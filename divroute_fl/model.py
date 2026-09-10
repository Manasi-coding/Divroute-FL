import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    """
    Weight Standardized Convolution.
    w_hat = (w - mean(w)) / (std(w, unbiased=False) + 1e-5)
    Standardization is performed independently per output channel.
    """
    def forward(self, input):
        # Weight shape: (out_channels, in_channels, kernel_size[0], kernel_size[1])
        # Normalize over in_channels, kernel_size[0], kernel_size[1]
        weight_mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        weight_std = self.weight.std(dim=[1, 2, 3], unbiased=False, keepdim=True)
        weight = (self.weight - weight_mean) / (weight_std + 1e-5)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class SimpleCNN(nn.Module):
    """
    Small CNN for CIFAR-10. ~200k parameters.

    Architecture:
        conv(3->16, 3x3) -> relu -> maxpool
        conv(16->32, 3x3) -> relu -> maxpool
        fc(32*8*8 -> 128) -> relu
        fc(128 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _pick_num_groups(num_features: int) -> int:
    """Pick a GroupNorm group count that evenly divides num_features.

    Prefers 32 (the original hardcoded choice — correct for ResNet-18, whose
    channel counts of 64/128/256/512 are all divisible by 32), falling back
    through common power-of-two divisors for architectures with uneven
    channel counts. This matters for EfficientNet-B0: its MBConv stages use
    channel counts (16, 24, 40, 80, 112, ...) that are NOT all divisible by
    32 — an earlier version of this function hardcoded 32 unconditionally and
    crashed with "num_channels must be divisible by num_groups" on those
    layers. Verified empirically, not assumed.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if num_features % g == 0:
            return g
    return 1  # unreachable — g=1 always divides evenly


def _replace_bn(module: nn.Module) -> None:
    """Recursively replace every nn.BatchNorm2d in `module` with GroupNorm.

    Architecture-agnostic (walks named_children()), so it works unmodified on
    any model built from standard BatchNorm2d layers — ResNet-18, EfficientNet's
    MBConv blocks, etc. Group count is chosen per-layer via _pick_num_groups()
    since a fixed 32 does not evenly divide every layer's channel count for
    every architecture (see that function's docstring).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(_pick_num_groups(child.num_features), child.num_features)
            setattr(module, name, gn)
        else:
            _replace_bn(child)


def _replace_ws_gn(module: nn.Module) -> None:
    """Recursively replace nn.Conv2d -> WSConv2d and nn.BatchNorm2d -> GroupNorm.

    Architecture-agnostic, same rationale as _replace_bn above.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(_pick_num_groups(child.num_features), child.num_features)
            setattr(module, name, gn)
        elif isinstance(child, nn.Conv2d):
            ws_conv = WSConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                child.bias is not None, child.padding_mode
            )
            ws_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ws_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, ws_conv)
        else:
            _replace_ws_gn(child)


def get_model(model_name: str, num_classes: int, bn_mode: str = "default") -> nn.Module:
    """
    Model factory.  Returns an initialised model.

    Supported values for model_name:
        "simplecnn"                 — lightweight CNN (~200k params), suitable for CIFAR-10
        "resnet18"                  — torchvision ResNet-18 (~11M params, random init), suitable for CIFAR-100
        "efficientnet_b0_pretrained" — torchvision EfficientNet-B0 (~5.3M params),
                                        ImageNet-pretrained (IMAGENET1K_V1). Reframes
                                        training as federated FINE-TUNING rather than
                                        federated learning-from-scratch. Requires
                                        inputs resized off CIFAR's native 32x32 and
                                        normalized with ImageNet statistics — handled
                                        automatically by data.py when this model_name
                                        is passed through (see _PRETRAINED_IMAGE_SIZE).

    Parameters
    ----------
    model_name  : one of {"simplecnn", "resnet18", "efficientnet_b0_pretrained"}
    num_classes : number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    bn_mode     : "default", "local_bn", "groupnorm", "ws_groupnorm"
                  For "efficientnet_b0_pretrained", leaving this at "default"
                  preserves the pretrained BatchNorm running statistics — do not
                  set "groupnorm"/"ws_groupnorm" for the main fine-tuning result,
                  since it discards that calibration (fine for a secondary ablation).
    """
    name = model_name.lower().strip()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "efficientnet_b0_pretrained":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Swap the ImageNet-1000-way classifier head for the target class count.
        # classifier = Sequential(Dropout, Linear) — only the Linear layer changes.
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        if bn_mode == "groupnorm":
            _replace_bn(model)
        elif bn_mode == "ws_groupnorm":
            _replace_ws_gn(model)
        # bn_mode == "default" (recommended for the main fine-tuning run):
        # BatchNorm2d layers and their pretrained running stats are left untouched.
        return model
    elif name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        # CIFAR adaptation (standard in FL research — FedProx, SCAFFOLD, FedNova etc.):
        # The default ResNet-18 uses a 7×7 conv (stride=2) followed by MaxPool (stride=2),
        # reducing a 32×32 CIFAR image to 8×8 before any residual block, then to 1×1
        # by layer4. This spatial collapse makes learning essentially impossible on
        # CIFAR-scale inputs. Replace with a 3×3 conv (stride=1) and remove MaxPool
        # so the feature map stays at 32×32 through the stem, matching published baselines.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()   # remove the stride-2 MaxPool
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # BN Ablation: Replace BatchNorm2d with GroupNorm if requested
        # Default ResNet uses BatchNorm2d which can cause issues in non-IID FL
        if bn_mode == "groupnorm":
            _replace_bn(model)
        elif bn_mode == "ws_groupnorm":
            _replace_ws_gn(model)

        return model
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported values: 'simplecnn', 'resnet18', 'efficientnet_b0_pretrained'."
        )