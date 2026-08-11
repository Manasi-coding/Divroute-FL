import torch.nn as nn
import torch.nn.functional as F


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


def get_model(model_name: str, num_classes: int, bn_mode: str = "default") -> nn.Module:
    """
    Model factory.  Returns an initialised (random-weight) model.

    Supported values for model_name:
        "simplecnn"  — lightweight CNN (~200k params), suitable for CIFAR-10
        "resnet18"   — torchvision ResNet-18 (~11M params), suitable for CIFAR-100

    Parameters
    ----------
    model_name  : one of {"simplecnn", "resnet18"}
    num_classes : number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    bn_mode     : "default", "local_bn", "groupnorm"
    """
    name = model_name.lower().strip()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
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
            def _replace_bn(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        num_features = child.num_features
                        # 32 groups is standard, but must divide num_features. 
                        # ResNet18 has 64, 128, 256, 512 channels, all divisible by 32.
                        gn = nn.GroupNorm(32, num_features)
                        setattr(module, name, gn)
                    else:
                        _replace_bn(child)
            _replace_bn(model)
            
        return model
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported values: 'simplecnn', 'resnet18'."
        )