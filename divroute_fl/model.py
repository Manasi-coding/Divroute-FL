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


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Model factory.  Returns an initialised (random-weight) model.

    Supported values for model_name:
        "simplecnn"  — lightweight CNN (~200k params), suitable for CIFAR-10
        "resnet18"   — torchvision ResNet-18 (~11M params), suitable for CIFAR-100

    Parameters
    ----------
    model_name  : one of {"simplecnn", "resnet18"}
    num_classes : number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    """
    name = model_name.lower().strip()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported values: 'simplecnn', 'resnet18'."
        )