"""
DivRoute-FL Lite — Model
Simple CNN for CIFAR-10 classification.
Architecture: Conv→ReLU→Pool → Conv→ReLU→Pool → FC→FC(10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A lightweight CNN suitable for CIFAR-10 (32×32×3 images, 10 classes).
    ~200 k parameters — small enough for fast CPU-based FL simulation.
    """

    def __init__(self):
        super().__init__()
        # --- Feature extractor ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32×32×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16×16×64

        # After two 2×2 max-pools the spatial dims are 8×8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)          # 32→16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          # 16→8
        x = x.view(x.size(0), -1)       # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
