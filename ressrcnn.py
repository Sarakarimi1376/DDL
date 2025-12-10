import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block used in deeper SRCNN-like model."""

    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ResSRCNN(nn.Module):
    """
    Deeper super-resolution model.
    ~0.7M parameters.
    """

    def __init__(self, num_res_blocks=8):
        super().__init__()

        # Feature extraction
        self.head = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # Final reconstruction
        self.tail = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.tail(x + residual)
        return x
