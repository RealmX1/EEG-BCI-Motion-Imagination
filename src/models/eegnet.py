"""
EEGNet implementation for FINGER-EEG-BCI.

Based on:
- Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-based
  Brain-Computer Interfaces", Journal of Neural Engineering, 2018

Configuration matches the original FINGER-EEG-BCI paper:
- EEGNet-8,2 (F1=8, D=2, F2=16)
- Input: 128 channels @ 100 Hz, 4 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with weight constraint (max norm)."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply max norm constraint during forward pass
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class EEGNet(nn.Module):
    """
    EEGNet-8,2 architecture.

    Architecture:
    1. Temporal convolution (extract frequency features)
    2. Depthwise convolution (learn spatial filters for each temporal feature)
    3. Separable convolution (learn temporal features)
    4. Classification head

    Input shape: [batch, 1, channels, time_samples]
    Output shape: [batch, n_classes]
    """

    def __init__(
        self,
        n_channels: int = 128,
        n_samples: int = 400,  # 4s @ 100Hz
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,  # 0.5s @ 128Hz
        dropout_rate: float = 0.5,
    ):
        """
        Initialize EEGNet.

        Args:
            n_channels: Number of EEG channels (default 128)
            n_samples: Number of time samples (default 400 for 4s @ 100Hz)
            n_classes: Number of output classes
            F1: Number of temporal filters (default 8)
            D: Depth multiplier for depthwise conv (default 2)
            F2: Number of pointwise filters (default 16, should be F1 * D)
            kernel_length: Length of temporal kernel in samples (default 64)
            dropout_rate: Dropout probability (default 0.5)
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal + Spatial filtering
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise (spatial) convolution
        self.spatial_conv = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution
        # Depthwise temporal convolution
        self.separable_conv1 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        # Pointwise convolution
        self.separable_conv2 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate output size
        self._calculate_output_size()

        # Classification head
        self.fc = nn.Linear(self.feature_size, n_classes)

    def _calculate_output_size(self):
        """Calculate the flattened feature size."""
        # Create dummy input
        x = torch.zeros(1, 1, self.n_channels, self.n_samples)

        # Forward through feature extraction
        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)

        self.feature_size = x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, time] or [batch, 1, channels, time]

        Returns:
            Output logits [batch, n_classes]
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, channels, time]

        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Classification
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)

        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EEGNetForCBraModComparison(EEGNet):
    """
    EEGNet configured to match CBraMod input format for fair comparison.

    Uses 19 channels @ 200 Hz instead of 128 channels @ 100 Hz.
    """

    def __init__(
        self,
        n_samples: int = 1000,  # 5s @ 200Hz
        n_classes: int = 2,
        **kwargs
    ):
        # Adjust kernel length for 200 Hz
        kernel_length = kwargs.pop('kernel_length', 128)  # 0.5s @ 200Hz

        super().__init__(
            n_channels=19,  # Standard 10-20 channels
            n_samples=n_samples,
            n_classes=n_classes,
            kernel_length=kernel_length,
            **kwargs
        )


if __name__ == '__main__':
    # Test EEGNet
    print("Testing EEGNet-8,2")
    print("-" * 40)

    # Original configuration (128 channels @ 100 Hz)
    model = EEGNet(
        n_channels=128,
        n_samples=400,  # 4s @ 100Hz
        n_classes=2,
    )

    x = torch.randn(8, 1, 128, 400)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")

    # CBraMod-compatible configuration (19 channels @ 200 Hz)
    print("\nTesting EEGNet for CBraMod comparison")
    print("-" * 40)

    model2 = EEGNetForCBraModComparison(
        n_samples=1000,  # 5s @ 200Hz
        n_classes=2,
    )

    x2 = torch.randn(8, 1, 19, 1000)
    y2 = model2(x2)

    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {y2.shape}")
    print(f"Parameters: {model2.count_parameters():,}")
