"""
CBraMod adapter for FINGER-EEG-BCI task.

This module provides:
1. Wrapper for loading pretrained CBraMod model from the official repository
2. Task-specific classification head for finger-level BCI decoding
3. Fine-tuning utilities following the paper's configuration

Based on: Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model
for EEG Decoding", ICLR 2025

CBraMod repository: https://github.com/wjq-learning/CBraMod
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging
import sys
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)

# CBraMod repository path - configurable via environment variable
# Default: parallel to EEG-BCI (github/CBraMod)
_default_cbramod_path = Path(__file__).parent.parent.parent.parent / 'CBraMod'
CBRAMOD_REPO_PATH = Path(os.environ.get('CBRAMOD_REPO_PATH', _default_cbramod_path))


def get_cbramod_model():
    """
    Import CBraMod model from the official repository using direct file import.

    This avoids conflicts with local 'models' package by loading directly from file path.

    Returns:
        CBraMod class or None if not available
    """
    if not CBRAMOD_REPO_PATH.exists():
        logger.warning(f"CBraMod repo not found at: {CBRAMOD_REPO_PATH}")
        return None

    try:
        # Import cbramod.py directly from file path to avoid module name conflicts
        cbramod_file = CBRAMOD_REPO_PATH / 'models' / 'cbramod.py'
        if not cbramod_file.exists():
            logger.warning(f"cbramod.py not found at: {cbramod_file}")
            return None

        # First, import the criss_cross_transformer dependency
        transformer_file = CBRAMOD_REPO_PATH / 'models' / 'criss_cross_transformer.py'
        if transformer_file.exists():
            spec = importlib.util.spec_from_file_location("criss_cross_transformer", transformer_file)
            transformer_module = importlib.util.module_from_spec(spec)
            # Register under multiple names for torch.dynamo compatibility
            sys.modules['criss_cross_transformer'] = transformer_module
            sys.modules['models.criss_cross_transformer'] = transformer_module
            spec.loader.exec_module(transformer_module)
            logger.debug(f"Loaded criss_cross_transformer from {transformer_file}")

        # Now import cbramod
        spec = importlib.util.spec_from_file_location("cbramod", cbramod_file)
        cbramod_module = importlib.util.module_from_spec(spec)
        # Register under multiple names for torch.dynamo compatibility
        sys.modules['cbramod'] = cbramod_module
        sys.modules['models.cbramod'] = cbramod_module
        spec.loader.exec_module(cbramod_module)

        CBraMod = cbramod_module.CBraMod
        logger.info(f"Successfully imported official CBraMod from {cbramod_file}")
        logger.info("CBraMod includes: ACPE (Asymmetric Conditional Positional Encoding), "
                   "Criss-Cross Attention, Time-Frequency Patch Encoding")
        return CBraMod

    except Exception as e:
        logger.warning(f"Failed to import CBraMod: {e}")
        logger.warning("Falling back to PlaceholderCBraMod (no ACPE, no Criss-Cross Attention)")
        return None


class FingerBCIClassifier(nn.Module):
    """
    Classification head for finger-level BCI decoding.

    Following CBraMod paper's design:
    - 3-layer MLP with ELU activation and dropout
    - Input: flattened patch embeddings (ch_num * patch_num * d_model)
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        d_model: int = 200,
        dropout: float = 0.1,
        classifier_type: str = 'two_layer',
    ):
        """
        Initialize classifier.

        Args:
            input_dim: Input feature dimension (ch_num * patch_num * d_model)
            n_classes: Number of output classes
            d_model: Model dimension (default 200)
            dropout: Dropout probability
            classifier_type: 'three_layer', 'two_layer', or 'one_layer'
        """
        super().__init__()

        if classifier_type == 'three_layer':
            # Default: 3-layer MLP (as used in most CBraMod fine-tuning)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 4 * d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif classifier_type == 'two_layer':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        else:  # one_layer
            self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features [batch, ch_num, patch_num, d_model]

        Returns:
            Logits [batch, n_classes]
        """
        # Flatten: [batch, ch_num * patch_num * d_model]
        x = x.flatten(1)
        return self.classifier(x)


class CBraModForFingerBCI(nn.Module):
    """
    CBraMod model adapted for FINGER-EEG-BCI task.

    This is the main model class that wraps the pretrained CBraMod
    encoder with a task-specific classification head.

    Input format: [batch, n_channels, n_patches, patch_size]
    - n_channels: Number of EEG channels (19 for 10-20, or 128 for full BioSemi)
    - n_patches: Number of time patches (e.g., 5 for 5-second trial)
    - patch_size: 200 (1 second @ 200 Hz, fixed)

    Output: [batch, n_classes]

    Channel Flexibility (ACPE):
        CBraMod uses Asymmetric Conditional Positional Encoding (ACPE) which
        generates position encodings dynamically via convolution. The ACPE kernel
        is (19, 7) with padding (9, 3), allowing it to handle ANY number of channels.
        This means you can use 19 channels (standard 10-20) or 128 channels (full
        BioSemi) without any architecture changes - only the classifier input
        dimension changes accordingly.

    Memory Considerations:
        - 19 channels: Attention map 19x19 = 361, classifier input 3,800
        - 128 channels: Attention map 128x128 = 16,384, classifier input 25,600
        When using 128 channels, consider reducing batch_size to avoid OOM.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_patches: int = 5,
        n_classes: int = 2,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layers: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
        classifier_type: str = 'two_layer',
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        """
        Initialize CBraMod for finger BCI.

        Args:
            n_channels: Number of EEG channels
            n_patches: Number of time patches (trial_duration / 1s)
            n_classes: Number of finger classes
            d_model: Model dimension (must be 200 to match pretrained)
            dim_feedforward: FFN dimension (must be 800 to match pretrained)
            n_layers: Number of transformer layers (must be 12)
            n_heads: Number of attention heads (must be 8)
            dropout: Dropout probability
            classifier_type: Type of classification head
            pretrained_path: Path to pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        # Validate input parameters
        if not isinstance(n_channels, int) or n_channels < 1:
            raise ValueError(f"n_channels must be a positive integer, got {n_channels}")
        if not isinstance(n_patches, int) or n_patches < 1:
            raise ValueError(f"n_patches must be a positive integer, got {n_patches}")
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        # Warn about very high channel counts (potential OOM)
        if n_channels > 256:
            logger.warning(
                f"n_channels={n_channels} is very high. "
                f"Attention map size: {n_channels}x{n_channels}={n_channels**2:,}. "
                f"This may cause out-of-memory errors."
            )

        self.n_channels = n_channels
        self.n_patches = n_patches
        self.n_classes = n_classes
        self.d_model = d_model
        self.patch_size = 200  # Fixed: 1 second @ 200 Hz

        # Get CBraMod class
        CBraMod = get_cbramod_model()

        if CBraMod is not None:
            # Use official CBraMod
            self.backbone = CBraMod(
                in_dim=200,  # patch_size
                out_dim=200,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                seq_len=30,  # Max sequence length (will handle any length)
                n_layer=n_layers,
                nhead=n_heads,
            )
            self._using_official = True
        else:
            # Use placeholder
            logger.warning("Using placeholder encoder (CBraMod not found)")
            self.backbone = PlaceholderCBraMod(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self._using_official = False

        # Load pretrained weights
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        else:
            logger.warning("No pretrained weights provided! CBraMod will be trained from scratch.")
            logger.warning("Expected weights at: github/CBraMod/pretrained_weights/pretrained_weights.pth")
            logger.warning("Download from: https://huggingface.co/weighting666/CBraMod")

        # Replace output projection with identity (for fine-tuning)
        if hasattr(self.backbone, 'proj_out'):
            self.backbone.proj_out = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone weights frozen")

        # Classification head
        classifier_input_dim = n_channels * n_patches * d_model
        self.classifier = FingerBCIClassifier(
            input_dim=classifier_input_dim,
            n_classes=n_classes,
            d_model=d_model,
            dropout=dropout,
            classifier_type=classifier_type,
        )

        logger.debug(f"CBraModForFingerBCI initialized:")
        logger.debug(f"  - Channels: {n_channels}, Patches: {n_patches}")
        logger.debug(f"  - Classes: {n_classes}")
        logger.debug(f"  - Classifier input dim: {classifier_input_dim:,}")
        logger.debug(f"  - Using official CBraMod: {self._using_official}")

        # Warn about memory usage for high channel counts
        if n_channels > 64:
            logger.info(f"Using {n_channels} channels - attention map size: {n_channels}x{n_channels}={n_channels**2:,}")
            logger.info("Consider reducing batch_size if OOM occurs")

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained backbone weights."""
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

            # Load weights
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)

            if missing:
                logger.warning(f"Missing keys in pretrained: {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys in pretrained: {unexpected[:5]}...")

            logger.info(f"Loaded pretrained weights from {pretrained_path}")

        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input EEG [batch, n_channels, n_samples]
               or [batch, n_channels, n_patches, patch_size]

        Returns:
            Logits [batch, n_classes]
        """
        # Handle 3D input (continuous EEG)
        if x.dim() == 3:
            batch_size, n_channels, n_samples = x.shape
            n_patches = n_samples // self.patch_size

            # Truncate to full patches
            x = x[:, :, :n_patches * self.patch_size]

            # Reshape to patches: [batch, channels, patches, patch_size]
            x = x.view(batch_size, n_channels, n_patches, self.patch_size)

        # Backbone forward
        features = self.backbone(x)  # [batch, channels, patches, d_model]

        # Classification
        logits = self.classifier(features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        if x.dim() == 3:
            batch_size, n_channels, n_samples = x.shape
            n_patches = n_samples // self.patch_size
            x = x[:, :, :n_patches * self.patch_size]
            x = x.view(batch_size, n_channels, n_patches, self.patch_size)

        return self.backbone(x)

    def get_parameter_groups(self, backbone_lr: float = 1e-4, classifier_lr: float = 5e-4):
        """
        Get parameter groups with different learning rates.

        Following CBraMod paper: classifier lr = 5x backbone lr
        """
        backbone_params = list(self.backbone.parameters())
        classifier_params = list(self.classifier.parameters())

        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': classifier_params, 'lr': classifier_lr},
        ]

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class PlaceholderCBraMod(nn.Module):
    """
    Placeholder encoder for testing when CBraMod is not available.

    This is NOT the actual CBraMod model - only for pipeline testing.
    """

    def __init__(
        self,
        d_model: int = 200,
        n_layers: int = 12,
        n_heads: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Simple embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(200, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Simple transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=min(4, n_layers))

        self.proj_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, patches, patch_size]
        Returns:
            [batch, channels, patches, d_model]
        """
        batch_size, n_channels, n_patches, patch_size = x.shape

        # Embed patches
        x = x.view(batch_size * n_channels * n_patches, patch_size)
        x = self.patch_embed(x)

        # Reshape for transformer
        x = x.view(batch_size, n_channels * n_patches, self.d_model)

        # Encode
        x = self.encoder(x)

        # Reshape back
        x = x.view(batch_size, n_channels, n_patches, self.d_model)

        return self.proj_out(x)


# Convenience functions

def load_cbramod_for_finger_bci(
    pretrained_path: str,
    n_channels: int = 19,
    n_patches: int = 5,
    n_classes: int = 2,
    freeze_backbone: bool = False,
    device: str = 'cuda',
) -> CBraModForFingerBCI:
    """
    Load pretrained CBraMod for finger BCI task.

    Args:
        pretrained_path: Path to pretrained weights
        n_channels: Number of EEG channels
        n_patches: Number of time patches
        n_classes: Number of finger classes
        freeze_backbone: Whether to freeze backbone
        device: Device to load model on

    Returns:
        Initialized model
    """
    model = CBraModForFingerBCI(
        n_channels=n_channels,
        n_patches=n_patches,
        n_classes=n_classes,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
    )

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model


def get_default_pretrained_path() -> Optional[str]:
    """Get default path to pretrained weights.

    Priority order:
    1. Local checkpoints directory (checkpoints/cbramod/pretrained_weights.pth)
    2. External CBraMod repository (if available)

    The pretrained weights are included in this repository to avoid external dependencies.
    Downloaded from: https://huggingface.co/weighting666/CBraMod
    """
    # Check common locations - local checkpoints first
    possible_paths = [
        # Priority 1: Local checkpoints (included in this repo)
        Path(__file__).parent.parent.parent / 'checkpoints' / 'cbramod' / 'pretrained_weights.pth',
        # Priority 2: External CBraMod repo (fallback)
        CBRAMOD_REPO_PATH / 'pretrained_weights' / 'pretrained_weights.pth',
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


# Legacy alias for backward compatibility
CBraModAdapter = CBraModForFingerBCI
load_pretrained_cbramod = load_cbramod_for_finger_bci


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing CBraMod for Finger BCI")
    print("=" * 50)

    # Check if pretrained weights exist
    pretrained_path = get_default_pretrained_path()
    print(f"Pretrained path: {pretrained_path}")

    # Create model
    model = CBraModForFingerBCI(
        n_channels=19,
        n_patches=5,  # 5 seconds
        n_classes=2,  # Binary: thumb vs pinky
        pretrained_path=pretrained_path,
    )

    print(f"\nModel created:")
    print(f"  Trainable parameters: {model.count_parameters():,}")
    print(f"  Total parameters: {model.count_parameters(trainable_only=False):,}")

    # Test forward pass
    # Input: 5 seconds of EEG @ 200 Hz = 1000 samples
    x = torch.randn(4, 19, 1000)  # [batch, channels, samples]

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {y.shape}")

    # Test with 4D input
    x_4d = torch.randn(4, 19, 5, 200)  # [batch, channels, patches, patch_size]

    print(f"\n4D Input shape: {x_4d.shape}")

    with torch.no_grad():
        y_4d = model(x_4d)

    print(f"4D Output shape: {y_4d.shape}")

    # Get parameter groups for different LR
    param_groups = model.get_parameter_groups(backbone_lr=1e-4, classifier_lr=5e-4)
    print(f"\nParameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {n_params:,} params, lr={group['lr']}")
