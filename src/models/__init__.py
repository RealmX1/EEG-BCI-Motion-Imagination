# Models module
"""
Neural network models for EEG-BCI project.

Models:
- EEGNet: Compact CNN baseline (Lawhern et al., 2018)
- CBraMod: Criss-Cross Transformer foundation model (Wang et al., ICLR 2025)
"""

from .eegnet import EEGNet
from .cbramod_adapter import CBraModForFingerBCI, load_cbramod_for_finger_bci

__all__ = [
    'EEGNet',
    'CBraModForFingerBCI',
    'load_cbramod_for_finger_bci',
]
