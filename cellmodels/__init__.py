"""
cellmodels — MSC confluency estimation from microscopy images.

Provides a U-Net-based deep learning pipeline for estimating mesenchymal
stem cell (MSC) confluency from phase-contrast and brightfield micrographs.
Supports multiple magnifications (currently 10x; 4x, 20x, 40x planned).
"""

__version__ = "0.1.0"

from cellmodels.confluency import MSCConfluency
from cellmodels.unet import UNet

__all__ = ["MSCConfluency", "UNet", "__version__"]
