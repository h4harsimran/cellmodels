"""
MSC Confluency estimation model.

Provides the ``MSCConfluency`` class for estimating mesenchymal stem cell
confluency from phase-contrast or brightfield microscopy images. Supports
multiple magnifications via per-magnification weight files.

Usage::

    from cellmodels import MSCConfluency

    model = MSCConfluency(magnification="40x")
    density_map, confluency_pct = model.predict(image)
"""

import inspect
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects

from cellmodels.base_model import Base

# Calibrated segmentation defaults per magnification.
# These were determined via grid search on held-out calibration sets.
# Each entry maps magnification → (threshold_factor, closing_radius, min_object_size)
_CALIBRATED_DEFAULTS: Dict[str, Tuple[float, int, int]] = {
    "40x": (0.8, 1, 200),
    # Future magnifications will be added here as models are trained:
    # "4x":  (t_factor, radius, min_size),
    # "10x": (t_factor, radius, min_size),
    # "20x": (t_factor, radius, min_size),
}

# Magnifications that have trained weights available
AVAILABLE_MAGNIFICATIONS = list(_CALIBRATED_DEFAULTS.keys())


class MSCConfluency(Base):
    """
    MSC confluency estimation from microscopy images.

    Uses a U-Net backbone to produce a density map, then applies calibrated
    thresholding and morphological operations to segment cells and compute
    confluency percentage.

    Parameters
    ----------
    magnification : str
        Objective magnification of the input images. Determines which
        weight file and calibrated segmentation parameters to use.
        Currently supported: ``"40x"``.
    checkpoint : str, optional
        Path to a custom ``.pt`` checkpoint file. If provided, overrides
        the default weights for the given magnification.
    device : torch.device, optional
        Compute device. Defaults to CPU if not specified.

    Examples
    --------
    >>> from cellmodels import MSCConfluency
    >>> model = MSCConfluency(magnification="40x")
    >>> density_map, confluency_pct = model.predict(image)
    >>> print(f"Confluency: {confluency_pct:.1f}%")
    """

    # U-Net configuration: 1 input channel (grayscale), 1 output (density),
    # 16 initial features
    model_args: Tuple[int, int, int] = (1, 1, 16)

    def __init__(
        self,
        magnification: str = "40x",
        checkpoint: Optional[str] = None,
        device=None,
    ):
        self.magnification = magnification

        # Set model_name to match the weight filename (e.g., "40x" → weights/40x.pt)
        self.model_name = magnification

        # Locate weights directory and config file
        weights_dir = Path(__file__).parent / "weights"
        config_path = weights_dir / f"{magnification}.json"
        default_weight_path = weights_dir / f"{magnification}.pt"

        # Check if magnification is supported
        if (
            magnification not in _CALIBRATED_DEFAULTS
            and checkpoint is None
            and not default_weight_path.exists()
        ):
            available_stems = [p.stem for p in weights_dir.glob("*.pt")]
            available = ", ".join(
                sorted(set(AVAILABLE_MAGNIFICATIONS + available_stems))
            )
            raise ValueError(
                f"Unsupported magnification '{magnification}'. "
                f"Available: {available}"
            )

        # Load calibrated segmentation defaults
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.method = config.get("method", "otsu_scaled")
                self.default_closing_radius = config.get("closing_radius", 3)
                self.default_min_object_size = config.get("min_object_size", 50)
                if self.method == "absolute_threshold":
                    self.default_prob_threshold = config.get("prob_threshold", 0.50)
                else:
                    self.default_threshold_factor = config.get("t_factor", 1.0)
            except Exception:
                # Fallback to hardcoded defaults on parsing error
                self.method = "otsu_scaled"
                t_factor, c_radius, min_size = _CALIBRATED_DEFAULTS.get(
                    magnification, (1.0, 3, 50)
                )
                self.default_threshold_factor = t_factor
                self.default_closing_radius = c_radius
                self.default_min_object_size = min_size
        else:
            self.method = "otsu_scaled"
            t_factor, c_radius, min_size = _CALIBRATED_DEFAULTS.get(
                magnification, (1.0, 3, 50)
            )
            self.default_threshold_factor = t_factor
            self.default_closing_radius = c_radius
            self.default_min_object_size = min_size

        super().__init__(checkpoint=checkpoint, device=device)

    def predict(
        self,
        image: np.ndarray,
        max_size: Optional[int] = None,
        threshold_factor: Optional[float] = None,
        closing_radius: Optional[int] = None,
        min_object_size: Optional[int] = None,
        prob_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference and return the density map and confluency percentage.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale image (float32, raw, or pre-normalised).
        max_size : int, optional
            Tile size for large images (must be a multiple of 16).
        threshold_factor : float, optional
            Multiplier for the Otsu threshold. Defaults to calibrated value.
        closing_radius : int, optional
            Disk radius for morphological closing. Defaults to calibrated value.
        min_object_size : int, optional
            Minimum connected-component size in pixels. Defaults to calibrated value.
        prob_threshold : float, optional
            Absolute probability threshold. If provided, overrides Otsu thresholding.

        Returns
        -------
        density_map : np.ndarray
            Raw U-Net output density map (2D, float32).
        confluency_pct : float
            Estimated confluency as a percentage (0–100).
        """
        density_map = self.get_density_map(image, max_size=max_size)
        cell_mask = self.segment(
            density_map,
            threshold_factor=threshold_factor,
            closing_radius=closing_radius,
            min_object_size=min_object_size,
            prob_threshold=prob_threshold,
        )
        confluency_pct = (cell_mask.sum() / cell_mask.size) * 100.0
        return density_map, confluency_pct

    def get_density_map(
        self,
        image: np.ndarray,
        max_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run the U-Net and return the raw density map without segmentation.

        Useful for custom post-processing pipelines.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale image.
        max_size : int, optional
            Tile size for large images (must be a multiple of 16).

        Returns
        -------
        np.ndarray
            Raw density map (2D, float32).
        """
        prediction = super().run_model(image, max_size)
        # Squeeze out the channel dimension (1, H, W) → (H, W)
        return prediction[0]

    def segment(
        self,
        density_map: np.ndarray,
        threshold_factor: Optional[float] = None,
        closing_radius: Optional[int] = None,
        min_object_size: Optional[int] = None,
        prob_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Segment a density map into a binary cell mask.

        Uses Otsu or absolute thresholding, morphological closing, and small-object
        removal with calibrated defaults for the configured magnification.

        Parameters
        ----------
        density_map : np.ndarray
            2D density map from the U-Net.
        threshold_factor : float, optional
            Multiplier for the Otsu threshold.
        closing_radius : int, optional
            Disk radius for morphological closing.
        min_object_size : int, optional
            Minimum connected-component size in pixels.
        prob_threshold : float, optional
            Absolute probability threshold. If provided, overrides Otsu thresholding.

        Returns
        -------
        np.ndarray
            Binary cell mask (bool).
        """
        if closing_radius is None:
            closing_radius = self.default_closing_radius
        if min_object_size is None:
            min_object_size = self.default_min_object_size

        # Handle degenerate (uniform) images
        if density_map.max() - density_map.min() < 1e-6:
            return np.zeros_like(density_map, dtype=bool)

        # Apply threshold based on method and overrides
        if prob_threshold is not None:
            cell_mask = density_map > prob_threshold
        elif threshold_factor is not None:
            t = threshold_otsu(density_map) * threshold_factor
            cell_mask = density_map > t
        elif getattr(self, "method", "otsu_scaled") == "absolute_threshold":
            t = getattr(self, "default_prob_threshold", 0.50)
            cell_mask = density_map > t
        else:
            factor = (
                threshold_factor
                if threshold_factor is not None
                else getattr(self, "default_threshold_factor", 1.0)
            )
            t = threshold_otsu(density_map) * factor
            cell_mask = density_map > t

        if closing_radius > 0:
            cell_mask = closing(cell_mask, footprint=disk(closing_radius))

        if min_object_size > 0:
            # Handle scikit-image API changes (min_size → max_size in 0.26+)
            sig = inspect.signature(remove_small_objects)
            if "max_size" in sig.parameters:
                cell_mask = remove_small_objects(
                    cell_mask, max_size=min_object_size - 1
                )
            else:
                cell_mask = remove_small_objects(cell_mask, min_size=min_object_size)

        return cell_mask
