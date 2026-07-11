"""
Base model class providing preprocessing and tiled inference.

Handles:
    - 99th-percentile intensity normalisation
    - Tiled inference for large images (non-overlapping tiles)
    - Automatic dimension padding to multiples of 16
"""

from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from cellmodels.unet import UNet


# Directory containing shipped weight files
_WEIGHTS_DIR = Path(__file__).parent / "weights"


class Base:
    model_name: str
    model_args: Tuple[int, int, int]

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model: torch.nn.Module = UNet(*self.model_args)
        self.model.eval()

        # Load weights: custom checkpoint or default shipped weights
        if checkpoint is not None:
            weight_path = Path(checkpoint)
        else:
            weight_path = _WEIGHTS_DIR / f"{self.model_name}.pt"

        if weight_path.exists():
            self.model.load_state_dict(
                torch.load(
                    str(weight_path),
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
        else:
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. "
                f"Provide a valid checkpoint path or ensure the default "
                f"weights are installed with the package."
            )

        if device is not None:
            self.model.to(device)
            self.device = device
        else:
            self.device = torch.device("cpu")

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Normalise input by 99th percentile and clip extreme bright pixels."""
        x_copy = x.copy().astype(np.float32)
        q = np.quantile(x_copy, 0.99)
        if q > 0:
            x_copy /= q
        x_copy[x_copy > 1.0] = 1.0
        return x_copy

    def run_model(self, x: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
        """Run the U-Net on a 2D grayscale image, optionally with tiling."""
        assert len(x.shape) == 2, "Image must be a 2D array"

        x = self.preprocess(x)

        h_img, w_img = x.shape

        if max_size is not None:
            n_rows = ceil(h_img / max_size)
            n_cols = ceil(w_img / max_size)
            max_size_x = max_size_y = max_size
            assert max_size % 16 == 0
        else:
            n_rows = n_cols = 1
            max_size_y = h_img
            max_size_x = w_img

        prediction = None
        for i in range(n_rows):
            for j in range(n_cols):
                min_row = i * max_size_y
                min_col = j * max_size_x

                tile_h = min(max_size_y, h_img - min_row)
                tile_w = min(max_size_x, w_img - min_col)

                tile = x[min_row : min_row + tile_h, min_col : min_col + tile_w]

                # Pad tile dimensions to multiples of 16 using edge replication
                pad_h = (16 - tile_h % 16) % 16
                pad_w = (16 - tile_w % 16) % 16

                if pad_h > 0 or pad_w > 0:
                    padded_tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="edge")
                else:
                    padded_tile = tile

                input_tensor = torch.Tensor(padded_tile[None, None, :, :]).to(self.device)
                with torch.no_grad():
                    # Apply sigmoid to model logits to get probabilities
                    model_output = torch.sigmoid(self.model(input_tensor)).cpu().numpy()[0]

                # Crop prediction back to its unpadded size
                if pad_h > 0 or pad_w > 0:
                    model_output = model_output[:, :tile_h, :tile_w]

                if prediction is None:
                    prediction = np.zeros((model_output.shape[0],) + x.shape, dtype=np.float32)
                prediction[
                    :, min_row : min_row + tile_h, min_col : min_col + tile_w
                ] = model_output

        assert prediction is not None
        return prediction
