"""
MSC Confluency Prediction — Production Inference Module.

Estimates mesenchymal stem cell confluency from microscopy images using
a trained U-Net checkpoint and calibrated post-processing parameters.

This module has ZERO visibility into ground-truth masks. It exposes
reusable functions that ``evaluate.py`` imports to prevent codebase drift.

Usage:
    # Single image
    python scripts/predict.py path/to/image.TIF

    # Directory of images
    python scripts/predict.py path/to/images/

    # With options
    python scripts/predict.py path/to/images/ --checkpoint output/training/best_msc_unet.pt

Output:
    - Per-image confluency percentage
    - 3-panel overlay images saved to output/confluency/
    - Summary CSV saved to output/confluency/confluency_results.csv
"""

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects
from tqdm import tqdm

from cellmodels.unet import UNet


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------------
# Core reusable functions (imported by evaluate.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, encoder_backbone, device):
    """Load a trained segmentation model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` state-dict file.
    encoder_backbone : str
        One of ``"scratch"``, ``"vgg16"``, ``"resnet34"``.
    device : torch.device
        Target compute device.

    Returns
    -------
    model : torch.nn.Module
        Model in eval mode on *device*.
    in_channels : int
        Number of input channels the model expects (1 for scratch, 3 for SMP).
    """
    if encoder_backbone == "scratch":
        model = UNet(in_channels=1, out_channels=1, init_features=16)
        in_channels = 1
    else:
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            print(
                "Error: segmentation-models-pytorch is required for "
                "transfer-learning backbones. Install with: "
                "pip install cellmodels[transfer]"
            )
            raise SystemExit(1)

        model = smp.Unet(
            encoder_name=encoder_backbone,
            encoder_weights=None,  # weights come from checkpoint
            in_channels=3,
            classes=1,
        )
        in_channels = 3

    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, in_channels


def preprocess_image(raw_image):
    """Normalise a grayscale image by its 99th percentile and clip outliers.

    Parameters
    ----------
    raw_image : np.ndarray
        2-D grayscale image (any dtype).

    Returns
    -------
    np.ndarray
        2-D float32 image with values in [0, 1].
    """
    x = raw_image.copy().astype(np.float32)
    q = np.quantile(x, 0.99)
    if q > 0:
        x /= q
    x[x > 1.0] = 1.0
    return x


def get_density_map(model, image, in_channels, device):
    """Run the model forward pass and return the sigmoid probability map.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model in eval mode.
    image : np.ndarray
        2-D float32 preprocessed grayscale image.
    in_channels : int
        Number of input channels (1 for scratch, 3 for SMP).
    device : torch.device
        Compute device.

    Returns
    -------
    np.ndarray
        2-D float32 probability map in [0, 1].
    """
    h, w = image.shape

    # Pad dimensions to multiples of 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        image_padded = image

    # Build input tensor: [1, C, H, W]
    tensor = torch.from_numpy(image_padded).unsqueeze(0).unsqueeze(0).float()
    if in_channels == 3:
        tensor = tensor.expand(-1, 3, -1, -1).contiguous()

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob_map = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # Crop back to original dimensions
    if pad_h > 0 or pad_w > 0:
        prob_map = prob_map[:h, :w]

    return prob_map


def segment_density(density_map, config):
    """Segment a probability density map into a binary cell mask.

    Reads the ``method`` key from *config* and applies either Otsu-scaled
    or absolute static thresholding, followed by morphological closing and
    small-object removal.

    Parameters
    ----------
    density_map : np.ndarray
        2-D float32 probability map from the U-Net.
    config : dict
        Calibration config containing at minimum:
        - ``method``: ``"otsu_scaled"`` or ``"absolute_threshold"``
        - ``closing_radius``: int
        - ``min_object_size``: int
        And one of:
        - ``t_factor``: float  (for ``otsu_scaled``)
        - ``prob_threshold``: float  (for ``absolute_threshold``)

    Returns
    -------
    np.ndarray
        Binary cell mask (bool).
    """
    method = config.get("method", "otsu_scaled")
    closing_radius = config.get("closing_radius", 3)
    min_object_size = config.get("min_object_size", 50)

    # Handle degenerate (uniform) density maps
    if density_map.max() - density_map.min() < 1e-6:
        return np.zeros_like(density_map, dtype=bool)

    # --- Thresholding step ---
    if method == "absolute_threshold":
        prob_threshold = config.get("prob_threshold", 0.50)
        cell_mask = density_map > prob_threshold
    else:
        # Default: otsu_scaled
        t_factor = config.get("t_factor", 1.0)
        otsu_val = threshold_otsu(density_map)
        cell_mask = density_map > (otsu_val * t_factor)

    # --- Morphological closing ---
    if closing_radius > 0:
        cell_mask = closing(cell_mask, footprint=disk(closing_radius))

    # --- Small-object removal ---
    if min_object_size > 0:
        sig = inspect.signature(remove_small_objects)
        if "max_size" in sig.parameters:
            cell_mask = remove_small_objects(
                cell_mask, max_size=min_object_size - 1
            )
        else:
            cell_mask = remove_small_objects(
                cell_mask, min_size=min_object_size
            )

    return cell_mask


# ---------------------------------------------------------------------------
# CLI-only helpers (not imported by evaluate.py)
# ---------------------------------------------------------------------------

def create_overlay(image, density_map, cell_mask, confluency_pct, filename):
    """Create a 3-panel visualisation: raw image, density heatmap, overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Input image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Probability map (heatmap)
    im = axes[1].imshow(density_map, cmap="hot")
    axes[1].set_title("U-Net Probability Map", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Confluency overlay
    if image.max() > 0:
        rgb = np.stack([image / image.max()] * 3, axis=-1)
    else:
        rgb = np.zeros(image.shape + (3,))
    overlay = rgb.copy()
    # Green tint for cell regions
    overlay[cell_mask, 1] = np.clip(overlay[cell_mask, 1] + 0.3, 0, 1)
    # Red border for cell boundary
    border = binary_dilation(cell_mask, structure=disk(1)) & ~cell_mask
    overlay[border, 0] = 1.0
    overlay[border, 1] = 0.2
    overlay[border, 2] = 0.2

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Confluency: {confluency_pct:.1f}%", fontsize=14, fontweight="bold"
    )
    axes[2].axis("off")

    fig.suptitle(filename, fontsize=10, color="gray")
    fig.tight_layout()
    return fig


def find_images(path):
    """Find all supported image files in a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(p.glob(f"*{ext}"))
            images.extend(p.glob(f"*{ext.upper()}"))
        return sorted(set(images))
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)


def select_device(device_str):
    """Resolve the compute device from a CLI string or auto-detect."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate MSC confluency from microscopy images"
    )
    parser.add_argument(
        "input",
        help="Path to a single image or directory of images",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/training/best_msc_unet.pt",
        help="Path to trained model weights .pt file",
    )
    parser.add_argument(
        "--optimal-config",
        default="output/calibration/optimal_config.json",
        help="Path to calibrated optimal_config.json (default: output/calibration/optimal_config.json)",
    )
    parser.add_argument(
        "--encoder-backbone",
        default="scratch",
        choices=["scratch", "vgg16", "resnet34"],
        help="Encoder backbone matching the trained checkpoint (default: scratch)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/confluency",
        help="Directory to save results (default: output/confluency)",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Skip saving overlay images (only produce CSV)",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    # Load calibration config
    config_path = Path(args.optimal_config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded calibration config from: {config_path}")
        print(f"  Method: {config.get('method', 'unknown')}")
    else:
        print(
            f"Warning: Config not found at {config_path}. "
            f"Using Otsu-scaled defaults (t_factor=1.0, closing=3, min_obj=50)."
        )
        config = {
            "method": "otsu_scaled",
            "t_factor": 1.0,
            "closing_radius": 3,
            "min_object_size": 50,
        }

    # Verify checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    # Find images
    images = find_images(args.input)
    if not images:
        print(f"No supported images found in {args.input}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(images)} image(s)")

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "confluency_results.csv")

    # Load model
    print(f"Loading model checkpoint: {args.checkpoint}")
    model, in_channels = load_model(args.checkpoint, args.encoder_backbone, device)

    # Process images
    results = []
    for img_path in tqdm(images, desc="Processing"):
        name = img_path.name

        # Load and preprocess
        raw_image = plt.imread(str(img_path))
        if raw_image.ndim == 3:
            raw_image = raw_image.mean(axis=-1)

        image = preprocess_image(raw_image)

        # Get density map
        density_map = get_density_map(model, image, in_channels, device)

        # Segment using calibrated config
        cell_mask = segment_density(density_map, config)

        # Confluency from binary mask pixel ratio
        confluency_pct = (cell_mask.sum() / cell_mask.size) * 100.0

        result = {
            "filename": name,
            "confluency_pct": round(confluency_pct, 2),
        }
        print(f"  {name}: {confluency_pct:.1f}% confluent")

        # Save overlay
        if not args.no_save_images:
            fig = create_overlay(image, density_map, cell_mask, confluency_pct, name)
            fig.savefig(
                os.path.join(args.output_dir, f"confluency_{name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        results.append(result)

    # Write CSV
    fieldnames = ["filename", "confluency_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 50)
    print("CONFLUENCY SUMMARY")
    print("=" * 50)
    confluencies = [r["confluency_pct"] for r in results]
    print(f"  Images processed: {len(results)}")
    print(f"  Mean confluency:  {np.mean(confluencies):.1f}%")
    print(f"  Min confluency:   {np.min(confluencies):.1f}%")
    print(f"  Max confluency:   {np.max(confluencies):.1f}%")
    print(f"  Std deviation:    {np.std(confluencies):.1f}%")
    print(f"\n  Results saved to: {csv_path}")
    if not args.no_save_images:
        print(f"  Overlays saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
