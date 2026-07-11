"""
Calibrate post-processing parameters on the validation set.

Loads a trained U-Net checkpoint, runs forward inference on every
validation image, and performs a grid search over two distinct
post-processing techniques to find the optimal segmentation parameters.

Techniques:
    1. Otsu Threshold Scaling  -- adaptive threshold via Otsu * factor
    2. Absolute Thresholding   -- static probability cutoff

Usage:
    python scripts/calibrate.py --val-dir dataset/val
    python scripts/calibrate.py --val-dir dataset/val --checkpoint output/training/best_msc_unet.pt
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects
from tqdm import tqdm

from cellmodels.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate segmentation parameters via grid search on validation set"
    )
    parser.add_argument(
        "--val-dir",
        default="dataset/val",
        help="Path to the validation split directory (containing images/ and masks/)",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/training/best_msc_unet.pt",
        help="Path to trained model checkpoint (default: output/training/best_msc_unet.pt)",
    )
    parser.add_argument(
        "--encoder-backbone",
        default="scratch",
        choices=["scratch", "vgg16", "resnet34"],
        help="Encoder backbone matching the trained checkpoint (default: scratch)",
    )
    parser.add_argument(
        "--bias-weight",
        type=float,
        default=0.01,
        help=(
            "Penalty weight for mean absolute confluency bias in "
            "grid search scoring (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--calibrate-size",
        type=int,
        default=15,
        help="Number of validation images to use for parameter calibration (default: 15, use -1 for all)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/calibration",
        help="Directory to save optimal_config.json (default: output/calibration)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )
    return parser.parse_args()


def load_model(checkpoint_path, encoder_backbone, device):
    """Load a trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` state-dict file.
    encoder_backbone : str
        One of ``"scratch"``, ``"vgg16"``, ``"resnet34"``.
    device : torch.device
        Target device.

    Returns
    -------
    model : torch.nn.Module
        Model in eval mode on *device*.
    in_channels : int
        Number of input channels the model expects.
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
            encoder_weights=None,  # weights loaded from checkpoint
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
    """Normalise by 99th percentile and clip extreme bright pixels."""
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

    # Pad to multiples of 16
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

    # Crop back to original size
    if pad_h > 0 or pad_w > 0:
        prob_map = prob_map[:h, :w]

    return prob_map


def load_val_pairs(val_dir):
    """Find and pair all image/mask files in the validation directory."""
    images_dir = Path(val_dir) / "images"
    masks_dir = Path(val_dir) / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(
            f"Error: subdirectories 'images' and 'masks' "
            f"must exist under {val_dir}"
        )
        raise SystemExit(1)

    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    print(f"Found {len(pairs)} matched image-mask pairs in {val_dir}")
    return pairs


def calculate_metrics(y_pred, y_true):
    """Compute pixel-wise Dice coefficient between two binary masks."""
    intersection = np.logical_and(y_pred, y_true).sum()
    pred_sum = y_pred.sum()
    true_sum = y_true.sum()
    dice = (
        (2.0 * intersection) / (pred_sum + true_sum)
        if (pred_sum + true_sum) > 0
        else 1.0
    )
    return dice


def run_grid_search(density_maps, gt_masks, gt_confluencies, bias_weight):
    """Grid search over Otsu-scaled and absolute-threshold techniques.

    Returns the best candidate dict containing the winning method and
    its parameters.
    """
    print("\nRunning grid search for optimal parameters...")

    # scipy.ndimage.label is faster and matches default 4-connectivity
    from scipy.ndimage import label

    # Pre-calculate Otsu thresholds
    otsu_vals = []
    for dm in density_maps:
        if dm.max() - dm.min() < 1e-6:
            otsu_vals.append(0.0)
        else:
            otsu_vals.append(threshold_otsu(dm))

    # ---- Technique 1: Otsu Threshold Scaling ----
    t_factors = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    closing_radii = [1, 3, 5]
    min_object_sizes = [0, 50, 100, 200]

    otsu_combos = len(t_factors) * len(closing_radii) * len(min_object_sizes)

    # ---- Technique 2: Absolute Thresholding ----
    prob_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    abs_combos = len(prob_thresholds) * len(closing_radii) * len(min_object_sizes)

    total_combos = otsu_combos + abs_combos
    print(
        f"  Otsu-scaled combinations: {otsu_combos}, "
        f"Absolute-threshold combinations: {abs_combos}, "
        f"Total: {total_combos}"
    )

    candidates = []
    pbar = tqdm(total=total_combos, desc="Grid Search")

    # ---------- Otsu-scaled search ----------
    for t_factor in t_factors:
        # Pre-compute thresholded masks
        t_masks = []
        for dm, otsu_val in zip(density_maps, otsu_vals):
            t = otsu_val * t_factor
            t_masks.append(dm > t)

        for cr in closing_radii:
            closed_masks = []
            labeled_masks = []
            comp_sizes_list = []
            for t_mask in t_masks:
                if cr > 0:
                    closed = closing(t_mask, footprint=disk(cr))
                else:
                    closed = t_mask
                closed_masks.append(closed)
                
                # Precompute label and component sizes
                labeled, num = label(closed)
                sizes = np.bincount(labeled.ravel())
                labeled_masks.append(labeled)
                comp_sizes_list.append(sizes)

            for min_obj in min_object_sizes:
                dice_scores = []
                biases = []
                for closed_mask, labeled_mask, comp_sizes, gt_mask, gt_conf in zip(
                    closed_masks, labeled_masks, comp_sizes_list, gt_masks, gt_confluencies
                ):
                    if min_obj > 0:
                        keep = comp_sizes >= min_obj
                        keep[0] = False
                        pred_mask = keep[labeled_mask]
                    else:
                        pred_mask = closed_mask

                    dice = calculate_metrics(pred_mask, gt_mask)
                    dice_scores.append(dice)

                    pred_conf = (pred_mask.sum() / pred_mask.size) * 100
                    biases.append(pred_conf - gt_conf)

                mean_dice = np.mean(dice_scores)
                mean_bias = np.mean(biases)
                score = mean_dice - bias_weight * abs(mean_bias)

                candidates.append({
                    "method": "otsu_scaled",
                    "t_factor": t_factor,
                    "closing_radius": cr,
                    "min_object_size": min_obj,
                    "mean_dice": float(mean_dice),
                    "mean_bias": float(mean_bias),
                    "score": float(score),
                })
                pbar.update(1)

    # ---------- Absolute threshold search ----------
    for prob_thresh in prob_thresholds:
        t_masks = []
        for dm in density_maps:
            t_masks.append(dm > prob_thresh)

        for cr in closing_radii:
            closed_masks = []
            labeled_masks = []
            comp_sizes_list = []
            for t_mask in t_masks:
                if cr > 0:
                    closed = closing(t_mask, footprint=disk(cr))
                else:
                    closed = t_mask
                closed_masks.append(closed)
                
                # Precompute label and component sizes
                labeled, num = label(closed)
                sizes = np.bincount(labeled.ravel())
                labeled_masks.append(labeled)
                comp_sizes_list.append(sizes)

            for min_obj in min_object_sizes:
                dice_scores = []
                biases = []
                for closed_mask, labeled_mask, comp_sizes, gt_mask, gt_conf in zip(
                    closed_masks, labeled_masks, comp_sizes_list, gt_masks, gt_confluencies
                ):
                    if min_obj > 0:
                        keep = comp_sizes >= min_obj
                        keep[0] = False
                        pred_mask = keep[labeled_mask]
                    else:
                        pred_mask = closed_mask

                    dice = calculate_metrics(pred_mask, gt_mask)
                    dice_scores.append(dice)

                    pred_conf = (pred_mask.sum() / pred_mask.size) * 100
                    biases.append(pred_conf - gt_conf)

                mean_dice = np.mean(dice_scores)
                mean_bias = np.mean(biases)
                score = mean_dice - bias_weight * abs(mean_bias)

                candidates.append({
                    "method": "absolute_threshold",
                    "prob_threshold": prob_thresh,
                    "closing_radius": cr,
                    "min_object_size": min_obj,
                    "mean_dice": float(mean_dice),
                    "mean_bias": float(mean_bias),
                    "score": float(score),
                })
                pbar.update(1)

    pbar.close()

    # Sort by score descending, then mean_dice descending, then abs(bias) ascending
    candidates.sort(
        key=lambda x: (x["score"], x["mean_dice"], -abs(x["mean_bias"])),
        reverse=True,
    )

    # Print top 10 candidates
    print("\n" + "-" * 95)
    print(
        f"Top 10 Parameter Combinations "
        f"(Score = Dice - {bias_weight} * |Bias|):"
    )
    print("-" * 95)
    header = (
        f"{'Rank':<5} | {'Method':<18} | {'Param':<12} | "
        f"{'Closing':<7} | {'MinSize':<7} | "
        f"{'Dice':<8} | {'Bias (%)':<12} | {'Score':<8}"
    )
    print(header)
    print("-" * 95)

    for rank, cand in enumerate(candidates[:10], 1):
        if cand["method"] == "otsu_scaled":
            param_str = f"tf={cand['t_factor']:.1f}"
        else:
            param_str = f"pt={cand['prob_threshold']:.2f}"

        print(
            f"{rank:<5} | {cand['method']:<18} | {param_str:<12} | "
            f"{cand['closing_radius']:<7} | {cand['min_object_size']:<7} | "
            f"{cand['mean_dice']:<8.4f} | "
            f"{cand['mean_bias']:>+11.2f}% | {cand['score']:<8.4f}"
        )
    print("-" * 95)

    return candidates[0]


def main():
    args = parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Encoder backbone: {args.encoder_backbone}")

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        raise SystemExit(1)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, in_channels = load_model(args.checkpoint, args.encoder_backbone, device)

    # Load validation pairs
    val_pairs = load_val_pairs(args.val_dir)
    if not val_pairs:
        print("Error: No validation pairs found.")
        raise SystemExit(1)

    # Perform stratified subsampling of validation pairs if requested
    if args.calibrate_size > 0 and args.calibrate_size < len(val_pairs):
        print(f"Selecting a representative subset of {args.calibrate_size} images using stratified sampling...")
        pair_confluencies = []
        for img_path, mask_path in val_pairs:
            gt_mask = plt.imread(str(mask_path))
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.mean(axis=-1)
            gt_mask = gt_mask > 0.5
            gt_conf = (gt_mask.sum() / gt_mask.size) * 100
            pair_confluencies.append(gt_conf)

        # Sort pairs by confluency (with filename as tiebreaker for strict determinism)
        sorted_pairs_with_conf = sorted(
            zip(val_pairs, pair_confluencies), key=lambda x: (x[1], x[0][0].name)
        )

        n_total = len(sorted_pairs_with_conf)
        indices = np.linspace(0, n_total - 1, args.calibrate_size)
        calibrate_indices = sorted(list(set(int(i) for i in indices)))
        # Handle rounding edge cases where we might have fewer unique indices
        for i in range(n_total):
            if len(calibrate_indices) == args.calibrate_size:
                break
            if i not in calibrate_indices:
                calibrate_indices.append(i)
        calibrate_indices = sorted(calibrate_indices)

        selected_val_pairs = [sorted_pairs_with_conf[i][0] for i in calibrate_indices]
        print(f"Sampled validation subset: {len(selected_val_pairs)} images (original val set: {len(val_pairs)})")
    else:
        selected_val_pairs = val_pairs

    # Forward pass: compute density maps and load ground-truth masks
    print("Computing density maps for selected validation images...")
    density_maps = []
    gt_masks = []
    gt_confluencies = []

    for img_path, mask_path in tqdm(selected_val_pairs, desc="Forward Pass"):
        # Load and preprocess image
        raw_image = plt.imread(str(img_path))
        if raw_image.ndim == 3:
            raw_image = raw_image.mean(axis=-1)
        image = preprocess_image(raw_image)

        # Get density map
        density_map = get_density_map(model, image, in_channels, device)
        density_maps.append(density_map)

        # Load ground-truth mask
        gt_mask = plt.imread(str(mask_path))
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.mean(axis=-1)
        gt_mask = gt_mask > 0.5
        gt_masks.append(gt_mask)

        gt_conf = (gt_mask.sum() / gt_mask.size) * 100
        gt_confluencies.append(gt_conf)

    # Run grid search
    best = run_grid_search(density_maps, gt_masks, gt_confluencies, args.bias_weight)

    # Build output config
    config = {
        "method": best["method"],
        "closing_radius": best["closing_radius"],
        "min_object_size": best["min_object_size"],
        "calibration_score": best["score"],
        "calibration_mean_dice": best["mean_dice"],
        "calibration_mean_bias": best["mean_bias"],
    }
    if best["method"] == "otsu_scaled":
        config["t_factor"] = best["t_factor"]
    else:
        config["prob_threshold"] = best["prob_threshold"]

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "optimal_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nOptimal configuration saved to: {config_path}")
    print("Winning parameters:")
    print(f"  Method:           {config['method']}")
    if config["method"] == "otsu_scaled":
        print(f"  Threshold factor: {config['t_factor']}")
    else:
        print(f"  Prob threshold:   {config['prob_threshold']}")
    print(f"  Closing radius:   {config['closing_radius']}")
    print(f"  Min object size:  {config['min_object_size']}")
    print(f"  Composite score:  {config['calibration_score']:.4f}")
    print(f"  Mean Dice:        {config['calibration_mean_dice']:.4f}")
    print(f"  Mean Bias:        {config['calibration_mean_bias']:.2f}%")


if __name__ == "__main__":
    main()
