"""
Evaluate segmentation performance on the independent test split.

Imports the exact segmentation and inference routines from ``predict.py``
to prevent codebase drift between production inference and evaluation.

Workflow:
    1. Loop through test images blindly (prediction first).
    2. Un-blind: load ground-truth masks and compare pixel-by-pixel.
    3. Compute per-image and aggregate metrics.

Usage:
    python scripts/evaluate.py --test-dir dataset/test
    python scripts/evaluate.py --test-dir dataset/test \\
        --optimal-config output/calibration/optimal_config.json
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import the exact production routines from predict.py
from predict import (
    load_model,
    preprocess_image,
    get_density_map,
    segment_density,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation on the independent test split"
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path to the test split directory (containing images/ and masks/)",
    )
    parser.add_argument(
        "--optimal-config",
        default="output/calibration/optimal_config.json",
        help="Path to calibrated optimal_config.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/training/best_msc_unet.pt",
        help="Path to trained model weights .pt file",
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
        default="output/evaluation",
        help="Directory to save evaluation outputs (default: output/evaluation)",
    )
    return parser.parse_args()


def load_test_pairs(test_dir):
    """Find and pair all image/mask files in the test directory."""
    images_dir = Path(test_dir) / "images"
    masks_dir = Path(test_dir) / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(
            f"Error: subdirectories 'images' and 'masks' "
            f"must exist under {test_dir}"
        )
        raise SystemExit(1)

    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    print(f"Found {len(pairs)} matched image-mask pairs in {test_dir}")
    return pairs


def calculate_metrics(y_pred, y_true):
    """Compute pixel-wise segmentation metrics between two binary masks.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted binary mask (bool).
    y_true : np.ndarray
        Ground-truth binary mask (bool).

    Returns
    -------
    dict
        Dictionary with dice, jaccard, precision, recall.
    """
    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    pred_sum = y_pred.sum()
    true_sum = y_true.sum()

    jaccard = float(intersection / union) if union > 0 else 1.0
    dice = (
        float((2.0 * intersection) / (pred_sum + true_sum))
        if (pred_sum + true_sum) > 0
        else 1.0
    )
    precision = float(intersection / pred_sum) if pred_sum > 0 else 1.0
    recall = float(intersection / true_sum) if true_sum > 0 else 1.0

    return {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
    }


def compute_aggregate_metrics(results):
    """Compute overall summary metrics from per-image results.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys: dice, jaccard, precision, recall,
        gt_confluency, pred_confluency.

    Returns
    -------
    dict
        Aggregate metrics.
    """
    gt_confs = np.array([r["gt_confluency"] for r in results])
    pred_confs = np.array([r["pred_confluency"] for r in results])
    errors = pred_confs - gt_confs

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Pearson correlation
    if len(results) > 1 and np.std(gt_confs) > 0 and np.std(pred_confs) > 0:
        pearson_r = float(np.corrcoef(gt_confs, pred_confs)[0, 1])
    else:
        pearson_r = 0.0

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((gt_confs - np.mean(gt_confs)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mean_dice": float(np.mean([r["dice"] for r in results])),
        "mean_jaccard": float(np.mean([r["jaccard"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_recall": float(np.mean([r["recall"] for r in results])),
        "confluency_mae": mae,
        "confluency_rmse": rmse,
        "confluency_pearson_r": pearson_r,
        "confluency_r2": r2,
    }


def save_correlation_plot(results, metrics, output_dir):
    """Generate and save the predicted-vs-ground-truth correlation scatter."""
    gt_confs = [r["gt_confluency"] for r in results]
    pred_confs = [r["pred_confluency"] for r in results]

    plt.figure(figsize=(8, 7))
    plt.scatter(
        gt_confs,
        pred_confs,
        color="#2b5c8f",
        alpha=0.7,
        edgecolors="k",
        s=60,
        label="Images",
    )

    # Identity line (perfect agreement)
    lims = [0, 100]
    plt.plot(
        lims,
        lims,
        "r--",
        alpha=0.6,
        linewidth=1.5,
        label="Identity (Perfect Agreement)",
    )

    # Trend line
    if len(gt_confs) > 1:
        m, b = np.polyfit(gt_confs, pred_confs, 1)
        x_vals = np.linspace(min(gt_confs), max(gt_confs), 100)
        plt.plot(
            x_vals,
            m * x_vals + b,
            color="#2ca02c",
            linestyle="-",
            linewidth=2,
            label=f"Trendline (y = {m:.2f}x + {b:.2f})",
        )

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("Ground Truth Confluency (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Confluency (%)", fontsize=12, fontweight="bold")
    plt.title(
        f"Test Evaluation: Predicted vs Ground Truth\n"
        f"MAE: {metrics['confluency_mae']:.2f}% | "
        f"R-squared: {metrics['confluency_r2']:.3f} | "
        f"Pearson r: {metrics['confluency_pearson_r']:.3f}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plot_path = os.path.join(output_dir, "test_correlation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation scatter plot to: {plot_path}")


def main():
    args = parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Encoder backbone: {args.encoder_backbone}")

    # Load calibration config
    config_path = Path(args.optimal_config)
    if not config_path.exists():
        print(f"Error: Calibration config not found at {config_path}")
        print("Run scripts/calibrate.py first to generate optimal_config.json.")
        raise SystemExit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded calibration config from: {config_path}")
    print(f"  Method: {config.get('method', 'unknown')}")

    # Verify checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        raise SystemExit(1)

    # Load model (using the exact same function as predict.py)
    print(f"Loading model checkpoint: {args.checkpoint}")
    model, in_channels = load_model(args.checkpoint, args.encoder_backbone, device)

    # Load test pairs
    test_pairs = load_test_pairs(args.test_dir)
    if not test_pairs:
        print("Error: No test pairs found.")
        raise SystemExit(1)

    # ---------------------------------------------------------------
    # Step 1: Blind prediction pass — generate masks without GT
    # Step 2: Un-blind — load GT and compare
    # ---------------------------------------------------------------
    print(f"\nEvaluating {len(test_pairs)} test images...")
    per_image_results = []

    for img_path, mask_path in tqdm(test_pairs, desc="Test Evaluation"):
        name = img_path.name

        # --- Blind prediction (no GT loaded yet) ---
        raw_image = plt.imread(str(img_path))
        if raw_image.ndim == 3:
            raw_image = raw_image.mean(axis=-1)

        image = preprocess_image(raw_image)
        density_map = get_density_map(model, image, in_channels, device)
        pred_mask = segment_density(density_map, config)
        pred_conf = (pred_mask.sum() / pred_mask.size) * 100.0

        # --- Un-blind: load ground-truth mask ---
        gt_mask_raw = plt.imread(str(mask_path))
        if gt_mask_raw.ndim == 3:
            gt_mask_raw = gt_mask_raw.mean(axis=-1)
        gt_mask = gt_mask_raw > 0.5
        gt_conf = (gt_mask.sum() / gt_mask.size) * 100.0

        # --- Pixel-wise comparison ---
        metrics = calculate_metrics(pred_mask, gt_mask)
        metrics.update({
            "filename": name,
            "gt_confluency": round(gt_conf, 2),
            "pred_confluency": round(pred_conf, 2),
            "confluency_error": round(pred_conf - gt_conf, 2),
        })
        per_image_results.append(metrics)

    # ---------------------------------------------------------------
    # Step 3: Aggregate metrics
    # ---------------------------------------------------------------
    agg = compute_aggregate_metrics(per_image_results)

    # Print summary table
    print("\n" + "=" * 65)
    print("TEST EVALUATION SUMMARY")
    print("=" * 65)
    print("Calibration config used:")
    print(f"  Method:          {config.get('method', 'N/A')}")
    if config.get("method") == "otsu_scaled":
        print(f"  Threshold factor: {config.get('t_factor', 'N/A')}")
    elif config.get("method") == "absolute_threshold":
        print(f"  Prob threshold:   {config.get('prob_threshold', 'N/A')}")
    print(f"  Closing radius:   {config.get('closing_radius', 'N/A')}")
    print(f"  Min object size:  {config.get('min_object_size', 'N/A')}")

    print("\nSegmentation Metrics (pixel-wise):")
    print(f"  Mean Dice Coefficient:    {agg['mean_dice']:.4f}")
    print(f"  Mean Jaccard Index (IoU): {agg['mean_jaccard']:.4f}")
    print(f"  Mean Precision:           {agg['mean_precision']:.4f}")
    print(f"  Mean Recall:              {agg['mean_recall']:.4f}")

    print("\nConfluency Estimation Metrics:")
    print(f"  Mean Absolute Error (MAE):  {agg['confluency_mae']:.2f}%")
    print(f"  Root Mean Squared Error:    {agg['confluency_rmse']:.2f}%")
    print(f"  Pearson Correlation (r):    {agg['confluency_pearson_r']:.3f}")
    print(f"  R-squared (R^2):            {agg['confluency_r2']:.3f}")

    print("\nPer-image details:")
    print(
        f"  {'Filename':<40} | {'GT%':>6} | {'Pred%':>6} | "
        f"{'Err%':>7} | {'Dice':>6} | {'IoU':>6}"
    )
    print("-" * 85)
    for r in per_image_results:
        print(
            f"  {r['filename']:<40} | {r['gt_confluency']:>6.1f} | "
            f"{r['pred_confluency']:>6.1f} | {r['confluency_error']:>+7.1f} | "
            f"{r['dice']:>6.4f} | {r['jaccard']:>6.4f}"
        )
    print("=" * 65)

    # ---------------------------------------------------------------
    # Step 4: Save outputs
    # ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # JSON summary
    summary = {
        "config_used": config,
        "num_test_images": len(per_image_results),
        "aggregate_metrics": agg,
        "per_image_results": per_image_results,
    }
    summary_path = os.path.join(args.output_dir, "test_summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved test summary metrics to: {summary_path}")

    # Correlation scatter plot
    save_correlation_plot(per_image_results, agg, args.output_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
