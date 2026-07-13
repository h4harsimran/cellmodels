# Post-Processing Calibration Guide

This document describes the parameters and optimization algorithms used to calibrate the `cellmodels` segmentation post-processing pipeline on validation data.

---

## Why Calibration is Necessary

Deep learning segmentation models output a continuous probability value $P(\text{cell}) \in [0, 1]$ for each pixel. To convert this probability map into a binary cell mask (and compute confluency), we must apply thresholding and morphology.

Optimizing for the highest pixel-wise **Dice Coefficient** (intersection-over-union) often leads to a systematic **confluency bias**. For example, a model might segment cell shapes perfectly (achieving high Dice) but consistently undercount borders slightly, leading to a systematic -5% error in total confluency area.

To resolve this, `cellmodels` uses a grid search on validation data that optimizes a composite score penalizing confluency area estimation bias.

---

## Post-Processing Parameters

The post-processing pipeline uses three parameters:

1. **Threshold Factor (`t_factor`)**: A multiplier applied to the Otsu threshold value calculated from the probability map (referred to in code as the density map). It scales the segmentation threshold dynamically based on image contrast:
   $$\text{Threshold} = \text{Otsu Threshold} \times t_{\text{factor}}$$
2. **Closing Disk Radius (`closing_radius`)**: The radius of a disk-shaped structural element used in a binary morphological closing operation. Closing fills small dark holes and gaps inside cell bodies.
3. **Minimum Object Size (`min_object_size`)**: Connected components in the binary mask with an area smaller than this threshold (in pixels) are discarded, removing high-frequency background noise.

---

## Stratified Calibration Subsetting

To ensure calibration is robust across all stages of cell growth, the calibration script ([calibrate.py](../scripts/calibrate.py)) uses confluency-based stratification to sample validation images:

1. **Read Masks**: The script pre-reads the ground-truth masks for all validation images.
2. **Compute Confluency**: It calculates the ground-truth confluency percentage for each image.
3. **Stratified Selection**: It sorts the validation images by confluency and samples `calibrate_size` (default: 15) images at equal intervals across the entire range (e.g. from 3.6% to 67% confluency).

This ensures that the calibration set contains a balanced representation of low, medium, and high-density cell cultures.

---

## Bias-Penalized Optimization Score

The grid search evaluates combinations of parameters over the stratified validation subset. For each parameter combination, it computes:

* **Mean Dice**: The average pixel-level Dice coefficient across the subset.
* **Mean Bias**: The average difference between the predicted confluency and ground-truth confluency (predicted % - ground truth %).

The optimal parameter set is chosen by maximizing a composite **Calibration Score**:

$$\text{Score} = \text{Mean Dice} - \beta \times |\text{Mean Bias}|$$

where $\beta$ is the `--bias-weight` parameter (default: `0.01`).

### Rationale for the Bias Weight $\beta = 0.01$
* A bias weight of $0.01$ means that a **1.0%** systematic confluency error (Mean Bias) reduces the overall score by $0.01$.
* This allows the optimizer to trade off a very small amount of pixel-wise overlap (Dice) if it significantly reduces systematic under/over-estimation of cell area, producing a highly accurate confluency estimator.

---

## Running Calibration & Registration

### 1. Execute Grid Search
Run the calibration script. By default, it tests combinations of:
* `t_factor` in $[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]$
* `closing_radius` in $[0, 1, 2, 3, 4, 5]$
* `min_object_size` in $[0, 50, 100, 150, 200, 300]$

```bash
python scripts/calibrate.py \
    --val-dir dataset/val \
    --checkpoint output/training/best_msc_unet.pt \
    --bias-weight 0.01 \
    --calibrate-size 15 \
    --output-dir output/calibration
```

The script prints a summary table of the top 10 parameter combinations ranked by composite score, and writes the best configuration to `output/calibration/optimal_config.json`:

```json
{
    "method": "otsu_scaled",
    "t_factor": 1.2,
    "closing_radius": 5,
    "min_object_size": 0,
    "calibration_score": 0.9069,
    "calibration_mean_dice": 0.9088,
    "calibration_mean_bias": -0.1918
}
```

### 2. Verify on Holdout Test Split
Once calibration is complete, evaluate the model on the independent holdout test split using the optimal configuration:

```bash
python scripts/evaluate.py \
    --test-dir dataset/test \
    --optimal-config output/calibration/optimal_config.json \
    --checkpoint output/training/best_msc_unet.pt \
    --output-dir output/evaluation
```
This produces final, unbiased evaluation metrics (MAE, RMSE, Pearson $r$, $R^2$) to verify the model's accuracy.
