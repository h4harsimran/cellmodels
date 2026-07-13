# cellmodels Documentation Hub

Welcome to the extended documentation for `cellmodels`. Here you will find deep dives into the model architecture, dataset handling, and parameter calibration strategies.

## Documentation Index

| Document | Description |
|----------|-------------|
| 🔬 [Architecture & Algorithms](architecture.md) | U-Net network topology, loss functions, tiled inference, and post-processing pipeline. |
| 📊 [Dataset & Split Logic](dataset_guide.md) | [MSU-Smooth-1-20](https://www.kaggle.com/datasets/maximsolopov/msu-smooth-1-20) dataset details and group-aware train/val/test splitting to prevent data leakage. |
| 🎛️ [Post-Processing Calibration](calibration_guide.md) | Parameter grid search and confluency bias penalization logic. |

---

## Workflow: Training & Calibrating a New Magnification

When you acquire images at a new objective magnification (e.g., `10x`, `20x`), follow this workflow to train, calibrate, register, and evaluate the new model.

### 1. Folder Structure Setup
Before training, construct a structured dataset directory. You should use `split_dataset.py` to copy your raw flat data into group-aware leakage-free train, val, and test splits:

```bash
# Run dataset splitter
python scripts/split_dataset.py --src dataset_raw --dst dataset_20x --train-ratio 0.7 --val-ratio 0.15
```

This creates the following tree structure under `dataset_20x/`:
```
dataset_20x/
├── train/
│   ├── images/  # training images
│   └── masks/   # training masks (e.g., image_stem_mask.png)
├── val/
│   ├── images/  # validation images
│   └── masks/   # validation masks
└── test/
    ├── images/  # holdout test images
    └── masks/   # holdout test masks
```
For more information on the splitting logic, see the [Dataset & Split Logic Guide](dataset_guide.md).

---

### 2. Training the Model
Run the training script pointing to your train and validation directories.

```bash
# Activate your environment and run training
python scripts/train_unet.py \
    --train-dir dataset_20x/train \
    --val-dir dataset_20x/val \
    --epochs 50 \
    --batch-size 8 \
    --crop-size 256 \
    --lr 1e-4 \
    --encoder-backbone scratch \
    --output-dir output/training_20x
```

#### Key Command Parameters:
* `--train-dir` / `--val-dir`: Path to train and validation split directories.
* `--encoder-backbone`: Backbone architecture (`scratch`, `vgg16`, `resnet34`).
* `--epochs`: Number of epochs to train (20-50 is usually sufficient).
* `--crop-size`: Side length of random training tiles in pixels (e.g., `256` or `512`). **Must be a multiple of 16**.
* `--batch-size`: Training batch size (default: `8`).
* `--output-dir`: Folder where checkpoints and training curves are saved.

After training, the best checkpoint is saved to `output/training_20x/best_msc_unet.pt`.

---

### 3. Parameter Calibration
Run the calibration script on the validation split. This performs an automated grid search to find the optimal post-processing parameters (Otsu threshold factor, closing disk radius, and minimum object size):

```bash
python scripts/calibrate.py \
    --val-dir dataset_20x/val \
    --checkpoint output/training_20x/best_msc_unet.pt \
    --encoder-backbone scratch \
    --bias-weight 0.01 \
    --calibrate-size 15 \
    --output-dir output/calibration_20x
```

#### Key Command Parameters:
* `--checkpoint`: Path to the newly trained model weights.
* `--bias-weight`: Penalty weight for mean absolute confluency bias in grid search (default: `0.01`).
* `--calibrate-size`: Number of validation images to sample using stratified confluency spacing (default: `15`).

Look at the console output or `output/calibration_20x/optimal_config.json` to find the **Optimal Parameters** e.g.:
```json
{
    "method": "otsu_scaled",
    "t_factor": 1.2,
    "closing_radius": 1,
    "min_object_size": 200
}
```
For more information on the scoring function and grid search, see the [Post-Processing Calibration Guide](calibration_guide.md).

---

### 4. Test Evaluation
Evaluate the model against the independent holdout test set using your calibrated parameters to obtain final metrics:

```bash
python scripts/evaluate.py \
    --test-dir dataset_20x/test \
    --optimal-config output/calibration_20x/optimal_config.json \
    --checkpoint output/training_20x/best_msc_unet.pt \
    --encoder-backbone scratch \
    --output-dir output/evaluation_20x
```

#### Output Artifacts:
Check the output directory (`output/evaluation_20x/`) for:
1. `test_summary_metrics.json`: Contains calibration config, per-image results, and final holdout set accuracy (Dice, Jaccard, MAE, RMSE, Pearson $r$, $R^2$).
2. `test_correlation.png`: Scatter plot comparing predicted vs ground-truth confluency percentages.

---

### 5. Registering the Model in the Package
To integrate the newly calibrated model into the public `cellmodels` package API:

1. **Ship weights**: Copy the trained checkpoint `best_msc_unet.pt` to the shipped weights directory:
   ```bash
   cp output/training_20x/best_msc_unet.pt cellmodels/weights/20x.pt
   ```
2. **Register parameters**: Open [confluency.py](../cellmodels/confluency.py) and locate `_CALIBRATED_DEFAULTS` around line 29. Add the new magnification, using the parameters identified in step 3:
   ```python
    _CALIBRATED_DEFAULTS: Dict[str, Tuple[float, int, int]] = {
        "10x": (1.2, 1, 200),
        "20x": (1.2, 1, 200),  # Add your new magnification parameters here: (t_factor, radius, min_size)
    }
   ```

---

## Running Predictions (Inference)

You can predict confluency on new micrographs using either the Command Line Interface (CLI) or the Python API.

### 1. Python API

#### Grayscale & Size Constraints
Images can be raw uint8 or float32. The library automatically handles grayscale conversion and pads the boundaries to multiples of 16 prior to running the model.

```python
from cellmodels import MSCConfluency
import matplotlib.pyplot as plt

# 1. Load the model (specifying the target magnification)
model = MSCConfluency(magnification="10x")

# 2. Read your image (2D numpy array)
image = plt.imread("path/to/my_image.png")
if image.ndim == 3:
    image = image.mean(axis=-1)  # Collapse to grayscale if RGB

# 3. Predict cell probability map & confluency percentage
density_map, confluency_pct = model.predict(image)
print(f"Cell Confluency: {confluency_pct:.2f}%")

# (Optional) Run with custom parameters overriding the calibrated defaults
density_map, confluency_pct = model.predict(
    image, 
    threshold_factor=1.0, 
    closing_radius=3, 
    min_object_size=50
)

# (Optional) Run tiled inference for large images to save memory
density_map, confluency_pct = model.predict(image, max_size=384)
```

---

### 2. Command Line Interface (CLI)
Use the `predict.py` script to run inference on single or multiple images.

#### Folder Structure for CLI Outputs:
Predictions create an output directory (defaulting to `output/confluency/`) structured as:
```
output/confluency/
├── confluency_results.csv           # Summary CSV file
├── confluency_image_001.png.png     # Visual panel overlays
└── confluency_image_002.png.png
```

#### Run on a Single Image:
```bash
python scripts/predict.py path/to/micrograph.png --checkpoint cellmodels/weights/10x.pt --optimal-config cellmodels/weights/10x.json
```

#### Run on a Directory of Images:
```bash
python scripts/predict.py path/to/images_folder/ --checkpoint cellmodels/weights/10x.pt --optimal-config cellmodels/weights/10x.json --output-dir output/predictions_10x
```

#### CLI Parameters:
* `input`: Path to a single image or directory of images.
* `--checkpoint`: Path to the trained model weights `.pt` file (default: `output/training/best_msc_unet.pt`).
* `--optimal-config`: Path to the calibrated `optimal_config.json` containing segmentation threshold, closing, and noise filtering parameters (default: `output/calibration/optimal_config.json`).
* `--encoder-backbone`: Backbone encoder architecture matching the checkpoint (`scratch`, `vgg16`, `resnet34`, default: `scratch`).
* `--device`: Compute device to use (`cpu`, `cuda`, `mps`, default: auto-detect).
* `--output-dir`: Folder to save results (default: `output/confluency`).
* `--no-save-images`: Skips generating overlays, only writes to `confluency_results.csv` for maximum speed.
