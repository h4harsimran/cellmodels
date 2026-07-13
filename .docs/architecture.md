# Architecture Deep Dive

This document provides a technical deep dive into the network architecture, inference data flow, and training pipeline of `cellmodels`.

---

## System Overview

`cellmodels` employs a **U-Net** encoder-decoder convolutional network to estimate mesenchymal stem cell (MSC) confluency from grayscale micrographs. Rather than predicting a single scalar confluency value directly from the image, the pipeline uses a two-stage approach:

1. **Pixel-Level Prediction**: The U-Net generates a continuous **density map** (probability map of cell presence).
2. **Post-Processing & Area Estimation**: The density map is converted to a binary cell mask via thresholding and morphology, and confluency is calculated as the ratio of cell pixels to total pixels.

```
Raw Image (2D) ──► Preprocessing ──► Tiled Inference ──► Density Map (2D) ──► Segmentation ──► Confluency %
```

---

## U-Net Architecture

The network implementation ([unet.py](../cellmodels/unet.py)) is a classic U-Net featuring symmetric contracting (encoder) and expanding (decoder) paths, connected by skip connections to preserve high-resolution spatial details.

```
Input Image (1 × H × W)
     │
     ▼
┌───────────┐     ┌───────────┐
│ Encoder 1 │────►│ Decoder 1 │──► 1×1 Conv ──► Sigmoid ──► Density Map (1 × H × W)
│ (16 feat) │     │ (16 feat) │
└─────┬─────┘     └─────▲─────┘
      │ pool            │ upconv + concat
┌─────▼─────┐     ┌─────┴─────┐
│ Encoder 2 │────►│ Decoder 2 │
│ (32 feat) │     │ (32 feat) │
└─────┬─────┘     └─────▲─────┘
      │ pool            │ upconv + concat
┌─────▼─────┐     ┌─────┴─────┐
│ Encoder 3 │────►│ Decoder 3 │
│ (64 feat) │     │ (64 feat) │
└─────┬─────┘     └─────▲─────┘
      │ pool            │ upconv + concat
┌─────▼─────┐     ┌─────┴─────┐
│ Encoder 4 │────►│ Decoder 4 │
│ (128 feat)│     │ (128 feat)│
└─────┬─────┘     └─────▲─────┘
      │ pool            │ upconv
┌─────▼─────────────────┴─────┐
│      Bottleneck (256 feat)  │
└─────────────────────────────┘
```

### 1. Encoder Path (Contracting)
The contracting path extracts high-level semantic features while reducing spatial dimensions. It consists of four resolution stages. Each stage contains:
* Two consecutive 3×3 convolutions (padding=1, no bias).
* Batch normalization (`BatchNorm2d`) after each convolution.
* In-place Rectified Linear Unit (`ReLU`) activations.
* A 2×2 Max Pooling layer (stride=2) for spatial downsampling.

The initial layer projects the 1-channel grayscale input to `init_features=16`. The number of feature channels doubles at each downsampling step: `16 → 32 → 64 → 128`.

### 2. Bottleneck
The bottleneck connects the encoder and decoder paths. It applies two 3×3 convolutions with batch normalization and ReLU activations, increasing the feature depth to `256` channels without changing the spatial dimensions.

### 3. Decoder Path (Expanding)
The expanding path restores spatial dimensions, enabling precise pixel-level localization. It features four stages matching the encoder resolution:
* Upsampling is performed via a 2×2 transposed convolution (stride=2), which halves the channel depth.
* The upsampled features are concatenated with the corresponding high-resolution feature map from the encoder path (the **skip connection**).
* Two 3×3 convolutions, each followed by batch normalization and ReLU, process the concatenated features.

Skip connections are crucial: they inject fine-grained spatial details directly into the decoder, helping resolve precise cell borders that are otherwise lost during pooling.

### 4. Output Layer
The final layer is a 1×1 convolution followed by a `Sigmoid` activation function. It maps the 16-channel feature map of the final decoder stage to a single-channel density map of shape `(1, H, W)`, representing the probability $P(\text{cell} \mid x, y) \in [0, 1]$ for each pixel.

---

## Inference Pipeline

### 1. Preprocessing
Images are normalized using a robust quantile-scaling method implemented in the `Base` class ([base_model.py](../cellmodels/base_model.py)):

```python
def preprocess(x: np.ndarray) -> np.ndarray:
    x_copy = x.copy().astype(np.float32)
    q = np.quantile(x_copy, 0.99)
    if q > 0:
        x_copy /= q              # Normalize by 99th percentile
    x_copy[x_copy > 1.0] = 1.0   # Clip extreme bright outliers
    return x_copy
```
This approach makes inference robust to differences in light intensity, exposure times, and camera noise.

### 2. Tiled Inference
For large micrographs that could exceed GPU or CPU memory limitations, the inference module supports tiling via the `max_size` argument:
* **Standard Inference**: Images are padded so that dimensions are multiples of 16 (required by the U-Net's 4-stage downsampling) and processed as a single tensor.
* **Tiled Inference**: The input is divided into non-overlapping tiles of size `max_size × max_size` (which must be a multiple of 16). Tiles are run through the U-Net independently and stitched back together.
> [!NOTE]
> Currently, tiles are stitched with simple hard boundaries. Overlap-and-crop or blending strategies are planned for future releases to prevent boundary prediction artifacts.

---

## Post-Processing & Segmentation

The `segment()` method converts the raw U-Net probability density map into a binary cell mask through three main steps:

1. **Thresholding**: A binary mask is created by applying a threshold.
2. **Morphological Closing**: A binary closing operation with a disk-shaped footprint is applied to fill small gaps/holes within cell bodies.
3. **Small Object Removal**: Connected components smaller than a specified pixel size are removed to eliminate noise.

```
Density Map ──► Thresholding ──► Morphological Closing ──► Small Object Removal ──► Binary Cell Mask
```

### Parameter Loading Precedence

To provide maximum flexibility and reliable defaults, the library implements a strict precedence order for post-processing parameters:

```
1. Explicit function arguments in predict() or segment()
       ▼ (if None)
2. Shipped JSON configuration file (e.g., cellmodels/weights/10x.json)
       ▼ (if JSON missing)
3. Hardcoded defaults in _CALIBRATED_DEFAULTS (cellmodels/confluency.py)
```

### Current Configured Parameter Values

For the **10x** magnification model, the parameters under each level of precedence are:

| Parameter | Shipped JSON (`10x.json`) | Package Fallback (`_CALIBRATED_DEFAULTS`) | Description |
|:---|:---:|:---:|:---|
| **Method** | `otsu_scaled` | `otsu_scaled` | Dynamic Otsu vs. absolute cutoff |
| **Threshold Factor (`t_factor`)** | `1.2` | `0.8` | Multiplier for the Otsu threshold value |
| **Closing Radius (`closing_radius`)**| `5` | `1` | Radius of disk footprint (pixels) |
| **Min Object Size (`min_object_size`)**| `0` | `200` | Area threshold for filtering small noise |

---

## Training Pipeline

The training pipeline ([train_unet.py](../scripts/train_unet.py)) is designed to be highly reproducible and computationally efficient.

### 1. Data Loader & Augmentation
* **Random Cropping**: Extracts tiles of size `512×512` from normalized training images during training to ensure uniform batch sizes.
* **Online Augmentations**: Applies random horizontal flips, vertical flips, and 90-degree rotations to artificially increase dataset diversity and prevent overfitting.

### 2. Hybrid Loss Function
The model is optimized using a hybrid loss combining Binary Cross-Entropy (BCE) and Dice Loss:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

* **BCE Loss**: Evaluates pixel-wise classification accuracy, providing stable gradients early in training.
* **Dice Loss**: Evaluates region overlap, directly optimizing the Dice coefficient and balancing the foreground-background class imbalance:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i y_i}{\sum_{i} p_i + \sum_{i} y_i}$$

where $p_i$ is the predicted probability and $y_i$ is the ground-truth binary label for pixel $i$.

### 3. Optimization details
* **Optimizer**: Adam optimizer with a base learning rate of $\eta_0 = 10^{-4}$.
* **LR Scheduler**: Cosine Annealing Learning Rate scheduler that decays the learning rate down to $10^{-6}$ over the course of training.
* **Hardware Optimizations**: Employs pinned memory (`pin_memory=True`) and non-blocking CUDA transfers (`non_blocking=True`) to maximize CPU-to-GPU data pipeline throughput.
