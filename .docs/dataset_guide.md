# Dataset & Split Logic Guide

This document describes the dataset used to train the `cellmodels` 40x magnification model and the group-aware partitioning logic designed to prevent spatial data leakage.

---

## The MSU-Smooth-1-20 Dataset

The 40x magnification model is trained on the public **MSU-Smooth-1-20 MSC dataset**, which consists of:
* **Micrographs**: Phase-contrast microscopy images of mesenchymal stem cells (MSCs).
* **Ground-Truth Masks**: Hand-segmented, pixel-level binary masks showing cell locations (foreground) vs. background.
* **Resolution**: Standard images are high-resolution ($1000 \times 1000$ pixels).
* **Characteristics**: Cells exhibit distinct phase halos (bright refractive borders) and varied, elongated morphology, presenting a challenging segmentation task compared to simple brightfield micrographs.

---

## The Spatial Data Leakage Problem

In biological imaging, datasets are often constructed by dividing large microscopy fields of view (or parent images) into smaller, manageable sub-images or "tiles" (e.g., $256 \times 256$ or $512 \times 512$ pixels).

```
┌───────────────────────────────────────┐
│              Parent Image             │
│  ┌─────────────────┬───────────────┐  │
│  │   Tile 1 (Train)│ Tile 2 (Test) │  │
│  │                 │               │  │
│  ├─────────────────┼───────────────┤  │
│  │   Tile 3 (Train)│ Tile 4 (Train)│  │
│  │                 │               │  │
│  └─────────────────┴───────────────┘  │
└───────────────────────────────────────┘
```

If these tiles are split randomly into training and testing sets:
1. **Structural Overfitting**: Neighboring tiles (sibling tiles) from the same physical culture dish share identical background illumination, optical artifacts, and local cell colony structures.
2. **Artificial Accuracy**: The model learns to recognize specific patterns of that specific well plate, rather than generalizing to unseen cell morphologies. Consequently, holdout test metrics will appear artificially high, but the model will perform poorly on entirely new datasets.

To qualify model performance for real-world laboratory use, **sibling tiles must never be split across train, validation, and test sets.**

---

## Group-Aware Split Logic (`split_dataset.py`)

The dataset preparation pipeline ([split_dataset.py](../scripts/split_dataset.py)) solves this problem by performing a **group-aware partition** based on tile filenames.

### 1. Macro-Part Identification
The script inspects each image filename and extracts a parent ID (or macro-part key) using a regular expression:

* **Naming Convention**: Tiles are expected to follow the format `[parent_id]_tile_[index].png` (e.g., `218-4_part_10_tile_3.png`).
* **Regex Pattern**: `^(.+?)_tile_\d+\.png$` extracts the group prefix `218-4_part_10`.
* **Fallback**: If a filename does not match the tile pattern, the script falls back to treating the entire file stem as its own unique group.

### 2. Group Partitioning
Instead of partitioning individual images, the script groups the filenames by their extracted parent ID and splits the *groups*:

1. **Group Identification**: All matching image-mask pairs are placed in a dictionary keyed by their parent ID.
2. **Deterministic Shuffling**: The unique parent IDs are sorted alphabetically and shuffled using a fixed random seed (`--seed 42` by default) to guarantee reproducibility.
3. **Partition Bounds**: The sorted and shuffled parent IDs are split according to the user-specified ratios:
   * **Train split**: Defaults to `70%` of groups.
   * **Validation split**: Defaults to `15%` of groups.
   * **Test split**: Defaults to `15%` of groups.
4. **Copying Files**: The script creates the split directory structure (`train/`, `val/`, `test/`) and copies both the images and their corresponding masks into their respective directories.

```
dataset_raw/
├── images/
│   ├── parentA_tile_0.png
│   ├── parentA_tile_1.png
│   └── parentB_tile_0.png
└── masks/
    ├── parentA_tile_0_mask.png
    ├── parentA_tile_1_mask.png
    └── parentB_tile_0_mask.png

            │  (Running split_dataset.py)
            ▼

dataset_split/
├── train/
│   ├── images/  --> Contains parentA_tile_0.png, parentA_tile_1.png (Group A)
│   └── masks/   --> Contains parentA_tile_0_mask.png, parentA_tile_1_mask.png
└── test/
    ├── images/  --> Contains parentB_tile_0.png (Group B)
    └── masks/   --> Contains parentB_tile_0_mask.png
```

This split strategy ensures that the holdout test split represents a completely independent set of biological cultures, providing an honest and rigorous evaluation of the model's ability to generalize.
