"""
Split the flat dataset into train / val / test directories.

Groups tiles by their macro-part identifier (everything before ``_tile_``)
so that **all** sibling tiles from the same parent image stay in the same
split.  Files are *copied* (not moved) so the original flat layout is
preserved.

Usage:
    python scripts/split_dataset.py --src dataset --dst dataset
    python scripts/split_dataset.py --src dataset --dst dataset --train-ratio 0.7 --val-ratio 0.15

The script creates the following tree under ``--dst``::

    dst/
      train/images/  train/masks/
      val/images/    val/masks/
      test/images/   test/masks/
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test with macro-part grouping"
    )
    parser.add_argument(
        "--src",
        default="dataset",
        help="Source dataset directory containing images/ and masks/ (default: dataset)",
    )
    parser.add_argument(
        "--dst",
        default="dataset",
        help="Destination root for train/val/test subdirectories (default: dataset)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of macro-part groups assigned to training (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of macro-part groups assigned to validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    return parser.parse_args()


_TILE_PATTERN = re.compile(r"^(.+?)_tile_\d+\.png$")


def extract_macro_id(filename: str) -> str:
    """Extract the macro-part identifier from a tile filename.

    Example: ``218-4_part_10_tile_3.png`` -> ``218-4_part_10``
    """
    match = _TILE_PATTERN.match(filename)
    if match:
        return match.group(1)
    # Fallback: treat the entire stem as the group key
    return Path(filename).stem


def main():
    args = parse_args()

    src_images = Path(args.src) / "images"
    src_masks = Path(args.src) / "masks"

    if not src_images.exists() or not src_masks.exists():
        print(
            f"Error: Expected images/ and masks/ subdirectories under {args.src}"
        )
        raise SystemExit(1)

    # --- Group tiles by macro-part identifier ---
    groups = defaultdict(list)
    for img_path in sorted(src_images.glob("*.png")):
        macro_id = extract_macro_id(img_path.name)
        groups[macro_id].append(img_path.name)

    group_ids = sorted(groups.keys())
    n_groups = len(group_ids)
    total_tiles = sum(len(v) for v in groups.values())

    print(f"Found {n_groups} macro-part groups ({total_tiles} total tiles)")
    print(f"Ratios  -- train: {args.train_ratio}  val: {args.val_ratio}  "
          f"test: {1.0 - args.train_ratio - args.val_ratio:.2f}")

    if args.train_ratio + args.val_ratio > 1.0:
        print("Error: train-ratio + val-ratio must be less than or equal to 1.0")
        raise SystemExit(1)

    # --- Group macro-parts by acquisition prefix (stratification) ---
    acq_to_macro_ids = defaultdict(list)
    for gid in group_ids:
        acq_prefix = gid.split("_")[0]
        acq_to_macro_ids[acq_prefix].append(gid)

    train_ids = []
    val_ids = []
    test_ids = []

    rng = np.random.default_rng(args.seed)
    for acq_prefix, ids in sorted(acq_to_macro_ids.items()):
        shuffled_ids = rng.permutation(ids)
        n_acq = len(shuffled_ids)
        
        n_train = max(1, round(n_acq * args.train_ratio))
        n_val = max(1, round(n_acq * args.val_ratio))
        n_test = n_acq - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = n_acq - n_train - n_test

        train_ids.extend(shuffled_ids[:n_train])
        val_ids.extend(shuffled_ids[n_train : n_train + n_val])
        test_ids.extend(shuffled_ids[n_train + n_val :])

    split_map = {}
    for gid in train_ids:
        split_map[gid] = "train"
    for gid in val_ids:
        split_map[gid] = "val"
    for gid in test_ids:
        split_map[gid] = "test"

    # --- Print split assignment ---
    print("\nGroup assignments:")
    for split_name in ("train", "val", "test"):
        assigned = sorted(gid for gid, s in split_map.items() if s == split_name)
        tile_count = sum(len(groups[gid]) for gid in assigned)
        print(f"  {split_name:5s}: {len(assigned)} groups, {tile_count} tiles")
        for gid in assigned:
            print(f"         - {gid} ({len(groups[gid])} tiles)")

    # --- Create directories and copy files ---
    dst = Path(args.dst)
    for split_name in ("train", "val", "test"):
        split_dir = dst / split_name
        if split_dir.exists():
            shutil.rmtree(str(split_dir))
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_masks = 0
    for gid, split_name in split_map.items():
        for tile_name in groups[gid]:
            # Copy image
            src_img = src_images / tile_name
            dst_img = dst / split_name / "images" / tile_name
            shutil.copy2(str(src_img), str(dst_img))

            # Copy corresponding mask
            mask_name = f"{Path(tile_name).stem}_mask.png"
            src_mask = src_masks / mask_name
            if src_mask.exists():
                dst_mask = dst / split_name / "masks" / mask_name
                shutil.copy2(str(src_mask), str(dst_mask))
            else:
                skipped_masks += 1

            copied += 1

    print(f"\nCopied {copied} image tiles into {dst}")
    if skipped_masks > 0:
        print(f"Warning: {skipped_masks} tiles had no matching mask file")

    # --- Verification summary ---
    for split_name in ("train", "val", "test"):
        img_count = len(list((dst / split_name / "images").glob("*.png")))
        mask_count = len(list((dst / split_name / "masks").glob("*.png")))
        print(f"  {split_name:5s}: {img_count} images, {mask_count} masks")

    print("\nDone. Dataset split complete.")


if __name__ == "__main__":
    main()
