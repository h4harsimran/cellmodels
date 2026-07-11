"""
Train U-Net model on MSC phase-contrast dataset.

Reads training and validation data from separate on-disk directories to
prevent spatial data leakage between sibling tiles.

Usage:
    # Phase 1 — scratch U-Net baseline
    python scripts/train_unet.py --train-dir dataset/train --val-dir dataset/val --epochs 50

    # Phase 2 — transfer-learning pivot with VGG16 encoder
    python scripts/train_unet.py --train-dir dataset/train --val-dir dataset/val \\
        --encoder-backbone vgg16 --epochs 50

    # With custom settings
    python scripts/train_unet.py --train-dir dataset/train --val-dir dataset/val \\
        --epochs 100 --lr 5e-5 --crop-size 256 --device cuda
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cellmodels.unet import UNet
from cellmodels.losses import BCEDiceLoss, compute_dice_coefficient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-Net model on MSC phase-contrast dataset"
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/train",
        help="Path to the training split directory (containing images/ and masks/)",
    )
    parser.add_argument(
        "--val-dir",
        default="dataset/val",
        help="Path to the validation split directory (containing images/ and masks/)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/training",
        help="Directory to save training checkpoints and loss curve",
    )
    parser.add_argument(
        "--encoder-backbone",
        default="scratch",
        choices=["scratch", "vgg16", "resnet34"],
        help=(
            "Encoder backbone to use. 'scratch' uses the custom U-Net; "
            "'vgg16' or 'resnet34' uses transfer learning with ImageNet "
            "pre-trained weights via segmentation-models-pytorch (default: scratch)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Crop size for training image tiles (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatability (default: 42)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )
    return parser.parse_args()


def load_pairs(split_dir):
    """Find and pair all image/mask files in a split directory."""
    images_dir = Path(split_dir) / "images"
    masks_dir = Path(split_dir) / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(
            f"Error: dataset subdirectories 'images' and 'masks' "
            f"must exist under {split_dir}"
        )
        raise SystemExit(1)

    pairs = []
    for img_path in images_dir.glob("*.png"):
        mask_name = f"{img_path.stem}_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    print(f"Found {len(pairs)} matched image-mask pairs in {split_dir}")
    return sorted(pairs)


class MSCDataset(Dataset):
    def __init__(self, pairs, crop_size=512, augment=False, channels=1):
        self.pairs = pairs
        self.crop_size = crop_size
        self.augment = augment
        self.channels = channels

        # Preload and cache all images and masks in memory to bypass disk I/O bottlenecks
        print(f"Preloading {len(pairs)} image-mask pairs into memory...")
        self.cached_imgs = []
        self.cached_masks = []
        for img_path, mask_path in pairs:
            # Load and preprocess image
            raw_img = plt.imread(str(img_path))
            if raw_img.ndim == 3:
                raw_img = raw_img.mean(axis=-1)
            img = self.preprocess(raw_img)
            self.cached_imgs.append(img)

            # Load and preprocess mask
            mask = plt.imread(str(mask_path))
            if mask.ndim == 3:
                mask = mask.mean(axis=-1)
            mask = (mask > 0.5).astype(np.float32)
            self.cached_masks.append(mask)

    def __len__(self):
        return len(self.pairs)

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Normalise input + ignore extreme bright pixels (matches cellmodels Base)."""
        x_copy = x.copy().astype(np.float32)
        q = np.quantile(x_copy, 0.99)
        if q > 0:
            x_copy /= q
        x_copy[x_copy > 1.0] = 1.0
        return x_copy

    def __getitem__(self, idx):
        img = self.cached_imgs[idx]
        mask = self.cached_masks[idx]

        # Apply random cropping if crop_size is smaller than image size
        h, w = img.shape
        if self.crop_size > 0 and self.crop_size < h and self.crop_size < w:
            if self.augment:
                # Random crop
                py = random.randint(0, h - self.crop_size)
                px = random.randint(0, w - self.crop_size)
            else:
                # Center crop
                py = (h - self.crop_size) // 2
                px = (w - self.crop_size) // 2

            img = img[py : py + self.crop_size, px : px + self.crop_size]
            mask = mask[py : py + self.crop_size, px : px + self.crop_size]

        # Apply augmentations (only during training)
        if self.augment:
            # Horizontal Flip
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            # Vertical Flip
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            # Random 90-degree rotations
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k)
                mask = np.rot90(mask, k=rot_k)

        # Convert to float PyTorch tensors with shape [C, H, W]
        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0)

        # Replicate grayscale to 3 channels for transfer-learning encoders
        if self.channels == 3:
            img_tensor = img_tensor.expand(3, -1, -1).contiguous()

        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)

        return img_tensor, mask_tensor


def build_model(encoder_backbone, device):
    """Construct the segmentation model based on the encoder backbone choice.

    Returns
    -------
    model : torch.nn.Module
        The initialised model moved to *device*.
    in_channels : int
        Number of input channels the model expects (1 for scratch, 3 for SMP).
    """
    if encoder_backbone == "scratch":
        print("Initializing custom U-Net (scratch, init_features=16)...")
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

        print(
            f"Initializing SMP U-Net with {encoder_backbone} encoder "
            f"(ImageNet pretrained)..."
        )
        model = smp.Unet(
            encoder_name=encoder_backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        in_channels = 3

    model = model.to(device)
    return model, in_channels


def plot_learning_curves(train_losses, val_losses, val_dices, output_dir):
    """Save the loss and dice curves plot."""
    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(
        val_losses,
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Epochs", fontsize=12, fontweight="bold")
    plt.ylabel("Loss (BCE + Dice)", fontsize=12, fontweight="bold")
    plt.title(
        "U-Net Training & Validation Loss Curves", fontsize=14, fontweight="bold"
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves plot to: {plot_path}")

    # Dice curves
    plt.figure(figsize=(10, 6))
    plt.plot(val_dices, label="Validation Dice", color="#2ca02c", linewidth=2)
    plt.xlabel("Epochs", fontsize=12, fontweight="bold")
    plt.ylabel("Dice Coefficient", fontsize=12, fontweight="bold")
    plt.title("U-Net Validation Dice Coefficient", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    dice_plot_path = os.path.join(output_dir, "dice_curve.png")
    plt.savefig(dice_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dice curve plot to: {dice_plot_path}")


def main():
    args = parse_args()

    # Set seed for repeatability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training on device: {device}")
    print(f"Encoder backbone: {args.encoder_backbone}")

    # Build model
    model, in_channels = build_model(args.encoder_backbone, device)

    # Load pairs from pre-separated directories (no runtime shuffling/splitting)
    train_pairs = load_pairs(args.train_dir)
    val_pairs = load_pairs(args.val_dir)

    if not train_pairs:
        print("Error: No training pairs found. Check --train-dir.")
        raise SystemExit(1)
    if not val_pairs:
        print("Error: No validation pairs found. Check --val-dir.")
        raise SystemExit(1)

    print(f"Dataset Split (from disk):")
    print(f"  - Training samples:    {len(train_pairs)}")
    print(f"  - Validation samples:  {len(val_pairs)}")

    # Create datasets and loaders
    train_dataset = MSCDataset(
        train_pairs,
        crop_size=args.crop_size,
        augment=True,
        channels=in_channels,
    )
    val_dataset = MSCDataset(
        val_pairs,
        crop_size=args.crop_size,
        augment=False,
        channels=in_channels,
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
    )

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss(bce_weight=0.5)

    # Cosine annealing LR schedule for faster convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Logs
    train_losses = []
    val_losses = []
    val_dices = []
    best_val_loss = float("inf")

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # 1. Training loop
        model.train()
        epoch_train_losses = []
        epoch_train_dices = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, masks in train_pbar:
            imgs = imgs.to(device, non_blocking=use_cuda)
            masks = masks.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            with torch.no_grad():
                dice = compute_dice_coefficient(outputs, masks)
            epoch_train_dices.append(dice)

            train_pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_dice = np.mean(epoch_train_dices)
        train_losses.append(avg_train_loss)

        # 2. Validation loop
        model.eval()
        epoch_val_losses = []
        epoch_val_dices = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=use_cuda)
                masks = masks.to(device, non_blocking=use_cuda)

                outputs = model(imgs)
                loss = criterion(outputs, masks)

                epoch_val_losses.append(loss.item())
                dice = compute_dice_coefficient(outputs, masks)
                epoch_val_dices.append(dice)

        avg_val_loss = np.mean(epoch_val_losses)
        avg_val_dice = np.mean(epoch_val_dices)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        # Step the LR scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch} Summary:")
        print(
            f"  - Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f}"
        )
        print(f"  - Val Loss:   {avg_val_loss:.4f} | Val Dice:   {avg_val_dice:.4f}")
        print(f"  - LR: {current_lr:.2e}")

        # Save checkpoint if val loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_msc_unet.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  => Saved new best checkpoint to: {checkpoint_path}")

    print("\nTraining completed successfully!")
    plot_learning_curves(train_losses, val_losses, val_dices, args.output_dir)


if __name__ == "__main__":
    main()
