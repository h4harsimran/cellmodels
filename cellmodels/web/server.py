import base64
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt

# Add workspace root to sys.path to import from scripts and cellmodels
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from cellmodels.confluency import MSCConfluency, _CALIBRATED_DEFAULTS
from scripts.train_unet import (
    load_pairs,
    MSCDataset,
    build_model,
    BCEDiceLoss,
    compute_dice_coefficient,
)

app = FastAPI(title="cellmodels UI Server")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cached model instances keyed by (magnification, checkpoint_path)
_model_cache: Dict[Tuple[str, Optional[str]], MSCConfluency] = {}


class TrainingState:
    def __init__(self):
        self.is_running = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.progress = 0.0
        self.metrics = []
        self.logs = ["Trainer initialized."]
        self.stop_requested = False
        self.lock = threading.Lock()

    def reset(self, total_epochs: int):
        with self.lock:
            self.is_running = True
            self.current_epoch = 0
            self.total_epochs = total_epochs
            self.progress = 0.0
            self.metrics = []
            self.logs = ["Training initialized."]
            self.stop_requested = False

    def add_log(self, message: str):
        with self.lock:
            self.logs.append(message)
            print(f"[Training] {message}")

    def update_epoch(self, epoch: int, metrics_dict: dict, progress: float):
        with self.lock:
            self.current_epoch = epoch
            self.metrics.append(metrics_dict)
            self.progress = progress

    def set_finished(self, success: bool, msg: str):
        with self.lock:
            self.is_running = False
            self.logs.append(msg)
            if success:
                self.progress = 1.0

    def stop_training(self):
        with self.lock:
            self.stop_requested = True


# Global training state
training_state = TrainingState()


def apply_colormap_base64(density_map: np.ndarray) -> str:
    """Map density map [0,1] to RGB 'hot' colormap and return base64 PNG."""
    cmap = plt.colormaps.get_cmap("hot")
    rgba = cmap(density_map)  # H x W x 4
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_to_base64(cell_mask: np.ndarray) -> str:
    """Convert boolean mask to base64 grayscale PNG."""
    mask_uint8 = cell_mask.astype(np.uint8) * 255
    img = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    magnification: str = Form("10x"),
    checkpoint_file: Optional[UploadFile] = File(None),
    threshold_factor: Optional[float] = Form(None),
    closing_radius: Optional[int] = Form(None),
    min_object_size: Optional[int] = Form(None),
    prob_threshold: Optional[float] = Form(None),
):
    try:
        # Load the uploaded micrograph
        image_bytes = await file.read()
        try:
            raw_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        if raw_img.mode != "L":
            raw_img = raw_img.convert("L")

        image_np = np.array(raw_img, dtype=np.float32)

        # Scale to [0.0, 1.0] if pixel values are uint8 [0, 255]
        if image_np.max() > 1.0:
            image_np /= 255.0

        # Handle optional custom checkpoint upload
        checkpoint_path = None
        if checkpoint_file is not None and checkpoint_file.filename:
            tmp_dir = Path("output/web_tmp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = tmp_dir / checkpoint_file.filename
            with open(checkpoint_path, "wb") as f:
                content = await checkpoint_file.read()
                f.write(content)
            checkpoint_path = str(checkpoint_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use cached model or create and cache a new one
        cache_key = (magnification, checkpoint_path)
        if cache_key not in _model_cache:
            try:
                _model_cache[cache_key] = MSCConfluency(
                    magnification=magnification,
                    checkpoint=checkpoint_path,
                    device=device,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to load model: {str(e)}"
                )
        model = _model_cache[cache_key]

        # Run density map + segmentation (single pass, not via predict())
        density_map = model.get_density_map(image_np)
        cell_mask = model.segment(
            density_map,
            threshold_factor=threshold_factor,
            closing_radius=closing_radius,
            min_object_size=min_object_size,
            prob_threshold=prob_threshold,
        )
        confluency_pct = (cell_mask.sum() / cell_mask.size) * 100.0

        # Convert arrays to base64 images
        density_b64 = apply_colormap_base64(density_map)
        mask_b64 = mask_to_base64(cell_mask)

        return {
            "confluency_pct": round(confluency_pct, 2),
            "density_map": f"data:image/png;base64,{density_b64}",
            "cell_mask": f"data:image/png;base64,{mask_b64}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


def run_training_thread(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    encoder_backbone: str,
    seed: int,
):
    try:
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_state.add_log(f"Training started on device: {device}")
        training_state.add_log(
            f"Backbone: {encoder_backbone} | Learning Rate: {lr} | Batch Size: {batch_size}"
        )

        # Load dataset pairs
        training_state.add_log("Loading image-mask pairs...")
        try:
            train_pairs = load_pairs(train_dir)
            val_pairs = load_pairs(val_dir)
        except Exception as e:
            training_state.set_finished(
                False, f"Failed to load data directories: {str(e)}"
            )
            return

        if not train_pairs:
            training_state.set_finished(
                False, "No training pairs found. Verify train folder structure."
            )
            return
        if not val_pairs:
            training_state.set_finished(
                False, "No validation pairs found. Verify validation folder structure."
            )
            return

        training_state.add_log(
            f"Found {len(train_pairs)} training and {len(val_pairs)} validation pairs."
        )

        # Build model
        model, in_channels = build_model(encoder_backbone, device)

        # Datasets & loaders
        training_state.add_log("Preloading datasets into memory...")
        train_dataset = MSCDataset(
            train_pairs,
            crop_size=512,
            augment=True,
            channels=in_channels,
        )
        val_dataset = MSCDataset(
            val_pairs,
            crop_size=512,
            augment=False,
            channels=in_channels,
        )

        use_cuda = device.type == "cuda"
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=use_cuda,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=use_cuda,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = BCEDiceLoss(bce_weight=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        os.makedirs(output_dir, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            if training_state.stop_requested:
                training_state.set_finished(False, "Training stopped by user request.")
                return

            model.train()
            epoch_train_losses = []

            # Simple manual batch loop so we can abort instantly
            for batch_idx, (imgs, masks) in enumerate(train_loader):
                if training_state.stop_requested:
                    training_state.set_finished(
                        False, "Training stopped by user request."
                    )
                    return

                imgs = imgs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_train_losses.append(loss.item())

                # Update fine-grained progress
                batch_progress = (epoch - 1) / epochs + (
                    batch_idx / len(train_loader)
                ) / epochs
                with training_state.lock:
                    training_state.progress = batch_progress

            # Validation
            model.eval()
            epoch_val_losses = []
            epoch_val_dices = []

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    epoch_val_losses.append(loss.item())
                    dice = compute_dice_coefficient(outputs, masks)
                    epoch_val_dices.append(dice)

            avg_train_loss = float(np.mean(epoch_train_losses))
            avg_val_loss = float(np.mean(epoch_val_losses))
            avg_val_dice = float(np.mean(epoch_val_dices))

            scheduler.step()

            # Log progress
            training_state.add_log(
                f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
            )

            # Save checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(output_dir, "best_msc_unet.pt")
                torch.save(model.state_dict(), checkpoint_path)
                training_state.add_log(
                    f"Saved new best checkpoint to: {checkpoint_path}"
                )

            metrics_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
            }
            training_state.update_epoch(epoch, metrics_dict, epoch / epochs)

        training_state.set_finished(True, "Training finished successfully.")

    except Exception as e:
        training_state.set_finished(False, f"Critical error during training: {str(e)}")


@app.post("/api/train/start")
async def api_train_start(
    train_dir: str = Form(...),
    val_dir: str = Form(...),
    output_dir: str = Form("output/web_training"),
    epochs: int = Form(10),
    lr: float = Form(1e-4),
    batch_size: int = Form(8),
    encoder_backbone: str = Form("scratch"),
    seed: int = Form(42),
):
    if training_state.is_running:
        raise HTTPException(status_code=400, detail="Training is already running.")

    # Verify paths exist
    if not os.path.isdir(train_dir):
        raise HTTPException(
            status_code=400, detail=f"Train directory does not exist: {train_dir}"
        )
    if not os.path.isdir(val_dir):
        raise HTTPException(
            status_code=400, detail=f"Validation directory does not exist: {val_dir}"
        )

    training_state.reset(epochs)

    # Start background thread
    t = threading.Thread(
        target=run_training_thread,
        args=(
            train_dir,
            val_dir,
            output_dir,
            epochs,
            lr,
            batch_size,
            encoder_backbone,
            seed,
        ),
        daemon=True,
    )
    t.start()

    return {"status": "started"}


@app.get("/api/train/status")
async def api_train_status():
    with training_state.lock:
        return {
            "is_running": training_state.is_running,
            "current_epoch": training_state.current_epoch,
            "total_epochs": training_state.total_epochs,
            "progress": round(training_state.progress, 4),
            "metrics": training_state.metrics,
            "logs": training_state.logs,
        }


@app.post("/api/train/stop")
async def api_train_stop():
    if not training_state.is_running:
        return {"status": "not running"}

    training_state.stop_training()
    return {"status": "stopping"}


@app.get("/api/parameters/{magnification}")
async def api_get_parameters(magnification: str):
    """Retrieve optimal/calibrated parameters for a given magnification.

    If the calibration file does not exist, returns fallback default values.
    """
    weights_dir = Path(__file__).parent.parent / "weights"
    config_path = weights_dir / f"{magnification}.json"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass  # Fall through to default options on parse failure

    # Try calibration defaults for this magnification
    if magnification in _CALIBRATED_DEFAULTS:
        t_factor, c_radius, min_size = _CALIBRATED_DEFAULTS[magnification]
        return {
            "method": "otsu_scaled",
            "t_factor": t_factor,
            "closing_radius": c_radius,
            "min_object_size": min_size,
        }

    # Generic fallback parameters if nothing is available
    return {
        "method": "otsu_scaled",
        "t_factor": 1.0,
        "closing_radius": 3,
        "min_object_size": 50,
    }


# Serve Frontend SPA
@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("index.html not found.", status_code=404)
    return FileResponse(index_path)
