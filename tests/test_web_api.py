import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from cellmodels.web.server import app

client = TestClient(app)

def test_index_endpoint():
    """Verify that root serves the index page successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "cellmodels" in response.text

def test_predict_endpoint():
    """Verify prediction endpoint with a real micrograph from the dataset."""
    test_image_path = ROOT_DIR / "dataset" / "images" / "218-4_part_1_tile_6.png"
    
    if not test_image_path.exists():
        pytest.skip(f"Test image not found at {test_image_path}, skipping predict API test.")
        
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
        
    response = client.post(
        "/api/predict",
        files={"file": ("test_tile.png", image_bytes, "image/png")},
        data={
            "magnification": "10x",
            "closing_radius": "3",
            "min_object_size": "50",
            "threshold_factor": "1.0"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "confluency_pct" in data
    assert isinstance(data["confluency_pct"], (int, float))
    assert "density_map" in data
    assert data["density_map"].startswith("data:image/png;base64,")
    assert "cell_mask" in data
    assert data["cell_mask"].startswith("data:image/png;base64,")

def test_train_status_endpoint():
    """Verify the training status endpoint returns the correct default idle state."""
    response = client.get("/api/train/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is False
    assert data["current_epoch"] == 0
    assert len(data["metrics"]) == 0
    assert len(data["logs"]) > 0

def test_train_start_invalid_folders():
    """Verify training rejects invalid dataset paths with status 400."""
    response = client.post(
        "/api/train/start",
        data={
            "train_dir": "nonexistent/train",
            "val_dir": "nonexistent/val",
            "epochs": "5"
        }
    )
    assert response.status_code == 400
    assert "directory does not exist" in response.json()["detail"]
