
from pathlib import Path
from ultralytics import YOLO

# Xác định đường dẫn project
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn model
MODEL_PATH = BASE_DIR / "models" / "best.pt"


def load_model():
    """
    Load model YOLOv8 đã train.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print("FoodAI YOLO model loaded successfully!")

    return model
