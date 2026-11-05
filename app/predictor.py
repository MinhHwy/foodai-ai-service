
from PIL import Image
from io import BytesIO


def predict_food(model, image_bytes: bytes, confidence: float = 0.25):
    """
    Nhận diện món ăn từ dữ liệu ảnh.

    Parameters:
        model: YOLO model
        image_bytes: Dữ liệu ảnh
        confidence: Ngưỡng confidence

    Returns:
        Danh sách món ăn được nhận diện
    """

    # Chuyển dữ liệu bytes thành ảnh
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Chạy YOLO
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    predictions = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            # Lấy tên class từ model
            food_name = result.names[class_id]

            # Tọa độ bounding box
            coordinates = box.xyxy[0].tolist()

            predictions.append({
                "food_name": food_name,
                "class_id": class_id,
                "confidence": round(score, 4),
                "bounding_box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2)
                }
            })

    return predictions
