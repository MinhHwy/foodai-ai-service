import os
from urllib.parse import urlparse
from io import BytesIO

import httpx
from PIL import Image

from app.schemas import (
    RecognitionResponse,
    PredictRequest,
    PredictResponse
)
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_model
from app.predictor import predict_food



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model khi khởi động API
    app.state.model = load_model()

    yield

    # Giải phóng model khi tắt API
    app.state.model = None


app = FastAPI(
    title="FoodAI AI Service",
    description="Vietnamese Food Recognition API using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)


# CORS: cho phép Frontend truy cập API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "message": "Welcome to FoodAI AI Service",
        "status": "running"
    }


@app.get("/health")
def health(request: Request):
    return {
        "success": True,
        "message": "FoodAI AI Service is healthy",
        "model_loaded": request.app.state.model is not None
    }


@app.post(
    "/api/v1/recognize",
    response_model=RecognitionResponse
)
async def recognize_food(
    request: Request,
    file: UploadFile = File(...)
):
    # Kiểm tra định dạng file
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WEBP images are supported"
        )

    try:
        # Đọc dữ liệu ảnh
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Image file is empty"
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB"
            )

        # Gọi model YOLO để nhận diện
        predictions = predict_food(
            model=request.app.state.model,
            image_bytes=image_bytes,
            confidence=0.25
        )

        return {
            "success": True,
            "total_detections": len(predictions),
            "predictions": predictions
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Food recognition failed: {str(error)}"
        )

    finally:
        await file.close()


@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict_from_url(
    data: PredictRequest,
    request: Request
):
    try:
        # Kiểm tra URL ảnh
        parsed_url = urlparse(data.imageUrl)

        if parsed_url.scheme != "https":
            raise HTTPException(
                status_code=400,
                detail="Image URL must use HTTPS"
            )

        if parsed_url.hostname != "res.cloudinary.com":
            raise HTTPException(
                status_code=400,
                detail="Only Cloudinary image URLs are supported"
            )

        # Tải ảnh từ Cloudinary
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=False
        ) as client:

            response = await client.get(data.imageUrl)

            response.raise_for_status()

            image_bytes = response.content

        # Kiểm tra kích thước
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Image is empty"
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB"
            )

        # Đọc kích thước ảnh
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = image.size

        # Chạy YOLO
        predictions = predict_food(
            model=request.app.state.model,
            image_bytes=image_bytes,
            confidence=0.25
        )

        results = []

        for item in predictions:
            box = item["bounding_box"]

            # Chuyển bounding box pixel sang tọa độ chuẩn hóa
            x1 = box["x1"] / image_width
            y1 = box["y1"] / image_height
            x2 = box["x2"] / image_width
            y2 = box["y2"] / image_height

            results.append({
                "label": item["food_name"],
                "confidence": item["confidence"],
                "estimatedGrams": None,

                # Tọa độ tâm bounding box
                "bboxX": round((x1 + x2) / 2, 6),
                "bboxY": round((y1 + y2) / 2, 6),
                "bboxWidth": round(x2 - x1, 6),
                "bboxHeight": round(y2 - y1, 6)
            })

        return {
            "modelVersion": os.getenv(
                "FOODAI_MODEL_VERSION",
                "foodai-yolov8-v1"
            ),
            "predictions": results
        }

    except HTTPException:
        raise

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot download image: {str(error)}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )