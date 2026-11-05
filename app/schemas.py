
from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FoodPrediction(BaseModel):
    food_name: str
    class_id: int
    confidence: float
    bounding_box: BoundingBox


class RecognitionResponse(BaseModel):
    success: bool
    total_detections: int
    predictions: List[FoodPrediction]

class PredictRequest(BaseModel):
    sessionId: int
    imageUrl: str


class AiPrediction(BaseModel):
    label: str
    confidence: float
    estimatedGrams: float | None = None
    bboxX: float | None = None
    bboxY: float | None = None
    bboxWidth: float | None = None
    bboxHeight: float | None = None


class PredictResponse(BaseModel):
    modelVersion: str
    predictions: List[AiPrediction]