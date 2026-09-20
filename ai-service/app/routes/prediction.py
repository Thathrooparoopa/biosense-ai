from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/api/predictions",
    tags=["Predictions"],
)


class PredictionRequest(BaseModel):
    sensor_data: Dict[str, float] = Field(
        ...,
        description="Sensor or VOC feature values used for model inference.",
        min_length=1,
    )


prediction_service = PredictionService()


@router.post("")
def predict(request: PredictionRequest):
    return prediction_service.predict(request.sensor_data)