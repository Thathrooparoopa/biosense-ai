from typing import Dict


class PredictionService:
    """
    Service responsible for model inference.

    The actual trained BioSense AI model will be connected
    in a later ML phase.
    """

    def predict(self, sensor_data: Dict[str, float]) -> dict:
        return {
            "status": "model_not_loaded",
            "message": (
                "Prediction pipeline is ready, but the trained "
                "BioSense AI model has not been loaded yet."
            ),
            "feature_count": len(sensor_data),
            "features_received": list(sensor_data.keys()),
        }