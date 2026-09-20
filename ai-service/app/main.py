from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="BioSense AI Service",
    description=(
        "AI service for the BioSense AI research platform. "
        "Provides endpoints for health checks and future sensor-pattern prediction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "service": "BioSense AI Service",
        "status": "running",
        "version": "1.0.0",
    }