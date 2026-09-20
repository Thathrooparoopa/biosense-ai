from fastapi import APIRouter


router = APIRouter(
    prefix="/api/health",
    tags=["Health"],
)


@router.get("")
def health_check():
    return {
        "service": "BioSense AI Service",
        "status": "healthy",
    }