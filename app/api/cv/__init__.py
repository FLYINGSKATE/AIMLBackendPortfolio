from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Import and include sub-routers
try:
    from .image_classification import router as image_classification_router
    router.include_router(
        image_classification_router,
        prefix="/classify",
        tags=["Image Classification"]
    )
except ImportError as e:
    logger.warning(f"Image classification not available: {e}")

# Only import object detection if dependencies are available
try:
    from .object_detection import router as object_detection_router
    router.include_router(
        object_detection_router,
        prefix="/object-detection",
        tags=["Object Detection"]
    )
except Exception as e:
    logger.warning(f"Object detection not available: {e}")
    
    # Create a simple router for the object detection endpoint that returns a helpful message
    @router.get("/object-detection/health")
    async def object_detection_health():
        return {
            "status": "unavailable",
            "message": "Object detection is not available. Required dependencies are missing.",
            "required_dependencies": ["ultralytics", "opencv-python"],
            "install_command": "pip install ultralytics opencv-python"
        }
