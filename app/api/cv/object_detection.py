import os
import cv2
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Model configuration
MODEL_PATH = "models/yolov5"
DETECTION_CONFIDENCE = 0.5  # Default confidence threshold

# Initialize model
model = None

class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

class DetectionResult(BaseModel):
    filename: str
    detections: List[DetectionBox]
    image_size: Dict[str, int]
    model: str
    inference_time: float

class DetectionConfig(BaseModel):
    confidence_threshold: float = Field(0.5, ge=0.1, le=0.9)
    max_detections: int = Field(100, ge=1, le=300)

def load_yolo_model():
    """Load YOLOv5 model with error handling."""
    global model
    try:
        if model is None:
            logger.info("Loading YOLOv5 model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.conf = DETECTION_CONFIDENCE
            logger.info("YOLOv5 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading YOLO model: {str(e)}"
        )

@router.post(
    "/detect",
    response_model=DetectionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect objects in an image",
    response_description="Object detection results with bounding boxes"
)
async def detect_objects(
    file: UploadFile = File(..., description="Image file for object detection"),
    config: DetectionConfig = DetectionConfig()
):
    """
    Detect objects in an uploaded image using YOLOv5.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence_threshold**: Minimum confidence score (0.1 to 0.9)
    - **max_detections**: Maximum number of detections to return
    
    Returns:
        JSON with detected objects and their bounding boxes
    """
    global model
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Load model if not already loaded
        model = load_yolo_model()
        model.conf = config.confidence_threshold
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Read and process image
        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read image"
            )
        
        # Run inference
        start_time = datetime.now()
        results = model(img)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Process results
        detections = []
        pred = results.pandas().xyxy[0]  # Predictions in pandas DataFrame
        
        for _, det in pred.iterrows():
            if len(detections) >= config.max_detections:
                break
                
            detections.append(DetectionBox(
                x1=int(det['xmin']),
                y1=int(det['ymin']),
                x2=int(det['xmax']),
                y2=int(det['ymax']),
                confidence=float(det['confidence']),
                class_id=int(det['class']),
                class_name=det['name']
            ))
        
        # Clean up
        os.unlink(temp_path)
        
        return DetectionResult(
            filename=file.filename,
            detections=detections,
            image_size={"width": img.shape[1], "height": img.shape[0]},
            model="YOLOv5s",
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"Error during object detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@router.get(
    "/classes",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="Get list of detectable classes",
    response_description="List of class names that the model can detect"
)
async def get_detectable_classes():
    """
    Get the list of classes that the object detection model can identify.
    """
    try:
        model = load_yolo_model()
        return model.names
    except Exception as e:
        logger.error(f"Error getting class names: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting class names: {str(e)}"
        )

# Load model when module is imported
load_yolo_model()
