from fastapi import APIRouter, HTTPException, status, Body, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize model and vectorizer
model = None
vectorizer = None
label_encoder = {}

class TrainingData(BaseModel):
    texts: List[str] = Field(..., example=["This is a positive text", "This is negative"])
    labels: List[str] = Field(..., example=["positive", "negative"])

class PredictionRequest(BaseModel):
    text: str = Field(..., example="This is a sample text to classify")

class TrainingConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.1, le=0.3)
    random_state: int = 42
    max_features: int = 5000

class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    class_distribution: Dict[str, int]
    model_summary: Dict[str, Any]

def load_model():
    global model, vectorizer, label_encoder
    try:
        model = joblib.load("models/text_classifier.joblib")
        vectorizer = joblib.load("models/text_vectorizer.joblib")
        label_encoder = joblib.load("models/label_encoder.joblib")
        logger.info("Loaded text classification model and vectorizer")
    except Exception as e:
        logger.warning(f"Error loading text classification model: {e}")
        # Initialize with empty model
        model = None
        vectorizer = TfidfVectorizer(max_features=5000)
        label_encoder = {}

@router.post(
    "/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Train or update the text classification model",
    response_description="Training results with model metrics"
)
async def train_model(
    data: TrainingData,
    config: TrainingConfig = TrainingConfig()
):
    """
    Train or update the text classification model with new data.
    
    - **texts**: List of text samples
    - **labels**: List of corresponding class labels
    - **test_size**: Fraction of data to use for testing (0.1 to 0.3)
    - **random_state**: Random seed for reproducibility
    - **max_features**: Maximum number of features for TF-IDF
    """
    global model, vectorizer, label_encoder
    
    try:
        # Encode labels
        unique_labels = sorted(set(data.labels))
        label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_encoder[label] for label in data.labels])
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=config.max_features)
        X = vectorizer.fit_transform(data.texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/text_classifier.joblib")
        joblib.dump(vectorizer, "models/text_vectorizer.joblib")
        joblib.dump(label_encoder, "models/label_encoder.joblib")
        
        # Prepare response
        return {
            "message": "Model trained successfully",
            "accuracy": float(accuracy),
            "class_distribution": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "model_summary": {
                "classes": list(label_encoder.keys()),
                "features": X.shape[1],
                "samples": X.shape[0]
            }
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}"
        )

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify text into predefined categories",
    response_description="Prediction results with confidence scores"
)
async def predict_text_class(request: PredictionRequest):
    """
    Classify input text into one of the trained categories.
    
    - **text**: The text to classify
    """
    global model, vectorizer, label_encoder
    
    if model is None or vectorizer is None:
        load_model()
    
    try:
        # Vectorize input
        X = vectorizer.transform([request.text])
        
        # Get predictions
        probs = model.predict_proba(X)[0]
        pred_class_idx = model.predict(X)[0]
        
        # Map indices back to labels
        idx_to_label = {v: k for k, v in label_encoder.items()}
        pred_class = idx_to_label.get(pred_class_idx, "unknown")
        
        # Get probabilities for each class
        class_probs = {}
        for idx, prob in enumerate(probs):
            class_name = idx_to_label.get(idx, f"class_{idx}")
            class_probs[class_name] = float(prob)
        
        return {
            "text": request.text,
            "predicted_class": pred_class,
            "confidence": float(max(probs)),
            "probabilities": class_probs
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

# Load model when module is imported
load_model()
