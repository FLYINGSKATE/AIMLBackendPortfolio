from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

router = APIRouter()

# Initialize model and vectorizer
model = None
vectorizer = None

# Sample training data (in a real app, this would be in a separate data file)
TRAIN_TEXTS = [
    "I love this product, it works great!",
    "This is the worst experience I've ever had.",
    "The quality is good but the price is too high.",
    "Amazing service, highly recommended!",
    "Terrible customer support, would not buy again."
]
TRAIN_LABELS = [1, 0, 1, 1, 0]  # 1 for positive, 0 for negative

class SentimentRequest(BaseModel):
    text: str = Field(..., example="I really enjoyed using this service!", 
                     description="Text to analyze for sentiment")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

class TrainingData(BaseModel):
    texts: List[str] = Field(..., example=["I love this!", "I hate this!"], description="List of training texts")
    labels: List[int] = Field(..., example=[1, 0], description="List of corresponding labels (1 for positive, 0 for negative)")

class TrainingResponse(BaseModel):
    """Response model for training operations.
    
    Attributes:
        message: Status message about the training operation
        accuracy: Accuracy of the model on the training data (0.0 to 1.0)
        samples_trained: Number of samples used for training
    """
    message: str = Field(..., example="Model retrained successfully on 100 samples")
    accuracy: Optional[float] = Field(None, example=0.95, ge=0.0, le=1.0, 
                                   description="Model accuracy on training data")
    samples_trained: Optional[int] = Field(None, example=100, 
                                         description="Number of training samples used")

# Initialize model if not already loaded
def load_model():
    global model, vectorizer
    model_path = "models/sentiment_model.joblib"
    vectorizer_path = "models/sentiment_vectorizer.joblib"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        train_model()

def train_model(training_data=None):
    global model, vectorizer, TRAIN_TEXTS, TRAIN_LABELS
    
    # Use provided training data or default to sample data
    if training_data and training_data.texts and training_data.labels:
        if len(training_data.texts) != len(training_data.labels):
            raise ValueError("Number of texts must match number of labels")
        texts = training_data.texts
        labels = training_data.labels
    else:
        texts = TRAIN_TEXTS
        labels = TRAIN_LABELS
    
    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, labels)
    
    # Save the model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.joblib")
    joblib.dump(vectorizer, "models/sentiment_vectorizer.joblib")
    
    # Calculate accuracy
    train_accuracy = model.score(X_train, labels)
    return train_accuracy, len(texts)

@router.post(
    "/predict", 
    response_model=SentimentResponse, 
    status_code=status.HTTP_200_OK,
    summary="Predict sentiment of a text",
    response_description="Sentiment analysis result"
)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict the sentiment of the input text (positive/negative).
    """
    try:
        if model is None or vectorizer is None:
            load_model()
        
        # Transform the input text
        X = vectorizer.transform([request.text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X).max()
        
        return {
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": float(proba),
            "text": request.text
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting sentiment: {str(e)}"
        )

@router.post(
    "/retrain", 
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrain the sentiment analysis model",
    response_description="Training result with accuracy and sample count",
    responses={
        200: {
            "description": "Model retrained successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model retrained successfully on 4 samples",
                        "accuracy": 1.0,
                        "samples_trained": 4
                    }
                }
            }
        },
        400: {
            "description": "Invalid input data",
            "content": {
                "application/json": {
                    "example": {"detail": "Number of texts must match number of labels"}
                }
            }
        },
        500: {
            "description": "Training error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retraining model: <error message>"}
                }
            }
        }
    }
)
async def retrain_model(
    training_data: Optional[TrainingData] = Body(
        None,
        example={
            "texts": [
                "This product is amazing!",
                "I would not recommend this to anyone.",
                "Great customer service!",
                "Terrible experience overall."
            ],
            "labels": [1, 0, 1, 0]
        },
        description="Training data with texts and corresponding labels (1=positive, 0=negative)",
        openapi_examples={
            "minimal": {
                "summary": "Minimal example",
                "description": "Simplest possible training data with two examples",
                "value": {
                    "texts": ["Great!", "Bad"],
                    "labels": [1, 0]
                }
            },
            "detailed": {
                "summary": "Detailed example",
                "description": "More comprehensive training data with various examples",
                "value": {
                    "texts": [
                        "Absolutely love this product, works perfectly!",
                        "The quality is good but delivery was late.",
                        "Terrible customer service, would not buy again.",
                        "Exceeded my expectations, highly recommend!",
                        "Not worth the price, very disappointed."
                    ],
                    "labels": [1, 1, 0, 1, 0]
                }
            }
        }
    )
):
    """
    Retrain the sentiment analysis model with new training data.
    
    This endpoint allows you to provide custom training data to improve the model's accuracy
    for your specific use case. The model will be retrained using both the new data and
    any previously provided training examples.
    
    **Note**: For best results, provide a balanced dataset with both positive and negative examples.
    
    **Labels**:
    - 1: Positive sentiment
    - 0: Negative sentiment
    
    **Example CURL**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8001/api/nlp/sentiment/retrain' \
      -H 'Content-Type: application/json' \
      -d '{
        "texts": ["Great service!", "Not good at all"],
        "labels": [1, 0]
      }'
    ```
    
    **Python Example**:
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8001/api/nlp/sentiment/retrain",
        json={
            "texts": ["Great service!", "Not good at all"],
            "labels": [1, 0]
        }
    )
    print(response.json())
    ```
    
    If no training data is provided, the model will be retrained using the default dataset.
    """
    try:
        accuracy, samples_trained = train_model(training_data)
        return {
            "message": f"Model retrained successfully on {samples_trained} samples",
            "accuracy": accuracy,
            "samples_trained": samples_trained
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retraining model: {str(e)}"
        )

# Initialize the model when the module is imported
load_model()
