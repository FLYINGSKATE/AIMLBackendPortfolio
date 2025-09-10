from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, confloat
from typing import Dict, Any, List, Optional
import logging

# Import our model
from .models.house_price_model import HousePriceModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the model
model = HousePriceModel()

# Request/Response Models
class HouseFeatures(BaseModel):
    """Input features for house price prediction."""
    MedInc: confloat(gt=0) = Field(
        ..., 
        example=8.3, 
        description="Median income in block group (in $10,000s)"
    )
    HouseAge: confloat(ge=0) = Field(
        ..., 
        example=21.0, 
        description="Median house age in block group (years)"
    )
    AveRooms: confloat(gt=0) = Field(
        ..., 
        example=6.2, 
        description="Average number of rooms per household"
    )
    AveBedrms: confloat(gt=0) = Field(
        ..., 
        example=1.1, 
        description="Average number of bedrooms per household"
    )
    Population: confloat(gt=0) = Field(
        ..., 
        example=240.0, 
        description="Block group population"
    )
    AveOccup: confloat(gt=0) = Field(
        ..., 
        example=3.0, 
        description="Average number of household members"
    )
    Latitude: confloat(ge=-90, le=90) = Field(
        ..., 
        example=37.8, 
        description="Block group latitude"
    )
    Longitude: confloat(ge=-180, le=180) = Field(
        ..., 
        example=-122.2, 
        description="Block group longitude"
    )

class PredictionResponse(BaseModel):
    """Response model for house price prediction."""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    features: Dict[str, float] = Field(..., description="Input features used for prediction")
    model: str = Field(..., description="Name of the model used for prediction")

class TrainingResponse(BaseModel):
    """Response model for model training/retraining."""
    status: str = Field(..., description="Training status")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    r2_score: Optional[float] = Field(None, description="R² Score")
    model_path: Optional[str] = Field(None, description="Path to the saved model")
    feature_importances: Optional[Dict[str, float]] = Field(
        None, 
        description="Feature importances from the model"
    )

# API Endpoints
@router.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_200_OK,
    summary="Predict house price",
    response_description="House price prediction with input features"
)
async def predict_house_price(features: HouseFeatures):
    """
    Predict house price based on input features.
    
    - **MedInc**: Median income in block group (in $10,000s)
    - **HouseAge**: Median house age in block group (years)
    - **AveRooms**: Average number of rooms per household
    - **AveBedrms**: Average number of bedrooms per household
    - **Population**: Block group population
    - **AveOccup**: Average number of household members
    - **Latitude**: Block group latitude (between -90 and 90)
    - **Longitude**: Block group longitude (between -180 and 180)
    """
    try:
        # Convert features to dict and make prediction
        input_data = features.dict()
        prediction = model.predict(input_data)
        
        logger.info(f"Prediction successful for features: {input_data}")
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error making prediction: {str(e)}"
        )

@router.post(
    "/retrain", 
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrain the house price prediction model",
    response_description="Model retraining results"
)
async def retrain_model():
    """
    Retrain the house price prediction model with the latest data.
    This will update the model with any new data that has been added.
    """
    try:
        # Retrain the model
        training_metrics = model.retrain()
        
        logger.info("Model retrained successfully")
        return {
            "status": "success",
            "mse": training_metrics['mse'],
            "r2_score": training_metrics['r2_score'],
            "model_path": training_metrics['model_path'],
            "feature_importances": training_metrics['feature_importances']
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retraining model: {str(e)}"
        )
