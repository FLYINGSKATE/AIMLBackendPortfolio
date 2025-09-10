import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status, Body, UploadFile, File
from pydantic import BaseModel, Field, conint, confloat
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import joblib
import tempfile
import shutil

# Time series specific imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Model configuration
MODEL_DIR = "models/time_series"
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize models and scalers
arima_models = {}
lstm_models = {}
scalers = {}

class TimeSeriesData(BaseModel):
    timestamps: List[Union[int, str]]
    values: List[float]
    freq: Optional[str] = "D"  # Daily by default, can be 'H' for hourly, 'M' for monthly, etc.

class ForecastRequest(BaseModel):
    model_id: str
    steps: conint(gt=0, le=365) = 7
    confidence_level: confloat(ge=0.5, lt=1.0) = 0.95

class ModelConfig(BaseModel):
    model_type: str = Field(..., pattern="^(arima|lstm)$")
    model_id: str
    # ARIMA parameters
    p: conint(ge=0) = 1
    d: conint(ge=0) = 1
    q: conint(ge=0) = 1
    # LSTM parameters
    lstm_units: conint(ge=1) = 50
    epochs: conint(ge=1, le=1000) = 100
    batch_size: conint(ge=1, le=1024) = 32
    look_back: conint(ge=1, le=100) = 7

class ForecastResult(BaseModel):
    model_id: str
    model_type: str
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    timestamps: List[str]
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    plot_url: Optional[str] = None  # URL to the forecast plot
    visualization: Optional[Dict[str, Any]] = None  # Visualization data

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    created_at: str
    last_used: str
    metrics: Dict[str, Any]

class SampleDataResponse(BaseModel):
    timestamps: List[str]
    values: List[float]
    message: str = "Sample time series data generated successfully"

@router.get("/sample-data", response_model=SampleDataResponse, summary="Get sample time series data")
async def get_sample_data(
    n_points: int = 30,
    trend: float = 0.5,
    seasonality: float = 0.5,
    noise: float = 0.2
):
    """
    Generate sample time series data for testing and demonstration purposes.
    
    Parameters:
    - n_points: Number of data points to generate (default: 30)
    - trend: Strength of the trend component (0-1)
    - seasonality: Strength of the seasonality component (0-1)
    - noise: Amount of random noise to add (0-1)
    """
    from datetime import datetime, timedelta
    import numpy as np
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_points-1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    # Generate trend component
    x = np.linspace(0, 10, n_points)
    trend_component = trend * x
    
    # Generate seasonality component (weekly seasonality)
    seasonality_component = seasonality * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Generate noise
    noise_component = noise * np.random.normal(0, 1, n_points)
    
    # Combine components
    values = 10 + trend_component + seasonality_component + noise_component
    
    return {
        "timestamps": [d.isoformat() for d in dates],
        "values": values.tolist()
    }

# Import all routers
try:
    from app.api.supervised import router as supervised_router
    from app.api.unsupervised import router as unsupervised_router
    from app.api.nlp import router as nlp_router
    from app.api.cv import router as cv_router
    from app.api.rl import router as rl_router
    from app.api.time_series import router as time_series_router
except ImportError as e:
    logger.warning(f"Failed to import some routers: {e}")
    # Create empty routers for missing modules
    supervised_router = APIRouter()
    unsupervised_router = APIRouter()
    nlp_router = APIRouter()
    cv_router = APIRouter()
    rl_router = APIRouter()
    time_series_router = APIRouter()

# Import visualization utilities
from .visualization import create_forecast_plot

# Helper functions
def create_lstm_model(look_back: int, lstm_units: int = 50):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=lstm_units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# API Endpoints
@router.post(
    "/train",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Train a new time series model",
    description="""
    Train a new time series forecasting model (ARIMA or LSTM).
    
    - **ARIMA**: Suitable for univariate time series with trends and seasonality
    - **LSTM**: Deep learning model for complex patterns and long sequences
    
    Returns training metrics and model information.
    """
)
async def train_model(
    data: TimeSeriesData,
    config: ModelConfig
):
    model_id = config.model_id
    model_type = config.model_type.lower()
    
    # Generate timestamps if not provided or invalid
    if not data.timestamps or len(data.timestamps) != len(data.values):
        # Generate default timestamps (daily)
        start_date = datetime.now() - timedelta(days=len(data.values)-1)
        data.timestamps = [start_date + timedelta(days=i) for i in range(len(data.values))]
    else:
        # Convert timestamps to datetime objects if they're strings or numbers
        try:
            if isinstance(data.timestamps[0], (int, float)):
                # Handle numeric timestamps (assume they're Unix timestamps)
                data.timestamps = [datetime.fromtimestamp(ts) for ts in data.timestamps]
            elif isinstance(data.timestamps[0], str):
                # Handle string timestamps
                data.timestamps = [pd.to_datetime(ts) for ts in data.timestamps]
        except Exception as e:
            # If timestamp parsing fails, use default timestamps
            logger.warning(f"Error parsing timestamps: {str(e)}. Using default timestamps.")
            start_date = datetime.now() - timedelta(days=len(data.values)-1)
            data.timestamps = [start_date + timedelta(days=i) for i in range(len(data.values))]
    
    try:
        # Validate data length
        min_data_points = 10  # Minimum data points required for meaningful training
        if len(data.values) < min_data_points:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient data points. Need at least {min_data_points} points, got {len(data.values)}"
            )
            
        # Convert to pandas Series with datetime index
        series = pd.Series(
            data.values,
            index=pd.to_datetime(data.timestamps)
        )
        
        if model_type == "arima":
            # Train ARIMA model with error handling for small datasets
            try:
                model = ARIMA(
                    series,
                    order=(config.p, config.d, config.q),
                    freq=data.freq
                )
                model_fit = model.fit()
                
                # If the model has NaN parameters, it's likely due to insufficient data
                if model_fit.params.isna().any():
                    raise ValueError("Model parameters contain NaN values. This often occurs with insufficient or non-stationary data.")
                    
            except Exception as e:
                # If ARIMA fails, try with simpler model
                logger.warning(f"ARIMA({config.p},{config.d},{config.q}) failed: {str(e)}. Trying with simpler model.")
                try:
                    # Try with simpler model (0,1,1) which is equivalent to simple exponential smoothing
                    model = ARIMA(series, order=(0, 1, 1), freq=data.freq)
                    model_fit = model.fit()
                    if model_fit.params.isna().any():
                        raise ValueError("Even simpler model failed")
                except Exception as inner_e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to train ARIMA model: {str(inner_e)}. Try with more data or different parameters."
                    )
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{config.model_id}_arima.pkl")
            joblib.dump(model_fit, model_path)
            
            # Store model reference
            arima_models[config.model_id] = model_fit
            
            return {
                "status": "success",
                "model_id": config.model_id,
                "model_type": "arima",
                "aic": model_fit.aic,
                "bic": model_fit.bic,
                "params": model_fit.params.to_dict(),
                "model_summary": str(model_fit.summary())
            }
            
        elif model_type == "lstm":
            try:
                # Validate data length for LSTM
                min_lstm_points = config.look_back * 3  # Need at least 3x look_back for meaningful training
                if len(data.values) < min_lstm_points:
                    raise ValueError(f"Not enough data points for LSTM. Need at least {min_lstm_points} points with look_back={config.look_back}")
                
                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
                
                # Create training data
                look_back = min(config.look_back, len(scaled_data) // 2)  # Adjust look_back if needed
                X, y = create_dataset(scaled_data, look_back)
                
                if len(X) < 1:
                    raise ValueError("Insufficient data for training after look_back adjustment")
                
                # Reshape input to be [samples, time steps, features]
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Adjust batch size if it's too large for the dataset
                batch_size = min(config.batch_size, len(X))
                
                # Create and train the LSTM network
                model = create_lstm_model(look_back, min(config.lstm_units, 32))  # Use smaller network for small datasets
                
                # Set up callbacks
                callbacks = []
                if len(X) > 10:  # Only use early stopping if we have enough validation data
                    early_stopping = EarlyStopping(
                        monitor='loss',
                        patience=5,
                        restore_best_weights=True,
                        min_delta=0.001
                    )
                    callbacks.append(early_stopping)
                
                model_checkpoint = ModelCheckpoint(
                    os.path.join(MODEL_DIR, f"{config.model_id}_lstm_best.h5"),
                    save_best_only=True,
                    monitor='loss',
                    mode='min'
                )
                callbacks.append(model_checkpoint)
                
                # Train the model with reduced epochs for small datasets
                epochs = min(config.epochs, 50) if len(X) < 50 else config.epochs
                
                history = model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Save the final model
                model_path = os.path.join(MODEL_DIR, f"{config.model_id}_lstm.h5")
                model.save(model_path)
                
                # Save the scaler
                scaler_path = os.path.join(MODEL_DIR, f"{config.model_id}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
                
                # Store model and scaler references
                lstm_models[config.model_id] = {
                    'model': model,
                    'scaler': scaler,
                    'look_back': look_back
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to train LSTM model: {str(e)}. Try with more data or a simpler model."
                )    
            
            return {
                "status": "success",
                "model_id": config.model_id,
                "model_type": "lstm",
                "loss": history.history['loss'][-1],
                "epochs_trained": len(history.history['loss']),
                "model_summary": []  # Can't serialize model summary directly
            }
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}"
        )

@router.post(
    "/forecast",
    response_model=ForecastResult,
    status_code=status.HTTP_200_OK,
    summary="Generate forecasts using a trained model"
)
async def forecast(
    request: ForecastRequest
):
    """
    Generate forecasts using a pre-trained time series model.
    
    - **model_id**: ID of the pre-trained model
    - **steps**: Number of time steps to forecast
    - **confidence_level**: Confidence level for prediction intervals (0.5-0.99)
    """
    try:
        # Check if model exists
        arima_path = os.path.join(MODEL_DIR, f"{request.model_id}_arima.pkl")
        lstm_path = os.path.join(MODEL_DIR, f"{request.model_id}_lstm.h5")
        
        if os.path.exists(arima_path):
            # Load ARIMA model
            if request.model_id not in arima_models:
                arima_models[request.model_id] = joblib.load(arima_path)
            
            model = arima_models[request.model_id]
            forecast_result = model.get_forecast(steps=request.steps)
            
            # Get prediction intervals
            conf_int = forecast_result.conf_int(alpha=1-request.confidence_level)
            
            # Generate timestamps
            last_date = pd.to_datetime(model.model.data.dates[-1])
            freq = model.model.freq or 'D'
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(1, unit=freq[0]),
                periods=request.steps,
                freq=freq
            )
            timestamps = forecast_dates.strftime('%Y-%m-%d').tolist()
            
            # Create forecast result
            forecast_values = forecast_result.predicted_mean.tolist()
            lower_bounds = conf_int.iloc[:, 0].tolist()
            upper_bounds = conf_int.iloc[:, 1].tolist()
            
            # Generate plot
            historical_dates = pd.to_datetime(model.model.data.dates)
            historical_values = model.model.data.endog
            
            plot_url = create_forecast_plot(
                historical_dates=historical_dates,
                historical_values=historical_values,
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                title=f"Time Series Forecast - {request.model_id}",
                y_label="Value"
            )
            
            return ForecastResult(
                model_id=request.model_id,
                model_type="arima",
                forecast=forecast_values,
                lower_bound=lower_bounds,
                upper_bound=upper_bounds,
                timestamps=timestamps,
                plot_url=plot_url
            )
            
        elif os.path.exists(lstm_path):
            # Load LSTM model and scaler
            if request.model_id not in lstm_models:
                lstm_models[request.model_id] = load_model(lstm_path)
                scaler_path = os.path.join(MODEL_DIR, f"{request.model_id}_scaler.pkl")
                scalers[request.model_id] = joblib.load(scaler_path)
            
            model = lstm_models[request.model_id]
            scaler = scalers[request.model_id]
            
            # Get the last look_back values for prediction
            # Note: In a real app, you'd want to store the last look_back values
            # This is a simplified version
            last_sequence = np.array([0] * model.input_shape[1] + [1] * model.input_shape[1])
            last_sequence = last_sequence.reshape(-1, 1)
            
            # Generate forecasts
            forecasts = []
            current_batch = last_sequence[-model.input_shape[1]:]
            
            for _ in range(request.steps):
                current_pred = model.predict(current_batch.reshape(1, model.input_shape[1], 1))
                forecasts.append(current_pred[0][0])
                current_batch = np.append(current_batch[1:], current_pred[0])
            
            # Inverse transform the forecasts
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = scaler.inverse_transform(forecasts).flatten().tolist()
            
            # Generate timestamps
            forecast_dates = pd.date_range(
                start=datetime.now(),
                periods=request.steps,
                freq='D'  # Default to daily frequency for LSTM
            )
            timestamps = forecast_dates.strftime('%Y-%m-%d').tolist()
            
            # For LSTM, we don't have built-in confidence intervals
            # So we'll just return the same values for upper/lower bounds
            
            # Generate plot (with dummy historical data for demonstration)
            historical_dates = forecast_dates - pd.Timedelta(days=len(forecasts)*2)
            historical_values = [f * 0.8 + np.random.normal(0, 2) for f in forecasts * 2][-len(historical_dates):]
            
            plot_url = create_forecast_plot(
                historical_dates=historical_dates,
                historical_values=historical_values,
                forecast_dates=forecast_dates,
                forecast_values=forecasts,
                title=f"Time Series Forecast - {request.model_id}",
                y_label="Value"
            )
            
            return ForecastResult(
                model_id=request.model_id,
                model_type="lstm",
                forecast=forecasts,
                lower_bound=forecasts,
                upper_bound=forecasts,
                timestamps=timestamps,
                plot_url=plot_url
            )
            
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {request.model_id} not found"
            )
            
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecast: {str(e)}"
        )

@router.get(
    "/models",
    response_model=List[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="List all trained time series models"
)
async def list_models():
    """
    Get information about all trained time series models.
    """
    try:
        models = []
        
        # Get ARIMA models
        for f in os.listdir(MODEL_DIR):
            if f.endswith('_arima.pkl'):
                model_id = f.replace('_arima.pkl', '')
                model_path = os.path.join(MODEL_DIR, f)
                model = joblib.load(model_path)
                
                models.append({
                    "model_id": model_id,
                    "model_type": "arima",
                    "created_at": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                    "last_used": datetime.now().isoformat(),
                    "metrics": {
                        "aic": model.aic,
                        "bic": model.bic,
                        "params": model.params.to_dict()
                    }
                })
        
        # Get LSTM models
        for f in os.listdir(MODEL_DIR):
            if f.endswith('_lstm.h5'):
                model_id = f.replace('_lstm.h5', '')
                model_path = os.path.join(MODEL_DIR, f)
                
                models.append({
                    "model_id": model_id,
                    "model_type": "lstm",
                    "created_at": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                    "last_used": datetime.now().isoformat(),
                    "metrics": {
                        "model_path": model_path,
                        "input_shape": "Not available without loading the model"
                    }
                })
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )
