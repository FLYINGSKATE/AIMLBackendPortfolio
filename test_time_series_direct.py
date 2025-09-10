import numpy as np
import pandas as pd
import asyncio
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from app.api.time_series.forecasting import TimeSeriesData, ModelConfig, ForecastRequest, ForecastResult, train_model, forecast

# Generate sample time series data
def generate_sample_data(days=100):
    """Generate sample time series data with trend and seasonality."""
    dates = pd.date_range(end=datetime.now(), periods=days).strftime('%Y-%m-%d').tolist()
    
    # Create a time series with trend and seasonality
    trend = np.linspace(0, 10, days)
    seasonal = 5 * np.sin(np.linspace(0, 10*np.pi, days))
    noise = np.random.normal(0, 1, days)
    values = (trend + seasonal + noise).tolist()
    
    return TimeSeriesData(
        timestamps=dates,
        values=values,
        freq="D"
    )

async def test_lstm_training():
    print("\n=== Testing LSTM Training ===")
    
    # Generate sample data
    data = generate_sample_data(200)  # LSTM typically needs more data
    
    # Create config
    config = ModelConfig(
        model_type="lstm",
        model_id="test_lstm",
        lstm_units=32,
        epochs=5,  # Reduced for testing
        batch_size=16,
        look_back=7
    )
    
    print("\nTraining LSTM model...")
    
    try:
        result = await train_model(data, config)
        print("Training successful!")
        if hasattr(result, 'loss'):
            print(f"Final loss: {result.loss}")
        return True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_forecasting():
    print("\n=== Testing Forecasting ===")
    
    # First, train a model
    if not await test_lstm_training():
        print("Skipping forecast test due to training failure")
        return
    
    # Create forecast request
    request = ForecastRequest(
        model_id="test_lstm",
        steps=14,
        confidence_level=0.95
    )
    
    try:
        print("\nGenerating forecast...")
        result = await forecast(request)
        print("Forecast successful!")
        print(f"Forecast values: {result.forecast[:5]}...")  # Print first 5 values
        return True
    except Exception as e:
        print(f"Error during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    # Make sure models directory exists
    import os
    os.makedirs("models/time_series", exist_ok=True)
    
    # Run tests
    print("Starting time series forecasting tests...")
    await test_lstm_training()
    await test_forecasting()

if __name__ == "__main__":
    asyncio.run(main())
