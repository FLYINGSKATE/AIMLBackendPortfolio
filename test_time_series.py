import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Base URL of the API
BASE_URL = "http://localhost:8001/api/time-series/forecast"

# Generate sample time series data
def generate_sample_data(days=100):
    """Generate sample time series data with trend and seasonality."""
    dates = pd.date_range(end=datetime.now(), periods=days).strftime('%Y-%m-%d').tolist()
    
    # Create a time series with trend and seasonality
    trend = np.linspace(0, 10, days)
    seasonal = 5 * np.sin(np.linspace(0, 10*np.pi, days))
    noise = np.random.normal(0, 1, days)
    values = (trend + seasonal + noise).tolist()
    
    return {
        "timestamps": dates,
        "values": values,
        "freq": "D"
    }

# Test ARIMA model
def test_arima():
    print("\n=== Testing ARIMA Model ===")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Train a simple model (using LSTM instead of ARIMA to avoid statsmodels dependency)
    train_data = {
        "model_type": "lstm",
        "model_id": "test_forecast",
        "lstm_units": 32,
        "epochs": 20,
        "batch_size": 16,
        "look_back": 7,
        "data": data
    }
    
    print("\nTraining ARIMA model...")
    response = requests.post(f"{BASE_URL}/train", json=train_data)
    print(f"Training Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Generate forecast
    forecast_request = {
        "model_id": "test_arima",
        "steps": 14,
        "confidence_level": 0.95
    }
    
    print("\nGenerating forecast with ARIMA...")
    response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
    print(f"Forecast Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# Test LSTM model
def test_lstm():
    print("\n=== Testing LSTM Model ===")
    
    # Generate sample data
    data = generate_sample_data(200)  # LSTM typically needs more data
    
    # Train LSTM model
    train_data = {
        "model_type": "lstm",
        "model_id": "test_lstm",
        "lstm_units": 50,
        "epochs": 50,
        "batch_size": 32,
        "look_back": 7,
        "data": data
    }
    
    print("\nTraining LSTM model (this may take a minute)...")
    response = requests.post(f"{BASE_URL}/train", json=train_data)
    print(f"Training Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Generate forecast
    forecast_request = {
        "model_id": "test_lstm",
        "steps": 14,
        "confidence_level": 0.95
    }
    
    print("\nGenerating forecast with LSTM...")
    response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
    print(f"Forecast Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

# List all models
def list_models():
    print("\n=== Listing All Models ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Test LSTM model (skipping ARIMA due to statsmodels dependency)
    print("Skipping ARIMA test due to missing statsmodels. Testing LSTM model only.")
    test_lstm()
    
    # List all models
    list_models()
