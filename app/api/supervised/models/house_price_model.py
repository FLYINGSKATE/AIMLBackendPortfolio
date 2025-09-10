import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple, Optional

from .base_model import BaseModel

class HousePriceModel(BaseModel):
    """House price prediction model using Random Forest."""
    
    def __init__(self):
        super().__init__(
            model_name="house_price_rf",
            model_dir=os.path.join("app", "api", "supervised", "models")
        )
        self.feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        self.target_name = 'PRICE'
        self._load_or_train()
    
    def _load_or_train(self) -> None:
        """Load the model if it exists, otherwise train a new one."""
        if not self.load():
            print("Training new house price model...")
            self._load_sample_data()
            self.train()
    
    def _load_sample_data(self) -> None:
        """Load sample data for training."""
        # This would typically come from a database or data file
        self.sample_data = {
            'MedInc': [8.3, 5.0, 7.3, 3.8, 5.4, 6.2, 4.1, 7.8, 4.5, 6.7],
            'HouseAge': [21.0, 34.0, 41.0, 52.0, 28.0, 35.0, 45.0, 30.0, 38.0, 42.0],
            'AveRooms': [6.2, 4.5, 5.8, 3.7, 6.1, 5.5, 4.8, 6.8, 5.2, 6.0],
            'AveBedrms': [1.1, 0.9, 1.2, 1.0, 1.1, 1.0, 0.9, 1.3, 1.0, 1.2],
            'Population': [240.0, 1000.0, 800.0, 1200.0, 500.0, 750.0, 950.0, 600.0, 850.0, 700.0],
            'AveOccup': [3.0, 2.5, 2.8, 2.0, 3.2, 2.7, 2.3, 3.1, 2.6, 2.9],
            'Latitude': [37.8, 34.2, 36.5, 38.1, 37.7, 35.9, 37.2, 36.8, 35.5, 37.0],
            'Longitude': [-122.2, -118.5, -119.8, -120.3, -121.9, -119.2, -121.5, -120.8, -119.9, -121.0],
            'PRICE': [452600, 358500, 352100, 341300, 342200, 365000, 378900, 412300, 325600, 389700]
        }
    
    def train(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the house price prediction model.
        
        Args:
            X: Features DataFrame (optional, uses sample data if None)
            y: Target Series (optional, uses sample data if None)
            
        Returns:
            Dictionary containing training metrics
        """
        if X is None or y is None:
            # Use sample data if none provided
            df = pd.DataFrame(self.sample_data)
            X = df[self.feature_names]
            y = df[self.target_name]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the trained model
        model_path = self.save()
        
        return {
            'mse': mse,
            'r2_score': r2,
            'model_path': model_path,
            'feature_importances': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict house price from input features.
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            Dictionary containing prediction and input features
        """
        if not self.is_trained():
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Convert input to DataFrame with correct feature order
        input_df = pd.DataFrame([{
            feature: input_data[feature] for feature in self.feature_names
        }])
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        return {
            'predicted_price': round(float(prediction), 2),
            'features': input_data,
            'model': self.model_name
        }
    
    def retrain(self, new_data: Optional[Dict[str, list]] = None) -> Dict[str, Any]:
        """
        Retrain the model with new data.
        
        Args:
            new_data: Optional new data to include in training
            
        Returns:
            Dictionary containing training metrics
        """
        if new_data is not None:
            # Update sample data with new data
            for key, values in new_data.items():
                if key in self.sample_data:
                    self.sample_data[key].extend(values)
        
        # Retrain with updated data
        df = pd.DataFrame(self.sample_data)
        X = df[self.feature_names]
        y = df[self.target_name]
        
        return self.train(X, y)
