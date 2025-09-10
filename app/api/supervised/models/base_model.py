from abc import ABC, abstractmethod
import joblib
import os
from typing import Any, Dict, Optional

class BaseModel(ABC):
    """Base class for all ML models in the application."""
    
    def __init__(self, model_name: str, model_dir: str = "models"):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model file (without extension)
            model_dir: Directory to save/load the model
        """
        self.model = None
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"{model_name}.joblib")
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Train the model with the given data."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the trained model."""
        pass
    
    def save(self) -> str:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, self.model_path)
        return self.model_path
    
    def load(self) -> bool:
        """Load the model from disk."""
        if not os.path.exists(self.model_path):
            return False
        self.model = joblib.load(self.model_path)
        return True
    
    def is_trained(self) -> bool:
        """Check if the model has been trained and loaded."""
        return self.model is not None
