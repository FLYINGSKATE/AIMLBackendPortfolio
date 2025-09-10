import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # App settings
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # API settings
    API_PREFIX = "/api"
    PROJECT_NAME = "AI Portfolio API"
    VERSION = "1.0.0"
    
    # CORS settings
    BACKEND_CORS_ORIGINS = os.getenv(
        "BACKEND_CORS_ORIGINS",
        "http://localhost:8001,http://localhost:3000,http://localhost:3001"
    ).split(",")
    
    # Model paths
    MODEL_DIR = "models"
    
    # Time series settings
    TIME_SERIES_MODEL_DIR = os.path.join(MODEL_DIR, "time_series")
    
    # File upload settings
    UPLOAD_FOLDER = "static/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'json'}

# Create instance of config
config = Config()
