from fastapi import FastAPI, APIRouter, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
import os
import logging
from pathlib import Path
from typing import Optional, List, Union

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import config as app_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="🤖 AI Portfolio API",
    description="""
    A comprehensive collection of machine learning models served as RESTful APIs.
    
    ## Features
    - 🏠 Supervised Learning (House Price Prediction, Sentiment Analysis)
    - 🔍 Unsupervised Learning (Clustering)
    - 📝 NLP (Text Classification, NER)
    - 👁️ Computer Vision (Image Classification, Object Detection)
    - 📈 Time Series Forecasting (ARIMA, LSTM)
    - 🎮 Reinforcement Learning (CartPole with DQN)
    """,
    version=app_config.VERSION,
    contact={
        "name": "Ashraf Khan",
        "email": "ashrafksalim1@gmail.com",
    },
    docs_url=f"{app_config.API_PREFIX}/docs",
    redoc_url=f"{app_config.API_PREFIX}/redoc",
)

# Create necessary directories
os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app_config.MODEL_DIR, exist_ok=True)

# Mount static files for the portfolio
PORTFOLIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "portfolio"))

# Create directories if they don't exist
os.makedirs(os.path.join(PORTFOLIO_DIR, "static"), exist_ok=True)

# Serve portfolio static files from the static directory
app.mount("/static", StaticFiles(directory=os.path.join(PORTFOLIO_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=app_config.UPLOAD_FOLDER), name="uploads")

# Serve CSS and JS files from the portfolio root
app.mount("/css", StaticFiles(directory=PORTFOLIO_DIR), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(PORTFOLIO_DIR, "js")), name="js")

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

# Add license info to OpenAPI schema
def custom_openapi():
    if not app.openapi_schema:
        # Get the default schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        # Add license info
        openapi_schema["info"]["license"] = {
            "name": "MIT"
        }
        app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(supervised_router, prefix="/api/supervised", tags=["Supervised Learning"])
app.include_router(unsupervised_router, prefix="/api/unsupervised", tags=["Unsupervised Learning"])
app.include_router(nlp_router, prefix="/api/nlp", tags=["Natural Language Processing"])
app.include_router(cv_router, prefix="/api/cv", tags=["Computer Vision"])
app.include_router(time_series_router, prefix="/api/time-series", tags=["Time Series"])
app.include_router(rl_router, prefix="/api/rl", tags=["Reinforcement Learning"])

# Serve the main portfolio page
@app.get("/", response_class=HTMLResponse)
async def serve_portfolio():
    index_path = os.path.join(PORTFOLIO_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return FileResponse(index_path)

# Serve other HTML pages
@app.get("/{page}", response_class=HTMLResponse)
async def serve_pages(page: str):
    # Check if the requested page exists in the portfolio directory
    page_path = os.path.join(PORTFOLIO_DIR, f"{page}.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    
    # If not found, try with .html extension
    if not page.endswith('.html'):
        page_path = os.path.join(PORTFOLIO_DIR, f"{page}.html")
        if os.path.exists(page_path):
            return FileResponse(page_path)
    
    # If still not found, return 404
    raise HTTPException(status_code=404, detail="Page not found")

# API root endpoint
@app.get("/api")
async def api_root():
    return {"message": "Welcome to the AI Portfolio API! Visit /docs for API documentation."}

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
