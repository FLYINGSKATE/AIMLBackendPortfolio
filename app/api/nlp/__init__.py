from fastapi import APIRouter

router = APIRouter()

# Import and include sub-routers
from .sentiment import router as sentiment_router
from .text_classification import router as text_classification_router

router.include_router(sentiment_router, prefix="/sentiment", tags=["Sentiment Analysis"])
router.include_router(text_classification_router, prefix="/text-classification", tags=["Text Classification"])
