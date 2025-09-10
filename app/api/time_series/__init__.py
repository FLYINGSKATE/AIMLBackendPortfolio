from fastapi import APIRouter

router = APIRouter()

# Import and include sub-routers
from .forecasting import router as forecasting_router

router.include_router(
    forecasting_router,
    prefix="/forecast",
    tags=["Time Series Forecasting"]
)
