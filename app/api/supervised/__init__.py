from fastapi import APIRouter
from .house_price import router as house_price_router

router = APIRouter()
router.include_router(house_price_router, prefix="/house-price", tags=["House Price Prediction"])
