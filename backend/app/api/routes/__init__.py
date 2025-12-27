"""API Routes module exports."""

from fastapi import APIRouter

from app.api.routes import auth, predictions, users, health

# Create main router
api_router = APIRouter()

# Include route modules
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(predictions.router)
api_router.include_router(users.router)

__all__ = ["api_router"]
