"""
Health Check and System Status Routes
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_db
from app.services import get_ml_service
from app.schemas import HealthCheck, ModelInfo

router = APIRouter(tags=["Health"])
settings = get_settings()

# Track server start time
_start_time = time.time()


@router.get("/health", response_model=HealthCheck)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Check API health status.
    
    Returns:
    - API status
    - Version
    - Model status
    - Database connection status
    - Uptime
    """
    # Check database connection
    db_connected = False
    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass
    
    # Check model
    ml_service = get_ml_service()
    model_loaded = ml_service.is_loaded
    
    # Calculate uptime
    uptime = time.time() - _start_time
    
    # Determine overall status
    if db_connected and model_loaded:
        status = "healthy"
    elif db_connected or model_loaded:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthCheck(
        status=status,
        version=settings.APP_VERSION,
        model_loaded=model_loaded,
        database_connected=db_connected,
        uptime_seconds=uptime
    )


@router.get("/health/model", response_model=ModelInfo)
async def model_info():
    """
    Get information about the loaded model.
    """
    ml_service = get_ml_service()
    
    return ModelInfo(
        version=ml_service.model_version or "unknown",
        model_type=ml_service.model_type or "unknown",
        accuracy=None,  # Would be loaded from model metadata
        f1_score=None,
        is_active=ml_service.is_loaded,
        deployed_at=datetime.utcnow() if ml_service.is_loaded else None
    )


@router.get("/")
async def root():
    """
    API root endpoint.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }
