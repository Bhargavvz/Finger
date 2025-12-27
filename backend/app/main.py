"""
FastAPI Application Entry Point

Production-ready API for Fingerprint Blood Group Detection.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.config import get_settings
from app.db import init_db, close_db
from app.services import get_ml_service
from app.api import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")
    
    # Load ML model
    logger.info("Loading ML model...")
    ml_service = get_ml_service()
    model_path = Path(settings.MODEL_PATH)
    
    if model_path.exists():
        success = ml_service.load_model(str(model_path))
        if success:
            logger.info(f"Model loaded: {model_path}")
        else:
            logger.warning("Failed to load model - predictions will be unavailable")
    else:
        logger.warning(f"Model not found at {model_path}")
        logger.info("To enable predictions, train a model and place it at the configured path")
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {upload_dir}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Fingerprint Blood Group Detection API
    
    This API provides blood group prediction from fingerprint images using deep learning.
    
    ### Features:
    - 🔐 **Authentication**: JWT-based authentication with refresh tokens
    - 🩸 **Prediction**: Upload fingerprint images to predict blood group
    - 📊 **History**: Track prediction history with statistics
    - 📈 **Analytics**: View blood group distribution and accuracy metrics
    
    ### Blood Groups:
    The model can predict 8 blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
    
    ### Image Requirements:
    - Supported formats: BMP, PNG, JPG, JPEG
    - Maximum file size: 10 MB
    - Recommended resolution: 224x224 or higher
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Mount static files for uploads (in production, use nginx or CDN)
if Path(settings.UPLOAD_DIR).exists():
    app.mount(
        "/uploads",
        StaticFiles(directory=settings.UPLOAD_DIR),
        name="uploads"
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
