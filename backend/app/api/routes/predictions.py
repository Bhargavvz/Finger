"""
Prediction API Routes
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings, BLOOD_GROUPS
from app.db import get_db, User
from app.services import get_ml_service, PredictionService
from app.schemas import (
    PredictionCreate,
    PredictionResult,
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    PredictionRequest,
    BatchPredictionResult,
    ErrorResponse
)
from app.api.deps import get_current_user, get_current_user_optional

router = APIRouter(prefix="/predict", tags=["Prediction"])
settings = get_settings()


def validate_image_extension(filename: str) -> bool:
    """Validate image file extension."""
    ext = Path(filename).suffix.lower()
    return ext in settings.ALLOWED_EXTENSIONS


async def save_uploaded_image(file: UploadFile, user_id: str) -> str:
    """Save uploaded image and return path."""
    upload_dir = Path(settings.UPLOAD_DIR) / str(user_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    ext = Path(file.filename).suffix
    filename = f"{uuid4()}{ext}"
    filepath = upload_dir / filename
    
    # Save file
    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    return str(filepath)


@router.post(
    "/",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    }
)
async def predict_blood_group(
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPG)"),
    save_result: bool = Query(True, description="Save prediction to history"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict blood group from a fingerprint image.
    
    Upload a fingerprint image to get the predicted blood group.
    Supported formats: BMP, PNG, JPG, JPEG
    
    Returns:
    - **blood_group**: Predicted blood group (A+, A-, B+, B-, AB+, AB-, O+, O-)
    - **confidence**: Prediction confidence (0-1)
    - **probabilities**: Probability distribution across all classes
    - **inference_time_ms**: Inference time in milliseconds
    """
    # Validate file extension
    if not validate_image_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Check file size
    content = await file.read()
    await file.seek(0)
    
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // (1024*1024)} MB"
        )
    
    # Get ML service
    ml_service = get_ml_service()
    
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Load image
        image = Image.open(io.BytesIO(content))
        
        # Run prediction
        result = ml_service.predict(image, return_probabilities=True)
        
        prediction_id = None
        
        # Save prediction if user is authenticated and save_result is True
        if current_user and save_result:
            # Save image file
            await file.seek(0)
            image_path = await save_uploaded_image(file, str(current_user.id))
            
            # Compute image hash
            image_hash = ml_service.compute_image_hash(content)
            
            # Save to database
            prediction_service = PredictionService(db)
            prediction_data = PredictionCreate(
                predicted_class=result["blood_group"],
                confidence=result["confidence"],
                image_path=image_path,
                image_hash=image_hash,
                probabilities=result.get("probabilities"),
                model_version=ml_service.model_version,
                inference_time_ms=result["inference_time_ms"]
            )
            prediction = await prediction_service.create(
                current_user.id,
                prediction_data
            )
            prediction_id = prediction.id
        
        return PredictionResult(
            blood_group=result["blood_group"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            inference_time_ms=result["inference_time_ms"],
            prediction_id=prediction_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/base64",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    }
)
async def predict_from_base64(
    request: PredictionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    """
    Predict blood group from a base64-encoded fingerprint image.
    
    Useful for mobile apps and web applications that encode images as base64.
    """
    ml_service = get_ml_service()
    
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Decode base64
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Run prediction
        result = ml_service.predict(
            image,
            return_probabilities=request.return_probabilities
        )
        
        return PredictionResult(
            blood_group=result["blood_group"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            inference_time_ms=result["inference_time_ms"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/history",
    response_model=PredictionHistoryResponse
)
async def get_prediction_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    blood_group: Optional[str] = Query(None, description="Filter by blood group"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get authenticated user's prediction history.
    
    Supports pagination and filtering by blood group.
    """
    # Validate blood group filter
    if blood_group and blood_group not in BLOOD_GROUPS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid blood group. Must be one of: {BLOOD_GROUPS}"
        )
    
    prediction_service = PredictionService(db)
    predictions, total = await prediction_service.get_user_predictions(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        blood_group=blood_group
    )
    
    items = [
        PredictionHistoryItem(
            id=p.id,
            predicted_class=p.predicted_class,
            confidence=p.confidence,
            created_at=p.created_at
        )
        for p in predictions
    ]
    
    total_pages = (total + page_size - 1) // page_size
    
    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=total_pages
    )


@router.get(
    "/history/{prediction_id}",
    response_model=PredictionResponse
)
async def get_prediction_detail(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details of a specific prediction.
    """
    prediction_service = PredictionService(db)
    prediction = await prediction_service.get_by_id(
        prediction_id,
        user_id=current_user.id
    )
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    return prediction


@router.delete(
    "/history/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a prediction from history.
    """
    prediction_service = PredictionService(db)
    deleted = await prediction_service.delete(
        prediction_id,
        user_id=current_user.id
    )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )


@router.get("/blood-groups", response_model=list)
async def get_blood_groups():
    """
    Get list of all blood groups the model can predict.
    """
    return BLOOD_GROUPS
