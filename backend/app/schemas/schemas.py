"""
Pydantic Schemas for API Request/Response Models
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ==================== Auth Schemas ====================

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """Schema for user registration."""
    password: str = Field(..., min_length=8)
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schema for user profile update."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=255)


class UserResponse(BaseModel):
    """Schema for user response."""
    id: UUID
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """Schema for user statistics."""
    total_predictions: int
    predictions_this_month: int
    most_common_result: Optional[str]
    average_confidence: Optional[float]


class LoginRequest(BaseModel):
    """Schema for login request."""
    username: str
    password: str


class Token(BaseModel):
    """Schema for JWT tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ==================== Prediction Schemas ====================

class PredictionCreate(BaseModel):
    """Schema for creating a prediction record."""
    predicted_class: str
    confidence: float
    image_path: str
    image_hash: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    inference_time_ms: Optional[float] = None


class PredictionResult(BaseModel):
    """Schema for prediction result."""
    blood_group: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Optional[Dict[str, float]] = None
    inference_time_ms: float
    prediction_id: Optional[UUID] = None


class PredictionRequest(BaseModel):
    """Schema for base64 prediction request."""
    image_base64: str
    return_probabilities: bool = True


class BatchPredictionResult(BaseModel):
    """Schema for batch prediction results."""
    results: List[PredictionResult]
    total_inference_time_ms: float


class PredictionResponse(BaseModel):
    """Schema for prediction response from database."""
    id: UUID
    predicted_class: str
    confidence: float
    probabilities: Optional[Dict[str, float]]
    model_version: Optional[str]
    inference_time_ms: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    """Schema for prediction history item."""
    id: UUID
    predicted_class: str
    confidence: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Schema for paginated prediction history."""
    items: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    pages: int


# ==================== Health Schemas ====================

class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    model_loaded: bool
    database_connected: bool
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Schema for model information."""
    version: str
    model_type: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    is_active: bool
    deployed_at: Optional[datetime]


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None


class ValidationError(BaseModel):
    """Schema for validation error details."""
    loc: List[str]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Schema for validation error response."""
    detail: List[ValidationError]
