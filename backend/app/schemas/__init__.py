"""
Pydantic Schemas for API Request/Response Validation
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """Schema for user registration."""
    password: str = Field(..., min_length=8, max_length=100)
    
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
    full_name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserInDB(UserResponse):
    """Schema for user with hashed password (internal use)."""
    hashed_password: str


# ==================== Auth Schemas ====================

class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str


class LoginRequest(BaseModel):
    """Schema for login request."""
    username: str
    password: str


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
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

class PredictionBase(BaseModel):
    """Base prediction schema."""
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionCreate(PredictionBase):
    """Schema for creating prediction (internal use)."""
    image_path: str
    image_hash: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    inference_time_ms: Optional[float] = None


class PredictionResponse(PredictionBase):
    """Schema for single prediction response."""
    id: UUID
    probabilities: Dict[str, float]
    inference_time_ms: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Schema for prediction request (if using base64)."""
    image_base64: str
    return_probabilities: bool = True


class PredictionResult(BaseModel):
    """Schema for API prediction response."""
    blood_group: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None
    inference_time_ms: float
    prediction_id: Optional[UUID] = None


class BatchPredictionResult(BaseModel):
    """Schema for batch prediction response."""
    results: List[PredictionResult]
    total_time_ms: float


# ==================== History Schemas ====================

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


# ==================== Statistics Schemas ====================

class UserStats(BaseModel):
    """Schema for user statistics."""
    total_predictions: int
    predictions_this_month: int
    most_common_result: Optional[str] = None
    average_confidence: Optional[float] = None


class BloodGroupDistribution(BaseModel):
    """Schema for blood group distribution."""
    blood_group: str
    count: int
    percentage: float


class SystemStats(BaseModel):
    """Schema for system statistics (admin)."""
    total_users: int
    total_predictions: int
    predictions_today: int
    average_inference_time_ms: float
    blood_group_distribution: List[BloodGroupDistribution]


# ==================== API Key Schemas ====================

class APIKeyCreate(BaseModel):
    """Schema for API key creation."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    id: UUID
    name: str
    key: str  # Only returned once on creation
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class APIKeyListItem(BaseModel):
    """Schema for API key list item (without key value)."""
    id: UUID
    name: str
    is_active: bool
    total_requests: int
    last_used_at: Optional[datetime] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== Model Schemas ====================

class ModelInfo(BaseModel):
    """Schema for model information."""
    version: str
    model_type: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    is_active: bool
    deployed_at: Optional[datetime] = None


# ==================== Health Check Schemas ====================

class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    model_loaded: bool
    database_connected: bool
    uptime_seconds: float


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
