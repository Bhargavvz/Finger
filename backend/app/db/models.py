"""
SQLAlchemy Database Models
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    """User model for authentication."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"


class Prediction(Base):
    """Prediction history model."""
    
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Input data
    image_path = Column(String(512), nullable=False)
    image_hash = Column(String(64), nullable=True)  # SHA256 hash of image
    
    # Prediction results
    predicted_class = Column(String(10), nullable=False)  # e.g., "A+", "O-"
    confidence = Column(Float, nullable=False)
    
    # Full probability distribution
    probabilities = Column(JSON, nullable=True)  # {"A+": 0.85, "A-": 0.05, ...}
    
    # Metadata
    model_version = Column(String(50), nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction {self.id} - {self.predicted_class} ({self.confidence:.2f})>"


class APIKey(Base):
    """API Key model for programmatic access."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    
    is_active = Column(Boolean, default=True)
    
    # Permissions
    can_predict = Column(Boolean, default=True)
    can_view_history = Column(Boolean, default=True)
    
    # Rate limiting
    rate_limit = Column(Integer, default=100)  # requests per minute
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User")
    
    def __repr__(self):
        return f"<APIKey {self.name}>"


class ModelMetadata(Base):
    """Model version and metadata tracking."""
    
    __tablename__ = "model_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    version = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)  # "efficientnet", "resnet", etc.
    
    # Training info
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    
    # File info
    file_path = Column(String(512), nullable=False)
    file_size_mb = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False)  # Currently deployed model
    is_available = Column(Boolean, default=True)
    
    # Metadata
    config = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<ModelMetadata {self.version}>"


class AuditLog(Base):
    """Audit log for tracking important actions."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    action = Column(String(100), nullable=False)  # e.g., "login", "predict", "export"
    resource_type = Column(String(50), nullable=True)  # e.g., "prediction", "user"
    resource_id = Column(String(100), nullable=True)
    
    details = Column(JSON, nullable=True)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AuditLog {self.action} at {self.created_at}>"
