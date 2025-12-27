"""
User Service - Business Logic for User Operations
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash, verify_password
from app.db.models import User, Prediction
from app.schemas import UserCreate, UserUpdate


class UserService:
    """Service class for user operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def create(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user object
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password
        )
        
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        
        return user
    
    async def update(self, user: User, user_data: UserUpdate) -> User:
        """Update user profile."""
        update_data = user_data.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        await self.db.flush()
        await self.db.refresh(user)
        
        return user
    
    async def update_password(self, user: User, new_password: str) -> User:
        """Update user password."""
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        
        await self.db.flush()
        
        return user
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        # Try username first
        user = await self.get_by_username(username)
        
        # Try email if username not found
        if not user:
            user = await self.get_by_email(username)
        
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def update_last_login(self, user: User) -> None:
        """Update user's last login timestamp."""
        user.last_login = datetime.utcnow()
        await self.db.flush()
    
    async def get_user_stats(self, user_id: UUID) -> dict:
        """Get statistics for a user."""
        # Total predictions
        total_result = await self.db.execute(
            select(func.count(Prediction.id)).where(
                Prediction.user_id == user_id
            )
        )
        total_predictions = total_result.scalar() or 0
        
        # Predictions this month
        month_start = datetime.utcnow().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        month_result = await self.db.execute(
            select(func.count(Prediction.id)).where(
                Prediction.user_id == user_id,
                Prediction.created_at >= month_start
            )
        )
        predictions_this_month = month_result.scalar() or 0
        
        # Most common result
        common_result = await self.db.execute(
            select(
                Prediction.predicted_class,
                func.count(Prediction.id).label("count")
            ).where(
                Prediction.user_id == user_id
            ).group_by(
                Prediction.predicted_class
            ).order_by(
                func.count(Prediction.id).desc()
            ).limit(1)
        )
        most_common = common_result.first()
        most_common_result = most_common[0] if most_common else None
        
        # Average confidence
        avg_result = await self.db.execute(
            select(func.avg(Prediction.confidence)).where(
                Prediction.user_id == user_id
            )
        )
        average_confidence = avg_result.scalar()
        
        return {
            "total_predictions": total_predictions,
            "predictions_this_month": predictions_this_month,
            "most_common_result": most_common_result,
            "average_confidence": float(average_confidence) if average_confidence else None
        }
