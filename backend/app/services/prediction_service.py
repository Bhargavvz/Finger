"""
Prediction Service - Business Logic for Predictions
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction
from app.schemas import PredictionCreate


class PredictionService:
    """Service class for prediction operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        user_id: UUID,
        prediction_data: PredictionCreate
    ) -> Prediction:
        """Create a new prediction record."""
        prediction = Prediction(
            user_id=user_id,
            **prediction_data.model_dump()
        )
        
        self.db.add(prediction)
        await self.db.flush()
        await self.db.refresh(prediction)
        
        return prediction
    
    async def get_by_id(
        self,
        prediction_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[Prediction]:
        """Get prediction by ID."""
        query = select(Prediction).where(Prediction.id == prediction_id)
        
        if user_id:
            query = query.where(Prediction.user_id == user_id)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_predictions(
        self,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
        blood_group: Optional[str] = None
    ) -> tuple[List[Prediction], int]:
        """Get paginated predictions for a user."""
        # Base query
        query = select(Prediction).where(Prediction.user_id == user_id)
        count_query = select(func.count(Prediction.id)).where(
            Prediction.user_id == user_id
        )
        
        # Filter by blood group if specified
        if blood_group:
            query = query.where(Prediction.predicted_class == blood_group)
            count_query = count_query.where(Prediction.predicted_class == blood_group)
        
        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Order and paginate
        query = query.order_by(desc(Prediction.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        predictions = list(result.scalars().all())
        
        return predictions, total
    
    async def delete(
        self,
        prediction_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete a prediction."""
        prediction = await self.get_by_id(prediction_id, user_id)
        
        if not prediction:
            return False
        
        await self.db.delete(prediction)
        await self.db.flush()
        
        return True
    
    async def get_blood_group_distribution(
        self,
        user_id: Optional[UUID] = None
    ) -> List[dict]:
        """Get distribution of blood group predictions."""
        query = select(
            Prediction.predicted_class,
            func.count(Prediction.id).label("count")
        )
        
        if user_id:
            query = query.where(Prediction.user_id == user_id)
        
        query = query.group_by(Prediction.predicted_class)
        
        result = await self.db.execute(query)
        rows = result.all()
        
        total = sum(row.count for row in rows)
        
        return [
            {
                "blood_group": row.predicted_class,
                "count": row.count,
                "percentage": (row.count / total * 100) if total > 0 else 0
            }
            for row in rows
        ]
    
    async def get_recent_predictions(
        self,
        user_id: UUID,
        limit: int = 5
    ) -> List[Prediction]:
        """Get most recent predictions for a user."""
        query = (
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(desc(Prediction.created_at))
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def count_predictions_today(self, user_id: Optional[UUID] = None) -> int:
        """Count predictions made today."""
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        query = select(func.count(Prediction.id)).where(
            Prediction.created_at >= today_start
        )
        
        if user_id:
            query = query.where(Prediction.user_id == user_id)
        
        result = await self.db.execute(query)
        return result.scalar() or 0
