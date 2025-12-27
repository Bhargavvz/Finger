"""
Authentication Dependencies
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_token
from app.db import get_db, User
from app.services import UserService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Raises HTTPException if not authenticated.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    user_id = verify_token(token, token_type="access")
    
    if user_id is None:
        raise credentials_exception
    
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    
    Useful for endpoints that work both with and without auth.
    """
    if not token:
        return None
    
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify they are an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Verify API key and return associated user.
    """
    if not api_key:
        return None
    
    # Hash the key and look it up
    from app.core.security import get_password_hash
    from sqlalchemy import select
    from app.db import APIKey
    
    # Note: In production, you'd want to use a different hashing approach
    # that allows for key lookup (e.g., prefix + hash)
    result = await db.execute(
        select(APIKey).where(APIKey.is_active == True)
    )
    
    for key_record in result.scalars():
        # This is simplified - in production use proper key comparison
        if api_key == str(key_record.id):  # Simplified for demo
            user_service = UserService(db)
            return await user_service.get_by_id(key_record.user_id)
    
    return None


async def get_current_user_or_api_key(
    user: Optional[User] = Depends(get_current_user_optional),
    api_key_user: Optional[User] = Depends(verify_api_key)
) -> User:
    """
    Get user from either JWT token or API key.
    """
    current_user = user or api_key_user
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (JWT or API Key)"
        )
    
    return current_user
