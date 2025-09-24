"""
Authentication endpoints for the Agentic RAG API.

This module provides authentication endpoints including login and token refresh.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from agentic_rag.api.models.auth import LoginRequest, LoginResponse, RefreshTokenRequest, TokenResponse
from agentic_rag.api.models.responses import SuccessResponse, ErrorResponse
from agentic_rag.services.auth import get_auth_service, AuthService
from agentic_rag.api.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=SuccessResponse[LoginResponse])
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    User login endpoint.

    Authenticates user with email and password, returns JWT tokens.
    """
    try:
        result = auth_service.login(request.email, request.password)

        return SuccessResponse(
            message="Login successful",
            data=LoginResponse(**result)
        )

    except AuthenticationError as e:
        logger.warning("Login failed", email=request.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    except Exception as e:
        logger.error("Login error", email=request.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/refresh", response_model=SuccessResponse[TokenResponse])
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Token refresh endpoint.

    Refreshes access token using refresh token.
    """
    try:
        result = auth_service.refresh_access_token(request.refresh_token)

        return SuccessResponse(
            message="Token refreshed successfully",
            data=TokenResponse(**result)
        )

    except AuthenticationError as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
