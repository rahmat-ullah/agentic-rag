"""
Authentication service for Agentic RAG System.

This module provides JWT-based authentication with secure password hashing
and token management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import structlog

from agentic_rag.config import get_settings
from agentic_rag.models.database import User, Tenant
from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.api.exceptions import AuthenticationError, AuthorizationError


logger = structlog.get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service for JWT token management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db = get_database_adapter()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.security.jwt_access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.jwt_secret_key,
            algorithm=self.settings.security.jwt_algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        
        expire = datetime.utcnow() + timedelta(
            days=self.settings.security.jwt_refresh_token_expire_days
        )
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.jwt_secret_key,
            algorithm=self.settings.security.jwt_algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.settings.security.jwt_secret_key,
                algorithms=[self.settings.security.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != token_type:
                raise AuthenticationError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None or datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise AuthenticationError("Token has expired")
            
            return payload
            
        except JWTError as e:
            logger.warning("JWT verification failed", error=str(e))
            raise AuthenticationError("Invalid token")
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            with self.db.get_session() as session:
                user = session.query(User).filter(User.email == email).first()
                
                if not user:
                    logger.warning("Authentication failed - user not found", email=email)
                    return None
                
                if not user.is_active:
                    logger.warning("Authentication failed - user inactive", email=email)
                    return None
                
                if not self.verify_password(password, user.password_hash):
                    logger.warning("Authentication failed - invalid password", email=email)
                    return None
                
                logger.info("User authenticated successfully", user_id=str(user.id), email=email)
                return user
                
        except Exception as e:
            logger.error("Authentication error", error=str(e), email=email)
            return None
    
    def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID."""
        try:
            with self.db.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                return user
        except Exception as e:
            logger.error("Error getting user by ID", error=str(e), user_id=str(user_id))
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            with self.db.get_session() as session:
                user = session.query(User).filter(User.email == email).first()
                return user
        except Exception as e:
            logger.error("Error getting user by email", error=str(e), email=email)
            return None
    
    def create_user(self, email: str, password: str, role: str, tenant_id: uuid.UUID) -> User:
        """Create a new user."""
        try:
            with self.db.get_session() as session:
                # Check if user already exists
                existing_user = session.query(User).filter(User.email == email).first()
                if existing_user:
                    raise AuthenticationError("User with this email already exists")
                
                # Create new user
                user = User(
                    tenant_id=tenant_id,
                    email=email,
                    password_hash=self.get_password_hash(password),
                    role=role,
                    is_active=True
                )
                
                session.add(user)
                session.flush()
                
                logger.info("User created successfully", user_id=str(user.id), email=email)
                return user
                
        except Exception as e:
            logger.error("Error creating user", error=str(e), email=email)
            raise
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login user and return tokens."""
        user = self.authenticate_user(email, password)
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        # Create token payload
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
            "tenant_id": str(user.tenant_id),
            "iat": datetime.utcnow()
        }
        
        # Generate tokens
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token({"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.settings.security.jwt_access_token_expire_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role.value,
                "tenant_id": str(user.tenant_id),
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat()
            }
        }
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, "refresh")
            user_id = uuid.UUID(payload.get("sub"))
            
            # Get user
            user = self.get_user_by_id(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Create new access token
            token_data = {
                "sub": str(user.id),
                "email": user.email,
                "role": user.role.value,
                "tenant_id": str(user.tenant_id),
                "iat": datetime.utcnow()
            }
            
            access_token = self.create_access_token(token_data)
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.settings.security.jwt_access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error("Error refreshing token", error=str(e))
            raise AuthenticationError("Invalid refresh token")


# Global auth service instance
auth_service = AuthService()


def get_auth_service() -> AuthService:
    """Get authentication service instance."""
    return auth_service
