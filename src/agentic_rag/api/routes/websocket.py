"""
WebSocket routes for real-time upload progress tracking.

This module defines WebSocket endpoints for real-time communication
with clients during file upload operations.
"""

import json
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer

from agentic_rag.api.dependencies.auth import get_current_user_websocket
from agentic_rag.models.database import User
from agentic_rag.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
router = APIRouter(prefix="/ws", tags=["WebSocket"])

# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance."""
    return websocket_manager


# Note: WebSocket manager lifecycle is now handled in the main app lifespan
# These event handlers are deprecated in favor of lifespan events


@router.websocket("/upload/progress/{upload_id}")
async def websocket_upload_progress(
    websocket: WebSocket,
    upload_id: UUID,
    token: Optional[str] = None,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time upload progress tracking.
    
    Clients can connect to this endpoint to receive real-time progress updates
    for a specific upload session.
    
    Args:
        websocket: WebSocket connection
        upload_id: Upload session ID to track
        token: Optional JWT token for authentication
        manager: WebSocket manager instance
    """
    connection_id = f"upload-{upload_id}-{id(websocket)}"
    
    try:
        # Authenticate user if token provided
        current_user = None
        if token:
            try:
                # Note: This is a simplified authentication for WebSocket
                # In production, you might want more robust WebSocket auth
                current_user = await get_current_user_websocket(token)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        
        # For now, allow unauthenticated connections for development
        # In production, you should require authentication
        if not current_user:
            # Create a mock user for development
            # In production, this should be removed and authentication required
            logger.warning("WebSocket connection without authentication - development mode")
            tenant_id = UUID("00000000-0000-0000-0000-000000000000")
            user_id = UUID("00000000-0000-0000-0000-000000000000")
        else:
            tenant_id = current_user.tenant_id
            user_id = current_user.id
        
        # Connect to WebSocket manager
        success = await manager.connect(
            websocket=websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=[upload_id]
        )
        
        if not success:
            logger.error(f"Failed to establish WebSocket connection {connection_id}")
            return
        
        logger.info(f"WebSocket connected for upload {upload_id}: {connection_id}")
        
        # Keep connection alive and handle incoming messages
        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": message.get("timestamp")
                        }))
                    elif message_type == "subscribe":
                        # Subscribe to additional upload IDs
                        additional_upload_ids = message.get("upload_ids", [])
                        if additional_upload_ids:
                            await manager.add_upload_tracking(
                                connection_id, 
                                [UUID(uid) for uid in additional_upload_ids]
                            )
                    elif message_type == "unsubscribe":
                        # Unsubscribe from upload IDs
                        remove_upload_ids = message.get("upload_ids", [])
                        if remove_upload_ids:
                            await manager.remove_upload_tracking(
                                connection_id,
                                [UUID(uid) for uid in remove_upload_ids]
                            )
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from WebSocket {connection_id}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            # Ensure cleanup
            await manager.disconnect(connection_id)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


@router.websocket("/upload/status")
async def websocket_upload_status(
    websocket: WebSocket,
    token: Optional[str] = None,
    manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for general upload status updates.
    
    Clients can connect to this endpoint to receive updates about all
    their upload sessions.
    
    Args:
        websocket: WebSocket connection
        token: Optional JWT token for authentication
        manager: WebSocket manager instance
    """
    connection_id = f"status-{id(websocket)}"
    
    try:
        # Authenticate user if token provided
        current_user = None
        if token:
            try:
                current_user = await get_current_user_websocket(token)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        
        # For now, allow unauthenticated connections for development
        if not current_user:
            logger.warning("WebSocket connection without authentication - development mode")
            tenant_id = UUID("00000000-0000-0000-0000-000000000000")
            user_id = UUID("00000000-0000-0000-0000-000000000000")
        else:
            tenant_id = current_user.tenant_id
            user_id = current_user.id
        
        # Connect to WebSocket manager (no specific upload IDs)
        success = await manager.connect(
            websocket=websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=[]
        )
        
        if not success:
            logger.error(f"Failed to establish WebSocket connection {connection_id}")
            return
        
        logger.info(f"WebSocket connected for general status: {connection_id}")
        
        # Keep connection alive and handle incoming messages
        try:
            while True:
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": message.get("timestamp")
                        }))
                    elif message_type == "subscribe":
                        # Subscribe to specific upload IDs
                        upload_ids = message.get("upload_ids", [])
                        if upload_ids:
                            await manager.add_upload_tracking(
                                connection_id,
                                [UUID(uid) for uid in upload_ids]
                            )
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from WebSocket {connection_id}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            await manager.disconnect(connection_id)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass
