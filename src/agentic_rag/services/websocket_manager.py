"""
WebSocket Manager for real-time upload progress tracking.

This module provides WebSocket connection management and real-time progress
updates for file upload operations.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agentic_rag.api.models.upload import UploadProgressUpdate, UploadStatus

logger = logging.getLogger(__name__)


class WebSocketConnection(BaseModel):
    """WebSocket connection information."""
    
    websocket: WebSocket
    tenant_id: UUID
    user_id: UUID
    upload_ids: Set[UUID]
    connected_at: datetime
    last_ping: datetime
    
    class Config:
        arbitrary_types_allowed = True


class ProgressMessage(BaseModel):
    """Progress message for WebSocket communication."""
    
    type: str = "progress_update"
    upload_id: UUID
    progress: UploadProgressUpdate
    timestamp: datetime


class WebSocketManager:
    """Manager for WebSocket connections and real-time progress updates."""
    
    def __init__(self):
        # Active connections by connection ID
        self.connections: Dict[str, WebSocketConnection] = {}
        
        # Upload ID to connection IDs mapping for efficient lookup
        self.upload_connections: Dict[UUID, Set[str]] = {}
        
        # Tenant-based connection grouping
        self.tenant_connections: Dict[UUID, Set[str]] = {}
        
        # Background task for connection cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the WebSocket manager."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_connections())
            logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop the WebSocket manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)
        
        logger.info("WebSocket manager stopped")
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        tenant_id: UUID,
        user_id: UUID,
        upload_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: WebSocket connection
            connection_id: Unique connection identifier
            tenant_id: Tenant ID for isolation
            user_id: User ID for authorization
            upload_ids: Optional list of upload IDs to track
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                tenant_id=tenant_id,
                user_id=user_id,
                upload_ids=set(upload_ids or []),
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow()
            )
            
            self.connections[connection_id] = connection
            
            # Add to tenant connections
            if tenant_id not in self.tenant_connections:
                self.tenant_connections[tenant_id] = set()
            self.tenant_connections[tenant_id].add(connection_id)
            
            # Add to upload connections
            for upload_id in connection.upload_ids:
                if upload_id not in self.upload_connections:
                    self.upload_connections[upload_id] = set()
                self.upload_connections[upload_id].add(connection_id)
            
            logger.info(f"WebSocket connected: {connection_id} for tenant {tenant_id}")
            
            # Send connection confirmation
            await self._send_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket {connection_id}: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket {connection_id}: {e}")
        
        # Remove from upload connections
        for upload_id in connection.upload_ids:
            if upload_id in self.upload_connections:
                self.upload_connections[upload_id].discard(connection_id)
                if not self.upload_connections[upload_id]:
                    del self.upload_connections[upload_id]
        
        # Remove from tenant connections
        if connection.tenant_id in self.tenant_connections:
            self.tenant_connections[connection.tenant_id].discard(connection_id)
            if not self.tenant_connections[connection.tenant_id]:
                del self.tenant_connections[connection.tenant_id]
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def add_upload_tracking(self, connection_id: str, upload_id: UUID):
        """Add upload tracking to an existing connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.upload_ids.add(upload_id)
        
        # Add to upload connections
        if upload_id not in self.upload_connections:
            self.upload_connections[upload_id] = set()
        self.upload_connections[upload_id].add(connection_id)
        
        logger.debug(f"Added upload tracking {upload_id} to connection {connection_id}")
        return True
    
    async def send_progress_update(self, upload_id: UUID, progress: UploadProgressUpdate):
        """Send progress update to all connections tracking the upload."""
        if upload_id not in self.upload_connections:
            return
        
        message = ProgressMessage(
            upload_id=upload_id,
            progress=progress,
            timestamp=datetime.utcnow()
        )
        
        # Send to all connections tracking this upload
        connection_ids = list(self.upload_connections[upload_id])
        for connection_id in connection_ids:
            await self._send_message(connection_id, message.dict())
    
    async def send_upload_complete(self, upload_id: UUID, document_id: UUID, message: str):
        """Send upload completion notification."""
        if upload_id not in self.upload_connections:
            return
        
        completion_message = {
            "type": "upload_complete",
            "upload_id": str(upload_id),
            "document_id": str(document_id),
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        connection_ids = list(self.upload_connections[upload_id])
        for connection_id in connection_ids:
            await self._send_message(connection_id, completion_message)
    
    async def send_upload_error(self, upload_id: UUID, error: str):
        """Send upload error notification."""
        if upload_id not in self.upload_connections:
            return
        
        error_message = {
            "type": "upload_error",
            "upload_id": str(upload_id),
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        connection_ids = list(self.upload_connections[upload_id])
        for connection_id in connection_ids:
            await self._send_message(connection_id, error_message)
    
    async def _send_message(self, connection_id: str, message: dict):
        """Send message to a specific connection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message, default=str))
        except WebSocketDisconnect:
            logger.info(f"WebSocket {connection_id} disconnected during send")
            await self.disconnect(connection_id)
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def _cleanup_connections(self):
        """Background task to clean up stale connections."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check if connection is stale (no ping for 5 minutes)
                    if (current_time - connection.last_ping).total_seconds() > 300:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.info(f"Cleaning up stale connection: {connection_id}")
                    await self.disconnect(connection_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connections)
    
    def get_tenant_connection_count(self, tenant_id: UUID) -> int:
        """Get number of connections for a specific tenant."""
        return len(self.tenant_connections.get(tenant_id, set()))
    
    def get_upload_connection_count(self, upload_id: UUID) -> int:
        """Get number of connections tracking a specific upload."""
        return len(self.upload_connections.get(upload_id, set()))

    async def add_upload_tracking(self, connection_id: str, upload_ids: List[UUID]):
        """Add upload IDs to track for an existing connection."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        for upload_id in upload_ids:
            # Add to connection's tracked uploads
            connection.upload_ids.add(upload_id)

            # Add to upload connections mapping
            if upload_id not in self.upload_connections:
                self.upload_connections[upload_id] = set()
            self.upload_connections[upload_id].add(connection_id)

        logger.info(f"Added upload tracking for {connection_id}: {upload_ids}")

    async def remove_upload_tracking(self, connection_id: str, upload_ids: List[UUID]):
        """Remove upload IDs from tracking for an existing connection."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        for upload_id in upload_ids:
            # Remove from connection's tracked uploads
            connection.upload_ids.discard(upload_id)

            # Remove from upload connections mapping
            if upload_id in self.upload_connections:
                self.upload_connections[upload_id].discard(connection_id)

                # Clean up empty sets
                if not self.upload_connections[upload_id]:
                    del self.upload_connections[upload_id]

        logger.info(f"Removed upload tracking for {connection_id}: {upload_ids}")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
