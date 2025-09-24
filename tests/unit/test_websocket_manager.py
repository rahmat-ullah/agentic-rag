"""
Unit tests for the WebSocket manager.
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from agentic_rag.api.models.upload import UploadProgressUpdate, UploadStatus
from agentic_rag.services.websocket_manager import WebSocketManager


@pytest.fixture
async def websocket_manager():
    """Create WebSocket manager instance."""
    manager = WebSocketManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


class TestWebSocketManager:
    """Test cases for WebSocketManager."""
    
    async def test_start_and_stop(self):
        """Test starting and stopping the WebSocket manager."""
        manager = WebSocketManager()
        
        # Start manager
        await manager.start()
        assert manager._running is True
        assert manager._cleanup_task is not None
        
        # Stop manager
        await manager.stop()
        assert manager._running is False
    
    async def test_connect_websocket(self, websocket_manager, mock_websocket):
        """Test connecting a WebSocket client."""
        connection_id = "test-connection-1"
        tenant_id = uuid4()
        user_id = uuid4()
        upload_ids = [uuid4(), uuid4()]
        
        success = await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=upload_ids
        )
        
        assert success is True
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_text.assert_called_once()
        
        # Verify connection is stored
        assert connection_id in websocket_manager.connections
        connection = websocket_manager.connections[connection_id]
        assert connection.tenant_id == tenant_id
        assert connection.user_id == user_id
        assert connection.upload_ids == set(upload_ids)
        
        # Verify mappings
        assert tenant_id in websocket_manager.tenant_connections
        assert connection_id in websocket_manager.tenant_connections[tenant_id]
        
        for upload_id in upload_ids:
            assert upload_id in websocket_manager.upload_connections
            assert connection_id in websocket_manager.upload_connections[upload_id]
    
    async def test_connect_websocket_failure(self, websocket_manager):
        """Test WebSocket connection failure."""
        mock_websocket = AsyncMock()
        mock_websocket.accept.side_effect = Exception("Connection failed")
        
        success = await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id="test-connection",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        assert success is False
    
    async def test_disconnect_websocket(self, websocket_manager, mock_websocket):
        """Test disconnecting a WebSocket client."""
        connection_id = "test-connection-1"
        tenant_id = uuid4()
        user_id = uuid4()
        upload_ids = [uuid4()]
        
        # Connect first
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=upload_ids
        )
        
        # Disconnect
        await websocket_manager.disconnect(connection_id)
        
        # Verify connection is removed
        assert connection_id not in websocket_manager.connections
        
        # Verify mappings are cleaned up
        if tenant_id in websocket_manager.tenant_connections:
            assert connection_id not in websocket_manager.tenant_connections[tenant_id]
        
        for upload_id in upload_ids:
            if upload_id in websocket_manager.upload_connections:
                assert connection_id not in websocket_manager.upload_connections[upload_id]
        
        mock_websocket.close.assert_called_once()
    
    async def test_add_upload_tracking(self, websocket_manager, mock_websocket):
        """Test adding upload tracking to existing connection."""
        connection_id = "test-connection-1"
        tenant_id = uuid4()
        user_id = uuid4()
        initial_upload_id = uuid4()
        new_upload_id = uuid4()
        
        # Connect with initial upload ID
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=[initial_upload_id]
        )
        
        # Add new upload tracking
        success = await websocket_manager.add_upload_tracking(connection_id, new_upload_id)
        assert success is True
        
        # Verify upload ID was added
        connection = websocket_manager.connections[connection_id]
        assert new_upload_id in connection.upload_ids
        assert initial_upload_id in connection.upload_ids
        
        # Verify mapping
        assert new_upload_id in websocket_manager.upload_connections
        assert connection_id in websocket_manager.upload_connections[new_upload_id]
    
    async def test_add_upload_tracking_nonexistent_connection(self, websocket_manager):
        """Test adding upload tracking to nonexistent connection."""
        success = await websocket_manager.add_upload_tracking("nonexistent", uuid4())
        assert success is False
    
    async def test_send_progress_update(self, websocket_manager, mock_websocket):
        """Test sending progress update to connected clients."""
        connection_id = "test-connection-1"
        tenant_id = uuid4()
        user_id = uuid4()
        upload_id = uuid4()
        
        # Connect client
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=tenant_id,
            user_id=user_id,
            upload_ids=[upload_id]
        )
        
        # Send progress update
        progress = UploadProgressUpdate(
            upload_id=upload_id,
            status=UploadStatus.PROCESSING,
            progress_percent=50.0,
            bytes_uploaded=512000,
            total_bytes=1024000,
            message="Processing file..."
        )
        
        await websocket_manager.send_progress_update(upload_id, progress)
        
        # Verify message was sent
        assert mock_websocket.send_text.call_count >= 2  # Connection confirmation + progress update
        
        # Check the progress update message
        calls = mock_websocket.send_text.call_args_list
        progress_call = calls[-1]  # Last call should be progress update
        message_data = json.loads(progress_call[0][0])
        
        assert message_data["type"] == "progress_update"
        assert message_data["upload_id"] == str(upload_id)
        assert message_data["progress"]["status"] == "processing"
        assert message_data["progress"]["progress_percent"] == 50.0
    
    async def test_send_progress_update_no_connections(self, websocket_manager):
        """Test sending progress update when no connections are tracking the upload."""
        upload_id = uuid4()
        progress = UploadProgressUpdate(
            upload_id=upload_id,
            status=UploadStatus.PROCESSING,
            progress_percent=50.0,
            bytes_uploaded=512000,
            total_bytes=1024000
        )
        
        # Should not raise an exception
        await websocket_manager.send_progress_update(upload_id, progress)
    
    async def test_send_upload_complete(self, websocket_manager, mock_websocket):
        """Test sending upload completion notification."""
        connection_id = "test-connection-1"
        upload_id = uuid4()
        document_id = uuid4()
        
        # Connect client
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=uuid4(),
            user_id=uuid4(),
            upload_ids=[upload_id]
        )
        
        # Send completion notification
        await websocket_manager.send_upload_complete(
            upload_id=upload_id,
            document_id=document_id,
            message="Upload completed successfully"
        )
        
        # Verify message was sent
        calls = mock_websocket.send_text.call_args_list
        completion_call = calls[-1]  # Last call should be completion
        message_data = json.loads(completion_call[0][0])
        
        assert message_data["type"] == "upload_complete"
        assert message_data["upload_id"] == str(upload_id)
        assert message_data["document_id"] == str(document_id)
        assert message_data["message"] == "Upload completed successfully"
    
    async def test_send_upload_error(self, websocket_manager, mock_websocket):
        """Test sending upload error notification."""
        connection_id = "test-connection-1"
        upload_id = uuid4()
        
        # Connect client
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=uuid4(),
            user_id=uuid4(),
            upload_ids=[upload_id]
        )
        
        # Send error notification
        await websocket_manager.send_upload_error(
            upload_id=upload_id,
            error="Upload failed due to validation error"
        )
        
        # Verify message was sent
        calls = mock_websocket.send_text.call_args_list
        error_call = calls[-1]  # Last call should be error
        message_data = json.loads(error_call[0][0])
        
        assert message_data["type"] == "upload_error"
        assert message_data["upload_id"] == str(upload_id)
        assert message_data["error"] == "Upload failed due to validation error"
    
    async def test_connection_counts(self, websocket_manager, mock_websocket):
        """Test connection counting methods."""
        tenant1_id = uuid4()
        tenant2_id = uuid4()
        upload_id = uuid4()
        
        # Connect multiple clients
        connections = []
        for i in range(3):
            connection_id = f"connection-{i}"
            tenant_id = tenant1_id if i < 2 else tenant2_id
            upload_ids = [upload_id] if i == 0 else []
            
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_text = AsyncMock()
            
            await websocket_manager.connect(
                websocket=mock_ws,
                connection_id=connection_id,
                tenant_id=tenant_id,
                user_id=uuid4(),
                upload_ids=upload_ids
            )
            connections.append((connection_id, mock_ws))
        
        # Test counts
        assert websocket_manager.get_connection_count() == 3
        assert websocket_manager.get_tenant_connection_count(tenant1_id) == 2
        assert websocket_manager.get_tenant_connection_count(tenant2_id) == 1
        assert websocket_manager.get_upload_connection_count(upload_id) == 1
    
    async def test_websocket_disconnect_during_send(self, websocket_manager):
        """Test handling WebSocket disconnect during message send."""
        from fastapi import WebSocketDisconnect
        
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=WebSocketDisconnect())
        
        connection_id = "test-connection"
        upload_id = uuid4()
        
        # Connect client
        await websocket_manager.connect(
            websocket=mock_websocket,
            connection_id=connection_id,
            tenant_id=uuid4(),
            user_id=uuid4(),
            upload_ids=[upload_id]
        )
        
        # Try to send message (should handle disconnect gracefully)
        progress = UploadProgressUpdate(
            upload_id=upload_id,
            status=UploadStatus.PROCESSING,
            progress_percent=50.0,
            bytes_uploaded=512000,
            total_bytes=1024000
        )
        
        await websocket_manager.send_progress_update(upload_id, progress)
        
        # Connection should be automatically removed
        assert connection_id not in websocket_manager.connections
