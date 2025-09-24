"""
Progress Tracking Service for upload operations.

This module provides comprehensive progress tracking, session management,
and resumption capabilities for file uploads.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import UUID

from agentic_rag.api.models.upload import (
    UploadProgressUpdate, 
    UploadSession, 
    UploadStatus,
    FileValidationError
)
from agentic_rag.config import Settings
from agentic_rag.services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Service for tracking upload progress and managing sessions."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: Dict[UUID, UploadSession] = {}
        self.session_locks: Dict[UUID, asyncio.Lock] = {}
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the progress tracker."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            logger.info("Progress tracker started")
    
    async def stop(self):
        """Stop the progress tracker."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Progress tracker stopped")
    
    async def create_session(self, session: UploadSession) -> UploadSession:
        """Create a new upload session."""
        session_id = session.id
        
        # Initialize session lock
        self.session_locks[session_id] = asyncio.Lock()
        
        # Store session
        self.sessions[session_id] = session
        
        # Send initial progress update
        await self._send_progress_update(session)
        
        logger.info(f"Created upload session {session_id} for file {session.filename}")
        return session
    
    async def get_session(self, session_id: UUID) -> Optional[UploadSession]:
        """Get upload session by ID."""
        return self.sessions.get(session_id)
    
    async def update_progress(
        self,
        session_id: UUID,
        bytes_uploaded: int,
        status: Optional[UploadStatus] = None,
        message: Optional[str] = None,
        error: Optional[FileValidationError] = None,
        current_chunk: Optional[int] = None
    ) -> bool:
        """
        Update upload progress for a session.
        
        Args:
            session_id: Upload session ID
            bytes_uploaded: Number of bytes uploaded
            status: New upload status
            message: Status message
            error: Error details if failed
            current_chunk: Current chunk number for chunked uploads
            
        Returns:
            True if update successful, False if session not found
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found for progress update")
            return False
        
        # Get session lock to prevent race conditions
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            
            # Update session
            session.bytes_uploaded = bytes_uploaded
            session.updated_at = datetime.utcnow()
            
            if status:
                session.status = status
            
            if message:
                session.message = message
            
            if error:
                session.error = error
            
            if current_chunk is not None:
                session.current_chunk = current_chunk
            
            # Calculate progress percentage
            if session.total_size > 0:
                session.progress_percent = min(100.0, (bytes_uploaded / session.total_size) * 100)
            else:
                session.progress_percent = 0.0
            
            # Send progress update via WebSocket
            await self._send_progress_update(session)
            
            logger.debug(f"Updated progress for session {session_id}: {session.progress_percent:.1f}%")
            return True
    
    async def complete_upload(
        self,
        session_id: UUID,
        document_id: UUID,
        message: str = "Upload completed successfully"
    ) -> bool:
        """Mark upload as complete."""
        if session_id not in self.sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            session.status = UploadStatus.COMPLETE
            session.progress_percent = 100.0
            session.message = message
            session.updated_at = datetime.utcnow()
            session.document_id = document_id
            
            # Send completion notification
            await websocket_manager.send_upload_complete(session_id, document_id, message)
            await self._send_progress_update(session)
            
            logger.info(f"Upload session {session_id} completed successfully")
            return True
    
    async def fail_upload(
        self,
        session_id: UUID,
        error: str,
        error_details: Optional[FileValidationError] = None
    ) -> bool:
        """Mark upload as failed."""
        if session_id not in self.sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            session.status = UploadStatus.FAILED
            session.error_message = error
            session.error = error_details
            session.updated_at = datetime.utcnow()
            
            # Send error notification
            await websocket_manager.send_upload_error(session_id, error)
            await self._send_progress_update(session)
            
            logger.error(f"Upload session {session_id} failed: {error}")
            return True
    
    async def pause_upload(self, session_id: UUID) -> bool:
        """Pause an upload session."""
        if session_id not in self.sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            session.status = UploadStatus.PAUSED
            session.updated_at = datetime.utcnow()
            
            await self._send_progress_update(session)
            
            logger.info(f"Upload session {session_id} paused")
            return True
    
    async def resume_upload(self, session_id: UUID) -> bool:
        """Resume a paused upload session."""
        if session_id not in self.sessions:
            return False
        
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            
            # Check if session can be resumed
            if session.status not in [UploadStatus.PAUSED, UploadStatus.FAILED]:
                logger.warning(f"Cannot resume session {session_id} with status {session.status}")
                return False
            
            # Check if session hasn't expired
            if datetime.utcnow() > session.expires_at:
                logger.warning(f"Cannot resume expired session {session_id}")
                return False
            
            session.status = UploadStatus.PROCESSING
            session.updated_at = datetime.utcnow()
            
            await self._send_progress_update(session)
            
            logger.info(f"Upload session {session_id} resumed")
            return True
    
    async def extend_session(self, session_id: UUID, additional_seconds: int = None) -> bool:
        """Extend session expiration time."""
        if session_id not in self.sessions:
            return False
        
        if additional_seconds is None:
            additional_seconds = self.settings.upload.upload_session_timeout
        
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            session.expires_at = datetime.utcnow() + timedelta(seconds=additional_seconds)
            session.updated_at = datetime.utcnow()
            
            logger.debug(f"Extended session {session_id} by {additional_seconds} seconds")
            return True
    
    async def cleanup_session(self, session_id: UUID):
        """Clean up a completed or expired session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
        logger.debug(f"Cleaned up session {session_id}")
    
    async def get_resumable_sessions(self, tenant_id: UUID, user_id: UUID) -> list[UploadSession]:
        """Get resumable sessions for a user."""
        resumable = []
        current_time = datetime.utcnow()
        
        for session in self.sessions.values():
            if (session.tenant_id == tenant_id and 
                session.user_id == user_id and
                session.status in [UploadStatus.PAUSED, UploadStatus.FAILED] and
                current_time < session.expires_at):
                resumable.append(session)
        
        return resumable
    
    async def _send_progress_update(self, session: UploadSession):
        """Send progress update via WebSocket."""
        progress = UploadProgressUpdate(
            upload_id=session.id,
            status=session.status,
            progress_percent=session.progress_percent,
            bytes_uploaded=session.bytes_uploaded,
            total_bytes=session.total_size,
            current_chunk=session.current_chunk,
            total_chunks=session.total_chunks,
            message=session.message,
            error=session.error
        )
        
        await websocket_manager.send_progress_update(session.id, progress)

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired upload sessions."""
        now = datetime.utcnow()
        expired_count = 0

        # Find expired sessions
        expired_session_ids = []
        for session_id, session in self.sessions.items():
            if session.expires_at and session.expires_at < now:
                expired_session_ids.append(session_id)

        # Remove expired sessions
        for session_id in expired_session_ids:
            try:
                async with self.session_locks[session_id]:
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                        expired_count += 1

                # Clean up lock
                if session_id in self.session_locks:
                    del self.session_locks[session_id]

            except Exception as e:
                logger.error(f"Error cleaning up expired session {session_id}: {e}")

        logger.info(f"Cleaned up {expired_count} expired sessions")
        return expired_count

    async def count_expired_sessions(self) -> int:
        """Count the number of expired sessions."""
        now = datetime.utcnow()
        expired_count = 0

        for session in self.sessions.values():
            if session.expires_at and session.expires_at < now:
                expired_count += 1

        return expired_count

    async def cleanup_tenant_sessions(self, tenant_id: UUID) -> int:
        """Clean up all sessions for a specific tenant."""
        tenant_session_ids = []

        # Find sessions for the tenant
        for session_id, session in self.sessions.items():
            if session.tenant_id == tenant_id:
                tenant_session_ids.append(session_id)

        # Remove tenant sessions
        cleaned_count = 0
        for session_id in tenant_session_ids:
            try:
                async with self.session_locks[session_id]:
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                        cleaned_count += 1

                # Clean up lock
                if session_id in self.session_locks:
                    del self.session_locks[session_id]

            except Exception as e:
                logger.error(f"Error cleaning up tenant session {session_id}: {e}")

        logger.info(f"Cleaned up {cleaned_count} sessions for tenant {tenant_id}")
        return cleaned_count
    
    async def _cleanup_expired_sessions(self):
        """Background task to clean up expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(self.settings.upload.upload_cleanup_interval)
                
                current_time = datetime.utcnow()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    # Clean up expired sessions or completed sessions older than 1 hour
                    if (current_time > session.expires_at or 
                        (session.status == UploadStatus.COMPLETE and 
                         (current_time - session.updated_at).total_seconds() > 3600)):
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    await self.cleanup_session(session_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)
    
    def get_tenant_session_count(self, tenant_id: UUID) -> int:
        """Get number of sessions for a specific tenant."""
        return sum(1 for session in self.sessions.values() if session.tenant_id == tenant_id)


# Global progress tracker instance factory
def create_progress_tracker(settings: Settings) -> ProgressTracker:
    """Create a progress tracker instance with the given settings."""
    return ProgressTracker(settings)
