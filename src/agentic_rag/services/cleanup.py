"""
Cleanup service for managing expired sessions, temporary files, and storage optimization.

This module provides cleanup policies and automated maintenance tasks
for the file upload and storage system.
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from agentic_rag.config import Settings
from agentic_rag.models.database import Document
from agentic_rag.services.progress_tracker import ProgressTracker
from agentic_rag.services.storage import StorageService

logger = logging.getLogger(__name__)


class CleanupService:
    """Service for cleanup policies and maintenance tasks."""
    
    def __init__(self, settings: Settings, storage_service: StorageService, progress_tracker: ProgressTracker):
        self.settings = settings
        self.storage_service = storage_service
        self.progress_tracker = progress_tracker
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the cleanup service with periodic tasks."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Cleanup service started")
    
    async def stop(self):
        """Stop the cleanup service."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cleanup service stopped")
    
    async def _periodic_cleanup(self):
        """Run periodic cleanup tasks."""
        cleanup_interval = self.settings.upload.upload_cleanup_interval
        
        while self._running:
            try:
                await self.run_cleanup_tasks()
                await asyncio.sleep(cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(cleanup_interval)
    
    async def run_cleanup_tasks(self):
        """Run all cleanup tasks."""
        logger.info("Starting cleanup tasks")
        
        # Clean up expired upload sessions
        expired_sessions = await self.cleanup_expired_sessions()
        
        # Clean up temporary chunk files
        cleaned_chunks = await self.cleanup_temporary_chunks()
        
        # Clean up orphaned files (files without database records)
        # Note: This is a more complex operation that should be run less frequently
        # orphaned_files = await self.cleanup_orphaned_files()
        
        logger.info(f"Cleanup completed: {expired_sessions} expired sessions, {cleaned_chunks} chunk files")
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired upload sessions."""
        try:
            expired_count = await self.progress_tracker.cleanup_expired_sessions()
            logger.info(f"Cleaned up {expired_count} expired upload sessions")
            return expired_count
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    async def cleanup_temporary_chunks(self) -> int:
        """Clean up temporary chunk files."""
        try:
            chunk_dir = Path(self.settings.upload.chunk_storage_path)
            if not chunk_dir.exists():
                return 0
            
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(
                seconds=self.settings.upload.chunk_storage_cleanup_interval
            )
            
            # Clean up old chunk directories
            for session_dir in chunk_dir.iterdir():
                if session_dir.is_dir():
                    try:
                        # Check if directory is old enough to clean up
                        dir_mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                        if dir_mtime < cutoff_time:
                            shutil.rmtree(session_dir)
                            cleaned_count += 1
                            logger.debug(f"Cleaned up chunk directory: {session_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up chunk directory {session_dir}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} temporary chunk directories")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary chunks: {e}")
            return 0
    
    async def cleanup_orphaned_files(self, db_session: Session, tenant_id: Optional[UUID] = None) -> int:
        """
        Clean up orphaned files (files in storage without database records).
        
        This is a more expensive operation that should be run less frequently.
        """
        try:
            # Get all document URIs from database
            query = db_session.query(Document.source_uri).filter(Document.source_uri.isnot(None))
            if tenant_id:
                query = query.filter(Document.tenant_id == tenant_id)
            
            db_uris = {uri[0] for uri in query.all()}
            
            # List all files in storage
            # Note: This is a simplified implementation
            # In production, you'd want to paginate through storage objects
            storage_files = await self.storage_service.list_all_files()
            
            orphaned_count = 0
            for file_uri in storage_files:
                if file_uri not in db_uris:
                    try:
                        await self.storage_service.delete_file(file_uri)
                        orphaned_count += 1
                        logger.debug(f"Cleaned up orphaned file: {file_uri}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up orphaned file {file_uri}: {e}")
            
            logger.info(f"Cleaned up {orphaned_count} orphaned files")
            return orphaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned files: {e}")
            return 0
    
    async def cleanup_tenant_data(self, tenant_id: UUID, db_session: Session) -> Dict[str, int]:
        """
        Clean up all data for a specific tenant.
        
        This is used when a tenant is deleted or for data retention compliance.
        """
        cleanup_stats = {
            "documents": 0,
            "files": 0,
            "sessions": 0
        }
        
        try:
            # Get all documents for the tenant
            documents = db_session.query(Document).filter(
                Document.tenant_id == tenant_id
            ).all()
            
            # Delete files from storage
            for document in documents:
                if document.source_uri:
                    try:
                        await self.storage_service.delete_file(document.source_uri)
                        cleanup_stats["files"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete file {document.source_uri}: {e}")
            
            # Delete documents from database
            db_session.query(Document).filter(
                Document.tenant_id == tenant_id
            ).delete()
            cleanup_stats["documents"] = len(documents)
            
            # Clean up upload sessions for the tenant
            sessions_cleaned = await self.progress_tracker.cleanup_tenant_sessions(tenant_id)
            cleanup_stats["sessions"] = sessions_cleaned
            
            db_session.commit()
            
            logger.info(f"Cleaned up tenant {tenant_id} data: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error cleaning up tenant data: {e}")
            raise
    
    async def get_cleanup_statistics(self, db_session: Session) -> Dict[str, int]:
        """Get statistics about items that could be cleaned up."""
        stats = {
            "expired_sessions": 0,
            "temporary_chunks": 0,
            "total_documents": 0,
            "total_storage_size": 0
        }
        
        try:
            # Count expired sessions
            stats["expired_sessions"] = await self.progress_tracker.count_expired_sessions()
            
            # Count temporary chunk directories
            chunk_dir = Path(self.settings.upload.chunk_storage_path)
            if chunk_dir.exists():
                stats["temporary_chunks"] = len([
                    d for d in chunk_dir.iterdir() 
                    if d.is_dir()
                ])
            
            # Count total documents
            stats["total_documents"] = db_session.query(Document).count()
            
            # Estimate total storage size (simplified)
            stats["total_storage_size"] = stats["total_documents"] * 1024 * 1024  # 1MB per doc estimate
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cleanup statistics: {e}")
            return stats


# Global cleanup service instance
cleanup_service: Optional[CleanupService] = None


def get_cleanup_service(settings: Settings, storage_service: StorageService, progress_tracker: ProgressTracker) -> CleanupService:
    """Get or create cleanup service instance."""
    global cleanup_service
    if cleanup_service is None:
        cleanup_service = CleanupService(settings, storage_service, progress_tracker)
    return cleanup_service
