"""
Unit tests for the progress tracking service.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from agentic_rag.api.models.upload import UploadSession, UploadStatus, FileValidationError, UploadRequest
from agentic_rag.config import Settings
from agentic_rag.services.progress_tracker import ProgressTracker


@pytest.fixture
def settings():
    """Create test settings."""
    settings = Settings()
    settings.upload.upload_session_timeout = 3600
    settings.upload.upload_cleanup_interval = 300
    return settings


@pytest.fixture
async def progress_tracker(settings):
    """Create progress tracker instance."""
    tracker = ProgressTracker(settings)
    await tracker.start()
    yield tracker
    await tracker.stop()


@pytest.fixture
def sample_session():
    """Create a sample upload session."""
    session_id = uuid4()
    tenant_id = uuid4()
    user_id = uuid4()
    
    return UploadSession(
        id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        status=UploadStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=1),
        filename="test.pdf",
        content_type="application/pdf",
        total_size=1024000,
        chunk_size=None,
        total_chunks=None,
        upload_options=UploadRequest()
    )


class TestProgressTracker:
    """Test cases for ProgressTracker."""
    
    async def test_create_session(self, progress_tracker, sample_session):
        """Test creating a new upload session."""
        created_session = await progress_tracker.create_session(sample_session)
        
        assert created_session.id == sample_session.id
        assert created_session.status == UploadStatus.PENDING
        
        # Verify session is stored
        retrieved_session = await progress_tracker.get_session(sample_session.id)
        assert retrieved_session is not None
        assert retrieved_session.id == sample_session.id
    
    async def test_get_nonexistent_session(self, progress_tracker):
        """Test getting a session that doesn't exist."""
        nonexistent_id = uuid4()
        session = await progress_tracker.get_session(nonexistent_id)
        assert session is None
    
    async def test_update_progress(self, progress_tracker, sample_session):
        """Test updating upload progress."""
        await progress_tracker.create_session(sample_session)
        
        # Update progress
        success = await progress_tracker.update_progress(
            session_id=sample_session.id,
            bytes_uploaded=512000,
            status=UploadStatus.PROCESSING,
            message="Processing file..."
        )
        
        assert success is True
        
        # Verify progress was updated
        session = await progress_tracker.get_session(sample_session.id)
        assert session.bytes_uploaded == 512000
        assert session.status == UploadStatus.PROCESSING
        assert session.message == "Processing file..."
        assert session.progress_percent == 50.0  # 512000 / 1024000 * 100
    
    async def test_update_progress_nonexistent_session(self, progress_tracker):
        """Test updating progress for nonexistent session."""
        nonexistent_id = uuid4()
        success = await progress_tracker.update_progress(
            session_id=nonexistent_id,
            bytes_uploaded=1000
        )
        assert success is False
    
    async def test_complete_upload(self, progress_tracker, sample_session):
        """Test completing an upload."""
        await progress_tracker.create_session(sample_session)
        document_id = uuid4()
        
        success = await progress_tracker.complete_upload(
            session_id=sample_session.id,
            document_id=document_id,
            message="Upload completed successfully"
        )
        
        assert success is True
        
        # Verify completion
        session = await progress_tracker.get_session(sample_session.id)
        assert session.status == UploadStatus.COMPLETE
        assert session.progress_percent == 100.0
        assert session.message == "Upload completed successfully"
        assert session.document_id == document_id
    
    async def test_fail_upload(self, progress_tracker, sample_session):
        """Test failing an upload."""
        await progress_tracker.create_session(sample_session)
        
        error_details = FileValidationError(
            code="INVALID_FORMAT",
            message="Invalid file format",
            field="content_type"
        )
        
        success = await progress_tracker.fail_upload(
            session_id=sample_session.id,
            error="Upload failed due to validation error",
            error_details=error_details
        )
        
        assert success is True
        
        # Verify failure
        session = await progress_tracker.get_session(sample_session.id)
        assert session.status == UploadStatus.FAILED
        assert session.error_message == "Upload failed due to validation error"
        assert session.error == error_details
    
    async def test_pause_and_resume_upload(self, progress_tracker, sample_session):
        """Test pausing and resuming an upload."""
        await progress_tracker.create_session(sample_session)
        
        # Pause upload
        success = await progress_tracker.pause_upload(sample_session.id)
        assert success is True
        
        session = await progress_tracker.get_session(sample_session.id)
        assert session.status == UploadStatus.PAUSED
        
        # Resume upload
        success = await progress_tracker.resume_upload(sample_session.id)
        assert success is True
        
        session = await progress_tracker.get_session(sample_session.id)
        assert session.status == UploadStatus.PROCESSING
    
    async def test_resume_invalid_status(self, progress_tracker, sample_session):
        """Test resuming upload with invalid status."""
        await progress_tracker.create_session(sample_session)
        
        # Complete the upload first
        await progress_tracker.complete_upload(
            session_id=sample_session.id,
            document_id=uuid4()
        )
        
        # Try to resume completed upload
        success = await progress_tracker.resume_upload(sample_session.id)
        assert success is False
    
    async def test_extend_session(self, progress_tracker, sample_session):
        """Test extending session expiration."""
        await progress_tracker.create_session(sample_session)
        original_expiry = sample_session.expires_at
        
        success = await progress_tracker.extend_session(
            session_id=sample_session.id,
            additional_seconds=1800  # 30 minutes
        )
        
        assert success is True
        
        session = await progress_tracker.get_session(sample_session.id)
        assert session.expires_at > original_expiry
    
    async def test_get_resumable_sessions(self, progress_tracker):
        """Test getting resumable sessions for a user."""
        tenant_id = uuid4()
        user_id = uuid4()
        
        # Create multiple sessions with different statuses
        sessions = []
        for i, status in enumerate([UploadStatus.PAUSED, UploadStatus.FAILED, UploadStatus.COMPLETE]):
            session = UploadSession(
                id=uuid4(),
                tenant_id=tenant_id,
                user_id=user_id,
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=1),
                filename=f"test{i}.pdf",
                content_type="application/pdf",
                total_size=1024000,
                chunk_size=None,
                total_chunks=None,
                upload_options=UploadRequest()
            )
            await progress_tracker.create_session(session)
            sessions.append(session)
        
        # Get resumable sessions
        resumable = await progress_tracker.get_resumable_sessions(tenant_id, user_id)
        
        # Should only return paused and failed sessions
        assert len(resumable) == 2
        statuses = {session.status for session in resumable}
        assert statuses == {UploadStatus.PAUSED, UploadStatus.FAILED}
    
    async def test_cleanup_session(self, progress_tracker, sample_session):
        """Test cleaning up a session."""
        await progress_tracker.create_session(sample_session)
        
        # Verify session exists
        session = await progress_tracker.get_session(sample_session.id)
        assert session is not None
        
        # Clean up session
        await progress_tracker.cleanup_session(sample_session.id)
        
        # Verify session is removed
        session = await progress_tracker.get_session(sample_session.id)
        assert session is None
    
    async def test_session_counts(self, progress_tracker):
        """Test session counting methods."""
        tenant_id = uuid4()
        other_tenant_id = uuid4()
        user_id = uuid4()
        
        # Create sessions for different tenants
        for i in range(3):
            session = UploadSession(
                id=uuid4(),
                tenant_id=tenant_id if i < 2 else other_tenant_id,
                user_id=user_id,
                status=UploadStatus.PENDING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=1),
                filename=f"test{i}.pdf",
                content_type="application/pdf",
                total_size=1024000,
                chunk_size=None,
                total_chunks=None,
                upload_options=UploadRequest()
            )
            await progress_tracker.create_session(session)
        
        # Test counts
        assert progress_tracker.get_session_count() == 3
        assert progress_tracker.get_tenant_session_count(tenant_id) == 2
        assert progress_tracker.get_tenant_session_count(other_tenant_id) == 1
    
    async def test_concurrent_progress_updates(self, progress_tracker, sample_session):
        """Test concurrent progress updates don't cause race conditions."""
        await progress_tracker.create_session(sample_session)
        
        # Create multiple concurrent update tasks
        async def update_progress(bytes_uploaded):
            await progress_tracker.update_progress(
                session_id=sample_session.id,
                bytes_uploaded=bytes_uploaded,
                status=UploadStatus.PROCESSING
            )
        
        # Run concurrent updates
        tasks = [update_progress(i * 1000) for i in range(1, 11)]
        await asyncio.gather(*tasks)
        
        # Verify final state is consistent
        session = await progress_tracker.get_session(sample_session.id)
        assert session.status == UploadStatus.PROCESSING
        assert session.bytes_uploaded in range(1000, 11000)  # One of the update values
