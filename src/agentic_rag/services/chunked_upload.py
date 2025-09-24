"""
Chunked Upload Service for handling large file uploads.

This module provides chunked upload functionality with resumption support,
progress tracking, and integrity validation.
"""

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, BinaryIO
from uuid import UUID

from fastapi import HTTPException, UploadFile

from agentic_rag.api.models.upload import (
    ChunkUploadRequest,
    ChunkUploadResponse,
    UploadSession,
    UploadStatus
)
from agentic_rag.config import Settings
from agentic_rag.services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class ChunkInfo:
    """Information about an uploaded chunk."""
    
    def __init__(self, chunk_number: int, size: int, checksum: str, file_path: str):
        self.chunk_number = chunk_number
        self.size = size
        self.checksum = checksum
        self.file_path = file_path
        self.uploaded_at = None


class ChunkedUploadService:
    """Service for handling chunked file uploads."""
    
    def __init__(self, settings: Settings, progress_tracker: ProgressTracker):
        self.settings = settings
        self.progress_tracker = progress_tracker
        
        # Chunk storage: session_id -> {chunk_number -> ChunkInfo}
        self.chunks: Dict[UUID, Dict[int, ChunkInfo]] = {}
        
        # Temporary directory for chunk storage
        self.temp_dir = Path(tempfile.gettempdir()) / "agentic_rag_chunks"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def upload_chunk(
        self,
        session_id: UUID,
        chunk_request: ChunkUploadRequest,
        chunk_file: UploadFile
    ) -> ChunkUploadResponse:
        """
        Upload a single chunk of a file.
        
        Args:
            session_id: Upload session ID
            chunk_request: Chunk upload request details
            chunk_file: Chunk file data
            
        Returns:
            ChunkUploadResponse with upload status
        """
        # Get upload session
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        # Validate session status
        if session.status not in [UploadStatus.PENDING, UploadStatus.PROCESSING, UploadStatus.PAUSED]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot upload chunk for session with status {session.status}"
            )
        
        # Validate chunk request
        if chunk_request.chunk_number >= chunk_request.total_chunks:
            raise HTTPException(status_code=400, detail="Invalid chunk number")
        
        if chunk_request.total_chunks != session.total_chunks:
            raise HTTPException(status_code=400, detail="Total chunks mismatch")
        
        try:
            # Read chunk data
            chunk_data = await chunk_file.read()
            
            # Validate chunk size
            if len(chunk_data) != chunk_request.chunk_size:
                raise HTTPException(status_code=400, detail="Chunk size mismatch")
            
            # Calculate chunk checksum
            chunk_checksum = hashlib.sha256(chunk_data).hexdigest()
            
            # Store chunk to temporary file
            chunk_file_path = self._get_chunk_file_path(session_id, chunk_request.chunk_number)
            chunk_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(chunk_file_path, 'wb') as f:
                f.write(chunk_data)
            
            # Store chunk info
            if session_id not in self.chunks:
                self.chunks[session_id] = {}
            
            self.chunks[session_id][chunk_request.chunk_number] = ChunkInfo(
                chunk_number=chunk_request.chunk_number,
                size=len(chunk_data),
                checksum=chunk_checksum,
                file_path=str(chunk_file_path)
            )
            
            # Update session progress
            total_uploaded = sum(chunk.size for chunk in self.chunks[session_id].values())
            await self.progress_tracker.update_progress(
                session_id=session_id,
                bytes_uploaded=total_uploaded,
                status=UploadStatus.PROCESSING,
                current_chunk=chunk_request.chunk_number,
                message=f"Uploaded chunk {chunk_request.chunk_number + 1}/{chunk_request.total_chunks}"
            )
            
            # Check if all chunks are uploaded
            if len(self.chunks[session_id]) == chunk_request.total_chunks:
                # All chunks uploaded, ready for assembly
                await self.progress_tracker.update_progress(
                    session_id=session_id,
                    bytes_uploaded=total_uploaded,
                    status=UploadStatus.ASSEMBLING,
                    message="All chunks uploaded, assembling file..."
                )
            
            logger.info(f"Uploaded chunk {chunk_request.chunk_number} for session {session_id}")
            
            return ChunkUploadResponse(
                upload_id=session_id,
                chunk_number=chunk_request.chunk_number,
                status=session.status,
                bytes_uploaded=total_uploaded,
                total_bytes=session.total_size,
                chunks_uploaded=len(self.chunks[session_id]),
                total_chunks=chunk_request.total_chunks,
                is_complete=len(self.chunks[session_id]) == chunk_request.total_chunks
            )
            
        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk_request.chunk_number} for session {session_id}: {e}")
            await self.progress_tracker.fail_upload(
                session_id=session_id,
                error=f"Chunk upload failed: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=f"Chunk upload failed: {str(e)}")
    
    async def assemble_file(self, session_id: UUID) -> bytes:
        """
        Assemble all chunks into the complete file.
        
        Args:
            session_id: Upload session ID
            
        Returns:
            Complete file content as bytes
        """
        if session_id not in self.chunks:
            raise HTTPException(status_code=404, detail="No chunks found for session")
        
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        chunks = self.chunks[session_id]
        
        # Verify all chunks are present
        expected_chunks = set(range(session.total_chunks))
        uploaded_chunks = set(chunks.keys())
        
        if expected_chunks != uploaded_chunks:
            missing_chunks = expected_chunks - uploaded_chunks
            raise HTTPException(
                status_code=400, 
                detail=f"Missing chunks: {sorted(missing_chunks)}"
            )
        
        try:
            # Assemble file in correct order
            assembled_data = bytearray()
            
            for chunk_number in sorted(chunks.keys()):
                chunk_info = chunks[chunk_number]
                
                # Read chunk data
                with open(chunk_info.file_path, 'rb') as f:
                    chunk_data = f.read()
                
                # Verify chunk integrity
                chunk_checksum = hashlib.sha256(chunk_data).hexdigest()
                if chunk_checksum != chunk_info.checksum:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Chunk {chunk_number} integrity check failed"
                    )
                
                assembled_data.extend(chunk_data)
                
                # Update progress
                progress = (chunk_number + 1) / len(chunks) * 100
                await self.progress_tracker.update_progress(
                    session_id=session_id,
                    bytes_uploaded=len(assembled_data),
                    status=UploadStatus.ASSEMBLING,
                    message=f"Assembling file... {progress:.1f}%"
                )
            
            # Verify total file size
            if len(assembled_data) != session.total_size:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Assembled file size mismatch: expected {session.total_size}, got {len(assembled_data)}"
                )
            
            logger.info(f"Successfully assembled file for session {session_id}")
            return bytes(assembled_data)
            
        except Exception as e:
            logger.error(f"Failed to assemble file for session {session_id}: {e}")
            await self.progress_tracker.fail_upload(
                session_id=session_id,
                error=f"File assembly failed: {str(e)}"
            )
            raise
    
    async def get_upload_status(self, session_id: UUID) -> Dict:
        """Get detailed upload status for a chunked upload."""
        if session_id not in self.chunks:
            return {"chunks_uploaded": 0, "missing_chunks": []}
        
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        chunks = self.chunks[session_id]
        uploaded_chunks = set(chunks.keys())
        expected_chunks = set(range(session.total_chunks))
        missing_chunks = sorted(expected_chunks - uploaded_chunks)
        
        return {
            "chunks_uploaded": len(uploaded_chunks),
            "total_chunks": session.total_chunks,
            "missing_chunks": missing_chunks,
            "uploaded_chunks": sorted(uploaded_chunks),
            "bytes_uploaded": sum(chunk.size for chunk in chunks.values()),
            "total_bytes": session.total_size
        }
    
    async def cleanup_chunks(self, session_id: UUID):
        """Clean up temporary chunk files for a session."""
        if session_id not in self.chunks:
            return
        
        chunks = self.chunks[session_id]
        
        # Delete chunk files
        for chunk_info in chunks.values():
            try:
                chunk_path = Path(chunk_info.file_path)
                if chunk_path.exists():
                    chunk_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete chunk file {chunk_info.file_path}: {e}")
        
        # Remove from memory
        del self.chunks[session_id]
        
        # Clean up session directory
        session_dir = self.temp_dir / str(session_id)
        if session_dir.exists():
            try:
                session_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to remove session directory {session_dir}: {e}")
        
        logger.debug(f"Cleaned up chunks for session {session_id}")
    
    def _get_chunk_file_path(self, session_id: UUID, chunk_number: int) -> Path:
        """Get file path for storing a chunk."""
        return self.temp_dir / str(session_id) / f"chunk_{chunk_number:06d}.bin"
    
    async def resume_upload(self, session_id: UUID) -> Dict:
        """Resume a chunked upload by returning the current status."""
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        # Resume the session
        await self.progress_tracker.resume_upload(session_id)
        
        # Return current status
        return await self.get_upload_status(session_id)
