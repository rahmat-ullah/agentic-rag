"""
Upload service for handling file uploads and storage operations.

This module provides the core business logic for file uploads, including
validation, storage, duplicate detection, and progress tracking.
"""

import hashlib
import mimetypes
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from agentic_rag.api.models.upload import (
    ChunkUploadRequest,
    ChunkUploadResponse,
    DuplicateFileInfo,
    DuplicateStatistics,
    DocumentVersion,
    DocumentVersionHistory,
    FileMetadata,
    FileValidationError,
    UploadProgressUpdate,
    UploadRequest,
    UploadResponse,
    UploadSession,
    UploadStatus,
)
from agentic_rag.config import Settings
from agentic_rag.models.database import Document, DocumentKind
from agentic_rag.services.duplicate_detection import DuplicateDetectionService, DuplicateAction
from agentic_rag.services.storage import StorageService
from agentic_rag.services.security import SecurityService, SecurityValidationResult
from agentic_rag.services.progress_tracker import ProgressTracker
from agentic_rag.services.chunked_upload import ChunkedUploadService


class FileValidator:
    """File validation utilities."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.allowed_mime_types = set(settings.upload.allowed_mime_types)
        self.allowed_extensions = set(settings.upload.allowed_extensions)
        self.max_file_size = settings.upload.max_file_size
    
    def validate_file(self, file: UploadFile, content: bytes) -> List[FileValidationError]:
        """Validate uploaded file."""
        errors = []
        
        # Check file size
        if len(content) > self.max_file_size:
            errors.append(FileValidationError(
                code="FILE_TOO_LARGE",
                message=f"File size {len(content)} exceeds maximum allowed size {self.max_file_size}",
                field="file"
            ))
        
        # Check file extension
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in self.allowed_extensions:
            errors.append(FileValidationError(
                code="INVALID_EXTENSION",
                message=f"File extension '{file_ext}' is not allowed. Allowed: {', '.join(self.allowed_extensions)}",
                field="filename"
            ))
        
        # Check MIME type
        if file.content_type not in self.allowed_mime_types:
            errors.append(FileValidationError(
                code="INVALID_MIME_TYPE",
                message=f"MIME type '{file.content_type}' is not allowed. Allowed: {', '.join(self.allowed_mime_types)}",
                field="content_type"
            ))
        
        # Validate MIME type matches file content (magic bytes)
        detected_mime = self._detect_mime_type(content, file.filename or "")
        if detected_mime and detected_mime != file.content_type:
            errors.append(FileValidationError(
                code="MIME_TYPE_MISMATCH",
                message=f"Declared MIME type '{file.content_type}' does not match detected type '{detected_mime}'",
                field="content_type"
            ))
        
        return errors
    
    def _detect_mime_type(self, content: bytes, filename: str) -> Optional[str]:
        """Detect MIME type from file content and filename."""
        # Use python-magic if available, otherwise fall back to mimetypes
        try:
            import magic
            return magic.from_buffer(content, mime=True)
        except ImportError:
            # Fall back to filename-based detection
            mime_type, _ = mimetypes.guess_type(filename)
            return mime_type


class UploadService:
    """Service for handling file uploads."""

    def __init__(self, settings: Settings, storage_service: StorageService, progress_tracker: ProgressTracker):
        self.settings = settings
        self.storage_service = storage_service
        self.validator = FileValidator(settings)
        self.security_service = SecurityService(settings, storage_service)
        self.duplicate_service = DuplicateDetectionService(settings)
        self.progress_tracker = progress_tracker
        self.chunked_upload_service = ChunkedUploadService(settings, progress_tracker)
        self._upload_sessions: Dict[UUID, UploadSession] = {}
    
    async def create_upload_session(
        self,
        tenant_id: UUID,
        user_id: UUID,
        filename: str,
        content_type: str,
        file_size: int,
        upload_request: UploadRequest
    ) -> UploadSession:
        """Create a new upload session."""
        session_id = uuid4()
        expires_at = datetime.utcnow() + timedelta(seconds=self.settings.upload.upload_session_timeout)

        # Calculate chunks if chunked upload
        total_chunks = None
        chunk_size = None
        if upload_request.chunk_upload:
            chunk_size = self.settings.upload.max_chunk_size
            total_chunks = (file_size + chunk_size - 1) // chunk_size

        session = UploadSession(
            id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status=UploadStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            expires_at=expires_at,
            filename=filename,
            content_type=content_type,
            total_size=file_size,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            upload_options=upload_request
        )

        # Store in both local cache and progress tracker
        self._upload_sessions[session_id] = session
        await self.progress_tracker.create_session(session)

        return session

    async def upload_chunk(
        self,
        session_id: UUID,
        chunk_request: ChunkUploadRequest,
        chunk_file: UploadFile
    ) -> ChunkUploadResponse:
        """Upload a single chunk of a file."""
        return await self.chunked_upload_service.upload_chunk(session_id, chunk_request, chunk_file)

    async def get_upload_progress(self, session_id: UUID) -> UploadProgressUpdate:
        """Get current upload progress."""
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        return UploadProgressUpdate(
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

    async def pause_upload(self, session_id: UUID) -> bool:
        """Pause an upload session."""
        return await self.progress_tracker.pause_upload(session_id)

    async def resume_upload(self, session_id: UUID) -> bool:
        """Resume a paused upload session."""
        return await self.progress_tracker.resume_upload(session_id)

    async def get_resumable_uploads(self, tenant_id: UUID, user_id: UUID) -> List[UploadSession]:
        """Get resumable upload sessions for a user."""
        return await self.progress_tracker.get_resumable_sessions(tenant_id, user_id)

    async def get_chunk_status(self, session_id: UUID) -> dict:
        """Get the status of uploaded chunks for a chunked upload."""
        # Get session
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Get chunk status from chunked upload service
        chunk_status = await self.chunked_upload_service.get_upload_status(session_id)

        return {
            "upload_id": session_id,
            "session_status": session.status,
            "progress_percent": session.progress_percent,
            "bytes_uploaded": session.bytes_uploaded,
            "total_bytes": session.total_size,
            "chunks_uploaded": len(session.uploaded_chunks) if session.uploaded_chunks else 0,
            "total_chunks": session.total_chunks,
            "uploaded_chunks": session.uploaded_chunks or [],
            "missing_chunks": chunk_status.get("missing_chunks", []),
            "next_chunk_number": chunk_status.get("next_chunk_number"),
            "is_complete": chunk_status.get("is_complete", False),
            "message": session.message
        }

    async def complete_chunked_upload(self, session_id: UUID, db: Session) -> UploadResponse:
        """Complete a chunked upload by assembling chunks and processing the file."""
        # Get session
        session = await self.progress_tracker.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        try:
            # Update status to assembling
            await self.progress_tracker.update_progress(
                session_id=session_id,
                bytes_uploaded=session.bytes_uploaded,
                status=UploadStatus.ASSEMBLING,
                message="Assembling file from chunks..."
            )

            # Assemble file from chunks
            file_content = await self.chunked_upload_service.assemble_file(session_id)

            # Update status to processing
            await self.progress_tracker.update_progress(
                session_id=session_id,
                bytes_uploaded=len(file_content),
                status=UploadStatus.PROCESSING,
                message="Processing assembled file..."
            )

            # Process the assembled file
            response = await self._process_file_content(
                session=session,
                file_content=file_content,
                db=db
            )

            # Clean up chunks
            await self.chunked_upload_service.cleanup_chunks(session_id)

            return response

        except Exception as e:
            await self.progress_tracker.fail_upload(session_id, str(e))
            raise

    async def upload_file(
        self,
        tenant_id: UUID,
        user_id: UUID,
        file: UploadFile,
        upload_request: UploadRequest,
        db_session: Session
    ) -> UploadResponse:
        """Handle single file upload."""
        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Validate file
        validation_errors = self.validator.validate_file(file, content)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "File validation failed",
                    "errors": [error.dict() for error in validation_errors]
                }
            )
        
        # Create upload session
        session = await self.create_upload_session(
            tenant_id=tenant_id,
            user_id=user_id,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            file_size=len(content),
            upload_request=upload_request
        )
        
        try:
            # Update session status to processing with progress tracking
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.PROCESSING,
                message="Processing file..."
            )

            # Calculate SHA256 hash
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.PROCESSING,
                message="Calculating file hash..."
            )
            sha256_hash = hashlib.sha256(content).hexdigest()

            # Check for duplicates using enhanced duplicate detection service
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.PROCESSING,
                message="Checking for duplicates..."
            )
            duplicate_result = await self.duplicate_service.detect_duplicate(
                tenant_id=tenant_id,
                sha256_hash=sha256_hash,
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                file_size=len(content),
                db_session=db_session,
                overwrite_existing=upload_request.overwrite_existing,
                create_version=getattr(upload_request, 'create_version', False)
            )

            # Handle duplicate detection results
            if duplicate_result.is_duplicate and duplicate_result.action_taken == DuplicateAction.SKIPPED:
                await self.progress_tracker.complete_upload(
                    session_id=session.id,
                    document_id=duplicate_result.existing_document_id,
                    message=f"Duplicate file detected: {duplicate_result.existing_filename} (version {duplicate_result.existing_version})"
                )
                return UploadResponse(
                    upload_id=session.id,
                    status=UploadStatus.COMPLETE,
                    message=f"Duplicate file detected: {duplicate_result.existing_filename} (version {duplicate_result.existing_version})",
                    document_id=duplicate_result.existing_document_id,
                    progress_percent=100.0,
                    duplicate_info=DuplicateFileInfo(
                        is_duplicate=duplicate_result.is_duplicate,
                        existing_document_id=duplicate_result.existing_document_id,
                        existing_filename=duplicate_result.existing_filename,
                        existing_upload_date=duplicate_result.existing_upload_date,
                        existing_version=duplicate_result.existing_version,
                        new_version=duplicate_result.new_version,
                        sha256_hash=duplicate_result.sha256_hash,
                        action_taken=duplicate_result.action_taken.value,
                        duplicate_count=duplicate_result.duplicate_count,
                        tenant_duplicate_count=duplicate_result.tenant_duplicate_count
                    )
                )

            # Comprehensive security validation
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.VALIDATING,
                message="Performing security validation..."
            )
            security_result = await self.security_service.validate_file_security(
                file=file,
                content=content,
                tenant_id=tenant_id,
                user_id=user_id,
                ip_address=None,  # Would be extracted from request context
                user_agent=None   # Would be extracted from request context
            )

            # Block upload if security validation fails
            if not security_result.is_safe:
                violation_messages = [v.get("message", "Unknown violation") for v in security_result.violations]
                error_msg = f"Security validation failed: {security_result.threat_level.value}"

                await self.progress_tracker.fail_upload(
                    session_id=session.id,
                    error=error_msg
                )

                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "File rejected due to security violations",
                        "threat_level": security_result.threat_level.value,
                        "violations": violation_messages,
                        "quarantine_id": security_result.quarantine_id
                    }
                )

            # Store file
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.STORING,
                message="Storing file to object storage..."
            )
            storage_path = await self._store_file(tenant_id, sha256_hash, content, file.filename or "unknown")

            # Create file metadata
            file_metadata = FileMetadata(
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                size=len(content),
                sha256=sha256_hash,
                extension=Path(file.filename or "").suffix.lower()
            )

            # Create document record with version handling
            await self.progress_tracker.update_progress(
                session_id=session.id,
                bytes_uploaded=len(content),
                status=UploadStatus.PROCESSING,
                message="Creating document record..."
            )
            document_id = await self._create_document_record(
                tenant_id=tenant_id,
                user_id=user_id,
                file_metadata=file_metadata,
                storage_path=storage_path,
                upload_request=upload_request,
                duplicate_result=duplicate_result,
                db_session=db_session
            )

            # Complete upload
            await self.progress_tracker.complete_upload(
                session_id=session.id,
                document_id=document_id,
                message="File uploaded successfully"
            )
            
            return UploadResponse(
                upload_id=session.id,
                status=UploadStatus.COMPLETE,
                message="File uploaded successfully",
                file_metadata=file_metadata,
                document_id=document_id,
                progress_percent=100.0
            )
            
        except Exception as e:
            await self.progress_tracker.fail_upload(session.id, str(e))
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Duplicate detection is now handled by DuplicateDetectionService

    async def get_duplicate_statistics(self, tenant_id: UUID, db_session: Session) -> DuplicateStatistics:
        """Get duplicate statistics for a tenant."""
        stats = await self.duplicate_service.get_duplicate_statistics(tenant_id, db_session)
        return DuplicateStatistics(
            tenant_id=stats.tenant_id,
            total_documents=stats.total_documents,
            unique_documents=stats.unique_documents,
            duplicate_documents=stats.duplicate_documents,
            duplicate_percentage=stats.duplicate_percentage,
            most_duplicated_hash=stats.most_duplicated_hash,
            most_duplicated_count=stats.most_duplicated_count,
            recent_duplicates=stats.recent_duplicates,
            storage_saved_bytes=stats.storage_saved_bytes
        )

    async def get_document_version_history(
        self,
        tenant_id: UUID,
        sha256_hash: str,
        db_session: Session
    ) -> DocumentVersionHistory:
        """Get version history for a document by SHA256 hash."""
        versions = await self.duplicate_service.get_document_versions(tenant_id, sha256_hash, db_session)

        if not versions:
            raise HTTPException(status_code=404, detail="Document not found")

        version_list = [
            DocumentVersion(
                document_id=doc.id,
                version=doc.version,
                title=doc.title,
                created_at=doc.created_at,
                created_by=doc.created_by,
                file_size=None  # Would need to be stored in metadata
            )
            for doc in versions
        ]

        return DocumentVersionHistory(
            sha256_hash=sha256_hash,
            total_versions=len(versions),
            versions=version_list,
            latest_version=max(v.version for v in versions),
            first_upload=min(v.created_at for v in versions),
            last_upload=max(v.created_at for v in versions)
        )


    async def _store_file(self, tenant_id: UUID, sha256_hash: str, content: bytes, filename: str) -> str:
        """Store file in object storage with secure naming and metadata."""
        # Generate secure storage path using the storage service
        storage_path = self.storage_service.generate_secure_object_name(
            tenant_id=tenant_id,
            file_hash=sha256_hash,
            filename=filename
        )

        # Prepare metadata
        metadata = {
            "tenant_id": str(tenant_id),
            "original_filename": filename,
            "sha256": sha256_hash,
            "upload_timestamp": datetime.utcnow().isoformat()
        }

        # Store file using enhanced storage service
        storage_uri = await self.storage_service.store_file(
            object_name=storage_path,
            content=content,
            metadata=metadata
        )

        return storage_path
    
    async def _create_document_record(
        self,
        tenant_id: UUID,
        user_id: UUID,
        file_metadata: FileMetadata,
        storage_path: str,
        upload_request: UploadRequest,
        duplicate_result,
        db_session: Session
    ) -> UUID:
        """Create document record in database with version handling."""
        # Determine version number based on duplicate detection result
        version_number = 1
        document_id = uuid4()

        if duplicate_result.is_duplicate:
            if duplicate_result.action_taken == DuplicateAction.OVERWRITTEN:
                # Update existing document
                existing_doc = db_session.query(Document).filter(
                    Document.id == duplicate_result.existing_document_id
                ).first()
                if existing_doc:
                    existing_doc.title = upload_request.title or file_metadata.filename
                    existing_doc.source_uri = f"s3://{self.settings.storage.minio_bucket_documents}/{storage_path}"
                    existing_doc.pages = file_metadata.page_count
                    existing_doc.updated_at = datetime.utcnow()
                    db_session.commit()
                    return existing_doc.id
            elif duplicate_result.action_taken == DuplicateAction.VERSIONED:
                # Create new version
                version_number = duplicate_result.new_version or 1

        # Create new document record
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            kind=DocumentKind(upload_request.kind) if upload_request.kind else DocumentKind.OTHER,
            title=upload_request.title or file_metadata.filename,
            source_uri=f"s3://{self.settings.storage.minio_bucket_documents}/{storage_path}",
            sha256=file_metadata.sha256,
            version=version_number,
            pages=file_metadata.page_count,
            created_by=user_id
        )

        db_session.add(document)
        db_session.commit()
        return document.id
