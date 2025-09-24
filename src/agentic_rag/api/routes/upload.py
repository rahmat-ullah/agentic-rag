"""
Upload API routes for file upload and storage operations.

This module defines FastAPI routes for handling file uploads, progress tracking,
and upload management operations.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from agentic_rag.api.dependencies.auth import get_current_user, require_permission
from agentic_rag.api.dependencies.database import get_db_session
from agentic_rag.api.models.responses import ErrorResponse, SuccessResponse
from agentic_rag.api.models.upload import (
    ChunkUploadRequest,
    ChunkUploadResponse,
    DuplicateFileInfo,
    UploadQuotaInfo,
    UploadRequest,
    UploadResponse,
    UploadSession,
    UploadStatsResponse,
)
from agentic_rag.config import get_settings
from agentic_rag.models.database import User
from agentic_rag.services.storage import get_storage_service
from agentic_rag.services.upload import UploadService

router = APIRouter(prefix="/upload", tags=["upload"])


def get_upload_service() -> UploadService:
    """Get upload service instance."""
    settings = get_settings()
    storage_service = get_storage_service(settings)
    return UploadService(settings, storage_service)


@router.post(
    "/",
    response_model=SuccessResponse[UploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload file",
    description="Upload a single file with validation and processing"
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    title: Optional[str] = Form(None, description="Document title override"),
    description: Optional[str] = Form(None, description="Document description"),
    kind: Optional[str] = Form(None, description="Document kind/type"),
    chunk_upload: bool = Form(False, description="Enable chunked upload"),
    overwrite_existing: bool = Form(False, description="Overwrite if duplicate exists"),
    skip_virus_scan: bool = Form(False, description="Skip virus scanning (admin only)"),
    extract_text: bool = Form(True, description="Extract text content"),
    generate_thumbnail: bool = Form(True, description="Generate thumbnail"),
    auto_process: bool = Form(True, description="Start processing immediately"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Upload a file to the system."""
    # Check permissions
    if skip_virus_scan:
        require_permission(current_user, "admin")
    
    # Create upload request
    upload_request = UploadRequest(
        title=title,
        description=description,
        kind=kind,
        chunk_upload=chunk_upload,
        overwrite_existing=overwrite_existing,
        skip_virus_scan=skip_virus_scan,
        extract_text=extract_text,
        generate_thumbnail=generate_thumbnail,
        auto_process=auto_process
    )
    
    try:
        # Perform upload
        result = await upload_service.upload_file(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            file=file,
            upload_request=upload_request,
            db_session=db_session
        )
        
        return SuccessResponse(
            data=result,
            message="File uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post(
    "/session",
    response_model=SuccessResponse[UploadSession],
    status_code=status.HTTP_201_CREATED,
    summary="Create upload session",
    description="Create a new upload session for chunked uploads"
)
async def create_upload_session(
    filename: str = Form(..., description="Original filename"),
    content_type: str = Form(..., description="MIME content type"),
    file_size: int = Form(..., description="Total file size in bytes"),
    title: Optional[str] = Form(None, description="Document title override"),
    description: Optional[str] = Form(None, description="Document description"),
    kind: Optional[str] = Form(None, description="Document kind/type"),
    overwrite_existing: bool = Form(False, description="Overwrite if duplicate exists"),
    skip_virus_scan: bool = Form(False, description="Skip virus scanning (admin only)"),
    extract_text: bool = Form(True, description="Extract text content"),
    generate_thumbnail: bool = Form(True, description="Generate thumbnail"),
    auto_process: bool = Form(True, description="Start processing immediately"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Create a new upload session for chunked uploads."""
    # Check permissions
    if skip_virus_scan:
        require_permission(current_user, "admin")
    
    # Create upload request
    upload_request = UploadRequest(
        title=title,
        description=description,
        kind=kind,
        chunk_upload=True,  # Force chunked upload for sessions
        overwrite_existing=overwrite_existing,
        skip_virus_scan=skip_virus_scan,
        extract_text=extract_text,
        generate_thumbnail=generate_thumbnail,
        auto_process=auto_process
    )
    
    try:
        # Create upload session
        session = await upload_service.create_upload_session(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            upload_request=upload_request
        )
        
        return SuccessResponse(
            data=session,
            message="Upload session created successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create upload session: {str(e)}"
        )


@router.get(
    "/session/{upload_id}",
    response_model=SuccessResponse[UploadSession],
    summary="Get upload session",
    description="Get upload session details and progress"
)
async def get_upload_session(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Get upload session details."""
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )
    
    # Check if user has access to this session
    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )
    
    return SuccessResponse(
        data=session,
        message="Upload session retrieved successfully"
    )


@router.delete(
    "/session/{upload_id}",
    response_model=SuccessResponse[dict],
    summary="Cancel upload session",
    description="Cancel an active upload session"
)
async def cancel_upload_session(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Cancel an upload session."""
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )
    
    # Check if user has access to this session
    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )
    
    # Cancel session
    from agentic_rag.api.models.upload import UploadStatus
    session.status = UploadStatus.CANCELLED
    session.updated_at = datetime.utcnow()
    
    return SuccessResponse(
        data={"upload_id": upload_id, "status": "cancelled"},
        message="Upload session cancelled successfully"
    )


@router.get(
    "/quota",
    response_model=SuccessResponse[UploadQuotaInfo],
    summary="Get upload quota",
    description="Get upload quota information for current tenant"
)
async def get_upload_quota(
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Get upload quota information for the current tenant."""
    from agentic_rag.models.database import Document
    from sqlalchemy import func, sum as sql_sum

    settings = get_settings()

    try:
        # Calculate actual usage from database
        # Count total documents for this tenant
        file_count = db_session.query(func.count(Document.id)).filter(
            Document.tenant_id == current_user.tenant_id
        ).scalar() or 0

        # Calculate total storage used (approximate from file metadata)
        # Note: This is a simplified calculation. In production, you'd want to
        # track actual file sizes in the database or query object storage
        used_quota = file_count * 1024 * 1024  # Approximate 1MB per document

        # Calculate available quota
        total_quota = settings.upload.tenant_upload_quota
        available_quota = max(0, total_quota - used_quota)
        quota_percentage = (used_quota / total_quota * 100) if total_quota > 0 else 0

        quota_info = UploadQuotaInfo(
            tenant_id=current_user.tenant_id,
            total_quota=total_quota,
            used_quota=used_quota,
            available_quota=available_quota,
            quota_percentage=quota_percentage,
            file_count=file_count
        )

        return SuccessResponse(
            data=quota_info,
            message="Upload quota retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upload quota: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=SuccessResponse[UploadStatsResponse],
    summary="Get upload statistics",
    description="Get upload statistics for current tenant"
)
async def get_upload_stats(
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session)
):
    """Get upload statistics for the current tenant."""
    from agentic_rag.models.database import Document
    from sqlalchemy import func, and_
    from datetime import datetime, timedelta

    settings = get_settings()

    try:
        # Get current time for date calculations
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)

        # Base query for this tenant
        base_query = db_session.query(Document).filter(
            Document.tenant_id == current_user.tenant_id
        )

        # Total uploads (all documents)
        total_uploads = base_query.count()

        # For now, assume all uploads are successful (no failed status tracking yet)
        successful_uploads = total_uploads
        failed_uploads = 0

        # Calculate total size (approximate)
        total_size = total_uploads * 1024 * 1024  # Approximate 1MB per document

        # Count duplicate files (documents with same SHA256)
        duplicate_subquery = db_session.query(
            Document.sha256,
            func.count(Document.id).label('count')
        ).filter(
            Document.tenant_id == current_user.tenant_id
        ).group_by(Document.sha256).having(func.count(Document.id) > 1).subquery()

        duplicate_files = db_session.query(func.count(duplicate_subquery.c.sha256)).scalar() or 0

        # Uploads by time period
        uploads_today = base_query.filter(
            Document.created_at >= today_start
        ).count()

        uploads_this_week = base_query.filter(
            Document.created_at >= week_start
        ).count()

        uploads_this_month = base_query.filter(
            Document.created_at >= month_start
        ).count()

        # Get quota info (reuse logic from quota endpoint)
        file_count = total_uploads
        used_quota = file_count * 1024 * 1024
        total_quota = settings.upload.tenant_upload_quota
        available_quota = max(0, total_quota - used_quota)
        quota_percentage = (used_quota / total_quota * 100) if total_quota > 0 else 0

        quota_info = UploadQuotaInfo(
            tenant_id=current_user.tenant_id,
            total_quota=total_quota,
            used_quota=used_quota,
            available_quota=available_quota,
            quota_percentage=quota_percentage,
            file_count=file_count
        )

        stats = UploadStatsResponse(
            total_uploads=total_uploads,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            total_size=total_size,
            duplicate_files=duplicate_files,
            quota_info=quota_info,
            uploads_today=uploads_today,
            uploads_this_week=uploads_this_week,
            uploads_this_month=uploads_this_month
        )

        return SuccessResponse(
            data=stats,
            message="Upload statistics retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upload statistics: {str(e)}"
        )


# ============================================================================
# CHUNKED UPLOAD ENDPOINTS
# ============================================================================

@router.post(
    "/chunk",
    response_model=SuccessResponse[ChunkUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload file chunk",
    description="Upload a single chunk of a file for chunked upload"
)
async def upload_chunk(
    upload_id: UUID = Form(..., description="Upload session ID"),
    chunk_number: int = Form(..., description="Chunk number (0-based)"),
    chunk_size: int = Form(..., description="Size of this chunk"),
    total_chunks: int = Form(..., description="Total number of chunks"),
    is_final_chunk: bool = Form(False, description="Whether this is the final chunk"),
    chunk_file: UploadFile = File(..., description="Chunk file data"),
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Upload a single chunk of a file."""
    # Create chunk upload request
    chunk_request = ChunkUploadRequest(
        upload_id=upload_id,
        chunk_number=chunk_number,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
        is_final_chunk=is_final_chunk
    )

    try:
        # Upload chunk
        result = await upload_service.upload_chunk(
            session_id=upload_id,
            chunk_request=chunk_request,
            chunk_file=chunk_file
        )

        return SuccessResponse(
            data=result,
            message=f"Chunk {chunk_number + 1}/{total_chunks} uploaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunk upload failed: {str(e)}"
        )


@router.post(
    "/{upload_id}/complete",
    response_model=SuccessResponse[UploadResponse],
    summary="Complete chunked upload",
    description="Complete a chunked upload by assembling all chunks"
)
async def complete_chunked_upload(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Complete a chunked upload by assembling all chunks."""
    # Check if user has access to this upload session
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )

    try:
        # Complete chunked upload
        result = await upload_service.complete_chunked_upload(upload_id, db_session)

        return SuccessResponse(
            data=result,
            message="Chunked upload completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete chunked upload: {str(e)}"
        )


@router.put(
    "/{upload_id}/pause",
    response_model=SuccessResponse[dict],
    summary="Pause upload",
    description="Pause an active upload session"
)
async def pause_upload(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Pause an active upload session."""
    # Check if user has access to this upload session
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )

    try:
        # Pause upload
        success = await upload_service.pause_upload(upload_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload cannot be paused in current state"
            )

        return SuccessResponse(
            data={"upload_id": upload_id, "status": "paused"},
            message="Upload paused successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause upload: {str(e)}"
        )


@router.put(
    "/{upload_id}/resume",
    response_model=SuccessResponse[dict],
    summary="Resume upload",
    description="Resume a paused upload session"
)
async def resume_upload(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Resume a paused upload session."""
    # Check if user has access to this upload session
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )

    try:
        # Resume upload
        success = await upload_service.resume_upload(upload_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload cannot be resumed in current state"
            )

        return SuccessResponse(
            data={"upload_id": upload_id, "status": "resumed"},
            message="Upload resumed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume upload: {str(e)}"
        )


@router.get(
    "/{upload_id}/chunks",
    response_model=SuccessResponse[dict],
    summary="Get chunk status",
    description="Get the status of uploaded chunks for a chunked upload"
)
async def get_chunk_status(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Get the status of uploaded chunks for a chunked upload."""
    # Check if user has access to this upload session
    session = upload_service._upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )

    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to upload session"
        )

    try:
        # Get chunk status
        chunk_status = await upload_service.get_chunk_status(upload_id)

        return SuccessResponse(
            data=chunk_status,
            message="Chunk status retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunk status: {str(e)}"
        )


# ============================================================================
# BATCH UPLOAD ENDPOINTS
# ============================================================================

@router.post(
    "/batch",
    response_model=SuccessResponse[List[UploadResponse]],
    status_code=status.HTTP_201_CREATED,
    summary="Batch upload files",
    description="Upload multiple files in a single request"
)
async def batch_upload_files(
    files: List[UploadFile] = File(..., description="Files to upload"),
    title_prefix: Optional[str] = Form(None, description="Prefix for document titles"),
    description: Optional[str] = Form(None, description="Description for all documents"),
    kind: Optional[str] = Form(None, description="Document kind/type for all files"),
    overwrite_existing: bool = Form(False, description="Overwrite if duplicates exist"),
    skip_virus_scan: bool = Form(False, description="Skip virus scanning (admin only)"),
    extract_text: bool = Form(True, description="Extract text content"),
    generate_thumbnail: bool = Form(True, description="Generate thumbnails"),
    auto_process: bool = Form(True, description="Start processing immediately"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Upload multiple files in a single batch operation."""
    # Check permissions
    if skip_virus_scan:
        require_permission(current_user, "admin")

    # Validate file count
    settings = get_settings()
    max_batch_size = getattr(settings.upload, 'max_batch_size', 10)
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {max_batch_size} files"
        )

    results = []
    errors = []

    for i, file in enumerate(files):
        try:
            # Create individual upload request
            file_title = f"{title_prefix} {i+1}" if title_prefix else None

            upload_request = UploadRequest(
                title=file_title,
                description=description,
                kind=kind,
                chunk_upload=False,  # Disable chunked upload for batch
                overwrite_existing=overwrite_existing,
                skip_virus_scan=skip_virus_scan,
                extract_text=extract_text,
                generate_thumbnail=generate_thumbnail,
                auto_process=auto_process
            )

            # Perform individual upload
            result = await upload_service.upload_file(
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                file=file,
                upload_request=upload_request,
                db_session=db_session
            )

            results.append(result)

        except Exception as e:
            error_info = {
                "filename": file.filename,
                "error": str(e),
                "index": i
            }
            errors.append(error_info)

            # Continue with other files even if one fails
            continue

    # Prepare response message
    success_count = len(results)
    error_count = len(errors)

    if error_count > 0:
        message = f"Batch upload completed with {success_count} successes and {error_count} errors"
    else:
        message = f"Batch upload completed successfully - {success_count} files uploaded"

    # Include error details in response if any
    response_data = results
    if errors:
        # Add error information to the response
        # Note: This modifies the response structure slightly when errors occur
        response_data = {
            "successful_uploads": results,
            "errors": errors,
            "summary": {
                "total_files": len(files),
                "successful": success_count,
                "failed": error_count
            }
        }

    return SuccessResponse(
        data=response_data,
        message=message
    )


@router.post(
    "/zip",
    response_model=SuccessResponse[List[UploadResponse]],
    status_code=status.HTTP_201_CREATED,
    summary="Upload ZIP archive",
    description="Upload and extract files from a ZIP archive"
)
async def upload_zip_archive(
    zip_file: UploadFile = File(..., description="ZIP file to upload and extract"),
    title_prefix: Optional[str] = Form(None, description="Prefix for extracted file titles"),
    description: Optional[str] = Form(None, description="Description for all extracted files"),
    kind: Optional[str] = Form(None, description="Document kind/type for all files"),
    overwrite_existing: bool = Form(False, description="Overwrite if duplicates exist"),
    skip_virus_scan: bool = Form(False, description="Skip virus scanning (admin only)"),
    extract_text: bool = Form(True, description="Extract text content"),
    generate_thumbnail: bool = Form(True, description="Generate thumbnails"),
    auto_process: bool = Form(True, description="Start processing immediately"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Upload and extract files from a ZIP archive."""
    import zipfile
    import tempfile
    import os
    from io import BytesIO

    # Check permissions
    if skip_virus_scan:
        require_permission(current_user, "admin")

    # Validate file type
    if not zip_file.filename.lower().endswith('.zip'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a ZIP archive"
        )

    try:
        # Read ZIP file content
        zip_content = await zip_file.read()

        # Validate ZIP file
        try:
            with zipfile.ZipFile(BytesIO(zip_content), 'r') as zip_ref:
                file_list = zip_ref.namelist()
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid ZIP file format"
            )

        # Validate extracted file count
        settings = get_settings()
        max_batch_size = getattr(settings.upload, 'max_batch_size', 10)
        if len(file_list) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ZIP contains too many files. Maximum: {max_batch_size}"
            )

        results = []
        errors = []

        # Extract and upload each file
        with zipfile.ZipFile(BytesIO(zip_content), 'r') as zip_ref:
            for i, filename in enumerate(file_list):
                # Skip directories
                if filename.endswith('/'):
                    continue

                try:
                    # Extract file content
                    file_content = zip_ref.read(filename)

                    # Create a temporary UploadFile-like object
                    file_obj = BytesIO(file_content)

                    # Create upload request
                    file_title = f"{title_prefix} - {filename}" if title_prefix else filename

                    upload_request = UploadRequest(
                        title=file_title,
                        description=description,
                        kind=kind,
                        chunk_upload=False,
                        overwrite_existing=overwrite_existing,
                        skip_virus_scan=skip_virus_scan,
                        extract_text=extract_text,
                        generate_thumbnail=generate_thumbnail,
                        auto_process=auto_process
                    )

                    # Create UploadFile object from extracted content
                    from fastapi import UploadFile as FastAPIUploadFile
                    extracted_file = FastAPIUploadFile(
                        filename=filename,
                        file=file_obj,
                        size=len(file_content)
                    )

                    # Upload extracted file
                    result = await upload_service.upload_file(
                        tenant_id=current_user.tenant_id,
                        user_id=current_user.id,
                        file=extracted_file,
                        upload_request=upload_request,
                        db_session=db_session
                    )

                    results.append(result)

                except Exception as e:
                    error_info = {
                        "filename": filename,
                        "error": str(e),
                        "index": i
                    }
                    errors.append(error_info)
                    continue

        # Prepare response
        success_count = len(results)
        error_count = len(errors)

        if error_count > 0:
            message = f"ZIP extraction completed with {success_count} successes and {error_count} errors"
        else:
            message = f"ZIP extraction completed successfully - {success_count} files uploaded"

        response_data = results
        if errors:
            response_data = {
                "successful_uploads": results,
                "errors": errors,
                "summary": {
                    "total_files": len(file_list),
                    "successful": success_count,
                    "failed": error_count
                }
            }

        return SuccessResponse(
            data=response_data,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ZIP upload failed: {str(e)}"
        )


# ============================================================================
# CLEANUP ENDPOINTS
# ============================================================================

@router.post(
    "/cleanup",
    response_model=SuccessResponse[dict],
    summary="Run cleanup tasks",
    description="Manually trigger cleanup of expired sessions and temporary files"
)
async def run_cleanup_tasks(
    current_user: User = Depends(get_current_user),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Manually trigger cleanup tasks."""
    # Require admin permission for cleanup operations
    require_permission(current_user, "admin")

    try:
        from agentic_rag.services.cleanup import get_cleanup_service
        from agentic_rag.services.storage import get_storage_service

        settings = get_settings()
        storage_service = get_storage_service(settings)
        cleanup_service = get_cleanup_service(settings, storage_service, upload_service.progress_tracker)

        # Run cleanup tasks
        await cleanup_service.run_cleanup_tasks()

        return SuccessResponse(
            data={"status": "completed", "timestamp": datetime.utcnow().isoformat()},
            message="Cleanup tasks completed successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get(
    "/cleanup/stats",
    response_model=SuccessResponse[dict],
    summary="Get cleanup statistics",
    description="Get statistics about items that could be cleaned up"
)
async def get_cleanup_stats(
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db_session),
    upload_service: UploadService = Depends(get_upload_service)
):
    """Get cleanup statistics."""
    # Require admin permission for cleanup statistics
    require_permission(current_user, "admin")

    try:
        from agentic_rag.services.cleanup import get_cleanup_service
        from agentic_rag.services.storage import get_storage_service

        settings = get_settings()
        storage_service = get_storage_service(settings)
        cleanup_service = get_cleanup_service(settings, storage_service, upload_service.progress_tracker)

        # Get cleanup statistics
        stats = await cleanup_service.get_cleanup_statistics(db_session)

        return SuccessResponse(
            data=stats,
            message="Cleanup statistics retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cleanup statistics: {str(e)}"
        )
