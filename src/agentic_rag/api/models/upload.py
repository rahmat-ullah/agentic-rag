"""
Upload API models for file upload and storage operations.

This module defines Pydantic models for file upload requests, responses,
and related data structures used in the upload API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UploadStatus(str, Enum):
    """Upload status enumeration."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    VALIDATING = "validating"
    STORING = "storing"
    ASSEMBLING = "assembling"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileValidationError(BaseModel):
    """File validation error details."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")


class UploadProgressUpdate(BaseModel):
    """Upload progress update for real-time tracking."""
    
    upload_id: UUID = Field(..., description="Upload session ID")
    status: UploadStatus = Field(..., description="Current upload status")
    progress_percent: float = Field(..., ge=0, le=100, description="Upload progress percentage")
    bytes_uploaded: int = Field(..., ge=0, description="Number of bytes uploaded")
    total_bytes: int = Field(..., ge=0, description="Total file size in bytes")
    current_chunk: Optional[int] = Field(None, description="Current chunk number")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[FileValidationError] = Field(None, description="Error details if failed")


class FileMetadata(BaseModel):
    """File metadata extracted during upload."""
    
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    size: int = Field(..., ge=0, description="File size in bytes")
    sha256: str = Field(..., description="SHA256 hash of file content")
    extension: str = Field(..., description="File extension")
    
    # Optional metadata
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    created_date: Optional[datetime] = Field(None, description="Document creation date")
    modified_date: Optional[datetime] = Field(None, description="Document modification date")
    page_count: Optional[int] = Field(None, description="Number of pages (for documents)")
    
    @field_validator("extension")
    @classmethod
    def validate_extension(cls, v):
        """Ensure extension starts with a dot."""
        if not v.startswith("."):
            return f".{v}"
        return v.lower()


class UploadRequest(BaseModel):
    """File upload request parameters."""
    
    # Document metadata
    title: Optional[str] = Field(None, description="Document title override")
    description: Optional[str] = Field(None, description="Document description")
    kind: Optional[str] = Field(None, description="Document kind/type")
    
    # Upload options
    chunk_upload: bool = Field(default=False, description="Enable chunked upload")
    overwrite_existing: bool = Field(default=False, description="Overwrite if duplicate exists")
    create_version: bool = Field(default=False, description="Create new version if duplicate exists")
    skip_virus_scan: bool = Field(default=False, description="Skip virus scanning (admin only)")
    
    # Processing options
    extract_text: bool = Field(default=True, description="Extract text content")
    generate_thumbnail: bool = Field(default=True, description="Generate thumbnail")
    auto_process: bool = Field(default=True, description="Start processing immediately")


class UploadSession(BaseModel):
    """Upload session information."""
    
    id: UUID = Field(..., description="Upload session ID")
    tenant_id: UUID = Field(..., description="Tenant ID")
    user_id: UUID = Field(..., description="User ID who initiated upload")
    
    # Session details
    status: UploadStatus = Field(..., description="Current session status")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    expires_at: datetime = Field(..., description="Session expiration time")
    
    # File information
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="Expected MIME type")
    total_size: int = Field(..., ge=0, description="Total file size")
    uploaded_size: int = Field(default=0, ge=0, description="Bytes uploaded so far")
    
    # Chunked upload details
    chunk_size: Optional[int] = Field(None, description="Chunk size for chunked uploads")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks")
    uploaded_chunks: List[int] = Field(default_factory=list, description="List of uploaded chunk numbers")
    
    # Upload options
    upload_options: UploadRequest = Field(..., description="Upload configuration")
    
    # Progress tracking
    progress_percent: float = Field(default=0.0, ge=0, le=100, description="Upload progress")
    bytes_uploaded: int = Field(default=0, ge=0, description="Bytes uploaded so far")
    current_chunk: Optional[int] = Field(None, description="Current chunk being processed")
    message: Optional[str] = Field(None, description="Current status message")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error: Optional[FileValidationError] = Field(None, description="Detailed error information")
    document_id: Optional[UUID] = Field(None, description="Created document ID")


class UploadResponse(BaseModel):
    """File upload response."""
    
    upload_id: UUID = Field(..., description="Upload session ID")
    status: UploadStatus = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    
    # File information
    file_metadata: Optional[FileMetadata] = Field(None, description="File metadata")
    document_id: Optional[UUID] = Field(None, description="Created document ID")
    
    # Upload details
    upload_url: Optional[str] = Field(None, description="Upload URL for chunked uploads")
    chunk_size: Optional[int] = Field(None, description="Recommended chunk size")
    expires_at: Optional[datetime] = Field(None, description="Upload session expiration")
    
    # Progress tracking
    progress_percent: float = Field(default=0.0, ge=0, le=100, description="Current progress")
    websocket_url: Optional[str] = Field(None, description="WebSocket URL for progress updates")


class ChunkUploadRequest(BaseModel):
    """Chunked upload request."""
    
    upload_id: UUID = Field(..., description="Upload session ID")
    chunk_number: int = Field(..., ge=0, description="Chunk number (0-based)")
    chunk_size: int = Field(..., gt=0, description="Size of this chunk")
    total_chunks: int = Field(..., gt=0, description="Total number of chunks")
    is_final_chunk: bool = Field(default=False, description="Whether this is the final chunk")


class ChunkUploadResponse(BaseModel):
    """Chunked upload response."""

    upload_id: UUID = Field(..., description="Upload session ID")
    chunk_number: int = Field(..., description="Uploaded chunk number")
    status: UploadStatus = Field(..., description="Upload status")
    progress_percent: float = Field(..., ge=0, le=100, description="Overall progress")
    bytes_uploaded: int = Field(..., ge=0, description="Total bytes uploaded so far")
    total_bytes: int = Field(..., ge=0, description="Total file size")
    chunks_uploaded: int = Field(..., ge=0, description="Number of chunks uploaded")
    total_chunks: int = Field(..., gt=0, description="Total number of chunks")
    next_chunk_number: Optional[int] = Field(None, description="Next expected chunk number")
    is_complete: bool = Field(default=False, description="Whether all chunks are uploaded")


class DuplicateFileInfo(BaseModel):
    """Information about duplicate file detection."""

    is_duplicate: bool = Field(..., description="Whether file is a duplicate")
    existing_document_id: Optional[UUID] = Field(None, description="ID of existing document")
    existing_filename: Optional[str] = Field(None, description="Filename of existing document")
    existing_upload_date: Optional[datetime] = Field(None, description="Upload date of existing document")
    existing_version: Optional[int] = Field(None, description="Version of existing document")
    new_version: Optional[int] = Field(None, description="New version number if versioned")
    sha256_hash: str = Field(..., description="SHA256 hash used for comparison")
    action_taken: str = Field(..., description="Action taken (skipped, overwritten, versioned, uploaded)")
    duplicate_count: int = Field(default=0, description="Number of duplicates found")
    tenant_duplicate_count: int = Field(default=0, description="Total duplicates in tenant")


class UploadQuotaInfo(BaseModel):
    """Upload quota information for tenant."""
    
    tenant_id: UUID = Field(..., description="Tenant ID")
    total_quota: int = Field(..., description="Total quota in bytes")
    used_quota: int = Field(..., description="Used quota in bytes")
    available_quota: int = Field(..., description="Available quota in bytes")
    quota_percentage: float = Field(..., ge=0, le=100, description="Quota usage percentage")
    file_count: int = Field(..., description="Number of uploaded files")


class UploadStatsResponse(BaseModel):
    """Upload statistics response."""
    
    total_uploads: int = Field(..., description="Total number of uploads")
    successful_uploads: int = Field(..., description="Number of successful uploads")
    failed_uploads: int = Field(..., description="Number of failed uploads")
    total_size: int = Field(..., description="Total size of uploaded files")
    duplicate_files: int = Field(..., description="Number of duplicate files detected")
    quota_info: UploadQuotaInfo = Field(..., description="Quota information")
    
    # Recent activity
    uploads_today: int = Field(..., description="Uploads today")
    uploads_this_week: int = Field(..., description="Uploads this week")
    uploads_this_month: int = Field(..., description="Uploads this month")


class DuplicateStatistics(BaseModel):
    """Statistics about duplicate detection for a tenant."""

    tenant_id: UUID = Field(..., description="Tenant ID")
    total_documents: int = Field(..., description="Total number of documents")
    unique_documents: int = Field(..., description="Number of unique documents")
    duplicate_documents: int = Field(..., description="Number of duplicate documents")
    duplicate_percentage: float = Field(..., ge=0, le=100, description="Percentage of duplicates")
    most_duplicated_hash: Optional[str] = Field(None, description="SHA256 hash with most duplicates")
    most_duplicated_count: int = Field(default=0, description="Count of most duplicated file")
    recent_duplicates: int = Field(default=0, description="Recent duplicates (last 30 days)")
    storage_saved_bytes: int = Field(default=0, description="Estimated storage saved by deduplication")


class DocumentVersion(BaseModel):
    """Information about a document version."""

    document_id: UUID = Field(..., description="Document ID")
    version: int = Field(..., description="Version number")
    title: str = Field(..., description="Document title")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by: UUID = Field(..., description="User who created this version")
    file_size: Optional[int] = Field(None, description="File size in bytes")


class DocumentVersionHistory(BaseModel):
    """Complete version history for a document."""

    sha256_hash: str = Field(..., description="SHA256 hash of the document")
    total_versions: int = Field(..., description="Total number of versions")
    versions: List[DocumentVersion] = Field(..., description="List of all versions")
    latest_version: int = Field(..., description="Latest version number")
    first_upload: datetime = Field(..., description="First upload timestamp")
    last_upload: datetime = Field(..., description="Last upload timestamp")
