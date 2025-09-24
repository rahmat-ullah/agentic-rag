"""
Security service for file validation and threat detection.

This module provides comprehensive security validation including virus scanning,
content validation, malicious file detection, and quarantine management.
"""

import asyncio
import hashlib
import logging
import socket
import struct
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from fastapi import UploadFile

from agentic_rag.config import Settings

# Try to import magic, fall back to None if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    magic = None
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat level enumeration."""
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    QUARANTINED = "quarantined"


class SecurityViolationType(str, Enum):
    """Security violation types."""
    VIRUS_DETECTED = "virus_detected"
    MALICIOUS_CONTENT = "malicious_content"
    CONTENT_MISMATCH = "content_mismatch"
    SUSPICIOUS_STRUCTURE = "suspicious_structure"
    OVERSIZED_FILE = "oversized_file"
    FORBIDDEN_TYPE = "forbidden_type"


class SecurityValidationResult:
    """Result of security validation."""
    
    def __init__(
        self,
        is_safe: bool,
        threat_level: ThreatLevel,
        violations: List[Dict],
        scan_details: Dict,
        quarantine_id: Optional[str] = None
    ):
        self.is_safe = is_safe
        self.threat_level = threat_level
        self.violations = violations
        self.scan_details = scan_details
        self.quarantine_id = quarantine_id
        self.timestamp = datetime.utcnow()


class ClamAVScanner:
    """ClamAV virus scanner integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.host = settings.upload.clamav_host
        self.port = settings.upload.clamav_port
        self.socket_path = settings.upload.clamav_socket_path
        self.timeout = settings.upload.virus_scan_timeout
    
    async def scan_content(self, content: bytes) -> Tuple[bool, str, Dict]:
        """Scan content for viruses using ClamAV."""
        if not self.settings.upload.virus_scan_enabled:
            return True, "SCAN_DISABLED", {"status": "disabled"}
        
        try:
            if self.socket_path:
                return await self._scan_via_socket(content)
            else:
                return await self._scan_via_tcp(content)
        except Exception as e:
            logger.error(f"Virus scan failed: {e}")
            # Fail secure - treat scan failure as potential threat
            return False, f"SCAN_ERROR: {str(e)}", {"error": str(e)}
    
    async def _scan_via_tcp(self, content: bytes) -> Tuple[bool, str, Dict]:
        """Scan via TCP connection to ClamAV daemon."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
            
            # Send INSTREAM command
            writer.write(b"zINSTREAM\0")
            
            # Send content in chunks
            chunk_size = 4096
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                writer.write(struct.pack("!L", len(chunk)))
                writer.write(chunk)
            
            # Send end marker
            writer.write(struct.pack("!L", 0))
            await writer.drain()
            
            # Read response
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=self.timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            response_str = response.decode().strip()
            is_clean = "OK" in response_str
            
            return is_clean, response_str, {
                "method": "tcp",
                "host": self.host,
                "port": self.port,
                "response": response_str
            }
            
        except asyncio.TimeoutError:
            raise Exception(f"ClamAV scan timeout after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"ClamAV TCP scan failed: {e}")
    
    async def _scan_via_socket(self, content: bytes) -> Tuple[bool, str, Dict]:
        """Scan via Unix socket connection to ClamAV daemon."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=self.timeout
            )
            
            # Send INSTREAM command
            writer.write(b"zINSTREAM\0")
            
            # Send content in chunks
            chunk_size = 4096
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                writer.write(struct.pack("!L", len(chunk)))
                writer.write(chunk)
            
            # Send end marker
            writer.write(struct.pack("!L", 0))
            await writer.drain()
            
            # Read response
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=self.timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            response_str = response.decode().strip()
            is_clean = "OK" in response_str
            
            return is_clean, response_str, {
                "method": "socket",
                "socket_path": self.socket_path,
                "response": response_str
            }
            
        except asyncio.TimeoutError:
            raise Exception(f"ClamAV scan timeout after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"ClamAV socket scan failed: {e}")


class ContentValidator:
    """Advanced content validation for file security."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.magic_enabled = settings.upload.magic_bytes_validation and MAGIC_AVAILABLE
        self.structure_validation = settings.upload.structure_validation_enabled

        if settings.upload.magic_bytes_validation and not MAGIC_AVAILABLE:
            logger.warning("Magic bytes validation requested but python-magic not available. Install python-magic for full content validation.")
    
    def validate_content(self, file: UploadFile, content: bytes) -> Tuple[bool, List[Dict]]:
        """Perform comprehensive content validation."""
        violations = []
        
        if self.magic_enabled:
            magic_violations = self._validate_magic_bytes(file, content)
            violations.extend(magic_violations)
        
        if self.structure_validation:
            structure_violations = self._validate_file_structure(file, content)
            violations.extend(structure_violations)
        
        # Check for suspicious patterns
        suspicious_violations = self._detect_suspicious_patterns(content)
        violations.extend(suspicious_violations)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _validate_magic_bytes(self, file: UploadFile, content: bytes) -> List[Dict]:
        """Validate file content matches declared MIME type using magic bytes."""
        violations = []

        if not MAGIC_AVAILABLE:
            logger.warning("Magic bytes validation skipped: python-magic not available")
            return violations

        try:
            # Get actual MIME type from content
            detected_mime = magic.from_buffer(content, mime=True)
            declared_mime = file.content_type
            
            # Define MIME type mappings for common mismatches
            mime_mappings = {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
                    "application/zip",  # DOCX files are ZIP archives
                ],
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
                    "application/zip",  # PPTX files are ZIP archives
                ],
            }
            
            # Check if detected MIME type matches declared or is acceptable
            acceptable_types = mime_mappings.get(declared_mime, [declared_mime])
            
            if detected_mime not in acceptable_types:
                violations.append({
                    "type": SecurityViolationType.CONTENT_MISMATCH,
                    "severity": "high",
                    "message": f"Content type mismatch: declared {declared_mime}, detected {detected_mime}",
                    "declared_type": declared_mime,
                    "detected_type": detected_mime
                })
                
        except Exception as e:
            logger.warning(f"Magic bytes validation failed: {e}")
            violations.append({
                "type": SecurityViolationType.CONTENT_MISMATCH,
                "severity": "medium",
                "message": f"Could not validate content type: {e}",
                "error": str(e)
            })
        
        return violations
    
    def _validate_file_structure(self, file: UploadFile, content: bytes) -> List[Dict]:
        """Validate file structure for known formats."""
        violations = []
        filename = file.filename or ""
        
        try:
            # PDF structure validation
            if filename.lower().endswith('.pdf'):
                if not content.startswith(b'%PDF-'):
                    violations.append({
                        "type": SecurityViolationType.SUSPICIOUS_STRUCTURE,
                        "severity": "high",
                        "message": "PDF file missing required header",
                        "expected": "%PDF-",
                        "found": content[:10].decode('utf-8', errors='ignore')
                    })
            
            # ZIP-based formats (DOCX, PPTX)
            elif filename.lower().endswith(('.docx', '.pptx', '.xlsx')):
                if not content.startswith(b'PK'):
                    violations.append({
                        "type": SecurityViolationType.SUSPICIOUS_STRUCTURE,
                        "severity": "high",
                        "message": "Office document missing ZIP signature",
                        "expected": "PK",
                        "found": content[:10].decode('utf-8', errors='ignore')
                    })
            
            # Image format validation
            elif filename.lower().endswith('.png'):
                if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                    violations.append({
                        "type": SecurityViolationType.SUSPICIOUS_STRUCTURE,
                        "severity": "high",
                        "message": "PNG file missing required signature",
                        "expected": "PNG signature",
                        "found": content[:10].hex()
                    })
            
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                if not content.startswith(b'\xff\xd8\xff'):
                    violations.append({
                        "type": SecurityViolationType.SUSPICIOUS_STRUCTURE,
                        "severity": "high",
                        "message": "JPEG file missing required signature",
                        "expected": "JPEG signature",
                        "found": content[:10].hex()
                    })
                    
        except Exception as e:
            logger.warning(f"Structure validation failed: {e}")
            violations.append({
                "type": SecurityViolationType.SUSPICIOUS_STRUCTURE,
                "severity": "low",
                "message": f"Structure validation error: {e}",
                "error": str(e)
            })
        
        return violations
    
    def _detect_suspicious_patterns(self, content: bytes) -> List[Dict]:
        """Detect suspicious patterns in file content."""
        violations = []
        
        # Check for embedded executables
        suspicious_patterns = [
            (b'MZ', "Possible embedded Windows executable"),
            (b'\x7fELF', "Possible embedded Linux executable"),
            (b'<script', "Possible embedded script"),
            (b'javascript:', "Possible JavaScript injection"),
            (b'vbscript:', "Possible VBScript injection"),
        ]
        
        for pattern, description in suspicious_patterns:
            if pattern in content:
                violations.append({
                    "type": SecurityViolationType.MALICIOUS_CONTENT,
                    "severity": "high",
                    "message": description,
                    "pattern": pattern.decode('utf-8', errors='ignore'),
                    "position": content.find(pattern)
                })
        
        return violations


class QuarantineManager:
    """Quarantine system for suspicious files."""

    def __init__(self, settings: Settings, storage_service):
        self.settings = settings
        self.storage_service = storage_service
        self.quarantine_enabled = settings.upload.quarantine_enabled
        self.retention_days = settings.upload.quarantine_retention_days

    async def quarantine_file(
        self,
        content: bytes,
        filename: str,
        tenant_id: UUID,
        violations: List[Dict],
        metadata: Optional[Dict] = None
    ) -> str:
        """Quarantine a suspicious file."""
        if not self.quarantine_enabled:
            logger.warning("Quarantine disabled, file not quarantined")
            return None

        quarantine_id = str(uuid4())
        quarantine_timestamp = datetime.utcnow()

        # Create quarantine metadata
        quarantine_metadata = {
            "quarantine_id": quarantine_id,
            "original_filename": filename,
            "tenant_id": str(tenant_id),
            "quarantine_timestamp": quarantine_timestamp.isoformat(),
            "violations": violations,
            "retention_until": (quarantine_timestamp + timedelta(days=self.retention_days)).isoformat(),
            "status": "quarantined",
            "file_size": len(content),
            "file_hash": hashlib.sha256(content).hexdigest()
        }

        if metadata:
            quarantine_metadata.update(metadata)

        # Store in quarantine bucket with special naming
        quarantine_path = f"quarantine/{tenant_id}/{quarantine_timestamp.strftime('%Y/%m/%d')}/{quarantine_id}_{filename}"

        try:
            await self.storage_service.store_file(
                object_name=quarantine_path,
                content=content,
                bucket=self.storage_service.bucket_exports,  # Use exports bucket for quarantine
                metadata=quarantine_metadata
            )

            logger.warning(f"File quarantined: {quarantine_id} - {filename}")
            return quarantine_id

        except Exception as e:
            logger.error(f"Failed to quarantine file {filename}: {e}")
            raise Exception(f"Quarantine operation failed: {e}")

    async def get_quarantine_info(self, quarantine_id: str) -> Optional[Dict]:
        """Get information about a quarantined file."""
        # This would query the storage service for quarantine metadata
        # Implementation depends on storage backend capabilities
        return {
            "quarantine_id": quarantine_id,
            "status": "quarantined",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def cleanup_expired_quarantine(self) -> int:
        """Clean up expired quarantined files."""
        if not self.quarantine_enabled:
            return 0

        # This would implement cleanup logic for expired quarantine files
        # Implementation depends on storage backend capabilities
        logger.info("Quarantine cleanup completed")
        return 0


class SecurityAuditLogger:
    """Audit logging for security events."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.audit_enabled = settings.upload.security_audit_enabled
        self.violation_alerts = settings.upload.security_violation_alerts

    async def log_upload_attempt(
        self,
        tenant_id: UUID,
        user_id: UUID,
        filename: str,
        file_size: int,
        file_hash: str,
        result: SecurityValidationResult,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log file upload attempt with security details."""
        if not self.audit_enabled:
            return

        audit_entry = {
            "event_type": "file_upload_attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "filename": filename,
            "file_size": file_size,
            "file_hash": file_hash,
            "security_result": {
                "is_safe": result.is_safe,
                "threat_level": result.threat_level.value,
                "violations": result.violations,
                "scan_details": result.scan_details,
                "quarantine_id": result.quarantine_id
            },
            "client_info": {
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        }

        # Log to structured logger
        if result.is_safe:
            logger.info("File upload successful", extra=audit_entry)
        else:
            logger.warning("File upload blocked due to security violations", extra=audit_entry)

            # Send alerts for security violations
            if self.violation_alerts and result.violations:
                await self._send_security_alert(audit_entry)

    async def log_security_violation(
        self,
        violation_type: SecurityViolationType,
        tenant_id: UUID,
        details: Dict,
        severity: str = "medium"
    ) -> None:
        """Log security violation."""
        if not self.audit_enabled:
            return

        violation_entry = {
            "event_type": "security_violation",
            "timestamp": datetime.utcnow().isoformat(),
            "violation_type": violation_type.value,
            "tenant_id": str(tenant_id),
            "severity": severity,
            "details": details
        }

        logger.warning("Security violation detected", extra=violation_entry)

        if self.violation_alerts and severity in ["high", "critical"]:
            await self._send_security_alert(violation_entry)

    async def _send_security_alert(self, alert_data: Dict) -> None:
        """Send security alert notification."""
        # This would integrate with alerting system (email, Slack, etc.)
        logger.critical(f"SECURITY ALERT: {alert_data.get('event_type', 'unknown')}")


class SecurityService:
    """Main security service coordinating all security validations."""

    def __init__(self, settings: Settings, storage_service):
        self.settings = settings
        self.virus_scanner = ClamAVScanner(settings)
        self.content_validator = ContentValidator(settings)
        self.quarantine_manager = QuarantineManager(settings, storage_service)
        self.audit_logger = SecurityAuditLogger(settings)

    async def validate_file_security(
        self,
        file: UploadFile,
        content: bytes,
        tenant_id: UUID,
        user_id: UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityValidationResult:
        """Perform comprehensive security validation on uploaded file."""
        violations = []
        scan_details = {}
        threat_level = ThreatLevel.CLEAN
        quarantine_id = None

        # Calculate file hash for audit
        file_hash = hashlib.sha256(content).hexdigest()

        try:
            # 1. Virus scanning
            virus_clean, virus_result, virus_details = await self.virus_scanner.scan_content(content)
            scan_details["virus_scan"] = virus_details

            if not virus_clean:
                violations.append({
                    "type": SecurityViolationType.VIRUS_DETECTED,
                    "severity": "critical",
                    "message": f"Virus detected: {virus_result}",
                    "scanner_result": virus_result
                })
                threat_level = ThreatLevel.MALICIOUS

            # 2. Content validation
            content_valid, content_violations = self.content_validator.validate_content(file, content)
            violations.extend(content_violations)

            if content_violations:
                # Determine threat level based on violations
                high_severity_violations = [v for v in content_violations if v.get("severity") == "high"]
                if high_severity_violations:
                    threat_level = ThreatLevel.SUSPICIOUS if threat_level == ThreatLevel.CLEAN else threat_level

            # 3. Determine if file should be quarantined
            should_quarantine = (
                threat_level in [ThreatLevel.MALICIOUS, ThreatLevel.SUSPICIOUS] or
                any(v.get("severity") in ["critical", "high"] for v in violations)
            )

            # 4. Quarantine if necessary
            if should_quarantine:
                quarantine_id = await self.quarantine_manager.quarantine_file(
                    content=content,
                    filename=file.filename or "unknown",
                    tenant_id=tenant_id,
                    violations=violations,
                    metadata={
                        "user_id": str(user_id),
                        "ip_address": ip_address,
                        "user_agent": user_agent
                    }
                )
                threat_level = ThreatLevel.QUARANTINED

            # 5. Create result
            is_safe = threat_level == ThreatLevel.CLEAN
            result = SecurityValidationResult(
                is_safe=is_safe,
                threat_level=threat_level,
                violations=violations,
                scan_details=scan_details,
                quarantine_id=quarantine_id
            )

            # 6. Audit logging
            await self.audit_logger.log_upload_attempt(
                tenant_id=tenant_id,
                user_id=user_id,
                filename=file.filename or "unknown",
                file_size=len(content),
                file_hash=file_hash,
                result=result,
                ip_address=ip_address,
                user_agent=user_agent
            )

            return result

        except Exception as e:
            logger.error(f"Security validation failed: {e}")

            # Log security validation failure
            await self.audit_logger.log_security_violation(
                violation_type=SecurityViolationType.MALICIOUS_CONTENT,
                tenant_id=tenant_id,
                details={
                    "error": str(e),
                    "filename": file.filename,
                    "file_size": len(content),
                    "user_id": str(user_id)
                },
                severity="high"
            )

            # Fail secure - treat validation failure as potential threat
            return SecurityValidationResult(
                is_safe=False,
                threat_level=ThreatLevel.SUSPICIOUS,
                violations=[{
                    "type": SecurityViolationType.MALICIOUS_CONTENT,
                    "severity": "high",
                    "message": f"Security validation failed: {e}",
                    "error": str(e)
                }],
                scan_details={"error": str(e)}
            )
