"""
Unit tests for security service.

This module contains comprehensive tests for the security service including
virus scanning, content validation, quarantine, and audit logging.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import UploadFile

from agentic_rag.config import Settings
from agentic_rag.services.security import (
    ClamAVScanner, ContentValidator, QuarantineManager, SecurityAuditLogger,
    SecurityService, SecurityValidationResult, ThreatLevel, SecurityViolationType
)


class TestClamAVScanner:
    """Test cases for ClamAVScanner."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            upload={
                "virus_scan_enabled": True,
                "virus_scan_timeout": 30,
                "clamav_host": "localhost",
                "clamav_port": 3310,
                "clamav_socket_path": None
            }
        )
    
    @pytest.fixture
    def scanner(self, settings):
        """Create ClamAV scanner instance."""
        return ClamAVScanner(settings)
    
    @pytest.mark.asyncio
    async def test_scan_disabled(self, settings):
        """Test virus scan when disabled."""
        settings.upload.virus_scan_enabled = False
        scanner = ClamAVScanner(settings)
        
        is_clean, result, details = await scanner.scan_content(b"test content")
        
        assert is_clean is True
        assert result == "SCAN_DISABLED"
        assert details["status"] == "disabled"
    
    @pytest.mark.asyncio
    async def test_scan_tcp_success(self, scanner):
        """Test successful TCP virus scan."""
        with patch('asyncio.open_connection') as mock_connect:
            # Mock successful connection and clean scan result
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_reader.read.return_value = b"stream: OK\n"
            mock_connect.return_value = (mock_reader, mock_writer)
            
            is_clean, result, details = await scanner.scan_content(b"clean content")
            
            assert is_clean is True
            assert "OK" in result
            assert details["method"] == "tcp"
            assert details["host"] == "localhost"
            assert details["port"] == 3310
    
    @pytest.mark.asyncio
    async def test_scan_tcp_virus_detected(self, scanner):
        """Test TCP virus scan with virus detected."""
        with patch('asyncio.open_connection') as mock_connect:
            # Mock virus detection
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_reader.read.return_value = b"stream: Eicar-Test-Signature FOUND\n"
            mock_connect.return_value = (mock_reader, mock_writer)
            
            is_clean, result, details = await scanner.scan_content(b"virus content")
            
            assert is_clean is False
            assert "FOUND" in result
            assert details["method"] == "tcp"
    
    @pytest.mark.asyncio
    async def test_scan_socket_success(self, settings):
        """Test successful Unix socket virus scan."""
        settings.upload.clamav_socket_path = "/var/run/clamav/clamd.ctl"
        scanner = ClamAVScanner(settings)

        # Skip this test on Windows where Unix sockets aren't available
        if not hasattr(asyncio, 'open_unix_connection'):
            pytest.skip("Unix sockets not available on this platform")

        with patch('asyncio.open_unix_connection') as mock_connect:
            # Mock successful connection and clean scan result
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_reader.read.return_value = b"stream: OK\n"
            mock_connect.return_value = (mock_reader, mock_writer)
            
            is_clean, result, details = await scanner.scan_content(b"clean content")
            
            assert is_clean is True
            assert "OK" in result
            assert details["method"] == "socket"
            assert details["socket_path"] == "/var/run/clamav/clamd.ctl"
    
    @pytest.mark.asyncio
    async def test_scan_timeout(self, scanner):
        """Test virus scan timeout handling."""
        with patch('asyncio.open_connection') as mock_connect:
            # Mock timeout
            mock_connect.side_effect = asyncio.TimeoutError()

            # The scanner should handle timeout gracefully and return a result
            is_clean, result, details = await scanner.scan_content(b"test content")

            # Should return False for safety when scan times out
            assert is_clean is False
            assert "timeout" in result.lower() or "error" in result.lower()
    
    @pytest.mark.asyncio
    async def test_scan_connection_error(self, scanner):
        """Test virus scan connection error handling."""
        with patch('asyncio.open_connection') as mock_connect:
            # Mock connection error
            mock_connect.side_effect = ConnectionRefusedError("Connection refused")

            # The scanner should handle the error gracefully and return a result
            is_clean, result, details = await scanner.scan_content(b"test content")

            # Should return False for safety when scan fails
            assert is_clean is False
            assert "scan failed" in result.lower() or "error" in result.lower()


class TestContentValidator:
    """Test cases for ContentValidator."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            upload={
                "content_validation_enabled": True,
                "magic_bytes_validation": True,
                "structure_validation_enabled": True
            }
        )
    
    @pytest.fixture
    def validator(self, settings):
        """Create content validator instance."""
        return ContentValidator(settings)
    
    @pytest.fixture
    def mock_upload_file(self):
        """Create mock upload file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.content_type = "application/pdf"
        return file
    
    def test_validate_pdf_structure_valid(self, validator, mock_upload_file):
        """Test valid PDF structure validation."""
        mock_upload_file.filename = "test.pdf"
        content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"  # Valid PDF header
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should be valid (no structure violations for valid PDF)
        structure_violations = [v for v in violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
        assert len(structure_violations) == 0
    
    def test_validate_pdf_structure_invalid(self, validator, mock_upload_file):
        """Test invalid PDF structure validation."""
        mock_upload_file.filename = "test.pdf"
        content = b"Not a PDF file"  # Invalid PDF content
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should detect structure violation
        structure_violations = [v for v in violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
        assert len(structure_violations) > 0
        assert "PDF file missing required header" in structure_violations[0]["message"]
    
    def test_validate_docx_structure_valid(self, validator, mock_upload_file):
        """Test valid DOCX structure validation."""
        mock_upload_file.filename = "test.docx"
        content = b"PK\x03\x04"  # Valid ZIP signature (DOCX is ZIP-based)
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should be valid (no structure violations for valid ZIP signature)
        structure_violations = [v for v in violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
        assert len(structure_violations) == 0
    
    def test_validate_docx_structure_invalid(self, validator, mock_upload_file):
        """Test invalid DOCX structure validation."""
        mock_upload_file.filename = "test.docx"
        content = b"Not a ZIP file"  # Invalid DOCX content
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should detect structure violation
        structure_violations = [v for v in violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
        assert len(structure_violations) > 0
        assert "Office document missing ZIP signature" in structure_violations[0]["message"]
    
    def test_detect_suspicious_patterns(self, validator, mock_upload_file):
        """Test detection of suspicious patterns."""
        # Content with embedded executable signature
        content = b"Some content MZ\x90\x00 more content"
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should detect malicious content
        malicious_violations = [v for v in violations if v["type"] == SecurityViolationType.MALICIOUS_CONTENT]
        assert len(malicious_violations) > 0
        assert "embedded" in malicious_violations[0]["message"].lower() and "executable" in malicious_violations[0]["message"].lower()
    
    def test_detect_script_injection(self, validator, mock_upload_file):
        """Test detection of script injection attempts."""
        # Content with script tags
        content = b"<script>alert('xss')</script>"
        
        is_valid, violations = validator.validate_content(mock_upload_file, content)
        
        # Should detect malicious content
        malicious_violations = [v for v in violations if v["type"] == SecurityViolationType.MALICIOUS_CONTENT]
        assert len(malicious_violations) > 0
        assert "embedded script" in malicious_violations[0]["message"].lower()
    


    def test_magic_bytes_validation_disabled_when_unavailable(self, validator, mock_upload_file):
        """Test magic bytes validation when python-magic is not available."""
        # Magic should be disabled when not available
        assert validator.magic_enabled is False

        content = b"Any content"

        is_valid, violations = validator.validate_content(mock_upload_file, content)

        # Should not have content mismatch violations when magic is disabled
        mismatch_violations = [v for v in violations if v["type"] == SecurityViolationType.CONTENT_MISMATCH]
        assert len(mismatch_violations) == 0


class TestQuarantineManager:
    """Test cases for QuarantineManager."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            upload={
                "quarantine_enabled": True,
                "quarantine_retention_days": 30
            }
        )
    
    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service."""
        storage = MagicMock()
        storage.bucket_exports = "test-exports"
        storage.store_file = AsyncMock()
        return storage
    
    @pytest.fixture
    def quarantine_manager(self, settings, mock_storage_service):
        """Create quarantine manager instance."""
        return QuarantineManager(settings, mock_storage_service)
    
    @pytest.mark.asyncio
    async def test_quarantine_file_success(self, quarantine_manager, mock_storage_service):
        """Test successful file quarantine."""
        content = b"suspicious file content"
        filename = "suspicious.exe"
        tenant_id = uuid4()
        violations = [{"type": "virus_detected", "message": "Virus found"}]
        
        quarantine_id = await quarantine_manager.quarantine_file(
            content=content,
            filename=filename,
            tenant_id=tenant_id,
            violations=violations
        )
        
        assert quarantine_id is not None
        assert mock_storage_service.store_file.called
        
        # Verify storage call parameters
        call_args = mock_storage_service.store_file.call_args
        assert call_args[1]["bucket"] == "test-exports"
        assert "quarantine/" in call_args[1]["object_name"]
        assert str(tenant_id) in call_args[1]["object_name"]
        assert filename in call_args[1]["object_name"]
    
    @pytest.mark.asyncio
    async def test_quarantine_disabled(self, settings, mock_storage_service):
        """Test quarantine when disabled."""
        settings.upload.quarantine_enabled = False
        quarantine_manager = QuarantineManager(settings, mock_storage_service)
        
        quarantine_id = await quarantine_manager.quarantine_file(
            content=b"content",
            filename="test.txt",
            tenant_id=uuid4(),
            violations=[]
        )
        
        assert quarantine_id is None
        assert not mock_storage_service.store_file.called
    
    @pytest.mark.asyncio
    async def test_quarantine_storage_failure(self, quarantine_manager, mock_storage_service):
        """Test quarantine when storage fails."""
        mock_storage_service.store_file.side_effect = Exception("Storage failed")
        
        with pytest.raises(Exception) as exc_info:
            await quarantine_manager.quarantine_file(
                content=b"content",
                filename="test.txt",
                tenant_id=uuid4(),
                violations=[]
            )
        
        assert "Quarantine operation failed" in str(exc_info.value)


class TestSecurityAuditLogger:
    """Test cases for SecurityAuditLogger."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            upload={
                "security_audit_enabled": True,
                "security_violation_alerts": True
            }
        )

    @pytest.fixture
    def audit_logger(self, settings):
        """Create security audit logger instance."""
        return SecurityAuditLogger(settings)

    @pytest.mark.asyncio
    async def test_log_upload_attempt_safe(self, audit_logger):
        """Test logging safe upload attempt."""
        result = SecurityValidationResult(
            is_safe=True,
            threat_level=ThreatLevel.CLEAN,
            violations=[],
            scan_details={"virus_scan": {"status": "clean"}},
            quarantine_id=None
        )

        with patch('agentic_rag.services.security.logger') as mock_logger:
            await audit_logger.log_upload_attempt(
                tenant_id=uuid4(),
                user_id=uuid4(),
                filename="safe.pdf",
                file_size=1024,
                file_hash="abc123",
                result=result,
                ip_address="192.168.1.1",
                user_agent="test-agent"
            )

            # Should log as info for safe files
            assert mock_logger.info.called
            assert not mock_logger.warning.called

    @pytest.mark.asyncio
    async def test_log_upload_attempt_unsafe(self, audit_logger):
        """Test logging unsafe upload attempt."""
        result = SecurityValidationResult(
            is_safe=False,
            threat_level=ThreatLevel.MALICIOUS,
            violations=[{"type": "virus_detected", "message": "Virus found"}],
            scan_details={"virus_scan": {"status": "infected"}},
            quarantine_id="quarantine-123"
        )

        with patch('agentic_rag.services.security.logger') as mock_logger:
            await audit_logger.log_upload_attempt(
                tenant_id=uuid4(),
                user_id=uuid4(),
                filename="malicious.exe",
                file_size=2048,
                file_hash="def456",
                result=result
            )

            # Should log as warning for unsafe files
            assert mock_logger.warning.called

    @pytest.mark.asyncio
    async def test_log_security_violation(self, audit_logger):
        """Test logging security violation."""
        with patch('agentic_rag.services.security.logger') as mock_logger:
            await audit_logger.log_security_violation(
                violation_type=SecurityViolationType.VIRUS_DETECTED,
                tenant_id=uuid4(),
                details={"filename": "virus.exe", "scanner": "clamav"},
                severity="high"
            )

            # Should log as warning
            assert mock_logger.warning.called

    @pytest.mark.asyncio
    async def test_audit_disabled(self, settings):
        """Test audit logging when disabled."""
        settings.upload.security_audit_enabled = False
        audit_logger = SecurityAuditLogger(settings)

        result = SecurityValidationResult(
            is_safe=True,
            threat_level=ThreatLevel.CLEAN,
            violations=[],
            scan_details={},
            quarantine_id=None
        )

        with patch('agentic_rag.services.security.logger') as mock_logger:
            await audit_logger.log_upload_attempt(
                tenant_id=uuid4(),
                user_id=uuid4(),
                filename="test.pdf",
                file_size=1024,
                file_hash="abc123",
                result=result
            )

            # Should not log when disabled
            assert not mock_logger.info.called
            assert not mock_logger.warning.called


class TestSecurityService:
    """Test cases for SecurityService."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            upload={
                "virus_scan_enabled": True,
                "content_validation_enabled": True,
                "quarantine_enabled": True,
                "security_audit_enabled": True
            }
        )

    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service."""
        storage = MagicMock()
        storage.bucket_exports = "test-exports"
        storage.store_file = AsyncMock()
        return storage

    @pytest.fixture
    def security_service(self, settings, mock_storage_service):
        """Create security service instance."""
        return SecurityService(settings, mock_storage_service)

    @pytest.fixture
    def mock_upload_file(self):
        """Create mock upload file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.content_type = "application/pdf"
        return file

    @pytest.mark.asyncio
    async def test_validate_file_security_clean(self, security_service, mock_upload_file):
        """Test security validation for clean file."""
        content = b"%PDF-1.4\nClean PDF content"

        with patch.object(security_service.virus_scanner, 'scan_content') as mock_virus_scan:
            mock_virus_scan.return_value = (True, "OK", {"status": "clean"})

            result = await security_service.validate_file_security(
                file=mock_upload_file,
                content=content,
                tenant_id=uuid4(),
                user_id=uuid4(),
                ip_address="192.168.1.1",
                user_agent="test-agent"
            )

            assert result.is_safe is True
            assert result.threat_level == ThreatLevel.CLEAN
            assert len(result.violations) == 0
            assert result.quarantine_id is None

    @pytest.mark.asyncio
    async def test_validate_file_security_virus_detected(self, security_service, mock_upload_file):
        """Test security validation when virus is detected."""
        content = b"Malicious content"

        # Ensure quarantine is enabled for this test
        security_service.quarantine_manager.quarantine_enabled = True

        with patch.object(security_service.virus_scanner, 'scan_content') as mock_virus_scan:
            mock_virus_scan.return_value = (False, "Eicar-Test-Signature FOUND", {"status": "infected"})

            with patch.object(security_service.quarantine_manager, 'quarantine_file') as mock_quarantine:
                mock_quarantine.return_value = "quarantine-123"

                result = await security_service.validate_file_security(
                    file=mock_upload_file,
                    content=content,
                    tenant_id=uuid4(),
                    user_id=uuid4()
                )

                assert result.is_safe is False
                assert result.threat_level == ThreatLevel.QUARANTINED
                assert len(result.violations) > 0
                assert result.quarantine_id == "quarantine-123"

                # Verify virus violation
                virus_violations = [v for v in result.violations if v["type"] == SecurityViolationType.VIRUS_DETECTED]
                assert len(virus_violations) > 0

    @pytest.mark.asyncio
    async def test_validate_file_security_structure_violation(self, security_service, mock_upload_file):
        """Test security validation with file structure violation."""
        mock_upload_file.filename = "fake.pdf"
        content = b"Plain text, not PDF"

        with patch.object(security_service.virus_scanner, 'scan_content') as mock_virus_scan:
            mock_virus_scan.return_value = (True, "OK", {"status": "clean"})

            result = await security_service.validate_file_security(
                file=mock_upload_file,
                content=content,
                tenant_id=uuid4(),
                user_id=uuid4()
            )

            # Should detect structure violation for invalid PDF
            structure_violations = [v for v in result.violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
            assert len(structure_violations) > 0

    @pytest.mark.asyncio
    async def test_validate_file_security_exception_handling(self, security_service, mock_upload_file):
        """Test security validation exception handling."""
        content = b"Test content"

        with patch.object(security_service.virus_scanner, 'scan_content') as mock_virus_scan:
            mock_virus_scan.side_effect = Exception("Scanner failed")

            result = await security_service.validate_file_security(
                file=mock_upload_file,
                content=content,
                tenant_id=uuid4(),
                user_id=uuid4()
            )

            # Should fail secure
            assert result.is_safe is False
            assert result.threat_level == ThreatLevel.SUSPICIOUS
            assert len(result.violations) > 0
            assert "Security validation failed" in result.violations[0]["message"]
