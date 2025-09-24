"""
Integration tests for security service.

This module contains integration tests for the security service that test
the actual security validation workflow with realistic scenarios.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import UploadFile

from agentic_rag.config import get_settings
from agentic_rag.services.security import SecurityService, ThreatLevel, SecurityViolationType
from agentic_rag.services.storage import StorageService


class TestSecurityIntegration:
    """Integration test cases for SecurityService."""
    
    @pytest.fixture
    def settings(self):
        """Get test settings with security enabled."""
        settings = get_settings()
        # Ensure security features are enabled for testing
        settings.upload.virus_scan_enabled = True
        settings.upload.content_validation_enabled = True
        settings.upload.quarantine_enabled = True
        settings.upload.security_audit_enabled = True
        return settings
    
    @pytest.fixture
    def mock_storage_service(self):
        """Create mock storage service for testing."""
        storage = MagicMock(spec=StorageService)
        storage.bucket_exports = "test-exports"
        storage.store_file = MagicMock()
        return storage
    
    @pytest.fixture
    def security_service(self, settings, mock_storage_service):
        """Create security service instance."""
        return SecurityService(settings, mock_storage_service)
    
    def create_upload_file(self, filename: str, content_type: str) -> UploadFile:
        """Create mock upload file."""
        file = MagicMock(spec=UploadFile)
        file.filename = filename
        file.content_type = content_type
        return file
    
    @pytest.mark.asyncio
    async def test_security_service_initialization(self, security_service):
        """Test security service can be initialized with all components."""
        assert security_service is not None
        assert security_service.virus_scanner is not None
        assert security_service.content_validator is not None
        assert security_service.quarantine_manager is not None
        assert security_service.audit_logger is not None
    
    @pytest.mark.asyncio
    async def test_pdf_file_validation_clean(self, security_service):
        """Test validation of clean PDF file."""
        file = self.create_upload_file("document.pdf", "application/pdf")
        content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nClean PDF content"
        
        # Mock virus scanner to return clean
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4(),
            ip_address="192.168.1.1",
            user_agent="test-browser"
        )
        
        assert result.is_safe is True
        assert result.threat_level == ThreatLevel.CLEAN
        assert len(result.violations) == 0
        assert result.quarantine_id is None
        assert "virus_scan" in result.scan_details
    
    @pytest.mark.asyncio
    async def test_docx_file_validation_clean(self, security_service):
        """Test validation of clean DOCX file."""
        file = self.create_upload_file("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        content = b"PK\x03\x04\x14\x00\x06\x00"  # ZIP signature for DOCX
        
        # Mock virus scanner to return clean
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        assert result.is_safe is True
        assert result.threat_level == ThreatLevel.CLEAN
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_file_with_virus_detected(self, security_service):
        """Test validation when virus is detected."""
        file = self.create_upload_file("malicious.exe", "application/octet-stream")
        content = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        
        # Mock virus scanner to detect virus
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (False, "Eicar-Test-Signature FOUND", {"status": "infected"})
        
        # Mock quarantine
        security_service.quarantine_manager.quarantine_file = MagicMock()
        security_service.quarantine_manager.quarantine_file.return_value = "quarantine-123"
        
        result = await security_service.validate_file_security(
            file=file,
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
        assert "virus detected" in virus_violations[0]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_file_structure_validation_failure(self, security_service):
        """Test validation when file structure is invalid."""
        file = self.create_upload_file("fake.pdf", "application/pdf")
        content = b"This is not a PDF file"  # Invalid PDF content
        
        # Mock virus scanner to return clean
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Should detect structure violations
        structure_violations = [v for v in result.violations if v["type"] == SecurityViolationType.SUSPICIOUS_STRUCTURE]
        assert len(structure_violations) > 0
        assert "PDF file missing required header" in structure_violations[0]["message"]
    
    @pytest.mark.asyncio
    async def test_suspicious_content_detection(self, security_service):
        """Test detection of suspicious content patterns."""
        file = self.create_upload_file("suspicious.txt", "text/plain")
        content = b"Normal content MZ\x90\x00 with embedded executable signature"
        
        # Mock virus scanner to return clean
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Should detect malicious content patterns
        malicious_violations = [v for v in result.violations if v["type"] == SecurityViolationType.MALICIOUS_CONTENT]
        assert len(malicious_violations) > 0
        assert "embedded windows executable" in malicious_violations[0]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_quarantine_workflow(self, security_service, mock_storage_service):
        """Test complete quarantine workflow."""
        file = self.create_upload_file("quarantine_test.exe", "application/octet-stream")
        content = b"Suspicious executable content"
        tenant_id = uuid4()
        user_id = uuid4()
        
        # Mock virus scanner to detect threat
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (False, "Threat detected", {"status": "threat"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # Verify quarantine was triggered
        assert result.is_safe is False
        assert result.threat_level == ThreatLevel.QUARANTINED
        assert result.quarantine_id is not None
        
        # Verify storage service was called for quarantine
        assert mock_storage_service.store_file.called
        call_args = mock_storage_service.store_file.call_args
        assert "quarantine/" in call_args[1]["object_name"]
        assert str(tenant_id) in call_args[1]["object_name"]
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self, security_service):
        """Test audit logging integration."""
        file = self.create_upload_file("audit_test.pdf", "application/pdf")
        content = b"%PDF-1.4\nTest content for audit"
        
        # Mock virus scanner
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        # Mock audit logger
        security_service.audit_logger.log_upload_attempt = MagicMock()
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4(),
            ip_address="192.168.1.100",
            user_agent="test-client"
        )
        
        # Verify audit logging was called
        assert security_service.audit_logger.log_upload_attempt.called
        call_args = security_service.audit_logger.log_upload_attempt.call_args
        assert call_args[1]["filename"] == "audit_test.pdf"
        assert call_args[1]["ip_address"] == "192.168.1.100"
        assert call_args[1]["user_agent"] == "test-client"
    
    @pytest.mark.asyncio
    async def test_security_validation_error_handling(self, security_service):
        """Test error handling in security validation."""
        file = self.create_upload_file("error_test.pdf", "application/pdf")
        content = b"Test content"
        
        # Mock virus scanner to raise exception
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.side_effect = Exception("Scanner error")
        
        # Mock audit logger
        security_service.audit_logger.log_security_violation = MagicMock()
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Should fail secure
        assert result.is_safe is False
        assert result.threat_level == ThreatLevel.SUSPICIOUS
        assert len(result.violations) > 0
        assert "Security validation failed" in result.violations[0]["message"]
        
        # Verify security violation was logged
        assert security_service.audit_logger.log_security_violation.called
    
    @pytest.mark.asyncio
    async def test_multiple_violation_types(self, security_service):
        """Test handling of multiple violation types in single file."""
        file = self.create_upload_file("multi_threat.pdf", "application/pdf")
        content = b"Not PDF MZ\x90\x00 executable signature <script>alert('xss')</script>"
        
        # Mock virus scanner to return clean (to test content validation)
        security_service.virus_scanner.scan_content = MagicMock()
        security_service.virus_scanner.scan_content.return_value = (True, "OK", {"status": "clean"})
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Should detect multiple violation types
        assert len(result.violations) > 1
        
        violation_types = [v["type"] for v in result.violations]
        assert SecurityViolationType.SUSPICIOUS_STRUCTURE in violation_types  # Invalid PDF structure
        assert SecurityViolationType.MALICIOUS_CONTENT in violation_types     # Embedded executable/script
    
    @pytest.mark.asyncio
    async def test_configuration_impact(self, settings, mock_storage_service):
        """Test how configuration changes impact security validation."""
        # Test with virus scanning disabled
        settings.upload.virus_scan_enabled = False
        security_service = SecurityService(settings, mock_storage_service)
        
        file = self.create_upload_file("config_test.pdf", "application/pdf")
        content = b"%PDF-1.4\nTest content"
        
        result = await security_service.validate_file_security(
            file=file,
            content=content,
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Should still validate content but skip virus scan
        assert "virus_scan" in result.scan_details
        assert result.scan_details["virus_scan"]["status"] == "disabled"
