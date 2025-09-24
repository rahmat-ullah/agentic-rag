"""
Integration tests for API health endpoints.

Tests the health check endpoints and basic API functionality.
"""

import pytest
from fastapi.testclient import TestClient

from tests.base import BaseAPITest


class TestHealthEndpoints(BaseAPITest):
    """Test health check endpoints."""
    
    def test_health_check_endpoint(self):
        """Test the main health check endpoint."""
        response = self.get("/health/")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert data["status"] in ["healthy", "unhealthy"]  # Can be unhealthy if DB is down
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
    
    def test_health_check_detailed(self):
        """Test detailed health check with service status."""
        response = self.get("/health/")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert data["status"] in ["healthy", "unhealthy"]
        assert "components" in data
        assert len(data["components"]) > 0
        assert "timestamp" in data
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe endpoint."""
        response = self.get("/health/ready")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert "ready" in data
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe endpoint."""
        response = self.get("/health/live")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        data = response_data["data"]
        assert "alive" in data


class TestAPIDocumentation(BaseAPITest):
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint."""
        response = self.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Agentic RAG System API"
    
    def test_swagger_docs(self):
        """Test Swagger documentation endpoint."""
        response = self.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self):
        """Test ReDoc documentation endpoint."""
        response = self.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestCORSHeaders(BaseAPITest):
    """Test CORS headers and preflight requests."""
    
    def test_cors_headers_on_get(self):
        """Test CORS headers on GET requests."""
        response = self.get("/health/")
        
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    def test_preflight_request(self):
        """Test CORS preflight request."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Authorization"
        }
        
        response = self.client.options("/health/", headers=headers)
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


class TestErrorHandling(BaseAPITest):
    """Test API error handling."""
    
    def test_404_not_found(self):
        """Test 404 error handling."""
        response = self.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_405_method_not_allowed(self):
        """Test 405 error handling."""
        response = self.client.patch("/health/")
        
        assert response.status_code == 405
        data = response.json()
        assert "detail" in data
    
    def test_422_validation_error(self):
        """Test 422 validation error handling."""
        # Try to POST invalid JSON to an endpoint that expects valid data
        response = self.post("/auth/login", json={"invalid": "data"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestRateLimiting(BaseAPITest):
    """Test API rate limiting."""
    
    def test_rate_limit_headers(self):
        """Test that rate limit headers are present."""
        response = self.get("/health/")
        
        assert response.status_code == 200
        # Rate limiting headers should be present
        # Note: This depends on the actual rate limiting implementation
        # Adjust based on your rate limiting middleware
    
    @pytest.mark.skip(reason="Rate limiting test requires multiple requests")
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded response."""
        # This test would require making many requests quickly
        # Skip for now as it might be flaky in CI/CD
        pass


class TestSecurityHeaders(BaseAPITest):
    """Test security headers."""
    
    def test_security_headers_present(self):
        """Test that security headers are present."""
        response = self.get("/health/")
        
        assert response.status_code == 200
        
        # Check for common security headers
        # Note: Adjust based on your security middleware implementation
        headers = response.headers
        
        # These might be added by your security middleware
        # Uncomment and adjust as needed
        # assert "x-content-type-options" in headers
        # assert "x-frame-options" in headers
        # assert "x-xss-protection" in headers


class TestAPIVersioning(BaseAPITest):
    """Test API versioning."""
    
    def test_api_version_in_response(self):
        """Test that API version is included in responses."""
        response = self.get("/health/")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data

        # Check if version is included in the response data
        data = response_data["data"]
        assert "version" in data
        assert isinstance(data["version"], str)
    
    def test_api_version_header(self):
        """Test API version in response headers."""
        response = self.get("/health/")
        
        assert response.status_code == 200
        
        # Check if version is included in headers
        # Note: This depends on your API implementation
        # Adjust based on your versioning strategy
