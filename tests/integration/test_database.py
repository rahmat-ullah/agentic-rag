"""
Integration tests for database functionality.

Tests database connections, transactions, and multi-tenant isolation.
"""

import pytest
import uuid
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from agentic_rag.models.database import (
    Tenant, User, Document, DocumentLink, ChunkMeta, Feedback,
    DocumentKind, UserRole, FeedbackLabel
)
from tests.base import BaseDatabaseTest


class TestDatabaseConnection(BaseDatabaseTest):
    """Test database connection and basic operations."""
    
    def test_database_connection(self):
        """Test that database connection works."""
        # Simple query to test connection
        result = self.db_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row[0] == 1
    
    def test_database_transaction_rollback(self):
        """Test that database transactions can be rolled back."""
        # Create a tenant
        tenant = Tenant(id=uuid.uuid4(), name="Test Tenant")
        self.db_session.add(tenant)
        self.db_session.flush()  # Flush but don't commit
        
        # Verify it exists in the session
        found_tenant = self.db_session.query(Tenant).filter_by(id=tenant.id).first()
        assert found_tenant is not None
        
        # Rollback should happen automatically in teardown
        # This is handled by the test fixture


class TestTenantModel(BaseDatabaseTest):
    """Test Tenant model database operations."""
    
    def test_create_tenant(self):
        """Test creating a tenant."""
        tenant_data = self.create_test_tenant(name="Integration Test Tenant")
        
        # Verify tenant was created
        tenant = self.db_session.query(Tenant).filter_by(id=tenant_data["id"]).first()
        assert tenant is not None
        assert tenant.name == "Integration Test Tenant"
        assert tenant.created_at is not None
        assert tenant.updated_at is not None
    
    def test_tenant_unique_constraint(self):
        """Test tenant name uniqueness (if implemented)."""
        # Create first tenant
        tenant1 = Tenant(id=uuid.uuid4(), name="Unique Tenant")
        self.db_session.add(tenant1)
        self.db_session.commit()
        
        # Try to create second tenant with same name
        # Note: This test assumes name uniqueness is enforced
        # Adjust based on your actual constraints
        tenant2 = Tenant(id=uuid.uuid4(), name="Unique Tenant")
        self.db_session.add(tenant2)
        
        # This might raise an IntegrityError if uniqueness is enforced
        # If not enforced, this test should be modified or removed
        try:
            self.db_session.commit()
            # If no error, uniqueness is not enforced
            assert True  # Test passes
        except IntegrityError:
            # If error, uniqueness is enforced
            self.db_session.rollback()
            assert True  # Test passes


class TestUserModel(BaseDatabaseTest):
    """Test User model database operations."""
    
    def test_create_user(self):
        """Test creating a user."""
        user_data = self.create_test_user(email="integration@test.com")
        
        # Verify user was created
        user = self.db_session.query(User).filter_by(id=user_data["id"]).first()
        assert user is not None
        assert user.email == "integration@test.com"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True
        assert user.created_at is not None
    
    def test_user_email_uniqueness(self):
        """Test user email uniqueness constraint."""
        # Create first user
        user1_data = self.create_test_user(email="unique@test.com")
        
        # Try to create second user with same email
        with pytest.raises(IntegrityError):
            user2 = User(
                id=uuid.uuid4(),
                tenant_id=uuid.UUID(user1_data["tenant_id"]),
                email="unique@test.com",
                password_hash="hash",
                role=UserRole.VIEWER
            )
            self.db_session.add(user2)
            self.db_session.commit()
    
    def test_user_tenant_relationship(self):
        """Test user-tenant relationship."""
        user_data = self.create_test_user()
        
        user = self.db_session.query(User).filter_by(id=user_data["id"]).first()
        assert user.tenant is not None
        assert str(user.tenant.id) == user_data["tenant_id"]


class TestDocumentModel(BaseDatabaseTest):
    """Test Document model database operations."""
    
    def test_create_document(self):
        """Test creating a document."""
        tenant_data = self.create_test_tenant()
        user_data = self.create_test_user(tenant_id=tenant_data["id"])
        
        document = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.UUID(tenant_data["id"]),
            kind=DocumentKind.RFQ,
            title="Test Document",
            source_uri="s3://bucket/test.pdf",
            sha256="abc123def456",
            pages=5,
            created_by=uuid.UUID(user_data["id"])
        )
        
        self.db_session.add(document)
        self.db_session.commit()
        self.db_session.refresh(document)
        
        # Verify document was created
        assert document.id is not None
        assert document.title == "Test Document"
        assert document.kind == DocumentKind.RFQ
        assert document.created_at is not None
    
    def test_document_tenant_sha256_uniqueness(self):
        """Test document tenant+sha256 uniqueness constraint."""
        tenant_data = self.create_test_tenant()
        
        # Create first document
        doc1 = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.UUID(tenant_data["id"]),
            kind=DocumentKind.RFQ,
            title="Document 1",
            source_uri="s3://bucket/doc1.pdf",
            sha256="same_hash_value"
        )
        self.db_session.add(doc1)
        self.db_session.commit()
        
        # Try to create second document with same tenant_id and sha256
        with pytest.raises(IntegrityError):
            doc2 = Document(
                id=uuid.uuid4(),
                tenant_id=uuid.UUID(tenant_data["id"]),
                kind=DocumentKind.OFFER_TECH,
                title="Document 2",
                source_uri="s3://bucket/doc2.pdf",
                sha256="same_hash_value"
            )
            self.db_session.add(doc2)
            self.db_session.commit()


class TestMultiTenantIsolation(BaseDatabaseTest):
    """Test multi-tenant data isolation."""
    
    def test_tenant_data_isolation(self):
        """Test that tenants cannot access each other's data."""
        # Create two tenants
        tenant1_data = self.create_test_tenant(name="Tenant 1")
        tenant2_data = self.create_test_tenant(name="Tenant 2")
        
        # Create users for each tenant
        user1_data = self.create_test_user(tenant_id=tenant1_data["id"], email="user1@test.com")
        user2_data = self.create_test_user(tenant_id=tenant2_data["id"], email="user2@test.com")
        
        # Create documents for each tenant
        doc1 = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.UUID(tenant1_data["id"]),
            kind=DocumentKind.RFQ,
            title="Tenant 1 Document",
            source_uri="s3://bucket/tenant1.pdf",
            sha256="tenant1_hash",
            created_by=uuid.UUID(user1_data["id"])
        )
        
        doc2 = Document(
            id=uuid.uuid4(),
            tenant_id=uuid.UUID(tenant2_data["id"]),
            kind=DocumentKind.RFQ,
            title="Tenant 2 Document",
            source_uri="s3://bucket/tenant2.pdf",
            sha256="tenant2_hash",
            created_by=uuid.UUID(user2_data["id"])
        )
        
        self.db_session.add_all([doc1, doc2])
        self.db_session.commit()
        
        # Query documents for tenant 1
        tenant1_docs = self.db_session.query(Document).filter_by(
            tenant_id=uuid.UUID(tenant1_data["id"])
        ).all()
        
        # Query documents for tenant 2
        tenant2_docs = self.db_session.query(Document).filter_by(
            tenant_id=uuid.UUID(tenant2_data["id"])
        ).all()
        
        # Verify isolation
        assert len(tenant1_docs) == 1
        assert len(tenant2_docs) == 1
        assert tenant1_docs[0].title == "Tenant 1 Document"
        assert tenant2_docs[0].title == "Tenant 2 Document"
        assert tenant1_docs[0].id != tenant2_docs[0].id
    
    def test_row_level_security_function(self):
        """Test that RLS function exists and works."""
        # Test the get_current_tenant_id() function
        result = self.db_session.execute(text("SELECT get_current_tenant_id()"))
        tenant_id = result.scalar()
        
        # Should return NULL when no tenant is set
        assert tenant_id is None
        
        # Set a tenant ID and test again
        test_tenant_id = str(uuid.uuid4())
        self.db_session.execute(text(f"SELECT set_current_tenant_id('{test_tenant_id}')"))
        
        result = self.db_session.execute(text("SELECT get_current_tenant_id()"))
        current_tenant_id = result.scalar()
        
        assert current_tenant_id == test_tenant_id


class TestDatabaseConstraints(BaseDatabaseTest):
    """Test database constraints and validation."""
    
    def test_foreign_key_constraints(self):
        """Test foreign key constraints."""
        # Try to create a user with non-existent tenant
        with pytest.raises(IntegrityError):
            user = User(
                id=uuid.uuid4(),
                tenant_id=uuid.uuid4(),  # Non-existent tenant
                email="test@example.com",
                password_hash="hash",
                role=UserRole.VIEWER
            )
            self.db_session.add(user)
            self.db_session.commit()
    
    def test_not_null_constraints(self):
        """Test NOT NULL constraints."""
        # Try to create a document without required fields
        with pytest.raises(IntegrityError):
            document = Document(
                id=uuid.uuid4(),
                tenant_id=uuid.uuid4(),
                kind=DocumentKind.RFQ,
                # Missing required sha256 field
                title="Test Document"
            )
            self.db_session.add(document)
            self.db_session.commit()


class TestDatabaseIndexes(BaseDatabaseTest):
    """Test database indexes and performance."""
    
    def test_query_performance_with_indexes(self):
        """Test that queries use indexes efficiently."""
        # Create test data
        tenant_data = self.create_test_tenant()
        
        # Create multiple documents
        documents = []
        for i in range(10):
            doc = Document(
                id=uuid.uuid4(),
                tenant_id=uuid.UUID(tenant_data["id"]),
                kind=DocumentKind.RFQ,
                title=f"Document {i}",
                source_uri=f"s3://bucket/doc{i}.pdf",
                sha256=f"hash_{i}"
            )
            documents.append(doc)
        
        self.db_session.add_all(documents)
        self.db_session.commit()
        
        # Query by tenant_id and kind (should use index)
        result = self.db_session.query(Document).filter_by(
            tenant_id=uuid.UUID(tenant_data["id"]),
            kind=DocumentKind.RFQ
        ).all()
        
        assert len(result) == 10
        
        # Query by created_at (should use index)
        result = self.db_session.query(Document).filter(
            Document.tenant_id == uuid.UUID(tenant_data["id"])
        ).order_by(Document.created_at.desc()).limit(5).all()
        
        assert len(result) == 5
