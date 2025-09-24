#!/usr/bin/env python3
"""
Database schema testing script for Agentic RAG System.

This script tests the database schema, RLS policies, and utility functions
to ensure proper multi-tenant isolation and functionality.
"""

import sys
import uuid
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.models.database import (
    Tenant, User, Document, DocumentLink, ChunkMeta, Feedback,
    DocumentKind, UserRole, FeedbackLabel
)


def test_tenant_isolation():
    """Test tenant isolation with RLS policies."""
    print("🔒 Testing tenant isolation...")
    
    db = get_database_adapter()
    
    # Create test tenants
    tenant1_id = uuid.uuid4()
    tenant2_id = uuid.uuid4()
    
    try:
        # Create tenants
        with db.get_tenant_session(tenant1_id, "admin") as session:
            tenant1 = Tenant(id=tenant1_id, name="Test Tenant 1")
            session.add(tenant1)
            session.flush()
        
        with db.get_tenant_session(tenant2_id, "admin") as session:
            tenant2 = Tenant(id=tenant2_id, name="Test Tenant 2")
            session.add(tenant2)
            session.flush()
        
        # Test tenant 1 can only see its own data
        with db.get_tenant_session(tenant1_id, "viewer") as session:
            tenants = session.query(Tenant).all()
            assert len(tenants) == 1
            assert tenants[0].id == tenant1_id
            print("✓ Tenant 1 can only see its own data")
        
        # Test tenant 2 can only see its own data
        with db.get_tenant_session(tenant2_id, "viewer") as session:
            tenants = session.query(Tenant).all()
            assert len(tenants) == 1
            assert tenants[0].id == tenant2_id
            print("✓ Tenant 2 can only see its own data")
        
        print("✅ Tenant isolation test passed")
        
    except Exception as e:
        print(f"❌ Tenant isolation test failed: {e}")
        return False
    
    return True


def test_role_based_access():
    """Test role-based access control."""
    print("👤 Testing role-based access control...")
    
    db = get_database_adapter()
    tenant_id = uuid.uuid4()
    
    try:
        # Create tenant
        with db.get_tenant_session(tenant_id, "admin") as session:
            tenant = Tenant(id=tenant_id, name="Role Test Tenant")
            session.add(tenant)
            session.flush()
        
        # Test admin can create users
        with db.get_tenant_session(tenant_id, "admin") as session:
            user = User(
                tenant_id=tenant_id,
                email="admin@test.com",
                role=UserRole.ADMIN
            )
            session.add(user)
            session.flush()
            print("✓ Admin can create users")
        
        # Test viewer cannot create users (should fail)
        try:
            with db.get_tenant_session(tenant_id, "viewer") as session:
                user = User(
                    tenant_id=tenant_id,
                    email="viewer@test.com",
                    role=UserRole.VIEWER
                )
                session.add(user)
                session.flush()
            print("❌ Viewer should not be able to create users")
            return False
        except Exception:
            print("✓ Viewer correctly blocked from creating users")
        
        # Test analyst can create documents
        with db.get_tenant_session(tenant_id, "analyst") as session:
            document = Document(
                tenant_id=tenant_id,
                kind=DocumentKind.RFQ,
                title="Test RFQ",
                sha256="a" * 64,
                version=1
            )
            session.add(document)
            session.flush()
            print("✓ Analyst can create documents")
        
        print("✅ Role-based access control test passed")
        
    except Exception as e:
        print(f"❌ Role-based access control test failed: {e}")
        return False
    
    return True


def test_utility_functions():
    """Test database utility functions."""
    print("🔧 Testing utility functions...")
    
    db = get_database_adapter()
    tenant_id = uuid.uuid4()
    
    try:
        # Create test data
        with db.get_tenant_session(tenant_id, "admin") as session:
            tenant = Tenant(id=tenant_id, name="Utility Test Tenant")
            session.add(tenant)
            
            # Create documents
            doc1 = Document(
                tenant_id=tenant_id,
                kind=DocumentKind.RFQ,
                title="Test RFQ 1",
                sha256="b" * 64,
                version=1,
                pages=10
            )
            doc2 = Document(
                tenant_id=tenant_id,
                kind=DocumentKind.OFFER_TECH,
                title="Test Offer 1",
                sha256="c" * 64,
                version=1,
                pages=15
            )
            session.add_all([doc1, doc2])
            session.flush()
            
            # Create chunks
            chunk1 = ChunkMeta(
                tenant_id=tenant_id,
                document_id=doc1.id,
                token_count=100,
                hash="chunk1hash",
                is_table=False
            )
            chunk2 = ChunkMeta(
                tenant_id=tenant_id,
                document_id=doc2.id,
                token_count=150,
                hash="chunk2hash",
                is_table=True
            )
            session.add_all([chunk1, chunk2])
            session.flush()
        
        # Test document statistics
        with db.get_tenant_session(tenant_id, "viewer") as session:
            stats = db.get_document_stats(session, tenant_id)
            print(f"✓ Document stats: {len(stats)} document types")
            
            chunk_stats = db.get_chunk_stats(session, tenant_id)
            print(f"✓ Chunk stats: {len(chunk_stats)} document types with chunks")
        
        print("✅ Utility functions test passed")
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False
    
    return True


def test_constraints_and_validation():
    """Test database constraints and validation."""
    print("✅ Testing constraints and validation...")
    
    db = get_database_adapter()
    tenant_id = uuid.uuid4()
    
    try:
        # Create tenant
        with db.get_tenant_session(tenant_id, "admin") as session:
            tenant = Tenant(id=tenant_id, name="Constraint Test Tenant")
            session.add(tenant)
            session.flush()
        
        # Test invalid SHA256 (should fail)
        try:
            with db.get_tenant_session(tenant_id, "admin") as session:
                document = Document(
                    tenant_id=tenant_id,
                    kind=DocumentKind.RFQ,
                    title="Invalid SHA256 Doc",
                    sha256="invalid_sha256",  # Invalid format
                    version=1
                )
                session.add(document)
                session.flush()
            print("❌ Invalid SHA256 should be rejected")
            return False
        except Exception:
            print("✓ Invalid SHA256 correctly rejected")
        
        # Test invalid email (should fail)
        try:
            with db.get_tenant_session(tenant_id, "admin") as session:
                user = User(
                    tenant_id=tenant_id,
                    email="invalid_email",  # Invalid format
                    role=UserRole.VIEWER
                )
                session.add(user)
                session.flush()
            print("❌ Invalid email should be rejected")
            return False
        except Exception:
            print("✓ Invalid email correctly rejected")
        
        # Test confidence range validation
        with db.get_tenant_session(tenant_id, "admin") as session:
            # Create valid documents first
            rfq_doc = Document(
                tenant_id=tenant_id,
                kind=DocumentKind.RFQ,
                title="Test RFQ",
                sha256="d" * 64,
                version=1
            )
            offer_doc = Document(
                tenant_id=tenant_id,
                kind=DocumentKind.OFFER_TECH,
                title="Test Offer",
                sha256="e" * 64,
                version=1
            )
            session.add_all([rfq_doc, offer_doc])
            session.flush()
            
            # Test valid confidence
            link = DocumentLink(
                tenant_id=tenant_id,
                rfq_id=rfq_doc.id,
                offer_id=offer_doc.id,
                offer_type="technical",
                confidence=0.85
            )
            session.add(link)
            session.flush()
            print("✓ Valid confidence accepted")
        
        print("✅ Constraints and validation test passed")
        
    except Exception as e:
        print(f"❌ Constraints and validation test failed: {e}")
        return False
    
    return True


def main():
    """Run all database schema tests."""
    print("🧪 Running Database Schema Tests")
    print("=" * 50)
    
    tests = [
        test_tenant_isolation,
        test_role_based_access,
        test_utility_functions,
        test_constraints_and_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
