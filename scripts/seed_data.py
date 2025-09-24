#!/usr/bin/env python3
"""
Database seeding script for Agentic RAG System.

This script creates sample data for development and testing purposes.
"""

import sys
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.adapters.database import get_database_adapter
from agentic_rag.models.database import (
    Tenant, User, Document, DocumentLink, ChunkMeta, Feedback,
    DocumentKind, UserRole, FeedbackLabel
)


def create_sample_tenants(db):
    """Create sample tenants."""
    print("🏢 Creating sample tenants...")
    
    tenants = [
        {"name": "Acme Corporation", "id": uuid.uuid4()},
        {"name": "TechCorp Industries", "id": uuid.uuid4()},
        {"name": "Global Solutions Ltd", "id": uuid.uuid4()},
    ]
    
    created_tenants = []
    
    for tenant_data in tenants:
        with db.get_tenant_session(tenant_data["id"], "admin") as session:
            tenant = Tenant(
                id=tenant_data["id"],
                name=tenant_data["name"]
            )
            session.add(tenant)
            session.flush()
            created_tenants.append(tenant)
            print(f"✓ Created tenant: {tenant.name}")
    
    return created_tenants


def create_sample_users(db, tenants):
    """Create sample users for each tenant."""
    print("👥 Creating sample users...")
    
    user_templates = [
        {"email": "admin@{domain}", "role": UserRole.ADMIN},
        {"email": "analyst1@{domain}", "role": UserRole.ANALYST},
        {"email": "analyst2@{domain}", "role": UserRole.ANALYST},
        {"email": "viewer1@{domain}", "role": UserRole.VIEWER},
        {"email": "viewer2@{domain}", "role": UserRole.VIEWER},
    ]
    
    created_users = {}
    
    for tenant in tenants:
        domain = tenant.name.lower().replace(" ", "").replace(".", "") + ".com"
        tenant_users = []
        
        with db.get_tenant_session(tenant.id, "admin") as session:
            for user_template in user_templates:
                user = User(
                    tenant_id=tenant.id,
                    email=user_template["email"].format(domain=domain),
                    role=user_template["role"],
                    is_active=True
                )
                session.add(user)
                tenant_users.append(user)
            
            session.flush()
            created_users[tenant.id] = tenant_users
            print(f"✓ Created {len(tenant_users)} users for {tenant.name}")
    
    return created_users


def create_sample_documents(db, tenants, users):
    """Create sample documents for each tenant."""
    print("📄 Creating sample documents...")
    
    document_templates = [
        {"kind": DocumentKind.RFQ, "title": "Network Infrastructure RFQ", "pages": 25},
        {"kind": DocumentKind.RFQ, "title": "Cloud Migration RFQ", "pages": 18},
        {"kind": DocumentKind.RFP, "title": "Software Development RFP", "pages": 32},
        {"kind": DocumentKind.OFFER_TECH, "title": "Technical Proposal - Network", "pages": 45},
        {"kind": DocumentKind.OFFER_TECH, "title": "Technical Proposal - Cloud", "pages": 38},
        {"kind": DocumentKind.OFFER_COMM, "title": "Commercial Proposal - Network", "pages": 12},
        {"kind": DocumentKind.OFFER_COMM, "title": "Commercial Proposal - Cloud", "pages": 15},
        {"kind": DocumentKind.PRICING, "title": "Pricing Schedule - Network", "pages": 8},
        {"kind": DocumentKind.PRICING, "title": "Pricing Schedule - Cloud", "pages": 6},
    ]
    
    created_documents = {}
    
    for tenant in tenants:
        tenant_documents = []
        analyst_user = next(
            (u for u in users[tenant.id] if u.role == UserRole.ANALYST), 
            None
        )
        
        with db.get_tenant_session(tenant.id, "analyst") as session:
            for i, doc_template in enumerate(document_templates):
                # Generate unique SHA256 for each document
                content = f"{tenant.name}-{doc_template['title']}-{i}"
                sha256 = hashlib.sha256(content.encode()).hexdigest()
                
                document = Document(
                    tenant_id=tenant.id,
                    kind=doc_template["kind"],
                    title=doc_template["title"],
                    source_uri=f"s3://documents/{tenant.id}/{sha256}.pdf",
                    sha256=sha256,
                    version=1,
                    pages=doc_template["pages"],
                    created_by=analyst_user.id if analyst_user else None
                )
                session.add(document)
                tenant_documents.append(document)
            
            session.flush()
            created_documents[tenant.id] = tenant_documents
            print(f"✓ Created {len(tenant_documents)} documents for {tenant.name}")
    
    return created_documents


def create_sample_document_links(db, tenants, documents):
    """Create sample document links between RFQs and offers."""
    print("🔗 Creating sample document links...")
    
    created_links = {}
    
    for tenant in tenants:
        tenant_docs = documents[tenant.id]
        tenant_links = []
        
        # Find RFQs and offers
        rfqs = [d for d in tenant_docs if d.kind in [DocumentKind.RFQ, DocumentKind.RFP]]
        offers = [d for d in tenant_docs if d.kind in [
            DocumentKind.OFFER_TECH, DocumentKind.OFFER_COMM, DocumentKind.PRICING
        ]]
        
        with db.get_tenant_session(tenant.id, "analyst") as session:
            for rfq in rfqs:
                for offer in offers:
                    # Determine offer type
                    if offer.kind == DocumentKind.OFFER_TECH:
                        offer_type = "technical"
                    elif offer.kind == DocumentKind.OFFER_COMM:
                        offer_type = "commercial"
                    else:  # PRICING
                        offer_type = "pricing"
                    
                    # Generate realistic confidence score
                    base_confidence = 0.7
                    if "Network" in rfq.title and "Network" in offer.title:
                        confidence = base_confidence + 0.2
                    elif "Cloud" in rfq.title and "Cloud" in offer.title:
                        confidence = base_confidence + 0.15
                    else:
                        confidence = base_confidence - 0.1
                    
                    confidence = min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
                    
                    link = DocumentLink(
                        tenant_id=tenant.id,
                        rfq_id=rfq.id,
                        offer_id=offer.id,
                        offer_type=offer_type,
                        confidence=confidence
                    )
                    session.add(link)
                    tenant_links.append(link)
            
            session.flush()
            created_links[tenant.id] = tenant_links
            print(f"✓ Created {len(tenant_links)} document links for {tenant.name}")
    
    return created_links


def create_sample_chunks(db, tenants, documents):
    """Create sample chunk metadata."""
    print("📝 Creating sample chunk metadata...")
    
    created_chunks = {}
    
    for tenant in tenants:
        tenant_docs = documents[tenant.id]
        tenant_chunks = []
        
        with db.get_tenant_session(tenant.id, "analyst") as session:
            for doc in tenant_docs:
                # Create 3-5 chunks per document
                num_chunks = min(5, max(3, doc.pages // 5))
                
                for i in range(num_chunks):
                    chunk_hash = hashlib.sha256(
                        f"{doc.id}-chunk-{i}".encode()
                    ).hexdigest()[:16]
                    
                    chunk = ChunkMeta(
                        tenant_id=tenant.id,
                        document_id=doc.id,
                        page_from=i * (doc.pages // num_chunks) + 1,
                        page_to=min((i + 1) * (doc.pages // num_chunks), doc.pages),
                        section_path=[f"Section {i+1}", f"Subsection {i+1}.1"],
                        token_count=150 + (i * 50),
                        hash=chunk_hash,
                        is_table=(i % 3 == 0),  # Every 3rd chunk is a table
                        retired=False,
                        embedding_model_version="text-embedding-3-large",
                        embedding_created_at=datetime.utcnow() - timedelta(days=i)
                    )
                    session.add(chunk)
                    tenant_chunks.append(chunk)
            
            session.flush()
            created_chunks[tenant.id] = tenant_chunks
            print(f"✓ Created {len(tenant_chunks)} chunks for {tenant.name}")
    
    return created_chunks


def create_sample_feedback(db, tenants, documents, chunks, users):
    """Create sample feedback data."""
    print("💬 Creating sample feedback...")
    
    feedback_queries = [
        "What are the network requirements?",
        "Show me pricing for cloud migration",
        "Technical specifications for the proposal",
        "What is the timeline for implementation?",
        "Security requirements and compliance",
    ]
    
    created_feedback = {}
    
    for tenant in tenants:
        tenant_docs = documents[tenant.id]
        tenant_chunks = chunks[tenant.id]
        tenant_users = users[tenant.id]
        tenant_feedback = []
        
        analyst_users = [u for u in tenant_users if u.role == UserRole.ANALYST]
        
        with db.get_tenant_session(tenant.id, "analyst") as session:
            for i, query in enumerate(feedback_queries):
                if i < len(tenant_docs) and i < len(tenant_chunks):
                    feedback = Feedback(
                        tenant_id=tenant.id,
                        query=query,
                        rfq_id=tenant_docs[i % len(tenant_docs)].id,
                        offer_id=tenant_docs[(i + 1) % len(tenant_docs)].id,
                        chunk_id=tenant_chunks[i].id,
                        label=list(FeedbackLabel)[i % len(FeedbackLabel)],
                        notes=f"Sample feedback note for query: {query}",
                        created_by=analyst_users[0].id if analyst_users else None
                    )
                    session.add(feedback)
                    tenant_feedback.append(feedback)
            
            session.flush()
            created_feedback[tenant.id] = tenant_feedback
            print(f"✓ Created {len(tenant_feedback)} feedback entries for {tenant.name}")
    
    return created_feedback


def main():
    """Seed the database with sample data."""
    print("🌱 Seeding Database with Sample Data")
    print("=" * 50)
    
    db = get_database_adapter()
    
    try:
        # Create sample data
        tenants = create_sample_tenants(db)
        users = create_sample_users(db, tenants)
        documents = create_sample_documents(db, tenants, users)
        links = create_sample_document_links(db, tenants, documents)
        chunks = create_sample_chunks(db, tenants, documents)
        feedback = create_sample_feedback(db, tenants, documents, chunks, users)
        
        print("\n" + "=" * 50)
        print("📊 Summary:")
        print(f"   • {len(tenants)} tenants created")
        print(f"   • {sum(len(u) for u in users.values())} users created")
        print(f"   • {sum(len(d) for d in documents.values())} documents created")
        print(f"   • {sum(len(l) for l in links.values())} document links created")
        print(f"   • {sum(len(c) for c in chunks.values())} chunks created")
        print(f"   • {sum(len(f) for f in feedback.values())} feedback entries created")
        
        print("\n🎉 Database seeding completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Database seeding failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
