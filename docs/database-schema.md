# Database Schema Documentation

## Overview

The Agentic RAG System uses a PostgreSQL database with Row-Level Security (RLS) for multi-tenant isolation. The schema supports the three-hop retrieval pattern: RFQ → Offers → Offer Chunks.

## Schema Design

### Core Tables

#### `tenant`
- **Purpose**: Multi-tenant isolation root table
- **Key Fields**: `id` (UUID), `name`, timestamps
- **RLS**: Tenant-scoped access

#### `app_user`
- **Purpose**: User management with role-based access
- **Key Fields**: `tenant_id`, `email`, `role` (admin/analyst/viewer)
- **RLS**: Tenant-scoped, role-based modification policies

#### `document`
- **Purpose**: Document metadata storage
- **Key Fields**: `tenant_id`, `kind` (RFQ/RFP/Tender/OfferTech/OfferComm/Pricing), `sha256`, `version`
- **Features**: Version control, deduplication via SHA256
- **RLS**: Tenant-scoped, role-based modification

#### `document_link`
- **Purpose**: RFQ-to-Offer relationships with confidence scoring
- **Key Fields**: `tenant_id`, `rfq_id`, `offer_id`, `offer_type`, `confidence`
- **Features**: Three-hop retrieval support, ML confidence tracking
- **RLS**: Tenant-scoped access

#### `chunk_meta`
- **Purpose**: Vector chunk metadata (vectors stored in ChromaDB)
- **Key Fields**: `tenant_id`, `document_id`, `hash`, `embedding_model_version`
- **Features**: Page ranges, section paths, table detection, retirement support
- **RLS**: Tenant-scoped access

#### `feedback`
- **Purpose**: User feedback for continuous learning
- **Key Fields**: `tenant_id`, `query`, `label` (up/down/edit/bad_link/good_link)
- **Features**: Query-result feedback loop
- **RLS**: Tenant-scoped, role-based modification

## Multi-Tenant Security

### Row-Level Security (RLS)

All tables implement RLS policies for tenant isolation:

```sql
-- Example policy
CREATE POLICY tenant_isolation_policy ON document
FOR ALL TO PUBLIC
USING (tenant_id = get_current_tenant_id());
```

### Role-Based Access Control

- **Admin**: Full access to tenant data and user management
- **Analyst**: Can create/modify documents and feedback
- **Viewer**: Read-only access to tenant data

### Context Functions

```sql
-- Set tenant context
SELECT set_current_tenant_id('tenant-uuid');
SELECT set_current_user_role('analyst');

-- Get current context
SELECT get_current_tenant_id();
SELECT get_current_user_role();
```

## Performance Optimization

### Indexes

#### Primary Indexes
- `idx_document_tenant_kind`: Fast document filtering by tenant and type
- `idx_chunk_meta_tenant_document`: Efficient chunk lookups
- `idx_document_link_tenant_rfq_type`: Three-hop retrieval optimization

#### Composite Indexes
- `idx_document_tenant_kind_created`: Time-based document queries
- `idx_chunk_meta_tenant_retired_embedding`: Active chunk filtering
- `idx_feedback_tenant_label_created`: Feedback analysis queries

#### Partial Indexes
- `idx_app_user_active_tenant`: Active users only
- `idx_chunk_meta_active_tenant`: Non-retired chunks only

## Data Validation

### Check Constraints
- SHA256 format validation: `validate_sha256(sha256)`
- Email format validation: `validate_email(email)`
- Confidence range: `confidence BETWEEN 0 AND 1`
- Positive values: `version > 0`, `pages > 0`, `token_count > 0`
- Page ranges: `page_from <= page_to`

### Enum Types
- `DocumentKind`: RFQ, RFP, Tender, OfferTech, OfferComm, Pricing
- `UserRole`: admin, analyst, viewer
- `FeedbackLabel`: up, down, edit, bad_link, good_link

## Utility Functions

### Document Management
```sql
-- Get next version for document
SELECT get_next_document_version(tenant_id, sha256);

-- Update document link confidence
SELECT update_document_link_confidence(tenant_id, rfq_id, offer_id, offer_type, confidence);
```

### Statistics
```sql
-- Document statistics by type
SELECT * FROM get_document_stats(tenant_id);

-- Chunk statistics by document type
SELECT * FROM get_chunk_stats(tenant_id);

-- Feedback summary
SELECT * FROM get_feedback_summary(tenant_id);
```

### Maintenance
```sql
-- Clean up old retired chunks
SELECT cleanup_retired_chunks(tenant_id, days_old);
```

## Migration Files

1. **001_initial_database_schema**: Core tables and relationships
2. **002_add_indexes_and_constraints**: Performance optimization
3. **003_add_row_level_security**: Multi-tenant isolation
4. **004_add_utility_functions**: Helper functions and validation

## Usage Examples

### Setting Up Tenant Context
```python
from agentic_rag.adapters.database import get_database_adapter

db = get_database_adapter()

# Work with tenant-scoped session
with db.get_tenant_session(tenant_id, "analyst") as session:
    # All queries automatically filtered by tenant
    documents = session.query(Document).all()
```

### Creating Documents
```python
document = Document(
    tenant_id=tenant_id,
    kind=DocumentKind.RFQ,
    title="Network Infrastructure RFQ",
    sha256="abc123...",
    version=1,
    pages=25
)
session.add(document)
```

### Linking Documents
```python
link = DocumentLink(
    tenant_id=tenant_id,
    rfq_id=rfq_document.id,
    offer_id=offer_document.id,
    offer_type="technical",
    confidence=0.85
)
session.add(link)
```

## Testing

### Schema Testing
```bash
python scripts/test_schema.py
```

### Sample Data
```bash
python scripts/seed_data.py
```

## Security Considerations

1. **Tenant Isolation**: RLS ensures complete data separation
2. **Role-Based Access**: Granular permissions by user role
3. **Data Validation**: Comprehensive constraints prevent invalid data
4. **Audit Trail**: Timestamps and user tracking for all modifications
5. **Secure Functions**: SECURITY DEFINER for controlled access

## Performance Considerations

1. **Optimized Indexes**: Covering common query patterns
2. **Partial Indexes**: Reduce index size for filtered queries
3. **Connection Pooling**: Efficient database connection management
4. **Query Optimization**: RLS policies designed for performance
5. **Maintenance Functions**: Automated cleanup of old data
