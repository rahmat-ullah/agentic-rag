# Sprint 4: Contextual Retrieval & Three-Hop Search (2 weeks)

## Sprint Goal
Implement the sophisticated contextual retrieval system with three-hop search pattern (RFQ → Offers → Offer Chunks) and contextual chunking for improved search accuracy.

## Sprint Objectives
- Implement contextual chunking with local and global context
- Create three-hop retrieval pipeline (H1/H2/H3)
- Develop document linking system for RFQ-Offer relationships
- Implement LLM-based reranking for improved relevance
- Create contextual query enhancement

## Deliverables
- Contextual chunking implementation
- Three-hop retrieval pipeline
- Document linking and relationship management
- LLM reranking system
- Enhanced query processing with context
- Citation and provenance tracking

## User Stories

### Story 4-01: Contextual Chunking Implementation
**As a system, I want to create contextual chunks that include surrounding context so that search results are more accurate and meaningful.**

**File:** [4-01-contextual-chunking-story.md](4-01-contextual-chunking-story.md)

**Acceptance Criteria:**
- [ ] Local context extraction (neighbors, siblings)
- [ ] Global context extraction (document title, section trail)
- [ ] Context fusion with token limit management
- [ ] Contextual text used for embeddings
- [ ] Original text preserved for citations

**Story Points:** 13

### Story 4-02: Document Linking System
**As a user, I want to link RFQ documents with their corresponding offers so that searches can find related information across document types.**

**File:** [4-02-document-linking-story.md](4-02-document-linking-story.md)

**Acceptance Criteria:**
- [ ] Manual linking interface for RFQ-Offer relationships
- [ ] Automatic linking suggestions based on content similarity
- [ ] Confidence scoring for document links
- [ ] Link validation and quality assessment
- [ ] Bulk linking operations for efficiency

**Story Points:** 8

### Story 4-03: Three-Hop Retrieval Pipeline
**As a system, I want to implement the three-hop search pattern so that queries find the most relevant information by following document relationships.**

**File:** [4-03-three-hop-retrieval-story.md](4-03-three-hop-retrieval-story.md)

**Acceptance Criteria:**
- [ ] H1: RFQ anchor search implementation
- [ ] H2: Linked offer discovery via relationships
- [ ] H3: Targeted chunk retrieval from linked offers
- [ ] Configurable parameters for each hop
- [ ] Performance optimization for multi-hop queries

**Story Points:** 13

### Story 4-04: LLM-Based Reranking
**As a system, I want to use LLM-based reranking so that search results are ordered by true relevance rather than just vector similarity.**

**File:** [4-04-llm-reranking-story.md](4-04-llm-reranking-story.md)

**Acceptance Criteria:**
- [ ] LLM reranking integration with OpenAI
- [ ] Reranking prompts optimized for procurement content
- [ ] Scoring criteria including relevance, specificity, completeness
- [ ] Batch reranking for performance
- [ ] Fallback to vector ranking if LLM unavailable

**Story Points:** 8

### Story 4-05: Enhanced Query Processing
**As a user, I want my queries to be enhanced with domain context so that search results are more relevant to procurement scenarios.**

**File:** [4-05-enhanced-query-processing-story.md](4-05-enhanced-query-processing-story.md)

**Acceptance Criteria:**
- [ ] Query expansion with procurement terminology
- [ ] Context injection based on query type
- [ ] RFQ hint processing for targeted search
- [ ] Query intent classification
- [ ] Search strategy selection based on query analysis

**Story Points:** 5

## Dependencies
- Sprint 1: Foundation & Core Infrastructure
- Sprint 2: Document Ingestion Pipeline
- Sprint 3: Basic Retrieval & Vector Search

## Risks & Mitigation
- **Risk**: Contextual chunking complexity affecting performance
  - **Mitigation**: Incremental implementation, performance testing
- **Risk**: Three-hop search latency too high
  - **Mitigation**: Caching, parallel processing, optimization
- **Risk**: LLM reranking costs and latency
  - **Mitigation**: Selective reranking, caching, cost monitoring

## Technical Architecture

### Contextual Chunking Flow
1. Document parsing → span extraction
2. Local context collection (neighbors, siblings)
3. Global context collection (title, section trail)
4. Context fusion with token limits
5. Embedding generation for contextual text

### Three-Hop Search Flow
1. **H1**: Query → RFQ collection search → top RFQs
2. **H2**: RFQ IDs → document_link table → linked offers
3. **H3**: Offer IDs → offer collection search → ranked chunks

### Key Components
- **Context Builder**: Local and global context extraction
- **Link Manager**: Document relationship management
- **Hop Orchestrator**: Three-hop search coordination
- **Reranker**: LLM-based result reranking
- **Query Enhancer**: Context-aware query processing

## Definition of Done
- [ ] All user stories completed with acceptance criteria met
- [ ] Three-hop search tested with realistic document sets
- [ ] Contextual chunking improves search relevance
- [ ] Performance benchmarks met for complex queries
- [ ] Document linking accuracy validated
- [ ] LLM reranking improves result quality
