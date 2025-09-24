# User Story: Three-Hop Retrieval Pipeline

## Story Details
**As a system, I want to implement the three-hop search pattern so that queries find the most relevant information by following document relationships.**

**Story Points:** 13  
**Priority:** High  
**Sprint:** 4

## Acceptance Criteria
- [ ] H1: RFQ anchor search implementation
- [ ] H2: Linked offer discovery via relationships
- [ ] H3: Targeted chunk retrieval from linked offers
- [ ] Configurable parameters for each hop
- [ ] Performance optimization for multi-hop queries

## Tasks

### Task 1: H1 - RFQ Anchor Search
**Estimated Time:** 6 hours

**Description:** Implement the first hop that searches RFQ collection to find anchor documents.

**Implementation Details:**
- Create RFQ-specific search endpoint
- Implement query processing for RFQ content
- Add RFQ collection filtering and ranking
- Create anchor document selection logic
- Implement configurable result limits for H1

**Acceptance Criteria:**
- [ ] RFQ search returns relevant anchor documents
- [ ] Query processing optimized for RFQ content
- [ ] Filtering works correctly for RFQ collection
- [ ] Anchor selection logic provides good starting points
- [ ] Result limits configurable and enforced

### Task 2: H2 - Linked Offer Discovery
**Estimated Time:** 5 hours

**Description:** Implement the second hop that discovers linked offers from RFQ anchors.

**Implementation Details:**
- Create link traversal from RFQ to offers
- Implement confidence-based offer filtering
- Add offer ranking by link quality
- Create fallback for unlinked RFQs
- Implement configurable link confidence thresholds

**Acceptance Criteria:**
- [ ] Link traversal finds all connected offers
- [ ] Confidence filtering improves result quality
- [ ] Offer ranking prioritizes best matches
- [ ] Fallback handles unlinked RFQs gracefully
- [ ] Confidence thresholds configurable

### Task 3: H3 - Targeted Chunk Retrieval
**Estimated Time:** 7 hours

**Description:** Implement the third hop that retrieves specific chunks from linked offers.

**Implementation Details:**
- Create chunk-level search within offers
- Implement query-chunk relevance scoring
- Add chunk ranking and selection
- Create result aggregation and deduplication
- Implement final result ranking across all hops

**Acceptance Criteria:**
- [ ] Chunk search finds relevant content within offers
- [ ] Relevance scoring accurately ranks chunks
- [ ] Selection logic chooses best chunks
- [ ] Aggregation combines results effectively
- [ ] Final ranking provides optimal result order

### Task 4: Configurable Hop Parameters
**Estimated Time:** 4 hours

**Description:** Implement configurable parameters for each hop to optimize search behavior.

**Implementation Details:**
- Create parameter configuration system
- Implement hop-specific parameter validation
- Add parameter tuning interface
- Create parameter preset management
- Implement parameter impact monitoring

**Acceptance Criteria:**
- [ ] Parameters easily configurable per hop
- [ ] Validation prevents invalid parameter combinations
- [ ] Tuning interface allows optimization
- [ ] Presets provide common configurations
- [ ] Monitoring shows parameter impact

### Task 5: Performance Optimization
**Estimated Time:** 6 hours

**Description:** Optimize three-hop search performance for production workloads.

**Implementation Details:**
- Implement parallel processing for hops
- Add caching for intermediate results
- Create query optimization strategies
- Implement timeout and circuit breaker patterns
- Add performance monitoring and alerting

**Acceptance Criteria:**
- [ ] Parallel processing improves search speed
- [ ] Caching reduces redundant computations
- [ ] Query optimization minimizes processing time
- [ ] Timeout prevents hanging queries
- [ ] Performance monitoring identifies bottlenecks

## Dependencies
- Sprint 4: Document Linking System (for H2 link traversal)
- Sprint 3: Basic Search API (for search infrastructure)
- Sprint 3: ChromaDB Integration (for vector operations)

## Technical Considerations

### Three-Hop Search Flow
```
Query → H1: RFQ Search → H2: Link Traversal → H3: Chunk Retrieval → Results
```

### Hop Configuration
```yaml
three_hop_config:
  h1_rfq_search:
    max_results: 10
    similarity_threshold: 0.7
    boost_recent: true
  h2_link_discovery:
    min_confidence: 0.6
    max_offers_per_rfq: 5
    include_suggestions: false
  h3_chunk_retrieval:
    max_chunks_per_offer: 3
    chunk_similarity_threshold: 0.8
    enable_reranking: true
```

### Performance Requirements
- Complete three-hop search < 5 seconds (95th percentile)
- Support 50 concurrent three-hop queries
- Handle queries with 100+ linked documents
- Efficient memory usage during processing

### Result Structure
```json
{
  "query": "user query text",
  "three_hop_results": {
    "h1_anchors": [
      {
        "document_id": "rfq-uuid",
        "title": "RFQ Title",
        "score": 0.95,
        "linked_offers_count": 5
      }
    ],
    "h2_offers": [
      {
        "document_id": "offer-uuid",
        "title": "Offer Title",
        "link_confidence": 0.85,
        "chunks_found": 3
      }
    ],
    "h3_chunks": [
      {
        "chunk_id": "chunk-uuid",
        "content": "Relevant chunk content...",
        "score": 0.92,
        "source_path": "RFQ → Offer → Chunk"
      }
    ]
  },
  "search_time_ms": 2450,
  "hop_timings": {
    "h1": 450,
    "h2": 200,
    "h3": 1800
  }
}
```

## Error Handling

### Hop Failures
- H1 no results → Return empty result with explanation
- H2 no links → Fall back to direct offer search
- H3 no chunks → Return offer-level results
- Timeout → Return partial results with warning

### Performance Degradation
- High latency → Enable aggressive caching
- Memory pressure → Reduce batch sizes
- API failures → Implement graceful degradation

## Definition of Done
- [ ] All tasks completed with acceptance criteria met
- [ ] Three-hop search tested with realistic document sets
- [ ] Performance requirements validated
- [ ] Configuration system tested with various parameters
- [ ] Error handling tested for all failure scenarios
- [ ] Integration with existing search infrastructure complete

## Notes
- Consider implementing adaptive hop parameters based on query type
- Plan for search result explanation showing hop path
- Monitor hop performance to identify optimization opportunities
- Ensure three-hop search respects multi-tenant data isolation
