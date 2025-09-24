# Agentic RAG System — Complete Project Documentation (with Pseudocode)

> **Purpose**
> A production-grade, **agentic Retrieval-Augmented Generation (RAG)** system that:
>
> * Continuously **learns from users** (feedback, edits, link confirmations).
> * Uses **IBM Granite-Docling-258M** to parse/convert documents (PDFs, images, Office).
> * Uses **OpenAI embedding models** for vectorization.
> * Stores vectors in **ChromaDB** and system truth in **PostgreSQL**.
> * Implements **contextual retrieval** (contextual chunks & contextual queries).
> * Keeps **two logical vector partitions**:
>
>   1. RFQ/RFP/Tender
>   2. OfferTechnical / OfferCommercial / Pricing
> * Query flow: find relevant RFQ/RFP → follow linked offers → retrieve relevant chunks → answer with citations.

---

## 0. Executive Summary

This system ingests procurement artifacts (RFQs/RFPs/Tenders) and their corresponding offers (technical, commercial, pricing). During inference, it **anchors** on the governing RFQ first, then **filters** to the correct paired offers, then retrieves **contextual chunks** for high-precision answers with citations, while collecting user feedback to improve future results.

---

## 1. Architecture Overview

### 1.1 High-Level Components

* **API Gateway** (REST/GraphQL, e.g., FastAPI/Express)
* **Ingestion Service** (file uploads, Granite-Docling parsing, chunking, embeddings, Chroma upsert, Postgres writes)
* **Retrieval Service** (3-hop retrieval: RFQ → Offers → Offer Chunks; rerank; synthesis)
* **Agent Orchestrator** (Planner, Retriever Tool, Link-Fixer, Redactor, Pricer)
* **Feedback Service** (thumbs up/down, edits, re-embed, link confirmation)
* **Storage**:

  * **PostgreSQL** (documents, linkage, metadata, users, feedback)
  * **ChromaDB** (two collections: `rfq_collection`, `offer_collection`)
  * **Object store** (original files, thumbnails)
* **Observability** (metrics, traces, logs, dashboards)

### 1.2 Data Flow

1. **Upload** → Granite-Docling → structured JSON
2. **Contextual chunking** → build contextual text per span
3. **OpenAI embeddings** → upsert to Chroma (+ Postgres metadata)
4. **Linking** → RFQ ↔ Offer mappings (manual + auto)
5. **Query** → RFQ retrieval → filter offers → offer-chunk retrieval → LLM synthesis
6. **Feedback** → update links/embeddings/weights; learn continuously

---

## 2. Environments & Dependencies

* **Runtime**: Python 3.11+ (services), Node 20+ (optional gateway/UI)
* **LLM/Embeddings**: OpenAI embeddings (`text-embedding-3-*` or successor)
* **Parsing**: Granite-Docling-258M & Docling pipeline (local/serving container)
* **Vector DB**: ChromaDB (persistent mode)
* **Relational DB**: PostgreSQL 14+
* **Message Queue**: Redis/RabbitMQ (async ingestion pipeline)
* **Storage**: S3-compatible bucket (MinIO/AWS S3)
* **Infra**: Docker Compose / Kubernetes (Helm)
* **Observability**: Prometheus + Grafana, OpenTelemetry tracing

---

## 3. Security, Tenancy, Compliance

* **Multi-tenancy**: `tenant_id` in every table and Chroma metadata filter.
* **Row-level Security (RLS)** in Postgres; **namespaces/collections per tenant** or global with strict metadata filters in Chroma.
* **Secrets** via Vault/KMS; API keys never logged.
* **PII/Secrets Redaction** pre-persistence; role-based masking for pricing.
* **Audit trail**: store chunk IDs, model versions, prompt hashes per answer.

---

## 4. Data Model (PostgreSQL)

> *Note*: Minimal DDL shown; adapt to your conventions.

```sql
-- Tenancy & Users
CREATE TABLE tenant (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE app_user (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant(id),
  email TEXT UNIQUE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('admin','analyst','viewer')),
  created_at TIMESTAMP DEFAULT now()
);

-- Document registry
CREATE TYPE doc_kind AS ENUM ('RFQ','RFP','Tender','OfferTech','OfferComm','Pricing');

CREATE TABLE document (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant(id),
  kind doc_kind NOT NULL,
  title TEXT,
  source_uri TEXT,                 -- object store URI
  sha256 CHAR(64) NOT NULL,
  version INT DEFAULT 1,
  pages INT,
  created_by UUID REFERENCES app_user(id),
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);
CREATE UNIQUE INDEX ON document(tenant_id, sha256);

-- RFQ <-> Offer links (one RFQ can have multiple offers)
CREATE TABLE document_link (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant(id),
  rfq_id UUID NOT NULL REFERENCES document(id),
  offer_id UUID NOT NULL REFERENCES document(id),
  offer_type TEXT NOT NULL CHECK (offer_type IN ('technical','commercial','pricing')),
  confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX ON document_link(tenant_id, rfq_id, offer_type);

-- Chunk metadata (vectors live in Chroma)
CREATE TABLE chunk_meta (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant(id),
  document_id UUID NOT NULL REFERENCES document(id),
  page_from INT, page_to INT,
  section_path TEXT[],             -- ["Section 2", "Scope", "2.1 Electrical"]
  token_count INT,
  hash CHAR(64),                   -- dedupe per version
  is_table BOOLEAN DEFAULT FALSE,
  retired BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT now()
);
CREATE INDEX ON chunk_meta(tenant_id, document_id);
CREATE UNIQUE INDEX ON chunk_meta(tenant_id, hash) WHERE retired = FALSE;

-- Feedback / Learning
CREATE TABLE feedback (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenant(id),
  query TEXT NOT NULL,
  rfq_id UUID REFERENCES document(id),
  offer_id UUID REFERENCES document(id),
  chunk_id UUID REFERENCES chunk_meta(id),
  label TEXT NOT NULL CHECK (label IN ('up','down','edit','bad_link','good_link')),
  notes TEXT,
  created_by UUID REFERENCES app_user(id),
  created_at TIMESTAMP DEFAULT now()
);
```

---

## 5. Vector Store Layout (ChromaDB)

* **Collections**

  * `rfq_collection`: RFQ/RFP/Tender
  * `offer_collection`: OfferTech, OfferComm, Pricing

* **Common metadata fields**
  `tenant_id, document_id, section_path, page_span, version, hash`
  Plus for offers: `offer_type ∈ {technical, commercial, pricing}`, `rfq_id` (if linked), `link_confidence`.

---

## 6. Contextual Retrieval Design

### 6.1 Contextual Chunks

Each chunk stores:

* `C_text` (atomic span content for display/citation)
* `LocalCtx` (neighbors, sibling headings)
* `GlobalCtx` (document title, section trail, key definitions)
* `ctx_text = fuse(GlobalCtx, LocalCtx, C_text, limit=N tokens)` → **embedded**
* `term_index = terms(C_text + headings + key terms)` → lightweight lexical support

### 6.2 Retrieval Plan (Three Hops)

1. **H1 — RFQ Anchor**: contextual query → search `rfq_collection` → k₁ results → rerank → top RFQs
2. **H2 — Follow Links**: from chosen RFQs, pull linked offers via Postgres (`document_link`) with `confidence ≥ τ`
3. **H3 — Offer Chunks**: constrained search in `offer_collection` (by `offer_id` set) → k₂ results → LLM rerank → answer pack

---

## 7. API Design (Minimal)

### 7.1 Endpoints

* **POST `/ingest`** → `{document_id}`

  * form-data: `file`, `kind ∈ {RFQ,RFP,Tender,OfferTech,OfferComm,Pricing}`, `tenant_id`, `rfq_ref?`
* **POST `/query`** → `{answer, citations[], debug?}`

  * json: `tenant_id, question, rfq_hint?, types?`
* **POST `/feedback`** → `{ok:true}`

  * json: `tenant_id, query_id, label, notes?, rfq_id?, offer_id?, chunk_id?`
* **GET  `/documents/:id`** → doc metadata
* **GET  `/links?rfq_id=...`** → linked offers (with confidences)

### 7.2 Response Citation Schema

```json
{
  "answer": "…",
  "citations": [
    {
      "doc_id": "UUID",
      "doc_title": "RFP-XYZ 2025",
      "doc_kind": "OfferTech",
      "section": "2.1 Scope",
      "pages": "12-13",
      "chunk_id": "UUID",
      "quote": "…"
    }
  ]
}
```

---

## 8. Agent Orchestration

* **Planner**: decides the tool chain (analyze vs compare vs summarize vs table extraction).
* **Retriever Tool**: implements H1/H2/H3 with tunable `k` and thresholds.
* **Link-Fixer**: proposes/updates `document_link` when confidence low or flagged.
* **Redactor**: masks PII/secrets by role.
* **Pricer**: normalizes pricing tables, currency, totals.

---

## 9. Ingestion Pipeline (Pseudocode)

```pseudo
FUNCTION ingest_document(file, kind, tenant_id, rfq_ref=None, user_id=None):
    # 1) Store original file
    uri = object_store.put(file)
    sha = sha256(file.bytes)

    # 2) Idempotency
    IF postgres.document_exists(tenant_id, sha):
        RETURN existing_document_id

    # 3) Create document row
    doc_id = postgres.insert_document(
        tenant_id=tenant_id, kind=kind, title=guess_title(file),
        source_uri=uri, sha256=sha, created_by=user_id
    )

    # 4) Parse with Granite-Docling
    parsed = docling.process(file.bytes)      # returns structured blocks, pages, layout

    # 5) Build contextual chunks
    chunks = []
    FOR span IN iterate_spans(parsed):
        C_text = clean(span.text)
        LocalCtx = collect_neighbors(span)
        GlobalCtx = collect_global_context(parsed, span)
        ctx_text = fuse(GlobalCtx, LocalCtx, C_text, limit_tokens=1024)
        chunk = {
            "text": C_text,
            "ctx_text": ctx_text,
            "meta": {
                "page_span": span.page_span,
                "section_path": span.section_trail,
                "is_table": span.is_table
            }
        }
        chunks.append(chunk)

    # 6) Deduplicate per document-version
    unique_chunks = dedupe_by_hash(chunks)

    # 7) Embed contextual text
    embeddings = openai.embed([c.ctx_text for c in unique_chunks])

    # 8) Persist chunk_meta (Postgres) and vectors (Chroma)
    col = choose_collection(kind)  # rfq_collection or offer_collection
    FOR c, emb IN zip(unique_chunks, embeddings):
        chunk_id = postgres.insert_chunk_meta(
            tenant_id, doc_id, c.meta.page_span, c.meta.section_path, token_count(c.ctx_text), hash(c.text), c.meta.is_table
        )
        chroma[col].add(
            id=chunk_id,
            embedding=emb,
            metadata={
                "tenant_id": tenant_id,
                "document_id": doc_id,
                "section_path": c.meta.section_path,
                "page_span": c.meta.page_span,
                "version": 1,
                "hash": hash(c.text),
                "offer_type": kind_to_offer_type(kind),
                "rfq_id": rfq_ref
            },
            document=c.text
        )

    # 9) Optional auto-linking (if kind is Offer*)
    IF kind IN {OfferTech, OfferComm, Pricing}:
        rfq_id, confidence = try_autolink(parsed, tenant_id, rfq_ref)
        IF rfq_id:
            postgres.insert_document_link(tenant_id, rfq_id, doc_id, offer_type=kind_to_offer_type(kind), confidence)

    RETURN doc_id
```

---

## 10. Retrieval & Answering (Pseudocode)

```pseudo
FUNCTION contextual_query(user_question, rfq_hint=None):
    # Expand with fields likely needed for procurement answers
    base = user_question
    IF rfq_hint:
        base += "\nRFQ hint: " + rfq_hint
    base += "\nRequired: scope, delivery, compliance, pricing terms if applicable."
    RETURN base

FUNCTION retrieve_answer(tenant_id, question, rfq_hint=None, types=None):
    q = contextual_query(question, rfq_hint)

    # H1: Find governing RFQ/RFP/Tender
    rfq_hits = chroma["rfq_collection"].query(
        query_text=q,
        n_results=12,
        where={"tenant_id": tenant_id, "retired": False}
    )
    top_rfq_ids = rerank_llm(rfq_hits, purpose="Select governing RFQs").top_ids(3)

    # H2: Follow links to offers with confidence threshold
    offer_rows = postgres.query(
        "SELECT offer_id, offer_type, confidence FROM document_link "
        "WHERE tenant_id = $1 AND rfq_id = ANY($2) AND confidence >= $3",
        [tenant_id, top_rfq_ids, 0.6]
    )
    offer_ids_by_type = group_by_type(offer_rows)   # {technical:[...], commercial:[...], pricing:[...]}

    # Filter to requested types, if any
    allowed_types = types OR ["technical","commercial","pricing"]
    chosen_offer_ids = flatten(offer_ids_by_type[t] FOR t IN allowed_types)

    IF empty(chosen_offer_ids):
        RETURN fallback_answer("No linked offers found. Please confirm linkage.", rfq_candidates=top_rfq_ids)

    # H3: Retrieve best chunks from linked offers
    offer_hits = chroma["offer_collection"].query(
        query_text=q,
        n_results=40,
        where={"tenant_id": tenant_id, "document_id": {"$in": chosen_offer_ids}}
    )

    # Rerank with LLM judge
    top_chunks = rerank_llm(
        offer_hits,
        purpose="Prefer grounded, specific, section-rich chunks; include numerics when asked"
    ).top_k(12)

    # Construct Context Pack with citations
    context_pack = []
    FOR ch IN top_chunks:
        meta = ch.metadata
        doc = postgres.get_document(meta.document_id)
        context_pack.append({
            "doc_id": doc.id,
            "doc_kind": doc.kind,
            "doc_title": doc.title,
            "section": join(meta.section_path, " > "),
            "pages": as_page_span(meta.page_span),
            "chunk_id": ch.id,
            "quote": ch.document
        })

    # Synthesize with strict citation template
    answer = llm_generate(
        system="""
            You are a compliance-minded procurement analyst.
            Use ONLY provided context. Cite like [Doc:Title §Section p.Pages].
            If unsure, say so. Show assumptions and gaps separately.
        """,
        user=question,
        context=context_pack
    )

    RETURN {
        "answer": answer,
        "citations": context_pack
    }
```

---

## 11. Feedback & Learning (Pseudocode)

```pseudo
FUNCTION submit_feedback(tenant_id, query_id, label, notes=None, rfq_id=None, offer_id=None, chunk_id=None, user_id=None):
    postgres.insert_feedback(tenant_id, query_id, label, notes, rfq_id, offer_id, chunk_id, user_id)

    SWITCH(label):
        CASE "good_link":
            IF rfq_id AND offer_id:
                increase_link_confidence(tenant_id, rfq_id, offer_id, delta=0.1)
        CASE "bad_link":
            IF rfq_id AND offer_id:
                decrease_link_confidence(tenant_id, rfq_id, offer_id, delta=0.2)
        CASE "edit":
            IF chunk_id:
                # user provided corrected text elsewhere; re-embed and upsert
                new_text = pull_user_edit(query_id, chunk_id)
                new_ctx_text = rebuild_context_text(chunk_id, new_text)
                new_emb = openai.embed([new_ctx_text])[0]
                chroma.update_embedding(collection="offer_collection", id=chunk_id, embedding=new_emb, document=new_text)
                postgres.bump_chunk_version(chunk_id)
        CASE "up":
            # Optional: boost learning-to-rank weights for chunk_id
            boost_chunk_weight(chunk_id, +0.05)
        CASE "down":
            penalize_chunk_weight(chunk_id, -0.05)

    RETURN {"ok": true}
```

---

## 12. Prompts (Synthesis & Reranking)

**LLM Reranker Prompt (sketch):**

```
Given: user query Q and candidate chunks with metadata.
Score each candidate from 0–5 on:
(1) Direct relevance to Q,
(2) Specificity (numbers, clauses, named sections),
(3) Completeness (does it answer the Q alone?),
(4) Reliability signals (is it from the matched RFQ's offers? has section/page?).

Return top-N ids with scores and short reasons.
```

**Answer Synthesis Prompt (sketch):**

```
System: Compliance-minded analyst. Use ONLY context. Cite each bullet as [Doc:Title §Section p.Pages].
If pricing is included, state currency, date/version. List gaps/assumptions.

User: {{question}}

Context: {{JSON lines of {doc_title, doc_kind, section, pages, quote}}}
```

---

## 13. Directory Layout (Reference)

```
/agentic-rag/
  api/
    main.py                 # REST endpoints
    auth.py                 # JWT / OAuth2 (optional)
    schemas.py              # Pydantic DTOs
  services/
    ingest.py               # ingestion pipeline
    retrieval.py            # H1/H2/H3 logic
    feedback.py             # feedback handlers
    orchestrator.py         # Planner, tools
    redactor.py
    pricer.py
  adapters/
    docling_client.py       # Granite-Docling wrapper
    openai_client.py        # embeddings & completions
    chroma_client.py
    postgres.py
    s3_client.py
  models/
    ddl.sql                 # DB schema
  ops/
    docker-compose.yml
    k8s/helm/               # charts
    grafana/                # dashboards
  tests/
    e2e/
    unit/
  README.md
```

---

## 14. Configuration

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8080

db:
  postgres_url: ${POSTGRES_URL}
  pool_size: 10

vector:
  chroma_path: /var/chroma
  rfq_collection: rfq_collection
  offer_collection: offer_collection

providers:
  openai_api_key: ${OPENAI_API_KEY}
  embedding_model: text-embedding-3-large  # or current best

docling:
  endpoint: http://docling:9000  # or local library path
  max_pages: 2000

retrieval:
  k1_rfq: 12
  k1_rerank_top: 3
  k3_offer_candidates: 40
  k3_rerank_top: 12
  link_conf_threshold: 0.6

security:
  enable_rls: true
  redact_roles: ["viewer"]
```

---

## 15. Deployment

### 15.1 Docker Compose (outline)

* Services: `api`, `ingestion-worker`, `docling`, `chroma`, `postgres`, `minio`, `prometheus`, `grafana`.

### 15.2 Kubernetes

* StatefulSets for `chroma`, `postgres`, `minio`.
* HPAs on `api`, `ingestion-worker`, `retrieval`.
* Nightly backup CronJobs (Postgres dump & Chroma snapshot).

---

## 16. Observability & Testing

* **Metrics**: ingestion latency, embed throughput, query p50/p95, hit\@k, citation precision, cost per query.
* **Tracing**: upload→parse→embed→query→rerank→synthesize.
* **Tests**:

  * Unit: chunker, reranker rubric, redactor.
  * Golden-set E2E: Q→A with expected citations; regression watch.
  * Load: concurrent queries with large offer sets.

---

## 17. Performance & Scaling Tips

* Batch embeddings (512–1024 items); parallel shards per document.
* Pre-warm **active RFQs** and **current offers**; pin in Chroma cache.
* Add **hybrid retrieval** (lexical + vector) for names/numbers precision.
* Promote **table-aware retrieval** (header signatures) for pricing questions.
* Snapshot **RFQ↔Offer** links monthly for auditability.

---

## 18. Risk Management & Safeguards

* **Mismatched links** → show RFQ candidates and request confirmation.
* **Outdated pricing** → mark with version date; require role to reveal.
* **OCR noise** → accept user edits → re-embed; store quality score per page.
* **Hallucination** → strict “context-only” prompt; refuse without evidence.

---

## 19. Example End-to-End Scenarios

### 19.1 “What are delivery terms for RFQ-ACME-2025?”

1. H1 finds RFQ-ACME-2025 sections on Terms & Delivery
2. H2 pulls linked OfferTech + OfferComm
3. H3 retrieves chunks; synthesis lists terms, incoterms, timelines with citations.

### 19.2 “Total cost for Option-B per unit?”

* Constrain to Pricing offers; extract table rows; normalize currency; cite line items + pages.

---

## 20. Migration & Backfills

* **Initial import**: run bulk parse + chunk + embed.
* **Versioning**: new document versions soft-retire old chunks (keep for audit).
* **Re-embedding**: when upgrading embedding model, re-embed incrementally (most viewed docs first).

---

## 21. Glossary

* **Contextual Chunk**: primary text plus local/global context used for embedding.
* **Context Pack**: curated set of top chunks with rich metadata for synthesis.
* **H1/H2/H3**: three-hop retrieval pipeline.
* **Offer Types**: `technical`, `commercial`, `pricing`.

---

# Appendix A — Pseudocode Snippets (Copy-Paste Ready)

### A.1 API: `/ingest`

```pseudo
POST /ingest (multipart form):

INPUT:
  tenant_id (string UUID)
  kind (enum: RFQ|RFP|Tender|OfferTech|OfferComm|Pricing)
  rfq_ref (optional string or UUID)
  file (binary)

PROCESS:
  doc_id = ingest_document(file, kind, tenant_id, rfq_ref, user_id=requester)

OUTPUT:
  200 OK { "document_id": doc_id }
ERRORS:
  400 invalid_kind | file_missing
  409 document_exists
  500 parse_failed | embed_failed | storage_failed
```

### A.2 API: `/query`

```pseudo
POST /query (application/json):

INPUT:
  tenant_id (UUID)
  question (string)
  rfq_hint (string, optional)
  types (array of "technical"|"commercial"|"pricing", optional)

PROCESS:
  result = retrieve_answer(tenant_id, question, rfq_hint, types)

OUTPUT:
  200 OK { "answer": "...", "citations": [ ... ], "debug": { optional } }
ERRORS:
  404 no_linked_offers
  422 ambiguous_documents
  500 retrieval_failed | synthesis_failed
```

### A.3 API: `/feedback`

```pseudo
POST /feedback

INPUT:
  tenant_id, query_id, label ∈ {up,down,edit,bad_link,good_link}
  notes?, rfq_id?, offer_id?, chunk_id?

PROCESS:
  submit_feedback(...)

OUTPUT:
  200 OK { "ok": true }
```

---

# Appendix B — Contextual Chunking Pseudocode

```pseudo
FUNCTION iterate_spans(parsed_doc):
    # yield spans: text block, table cell group, list item, header block
    ...

FUNCTION collect_neighbors(span):
    prev_headings = headings(previous(span))
    next_headings = headings(next(span))
    siblings = sibling_headings(span)
    RETURN normalize(prev_headings + next_headings + siblings)

FUNCTION collect_global_context(doc, span):
    title = doc.title OR guess_title(doc)
    trail = " > ".join(span.section_trail)
    glossary = extract_key_definitions(doc)  # optional, cached per doc
    RETURN normalize(title + " | " + trail + " | " + glossary)

FUNCTION fuse(global, local, core, limit_tokens):
    text = global + "\n" + local + "\n" + core
    RETURN truncate_to_token_limit(text, limit_tokens)
```

---

# Appendix C — Reranker & Synthesis Pseudocode

```pseudo
FUNCTION rerank_llm(hits, purpose):
    # hits: [{id, document, metadata, score_vector}]
    prompt = build_rerank_prompt(purpose, hits)
    llm_out = llm.call(prompt)
    RETURN parse_ranked_ids(llm_out)

FUNCTION llm_generate(system, user, context):
    prompt = render_citation_template(system, user, context)
    return llm.call(prompt)
```

---

# Appendix D — Link Auto-Detection Pseudocode

```pseudo
FUNCTION try_autolink(parsed, tenant_id, rfq_ref):
    IF rfq_ref:
        RETURN rfq_ref, 0.9

    rfq_candidates = postgres.search_documents(
        tenant_id, kind ∈ {RFQ,RFP,Tender}, terms=extract_ids_and_titles(parsed)
    )
    best, conf = fuzzy_best_match(parsed.title, rfq_candidates)
    IF conf >= 0.7:
        RETURN best.id, conf
    RETURN None, 0.0
```

---

# Appendix E — Pricing Table Normalization Pseudocode

```pseudo
FUNCTION normalize_pricing(chunks):
    rows = []
    FOR ch IN chunks WHERE ch.meta.is_table:
        tbl = parse_table(ch.quote)               # headers, cells, units
        rows += extract_line_items(tbl)
    rows = standardize_currency(rows, target="USD")
    RETURN aggregate_by_item(rows)
```

---

## Final Notes

* This documentation provides a ready-to-implement blueprint: schemas, APIs, contextual-retrieval algorithms, agent roles, and ops guidance.
* You can map the pseudocode into your preferred stack (FastAPI + SQLAlchemy + Chroma client + OpenAI SDK + Docling wrapper).
* If you want, I can turn this into a starter repo layout (FastAPI service + workers + Docker Compose) with the outlined endpoints and stubbed adapters.
