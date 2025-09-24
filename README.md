# Agentic RAG System

A production-grade, **agentic Retrieval-Augmented Generation (RAG)** system designed for procurement document processing and analysis.

## 🎯 Purpose

This system provides intelligent document processing and retrieval capabilities that:

- **Continuously learns from users** through feedback, edits, and link confirmations
- **Uses IBM Granite-Docling-258M** for advanced document parsing and conversion
- **Leverages OpenAI embeddings** for high-quality vectorization
- **Stores vectors in ChromaDB** and metadata in PostgreSQL
- **Implements contextual retrieval** with three-hop search pattern
- **Supports multi-tenant isolation** for enterprise deployment

## 🏗️ Architecture Overview

### Core Components

- **API Gateway**: FastAPI-based REST API with authentication
- **Ingestion Service**: Document upload, parsing, chunking, and embedding
- **Retrieval Service**: Three-hop retrieval (RFQ → Offers → Offer Chunks)
- **Agent Orchestrator**: Specialized agents for different tasks
- **Feedback Service**: Continuous learning from user interactions
- **Storage Layer**: PostgreSQL + ChromaDB + Object Storage

### Data Flow

```
Upload → Granite-Docling → Contextual Chunking → OpenAI Embeddings → ChromaDB/PostgreSQL
Query → RFQ Search → Link Traversal → Chunk Retrieval → LLM Synthesis → Response
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-contextual-rag
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start development environment**
   ```bash
   docker-compose up -d
   ```

4. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the development server**
   ```bash
   uvicorn agentic_rag.api.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
agentic-contextual-rag/
├── src/agentic_rag/           # Main application code
│   ├── api/                   # REST API endpoints
│   ├── services/              # Business logic and orchestration
│   ├── adapters/              # External service integrations
│   └── models/                # Data models and database schemas
├── tests/                     # Test suites
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── ops/                       # Operations and deployment
│   ├── docker/                # Docker configurations
│   ├── k8s/                   # Kubernetes manifests
│   ├── helm/                  # Helm charts
│   └── grafana/               # Monitoring dashboards
├── sprints/                   # Sprint planning and documentation
└── docs/                      # Additional documentation
```

## 🔧 Technology Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, Alembic
- **Databases**: PostgreSQL 14+, ChromaDB, Redis
- **AI/ML**: OpenAI GPT-4, text-embedding-3-large, IBM Granite-Docling-258M
- **Storage**: MinIO (S3-compatible), Object Storage
- **Infrastructure**: Docker, Kubernetes, Helm
- **Monitoring**: Prometheus, Grafana, OpenTelemetry

## 🏢 Multi-Tenancy

The system supports multi-tenant deployment with:

- **Row-Level Security (RLS)** in PostgreSQL
- **Tenant-isolated collections** in ChromaDB
- **Secure API access** with tenant-scoped authentication
- **Data isolation** across all storage layers

## 📊 Key Features

### Document Processing
- Support for PDFs, images, Office documents
- Advanced layout analysis and structure extraction
- Intelligent chunking with contextual information
- Automatic document linking and relationship detection

### Search & Retrieval
- Three-hop search pattern for precision
- Contextual query processing
- LLM-based result reranking
- Citation-rich answer synthesis

### Learning & Feedback
- User feedback collection and processing
- Continuous improvement through learning algorithms
- Quality metrics and analytics
- Automated system optimization

## 🔒 Security

- JWT-based authentication and authorization
- Role-based access control (RBAC)
- Data encryption at rest and in transit
- PII/sensitive data redaction
- Comprehensive audit logging

## 📈 Monitoring & Observability

- Application metrics with Prometheus
- Distributed tracing with OpenTelemetry
- Log aggregation and analysis
- Custom dashboards and alerting
- Performance monitoring and optimization

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# End-to-end tests
pytest tests/e2e

# All tests with coverage
pytest --cov=agentic_rag tests/
```

## 📚 Documentation

- [Project Documentation](project-documentation.md) - Complete technical specification
- [Sprint Planning](sprints/README.md) - Development roadmap and sprint breakdown
- [API Documentation](http://localhost:8000/docs) - Interactive API documentation (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the GitHub repository
- Check the [documentation](project-documentation.md)
- Review the [sprint planning](sprints/README.md) for implementation details

## 🗺️ Roadmap

The project is organized into 7 sprints over 14 weeks:

1. **Sprint 1**: Foundation & Core Infrastructure
2. **Sprint 2**: Document Ingestion Pipeline
3. **Sprint 3**: Basic Retrieval & Vector Search
4. **Sprint 4**: Contextual Retrieval & Three-Hop Search
5. **Sprint 5**: Agent Orchestration & Advanced Features
6. **Sprint 6**: Feedback System & Learning
7. **Sprint 7**: Production Deployment & Observability

See [sprints/README.md](sprints/README.md) for detailed sprint planning and progress tracking.
