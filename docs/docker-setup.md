# Docker Setup for Agentic RAG System

This document provides comprehensive instructions for setting up the Agentic RAG System using Docker Compose.

## Overview

The Docker Compose configuration includes the following services:

- **PostgreSQL 16** - Primary database with multi-tenant Row-Level Security
- **ChromaDB 0.4.22** - Vector database for document embeddings
- **MinIO** - S3-compatible object storage for documents
- **Redis 7.2** - Caching and session management
- **API Service** - FastAPI application (development mode)

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- 10GB free disk space for data volumes

## Quick Start

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd agentic-contextual-rag
   ```

2. **Copy the environment file and customize:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. **Create data directories:**
   ```bash
   mkdir -p data/{postgres,chromadb,minio,redis}
   ```

4. **Start all services:**
   ```bash
   docker-compose up -d
   ```

5. **Check service health:**
   ```bash
   docker-compose ps
   ```

6. **View logs:**
   ```bash
   docker-compose logs -f
   ```

## Service Details

### PostgreSQL Database

- **Port:** 5432
- **Database:** agentic_rag
- **User:** postgres
- **Password:** postgres (change in production)
- **Features:**
  - Row-Level Security enabled
  - UUID and crypto extensions
  - Performance monitoring with pg_stat_statements
  - Audit logging configured
  - Optimized for development workloads

**Connection String:**
```
postgresql://postgres:postgres@localhost:5432/agentic_rag
```

### ChromaDB Vector Database

- **Port:** 8001 (HTTP), 50051 (gRPC)
- **Features:**
  - Persistent storage
  - CORS configured for development
  - Optional authentication for production
  - Performance optimizations

**Health Check:**
```bash
curl http://localhost:8001/api/v1/heartbeat
```

### MinIO Object Storage

- **API Port:** 9000
- **Console Port:** 9001
- **Access Key:** minioadmin
- **Secret Key:** minioadmin
- **Features:**
  - Automatic bucket creation (documents, thumbnails, exports)
  - Compression enabled for documents
  - Web console for management

**Console Access:**
http://localhost:9001

### Redis Cache

- **Port:** 6379
- **Password:** redis_password
- **Features:**
  - Persistence enabled (AOF + RDB)
  - Memory optimization (256MB limit)
  - LRU eviction policy

## Environment Configuration

Key environment variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DATA_PATH=./data/postgres

# Vector Database
CHROMADB_DATA_PATH=./data/chromadb
CHROMA_AUTH_TOKEN=your-auth-token  # For production

# Object Storage
MINIO_ROOT_PASSWORD=your-secure-password
MINIO_DATA_PATH=./data/minio

# Cache
REDIS_PASSWORD=your-secure-password
REDIS_DATA_PATH=./data/redis

# Security
JWT_SECRET_KEY=your-jwt-secret
SECRET_KEY=your-app-secret
```

## Data Persistence

Data is persisted using Docker volumes with bind mounts to local directories:

```
./data/
├── postgres/     # PostgreSQL data
├── chromadb/     # ChromaDB collections
├── minio/        # Object storage files
└── redis/        # Redis persistence files
```

## Development Workflow

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres chromadb minio redis

# Start with logs
docker-compose up postgres chromadb minio redis
```

### Database Operations

```bash
# Run database migrations
docker-compose exec api python scripts/migrate.py upgrade

# Access PostgreSQL shell
docker-compose exec postgres psql -U postgres -d agentic_rag

# View database logs
docker-compose logs postgres
```

### Object Storage Operations

```bash
# Access MinIO client
docker-compose exec minio-init mc --help

# List buckets
docker-compose exec minio-init mc ls myminio/

# Upload test file
docker-compose exec minio-init mc cp /tmp/test.pdf myminio/documents/
```

### Monitoring and Debugging

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f chromadb

# Check service health
docker-compose ps
docker-compose exec postgres pg_isready -U postgres

# Monitor resource usage
docker stats
```

## Production Deployment

For production, use the production override:

```bash
# Copy production environment
cp .env.example .env.prod
# Edit .env.prod with production values

# Deploy with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS for all services
- [ ] Configure proper CORS origins
- [ ] Enable ChromaDB authentication
- [ ] Set up proper backup strategies
- [ ] Configure monitoring and alerting
- [ ] Use external managed databases if available
- [ ] Set up log aggregation
- [ ] Configure firewall rules
- [ ] Enable container security scanning

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :5432
   # Change ports in .env file
   ```

2. **Permission issues:**
   ```bash
   # Fix data directory permissions
   sudo chown -R $USER:$USER ./data/
   chmod -R 755 ./data/
   ```

3. **Out of disk space:**
   ```bash
   # Clean up Docker
   docker system prune -a
   docker volume prune
   ```

4. **Memory issues:**
   ```bash
   # Check container memory usage
   docker stats
   # Adjust resource limits in docker-compose.yml
   ```

### Service-Specific Issues

**PostgreSQL:**
- Check logs: `docker-compose logs postgres`
- Verify connection: `docker-compose exec postgres pg_isready`
- Reset data: `docker-compose down -v && docker-compose up -d`

**ChromaDB:**
- Health check: `curl http://localhost:8001/api/v1/heartbeat`
- Reset collections: Remove `./data/chromadb/` directory

**MinIO:**
- Console access: http://localhost:9001
- Check buckets: `docker-compose exec minio-init mc ls myminio/`

**Redis:**
- Test connection: `docker-compose exec redis redis-cli ping`
- Monitor: `docker-compose exec redis redis-cli monitor`

## Security Considerations

1. **Change default passwords** in production
2. **Enable SSL/TLS** for all external connections
3. **Use secrets management** for sensitive data
4. **Regular security updates** for base images
5. **Network isolation** between environments
6. **Backup encryption** for data at rest
7. **Access logging** and monitoring
8. **Container vulnerability scanning**

## Backup and Recovery

### Database Backup
```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U postgres agentic_rag > backup.sql

# Restore PostgreSQL
docker-compose exec -T postgres psql -U postgres agentic_rag < backup.sql
```

### Full Data Backup
```bash
# Stop services
docker-compose down

# Backup data directory
tar -czf agentic-rag-backup-$(date +%Y%m%d).tar.gz ./data/

# Restart services
docker-compose up -d
```

## Performance Tuning

### PostgreSQL
- Adjust `shared_buffers` based on available memory
- Tune `effective_cache_size` to 75% of system memory
- Monitor with `pg_stat_statements`

### ChromaDB
- Increase `CHROMA_SEGMENT_CACHE_SIZE` for large collections
- Use SSD storage for better performance

### MinIO
- Enable compression for document storage
- Use multiple drives for better I/O performance

### Redis
- Adjust `maxmemory` based on usage patterns
- Monitor memory usage and eviction policies
