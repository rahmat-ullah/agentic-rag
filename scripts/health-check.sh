#!/bin/bash

# Health check script for Agentic RAG System services
# This script checks the health of all required services

set -e

echo "🔍 Checking Agentic RAG System Health..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service_name... "
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

# Function to check TCP port
check_port() {
    local service_name=$1
    local host=$2
    local port=$3
    
    echo -n "Checking $service_name port... "
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓ Port $port open${NC}"
        return 0
    else
        echo -e "${RED}✗ Port $port closed${NC}"
        return 1
    fi
}

# Check if Docker Compose is running
echo "📋 Checking Docker Compose services..."
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}❌ Docker Compose services are not running${NC}"
    echo "Run: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose services are running${NC}"
echo ""

# Initialize health check results
all_healthy=true

# Check PostgreSQL
if ! check_port "PostgreSQL" "localhost" "5432"; then
    all_healthy=false
fi

# Check ChromaDB
if ! check_service "ChromaDB" "http://localhost:8001/api/v1/heartbeat"; then
    all_healthy=false
fi

# Check MinIO
if ! check_service "MinIO" "http://localhost:9000/minio/health/live"; then
    all_healthy=false
fi

# Check Redis
if ! check_port "Redis" "localhost" "6379"; then
    all_healthy=false
fi

# Check API (if running)
if check_port "API" "localhost" "8000"; then
    if ! check_service "API Health" "http://localhost:8000/health"; then
        echo -e "${YELLOW}⚠ API port is open but health check failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ API is not running (this is normal if not started yet)${NC}"
fi

echo ""

# Summary
if $all_healthy; then
    echo -e "${GREEN}🎉 All core services are healthy!${NC}"
    echo ""
    echo "📊 Service URLs:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Database Admin: http://localhost:8080"
    echo "  • MinIO Console: http://localhost:9001"
    echo "  • Redis Commander: http://localhost:8081"
    echo "  • ChromaDB: http://localhost:8001"
    exit 0
else
    echo -e "${RED}❌ Some services are unhealthy${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  • Check logs: docker-compose logs [service-name]"
    echo "  • Restart services: docker-compose restart"
    echo "  • Rebuild services: docker-compose up --build"
    exit 1
fi
