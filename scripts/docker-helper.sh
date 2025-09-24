#!/bin/bash

# Docker Helper Script for Agentic RAG System
# This script provides convenient commands for managing the Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic-rag"
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found, copying from .env.example"
        cp .env.example .env
        log_info "Please edit .env file with your configuration"
    fi
    
    log_success "Requirements check passed"
}

create_data_dirs() {
    log_info "Creating data directories..."
    mkdir -p data/{postgres,chromadb,minio,redis,docling/models,docling/temp}
    chmod 755 data/*
    chmod 755 data/docling/*
    log_success "Data directories created"
}

start_services() {
    local mode=${1:-dev}
    
    check_requirements
    create_data_dirs
    
    log_info "Starting services in $mode mode..."
    
    if [ "$mode" = "prod" ]; then
        docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE up -d
    else
        docker-compose up -d
    fi
    
    log_success "Services started"
    show_status
}

stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."
    docker-compose restart
    log_success "Services restarted"
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo ""
    log_info "Service URLs:"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  MinIO Console: http://localhost:9001"
    echo "  PostgreSQL: localhost:5432"
    echo "  ChromaDB: http://localhost:8001"
    echo "  Redis: localhost:6379"
}

show_logs() {
    local service=${1:-}
    
    if [ -n "$service" ]; then
        log_info "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        log_info "Showing logs for all services..."
        docker-compose logs -f
    fi
}

run_migrations() {
    log_info "Running database migrations..."
    docker-compose exec api python scripts/migrate.py upgrade
    log_success "Migrations completed"
}

backup_data() {
    local backup_name="agentic-rag-backup-$(date +%Y%m%d-%H%M%S)"
    
    log_info "Creating backup: $backup_name"
    
    # Stop services to ensure consistent backup
    docker-compose down
    
    # Create backup
    tar -czf "${backup_name}.tar.gz" ./data/
    
    # Restart services
    docker-compose up -d
    
    log_success "Backup created: ${backup_name}.tar.gz"
}

restore_data() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        log_error "Please specify backup file"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_warning "This will overwrite existing data. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Restoring from backup: $backup_file"
    
    # Stop services
    docker-compose down
    
    # Remove existing data
    rm -rf ./data/
    
    # Extract backup
    tar -xzf "$backup_file"
    
    # Restart services
    docker-compose up -d
    
    log_success "Restore completed"
}

cleanup() {
    log_warning "This will remove all containers, volumes, and data. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    log_info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove data directories
    rm -rf ./data/
    
    # Clean up Docker system
    docker system prune -f
    
    log_success "Cleanup completed"
}

health_check() {
    log_info "Performing health checks..."
    
    # Check PostgreSQL
    if docker-compose exec postgres pg_isready -U postgres -d agentic_rag > /dev/null 2>&1; then
        log_success "PostgreSQL: Healthy"
    else
        log_error "PostgreSQL: Unhealthy"
    fi
    
    # Check ChromaDB
    if curl -f http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
        log_success "ChromaDB: Healthy"
    else
        log_error "ChromaDB: Unhealthy"
    fi
    
    # Check MinIO
    if curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        log_success "MinIO: Healthy"
    else
        log_error "MinIO: Unhealthy"
    fi
    
    # Check Redis
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis: Healthy"
    else
        log_error "Redis: Unhealthy"
    fi
}

show_help() {
    echo "Agentic RAG System Docker Helper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [dev|prod]    Start all services (default: dev)"
    echo "  stop                Stop all services"
    echo "  restart             Restart all services"
    echo "  status              Show service status and URLs"
    echo "  logs [service]      Show logs (all services or specific service)"
    echo "  migrate             Run database migrations"
    echo "  backup              Create data backup"
    echo "  restore <file>      Restore from backup file"
    echo "  health              Perform health checks"
    echo "  cleanup             Remove all containers and data"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start in development mode"
    echo "  $0 start prod      # Start in production mode"
    echo "  $0 logs postgres   # Show PostgreSQL logs"
    echo "  $0 backup          # Create backup"
    echo "  $0 restore backup.tar.gz  # Restore from backup"
}

# Main script logic
case "${1:-help}" in
    start)
        start_services "${2:-dev}"
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    migrate)
        run_migrations
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data "$2"
        ;;
    health)
        health_check
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
