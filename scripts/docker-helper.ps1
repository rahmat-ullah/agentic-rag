# Docker Helper Script for Agentic RAG System (PowerShell)
# This script provides convenient commands for managing the Docker environment

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Option = ""
)

# Configuration
$ProjectName = "agentic-rag"
$ComposeFile = "docker-compose.yml"
$ProdComposeFile = "docker-compose.prod.yml"
$EnvFile = ".env"

# Helper functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Requirements {
    Write-Info "Checking requirements..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed"
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose is not installed"
        exit 1
    }
    
    if (-not (Test-Path $EnvFile)) {
        Write-Warning ".env file not found, copying from .env.example"
        Copy-Item .env.example .env
        Write-Info "Please edit .env file with your configuration"
    }
    
    Write-Success "Requirements check passed"
}

function New-DataDirectories {
    Write-Info "Creating data directories..."
    $dirs = @("data\postgres", "data\chromadb", "data\minio", "data\redis", "data\docling\models", "data\docling\temp", "data\clamav", "data\clamav\logs")
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Write-Success "Data directories created"
}

function Start-Services {
    param([string]$Mode = "dev")
    
    Test-Requirements
    New-DataDirectories
    
    Write-Info "Starting services in $Mode mode..."
    
    if ($Mode -eq "prod") {
        docker-compose -f $ComposeFile -f $ProdComposeFile up -d
    } else {
        docker-compose up -d
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started"
        Show-Status
    } else {
        Write-Error "Failed to start services"
    }
}

function Stop-Services {
    Write-Info "Stopping services..."
    docker-compose down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services stopped"
    } else {
        Write-Error "Failed to stop services"
    }
}

function Restart-Services {
    Write-Info "Restarting services..."
    docker-compose restart
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services restarted"
    } else {
        Write-Error "Failed to restart services"
    }
}

function Show-Status {
    Write-Info "Service status:"
    docker-compose ps
    
    Write-Host ""
    Write-Info "Service URLs:"
    Write-Host "  API Documentation: http://localhost:8000/docs"
    Write-Host "  MinIO Console: http://localhost:9001"
    Write-Host "  PostgreSQL: localhost:5432"
    Write-Host "  ChromaDB: http://localhost:8001"
    Write-Host "  Redis: localhost:6379"
}

function Show-Logs {
    param([string]$Service = "")
    
    if ($Service) {
        Write-Info "Showing logs for $Service..."
        docker-compose logs -f $Service
    } else {
        Write-Info "Showing logs for all services..."
        docker-compose logs -f
    }
}

function Invoke-Migrations {
    Write-Info "Running database migrations..."
    docker-compose exec api python scripts/migrate.py upgrade
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Migrations completed"
    } else {
        Write-Error "Migration failed"
    }
}

function Backup-Data {
    $BackupName = "agentic-rag-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    
    Write-Info "Creating backup: $BackupName"
    
    # Stop services to ensure consistent backup
    docker-compose down
    
    # Create backup using tar (if available) or 7zip
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -czf "$BackupName.tar.gz" .\data\
    } elseif (Get-Command 7z -ErrorAction SilentlyContinue) {
        7z a "$BackupName.7z" .\data\
    } else {
        # Fallback to PowerShell compression
        Compress-Archive -Path .\data\ -DestinationPath "$BackupName.zip"
    }
    
    # Restart services
    docker-compose up -d
    
    Write-Success "Backup created: $BackupName"
}

function Restore-Data {
    param([string]$BackupFile)
    
    if (-not $BackupFile) {
        Write-Error "Please specify backup file"
        return
    }
    
    if (-not (Test-Path $BackupFile)) {
        Write-Error "Backup file not found: $BackupFile"
        return
    }
    
    Write-Warning "This will overwrite existing data. Continue? (y/N)"
    $response = Read-Host
    if ($response -notmatch '^[Yy]$') {
        Write-Info "Restore cancelled"
        return
    }
    
    Write-Info "Restoring from backup: $BackupFile"
    
    # Stop services
    docker-compose down
    
    # Remove existing data
    if (Test-Path .\data\) {
        Remove-Item .\data\ -Recurse -Force
    }
    
    # Extract backup
    $extension = [System.IO.Path]::GetExtension($BackupFile)
    switch ($extension) {
        ".gz" { tar -xzf $BackupFile }
        ".7z" { 7z x $BackupFile }
        ".zip" { Expand-Archive -Path $BackupFile -DestinationPath . }
        default { Write-Error "Unsupported backup format: $extension" }
    }
    
    # Restart services
    docker-compose up -d
    
    Write-Success "Restore completed"
}

function Remove-AllData {
    Write-Warning "This will remove all containers, volumes, and data. Continue? (y/N)"
    $response = Read-Host
    if ($response -notmatch '^[Yy]$') {
        Write-Info "Cleanup cancelled"
        return
    }
    
    Write-Info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove data directories
    if (Test-Path .\data\) {
        Remove-Item .\data\ -Recurse -Force
    }
    
    # Clean up Docker system
    docker system prune -f
    
    Write-Success "Cleanup completed"
}

function Test-Health {
    Write-Info "Performing health checks..."
    
    # Check PostgreSQL
    $pgResult = docker-compose exec postgres pg_isready -U postgres -d agentic_rag 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "PostgreSQL: Healthy"
    } else {
        Write-Error "PostgreSQL: Unhealthy"
    }
    
    # Check ChromaDB
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8001/api/v1/heartbeat" -TimeoutSec 5 -ErrorAction Stop
        Write-Success "ChromaDB: Healthy"
    } catch {
        Write-Error "ChromaDB: Unhealthy"
    }
    
    # Check MinIO
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -TimeoutSec 5 -ErrorAction Stop
        Write-Success "MinIO: Healthy"
    } catch {
        Write-Error "MinIO: Unhealthy"
    }
    
    # Check Redis
    $redisResult = docker-compose exec redis redis-cli ping 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Redis: Healthy"
    } else {
        Write-Error "Redis: Unhealthy"
    }
}

function Show-Help {
    Write-Host "Agentic RAG System Docker Helper (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\scripts\docker-helper.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start [dev|prod]    Start all services (default: dev)"
    Write-Host "  stop                Stop all services"
    Write-Host "  restart             Restart all services"
    Write-Host "  status              Show service status and URLs"
    Write-Host "  logs [service]      Show logs (all services or specific service)"
    Write-Host "  migrate             Run database migrations"
    Write-Host "  backup              Create data backup"
    Write-Host "  restore <file>      Restore from backup file"
    Write-Host "  health              Perform health checks"
    Write-Host "  cleanup             Remove all containers and data"
    Write-Host "  help                Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\docker-helper.ps1 start           # Start in development mode"
    Write-Host "  .\scripts\docker-helper.ps1 start prod      # Start in production mode"
    Write-Host "  .\scripts\docker-helper.ps1 logs postgres   # Show PostgreSQL logs"
    Write-Host "  .\scripts\docker-helper.ps1 backup          # Create backup"
    Write-Host "  .\scripts\docker-helper.ps1 restore backup.zip  # Restore from backup"
}

# Main script logic
switch ($Command.ToLower()) {
    "start" { Start-Services $Option }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "status" { Show-Status }
    "logs" { Show-Logs $Option }
    "migrate" { Invoke-Migrations }
    "backup" { Backup-Data }
    "restore" { Restore-Data $Option }
    "health" { Test-Health }
    "cleanup" { Remove-AllData }
    "help" { Show-Help }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
    }
}
