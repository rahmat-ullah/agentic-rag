@echo off
REM Health check script for Agentic RAG System services (Windows)
REM This script checks the health of all required services

echo 🔍 Checking Agentic RAG System Health...

REM Check if Docker Compose is running
echo 📋 Checking Docker Compose services...
docker-compose ps | findstr "Up" >nul
if %errorlevel% neq 0 (
    echo ❌ Docker Compose services are not running
    echo Run: docker-compose up -d
    exit /b 1
)

echo ✓ Docker Compose services are running
echo.

REM Initialize health check results
set all_healthy=true

REM Check PostgreSQL port
echo Checking PostgreSQL port...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet" >nul
if %errorlevel% neq 0 (
    echo ✗ PostgreSQL port 5432 closed
    set all_healthy=false
) else (
    echo ✓ PostgreSQL port 5432 open
)

REM Check ChromaDB
echo Checking ChromaDB...
curl -s -o nul -w "%%{http_code}" "http://localhost:8001/api/v1/heartbeat" | findstr "200" >nul
if %errorlevel% neq 0 (
    echo ✗ ChromaDB unhealthy
    set all_healthy=false
) else (
    echo ✓ ChromaDB healthy
)

REM Check MinIO
echo Checking MinIO...
curl -s -o nul -w "%%{http_code}" "http://localhost:9000/minio/health/live" | findstr "200" >nul
if %errorlevel% neq 0 (
    echo ✗ MinIO unhealthy
    set all_healthy=false
) else (
    echo ✓ MinIO healthy
)

REM Check Redis port
echo Checking Redis port...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet" >nul
if %errorlevel% neq 0 (
    echo ✗ Redis port 6379 closed
    set all_healthy=false
) else (
    echo ✓ Redis port 6379 open
)

REM Check API (if running)
echo Checking API port...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 8000 -InformationLevel Quiet" >nul
if %errorlevel% neq 0 (
    echo ⚠ API is not running (this is normal if not started yet)
) else (
    echo ✓ API port 8000 open
    curl -s -o nul -w "%%{http_code}" "http://localhost:8000/health" | findstr "200" >nul
    if %errorlevel% neq 0 (
        echo ⚠ API port is open but health check failed
    ) else (
        echo ✓ API health check passed
    )
)

echo.

REM Summary
if "%all_healthy%"=="true" (
    echo 🎉 All core services are healthy!
    echo.
    echo 📊 Service URLs:
    echo   • API Documentation: http://localhost:8000/docs
    echo   • Database Admin: http://localhost:8080
    echo   • MinIO Console: http://localhost:9001
    echo   • Redis Commander: http://localhost:8081
    echo   • ChromaDB: http://localhost:8001
    exit /b 0
) else (
    echo ❌ Some services are unhealthy
    echo.
    echo 🔧 Troubleshooting:
    echo   • Check logs: docker-compose logs [service-name]
    echo   • Restart services: docker-compose restart
    echo   • Rebuild services: docker-compose up --build
    exit /b 1
)
