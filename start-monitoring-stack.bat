@echo off
REM Agentic RAG Monitoring Stack Startup Script
REM This script starts the complete monitoring and observability system

echo.
echo ========================================
echo  Agentic RAG Monitoring Stack Startup
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Check if we're in the right directory
if not exist "docker-compose.yml" (
    echo ERROR: docker-compose.yml not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo ✅ Found docker-compose.yml

REM Create data directories if they don't exist
echo.
echo 📁 Creating data directories...
if not exist "data" mkdir data
if not exist "data\monitoring" mkdir data\monitoring
if not exist "data\monitoring\prometheus" mkdir data\monitoring\prometheus
if not exist "data\monitoring\grafana" mkdir data\monitoring\grafana
if not exist "data\monitoring\elasticsearch" mkdir data\monitoring\elasticsearch
if not exist "data\monitoring\alertmanager" mkdir data\monitoring\alertmanager
if not exist "data\monitoring\filebeat" mkdir data\monitoring\filebeat

echo ✅ Data directories created

REM Start core services first
echo.
echo 🚀 Starting core services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start core services
    pause
    exit /b 1
)

echo ✅ Core services started

REM Wait a bit for core services to initialize
echo.
echo ⏳ Waiting for core services to initialize (30 seconds)...
timeout /t 30 /nobreak >nul

REM Start monitoring stack
echo.
echo 📊 Starting monitoring stack...
docker-compose -f docker-compose.monitoring.yml up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start monitoring services
    pause
    exit /b 1
)

echo ✅ Monitoring stack started

REM Wait for services to be ready
echo.
echo ⏳ Waiting for services to be ready...
set /a counter=0
:wait_loop
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 goto services_ready
set /a counter+=1
if %counter% geq 24 (
    echo WARNING: Services taking longer than expected to start
    echo You can continue manually or wait longer
    goto show_status
)
timeout /t 5 /nobreak >nul
goto wait_loop

:services_ready
echo ✅ Services are ready!

:show_status
echo.
echo 📋 Service Status:
echo.
docker-compose ps
echo.
docker-compose -f docker-compose.monitoring.yml ps

echo.
echo 🌐 Service URLs:
echo.
echo   Main Application:
echo     API:           http://localhost:8000
echo     Documentation: http://localhost:8000/docs
echo.
echo   Monitoring Dashboards:
echo     Grafana:       http://localhost:3000 (admin/agentic-rag-admin)
echo     Prometheus:    http://localhost:9090
echo     Kibana:        http://localhost:5601
echo     Jaeger:        http://localhost:16686
echo     Alertmanager:  http://localhost:9093
echo.
echo   System Monitoring:
echo     Node Exporter: http://localhost:9100/metrics
echo     cAdvisor:      http://localhost:8080
echo.
echo   Storage:
echo     MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

echo.
echo 🧪 Testing Options:
echo.
echo   1. Run automated tests:
echo      python scripts\test_monitoring_stack.py
echo.
echo   2. Generate test data:
echo      python scripts\demo_monitoring_system.py
echo.
echo   3. Manual testing:
echo      - Open Grafana dashboards
echo      - Check Prometheus targets
echo      - Search logs in Kibana
echo      - View traces in Jaeger

echo.
echo ✅ Monitoring stack is ready!
echo.
echo Press any key to run automated tests, or Ctrl+C to exit...
pause >nul

REM Run automated tests
echo.
echo 🧪 Running automated tests...
python scripts\test_monitoring_stack.py
if %errorlevel% equ 0 (
    echo.
    echo 🎉 All tests passed! Your monitoring stack is working correctly.
) else (
    echo.
    echo ⚠️  Some tests failed. Check the output above for details.
)

echo.
echo Press any key to exit...
pause >nul
