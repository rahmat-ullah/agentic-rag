@echo off
REM Development setup script for Agentic RAG System (Windows)
REM This script sets up the complete development environment

echo 🚀 Setting up Agentic RAG Development Environment...
echo.

REM Check if Docker is running
echo 📋 Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not running
    echo Please install Docker Desktop and ensure it's running
    exit /b 1
)
echo ✓ Docker is available

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not available
    exit /b 1
)
echo ✓ Docker Compose is available
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env >nul
    echo ✓ Created .env file
    echo ⚠ Please review and update .env file with your configuration
    echo.
) else (
    echo ✓ .env file already exists
)

REM Pull and build Docker images
echo 🐳 Building Docker images...
docker-compose build
if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker images
    exit /b 1
)
echo ✓ Docker images built successfully
echo.

REM Start services
echo 🔄 Starting services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    exit /b 1
)
echo ✓ Services started successfully
echo.

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Run health check
echo 🔍 Running health check...
call scripts\health-check.bat
if %errorlevel% neq 0 (
    echo ❌ Health check failed
    echo 🔧 Check service logs: docker-compose logs
    exit /b 1
)

echo.
echo 🎉 Development environment setup complete!
echo.
echo 📊 Available Services:
echo   • API Documentation: http://localhost:8000/docs
echo   • Database Admin (Adminer): http://localhost:8080
echo   • MinIO Console: http://localhost:9001
echo   • Redis Commander: http://localhost:8081
echo   • ChromaDB: http://localhost:8001
echo.
echo 🛠️ Next Steps:
echo   1. Install Python dependencies: pip install -e ".[dev]"
echo   2. Run database migrations: alembic upgrade head
echo   3. Start the API: docker-compose up api
echo   4. Visit http://localhost:8000/docs to see the API documentation
echo.
echo 📚 Useful Commands:
echo   • View logs: docker-compose logs [service-name]
echo   • Stop services: docker-compose down
echo   • Restart services: docker-compose restart
echo   • Health check: scripts\health-check.bat
