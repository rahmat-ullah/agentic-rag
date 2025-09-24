#!/usr/bin/env python3
"""
Development environment setup script for Agentic RAG System.

This script sets up the development environment including:
- Installing Python dependencies
- Setting up pre-commit hooks
- Configuring development tools
- Running initial health checks
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")


def check_git():
    """Check if Git is available."""
    print("📋 Checking Git...")
    
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, check=True)
        print(f"✓ {result.stdout.decode().strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not available")
        sys.exit(1)


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Install the package in development mode
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    
    print("✓ Dependencies installed successfully")


def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    # Install pre-commit hooks
    run_command([sys.executable, "-m", "pre_commit", "install"])
    run_command([sys.executable, "-m", "pre_commit", "install", "--hook-type", "commit-msg"])
    
    # Run pre-commit on all files to ensure everything works
    print("🧪 Running pre-commit on all files...")
    result = run_command([sys.executable, "-m", "pre_commit", "run", "--all-files"], check=False)
    
    if result.returncode != 0:
        print("⚠️  Pre-commit found issues. This is normal for initial setup.")
        print("   The issues have been automatically fixed where possible.")
        print("   Please review the changes and commit them.")
    else:
        print("✓ Pre-commit hooks are working correctly")


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    print("📝 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("✓ Created .env file from template")
            print("⚠️  Please review and update .env file with your configuration")
        else:
            print("❌ .env.example file not found")
            sys.exit(1)
    else:
        print("✓ .env file already exists")


def check_docker():
    """Check if Docker is available."""
    print("🐳 Checking Docker...")
    
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print(f"✓ {result.stdout.decode().strip()}")
        
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
        print(f"✓ {result.stdout.decode().strip()}")
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Docker is not installed or not available")
        print("   Docker is required for running the full development environment")
        return False


def setup_git_hooks():
    """Set up additional Git hooks."""
    print("🔧 Setting up Git configuration...")
    
    # Set up Git hooks directory
    git_hooks_dir = Path(".git/hooks")
    if git_hooks_dir.exists():
        # Create a simple commit message template
        commit_template = """# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type should be one of the following:
# * feat: A new feature
# * fix: A bug fix
# * docs: Documentation only changes
# * style: Changes that do not affect the meaning of the code
# * refactor: A code change that neither fixes a bug nor adds a feature
# * perf: A code change that improves performance
# * test: Adding missing tests or correcting existing tests
# * chore: Changes to the build process or auxiliary tools
"""
        
        template_file = Path(".gitmessage")
        template_file.write_text(commit_template)
        
        # Configure Git to use the template
        run_command(["git", "config", "commit.template", ".gitmessage"], check=False)
        
        print("✓ Git commit template configured")


def run_initial_tests():
    """Run initial tests to verify setup."""
    print("🧪 Running initial tests...")
    
    # Check if tests directory exists
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("⚠️  Tests directory not found, skipping test run")
        return
    
    # Run a simple import test
    try:
        result = run_command([
            sys.executable, "-c", 
            "import agentic_rag; print('✓ Package imports successfully')"
        ], check=False)
        
        if result.returncode == 0:
            print("✓ Package imports successfully")
        else:
            print("⚠️  Package import failed, but this might be expected during initial setup")
    
    except Exception as e:
        print(f"⚠️  Could not run import test: {e}")


def print_next_steps():
    """Print next steps for the developer."""
    print("\n🎉 Development environment setup complete!")
    print("\n📋 Next Steps:")
    print("   1. Review and update .env file with your configuration")
    print("   2. Start Docker services: docker-compose up -d")
    print("   3. Run health check: scripts/health-check.bat (Windows) or scripts/health-check.sh (Linux)")
    print("   4. Run database migrations: python scripts/migrate.py upgrade head")
    print("   5. Start the API: uvicorn agentic_rag.api.main:app --reload")
    print("\n🔧 Useful Commands:")
    print("   • Run tests: pytest tests/")
    print("   • Format code: black src/ tests/")
    print("   • Check types: mypy src/")
    print("   • Run all checks: pre-commit run --all-files")
    print("\n📚 Documentation:")
    print("   • API docs: http://localhost:8000/docs (when API is running)")
    print("   • Project README: README.md")
    print("   • Sprint documentation: sprints/")


def main():
    """Main setup function."""
    print("🚀 Setting up Agentic RAG Development Environment...")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        # Run setup steps
        check_python_version()
        check_git()
        install_dependencies()
        create_env_file()
        setup_pre_commit()
        setup_git_hooks()
        check_docker()
        run_initial_tests()
        
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
