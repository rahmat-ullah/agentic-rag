#!/usr/bin/env python3
"""
Database migration management script for Agentic RAG System.

This script provides convenient commands for managing database migrations
using Alembic with proper environment configuration.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.config import get_settings


def run_command(command: list[str], env_vars: dict = None) -> int:
    """Run a command with optional environment variables."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, env=env)
    return result.returncode


def get_database_url(environment: str = None) -> str:
    """Get database URL for the specified environment."""
    if environment:
        os.environ["ENVIRONMENT"] = environment
    
    settings = get_settings()
    return str(settings.database.postgres_url)


def create_migration(message: str, autogenerate: bool = True, environment: str = None) -> int:
    """Create a new migration."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    
    command = ["alembic", "revision"]
    if autogenerate:
        command.append("--autogenerate")
    command.extend(["-m", message])
    
    return run_command(command, env_vars)


def upgrade_database(revision: str = "head", environment: str = None) -> int:
    """Upgrade database to specified revision."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    command = ["alembic", "upgrade", revision]
    return run_command(command, env_vars)


def downgrade_database(revision: str, environment: str = None) -> int:
    """Downgrade database to specified revision."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    command = ["alembic", "downgrade", revision]
    return run_command(command, env_vars)


def show_history(environment: str = None) -> int:
    """Show migration history."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    command = ["alembic", "history", "--verbose"]
    return run_command(command, env_vars)


def show_current(environment: str = None) -> int:
    """Show current migration revision."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    command = ["alembic", "current", "--verbose"]
    return run_command(command, env_vars)


def check_migrations(environment: str = None) -> int:
    """Check if migrations are up to date."""
    env_vars = {"POSTGRES_URL": get_database_url(environment)}
    
    # Get current revision
    result = subprocess.run(
        ["alembic", "current"], 
        env={**os.environ, **env_vars},
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error getting current revision")
        return 1
    
    current = result.stdout.strip()
    
    # Get head revision
    result = subprocess.run(
        ["alembic", "heads"], 
        env={**os.environ, **env_vars},
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error getting head revision")
        return 1
    
    head = result.stdout.strip()
    
    if current == head:
        print("✓ Database is up to date")
        return 0
    else:
        print(f"✗ Database needs migration: current={current}, head={head}")
        return 1


def reset_database(environment: str = None) -> int:
    """Reset database (downgrade to base and upgrade to head)."""
    print("⚠️  WARNING: This will reset the entire database!")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() != "yes":
        print("Aborted")
        return 0
    
    # Downgrade to base
    result = downgrade_database("base", environment)
    if result != 0:
        return result
    
    # Upgrade to head
    return upgrade_database("head", environment)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Database migration management")
    parser.add_argument(
        "--env", 
        choices=["development", "testing", "staging", "production"],
        help="Environment to use for database connection"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create migration
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("message", help="Migration message")
    create_parser.add_argument("--no-autogenerate", action="store_true", help="Don't auto-generate migration")
    
    # Upgrade
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument("revision", nargs="?", default="head", help="Target revision (default: head)")
    
    # Downgrade
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("revision", help="Target revision")
    
    # History
    subparsers.add_parser("history", help="Show migration history")
    
    # Current
    subparsers.add_parser("current", help="Show current revision")
    
    # Check
    subparsers.add_parser("check", help="Check if migrations are up to date")
    
    # Reset
    subparsers.add_parser("reset", help="Reset database (DANGEROUS)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "create":
            return create_migration(
                args.message, 
                autogenerate=not args.no_autogenerate,
                environment=args.env
            )
        elif args.command == "upgrade":
            return upgrade_database(args.revision, args.env)
        elif args.command == "downgrade":
            return downgrade_database(args.revision, args.env)
        elif args.command == "history":
            return show_history(args.env)
        elif args.command == "current":
            return show_current(args.env)
        elif args.command == "check":
            return check_migrations(args.env)
        elif args.command == "reset":
            return reset_database(args.env)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
