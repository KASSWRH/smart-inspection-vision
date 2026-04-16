"""Shared pytest fixtures."""

import os

import pytest

# Set test environment before any app imports
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret-key-32-characters-long-enough")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("AUDIT_LOG_ENABLED", "false")
