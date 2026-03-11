"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    """Return the default config path."""
    return project_root / "config" / "default.yaml"


@pytest.fixture
def mock_openai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock OpenAI API key for testing."""
    key = "sk-test-mock-key-for-testing"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    return key


@pytest.fixture
def mock_gemini_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a mock Gemini API key for testing."""
    key = "test-mock-gemini-key"
    monkeypatch.setenv("GOOGLE_API_KEY", key)
    return key


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Remove LLM-related environment variables for clean testing."""
    env_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


# Skip markers for integration tests
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take > 10s)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip integration tests if API keys are not set."""
    skip_integration = pytest.mark.skip(reason="API keys not configured")

    for item in items:
        if "integration" in item.keywords:
            # Check if required API key is set
            if "openai" in item.name.lower():
                if not os.getenv("OPENAI_API_KEY"):
                    item.add_marker(skip_integration)
            elif "gemini" in item.name.lower():
                if not os.getenv("GOOGLE_API_KEY"):
                    item.add_marker(skip_integration)
