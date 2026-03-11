"""Configuration management for the Text-to-SQL Agent.

Loads configuration from YAML files with environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseModel):
    """LLM provider settings."""
    provider: str = "openai"
    model: str = "gpt-5.1-mini"
    temperature: float = 0.0
    max_tokens: int = 4096


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    url: str = "postgresql://postgres:postgres@localhost:5432/chinook"
    pool_size: int = 5
    max_overflow: int = 10


class SafetySettings(BaseModel):
    """Safety and security settings."""
    max_result_rows: int = 1000
    query_timeout_seconds: int = 30
    pii_detection_enabled: bool = True
    allowed_operations: list[str] = Field(default_factory=lambda: ["SELECT"])
    blocked_tables: list[str] = Field(default_factory=list)


class RetrievalSettings(BaseModel):
    """Few-shot retrieval settings."""
    method: str = "hybrid"
    top_k: int = 5
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    rerank_enabled: bool = True
    rerank_model: str = "gpt-5.1-mini"


class AgentSettings(BaseModel):
    """Agent behavior settings."""
    max_retries: int = 3
    validate_before_execute: bool = True
    ask_on_ambiguity: bool = True
    require_approval_for: list[str] = Field(
        default_factory=lambda: ["DELETE", "UPDATE", "INSERT"]
    )


class ObservabilitySettings(BaseModel):
    """Logging and tracing settings."""
    log_level: str = "INFO"
    trace_enabled: bool = False
    metrics_enabled: bool = True
    metrics_prefix: str = "sql_agent"


class Settings(BaseSettings):
    """Main settings container."""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR:default} patterns with environment values."""
    if isinstance(value, str):
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    return value


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from YAML file with env var substitution.

    Args:
        config_path: Path to config file. Defaults to config/default.yaml

    Returns:
        Settings instance with resolved configuration.

    Example:
        settings = load_config()
        print(settings.llm.model)  # "gpt-5.1-mini"

        settings = load_config("config/production.yaml")
    """
    if config_path is None:
        # Find project root (where pyproject.toml is)
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                config_path = current / "config" / "default.yaml"
                break
            current = current.parent
        else:
            config_path = Path("config/default.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        # Return defaults if no config file
        return Settings()

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    resolved_config = _substitute_env_vars(raw_config)

    return Settings(**resolved_config)
