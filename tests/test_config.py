"""Tests for configuration management."""

import pytest
from pathlib import Path

from src.utils.config import load_config, Settings, _substitute_env_vars


class TestEnvVarSubstitution:
    """Tests for environment variable substitution."""

    def test_substitute_simple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should substitute simple env var."""
        monkeypatch.setenv("TEST_VAR", "hello")
        result = _substitute_env_vars("${TEST_VAR}")
        assert result == "hello"

    def test_substitute_with_default(self) -> None:
        """Should use default when env var not set."""
        result = _substitute_env_vars("${UNSET_VAR:default_value}")
        assert result == "default_value"

    def test_substitute_in_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should substitute in nested dict."""
        monkeypatch.setenv("DB_HOST", "localhost")
        data = {"database": {"host": "${DB_HOST:127.0.0.1}"}}
        result = _substitute_env_vars(data)
        assert result["database"]["host"] == "localhost"

    def test_substitute_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should substitute in list items."""
        monkeypatch.setenv("ITEM", "value")
        data = ["${ITEM}", "${UNSET:default}"]
        result = _substitute_env_vars(data)
        assert result == ["value", "default"]

    def test_no_substitution_needed(self) -> None:
        """Should return unchanged if no env vars."""
        result = _substitute_env_vars("plain string")
        assert result == "plain string"


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self) -> None:
        """Should create settings with defaults."""
        settings = Settings()

        assert settings.llm.provider == "openai"
        assert settings.llm.temperature == 0.0
        assert settings.safety.max_result_rows == 1000
        assert settings.agent.validate_before_execute is True

    def test_settings_override(self) -> None:
        """Should allow overriding defaults."""
        from src.utils.config import LLMSettings

        settings = Settings(
            llm=LLMSettings(provider="gemini", model="gemini-2.5-pro")
        )

        assert settings.llm.provider == "gemini"
        assert settings.llm.model == "gemini-2.5-pro"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self, config_path: Path) -> None:
        """Should load config from default path."""
        if config_path.exists():
            settings = load_config(config_path)
            assert isinstance(settings, Settings)

    def test_load_missing_config(self, tmp_path: Path) -> None:
        """Should return defaults for missing config."""
        settings = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(settings, Settings)
        assert settings.llm.provider == "openai"

    def test_load_config_with_env_substitution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should substitute env vars in config."""
        monkeypatch.setenv("TEST_MODEL", "gpt-4o")

        config_content = """
llm:
  provider: openai
  model: ${TEST_MODEL:gpt-4o-mini}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        settings = load_config(config_file)
        assert settings.llm.model == "gpt-4o"
