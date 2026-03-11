"""Tests for LLM provider adapters."""

import pytest

from src.adapters.base import LLMAdapter, LLMResponse
from src.adapters.factory import create_adapter, get_available_providers
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.gemini_adapter import GeminiAdapter


class TestAdapterFactory:
    """Tests for the adapter factory."""

    def test_get_available_providers(self) -> None:
        """Should return list of supported providers."""
        providers = get_available_providers()
        assert "openai" in providers
        assert "gemini" in providers

    def test_create_openai_adapter(self, mock_openai_key: str) -> None:
        """Should create OpenAI adapter with default model."""
        adapter = create_adapter(provider="openai")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.provider_name == "openai"
        assert adapter.model == "gpt-4o-mini"

    def test_create_gemini_adapter(self, mock_gemini_key: str) -> None:
        """Should create Gemini adapter with default model."""
        adapter = create_adapter(provider="gemini")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter.provider_name == "gemini"
        assert adapter.model == "gemini-2.5-flash"

    def test_create_adapter_with_custom_model(self, mock_openai_key: str) -> None:
        """Should respect custom model parameter."""
        adapter = create_adapter(provider="openai", model="gpt-4o")
        assert adapter.model == "gpt-4o"

    def test_create_adapter_with_temperature(self, mock_openai_key: str) -> None:
        """Should respect temperature parameter."""
        adapter = create_adapter(provider="openai", temperature=0.7)
        assert adapter.temperature == 0.7

    def test_create_adapter_invalid_provider(self) -> None:
        """Should raise ValueError for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_adapter(provider="invalid")

    def test_create_adapter_from_env(
        self, monkeypatch: pytest.MonkeyPatch, mock_openai_key: str
    ) -> None:
        """Should use environment variables when no args provided."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")

        adapter = create_adapter()
        assert adapter.provider_name == "openai"
        assert adapter.model == "gpt-4o"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_properties(self) -> None:
        """Should correctly compute token properties."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o-mini",
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.total_tokens == 15

    def test_response_empty_usage(self) -> None:
        """Should handle empty usage dict."""
        response = LLMResponse(
            content="Hello",
            model="gpt-4o-mini",
        )

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.total_tokens == 0


class TestOpenAIAdapter:
    """Tests for OpenAI adapter (unit tests, no API calls)."""

    def test_init(self, mock_openai_key: str) -> None:
        """Should initialize with correct settings."""
        adapter = OpenAIAdapter(model="gpt-4o", temperature=0.5)
        assert adapter.model == "gpt-4o"
        assert adapter.temperature == 0.5
        assert adapter.provider_name == "openai"

    def test_repr(self, mock_openai_key: str) -> None:
        """Should have meaningful repr."""
        adapter = OpenAIAdapter()
        assert "OpenAIAdapter" in repr(adapter)
        assert "gpt-4o-mini" in repr(adapter)


class TestGeminiAdapter:
    """Tests for Gemini adapter (unit tests, no API calls)."""

    def test_init(self, mock_gemini_key: str) -> None:
        """Should initialize with correct settings."""
        adapter = GeminiAdapter(model="gemini-2.5-pro", temperature=0.3)
        assert adapter.model == "gemini-2.5-pro"
        assert adapter.temperature == 0.3
        assert adapter.provider_name == "gemini"

    def test_repr(self, mock_gemini_key: str) -> None:
        """Should have meaningful repr."""
        adapter = GeminiAdapter()
        assert "GeminiAdapter" in repr(adapter)
        assert "gemini-2.5-flash" in repr(adapter)


# Integration tests (require actual API keys)
@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI adapter."""

    @pytest.mark.asyncio
    async def test_generate_simple(self) -> None:
        """Should generate text response."""
        adapter = create_adapter(provider="openai")
        response = await adapter.generate("What is 2+2? Answer with just the number.")

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
        assert response.total_tokens > 0


@pytest.mark.integration
class TestGeminiIntegration:
    """Integration tests for Gemini adapter."""

    @pytest.mark.asyncio
    async def test_generate_simple(self) -> None:
        """Should generate text response."""
        adapter = create_adapter(provider="gemini")
        response = await adapter.generate("What is 2+2? Answer with just the number.")

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
