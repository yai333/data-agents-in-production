"""Adapter factory for creating LLM adapters.

This module provides a simple factory function to create the appropriate
adapter based on configuration.
"""

import os
from typing import Any

from src.adapters.base import LLMAdapter
from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.openai_adapter import OpenAIAdapter

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
}


def create_adapter(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMAdapter:
    """Create an LLM adapter for the specified provider.

    Args:
        provider: The provider name ("openai" or "gemini").
                  Defaults to LLM_PROVIDER env var or "openai".
        model: The model identifier. Defaults to LLM_MODEL env var
               or the provider's default model.
        **kwargs: Additional configuration passed to the adapter.

    Returns:
        An initialized LLMAdapter instance.

    Raises:
        ValueError: If the provider is not supported.

    Example:
        # Use environment defaults
        adapter = create_adapter()

        # Explicit configuration
        adapter = create_adapter(provider="gemini", model="gemini-2.5-pro")

        # With custom settings
        adapter = create_adapter(
            provider="openai",
            model="gpt-5.1",
            temperature=0.7,
        )
    """
    # Resolve provider from env if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Resolve model from env if not specified
    if model is None:
        model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, "gpt-5.1-mini"))

    # Create the appropriate adapter
    if provider == "openai":
        api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
        return OpenAIAdapter(model=model, api_key=api_key, **kwargs)

    elif provider == "gemini":
        api_key = kwargs.pop("api_key", None) or os.getenv("GOOGLE_API_KEY")
        return GeminiAdapter(model=model, api_key=api_key, **kwargs)

    else:
        supported = ", ".join(DEFAULT_MODELS.keys())
        raise ValueError(
            f"Unsupported provider: {provider!r}. Supported providers: {supported}"
        )


def get_available_providers() -> list[str]:
    """Return list of available provider names."""
    return list(DEFAULT_MODELS.keys())


def get_model_name(
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Get the configured model name for use with LangChain.

    This returns the model string (e.g., "gpt-5.1") that can be passed
    to LangChain's create_agent() or other LangChain functions.

    Args:
        provider: The provider name ("openai" or "gemini").
                  Defaults to LLM_PROVIDER env var or "openai".
        model: The model identifier. Defaults to LLM_MODEL env var
               or the provider's default model.

    Returns:
        The model name string.

    Example:
        from langchain.agents import create_agent
        from src.adapters import get_model_name

        model = get_model_name()  # Returns e.g., "gpt-5.1"
        agent = create_agent(model=model, tools=[...])
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if model is None:
        model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, "gpt-5.1-mini"))

    return model


def get_provider_name() -> str:
    """Get the configured provider name.

    Returns:
        The provider name string (e.g., "openai", "gemini").
    """
    return os.getenv("LLM_PROVIDER", "openai").lower()
