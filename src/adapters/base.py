"""Base adapter interface for LLM providers.

This module defines the abstract interface that all LLM adapters must implement.
The adapter pattern allows the book's code to work with both OpenAI and Gemini
without changing the core logic.

See Appendix F for full documentation on the adapter interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Standard response from any LLM provider.

    Attributes:
        content: The text content of the response
        model: The model that generated the response
        usage: Token usage statistics
        raw_response: The original provider-specific response object
    """
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StructuredOutput(Generic[T]):
    """Response containing a parsed Pydantic model.

    Attributes:
        data: The parsed Pydantic model instance
        raw_response: The underlying LLMResponse
    """
    data: T
    raw_response: LLMResponse


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    All provider adapters (OpenAI, Gemini, etc.) must implement this interface.
    This ensures the book's code works identically regardless of which
    provider is configured.

    Example:
        adapter = create_adapter("openai", model="gpt-5.1-mini")
        response = await adapter.generate("What is 2+2?")
        print(response.content)  # "4"
    """

    def __init__(self, model: str, temperature: float = 0.0, **kwargs: Any):
        """Initialize the adapter.

        Args:
            model: The model identifier (e.g., "gpt-5.1-mini", "gemini-2.5-flash")
            temperature: Sampling temperature (0.0 for deterministic)
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a text response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the generated text
        """
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> StructuredOutput[T]:
        """Generate a structured response matching a Pydantic model.

        This uses constrained decoding (where available) to ensure
        the response exactly matches the schema.

        Args:
            prompt: The user prompt
            response_model: A Pydantic model class defining the output schema
            system_prompt: Optional system prompt for context
            **kwargs: Additional provider-specific parameters

        Returns:
            StructuredOutput containing the parsed model instance
        """
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with tool/function calling capability.

        Args:
            prompt: The user prompt
            tools: List of tool definitions in OpenAI function format
            system_prompt: Optional system prompt for context
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with tool calls in the content or raw_response
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini')."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, temperature={self.temperature})"
