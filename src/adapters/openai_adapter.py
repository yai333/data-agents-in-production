"""OpenAI adapter implementation.

Supports GPT-4o, GPT-4o-mini, GPT-5-mini and other OpenAI models.
Uses structured outputs with response_format for constrained decoding.
"""

import json
import os
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.adapters.base import LLMAdapter, LLMResponse, StructuredOutput

T = TypeVar("T", bound=BaseModel)


class OpenAIAdapter(LLMAdapter):
    """OpenAI API adapter.

    Example:
        adapter = OpenAIAdapter(model="gpt-5.1-mini")
        response = await adapter.generate("Explain SQL joins")

    Supports custom base URL via OPENAI_API_BASE environment variable.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-5.1-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        # Support custom base URL from env or parameter, default to official OpenAI
        base_url = base_url or os.getenv("OPENAI_API_BASE", self.DEFAULT_BASE_URL)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using OpenAI Chat Completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> StructuredOutput[T]:
        """Generate structured output using OpenAI's response_format.

        Uses JSON mode with schema enforcement for 100% compliance.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use OpenAI's structured output feature
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=response_model,
            **kwargs,
        )

        choice = response.choices[0]
        parsed_data = choice.message.parsed

        llm_response = LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response,
        )

        return StructuredOutput(data=parsed_data, raw_response=llm_response)

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with function/tool calling."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=tools,
            **kwargs,
        )

        choice = response.choices[0]

        # Extract tool calls if present
        content = choice.message.content or ""
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]
            content = json.dumps({"tool_calls": tool_calls})

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response,
        )
