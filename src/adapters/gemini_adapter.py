"""Google Gemini adapter implementation.

Supports Gemini 2.5 Flash, Gemini 2.5 Pro and other Gemini models.
Uses response_schema for constrained decoding.
"""

import json
from typing import Any, TypeVar

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel

from src.adapters.base import LLMAdapter, LLMResponse, StructuredOutput

T = TypeVar("T", bound=BaseModel)


class GeminiAdapter(LLMAdapter):
    """Google Gemini API adapter.

    Example:
        adapter = GeminiAdapter(model="gemini-2.5-flash")
        response = await adapter.generate("Explain SQL joins")
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        if api_key:
            genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        config = GenerationConfig(
            temperature=self.temperature,
            **kwargs,
        )

        response = await self.client.generate_content_async(
            full_prompt,
            generation_config=config,
        )

        # Extract usage metadata
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        return LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage,
            raw_response=response,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> StructuredOutput[T]:
        """Generate structured output using Gemini's response_schema.

        Uses JSON mode with schema enforcement for 100% compliance.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Convert Pydantic model to JSON schema for Gemini
        schema = response_model.model_json_schema()

        config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
            **kwargs,
        )

        response = await self.client.generate_content_async(
            full_prompt,
            generation_config=config,
        )

        # Parse the JSON response into the Pydantic model
        parsed_data = response_model.model_validate_json(response.text)

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        llm_response = LLMResponse(
            content=response.text,
            model=self.model,
            usage=usage,
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
        """Generate with function/tool calling.

        Converts OpenAI-style tool definitions to Gemini format.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Convert OpenAI tool format to Gemini function declarations
        gemini_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                gemini_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })

        config = GenerationConfig(
            temperature=self.temperature,
            **kwargs,
        )

        response = await self.client.generate_content_async(
            full_prompt,
            generation_config=config,
            tools=gemini_tools if gemini_tools else None,
        )

        # Extract function calls if present
        content = response.text
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "function_call"):
                        fc = part.function_call
                        content = json.dumps({
                            "tool_calls": [{
                                "name": fc.name,
                                "arguments": dict(fc.args),
                            }]
                        })
                        break

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            raw_response=response,
        )
