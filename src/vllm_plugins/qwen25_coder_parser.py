"""Custom vLLM tool parser for Qwen2.5-Coder models.

Qwen2.5-Coder emits tool calls in formats different from the standard
Hermes <tool_call> tags:
  - <tool_call>...</tool_call> (standard Hermes, sometimes used)
  - <tools>...</tools> XML tags (Qwen2.5-Coder variant)
  - fenced ```json code blocks
  - raw JSON {"name": ..., "arguments": ...}

This parser handles all variants and converts them to OpenAI-style
tool_calls so LangGraph/AGL can follow the tool branch normally.

Registered as a vllm.general_plugins entry_point so it loads
automatically in ALL vLLM processes (engine core + workers),
including Ray worker processes used by VERL.
"""

import json
import re
from typing import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

# Patterns ordered from most specific to least specific
_TOOL_CALL_PATTERNS = [
    # 1. Standard Hermes format
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),
    # 2. Qwen2.5-Coder <tools> variant
    re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL),
    # 3. Fenced JSON code block
    re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL),
]


def _normalize_calls(obj):
    """Accept various JSON shapes for tool calls."""
    if isinstance(obj, dict):
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
            return obj["tool_calls"]
        if "name" in obj:
            return [obj]
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict) and "name" in item]
    return None


def _try_parse_tool_calls(text: str) -> list[dict] | None:
    """Try to extract tool call dicts from model output text."""
    for pattern in _TOOL_CALL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            calls = []
            for match in matches:
                match = match.strip()
                try:
                    obj = json.loads(match)
                except json.JSONDecodeError:
                    continue
                normalized = _normalize_calls(obj)
                if normalized:
                    calls.extend(normalized)
            if calls:
                return calls

    # Last resort: raw JSON object in the full text
    text = text.strip()
    try:
        obj = json.loads(text)
        normalized = _normalize_calls(obj)
        if normalized:
            return normalized
    except json.JSONDecodeError:
        pass

    return None


class Qwen25CoderToolParser(ToolParser):
    """Handles Qwen2.5-Coder's various tool-call output formats."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def adjust_request(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionRequest:
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        parsed = _try_parse_tool_calls(model_output)
        if not parsed:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls = []
        for i, call in enumerate(parsed):
            fn_name = (
                call.get("name")
                or call.get("function", {}).get("name")
            )
            args = (
                call.get("arguments")
                or call.get("parameters")
                or call.get("function", {}).get("arguments", {})
            )
            if fn_name is None:
                continue
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)

            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=fn_name,
                        arguments=args,
                    ),
                )
            )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        # Strip tool-call markup from content
        content = model_output
        for pattern in _TOOL_CALL_PATTERNS:
            content = pattern.sub("", content)
        content = content.strip() or None

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Streaming not needed for VERL training rollouts.
        return DeltaMessage(content=delta_text)


def register():
    """Entry point called by vLLM's load_general_plugins().

    Runs in every vLLM process (engine core + workers), so the
    parser is available even in Ray worker processes.
    """
    ToolParserManager.register_module(
        "qwen25_coder",
        module=Qwen25CoderToolParser,
    )
