"""Quick test for the custom Qwen2.5-Coder vLLM tool parser.

Tests:
  1. Plugin loads via vllm.general_plugins entry_point
  2. Parser is registered as "qwen25_coder"
  3. All 4 output formats parse correctly
  4. Edge cases (no tool calls, malformed JSON) handled gracefully

Usage:
    source .venv/bin/activate && python -m pytest tests/test_tool_parser.py
"""

import json
import sys


def test_plugin_registration():
    """Verify the parser registers via vllm.general_plugins."""
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.general_plugins")
    names = [ep.name for ep in eps]
    assert "qwen25_coder_parser" in names, (
        f"Entry point not found. Registered: {names}\n"
        f"Run: pip install -e .  (using venv pip, not conda pip)"
    )
    print("[PASS] Entry point registered")

    # Load the plugin
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParserManager,
    )
    registered = ToolParserManager.list_registered()
    assert "qwen25_coder" in registered, (
        f"Parser 'qwen25_coder' not in registered parsers: {registered}"
    )
    print("[PASS] Parser registered as 'qwen25_coder'")


def test_parse_formats():
    """Test all 4 output formats the parser handles."""
    from src.vllm_plugins.qwen25_coder_parser import _try_parse_tool_calls

    # Format 1: Standard Hermes <tool_call> tags
    text1 = """I'll look up the table details.
<tool_call>
{"name": "get_table_details", "arguments": {"table_name": "artist"}}
</tool_call>"""
    result1 = _try_parse_tool_calls(text1)
    assert result1 is not None, "Failed to parse <tool_call> format"
    assert result1[0]["name"] == "get_table_details"
    assert result1[0]["arguments"]["table_name"] == "artist"
    print("[PASS] Format 1: <tool_call> tags")

    # Format 2: Qwen2.5-Coder <tools> tags
    text2 = """Let me check the schema.
<tools>
{"name": "get_table_details", "arguments": {"table_name": "invoice"}}
</tools>"""
    result2 = _try_parse_tool_calls(text2)
    assert result2 is not None, "Failed to parse <tools> format"
    assert result2[0]["name"] == "get_table_details"
    print("[PASS] Format 2: <tools> tags")

    # Format 3: Fenced JSON code block
    text3 = """Here is the tool call:
```json
{"name": "sql_db_query", "arguments": {"query": "SELECT COUNT(*) FROM artist"}}
```"""
    result3 = _try_parse_tool_calls(text3)
    assert result3 is not None, "Failed to parse fenced JSON format"
    assert result3[0]["name"] == "sql_db_query"
    print("[PASS] Format 3: fenced ```json block")

    # Format 4: Raw JSON
    text4 = '{"name": "get_metrics", "arguments": {"table_name": "track"}}'
    result4 = _try_parse_tool_calls(text4)
    assert result4 is not None, "Failed to parse raw JSON format"
    assert result4[0]["name"] == "get_metrics"
    print("[PASS] Format 4: raw JSON")

    # Format 5: tool_calls wrapper
    text5 = '{"tool_calls": [{"name": "get_table_details", "arguments": {"table_name": "album"}}]}'
    result5 = _try_parse_tool_calls(text5)
    assert result5 is not None, "Failed to parse tool_calls wrapper"
    assert result5[0]["name"] == "get_table_details"
    print("[PASS] Format 5: tool_calls wrapper")

    # Format 6: "parameters" instead of "arguments"
    text6 = '{"name": "get_relationships", "parameters": {"table_name": "track"}}'
    result6 = _try_parse_tool_calls(text6)
    assert result6 is not None, "Failed to parse parameters variant"
    assert result6[0]["name"] == "get_relationships"
    print("[PASS] Format 6: parameters key variant")


def test_extract_tool_calls():
    """Test the full extract_tool_calls method returns proper ToolCall objects."""
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParserManager,
    )
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    parser_cls = ToolParserManager.tool_parsers.get("qwen25_coder")
    assert parser_cls is not None, "Parser class not found"

    # Create parser (tokenizer not used for non-streaming)
    parser = parser_cls(tokenizer=None)

    # Create a minimal request
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "test"}],
    )

    # Test with tool call
    output = '<tool_call>\n{"name": "sql_db_query", "arguments": {"query": "SELECT 1"}}\n</tool_call>'
    result = parser.extract_tool_calls(output, request)
    assert result.tools_called is True, f"Expected tools_called=True, got {result.tools_called}"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.function.name == "sql_db_query"
    args = json.loads(tc.function.arguments)
    assert args["query"] == "SELECT 1"
    print("[PASS] extract_tool_calls: proper ToolCall objects returned")

    # Test with no tool call (plain text)
    plain = "I don't need any tools for this. The answer is 42."
    result2 = parser.extract_tool_calls(plain, request)
    assert result2.tools_called is False
    assert result2.content == plain
    print("[PASS] extract_tool_calls: plain text returns tools_called=False")

    # Test with malformed JSON
    bad = "<tool_call>not valid json</tool_call>"
    result3 = parser.extract_tool_calls(bad, request)
    assert result3.tools_called is False
    print("[PASS] extract_tool_calls: malformed JSON handled gracefully")

    # Test multiple tool calls
    multi = """<tool_call>
{"name": "get_table_details", "arguments": {"table_name": "artist"}}
</tool_call>
<tool_call>
{"name": "get_table_details", "arguments": {"table_name": "album"}}
</tool_call>"""
    result4 = parser.extract_tool_calls(multi, request)
    assert result4.tools_called is True
    assert len(result4.tool_calls) == 2
    assert result4.tool_calls[0].function.name == "get_table_details"
    assert result4.tool_calls[1].function.name == "get_table_details"
    print("[PASS] extract_tool_calls: multiple tool calls parsed")


def test_edge_cases():
    """Test edge cases."""
    from src.vllm_plugins.qwen25_coder_parser import _try_parse_tool_calls

    # Empty string
    assert _try_parse_tool_calls("") is None
    print("[PASS] Edge: empty string")

    # Just whitespace
    assert _try_parse_tool_calls("   \n  ") is None
    print("[PASS] Edge: whitespace only")

    # JSON without "name" key
    assert _try_parse_tool_calls('{"foo": "bar"}') is None
    print("[PASS] Edge: JSON without name key")

    # Nested thinking + tool call (common pattern)
    thinking = """Let me think about this...

I need to look at the artist table first.

<tool_call>
{"name": "get_table_details", "arguments": {"table_name": "artist"}}
</tool_call>"""
    result = _try_parse_tool_calls(thinking)
    assert result is not None
    assert result[0]["name"] == "get_table_details"
    print("[PASS] Edge: text before tool call preserved")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Qwen2.5-Coder vLLM Tool Parser Plugin")
    print("=" * 50)
    print()

    try:
        test_plugin_registration()
        print()
        test_parse_formats()
        print()
        test_extract_tool_calls()
        print()
        test_edge_cases()
        print()
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
