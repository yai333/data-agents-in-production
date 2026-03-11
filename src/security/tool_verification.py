"""Tool signature verification for MCP.

Prevents tool poisoning attacks by verifying tool definitions
match known-good signatures.

See 2.1 for tool poisoning attack vectors.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSignature:
    """Cryptographic signature for an MCP tool.

    Used to verify that a tool definition hasn't been modified
    from its known-good state.

    Attributes:
        tool_name: Name of the tool
        version: Version string
        description_hash: SHA-256 hash of the description
        schema_hash: SHA-256 hash of the parameters schema
        publisher: Publisher identifier
        signature: Cryptographic signature (for production use)
    """

    tool_name: str
    version: str
    description_hash: str
    schema_hash: str
    publisher: str
    signature: str = ""  # For production: Ed25519 signature


def create_signature(
    tool_definition: dict[str, Any],
    publisher: str,
    version: str = "1.0.0",
) -> ToolSignature:
    """Create a signature for a tool definition.

    Args:
        tool_definition: The MCP tool definition
        publisher: Publisher identifier
        version: Version string

    Returns:
        ToolSignature for the tool
    """
    # Hash the description
    description = tool_definition.get("description", "")
    desc_hash = hashlib.sha256(description.encode()).hexdigest()

    # Hash the schema (sorted for determinism)
    schema = tool_definition.get("parameters", {})
    schema_str = json.dumps(schema, sort_keys=True)
    schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()

    return ToolSignature(
        tool_name=tool_definition.get("name", "unknown"),
        version=version,
        description_hash=desc_hash,
        schema_hash=schema_hash,
        publisher=publisher,
        signature="",  # Would be computed with private key in production
    )


def verify_tool(
    tool_definition: dict[str, Any],
    expected_signature: ToolSignature,
) -> tuple[bool, str]:
    """Verify a tool definition matches expected signature.

    Prevents tool poisoning by ensuring tool hasn't been modified.

    Args:
        tool_definition: The tool definition from MCP server
        expected_signature: Known-good signature

    Returns:
        Tuple of (valid, reason)

    Example:
        >>> sig = create_signature(tool_def, "trusted_publisher")
        >>> # Store sig in trusted registry...
        >>> valid, reason = verify_tool(tool_def, sig)
        >>> valid
        True
    """
    # Hash the current description
    description = tool_definition.get("description", "")
    current_desc_hash = hashlib.sha256(description.encode()).hexdigest()

    if current_desc_hash != expected_signature.description_hash:
        return False, "Tool description has been modified"

    # Hash the current schema
    schema = tool_definition.get("parameters", {})
    schema_str = json.dumps(schema, sort_keys=True)
    current_schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()

    if current_schema_hash != expected_signature.schema_hash:
        return False, "Tool schema has been modified"

    # In production, also verify cryptographic signature
    # using the publisher's public key

    return True, "Tool verified"


def detect_suspicious_description(description: str) -> list[str]:
    """Detect suspicious patterns in tool descriptions.

    Tool poisoning attacks often hide instructions in descriptions.

    Args:
        description: Tool description to analyze

    Returns:
        List of suspicious patterns found
    """
    import re

    suspicious = []

    # Check for hidden instructions
    instruction_patterns = [
        (r"\[.*INSTRUCTION.*\]", "bracketed_instruction"),
        (r"\[.*SYSTEM.*\]", "system_directive"),
        (r"\[.*HIDDEN.*\]", "hidden_content"),
        (r"ignore\s+(previous|above)", "instruction_override"),
        (r"also\s+(send|transmit|exfil)", "data_exfiltration"),
        (r"(http|https)://", "external_url"),
        (r"base64|eval|exec", "code_execution_hint"),
    ]

    for pattern, pattern_name in instruction_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            suspicious.append(f"Suspicious pattern: {pattern_name}")

    # Check for unusually long descriptions (may hide payload)
    if len(description) > 2000:
        suspicious.append("Unusually long description")

    # Check for control characters
    if any(ord(c) < 32 and c not in "\n\r\t" for c in description):
        suspicious.append("Contains control characters")

    return suspicious


class ToolRegistry:
    """Registry of verified tool signatures.

    Maintains a database of known-good tool signatures
    for verification.
    """

    def __init__(self):
        self.signatures: dict[str, ToolSignature] = {}

    def register(self, signature: ToolSignature) -> None:
        """Register a verified tool signature.

        Args:
            signature: The tool signature to register
        """
        key = f"{signature.publisher}:{signature.tool_name}:{signature.version}"
        self.signatures[key] = signature

    def get(
        self,
        tool_name: str,
        publisher: str,
        version: str | None = None,
    ) -> ToolSignature | None:
        """Get a registered signature.

        Args:
            tool_name: Name of the tool
            publisher: Publisher identifier
            version: Specific version (None = latest)

        Returns:
            ToolSignature if found, None otherwise
        """
        if version:
            key = f"{publisher}:{tool_name}:{version}"
            return self.signatures.get(key)

        # Find latest version
        prefix = f"{publisher}:{tool_name}:"
        matching = [
            (k, v) for k, v in self.signatures.items()
            if k.startswith(prefix)
        ]

        if not matching:
            return None

        # Sort by version (simple string comparison)
        matching.sort(key=lambda x: x[0], reverse=True)
        return matching[0][1]

    def verify(
        self,
        tool_definition: dict[str, Any],
        publisher: str,
    ) -> tuple[bool, str]:
        """Verify a tool against the registry.

        Args:
            tool_definition: Tool definition to verify
            publisher: Expected publisher

        Returns:
            Tuple of (valid, reason)
        """
        tool_name = tool_definition.get("name", "")

        signature = self.get(tool_name, publisher)
        if not signature:
            return False, f"No registered signature for {publisher}:{tool_name}"

        return verify_tool(tool_definition, signature)
