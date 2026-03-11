"""Security module for SQL agent protection.

This module provides defense-in-depth security controls:
- Input sanitization (prompt injection defense)
- Output filtering (sensitive data protection)
- Audit logging (forensics and anomaly detection)
- MCP firewall (tool governance)
- JIT permissions (zero-trust architecture)

See 2.1 for threat modeling and security patterns.
"""

from src.security.input_sanitizer import sanitize_input, SanitizationResult
from src.security.output_filter import filter_output, FilteredOutput
from src.security.audit import SecurityAuditor, AuditEvent
from src.security.mcp_firewall import MCPFirewall, ToolPolicy, ToolPermission
from src.security.jit_permissions import JITPermissionManager, PermissionGrant

__all__ = [
    # Input sanitization
    "sanitize_input",
    "SanitizationResult",
    # Output filtering
    "filter_output",
    "FilteredOutput",
    # Audit logging
    "SecurityAuditor",
    "AuditEvent",
    # MCP firewall
    "MCPFirewall",
    "ToolPolicy",
    "ToolPermission",
    # JIT permissions
    "JITPermissionManager",
    "PermissionGrant",
]
