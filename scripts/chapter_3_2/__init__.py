"""Chapter 3.2: PII Protection - Three-Phase Pipeline.

This module implements the PII protection mechanisms from Chapter 3.2:
    PHASE 1: PROTECT  - Detect and pseudonymize PII before LLM sees it
    PHASE 2: CONSTRAIN - Build dynamic allow-list for structured output
    PHASE 3: GENERATE - LLM generates SQL with placeholders only

Plus: DatabaseResultsAnonymizer for re-anonymizing query results

Requirements:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
"""

from .storage import PIIMappingStore
from .detector import PresidioPIIDetector, PIIDetector, PIIEntity, create_detector
from .anonymizer import DatabaseResultsAnonymizer
from .core import (
    detect_and_pseudonymize,
    extract_placeholders,
    build_allow_list,
    resolve_parameters,
    deanonymize,
    SQLParameter,
    ParameterType,
)

__all__ = [
    # Storage
    "PIIMappingStore",
    # Detection (Presidio-based)
    "PresidioPIIDetector",
    "PIIDetector",
    "PIIEntity",
    "create_detector",
    # Anonymization
    "DatabaseResultsAnonymizer",
    # Core functions
    "detect_and_pseudonymize",
    "extract_placeholders",
    "build_allow_list",
    "resolve_parameters",
    "deanonymize",
    "SQLParameter",
    "ParameterType",
]
