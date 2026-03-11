"""PII protection module.

Provides pseudonymization, detection, and safe execution
for PII-aware SQL generation.

See 2.3 for the complete PII protection pattern.
"""

from src.pii.detector import PIIDetector, PIIEntity, DetectionResult
from src.pii.pseudonymizer import (
    Pseudonymizer,
    PseudonymMapping,
    PseudonymizationResult,
)
from src.pii.aggregate_detector import detect_aggregate_leakage

__all__ = [
    # Detection
    "PIIDetector",
    "PIIEntity",
    "DetectionResult",
    # Pseudonymization
    "Pseudonymizer",
    "PseudonymMapping",
    "PseudonymizationResult",
    # Aggregate protection
    "detect_aggregate_leakage",
]
