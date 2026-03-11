"""PII detection using Microsoft Presidio.

Detects personally identifiable information in text for
pseudonymization before LLM processing.

See 2.3 for the PII protection pattern.
"""

from dataclasses import dataclass, field
from typing import Any

# Presidio imports are optional - provide fallback
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


@dataclass
class PIIEntity:
    """A detected PII entity.

    Attributes:
        entity_type: Type of PII (PERSON, EMAIL, etc.)
        text: The actual PII value
        start: Start position in text
        end: End position in text
        score: Confidence score (0-1)
    """

    entity_type: str
    text: str
    start: int
    end: int
    score: float


@dataclass
class DetectionResult:
    """Result of PII detection.

    Attributes:
        original_text: The input text
        entities: Detected PII entities
        has_pii: Whether any PII was detected
    """

    original_text: str
    entities: list[PIIEntity]
    has_pii: bool = field(init=False)

    def __post_init__(self):
        self.has_pii = len(self.entities) > 0


class PIIDetector:
    """Detect PII in text using Presidio.

    Supports detection of:
    - PERSON: Names
    - EMAIL_ADDRESS: Email addresses
    - PHONE_NUMBER: Phone numbers
    - CREDIT_CARD: Credit card numbers
    - US_SSN: Social Security Numbers
    - LOCATION: Addresses, cities, countries
    - And many more entity types

    Example:
        >>> detector = PIIDetector()
        >>> result = detector.detect("Contact John at john@example.com")
        >>> result.has_pii
        True
        >>> [e.entity_type for e in result.entities]
        ['PERSON', 'EMAIL_ADDRESS']
    """

    # Entity types we care about for SQL agents
    DEFAULT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
        "LOCATION",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
    ]

    def __init__(
        self,
        entities: list[str] | None = None,
        min_score: float = 0.5,
    ):
        """Initialize the detector.

        Args:
            entities: Entity types to detect (None = defaults)
            min_score: Minimum confidence score

        Raises:
            ImportError: If Presidio is not installed
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.min_score = min_score
        self.analyzer = None

        if PRESIDIO_AVAILABLE:
            self._init_analyzer()

    def _init_analyzer(self):
        """Initialize the Presidio analyzer."""
        try:
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception:
            # Fall back to simpler spaCy model if lg not available
            try:
                nlp_config = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
                provider = NlpEngineProvider(nlp_configuration=nlp_config)
                nlp_engine = provider.create_engine()
                self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            except Exception:
                self.analyzer = None

    def detect(self, text: str) -> DetectionResult:
        """Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            DetectionResult with detected entities
        """
        if not PRESIDIO_AVAILABLE or self.analyzer is None:
            # Fallback to regex-based detection
            return self._detect_fallback(text)

        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en",
        )

        entities = [
            PIIEntity(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=r.score,
            )
            for r in results
            if r.score >= self.min_score
        ]

        # Sort by position for consistent processing
        entities.sort(key=lambda e: e.start)

        return DetectionResult(
            original_text=text,
            entities=entities,
        )

    def _detect_fallback(self, text: str) -> DetectionResult:
        """Fallback regex-based PII detection.

        Used when Presidio is not available.
        """
        import re

        entities = []

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(PIIEntity(
                entity_type="EMAIL_ADDRESS",
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.9,
            ))

        # Phone pattern (US)
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append(PIIEntity(
                entity_type="PHONE_NUMBER",
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.85,
            ))

        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            entities.append(PIIEntity(
                entity_type="US_SSN",
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.95,
            ))

        # Credit card pattern
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        for match in re.finditer(cc_pattern, text):
            entities.append(PIIEntity(
                entity_type="CREDIT_CARD",
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.9,
            ))

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return DetectionResult(
            original_text=text,
            entities=entities,
        )

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains PII.

        Args:
            text: Text to check

        Returns:
            True if PII detected
        """
        return self.detect(text).has_pii


def is_presidio_available() -> bool:
    """Check if Presidio is available.

    Returns:
        True if Presidio can be used
    """
    return PRESIDIO_AVAILABLE
