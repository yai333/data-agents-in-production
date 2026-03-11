"""Pseudonymization for PII protection.

Replaces PII with reversible pseudonyms to protect sensitive
data during LLM processing.

See 2.3 for the PII protection pattern.
"""

from dataclasses import dataclass, field
from typing import Any

from src.pii.detector import PIIDetector, DetectionResult, PIIEntity


@dataclass
class PseudonymMapping:
    """Mapping between pseudonyms and real values.

    WARNING: This mapping contains sensitive data.
    Never log or persist this mapping.

    Attributes:
        pseudonym_to_real: Map pseudonym → real value
        real_to_pseudonym: Map real value → pseudonym
        entity_types: Map pseudonym → entity type
    """

    pseudonym_to_real: dict[str, str] = field(default_factory=dict)
    real_to_pseudonym: dict[str, str] = field(default_factory=dict)
    entity_types: dict[str, str] = field(default_factory=dict)

    def add(self, pseudonym: str, real_value: str, entity_type: str) -> None:
        """Add a mapping.

        Args:
            pseudonym: The pseudonym (e.g., PERSON_1)
            real_value: The real PII value
            entity_type: Type of PII
        """
        self.pseudonym_to_real[pseudonym] = real_value
        self.real_to_pseudonym[real_value] = pseudonym
        self.entity_types[pseudonym] = entity_type

    def get_real(self, pseudonym: str) -> str | None:
        """Get real value from pseudonym.

        Args:
            pseudonym: The pseudonym

        Returns:
            Real value or None
        """
        return self.pseudonym_to_real.get(pseudonym)

    def get_pseudonym(self, real_value: str) -> str | None:
        """Get pseudonym from real value.

        Args:
            real_value: The real PII value

        Returns:
            Pseudonym or None
        """
        return self.real_to_pseudonym.get(real_value)

    def __len__(self) -> int:
        """Return number of mappings."""
        return len(self.pseudonym_to_real)


@dataclass
class PseudonymizationResult:
    """Result of pseudonymization.

    Attributes:
        original_text: The input text
        pseudonymized_text: Text with PII replaced
        mapping: The pseudonym ↔ real value mapping
        entity_count: Number of entities pseudonymized
    """

    original_text: str
    pseudonymized_text: str
    mapping: PseudonymMapping
    entity_count: int


class Pseudonymizer:
    """Replace PII with reversible pseudonyms.

    Example:
        >>> p = Pseudonymizer()
        >>> result = p.pseudonymize("Contact John at john@example.com")
        >>> result.pseudonymized_text
        'Contact PERSON_1 at EMAIL_1'
        >>> result.mapping.get_real("PERSON_1")
        'John'
    """

    # Prefixes for each entity type
    ENTITY_PREFIXES = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CARD",
        "US_SSN": "SSN",
        "LOCATION": "LOCATION",
        "IBAN_CODE": "IBAN",
        "IP_ADDRESS": "IP",
        "DATE_TIME": "DATE",
        "US_PASSPORT": "PASSPORT",
        "US_DRIVER_LICENSE": "LICENSE",
    }

    def __init__(self, detector: PIIDetector | None = None):
        """Initialize the pseudonymizer.

        Args:
            detector: PII detector (creates default if None)
        """
        self.detector = detector or PIIDetector()

    def pseudonymize(self, text: str) -> PseudonymizationResult:
        """Pseudonymize PII in text.

        Args:
            text: Text to pseudonymize

        Returns:
            PseudonymizationResult with pseudonymized text and mapping
        """
        detection = self.detector.detect(text)

        if not detection.has_pii:
            return PseudonymizationResult(
                original_text=text,
                pseudonymized_text=text,
                mapping=PseudonymMapping(),
                entity_count=0,
            )

        mapping = PseudonymMapping()
        counters: dict[str, int] = {}

        # Process entities from end to start (to preserve positions)
        sorted_entities = sorted(
            detection.entities,
            key=lambda e: e.start,
            reverse=True,
        )
        result_text = text

        for entity in sorted_entities:
            # Check if we already have a pseudonym for this value
            existing = mapping.get_pseudonym(entity.text)
            if existing:
                pseudonym = existing
            else:
                # Generate new pseudonym
                prefix = self.ENTITY_PREFIXES.get(entity.entity_type, "ENTITY")
                counter = counters.get(prefix, 0) + 1
                counters[prefix] = counter
                pseudonym = f"{prefix}_{counter}"
                mapping.add(pseudonym, entity.text, entity.entity_type)

            # Replace in text
            result_text = (
                result_text[:entity.start] +
                pseudonym +
                result_text[entity.end:]
            )

        return PseudonymizationResult(
            original_text=text,
            pseudonymized_text=result_text,
            mapping=mapping,
            entity_count=len(mapping),
        )

    def depseudonymize(self, text: str, mapping: PseudonymMapping) -> str:
        """Restore pseudonymized text to original.

        Args:
            text: Pseudonymized text
            mapping: The pseudonym mapping

        Returns:
            Text with real values restored

        Warning:
            Only call this with trusted text (e.g., database results),
            never with LLM output that might contain injected pseudonyms.
        """
        result = text
        # Sort by length (longest first) to avoid partial replacements
        sorted_pseudonyms = sorted(
            mapping.pseudonym_to_real.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )
        for pseudonym, real_value in sorted_pseudonyms:
            result = result.replace(pseudonym, real_value)
        return result


def create_pseudonymizer(min_score: float = 0.5) -> Pseudonymizer:
    """Create a pseudonymizer with custom settings.

    Args:
        min_score: Minimum confidence for PII detection

    Returns:
        Configured Pseudonymizer
    """
    detector = PIIDetector(min_score=min_score)
    return Pseudonymizer(detector=detector)
