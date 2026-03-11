"""PII detection using Microsoft Presidio.

Presidio provides robust NER-based detection for names and pattern-based
detection for structured PII like emails and phone numbers.

Installation:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
"""

from dataclasses import dataclass
from typing import Protocol
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


@dataclass
class PIIEntity:
    """A detected PII entity."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float


class PIIDetector(Protocol):
    """Protocol for PII detectors."""

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII entities in text."""
        ...


class PresidioPIIDetector:
    """PII detector using Microsoft Presidio with spaCy NER.

    Uses spaCy's en_core_web_lg model for robust name detection
    and Presidio's pattern recognizers for structured PII.
    """

    # Entity types we detect
    DEFAULT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
    ]

    def __init__(
        self,
        entities: list[str] | None = None,
        spacy_model: str = "en_core_web_lg"
    ):
        """Initialize Presidio detector.

        Args:
            entities: Entity types to detect. Defaults to DEFAULT_ENTITIES.
            spacy_model: spaCy model for NER. Defaults to en_core_web_lg.
        """

        self.entities = entities or self.DEFAULT_ENTITIES

        # Configure spaCy NLP engine
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}]
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII entities using Presidio.

        Args:
            text: Input text to analyze.

        Returns:
            List of detected PII entities, sorted by position.
        """
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities
        )

        entities = []
        for result in results:
            entities.append(PIIEntity(
                entity_type=result.entity_type,
                text=text[result.start:result.end],
                start=result.start,
                end=result.end,
                score=result.score
            ))

        return sorted(entities, key=lambda e: e.start)


def create_detector(use_presidio: bool = True) -> PIIDetector:
    return PresidioPIIDetector()
