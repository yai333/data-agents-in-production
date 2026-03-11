"""Core PII protection functions for the three-phase pipeline."""

import re
from typing import Any
from enum import Enum
from dataclasses import dataclass

from .storage import PIIMappingStore
from .detector import PIIDetector


class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass
class SQLParameter:
    param_name: str
    placeholder_ref: str
    param_type: ParameterType = ParameterType.STRING


def detect_and_pseudonymize(
    text: str,
    mapping: PIIMappingStore,
    detector: PIIDetector | None = None
) -> str:
    """PHASE 1: PROTECT - Detect PII and replace with placeholders."""
    if detector is None:
        from .detector import PresidioPIIDetector
        detector = PresidioPIIDetector()

    entities = detector.detect(text)
    entities = sorted(entities, key=lambda e: e.start, reverse=True)

    pseudonymized = text
    for entity in entities:
        entity_type = entity.entity_type
        existing = mapping.find_placeholder(entity_type, entity.text)
        if existing:
            placeholder = existing
        else:
            count = mapping.get_next_counter(entity_type)
            placeholder = f"<{entity_type}_{count}>"
            mapping.add(entity_type, placeholder, entity.text)

        pseudonymized = (
            pseudonymized[:entity.start] +
            placeholder +
            pseudonymized[entity.end:]
        )

    return pseudonymized


def extract_placeholders(text: str) -> list[str]:
    return re.findall(r'<[A-Z_]+_\d+>', text)


def build_allow_list(pseudonymized_question: str) -> list[str]:
    """PHASE 2: CONSTRAIN - Build allow-list from detected placeholders."""
    return extract_placeholders(pseudonymized_question)


def resolve_parameters(
    parameters: list[SQLParameter],
    mapping: PIIMappingStore
) -> dict[str, Any]:
    """Resolve placeholders to real values for database execution."""
    bound_params = {}
    placeholder_pattern = re.compile(r'^<[A-Z_]+_\d+>$')

    for param in parameters:
        ref = param.placeholder_ref

        if placeholder_pattern.match(ref):
            original = mapping.get_original(ref)
            if original is None:
                raise ValueError(f"Unknown placeholder: {ref}")
            bound_params[param.param_name] = original
        else:
            if param.param_type == ParameterType.INTEGER:
                bound_params[param.param_name] = int(ref)
            elif param.param_type == ParameterType.FLOAT:
                bound_params[param.param_name] = float(ref)
            else:
                bound_params[param.param_name] = ref

    return bound_params


def deanonymize(text: str, mapping: PIIMappingStore) -> str:
    """Restore original PII values from placeholders."""
    result = text
    for entity_type, entities in mapping.mappings.items():
        for placeholder, original in entities.items():
            result = result.replace(placeholder, original)
    return result
