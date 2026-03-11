"""Database results anonymizer for re-anonymizing query results."""

from typing import Any

from .storage import PIIMappingStore


class DatabaseResultsAnonymizer:
    """
    Re-anonymize database query results before sending to LLM.

    Query results may contain NEW PII not in the original question.
    This ensures the LLM never sees raw PII values in results.

    Uses the SHARED mapping to maintain placeholder consistency.
    """

    COLUMN_ENTITY_MAPPING = {
        "firstname": "PERSON",
        "first_name": "PERSON",
        "lastname": "PERSON",
        "last_name": "PERSON",
        "name": "PERSON",
        "customername": "PERSON",
        "customer_name": "PERSON",
        "employeename": "PERSON",
        "employee_name": "PERSON",
        "email": "EMAIL_ADDRESS",
        "emailaddress": "EMAIL_ADDRESS",
        "email_address": "EMAIL_ADDRESS",
        "phone": "PHONE_NUMBER",
        "phonenumber": "PHONE_NUMBER",
        "phone_number": "PHONE_NUMBER",
        "address": "LOCATION",
        "city": "LOCATION",
        "country": "LOCATION",
    }

    def __init__(self, mapping: PIIMappingStore):
        self.mapping = mapping
        self.new_mappings: dict[str, dict[str, str]] = {}

    def _get_entity_type(self, column: str) -> str | None:
        """Get entity type for a column using case-insensitive matching."""
        return self.COLUMN_ENTITY_MAPPING.get(column.lower())

    def anonymize_results(
        self,
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Anonymize PII columns in query results.

        - Uses case-insensitive column matching
        - Reuses existing placeholder if value matches (session consistency)
        - Generates new placeholder for new PII values
        """
        self.new_mappings = {}
        anonymized_results = []

        for row in results:
            anonymized_row = {}
            for column, value in row.items():
                entity_type = self._get_entity_type(column)
                if entity_type and value:
                    placeholder = self._get_or_create_placeholder(
                        entity_type, str(value)
                    )
                    anonymized_row[column] = placeholder
                else:
                    anonymized_row[column] = value
            anonymized_results.append(anonymized_row)

        return anonymized_results

    def _get_or_create_placeholder(self, entity_type: str, value: str) -> str:
        """Get existing placeholder or create new one."""
        existing = self.mapping.find_placeholder(entity_type, value)
        if existing:
            return existing

        count = self.mapping.get_next_counter(entity_type)
        placeholder = f"<{entity_type}_{count}>"
        self.mapping.add(entity_type, placeholder, value)

        if entity_type not in self.new_mappings:
            self.new_mappings[entity_type] = {}
        self.new_mappings[entity_type][placeholder] = value

        return placeholder

    def get_new_mappings(self) -> dict[str, dict[str, str]]:
        """Return mappings created during result anonymization."""
        return self.new_mappings
