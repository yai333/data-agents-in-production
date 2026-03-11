"""Retry logic for SQL generation.

This module determines when to retry after an error and when to give up.
Bounded retries prevent infinite loops.

See 1.6 for retry configuration.
"""

from dataclasses import dataclass

from src.agent.error_classifier import ClassifiedError, ErrorType


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    retry_syntax: bool = True
    retry_schema: bool = True
    retry_type: bool = True
    retry_timeout: bool = False  # Usually needs query redesign, not retry
    retry_unknown: bool = True


def should_retry(
    error: ClassifiedError,
    attempt: int,
    config: RetryConfig | None = None,
) -> bool:
    """Determine if we should retry after an error.

    Args:
        error: The classified error
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        True if should retry, False otherwise

    Example:
        >>> error = ClassifiedError(ErrorType.SYNTAX, "...", "...", True)
        >>> should_retry(error, attempt=1)
        True
        >>> should_retry(error, attempt=3)
        False
    """
    if config is None:
        config = RetryConfig()

    # Check max attempts
    if attempt >= config.max_attempts:
        return False

    # Check if error is retryable at all
    if not error.retryable:
        return False

    # Check error-type-specific retry settings
    retry_map = {
        ErrorType.SYNTAX: config.retry_syntax,
        ErrorType.SCHEMA: config.retry_schema,
        ErrorType.TYPE: config.retry_type,
        ErrorType.TIMEOUT: config.retry_timeout,
        ErrorType.UNKNOWN: config.retry_unknown,
        ErrorType.PERMISSION: False,  # Never retry permission errors
        ErrorType.CONSTRAINT: False,  # Never retry constraint errors
    }

    return retry_map.get(error.error_type, False)


def get_retry_delay_ms(attempt: int) -> int:
    """Calculate delay before retry (exponential backoff).

    Args:
        attempt: Current attempt number (1-indexed)

    Returns:
        Delay in milliseconds
    """
    # Exponential backoff: 100ms, 200ms, 400ms, ...
    return min(100 * (2 ** (attempt - 1)), 5000)  # Cap at 5 seconds
