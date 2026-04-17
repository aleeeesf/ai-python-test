"""Centralized logging configuration with colored output for the notification service.

This module provides a standardized logger that can be imported across all modules.
Any changes to logging behavior here will automatically apply to all modules.

Example:
    >>> from infrastructure.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Request processed | request_id=req-123 | status=sent")
    [INFO] [2026-04-17 10:30:45] [process_request.py:42] Request processed | request_id=req-123 | status=sent

Features:
    - Colored output for easy visual distinction
    - Centralized configuration
    - Module-specific log level overrides during development
    - Environment-based enable/disable
"""

import logging
import sys
from typing import ClassVar

from core.settings import settings


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log level names.

    Provides visual distinction for different log levels in terminal output,
    making it easier to spot errors and warnings during development and debugging.

    Colors:
        DEBUG: Cyan | INFO: Green | WARNING: Yellow | ERROR: Red | CRITICAL: Bold Red
    """

    COLORS: ClassVar = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET: ClassVar = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colored level name.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with ANSI color codes.
        """
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str = __name__, log_level: str | None = None) -> logging.Logger:
    """Create or retrieve a configured logger with colored output.

    This is the primary entry point for all modules. Import and use this function
    to get a standardized logger instance. Any changes to the logging format or
    behavior will automatically propagate to all modules using this function.

    Args:
        name: Logger name, typically __name__ of the calling module.
              This provides automatic module identification in logs.
        log_level: Optional log level override for this specific logger.
                   If provided, it overrides the global settings.log_level.
                   Valid values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
                   Use this to enable detailed logging in specific modules during development.

    Returns:
        Configured logger instance with colored level output to stdout.
        If LOGGING_ENABLED is False in settings, returns a no-op logger.

    Example:
        >>> from infrastructure.logging import get_logger
        >>>
        >>> # Default: uses settings.log_level (e.g., INFO)
        >>> logger = get_logger(__name__)
        >>>
        >>> # Module-specific override: force DEBUG level for this module only
        >>> logger = get_logger(__name__, log_level="DEBUG")
        >>>
        >>> # Simple logging
        >>> logger.info("Processing started")
        [INFO] [2026-04-17 13:44:25] [processor.py:42] Processing started
        >>>
        >>> # Logging with context (manually formatted)
        >>> logger.info(f"Notification delivered | request_id={request_id} | provider_id={provider_id}")
        [INFO] [2026-04-17 13:44:30] [process.py:58] Notification delivered | request_id=req-123 | provider_id=p-999
        >>>
        >>> # Error logging with exception trace
        >>> try:
        ...     raise ValueError("Invalid request")
        ... except Exception as e:
        ...     logger.error(f"Delivery failed | request_id={request_id} | error={e}", exc_info=True)

    Best Practices:
        - Always include request_id for tracing operations
        - Use f-strings to include context: f"Message | key1={value1} | key2={value2}"
        - Use exc_info=True for error logging to capture full stack traces
        - Use log_level parameter only during development to debug specific modules
        - Never log sensitive data directly
    """

    # Return no-op logger if logging is disabled
    if not settings.logging_enabled:
        return logging.getLogger(name)

    # Determine log level: module override > global settings
    level_str = log_level if log_level is not None else settings.log_level
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = (
        False  # Prevent log messages from being propagated to the root logger
    )

    # Add handler only if it doesn't exist (avoid duplicate logs)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = ColorFormatter(
            fmt="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # If logger already exists but log_level is overridden, update the level
        if log_level is not None:
            logger.setLevel(level)
            for handler in logger.handlers:  # type: ignore
                handler.setLevel(level)

    return logger
