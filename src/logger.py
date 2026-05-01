"""
Logging utilities for BSS-Test framework.

Provides centralized logging configuration with:
- Console and file output
- Log level configuration
- Formatted output with timestamps
- Module-specific loggers

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
    log_dir: str = "outputs/logs",
    console_output: bool = True,
    file_output: bool = True,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Configure logging for the entire application.

    Parameters
    ----------
    level : str
        Logging level: "debug", "info", "warning", "error", "critical"
    log_file : str or None
        Specific log file path. If None, auto-generates based on timestamp.
    log_dir : str
        Directory for log files (used if log_file is None).
    console_output : bool
        Enable console output.
    file_output : bool
        Enable file output.
    log_format : str
        Log message format string.
    date_format : str
        Date format string.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    global _logging_configured

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_file is None:
            # Auto-generate log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir_path / f"bss_test_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logging_configured = True

    # Log startup message
    root_logger.info(f"Logging initialized (level={level.upper()})")
    if file_output and log_file:
        root_logger.info(f"Log file: {log_file}")

    return root_logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger instance.

    If logging hasn't been configured yet, sets up with default settings.

    Parameters
    ----------
    name : str
        Logger name, typically __name__.
    level : str or None
        Optional specific level for this logger.

    Returns
    -------
    logging.Logger
        Named logger instance.
    """
    global _logging_configured

    if not _logging_configured:
        setup_logging()

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))

    return logger


def get_log_file_path() -> Optional[str]:
    """
    Get the current log file path.

    Returns
    -------
    str or None
        Path to current log file, or None if not logging to file.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


class LoggerMixin:
    """
    Mixin class that adds logging capability to any class.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# Convenience functions for quick logging
def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function entry and exit.

    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            _logger.debug(f"Calling {func.__name__}({args}, {kwargs})")
            try:
                result = func(*args, **kwargs)
                _logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                _logger.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Usage:
        @log_execution_time()
        def my_function():
            pass
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                _logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator
