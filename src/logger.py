# src/logger.py
"""
Educational Goal:
- Configure logging once so every module gets consistent, timestamped, leveled logs
- Log to console (developer and Render visibility) and, optionally, to a file (post-run audit trail)

How other modules use it
- Each module only does: logger = logging.getLogger(__name__)
- The entry point (main.py or api.py) calls configure_logging() once at startup
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(*, log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure root logging once for the whole process

    Inputs
    - log_level: "DEBUG", "INFO", "WARNING", "ERROR"
    - log_file: Path to a log file. None means console only (useful inside containers)

    Outputs
    - None. Every module logger now writes timestamped lines to the console (and the file)
    """
    numeric_level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers = [console_handler]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # force=True replaces any earlier logging setup, so we never get duplicated lines
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
