"""Logger setup used by the command-line pipeline and tests."""

import logging
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_file: str | None = None,
                 format_string: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    formatter = logging.Formatter(format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
