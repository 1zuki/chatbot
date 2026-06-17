from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    return configure_file_logger("vista_chatbot", log_file, level=level, stream=True)


def configure_file_logger(
    logger_name: str,
    log_file: Path,
    level: str = "INFO",
    *,
    stream: bool = False,
) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    return logger
