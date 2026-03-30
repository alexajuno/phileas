"""Structured JSON logging for Phileas operations."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter

LOG_DIR = Path.home() / ".phileas"
LOG_FILE = LOG_DIR / "phileas.log"

# 5 MB per file, keep 3 rotated files
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "op": getattr(record, "op", None),
            "msg": record.getMessage(),
        }
        # Merge extra structured data
        data = getattr(record, "data", None)
        if data:
            entry["data"] = data
        return json.dumps(entry, default=str)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("phileas")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


class OpTimer:
    """Context manager that times an operation and logs the result."""

    def __init__(self, logger: logging.Logger, op: str, **extra: object) -> None:
        self.logger = logger
        self.op = op
        self.extra = extra
        self.start = 0.0

    def __enter__(self) -> "OpTimer":
        self.start = perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed_ms = (perf_counter() - self.start) * 1000
        data = {**self.extra, "elapsed_ms": round(elapsed_ms, 2)}
        self.logger.info(
            self.op,
            extra={"op": self.op, "data": data},
        )
