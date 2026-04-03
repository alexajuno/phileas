"""Structured JSON logging for Phileas operations."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter

_DEFAULT_LOG_DIR = Path.home() / ".phileas"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3


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


def get_logger(
    *,
    log_dir: Path | None = None,
    level: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Logger:
    """Return the Phileas logger, creating it on first call.

    All parameters are optional and default to the original hardcoded values:
      log_dir     – directory for log files (default ``~/.phileas``)
      level       – logging level name (default ``"INFO"``)
      max_bytes   – max bytes per rotated log file (default 5 MB)
      backup_count – number of rotated backup files (default 3)
    """
    logger = logging.getLogger("phileas")
    if logger.handlers:
        return logger

    resolved_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    resolved_level = level if level is not None else "INFO"
    resolved_max_bytes = max_bytes if max_bytes is not None else _DEFAULT_MAX_BYTES
    resolved_backup_count = backup_count if backup_count is not None else _DEFAULT_BACKUP_COUNT

    logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))

    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_dir / "phileas.log"
    handler = RotatingFileHandler(
        log_file, maxBytes=resolved_max_bytes, backupCount=resolved_backup_count
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
