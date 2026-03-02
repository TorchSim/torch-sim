"""Minimal logging for torch-sim.

By default all log output is written to ``~/.torchsim/torch_sim.log`` (rotating,
up to 10 MB x 5 backups, JSON-lines format) **and** to the console at INFO level.
The file always captures DEBUG; the console level can be changed.

The default log file accumulates entries across runs, which makes it easy to
review history, but means you need timestamps or ``--grep`` to isolate a single
run.  To direct logs for a specific script to its own file, call
``configure_logging`` at the top of that script::

    from torch_sim.telemetry import configure_logging, get_logger

    configure_logging(log_level="DEBUG", log_file="my_run.log")
    log = get_logger(__name__)
    log.info("starting simulation")

``configure_logging`` is idempotent — calling it again replaces the handlers, so
you can safely reconfigure mid-script if needed.  If you never call it explicitly,
logging auto-configures on the first ``get_logger()`` call using the defaults.
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from torch_sim import TORCH_SIM_CONFIG_DIR


_DEFAULT_LOG_FILE = TORCH_SIM_CONFIG_DIR / "torch_sim.log"
_configured = False


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    log_level: str = "INFO",
    log_file: Path | str | None = None,
) -> None:
    """Configure root logger with a console handler and a rotating JSON file handler.

    Args:
        log_level: Minimum level shown on the console (``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``). The file always captures ``DEBUG`` regardless.
        log_file: Path to the log file. Defaults to ``~/.torchsim/torch_sim.log``.
            Pass a relative or absolute path to redirect a specific run.

    Notes:
        - Safe to call multiple times; subsequent calls replace the existing handlers.
    """
    global _configured  # noqa: PLW0603
    log_file = Path(log_file) if log_file is not None else _DEFAULT_LOG_FILE

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # handlers filter individually

    # Rotating JSON file — always DEBUG so nothing is lost
    fh = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())

    # Human-readable console at the requested level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )

    root.addHandler(fh)
    root.addHandler(ch)
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, auto-configuring on first call."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
