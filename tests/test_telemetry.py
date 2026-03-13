"""Tests for torch_sim.telemetry."""

import json
import logging
from pathlib import Path

import pytest

from torch_sim import telemetry


@pytest.fixture(autouse=True)
def reset_logging(tmp_path: Path):
    """Restore root logger state and telemetry._configured after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    original_configured = telemetry._configured  # noqa: SLF001

    yield tmp_path

    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)
    telemetry._configured = original_configured  # noqa: SLF001


def test_configure_logging_adds_two_handlers(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="DEBUG", log_file=log_file)

    root = logging.getLogger()
    assert len(root.handlers) == 2


def test_configure_logging_creates_log_file(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_file=log_file)

    logging.getLogger("test").info("hello")
    assert log_file.exists()


def test_file_handler_writes_valid_json(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="DEBUG", log_file=log_file)

    logging.getLogger("mymodule").info("structured message", extra={})
    logging.getLogger().handlers[0].flush()

    lines = log_file.read_text().splitlines()
    assert lines, "log file is empty"
    record = json.loads(lines[0])
    assert record["level"] == "INFO"
    assert record["message"] == "structured message"
    assert "time" in record
    assert "name" in record


def test_json_record_includes_exception(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="DEBUG", log_file=log_file)

    try:
        raise ValueError("boom")  # noqa: TRY301
    except ValueError:
        logging.getLogger("err").exception("caught")

    logging.getLogger().handlers[0].flush()
    record = json.loads(log_file.read_text().splitlines()[0])
    assert "exception" in record
    assert "ValueError" in record["exception"]


def test_console_handler_respects_log_level(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="WARNING", log_file=log_file)

    logging.getLogger("quiet").debug("should not appear")
    logging.getLogger("quiet").warning("should appear")

    captured = capsys.readouterr()
    assert "should not appear" not in captured.out
    assert "should appear" in captured.out


def test_file_handler_captures_debug_regardless_of_console_level(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="WARNING", log_file=log_file)

    logging.getLogger("verbose").debug("debug only in file")
    logging.getLogger().handlers[0].flush()

    record = json.loads(log_file.read_text().splitlines()[0])
    assert record["message"] == "debug only in file"


def test_configure_logging_is_idempotent(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_level="INFO", log_file=log_file)
    telemetry.configure_logging(log_level="INFO", log_file=log_file)

    assert len(logging.getLogger().handlers) == 2


def test_get_logger_returns_named_logger(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_file=log_file)

    logger = telemetry.get_logger("mymodule")
    assert logger.name == "mymodule"


def test_get_logger_auto_configures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(telemetry, "_configured", False)
    monkeypatch.setattr(telemetry, "_DEFAULT_LOG_FILE", tmp_path / "auto.log")

    logger = telemetry.get_logger("auto")
    assert telemetry._configured  # noqa: SLF001
    assert isinstance(logger, logging.Logger)


def test_get_logger_none_returns_root(tmp_path: Path):
    log_file = tmp_path / "test.log"
    telemetry.configure_logging(log_file=log_file)

    logger = telemetry.get_logger()
    assert logger is logging.getLogger()
