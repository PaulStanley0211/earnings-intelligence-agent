"""Structured logging via loguru with trace-id propagation and secret scrubbing.

Loguru is configured to emit one JSON record per line on stdout. A context
variable carries the :data:`trace_id` through async boundaries, and a filter
strips obvious secrets (API keys, bearer tokens) from any field before the
record is serialised - PLAN.md section 7 requires this for any logs persisted
to disk.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final
from uuid import uuid4

from loguru import logger as _logger

_TRACE_ID: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Patterns we never want to see in a persisted log line. Matching is conservative
# on purpose: false positives only blank a value, false negatives leak secrets.
# The generic ``sk-`` prefix covers both Anthropic (``sk-ant-*``) and OpenAI
# (``sk-proj-*`` / ``sk-*``) provider keys, so all LLM provider keys are
# redacted by the first pattern. The ``sk-ant-`` entry below is retained for
# defence-in-depth in case the tuple order is ever changed.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=]+", re.IGNORECASE),
    re.compile(r"xoxb-[A-Za-z0-9\-]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)

_REDACTED: Final[str] = "[REDACTED]"


def _scrub(value: str) -> str:
    """Replace any matched secret pattern in ``value`` with a placeholder."""
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(_REDACTED, value)
    return value


def _serializer(record: Mapping[str, Any]) -> str:
    """Render a loguru record as a single JSON line, scrubbing the message.

    Loguru already serialises records via ``serialize=True``; this layer adds
    secret scrubbing on top.
    """
    payload: dict[str, Any] = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "msg": _scrub(record["message"]),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "trace_id": _TRACE_ID.get(),
    }
    extras = record.get("extra") or {}
    for key, raw in extras.items():
        payload[key] = _scrub(str(raw)) if isinstance(raw, str) else raw
    exception = record.get("exception")
    if exception is not None:
        payload["exception"] = _scrub(exception.repr)
    return json.dumps(payload, default=str) + "\n"


def _json_sink(message: Any) -> None:
    """Loguru sink that writes one scrubbed JSON record per line to stdout."""
    sys.stdout.write(_serializer(message.record))


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON sink, replacing loguru's default stderr handler.

    Idempotent - safe to call from application startup and from tests.
    """
    _logger.remove()
    _logger.add(
        _json_sink,
        level=level,
        format="{message}",
        backtrace=False,
        diagnose=False,
        catch=True,
    )


def get_logger() -> Any:
    """Return the loguru logger.

    Wrapped so callers do not import loguru directly - keeps the option open to
    swap implementations later without churn.
    """
    return _logger


def new_trace_id() -> str:
    """Generate a new opaque trace id."""
    return uuid4().hex


@contextmanager
def with_trace_id(trace_id: str | None = None) -> Iterator[str]:
    """Bind ``trace_id`` for the duration of the ``with`` block.

    A new id is generated when none is supplied. The previous value (often
    ``None``) is restored on exit.
    """
    effective = trace_id or new_trace_id()
    token = _TRACE_ID.set(effective)
    try:
        yield effective
    finally:
        _TRACE_ID.reset(token)


def current_trace_id() -> str | None:
    """Return the trace id currently bound in this async context, if any."""
    return _TRACE_ID.get()
