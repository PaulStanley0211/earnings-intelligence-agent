"""Postgres + pgvector access layer.

The project rule is that every interaction with the database goes through
this package - agents talk to :class:`~app.memory.repository.Repository`
and never import SQLAlchemy directly. The ORM models in
:mod:`app.memory.models` describe what is on disk; the DTOs in
:mod:`app.memory.schemas` describe what travels across the package
boundary.
"""

from app.memory.db import (
    build_engine,
    dispose_engine,
    get_engine,
    get_session,
    get_session_factory,
)
from app.memory.models import (
    Base,
    DailyLLMSpend,
    EdgarPollLog,
    Filing,
    FinancialFact,
    WatchlistEntry,
)
from app.memory.repository import Repository
from app.memory.schemas import (
    FilingRecord,
    FilingStatus,
    FinancialFactRecord,
    NewFiling,
    NewFinancialFact,
    NewPollLog,
    PollLogRecord,
    PollStatus,
    WatchlistRecord,
)

__all__ = [
    "Base",
    "DailyLLMSpend",
    "EdgarPollLog",
    "Filing",
    "FilingRecord",
    "FilingStatus",
    "FinancialFact",
    "FinancialFactRecord",
    "NewFiling",
    "NewFinancialFact",
    "NewPollLog",
    "PollLogRecord",
    "PollStatus",
    "Repository",
    "WatchlistEntry",
    "WatchlistRecord",
    "build_engine",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
]
