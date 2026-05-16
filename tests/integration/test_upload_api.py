"""End-to-end tests for /api/advise and /api/upload."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_compiled_graph, get_edgar_client
from app.main import create_app
from app.memory.db import build_engine, dispose_engine, get_engine, get_session_factory
from app.memory.models import Base
from app.memory.repository import Repository
from app.models.state import AgentState
from app.tools.edgar import RecentFiling, SubmissionsResponse

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "uploaded_pdfs"


class _FakeEdgar:
    """Stub EDGAR client matching :class:`EdgarClient`'s submission contract."""

    async def get_submissions(self, *, cik: str) -> SubmissionsResponse:
        """Return a fixed pair of recent filings (one 8-K, one 10-Q)."""
        return SubmissionsResponse(
            cik=cik,
            entity_name="Microsoft Corp",
            tickers=["MSFT"],
            sic_description=None,
            recent_filings=[
                RecentFiling(
                    accession_number="0001193125-26-191457",
                    form="8-K",
                    filing_date=date(2026, 4, 29),
                    report_date=date(2026, 4, 29),
                    primary_document="msft-20260429.htm",
                ),
                RecentFiling(
                    accession_number="0001193125-26-027207",
                    form="10-Q",
                    filing_date=date(2026, 1, 28),
                    report_date=date(2025, 12, 31),
                    primary_document="msft-20260128.htm",
                ),
            ],
        )


class _StubCompiledGraph:
    """Mimics a compiled LangGraph: ``ainvoke(state) -> dict``.

    Returns a state dict with the synthesised fields populated so the
    upload route can build a complete :class:`UploadResponse` without
    actually exercising specialist nodes (those have their own tests).

    NOTE: this stub bypasses LangGraph's reducer entirely and therefore
    does NOT exercise the per-field ownership contract enforced by
    :data:`app.models.state._FIELD_OWNERS` via :class:`StateUpdate`. A
    passing test here does not rule out regressions in the real compiled
    graph's field-ownership behaviour - those need to be caught by
    node-level unit tests and the full-graph integration tests that
    build via ``build_graph``.
    """

    async def ainvoke(
        self, state: AgentState | dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        """Return a successful final-state dict.

        The synthesised fields are written unconditionally so the response
        does not pick up the ``None`` defaults that come back through
        :meth:`AgentState.model_dump`.
        """
        if isinstance(state, AgentState):
            payload: dict[str, Any] = state.model_dump()
        else:
            payload = dict(state)
        payload["financials"] = {
            "source": "uploaded",
            "revenue_usd": 61_858_000_000,
        }
        payload["comparisons"] = {"consensus_source": "finnhub", "metrics": []}
        payload["language_diffs"] = []
        payload["draft_note"] = "MSFT reported revenue of $61.9 billion [F1]."
        payload["final_note"] = "MSFT reported revenue of $61.9 billion [F1]."
        payload["critic_verdict"] = "accepted"
        return payload


@pytest_asyncio.fixture()
async def fresh_schema() -> AsyncIterator[None]:
    """Reset the schema and process-wide engine before each test."""
    await dispose_engine()
    engine = build_engine(echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    await dispose_engine()
    _ = get_engine()
    yield
    await dispose_engine()


@pytest_asyncio.fixture()
async def seed_watchlist_msft(fresh_schema: None) -> AsyncIterator[None]:
    """Insert MSFT into the watchlist so the advisor can resolve it."""
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await Repository(session).upsert_watchlist_entry(
            ticker="MSFT",
            cik="0000789019",
            company_name="Microsoft Corp",
            active=True,
        )
        await session.commit()
    yield


async def _stub_edgar() -> AsyncIterator[_FakeEdgar]:
    """Async generator yielding a fake EDGAR client."""
    yield _FakeEdgar()


def _stub_compiled_graph() -> _StubCompiledGraph:
    """Sync dependency returning the stubbed compiled graph."""
    return _StubCompiledGraph()


@pytest_asyncio.fixture()
async def app_with_fake_edgar() -> AsyncIterator[AsyncClient]:
    """Build a FastAPI app with the EDGAR dependency overridden to the fake."""
    app = create_app()
    app.dependency_overrides[get_edgar_client] = _stub_edgar
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def app_with_stubbed_graph() -> AsyncIterator[AsyncClient]:
    """Build the FastAPI app with /api/upload's graph dependency stubbed.

    The EDGAR client is also stubbed for symmetry, even though /api/upload
    does not invoke it directly: the production ``get_compiled_graph`` wires
    a live :class:`EdgarClient` into the graph, and the override here keeps
    test environments off the SEC network.
    """
    app = create_app()
    app.dependency_overrides[get_edgar_client] = _stub_edgar
    app.dependency_overrides[get_compiled_graph] = _stub_compiled_graph
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_advise_for_msft_returns_checklist(
    seed_watchlist_msft: None, app_with_fake_edgar: AsyncClient
) -> None:
    """A seeded ticker yields a 200 with at least the 8-K in the checklist."""
    response = await app_with_fake_edgar.post("/api/advise", json={"ticker": "MSFT"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert any(s["filing_type"] == "8-K" for s in payload["suggested"])
    assert "transcript" in payload["transcript_hint"].lower()


async def test_advise_unknown_ticker_404(
    fresh_schema: None, app_with_fake_edgar: AsyncClient
) -> None:
    """An unseeded ticker yields a 404 mentioning 'watchlist'."""
    response = await app_with_fake_edgar.post(
        "/api/advise", json={"ticker": "ZZZZZZ"}
    )
    assert response.status_code == 404
    assert "watchlist" in response.json()["detail"].lower()


async def test_upload_msft_8k_runs_pipeline_to_final_note(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """Uploading a real MSFT 8-K PDF returns a populated analysis payload."""
    pdf_bytes = (_FIXTURES / "0001193125-26-027198.pdf").read_bytes()
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("msft-8k.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["upload_id"]
    assert payload["trace_id"]
    assert payload["analysis"]["final_note"] is not None
    assert payload["analysis"]["critic_verdict"] == "accepted"


async def test_upload_rejects_too_large(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversize body yields 413 before any parsing happens."""
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    from app.config import reset_settings_cache

    reset_settings_cache()
    try:
        big = b"%PDF-1.4\n" + b"A" * 2048
        response = await app_with_stubbed_graph.post(
            "/api/upload",
            data={"ticker": "MSFT", "filing_type": "8-K"},
            files={"file": ("big.pdf", big, "application/pdf")},
        )
        assert response.status_code == 413
    finally:
        monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)
        reset_settings_cache()


async def test_upload_rejects_wrong_content_type(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """A non-PDF/non-plain-text content type yields 415."""
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


async def test_upload_rejects_scanned_pdf(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """A PDF with no extractable text yields 422 with a clear message."""
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000055 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n156\n%%EOF"
    )
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "8-K"},
        files={"file": ("scan.pdf", empty_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "scanned" in detail or "extractable" in detail


async def test_upload_accepts_transcript_filing_type_plain_text(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """``filing_type=TRANSCRIPT`` with text/plain returns 200 (spec §3.5)."""
    transcript_body = (
        b"Operator: Good afternoon and welcome to the Microsoft fiscal Q4 "
        b"earnings call. With us today are Satya Nadella, chief executive "
        b"officer, and Amy Hood, chief financial officer.\n\n"
        b"Satya Nadella: Thank you, and good afternoon, everyone. We had a "
        b"strong quarter, with revenue of 61.9 billion dollars, up 17 percent "
        b"year over year. Cloud revenue grew 23 percent. AI-driven workloads "
        b"continue to drive meaningful customer adoption across all clouds.\n\n"
        b"Amy Hood: Thanks, Satya. For the full year, we generated 245 "
        b"billion in revenue. Operating margin expanded 100 basis points. "
        b"Looking ahead, we expect double-digit revenue and operating-income "
        b"growth in the coming year as we continue to invest in capacity to "
        b"meet AI demand.\n\n"
        b"Analyst (Keith Weiss, Morgan Stanley): Thanks for taking the "
        b"question. Can you talk about the trajectory of Azure growth into "
        b"the next year and how to think about capex pacing?\n\n"
        b"Satya Nadella: Sure. We remain confident in the Azure growth "
        b"trajectory and we will continue to scale infrastructure to match "
        b"demand. Amy can speak to the capex framing."
    )
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "TRANSCRIPT"},
        files={
            "file": ("msft-call-q4.txt", transcript_body, "text/plain"),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["upload_id"]


async def test_upload_accepts_transcript_filing_type_pdf(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """``filing_type=TRANSCRIPT`` with application/pdf returns 200.

    Re-uses an existing 8-K fixture for body bytes; the bytes themselves
    are an arbitrary readable PDF -- the test exercises the API surface,
    not transcript-specific extraction (the stubbed graph short-circuits
    the analyzer).
    """
    pdf_bytes = (_FIXTURES / "0001193125-26-027198.pdf").read_bytes()
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "TRANSCRIPT"},
        files={
            "file": ("msft-call-q4.pdf", pdf_bytes, "application/pdf"),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"


async def test_upload_rejects_unknown_filing_type(
    seed_watchlist_msft: None,
    app_with_stubbed_graph: AsyncClient,
) -> None:
    """A filing_type outside the allowlist (8-K/10-Q/10-K/TRANSCRIPT) is rejected."""
    response = await app_with_stubbed_graph.post(
        "/api/upload",
        data={"ticker": "MSFT", "filing_type": "FOO"},
        files={"file": ("x.txt", b"hello world", "text/plain")},
    )
    # FastAPI/Pydantic returns 422 for a Literal-form-field allowlist miss.
    assert response.status_code == 422, response.text


class _VisibilityCheckingGraph:
    """Stub graph that probes uploaded_documents from a fresh session.

    The probe runs inside ``ainvoke`` so it executes BEFORE the route
    finishes and the ``get_session`` dependency's tail-commit fires. The
    fresh session opened from the process-wide :func:`get_session_factory`
    represents what every real graph node sees: an independent connection
    that cannot observe writes still pending on the route's session.

    Used by :func:`test_upload_commits_before_graph_invoke` to lock in the
    Phase 4B Task 11c fix -- without the explicit commit in
    :func:`app.api.upload.post_upload`, the probe finds no row and the
    transcript analyzer (and any other node opening its own session) self-
    skips silently.
    """

    def __init__(self) -> None:
        """Record whether the probed upload row was visible from a fresh session."""
        self.upload_visible: bool = False
        self.probed_upload_id: str | None = None

    async def ainvoke(
        self, state: AgentState | dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        """Probe ``uploaded_documents`` from a fresh session, then return a final state."""
        if isinstance(state, AgentState):
            payload: dict[str, Any] = state.model_dump()
        else:
            payload = dict(state)
        filing_event = (
            state.filing_event
            if isinstance(state, AgentState)
            else state.get("filing_event")
        )
        assert filing_event is not None, "test stub requires a FilingEvent on state"
        accession_number = (
            filing_event.accession_number
            if hasattr(filing_event, "accession_number")
            else filing_event["accession_number"]
        )
        upload_id = accession_number.removeprefix("upload-")
        self.probed_upload_id = upload_id

        factory = get_session_factory()
        async with factory() as fresh_session:
            doc = await Repository(fresh_session).get_uploaded_document(upload_id)
            self.upload_visible = doc is not None

        # Keep the response shape valid so the route returns 200.
        payload["financials"] = {"source": "uploaded"}
        payload["comparisons"] = {"consensus_source": "finnhub", "metrics": []}
        payload["language_diffs"] = []
        payload["draft_note"] = "stub draft [F1]"
        payload["final_note"] = "stub draft [F1]"
        payload["critic_verdict"] = "accepted"
        return payload


async def test_upload_commits_before_graph_invoke(
    seed_watchlist_msft: None,
) -> None:
    """The uploaded_documents row must be visible to a separately-opened session
    before ``graph.ainvoke`` runs, so the transcript_analyzer (or any other graph
    node opening its own session via :func:`get_session_factory`) can read it.

    Regression for the Phase 4B Task 11c visibility bug: prior to the explicit
    ``await session.commit()`` between :func:`intake_upload` and
    ``graph.ainvoke``, the route's session held the INSERT until the request
    finished, so a fresh session opened inside the graph saw no row.
    """
    visibility_graph = _VisibilityCheckingGraph()

    def _provide_visibility_graph() -> _VisibilityCheckingGraph:
        """Sync dependency override returning the shared probe graph."""
        return visibility_graph

    app = create_app()
    app.dependency_overrides[get_edgar_client] = _stub_edgar
    app.dependency_overrides[get_compiled_graph] = _provide_visibility_graph
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            pdf_bytes = (_FIXTURES / "0001193125-26-027198.pdf").read_bytes()
            response = await client.post(
                "/api/upload",
                data={"ticker": "MSFT", "filing_type": "8-K"},
                files={"file": ("msft-8k.pdf", pdf_bytes, "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert visibility_graph.probed_upload_id is not None, (
        "stub graph should have observed a FilingEvent on state"
    )
    assert visibility_graph.upload_visible, (
        "uploaded_documents row was NOT visible to a fresh session opened "
        "inside graph.ainvoke -- the route must commit its session BEFORE "
        "invoking the graph so transcript_analyzer and other nodes that open "
        "their own sessions via get_session_factory can read the row."
    )


async def test_chat_returns_501_stub(app_with_stubbed_graph: AsyncClient) -> None:
    """``POST /api/chat`` reserves the route; Phase 6 ships the real agent."""
    response = await app_with_stubbed_graph.post(
        "/api/chat", json={"trace_id": "x", "message": "What was revenue?"}
    )
    assert response.status_code == 501
    detail = response.json()["detail"].lower()
    assert "phase 6" in detail
