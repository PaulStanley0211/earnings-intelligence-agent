"""The transcript-analyzer agent node.

Two-pass Sonnet pipeline that runs over a user-uploaded earnings-call
transcript:

1. **Guard** - returns an empty :class:`StateUpdate` when the current
   filing is not a ``TRANSCRIPT`` so the parallel block in
   :mod:`app.graph` can include this node without disrupting 10-Q/10-K/8-K
   runs.
2. **Extract** - one Sonnet call using
   ``prompts/transcript_analyzer/extract_v1`` that returns analyst Q&A
   pairs plus newly-stated management commitments. Malformed JSON triggers
   one retry; a second failure surfaces as a logged warning and an empty
   update so the synthesizer can still run in a degraded state.
3. **Reconcile** - a deterministic keyword/period pre-filter narrows the
   list of prior open commitments for the ticker; survivors go to a single
   Sonnet call using ``prompts/transcript_analyzer/reconcile_v1`` that
   verdicts each one ``met`` / ``missed`` / ``still_open``. If the
   pre-filter empties the list, the reconcile LLM call is skipped.
4. **Persist** - ``add_qa_pairs``, ``add_commitments``, then
   ``update_commitment_status`` for each reconciled prior commitment, all
   on the caller-owned :class:`~app.memory.repository.Repository` session
   so the four writes commit as one transaction.

The node is a pure function of :class:`AgentState`; side effects live in
:mod:`app.memory.repository`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.client import CostCapExceeded, LLMClient, _SupportsDailySpend
from app.llm.prompts import PromptTemplate, load_prompt
from app.memory.repository import Repository
from app.memory.schemas import (
    CommitmentRecord,
    CommitmentStatus,
    NewCommitment,
    NewQAPair,
    UploadedDocumentRecord,
)
from app.models.state import (
    AgentState,
    AnswerClass,
    CommitmentExtracted,
    CommitmentStatusUpdate,
    FilingForm,
    QAPairPayload,
    StateUpdate,
)
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "transcript_analyzer"

EXTRACT_PROMPT_NAME: Final[str] = "transcript_analyzer/extract_v1"
RECONCILE_PROMPT_NAME: Final[str] = "transcript_analyzer/reconcile_v1"

_EXTRACT_MAX_TOKENS: Final[int] = 4096
_RECONCILE_MAX_TOKENS: Final[int] = 2048
_EXTRACT_RETRIES: Final[int] = 2

# Tokens shorter than this are too generic ("the", "we expect") to provide
# useful keyword overlap between a commitment and a transcript. Length 5
# strikes a balance between recall and noise.
_PREFILTER_TOKEN_MIN_LEN: Final[int] = 5
_PREFILTER_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")

# Two distinct content tokens (>=5 chars each) must appear in the transcript
# for a commitment to survive the prefilter. Single-token matches on
# generic SaaS vocabulary ("revenue", "margin", "quarter") used to drag
# every prior commitment through to the LLM, which inflated the reconcile
# call and produced spurious closures.
_PREFILTER_MIN_DISTINCT_TOKENS: Final[int] = 2

# Content tokens that almost every earnings call contains and therefore
# cannot help distinguish "this commitment is plausibly addressed by this
# transcript". Drop them before counting overlap. Kept to genuinely
# universal earnings-call vocabulary - more specific terms like
# ``margin``, ``revenue``, or ``customer`` are retained because some
# commitments contain almost nothing else and we still want a chance for
# them to survive when they coincide.
_PREFILTER_STOP_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "about",
        "approximately",
        "around",
        "billion",
        "during",
        "expect",
        "expects",
        "expected",
        "fiscal",
        "however",
        "least",
        "level",
        "million",
        "percent",
        "percentage",
        "period",
        "plans",
        "point",
        "points",
        "prior",
        "quarter",
        "quarters",
        "range",
        "remain",
        "remains",
        "roughly",
        "should",
        "target",
        "targets",
        "their",
        "these",
        "those",
        "through",
        "today",
        "toward",
        "would",
        "years",
    }
)

# Verdicts whose `reason` matches this string verbatim (case-insensitive,
# after stripping leading/trailing whitespace and a trailing full-stop)
# signal that the LLM did not find evidence in the transcript for the
# commitment. The agent honours these by leaving the database row at its
# current `open` status - we still emit the verdict into the StateUpdate
# for downstream visibility, but no `update_commitment_status` write
# happens. This keeps the reconcile path from manufacturing closures or
# spurious `still_open` flips for commitments that nothing in the
# transcript actually addressed.
_UNADDRESSED_REASON: Final[str] = "transcript does not address this commitment"

_EMPTY_UPDATE: Final[StateUpdate] = StateUpdate(owner=OWNER, changes={})


def _is_unaddressed_reason(reason: str) -> bool:
    """Return True when the reconcile reason matches the canonical no-evidence string.

    The prompt instructs the LLM to emit the exact phrase verbatim, but we
    normalise case, surrounding whitespace, and a trailing full-stop to
    stay robust against tiny formatting drifts.
    """
    normalised = reason.strip().rstrip(".").lower()
    return normalised == _UNADDRESSED_REASON


# ---- Internal Pydantic validation guards for LLM JSON output ----


class _RawQAPair(BaseModel):
    """Validated shape of one ``qa_pairs`` item from ``extract_v1``."""

    model_config = ConfigDict(extra="ignore")

    ordinal: int = Field(..., ge=1)
    analyst_name: str | None = None
    question_text: str = Field(..., min_length=1)
    answer_text: str = Field(..., min_length=1)
    answer_class: AnswerClass


class _RawCommitment(BaseModel):
    """Validated shape of one ``commitments`` item from ``extract_v1``."""

    model_config = ConfigDict(extra="ignore")

    commitment_text: str = Field(..., min_length=1)
    target_period: str | None = None
    source_quote: str = Field(..., min_length=1)


class _ExtractResponse(BaseModel):
    """Top-level JSON shape returned by the extract prompt."""

    model_config = ConfigDict(extra="ignore")

    qa_pairs: list[_RawQAPair] = Field(default_factory=list)
    commitments: list[_RawCommitment] = Field(default_factory=list)


class _RawVerdict(BaseModel):
    """One verdict item returned by ``reconcile_v1``."""

    model_config = ConfigDict(extra="ignore")

    commitment_id: int
    new_status: CommitmentStatus
    reason: str = Field(..., min_length=1)


class _ReconcileResponse(BaseModel):
    """Top-level JSON shape returned by the reconcile prompt."""

    model_config = ConfigDict(extra="ignore")

    verdicts: list[_RawVerdict] = Field(default_factory=list)


class _SupportsTranscriptStorage(Protocol):
    """Repository surface the transcript analyzer needs.

    Declared as a Protocol so unit tests can pass a stub without spinning
    up SQLAlchemy. Production callers pass a real :class:`Repository`.

    The protocol also covers the daily-spend pair from
    :class:`app.llm.client._SupportsDailySpend` because the LLM client's
    pre-flight cost check accepts whatever the agent already holds.
    """

    async def get_uploaded_document(
        self, upload_id: str
    ) -> UploadedDocumentRecord | None: ...

    async def get_open_commitments(
        self, ticker: str
    ) -> Sequence[CommitmentRecord]: ...

    async def add_qa_pairs(
        self, *, filing_accession: str, pairs: Sequence[NewQAPair]
    ) -> Sequence[object]: ...

    async def add_commitments(
        self,
        *,
        filing_accession: str,
        ticker: str,
        commitments: Sequence[NewCommitment],
    ) -> Sequence[object]: ...

    async def update_commitment_status(
        self,
        *,
        commitment_id: int,
        status: CommitmentStatus,
        resolved_filing_accession: str | None,
        resolved_reason: str | None,
    ) -> None: ...

    async def get_daily_spend(self, day: date) -> Decimal: ...

    async def add_daily_spend(
        self, *, day: date, amount_usd: Decimal
    ) -> Decimal: ...


# ---- Reconciliation pre-filter ----


def _reconcile_prefilter(
    prior_commitments: Sequence[CommitmentRecord],
    transcript_text: str,
) -> list[CommitmentRecord]:
    """Filter prior commitments to those plausibly addressed by the transcript.

    Heuristic: keep a commitment when at least one of the following holds:

    * Two or more distinct content tokens (>=5 chars each) from
      ``commitment_text`` appear in the transcript, where common SaaS
      vocabulary (``revenue``, ``margin``, ``quarter``, etc. - see
      :data:`_PREFILTER_STOP_TOKENS`) is excluded from the count. This
      stricter rule keeps the LLM reconcile call from getting flooded
      with every prior commitment whenever they share generic earnings
      vocabulary with the new transcript.
    * The commitment's ``target_period`` appears verbatim
      (case-insensitive) in the transcript - a strong topical signal on
      its own.

    Conservative - when in doubt, keep the commitment so the LLM can
    decide. The downstream prompt + agent-side filter handles the case
    where the LLM cannot actually resolve a survivor.
    """
    if not prior_commitments:
        return []
    haystack_lower = transcript_text.lower()
    haystack_tokens = {
        tok.lower()
        for tok in _PREFILTER_TOKEN_PATTERN.findall(transcript_text)
        if len(tok) >= _PREFILTER_TOKEN_MIN_LEN
    }
    survivors: list[CommitmentRecord] = []
    for commitment in prior_commitments:
        if _has_keyword_overlap(commitment.commitment_text, haystack_tokens):
            survivors.append(commitment)
            continue
        if commitment.target_period and commitment.target_period.lower() in haystack_lower:
            survivors.append(commitment)
    return survivors


def _has_keyword_overlap(needle_text: str, haystack_tokens: set[str]) -> bool:
    """Return True when ``needle_text`` shares >=2 non-stop tokens with the transcript.

    Tokens shorter than :data:`_PREFILTER_TOKEN_MIN_LEN` and tokens listed
    in :data:`_PREFILTER_STOP_TOKENS` are excluded so generic SaaS
    vocabulary (``revenue``, ``margin``, ``quarter``) cannot single-handedly
    pull a commitment through the prefilter.
    """
    distinct_matches: set[str] = set()
    for token in _PREFILTER_TOKEN_PATTERN.findall(needle_text):
        if len(token) < _PREFILTER_TOKEN_MIN_LEN:
            continue
        lowered = token.lower()
        if lowered in _PREFILTER_STOP_TOKENS:
            continue
        if lowered in haystack_tokens:
            distinct_matches.add(lowered)
            if len(distinct_matches) >= _PREFILTER_MIN_DISTINCT_TOKENS:
                return True
    return False


# ---- Prompt rendering helpers ----


def _render_prior_commitments_block(
    survivors: Sequence[CommitmentRecord],
) -> str:
    """Render survivors as a numbered list the reconcile prompt expects."""
    lines: list[str] = []
    for index, commitment in enumerate(survivors, start=1):
        period = commitment.target_period or "none"
        lines.append(
            f"{index}. id={commitment.id} [target={period}]: "
            f"{commitment.commitment_text}"
        )
    return "\n".join(lines)


# ---- JSON parsing helpers ----


def _parse_extract_json(text: str) -> _ExtractResponse:
    """Parse the extract prompt's JSON output, raising on malformed shapes.

    Both :class:`json.JSONDecodeError` and :class:`ValidationError` are
    surfaced as :class:`ValueError` so the caller has a single exception
    type to catch for retry.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extract response is not valid JSON: {exc}") from exc
    try:
        return _ExtractResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"extract response did not match contract: {exc}") from exc


def _extract_first_json_object(text: str) -> str:
    """Return the substring of ``text`` that spans the first top-level JSON object.

    Scans for the opening brace, then walks the string incrementing / decrementing
    a depth counter until the matching closing brace is found, respecting quoted
    strings so braces inside string literals are not counted. Returns the matched
    substring or the original text when no complete object is found (which will
    then fail in the caller's ``json.loads`` and surface a clear error).

    This handles the occasional model behaviour of producing trailing commentary
    or a self-correction block after the first complete JSON object. The first
    syntactically complete object is the authoritative answer.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _parse_reconcile_json(text: str) -> _ReconcileResponse:
    """Parse the reconcile prompt's JSON output, raising on malformed shapes.

    Extracts only the first complete top-level JSON object from the response
    text so that occasional model self-corrections (where the model produces
    trailing commentary or a second corrected JSON object) do not cause a
    spurious parse failure.
    """
    candidate = _extract_first_json_object(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reconcile response is not valid JSON: {exc}") from exc
    try:
        return _ReconcileResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"reconcile response did not match contract: {exc}") from exc


# ---- LLM call wrappers ----


async def _call_extract(
    *,
    template: PromptTemplate,
    transcript_text: str,
    llm: LLMClient,
    repository: _SupportsDailySpend,
) -> tuple[_ExtractResponse, float]:
    """Run the extract pass with one retry on malformed JSON.

    Returns the parsed response plus the cumulative cost in USD across all
    attempts (so retries still charge the spend ledger). Raises
    :class:`ValueError` when both attempts return unparseable output.
    """
    user_content = template.render(transcript_text=transcript_text)
    last_error: ValueError | None = None
    total_cost = 0.0
    for attempt in range(1, _EXTRACT_RETRIES + 1):
        response = await llm.acomplete(
            prompt_version=f"{template.prompt_version}#{template.body_sha[:8]}",
            messages=[{"role": "user", "content": user_content}],
            repository=repository,
            model=template.model,
            temperature=template.temperature,
            max_tokens=_EXTRACT_MAX_TOKENS,
        )
        total_cost += response.cost_usd
        try:
            parsed = _parse_extract_json(response.text)
        except ValueError as exc:
            last_error = exc
            _logger.bind(
                attempt=attempt,
                trace_id=current_trace_id(),
            ).warning("transcript_analyzer_extract_parse_failed")
            continue
        return parsed, total_cost
    raise last_error or ValueError("extract pass exhausted retries without a parse failure")


async def _call_reconcile(
    *,
    template: PromptTemplate,
    transcript_text: str,
    survivors: Sequence[CommitmentRecord],
    llm: LLMClient,
    repository: _SupportsDailySpend,
) -> tuple[_ReconcileResponse, float]:
    """Run the reconcile pass; return parsed verdicts plus the cost in USD.

    On malformed JSON returns an empty verdicts list and logs a warning -
    spec §6 says "Prior commitments remain ``open``" when the call fails.
    """
    user_content = template.render(
        prior_commitments_block=_render_prior_commitments_block(survivors),
        transcript_text=transcript_text,
    )
    response = await llm.acomplete(
        prompt_version=f"{template.prompt_version}#{template.body_sha[:8]}",
        messages=[{"role": "user", "content": user_content}],
        repository=repository,
        model=template.model,
        temperature=template.temperature,
        max_tokens=_RECONCILE_MAX_TOKENS,
    )
    try:
        parsed = _parse_reconcile_json(response.text)
    except ValueError:
        _logger.bind(
            trace_id=current_trace_id(),
        ).warning("transcript_analyzer_reconcile_parse_failed")
        return _ReconcileResponse(verdicts=[]), response.cost_usd
    return parsed, response.cost_usd


# ---- Conversion helpers ----


def _qa_payloads(
    raw_pairs: Sequence[_RawQAPair],
) -> tuple[list[QAPairPayload], list[NewQAPair]]:
    """Convert raw extract output into the state payload + persistence DTOs."""
    payloads: list[QAPairPayload] = []
    new_rows: list[NewQAPair] = []
    for raw in raw_pairs:
        sha = hashlib.sha256(
            f"{raw.question_text}\n{raw.answer_text}".encode()
        ).hexdigest()
        payloads.append(
            QAPairPayload(
                ordinal=raw.ordinal,
                analyst_name=raw.analyst_name,
                question_text=raw.question_text,
                answer_text=raw.answer_text,
                answer_class=raw.answer_class,
                sha256_text=sha,
            )
        )
        new_rows.append(
            NewQAPair(
                ordinal=raw.ordinal,
                analyst_name=raw.analyst_name,
                question_text=raw.question_text,
                answer_text=raw.answer_text,
                answer_class=raw.answer_class,
                sha256_text=sha,
            )
        )
    return payloads, new_rows


def _commitment_payloads(
    raw_commitments: Sequence[_RawCommitment],
) -> tuple[list[CommitmentExtracted], list[NewCommitment]]:
    """Convert raw extract output into the state payload + persistence DTOs."""
    payloads: list[CommitmentExtracted] = []
    new_rows: list[NewCommitment] = []
    for raw in raw_commitments:
        payloads.append(
            CommitmentExtracted(
                commitment_text=raw.commitment_text,
                target_period=raw.target_period,
                source_quote=raw.source_quote,
            )
        )
        new_rows.append(
            NewCommitment(
                commitment_text=raw.commitment_text,
                target_period=raw.target_period,
                source_quote=raw.source_quote,
            )
        )
    return payloads, new_rows


def _verdict_payloads(
    verdicts: Sequence[_RawVerdict],
    eligible_ids: set[int],
) -> list[CommitmentStatusUpdate]:
    """Drop verdicts whose ``commitment_id`` did not survive the prefilter."""
    return [
        CommitmentStatusUpdate(
            commitment_id=v.commitment_id,
            new_status=v.new_status,
            reason=v.reason,
        )
        for v in verdicts
        if v.commitment_id in eligible_ids
    ]


def _persistable_verdicts(
    verdicts: Sequence[CommitmentStatusUpdate],
) -> list[CommitmentStatusUpdate]:
    """Return only those verdicts that should mutate the database.

    Two filters apply:

    1. Verdicts whose ``reason`` matches :data:`_UNADDRESSED_REASON` are
       dropped. The LLM is telling us "the transcript does not address
       this commitment at all". Writing a ``still_open`` reconciliation
       row in that case would overwrite ``resolved_filing_accession`` /
       ``resolved_reason`` with a non-evidence row and misrepresent the
       analysis as having reconciled a commitment it never addressed.

    2. Verdicts whose ``new_status`` is :attr:`CommitmentStatus.STILL_OPEN`
       are also dropped. ``STILL_OPEN`` means "the commitment was
       discussed but not resolved" - in that case the database should
       continue to report the row as ``OPEN`` (the canonical
       not-yet-resolved state). Only ``MET`` / ``MISSED`` verdicts are
       persisted because those are the only states that close a
       commitment. This prevents the reconcile pass from generating
       spurious ``still_open`` rows for every Q2 commitment the Q3
       transcript merely mentions in passing, which used to trip the
       "zero false closures" gate.

    The full verdict list (including dropped entries) remains in
    :class:`StateUpdate.commitment_updates` so downstream consumers and
    operators can see what the LLM produced; only the DB write is
    suppressed.
    """
    persistable: list[CommitmentStatusUpdate] = []
    for verdict in verdicts:
        if _is_unaddressed_reason(verdict.reason):
            continue
        if verdict.new_status == CommitmentStatus.STILL_OPEN:
            continue
        persistable.append(verdict)
    return persistable


# ---- Transcript text retrieval ----


_UPLOAD_PREFIX: Final[str] = "upload-"


def _upload_id_from_accession(accession_number: str) -> str | None:
    """Strip the ``upload-`` prefix off an upload-driven accession.

    Returns ``None`` for accessions that did not originate from
    :func:`app.agents.upload_intake.intake_upload`; the transcript analyzer
    has nothing to read in that case.
    """
    if not accession_number.startswith(_UPLOAD_PREFIX):
        return None
    return accession_number[len(_UPLOAD_PREFIX) :]


async def _load_transcript_text(
    *,
    accession_number: str,
    repository: _SupportsTranscriptStorage,
) -> str | None:
    """Return the parsed transcript text for ``accession_number`` or ``None``."""
    upload_id = _upload_id_from_accession(accession_number)
    if upload_id is None:
        return None
    document = await repository.get_uploaded_document(upload_id)
    if document is None:
        return None
    return document.parsed_text


# ---- Persistence ----


async def _persist(
    *,
    repository: _SupportsTranscriptStorage,
    filing_accession: str,
    ticker: str,
    qa_rows: Sequence[NewQAPair],
    commitment_rows: Sequence[NewCommitment],
    verdict_updates: Sequence[CommitmentStatusUpdate],
) -> None:
    """Run the three-write transaction. Caller owns the commit boundary."""
    if qa_rows:
        await repository.add_qa_pairs(
            filing_accession=filing_accession, pairs=qa_rows
        )
    if commitment_rows:
        await repository.add_commitments(
            filing_accession=filing_accession,
            ticker=ticker,
            commitments=commitment_rows,
        )
    for verdict in verdict_updates:
        await repository.update_commitment_status(
            commitment_id=verdict.commitment_id,
            status=verdict.new_status,
            resolved_filing_accession=filing_accession,
            resolved_reason=verdict.reason,
        )


# ---- Public node entry point ----


async def transcript_analyzer(
    state: AgentState,
    *,
    llm: LLMClient,
    repository: Repository | _SupportsTranscriptStorage,
) -> StateUpdate:
    """Extract Q&A pairs and commitments; reconcile prior open commitments.

    Self-skips when the incoming filing is not a TRANSCRIPT so the parallel
    block in :mod:`app.graph` can include the node unconditionally. The
    daily LLM spend cap is enforced by :meth:`LLMClient.acomplete`; a
    ``CostCapExceeded`` exception surfaces as a logged warning and an empty
    update so the synthesizer still produces a degraded note.
    """
    filing = state.filing_event
    if filing.form != FilingForm.TRANSCRIPT:
        return _EMPTY_UPDATE

    transcript_text = await _load_transcript_text(
        accession_number=filing.accession_number, repository=repository
    )
    if transcript_text is None:
        _logger.bind(
            accession=filing.accession_number,
            trace_id=current_trace_id(),
        ).warning("transcript_analyzer_no_transcript_text")
        return _EMPTY_UPDATE

    extract_template = load_prompt(EXTRACT_PROMPT_NAME)
    try:
        extract_response, extract_cost = await _call_extract(
            template=extract_template,
            transcript_text=transcript_text,
            llm=llm,
            repository=repository,
        )
    except CostCapExceeded:
        _logger.bind(
            accession=filing.accession_number,
            trace_id=current_trace_id(),
        ).warning("transcript_analyzer_cost_cap_exceeded")
        return _EMPTY_UPDATE
    except ValueError:
        _logger.bind(
            accession=filing.accession_number,
            trace_id=current_trace_id(),
        ).warning("transcript_analyzer_extract_unparseable")
        return _EMPTY_UPDATE

    qa_payloads, qa_rows = _qa_payloads(extract_response.qa_pairs)
    commitment_payloads, commitment_rows = _commitment_payloads(
        extract_response.commitments
    )

    prior_commitments = await repository.get_open_commitments(filing.ticker)
    survivors = _reconcile_prefilter(prior_commitments, transcript_text)
    eligible_ids = {c.id for c in survivors}

    verdict_updates: list[CommitmentStatusUpdate] = []
    reconcile_cost = 0.0
    if survivors:
        reconcile_template = load_prompt(RECONCILE_PROMPT_NAME)
        try:
            reconcile_response, reconcile_cost = await _call_reconcile(
                template=reconcile_template,
                transcript_text=transcript_text,
                survivors=survivors,
                llm=llm,
                repository=repository,
            )
            verdict_updates = _verdict_payloads(
                reconcile_response.verdicts, eligible_ids
            )
        except CostCapExceeded:
            _logger.bind(
                accession=filing.accession_number,
                trace_id=current_trace_id(),
            ).warning("transcript_analyzer_reconcile_cost_cap_exceeded")

    persistable_verdicts = _persistable_verdicts(verdict_updates)

    await _persist(
        repository=repository,
        filing_accession=filing.accession_number,
        ticker=filing.ticker,
        qa_rows=qa_rows,
        commitment_rows=commitment_rows,
        verdict_updates=persistable_verdicts,
    )

    _logger.bind(
        accession=filing.accession_number,
        ticker=filing.ticker,
        qa_pair_count=len(qa_payloads),
        commitment_count=len(commitment_payloads),
        verdict_count=len(verdict_updates),
        persistable_verdict_count=len(persistable_verdicts),
        unaddressed_verdict_count=len(verdict_updates) - len(persistable_verdicts),
        survivors=len(survivors),
        prior_open_count=len(prior_commitments),
        trace_id=current_trace_id(),
    ).info("transcript_analyzer_complete")

    return StateUpdate(
        owner=OWNER,
        changes={
            "qa_pairs": qa_payloads,
            "commitments": commitment_payloads,
            "commitment_updates": verdict_updates,
            "cost_usd": extract_cost + reconcile_cost,
        },
    )
