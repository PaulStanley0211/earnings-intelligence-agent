"""The note_writer terminal node.

Persists the accepted synthesized note into the ``notes`` table. Runs only
after the critic returns ACCEPTED. On LOOP_EXCEEDED the node yields an
empty StateUpdate so the graph proceeds to END with no row written -- the
note is held for manual review per the runbook.

A DB failure here does not propagate: the user already has the note in
their API response, so we degrade gracefully and set ``persisted_note_id``
to ``None`` for trace visibility.
"""

from __future__ import annotations

from app.memory.repository import Repository
from app.memory.schemas import NoteCreate
from app.models.state import AgentState, CriticVerdict, StateUpdate
from app.observability.logging import current_trace_id, get_logger

_logger = get_logger()

OWNER = "note_writer"


async def write_note(
    state: AgentState,
    *,
    repository: Repository,
    prompt_template_name: str,
    prompt_template_sha: str,
) -> StateUpdate:
    """Persist the final note when the critic accepted it, otherwise no-op."""
    if state.critic_verdict is not CriticVerdict.ACCEPTED:
        _logger.bind(
            accession=state.filing_event.accession_number,
            verdict=state.critic_verdict.value if state.critic_verdict else None,
            trace_id=current_trace_id(),
        ).info("note_writer_skipped")
        return StateUpdate(owner=OWNER, changes={})

    if state.final_note is None:
        _logger.bind(
            accession=state.filing_event.accession_number,
            trace_id=current_trace_id(),
        ).warning("note_writer_no_final_note")
        return StateUpdate(owner=OWNER, changes={})

    payload = NoteCreate(
        filing_accession=state.filing_event.accession_number,
        ticker=state.filing_event.ticker,
        markdown_body=state.final_note,
        prompt_template_name=prompt_template_name,
        prompt_template_sha=prompt_template_sha,
        critic_attempts=state.critic_attempts,
    )

    try:
        note_id: int | None = await repository.insert_note(payload)
    except Exception as exc:  # degrade gracefully - DB failure must not block the API response
        _logger.bind(
            accession=state.filing_event.accession_number,
            error=str(exc),
            trace_id=current_trace_id(),
        ).error("note_writer_persist_failed")
        note_id = None

    _logger.bind(
        accession=state.filing_event.accession_number,
        note_id=note_id,
        trace_id=current_trace_id(),
    ).info("note_writer_complete")
    return StateUpdate(owner=OWNER, changes={"persisted_note_id": note_id})
