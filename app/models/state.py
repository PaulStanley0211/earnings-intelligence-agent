"""The :class:`AgentState` contract carried through the LangGraph nodes.

Every graph node is a pure function of an ``AgentState`` and returns a
:class:`StateUpdate` describing only the fields the node owns. Mutating a field
that the node does not own raises a validation error - this prevents nodes
from stepping on each other's outputs by accident.

Field-level detail (Financials, LanguageDiff, QAPair, ...) is filled in by the
phases that produce them. For Phase 0 the placeholders are typed as generic
mappings; their richer schemas land in Phase 1+ next to the agents that emit
them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FilingForm(StrEnum):
    """SEC filing forms the system understands."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"


class FilingEvent(BaseModel):
    """The triggering filing for an agent run.

    Populated by the EDGAR watcher and never mutated by downstream nodes.
    """

    model_config = ConfigDict(frozen=True)

    accession_number: str = Field(..., description="SEC accession number, dashes-as-dashed.")
    cik: str = Field(..., description="SEC central index key, zero-padded.")
    ticker: str = Field(..., description="Primary trading symbol.")
    form: FilingForm
    filed_at: datetime
    source_url: str = Field(..., description="EDGAR URL of the filing index.")


class CriticVerdict(StrEnum):
    """Outcome of a critic pass over a draft note."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    LOOP_EXCEEDED = "loop_exceeded"


class CriticFinding(BaseModel):
    """A single issue identified by the critic.

    Captured as part of :attr:`AgentState.critic_findings` so the synthesiser
    can address them on retry and so the audit log records why a note was held.
    """

    layer: str = Field(..., description="Critic layer that fired: numbers, quote, llm.")
    severity: str = Field(..., description="info | warning | error.")
    message: str
    citation_id: int | None = None


class AgentState(BaseModel):
    """The single object passed between LangGraph nodes during one run.

    Every field has a designated owning node. :class:`StateUpdate` enforces
    that ownership at runtime so we can debug who clobbered what.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- Run metadata ----
    trace_id: str
    started_at: datetime
    cost_usd: float = 0.0

    # ---- Inputs ----
    filing_event: FilingEvent

    # ---- Planner output ----
    plan: list[str] = Field(
        default_factory=list,
        description="Specialist node names the planner has selected to invoke.",
    )

    # ---- Specialist outputs. Detailed Pydantic models land in their phases. ----
    financials: dict[str, Any] | None = None
    comparisons: dict[str, Any] | None = None
    language_diffs: list[dict[str, Any]] = Field(default_factory=list)
    qa_pairs: list[dict[str, Any]] = Field(default_factory=list)
    peer_context: dict[str, Any] | None = None

    # ---- Synthesizer + critic loop ----
    draft_note: str | None = None
    critic_findings: list[CriticFinding] = Field(default_factory=list)
    critic_verdict: CriticVerdict | None = None
    critic_attempts: int = 0
    final_note: str | None = None


# Lookup of which node is allowed to mutate which AgentState field.
# Mutating any other field via :class:`StateUpdate` is a validation error.
_FIELD_OWNERS: dict[str, frozenset[str]] = {
    "planner": frozenset({"plan", "cost_usd"}),
    "financial_extractor": frozenset({"financials", "cost_usd"}),
    "comparator": frozenset({"comparisons", "cost_usd"}),
    "language_differ": frozenset({"language_diffs", "cost_usd"}),
    "transcript_analyzer": frozenset({"qa_pairs", "cost_usd"}),
    "answer_classifier": frozenset({"qa_pairs", "cost_usd"}),
    "commitment_extractor": frozenset({"qa_pairs", "cost_usd"}),
    "commitment_resolver": frozenset({"qa_pairs", "cost_usd"}),
    "peer_reader": frozenset({"peer_context", "cost_usd"}),
    "synthesizer": frozenset({"draft_note", "cost_usd"}),
    "critic": frozenset(
        {"critic_findings", "critic_verdict", "critic_attempts", "final_note", "cost_usd"}
    ),
}


class StateUpdate(BaseModel):
    """A typed patch returned by a LangGraph node.

    The :attr:`owner` field names the node producing the update and is checked
    against :data:`_FIELD_OWNERS`. Any attempt to set a field outside the
    owner's allowlist fails fast. ``cost_usd`` may be added by any node and is
    accumulated, not overwritten.
    """

    model_config = ConfigDict(extra="forbid")

    # Sentinel: keys allowed to appear in ``changes``. Mirrors AgentState.
    _allowed_keys: ClassVar[frozenset[str]] = frozenset(
        set(AgentState.model_fields.keys()) - {"trace_id", "started_at", "filing_event"}
    )

    owner: str = Field(..., description="Node name producing this update.")
    changes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ownership(self) -> StateUpdate:
        """Reject changes the named owner is not allowed to write."""
        allowed = _FIELD_OWNERS.get(self.owner)
        if allowed is None:
            raise ValueError(f"Unknown StateUpdate owner: {self.owner!r}.")
        offending = set(self.changes) - allowed
        if offending:
            raise ValueError(
                f"Node {self.owner!r} cannot mutate fields {sorted(offending)}; "
                f"owned fields are {sorted(allowed)}."
            )
        unknown = set(self.changes) - self._allowed_keys
        if unknown:
            raise ValueError(f"StateUpdate has unknown AgentState fields: {sorted(unknown)}.")
        return self

    def apply(self, state: AgentState) -> AgentState:
        """Return a new :class:`AgentState` with the update applied.

        ``cost_usd`` is summed; other fields are overwritten. The input state is
        not mutated.
        """
        merged = state.model_dump()
        for key, value in self.changes.items():
            if key == "cost_usd":
                merged["cost_usd"] = float(merged.get("cost_usd", 0.0)) + float(value)
            else:
                merged[key] = value
        return AgentState.model_validate(merged)
