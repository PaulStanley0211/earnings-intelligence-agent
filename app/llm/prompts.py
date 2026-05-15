"""Versioned prompt templates.

Prompts are code. Every prompt the system ships lives under
:file:`prompts/<name>.md` with a YAML-ish frontmatter block declaring the
default model, temperature, and a human-readable version string. The loader
hashes the body to produce a stable ``body_sha`` so the LLM client can tag
every call with which exact prompt content handled it; a typo in the prompt
file flips the hash and the next eval traces back to the bad commit.

The frontmatter parser is intentionally narrow: only the keys we know about
are accepted, and the file format is the documented contract. Adding new
metadata is one schema change here and one frontmatter key in the template.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

_FRONTMATTER_FENCE: Final[re.Pattern[str]] = re.compile(
    r"\A---\n(?P<front>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL
)
_FRONTMATTER_LINE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$"
)

# Resolved at import time so the loader works regardless of the process CWD.
_DEFAULT_PROMPTS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    """A parsed prompt template.

    Attributes mirror the on-disk frontmatter plus a derived ``body_sha`` so
    callers can pass ``f"{name}@{body_sha[:8]}"`` to the LLM client and have
    cassette keys move with prompt content.
    """

    name: str
    version: str
    model: str
    temperature: float
    body: str
    body_sha: str

    @property
    def prompt_version(self) -> str:
        """Return the cassette-friendly ``name@version`` identifier."""
        return f"{self.name}@{self.version}"

    def render(self, **substitutions: str) -> str:
        """Substitute ``{key}``-style placeholders in :attr:`body`.

        Raises ``KeyError`` for missing substitutions to surface drift
        between the template and the calling agent. Unknown substitution
        keys are silently ignored so harmless extra context does not break
        the call site.
        """
        try:
            return self.body.format(**substitutions)
        except KeyError as exc:
            raise KeyError(
                f"prompt {self.name!r} expects substitution {exc.args[0]!r} "
                "but the caller did not provide it"
            ) from exc


def load_prompt(name: str, *, prompts_dir: Path | None = None) -> PromptTemplate:
    """Load and parse the template at ``prompts/<name>.md``.

    ``name`` is a slash-separated path under :file:`prompts/` without the
    ``.md`` suffix (e.g., ``"synthesizer/numbers_v1"``). The result is
    cached so repeated calls during one process are cheap.
    """
    base = prompts_dir or _DEFAULT_PROMPTS_DIR
    return _load_prompt_cached(name, str(base.resolve()))


@lru_cache(maxsize=64)
def _load_prompt_cached(name: str, base: str) -> PromptTemplate:
    """Disk-bound implementation of :func:`load_prompt`, cached by key."""
    path = Path(base) / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_FENCE.match(raw)
    if match is None:
        raise ValueError(
            f"prompt {path} is missing frontmatter; "
            "wrap declarations in '---' fences at the top of the file."
        )
    metadata = _parse_frontmatter(match.group("front"))
    body = match.group("body").rstrip() + "\n"
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return PromptTemplate(
        name=name,
        version=_require(metadata, "version", path),
        model=_require(metadata, "model", path),
        temperature=float(_require(metadata, "temperature", path)),
        body=body,
        body_sha=body_sha,
    )


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Parse the ``key: value`` lines inside a prompt's frontmatter fence."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _FRONTMATTER_LINE.match(line)
        if match is None:
            raise ValueError(
                f"unparseable frontmatter line: {line!r}; "
                "lines must be of the form 'key: value'."
            )
        out[match.group(1)] = match.group(2)
    return out


def _require(metadata: dict[str, str], key: str, path: Path) -> str:
    """Return ``metadata[key]`` or raise a context-rich error."""
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"prompt {path} is missing required frontmatter key: {key}")
    return value


def clear_prompt_cache() -> None:
    """Drop the in-process prompt cache.

    Tests that mutate templates on disk between cases call this so the next
    :func:`load_prompt` re-reads the file. Production code never needs it.
    """
    _load_prompt_cached.cache_clear()
