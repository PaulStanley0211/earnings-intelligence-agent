"""Unit tests for :mod:`app.llm.prompts`."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.prompts import clear_prompt_cache, load_prompt


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_prompt_cache()


def _write_prompt(directory: Path, name: str, content: str) -> Path:
    path = directory / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_prompt_parses_frontmatter_and_hashes_body(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        "synthesizer/numbers_v1",
        """---
version: v1
model: claude-opus-4-7
temperature: 0.0
---

Hello {name}.
""",
    )

    template = load_prompt("synthesizer/numbers_v1", prompts_dir=tmp_path)

    assert template.name == "synthesizer/numbers_v1"
    assert template.version == "v1"
    assert template.model == "claude-opus-4-7"
    assert template.temperature == 0.0
    assert template.body.strip() == "Hello {name}."
    assert template.prompt_version == "synthesizer/numbers_v1@v1"
    assert len(template.body_sha) == 64
    # Same content → same hash.
    twin = load_prompt("synthesizer/numbers_v1", prompts_dir=tmp_path)
    assert twin.body_sha == template.body_sha


def test_render_substitutes_placeholders(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        "demo",
        """---
version: v1
model: m
temperature: 0.5
---

Hello {name}!
""",
    )
    template = load_prompt("demo", prompts_dir=tmp_path)
    rendered = template.render(name="world")
    assert "Hello world!" in rendered
    assert "{name}" not in rendered


def test_render_raises_on_missing_substitution(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        "demo",
        """---
version: v1
model: m
temperature: 0
---

Hi {name}
""",
    )
    template = load_prompt("demo", prompts_dir=tmp_path)
    with pytest.raises(KeyError):
        template.render()


def test_missing_frontmatter_raises(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "demo", "no frontmatter here\n")
    with pytest.raises(ValueError, match="frontmatter"):
        load_prompt("demo", prompts_dir=tmp_path)


def test_unknown_required_field_raises(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        "demo",
        """---
version: v1
temperature: 0
---

body
""",
    )
    with pytest.raises(ValueError, match="model"):
        load_prompt("demo", prompts_dir=tmp_path)


def test_unparseable_frontmatter_line_raises(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        "demo",
        """---
this is not key value
model: m
version: v1
temperature: 0
---

body
""",
    )
    with pytest.raises(ValueError, match="unparseable"):
        load_prompt("demo", prompts_dir=tmp_path)


def test_real_synthesizer_and_critic_templates_load() -> None:
    synth = load_prompt("synthesizer/numbers_v1")
    critic = load_prompt("critic/numbers_v0")
    assert synth.model == "claude-opus-4-7"
    assert critic.model == "deterministic"
    assert synth.temperature == 0.0
