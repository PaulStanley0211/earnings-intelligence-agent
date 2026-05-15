"""Unit tests for the 10-Q / 10-K section parser."""

from __future__ import annotations

from pathlib import Path

from app.tools.sections import SectionKind, parse_sections

_FIXTURE_DIR = Path("tests/fixtures/edgar_html")


def test_parse_sections_finds_mda_and_risk_factors_in_minimal_10q():
    html = (_FIXTURE_DIR / "synthetic_10q_minimal.html").read_text(encoding="utf-8")
    sections = parse_sections(html, form="10-Q")
    kinds = sorted(s.kind for s in sections)
    assert kinds == [SectionKind.MDA, SectionKind.RISK_FACTORS]


def test_parse_sections_splits_paragraphs():
    html = (_FIXTURE_DIR / "synthetic_10q_minimal.html").read_text(encoding="utf-8")
    sections = parse_sections(html, form="10-Q")
    mda = next(s for s in sections if s.kind == SectionKind.MDA)
    # Three substantive sentences in the fixture.
    assert len(mda.paragraphs) == 3
    assert mda.paragraphs[0].startswith("Our revenue grew")


def test_parse_sections_drops_short_boilerplate():
    html = """<html><body>
    <p>Item 2. Management's Discussion and Analysis</p>
    <p>x</p>
    <p>This is a substantive paragraph well above the 40-character floor.</p>
    <p>Item 3. Other</p>
    </body></html>"""
    sections = parse_sections(html, form="10-Q")
    mda = next(s for s in sections if s.kind == SectionKind.MDA)
    assert len(mda.paragraphs) == 1
    assert mda.paragraphs[0].startswith("This is a substantive")


def test_parse_sections_handles_missing_risk_factors_in_10q():
    """10-Q Item 1A is optional; absence is normal, not degraded."""
    html = """<html><body>
    <p>Item 2. Management's Discussion and Analysis</p>
    <p>Revenue grew driven by enterprise demand for our cloud platform.</p>
    <p>Item 6. Exhibits</p>
    </body></html>"""
    sections = parse_sections(html, form="10-Q")
    kinds = [s.kind for s in sections]
    assert SectionKind.MDA in kinds
    assert SectionKind.RISK_FACTORS not in kinds


def test_parse_sections_10k_uses_item_7_for_mda():
    html = """<html><body>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Annual revenue grew supported by sustained enterprise adoption.</p>
    <p>Item 8. Financial Statements</p>
    </body></html>"""
    sections = parse_sections(html, form="10-K")
    mda = next(s for s in sections if s.kind == SectionKind.MDA)
    assert len(mda.paragraphs) == 1


def test_parse_sections_collapses_tables_to_sentinel_drops_under_filter():
    html = """<html><body>
    <p>Item 2. Management's Discussion and Analysis</p>
    <p>Revenue grew driven by enterprise demand for our cloud platform.</p>
    <table><tr><td>x</td><td>1</td></tr></table>
    <p>Item 3. Other</p>
    </body></html>"""
    sections = parse_sections(html, form="10-Q")
    mda = next(s for s in sections if s.kind == SectionKind.MDA)
    # The table sentinel is below the 40-char floor and is dropped.
    assert len(mda.paragraphs) == 1
