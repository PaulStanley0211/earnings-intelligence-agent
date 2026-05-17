"""Sanity checks on the advisor accuracy cassettes.

These tests guard the cassette set that backs the Phase 4B advisor accuracy
gate (Task 11b). They do not exercise the advisor itself - they just verify
that the fixture files exist, are consistent, and that every expected 8-K
accession recorded in ``_test_cases.json`` actually appears in its companion
cassette at a row whose form is ``8-K`` and whose filing date is on or before
the case's ``as_of_date``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

_CASSETTES: Path = (
    Path(__file__).resolve().parents[1] / "fixtures" / "edgar" / "advisor"
)


def _load_index() -> list[dict[str, str]]:
    cases_file = _CASSETTES / "_test_cases.json"
    assert cases_file.is_file(), "advisor cassette index missing"
    return list(json.loads(cases_file.read_text(encoding="utf-8")))


def test_ten_cassettes_present() -> None:
    """The advisor gate test requires 10 (ticker, as_of_date) cassettes."""
    cases = _load_index()
    assert len(cases) == 10


def test_every_case_has_a_cassette() -> None:
    """For every entry in _test_cases.json, the corresponding cassette JSON exists."""
    for case in _load_index():
        cassette_path = _CASSETTES / f"{case['ticker']}_{case['as_of_date']}.json"
        assert cassette_path.is_file(), f"missing cassette for {case}"


def test_every_case_expected_accession_appears_in_its_cassette() -> None:
    """The expected accession must exist in the cassette, be an 8-K, and predate the cutoff."""
    for case in _load_index():
        cassette = json.loads(
            (_CASSETTES / f"{case['ticker']}_{case['as_of_date']}.json").read_text(
                encoding="utf-8"
            )
        )
        recent = cassette["filings"]["recent"]
        accessions: list[str] = recent["accessionNumber"]
        forms: list[str] = recent["form"]
        filing_dates: list[str] = recent["filingDate"]
        expected: str = case["expected_latest_8k_accession"]
        as_of = date.fromisoformat(case["as_of_date"])

        assert expected in accessions, f"expected accession missing from cassette: {case}"
        idx = accessions.index(expected)
        assert forms[idx] == "8-K", f"expected accession is not an 8-K: {case}"
        assert date.fromisoformat(filing_dates[idx]) <= as_of, (
            f"expected accession filed after the as_of cutoff: {case}"
        )


def test_cassette_arrays_are_parallel() -> None:
    """Every cassette's filings.recent arrays must share a single length."""
    for case in _load_index():
        cassette = json.loads(
            (_CASSETTES / f"{case['ticker']}_{case['as_of_date']}.json").read_text(
                encoding="utf-8"
            )
        )
        recent = cassette["filings"]["recent"]
        lengths = {
            key: len(value)
            for key, value in recent.items()
            if isinstance(value, list)
        }
        assert len(set(lengths.values())) == 1, (
            f"non-parallel arrays in {case['ticker']}_{case['as_of_date']}: {lengths}"
        )
