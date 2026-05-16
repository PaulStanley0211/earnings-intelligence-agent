# Sample uploaded PDFs (Phase 4 fixture set)

Real Microsoft SEC filings used as fixtures for the Phase 4 upload-intake and document-advisor tests.

All filings are public, available on SEC EDGAR. The canonical archive URL pattern is:

```
https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/
```

Microsoft's CIK is `789019`.

## Filings in this directory

| Accession (filename) | Type | Filed | Period / event | Size | Tracked in git? | EDGAR archive |
|---|---|---|---|---|---|---|
| `0000950170-25-100235.pdf` | 10-K | 2025 | FY2025 (ended Jun 30, 2025) | ~2.9 MB | No — fetch on demand | https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/ |
| `0001193125-26-027198.pdf` | 8-K | 2026-01-28 | Q2 FY2026 earnings release | ~360 KB | Yes | https://www.sec.gov/Archives/edgar/data/789019/000119312526027198/ |
| `0001193125-26-027207.pdf` | 10-Q | 2026-01-28 | Q2 FY2026 (ended Dec 31, 2025) | ~2.2 MB | No — fetch on demand | https://www.sec.gov/Archives/edgar/data/789019/000119312526027207/ |
| `0001193125-26-191457.pdf` | 8-K | 2026-04-29 | Q3 FY2026 earnings release | ~342 KB | Yes | https://www.sec.gov/Archives/edgar/data/789019/000119312526191457/ |
| `0001193125-26-224155.pdf` | 8-K | 2026-05-13 | Small item (~5 pages) | ~105 KB | Yes | https://www.sec.gov/Archives/edgar/data/789019/000119312526224155/ |

The three small 8-Ks are checked in so unit tests are hermetic. The 10-K and 10-Q are gitignored — they're easy to re-fetch and would bloat the repo.

## Re-fetching the gitignored files

Either click the EDGAR archive URL above and download the PDF, or use the Phase 1 EDGAR client (`app/tools/edgar.py`) to fetch programmatically. Save the result into this directory using the exact accession-number filenames above so the tests find them.

## Why MSFT only

Phase 4's gates explicitly need (a) commitments persisting across consecutive quarters and (b) at least one transcript per company. The Jan-28 and Apr-29 8-Ks are consecutive MSFT earnings events — perfect for the cross-quarter commitment test. Additional tickers can be added here as Phase 4 fixtures expand (e.g. NVDA, GOOGL).

## Not in this directory

Earnings-call transcripts. The earlier scanned-PDF transcripts were unusable (no embedded text) and were removed. Phase 4 expects transcripts as plain text or HTML, dropped in by the user via the upload UI — not stored in this fixtures directory. The transcript fixtures used to label the 50 Q&A pairs for the F1 gate will live next to the labelling docs in `tests/fixtures/transcripts/` once Phase 4 begins.
