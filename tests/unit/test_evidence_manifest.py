"""The evidence record, checked against itself.

``docs/drtmle/study-manifest.md`` carries every dispatched study's artefact identifiers and
digests in prose, and ``evidence/manifest.json`` carries the same record in the form
``scripts/fetch_evidence.sh`` reads.  **Two representations of one fact is a duplication, and
it is deliberate**: a shell script cannot check a digest it has to parse out of a markdown
table, and a reader cannot audit a study from a JSON file.  What stops them drifting is this
module, which asserts they cover each other in *both* directions -- the same instrument
``tests/unit/test_registry.py`` points at the target registry and its oracle laws.

Both directions, because the two failures are different and only one of them is loud.  A row
in the JSON with no row in the prose is a digest nothing documents; a row in the prose with no
row in the JSON is a digest the fetch will silently skip, and the study would come back four
artefacts short with an exit status of zero.

**The digest is the anchor these are matched on.**  A ``sha256`` is the one field in either
table that cannot be reconstructed from the others, so parsing on it makes the check
independent of the columns each table happens to carry -- ``c3c``'s rows are keyed by batch
and ``e1b``'s by tier and size, and no common key exists that is not the artefact itself.

What this does *not* check is that the digests are right.  Nothing in this repository can:
they were read off the GitHub Actions API on 2026-08-05 and the payload they describe is not
here.  ``scripts/fetch_evidence.sh`` is what checks them, on a machine that can reach blob
storage, and until it has run this manifest names evidence the repository does not carry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "evidence" / "manifest.json"
PROSE = ROOT / "docs" / "drtmle" / "study-manifest.md"

#: A markdown table row carrying a digest, which is the only row shape either table has in
#: common.  ``artefact`` and ``bytes`` are read relative to it rather than by column index,
#: since the two tables put them in different positions.
_ROW = re.compile(
    r"^\|.*\|\s*`?(?P<artefact>\d{6,})`?\s*\|\s*(?P<bytes>[\d,]+)\s*\|\s*`(?P<sha>[0-9a-f]{64})`\s*\|\s*$"
)


def _prose_rows() -> set[tuple[int, int, str]]:
    """``(artefact, bytes, sha256)`` for every digest row of the prose manifest."""
    rows = set()
    for line in PROSE.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line.strip())
        if match is not None:
            rows.add(
                (
                    int(match["artefact"]),
                    int(match["bytes"].replace(",", "")),
                    match["sha"],
                )
            )
    return rows


def _json_rows() -> set[tuple[int, int, str]]:
    """The same triples, out of every study in the machine-readable manifest."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        (int(each["artefact"]), int(each["bytes"]), each["sha256"])
        for study in payload["studies"].values()
        for each in study["artefacts"]
    }


class TestTheTwoManifestsCoverEachOther:
    """The both-directions check, one test each way so a failure says which way round."""

    def test_every_json_artefact_is_documented(self) -> None:
        missing = _json_rows() - _prose_rows()
        assert not missing, f"in evidence/manifest.json and not in the prose: {sorted(missing)}"

    def test_every_documented_artefact_is_fetchable(self) -> None:
        """A prose row with no JSON row is the quiet failure: the fetch would skip it."""
        missing = _prose_rows() - _json_rows()
        assert not missing, f"documented and not fetchable: {sorted(missing)}"

    def test_the_parser_found_something(self) -> None:
        """Guards the check itself.

        Both assertions above pass vacuously if the regex stops matching -- a table reformatted
        or a digest backticked differently -- and would then report agreement between two empty
        sets.  Twelve is what is on record: four for ``c3c`` and eight for ``e1b``.
        """
        assert len(_prose_rows()) == 12
        assert len(_json_rows()) == 12


class TestTheManifestSaysWhatTheFetchNeeds:
    """The fields ``scripts/fetch_evidence.sh`` reads, and the expiry a reader is owed."""

    def test_the_studies_are_the_two_on_record(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert sorted(payload["studies"]) == ["c3c", "e1b"]

    def test_every_artefact_carries_the_fields_the_script_reads(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, study in payload["studies"].items():
            for each in study["artefacts"]:
                for field in ("artefact", "sha256", "bytes"):
                    assert field in each, f"{name}: {each} has no {field}"

    def test_the_retention_date_is_the_one_the_prose_states(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        expires = payload["retention_expires"]
        assert expires == "2026-11-03"
        # Twice, because both studies expire on it and each says so where it is read.
        assert PROSE.read_text(encoding="utf-8").count(expires) >= 2

    @pytest.mark.parametrize(("study", "count"), [("c3c", 4), ("e1b", 8)])
    def test_each_study_carries_the_artefact_count_it_dispatched(
        self, study: str, count: int
    ) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert len(payload["studies"][study]["artefacts"]) == count


class TestTheDispatchedDrawCountIsTheOneOnRecord:
    """The correction this module landed beside, pinned so it cannot silently revert.

    The E1b section said ``16`` draws in three places while its own spot checks, the
    investigation log and the roadmap all said ``32`` -- and the dispatched runs' logs print
    ``draws 32``, ``fits 32`` and 1,024 or 1,280 replicate rows.  The arithmetic is what
    settles it: a fit contributes ``8 * rungs + 8`` rows, so tier 2's three rungs give 32 rows
    a fit and ``1,024 / 32 = 32`` fits.

    Pinned as a *string* check on the document rather than recomputed, because what failed here
    was a transcription and not a calculation.
    """

    def test_the_manifest_does_not_claim_sixteen_draws(self) -> None:
        text = PROSE.read_text(encoding="utf-8")
        assert "| draws | **32** per" in text
        assert "--draws 32" in text
        assert "--draws 16" not in text

    def test_the_superseded_runs_keep_their_own_draw_count(self) -> None:
        """The two earlier runs really did run at 16, and that sentence is history, not a bug."""
        text = PROSE.read_text(encoding="utf-8")
        assert "ran at `d2982501` with 16 draws" in text

    def test_the_fit_total_follows_the_draw_count(self) -> None:
        """32 draws over two cells and two sizes is 128 fits a tier, not 64."""
        text = PROSE.read_text(encoding="utf-8")
        assert "128 fits a tier" in text
