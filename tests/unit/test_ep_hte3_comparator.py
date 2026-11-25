"""The hte3 comparison of the EP curve, in its pending state and after its run (P1 plan, PR 4b).

Before the run: no run file is committed, the parent pages say the comparison is pending, the
driver refuses a declared run, the image pins the tmle3 study's base and commits, and the
readings publish a failed cell as failed.  After the run (the tests skip until
``tests/canonical/hte3_ep/manifest.json`` exists): the committed hashes are the bytes on disk,
and the readings recompute from the committed rows.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.hte3_ep import regenerate
from tests.studies import ep_hte3 as study
from tests.studies.evidence.manifest import reference_hashes
from tests.studies.evidence.registry import ROOT

MANIFEST = study.ARTIFACTS / "manifest.json"
PAGES = {
    "ep-cate": ROOT / "docs" / "technical-reference" / "method-evidence" / "ep-cate.md",
    "ep-crr": ROOT / "docs" / "technical-reference" / "method-evidence" / "ep-crr.md",
}
ran = pytest.mark.skipif(not MANIFEST.exists(), reason="the hte3 comparison has not been run")


def synthetic_rows(cell: str, *, d: float, hte3_ratio: float, t_gap: float) -> pd.DataFrame:
    """Forty rows of one cell with the given mean differences and unit-scale noise of 0.05."""
    rng = np.random.default_rng(0)
    m = 40
    oracle = rng.normal(-3.0, 0.05, m)
    hte3 = oracle + hte3_ratio + rng.normal(0.0, 0.05, m)
    return pd.DataFrame(
        {
            "cell": cell,
            "replicate": np.arange(m),
            "log_ise_cleverly": hte3 + d + rng.normal(0.0, 0.05, m),
            "log_ise_hte3": hte3,
            "log_ise_oracle": oracle,
            "log_ise_t": hte3 + t_gap + rng.normal(0.0, 0.05, m),
            "curve_distance": rng.uniform(0.0, 0.1, m),
            "drift": rng.normal(0.0, 1e-9, m),
            "emulation_gap": rng.uniform(0.0, 1e-8, m),
        },
        columns=list(study.REPLICATE_COLUMNS),
    )


class TestThePendingState:
    @pytest.mark.skipif(MANIFEST.exists(), reason="the comparison has been run")
    def test_no_run_file_is_committed(self) -> None:
        present = [name for name in study.RUN_FILES if (study.ARTIFACTS / name).exists()]
        assert present == [], "a run file is committed while REPLICATES is empty"
        assert not study.REPLICATES and not study.REPORTED

    @pytest.mark.skipif(MANIFEST.exists(), reason="the comparison has been run")
    @pytest.mark.parametrize("parent", sorted(PAGES))
    def test_the_parent_page_says_the_comparison_is_pending(self, parent: str) -> None:
        text = PAGES[parent].read_text(encoding="utf-8")
        section = text.split("## Comparison with hte3", 1)
        assert len(section) == 2, f"{parent} has no Comparison with hte3 section"
        assert "comparator runs pending" in section[1].split("\n## ", 1)[0]
        body = section[1].split("\n## ", 1)[0]
        # The table names each cell, alone or as an end of a range such as "C1 to C4".
        cells = [cell for cell in study.CELLS if cell.parent == parent]
        named = {cells[0].label, cells[-1].label} | {
            cell.label for cell in cells if cell.role != study.GATED
        }
        for label in named:
            assert re.search(rf"\b{label}\b", body), f"{label} is not in the table"

    @pytest.mark.skipif(bool(study.REPLICATES), reason="the comparison has been declared")
    def test_the_driver_refuses_a_declared_run(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="pending its declaration"):
            regenerate.main(["--output", str(tmp_path / "run")])

    @pytest.mark.parametrize("smoke", [0, study.MUTATION_REPLICATES + 1])
    def test_the_driver_refuses_a_smoke_outside_its_range(self, tmp_path: Path, smoke: int) -> None:
        with pytest.raises(SystemExit, match="--smoke takes"):
            regenerate.main(["--output", str(tmp_path / "smoke"), "--smoke", str(smoke)])

    def test_the_driver_refuses_an_output_in_the_repository(self) -> None:
        with pytest.raises(SystemExit, match="outside the repository"):
            regenerate.main(["--output", str(study.ARTIFACTS / "scratch"), "--smoke", "1"])


class TestTheImage:
    def test_the_base_and_the_tlverse_commits_are_the_tmle3_studys(self) -> None:
        ours = study.DOCKERFILE.read_text(encoding="utf-8")
        theirs = (ROOT / "tests" / "canonical" / "tmle3" / "Dockerfile").read_text(encoding="utf-8")
        base = re.search(r"^FROM (\S+)$", theirs, re.MULTILINE)
        assert base is not None and f"FROM {base.group(1)}\n" in ours
        for name in ("SL3_COMMIT", "TMLE3_COMMIT"):
            pinned = re.search(rf"^ARG {name}=([0-9a-f]{{40}})$", theirs, re.MULTILINE)
            assert pinned is not None and f"ARG {name}={pinned.group(1)}\n" in ours
        assert f"ARG HTE3_COMMIT={study.HTE3_COMMIT}\n" in ours
        assert f"ARG SL3_COMMIT={study.SL3_COMMIT}\n" in ours
        assert f"ARG TMLE3_COMMIT={study.TMLE3_COMMIT}\n" in ours

    @pytest.mark.parametrize(
        "package", ["tmle3", "hte3", "Sieve", "glmnet", "data.table", "uuid", "R6", "combinat"]
    )
    def test_each_package_hte3_calls_is_checked_against_the_snapshot(self, package: str) -> None:
        text = study.DOCKERFILE.read_text(encoding="utf-8")
        assert f"packageVersion('{package}')) == '" in text

    def test_the_runner_hands_hte3_the_supplied_nuisances(self) -> None:
        text = study.RUNNER.read_text(encoding="utf-8")
        assert "cross_fit_and_cv = FALSE" in text
        assert 'targeting_style = "dr"' in text
        assert "check_round_trip(task, mu, pi, label)" in text
        assert "sieve_interaction_order = 1" in text
        assert 'source("/fixture/study_harness.R")' in text
        # The round trip stops the run: it is not wrapped in a stage, which would cost one
        # replicate instead.
        assert not re.search(r"study_stage\([^)]*check_round_trip", text)


class TestTheReadings:
    def test_a_red_noninferiority_reading_is_published_false(self) -> None:
        rows = synthetic_rows("E1", d=math.log(1.6), hte3_ratio=0.0, t_gap=0.0)
        reading = study.readings(rows).set_index("cell").loc["E1"]
        assert reading["role"] == study.GATED
        assert not bool(reading["noninferior"])
        assert reading["d_upper"] > study.NONINFERIORITY_MARGIN

    def test_a_green_reading_is_published_true(self) -> None:
        rows = synthetic_rows("E1", d=0.0, hte3_ratio=0.0, t_gap=0.0)
        reading = study.readings(rows).set_index("cell").loc["E1"]
        assert bool(reading["noninferior"]) and bool(reading["similar"])

    def test_the_witness_fails_when_hte3_is_replaced_by_the_t_learner(self) -> None:
        rows = synthetic_rows("E8", d=0.0, hte3_ratio=0.0, t_gap=math.log(3.0))
        reading = study.readings(rows).set_index("cell").loc["E8"]
        assert bool(reading["witness"]), "the synthetic hte3 beats its T-learner"
        copied = rows.assign(log_ise_hte3=rows["log_ise_t"])
        reading = study.readings(copied).set_index("cell").loc["E8"]
        assert not bool(reading["witness"]), "a comparator that is its T-learner passes"

    def test_a_mutation_reads_failed_when_it_moves_the_curve(self) -> None:
        rows = synthetic_rows("M2", d=0.0, hte3_ratio=math.log(5.0), t_gap=0.0)
        reading = study.readings(rows).set_index("cell").loc["M2"]
        assert reading["role"] == study.MUTATION and bool(reading["mutation_failed"])
        unmoved = synthetic_rows("M2", d=0.0, hte3_ratio=0.0, t_gap=0.0)
        reading = study.readings(unmoved).set_index("cell").loc["M2"]
        assert not bool(reading["mutation_failed"])

    def test_a_reported_cell_keeps_its_reading(self) -> None:
        rows = synthetic_rows("E7", d=math.log(1.6), hte3_ratio=0.0, t_gap=0.0)
        reading = study.readings(rows, frozenset({"E7"})).set_index("cell").loc["E7"]
        assert reading["role"] == "reported" and not bool(reading["noninferior"])


@ran
class TestTheRun:
    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_the_hashed_sources_are_the_bytes_on_disk(self, manifest: dict) -> None:
        assert manifest["reference_sha256"] == reference_hashes(regenerate.SOURCES)

    def test_the_run_used_the_declared_replicates(self, manifest: dict) -> None:
        assert manifest["replicates"] == dict(study.REPLICATES)
        assert manifest["reported"] == sorted(study.REPORTED)

    def test_the_readings_recompute_from_the_rows(self) -> None:
        rows = pd.read_csv(study.ARTIFACTS / "replicates.csv.gz")
        committed = pd.read_csv(study.ARTIFACTS / "comparison.csv")
        recomputed = study.readings(rows)
        pd.testing.assert_frame_equal(
            committed.reset_index(drop=True),
            recomputed.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-9,
        )

    def test_each_mutation_control_failed(self) -> None:
        committed = pd.read_csv(study.ARTIFACTS / "comparison.csv").set_index("cell")
        for cell in study.MUTATION_CELLS:
            assert bool(committed.loc[cell.label, "mutation_failed"]), cell.label

    def test_the_emulation_tracks_hte3(self) -> None:
        rows = pd.read_csv(study.ARTIFACTS / "replicates.csv.gz")
        assert float(rows["emulation_gap"].max()) <= study.EMULATION_TOLERANCE
