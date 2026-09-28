"""The RM30 study harness, seed rules, reading table and run form, without a study run.

Every fit here draws from a throwaway seed at a small size, and no test prints or asserts a
bias, a coverage or an SE ratio.  The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.canonical import learned_rule_run
from tests.canonical.regenerate import ARTIFACT_NAMES
from tests.diagnostics.rm18_shared import THREAD_VARIABLES
from tests.studies import _learned_rule_law as law
from tests.studies import learned_rule_cvtmle as gated
from tests.studies import learned_rule_cvtmle_boundary as boundary
from tests.studies import learned_rule_cvtmle_properties as properties
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.manifest import write_manifest
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_990_000
N = 400


@pytest.fixture(scope="module")
def fitted() -> tuple[pd.DataFrame, object]:
    frame = law.draw("non_exceptional", N, THROWAWAY)
    return frame, law.fit(frame, law.outcome_learner(), THROWAWAY + 1)


class TestTheSeedRules:
    def test_rule_l3_moves_only_the_calibration_cell(self) -> None:
        assert properties.suffixes(gated.STUDY) == properties.DECLARED_SUFFIXES
        moved = {key for key, suffix in properties.DECLARED_SUFFIXES.items() if suffix}
        assert moved == {("interval_calibration", "non_exceptional__correctly_specified")}

    def test_the_collision_that_moved_it_is_real(self) -> None:
        """The nonzero witness: the unmoved label repeats a primary sample."""
        label = ("property_sample", "interval_calibration", "non_exceptional__correctly_specified")
        assert stream_seed(gated.STUDY, *label, 4_893) == replicate_seed(
            gated.STUDY, "non_exceptional", 4_696
        )
        assert stream_seed(gated.STUDY, *label, 4_893, "retry", 1) != replicate_seed(
            gated.STUDY, "non_exceptional", 4_696
        )

    def test_the_declared_fold_seeds_hold_one_equal_pair(self) -> None:
        seeds = [
            stream_seed(record, "fold_partition", "primary", scenario, r)
            for record in (gated.STUDY, boundary.STUDY)
            for scenario in record.scenarios
            for r in range(6_000)
        ]
        resolved = properties.suffixes(gated.STUDY)
        for draw in properties.PROPERTY_DRAWS:
            suffix = resolved[draw.family, draw.label]
            seeds += [
                stream_seed(gated.STUDY, "fold_partition", draw.family, draw.label, r, *suffix)
                for r in range(draw.replicates)
            ]
        assert (len(seeds), len(set(seeds))) == (64_310, 64_309)

    def test_no_smoke_sample_seed_is_declared(self) -> None:
        declared = learned_rule_run.declared_sample_seeds()
        assert len(declared) == 46_310 + 6_000 + 12_000
        for study in (gated, boundary):
            smoke = law.smoke_record(study.STUDY)
            seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
            assert not seeds & declared
            # The deliberate mutation: the declared record run as a smoke would be refused.
            assert replicate_seed(study.STUDY, next(iter(study.SCENARIOS)), 0) in declared


class TestTheHarness:
    def test_the_refit_reproduces_the_fits_rule_on_every_row(self, fitted) -> None:
        _, result = fitted
        truth = law.harness(result, law.outcome_learner(), "non_exceptional")
        assert truth.rows_checked == N
        assert len(truth.fold_values) == law.N_FOLDS
        assert truth.truth == pytest.approx(np.mean(truth.fold_values), rel=1e-15)
        assert truth.oracle_se > 0.0

    def test_a_rule_that_differs_on_one_row_stops_the_replication(
        self, fitted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, result = fitted
        honest = law.rule_of(result)

        def flipped(result: object) -> np.ndarray:
            rule = honest.copy()
            rule[7] = ~rule[7]
            return rule

        monkeypatch.setattr(law, "rule_of", flipped)
        with pytest.raises(law.HarnessMismatch, match="1 of"):
            law.harness(result, law.outcome_learner(), "non_exceptional")

    def test_the_validation_row_mutation_learns_a_different_rule(self) -> None:
        frame = law.draw("non_exceptional", N, THROWAWAY + 2)
        local = law.fit(frame, law.outcome_learner(), THROWAWAY + 3)
        mutated = law.fit(frame, law.outcome_learner(), THROWAWAY + 3, mutation=True)
        assert np.array_equal(local.nuisance.folds.assignment, mutated.nuisance.folds.assignment), (
            "the two arms must share one partition"
        )
        assert (law.rule_of(local) != law.rule_of(mutated)).any()
        truth = law.harness(mutated, law.outcome_learner(), "non_exceptional", source="validation")
        assert truth.rows_checked == N
        with pytest.raises(law.HarnessMismatch):
            law.harness(mutated, law.outcome_learner(), "non_exceptional")

    def test_the_untargeted_control_is_the_fit_without_its_fluctuation(self, fitted) -> None:
        _, result = fitted
        nuisance = result.nuisance
        rule = law.rule_of(result)
        plug_in = np.where(rule, nuisance.outcome.arms[1.0], nuisance.outcome.arms[0.0])
        folds = nuisance.folds.assignment
        expected = np.mean([plug_in[folds == v].mean() for v in range(law.N_FOLDS)])
        assert law.untargeted_estimate(result) == pytest.approx(expected, rel=1e-14)
        assert law.untargeted_estimate(result) != result.estimates[law.ESTIMAND].psi

    def test_rule_value_integrates_a_rule_on_the_declared_grid(self) -> None:
        assert law.GRID.size == 4_001
        grid = law.fixed_rule_grid(lambda w1, w2: np.ones_like(w1, dtype=bool))
        value, _ = law.rule_value(grid, "non_exceptional")
        treated = [law.qbar0(1.0, law.GRID, w2, "non_exceptional") for w2 in law.W2_LEVELS]
        assert value == pytest.approx(
            np.mean([np.trapezoid(q, law.GRID) / 2.0 for q in treated]), rel=1e-15
        )

    def test_a_primary_row_carries_its_own_truth_and_the_harness_columns(self) -> None:
        smoke = law.smoke_record(gated.STUDY)
        row = law.primary_row(smoke, "misspecified_limit", 0, N)
        assert row["rule_rows_checked"] == N
        assert row["solver_warnings"] == 0
        assert row["covered"] == int(row["ci_lower"] <= row["truth"] <= row["ci_upper"])
        assert set(law.HARNESS_COLUMNS) <= set(row)


class TestTheBoundaryReading:
    @staticmethod
    def _rows(covered: dict[str, int], replicates: int = 6_000) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"scenario": scenario, "covered": int(index < hits)}
                for scenario, hits in covered.items()
                for index in range(replicates)
            ]
        )

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            # Both conditions hold, and the first one read names the reading.
            (0.91, 0.94, "under-covers at the exceptional law"),
            (0.92, 0.96, "no under-coverage resolved at the declared budget"),
            (0.89, 0.96, "unresolved"),
        ],
    )
    def test_each_reading_is_the_first_condition_that_holds(
        self, low: float, high: float, expected: str
    ) -> None:
        assert boundary.read(Interval(low, high), "exceptional") == expected

    @pytest.mark.parametrize(
        ("hits", "expected"),
        [
            (5_400, "under-covers at the exceptional law"),
            (5_700, "no under-coverage resolved at the declared budget"),
        ],
    )
    def test_the_table_reads_the_99_percent_exact_interval(self, hits: int, expected: str) -> None:
        table = boundary.reading(self._rows({"exceptional": hits, "weak_blip": 5_700}))
        row = table.set_index("scenario").loc["exceptional"]
        assert row["reading"] == expected
        assert row["confidence_level"] == 0.99
        assert row["coverage"] == hits / 6_000

    def test_a_smoke_run_writes_no_reading(self) -> None:
        table = boundary.reading(self._rows({"exceptional": 3, "weak_blip": 4}, replicates=4))
        assert set(table["reading"]) == {boundary.SMOKE}


def _git_status() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _writing_driver(*, fail: str | None = None) -> Any:
    """A stand-in for the shared driver that writes every artefact and a real manifest.

    ``fail="before"`` raises before the manifest exists, as a failed fit does (rule L4).
    ``fail="after"`` raises after it, as a failed gated verdict does.
    """

    def driver(study: Any, properties: Any, *, here: Path) -> None:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if fail == "before":
            raise law.HarnessMismatch("fold 0: the refit rule differs")
        record = study.STUDY
        paths = [output / name for name in (*ARTIFACT_NAMES, *record.extra_artifacts)]
        for path in paths:
            path.write_text("artefact\n", encoding="utf-8", newline="\n")
        write_manifest(output / "manifest.json", record, paths)
        if fail == "after":
            raise RuntimeError("independent performance gates failed")

    return driver


class TestTheRunForm:
    @staticmethod
    def _declared(
        monkeypatch: pytest.MonkeyPatch, argv: list[str], *, guard: list[str] | None = None
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["regenerate", *argv])
        monkeypatch.setattr(learned_rule_run, "refusals", lambda smoke: list(guard or []))
        monkeypatch.setattr(learned_rule_run, "runtime_refusals", list)

    def test_a_declared_run_refuses_what_rule_r6_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._declared(
            monkeypatch, ["--output", str(tmp_path / "run")], guard=["the tree has changes"]
        )
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="the tree has changes"):
            learned_rule_run.run(gated, properties, here=tmp_path / "here")

    @pytest.mark.parametrize(
        "extra",
        [
            ["--n", "500"],
            ["--primary-only"],
            ["--skip-properties"],
            ["--allow-failures"],
            ["--cache", "somewhere"],
            ["--refresh-python"],
        ],
    )
    def test_a_declared_run_refuses_every_flag_but_output_and_jobs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]
    ) -> None:
        self._declared(monkeypatch, ["--output", str(tmp_path / "run"), "--jobs", "2", *extra])
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="--output and --jobs only"):
            learned_rule_run.run(gated, properties, here=tmp_path / "here")

    def test_a_declared_run_refuses_the_declared_count_named_explicitly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._declared(monkeypatch, ["--replicates", "6000", "--output", str(tmp_path)])
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="passes no --replicates"):
            learned_rule_run.run(gated, properties, here=tmp_path / "here")

    @pytest.mark.parametrize(
        ("argv", "match"),
        [
            ([], "needs a scratch --output"),
            (["--output", str(ROOT / "tests" / "canonical" / "learned_rule_cvtmle")], "outside"),
        ],
    )
    def test_a_declared_run_writes_to_scratch_outside_the_repository(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], match: str
    ) -> None:
        self._declared(monkeypatch, argv)
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match=match):
            learned_rule_run.run(gated, properties, here=tmp_path / "here")

    def test_a_declared_run_refuses_a_scratch_output_that_holds_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "stale.csv").write_text("x\n", encoding="utf-8")
        self._declared(monkeypatch, ["--output", str(tmp_path)])
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="not empty"):
            learned_rule_run.run(gated, properties, here=tmp_path / "here")

    def test_the_runtime_refusals_name_threads_and_versions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in THREAD_VARIABLES:
            monkeypatch.setenv(name, "1")
        monkeypatch.setattr(learned_rule_run, "PYTHON", platform.python_version())
        monkeypatch.setattr(
            learned_rule_run,
            "PACKAGES",
            {name: importlib.import_module(name).__version__ for name in learned_rule_run.PACKAGES},
        )
        assert learned_rule_run.runtime_refusals() == []
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setitem(learned_rule_run.PACKAGES, "sklearn", "0.0.0")
        refused = learned_rule_run.runtime_refusals()
        assert refused == [
            f"sklearn is {importlib.import_module('sklearn').__version__}, not 0.0.0 (rule L7)",
            "OMP_NUM_THREADS is 4, not 1 (rule R6)",
        ]

    def test_the_declared_packages_are_the_l7_list(self) -> None:
        assert learned_rule_run.PYTHON == "3.13.7"
        assert learned_rule_run.PACKAGES == {
            "numpy": "2.4.6",
            "scipy": "1.18.0",
            "pandas": "3.0.5",
            "sklearn": "1.9.0",
            "joblib": "1.5.3",
        }

    def test_the_manifest_sees_the_tree_as_the_run_found_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """R1: nothing is written inside the repository before the manifest records its state.

        On a clean pushed tree the manifest therefore records ``cleverly_worktree_clean: true``.
        The driver used to write into the study directory first, which made every declared
        manifest record ``false``.
        """
        before = _git_status()
        here = tmp_path / "here"
        self._declared(monkeypatch, ["--output", str(tmp_path / "run"), "--jobs", "3"])
        calls: list[list[str]] = []
        driver = _writing_driver()

        def recording(study: Any, properties: Any, *, here: Path) -> None:
            calls.append(list(sys.argv))
            driver(study, properties, here=here)

        monkeypatch.setattr(learned_rule_run, "regenerate", recording)
        learned_rule_run.run(gated, properties, here=here)
        assert calls == [["regenerate", "--jobs", "3", "--output", str(tmp_path / "run")]]
        manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
        clean = manifest["generated_with"]["subject"]["cleverly_worktree_clean"]
        assert clean is (before == "")
        assert _git_status() == before
        expected = {*ARTIFACT_NAMES, *gated.STUDY.extra_artifacts, "manifest.json", "run.log"}
        assert {path.name for path in here.iterdir()} == expected
        log = (here / "run.log").read_text(encoding="utf-8")
        assert "declared run" in log
        assert "exit code: 0" in log

    def test_the_output_of_the_old_path_would_have_dirtied_the_tree(self) -> None:
        """The control: the study directory is inside the repository, so writing there first
        is what the manifest's ``git status`` saw."""
        assert gated.STUDY.artifacts.resolve().is_relative_to(ROOT.resolve())

    def test_a_failed_gated_verdict_still_publishes_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        here = tmp_path / "here"
        self._declared(monkeypatch, ["--output", str(tmp_path / "run")])
        monkeypatch.setattr(learned_rule_run, "regenerate", _writing_driver(fail="after"))
        with pytest.raises(RuntimeError, match="gates failed"):
            learned_rule_run.run(gated, properties, here=here)
        assert (here / "manifest.json").exists()
        assert "exit code: 1" in (here / "run.log").read_text(encoding="utf-8")

    def test_a_run_that_stops_before_its_manifest_publishes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        here = tmp_path / "here"
        self._declared(monkeypatch, ["--output", str(tmp_path / "run")])
        monkeypatch.setattr(learned_rule_run, "regenerate", _writing_driver(fail="before"))
        with pytest.raises(law.HarnessMismatch):
            learned_rule_run.run(gated, properties, here=here)
        assert not here.exists()

    def test_a_smoke_run_refuses_an_output_inside_the_repository(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inside = ROOT / "tests" / "canonical" / "learned_rule_cvtmle"
        monkeypatch.setattr(
            sys, "argv", ["regenerate", "--replicates", "4", "--output", str(inside)]
        )
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="outside the repository"):
            learned_rule_run.run(gated, properties, here=gated.STUDY.artifacts)

    def test_the_harness_check_flag_is_refused_on_a_declared_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["regenerate", "--property-harness-check", "2"])
        monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)
        with pytest.raises(SystemExit, match="smoke-only"):
            learned_rule_run.run(gated, properties, here=gated.STUDY.artifacts)

    def test_a_smoke_study_draws_from_the_throwaway_record(self) -> None:
        smoke = law.smoke_record(gated.STUDY)
        proxy = learned_rule_run._smoke_study(gated, smoke)
        assert proxy.STUDY is smoke
        assert proxy.draw_and_fit.keywords == {"record": smoke}
        assert smoke.seed != gated.STUDY.seed
        assert smoke.resampling_seed != gated.STUDY.resampling_seed


def test_a_non_finite_untargeted_estimate_stops_the_replication(
    fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = fitted
    monkeypatch.setattr(law.np, "mean", lambda values: float("nan"))
    with pytest.raises(RuntimeError, match="non-finite untargeted"):
        law.untargeted_estimate(result)
