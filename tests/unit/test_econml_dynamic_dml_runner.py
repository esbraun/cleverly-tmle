"""The EconML ``DynamicDML`` study runner of ``structural-nested-mean`` (X44, row 43b).

The runner's image installs no ``cleverly``, so it carries copies of the ``GLM`` design, the
failure classifier and the row rules.  These tests pin each copy to its source and check the
long-panel layout and the failure handling without EconML: a stub stands in for the fit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.learners import GLM
from cleverly.validation import _failures
from tests.canonical.econml_dynamic_dml import run_study as runner
from tests.studies import canonical_structural_nested_mean as study
from tests.studies.evidence.replication import invalid_rows
from tests.studies.evidence.schema import REPLICATE_COLUMNS


def _frame(n: int = 6, *, levels: int = 2, replicate: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(replicate)
    return pd.DataFrame(
        {
            "scenario": "s",
            "replicate": replicate,
            "W": rng.integers(0, 2, n).astype(float),
            "S1": rng.integers(0, 2, n).astype(float),
            "A1": rng.integers(0, levels, n).astype(float),
            "S2": rng.integers(0, 2, n).astype(float),
            "A2": rng.integers(0, levels, n).astype(float),
            "Y": rng.normal(size=n),
            "fold": np.arange(n) % 3,
        }
    )


def _task(variant: str = "paired", kind: str = "binary", **change: Any) -> runner.Task:
    settings: dict[str, Any] = {
        "scenario": "s",
        "variant": variant,
        "kind": kind,
        "nodes": 2,
        "order": 2,
        "features": (),
        "estimands": (("blip[A1:1.0:intercept]", 1.0, None, None),),
        **change,
    }
    return runner.Task(**settings)


class TestCopies:
    def test_the_failure_lists_are_cleverlys_without_its_own_classes(self) -> None:
        def foreign(classes: tuple[type[BaseException], ...]) -> tuple[type[BaseException], ...]:
            return tuple(c for c in classes if not c.__module__.startswith("cleverly"))

        assert foreign(_failures.FATAL) == runner.FATAL
        assert foreign(_failures.RECORDED) == runner.RECORDED
        assert (*_failures.LIBRARY_MODULES, "econml") == runner.LIBRARY_MODULES

    @pytest.mark.parametrize(
        "change",
        [{}, {"estimate": float("nan")}, {"std_error": 0.0}, {"ci_lower": 9.0}, {"covered": 0}],
    )
    def test_the_row_rules_agree_with_the_python_phase(self, change: dict[str, Any]) -> None:
        row = {
            "scenario": "s",
            "estimand": "ate",
            "truth": 0.0,
            "estimate": 0.1,
            "inference_estimate": 0.1,
            "std_error": 0.1,
            "ci_lower": -0.2,
            "ci_upper": 0.4,
            "covered": 1,
            **change,
        }
        assert runner.invalid(row) == invalid_rows([row])

    def test_the_columns_are_the_studys(self) -> None:
        assert runner.ROW_COLUMNS == REPLICATE_COLUMNS
        assert runner.TASK_COLUMNS == study.TASK_COLUMNS
        assert runner.FAILURE_COLUMNS == study.MARKOV_FAILURE_COLUMNS

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_the_design_is_the_glm_design(self, order: int) -> None:
        x = np.random.default_rng(order).normal(size=(12, 4))
        y = x @ np.arange(1.0, 5.0) + x[:, 0] * x[:, 1]
        learner = GLM(family="gaussian", interactions=order).fit(x, y)
        np.testing.assert_allclose(
            runner.interaction_design(x, order), np.asarray(learner.full_design(x))
        )
        np.testing.assert_allclose(
            runner.InteractionLinear(order).fit(x, y).predict(x), learner.predict(x)
        )

    def test_the_indicator_logistic_is_the_glm_fit_per_level(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 2))
        labels = rng.integers(0, 3, 400)
        probabilities = runner.IndicatorLogistic(2).fit(x, labels).predict_proba(x)
        for level in (1, 2):
            target = (labels == level).astype(float)
            expected = GLM(interactions=2).fit(x, target).predict_proba(x)[:, 1]
            np.testing.assert_allclose(probabilities[:, level], expected, atol=1e-8)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_a_separated_logistic_fit_is_a_recorded_failure(self) -> None:
        x = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        with pytest.raises(np.linalg.LinAlgError) as raised:
            runner.IndicatorLogistic(1).fit(x, (x[:, 0] > 0).astype(int))
        assert runner.recordable(raised.value)


class TestLayout:
    def test_the_paired_state_is_the_zero_padded_history(self) -> None:
        frame = _frame(4, levels=3)
        rows = runner.states(_task(kind="categorical"), frame)
        # [W, S1, S2, A1 == 1, A1 == 2] per period; row 2 * i + t - 1 is unit i at period t.
        assert rows.shape == (8, 5)
        first, second = rows[0::2], rows[1::2]
        np.testing.assert_array_equal(first[:, 2:], 0.0)
        np.testing.assert_array_equal(first[:, :2], frame[["W", "S1"]].to_numpy())
        np.testing.assert_array_equal(second[:, :3], frame[["W", "S1", "S2"]].to_numpy())
        np.testing.assert_array_equal(second[:, 3], frame["A1"].to_numpy() == 1)
        np.testing.assert_array_equal(second[:, 4], frame["A1"].to_numpy() == 2)

    def test_the_markov_state_reads_the_current_covariate_and_last_treatment(self) -> None:
        frame = _frame(5)
        rows = runner.states(_task(variant="markov"), frame)
        assert rows.shape == (10, 3)
        np.testing.assert_array_equal(
            rows[0::2], np.column_stack([frame["W"], frame["S1"], 0.0 * frame["W"]])
        )
        np.testing.assert_array_equal(rows[1::2], frame[["W", "S2", "A1"]].to_numpy())

    def test_the_splits_keep_each_unit_whole_in_its_own_fold(self) -> None:
        folds = np.array([0, 1, 2, 0, 1, 2])
        for train, test in runner.splits(folds, 2):
            units = np.unique(test // 2)
            assert len(np.unique(folds[units])) == 1
            np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(12))
            assert set(test // 2).isdisjoint(train // 2)

    def test_the_tasks_round_trip_from_the_studys_table(self, tmp_path: Path) -> None:
        table = study.comparator_tasks()
        table.to_csv(tmp_path / "truth.csv", index=False)
        tasks = runner.read_tasks(
            pd.read_csv(tmp_path / "truth.csv", keep_default_na=False, na_values=[""])
        )
        assert set(tasks) == {
            *((s, "paired") for s in study.REFERENCE_SCENARIOS),
            *((s, "markov") for s in study.MARKOV_ESTIMANDS),
        }
        k = tasks[("k", "paired")]
        assert (k.kind, k.nodes, k.order, k.features) == ("categorical", 2, 2, ())
        contrast = {name: (t1, t0) for name, _, t1, t0 in k.estimands}
        assert contrast["ate_regimen[r12 vs r00]"] == ((1.0, 2.0), (0.0, 0.0))
        assert contrast["blip[A1:1.0:intercept]"] == (None, None)
        assert tasks[("h", "markov")].features == ("W",)

    @pytest.mark.parametrize(
        ("kind", "estimand", "position"),
        [
            ("binary", "blip[A2:1.0:intercept]", (1, "intercept")),
            ("categorical", "blip[A1:2.0:intercept]", (1, "intercept")),
            ("categorical", "blip[A2:1.0:intercept]", (2, "intercept")),
            ("continuous", "blip[A2:dose:intercept]", (1, "intercept")),
            ("binary", "blip[A1:1.0:W]", (0, "W")),
        ],
    )
    def test_a_blip_reads_its_period_major_coefficient(
        self, kind: str, estimand: str, position: tuple[int, str]
    ) -> None:
        assert runner._blip_position(_task(kind=kind), estimand, 3) == position


class TestFailures:
    @pytest.fixture
    def plan(self, monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
        plan: dict[int, str] = {}

        def rows_of(task: runner.Task, frame: pd.DataFrame) -> list[dict[str, Any]]:
            replicate = int(frame["replicate"].iloc[0])
            effect = plan.get(replicate)
            if effect == "singular":
                np.linalg.inv(np.zeros((2, 2)))
            if effect == "type":
                raise TypeError("a bug")
            se = 0.0 if effect == "zero se" else 0.1
            return [
                {
                    "implementation": runner.IMPLEMENTATION,
                    "scenario": task.scenario,
                    "replicate": replicate,
                    "n": len(frame),
                    "estimand": name,
                    "truth": truth,
                    "estimate": truth,
                    "inference_estimate": truth,
                    "std_error": se,
                    "ci_lower": truth - 0.2,
                    "ci_upper": truth + 0.2,
                    "inference_scale": "identity",
                    "covered": 1,
                    "initial_estimate": truth + 1.0,
                }
                for name, truth, _, _ in task.estimands
            ]

        monkeypatch.setattr(runner, "rows_of", rows_of)
        return plan

    def _jobs(self, replicates: int) -> list[runner.Job]:
        tasks = {("s", "paired"): _task(), ("s", "markov"): _task(variant="markov")}
        samples = pd.concat([_frame(replicate=r) for r in range(replicates)], ignore_index=True)
        return runner.jobs_of(samples, tasks)

    def test_a_recorded_failure_costs_its_replicate_in_both_variants(
        self, plan: dict[int, str]
    ) -> None:
        plan.update({1: "singular", 2: "zero se"})
        out = runner.fit_all(self._jobs(4), cores=1)
        for variant in runner.VARIANTS:
            rows, failures = out[variant]
            assert [row["replicate"] for row in rows] == [0, 3]
            assert [(f["class"], f["replicates"]) for f in failures] == [
                ("fitting_failure", "1"),
                ("invalid_result", "2"),
            ]
        assert out["paired"][1][0]["group_id"] == "g000000003"
        assert out["markov"][1][0]["group_id"] == "g000000004"

    def test_a_bug_stops_the_run(self, plan: dict[int, str]) -> None:
        plan[0] = "type"
        with pytest.raises(TypeError):
            runner.fit_all(self._jobs(2), cores=1)

    def test_the_systematic_stop(self, plan: dict[int, str]) -> None:
        plan.update(dict.fromkeys(range(20), "singular"))
        with pytest.raises(RuntimeError, match="systematic"):
            runner.fit_all(self._jobs(22), cores=1)

    def test_main_writes_both_variants_and_their_records(
        self, plan: dict[int, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan[1] = "singular"
        samples = pd.concat([_frame(replicate=r) for r in range(3)], ignore_index=True)
        samples.to_csv(tmp_path / "samples.csv.gz", index=False)
        tasks = pd.DataFrame(
            [
                {
                    "scenario": "s",
                    "variant": variant,
                    "estimand": "blip[A1:1.0:intercept]",
                    "truth": 1.0,
                    "kind": "binary",
                    "nodes": 2,
                    "order": 2,
                    "features": "",
                    "t1": "",
                    "t0": "",
                }
                for variant in runner.VARIANTS
            ]
        )
        tasks.to_csv(tmp_path / "truth.csv", index=False)
        output = tmp_path / "reference-results.csv"
        monkeypatch.setenv("CLEVERLY_REFERENCE_CORES", "1")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run", str(tmp_path / "samples.csv.gz"), str(tmp_path / "truth.csv"), str(output)],
        )
        runner.main()
        assert pd.read_csv(output)["replicate"].tolist() == [0, 2]
        assert pd.read_csv(tmp_path / "reference-results.markov.csv")["replicate"].tolist() == [
            0,
            2,
        ]
        for name in ("failures", "markov-failures"):
            records = pd.read_csv(tmp_path / f"reference-results.{name}.csv", dtype=str)
            assert list(records.columns) == list(runner.FAILURE_COLUMNS)
            assert records["replicates"].tolist() == ["1"]
