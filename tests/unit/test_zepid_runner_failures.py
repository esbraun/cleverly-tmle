"""The zEpid runner records a failed fit as one lost replicate (RM39b).

The runner's image installs no ``cleverly``, so it carries a copy of the classifier.  The parity
tests pin the copy to ``cleverly.validation._failures``.  The behavior tests run the runner
against a stub ``zepid`` package whose fit raises on chosen replicates.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.validation import _failures
from tests.canonical.zepid_cvtmle import run_zepid_cvtmle as runner
from tests.studies.evidence.replication import invalid_rows

_STUB = """
import numpy as np

from zepid.causal.doublyrobust import crossfit

PLAN = {}


class SingleCrossfitTMLE:
    def __init__(self, frame, exposure, outcome, alpha):
        self.frame = frame

    def exposure_model(self, covariates, estimator, bound):
        pass

    def outcome_model(self, formula, estimator):
        pass

    def fit(self, n_splits, n_partitions, method, random_state):
        replicate = int(self.frame["replicate"].iloc[0])
        effect = PLAN.get(replicate)
        if effect == "singular":
            np.linalg.inv(np.zeros((2, 2)))
        if effect == "zepid value":
            raise ValueError("positivity violated")
        if effect == "type":
            raise TypeError("a bug")
        y = self.frame["Y"].to_numpy(float)
        crossfit.targeting_step(None, None, y, y * 0.0)
        self.risk_difference = float(y.mean())
        self.risk_difference_se = 0.0 if effect == "zero se" else 0.1
        self.risk_difference_ci = (self.risk_difference - 0.2, self.risk_difference + 0.2)
"""


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    """A stub ``zepid`` whose fit does what the returned plan says for each replicate."""
    names = ("zepid", "zepid.causal", "zepid.causal.doublyrobust")
    modules = {name: types.ModuleType(name) for name in names}
    crossfit = types.ModuleType("zepid.causal.doublyrobust.crossfit")
    crossfit._sample_split_ = lambda data, n_splits, random_state: []  # type: ignore[attr-defined]
    crossfit.targeting_step = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    modules["zepid.causal.doublyrobust.crossfit"] = crossfit
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    modules["zepid.causal.doublyrobust"].crossfit = crossfit  # type: ignore[attr-defined]
    # The fit's frames must read as zEpid's own, as a real zEpid raise would.
    fit = types.ModuleType("zepid.causal.doublyrobust._stub")
    exec(compile(_STUB, "zepid/causal/doublyrobust/_stub.py", "exec"), fit.__dict__)
    modules["zepid.causal.doublyrobust"].SingleCrossfitTMLE = fit.SingleCrossfitTMLE  # type: ignore[attr-defined]
    return fit.PLAN  # type: ignore[no-any-return]


def _groups(scenarios: dict[str, int]) -> list[tuple[pd.DataFrame, float]]:
    out = []
    for scenario, replicates in scenarios.items():
        for replicate in range(replicates):
            y = np.random.default_rng(replicate).integers(0, 2, size=20).astype(float)
            frame = pd.DataFrame(
                {
                    "scenario": scenario,
                    "replicate": replicate,
                    "partition_random_state": 1,
                    "W1": 0.0,
                    "A": 1.0,
                    "Y": y,
                }
            )
            out.append((frame, float(y.mean())))
    return out


class TestParity:
    def test_the_lists_are_cleverlys_without_its_own_classes(self) -> None:
        def foreign(classes: tuple[type[BaseException], ...]) -> tuple[type[BaseException], ...]:
            return tuple(c for c in classes if not c.__module__.startswith("cleverly"))

        assert foreign(_failures.FATAL) == runner.FATAL
        assert foreign(_failures.RECORDED) == runner.RECORDED
        assert (*_failures.LIBRARY_MODULES, "zepid") == runner.LIBRARY_MODULES

    @pytest.mark.parametrize(
        "change",
        [
            {},
            {"estimate": float("nan")},
            {"std_error": 0.0},
            {"ci_lower": 9.0},
            {"covered": 0},
        ],
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


class TestBehavior:
    def test_a_recorded_failure_costs_its_replicate(self, plan: dict[int, str]) -> None:
        plan.update({1: "singular", 3: "zepid value", 4: "zero se"})
        rows, failures = runner.fit_all(_groups({"a": 6}), cores=1)
        assert [row["replicate"] for row in rows] == [0, 2, 5]
        assert [(f["group_id"], f["class"], f["replicates"]) for f in failures] == [
            ("g000000002", "fitting_failure", "1"),
            ("g000000004", "fitting_failure", "3"),
            ("g000000005", "invalid_result", "4"),
        ]
        assert failures[1]["call"] == "zepid.causal.doublyrobust._stub:fit"
        assert failures[1]["message"] == "ValueError: positivity violated"

    def test_a_bug_and_a_foreign_value_error_stop_the_run(self, plan: dict[int, str]) -> None:
        plan[2] = "type"
        with pytest.raises(TypeError):
            runner.fit_all(_groups({"a": 4}), cores=1)

        def own(payload: Any) -> dict[str, Any]:
            raise ValueError("the runner's own check")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner, "_fit_one", own)
            with pytest.raises(ValueError, match="own check"):
                runner.fit_all(_groups({"a": 2}), cores=1)

    def test_the_systematic_and_every_failed_stops(self, plan: dict[int, str]) -> None:
        plan.update(dict.fromkeys(range(20), "singular"))
        with pytest.raises(RuntimeError, match="the first 20 groups of a failed"):
            runner.fit_all(_groups({"a": 25}), cores=1)
        # Nineteen of 20 is not systematic.
        plan.pop(19)
        rows, failures = runner.fit_all(_groups({"a": 25}), cores=1)
        assert (len(rows), len(failures)) == (6, 19)
        with pytest.raises(RuntimeError, match="every group failed"):
            runner.fit_all(_groups({"a": 3}), cores=1)

    def test_main_writes_the_records_beside_the_rows(
        self, plan: dict[int, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan[1] = "singular"
        groups = _groups({"a": 3})
        samples = pd.concat([frame for frame, _ in groups], ignore_index=True)
        truths = pd.DataFrame(
            {"scenario": "a", "replicate": range(3), "truth_ate": [t for _, t in groups]}
        )
        samples.to_csv(tmp_path / "samples.csv.gz", index=False)
        truths.to_csv(tmp_path / "truth.csv", index=False)
        output = tmp_path / "reference-results.csv"
        monkeypatch.setenv("CLEVERLY_REFERENCE_CORES", "1")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run", str(tmp_path / "samples.csv.gz"), str(tmp_path / "truth.csv"), str(output)],
        )
        runner.main()
        assert pd.read_csv(output)["replicate"].tolist() == [0, 2]
        records = pd.read_csv(tmp_path / "reference-results.failures.csv", dtype=str)
        assert list(records.columns) == list(runner.FAILURE_COLUMNS)
        assert records[["group_id", "replicates"]].values.tolist() == [["g000000002", "1"]]
