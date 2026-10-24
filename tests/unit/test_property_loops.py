"""The property loops of the E3 modules and ``known_mechanism``, and every module's bound.

The loops run in process (``n_jobs=1``) on stubbed fits, so the tests read what each module
sends to :func:`~tests.studies.evidence.replication.fit_properties`: its cells, its row rules and
its errors.  RM39 part b2.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import clustered_few_cluster_properties as clustered
from tests.studies import few_cluster_crossfit_properties as crossfit
from tests.studies import known_mechanism_properties as known
from tests.studies.evidence import property_failures, replication
from tests.studies.evidence.registry import registered


class _Estimate:
    """The part of a ``ParameterEstimate`` the few-cluster rows read."""

    def __init__(self, psi: float, std_error: float, reference_df: int) -> None:
        self.psi = psi
        self.plugin_std_error = std_error
        self.plugin_interval = (psi - 2.0 * std_error, psi + 2.0 * std_error)
        self.reference_df = reference_df
        self.influence_curve = np.linspace(-1.0, 1.0, 40)

    def _plugin_pvalue(self) -> float:
        return 0.5


def _stub_crossfit(
    monkeypatch: pytest.MonkeyPatch, estimate_of: Any, *, truth: float = 0.1
) -> None:
    """Replace the draw by the replicate index and the fit by ``estimate_of(replicate, J)``."""
    monkeypatch.setattr(crossfit, "draw_cell", lambda sizes, clusters, replicate: (replicate, None))
    monkeypatch.setattr(
        crossfit,
        "fit_drawn",
        lambda drawn: (estimate_of(drawn[0]), truth),
    )


class TestFewClusterCrossfit:
    def test_a_non_finite_standard_error_is_an_invalid_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def estimate(replicate: int) -> _Estimate:
            # The reference df is checked per key below, so fix it to the J = 20 cells and
            # keep the stub to one cluster count.
            return _Estimate(0.1, float("nan") if replicate == 3 else 0.05, 18)

        _stub_crossfit(monkeypatch, estimate)
        monkeypatch.setattr(crossfit, "CLUSTER_COUNTS", (20,))
        with replication.recording("subject") as recorded:
            rows = crossfit.generate_property_rows(n_jobs=1, budget=5)
        failures = recorded.failures
        assert failures is not None
        assert set(failures["class"]) == {"invalid_result"}
        # One key per size law lost replicate 3, and each record names the three arms.
        assert len(failures) == len(crossfit.SIZE_LAWS)
        assert all(len(scenarios.split(";")) == 3 for scenarios in failures["scenarios"])
        assert 3 not in set(rows["replicate"][rows["cell"].str.contains("equal40")])
        assert set(rows["failed_replicates"]) == {0}

    def test_an_error_that_is_not_recordable_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def estimate(replicate: int) -> _Estimate:
            if replicate == 2:
                raise TypeError("a defect in the fit, not a degenerate sample")
            return _Estimate(0.1, 0.05, 18)

        _stub_crossfit(monkeypatch, estimate)
        monkeypatch.setattr(crossfit, "CLUSTER_COUNTS", (20,))
        with replication.recording("subject"), pytest.raises(TypeError, match="defect"):
            crossfit.generate_property_rows(n_jobs=1, budget=5)

    def test_the_reference_df_assertion_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_crossfit(monkeypatch, lambda replicate: _Estimate(0.1, 0.05, 17))
        monkeypatch.setattr(crossfit, "CLUSTER_COUNTS", (20,))
        with replication.recording("subject"), pytest.raises(AssertionError, match="reference_df"):
            crossfit.generate_property_rows(n_jobs=1, budget=3)


class TestClusteredFewCluster:
    def test_one_failed_fit_costs_only_that_fits_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lost = "tmle_crossfit"

        def fit(name: str, frame: int, clusters: int) -> _Estimate:
            if name == lost and frame == 3:
                raise np.linalg.LinAlgError("singular")
            return _Estimate(0.1, 0.05, clustered.expected_reference_df(name, clusters))

        monkeypatch.setattr(clustered, "draw", lambda sizes, clusters, seed: seed % 1000)
        monkeypatch.setattr(clustered, "_seed", lambda sizes, clusters, replicate: replicate)
        monkeypatch.setattr(clustered, "fit_estimate", fit)
        monkeypatch.setattr(clustered, "CLUSTER_COUNTS", (10,))
        monkeypatch.setattr(clustered, "SIZE_LAWS", ("equal10",))
        with replication.recording("subject") as recorded:
            rows = clustered.generate_property_rows(n_jobs=1, budget=5)
        failures = recorded.failures
        assert failures is not None and len(failures) == 1
        record = failures.iloc[0]
        assert record["class"] == "fitting_failure"
        assert record["scenarios"].split(";") == [
            f"{clustered.FAMILY}/{clustered.cell_name(lost, 'equal10', 10, arm)}"
            for arm, _ in clustered.arms(lost)
        ]
        # The other fits of the same draw published their rows.
        other = [name for name in clustered.fits_at(10) if name != lost]
        for name in other:
            cell = clustered.cell_name(name, "equal10", 10, "t_reference")
            assert set(rows.loc[rows["cell"] == cell, "replicate"]) == set(range(5))
        cell = clustered.cell_name(lost, "equal10", 10, "t_reference")
        assert set(rows.loc[rows["cell"] == cell, "replicate"]) == {0, 1, 2, 4}

    def test_the_fold_evaluated_fit_names_its_fourth_arm(self) -> None:
        key = ("equal10", 10, 0, "tmle_cv_evaluation", 5)
        names = clustered._cells_of(key)
        assert len(names) == 4 and names[-1].endswith("t_j_minus_2_reference")
        assert len(clustered._cells_of(("equal10", 10, 0, "tmle_crossfit", 5))) == 3


class TestKnownMechanismBootstrap:
    @staticmethod
    def _stub(monkeypatch: pytest.MonkeyPatch, lost: int) -> None:
        """A bootstrap that loses ``lost`` draws on replicate 4 and none elsewhere."""
        estimate = SimpleNamespace(psi=0.1)
        summary = SimpleNamespace(ci=(0.0, 0.2), std_error=0.05)

        class _Result(SimpleNamespace):
            def __getitem__(self, name: str) -> Any:
                return estimate

        monkeypatch.setattr(known.law, "sample", lambda n, seed: seed % 7)
        monkeypatch.setattr(known, "_seed", lambda family, stream, replicate: replicate)
        monkeypatch.setattr(
            known,
            "fit_cleverly",
            lambda frame, **settings: _Result(
                bootstrap=SimpleNamespace(
                    n_failed=lost if frame == 4 else 0, interval=lambda name, alpha: summary
                )
            ),
        )

    def _run(self) -> list[Any]:
        return replication.fit_properties(
            [(("bootstrap", r),) for r in range(6)],
            replication.splat_drawn(known._dispatch),
            draw=replication.splat_draw(known._dispatch_draw),
            cells_of=known._cells_of,
            cell_of=known._cell_of,
            replicate_of=known._replicate_of,
            n_jobs=1,
            extra=known._lost_draws,
            derived=known.DERIVED_CELLS,
            checkpoint=None,
        )

    def test_a_lost_draw_is_an_invalid_result_and_the_row_carries_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(monkeypatch, 3)
        with replication.recording("subject") as recorded:
            values = self._run()
        failures = recorded.failures
        assert failures is not None and len(failures) == 1
        record = failures.iloc[0]
        assert (record["class"], record["message"]) == (
            "invalid_result",
            "the bootstrap lost 3 draws",
        )
        assert len(values) == 5
        rows = pd.DataFrame([row for value in values for row in value])
        assert set(rows["failed_replicates"]) == {0}
        assert set(rows[known.BOOTSTRAP_LOST]) == {0}

    def test_a_complete_bootstrap_publishes_every_replicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(monkeypatch, 0)
        with replication.recording("subject") as recorded:
            values = self._run()
        assert recorded.failures is None and len(values) == 6

    def test_the_rule_is_silent_on_rows_without_the_column(self) -> None:
        assert known._lost_draws(pd.DataFrame({"estimate": [0.0]})) is None


def test_every_registered_study_declares_a_failure_bound_inside_its_range() -> None:
    for study in registered():
        module = study.properties()
        assert hasattr(module, "FAILURE_BOUND"), f"{study.slug} declares no FAILURE_BOUND"
        assert 0.0 <= float(module.FAILURE_BOUND) <= property_failures.MAXIMUM_BOUND, study.slug


#: The property loops that pass no ``draw=``, with the reason each one needs none.  Every other
#: ``fit_properties`` call in ``tests/studies`` draws its sample outside the classifier (F5).
UNSPLIT_LOOPS = {
    ("repeated_crossfit_properties", "_stability_trial"): (
        "the trial's one sample is drawn before the loop; a key varies only the fold seed"
    ),
}


def test_every_property_loop_but_the_declared_ones_draws_outside_the_fit() -> None:
    studies = Path(__file__).resolve().parents[1] / "studies"
    unsplit: set[tuple[str, str]] = set()
    calls = 0
    for path in sorted(studies.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name != "fit_properties" or path.stem == "replication":
                continue
            calls += 1
            if not any(keyword.arg == "draw" for keyword in node.keywords):
                worker = node.args[1]
                inner = worker.args[0] if isinstance(worker, ast.Call) and worker.args else worker
                unsplit.add((path.stem, ast.unparse(inner)))
    assert calls > 50
    assert unsplit == set(UNSPLIT_LOOPS)
