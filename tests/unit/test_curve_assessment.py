"""The assessment of a fitted effect curve (P1 plan 3.5): rows, fold locality and readings."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly._curve_assessment import STABILITY_WARNING, CurveDiagnostics
from cleverly.assessment import CURVE_OPERATIONS
from cleverly.bases import Linear, Modifiers
from cleverly.exceptions import CapabilityError

MODIFIERS = Modifiers(["W1", "W2"])


def frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w[:, 0])))
    y = rng.binomial(1, 0.3 + 0.1 * w[:, 1] + a * (0.15 + 0.1 * w[:, 0]))
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})


def curve(data: pd.DataFrame, scale: str = "difference", basis: Any = None) -> Any:
    study = CausalStudy(
        data, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    contrast = ConditionalContrast(
        MODIFIERS,
        scale=scale,  # type: ignore[arg-type]
        model=Linear(["W1", "W2"]) if basis is None else basis,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            study.identify(contrast)
            .estimate(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                n_folds=2,
                random_state=0,
            )
            .effect_curve()
        )


@pytest.fixture(scope="module")
def fitted() -> Any:
    return curve(frame())


def test_the_rows_are_the_curve_family(fitted: Any) -> None:
    rows = {row.operation: row for row in fitted.diagnostics.capabilities}
    assert set(CURVE_OPERATIONS) <= set(rows)
    assert all(rows[name].available for name in (*CURVE_OPERATIONS, "support"))
    assert not rows["refute"].available and rows["refute"].refusal.roadmap == "P1"
    assert rows["calibration"].execution == "refit" and rows["calibration"].accepts_random_state


def test_a_held_out_fold_never_sees_its_own_outcomes(fitted: Any) -> None:
    """Flip the outcomes of fold 0: its out-of-fold predictions do not move.

    Mutation control: predicting fold 0 from a refit on every row moves them.
    """
    diagnostics = fitted.diagnostics
    predictions, folds = diagnostics._out_of_fold(2, 0)
    fit = fitted.fit
    held = folds == 0
    flipped = fit.data
    from dataclasses import replace

    outcome = np.asarray(flipped.outcome, dtype=float).copy()
    outcome[held] = 1.0 - outcome[held]
    moved = replace(fitted, fit=_with_data(fit, replace(flipped, outcome=outcome)))
    again, _ = CurveDiagnostics(moved)._out_of_fold(2, 0)
    np.testing.assert_allclose(again[held], predictions[held], atol=1e-12)
    assert np.max(np.abs(again[~held] - predictions[~held])) > 1e-3
    whole = fit.estimator.refit(replace(flipped, outcome=outcome), random_state=0)
    frame_held = pd.DataFrame(np.asarray(flipped.covariates)[held], columns=flipped.covariate_names)
    leaked = whole.effect_curve().predict(frame_held)
    assert np.max(np.abs(leaked - predictions[held])) > 1e-3


class _Fit:
    """The producing fit, with its data replaced and everything else read through."""

    def __init__(self, fit: Any, data: Any) -> None:
        self._fit, self.data = fit, data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fit, name)


def _with_data(fit: Any, data: Any) -> Any:
    return _Fit(fit, data)


def test_the_calibration_is_the_least_squares_line_of_the_pseudo_outcome(fitted: Any) -> None:
    diagnostics = fitted.diagnostics
    report = diagnostics.calibration(n_folds=2, random_state=0)
    predictions, _ = diagnostics._out_of_fold(2, 0)
    _, _, d_t, d_r = diagnostics._pieces()[3]
    line = np.polyfit(predictions, d_t - d_r, 1)
    assert report.slope == pytest.approx(line[0], rel=1e-10)
    assert report.intercept == pytest.approx(line[1], abs=1e-12)
    table = report.to_frame()
    assert table["rows"].sum() == len(predictions) and len(table) == 5


def test_the_risk_comparison_names_its_comparators(fitted: Any) -> None:
    report = fitted.diagnostics.risk_comparison(n_folds=2, random_state=0)
    assert set(report.risks) == {"curve", "constant", "t_learner", "dr_learner"}
    assert report.omitted == ""
    ratio = curve(frame(), "log_ratio").diagnostics.risk_comparison(n_folds=2, random_state=0)
    assert set(ratio.risks) == {"curve", "constant", "t_learner"}
    assert "No DR-learner" in ratio.omitted


def test_stability_and_its_threshold(fitted: Any) -> None:
    report = fitted.diagnostics.stability(n_seeds=2, random_state=0)
    assert report.warning == (report.relative_sd > STABILITY_WARNING)
    assert report.alternative.startswith("none: a Linear basis")
    assert report.alternative_relative_sd is None


def test_modifier_stability_drops_each_modifier(fitted: Any) -> None:
    table = fitted.diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame()
    assert list(table["dropped"]) == ["W1", "W2"]
    one = curve(frame(), basis=Linear(["W1"]))
    with pytest.raises(CapabilityError, match="fewer than two modifiers"):
        one.diagnostics.modifier_stability()


def test_a_curve_without_its_fit_refuses(fitted: Any) -> None:
    from dataclasses import replace

    with pytest.raises(CapabilityError, match="keeps no producing fit"):
        replace(fitted, fit=None).diagnostics.calibration()


def test_every_reading_becomes_one_report_row(fitted: Any) -> None:
    """Each curve operation's report has an interpreter, and a warning reaches its status.

    The mutation control is the stability row: the same report with ``warning`` flipped
    must change status, so the row reads the flag rather than always completing.
    """
    from dataclasses import replace

    from cleverly.assessment import INTERPRETERS, AssessmentStatus

    diagnostics = fitted.diagnostics
    reports = {
        "calibration": diagnostics.calibration(n_folds=2, random_state=0),
        "risk_comparison": diagnostics.risk_comparison(n_folds=2, random_state=0),
        "stability": diagnostics.stability(n_seeds=2, random_state=0),
        "modifier_stability": diagnostics.modifier_stability(n_folds=2, random_state=0),
        "targeting_curve": diagnostics.targeting_curve(n_half_samples=20, random_state=0),
    }
    assert set(reports) == set(CURVE_OPERATIONS)
    for name, report in reports.items():
        item = INTERPRETERS[name](report, fitted, {})
        assert item.name == name and item.detail
    stability = reports["stability"]
    calm = INTERPRETERS["stability"](replace(stability, warning=False), fitted, {})
    loud = INTERPRETERS["stability"](replace(stability, warning=True), fitted, {})
    assert calm.status is AssessmentStatus.COMPLETED
    assert loud.status is AssessmentStatus.WARNING and loud.next_steps


def test_the_tables_come_back_in_the_fit_backend() -> None:
    """A polars fit's curve readings emit polars, as every other result table does."""
    pl = pytest.importorskip("polars")
    polars_curve = curve(pl.from_pandas(frame()))
    assert polars_curve.backend == "polars"
    diagnostics = polars_curve.diagnostics
    tables = (
        diagnostics.calibration(n_folds=2, random_state=0).to_frame(),
        diagnostics.risk_comparison(n_folds=2, random_state=0).to_frame(),
        diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame(),
    )
    assert all(isinstance(table, pl.DataFrame) for table in tables)
