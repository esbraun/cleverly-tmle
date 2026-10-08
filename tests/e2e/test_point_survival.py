"""Point-treatment survival end to end: the study path and the shipped compositions.

Each test fits ``make_point_survival`` with explicit, fast learners and checks a statement
the held design makes about a composition the longitudinal engine already ships: several
arms, competing causes, weights, clusters, cross-fitting, a working model, the truncation
replay, the bootstrap diagnostic, the nuisance report and the summary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast, TimeToEvent
from cleverly.datasets import make_point_survival
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import bootstrap_design_kind, longitudinal_truncation_curve
from cleverly.msm import MSM
from cleverly.validation.longitudinal import (
    LONGITUDINAL_HELD_DECISION,
    LONGITUDINAL_NO_CENSORING,
)

LEARNERS = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
}


def _long(n: int = 1500, seed: int = 3, **kwargs: object) -> LongitudinalData:
    frame, _ = make_point_survival(n=n, seed=seed, n_times=4, **kwargs)  # type: ignore[arg-type]
    causes = {1: "relapse", 2: "death"} if kwargs.get("causes") == 2 else None
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=[1, 2, 3, 4],
        causes=causes,
    )


def test_the_study_path_reports_the_contrast_and_the_omissions() -> None:
    frame, truth = make_point_survival(n=2000, seed=5, n_times=4)
    design = TimeToEvent(
        time="time", event="event", treatment="A", baseline=["W1", "W2"], grid=[1, 2, 3, 4]
    )
    result = CausalStudy(frame, design=design).estimate(
        RegimeContrast(regimens={"arm1": 1, "arm0": 0}, reference="arm0"),
        cross_fit=False,
        **LEARNERS,
    )
    name = "ate_regimen[arm1 vs arm0 @ t=4]"
    estimate = result[name]
    low, high = estimate.ci
    assert low - 4 * estimate.std_error < truth[name] < high + 4 * estimate.std_error
    omissions = result.diagnostics.nuisance_models().omissions
    reasons = {(item.role, item.time, item.reason) for item in omissions}
    assert ("censoring", 1, LONGITUDINAL_NO_CENSORING) in reasons
    for node in (2, 3, 4):
        assert ("treatment", node, LONGITUDINAL_HELD_DECISION) in reasons
    text = result.summary()
    assert "one baseline decision 'A', held over 4 node(s)" in text
    assert "time grid: 1, 2, 3, 4" in text


def test_three_arms_and_two_causes() -> None:
    frame, truth = make_point_survival(
        n=3000, seed=8, n_times=3, arms=3, causes=2, layout="wide", time_varying=True
    )
    design = LongitudinalTreatment(
        outcome={"relapse": ["R1", "R2", "R3"], "death": ["D1", "D2", "D3"]},
        treatment="A",
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
    )
    result = CausalStudy(frame, design=design).estimate(
        RegimeContrast(regimens={"arm0": 0, "arm1": 1, "arm2": 2}, reference="arm0"),
        cross_fit=False,
        **LEARNERS,
    )
    for name in (
        "ate_regimen[arm2 vs arm0, death @ t=3]",
        "ate_regimen[arm1 vs arm0, relapse @ t=3]",
    ):
        estimate = result[name]
        assert abs(estimate.psi - truth[name]) < 5 * estimate.std_error


def test_weights_and_clusters_and_folds_compose() -> None:
    frame, _ = make_point_survival(n=1200, seed=4, n_times=3)
    frame["w"] = np.where(frame["W1"] == 1, 2.0, 1.0)
    frame["id"] = np.arange(len(frame)) // 20
    data = LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=[1, 2, 3],
        weights="w",
        id="id",
    )
    result = LTMLE({"arm1": 1, "arm0": 0}, n_folds=2, random_state=1, **LEARNERS).fit(data)
    assert result.data.is_weighted and result.data.cluster is not None
    assert all(np.isfinite(estimate.psi) for estimate in result.estimates.values())
    assert result.folds.n_folds == 2


def _treated_by_horizon(label: object, horizon: int, frame: object) -> np.ndarray:
    n = len(frame)  # type: ignore[arg-type]
    return np.column_stack(
        [np.ones(n), np.full(n, float(label == "arm1")), np.full(n, float(horizon))]
    )


def test_a_working_model_over_arms_and_horizons() -> None:
    msm = MSM(
        design=_treated_by_horizon, terms=("(intercept)", "treated", "t"), design_kind="known"
    )
    result = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, msm=msm, **LEARNERS).fit(_long())
    assert len(result.estimates) == 3
    assert all(np.isfinite(estimate.psi) for estimate in result.estimates.values())


def test_the_truncation_replay_and_the_bootstrap_diagnostic() -> None:
    data = _long(n=800)
    fit = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, n_bootstrap=5, random_state=2, **LEARNERS).fit(
        data
    )
    assert bootstrap_design_kind(fit.data, fit.folds, fit.config.regimens, None) is None
    assert fit.bootstrap is not None
    curve = longitudinal_truncation_curve(fit, [(0.01, 1.0), (0.05, 1.0)])
    fitted = curve.set_index("is_fitted_bound").loc[True, "delta_from_fitted"]
    assert np.all(np.asarray(fitted) == 0.0)


def test_the_wide_held_and_long_paths_agree() -> None:
    frame, _ = make_point_survival(n=1500, seed=9, n_times=3)
    long = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, **LEARNERS).fit(
        LongitudinalData.from_time_to_event(
            frame,
            time="time",
            event="event",
            treatment="A",
            baseline=["W1", "W2"],
            grid=[1, 2, 3],
        )
    )
    wide_frame, _ = make_point_survival(n=1500, seed=9, n_times=3, layout="wide")
    wide = LTMLE({"arm1": 1, "arm0": 0}, n_folds=1, **LEARNERS).fit(
        wide_frame,
        outcome=["Y1", "Y2", "Y3"],
        treatment="A",
        baseline=["W1", "W2"],
        censoring=["C1", "C2", "C3"],
    )
    for name in long.estimates:
        assert long[name].psi == pytest.approx(wide[name].psi, rel=1e-10)
