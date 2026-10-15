"""Hazard targeting end to end: the study path and the compositions it ships with (X14).

One ``TimeToEvent`` study fits three arms, two causes, a known policy, weights and clusters
with ``n_folds=2`` and ``survival_model="hazard"``.  The other tests check the truncation
replay, the bootstrap diagnostic, the summary, the nuisance report and ``save``/``load`` on a
smaller hazard fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import cleverly
from cleverly import CausalStudy, RegimeMean, TimeToEvent
from cleverly.datasets import make_point_survival
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import longitudinal_truncation_curve

LEARNERS = {
    "outcome_learner": LogisticRegression(max_iter=1000),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
}
CAUSES = {1: "relapse", 2: "death"}


def _three_arm_policy(frame: Any) -> np.ndarray:
    """A known policy over three arms: arm 2 when ``W1 = 1``, else arms 0 and 1 evenly."""
    w1 = np.asarray(frame["W1"], dtype=float)
    return np.column_stack([0.5 * (1.0 - w1), 0.5 * (1.0 - w1), w1])


@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_the_study_path_with_three_arms_two_causes_a_policy_weights_and_clusters(
    targeting: str,
) -> None:
    frame, truth = make_point_survival(n=1500, seed=11, n_times=3, arms=3, causes=2)
    frame["w"] = np.where(frame["W1"] == 1, 1.5, 1.0)
    frame["id"] = np.arange(len(frame)) // 4
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=[1, 2, 3],
        causes=CAUSES,
        weights="w",
        cluster="id",
    )
    regimens = {
        "arm0": 0,
        "arm1": 1,
        "arm2": 2,
        "policy": Stochastic(_three_arm_policy, "q", density_kind="known"),
    }
    result = CausalStudy(frame, design=design).estimate(
        RegimeMean(regimens=regimens),
        n_folds=2,
        survival_model="hazard",
        targeting=targeting,
        random_state=0,
        **LEARNERS,
    )
    assert result.data.is_weighted and result.data.cluster is not None
    assert result.folds.n_folds == 2
    for cause in CAUSES.values():
        for label in regimens:
            curve = [result[f"cif_regimen[{label}, {cause} @ t={t}]"].psi for t in (1, 2, 3)]
            if targeting == "one_step":
                assert np.all(np.diff(curve) >= 0.0)
    for name in (
        "cif_regimen[arm2, death @ t=3]",
        "cif_regimen[arm1, relapse @ t=3]",
    ):
        estimate = result[name]
        assert np.isfinite(estimate.std_error)
        # The weighted fit targets the weighted law, so the unweighted truth is a sanity band.
        assert abs(estimate.psi - truth[name]) < 0.1


def _data(n: int = 800, seed: int = 3) -> LongitudinalData:
    frame, _ = make_point_survival(n=n, seed=seed, n_times=4)
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W1", "W2"], grid=[1, 2, 3, 4]
    )


@pytest.fixture(scope="module")
def one_step() -> Any:
    return LTMLE(
        {"arm1": 1, "arm0": 0},
        n_folds=1,
        survival_model="hazard",
        targeting="one_step",
        n_bootstrap=5,
        random_state=2,
        **LEARNERS,
    ).fit(_data())


def test_the_truncation_replay_reruns_the_hazard_targeting(one_step: Any) -> None:
    curve = longitudinal_truncation_curve(one_step, [(0.01, 1.0), (0.05, 1.0)])
    fitted = curve.set_index("is_fitted_bound").loc[True, "delta_from_fitted"]
    assert np.all(np.asarray(fitted) == 0.0)


def test_the_bootstrap_diagnostic_refits(one_step: Any) -> None:
    assert one_step.bootstrap is not None


def test_the_summary_reports_the_walk(one_step: Any) -> None:
    text = one_step.summary()
    assert "hazard targeting: one step over 8 parameter(s)" in text
    assert "converged" in text


def test_the_nuisance_report_has_hazard_rows(one_step: Any) -> None:
    report = one_step.diagnostics.nuisance_models()
    roles = {row.role for row in report.rows}
    assert "hazard" in roles


def test_save_and_load_keep_the_estimates(one_step: Any, tmp_path: Any) -> None:
    path = one_step.save(tmp_path / "hazard.pkl")
    loaded = cleverly.load(path)
    for name, estimate in one_step.estimates.items():
        assert loaded[name].psi == estimate.psi
        assert loaded[name].std_error == estimate.std_error
    assert loaded.initial_hazards is not None
    curve = longitudinal_truncation_curve(loaded, [(0.05, 1.0)])
    assert np.all(np.isfinite(curve["psi"]))
