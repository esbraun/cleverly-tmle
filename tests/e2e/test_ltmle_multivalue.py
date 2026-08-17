"""The categorical treatment path reaches every existing longitudinal report family."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM

from ..conftest import FAST_KWARGS

SPEC = {
    "inactive": "none",
    "active": "active",
    "step_up": ("medium", "active"),
}
SETTINGS = {
    **FAST_KWARGS,
    "outcome_learner": LogisticRegression(max_iter=500),
    "pseudo_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=500),
    "censoring_learner": LogisticRegression(max_iter=500),
    "n_folds": 2,
    "learner_folds": 2,
}
COLUMNS: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def categorical_treatments(frame: Any) -> Any:
    """Relabel both binary nodes and give a supported subset a third arm."""
    out = frame.copy()
    for name in ("A1", "A2"):
        observed = out[name].notna()
        out[name] = out[name].map({0.0: "none", 1.0: "active"})
        eligible = np.flatnonzero(observed.to_numpy())
        out.loc[out.index[eligible[::7]], name] = "medium"
    return out


def competing_frame(n: int = 700) -> Any:
    frame, _ = make_longitudinal_survival(n=n, seed=31)
    rng = np.random.default_rng(31)
    relapse = rng.integers(0, 2, len(frame)) == 0
    for time in (1, 2):
        event = frame[f"Y{time}"].to_numpy()
        frame[f"R{time}"] = np.where(np.isnan(event), np.nan, event * relapse)
        frame[f"D{time}"] = np.where(np.isnan(event), np.nan, event * ~relapse)
        frame = frame.drop(columns=[f"Y{time}"])
    return categorical_treatments(frame)


def test_end_outcome_with_censoring_and_categorical_labels() -> None:
    frame, _ = make_longitudinal(n=700, seed=30)
    result = LTMLE(SPEC, reference="inactive", **SETTINGS).fit(
        categorical_treatments(frame), outcome="Y", **COLUMNS
    )
    assert result.data.treatment_levels == (
        ("active", "medium", "none"),
        ("active", "medium", "none"),
    )
    assert set(result) == {
        "ey_regimen[inactive]",
        "ey_regimen[active]",
        "ey_regimen[step_up]",
        "ate_regimen[active vs inactive]",
        "ate_regimen[step_up vs inactive]",
    }
    assert result.converged


def test_a_saturated_msm_accepts_the_same_categorical_plans() -> None:
    frame, _ = make_longitudinal(n=700, seed=32)
    labels = tuple(SPEC)

    def design(label: Any, horizon: int, baseline: Any) -> np.ndarray:
        del horizon
        return np.eye(len(labels))[labels.index(label)] * np.ones((len(baseline), 1))

    result = LTMLE(
        SPEC,
        msm=MSM(design=design, terms=labels),
        **SETTINGS,
    ).fit(categorical_treatments(frame), outcome="Y", **COLUMNS)
    assert set(result) == {f"msm_regimen[{label}]" for label in labels}
    assert result.converged


def test_survival_curve_accepts_the_same_categorical_plans() -> None:
    frame, _ = make_longitudinal_survival(n=700, seed=33)
    result = LTMLE(SPEC, reference="inactive", **SETTINGS).fit(
        categorical_treatments(frame), outcome=["Y1", "Y2"], **COLUMNS
    )
    assert sum(name.startswith("risk_regimen[") for name in result) == 6
    assert sum(name.startswith("ate_regimen[") for name in result) == 4
    assert result.converged


def test_competing_risks_accept_the_same_categorical_plans() -> None:
    result = LTMLE(SPEC, reference="inactive", **SETTINGS).fit(
        competing_frame(),
        outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
        **COLUMNS,
    )
    assert sum(name.startswith("cif_regimen[") for name in result) == 12
    assert sum(name.startswith("ate_regimen[") for name in result) == 8
    assert result.config.causes == ("relapse", "death")
    assert result.converged
