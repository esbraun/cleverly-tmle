"""The categorical treatment path reaches every existing longitudinal report family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen
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


def test_the_diagnostics_report_shares_per_label_not_a_binary_share() -> None:
    """``share_assigned_1`` cannot answer this question, so the frame asks a different one.

    On three arms "the fraction assigned arm 1" is the fraction assigned whichever label
    sorts second -- here ``medium`` -- so a static plan on ``none`` and a static plan on
    ``active`` would both report ``0.0`` and be indistinguishable, which is the opposite
    of what the column exists for. Every level appears, including the ones a plan never
    assigns, so the shares in a row sum to one.
    """
    frame, _ = make_longitudinal(n=700, seed=30)
    result = LTMLE(SPEC, reference="inactive", **SETTINGS).fit(
        categorical_treatments(frame), outcome="Y", **COLUMNS
    )
    diagnostics = result.diagnostics.support().to_frame()
    assert "assigned_shares" in diagnostics.columns
    assert "share_assigned_1" not in diagnostics.columns

    keys = zip(diagnostics["regimen"], diagnostics["time"], strict=True)
    shares = dict(zip(keys, diagnostics["assigned_shares"], strict=True))
    assert shares[("inactive", 1)] == "active=0, medium=0, none=1"
    assert shares[("active", 1)] == "active=1, medium=0, none=0"
    assert shares[("step_up", 1)] == "active=0, medium=1, none=0"
    assert shares[("step_up", 2)] == "active=1, medium=0, none=0"


def test_a_binary_panel_keeps_the_share_it_always_reported() -> None:
    """The other side of the switch: nothing changes for a wholly two-level fit."""
    frame, _ = make_longitudinal(n=700, seed=30)
    result = LTMLE({"never": 0, "always": 1}, reference="never", **SETTINGS).fit(
        frame, outcome="Y", **COLUMNS
    )
    diagnostics = result.diagnostics.support().to_frame()
    assert "share_assigned_1" in diagnostics.columns
    assert "assigned_shares" not in diagnostics.columns
    assert set(diagnostics["share_assigned_1"]) == {0.0, 1.0}


def test_a_saturated_msm_accepts_the_same_categorical_plans() -> None:
    frame, _ = make_longitudinal(n=700, seed=32)
    labels = tuple(SPEC)

    def design(label: Any, horizon: int, baseline: Any) -> np.ndarray:
        del horizon
        return np.eye(len(labels))[labels.index(label)] * np.ones((len(baseline), 1))

    result = LTMLE(
        SPEC,
        msm=MSM(design=design, terms=labels, design_kind="known"),
        **{**SETTINGS, "n_folds": 1},
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


# ------------------------------------------------------------------ known policies


def _policy_regimen() -> Any:
    """Draw ``active`` at a quarter and ``none`` at a half at both nodes, ``medium`` the rest."""

    def draw(history: Any) -> Any:
        return pd.DataFrame(
            np.tile([0.25, 0.25, 0.5], (len(history), 1)), columns=["active", "medium", "none"]
        )

    node = Stochastic(draw, "draw", density_kind="known")
    return DynamicRegimen("mix", (node, node))


def test_a_clustered_policy_fit_reports_at_the_cluster_in_sample_and_cross_fitted() -> None:
    """T11: the cluster-summed curve and the cluster floors are target-agnostic."""
    frame, _ = make_longitudinal(n=1200, seed=41, cluster_size=20)
    frame = categorical_treatments(frame)
    for folds in (1, 2):
        result = LTMLE(
            {"inactive": "none", "mix": _policy_regimen()},
            **{**SETTINGS, "n_folds": folds, "simultaneous": False},
        ).fit(frame, outcome="Y", id="id", **COLUMNS)
        estimate = result["ate_regimen[mix vs inactive]"]
        assert estimate.n_clusters == 60
        clusters = result.data.cluster
        summed = np.bincount(clusters, weights=estimate.influence_curve)
        # The cluster-robust variance carries the J / (J - 1) small-sample factor.
        expected = np.sqrt(np.sum(summed**2) * 60 / 59) / result.data.n
        assert estimate.std_error == pytest.approx(float(expected), rel=1e-12)


def test_a_policy_fit_below_the_cluster_floor_reports_no_interval() -> None:
    frame, _ = make_longitudinal(n=320, seed=42, cluster_size=40)
    frame = categorical_treatments(frame)
    result = LTMLE(
        {"inactive": "none", "mix": _policy_regimen()},
        **{**SETTINGS, "n_folds": 1, "simultaneous": False},
    ).fit(frame, outcome="Y", id="id", **COLUMNS)
    estimate = result["ey_regimen[mix]"]
    assert estimate.n_clusters == 8
    assert estimate.inference == "few_cluster_plugin"


def test_a_policy_closing_over_a_lambda_saves_as_a_rule_lambda_does(tmp_path: Any) -> None:
    """Q12: the same refusal a rule written as a lambda meets."""

    frame, _ = make_longitudinal(n=300, seed=43)
    frame = categorical_treatments(frame)
    plans = {
        "rule": DynamicRegimen(
            "rule", ("none", lambda h: np.where(h["L2"] > 0, "active", "none")), rule_kind="known"
        ),
        "policy": DynamicRegimen(
            "policy",
            (
                "none",
                Stochastic(lambda h: np.full((len(h), 3), 1.0 / 3.0), "u", density_kind="known"),
            ),
        ),
    }
    messages = []
    for label, plan in plans.items():
        result = LTMLE({label: plan}, **{**SETTINGS, "n_folds": 1, "simultaneous": False}).fit(
            frame, outcome="Y", **COLUMNS
        )
        try:
            result.save(tmp_path / f"{label}.joblib")
        except TypeError as error:
            messages.append(str(error))
    assert len(messages) == 2 and messages[0] == messages[1]


def test_a_causal_study_states_positivity_where_the_policy_puts_mass() -> None:
    """T17: the study path passes a policy regimen through and words its identification."""
    frame, _ = make_longitudinal(n=600, seed=44)
    frame = categorical_treatments(frame)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    )
    effect = study.identify(
        RegimeContrast({"inactive": "none", "mix": _policy_regimen()}, reference="inactive")
    )
    assert any(
        "wherever a regimen or its known policy puts mass" in item
        for item in effect.identification.assumptions
    )
    result = effect.estimate(
        **{**SETTINGS, "n_folds": 1, "simultaneous": False, "cross_fit": False}
    )
    engine = LTMLE(
        {"inactive": "none", "mix": _policy_regimen()},
        reference="inactive",
        **{**SETTINGS, "n_folds": 1, "simultaneous": False},
    ).fit(frame, outcome="Y", **COLUMNS)
    assert result.psi("ate_regimen[mix vs inactive]") == engine.psi("ate_regimen[mix vs inactive]")
