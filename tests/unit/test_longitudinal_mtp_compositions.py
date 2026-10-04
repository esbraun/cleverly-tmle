"""Longitudinal modified treatment policies with censoring, events, weights and clusters.

T7: the cumulative risk and a competing cause's incidence under a binary-node plan of
modified treatment policies, at both horizons, with censoring at both nodes, against the
longhand law; dropping the censoring factor must fail.  T8: integer observation weights
equal the expanded sample.  T9: a clustered fit, in sample and with grouped folds.  T12:
each longitudinal refusal of a policy, before any learner.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import ModifiedPolicy, Piece, Shift, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, sequential
from cleverly.utils.bounds import bound

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import discrete_law_longitudinal_policy as policy
from .. import discrete_law_survival as survival
from .. import longitudinal_mtp as mtp

CAUGHT = 1e-4
NO_TRUNCATION = (1e-8, 1.0)

SURVIVAL_COLUMNS: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "censoring": ["C1", "C2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}


SURVIVAL_PLAN = mtp.SURVIVAL_PLAN


def _survival_fit(module: Any, outcome: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"never": 0, "mtp": SURVIVAL_PLAN},
            outcome_learner=module.CellMeans(),
            pseudo_learner=module.CellMeans(),
            treatment_learner=module.CellMeans(),
            censoring_learner=module.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ).fit(module.frame(), **{**SURVIVAL_COLUMNS, "outcome": outcome})


def _survival_deviation() -> float:
    result = _survival_fit(survival, ["Y1", "Y2"])
    rows = survival.first_row_of()
    worst = 0.0
    for horizon in survival.HORIZONS:
        name = f"risk_regimen[mtp @ t={horizon}]"
        function = partial(law.functional_mtp_survival, horizon=horizon)
        worst = max(worst, abs(result.psi(name) - function(survival.PROBS)))
        reported = result.influence_curves[name][rows]
        worst = max(worst, float(np.max(np.abs(reported - policy.eif(function, survival.PROBS)))))
    return worst


def test_t7_a_survival_plan_of_policies_is_the_cumulative_risk() -> None:
    assert _survival_deviation() < 1e-10
    never = survival.TRUTH["risk_regimen[never @ t=2]"]
    assert abs(law.functional_mtp_survival(survival.PROBS, 2) - never) > 0.01


def test_t7_dropping_the_censoring_factor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def without_censoring(self: Any, data: Any, plan: Any, bounds: Any) -> Any:
        lower, upper = bounds
        running = np.ones(data.n)
        raw, bounded = [], []
        for time in range(1, data.n_times + 1):
            running = running * self.treatment[time - 1][plan.label]
            raw.append(running)
            bounded.append(bound(running, lower, upper))
        return np.column_stack(raw), np.column_stack(bounded)

    monkeypatch.setattr(sequential.Mechanism, "cumulative_with_unbounded", without_censoring)
    assert _survival_deviation() > CAUGHT


@pytest.mark.parametrize("cause", competing.CAUSES)
def test_t7_a_competing_cause_under_a_plan_of_policies(cause: str) -> None:
    result = _survival_fit(competing, competing.outcome_columns())
    rows = competing.first_row_of()
    for horizon in competing.HORIZONS:
        name = f"cif_regimen[mtp, {cause} @ t={horizon}]"
        function = partial(law.functional_mtp_survival, horizon=horizon, cause=cause)
        assert result.psi(name) == pytest.approx(function(competing.PROBS), abs=1e-12)
        np.testing.assert_allclose(
            result.influence_curves[name][rows],
            policy.eif(function, competing.PROBS),
            atol=1e-10,
            rtol=0,
        )


# ------------------------------------------------------------------ T8 weights


def _settings(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
        **overrides,
    }


CONTINUOUS_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}


def test_t8_integer_weights_equal_the_expanded_sample() -> None:
    """A weight of two is two copies of the row: estimate and summed curve agree."""
    frame = law.sample(600, 3)
    weights = 1.0 + (np.arange(len(frame)) % 3 == 0)
    weighted = frame.assign(w=weights)
    expanded = pd.concat([frame, frame[weights == 2.0]], ignore_index=True)
    labels = ("natural", "up", "random")
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        by_weight = LTMLE(mtp.regimens(labels), **_settings()).fit(
            weighted, weights="w", **CONTINUOUS_COLUMNS
        )
        by_copy = LTMLE(mtp.regimens(labels), **_settings()).fit(expanded, **CONTINUOUS_COLUMNS)
    for label in labels:
        name = f"ey_regimen[{label}]"
        assert by_weight.psi(name) == pytest.approx(by_copy.psi(name), abs=1e-10)


# ------------------------------------------------------------------ T9 clusters


@pytest.mark.parametrize("folds", [1, 3])
def test_t9_a_clustered_fit_runs_with_whole_clusters(folds: int) -> None:
    frame = law.sample(900, 4).assign(cluster=lambda f: np.arange(len(f)) // 3)
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        result = LTMLE(
            mtp.regimens(("natural", "up")), **_settings(n_folds=folds, random_state=1)
        ).fit(frame, id="cluster", **CONTINUOUS_COLUMNS)
    assert np.isfinite(result.psi("ate_regimen[up vs natural]"))
    if folds > 1:
        for train, test in result.folds:
            assert not set(frame["cluster"].to_numpy()[train]) & set(
                frame["cluster"].to_numpy()[test]
            )


# ------------------------------------------------------------------ T12 refusals


class NeverFit:
    """A learner that raises if fitted, the witness that a refusal precedes every learner."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> NeverFit:
        return self

    def fit(self, *args: Any, **kwargs: Any) -> NeverFit:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, x: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")

    def predict_proba(self, x: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")


def _refused(regimens: Any, frame: Any = None, **columns: Any) -> Any:
    settings = {
        "outcome_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "n_folds": 1,
        "simultaneous": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(regimens, **settings).fit(
            law.frame() if frame is None else frame, **{**CONTINUOUS_COLUMNS, **columns}
        )


def test_r10_a_label_at_a_continuous_node() -> None:
    with pytest.raises(CapabilityError, match="sets the continuous node 'A1' to a fixed dose"):
        _refused({"static": 1.0})


def test_r10_a_known_density_at_a_continuous_node() -> None:
    density = Stochastic(lambda h: np.ones((len(h), 2)) / 2, "q", density_kind="known")
    with pytest.raises(CapabilityError, match="not pathwise differentiable"):
        _refused({"q": DynamicRegimen("q", (density, Shift(0.0, cap=None)))})


def test_r3_a_threshold_policy_at_a_continuous_node() -> None:
    floor = ModifiedPolicy(
        "floor",
        pieces=(
            Piece(
                -np.inf, np.inf, lambda a, h: np.minimum(a, 1.0), lambda b, h: b, lambda b, h: 1.0
            ),
        ),
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match="not pathwise differentiable"):
        _refused({"floor": floor})


def test_r11_a_policy_off_the_levels_of_a_categorical_node() -> None:
    columns = {"continuous_treatment": ()}
    with pytest.raises(DataError, match="that are not levels of the node"):
        _refused({"up": Shift(1.0, cap=None)}, **columns)


def test_r5_an_undeclared_policy() -> None:
    undeclared = ModifiedPolicy("u", apply=lambda a, h: a)
    with pytest.raises(CapabilityError, match="inspect a closure"):
        _refused({"u": undeclared}, continuous_treatment=())


def test_r13_a_vector_node_with_a_continuous_component() -> None:
    frame = law.frame(vector=True).assign(A1a=lambda f: f["A1a"] + 0.25)
    with pytest.raises(DataError, match="vector with continuous components"):
        _refused(
            {"natural": Shift(0.0, cap=None)},
            frame=frame,
            treatment=[["A1a", "A1b"], ["A2a", "A2b"]],
            continuous_treatment=["A1a"],
        )


def test_the_point_working_model_refusal_names_the_longitudinal_route() -> None:
    from cleverly.estimators import TMLE
    from cleverly.msm import MSM

    model = MSM(
        design=lambda a, v: np.column_stack([np.ones(len(a)), a]),
        terms=("intercept", "a"),
        design_kind="known",
    )
    with pytest.raises(CapabilityError, match=r"LTMLE\(regimens=\{\.\.\.\}, msm=MSM\(\.\.\.\)\)"):
        TMLE(msm=model, policies=[Shift(0.0, cap=None)])


# ------------------------------------------------------------------ T10 truncation replay, diagnostics


def _sample_fit(reading: str, labels: tuple[str, ...] = ("natural", "up")) -> Any:
    from sklearn.dummy import DummyRegressor

    from ..studies.canonical_ltmle import QuasiBinomialGLM

    columns = dict(CONTINUOUS_COLUMNS)
    if reading == "categorical":
        columns["continuous_treatment"] = ()
    settings = _settings(
        g_bounds=(0.01, 1.0), outcome_learner=QuasiBinomialGLM(), pseudo_learner=DummyRegressor()
    )
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return LTMLE(mtp.regimens(labels), **settings).fit(law.sample(800, 9), **columns)


def test_t10_the_replay_reproduces_the_fit_and_the_policy_ratio_has_no_bound() -> None:
    """At a continuous node the bound reaches no factor: the ratio is the numerator."""
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    result = _sample_fit("continuous")
    with mtp.exact_bins():
        curve = longitudinal_truncation_curve(
            result, [(0.01, 1.0), (0.2, 1.0)], estimands=["ey_regimen[up]"]
        )
    curve = curve.set_index("is_fitted_bound")
    assert curve.loc[True, "delta_from_fitted"] == 0.0
    assert abs(curve.loc[False, "delta_from_fitted"]) < 1e-12


def test_t10_the_bound_moves_a_categorical_reading() -> None:
    """The witness that the sweep reaches the categorical factor it bounds."""
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    result = _sample_fit("categorical")
    curve = longitudinal_truncation_curve(
        result, [(0.01, 1.0), (0.3, 1.0)], estimands=["ey_regimen[up]"]
    )
    curve = curve.set_index("is_fitted_bound")
    assert abs(curve.loc[False, "delta_from_fitted"]) > CAUGHT


def test_c5_the_stage_report_names_the_policy_nodes_and_their_mean_ratio() -> None:
    result = _sample_fit("continuous")
    rows = [row for row in result.diagnostics.support().rows if row.regimen == "up"]
    assert {row.node_kind for row in rows} == {"mtp"}
    fit = result.fits["up"]
    assert fit.node_ratio is not None
    for row in rows:
        at_risk = fit.steps[row.time - 1].at_risk
        assert row.assignment == pytest.approx(
            float(np.mean(fit.node_ratio[at_risk, row.time - 1]))
        )
    treatment = [
        row for row in result.diagnostics.nuisance_models().rows if row.role == "treatment"
    ]
    assert [row.model.kind for row in treatment] == ["conditional density"] * 2


# ------------------------------------------------------------------ T11 CausalStudy


def test_t11_a_study_identifies_and_estimates_a_contrast_of_policies() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast

    study = CausalStudy(
        law.frame(),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W",),
            time_varying=((), ("L2",)),
            continuous_treatment=("A1", "A2"),
        ),
    )
    identified = study.identify(
        RegimeContrast(mtp.regimens(("natural", "up")), reference="natural")
    )
    text = " ".join(identified.identification.assumptions)
    assert "Assumption 3" in text and "LMTP-SI" in text and "Assumption 4" in text
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        result = identified.estimate(
            outcome_learner=binary_law.CellMeans(),
            pseudo_learner=binary_law.CellMeans(),
            treatment_learner=multivalue.CellProbabilities(),
            g_bounds=NO_TRUNCATION,
            cross_fit=False,
        )
    name = "ate_regimen[up vs natural]"
    assert result.psi(name) == pytest.approx(law.truth(name), abs=1e-12)
