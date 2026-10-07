"""Declared treatment and censoring factors of a longitudinal fit.

With every treatment and censoring factor known, the ICE-TMLE's remainder is zero and its
curve is ``D*(Qbar_inf, g0)`` for any outcome regression (van der Laan and Gruber 2012,
Theorem 2 and Section 4).  The exact-law tests run on the two-node SMART law of
:mod:`tests.unit._known_mechanism_support`, whose empirical law is the law, so a fit with a
wrong outcome regression must return the truth.  A declared node factor is the declared
probability of the arm each row is *assigned*, read where the assigned history equals the
observed one, which is ``ltmle``'s numeric ``gform``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CapabilityError, CausalStudy, DataError, LongitudinalTreatment, RegimeMean
from cleverly.assessment import LONGITUDINAL_KNOWN_MECHANISM
from cleverly.data.known_mechanism import CONTINUOUS_REFUSAL
from cleverly.longitudinal import LTMLE, LongitudinalData
from tests.unit import _known_mechanism_support as law
from tests.unit._natural_course_support import NeverFit

#: The exact-law tolerance: each node's fluctuation stops at ``tol = 1e-10``.
EXACT = 1e-9
REGIMENS = {"always": 1, "never": 0, "early": (1, 0)}
TRUTH = {
    "ey_regimen[always]": law.smart_truth(1, 1),
    "ey_regimen[never]": law.smart_truth(0, 0),
    "ey_regimen[early]": law.smart_truth(1, 0),
}


class SmartMechanism(BaseEstimator, ClassifierMixin):
    """A treatment learner that returns the SMART law's mechanism from the history design.

    The node-one design is ``[L0]`` and the stage-one randomization is a half.  The node-two
    design is ``[L0, L1, A1]`` and the second stage randomizes by ``L1``.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> SmartMechanism:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if values.shape[1] == 1:
            one = np.full(values.shape[0], law.SMART_A1)
        else:
            one = np.where(values[:, 1] == 1.0, law.SMART_A2[1], law.SMART_A2[0])
        return np.column_stack([1.0 - one, one])


def _ltmle(**settings: Any) -> LTMLE:
    base: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "simultaneous": False,
    }
    base.update(settings)
    return LTMLE(REGIMENS, **base)


def _columns(censoring: bool) -> dict[str, Any]:
    columns = dict(law.SMART_COLUMNS)
    if censoring:
        columns["censoring"] = ["C1", "C2"]
    return columns


def _arrays(frame: pd.DataFrame, censoring: bool) -> dict[str, Any]:
    declared: dict[str, Any] = {
        "treatment_probabilities": frame[["g1_1", "g2_1"]].to_numpy(dtype=float)
    }
    if censoring:
        declared["censoring_probabilities"] = frame[["r1", "r2"]].to_numpy(dtype=float)
    return declared


def _mappings(censoring: bool) -> dict[str, Any]:
    declared: dict[str, Any] = {"treatment_probabilities": law.SMART_TREATMENT}
    if censoring:
        declared["censoring_probabilities"] = law.SMART_CENSORING
    return declared


@pytest.mark.parametrize("form", ["mapping", "array"])
@pytest.mark.parametrize("censoring", [False, True], ids=["complete", "known_censoring"])
def test_a_wrong_outcome_regression_with_known_factors_recovers_the_truth(
    censoring: bool, form: str
) -> None:
    """E10: every regimen mean equals the truth, with or without a declared censoring factor.

    No treatment or censoring learner runs: both are ``NeverFit``.
    """
    frame = law.smart_frame(censoring=censoring)
    declared = _mappings(censoring) if form == "mapping" else _arrays(frame, censoring)
    result = _ltmle().fit(frame, **_columns(censoring), **declared)
    for name, truth in TRUTH.items():
        assert abs(result[name].psi - truth) <= EXACT, name


def test_the_mapping_and_array_forms_are_bit_identical() -> None:
    frame = law.smart_frame(censoring=True)
    mapped = _ltmle().fit(frame, **_columns(True), **_mappings(True))
    arrayed = _ltmle().fit(frame, **_columns(True), **_arrays(frame, True))
    for name in mapped:
        assert mapped[name].psi == arrayed[name].psi
        np.testing.assert_array_equal(mapped[name].influence_curve, arrayed[name].influence_curve)


def _finite_smart(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = law.smart_frame(censoring=False)
    return frame.iloc[rng.choice(len(frame), size=n, replace=True)].reset_index(drop=True)


@pytest.mark.parametrize("n_folds", [1, 3])
def test_known_factors_equal_a_learner_that_returns_them(n_folds: int) -> None:
    """E4: the declaration and a treatment learner returning it give the same fit."""
    frame = _finite_smart()
    declared = _ltmle(n_folds=n_folds).fit(frame, **_columns(False), **_mappings(False))
    fitted = _ltmle(n_folds=n_folds, treatment_learner=SmartMechanism()).fit(
        frame, **_columns(False)
    )
    for name in declared:
        np.testing.assert_allclose(declared[name].psi, fitted[name].psi, rtol=1e-12)
        np.testing.assert_allclose(
            declared[name].influence_curve, fitted[name].influence_curve, rtol=1e-12, atol=1e-14
        )


def test_a_perturbed_node_moves_the_clever_covariate_from_that_node_on() -> None:
    """W2: changing node two's declaration changes node two's covariate and not node one's."""
    frame = _finite_smart()
    base = _arrays(frame, False)["treatment_probabilities"]
    moved = base.copy()
    moved[:, 1] = np.clip(moved[:, 1] + 0.05, 0.05, 0.95)
    first = _ltmle().fit(frame, **_columns(False), treatment_probabilities=base)
    second = _ltmle().fit(frame, **_columns(False), treatment_probabilities=moved)
    before = first.fits["always"].steps
    after = second.fits["always"].steps
    np.testing.assert_array_equal(before[0].clever, after[0].clever)
    followers = before[1].trained_on
    assert np.all(np.abs(before[1].clever - after[1].clever)[followers] > 1e-3)


def test_a_known_censoring_factor_is_read() -> None:
    """M8's detector: replacing the declared retention by one moves the estimate."""
    frame = law.smart_frame(censoring=True)
    known = _ltmle().fit(frame, **_columns(True), **_mappings(True))
    ones = _ltmle().fit(
        frame,
        **_columns(True),
        treatment_probabilities=law.SMART_TREATMENT,
        censoring_probabilities=np.ones((len(frame), 2)),
    )
    assert abs(known["ey_regimen[always]"].psi - ones["ey_regimen[always]"].psi) > 1e-3


def _long_frame(nodes: int = 7, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {"L0": rng.normal(size=n)}
    for node in range(1, nodes + 1):
        data[f"A{node}"] = rng.binomial(1, 0.5, size=n).astype(float)
    data["Y"] = rng.binomial(1, 0.5, size=n).astype(float)
    data["Y"][:4] = 1.0
    frame = pd.DataFrame(data)
    # Four rows follow "always" at every node, so the regimen is not empty.
    for node in range(1, nodes + 1):
        frame.loc[:3, f"A{node}"] = 1.0
    return frame


def test_a_long_known_product_below_the_bound_is_refused_before_any_learner() -> None:
    """W4, longitudinal: 0.5 at each of seven nodes reaches 0.0078, below 0.01."""
    nodes = 7
    frame = _long_frame(nodes)
    columns = {
        "outcome": "Y",
        "treatment": [f"A{t}" for t in range(1, nodes + 1)],
        "baseline": ["L0"],
    }
    probabilities = np.full((len(frame), nodes), 0.5)
    estimator = LTMLE(
        {"always": 1},
        n_folds=1,
        outcome_learner=NeverFit(),
        pseudo_learner=NeverFit(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        simultaneous=False,
    )
    NeverFit.calls = 0
    with pytest.raises(CapabilityError) as raised:
        estimator.fit(frame, **columns, treatment_probabilities=probabilities)
    message = str(raised.value)
    assert "g_bounds=[0.01, 1] first at node 7" in message
    assert "Truncating a known design mechanism moves the estimate" in message
    assert NeverFit.calls == 0


def test_a_missing_value_on_an_at_risk_row_is_refused() -> None:
    frame = law.smart_frame(censoring=True)
    arrays = _arrays(frame, True)
    treatment = arrays["treatment_probabilities"].copy()
    observed = np.flatnonzero(frame["C1"].to_numpy() == 1.0)
    treatment[observed[:3], 1] = np.nan
    NeverFit.calls = 0
    with pytest.raises(DataError, match="missing on 3 row\\(s\\) at risk there"):
        _ltmle().fit(
            frame,
            **_columns(True),
            treatment_probabilities=treatment,
            censoring_probabilities=arrays["censoring_probabilities"],
        )
    assert NeverFit.calls == 0


def test_a_zero_for_an_arm_a_unit_took_is_refused() -> None:
    frame = law.smart_frame(censoring=False)
    treatment = _arrays(frame, False)["treatment_probabilities"].copy()
    took = np.flatnonzero(frame["A1"].to_numpy() == 1.0)[:2]
    treatment[took, 0] = 0.0
    with pytest.raises(DataError, match="gives probability 0 to the arm that 2 at-risk unit"):
        _ltmle().fit(frame, **_columns(False), treatment_probabilities=treatment)


def test_a_continuous_node_cannot_be_declared() -> None:
    frame = _finite_smart().assign(
        D=lambda f: f["A1"] + np.random.default_rng(0).normal(size=len(f))
    )
    with pytest.raises(CapabilityError, match="a continuous treatment has a conditional density"):
        LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["D"],
            baseline=["L0"],
            continuous_treatment=["D"],
            treatment_probabilities={"D": {0.0: "g1_0", 1.0: "g1_1"}},
        )
    assert CONTINUOUS_REFUSAL.startswith("treatment_probabilities= gives arm probabilities")


def test_a_declaration_on_a_container_and_at_fit_is_refused() -> None:
    frame = _finite_smart()
    data = LongitudinalData.from_frame(frame, **_columns(False), **_mappings(False))
    with pytest.raises(ValueError, match="declared twice"):
        _ltmle().fit(data, treatment_probabilities=np.full((len(frame), 2), 0.5))


def test_a_bootstrap_replicate_carries_its_rows_and_fits_no_mechanism() -> None:
    frame = _finite_smart()
    data = LongitudinalData.from_frame(frame, **_columns(False), **_mappings(False))
    index = np.array([3, 3, 0, *range(10, 60)])
    sub = data.subset(index)
    assert sub.known_mechanisms is not None and data.known_mechanisms is not None
    np.testing.assert_array_equal(
        sub.known_mechanisms.treatment[1], data.known_mechanisms.treatment[1][index]
    )
    result = _ltmle(n_bootstrap=3).fit(data)
    assert result.bootstrap is not None
    assert result.bootstrap.n_failed == 0


def test_the_nuisance_report_names_each_declared_factor() -> None:
    frame = law.smart_frame(censoring=True)
    result = _ltmle().fit(frame, **_columns(True), **_mappings(True))
    reasons = {
        (item.role, item.time, item.reason)
        for item in result.diagnostics.nuisance_models().omissions
    }
    for role in ("treatment", "censoring"):
        for time in (1, 2):
            assert (role, time, LONGITUDINAL_KNOWN_MECHANISM) in reasons


def test_known_treatment_beside_an_estimated_censoring_factor_fits_the_censoring_learner() -> None:
    frame = law.smart_frame(censoring=True)
    result = _ltmle(censoring_learner=LogisticRegression(max_iter=1000)).fit(
        frame, **_columns(True), treatment_probabilities=law.SMART_TREATMENT
    )
    assert result.mechanism.known_treatment_nodes == (1, 2)
    assert result.mechanism.known_censoring_nodes == ()


def test_a_longitudinal_design_declares_the_columns() -> None:
    frame = law.smart_frame(censoring=True)
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["L0"],
        time_varying=[[], ["L1"]],
        censoring=["C1", "C2"],
        treatment_probabilities=law.SMART_TREATMENT,
        censoring_probabilities=law.SMART_CENSORING,
    )
    study = CausalStudy(frame, design=design)
    assert study.data.known_mechanisms is not None
    result = study.identify(RegimeMean({"always": 1})).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        cross_fit=False,
    )
    (estimate,) = result.estimates.values()
    assert abs(estimate.psi - law.smart_truth(1, 1)) <= 1e-6


def test_a_modified_treatment_policy_at_a_declared_node_equals_a_learner_that_returns_it() -> None:
    """The ratio numerator reads the declared matrix, as it reads a fitted one.

    A declared node is the degenerate estimate ``g_n = g0``, so the remainder, a product of
    the ratio error and the outcome error, is zero there.
    """
    from cleverly.interventions import ModifiedPolicy
    from cleverly.longitudinal import DynamicRegimen

    flip = ModifiedPolicy(
        "flip", apply=lambda a, h: 1.0 - np.asarray(a, dtype=float), policy_kind="known"
    )
    regimens = {"flip": DynamicRegimen("flip", (1, flip)), "always": 1}
    frame = law.smart_frame(censoring=False)
    declared_regimens = LTMLE(
        regimens,
        n_folds=1,
        random_state=0,
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        simultaneous=False,
    ).fit(frame, **_columns(False), **_mappings(False))
    fitted = LTMLE(
        regimens,
        n_folds=1,
        random_state=0,
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=SmartMechanism(),
        censoring_learner=NeverFit(),
        simultaneous=False,
    ).fit(frame, **_columns(False))
    assert declared_regimens.mechanism.known_treatment_nodes == (1, 2)
    for name in fitted.estimates:
        np.testing.assert_allclose(
            declared_regimens[name].psi, fitted[name].psi, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(
            declared_regimens[name].influence_curve,
            fitted[name].influence_curve,
            rtol=1e-11,
            atol=1e-13,
        )


def test_a_known_stochastic_policy_at_a_declared_node_recovers_the_truth() -> None:
    """A known policy reads no mechanism, so it composes with a declared one (F1 policies)."""
    from cleverly.interventions import Stochastic
    from cleverly.longitudinal import DynamicRegimen

    half = Stochastic(lambda frame: np.full((len(frame), 2), 0.5), "half", density_kind="known")
    frame = law.smart_frame(censoring=False)
    result = LTMLE(
        {"half": DynamicRegimen("half", (1, half))},
        n_folds=1,
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        simultaneous=False,
    ).fit(frame, **_columns(False), **_mappings(False))
    truth = 0.5 * law.smart_truth(1, 0) + 0.5 * law.smart_truth(1, 1)
    assert abs(result["ey_regimen[half]"].psi - truth) <= EXACT


# ----------------------------------------------------------------------------- compositions


def _panel_mechanism(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The declared factors of ``make_longitudinal``'s law, at each row's observed history."""
    from scipy.special import expit

    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    a1 = frame["A1"].to_numpy()
    l2 = frame["L2"].to_numpy()
    return {
        "treatment_probabilities": np.column_stack(
            [expit(0.3 * w1 - 0.4 * w2), expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2)]
        ),
        "censoring_probabilities": np.column_stack(
            [expit(2.2 + 0.3 * w1 - 0.3 * a1), expit(2.4 + 0.2 * l2)]
        ),
    }


PANEL = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _paired_panel_fits(frame: pd.DataFrame, outcome: Any, **settings: Any) -> tuple[Any, Any]:
    from tests.studies.canonical_ltmle import KnownLongitudinalMechanism

    base: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "random_state": 0,
        "simultaneous": False,
        "n_folds": 1,
    }
    base.update(settings)
    regimens = base.pop("regimens", {"always": 1, "never": 0})
    declared = LTMLE(
        regimens, treatment_learner=NeverFit(), censoring_learner=NeverFit(), **base
    ).fit(frame, outcome=outcome, **PANEL, **_panel_mechanism(frame))
    fitted = LTMLE(
        regimens,
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        **base,
    ).fit(frame, outcome=outcome, **PANEL)
    return declared, fitted


@pytest.mark.parametrize(
    "case",
    ["end_of_study", "survival", "cross_fitted", "clustered", "msm"],
)
def test_declared_factors_equal_learners_that_return_them(case: str) -> None:
    """Each composition the contract admits: the declaration is the mechanism a learner returns."""
    from cleverly.datasets import make_longitudinal, make_longitudinal_survival
    from cleverly.msm import MSM

    settings: dict[str, Any] = {}
    outcome: Any = "Y"
    if case == "survival":
        frame = make_longitudinal_survival(n=800, seed=4)[0]
        outcome = ["Y1", "Y2"]
    elif case == "clustered":
        frame = make_longitudinal(n=800, seed=5, cluster_size=8)[0]
        settings = {"n_folds": 4}
    else:
        frame = make_longitudinal(n=800, seed=6)[0]
    if case == "cross_fitted":
        settings = {"n_folds": 4}
    if case == "msm":

        def design(label: Any, horizon: int, data: Any) -> np.ndarray:
            n = len(data)
            return np.column_stack([np.ones(n), np.full(n, 1.0 if label == "always" else 0.0)])

        settings = {"msm": MSM(design=design, terms=("(intercept)", "always"), design_kind="known")}
    if case == "clustered":
        declared, fitted = _clustered_pair(frame, settings)
    else:
        declared, fitted = _paired_panel_fits(frame, outcome, **settings)
    assert declared.estimates.keys() == fitted.estimates.keys()
    for name in declared.estimates:
        np.testing.assert_allclose(declared[name].psi, fitted[name].psi, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            declared[name].influence_curve, fitted[name].influence_curve, rtol=1e-11, atol=1e-13
        )
    assert declared.mechanism.known_treatment_nodes == (1, 2)


def _clustered_pair(frame: pd.DataFrame, settings: dict[str, Any]) -> tuple[Any, Any]:
    from tests.studies.canonical_ltmle import KnownLongitudinalMechanism

    base: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "random_state": 0,
        "simultaneous": False,
        **settings,
    }
    regimens = {"always": 1, "never": 0}
    declared = LTMLE(
        regimens, treatment_learner=NeverFit(), censoring_learner=NeverFit(), **base
    ).fit(frame, outcome="Y", id="id", **PANEL, **_panel_mechanism(frame))
    fitted = LTMLE(
        regimens,
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        **base,
    ).fit(frame, outcome="Y", id="id", **PANEL)
    return declared, fitted


# ----------------------------------------------------------------------------- identification


def _certain_responders() -> pd.DataFrame:
    """The SMART law with responders after A1 = 1 kept on arm 1 with certainty.

    ``P(A2 = 0 | L1 = 1, A1 = 1) = 0``, so the regimen ``(1, 0)`` is not identified for
    those responders, and the declaration says so.
    """
    frame = law.smart_frame(censoring=False)
    certain = (frame["L1"] == 1.0) & (frame["A1"] == 1.0)
    frame.loc[certain, "A2"] = 1.0
    frame.loc[certain, "g2_0"] = 0.0
    frame.loc[certain, "g2_1"] = 1.0
    return frame


def test_a_regimen_the_declared_design_cannot_follow_is_refused_before_any_learner() -> None:
    """H1: a declared zero on the arm a regimen assigns at-risk rows refuses that regimen."""
    frame = _certain_responders()
    count = int(((frame["L1"] == 1.0) & (frame["A1"] == 1.0)).sum())
    NeverFit.calls = 0
    with pytest.raises(DataError) as raised:
        _ltmle(outcome_learner=NeverFit(), pseudo_learner=NeverFit()).fit(
            frame, **_columns(False), **_mappings(False)
        )
    message = str(raised.value)
    assert "regimen 'early' is not identified under the declared design" in message
    assert "at node 'A2' it assigns an arm the declared mechanism gives probability 0" in message
    assert f"on {count} row(s) at risk under the regimen" in message
    assert NeverFit.calls == 0
    identified = LTMLE(
        {"always": 1, "never": 0},
        n_folds=1,
        random_state=0,
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
        simultaneous=False,
    ).fit(frame, **_columns(False), **_mappings(False))
    assert np.isfinite(identified["ey_regimen[always]"].psi)


def test_a_stochastic_policy_on_a_declared_zero_is_refused() -> None:
    from cleverly.interventions import Stochastic
    from cleverly.longitudinal import DynamicRegimen

    half = Stochastic(lambda frame: np.full((len(frame), 2), 0.5), "half", density_kind="known")
    NeverFit.calls = 0
    with pytest.raises(
        DataError, match="it draws an arm the declared mechanism gives probability 0"
    ):
        LTMLE(
            {"half": DynamicRegimen("half", (1, half))},
            n_folds=1,
            outcome_learner=NeverFit(),
            pseudo_learner=NeverFit(),
            treatment_learner=NeverFit(),
            censoring_learner=NeverFit(),
        ).fit(_certain_responders(), **_columns(False), **_mappings(False))
    assert NeverFit.calls == 0


def _all_censored_after_treatment() -> pd.DataFrame:
    """The censored SMART law with every ``L0 = 1, A1 = 1`` unit censored at ``C1``."""
    frame = law.smart_frame(censoring=True)
    lost = (frame["L0"] == 1.0) & (frame["A1"] == 1.0)
    frame.loc[lost, "C1"] = 0.0
    frame.loc[lost, ["L1", "A2", "C2", "Y", "g2_0", "g2_1", "r2"]] = np.nan
    frame.loc[lost, "r1"] = 0.0
    return frame


def test_a_declared_zero_retention_on_a_regimen_s_followers_is_refused() -> None:
    frame = _all_censored_after_treatment()
    NeverFit.calls = 0
    with pytest.raises(DataError) as raised:
        _ltmle(outcome_learner=NeverFit(), pseudo_learner=NeverFit()).fit(
            frame, **_columns(True), **_mappings(True)
        )
    message = str(raised.value)
    assert "regimen 'always' is not identified under the declared design" in message
    assert "at censoring node 'C1'" in message
    assert NeverFit.calls == 0


def test_a_declared_zero_retention_for_a_unit_that_stayed_is_refused() -> None:
    """L8: a contradiction, not a bound the caller could widen."""
    frame = law.smart_frame(censoring=True)
    stayed = np.flatnonzero(frame["C1"].to_numpy() == 1.0)[:3]
    frame.loc[stayed, "r1"] = 0.0
    with pytest.raises(DataError, match="retention probability 0 to 3 unit\\(s\\) that stayed"):
        _ltmle().fit(frame, **_columns(True), **_mappings(True))


def test_a_rare_level_at_a_declared_node_needs_no_training_fold() -> None:
    """M2: a declared node fits no learner, so the fold-support check skips it."""
    frame = law.smart_frame(censoring=False)
    at_risk = frame["A2"].notna().to_numpy()
    frame["g2_2"] = np.where(at_risk, 0.05, np.nan)
    frame["g2_0"] = frame["g2_0"] * 0.95
    frame["g2_1"] = frame["g2_1"] * 0.95
    rare = np.flatnonzero(at_risk)[:1]
    frame.loc[rare, "A2"] = 2.0
    treatment = {
        "A1": law.SMART_TREATMENT["A1"],
        "A2": {0.0: "g2_0", 1.0: "g2_1", 2.0: "g2_2"},
    }
    NeverFit.calls = 0
    result = _ltmle(n_folds=3).fit(frame, **_columns(False), treatment_probabilities=treatment)
    assert NeverFit.calls == 0
    assert np.isfinite(result["ey_regimen[always]"].psi)


def test_the_record_names_each_declared_factor() -> None:
    frame = law.smart_frame(censoring=True)
    result = _ltmle().fit(frame, **_columns(True), **_mappings(True))
    assert result.config.known_factors == ("A1", "C1", "A2", "C2")
    assert "known factors (declared, no learner): A1, C1, A2, C2" in result.summary()
    estimated = _ltmle(
        treatment_learner=SmartMechanism(), censoring_learner=LogisticRegression()
    ).fit(frame, **_columns(True))
    assert estimated.config.known_factors == ()


# ----------------------------------------------------------------------------- held designs


def _held_mechanism(frame: pd.DataFrame) -> np.ndarray:
    from tests.discrete_law_point_survival import G

    return G[frame["W"].to_numpy().astype(int)]


def _held_fits(
    declared_input: Any,
    fitted_input: Any,
    regimens: dict[str, Any],
    declaration: dict[str, Any] | None = None,
    **columns: Any,
) -> tuple[Any, Any]:
    """A fit on the declaration and a fit whose treatment learner returns it."""
    from tests.discrete_law_point_survival import CellMeans, CellProbabilities

    common: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": CellMeans(),
        "censoring_learner": CellMeans(),
    }
    declared = LTMLE(regimens, treatment_learner=NeverFit(), **common).fit(
        declared_input, **columns, **(declaration or {})
    )
    fitted = LTMLE(regimens, treatment_learner=CellProbabilities(), **common).fit(
        fitted_input, **columns
    )
    return declared, fitted


def _assert_same(declared: Any, fitted: Any) -> None:
    assert declared.estimates.keys() == fitted.estimates.keys()
    for name in declared.estimates:
        np.testing.assert_allclose(declared[name].psi, fitted[name].psi, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            declared[name].influence_curve, fitted[name].influence_curve, rtol=1e-11, atol=1e-13
        )


@pytest.mark.parametrize("causes", [1, 2], ids=["survival", "competing_risks"])
def test_a_held_declaration_equals_a_learner_that_returns_it(causes: int) -> None:
    """E4 on a held three-level baseline decision, with one cause or two.

    The single-cause case adds a categorical MTP at the declared decision node.
    """
    from cleverly.interventions import ModifiedPolicy
    from tests.discrete_law_point_survival import survival_law

    law_ = survival_law(causes=causes)
    frame = law_.frame()
    matrix = _held_mechanism(frame)
    regimens: dict[str, Any] = {"a0": 0, "a2": 2}
    if causes == 1:
        regimens["up"] = ModifiedPolicy(
            "up",
            apply=lambda a, h: np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0),
            policy_kind="known",
        )
    declaration = {"treatment_probabilities": {"A": {float(a): matrix[:, a] for a in range(3)}}}
    declared, fitted = _held_fits(
        frame, frame, regimens, declaration, **law_.fit_columns(held=True)
    )
    assert declared.mechanism.known_treatment_nodes == (1,)
    assert declared.config.known_factors == ("A",)
    _assert_same(declared, fitted)


def test_a_time_to_event_container_declares_its_baseline_mechanism() -> None:
    """``from_time_to_event`` and ``TimeToEvent`` take the column form, and the fit agrees."""
    from cleverly import TimeToEvent
    from tests.discrete_law_point_survival import survival_law

    law_ = survival_law(censor_first=False)
    grid = (1.0, 2.0, 3.0)
    long = law_.long_frame(grid)
    matrix = _held_mechanism(long)
    long = long.assign(p0=matrix[:, 0], p1=matrix[:, 1], p2=matrix[:, 2])
    columns = {0.0: "p0", 1.0: "p1", 2.0: "p2"}
    roles: dict[str, Any] = {
        "time": "time",
        "event": "event",
        "treatment": "A",
        "baseline": ["W"],
        "grid": grid,
    }
    declared_data = LongitudinalData.from_time_to_event(
        long, **roles, treatment_probabilities=columns
    )
    plain = LongitudinalData.from_time_to_event(long, **roles)
    assert list(declared_data.baseline_names) == ["W"]
    assert declared_data.known_mechanisms is not None
    declared, fitted = _held_fits(declared_data, plain, {"a0": 0, "a2": 2})
    _assert_same(declared, fitted)

    design = TimeToEvent(**roles, treatment_probabilities=columns)
    assert design.treatment_probabilities == tuple(columns.items())
    prepared = design.prepare(long)
    assert prepared.known_mechanisms is not None
    design.prepare(declared_data)
    with pytest.raises(DataError, match="names columns"):
        TimeToEvent(**roles, treatment_probabilities=matrix)
