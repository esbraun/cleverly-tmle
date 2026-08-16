"""The separate randomized missing-outcome DR-TMLE construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor

from cleverly import DRTMLE, load
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.reduced import MissingOutcomeReducedSet
from cleverly.estimators.tmle import build_submodel, correction_parts
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference.influence import missing_outcome_correction_parts


def _trial(n: int = 320, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.1 + 0.4 * a + 0.3 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 0.8 + 1.1 * a + 0.4 * w1 - 0.2 * w2 + rng.normal(scale=0.6, size=n)
    y[observed == 0.0] = np.nan
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": y})


def _estimator(**settings: object) -> DRTMLE:
    return DRTMLE(
        randomized=True,
        cross_fit=False,
        outcome_learner="glm",
        treatment_learner="glm",
        missingness_learner="glm",
        reduced_outcome_learner="glm",
        reduced_treatment_learner="glm",
        estimands=("ate", "ey1", "ey0"),
        simultaneous=False,
        random_state=0,
        **settings,
    )


@pytest.fixture(scope="module")
def randomized_fit():
    return (
        _estimator()
        .fit(
            _trial(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
        )
        .single()
    )


def test_randomized_missing_outcomes_solve_the_reported_equations(randomized_fit) -> None:
    assert randomized_fit.validation.score_check().passed
    check = randomized_fit.validation.correction_check()
    assert check.passed
    assert {row.equation for row in check.rows} == {"D*_A", "D*_M", "D*_Y"}
    assert len(check.rows) == 6


def test_each_arms_covariate_divides_by_separately_targeted_mechanisms(randomized_fit) -> None:
    data = randomized_fit.data
    repeat = randomized_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    lower, upper = fluctuation.reduction.bounds
    targeted_upper = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    targeted_g = np.column_stack([1.0 - targeted_upper, targeted_upper])
    targeted_m = np.asarray(fluctuation.reduction.observation.propensity, dtype=float)
    current = replace(
        repeat.nuisance,
        propensity=Propensity(targeted_g, data.arm_codes),
        missingness=targeted_m,
    )

    submodel = build_submodel(
        data,
        current,
        "mean",
        bounds=fluctuation.reduction.bounds,
        nuisance_bound=randomized_fit.config.missingness_bound,
    )
    for column, arm in enumerate(data.arm_codes):
        expected = 1.0 / (
            np.clip(targeted_g[:, column], lower, upper)
            * np.clip(targeted_m[:, column], randomized_fit.config.missingness_bound, 1.0)
        )
        np.testing.assert_array_equal(submodel.arms[arm][:, column], expected)
    assert fluctuation.mechanism.propensity.ndim == 1
    assert fluctuation.reduction.observation.propensity.shape == targeted_m.shape


def test_the_five_reductions_and_observation_tilt_survive_serialization(
    randomized_fit, tmp_path
) -> None:
    path = tmp_path / "drtmle-missing.cleverly"
    randomized_fit.save(path)
    restored = load(path)
    before = randomized_fit.repeats[0].fluctuations["mean"].reduction
    after = restored.repeats[0].fluctuations["mean"].reduction
    assert isinstance(after.reduced, MissingOutcomeReducedSet)
    for name in ("gamma_a", "gamma_m", "r_a", "r_m", "e"):
        np.testing.assert_array_equal(getattr(after.reduced, name), getattr(before.reduced, name))
    np.testing.assert_array_equal(after.observation.propensity, before.observation.propensity)
    assert restored.validation.score_check() == randomized_fit.validation.score_check()


def test_the_report_shows_the_product_the_covariate_divides_by(randomized_fit) -> None:
    mechanisms = randomized_fit.sensitivity.positivity().mechanisms
    joint = randomized_fit.nuisance.propensity.values * randomized_fit.nuisance.missingness
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] == pytest.approx(float(joint.min()))
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] < mechanisms["P(Delta=1|A,W)"]["min"]


def test_treatment_truncation_sweep_counts_only_the_treatment_mechanism(randomized_fit) -> None:
    propensity = randomized_fit.nuisance.propensity.values
    bound = 0.1
    curve = randomized_fit.sensitivity.truncation_curve(bounds=[bound])
    reported = float(np.asarray(curve["truncated_fraction"])[0])
    assert reported == pytest.approx(float(np.mean((propensity < bound) | (propensity > 0.9))))


def test_observation_truncation_sweep_refits_at_and_counts_the_selected_bound(
    randomized_fit,
) -> None:
    missingness = randomized_fit.nuisance.missingness
    assert missingness is not None
    bound = 0.2
    curve = randomized_fit.sensitivity.truncation_curve(bounds=[bound], mechanism=True)
    reported = float(np.asarray(curve["truncated_fraction"])[0])
    assert reported == pytest.approx(float(np.mean(missingness < bound)))

    estimator = randomized_fit.estimator
    assert estimator is not None
    _, fluctuations = estimator.retarget(
        randomized_fit.data,
        randomized_fit.repeats[0].nuisance,
        estimands=("ate",),
        nuisance_bound=bound,
    )
    reduced = fluctuations["mean"].reduction.reduced
    assert isinstance(reduced, MissingOutcomeReducedSet)
    assert reduced.missingness_bound == bound


def test_corrections_keep_the_three_paper_blocks_separate(randomized_fit) -> None:
    repeat = randomized_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    parts = correction_parts(
        randomized_fit.data,
        repeat.nuisance,
        fluctuation,
        fluctuation.targeted,
        repeat.nuisance.scaler.scale(randomized_fit.data.outcome),
    )
    assert parts.d_a is not None and parts.d_m is not None and parts.d_y is not None
    for arm in randomized_fit.data.arm_codes:
        np.testing.assert_array_equal(parts.d_g[arm], parts.d_a[arm] + parts.d_m[arm])


def test_treatment_correction_has_a_nonzero_independent_witness() -> None:
    reduced = MissingOutcomeReducedSet(
        gamma_a=np.full((4, 2), 0.5),
        gamma_m=np.full((4, 2), 0.7),
        r_a=np.full((4, 2), 0.2),
        r_m=np.full((4, 2), -0.1),
        e=np.array([[0.2, -0.1], [0.3, -0.2], [0.4, -0.3], [0.5, -0.4]]),
        arms=(0.0, 1.0),
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
    )
    treatment = np.array([0.0, 0.0, 1.0, 1.0])
    observed = np.array([True, False, True, False])
    targeted = InitialFit(
        np.array([0.2, 0.3, 0.4, 0.5]),
        {0.0: np.zeros(4), 1.0: np.ones(4)},
    )
    parts = missing_outcome_correction_parts(
        np.array([0.1, 0.2, 0.3, 0.4]),
        targeted,
        treatment,
        observed,
        reduced,
        np.array([0.45, 0.55, 0.60, 0.50]),
        np.full((4, 2), 0.7),
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
        guard=("Q", "g"),
    )
    assert parts.d_a is not None and parts.d_m is not None
    assert np.max(np.abs(parts.d_a[1.0])) > 0.1
    collapsed = parts.d_a[1.0] + parts.d_m[1.0]
    np.testing.assert_array_equal(parts.d_g[1.0], collapsed)
    assert np.max(np.abs(parts.d_a[1.0] - collapsed)) > 0.01


@pytest.mark.parametrize("guard", [("Q",), ("g",)])
def test_partial_guards_are_refused_for_missing_outcomes(guard) -> None:
    with pytest.raises(NotImplementedError, match="requires guard"):
        _estimator(guard=guard).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


class _FailIfFit(BaseEstimator):
    def fit(self, X, y, sample_weight=None):  # pragma: no cover - failure is the assertion
        raise AssertionError("known randomization probabilities must bypass treatment fitting")


def test_known_probabilities_bypass_the_treatment_learner() -> None:
    frame = _trial(n=260, seed=17)
    estimator = DRTMLE(
        randomized=False,
        cross_fit=False,
        outcome_learner="glm",
        treatment_learner=_FailIfFit(),
        missingness_learner="glm",
        reduced_outcome_learner="glm",
        reduced_treatment_learner="glm",
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        delta="Delta",
        treatment_probabilities=np.full(len(frame), 0.5),
    ).single()
    np.testing.assert_array_equal(result.nuisance.propensity.values, np.full((len(frame), 2), 0.5))
    assert result.validation.score_check().passed


def test_two_dimensional_known_probabilities_are_copied() -> None:
    frame = _trial(n=140, seed=29)
    supplied = np.full((len(frame), 2), 0.5)
    result = (
        _estimator()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=supplied,
        )
        .single()
    )
    before = result.nuisance.propensity.values.copy()
    assert not np.shares_memory(supplied, result.nuisance.propensity.values)
    supplied[:, :] = (0.2, 0.8)
    np.testing.assert_array_equal(result.nuisance.propensity.values, before)


def test_known_probability_recipe_is_explicitly_unreconstructible(tmp_path) -> None:
    frame = _trial(n=120, seed=31)
    result = (
        _estimator()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full(len(frame), 0.5),
        )
        .single()
    )
    path = tmp_path / "known-probabilities.cleverly"
    result.save(path)
    restored = load(path)
    assert restored.validation.score_check() == result.validation.score_check()
    curve = restored.sensitivity.truncation_curve(bounds=[0.05])
    assert len(curve) == len(result.estimates)
    with pytest.raises(ValueError, match="row-aligned known treatment probabilities"):
        restored.estimator.refit(restored.data)


def test_observational_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="randomized trial"):
        DRTMLE(cross_fit=False, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


def test_cross_fitted_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="cross-validated extension"):
        DRTMLE(randomized=True, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (np.full(99, 0.5), "99 rows"),
        (np.full((100, 3), 1 / 3), r"must be \(n,\)"),
        (np.zeros(100), "strictly between"),
        (np.column_stack([np.full(100, 0.5), np.full(100, 0.4)]), "sum to one"),
        ({"nope": np.full(100, 0.5), 1: np.full(100, 0.5)}, "not a level"),
        ({1: np.full(100, 0.6)}, "must name every arm"),
        ({0: np.full(3, 0.4), 1: np.full(100, 0.6)}, r"\[0\] has 3 rows"),
    ],
)
def test_invalid_known_probability_shapes_are_refused(probabilities, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _estimator().fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=probabilities,
        )


def test_bootstrapping_known_probabilities_is_refused() -> None:
    """A replicate refits on resampled rows the supplied array cannot follow.

    An n-out-of-n resample even passes the length check, so nothing downstream would
    notice -- and ``run_bootstrap`` swallows replicate failures, so raising inside one
    would come back as "the fit is too unstable to bootstrap".
    """
    with pytest.raises(NotImplementedError, match="n_bootstrap"):
        _estimator(n_bootstrap=5).fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full(100, 0.5),
        )


def _labelled_trial(n: int = 300, seed: int = 21) -> tuple[pd.DataFrame, np.ndarray]:
    """A stratified trial whose arms are named, so the positional binding is asymmetric.

    ``active`` sorts before ``placebo``, so arm code ``1`` is ``placebo`` and a caller who
    reads the one-dimensional form as "the probability of treatment" inverts the design.
    The allocation is unequal *and* varies with ``W1``: a constant one would make the
    inversion invisible in the point estimate, because rescaling both arms' clever
    covariates by constants is absorbed by the fluctuation's own coefficients.
    """
    frame = _trial(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    active = np.where(frame["W1"].to_numpy() > 0.0, 0.75, 0.55)
    treated = rng.binomial(1, active).astype(float)
    frame["A"] = treated
    frame["arm"] = np.where(treated == 1.0, "active", "placebo")
    return frame, active


def test_known_probabilities_may_be_keyed_by_the_arm_they_belong_to() -> None:
    """The mapping form says which arm each column is, and matches the positional one.

    ``{"active": p_active, "placebo": p_placebo}`` and the ``(n,)`` vector
    ``P(A = 'placebo' | W)`` are the same design stated two ways, so they must give the
    same mechanism.  Writing the vector as ``P(A = 'active' | W)`` instead is a different
    design, and the assertion at the end is that the two really do differ -- otherwise the
    agreement above would be vacuous.
    """
    frame, active = _labelled_trial()
    keyword = {"outcome": "Y", "treatment": "arm", "covariates": ["W1", "W2"], "delta": "Delta"}
    keyed = (
        _estimator()
        .fit(frame, **keyword, treatment_probabilities={"active": active, "placebo": 1.0 - active})
        .single()
    )
    positional = _estimator().fit(frame, **keyword, treatment_probabilities=1.0 - active).single()
    np.testing.assert_array_equal(
        keyed.nuisance.propensity.values, positional.nuisance.propensity.values
    )
    # Column 1 is `placebo`, the second sorted level -- not "the treated arm".
    np.testing.assert_allclose(keyed.nuisance.propensity.arm(1.0), 1.0 - active)
    inverted = _estimator().fit(frame, **keyword, treatment_probabilities=active).single()
    assert abs(inverted.estimates["ate"].psi - keyed.estimates["ate"].psi) > 1e-8


def test_an_unnamed_probability_vector_says_which_arm_it_bound_to() -> None:
    """The message names the resolved level, since the code is what the caller cannot see."""
    frame, _ = _labelled_trial(n=120)
    with pytest.raises(ValueError, match="P\\(arm = 'placebo' \\| W\\)"):
        _estimator().fit(
            frame,
            outcome="Y",
            treatment="arm",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full((len(frame), 3), 1 / 3),
        )


@pytest.mark.slow
def test_the_estimator_is_consistent_when_only_the_outcome_model_is_wrong() -> None:
    """Double robustness through the treatment and observation mechanisms.

    The check the exact identities above cannot make: a deliberately misspecified outcome
    regression against correctly specified treatment and missingness mechanisms, where
    only a covariate that divides by the right per-arm probability recovers the truth.
    Both arms are asserted, because the arm-1 column is right under either truncation rule
    and it is arm 0 that carries the evidence.
    """
    rng = np.random.default_rng(7)
    n = 20000
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.4 + 1.2 * a + 0.8 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 2.0 + 4.0 * a + 1.5 * w1 - 0.8 * w2 + 2.0 * w1 * w2 + rng.normal(scale=0.5, size=n)
    truth = {"ey0": 2.0, "ey1": 6.0, "ate": 4.0}
    frame = pd.DataFrame(
        {"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": np.where(observed == 1.0, y, np.nan)}
    )
    fit = (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            # Misspecified on purpose, and as badly as the class allows: an intercept, blind to
            # both the interaction and the covariate that drives the chance of being seen. Only
            # the mechanism can carry consistency from here.
            outcome_learner=DummyRegressor(strategy="mean"),
            treatment_learner="glm",
            missingness_learner="glm",
            reduced_outcome_learner="glm",
            reduced_treatment_learner="glm",
            estimands=("ate", "ey1", "ey0"),
            simultaneous=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta")
        .single()
    )
    for name, value in truth.items():
        assert abs(fit.estimates[name].psi - value) < 0.15, name
