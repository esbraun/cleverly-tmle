"""The theorem-backed randomized missing-outcome DR-TMLE surface.

The canonical R implementation writes the mechanism correction with
``I(A=a, Delta=1)`` while its ordinary complete-data reduction is easy to read as
``I(A=a)``.  The nonzero array witness below is the acceptance evidence for that mask;
the end-to-end tests cover the public eligibility and known-randomization contracts.

Everything the fit solves it solves *against the covariate it was handed*, so a score of
zero says nothing about whether that covariate was the right one -- the alternation drives
whatever it is given to zero, and ``gr2``'s target and the correction's ratio are truncated
by the same array, so their identity holds either way.  The tests that pin the denominator
therefore compare the fit against something outside it: the joint mechanism's own columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor

from cleverly import DRTMLE, load
from cleverly.estimators.reduced import ReducedSet
from cleverly.estimators.targeting import _retargeted_mechanism
from cleverly.estimators.tmle import build_submodel, correction_parts
from cleverly.exceptions import PositivityWarning
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference.influence import reduced_correction_parts


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
    assert randomized_fit.validation.correction_check().passed
    joint = randomized_fit.nuisance.reduction_mechanism
    assert joint is not None
    np.testing.assert_allclose(
        joint.values,
        randomized_fit.nuisance.propensity.values * randomized_fit.nuisance.missingness,
        rtol=0,
        atol=0,
    )


def test_each_arms_covariate_divides_by_its_own_joint_probability(randomized_fit) -> None:
    """Equation (8)'s covariate is ``1 / g_a pi_a``, arm by arm -- not by a complement.

    The fitted submodel is not kept on the result, but every input the closing pass gave
    :func:`build_submodel` is, so this rebuilds it through the production functions rather
    than re-deriving the arithmetic.  ``_retargeted_mechanism`` sets ``missingness=None``
    on the joint path, so ``pi`` is one inside ``mean_submodel`` and the covariate is
    exactly the reciprocal of the truncated joint -- ``nuisance_bound`` cannot enter it.
    """
    data = randomized_fit.data
    repeat = randomized_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    lower, upper = fluctuation.reduction.bounds
    joint = np.asarray(fluctuation.mechanism.propensity, dtype=float)

    submodel = build_submodel(
        data,
        _retargeted_mechanism(repeat.nuisance, joint, data.arm_codes, joint=True),
        "mean",
        bounds=fluctuation.reduction.bounds,
        nuisance_bound=randomized_fit.config.missingness_bound,
    )
    for column, arm in enumerate(data.arm_codes):
        np.testing.assert_array_equal(
            submodel.arms[arm][:, column], 1.0 / np.clip(joint[:, column], lower, upper)
        )
    # The mutation control. Arm 0's denominator under the complement rule is the
    # probability of *not* being treated-and-observed, which is a far larger number, so the
    # equality above is one only the per-arm rule can satisfy.
    complement = 1.0 / (1.0 - np.clip(joint[:, 1], lower, upper))
    assert np.max(np.abs(1.0 / np.clip(joint[:, 0], lower, upper) - complement)) > 1.0


def test_the_joint_mechanism_truncates_arm_by_arm(randomized_fit) -> None:
    """The other site the same rule reaches: ``gr2``'s target in ``reduced._roles``."""
    joint = randomized_fit.nuisance.reduction_mechanism
    bounds = randomized_fit.repeats[0].fluctuations["mean"].reduction.bounds
    assert joint.simplex is False
    np.testing.assert_array_equal(joint.bounded(bounds), np.clip(joint.values, *bounds))


def test_the_joint_mechanism_survives_serialization(randomized_fit, tmp_path) -> None:
    path = tmp_path / "drtmle-missing.cleverly"
    randomized_fit.save(path)
    restored = load(path)
    assert restored.nuisance.reduction_mechanism is not None
    np.testing.assert_array_equal(
        restored.nuisance.reduction_mechanism.values,
        randomized_fit.nuisance.reduction_mechanism.values,
    )
    # Not merely the array: a reloaded joint that forgot it was off the simplex would
    # truncate by the complement rule and give a retarget a different denominator.
    assert restored.nuisance.reduction_mechanism.simplex is False
    assert restored.validation.score_check() == randomized_fit.validation.score_check()


def test_the_report_shows_the_product_the_covariate_divides_by(randomized_fit) -> None:
    """``g`` and ``pi`` stay separately reported, and their product gets its own row."""
    mechanisms = randomized_fit.sensitivity.positivity().mechanisms
    joint = randomized_fit.nuisance.reduction_mechanism.values
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] == pytest.approx(float(joint.min()))
    # Strictly smaller than either factor, which is the whole reason it needs its own row.
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] < mechanisms["P(Delta=1|A,W)"]["min"]


def test_the_truncation_sweep_counts_the_joint_it_would_clip(randomized_fit) -> None:
    """``g_bounds`` is applied to the joint, so the swept fraction must be the joint's."""
    joint = randomized_fit.nuisance.reduction_mechanism.values
    bound = float(np.quantile(joint, 0.25))
    curve = randomized_fit.sensitivity.truncation_curve(bounds=[bound])
    reported = float(np.asarray(curve["truncated_fraction"])[0])
    assert reported == pytest.approx(float(np.mean(joint < bound)))
    # The marginal propensity is a constant one half here, so it would report nothing at
    # all -- the number this column used to show.
    assert reported > 0.0
    assert np.mean(randomized_fit.nuisance.propensity.values < bound) == 0.0


@pytest.fixture(scope="module")
def pinched_fit():
    """A fit whose *joint* mechanism binds against ``g_bounds`` while its ``g`` does not.

    Randomization makes the propensity a flat one half, so a bound between the joint's
    range and that half is active for the reductions and invisible in the propensity --
    which is exactly the gap a diagnostic reading the propensity cannot see.
    """
    with pytest.warns(PositivityWarning, match="P\\(A = a, Delta = 1"):
        return (
            _estimator(guard=("g",), g_bounds=0.25)
            .fit(_trial(), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta")
            .single()
        )


def test_the_correction_is_formed_at_the_mechanism_the_fit_divided_by(pinched_fit) -> None:
    """With no ``"Q"`` guard nothing was tilted, so the corrections read the initial fit.

    The initial mechanism the covariates divided by is the joint one, not the separately
    reported propensity -- and on a randomized trial those are far apart, because the
    propensity is a constant one half.
    """
    repeat = pinched_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    assert fluctuation.mechanism is None  # no "Q" guard, so nothing was tilted
    parts = correction_parts(
        pinched_fit.data,
        repeat.nuisance,
        fluctuation,
        fluctuation.targeted,
        repeat.nuisance.scaler.scale(pinched_fit.data.outcome),
    )
    reduced = fluctuation.reduction.reduced
    joint = np.clip(repeat.nuisance.reduction_mechanism.values, *fluctuation.reduction.bounds)
    treatment = np.asarray(pinched_fit.data.treatment, dtype=float)
    observed = np.asarray(pinched_fit.data.observed, dtype=float)
    for column, arm in enumerate(reduced.arms):
        indicator = (treatment == arm).astype(float) * observed
        expected = reduced.qr[:, column] / joint[:, column] * (indicator - joint[:, column])
        np.testing.assert_allclose(parts.d_g[arm], expected, rtol=0, atol=1e-14)
    # The mutation control: the propensity is a flat one half here, so a correction formed
    # at it is a different array entirely rather than a rounding away.
    marginal = repeat.nuisance.propensity.arm(1.0)
    assert np.max(np.abs(marginal - joint[:, 1])) > 0.1


def test_the_contract_witness_reports_the_bound_that_actually_bound(pinched_fit) -> None:
    """``initial_clipped`` must come off the joint, or ``contract`` overstates the fit.

    The joint is uniformly the smaller of the two mechanisms, so a witness reading the
    propensity can only ever under-report -- and under-reporting here means claiming the
    theorem's estimator when a bound was active.
    """
    check = pinched_fit.validation.correction_check()
    assert check.initial_clip_share > 0.0
    assert "g-hat at the initial fit" in check.truncations_active
    assert check.contract == "bound-active"
    # Nothing about the reported propensity says so, which is the point.
    lower, upper = pinched_fit.repeats[0].fluctuations["mean"].reduction.bounds
    marginal = pinched_fit.nuisance.propensity.values
    assert not np.any((marginal < lower) | (marginal > upper))


def test_the_mechanism_correction_uses_the_observation_mask() -> None:
    observed = np.array([True, False, True, False])
    treatment = np.array([0.0, 0.0, 1.0, 1.0])
    joint = np.array([[0.30, 0.40], [0.25, 0.35], [0.30, 0.40], [0.25, 0.35]])
    reduced = ReducedSet(
        qr=np.array([[0.2, -0.1], [0.3, -0.2], [0.4, -0.3], [0.5, -0.4]]),
        gr1=np.full((4, 2), 0.5),
        gr2=np.full((4, 2), 0.1),
        arms=(0.0, 1.0),
        g_bounds=(1e-6, 1 - 1e-6),
    )
    targeted = InitialFit(np.array([0.2, 0.3, 0.4, 0.5]), {0.0: np.zeros(4), 1.0: np.ones(4)})
    parts = reduced_correction_parts(
        np.array([0.1, 0.2, 0.3, 0.4]),
        targeted,
        treatment,
        reduced,
        joint,
        bounds=(1e-6, 1 - 1e-6),
        observed=observed,
        guard=("Q", "g"),
    )
    for column, arm in enumerate(reduced.arms):
        indicator = ((treatment == arm) & observed).astype(float)
        expected = reduced.qr[:, column] / joint[:, column] * (indicator - joint[:, column])
        np.testing.assert_allclose(parts.d_g[arm], expected, rtol=0, atol=1e-15)
        wrong = (
            reduced.qr[:, column]
            / joint[:, column]
            * ((treatment == arm).astype(float) - joint[:, column])
        )
        assert np.max(np.abs(parts.d_g[arm] - wrong)) > 0.1


def test_joint_mechanism_correction_is_diaz_theorem_decomposition() -> None:
    """D_M + D_A collapses to e/g {I(A=a, Delta=1) - g}."""
    treatment = np.array([0.0, 1.0, 1.0, 0.0])
    observed = np.array([1.0, 0.0, 1.0, 0.0])
    arm = 1.0
    indicator = (treatment == arm).astype(float)
    g_a = np.array([0.42, 0.57, 0.63, 0.48])
    g_delta = np.array([0.76, 0.69, 0.81, 0.72])
    e = np.array([0.17, -0.23, 0.31, -0.14])
    joint = g_a * g_delta

    collapsed = e / joint * (indicator * observed - joint)
    d_a = e / g_a * (indicator - g_a)
    d_m = indicator * e / joint * (observed - g_delta)

    np.testing.assert_allclose(collapsed, d_a + d_m, rtol=0, atol=2e-16)
    assert np.max(np.abs(collapsed)) > 0.1


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
    """Double robustness through the joint mechanism, at a size where bias would show.

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
