r"""Multi-arm evidence for the two collaborative point-treatment estimators.

**The exact law is a plug-in check and nothing more here**, which is the first thing this
module has to say about itself.  Handed the oracle nuisances,
:mod:`tests.discrete_law_multi` makes every term the multi-arm extension added vanish --
``Qr`` is ``1e-17``, ``gr2`` is numerical zero, and no fluctuation moves -- so recovering the
five parameters to ``2e-15`` there is evidence that the *plug-in and the reported names*
are right and evidence about nothing else.
:func:`test_the_exact_law_leaves_the_new_terms_at_zero` pins that, so a later reader
cannot mistake the oracle fits for coverage of the equations, and each construction below
carries its own nonzero instrument instead:

* equation (9) is compared against an independent ``brentq`` solve of R ``drtmle``'s own
  ``fluctuateG`` score equation, arm by arm, sharing no code with the solver;
* the corrections are exercised on a **misspecified** fit, where ``Qr`` is order ``4e-2``
  and the targeted mechanism visibly leaves the simplex;
* the exit state's arm alignment is checked by recomputing equation (9) from the reported
  arrays and asserting the same expression under a column permutation is *not* solved;
* the ``oat`` treatment design is checked by content on a law where ``Qbar(., W)`` is a
  bijection of ``W``, so a saturated mechanism has to reproduce ``g_0`` exactly.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
import sklearn.linear_model
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator

from cleverly.estimators import CTMLE, DRTMLE
from cleverly.estimators.reduced import ReducedSet
from cleverly.estimators.tmle import correction_parts
from cleverly.exceptions import PositivityWarning
from cleverly.fluctuation.mechanism import solve_armwise_bounded_mechanism
from cleverly.fluctuation.reduced import reduced_mechanism_covariate
from tests import discrete_law_multi as law

COMMON = {
    "outcome_learner": law.OracleMultiOutcome(),
    "treatment_learner": law.OracleMultiTreatment(),
    "cross_fit": False,
    "simultaneous": False,
    "estimands": ("ey", "ate"),
    "random_state": 0,
}

#: The same fit with *learned* nuisances.  Everything about the multi-arm equations that
#: the oracle law cannot see is measured on this one: ``glm`` cannot reproduce this law's
#: non-additive ``Qbar``, so the reduced regressions carry signal and equation (9) has
#: something to solve.
LEARNED = {
    **COMMON,
    "outcome_learner": sklearn.linear_model.LinearRegression(),
    "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
}

REDUCED_LEARNERS = {
    "reduced_outcome_learner": sklearn.linear_model.LinearRegression(),
    "reduced_treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
}

ORACLES = ("ey[0]", "ey[1]", "ey[2]", "ate[0 vs 2]", "ate[1 vs 2]")


@pytest.fixture(scope="module")
def oat_fit():
    return CTMLE(strategy="oat", **COMMON).fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def dr_fit():
    return (
        DRTMLE(**REDUCED_LEARNERS, **COMMON).fit(law.frame(), outcome="Y", treatment="A").single()
    )


@pytest.fixture(scope="module")
def learned_dr_fit():
    """The misspecified three-armed fit -- nonzero ``Qr``, and a mechanism that moves."""
    return (
        DRTMLE(**REDUCED_LEARNERS, **LEARNED).fit(law.frame(), outcome="Y", treatment="A").single()
    )


@pytest.fixture(scope="module")
def learned_bivariate_dr_fit():
    """The same nonzero three-arm witness through the bivariate R branch."""
    return (
        DRTMLE(reduction="bivariate", **REDUCED_LEARNERS, **LEARNED)
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


# --------------------------------------------------------- what the exact law does prove


@pytest.mark.parametrize("oracle", ORACLES)
def test_oat_recovers_each_exact_law_parameter(oat_fit, oracle: str) -> None:
    reported = law.reported_name(oracle)
    assert oat_fit.estimates[reported].psi == pytest.approx(
        float(law.functional(law.PROBS, oracle)), abs=2e-15
    )


@pytest.mark.parametrize("oracle", ORACLES)
def test_drtmle_recovers_each_exact_law_parameter(dr_fit, oracle: str) -> None:
    reported = law.reported_name(oracle)
    assert dr_fit.estimates[reported].psi == pytest.approx(
        float(law.functional(law.PROBS, oracle)), abs=2e-15
    )


def test_the_exact_law_leaves_the_new_terms_at_zero(dr_fit, oat_fit) -> None:
    """The scope of the two tests above, asserted rather than left to be rediscovered.

    Every quantity the multi-arm extension introduced is numerically absent on the oracle
    fits, so no assertion about ``psi`` there can fail because of them.  If a change ever
    makes this test fail, the two exact-law tests have started carrying weight and this
    module's docstring needs rewriting -- which is the point of pinning it.
    """
    reduced = dr_fit.fluctuations["mean"].reduction.reduced
    mechanism = dr_fit.fluctuations["mean"].mechanism
    assert np.abs(reduced.qr).max() < 1e-15
    assert np.abs(reduced.gr2).max() < 1e-15
    np.testing.assert_array_equal(mechanism.epsilon, np.zeros(law.K))
    assert np.abs(mechanism.propensity - dr_fit.nuisance.propensity.values).max() < 1e-15
    np.testing.assert_array_equal(oat_fit.fluctuations["mean"].epsilon, np.zeros(law.K))


# ------------------------------------------------- equation (9) against the R derivation


def test_armwise_mechanism_matches_an_independent_glm_solve() -> None:
    r"""``solve_armwise_bounded_mechanism`` against R ``drtmle``'s ``fluctuateG``.

    ``R/fluctuate.R`` fits, once per level, a weighted logistic regression of
    ``1(A = a)`` on ``H_1 = Qr_a / g_a`` with offset ``logit(g_a)`` and no intercept.  A
    no-intercept logistic MLE in one covariate is the root of a scalar monotone score, so
    ``brentq`` solves it to machine precision without borrowing anything from the module
    under test -- no Newton step, no warm start, no convergence vocabulary in common.
    This is the check the exact law cannot supply: it fails on a changed response, offset,
    covariate or arm alignment, all of which vanish at the truth.

    Bounds are inert, so :func:`solve_bounded_mechanism`'s fast path returns the plain
    unconstrained solve and the two are answering the identical question.
    """
    rng = np.random.default_rng(0)
    n, arms = 400, (0.0, 1.0, 2.0)
    mechanism = rng.dirichlet(np.ones(len(arms)) * 2.0, size=n)
    treatment = np.array([rng.choice(len(arms), p=row) for row in mechanism], dtype=float)
    # Stands in for ``Qr / g*``, signed and of both magnitudes, so no arm's equation is
    # solved by its covariate being small.
    covariate = rng.normal(size=(n, len(arms))) * 0.5
    weights = rng.uniform(0.5, 1.5, size=n)

    fitted = solve_armwise_bounded_mechanism(
        treatment, mechanism, covariate, weights, arms, bounds=(1e-6, 1.0 - 1e-6)
    )

    reference = []
    for column, arm in enumerate(arms):
        response = (treatment == arm).astype(float)
        offset = logit(mechanism[:, column])
        here = covariate[:, column]

        def score(epsilon: float, response=response, offset=offset, here=here) -> float:
            return float(np.sum(weights * here * (response - expit(offset + epsilon * here))))

        reference.append(brentq(score, -50.0, 50.0, xtol=1e-14, rtol=1e-15))

    np.testing.assert_allclose(fitted.epsilon, reference, rtol=0.0, atol=1e-11)
    tilted = np.column_stack(
        [
            expit(logit(mechanism[:, column]) + reference[column] * covariate[:, column])
            for column in range(len(arms))
        ]
    )
    np.testing.assert_allclose(fitted.propensity, tilted, rtol=0.0, atol=1e-12)
    # The signature of the construction: R fluctuates each margin on its own, so the
    # targeted mechanism is no longer a distribution over the arms.
    assert np.abs(np.asarray(fitted.propensity).sum(axis=1) - 1.0).max() > 1e-3


def test_multi_arm_reduced_mechanism_covariate_has_the_r_formula() -> None:
    """``H_1 = Qr / g``, column by column and with no sign -- ``fluctuateG``'s covariate.

    On a nonzero ``Qr``, which is where the binary route's ``signs = (-1, +1)`` would show:
    at the truth ``Qr`` vanishes and every sign convention agrees.
    """
    propensity = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    qr = np.array([[0.12, -0.06, 0.15], [-0.04, 0.21, -0.08]])
    reduced = ReducedSet(
        qr=qr,
        gr1=np.full_like(qr, 0.5),
        gr2=np.zeros_like(qr),
        arms=(0.0, 1.0, 2.0),
        g_bounds=(0.05, 0.95),
    )
    actual = reduced_mechanism_covariate(reduced, propensity, bounds=(0.05, 0.95))
    np.testing.assert_array_equal(actual, qr / propensity)


def test_the_multi_arm_covariate_bounds_the_mechanism_column_by_column() -> None:
    """Not the binary complement rule, which is only defined at two arms."""
    propensity = np.array([[0.01, 0.30, 0.69], [0.10, 0.02, 0.88]])
    qr = np.ones_like(propensity)
    reduced = ReducedSet(
        qr=qr,
        gr1=np.full_like(qr, 0.5),
        gr2=np.zeros_like(qr),
        arms=(0.0, 1.0, 2.0),
        g_bounds=(0.05, 0.95),
    )
    actual = reduced_mechanism_covariate(reduced, propensity, bounds=(0.05, 0.95))
    np.testing.assert_allclose(actual, qr / np.clip(propensity, 0.05, 0.95))


# ------------------------------------------- the corrections, where they are not degenerate


def test_drtmle_corrections_are_nonzero_and_solved_under_misspecification(
    learned_dr_fit,
) -> None:
    """The identity, on a fit where both sides of it have something in them.

    ``glm`` cannot fit this law's ``Qbar``, so ``Qr`` and ``gr2`` carry real signal and
    equation (9) has a root away from ``epsilon = 0``.  Without this fixture every
    correction assertion in this module compares zero against zero.
    """
    fluctuation = learned_dr_fit.fluctuations["mean"]
    reduced = fluctuation.reduction.reduced
    assert np.abs(reduced.qr).max() > 1e-3
    assert np.abs(reduced.gr2).max() > 1e-3

    mechanism = fluctuation.mechanism
    assert mechanism is not None
    assert mechanism.propensity.shape == (law.N, law.K)
    assert mechanism.score.shape == (law.K,)
    # The mechanism genuinely moved, and moved off the simplex, which is the armwise
    # construction's fingerprint and the thing a renormalising implementation would lose.
    moved = np.abs(mechanism.propensity - learned_dr_fit.nuisance.propensity.values).max()
    assert moved > 1e-3
    assert np.abs(np.asarray(mechanism.propensity).sum(axis=1) - 1.0).max() > 1e-4

    assert learned_dr_fit.validation.score_check().passed
    check = learned_dr_fit.validation.correction_check()
    assert check.passed
    assert {row.arm for row in check.rows} == set(learned_dr_fit.nuisance.arms)

    # `check.passed` compares two *means*, and a solved equation drives both to zero -- so
    # it would read the same on a curve whose corrections were identically zero.  The
    # correction this fit actually subtracts is pointwise substantial at every arm, which
    # is the part that has to be true for the interval to be the doubly-robust one.
    scaled = learned_dr_fit.nuisance.scaler.scale(learned_dr_fit.data.outcome)
    parts = correction_parts(
        learned_dr_fit.data,
        learned_dr_fit.nuisance,
        fluctuation,
        fluctuation.targeted,
        scaled,
    )
    assert parts is not None
    for arm in learned_dr_fit.nuisance.arms:
        assert np.abs(parts.total()[arm]).max() > 1e-3
        assert abs(float(np.mean(parts.total()[arm]))) < 1e-8


def test_multi_arm_bivariate_corrections_are_nonzero_and_solved(
    learned_bivariate_dr_fit,
) -> None:
    """Pinned R's bivariate branch is one joint reduction and correction per arm.

    This is deliberately the misspecified fit: at the exact law ``Qr`` and every
    correction vanish, so merely recovering the three means cannot distinguish an
    implemented armwise correction from a no-op.
    """
    fit = learned_bivariate_dr_fit
    fluctuation = fit.fluctuations["mean"]
    reduced = fluctuation.reduction.reduced
    assert reduced.reduction == "bivariate"
    assert reduced.gr1.shape == (law.N, law.K)
    assert np.isnan(reduced.gr2).all()
    assert np.abs(reduced.qr).max() > 1e-3

    mechanism = fluctuation.mechanism
    assert mechanism is not None
    assert mechanism.propensity.shape == (law.N, law.K)
    assert fit.validation.score_check().passed
    assert fit.validation.correction_check().passed

    scaled = fit.nuisance.scaler.scale(fit.data.outcome)
    parts = correction_parts(fit.data, fit.nuisance, fluctuation, fluctuation.targeted, scaled)
    assert parts is not None
    for arm in fit.nuisance.arms:
        assert np.abs(parts.d_q[arm]).max() > 1e-3
        assert abs(float(np.mean(parts.total()[arm]))) < 1e-8


def test_multi_arm_exit_state_solves_each_arms_equation(learned_dr_fit) -> None:
    r"""Equation (9) recomputed from the reported arrays, with a permutation witness.

    .. math:: P_n\bigl[w\,Q_r(a, W)/g^*_a\,\{1(A = a) - g^*_a\}\bigr] = 0

    written out here from ``fluctuateG``'s definition rather than read off the
    fluctuation, so agreement means the reported ``Qr`` column, the reported mechanism
    column and the arm each is claimed to belong to line up.  The permuted expression is
    the control: were any per-arm quantity read at the wrong arm, the *solved* score would
    be the permuted one and the assertions would swap places.
    """
    fluctuation = learned_dr_fit.fluctuations["mean"]
    reduced = fluctuation.reduction.reduced
    lower, upper = fluctuation.reduction.bounds
    mechanism = np.clip(np.asarray(fluctuation.mechanism.propensity, dtype=float), lower, upper)
    qr = np.asarray(reduced.qr, dtype=float)
    weights = np.asarray(learned_dr_fit.data.weights, dtype=float)
    treatment = np.asarray(learned_dr_fit.data.treatment, dtype=float)

    def scores(columns: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.sum(
                    weights
                    * (columns[:, j] / mechanism[:, j])
                    * ((treatment == arm).astype(float) - mechanism[:, j])
                )
                / weights.sum()
                for j, arm in enumerate(reduced.arms)
            ]
        )

    solved = scores(qr)
    assert np.abs(solved).max() < 1e-8
    np.testing.assert_allclose(solved, fluctuation.mechanism.score, rtol=0.0, atol=1e-8)

    permuted = scores(np.roll(qr, 1, axis=1))
    assert np.abs(permuted).min() > 1e-6
    assert np.abs(permuted).max() > 1e-4


def test_multi_arm_single_guard_uses_the_initial_mechanism_matrix(learned_dr_fit) -> None:
    fit = (
        DRTMLE(guard=("g",), **REDUCED_LEARNERS, **LEARNED)
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )
    assert fit.fluctuations["mean"].mechanism is None
    assert fit.validation.correction_check().passed
    # And the other guard did fluctuate, so the branch above is a branch rather than the
    # only thing this construction ever does.
    assert learned_dr_fit.fluctuations["mean"].mechanism is not None


def test_multi_arm_truncation_is_reported_per_row_not_per_cell() -> None:
    """``clipped`` counts rows at ``K`` arms as it does at two.

    The mechanism is bounded column by column here, so a row can clip on more than one
    arm.  Counting the cells made ``correction_check`` print "binds on up to 3800 row(s)
    of 2000" on this very fixture -- a diagnostic that reads as broken when it is not.
    """
    estimator = DRTMLE(guard=("g",), g_bounds=(0.34, 0.36), **REDUCED_LEARNERS, **LEARNED)
    with pytest.warns(PositivityWarning, match="outside the truncation bounds"):
        fit = estimator.fit(law.frame(), outcome="Y", treatment="A").single()
    check = fit.validation.correction_check()
    # The bound is narrower than any margin this law produces, so every row clips on at
    # least two arms -- the case that used to over-count.
    assert check.clipped == check.n


# ----------------------------------------------------------- the outcome-adaptive design


class _SaturatedCategorical(BaseEstimator):
    """``P(A = a | design)`` as the weighted class frequencies within each design row.

    A ``K``-class sibling of :class:`tests.discrete_law_longitudinal.CellMeans`, and here
    for the same reason: on a sample that realises its law exactly, an unpenalised
    saturated fit *is* the oracle, so "did the mechanism come out right" becomes an exact
    assertion rather than a tolerance.

    :attr:`designs` collects what every ``fit`` was handed, so one test can check what the
    treatment model was *given* and another what it learned from it.  It is a **class**
    attribute rather than a constructor argument because ``sklearn.clone`` rebuilds an
    estimator from ``get_params`` and deep-copies anything that is not itself an
    estimator: a list passed in would arrive at the fitted clone as a copy, and the caller
    would watch an empty one.
    """

    designs: ClassVar[list[np.ndarray]] = []

    def __init__(self, record: bool = False) -> None:
        self.record = record

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _SaturatedCategorical:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if self.record:
            type(self).designs.append(design.copy())
        self.classes_ = np.unique(target)
        keys, inverse = np.unique(np.round(design, 9), axis=0, return_inverse=True)
        totals = np.zeros((keys.shape[0], self.classes_.size))
        for column, level in enumerate(self.classes_):
            totals[:, column] = np.bincount(
                inverse, weights=weights * (target == level), minlength=keys.shape[0]
            )
        sizes = totals.sum(axis=1, keepdims=True)
        self.keys_ = keys
        self.frequencies_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        self.default_ = totals.sum(axis=0) / totals.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.round(np.asarray(X, dtype=float), 9)
        out = np.tile(self.default_, (design.shape[0], 1))
        for position, key in enumerate(self.keys_):
            out[np.all(design == key, axis=1)] = self.frequencies_[position]
        return out


def test_oat_fits_the_treatment_model_on_the_arm_specific_qbar_matrix() -> None:
    """The design is ``[Qbar(a, W)]`` by content, not merely by the names recorded.

    ``ctmle3::LF_oat`` builds its task from ``Q%sW`` over the levels and nothing else, so
    what has to be checked is the matrix -- a design of the *observed* predictions, or of
    the covariates, would leave ``treatment_features`` reading exactly the same.
    """
    _SaturatedCategorical.designs.clear()
    fit = (
        CTMLE(
            strategy="oat",
            **{**COMMON, "treatment_learner": _SaturatedCategorical(record=True)},
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )
    seen = list(_SaturatedCategorical.designs)
    _SaturatedCategorical.designs.clear()

    expected = np.column_stack([fit.nuisance.outcome.arms[arm] for arm in fit.nuisance.arms])
    # The shared nuisance pass is outcome-first for every CTMLE strategy, so this is the
    # only treatment-model fit: no ordinary g(W) is fitted and discarded beforehand.
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], expected)
    assert expected.shape == (law.N, law.K)
    # Not the observed-arm prediction repeated, and not the covariates: both are the
    # plausible slips, and both would leave `treatment_features` reading the same.
    assert not np.array_equal(seen[0][:, 0], fit.nuisance.outcome.observed)
    assert np.unique(seen[0], axis=0).shape[0] == 3


def test_oat_recovers_a_mechanism_generated_by_qbar() -> None:
    r"""When ``sigma(Qbar) = sigma(W)``, the outcome-adaptive mechanism *is* ``g_0``.

    This law's three ``Qbar(., w)`` triples are distinct, so the design generates the same
    sigma-algebra ``W`` does and the projection ``LF_oat`` takes loses nothing.  A
    saturated fit on a sample that realises the law exactly then reproduces ``g_0`` to
    machine precision -- which is the assertion that fails if the design is zeroed,
    permuted, or replaced by the observed predictions, none of which the exact-law
    parameter checks can see.

    It is also the boundary of what ``oat`` promises: the equality holds *because* the
    outcome regression happens to carry all of ``W`` here.  Where it does not, the fitted
    mechanism is a strictly coarser projection and consistency for ``g_0`` is gone -- see
    the module docstring of :mod:`cleverly.estimators.ctmle`.
    """
    fit = (
        CTMLE(strategy="oat", **{**COMMON, "treatment_learner": _SaturatedCategorical()})
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )
    covariate = np.rint(fit.data.covariates[:, 0]).astype(int)
    truth = np.column_stack(
        [law.G_EXACT[covariate, law.ARM_OF_CODE[int(code)]] for code in fit.nuisance.arms]
    )
    np.testing.assert_allclose(fit.nuisance.propensity.values, truth, rtol=0.0, atol=1e-15)
    # Which is worth saying out loud, because the assertion above would also pass on a
    # law whose ``g_0`` happened to be constant.
    assert np.unique(truth.round(12), axis=0).shape[0] == 3


def test_oat_records_the_shared_treatment_model_api(oat_fit) -> None:
    record = oat_fit.extra["ctmle"]
    assert record.strategy == "oat"
    assert record.treatment_features == ("Qbar[high]", "Qbar[low]", "Qbar[mid]")
    assert record.treatment_risk_selected == record.treatment_risk
    assert oat_fit.nuisance.propensity.values.shape == (law.N, law.K)
    np.testing.assert_allclose(oat_fit.nuisance.propensity.values.sum(axis=1), 1.0, atol=1e-15)


def test_oat_refuses_selector_only_controls() -> None:
    with pytest.raises(ValueError, match="penalty= configure selector strategies"):
        CTMLE(strategy="oat", penalty=False)
    with pytest.raises(ValueError, match="selection_folds= configure selector strategies"):
        CTMLE(strategy="oat", selection_folds=3)


def test_oat_accepts_a_selector_setting_written_at_its_default() -> None:
    """Which is what a reloaded fit hands back.

    Whole-result persistence retains every constructor setting, so a restored ``oat``
    fit supplies all four selector-only settings
    explicitly.  The guard is therefore on the *value* rather than on whether the argument
    was written, and it reads the defaults off the signature so it cannot invert when one
    of them changes.
    """
    estimator = CTMLE(
        strategy="oat", selection_folds=5, selection_inner_folds=2, loss="auto", penalty=True
    )
    assert estimator.strategy == "oat"
