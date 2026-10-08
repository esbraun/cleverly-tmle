r"""The per-arm outcome-adaptive design of ``CTMLE(strategy="oat")``, checked by content.

``oat_design="per_arm"`` regresses ``1{A = a}`` on the one column ``Qbar_n(a, W)`` for each
arm.  Column ``a`` of the mechanism holds ``P(A = a | Qbar_n(a, W))``, off the simplex.
Benkeser, Cai and van der Laan (2020), Theorem 1, give the influence curve of that
construction for one treatment-specific mean:
``D(Q_0, G_0(. | Q_0)) = 1{A = a} / G_0(a | Qbar_0(a, W)) (Y - Qbar_0(a, W)) + Qbar_0(a, W) - psi_a``.
It is not the efficient influence function.

The checks, and the instrument that makes each one able to fail:

* the design, on :mod:`tests.discrete_law_oat`, where the per-arm design of arm 0 ties two
  covariate values that the shared design separates;
* the reported curve equals ``D(Q_0, G_0(. | Q_0))`` row by row on that law, and differs
  from the efficient influence function on the tied rows;
* a longhand of the in-sample construction (the binary regressions, the bounds, one scalar
  fluctuation per arm, the curve, the standard error and the interval) on two sampled laws;
* a longhand of the cross-fitted construction with fixed weights: each fold's outcome
  regression and each fold's per-arm mechanism refitted on that fold's training rows, then the
  pooled weighted fluctuation;
* each arm's coefficient equals a scalar fluctuation on that arm's column alone;
* the joint covariance of the stacked curves, with a nonzero cross-arm covariance;
* fixed integer weights equal duplicated rows;
* the per-arm diagnostics read each arm's own column at two and at three arms.

Every check has a mutation control in :class:`TestEachMutationFailsItsCheck`.  Each control
breaks one line of the shipped source through
:func:`tests.unit._source_mutation.mutate` and requires the check to fail.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CollaborativeTMLEMethod
from cleverly._inference_status import NON_INFERENTIAL
from cleverly.datasets import make_missing_outcome
from cleverly.estimators import CTMLE
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators import targeting as targeting_module
from cleverly.fluctuation.iterative import InitialFit, solve_fluctuation
from cleverly.fluctuation.submodel import Submodel, mean_submodel
from cleverly.inference.multiplier import multiplier_critical_value
from cleverly.learners.crossfit import SplitPlan, random_partition
from cleverly.targets import builtin
from tests import discrete_law_oat as law
from tests.studies.oat_per_arm_laws import (
    BINARY_ACTIVE,
    THREE_ARM_ACTIVE,
    OatLaw,
    arm_interactions,
    cubic_logit_learner,
    interaction_outcome,
    logit_basis,
)
from tests.unit._source_mutation import mutate

pytestmark = pytest.mark.xdist_group("outcome_adaptive_per_arm")

#: The submodel bound of the targeting step, ``R``'s ``tmle`` convention.
ALPHA = 0.9995
#: The ``1 - 0.05 / 2`` normal quantile.
Z = float(norm.ppf(0.975))
#: The cross-fitted longhand's fold count and split seed.
FOLDS = 3
SPLIT_SEED = 31


def exact_fit(design: str, learner: Any = None, shift: float = 0.0, **extra: Any) -> Any:
    """A fit on the exact law, with the oracle outcome model moved by ``shift``."""
    return (
        CTMLE(
            strategy="oat",
            oat_design=design,
            outcome_learner=law.OracleOutcome(shift=shift),
            treatment_learner=law.SaturatedCategorical() if learner is None else learner,
            cross_fit=False,
            simultaneous=False,
            estimands=("ey", "ate"),
            random_state=0,
            **extra,
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


def glm_fit(frame: pd.DataFrame, **extra: Any) -> Any:
    """The in-sample per-arm fit with the study's learners.

    The outcome model is the correctly specified ``A x W`` logistic GLM of the study laws.
    A main-terms outcome model would make the two designs coincide at two arms: both
    ``logit Qbar(a, W)`` are then shifts of one index, and a cubic in either spans the same
    functions.
    """
    k = frame["A"].nunique()
    settings: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": interaction_outcome(k),
        "treatment_learner": cubic_logit_learner(),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ey0", "ey1", "ate") if k == 2 else ("ey", "ate"),
        **({} if k == 2 else {"reference": "0"}),
        **extra,
    }
    roles = {key: settings.pop(key) for key in ("weights",) if key in settings}
    return CTMLE(**settings).fit(frame, outcome="Y", treatment="A", **roles).single()


def binary_frame(n: int = 500, seed: int = 11) -> pd.DataFrame:
    return BINARY_ACTIVE.sample(n, seed)[0]


def three_arm_frame(n: int = 600, seed: int = 12) -> pd.DataFrame:
    return THREE_ARM_ACTIVE.sample(n, seed)[0]


@pytest.fixture(scope="module")
def binary() -> Any:
    return glm_fit(binary_frame())


@pytest.fixture(scope="module")
def three_arm() -> Any:
    return glm_fit(three_arm_frame(), estimands=("ey", "ate", "rr"))


def _mean_name(data: Any, arm: float) -> str:
    if data.is_binary_treatment:
        return f"ey{int(arm)}"
    return f"ey[{data.arm_label(arm)}]"


def _contrast(data: Any, stem: str, arm: float, reference: float) -> str:
    return f"{stem}[{data.arm_label(arm)} vs {data.arm_label(reference)}]"


def _reference(result: Any) -> float:
    """The reference arm the fit's contrasts name."""
    data = result.data
    if data.is_binary_treatment:
        return float(data.arm_codes[0])
    for name in result.estimates:
        match = re.fullmatch(r"ate\[(.+) vs (.+)\]", name)
        if match:
            return next(code for code in data.arm_codes if data.arm_label(code) == match[2])
    raise AssertionError("no contrast")


# ------------------------------------------------------------------------- the checks


def check_tied_projection() -> None:
    """Arm 0's design ties ``W = 0`` and ``W = 1``; the shared design separates them."""
    frame = law.frame()
    w = frame["W"].to_numpy().astype(int)
    per_arm = exact_fit("per_arm").nuisance.propensity
    shared = exact_fit("shared").nuisance.propensity
    assert not per_arm.simplex
    assert shared.simplex
    np.testing.assert_allclose(per_arm.arm(0.0), law.projection(0)[w], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(shared.arm(0.0), (1.0 - law.G1)[w], rtol=0.0, atol=1e-12)
    # The nonzero witness: the two designs differ on the tied rows by 0.15.
    assert np.max(np.abs(per_arm.values - shared.values)) > 0.05
    # Arm 1's design is injective, so both designs recover g_0 there.
    np.testing.assert_allclose(per_arm.arm(1.0), law.G1[w], rtol=0.0, atol=1e-12)


def check_curve_is_theorem_ones() -> None:
    """The reported curve is ``D(Q_0, G_0(. | Q_0))`` row by row, and not the EIF."""
    frame = law.frame()
    w = frame["W"].to_numpy().astype(int)
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy()
    fit = exact_fit("per_arm")
    for arm in (0, 1):
        g = law.projection(arm)[w]
        q = law.Q[w, arm]
        expected = (a == arm) / g * (y - q) + q - law.TRUTH[arm]
        reported = np.asarray(fit[f"ey[{arm}]"].influence_curve)
        np.testing.assert_allclose(reported, expected, rtol=0.0, atol=1e-12)
    g0 = (1.0 - law.G1)[w]
    q = law.Q[w, 0]
    eif = (a == 0) / g0 * (y - q) + q - law.TRUTH[0]
    reported = np.asarray(fit["ey[0]"].influence_curve)
    tied = w <= 1
    # The witness: on the tied rows the curve is not the efficient influence function.
    assert np.max(np.abs(reported - eif)[tied]) > 1e-3
    np.testing.assert_allclose(reported[~tied], eif[~tied], rtol=0.0, atol=1e-12)


def _irls(design: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """An unpenalized weighted logistic regression by Newton's method; the coefficients.

    The intercept comes first.
    """
    x = np.column_stack([np.ones(design.shape[0]), design])
    w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        p = expit(x @ beta)
        gradient = x.T @ (w * (target - p))
        hessian = (x * (w * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return beta


def _predict(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(expit(beta[0] + design @ beta[1:]))


def _shrink(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1.0 - ALPHA, ALPHA)


def _target(
    q: dict[float, np.ndarray], g: np.ndarray, result: Any
) -> dict[float, tuple[float, np.ndarray]]:
    """One weighted scalar logistic fluctuation per arm, then the mean and the curve.

    ``q`` holds the initial regressions and ``g`` the untruncated per-arm mechanism.
    """
    data = result.data
    a = np.asarray(data.treatment, dtype=float)
    y = np.asarray(data.outcome, dtype=float)
    w = np.asarray(data.weights, dtype=float)
    lower, upper = result.config.g_bounds
    out: dict[float, tuple[float, np.ndarray]] = {}
    for j, arm in enumerate(data.arm_codes):
        hit = (a == arm).astype(float)
        h = 1.0 / np.clip(g[:, j], lower, upper)
        offset = logit(_shrink(q[arm]))
        epsilon = 0.0
        for _ in range(200):
            fitted = expit(offset + epsilon * h)
            score = float(np.sum(w * hit * h * (y - fitted)))
            curvature = float(np.sum(w * hit * h**2 * fitted * (1.0 - fitted)))
            step = score / curvature
            epsilon += step
            if abs(step) < 1e-15:
                break
        targeted = _shrink(expit(offset + epsilon * h))
        psi = float(np.average(targeted, weights=w))
        curve = w / np.mean(w) * (hit * h * (y - targeted) + targeted - psi)
        out[arm] = (psi, curve)
    return out


def longhand(result: Any) -> dict[float, tuple[float, np.ndarray]]:
    """Steps 1-7 of Benkeser, Cai and van der Laan (2020), Section 3, arm by arm, in sample.

    Step 1 is the fit's own initial ``Qbar_n``, which is the input of the construction.
    Every later step is written here: the binary regression of ``1{A = a}`` on the cubic
    basis of ``logit Qbar_n(a, W)``, the bound, one scalar logistic fluctuation by Newton's
    method, the targeted mean and the curve.
    """
    data = result.data
    a = np.asarray(data.treatment, dtype=float)
    q = {arm: np.asarray(result.nuisance.outcome.arms[arm], dtype=float) for arm in data.arm_codes}
    g = np.column_stack(
        [
            _predict(logit_basis(q[arm]), _irls(logit_basis(q[arm]), (a == arm).astype(float)))
            for arm in data.arm_codes
        ]
    )
    return _target(q, g, result)


def _assert_matches(result: Any, hand: dict[float, tuple[float, np.ndarray]]) -> None:
    """The fit's means, contrasts, curves, standard errors and intervals are the longhand's."""
    data = result.data
    n = data.n
    for arm, (psi, curve) in hand.items():
        estimate = result[_mean_name(data, arm)]
        assert estimate.psi == pytest.approx(psi, abs=1e-8)
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-8)
        assert estimate.std_error == pytest.approx(np.sqrt(np.var(curve, ddof=1) / n), abs=1e-8)
    reference = _reference(result)
    for arm in data.arm_codes:
        if arm == reference:
            continue
        name = "ate" if data.is_binary_treatment else _contrast(data, "ate", arm, reference)
        estimate = result[name]
        psi = hand[arm][0] - hand[reference][0]
        se = np.sqrt(np.var(hand[arm][1] - hand[reference][1], ddof=1) / n)
        assert estimate.psi == pytest.approx(psi, abs=1e-8)
        assert estimate.std_error == pytest.approx(se, abs=1e-8)
        assert estimate.ci == pytest.approx((psi - Z * se, psi + Z * se), abs=1e-8)


def check_longhand(which: str) -> None:
    if which == "binary":
        result = glm_fit(binary_frame())
    else:
        result = glm_fit(three_arm_frame(), estimands=("ey", "ate", "rr"))
    _assert_matches(result, longhand(result))


def cv_fit(source: OatLaw, n: int = 600, seed: int = 31) -> tuple[Any, np.ndarray]:
    """A cross-fitted per-arm fit with fixed weights ``0.5 + 1{W1 > 0}`` on a declared split."""
    frame = source.sample(n, seed)[0]
    frame = frame.assign(wt=0.5 + (frame["W1"] > 0).astype(float))
    drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
    result = glm_fit(
        frame,
        cross_fit=True,
        n_folds=FOLDS,
        split_plan=SplitPlan.from_folds([drawn]),
        weights="wt",
    )
    return result, np.asarray(drawn.assignment)


def cv_longhand(
    result: Any, assignment: np.ndarray
) -> tuple[dict[float, np.ndarray], np.ndarray, dict[float, tuple[float, np.ndarray]]]:
    """The cross-fitted construction, fold by fold, with the fit's weights.

    Fold ``v``'s weighted ``A x W`` logistic GLM is fitted on its training rows and evaluated
    at every row and arm.  Arm ``a``'s mechanism is the weighted cubic-logit regression of
    ``1{A = a}`` on that fold's ``Qbar_v(a, W)`` over the training rows, predicted on the
    held-out rows.  The pooled weighted fluctuation follows.
    """
    data = result.data
    arms = data.arm_codes
    k = len(arms)
    a = np.asarray(data.treatment, dtype=float)
    y = np.asarray(data.outcome, dtype=float)
    w = np.asarray(data.weights, dtype=float)
    design = arm_interactions(data.treatment_design(), indicators=k - 1)
    q = {arm: np.empty(data.n) for arm in arms}
    g = np.empty((data.n, k))
    for fold in range(FOLDS):
        test = assignment == fold
        train = ~test
        beta = _irls(design[train], y[train], w[train])
        for j, arm in enumerate(arms):
            counterfactual = arm_interactions(data.counterfactual_design(arm), indicators=k - 1)
            q_all = np.clip(_predict(counterfactual, beta), 0.0, 1.0)
            q[arm][test] = q_all[test]
            basis = logit_basis(q_all)
            gamma = _irls(basis[train], (a[train] == arm).astype(float), w[train])
            g[test, j] = np.clip(_predict(basis[test], gamma), 0.0, 1.0)
    return q, g, _target(q, g, result)


def check_cv_longhand(source: OatLaw) -> None:
    result, assignment = cv_fit(source)
    q, g, hand = cv_longhand(result, assignment)
    for arm in result.data.arm_codes:
        np.testing.assert_allclose(result.nuisance.outcome.arms[arm], q[arm], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.nuisance.propensity.values, g, rtol=0.0, atol=1e-8)
    _assert_matches(result, hand)


def _scalar_submodel(submodel: Submodel, column: int, arm: float) -> Submodel:
    """Column ``column`` of a mean submodel alone, as a one-arm submodel."""
    return Submodel(
        submodel.observed[:, [column]],
        {level: values[:, [column]] for level, values in submodel.arms.items()},
        (submodel.names[column],),
        "mean",
        {arm: 0},
    )


def check_scalar_fluctuation(kind: str) -> None:
    """At convergence the joint maximum equals the ``K`` scalar maxima, column by column."""
    result = glm_fit(three_arm_frame(), fluctuation=kind)
    data = result.data
    fluctuation = result.fluctuations["mean"]
    bounded = result.nuisance.propensity.bounded(result.config.g_bounds)
    joint = mean_submodel(data.treatment, bounded, arms=data.arm_codes)
    initial = InitialFit(result.nuisance.outcome.observed, dict(result.nuisance.outcome.arms))
    for column, arm in enumerate(data.arm_codes):
        scalar = solve_fluctuation(
            data.outcome,
            initial,
            _scalar_submodel(joint, column, arm),
            data.weights,
            kind=kind,
            warn=False,
        )
        assert fluctuation.epsilon[column] == pytest.approx(scalar.epsilon[0], abs=1e-9)
        np.testing.assert_allclose(
            fluctuation.targeted.arms[arm], scalar.targeted.arms[arm], rtol=0.0, atol=1e-9
        )


def _arm_scores(result: Any, *, targeted: bool) -> dict[float, float]:
    """``mean(1{A = a} / g_a (Y - Q))`` per arm, at the targeted or the initial fit."""
    data = result.data
    bounded = result.nuisance.propensity.bounded(result.config.g_bounds)
    a = np.asarray(data.treatment, dtype=float)
    y = np.asarray(data.outcome, dtype=float)
    fit = result.fluctuations["mean"].targeted if targeted else result.nuisance.outcome
    return {
        arm: float(np.mean((a == arm) / bounded[:, j] * (y - fit.arms[arm])))
        for j, arm in enumerate(data.arm_codes)
    }


def check_scores() -> None:
    result = glm_fit(three_arm_frame(), estimands=("ey", "ate", "rr"))
    initial = _arm_scores(result, targeted=False)
    # The witness: the initial scores differ in sign, so no single shared coefficient
    # could solve every arm's equation at once.
    assert min(initial.values()) < -1e-4
    assert max(initial.values()) > 1e-4
    for arm, score in _arm_scores(result, targeted=True).items():
        assert abs(score) < 1e-8, arm


def check_joint_covariance() -> None:
    result = glm_fit(binary_frame())
    n = result.data.n
    d1 = np.asarray(result["ey1"].influence_curve)
    d0 = np.asarray(result["ey0"].influence_curve)
    # The witness: the two arm curves covary, so a block-diagonal covariance is wrong.
    assert abs(np.cov(d1, d0)[0, 1]) > 1e-3 * np.sqrt(np.var(d1) * np.var(d0))
    assert result["ate"].std_error == pytest.approx(np.sqrt(np.var(d1 - d0, ddof=1) / n), rel=1e-12)


def check_weights_equal_duplicates() -> None:
    frame = binary_frame(300, 21)
    counts = 1 + np.arange(len(frame)) % 3
    weighted = glm_fit(frame.assign(wt=counts.astype(float)), weights="wt")
    duplicated = glm_fit(frame.loc[frame.index.repeat(counts)].reset_index(drop=True))
    assert weighted.inference_status == "influence_curve"
    for name in ("ey0", "ey1", "ate"):
        assert weighted[name].psi == pytest.approx(duplicated[name].psi, abs=1e-7)
    # The witness: the weights move the estimate, so equality is not the unweighted fit.
    unweighted = glm_fit(frame)
    assert abs(unweighted["ate"].psi - weighted["ate"].psi) > 1e-4


# -------------------------------------------------------------------------- the tests


class TestTheDesignIsPerArm:
    def test_the_tied_arm_takes_the_pooled_projection(self) -> None:
        check_tied_projection()

    def test_each_arm_fits_on_its_own_prediction_alone(self) -> None:
        law.SaturatedCategorical.designs.clear()
        fit = exact_fit("per_arm", law.SaturatedCategorical(record=True))
        seen = list(law.SaturatedCategorical.designs)
        law.SaturatedCategorical.designs.clear()
        assert len(seen) == 2
        for design, arm in zip(seen, fit.nuisance.arms, strict=True):
            assert design.shape == (law.N, 1)
            np.testing.assert_array_equal(design[:, 0], fit.nuisance.outcome.arms[arm])
        assert not np.array_equal(seen[0], seen[1])

    def test_the_point_estimates_of_the_two_designs_differ(self) -> None:
        """At the exact ``Qbar`` both equal the truth; a moved ``Qbar`` separates them.

        A saturated mechanism would hide the difference: both designs then give an
        unbiased estimate on this law, the per-arm one because the error of a constant
        shift is a function of its design.  A logistic mechanism is not saturated, so each
        design leaves its own bias.
        """
        learner = LogisticRegression(C=np.inf, max_iter=1000)
        for design in ("per_arm", "shared"):
            fit = exact_fit(design, learner)
            assert fit["ey[0]"].psi == pytest.approx(law.TRUTH[0], abs=1e-12)
            assert fit["ey[1]"].psi == pytest.approx(law.TRUTH[1], abs=1e-12)
        per_arm = exact_fit("per_arm", learner, shift=-0.1)
        shared = exact_fit("shared", learner, shift=-0.1)
        assert abs(per_arm["ate"].psi - shared["ate"].psi) > 1e-3


def test_the_curve_is_theorem_ones() -> None:
    check_curve_is_theorem_ones()


@pytest.mark.parametrize("which", ["binary", "three_arm"])
def test_the_fit_is_the_longhand_construction(which: str) -> None:
    check_longhand(which)


@pytest.mark.parametrize("source", [BINARY_ACTIVE, THREE_ARM_ACTIVE], ids=lambda law: law.name)
def test_the_cross_fitted_fit_is_the_longhand_construction(source: OatLaw) -> None:
    check_cv_longhand(source)


@pytest.mark.parametrize("kind", ["logistic", "linear"])
def test_each_arm_coefficient_is_a_scalar_fluctuation(kind: str) -> None:
    check_scalar_fluctuation(kind)


def test_the_one_step_update_solves_each_arms_equation() -> None:
    result = glm_fit(three_arm_frame(), targeting="one_step")
    for arm, score in _arm_scores(result, targeted=True).items():
        assert abs(score) < 1e-8, arm


def test_an_ey1_only_fit_is_the_full_fit() -> None:
    """A plumbing check: the design does not depend on the requested estimands."""
    frame = binary_frame()
    alone = glm_fit(frame, estimands=("ey1",))
    full = glm_fit(frame)
    assert alone["ey1"].psi == full["ey1"].psi
    np.testing.assert_array_equal(alone["ey1"].influence_curve, full["ey1"].influence_curve)


def test_every_arms_score_is_solved() -> None:
    check_scores()


def test_the_contrast_reads_the_joint_covariance() -> None:
    check_joint_covariance()


def test_the_three_arm_ratio_and_band_read_the_stacked_curves(three_arm: Any) -> None:
    data = three_arm.data
    reference = _reference(three_arm)
    n = data.n
    for arm in data.arm_codes:
        if arm == reference:
            continue
        top = three_arm[_mean_name(data, arm)]
        bottom = three_arm[_mean_name(data, reference)]
        curve = (
            np.asarray(top.influence_curve) / top.psi
            - np.asarray(bottom.influence_curve) / bottom.psi
        )
        ratio = three_arm[_contrast(data, "rr", arm, reference)]
        assert ratio.log_psi == pytest.approx(np.log(top.psi / bottom.psi), rel=1e-12)
        assert ratio.std_error == pytest.approx(np.sqrt(np.var(curve, ddof=1) / n), rel=1e-10)

    banded = glm_fit(three_arm_frame(), estimands=("ey", "ate"), simultaneous=True)
    estimates = list(banded.estimates.values())
    curves = np.column_stack([np.asarray(e.influence_curve) for e in estimates])
    errors = np.array([e.std_error for e in estimates])
    expected = multiplier_critical_value(
        curves,
        errors,
        n=n,
        alpha=banded.estimator.alpha_sig,
        n_replicates=banded.estimator.n_multiplier,
        kind=banded.estimator.multiplier_kind,
        random_state=banded.estimator.random_state,
    )
    assert banded.simultaneous is not None
    assert banded.simultaneous.critical_value == pytest.approx(expected, rel=1e-12)


def test_integer_weights_equal_duplicated_rows() -> None:
    check_weights_equal_duplicates()


# ----------------------------------------------------------------- the mutation controls

_CTMLE = (ctmle_module, CTMLE)
_IN_SAMPLE = "_outcome_adaptive_nuisances"
_CROSS_FITTED = "_cross_fit_outcome_adaptive"


def _shared_coefficient(monkeypatch: pytest.MonkeyPatch) -> None:
    """One coefficient for every arm: the joint submodel collapsed to one column."""
    original = targeting_module.solve_fluctuation

    def collapsed(scaled: Any, initial: Any, submodel: Any, weights: Any, *args: Any, **kw: Any):  # type: ignore[no-untyped-def]
        if submodel.group != "mean" or submodel.dim < 2:
            return original(scaled, initial, submodel, weights, *args, **kw)
        one = Submodel(
            submodel.observed.sum(axis=1, keepdims=True),
            {arm: values.sum(axis=1, keepdims=True) for arm, values in submodel.arms.items()},
            ("h",),
            "mean",
            dict.fromkeys(submodel.arms, 0),
        )
        fit = original(scaled, initial, one, weights, *args, **kw)
        return replace(
            fit,
            epsilon=np.repeat(fit.epsilon, submodel.dim),
            score=np.repeat(fit.score, submodel.dim),
            names=submodel.names,
        )

    monkeypatch.setattr(targeting_module, "solve_fluctuation", collapsed)


#: Each control: the mutation, applied by ``monkeypatch``, and the check it must fail.
MUTATIONS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], Callable[[], None]]] = {
    "another_arms_column_in_sample": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [
                (
                    "design = base.outcome.arms[arm][:, None]",
                    "design = base.outcome.arms[arms[(list(arms).index(arm) + 1) % len(arms)]]"
                    "[:, None]",
                )
            ],
        ),
        lambda: check_longhand("binary"),
    ),
    "another_arms_column_on_the_exact_law": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [
                (
                    "design = base.outcome.arms[arm][:, None]",
                    "design = base.outcome.arms[arms[(list(arms).index(arm) + 1) % len(arms)]]"
                    "[:, None]",
                )
            ],
        ),
        check_tied_projection,
    ),
    "the_complement_on_the_simplex": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [
                (
                    "propensity = Propensity(propensity_values, arms, simplex=False)",
                    "propensity = Propensity(np.column_stack([1.0 - propensity_values[:, 1], "
                    "propensity_values[:, 1]]), arms)",
                )
            ],
        ),
        check_tied_projection,
    ),
    "the_observed_arm_prediction_as_design": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [
                (
                    "design = base.outcome.arms[arm][:, None]",
                    "design = base.outcome.observed[:, None]",
                )
            ],
        ),
        lambda: check_longhand("three_arm"),
    ),
    "the_shared_design_reported_as_per_arm": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [("if base.folds.is_single and per_arm:", "if False:")],
        ),
        check_curve_is_theorem_ones,
    ),
    "unweighted_in_sample_mechanism": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _IN_SAMPLE,
            [
                (
                    "(data.treatment == arm).astype(float),\n                    data.weights,",
                    "(data.treatment == arm).astype(float),\n                    np.ones(data.n),",
                )
            ],
        ),
        check_weights_equal_duplicates,
    ),
    "one_coefficient_for_every_arm": (_shared_coefficient, check_scores),
    "one_coefficient_against_the_scalar_fluctuations": (
        _shared_coefficient,
        lambda: check_scalar_fluctuation("logistic"),
    ),
    "a_block_diagonal_covariance": (
        lambda mp: mutate(
            mp,
            builtin,
            builtin,
            "_difference_against_reference",
            [
                (
                    "ctx.means[arm].influence_curve - reference.influence_curve,",
                    "ctx.means[arm].influence_curve"
                    " - np.random.default_rng(0).permutation(reference.influence_curve),",
                )
            ],
        ),
        check_joint_covariance,
    ),
    "another_arms_column_in_each_fold": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _CROSS_FITTED,
            [
                ("generated[:, [j]],", "generated[:, [(j + 1) % len(arms)]],"),
                ("generated[test][:, [j]]", "generated[test][:, [(j + 1) % len(arms)]]"),
            ],
        ),
        lambda: check_cv_longhand(BINARY_ACTIVE),
    ),
    "unweighted_mechanism_in_each_fold": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _CROSS_FITTED,
            [
                (
                    "indicators[arm],\n                        data.weights,",
                    "indicators[arm],\n                        np.ones(data.n),",
                )
            ],
        ),
        lambda: check_cv_longhand(BINARY_ACTIVE),
    ),
    "the_first_column_predicted_in_each_fold": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _CROSS_FITTED,
            [("generated[test][:, [j]]", "generated[test][:, [0]]")],
        ),
        lambda: check_cv_longhand(THREE_ARM_ACTIVE),
    ),
    "stitched_global_out_of_fold_design": (
        lambda mp: mutate(
            mp,
            *_CTMLE,
            _CROSS_FITTED,
            [
                (
                    "generated = np.column_stack([by_arm[arm] for arm in arms])",
                    "generated = np.column_stack([base.outcome.arms[arm] for arm in arms])",
                )
            ],
        ),
        lambda: check_cv_longhand(BINARY_ACTIVE),
    ),
}


class TestEachMutationFailsItsCheck:
    """Each check above fails when one line of the construction it checks is broken."""

    @pytest.mark.parametrize("name", list(MUTATIONS))
    def test_the_mutation_fails_the_check(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        apply, check = MUTATIONS[name]
        apply(monkeypatch)
        with pytest.raises(AssertionError):
            check()


# ------------------------------------------------------------------ API and the bootstrap


class TestTheSetting:
    def test_a_selector_strategy_refuses_the_setting(self) -> None:
        with pytest.raises(ValueError, match="oat_design= only applies to strategy='oat'"):
            CTMLE(strategy="greedy", oat_design="shared")
        with pytest.raises(ValueError, match="oat_design must be 'per_arm' or 'shared'"):
            CTMLE(strategy="oat", oat_design="joint")  # type: ignore[arg-type]

    def test_the_default_resolves_by_strategy(self) -> None:
        assert CTMLE(strategy="greedy").oat_design is None
        assert CTMLE(strategy="oat").oat_design == "per_arm"
        assert CTMLE(strategy="oat", oat_design="shared").oat_design == "shared"

    def test_the_method_passes_the_setting_through(self) -> None:
        method = CollaborativeTMLEMethod(strategy="oat", oat_design="shared")
        assert method.estimator_kwargs()["oat_design"] == "shared"
        assert CollaborativeTMLEMethod(strategy="oat").estimator_kwargs()["oat_design"] is None


def test_the_bootstrap_stays_a_diagnostic_on_an_admitted_fit() -> None:
    """Theorem 1 does not cover the bootstrap of this superefficient estimator."""
    result = glm_fit(binary_frame(300, 21), estimands=("ate",), n_bootstrap=20)
    assert result.inference_status == "influence_curve"
    summary = result["ate"].bootstrap
    assert summary is not None
    assert summary.inferential is False
    columns = set(result.to_frame().columns)
    assert {"bootstrap_sd", "bootstrap_range_lower", "bootstrap_range_upper"} <= columns
    assert not {"bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_std_err"} & columns


def test_no_reachable_text_calls_the_curve_efficient() -> None:
    """The reported curve is Theorem 1's, which is not the efficient influence function."""
    result = glm_fit(binary_frame(300, 21))
    texts = [
        result.summary(),
        result.extra["ctmle"].summary(),
        result.extra["ctmle"].describe(),
        result.diagnostics.nuisance_models().summary(),
        NON_INFERENTIAL["generated_design_plugin"].reason,
        NON_INFERENTIAL["generated_design_plugin"].assessment_note,
    ]
    for text in texts:
        assert "efficien" not in text.lower(), text


def test_the_weight_report_of_a_weighted_fit_names_theorem_ones_curve() -> None:
    """A weighted fit's summary points at the weight report, which names the per-arm curve."""
    frame = binary_frame(300, 21)
    result = glm_fit(frame.assign(wt=0.5 + (frame["W1"] > 0).astype(float)), weights="wt")
    assert result.inference_status == "influence_curve"
    assert "weight_report()" in result.summary()
    report = result.data.weight_report().summary()
    assert "Theorem 1, which is not the efficient one" in report
    for text in (result.summary(), result.extra["ctmle"].summary()):
        assert "efficien" not in text.lower(), text


# --------------------------------------------------------------- the per-arm diagnostics


@pytest.mark.parametrize("which", ["binary", "three_arm"])
def test_the_diagnostics_read_each_arms_own_column(
    which: str, request: pytest.FixtureRequest
) -> None:
    """The simplex audit: a consumer that took arm 0 as the complement of arm 1 would fail.

    Each arm's Brier score and each truncation-curve row are recomputed from that arm's own
    column.  At two arms the complement reading gives a different number, so the comparison
    can fail.
    """
    result = request.getfixturevalue(which)
    fresh = replace(result)
    data = fresh.data
    propensity = fresh.nuisance.propensity
    assert not propensity.simplex
    a = np.asarray(data.treatment, dtype=float)
    w = np.asarray(data.weights, dtype=float)
    models = fresh.diagnostics.nuisance_models()
    for arm in data.arm_codes:
        own = np.clip(propensity.arm(arm), 1e-12, 1.0 - 1e-12)
        hit = (a == arm).astype(float)
        brier = float(np.average((own - hit) ** 2, weights=w))
        report = models[f"propensity[{data.arm_label(arm)}]"]
        assert report.metrics["brier"] == pytest.approx(brier, rel=1e-12)
        if data.is_binary_treatment and arm == data.arm_codes[0]:
            # The witness: the complement of the other column scores differently.
            complement = 1.0 - propensity.arm(data.arm_codes[1])
            assert abs(float(np.average((complement - hit) ** 2, weights=w)) - brier) > 1e-6
    values = np.asarray(propensity.values, dtype=float)
    curve = fresh.diagnostics.truncation_curve(bounds=[0.05, 0.2])
    for row in curve.itertuples(index=False):
        outside = (values < row.bound) | (values > row.upper_bound)
        assert row.truncated_fraction == pytest.approx(float(np.mean(outside.any(axis=1))))
    if data.is_binary_treatment:
        complement = np.column_stack([1.0 - values[:, 1], values[:, 1]])
        fractions = [
            float(np.mean(((complement < b) | (complement > u)).any(axis=1)))
            for b, u in zip(curve["bound"], curve["upper_bound"], strict=True)
        ]
        assert np.max(np.abs(np.asarray(fractions) - curve["truncated_fraction"])) > 1e-3
    assert fresh.diagnostics.support().n == data.n
    assert fresh.assess().summary()


def test_the_missingness_tilt_runs_on_a_withheld_per_arm_fit() -> None:
    frame, _ = make_missing_outcome(n=400, seed=4)
    result = (
        CTMLE(
            strategy="oat",
            outcome_learner=LinearRegression(),
            treatment_learner=cubic_logit_learner(),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
            estimands=("ate", "ey1", "ey0"),
        )
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )
    assert result.inference_status == "generated_design_plugin"
    assert not result.nuisance.propensity.simplex
    tilt = result.sensitivity.missingness()
    assert len(tilt) > 0


def test_the_revert_flag_is_declared_true() -> None:
    """The flag ships ``True``; the registered study's declaration names its revert cells."""
    assert ctmle_module.OAT_PER_ARM_INFERENTIAL is True
