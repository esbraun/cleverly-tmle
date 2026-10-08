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
* a longhand of the construction (the binary regressions, the bounds, one scalar
  fluctuation per arm, the curve, the standard error and the interval) on two sampled laws;
* each arm's coefficient equals a scalar fluctuation on that arm's column alone;
* the joint covariance of the stacked curves, with a nonzero cross-arm covariance;
* fixed integer weights equal duplicated rows;
* the per-arm diagnostics read each arm's own column at two and at three arms.
"""

from __future__ import annotations

import re
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
from cleverly.fluctuation.iterative import InitialFit, solve_fluctuation
from cleverly.fluctuation.submodel import Submodel, mean_submodel
from cleverly.inference.multiplier import multiplier_critical_value
from tests import discrete_law_oat as law
from tests.studies.oat_per_arm_laws import (
    BINARY_ACTIVE,
    THREE_ARM_ACTIVE,
    cubic_logit_learner,
    interaction_outcome,
    logit_basis,
)

pytestmark = pytest.mark.xdist_group("outcome_adaptive_per_arm")

#: The submodel bound of the targeting step, ``R``'s ``tmle`` convention.
ALPHA = 0.9995
#: The ``1 - 0.05 / 2`` normal quantile.
Z = float(norm.ppf(0.975))


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


# ------------------------------------------------------------------- the design (exact law)


class TestTheDesignIsPerArm:
    def test_the_tied_arm_takes_the_pooled_projection(self) -> None:
        """Arm 0's design ties ``W = 0`` and ``W = 1``; the shared design separates them."""
        frame = law.frame()
        w = frame["W"].to_numpy().astype(int)
        per_arm = exact_fit("per_arm").nuisance.propensity
        shared = exact_fit("shared").nuisance.propensity
        assert not per_arm.simplex
        assert shared.simplex
        pooled = law.projection(0)[w]
        np.testing.assert_allclose(per_arm.arm(0.0), pooled, rtol=0.0, atol=1e-12)
        separated = (1.0 - law.G1)[w]
        np.testing.assert_allclose(shared.arm(0.0), separated, rtol=0.0, atol=1e-12)
        # The nonzero witness: the two designs differ on the tied rows by 0.15.
        assert np.max(np.abs(per_arm.values - shared.values)) > 0.05
        # Arm 1's design is injective, so both designs recover g_0 there.
        np.testing.assert_allclose(per_arm.arm(1.0), law.G1[w], rtol=0.0, atol=1e-12)

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


class TestTheCurveIsTheoremOnes:
    """The reported curve is ``D(Q_0, G_0(. | Q_0))``, and it is not the EIF."""

    def test_the_curve_equals_the_projection_curve_row_by_row(self) -> None:
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

    def test_it_differs_from_the_efficient_influence_function_on_the_tied_rows(self) -> None:
        frame = law.frame()
        w = frame["W"].to_numpy().astype(int)
        a = frame["A"].to_numpy()
        y = frame["Y"].to_numpy()
        g0 = (1.0 - law.G1)[w]
        q = law.Q[w, 0]
        eif = (a == 0) / g0 * (y - q) + q - law.TRUTH[0]
        reported = np.asarray(exact_fit("per_arm")["ey[0]"].influence_curve)
        tied = w <= 1
        assert np.max(np.abs(reported - eif)[tied]) > 1e-3
        np.testing.assert_allclose(reported[~tied], eif[~tied], rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- the longhand


def _irls(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An unpenalized logistic regression by Newton's method, with an intercept."""
    x = np.column_stack([np.ones(design.shape[0]), design])
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = expit(x @ beta)
        gradient = x.T @ (target - p)
        hessian = (x * (p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return np.asarray(expit(x @ beta))


def _shrink(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1.0 - ALPHA, ALPHA)


def longhand(result: Any) -> dict[float, tuple[float, np.ndarray]]:
    """Steps 1-7 of Benkeser, Cai and van der Laan (2020), Section 3, arm by arm.

    Step 1 is the fit's own initial ``Qbar_n``, which is the input of the construction.
    Every later step is written here: the binary regression of ``1{A = a}`` on the cubic
    basis of ``logit Qbar_n(a, W)``, the bound, one scalar logistic fluctuation by Newton's
    method, the targeted mean and the curve.
    """
    data = result.data
    a = np.asarray(data.treatment, dtype=float)
    y = np.asarray(data.outcome, dtype=float)
    lower, upper = result.config.g_bounds
    out: dict[float, tuple[float, np.ndarray]] = {}
    for arm in data.arm_codes:
        q = np.asarray(result.nuisance.outcome.arms[arm], dtype=float)
        hit = (a == arm).astype(float)
        g = np.clip(_irls(logit_basis(q), hit), lower, upper)
        offset = logit(_shrink(q))
        epsilon = 0.0
        for _ in range(100):
            fitted = expit(offset + epsilon / g)
            score = float(np.sum(hit / g * (y - fitted)))
            curvature = float(np.sum(hit / g**2 * fitted * (1.0 - fitted)))
            step = score / curvature
            epsilon += step
            if abs(step) < 1e-15:
                break
        targeted = _shrink(expit(offset + epsilon / g))
        psi = float(np.mean(targeted))
        curve = hit / g * (y - targeted) + targeted - psi
        out[arm] = (psi, curve)
    return out


@pytest.mark.parametrize("which", ["binary", "three_arm"])
def test_the_fit_is_the_longhand_construction(which: str, request: pytest.FixtureRequest) -> None:
    result = request.getfixturevalue(which)
    data = result.data
    hand = longhand(result)
    n = data.n
    for arm, (psi, curve) in hand.items():
        estimate = result[_mean_name(data, arm)]
        assert estimate.psi == pytest.approx(psi, abs=1e-8)
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-8)
        assert estimate.std_error == pytest.approx(np.sqrt(np.var(curve, ddof=1) / n), abs=1e-8)
    reference = result.data.arm_codes[0] if which == "binary" else _reference(result)
    for arm in data.arm_codes:
        if arm == reference:
            continue
        name = "ate" if which == "binary" else _contrast(data, "ate", arm, reference)
        estimate = result[name]
        psi = hand[arm][0] - hand[reference][0]
        se = np.sqrt(np.var(hand[arm][1] - hand[reference][1], ddof=1) / n)
        assert estimate.psi == pytest.approx(psi, abs=1e-8)
        assert estimate.std_error == pytest.approx(se, abs=1e-8)
        assert estimate.ci == pytest.approx((psi - Z * se, psi + Z * se), abs=1e-8)


def _mean_name(data: Any, arm: float) -> str:
    if data.is_binary_treatment:
        return f"ey{int(arm)}"
    return f"ey[{data.arm_label(arm)}]"


def _contrast(data: Any, stem: str, arm: float, reference: float) -> str:
    return f"{stem}[{data.arm_label(arm)} vs {data.arm_label(reference)}]"


def _reference(result: Any) -> float:
    """The reference arm the fit's contrasts name."""
    data = result.data
    for name in result.estimates:
        match = re.fullmatch(r"ate\[(.+) vs (.+)\]", name)
        if match:
            return next(code for code in data.arm_codes if data.arm_label(code) == match[2])
    raise AssertionError("no contrast")


# ---------------------------------------------------- K columns are K scalar fluctuations


def _scalar_submodel(submodel: Submodel, column: int, arm: float) -> Submodel:
    """Column ``column`` of a mean submodel alone, as a one-arm submodel."""
    return Submodel(
        submodel.observed[:, [column]],
        {level: values[:, [column]] for level, values in submodel.arms.items()},
        (submodel.names[column],),
        "mean",
        {arm: 0},
    )


@pytest.mark.parametrize("kind", ["logistic", "linear"])
def test_each_arm_coefficient_is_a_scalar_fluctuation(kind: str) -> None:
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


def test_every_arms_score_is_solved(three_arm: Any) -> None:
    initial = _arm_scores(three_arm, targeted=False)
    # The witness: the initial scores differ in sign, so no single shared coefficient
    # could solve every arm's equation at once.
    assert min(initial.values()) < -1e-4
    assert max(initial.values()) > 1e-4
    for arm, score in _arm_scores(three_arm, targeted=True).items():
        assert abs(score) < 1e-8, arm


# ---------------------------------------------------------------------- joint covariance


def test_the_contrast_reads_the_joint_covariance(binary: Any) -> None:
    n = binary.data.n
    d1 = np.asarray(binary["ey1"].influence_curve)
    d0 = np.asarray(binary["ey0"].influence_curve)
    # The witness: the two arm curves covary, so a block-diagonal covariance is wrong.
    assert abs(np.cov(d1, d0)[0, 1]) > 1e-3 * np.sqrt(np.var(d1) * np.var(d0))
    assert binary["ate"].std_error == pytest.approx(np.sqrt(np.var(d1 - d0, ddof=1) / n), rel=1e-12)


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


# ----------------------------------------------------------------------- fixed weights


def test_integer_weights_equal_duplicated_rows() -> None:
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


# --------------------------------------------------------------- the per-arm diagnostics


@pytest.mark.parametrize("which", ["binary", "three_arm"])
def test_the_diagnostics_read_each_arms_own_column(
    which: str, request: pytest.FixtureRequest
) -> None:
    """The simplex audit: a consumer that took arm 0 as the complement of arm 1 would fail."""
    result = request.getfixturevalue(which)
    fresh = replace(result)
    data = fresh.data
    propensity = fresh.nuisance.propensity
    assert not propensity.simplex
    if data.is_binary_treatment:
        # The witness: the per-arm columns are not complements.
        assert np.max(np.abs(propensity.values.sum(axis=1) - 1.0)) > 1e-3
    models = fresh.diagnostics.nuisance_models()
    for arm in data.arm_codes:
        report = models[f"propensity[{data.arm_label(arm)}]"]
        assert np.isfinite(report.metrics["brier"])
    support = fresh.diagnostics.support()
    assert support.n == data.n
    curve = fresh.diagnostics.truncation_curve()
    assert len(curve) > 0
    assessment = fresh.assess()
    assert assessment.summary()


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
