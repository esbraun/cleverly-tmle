r"""The per-arm outcome-adaptive design inside baseline strata and with missing data.

``CTMLE(strategy="oat", oat_design="per_arm")`` fits, for each arm ``a`` and on the one column
``q_a = Qbar_n(a, W)``, the treatment factor ``P(A = a | Delta_A = 1, q_a)``, the outcome
observation factor ``P(Delta = 1 | A = a, Delta_A = 1, q_a)`` with missing outcomes, and the
treatment observation factor ``P(Delta_A = 1 | q_a)`` with a missing treatment.  With baseline
strata each factor is fitted inside each stratum.  By the chain rule the product is
``P(C_a = 1 | q_a)`` (or ``P(C_a = 1 | q_a, S)``) for ``C_a = Delta_A Delta 1{A = a}``: the
construction of Benkeser, Cai and van der Laan (2020), Theorem 1, applied to the composite
indicator.  Every ``oat`` fit withholds its interval (``generated_design_plugin``), so the
checks below read points and the retained ``plugin_std_error`` diagnostic.

The checks, and what makes each able to fail:

* a longhand of the in-sample construction (the factor regressions per arm and stratum, the
  product, one scalar fluctuation per arm and stratum, the means and the curves);
* a longhand of the cross-fitted construction with a declared split;
* each stratum point equals an unstratified fit on the stratum's rows, and the marginal point
  is the ``P_n(S = s)`` mixture;
* the natural-course mean reads the observation factor at the realised arm;
* an exact finite law: the factor product is the population cell share of ``C_a = 1``, and
  with the oracle ``Qbar`` the point is the truth and the curve is the paper's;
* every arm-and-stratum score is solved under each admitted fluctuation, and a longhand
  of ``target_weights=True``;
* the missing-outcome route's covariate is the composite construction's;
* fold-local nesting, fixed weights, the status matrix, the refusals, the constant-target
  rule, the missingness tilt, the support report's labels, and save and load.

Each design check has a mutation control in :class:`TestEachMutationFailsItsCheck`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

import cleverly
from cleverly.estimators import CTMLE
from cleverly.estimators import ctmle as ctmle_module
from cleverly.exceptions import CapabilityError
from cleverly.learners.crossfit import _POST_DRAW_REMEDY, SplitPlan, random_partition
from cleverly.sensitivity import missingness as missingness_module
from tests import discrete_law_oat as law
from tests.studies.oat_per_arm_laws import (
    BINARY_ACTIVE,
    THREE_ARM_ACTIVE,
    OatLaw,
    OracleOutcome,
    cubic_logit_learner,
    interaction_outcome,
    logit_basis,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_outcome_adaptive_per_arm import (
    _irls,
    _mean_name,
    _predict,
    _reference,
    _shrink,
)

pytestmark = pytest.mark.xdist_group("outcome_adaptive_per_arm")

#: The cross-fitted longhand's fold count and split seed.
FOLDS = 3
SPLIT_SEED = 41

Longhand = dict[str, tuple[float, np.ndarray]]


# ----------------------------------------------------------------------------- the data


def extended_frame(
    source: OatLaw,
    n: int,
    seed: int,
    *,
    delta: bool = False,
    treatment_delta: bool = False,
    strata: bool = False,
) -> pd.DataFrame:
    """A study-law sample with missing outcomes, a missing treatment or baseline strata.

    ``Delta`` depends on the arm, on ``W1`` and on ``W3``, which also drives the treatment,
    so ``P(Delta = 1 | A = a, W)`` varies inside a level set of ``Qbar(a, W)``.  ``DA``
    depends on ``W1`` and ``W2``.  ``S`` cuts ``W1`` into three strata at -0.5 and 0.6.
    ``S`` is the last covariate, and
    :class:`~tests.studies.oat_per_arm_laws.OracleOutcome` ignores it.
    """
    frame = source.sample(n, seed)[0]
    rng = np.random.default_rng(seed + 1000)
    w1 = frame["W1"].to_numpy()
    w2 = frame["W2"].to_numpy()
    w3 = frame["W3"].to_numpy()
    arm = pd.Series(frame["A"]).map({label: i for i, label in enumerate(source.arm_labels)})
    if source.k == 2:
        arm = frame["A"].astype(int)
    index = np.asarray(arm, dtype=float)
    if strata:
        frame["S"] = np.digitize(w1, [-0.5, 0.6]).astype(float)
    if delta:
        respond = rng.random(n) < expit(1.2 + 0.4 * index - 0.5 * w1 + 0.5 * w3)
        frame["Delta"] = respond.astype(float)
        frame.loc[~respond, "Y"] = np.nan
    if treatment_delta:
        recorded = rng.random(n) < expit(2.0 - 0.5 * w1 + 0.6 * w2)
        frame["DA"] = recorded.astype(float)
        frame["A"] = frame["A"].astype(object)
        frame.loc[~recorded, "A"] = np.nan
    return frame


def roles(frame: pd.DataFrame) -> dict[str, Any]:
    """The declared roles a frame from :func:`extended_frame` carries."""
    out: dict[str, Any] = {}
    if "Delta" in frame:
        out["delta"] = "Delta"
    if "DA" in frame:
        out["treatment_delta"] = "DA"
    if "S" in frame:
        out["strata"] = "S"
    if "wt" in frame:
        out["weights"] = "wt"
    return out


def per_arm_fit(frame: pd.DataFrame, **extra: Any) -> Any:
    """The per-arm fit with the study's learners, in sample unless ``extra`` says otherwise."""
    k = int(pd.Series(frame["A"]).dropna().nunique())
    settings: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": interaction_outcome(k),
        "treatment_learner": cubic_logit_learner(),
        "missingness_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        # Declared bounds: the "auto" bounds read n, so a subset or a duplicated frame
        # would bound differently.
        "g_bounds": (0.01, 0.99),
        "estimands": ("ey0", "ey1", "ate") if k == 2 else ("ey", "ate"),
        **({} if k == 2 else {"reference": "0"}),
        **extra,
    }
    return CTMLE(**settings).fit(frame, outcome="Y", treatment="A", **roles(frame)).single()


# -------------------------------------------------------------------------- the longhand


def _regions(data: Any) -> list[np.ndarray]:
    if data.strata is None:
        return [np.ones(data.n, dtype=bool)]
    return [np.asarray(data.strata == code) for code in range(data.n_strata)]


def _factor(
    basis: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    *,
    constant: bool,
) -> np.ndarray:
    """One logistic factor fitted on the ``fit`` rows and predicted on the ``predict`` rows."""
    values = np.unique(target[fit])
    if constant and values.size == 1:
        return np.full(int(predict.sum()), float(values[0]))
    beta = _irls(basis[fit], target[fit], weights[fit])
    return np.clip(_predict(basis[predict], beta), 0.0, 1.0)


def longhand_factors(
    data: Any, q: dict[float, np.ndarray], train: np.ndarray, test: np.ndarray
) -> dict[str, np.ndarray]:
    """Each factor of each arm, fitted on ``train`` and predicted on ``test``, per stratum.

    Written from the definitions of the module docstring: the treatment factor on the rows
    with a recorded treatment, the outcome observation factor on the recorded rows of the
    arm, and the treatment observation factor on every row.
    """
    a = np.asarray(data.treatment, dtype=float)
    recorded = np.asarray(data.treatment_recorded, dtype=bool)
    observed = np.asarray(data.observed, dtype=bool)
    weights = np.asarray(data.weights, dtype=float)
    arms = data.arm_codes
    out = {name: np.full((data.n, len(arms)), np.nan) for name in ("g", "pi", "recording")}
    for j, arm in enumerate(arms):
        basis = logit_basis(q[arm])
        hit = (a == arm).astype(float)
        for region in _regions(data):
            at = region & test
            if not at.any():
                continue
            out["g"][at, j] = _factor(
                basis, hit, weights, region & train & recorded, at, constant=False
            )
            if data.has_missing_outcome:
                out["pi"][at, j] = _factor(
                    basis,
                    observed.astype(float),
                    weights,
                    region & train & recorded & (a == arm),
                    at,
                    constant=True,
                )
            if data.has_missing_treatment:
                out["recording"][at, j] = _factor(
                    basis, recorded.astype(float), weights, region & train, at, constant=True
                )
    return out


def composite(result: Any, factors: dict[str, np.ndarray]) -> np.ndarray:
    """The bounded product ``P(C_a = 1 | q_a)``, each factor bounded as the fit bounds it."""
    data = result.data
    lower, upper = result.config.g_bounds
    floor = result.config.missingness_bound
    product = np.clip(factors["g"], lower, upper)
    if data.has_missing_outcome:
        product = product * np.clip(factors["pi"], floor, 1.0)
    if data.has_missing_treatment:
        product = product * np.clip(factors["recording"], floor, 1.0)
    return product


def target(
    result: Any, q: dict[float, np.ndarray], product: np.ndarray, *, weighted: bool = False
) -> Longhand:
    """One scalar logistic fluctuation per arm and stratum, then the means and curves.

    The covariate is ``C_a / P(C_a = 1 | q_a)``.  With ``weighted``, the form of
    ``target_weights=True``, the covariate is ``C_a`` and ``1 / P(C_a = 1 | q_a)`` joins the
    weights.  A stratum mean is the weighted mean of the
    targeted fit over its rows, with its curve embedded as ``1{S = s} D_s / P_n(S = s)``.
    The marginal mean is the weighted mean over every row, with the curve
    ``C_a / G (Y - Q*) + Q* - psi``.
    """
    data = result.data
    a = np.asarray(data.treatment, dtype=float)
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    w = np.asarray(data.weights, dtype=float)
    hit_all = np.asarray(data.observed, dtype=bool) & np.asarray(data.treatment_recorded)
    out: Longhand = {}
    for j, arm in enumerate(data.arm_codes):
        hit = (hit_all & (a == arm)).astype(float)
        h = 1.0 / product[:, j]
        covariate = np.ones(data.n) if weighted else h
        mass = w * hit * (h if weighted else 1.0)
        offset = logit(_shrink(q[arm]))
        targeted = np.empty(data.n)
        for region in _regions(data):
            epsilon = 0.0
            for _ in range(200):
                fitted = expit(offset[region] + epsilon * covariate[region])
                score = float(np.sum((mass * covariate)[region] * (y[region] - fitted)))
                curvature = float(np.sum((mass * covariate**2)[region] * fitted * (1.0 - fitted)))
                step = score / curvature
                epsilon += step
                if abs(step) < 1e-15:
                    break
            targeted[region] = _shrink(expit(offset[region] + epsilon * covariate[region]))
        piece = hit * h * (y - targeted) + targeted
        name = _mean_name(data, arm)
        psi = float(np.average(targeted, weights=w))
        out[name] = (psi, w / np.mean(w) * (piece - psi))
        if data.strata is None:
            continue
        for code, region in enumerate(_regions(data)):
            psi_s = float(np.average(targeted[region], weights=w[region]))
            local = w[region] / np.mean(w[region]) * (piece[region] - psi_s)
            curve = np.zeros(data.n)
            curve[region] = local * data.n / int(region.sum())
            out[f"{name}[{data.stratum_label(code)}]"] = (psi_s, curve)
    return out


def in_sample_longhand(
    result: Any, *, weighted: bool = False
) -> tuple[dict[str, np.ndarray], Longhand]:
    data = result.data
    q = {arm: np.asarray(result.nuisance.outcome.arms[arm], dtype=float) for arm in data.arm_codes}
    every = np.ones(data.n, dtype=bool)
    factors = longhand_factors(data, q, every, every)
    return factors, target(result, q, composite(result, factors), weighted=weighted)


def assert_matches(result: Any, hand: Longhand, atol: float = 1e-8) -> None:
    """Every longhand mean, curve and standard error, and the contrasts, are the fit's."""
    n = result.data.n
    for name, (psi, curve) in hand.items():
        estimate = result[name]
        assert estimate.psi == pytest.approx(psi, abs=atol), name
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=atol)
        assert estimate.plugin_std_error == pytest.approx(
            np.sqrt(np.var(curve, ddof=1) / n), abs=atol
        )
    data = result.data
    reference = _reference(result)
    for arm in data.arm_codes:
        if arm == reference:
            continue
        name = (
            "ate"
            if data.is_binary_treatment
            else f"ate[{data.arm_label(arm)} vs {data.arm_label(reference)}]"
        )
        treated, control = hand[_mean_name(data, arm)], hand[_mean_name(data, reference)]
        assert result[name].psi == pytest.approx(treated[0] - control[0], abs=atol)


def assert_factors(result: Any, factors: dict[str, np.ndarray], atol: float = 1e-8) -> None:
    nuisance = result.nuisance
    np.testing.assert_allclose(nuisance.propensity.values, factors["g"], rtol=0.0, atol=atol)
    if result.data.has_missing_outcome:
        np.testing.assert_allclose(nuisance.missingness, factors["pi"], rtol=0.0, atol=atol)
    if result.data.has_missing_treatment:
        np.testing.assert_allclose(
            nuisance.treatment_observation, factors["recording"], rtol=0.0, atol=atol
        )


# --------------------------------------------------------------------------- the checks

LONGHAND_CASES: dict[str, Callable[[], pd.DataFrame]] = {
    "binary_delta": lambda: extended_frame(BINARY_ACTIVE, 700, 11, delta=True),
    "binary_delta_strata": lambda: extended_frame(BINARY_ACTIVE, 900, 12, delta=True, strata=True),
    "three_arm_delta": lambda: extended_frame(THREE_ARM_ACTIVE, 900, 13, delta=True),
    "binary_treatment_delta": lambda: extended_frame(
        BINARY_ACTIVE, 800, 14, delta=True, treatment_delta=True
    ),
    "binary_treatment_delta_strata": lambda: extended_frame(
        BINARY_ACTIVE, 1100, 15, delta=True, treatment_delta=True, strata=True
    ),
    "binary_strata": lambda: extended_frame(BINARY_ACTIVE, 900, 16, strata=True),
}


def check_longhand(case: str) -> None:
    result = per_arm_fit(LONGHAND_CASES[case]())
    factors, hand = in_sample_longhand(result)
    assert_factors(result, factors)
    assert_matches(result, hand)
    # The witness: the outcome model is estimated, so the targeting step moves.
    epsilon = np.asarray(result.fluctuations["mean"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3, epsilon


def cv_fit(frame: pd.DataFrame, **extra: Any) -> tuple[Any, np.ndarray]:
    drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
    result = per_arm_fit(
        frame,
        cross_fit=True,
        n_folds=FOLDS,
        split_plan=SplitPlan.from_folds([drawn]),
        **extra,
    )
    return result, np.asarray(drawn.assignment)


def check_cv_longhand() -> None:
    """Each fold's outcome GLM and factors refitted on its training rows, then the pooled step.

    Binary, with ``delta=`` and baseline strata.  The fold's outcome model trains on the
    fold's respondents, and every factor of every arm and stratum on the fold's training rows.
    """
    from tests.studies.oat_per_arm_laws import arm_interactions

    frame = extended_frame(BINARY_ACTIVE, 1500, 26, delta=True, strata=True)
    result, assignment = cv_fit(frame)
    # The premise: no targeted value reaches the submodel bound, where the solver's
    # shrinkage after each Newton step would make the path matter.
    for arm in result.data.arm_codes:
        targeted = np.asarray(result.fluctuations["mean"].targeted.arms[arm])
        assert np.all(np.abs(targeted - 0.5) < 0.4995)
    data = result.data
    arms = data.arm_codes
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    w = np.asarray(data.weights, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    design = arm_interactions(data.treatment_design(), indicators=1)
    q = {arm: np.empty(data.n) for arm in arms}
    factors = {name: np.full((data.n, len(arms)), np.nan) for name in ("g", "pi", "recording")}
    for fold in range(FOLDS):
        test = assignment == fold
        train = ~test
        beta = _irls(design[train & observed], y[train & observed], w[train & observed])
        q_fold = {
            arm: np.clip(
                _predict(arm_interactions(data.counterfactual_design(arm), indicators=1), beta),
                0.0,
                1.0,
            )
            for arm in arms
        }
        for arm in arms:
            q[arm][test] = q_fold[arm][test]
        piece = longhand_factors(data, q_fold, train, test)
        for name in factors:
            factors[name][test] = piece[name][test]
    for arm in arms:
        np.testing.assert_allclose(result.nuisance.outcome.arms[arm], q[arm], rtol=0, atol=1e-8)
    assert_factors(result, factors)
    assert_matches(result, target(result, q, composite(result, factors)))


def _level(label: str) -> float:
    """The value of ``S`` that a stratum label such as ``"S=1.0"`` names."""
    return float(label.split("=")[1])


def check_stratum_is_the_subset_fit() -> None:
    """Each stratum point is an unstratified per-arm fit on that stratum's rows.

    The outcome learner is the law's own ``Qbar_0``, which fits nothing, so both fits see
    the same design.  In sample, complete, with ``delta=`` and with ``treatment_delta=``,
    at two and three arms.
    """
    cases = [
        (BINARY_ACTIVE, {}),
        (BINARY_ACTIVE, {"delta": True}),
        (BINARY_ACTIVE, {"delta": True, "treatment_delta": True}),
        (THREE_ARM_ACTIVE, {"delta": True}),
    ]
    for source, flags in cases:
        frame = extended_frame(source, 1200, 31, strata=True, **flags)
        oracle = {"outcome_learner": OracleOutcome(source.name)}
        stratified = per_arm_fit(frame, **oracle)
        pooled = per_arm_fit(frame.drop(columns="S"), **oracle)
        for code in range(stratified.data.n_strata):
            label = stratified.data.stratum_label(code)
            subset = per_arm_fit(frame[frame["S"] == _level(label)].drop(columns="S"), **oracle)
            for arm in stratified.data.arm_codes:
                name = _mean_name(stratified.data, arm)
                assert stratified[f"{name}[{label}]"].psi == pytest.approx(
                    subset[name].psi, abs=1e-9
                ), (source.name, flags, name, label)
        # The witness: the design fitted on every row moves the marginal arm means.
        moved = max(
            abs(stratified[name].psi - pooled[name].psi)
            for name in (_mean_name(stratified.data, arm) for arm in stratified.data.arm_codes)
        )
        # Five orders of magnitude above the equality tolerance.
        assert moved > 1e-4, (source.name, flags)


def check_marginal_is_the_mixture() -> None:
    frame = extended_frame(BINARY_ACTIVE, 900, 12, delta=True, strata=True)
    result = per_arm_fit(frame)
    data = result.data
    mass = [float(np.mean(data.strata == code)) for code in range(data.n_strata)]
    for name in ("ey0", "ey1", "ate"):
        mixture = sum(
            p * result[f"{name}[{data.stratum_label(code)}]"].psi for code, p in enumerate(mass)
        )
        assert result[name].psi == pytest.approx(mixture, abs=1e-12)
    # The marginal curve is the linearity curve of the mixture: the stratum curves, each
    # weighted by its mass, plus the mass term psi_S - psi.
    codes = np.asarray(data.strata)
    for name in ("ey0", "ey1", "ate"):
        stacked = sum(
            p * np.asarray(result[f"{name}[{data.stratum_label(code)}]"].influence_curve)
            for code, p in enumerate(mass)
        )
        by_stratum = np.array(
            [result[f"{name}[{data.stratum_label(code)}]"].psi for code in range(data.n_strata)]
        )
        mass_term = by_stratum[codes] - result[name].psi
        np.testing.assert_allclose(
            result[name].influence_curve, stacked + mass_term, rtol=0.0, atol=1e-10
        )
        # The witness: the mass term is not negligible, so the identity reads it.
        assert np.sqrt(np.mean(mass_term**2)) > 1e-3, name


def natural_course_longhand(result: Any) -> float:
    """``ey_obs`` with the design ``(A, Qbar_n(A, W))``: one pooled scalar fluctuation.

    The covariate is ``Delta / pi_A`` with ``pi_A`` the outcome observation factor of the
    realised arm, bounded below by the fit's ``nuisance_bound``.
    """
    data = result.data
    a = np.asarray(data.treatment, dtype=float)
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    w = np.asarray(data.weights, dtype=float)
    observed = np.asarray(data.observed, dtype=float)
    q = {arm: np.asarray(result.nuisance.outcome.arms[arm], dtype=float) for arm in data.arm_codes}
    every = np.ones(data.n, dtype=bool)
    factors = longhand_factors(data, q, every, every)
    realised = np.array([data.arm_codes.index(value) for value in a])
    pi = np.clip(factors["pi"][np.arange(data.n), realised], result.config.missingness_bound, 1)
    h = 1.0 / pi
    offset = logit(_shrink(np.asarray(result.nuisance.outcome.observed, dtype=float)))
    epsilon = 0.0
    for _ in range(200):
        fitted = expit(offset + epsilon * h)
        score = float(np.sum(w * observed * h * (y - fitted)))
        curvature = float(np.sum(w * observed * h**2 * fitted * (1.0 - fitted)))
        step = score / curvature
        epsilon += step
        if abs(step) < 1e-15:
            break
    assert abs(epsilon) > 1e-3
    return float(np.average(_shrink(expit(offset + epsilon * h)), weights=w))


def check_natural_course() -> None:
    frame = extended_frame(BINARY_ACTIVE, 900, 41, delta=True)
    result = per_arm_fit(frame, estimands=("ey_obs", "par", "paf", "ey0"))
    assert result["ey_obs"].psi == pytest.approx(natural_course_longhand(result), abs=1e-8)
    assert result["par"].psi == pytest.approx(result["ey_obs"].psi - result["ey0"].psi, abs=1e-12)
    assert result["paf"].psi == pytest.approx(
        1.0 - result["ey0"].psi / result["ey_obs"].psi, abs=1e-12
    )


def check_natural_course_strata(cross_fit: bool) -> None:
    frame = extended_frame(BINARY_ACTIVE, 1200, 42, delta=True, strata=True)
    extra: dict[str, Any] = {
        "estimands": ("ey_obs",),
        "outcome_learner": OracleOutcome("binary_active"),
    }
    if cross_fit:
        # The natural-course contract needs package-generated folds.
        result = per_arm_fit(frame, cross_fit=True, n_folds=FOLDS, **extra)
    else:
        result = per_arm_fit(frame, **extra)
    data = result.data
    mass = [float(np.mean(data.strata == code)) for code in range(data.n_strata)]
    labels = [data.stratum_label(code) for code in range(data.n_strata)]
    mixture = sum(p * result[f"ey_obs[{label}]"].psi for p, label in zip(mass, labels, strict=True))
    assert result["ey_obs"].psi == pytest.approx(mixture, abs=1e-12)
    if cross_fit:
        return
    for label in labels:
        subset = per_arm_fit(frame[frame["S"] == _level(label)].drop(columns="S"), **extra)
        assert result[f"ey_obs[{label}]"].psi == pytest.approx(subset["ey_obs"].psi, abs=1e-9)


def exact_frame() -> pd.DataFrame:
    """:mod:`tests.discrete_law_oat`, with half of every ``W = 1``, ``A = 0`` cell missing.

    ``Qbar(0, .)`` ties ``W = 0`` and ``W = 1``, and the outcome observation probability
    ``pi(0, W)`` is 1 at ``W = 0`` and 1/2 at ``W = 1``, so ``pi(0, W)`` varies inside the
    tie cell of the arm-0 design.  Every such cell count is even, so the respondents of
    each cell keep the cell's outcome share, and the realised sample is exactly a MAR law
    with ``E[Y | A, Delta = 1, W] = Qbar_0(A, W)``.
    """
    frame = law.frame()
    respond = np.ones(len(frame), dtype=bool)
    for (w, a, _), rows in frame.groupby(["W", "A", "Y"]).groups.items():
        if a == 0 and w == 1:
            index = np.asarray(list(rows))
            assert index.size % 2 == 0
            respond[index[: index.size // 2]] = False
    frame["Delta"] = respond.astype(float)
    frame.loc[~respond, "Y"] = np.nan
    return frame


class _FlatArmOne(law.OracleOutcome):
    """The law's ``Qbar_0``, except a constant 0.5 for arm 1: a wrong outcome regression.

    ``Qbar_0(1, .)`` is not a function of this constant, so the arm-1 design pools every
    ``W`` and the remainder of the arm-1 point is not zero.
    """

    def predict(self, X: Any) -> np.ndarray:
        values = super().predict(X)
        arm = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        return np.where(arm == 1, 0.5, values)


def exact_fit(frame: pd.DataFrame, outcome_learner: Any) -> Any:
    """The per-arm fit on the exact law, with the saturated learner for every factor."""
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=outcome_learner,
            treatment_learner=law.SaturatedCategorical(),
            missingness_learner=law.SaturatedCategorical(),
            cross_fit=False,
            simultaneous=False,
            estimands=("ey1", "ey0", "ey_obs"),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def cell_share(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """The mean of ``values`` over the rows that share each row's ``keys``."""
    _, inverse = np.unique(np.round(keys, 9), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    totals = np.bincount(inverse, weights=values)
    return totals[inverse] / np.bincount(inverse)[inverse]


def check_exact_law() -> None:
    """The design is the population ``P(C_a = 1 | q_a)``, and the oracle ``Qbar`` is exact.

    1. The factor product equals the cell share of ``C_a = 1`` given ``q_a`` to 1e-12.  On
       the arm-0 tie cell it differs from the shipped ``g_a(q_a) pi(a, W)`` by more than
       0.05, so the check reads the observation factor's design.
    2. With the oracle ``Qbar`` the arm means and the natural course equal the truth to
       1e-12, and each plug-in curve is the paper's curve on ``O'_a`` row by row.
    3. The contrast: a wrong outcome regression with the same saturated design misses the
       arm-1 truth by more than 1e-3.
    """
    frame = exact_frame()
    exact = exact_fit(frame, law.OracleOutcome())
    data = exact.data
    nuisance = exact.nuisance
    a = np.asarray(data.treatment, dtype=float)
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    observed = np.asarray(data.observed, dtype=float)
    q0 = {arm: np.asarray(nuisance.outcome.arms[arm], dtype=float) for arm in data.arm_codes}
    for j, arm in enumerate(data.arm_codes):
        hit = observed * (a == arm)
        share = cell_share(hit, q0[arm])
        product = nuisance.propensity.values[:, j] * nuisance.missingness[:, j]
        np.testing.assert_allclose(product, share, rtol=0.0, atol=1e-12)
        name = f"ey{int(arm)}"
        assert exact[name].psi == pytest.approx(law.TRUTH[int(arm)], abs=1e-12)
        curve = hit / share * (y - q0[arm]) + q0[arm] - exact[name].psi
        np.testing.assert_allclose(exact[name].influence_curve, curve, rtol=0.0, atol=1e-12)
    # The full-covariate product g_0(q_0) pi(0, W) on the tie cell W in {0, 1}.
    tie = np.asarray(frame["W"] <= 1)
    shipped = nuisance.propensity.values[:, 0] * nuisance.tilt_missingness[:, 0]
    product = nuisance.propensity.values[:, 0] * nuisance.missingness[:, 0]
    assert np.min(np.abs(shipped - product)[tie]) > 0.05
    # The natural course: the design (A, Qbar(A, W)), read at the realised arm.
    realised = np.array([data.arm_codes.index(value) for value in a])
    q_obs = np.asarray(nuisance.outcome.observed, dtype=float)
    pi_obs = cell_share(observed, np.column_stack([a, q_obs]))
    np.testing.assert_allclose(
        nuisance.missingness[np.arange(data.n), realised], pi_obs, rtol=0.0, atol=1e-12
    )
    truth = float(np.sum(law.COUNTS[..., 1]) / law.N)
    assert exact["ey_obs"].psi == pytest.approx(truth, abs=1e-12)
    curve = observed / pi_obs * (y - q_obs) + q_obs - exact["ey_obs"].psi
    np.testing.assert_allclose(exact["ey_obs"].influence_curve, curve, rtol=0.0, atol=1e-12)
    # The contrast: the wrong outcome regression misses the arm-1 truth.
    wrong = exact_fit(frame, _FlatArmOne())
    assert abs(wrong["ey1"].psi - law.TRUTH[1]) > 1e-3


def check_fold_local_nesting() -> None:
    """Changing every validation outcome and response of a fold leaves that fold's nuisances."""
    frame = extended_frame(BINARY_ACTIVE, 600, 51, delta=True, strata=True)
    result, assignment = cv_fit(frame)
    held = assignment == 0
    changed = frame.copy()
    flip = held & (changed["Delta"] == 1).to_numpy()
    changed.loc[flip, "Y"] = 1.0 - changed.loc[flip, "Y"]
    respond = held & (changed["Delta"] == 0).to_numpy()
    changed.loc[respond, "Delta"] = 1.0
    changed.loc[respond, "Y"] = 1.0
    moved, _ = cv_fit(changed)
    for name in ("propensity", "missingness"):
        before = np.asarray(
            getattr(result.nuisance, name).values
            if name == "propensity"
            else result.nuisance.missingness
        )
        after = np.asarray(
            getattr(moved.nuisance, name).values
            if name == "propensity"
            else moved.nuisance.missingness
        )
        np.testing.assert_array_equal(before[held], after[held])
    for arm in result.data.arm_codes:
        np.testing.assert_array_equal(
            result.nuisance.outcome.arms[arm][held], moved.nuisance.outcome.arms[arm][held]
        )
    # The witness: the change does move the other folds' nuisances.
    assert (
        np.max(np.abs(result.nuisance.missingness[~held] - moved.nuisance.missingness[~held])) > 0
    )


def check_weights_equal_duplicates() -> None:
    frame = extended_frame(BINARY_ACTIVE, 400, 61, delta=True, strata=True)
    counts = 1 + np.arange(len(frame)) % 3
    weighted = per_arm_fit(frame.assign(wt=counts.astype(float)))
    duplicated = per_arm_fit(frame.loc[frame.index.repeat(counts)].reset_index(drop=True))
    names = [name for name in weighted.estimates if name.startswith(("ey", "ate"))]
    for name in names:
        assert weighted[name].psi == pytest.approx(duplicated[name].psi, abs=1e-7), name
    # The witness: the weights move the estimate.
    assert abs(per_arm_fit(frame)["ate"].psi - weighted["ate"].psi) > 1e-4


class _CountingLogistic(LogisticRegression):
    """A logistic regression that records each binary target it is fitted to.

    A target with one class predicts that class, as the constant-target rule does, where
    the parent class would raise.  So a fit that hands the learner a constant target
    returns the same factor, and only the record of targets shows that it fitted one.
    """

    targets: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _CountingLogistic:
        target = np.asarray(y, dtype=float).reshape(-1)
        type(self).targets.append(target.copy())
        if np.unique(target).size == 1:
            self.constant_ = float(target[0])
            self.classes_ = np.array([0.0, 1.0])
            return self
        self.constant_ = None
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        if getattr(self, "constant_", None) is None:
            return super().predict_proba(X)
        level = np.full(np.asarray(X).shape[0], self.constant_)
        return np.column_stack([1.0 - level, level])


def check_constant_target() -> None:
    """A stratum where every arm-1 row responds: its factor is 1 and fits no learner."""
    frame = extended_frame(BINARY_ACTIVE, 800, 71, delta=True, strata=True)
    full = (frame["S"] == 2) & (frame["A"] == 1)
    frame.loc[full, "Delta"] = 1.0
    frame.loc[full, "Y"] = frame.loc[full, "Y"].fillna(1.0)
    _CountingLogistic.targets = []
    result = per_arm_fit(frame, treatment_learner=_CountingLogistic(C=np.inf, max_iter=1000))
    assert np.all(result.nuisance.missingness[full.to_numpy(), 1] == 1.0)
    # Every fit's target has both classes: the constant cell never reached the learner.
    assert all(np.unique(target).size == 2 for target in _CountingLogistic.targets)
    expected = 2 * 3 + 2 * 3 - 1  # treatment and observation factors, less the constant one
    assert len(_CountingLogistic.targets) == expected


def check_tilt_reads_the_full_covariate_model(repeats: int = 1) -> None:
    """The tilt reads ``P(Delta = 1 | A = a, W)``, not the design factor."""
    frame = extended_frame(BINARY_ACTIVE, 700, 81, delta=True)
    result = per_arm_fit(frame, repeats=repeats, **({"cross_fit": True} if repeats > 1 else {}))
    assert result.nuisance.tilt_missingness is not None
    frame_out = result.sensitivity.missingness([-1.0, 0.0, 1.0], estimands=["ey1"])
    by_gamma = dict(zip(frame_out["gamma"], frame_out["psi"], strict=True))
    assert by_gamma[0.0] == pytest.approx(result["ey1"].psi, abs=1e-10)
    floor = result.config.missingness_bound
    for gamma in (-1.0, 1.0):
        values = []
        design_values = []
        for repeat in result.repeats:
            q = np.asarray(repeat.fluctuations["mean"].targeted.arms[1.0], dtype=float)
            full = np.clip(repeat.nuisance.tilt_missingness[:, 1], floor, 1.0)
            design = np.clip(repeat.nuisance.missingness[:, 1], floor, 1.0)
            tilted = expit(logit(q) + gamma)
            values.append(float(np.mean(full * q + (1.0 - full) * tilted)))
            design_values.append(float(np.mean(design * q + (1.0 - design) * tilted)))
        assert by_gamma[gamma] == pytest.approx(float(np.median(values)), abs=1e-10)
        # The witness: the design factor gives a different tilt.
        assert abs(by_gamma[gamma] - float(np.median(design_values))) > 1e-3


#: Every admitted configuration of the status matrix, as ``(frame, extra settings)``.
STATUS_CASES: dict[str, Callable[[], tuple[pd.DataFrame, dict[str, Any]]]] = {
    "strata": lambda: (extended_frame(BINARY_ACTIVE, 500, 91, strata=True), {}),
    "strata_cross_fit": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 91, strata=True),
        {"cross_fit": True, "n_folds": 3},
    ),
    "delta_cross_fit": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True),
        {"cross_fit": True, "n_folds": 3},
    ),
    "delta_natural_course_cross_fit": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True),
        {"cross_fit": True, "n_folds": 3, "estimands": ("ey_obs", "par", "paf", "ey0")},
    ),
    "treatment_delta": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 93, delta=True, treatment_delta=True),
        {},
    ),
    "linear": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True),
        {"cross_fit": True, "n_folds": 3, "fluctuation": "linear"},
    ),
    "one_step": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True),
        {"cross_fit": True, "n_folds": 3, "targeting": "one_step"},
    ),
    "target_weights": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True),
        {"cross_fit": True, "n_folds": 3, "target_weights": True},
    ),
    "repeats": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True, strata=True),
        {"cross_fit": True, "n_folds": 3, "repeats": 3},
    ),
    "weights": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 94, delta=True, strata=True).assign(
            wt=lambda f: 0.5 + (f["W1"] > 0).astype(float)
        ),
        {},
    ),
    "bootstrap": lambda: (
        extended_frame(BINARY_ACTIVE, 400, 92, delta=True, strata=True),
        {"n_bootstrap": 3},
    ),
    "bootstrap_cross_fit": lambda: (
        extended_frame(BINARY_ACTIVE, 400, 92, delta=True),
        {"cross_fit": True, "n_folds": 3, "n_bootstrap": 3},
    ),
    "three_arm_strata_cross_fit": lambda: (
        extended_frame(THREE_ARM_ACTIVE, 600, 97, delta=True, strata=True),
        {"cross_fit": True, "n_folds": 3},
    ),
    "treatment_delta_strata": lambda: (
        extended_frame(BINARY_ACTIVE, 600, 98, delta=True, treatment_delta=True, strata=True),
        {},
    ),
    "natural_course_strata": lambda: (
        extended_frame(BINARY_ACTIVE, 500, 92, delta=True, strata=True),
        {"estimands": ("ey_obs", "par", "paf", "ey0")},
    ),
}


def check_status(case: str) -> None:
    """Every estimate reports a point and the diagnostic, and refuses all inference."""
    frame, extra = STATUS_CASES[case]()
    result = per_arm_fit(frame, **extra)
    assert result.inference_status == "generated_design_plugin"
    for name in result.estimates:
        estimate = result[name]
        assert np.isfinite(estimate.psi)
        assert np.isfinite(estimate.plugin_std_error)
        for attribute in ("ci", "pvalue", "std_error"):
            with pytest.raises(CapabilityError, match="generated"):
                getattr(estimate, attribute)
    if extra.get("n_bootstrap"):
        # The bootstrap keeps its diagnostic framing: a spread, no interval.
        for name in result.estimates:
            summary = result[name].bootstrap
            assert summary is not None, name
            assert summary.inferential is False, name
        columns = set(result.to_frame().columns)
        assert {"bootstrap_sd", "bootstrap_range_lower", "bootstrap_range_upper"} <= columns
        assert not {"bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_std_err"} & columns


def fitted_factors(result: Any) -> dict[str, np.ndarray]:
    """The fit's own factors, in the keys :func:`composite` reads."""
    nuisance = result.nuisance
    return {
        "g": nuisance.propensity.values,
        "pi": nuisance.missingness,
        "recording": nuisance.treatment_observation,
    }


def arm_scores(result: Any, *, initial: bool = False) -> dict[tuple[float, int], float]:
    """``P_n 1{S = s} C_a / G (Y - Q)`` for each arm and stratum, at ``Q*`` or at ``Qbar_n``."""
    data = result.data
    product = composite(result, fitted_factors(result))
    a = np.asarray(data.treatment, dtype=float)
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    w = np.asarray(data.weights, dtype=float)
    hit_all = np.asarray(data.observed, dtype=bool) & np.asarray(data.treatment_recorded)
    fit = result.nuisance.outcome if initial else result.fluctuations["mean"].targeted
    out = {}
    for j, arm in enumerate(data.arm_codes):
        q = np.asarray(fit.arms[arm], dtype=float)
        terms = w * (hit_all & (a == arm)) / product[:, j] * (y - q)
        for code, region in enumerate(_regions(data)):
            out[arm, code] = float(np.sum(terms[region])) / data.n
    return out


def check_scores(extra: dict[str, Any]) -> None:
    """Every arm-by-stratum score is solved, and the initial scores are not.

    The initial scores are nonzero and differ in sign across the strata of some arm, so
    one coefficient shared across the strata could not solve every block.
    """
    result = per_arm_fit(LONGHAND_CASES["binary_delta_strata"](), **extra)
    solved = arm_scores(result)
    assert max(abs(value) for value in solved.values()) < 1e-8, solved
    initial = arm_scores(result, initial=True)
    assert min(abs(value) for value in initial.values()) > 1e-3, initial
    signs = {arm: {np.sign(initial[arm, code]) for code in range(3)} for arm in (0.0, 1.0)}
    assert any(len(found) == 2 for found in signs.values()), initial


def check_target_weights(case: str) -> None:
    """``target_weights=True``: the covariate ``C_a`` with the weights ``1 / G``."""
    result = per_arm_fit(LONGHAND_CASES[case](), target_weights=True)
    factors, hand = in_sample_longhand(result, weighted=True)
    assert_factors(result, factors)
    assert_matches(result, hand)
    # The witness: the unweighted form gives another targeted fit, so other curves.
    _, plain = in_sample_longhand(result)
    moved = max(np.max(np.abs(plain[name][1] - curve)) for name, (_, curve) in hand.items())
    assert moved > 1e-4


def check_route_identity(case: str) -> None:
    """The missing-outcome route's arm covariate is the composite construction's.

    Without a missing treatment the fit takes the missing-outcome route, with the
    treatment factor as ``g`` and the outcome observation factor as ``pi``. The composite
    route of :func:`~cleverly.estimators.composite.composite_state` divides by the bounded
    product on the composite view. The two agree on every row whose residual the score
    reads, and the composite covariate is zero on the rest.
    """
    from cleverly.estimators.composite import composite_state
    from cleverly.estimators.targeting import build_submodel

    result = per_arm_fit(LONGHAND_CASES[case]())
    assert result.extra.get("missing_data") == "missing_outcome"
    bounds = result.config.g_bounds
    floor = result.config.missingness_bound
    route = build_submodel(
        result.data, result.nuisance, "mean", bounds=bounds, nuisance_bound=floor
    )
    state = composite_state(result.data, result.nuisance, g_bounds=bounds, nuisance_bound=floor)
    view = build_submodel(
        state.data, state.nuisance, "mean", bounds=state.bounds, nuisance_bound=floor
    )
    observed = np.asarray(result.data.observed, dtype=bool)
    np.testing.assert_allclose(route.observed[observed], view.observed[observed], rtol=1e-12)
    assert np.all(view.observed[~observed] == 0.0)
    for arm in route.arms:
        np.testing.assert_allclose(route.arms[arm], view.arms[arm], rtol=1e-12)
    # The witness: the covariate divides by the observation factor, not by g alone.
    data = result.data
    g = np.clip(result.nuisance.propensity.values, *bounds)
    a = np.asarray(data.treatment, dtype=float)
    alone = np.column_stack([(a == arm) / g[:, j] for j, arm in enumerate(data.arm_codes)])
    assert np.max(np.abs(route.observed - alone)[observed]) > 1e-3


# --------------------------------------------------------------------------- the tests


@pytest.mark.parametrize("case", sorted(LONGHAND_CASES))
def test_the_fit_is_the_longhand_construction(case: str) -> None:
    check_longhand(case)


def test_the_cross_fitted_fit_is_the_longhand_construction() -> None:
    check_cv_longhand()


def test_each_stratum_point_is_the_subset_fit() -> None:
    check_stratum_is_the_subset_fit()


def test_the_marginal_point_is_the_stratum_mixture() -> None:
    check_marginal_is_the_mixture()


def test_the_natural_course_reads_the_realised_arms_factor() -> None:
    check_natural_course()


@pytest.mark.parametrize("cross_fit", [False, True])
def test_the_stratified_natural_course(cross_fit: bool) -> None:
    check_natural_course_strata(cross_fit)


def test_the_oracle_outcome_gives_the_truth_on_the_exact_law() -> None:
    check_exact_law()


def test_a_validation_fold_never_reaches_its_own_factors() -> None:
    check_fold_local_nesting()


def test_integer_weights_equal_duplicated_rows() -> None:
    check_weights_equal_duplicates()


def test_a_constant_target_fits_no_learner() -> None:
    check_constant_target()


@pytest.mark.parametrize("repeats", [1, 3])
def test_the_tilt_reads_the_full_covariate_model(repeats: int) -> None:
    check_tilt_reads_the_full_covariate_model(repeats)


@pytest.mark.parametrize("case", sorted(STATUS_CASES))
def test_every_admitted_configuration_withholds_inference(case: str) -> None:
    check_status(case)


@pytest.mark.parametrize(
    "extra",
    [{}, {"fluctuation": "linear"}, {"targeting": "one_step"}, {"target_weights": True}],
    ids=["iterative", "linear", "one_step", "target_weights"],
)
def test_every_arm_and_stratum_score_is_solved(extra: dict[str, Any]) -> None:
    check_scores(extra)


@pytest.mark.parametrize("case", ["binary_delta", "binary_delta_strata", "binary_treatment_delta"])
def test_target_weights_are_the_weighted_longhand(case: str) -> None:
    check_target_weights(case)


@pytest.mark.parametrize("case", ["binary_delta", "three_arm_delta"])
def test_the_missing_outcome_route_is_the_composite_construction(case: str) -> None:
    check_route_identity(case)


def test_the_support_report_names_the_design_product() -> None:
    """The derived row of a per-arm fit conditions on ``Qbar(a, W)``, not on ``W``."""
    cases = {
        "P(A=a,Delta=1|Qbar(a,W))": {"delta": True},
        "P(A=a,Delta=1|Qbar(a,W),S)": {"delta": True, "strata": True},
        "P(A=a,Delta_A=1,Delta=1|Qbar(a,W),S)": {
            "delta": True,
            "treatment_delta": True,
            "strata": True,
        },
    }
    for label, flags in cases.items():
        result = per_arm_fit(extended_frame(BINARY_ACTIVE, 500, 11, **flags))
        report = result.diagnostics.support()
        assert label in report.mechanisms, (label, list(report.mechanisms))
        assert not any(name.endswith("|W)") for name in report.mechanisms), report.mechanisms
        # The row is the product of the fit's own factors.
        raw = [value for value in fitted_factors(result).values() if value is not None]
        product = np.prod(np.stack(raw), axis=0)
        assert report.mechanisms[label]["min"] == pytest.approx(float(product.min()), abs=1e-12)
        assert f"{label} truncated to [" in report.summary()


def test_the_revert_key_still_excludes_every_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the per-arm flag set, a fit with strata or missing data still withholds."""
    monkeypatch.setattr(ctmle_module, "OAT_PER_ARM_INFERENTIAL", True)
    for flags in ({"strata": True}, {"delta": True}, {"delta": True, "treatment_delta": True}):
        result = per_arm_fit(extended_frame(BINARY_ACTIVE, 400, 95, **flags))
        assert result.inference_status == "generated_design_plugin", flags
    complete = per_arm_fit(extended_frame(BINARY_ACTIVE, 400, 95))
    assert complete.inference_status != "generated_design_plugin"


def test_the_diagnostics_name_each_factor_and_stratum() -> None:
    result = per_arm_fit(
        extended_frame(BINARY_ACTIVE, 900, 15, delta=True, treatment_delta=True, strata=True)
    )
    fit = result.ctmle_selection
    assert fit.factors == (
        "P(A=a|Delta_A=1,Qbar(a,W),S)",
        "P(Delta=1|A=a,Delta_A=1,Qbar(a,W),S)",
        "P(Delta_A=1|Qbar(a,W),S)",
    )
    assert len(fit.strata) == 3
    assert set(fit.risk_by_factor) == set(fit.factors)
    assert "3 factors per arm" in fit.describe()
    assert "inside each of 3 baseline strata" in fit.describe()
    summary = fit.summary()
    assert "each factor below ~ Qbar(a, W), inside each baseline stratum" in summary
    assert "1{A = a} ~ Qbar(a, W), one per arm" not in summary
    for factor in fit.factors:
        assert f"  {factor}: " in summary
    text = summary + fit.describe() + result.summary()
    assert "efficien" not in text.lower()
    assert result.nuisance.treatment_observation.shape == (result.data.n, 2)
    result.validate()


def test_save_and_load_keep_the_design(tmp_path: Any) -> None:
    result = per_arm_fit(extended_frame(BINARY_ACTIVE, 500, 96, delta=True, strata=True))
    path = tmp_path / "fit.cleverly"
    result.save(path)
    loaded = cleverly.load(path)
    assert loaded.inference_status == "generated_design_plugin"
    assert loaded.ctmle_selection == result.ctmle_selection
    np.testing.assert_array_equal(
        loaded.nuisance.tilt_missingness, result.nuisance.tilt_missingness
    )
    assert loaded["ate"].psi == result["ate"].psi


# ------------------------------------------------------------------------- the refusals


def _refusal(frame: pd.DataFrame, **extra: Any) -> str:
    settings = {**never_fit_learners(), **extra}
    with pytest.raises((CapabilityError, ValueError)) as caught:
        per_arm_fit(frame, **settings)
    return str(caught.value)


class TestTheRefusals:
    def test_a_selector_strategy_with_a_missing_treatment(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 101, delta=True, treatment_delta=True)
        for extra in ({"strategy": "greedy"}, {"oat_design": "shared"}):
            message = _refusal(frame, **extra)
            assert "The C-TMLE selector strategies have no collaborative score for the " in message
            assert "CTMLE(strategy='oat', oat_design='per_arm') for point estimates." in message

    def test_a_cross_fitted_missing_treatment(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 101, delta=True, treatment_delta=True)
        message = _refusal(frame, cross_fit=True)
        assert "in-sample" in message
        assert "roadmap X53" in message

    def test_the_natural_course_with_a_missing_treatment(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 101, delta=True, treatment_delta=True)
        message = _refusal(frame, estimands=("ey_obs",))
        assert "ey_obs" in message

    def test_the_shared_design_with_cross_fitted_missing_outcomes(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 102, delta=True)
        message = _refusal(frame, cross_fit=True, oat_design="shared")
        assert (
            "the selector strategies and the shared outcome-adaptive design "
            "(oat_design='shared') of C-TMLE (CTMLE, or CollaborativeTMLEMethod) have no "
            "audited selection and inference result with cross-fitted arm-indexed missing "
            "outcomes (X51 in docs/roadmap.md tracks their point estimates). Use the ordinary "
            "TMLE (TMLE, or "
            "TMLEMethod), or CTMLE(strategy='oat', oat_design='per_arm') for point estimates"
        ) in message

    def test_the_shared_design_with_the_natural_course(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 102, delta=True)
        message = _refusal(frame, oat_design="shared", estimands=("ey_obs",))
        assert message.startswith(
            "NaturalCourseMean, PAR and PAF with missing outcomes currently support ordinary "
            "TMLE and, for point estimates, the per-arm outcome-adaptive CTMLE under their "
            "audited implementation contracts; use ordinary TMLE, or "
            "CTMLE(strategy='oat', oat_design='per_arm') for point estimates; the C-TMLE "
            "selector strategies, the shared outcome-adaptive design and DR-TMLE need "
            "separate targeting and inference results (X51 in docs/roadmap.md tracks the "
            "C-TMLE point estimates"
        )

    @pytest.mark.parametrize("source", [BINARY_ACTIVE, THREE_ARM_ACTIVE], ids=["k2", "k3"])
    def test_a_stratum_without_an_arm_in_sample(self, source: OatLaw) -> None:
        frame = extended_frame(source, 600, 103, delta=True, strata=True)
        first = 0.0 if source.k == 2 else source.arm_labels[0]
        frame = frame[~((frame["S"] == 0) & (frame["A"] == first))]
        label = "0" if source.k == 2 else "'0'"
        assert _refusal(frame) == (
            f"inside baseline stratum S=0.0: no recorded row is in arm {label}, so the "
            f"outcome-adaptive treatment design of arm {label} cannot be fitted. Declare fewer "
            "baseline strata."
        )

    def test_an_arm_without_a_respondent_in_sample(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 600, 104, delta=True)
        arm1 = frame["A"] == 1
        frame.loc[arm1, "Delta"] = 0.0
        frame.loc[arm1, "Y"] = np.nan
        message = _refusal(frame)
        assert "no recorded row of arm 1 has an observed outcome" in message
        assert "observation design" in message

    def test_a_fold_without_a_respondent_in_an_arm(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 600, 105, delta=True, strata=True)
        drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        keep = (frame["S"] == 1) & (frame["A"] == 1) & (frame["Delta"] == 1)
        lone = keep & (np.asarray(drawn.assignment) == 0)
        silence = keep & ~lone
        frame.loc[silence, "Delta"] = 0.0
        frame.loc[silence, "Y"] = np.nan
        message = _refusal(
            frame, cross_fit=True, n_folds=FOLDS, split_plan=SplitPlan.from_folds([drawn])
        )
        assert message.startswith(
            "inside baseline stratum S=1.0: a cross-fitting fold has no trainable rows for "
            "the outcome-adaptive observation design of arm 1."
        )

    @pytest.mark.parametrize("source", [BINARY_ACTIVE, THREE_ARM_ACTIVE], ids=["k2", "k3"])
    @pytest.mark.parametrize("stratified", [True, False], ids=["strata", "no_strata"])
    def test_a_fold_without_a_respondent_at_two_and_three_arms(
        self, source: OatLaw, stratified: bool
    ) -> None:
        frame = extended_frame(source, 600, 108, delta=True, strata=stratified)
        drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        arm = source.arm_labels[-1] if source.k == 3 else 1.0
        keep = (frame["A"] == arm) & (frame["Delta"] == 1)
        if stratified:
            keep &= frame["S"] == 1
        lone = keep & (np.asarray(drawn.assignment) == 0)
        silence = keep & ~lone
        frame.loc[silence, "Delta"] = 0.0
        frame.loc[silence, "Y"] = np.nan
        message = _refusal(
            frame, cross_fit=True, n_folds=FOLDS, split_plan=SplitPlan.from_folds([drawn])
        )
        label = repr(arm) if source.k == 3 else "1"
        where = "baseline stratum S=1.0" if stratified else "the sample"
        remedy = (
            "declare fewer baseline strata, or fit with cross_fit=False"
            if stratified
            else "fit with cross_fit=False"
        )
        assert message == (
            f"inside {where}: a cross-fitting fold has no trainable rows for the "
            f"outcome-adaptive observation design of arm {label}. "
            + _POST_DRAW_REMEDY.format(remedy=remedy)
        )
        assert NeverFit.calls == 0

    def test_a_later_draw_refuses_before_the_first_draw_fits(self) -> None:
        """With ``repeats=2``, only the second draw leaves a complement with no respondent.

        Two respondents remain in stratum 1 and arm 1. The first draw puts them in two
        folds, so every complement keeps one. The second puts them in one fold, so that
        fold's complement keeps none. The refusal must come before any learner of the
        first draw.
        """
        frame = extended_frame(BINARY_ACTIVE, 600, 105, delta=True, strata=True)
        first = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        second = random_partition(len(frame), FOLDS, seed=SPLIT_SEED + 1)
        one, two = np.asarray(first.assignment), np.asarray(second.assignment)
        cell = (frame["S"] == 1) & (frame["A"] == 1) & (frame["Delta"] == 1)
        rows = np.flatnonzero(cell.to_numpy())
        pair = next((i, j) for i in rows for j in rows if one[i] != one[j] and two[i] == two[j])
        silence = frame.index[np.setdiff1d(rows, pair)]
        frame.loc[silence, "Delta"] = 0.0
        frame.loc[silence, "Y"] = np.nan
        settings = {"cross_fit": True, "n_folds": FOLDS}
        message = _refusal(
            frame, repeats=2, split_plan=SplitPlan.from_folds([first, second]), **settings
        )
        assert message.startswith(
            "inside baseline stratum S=1.0: a cross-fitting fold has no trainable rows for "
            "the outcome-adaptive observation design of arm 1."
        )
        assert NeverFit.calls == 0
        # The control: the first draw alone passes the preflight and reaches a learner.
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            per_arm_fit(
                frame, **never_fit_learners(), split_plan=SplitPlan.from_folds([first]), **settings
            )
        assert NeverFit.calls == 1

    def test_the_scale_rule_still_refuses(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 107, delta=True)
        frame["Y"] = frame["Y"] + 0.37 * frame["W1"]
        message = _refusal(frame, cross_fit=True, n_folds=FOLDS, q_bounds=None)
        assert message.startswith(
            "a cross-fitted fit of a continuous outcome (Y, family='gaussian') needs a "
            "declared q_bounds."
        )
        assert NeverFit.calls == 0

    def test_the_generator_record_rule_still_refuses(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 107, delta=True)
        drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        plan = SplitPlan([list(np.asarray(drawn.assignment))])
        message = _refusal(frame, cross_fit=True, n_folds=FOLDS, split_plan=plan)
        assert message.startswith("split_plan carries no generator record")
        assert NeverFit.calls == 0

    @pytest.mark.parametrize("delta", [True, False], ids=["delta", "complete"])
    def test_the_drtmle_message_is_unchanged(self, delta: bool) -> None:
        """The stratified DR-TMLE preflight keeps its reduced-regression sentence bit for bit.

        A stratum with no row of arm 1 leaves every reduced regression of that arm with no
        trainable row. In sample, with ``delta=``, and cross-fitted with complete outcomes,
        the guarded DR-TMLE refuses with the shipped sentence before any learner.
        """
        from cleverly.estimators import DRTMLE

        frame = extended_frame(BINARY_ACTIVE, 600, 106, delta=delta, strata=True)
        frame = frame[~((frame["S"] == 1) & (frame["A"] == 1))].reset_index(drop=True)
        settings: dict[str, Any] = (
            {"cross_fit": False} if delta else {"cross_fit": True, "n_folds": FOLDS}
        )
        with pytest.raises(ValueError) as caught:
            DRTMLE(**never_fit_learners(), random_state=0, estimands=("ate",), **settings).fit(
                frame, outcome="Y", treatment="A", **roles(frame)
            )
        assert str(caught.value) == (
            "inside baseline stratum S=1.0: a cross-fitting fold has no trainable rows for a "
            "reduced regression of arm 1. "
            + _POST_DRAW_REMEDY.format(
                remedy="declare fewer baseline strata, or fit with cross_fit=False"
            )
        )
        assert NeverFit.calls == 0


# -------------------------------------------------------------------- the mutation controls


def _with_factor_override(
    monkeypatch: pytest.MonkeyPatch, edit: Callable[[dict[str, np.ndarray]], None]
) -> None:
    original = ctmle_module._fit_per_arm_factors

    def mutated(*args: Any, **kwargs: Any) -> Any:
        factors, diagnostics = original(*args, **kwargs)
        edit(factors)
        return factors, diagnostics

    monkeypatch.setattr(ctmle_module, "_fit_per_arm_factors", mutated)


def _with_specs(monkeypatch: pytest.MonkeyPatch, edit: Callable[[Any, Any], Any]) -> None:
    original = ctmle_module._per_arm_factor_specs

    def mutated(data: Any) -> Any:
        return tuple(edit(spec, data) for spec in original(data))

    monkeypatch.setattr(ctmle_module, "_per_arm_factor_specs", mutated)


def _pool_strata(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ctmle_module._fit_per_arm_factors

    def mutated(data: Any, *args: Any) -> Any:
        return original(replace(data, strata=None), *args)

    monkeypatch.setattr(ctmle_module, "_fit_per_arm_factors", mutated)


def _drop_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    def edit(factors: dict[str, np.ndarray]) -> None:
        if "outcome_observation" in factors:
            factors["outcome_observation"][:] = 1.0

    _with_factor_override(monkeypatch, edit)


def _observation_on_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    every = lambda spec, data: (  # noqa: E731
        replace(spec, rows=lambda arm: np.asarray(data.treatment_recorded, dtype=bool))
        if spec.name == "outcome_observation"
        else spec
    )
    _with_specs(monkeypatch, every)


def _treatment_on_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    every = lambda spec, data: (  # noqa: E731
        replace(spec, rows=lambda arm: np.ones(data.n, dtype=bool))
        if spec.name == "treatment"
        else spec
    )
    _with_specs(monkeypatch, every)


def _broadcast_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    def edit(factors: dict[str, np.ndarray]) -> None:
        if "treatment_observation" in factors:
            column = factors["treatment_observation"][:, [0]].copy()
            factors["treatment_observation"][:] = column

    _with_factor_override(monkeypatch, edit)


def _natural_course_at_arm_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row's observation factor is arm 0's, so the natural course reads arm 0."""

    def edit(factors: dict[str, np.ndarray]) -> None:
        if "outcome_observation" in factors:
            column = factors["outcome_observation"][:, [0]].copy()
            factors["outcome_observation"][:] = column

    _with_factor_override(monkeypatch, edit)


def _constant_fits_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_specs(monkeypatch, lambda spec, data: replace(spec, constant=False))


def _drop_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ctmle_module._fit_per_arm_factors

    def mutated(data: Any, *args: Any) -> Any:
        return original(replace(data, weights=np.ones(data.n)), *args)

    monkeypatch.setattr(ctmle_module, "_fit_per_arm_factors", mutated)


def _full_covariate_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The observation factor is the full-``W`` ``pi(a, W)``, as before the per-arm design."""
    original = CTMLE._outcome_adaptive_nuisances

    def mutated(self: CTMLE, data: Any, base: Any, *, seed: int | None) -> Any:
        nuisance, extra = original(self, data, base, seed=seed)
        if nuisance.tilt_missingness is not None:
            nuisance = replace(nuisance, missingness=nuisance.tilt_missingness)
        return nuisance, extra

    monkeypatch.setattr(CTMLE, "_outcome_adaptive_nuisances", mutated)


def _stitched_design(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-fitted observation factors fitted on the stitched out-of-fold ``Qbar``.

    The base fit's outcome predictions on a fold's training rows come from other folds'
    models, each trained on rows that include the fold's own validation rows.
    """
    original_cross_fit = CTMLE._cross_fit_outcome_adaptive
    original_fit = ctmle_module._fit_per_arm_factors
    stitched: dict[str, np.ndarray] = {}

    def cross_fit(self: CTMLE, data: Any, base: Any, learner: Any, *, seed: int | None) -> Any:
        stitched["q"] = np.column_stack([base.outcome.arms[arm] for arm in data.arm_codes])
        return original_cross_fit(self, data, base, learner, seed=seed)

    def fit(data: Any, generated: Any, learner: Any, train: Any, test: Any) -> Any:
        factors, diagnostics = original_fit(data, generated, learner, train, test)
        if "q" in stitched and "outcome_observation" in factors:
            wrong, _ = original_fit(data, stitched["q"], learner, train, test)
            factors["outcome_observation"] = wrong["outcome_observation"]
        return factors, diagnostics

    monkeypatch.setattr(CTMLE, "_cross_fit_outcome_adaptive", cross_fit)
    monkeypatch.setattr(ctmle_module, "_fit_per_arm_factors", fit)


def _drop_mass_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each marginal curve is the mass-weighted sum of the stratum curves, without the mass term.

    The marginal estimates come from :meth:`~cleverly.TMLE._estimates_for` before the
    stratum estimates, so the first wrapper keeps the marginal mapping and the second
    replaces each marginal curve once the stratum curves exist.
    """
    from cleverly.estimators import TMLE
    from cleverly.estimators._strata import stratum_probabilities
    from cleverly.targets.base import stratum_alias

    original_for = TMLE._estimates_for
    original_strata = TMLE._stratum_estimates
    held: dict[str, Any] = {}

    def estimates_for(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        out = original_for(self, data, *args, **kwargs)
        if data.has_strata and kwargs.get("index") is None:
            held["marginal"] = out
        return out

    def stratum_estimates(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        out, codes = original_strata(self, data, *args, **kwargs)
        marginal = held.pop("marginal")
        mass = stratum_probabilities(data)
        for name, estimate in list(marginal.items()):
            stacked = sum(
                p * out[stratum_alias(name, data.stratum_label(code))].influence_curve
                for code, p in enumerate(mass)
            )
            marginal[name] = replace(estimate, influence_curve=stacked)
        return out, codes

    monkeypatch.setattr(TMLE, "_estimates_for", estimates_for)
    monkeypatch.setattr(TMLE, "_stratum_estimates", stratum_estimates)


def _tilt_reads_the_design(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        missingness_module,
        "tilt_missingness",
        lambda nuisance, lower: nuisance.bounded_missingness(lower),
    )


def _shared_is_per_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ctmle_module,
        "is_per_arm_oat",
        lambda estimator: getattr(estimator, "strategy", "") == "oat",
    )


def _admits_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ctmle_module, "OAT_PER_ARM_INFERENTIAL", True)
    monkeypatch.setattr(ctmle_module, "per_arm_design_admits", lambda estimator, data: True)


def _shared_refusal() -> None:
    TestTheRefusals().test_the_shared_design_with_the_natural_course()


MUTATIONS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], Callable[[], None]]] = {
    "factors_pooled_over_strata": (_pool_strata, check_stratum_is_the_subset_fit),
    "observation_factor_dropped": (_drop_observation, lambda: check_longhand("binary_delta")),
    "observation_factor_on_every_recorded_row": (
        _observation_on_every_row,
        lambda: check_longhand("three_arm_delta"),
    ),
    "treatment_factor_on_every_row": (
        _treatment_on_every_row,
        lambda: check_longhand("binary_treatment_delta"),
    ),
    "recording_broadcast_from_arm_zero": (
        _broadcast_recording,
        lambda: check_longhand("binary_treatment_delta"),
    ),
    "natural_course_reads_arm_zero": (_natural_course_at_arm_zero, check_natural_course),
    "marginal_curve_drops_the_mass_term": (_drop_mass_term, check_marginal_is_the_mixture),
    "observation_factor_on_the_full_covariates": (_full_covariate_observation, check_exact_law),
    "observation_factor_on_stitched_out_of_fold_qbar": (
        _stitched_design,
        check_fold_local_nesting,
    ),
    "constant_target_fits_the_learner": (_constant_fits_anyway, check_constant_target),
    "factor_fits_drop_the_weights": (_drop_weights, check_weights_equal_duplicates),
    "tilt_reads_the_design_factor": (
        _tilt_reads_the_design,
        check_tilt_reads_the_full_covariate_model,
    ),
    "shared_design_is_per_arm": (_shared_is_per_arm, _shared_refusal),
    "status_key_admits_extensions": (_admits_extensions, lambda: check_status("strata")),
}


class TestEachMutationFailsItsCheck:
    @pytest.mark.parametrize("name", sorted(MUTATIONS))
    def test_the_mutation_fails_the_check(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        mutation, check = MUTATIONS[name]
        mutation(monkeypatch)
        # Only an assertion of the check counts: an exception from the mutated fit would
        # say nothing about whether the check reads the mutated component.
        with pytest.raises(AssertionError):
            check()
