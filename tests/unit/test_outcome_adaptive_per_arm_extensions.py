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
* an exact finite law: with the oracle ``Qbar`` the point is the truth whatever the design;
* fold-local nesting, fixed weights, the status matrix, the refusals, the constant-target
  rule, the missingness tilt, and save and load.

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
from cleverly.learners.crossfit import SplitPlan, random_partition
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
from tests.unit._natural_course_support import never_fit_learners
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
    depends on ``W1`` and ``W2``.  ``S`` cuts ``W3`` into three strata, so the treatment
    depends on ``S`` given ``Qbar(a, W)``.  ``S`` and is the last
    covariate, which :class:`~tests.studies.oat_per_arm_laws.OracleOutcome` ignores.
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


def target(result: Any, q: dict[float, np.ndarray], product: np.ndarray) -> Longhand:
    """One scalar logistic fluctuation per arm and stratum, then the means and curves.

    The covariate is ``C_a / P(C_a = 1 | q_a)``.  A stratum mean is the weighted mean of the
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
        offset = logit(_shrink(q[arm]))
        targeted = np.empty(data.n)
        for region in _regions(data):
            epsilon = 0.0
            for _ in range(200):
                fitted = expit(offset[region] + epsilon * h[region])
                score = float(np.sum((w * hit * h)[region] * (y[region] - fitted)))
                curvature = float(np.sum((w * hit * h**2)[region] * fitted * (1.0 - fitted)))
                step = score / curvature
                epsilon += step
                if abs(step) < 1e-15:
                    break
            targeted[region] = _shrink(expit(offset[region] + epsilon * h[region]))
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


def in_sample_longhand(result: Any) -> tuple[dict[str, np.ndarray], Longhand]:
    data = result.data
    q = {arm: np.asarray(result.nuisance.outcome.arms[arm], dtype=float) for arm in data.arm_codes}
    every = np.ones(data.n, dtype=bool)
    factors = longhand_factors(data, q, every, every)
    return factors, target(result, q, composite(result, factors))


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
    # The marginal curve carries the mass term: it is not the sum of the stratum curves.
    stacked = sum(
        p * np.asarray(result[f"ey1[{data.stratum_label(code)}]"].influence_curve)
        for code, p in enumerate(mass)
    )
    mass_term = np.asarray(result["ey1"].influence_curve) - stacked
    assert np.sqrt(np.mean(mass_term**2)) > 1e-3


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
    """:mod:`tests.discrete_law_oat`, with half of every ``W in {0, 1}``, ``A = 0`` cell missing.

    Every such cell count is even, so the respondents of each cell keep the cell's outcome
    share and the realised sample is exactly a MAR law with ``E[Y | A, Delta = 1, W] =
    Qbar_0(A, W)``.
    """
    frame = law.frame()
    respond = np.ones(len(frame), dtype=bool)
    for (w, a, _), rows in frame.groupby(["W", "A", "Y"]).groups.items():
        if a == 0 and w in (0, 1):
            index = np.asarray(list(rows))
            assert index.size % 2 == 0
            respond[index[: index.size // 2]] = False
    frame["Delta"] = respond.astype(float)
    frame.loc[~respond, "Y"] = np.nan
    return frame


def check_exact_law() -> None:
    """With the oracle ``Qbar`` the point is the truth for any design, and a wrong one misses."""
    frame = exact_frame()
    for estimands, truth in ((("ey1", "ey0"), law.TRUTH),):
        settings = {
            "strategy": "oat",
            "treatment_learner": law.SaturatedCategorical(),
            "missingness_learner": LogisticRegression(),
            "cross_fit": False,
            "simultaneous": False,
            "estimands": estimands,
            "random_state": 0,
        }
        exact = (
            CTMLE(outcome_learner=law.OracleOutcome(), **settings)
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        for arm in (0, 1):
            assert exact[f"ey{arm}"].psi == pytest.approx(truth[arm], abs=1e-12)
        # The contrast: the respondents alone miss the truth, so the missing outcomes matter.
        respondents = frame[(frame["A"] == 0) & (frame["Delta"] == 1)]
        assert abs(float(respondents["Y"].mean()) - truth[0]) > 1e-3


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
    """A logistic regression that records each binary target it is fitted to."""

    targets: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _CountingLogistic:
        type(self).targets.append(np.asarray(y, dtype=float).copy())
        return super().fit(X, y, sample_weight=sample_weight)


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
}


def check_status(case: str) -> None:
    frame, extra = STATUS_CASES[case]()
    result = per_arm_fit(frame, **extra)
    assert result.inference_status == "generated_design_plugin"
    for name in result.estimates:
        estimate = result[name]
        assert np.isfinite(estimate.psi)
        assert np.isfinite(estimate.plugin_std_error)
        with pytest.raises(Exception, match="generated"):
            _ = estimate.ci


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
    text = fit.summary() + fit.describe() + result.summary()
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
        assert "F21" in message

    def test_the_natural_course_with_a_missing_treatment(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 101, delta=True, treatment_delta=True)
        message = _refusal(frame, estimands=("ey_obs",))
        assert "ey_obs" in message

    def test_the_shared_design_with_cross_fitted_missing_outcomes(self) -> None:
        frame = extended_frame(BINARY_ACTIVE, 300, 102, delta=True)
        message = _refusal(frame, cross_fit=True, oat_design="shared")
        assert (
            "C-TMLE selector strategies and the shared outcome-adaptive design "
            "(oat_design='shared') have no audited selection and inference result with "
            "cross-fitted arm-indexed missing outcomes. Use the ordinary TMLE (TMLE, or "
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
            "separate targeting and inference results"
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

    def test_the_drtmle_message_is_unchanged(self) -> None:
        from cleverly.estimators import DRTMLE

        frame = extended_frame(BINARY_ACTIVE, 600, 106, delta=True, strata=True)
        drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        cell = (frame["S"] == 1) & (frame["A"] == 1) & (np.asarray(drawn.assignment) != 0)
        frame = frame[~cell].reset_index(drop=True)
        drawn = random_partition(len(frame), FOLDS, seed=SPLIT_SEED)
        cell = (frame["S"] == 1) & (frame["A"] == 1)
        frame = frame[~(cell & (np.asarray(drawn.assignment) != 0))].reset_index(drop=True)
        with pytest.raises((CapabilityError, ValueError)) as caught:
            DRTMLE(
                **never_fit_learners(),
                cross_fit=True,
                n_folds=FOLDS,
                random_state=0,
                estimands=("ate",),
            ).fit(frame, outcome="Y", treatment="A", delta="Delta", strata="S")
        message = str(caught.value)
        if "reduced regression" in message:
            assert "a cross-fitting fold has no trainable rows for a reduced regression" in message


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
        # A crash of the mutated fit also fails the check, as a learner refusing a
        # one-class target does.
        with pytest.raises((AssertionError, pytest.fail.Exception, ValueError)):
            check()
