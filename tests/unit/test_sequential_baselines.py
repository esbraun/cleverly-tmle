"""Sequential g-computation and IPW: exact laws, stacked curves and their witnesses.

Every exact law here realises its distribution in a finite sample, so a saturated GLM is the
population fit.  A saturated parametric model is the nonparametric one, so the stacked
estimating-equation curve of either baseline equals the efficient influence function there,
which each law module states by a Gateaux derivative of the functional alone.  That makes the
exact-law curve checks independent of the code they check.

The random-sample checks (W5, W6, W7, W10) use main-terms GLMs, which do not saturate, so the
stacked terms are not the efficient influence function there.  W6 differentiates the estimator
in each observation weight, and W10 rebuilds both estimators and their curves in numpy.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, DataError, MethodConfigurationError
from cleverly.learners import GLM
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.estimator import _refit_bound, longitudinal_truncation_curve
from cleverly.longitudinal.regimen import resolve_plans
from cleverly.longitudinal.sequential import (
    FoldRecursionCell,
    _untargeted_recursion_in_fold,
)
from cleverly.methods import (
    GComputationMethod,
    IPWMethod,
    ModelSpec,
    ParametricGFormulaMethod,
    Targeting,
)

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_mtp as mtp_law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import discrete_law_longitudinal_policy as policy
from .. import discrete_law_survival as survival
from .. import longitudinal_mtp as mtp
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import declared_regimens

#: A GLM that saturates every design of the exact laws: products of every subset of the
#: binary columns, with each impossible product aliased out.
SATURATED = GLM(interactions=8, tol=1e-13)
#: Bounds that never bind on the exact laws, whose conditionals lie in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
#: The three estimators each witness runs.
VARIANTS: dict[str, dict[str, Any]] = {
    "gcomp": {},
    "hajek": {"normalization": "hajek"},
    "horvitz_thompson": {"normalization": "horvitz_thompson"},
}
END_OF_STUDY = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}
CENSORED = {**END_OF_STUDY, "censoring": ["C1", "C2"]}


def _estimator(variant: str, regimens: Any, learner: Any = SATURATED, **settings: Any) -> Any:
    settings = {"g_bounds": NO_TRUNCATION, "simultaneous": False, **settings}
    if variant == "gcomp":
        return SequentialGComputation(
            regimens,
            outcome_learner=learner,
            treatment_learner=learner,
            censoring_learner=learner,
            **settings,
        )
    return SequentialIPW(
        regimens,
        treatment_learner=learner,
        censoring_learner=learner,
        **VARIANTS[variant],
        **settings,
    )


def _fit(variant: str, regimens: Any, frame: Any, columns: dict[str, Any], **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(variant, regimens, **settings).fit(frame, **columns)


# ---------------------------------------------------------------- exact laws


def _binary_case() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Callable[[str], Any]]:
    return (
        declared_regimens(binary.REGIMEN_SPEC),
        binary.frame(),
        CENSORED,
        {"reference": binary.REGIMEN_REFERENCE},
        binary.eif,
    )


def _survival_case() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Callable[[str], Any]]:
    return (
        declared_regimens(survival.REGIMEN_SPEC),
        survival.frame(),
        {**CENSORED, "outcome": ["Y1", "Y2"]},
        {"reference": survival.REGIMEN_REFERENCE},
        survival.eif,
    )


def _competing_case() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Callable[[str], Any]]:
    return (
        declared_regimens(competing.REGIMEN_SPEC),
        competing.frame(),
        {**CENSORED, "outcome": competing.outcome_columns()},
        {"reference": competing.REGIMEN_REFERENCE},
        competing.eif,
    )


def _multivalue_case() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Callable[[str], Any]]:
    return (
        declared_regimens(multivalue.REGIMEN_SPEC),
        multivalue.frame(),
        END_OF_STUDY,
        {"reference": multivalue.REFERENCE},
        lambda name: multivalue.eif_at(multivalue.PROBS, name),
    )


def _policy_case() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Callable[[str], Any]]:
    labels = tuple(policy.PLANS)
    return (
        policies.regimens(labels),
        multivalue.frame(),
        END_OF_STUDY,
        {"reference": labels[0]},
        lambda name: policy.eif_policy(policy.PROBS, name),
    )


CASES: dict[str, tuple[Callable[[], Any], dict[str, float], Any]] = {
    "binary": (_binary_case, binary.TRUTH, binary.first_row_of),
    "survival": (_survival_case, survival.TRUTH, survival.first_row_of),
    "competing": (_competing_case, competing.TRUTH, competing.first_row_of),
    "three_arm": (_multivalue_case, multivalue.TRUTH, multivalue.first_row_of),
    "policy": (_policy_case, policy.TRUTH, multivalue.first_row_of),
}


@pytest.fixture(scope="module")
def exact_fits() -> dict[tuple[str, str], Any]:
    fits = {}
    for case, (build, _, _) in CASES.items():
        regimens, frame, columns, settings, _ = build()
        for variant in VARIANTS:
            fits[(case, variant)] = _fit(variant, regimens, frame, columns, **settings)
    return fits


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", list(CASES))
def test_w1_w2_w4_every_estimate_is_the_g_formula(
    exact_fits: dict[tuple[str, str], Any], case: str, variant: str
) -> None:
    """W1, W2, W4: the estimate equals the truth on every exact law, both directions.

    The three-arm law reads arm 2 at each node, the policy law averages per-arm predictions
    under a known density, and the survival and competing laws compose the event and the
    all-cause survival factor.  A wrong arm index, a dropped policy weight, a missing factor
    of the cumulative ratio or a wrong Hajek denominator each moves an estimate here.
    """
    result = exact_fits[(case, variant)]
    truth = CASES[case][1]
    assert set(result.estimates) == set(truth)
    for name, value in truth.items():
        assert result.psi(name) == pytest.approx(value, abs=1e-10)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("case", list(CASES))
def test_the_stacked_curve_is_the_gateaux_derivative_on_a_saturated_law(
    exact_fits: dict[tuple[str, str], Any], case: str, variant: str
) -> None:
    """A saturated GLM is the nonparametric model, so the stacked curve is the EIF.

    This reads every term of the stacked curves: the chain of the g-computation curve and
    the mechanism projection of the IPW curve.  Dropping either moves the curve here.
    """
    result = exact_fits[(case, variant)]
    _, _, _, _, eif = CASES[case][0]()
    rows = CASES[case][2]()
    assert result.inference_status == "influence_curve"
    for name in CASES[case][1]:
        np.testing.assert_allclose(result.influence_curves[name][rows], eif(name), atol=1e-10)


def test_g_computation_of_a_modified_treatment_policy_is_exact() -> None:
    """Q7: g-computation predicts each node at the policy's level, with no density ratio."""
    labels = ("natural", "up", "history", "random")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _estimator("gcomp", mtp.regimens(labels), reference=labels[0]).fit(
            mtp_law.frame(), **END_OF_STUDY
        )
    rows = mtp_law.first_row_of()
    for label in labels:
        name = f"ey_regimen[{label}]"
        assert result.psi(name) == pytest.approx(mtp_law.truth(name), abs=1e-10)
        np.testing.assert_allclose(
            result.influence_curves[name][rows], mtp_law.eif(name), atol=1e-10
        )


def test_ipw_refuses_a_modified_treatment_policy() -> None:
    with pytest.raises(CapabilityError, match="density ratio of the shifted law"):
        _fit("hajek", mtp.regimens(("natural", "up")), mtp_law.frame(), END_OF_STUDY)


# ------------------------------------------------------------- W3 point masses


def _one_hot(level: str) -> Any:
    from cleverly.interventions import Stochastic

    def density(frame: Any) -> Any:
        values = np.zeros((len(frame), len(policies.ARM_LABELS)))
        values[:, list(policies.ARM_LABELS).index(level)] = 1.0
        return pd.DataFrame(values, columns=list(policies.ARM_LABELS))

    return Stochastic(density, f"one_hot_{level}", density_kind="known")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_w3_the_policy_path_on_a_point_mass_is_the_regimen(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """W3: with collapse off, a one-hot policy runs the policy path and equals the regimen.

    The policy path adds an arm block to the outcome design.  It is constant on the fit rows,
    so the GLM aliases it, and the estimate and curve equal the label path's.
    """
    high = policies.ARM_LABELS[-1]
    original = resolve_plans
    monkeypatch.setattr(
        estimator_module,
        "resolve_plans",
        lambda regimens, data, **kw: original(regimens, data, _collapse_point_masses=False, **kw),
    )
    point = DynamicRegimen("plan", (_one_hot(high), _one_hot(high)))
    stochastic = _fit(variant, {"plan": point}, multivalue.frame(), END_OF_STUDY)
    assert stochastic.fits["plan"].policy != ()
    monkeypatch.undo()
    label = _fit(variant, {"plan": (high, high)}, multivalue.frame(), END_OF_STUDY)
    name = "ey_regimen[plan]"
    assert stochastic.psi(name) == pytest.approx(label.psi(name), abs=1e-12)
    np.testing.assert_allclose(
        stochastic.influence_curves[name], label.influence_curves[name], atol=1e-12
    )


# --------------------------------------------------------------- random laws


def _draw_binary(n: int, seed: int, *, weighted: bool = True) -> pd.DataFrame:
    """A toy law: continuous W and L2, binary nodes, censoring."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.binomial(1, _expit(0.3 + 0.5 * w))
    c1 = rng.binomial(1, _expit(1.8 + 0.4 * w - 0.5 * a1))
    l2 = 0.5 * w + 0.4 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, _expit(-0.2 + 0.6 * l2 + 0.5 * a1))
    c2 = rng.binomial(1, _expit(1.6 + 0.4 * l2 + 0.4 * a2))
    y = rng.binomial(1, _expit(-0.5 + 0.4 * w + 0.6 * l2 + 0.4 * a1 + 0.5 * a2))
    frame = pd.DataFrame(
        {
            "W": w,
            "A1": a1.astype(float),
            "C1": c1.astype(float),
            "L2": np.where(c1 == 1, l2, np.nan),
            "A2": np.where(c1 == 1, a2, np.nan),
            "C2": np.where(c1 == 1, c2, np.nan),
            "Y": np.where((c1 == 1) & (c2 == 1), y, np.nan),
        }
    )
    frame["wt"] = rng.uniform(0.5, 2.0, n) if weighted else 1.0
    return frame


def _draw_survival(n: int, seed: int, *, causes: int = 1) -> tuple[pd.DataFrame, Any]:
    """Two nodes, an absorbing event after each, one or two causes, censoring, weights."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.binomial(1, _expit(0.2 + 0.5 * w))
    c1 = rng.binomial(1, _expit(2.0 + 0.3 * w - 0.4 * a1))
    e1 = rng.binomial(1, _expit(-1.5 + 0.5 * w - 0.6 * a1))
    k1 = rng.binomial(1, 0.4, n) + 1
    l2 = 0.4 * w - 0.3 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, _expit(-0.1 + 0.5 * l2 + 0.6 * a1))
    c2 = rng.binomial(1, _expit(1.8 + 0.3 * l2 + 0.3 * a2))
    e2 = rng.binomial(1, _expit(-1.2 + 0.5 * l2 - 0.5 * a2 + 0.3 * w))
    k2 = rng.binomial(1, 0.5, n) + 1
    in2 = (c1 == 1) & (e1 == 0)
    observed2 = in2 & (c2 == 1)
    frame = pd.DataFrame(
        {
            "W": w,
            "A1": a1.astype(float),
            "C1": c1.astype(float),
            "L2": np.where(in2, l2, np.nan),
            "A2": np.where(in2, a2, np.nan),
            "C2": np.where(in2, c2, np.nan),
            "wt": rng.uniform(0.5, 2.0, n),
        }
    )
    event1 = np.where(c1 == 1, e1, np.nan)
    event2 = np.where(observed2, e2, np.nan)
    if causes == 1:
        frame["Y1"], frame["Y2"] = event1, event2
        return frame, ["Y1", "Y2"]
    for cause in (1, 2):
        frame[f"R{cause}_1"] = np.where(c1 == 1, (e1 == 1) & (k1 == cause), np.nan)
        frame[f"R{cause}_2"] = np.where(observed2, (e2 == 1) & (k2 == cause), np.nan)
    return frame, {"r1": ["R1_1", "R1_2"], "r2": ["R2_1", "R2_2"]}


def _draw_gaussian(n: int, seed: int) -> pd.DataFrame:
    """Law G, two nodes: bounded linear L and Y, logistic treatment and retention."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, n)
    v = rng.binomial(1, 0.5, n)
    a1 = rng.binomial(1, _expit(0.2 + 0.6 * w - 0.4 * v))
    c1 = rng.binomial(1, _expit(1.8 + 0.4 * w - 0.3 * a1))
    l2 = 0.3 * w - 0.4 * a1 + rng.uniform(-1.0, 1.0, n)
    a2 = rng.binomial(1, _expit(-0.1 + 0.5 * l2 + 0.6 * a1 - 0.3 * v))
    c2 = rng.binomial(1, _expit(1.6 + 0.3 * l2 + 0.3 * a2))
    y = 1.0 + 0.5 * w + 0.3 * v + 0.6 * l2 - 0.8 * a2 + rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame(
        {
            "W": w,
            "V": v.astype(float),
            "A1": a1.astype(float),
            "C1": c1.astype(float),
            "L2": np.where(c1 == 1, l2, np.nan),
            "A2": np.where(c1 == 1, a2, np.nan),
            "C2": np.where(c1 == 1, c2, np.nan),
            "Y": np.where((c1 == 1) & (c2 == 1), y, np.nan),
            "wt": rng.uniform(0.5, 2.0, n),
        }
    )


#: The support of law G's outcome, which declares its scale.
G_BOUNDS = (-2.5, 4.0)


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


TIGHT = GLM(tol=1e-14)
GAUSSIAN = GLM(family="gaussian", tol=1e-14)


def _random_case(
    kind: str, n: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """A random draw, its columns, and the settings each estimator adds."""
    if kind == "binary":
        frame = _draw_binary(n, seed)
        return frame, {**CENSORED, "baseline": ["W"], "weights": "wt"}, {}
    if kind in ("survival", "competing"):
        frame, outcome = _draw_survival(n, seed, causes=1 if kind == "survival" else 2)
        return frame, {**CENSORED, "outcome": outcome, "weights": "wt"}, {}
    frame = _draw_gaussian(n, seed)
    columns = {**CENSORED, "baseline": ["W", "V"], "weights": "wt"}
    return frame, columns, {"q_bounds": G_BOUNDS}


def _random_fit(
    variant: str, frame: pd.DataFrame, columns: dict[str, Any], settings: dict[str, Any]
) -> Any:
    outcome = GAUSSIAN if "q_bounds" in settings else TIGHT
    regimens = {"always": 1, "never": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if variant == "gcomp":
            estimator = SequentialGComputation(
                regimens,
                outcome_learner=outcome,
                treatment_learner=TIGHT,
                censoring_learner=TIGHT,
                simultaneous=False,
                **settings,
            )
        else:
            estimator = SequentialIPW(
                regimens,
                treatment_learner=TIGHT,
                censoring_learner=TIGHT,
                simultaneous=False,
                **VARIANTS[variant],
                **settings,
            )
        return estimator.fit(frame, **columns)


RANDOM_KINDS = ("binary", "survival", "competing", "gaussian")


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("kind", RANDOM_KINDS)
def test_w5_every_stacked_curve_has_mean_zero(kind: str, variant: str) -> None:
    """W5: each block of the stack uses the fit's own empirical measure."""
    frame, columns, settings = _random_case(kind, 400, 11)
    result = _random_fit(variant, frame, columns, settings)
    for name, curve in result.influence_curves.items():
        assert abs(float(np.mean(curve))) <= 1e-10 * float(np.mean(np.abs(curve))), name


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("kind", RANDOM_KINDS)
def test_w6_the_curve_is_the_derivative_in_each_observation_weight(kind: str, variant: str) -> None:
    """W6: ``IF_i / (n w_i)`` is the derivative of the estimate in unit ``i``'s weight.

    The weights are normalized to mean one inside the fit, so the derivative is taken in the
    raw column and divided by the raw mean.  Every term of the stacked curve enters it.
    """
    frame, columns, settings = _random_case(kind, 200, 6)
    _assert_weight_derivative(lambda data: _random_fit(variant, data, columns, settings), frame)


def _assert_weight_derivative(
    fit: Callable[[pd.DataFrame], Any], frame: pd.DataFrame, *, rows: int = 4, seed: int = 3
) -> Any:
    """Assert W6 at ``rows`` random units, for the first two reported names, and return the fit."""
    base = fit(frame)
    raw = frame["wt"].to_numpy()
    scale = raw.mean()
    normalized = raw / scale
    n = raw.size
    rng = np.random.default_rng(seed)
    step = 1e-5
    for name in list(base.estimates)[:2]:
        curve = base.influence_curves[name]
        for row in rng.choice(n, rows, replace=False):
            values = []
            for sign in (1.0, -1.0):
                moved = frame.copy()
                moved.loc[moved.index[row], "wt"] = raw[row] + sign * step
                values.append(fit(moved).psi(name))
            numeric = (values[0] - values[1]) / (2.0 * step)
            analytic = curve[row] / (n * normalized[row]) / scale
            assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-9), (name, row)
    return base


@pytest.mark.parametrize("variant", ["hajek", "horvitz_thompson"])
def test_w6_holds_where_g_bounds_bind(variant: str) -> None:
    """A bounded prefix has derivative 0 in the mechanism, and the curve drops its term."""
    frame = _draw_binary(200, 6)
    columns = {**CENSORED, "baseline": ["W"], "weights": "wt"}
    settings = {"g_bounds": (0.08, 1.0)}

    def fit(data: pd.DataFrame) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SequentialIPW(
                {"always": 1, "never": 0},
                treatment_learner=TIGHT,
                censoring_learner=TIGHT,
                simultaneous=False,
                **VARIANTS[variant],
                **settings,
            ).fit(data, **columns)

    base = _assert_weight_derivative(fit, frame, rows=6)
    bound = [
        np.any(f.cumulative[:, -1] != f.cumulative_unbounded[:, -1]) for f in base.fits.values()
    ]
    assert any(bound)


def test_w6_holds_where_the_gaussian_clip_binds() -> None:
    """A Gaussian prediction outside [0, 1] is clipped, and the chain drops its row."""
    frame = _draw_three_arm(300, 7)
    rng = np.random.default_rng(1)
    observed = frame["Y"].notna()
    frame.loc[observed, "Y"] = frame.loc[observed, "L2"] + 0.1 * rng.normal(size=observed.sum())
    leavers = frame.index[(frame["C1"] == 1) & (frame["C2"] == 0)][:6]
    frame.loc[leavers, "L2"] = frame["L2"].max() + 2.0 + np.arange(6)
    regimens = {"bc": ("b", "c"), "cc": ("c", "c")}

    def fit(data: pd.DataFrame) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SequentialGComputation(
                regimens,
                outcome_learner=GAUSSIAN,
                treatment_learner=TIGHT,
                censoring_learner=TIGHT,
                simultaneous=False,
            ).fit(data, **THREE_ARM)

    base = _assert_weight_derivative(fit, frame, rows=8, seed=11)
    clipped = []
    for regimen in base.fits.values():
        step = next(step for step in regimen.steps if step.time == 2)
        model_scale = step.initial[step.at_risk]
        clipped.append(np.any((model_scale <= 0.0) | (model_scale >= 1.0)))
    assert any(clipped)


ARMS = ("a", "b", "c")
THREE_ARM = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
    "weights": "wt",
}


def _draw_three_arm(n: int, seed: int) -> pd.DataFrame:
    """Multinomial treatment nodes with continuous W and L2, censoring and weights."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)

    def multinomial(first: Any, second: Any) -> Any:
        probability = np.exp(np.column_stack([np.zeros(n), first, second]))
        probability /= probability.sum(axis=1, keepdims=True)
        return (rng.uniform(size=n)[:, None] > probability.cumsum(axis=1)).sum(axis=1)

    a1 = multinomial(0.2 + 0.4 * w, -0.1 - 0.3 * w)
    c1 = rng.binomial(1, _expit(2.0 + 0.3 * w - 0.3 * (a1 == 2)))
    l2 = 0.5 * w + 0.3 * (a1 == 1) - 0.2 * (a1 == 2) + rng.normal(size=n)
    a2 = multinomial(0.1 + 0.5 * l2 + 0.3 * (a1 == 1), -0.2 - 0.4 * l2)
    c2 = rng.binomial(1, _expit(1.8 + 0.3 * l2 + 0.2 * (a2 == 1)))
    y = rng.binomial(1, _expit(-0.3 + 0.4 * w + 0.5 * l2 + 0.3 * (a2 == 1) - 0.4 * (a2 == 2)))
    labels = np.asarray(ARMS, dtype=object)
    return pd.DataFrame(
        {
            "W": w,
            "A1": labels[a1],
            "C1": c1.astype(float),
            "L2": np.where(c1 == 1, l2, np.nan),
            "A2": np.where(c1 == 1, labels[a2], None),
            "C2": np.where(c1 == 1, c2, np.nan),
            "Y": np.where((c1 == 1) & (c2 == 1), y, np.nan),
            "wt": rng.uniform(0.5, 2.0, n),
        }
    )


def _first_policy(frame: pd.DataFrame) -> pd.DataFrame:
    w = np.asarray(frame["W"], dtype=float)
    share = 0.6 * _expit(w)
    return pd.DataFrame(np.column_stack([share, np.full(w.size, 0.2), 0.8 - share]), columns=ARMS)


def _second_policy(frame: pd.DataFrame) -> pd.DataFrame:
    w = np.asarray(frame["W"], dtype=float)
    share = 0.5 * _expit(-w)
    return pd.DataFrame(np.column_stack([np.full(w.size, 0.3), share, 0.7 - share]), columns=ARMS)


def _second_rule(frame: pd.DataFrame) -> Any:
    return np.where(np.asarray(frame["L2"]) > 0, "b", "c")


def _three_arm_regimens(kind: str) -> dict[str, Any]:
    from cleverly.interventions import Stochastic

    if kind == "static":
        first: Any = ("b", "c")
    elif kind == "rule":
        first = DynamicRegimen("rule", ("b", _second_rule), rule_kind="known")
    else:
        first = DynamicRegimen(
            "policy",
            (
                Stochastic(_first_policy, "p1", density_kind="known"),
                Stochastic(_second_policy, "p2", density_kind="known"),
            ),
        )
    return {kind: first, "aa": ("a", "a")}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("kind", ["static", "rule", "policy"])
def test_w5_w6_hold_on_multinomial_nodes(kind: str, variant: str) -> None:
    """The multinomial b-term and the rule and policy nodes, off a saturated law."""
    frame = _draw_three_arm(200, 5)
    regimens = _three_arm_regimens(kind)

    def fit(data: pd.DataFrame) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if variant == "gcomp":
                estimator = SequentialGComputation(
                    regimens,
                    outcome_learner=TIGHT,
                    treatment_learner=TIGHT,
                    censoring_learner=TIGHT,
                    simultaneous=False,
                )
            else:
                estimator = SequentialIPW(
                    regimens,
                    treatment_learner=TIGHT,
                    censoring_learner=TIGHT,
                    simultaneous=False,
                    **VARIANTS[variant],
                )
            return estimator.fit(data, **THREE_ARM)

    base = _assert_weight_derivative(fit, frame, rows=3)
    assert base.inference_status == "influence_curve"
    for name, curve in base.influence_curves.items():
        assert abs(float(np.mean(curve))) <= 1e-10 * float(np.mean(np.abs(curve))), name


def test_w7_each_stacked_term_moves_the_standard_error() -> None:
    """W7: the IPW projection and the g-computation chain are not vanishing terms.

    On law G the fixed-mechanism IPW curve, and the g-computation curve without its node
    terms, each give a standard error that differs from the stacked one.
    """
    frame, columns, settings = _random_case("gaussian", 2000, 21)
    for variant in VARIANTS:
        stacked = _random_fit(variant, frame, columns, settings)
        name = "ey_regimen[always]"
        fit = stacked.fits["always"]
        full = float(np.std(fit.influence_curve_scaled))
        if variant == "gcomp":
            first = fit.steps[0]
            weights = stacked.data.weights
            reduced = float(np.std(weights * (first.initial - fit.psi_scaled)))
            assert reduced < 0.9 * full, (variant, reduced, full)
        else:
            fixed = _fixed_mechanism_curve(stacked, name)
            assert abs(float(np.std(fixed)) - full) > 0.005 * full, (variant, fixed, full)


def _fixed_mechanism_curve(result: Any, name: str) -> Any:
    from cleverly.longitudinal.baselines import fit_ipw_cell

    plan = result.replay_recipe.plans[0]
    fit = fit_ipw_cell(
        result.data,
        plan,
        result.mechanism,
        scaler=result.scaler,
        g_bounds=result.config.g_bounds,
        horizon=result.config.horizons[0],
        cause=None,
        normalization=result.config.normalization,
        stacked=False,
    )
    assert fit.psi_scaled == result.fits["always"].psi_scaled
    return fit.influence_curve_scaled


@pytest.mark.parametrize("kind", ("binary", "survival"))
def test_w8_g_computation_is_the_shared_untargeted_recursion(kind: str) -> None:
    """W8: the estimate is the first-node mean of the fold recursion run on every row."""
    frame, columns, settings = _random_case(kind, 300, 9)
    result = _random_fit("gcomp", frame, columns, settings)
    data = result.data
    plan = result.replay_recipe.plans[0]
    horizon = result.config.horizons[-1]
    cell = FoldRecursionCell(plan, plan.masks(data), horizon)
    outputs = _untargeted_recursion_in_fold(
        data,
        cell,
        scaler=result.scaler,
        cause=None,
        outcome_learner=TIGHT,
        pseudo_learner=TIGHT,
        outer_train=np.ones(data.n, dtype=bool),
        fold=0,
        test=np.arange(data.n),
    )
    first = outputs[1][0]
    expected = float(np.average(first, weights=data.weights))
    key = "always" if not data.is_survival else f"always @ t={horizon}"
    assert result.fits[key].psi_scaled == expected


# ------------------------------------------------------------------ W10 numpy


def _newton(design: Any, target: Any, weights: Any) -> Any:
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        mean = _expit(design @ beta)
        score = design.T @ (weights * (target - mean))
        hessian = (design * (weights * mean * (1.0 - mean))[:, None]).T @ design
        move = np.linalg.solve(hessian, score)
        beta = beta + move
        if np.max(np.abs(move)) < 1e-14:
            break
    return beta


def _numpy_baselines(frame: pd.DataFrame) -> dict[str, tuple[float, Any]]:
    """ICE, HT and Hajek of the plan (1, 1) with their stacked curves, written in numpy."""
    n = len(frame)
    one = np.ones(n)
    d = {key: np.nan_to_num(frame[key].to_numpy(dtype=float)) for key in frame}
    x = {
        "g1": np.column_stack([one, d["W"]]),
        "c1": np.column_stack([one, d["W"], d["A1"]]),
        "g2": np.column_stack([one, d["W"], d["L2"], d["A1"]]),
        "c2": np.column_stack([one, d["W"], d["L2"], d["A1"], d["A2"]]),
        "q1": np.column_stack([one, d["W"]]),
        "q2": np.column_stack([one, d["W"], d["L2"]]),
    }
    r1 = (d["A1"] == 1) & (d["C1"] == 1)
    r2 = r1 & (d["A2"] == 1) & (d["C2"] == 1)
    unit = np.ones(n)
    b2 = _newton(x["q2"][r2], d["Y"][r2], unit[r2])
    m2 = _expit(x["q2"] @ b2)
    b1 = _newton(x["q1"][r1], m2[r1], unit[r1])
    m1 = _expit(x["q1"] @ b1)
    psi = float(np.mean(m1))
    information2 = (x["q2"] * (r2 * m2 * (1 - m2))[:, None]).T @ x["q2"] / n
    information1 = (x["q1"] * (r1 * m1 * (1 - m1))[:, None]).T @ x["q1"] / n
    a1 = (x["q1"] * (m1 * (1 - m1))[:, None]).sum(0) / n
    link = (x["q1"] * r1[:, None]).T @ (x["q2"] * (m2 * (1 - m2))[:, None]) / n
    a2 = np.linalg.solve(information1.T, a1) @ link
    ice = (
        m1
        - psi
        + (x["q1"] * (r1 * (m2 - m1))[:, None]) @ np.linalg.solve(information1, a1)
        + (x["q2"] * (r2 * (d["Y"] - m2))[:, None]) @ np.linalg.solve(information2, a2)
    )
    rows = {"g1": unit > 0, "c1": unit > 0, "g2": d["C1"] == 1, "c2": d["C1"] == 1}
    target = {"g1": d["A1"], "c1": d["C1"], "g2": d["A2"], "c2": d["C2"]}
    projected, log_derivative, factor = [], [], np.ones(n)
    for key in ("g1", "c1", "g2", "c2"):
        coefficient = _newton(x[key][rows[key]], target[key][rows[key]], unit[rows[key]])
        mean = _expit(x[key] @ coefficient)
        factor = factor * mean
        score = x[key] * (rows[key] * (target[key] - mean))[:, None]
        information = (x[key] * (rows[key] * mean * (1 - mean))[:, None]).T @ x[key] / n
        projected.append(score @ np.linalg.inv(information))
        log_derivative.append(x[key] * (1 - mean)[:, None])
    projection = np.hstack(projected)
    slope = np.hstack(log_derivative)
    numerator = np.where(r2, d["Y"] / factor, 0.0)
    denominator = np.where(r2, 1.0 / factor, 0.0)
    ht = float(np.mean(numerator))
    hajek = float(np.sum(numerator) / np.sum(denominator))
    b_numerator = -(slope * numerator[:, None]).sum(0) / n
    b_denominator = -(slope * denominator[:, None]).sum(0) / n
    ht_curve = numerator - ht + projection @ b_numerator
    hajek_curve = (
        numerator - hajek * denominator + projection @ (b_numerator - hajek * b_denominator)
    ) / np.mean(denominator)
    return {"gcomp": (psi, ice), "horvitz_thompson": (ht, ht_curve), "hajek": (hajek, hajek_curve)}


def test_w10_an_independent_numpy_construction_agrees() -> None:
    """W10: points to 1e-10 and curves to 1e-8 against a construction written here."""
    frame = _draw_binary(500, 17, weighted=False)
    expected = _numpy_baselines(frame)
    columns = {**CENSORED, "baseline": ["W"]}
    for variant in VARIANTS:
        result = _random_fit(variant, frame, columns, {"g_bounds": (1e-8, 1.0)})
        psi, curve = expected[variant]
        assert result.psi("ey_regimen[always]") == pytest.approx(psi, abs=1e-10)
        np.testing.assert_allclose(
            result.influence_curves["ey_regimen[always]"], curve, atol=1e-8, rtol=0
        )


# --------------------------------------------------------- status and refusals


def _small() -> tuple[pd.DataFrame, dict[str, Any]]:
    return _draw_binary(400, 2), {**CENSORED, "baseline": ["W"]}


def test_a_data_adaptive_learner_withholds_the_interval() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeRegressor

    frame, columns = _small()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcomp = SequentialGComputation(
            {"always": 1},
            outcome_learner=DecisionTreeRegressor(max_depth=2, random_state=0),
            simultaneous=False,
        ).fit(frame, **columns)
        ipw = SequentialIPW(
            {"always": 1}, treatment_learner=LogisticRegression(), simultaneous=False
        ).fit(frame, **columns)
    for result in (gcomp, ipw):
        estimate = result.estimates["ey_regimen[always]"]
        assert estimate.inference == "adaptive_baseline_plugin"
        assert np.isfinite(estimate.plugin_std_error)
        with pytest.raises(CapabilityError, match="F35"):
            _ = estimate.std_error


def test_g_computation_reads_no_mechanism_learner_for_its_status() -> None:
    from sklearn.linear_model import LogisticRegression

    frame, columns = _small()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SequentialGComputation(
            {"always": 1}, treatment_learner=LogisticRegression(), simultaneous=False
        ).fit(frame, **columns)
    assert result.inference_status == "influence_curve"


def test_the_status_and_every_cell_curve_agree() -> None:
    """A data-adaptive pseudo learner makes every cell diagnostic, the horizon-1 cell too.

    The horizon-1 cell reads only the GLM outcome regression, so its stacked curve exists.
    The fit takes ``adaptive_baseline_plugin``, so the cell must report the diagnostic curve.
    """
    from sklearn.tree import DecisionTreeRegressor

    frame, columns, _ = _random_case("survival", 400, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = SequentialGComputation(
            {"always": 1},
            outcome_learner=GLM(),
            pseudo_learner=DecisionTreeRegressor(max_depth=2, random_state=0),
            simultaneous=False,
        ).fit(frame, **columns)
        stacked = SequentialGComputation(
            {"always": 1}, outcome_learner=GLM(), pseudo_learner=GLM(), simultaneous=False
        ).fit(frame, **columns)
    name = "risk_regimen[always @ t=1]"
    assert {e.inference for e in mixed.estimates.values()} == {"adaptive_baseline_plugin"}
    assert mixed.psi(name) == pytest.approx(stacked.psi(name), abs=1e-12)
    difference = np.max(np.abs(mixed.influence_curves[name] - stacked.influence_curves[name]))
    assert difference > 1e-3


def test_estimated_weights_take_the_estimated_weight_status() -> None:
    frame, columns = _small()
    frame["wt"] = np.random.default_rng(0).uniform(0.5, 2.0, len(frame))
    for variant in VARIANTS:
        result = _random_fit(
            variant,
            frame,
            {**columns, "weights": "wt", "weights_estimated": True},
            {},
        )
        assert result.inference_status == "estimated_weight_plugin"


def test_an_ipw_node_with_no_event_takes_the_constant_node_status() -> None:
    frame, outcome = _draw_survival(400, 4)
    # Drop the treated units who fail at node 1, so the "always" arm has no event there.
    frame = frame[~((frame["A1"] == 1) & (frame["Y1"] == 1))].reset_index(drop=True)
    columns = {**CENSORED, "outcome": outcome}
    result = _random_fit("hajek", frame, columns, {})
    assert result.estimates["risk_regimen[always @ t=1]"].inference == "constant_node_plugin"
    assert result.estimates["risk_regimen[always @ t=2]"].inference == "constant_node_plugin"
    assert result.estimates["risk_regimen[never @ t=1]"].inference == "influence_curve"


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (lambda: SequentialGComputation({"a": 1}, n_folds=2), CapabilityError, "fits in sample"),
        (lambda: SequentialIPW({"a": 1}, n_folds=5), CapabilityError, "fits in sample"),
        (
            lambda: SequentialGComputation({"a": 1}, alpha=0.99),
            MethodConfigurationError,
            "fit no fluctuation",
        ),
        (
            lambda: SequentialIPW({"a": 1}, max_iter=3),
            MethodConfigurationError,
            "fit no fluctuation",
        ),
        (
            lambda: SequentialIPW({"a": 1}, outcome_learner=GLM()),
            CapabilityError,
            "fits no outcome regression",
        ),
        (
            lambda: SequentialIPW({"a": 1}, pseudo_learner=GLM()),
            CapabilityError,
            "fits no outcome regression",
        ),
        (lambda: SequentialIPW({"a": 1}, density_bins=5), CapabilityError, "no density ratio"),
        (lambda: SequentialIPW({"a": 1}, ratio="classifier"), CapabilityError, "no density ratio"),
        (lambda: SequentialIPW({"a": 1}, normalization="n"), ValueError, "normalization"),
    ],
)
def test_each_construction_refusal(make: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        make()


def test_a_working_model_is_refused() -> None:
    from tests.unit.test_longitudinal_msm import DOSE

    with pytest.raises(CapabilityError, match=r"LTMLE\(msm=\.\.\.\)"):
        SequentialGComputation({"a": 1}, msm=DOSE)


@pytest.mark.parametrize("engine", [SequentialGComputation, SequentialIPW, ParametricGFormula])
def test_a_ratio_trim_and_pooled_hazards_are_refused(engine: Any) -> None:
    from cleverly.methods import BASELINE_HAZARD_POOLING_REFUSAL, BASELINE_RATIO_TRIM_REFUSAL

    with pytest.raises(CapabilityError) as trim:
        engine({"a": 1}, ratio_trim=0.9)
    assert trim.value.refusal == BASELINE_RATIO_TRIM_REFUSAL
    with pytest.raises(CapabilityError) as pooled:
        engine({"a": 1}, hazard_pooling="pooled")
    assert pooled.value.refusal == BASELINE_HAZARD_POOLING_REFUSAL
    # The defaults construct.
    engine({"a": 1}, ratio_trim=None, hazard_pooling="node")


@pytest.mark.parametrize("method", [GComputationMethod, IPWMethod, ParametricGFormulaMethod])
def test_the_methods_refuse_a_ratio_trim_and_pooled_hazards(method: Any) -> None:
    with pytest.raises(MethodConfigurationError, match="Omit ratio_trim="):
        method(targeting=Targeting(ratio_trim=0.9)).estimator_kwargs(longitudinal=True)
    with pytest.raises(MethodConfigurationError, match="hazard_pooling='node'"):
        method(models=ModelSpec(hazard_pooling="pooled")).estimator_kwargs(longitudinal=True)


def test_a_binomial_glm_on_an_unbounded_scale_is_refused() -> None:
    frame, columns, _ = _random_case("gaussian", 300, 1)
    with pytest.raises(CapabilityError, match="Declare q_bounds="):
        _fit("gcomp", {"always": 1}, frame, {**columns}, learner=GLM())
    # The Gaussian family and a declared scale both fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SequentialGComputation({"always": 1}, outcome_learner=GAUSSIAN, simultaneous=False).fit(
            frame, **columns
        )
    _fit("gcomp", {"always": 1}, frame, columns, learner=GLM(), q_bounds=G_BOUNDS)


def test_horvitz_thompson_reads_the_outcome_scale() -> None:
    """The Horvitz-Thompson sum does not move with the declared scale.

    The scaled sum maps back to :math:`P_n[w \\lambda R Y] - a (P_n[w \\lambda R] - 1)` unless
    the fit restores the offset.  Two scales, and a non-follower's outcome, would then move it.
    """
    frame, columns, settings = _random_case("gaussian", 400, 4)
    name = "ey_regimen[always]"
    base = _random_fit("horvitz_thompson", frame, columns, settings)
    wider = _random_fit("horvitz_thompson", frame, columns, {"q_bounds": (-6.0, 9.0)})
    assert wider.psi(name) == pytest.approx(base.psi(name), abs=1e-10)
    np.testing.assert_allclose(
        wider.influence_curves[name], base.influence_curves[name], rtol=1e-8, atol=1e-10
    )
    observed = frame["Y"].notna().to_numpy()
    leaver = np.flatnonzero(observed & (frame["A2"].to_numpy() == 0.0))[0]
    moved = frame.copy()
    moved.loc[moved.index[leaver], "Y"] = 3.5
    assert _random_fit("horvitz_thompson", moved, columns, settings).psi(name) == pytest.approx(
        base.psi(name), abs=1e-10
    )
    # The offset is nonzero here: a scaled sum without it would differ by this much.
    mass = base.fits["always"].steps[-1].clever
    weights = base.data.weights
    assert abs(G_BOUNDS[0] * (float(np.mean(weights * mass)) - 1.0)) > 1e-3


def test_an_estimated_tilt_is_refused() -> None:
    from cleverly.interventions import Incremental

    frame, columns = _small()
    for variant in ("gcomp", "hajek"):
        with pytest.raises(CapabilityError, match="Fit IPSI"):
            _fit(variant, {"tilt": DynamicRegimen("tilt", (Incremental(2.0), 1))}, frame, columns)


def test_a_truncation_curve_refuses_g_computation_and_replays_ipw() -> None:
    frame, columns = _small()
    gcomp = _random_fit("gcomp", frame, columns, {})
    with pytest.raises(CapabilityError, match="no truncation curve exists"):
        longitudinal_truncation_curve(gcomp, [0.01, 0.05])
    ipw = _random_fit("hajek", frame, columns, {})
    curve = longitudinal_truncation_curve(ipw, [0.01, 0.2])
    fitted = curve[curve["is_fitted_bound"]]
    assert np.allclose(fitted["delta_from_fitted"], 0.0)


def test_the_replay_restamps_the_adaptive_status() -> None:
    from sklearn.linear_model import LogisticRegression

    frame, columns = _small()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SequentialIPW(
            {"always": 1}, treatment_learner=LogisticRegression(), simultaneous=False
        ).fit(frame, **columns)
        curve = longitudinal_truncation_curve(result, [0.01])
        replay = _refit_bound(result, result.repeats[0], result.config.g_bounds)
    assert result.replay_recipe.parametric is False
    assert curve["delta_from_fitted"].iloc[0] == 0.0
    name = "ey_regimen[always]"
    assert replay.estimates[name].inference == "adaptive_baseline_plugin"
    # The replay keeps the fixed-mechanism curve; a rebuilt stacked curve would differ.
    np.testing.assert_array_equal(
        replay.fits["always"].influence_curve_scaled,
        result.fits["always"].influence_curve_scaled,
    )


def test_a_diagnostic_mechanism_that_cannot_fit_leaves_the_estimate() -> None:
    """m5: a separated treatment GLM warns, and g-computation still reports its estimate.

    ``A1 = 1`` exactly when ``W > 0``.  The followers of "always" still vary in ``W``, so the
    outcome GLM keeps its coefficient and extrapolates it: the declared model reliance.
    """
    frame, columns = _small()
    frame["A1"] = (frame["W"] > 0.0).astype(float)
    with pytest.warns(Warning, match="did not fit"):
        result = SequentialGComputation(
            {"always": 1}, outcome_learner=GLM(), simultaneous=False
        ).fit(frame, **columns)
    assert np.isfinite(result.psi("ey_regimen[always]"))
    assert result.inference_status == "influence_curve"


def test_a_prediction_through_an_aliased_column_is_refused() -> None:
    """B3: ``W`` is constant on the followers, so their fit cannot say what ``W`` does.

    ``W = 1`` exactly when ``A1 = 1``.  No unit with ``W = -1`` follows "always", and the GLM
    aliases ``W``.  Predicting the ``W = -1`` half at the intercept would assume ``W`` has no
    effect, so the fit refuses.
    """
    frame, columns = _small()
    frame["W"] = np.where(frame["A1"] == 1, 1.0, -1.0)
    with (
        warnings.catch_warnings(),
        pytest.raises(DataError, match="not identified by sequential g-computation"),
    ):
        warnings.simplefilter("ignore")
        SequentialGComputation({"always": 1}, outcome_learner=GLM(), simultaneous=False).fit(
            frame, **columns
        )


def test_a_mechanism_factor_off_its_fit_rows_is_refused() -> None:
    """S3: the factor rebuilds the fit rows, so a drift from the fit must not pass silently."""
    from cleverly.longitudinal.baselines import _factor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    label = rng.binomial(1, _expit(x[:, 0])).astype(float)
    weights = np.ones(300)
    model = GLM(tol=1e-14).fit(x, label, sample_weight=weights)
    _factor(1, model, x, label, weights, weights, 300)
    drifted = np.where(np.arange(300) < 20, 0.0, weights)
    with pytest.raises(RuntimeError, match="does not solve its score equation"):
        _factor(1, model, x, label, drifted, weights, 300)


def test_save_and_load_round_trip(tmp_path: Any) -> None:
    import cleverly

    frame, columns = _small()
    result = _random_fit("hajek", frame, columns, {})
    path = tmp_path / "ipw.cleverly"
    result.save(path)
    loaded = cleverly.load(path)
    assert loaded.psi("ey_regimen[always]") == result.psi("ey_regimen[always]")
    assert loaded.config.normalization == "hajek"


def test_the_summary_names_the_estimator() -> None:
    frame, columns = _small()
    assert "sequential g-computation" in _random_fit("gcomp", frame, columns, {}).summary()
    assert "Horvitz-Thompson" in _random_fit("horvitz_thompson", frame, columns, {}).summary()


def test_the_fit_reads_its_scale_from_a_declared_container() -> None:
    frame, columns = _small()
    data = LongitudinalData.from_frame(frame, **columns)
    for variant in VARIANTS:
        assert _fit(variant, {"always": 1}, data, {}).psi("ey_regimen[always]") == pytest.approx(
            _fit(variant, {"always": 1}, frame, columns).psi("ey_regimen[always]"), abs=0
        )


def test_ltmle_and_the_baselines_share_a_target() -> None:
    """A smoke check: on a large draw the three estimates agree within their spread."""
    frame, columns = _draw_binary(4000, 31, weighted=False), {**CENSORED, "baseline": ["W"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = LTMLE(
            {"always": 1},
            n_folds=1,
            outcome_learner=GLM(),
            treatment_learner=GLM(),
            censoring_learner=GLM(),
            simultaneous=False,
        ).fit(frame, **columns)
    for variant in VARIANTS:
        result = _random_fit(variant, frame, columns, {})
        estimate = result.estimates["ey_regimen[always]"]
        assert abs(estimate.psi - reference.psi("ey_regimen[always]")) < 3 * estimate.std_error
