"""The parametric g-formula (X45): exact-law witnesses, refusals, statuses and the surface.

The estimator simulates each unit's history forward from its observed baseline under a regimen
and averages the simulated outcomes.  Its curve is the two-step curve of the simulated
estimating equation, so every piece has a closed form on a law the test can enumerate.  The
witnesses below enumerate that law with complex-step arithmetic, and so they are independent of
the simulation code they check.

* W1, W2: the estimate equals the enumerated functional at the fitted parameters.
* W3a: with the exact terms in place of the simulation, the curve equals the Gateaux derivative
  of the enumerated functional in each observation weight.
* W3b: the production derivative of the simulation equals the exact one, per draw kind.
* W4, W5, W6: the Monte Carlo variance term, the mean of the curve, common random numbers.
* W9: on a saturated law the estimate equals the nonparametric plug-in of
  ``SequentialGComputation``, and the statistical curve equals the efficient influence curve.
* W10: a fixed dose on a linear-Gaussian law has a closed form.
* W11: the refusals before any learner.

``tests/unit/test_parametric_gformula.py`` also carries the Philox known-answer test, the status
table and the surface checks.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import cleverly
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    MonteCarloWarning,
    RefusalReason,
    WeightingWarning,
)
from cleverly.inference.cluster import influence_variance
from cleverly.interventions import Incremental, Shift
from cleverly.learners import GLM
from cleverly.longitudinal import (
    DynamicRegimen,
    ParametricGFormula,
    Regimen,
    Repeated,
    SequentialGComputation,
)
from cleverly.longitudinal.estimator import longitudinal_truncation_curve

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import discrete_law_longitudinal_policy as policy
from .. import discrete_law_survival as survival
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import declared_regimens

#: The module, not the one-call function of the same name that ``cleverly.longitudinal`` exports.
gf = importlib.import_module("cleverly.longitudinal.gformula")
# ----------------------------------------------------------------------------- the machinery


def expit(z: Any) -> Any:
    """The logistic function, valid for complex arguments."""
    return 1.0 / (1.0 + np.exp(-z))


STEP = 1e-30


def theta_of(book: gf.ModelBook) -> dict[str, np.ndarray]:
    """The parameter vector of each stacked model: coefficients, then a Gaussian variance."""
    out = {}
    for model in book.stacked:
        vector = np.asarray(model.learner.coef_, dtype=float).reshape(-1)
        if model.response == "normal":
            vector = np.append(vector, model.sigma2)
        out[model.key] = vector
    return out


def design_of(model: gf.FittedModel, x: np.ndarray) -> np.ndarray:
    """The active GLM design of ``x``, keeping a complex part that the library's would drop."""
    glm = model.learner
    if glm.interactions == 1:
        columns = [np.ones(x.shape[0]), *np.asarray(x).T]
        return np.column_stack(columns)[:, glm.active_]
    return glm.design(np.real(x))


def linear(book: gf.ModelBook, key: str, vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The linear predictor of a binomial or Gaussian model at parameter ``vector``."""
    model = next(m for m in book.models if m.key == key)
    design = design_of(model, x)
    return design @ vector[: design.shape[1]]


def logits(book: gf.ModelBook, key: str, vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The class logits ``(N, K)`` of a multinomial model, reference class first."""
    model = next(m for m in book.models if m.key == key)
    design = design_of(model, x)
    k = len(model.present) - 1
    coef = vector.reshape(k, design.shape[1])
    return np.column_stack([np.zeros(design.shape[0], dtype=vector.dtype), design @ coef.T])


def hook_from(
    phi: Callable[[gf.ModelBook, Any, dict[str, np.ndarray]], np.ndarray],
    store: dict[str, Any] | None = None,
) -> Callable[..., gf.ExactTerms]:
    """Exact terms from a complex-step enumerator ``phi(book, plan, theta) -> (n, C)``.

    The derivative of each stacked model is the complex-step derivative of
    ``(1/n) sum_i w_i Phi_i``, which is what the simulation estimates.
    """

    def hook(book: gf.ModelBook, plan: Any, cells: Any) -> gf.ExactTerms:
        if store is not None:
            store.update(book=book, plan=plan, cells=cells)
            store.setdefault("plans", []).append(plan)
        theta = theta_of(book)
        base = np.real(phi(book, plan, {k: v.astype(complex) for k, v in theta.items()}))
        weights = np.asarray(book.data.weights, dtype=float)
        n = book.data.n
        derivatives = {}
        for key, vector in theta.items():
            block = np.zeros((base.shape[1], vector.size))
            for position in range(vector.size):
                moved = {k: v.astype(complex) for k, v in theta.items()}
                moved[key][position] += 1j * STEP
                value = phi(book, plan, moved)
                block[:, position] = (weights[:, None] * value.imag / STEP).sum(axis=0) / n
            derivatives[key] = block
        return gf.ExactTerms(base, derivatives)

    return hook


def fit_with(
    frame: pd.DataFrame,
    regimens: Any,
    roles: Mapping[str, Any],
    *,
    hook: Callable[..., gf.ExactTerms] | None = None,
    **settings: Any,
) -> Any:
    """Fit :class:`ParametricGFormula`, optionally with exact terms in place of the draws."""
    settings.setdefault("random_state", 1)
    settings.setdefault("simultaneous", False)
    estimator = ParametricGFormula(regimens, **settings)
    estimator._exact = hook
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MonteCarloWarning)
        # The exact laws are weighted by cell counts, which the weight check reads as counts.
        warnings.simplefilter("ignore", WeightingWarning)
        return estimator.fit(frame, **roles)


def monte_carlo_se(result: Any, name: str) -> float:
    """The standard error of the simulation part of ``name``: ``se * sqrt(share)``."""
    share = result.simulation.monte_carlo_share[name]
    return float(result.estimates[name].std_error * np.sqrt(share))


# --------------------------------------------------------------------------- law B: binary L2
#
#   (W1, W2) -> A1 -> C1 -> L2 -> A2 -> C2 -> Y         every node but W1 binary

ROLES_B: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def panel_b(n: int, seed: int, *, weighted: bool = False, clustered: bool = False) -> pd.DataFrame:
    """A two-node panel with a binary time-varying covariate, censoring and an interaction."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a1 = rng.binomial(1, expit(0.4 * w1 - 0.3 * w2)).astype(float)
    c1 = rng.binomial(1, expit(1.5 - 0.4 * a1 + 0.3 * w1)).astype(float)
    l2 = rng.binomial(1, expit(-0.3 + 0.9 * a1 + 0.5 * w1)).astype(float)
    a2 = rng.binomial(1, expit(0.2 - 0.8 * l2 + 0.5 * a1 + 0.3 * w2)).astype(float)
    c2 = rng.binomial(1, expit(1.2 + 0.3 * l2 - 0.3 * a2)).astype(float)
    y = rng.binomial(
        1, expit(-0.4 + 0.7 * a1 + 0.9 * l2 + 0.6 * a2 + 0.5 * w1 - 0.8 * a1 * l2)
    ).astype(float)
    frame = pd.DataFrame(
        {"W1": w1, "W2": w2, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y}
    )
    frame.loc[c1 == 0, ["L2", "A2", "C2", "Y"]] = np.nan
    frame.loc[(c1 == 1) & (c2 == 0), ["Y"]] = np.nan
    if weighted:
        frame["wt"] = 0.5 + rng.uniform(size=n) * (1.0 + 0.8 * w2)
    if clustered:
        frame["cl"] = np.arange(n) // 4
    return frame


def baseline_history(data: Any) -> dict[str, np.ndarray]:
    """The baseline columns by name, the start of every history a rule reads."""
    return {name: data.baseline[:, j] for j, name in enumerate(data.baseline_names)}


def design_b(data: Any, a1: np.ndarray, l2: Any = None, a2: Any = None) -> dict[str, np.ndarray]:
    """The design a model of law B reads: baseline, ``A1``, then ``L2`` and ``A2``."""
    base = np.column_stack([data.baseline, a1])
    out = {"covariate:2:L2": base}
    if l2 is not None:
        out["outcome"] = np.column_stack([base, l2, a2])
    return out


def assigned(plan: Any, data: Any, node: int, history: dict[str, Any]) -> np.ndarray:
    """The treatment a static or rule plan assigns at ``node`` on a simulated history."""
    kind, spec = gf.plan_node(plan.regimen, node)
    if kind == "label":
        return np.full(data.n, float(spec))
    return np.asarray(spec(pd.DataFrame({k: np.real(v) for k, v in history.items()})), dtype=float)


def phi_b(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law B's g-formula by enumeration over ``L2``: ``(n, 1)`` unit means, complex-step ready."""
    data = book.data
    start = baseline_history(data)
    a1 = assigned(plan, data, 1, start)
    x2 = design_b(data, a1)["covariate:2:L2"]
    p_l2 = expit(linear(book, "covariate:2:L2", theta["covariate:2:L2"], x2))
    total = np.zeros(data.n, dtype=complex)
    for value, prob in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
        history = {**start, "A1": a1, "L2": np.full(data.n, value)}
        a2 = assigned(plan, data, 2, history)
        x = design_b(data, a1, history["L2"], a2)["outcome"]
        total = total + prob * expit(linear(book, "outcome", theta["outcome"], x))
    return total[:, None]


#: The plans of law B: the regimen the estimator is handed, and the same plan as ``(a1, a2(l2))``.
PLANS_B: dict[str, tuple[Any, tuple[float, Callable[[np.ndarray], np.ndarray]]]] = {
    "never": (0, (0.0, lambda l2: np.zeros_like(l2))),
    "always": (1, (1.0, lambda l2: np.ones_like(l2))),
    "respond": (
        DynamicRegimen("respond", (1, lambda h: 1.0 - h["L2"]), rule_kind="known"),
        (1.0, lambda l2: 1.0 - l2),
    ),
}


def regimens_b(*labels: str) -> list[Any]:
    """The regimens of ``labels`` as a sequence the estimator takes."""
    return [
        spec if isinstance(spec, DynamicRegimen) else Regimen(label, (spec, spec))
        for label in labels
        for spec in [PLANS_B[label][0]]
    ]


class TestExactTermsLawB:
    """W3a on law B: the curve is the Gateaux derivative of the enumerated functional."""

    N = 120

    @staticmethod
    def _refit_functional(frame: pd.DataFrame, weights: np.ndarray, label: str) -> float:
        """psi(w): refit both GLMs at the weights and enumerate, in plain numpy."""
        w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
        a1o = frame["A1"].to_numpy()
        c1, c2 = frame["C1"].to_numpy() == 1, frame["C2"].to_numpy() == 1
        l2o, a2o, y = (frame[c].to_numpy() for c in ("L2", "A2", "Y"))
        glm2 = GLM(tol=1e-15, max_iter=200).fit(
            np.column_stack([w1, w2, a1o])[c1], l2o[c1].astype(int), weights[c1]
        )
        glmy = GLM(tol=1e-15, max_iter=200).fit(
            np.column_stack([w1, w2, a1o, l2o, a2o])[c2], y[c2].astype(int), weights[c2]
        )
        first, second = PLANS_B[label][1]
        a1 = np.full(len(frame), first)
        p_l2 = glm2.predict(np.column_stack([w1, w2, a1]))
        total = np.zeros(len(frame))
        for value, prob in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
            l2 = np.full(len(frame), value)
            total += prob * glmy.predict(np.column_stack([w1, w2, a1, l2, second(l2)]))
        return float(np.sum(weights * total) / np.sum(weights))

    @pytest.mark.parametrize("label", ["never", "always", "respond"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_the_curve_is_the_gateaux_derivative(self, label: str, weighted: bool) -> None:
        frame = panel_b(self.N, seed=3, weighted=weighted)
        roles = dict(ROLES_B, **({"weights": "wt"} if weighted else {}))
        result = fit_with(frame, regimens_b(label), roles, hook=hook_from(phi_b), n_draws=2)
        name = f"ey_regimen[{label}]"
        weights = frame["wt"].to_numpy() if weighted else np.ones(len(frame))
        step = 1e-4
        expected = np.zeros(len(frame))
        for i in range(len(frame)):
            up, down = weights.copy(), weights.copy()
            up[i] += step
            down[i] -= step
            slope = (
                self._refit_functional(frame, up, label)
                - self._refit_functional(frame, down, label)
            ) / (2 * step)
            expected[i] = len(frame) * weights[i] * slope
        np.testing.assert_allclose(result.influence_curves[name], expected, rtol=0, atol=1e-6)
        assert np.max(np.abs(expected)) > 0.1


# ------------------------------------------------------------ generic exact-law checks


def refit_theta(book: gf.ModelBook, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Each stacked model refitted at ``weights`` on its own rows, design and target.

    The rows, designs and targets come from the book, so what the Gateaux check isolates is the
    weight convention, the scores, the information and the stacking of the curve.
    """
    out = {}
    for model in book.stacked:
        glm = GLM(
            family=model.learner.family_,
            interactions=model.learner.interactions,
            tol=1e-15,
            max_iter=200,
        )
        rows = model.rows
        x, y, w = book.design(model)[rows], model.target[rows], weights[rows]
        glm.fit(x, y, w)
        vector = np.asarray(glm.coef_, dtype=float).reshape(-1)
        if model.response == "normal":
            residual = y - glm.predict(x)
            vector = np.append(vector, np.sum(w * residual**2) / np.sum(w))
        out[model.key] = vector
    return out


def gateaux_curve(
    book: gf.ModelBook,
    phi: Callable[[gf.ModelBook, Any, dict[str, np.ndarray]], np.ndarray],
    plan: Any,
    *,
    cell: int = 0,
    step: float = 1e-4,
    units: np.ndarray | None = None,
) -> np.ndarray:
    """``n w_i dpsi/dw_i``: the curve of ``psi(w) = sum w Phi(theta(w)) / sum w`` at ``units``."""
    weights = np.asarray(book.data.weights, dtype=float)
    n = book.data.n

    def psi(w: np.ndarray) -> float:
        values = np.real(phi(book, plan, refit_theta(book, w)))[:, cell]
        return float(np.sum(w * values) / np.sum(w))

    chosen = np.arange(n) if units is None else np.asarray(units)
    curve = np.zeros(chosen.size)
    for position, i in enumerate(chosen):
        up, down = weights.copy(), weights.copy()
        up[i] += step * weights[i]
        down[i] -= step * weights[i]
        curve[position] = n * weights[i] * (psi(up) - psi(down)) / (2.0 * step * weights[i])
    return curve


def seeded_settings(
    result: Any, *, n_draws: int, n_derivative_draws: int, seed: int
) -> gf.GFormulaSettings:
    """Simulation settings that read the learners of a finished fit, at a chosen seed."""
    return gf.GFormulaSettings(
        n_draws=n_draws,
        n_derivative_draws=n_derivative_draws,
        covariate_types=None,
        covariate_learner=None,
        outcome_learner=None,
        hazard_learner=None,
        treatment_learner=None,
        seed=seed,
    )


def replicate_streams(
    store: dict[str, Any],
    *,
    replicates: int,
    n_draws: int,
    n_derivative_draws: int,
    first_seed: int = 10_000,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Production unit means and derivatives at ``replicates`` independent seeds.

    Returns the ``(R, n, C)`` unit means of the estimate stream and, per model, the
    ``(R, C, P)`` derivatives of the derivative stream.
    """
    phis, derivatives = [], {}
    for r in range(replicates):
        settings = seeded_settings(
            None, n_draws=n_draws, n_derivative_draws=n_derivative_draws, seed=first_seed + r
        )
        out = gf._simulate_plan(store["book"], store["plan"], store["cells"], settings)
        phis.append(out.phi)
        for key, value in out.derivatives.items():
            derivatives.setdefault(key, []).append(value)
    return np.array(phis), {key: np.array(value) for key, value in derivatives.items()}


def assert_within_standard_errors(
    replicates: np.ndarray, exact: np.ndarray, *, z: float = 4.0, label: str = ""
) -> None:
    """The replicate mean equals ``exact`` within ``z`` standard errors of that mean."""
    mean = replicates.mean(axis=0)
    error = replicates.std(axis=0, ddof=1) / np.sqrt(replicates.shape[0])
    miss = np.abs(mean - exact) - z * error
    assert np.all(miss <= 1e-10), (
        f"{label}: worst miss {np.max(miss):.3g} at "
        f"{np.unravel_index(np.argmax(miss), miss.shape)}; exact {exact.ravel()[:4]}, "
        f"mean {mean.ravel()[:4]}"
    )


def run_exact(
    frame: pd.DataFrame,
    regimens: Any,
    roles: Mapping[str, Any],
    phi: Callable[..., np.ndarray],
    **settings: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit with the exact terms of ``phi``, keeping the book, plan and cells the hook saw."""
    store: dict[str, Any] = {}
    result = fit_with(frame, regimens, roles, hook=hook_from(phi, store), **settings)
    return result, store


# ------------------------------------------------------------------ law R: rich covariates
#
#   W1 -> A1 -> L2 (normal) -> A2 -> L3 (three categories, non-monotone effect) -> A3 -> Y

ROLES_R: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W1"],
    "time_varying": [[], ["L2"], ["L3"]],
    "censoring": ["C1", "C2", "C3"],
}
TYPES_R = {"L2": "normal", "L3": "categorical"}


def panel_r(n: int, seed: int, *, weighted: bool = False) -> pd.DataFrame:
    """Law R: a Gaussian covariate that drives a categorical one that drives the outcome."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    a1 = rng.binomial(1, expit(0.4 * w1)).astype(float)
    c1 = rng.binomial(1, expit(1.6 - 0.3 * a1)).astype(float)
    l2 = 0.3 + 0.6 * w1 + 0.7 * a1 + rng.normal(scale=0.8, size=n)
    a2 = rng.binomial(1, expit(0.3 * l2 - 0.2 * a1)).astype(float)
    c2 = rng.binomial(1, expit(1.5 - 0.2 * a2)).astype(float)
    eta = np.column_stack([np.zeros(n), -0.2 + 1.2 * l2 + 0.4 * a2, 0.1 - 1.0 * l2 + 0.5 * a1])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)
    l3 = (u[:, None] > np.cumsum(prob, axis=1)).sum(axis=1).astype(float)
    a3 = rng.binomial(1, expit(0.2 + 0.5 * (l3 == 2) - 0.3 * a2)).astype(float)
    c3 = rng.binomial(1, expit(1.4 - 0.2 * a3)).astype(float)
    y = rng.binomial(
        1,
        expit(-0.5 + 0.6 * a1 + 0.4 * l2 + 0.5 * a2 + 1.2 * (l3 == 1) - 0.4 * (l3 == 2) + 0.5 * a3),
    ).astype(float)
    frame = pd.DataFrame(
        {
            "W1": w1,
            "A1": a1,
            "C1": c1,
            "L2": l2,
            "A2": a2,
            "C2": c2,
            "L3": l3,
            "A3": a3,
            "C3": c3,
            "Y": y,
        }
    )
    alive = np.ones(n, dtype=bool)
    for node, later in ((1, ["L2", "A2", "C2"]), (2, ["L3", "A3", "C3"]), (3, ["Y"])):
        dropped = alive & (frame[f"C{node}"] == 0)
        frame.loc[dropped, later] = np.nan
        alive &= frame[f"C{node}"] == 1
        if node < 3:
            frame.loc[~alive, later] = np.nan
    frame.loc[~alive, ["Y"]] = np.nan
    if weighted:
        frame["wt"] = 0.6 + rng.uniform(size=n) * (1.0 + 0.5 * (a1 == 1))
    return frame


PLANS_R: dict[str, Any] = {
    "static": Regimen("static", (1, 0, 1)),
    "never": Regimen("never", (0, 0, 0)),
    "dynamic": DynamicRegimen(
        "dynamic", (1, 0, lambda h: (h["L3"] == 2).astype(float)), rule_kind="known"
    ),
}


def phi_r(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law R by Gauss-Hermite over ``L2`` and enumeration over the categories of ``L3``."""
    data = book.data
    nodes, weights = np.polynomial.hermite_e.hermegauss(30)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = data.baseline[:, 0]
    a1 = assigned(plan, data, 1, {"W1": w1})
    th2 = theta["covariate:2:L2"]
    mean2 = linear(book, "covariate:2:L2", th2, np.column_stack([w1, a1]))
    scale2 = np.sqrt(th2[-1])
    total = np.zeros(data.n, dtype=complex)
    for z, weight in zip(nodes, weights, strict=True):
        l2 = mean2 + scale2 * z
        history = {"W1": w1, "A1": a1, "L2": np.real(l2)}
        a2 = assigned(plan, data, 2, history)
        x3 = np.column_stack([w1, a1, l2, a2])
        p3 = np.exp(logits(book, "covariate:3:L3", theta["covariate:3:L3"], x3))
        p3 = p3 / p3.sum(axis=1, keepdims=True)
        for code in range(3):
            a3 = assigned(plan, data, 3, {**history, "A2": a2, "L3": np.full(data.n, float(code))})
            x = np.column_stack(
                [
                    w1,
                    a1,
                    l2,
                    a2,
                    np.full(data.n, float(code == 1)),
                    np.full(data.n, float(code == 2)),
                    a3,
                ]
            )
            total = total + weight * p3[:, code] * expit(
                linear(book, "outcome", theta["outcome"], x)
            )
    return total[:, None]


class TestGenericGateauxCurves:
    """W3a on the rich law: Gaussian mean and variance, multinomial, outcome, dynamic rule."""

    @pytest.mark.parametrize("label", ["static", "dynamic"])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_law_r(self, label: str, weighted: bool) -> None:
        frame = panel_r(100, seed=5, weighted=weighted)
        roles = dict(ROLES_R, **({"weights": "wt"} if weighted else {}))
        result, store = run_exact(
            frame,
            [PLANS_R[label]],
            roles,
            phi_r,
            covariate_types=TYPES_R,
            n_draws=2,
        )
        units = np.random.default_rng(0).choice(len(frame), 20, replace=False)
        expected = gateaux_curve(store["book"], phi_r, store["plan"], units=units)
        np.testing.assert_allclose(
            result.influence_curves[f"ey_regimen[{label}]"][units], expected, rtol=0, atol=1e-6
        )
        assert np.max(np.abs(expected)) > 0.1


def phi_b_moments(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law B's ``(n, 2)`` first and second moments of the simulated outcome mean ``f``."""
    data = book.data
    start = baseline_history(data)
    a1 = assigned(plan, data, 1, start)
    x2 = design_b(data, a1)["covariate:2:L2"]
    p_l2 = expit(linear(book, "covariate:2:L2", theta["covariate:2:L2"], x2))
    first = np.zeros(data.n, dtype=complex)
    second = np.zeros(data.n, dtype=complex)
    for value, prob in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
        history = {**start, "A1": a1, "L2": np.full(data.n, value)}
        a2 = assigned(plan, data, 2, history)
        x = design_b(data, a1, history["L2"], a2)["outcome"]
        mean = expit(linear(book, "outcome", theta["outcome"], x))
        first = first + prob * mean
        second = second + prob * mean**2
    return np.column_stack([first, second])


def simulation_variance_of_units(
    book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]
) -> np.ndarray:
    """The exact variance of ``f`` given each unit's observed start, on law B."""
    moments = np.real(phi_b_moments(book, plan, {k: v.astype(complex) for k, v in theta.items()}))
    return moments[:, 1] - moments[:, 0] ** 2


class TestExactTermsClusterLawB:
    """W3a, cluster variant: the standard error sums the curve within a cluster."""

    def test_cluster_robust_standard_error(self) -> None:
        frame = panel_b(120, seed=3, clustered=True)
        roles = dict(ROLES_B, id="cl")
        result, store = run_exact(frame, regimens_b("always"), roles, phi_b, n_draws=2)
        curve = gateaux_curve(store["book"], phi_b, store["plan"])
        name = "ey_regimen[always]"
        np.testing.assert_allclose(result.influence_curves[name], curve, rtol=0, atol=1e-6)
        expected = np.sqrt(influence_variance(curve, result.data.cluster))
        unit = np.sqrt(influence_variance(curve, None))
        assert result.estimates[name].std_error == pytest.approx(expected, rel=1e-6)
        # The nonzero witness: clustering moves the standard error.
        assert abs(expected - unit) > 0.02 * unit


class TestProductionLawB:
    """W2, W3b, W4, W5, W6 on law B."""

    N = 150

    @pytest.fixture(scope="class")
    def fitted(self) -> tuple[Any, dict[str, Any]]:
        frame = panel_b(self.N, seed=11)
        return run_exact(
            frame,
            regimens_b("never", "always", "respond"),
            ROLES_B,
            phi_b,
            n_draws=2,
        )

    @pytest.mark.parametrize("m_d", [2, 10])
    def test_w3b_the_production_derivatives_are_the_exact_ones(
        self, fitted: tuple[Any, dict[str, Any]], m_d: int
    ) -> None:
        """The binary draw and the outcome mean, at the smallest and a usual stream size."""
        _, store = fitted
        book = store["book"]
        for plan in store["plans"][-3:]:
            store["plan"] = plan
            exact = hook_from(phi_b)(book, plan, store["cells"])
            _, reps = replicate_streams(
                store, replicates=250 if m_d == 2 else 80, n_draws=2, n_derivative_draws=m_d
            )
            assert set(reps) >= set(exact.derivatives)
            for key, block in exact.derivatives.items():
                assert np.max(np.abs(block)) > 0.01, key
                assert_within_standard_errors(reps[key], block, label=f"{plan.label} {key}")

    def test_w2_the_estimate_equals_the_enumerated_functional(
        self, fitted: tuple[Any, dict[str, Any]]
    ) -> None:
        _, store = fitted
        book = store["book"]
        weights = np.asarray(book.data.weights, dtype=float)
        for plan in store["plans"][-3:]:
            store["plan"] = plan
            exact = hook_from(phi_b)(book, plan, store["cells"])
            phis, _ = replicate_streams(store, replicates=40, n_draws=25, n_derivative_draws=2)
            estimates = (weights[None, :, None] * phis).sum(axis=1) / weights.sum()
            target = (weights[:, None] * exact.phi).sum(axis=0) / weights.sum()
            assert_within_standard_errors(estimates, target, label=plan.label)

    def test_w5_the_curve_has_mean_zero(self) -> None:
        frame = panel_b(self.N, seed=11, weighted=True)
        roles = dict(ROLES_B, weights="wt")
        result = fit_with(frame, regimens_b("never", "always", "respond"), roles, n_draws=3)
        for name in ("ey_regimen[never]", "ey_regimen[always]", "ey_regimen[respond]"):
            curve = result.influence_curves[name]
            assert abs(np.mean(curve)) < 1e-10 * max(1.0, np.max(np.abs(curve)))
            assert np.std(curve) > 0.05

    def test_w6_the_contrast_curve_is_the_difference_of_the_level_curves(self) -> None:
        frame = panel_b(self.N, seed=11)
        result = fit_with(frame, regimens_b("never", "always"), ROLES_B, n_draws=3)
        contrast = "ate_regimen[always vs never]"
        np.testing.assert_allclose(
            result.influence_curves[contrast],
            result.influence_curves["ey_regimen[always]"]
            - result.influence_curves["ey_regimen[never]"],
            rtol=0,
            atol=1e-14,
        )
        assert result.psi(contrast) == pytest.approx(
            result.psi("ey_regimen[always]") - result.psi("ey_regimen[never]"), abs=1e-14
        )

    def test_w6_regimens_share_their_uniforms(self, fitted: tuple[Any, dict[str, Any]]) -> None:
        """Two plans equal before node 2 draw bitwise-identical covariates there."""
        _, store = fitted
        book = store["book"]
        always = next(p for p in store["plans"] if p.label == "always")
        respond = next(p for p in store["plans"] if p.label == "respond")
        units = np.arange(book.data.n, dtype=np.int64)
        weight = np.full(units.size * 4, 1.0)
        tables = []
        for plan in (always, respond):
            simulator = gf._Simulator(book, plan, 77, gf.STREAM_ESTIMATE)
            tables.append(simulator.simulate(units, 4, weight).table)
        column = book.layout.variable_start[(2, 0)]
        np.testing.assert_array_equal(tables[0][:, column], tables[1][:, column])
        assert 0.0 < tables[0][:, column].mean() < 1.0

    def test_w6_common_random_numbers_shrink_the_simulation_part_of_a_contrast(
        self, fitted: tuple[Any, dict[str, Any]]
    ) -> None:
        """On a law where both arms move together, the paired difference varies less."""
        _, store = fitted
        book = store["book"]
        never = next(p for p in store["plans"] if p.label == "never")
        always = next(p for p in store["plans"] if p.label == "always")
        settings = seeded_settings(None, n_draws=2, n_derivative_draws=60, seed=5)
        copies = [
            gf._simulate_plan(book, plan, store["cells"], settings).copy_values[:, :, 0]
            for plan in (never, always)
        ]
        spread = [np.mean(np.var(c, axis=1, ddof=1)) for c in copies]
        paired = np.mean(np.var(copies[1] - copies[0], axis=1, ddof=1))
        assert paired < 0.8 * (spread[0] + spread[1])
        assert min(spread) > 1e-4

    @pytest.mark.parametrize("m", [1, 200])
    def test_w4_the_simulation_variance_term(self, m: int) -> None:
        """The curve of simulated unit means minus the exact curve has the predicted size."""
        frame = panel_b(1000, seed=13, weighted=True)
        roles = dict(ROLES_B, weights="wt")
        exact_store: dict[str, Any] = {}
        result_exact = fit_with(
            frame,
            regimens_b("always"),
            roles,
            hook=hook_from(phi_b, exact_store),
            n_draws=m,
        )
        exact_terms = hook_from(phi_b)(
            exact_store["book"], exact_store["plan"], exact_store["cells"]
        )

        def hybrid(book: Any, plan: Any, cells: Any) -> gf.ExactTerms:
            settings = seeded_settings(None, n_draws=m, n_derivative_draws=2, seed=21)
            simulated = gf._simulate_plan(book, plan, cells, settings)
            return gf.ExactTerms(simulated.phi, exact_terms.derivatives)

        result_hybrid = fit_with(frame, regimens_b("always"), roles, hook=hybrid, n_draws=m)
        name = "ey_regimen[always]"
        difference = result_hybrid.influence_curves[name] - result_exact.influence_curves[name]
        book = exact_store["book"]
        sigma2 = simulation_variance_of_units(book, exact_store["plan"], theta_of(book))
        weights = np.asarray(book.data.weights, dtype=float)
        scaled = weights / weights.mean()
        expected = np.mean(scaled**2 * sigma2) / m
        observed = np.mean(difference**2)
        error = np.std(difference**2, ddof=1) / np.sqrt(difference.size)
        assert abs(observed - expected) < 4.0 * error + 0.02 * expected
        assert expected * m > 5e-3
        # The reported share is the same quantity from the derivative stream's copies.
        settings = seeded_settings(None, n_draws=m, n_derivative_draws=40, seed=22)
        wide = fit_with(frame, regimens_b("always"), roles, n_draws=m, n_derivative_draws=40)
        curve = wide.influence_curves[name]
        share = wide.simulation.monte_carlo_share[name]
        assert share * np.sum(curve**2) / curve.size == pytest.approx(expected, rel=0.08)
        del settings, result_exact, result_hybrid


def assert_production_matches_exact(
    store: dict[str, Any],
    phi: Callable[..., np.ndarray],
    *,
    replicates: int = 80,
    n_derivative_draws: int = 10,
    n_draws: int = 20,
    nonzero: float = 0.01,
) -> None:
    """W2 and W3b for every plan the hook saw: the estimate and each model's derivative."""
    book = store["book"]
    weights = np.asarray(book.data.weights, dtype=float)
    for plan in store["plans"]:
        store["plan"] = plan
        exact = hook_from(phi)(book, plan, store["cells"])
        phis, derivatives = replicate_streams(
            store,
            replicates=replicates,
            n_draws=n_draws,
            n_derivative_draws=n_derivative_draws,
        )
        estimates = (weights[None, :, None] * phis).sum(axis=1) / weights.sum()
        target = (weights[:, None] * exact.phi).sum(axis=0) / weights.sum()
        assert_within_standard_errors(estimates, target, label=f"{plan.label} estimate")
        assert set(derivatives) >= set(exact.derivatives)
        for key, block in exact.derivatives.items():
            assert np.max(np.abs(block)) > nonzero, key
            assert_within_standard_errors(derivatives[key], block, label=f"{plan.label} {key}")


class TestProductionLawR:
    """W2 and W3b on the rich law: each draw kind and the outcome mean."""

    def test_the_production_terms_are_the_exact_terms(self) -> None:
        frame = panel_r(100, seed=5)
        _, store = run_exact(
            frame,
            [PLANS_R["static"], PLANS_R["dynamic"]],
            ROLES_R,
            phi_r,
            covariate_types=TYPES_R,
            n_draws=2,
        )
        keys = {model.key for model in store["book"].stacked}
        assert keys == {"covariate:2:L2", "covariate:3:L3", "outcome"}
        assert_production_matches_exact(store, phi_r, replicates=60, n_derivative_draws=10)


# ------------------------------------------------------- law M: a modified treatment policy
#
#   W1 -> A1 (dose) -> L2 (binary) -> A2 (dose) -> Y

ROLES_M: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
    "continuous_treatment": ["A1", "A2"],
}
SHIFT = 0.5


def panel_m(n: int, seed: int) -> pd.DataFrame:
    """Law M: two continuous doses, a binary covariate between them, and a binary outcome."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    a1 = 0.3 * w1 + rng.normal(size=n)
    c1 = rng.binomial(1, expit(1.6 - 0.2 * a1)).astype(float)
    l2 = rng.binomial(1, expit(-0.2 + 0.6 * a1 + 0.4 * w1)).astype(float)
    a2 = 0.2 + 0.5 * a1 + 0.6 * l2 + 0.2 * w1 + rng.normal(scale=0.9, size=n)
    c2 = rng.binomial(1, expit(1.5 - 0.1 * a2)).astype(float)
    y = rng.binomial(1, expit(-0.3 + 0.5 * a1 + 0.6 * a2 + 0.5 * l2 - 0.2 * w1)).astype(float)
    frame = pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})
    frame.loc[c1 == 0, ["L2", "A2", "C2", "Y"]] = np.nan
    frame.loc[(c1 == 1) & (c2 == 0), ["Y"]] = np.nan
    return frame


def regimens_m() -> dict[str, Any]:
    """The natural course and a uniform shift at both nodes."""
    return {"natural": Shift(0.0, cap=None), "shift": Shift(SHIFT, cap=None)}


def phi_m(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law M by enumeration of ``L2`` and Gauss-Hermite over the natural dose ``A2``.

    Node 1 takes the observed dose as the natural value.  Node 2 draws it from the fitted
    model given the intervened history.  Both are then shifted.
    """
    data = book.data
    delta = {"natural": 0.0, "shift": SHIFT}[plan.label]
    nodes, weights = np.polynomial.hermite_e.hermegauss(30)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = data.baseline[:, 0]
    a1 = data.treatment[:, 0] + delta
    p_l2 = expit(linear(book, "covariate:2:L2", theta["covariate:2:L2"], np.column_stack([w1, a1])))
    th_a = theta["treatment:2"]
    scale = np.sqrt(th_a[-1])
    total = np.zeros(data.n, dtype=complex)
    for value, prob in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
        l2 = np.full(data.n, value)
        mean = linear(book, "treatment:2", th_a, np.column_stack([w1, a1, l2]))
        for z, weight in zip(nodes, weights, strict=True):
            a2 = mean + scale * z + delta
            x = np.column_stack([w1, a1, l2, a2])
            total = total + weight * prob * expit(linear(book, "outcome", theta["outcome"], x))
    return total[:, None]


class TestModifiedTreatmentPolicy:
    """W3a, W3b and W2 for a shift: the natural-treatment model joins the stack."""

    def test_the_curve_is_the_gateaux_derivative(self) -> None:
        frame = panel_m(100, seed=7)
        result, store = run_exact(frame, regimens_m(), ROLES_M, phi_m, n_draws=2)
        assert {m.key for m in store["book"].stacked} == {
            "covariate:2:L2",
            "treatment:2",
            "outcome",
        }
        units = np.random.default_rng(1).choice(100, 20, replace=False)
        for plan in store["plans"][-2:]:
            expected = gateaux_curve(store["book"], phi_m, plan, units=units)
            np.testing.assert_allclose(
                result.influence_curves[f"ey_regimen[{plan.label}]"][units],
                expected,
                rtol=0,
                atol=1e-6,
            )
            assert np.max(np.abs(expected)) > 0.05

    def test_the_production_terms_are_the_exact_terms(self) -> None:
        frame = panel_m(100, seed=7)
        _, store = run_exact(frame, regimens_m(), ROLES_M, phi_m, n_draws=2)
        store["plans"] = store["plans"][-2:]
        assert_production_matches_exact(store, phi_m, replicates=60)

    def test_the_first_node_keeps_the_observed_dose(self) -> None:
        """MTP node 1 shifts the observed dose and draws none, so its model is not fitted."""
        frame = panel_m(100, seed=7)
        _, store = run_exact(frame, regimens_m(), ROLES_M, phi_m, n_draws=2)
        assert set(store["book"].treatment) == {2}


# -------------------------------------------------------------------- law T: an odds tilt

TILTS = {"x2": 2.0, "x0.5": 0.5}


def tilts() -> list[Incremental]:
    """Two odds tilts, each broadcast to every node."""
    return [Incremental(delta, name=name) for name, delta in TILTS.items()]


def phi_t(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law B under an odds tilt at both nodes: the treatment is drawn, not assigned."""
    data = book.data
    delta = TILTS[plan.label]
    w1, w2 = data.baseline[:, 0], data.baseline[:, 1]

    def tilted(key: str, x: np.ndarray) -> np.ndarray:
        g = expit(linear(book, key, theta[key], x))
        return delta * g / (delta * g + 1.0 - g)

    total = np.zeros(data.n, dtype=complex)
    q1 = tilted("treatment:1", np.column_stack([w1, w2]))
    for a1, p1 in ((0.0, 1.0 - q1), (1.0, q1)):
        a1v = np.full(data.n, a1)
        x2 = np.column_stack([w1, w2, a1v])
        p_l2 = expit(linear(book, "covariate:2:L2", theta["covariate:2:L2"], x2))
        for l2, pl in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
            l2v = np.full(data.n, l2)
            q2 = tilted("treatment:2", np.column_stack([w1, w2, a1v, l2v]))
            for a2, p2 in ((0.0, 1.0 - q2), (1.0, q2)):
                x = np.column_stack([w1, w2, a1v, l2v, np.full(data.n, a2)])
                total = total + p1 * pl * p2 * expit(linear(book, "outcome", theta["outcome"], x))
    return total[:, None]


class TestOddsTilt:
    """An odds tilt draws the treatment from the tilted law and scores the tilted draw."""

    def test_the_curve_is_the_gateaux_derivative(self) -> None:
        frame = panel_b(100, seed=9)
        result, store = run_exact(frame, tilts(), ROLES_B, phi_t, n_draws=2)
        assert {m.key for m in store["book"].stacked} == {
            "treatment:1",
            "covariate:2:L2",
            "treatment:2",
            "outcome",
        }
        units = np.random.default_rng(2).choice(100, 25, replace=False)
        for plan in store["plans"][-2:]:
            expected = gateaux_curve(store["book"], phi_t, plan, units=units)
            np.testing.assert_allclose(
                result.influence_curves[f"ey_regimen[{plan.label}]"][units],
                expected,
                rtol=0,
                atol=1e-6,
            )
            assert np.max(np.abs(expected)) > 0.05

    def test_the_production_terms_are_the_exact_terms(self) -> None:
        """The tilted draw's score is ``x (a - q)``, not the score of the natural law."""
        frame = panel_b(100, seed=9)
        _, store = run_exact(frame, tilts(), ROLES_B, phi_t, n_draws=2)
        store["plans"] = store["plans"][-2:]
        assert_production_matches_exact(store, phi_t, replicates=80)


# ---------------------------------------------------- law S: survival and competing risks
#
#   W1 -> A1 -> C1 -> Y1 -> L2 -> A2 -> C2 -> Y2 -> L3 -> A3 -> C3 -> Y3

HORIZONS_S = (1, 2, 3)


def outcome_roles_s(causes: int) -> Any:
    """``outcome=`` for one event column per node, or a cause-to-columns mapping."""
    if causes == 1:
        return ["Y1", "Y2", "Y3"]
    return {"A": ["YA1", "YA2", "YA3"], "B": ["YB1", "YB2", "YB3"]}


def roles_s(causes: int) -> dict[str, Any]:
    """The roles of law S."""
    return {
        "outcome": outcome_roles_s(causes),
        "treatment": ["A1", "A2", "A3"],
        "baseline": ["W1"],
        "time_varying": [[], ["L2"], ["L3"]],
        "censoring": ["C1", "C2", "C3"],
    }


def panel_s(n: int, seed: int, *, causes: int = 1) -> pd.DataFrame:
    """Law S: discrete-time hazards driven by the treatment and by two binary covariates."""
    rng = np.random.default_rng(seed)
    alive = np.ones(n, dtype=bool)
    columns: dict[str, np.ndarray] = {}
    w1 = rng.normal(size=n)
    columns["W1"] = w1
    state = {"A": np.zeros(n), "L": np.zeros(n)}
    history: dict[str, np.ndarray] = {}

    def nan_unless(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
        return np.where(keep, values, np.nan)

    for node in (1, 2, 3):
        if node == 1:
            lnode = np.zeros(n)
        else:
            lnode = rng.binomial(
                1, expit(-0.2 + 0.7 * history["A1"] * (node == 2) + 0.4 * w1 + 0.8 * state["L"])
            ).astype(float)
            columns[f"L{node}"] = nan_unless(lnode, alive)
        a = rng.binomial(1, expit(0.1 + 0.4 * w1 + 0.6 * lnode - 0.4 * state["A"])).astype(float)
        columns[f"A{node}"] = nan_unless(a, alive)
        c = rng.binomial(1, expit(2.0 - 0.3 * a)).astype(float)
        columns[f"C{node}"] = nan_unless(c, alive)
        observed = alive & (c == 1)
        eta1 = -1.7 + 0.5 * a + 0.3 * w1 + 0.7 * lnode
        if causes == 1:
            event = rng.binomial(1, expit(eta1)).astype(float)
            columns[f"Y{node}"] = nan_unless(event, observed)
            happened = event == 1
        else:
            eta2 = -2.0 - 0.4 * a + 0.5 * w1 + 0.3 * lnode
            weights = np.column_stack([np.ones(n), np.exp(eta1), np.exp(eta2)])
            weights = weights / weights.sum(axis=1, keepdims=True)
            u = rng.uniform(size=n)
            code = (u[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
            columns[f"YA{node}"] = nan_unless((code == 1).astype(float), observed)
            columns[f"YB{node}"] = nan_unless((code == 2).astype(float), observed)
            happened = code > 0
        alive = observed & ~happened
        history[f"A{node}"] = a
        state = {"A": a, "L": lnode}
    order = ["W1"]
    for node in (1, 2, 3):
        order += [f"L{node}"] if node > 1 else []
        order += [f"A{node}", f"C{node}"]
        order += [f"Y{node}"] if causes == 1 else [f"YA{node}", f"YB{node}"]
    return pd.DataFrame({name: columns[name] for name in order})


PLANS_S: dict[str, Any] = {
    "always": Regimen("always", (1, 1, 1)),
    "never": Regimen("never", (0, 0, 0)),
    "dynamic": DynamicRegimen(
        "dynamic", (1, lambda h: h["L2"], lambda h: 1.0 - h["L3"]), rule_kind="known"
    ),
}


def hazards(
    book: gf.ModelBook, node: int, theta: dict[str, np.ndarray], x: np.ndarray
) -> np.ndarray:
    """``(n, K)`` cause-specific discrete hazards at ``node`` from the fitted hazard model."""
    key = f"hazard:{node}"
    if len(book.data.cause_labels) <= 1:
        return expit(linear(book, key, theta[key], x))[:, None]
    z = np.exp(logits(book, key, theta[key], x))
    return (z / z.sum(axis=1, keepdims=True))[:, 1:]


def phi_s(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Cumulative incidence at horizons 1 to 3 by enumeration of ``L2`` and ``L3``."""
    data = book.data
    n_causes = max(len(data.cause_labels), 1)
    w1 = data.baseline[:, 0]
    a1 = assigned(plan, data, 1, {"W1": w1})
    h1 = hazards(book, 1, theta, np.column_stack([w1, a1]))
    cif = np.zeros((data.n, n_causes, 3), dtype=complex)
    s1 = 1.0 - h1.sum(axis=1)
    p_l2_all = expit(
        linear(book, "covariate:2:L2", theta["covariate:2:L2"], np.column_stack([w1, a1]))
    )
    for l2, p_l2 in ((0.0, 1.0 - p_l2_all), (1.0, p_l2_all)):
        l2v = np.full(data.n, l2)
        a2 = assigned(plan, data, 2, {"W1": w1, "A1": a1, "L2": l2v})
        h2 = hazards(book, 2, theta, np.column_stack([w1, a1, l2v, a2]))
        s2 = s1 * (1.0 - h2.sum(axis=1))
        x3 = np.column_stack([w1, a1, l2v, a2])
        p_l3_all = expit(linear(book, "covariate:3:L3", theta["covariate:3:L3"], x3))
        for l3, p_l3 in ((0.0, 1.0 - p_l3_all), (1.0, p_l3_all)):
            l3v = np.full(data.n, l3)
            a3 = assigned(plan, data, 3, {"W1": w1, "A1": a1, "L2": l2v, "A2": a2, "L3": l3v})
            h3 = hazards(book, 3, theta, np.column_stack([w1, a1, l2v, a2, l3v, a3]))
            path = p_l2 * p_l3
            first = h1
            second = h1 + s1[:, None] * h2
            third = second + s2[:, None] * h3
            for k, value in enumerate((first, second, third)):
                cif[:, :, k] += path[:, None] * value
    return cif.reshape(data.n, n_causes * 3)


def cell_order(store: dict[str, Any]) -> list[tuple[str | None, int]]:
    """``(cause, horizon)`` of each cell, in the order the hook returns them."""
    return [(cell.cause, cell.horizon) for cell in store["cells"]]


class TestSurvivalAndCompetingRisks:
    """W3a and W3b on discrete hazards: binomial for one cause, multinomial for several."""

    @pytest.mark.parametrize("causes", [1, 2])
    def test_the_curve_is_the_gateaux_derivative(self, causes: int) -> None:
        frame = panel_s(200, seed=17, causes=causes)
        result, store = run_exact(
            frame,
            [PLANS_S["always"], PLANS_S["dynamic"]],
            roles_s(causes),
            phi_s,
            covariate_types=None,
            n_draws=2,
        )
        cells = cell_order(store)
        assert [horizon for _, horizon in cells[:3]] == [1, 2, 3]
        keys = {m.key for m in store["book"].stacked}
        assert {"hazard:1", "hazard:2", "hazard:3", "covariate:2:L2", "covariate:3:L3"} <= keys
        units = np.random.default_rng(3).choice(200, 12, replace=False)
        plans = store["plans"][-2:]
        for plan in plans:
            for position, (cause, horizon) in enumerate(cells):
                expected = gateaux_curve(store["book"], phi_s, plan, cell=position, units=units)
                name = (
                    f"risk_regimen[{plan.label} @ t={horizon}]"
                    if cause is None
                    else f"cif_regimen[{plan.label}, {cause} @ t={horizon}]"
                )
                np.testing.assert_allclose(
                    result.influence_curves[name][units], expected, rtol=0, atol=1e-6
                )
                if horizon == 3:
                    assert np.max(np.abs(expected)) > 0.05

    @pytest.mark.parametrize("causes", [1, 2])
    def test_the_production_terms_are_the_exact_terms(self, causes: int) -> None:
        frame = panel_s(200, seed=17, causes=causes)
        _, store = run_exact(
            frame,
            [PLANS_S["always"], PLANS_S["dynamic"]],
            roles_s(causes),
            phi_s,
            n_draws=2,
        )
        store["plans"] = store["plans"][-2:]
        assert_production_matches_exact(store, phi_s, replicates=60, nonzero=0.002)


# ------------------------------------------------------------ the exact laws (W1 and W9)
#
# Each oracle law realises its distribution in a finite sample, so a saturated GLM is the
# population fit and the g-formula at the fitted parameters is the nonparametric functional.
# One row per support point, weighted by its count, is the same weighted empirical law, and
# the curve of a row is then ``(w / mean w)`` times the efficient influence function there.

SATURATED = GLM(interactions=8, tol=1e-13)
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
CENSORED_ROLES = {
    "treatment": ["A1", "A2"],
    "censoring": ["C1", "C2"],
    "time_varying": [[], ["L2"]],
    "baseline": ["W"],
}


def phi_s2(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Two-node cumulative incidence by enumeration of ``L2``: ``(n, K * 2)``, cause-major."""
    data = book.data
    n_causes = max(len(data.cause_labels), 1)
    start = baseline_history(data)
    a1 = assigned(plan, data, 1, start)
    x1 = np.column_stack([data.baseline, a1])
    h1 = hazards(book, 1, theta, x1)
    s1 = 1.0 - h1.sum(axis=1)
    p_l2 = expit(linear(book, "covariate:2:L2", theta["covariate:2:L2"], x1))
    cif = np.zeros((data.n, n_causes, 2), dtype=complex)
    for value, prob in ((0.0, 1.0 - p_l2), (1.0, p_l2)):
        l2 = np.full(data.n, value)
        a2 = assigned(plan, data, 2, {**start, "A1": a1, "L2": l2})
        h2 = hazards(book, 2, theta, np.column_stack([data.baseline, a1, l2, a2]))
        cif[:, :, 0] += prob[:, None] * h1
        cif[:, :, 1] += prob[:, None] * (h1 + s1[:, None] * h2)
    return cif.reshape(data.n, n_causes * 2)


def compressed(module: Any) -> pd.DataFrame:
    """One row per support point of an oracle law, weighted by its count."""
    frame = module.frame().iloc[module.first_row_of()].reset_index(drop=True)
    frame["wt"] = np.asarray(module.COUNTS, dtype=float)
    return frame


def oracle_case(name: str) -> dict[str, Any]:
    """The regimens, data, roles, truth and enumerator of one exact law."""
    if name == "binary":
        return {
            "regimens": declared_regimens(binary.REGIMEN_SPEC),
            "frame": compressed(binary),
            "roles": {**CENSORED_ROLES, "outcome": "Y"},
            "reference": binary.REGIMEN_REFERENCE,
            "truth": binary.TRUTH,
            "eif": binary.eif,
            "phi": phi_b,
        }
    if name == "survival":
        return {
            "regimens": declared_regimens(survival.REGIMEN_SPEC),
            "frame": compressed(survival),
            "roles": {**CENSORED_ROLES, "outcome": ["Y1", "Y2"]},
            "reference": survival.REGIMEN_REFERENCE,
            "truth": survival.TRUTH,
            "eif": survival.eif,
            "phi": phi_s2,
        }
    if name == "competing":
        return {
            "regimens": declared_regimens(competing.REGIMEN_SPEC),
            "frame": compressed(competing),
            "roles": {**CENSORED_ROLES, "outcome": competing.outcome_columns()},
            "reference": competing.REGIMEN_REFERENCE,
            "truth": competing.TRUTH,
            "eif": competing.eif,
            "phi": phi_s2,
        }
    if name == "three_arm":
        labels = ("never", "always")
        return {
            "regimens": declared_regimens(multivalue.REGIMEN_SPEC),
            "frame": compressed(multivalue),
            "roles": {
                "outcome": "Y",
                "treatment": ["A1", "A2"],
                "baseline": ["W"],
                "time_varying": [[], ["L2"]],
            },
            "reference": multivalue.REFERENCE,
            "truth": multivalue.TRUTH,
            "eif": None,
            "phi": None,
            "labels": labels,
        }
    labels = tuple(policy.PLANS)
    return {
        "regimens": policies.regimens(labels),
        "frame": compressed(multivalue),
        "roles": {
            "outcome": "Y",
            "treatment": ["A1", "A2"],
            "baseline": ["W"],
            "time_varying": [[], ["L2"]],
        },
        "reference": labels[0],
        "truth": policy.TRUTH,
        "eif": None,
        "phi": None,
    }


LEARNERS_BY_CASE = {
    "binary": {"outcome_learner": SATURATED},
    "survival": {"hazard_learner": SATURATED},
    "competing": {"hazard_learner": SATURATED},
    "three_arm": {"outcome_learner": SATURATED},
    "policy": {"outcome_learner": SATURATED},
}


class TestExactLaws:
    """W1 and W9 on the oracle laws."""

    @pytest.mark.parametrize("case", list(LEARNERS_BY_CASE))
    def test_w1_the_simulated_estimate_is_the_g_formula(self, case: str) -> None:
        spec = oracle_case(case)
        result = fit_with(
            spec["frame"],
            spec["regimens"],
            {**spec["roles"], "weights": "wt"},
            reference=spec["reference"],
            covariate_learner=SATURATED,
            treatment_learner=SATURATED,
            censoring_learner=SATURATED,
            g_bounds=NO_TRUNCATION,
            n_draws=400,
            **LEARNERS_BY_CASE[case],
        )
        assert set(result.estimates) == set(spec["truth"])
        errors = {name: monte_carlo_se(result, name) for name in spec["truth"]}
        for name, value in spec["truth"].items():
            assert abs(result.psi(name) - value) <= 4.0 * errors[name] + 1e-9, name
        # The nonzero witness: the simulation is noisy for some parameters, so the 4 SE band
        # is a statement about draws and not only about a deterministic number.
        assert sum(error > 1e-4 for error in errors.values()) >= 3
        assert result.inference_status == "influence_curve"

    @pytest.mark.parametrize("case", ["binary", "survival", "competing"])
    def test_w9_the_exact_curve_is_the_efficient_influence_curve(self, case: str) -> None:
        spec = oracle_case(case)
        frame = spec["frame"]
        roles = {**spec["roles"], "weights": "wt"}
        settings = {
            "reference": spec["reference"],
            "covariate_learner": SATURATED,
            "treatment_learner": SATURATED,
            "censoring_learner": SATURATED,
            "g_bounds": NO_TRUNCATION,
            "n_draws": 2,
            **LEARNERS_BY_CASE[case],
        }
        exact = fit_with(frame, spec["regimens"], roles, hook=hook_from(spec["phi"]), **settings)
        scale = frame["wt"].to_numpy() / frame["wt"].mean()
        gcomp_learners = {
            "outcome_learner": SATURATED,
            "pseudo_learner": SATURATED,
            "treatment_learner": SATURATED,
            "censoring_learner": SATURATED,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WeightingWarning)
            gcomp = SequentialGComputation(
                spec["regimens"],
                reference=spec["reference"],
                g_bounds=NO_TRUNCATION,
                simultaneous=False,
                **gcomp_learners,
            ).fit(frame, **roles)
        for name, value in spec["truth"].items():
            assert exact.psi(name) == pytest.approx(value, abs=1e-10), name
            assert gcomp.psi(name) == pytest.approx(value, abs=1e-10), name
            reported = exact.influence_curves[name] / scale
            np.testing.assert_allclose(reported, spec["eif"](name), rtol=0, atol=1e-8)
            np.testing.assert_allclose(
                exact.influence_curves[name], gcomp.influence_curves[name], rtol=0, atol=1e-8
            )


# ------------------------------------------------------------- law F: a fixed continuous dose
#
#   W1 -> A1 -> L2 -> A2 -> Y, every conditional law linear-Gaussian

ROLES_F: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
    "continuous_treatment": ["A1", "A2"],
}
BOUNDS_F = (-12.0, 16.0)


def panel_f(n: int, seed: int) -> pd.DataFrame:
    """Law F: linear-Gaussian, with censoring that depends on the history."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    a1 = 0.3 * w1 + rng.normal(size=n)
    c1 = rng.binomial(1, expit(1.8 - 0.2 * a1)).astype(float)
    l2 = 0.5 + 0.6 * w1 + 0.8 * a1 + rng.normal(scale=0.7, size=n)
    a2 = 0.2 + 0.5 * a1 + 0.4 * l2 + rng.normal(size=n)
    c2 = rng.binomial(1, expit(1.6 - 0.1 * a2)).astype(float)
    y = 1.0 + 0.5 * w1 + 0.7 * a1 + 0.6 * l2 + 0.9 * a2 + rng.normal(size=n)
    frame = pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})
    frame.loc[c1 == 0, ["L2", "A2", "C2", "Y"]] = np.nan
    frame.loc[(c1 == 1) & (c2 == 0), ["Y"]] = np.nan
    return frame


def dose_plan(label: str, first: float, second: float) -> Regimen:
    """A fixed dose at each of the two continuous nodes."""
    return Regimen(label, (first, second))


def phi_f(book: gf.ModelBook, plan: Any, theta: dict[str, np.ndarray]) -> np.ndarray:
    """Law F's unit means on the scaled outcome: the outcome is linear in ``L2``."""
    data = book.data
    d1, d2 = (float(gf.plan_node(plan.regimen, node)[1]) for node in (1, 2))
    w1 = data.baseline[:, 0]
    a1 = np.full(data.n, d1)
    mean_l2 = linear(book, "covariate:2:L2", theta["covariate:2:L2"], np.column_stack([w1, a1]))
    x = np.column_stack([w1, a1, mean_l2, np.full(data.n, d2)])
    return linear(book, "outcome", theta["outcome"], x)[:, None]


def closed_form_dose(frame: pd.DataFrame, d1: float, d2: float) -> tuple[float, np.ndarray]:
    """The scaled estimate and its curve for a fixed dose, from two OLS fits and a mean."""
    lo, hi = BOUNDS_F
    n = len(frame)
    w1, a1 = frame["W1"].to_numpy(), frame["A1"].to_numpy()
    rows1 = frame["C1"].to_numpy() == 1
    rows2 = frame["C2"].to_numpy() == 1
    l2, a2 = frame["L2"].to_numpy(), frame["A2"].to_numpy()
    y = (frame["Y"].to_numpy() - lo) / (hi - lo)
    x1 = np.column_stack([np.ones(n), w1, a1])
    a, *_ = np.linalg.lstsq(x1[rows1], l2[rows1], rcond=None)
    r1 = np.where(rows1, np.nan_to_num(l2) - x1 @ a, 0.0)
    x2 = np.column_stack([np.ones(n), w1, a1, np.nan_to_num(l2), np.nan_to_num(a2)])
    b, *_ = np.linalg.lstsq(x2[rows2], y[rows2], rcond=None)
    r2 = np.where(rows2, np.nan_to_num(y) - x2 @ b, 0.0)
    w_bar = w1.mean()

    def at(w: Any) -> Any:
        mean_l2 = a[0] + a[1] * w + a[2] * d1
        return b[0] + b[1] * w + b[2] * d1 + b[3] * mean_l2 + b[4] * d2

    psi = float(at(w_bar))
    info1 = (x1[rows1].T @ x1[rows1]) / n
    info2 = (x2[rows2].T @ x2[rows2]) / n
    grad_a = b[3] * np.array([1.0, w_bar, d1])
    grad_b = np.array([1.0, w_bar, d1, a[0] + a[1] * w_bar + a[2] * d1, d2])
    curve = (
        at(w1)
        - psi
        + (x1 * r1[:, None]) @ np.linalg.solve(info1, grad_a)
        + (x2 * r2[:, None]) @ np.linalg.solve(info2, grad_b)
    )
    return psi, curve


class TestFixedDose:
    """W10: a fixed dose on a linear-Gaussian law has a closed form."""

    N = 600
    DOSES: ClassVar[dict[str, tuple[float, float]]] = {"central": (0.5, 1.0), "far": (5.0, 10.0)}

    def _plans(self) -> list[Regimen]:
        return [dose_plan(label, *doses) for label, doses in self.DOSES.items()]

    def test_the_exact_curve_is_the_delta_method_curve(self) -> None:
        frame = panel_f(self.N, seed=23)
        result, store = run_exact(
            frame,
            self._plans(),
            ROLES_F,
            phi_f,
            covariate_types={"L2": "normal"},
            q_bounds=BOUNDS_F,
            n_draws=2,
        )
        # The dose is not drawn, so the natural-treatment models are fitted and not stacked.
        assert {m.key for m in store["book"].stacked} == {"covariate:2:L2", "outcome"}
        for label, (d1, d2) in self.DOSES.items():
            psi, curve = closed_form_dose(frame, d1, d2)
            name = f"ey_regimen[{label}]"
            assert result.psi(name) == pytest.approx(
                BOUNDS_F[0] + (BOUNDS_F[1] - BOUNDS_F[0]) * psi, abs=1e-9
            )
            scaled = (BOUNDS_F[1] - BOUNDS_F[0]) * curve
            np.testing.assert_allclose(result.influence_curves[name], scaled, rtol=0, atol=1e-8)
            closed = np.sqrt(np.var(scaled, ddof=1) / self.N)
            assert result.estimates[name].std_error == pytest.approx(closed, rel=1e-6)

    def test_the_simulated_estimate_and_the_support_diagnostic(self) -> None:
        frame = panel_f(self.N, seed=23)
        result = fit_with(
            frame,
            self._plans(),
            ROLES_F,
            covariate_types={"L2": "normal"},
            q_bounds=BOUNDS_F,
            n_draws=200,
            reference="central",
        )
        for label, (d1, d2) in self.DOSES.items():
            psi, _ = closed_form_dose(frame, d1, d2)
            name = f"ey_regimen[{label}]"
            truth = BOUNDS_F[0] + (BOUNDS_F[1] - BOUNDS_F[0]) * psi
            assert abs(result.psi(name) - truth) <= 4.0 * monte_carlo_se(result, name)
        report = result.simulation
        assert report.model_reliance is not None
        rows = {(row.plan, row.node): row for row in report.support}
        assert set(rows) == {("central", 1), ("central", 2), ("far", 1), ("far", 2)}
        assert rows[("central", 1)].off_support < 0.25
        assert rows[("far", 1)].off_support > 0.9
        assert rows[("far", 2)].off_support > 0.9
        assert "model reliance" in result.summary()


# ------------------------------------------------------------------------- the random numbers

#: Known answers of Philox4x32-10 from the Random123 distribution (Salmon et al. 2011).
PHILOX_KAT = [
    ((0, 0, 0, 0), (0, 0), ("6627e8d5", "e169c58d", "bc57ac4c", "9b00dbd8")),
    (
        (0xFFFFFFFF,) * 4,
        (0xFFFFFFFF,) * 2,
        ("408f276d", "41c83b0e", "a20bc7c6", "6d5451fd"),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        ("d16cfe09", "94fdcceb", "5001e420", "24126ea1"),
    ),
]


class TestPhilox:
    @pytest.mark.parametrize(("counter", "key", "words"), PHILOX_KAT)
    def test_the_block_function_matches_the_published_vectors(
        self, counter: Any, key: Any, words: Any
    ) -> None:
        out = gf.philox4x32(counter, key)
        assert tuple(format(int(word), "08x") for word in out) == words

    def test_a_draw_is_a_pure_function_of_its_five_indices(self) -> None:
        units = np.arange(50)
        whole = gf.counter_uniforms(7, 0, units, 3, 2, 5)
        # The same unit and copy give the same draw however the call batches the units.
        part = gf.counter_uniforms(7, 0, units[10:13], 3, 2, 5)
        np.testing.assert_array_equal(whole[10:13], part)
        np.testing.assert_array_equal(whole, gf.counter_uniforms(7, 0, units, 3, 2, 5))
        shuffled = gf.counter_uniforms(7, 0, units[::-1], 3, 2, 5)
        np.testing.assert_array_equal(shuffled[::-1], whole)

    def test_each_index_separates_the_draws(self) -> None:
        units = np.arange(200)
        base = gf.counter_uniforms(7, 0, units, 3, 2, 5)
        others = [
            gf.counter_uniforms(8, 0, units, 3, 2, 5),
            gf.counter_uniforms(7, 1, units, 3, 2, 5),
            gf.counter_uniforms(7, 0, units, 4, 2, 5),
            gf.counter_uniforms(7, 0, units, 3, 3, 5),
            gf.counter_uniforms(7, 0, units, 3, 2, 6),
            gf.counter_uniforms(7 + 2**40, 0, units, 3, 2, 5),
        ]
        for other in others:
            assert not np.any(np.isclose(base, other, atol=0, rtol=0))

    def test_the_uniforms_are_on_the_open_interval_and_uniform(self) -> None:
        draws = gf.counter_uniforms(11, 0, np.arange(200_000), 0, 1, 0)
        assert draws.min() > 0.0 and draws.max() < 1.0
        assert abs(draws.mean() - 0.5) < 4 * np.sqrt(1 / 12 / draws.size)
        assert abs(draws.var() - 1 / 12) < 2e-3
        # Adjacent units are not correlated, which a counter-based generator could get wrong.
        assert abs(np.corrcoef(draws[:-1], draws[1:])[0, 1]) < 4 / np.sqrt(draws.size)


# --------------------------------------------------------------- W11: refusals before a learner


class NeverFit:
    """A learner that fails the test if a refusal comes after the nuisance fit."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")


def small_b(n: int = 300) -> pd.DataFrame:
    return panel_b(n, seed=5)


def quiet_fit(estimator: ParametricGFormula, frame: pd.DataFrame, **roles: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, **(roles or ROLES_B))


def panel_l2(kind: str, n: int = 300) -> pd.DataFrame:
    """Law B with ``L2`` replaced by a column of another type."""
    frame = panel_b(n, seed=5)
    rng = np.random.default_rng(3)
    seen = frame["L2"].notna()
    if kind == "continuous":
        frame.loc[seen, "L2"] = frame.loc[seen, "L2"] + rng.normal(size=int(seen.sum()))
    else:
        frame.loc[seen, "L2"] = rng.integers(0, 3, int(seen.sum())).astype(float)
    return frame


class TestRefusals:
    def test_an_undeclared_covariate_is_refused_and_names_the_declaration(self) -> None:
        frame = panel_l2("continuous")
        with pytest.raises(CapabilityError, match=r"covariate_types=\{'L2': 'normal'\}") as raised:
            ParametricGFormula({"a": 1}, outcome_learner=NeverFit()).fit(frame, **ROLES_B)
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    @pytest.mark.parametrize(
        ("declared", "match"),
        [
            ({"nope": "normal"}, "not a time-varying covariate column"),
            ({"L2": "ordinal"}, "a type is one of"),
        ],
    )
    def test_a_bad_declaration_is_refused(self, declared: Any, match: str) -> None:
        error = ValueError if declared == {"L2": "ordinal"} else DataError
        with pytest.raises(error, match=match):
            ParametricGFormula({"a": 1}, covariate_types=declared).fit(small_b(), **ROLES_B)

    def test_a_declaration_the_data_contradicts_is_refused(self) -> None:
        with pytest.raises(DataError, match="declares 'L2' binary"):
            ParametricGFormula({"a": 1}, covariate_types={"L2": "binary"}).fit(
                panel_l2("categorical"), **ROLES_B
            )

    def test_a_gaussian_covariate_needs_a_glm(self) -> None:
        from sklearn.linear_model import LinearRegression

        with pytest.raises(CapabilityError, match="L2"):
            ParametricGFormula(
                {"a": 1},
                covariate_types={"L2": "normal"},
                covariate_learner=LinearRegression(),
            ).fit(panel_l2("continuous"), **ROLES_B)

    @pytest.mark.parametrize("argument", ["covariate_learner", "treatment_learner"])
    def test_a_glm_family_is_refused_because_the_fit_chooses_it(self, argument: str) -> None:
        with pytest.raises(CapabilityError, match="names a family") as raised:
            ParametricGFormula({"a": 1}, **{argument: GLM(family="binomial")}).fit(
                small_b(), **ROLES_B
            )
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_learner_that_nothing_reads_is_refused(self) -> None:
        frame = panel_s(200, seed=1)
        with pytest.raises(CapabilityError, match="nothing to fit on a survival outcome"):
            ParametricGFormula({"a": Regimen("a", (1, 1, 1))}, outcome_learner=GLM()).fit(
                frame, **roles_s(1)
            )
        with pytest.raises(CapabilityError, match="nothing to fit on an end-of-study"):
            ParametricGFormula({"a": 1}, hazard_learner=GLM()).fit(small_b(), **ROLES_B)

    def test_a_pseudo_learner_is_refused_at_construction(self) -> None:
        with pytest.raises(CapabilityError, match="pseudo") as raised:
            ParametricGFormula({"a": 1}, pseudo_learner=GLM())
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_cross_fitting_is_an_implementation_gap(self) -> None:
        with pytest.raises(CapabilityError, match="fits in sample") as raised:
            ParametricGFormula({"a": 1}, n_folds=2)
        assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_a_working_model_is_refused(self) -> None:
        from tests.unit.test_longitudinal_msm import DOSE

        with pytest.raises(CapabilityError, match=r"LTMLE\(msm=\.\.\.\)"):
            ParametricGFormula({"a": 1}, msm=DOSE)

    @pytest.mark.parametrize("setting", [{"alpha": 0.99}, {"max_iter": 3}, {"tol": 1e-3}])
    def test_a_fluctuation_setting_is_refused(self, setting: Any) -> None:
        with pytest.raises(MethodConfigurationError, match="fit no fluctuation"):
            ParametricGFormula({"a": 1}, **setting)

    @pytest.mark.parametrize("n_draws", [0, -1, 2.5, True])
    def test_n_draws_must_be_a_positive_integer(self, n_draws: Any) -> None:
        with pytest.raises(ValueError, match="n_draws"):
            ParametricGFormula({"a": 1}, n_draws=n_draws)

    @pytest.mark.parametrize("n_derivative_draws", [1, 0, 3.5, False])
    def test_the_derivative_draws_need_a_second_copy(self, n_derivative_draws: Any) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ParametricGFormula({"a": 1}, n_derivative_draws=n_derivative_draws)

    def test_the_default_derivative_draws_never_fall_below_ten(self) -> None:
        assert ParametricGFormula({"a": 1}, n_draws=3).n_derivative_draws == 10
        assert ParametricGFormula({"a": 1}, n_draws=40).n_derivative_draws == 40
        assert ParametricGFormula({"a": 1}, n_draws=3, n_derivative_draws=2).n_derivative_draws == 2

    def test_a_risk_ratio_tilt_is_redirected_before_any_learner(self) -> None:
        from cleverly.interventions import RiskRatioTilt

        plan = DynamicRegimen("rr", (RiskRatioTilt(2.0), 1))
        with pytest.raises(CapabilityError, match="risk-ratio tilt") as raised:
            ParametricGFormula({"rr": plan}, outcome_learner=NeverFit()).fit(small_b(), **ROLES_B)
        assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION

    def test_a_repeated_outcome_is_an_implementation_gap_of_x71(self) -> None:
        from .test_repeated_outcome import panel

        roles = {
            "outcome": Repeated(["Y1", "Y2", "Y3"]),
            "treatment": ["A1", "A2", "A3"],
            "baseline": ["W"],
            "time_varying": [["L1"], ["L2"], ["L3"]],
            "censoring": ["C1", "C2", "C3"],
        }
        with pytest.raises(CapabilityError, match="no per-horizon reduction") as raised:
            ParametricGFormula({"a": 1}, outcome_learner=NeverFit()).fit(panel(), **roles)
        assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "X71" in str(raised.value)

    def test_a_binomial_glm_on_an_unbounded_outcome_is_refused(self) -> None:
        frame = panel_f(200, seed=3)
        roles = {**ROLES_F}
        with pytest.raises(CapabilityError, match="Declare q_bounds="):
            ParametricGFormula(
                {"a": Regimen("a", (1.0, 1.0))},
                covariate_types={"L2": "normal"},
                outcome_learner=GLM(),
            ).fit(frame, **roles)

    def test_a_truncation_curve_is_redirected(self) -> None:
        result = quiet_fit(ParametricGFormula({"a": 1}, random_state=0), small_b())
        with pytest.raises(CapabilityError, match="truncation") as raised:
            longitudinal_truncation_curve(result, [0.01, 0.05])
        assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION

    def test_an_aliased_column_the_regimen_leaves_is_refused(self) -> None:
        """``W1`` equals the sign of ``A1``, so no unit has ``W1 = -1`` and ``A1 = 1``.

        The regimen "always" gives every unit ``A1 = 1``.  A unit with ``W1 = -1`` then lies off
        the fit rows' relation, and the GLM's aliased coefficient would silently set the effect
        of ``W1`` to zero there.
        """
        frame = small_b()
        frame["W1"] = np.where(frame["A1"] == 1, 1.0, -1.0)
        with pytest.raises(DataError, match="not identified by the parametric g-formula"):
            quiet_fit(ParametricGFormula({"always": 1}, random_state=0), frame)


# ------------------------------------------------------------------------------ the status table


class TestStatus:
    def test_every_model_a_glm_gives_the_influence_curve_status(self) -> None:
        result = quiet_fit(ParametricGFormula({"a": 1, "b": 0}, random_state=0), small_b())
        assert result.inference_status == "influence_curve"
        assert {e.inference for e in result.estimates.values()} == {"influence_curve"}
        assert result.simulation.parametric is True
        assert np.isfinite(result.estimates["ey_regimen[a]"].std_error)

    @pytest.mark.parametrize(
        "learner",
        ["outcome_learner", "covariate_learner"],
    )
    def test_another_learner_gives_a_diagnostic_interval(self, learner: str) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeRegressor

        given = (
            DecisionTreeRegressor(max_depth=2, random_state=0)
            if learner == "outcome_learner"
            else LogisticRegression()
        )
        result = quiet_fit(
            ParametricGFormula({"a": 1}, random_state=0, **{learner: given}), small_b()
        )
        estimate = result.estimates["ey_regimen[a]"]
        assert result.inference_status == "adaptive_baseline_plugin"
        assert estimate.inference == "adaptive_baseline_plugin"
        assert np.isfinite(estimate.plugin_std_error)
        assert result.simulation.parametric is False
        with pytest.raises(CapabilityError, match="F35"):
            _ = estimate.std_error

    def test_a_learner_the_estimate_does_not_read_leaves_the_status(self) -> None:
        from sklearn.linear_model import LogisticRegression

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ParametricGFormula(
                {"a": 1}, treatment_learner=LogisticRegression(), random_state=0
            ).fit(small_b(), **ROLES_B)
        assert result.inference_status == "influence_curve"

    def test_estimated_weights_take_the_estimated_weight_status(self) -> None:
        frame = panel_b(300, seed=5, weighted=True)
        result = quiet_fit(
            ParametricGFormula({"a": 1}, random_state=0),
            frame,
            **{**ROLES_B, "weights": "wt", "weights_estimated": True},
        )
        assert result.inference_status == "estimated_weight_plugin"

    def test_a_low_draw_count_warns_and_a_high_one_does_not(self) -> None:
        # The outcome mean is not drawn, so the simulation varies only through L2.  A strong
        # effect of L2 on Y makes one simulated history per unit a visible share of the interval.
        frame = small_b()
        seen = frame["Y"].notna()
        frame.loc[seen, "Y"] = (
            np.random.default_rng(1)
            .binomial(1, expit(-3.0 + 6.0 * frame.loc[seen, "L2"].to_numpy()))
            .astype(float)
        )
        with pytest.warns(MonteCarloWarning, match="simulation"):
            low = ParametricGFormula(
                {"a": 1, "b": 0}, n_draws=1, n_derivative_draws=2, random_state=0
            ).fit(frame, **ROLES_B)
        assert low.simulation.max_share > gf_threshold()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MonteCarloWarning)
            warnings.simplefilter("ignore", UserWarning)
            high = ParametricGFormula({"a": 1, "b": 0}, n_draws=400, random_state=0).fit(
                frame, **ROLES_B
            )
        assert high.simulation.max_share < gf_threshold()
        assert high.simulation.max_share < low.simulation.max_share

    def test_a_fit_repeats_under_one_seed_and_moves_under_another(self) -> None:
        frame = small_b()
        first = quiet_fit(ParametricGFormula({"a": 1}, n_draws=3, random_state=4), frame)
        again = quiet_fit(ParametricGFormula({"a": 1}, n_draws=3, random_state=4), frame)
        other = quiet_fit(ParametricGFormula({"a": 1}, n_draws=3, random_state=5), frame)
        name = "ey_regimen[a]"
        assert first.psi(name) == again.psi(name)
        np.testing.assert_array_equal(first.influence_curves[name], again.influence_curves[name])
        assert first.psi(name) != other.psi(name)
        assert first.simulation.seed != other.simulation.seed


def gf_threshold() -> float:
    from cleverly.longitudinal.estimator import MONTE_CARLO_WARN_SHARE

    return float(MONTE_CARLO_WARN_SHARE)


# ----------------------------------------------------------------------------------- W11: typing


def panel_cat(n: int, seed: int) -> pd.DataFrame:
    """Binary baseline, a three-level covariate with a non-monotone effect, and no censoring."""
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, n).astype(float)
    a1 = rng.binomial(1, expit(0.5 * w - 0.2)).astype(float)
    pick = np.exp(np.column_stack([np.zeros(n), 0.8 * a1 - 0.2, -0.6 * a1 + 0.4 * w]))
    pick = pick / pick.sum(axis=1, keepdims=True)
    l2 = (rng.uniform(size=n)[:, None] > np.cumsum(pick, axis=1)).sum(axis=1).astype(float)
    a2 = rng.binomial(1, expit(0.3 - 0.4 * (l2 == 1) + 0.5 * a1)).astype(float)
    # The effect of the level is not monotone: the middle level is the harmful one.
    eta = -0.3 + 0.5 * a1 + 0.6 * a2 + 1.2 * (l2 == 1) - 0.7 * (l2 == 2) * a1 + 0.4 * w
    y = rng.binomial(1, expit(eta)).astype(float)
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})


ROLES_CAT: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}


class TestCategoricalCovariate:
    def test_a_saturated_law_matches_the_nonparametric_g_formula(self) -> None:
        """A three-level covariate with a non-monotone effect: a numeric 0/1/2 column would fail."""
        frame = panel_cat(1500, seed=12)
        # The empirical g-formula for "always": the sum over W and L2 of the cell means.
        truth = 0.0
        for _, group in frame.groupby("W"):
            weight = len(group) / len(frame)
            treated = group[group["A1"] == 1]
            for _, cell in treated.groupby("L2"):
                follow = cell[cell["A2"] == 1]
                truth += weight * len(cell) / len(treated) * follow["Y"].mean()
        result = fit_with(
            frame,
            {"always": 1},
            ROLES_CAT,
            covariate_types={"L2": "categorical"},
            covariate_learner=SATURATED,
            outcome_learner=SATURATED,
            n_draws=100,
        )
        name = "ey_regimen[always]"
        assert abs(result.psi(name) - truth) <= 4.0 * monte_carlo_se(result, name) + 1e-9
        assert monte_carlo_se(result, name) > 1e-4
        # The nonzero witness: the g-formula moves the mean away from the treated units' raw mean.
        assert abs(truth - frame.loc[frame["A1"] == 1, "Y"].mean()) > 1e-3
        assert result.simulation.covariate_types == ((2, "L2", "categorical"),)


# ------------------------------------------------------------------------------------ the surface

STUDY_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}


class TestSurface:
    def test_the_one_call_form_equals_the_class(self) -> None:
        from cleverly.longitudinal import gformula

        frame = small_b()
        direct = quiet_fit(
            ParametricGFormula({"always": 1, "never": 0}, n_draws=5, random_state=2), frame
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call = gformula(
                frame,
                regimens={"always": 1, "never": 0},
                n_draws=5,
                random_state=2,
                **ROLES_B,
            )
        assert set(call.estimates) == set(direct.estimates)
        for name in direct.estimates:
            assert call.psi(name) == direct.psi(name)

    def test_the_study_method_is_the_direct_fit(self) -> None:
        frame = small_b()
        design = cleverly.LongitudinalTreatment(**STUDY_COLUMNS)
        effect = cleverly.CausalStudy(frame, design=design).identify(
            cleverly.RegimeContrast({"always": 1, "never": 0}, reference="never")
        )
        names = {record.name: record for record in effect.available_methods()}
        assert names["parametric_gformula"].available
        assert names["parametric_gformula"].reason is None
        method = cleverly.ParametricGFormulaMethod(
            n_draws=5, runtime=cleverly.Runtime(random_state=2)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow = effect.estimate(method=method)
            direct = ParametricGFormula(
                {"always": 1, "never": 0}, reference="never", n_draws=5, random_state=2
            ).fit(frame, **ROLES_B)
        assert workflow.fitted_method == "parametric_gformula"
        assert workflow.config.estimator == "parametric_gformula"
        assert workflow.estimates
        for name, workflow_estimate in workflow.estimates.items():
            assert workflow_estimate.psi == direct.estimates[name].psi
            assert workflow_estimate.std_error == direct.estimates[name].std_error

    def test_a_point_design_does_not_list_the_method(self) -> None:
        from cleverly.datasets import make_linear_ate

        frame, _ = make_linear_ate(n=120, seed=19)
        effect = cleverly.CausalStudy(
            frame,
            design=cleverly.PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
        ).identify(cleverly.ATE())
        row = {item.name: item for item in effect.available_methods()}["parametric_gformula"]
        assert not row.available
        assert row.refusal is not None

    def test_the_method_refuses_a_fluctuation_setting(self) -> None:
        method = cleverly.ParametricGFormulaMethod(targeting=cleverly.Targeting(max_iter=3))
        with pytest.raises(MethodConfigurationError, match="fit no fluctuation"):
            method.estimator_kwargs(longitudinal=True)

    def test_the_summary_names_the_estimator_and_its_simulation(self) -> None:
        text = quiet_fit(ParametricGFormula({"a": 1}, random_state=0), small_b()).summary()
        assert "parametric g-formula" in text
        assert "simulation:" in text
        assert "Philox" in text

    def test_save_and_load_round_trip(self, tmp_path: Any) -> None:
        result = quiet_fit(ParametricGFormula({"a": 1, "b": 0}, random_state=0), small_b())
        path = tmp_path / "gformula.cleverly"
        result.save(path)
        loaded = cleverly.load(path)
        assert loaded.psi("ey_regimen[a]") == result.psi("ey_regimen[a]")
        assert loaded.config.estimator == "parametric_gformula"
        assert loaded.simulation.seed == result.simulation.seed
        assert loaded.simulation.monte_carlo_share == result.simulation.monte_carlo_share

    def test_the_assessment_rows_say_what_a_simulation_fit_has(self) -> None:
        from cleverly.assessment import assessment_capabilities

        result = quiet_fit(ParametricGFormula({"a": 1}, random_state=0), small_b())
        rows = {row.operation: row for row in assessment_capabilities(result)}
        assert "parametric_gformula" in rows["support"].methods
        assert not rows["score_equations"].available
        assert not rows["truncation_curve"].available
