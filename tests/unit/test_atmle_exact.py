"""Exact-law checks of the adaptive TMLE engine (X90 plan, section 4).

Every check runs on a finite law F: ``W = (W1, W2)`` in ``{0, 1, 2} x {0, 1}``, trial
membership ``S`` that depends on ``W``, a randomized trial arm and an external arm share that
depends on ``W``, and a two-point outcome given ``(S, W, A)`` whose trial effect and external
bias are not in the span of the declared dictionary ``phi = [1, W1, W2]``,
``psi = [1, A, W1, W2, A W1]``. Optionally the outcome is missing with a probability that
depends on ``(S, W, A)``.

The engine's own :func:`~cleverly.estimators.atmle.refit_unit` and
:func:`~cleverly.estimators.atmle.evaluate` run with the law's masses as the measure, so the
projection parameter ``Psi_w(P)`` and its influence function ``D_w(P)`` below are the code
under test, evaluated exactly. The law's nuisances are computed here from the masses, without
the engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.special import expit

from cleverly.estimators import atmle
from cleverly.estimators._working_basis import bias_columns
from cleverly.estimators.atmle import (
    Nuisances,
    WorkingUnit,
    bias_term,
    evaluate,
    refit_unit,
)
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import mean_submodel
from cleverly.inference.influence import counterfactual_means

W_CELLS = [(w1, w2) for w1 in (0.0, 1.0, 2.0) for w2 in (0.0, 1.0)]
W_MASS = np.array([0.20, 0.12, 0.18, 0.15, 0.17, 0.18])


def _trial_share(w1: float, w2: float) -> float:
    return float(expit(-0.2 + 0.6 * w1 - 0.5 * w2))


def _external_arm(w1: float, w2: float) -> float:
    return float(expit(-0.8 + 0.7 * w1 + 0.4 * w2))


def _mean(s: float, w1: float, w2: float, a: float) -> float:
    effect = 1.2 + 0.4 * w1 - 0.5 * w1 * w2 + 0.3 * w1**2
    bias = 0.4 + 0.6 * w1 * (1 - a) - 0.7 * a * w2 + 0.3 * w1**2 * w2
    return 1.0 + 0.5 * w1 - 0.3 * w2 + a * effect + (1.0 - s) * bias


def _observation(s: float, w1: float, w2: float, a: float) -> float:
    return float(expit(1.0 - 0.4 * w1 + 0.5 * a - 0.6 * (1.0 - s) + 0.3 * w2))


@dataclass(frozen=True)
class Law:
    """A finite law: one row per support point, with its mass."""

    w: np.ndarray
    s: np.ndarray
    a: np.ndarray
    y: np.ndarray
    observed: np.ndarray | None
    mass: np.ndarray
    controls_only: bool

    def key(self, *names: str) -> np.ndarray:
        """Integer codes of the cells defined by ``names``."""
        columns = {
            "w": self.w[:, 0] * 2 + self.w[:, 1],
            "s": self.s,
            "a": self.a,
            "d": np.zeros_like(self.s) if self.observed is None else self.observed.astype(float),
        }
        code = np.zeros(self.s.size)
        for name in names:
            code = code * 10 + columns[name]
        return np.unique(code, return_inverse=True)[1]

    def with_mass(self, mass: np.ndarray) -> Law:
        return replace(self, mass=mass)


def finite_law(*, missing: bool = False, controls_only: bool = False) -> Law:
    """The law F, with optional missing outcomes and optional external controls only."""
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    for (w1, w2), pw in zip(W_CELLS, W_MASS, strict=True):
        share = _trial_share(w1, w2)
        for s, ps in ((1.0, share), (0.0, 1.0 - share)):
            arm = 0.5 if s == 1.0 else (0.0 if controls_only else _external_arm(w1, w2))
            for a, pa in ((1.0, arm), (0.0, 1.0 - arm)):
                if pa == 0.0:
                    continue
                mu = _mean(s, w1, w2, a)
                base = pw * ps * pa
                if missing:
                    seen = _observation(s, w1, w2, a)
                    rows.append((w1, w2, s, a, 1.0, mu + 1.0, base * seen * 0.5))
                    rows.append((w1, w2, s, a, 1.0, mu - 1.0, base * seen * 0.5))
                    rows.append((w1, w2, s, a, 0.0, 0.0, base * (1.0 - seen)))
                else:
                    rows.append((w1, w2, s, a, 1.0, mu + 1.0, base * 0.5))
                    rows.append((w1, w2, s, a, 1.0, mu - 1.0, base * 0.5))
    table = np.array(rows)
    return Law(
        w=table[:, :2],
        s=table[:, 2],
        a=table[:, 3],
        y=table[:, 5],
        observed=table[:, 4].astype(bool) if missing else None,
        mass=table[:, 6] / table[:, 6].sum(),
        controls_only=controls_only,
    )


def _group_mean(values: np.ndarray, mass: np.ndarray, key: np.ndarray) -> np.ndarray:
    """``E(values | key)`` at each support point."""
    total = np.bincount(key, weights=mass)
    weighted = np.bincount(key, weights=mass * values)
    return (weighted / total)[key]


def _cell_values(
    law: Law, values: np.ndarray, at_key: np.ndarray, from_key: np.ndarray
) -> np.ndarray:
    """Look up a per-cell quantity, known on the points of ``from_key``, at ``at_key`` cells."""
    table = {int(k): float(v) for k, v in zip(from_key, values, strict=True)}
    return np.array([table[int(k)] for k in at_key])


@dataclass(frozen=True)
class Truth:
    """The law's nuisances at each point, and its regressions at every ``(s, a)``."""

    nuisances: Nuisances
    q: dict[tuple[float, float], np.ndarray]


def law_truth(law: Law) -> Truth:
    """Every nuisance of the law at each support point, from the masses alone."""
    m = law.mass
    wkey = law.key("w")
    g = _group_mean(law.a, m, wkey)
    pibar = _group_mean(law.s, m, wkey)
    source = {}
    for arm in (0.0, 1.0):
        rows = law.a == arm
        if not np.any(rows & (law.s == 0.0)):
            source[arm] = np.ones(law.s.size)
            continue
        sub = _group_mean(law.s[rows], m[rows], wkey[rows])
        source[arm] = _cell_values(law, sub, wkey, wkey[rows])
    seen = np.ones(law.s.size, bool) if law.observed is None else law.observed
    swa = law.key("w", "s", "a")
    q_own = _cell_values(law, _group_mean(law.y[seen], m[seen], swa[seen]), swa, swa[seen])
    q: dict[tuple[float, float], np.ndarray] = {}
    for member in (0.0, 1.0):
        for arm in (0.0, 1.0):
            cell = (law.s == member) & (law.a == arm)
            values = np.array([_mean(member, w1, w2, arm) for w1, w2 in law.w])
            if np.any(cell):
                values = _cell_values(law, q_own[cell], wkey, wkey[cell])
            q[(member, arm)] = values
    qt = {
        arm: source[arm] * q[(1.0, arm)] + (1.0 - source[arm]) * q[(0.0, arm)] for arm in (0.0, 1.0)
    }
    if law.observed is None:
        theta = _group_mean(law.y, m, wkey)
        qt_own = _group_mean(law.y, m, law.key("w", "a"))
        np.testing.assert_allclose(theta, g * qt[1.0] + (1 - g) * qt[0.0], rtol=1e-12)
        np.testing.assert_allclose(qt_own, np.where(law.a == 1, qt[1.0], qt[0.0]), rtol=1e-12)
        nuisances = Nuisances(g, theta, source[0.0], source[1.0], qt_own, pibar)
    else:
        g_delta = _group_mean(law.observed.astype(float), m, swa)
        nuisances = Nuisances(
            g,
            g * qt[1.0] + (1 - g) * qt[0.0],
            source[0.0],
            source[1.0],
            np.where(law.a == 1, qt[1.0], qt[0.0]),
            pibar,
            q_own,
            g_delta,
        )
    return Truth(nuisances, q)


def main_terms(w: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(w.shape[0]), w])


def cell_indicators(w: np.ndarray) -> np.ndarray:
    cells = np.array(W_CELLS)
    return np.column_stack(
        [np.ones(w.shape[0])] + [np.all(w == cell, axis=1).astype(float) for cell in cells[1:]]
    )


def intercept_only(w: np.ndarray) -> np.ndarray:
    return np.ones((w.shape[0], 1))


def intercept_and_arm(phi: np.ndarray, a: np.ndarray | float, controls_only: bool) -> np.ndarray:
    arm = np.broadcast_to(np.asarray(a, float), (phi.shape[0],))
    return phi if controls_only else np.column_stack([phi[:, 0], arm])


def declared_bias(phi: np.ndarray, a: np.ndarray | float, controls_only: bool) -> np.ndarray:
    """The plan's bias dictionary for F: ``[1, A, W1, W2, A W1]``, or ``phi`` for controls."""
    arm = np.broadcast_to(np.asarray(a, float), (phi.shape[0],))
    if controls_only:
        return phi
    return np.column_stack([phi[:, 0], arm, phi[:, 1], phi[:, 2], arm * phi[:, 1]])


Dictionary = Callable[[np.ndarray, np.ndarray | float, bool], np.ndarray]


def law_unit(
    law: Law,
    nuisances: Nuisances | None = None,
    *,
    effect: Callable[[np.ndarray], np.ndarray] = main_terms,
    bias: Dictionary = declared_bias,
) -> WorkingUnit:
    phi = effect(law.w)
    return WorkingUnit(
        outcome=law.y,
        treatment=law.a,
        source=law.s,
        observed=law.observed,
        phi=phi,
        psi=bias(phi, law.a, law.controls_only),
        psi0=bias(phi, 0.0, law.controls_only),
        psi1=bias(phi, 1.0, law.controls_only),
        nuisances=law_truth(law).nuisances if nuisances is None else nuisances,
        weights=law.mass,
        controls_only=law.controls_only,
    )


def projection(
    law: Law, population: atmle.Population, **dictionary: object
) -> tuple[float, atmle.InfluenceParts, np.ndarray, np.ndarray]:
    """``Psi_w(P)``, the parts of ``D_w(P)``, and the projection coefficients, under the law."""
    unit = law_unit(law, **dictionary)  # type: ignore[arg-type]
    effect_fit, bias_fit = refit_unit(unit)
    assert effect_fit is not None
    estimate, parts = evaluate(unit, effect_fit.beta, bias_fit.beta, population)
    return estimate, parts, effect_fit.beta, bias_fit.beta


def trial_effects(law: Law) -> tuple[float, float]:
    """``Psi`` and ``Psi_2`` of the law, from its regressions."""
    q = law_truth(law).q
    effect = q[(1.0, 1.0)] - q[(1.0, 0.0)]
    p = float(np.sum(law.mass * law.s))
    return float(np.sum(law.mass * effect)), float(np.sum(law.mass * law.s * effect)) / p


LAWS = {
    "complete": {},
    "missing": {"missing": True},
    "controls": {"controls_only": True},
    "controls-missing": {"missing": True, "controls_only": True},
}
POPULATIONS = ("pooled", "trial")


# --------------------------------------------------------------------------- Lemma 2


def _decomposition(law: Law) -> tuple[float, np.ndarray, np.ndarray, Truth]:
    truth = law_truth(law)
    nu, q = truth.nuisances, truth.q
    qt1 = nu.source1 * q[(1.0, 1.0)] + (1 - nu.source1) * q[(0.0, 1.0)]
    qt0 = nu.source0 * q[(1.0, 0.0)] + (1 - nu.source0) * q[(0.0, 0.0)]
    pooled = float(np.sum(law.mass * (qt1 - qt0)))
    return pooled, q[(1.0, 0.0)] - q[(0.0, 0.0)], q[(1.0, 1.0)] - q[(0.0, 1.0)], truth


def test_lemma2_decomposes_the_trial_effect_into_the_pooled_effect_minus_the_bias() -> None:
    law = finite_law()
    psi, psi2 = trial_effects(law)
    pooled, bias0, bias1, truth = _decomposition(law)
    nu = truth.nuisances
    sharp = bias_term(nu.source0, nu.source1, bias0, bias1)
    bias = float(np.sum(law.mass * sharp))
    assert abs(bias) > 0.1
    np.testing.assert_allclose(pooled - bias, psi, rtol=1e-12)
    p = float(np.sum(law.mass * law.s))
    q = truth.q
    tau_a = (nu.source1 * q[(1.0, 1.0)] + (1 - nu.source1) * q[(0.0, 1.0)]) - (
        nu.source0 * q[(1.0, 0.0)] + (1 - nu.source0) * q[(0.0, 0.0)]
    )
    np.testing.assert_allclose(np.sum(law.mass * law.s * (tau_a - sharp)) / p, psi2, rtol=1e-12)
    # Mutations: the sign of Psi# flipped, and Pi(0 | W, 0) swapped with Pi(0 | W, 1). The
    # swap is visible only because A depends on S given W in F.
    assert np.max(np.abs(nu.source0 - nu.source1)) > 0.05
    assert abs(pooled + bias - psi) > 1e-3
    swapped = float(np.sum(law.mass * bias_term(nu.source1, nu.source0, bias0, bias1)))
    assert abs(pooled - swapped - psi) > 1e-3


def test_controls_only_drops_the_external_treated_bias() -> None:
    law = finite_law(controls_only=True)
    psi, _ = trial_effects(law)
    pooled, bias0, _, truth = _decomposition(law)
    nu = truth.nuisances
    np.testing.assert_array_equal(nu.source1, 1.0)
    for unidentified in (np.zeros(law.s.size), np.linspace(-3.0, 5.0, law.s.size)):
        sharp = bias_term(nu.source0, nu.source1, bias0, unidentified)
        np.testing.assert_allclose(pooled - np.sum(law.mass * sharp), psi, rtol=1e-12)
    # Mutation: the both-arms formula on a controls-only fit, which evaluates the control
    # rows' source model at A = 1 and keeps a bias at A = 1 (the bias model without A).
    kept = bias_term(nu.source0, nu.source0, bias0, bias0)
    assert abs(pooled - np.sum(law.mass * kept) - psi) > 1e-3


# --------------------------------------------------------------------------- the projection


@pytest.mark.parametrize("law_name", list(LAWS))
def test_a_saturated_dictionary_projects_onto_the_trial_effect(law_name: str) -> None:
    law = finite_law(**LAWS[law_name])
    psi, psi2 = trial_effects(law)
    saturated = {"effect": cell_indicators, "bias": bias_columns}
    np.testing.assert_allclose(projection(law, "pooled", **saturated)[0], psi, rtol=1e-11)
    np.testing.assert_allclose(projection(law, "trial", **saturated)[0], psi2, rtol=1e-11)
    # Witness: a dictionary too small for F misses both, so the check can fail.
    small = {"effect": intercept_only, "bias": intercept_and_arm}
    assert abs(projection(law, "pooled", **small)[0] - psi) > 1e-3
    assert abs(projection(law, "trial", **small)[0] - psi2) > 1e-3


# --------------------------------------------------------------------------- Gateaux


def _center(law: Law, values: np.ndarray, *names: str) -> np.ndarray:
    """``values - E(values | names)``; the overall mean when no name is given."""
    key = law.key(*names) if names else np.zeros(law.s.size, dtype=np.int64)
    return values - _group_mean(values, law.mass, key)


def directions(law: Law) -> dict[str, np.ndarray]:
    """Score directions, one per factor of the law, and a generic one."""
    rng = np.random.default_rng(7)

    def by(*names: str) -> np.ndarray:
        key = law.key(*names)
        return rng.normal(size=int(key.max()) + 1)[key]

    out = {
        "Y given (S, W, A)": _center(law, rng.normal(size=law.s.size), "w", "s", "a", "d"),
        "S given (W, A)": _center(law, by("w", "s", "a"), "w", "a"),
        "A given W": _center(law, by("w", "a"), "w"),
        "W": _center(law, by("w")),
        "generic": _center(law, rng.normal(size=law.s.size)),
    }
    if law.observed is not None:
        out["Delta given (S, W, A)"] = _center(law, by("w", "s", "a", "d"), "w", "s", "a")
    return out


def _derivative(law: Law, h: np.ndarray, population: atmle.Population) -> float:
    """The five-point derivative of ``Psi_w`` along ``P_e = P (1 + e h)`` at 0."""
    step = 1e-4

    def at(e: float) -> float:
        return projection(law.with_mass(law.mass * (1.0 + e * h)), population)[0]

    return (-at(2 * step) + 8 * at(step) - 8 * at(-step) + at(-2 * step)) / (12 * step)


GATEAUX = [(name, population) for name in LAWS for population in POPULATIONS]


@pytest.mark.parametrize(("law_name", "population"), GATEAUX)
def test_the_influence_function_is_the_gateaux_derivative(law_name: str, population: str) -> None:
    law = finite_law(**LAWS[law_name])
    _, parts, _, _ = projection(law, population)  # type: ignore[arg-type]
    for name, h in directions(law).items():
        derivative = _derivative(law, h, population)  # type: ignore[arg-type]
        expected = float(np.sum(law.mass * parts.total * h))
        assert abs(derivative - expected) <= 1e-8 * max(1.0, abs(expected)), name


@pytest.mark.parametrize(("law_name", "population"), GATEAUX)
def test_each_influence_component_is_needed(law_name: str, population: str) -> None:
    """Witness and mutation: every component is nonzero, and dropping it breaks the identity."""
    law = finite_law(**LAWS[law_name])
    _, parts, _, _ = projection(law, population)  # type: ignore[arg-type]
    paths = directions(law)
    derivatives = {name: _derivative(law, h, population) for name, h in paths.items()}  # type: ignore[arg-type]
    components = {
        "m_A'U_A": parts.effect_model,
        "m_S'U_S": parts.bias_model,
        "(S - Pi) C": parts.source,
    }
    for label, component in components.items():
        assert float(np.sum(law.mass * component**2)) > 1e-4, label
        gaps = [
            abs(float(np.sum(law.mass * (parts.total - component) * h)) - derivatives[name])
            for name, h in paths.items()
        ]
        assert max(gaps) > 1e-4, label
    if population == "trial":
        # Mutation: the pi(W) / p factor dropped from the source term. S depends on W in F.
        nu = law_truth(law).nuisances
        p = float(np.sum(law.mass * law.s))
        assert np.ptp(nu.pibar) > 0.1
        mutated = parts.total - parts.source + parts.source * p / nu.pibar
        gaps = [
            abs(float(np.sum(law.mass * mutated * h)) - derivatives[name])
            for name, h in paths.items()
        ]
        assert max(gaps) > 1e-4


# --------------------------------------------------------------------------- remainder


def _perturbed(law: Law, truth: Truth, t: float, moves: dict[str, np.ndarray]) -> Nuisances:
    """The law's nuisances moved by ``t`` along ``moves``; built outcome means follow."""
    nu = truth.nuisances

    def moved(name: str, value: np.ndarray) -> np.ndarray:
        return value + t * moves[name] if name in moves else value

    g = moved("g", nu.g)
    source0 = moved("source0", nu.source0)
    source1 = nu.source1 if law.controls_only else moved("source1", nu.source1)
    pibar = moved("pibar", nu.pibar)
    if law.observed is None:
        return Nuisances(g, moved("theta", nu.theta), source0, source1, moved("qt", nu.qt), pibar)
    q = {key: moved(f"q{key}", value) for key, value in truth.q.items()}
    qt1 = source1 * q[(1.0, 1.0)] + (1 - source1) * q[(0.0, 1.0)]
    qt0 = source0 * q[(1.0, 0.0)] + (1 - source0) * q[(0.0, 0.0)]
    own = {(s, a): (law.s == s) & (law.a == a) for s, a in q}
    q_own = sum(np.where(cell, q[key], 0.0) for key, cell in own.items())
    return Nuisances(
        g,
        g * qt1 + (1 - g) * qt0,
        source0,
        source1,
        np.where(law.a == 1, qt1, qt0),
        pibar,
        np.asarray(q_own, dtype=float),
        moved("g_delta", nu.g_delta),  # type: ignore[arg-type]
    )


def remainder_slope(
    law: Law, population: atmle.Population, nuisances: tuple[str, ...]
) -> tuple[float, float]:
    """The slope of ``log |R(t)|`` on ``log t``, and ``|R|`` at the smallest ``t``.

    ``R(t) = Psi_w(P_t) - Psi_w(P0) + P0 D_w(P_t)``, where ``P_t`` moves the named nuisances
    and both working-model coefficient vectors by ``t`` along fixed random directions, and
    keeps ``P0``'s covariate law.
    """
    truth = law_truth(law)
    unit0 = law_unit(law, truth.nuisances)
    effect_fit, bias_fit = refit_unit(unit0)
    assert effect_fit is not None
    beta_a, beta_s = effect_fit.beta, bias_fit.beta
    base = evaluate(unit0, beta_a, beta_s, population)[0]
    rng = np.random.default_rng(11)
    wkey, wa, swa = law.key("w"), law.key("w", "a"), law.key("w", "s", "a")
    shapes = {
        "g": wkey,
        "theta": wkey,
        "source0": wkey,
        "source1": wkey,
        "pibar": wkey,
        "qt": wa,
        "g_delta": swa,
    }
    moves: dict[str, np.ndarray] = {}
    for name in nuisances:
        if name == "q":
            for key in truth.q:
                moves[f"q{key}"] = rng.normal(size=6)[wkey]
            continue
        if name == "source":
            continue
        key = shapes[name]
        moves[name] = 0.5 * rng.uniform(-1, 1, size=int(key.max()) + 1)[key]
    if "source" in nuisances:
        moves["source0"] = 0.5 * rng.uniform(-1, 1, size=6)[wkey]
        moves["source1"] = 0.5 * rng.uniform(-1, 1, size=6)[wkey]
    move_a = rng.normal(size=beta_a.size)
    move_s = rng.normal(size=beta_s.size)
    ts = 1e-3 * 2.0 ** -np.arange(5)
    values = []
    for t in ts:
        unit = replace(unit0, nuisances=_perturbed(law, truth, float(t), moves))
        estimate, parts = evaluate(unit, beta_a + t * move_a, beta_s + t * move_s, population)
        values.append(estimate - base + float(np.sum(law.mass * parts.total)))
    logs = np.log(np.abs(values))
    slope = float(np.polyfit(np.log(ts), logs, 1)[0])
    return slope, float(abs(values[-1]))


REMAINDER_PAIRS = [
    ("complete", "pooled", ("g", "theta")),
    ("complete", "pooled", ("source", "qt")),
    ("complete", "pooled", ("g", "source")),
    ("complete", "trial", ("g", "theta")),
    ("complete", "trial", ("source", "qt")),
    ("complete", "trial", ("g", "source")),
    ("complete", "trial", ("source", "pibar")),
    ("complete", "pooled", ("g", "theta", "source", "qt", "pibar")),
    ("missing", "pooled", ("g_delta", "q")),
    ("missing", "trial", ("g_delta", "q")),
    ("missing", "pooled", ("g", "source", "g_delta", "q")),
    ("controls", "pooled", ("g", "source")),
    ("controls", "trial", ("source", "qt", "pibar")),
]


@pytest.mark.parametrize(("law_name", "population", "nuisances"), REMAINDER_PAIRS)
def test_the_remainder_is_second_order(
    law_name: str, population: str, nuisances: tuple[str, ...]
) -> None:
    law = finite_law(**LAWS[law_name])
    slope, size = remainder_slope(law, population, nuisances)  # type: ignore[arg-type]
    assert size > 1e-12
    assert 1.85 < slope < 2.15


@pytest.mark.parametrize("population", POPULATIONS)
def test_a_trial_treatment_mechanism_in_the_clever_covariate_leaves_a_first_order_remainder(
    population: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: ``g(a | S = 1, W)`` in place of the pooled ``g(a | W)`` in ``C``."""
    law = finite_law()
    truth = law_truth(law)
    trial_g = np.full(law.s.size, 0.5)
    assert np.max(np.abs(truth.nuisances.g - trial_g)) > 0.05
    original = atmle.source_covariates
    pooled_g = truth.nuisances.g

    def mutated(g, bias0, bias1, controls_only):  # type: ignore[no-untyped-def]
        return original(trial_g + (g - pooled_g), bias0, bias1, controls_only)

    monkeypatch.setattr(atmle, "source_covariates", mutated)
    slope, _ = remainder_slope(law, population, ("g", "source"))  # type: ignore[arg-type]
    assert 0.85 < slope < 1.15


@pytest.mark.parametrize("population", POPULATIONS)
def test_an_inverse_weighted_residual_leaves_a_first_order_remainder(
    population: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: App B's plain IPCW residuals ``Delta / g_delta r`` in place of the augmented."""
    law = finite_law(missing=True)

    def ipcw(unit: WorkingUnit, beta_a: np.ndarray, beta_s: np.ndarray):  # type: ignore[no-untyped-def]
        nu = unit.nuisances
        seen = unit.observed
        y = np.where(seen, unit.outcome, 0.0)
        weight = np.where(seen, 1.0 / nu.g_delta, 0.0)
        source = nu.source_at(unit.treatment)
        return (
            weight * (y - nu.theta - (unit.treatment - nu.g) * (unit.phi @ beta_a)),
            weight * (y - nu.qt - (unit.source - source) * (unit.psi @ beta_s)),
        )

    monkeypatch.setattr(atmle, "residuals", ipcw)
    slope, _ = remainder_slope(law, population, ("g_delta", "q"))  # type: ignore[arg-type]
    assert 0.85 < slope < 1.15


# --------------------------------------------------------------------------- Lemma 1


def test_lemma1_regular_tmle_equals_the_shipped_missing_outcome_influence_curve() -> None:
    """With ``Delta = S``, the shipped missing-outcome curve is Lemma 1's ``D_Psi``."""
    law = finite_law()
    truth = law_truth(law)
    nu, q = truth.nuisances, truth.q
    psi, _ = trial_effects(law)
    mechanism = np.column_stack([1 - nu.g, nu.g])
    observation = np.column_stack([nu.source0, nu.source1])
    fit = InitialFit(
        observed=np.where(law.a == 1, q[(1.0, 1.0)], q[(1.0, 0.0)]),
        arms={0.0: q[(1.0, 0.0)], 1.0: q[(1.0, 1.0)]},
    )

    def shipped(missingness: np.ndarray) -> np.ndarray:
        submodel = mean_submodel(law.a, mechanism, missingness=missingness)
        arms = counterfactual_means(law.y, fit, submodel, law.mass, observed=law.s == 1.0)
        return (arms[1.0].influence_curve - arms[0.0].influence_curve) / law.mass

    trial_arm = np.where(law.a == 1, 0.5, 0.5)
    q_own = np.where(law.a == 1, q[(1.0, 1.0)], q[(1.0, 0.0)])
    q_own = np.where(law.s == 1, q_own, np.where(law.a == 1, q[(0.0, 1.0)], q[(0.0, 0.0)]))
    lemma1 = (
        q[(1.0, 1.0)]
        - q[(1.0, 0.0)]
        - psi
        + law.s / nu.pibar * (2 * law.a - 1) / trial_arm * (law.y - q_own)
    )
    np.testing.assert_allclose(shipped(observation), lemma1, rtol=1e-12, atol=1e-12)
    # Mutation: P(S = 1 | W) in place of P(S = 1 | W, A). The identity rests on
    # g(A | W) P(S = 1 | A, W) = pi(W) g(A | S = 1, W), which the mutation breaks.
    wrong = np.column_stack([nu.pibar, nu.pibar])
    assert np.max(np.abs(shipped(wrong) - lemma1)) > 1e-3


# --------------------------------------------------------------------------- the fluctuation


@pytest.mark.parametrize(("law_name", "population"), GATEAUX)
def test_the_fluctuation_solves_the_source_term_of_the_influence_function(
    law_name: str, population: str
) -> None:
    """After one step from a wrong ``Pi``, ``P (f (S - Pi*) C) = 0`` for the trial's factor too."""
    law = finite_law(**LAWS[law_name])
    truth = law_truth(law)
    nu = truth.nuisances
    wrong = replace(
        nu,
        source0=np.clip(nu.source0 * 0.8 + 0.05, 0.01, 0.99),
        source1=nu.source1 if law.controls_only else np.clip(nu.source1 * 1.1, 0.01, 0.99),
    )
    unit = law_unit(law, wrong)
    effect_fit, bias_fit = refit_unit(unit)
    assert effect_fit is not None
    _, before = evaluate(unit, effect_fit.beta, bias_fit.beta, population)  # type: ignore[arg-type]
    k0, k1, covariate, rows = atmle._source_step(unit, bias_fit.beta, population)  # type: ignore[arg-type]
    assert abs(float(np.sum(law.mass * before.source))) > 1e-4
    epsilon = atmle.target_source(
        unit.nuisances.source_at(law.a)[rows], covariate[rows], law.s[rows], law.mass[rows]
    )
    moved = atmle._fluctuated(unit, epsilon, k0, k1)
    _, after = evaluate(moved, effect_fit.beta, bias_fit.beta, population)  # type: ignore[arg-type]
    scale = float(np.sum(law.mass * np.abs(after.source)))
    assert abs(float(np.sum(law.mass * after.source))) <= 1e-10 * scale
    if law.controls_only:
        np.testing.assert_array_equal(moved.nuisances.source1, 1.0)
