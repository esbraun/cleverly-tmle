"""Population machinery of the ``adaptive-iptw`` design test (``X21-plan.md`` 3.5, items 1-8).

The test fits no estimator.  It builds each law as a table of cells with exact probabilities
and computes, for each key, the population fixed point of the targeting loop at the declared
learner's limit.  A :class:`Term` is one indicator, one target and one mechanism: the unit the
loop acts on.  A key is a linear (or, for ``paf``, a smooth) function of terms.

For a term with indicator :math:`I`, target :math:`T`, true mechanism :math:`g_0(x)` and learner
limit :math:`\\tilde g(x)`, the loop is the estimator's own.  Round ``k`` regresses ``T`` on a
polynomial of ``logit g`` among ``I = 1`` (the declared basis, capped at the distinct count minus
one, which is the saturated regression), forms :math:`H = Q_r / g`, and moves ``g`` along ``H``
to the root of :math:`E[H (I - g)] = 0`.  The loop stops at a score below ``1e-13``.  The
fixed point is :math:`\\psi^* = E[I T / g^*]` and the bias is :math:`\\psi^* - \\psi_0`.  A
learner whose limit is :math:`g_0` has score zero at the start, so its bias is exactly zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm
from scipy.stats import t as student

# ------------------------------------------------------------------------------ pass probabilities

MARGIN = 0.25
CONFIDENCE = 0.99


def bias_pass_probability(b: float, replicates: int, *, margin: float = MARGIN) -> float:
    """The chance a positive row's 99% standardized-bias interval lies inside the margin.

    The standardized bias estimate is normal about ``b`` with variance ``1 / R``, and the
    Student interval adds ``t / sqrt(R)``.
    """
    half = float(student.ppf(0.5 + CONFIDENCE / 2.0, replicates - 1)) / np.sqrt(replicates)
    limit = margin - half
    scale = 1.0 / np.sqrt(replicates)
    return float(norm.cdf((limit - b) / scale) - norm.cdf((-limit - b) / scale))


def control_fail_probability(b: float, replicates: int, *, margin: float = MARGIN) -> float:
    """The chance a control's 99% interval lies wholly beyond the margin on the side of ``b``."""
    half = float(student.ppf(0.5 + CONFIDENCE / 2.0, replicates - 1)) / np.sqrt(replicates)
    scale = 1.0 / np.sqrt(replicates)
    return float(norm.cdf((abs(b) - margin - half) / scale))


def spread_exclusion_probability(ratio: float, replicates: int) -> float:
    """The chance the 99% interval of an SE ratio with true value ``ratio`` excludes 1."""
    scale = 1.0 / np.sqrt(2.0 * replicates)
    z = float(norm.ppf(0.5 + CONFIDENCE / 2.0))
    return float(norm.cdf((ratio - 1.0 - z * scale) / scale))


def displacement_pass_probability(
    displacement: float, paired_variance: float, replicates: int, *, floor: float = 1.0
) -> float:
    """The chance a paired displacement estimate reaches ``floor``.

    ``displacement`` is the exact displacement in units of the targeted arm's SD, and
    ``paired_variance`` is the variance of the per-replicate difference of the two arms in the
    same units squared.  The estimate is the difference of means over the targeted arm's
    empirical SD, so its variance is ``V / R + D^2 / (2 R)``.
    """
    scale = np.sqrt((paired_variance + displacement**2 / 2.0) / replicates)
    return float(norm.cdf((displacement - floor) / scale))


# ----------------------------------------------------------------------------------- population


@dataclass
class Term:
    """One indicator, one target and one mechanism of a population.

    Parameters
    ----------
    name : str
        The term name.
    point : numpy.ndarray
        The covariate point of each cell, an integer index into the term's points.
    ind : numpy.ndarray
        The indicator ``I`` of each cell.
    target : numpy.ndarray
        The target ``T`` of each cell; read where ``I = 1``.
    limit : numpy.ndarray
        The learner's limit of ``P(I = 1 | point)``.
    degree : int
        The declared degree of the basis in ``logit g``.
    binary : bool
        Whether the regression is logistic (a 0/1 target) or least squares.
    dlogit : numpy.ndarray or None
        The derivative of ``logit g`` at each point with respect to the parameters of a
        parametric mechanism model, for the projected SD.
    exact_qr : numpy.ndarray or None
        The exact ``Qr_0`` at each point, for a law whose level sets of ``g_0`` the grid does not
        resolve (laws VL1 and VL0).  ``None`` groups the points by the value of ``g_0``.
    """

    name: str
    point: np.ndarray
    ind: np.ndarray
    target: np.ndarray
    limit: np.ndarray
    degree: int
    binary: bool = True
    dlogit: np.ndarray | None = None
    exact_qr: np.ndarray | None = None


@dataclass
class Key:
    """A reported key: coefficients on terms, or the attributable fraction of two terms."""

    coefficients: Mapping[str, float]
    paf: tuple[str, str] | None = None

    def value(self, psi: Mapping[str, float]) -> float:
        if self.paf is not None:
            observed, zero = self.paf
            return float(1.0 - psi[zero] / psi[observed])
        return float(sum(c * psi[name] for name, c in self.coefficients.items()))

    def gradient(self, psi: Mapping[str, float]) -> dict[str, float]:
        if self.paf is not None:
            observed, zero = self.paf
            return {observed: psi[zero] / psi[observed] ** 2, zero: -1.0 / psi[observed]}
        return dict(self.coefficients)


@dataclass
class Population:
    """A law as a table of cells, with its terms and keys.

    Parameters
    ----------
    prob : numpy.ndarray
        The probability of each cell.
    terms : dict
        The terms.
    keys : dict
        The reported keys.
    score : numpy.ndarray or None
        The score of a parametric treatment model at each cell, ``(cells, k)``, for the
        projected SD.
    """

    prob: np.ndarray
    terms: dict[str, Term]
    keys: dict[str, Key]
    score: np.ndarray | None = None
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- masses ----------------------------------------------------------------------------

    def masses(self, name: str) -> dict[str, np.ndarray]:
        """``pw``, the true ``g``, the conditional mean ``q`` and the cell mass of ``I = 1``."""
        term = self.terms[name]
        points = int(term.point.max()) + 1
        pw = np.bincount(term.point, weights=self.prob, minlength=points)
        weight = self.prob * term.ind
        mass = np.bincount(term.point, weights=weight, minlength=points)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(pw > 0, mass / pw, 0.0)
            q = np.where(
                mass > 0,
                np.bincount(term.point, weights=weight * term.target, minlength=points) / mass,
                0.0,
            )
        return {"pw": pw, "g": g, "q": q, "mass": mass}

    def truth(self, name: str) -> float:
        """``psi_0 = E[I T / g_0]`` of a term."""
        m = self.masses(name)
        return float(np.sum(m["pw"] * m["q"]))

    # ---- regression and loop ---------------------------------------------------------------

    def regress(self, name: str, g_current: np.ndarray) -> np.ndarray:
        """The declared-basis regression of ``T`` on ``logit g_current`` among ``I = 1``."""
        term = self.terms[name]
        m = self.masses(name)
        return _regression(
            logit(np.clip(g_current, 1e-12, 1.0 - 1e-12)),
            m["q"],
            m["mass"],
            term.degree,
            term.binary,
        )

    def fixed_point(
        self, name: str, *, tolerance: float = 1e-13, max_rounds: int = 200
    ) -> dict[str, Any]:
        """The population targeting loop of one term at its learner's limit."""
        term = self.terms[name]
        m = self.masses(name)
        pw, g_true, q = m["pw"], m["g"], m["q"]
        g = np.clip(np.asarray(term.limit, dtype=float), 1e-12, 1.0 - 1e-12)
        qr = self.regress(name, g)
        rounds = 0
        score = 0.0
        for step_count in range(max_rounds + 1):
            rounds = step_count
            h = qr / g
            score = float(np.sum(pw * h * (g_true - g)))
            if abs(score) < tolerance:
                break
            eps = 0.0
            for _ in range(100):
                tilted = expit(logit(g) + eps * h)
                f = float(np.sum(pw * h * (g_true - tilted)))
                df = -float(np.sum(pw * h * h * tilted * (1.0 - tilted)))
                eps -= f / df
                if abs(f) < 1e-15:
                    break
            g = expit(logit(g) + eps * h)
            qr = self.regress(name, g)
        else:  # pragma: no cover - a loop that does not converge is a design failure
            raise AssertionError(f"{name}: the population loop did not converge ({score})")
        psi = float(np.sum(pw * g_true * q / g))
        untargeted = float(np.sum(pw * g_true * q / np.clip(term.limit, 1e-12, 1.0)))
        return {"psi": psi, "untargeted": untargeted, "g": g, "qr": qr, "rounds": rounds}

    def solved(self) -> dict[str, dict[str, Any]]:
        if "solved" not in self._cache:
            self._cache["solved"] = {name: self.fixed_point(name) for name in self.terms}
        return dict(self._cache["solved"])

    # ---- curves ----------------------------------------------------------------------------

    def curve(self, name: str, *, g_current: np.ndarray | None = None) -> np.ndarray:
        """The Theorem 1 curve of a term at the true mechanism, with the declared-basis ``Qr``."""
        term = self.terms[name]
        m = self.masses(name)
        g = m["g"] if g_current is None else g_current
        qr = self.regress(name, g)
        psi0 = self.truth(name)
        point = term.point
        return np.asarray(
            term.ind / g[point] * (term.target - qr[point]) + qr[point] - psi0, dtype=float
        )

    def truths(self) -> dict[str, float]:
        return {name: self.truth(name) for name in self.terms}

    def key_truth(self, key: str) -> float:
        return self.keys[key].value(self.truths())

    def key_curve(self, key: str, *, projected: bool = False) -> np.ndarray:
        spec = self.keys[key]
        psi0 = self.truths()
        gradient = spec.gradient(psi0)
        out = np.zeros(len(self.prob))
        for name, coefficient in gradient.items():
            out += coefficient * self.curve(name)
        if projected:
            if self.score is None:
                raise ValueError("this population has no parametric mechanism score")
            out += self._projection(gradient)
        return out

    def _projection(self, gradient: Mapping[str, float]) -> np.ndarray:
        """``J' I^{-1} s``: the influence of estimating a parametric mechanism, first order."""
        assert self.score is not None
        jacobian = 0.0
        for name, coefficient in gradient.items():
            term = self.terms[name]
            if term.dlogit is None:
                raise ValueError(f"{name} declares no mechanism derivative")
            m = self.masses(name)
            qr = self.regress(name, m["g"])
            weight = m["pw"] * (m["q"] - qr) * (1.0 - m["g"])
            jacobian = jacobian + coefficient * -(weight @ term.dlogit)
        information = (self.score * self.prob[:, None]).T @ self.score
        return np.asarray(self.score @ np.linalg.solve(information, np.atleast_1d(jacobian)))

    def key_sd(self, key: str, *, projected: bool = False) -> float:
        curve = self.key_curve(key, projected=projected)
        return float(np.sqrt(np.sum(self.prob * curve**2)))

    # ---- the bias of a key -----------------------------------------------------------------

    def key_bias(self, key: str, n: int, *, projected: bool = False) -> dict[str, float]:
        """``sqrt(n)`` times the targeted and the untargeted bias, in units of the key's SD."""
        solved = self.solved()
        truth = self.key_truth(key)
        sd = self.key_sd(key, projected=projected)
        spec = self.keys[key]
        targeted = spec.value({name: float(s["psi"]) for name, s in solved.items()})
        untargeted = spec.value({name: float(s["untargeted"]) for name, s in solved.items()})
        scale = np.sqrt(n) / sd
        return {
            "truth": truth,
            "sd": sd,
            "b": (targeted - truth) * scale,
            "untargeted": (untargeted - truth) * scale,
        }

    def paired_variance(self, key: str) -> float:
        """The variance of the untargeted-minus-targeted per-replicate difference, in SD^2.

        Both arms are functions of one draw.  The untargeted curve is the Horvitz-Thompson
        curve at the learner's limit and the targeted curve is the Theorem 1 curve; the
        mechanism's own estimation is not in either.
        """
        spec = self.keys[key]
        psi0 = self.truths()
        gradient = spec.gradient(psi0)
        diff = np.zeros(len(self.prob))
        for name, coefficient in gradient.items():
            term = self.terms[name]
            limit = np.clip(term.limit, 1e-12, 1.0)
            untargeted = term.ind * term.target / limit[term.point] - psi0[name]
            diff += coefficient * (untargeted - self.curve(name))
        sd = self.key_sd(key)
        return float(np.sum(self.prob * diff**2) / sd**2)

    def untargeted_sd(self, key: str) -> float:
        """The SD of the Horvitz-Thompson curve at the true mechanism (a known-weights bound)."""
        spec = self.keys[key]
        psi0 = self.truths()
        out = np.zeros(len(self.prob))
        for name, coefficient in spec.gradient(psi0).items():
            term = self.terms[name]
            g_true = self.masses(name)["g"]
            out += coefficient * (term.ind * term.target / g_true[term.point] - psi0[name])
        return float(np.sqrt(np.sum(self.prob * out**2)))

    def l2_gap(self, name: str, *, declared: bool) -> float:
        """``L2(Q_0 - Qr)`` among ``I = 1``: against the exact conditional mean or the basis."""
        m = self.masses(name)
        term = self.terms[name]
        if declared:
            qr = self.regress(name, m["g"])
        elif term.exact_qr is not None:
            qr = term.exact_qr
        else:
            qr = _conditional_mean(m, term)
        weight = m["mass"]
        return float(np.sqrt(np.sum(weight * (m["q"] - qr) ** 2) / np.sum(weight)))

    def mechanism_range(self, name: str) -> tuple[float, float]:
        """The smallest and the largest value of the term's limit and of its fixed point."""
        solved = self.solved()[name]
        values = np.concatenate([self.terms[name].limit, solved["g"]])
        return float(values.min()), float(values.max())

    def outside_mass(self, name: str, low: float, high: float) -> float:
        """The probability of a covariate point where the true, limit or fixed-point ``g`` leaves.

        A draw clips only on such a point.  A law with Gaussian covariates reaches a bound
        only in the tail, where this mass is the chance one row clips.
        """
        term = self.terms[name]
        solved = self.solved()[name]
        masses = self.masses(name)
        out = np.zeros(len(masses["pw"]), dtype=bool)
        for g in (masses["g"], term.limit, solved["g"]):
            out |= (g <= low) | (g >= high)
        return float(masses["pw"][out].sum())


def _conditional_mean(m: Mapping[str, np.ndarray], term: Term) -> np.ndarray:
    """``E(T | I = 1, g_0(x))``: the mean of ``q`` over points with equal true ``g_0``."""
    g = np.round(m["g"], 12)
    out = np.empty_like(m["q"])
    for value in np.unique(g):
        rows = g == value
        weight = m["mass"][rows]
        out[rows] = np.sum(weight * m["q"][rows]) / np.sum(weight) if weight.sum() > 0 else 0.0
    return out


def _regression(
    x: np.ndarray, q: np.ndarray, weight: np.ndarray, degree: int, binary: bool
) -> np.ndarray:
    """The weighted polynomial regression of ``q`` on ``x``, the estimator's reduced learner.

    The basis is capped at the number of distinct values of ``x`` (among points with weight)
    minus one, where it is the saturated regression: the weighted mean of ``q`` at each value.
    """
    active = weight > 0
    keys = np.round(x, 10)
    distinct = len(np.unique(keys[active]))
    effective = max(1, min(int(degree), distinct - 1))
    out = np.empty_like(q)
    if effective >= distinct - 1:
        for value in np.unique(keys[active]):
            rows = active & (keys == value)
            out[keys == value] = np.sum(weight[rows] * q[rows]) / np.sum(weight[rows])
        return out
    centre = float(np.sum(weight[active] * x[active]) / np.sum(weight[active]))
    spread = float(
        np.sqrt(np.sum(weight[active] * (x[active] - centre) ** 2) / np.sum(weight[active]))
    )
    z = (x - centre) / (spread if spread > 0 else 1.0)
    basis = np.column_stack([z**k for k in range(effective + 1)])
    w = weight / weight.sum()
    if not binary:
        gram = (basis[active] * w[active, None]).T @ basis[active]
        beta = np.linalg.solve(gram, (basis[active] * w[active, None]).T @ q[active])
        return np.asarray(basis @ beta)
    mean = float(np.clip(np.sum(w * q), 1e-9, 1.0 - 1e-9))
    beta = np.zeros(basis.shape[1])
    beta[0] = logit(mean)
    for _ in range(200):
        mu = expit(basis @ beta)
        variance = np.clip(mu * (1.0 - mu), 1e-14, None)
        gradient = basis.T @ (w * (q - mu))
        hessian = (basis * (w * variance)[:, None]).T @ basis
        step = np.linalg.solve(hessian + 1e-14 * np.eye(len(beta)), gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-13:
            break
    return np.asarray(expit(basis @ beta))


# ------------------------------------------------------------------------------------- builders


def rate_step(n: int, size: float = 2.0) -> float:
    """``size * n^(-1/3)``: the rate perturbation of a declared size."""
    return float(size) * float(n) ** (-1.0 / 3.0)


def _cells(*sizes: int) -> tuple[np.ndarray, ...]:
    """Index arrays of a product table, one per axis, in C order."""
    grids = np.meshgrid(*[np.arange(size) for size in sizes], indexing="ij")
    return tuple(grid.ravel() for grid in grids)


def p_mle_main_terms(w1: np.ndarray, w2: np.ndarray, pw: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """The population limit of the main-terms logistic regression of ``A`` on ``(1, W1, W2)``."""
    design = np.column_stack([np.ones_like(w1), w1, w2])
    beta = np.zeros(3)
    for _ in range(200):
        mu = expit(design @ beta)
        gradient = design.T @ (pw * (g1 - mu))
        hessian = (design * (pw * mu * (1.0 - mu))[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return np.asarray(expit(design @ beta))


def build_p(
    law: str, learner: str, n: int, laws: Any, degree: int, *, regimes: bool = False
) -> Population:
    """Laws P (``law="p"``) and PD (``law="pd"``) on the 800-point quadrature.

    Parameters
    ----------
    law : {"p", "pd"}
        The law.
    learner : {"correct", "rate", "rate_scaled", "wrong"}
        The treatment learner whose limit the loop starts from.
    n : int
        The declared sample size in the rate perturbation.
    laws : module
        :mod:`tests.studies.adaptive_iptw_laws`.
    degree : int
        The declared degree of the basis.
    regimes : bool
        Add the regime terms and keys of law PD.
    """
    grid = laws.p_grid()
    w1, w2, pw = (grid[c].to_numpy() for c in ("W1", "W2", "weight"))
    points = len(pw)
    lg = laws.p_logit_g(w1, w2)
    g1 = expit(lg)
    outcome = laws.pd_outcome_logit if law == "pd" else laws.p_outcome_logit
    q = {a: expit(outcome(float(a), w1, w2)) for a in (0, 1)}
    a_idx, y_idx, i_idx = _cells(2, 2, points)
    prob = pw[i_idx] * np.where(a_idx == 1, g1[i_idx], 1.0 - g1[i_idx]) * np.where(
        y_idx == 1, np.where(a_idx == 1, q[1][i_idx], q[0][i_idx]), 0.0
    ) + pw[i_idx] * np.where(a_idx == 1, g1[i_idx], 1.0 - g1[i_idx]) * np.where(
        y_idx == 0, np.where(a_idx == 1, 1.0 - q[1][i_idx], 1.0 - q[0][i_idx]), 0.0
    )
    if learner == "correct":
        limit1 = g1
    elif learner == "rate":
        limit1 = expit(lg + rate_step(n) * (1.0 + w2))
    elif learner == "rate_scaled":
        limit1 = expit((1.0 + rate_step(n)) * lg)
    elif learner == "wrong":
        limit1 = p_mle_main_terms(w1, w2, pw, g1)
    else:
        raise KeyError(learner)
    design = np.column_stack([np.ones(points), w1, w2, w1 * w2])
    y = y_idx.astype(float)
    rule = (w1 > 0.0).astype(float)
    tilt = expit(w1)
    ones = np.ones(points)

    def term(
        name: str,
        ind: np.ndarray,
        target: np.ndarray,
        limit: np.ndarray,
        sign: np.ndarray,
        *,
        binary: bool = True,
    ) -> Term:
        return Term(name, i_idx, ind, target, limit, degree, binary, dlogit=sign[:, None] * design)

    terms = {
        "a0": term("a0", (a_idx == 0).astype(float), y, 1.0 - limit1, -ones),
        "a1": term("a1", (a_idx == 1).astype(float), y, limit1, ones),
    }
    keys = {
        "ey0": Key({"a0": 1.0}),
        "ey1": Key({"a1": 1.0}),
        "ate": Key({"a1": 1.0, "a0": -1.0}),
    }
    if regimes:
        on_rule = np.where(rule == 1.0, limit1, 1.0 - limit1)
        rule_cell = rule[i_idx]
        terms["d"] = term(
            "d",
            (a_idx == rule_cell).astype(float),
            y,
            on_rule,
            np.where(rule == 1.0, 1.0, -1.0),
        )
        terms["s1"] = term(
            "s1", (a_idx == 1).astype(float), tilt[i_idx] * y, limit1, ones, binary=False
        )
        terms["s0"] = term(
            "s0",
            (a_idx == 0).astype(float),
            (1.0 - tilt[i_idx]) * y,
            1.0 - limit1,
            -ones,
            binary=False,
        )
        keys.update(
            {
                f"ey_regime[{laws.NEVER_NAME}]": Key({"a0": 1.0}),
                f"ey_regime[{laws.RULE_NAME}]": Key({"d": 1.0}),
                f"ey_regime[{laws.STOCHASTIC_NAME}]": Key({"s1": 1.0, "s0": 1.0}),
                f"ate_regime[{laws.RULE_NAME} vs {laws.NEVER_NAME}]": Key({"d": 1.0, "a0": -1.0}),
                f"ate_regime[{laws.STOCHASTIC_NAME} vs {laws.NEVER_NAME}]": Key(
                    {"s1": 1.0, "s0": 1.0, "a0": -1.0}
                ),
            }
        )
    treated = a_idx.astype(float)
    score = (treated - g1[i_idx])[:, None] * design[i_idx]
    return Population(prob, terms, keys, score)


def _vl_exact_qr(law: str, w1: np.ndarray, w2: np.ndarray, laws: Any) -> np.ndarray:
    """``Qr_0(g_0(w)) = E[Q(1, W) | g_0(W) = g_0(w)]`` of a validity law, by rotation.

    ``g_0`` depends on ``W`` through the index ``0.6 W1 + 0.3 W2`` only.  Rotate ``W`` into
    the unit index ``s`` and its orthogonal complement ``t``, which is standard normal and
    independent of ``s``.  The conditional mean averages ``Q`` over ``t`` at the point's ``s``.
    """
    scale = float(np.hypot(0.6, 0.3))
    s_value = (0.6 * w1 + 0.3 * w2) / scale
    t_nodes, t_weights = np.polynomial.hermite_e.hermegauss(80)
    t_weights = t_weights / t_weights.sum()
    out = np.empty(len(w1))
    for start in range(0, len(w1), 2_000):
        s = s_value[start : start + 2_000, None]
        t = t_nodes[None, :]
        a = (0.6 * s - 0.3 * t) / scale
        b = (0.3 * s + 0.6 * t) / scale
        out[start : start + 2_000] = laws.vl_q1(law, a, b) @ t_weights
    return out


def build_vl(law: str, learner: str, n: int, laws: Any, degree: int) -> Population:
    """The validity laws VL1 and VL0 on the 80 x 80 Gauss-Hermite grid."""
    grid = laws.gauss_hermite_grid()
    keep = grid["weight"].to_numpy() > 1e-13
    grid = grid.loc[keep]
    w1, w2, pw = (grid[c].to_numpy() for c in ("W1", "W2", "weight"))
    pw = pw / pw.sum()
    points = len(pw)
    g = laws.vl_g(w1, w2)
    q1 = laws.vl_q1(law, w1, w2)
    a_idx, y_idx, i_idx = _cells(2, 2, points)
    q = np.where(a_idx == 1, q1[i_idx], 0.5)
    prob = (
        pw[i_idx]
        * np.where(a_idx == 1, g[i_idx], 1.0 - g[i_idx])
        * np.where(y_idx == 1, q, 1.0 - q)
    )
    lg = logit(g)
    limit = g if learner == "glm" else expit(lg + rate_step(n, 1.0) * w2)
    design = np.column_stack([np.ones(points), w1, w2])
    term = Term(
        "a1",
        i_idx,
        (a_idx == 1).astype(float),
        y_idx.astype(float),
        limit,
        degree,
        dlogit=design,
        exact_qr=_vl_exact_qr(law, w1, w2, laws),
    )
    score = (a_idx.astype(float) - g[i_idx])[:, None] * design[i_idx]
    return Population(prob, {"a1": term}, {"ey1": Key({"a1": 1.0})}, score)


def build_m(learner: str, n: int, multi: Any, degree: int, *, nodes: int = 20) -> Population:
    """Law M on a three-dimensional Gauss-Hermite grid; the arms in code order."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    i0, i1, i2 = _cells(nodes, nodes, nodes)
    w0, w1, w2 = x[i0], x[i1], x[i2]
    pw = w[i0] * w[i1] * w[i2]
    points = len(pw)
    labels = multi.LABELS
    # ``CausalData`` sorts the labels: high, low, medium.  The law's logits are for
    # (low, medium, high).
    eta = {
        "low": np.zeros(points),
        "medium": 0.25 * w0 - 0.15 * w1,
        "high": -0.15 * w0 + 0.25 * w1,
    }
    codes = {"high": 0.0, "low": 1.0, "medium": 2.0}
    order = sorted(labels)

    top = np.max(np.column_stack([eta[k] for k in order]), axis=1)
    weights = {k: np.exp(eta[k] - top) for k in order}
    total = sum(weights.values())
    g = {k: weights[k] / total for k in order}
    if learner == "correct":
        limit = g
    else:
        # The learner shrinks every arm probability to uniform by ``2 n^(-1/3)`` (a function of
        # the true ``g_a`` alone, so each level set of ``g_a`` survives).
        step = rate_step(n)
        limit = {k: (1.0 - step) * g[k] + step / len(order) for k in order}
    q = {k: expit(-0.5 + 0.6 * codes[k] + 0.6 * w0 - 0.3 * w1 + 0.2 * w2) for k in order}
    arm_idx, y_idx, i_idx = _cells(len(order), 2, points)
    prob = np.zeros(len(arm_idx))
    for j, k in enumerate(order):
        rows = arm_idx == j
        prob[rows] = (
            pw[i_idx[rows]]
            * g[k][i_idx[rows]]
            * np.where(y_idx[rows] == 1, q[k][i_idx[rows]], 1.0 - q[k][i_idx[rows]])
        )
    terms = {
        k: Term(
            k,
            i_idx,
            (arm_idx == j).astype(float),
            y_idx.astype(float),
            limit[k],
            degree,
        )
        for j, k in enumerate(order)
    }
    keys = {f"ey[{k}]": Key({k: 1.0}) for k in order}
    keys["ate[low vs high]"] = Key({"low": 1.0, "high": -1.0})
    keys["ate[medium vs high]"] = Key({"medium": 1.0, "high": -1.0})
    return Population(prob, terms, keys)


def _l1_table(law: Any) -> tuple[np.ndarray, ...]:
    """Cells ``(w, a, delta, y)`` of a two-arm finite law with a missing outcome."""
    w, a, d, y = _cells(len(law.p_w), 2, 2, 2)
    mu = law.mu[w, a]
    prob = (
        law.p_w[w]
        * law.g[w, a]
        * np.where(d == 1, law.pi[w, a], 1.0 - law.pi[w, a])
        * np.where(d == 1, np.where(y == 1, mu, 1.0 - mu), 0.5)
    )
    return w, a, d, y, prob


def build_cn(n: int, mar: Any, degrees: tuple[int, int]) -> Population:
    """Law CN: law C ``l1`` with a rate-perturbed response; ``ey_obs``, ``ey0``, ``par``, ``paf``."""
    law = mar.LAWS["l1"]
    w, a, d, y, prob = _l1_table(law)
    step = rate_step(n)
    pi_hat = expit(logit(law.pi) + step * (1.0 + np.array([0.0, 1.0]))[None, :])
    observed = Term(
        "obs", w * 2 + a, d.astype(float), y.astype(float), pi_hat.reshape(-1), degrees[0]
    )
    zero_limit = law.g[:, 0] * pi_hat[:, 0]
    zero = Term("zero", w, (d * (a == 0)).astype(float), y.astype(float), zero_limit, degrees[1])
    keys = {
        "ey_obs": Key({"obs": 1.0}),
        "ey0": Key({"zero": 1.0}),
        "par": Key({"obs": 1.0, "zero": -1.0}),
        "paf": Key({}, paf=("obs", "zero")),
    }
    return Population(prob, {"obs": observed, "zero": zero}, keys)


def build_rm(n: int, mar: Any, degree: int, rule: np.ndarray, tilt: np.ndarray) -> Population:
    """Law RM: law C ``l1`` with the rule ``d`` and the stochastic regime ``s`` (in sample)."""
    law = mar.LAWS["l1"]
    w, a, d, y, prob = _l1_table(law)
    step = rate_step(n)
    g1 = law.g[:, 1]
    g1_hat = expit(logit(g1) + step * (2.0 * rule - 1.0))
    g_hat = np.column_stack([1.0 - g1_hat, g1_hat])
    pi = law.pi
    on_rule = g_hat[np.arange(3), rule] * pi[np.arange(3), rule]
    yy = y.astype(float)
    terms = {
        "d": Term("d", w, (d * (a == rule[w])).astype(float), yy, on_rule, degree),
        "s1": Term(
            "s1",
            w,
            (d * (a == 1)).astype(float),
            tilt[w] * yy,
            g_hat[:, 1] * pi[:, 1],
            degree,
            binary=False,
        ),
        "s0": Term(
            "s0",
            w,
            (d * (a == 0)).astype(float),
            (1.0 - tilt[w]) * yy,
            g_hat[:, 0] * pi[:, 0],
            degree,
            binary=False,
        ),
    }
    keys = {
        "ey_regime[d]": Key({"d": 1.0}),
        "ey_regime[s]": Key({"s1": 1.0, "s0": 1.0}),
        "ate_regime[s vs d]": Key({"s1": 1.0, "s0": 1.0, "d": -1.0}),
    }
    return Population(prob, terms, keys)


def build_ma(two: Any, p_w: np.ndarray, degree: int, rule: np.ndarray) -> Population:
    """Law MA: the two-arm law with a missing treatment and a missing outcome (oracle factors)."""
    w, a, recorded, response, y = _cells(3, 2, 2, 2, 2)
    seen = recorded == 1
    prob = (
        p_w[w]
        * two.g[w, a]
        * np.where(seen, two.pi_treatment[w, a], 1.0 - two.pi_treatment[w, a])
        * np.where(
            seen,
            np.where(response == 1, two.pi[w, a], 1.0 - two.pi[w, a]),
            0.5,
        )
        * np.where(seen & (response == 1), np.where(y == 1, two.q[w, a], 1.0 - two.q[w, a]), 0.5)
    )
    composite = two.g * two.pi_treatment * two.pi
    observed = (seen & (response == 1)).astype(float)
    yy = y.astype(float)
    terms = {
        f"a{arm}": Term(f"a{arm}", w, observed * (a == arm), yy, composite[:, arm], degree)
        for arm in (0, 1)
    }
    terms["d"] = Term(
        "d",
        w,
        observed * (a == rule[w]),
        yy,
        composite[np.arange(3), rule],
        degree,
    )
    keys = {
        "ey0": Key({"a0": 1.0}),
        "ey1": Key({"a1": 1.0}),
        "ate": Key({"a1": 1.0, "a0": -1.0}),
        "ey_regime[d]": Key({"d": 1.0}),
    }
    return Population(prob, terms, keys)


# --------------------------------------------------------------------------- identifiability


def _logit_spread(
    cell: np.ndarray, numerator: np.ndarray, denominator: np.ndarray, cells: int, fold: np.ndarray
) -> np.ndarray:
    """The across-fold SD of the out-of-fold logit of ``P(numerator | denominator)`` per cell."""
    folds = int(fold.max()) + 1

    def counts(mask: np.ndarray) -> np.ndarray:
        flat = np.bincount(fold[mask] * cells + cell[mask], minlength=folds * cells)
        return flat.reshape(folds, cells).astype(float)

    top, bottom = counts(numerator), counts(denominator)
    held_in = (top.sum(axis=0) - top) / (bottom.sum(axis=0) - bottom)
    return np.asarray(logit(held_in).std(axis=0, ddof=1))


def smallest_gap_ratio(truth_logit: np.ndarray, spread: np.ndarray) -> float:
    """The smallest gap between adjacent cell logits over five times the larger SD of the pair."""
    order = np.argsort(truth_logit)
    gaps = np.diff(truth_logit[order])
    sd = spread[order]
    return float(np.min(gaps / (5.0 * np.maximum(sd[:-1], sd[1:]))))


def composite_gap_ratios(
    p_w: np.ndarray,
    g: np.ndarray,
    recorded: np.ndarray | None,
    response: np.ndarray,
    *,
    n: int,
    draws: int = 200,
    folds: int = 10,
    seed: int = 34,
) -> dict[str, float]:
    """The identifiability ratio of each arm's composite cells and of the response cells.

    Draw ``(W, A, Delta_A, Delta)`` from the finite law and cross-fit the saturated factors on
    the cells.  The out-of-fold composite of arm ``a`` in cell ``w`` is the training share of
    the rows of ``w`` with ``A = a``, ``Delta_A = 1`` and ``Delta = 1``, because the three
    factors telescope on one training set.  The fold-to-fold SD of a cell is the within-draw SD
    of its out-of-fold logit across the folds, averaged over the draws.  The ratio is
    :func:`smallest_gap_ratio`.

    Parameters
    ----------
    p_w, g : numpy.ndarray
        ``P(W = w)`` and ``P(A = a | W = w)``.
    recorded : numpy.ndarray or None
        ``P(Delta_A = 1 | A = a, W = w)``, or ``None`` when the treatment is always recorded.
    response : numpy.ndarray
        ``P(Delta = 1 | A = a, W = w)``.
    n : int
        The declared sample size.
    draws, folds, seed : int
        The number of draws, the number of folds and the seed of the generator.

    Returns
    -------
    dict
        ``arm0``, ``arm1`` (and ``arm2``): the composite cells of each arm over ``W``, and
        ``response``: the ``(W, A)`` cells of the response factor.
    """
    rng = np.random.default_rng(seed)
    w_levels, arms = g.shape
    rec_prob = np.ones_like(g) if recorded is None else recorded
    spreads = {arm: np.zeros(w_levels) for arm in range(arms)}
    response_spread = np.zeros(w_levels * arms)
    everyone = np.ones(n, dtype=bool)
    for _ in range(draws):
        w = rng.choice(w_levels, size=n, p=p_w)
        a = (rng.uniform(size=n)[:, None] > np.cumsum(g[w], axis=1)).sum(axis=1)
        rec = rng.uniform(size=n) < rec_prob[w, a]
        resp = rng.uniform(size=n) < response[w, a]
        fold = rng.permutation(n) % folds
        for arm in range(arms):
            spreads[arm] += _logit_spread(w, (a == arm) & rec & resp, everyone, w_levels, fold)
        response_spread += _logit_spread(w * arms + a, rec & resp, rec, w_levels * arms, fold)
    out = {
        f"arm{arm}": smallest_gap_ratio(
            logit((g * rec_prob * response)[:, arm]), spreads[arm] / draws
        )
        for arm in range(arms)
    }
    out["response"] = smallest_gap_ratio(logit(response.reshape(-1)), response_spread / draws)
    return out
