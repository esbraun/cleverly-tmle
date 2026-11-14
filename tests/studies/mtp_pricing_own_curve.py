r"""The own influence curve of the pricing law's ``LTMLE`` and ``SDR`` fits, in numpy.

The tail size of a Wald cell reads the curve its fit converges to, :math:`D(Q_\infty, r_0)`,
not only the efficient influence function: the primary learners of ``mtp-pricing`` are
misspecified at nodes 1 to 4, and their limit curve has a heavier tail than the EIF.  This
module re-implements the single-fold recursions of :mod:`cleverly.longitudinal.sequential`
(``LTMLE``, with its bounded intercept fluctuation) and :mod:`cleverly.longitudinal.sdr`
(``SDR``) for the declared learners of law P (:mod:`tests.studies.mtp_pricing_common`), so
that a fit on :math:`10^7` rows fixes the limit nuisances and the curve can then be evaluated
on any number of fresh draws:

.. math::

    D = Q^*_1(d_1(A_1), H_1) - \psi + \sum_t \omega_t \bigl(Z_{t+1} - Q^*_t(A_t, H_t)\bigr),

with :math:`Z_{t+1} = Q^*_{t+1}(d_{t+1}(A_{t+1}), H_{t+1})`, :math:`Z_{\tau+1} = Y` and
:math:`\omega_t` the cumulative exact ratio (``LTMLE``), and
:math:`D = \varphi_1 - \psi` with :math:`\varphi_t = Q_t(d_t(A_t)) + r_t
(\varphi_{t+1} - Q_t(A_t))`, :math:`\varphi_{\tau+1} = Y` (``SDR``, the sum of
``cleverly.longitudinal.sdr.partial_weights`` written as a recursion).

Learners (``canonical_mtp_pricing.fit`` and ``mtp_pricing_properties.declared_learners``):

* ``"primary"``: the quasibinomial GLM on the outcome design at the last node, least squares at
  every earlier one; ``LTMLE`` with :data:`~tests.studies.canonical_mtp_pricing.PRIMARY_ALPHA`.
* ``"property"``: :class:`~tests.studies.mtp_pricing_common.FinalOutcome` and
  :class:`~tests.studies.mtp_pricing_common.SplinePseudo`, ``LTMLE`` at the default bound.
  With the exact ratio it is the curve of the ``up_oracle_tau*`` cells, and with its
  correctly specified last node it stands in for the EIF, which has no closed form here.
* ``"revenue"``: the property learners on the revenue ``R / 10``.

Every fit reads the exact ratio (:func:`~tests.studies.mtp_pricing_common.ratio`), the limit of
:class:`~tests.studies.mtp_pricing_common.OracleRatio`.  The regressions solve their normal
equations accumulated over chunks of rows, so a :math:`10^7`-row fit holds no full design.
``tests/unit/test_mtp_tail_size.py`` checks the curve of a 5,000-row fit against
``cleverly`` to 1e-8.  This module imports nothing from ``cleverly``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import expit, logit
from scipy.stats import norm

from tests.studies import mtp_pricing_common as law

Learners = Literal["primary", "property", "revenue"]
Estimator = Literal["ltmle", "sdr"]

#: The submodel bounds: ``PRIMARY_ALPHA`` of the primary ``LTMLE`` rows, the default otherwise.
PRIMARY_ALPHA = 1.0 - 1e-5
DEFAULT_ALPHA = 0.9995
#: The rows of one accumulated block of the normal equations.
CHUNK = 1_000_000
#: ``QuasiBinomialGLM``'s defaults.
GLM_MAX_ITER = 100
GLM_TOL = 1e-10

Columns = Mapping[str, np.ndarray]


def columns(frame: pd.DataFrame | Columns, tau: int) -> dict[str, np.ndarray]:
    """The arrays the recursions read, with ``Y`` the outcome on the ``[0, 1]`` scale."""
    names = law.columns(tau)
    out = {name: np.asarray(frame[name], dtype=float) for name in names}
    out["Y"] = np.asarray(frame["Y"], dtype=float)
    out["R"] = np.asarray(frame["R"], dtype=float) if "R" in frame else out["Y"] * out[f"A{tau}"]
    return out


def outcome_design(cols: Columns, t: int, dose: np.ndarray | None = None) -> np.ndarray:
    """``[W1, L_1..L_t, A_1..A_t]``, the column order of :func:`~law.node_columns`."""
    blocks = [cols["W1"], *(cols[f"L{s}"] for s in range(1, t + 1))]
    blocks += [cols[f"A{s}"] for s in range(1, t)]
    blocks.append(cols[f"A{t}"] if dose is None else dose)
    return np.column_stack(blocks)


def node_mean(cols: Columns, t: int) -> np.ndarray:
    """The truncated normal's mean of ``A_t`` given the history."""
    gap = np.zeros_like(cols["W1"]) if t == 1 else cols[f"A{t - 1}"] - law.centre(cols[f"L{t - 1}"])
    return np.asarray(law.price_mean(cols["W1"], cols[f"L{t}"], gap), dtype=float)


def ratios(cols: Columns, label: str, tau: int) -> list[np.ndarray]:
    """Each node's exact ratio ``r_t`` at the observed dose."""
    return [
        law.ratio(label, cols[f"A{t}"], node_mean(cols, t), cols[f"L{t}"])
        for t in range(1, tau + 1)
    ]


def policy_dose(cols: Columns, label: str, t: int) -> np.ndarray:
    return np.asarray(law.MAPS[label](cols[f"A{t}"], cols[f"L{t}"]), dtype=float)


# ----------------------------------------------------------------------------- learners


@dataclass
class NodeModel:
    """One node's fitted regression: least squares or a quasibinomial GLM on a basis.

    Parameters
    ----------
    kind : {"linear", "glm"}
        The family.
    basis : callable
        Maps the outcome design to the regression's columns, without the intercept.
    revenue : bool
        :class:`~law.FinalOutcome` on the revenue scale: the GLM fits ``Y`` and predicts
        ``A_t P(Y = 1) / 10``.
    coef : numpy.ndarray
        The intercept, then the basis coefficients.
    """

    kind: Literal["linear", "glm"]
    basis: Callable[[np.ndarray], np.ndarray]
    revenue: bool = False
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def predict(self, design: np.ndarray) -> np.ndarray:
        eta = self.coef[0] + self.basis(design) @ self.coef[1:]
        if self.kind == "linear":
            return np.asarray(eta)
        p = np.asarray(expit(eta))
        if self.revenue:
            return design[:, -1] * p / law.REVENUE_BOUNDS[1]
        return p

    def target(self, design: np.ndarray, z: np.ndarray) -> np.ndarray:
        if not self.revenue:
            return z
        return np.clip(law.REVENUE_BOUNDS[1] * z / design[:, -1], 0.0, 1.0)


def _chunks(n: int, chunk: int) -> Iterable[slice]:
    return (slice(start, min(start + chunk, n)) for start in range(0, n, chunk))


def fit_node(
    model: NodeModel,
    design_of: Callable[[slice], np.ndarray],
    z: np.ndarray,
    weight: np.ndarray | None = None,
    chunk: int = CHUNK,
) -> NodeModel:
    """Fit ``model`` to ``z`` by weighted least squares accumulated over row chunks.

    Each solve is a chunked QR (:func:`_least_squares`), as accurate as the ``lstsq`` of
    ``LinearRegression`` and ``QuasiBinomialGLM``.  The GLM takes ``QuasiBinomialGLM``'s
    iterates: the weighted mean's logit as the start, the working response, and the stop at a
    largest coefficient change of 1e-10.
    """
    n = z.shape[0]
    w = np.ones(n) if weight is None else weight

    def blocks() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for rows in _chunks(n, chunk):
            design = design_of(rows)
            features = np.column_stack([np.ones(design.shape[0]), model.basis(design)])
            yield features, model.target(design, z[rows]), w[rows]

    if model.kind == "linear":
        model.coef = _least_squares(blocks())
        return model
    total, mass = 0.0, 0.0
    width = 0
    for features, target, weights in blocks():
        total += float(np.sum(weights * target))
        mass += float(np.sum(weights))
        width = features.shape[1]
    mean = total / mass
    coef = np.zeros(width)
    coef[0] = np.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))

    def working(coef: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for features, target, weights in blocks():
            eta = features @ coef
            fitted = expit(eta)
            variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
            yield features, eta + (target - fitted) / variance, weights * variance

    for _ in range(GLM_MAX_ITER):
        updated = _least_squares(working(coef))
        if np.max(np.abs(updated - coef)) <= GLM_TOL:
            coef = updated
            break
        coef = updated
    else:
        raise RuntimeError("the quasibinomial IRLS did not converge")
    model.coef = coef
    return model


def _least_squares(blocks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Weighted least squares by a QR updated block by block (TSQR)."""
    upper: np.ndarray | None = None
    rotated: np.ndarray | None = None
    for features, target, weights in blocks:
        root = np.sqrt(weights)
        a, b = features * root[:, None], target * root
        if upper is not None and rotated is not None:
            a, b = np.vstack([upper, a]), np.concatenate([rotated, b])
        q, upper = np.linalg.qr(a)
        rotated = q.T @ b
    assert upper is not None and rotated is not None
    return np.asarray(scipy.linalg.solve_triangular(upper, rotated))


def _identity(design: np.ndarray) -> np.ndarray:
    return design


def _final_outcome(design: np.ndarray) -> np.ndarray:
    features, _ = law.FinalOutcome._design(design)
    return features


@dataclass(frozen=True)
class _Spline:
    """:class:`~law.SplinePseudo`'s basis at knots fixed from the fit's rows."""

    knots: Mapping[str, np.ndarray]

    def __call__(self, design: np.ndarray) -> np.ndarray:
        learner = law.SplinePseudo()
        learner.knots_ = dict(self.knots)
        return learner._basis(design, False)


def _spline_knots(cols: Columns, t: int) -> dict[str, np.ndarray]:
    names = [f"A{t}", "W1"] + ([f"A{t - 1}"] if t > 1 else [])
    return {name: np.quantile(cols[name], law.SPLINE_QUANTILES) for name in names}


def node_learner(learners: Learners, t: int, tau: int, cols: Columns) -> NodeModel:
    """The declared learner of node ``t``, unfitted."""
    if learners == "primary":
        return NodeModel("glm" if t == tau else "linear", _identity)
    if t == tau:
        return NodeModel("glm", _final_outcome, revenue=learners == "revenue")
    return NodeModel("glm", _Spline(_spline_knots(cols, t)))


# ------------------------------------------------------------------------- fluctuation


def _shrink(p: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(p, 1.0 - alpha, alpha)


def _intercept(z: np.ndarray, offset: np.ndarray, weight: np.ndarray) -> float:
    """The weighted logistic intercept with an offset, by Newton to a 1e-12 gradient."""
    eps, total = 0.0, float(np.sum(weight))
    for _ in range(50):
        p = expit(offset + eps)
        gradient = float(np.sum(weight * (z - p)))
        if abs(gradient) / total <= 1e-12:
            break
        eps += gradient / float(np.sum(weight * p * (1.0 - p)))
    return eps


@dataclass(frozen=True)
class Fluctuation:
    """The steps of one node's bounded intercept fluctuation, replayable on any rows.

    ``cleverly`` shrinks the predictions into ``[1 - alpha, alpha]``, takes a full Newton step,
    and shrinks again after each step, until the score is solved
    (``cleverly.fluctuation.iterative.solve_fluctuation``).

    Parameters
    ----------
    alpha : float
        The submodel bound.
    steps : tuple of float
        Each iteration's step.
    """

    alpha: float
    steps: tuple[float, ...]

    def apply(self, q: np.ndarray) -> np.ndarray:
        current = _shrink(q, self.alpha)
        for step in self.steps:
            current = _shrink(expit(logit(current) + step), self.alpha)
        return current

    @classmethod
    def fit(
        cls, z: np.ndarray, q: np.ndarray, weight: np.ndarray, alpha: float, max_iter: int = 100
    ) -> Fluctuation:
        steps: list[float] = []
        current = _shrink(q, alpha)
        scale = float(np.sum(weight))
        for _ in range(max_iter):
            step = _intercept(z, logit(current), weight)
            steps.append(step)
            current = _shrink(expit(logit(current) + step), alpha)
            score = abs(float(np.sum(weight * (z - current)))) / scale
            if score <= 1e-13 or abs(step) <= 1e-13:
                break
        return cls(alpha, tuple(steps))


# ----------------------------------------------------------------------------- the fit


@dataclass
class PlanFit:
    """One plan's fitted recursion: the node models, fluctuations and the estimate.

    Parameters
    ----------
    label : str
        The plan.
    models : list of NodeModel
        Node 1 first.
    fluctuations : list of Fluctuation or None
        ``LTMLE``'s per node; ``None`` entries under ``SDR``.
    psi : float
        The estimate on the fit's rows.
    """

    label: str
    models: list[NodeModel]
    fluctuations: list[Fluctuation | None]
    psi: float = float("nan")


@dataclass
class OwnCurve:
    """The own curve of one estimator with one set of learners on law P at ``tau`` nodes.

    Parameters
    ----------
    tau : int
        Nodes.
    estimator : {"ltmle", "sdr"}
        The recursion.
    learners : {"primary", "property", "revenue"}
        The declared learners.
    labels : tuple of str
        The plans; ``natural_prices`` first.
    """

    tau: int
    estimator: Estimator
    learners: Learners
    labels: tuple[str, ...] = law.LABELS
    plans: dict[str, PlanFit] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return PRIMARY_ALPHA if self.learners == "primary" else DEFAULT_ALPHA

    def outcome(self, cols: Columns) -> np.ndarray:
        if self.learners == "revenue":
            return np.asarray(cols["R"]) / law.REVENUE_BOUNDS[1]
        return np.asarray(cols["Y"])

    def fit(self, cols: Columns, chunk: int = CHUNK) -> OwnCurve:
        """Fit every plan's recursion on ``cols``; the curves then read :meth:`curves`."""
        for label in self.labels:
            self.plans[label] = self._fit_plan(cols, label, chunk)
        return self

    def _fit_plan(self, cols: Columns, label: str, chunk: int) -> PlanFit:
        tau = self.tau
        r = ratios(cols, label, tau)
        omega = np.cumprod(np.column_stack(r), axis=1)
        models: list[NodeModel] = [None] * tau  # type: ignore[list-item]
        flucts: list[Fluctuation | None] = [None] * tau
        z = self.outcome(cols)
        for t in range(tau, 0, -1):
            dose = policy_dose(cols, label, t)

            def observed(rows: slice, t: int = t) -> np.ndarray:
                return outcome_design({k: v[rows] for k, v in cols.items()}, t)

            model = fit_node(node_learner(self.learners, t, tau, cols), observed, z, chunk=chunk)
            models[t - 1] = model
            q_obs = self._predict(model, cols, t, None, chunk)
            q_pol = self._predict(model, cols, t, dose, chunk)
            if self.estimator == "ltmle":
                flucts[t - 1] = Fluctuation.fit(z, q_obs, omega[:, t - 1], self.alpha)
                z = flucts[t - 1].apply(q_pol)  # type: ignore[union-attr]
            else:
                z = q_pol + r[t - 1] * (z - q_obs)
        plan = PlanFit(label, models, flucts)
        plan.psi = float(np.mean(z))
        return plan

    def _predict(
        self, model: NodeModel, cols: Columns, t: int, dose: np.ndarray | None, chunk: int
    ) -> np.ndarray:
        n = cols["W1"].shape[0]
        out = np.empty(n)
        for rows in _chunks(n, chunk):
            part = {k: v[rows] for k, v in cols.items()}
            values = model.predict(outcome_design(part, t, None if dose is None else dose[rows]))
            out[rows] = np.clip(values, 0.0, 1.0) if self.estimator == "ltmle" else values
        return out

    def curves(
        self, cols: Columns, known: Mapping[str, list[np.ndarray]] | None = None
    ) -> dict[str, np.ndarray]:
        """Each plan's curve ``D`` at the rows of ``cols``, with the fit's limit nuisances.

        ``known`` holds each plan's :func:`ratios` at these rows, when several fits read them.
        """
        return {
            label: self._curve(cols, plan, None if known is None else known[label])
            for label, plan in self.plans.items()
        }

    def _curve(
        self, cols: Columns, plan: PlanFit, known: list[np.ndarray] | None = None
    ) -> np.ndarray:
        tau = self.tau
        r = ratios(cols, plan.label, tau) if known is None else known
        z = self.outcome(cols)
        if self.estimator == "sdr":
            for t in range(tau, 0, -1):
                q_obs = plan.models[t - 1].predict(outcome_design(cols, t))
                q_pol = plan.models[t - 1].predict(
                    outcome_design(cols, t, policy_dose(cols, plan.label, t))
                )
                z = q_pol + r[t - 1] * (z - q_obs)
            return z - plan.psi
        omega = np.cumprod(np.column_stack(r), axis=1)
        curve = np.zeros_like(z)
        for t in range(tau, 0, -1):
            fluctuation = plan.fluctuations[t - 1]
            assert fluctuation is not None
            model = plan.models[t - 1]
            q_obs = np.clip(model.predict(outcome_design(cols, t)), 0.0, 1.0)
            q_pol = np.clip(
                model.predict(outcome_design(cols, t, policy_dose(cols, plan.label, t))), 0.0, 1.0
            )
            curve = curve + omega[:, t - 1] * (z - fluctuation.apply(q_obs))
            z = fluctuation.apply(q_pol)
        return z - plan.psi + curve

    def quantities(
        self, cols: Columns, known: Mapping[str, list[np.ndarray]] | None = None
    ) -> dict[str, np.ndarray]:
        """The curves of every mean and every contrast with ``natural_prices``."""
        curves = self.curves(cols, known)
        out = {f"ey[{label}]": curve for label, curve in curves.items()}
        for label in self.labels:
            if label != law.REFERENCE:
                out[f"ate[{label}]"] = curves[label] - curves[law.REFERENCE]
        return out


# ----------------------------------------------------------------- the tau = 2 quadrature


def _dose_nodes(l: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre on ``[l_t, u_t]`` split where a plan's map or its ratio breaks."""
    low, high = float(law.lower(l)), float(law.upper(l))
    breaks = sorted({low, high, law.UP * low, high / law.UP, low / law.DOWN, law.DOWN * high})
    base, weights = np.polynomial.legendre.leggauss(order)
    nodes, mass = [], []
    for a, b in itertools.pairwise(breaks):
        half = 0.5 * (b - a)
        nodes.append(a + half * (base + 1.0))
        mass.append(half * weights)
    return np.concatenate(nodes), np.concatenate(mass)


def quadrature_columns(order: int) -> Iterable[tuple[dict[str, np.ndarray], np.ndarray]]:
    """The law at ``tau = 2`` as weighted rows: one batch per ``W1`` node.

    Every ``(W1, L_1, A_1, L_2, A_2, Y)`` point of a composite Gauss-Legendre grid, split at
    every break of the plans' maps and ratios, with its probability as the weight; the
    weights sum to 1 to quadrature precision.
    """
    base, weights = np.polynomial.legendre.leggauss(order)
    w1_nodes = law.W_BOUND * base
    w1_mass = law.W_BOUND * weights * np.exp(-0.5 * w1_nodes**2) / np.sqrt(2.0 * np.pi)
    w1_mass = w1_mass / (2.0 * norm.cdf(law.W_BOUND) - 1.0)
    for w1, mass_w in zip(w1_nodes, w1_mass, strict=True):
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("W1", "L1", "A1", "L2", "A2")}
        parts["Y"], parts["weight"] = [], []
        for l1 in (0.0, 1.0):
            p1 = float(law.p_engaged(w1, 0.0, 0.0))
            a1, m1 = _dose_nodes(l1, order)
            g1 = law.tn_pdf(a1, law.price_mean(w1, l1, 0.0), law.lower(l1), law.upper(l1))
            gap1 = a1 - law.centre(l1)
            for l2 in (0.0, 1.0):
                p2 = law.p_engaged(w1, gap1, l1)
                a2, m2 = _dose_nodes(l2, order)
                g2 = law.tn_pdf(
                    a2[None, :],
                    law.price_mean(w1, l2, gap1[:, None]),
                    law.lower(l2),
                    law.upper(l2),
                )
                buy = law.purchase(w1, l2, a2[None, :] - law.centre(l2), gap1[:, None])
                joint = (
                    mass_w
                    * (p1 if l1 else 1.0 - p1)
                    * (m1 * g1 * (p2 if l2 else 1.0 - p2))[:, None]
                    * (m2 * g2)
                )
                shape = joint.shape
                for y in (0.0, 1.0):
                    parts["W1"].append(np.full(joint.size, w1))
                    parts["L1"].append(np.full(joint.size, l1))
                    parts["A1"].append(np.broadcast_to(a1[:, None], shape).ravel())
                    parts["L2"].append(np.full(joint.size, l2))
                    parts["A2"].append(np.broadcast_to(a2[None, :], shape).ravel())
                    parts["Y"].append(np.full(joint.size, y))
                    parts["weight"].append((joint * (buy if y else 1.0 - buy)).ravel())
        cols = {k: np.concatenate(v) for k, v in parts.items()}
        weight = cols.pop("weight")
        cols["R"] = cols["Y"] * cols["A2"]
        yield cols, weight
