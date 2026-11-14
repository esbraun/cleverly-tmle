r"""The own influence curve of the boundaries law's ``LTMLE`` and ``SDR`` fits, by quadrature.

A Wald cell's tail size reads the curve its fit converges to, :math:`D(Q_\infty, r_0)`, as
well as the efficient influence function.  On the two-node boundaries laws
(:mod:`tests.studies.mtp_boundaries_common`) the limit nuisances are the solutions of the
population normal equations of the declared learners, and this module solves them on a
weighted grid, then integrates the moments of

.. math::

    D = Q^*_1(d_1(A_1)) - \psi + r_1 \bigl(Z_2 - Q^*_1(A_1)\bigr)
        + r_1 r_2 \bigl(Y - Q^*_2(A_2)\bigr),\qquad Z_2 = Q^*_2(d_2(A_2))

(``LTMLE``: :math:`Q^*` the bounded intercept fluctuation of each node's fit, weighted by the
cumulative ratio) or of :math:`D = \varphi_1 - \psi`, :math:`\varphi_1 = Q_1(d_1(A_1)) +
r_1(\varphi_2 - Q_1(A_1))`, :math:`\varphi_2 = Q_2(d_2(A_2)) + r_2 (Y - Q_2(A_2))` (``SDR``).

The data enter as :class:`Block` objects: node-1 points ``(W1, W2, A1)`` with their mass, and,
under each, inner points ``(L2, A2)`` with their conditional mass and ``E[Y | H_2, A_2]``.  The
quadrature of :func:`tests.studies.mtp_tail_size_common.law_b_own_moments` yields them, and a
sample is one block with one inner point per row, ``Y`` itself and mass ``1 / n``.  Every
score is linear in ``Y`` and in the node-1 target, so the grid's conditional means solve the
population equations exactly.  ``tests/unit/test_mtp_tail_size.py`` checks a sample fit
against ``cleverly`` to 1e-8.

Learners (``canonical_mtp_boundaries.learners`` and ``mtp_boundaries_properties``):

* ``"primary"``: the quasibinomial GLM on ``[W1, W2, L2, A1, A2]`` (and ``1{A2 = 3}`` on law
  B') at node 2, least squares on ``[W1, W2, A1]`` at node 1;
* ``"correct"``: :class:`~tests.studies.mtp_boundaries_common.CorrectOutcome` at node 2 and
  :class:`~tests.studies.mtp_boundaries_common.RichPseudo` at node 1 (``LTMLE``) or
  :class:`~tests.studies.mtp_boundaries_common.UnitRangePseudo` (``SDR``), the calibration
  cells' learners.

The ratio is the exact one (:func:`~tests.studies.mtp_boundaries_common.node_ratio`): the
studies read the oracle binned hazard, whose bins grow with n and whose limit it is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from tests.studies import mtp_boundaries_common as law
from tests.studies.mtp_pricing_own_curve import Fluctuation, _least_squares

Learners = Literal["primary", "correct"]
Estimator = Literal["ltmle", "sdr"]

#: ``LTMLE``'s default submodel bound, which the boundaries fits keep.
ALPHA = 0.9995
GLM_MAX_ITER = 100
GLM_TOL = 1e-10


@dataclass(frozen=True)
class Block:
    """Node-1 points with their inner node-2 points.

    Parameters
    ----------
    w1 : numpy.ndarray
        ``(R,)`` the node-1 points' ``W1``.
    w2 : numpy.ndarray
        ``(R,)`` their ``W2``.
    a1 : numpy.ndarray
        ``(R,)`` their ``A1``.
    mass : numpy.ndarray
        ``(R,)`` the probability of each node-1 point.
    l2 : numpy.ndarray
        ``(R, K)`` the inner points' ``L2``.
    a2 : numpy.ndarray
        ``(R, K)`` their ``A2``.
    inner : numpy.ndarray
        ``(R, K)`` each inner point's conditional probability given its node-1 point.
    y : numpy.ndarray
        ``(R, K)`` ``E[Y | H_2, A_2]`` (``Y`` itself on a sample).
    atom : bool
        Law B': the node-2 design holds ``1{A2 = 3}`` and the ratio reads the atom.
    ratios : dict or None
        ``label -> (r1, r2)`` read in place of the exact ratios (a check against a fit's own).
    """

    w1: np.ndarray
    w2: np.ndarray
    a1: np.ndarray
    mass: np.ndarray
    l2: np.ndarray
    a2: np.ndarray
    inner: np.ndarray
    y: np.ndarray
    atom: bool = False
    ratios: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def u(self) -> np.ndarray:
        return np.asarray(law.upper(self.w2), dtype=float)

    def ratio(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """``(r1, r2)`` of ``label`` at the observed doses, ``(R,)`` and ``(R, K)``."""
        if self.ratios is not None:
            return self.ratios[label]
        u = self.u
        r1 = law.node_ratio(label, 1, self.a1, law.mean1(self.w1, self.w2), u)
        mean2 = law.mean2(self.w1[:, None], self.w2[:, None], self.a1[:, None], self.l2)
        p = law.p_atom(self.w1[:, None], self.l2) if self.atom else None
        r2 = law.node_ratio(label, 2, self.a2, mean2, u[:, None], p)
        return np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)

    def dose1(self, label: str) -> np.ndarray:
        return np.asarray(law.MAPS[label][0](self.a1, self.u), dtype=float)

    def dose2(self, label: str) -> np.ndarray:
        return np.asarray(law.MAPS[label][1](self.a2, self.u[:, None]), dtype=float)

    def design1(self, dose: np.ndarray | None = None) -> np.ndarray:
        """``[W1, W2, A1]`` at the observed dose or at ``dose``."""
        return np.column_stack([self.w1, self.w2, self.a1 if dose is None else dose])

    def design2(self, dose: np.ndarray | None = None) -> np.ndarray:
        """``[W1, W2, L2, A1, A2]``, with ``1{A2 = 3}`` on law B', flattened to ``R K`` rows."""
        a2 = self.a2 if dose is None else dose
        shape = a2.shape
        blocks = [
            np.broadcast_to(self.w1[:, None], shape),
            np.broadcast_to(self.w2[:, None], shape),
            self.l2,
            np.broadcast_to(self.a1[:, None], shape),
            a2,
        ]
        if self.atom:
            blocks.append(np.abs(a2 - law.ATOM) <= 1e-12 * law.ATOM)
        return np.column_stack([np.ravel(b).astype(float) for b in blocks])

    @property
    def weight(self) -> np.ndarray:
        """``(R, K)`` the probability of each inner point."""
        return self.mass[:, None] * self.inner


# ----------------------------------------------------------------------------- learners


def _identity(design: np.ndarray) -> np.ndarray:
    return design


def _correct(design: np.ndarray) -> np.ndarray:
    return law.CorrectOutcome._design(design)


def _rich(design: np.ndarray) -> np.ndarray:
    return law.RichPseudo._design(design)


@dataclass
class Model:
    """A fitted least-squares or quasibinomial regression on a basis, with the intercept first.

    Parameters
    ----------
    kind : {"linear", "glm"}
        The family.
    basis : callable
        The regression's columns from the design, without the intercept.
    clip : bool
        Project each prediction onto ``[0, 1]`` (:class:`~law.UnitRangePseudo`).
    coef : numpy.ndarray
        The coefficients.
    """

    kind: Literal["linear", "glm"]
    basis: Callable[[np.ndarray], np.ndarray]
    clip: bool = False
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def features(self, design: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(design.shape[0]), self.basis(design)])

    def predict(self, design: np.ndarray) -> np.ndarray:
        eta = self.features(design) @ self.coef
        value = np.asarray(eta if self.kind == "linear" else expit(eta))
        return np.clip(value, 0.0, 1.0) if self.clip else value


def learners(kind: Learners, estimator: Estimator) -> tuple[Model, Model]:
    """``(node 1, node 2)`` declared learners, unfitted."""
    if kind == "primary":
        return Model("linear", _identity), Model("glm", _identity)
    return Model("linear", _rich, clip=estimator == "sdr"), Model("glm", _correct)


def fit_glm(
    model: Model, parts: Callable[[], Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]]
) -> Model:
    """``QuasiBinomialGLM``'s iterates on weighted ``(design, target, weight)`` parts."""
    total = mass = 0.0
    width = 0
    for design, target, weight in parts():
        total += float(np.sum(weight * target))
        mass += float(np.sum(weight))
        width = model.features(design[:1]).shape[1]
    mean = total / mass
    coef = np.zeros(width)
    coef[0] = np.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))

    def working(coef: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for design, target, weight in parts():
            features = model.features(design)
            eta = features @ coef
            fitted = expit(eta)
            variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
            yield features, eta + (target - fitted) / variance, weight * variance

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


def fit_linear(model: Model, design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> Model:
    model.coef = _least_squares([(model.features(design), target, weight)])
    return model


def fit_fluctuation(
    parts: Callable[[], Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    alpha: float = ALPHA,
    max_iter: int = 100,
) -> Fluctuation:
    """:meth:`Fluctuation.fit` over weighted ``(target, prediction, weight)`` parts."""
    steps: list[float] = []
    for _ in range(max_iter):
        current = Fluctuation(alpha, tuple(steps))
        eps = 0.0
        for _newton in range(50):
            gradient = hessian = total = 0.0
            for z, q, weight in parts():
                p = expit(logit(current.apply(q)) + eps)
                gradient += float(np.sum(weight * (z - p)))
                hessian += float(np.sum(weight * p * (1.0 - p)))
                total += float(np.sum(weight))
            if abs(gradient) / total <= 1e-12:
                break
            eps += gradient / hessian
        steps.append(eps)
        score = total = 0.0
        moved = Fluctuation(alpha, tuple(steps))
        for z, q, weight in parts():
            score += float(np.sum(weight * (z - moved.apply(q))))
            total += float(np.sum(weight))
        if abs(score) / total <= 1e-13 or abs(eps) <= 1e-13:
            break
    return Fluctuation(alpha, tuple(steps))


# ----------------------------------------------------------------------------- the fit


@dataclass
class PlanFit:
    """One plan's limit nuisances.

    Parameters
    ----------
    label : str
        The plan.
    node1 : Model
        The node-1 regression.
    fluctuations : tuple
        ``LTMLE``'s ``(node 1, node 2)`` fluctuations, or ``(None, None)`` under ``SDR``.
    psi : float
        The plan's value under the limit nuisances.
    """

    label: str
    node1: Model
    fluctuations: tuple[Fluctuation | None, Fluctuation | None]
    psi: float


@dataclass
class OwnCurve:
    """The own curve of one estimator and one set of learners on law B or B'.

    Parameters
    ----------
    estimator : {"ltmle", "sdr"}
        The recursion.
    learners : {"primary", "correct"}
        The declared learners.
    labels : tuple of str
        The plans, ``natural`` first.
    """

    estimator: Estimator
    learners: Learners
    labels: tuple[str, ...]
    node2: Model | None = None
    plans: dict[str, PlanFit] = field(default_factory=dict)

    def _q2(self, block: Block, dose: np.ndarray | None = None) -> np.ndarray:
        assert self.node2 is not None
        q = self.node2.predict(block.design2(dose)).reshape(block.a2.shape)
        return np.clip(q, 0.0, 1.0) if self.estimator == "ltmle" else q

    def _q1(self, plan: PlanFit, block: Block, dose: np.ndarray | None = None) -> np.ndarray:
        q = plan.node1.predict(block.design1(dose))
        return np.clip(q, 0.0, 1.0) if self.estimator == "ltmle" else q

    def fit(self, blocks: Callable[[], Iterable[Block]]) -> OwnCurve:
        """Solve every regression's population equations over ``blocks()``."""
        node1_kind, node2 = learners(self.learners, self.estimator)
        self.node2 = fit_glm(
            node2,
            lambda: (
                (block.design2(), np.ravel(block.y), np.ravel(block.weight)) for block in blocks()
            ),
        )
        for label in self.labels:
            self.plans[label] = self._fit_plan(label, blocks, node1_kind)
        return self

    def _fit_plan(self, label: str, blocks: Callable[[], Iterable[Block]], kind: Model) -> PlanFit:
        node1 = Model(kind.kind, kind.basis, kind.clip)
        fluct2 = None
        if self.estimator == "ltmle":

            def parts2() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
                for block in blocks():
                    r1, r2 = block.ratio(label)
                    yield block.y, self._q2(block), block.weight * r1[:, None] * r2

            fluct2 = fit_fluctuation(parts2)
        designs, targets, masses, ratios1 = [], [], [], []
        for block in blocks():
            r1, r2 = block.ratio(label)
            v2 = self._q2(block, block.dose2(label))
            # LTMLE carries the targeted value; SDR the transform phi_2.
            z2 = fluct2.apply(v2) if fluct2 is not None else v2 + r2 * (block.y - self._q2(block))
            designs.append(block.design1())
            targets.append(np.sum(block.inner * z2, axis=1) / np.sum(block.inner, axis=1))
            masses.append(block.mass * np.sum(block.inner, axis=1))
            ratios1.append(r1)
        design, target, mass, r1 = (
            np.concatenate(designs),
            np.concatenate(targets),
            np.concatenate(masses),
            np.concatenate(ratios1),
        )
        fit_linear(node1, design, target, mass)
        fluct1 = None
        q1 = node1.predict(design)
        if self.estimator == "ltmle":
            q1 = np.clip(q1, 0.0, 1.0)
            fluct1 = Fluctuation.fit(target, q1, mass * r1, ALPHA)
        plan = PlanFit(label, node1, (fluct1, fluct2), 0.0)
        value = np.concatenate([self._value1(plan, block) for block in blocks()])
        if self.estimator == "sdr":
            # psi is the mean of phi_1, whose node-1 residual a least-squares fit does not zero
            # under the ratio weight.
            value = value + r1 * (target - q1)
        plan.psi = float(np.sum(value * mass) / np.sum(mass))
        return plan

    def _value1(self, plan: PlanFit, block: Block) -> np.ndarray:
        v1 = self._q1(plan, block, block.dose1(plan.label))
        fluct1 = plan.fluctuations[0]
        return v1 if fluct1 is None else fluct1.apply(v1)

    def parts(self, plan: PlanFit, block: Block) -> tuple[np.ndarray, np.ndarray]:
        """``D = base + slope Y`` on the block's inner points: ``(base, slope)``."""
        r1, r2 = block.ratio(plan.label)
        value1 = self._value1(plan, block)
        fluct1, fluct2 = plan.fluctuations
        q1 = self._q1(plan, block)
        q2 = self._q2(block)
        v2 = self._q2(block, block.dose2(plan.label))
        if self.estimator == "ltmle":
            assert fluct1 is not None and fluct2 is not None
            z2, q2s, q1s = fluct2.apply(v2), fluct2.apply(q2), fluct1.apply(q1)
            base = (value1 - plan.psi - r1 * q1s)[:, None] + r1[:, None] * (z2 - r2 * q2s)
            return base, r1[:, None] * r2
        base = (value1 - plan.psi - r1 * q1)[:, None] + r1[:, None] * (v2 - r2 * q2)
        return base, r1[:, None] * r2

    def curve(self, label: str, block: Block) -> np.ndarray:
        """``D`` at a sample block, whose ``y`` holds the observed outcome."""
        base, slope = self.parts(self.plans[label], block)
        return base + slope * block.y


def sample_block(frame: pd.DataFrame, atom: bool = False, ratios: dict | None = None) -> Block:
    """A sample as one block: each row a node-1 point of mass ``1 / n`` with its own inner point."""

    def get(name: str) -> np.ndarray:
        return np.asarray(frame[name], dtype=float)

    n = len(get("W1"))
    return Block(
        w1=get("W1"),
        w2=get("W2"),
        a1=get("A1"),
        mass=np.full(n, 1.0 / n),
        l2=get("L2")[:, None],
        a2=get("A2")[:, None],
        inner=np.ones((n, 1)),
        y=get("Y")[:, None],
        atom=atom,
        ratios=ratios,
    )
