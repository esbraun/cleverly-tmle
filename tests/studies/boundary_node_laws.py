r"""The laws of ``boundary-node-ltmle``: a survival curve whose last node is often flagged.

Baseline :math:`W \sim N(0, 1)`. Three treatment decisions, before event nodes 1, 3 and 4, each
with the known mechanism :math:`g_k(W) = \mathrm{expit}(\gamma_k + 0.3W)`; the panel carries a
fourth treatment column ``A2`` that copies ``A1``, so that every event node has a treatment
column, with the known probability :math:`P(A_2 = 1 \mid A_1) = A_1`. The decision that node
:math:`t` reads is :math:`k(t) = (1, 1, 2, 3)`. A discrete-time event process over four nodes has
the hazard

.. math::

    h_t(W, \bar A) = \mathrm{expit}(\alpha_t + 0.4W - 0.5A_{k(t)} + b),

with no censoring, and :math:`b = 0` except in the clustered cell, where :math:`b \sim
N(0, \sigma_b^2)` is shared by the ten rows of a cluster. :math:`\gamma` sets the follow
probabilities of ``always`` to :math:`(0.5, 0.5, 0.25)`, so the third decision shrinks node 4's
risk set under ``always`` to about :math:`n/16`: node 4 is flagged often, while it carries most
of the curve's variance.

Everything the design test checks is exact: the flag probability of each node at ``n`` is
:math:`(1-q_1)^n + (1-q_0)^n - 2(1-p)^n`, with :math:`p` the probability that a unit sits in the
node's risk set (it followed the regimen and had no earlier event), :math:`q_1` that it does so
and has the event there, and :math:`q_0 = p - q_1` (no event among the node's rows, or only
events, and the risk set not empty). Each probability is a Gauss-Hermite integral over
:math:`W` (and :math:`b`). So is the truth, the risk at horizon 4 under each static regimen, and
the variance share of each node's term in the efficient influence curve.

:data:`ALPHA_GRID` is declared before any pilot, and :func:`chosen_alpha` picks from it by a rule
fixed with it: among the points that meet the three design constraints, the one whose node 4
flag probability under ``always`` at :math:`n = 200` is closest to 0.5.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

#: The mechanism's slope on :math:`W`, and the hazard's on :math:`W` and on the decision.
MECHANISM_SLOPE: Final[float] = 0.3
HAZARD_SLOPE: Final[float] = 0.4
TREATMENT_EFFECT: Final[float] = -0.5
#: The marginal follow probabilities of ``always`` at the three decisions.
FOLLOW: Final[tuple[float, float, float]] = (0.5, 0.5, 0.25)
#: The decision each event node reads, one-based.
DECISION_OF_NODE: Final[tuple[int, int, int, int]] = (1, 1, 2, 3)
#: The SD of the shared cluster effect on the hazard's logit in the clustered cell.
CLUSTER_SD: Final[float] = 0.5
NODES: Final[int] = 4
HORIZON: Final[int] = 4
REGIMENS: Final[dict[str, int]] = {"always": 1, "never": 0}
#: Gauss-Hermite nodes per normal integral.
QUADRATURE: Final[int] = 80

#: The declared grid of node hazards at :math:`W = 0, A = 0`, from which :math:`\alpha_t =
#: \mathrm{logit}` is taken: nodes 1 to 3 share one value, node 4 takes its own.
EARLY_HAZARDS: Final[tuple[float, ...]] = (0.06, 0.08, 0.10, 0.12)
LAST_HAZARDS: Final[tuple[float, ...]] = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
ALPHA_GRID: Final[tuple[tuple[float, float, float, float], ...]] = tuple(
    (float(logit(early)),) * 3 + (float(logit(last)),)
    for early, last in itertools.product(EARLY_HAZARDS, LAST_HAZARDS)
)

#: The three design constraints, at :data:`DESIGN_N`.
DESIGN_N: Final[int] = 200
MINIMUM_LAST_SHARE: Final[float] = 0.4
LAST_FLAG_RANGE: Final[tuple[float, float]] = (0.3, 0.7)
MAXIMUM_EARLY_FLAG: Final[float] = 0.10


def _gauss_hermite(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of :math:`E[f(Z)]`, :math:`Z \\sim N(0, 1)`."""
    points, weights = np.polynomial.hermite_e.hermegauss(size)
    return points, weights / weights.sum()


def _gamma(follow: float) -> float:
    """The intercept whose mechanism has marginal mean ``follow`` over :math:`W`."""
    points, weights = _gauss_hermite(QUADRATURE)
    return float(
        brentq(lambda g: float(weights @ expit(g + MECHANISM_SLOPE * points)) - follow, -20.0, 20.0)
    )


GAMMA: Final[tuple[float, float, float]] = tuple(_gamma(p) for p in FOLLOW)  # type: ignore[assignment]


@dataclass(frozen=True)
class Law:
    """One law of the study.

    Parameters
    ----------
    alpha : tuple of float
        :math:`\\alpha_1, \\dots, \\alpha_4`.
    zero : bool
        Whether node 4's hazard is exactly 0 for a unit treated at the third decision (the
        ``zero`` variant): ``always`` then has a true hazard of 0 at node 4.
    cluster_sd : float
        :math:`\\sigma_b`, 0 off the clustered cell.
    """

    alpha: tuple[float, float, float, float]
    zero: bool = False
    cluster_sd: float = 0.0

    def mechanism(self, decision: int, w: np.ndarray) -> np.ndarray:
        """:math:`g_k(W) = P(A = 1 \\mid W)` at decision ``k`` (one-based)."""
        return np.asarray(expit(GAMMA[decision - 1] + MECHANISM_SLOPE * w), dtype=float)

    def hazard(
        self, node: int, w: np.ndarray, a: np.ndarray, b: np.ndarray | float = 0.0
    ) -> np.ndarray:
        """:math:`h_t(W, A, b)` at node ``t`` (one-based), with ``a`` the decision it reads."""
        value = expit(self.alpha[node - 1] + HAZARD_SLOPE * w + TREATMENT_EFFECT * a + b)
        if self.zero and node == NODES:
            value = np.where(np.asarray(a) == 1, 0.0, value)
        return np.asarray(value, dtype=float)

    # ------------------------------------------------------------------------------ draws

    def draw(
        self, n: int, rng: np.random.Generator, cluster_size: int | None = None
    ) -> pd.DataFrame:
        """One wide panel of ``n`` rows, with ``cluster`` when ``cluster_size`` is set.

        Columns: ``W``, ``A1`` to ``A4``, ``Y1`` to ``Y4`` (missing after the event), and
        ``p1`` to ``p4``, the known :math:`P(A_t = 1 \\mid \\text{past})` (``p2`` is ``A1``).
        """
        w = rng.normal(size=n)
        if cluster_size is None:
            b = np.zeros(n)
        else:
            clusters = -(-n // cluster_size)
            b = np.repeat(rng.normal(scale=self.cluster_sd, size=clusters), cluster_size)[:n]
        decisions = [rng.binomial(1, self.mechanism(k, w)) for k in (1, 2, 3)]
        frame = pd.DataFrame({"W": w})
        at_risk = np.ones(n, dtype=bool)
        for node in range(1, NODES + 1):
            k = DECISION_OF_NODE[node - 1]
            a = decisions[k - 1]
            frame[f"A{node}"] = np.where(at_risk, a, np.nan).astype(float)
            frame[f"p{node}"] = a.astype(float) if node == 2 else self.mechanism(k, w)
            event = rng.binomial(1, self.hazard(node, w, a, b)).astype(float)
            frame[f"Y{node}"] = np.where(at_risk, event, np.nan)
            at_risk &= event == 0
        if cluster_size is not None:
            frame["cluster"] = np.arange(n) // cluster_size
        return frame

    # ---------------------------------------------------------------------- exact moments

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The quadrature points of :math:`(W, b)` and their weights, flattened."""
        w, weight = _gauss_hermite(QUADRATURE)
        if self.cluster_sd == 0.0:
            return w, np.zeros_like(w), weight
        z, zweight = _gauss_hermite(QUADRATURE // 2)
        ww, zz = np.meshgrid(w, z, indexing="ij")
        return ww.ravel(), self.cluster_sd * zz.ravel(), np.outer(weight, zweight).ravel()

    def _paths(self, regimen: int) -> dict[str, np.ndarray]:
        """Per quadrature point: cumulative follow, survival before each node, and hazards."""
        w, b, weight = self._grid()
        a = np.full_like(w, float(regimen))
        follow = [
            self.mechanism(k, w) if regimen == 1 else 1.0 - self.mechanism(k, w) for k in (1, 2, 3)
        ]
        cumulative, survival, hazards = [], [], []
        before = np.ones_like(w)
        for node in range(1, NODES + 1):
            k = DECISION_OF_NODE[node - 1]
            pi = np.prod([follow[j] for j in range(k)], axis=0)
            h = self.hazard(node, w, a, b)
            cumulative.append(pi)
            survival.append(before.copy())
            hazards.append(h)
            before = before * (1.0 - h)
        return {
            "weight": weight,
            "cumulative": np.array(cumulative),
            "survival": np.array(survival),
            "hazard": np.array(hazards),
            "end": before,
        }

    def risk(self, regimen: int) -> float:
        """The risk at horizon 4 under the static ``regimen``."""
        paths = self._paths(regimen)
        return float(paths["weight"] @ (1.0 - paths["end"]))

    def truths(self) -> dict[str, float]:
        """The risk of each regimen at horizon 4, and their contrast."""
        always, never = self.risk(1), self.risk(0)
        return {"always": always, "never": never, "contrast": always - never}

    def membership(self, node: int, regimen: int) -> tuple[float, float]:
        """:math:`(p, q_1)`: a unit is in node ``t``'s risk set, and has the event there.

        Unclustered: the risk set of the regimen's node regression is the rows that followed it
        through the node's decision and had no earlier event. In the clustered cell the units
        are clusters and these are per-row probabilities, which the flag probability does not
        use.
        """
        paths = self._paths(regimen)
        i = node - 1
        p = paths["cumulative"][i] * paths["survival"][i]
        return float(paths["weight"] @ p), float(paths["weight"] @ (p * paths["hazard"][i]))

    def flag_probability(self, node: int, regimen: int, n: int) -> float:
        """The probability that node ``t``'s risk set is not empty and holds one value.

        :math:`(1-q_1)^n + (1-q_0)^n - 2(1-p)^n`, for ``n`` independent rows.
        """
        p, q1 = self.membership(node, regimen)
        q0 = p - q1
        return float((1.0 - q1) ** n + (1.0 - q0) ** n - 2.0 * (1.0 - p) ** n)

    def node_variances(self, regimen: int) -> tuple[np.ndarray, float]:
        """The variance of each node's term of the risk's efficient curve, and the baseline's.

        Node ``t``'s term of :math:`1 - S_4` under a static regimen with a known mechanism is
        :math:`1\\{R_t\\}\\,(\\pi_t S_{t-1}(W))^{-1} S_4(W) / (1 - h_t(W))\\,(Y_t - h_t(W))`, with :math:`R_t`
        the risk-set indicator and :math:`\\pi_t` the cumulative follow probability, whose
        variance is :math:`E[S_4^2 h_t / ((1-h_t)\\,\\pi_t S_{t-1})]`. The baseline term is
        :math:`\\mathrm{Var}_W(S_4(W))`. Unclustered only.
        """
        if self.cluster_sd != 0.0:
            raise ValueError("the node variance shares are defined for the unclustered law")
        paths = self._paths(regimen)
        end, weight = paths["end"], paths["weight"]
        out = np.zeros(NODES)
        for i in range(NODES):
            h, pi, before = paths["hazard"][i], paths["cumulative"][i], paths["survival"][i]
            out[i] = float(weight @ (end**2 * h / ((1.0 - h) * pi * before)))
        baseline = float(weight @ end**2 - (weight @ end) ** 2)
        return out, baseline

    def last_share(self, regimen: int = 1) -> float:
        """Node 4's share of the risk's efficient variance when node 4 is not flagged."""
        nodes, baseline = self.node_variances(regimen)
        return float(nodes[-1] / (nodes.sum() + baseline))


def meets_constraints(law: Law, n: int = DESIGN_N) -> dict[str, bool]:
    """The three design constraints of the interior law at ``n``."""
    flag = law.flag_probability(NODES, 1, n)
    early = max(law.flag_probability(t, r, n) for t in (1, 2, 3) for r in (0, 1))
    return {
        "last_share": law.last_share() >= MINIMUM_LAST_SHARE,
        "last_flag": LAST_FLAG_RANGE[0] <= flag <= LAST_FLAG_RANGE[1],
        "early_flag": early <= MAXIMUM_EARLY_FLAG,
    }


@cache
def chosen_alpha() -> tuple[float, float, float, float]:
    """The declared grid point: constraints met, node 4 flagged closest to half the time."""
    feasible = [alpha for alpha in ALPHA_GRID if all(meets_constraints(Law(alpha)).values())]
    if not feasible:  # pragma: no cover - the design test pins a feasible grid
        raise ValueError("no point of the declared grid meets the design constraints")
    return min(
        feasible, key=lambda alpha: abs(Law(alpha).flag_probability(NODES, 1, DESIGN_N) - 0.5)
    )


def law_of(variant: str) -> Law:
    """``interior``, ``clustered`` or ``zero``, at the chosen :math:`\\alpha`."""
    base = Law(chosen_alpha())
    if variant == "interior":
        return base
    if variant == "clustered":
        return replace(base, cluster_sd=CLUSTER_SD)
    if variant == "zero":
        return replace(base, zero=True)
    raise KeyError(variant)


def oracle_spread(law: Law, frame: pd.DataFrame, node: int, regimen: int) -> np.ndarray:
    """Each row's true residual SD at ``node`` under ``regimen``, :math:`\\sqrt{h(1-h)}`.

    The clustered law's effect :math:`b` is latent, so its hazard is the mean over :math:`b`
    given :math:`W`.
    """
    w = frame["W"].to_numpy(dtype=float)
    a = np.full_like(w, float(regimen))
    if law.cluster_sd == 0.0:
        h = law.hazard(node, w, a)
    else:
        z, weight = _gauss_hermite(QUADRATURE // 2)
        h = np.sum(
            weight[:, None] * law.hazard(node, w[None, :], a[None, :], law.cluster_sd * z[:, None]),
            axis=0,
        )
    return np.sqrt(h * (1.0 - h))


def summary(law: Law, n: int) -> Mapping[str, float]:
    """The law's design figures at ``n``: flag probabilities, share and truths."""
    out: dict[str, float] = {
        f"flag_always_node{t}": law.flag_probability(t, 1, n) for t in range(1, 5)
    }
    out.update({f"flag_never_node{t}": law.flag_probability(t, 0, n) for t in range(1, 5)})
    if law.cluster_sd == 0.0:
        out["last_share"] = law.last_share()
    out.update({f"truth_{k}": v for k, v in law.truths().items()})
    return out
