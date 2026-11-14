r"""The exact laws, regimens, truths and efficiency bounds of ``hazard-multinode``.

**The laws.**  :math:`W_1 \in \{0, 1\}` and :math:`W_2 \in \{0, \dots, 3\}` are independent and
uniform.  A binary treatment :math:`A_k` is decided at every node :math:`k = 1, \dots, T`, among
the units still at risk, with

.. math::

    P(A_1 = 1 \mid W) = \operatorname{expit}(\alpha_0 + \alpha_1 W_1 + \alpha_2 W_2), \qquad
    P(A_k = 1 \mid W, \bar A_{k-1}) = \operatorname{expit}(\beta_0 + \beta_1 W_1 + \beta_2 W_2
        + \beta_3 A_{k-1}),

then the unit stays in follow-up with probability
:math:`1 - \operatorname{expit}(c_0 + c_1 W_1 + c_2 W_2 + c_3 A_k + c_4 A_{k-1})` and, if it
stays, has the event with probability

.. math::

    \lambda(k \mid \bar A_k, W) = \operatorname{expit}\bigl(h_0 + h_t (k - 1) + h_1 W_1
        + h_2 W_2 + h_3 A_k + h_4 A_{k-1}\bigr),

with :math:`A_0 = 0`.  This is the base law of plan X70 section 4.3: the treatment depends on
:math:`(W, A_{k-1})`, dropout on :math:`(W, \bar A_k)` and the hazard on
:math:`(W, A_k, A_{k-1})`.  A fit hands its learners the history ``[W1, W2, A_1, ..., A_k]``
(the treatment learner ``[W1, W2, A_1, ..., A_{k-1}]``) at each node, so an unpenalised
logistic regression on those columns is correctly specified and not saturated: the factors
leave every earlier arm but the last out, and every interaction out.  :data:`BASE` has three
nodes and :data:`FOUR` has four, for the policy whose 16 paths the fit enumerates; the four-node
law has the same form with its own coefficients (:data:`FOUR_MECHANISM`, :data:`FOUR_HAZARD`),
so that every treatment prefix keeps units at risk and events at n = 2,000.

**The regimens** (:data:`REGIMENS`) are the static arms ``always`` and ``never``, the known
rule ``rule`` (treat at every node when :math:`W_2 \ge 2`), the known policy ``policy``
(treat with probability :math:`0.3 + 0.4 A_{k-1} + 0.1 W_1`, :math:`A_0 = 0`), and ``stay``
(treat at node 1 with probability :math:`0.3 + 0.2 W_1`, then keep the previous arm), which the
shipped plan resolution would collapse at the observed arms.  :func:`static_paths` lists the
eight static paths of three nodes, which the working-model cell reads.

**Truths.**  :meth:`Law.risk` enumerates the paths: :math:`\psi_r(t) = \sum_W P(W) \sum_u
q_r(u \mid W) F(t \mid u, W)`, with :math:`F` built from the law's own hazards.  The working
model's coefficients are the projection of those cell risks by Newton's method
(:func:`exact_beta`).  Nothing here calls the estimator.

**Efficiency bounds.**  :meth:`Law.curve` evaluates the efficient influence function at one
observed outcome in its iterated-mean form,

.. math::

    D_{r,t} = V_1(W) - \psi_r(t) + \sum_{k \le t} \frac{\prod_{s \le k} q_{r,s}(A_s \mid
        \bar A_{s-1}, W)}{\prod_{s \le k} g_s(A_s \mid \bar A_{s-1}, W)\, c_s(\bar A_s, W)}
        R_k \bigl(Z_k - Q_k(\bar A_k, W)\bigr),

with :math:`V_k(\bar a_{k-1}, W) = \sum_a q_{r,k}(a \mid \bar a_{k-1}, W) Q_k(\bar a_{k-1} a, W)`,
:math:`Q_k = \lambda_k + (1 - \lambda_k) V_{k+1}`, :math:`V_{t+1} = 0`, :math:`R_k` the
indicator of being at risk and uncensored at node :math:`k`, and :math:`Z_k` the event
indicator at an event and :math:`V_{k+1}(\bar A_k, W)` otherwise.  This is the sequential
regression form (Bang and Robins 2005) and shares nothing with the estimator's hazard form; the
design test checks the two agree through an exact-sample fit.  :func:`efficiency_sd`
enumerates every support point of the observed law.

**The wrong hazard learner** (:class:`LastArmLogistic`) regresses each node's event on
:math:`A_k` alone: no :math:`A_{k-1}` and no :math:`W`, the wrong learner of plan X70
section 5.  On a binary :math:`A_k` it is the at-risk event share of each arm, so its limit
(:func:`wrong_hazard_limit`) is an exact weighted mean over the observed law.  **The node-only
hazard learner** (the prior share at each node, :func:`node_hazards`) reads no arm at all, so
its plug-in contrast of two regimens is zero in every sample (:func:`node_hazard_limit`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.interventions import Stochastic
from cleverly.longitudinal import DynamicRegimen
from cleverly.msm import MSM

#: The baseline strata and their mass.
STRATA = tuple((w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3))
MASS = 0.125
#: The regimen labels of the primary fit, in report order.
LABELS = ("always", "never", "rule", "policy", "stay")
#: The working model's terms, in design-column order.
TERMS = ("(intercept)", "dose", "t", "dose_W1")


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ------------------------------------------------------------------ the regimens, law side


def treat_probability(label: str, node: int, w1: int, w2: int, previous: int) -> float:
    """The probability that regimen ``label`` treats at ``node``, written from its definition.

    A static path label ``pXYZ`` treats at node ``k`` when its ``k``-th digit is one.
    """
    if label == "always":
        return 1.0
    if label == "never":
        return 0.0
    if label == "rule":
        return float(w2 >= 2)
    if label == "policy":
        return 0.3 + 0.4 * previous + 0.1 * w1
    if label == "stay":
        return 0.3 + 0.2 * w1 if node == 1 else float(previous)
    if label.startswith("p"):
        return float(label[node] == "1")
    raise KeyError(label)


def static_paths(n_times: int = 3) -> tuple[str, ...]:
    """The static path labels ``pXYZ`` of ``n_times`` nodes, in binary order."""
    return tuple(
        "p" + "".join(str(a) for a in path) for path in itertools.product((0, 1), repeat=n_times)
    )


def dose(label: str, horizon: int) -> float:
    """The cumulative dose of a static path through ``horizon``: its treated nodes."""
    return float(sum(int(digit) for digit in label[1 : horizon + 1]))


# ------------------------------------------------------------------ the law


@dataclass(frozen=True)
class Law:
    """A finite law with a binary treatment at every node, dropout and one event.

    Parameters
    ----------
    name : str
        The law's label.
    n_times : int
        The nodes.
    first : tuple of float
        :math:`(\\alpha_0, \\alpha_1, \\alpha_2)` of :math:`P(A_1 = 1 \\mid W)`.
    later : tuple of float
        :math:`(\\beta_0, \\beta_1, \\beta_2, \\beta_3)` of :math:`P(A_k = 1 \\mid W, A_{k-1})`.
    dropout : tuple of float
        :math:`(c_0, c_1, c_2, c_3, c_4)` of the dropout probability on
        ``(1, W1, W2, A_k, A_{k-1})``.
    hazard : tuple of float
        :math:`(h_0, h_t, h_1, h_2, h_3, h_4)` of the event hazard on
        ``(1, k - 1, W1, W2, A_k, A_{k-1})``.
    """

    name: str
    n_times: int
    first: tuple[float, float, float]
    later: tuple[float, float, float, float]
    dropout: tuple[float, float, float, float, float]
    hazard: tuple[float, float, float, float, float, float]
    _support: list[tuple[tuple[float, ...], float]] = field(
        init=False, repr=False, compare=False, default_factory=list
    )

    # -------------------------------------------------------------- factors

    def treated(self, node: int, w1: Any, w2: Any, previous: Any) -> Any:
        """:math:`P(A_k = 1 \\mid W, A_{k-1})`."""
        x1, x2, prev = (np.asarray(value, dtype=float) for value in (w1, w2, previous))
        if node == 1:
            a0, a1, a2 = self.first
            return _expit(a0 + a1 * x1 + a2 * x2)
        b0, b1, b2, b3 = self.later
        return _expit(b0 + b1 * x1 + b2 * x2 + b3 * prev)

    def g(self, node: int, a: int, w1: int, w2: int, previous: int) -> float:
        p = float(self.treated(node, w1, w2, previous))
        return p if a == 1 else 1.0 - p

    def lost(self, w1: Any, w2: Any, a: Any, previous: Any) -> Any:
        """The dropout probability at a node, after its treatment and before its event."""
        c0, c1, c2, c3, c4 = self.dropout
        x1, x2, arm, prev = (np.asarray(value, dtype=float) for value in (w1, w2, a, previous))
        return _expit(c0 + c1 * x1 + c2 * x2 + c3 * arm + c4 * prev)

    def retained(self, w1: int, w2: int, a: int, previous: int) -> float:
        return 1.0 - float(self.lost(w1, w2, a, previous))

    def event(self, node: int, w1: Any, w2: Any, a: Any, previous: Any) -> Any:
        """:math:`\\lambda(k \\mid \\bar A_k, W)`, the event hazard among the units at risk."""
        h0, ht, h1, h2, h3, h4 = self.hazard
        x1, x2, arm, prev = (np.asarray(value, dtype=float) for value in (w1, w2, a, previous))
        return _expit(h0 + ht * (node - 1) + h1 * x1 + h2 * x2 + h3 * arm + h4 * prev)

    # -------------------------------------------------------------- truths

    def path_risk(self, path: Sequence[int], w1: int, w2: int, horizon: int) -> float:
        """:math:`F(t \\mid u, W)` of one treatment path."""
        survival, risk, previous = 1.0, 0.0, 0
        for node in range(1, horizon + 1):
            lam = float(self.event(node, w1, w2, path[node - 1], previous))
            risk += survival * lam
            survival *= 1.0 - lam
            previous = path[node - 1]
        return risk

    def path_mass(self, label: str, path: Sequence[int], w1: int, w2: int) -> float:
        """:math:`q_r(u \\mid W)` through the path's length."""
        mass, previous = 1.0, 0
        for node, a in enumerate(path, start=1):
            p = treat_probability(label, node, w1, w2, previous)
            mass *= p if a == 1 else 1.0 - p
            previous = a
        return mass

    def risk(self, label: str, horizon: int) -> float:
        """:math:`\\psi_r(t)` by enumerating the paths through ``horizon``."""
        total = 0.0
        for w1, w2 in STRATA:
            for path in itertools.product((0, 1), repeat=horizon):
                mass = self.path_mass(label, path, w1, w2)
                if mass > 0.0:
                    total += MASS * mass * self.path_risk(path, w1, w2, horizon)
        return total

    # -------------------------------------------------------------- the observed law

    def support(self) -> list[tuple[tuple[float, ...], float]]:
        """Every observed outcome and its probability, as wide rows.

        A row is ``(W1, W2, A1, C1, Y1, ..., AT, CT, YT)``, with ``nan`` after the unit
        leaves: ``C_k = 0`` at a dropout (``Y_k`` missing) and ``Y_k = 1`` at an event.
        """
        if self._support:
            return self._support
        width = 2 + 3 * self.n_times
        out: list[tuple[tuple[float, ...], float]] = []

        def walk(prefix: tuple[float, ...], mass: float, node: int, previous: int) -> None:
            if node > self.n_times:
                out.append((prefix, mass))
                return
            w1, w2 = int(prefix[0]), int(prefix[1])
            for a in (1, 0):
                pa = self.g(node, a, w1, w2, previous)
                c = self.retained(w1, w2, a, previous)
                here = mass * pa
                pad = (np.nan,) * (width - len(prefix) - 2)
                out.append(((*prefix, float(a), 0.0, *pad), here * (1.0 - c)))
                lam = float(self.event(node, w1, w2, a, previous))
                pad = (np.nan,) * (width - len(prefix) - 3)
                out.append(((*prefix, float(a), 1.0, 1.0, *pad), here * c * lam))
                walk((*prefix, float(a), 1.0, 0.0), here * c * (1.0 - lam), node + 1, a)

        for w1, w2 in STRATA:
            walk((float(w1), float(w2)), MASS, 1, 0)
        total = sum(mass for _, mass in out)
        if abs(total - 1.0) > 1e-12:  # pragma: no cover - guards the enumeration
            raise AssertionError(f"the support sums to {total}")
        self._support.extend(out)
        return self._support

    def columns(self) -> list[str]:
        out = ["W1", "W2"]
        for node in range(1, self.n_times + 1):
            out += [f"A{node}", f"C{node}", f"Y{node}"]
        return out

    def exact_frame(self) -> pd.DataFrame:
        """The law laid out as weighted wide rows; ``wt`` is each outcome's mass."""
        points = self.support()
        frame = pd.DataFrame([row for row, _ in points], columns=self.columns())
        frame["wt"] = np.array([mass for _, mass in points])
        return frame

    def fit_columns(self, *, weights: str | None = None) -> dict[str, Any]:
        """The ``LTMLE.fit`` keywords of a wide frame of this law."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{k}" for k in nodes],
            "outcome": [f"Y{k}" for k in nodes],
            "censoring": [f"C{k}" for k in nodes],
            "baseline": ["W1", "W2"],
        }
        if weights is not None:
            out["weights"] = weights
        return out

    def draw(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` units as wide rows, coded as :meth:`support` codes them."""
        rng = np.random.default_rng(seed)
        w1 = rng.integers(0, 2, size=n).astype(float)
        w2 = rng.integers(0, 4, size=n).astype(float)
        out: dict[str, np.ndarray] = {"W1": w1, "W2": w2}
        open_ = np.ones(n, dtype=bool)
        previous = np.zeros(n)
        for node in range(1, self.n_times + 1):
            a = (rng.random(n) < self.treated(node, w1, w2, previous)).astype(float)
            stays = rng.random(n) >= self.lost(w1, w2, a, previous)
            fires = rng.random(n) < self.event(node, w1, w2, a, previous)
            out[f"A{node}"] = np.where(open_, a, np.nan)
            out[f"C{node}"] = np.where(open_, stays.astype(float), np.nan)
            out[f"Y{node}"] = np.where(open_ & stays, fires.astype(float), np.nan)
            open_ = open_ & stays & ~fires
            previous = a
        return pd.DataFrame(out)

    # -------------------------------------------------------------- the efficient curve

    def _value(self, label: str, history: tuple[int, ...], w1: int, w2: int, horizon: int) -> float:
        """:math:`V_k(\\bar a_{k-1}, W)` with ``k = len(history) + 1``; zero past ``horizon``."""
        node = len(history) + 1
        if node > horizon:
            return 0.0
        previous = history[-1] if history else 0
        p = treat_probability(label, node, w1, w2, previous)
        total = 0.0
        for a, share in ((1, p), (0, 1.0 - p)):
            if share > 0.0:
                total += share * self._regression(label, (*history, a), w1, w2, horizon)
        return total

    def _regression(
        self, label: str, history: tuple[int, ...], w1: int, w2: int, horizon: int
    ) -> float:
        """:math:`Q_k(\\bar a_k, W) = \\lambda_k + (1 - \\lambda_k) V_{k+1}`."""
        node = len(history)
        previous = history[-2] if node > 1 else 0
        lam = float(self.event(node, w1, w2, history[-1], previous))
        return lam + (1.0 - lam) * self._value(label, history, w1, w2, horizon)

    def curve(self, label: str, horizon: int, row: Sequence[float]) -> float:
        """The uncentred efficient influence curve :math:`D_{r,t} + \\psi_r(t)` at one row."""
        w1, w2 = int(row[0]), int(row[1])
        total = self._value(label, (), w1, w2, horizon)
        weight = 1.0
        history: tuple[int, ...] = ()
        for node in range(1, horizon + 1):
            base = 2 + 3 * (node - 1)
            if np.isnan(row[base]):
                break
            a = int(row[base])
            previous = history[-1] if history else 0
            q = treat_probability(label, node, w1, w2, previous)
            weight *= (q if a == 1 else 1.0 - q) / self.g(node, a, w1, w2, previous)
            if row[base + 1] == 0.0:
                break  # dropped out before the node's event
            weight /= self.retained(w1, w2, a, previous)
            history = (*history, a)
            regression = self._regression(label, history, w1, w2, horizon)
            y = row[base + 2]
            z = 1.0 if y == 1.0 else self._value(label, history, w1, w2, horizon)
            total += weight * (z - regression)
            if y == 1.0:
                break
        return total


# ------------------------------------------------------------------ the declared laws

#: The treatment and dropout mechanism of the three-node law.  :math:`P(A_1 = 1 \mid W)` runs from 0.21
#: to 0.75 and later treatment depends on the previous arm, so the arm is confounded by ``W``
#: and by the earlier arm.  Dropout runs from about 0.03 to 0.10 per node.
MECHANISM: dict[str, Any] = {
    "first": (-1.3, 0.4, 0.55),
    "later": (-1.4, 0.3, 0.45, 1.3),
    "dropout": (-3.4, 0.3, 0.15, 0.25, 0.2),
}
#: The event hazard of the three-node law.
HAZARD = (-2.2, 0.1, 0.4, 0.35, -0.55, -0.45)
#: The factors of the four-node law.  The shipped hazard fit holds a prefix cell at zero when
#: none of the units at risk with that treatment prefix has the event (and at one when all of
#: them do), and a regimen that reads a held cell reports no interval.  The policy reads all
#: 16 prefixes at node 4, so the four-node law keeps every prefix populated at n = 2,000: the
#: treatment is close to balanced, dropout is rarer, and the hazard rises from about 0.05 at
#: node 1 to about 0.3 at node 4, with smaller arm effects.  The expected number of held cells
#: per fit at n = 2,000 is about 2e-9 (``tests/unit/test_hazard_multinode_study_design.py``
#: sums it over every fit of the study).  Every path keeps a cumulative treatment and
#: retention probability of at least 0.042.
FOUR_MECHANISM: dict[str, Any] = {
    "first": (0.0, 0.2, 0.0),
    "later": (-0.1, 0.1, 0.0, 0.2),
    "dropout": (-4.0, 0.3, 0.15, 0.25, 0.2),
}
FOUR_HAZARD = (-3.0, 0.75, 0.2, 0.15, -0.25, -0.15)
#: The primary law: three nodes.
BASE = Law("base", 3, hazard=HAZARD, **MECHANISM)
#: The four-node variant for the enumerated policy (16 paths).
FOUR = Law("four", 4, hazard=FOUR_HAZARD, **FOUR_MECHANISM)
LAWS = {"base": BASE, "four": FOUR}


# ------------------------------------------------------------------ the regimens, fit side


def _rule_node(node: int) -> Callable[[Any], Any]:
    def assign(history: Any) -> Any:
        return (np.asarray(history["W2"], dtype=float) >= 2.0).astype(int)

    del node
    return assign


def _policy_node(node: int) -> Stochastic:
    def density(history: Any) -> np.ndarray:
        if node == 1:
            previous = np.zeros(len(history))
        else:
            previous = np.asarray(history[f"A{node - 1}"], dtype=float)
        p = 0.3 + 0.4 * previous + 0.1 * np.asarray(history["W1"], dtype=float)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, f"policy{node}", density_kind="known")


def _stay_node(node: int) -> Stochastic:
    def density(history: Any) -> np.ndarray:
        if node == 1:
            p = 0.3 + 0.2 * np.asarray(history["W1"], dtype=float)
        else:
            p = np.asarray(history[f"A{node - 1}"], dtype=float)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, f"stay{node}", density_kind="known")


def regimen(label: str, n_times: int) -> Any:
    """The fit-side regimen of ``label`` over ``n_times`` nodes."""
    nodes = range(1, n_times + 1)
    if label == "always":
        return (1,) * n_times
    if label == "never":
        return (0,) * n_times
    if label == "rule":
        return DynamicRegimen("rule", tuple(_rule_node(k) for k in nodes), rule_kind="known")
    if label == "policy":
        return DynamicRegimen("policy", tuple(_policy_node(k) for k in nodes))
    if label == "stay":
        return DynamicRegimen("stay", tuple(_stay_node(k) for k in nodes))
    if label.startswith("p"):
        return tuple(int(digit) for digit in label[1:])
    raise KeyError(label)


def regimens(labels: Sequence[str], n_times: int) -> dict[str, Any]:
    return {label: regimen(label, n_times) for label in labels}


# ------------------------------------------------------------------ linear targets


@dataclass(frozen=True)
class Target:
    """A linear combination :math:`\\sum_i c_i \\psi_{r_i}(t_i)` of regimen risks.

    Parameters
    ----------
    terms : tuple of tuple
        ``(coefficient, regimen label, horizon)`` per term.
    """

    terms: tuple[tuple[float, str, int], ...]

    def value(self, law: Law) -> float:
        return float(sum(c * law.risk(label, t) for c, label, t in self.terms))

    def curve(self, law: Law, row: Sequence[float]) -> float:
        return float(sum(c * law.curve(label, t, row) for c, label, t in self.terms))


def efficiency_sd(law: Law, target: Target) -> float:
    """The exact efficiency bound of a linear target, as a standard deviation."""
    psi = target.value(law)
    first = second = 0.0
    for row, mass in law.support():
        centred = target.curve(law, row) - psi
        first += mass * centred
        second += mass * centred**2
    if abs(first) > 1e-10:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function has mean {first}, not zero")
    return float(np.sqrt(second))


# ------------------------------------------------------------------ the working model


def msm_cells(law: Law) -> tuple[tuple[str, int], ...]:
    """Every ``(static path, horizon)`` cell of the working model."""
    return tuple(
        (label, horizon)
        for horizon in range(1, law.n_times + 1)
        for label in static_paths(law.n_times)
    )


def phi(label: str, horizon: int, w1: float) -> np.ndarray:
    """The design row :math:`(1, d, t, d W_1)` of one cell, with ``d`` its cumulative dose."""
    d = dose(label, horizon)
    return np.array([1.0, d, float(horizon), d * float(w1)])


def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """The working model's design over a fit's baseline frame."""
    w1 = np.asarray(frame["W1"], dtype=float)
    d = dose(str(label), horizon)
    return np.column_stack(
        [np.ones_like(w1), np.full_like(w1, d), np.full_like(w1, float(horizon)), d * w1]
    )


def working_model() -> MSM:
    """The declared logit working model over the static paths, with uniform cell weights."""
    return MSM(
        design=cast(Callable[[Any, Any], Any], design),
        terms=TERMS,
        link="logit",
        design_kind="known",
    )


def coefficient_name(term: str) -> str:
    return f"msm_regimen[{term}]"


def _rows(law: Law, w1: int) -> np.ndarray:
    return np.array([phi(label, horizon, w1) for label, horizon in msm_cells(law)])


def _stratum_risks(law: Law) -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for w1, w2 in STRATA:
        out[(w1, w2)] = np.array(
            [
                law.path_risk([int(d) for d in label[1:]], w1, w2, horizon)
                for label, horizon in msm_cells(law)
            ]
        )
    return out


def project(law: Law, risks: Mapping[tuple[int, int], np.ndarray]) -> np.ndarray:
    """The logit working-model coefficients of cell risks ``W -> (Psi_c(W))_c``."""
    beta = np.zeros(len(TERMS))
    step = np.full(len(TERMS), np.inf)
    for _ in range(100):
        score = np.zeros(len(TERMS))
        jacobian = np.zeros((len(TERMS), len(TERMS)))
        for stratum in STRATA:
            x = _rows(law, stratum[0])
            mean = _expit(x @ beta)
            score += MASS * x.T @ (risks[stratum] - mean)
            jacobian += MASS * (x.T * (mean * (1.0 - mean))) @ x
        step = np.linalg.solve(jacobian, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-15:
            break
    if np.max(np.abs(step)) > 1e-12:  # pragma: no cover - the projection is well posed
        raise AssertionError(f"the projection did not converge: last step {step}")
    return beta


def exact_beta(law: Law) -> np.ndarray:
    """The true coefficients: the projection of the law's own static-path cell risks."""
    return project(law, _stratum_risks(law))


def beta_sd(law: Law) -> np.ndarray:
    r"""The exact efficiency bound of each coefficient, as a standard deviation.

    The efficient influence function is :math:`M^{-1} \sum_c \varphi(c, W_1) [D_c + \psi_c -
    m(c, W_1; \beta)]` (Rosenblum and van der Laan 2010, (17)), with :math:`D_c + \psi_c` the
    cell's uncentred curve :meth:`Law.curve`.
    """
    beta = exact_beta(law)
    information = np.zeros((len(TERMS), len(TERMS)))
    for w1, _ in STRATA:
        x = _rows(law, w1)
        mean = _expit(x @ beta)
        information += MASS * (x.T * (mean * (1.0 - mean))) @ x
    inverse = np.linalg.inv(information)
    cells = msm_cells(law)
    first = np.zeros(len(TERMS))
    second = np.zeros(len(TERMS))
    for row, mass in law.support():
        w1 = int(row[0])
        x = _rows(law, w1)
        pieces = np.array([law.curve(label, horizon, row) for label, horizon in cells])
        curve = inverse @ (x.T @ (pieces - _expit(x @ beta)))
        first += mass * curve
        second += mass * curve**2
    if np.max(np.abs(first)) > 1e-10:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function has mean {first}, not zero")
    return np.sqrt(second)


# ------------------------------------------------------------------ learners


class HistoryLogistic(BaseEstimator, ClassifierMixin):
    """The correct learner of every factor: unpenalised logistic on the node's history.

    Each factor of :data:`BASE` and :data:`FOUR` is logistic in ``W1``, ``W2`` and the last one
    or two arms of the history the fit hands the learner, so the model is correct; it has no
    interaction, so it is not saturated, and targeting moves every fit.
    """

    def _features(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> HistoryLogistic:
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        self.model_.fit(self._features(X), np.asarray(y, dtype=float), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._features(X)))


class LastArmLogistic(HistoryLogistic):
    """The wrong hazard learner: logistic on the node's own arm ``A_k`` alone.

    A hazard design is ``[W1, W2, A_1, ..., A_k]``; this learner reads its last column.  The
    law's hazard depends on ``W`` and on ``A_{k-1}``, both of which confound ``A_k``.
    """

    def _features(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, -1:]


def wrong_hazards(law: Law) -> np.ndarray:
    """``(T, 2)`` hazards at the limit of :class:`LastArmLogistic`, by node and arm.

    At each node and arm the limit is the event share among the units at risk with that arm,
    over the observed law (the logistic fit on one binary column is saturated in it).
    """
    events = np.zeros((law.n_times, 2))
    totals = np.zeros((law.n_times, 2))
    for row, mass in law.support():
        for node in range(1, law.n_times + 1):
            base = 2 + 3 * (node - 1)
            if np.isnan(row[base]) or row[base + 1] == 0.0:
                break
            a = int(row[base])
            totals[node - 1, a] += mass
            events[node - 1, a] += mass * float(row[base + 2] == 1.0)
    return events / totals


def wrong_risk(law: Law, label: str, horizon: int) -> float:
    """The plug-in risk of a regimen at the limit of :class:`LastArmLogistic`."""
    hazards = wrong_hazards(law)
    total = 0.0
    for w1, w2 in STRATA:
        for path in itertools.product((0, 1), repeat=horizon):
            mass = law.path_mass(label, path, w1, w2)
            if mass == 0.0:
                continue
            survival, risk = 1.0, 0.0
            for node in range(1, horizon + 1):
                lam = hazards[node - 1, path[node - 1]]
                risk += survival * lam
                survival *= 1.0 - lam
            total += MASS * mass * risk
    return total


def wrong_hazard_limit(law: Law, target: Target) -> float:
    """The plug-in value of a linear target at the limit of :class:`LastArmLogistic`."""
    return float(sum(c * wrong_risk(law, label, t) for c, label, t in target.terms))


def node_hazards(law: Law) -> np.ndarray:
    """``(T,)`` hazards at the limit of the node-only learner: the at-risk event share by node.

    The learner is the prior share of the node's event (``DummyClassifier(strategy="prior")``
    on each node's hazard fit), so it reads neither the arms nor ``W``.
    """
    events = np.zeros(law.n_times)
    totals = np.zeros(law.n_times)
    for row, mass in law.support():
        for node in range(1, law.n_times + 1):
            base = 2 + 3 * (node - 1)
            if np.isnan(row[base]) or row[base + 1] == 0.0:
                break
            totals[node - 1] += mass
            events[node - 1] += mass * float(row[base + 2] == 1.0)
    return events / totals


def node_hazard_limit(law: Law, target: Target) -> float:
    """The plug-in value of a linear target at the limit of the node-only learner.

    Every regimen has the same plug-in risk curve, so a contrast's plug-in is zero.
    """
    hazards = node_hazards(law)

    def risk(horizon: int) -> float:
        survival, total = 1.0, 0.0
        for lam in hazards[:horizon]:
            total += survival * lam
            survival *= 1.0 - lam
        return total

    return float(sum(c * risk(t) for c, _, t in target.terms))


def wrong_beta(law: Law) -> np.ndarray:
    """The plug-in coefficients at the limit of :class:`LastArmLogistic`.

    The wrong hazards do not depend on ``W``, so every stratum's cell risks are the same.
    """
    hazards = wrong_hazards(law)
    cells = msm_cells(law)
    values = []
    for label, horizon in cells:
        survival, risk = 1.0, 0.0
        for node in range(1, horizon + 1):
            lam = hazards[node - 1, int(label[node])]
            risk += survival * lam
            survival *= 1.0 - lam
        values.append(risk)
    risks = dict.fromkeys(STRATA, np.array(values))
    return project(law, risks)
