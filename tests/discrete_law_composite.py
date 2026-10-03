r"""Finite-support laws with an outcome and a treatment missing at random, realised exactly.

The exact-law checks of the composite-indicator construction in
``tests/unit/test_composite_missing_data.py`` run on these laws.  A row reveals ``W``,
``Delta_A`` and ``Delta`` always, ``A`` only when ``Delta_A = 1`` and ``Y`` only when
``Delta = 1``.  The support is

.. math::

    (w, c, k), \qquad c \in \{a_0, \ldots, a_{K-1}, \text{unrecorded}\},
    \qquad k \in \{\,\Delta = 1, Y = 0;\ \ \Delta = 1, Y = 1;\ \ \Delta = 0\,\},

a ``(3, K + 1, 3)`` array whose column ``K`` holds the rows with ``Delta_A = 0``.  Each cell
probability is a multiple of ``1 / N``, so :meth:`CompositeLaw.frame` *is* the law and not a
sample from it.  The module asserts that at import.

**The composite identity.**  With :math:`\pi_A(a, w) = P(\Delta_A = 1 \mid A = a, W = w)` the
probability of :math:`C_a = \Delta_A \Delta 1\{A = a\}` is

.. math::

    g_{c,a}(w) = g(a \mid w)\,\pi_A(a, w)\,\pi(a, w)
               = P(\Delta_A = 1 \mid w)\,P(A = a \mid \Delta_A = 1, w)\,\pi(a, w).

The second form is the three factors the fit estimates.  The first is how the tables are
written, so no separate table of :math:`P(A = a \mid \Delta_A = 1, w)` is needed.
:math:`\pi_A` depends on both arguments, so the recording of the treatment depends on the
treatment, and :math:`P(A = a \mid w)` is not identified from the observed data.

**Why ``W`` has three levels.**  With two levels each arm's outcome tilt saturates, and a
both-wrong control cannot fail (F4 finding 14).

**What an unrecorded row holds.**  Its outcome is recorded with probability one half and is
one with probability one half.  No functional here reads it, which is the point: a
construction that reads such a row moves the estimate.

**Witness margins.**  :data:`WITNESS_LIMITS` holds, for each law, the large-sample limit of
the composite TMLE under every mutation the tests apply, each a finite sum.  The module
asserts at import that each limit is at least :data:`MARGIN` from the truth, so a law edit
that collapses a witness fails here rather than in a test that then passes vacuously.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from tests.studies.mar_arm_indexed_laws import Law

#: Rows in each realised sample.
N = 2048

#: ``P(W = w)``.
P_W = np.array([1 / 2, 1 / 4, 1 / 4])

#: The smallest distance from the truth each witness limit must keep.
MARGIN = 0.02

OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = 0, 1, 2

#: ``P(Delta = 1 | Delta_A = 0, W)`` and ``P(Y = 1 | Delta_A = 0, Delta = 1, W)``.
UNRECORDED_RESPONSE = 0.5
UNRECORDED_OUTCOME = 0.5


@dataclass(frozen=True)
class CompositeLaw:
    """One exact law, every table indexed ``[w, a]`` in ``labels`` order.

    Parameters
    ----------
    key : str
        ``"two"`` or ``"three"``.
    labels : tuple
        The treatment value of each table column.
    g : ndarray
        ``P(A = a | W = w)``.
    pi_treatment : ndarray
        ``P(Delta_A = 1 | A = a, W = w)``.
    pi : ndarray
        ``P(Delta = 1 | A = a, Delta_A = 1, W = w)``.
    q : ndarray
        ``P(Y = 1 | A = a, Delta_A = 1, Delta = 1, W = w)``.
    wrong_g : ndarray
        A wrong ``P(A = a | Delta_A = 1, W)``, the mechanism drift's treatment factor.
    wrong_recorded : ndarray
        A wrong ``P(Delta_A = 1 | W)``, the mechanism drift's treatment observation factor.
    weight : ndarray
        A fixed weight that depends on ``W``, indexed ``[w]``.
    """

    key: str
    labels: tuple[Any, ...]
    g: np.ndarray
    pi_treatment: np.ndarray
    pi: np.ndarray
    q: np.ndarray
    wrong_g: np.ndarray
    wrong_recorded: np.ndarray
    weight: np.ndarray
    counts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", self._cell_counts())

    @property
    def k(self) -> int:
        return len(self.labels)

    @property
    def codes(self) -> tuple[Any, ...]:
        """The treatment values in the order ``CausalData`` codes them."""
        return tuple(sorted(self.labels))

    @property
    def reference(self) -> Any:
        return self.codes[0]

    @property
    def support(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(itertools.product(range(3), range(self.k + 1), range(3)))

    @property
    def recorded(self) -> np.ndarray:
        """``P(Delta_A = 1 | W = w)``."""
        return np.asarray((self.g * self.pi_treatment).sum(axis=1), dtype=float)

    @property
    def conditional_g(self) -> np.ndarray:
        """``P(A = a | Delta_A = 1, W = w)``."""
        joint = self.g * self.pi_treatment
        return np.asarray(joint / joint.sum(axis=1, keepdims=True), dtype=float)

    @property
    def composite(self) -> np.ndarray:
        """``g_c(a, w) = g pi_A pi``."""
        return np.asarray(self.g * self.pi_treatment * self.pi, dtype=float)

    def _cell_counts(self) -> np.ndarray:
        counts = np.empty((3, self.k + 1, 3))
        for w, c, kind in self.support:
            if c < self.k:
                base = P_W[w] * self.g[w, c] * self.pi_treatment[w, c]
                response, outcome = self.pi[w, c], self.q[w, c]
            else:
                base = P_W[w] * (1.0 - self.recorded[w])
                response, outcome = UNRECORDED_RESPONSE, UNRECORDED_OUTCOME
            if kind == UNOBSERVED:
                counts[w, c, kind] = base * (1.0 - response) * N
            else:
                share = outcome if kind == OBSERVED_ONE else 1.0 - outcome
                counts[w, c, kind] = base * response * share * N
        rounded = np.rint(counts)
        if np.max(np.abs(counts - rounded)) > 1e-9:  # pragma: no cover - guards the constants
            raise AssertionError(f"law {self.key}: a cell probability is not a multiple of 1/N")
        if rounded.min() < 2:  # pragma: no cover - guards the constants
            raise AssertionError(f"law {self.key}: every cell needs at least two rows")
        return rounded.astype(int)

    @property
    def probs(self) -> np.ndarray:
        """``P(W, C, K)``, bit for bit the empirical law of :meth:`frame`."""
        return self.counts / N

    def cell_of_row(self) -> np.ndarray:
        """The support index of each row of :meth:`frame`."""
        counts = [self.counts[w, c, kind] for w, c, kind in self.support]
        return np.repeat(np.arange(len(self.support)), counts)

    def frame(self, *, coded_unrecorded: bool = False) -> pd.DataFrame:
        """The ``N``-row sample whose empirical distribution is exactly this law.

        ``A`` is missing wherever ``DeltaA = 0``, and ``Y`` wherever ``Delta = 0``.  With
        ``coded_unrecorded=True`` an unrecorded row instead holds the first table label,
        which the fit must ignore (E12).  ``weight`` is the fixed ``W``-dependent weight.
        """
        cells = self.cell_of_row()
        columns = np.array(self.support, dtype=float)[cells]
        w = columns[:, 0].astype(int)
        c = columns[:, 1].astype(int)
        kind = columns[:, 2]
        recorded = c < self.k
        labels = np.asarray(self.labels, dtype=object)
        fill: Any = labels[0] if coded_unrecorded else None
        treatment: Any = np.where(recorded, labels[np.minimum(c, self.k - 1)], fill)
        if self.k == 2:
            treatment = np.asarray(
                [np.nan if value is None else float(value) for value in treatment], dtype=float
            )
        return pd.DataFrame(
            {
                "W": w.astype(float),
                "A": treatment,
                "Y": np.where(kind == UNOBSERVED, np.nan, kind),
                "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
                "DeltaA": recorded.astype(float),
                "weight": self.weight[w],
            }
        )

    # ------------------------------------------------------------------ functional

    def names(self) -> tuple[str, ...]:
        """Every reported name on a binary outcome, in the engine's report order by stem."""
        if self.k == 2:
            return ("ey0", "ey1", "ate", "rr", "or")
        others = [label for label in self.codes if label != self.reference]
        return (
            *(f"ey[{label}]" for label in self.codes),
            *(f"ate[{label} vs {self.reference}]" for label in others),
            *(f"rr[{label} vs {self.reference}]" for label in others),
            *(f"or[{label} vs {self.reference}]" for label in others),
        )

    def _arms_of(self, name: str) -> tuple[int, int | None]:
        head, _, rest = name.partition("[")
        if self.k == 2:
            if head in ("ey0", "ey1"):
                return self.labels.index(float(head[-1])), None
            return self.labels.index(1.0), self.labels.index(0.0)
        if head == "ey":
            return self.labels.index(rest.rstrip("]")), None
        arm, reference = rest.rstrip("]").split(" vs ")
        return self.labels.index(arm), self.labels.index(reference)

    def functional(self, probs: Any, name: str, *, weighted: bool = False) -> Any:
        r"""The target as a closed-form function of the cell probabilities.

        ``E(Y_a) = sum_w P(W = w) P(Y = 1 | A = a, Delta_A = 1, Delta = 1, W = w)``, or the
        weight-tilted mean with ``weighted=True``.  ``rr`` and ``or`` are on the log scale.
        Every operation is arithmetic or :func:`numpy.log`, so the complex step in
        :meth:`gateaux` stays exact.
        """
        p = np.asarray(probs)
        p_w = p.sum(axis=(1, 2))
        if weighted:
            p_w = p_w * self.weight / (p_w * self.weight).sum()
        observed = p[:, : self.k, OBSERVED_ZERO] + p[:, : self.k, OBSERVED_ONE]
        q = p[:, : self.k, OBSERVED_ONE] / observed
        arm, reference = self._arms_of(name)
        one = (p_w * q[:, arm]).sum()
        if reference is None:
            return one
        zero = (p_w * q[:, reference]).sum()
        head = name.partition("[")[0]
        if head == "ate":
            return one - zero
        if head == "rr":
            return np.log(one) - np.log(zero)
        if head == "or":
            return np.log(one / (1.0 - one)) - np.log(zero / (1.0 - zero))
        raise ValueError(f"unknown estimand {name!r}")

    def truth(self, name: str, *, weighted: bool = False) -> float:
        """The truth on the reported scale: ``rr`` and ``or`` as ratios."""
        value = float(self.functional(self.probs, name, weighted=weighted))
        head = name.partition("[")[0]
        return float(np.exp(value)) if head in ("rr", "or") else value

    def gateaux(self, name: str, point: int, *, step: float = 1e-30) -> float:
        """The Gateaux derivative of ``name`` at support point ``point``, by complex step."""
        base = self.probs.astype(complex)
        mass = np.zeros_like(base)
        mass[self.support[point]] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        return float(np.imag(self.functional(perturbed, name)) / step)

    def eif(self, name: str) -> np.ndarray:
        """The EIF of ``name`` at every support point, in support order."""
        return np.array([self.gateaux(name, point) for point in range(len(self.support))])

    # ----------------------------------------------------------------- the oracles

    def law(self) -> Law:
        """The law as the arm-indexed oracle learners read it."""
        return Law(
            key=f"c{self.k}",
            scenario=f"composite_{self.key}_exact",
            p_w=P_W,
            g=self.g,
            pi=self.pi,
            mu=self.q,
            labels=self.labels,
            continuous=False,
            pi_treatment=self.pi_treatment,
        )

    # ---------------------------------------------------------- the witness limits

    def tmle_limit(self, composite: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""The composite TMLE's arm means under a working denominator and regression.

        Each arm's logistic fluctuation is fitted on the rows with :math:`C_a = 1`, of mass
        :math:`P(w) g_{c,a}(w)`, along :math:`1 / \tilde g_{c,a}`.  On the exact frame the
        empirical score is this population score, so the fit lands on this value.
        """
        clever = 1.0 / composite
        mass = P_W[:, None] * self.composite
        offset = logit(mu)
        psi = np.empty(self.k)
        for arm in range(self.k):

            def score(epsilon: float, arm: int = arm) -> float:
                fitted = expit(offset[:, arm] + epsilon * clever[:, arm])
                return float(np.sum(mass[:, arm] * clever[:, arm] * (self.q[:, arm] - fitted)))

            epsilon = brentq(score, -50.0, 50.0, xtol=1e-15, rtol=1e-15)
            psi[arm] = float(P_W @ expit(offset[:, arm] + epsilon * clever[:, arm]))
        return psi

    def complement_limit(self, mu: np.ndarray) -> np.ndarray:
        r"""The two-arm limit when arm 0's denominator is taken as :math:`1 - g_{c,1}`.

        The complement form tilts one margin, so arm 1 keeps its own column and arm 0
        divides by the complement.  The arm-0 rows still fit along their own covariate.
        """
        upper = self.labels.index(1.0)
        lower = self.labels.index(0.0)
        working = np.array(self.composite, copy=True)
        working[:, lower] = 1.0 - self.composite[:, upper]
        return self.tmle_limit(working, mu)

    def mutations(self) -> dict[str, np.ndarray]:
        """Every working denominator a mutation control substitutes, as ``(w, a)``."""
        conditional = self.conditional_g
        recorded = self.recorded[:, None]
        reference = self.labels.index(self.reference)
        # The factor-order mutation: a categorical fit on every row, with an unrecorded
        # row read as the reference arm.
        everywhere = self.g * self.pi_treatment
        everywhere[:, reference] += 1.0 - self.recorded
        out = {
            "omit_recorded": conditional * self.pi,
            "omit_response": conditional * recorded,
            "renormalized": self.composite / self.composite.sum(axis=1, keepdims=True),
            "factor_order": everywhere * recorded * self.pi,
            "mechanism_drift": self.wrong_g * self.wrong_recorded[:, None] * self.pi,
        }
        if self.k >= 3:
            out["rolled"] = np.roll(self.composite, 1, axis=1)
        return out

    def witness_limits(self) -> dict[str, np.ndarray]:
        """Each mutation's arm means under the outcome drift, and the both-wrong limit."""
        drift = 1.0 - self.q
        out = {name: self.tmle_limit(table, drift) for name, table in self.mutations().items()}
        if self.k == 2:
            out["complement"] = self.complement_limit(drift)
        out["truth"] = P_W @ self.q
        return out


def _check_margins(law: CompositeLaw) -> dict[str, np.ndarray]:
    limits = law.witness_limits()
    truth = limits["truth"]
    for name, values in limits.items():
        if name == "truth":
            continue
        gap = float(np.max(np.abs(values - truth)))
        if gap < MARGIN:  # pragma: no cover - guards the constants
            raise AssertionError(
                f"law {law.key}: the {name} witness limit is within {gap:.4f} of the truth"
            )
    # The correct composite under the drift lands on the truth: the control of every witness.
    exact = law.tmle_limit(law.composite, 1.0 - law.q)
    if float(np.max(np.abs(exact - truth))) > 1e-10:  # pragma: no cover - guards the solver
        raise AssertionError(f"law {law.key}: the correct composite does not reach the truth")
    return limits


TWO = CompositeLaw(
    key="two",
    labels=(0.0, 1.0),
    g=np.array([[5 / 8, 3 / 8], [2 / 8, 6 / 8], [4 / 8, 4 / 8]]),
    pi_treatment=np.array([[3 / 4, 3 / 4], [2 / 4, 3 / 4], [2 / 4, 2 / 4]]),
    pi=np.array([[2 / 4, 3 / 4], [3 / 4, 1 / 4], [1 / 4, 2 / 4]]),
    q=np.array([[1 / 4, 3 / 4], [2 / 4, 1 / 4], [3 / 4, 2 / 4]]),
    wrong_g=np.array([[0.30, 0.70], [0.65, 0.35], [0.20, 0.80]]),
    wrong_recorded=np.array([0.90, 0.40, 0.95]),
    weight=np.array([1.0, 2.0, 0.5]),
)

THREE = CompositeLaw(
    key="three",
    labels=("low", "mid", "high"),
    g=np.array([[4 / 8, 3 / 8, 1 / 8], [2 / 8, 1 / 8, 5 / 8], [1 / 8, 4 / 8, 3 / 8]]),
    pi_treatment=np.array([[3 / 4, 2 / 4, 3 / 4], [2 / 4, 3 / 4, 2 / 4], [3 / 4, 3 / 4, 2 / 4]]),
    pi=np.array([[2 / 4, 3 / 4, 1 / 4], [1 / 4, 2 / 4, 3 / 4], [3 / 4, 1 / 4, 2 / 4]]),
    q=np.array([[1 / 4, 2 / 4, 3 / 4], [3 / 4, 1 / 4, 2 / 4], [2 / 4, 3 / 4, 1 / 4]]),
    wrong_g=np.array([[0.25, 0.35, 0.40], [0.45, 0.20, 0.35], [0.40, 0.40, 0.20]]),
    wrong_recorded=np.array([0.90, 0.40, 0.95]),
    weight=np.array([1.0, 2.0, 0.5]),
)

LAWS: dict[str, CompositeLaw] = {"two": TWO, "three": THREE}

#: Each law's witness limits, computed and checked at import.
WITNESS_LIMITS: dict[str, dict[str, np.ndarray]] = {
    key: _check_margins(law) for key, law in LAWS.items()
}

#: The smallest composite probability over both laws: every factor and composite is at
#: least 1/64, above every default bound.
_SMALLEST = min(float(law.composite.min()) for law in LAWS.values())
if _SMALLEST < 1 / 64:  # pragma: no cover - guards the constants
    raise AssertionError(f"a composite probability is {_SMALLEST}, below 1/64")
