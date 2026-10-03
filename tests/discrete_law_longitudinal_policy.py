r"""Known stochastic policies on exactly realised longitudinal laws.

The implementation never enters this module.  Each parameter below is the longitudinal
g-formula under a policy, written directly as ratios of finite-support masses, and
complex-step derivatives of it supply the efficient influence function independently of
the fitted ratio, recursion and fluctuation.

Three laws are read, and none is changed:

* :mod:`tests.discrete_law_longitudinal_multivalue`, the two-node three-level end-of-study
  law (512 rows).  The policies :data:`POLICY1`, :data:`POLICY2` and
  :data:`POLICY2_PARTIAL` are stated over it.
* :mod:`tests.discrete_law_survival`, the binary survival law with censoring.
  :data:`SURVIVAL_Q1` and :data:`SURVIVAL_Q2` are stated over it.
* :mod:`tests.discrete_law_competing`, the competing-risk law, with the same binary
  policies.

Every policy table is a fixed constant and a multiple of one quarter.  It is not a function
of the cell masses, so the complex step does not move it.  The policy tables of the
end-of-study law list their probabilities in :data:`ARM_LABELS` order, the raw support
order ``("standard", "high", "low")``.  That is not the data layer's sorted level order
``("high", "low", "standard")``, so the estimator side hands each policy over as a dataframe
keyed by label, which :class:`~cleverly.longitudinal.DynamicRegimen` reorders by label.
That side lives in :mod:`tests.longitudinal_policies`, because an oracle must not import
the library it checks (``tests/unit/test_oracle_independence.py``).

The augmented law of :func:`augmented_frame` records the randomizer: one uniform
:math:`\varepsilon_t \in \{0, 1, 2, 3\}` per node, independent of everything, realised
exactly in ``16 * 512`` rows.  The rule that reads it is the inverse-CDF draw of the policy,
so the deterministic regimen of that law is Theorem 3 of Díaz, Williams, Hoffman and
Schenck (2023) as stated, with :math:`\varepsilon_t` in :math:`L_t`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from . import discrete_law_competing as competing
from . import discrete_law_longitudinal_multivalue as law
from . import discrete_law_survival as survival

ARM_LABELS = law.ARM_LABELS
SUPPORT = law.SUPPORT
PROBS = law.PROBS
N = law.N

#: ``q_1(a | W = w)``, indexed ``[w, a]`` with ``a`` in :data:`ARM_LABELS` order.  It favours
#: ``high`` at both ``W``, where the mechanism :data:`G1 <tests.discrete_law_longitudinal_multivalue.G1>`
#: favours ``low`` at ``W = 0`` and ``standard`` at ``W = 1``.
#:
#: The tables were chosen by a search over the quarter-grid rotations of ``(1/2, 1/4, 1/4)``
#: for the policy whose mean is farthest from every deterministic mean of the law, from
#: the observed mean, from its own label permutation and from the ``partial`` plan.  On this
#: law every such mean lies in ``[0.4375, 0.53125]``, so the best separation is 0.0137.
POLICY1 = np.array([[0.25, 0.50, 0.25], [0.25, 0.50, 0.25]])

_BASE = np.array([0.50, 0.25, 0.25])

#: ``q_2(a | W = w, A_1 = a_1, L_2 = l)``, indexed ``[w, a1, l, a]`` in
#: :data:`ARM_LABELS` order.  It reads the earlier arm: each ``a1`` rotates the base row by
#: a different amount, so a policy frame without ``A1`` reads a different policy.
POLICY2 = np.array(
    [
        [[np.roll(_BASE, (a1 + 2 * l2 + 2 * w) % 3) for l2 in range(2)] for a1 in range(3)]
        for w in range(2)
    ]
)

#: A node-2 policy with zero cells that are not one-hot.  At ``L2 = 1`` it draws one of two
#: arms with probability one half each, so a third arm has zero probability there and the
#: rows that received it leave the plan.  It is the witness of the support mask.
POLICY2_PARTIAL = np.array(
    [
        [
            [
                np.roll(np.array([0.50, 0.50, 0.0]), a1) if l2 == 1 else POLICY2[w, a1, l2]
                for l2 in range(2)
            ]
            for a1 in range(3)
        ]
        for w in range(2)
    ]
)

#: ``d_2(W, L_2)`` of the plan ``policy_then_rule``: ``high`` at ``L2 = 1`` and ``low``
#: otherwise, as raw codes indexed ``[w, l2]``.  A rule frame holds no earlier treatment.
RULE2 = np.array([[2, 1], [2, 1]])

#: The plans, oracle side.  A node is ``("label", raw code)``, ``("rule", array)`` or
#: ``("policy", table)``.  A rule at node 2 is indexed ``[w, l2]``.
PLANS: dict[str, tuple[tuple[str, Any], tuple[str, Any]]] = {
    "low": (("label", 2), ("label", 2)),
    "mix": (("policy", POLICY1), ("policy", POLICY2)),
    "taper": (("label", 1), ("policy", POLICY2)),
    "policy_then_rule": (("policy", POLICY1), ("rule", RULE2)),
    "partial": (("policy", POLICY1), ("policy", POLICY2_PARTIAL)),
}
REFERENCE = "low"

_CODE = {label: code for code, label in enumerate(ARM_LABELS)}


def _density(entry: tuple[str, Any], node: int, a: int, w: int, a1: int, l2: int) -> float:
    kind, value = entry
    if kind == "label":
        return 1.0 if a == value else 0.0
    if kind == "rule":
        target = value[w] if node == 1 else value[w, l2]
        return 1.0 if a == int(target) else 0.0
    return float(value[w, a] if node == 1 else value[w, a1, l2, a])


@cache
def _index(**pattern: int) -> tuple[int, ...]:
    names = ("w", "a1", "l2", "a2", "y")
    return tuple(
        index
        for index, point in enumerate(SUPPORT)
        if all(point[names.index(name)] == value for name, value in pattern.items())
    )


def _mass(probs: Any, **pattern: int) -> Any:
    """Total probability of the matching support points, a linear form in ``probs``."""
    return sum(probs[index] for index in _index(**pattern))


def functional_policy(probs: Any, plan: tuple[tuple[str, Any], tuple[str, Any]]) -> Any:
    r""":math:`E[Y^{\bar\pi}]` under a plan, directly from the finite-support g-formula.

    .. math::

        \Psi = \sum_w P(w) \sum_{a_1} \pi_1(a_1 \mid w) \sum_{l} P(l \mid w, a_1)
               \sum_{a_2} \pi_2(a_2 \mid w, a_1, l)\, E[Y \mid w, a_1, l, a_2]

    Every conditional is a quotient of sums of cell probabilities, and every policy weight
    is a constant, so this stays analytic in ``probs``.
    """
    total = _mass(probs)
    psi = 0.0
    for w in range(2):
        for a1 in range(3):
            first = _density(plan[0], 1, a1, w, 0, 0)
            if first == 0.0:
                continue
            reached = _mass(probs, w=w, a1=a1)
            for l2 in range(2):
                p_l2 = _mass(probs, w=w, a1=a1, l2=l2) / reached
                for a2 in range(3):
                    second = _density(plan[1], 2, a2, w, a1, l2)
                    if second == 0.0:
                        continue
                    treated = _mass(probs, w=w, a1=a1, l2=l2, a2=a2)
                    events = _mass(probs, w=w, a1=a1, l2=l2, a2=a2, y=1)
                    psi = psi + (_mass(probs, w=w) / total) * first * p_l2 * second * (
                        events / treated
                    )
    return psi


def functional(probs: Any, estimand: str) -> Any:
    """An ``ey_regimen`` or ``ate_regimen`` name of a fit over :data:`PLANS`."""
    if estimand.startswith("ate_regimen["):
        left, right = estimand[len("ate_regimen[") : -1].split(" vs ")
        return functional(probs, f"ey_regimen[{left}]") - functional(probs, f"ey_regimen[{right}]")
    if not estimand.startswith("ey_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")
    return functional_policy(probs, PLANS[estimand[len("ey_regimen[") : -1]])


def names(labels: tuple[str, ...] = tuple(PLANS)) -> tuple[str, ...]:
    """The names a fit over ``labels`` reports, the first label being the reference."""
    reference = labels[0]
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {reference}]" for label in labels if label != reference
    )


NAMES = names()
TRUTH = {name: float(functional(PROBS, name)) for name in NAMES}


def gateaux_at(probs: np.ndarray, estimand: str, point: int, *, step: float = 1e-30) -> float:
    """The complex-step Gateaux derivative of ``estimand`` at one support point."""
    base = np.asarray(probs, dtype=complex)
    mass = np.zeros_like(base)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(functional(perturbed, estimand)) / step)


def eif_policy(probs: np.ndarray, estimand: str) -> np.ndarray:
    """The efficient influence function of ``estimand`` at every support point."""
    return np.array([gateaux_at(probs, estimand, point) for point in range(len(SUPPORT))])


def deterministic_means() -> dict[str, float]:
    """``E[Y^{a1, a2}]`` for every static plan of the law, the policy witnesses' foil."""
    means = {}
    for a1, a2 in itertools.product(range(3), range(3)):
        plan = (("label", a1), ("label", a2))
        means[f"{ARM_LABELS[a1]}/{ARM_LABELS[a2]}"] = float(functional_policy(PROBS, plan))
    return means


def observed_mean() -> float:
    """``E[Y]`` under the law, the natural course."""
    return float(_mass(PROBS, y=1))


# ------------------------------------------------------------------ recorded randomizer

#: Draws per node of the recorded randomizer: ``epsilon_t`` is uniform on ``0..3``.
RANDOMIZER_LEVELS = 4


def _draw(row: np.ndarray, epsilon: int) -> int:
    """The inverse-CDF draw: the first raw arm whose quarter count exceeds ``epsilon``."""
    counts = np.rint(np.cumsum(row) * RANDOMIZER_LEVELS).astype(int)
    return int(np.flatnonzero(counts > epsilon)[0])


def augmented_frame() -> pd.DataFrame:
    """The 512-row law with an independent uniform randomizer per node, ``16 * 512`` rows.

    Row block ``c`` of :func:`tests.discrete_law_longitudinal_multivalue.frame` is repeated
    once per ``(e1, e2)`` pair, so each ``O`` cell carries every randomizer pair exactly
    once per unit and the empirical law of ``(O, E1, E2)`` is the product law.
    """
    base = law.frame()
    pairs = list(itertools.product(range(RANDOMIZER_LEVELS), repeat=2))
    blocks = []
    for e1, e2 in pairs:
        block = base.copy()
        block["E1"] = float(e1)
        block["E2"] = float(e2)
        blocks.append(block)
    stacked = pd.concat(blocks, ignore_index=True)
    stacked["row"] = np.tile(np.arange(len(base)), len(pairs))
    return stacked.sort_values(["row", "E1", "E2"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ survival


#: ``q_1(A_1 = 1 | W = w)`` on the binary survival and competing laws.
SURVIVAL_Q1 = np.array([0.25, 0.75])

#: ``q_2(A_2 = 1 | W = w, A_1 = a_1, L_2 = l)``, indexed ``[w, a1, l2]``.
SURVIVAL_Q2 = np.array([[[0.75, 0.25], [0.50, 0.75]], [[0.25, 0.50], [0.75, 0.25]]])


def _binary(p: Any, a: int) -> Any:
    return p if a == 1 else 1.0 - p


def functional_policy_survival(probs: Any, horizon: int) -> Any:
    r"""The cumulative risk at ``horizon`` under the survival policy, longhand.

    .. math::

        \Psi_1 &= \sum_w P(w) \sum_{a_1} q_1(a_1 \mid w)\, h_1(w, a_1) \\
        \Psi_2 &= \sum_w P(w) \sum_{a_1} q_1(a_1 \mid w) \Bigl[h_1 + (1 - h_1)
                  \sum_l P(l \mid w, a_1, C_1 = 1, Y_1 = 0)
                  \sum_{a_2} q_2(a_2 \mid w, a_1, l)\, h_2(w, a_1, l, a_2)\Bigr]
    """
    mass = survival._mass
    total = mass(probs)
    psi = 0.0
    for w in (0, 1):
        share = mass(probs, w=w) / total
        for a1 in (0, 1):
            weight1 = _binary(SURVIVAL_Q1[w], a1)
            reached = mass(probs, w=w, a1=a1, c1=1)
            hazard1 = mass(probs, w=w, a1=a1, c1=1, y1=1) / reached
            if horizon == 1:
                psi = psi + share * weight1 * hazard1
                continue
            survived = mass(probs, w=w, a1=a1, c1=1, y1=0)
            later = 0.0
            for l2 in (0, 1):
                density = mass(probs, w=w, a1=a1, c1=1, y1=0, l2=l2) / survived
                for a2 in (0, 1):
                    weight2 = _binary(SURVIVAL_Q2[w, a1, l2], a2)
                    uncensored = mass(probs, w=w, a1=a1, c1=1, y1=0, l2=l2, a2=a2, c2=1)
                    events = mass(probs, w=w, a1=a1, c1=1, y1=0, l2=l2, a2=a2, c2=1, y2=1)
                    later = later + density * weight2 * (events / uncensored)
            psi = psi + share * weight1 * (hazard1 + (1.0 - hazard1) * later)
    return psi


def functional_policy_competing(probs: Any, cause: str, horizon: int) -> Any:
    """The cause-specific cumulative incidence under the survival policy, longhand.

    The numerator is cause-specific and the survival factor all-cause, as in
    :func:`tests.discrete_law_competing.functional`.
    """
    mass = competing._mass
    j = competing.CAUSES.index(cause) + 1
    total = mass(probs)
    psi = 0.0
    for w in (0, 1):
        share = mass(probs, w=w) / total
        for a1 in (0, 1):
            weight1 = _binary(SURVIVAL_Q1[w], a1)
            reached = mass(probs, w=w, a1=a1, c1=1)
            hazard1 = mass(probs, w=w, a1=a1, c1=1, j1=j) / reached
            if horizon == 1:
                psi = psi + share * weight1 * hazard1
                continue
            survived = mass(probs, w=w, a1=a1, c1=1, j1=0)
            later = 0.0
            for l2 in (0, 1):
                density = mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2) / survived
                for a2 in (0, 1):
                    weight2 = _binary(SURVIVAL_Q2[w, a1, l2], a2)
                    uncensored = mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
                    events = mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=j)
                    later = later + density * weight2 * (events / uncensored)
            psi = psi + share * weight1 * (hazard1 + (survived / reached) * later)
    return psi


def gateaux(
    function: Callable[[Any], Any], probs: np.ndarray, point: int, *, step: float = 1e-30
) -> float:
    """The complex-step Gateaux derivative of ``function`` at one support point."""
    base = np.asarray(probs, dtype=complex)
    mass = np.zeros_like(base)
    mass[point] = 1.0
    perturbed = (1.0 - 1j * step) * base + 1j * step * mass
    return float(np.imag(function(perturbed)) / step)


def eif(function: Callable[[Any], Any], probs: np.ndarray) -> np.ndarray:
    """The influence function of ``function`` at every point of a law's support."""
    return np.array([gateaux(function, probs, point) for point in range(len(probs))])


def policy_tables() -> Mapping[str, np.ndarray]:
    """Every policy table this module declares, by name, for the transcription gates."""
    return {
        "POLICY1": POLICY1,
        "POLICY2": POLICY2,
        "POLICY2_PARTIAL": POLICY2_PARTIAL,
        "SURVIVAL_Q1": SURVIVAL_Q1,
        "SURVIVAL_Q2": SURVIVAL_Q2,
    }
