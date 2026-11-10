r"""The two laws of the ``eliminated-competing-events`` study, with their eliminated truths.

The parameter is the risk of relapse had death been prevented, Young et al. (2020)
g-formula (23), with the order censoring, death, relapse within a node.  Both laws exist
already and realise their g-formula exactly in a finite sample; this module adds the
eliminated truth of the time-varying law, the support samplers, and the efficient influence
function of every reported parameter by a complex step.  The implementation never enters it.

==================  =======================================================================
law                 what it adds
==================  =======================================================================
``time_varying``    :mod:`tests.discrete_law_competing`: ``W, A1, C1, J1, L2, A2, C2, J2``,
                    a dynamic plan and a second-node covariate that treatment moves.  Its
                    hazards of both causes read ``W``, ``A1``, ``L2`` and ``A2``
``held_baseline``   :func:`tests.discrete_law_point_survival.survival_law` with two causes and
                    no ``L2``: a held three-level baseline treatment and three nodes, the only
                    design the hazard path of ``LTMLE`` takes
==================  =======================================================================

On the time-varying law the eliminated risk is

.. math::

    \psi_a(2) = \sum_w P(w)\Bigl[\tilde h_1 + \tilde S_1
                \sum_l P(L_2 = l \mid w, a_1, C_1 = 1, J_1 = 0)\, \tilde h_2\Bigr]

where every tilde hazard conditions on having no death at its node:
:math:`\tilde h_k = P(J_k = 1 \mid \cdot, C_k = 1) / P(J_k \ne 2 \mid \cdot, C_k = 1)` and
:math:`\tilde S_1 = P(J_1 = 0 \mid \cdot) / P(J_1 \ne 2 \mid \cdot)`.  It is written as
quotients of cell masses, as :func:`tests.discrete_law_competing.functional` is, so the
complex step differentiates it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests import discrete_law_competing as competing
from tests import discrete_law_point_survival as point
from tests.studies.canonical_ltmle import declared_regimens

#: The causes, by the code each law gives them.
RELAPSE, DEATH = 1, 2

# ------------------------------------------------------------------------- time-varying law

TV_REGIMENS: dict[str, Any] = declared_regimens(competing.REGIMEN_SPEC)
TV_REFERENCE = competing.REGIMEN_REFERENCE
TV_HORIZONS: tuple[int, ...] = competing.HORIZONS

#: ``continue_if_l2`` is ``always`` at the first node, so its ``t = 1`` risk is reported once.
TV_MEANS: tuple[str, ...] = (
    "risk_regimen[never @ t=1]",
    "risk_regimen[never @ t=2]",
    "risk_regimen[always @ t=1]",
    "risk_regimen[always @ t=2]",
    "risk_regimen[continue_if_l2 @ t=2]",
)
TV_CONTRASTS: tuple[str, ...] = (
    "ate_regimen[always vs never @ t=1]",
    "ate_regimen[always vs never @ t=2]",
    "ate_regimen[continue_if_l2 vs never @ t=2]",
)
TV_ESTIMANDS: tuple[str, ...] = (*TV_MEANS, *TV_CONTRASTS)


def _parse(estimand: str) -> tuple[str, str | None, int]:
    """``"ate_regimen[always vs never @ t=2]"`` -> ``("always", "never", 2)``."""
    inside = estimand[estimand.index("[") + 1 : -1]
    label, _, horizon = inside.rpartition(" @ t=")
    if estimand.startswith("ate_regimen["):
        left, right = label.split(" vs ")
        return left, right, int(horizon)
    if not estimand.startswith("risk_regimen["):
        raise ValueError(f"unknown estimand {estimand!r}")
    return label, None, int(horizon)


def tv_risk(probs: Any, label: str, horizon: int) -> Any:
    """The risk of relapse by ``horizon`` under ``label`` had death been prevented."""
    mass = competing._mass
    arm = competing._arm
    node1, node2 = competing.REGIMEN_ARMS[label]
    total = mass(probs)
    psi: Any = 0.0
    for w in (0, 1):
        a1 = arm(node1, w)
        share = mass(probs, w=w) / total
        kept = {"w": w, "a1": a1, "c1": 1}
        reached = mass(probs, **kept) - mass(probs, **kept, j1=DEATH)
        fired = mass(probs, **kept, j1=RELAPSE) / reached
        if horizon == 1:
            psi = psi + share * fired
            continue
        survived = mass(probs, **kept, j1=0)
        later: Any = 0.0
        for l2 in (0, 1):
            a2 = arm(node2, w, l2)
            density = mass(probs, **kept, j1=0, l2=l2) / survived
            second = {**kept, "j1": 0, "l2": l2, "a2": a2, "c2": 1}
            at_risk = mass(probs, **second) - mass(probs, **second, j2=DEATH)
            later = later + density * mass(probs, **second, j2=RELAPSE) / at_risk
        psi = psi + share * (fired + survived / reached * later)
    return psi


def tv_functional(probs: Any, estimand: str) -> Any:
    """Any reported parameter of the time-varying law, analytic in ``probs``."""
    left, right, horizon = _parse(estimand)
    if right is None:
        return tv_risk(probs, left, horizon)
    return tv_risk(probs, left, horizon) - tv_risk(probs, right, horizon)


def tv_eif(estimand: str, probs: np.ndarray = competing.PROBS) -> np.ndarray:
    """The EIF of ``estimand`` at every support point of the time-varying law."""
    base = probs.astype(complex)
    step = 1e-30
    values = np.empty(len(competing.SUPPORT))
    for position in range(len(competing.SUPPORT)):
        mass = np.zeros_like(base)
        mass[position] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        values[position] = float(np.imag(tv_functional(perturbed, estimand)) / step)
    return values


TV_TRUTH: dict[str, float] = {
    name: float(tv_functional(competing.PROBS, name)) for name in TV_ESTIMANDS
}

_TV_SUPPORT_FRAME = competing.frame().iloc[competing.first_row_of()].reset_index(drop=True)

#: The columns a time-varying fit is handed, beside ``eliminate=["death"]``.
TV_FIT_COLUMNS: dict[str, Any] = {
    "outcome": competing.outcome_columns(),
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def tv_sample(n: int, seed: int, probs: np.ndarray = competing.PROBS) -> pd.DataFrame:
    """``n`` independent rows of the time-varying law."""
    cells = np.random.default_rng(seed).choice(len(competing.SUPPORT), size=n, p=probs)
    return _TV_SUPPORT_FRAME.iloc[cells].reset_index(drop=True)


def tv_exact_frame() -> pd.DataFrame:
    """The ``4**8``-row sample whose empirical law is the time-varying law exactly."""
    return competing.frame()


# ------------------------------------------------------------------------ held-baseline law

HB_LAW = point.survival_law(causes=2)
HB_REGIMENS: dict[str, int] = {"a0": 0, "a1": 1, "a2": 2}
HB_REFERENCE = "a0"
HB_HORIZONS: tuple[int, ...] = (1, 2, 3)
HB_MEANS: tuple[str, ...] = tuple(
    f"risk_regimen[{label} @ t={horizon}]" for label in HB_REGIMENS for horizon in HB_HORIZONS
)
HB_CONTRASTS: tuple[str, ...] = tuple(
    f"ate_regimen[{label} vs {HB_REFERENCE} @ t={horizon}]"
    for label in HB_REGIMENS
    if label != HB_REFERENCE
    for horizon in HB_HORIZONS
)
HB_ESTIMANDS: tuple[str, ...] = (*HB_MEANS, *HB_CONTRASTS)


def _hb_assign(label: str) -> Any:
    return point.static(HB_REGIMENS[label])


def hb_functional(probs: Any, estimand: str) -> Any:
    """Any reported parameter of the held-baseline law, analytic in ``probs``."""
    left, right, horizon = _parse(estimand)

    def risk(label: str) -> Any:
        return HB_LAW.functional(probs, _hb_assign(label), horizon, cause=RELAPSE, eliminate=DEATH)

    return risk(left) if right is None else risk(left) - risk(right)


def hb_eif(estimand: str) -> np.ndarray:
    """The EIF of ``estimand`` at every support point of the held-baseline law."""
    left, right, horizon = _parse(estimand)
    curve = HB_LAW.eif(_hb_assign(left), horizon, RELAPSE, eliminate=DEATH)
    if right is not None:
        curve = curve - HB_LAW.eif(_hb_assign(right), horizon, RELAPSE, eliminate=DEATH)
    return curve


HB_TRUTH: dict[str, float] = {
    name: float(hb_functional(HB_LAW.probs, name)) for name in HB_ESTIMANDS
}

_HB_FRAME = HB_LAW.frame()
_HB_SUPPORT_FRAME = _HB_FRAME.iloc[HB_LAW.first_row_of()].reset_index(drop=True)


def hb_fit_columns() -> dict[str, Any]:
    """The columns a held-baseline fit is handed, beside ``eliminate=["death"]``."""
    return HB_LAW.fit_columns()


def hb_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` independent rows of the held-baseline law."""
    cells = np.random.default_rng(seed).choice(len(HB_LAW.support), size=n, p=HB_LAW.probs)
    return _HB_SUPPORT_FRAME.iloc[cells].reset_index(drop=True)


def hb_exact_frame() -> pd.DataFrame:
    """The sample whose empirical law is the held-baseline law exactly."""
    return _HB_FRAME.copy()


def efficiency_sd(eif: np.ndarray, probs: np.ndarray) -> float:
    r""":math:`\sqrt{E_P[D^*(O)^2]}`, the efficient spread of one observation."""
    return float(np.sqrt(np.sum(probs * eif**2)))
