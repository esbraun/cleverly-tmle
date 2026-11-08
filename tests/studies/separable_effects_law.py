r"""The law of the ``separable-effects`` study (X91 part D), its truths and its engine inputs.

The law is :func:`tests.discrete_law_point_survival.survival_law` with two causes: a binary
baseline ``W``, a three-level baseline treatment ``A`` held over three nodes, censoring at every
node, and the causes relapse (code 1) and death (code 2).  Every conditional probability is a
multiple of one quarter, so the ``4**k``-row exact sample realises the law.

Every truth is computed from cell masses by the law itself, never by the implementation:
the static incidences by :meth:`~tests.discrete_law_point_survival.PointLaw.functional` and
the separable incidences by :meth:`~tests.discrete_law_point_survival.PointLaw.separable`,
Stensrud et al. (2022) identification (8) with death counted first within a node.  The
efficient influence functions are complex-step Gateaux derivatives of the same formulas.

=================  =======================================================================
regimen            arms
=================  =======================================================================
``a0``, ``a1``     the static arms ``A = 0`` and ``A = 1``; ``a0`` is the reference
``sep10``          relapse component at 1, death component at 0: the separable direct
                   effect against ``a0`` and the separable indirect effect against ``a1``
``sep01``          relapse component at 0, death component at 1
=================  =======================================================================

:func:`engine_inputs` builds the separable engine's inputs on a sample with the *exact*
treatment and censoring mechanisms and its empirical cell hazards, so the robustness cells can
start the targeting from a hazard that is wrong in one factor while the mechanism is right.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import Separable
from cleverly.longitudinal.separable import SeparableInputs, SeparableState
from tests import discrete_law_point_survival as point

RELAPSE, DEATH = 1, 2
CAUSES = {"relapse": RELAPSE, "death": DEATH}
FIRST = "death"
HORIZONS: tuple[int, ...] = (1, 2, 3)
N_LEVELS = 3

LAW = point.survival_law(causes=2)

STATIC: dict[str, int] = {"a0": 0, "a1": 1}
SEPARABLE_ARMS: dict[str, dict[str, int]] = {
    "sep10": {"relapse": 1, "death": 0},
    "sep01": {"relapse": 0, "death": 1},
}
REGIMENS: dict[str, Any] = {
    **STATIC,
    **{label: Separable(arms=arms, first=FIRST) for label, arms in SEPARABLE_ARMS.items()},
}
REFERENCE = "a0"

LEVELS = tuple(REGIMENS)
MEANS: tuple[str, ...] = tuple(
    f"cif_regimen[{label}, {cause} @ t={t}]"
    for label in LEVELS
    for cause in CAUSES
    for t in HORIZONS
)
#: ``sep10`` against ``a0`` for death at ``t = 1`` is left out: both read
#: :math:`\lambda_D(1 \mid 0)` alone, so the fit reports zero with no spread on every sample.
CONTRASTS: tuple[str, ...] = tuple(
    f"ate_regimen[{label} vs {REFERENCE}, {cause} @ t={t}]"
    for label in LEVELS
    if label != REFERENCE
    for cause in CAUSES
    for t in HORIZONS
    if (label, cause, t) != ("sep10", "death", 1)
)
ESTIMANDS: tuple[str, ...] = (*MEANS, *CONTRASTS)

_NAME = re.compile(r"^(cif|ate)_regimen\[(\w+)(?: vs (\w+))?, (\w+) @ t=(\d+)\]$")


def parse(estimand: str) -> tuple[str, str | None, str, int]:
    """``(regimen, reference or None, cause, horizon)`` of a reported name."""
    match = _NAME.match(estimand)
    if match is None:  # pragma: no cover - guards the declared names
        raise KeyError(estimand)
    _, left, right, cause, horizon = match.groups()
    return left, right, cause, int(horizon)


def _codes(label: str) -> dict[int, int]:
    """The arms of a separable regimen by event code."""
    return {CAUSES[cause]: level for cause, level in SEPARABLE_ARMS[label].items()}


def incidence(probs: Any, label: str, cause: str, horizon: int) -> Any:
    """One regimen's incidence of ``cause`` by ``horizon``, analytic in ``probs``."""
    code = CAUSES[cause]
    if label in STATIC:
        return LAW.functional(probs, point.static(STATIC[label]), horizon, cause=code)
    return LAW.separable(probs, _codes(label), CAUSES[FIRST], horizon, code)


def functional(probs: Any, estimand: str) -> Any:
    """Any reported parameter, analytic in ``probs``."""
    left, right, cause, horizon = parse(estimand)
    value = incidence(probs, left, cause, horizon)
    return value if right is None else value - incidence(probs, right, cause, horizon)


def _incidence_eif(label: str, cause: str, horizon: int) -> np.ndarray:
    code = CAUSES[cause]
    if label in STATIC:
        return LAW.eif(point.static(STATIC[label]), horizon, code)
    return LAW.separable_eif(_codes(label), CAUSES[FIRST], horizon, code)


def eif(estimand: str) -> np.ndarray:
    """The EIF of ``estimand`` at every support point."""
    left, right, cause, horizon = parse(estimand)
    curve = _incidence_eif(left, cause, horizon)
    if right is not None:
        curve = curve - _incidence_eif(right, cause, horizon)
    return curve


def efficiency_sd(estimand: str) -> float:
    r""":math:`\sqrt{E_P[D^*(O)^2]}`, the efficient spread of one observation."""
    return float(np.sqrt(np.sum(LAW.probs * eif(estimand) ** 2)))


TRUTH: dict[str, float] = {name: float(functional(LAW.probs, name)) for name in ESTIMANDS}

_FRAME = LAW.frame()
_SUPPORT_FRAME = _FRAME.iloc[LAW.first_row_of()].reset_index(drop=True)


def fit_columns() -> dict[str, Any]:
    """The column keywords of ``LTMLE.fit`` for this law."""
    return LAW.fit_columns()


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` independent rows of the law."""
    cells = np.random.default_rng(seed).choice(len(LAW.support), size=n, p=LAW.probs)
    return _SUPPORT_FRAME.iloc[cells].reset_index(drop=True)


def exact_frame() -> pd.DataFrame:
    """The sample whose empirical law is the law exactly."""
    return _FRAME.copy()


# ------------------------------------------------------------------ engine inputs


def _indicators(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n, T)`` at risk, death and relapse indicators, read from the wide frame."""
    n, n_times = len(frame), len(HORIZONS)
    at_risk = np.zeros((n, n_times), dtype=bool)
    event_d = np.zeros((n, n_times))
    event_y = np.zeros((n, n_times))
    alive = np.ones(n, dtype=bool)
    for k in range(n_times):
        at_risk[:, k] = alive & (frame[f"C{k + 1}"].to_numpy() == 1)
        died = np.nan_to_num(frame[f"D{k + 1}"].to_numpy()) == 1
        relapsed = np.nan_to_num(frame[f"R{k + 1}"].to_numpy()) == 1
        event_d[:, k] = np.where(at_risk[:, k], died, 0.0)
        event_y[:, k] = np.where(at_risk[:, k], relapsed, 0.0)
        alive = at_risk[:, k] & ~died & ~relapsed
    return at_risk, event_d, event_y


def exact_inverse(frame: pd.DataFrame) -> np.ndarray:
    r"""``(n, T, L)`` :math:`1 / (g(a \mid W) G_k(a, W))` from the law's own tables."""
    w = frame["W"].to_numpy(dtype=int)
    inverse = np.zeros((len(frame), len(HORIZONS), N_LEVELS))
    for level in range(N_LEVELS):
        cumulative = point.G[w, level].copy()
        for node in HORIZONS:
            cumulative = cumulative * np.array([point._retained(node, x, level, 0) for x in w])
            inverse[:, node - 1, level] = 1.0 / cumulative
    return inverse


def empirical_hazards(frame: pd.DataFrame) -> np.ndarray:
    r"""``(n, T, L, 2)`` cell-mean cause-specific hazards :math:`\lambda_l(k \mid a, W)`.

    The share of the units at risk in each ``(W, A, node)`` cell with each cause, which is
    the saturated maximum-likelihood fit, computed here without the package.  An empty cell
    takes the law's own hazard, so a draw that leaves a cell empty still has a defined start.
    """
    at_risk, event_d, event_y = _indicators(frame)
    w = frame["W"].to_numpy(dtype=int)
    a = frame["A"].to_numpy(dtype=int)
    hazards = np.zeros((len(frame), len(HORIZONS), N_LEVELS, 2))
    for ww in (0, 1):
        for level in range(N_LEVELS):
            cell = (w == ww) & (a == level)
            for node in HORIZONS:
                rows = cell & at_risk[:, node - 1]
                for code, events in ((RELAPSE, event_y), (DEATH, event_d)):
                    if rows.any():
                        value = float(events[rows, node - 1].mean())
                    else:
                        value = float(LAW._node_hazard(LAW.probs, ww, level, node, code))
                    hazards[w == ww, node - 1, level, code - 1] = value
    return hazards


def engine_inputs(frame: pd.DataFrame) -> tuple[SeparableInputs, SeparableState]:
    """The engine's inputs at the exact mechanism, and the empirical factors."""
    at_risk, event_d, event_y = _indicators(frame)
    inputs = SeparableInputs(
        at_risk=at_risk,
        event_d=event_d,
        event_y=event_y,
        observed=frame["A"].to_numpy(dtype=int),
        inverse=exact_inverse(frame),
        weights=np.ones(len(frame)),
        alpha=1.0 - 1e-6,
    )
    none = np.zeros((len(HORIZONS), N_LEVELS), dtype=bool)
    state = SeparableState.from_hazards(
        empirical_hazards(frame), DEATH - 1, inputs.bounds, (none,) * 4
    )
    return inputs, state


def logit_shift(frame: pd.DataFrame, by_level: tuple[float, float, float]) -> np.ndarray:
    """``(n, T, L)`` a logit shift that depends on ``W`` and the level, the same at each node."""
    w = frame["W"].to_numpy(dtype=int)
    per_unit = np.where(w == 0, 0.6, -0.5)
    stacked = np.stack([per_unit] * len(HORIZONS), axis=1)
    return stacked[:, :, None] * np.asarray(by_level)[None, None]


#: The declared wrong-factor shifts: death's hazard and relapse's hazard among the death-free.
SHIFT_D = (1.5, -1.0, 0.5)
SHIFT_Y = (0.8, 2.0, -1.0)
