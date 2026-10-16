"""Laws and helpers for the declared support and declared atoms of a continuous dose (X40).

The doses here are unrounded draws, so the undeclared-atom check (R-b4) passes on them with
its seam on.  A declared atom is planted as an exact repeated value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


class NeverFit:
    """A learner that raises if fitted, the witness that a refusal precedes every learner."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> NeverFit:
        return self

    def fit(self, *args: Any, **kwargs: Any) -> NeverFit:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, x: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")

    def predict_proba(self, x: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used before the refusal")


def upper(h: Any) -> np.ndarray:
    """``u(h) = 1 + W``: the declared upper support bound of every dose below."""
    return 1.0 + np.asarray(h["W"], dtype=float)


def point_frame(n: int = 400, seed: int = 0, *, atom: float | None = None) -> pd.DataFrame:
    """``W`` binary, ``A ~ U(0, 1 + W)``, ``Y`` binary; the first row has ``W = 1``.

    ``atom`` puts that dose on a quarter of the rows.
    """
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    w[0] = 1.0
    a = rng.uniform(0.0, 1.0 + w)
    if atom is not None:
        a = np.where(rng.random(n) < 0.25, atom, a)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * a - 0.4 * w)))
    return pd.DataFrame({"W": w, "A": a, "Y": rng.binomial(1, p).astype(float)})


def panel_frame(n: int = 400, seed: int = 0, *, atom: float | None = None) -> pd.DataFrame:
    """Two nodes, each ``A_t ~ U(0, 1 + W)``; ``atom`` plants that dose at node 2."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    w[0] = 1.0
    a1 = rng.uniform(0.0, 1.0 + w)
    l2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(a1 - 0.7)))).astype(float)
    a2 = rng.uniform(0.0, 1.0 + w)
    if atom is not None:
        a2 = np.where(rng.random(n) < 0.25, atom, a2)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.5 * a1 + 0.6 * a2 - 0.3 * l2)))
    return pd.DataFrame(
        {"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": rng.binomial(1, p).astype(float)}
    )


#: The fit columns of :func:`panel_frame`.
PANEL: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}


def point_learners(**overrides: Any) -> dict[str, Any]:
    """Fast learners for a point policy fit."""
    return {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "cross_fit": False,
        **overrides,
    }


def panel_learners(**overrides: Any) -> dict[str, Any]:
    """Fast learners for a longitudinal policy fit."""
    return {
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 3,
        **overrides,
    }


# ------------------------------------------------------------------ R-b4 witness laws
#
# The laws of ``X40-plan.md`` section 4.1, T-b8 (its probe ``rev3.1/rule_isolation31.py``):
# a dose of a truncated normal on [3, 9], and one law per rule of the undeclared-atom check
# on which that rule alone fires.


def smooth_dose(n: int, rng: np.random.Generator) -> np.ndarray:
    """A truncated normal on ``[3, 9]``, mean 6, scale 1.5."""
    from scipy.stats import truncnorm

    return np.asarray(truncnorm.rvs(-2, 2, loc=6, scale=1.5, size=n, random_state=rng))


def one_atom(n: int, rng: np.random.Generator) -> np.ndarray:
    """Rule (A): the smooth dose with 1.5% of rows at the unrounded price 6.37."""
    a = smooth_dose(n, rng)
    a[rng.random(n) < 0.015] = 6.37
    return a


def grid(step: float) -> Any:
    """Rules (B) and (C): the smooth dose rounded to a grid of ``step``."""

    def law(n: int, rng: np.random.Generator) -> np.ndarray:
        return np.round(smooth_dose(n, rng) / step) * step

    return law


def list_prices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Rule (D): 40% of rows on 200 list prices in [3, 9], the dose rounded to cents."""
    a = smooth_dose(n, rng)
    on = rng.random(n) < 0.40
    a[on] = np.linspace(3.0, 9.0, 200)[rng.integers(0, 200, on.sum())]
    return np.round(a, 2)


def cents(n: int, rng: np.random.Generator) -> np.ndarray:
    """The null: the smooth dose rounded to cents, with no atom."""
    return np.round(smooth_dose(n, rng), 2)


def dose_frame(dose: np.ndarray, seed: int, *, jump: float = 0.0, atom: float = 0.0) -> Any:
    """A point frame around a given dose: ``W`` binary and ``Y`` binary.

    ``jump`` adds that much to the logit of ``Y`` on the rows at ``atom``.
    """
    rng = np.random.default_rng(seed)
    n = dose.size
    w = rng.integers(0, 2, n).astype(float)
    logit = -2.0 + 0.3 * dose + 0.4 * w + jump * (np.abs(dose - atom) < 1e-12)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return pd.DataFrame({"W": w, "A": dose, "Y": y})


# ------------------------------------------------------------------ a jump at the atom (T-b4)

#: The declared atom of the jump laws, and the jump of the outcome logit there.
JUMP_ATOM = 3.0
JUMP = 2.5
#: The share of rows at the atom.
JUMP_SHARE = 0.3


def jump_regression(a: Any, w: Any) -> np.ndarray:
    """``m(a, w)``: a logistic regression in the dose with a jump of :data:`JUMP` at the atom."""
    a = np.asarray(a, dtype=float)
    logit = -1.0 + 0.3 * a + 0.4 * np.asarray(w, dtype=float) + JUMP * (a == JUMP_ATOM)
    return np.asarray(1.0 / (1.0 + np.exp(-logit)))


def jump_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """``W`` binary, ``A`` at :data:`JUMP_ATOM` on 30% of rows and ``U(0, 4)`` otherwise."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a = np.where(rng.random(n) < JUMP_SHARE, JUMP_ATOM, rng.uniform(0.0, 4.0, n))
    y = rng.binomial(1, jump_regression(a, w)).astype(float)
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def jump_truth() -> float:
    """``E[Y^d]`` on :func:`jump_frame` under "to atom": ``a + 1`` below 1.5, the atom on
    ``[1.5, 2.5)``, the identity above, by quadrature."""
    grid = np.linspace(0.0, 4.0, 400_001)
    total = 0.0
    for w in (0.0, 1.0):
        assigned = np.where(grid < 1.5, grid + 1.0, np.where(grid < 2.5, JUMP_ATOM, grid))
        values = jump_regression(assigned, np.full(grid.size, w))
        continuous = float(np.trapezoid(values, grid) / 4.0)
        atom = float(jump_regression(JUMP_ATOM, w))
        total += 0.5 * (JUMP_SHARE * atom + (1.0 - JUMP_SHARE) * continuous)
    return total


def jump_panel(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Two nodes; node 2 is :func:`jump_frame`'s dose, and the outcome jumps at its atom."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a1 = rng.uniform(0.0, 1.0 + w)
    l2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-(a1 - 0.7)))).astype(float)
    a2 = np.where(rng.random(n) < JUMP_SHARE, JUMP_ATOM, rng.uniform(0.0, 4.0, n))
    p = jump_regression(a2, w) * (0.8 + 0.2 * l2) * (0.9 + 0.05 * a1)
    return pd.DataFrame(
        {"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": rng.binomial(1, np.clip(p, 0, 1)).astype(float)}
    )
