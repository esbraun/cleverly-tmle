"""Shared finite-law helpers for missing-outcome evidence studies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tests import discrete_law_mar as mar
from tests.studies.evidence.properties import finite_support_sample


def probabilities(
    q: np.ndarray = mar.Q,
    *,
    g: np.ndarray = mar.G,
    pi: np.ndarray = mar.PI,
    p_w: np.ndarray = mar.P_W,
) -> np.ndarray:
    """Build an observed-data MAR law from its three nuisance functions."""
    out = np.empty((len(p_w), 2, 3), dtype=float)
    for w, a, kind in mar.SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        observed = pi[w, a]
        if kind == mar.UNOBSERVED:
            cell = 1.0 - observed
        else:
            outcome = q[w, a] if kind == mar.OBSERVED_ONE else 1.0 - q[w, a]
            cell = observed * outcome
        out[w, a, kind] = p_w[w] * arm * cell
    if not np.isclose(out.sum(), 1.0):  # pragma: no cover - declaration guard
        raise AssertionError("the declared MAR cell probabilities do not sum to one")
    return out


def sample_discrete(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` observed-data rows from a declared finite MAR law."""
    return finite_support_sample(
        probs,
        mar.SUPPORT,
        n,
        seed,
        columns=("W", "A"),
        kind_axis=2,
        unobserved=mar.UNOBSERVED,
    )


def truths(probs: np.ndarray, estimands: Sequence[str]) -> dict[str, float]:
    """Evaluate the independent observed-data oracle for each estimand."""
    return {name: float(mar.functional(probs, name)) for name in estimands}


def efficiency_sd(probs: np.ndarray, estimand: str) -> float:
    r"""Standard deviation of the oracle observed-data influence curve.

    :math:`\sqrt{E_P[D^*(O)^2]}`, and the curve is the law's own Gateaux derivative rather
    than a second complex-step loop written beside it.  The mixture the derivative is taken
    along already centres :math:`D^*`, so the raw second moment is the variance.
    """
    curve = mar.eif(estimand, probs=probs)
    return float(np.sqrt(np.sum(np.asarray(probs).reshape(-1) * curve**2)))
