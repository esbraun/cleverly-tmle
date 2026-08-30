"""Shared exact-law helpers for controlled direct-effect evidence studies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tests import discrete_law_cde as cde


def probabilities(
    qbar: np.ndarray = cde.QBAR,
    *,
    g: np.ndarray = cde.G,
    qz: np.ndarray = cde.QZ,
    pi: np.ndarray = cde.PI,
    p_w: np.ndarray = cde.P_W,
) -> np.ndarray:
    """Build an observed-data CDE law from its four nuisance functions."""
    out = np.empty((len(p_w), 2, 2, 3), dtype=float)
    for w, a, z, kind in cde.SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        level = qz[w, a] if z == 1 else 1.0 - qz[w, a]
        observed = pi[w, a]
        if kind == cde.UNOBSERVED:
            cell = 1.0 - observed
        else:
            outcome = qbar[w, a, z] if kind == cde.OBSERVED_ONE else 1.0 - qbar[w, a, z]
            cell = observed * outcome
        out[w, a, z, kind] = p_w[w] * arm * level * cell
    if not np.isclose(out.sum(), 1.0):  # pragma: no cover - declaration guard
        raise AssertionError("the declared CDE cell probabilities do not sum to one")
    if np.any(out <= 0.0):  # pragma: no cover - declaration guard
        raise AssertionError("the declared CDE law must give every support point positive mass")
    return out


def sample_discrete(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows from a declared finite controlled direct-effect law."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(cde.SUPPORT), size=n, p=np.asarray(probs).reshape(-1))
    values = np.asarray(cde.SUPPORT, dtype=float)[cells]
    kind = values[:, 3]
    return pd.DataFrame(
        {
            "W": values[:, 0],
            "A": values[:, 1],
            "Z": values[:, 2],
            "Y": np.where(kind == cde.UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == cde.UNOBSERVED, 0.0, 1.0),
        }
    )


def truths(probs: np.ndarray, estimands: Sequence[str], level: int) -> dict[str, float]:
    """Evaluate each exact controlled parameter on its public reporting scale."""
    out = {name: float(cde.functional(probs, name, level)) for name in estimands}
    for name in ("rr", "or"):
        if name in out:
            out[name] = float(np.exp(out[name]))
    return out


def efficiency_sd(probs: np.ndarray, estimand: str, level: int) -> float:
    r"""Return :math:`\sqrt{E_P[D^*(O)^2]}` on the inference scale."""
    law = cde.DiscreteLaw(probs)
    # ``cde.eif`` is defined at the module's base law.  The registered efficiency claim
    # uses that law only; refuse a quiet extension to a different probability array.
    if not np.array_equal(law.probs, cde.PROBS):
        raise ValueError("the exact CDE efficiency bound is registered only for the base law")
    curve = cde.eif(estimand, level)
    return float(np.sqrt(np.sum(np.asarray(probs).reshape(-1) * curve**2)))


def level_rows(levels: Sequence[int], suffixes: Sequence[str]) -> tuple[str, ...]:
    """Build the shared ``z0__...`` and ``z1__...`` property-cell names."""
    return tuple(f"z{level}__{suffix}" for level in levels for suffix in suffixes)
