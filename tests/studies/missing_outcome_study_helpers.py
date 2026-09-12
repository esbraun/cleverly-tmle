"""Shared finite-law helpers for missing-outcome evidence studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

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


class FailTreatment(BaseEstimator):
    """A treatment learner whose use is the failure the natural-course target must expose.

    Shared because four call sites need the same class: the canonical driver, the
    property study, and the two exact-law modules. The natural-course mean fits no
    treatment mechanism at all, so any call here is a defect rather than a worse fit.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FailTreatment:
        raise AssertionError("the natural-course mean must not fit a treatment mechanism")

    def predict_proba(self, X: Any) -> Any:
        raise AssertionError("the natural-course mean must not predict a treatment mechanism")


#: The misspecified outcome regression the MAR property studies declare: ``1 - Qbar``.
WRONG_Q = 1.0 - mar.Q
#: The misspecified response mechanism the MAR property studies declare, indexed ``[w, a]``.
WRONG_PI = np.array([[0.80, 0.75], [0.55, 0.45], [0.30, 0.25]])


def _levels(w: Any) -> np.ndarray:
    """Read the ``W`` level from a vector or from the first column of a design block.

    Any later columns are covariates the finite law ignores, such as noise controls.
    """
    values = np.asarray(w, dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    return np.rint(values.reshape(-1)).astype(int)


class NaturalCourseLaw:
    """The finite MAR nuisance functions used by both primary outcome laws."""

    def __init__(self, *, q: np.ndarray = mar.Q, pi: np.ndarray = mar.PI) -> None:
        self.q = np.asarray(q, dtype=float)
        self.pi = np.asarray(pi, dtype=float)

    def outcome_mean(self, w: Any, a: Any, z: Any = None) -> np.ndarray:
        levels = _levels(w)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape).astype(int)
        return self.q[levels, arms]

    def missingness(self, w: Any, a: Any) -> np.ndarray:
        levels = _levels(w)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape).astype(int)
        return self.pi[levels, arms]

    def propensity(self, w: Any) -> np.ndarray:
        return mar.G[_levels(w)]


def natural_course_law(configuration: str) -> NaturalCourseLaw:
    """Return the oracle law for one double-robustness nuisance configuration."""
    q = mar.Q if configuration in {"both_correct", "outcome_correct"} else WRONG_Q
    pi = mar.PI if configuration in {"both_correct", "response_correct"} else WRONG_PI
    return NaturalCourseLaw(q=q, pi=pi)
