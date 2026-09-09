"""Exact continuous law for learned weighted point-treatment studies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SELECTION_SLOPE = 0.75
EFFECT_MODIFICATION = 2.0
TARGET_ATE = 1.0
TREATMENT_PROBABILITY = 0.5
OUTCOME_NOISE_SD = 1.0
SELECTED_W1_MEAN = SELECTION_SLOPE / 3.0
SELECTED_ATE = TARGET_ATE + EFFECT_MODIFICATION * SELECTED_W1_MEAN


def weighted_ate_efficiency_sd() -> float:
    """Exact selected-law SD of the target-population ATE influence curve.

    The estimator normalizes the observation weights.  ``cleverly`` averages the targeted
    predictions with :func:`numpy.average`, so the reported functional is
    ``E_sel[h * b] / E_sel[h]`` rather than ``E_sel[h * b]``.  The pathwise derivative of a
    ratio functional carries the centering *inside* the weight, so the gradient this estimator
    has is ``h * (cc * (Y - Q) + b - psi)`` with ``h`` the density ratio, ``cc`` the clever
    covariate and ``b`` the conditional effect.  This function returns the selected-law
    standard deviation of that curve.

    The unnormalized alternative ``h * (cc * (Y - Q) + b) - psi`` is the Horvitz-Thompson
    gradient.  It returns 2.4525 here, and it is not the gradient of the functional this
    estimator reports.  The sibling ``tmle_weighted`` study is the witness: it fits
    oracle-correct nuisances against the same inside-the-weight convention in
    ``tests/studies/weighted_point_common.py``, and it publishes an ``efficiency_reported``
    interval of 0.99651 to 0.99972 against the bound that convention gives.

    This study uses the returned value as one noise unit for its calibration control.  It does
    not claim the estimator attains the bound, because the study fits a main-effects outcome
    regression that omits the treatment-effect modification.

    Returns
    -------
    float
        The selected-law standard deviation of the target-population ATE influence curve.
    """
    # A coincidence of this law's constants, recorded so the next reader is not misled.  With
    # ``EFFECT_MODIFICATION == 2`` and ``TREATMENT_PROBABILITY == 0.5``, the variance the study
    # achieves under its misspecified main-effects regression, ``E[h**2 * 4 * (1 + W1**2)]``,
    # is algebraically the same number as the bound at the true regression,
    # ``E[h**2 * (4 + EFFECT_MODIFICATION**2 * W1**2)]``.  This study's own artifacts therefore
    # cannot separate "computes the bound" from "computes the achieved variance".  The
    # ``tmle_weighted`` study separates them, because its nuisances are oracle-correct.
    slope = SELECTION_SLOPE
    probability = TREATMENT_PROBABILITY
    log_ratio = math.log((1.0 + slope) / (1.0 - slope))
    # The two moments of ``1 / (1 + slope * w)`` against ``w**0`` and ``w**2`` on ``[-1, 1]``.
    first = log_ratio / slope
    second = log_ratio / slope**3 - 2.0 / slope**2
    residual = OUTCOME_NOISE_SD**2 * (1.0 / probability + 1.0 / (1.0 - probability))
    return math.sqrt(0.5 * residual * first + 0.5 * EFFECT_MODIFICATION**2 * second)


def selected_density(w1: np.ndarray) -> np.ndarray:
    """Density of the selected ``W1`` law on ``[-1, 1]``."""
    values = np.asarray(w1, dtype=float)
    return (1.0 + SELECTION_SLOPE * values) / 2.0


def inverse_selection_weight(w1: np.ndarray) -> np.ndarray:
    """Radon-Nikodym derivative from the selected law to the uniform target law."""
    values = np.asarray(w1, dtype=float)
    return 1.0 / (1.0 + SELECTION_SLOPE * values)


def outcome_mean(
    w1: np.ndarray, w2: np.ndarray, treatment: np.ndarray, *, effect: float
) -> np.ndarray:
    """Conditional outcome mean under the declared effect and effect modification."""
    return (
        0.5
        + 0.5 * np.asarray(w1, dtype=float)
        + 0.25 * np.asarray(w2, dtype=float)
        + np.asarray(treatment, dtype=float)
        * (effect + EFFECT_MODIFICATION * np.asarray(w1, dtype=float))
    )


def truths(*, effect: float = TARGET_ATE) -> dict[str, float]:
    """Exact target-law arm means and ATE."""
    return {"ey0": 0.5, "ey1": 0.5 + effect, "ate": effect}


def sample_selected(n: int, seed: int, *, effect: float = TARGET_ATE) -> pd.DataFrame:
    """Draw exactly ``n`` rows from the selected continuous law."""
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(size=n)
    slope = SELECTION_SLOPE
    # The other quadratic root is outside [-1, 1], so this branch is exact.
    w1 = (-1.0 + np.sqrt((1.0 - slope) ** 2 + 4.0 * slope * uniform)) / slope
    w2 = rng.uniform(-1.0, 1.0, size=n)
    treatment = rng.binomial(1, TREATMENT_PROBABILITY, size=n).astype(float)
    outcome = outcome_mean(w1, w2, treatment, effect=effect) + rng.normal(
        scale=OUTCOME_NOISE_SD, size=n
    )
    return pd.DataFrame(
        {
            "Y": outcome,
            "A": treatment,
            "W1": w1,
            "W2": w2,
            "obs_weight": inverse_selection_weight(w1),
        }
    )
