"""Sizing rules and a resumable runner shared by ``mtp-boundaries``, ``mtp-pricing`` and
``mtp-tuning``.

**The arm-pooled calibration rule.**  A calibration positive passes when the 99% bootstrap
interval of its SE ratio lies inside ``[0.93, 1.07]`` and its coverage interval lies inside
``[0.92, 0.98]``.  The SE-ratio half binds.  With ``R`` replications the estimated ratio is
about ``N(rho, sigma_R^2)``, ``sigma_R = c / sqrt(2 R)``, and the interval is
``rho_hat +/- 2.576 sigma_R``.  :data:`C_FIT` is fitted so that a perfectly calibrated cell
passes with probability 0.95 at ``R = 2,400``, the pass probability the beta brief records.

The arms' true ratios are unknown before the run, so a pilot of ``P`` draws per arm estimates
them.  The predictive model is ``rho_j = m + e + u_j``: ``m`` is the observed pooled pilot
mean, ``e ~ N(0, sigma_P^2 / arms)`` is the error of that pooled mean shared by every arm,
and ``u_j ~ N(0, tau^2)`` is each arm's own departure.  ``sigma_P = c / sqrt(2 P)``, and
``tau`` sits at its 80% upper bound from the chi-square of the arm means.  The family-wise
pass probability is ``E_e[prod_j P_j(e)]``, computed by Gauss-Hermite quadrature with 80
nodes.  ``R`` is the smallest multiple of 400 up to 40,000 with family-wise probability at
least 0.95 and per-cell probability at least 0.90.  The model is the reviewer's
(``reviews/x40/rev2/pass_prob_lib.py`` in the beta plans); ``tests/unit/test_mtp_boundaries_
design.py`` reproduces its recorded outputs.

**The per-arm screen.**  Before pooling, each arm's pilot SE ratio must satisfy
``|rho_hat_j - 1| <= 2.576 sigma_P + 0.01`` (:func:`screen_half_width`).  An arm outside the
band is diagnosed alone before ``R`` is set from the others.

**Parts.**  :func:`run_in_parts` fits a list of keys in chunks, writes each finished chunk to
its own compressed CSV, and skips a chunk whose file exists, so a run that a container restart
stops resumes where it stopped.  The assembled rows do not depend on the chunking.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from cleverly.utils.parallel import map_parallel

#: The two-sided 99% critical value of the SE-ratio interval.
Z = float(norm.ppf(0.995))
#: The half-width of the calibration SE-ratio band, ``[0.93, 1.07]``.
HALF = 0.07
#: The smallest and largest ``R`` the rule considers, and its step.
R_STEP = 400
R_MAX = 40_000
#: The family-wise and per-cell pass probabilities the rule requires.
FAMILY_TARGET = 0.95
CELL_TARGET = 0.90
#: The pilot mean's distance from 1 at which a family is a defect suspect.
OFF_MEAN = 0.015
#: The slack the per-arm screen adds to ``2.576 sigma_P``.
SCREEN_SLACK = 0.01
#: The learner screen's band on the pilot SE ratio (``mtp-pricing``).
LEARNER_BAND = (0.95, 1.05)

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def cell_pass(rho: Any, replicates: int, c: float) -> np.ndarray:
    """The probability that a cell with true SE ratio ``rho`` passes at ``replicates``."""
    rho = np.asarray(rho, dtype=float)
    sigma = c / np.sqrt(2.0 * replicates)
    slack = HALF - Z * sigma
    if slack <= 0:
        return np.zeros_like(rho)
    return np.asarray(
        norm.cdf((slack - (rho - 1.0)) / sigma) - norm.cdf((-slack - (rho - 1.0)) / sigma)
    )


def _calibrate_c() -> float:
    low, high = 0.9, 1.3
    for _ in range(60):
        middle = 0.5 * (low + high)
        if cell_pass(np.array([1.0]), 2400, middle)[0] > 0.95:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


#: The spread constant of the SE-ratio estimate, ``1.0692``.
C_FIT = _calibrate_c()


def pilot_sigma(pilot: int, c: float = C_FIT) -> float:
    """``sigma_P``: the standard deviation of one arm's pilot SE ratio at ``pilot`` draws."""
    return float(c / np.sqrt(2.0 * pilot))


def tau_upper(pilot: int, arms: int, c: float = C_FIT) -> float:
    """The 80% upper bound of the between-arm spread ``tau`` when the arms agree."""
    sigma = pilot_sigma(pilot, c)
    factor = chi2.median(arms - 1) / chi2.ppf(0.2, arms - 1)
    return float(sigma * np.sqrt(max(factor - 1.0, 0.0)))


def pass_probabilities(
    replicates: int, mean: float, shared: float, tau: float, arms: int, c: float = C_FIT
) -> tuple[float, float]:
    """``(per-cell, family-wise)`` pass probability under the arm-pooled model.

    ``shared`` is the variance of the pooled pilot mean, ``sigma_P^2 / arms``.
    """
    per_e = []
    for e in np.sqrt(shared) * _GH_X:
        rho = mean + e + tau * _GH_X
        per_e.append(np.sum(_GH_W * cell_pass(rho, replicates, c)))
    values = np.asarray(per_e)
    return float(np.sum(_GH_W * values)), float(np.sum(_GH_W * values**arms))


def smallest_replicates(
    mean: float, pilot: int, arms: int, *, tau: float | None = None, c: float = C_FIT
) -> tuple[int | None, float, float]:
    """``(R, per-cell, family-wise)``: the smallest ``R`` the rule accepts, or ``None``.

    ``tau`` defaults to :func:`tau_upper`.  Where no ``R`` up to :data:`R_MAX` is accepted the
    probabilities are those at :data:`R_MAX`.
    """
    sigma = pilot_sigma(pilot, c)
    spread = tau_upper(pilot, arms, c) if tau is None else tau
    for replicates in range(R_STEP, R_MAX + 1, R_STEP):
        cell, family = pass_probabilities(replicates, mean, sigma**2 / arms, spread, arms, c)
        if family >= FAMILY_TARGET and cell >= CELL_TARGET:
            return replicates, cell, family
    return None, *pass_probabilities(R_MAX, mean, sigma**2 / arms, spread, arms, c)


def tau_from_pilot(ratios: Sequence[float], pilot: int, c: float = C_FIT) -> float:
    """The 80% upper bound of ``tau`` from observed pilot SE ratios, one per arm.

    The arm means' sum of squares about their mean, over ``sigma_P^2``, is chi-square with
    ``arms - 1`` degrees of freedom plus the ``tau^2`` excess.  The bound solves
    ``(sum of squares) / (sigma_P^2 + tau^2) = chi2.ppf(0.2, arms - 1)``.
    """
    values = np.asarray(ratios, dtype=float)
    sigma = pilot_sigma(pilot, c)
    squares = float(np.sum((values - values.mean()) ** 2))
    bound = squares / chi2.ppf(0.2, values.size - 1) - sigma**2
    return float(np.sqrt(max(bound, 0.0)))


def replicates_from_pilot(ratios: Sequence[float], pilot: int, c: float = C_FIT) -> dict[str, Any]:
    """The declared rule applied to a pilot: the screen, the contingency and ``R``.

    Returns the screened arms, the pooled mean of the arms inside the band, whether the
    family is a defect suspect, and ``(R, per-cell, family-wise)`` from the arms inside the
    band.  The predictive ``tau`` is the larger of :func:`tau_upper` (agreeing arms) and
    :func:`tau_from_pilot` (the observed spread), so a spread pilot never lowers ``R``.
    """
    values = np.asarray(ratios, dtype=float)
    half = screen_half_width(pilot, c)
    inside = np.abs(values - 1.0) <= half
    kept = values[inside]
    if kept.size < 2:
        return {"screened": inside.tolist(), "mean": float("nan"), "suspect": True, "R": None}
    mean = float(kept.mean())
    spread = max(tau_upper(pilot, kept.size, c), tau_from_pilot(kept, pilot, c))
    replicates, cell, family = smallest_replicates(mean, pilot, kept.size, tau=spread, c=c)
    return {
        "screened": inside.tolist(),
        "mean": mean,
        "tau": spread,
        "suspect": bool(abs(mean - 1.0) >= OFF_MEAN or replicates is None),
        "R": replicates,
        "per_cell": cell,
        "family_wise": family,
    }


def screen_half_width(pilot: int, c: float = C_FIT) -> float:
    """The per-arm screen's half-width, ``2.576 sigma_P + 0.01``."""
    return Z * pilot_sigma(pilot, c) + SCREEN_SLACK


def learner_screen_pass(pilot: int, c: float = C_FIT) -> float:
    """The probability that a calibrated candidate's pilot SE ratio lands in the band."""
    sigma = pilot_sigma(pilot, c)
    return float(2.0 * norm.cdf((LEARNER_BAND[1] - 1.0) / sigma) - 1.0)


def necessity_power(
    standardized_bias: float, replicates: int, margin: float = 0.25, z: float = Z
) -> float:
    """The probability that a control's bias is established outside the margin.

    ``bias_discriminated`` needs the 99% interval of the standardized bias to clear
    ``margin``.  The standardized bias has standard error about ``1 / sqrt(R)``, so the
    probability is ``Phi(sqrt(R) (|delta| - margin) - z)``.
    """
    return float(norm.cdf(np.sqrt(replicates) * (abs(standardized_bias) - margin) - z))


def equivalence_power(
    standardized_bias: float, replicates: int, margin: float = 0.25, z: float = Z
) -> float:
    """The probability that a positive arm's bias is established inside the margin."""
    root = np.sqrt(replicates)
    upper = norm.cdf(root * (margin - standardized_bias) - z)
    lower = norm.cdf(root * (-margin - standardized_bias) + z)
    return float(max(upper - lower, 0.0))


# ---------------------------------------------------------------------------- parts


def parts_root(default: Path | None = None) -> Path | None:
    """Where a run's parts go: the driver's checkpoint root, else ``default``.

    Under ``tests/canonical/regenerate.py`` the property phase runs inside the driver's
    checkpoint context, whose scratch survives a container restart.  Outside it a caller
    names the directory, or the environment variable ``CLEVERLY_STUDY_PARTS`` does.
    """
    from tests.studies.evidence.replication import checkpoint_root

    root = checkpoint_root()
    if root is not None:
        return root / "properties"
    if default is not None:
        return default
    named = os.environ.get("CLEVERLY_STUDY_PARTS")
    return Path(named) if named else None


def run_in_parts(
    fit_one: Callable[..., list[dict[str, Any]]],
    keys: Sequence[tuple[Hashable, ...]],
    *,
    label: str,
    root: Path | None,
    chunk: int = 50,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit every key, ``chunk`` keys to a part, and resume from the parts on disk.

    ``fit_one(*key)`` returns a list of row dictionaries.  Each finished part is written to
    ``root / label / part-<index>.csv.gz`` through a temporary file and an atomic rename, so
    a part on disk is always complete.  With ``root=None`` nothing is written.
    """
    frames: list[pd.DataFrame] = []
    directory = None if root is None else Path(root) / label
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(keys), chunk)):
        batch = list(keys[start : start + chunk])
        path = None if directory is None else directory / f"part-{index:05d}.csv.gz"
        if path is not None and path.exists():
            frames.append(pd.read_csv(path, float_precision="round_trip"))
            continue
        outcomes = map_parallel(fit_one, batch, n_jobs=n_jobs)
        part = pd.DataFrame([row for rows in outcomes for row in rows])
        if path is not None:
            temporary = path.with_suffix(".tmp")
            part.to_csv(temporary, index=False, compression="gzip", float_format="%.17g")
            os.replace(temporary, path)
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
