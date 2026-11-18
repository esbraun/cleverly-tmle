r"""The allocation preflight of the continuous-time engine.

:class:`~cleverly.survival.ContinuousSurvivalTMLE` holds hazards, survival curves, incidences
and weights for both arms of every unit at every distinct observed time up to the last time a
target reads: arrays of shape ``(2, n, M)`` and ``(2, J, n, M)``.  When ``M`` grows with ``n``
their size grows as ``n**2``.  :func:`peak_bytes` predicts the largest number of bytes the fit
holds at once, from the dimensions alone, and :func:`check_memory` refuses a fit whose prediction
exceeds the memory available, before any nuisance is fitted.

The prediction is the largest, over the phases of a fit, of the bytes live at that phase's peak.
Each phase is a sum of the arrays the code holds there, shape times item size, so each term can
be checked against the line that allocates it.  Write ``A = n M`` and ``JA = J n M``:

* ``nuisance``: the hazard jumps ``(2, J, n, M)`` and censoring jumps ``(2, n, M)`` being
  filled, beside one learner's ``(n_f, M)`` cumulative hazard and the two copies
  :func:`~cleverly.survival.engine._jumps` makes of it, for the largest prediction fold
  ``n_f``; a stratified Nelson-Aalen learner adds a per-cell cache of at most ``(n_f, M)``.
* ``curves``: the jumps beside the censoring jumps, survival, retention, the retention just
  before each time and the product of treatment and retention probabilities: ``16 JA + 80 A``.
* ``indicators``: the jumps and the product beside the at-risk and event indicators, which are
  boolean.
* ``weights``: the bounded inverse weight built from the product, beside it.
* ``walk``: the held indicators and weight, the current jumps and their evaluation, a trial's
  jumps and the trial evaluation's temporaries (:func:`~cleverly.survival._targeting.evaluate`),
  or the direction's (:func:`~cleverly.survival._targeting._direction`), whichever is larger.
* ``off_grid``: the walk's result beside one more evaluation at up to
  :data:`~cleverly.survival.engine.OFF_GRID_TIMES` random times.
* ``band``: the multiplier buffer of :func:`~cleverly.inference.simultaneous_bands`.

A truncation curve (``retained=True``) keeps the initial jumps and the product alive through
every walk, ``16 JA + 16 A`` more.  Terms in ``n P``, ``P^2`` and ``n p`` keep the prediction an
upper bound at small ``n``.  The prediction does not cover a user learner's own allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..exceptions import MethodConfigurationError
from ..inference.multiplier import _BLOCK_BYTES
from ..utils.memory import available_memory

#: Bytes of one float64.
F8: Final = 8
#: The band's multiplier buffer with its packed draw beside it.
BAND_BUFFER: Final = 2 * _BLOCK_BYTES
#: Allocations no array shape accounts for (small arrays, Python objects, a learner's fit),
#: with a margin.  It keeps the prediction an upper bound at small ``n`` and is negligible where
#: the guard can refuse.
FIXED: Final = 4 << 20
#: Random times of the off-grid diagnostic, ``engine.OFF_GRID_TIMES``; the engine imports this
#: module, so a test pins the two equal.
OFF_GRID: Final = 100


class ContinuousSurvivalMemoryError(MethodConfigurationError):  # numpydoc ignore=PR01
    """A continuous-time fit predicted to need more memory than its budget.

    Raised before any nuisance is fitted.  ``predicted`` and ``budget`` are in bytes, ``None``
    on a copy rebuilt from the refusal alone.
    """

    predicted: int | None = None
    budget: int | None = None


@dataclass(frozen=True)
class PeakEstimate:
    """The predicted peak of a continuous-time fit and the phase that reaches it.

    Parameters
    ----------
    phases : dict of str to int
        The predicted bytes live at the peak of each phase.
    n : int
        Units.
    causes : int
        Causes ``J``.
    times : int
        Grid size ``M``.
    rows : int
        Weight rows ``Q`` of one arm and cause.
    """

    phases: dict[str, int]
    n: int
    causes: int
    times: int
    rows: int

    @property
    def bytes(self) -> int:
        """The predicted peak, the largest phase."""
        return max(self.phases.values())

    @property
    def phase(self) -> str:
        """The phase that reaches the peak."""
        return max(self.phases, key=self.phases.__getitem__)


def peak_bytes(
    n: int,
    causes: int,
    times: int,
    rows: int,
    *,
    covariates: int,
    fold_size: int,
    simultaneous: bool,
    n_multiplier: int,
    retained: bool = False,
    factual_tilt: bool = False,
    stratified_learner: bool = False,
) -> PeakEstimate:
    """Predict the bytes a continuous-time fit holds at once, by phase.

    Parameters
    ----------
    n : int
        Units.
    causes : int
        Causes ``J``.
    times : int
        Grid size ``M``: the distinct observed times up to the last time a target reads.
    rows : int
        Weight rows ``Q``: target times plus restricted horizons.
    covariates : int
        Baseline covariates ``p``.
    fold_size : int
        The most units any one nuisance prediction covers: the largest outer fold, or ``n``
        without cross-fitting.
    simultaneous : bool
        Whether the fit builds a simultaneous band.
    n_multiplier : int
        The band's multiplier draws.
    retained : bool
        Whether the initial jumps and the weight product stay alive through every walk, as a
        truncation curve keeps them.
    factual_tilt : bool
        Whether the walk tilts only the received arm (``tilt="factual"``).
    stratified_learner : bool
        Whether a library holds a stratified Nelson-Aalen learner, whose prediction caches a
        curve per stratum.

    Returns
    -------
    PeakEstimate
        The bytes of each phase.
    """
    J, M = int(causes), int(times)
    A = int(n) * M  # cells of one (n, M) array
    JA = J * A
    P = 2 * J * int(rows)
    # The last two terms: the (2, n, K) retention at the targets and the (Q, M) target weights,
    # which live from the curves phase on.
    small = (
        FIXED
        + F8 * int(n) * (int(covariates) + 4) * 3
        + F8 * P * P * 4
        + 16 * int(n) * int(rows)
        + F8 * int(rows) * M
    )
    influence = F8 * int(n) * P  # one (n, P) float array
    fold = F8 * int(fold_size) * M * (4 if stratified_learner else 3)
    # A truncation curve also keeps the first replay's estimates and record beside each walk.
    held = 16 * A + 16 * JA + 8 * influence if retained else 0
    # evaluate's peak over its inputs: the survival and incidences beside either the first
    # arm's temporaries or the moment the second arm's weighted jumps replace the first's.
    evaluate = max(F8 * (4 * JA + 6 * A), F8 * (3 * JA + 8 * A)) + 3 * influence
    walk_held = 17 * A + JA  # boolean at-risk and events, float weight
    current = 16 * JA + 16 * A + 16 * JA + 2 * influence  # jumps, evaluation, initial curves
    trial = walk_held + current + 16 * JA + evaluate
    direction = walk_held + current + 16 * JA + 24 * JA + 16 * A + 16 * JA * factual_tilt
    tilt = walk_held + current + 32 * JA + 32 * A
    off_P = 2 * J * OFF_GRID
    off_grid = (
        walk_held
        + current
        + max(F8 * (4 * JA + 6 * A), F8 * (3 * JA + 8 * A))
        + 3 * F8 * int(n) * off_P
        + F8 * OFF_GRID * M * 2
    )
    band = (BAND_BUFFER + F8 * int(n_multiplier) * P * 3 + 3 * influence) if simultaneous else 0
    phases = {
        "nuisance": 16 * JA + 16 * A + fold,
        "curves": 16 * JA + 80 * A,
        "indicators": 16 * JA + 18 * A + 2 * JA,
        "weights": 16 * JA + JA + 49 * A,
        "walk": max(trial, direction, tilt),
        "off_grid": off_grid,
        "band": walk_held + current + band,
    }
    return PeakEstimate(
        {
            name: value + small + (held if name not in ("nuisance", "curves") else 0)
            for name, value in phases.items()
        },
        int(n),
        J,
        M,
        int(rows),
    )


def resident_count(plan: object, max_memory_bytes: int | None) -> int:
    """The blocks a walk keeps (placeholder: every block)."""
    return len(plan.bounds)  # type: ignore[attr-defined]


def _gib(value: int) -> str:
    return f"{value / 1024.0**3:.2f} GiB ({value:,} bytes)"


def check_memory(estimate: PeakEstimate, max_memory_bytes: int | None) -> None:
    """Refuse a fit whose predicted peak exceeds its memory budget.

    Parameters
    ----------
    estimate : PeakEstimate
        The prediction.
    max_memory_bytes : int or None
        The budget.  ``None`` reads the memory available now
        (:func:`~cleverly.utils.memory.available_memory`), and checks nothing where the
        platform does not report it.

    Raises
    ------
    ContinuousSurvivalMemoryError
        When the prediction exceeds the budget.
    """
    explicit = max_memory_bytes is not None
    budget = available_memory() if max_memory_bytes is None else int(max_memory_bytes)
    if budget is None or estimate.bytes <= budget:
        return
    source = f"max_memory_bytes={budget}" if explicit else "the memory available now"
    condition = (
        f"The continuous-time fit is predicted to hold {_gib(estimate.bytes)} at once (its "
        f"{estimate.phase} phase) for {estimate.n:,} units, {estimate.causes} cause(s), "
        f"{estimate.rows} target row(s) and M = {estimate.times:,} grid times, and the budget is "
        f"{_gib(budget)}, {source}. Its arrays hold every unit at every grid time for both arms, "
        "so they grow as n x M; M counts every distinct observed time up to the last target "
        "time or restricted horizon, so an earlier last time shrinks it. Nothing was fitted. "
        "The prediction may exceed the true peak by about 10% when the walk takes a step, and "
        "by up to about 20% when the initial fit already meets the stopping rule; it does not "
        "count a user learner's own memory, and the available memory is read at one instant."
    )
    alternative = (
        "Pass max_memory_bytes= to ContinuousSurvivalTMLE to set the budget yourself, higher "
        "or lower."
    )
    error = ContinuousSurvivalMemoryError(condition, alternative=alternative)
    error.predicted = estimate.bytes
    error.budget = budget
    raise error


def largest_fold(folds: object, n: int) -> int:
    """The size of the largest group of equal labels in ``folds``.

    Parameters
    ----------
    folds : array-like or None
        One fold label per unit.
    n : int
        Units, returned when ``folds`` is ``None``.

    Returns
    -------
    int
        The most units one prediction fold covers.
    """
    import numpy as np

    if folds is None:
        return int(n)
    _, counts = np.unique(np.asarray(folds), return_counts=True)
    return int(counts.max()) if counts.size else int(n)


__all__ = [
    "ContinuousSurvivalMemoryError",
    "PeakEstimate",
    "check_memory",
    "largest_fold",
    "peak_bytes",
]
