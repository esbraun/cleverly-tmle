r"""The memory model and allocation preflight of the continuous-time engine.

:class:`~cleverly.survival.ContinuousSurvivalTMLE` walks over blocks of ``c`` units
(:mod:`cleverly.survival._chunked`).  It holds every unit only in vectors and ``(n, P)`` arrays,
one block of ``(2, J, c, M)`` arrays at a time, and every unit at every grid time only for the
blocks it keeps between passes ("resident" blocks), as many as the budget allows.
:func:`peak_bytes` predicts the largest number of bytes the fit holds at once in each phase, from
the dimensions alone, and :func:`check_memory` refuses a fit only when its prediction with no
block kept exceeds the budget, before any nuisance is fitted.  :func:`resident_count` sizes the
kept blocks to the rest of the budget; it never refuses and never changes a result.

Each phase is a sum of the arrays the code holds there, shape times item size, so each term can
be checked against the line that allocates it.  Write ``n`` units, ``J`` causes, ``M`` grid
times, ``G`` the times of the event-support grid the walk runs on (at most ``M``), ``c`` the
units of a block, ``P = 2 J Q`` targets and ``E`` the trial steps a pass evaluates:

* ``nuisance``: the data, the fitted models' per-unit factors, and the largest Cox fit's
  ``(n_train, d, d)`` arrays (:func:`~cleverly.survival.cox._loglik_parts`) beside the treatment
  model's selection.
* ``initial``: one block's initial jumps on the full grid with their survival, then its
  censoring weights, as the pass that checks the initial fit builds them.
* ``walk``: the kept blocks, a rebuilt block or a block's trial step (its direction, candidate,
  curves and score temporaries), and the full ``(n, P)`` arrays of each trial.
* ``off_grid``: the final pass: the walk's state beside one ``(n, 2J x 100)`` array of the
  off-grid diagnostic, and under ``variance="switched"`` the predictable variation
  (``engine._Predictable``): every unit's ``(n, P)`` plug-ins and one
  buffer of targeted jumps on the full grid, ``(2, J, b, M)`` for the ``b`` units of
  :func:`~cleverly.survival._targeting.predictable_rows`, with that buffer's weights, curves
  and moment temporaries, which hold about ``_PV_BLOCK_CELLS`` cells.  It grows with ``n``
  only through the plug-ins.
* ``band``: the multiplier buffer of :func:`~cleverly.inference.simultaneous_bands`.

The prediction does not cover a user learner's own allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from ..exceptions import MethodConfigurationError
from ..inference.multiplier import _BLOCK_BYTES
from ..utils.memory import available_memory
from ._targeting import predictable_rows

#: Bytes of one float64.
F8: Final = 8
#: The band's multiplier buffer with its packed draw beside it.
BAND_BUFFER: Final = 2 * _BLOCK_BYTES
#: Allocations no array shape accounts for (small arrays, Python objects, the fitted models),
#: with a margin.  It keeps the prediction an upper bound at small ``n``.
FIXED: Final = 1 << 20
#: Random times of the off-grid diagnostic, ``engine.OFF_GRID_TIMES``; the engine imports this
#: module, so a test pins the two equal.
OFF_GRID: Final = 100
#: Bytes per training row and squared Cox design column of the Breslow fit's arrays
#: (``cox._loglik_parts``: the ``(n, d, d)`` outer products, their reversed cumulative sum and
#: its padded copy; the rows taken at the event times replace the sum).
COX_FIT: Final = 3 * F8
#: Bytes per training row and covariate of the default treatment model's selection (the
#: scaler's copies and the solver's).
TREATMENT_FIT: Final = 6 * F8


#: The resident set over the array bytes the model counts, which the kept blocks are sized to
#: leave room for.  The C library's allocator keeps freed block-sized arrays in its heap once
#: its mmap threshold has risen, so the resident set of a walk read 7 to 9% above its predicted
#: arrays at 8,000 to 32,000 units (and under them with the threshold held fixed).
ALLOCATOR_MARGIN: Final = 1.15


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
    influence_curve : int or None
        Under ``variance="switched"``, the predicted peak of the same fit with
        ``variance="influence_curve"``, which has no predictable-variation pass; ``None``
        otherwise.
    """

    phases: dict[str, int]
    n: int
    causes: int
    times: int
    rows: int
    influence_curve: int | None = None

    @property
    def bytes(self) -> int:
        """The predicted peak, the largest phase."""
        return max(self.phases.values())

    @property
    def phase(self) -> str:
        """The phase that reaches the peak."""
        return max(self.phases, key=self.phases.__getitem__)


def slot_bytes(units: int, causes: int, kept: int) -> int:
    """The bytes a kept block of ``units`` holds: its jumps, curves, weight and indicators.

    Parameters
    ----------
    units : int
        The block's units.
    causes : int
        Causes ``J``.
    kept : int
        The walk's grid size ``G``.

    Returns
    -------
    int
        ``(33 J + 33) units G``: jumps and incidences ``(2, J, c, G)``, survival and weight
        ``(2, c, G)``, and the boolean at-risk ``(c, G)`` and event ``(J, c, G)`` indicators.
    """
    return (33 * int(causes) + 33) * int(units) * int(kept)


def _rebuild(c: int, J: int, M: int, G: int) -> int:
    """A block built from the fitted models: its jumps, curves and weight, at their peak."""
    cM, cG = c * M, c * G
    return max(
        # The jumps on the full grid, then the grid's kept columns copied out of them.
        16 * J * cM + max(24 * cM, 16 * J * cG),
        # The curves of the jumps: survival, the survival just before each time, incidences.
        32 * J * cG + 32 * cG,
        # The censoring weight on the full grid beside the jumps and curves: the censoring
        # jumps, their product limit, the product with the propensity and its bounded inverse,
        # and the first build's count of bounded cells.
        32 * J * cG + 16 * cG + 48 * cM,
    )


def _direction(c: int, J: int, M: int, G: int, *, factual: bool, restricted: bool) -> int:
    """A block's direction being formed beside its state."""
    cG = c * G
    # The direction beside the summed quotient and one conditional_risk call's increments,
    # their reverse sums and the quotient; the factual tilt's masked copy beside the direction.
    # The quotient reads only the walk's grid, a restricted mean's weights too.
    peak = 16 * J * cG + 32 * cG
    if factual:
        peak = max(peak, 32 * J * cG)
    return peak


def _trial(c: int, J: int, M: int, G: int, *, factual: bool, restricted: bool) -> int:
    """One trial step on a block beside its state and direction, at its peak."""
    cM, cG = c * M, c * G
    # The score temporaries of evaluate_rows: one arm's weighted jumps, the summed ratio, the
    # ratio cumulated before each time, the survival before each time, two causes' increments
    # and one cause's cumulated sum beside the next, and on a reduced grid with a restricted
    # mean one cumulated sum read at every full-grid time.
    score = 8 * J * cG + 48 * cG + (16 * cM if restricted else 0)
    return max(
        _direction(c, J, M, G, factual=factual, restricted=restricted),
        # The direction beside the overflow bound's survival before each time, one cause's
        # increments and their reverse sums, and its quotients and factor at the runs of dropped
        # times (Walker.reach), at most one per walk-grid time.
        16 * J * cG + 48 * cG,
        # The direction beside the increment, the tilted candidate and its denominators, then
        # the candidate's curves and their score temporaries.
        16 * J * cG + max(32 * J * cG + 32 * cG, 32 * J * cG + 16 * cG + score),
        # The direction beside the exact overflow check of one run of units on the full grid:
        # its weights, filled curves, indicators, direction and the exp at the dropped times.
        16 * J * cG + ((65 + 60 * J) * overflow_rows(c, G, M) * M if G < M else 0),
    )


def overflow_rows(block: int, kept: int, times: int) -> int:
    """The units one run of the exact overflow check takes on the full grid.

    A quarter of a block's cells on the walk's grid, spread over the full grid's times, and at
    least one unit.

    Parameters
    ----------
    block : int
        The block's units.
    kept : int
        The walk's grid size ``G``.
    times : int
        The full grid size ``M``.

    Returns
    -------
    int
        Units per run.
    """
    return max(1, int(block) * int(kept) // (4 * int(times)))


def _replay(c: int, J: int, M: int, G: int, *, factual: bool, restricted: bool) -> int:
    """One accepted step applied to a block beside its old state."""
    cG = c * G
    return max(
        _direction(c, J, M, G, factual=factual, restricted=restricted),
        # The direction and the tilted jumps, then the new jumps and their curves.
        32 * J * cG + 32 * cG,
    )


def _variation(n: int, J: int, M: int, Q: int) -> tuple[int, int]:
    """The switched variance's predictable pass in the final pass: what it keeps and adds.

    Returns the bytes it keeps through the final pass and the most its buffer's moment adds
    beside them, both bounded by one buffer of ``predictable_rows`` units, not by ``n M``.
    """
    b = min(n, predictable_rows(J, Q, M))
    bM = b * M
    # Every unit's plug-ins (n, P) and the buffer of targeted jumps on the full grid (2, J, b, M).
    kept = F8 * n * 2 * J * Q + 16 * J * bM
    working = max(
        # The buffer's censoring weight: the censoring jumps, their product limit, the product
        # with the propensity and its bounded inverse, then the floored ratio.
        64 * bM,
        # The bounded weight and the floored ratio beside the curves being formed (survival,
        # the survival just before each time, its combined jump and the incidences).
        32 * bM + 48 * bM + 16 * J * bM,
        # The weight, the ratio and the survival beside one arm's moment: the clever
        # covariates (J, J Q, b, M) with the first moment, the scaled covariates and one
        # product (J Q, b, M each), and the arm's weights and their square roots.  Forming the
        # covariates, one cause's conditional risks and their reverse sums (3 Q) hold less.
        48 * bM + F8 * bM * (J * J * Q + 3 * J * Q + 2),
    )
    return kept, working


def _final(c: int, J: int, G: int, width: int) -> int:
    """The off-grid rows of one block beside its targeted state."""
    cG = c * G
    # The score temporaries, the block's parts and plug-ins, and the padded copy that carries
    # them into the sums.
    return 11 * J * cG + 48 * cG + 3 * F8 * c * width


def peak_bytes(
    n: int,
    causes: int,
    times: int,
    rows: int,
    *,
    covariates: int,
    train_size: int,
    design_width: int,
    simultaneous: bool,
    n_multiplier: int,
    factual_tilt: bool = False,
    stratified_learner: bool = False,
    restricted: bool = False,
    kept: int | None = None,
    resident_rows: int = 0,
    resident_blocks: int = 0,
    rungs: int | None = None,
    block: int | None = None,
    folds: int = 1,
    switched: bool,
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
    train_size : int
        The most units a nuisance model is fitted on: ``n`` without cross-fitting, otherwise
        ``n`` minus the smallest fold.
    design_width : int
        The widest design column count of a Cox model in either library, ``0`` with none.
    simultaneous : bool
        Whether the fit builds a simultaneous band.
    n_multiplier : int
        The band's multiplier draws.
    factual_tilt : bool
        Whether the walk tilts only the received arm (``tilt="factual"``).
    stratified_learner : bool
        Whether a library holds a stratified Nelson-Aalen learner, which keeps a cell index
        per unit.
    restricted : bool
        Whether a target has weights other than a unit vector (a restricted mean), whose sums
        run on the full grid.
    kept : int or None
        The walk's grid size ``G``; ``None`` takes ``M``, an upper bound known before fitting.
    resident_rows : int
        The units of the kept blocks.
    resident_blocks : int
        The kept blocks.
    rungs : int or None
        The trial steps a pass evaluates; ``None`` takes one with every block kept and
        :data:`~cleverly.survival._chunked.LADDER` otherwise.
    block : int or None
        Units per block; ``None`` takes :func:`~cleverly.survival._chunked.chunk_size`.
    folds : int
        The prediction folds: a stratified learner keeps each fold's cell tables, at most a
        block's units each.
    switched : bool
        Whether the fit reports the switched variance (``variance="switched"``), whose final
        pass adds the targeted jumps on the full grid to the predictable variation, one buffer
        of units at a time.

    Returns
    -------
    PeakEstimate
        The bytes of each phase.
    """
    from ._chunked import LADDER, chunk_size

    n, J, M, Q = int(n), int(causes), int(times), int(rows)
    G = M if kept is None else int(kept)
    c = chunk_size(n, J, M) if block is None else int(block)
    K = -(-n // c)
    R = int(resident_blocks)
    all_kept = R >= K
    E = (1 if all_kept else LADDER) if rungs is None else int(rungs)
    P = 2 * J * Q
    p = int(covariates)
    width = 2 * J * OFF_GRID
    # The data, the per-unit vectors and the record arrays, live through every phase: the
    # design and its inputs, the folds, the propensities, the retention readers, each model's
    # scales (and cell indices), the target weights and the metric.
    fixed = (
        FIXED
        + F8 * n * (p + 4) * 3
        + F8 * n * 8
        + 16 * n * Q
        + 16 * (J + 1) * n * (2 if stratified_learner else 1)
        + F8 * Q * M
        + F8 * P * P * 4
        # The off-grid diagnostic's weight rows, drawn before the walk.
        + F8 * OFF_GRID * G
        # A stratified learner's cell tables: both arms, every source and fold, a block's
        # units at most.
        + (16 * (J + 1) * int(folds) * c * M if stratified_learner else 0)
    )
    influence = F8 * n * P  # one (n, P) float array
    nuisance = (
        COX_FIT * int(train_size) * int(design_width) ** 2
        + TREATMENT_FIT * int(train_size) * (p + 1)
        + F8 * int(train_size) * (p + 1) * 4
    )
    cM = c * M
    initial = (
        max(
            # The block's jumps on the full grid, their combined jump and survival.
            16 * J * cM + 48 * cM,
            # The censoring jumps, their product limit and the product with the propensity.
            48 * cM,
        )
        + 2 * J * cM
    )
    resident = slot_bytes(int(resident_rows), J, G)
    slot = slot_bytes(c, J, G)
    rebuild = _rebuild(c, J, M, G)
    flags = {"factual": factual_tilt, "restricted": restricted}
    trial = slot + _trial(c, J, M, G, **flags)
    replay = slot + _replay(c, J, M, G, **flags)
    # The (n, P) arrays: the initial and current influence curves, each trial's parts and
    # plug-ins and its retry's, a summary being formed, and the low masks.
    held = influence * (4 + 4 * E) + 2 * n * (E + 1)

    def final_phase(rows: int, kept: int) -> int:
        """The final pass, whose largest working arrays beside a block's state are ``rows``."""
        if all_kept:
            # Every block's state is in ``resident``; the pass works on each in place.
            out = resident - slot + max(replay, rows)
        else:
            out = resident + max(rebuild, replay, rows)
        return out + 2 * influence + F8 * n * width + kept

    if all_kept:
        # Every block's state is in ``resident``: the first pass builds the last block beside
        # the others, and a later pass works on a kept block in place.
        last = slot_bytes(n - (K - 1) * c, J, G)
        walk = held + max(resident - last + rebuild, resident - slot + max(trial, replay))
    else:
        walk = held + resident + max(rebuild, replay, trial)
    final_rows = slot + _final(c, J, G, width)
    # Under "switched" the final pass also adds each buffer of targeted jumps on the full grid
    # to the predictable variation, beside the block's state.
    variation, adding = _variation(n, J, M, Q) if switched else (0, 0)
    band = (BAND_BUFFER + F8 * int(n_multiplier) * P * 3) if simultaneous else 0
    phases = {
        "nuisance": nuisance,
        "initial": initial,
        "walk": walk,
        "off_grid": final_phase(max(final_rows, slot + adding), variation),
        "band": 5 * influence + band,
    }
    influence_curve = None
    if switched:
        # The same fit without the predictable variation: only the final pass differs.
        plain = {**phases, "off_grid": final_phase(final_rows, 0)}
        influence_curve = max(plain.values()) + fixed
    return PeakEstimate(
        {name: value + fixed for name, value in phases.items()}, n, J, M, Q, influence_curve
    )


#: Each phase of :class:`PeakEstimate`, as a refusal names it.
PHASES: Final = {
    "nuisance": "fitting the nuisance models",
    "initial": "the pass that checks the initial hazard fits",
    "walk": "the targeting walk",
    "off_grid": "the final pass (the off-grid diagnostic)",
    "band": "the estimates and their simultaneous band",
}


def _described(phase: str, switched: bool) -> str:
    """The phase in words; under the switched variance the final pass names its variation."""
    if phase == "off_grid" and switched:
        return (
            "the final pass (the off-grid diagnostic and the switched variance's predictable "
            "variation)"
        )
    return PHASES[phase]


def _gib(value: int) -> str:
    return f"{value / 1024.0**3:.2f} GiB ({value:,} bytes)"


def check_memory(estimate: PeakEstimate, max_memory_bytes: int | None) -> None:
    """Refuse a fit whose predicted peak with no block kept exceeds its memory budget.

    Parameters
    ----------
    estimate : PeakEstimate
        The prediction with no block kept and the walk on the full grid.
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
    switched = estimate.influence_curve is not None
    condition = (
        f"The continuous-time fit is predicted to hold at least {_gib(estimate.bytes)} at once, "
        f"in {_described(estimate.phase, switched)}, for {estimate.n:,} units, "
        f"{estimate.causes} cause(s), {estimate.rows} target row(s) and "
        f"M = {estimate.times:,} grid times, and the budget is {_gib(budget)}, {source}. No "
        "budget lowers that with these settings: the prediction already walks over blocks of "
        "units and keeps no block between passes, so what remains is one block of units' "
        "arrays at a time and arrays with one row per unit. "
        "Nothing was fitted. The prediction does not count a user learner's own memory, and "
        "the available memory is read at one instant."
    )
    alternative = (
        "Pass max_memory_bytes= to ContinuousSurvivalTMLE to set the budget yourself, higher "
        "or lower."
    )
    if estimate.influence_curve is not None and estimate.influence_curve < estimate.bytes:
        alternative += (
            ' With variance="influence_curve" the fit needs less, a predicted '
            f"{_gib(estimate.influence_curve)}: it skips the predictable variation and reports "
            "the influence-curve variance alone, without raising it where the influence curve "
            "is sparse."
        )
    error = ContinuousSurvivalMemoryError(condition, alternative=alternative)
    error.predicted = estimate.bytes
    error.budget = budget
    raise error


def resident_count(shape: dict[str, Any], bounds: Any, max_memory_bytes: int | None) -> int:
    """The blocks a walk keeps between passes: as many as the rest of the budget holds.

    The budget is the explicit one, or under the default
    :data:`~cleverly.survival._chunked.RESIDENT_SHARE` of the memory available now; the kept
    blocks are as many as leave the walk's prediction times :data:`ALLOCATOR_MARGIN` within it.
    Every block is kept where the platform reports no memory.  The count changes only how often
    a pass rebuilds a block, never a result.

    Parameters
    ----------
    shape : dict
        The keyword arguments of :func:`peak_bytes` for the walk, without the kept blocks.
    bounds : sequence of tuple of int
        Each block's row range, in row order.
    max_memory_bytes : int or None
        The fit's budget.

    Returns
    -------
    int
        The kept blocks, the first ones in row order.
    """
    from ._chunked import RESIDENT_SHARE

    if max_memory_bytes is None:
        available = available_memory()
        if available is None:
            return len(bounds)
        budget = RESIDENT_SHARE * available
    else:
        budget = float(max_memory_bytes)
    budget /= ALLOCATOR_MARGIN
    estimate = peak_bytes(**shape)
    free = budget - max(estimate.phases["walk"], estimate.phases["off_grid"])
    J, G = int(shape["causes"]), int(shape["kept"])
    count = units = 0
    for lo, hi in bounds:
        units += hi - lo
        if slot_bytes(units, J, G) > free:
            break
        count += 1
    return count


def train_size(folds: object, n: int, cross_fit: bool) -> int:
    """The most units one nuisance model is fitted on.

    Parameters
    ----------
    folds : array-like
        One fold label per unit.
    n : int
        Units.
    cross_fit : bool
        Whether each model is fitted out of fold.

    Returns
    -------
    int
        ``n`` without cross-fitting, otherwise ``n`` minus the smallest fold.
    """
    import numpy as np

    if not cross_fit:
        return int(n)
    _, counts = np.unique(np.asarray(folds), return_counts=True)
    return int(n - counts.min()) if counts.size else int(n)


def design_width(library: Any, covariates: int) -> int:
    """The widest design of a Cox model in ``library``, ``0`` with none.

    Parameters
    ----------
    library : iterable
        Hazard learners.
    covariates : int
        Baseline covariates ``p``; the design adds the treatment first.

    Returns
    -------
    int
        Columns: one for the treatment alone, ``p + 1`` for main terms and ``2 p + 1`` with
        the treatment interactions.
    """
    from .cox import CoxHazard

    widths = {"treatment": 1, "main": covariates + 1, "interaction": 2 * covariates + 1}
    return max(
        (widths[member.terms] for member in library if isinstance(member, CoxHazard)), default=0
    )


__all__ = [
    "ContinuousSurvivalMemoryError",
    "PeakEstimate",
    "check_memory",
    "design_width",
    "peak_bytes",
    "resident_count",
    "slot_bytes",
    "train_size",
]
