"""The memory guard of ``LTMLE``, ``SDR`` and ``IPSI`` (X106 part J2d).

Before any learner, :meth:`~cleverly.longitudinal.estimator._SequentialEstimator._preflight`
predicts the traced peak of the whole call from counts known once the plans, the horizons and
the draws are resolved (:class:`FitCounts`), and refuses through
:func:`~cleverly.utils.memory.check_memory_budget` when the prediction exceeds
``max_memory_bytes``, or the memory available when that is ``None``.

One draw is predicted as ``c0 + 8 n * sum_k c_k x_k`` over the count terms of :data:`TERMS`,
with one constant set per family (:data:`CONSTANTS`), fitted to ``tracemalloc`` peaks over a
declared grid (``rows/X106/J2/evidence/`` in the plans repository).  The whole call is
``MARGIN * (repeats * draw + max(workers, bootstrap))``.  The prediction covers the arrays the
estimator holds with the calibration learners; a learner's own working memory beyond them is not
predicted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import MethodConfigurationError

__all__ = [
    "CONSTANTS",
    "MARGIN",
    "TERMS",
    "FitCounts",
    "LongitudinalMemoryError",
    "draw_bytes",
    "predicted_bytes",
]

#: The factor the prediction is raised by over the fitted constants (plan rev 9).
MARGIN = 1.25


class LongitudinalMemoryError(MethodConfigurationError):  # numpydoc ignore=PR01
    """A longitudinal fit predicted to need more memory than its budget.

    Raised before any learner.  ``predicted`` and ``budget`` are in bytes, ``None`` on a copy
    rebuilt from the refusal alone.
    """

    predicted: int | None = None
    budget: int | None = None


@dataclass(frozen=True)
class FitCounts:
    """The counts the memory guard reads, resolved before any learner.

    Parameters
    ----------
    estimator : str
        The class name the refusal names.
    family : str
        The key of :data:`CONSTANTS`.
    n : int
        Units.
    n_times : int
        Nodes.
    plans : int
        The plans the recursion fits (the slots on a path fit).
    horizons : tuple of int
        The reported horizons; ``(T,)`` at the end of the study.
    causes : int
        Causes, 1 without competing events.
    folds : int
        Outer folds.
    width : int
        Columns of the widest history design.
    bins : int
        Density bins at continuous nodes, 0 without.
    msm : bool
        Whether the fit projects on a regimen working model.
    repeats : int
        Draws, every one kept by the result.
    bootstrap : int
        Full-refit bootstrap replicates.
    jobs : int
        The worker count joblib resolves ``n_jobs`` to.
    panel_bytes : int
        Bytes of the prepared panel's arrays, which a parallel replicate copies.
    paths : int or None
        The slots of a path fit whose regimens draw from a known policy, ``None`` otherwise.
    """

    estimator: str
    family: str
    n: int
    n_times: int
    plans: int
    horizons: tuple[int, ...]
    causes: int
    folds: int
    width: int
    bins: int
    msm: bool
    repeats: int
    bootstrap: int
    jobs: int
    panel_bytes: int
    paths: int | None = None

    @property
    def steps(self) -> int:
        """The horizon-node pairs the recursion fits: the sum of the reported horizons."""
        return int(sum(self.horizons))


#: The count terms, each a function of :class:`FitCounts`, in units of ``8 n`` bytes.
TERMS: dict[str, Callable[[FitCounts], float]] = {
    "RSJ": lambda c: c.plans * c.steps * c.causes,
    "RT": lambda c: c.plans * c.n_times,
    "RTJ": lambda c: c.plans * c.n_times * c.causes,
    "T": lambda c: c.n_times,
    "TJ": lambda c: c.n_times * c.causes,
    "RHJ": lambda c: c.plans * len(c.horizons) * c.causes,
    "p": lambda c: c.width,
    "Rp": lambda c: c.plans * c.width,
    "Tp": lambda c: c.n_times * c.width,
    "one": lambda c: 1,
    "FRT": lambda c: c.folds * c.plans * c.n_times,
    "FT": lambda c: c.folds * c.n_times,
    "TB": lambda c: c.n_times * c.bins,
    "RTB": lambda c: c.plans * c.n_times * c.bins,
    "FTB": lambda c: c.folds * c.n_times * c.bins,
    "BB": lambda c: c.bins * c.bins,
    "TBB": lambda c: c.n_times * c.bins * c.bins,
    "RS_msm": lambda c: c.plans * c.steps if c.msm else 0,
    "TU": lambda c: c.n_times * (c.paths or 0),
    "TUJ": lambda c: c.n_times * (c.paths or 0) * c.causes,
    "Up": lambda c: (c.paths or 0) * c.width,
}

#: Bytes per unit per count, and ``c0`` in bytes, per family.  Filled from the calibration fit.
CONSTANTS: dict[str, dict[str, float]] = {}

#: The share of one draw a fold worker holds at ``n_jobs > 1``, per family.
WORKER_SHARE: dict[str, float] = {}


def draw_bytes(counts: FitCounts) -> float:
    """The predicted traced peak of one draw, before :data:`MARGIN`.

    Parameters
    ----------
    counts : FitCounts
        The fit's counts.

    Returns
    -------
    float
        Bytes.
    """
    constants = CONSTANTS.get(counts.family, {})
    total = constants.get("c0", 0.0) + 8.0 * counts.n * sum(
        constants.get(name, 0.0) * term(counts) for name, term in TERMS.items()
    )
    if counts.paths is not None:
        from .hazard import path_peak

        total = max(
            total,
            path_peak(counts.n, counts.n_times, counts.paths, counts.causes, counts.width),
        )
    return float(total)


def predicted_bytes(counts: FitCounts) -> int:
    """The predicted traced peak of the whole call, with :data:`MARGIN`.

    ``repeats`` draws are kept; at one worker a bootstrap replicate lives beside the kept
    result, and at ``j > 1`` workers the larger of the fold fan-out (``min(j, F)`` workers,
    each a share of a draw) and the parallel replicates (``min(j, B)``, each a draw and a copy
    of the panel) runs on top.

    Parameters
    ----------
    counts : FitCounts
        The fit's counts.

    Returns
    -------
    int
        Bytes.
    """
    draw = draw_bytes(counts)
    jobs = max(1, counts.jobs)
    workers = 0.0
    if jobs > 1 and counts.folds > 1:
        workers = min(jobs, counts.folds) * WORKER_SHARE.get(counts.family, 1.0) * draw
    bootstrap = 0.0
    if counts.bootstrap:
        bootstrap = (
            draw
            if jobs == 1
            else min(jobs, counts.bootstrap) * (draw + counts.panel_bytes)
        )
    return int(np.ceil(MARGIN * (counts.repeats * draw + max(workers, bootstrap))))


def panel_bytes(data: Any) -> int:
    """Bytes of the numpy arrays a prepared panel holds."""
    total = 0
    for value in vars(data).values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
        elif isinstance(value, tuple):
            total += sum(int(v.nbytes) for v in value if isinstance(v, np.ndarray))
    return total


def effective_jobs(n_jobs: int | None) -> int:
    """The worker count joblib runs ``n_jobs`` at (a negative value counts back from the CPUs)."""
    from joblib import effective_n_jobs

    return max(1, int(effective_n_jobs(1 if n_jobs is None else n_jobs)))


def fit_counts(
    estimator: Any,
    data: Any,
    *,
    plans: int,
    horizons: Sequence[int],
    folds: int,
    repeats: int,
    paths: int | None,
    multinode: bool,
    later: bool,
) -> FitCounts:
    """The counts of one guarded call (see :class:`FitCounts`).

    Parameters
    ----------
    estimator : _SequentialEstimator
        The estimator; its settings give the family, bins, bootstrap and workers.
    data : LongitudinalData
        The prepared panel.
    plans : int
        The plans the recursion fits.
    horizons : sequence of int
        The reported horizons.
    folds : int
        Outer folds of the first draw.
    repeats : int
        Draws.
    paths : int or None
        The slots of a policy path fit.
    multinode : bool
        Whether the design has a treatment at several nodes.
    later : bool
        Whether it has covariates after baseline.

    Returns
    -------
    FitCounts
        The counts.
    """
    from ..learners.density import default_density_bins

    n, n_times = int(data.n), int(data.n_times)
    continuous = any(
        data.is_continuous_node(t) or data.node_kind(t) == "mixed" for t in range(1, n_times + 1)
    )
    kind = estimator._estimator
    if kind == "tmle" and getattr(estimator, "survival_model", "iterated_mean") == "hazard":
        family = (
            "hazard_one_node"
            if not multinode
            else "hazard_backward"
            if later
            else "hazard_paths"
        )
    elif kind == "ipsi":
        family = "ipsi"
    else:
        family = ("ltmle" if kind == "tmle" else "sdr") + ("_continuous" if continuous else "")
    bins = 0
    if continuous:
        bins = int(estimator.density_bins or default_density_bins(n))
    width = int(data.history_design(n_times, include_current=True).shape[1])
    return FitCounts(
        estimator=type(estimator).__name__,
        family=family,
        n=n,
        n_times=n_times,
        plans=int(plans),
        horizons=tuple(int(h) for h in horizons),
        causes=len(data.cause_labels) if data.cause_labels else 1,
        folds=int(folds),
        width=width,
        bins=bins,
        msm=getattr(estimator, "msm", None) is not None,
        repeats=int(repeats),
        bootstrap=int(getattr(estimator, "n_bootstrap", 0) or 0),
        jobs=effective_jobs(getattr(estimator, "n_jobs", 1)),
        panel_bytes=panel_bytes(data),
        paths=paths,
    )


#: How far the margined prediction may exceed the true peak, which the refusal states.
FACTOR = 3.0


def check_fit(estimator: Any, data: Any, **counts: Any) -> int:
    """Refuse a guarded call whose predicted peak exceeds the budget, before any learner.

    Parameters
    ----------
    estimator : _SequentialEstimator
        The estimator; ``max_memory_bytes`` is its budget, ``None`` for the memory available.
    data : LongitudinalData
        The prepared panel.
    **counts : Any
        The keyword arguments of :func:`fit_counts` after ``data``.

    Returns
    -------
    int
        The predicted bytes.

    Raises
    ------
    LongitudinalMemoryError
        When the prediction exceeds the budget.
    """
    from ..utils.memory import check_memory_budget

    resolved = fit_counts(estimator, data, **counts)
    predicted = predicted_bytes(resolved)
    name = resolved.estimator
    horizons = len(resolved.horizons)
    what = (
        f"{name}'s fit ({resolved.n:,} units, {resolved.n_times} nodes, {resolved.plans} "
        f"{'paths' if resolved.paths else 'regimens'}, {horizons} "
        f"horizon{'s' if horizons != 1 else ''}, {resolved.causes} "
        f"cause{'s' if resolved.causes != 1 else ''}, {resolved.folds} folds, "
        f"{resolved.repeats} draw{'s' if resolved.repeats != 1 else ''}, "
        f"{resolved.bootstrap} bootstrap replicates, {resolved.jobs} "
        f"worker{'s' if resolved.jobs != 1 else ''})"
    )
    remedies = [f"Pass {name}(max_memory_bytes=...) (in bytes) to set the budget yourself"]
    if resolved.jobs > 1:
        remedies.append("fit with n_jobs=1, which runs one fold or replicate at a time")
    if resolved.bins:
        remedies.append("declare fewer density_bins=")
    remedies.append("or fit fewer regimens or horizons per call")
    check_memory_budget(
        predicted,
        max_memory_bytes=estimator.max_memory_bytes,
        what=what,
        factor=FACTOR,
        alternative=", ".join(remedies) + ".",
        error=LongitudinalMemoryError,
        held=False,
    )
    return predicted
