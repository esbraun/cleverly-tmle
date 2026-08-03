"""The mode table, and the thread plan each mode resolves to at a given core count.

There is no numpy code here.  Each kernel's numpy reference lives beside its numba
variants in :mod:`..kernels`, because the property a reader has to check is that the two
compute the same quantity and that check needs them on one screen.  What this module owns
is the part that is genuinely shared: which modes exist, and how many threads each gives
to which pool.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..resources import ThreadPlan

__all__ = ["MODES", "Mode", "resolve_plan"]


@dataclass(frozen=True)
class Mode:
    """One implementation-and-threading configuration."""

    name: str
    #: Which implementation of a kernel this mode calls.
    implementation: str
    #: What it is a control for, quoted into the summary so a negative result is legible.
    purpose: str
    #: Whether the mode varies with the core count at all.  A serial mode measured at
    #: four cores is the same measurement as at one, so the runner takes it once and says
    #: so rather than reporting a flat "speed-up" curve that looks like a finding.
    scales: bool

    def plan(self, cores: int, *, workers: int | None = None) -> ThreadPlan:
        return resolve_plan(self, cores, workers=workers)


MODES: tuple[Mode, ...] = (
    Mode(
        name="numpy_serial",
        implementation="numpy",
        purpose="the reference: one BLAS thread, one worker",
        scales=False,
    ),
    Mode(
        name="numpy_threaded",
        implementation="numpy",
        purpose="BLAS at p threads -- how much of a parallel gain is just cores",
        scales=True,
    ),
    Mode(
        name="numba_serial",
        implementation="numba",
        purpose="what compilation alone buys: fused passes, no temporaries",
        scales=False,
    ),
    Mode(
        name="numba_parallel",
        implementation="numba_parallel",
        purpose="prange over rows or replicates, BLAS pinned to one",
        scales=True,
    ),
    Mode(
        name="task_parallel",
        implementation="numpy",
        purpose="p single-threaded workers over independent jobs",
        scales=True,
    ),
    Mode(
        name="hybrid",
        implementation="numba_parallel",
        purpose="w workers x t threads -- measured because it is assumed to help",
        scales=True,
    ),
)

_BY_NAME = {mode.name: mode for mode in MODES}


def resolve_plan(mode: Mode | str, cores: int, *, workers: int | None = None) -> ThreadPlan:
    """The thread assignment ``mode`` asks for at ``cores`` cores.

    ``hybrid`` splits the cores between workers and threads; with ``workers=None`` it
    takes the squarest split, which is the configuration a reader would try first.  Any
    other split is reachable by passing ``workers`` explicitly, and the runner sweeps a
    few.
    """
    resolved = _BY_NAME[mode] if isinstance(mode, str) else mode
    cores = max(1, int(cores))
    if resolved.name == "numpy_serial":
        return ThreadPlan(numba_threads=1, blas_threads=1, workers=1)
    if resolved.name == "numpy_threaded":
        return ThreadPlan(numba_threads=1, blas_threads=cores, workers=1)
    if resolved.name == "numba_serial":
        return ThreadPlan(numba_threads=1, blas_threads=1, workers=1)
    if resolved.name == "numba_parallel":
        return ThreadPlan(numba_threads=cores, blas_threads=1, workers=1)
    if resolved.name == "task_parallel":
        return ThreadPlan(numba_threads=1, blas_threads=1, workers=cores)
    if resolved.name == "hybrid":
        count = workers if workers else max(1, round(cores**0.5))
        count = max(1, min(count, cores))
        return ThreadPlan(numba_threads=max(1, cores // count), blas_threads=1, workers=count)
    raise KeyError(f"unknown mode {resolved.name!r}")
