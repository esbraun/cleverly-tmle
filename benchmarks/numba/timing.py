"""Measurement: cold compile, warm runtime, amortisation, and the statistics of both.

Three commitments here, each of which a naive timing loop gets wrong.

**Cold and warm are separate numbers, not one number with a caveat.**  A numba kernel's
first call includes its compilation, which is seconds where the kernel is milliseconds.
Averaging that into the reported time makes a fast kernel look catastrophic; discarding it
silently makes a kernel that is called once per process look free.  Both are reported, and
:func:`break_even_calls` says how many calls it takes for the second to pay for the first.

**Best-of-three is not the estimate.**  It is the *minimum*, which is an estimate of the
machine's best behaviour rather than of the kernel's cost, and on a shared box it is the
statistic most easily moved by luck.  The median is reported as the estimate, with the
interquartile range beside it, and the minimum and maximum kept so a reader can see how
noisy the box was.

**Implementations are timed in randomised order.**  Timing A ten times then B ten times
attributes any drift in the machine -- a neighbouring container waking up, a thermal
change -- entirely to B.  Interleaving in a shuffled order spreads it over both.
"""

from __future__ import annotations

import gc
import random
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .resources import cpu_seconds, peak_rss_bytes

__all__ = [
    "Measurement",
    "break_even_calls",
    "measure",
    "measure_amortised",
    "peak_allocation",
    "shuffled",
    "speedup_interval",
]


@dataclass(frozen=True)
class Measurement:
    """One implementation's timing at one configuration."""

    #: Seconds for the first call in a fresh process-or-cache state: compilation included.
    #: ``None`` when the caller did not ask for a cold measurement (it costs a process).
    cold_seconds: float | None
    #: Every warm repetition, in the order run.
    samples: tuple[float, ...]
    cpu_seconds: float
    peak_rss_bytes: int
    rss_delta_bytes: int
    #: Largest allocation held at any moment during one *untimed* call, from
    #: :mod:`tracemalloc` -- which sees numpy's allocations and numba's alike; see
    #: :func:`peak_allocation`.  This is the memory number that means something here: peak RSS
    #: is a process high-water mark that never falls, so once the interpreter has touched
    #: a page it counts forever and every implementation after the first reads zero.  What
    #: a caller wants to know is what *this call* allocates -- the multiplier bootstrap's
    #: ``(chunk, n)`` array against a fused kernel's ``block x m`` accumulator -- and that
    #: is a per-call peak, not a process one.
    peak_alloc_bytes: int = 0

    @property
    def median(self) -> float:
        return float(statistics.median(self.samples))

    @property
    def iqr(self) -> float:
        if len(self.samples) < 4:
            return 0.0
        quartiles = statistics.quantiles(self.samples, n=4, method="inclusive")
        return float(quartiles[2] - quartiles[0])

    @property
    def minimum(self) -> float:
        return float(min(self.samples))

    @property
    def maximum(self) -> float:
        return float(max(self.samples))

    @property
    def compile_seconds(self) -> float | None:
        """Cold time net of the warm time: the compilation itself."""
        if self.cold_seconds is None:
            return None
        return max(0.0, self.cold_seconds - self.median)


def shuffled(items: Sequence[Any], seed: int) -> list[Any]:
    """``items`` in a deterministic shuffled order.

    Deterministic so a run is reproducible, shuffled so the order in which
    implementations are timed does not systematically favour one of them.
    """
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def measure(
    call: Callable[[], Any],
    *,
    warmups: int = 3,
    repeats: int = 10,
    min_total_seconds: float = 0.0,
    max_repeats: int = 10_000,
    cold: Callable[[], Any] | None = None,
    measure_memory: bool = True,
) -> Measurement:
    """Time ``call``, warm, with the statistics this package reports.

    Parameters
    ----------
    warmups:
        Calls made and discarded first.  For a numba kernel this is what moves the
        compilation out of the measured region; for a numpy one it settles the allocator
        and warms the caches the kernel will use.
    repeats:
        Measured calls, at least.
    min_total_seconds:
        Keep repeating past ``repeats`` until the measured calls total at least this
        much.  A kernel that runs in 40 microseconds needs hundreds of repetitions before
        the clock's own resolution stops dominating; one that runs in four seconds needs
        none.  This is what lets one setting serve both.
    measure_memory:
        Take a :mod:`tracemalloc` pass after the timed region.  On by default; it costs
        one extra call and roughly doubles that call's runtime, which is why it is one
        call and not part of the loop.
    cold:
        Called *once, first*, and timed separately.  Pass a callable that forces a fresh
        compilation (a freshly built kernel, cache disabled) to get a compile time; pass
        ``None`` to skip it.

    The garbage collector is disabled across the measured region and a full collection is
    forced before it, so a collection triggered by an earlier allocation is not charged to
    whichever repetition happens to trip it.
    """
    cold_seconds: float | None = None
    if cold is not None:
        gc.collect()
        start = time.perf_counter()
        cold()
        cold_seconds = time.perf_counter() - start

    for _ in range(warmups):
        call()

    gc.collect()
    rss_before = peak_rss_bytes()
    cpu_before = cpu_seconds()
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        total = 0.0
        while len(samples) < repeats or (total < min_total_seconds and len(samples) < max_repeats):
            start = time.perf_counter()
            call()
            elapsed = time.perf_counter() - start
            samples.append(elapsed)
            total += elapsed
    finally:
        if gc_was_enabled:
            gc.enable()
    cpu_used = cpu_seconds() - cpu_before
    rss_after = peak_rss_bytes()

    return Measurement(
        cold_seconds=cold_seconds,
        samples=tuple(samples),
        cpu_seconds=cpu_used / max(1, len(samples)),
        peak_rss_bytes=rss_after,
        rss_delta_bytes=max(0, rss_after - rss_before),
        peak_alloc_bytes=peak_allocation(call) if measure_memory else 0,
    )


def peak_allocation(call: Callable[[], Any]) -> int:
    """Largest Python-level allocation held at once during one call to ``call``.

    Taken **outside** the timed region and in its own call, because
    :mod:`tracemalloc` traces every allocation and roughly doubles the runtime of an
    allocation-heavy kernel.  Timing under it would measure the tracer.

    It sees numpy arrays -- numpy allocates through the CPython allocator hooks
    ``tracemalloc`` installs -- which is what makes it the right instrument for the
    question this package asks about memory.

    **It sees numba's allocations too**, which this docstring used to deny.  numba's NRT
    allocates through ``PyMem_RawMalloc``, and ``tracemalloc`` traces all three CPython
    allocator domains, so an ``np.empty`` inside an ``@njit`` function is traced exactly as
    the same call outside one is.  Measured on numba 0.66: 80 MB allocated inside a jitted
    function reads a peak of 80,000,224 bytes, against 80,000,240 for the identical numpy
    calls; and ``numba_cluster_sums_threadlocal`` at ``C = 50,000``, ``m = 20``, four
    threads reads 40.0 MB against the serial kernel's 9.99 MB -- the 32 MB thread-local
    block the old text called invisible is exactly what the difference is made of.

    What it genuinely does not see is a library that calls ``malloc`` directly rather than
    through CPython -- OpenBLAS scratch is the case that arises here -- and it is a
    high-water mark of *allocation* rather than of resident memory, so it does not answer
    "will this be killed".  For that, take an incremental peak RSS in a fresh process;
    ``benchmarks/results/production_plan.md`` §1.3 says why that is a confirmation at one
    configuration rather than a replacement for this column.
    """
    gc.collect()
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def measure_amortised(
    call: Callable[[], Any],
    counts: Sequence[int] = (1, 10, 100, 1000),
    *,
    cold: Callable[[], Any] | None = None,
) -> dict[int, float]:
    """Seconds per call when the kernel is called ``k`` times, for each ``k``.

    The repeated-call figure is the one that matters for the workloads a compiled kernel
    is most plausible for -- a truncation sweep, a simulation study, a bootstrap, a
    candidate path -- because there the compilation is paid once against hundreds of
    calls.  Reported as *per call at k* rather than as a total, so the series reads as a
    curve converging on the warm time.
    """
    out: dict[int, float] = {}
    first_cost = 0.0
    if cold is not None:
        start = time.perf_counter()
        cold()
        first_cost = time.perf_counter() - start
    for count in sorted(counts):
        start = time.perf_counter()
        for _ in range(count):
            call()
        elapsed = time.perf_counter() - start
        out[count] = (elapsed + first_cost) / count
    return out


def break_even_calls(compile_seconds: float, saving_per_call: float) -> float:
    """Calls before a compilation pays for itself, or ``inf`` when it never does."""
    if saving_per_call <= 0.0:
        return float("inf")
    return compile_seconds / saving_per_call


def speedup_interval(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and bootstrap interval for ``median(baseline) / median(candidate)``.

    A ratio of two noisy medians has no closed-form interval worth writing down, and the
    difference between "1.3x, interval 1.28-1.32" and "1.3x, interval 0.9-1.9" is the
    difference between a finding and a coin flip.  Resampling the two sample sets
    independently is the honest construction here: the repetitions are exchangeable
    within an implementation and the two implementations were timed on interleaved calls,
    so there is no pairing to preserve.
    """
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    point = float(np.median(base) / np.median(cand))
    if base.size < 3 or cand.size < 3:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        b = rng.choice(base, size=base.size, replace=True)
        c = rng.choice(cand, size=cand.size, replace=True)
        draws[i] = np.median(b) / np.median(c)
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(low), float(high)


@dataclass
class ScalingRow:
    """One core count's entry in a parallel-scaling table."""

    cores: int
    seconds: float
    peak_rss_bytes: int
    cpu_seconds: float
    baseline_seconds: float = field(default=float("nan"))

    @property
    def speedup(self) -> float:
        """``T_1 / T_p``."""
        if not self.seconds:
            return float("nan")
        return self.baseline_seconds / self.seconds

    @property
    def efficiency(self) -> float:
        """``S_p / p``: the share of the added cores that turned into speed."""
        return self.speedup / self.cores if self.cores else float("nan")

    @property
    def regression(self) -> bool:
        """More cores made it more than 5% slower -- reported, never averaged away."""
        return self.seconds > 1.05 * self.baseline_seconds
