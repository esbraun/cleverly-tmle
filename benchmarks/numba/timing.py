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

**Implementations are timed in a rotation, and this used to say something it did not do.**
Timing A ten times then B ten times attributes any drift in the machine -- a neighbouring
container waking up, a thermal change -- entirely to B.  Shuffling the order of the *whole
implementations* and then running each one's repetitions back to back does not fix that: it
randomises which arm gets which window, and every sample of an arm still comes from one
contiguous window, so drift slower than a block is confounded with arm rather than spread
over both.  That is randomised *block* order, it is what this module did while claiming to
interleave, and it is why a 1.02x is not resolved by it.

:func:`measure_interleaved` rotates instead: every arm is warmed, and then one step of
every arm is taken before a second step of any.  Samples are paired by round, which is what
lets :func:`speedup_interval` resample them together and what makes a small ratio mean
something.  Two things follow and neither is free -- the rotation's unit is a **plan
group** rather than an arm, since entering a thread plan costs more than a fast kernel's
call; and ``min_total_seconds`` becomes a **batch size** rather than a sample count, since
lockstep forbids one arm taking more samples than another.  Both are spelled out on
:func:`measure_interleaved`, and the second changes what a sample *is*: a per-call mean
over a batch, not one call's time.
"""

from __future__ import annotations

import contextlib
import gc
import math
import random
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .resources import cpu_seconds, peak_rss_bytes

__all__ = [
    "Arm",
    "Measurement",
    "break_even_calls",
    "measure",
    "measure_amortised",
    "measure_interleaved",
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
    #: Every warm repetition, in the order run.  Under :func:`measure_interleaved` these
    #: are paired by index across arms: sample ``r`` of each came from round ``r``.
    samples: tuple[float, ...]
    cpu_seconds: float
    peak_rss_bytes: int
    #: Process peak RSS gained across the *whole rotation*, not this arm's share.  It
    #: cannot be an arm's share: a high-water mark never falls, so once one arm has touched
    #: the pages every later one reads zero -- which is why the memory column has always
    #: been :attr:`peak_alloc_bytes` and why this stays a run-level number.
    rss_delta_bytes: int
    #: Largest allocation held at any moment during one *untimed* call, from
    #: :mod:`tracemalloc` -- which sees numpy's allocations and numba's alike; see
    #: :func:`peak_allocation`.  This is the memory number that means something here: peak RSS
    #: is a process high-water mark that never falls, so once the interpreter has touched
    #: a page it counts forever and every implementation after the first reads zero.  What
    #: a caller wants to know is what *this call* allocates -- the multiplier bootstrap's
    #: ``block x n`` buffer against a fused kernel's ``block x m`` accumulator -- and that
    #: is a per-call peak, not a process one.
    peak_alloc_bytes: int = 0
    #: Calls behind each entry of :attr:`samples`.  ``1`` is one call timed directly; above
    #: that, a sample is the per-call *mean* over a batch -- so the median is a median of
    #: batch means, :attr:`iqr` is narrowed by roughly ``sqrt(k)``, and :attr:`minimum` is
    #: the fastest batch per call rather than the fastest single call.  Recorded because a
    #: reader cannot otherwise tell ten samples of one call from ten samples of five
    #: hundred, and the two spreads are not comparable.
    calls_per_sample: int = 1

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


@contextlib.contextmanager
def _nothing() -> Iterator[None]:
    """The context an arm with no thread plan to apply runs under."""
    yield


@dataclass(frozen=True)
class Arm:
    """One thing to time, and the conditions it must be timed under.

    Attributes
    ----------
    key:
        How the caller identifies this arm in the returned mapping.
    call:
        The thing to time.  Called with no arguments and its result discarded.
    context:
        A zero-argument factory for the context manager the calls must run inside --
        ``lambda: applied(plan)``.  A factory rather than an object because the rotation
        enters it once per round.
    group:
        Arms with an equal ``group`` share one entry of that context per round.  This is
        what keeps the rotation affordable: entering a thread plan builds a
        ``ThreadpoolController`` at roughly 0.7 ms, which is an order of magnitude more
        than a fast kernel's call, so rotating *arm by arm* would spend the run switching
        plans.  A round is therefore group by group, and every arm still advances exactly
        one step per round.  The only cost is that arms sharing a plan are adjacent within
        a round.  Must be hashable; ``None`` puts every arm in one group.
    """

    key: Any
    call: Callable[[], Any]
    context: Callable[[], Any] = _nothing
    group: Any = None


def _grouped(arms: Sequence[Arm]) -> dict[Any, list[Arm]]:
    """``arms`` by group, both the groups and their members in first-seen order."""
    out: dict[Any, list[Arm]] = {}
    for arm in arms:
        out.setdefault(arm.group, []).append(arm)
    return out


def _batch_size(probe_seconds: float, repeats: int, minimum_total: float, cap: int) -> int:
    """Calls per sample, so that ``repeats`` samples cover ``minimum_total`` seconds.

    A probe that reads zero is *faster* than the clock, not free, so it takes the cap
    rather than a batch of one -- one would be the single case where the threshold is
    silently unmet, and it would happen to whichever arm most needs the batching.
    """
    if minimum_total <= 0.0:
        return 1
    if probe_seconds <= 0.0:  # pragma: no cover - perf_counter resolves a Python call
        return max(1, cap)
    wanted = math.ceil(minimum_total / (repeats * probe_seconds))
    return max(1, min(cap, wanted))


def measure_interleaved(
    arms: Sequence[Arm],
    *,
    warmups: int = 3,
    repeats: int = 10,
    min_total_seconds: float = 0.0,
    max_batch: int = 10_000,
    measure_memory: bool = True,
    seed: int = 0,
) -> dict[Any, Measurement]:
    """Time every arm in a rotation, and return one :class:`Measurement` each.

    One **round** takes a single step of every arm; ``repeats`` rounds are run.  Sample
    ``r`` of every arm therefore comes from round ``r``, under whatever the machine was
    doing then -- which is what "interleaved" has to mean if a common drift is to cancel
    out of a ratio rather than land on whichever arm held that window.

    Parameters
    ----------
    warmups:
        Calls made and discarded per arm, before any timing.  Done arm by arm rather than
        inside the rotation, so a numba kernel's compilation is out of the measured region
        entirely instead of landing in round zero.
    repeats:
        Rounds, which is exactly the sample count for every arm.
    min_total_seconds:
        Cover at least this much measured time per arm -- by **batching**, not by taking
        more samples.  A probe call sizes each arm's batch, so a 40-microsecond kernel and
        a four-second one both clear the threshold in the same number of rounds.  It has
        to work this way: lockstep is what pairs the samples, and an arm taking extra
        samples of its own is exactly what breaks it.  The cost is that a sample becomes a
        per-call mean over a batch -- see :attr:`Measurement.calls_per_sample`, which
        records the number so a spread is not read as though it were a single call's.
    measure_memory:
        Take a :mod:`tracemalloc` pass per arm after the rotation.  One extra call each,
        outside the timed region.
    seed:
        Fixes the order *within* each round, re-derived per round so no arm is permanently
        first.

    The collector is disabled once for the whole rotation and a full collection forced
    before it, rather than once per arm: under a rotation, whichever arm was running when a
    collection would have fired would otherwise be charged for it.
    """
    groups = _grouped(arms)

    probes: dict[Any, float] = {}
    for members in groups.values():
        with members[0].context():
            for arm in members:
                for _ in range(warmups):
                    arm.call()
                start = time.perf_counter()
                arm.call()
                probes[arm.key] = time.perf_counter() - start
    batches = {
        key: _batch_size(seconds, repeats, min_total_seconds, max_batch)
        for key, seconds in probes.items()
    }

    samples: dict[Any, list[float]] = {arm.key: [] for arm in arms}
    cpu: dict[Any, float] = {arm.key: 0.0 for arm in arms}

    gc.collect()
    rss_before = peak_rss_bytes()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_index in range(repeats):
            for group in shuffled(list(groups), seed + round_index):
                members = groups[group]
                with members[0].context():
                    for arm in shuffled(members, seed + round_index + len(groups)):
                        size = batches[arm.key]
                        # `cpu_seconds` is a `getrusage` call at ~1 microsecond, which is a
                        # quarter of the fastest kernel here -- so it is taken outside the
                        # window rather than inside it.
                        cpu_before = cpu_seconds()
                        start = time.perf_counter()
                        for _ in range(size):
                            arm.call()
                        elapsed = time.perf_counter() - start
                        cpu[arm.key] += cpu_seconds() - cpu_before
                        samples[arm.key].append(elapsed / size)
    finally:
        if gc_was_enabled:
            gc.enable()
    rss_after = peak_rss_bytes()

    allocations: dict[Any, int] = {}
    if measure_memory:
        for members in groups.values():
            with members[0].context():
                for arm in members:
                    allocations[arm.key] = peak_allocation(arm.call)

    return {
        arm.key: Measurement(
            cold_seconds=None,
            samples=tuple(samples[arm.key]),
            # Summed over every call in every batch, so the ~1 ms quantisation of
            # `ru_utime` -- which reads zero on a single 40-microsecond call -- cancels
            # rather than truncating each sample to nothing.
            cpu_seconds=cpu[arm.key] / max(1, repeats * batches[arm.key]),
            peak_rss_bytes=rss_after,
            rss_delta_bytes=max(0, rss_after - rss_before),
            peak_alloc_bytes=allocations.get(arm.key, 0),
            calls_per_sample=batches[arm.key],
        )
        for arm in arms
    }


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
    """Time one ``call``: :func:`measure_interleaved` with a single arm.

    Kept as the entry point for a caller with nothing to interleave *against*, and
    implemented through the rotation so there is one timing loop rather than two that
    drift apart.  ``max_repeats`` is the batch cap under the new spelling of
    ``min_total_seconds``; see :func:`measure_interleaved` for what that changed.

    ``cold`` is called once, first, and timed separately.  Pass a callable that forces a
    fresh compilation to get a compile time, or ``None`` to skip it.
    """
    cold_seconds: float | None = None
    if cold is not None:
        gc.collect()
        start = time.perf_counter()
        cold()
        cold_seconds = time.perf_counter() - start

    measurement = measure_interleaved(
        [Arm(key=None, call=call)],
        warmups=warmups,
        repeats=repeats,
        min_total_seconds=min_total_seconds,
        max_batch=max_repeats,
        measure_memory=measure_memory,
    )[None]
    return replace(measurement, cold_seconds=cold_seconds)


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
    ``docs/benchmarks/README.md`` says why that is a confirmation at one
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
    paired: bool = True,
) -> tuple[float, float, float]:
    """Point estimate and bootstrap interval for ``median(baseline) / median(candidate)``.

    A ratio of two noisy medians has no closed-form interval worth writing down, and the
    difference between "1.3x, interval 1.28-1.32" and "1.3x, interval 0.9-1.9" is the
    difference between a finding and a coin flip.

    Resampling is **paired** when the two sample sets are the same length, which under
    :func:`measure_interleaved` means sample ``r`` of each came from the same round: one
    index vector is drawn and both medians are taken over it, so a round in which the whole
    machine was slow moves both arms together and drops out of the ratio.  That correlation
    is the entire reason the rotation exists, and resampling the two sets independently
    would throw it away and report a wider interval than the design earns.  This function
    used to do exactly that, on the stated grounds that "the two implementations were timed
    on interleaved calls, so there is no pairing to preserve" -- which was true of the old
    block-order loop and is what the rotation retired.

    Unequal lengths fall back to independent resampling, which is the honest construction
    when there genuinely is no pairing.
    """
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    point = float(np.median(base) / np.median(cand))
    if base.size < 3 or cand.size < 3:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if paired and base.size == cand.size:
        index = rng.integers(0, base.size, size=(n_resamples, base.size))
        draws = np.median(base[index], axis=1) / np.median(cand[index], axis=1)
    else:
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
