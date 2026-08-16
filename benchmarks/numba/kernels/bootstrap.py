r"""The multiplier bootstrap: the kernel whose grounds for compiling have moved.

``multiplier_critical_value`` draws ``B`` Rademacher vectors of length ``n``, forms
``xi @ centred / n``, standardises, takes a row-wise maximum, and quantiles the result.

**Read ``docs/benchmarks/bootstrap_numpy.md`` before citing anything here.**  An earlier
profile called this path "92-95% multiplier *generation*", which reads as an argument
about the random draw and is not one.  Split at ``n = 100,000`` with a 256-replicate
block, the seeded draw is **2%**, ``np.unpackbits`` is **1%**, expanding those bits into a
float64 sign matrix is **89%**, and the ``dgemm`` is **7%**.  The expensive step was the
expansion; the generator -- the thing a compiler was going to replace -- did not need to
change at all.

That report then removed two of the three grounds this module was built on:

.. code-block:: text

    # what this module used to transcribe, and what its original grounds assumed
    xi = signs((chunk, n))        # 8 bytes/entry: 200 MB at chunk=256, n=1e6
    draws = (xi @ centred) / n    # a (chunk, n) x (n, m) dgemm for m ~ 5

    # what ships, and what :func:`numpy_multiplier` now transcribes
    np.copyto(out, bits); out *= 2; out -= 1   # into a reused, byte-budgeted buffer

The large temporary is gone -- the buffer is sized by a **32 MB budget with a
four-replicate floor**, and the measured allocation at ``n = 10^6`` fell from 1,881 MB to
92 MB -- so the memory criterion no longer selects this kernel.  The runtime ground moved
with it: the numpy path is now 3.4-3.9x its old self, which is *faster than the compiled
kernel's serial arm was* (2.4-2.5x) on the same box, with no dependency and no 2.0 s
compile.

What is genuinely left for a compiler is narrower, and worth stating exactly: a fused loop
never forms the sign array at all, which is the 89% step; multiplying by a sign is an add
or a subtract rather than a multiply; and replicates remain an independent parallel axis.
That is the margin this module measures, and it is now almost entirely a *parallel* one --
see ``docs/benchmarks/bootstrap_numpy.md``, which records the rerun against the corrected
reference and its box.

``findings.md``'s 2.4-2.5x serial and 7.4-7.6x four-core figures were taken against the
old spelling and are **not** comparable to anything this module reports now.  Neither is
any result recorded before the reference was brought forward; the fixture identity did not
change, so the two cannot be told apart by their dimensions and must be told apart by their
commit.

**Reproducibility across thread counts.**  The parallel implementation gives each
replicate its own counter-based stream, seeded from ``(seed, replicate_index)``, so the
draws depend on the replicate's index and not on which thread ran it or how many threads
there were.  A parallel bootstrap whose answer moves with the worker count is not a
bootstrap, and the validator compares against the serial fused kernel *value for value*
rather than distributionally.

The numpy reference here is the package's own path, transcribed rather than imported, so
that the comparison is against the code that ships and not against a re-derivation of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import make_influence
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from . import KernelSpec, register

__all__ = ["build", "numba_multiplier", "numba_multiplier_parallel", "numpy_multiplier"]

#: ``None`` lets the reference derive its block the way the package does, from a byte
#: budget and ``n``.  A fixed number here would pin the one thing the package stopped
#: fixing; pass ``--bootstrap-chunk-size`` to sweep it deliberately instead.
_CHUNK = None

#: 2**64, for turning a 64-bit counter hash into the one bit a Rademacher draw needs.
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)


def build(
    n: int = 100_000,
    n_estimands: int = 5,
    n_replicates: int = 2000,
    chunk: int | None = None,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Centred influence curves and the standard errors the statistic divides by."""
    fixture = make_influence(n, n_arms=2, seed=seed)
    rng = np.random.default_rng(seed)
    curves = rng.standard_normal((n, n_estimands))
    # One estimand that is an exact linear combination of two others, as `ate` is of
    # `ey1` and `ey0`: the covariance is then singular, which is the case the package's
    # own closed form has to handle and a fixture of independent columns would miss.
    if n_estimands >= 3:
        curves[:, 2] = curves[:, 0] - curves[:, 1]
    # Scale each row by its own inverse-probability weight, so the curves have the heavy
    # tail a `1/g(W)` clever covariate gives them. A max-t law is a statement about the
    # tail, and a fixture of homoscedastic normals is the case where every method agrees.
    leverage = (fixture.treatment_indicator / fixture.propensity).sum(axis=1, keepdims=True)
    curves *= np.sqrt(leverage)
    centred = curves - curves.mean(axis=0, keepdims=True)
    std_errors = np.sqrt((centred**2).mean(axis=0) / n)
    return {
        "centred": np.ascontiguousarray(centred),
        "std_errors": std_errors,
        "n": n,
        "n_replicates": n_replicates,
        "chunk": chunk,
        "seed": seed,
        "alpha": 0.05,
    }


# --------------------------------------------------------------------------- numpy

#: The package's block budget and bounds, from `cleverly.inference.multiplier`.  The shipped
#: path derives its block from a byte target and `n` rather than fixing a replicate count,
#: because what the timing tracks is the buffer's footprint against the cache.
_BLOCK_BYTES = 32 << 20
_MIN_BLOCK = 4
_MAX_BLOCK = 256


def _package_block(n_rows: int, n_replicates: int) -> int:
    """Transcribed from :func:`cleverly.inference.multiplier._block_size`."""
    wanted = _BLOCK_BYTES // max(1, n_rows * 8)
    block = min(_MAX_BLOCK, max(_MIN_BLOCK, int(wanted)))
    block -= block % 4
    return max(_MIN_BLOCK, min(block, max(1, n_replicates)))


def _summarise(statistics: np.ndarray, alpha: float) -> dict[str, float]:
    """The critical value, plus the standard error a comparison of two of them needs.

    The numpy path draws from PCG64 and the compiled paths from a counter hash, so the two
    critical values are two *estimates of the same quantile from different samples*.  They
    cannot be compared to a fixed tolerance at any replicate count.

    The standard error is taken by resampling the statistics rather than by the textbook
    ``sqrt(p(1-p)/B) / f(q)``, because the density at a 95th percentile of a max-t law is
    exactly the quantity that formula needs and does not supply.  Using ``spread/sqrt(B)``
    instead -- the obvious shortcut -- understates it by a factor of several precisely at
    the upper tail, which is where this quantile lives; it was tried and read a correct
    kernel as five standard errors out.  Resampling costs microseconds on ``B`` floats and
    is outside every timed region.
    """
    quantile = 1.0 - alpha
    rng = np.random.default_rng(0)
    size = statistics.size
    draws = np.quantile(rng.choice(statistics, size=(200, size), replace=True), quantile, axis=1)
    return {
        "critical_value": float(np.quantile(statistics, quantile)),
        "critical_value_se": float(draws.std(ddof=1)),
        "n_replicates": float(size),
    }


def numpy_multiplier(inputs: dict[str, Any]) -> dict[str, float]:
    """Pack bits, widen them in place into a reused buffer, one dgemm per block.

    Transcribed from :func:`cleverly.inference.multiplier._two_point_statistics` and
    :func:`~cleverly.inference.multiplier._fill_multipliers` rather than called, so the
    timed region is the arithmetic alone -- the package function also validates shapes,
    resolves the kind and (with a cluster) sums within clusters first, none of which the
    compiled variants do either.  A comparison whose two sides do different amounts of
    bookkeeping is a comparison of the bookkeeping.

    **Transcribed from the current path, which is the whole point of transcribing it.**
    This once read ``_SIGNS[np.unpackbits(...)]``, the fancy-index expansion that
    ``docs/benchmarks/bootstrap_numpy.md`` measured at 89% of the kernel and then removed,
    against a fresh ``(chunk, n)`` array per block.  Timing the compiled kernels against
    that spelling charged them for work the package had stopped doing, which is what
    ``docs/benchmarks/README.md``'s first measurement rule forbids: compare against a
    competent numpy baseline, not merely the previous spelling of a function.

    ``inputs["chunk"]`` overrides the block for a sweep; ``None`` derives it as the package
    does, and that is the configuration whose ratio is the one to quote.
    """
    centred = inputs["centred"]
    se = inputs["std_errors"]
    n = inputs["n"]
    n_replicates = inputs["n_replicates"]
    rows = centred.shape[0]
    chunk = inputs["chunk"] or _package_block(rows, n_replicates)
    rng = np.random.default_rng(inputs["seed"])

    usable = np.isfinite(se) & (se > 0)
    statistics = np.empty(n_replicates, dtype=float)
    scale = se[usable]
    # Allocated once and written over per block, as the package does: the bounded
    # footprint is the capability change, not an incidental tidy-up.
    buffer = np.empty((chunk, rows), dtype=float)
    packed_columns = (rows + 7) // 8
    done = 0
    while done < n_replicates:
        size = min(chunk, n_replicates - done)
        xi = buffer[:size]
        packed = rng.integers(0, 256, size=(size, packed_columns), dtype=np.uint8)
        np.copyto(xi, np.unpackbits(packed, axis=1, count=rows))
        xi *= 2.0
        xi -= 1.0
        draws = (xi @ centred) / n
        standardised = np.abs(draws[:, usable]) / scale
        statistics[done : done + size] = standardised.max(axis=1)
        done += size
    return _summarise(statistics, inputs["alpha"])


# --------------------------------------------------------------------------- numba


@njit(inline="always")
def _bit(state: np.uint64) -> np.uint64:
    """One splitmix64 round: a counter to a well-mixed 64-bit word.

    A counter-based generator rather than a stateful one is what makes the parallel
    kernel's draw depend on the replicate index alone.  A shared ``np.random`` state would
    make the answer a function of the interleaving, and a per-thread state would make it a
    function of the thread count; both would give a bootstrap whose value moves when the
    machine does.
    """
    z = state + _SPLITMIX_GAMMA
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


#: Replicates accumulated together in one pass over ``centred``.  The blocking is not
#: cosmetic and is the difference between losing to numpy and beating it: a one-replicate-
#: at-a-time loop reads the whole ``(n, m)`` array once per replicate, which at ``B=500``
#: and ``n=100,000`` is 2 GB of traffic for 250 MFLOP -- entirely memory-bound, and no
#: faster than the dgemm it was meant to replace (measured here at 405 ms against numpy's
#: 393 ms).  Accumulating 64 replicates per pass reuses each loaded row 64 times and takes
#: it to 188 ms.  This is what a blocked dgemm does, done where the sign draw can be fused
#: into it.
_BLOCK = 64


@njit()
def _block_statistics(
    centred: np.ndarray,
    se: np.ndarray,
    usable: np.ndarray,
    n: float,
    seed: np.uint64,
    first: int,
    count: int,
    statistics: np.ndarray,
) -> None:
    """Accumulate ``count`` replicates starting at ``first``, in one pass over the rows.

    ``xi`` never exists.  Each row draws its bit per replicate and is added to or
    subtracted from that replicate's ``m``-vector, so the sign matrix is not formed at
    all and the multiply by plus-or-minus one becomes a sign flip.  Against the numpy
    path that is a narrower advantage than it once was: numpy no longer materialises a
    fresh ``(chunk, n)`` array either, it writes into one bounded buffer, so what is left
    here is the widen-and-rescale pass over that buffer rather than the whole allocation.
    """
    rows, columns = centred.shape
    accumulator = np.zeros((count, columns))
    words = np.empty(count, dtype=np.uint64)
    signs = np.empty(count)
    for i in range(rows):
        if i % 64 == 0:
            for k in range(count):
                base = np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D) + np.uint64(first + k)
                words[k] = _bit(base + np.uint64(i // 64))
        shift = np.uint64(i % 64)
        for k in range(count):
            signs[k] = 1.0 if ((words[k] >> shift) & np.uint64(1)) else -1.0
        for j in range(columns):
            value = centred[i, j]
            for k in range(count):
                accumulator[k, j] += signs[k] * value
    for k in range(count):
        best = 0.0
        for j in range(columns):
            if usable[j]:
                statistic = abs(accumulator[k, j] / n) / se[j]
                if statistic > best:
                    best = statistic
        statistics[first + k] = best


@njit()
def _multiplier_serial(
    centred: np.ndarray,
    se: np.ndarray,
    usable: np.ndarray,
    n: float,
    n_replicates: int,
    seed: np.uint64,
    block: int,
) -> np.ndarray:
    statistics = np.empty(n_replicates)
    for first in range(0, n_replicates, block):
        _block_statistics(
            centred,
            se,
            usable,
            n,
            seed,
            first,
            min(block, n_replicates - first),
            statistics,
        )
    return statistics


@pjit()
def _multiplier_parallel(
    centred: np.ndarray,
    se: np.ndarray,
    usable: np.ndarray,
    n: float,
    n_replicates: int,
    seed: np.uint64,
    block: int,
) -> np.ndarray:
    """Replicates are independent, so this is the clean axis: one ``prange``, no reduction.

    Each iteration owns a block of replicates and writes only its own slots of the output,
    so there is no shared accumulator and nothing to contend on.  Because the draw is
    keyed on the replicate index rather than on a stream, the result is identical to the
    serial kernel's whatever the block boundaries and the thread count are -- which the
    equivalence test checks value for value rather than distributionally.
    """
    statistics = np.empty(n_replicates)
    n_blocks = (n_replicates + block - 1) // block
    for index in prange(n_blocks):
        first = index * block
        _block_statistics(
            centred,
            se,
            usable,
            n,
            seed,
            first,
            min(block, n_replicates - first),
            statistics,
        )
    return statistics


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, float]:
    centred = inputs["centred"]
    se = np.asarray(inputs["std_errors"], dtype=float)
    usable = np.isfinite(se) & (se > 0)
    safe = np.where(usable, se, 1.0)
    statistics = kernel(
        centred,
        safe,
        usable,
        float(inputs["n"]),
        int(inputs["n_replicates"]),
        np.uint64(inputs["seed"]),
        _BLOCK,
    )
    return _summarise(statistics, inputs["alpha"])


def numba_multiplier(inputs: dict[str, Any]) -> dict[str, float]:
    return _run(inputs, _multiplier_serial)


def numba_multiplier_parallel(inputs: dict[str, Any]) -> dict[str, float]:
    return _run(inputs, _multiplier_parallel)


def _compare(reference: dict[str, float], candidate: dict[str, float]) -> tuple[float, float]:
    """Disagreement in units of the quantile's own Monte Carlo standard error.

    The two values are differenced and divided by the standard error of the difference,
    so the kernel's tolerance is a number of standard errors rather than a number of units
    -- the only form of this comparison that means the same thing at ``B = 64`` and at
    ``B = 10,000``.

    Two compiled implementations of the same draw are checked far more sharply than this,
    bitwise, by the equivalence test in ``tests/unit/test_numba_benchmark.py``.  This gate
    is only for the cross-generator comparison, where nothing sharper is available.
    """
    error = abs(reference["critical_value"] - candidate["critical_value"])
    standard_error = np.sqrt(
        reference["critical_value_se"] ** 2 + candidate["critical_value_se"] ** 2
    )
    normalised = error / max(standard_error, 1e-12)
    return float(normalised), float(normalised)


# ------------------------------------------------------------------------ registry

_IMPLEMENTATIONS: dict[str, Any] = {"numpy": numpy_multiplier}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_multiplier
    _IMPLEMENTATIONS["numba_parallel"] = numba_multiplier_parallel

register(
    KernelSpec(
        name="multiplier_bootstrap",
        estimator="inference",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=_compare,
        # In units of the quantile's Monte Carlo standard error, per `_compare`. Four is
        # a bar that a correct kernel clears at any replicate count and that a kernel
        # computing the wrong statistic fails -- a sign error, a missed standardisation or
        # a dropped column moves the max-t quantile by far more than four standard errors.
        tolerance=(4.0, 4.0),
        parallel_axis="replicates",
        note=(
            "expansion-bound: numpy widens packed bits into a bounded reused buffer; the "
            "fused kernel never forms the sign matrix at all"
        ),
        dimensions={
            "n": 100_000,
            "n_estimands": 5,
            "n_replicates": 2000,
            "chunk": _CHUNK,
            "seed": 20260803,
        },
        amortise=True,
    )
)
