"""Whether numba and explicit parallelism buy anything *after* the nuisances are fitted.

`benchmarks/bench_tmle.py` answers a narrower question and answers it well: it times one
hand-fused ``njit`` Newton loop against the numpy one and reports that the ratio is a
wash.  What it cannot answer is whether that result generalises, because the Newton loop
is the *least* favourable case for a compiler that could still plausibly be called a
candidate -- its inner work is already ``x @ eps``, ``x.T @ (...)`` and a vectorised
``exp``, so a scalar loop has nothing left to remove.  Generalising from it would be
reasoning from a negative control.

This package is the wider instrument.  Its shape follows from three commitments.

**The nuisance fit is outside the timed region.**  The denominator is *cached nuisance
predictions -> targeting -> estimands -> influence curves -> inference*, which is the
half of a fit this package owns; a full-fit share is reported beside it as context and
never as the headline.  With the learners excluded, ``n = 1,000,000`` is a couple of
seconds rather than an afternoon, so the scaling questions are answered by direct
measurement rather than by extrapolation.

**The core count is a parameter, not an accident.**  Every measurement fixes numba's
thread count, BLAS's thread count and the worker count explicitly (:mod:`.resources`),
because the default is all three libraries claiming every core at once -- which measures
oversubscription rather than the kernel.  Speed-up and efficiency are reported per core
count, and a configuration that is *slower* than one core is flagged rather than averaged
away.

**A performance number is void until the outputs agree.**  Every kernel carries a
validator (:mod:`.validation`); an implementation whose output does not match the numpy
reference to the tolerance the kernel declares is reported as incorrect and its timings
are not summarised.

Layout
------
``config``
    the run description: sizes, core counts, repeats, per-scenario dimensions.
``resources``
    thread-pool control and the environment record.
``timing``
    cold-compile, warm and amortised measurement, with the statistics the plan asks for.
``fixtures``
    deterministic cached-nuisance inputs, generated outside every timed block.
``validation``
    numerical-equivalence gates and result digests.
``reporting``
    ``results.jsonl`` / ``results.csv`` / ``summary.md`` / ``environment.json``.
``kernels``
    one module per family.  Each holds a kernel's *specification* -- fixture, dimensions,
    validator -- **and all of its implementations, side by side**.  The plan's layout put
    the three implementations of every kernel in three separate modules; they are together
    here because the thing a reader has to check is that a numpy expression and a
    ``prange`` loop compute the same quantity, and that check is only possible when the
    two are on one screen.  What :mod:`.implementations` holds instead is the machinery
    the modes share: the ``njit`` decorators with the right flags, the availability probe,
    and the warm-up that separates compile time from run time.
``scenarios``
    complete post-nuisance pipelines per estimator flavour, so the answer is not only
    about isolated kernels.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
