"""The four ways a kernel can be run here, and what each one is a control for.

A "mode" is a pairing of an implementation with a thread plan, and the pairing matters as
much as the implementation: a numpy expression measured with four BLAS threads against a
``prange`` kernel measured with one is not a comparison of numpy against numba, it is a
comparison of four cores against one.  :mod:`..resources` enforces the plan; this module
names the modes and holds the machinery they share.

``numpy_serial``
    The reference.  NumPy/BLAS pinned to one thread, one worker.  Every correctness gate
    is stated against this, and every speed-up is a ratio to it.
``numpy_threaded``
    NumPy with BLAS at ``p`` threads.  Meaningful only where the kernel actually reaches
    threaded BLAS -- a matrix product with a large enough inner dimension.  Run anyway on
    the kernels that do not, because "numpy did not get faster with more cores here" is
    the fact that makes a numba-parallel speed-up attributable to numba rather than to the
    cores.
``numba_serial``
    ``njit`` with one thread.  Isolates what compilation buys: fused passes, no
    temporaries, no per-element dispatch.
``numba_parallel``
    ``njit(parallel=True)`` with ``p`` threads and BLAS at one, so the two do not contend.
``task_parallel``
    Independent jobs -- folds, candidates, regimens, bootstrap chunks -- over ``p``
    workers, each single-threaded.  The alternative to parallelising *inside* a kernel,
    and usually the better one when the jobs are large and numerous.
``hybrid``
    ``w`` workers times ``t`` threads.  Included because it is the configuration people
    assume helps; measured because it frequently does not.
"""

from __future__ import annotations

from .numba_parallel import PARALLEL_AVAILABLE, pjit
from .numba_serial import NUMBA_AVAILABLE, njit
from .numpy_reference import MODES, Mode

__all__ = [
    "MODES",
    "NUMBA_AVAILABLE",
    "PARALLEL_AVAILABLE",
    "Mode",
    "njit",
    "pjit",
]
