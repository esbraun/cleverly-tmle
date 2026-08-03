"""The correctness tier of the numba benchmark: small, one core, every kernel.

This is not a performance test and must never become one.  What it checks is the property
that makes a performance number mean anything -- that the compiled implementations compute
what the numpy reference computes -- and it checks it at sizes small enough to belong in
the fast tier.

Three things are pinned here that a timing run would not catch:

* **every registered kernel has a numpy reference**, because a speed-up with nothing to be
  a ratio to is not a measurement;
* **every implementation passes its own kernel's gate** at a small size and one core;
* **the parallel kernels are invariant to the thread count**, which is the property a
  reproducibility claim rests on and the one a single-threaded test suite would never
  exercise.  A bootstrap whose answer moves with the worker count is not a bootstrap.

The whole module skips without numba, which is the ordinary state of this repository's
environments -- ``numba`` lives in the ``bench`` extra and nothing under ``src/`` imports
it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

numba = pytest.importorskip("numba", reason="numba lives in the `bench` extra")

from benchmarks.numba.kernels import REGISTRY, resolve  # noqa: E402
from benchmarks.numba.validation import check  # noqa: E402

#: Dimensions small enough for the fast tier.  Compilation dominates at this size, which
#: is fine: what is under test is the answer, not the time.
_SMALL = {
    "n": 2_000,
    "n_replicates": 64,
    "n_candidates": 6,
    "n_times": 4,
    "n_regimens": 3,
    "n_clusters": 50,
    "n_rounds": 2,
    "max_steps": 25,
    "n_horizons": 2,
}


def _small(spec) -> dict:
    return {key: value for key, value in _SMALL.items() if key in spec.dimensions}


@pytest.fixture(scope="module")
def kernels():
    return resolve(None)


def test_every_kernel_has_a_numpy_reference(kernels):
    for spec in kernels:
        assert "numpy" in spec.implementations, spec.name


def test_the_registry_is_not_empty(kernels):
    # A kernel module that failed to import would leave the registry short and every
    # other test here passing vacuously.
    assert len(kernels) >= 9
    assert {spec.name for spec in kernels} == set(REGISTRY)


@pytest.mark.parametrize("kernel", sorted(REGISTRY) if REGISTRY else resolve(None))
def test_implementations_agree_with_the_numpy_reference(kernel):
    spec = REGISTRY[kernel] if isinstance(kernel, str) else kernel
    numba.set_num_threads(1)
    inputs = spec.inputs(**_small(spec))
    reference = spec.implementations["numpy"](inputs)
    for name, implementation in spec.implementations.items():
        if name == "numpy":
            continue
        verdict = check(
            reference,
            implementation(inputs),
            compare=spec.compare,
            tolerance=spec.tolerance,
        )
        assert verdict.correct, f"{spec.name}/{name}: {verdict.reason}"


@pytest.mark.parametrize(
    "kernel",
    [
        name
        for name, spec in (REGISTRY or {}).items()
        if any(key.startswith("numba_parallel") for key in spec.implementations)
    ]
    or ["multiplier_bootstrap"],
)
def test_a_parallel_kernel_does_not_depend_on_the_thread_count(kernel):
    """The reproducibility claim, stated as a test rather than as a docstring.

    Two threads and one thread must give the same answer.  For the bootstrap that is
    *bitwise* -- the draw is keyed on the replicate index rather than on a stream, so the
    scheduling cannot reach it.  For the reductions it is to the kernel's own tolerance,
    because summing in ``p`` blocks reassociates.
    """
    spec = REGISTRY[kernel]
    inputs = spec.inputs(**_small(spec))
    available = int(numba.config.NUMBA_NUM_THREADS)
    if available < 2:
        pytest.skip("needs at least two numba threads")

    for name, implementation in spec.implementations.items():
        if not name.startswith("numba_parallel"):
            continue
        numba.set_num_threads(1)
        one = implementation(inputs)
        numba.set_num_threads(2)
        two = implementation(inputs)
        numba.set_num_threads(1)
        verdict = check(one, two, compare=spec.compare, tolerance=spec.tolerance)
        assert verdict.correct, f"{spec.name}/{name} moved with the thread count: {verdict.reason}"


def test_the_bootstrap_draw_is_bitwise_reproducible_across_threads():
    """Sharper than the tolerance check above, and specific to the counter-based draw.

    The multiplier bootstrap's whole reason for using a counter hash instead of a stateful
    generator is that the replicate's draw is a function of its index alone.  So this is
    not "close enough": one thread and two threads must produce the same float.
    """
    from benchmarks.numba.kernels import bootstrap

    inputs = bootstrap.build(n=2_000, n_estimands=4, n_replicates=128)
    numba.set_num_threads(1)
    serial = bootstrap.numba_multiplier(inputs)
    one = bootstrap.numba_multiplier_parallel(inputs)
    if int(numba.config.NUMBA_NUM_THREADS) >= 2:
        numba.set_num_threads(2)
        two = bootstrap.numba_multiplier_parallel(inputs)
        numba.set_num_threads(1)
        assert one == two
    assert serial == one


def test_the_fused_multiplier_computes_the_statistic_it_claims_to():
    """An exact identity, which the cross-generator quantile gate cannot give.

    The benchmark's own correctness gate for this kernel is a Monte Carlo one -- the numpy
    path and the compiled path draw from different generators, so their critical values
    agree only as two estimates of one quantile do, and a gate at four standard errors
    would pass a kernel that dropped a column and got lucky.

    This is the sharp check.  The sign vector is reconstructed in numpy from the same
    splitmix counter, ``xi @ centred / n`` is formed the way the shipped path forms it, and
    the resulting max-t must equal the fused kernel's to rounding.  If the fused loop ever
    stops computing ``max_j |xi . centred_j / n| / se_j``, this fails deterministically.
    """
    from benchmarks.numba.kernels import bootstrap

    inputs = bootstrap.build(n=1_500, n_estimands=4, n_replicates=8)
    centred = inputs["centred"]
    se = np.asarray(inputs["std_errors"], dtype=float)
    usable = np.isfinite(se) & (se > 0)
    numba.set_num_threads(1)
    statistics = bootstrap._multiplier_serial(
        centred,
        np.where(usable, se, 1.0),
        usable,
        float(inputs["n"]),
        int(inputs["n_replicates"]),
        np.uint64(inputs["seed"]),
        bootstrap._BLOCK,
    )

    # Python ints with an explicit mask rather than np.uint64: numpy raises on the
    # wraparound these hashes are *defined* by, and the repository turns RuntimeWarning
    # into an error. The mask is the wraparound, written down.
    mask64 = (1 << 64) - 1

    def splitmix(state: int) -> int:
        z = (state + 0x9E3779B97F4A7C15) & mask64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask64
        return z ^ (z >> 31)

    rows = centred.shape[0]
    for replicate in range(int(inputs["n_replicates"])):
        base = (int(inputs["seed"]) * 0x2545F4914F6CDD1D + replicate) & mask64
        signs = np.empty(rows)
        for i in range(rows):
            word = splitmix((base + i // 64) & mask64)
            signs[i] = 1.0 if (word >> (i % 64)) & 1 else -1.0
        draw = (signs @ centred) / inputs["n"]
        expected = np.max(np.abs(draw[usable]) / se[usable])
        assert statistics[replicate] == pytest.approx(expected, rel=0, abs=1e-12)


def test_the_fused_curves_keep_the_influence_curve_identity():
    """``IC_ate == IC_ey1 - IC_ey0`` exactly, in the fused path as in the package.

    An exact identity rather than a tolerance, which is the instrument this repository
    prefers: the fused kernel computes the ATE curve as a difference of the two arm curves
    row by row, so if it ever stops being exactly that, this fails deterministically
    rather than drifting under a tolerance.
    """
    from benchmarks.numba.kernels import influence_curves

    inputs = influence_curves.build(n=2_000, n_estimands=7)
    for implementation in (
        influence_curves.numpy_estimands,
        influence_curves.numba_estimands,
        influence_curves.numba_estimands_parallel,
    ):
        out = implementation(inputs)
        assert np.array_equal(out["ic_ate"], out["ic_ey1"] - out["ic_ey0"])


def test_the_deferred_arm_update_matches_the_incremental_one_under_good_overlap():
    """The algorithmic arm's correctness claim, and the condition it holds under.

    ``logit`` is additive along the submodel, so applying the accumulated ``epsilon`` once
    is applying each step's increment in turn -- *until the shrink bound binds*, after
    which the incremental path clamps repeatedly and the deferred one clamps once.  Under
    good overlap it does not bind, and the two agree to rounding.  That conditionality is
    the finding, so it is pinned rather than left to a comment.
    """
    from benchmarks.numba.kernels import one_step

    inputs = one_step.build(n=2_000, regime="good", max_steps=40, step_size=1e-3)
    incremental = one_step.numpy_one_step(inputs)
    deferred = one_step.numpy_one_step_deferred(inputs)
    assert incremental["n_iter"] == deferred["n_iter"]
    assert np.allclose(incremental["targeted_arms"], deferred["targeted_arms"], rtol=0, atol=1e-12)


def test_a_kernel_refuses_a_dimension_it_does_not_take():
    """A typo in a config must fail rather than silently benchmark the defaults."""
    spec = REGISTRY["cluster_sums"]
    with pytest.raises(KeyError, match="does not take dimension"):
        spec.inputs(n_estimand=5)
