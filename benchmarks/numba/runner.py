"""Driving the sweep: kernels x dimensions x implementations x core counts.

Two things the loop order encodes, and both are methodological rather than incidental.

**The fixture is built once per configuration and handed to every implementation.**  Not
rebuilt per implementation with the same seed -- *the same object*.  A seed guarantees the
same values; sharing the object guarantees the same *pages*, so an implementation is not
charged for the page faults of a fresh allocation that its predecessor was not.

**Implementations are timed in a shuffled order, and the shuffle is seeded.**  Timing all
of A then all of B attributes any drift in the machine to B.  The order is deterministic so
a run is reproducible, and it is different per configuration so a systematic effect cannot
line up with one implementation across the sweep.

A serial implementation is measured once and reused across the core counts.  It does not
vary with them, and re-measuring it would produce a flat "scaling curve" that reads as a
finding about parallelism rather than as the tautology it is.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

from .config import Config
from .fixtures import digest
from .implementations.numpy_reference import MODES, resolve_plan
from .kernels import KernelSpec, resolve
from .reporting import Row
from .resources import (
    ThreadPlan,
    applied,
    environment_record,
    logical_cores,
    numba_available,
)
from .timing import Measurement, measure, measure_amortised, shuffled
from .validation import check

__all__ = ["run"]

#: Which mode each implementation name is measured under.  A name not listed is treated as
#: a numpy variant: measured serially, and once, because it has no thread pool to vary.
_MODE_FOR = {
    "numba": "numba_serial",
    "numba_deferred_arms": "numba_serial",
    "numba_parallel": "numba_parallel",
    "numba_parallel_deferred_arms": "numba_parallel",
    "numba_parallel_by_cluster": "numba_parallel",
}


def _plan_for(implementation: str, cores: int) -> tuple[ThreadPlan, bool]:
    """The thread plan and whether this implementation varies with the core count."""
    mode = _MODE_FOR.get(implementation)
    if mode is None:
        return resolve_plan("numpy_serial", cores), False
    if mode == "numba_serial":
        return resolve_plan("numba_serial", cores), False
    return resolve_plan("numba_parallel", cores), True


def run(config: Config) -> tuple[list[Row], Any]:
    """Execute the sweep and return the rows plus the environment record."""
    environment = environment_record()
    available = logical_cores()
    rows: list[Row] = []

    for spec in resolve(config.kernels):
        for settings in _settings_for(spec, config):
            rows.extend(_run_one(spec, settings, config, environment, available))
    return rows, environment


def _settings_for(spec: KernelSpec, config: Config) -> Iterator[dict[str, Any]]:
    """The dimension settings to run this kernel at.

    ``sizes`` is a top-level sweep and applies to every kernel that takes an ``n``; a
    per-kernel ``sweeps`` entry overrides it, so a kernel whose interesting axis is the
    horizon count rather than the row count can say so without the config repeating
    ``sizes`` for everything else.
    """
    sweep = config.sweep_for(spec.name)
    settings = sweep.settings()
    if "n" in spec.dimensions and not any("n" in setting for setting in settings):
        settings = [{**setting, "n": n} for setting in settings for n in config.sizes]
    yield from settings


def _run_one(
    spec: KernelSpec,
    settings: dict[str, Any],
    config: Config,
    environment: Any,
    available: int,
) -> list[Row]:
    rows: list[Row] = []
    inputs = spec.inputs(**settings)
    dimensions = {**spec.dimensions, **settings}
    size = int(dimensions.get("n", 0))

    reference_output: Any = None
    reference_error = ""
    try:
        reference_output = spec.implementations["numpy"](inputs)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        reference_error = f"{type(error).__name__}: {error}"

    wanted = _selected(spec, config)
    # Serial implementations are measured once; the parallel ones once per core count.
    jobs: list[tuple[str, int]] = []
    for name in wanted:
        _, scales = _plan_for(name, 1)
        jobs.extend(
            (name, cores) for cores in (config.num_cores if scales else (1,))
        )

    for name, cores in shuffled(jobs, config.seed + size + len(spec.name)):
        implementation = spec.implementations[name]
        plan, _ = _plan_for(name, cores)
        if plan.numba_threads > available or plan.workers > available:
            rows.append(
                _skipped(spec, name, size, cores, plan, dimensions, environment,
                         f"asked for {cores} cores; the box has {available}")
            )
            continue
        if reference_error and name != "numpy":
            rows.append(
                _skipped(spec, name, size, cores, plan, dimensions, environment,
                         f"numpy reference failed: {reference_error}")
            )
            continue

        with applied(plan):
            cold = None
            if config.cold_compile and name.startswith("numba"):
                # The compilation has already happened by the time the sweep reaches
                # here -- the module-level dispatchers compiled on the reference pass --
                # so a "cold" number taken now would be a warm one. Cold compile is
                # measured in its own process by `cli --cold-compile`, and left None here
                # rather than filled with something that looks like a measurement.
                cold = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                measurement = measure(
                    lambda: implementation(inputs),
                    warmups=config.warmups,
                    repeats=config.repeats,
                    min_total_seconds=config.min_total_seconds,
                    cold=cold,
                )
                output = implementation(inputs)
                amortised = (
                    measure_amortised(lambda: implementation(inputs))
                    if config.amortise and spec.amortise
                    else {}
                )

        verdict = (
            check(reference_output, output, compare=spec.compare, tolerance=spec.tolerance)
            if config.validate and reference_output is not None
            else None
        )
        rows.append(
            _row(
                spec, name, size, cores, plan, dimensions, environment, measurement,
                verdict, amortised, inputs,
            )
        )
    return rows


def _selected(spec: KernelSpec, config: Config) -> list[str]:
    names = list(spec.implementations)
    if config.implementations and list(config.implementations) != ["all"]:
        wanted = set(config.implementations)
        names = [name for name in names if name in wanted or name == "numpy"]
    if not numba_available():
        names = [name for name in names if not name.startswith("numba")]
    return names


def _common(
    spec: KernelSpec,
    name: str,
    size: int,
    cores: int,
    plan: ThreadPlan,
    dimensions: dict[str, Any],
    environment: Any,
) -> dict[str, Any]:
    from .implementations.numba_parallel import effective_threads

    return {
        "scenario": spec.estimator,
        "operation": spec.name,
        "implementation": name,
        "n": size,
        "num_cores_requested": cores,
        "num_cores_effective": effective_threads() if plan.numba_threads > 1 else plan.requested_cores,
        "blas_threads": plan.blas_threads,
        "numba_threads": plan.numba_threads,
        "workers": plan.workers,
        "git_sha": environment.git_sha,
        "python_version": environment.python_version,
        "numpy_version": environment.numpy_version,
        "numba_version": environment.numba_version,
        "blas_backend": environment.blas_backend,
        "cpu_model": environment.cpu_model,
        "parallel_axis": spec.parallel_axis,
        "negative_control": spec.negative_control,
        "dimensions": dimensions,
        "oversubscribed": plan.oversubscribed(logical_cores()),
    }


def _skipped(spec, name, size, cores, plan, dimensions, environment, reason) -> Row:
    return Row(
        **_common(spec, name, size, cores, plan, dimensions, environment),
        repeat_count=0,
        warm_seconds=float("nan"),
        warm_iqr_seconds=float("nan"),
        warm_min_seconds=float("nan"),
        warm_max_seconds=float("nan"),
        cpu_seconds=float("nan"),
        peak_rss_bytes=0,
        rss_delta_bytes=0,
        correct=False,
        max_abs_error=float("nan"),
        max_rel_error=float("nan"),
        skipped_reason=reason,
    )


def _row(
    spec, name, size, cores, plan, dimensions, environment,
    measurement: Measurement, verdict, amortised, inputs,
) -> Row:
    return Row(
        **_common(spec, name, size, cores, plan, dimensions, environment),
        repeat_count=len(measurement.samples),
        warm_seconds=measurement.median,
        warm_iqr_seconds=measurement.iqr,
        warm_min_seconds=measurement.minimum,
        warm_max_seconds=measurement.maximum,
        cpu_seconds=measurement.cpu_seconds,
        peak_rss_bytes=measurement.peak_rss_bytes,
        rss_delta_bytes=measurement.rss_delta_bytes,
        cold_compile_seconds=measurement.cold_seconds,
        correct=bool(verdict) if verdict is not None else True,
        max_abs_error=verdict.max_abs_error if verdict else 0.0,
        max_rel_error=verdict.max_rel_error if verdict else 0.0,
        result_digest=_digest_inputs(inputs),
        amortised={str(k): v for k, v in amortised.items()},
    )


def _digest_inputs(inputs: Any) -> str:
    """A content hash of whatever arrays the kernel was handed, best effort."""
    import numpy as np

    arrays = []
    stack = [inputs]
    while stack and len(arrays) < 4:
        item = stack.pop()
        if isinstance(item, np.ndarray):
            arrays.append(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())
    return digest(*arrays) if arrays else ""
