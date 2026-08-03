"""``python -m benchmarks.numba.cli`` -- the entry point.

The one thing this module does that a normal CLI does not: it sets the thread-pool
environment variables *before* importing numpy, numba or anything that imports them.
``NUMBA_NUM_THREADS`` and the BLAS variables are read at library load time, so a run that
asks for four threads has to say so before the first import.  Everything else here is
argument parsing.

Examples
--------
::

    # the whole sweep at the sizes and core counts that fit a four-core box
    python -m benchmarks.numba.cli --kernel all --sizes 10000 100000 --num-cores 1 2 4

    # one kernel, every implementation, with the amortisation curve
    python -m benchmarks.numba.cli --kernel multiplier_bootstrap --amortise

    # cold-compile times, which need a fresh process per kernel
    python -m benchmarks.numba.cli --cold-compile

    # from a file
    python -m benchmarks.numba.cli --config benchmarks/configs/full.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks.numba.cli",
        description="Does numba or explicit parallelism help after the nuisances are fitted?",
    )
    parser.add_argument("--config", type=Path, help="YAML or JSON config file")
    parser.add_argument(
        "--kernel", "--scenario", dest="kernels", nargs="+", default=None,
        help="kernel name(s), or 'all'",
    )
    parser.add_argument(
        "--implementation", dest="implementations", nargs="+", default=None,
        help="implementation name(s), or 'all'",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--num-cores", type=int, nargs="+", default=None,
        help="core counts to measure the parallel implementations at",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", dest="validate", action="store_true", default=None)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--memory", dest="memory", action="store_true", default=None)
    parser.add_argument("--amortise", action="store_true", default=None)
    parser.add_argument(
        "--cold-compile", action="store_true", default=None,
        help="measure first-call compilation in a fresh process per kernel, then exit",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the registered kernels and exit"
    )
    # Estimator-specific dimensions, forwarded to whichever kernels take them.
    for name in (
        "n-estimands", "n-arms", "n-folds", "n-candidates", "n-timepoints", "n-regimens",
        "n-bootstrap", "n-clusters", "n-causes", "n-horizons",
    ):
        parser.add_argument(f"--{name}", type=int, nargs="+", default=None)
    parser.add_argument("--regime", nargs="+", default=None)
    parser.add_argument("--cluster-shape", nargs="+", default=None)
    parser.add_argument("--targeting-method", nargs="+", default=None)
    parser.add_argument("--bootstrap-chunk-size", type=int, nargs="+", default=None)
    return parser


#: Command-line dimension flags mapped onto the kernels and dimension names they set.
#: One flag can reach several kernels under different names -- ``--n-timepoints`` is
#: ``n_times`` everywhere it applies -- which is why this is a table rather than a
#: ``setattr`` loop.
_DIMENSION_FLAGS: dict[str, list[tuple[str, str]]] = {
    "n_estimands": [
        ("fused_influence_curves", "n_estimands"),
        ("cluster_sums", "n_estimands"),
        ("multiplier_bootstrap", "n_estimands"),
    ],
    "n_arms": [
        ("one_step_walk", "n_arms"),
        ("newton_targeting", "n_arms"),
        ("msm_gram", "n_arms"),
    ],
    "n_folds": [("ctmle_candidate_scores", "n_folds")],
    "n_candidates": [("ctmle_candidate_scores", "n_candidates")],
    "n_timepoints": [
        ("ltmle_backward_recursion", "n_times"),
        ("survival_incidence", "n_times"),
    ],
    "n_regimens": [
        ("ltmle_backward_recursion", "n_regimens"),
        ("survival_incidence", "n_regimens"),
    ],
    "n_bootstrap": [("multiplier_bootstrap", "n_replicates")],
    "n_clusters": [("cluster_sums", "n_clusters")],
    "n_causes": [("survival_incidence", "n_causes")],
    "n_horizons": [("survival_incidence", "n_horizons")],
    "regime": [
        ("one_step_walk", "regime"),
        ("ltmle_backward_recursion", "regime"),
        ("survival_incidence", "regime"),
        ("drtmle_reduction_rounds", "regime"),
        ("ctmle_candidate_scores", "regime"),
        ("fused_influence_curves", "regime"),
    ],
    "cluster_shape": [("cluster_sums", "shape")],
    "bootstrap_chunk_size": [("multiplier_bootstrap", "chunk")],
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    from .config import Config, load

    config = load(args.config) if args.config else Config()
    overrides: dict[str, object] = {}
    for name in (
        "kernels", "implementations", "sizes", "num_cores", "repeats", "warmups",
        "seed", "output", "validate", "memory", "amortise",
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        overrides[name] = tuple(value) if isinstance(value, list) else value

    sweeps = {sweep.kernel: dict(sweep.dimensions) for sweep in config.sweeps}
    for flag, targets in _DIMENSION_FLAGS.items():
        values = getattr(args, flag, None)
        if not values:
            continue
        for kernel, dimension in targets:
            sweeps.setdefault(kernel, {})[dimension] = list(values)

    from dataclasses import replace

    from .config import KernelSweep

    config = replace(
        config,
        **overrides,
        sweeps=tuple(
            KernelSweep(kernel=name, dimensions=dims) for name, dims in sweeps.items()
        ),
    )

    # Must happen before numpy/numba are imported anywhere, which is why the imports
    # below this line are function-local and the ones above are not.
    from .resources import bootstrap_environment

    bootstrap_environment(config.max_cores)

    from .kernels import REGISTRY, resolve

    if args.list:
        resolve(None)
        for name, spec in sorted(REGISTRY.items()):
            control = " [negative control]" if spec.negative_control else ""
            print(f"{name:28} {spec.estimator:10} {spec.parallel_axis or '-':12}{control}")
            print(f"{'':28} {spec.note}")
            print(f"{'':28} implementations: {', '.join(spec.implementations)}")
        return 0

    if args.cold_compile:
        from .cold import report_cold_compile

        return report_cold_compile(config)

    from .reporting import write_all
    from .runner import run

    rows, environment = run(config)
    latest = write_all(rows, environment, config.output)
    print(f"wrote {len(rows)} row(s) to {latest}")
    print((latest / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
