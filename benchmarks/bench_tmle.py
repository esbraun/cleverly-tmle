"""Where does the time actually go?

Run with ``python benchmarks/bench_tmle.py``.  The point is to separate the two costs
that scale very differently:

* **nuisance estimation** -- fitting ``g`` and ``Qbar`` across folds.  This dominates, and
  it is already compiled code (scikit-learn, LightGBM), so there is nothing for a Rust
  extension to do here.
* **targeting and inference** -- the fluctuation solve, the influence curves, the
  multiplier bootstrap.  This is the part written in numpy, and therefore the part a
  native extension could speed up.

The output tells you whether that would be worth doing.  If targeting is 2% of runtime,
rewriting it in Rust buys 2% at best, and the honest conclusion is to leave it alone.
The kernels most likely to justify native code are the ones that scale in
``n_replicates x n``: the multiplier bootstrap and the targeted bootstrap.

What running this actually established (see the README):  every one of those kernels
turned out to be cheaper to *fix* than to rewrite.  The multiplier bootstrap spent
over 90% of its time generating multipliers rather than multiplying, and for Gaussian
multipliers the whole resampling loop has a closed form.  The cluster bootstrap was
rebuilding its membership index once per replicate.  Those rows are kept below so the
comparison stays reproducible rather than becoming folklore.

``--library`` defaults to ``default`` because that is what a real fit uses;  ``glm`` is
much faster but makes nuisance estimation look far cheaper than it is, which inflates
every other line's share.  Use ``--library glm`` for a quick pass, not for a verdict.

Usage::

    python benchmarks/bench_tmle.py                       # realistic, a few minutes
    python benchmarks/bench_tmle.py --library glm         # quick pass
    python benchmarks/bench_tmle.py --sizes 1000 10000 --library fast
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate
from cleverly.estimators.base import format_table
from cleverly.fluctuation import mean_submodel, solve_fluctuation, solve_one_step
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference import (
    bootstrap_indices,
    cluster_members,
    cluster_sums,
    make_estimate,
    simultaneous_bands,
)
from cleverly.utils.bounds import expit

DEFAULT_SIZES = (1_000, 5_000, 20_000)


@dataclass
class Timing:
    label: str
    n: int
    seconds: float
    note: str = ""

    def row(self, total: float | None = None) -> list[str]:
        share = "" if total is None else f"{100.0 * self.seconds / total:5.1f}%"
        return [self.label, f"{self.n:,}", f"{self.seconds:8.4f}", share, self.note]


def _time(func: Callable[[], Any], repeats: int = 3) -> float:
    """Best of ``repeats``: the least contaminated by scheduling noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def _synthetic(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    g1 = np.clip(expit(0.8 * w), 0.02, 0.98)
    a = rng.binomial(1, g1).astype(float)
    q = expit(0.4 + w)
    y = rng.binomial(1, q).astype(float)
    return {"a": a, "g1": g1, "y": y, "w": w}


def bench_targeting(sizes: tuple[int, ...]) -> list[Timing]:
    """The fluctuation solve on its own -- the innermost numerical kernel."""
    timings: list[Timing] = []
    for n in sizes:
        data = _synthetic(n)
        submodel = mean_submodel(data["a"], data["g1"])
        initial = InitialFit(np.full(n, 0.45), np.full(n, 0.45), np.full(n, 0.45))
        weights = np.ones(n)

        timings.append(
            Timing(
                "targeting (Newton)",
                n,
                _time(
                    lambda d=data, i=initial, sm=submodel, w=weights: solve_fluctuation(
                        d["y"], i, sm, w
                    )
                ),
            )
        )
        one_step = solve_fluctuation(data["y"], initial, submodel, weights)
        timings.append(
            Timing(
                "targeting (one-step)",
                n,
                _time(
                    lambda d=data, i=initial, sm=submodel, w=weights: solve_one_step(
                        d["y"], i, sm, w, step_size=5e-3, warn=False
                    ),
                    repeats=1,
                ),
                note=f"{one_step.n_iter} Newton iters",
            )
        )
    return timings


def bench_multiplier(sizes: tuple[int, ...], n_replicates: int = 1000) -> list[Timing]:
    """The multiplier bootstrap: the kernel that scales as replicates x n.

    Two rows per size.  ``rademacher`` resamples and therefore pays ``replicates x n``;
    ``normal`` samples the max-t law from its exact distribution and pays
    ``n m^2 + replicates m^2`` instead, so its cost barely moves with ``n``.
    """
    timings: list[Timing] = []
    for n in sizes:
        rng = np.random.default_rng(0)
        estimates = {
            name: make_estimate(name, 1.0, rng.normal(size=n), n=n)
            for name in ("ate", "att", "atc", "ey1", "ey0", "rr", "or")
        }
        for kind, note in (
            ("rademacher", "resampled"),
            ("normal", "exact, no (B, n) array"),
        ):
            timings.append(
                Timing(
                    f"multiplier bootstrap ({kind})",
                    n,
                    _time(
                        lambda e=estimates, k=kind: simultaneous_bands(
                            e, n_replicates=n_replicates, kind=k, random_state=0
                        )
                    ),
                    note=f"{n_replicates} draws x 7 estimands, {note}",
                )
            )
    return timings


def bench_clustered(sizes: tuple[int, ...], n_clusters: int = 500) -> list[Timing]:
    """Cluster-bootstrap resampling and cluster-summed influence curves.

    ``bootstrap_indices`` is called once per replicate, so its cost is multiplied by
    ``n_bootstrap``; the membership index it needs does not depend on the draw and is
    built once by ``run_bootstrap``.  The row below is the per-replicate cost with and
    without that prebuilt index.
    """
    timings: list[Timing] = []
    for n in sizes:
        rng = np.random.default_rng(0)
        codes = rng.integers(0, n_clusters, size=n)
        members = cluster_members(codes)
        ic = rng.normal(size=(n, 7))

        timings.append(
            Timing(
                "bootstrap_indices (rebuilt)",
                n,
                _time(lambda c=codes, size=n: bootstrap_indices(size, c, np.random.default_rng(1))),
                note=f"{n_clusters} clusters, per replicate",
            )
        )
        timings.append(
            Timing(
                "bootstrap_indices (prebuilt)",
                n,
                _time(
                    lambda c=codes, m=members, size=n: bootstrap_indices(
                        size, c, np.random.default_rng(1), m
                    )
                ),
                note="what run_bootstrap actually pays",
            )
        )
        timings.append(
            Timing(
                "cluster_sums",
                n,
                _time(lambda i=ic, c=codes: cluster_sums(i, c)),
                note="7 estimands",
            )
        )
    return timings


def bench_end_to_end(sizes: tuple[int, ...], library: str) -> list[Timing]:
    """A full fit, and the same fit with only the targeting step re-run."""
    timings: list[Timing] = []
    for n in sizes:
        frame, _ = make_nonlinear_ate(n=n, seed=0)
        estimator = TMLE(
            outcome_learner=library,
            treatment_learner=library,
            n_folds=5,
            learner_folds=3,
            estimands=("ate", "att", "atc", "ey1", "ey0"),
            simultaneous=False,
            random_state=0,
        )
        timings.append(
            Timing(
                f"full fit ({library})",
                n,
                _time(
                    lambda e=estimator, f=frame: e.fit(f, outcome="Y", treatment="A"),
                    repeats=1,
                ),
            )
        )

        result = estimator.fit(frame, outcome="Y", treatment="A")
        timings.append(
            Timing(
                "retarget (cached nuisances)",
                n,
                _time(
                    lambda e=estimator, r=result: e.retarget(
                        r.data,
                        r.nuisance,
                        estimands=("ate", "att", "atc", "ey1", "ey0"),
                    ),
                    repeats=3,
                ),
                note="what a sensitivity curve costs per point",
            )
        )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--library",
        default="default",
        help="nuisance learner library preset; 'glm' is quick but overstates every other share",
    )
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    sizes = tuple(args.sizes)

    sections = [
        ("Numerical kernels (numpy today)", bench_targeting(sizes)),
        ("Resampling kernels", bench_multiplier(sizes, args.replicates)),
        ("Clustered inference kernels", bench_clustered(sizes)),
        ("End to end", bench_end_to_end(sizes, args.library)),
    ]

    for title, timings in sections:
        print(f"\n{title}")
        print("=" * len(title))
        print(
            format_table(
                ["operation", "n", "seconds", "share", "note"],
                [timing.row() for timing in timings],
            )
        )

    print("\nReading the numbers")
    print("=" * 19)
    kernels = sum(t.seconds for t in sections[0][1] if t.label.startswith("targeting (Newton)"))
    full = sum(t.seconds for t in sections[-1][1] if t.label.startswith("full fit"))
    if full > 0:
        print(
            f"The targeting step is {100.0 * kernels / full:.2f}% of a full fit with "
            f"library={args.library!r}. Nuisance estimation dominates, and it already runs "
            "in compiled code -- so a native extension for the targeting step would buy "
            "almost nothing."
        )
    resampled = sum(t.seconds for t in sections[1][1] if t.label.endswith("(rademacher)"))
    exact = sum(t.seconds for t in sections[1][1] if t.label.endswith("(normal)"))
    if exact > 0:
        print(
            f"Gaussian multipliers are {resampled / exact:.0f}x cheaper than resampled ones "
            "here, because the max-t law has a closed form -- an algorithmic win no rewrite "
            "in any language competes with."
        )
    rebuilt = sum(t.seconds for t in sections[2][1] if t.label.endswith("(rebuilt)"))
    prebuilt = sum(t.seconds for t in sections[2][1] if t.label.endswith("(prebuilt)"))
    if prebuilt > 0:
        print(
            f"Prebuilding the cluster membership index is {rebuilt / prebuilt:.0f}x cheaper "
            "per replicate, which a cluster bootstrap pays n_bootstrap times over."
        )


if __name__ == "__main__":
    main()
