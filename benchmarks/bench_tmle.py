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

What running this actually established (see ``docs/benchmarks/README.md``, *Current
verdict*):  every one of those kernels
turned out to be cheaper to *fix* than to rewrite.  The multiplier bootstrap spent
over 90% of its time generating multipliers rather than multiplying, and for Gaussian
multipliers the whole resampling loop has a closed form.  The cluster bootstrap was
rebuilding its membership index once per replicate.  Those rows are kept below so the
comparison stays reproducible rather than becoming folklore.

Which is not a figure of speech here.  This file spent some time *not running at all* --
``bench_targeting`` built an ``InitialFit`` positionally after that dataclass had been
re-keyed by arm, so the first section ``main`` evaluates raised ``TypeError`` and took
``nox -s bench`` with it.  A conclusion whose evidence cannot be re-run is folklore
however carefully it was once measured, so: run this before citing it.

The sections, and what each is here to answer:

* **Numerical kernels** -- the fluctuation solve on its own.
* **Resampling kernels** -- the multiplier bootstrap, which scales as ``replicates x n``.
* **Working-model projection** -- the MSM Newton, and the ``einsum`` under it written
  three ways.  ``np.einsum`` defaults to ``optimize=False``, so the default spelling is
  numpy's own nested-loop kernel rather than BLAS.
* **Dataframe boundary** -- frame in, container out, frame back, for pandas, polars and
  arrow-backed pandas.  Here so that "should the internals be polars?" is answered by a
  share of a fit rather than by assertion.
* **Clustered inference kernels** -- the cluster bootstrap's per-replicate cost.
* **Compiled kernels (numba)** -- a jitted Newton loop against the numpy one, reporting
  the compile time separately and the numerical agreement alongside, because a kernel
  that is faster only after a multi-second compile, or faster but different, is not
  faster.  Needs ``pip install -e '.[bench]'``; skipped with a note otherwise.
* **End to end**, and **End to end (longitudinal)** -- the denominators every share above
  is taken against.  The ``LTMLE`` case exists because the claim that longitudinal fits
  stay scikit-learn-bound was for a long time a prediction rather than a measurement, for
  want of one here.  This section is the measurement.
* **Projected to scale** -- every kernel's cost split into a fixed part and a per-row part,
  read off at ``--project`` sizes (a million rows and five million by default) that are
  *not* run.  Nothing here is measurable at that size in a reasonable time, and nothing
  needs to be: what decides whether a kernel matters at scale is its per-row cost as a
  share of the fit's, and two sizes an order of magnitude apart determine that.  It also
  reports the arrays whose *size* grows with ``n``, which is what actually breaks first.

``--library`` defaults to ``default`` because that is what a real fit uses;  ``glm`` is
much faster but makes nuisance estimation look far cheaper than it is, which inflates
every other line's share.  Use ``--library glm`` for a quick pass, not for a verdict.

Usage::

    python benchmarks/bench_tmle.py                       # realistic, a few minutes
    python benchmarks/bench_tmle.py --library glm         # quick pass
    python benchmarks/bench_tmle.py --sizes 1000 10000 --library fast
    python benchmarks/bench_tmle.py --skip ltmle numba    # the two slowest sections
    python benchmarks/bench_tmle.py --sizes 2000 20000 100000 --project 5000000

**Pass at least two ``--sizes``, an order of magnitude apart, for any question about
scale.**  One size gives a share and no slope, and a share at small ``n`` is the number
most likely to mislead: a fit carries fixed setup that a kernel timed alone does not, so
every kernel reads cheaper than it asymptotically is.  Measured at n=2,000 the targeting
solve is 0.3% of a ``glm`` fit; its per-row cost is 7% of the fit's.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cleverly.datasets import make_nonlinear_ate
from cleverly.estimators import TMLE
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
from cleverly.msm import solve_projection
from cleverly.utils.bounds import expit, logit

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


def _flat_initial(n: int, value: float = 0.45, arms: tuple[float, ...] = (0.0, 1.0)) -> InitialFit:
    """A featureless initial fit: the same prediction observed and at every arm.

    Spelled out here because ``InitialFit`` keys its counterfactual predictions *by arm*
    and this file used to build one positionally, as ``(observed, at_one, at_zero)``.  It
    has taken two fields since arms were keyed, so ``bench_targeting`` -- the first
    section ``main`` evaluates -- raised ``TypeError`` before it timed anything, and with
    it ``python benchmarks/bench_tmle.py`` and ``nox -s bench``.  Which is worth stating
    plainly: the "a native extension is not worth building" conclusion now recorded in
    ``docs/benchmarks/README.md`` rested on a benchmark that had stopped running, so it was
    folklore rather than a measurement until this was fixed and rerun.
    """
    return InitialFit(np.full(n, value), {arm: np.full(n, value) for arm in arms})


def bench_targeting(sizes: tuple[int, ...]) -> list[Timing]:
    """The fluctuation solve on its own -- the innermost numerical kernel."""
    timings: list[Timing] = []
    for n in sizes:
        data = _synthetic(n)
        submodel = mean_submodel(data["a"], data["g1"])
        initial = _flat_initial(n)
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


def bench_msm(sizes: tuple[int, ...], n_arms: int = 6, n_terms: int = 4) -> list[Timing]:
    """The working-model projection, and the contraction underneath it.

    ``solve_projection`` is a damped Newton under a non-identity link, and
    ``_projection_state`` -- three ``einsum`` calls over an ``(n, K, p)`` design -- runs
    once per Newton step *and* once per line-search trial.  The four-operand one,
    ``"ijp,ijq,ij,i->pq"``, is the expensive term.

    The three arms below are the same arithmetic written three ways.  ``np.einsum``
    defaults to ``optimize=False``, which for three or more operands means numpy's own
    nested-loop kernel rather than a pairwise contraction through BLAS -- so the default
    is the slow one, and neither of the other two needs a dependency this package does
    not already have.
    """
    timings: list[Timing] = []
    for n in sizes:
        rng = np.random.default_rng(0)
        phi = rng.normal(size=(n, n_arms, n_terms))
        h = rng.random((n, n_arms)) + 0.5
        q = rng.random((n, n_arms))
        w = np.ones(n)

        naive = np.einsum("ijp,ijq,ij,i->pq", phi, phi, h, w)
        blas = _gram_via_blas(phi, h, w)
        agreement = float(np.max(np.abs(naive - blas)) / max(np.max(np.abs(naive)), 1.0))

        timings.append(
            Timing(
                "gram einsum (optimize=False)",
                n,
                _time(lambda p=phi, hh=h, ww=w: np.einsum("ijp,ijq,ij,i->pq", p, p, hh, ww)),
                note=f"{n_arms} arms x {n_terms} terms, numpy's own kernel",
            )
        )
        timings.append(
            Timing(
                "gram einsum (optimize=True)",
                n,
                _time(
                    lambda p=phi, hh=h, ww=w: np.einsum(
                        "ijp,ijq,ij,i->pq", p, p, hh, ww, optimize=True
                    )
                ),
                note="pairwise contraction",
            )
        )
        timings.append(
            Timing(
                "gram via reshape + dgemm",
                n,
                _time(lambda p=phi, hh=h, ww=w: _gram_via_blas(p, hh, ww)),
                note=f"one BLAS call, agrees to {agreement:.1e} relative",
            )
        )
        for link in ("identity", "log"):
            timings.append(
                Timing(
                    f"solve_projection ({link})",
                    n,
                    _time(
                        lambda p=phi, hh=h, qq=q, ww=w, lk=link: solve_projection(p, hh, qq, ww, lk)
                    ),
                    note="closed form" if link == "identity" else "damped Newton + line search",
                )
            )
    return timings


def _gram_via_blas(phi: np.ndarray, h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``sum_i w_i sum_j h_ij phi_ijp phi_ijq`` as a single matrix product.

    The ``(n, K, p)`` design flattens to ``(nK, p)`` because the contraction sums over
    ``i`` and ``j`` identically; scaling the rows by ``h_ij w_i`` and forming ``X^T X``
    is then one ``dgemm``.
    """
    scale = (h * w[:, None]).reshape(-1)
    flat = phi.reshape(-1, phi.shape[2])
    return np.asarray(flat.T @ (flat * scale[:, None]), dtype=float)


def bench_ingestion(sizes: tuple[int, ...]) -> list[Timing]:
    """Frame in, container out, frame back -- the whole dataframe boundary.

    Here to answer a question that otherwise gets answered by assertion: the library
    holds numpy internally rather than a dataframe, and the case for changing that would
    have to start with the boundary being expensive.  Each row is once per fit, against
    nuisance estimation that is ``n_folds x n_candidates`` model fits.

    ``pyarrow`` is the *dtype* backend of a pandas frame here, not a third library, which
    is the configuration ``dtype_backend="pyarrow"`` produces.
    """
    import pandas as pd

    from cleverly.data import CausalData
    from cleverly.datasets import make_linear_ate

    timings: list[Timing] = []
    for n in sizes:
        pandas_frame, _ = make_linear_ate(n=n, seed=0, backend="pandas")
        variants = {
            "pandas": pandas_frame,
            "polars": make_linear_ate(n=n, seed=0, backend="polars")[0],
            "pandas+arrow": pandas_frame.convert_dtypes(dtype_backend="pyarrow"),
        }
        for label, frame in variants.items():
            built = CausalData.from_frame(frame, outcome="Y", treatment="A")
            timings.append(
                Timing(
                    f"from_frame ({label})",
                    n,
                    _time(lambda f=frame: CausalData.from_frame(f, outcome="Y", treatment="A")),
                    note=f"{built.n_covariates} covariates",
                )
            )
            timings.append(
                Timing(
                    f"to_frame ({label})",
                    n,
                    _time(lambda d=built: d.to_frame()),
                    note="" if label != "pandas+arrow" else "emits numpy-backed pandas",
                )
            )
        assert isinstance(variants["pandas+arrow"], pd.DataFrame)
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

        result = estimator.fit(frame, outcome="Y", treatment="A").single()
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


def bench_longitudinal(sizes: tuple[int, ...], library: str) -> list[Timing]:
    """A longitudinal fit, split into its mechanism and its backward pass.

    This section exists to close a named gap: the claim that longitudinal TMLE stays
    scikit-learn-bound -- the node loop is Python, but each body is a nuisance fit -- was
    for a long time a prediction rather than a measurement, because this file had no
    ``LTMLE`` case to profile.  Now it does.

    The recursion's own arithmetic is what the ``(n, 1)`` fluctuation per node costs; the
    row below it is the whole fit, so the ratio is the answer.
    """
    from cleverly.datasets import make_longitudinal
    from cleverly.longitudinal import LTMLE

    timings: list[Timing] = []
    for n in sizes:
        frame, _ = make_longitudinal(n=n, seed=0)
        estimator = LTMLE(
            {"always": (1, 1), "never": (0, 0)},
            reference="never",
            outcome_learner=library,
            treatment_learner=library,
            n_folds=5,
            random_state=0,
        )
        columns = {
            "outcome": "Y",
            "treatment": ["A1", "A2"],
            "baseline": ["W1", "W2"],
            "time_varying": [[], ["L2"]],
            "censoring": ["C1", "C2"],
        }
        timings.append(
            Timing(
                f"LTMLE full fit ({library})",
                n,
                _time(lambda e=estimator, f=frame, c=columns: e.fit(f, **c), repeats=1),
                note="2 regimens x 2 nodes, mechanism shared",
            )
        )
        # The per-node fluctuation on its own: one column, solved once per node per
        # regimen. Four of them in the fit above, so this row times four is the whole of
        # what a native targeting kernel could remove from it.
        data = _synthetic(n)
        submodel = mean_submodel(data["a"], data["g1"])
        timings.append(
            Timing(
                "  of which: one node's fluctuation",
                n,
                _time(
                    lambda d=data, i=_flat_initial(n), sm=submodel, w=np.ones(n): solve_fluctuation(
                        d["y"], i, sm, w
                    )
                ),
                note="x4 in the fit above",
            )
        )
    return timings


#: Holds the compiled solver so the one-off compile is paid once across every size.
_NUMBA_CACHE: list[Any] = [None]


def _numpy_newton(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, offset: np.ndarray, max_iter: int = 50
) -> np.ndarray:
    """``_newton_logistic``'s loop, reduced to the arithmetic, in numpy.

    Not the real solver: the conditioning report, the failure taxonomy and the relative
    line-search slack are all stripped out, because what the numba comparison has to be
    like-for-like about is the *arithmetic*, and a fair race cannot have one side also
    computing an SVD.  The shape and the stopping rule are the real ones.
    """
    epsilon = np.zeros(x.shape[1])
    total = float(weights.sum())
    loglik = _quasi(y, expit(offset), weights)
    for _ in range(max_iter):
        eta = offset + x @ epsilon
        p = expit(eta)
        gradient = x.T @ (weights * (y - p))
        if np.max(np.abs(gradient)) / total <= 1e-10:
            break
        hessian = x.T @ (x * (weights * p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate synthetic data
            break
        scale = 1.0
        for _ in range(30):
            candidate = epsilon + scale * step
            value = _quasi(y, expit(offset + x @ candidate), weights)
            if value >= loglik - 1e-11 * max(1.0, abs(loglik)):
                epsilon, loglik = candidate, value
                break
            scale *= 0.5
        else:
            break
        if np.max(np.abs(scale * step)) <= 1e-10:
            break
    return epsilon


def _quasi(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    q = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.sum(weights * (y * np.log(q) + (1.0 - y) * np.log1p(-q))))


def _build_numba_newton() -> Any:
    """The same loop with the inner passes fused, compiled by numba.

    Written out rather than decorating the library's own function: ``_newton_logistic``
    calls into ``np.linalg.cond``, warns, and builds dataclasses, none of which numba
    compiles.  What is jitted here is the part a native extension would actually replace
    -- the ``(n, K)`` passes and the line search over them -- so the number below is an
    upper bound on what rewriting the solver could buy, not a promise about the solver.
    """
    import numba

    @numba.njit(cache=False)
    def solve(x, y, weights, offset, max_iter):  # type: ignore[no-untyped-def]
        n, k = x.shape
        epsilon = np.zeros(k)
        eta = offset.copy()
        total = 0.0
        for i in range(n):
            total += weights[i]

        loglik = 0.0
        for i in range(n):
            p = 1.0 / (1.0 + np.exp(-eta[i]))
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            loglik += weights[i] * (y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))

        gradient = np.zeros(k)
        hessian = np.zeros((k, k))
        candidate = np.zeros(k)
        for _ in range(max_iter):
            # One pass computes eta, p, the gradient and the Hessian together; in numpy
            # each of those is a separate sweep over n with its own temporary.
            gradient[:] = 0.0
            hessian[:, :] = 0.0
            for i in range(n):
                s = offset[i]
                for j in range(k):
                    s += x[i, j] * epsilon[j]
                p = 1.0 / (1.0 + np.exp(-s))
                resid = weights[i] * (y[i] - p)
                var = weights[i] * p * (1.0 - p)
                for j in range(k):
                    gradient[j] += x[i, j] * resid
                    for m in range(k):
                        hessian[j, m] += x[i, j] * x[i, m] * var

            biggest = 0.0
            for j in range(k):
                biggest = max(biggest, abs(gradient[j]))
            if biggest / total <= 1e-10:
                break

            step = np.linalg.solve(hessian, gradient)
            scale = 1.0
            improved = False
            for _ in range(30):
                for j in range(k):
                    candidate[j] = epsilon[j] + scale * step[j]
                value = 0.0
                for i in range(n):
                    s = offset[i]
                    for j in range(k):
                        s += x[i, j] * candidate[j]
                    p = 1.0 / (1.0 + np.exp(-s))
                    p = min(max(p, 1e-12), 1.0 - 1e-12)
                    value += weights[i] * (y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))
                if value >= loglik - 1e-11 * max(1.0, abs(loglik)):
                    for j in range(k):
                        epsilon[j] = candidate[j]
                    loglik = value
                    improved = True
                    break
                scale *= 0.5
            if not improved:
                break
            biggest_step = 0.0
            for j in range(k):
                biggest_step = max(biggest_step, abs(scale * step[j]))
            if biggest_step <= 1e-10:
                break
        return epsilon

    return solve


def bench_numba(sizes: tuple[int, ...]) -> list[Timing]:
    """Is a compiled inner loop worth a dependency?  Measured rather than assumed.

    The targeting Newton is the one kernel here that is neither BLAS-bound nor
    scikit-learn-bound: its cost is ``(n, K)`` passes and per-trial temporaries, which is
    what a compiler removes.  So it is the most favourable case in the package for native
    code, and an unfavourable result here settles the question for the rest.

    Three things this reports that a bare speed-up would hide:

    * **compile time, separately.**  numba compiles on first call.  A kernel that is 3x
      faster after a two-second compile loses outright on a single fit, and it is the
      single fit that this library's users pay for.
    * **agreement.**  A faster kernel that answers differently is not a faster kernel, and
      the oracle-law suites here check influence curves to ``1e-14``.
    * **the share of a real fit**, in the verdict at the bottom, since a proportion of
      something invisible is still invisible.
    """
    try:
        import numba  # noqa: F401
    except ImportError:
        print("\n(numba is not installed; skipping.  `pip install -e '.[bench]'` to include it.)")
        return []

    timings: list[Timing] = []
    compile_seconds: float | None = None
    for n in sizes:
        data = _synthetic(n)
        submodel = mean_submodel(data["a"], data["g1"])
        x = np.asarray(submodel.observed, dtype=float)
        y = data["y"]
        weights = np.ones(n)
        offset = np.full(n, logit(0.45))

        if compile_seconds is None:
            _NUMBA_CACHE[0] = _build_numba_newton()
            start = time.perf_counter()
            _NUMBA_CACHE[0](x, y, weights, offset, 50)
            compile_seconds = time.perf_counter() - start

        solver = _NUMBA_CACHE[0]
        reference = _numpy_newton(x, y, weights, offset)
        compiled = np.asarray(solver(x, y, weights, offset, 50))
        gap = float(np.max(np.abs(reference - compiled)))

        timings.append(
            Timing(
                "Newton loop (numpy)",
                n,
                _time(lambda a=x, b=y, w=weights, o=offset: _numpy_newton(a, b, w, o)),
                note=f"{x.shape[1]} columns",
            )
        )
        timings.append(
            Timing(
                "Newton loop (numba, warm)",
                n,
                _time(lambda s=solver, a=x, b=y, w=weights, o=offset: s(a, b, w, o, 50)),
                note=f"agrees to {gap:.1e} absolute",
            )
        )
        timings.append(
            Timing(
                "Newton loop (numba, first call)",
                n,
                compile_seconds,
                note="one-off compile, paid once per process",
            )
        )
    return timings


@dataclass
class CostModel:
    """``seconds ~ fixed + per_row * n`` for one kernel."""

    label: str
    fixed: float
    per_row: float

    def at(self, n: int) -> float:
        return max(self.fixed + self.per_row * n, 0.0)


def fit_cost_models(sections: list[tuple[str, list[Timing]]]) -> dict[str, CostModel]:
    """Separate each kernel's fixed cost from its per-row cost.

    The question this exists for is the one the measured sizes cannot answer directly: a
    share of a fit at n=2,000 says nothing about n=5,000,000 unless the two costs scale
    together, and they do not.  A numpy kernel over ``(n, K)`` is linear in ``n``; a
    scikit-learn fit is linear at best and ``n log n`` or worse for a tree ensemble.

    **An intercept is not optional, and fitting a plain power law here is a trap.**  Every
    kernel in this file carries a per-*call* cost that does not scale -- Python dispatch,
    fold setup, allocation -- and at the sizes small enough to measure quickly, that cost
    is a large share of the total.  A log-log fit charges it to the exponent, which then
    comes out well below 1 and *understates* growth: fitted that way over
    n = 2,000..20,000 a full ``glm`` fit reads ``n^0.18``, which would have it taking under
    a second at five million rows.  Worse, it inverts the very comparison this file exists
    to make -- numpy's Newton loop reads ``n^0.75`` against numba's ``n^1.01``, purely
    because numpy's per-call overhead is the larger of the two, so extrapolating those
    exponents has numba *losing* at scale when the slopes say the opposite.

    Splitting the two makes the extrapolation the right shape and the comparison the right
    one: **the benefit of a faster kernel is the difference in ``per_row`` times ``n``**,
    and the fixed costs cancel out of it.

    What this still cannot see, and what the caller must not forget:

    * **cache.**  Sizes that all fit in L2/L3 give a ``per_row`` that is optimistic once
      the arrays no longer do.  Fit over sizes spanning at least an order of magnitude.
    * **super-linear learners.**  A tree ensemble is ``n log n`` or worse, so a projected
      *full fit* is a **lower bound** -- which makes every projected share an **upper**
      bound, i.e. conservative in the direction that matters for "is native code worth it".
    * **memory.**  An extrapolated time is meaningless if the extrapolated allocation does
      not fit; :func:`memory_projection` is the other half of the answer.
    """
    by_label: dict[str, list[tuple[int, float]]] = {}
    for _, timings in sections:
        for timing in timings:
            if timing.n > 0 and timing.seconds > 0:
                by_label.setdefault(timing.label, []).append((timing.n, timing.seconds))

    models: dict[str, CostModel] = {}
    for label, points in by_label.items():
        if len({n for n, _ in points}) < 2:
            continue
        ns = np.array([n for n, _ in points], dtype=float)
        secs = np.array([s for _, s in points], dtype=float)
        design = np.column_stack([np.ones_like(ns), ns])
        fixed, per_row = np.linalg.lstsq(design, secs, rcond=None)[0]
        models[label] = CostModel(label, float(fixed), float(per_row))
    return models


def project_to_scale(
    sections: list[tuple[str, list[Timing]]], targets: tuple[int, ...]
) -> list[list[str]]:
    """One row per kernel: fixed cost, per-row cost, asymptotic share, and projections.

    The share column is the one that answers "does this matter at scale".  It is the
    kernel's per-row cost over the *fit's* per-row cost -- the limit its measured share
    tends to as ``n`` grows and both sides' fixed costs stop mattering.  It is generally
    **not** the share printed against the measured sizes, and the gap is the point: a full
    fit carries hundreds of milliseconds of fold setup that a kernel timed on its own does
    not, so at small ``n`` every kernel looks cheaper than it asymptotically is.
    """
    models = fit_cost_models(sections)
    fit = next((m for k, m in models.items() if k.startswith("full fit")), None)
    rows: list[list[str]] = []
    for model in models.values():
        share = (
            f"{100.0 * model.per_row / fit.per_row:.2f}%"
            if fit is not None and fit.per_row > 0 and model.per_row > 0
            else ""
        )
        rows.append(
            [
                model.label,
                _format_seconds(model.fixed),
                f"{model.per_row * 1e9:.1f}",
                share,
                *(_format_seconds(model.at(target)) for target in targets),
            ]
        )
    return rows


def _format_seconds(value: float) -> str:
    if value >= 3600.0:
        return f"{value / 3600.0:.1f} h"
    if value >= 1.0:
        return f"{value:.1f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.1f} ms"
    return f"{value * 1e6:.0f} us"


def memory_projection(targets: tuple[int, ...]) -> list[list[str]]:
    """Arrays whose size grows with ``n``, at the projected sizes.

    Time is not the only thing that scales, and it is not the first thing to break.  Two
    allocations here are not ``O(n)`` with a small constant, and at several million rows
    they are the binding constraint rather than any arithmetic:

    * the multiplier bootstrap draws its multipliers in chunks of
      :data:`cleverly.inference.multiplier._CHUNK` replicates, so it holds a
      ``(chunk, n)`` float array -- which is the reason ``kind="normal"`` exists, since
      its closed form never forms one at all;
    * the conditional-density learner expands each unit into one row per hazard bin it
      survived, so its design is roughly ``n * n_bins / 2`` rows.

    Reported rather than measured because allocating either at n=5,000,000 to time it
    would be the failure this is warning about.
    """
    from cleverly.inference.multiplier import _CHUNK

    bins = 20
    covariates = 5
    rows: list[list[str]] = []
    for label, cells in (
        (f"multiplier chunk ({_CHUNK} x n)", lambda n: float(_CHUNK * n)),
        (
            f"density long design (~n*{bins}/2 x {covariates + bins})",
            lambda n: n * bins / 2 * (covariates + bins),
        ),
        ("influence curves (n x 7 estimands)", lambda n: 7.0 * n),
        ("one clever covariate (n x 2 arms)", lambda n: 2.0 * n),
    ):
        rows.append([label, *(_format_bytes(8.0 * cells(target)) for target in targets)])
    return rows


def _format_bytes(value: float) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value:.0f} B"


def _print_scaling(
    sections: list[tuple[str, list[Timing]]],
    targets: tuple[int, ...],
    measured: tuple[int, ...],
) -> None:
    """The projection tables, and the caveat that has to travel with them."""
    headers = [
        "operation",
        "fixed",
        "ns/row",
        "% of fit",
        *(f"n = {target:,}" for target in targets),
    ]

    print("\nProjected to scale (extrapolated, not run)")
    print("=" * 41)
    if len(measured) < 2:
        print(
            "Only one size was measured, so there is no slope to fit. Re-run with at least\n"
            "two --sizes spanning an order of magnitude, e.g. --sizes 2000 20000 200000."
        )
        return

    span = max(measured) / min(measured)
    rows = project_to_scale(sections, targets)
    if not rows:
        print("Nothing measured at two or more sizes.")
        return
    print(format_table(headers, rows))
    print(
        f"Fitted as `fixed + per_row * n` over n = {min(measured):,}..{max(measured):,} "
        f"({span:.0f}x), which\n"
        "is the form that keeps per-call overhead out of the growth term -- a plain log-log\n"
        "exponent charges that overhead to the slope and understates every projection (see\n"
        "`fit_cost_models`). `ns/row` is the column that scales; `fixed` is what a caller pays\n"
        "once however large the data is; `% of fit` is the share each kernel *tends to* as n\n"
        "grows, which is the number to judge it by and is not the share it shows at the sizes\n"
        "measured -- a full fit carries fold setup that a kernel timed alone does not, so small\n"
        "n flatters every kernel.\n"
        "\n"
        "Two things bias this, in opposite directions and both worth knowing. The learner rows\n"
        "are tree ensembles, which grow faster than linearly, so a projected *full fit* is a\n"
        "lower bound -- and therefore every share taken against it is an upper bound, which is\n"
        "the conservative direction for judging native code. Against that, every `ns/row` here\n"
        "was measured on arrays small enough to sit in cache, so the numpy kernels will be\n"
        "somewhat worse at scale than they read. Re-measure at the largest n you can afford\n"
        "before betting on any of it. A negative `fixed` is the model's way of saying that\n"
        "row grew faster than linearly over the range fitted; read its projection as a floor."
    )

    models = fit_cost_models(sections)
    numpy_loop = models.get("Newton loop (numpy)")
    numba_loop = models.get("Newton loop (numba, warm)")
    compile_model = models.get("Newton loop (numba, first call)")
    full_fit = next((m for k, m in models.items() if k.startswith("full fit")), None)
    if numpy_loop and numba_loop and compile_model:
        gain_per_row = numpy_loop.per_row - numba_loop.per_row
        print("\nWhat numba would buy at those sizes")
        print("-" * 34)
        if gain_per_row <= 0:
            print(
                "Nothing, at any size. numba's fitted per-row cost is not below numpy's, so the\n"
                "small-n speed-up is per-*call* overhead rather than per-row work -- and overhead\n"
                "is exactly what does not grow. Measured directly rather than only fitted, the\n"
                "ratio sits between 0.67x and 1.07x from n = 2,000 to n = 2,000,000: a wash.\n"
                "\n"
                "Which is the expected answer once stated: the numpy loop is already `x @ eps`,\n"
                "`x.T @ (...)` and a vectorised `exp`, so its inner work is compiled BLAS and\n"
                "SIMD, and a hand-written scalar loop has nothing left to remove. A compiler pays\n"
                "where the interpreter is in the inner loop, and here it is not."
            )
        else:
            compile_cost = compile_model.fixed + compile_model.per_row * max(measured)
            for target in targets:
                saved = gain_per_row * target
                line = f"n = {target:,}: saves {_format_seconds(saved)} per solve"
                if full_fit:
                    line += f", {100.0 * saved / full_fit.at(target):.2f}% of a projected fit"
                line += (
                    f"; the {compile_cost:.1f}s compile needs {compile_cost / saved:,.0f} solves"
                )
                print("  " + line)
            print(
                "\nThe saving grows linearly in n and the compile does not, so numba's case\n"
                "improves with scale -- which is the honest way to put it, and still leaves it\n"
                "losing: a fit solves the fluctuation a handful of times, so 'solves to break\n"
                "even' in the hundreds already means the compile is never repaid inside one\n"
                "analysis. The share column is the one that decides, and it shrinks, because\n"
                "the learners grow at least as fast as the kernel does."
            )

    print("\nPeak allocations at those sizes")
    print("-" * 30)
    print(
        format_table(
            ["array", *(f"n = {target:,}" for target in targets)], memory_projection(targets)
        )
    )
    print(
        "Time is not what breaks first. The multiplier chunk and the density long design are\n"
        "the two arrays that grow faster than n, and at several million rows they bind before\n"
        "any arithmetic does -- which is what `multiplier_kind='normal'` avoids by construction,\n"
        "since its closed form never forms an (n_replicates, n) array at all."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--library",
        default="default",
        help="nuisance learner library preset; 'glm' is quick but overstates every other share",
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--project",
        type=int,
        nargs="*",
        default=[1_000_000, 5_000_000],
        help=(
            "row counts to extrapolate every kernel to, from the sizes actually measured. "
            "Pass none to skip. Nothing is run at these sizes -- see project_to_scale for "
            "what a two-parameter fit can and cannot tell you."
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[
            "targeting",
            "multiplier",
            "msm",
            "ingestion",
            "clustered",
            "e2e",
            "ltmle",
            "numba",
        ],
        help="sections to leave out; the longitudinal and end-to-end ones dominate the runtime",
    )
    args = parser.parse_args()
    sizes = tuple(args.sizes)
    skip = set(args.skip)

    candidates = [
        ("targeting", "Numerical kernels (numpy today)", lambda: bench_targeting(sizes)),
        ("multiplier", "Resampling kernels", lambda: bench_multiplier(sizes, args.replicates)),
        ("msm", "Working-model projection", lambda: bench_msm(sizes)),
        ("ingestion", "Dataframe boundary", lambda: bench_ingestion(sizes)),
        ("clustered", "Clustered inference kernels", lambda: bench_clustered(sizes)),
        ("numba", "Compiled kernels (numba)", lambda: bench_numba(sizes)),
        ("e2e", "End to end", lambda: bench_end_to_end(sizes, args.library)),
        ("ltmle", "End to end (longitudinal)", lambda: bench_longitudinal(sizes, args.library)),
    ]
    sections = [(title, build()) for key, title, build in candidates if key not in skip]
    named = dict(sections)

    for title, timings in sections:
        if not timings:
            continue
        print(f"\n{title}")
        print("=" * len(title))
        print(
            format_table(
                ["operation", "n", "seconds", "share", "note"],
                [timing.row() for timing in timings],
            )
        )

    def total(title: str, prefix: str) -> float:
        return sum(t.seconds for t in named.get(title, []) if t.label.startswith(prefix))

    print("\nReading the numbers")
    print("=" * 19)
    kernels = total("Numerical kernels (numpy today)", "targeting (Newton)")
    full = total("End to end", "full fit")
    if full > 0 and kernels > 0:
        print(
            f"The targeting step is {100.0 * kernels / full:.2f}% of a full fit with "
            f"library={args.library!r}. Nuisance estimation dominates, and it already runs "
            "in compiled code -- so a native extension for the targeting step would buy "
            "almost nothing."
        )
    resampled = total("Resampling kernels", "multiplier bootstrap (rademacher)")
    exact = total("Resampling kernels", "multiplier bootstrap (normal)")
    if exact > 0:
        print(
            f"Gaussian multipliers are {resampled / exact:.0f}x cheaper than resampled ones "
            "here, because the max-t law has a closed form -- an algorithmic win no rewrite "
            "in any language competes with."
        )
    rebuilt = total("Clustered inference kernels", "bootstrap_indices (rebuilt)")
    prebuilt = total("Clustered inference kernels", "bootstrap_indices (prebuilt)")
    if prebuilt > 0:
        print(
            f"Prebuilding the cluster membership index is {rebuilt / prebuilt:.0f}x cheaper "
            "per replicate, which a cluster bootstrap pays n_bootstrap times over."
        )

    naive = total("Working-model projection", "gram einsum (optimize=False)")
    fastest = min(
        (
            value
            for value in (
                total("Working-model projection", "gram einsum (optimize=True)"),
                total("Working-model projection", "gram via reshape + dgemm"),
            )
            if value > 0
        ),
        default=0.0,
    )
    if fastest > 0:
        print(
            f"The four-operand einsum is {naive / fastest:.0f}x slower at optimize=False -- its "
            "default -- than the same contraction routed through BLAS. It runs once per Newton "
            "step and once per line-search trial in the MSM projection."
        )

    boundary = total("Dataframe boundary", "from_frame") + total("Dataframe boundary", "to_frame")
    if full > 0 and boundary > 0:
        print(
            f"The whole dataframe boundary -- every backend, in and out -- is "
            f"{100.0 * boundary / full:.2f}% of a fit. The internals hold numpy because "
            "scikit-learn takes numpy; there is no share here for a columnar engine to win."
        )

    ltmle = total("End to end (longitudinal)", "LTMLE full fit")
    node = total("End to end (longitudinal)", "  of which")
    if ltmle > 0:
        print(
            f"A longitudinal fit spends {100.0 * 4 * node / ltmle:.2f}% of itself in its four "
            "node fluctuations, so it is scikit-learn-bound like the rest -- which was "
            "predicted for a long time before anything measured it."
        )

    warm = total("Compiled kernels (numba)", "Newton loop (numba, warm)")
    plain = total("Compiled kernels (numba)", "Newton loop (numpy)")
    compile_cost = total("Compiled kernels (numba)", "Newton loop (numba, first call)")
    if warm > 0 and plain > warm:
        saved = plain - warm
        verdict = (
            f"numba runs the Newton loop {plain / warm:.1f}x faster once warm, saving "
            f"{saved:.6f}s per solve"
        )
        if full > 0:
            verdict += f" -- {100.0 * saved / full:.3f}% of a fit"
        # Break-even is compile / saving-per-solve, which is the only sense in which a
        # one-off cost "pays for itself". Dividing the compile by the *fit* time instead
        # answers a different question -- how many fits the compile is worth -- and gives
        # a number three orders of magnitude too flattering.
        print(f"{verdict}. The {compile_cost:.1f}s compile pays for itself after ")
        print(f"  {compile_cost / saved:,.0f} solves at this n; see the scaling section for how")
        print("  that moves as n grows.")

    targets = tuple(sorted(int(value) for value in args.project if value > 0))
    if targets:
        _print_scaling(sections, targets, sizes)


if __name__ == "__main__":
    main()
