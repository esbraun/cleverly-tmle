"""One scenario per estimator flavour: fit once, then time the post-nuisance half.

Each of these calls the shipped API.  The fit is outside the timed region; what is timed is
``retarget`` -- the package's own name for "run the targeting step and build the estimates
from cached nuisance fits" -- or, where a flavour has no ``retarget``, the closest thing it
has and a note saying why.

**Two flavours do not have one, and the reason matters more than the workaround.**

``LTMLE`` refuses a ``retarget`` by name: ``g_bounds`` enters the *pseudo-outcome* of every
earlier node through the recursion, so re-solving the fluctuation alone would answer a
different question.  Its post-nuisance half is therefore interleaved with learner fits by
construction, and the honest measurement is the fit's own profile with the learner lines
subtracted -- which is what :func:`ltmle_scenario` reports, with the subtraction stated.

``CTMLE``'s candidate search is likewise not separable: the candidates *are* learner fits.
Its scenario reports the post-selection ``retarget`` and, beside it, the share of the fit
that is candidate fitting, because the second is what makes the first uninteresting.
"""

from __future__ import annotations

import cProfile
import pstats
import time
import warnings
from typing import Any

import numpy as np

from . import ScenarioResult, ScenarioSpec, register

__all__ = ["ALL_LIBRARIES"]

#: The two presets every scenario is run at.  Quoting a `glm` share as the verdict is the
#: standard way to mislead with this measurement: it is the cheapest preset available and
#: inflates every package-owned share several-fold.
ALL_LIBRARIES = ("glm", "default")

_GAUSS = ["ey1", "ey0", "ate", "att", "atc"]
_BIN = ["ey1", "ey0", "ate", "att", "atc", "rr", "or"]


def _data(maker: Any, n: int, seed: int = 1) -> Any:
    from cleverly.data import CausalData

    frame, _ = maker(n=n, seed=seed)
    covariates = [c for c in frame.columns if c not in ("Y", "A", "id")]
    return CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=covariates)


def _time(call: Any, repeats: int = 3) -> float:
    call()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - start) / repeats


def _learner_share(profile: cProfile.Profile) -> float:
    """Share of a profile's total time spent inside scikit-learn, LightGBM or threadpoolctl.

    Crude by construction -- it matches on the file path -- and that is fine for what it is
    used for: separating "the package's own arithmetic" from "a learner fit and the
    machinery around it" inside a fit that interleaves them.  It is reported as a share
    rather than subtracted silently, so a reader can see the split rather than a residual.
    """
    stats = pstats.Stats(profile)
    total = 0.0
    external = 0.0
    for (filename, _, _), (_, _, tottime, _, _) in stats.stats.items():
        total += tottime
        if any(
            marker in filename
            for marker in ("sklearn", "lightgbm", "threadpoolctl", "joblib", "scipy")
        ):
            external += tottime
    return external / total if total else float("nan")


# ------------------------------------------------------------------ point treatment


def tmle_scenario(
    n: int = 20_000, library: str = "glm", targeting: str = "iterative"
) -> ScenarioResult:
    """Pooled TMLE: cached nuisances through targeting, estimands, curves and inference."""
    from cleverly.datasets import make_binary_outcome
    from cleverly.estimators import TMLE

    data = _data(make_binary_outcome, n)
    estimator = TMLE(
        estimands=_BIN,
        outcome_learner=library,
        treatment_learner=library,
        targeting=targeting,
        n_folds=2,
        learner_folds=2,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(data).single()
    fit_seconds = time.perf_counter() - start
    nuisance = result.nuisance

    detail = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for count, names in ((1, ["ate"]), (3, _GAUSS[:3]), (5, _GAUSS), (7, _BIN)):
            detail[f"estimands_{count}"] = _time(
                lambda names=names: estimator.retarget(data, nuisance, estimands=names)
            )
        post = detail["estimands_7"]
    return ScenarioResult(
        name=f"tmle_{targeting}",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=post,
        detail=detail,
        note="detail keys are the same retarget at 1/3/5/7 requested estimands",
    )


def cvtmle_scenario(n: int = 20_000, library: str = "glm", n_folds: int = 10) -> ScenarioResult:
    """CV-TMLE: a separate fluctuation per validation fold, stitched back by index."""
    from cleverly.datasets import make_binary_outcome
    from cleverly.estimators import TMLE

    data = _data(make_binary_outcome, n)
    estimator = TMLE(
        estimands=_GAUSS,
        outcome_learner=library,
        treatment_learner=library,
        targeting_scheme="fold",
        cv_evaluation=True,
        n_folds=n_folds,
        learner_folds=2,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(data).single()
    fit_seconds = time.perf_counter() - start
    nuisance = result.nuisance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = _time(lambda: estimator.retarget(data, nuisance, estimands=_GAUSS))
    return ScenarioResult(
        name=f"cvtmle_{n_folds}folds",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=post,
        detail={"n_folds": float(n_folds)},
        note="_solve_by_fold is a serial Python loop over folds; the folds are independent",
    )


def sensitivity_scenario(n: int = 20_000, library: str = "glm", grid: int = 25) -> ScenarioResult:
    """A truncation sweep: the same nuisances retargeted at ``grid`` bounds.

    The workload a compiled kernel is most plausible for, and the one this whole package
    exists to price: the compilation is paid once against ``grid`` calls, and the nuisance
    fit is paid once against all of them.  Reported as latency for one point and throughput
    for the grid, because a sweep is judged on the second.
    """
    from cleverly.datasets import make_binary_outcome
    from cleverly.estimators import TMLE

    data = _data(make_binary_outcome, n)
    estimator = TMLE(
        estimands=_GAUSS,
        outcome_learner=library,
        treatment_learner=library,
        n_folds=2,
        learner_folds=2,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(data).single()
    fit_seconds = time.perf_counter() - start
    nuisance = result.nuisance
    bounds = [(lower, 1.0 - lower) for lower in np.linspace(0.001, 0.2, grid)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = _time(
            lambda: estimator.retarget(data, nuisance, estimands=_GAUSS, g_bounds=bounds[grid // 2])
        )
        start = time.perf_counter()
        for bound in bounds:
            estimator.retarget(data, nuisance, estimands=_GAUSS, g_bounds=bound)
        whole = time.perf_counter() - start
    return ScenarioResult(
        name=f"sensitivity_grid{grid}",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=whole,
        detail={
            "one_point_seconds": one,
            "grid_points": float(grid),
            "grid_over_fit": whole / fit_seconds,
        },
        note="the repeated workload: one fit, `grid` retargets, independent per point",
    )


# --------------------------------------------------------------------- longitudinal


def ltmle_scenario(n: int = 20_000, library: str = "glm") -> ScenarioResult:
    """LTMLE has no ``retarget``, so this profiles the fit and reports the external share.

    The refusal is deliberate and documented in the package: ``g_bounds`` enters the
    pseudo-outcome of every earlier node through the recursion, so there is no fluctuation
    to re-solve on its own.  What that means for this benchmark is that the post-nuisance
    half cannot be *called*; it can only be separated inside a profile.  ``detail`` carries
    the split, and ``post_nuisance_seconds`` is the fit net of the learner machinery.
    """
    from cleverly.datasets import make_longitudinal
    from cleverly.longitudinal import LTMLE, LongitudinalData

    frame, _ = make_longitudinal(n=n, seed=1)
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    estimator = LTMLE(
        {"always": [1, 1], "never": [0, 0]},
        outcome_learner=library,
        treatment_learner=library,
        n_folds=2,
        learner_folds=2,
    )
    profile = cProfile.Profile()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile.enable()
        estimator.fit(data)
        profile.disable()
    fit_seconds = time.perf_counter() - start
    external = _learner_share(profile)
    return ScenarioResult(
        name="ltmle",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=fit_seconds * (1.0 - external),
        detail={"external_share": external},
        note=(
            "no retarget by design: g_bounds enters every earlier node's pseudo-outcome, "
            "so the post-nuisance half is separable only inside a profile"
        ),
    )


def survival_scenario(n: int = 5_000, library: str = "glm") -> ScenarioResult:
    """A discrete-time survival fit: one backward pass per horizon per regimen."""
    from cleverly.datasets import make_longitudinal_survival
    from cleverly.longitudinal import LTMLE, LongitudinalData

    frame, _ = make_longitudinal_survival(n=n, seed=1)
    columns = list(frame.columns)
    outcomes = [c for c in columns if c.startswith("Y")]
    data = LongitudinalData.from_frame(
        frame,
        outcome=outcomes,
        treatment=[c for c in columns if c.startswith("A")],
        baseline=[c for c in columns if c.startswith("W")],
        time_varying=[[], ["L2"]] if "L2" in columns else None,
        censoring=[c for c in columns if c.startswith("C")] or None,
    )
    estimator = LTMLE(
        {"always": [1, 1], "never": [0, 0]},
        outcome_learner=library,
        treatment_learner=library,
        n_folds=2,
        learner_folds=2,
    )
    profile = cProfile.Profile()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile.enable()
        estimator.fit(data)
        profile.disable()
    fit_seconds = time.perf_counter() - start
    external = _learner_share(profile)
    return ScenarioResult(
        name="survival",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=fit_seconds * (1.0 - external),
        detail={"external_share": external, "horizons": float(len(outcomes))},
        note="T(T+1)/2 node regressions per regimen; the mechanism is fitted once",
    )


# -------------------------------------------------------------------- the variants


def drtmle_scenario(n: int = 5_000, library: str = "glm") -> ScenarioResult:
    """DR-TMLE: the alternation refits its reduced regressions, so ``retarget`` is not cheap.

    This is the scenario that produced the run's largest single finding, and it is not a
    numba one: the ``retarget`` costs *more than the fit*, and the majority of it is
    :mod:`threadpoolctl` constructing a controller per learner fit.  ``detail`` carries the
    external share so the claim is on the record rather than in a commit message.
    """
    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import DRTMLE

    data = _data(make_linear_ate, n)
    estimator = DRTMLE(
        outcome_learner=library,
        treatment_learner=library,
        estimands=["ey1", "ey0", "ate"],
        n_folds=2,
        learner_folds=2,
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(data).single()
    fit_seconds = time.perf_counter() - start
    nuisance = result.nuisance

    profile = cProfile.Profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.retarget(data, nuisance, estimands=["ey1", "ey0", "ate"])
        start = time.perf_counter()
        profile.enable()
        estimator.retarget(data, nuisance, estimands=["ey1", "ey0", "ate"])
        profile.disable()
        post = time.perf_counter() - start
    return ScenarioResult(
        name="drtmle",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=post,
        detail={"external_share": _learner_share(profile)},
        note="the alternation refits the reduced regressions, so retarget is not learner-free",
    )


def ctmle_scenario(n: int = 5_000, library: str = "glm") -> ScenarioResult:
    """CTMLE: the post-selection retarget, beside the share of the fit that is candidates."""
    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import CTMLE

    data = _data(make_linear_ate, n)
    estimator = CTMLE(
        outcome_learner=library,
        estimands=["ey1", "ey0", "ate"],
        n_folds=2,
        learner_folds=2,
    )
    profile = cProfile.Profile()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile.enable()
        result = estimator.fit(data).single()
        profile.disable()
    fit_seconds = time.perf_counter() - start
    nuisance = result.nuisance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = _time(lambda: estimator.retarget(data, nuisance, estimands=["ey1", "ey0", "ate"]))
    return ScenarioResult(
        name="ctmle",
        library=library,
        n=n,
        fit_seconds=fit_seconds,
        post_nuisance_seconds=post,
        detail={"external_share": _learner_share(profile)},
        note="the candidate search is candidate-fitting-bound; the retarget is milliseconds",
    )


for _spec in (
    ScenarioSpec("tmle", "tmle", tmle_scenario, "pooled targeting, Newton"),
    ScenarioSpec(
        "tmle_one_step",
        "tmle",
        lambda n=20_000, library="glm": tmle_scenario(n, library, targeting="one_step"),
        "pooled targeting, the universal least-favourable walk",
    ),
    ScenarioSpec("cvtmle", "cvtmle", cvtmle_scenario, "fold-specific targeting"),
    ScenarioSpec("sensitivity", "inference", sensitivity_scenario, "a truncation grid"),
    ScenarioSpec("ltmle", "ltmle", ltmle_scenario, "the sequential recursion"),
    ScenarioSpec("survival", "survival", survival_scenario, "cumulative risk by horizon"),
    ScenarioSpec("drtmle", "drtmle", drtmle_scenario, "the reduction alternation"),
    ScenarioSpec("ctmle", "ctmle", ctmle_scenario, "the candidate path"),
):
    register(_spec)
