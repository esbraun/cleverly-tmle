"""Builders for the kinds of fit whose capability rows RM23 checks against their calls.

A capability row reads ``available`` before any call. RM23 found rows that read
``available`` while their call then refused, and each such row was a kind of fit that
nobody had asked the row about. :data:`KINDS` names one builder per kind, so every test
that asks a row about a kind asks it about the same fit.

Each builder fits in sample with the explicit linear learners of
:func:`tests.conftest.linear_in_sample`, unless its docstring says otherwise. Each call
returns a fresh result, so no test reads a facade or a cached report that another test
filled.

:func:`problems` is the sweep's instrument, and :data:`MUTATIONS` holds the monkeypatched
defects it must see. Each mutation restores one pre-RM23 answer at the seam its fix added.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean, SplitPlan
from cleverly.assessment import (
    AssessmentStatus,
    DiagnosticsFacade,
    SensitivityFacade,
    replayability,
)
from cleverly.datasets import (
    make_binary_outcome,
    make_clustered,
    make_instrument,
    make_linear_ate,
    make_longitudinal,
    make_missing_outcome,
    make_missing_outcome_binary,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from tests import discrete_law, discrete_law_mar, regimes
from tests.conftest import (
    IN_SAMPLE,
    SELECTOR_CONFIGS,
    linear_ctmle,
    linear_in_sample,
    mean_one_weights,
)
from tests.unit._declaration_support import legacy_result
from tests.unit._direct_effect_support import COVARIATES as CDE_COVARIATES
from tests.unit._direct_effect_support import cde_frame
from tests.unit._simulated_confounding_support import _GRID

#: What the combined report writes when an operation it ran raised ``CapabilityError``
#: under a row that said the operation was available.
DECLINED = "the operation declined this request"


def _missing_frame() -> Any:
    """The missing-outcome law the RM23 probes fitted, ``make_missing_outcome(400, 4)``."""
    frame, _ = make_missing_outcome(n=400, seed=4)
    return frame


def _linear_frame() -> Any:
    """The complete-outcome law the RM23 probes fitted, ``make_linear_ate(400, 2)``."""
    frame, _ = make_linear_ate(n=400, seed=2)
    return frame


def fit_ordinary() -> Any:
    """A binary-treatment ATE fit with complete outcomes."""
    return TMLE(**linear_in_sample()).fit(_linear_frame(), outcome="Y", treatment="A").single()


def fit_binary() -> Any:
    """A binary-outcome fit, ``make_binary_outcome(400, 17)``."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()


def fit_missing() -> Any:
    """A binary-treatment fit with missing outcomes, which both tilt rows admit."""
    return (
        TMLE(**linear_in_sample())
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_shift() -> Any:
    """A capped shift of the continuous dose of ``make_shift_dose(300, 0)``."""
    frame, _ = make_shift_dose(n=300, seed=0)
    return (
        TMLE(**linear_in_sample(shifts=[Shift(0.5, cap=5.0)]))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def fit_cde() -> Any:
    """The controlled direct effect of :func:`tests.unit._direct_effect_support.cde_frame`.

    The fit reports one result per level of ``Z``, and this is the one at ``Z = 0``.
    """
    return TMLE(**linear_in_sample()).fit(
        cde_frame(),
        outcome="Y",
        treatment="A",
        covariates=CDE_COVARIATES,
        intermediate="Z",
    )[0.0]


def fit_multi_arm() -> Any:
    """A three-arm fit, ``make_multi_arm(600, 0)``."""
    frame, _ = make_multi_arm(n=600, seed=0)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()


def fit_ctmle_oat() -> Any:
    """An outcome-adaptive collaborative fit on ``make_instrument(500, 44)``."""
    frame, _ = make_instrument(n=500, seed=44)
    estimator = linear_ctmle("oat", estimands=("ate",))
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_msm() -> Any:
    """A linear working-model fit with complete outcomes."""
    return (
        TMLE(**linear_in_sample(msm=MSM.linear()))
        .fit(_linear_frame(), outcome="Y", treatment="A")
        .single()
    )


def fit_regime() -> Any:
    """The three regimes of :mod:`tests.regimes` on the discrete law."""
    return (
        TMLE(**linear_in_sample(interventions=regimes.interventions()))
        .fit(discrete_law.frame(), outcome="Y", treatment="A", covariates=("W",))
        .single()
    )


def fit_weighted() -> Any:
    """``make_linear_ate(400, 2)`` with nonconstant observation weights of mean one."""
    frame = _linear_frame()
    frame = frame.assign(weight=mean_one_weights(len(frame)))
    return (
        TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A", weights="weight").single()
    )


def fit_stratified() -> Any:
    """``make_linear_ate(400, 2)`` with a baseline stratum ``V`` that is also a covariate."""
    frame = _linear_frame()
    frame["V"] = np.where(frame["W1"] >= frame["W1"].median(), "high", "low")
    return (
        TMLE(**linear_in_sample())
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2", "W3", "W4", "V"),
            strata=("V",),
        )
        .single()
    )


def fit_clustered() -> Any:
    """``make_clustered(400, cluster_size=10, seed=7)``, 40 clusters of ten rows."""
    frame, _ = make_clustered(n=400, cluster_size=10, seed=7)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A", id="cluster").single()


def fit_incremental() -> Any:
    """An incremental fit with complete outcomes, whose estimand contains the propensity."""
    return (
        TMLE(**linear_in_sample(incremental=[Incremental(2.0)]))
        .fit(_linear_frame(), outcome="Y", treatment="A")
        .single()
    )


def fit_shift_missing() -> Any:
    """A shift fit with missing outcomes. The dose is the arm plus standard normal noise."""
    frame = _missing_frame()
    noise = np.random.default_rng(0).standard_normal(len(frame))
    dose = frame.assign(A=frame["A"] + noise)
    return (
        TMLE(**linear_in_sample(shifts=[Shift(0.5, cap=None)]))
        .fit(dose, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_incremental_missing() -> Any:
    """An incremental fit with missing outcomes, which has both truncation axes."""
    return (
        TMLE(**linear_in_sample(incremental=[Incremental(2.0)]))
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_regime_missing() -> Any:
    """The three regimes of :mod:`tests.regimes` on the missing-outcome discrete law."""
    return (
        TMLE(**linear_in_sample(interventions=regimes.interventions()))
        .fit(
            discrete_law_mar.frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
        .single()
    )


def fit_msm_missing() -> Any:
    """A linear working-model fit with missing outcomes, which reports coefficients only."""
    return (
        TMLE(**linear_in_sample(msm=MSM.linear()))
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_rr_missing() -> Any:
    """A risk-ratio-only fit with missing binary outcomes."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("rr",)))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_natural_course() -> Any:
    """The missing-outcome natural-course mean, which fits no treatment mechanism.

    The outcome is binary, because this target refuses a continuous outcome without
    declared ``q_bounds``.
    """
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("ey_obs",)))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_split_plan() -> Any:
    """A cross-fitted fit on the discrete law, given the split plan of a first fit.

    Two folds and a binary outcome, so no outcome support has to be declared. The plan
    labels the rows it was realised on, so ``refute`` refuses ``subset`` here.
    """
    frame = discrete_law.frame()
    folds = {"cross_fit": True, "n_folds": 2}
    first = TMLE(**linear_in_sample(**folds)).fit(frame, outcome="Y", treatment="A").single()
    supplied = TMLE(**linear_in_sample(**folds, split_plan=first.split_plan))
    return supplied.fit(frame, outcome="Y", treatment="A").single()


#: The covariate order :func:`fit_ctmle_ordered` declares, as ``make_instrument`` names it.
INSTRUMENT_ORDERING = ("W1", "W2", "W3")


def ctmle_ordered(ordering: tuple[str, ...] = INSTRUMENT_ORDERING) -> Any:
    """The ordered collaborative estimator of :func:`fit_ctmle_ordered`, at ``ordering``."""
    return linear_ctmle(
        "ordered",
        ordering=ordering,
        selection_folds=SELECTOR_CONFIGS["ordered"]["selection_folds"],
        estimands=("ate",),
    )


def fit_ctmle_ordered() -> Any:
    """An ordered collaborative fit with an explicit ordering, on ``make_instrument(500, 44)``."""
    frame, _ = make_instrument(n=500, seed=44)
    return ctmle_ordered().fit(frame, outcome="Y", treatment="A").single()


def drtmle_settings() -> dict[str, Any]:
    """Cross-fitted DR-TMLE with explicit linear learners, as the companion tests fit it."""
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 3,
        "learner_folds": 2,
        "random_state": 0,
        "simultaneous": False,
        "estimands": ("ate",),
    }


def fit_drtmle(*, companion: bool = False) -> Any:
    """A DR-TMLE fit of a binary outcome, with an ``evaluation=`` companion when asked.

    The outcome is binary, so the cross-fitted fit needs no declared ``q_bounds``. The
    companion is an independent draw of the same law.
    """
    frame, _ = make_binary_outcome(n=240, seed=11)
    evaluation = make_binary_outcome(n=120, seed=12)[0] if companion else None
    estimator = DRTMLE(**drtmle_settings(), evaluation=evaluation)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_drtmle_companion() -> Any:
    """:func:`fit_drtmle` with an ``evaluation=`` companion."""
    return fit_drtmle(companion=True)


# ------------------------------------------------------------------- restored results


def discrete_fit(**overrides: Any) -> Any:
    """A fit of the discrete law with the linear learners, in sample unless overridden."""
    estimator = TMLE(**linear_in_sample(**overrides))
    return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


def cross_fitted() -> Any:
    """A cross-fitted fit of the discrete law in two folds."""
    return discrete_fit(cross_fit=True, n_folds=2)


def restored(result: Any, **configuration: Any) -> Any:
    """``result`` saved, given ``configuration`` on its estimator, and loaded.

    ``__init__`` refuses each configuration the tests use. Only a result that an earlier
    version saved, or a copied estimator, can carry one.
    """
    old = loads(dumps(result))
    vars(old.estimator).update(configuration)
    return loads(dumps(old))


def as_saved_by_v011(result: Any) -> Any:
    """``result`` as release 0.1.1 saved it: its estimator holds no ``split_plan``."""
    return legacy_result(result, "split_plan", lambda old: [old.estimator])


def without_provenance() -> Any:
    """A supplied-plan fit restored with a plan that records no generator."""
    result = fit_split_plan()
    return restored(result, split_plan=SplitPlan(result.estimator.split_plan.assignments))


def ctmle_stratified() -> Any:
    """An in-sample greedy collaborative fit restored with a stratified fold policy."""
    estimator = linear_ctmle("greedy", **SELECTOR_CONFIGS["greedy"], estimands=("ate",))
    result = estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()
    return restored(result, stratify_folds="treatment")


def unbounded_scale() -> Any:
    """A cross-fitted continuous outcome with declared bounds, restored without them."""
    frame = _linear_frame()
    bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
    estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, q_bounds=bounds))
    return restored(estimator.fit(frame, outcome="Y", treatment="A").single(), q_bounds=None)


def restored_stratified() -> Any:
    """The two-fold cross-fitted fit restored under ``stratify_folds="treatment"``."""
    return restored(cross_fitted(), stratify_folds="treatment")


def restored_v011() -> Any:
    """The shape of the real 0.1.1 artifact: two stratified folds and no ``split_plan``.

    Release 0.1.1 accepted ``stratify_folds="treatment"`` and wrote no ``split_plan``, so
    this result needs the class default to be read at all, and then its refit is refused.
    """
    return as_saved_by_v011(restored_stratified())


# ----------------------------------------------------------------------- longitudinal


def fit_ltmle() -> Any:
    """The in-sample two-node regime fit of ``tests/unit/test_assessment_contract.py``."""
    frame, _ = make_longitudinal(n=400, seed=12)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
    )
    return study.identify(RegimeMean({"always": 1, "never": 0})).estimate(
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=5,
        simultaneous=False,
        **IN_SAMPLE,
    )


# ---------------------------------------------------------------------- replay slots


def replay_disagreements(result: Any, estimands: tuple[str, ...]) -> list[str]:
    """Every point replay slot of ``result`` that disagrees with the call it stands for.

    ``retarget_cached_nuisances`` stands for ``estimator.retarget`` on the cached
    nuisances, and ``refit_nuisances`` for ``estimator.refit`` on the result's own data.
    A :class:`~cleverly.exceptions.CapabilityError` or
    :class:`~cleverly.exceptions.DataError` is a refusal. Any other exception is a defect,
    so it propagates. An empty list is agreement.
    """
    replay = replayability(result)
    estimator = result.estimator
    calls: dict[str, Callable[[], Any]] = {
        "retarget_cached_nuisances": lambda: estimator.retarget(
            result.data, result.nuisance, estimands=estimands
        ),
        "refit_nuisances": lambda: estimator.refit(
            result.data, intermediate_value=result.intermediate_value
        ),
    }
    problems = []
    for slot, call in calls.items():
        try:
            call()
        except (CapabilityError, DataError) as error:
            refused: str | None = f"{type(error).__name__}: {error}"
        else:
            refused = None
        if getattr(replay, slot) != (refused is None):
            problems.append(f"{slot} reads {getattr(replay, slot)}, and the call gave {refused}")
    return problems


def assert_replay_agrees(result: Any, estimands: tuple[str, ...]) -> None:
    """Each point replay slot of ``result`` reads true exactly when its call runs."""
    assert replay_disagreements(result, estimands) == []


def assert_replay_rows_refused(result: Any) -> None:
    """Every row that needs a replay slot reads unavailable, and a combined report runs.

    For a restored result whose replay slots read false. The list of rows is not empty,
    and the ``refute`` row reads unavailable too. A longitudinal ``refute`` row declares
    no replay slot, because it is unavailable for every longitudinal result.
    """
    rows = [
        row
        for facade in (result.diagnostics, result.sensitivity)
        for row in facade.capabilities
        if row.requires_replay is not None
    ]
    assert rows
    assert [row.operation for row in rows if row.available] == []
    assert not result.diagnostics.capability("refute").available
    report = result.assess(include_refits=True, include_retargets=True, random_state=0)
    assert report.diagnostics["refute"].status is AssessmentStatus.UNAVAILABLE


# ------------------------------------------------------------------------- the kinds


@dataclass(frozen=True)
class Kind:
    """One kind of fit the sweep asks every row about.

    Parameters
    ----------
    build : callable
        Returns a fresh result of this kind.
    must_run : frozenset of str
        Rows whose operation must answer in the sweep's report. This is the nonzero
        witness: a row refused by mistake passes the no-decline check and fails here.
    live : bool
        Whether the result comes from a fit in this version rather than a restored
        artifact. Every live point result keeps ``refit_nuisances``.
    """

    build: Callable[[], Any]
    must_run: frozenset[str]
    live: bool = True


def _kind(build: Callable[[], Any], *must_run: str, live: bool = True) -> Kind:
    return Kind(build, frozenset(must_run), live)


#: The two RM23 rows every live point fit that admits the default axis answers.
_POINT = ("refute", "truncation_curve")
_TILT = ("missingness", "tipping_gamma")

#: One kind of fit per name. The first 16 are the kinds of the 2026-09-22 RM23 probe. The
#: next seven are the siblings the plan's probes found, then three restored results whose
#: refit this version refuses, and one longitudinal fit.
KINDS: dict[str, Kind] = {
    "ordinary": _kind(fit_ordinary, *_POINT),
    "binary": _kind(fit_binary, *_POINT),
    "missing": _kind(fit_missing, *_POINT, *_TILT),
    "shift": _kind(fit_shift, *_POINT),
    "cde": _kind(fit_cde, *_POINT),
    "multi_arm": _kind(fit_multi_arm, *_POINT),
    "ctmle_oat": _kind(fit_ctmle_oat, *_POINT),
    "msm": _kind(fit_msm, *_POINT),
    "regime": _kind(fit_regime, *_POINT),
    "weighted": _kind(fit_weighted, *_POINT),
    "stratified": _kind(fit_stratified, *_POINT),
    "clustered": _kind(fit_clustered, *_POINT),
    "drtmle": _kind(fit_drtmle, *_POINT),
    "ctmle_ordered": _kind(fit_ctmle_ordered, *_POINT),
    "shift+missing": _kind(fit_shift_missing, *_POINT),
    "incremental": _kind(fit_incremental, "refute"),
    "incremental+missing": _kind(fit_incremental_missing, *_POINT),
    "regime+missing": _kind(fit_regime_missing, *_POINT),
    "msm+missing": _kind(fit_msm_missing, *_POINT),
    "rr+missing": _kind(fit_rr_missing, *_POINT),
    "natural_course": _kind(fit_natural_course, *_POINT),
    "split_plan": _kind(fit_split_plan, *_POINT),
    "drtmle_companion": _kind(fit_drtmle_companion, *_POINT),
    "restored_stratified": _kind(restored_stratified, "truncation_curve", live=False),
    "restored_v011": _kind(restored_v011, "truncation_curve", live=False),
    "restored_unbounded_scale": _kind(unbounded_scale, "truncation_curve", live=False),
    "ltmle": _kind(fit_ltmle, "truncation_curve"),
}

#: The mismatches the sweep found outside RM23, which stay open, by kind. Each is the
#: exact line :func:`problems` reports, so the sweep fails when one is fixed, and the
#: entry is removed then.
#:
#: ``split_plan`` fits the discrete law, which has the one covariate ``W``. Its
#: ``benchmark`` row reads available and asks for ``covariates=``, but no value runs: the
#: call refits without the named covariates and raises ``DataError("cannot drop every
#: covariate")``. By the request rule of :func:`cleverly.assessment._argument_resolved`,
#: a row whose omitted argument has no value that runs reads ``unavailable``. The sweep
#: leaves the argument out, so the row stays deferred.
OPEN: dict[str, tuple[str, ...]] = {
    "split_plan": (
        "benchmark is still deferred: needs an explicit covariates argument, which a "
        "combined report has no basis to choose",
    ),
}


# ------------------------------------------------------------------------- the sweep


def _benchmark_covariates(result: Any) -> list[str] | None:
    """The first covariate, or ``None`` on a fit that has no other one to keep.

    ``benchmark`` refits without the named covariates, and a refit needs one left, so no
    ``covariates=`` value runs on a fit with a single covariate. See :data:`OPEN`.
    """
    names = list(result.data.covariate_names)
    return names[:1] if len(names) > 1 else None


def _fillers(result: Any) -> dict[str, Callable[[], Any]]:
    """The value the sweep supplies for each argument a row can defer on.

    ``None`` means that no value runs, so the sweep leaves the argument out.
    """
    return {
        "estimand": lambda: next(iter(result.estimates)),
        "covariates": lambda: _benchmark_covariates(result),
        "grid": lambda: _GRID,
        "bounds": lambda: (0.05,),
        "mechanism": lambda: True,
        "tests": lambda: ("random_common_cause",),
    }


def _facades(result: Any) -> tuple[Any, Any]:
    return result.diagnostics, result.sensitivity


def fill_deferred(result: Any, arguments: dict[str, dict[str, Any]]) -> bool:
    """Supply one value for every argument a row of ``result`` still waits on.

    A row waits on an argument when it defers on it, or when it runs and declares it
    required. Rows refused for the whole fit are left alone, because no argument lifts
    them. Returns whether anything was filled, since a filled argument can resolve a row
    to a new deferral.
    """
    fillers = _fillers(result)
    filled = False
    for facade in _facades(result):
        for declared in facade.capabilities:
            request = arguments.get(declared.operation, {})
            row = facade._capability_for_arguments(declared.operation, request)
            if not (row.available or row.status is AssessmentStatus.DEFERRED):
                continue
            for name in row.requires_arguments:
                value = None if name in request else fillers[name]()
                if value is not None:
                    request = {**request, name: value}
                    filled = True
            if request:
                arguments[declared.operation] = request
    return filled


def sweep_arguments(result: Any) -> dict[str, dict[str, Any]]:
    """The arguments that leave no row of ``result`` waiting on the caller.

    One replicate per refutation keeps the refits cheap. So does one bound for a
    point-treatment curve that refits at each bound, which a guarded DR-TMLE fit declares.
    The bound grid decides no refusal: the rows and the calls resolve the axis alone.
    """
    arguments: dict[str, dict[str, Any]] = {"refute": {"n_replicates": 1}}
    curve = result.diagnostics.capability("truncation_curve")
    if curve.result_family == "point" and curve.execution == "refit":
        arguments["truncation_curve"] = {"bounds": (0.05,)}
    while fill_deferred(result, arguments):
        pass
    return arguments


def problems(result: Any, arguments: dict[str, dict[str, Any]]) -> list[str]:
    """Every way a row of ``result`` and its call disagree under ``arguments``.

    Four checks. ``assess`` with every cost paid must return a report: any exception is a
    problem, because a combined report exists to name refusals rather than to raise them.
    No item may carry :data:`DECLINED`, which marks a row that said available and a call
    that refused. No item may stay deferred, since ``arguments`` supplies every argument a
    row named. And every row that the request resolves as available must run as a direct
    call with the same arguments. An empty list is agreement.
    """
    try:
        report = result.assess(
            include_refits=True, include_retargets=True, arguments=arguments, random_state=0
        )
    except Exception as error:
        return [f"assess raised {type(error).__name__}: {error}"]
    found = []
    items = (*report.validation.items, *report.diagnostics.items, *report.sensitivity.items)
    for item in items:
        if DECLINED in (item.detail or ""):
            found.append(f"{item.name} declined: {item.detail}")
        if item.status is AssessmentStatus.DEFERRED:
            found.append(f"{item.name} is still deferred: {item.detail}")
    for facade in _facades(result):
        for declared in facade.capabilities:
            request = dict(arguments.get(declared.operation, {}))
            if declared.accepts_random_state:
                request["random_state"] = 0
            row = facade._capability_for_arguments(declared.operation, request)
            if not row.available or not set(row.requires_arguments) <= set(request):
                # A row still waiting on an argument is reported as deferred above.
                continue
            try:
                getattr(facade, declared.operation)(**request)
            except CapabilityError as error:
                found.append(f"{declared.operation} reads available, and its call raised {error}")
    return found


#: The statuses of an item whose operation did not answer.
_OMITTED = frozenset(
    {AssessmentStatus.DEFERRED, AssessmentStatus.UNAVAILABLE, AssessmentStatus.NOT_APPLICABLE}
)


def sweep(result: Any) -> list[str]:
    """:func:`problems` under :func:`sweep_arguments`, for a result nobody has asked yet.

    Reading the rows to fill their arguments is part of the sweep, so an exception there
    is a problem too. Without the class default for ``split_plan``, a result saved by
    release 0.1.1 raises as soon as its first row is read.
    """
    try:
        arguments = sweep_arguments(result)
    except Exception as error:
        return [f"reading the rows raised {type(error).__name__}: {error}"]
    return problems(result, arguments)


def ran(result: Any, arguments: dict[str, dict[str, Any]]) -> frozenset[str]:
    """The operations that answered in the sweep's report of ``result``.

    ``tipping_gamma`` can answer ``None``, a tilt that never reaches the null, so an
    answer is read from the status rather than from a retained report.
    """
    report = result.assess(
        include_refits=True, include_retargets=True, arguments=arguments, random_state=0
    )
    items = (*report.validation.items, *report.diagnostics.items, *report.sensitivity.items)
    return frozenset(item.name for item in items if item.status not in _OMITTED)


# ------------------------------------------------------------------------ mutations

_TILT_RULE = SensitivityFacade._tilt_rule

#: The tilt rules the rows read before RM23: the longitudinal literal, the natural course,
#: and the missing-outcome flag. Every later rule was the call's alone.
_PRE_RM23_TILT_RULES = ("longitudinal", "natural_course", "missing_outcome")


def _pre_rm23_tilt_rule(self: SensitivityFacade) -> tuple[str, str] | None:
    """The tilt rows as they were: available on every other fit with missing outcomes."""
    rule = _TILT_RULE(self)
    return rule if rule is not None and rule[0] in _PRE_RM23_TILT_RULES else None


#: Each seam an RM23 fix added, as its owner and attribute name.
_SEAMS: tuple[tuple[type, str], ...] = (
    (SensitivityFacade, "_tilt_rule"),
    (DiagnosticsFacade, "_truncation_gated"),
    (DiagnosticsFacade, "_refute_gated"),
    (CTMLE, "_configured_for_refit"),
    (DRTMLE, "_configured_for_refit"),
    (TMLE, "_refit_configuration_refusal"),
)


def _passthrough(patch: pytest.MonkeyPatch) -> None:
    """Replace every seam with a wrapper that returns what the seam returns."""
    for owner, name in _SEAMS:
        original = vars(owner)[name]
        patch.setattr(owner, name, functools.wraps(original)(lambda *a, _f=original: _f(*a)))
    patch.setattr(TMLE, "split_plan", None)


def _identity_gate(self: Any, capability: Any, arguments: Any) -> Any:
    return capability


def _unconfigured(self: Any, data: Any) -> Any:
    return self


@dataclass(frozen=True)
class Mutation:
    """One monkeypatched defect, and the kinds of :data:`KINDS` whose sweep must fail.

    Parameters
    ----------
    describe : str
        What the mutation restores.
    apply : callable
        Applies the mutation to a ``pytest.MonkeyPatch``.
    fails_on : frozenset of str
        The kinds whose :func:`sweep` must report other problems than their :data:`OPEN`
        entry under the mutation. Every other kind must report exactly that entry.
    """

    describe: str
    apply: Callable[[pytest.MonkeyPatch], None]
    fails_on: frozenset[str]


_TILT_KINDS = frozenset(
    {"shift+missing", "incremental+missing", "regime+missing", "msm+missing", "rr+missing"}
)

#: M0 to M7 of the RM23 plan. M0 wraps every seam and changes nothing, so a failure there
#: is the wrapping and not a defect.
MUTATIONS: dict[str, Mutation] = {
    "M0": Mutation("every seam wrapped and unchanged", _passthrough, frozenset()),
    "M1": Mutation(
        "the tilt rows ignore every rule the rows did not read before RM23",
        lambda patch: patch.setattr(SensitivityFacade, "_tilt_rule", _pre_rm23_tilt_rule),
        _TILT_KINDS,
    ),
    "M2": Mutation(
        "the truncation row ignores the predicate",
        lambda patch: patch.setattr(DiagnosticsFacade, "_truncation_gated", _identity_gate),
        frozenset({"incremental", "incremental+missing"}),
    ),
    "M3": Mutation(
        "the refute row ignores the predicate",
        lambda patch: patch.setattr(DiagnosticsFacade, "_refute_gated", _identity_gate),
        frozenset({"split_plan", "natural_course"}),
    ),
    "M4": Mutation(
        "CTMLE ignores a covariate the refit adds",
        lambda patch: patch.setattr(CTMLE, "_configured_for_refit", _unconfigured),
        frozenset({"ctmle_ordered"}),
    ),
    "M5": Mutation(
        "DRTMLE keeps a companion that lacks a refit covariate",
        lambda patch: patch.setattr(DRTMLE, "_configured_for_refit", _unconfigured),
        frozenset({"drtmle_companion"}),
    ),
    "M6": Mutation(
        "the refit slot ignores the refit preflight",
        lambda patch: patch.setattr(TMLE, "_refit_configuration_refusal", lambda self, data: None),
        frozenset({"restored_stratified", "restored_v011", "restored_unbounded_scale"}),
    ),
    "M7": Mutation(
        "no class default for split_plan",
        lambda patch: patch.delattr(TMLE, "split_plan"),
        frozenset({"restored_v011"}),
    ),
}
