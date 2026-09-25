"""Builders for the kinds of fit whose capability rows RM23 checks against their calls.

A capability row reads ``available`` before any call. RM23 found rows that read
``available`` while their call then refused, and each such row was a kind of fit that
nobody had asked the row about. :data:`KINDS` names one builder per kind, so every test
that asks a row about a kind asks it about the same fit.

Each builder fits in sample with the explicit linear learners of
:func:`tests.conftest.linear_in_sample`, unless its docstring says otherwise.
:meth:`Kind.build` fits each kind once per process and returns a fresh copy of the
result, so no test reads a facade or a cached report that another test filled.

:func:`problems` is the sweep's instrument, and :data:`MUTATIONS` holds the monkeypatched
defects it must see. Each mutation restores one pre-RM23 answer at the seam its fix added.
"""

from __future__ import annotations

import dataclasses
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
)
from cleverly.data import CausalData
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
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from cleverly.sensitivity import omitted_variable
from tests import discrete_law, discrete_law_mar, regimes
from tests.conftest import (
    IN_SAMPLE,
    SELECTOR_CONFIGS,
    linear_ctmle,
    linear_drtmle,
    linear_in_sample,
    mean_one_weights,
)
from tests.unit._declaration_support import legacy_result
from tests.unit._direct_effect_support import COVARIATES as CDE_COVARIATES
from tests.unit._direct_effect_support import cde_frame
from tests.unit._simulated_confounding_support import (
    _GRID,
    _fit_attributable,
    _fit_shift_policies,
)

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


def dose_frame() -> Any:
    """The continuous-dose law of ``make_shift_dose(300, 0)``."""
    frame, _ = make_shift_dose(n=300, seed=0)
    return frame


def outcome_bounds(frame: Any) -> tuple[float, float]:
    """A declared outcome support one unit wider than the observed range of ``frame``."""
    return (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)


def fit_shift(frame: Any = None, **overrides: Any) -> Any:
    """A capped shift of the continuous dose, in sample unless ``overrides`` say otherwise.

    ``frame`` defaults to :func:`dose_frame`.
    """
    frame = dose_frame() if frame is None else frame
    return (
        TMLE(**linear_in_sample(shifts=[Shift(0.5, cap=5.0)], **overrides))
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


def drtmle_companion_frame() -> Any:
    """The ``evaluation=`` companion of :func:`fit_drtmle`, an independent draw of its law."""
    frame, _ = make_binary_outcome(n=80, seed=12)
    return frame


def fit_drtmle(evaluation: Any = None) -> Any:
    """A two-fold DR-TMLE fit of a binary outcome, ``make_binary_outcome(160, 11)``.

    The outcome is binary, so the cross-fitted fit needs no declared ``q_bounds``.
    ``evaluation`` is the companion the estimator declares, or ``None``. Two folds of 160
    rows keep each refit of the sweep cheap, and every row answers as it does at three
    folds of 240.
    """
    frame, _ = make_binary_outcome(n=160, seed=11)
    estimator = DRTMLE(**linear_drtmle(n_folds=2, estimands=("ate",)), evaluation=evaluation)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_drtmle_companion() -> Any:
    """:func:`fit_drtmle` with the frame of :func:`drtmle_companion_frame` as its companion."""
    return fit_drtmle(drtmle_companion_frame())


def fit_policy_means() -> Any:
    """A study fit of two policy means of a continuous dose, one of them zero-delta.

    A kind fitted by an estimator directly records no identification, so
    ``simulated_confounding`` refuses it for the whole fit. A study records the
    identification, so this kind's ``simulated_confounding`` row resolves each request.
    The first reported mean is the zero-delta one, which the call refuses, so the
    estimand the sweep supplies is a refused value.
    """
    return _fit_shift_policies(means=True)


def fit_natural_course_study() -> Any:
    """A study fit of the complete-outcome natural-course mean, which reports ``ey_obs`` alone.

    The fit is arm-indexed, and it reports no arm-indexed mean and no linear contrast, so the
    ``bound_parameters`` rule is the first omitted-variable rule that refuses it. The
    ``natural_course`` kind has missing outcomes, so the response rule refuses it first.
    Given the estimand, its ``refute`` row defers on ``tests=`` as that kind's does. The bare
    ``simulated_confounding`` row reads unavailable, because no reported name runs. The
    builder is cached, so each call returns a copy with an empty report cache.
    """
    return dataclasses.replace(_fit_attributable("ey_obs", family="gaussian", strata=False))


# ------------------------------------------------------------------- restored results


def discrete_fit(**overrides: Any) -> Any:
    """A fit of the discrete law with the linear learners, in sample unless overridden."""
    estimator = TMLE(**linear_in_sample(**overrides))
    return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


def cross_fitted() -> Any:
    """A cross-fitted fit of the discrete law in two folds."""
    return discrete_fit(cross_fit=True, n_folds=2)


def reconfigured(result: Any, **configuration: Any) -> Any:
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
    return reconfigured(result, split_plan=SplitPlan(result.estimator.split_plan.assignments))


def ctmle_stratified() -> Any:
    """An in-sample greedy collaborative fit restored with a stratified fold policy."""
    estimator = linear_ctmle("greedy", **SELECTOR_CONFIGS["greedy"], estimands=("ate",))
    result = estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()
    return reconfigured(result, stratify_folds="treatment")


def unbounded_scale_ate(engine: type[TMLE] = TMLE, **settings: Any) -> Any:
    """A cross-fitted binary-treatment fit with declared bounds, restored without them.

    No release saved this shape, because it cross-fitted a discrete treatment under
    ``"none"``. A copied estimator can carry it. ``engine`` is :class:`TMLE` or
    :class:`DRTMLE`, and ``settings`` extend its linear learners. The fit reports ``ate``
    among its estimands, so the replay slots read it with that estimand.
    """
    frame = _linear_frame()
    bounds = outcome_bounds(frame)
    if engine is DRTMLE:
        estimator: TMLE = DRTMLE(**linear_drtmle(n_folds=2, q_bounds=bounds, **settings))
    else:
        estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, q_bounds=bounds, **settings))
    return reconfigured(estimator.fit(frame, outcome="Y", treatment="A").single(), q_bounds=None)


def unbounded_scale() -> Any:
    """A cross-fitted continuous-dose fit with declared bounds, restored without them.

    The shape of the RM33 artifact on its outcome scale. It keeps ``stratify_folds="none"``,
    so its refit meets the scale refusal. ``restored_stratified`` holds the fold policy.
    """
    frame = dose_frame()
    result = fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame))
    return reconfigured(result, q_bounds=None)


def restored_ctmle_clustered() -> Any:
    """An in-sample outcome-adaptive collaborative fit of clustered data, saved and loaded.

    Release 0.1.1 fitted C-TMLE on clustered data, and this version refuses that design in
    the collaborative override of the refit preflight alone. The outcome-adaptive strategy
    meets no fold-policy refusal first. The fit adjusts for ``W1`` and ``W2``, and the
    restored result carries the same rows with ``cluster`` as their id, as the data of such
    a saved result does.
    """
    frame, _ = make_clustered(n=400, cluster_size=10, seed=7)
    estimator = linear_ctmle("oat", estimands=("ate",))
    fitted = estimator.fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2")).single()
    clustered = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=("W1", "W2"), id="cluster"
    )
    return loads(dumps(dataclasses.replace(fitted, data=clustered)))


def restored_stratified() -> Any:
    """The two-fold cross-fitted fit restored under ``stratify_folds="treatment"``."""
    return reconfigured(cross_fitted(), stratify_folds="treatment")


def v011_artifact(result: Any) -> Any:
    """``result`` in the shape of a 0.1.1 artifact: its default policy, and no ``split_plan``.

    Release 0.1.1 wrote ``stratify_folds="treatment"`` on every estimator, and it wrote no
    ``split_plan``. :func:`as_saved_by_v011` drops the plan alone.
    """
    return as_saved_by_v011(reconfigured(result, stratify_folds="treatment"))


def restored_v011() -> Any:
    """The shape of the real 0.1.1 artifact: two stratified folds and no ``split_plan``.

    Release 0.1.1 accepted ``stratify_folds="treatment"`` and wrote no ``split_plan``, so
    this result needs the class default to be read at all, and then its refit is refused.
    """
    return v011_artifact(cross_fitted())


# ----------------------------------------------------------------------- longitudinal


def fit_ltmle() -> Any:
    """The in-sample two-node regime fit of ``make_longitudinal(400, 12)``."""
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


# ------------------------------------------------------------------------- the kinds


@functools.cache
def _fitted(fit: Callable[[], Any]) -> Any:
    """``fit()``, run once per process. Only :meth:`Kind.build` reads it, and it copies."""
    return fit()


@dataclass(frozen=True)
class Kind:
    """One kind of fit the sweep asks every row about.

    Parameters
    ----------
    fit : callable
        Fits a result of this kind. :meth:`build` calls it once per process.
    must_run : frozenset of str
        Rows whose operation must answer in the sweep's report. This is the nonzero
        witness: a row refused by mistake passes the no-decline check and fails here.
        Each kind names every row it answered when the snapshot was measured.
    refits : bool
        Whether ``refit()`` runs on the result's own data, so ``refit_nuisances`` reads
        true. Every fit in this version refits, and a restored kind is one it refuses.
    must_not_run : frozenset of str
        Rows whose operation must not answer. A restored kind whose status supplies no
        inference names the E-value row here, so the sweep sees a status rule that stops
        applying. ``must_run`` alone would pass when an extra row answers.
    """

    fit: Callable[[], Any]
    must_run: frozenset[str]
    refits: bool = True
    must_not_run: frozenset[str] = frozenset()

    def build(self) -> Any:
        """A fresh copy of this kind's result, which shares the fit and nothing it reported.

        :func:`dataclasses.replace` gives the copy an empty ``assessment_cache`` and no
        memoized facade, so no report or row computed on another copy can answer for it.
        The estimator, the data and the nuisances are shared, and no call the tests make
        writes to them. The fit runs on the first call, so a test that builds before it
        applies a mutation fits without the mutation.
        """
        return dataclasses.replace(_fitted(self.fit))


def _kind(
    fit: Callable[[], Any], *must_run: str, refits: bool = True, must_not_run: tuple[str, ...] = ()
) -> Kind:
    return Kind(fit, frozenset(must_run), refits, frozenset(must_not_run))


#: The row a restored kind with a non-inferential status must not answer: the E-value
#: reads the interval that the status withholds.
_NO_INTERVAL = ("evalue",)


#: The rows that read a fit's own nuisances and scores.
_READ = ("nuisance_models", "score_equations", "support")
#: The two point rows that run a new computation: a refit and a truncation sweep.
_POINT = ("refute", "truncation_curve")
#: Every point row of a fit that admits the default truncation axis.
_LIVE = (*_READ, *_POINT)
#: The tilt rows, which a binary-treatment fit with missing outcomes answers.
_TILT = ("missingness", "tipping_gamma")
#: The bound rows that read the fit alone and answer without an interval. A fit that
#: reports no inference answers these rows and not the E-value, which needs its interval.
_PLUGIN_BOUND = ("contour", "elements", "omitted_confounding", "robustness_value")
#: The bound rows that read the fit alone. ``benchmark`` refits, so it is listed apart.
_BOUND = (*_PLUGIN_BOUND, "evalue")
#: Every row of an arm-indexed ATE fit with complete outcomes and two or more covariates.
_ARM = (*_LIVE, *_BOUND, "benchmark")

#: One kind of fit per name: live point fits, two study fits, restored results whose refit
#: this version refuses, and one longitudinal fit. Each kind's rows are the snapshot of the
#: rows it answered, so a row that starts to refuse by mistake fails its kind.
KINDS: dict[str, Kind] = {
    "ordinary": _kind(fit_ordinary, *_ARM),
    "binary": _kind(fit_binary, *_ARM),
    "missing": _kind(fit_missing, *_LIVE, *_TILT),
    "shift": _kind(fit_shift, *_LIVE),
    "cde": _kind(fit_cde, *_LIVE),
    "multi_arm": _kind(fit_multi_arm, *_ARM),
    "ctmle_oat": _kind(fit_ctmle_oat, *_LIVE),
    "msm": _kind(fit_msm, *_LIVE),
    "regime": _kind(fit_regime, *_LIVE),
    "weighted": _kind(fit_weighted, *_ARM),
    "stratified": _kind(fit_stratified, *_ARM),
    "clustered": _kind(fit_clustered, *_ARM),
    "drtmle": _kind(fit_drtmle, *_LIVE, "corrections"),
    "ctmle_ordered": _kind(fit_ctmle_ordered, *_LIVE),
    "shift+missing": _kind(fit_shift_missing, *_LIVE),
    "incremental": _kind(fit_incremental, *_READ, "refute"),
    "incremental+missing": _kind(fit_incremental_missing, *_LIVE),
    "regime+missing": _kind(fit_regime_missing, *_LIVE),
    "msm+missing": _kind(fit_msm_missing, *_LIVE),
    "rr+missing": _kind(fit_rr_missing, *_LIVE, "evalue"),
    # No treatment mechanism is fitted, so no support row answers.
    "natural_course": _kind(fit_natural_course, "nuisance_models", "score_equations", *_POINT),
    # One covariate, which benchmark cannot drop.
    "split_plan": _kind(fit_split_plan, *_LIVE, *_BOUND),
    "drtmle_companion": _kind(fit_drtmle_companion, *_LIVE, "corrections"),
    "policy_means": _kind(fit_policy_means, *_LIVE),
    "natural_course_study": _kind(fit_natural_course_study, *_LIVE),
    # A saved stratified split reports no interval (RM31), so the E-value row does not answer.
    "restored_stratified": _kind(
        restored_stratified,
        *_READ,
        "truncation_curve",
        *_PLUGIN_BOUND,
        refits=False,
        must_not_run=_NO_INTERVAL,
    ),
    "restored_v011": _kind(
        restored_v011,
        *_READ,
        "truncation_curve",
        *_PLUGIN_BOUND,
        refits=False,
        must_not_run=_NO_INTERVAL,
    ),
    # A saved undeclared scale reports no interval (RM33), and a shift fit answers no bound row.
    "restored_unbounded_scale": _kind(unbounded_scale, *_READ, "truncation_curve", refits=False),
    # The same status on an ATE fit, whose plug-in bound rows answer and whose E-value does not.
    "restored_unbounded_scale_ate": _kind(
        unbounded_scale_ate,
        *_READ,
        "truncation_curve",
        *_PLUGIN_BOUND,
        refits=False,
        must_not_run=_NO_INTERVAL,
    ),
    "restored_ctmle_clustered": _kind(
        restored_ctmle_clustered, *_READ, "truncation_curve", refits=False
    ),
    "ltmle": _kind(fit_ltmle, *_READ, "truncation_curve"),
}


# ------------------------------------------------------------------------- the sweep


def _fillers(result: Any) -> dict[str, Callable[[], Any]]:
    """The value the sweep supplies for each argument a row can defer on."""
    return {
        "estimand": lambda: next(iter(result.estimates)),
        "covariates": lambda: list(result.data.covariate_names[:1]),
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
                if name not in request:
                    request = {**request, name: fillers[name]()}
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


#: Each seam an RM23 fix added, as its owner and attribute name. Every seam is a function
#: but the omitted-variable rule table, which the rows and the calls read at call time.
SEAMS: tuple[tuple[Any, str], ...] = (
    (SensitivityFacade, "_tilt_rule"),
    (DiagnosticsFacade, "_truncation_gated"),
    (DiagnosticsFacade, "_refute_gated"),
    (SensitivityFacade, "_benchmark_gated"),
    (SensitivityFacade, "_simulated_confounding_gated"),
    (omitted_variable, "_FIT_WIDE_BOUND_RULES"),
    (CTMLE, "_configured_for_refit"),
    (DRTMLE, "_configured_for_refit"),
    (TMLE, "_refit_configuration_refusal"),
)

#: The class default that lets a result saved by release 0.1.1 be read. M7 removes it.
SPLIT_PLAN_DEFAULT = (TMLE, "split_plan")


def _passthrough(patch: pytest.MonkeyPatch) -> None:
    """Replace every seam with a wrapper that returns what the seam returns.

    The rule table is replaced by an equal copy, so a reader that holds the original
    object and a reader of the module attribute see the same rules.
    """
    for owner, name in SEAMS:
        original = vars(owner)[name]
        if callable(original):
            wrapper = functools.wraps(original)(lambda *a, _f=original, **k: _f(*a, **k))
            patch.setattr(owner, name, wrapper)
        else:
            patch.setattr(owner, name, tuple(original))
    patch.setattr(*SPLIT_PLAN_DEFAULT, None)


def _without_bound_parameters(patch: pytest.MonkeyPatch) -> None:
    """The omitted-variable rule table as it was, without its ``bound_parameters`` rule.

    The rows and the entry points read the one table, so both lose the rule.
    :func:`~cleverly.sensitivity.omitted_variable.resolve_parameter` still refuses the fit,
    and the call raises what the row no longer declares.
    """
    rules = tuple(
        (name, rule)
        for name, rule in omitted_variable._FIT_WIDE_BOUND_RULES
        if name != "bound_parameters"
    )
    patch.setattr(omitted_variable, "_FIT_WIDE_BOUND_RULES", rules)


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
        The kinds whose :func:`sweep` must report a problem under the mutation. Every
        other kind must report none.
    signature : str
        A regular expression that one problem of each kind in ``fails_on`` must match.
        It names the row and the refusal the mutated component lets through, so an
        unrelated failure, such as a ``TypeError`` from a stale seam, does not count as
        detection. M0 detects nothing, so its signature is empty.
    """

    describe: str
    apply: Callable[[pytest.MonkeyPatch], None]
    fails_on: frozenset[str]
    signature: str


#: The kinds whose tilt rows read ``available`` before RM23 while both tilt calls refused.
DECLINED_TILT_KINDS = ("shift+missing", "incremental+missing", "regime+missing")
DECLINED_TILT_KINDS += ("msm+missing", "rr+missing")

#: What the sweep reports for a row that reads available while its call refuses.
_RAISED = "{} reads available, and its call raised "

#: M0 to M7 of the RM23 plan, M8 for the benchmark row the sweep found, M9 for the
#: simulated-confounding row, and M10 for the five omitted-variable rows. M0 wraps every
#: seam and changes nothing, so a failure there is the wrapping and not a defect.
MUTATIONS: dict[str, Mutation] = {
    "M0": Mutation("every seam wrapped and unchanged", _passthrough, frozenset(), ""),
    "M1": Mutation(
        "the tilt rows ignore every rule the rows did not read before RM23",
        lambda patch: patch.setattr(SensitivityFacade, "_tilt_rule", _pre_rm23_tilt_rule),
        frozenset(DECLINED_TILT_KINDS),
        _RAISED.format("missingness") + "missingness_tilt ",
    ),
    "M2": Mutation(
        "the truncation row ignores the predicate",
        lambda patch: patch.setattr(DiagnosticsFacade, "_truncation_gated", _identity_gate),
        frozenset({"incremental", "incremental+missing"}),
        _RAISED.format("truncation_curve") + r"the propensity g is \*inside\* the estimand",
    ),
    "M3": Mutation(
        "the refute row ignores the predicate",
        lambda patch: patch.setattr(DiagnosticsFacade, "_refute_gated", _identity_gate),
        frozenset(
            {
                "split_plan",
                "natural_course",
                "natural_course_study",
                "shift",
                "msm",
                "regime",
                "shift+missing",
                "regime+missing",
                "msm+missing",
                "incremental",
                "policy_means",
                "incremental+missing",
            }
        ),
        # A supplied plan loses the row-set rule. Means and MSM coefficients lose
        # the fixed-null rule, including its natural-course placebo sentence.
        _RAISED.format("refute")
        + r"(refutation test\(s\) \['subset'\]|the placebo refutation|"
        + r"refutation test\(s\) \['placebo'\] require a fixed no-effect value)",
    ),
    "M4": Mutation(
        "CTMLE ignores a covariate the refit adds",
        lambda patch: patch.setattr(CTMLE, "_configured_for_refit", _unconfigured),
        frozenset({"ctmle_ordered"}),
        r"ValueError: ordering must cover every covariate; missing \['_noise_0'\]",
    ),
    "M5": Mutation(
        "DRTMLE keeps a companion that lacks a refit covariate",
        lambda patch: patch.setattr(DRTMLE, "_configured_for_refit", _unconfigured),
        frozenset({"drtmle_companion"}),
        r"DataError: covariate columns not found: \['_noise_0'\]",
    ),
    "M6": Mutation(
        "the refit slot ignores the refit preflight",
        lambda patch: patch.setattr(TMLE, "_refit_configuration_refusal", lambda self, data: None),
        frozenset(
            {
                "restored_stratified",
                "restored_v011",
                "restored_unbounded_scale",
                "restored_unbounded_scale_ate",
                "restored_ctmle_clustered",
            }
        ),
        # The fold policy, the outcome scale, and the collaborative clustered refusal.
        _RAISED.format("refute")
        + "(stratify_folds='treatment' balances|a cross-fitted fit of a continuous outcome"
        + "|C-TMLE has no clustered result)",
    ),
    "M7": Mutation(
        "no class default for split_plan",
        lambda patch: patch.delattr(*SPLIT_PLAN_DEFAULT),
        frozenset({"restored_v011"}),
        "reading the rows raised AttributeError: 'TMLE' object has no attribute 'split_plan'",
    ),
    "M8": Mutation(
        "the benchmark row ignores the predicate",
        lambda patch: patch.setattr(SensitivityFacade, "_benchmark_gated", _identity_gate),
        frozenset({"split_plan"}),
        _RAISED.format("benchmark") + "benchmark cannot drop every covariate",
    ),
    "M9": Mutation(
        "the simulated-confounding row ignores the predicate",
        lambda patch: patch.setattr(
            SensitivityFacade, "_simulated_confounding_gated", _identity_gate
        ),
        frozenset({"policy_means", "natural_course_study"}),
        _RAISED.format("simulated_confounding") + "(continuous )?simulated_confounding refuses",
    ),
    "M10": Mutation(
        "the omitted-variable rows ignore a fit that reports no parameter to bound",
        _without_bound_parameters,
        frozenset({"natural_course_study"}),
        _RAISED.format("omitted_confounding") + "the omitted-variable bound applies",
    ),
}
