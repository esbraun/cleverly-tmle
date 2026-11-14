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
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    IncrementalEffect,
    LongitudinalTreatment,
    ParametricGFormulaMethod,
    RegimeContrast,
    RegimeMean,
    SplitPlan,
    TimeToEvent,
)
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
    make_point_survival,
    make_shift_dose,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE, AdaptiveIPTW
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, Refusal
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
from tests.unit import _known_mechanism_support as known_mechanism
from tests.unit import _learned_rule_support as learned_rule_support
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


def _single_estimand(name: str, frame: Any) -> Any:
    return (
        TMLE(**linear_in_sample(estimands=(name,))).fit(frame, outcome="Y", treatment="A").single()
    )


def fit_att() -> Any:
    """A fit that reports the ATT alone (Fix-15: a bare marginal bound reads ``att``)."""
    return _single_estimand("att", _linear_frame())


def fit_atc() -> Any:
    """A fit that reports the ATC alone."""
    return _single_estimand("atc", _linear_frame())


def fit_rr() -> Any:
    """A binary-outcome fit that reports the risk ratio alone, ``make_binary_outcome(400, 17)``."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    return _single_estimand("rr", frame)


def fit_or() -> Any:
    """A binary-outcome fit that reports the odds ratio alone."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    return _single_estimand("or", frame)


def _survival_study_fit(method: Any, n: int = 600, *, mechanisms: bool = True) -> Any:
    from cleverly import RegimeContrast
    from tests.unit.test_survival_drtmle import sample_frame

    study = CausalStudy(
        sample_frame(n, 21, n_times=3),
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    effect = study.identify(RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"))
    settings: dict[str, Any] = {"outcome_learner": LogisticRegression(max_iter=1000)}
    if mechanisms:
        settings["treatment_learner"] = LogisticRegression(max_iter=1000)
        settings["censoring_learner"] = LogisticRegression(max_iter=1000)
    return effect.estimate(method=method, cross_fit=False, random_state=0, **settings)


def fit_survival_drtmle() -> Any:
    """A drift-corrected survival TMLE (``SurvivalDRTMLE``) through the study layer.

    Its truncation curve and simulated confounding refuse (X62), and the rows say so.
    """
    from cleverly.methods import DRTMLEMethod

    return _survival_study_fit(DRTMLEMethod())


def fit_survival_ctmle() -> Any:
    """A collaborative survival TMLE (``SurvivalCTMLE``) through the study layer."""
    from cleverly.methods import CollaborativeTMLEMethod

    return _survival_study_fit(CollaborativeTMLEMethod(selection_folds=2), n=300, mechanisms=False)


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
        TMLE(**linear_in_sample(policies=[Shift(0.5, cap=5.0)], **overrides))
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
    """A per-arm outcome-adaptive collaborative fit on ``make_instrument(500, 44)``.

    The per-arm design reports no inference (the declared revert of ``ctmle-oat-per-arm``),
    so its rows are the shared design's.
    """
    frame, _ = make_instrument(n=500, seed=44)
    estimator = linear_ctmle("oat", estimands=("ate",))
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_ctmle_oat_shared() -> Any:
    """The shared outcome-adaptive fit on the same frame, which reports no inference."""
    frame, _ = make_instrument(n=500, seed=44)
    estimator = linear_ctmle("oat", estimands=("ate",), oat_design="shared")
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_ctmle_oat_missing() -> Any:
    """A per-arm outcome-adaptive fit with missing outcomes, ``make_missing_outcome(400, 4)``.

    It reports points only, and the missingness tilt reads its full-covariate response
    model, so its rows are the shared design's and both tilt rows.
    """
    estimator = linear_ctmle("oat", estimands=("ate",))
    return estimator.fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta").single()


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


def fit_learned_rule() -> Any:
    """The fold-evaluated learned-rule fit of :mod:`tests.unit._learned_rule_support`.

    The fit cross-fits in five folds, the fold-evaluated report that the registered
    learned-rule studies measure.  The point rows of the omitted-variable bound run on the
    fit's fold rules, held fixed (X11 part (n)).  A refit relearns the rules, so
    ``benchmark`` reads unavailable, and the E-value and the missingness tilt do too (F27
    in ``docs/roadmap.md``).  ``refute`` asks for an explicit estimand, and this direct fit
    records no identification for ``simulated_confounding``, so both read unavailable here.
    The summarize rows and ``truncation_curve`` run.
    """
    return learned_rule_support.fit()


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
        TMLE(**linear_in_sample(policies=[Shift(0.5, cap=None)]))
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


def _iptw(**overrides: Any) -> AdaptiveIPTW:
    """An in-sample AdaptiveIPTW, whose reduced regression is the linear outcome learner."""
    return AdaptiveIPTW(
        **linear_in_sample(
            missingness_learner=LogisticRegression(max_iter=1000),
            **overrides,
        )
    )


def fit_iptw() -> Any:
    """An AdaptiveIPTW arm fit with complete outcomes and an estimated mechanism.

    It reports `estimated_mechanism_plugin`: the point estimates stand and the interval
    rows refuse by name. It fits no outcome regression, so the outcome-bound rows refuse.
    """
    return _iptw().fit(_linear_frame(), outcome="Y", treatment="A").single()


def fit_iptw_known() -> Any:
    """An AdaptiveIPTW arm fit with a declared mechanism (H2), which reports `influence_curve`."""
    frame = known_mechanism.point_frame()
    return (
        _iptw()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=frame[["p0", "p1"]].to_numpy(dtype=float),
        )
        .single()
    )


def fit_iptw_regime() -> Any:
    """A deterministic regime pair, a static and a rule, on the discrete law."""
    return (
        _iptw(interventions=regimes.interventions()[:2])
        .fit(discrete_law.frame(), outcome="Y", treatment="A", covariates=("W",))
        .single()
    )


def fit_iptw_stochastic() -> Any:
    """The three regimes of :mod:`tests.regimes`, one of them a known stochastic regime."""
    return (
        _iptw(interventions=regimes.interventions())
        .fit(discrete_law.frame(), outcome="Y", treatment="A", covariates=("W",))
        .single()
    )


def fit_iptw_natural_course() -> Any:
    """The missing-outcome natural-course mean, which fits no treatment mechanism."""
    return (
        _iptw(estimands=("ey_obs",))
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_iptw_missing() -> Any:
    """An AdaptiveIPTW composite fit with a missing outcome."""
    return _iptw().fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta").single()


def fit_iptw_missing_treatment() -> Any:
    """An AdaptiveIPTW composite fit with a missing treatment, fitted in sample."""
    return (
        _iptw()
        .fit(
            _with_missing_treatment(_linear_frame()),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3", "W4"],
            treatment_delta="DeltaA",
        )
        .single()
    )


def fit_iptw_missing_treatment_crossfit() -> Any:
    """An AdaptiveIPTW composite fit with a missing treatment, cross-fitted."""
    return (
        _iptw(cross_fit=True, n_folds=3)
        .fit(
            _with_missing_treatment(_linear_frame()),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3", "W4"],
            treatment_delta="DeltaA",
        )
        .single()
    )


def fit_iptw_regime_missing() -> Any:
    """A regime fit with missing outcomes, fitted in sample."""
    return (
        _iptw(interventions=regimes.interventions())
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


def fit_attributable_missing() -> Any:
    """PAR and PAF alone with missing binary outcomes: no arm mean for the tilt to re-mix."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("par", "paf")))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_joint_natural_course() -> Any:
    """The natural-course mean, one arm mean, PAR and PAF on one stack, with missing outcomes."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("ey_obs", "ey0", "par", "paf")))
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
    """A three-fold DR-TMLE fit of a binary outcome, ``make_binary_outcome(240, 11)``.

    The outcome is binary, so the cross-fitted fit needs no declared ``q_bounds``.
    ``evaluation`` is the companion the estimator declares, or ``None``. Three folds is the
    fewest the default foldwise reduced regressions accept.
    """
    frame, _ = make_binary_outcome(n=240, seed=11)
    estimator = DRTMLE(**linear_drtmle(n_folds=3, estimands=("ate",)), evaluation=evaluation)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_drtmle_companion() -> Any:
    """:func:`fit_drtmle` with the frame of :func:`drtmle_companion_frame` as its companion."""
    return fit_drtmle(drtmle_companion_frame())


def _missing_drtmle(frame: Any) -> Any:
    """A randomized in-sample missing-outcome DR-TMLE fit, which refits its own mechanism."""
    estimator = DRTMLE(
        **linear_in_sample(
            missingness_learner=LogisticRegression(max_iter=1000),
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        ),
        randomized=True,
        max_outer=10,
        estimands=("ate", "ey"),
    )
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], delta="Delta"
    ).single()


def fit_drtmle_missing() -> Any:
    """Missing-outcome DR-TMLE at two arms, on the law of :func:`fit_missing`."""
    return _missing_drtmle(_missing_frame())


def three_arm_missing_frame() -> Any:
    """A randomized three-arm trial with MAR outcomes, 450 rows."""
    rng = np.random.default_rng(23)
    n = 450
    w = rng.normal(size=(n, 3))
    code = rng.integers(0, 3, size=n)
    observed = rng.random(n) < 1.0 / (1.0 + np.exp(-(0.4 + 0.3 * code - 0.5 * w[:, 0])))
    y = 0.5 * code + w[:, 0] - 0.4 * w[:, 1] + rng.normal(scale=0.7, size=n)
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "W3": w[:, 2],
            "A": np.array(["low", "medium", "high"], dtype=object)[code],
            "Delta": observed.astype(float),
            "Y": np.where(observed, y, np.nan),
        }
    )


def fit_drtmle_missing_multi_arm() -> Any:
    """Missing-outcome DR-TMLE at three arms, with each arm's mechanism tilted alone."""
    return _missing_drtmle(three_arm_missing_frame())


def _with_missing_treatment(frame: Any, *, seed: int = 31) -> Any:
    """``frame`` with the treatment missing at random on about a fifth of its rows."""
    rng = np.random.default_rng(seed)
    w = frame["W1"].to_numpy(dtype=float)
    recorded = rng.random(len(frame)) < 1.0 / (1.0 + np.exp(-(1.5 + 0.4 * w)))
    labels = frame["A"].to_numpy()
    missing: Any = np.nan if labels.dtype.kind in "fiu" else None
    return frame.assign(
        A=np.where(recorded, labels, missing),
        DeltaA=recorded.astype(float),
        cluster=np.arange(len(frame)) // 4,
        weight=mean_one_weights(len(frame)),
    )


def _composite_drtmle(**settings: Any) -> DRTMLE:
    """An observational in-sample DR-TMLE, which the composite route fits."""
    return DRTMLE(
        **linear_in_sample(
            missingness_learner=LogisticRegression(max_iter=1000),
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        ),
        max_outer=10,
        **settings,
    )


def fit_drtmle_composite() -> Any:
    """An observational missing-outcome DR-TMLE at two arms: the composite route."""
    return (
        _composite_drtmle(estimands=("ate", "ey0", "ey1"))
        .fit(
            _missing_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
        )
        .single()
    )


def fit_missing_treatment() -> Any:
    """A composite TMLE at two arms with a missing treatment and complete outcomes."""
    frame = _with_missing_treatment(_linear_frame())
    return (
        TMLE(**linear_in_sample(missingness_learner=LogisticRegression(max_iter=1000)))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3", "W4"],
            treatment_delta="DeltaA",
        )
        .single()
    )


def fit_tmle_missing_treatment_weights() -> Any:
    """A weighted composite TMLE with a missing outcome and a missing treatment."""
    frame = _with_missing_treatment(_missing_frame())
    return (
        TMLE(**linear_in_sample(missingness_learner=LogisticRegression(max_iter=1000)))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
            treatment_delta="DeltaA",
            weights="weight",
        )
        .single()
    )


def fit_drtmle_missing_treatment_multi_arm() -> Any:
    """A composite DR-TMLE at three arms with a missing outcome and a missing treatment."""
    frame = _with_missing_treatment(three_arm_missing_frame())
    return (
        _composite_drtmle(estimands=("ate", "ey"))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
        .single()
    )


def fit_drtmle_missing_treatment_clustered() -> Any:
    """A clustered composite DR-TMLE at two arms with both indicators."""
    frame = _with_missing_treatment(_missing_frame())
    return (
        _composite_drtmle(estimands=("ate", "ey0", "ey1"))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3"],
            delta="Delta",
            treatment_delta="DeltaA",
            id="cluster",
        )
        .single()
    )


def _ctmle_selector(strategy: Any, **settings: Any) -> Any:
    """An in-sample collaborative fit of the ATE on ``make_instrument(250, 44)``."""
    frame, _ = make_instrument(n=250, seed=44)
    estimator = linear_ctmle(strategy, estimands=("ate",), **settings)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_ctmle_greedy() -> Any:
    """A greedy collaborative fit, which reports a point and no interval (F18)."""
    return _ctmle_selector("greedy", **SELECTOR_CONFIGS["greedy"])


def fit_ctmle_discrete() -> Any:
    """A discrete collaborative fit over two candidate adjustment sets."""
    return _ctmle_selector("discrete", **SELECTOR_CONFIGS["discrete"])


def fit_ctmle_lasso() -> Any:
    """A lasso collaborative fit on a three-step penalty path."""
    return _ctmle_selector("lasso", **SELECTOR_CONFIGS["lasso"])


def fit_ctmle_learners() -> Any:
    """A collaborative fit that selects among declared treatment learners."""
    return _ctmle_selector("learners", **SELECTOR_CONFIGS["learners"])


def fit_ctmle_derivative() -> Any:
    """C-TMLE0, the derivative strategy, which supplies an interval, on a two-step path."""
    return _ctmle_selector("derivative", lambdas=(0.01, 0.03))


def fit_ctmle_oat_stack() -> Any:
    """A per-arm outcome-adaptive fit of unpenalized GLMs, whose stacked curve supplies an interval.

    The outcome is binary, ``make_binary_outcome(400, 17)``, so the stack needs no declared
    ``q_bounds`` (X58).
    """
    from cleverly.learners import ParametricMechanism, ParametricOutcome

    frame, _ = make_binary_outcome(n=400, seed=17)
    estimator = CTMLE(
        strategy="oat",
        oat_design="per_arm",
        estimands=("ate",),
        **linear_in_sample(
            outcome_learner=ParametricOutcome(), treatment_learner=ParametricMechanism()
        ),
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_rr_tilt() -> Any:
    """Two risk-ratio tilts of a binary treatment, the natural course and ``delta = 0.5``.

    The fit reports each tilt's mean ``ey_rr_tilt`` and their contrast ``ate_rr_tilt``, on
    ``make_binary_outcome(400, 17)``.
    """
    from cleverly.interventions import RiskRatioTilt

    frame, _ = make_binary_outcome(n=400, seed=17)
    policies = [RiskRatioTilt(1.0), RiskRatioTilt(0.5)]
    return (
        TMLE(**linear_in_sample(policies=policies)).fit(frame, outcome="Y", treatment="A").single()
    )


def fit_learned_rule_known() -> Any:
    """The fold-evaluated learned rule beside the known regime ``Static(1)`` (X11 (c)).

    It reports the rule's value, ``ey_known_regime`` and their contrast ``ate_learned_rule``.
    """
    from cleverly.interventions import Static

    return learned_rule_support.fit(interventions=[Static(1)])


def fit_optimal_rule() -> Any:
    """The fold-evaluated value of the optimal rule, ``ey_optimal_rule`` (X11 (f), family P12)."""
    return learned_rule_support.fit(estimands=["ey_optimal_rule"])


def _modifier_frame(n: int = 400, seed: int = 5) -> Any:
    """A binary outcome whose effect moves with ``W1``, as ``test_curve_assessment`` draws it."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w[:, 0])))
    y = rng.binomial(1, 0.3 + 0.1 * w[:, 1] + a * (0.15 + 0.1 * w[:, 0]))
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})


def _conditional_contrast(scale: str = "difference") -> Any:
    """The linear projection of the conditional contrast on ``W1`` and ``W2``, identified."""
    from cleverly import ConditionalContrast, PointTreatment
    from cleverly.bases import Linear, Modifiers

    study = CausalStudy(
        _modifier_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    contrast = ConditionalContrast(
        Modifiers(["W1", "W2"]),
        scale=scale,  # type: ignore[arg-type]
        model=Linear(["W1", "W2"]),
    )
    return study.identify(contrast)


def _logistic_two_fold() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 0,
        "simultaneous": False,
    }


def fit_cate_projection() -> Any:
    """The stacked two-fold TMLE of the CATE projection (family P13)."""
    return _conditional_contrast().estimate(**_logistic_two_fold())


def fit_log_crr_projection() -> Any:
    """The stacked two-fold TMLE of the log conditional risk ratio projection (family P14)."""
    return _conditional_contrast("log_ratio").estimate(**_logistic_two_fold())


def fit_four_arm() -> Any:
    """An ATE fit at a randomized four-arm treatment, 600 rows (the ``arms4`` key)."""
    rng = np.random.default_rng(3)
    n = 600
    w = rng.normal(size=(n, 2))
    arm = rng.integers(0, 4, size=n)
    y = 0.3 * arm + w[:, 0] - 0.5 * w[:, 1] + rng.normal(scale=0.8, size=n)
    frame = pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": arm.astype(float), "Y": y})
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()


def fit_vector_policy() -> Any:
    """A joint policy at a vector treatment, a flag ``F`` and a continuous dose ``D``.

    The natural course and a half-unit dose shift, by the stacked classifier ratio, on 400
    rows with a binary outcome (the ``vector`` key, family P7).
    """
    from cleverly.interventions import JointPolicy

    rng = np.random.default_rng(9)
    n = 400
    w = rng.normal(size=n)
    flag = rng.binomial(1, 1 / (1 + np.exp(-0.4 * w)))
    dose = 1.0 + 0.5 * flag + 0.3 * w + rng.normal(scale=0.7, size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.3 + 0.4 * flag + 0.3 * dose + 0.5 * w))))
    frame = pd.DataFrame({"W": w, "F": flag.astype(float), "D": dose, "Y": y.astype(float)})
    policies = [
        JointPolicy("natural", components={}, policy_kind="known"),
        JointPolicy("dose up", components={"D": Shift(0.5, cap=None)}, policy_kind="known"),
    ]
    data = CausalData.from_frame(
        frame, outcome="Y", treatment=["F", "D"], covariates=["W"], continuous_treatment=["D"]
    )
    return TMLE(**linear_in_sample(policies=policies, ratio="classifier")).fit(data).single()


def fit_atmle() -> Any:
    """The adaptive TMLE of a trial ATE augmented by external controls and treated rows.

    A pooled-form fit with cross-fitting off on 300 rows of a trial and an external source,
    which ``test_atmle_workflow`` draws. Every bound row refuses the adaptive working-model
    fit by name.
    """
    from cleverly import (
        AdaptiveTMLEMethod,
        CrossFitting,
        ModelSpec,
        TrialATE,
        TrialWithExternalData,
    )

    rng = np.random.default_rng(0)
    n = 300
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, 0.3, n))
    y = 1.0 + a + w + 0.5 * (1 - s) * (1 + a) + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W": w, "S": s})
    method = AdaptiveTMLEMethod(
        form="pooled",
        models=ModelSpec(
            outcome_learner=LinearRegression(), treatment_learner=LogisticRegression()
        ),
        cross_fitting=CrossFitting(enabled=False, learner_folds=3),
    ).with_overrides(random_state=7)
    design = TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S")
    effect = CausalStudy(frame, design=design).identify(TrialATE(population="trial"))
    return effect.estimate(method=method)


def fit_few_clusters() -> Any:
    """``make_clustered(200, cluster_size=10, seed=7)``: 20 clusters, the ``t`` reference tier."""
    frame, _ = make_clustered(n=200, cluster_size=10, seed=7)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A", id="cluster").single()


def fit_fold_targeted_repeated() -> Any:
    """A fold-targeted, fold-evaluated two-fold fit repeated over two draws.

    The outcome is binary, ``make_binary_outcome(400, 17)``, so the cross-fitted fit needs
    no declared outcome support.
    """
    frame, _ = make_binary_outcome(n=400, seed=17)
    estimator = TMLE(
        **linear_in_sample(
            outcome_learner=LogisticRegression(max_iter=1000),
            cross_fit=True,
            n_folds=2,
            repeats=2,
            cv_evaluation=True,
            targeting_scheme="fold",
            estimands=("ate",),
        )
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


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


# ------------------------------------------------------------------ reconfigured results


def discrete_fit(**overrides: Any) -> Any:
    """A fit of the discrete law with the linear learners, in sample unless overridden."""
    estimator = TMLE(**linear_in_sample(**overrides))
    return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


def cross_fitted() -> Any:
    """A cross-fitted fit of the discrete law in two folds."""
    return discrete_fit(cross_fit=True, n_folds=2)


def reconfigured(result: Any, **configuration: Any) -> Any:
    """A copy of ``result`` whose estimator is given ``configuration`` after construction.

    ``__init__`` refuses each configuration the tests use, so only a copied or modified
    estimator can carry one.
    """
    old = loads(dumps(result))
    vars(old.estimator).update(configuration)
    return loads(dumps(old))


def without_provenance() -> Any:
    """A supplied-plan fit given a plan that records no generator."""
    result = fit_split_plan()
    return reconfigured(result, split_plan=SplitPlan(result.estimator.split_plan.assignments))


def ctmle_stratified() -> Any:
    """An in-sample greedy collaborative fit given a stratified fold policy."""
    estimator = linear_ctmle("greedy", **SELECTOR_CONFIGS["greedy"], estimands=("ate",))
    result = estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()
    return reconfigured(result, stratify_folds="treatment")


def unbounded_scale_ate(engine: type[TMLE] = TMLE, **settings: Any) -> Any:
    """A cross-fitted binary-treatment fit with declared bounds, given none after the fit.

    A copied estimator can carry this shape. ``engine`` is :class:`TMLE` or
    :class:`DRTMLE`, and ``settings`` extend its linear learners. The fit reports ``ate``
    among its estimands, so the replay slots read it with that estimand.
    """
    frame = _linear_frame()
    bounds = outcome_bounds(frame)
    if engine is DRTMLE:
        estimator: TMLE = DRTMLE(**linear_drtmle(n_folds=3, q_bounds=bounds, **settings))
    else:
        estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, q_bounds=bounds, **settings))
    return reconfigured(estimator.fit(frame, outcome="Y", treatment="A").single(), q_bounds=None)


def unbounded_scale() -> Any:
    """A cross-fitted continuous-dose fit with declared bounds, given none after the fit.

    It keeps ``stratify_folds="none"``, so its refit meets the scale refusal.
    ``reconfigured_stratified`` holds the fold policy.
    """
    frame = dose_frame()
    result = fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame))
    return reconfigured(result, q_bounds=None)


def reconfigured_ctmle_clustered() -> Any:
    """An in-sample outcome-adaptive collaborative fit, given clustered data after the fit.

    This version refuses C-TMLE on clustered data in the collaborative override of the
    refit preflight alone. The outcome-adaptive strategy meets no fold-policy refusal
    first. The fit adjusts for ``W1`` and ``W2``, and the copy carries the same rows with
    ``cluster`` as their id.
    """
    frame, _ = make_clustered(n=400, cluster_size=10, seed=7)
    estimator = linear_ctmle("oat", estimands=("ate",))
    fitted = estimator.fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2")).single()
    clustered = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=("W1", "W2"), id="cluster"
    )
    return loads(dumps(dataclasses.replace(fitted, data=clustered)))


def reconfigured_stratified() -> Any:
    """The two-fold cross-fitted fit given ``stratify_folds="treatment"`` after the fit."""
    return reconfigured(cross_fitted(), stratify_folds="treatment")


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


def _fit_point_survival(**settings: Any) -> Any:
    """A two-arm survival curve fit on ``make_point_survival(600, 3)``, in sample."""
    frame, _ = make_point_survival(n=600, seed=3, n_times=4)
    study = CausalStudy(
        frame,
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    return study.identify(RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0")).estimate(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=0),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=0),
        random_state=5,
        simultaneous=False,
        **IN_SAMPLE,
        **settings,
    )


def fit_hazard() -> Any:
    """The per-horizon hazard fit of the X14 survival curve."""
    return _fit_point_survival(survival_model="hazard")


def fit_hazard_one_step() -> Any:
    """The one-step whole-curve hazard fit."""
    return _fit_point_survival(survival_model="hazard", targeting="one_step")


def fit_isotonic() -> Any:
    """The iterated-mean fit with its curve projected."""
    return _fit_point_survival(isotonic=True, pseudo_learner=LinearRegression())


def fit_ipsi() -> Any:
    """The in-sample two-node odds-tilt fit of ``make_longitudinal(400, 12)``, by IPSI."""
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
    return study.identify(IncrementalEffect([Incremental(1.0), Incremental(2.0)])).estimate(
        method="ipsi",
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        learner_folds=2,
        random_state=5,
        simultaneous=False,
        **IN_SAMPLE,
    )


def _fit_baseline(method: str) -> Any:
    """The in-sample two-node regime fit of ``make_longitudinal(400, 12)`` by a baseline."""
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
        method=method, random_state=5, simultaneous=False
    )


def fit_gcomp() -> Any:
    """The two-node regime fit of ``make_longitudinal(400, 12)``, by g-computation."""
    return _fit_baseline("gcomp")


def fit_ipw() -> Any:
    """The two-node regime fit of ``make_longitudinal(400, 12)``, by IPW."""
    return _fit_baseline("ipw")


def fit_gformula() -> Any:
    """The two-node regime fit of ``make_longitudinal(400, 12)``, by the parametric g-formula."""
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
        method=ParametricGFormulaMethod(covariate_types={"L2": "normal"}),
        random_state=5,
        simultaneous=False,
    )


#: The roles of ``make_longitudinal``: two binary nodes, censored after each.
LONGITUDINAL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _longitudinal_settings(**overrides: Any) -> dict[str, Any]:
    """Logistic and linear learners, in sample (one fold), with no simultaneous band."""
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "learner_folds": 2,
        "random_state": 5,
        "simultaneous": False,
        "n_folds": 1,
        **overrides,
    }


def fit_sdr() -> Any:
    """The sequentially doubly robust estimator of two static regimens, two-fold cross-fitted.

    ``make_longitudinal(400, 12)``. The fit is the ``sdr`` engine and the longitudinal
    ``cross_fitted`` scheme.
    """
    from cleverly.longitudinal import SDR

    frame, _ = make_longitudinal(n=400, seed=12)
    settings = _longitudinal_settings(n_folds=2)
    return SDR({"always": 1, "never": 0}, reference="never", **settings).fit(
        frame, **LONGITUDINAL_COLUMNS
    )


def fit_learned_regimen() -> Any:
    """A learned two-node rule pair, ``LTMLE(LearnedRegimen())`` (family learned).

    The panel and settings of :mod:`tests.unit.test_longitudinal_learned_regimen`. Its
    truncation and refutation rows refuse with the F27 record, since each runs the fit again
    and relearns the rules.
    """
    from tests.unit import test_longitudinal_learned_regimen as learned_regimen

    return learned_regimen.fit()


#: The treated nodes of each regimen of :func:`fit_longitudinal_msm`.
_DURATION = {"always": 2.0, "never": 0.0}


def _duration_design(label: Any, horizon: int, rows: Any) -> Any:
    """An intercept and the regimen's treated nodes, the working model's design."""
    del horizon
    return np.column_stack([np.ones(len(rows)), np.full(len(rows), _DURATION[str(label)])])


def fit_longitudinal_msm() -> Any:
    """An LTMLE working model, an intercept and a duration, over always and never (family msm)."""
    from cleverly.longitudinal import LTMLE

    frame, _ = make_longitudinal(n=400, seed=12)
    msm = MSM(design=_duration_design, terms=("(intercept)", "duration"), design_kind="known")
    return LTMLE({"always": 1, "never": 0}, msm=msm, **_longitudinal_settings()).fit(
        frame, **LONGITUDINAL_COLUMNS
    )


def fit_competing() -> Any:
    """The cause-specific risks of relapse and death under always and never (family competing).

    ``make_longitudinal_competing(400, 5)``, two nodes, censored after each.
    """
    from cleverly.datasets import make_longitudinal_competing
    from cleverly.longitudinal import LTMLE

    frame, _ = make_longitudinal_competing(n=400, seed=5)
    columns = {**LONGITUDINAL_COLUMNS, "outcome": {"relapse": ("R1", "R2"), "death": ("D1", "D2")}}
    return LTMLE({"always": 1, "never": 0}, reference="never", **_longitudinal_settings()).fit(
        frame, **columns
    )


def fit_repeated() -> Any:
    """Always and never on a non-absorbing outcome measured at three nodes (family repeated).

    ``make_longitudinal_repeated(400, 7)``.
    """
    from cleverly.datasets import make_longitudinal_repeated
    from cleverly.longitudinal import LTMLE, Repeated

    frame, _ = make_longitudinal_repeated(n=400, seed=7, backend="pandas")
    return LTMLE({"always": 1, "never": 0}, reference="never", **_longitudinal_settings()).fit(
        frame,
        outcome=Repeated(["Y1", "Y2", "Y3"]),
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
    )


def _categorical_frame(levels: int, n: int = 400, seed: int = 8) -> Any:
    """Two randomized nodes of ``levels`` integer arms, a binary covariate between them.

    ``W`` is standard normal and ``Y`` binary. No row is censored.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.integers(0, levels, size=n)
    l2 = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * w + 0.2 * a1))))
    a2 = rng.integers(0, levels, size=n)
    logit = -0.5 + 0.2 * a1 + 0.2 * a2 + 0.3 * w + 0.3 * l2
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {"W": w, "A1": a1.astype(float), "L2": l2.astype(float), "A2": a2.astype(float), "Y": y}
    )


#: The roles of :func:`_categorical_frame`.
CATEGORICAL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}


def _tilted_density(rows: Any) -> Any:
    """A known stochastic node over three arms, the same on every row."""
    return np.tile([0.2, 0.3, 0.5], (len(rows), 1))


def fit_stochastic_categorical() -> Any:
    """A static plan and a known stochastic plan of a three-arm treatment at two nodes.

    The ``3`` treatment key and the ``stochastic`` regimen family, on
    :func:`_categorical_frame` at three arms.
    """
    from cleverly.interventions import Stochastic
    from cleverly.longitudinal import LTMLE, DynamicRegimen

    node = Stochastic(_tilted_density, "tilted", density_kind="known")
    regimens = {"low": 0.0, "tilted": DynamicRegimen("tilted", (node, node))}
    settings = _longitudinal_settings()
    settings.pop("censoring_learner")
    return LTMLE(regimens, reference="low", **settings).fit(
        _categorical_frame(3), **CATEGORICAL_COLUMNS
    )


def _minus_one(a: Any, h: Any) -> Any:
    """Each arm above the first moves down one arm."""
    values = np.asarray(a, dtype=float)
    return np.where(values >= 1, values - 1, values)


def fit_categorical_mtp() -> Any:
    """A modified treatment policy of a six-arm treatment at two nodes (the ``6`` key).

    The natural course and ``minus one`` on :func:`_categorical_frame` at six arms.
    """
    from cleverly.interventions import ModifiedPolicy
    from cleverly.longitudinal import LTMLE

    regimens = {
        "natural": Shift(0.0, cap=None),
        "minus one": ModifiedPolicy("minus one", apply=_minus_one, policy_kind="known"),
    }
    settings = _longitudinal_settings()
    settings.pop("censoring_learner")
    return LTMLE(regimens, reference="natural", **settings).fit(
        _categorical_frame(6), **CATEGORICAL_COLUMNS
    )


def fit_mixed_mtp() -> Any:
    """A joint policy at a mixed node, a flag and a continuous dose, at two nodes.

    The natural course and a half-unit dose shift at each node, by the classifier ratio, on
    400 rows (the ``mixed`` treatment key, family mtp).
    """
    from cleverly.interventions import JointPolicy
    from cleverly.longitudinal import LTMLE, DynamicRegimen

    rng = np.random.default_rng(10)
    n = 400
    w = rng.normal(size=n)
    f1 = rng.binomial(1, 1 / (1 + np.exp(-0.3 * w)))
    d1 = 1.0 + 0.5 * f1 + 0.3 * w + rng.normal(scale=0.7, size=n)
    l2 = rng.binomial(1, 1 / (1 + np.exp(-(0.2 * d1 - 0.2))))
    f2 = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * l2 - 0.1))))
    d2 = 1.0 + 0.5 * f2 + 0.2 * d1 + rng.normal(scale=0.7, size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 0.3 * d2 + 0.3 * w + 0.2 * f2))))
    frame = pd.DataFrame({"W": w, "F1": f1, "D1": d1, "L2": l2, "F2": f2, "D2": d2, "Y": y}).astype(
        float
    )

    def plan(label: str, components: Callable[[int], dict[str, Any]]) -> Any:
        nodes = tuple(
            JointPolicy(label, components=components(t), policy_kind="known") for t in (1, 2)
        )
        return DynamicRegimen(label, nodes)

    regimens = {
        "natural": plan("natural", lambda t: {}),
        "dose up": plan("dose up", lambda t: {f"D{t}": Shift(0.5, cap=None)}),
    }
    settings = _longitudinal_settings(ratio="classifier")
    settings.pop("censoring_learner")
    return LTMLE(regimens, reference="natural", **settings).fit(
        frame,
        outcome="Y",
        treatment=[["F1", "D1"], ["F2", "D2"]],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        continuous_treatment=["D1", "D2"],
    )


def fit_eliminated() -> Any:
    """The risk of relapse under always and never with death eliminated (``eliminated`` key).

    ``make_longitudinal_competing(400, 5)``. The E-value refuses a fit that eliminates a
    cause, because its identification also needs no unmeasured common cause of the
    prevented events and the outcome (F16).
    """
    from cleverly.datasets import make_longitudinal_competing
    from cleverly.longitudinal import LTMLE

    frame, _ = make_longitudinal_competing(n=400, seed=5)
    columns = {**LONGITUDINAL_COLUMNS, "outcome": {"relapse": ("R1", "R2"), "death": ("D1", "D2")}}
    return LTMLE({"always": 1, "never": 0}, reference="never", **_longitudinal_settings()).fit(
        frame, eliminate=["death"], **columns
    )


def fit_separable() -> Any:
    """A separable effect beside two static arms of a held treatment (family separable).

    300 rows drawn from the exact two-cause survival law of
    :mod:`tests.discrete_law_point_survival`, whose frame realises the law by equally weighted
    rows, fitted by the one-step hazard fit that ``Separable`` needs with logistic learners.
    """
    from cleverly.longitudinal import LTMLE, Separable
    from tests.discrete_law_point_survival import survival_law

    law = survival_law(causes=2)
    frame = law.frame().sample(n=300, random_state=3).reset_index(drop=True)
    regimens = {"a0": 0, "a1": 1, "sep": Separable(arms={"death": 0, "relapse": 1}, first="death")}
    settings = _longitudinal_settings(survival_model="hazard", targeting="one_step")
    settings.pop("pseudo_learner")
    return LTMLE(regimens, **settings).fit(frame, **law.fit_columns())


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
        true. Every fit in this version refits, and a reconfigured kind is one it refuses.
    """

    fit: Callable[[], Any]
    must_run: frozenset[str]
    refits: bool = True

    def build(self) -> Any:
        """A fresh copy of this kind's result, which shares the fit and nothing it reported.

        :func:`dataclasses.replace` gives the copy an empty ``assessment_cache`` and no
        memoized facade, so no report or row computed on another copy can answer for it.
        The estimator, the data and the nuisances are shared, and no call the tests make
        writes to them. The fit runs on the first call, so a test that builds before it
        applies a mutation fits without the mutation.
        """
        return dataclasses.replace(_fitted(self.fit))


def _kind(fit: Callable[[], Any], *must_run: str, refits: bool = True) -> Kind:
    return Kind(fit, frozenset(must_run), refits)


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

#: One kind of fit per name: live point fits, two study fits, reconfigured results whose
#: refit this version refuses, and one longitudinal fit. Each kind's rows are the snapshot of the
#: rows it answered, so a row that starts to refuse by mistake fails its kind.
KINDS: dict[str, Kind] = {
    "ordinary": _kind(fit_ordinary, "marginal_bounds", *_ARM),
    # AdaptiveIPTW fits no outcome regression: the bound, tilt, refit and truncation rows
    # refuse it by name, and the rows that read the treatment mechanism run. The E-value
    # runs on the fit whose mechanism is declared, because it needs an interval.
    "iptw": _kind(fit_iptw, *_READ),
    "iptw_known": _kind(fit_iptw_known, *_READ, "evalue"),
    "iptw_regime": _kind(fit_iptw_regime, *_READ),
    "iptw_stochastic": _kind(fit_iptw_stochastic, *_READ),
    "iptw_natural_course": _kind(fit_iptw_natural_course, "nuisance_models", "score_equations"),
    "iptw_missing": _kind(fit_iptw_missing, *_READ),
    "iptw_missing_treatment": _kind(fit_iptw_missing_treatment, *_READ),
    "iptw_missing_treatment_crossfit": _kind(fit_iptw_missing_treatment_crossfit, *_READ),
    "iptw_regime_missing": _kind(fit_iptw_regime_missing, *_READ),
    "binary": _kind(fit_binary, "marginal_bounds", *_ARM),
    "missing": _kind(fit_missing, "marginal_bounds", *_LIVE, *_TILT),
    "shift": _kind(fit_shift, *_LIVE),
    "cde": _kind(fit_cde, *_LIVE),
    "multi_arm": _kind(fit_multi_arm, "marginal_bounds", *_ARM),
    "ctmle_oat": _kind(fit_ctmle_oat, *_LIVE),
    "ctmle_oat_shared": _kind(fit_ctmle_oat_shared, "marginal_bounds", *_LIVE),
    "ctmle_oat+missing": _kind(fit_ctmle_oat_missing, *_LIVE, *_TILT),
    "msm": _kind(fit_msm, *_LIVE),
    "regime": _kind(fit_regime, *_LIVE),
    # The point bound holds the fold rules fixed (X11 part (n)). A refit relearns the
    # rules, so the benchmark row refuses; the E-value reads no regime contrast.
    "learned_rule": _kind(fit_learned_rule, *_READ, "truncation_curve", *_PLUGIN_BOUND),
    "weighted": _kind(fit_weighted, "marginal_bounds", *_ARM),
    "stratified": _kind(fit_stratified, "marginal_bounds", *_ARM),
    "clustered": _kind(fit_clustered, "marginal_bounds", *_ARM),
    "drtmle": _kind(fit_drtmle, "marginal_bounds", *_LIVE, "corrections"),
    # Both refit the treatment mechanism on their own rows, because randomized=True.
    "drtmle+missing": _kind(fit_drtmle_missing, "marginal_bounds", *_LIVE, *_TILT, "corrections"),
    "drtmle+missing+multi_arm": _kind(
        fit_drtmle_missing_multi_arm, "marginal_bounds", *_LIVE, *_TILT, "corrections"
    ),
    # The composite construction: an observational delta= and a declared missing
    # treatment. The omitted-variable rows refuse a missing treatment by name.
    "drtmle+composite": _kind(
        fit_drtmle_composite, "marginal_bounds", *_LIVE, *_TILT, "corrections"
    ),
    "missing_treatment": _kind(fit_missing_treatment, *_LIVE, "evalue"),
    "tmle+missing_treatment+weights": _kind(fit_tmle_missing_treatment_weights, *_LIVE, *_TILT),
    "drtmle+missing_treatment+multi_arm": _kind(
        fit_drtmle_missing_treatment_multi_arm, *_LIVE, *_TILT, "corrections"
    ),
    "drtmle+missing_treatment+id": _kind(
        fit_drtmle_missing_treatment_clustered, *_LIVE, *_TILT, "corrections"
    ),
    "ctmle_ordered": _kind(fit_ctmle_ordered, "marginal_bounds", *_LIVE),
    "shift+missing": _kind(fit_shift_missing, *_LIVE),
    "incremental": _kind(fit_incremental, *_READ, "refute"),
    "incremental+missing": _kind(fit_incremental_missing, *_LIVE),
    # Three regime parameters, so a bare tipping_gamma() defers on the estimand.
    "regime+missing": _kind(fit_regime_missing, *_LIVE, "missingness"),
    # Two coefficients, so a bare tipping_gamma() defers on the estimand.
    "msm+missing": _kind(fit_msm_missing, *_LIVE, "missingness"),
    "rr+missing": _kind(fit_rr_missing, "marginal_bounds", *_LIVE, *_TILT, "evalue"),
    # PAR and PAF, so a bare tipping_gamma() defers on the estimand.
    "par+missing": _kind(fit_attributable_missing, "marginal_bounds", *_LIVE, "missingness"),
    # Four tiltable parameters, so a bare tipping_gamma() defers on the estimand.
    "joint+missing": _kind(fit_joint_natural_course, "marginal_bounds", *_LIVE, "missingness"),
    # No treatment mechanism is fitted, so no support row answers.
    "natural_course": _kind(
        fit_natural_course, "nuisance_models", "score_equations", *_POINT, *_TILT
    ),
    # One covariate, which benchmark cannot drop.
    "split_plan": _kind(fit_split_plan, "marginal_bounds", *_LIVE, *_BOUND),
    "drtmle_companion": _kind(fit_drtmle_companion, "marginal_bounds", *_LIVE, "corrections"),
    "policy_means": _kind(fit_policy_means, *_LIVE),
    "natural_course_study": _kind(fit_natural_course_study, "marginal_bounds", *_LIVE),
    # Each reconfigured estimator keeps its cached nuisances, so every row but a refit runs.
    "reconfigured_stratified": _kind(
        reconfigured_stratified,
        "marginal_bounds",
        *_READ,
        "truncation_curve",
        *_BOUND,
        refits=False,
    ),
    # A shift fit answers no bound row.
    "reconfigured_unbounded_scale": _kind(
        unbounded_scale, *_READ, "truncation_curve", refits=False
    ),
    "reconfigured_unbounded_scale_ate": _kind(
        unbounded_scale_ate, "marginal_bounds", *_READ, "truncation_curve", *_BOUND, refits=False
    ),
    "reconfigured_ctmle_clustered": _kind(
        reconfigured_ctmle_clustered, *_READ, "truncation_curve", refits=False
    ),
    "ltmle": _kind(fit_ltmle, *_READ, *_POINT),
    "ltmle_hazard": _kind(fit_hazard, *_READ, *_POINT),
    "ltmle_hazard_one_step": _kind(fit_hazard_one_step, *_READ, *_POINT),
    "ltmle_isotonic": _kind(fit_isotonic, *_READ, *_POINT),
    # IPSI averages the influence function and solves no score equation.
    "ipsi": _kind(fit_ipsi, "nuisance_models", "support", *_POINT),
    # The baselines fit no fluctuation, so no score equation. A g-computation estimate reads
    # no treatment mechanism, so its bounds move nothing and no truncation curve exists.
    "gcomp": _kind(fit_gcomp, "nuisance_models", "support", "refute"),
    "ipw": _kind(fit_ipw, "nuisance_models", "support", *_POINT),
    # The parametric g-formula divides by no mechanism, so it answers the rows g-computation does.
    "gformula": _kind(fit_gformula, "nuisance_models", "support", "refute"),
    # A direct fit records no identification, so simulated confounding reads unavailable.
    "att": _kind(fit_att, *_LIVE, *_BOUND, "benchmark", "marginal_bounds"),
    "atc": _kind(fit_atc, *_LIVE, *_BOUND, "benchmark", "marginal_bounds"),
    # The omitted-variable bound covers linear functionals; a ratio refuses it.
    "rr": _kind(fit_rr, *_LIVE, "evalue", "marginal_bounds"),
    "or": _kind(fit_or, *_LIVE, "evalue", "marginal_bounds"),
    # No truncation curve and no replay (X62), so only the reading rows and refute run.
    "survival_drtmle": _kind(fit_survival_drtmle, *_READ, "refute"),
    "survival_ctmle": _kind(fit_survival_ctmle, *_READ, "refute"),
    # Fix-15: the selector paths report a point and no interval (F18), so the bound rows
    # and the E-value refuse them; C-TMLE0 and the stacked per-arm curve supply one.
    "ctmle_greedy": _kind(fit_ctmle_greedy, "marginal_bounds", *_LIVE),
    "ctmle_discrete": _kind(fit_ctmle_discrete, "marginal_bounds", *_LIVE),
    "ctmle_lasso": _kind(fit_ctmle_lasso, "marginal_bounds", *_LIVE),
    "ctmle_learners": _kind(fit_ctmle_learners, "marginal_bounds", *_LIVE),
    "ctmle_derivative": _kind(fit_ctmle_derivative, "marginal_bounds", *_ARM),
    # The stack's interval reads unpenalized GLMs; the omitted-variable rows refuse it.
    "ctmle_oat_stack": _kind(fit_ctmle_oat_stack, *_LIVE, "evalue"),
    # A tilt reports its own estimands, which no bound row covers.
    "rr_tilt": _kind(fit_rr_tilt, *_LIVE),
    # The point bound holds the fold rules fixed; the benchmark refit relearns them.
    "learned_rule_known": _kind(fit_learned_rule_known, *_LIVE, *_PLUGIN_BOUND),
    "optimal_rule": _kind(fit_optimal_rule, *_LIVE),
    # A projection coefficient has no refutation or sensitivity derivation (R9, P1).
    "cate_projection": _kind(fit_cate_projection, *_READ, "truncation_curve"),
    "log_crr_projection": _kind(fit_log_crr_projection, *_READ, "truncation_curve"),
    "four_arm": _kind(fit_four_arm, "marginal_bounds", *_ARM),
    "few_clusters": _kind(fit_few_clusters, "marginal_bounds", *_ARM),
    # The fold-evaluated report averages over folds, so the E-value reads no ratio interval.
    "fold_targeted_repeated": _kind(
        fit_fold_targeted_repeated, "marginal_bounds", *_LIVE, *_PLUGIN_BOUND, "benchmark"
    ),
    "vector_policy": _kind(fit_vector_policy, *_LIVE),
    # Every bound and refutation row refuses the adaptive working-model fit by name.
    "atmle": _kind(fit_atmle, "corrections", "nuisance_models", "score_equations", "support"),
    # SDR fits no fluctuation, so it solves no score equation.
    "sdr": _kind(fit_sdr, "nuisance_models", "support", *_POINT),
    "longitudinal_msm": _kind(fit_longitudinal_msm, *_READ, *_POINT),
    "competing": _kind(fit_competing, *_READ, *_POINT),
    "repeated": _kind(fit_repeated, *_READ, *_POINT),
    "stochastic_categorical": _kind(fit_stochastic_categorical, *_READ, *_POINT),
    "categorical_mtp": _kind(fit_categorical_mtp, *_READ, *_POINT),
    "mixed_mtp": _kind(fit_mixed_mtp, *_READ, *_POINT),
    "eliminated": _kind(fit_eliminated, *_READ, *_POINT),
    "separable": _kind(fit_separable, *_READ, *_POINT),
    "learned_regimen": _kind(fit_learned_regimen, *_READ),
}


# ------------------------------------------------------------------------- the sweep


def _fillers(result: Any) -> dict[str, Callable[[], Any]]:
    """The value the sweep supplies for each argument a row can defer on."""
    return {
        "estimand": lambda: next(iter(result.estimates)),
        "covariates": lambda: list(omitted_variable._benchmark_covariates(result.data)[:1]),
        "grid": lambda: _GRID,
        "bounds": lambda: (0.05,),
        "mechanism": lambda: True,
        "tests": lambda: ("random_common_cause",),
        # The elements describe one draw, so a fit with several draws names the first.
        "repeat": lambda: 0,
    }


def _facades(result: Any) -> tuple[Any, Any]:
    return result.diagnostics, result.sensitivity


def _estimand_choice(result: Any, facade: Any, operation: str) -> str:
    """The estimand the sweep supplies: the deferred row's own first candidate.

    A row that defers on its estimand names the candidates it chooses between, and the
    first reported estimate need not be one of them: a PAR fit reports ``par``, which the
    marginal bound cannot take, so it offers every arm parameter instead (Fix-15).
    """
    candidates = facade._estimand_candidates(operation)
    if candidates:
        return str(candidates[0])
    return str(next(iter(result.estimates)))


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
                    value = (
                        _estimand_choice(result, facade, declared.operation)
                        if name == "estimand"
                        else fillers[name]()
                    )
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
    is a problem too.
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

#: The tilt rules the rows read before RM23: the longitudinal literal and the
#: missing-outcome flag. Every later rule was the call's alone. The natural-course rule
#: the rows also read then no longer exists, because the tilt covers that mean now.
_PRE_RM23_TILT_RULES = ("longitudinal", "missing_outcome")


def _pre_rm23_tilt_rule(self: SensitivityFacade) -> tuple[str, Refusal] | None:
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


#: The kinds whose tilt rows read ``available`` before RM23 while both tilt calls refuse.
#: The regime, MSM, ratio and attributable kinds left this list when X82 part G gave
#: them a tilt.
DECLINED_TILT_KINDS = ("shift+missing", "incremental+missing")

#: What the sweep reports for a row that reads available while its call refuses.
_RAISED = "{} reads available, and its call raised "

#: M0 to M6 of the RM23 plan, M8 for the benchmark row the sweep found, M9 for the
#: simulated-confounding row, and M10 for the five omitted-variable rows. M0 wraps every
#: seam and changes nothing, so a failure there is the wrapping and not a defect.
MUTATIONS: dict[str, Mutation] = {
    "M0": Mutation("every seam wrapped and unchanged", _passthrough, frozenset(), ""),
    "M1": Mutation(
        "the tilt rows ignore every rule the rows did not read before RM23",
        lambda patch: patch.setattr(SensitivityFacade, "_tilt_rule", _pre_rm23_tilt_rule),
        # The learned-rule kind came after RM23. Its tilt rule is one the pre-RM23 rows did
        # not read, so M1 makes its tilt rows read available while both calls refuse.
        frozenset((*DECLINED_TILT_KINDS, "learned_rule", "learned_rule_known", "optimal_rule")),
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
                "natural_course",
                # The default refute request on a joint fit reads ey_obs, whose placebo has
                # no fixed null, so the row defers and the identity gate would not.
                "joint+missing",
                "natural_course_study",
                "shift",
                "regime",
                "shift+missing",
                "regime+missing",
                "incremental",
                "policy_means",
                "incremental+missing",
                # A longitudinal regimen fit reports its regimen means first, and a mean
                # has no fixed null (X82 part C).
                "ltmle",
                "gcomp",
                "ipw",
                "gformula",
                # Fix-15: a tilt, policy or longitudinal regimen mean reported first.
                "rr_tilt",
                "vector_policy",
                "sdr",
                "competing",
                "repeated",
                "stochastic_categorical",
                "categorical_mtp",
                "mixed_mtp",
                "eliminated",
                "separable",
            }
        ),
        # Means lose the fixed-null rule, including its natural-course placebo sentence. A
        # supplied plan no longer meets a default refusal, because ``subset`` keeps the
        # original fold labels, and an MSM coefficient has a placebo reference (X82 A2).
        _RAISED.format("refute")
        + r"(the placebo refutation|"
        + r"refutation test\(s\) \['placebo'\] require a fixed no-effect value|"
        # A mixed node's placebo meets its missing Markov kernel first (X82).
        + r"the longitudinal placebo has no Markov kernel)",
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
                "reconfigured_stratified",
                "reconfigured_unbounded_scale",
                "reconfigured_unbounded_scale_ate",
                "reconfigured_ctmle_clustered",
            }
        ),
        # The fold policy, the outcome scale, and the collaborative clustered refusal.
        _RAISED.format("refute")
        + "(stratify_folds='treatment' requests stratification|a cross-fitted fit of a continuous outcome"
        + "|C-TMLE has no clustered result)",
    ),
    "M8": Mutation(
        "the benchmark row ignores the predicate",
        lambda patch: patch.setattr(SensitivityFacade, "_benchmark_gated", _identity_gate),
        # A regime fit reports several parameters, and the benchmark refits it (X82 G2c).
        # A learned-rule fit runs the bound and refuses the benchmark refit (X11 part (n)).
        frozenset({"split_plan", "regime", "regime+missing", "learned_rule", "learned_rule_known"}),
        "("
        + _RAISED.format("benchmark")
        + "benchmark cannot drop every covariate"
        + "|benchmark declined: .*on a learned-rule fit the refit relearns the rule)",
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
        # Part G2a moved these kinds past the earlier rules, so this rule alone refuses them.
        frozenset({"natural_course_study", "natural_course", "rr+missing", "par+missing"}),
        _RAISED.format("omitted_confounding") + "the omitted-variable bound applies",
    ),
}
