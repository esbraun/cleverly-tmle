"""Every plot on one small fit of every result family: it draws, or it refuses with a reason.

A plot that meets a result it does not draw must say so: a ``CapabilityError`` (with the
roadmap item that tracks a missing view), or a ``ValueError`` or ``TypeError`` naming what it
takes. An ``AttributeError``, ``KeyError`` or ``IndexError`` from inside a plot is a crash, and
fails here with the cell that raised it.

``EXPECTED`` pins each cell: ``"draws"``, ``"refuses"`` (a refusal with no roadmap item, such
as a fit with no bootstrap replicates) or the roadmap id the refusal names. A cell that moves
from drawing to refusing, or to another item, fails.

To add a result family, add a fitting function to ``FAMILIES`` and one entry per plot to
``EXPECTED``. To add a plot, add it to ``PLOTS`` (or to ``OTHER_INPUTS`` when it takes an object
that is not a fitted result) and pin it for every family. The meta tests fail when
``cleverly.plotting.__all__`` or ``FAMILIES`` outgrows the matrix.
"""

from __future__ import annotations

import importlib
import inspect
import linecache
import pkgutil
import re
import traceback
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
import cleverly.plotting as plotting
from cleverly import (
    ATE,
    AdaptiveTMLEMethod,
    CausalStudy,
    CollaborativeTMLEMethod,
    ConditionalContrast,
    DRTMLEMethod,
    ModelSpec,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
    TimeToEvent,
    TrialATE,
    TrialWithExternalData,
)
from cleverly.bases import Linear, Modifiers
from cleverly.datasets import (
    make_linear_ate,
    make_longitudinal,
    make_longitudinal_survival,
    make_point_survival,
)
from cleverly.estimators.dose_response import DoseResponse, DoseResponseTMLE
from cleverly.exceptions import CapabilityError
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, StructuralNestedMean
from tests.unit import _capability_sweep_support as sweep
from tests.unit import _mtp_route_kinds as mtp_route_kinds
from tests.unit.test_plotting_core import assert_short_title

#: The tests that share the module's fits run on one worker, which fits each family once.
_SHARED = pytest.mark.xdist_group("plotting_result_matrix")

#: Plots that take a fitted result.
PLOTS = (
    "forest",
    "curve",
    "influence",
    "overlap",
    "weights",
    "balance",
    "learners",
    "calibration",
    "repeats",
    "bootstrap",
)
#: Public names of ``cleverly.plotting`` that take another object, or are not plots.
OTHER_INPUTS = {
    "PALETTE": "the palette, not a plot",
    "notes": "reads a figure",
    "style": "a style context, not a plot",
    "sensitivity": "takes a sensitivity analysis or a truncation curve",
    "confounding": "takes an omitted-variable bound or a benchmark",
    "targeting": "takes a targeting curve",
    "zipper": "takes a simulation study",
}


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _point_tmle() -> Any:
    frame, _ = make_linear_ate(n=300, seed=0)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            n_folds=3,
            repeats=2,
            q_bounds=(-10.0, 10.0),
            random_state=0,
        )
    )


def _ltmle_repeats() -> Any:
    frame, _ = make_longitudinal_survival(n=400, seed=0)
    return LTMLE(
        {"always": 1, "never": 0},
        n_folds=2,
        repeats=3,
        outcome_learner=LinearRegression(),
        treatment_learner=_logistic(),
        censoring_learner=_logistic(),
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _point_bootstrap() -> Any:
    frame, _ = make_linear_ate(n=200, seed=0)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            cross_fit=False,
            n_bootstrap=20,
            q_bounds=(-10.0, 10.0),
            random_state=0,
        )
    )


def _point_survival() -> tuple[pd.DataFrame, TimeToEvent]:
    frame, _ = make_point_survival(n=600, seed=4, n_times=4)
    design = TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
    )
    return frame, design


def _time_to_event() -> Any:
    frame, design = _point_survival()
    return CausalStudy(frame, design=design).estimate(
        RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=_logistic(),
        cross_fit=False,
    )


def _boundary_node() -> Any:
    """The hazard curve fit with no event in ``arm1`` at node 2, which adds the supplement."""
    frame, design = _point_survival()
    frame.loc[(frame["A"] == 1) & (frame["time"] == 2), "event"] = 0
    return CausalStudy(frame, design=design).estimate(
        RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
        outcome_learner=_logistic(),
        treatment_learner=_logistic(),
        survival_model="hazard",
        cross_fit=False,
    )


def _boundary_node_iterated() -> Any:
    """The clustered iterated-mean curve fit with no event in ``arm1`` at node 2."""
    frame, _ = _point_survival()
    frame.loc[(frame["A"] == 1) & (frame["time"] == 2), "event"] = 0
    frame["site"] = np.arange(len(frame)) // 10
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=("W1", "W2"),
        grid=(1, 2, 3, 4),
        cluster="site",
    )
    return CausalStudy(frame, design=design).estimate(
        RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
        outcome_learner=_logistic(),
        pseudo_learner=LinearRegression(),
        treatment_learner=_logistic(),
        cross_fit=False,
    )


def _boundary_node_bootstrap() -> Any:
    """The supplemented hazard curve fit with a full-refit bootstrap."""
    frame, design = _point_survival()
    frame.loc[(frame["A"] == 1) & (frame["time"] == 2), "event"] = 0
    return CausalStudy(frame, design=design).estimate(
        RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
        outcome_learner=_logistic(),
        treatment_learner=_logistic(),
        survival_model="hazard",
        cross_fit=False,
        n_bootstrap=19,
        random_state=0,
    )


def _survival_method(method: Any) -> Callable[[], Any]:
    def fit() -> Any:
        frame, design = _point_survival()
        return CausalStudy(frame, design=design).estimate(
            RegimeMean({"arm1": 1, "arm0": 0}, horizons=(2, 4)),
            method=method,
            outcome_learner=_logistic(),
            cross_fit=False,
        )

    return fit


def _continuous_survival() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    treated = rng.binomial(1, 1 / (1 + np.exp(-x)))
    relapse = np.exp(-1.5 + 0.5 * x - treated)
    death = np.exp(-2.0 - 0.5 * x - 0.5 * treated)
    event_time = rng.exponential(1 / (relapse + death))
    cause = np.where(rng.uniform(size=n) < relapse / (relapse + death), 1, 2)
    dropout = rng.exponential(1 / 0.15, size=n)
    frame = pd.DataFrame(
        {
            "time": np.minimum(event_time, dropout),
            "event": np.where(event_time <= dropout, cause, 0),
            "treated": treated,
            "x": x,
        }
    )
    design = TimeToEvent(
        time="time",
        event="event",
        treatment="treated",
        baseline=("x",),
        grid="continuous",
        causes={1: "relapse", 2: "death"},
    )
    return CausalStudy(frame, design=design).estimate(
        RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=(1, 2)),
        random_state=0,
    )


def _adaptive() -> Any:
    rng = np.random.default_rng(1)
    n = 500
    w = rng.normal(size=n)
    trial = rng.binomial(1, 0.4, size=n)
    treated = np.where(
        trial == 1, rng.binomial(1, 0.5, size=n), rng.binomial(1, 1 / (1 + np.exp(-w)), size=n)
    )
    y = w + 0.5 * treated + rng.normal(size=n)
    frame = pd.DataFrame({"score": y, "program": treated, "risk": w, "in_trial": trial})
    design = TrialWithExternalData(
        outcome="score", treatment="program", adjustment=("risk",), source="in_trial"
    )
    models = ModelSpec(outcome_learner=LinearRegression(), treatment_learner=_logistic())
    return (
        CausalStudy(frame, design=design)
        .identify(TrialATE(population="trial"))
        .estimate(method=AdaptiveTMLEMethod(models=models, runtime=Runtime(random_state=1)))
    )


def _effect_curve() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    w, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    treated = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * treated * (w > 0) + 0.05 * z)
    frame = pd.DataFrame({"Y": y, "A": treated, "W": w, "Z": z})
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z"))
    question = ConditionalContrast(Modifiers(["W"], domains={"W": (-1, 1)}), model=Linear(["W"]))
    fit = (
        CausalStudy(frame, design=design)
        .identify(question)
        .estimate(
            outcome_learner=_logistic(),
            treatment_learner=_logistic(),
            cross_fit=False,
            random_state=0,
        )
    )
    return fit.effect_curve()


def _ep_curve_log_ratio_learner() -> Any:
    """The EP log-ratio curve with a declared classifier as its final learner (P1 PR 2)."""
    from cleverly import EPMethod
    from cleverly.bases import Sieve

    rng = np.random.default_rng(0)
    n = 500
    w = rng.uniform(-1, 1, n)
    treated = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * treated * (w > 0))
    frame = pd.DataFrame({"Y": y, "A": treated, "W": w})
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    question = ConditionalContrast(Modifiers(["W"], domains={"W": (-1, 1)}), scale="log_ratio")
    method = EPMethod(sieve=Sieve(Linear(["W"]), size=1), final_learner=_logistic())
    return (
        CausalStudy(frame, design=design)
        .identify(question)
        .estimate(
            method.with_overrides(
                outcome_learner=_logistic(), treatment_learner=_logistic(), random_state=0
            )
        )
    )


def _dose_response() -> Any:
    rng = np.random.default_rng(0)
    n = 500
    w = rng.normal(size=n)
    dose = rng.normal(2.5 + 0.6 * w, 1.2)
    y = rng.binomial(1, expit(-1 + 0.5 * dose - 0.08 * dose * dose + 0.4 * w))
    return DoseResponseTMLE(
        DoseResponse.spanning((1.0, 4.0), 3, fractions=(0.25,)),
        outcome_learner=_logistic(),
        treatment_learner=_logistic(),
        random_state=0,
        n_folds=2,
    ).fit(pd.DataFrame({"Y": y, "A": dose, "W": w}), outcome="Y", treatment="A", covariates=["W"])


def _structural_nested_mean() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    return StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
    )


def _point_binary() -> Any:
    """Asymmetric intervals: range-logit arm means and a MOVER difference (Fix-3 PR 3)."""
    return sweep.fit_binary()


def _point_withheld() -> Any:
    """Withheld intervals: the treated arm shows one outcome value (constant_arm_cell_plugin)."""
    return sweep.fit_constant_arm_cell()


def _effect_curve_continuous() -> Any:
    """Skew-widened t rows: an effect curve of a continuous, right-skewed outcome."""
    rng = np.random.default_rng(0)
    n = 500
    w, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    treated = rng.binomial(1, 0.5, n)
    y = treated * (w > 0) + 0.2 * z + rng.lognormal(0.0, 1.0, n)
    frame = pd.DataFrame({"Y": y, "A": treated, "W": w, "Z": z})
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W", "Z"))
    question = ConditionalContrast(Modifiers(["W"], domains={"W": (-1, 1)}), model=Linear(["W"]))
    fit = (
        CausalStudy(frame, design=design)
        .identify(question)
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            cross_fit=False,
            random_state=0,
        )
    )
    return fit.effect_curve()


def _mtp_censored(estimator: str) -> Callable[[], Any]:
    """A continuous-treatment modified policy with dropout after each node (X77)."""

    def build() -> Any:
        from tests.unit import _capability_sweep_support as sweep

        return sweep._fit_mtp_censored(estimator)

    return build


#: One small fit per result family.
FAMILIES: dict[str, Callable[[], Any]] = {
    "point_tmle_repeats": _point_tmle,
    "point_tmle_bootstrap": _point_bootstrap,
    # The known-score ATT and ATC on a declared mechanism (X35): the same result class, whose
    # ATT and ATC carry a different plug-in and influence curve.
    "point_tmle_known_score": sweep.fit_known_score,
    # X54 part (h): a t-tier clustered fit with an adaptive outcome learner, whose key carries
    # adaptive_nuisance. The qualifier moves the support label alone, so every plot of a point
    # fit draws or refuses as it does without it.
    "point_tmle_adaptive_nuisance": sweep.fit_clustered_adaptive,
    # The same qualifier on a cross-fitted clustered LTMLE at 20 clusters (row G07).
    "ltmle_adaptive_nuisance": sweep.fit_clustered_ltmle_adaptive,
    "ltmle_repeats": _ltmle_repeats,
    "time_to_event_ltmle": _time_to_event,
    "time_to_event_boundary_node": _boundary_node,
    "time_to_event_boundary_node_iterated": _boundary_node_iterated,
    "time_to_event_boundary_node_bootstrap": _boundary_node_bootstrap,
    "time_to_event_ctmle": _survival_method(CollaborativeTMLEMethod()),
    "time_to_event_drtmle": _survival_method(DRTMLEMethod()),
    "continuous_survival": _continuous_survival,
    # X6 (c): restricted_horizons= adds restricted means, in time units, beside the risks.
    "continuous_survival_restricted": sweep.fit_continuous_survival_restricted,
    # X6 (d): the full-refit bootstrap, with the multiplier and the bootstrap band.
    "continuous_survival_bootstrap": sweep.fit_continuous_survival_bootstrap,
    "adaptive_tmle": _adaptive,
    "effect_curve": _effect_curve,
    "effect_curve_continuous": _effect_curve_continuous,
    "point_tmle_binary": _point_binary,
    "point_tmle_withheld": _point_withheld,
    "dose_response": _dose_response,
    "structural_nested_mean": _structural_nested_mean,
    # The P1 PR 1 compositions of the conditional-contrast projection, and the EP rule with a
    # declared sieve or final learner: the capability sweep's own fits.
    "projection_missing_treatment": sweep.fit_projection_missing_treatment,
    "projection_strata": sweep.fit_projection_strata,
    "projection_strata_log_ratio": sweep.fit_projection_strata_crr,
    "projection_repeats": sweep.fit_projection_repeated,
    "projection_bootstrap": sweep.fit_projection_bootstrap,
    "learned_rule_ep_sieve": sweep.fit_learned_rule_ep_sieve,
    "learned_rule_ep_learner": sweep.fit_learned_rule_ep_learner,
    # P1 PR 2: the projection of a controlled direct effect on both scales, and the EP
    # log-ratio curve with a declared classifier.
    "projection_intermediate": sweep.fit_projection_intermediate,
    "projection_intermediate_log_ratio": sweep.fit_projection_intermediate_crr,
    "ep_curve_log_ratio_learner": _ep_curve_log_ratio_learner,
    # A declared mechanism read in the weighted sample (Fix-16): an estimated tilt, so each
    # plot answers as it does on an estimated-mechanism fit of the same kind.
    "point_tmle_known_sample": sweep.fit_tmle_known_sample_weights,
    "ltmle_known_sample": sweep.fit_ltmle_known_sample_weights,
    "ltmle_mtp_censored": _mtp_censored("ltmle"),
    "sdr_mtp_censored": _mtp_censored("sdr"),
    # X70 (d) and (c): the hazard working model and the hazard modified treatment policy.
    "hazard_msm": sweep.fit_hazard_msm,
    "hazard_mtp": sweep.fit_hazard_mtp,
    # X70 (b1): the hazard path on a held design with covariates after baseline.
    "hazard_time_varying": sweep.fit_hazard_time_varying,
    # X70 (b2): the same backward pass with a treatment at several nodes.
    "hazard_time_varying_multinode": sweep.fit_hazard_time_varying_multinode,
    # X70 (a): the hazard path with a treatment at several nodes and baseline covariates.
    "hazard_multinode": sweep.fit_hazard_multinode,
    # A fit that estimates no treatment mechanism: no plot may promise its weights.
    "sequential_gcomp": sweep.fit_gcomp,
    # The modified-treatment-policy fits of each shape on each density-ratio route.
    **mtp_route_kinds.FITS,
    # X54 (c): the delete-one-cluster jackknife variance, which ships for LTMLE only (the
    # registered measurement refuses it for point TMLE). The interval plots read the jackknife
    # interval, and influence() says its curve is not the reported variance.
    "ltmle_jackknife": sweep.fit_ltmle_jackknife_cross_fitted,
    # A leave-one-cluster-out refit raises, so the jackknife reports no interval: the
    # interval plots read the withheld status, and influence() refuses.
    "ltmle_jackknife_withheld": sweep.fit_ltmle_jackknife_refit_withheld,
    # X123: engine options declared on a study's method object. Each result is the engine
    # route's result, so each plot draws or refuses as it does on the engine twin.
    "ltmle_bca_study": sweep.fit_study_ltmle_bca,
    "ltmle_policy_weighted_study": sweep.fit_study_ltmle_policy_weighted,
    "continuous_survival_settings_study": sweep.fit_study_continuous_survival_settings,
    "ipsi_bca_study": sweep.fit_study_ipsi_bca,
    # X54 B1: fits at their family's floor with mixed estimate statuses. The measured contrast
    # keeps the ordinary status and every other estimate shows its interval under
    # few_cluster_estimand_plugin; each plot draws or refuses as on an ordinary fit.
    "point_tmle_floor_mixed": sweep.fit_floor_mixed_tmle,
    "ltmle_floor_mixed": sweep.fit_floor_mixed_ltmle,
}

#: Public result classes that are not fits of one estimand, with the plot or fit that covers
#: them. Every other public ``*Result`` class must be the type of a ``FAMILIES`` fit.
NOT_FITS = {
    "BenchmarkResult": "an omitted-variable benchmark; confounding() draws it",
    "BootstrapResult": "a fit's replicates; bootstrap() reads it through its fit",
    "CausalResult": "a protocol, not a class with instances",
    "RefutationResult": "a refutation report of a fit",
    "SimulatedConfoundingResult": "a simulated-confounding report; sensitivity() draws it",
    "StudyResult": "a simulation study; zipper() draws it",
    "VariableImportanceResult": (
        "one TMLEResult per candidate, each covered as a point fit; a keep_fits=False "
        "result keeps no fit to plot"
    ),
    "TMLEResultSet": "one TMLEResult per level, each covered as a point fit",
}

#: Each modified-treatment-policy shape's cells, the same on every density-ratio route.
_MTP_EXPECTED: dict[str, dict[str, str]] = {
    "shift": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "DataError: A is continuous, so there",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "DataError: A is continuous, so there",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "CapabilityError: no probability model with a",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    # The banded fit's outcome is binary, so its outcome model has a calibration plot.
    "band": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "DataError: A is continuous, so there",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "DataError: A is continuous, so there",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "atoms": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "DataError: A is continuous, so there",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "DataError: A is continuous, so there",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "CapabilityError: no probability model with a",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "panel": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "DataError: A1 is a continuous treatment,",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
}

#: The pinned outcome of each plot on each family: "draws", the roadmap id a refusal
#: names, or the refusal's class and first five words.
EXPECTED: dict[str, dict[str, str]] = {
    "point_tmle_floor_mixed": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_floor_mixed": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_jackknife": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_bca_study": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "ltmle_policy_weighted_study": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "DataError: A1 is a continuous treatment,",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ipsi_bca_study": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "continuous_survival_settings_study": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "CapabilityError: calibration() draws a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "draws",
    },
    "ltmle_jackknife_withheld": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "ValueError: ey_regimen[always]: this estimate's interval is",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_mtp_censored": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "DataError: A1 is a continuous treatment,",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "sdr_mtp_censored": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "DataError: A1 is a continuous treatment,",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_repeats": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "draws",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_bootstrap": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "point_tmle_adaptive_nuisance": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_known_score": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_adaptive_nuisance": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_repeats": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "draws",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_ltmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_boundary_node": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_boundary_node_iterated": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_boundary_node_bootstrap": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "time_to_event_ctmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "ValueError: risk_regimen[arm1 @ t=2]: this estimate's",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "time_to_event_drtmle": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "continuous_survival": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "CapabilityError: calibration() draws a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "continuous_survival_bootstrap": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "CapabilityError: calibration() draws a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "draws",
    },
    "continuous_survival_restricted": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "draws",
        "calibration": "CapabilityError: calibration() draws a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "adaptive_tmle": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance: this report reads a",
        "learners": "CapabilityError: learners() reads a nuisance report's",
        "calibration": "CapabilityError: calibration() reads a nuisance report's",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "effect_curve": {
        "forest": "TypeError: forest() takes fitted results, result",
        "curve": "draws",
        "influence": "TypeError: influence() takes a ParameterEstimate or",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "TypeError: bootstrap() takes a fitted result",
    },
    "effect_curve_continuous": {
        "forest": "TypeError: forest() takes fitted results, result",
        "curve": "draws",
        "influence": "TypeError: influence() takes a ParameterEstimate or",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "TypeError: bootstrap() takes a fitted result",
    },
    "point_tmle_binary": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "point_tmle_withheld": {
        "forest": "draws",
        "curve": "draws",
        "influence": "ValueError: ate: this estimate's interval is",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "dose_response": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "structural_nested_mean": {
        "forest": "draws",
        "curve": "TypeError: curve() draws a longitudinal fit,",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "CapabilityError: diagnostic 'nuisance_models' is unavailable: a",
        "calibration": "CapabilityError: diagnostic 'nuisance_models' is unavailable: a",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_missing_treatment": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_strata": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_strata_log_ratio": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_repeats": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "draws",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_bootstrap": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "draws",
    },
    "learned_rule_ep_sieve": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "learned_rule_ep_learner": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_intermediate": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "projection_intermediate_log_ratio": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: the fit reports no estimand",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ep_curve_log_ratio_learner": {
        "forest": "TypeError: forest() takes fitted results, result",
        "curve": "draws",
        "influence": "TypeError: influence() takes a ParameterEstimate or",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "CapabilityError: repeats() draws the estimate of",
        "bootstrap": "TypeError: bootstrap() takes a fitted result",
    },
    "point_tmle_known_sample": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "CapabilityError: weights() draws the cumulative inverse-probability",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "ltmle_known_sample": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    # The coefficients have no horizon, so the curve refuses and forest draws them.
    "hazard_msm": {
        "forest": "draws",
        "curve": "CapabilityError: this fit reports the coefficients",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "hazard_mtp": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "hazard_time_varying": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "hazard_time_varying_multinode": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        # The known policy at both nodes is a ratio of probabilities, not one probability.
        "overlap": "CapabilityError: overlap() draws the probability of",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "hazard_multinode": {
        "forest": "draws",
        "curve": "draws",
        "influence": "draws",
        "overlap": "draws",
        "weights": "draws",
        "balance": "draws",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    "sequential_gcomp": {
        "forest": "draws",
        "curve": "CapabilityError: this fit has one end-of-study",
        "influence": "draws",
        "overlap": "CapabilityError: overlap() draws a point-treatment fit's",
        "weights": "CapabilityError: a g-computation or parametric g-formula",
        "balance": "CapabilityError: covariate_balance weighs the covariates by",
        "learners": "ValueError: no nuisance model here was",
        "calibration": "draws",
        "repeats": "ValueError: this fit has one cross-fitting",
        "bootstrap": "ValueError: this fit holds no bootstrap",
    },
    **{name: _MTP_EXPECTED[name.split("/")[1]] for name in mtp_route_kinds.FITS},
}


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    cache: dict[str, Any] = {}

    def get(family: str) -> Any:
        if family not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[family] = FAMILIES[family]()
        return cache[family]

    return get  # type: ignore[return-value]


def _raised_by_cleverly(error: BaseException) -> bool:
    """Whether a ``raise`` statement in ``cleverly``'s own source raised the error."""
    frame = traceback.extract_tb(error.__traceback__)[-1]
    in_package = "/src/cleverly/" in frame.filename.replace("\\", "/")
    line = linecache.getline(frame.filename, frame.lineno or 0).strip()
    return in_package and line.startswith("raise")


def outcome(plot: str, result: Any, **options: Any) -> str:
    """``"draws"``, the roadmap id a refusal names, or the refusal's class and first words.

    An exception that a ``raise`` statement of ``cleverly`` did not raise, or that is not a
    ``ValueError`` or ``TypeError`` (``CapabilityError`` is one), is a crash and propagates.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = getattr(plotting, plot)(result, **options)
    except (ValueError, TypeError) as error:
        if not _raised_by_cleverly(error):
            raise
        if isinstance(error, CapabilityError) and error.refusal.roadmap:
            return str(error.refusal.roadmap)
        return f"{type(error).__name__}: " + " ".join(str(error).split()[:5])
    assert isinstance(fig, Figure)
    assert_short_title(fig)
    plt.close(fig)
    return "draws"


@_SHARED
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("plot", PLOTS)
def test_each_plot_draws_or_refuses_each_family(fits: Any, family: str, plot: str) -> None:
    assert outcome(plot, fits(family)) == EXPECTED[family][plot]


#: Each display limit, the call that sets it small, and the note a limit that reduces adds.
#: A limit changes only what is drawn: the figure's numbers and refusals stay as they are.
LIMITED: dict[str, tuple[dict[str, Any], str]] = {
    "influence": ({"max_points": 16}, "Line drawn through"),
    "curve": ({"max_rug_points": 3}, "Rug: at most 3"),
    "balance": ({"max_covariates": 1, "page": 1}, "Page 1 of"),
}


def _drawn_arrays(fig: Figure) -> list[Any]:
    """Every line's and collection's drawn coordinates, panel by panel."""
    arrays: list[Any] = []
    for axis in fig.axes:
        arrays.extend(np.asarray(line.get_xydata()).tolist() for line in axis.lines)
        for item in axis.collections:
            segments = getattr(item, "get_segments", None)
            drawn = segments() if segments is not None else item.get_offsets()
            arrays.append([np.asarray(piece).tolist() for piece in drawn])
        arrays.append([label.get_text() for label in axis.get_yticklabels()])
    return arrays


def _limited_count(plot: str, fig: Figure) -> list[int]:
    """What each display limit bounds, per panel: line points, rug ticks or covariate rows."""
    counts = []
    for axis in fig.axes:
        if plot == "influence":
            counts.extend(
                len(line.get_xdata())
                for line in axis.lines
                if line.get_gid() == "cleverly-influence-curve"
            )
        elif plot == "curve":
            counts.extend(
                len(item.get_segments())
                for item in axis.collections
                if item.get_gid() == "cleverly-rug"
            )
        elif any(item.get_gid() == "cleverly-balance-after" for item in axis.collections):
            counts.append(len([t for t in axis.get_yticklabels() if t.get_text()]))
    return counts


def _quiet_plot(plot: str, result: Any, **options: Any) -> Figure:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(plotting, plot)(result, **options)


@_SHARED
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("plot", list(LIMITED))
def test_a_display_limit_bounds_the_drawing_and_nothing_else(
    fits: Any, family: str, plot: str
) -> None:
    """Where the plot refuses, the limited call refuses alike; where it draws, the limit holds.

    A limit that reduces leaves at most its count drawn and says so in the notes. Where it
    has nothing to reduce (a ``curve`` view with no rug, a short influence line), the figure
    is the plain one, so a limit that is ignored, or that changes another figure, fails.
    """
    options, note = LIMITED[plot]
    result = fits(family)
    plain = EXPECTED[family][plot]
    assert outcome(plot, result, **options) == plain
    if plain != "draws":
        return
    bound = next(iter(value for key, value in options.items() if key != "page"))
    before = _quiet_plot(plot, result)
    after = _quiet_plot(plot, result, **options)
    reduces = any(count > bound for count in _limited_count(plot, before))
    if plot == "balance" or reduces:
        assert all(count <= bound for count in _limited_count(plot, after))
        assert any(line.startswith(note) for line in plotting.notes(after)), plotting.notes(after)
    else:
        assert _drawn_arrays(after) == _drawn_arrays(before)
        assert plotting.notes(after) == plotting.notes(before)
    plt.close(before)
    plt.close(after)


def test_every_public_plot_is_in_the_matrix() -> None:
    assert set(plotting.__all__) == set(PLOTS) | set(OTHER_INPUTS)


def test_every_family_pins_every_plot() -> None:
    assert set(EXPECTED) == set(FAMILIES)
    for family, cells in EXPECTED.items():
        assert set(cells) == set(PLOTS), family


def test_a_crash_is_not_a_refusal() -> None:
    """Mutation control: an exception no ``raise`` of cleverly made propagates."""

    class Broken:
        estimates: dict[str, Any] = {"psi": 0.0}  # noqa: RUF012
        repeats = (object(), object())

    with pytest.raises(AttributeError):
        outcome("repeats", Broken())


@_SHARED
@pytest.mark.parametrize(
    "crash",
    [lambda: tuple(None), lambda: float("not a number")],  # type: ignore[arg-type]
    ids=["TypeError", "ValueError"],
)
def test_a_builtin_error_inside_a_plot_is_a_crash(
    fits: Any, monkeypatch: pytest.MonkeyPatch, crash: Callable[[], Any]
) -> None:
    """Mutation control: a built-in TypeError or ValueError from inside a plot fails the cell."""
    from cleverly.plotting import _diagnostics

    def broken(result: Any, name: str | None) -> Any:
        return crash()

    monkeypatch.setattr(_diagnostics, "_estimate_of", broken)
    with pytest.raises((TypeError, ValueError)):
        outcome("repeats", fits("point_tmle_repeats"))


def _public_result_classes() -> dict[str, type]:
    found: dict[str, type] = {}
    for module in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        if any(part.startswith("_") for part in module.name.split(".")[1:]):
            continue
        loaded = importlib.import_module(module.name)
        for name in getattr(loaded, "__all__", ()):
            value = getattr(loaded, name, None)
            if inspect.isclass(value) and name.endswith(("Result", "ResultSet")):
                found[name] = value
    return found


@_SHARED
def test_every_public_result_class_has_a_family_or_a_reason(fits: Any) -> None:
    covered = {type(fits(family)).__name__ for family in FAMILIES}
    shipped = set(_public_result_classes())
    assert shipped - covered - set(NOT_FITS) == set()
    assert set(NOT_FITS) <= shipped, "a NOT_FITS entry names no shipped class"
    assert not covered & set(NOT_FITS)


@pytest.mark.parametrize("kind", list(sweep.KINDS))
def test_no_plot_crashes_on_a_capability_sweep_kind(kind: str) -> None:
    """Every fit kind of the capability sweep: each plot draws or refuses, never crashes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.KINDS[kind].build()
    for plot in PLOTS:
        outcome(plot, result)


@pytest.mark.parametrize("kind", ["variable_importance", "variable_importance_no_fits"])
@pytest.mark.parametrize("plot", ["calibration", "learners"])
def test_a_variable_importance_run_redirects_its_nuisance_plots(kind: str, plot: str) -> None:
    """X104: ``result.diagnostics`` now exists on a run, and these plots would read it.

    Each candidate is a point fit, so the plot refuses the run with a redirect to one
    candidate's fit rather than drawing a report of several fits. A run that kept no fits has
    no candidate fit to plot, so it raises the keep-fits record instead.
    """
    from cleverly.exceptions import RefusalReason
    from cleverly.variable_importance import KEEP_FITS_REFUSAL

    result = sweep.KINDS[kind].build()
    with pytest.raises(CapabilityError) as caught:
        getattr(plotting, plot)(result)
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
    if kind == "variable_importance_no_fits":
        assert caught.value.refusal == KEEP_FITS_REFUSAL
    else:
        assert "result.fits[candidate]" in caught.value.refusal.alternative
    if kind == "variable_importance":
        with pytest.raises(CapabilityError) as caught:
            getattr(plotting, plot)(result.diagnostics.nuisance_models())
        assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION


def test_the_module_table_lists_every_plot() -> None:
    listed = set(re.findall(r"^:func:`(\w+)`", plotting.__doc__ or "", flags=re.M))
    assert listed == set(PLOTS) | {"sensitivity", "confounding", "targeting", "zipper"}


@_SHARED
def test_repeats_draws_each_longitudinal_draw(fits: Any) -> None:
    result = fits("ltmle_repeats")
    name = next(iter(result.estimates))
    wanted = [float(draw.estimates[name].psi) for draw in result.repeats]
    # The draws differ, so a reader that repeats one draw cannot pass.
    assert len(set(wanted)) == len(wanted)
    fig = plotting.repeats(result, name=name)
    drawn = [line for line in fig.axes[0].get_lines() if line.get_marker() == "o"]
    assert np.asarray(drawn[0].get_xdata(), dtype=float).tolist() == pytest.approx(wanted)
    # The reported estimate of a repeated fit is the median of its draws, drawn as a line.
    reported = float(result.estimates[name].psi)
    assert reported == pytest.approx(float(np.median(wanted)))
    flat = [
        float(np.asarray(line.get_xdata(), dtype=float)[0])
        for line in fig.axes[0].get_lines()
        if np.ptp(np.asarray(line.get_xdata(), dtype=float)) == 0
    ]
    assert any(value == pytest.approx(reported) for value in flat)
    plt.close(fig)


@_SHARED
def test_calibration_does_not_call_a_hazard_library_a_continuous_outcome(fits: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        plotting.calibration(fits("continuous_survival"))
    assert "continuous outcome" not in str(caught.value)


@_SHARED
def test_the_interval_families_carry_their_intervals(fits: Any) -> None:
    """Nonzero witnesses that the Fix-3 PR 3 families hold the intervals they stand for."""
    binary = fits("point_tmle_binary")
    assert binary["ate"].interval_scale == "mover"
    assert binary["ey1"].interval_scale == "range_logit"
    withheld = fits("point_tmle_withheld")
    assert withheld["ate"].inference == "constant_arm_cell_plugin"
    assert not withheld["ate"].supplies_inference
    curve = fits("effect_curve_continuous")
    assert curve.model.skew_widened()
    table = curve.predict({"W": np.linspace(-1.0, 1.0, 5)}, interval=True)
    estimate = np.asarray(table["estimate"], dtype=float)
    below = estimate - np.asarray(table["ci_lower"], dtype=float)
    above = np.asarray(table["ci_upper"], dtype=float) - estimate
    assert np.max(np.abs(above - below)) > 1e-3 * float(np.max(above + below))


@_SHARED
def test_the_curve_draws_a_supplemented_interval_and_names_the_band_gap(fits: Any) -> None:
    """A boundary-node estimate: the curve draws its own interval and says the band skips it."""
    result = fits("time_to_event_boundary_node")
    name = "risk_regimen[arm1 @ t=2]"
    estimate = result.estimates[name]
    assert estimate.node is not None
    # The supplement widens the interval past the curve's own, so a plot that redrew the
    # interval from the influence curve alone would miss these limits.
    assert estimate.variance > float(np.var(estimate.influence_curve)) / estimate.n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotting.curve(result)
    drawn = [
        vertex
        for collection in fig.axes[0].collections
        if collection.get_gid() == "cleverly-interval"
        for path in collection.get_paths()
        for vertex in path.vertices
        if 2.0 <= vertex[0] < 3.0
    ]
    heights = {round(float(y), 9) for _, y in drawn}
    low, high = estimate.ci
    assert round(float(low), 9) in heights
    assert round(float(high), 9) in heights
    gap = [note for note in plotting.notes(fig) if "boundary-node variance" in note]
    assert len(gap) == 1
    assert name in gap[0]
    assert "risk_regimen[arm1 @ t=1]" not in gap[0]
    plt.close(fig)


@_SHARED
def test_a_hazard_working_model_draws_its_coefficients_not_curves(fits: Any) -> None:
    """X70 (d): the plot of a hazard MSM fit is the forest of its terms, never a curve."""
    result = fits("hazard_msm")
    with pytest.raises(CapabilityError, match="coefficients of a working model"):
        plotting.curve(result)
    fig = result.plot()
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert len(fig.axes) == 1
    assert sorted(labels) == sorted(result.estimates)
    plt.close(fig)


@_SHARED
def test_influence_says_the_curve_is_not_a_supplemented_variance(fits: Any) -> None:
    result = fits("time_to_event_boundary_node")
    phrase = "so this curve is not the reported variance"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplemented = plotting.influence(result, name="risk_regimen[arm1 @ t=2]")
        plain = plotting.influence(result, name="risk_regimen[arm1 @ t=1]")
    assert any(
        "node with no event" in note and phrase in note for note in plotting.notes(supplemented)
    )
    assert not any(phrase in note for note in plotting.notes(plain))
    plt.close(supplemented)
    plt.close(plain)


@_SHARED
@pytest.mark.parametrize("family", ["ltmle_jackknife"])
def test_influence_says_the_curve_is_not_the_jackknife_variance(fits: Any, family: str) -> None:
    result = fits(family)
    estimates = getattr(result, "estimates", result)
    name = next(iter(estimates))
    assert estimates[name].variance_source == "jackknife"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = plotting.influence(result, name=name)
    assert any(
        "delete-one-cluster jackknife" in note and "not the reported variance" in note
        for note in plotting.notes(figure)
    )
    plt.close(figure)


@_SHARED
def test_the_bootstrap_of_a_supplemented_estimate_draws_as_a_named_diagnostic(fits: Any) -> None:
    """A row resample keeps the flagged node's one value: the plot and the band say so."""
    result = fits("time_to_event_boundary_node_bootstrap")
    reason = "every resample of rows keeps the flagged node's one value"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplemented = plotting.bootstrap(result, name="risk_regimen[arm1 @ t=2]")
        plain = plotting.bootstrap(result, name="risk_regimen[arm1 @ t=1]")
        curve = plotting.curve(result, band="bootstrap")
    assert any(reason in note for note in plotting.notes(supplemented))
    assert not any(reason in note for note in plotting.notes(plain))
    assert "risk_regimen[arm1 @ t=2]" not in result.bootstrap_band.bands
    assert any(
        "nor a row resample carries" in note and "risk_regimen[arm1 @ t=2]" in note
        for note in plotting.notes(curve)
    )
    for fig in (supplemented, plain, curve):
        plt.close(fig)


@_SHARED
def test_a_hazard_policy_names_its_shifted_weights(fits: Any) -> None:
    """X70 (c): the later nodes of a held policy are drawn as held, under the policy's note."""
    from cleverly.longitudinal import node_weights

    result = fits("hazard_mtp")
    kinds = node_weights(result, regimen="up", horizon=4).node_kinds
    assert kinds == ("mtp", "held", "held", "held")
    fig = plotting.weights(result, regimens=["up"])
    text = " ".join(plotting.notes(fig))
    assert "modified treatment policy" in text
    assert "held" not in text
    plt.close(fig)


@_SHARED
def test_a_static_arm_of_a_multinode_hazard_fit_draws_its_overlap(fits: Any) -> None:
    """X70 (b2): overlap of every regimen refuses on the known policy, so one static arm is
    drawn on its own, by the same path as the held designs."""
    result = fits("hazard_time_varying_multinode")
    fig = plotting.overlap(result, regimen="a1")
    assert fig.axes
    plt.close(fig)


def test_a_held_rule_is_capped_at_its_last_node() -> None:
    """A one-node known rule on a held design: later nodes are held, and the limit caps them.

    A held node repeats the baseline decision and adds no treatment factor, so the weight at
    the last node is still the data's weight over the bounded cumulative probability.
    """
    from cleverly.longitudinal import DynamicRegimen, node_weights
    from cleverly.plotting._weights import _capped

    frame, _ = make_point_survival(n=600, seed=3, n_times=4)
    rule = DynamicRegimen(
        "rule", (lambda h: np.asarray(h["W1"] > 0, dtype=int),), rule_kind="known"
    )
    study = CausalStudy(
        frame,
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.identify(
            RegimeContrast(regimens={"rule": rule, "a0": 0}, reference="a0")
        ).estimate(
            outcome_learner=LogisticRegression(max_iter=1000, random_state=0),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
            censoring_learner=LogisticRegression(max_iter=1000, random_state=0),
            survival_model="hazard",
            targeting="one_step",
            random_state=5,
            simultaneous=False,
            **sweep.IN_SAMPLE,
        )
    kinds = node_weights(result, regimen="rule", horizon=4).node_kinds
    assert kinds == ("rule", "held", "held", "held")
    assert _capped(kinds, len(kinds) - 1)
    # A static arm reports label at every node.
    assert node_weights(result, regimen="a0", horizon=4).node_kinds == ("label",) * 4
    fig = plotting.weights(result, regimens=["rule"])
    assert "held" not in " ".join(plotting.notes(fig))
    plt.close(fig)
