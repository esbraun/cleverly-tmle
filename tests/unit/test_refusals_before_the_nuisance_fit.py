"""Each composition refusal of RM24 raises ``CapabilityError`` before any learner.

Before RM24 in ``docs/roadmap.md`` at ``4ce96cda``, a stratified incremental fit and a stratified log- or
logit-link MSM fit raised ``NotImplementedError`` from the strata guard in
``TMLE._retarget_detailed`` after two learner fits.  A stratified ``DRTMLE`` fit at a non-empty
``guard`` raised there after eight, and ``DRTMLE`` with ``targeting="one_step"`` and
``reduced_crossfit="nested"`` raised from ``solve_submodel`` after 42.  The other composition
refusals of ``src/cleverly/estimators/`` raised ``ValueError`` or ``NotImplementedError`` before
any learner, and the refit replay slot read both as refusals.  X8 in ``docs/roadmap.md`` tracks the
stratified targeting, and F6 tracks the incremental-intermediate composition.  This module pins
these things:

* each stratified request refuses with ``CapabilityError`` before any learner and cites X8, and
  the ``DRTMLE`` message names ``guard=()`` rather than an arm, regime or shift target;
* a stratified ``DRTMLE`` request that ``guard=()`` does not repair, with ``att`` or with a
  cross-fitted ``delta=``, meets that refusal first, so no remedy names a refused request;
* ``CausalStudy.identify`` refuses the stratified incremental and MSM estimands, and
  ``estimate`` fits no learner, for those estimands and for ``DRTMLE``;
* the refit replay slot reads the stratified ``DRTMLE`` refusal on a copied estimator, and
  ``refit`` raises the same sentence before any learner;
* ``simulated_confounding`` refuses a stratum of a ``guard=()`` result, which fits strata, with
  the reason that holds for it: the surface covers marginal DR-TMLE targets only;
* the one-step nested composition refuses before any learner, on a copied estimator as well;
* each other moved refusal is a ``CapabilityError``, and the incremental-intermediate refusal
  cites F6;
* stratified arm, regime and shift targets, an identity-link MSM and a ``DRTMLE`` fit at
  ``guard=()`` still fit;
* a mutation that removes the estimator check lets a learner fit, and the targeting-loop
  backstop then refuses after the learners.  A mutation that removes the identification check
  moves the study refusal to ``estimate``, and removing both lets a learner fit.  A mutation that
  removes the ``DRTMLE`` check fails the ``DRTMLE`` witnesses.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.study as study_module
from cleverly import (
    ATE,
    CausalStudy,
    DRTMLEMethod,
    IncrementalEffect,
    IncrementalMean,
    MSMProjection,
    PointTreatment,
)
from cleverly.assessment import POINT_REPLAY_REFIT_CONFIGURATION, replayability
from cleverly.datasets import make_cde, make_linear_ate
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.reduced import refuse_unsupported
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Shift, Static
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.conftest import linear_in_sample
from tests.unit._declaration_support import (
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    tmle_module,
)
from tests.unit._natural_course_support import (
    Counting,
    CountingLinear,
    CountingLogistic,
    NeverFit,
    never_fit_learners,
)
from tests.unit.test_drtmle_missing import _binary_trial, _trial

drtmle_module = importlib.import_module("cleverly.estimators.drtmle")

X8 = "docs/roadmap.md X8"
F6 = "docs/roadmap.md F6"
COVARIATES = ["W1", "W2", "W3", "W4", "S"]
DOSES = (-1.0, 0.0, 1.0, 2.0)
IPSI = "the 'ipsi' group's alternating targeting equations"
MSM_GROUP = "the 'msm' group's alternating targeting equations"
MEAN = "the 'mean' group's alternating targeting equations"
CONTINUOUS = "continuous MSMs do not yet support baseline strata"
GUARD_REMEDY = "pass guard=(), which is the ordinary TMLE and accepts strata="
NESTED = "targeting='one_step' and reduced_crossfit='nested' are not combined"
#: Every non-empty ``guard``, each of which gives the ``mean`` group reduced regressions.
GUARDS = [("Q", "g"), ("Q",), ("g",)]


def strata_frame(n: int = 200) -> pd.DataFrame:
    """``make_linear_ate(n, seed=2)`` with the baseline stratum ``S = (W1 > 0)``."""
    frame, _ = make_linear_ate(n=n, seed=2)
    return frame.assign(S=(frame["W1"] > 0).astype(int))


def binary_strata_frame() -> pd.DataFrame:
    """:func:`strata_frame` with ``Y`` replaced by ``Y > median(Y)``, for the logit link."""
    frame = strata_frame()
    return frame.assign(Y=(frame["Y"] > frame["Y"].median()).astype(int))


def positive_strata_frame() -> pd.DataFrame:
    """:func:`strata_frame` with ``Y`` replaced by ``exp(Y / 10)``, for the log link."""
    frame = strata_frame()
    return frame.assign(Y=np.exp(frame["Y"] / 10.0))


def dose_strata_frame() -> pd.DataFrame:
    """:func:`strata_frame` with the dose ``D = A + N(0, 1)``."""
    frame = strata_frame()
    return frame.assign(D=frame["A"] + np.random.default_rng(0).normal(size=len(frame)))


def spies(**extra: Any) -> dict[str, Any]:
    """In-sample settings whose every point-treatment learner is a ``NeverFit``."""
    return linear_in_sample(**never_fit_learners(), **extra)


def drtmle_spies(**extra: Any) -> dict[str, Any]:
    """:func:`spies` with ``NeverFit`` reduced learners too."""
    return spies(reduced_outcome_learner=NeverFit(), reduced_treatment_learner=NeverFit(), **extra)


def fit(estimator: TMLE, frame: pd.DataFrame, **columns: Any) -> Any:
    return estimator.fit(frame, outcome="Y", covariates=COVARIATES, strata=["S"], **columns)


def fit_dose(estimator: TMLE) -> Any:
    return fit(estimator, dose_strata_frame(), treatment="D", treatment_kind="continuous")


def study(frame: pd.DataFrame | None = None, **design: Any) -> CausalStudy:
    design = {"treatment": "A", **design}
    return CausalStudy(
        strata_frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", adjustment=tuple(COVARIATES), strata=("S",), **design),
    )


def dose_study() -> CausalStudy:
    return study(dose_strata_frame(), treatment="D", treatment_kind="continuous")


TILTS = IncrementalEffect(
    (Incremental(1.0, name="one"), Incremental(2.0, name="two")), reference="one"
)

#: A stratified ``TMLE`` fit, built from its learner settings, and the fragments of its refusal.
TMLE_ROWS: dict[str, tuple[Callable[[dict[str, Any]], Any], tuple[str, ...]]] = {
    "incremental, in sample": (
        lambda s: fit(TMLE(incremental=[Incremental(2.0)], **s), strata_frame(), treatment="A"),
        (IPSI, "use an arm/regime/shift target", X8),
    ),
    "incremental, cross-fitted": (
        lambda s: fit(
            TMLE(
                incremental=[Incremental(2.0)],
                **{**s, "cross_fit": True, "n_folds": 2, "q_bounds": (-30.0, 30.0)},
            ),
            strata_frame(),
            treatment="A",
        ),
        (IPSI, X8),
    ),
    "log-link MSM": (
        lambda s: fit(
            TMLE(msm=MSM.linear(link="log"), **s), positive_strata_frame(), treatment="A"
        ),
        (MSM_GROUP, "use an arm/regime/shift target", X8),
    ),
    "logit-link MSM": (
        lambda s: fit(
            TMLE(msm=MSM.linear(link="logit"), **s), binary_strata_frame(), treatment="A"
        ),
        (MSM_GROUP, "use an arm/regime/shift target", X8),
    ),
    "continuous-dose MSM": (
        lambda s: fit_dose(TMLE(msm=MSM.linear(doses=DOSES), density_bins=6, **s)),
        (CONTINUOUS, "Fit the marginal MSM projection", X8),
    ),
}


def tmle_witness(name: str) -> None:
    build, fragments = TMLE_ROWS[name]
    assert_refused_before_any_call(lambda: build(spies()), None, "", *fragments)


def drtmle_witness(guard: tuple[str, ...]) -> None:
    """The ``DRTMLE`` message names ``guard=()``, and no target that ``DRTMLE`` refuses."""
    estimator = DRTMLE(guard=guard, estimands=("ate",), **drtmle_spies())
    raised = assert_refused_before_any_call(
        lambda: fit(estimator, strata_frame(), treatment="A"), None, "", MEAN, GUARD_REMEDY, X8
    )
    assert "arm/regime/shift" not in str(raised)


def _stratified_trial() -> pd.DataFrame:
    return _binary_trial(100).assign(S=lambda f: (f["W1"] > 0).astype(int))


#: A stratified ``DRTMLE`` request that ``guard=()`` does not repair, and its refusal.
GUARD_INDEPENDENT: dict[str, tuple[Callable[[], Any], str]] = {
    "att": (
        lambda: fit(
            DRTMLE(estimands=("ate", "att"), **drtmle_spies()), strata_frame(), treatment="A"
        ),
        "DRTMLE does not support estimand(s) ['att']",
    ),
    "cross-fitted delta=": (
        lambda: DRTMLE(
            randomized=True, estimands=("ate",), **drtmle_spies(cross_fit=True, n_folds=5)
        ).fit(
            _stratified_trial(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "S"],
            delta="Delta",
            strata=["S"],
        ),
        "does not establish its cross-validated extension",
    ),
}


def guard_independent_witness(name: str) -> None:
    """The refusal that also holds at ``guard=()`` comes before the strata refusal.

    The strata refusal names ``guard=()`` as the remedy, and this request is refused there too.
    """
    build, fragment = GUARD_INDEPENDENT[name]
    raised = assert_refused_before_any_call(build, None, "", fragment)
    assert "guard=()" not in str(raised)


IDENTIFY_ROWS: dict[str, tuple[Callable[[], CausalStudy], Any, str]] = {
    "incremental": (study, TILTS, IPSI),
    "incremental mean": (study, IncrementalMean((Incremental(2.0, name="two"),)), IPSI),
    "log-link MSM": (
        lambda: study(positive_strata_frame()),
        MSMProjection(MSM.linear(link="log")),
        MSM_GROUP,
    ),
    "logit-link MSM": (
        lambda: study(binary_strata_frame()),
        MSMProjection(MSM.linear(link="logit")),
        MSM_GROUP,
    ),
    "continuous-dose MSM": (dose_study, MSMProjection(MSM.linear(doses=DOSES)), CONTINUOUS),
}


def identify_witness(name: str) -> None:
    build, estimand, fragment = IDENTIFY_ROWS[name]
    assert_refused(lambda: build().identify(estimand), CapabilityError, fragment, X8)


ESTIMATE_ROWS: dict[str, tuple[Any, str, str]] = {
    "incremental": (TILTS, "tmle", IPSI),
    "DR-TMLE": (ATE(), "drtmle", MEAN),
}


def estimate_witness(name: str) -> None:
    """The study refuses before any learner, whichever check raises."""
    estimand, method, fragment = ESTIMATE_ROWS[name]
    assert_refused_before_any_call(
        lambda: study().estimate(
            estimand, method, cross_fit=False, simultaneous=False, **never_fit_learners()
        ),
        None,
        "",
        fragment,
        X8,
    )


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a learner was reached before the configuration refusal")


def refit_slot_witness(result: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``guard=()`` fit whose copied estimator is set to ``guard=("Q", "g")``.

    Before RM24 the slot read ``True``, and the refit raised after its learners.
    """
    estimator = copy.copy(result.estimator)
    estimator.guard = ("Q", "g")
    reconfigured = replace(result, estimator=estimator)
    monkeypatch.setattr(estimator, "_nuisances", _never)
    reason = estimator._refit_configuration_refusal(reconfigured.data)
    assert reason is not None and MEAN in reason and X8 in reason, reason
    replay = replayability(reconfigured)
    assert not replay.refit_nuisances
    assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
    try:
        estimator.refit(reconfigured.data)
    except CapabilityError as raised:
        assert str(raised) == reason
    else:
        raise AssertionError("the refit ran")


@pytest.fixture(scope="module")
def unguarded_result() -> Any:
    """A stratified ``DRTMLE(guard=())`` fit, which is the ordinary TMLE."""
    return fit(
        DRTMLE(guard=(), estimands=("ate",), **linear_in_sample()), strata_frame(), treatment="A"
    ).single()


class TestAStratifiedRequestRefusesBeforeAnyLearner:
    @pytest.mark.parametrize("name", list(TMLE_ROWS))
    def test_the_fit_names_x8(self, name: str) -> None:
        tmle_witness(name)

    @pytest.mark.parametrize("guard", GUARDS)
    def test_the_drtmle_fit_names_x8_and_the_empty_guard(self, guard: tuple[str, ...]) -> None:
        drtmle_witness(guard)

    @pytest.mark.parametrize("name", list(GUARD_INDEPENDENT))
    def test_a_refusal_that_guard_repairs_comes_last(self, name: str) -> None:
        guard_independent_witness(name)


class TestTheStudyRefusesAtIdentify:
    @pytest.mark.parametrize("name", list(IDENTIFY_ROWS))
    def test_identify_names_x8(self, name: str) -> None:
        identify_witness(name)

    @pytest.mark.parametrize("name", list(ESTIMATE_ROWS))
    def test_estimate_fits_no_learner(self, name: str) -> None:
        estimate_witness(name)


class TestTheRefitSlotReadsTheStratifiedRefusal:
    def test_a_reconfigured_drtmle_refit_is_refused_before_any_learner(
        self, unguarded_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        refit_slot_witness(unguarded_result, monkeypatch)


class TestTheSurfaceRefusesAStratumOfAnUnguardedFit:
    def test_the_reason_is_the_marginal_contract(self) -> None:
        """The surface guard is the only refusal of this request, so its reason must hold.

        The fit has no reduced regressions, so the reason is not stratified reduced-regression
        targeting.  The marginal alias of the same result replays.
        """
        result = study().estimate(
            ATE(),
            DRTMLEMethod(guard=()),
            cross_fit=False,
            simultaneous=False,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.02))
        with pytest.raises(CapabilityError) as raised:
            simulated_confounding(result, estimand="ate[S=1]", grid=grid, random_state=3)
        message = str(raised.value)
        assert "covers marginal arm means, ATE and ratios only" in message
        assert "fit it with TMLE" in message
        assert "reduced-regression" not in message
        simulated_confounding(result, estimand="ate", grid=grid, random_state=3)


class TestTheOneStepNestedComposition:
    @staticmethod
    def build(**extra: Any) -> DRTMLE:
        settings = drtmle_spies(cross_fit=True, n_folds=3, q_bounds=(-30.0, 30.0))
        return DRTMLE(reduced_crossfit="nested", estimands=("ate",), **settings, **extra)

    @pytest.mark.parametrize("guard", [("Q", "g"), ("Q",)])
    def test_the_fit_refuses_it_before_any_learner(self, guard: tuple[str, ...]) -> None:
        estimator = self.build(guard=guard, targeting="one_step")
        assert_refused_before_any_call(
            lambda: estimator.fit(strata_frame(), outcome="Y", treatment="A", covariates=["W1"]),
            None,
            "",
            NESTED,
            "Use targeting='iterative'",
        )

    def test_a_copied_estimator_is_refused_before_any_learner(self) -> None:
        """The constructor admits ``targeting="iterative"``, and a caller changes it after."""
        estimator = copy.copy(self.build())
        estimator.targeting = "one_step"
        assert_refused_before_any_call(
            lambda: estimator.fit(strata_frame(), outcome="Y", treatment="A", covariates=["W1"]),
            None,
            "",
            NESTED,
        )


class TestTheIncrementalIntermediateRefusal:
    def test_it_is_a_capability_error_before_any_learner(self) -> None:
        frame, _ = make_cde(n=200, seed=3)
        assert_refused_before_any_call(
            lambda: TMLE(incremental=[Incremental(2.0)], **spies()).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], intermediate="Z"
            ),
            None,
            "",
            "incremental= and intermediate= are not combined",
            F6,
        )


def _missing_drtmle(frame: pd.DataFrame, **columns: Any) -> Callable[[], Any]:
    """A randomized in-sample missing-outcome ``DRTMLE`` fit with ``NeverFit`` learners."""
    return lambda: DRTMLE(randomized=True, estimands=("ate",), **drtmle_spies()).fit(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta", **columns
    )


def _three_arms() -> pd.DataFrame:
    frame = _trial(120)
    return frame.assign(A=np.where(np.arange(len(frame)) < 30, 2.0, frame["A"]))


def _cde_fit(estimator: TMLE) -> Callable[[], Any]:
    frame, _ = make_cde(n=200, seed=3)
    return lambda: estimator.fit(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], intermediate="Z"
    )


def _plain_fit(estimator: TMLE, frame: pd.DataFrame | None = None, **columns: Any) -> Any:
    return estimator.fit(
        strata_frame() if frame is None else frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        **columns,
    )


def _dose_fit(estimator: TMLE) -> Any:
    return estimator.fit(
        dose_strata_frame(),
        outcome="Y",
        treatment="D",
        covariates=["W1", "W2"],
        treatment_kind="continuous",
    )


#: Each moved refusal that no other test pins as ``CapabilityError``, and one fragment of it.
#: A constructor row builds the estimator.  A fit row fits with ``NeverFit`` learners.
MOVED: dict[str, tuple[Callable[[], Any], str]] = {
    "TMLE fold targeting with incremental=": (
        lambda: TMLE(
            targeting_scheme="fold", incremental=[Incremental(2.0)], cross_fit=True, n_folds=2
        ),
        "targeting_scheme='fold' is not implemented for incremental interventions",
    ),
    "TMLE cv_evaluation with rr": (
        lambda: _plain_fit(
            TMLE(**spies(cross_fit=True, n_folds=2, cv_evaluation=True, estimands=("rr",))),
            binary_strata_frame(),
        ),
        "cv_evaluation=True does not yet support ['rr']",
    ),
    "CTMLE cv_evaluation": (
        lambda: CTMLE(cv_evaluation=True, cross_fit=True, n_folds=2),
        "CTMLE does not support cv_evaluation=True",
    ),
    "CTMLE fold targeting": (
        lambda: CTMLE(targeting_scheme="fold", cross_fit=True, n_folds=2),
        "CTMLE implements the published pooled collaborative estimator only",
    ),
    "CTMLE incremental=": (
        lambda: _plain_fit(CTMLE(incremental=[Incremental(2.0)], **spies())),
        "CTMLE and incremental= are not combined",
    ),
    "CTMLE continuous dose": (
        lambda: _dose_fit(CTMLE(msm=MSM.linear(doses=DOSES), density_bins=6, **spies())),
        "CTMLE strategies require a discrete treatment",
    ),
    "CTMLE intermediate=": (
        lambda: _cde_fit(CTMLE(**spies()))(),
        "CTMLE does not compose either collaborative strategy with an intermediate outcome",
    ),
    "CTMLE att": (
        lambda: _plain_fit(CTMLE(estimands=("ate", "att"), **spies())),
        "CTMLE does not support estimand(s) ['att']",
    ),
    "CTMLE no selector criterion": (
        lambda: _plain_fit(CTMLE(ctmle_estimand="par", estimands=("ate", "par"), **spies())),
        "ctmle_estimand='par' has no selector criterion",
    ),
    "DRTMLE evaluation= with repeats=": (
        lambda: DRTMLE(evaluation=strata_frame(), repeats=2, cross_fit=True, n_folds=2),
        "evaluation= and repeats= are not combined",
    ),
    "DRTMLE msm=": (lambda: DRTMLE(msm=MSM.linear()), "DRTMLE and msm= are not combined"),
    "DRTMLE interventions=": (
        lambda: DRTMLE(interventions=(Static(1), Static(0))),
        "DRTMLE and interventions= are not combined",
    ),
    "DRTMLE intermediate=": (
        lambda: _cde_fit(DRTMLE(estimands=("ate",), **drtmle_spies()))(),
        "DRTMLE and intermediate= are not combined",
    ),
    "DRTMLE treatment_probabilities= without delta=": (
        lambda: _plain_fit(
            DRTMLE(estimands=("ate",), **drtmle_spies()),
            treatment_probabilities=np.full(200, 0.5),
        ),
        "treatment_probabilities= is currently only used with delta=",
    ),
    "DRTMLE missing outcomes, three arms, observational": (
        lambda: DRTMLE(estimands=("ate",), **drtmle_spies()).fit(
            _three_arms(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
        ),
        "DRTMLE with delta= is supported only for a randomized trial",
    ),
    "DRTMLE missing outcomes, weighted": (
        lambda: _missing_drtmle(_trial(120).assign(wt=np.linspace(0.5, 1.5, 120)), weights="wt")(),
        "missing-outcome DRTMLE is not certified for a weight-tilted target law",
    ),
    "DRTMLE missing outcomes, evaluation=": (
        lambda: DRTMLE(
            randomized=True, estimands=("ate",), evaluation=_trial(40), **drtmle_spies()
        ).fit(_trial(120), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"),
        "missing-outcome DRTMLE supports the published pooled construction only",
    ),
    "DRTMLE missing outcomes, three arms, weighted": (
        lambda: _missing_drtmle(
            _three_arms().assign(wt=np.linspace(0.5, 1.5, 120)), weights="wt"
        )(),
        "missing-outcome DRTMLE is not certified for a weight-tilted target law",
    ),
    "DRTMLE missing outcomes, three arms, evaluation=": (
        lambda: DRTMLE(
            randomized=True, estimands=("ate",), evaluation=_trial(40), **drtmle_spies()
        ).fit(_three_arms(), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"),
        "missing-outcome DRTMLE supports the published pooled construction only",
    ),
    "DRTMLE missing outcomes, three arms, partial guard": (
        lambda: DRTMLE(randomized=True, guard=("Q",), estimands=("ate",), **drtmle_spies()).fit(
            _three_arms(), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        ),
        "missing-outcome DRTMLE requires guard=('Q', 'g')",
    ),
    "DRTMLE missing outcomes, three arms, bivariate": (
        lambda: DRTMLE(
            randomized=True, reduction="bivariate", estimands=("ate",), **drtmle_spies()
        ).fit(_three_arms(), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"),
        "reduction='bivariate' is the complete-outcome construction",
    ),
    "the continuous reduction": (
        lambda: refuse_unsupported("continuous"),
        "a continuous dose has no arms to index by",
    ),
}


class TestEachMovedRefusalIsACapabilityError:
    @pytest.mark.parametrize("name", list(MOVED))
    def test_the_refusal_is_a_capability_error(self, name: str) -> None:
        build, fragment = MOVED[name]
        NeverFit.calls = 0
        assert_refused_before_any_call(build, None, "", fragment)


class TestSupportedNeighboursStillFit:
    """Each control fits with ``strata=`` and reports a stratum-specific name."""

    @staticmethod
    def names(result: Any) -> list[str]:
        return list(result.single().estimates)

    def test_arm_targets(self) -> None:
        """The check keys on the group, not on ``strata=`` alone."""
        assert "ate[S=1]" in self.names(
            fit(TMLE(**linear_in_sample()), strata_frame(), treatment="A")
        )

    def test_an_identity_link_msm(self) -> None:
        """A check keyed on ``msm is not None`` refuses this fit."""
        result = fit(TMLE(msm=MSM.linear(), **linear_in_sample()), strata_frame(), treatment="A")
        assert "msm[a][S=1]" in self.names(result)

    def test_regimes(self) -> None:
        estimator = TMLE(interventions=(Static(1), Static(0)), **linear_in_sample())
        assert "ey_regime[always 1][S=1]" in self.names(
            fit(estimator, strata_frame(), treatment="A")
        )

    def test_shifts(self) -> None:
        """The continuous branch keys on ``msm=``, not on a continuous dose."""
        estimator = TMLE(shifts=[Shift(0.5, cap=None)], density_bins=6, **linear_in_sample())
        assert "ey_shift[+0.5][S=1]" in self.names(fit_dose(estimator))

    def test_an_empty_drtmle_guard(self, unguarded_result: Any) -> None:
        """A check keyed on ``guard is not None`` refuses this fit."""
        assert "ate[S=1]" in unguarded_result.estimates

    def test_the_one_step_nested_fit_at_an_empty_guard(self) -> None:
        """The one-step nested check keys on a non-empty ``guard``."""
        settings = linear_in_sample(cross_fit=True, n_folds=3, q_bounds=(-30.0, 30.0))
        estimator = DRTMLE(
            guard=(),
            targeting="one_step",
            reduced_crossfit="nested",
            estimands=("ate",),
            **settings,
        )
        result = estimator.fit(strata_frame(), outcome="Y", treatment="A", covariates=["W1"])
        assert self.names(result) == ["ate"]

    def test_an_incremental_fit_without_strata(self) -> None:
        estimator = TMLE(incremental=[Incremental(2.0)], **linear_in_sample())
        result = estimator.fit(strata_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
        assert any(name.startswith("ey_ipsi") for name in self.names(result))


class TestTheWitnessesHaveTeeth:
    @staticmethod
    def remove(monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
        for module in modules:
            monkeypatch.setattr(module, "refuse_stratified_targeting", lambda *a, **k: None)

    def test_removing_the_estimator_check_lets_a_learner_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each ``TMLE`` row reaches ``NeverFit.fit``.  The identify check still refuses."""
        self.remove(monkeypatch, tmle_module)
        for name in TMLE_ROWS:
            assert_every_witness_fails([lambda name=name: tmle_witness(name)])
            assert NeverFit.calls > 0, f"{name} failed before any learner"
        for name in IDENTIFY_ROWS:
            identify_witness(name)

    def test_the_backstop_refuses_after_the_learners(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The targeting-loop guard raises the same refusal, and only a spy catches its timing."""
        self.remove(monkeypatch, tmle_module)
        Counting.calls = 0
        estimator = TMLE(
            incremental=[Incremental(2.0)],
            **linear_in_sample(
                outcome_learner=CountingLinear(), treatment_learner=CountingLogistic(max_iter=1000)
            ),
        )
        with pytest.raises(CapabilityError) as raised:
            fit(estimator, strata_frame(), treatment="A")
        assert IPSI in str(raised.value) and X8 in str(raised.value)
        assert Counting.calls == 2

    def test_removing_the_identify_check_moves_the_refusal_to_estimate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.remove(monkeypatch, study_module)
        assert_every_witness_fails(
            [lambda name=name: identify_witness(name) for name in IDENTIFY_ROWS]
        )
        estimate_witness("incremental")

    def test_removing_both_checks_lets_a_learner_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.remove(monkeypatch, study_module, tmle_module)
        assert_every_witness_fails([lambda: estimate_witness("incremental")])
        assert NeverFit.calls > 0

    def test_removing_the_drtmle_check_fails_the_drtmle_witnesses(
        self, unguarded_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.remove(monkeypatch, drtmle_module)
        assert_every_witness_fails(
            [
                *(lambda guard=guard: drtmle_witness(guard) for guard in GUARDS),
                lambda: estimate_witness("DR-TMLE"),
                lambda: refit_slot_witness(unguarded_result, monkeypatch),
            ]
        )
