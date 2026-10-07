"""Each composition refusal raises ``CapabilityError`` or ``DataError`` before any learner.

Before RM24 in ``docs/roadmap.md`` at ``4ce96cda``, a stratified incremental fit and a stratified
log- or logit-link MSM fit raised ``NotImplementedError`` after two learner fits, and the other
composition refusals of ``src/cleverly/estimators/`` raised ``ValueError`` or
``NotImplementedError``.  The stratified incremental, linked, continuous-dose and DR-TMLE fits now
target one block per stratum, so this module pins these things:

* each formerly refused stratified request reaches a learner, on the estimator, on
  ``CausalStudy.identify`` and ``estimate``, and through the refit replay slot;
* each stratified refusal that remains refuses before any learner with its own reason: fold-wise
  targeting, fold evaluation, and a DR-TMLE stratum with no trainable rows in some training
  complement.  The incremental and MSM data refusals are pinned in their exact-law modules;
* a stratified ``DRTMLE`` request that ``guard=()`` does not repair, with ``att`` or with a
  cross-fitted ``delta=``, meets its own refusal;
* ``simulated_confounding`` replays a stratum of a ``guard=()`` result;
* the one-step nested composition refuses before any learner, on a copied estimator as well;
* each other moved refusal is a ``CapabilityError``, and the incremental-intermediate refusal
  cites F6;
* a mutation that removes a stratified data check lets a learner fit.
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

from cleverly import (
    ATE,
    CausalStudy,
    DRTMLEMethod,
    IncrementalEffect,
    IncrementalMean,
    MSMProjection,
    PointTreatment,
)
from cleverly.assessment import replayability
from cleverly.datasets import make_cde, make_linear_ate
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.reduced import refuse_unsupported
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, LearnedRule, Shift, Static
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.conftest import linear_in_sample
from tests.unit._declaration_support import (
    assert_refused_before_any_call,
    tmle_module,
)
from tests.unit._natural_course_support import (
    NeverFit,
    never_fit_learners,
)
from tests.unit.test_drtmle_missing import _binary_trial, _trial

drtmle_module = importlib.import_module("cleverly.estimators.drtmle")

F6 = "docs/roadmap.md F6"
COVARIATES = ["W1", "W2", "W3", "W4", "S"]
DOSES = (-1.0, 0.0, 1.0, 2.0)
NESTED = "targeting='one_step' and reduced_crossfit='nested' are not combined"
#: Every non-empty ``guard``, each of which gives the ``mean`` group reduced regressions.
GUARDS = [("Q", "g"), ("Q",), ("g",)]
#: What a ``NeverFit`` learner raises when a request reaches it.
REACHED = "before any learner is fitted"


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

#: Each stratified ``TMLE`` request that refused before the stratum targeting existed.
TMLE_ROWS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "incremental, in sample": lambda s: fit(
        TMLE(incremental=[Incremental(2.0)], **s), strata_frame(), treatment="A"
    ),
    "incremental, cross-fitted": lambda s: fit(
        TMLE(
            incremental=[Incremental(2.0)],
            **{**s, "cross_fit": True, "n_folds": 2, "q_bounds": (-30.0, 30.0)},
        ),
        strata_frame(),
        treatment="A",
    ),
    "log-link MSM": lambda s: fit(
        TMLE(msm=MSM.linear(link="log"), **s), positive_strata_frame(), treatment="A"
    ),
    "logit-link MSM": lambda s: fit(
        TMLE(msm=MSM.linear(link="logit"), **s), binary_strata_frame(), treatment="A"
    ),
    "continuous-dose MSM": lambda s: fit_dose(
        TMLE(msm=MSM.linear(doses=DOSES), density_bins=6, **s)
    ),
}


def reaches_a_learner(build: Callable[[], Any]) -> None:
    """``build`` passes every preflight and fails in the first ``NeverFit`` learner."""
    NeverFit.calls = 0
    with pytest.raises(AssertionError, match=REACHED):
        build()
    assert NeverFit.calls > 0


def tmle_witness(name: str) -> None:
    reaches_a_learner(lambda: TMLE_ROWS[name](spies()))


def drtmle_witness(guard: tuple[str, ...]) -> None:
    estimator = DRTMLE(guard=guard, estimands=("ate",), **drtmle_spies())
    reaches_a_learner(lambda: fit(estimator, strata_frame(), treatment="A"))


def _stratified_trial() -> pd.DataFrame:
    return _binary_trial(100).assign(S=lambda f: (f["W1"] > 0).astype(int))


#: A stratified ``DRTMLE`` request refused for a reason unrelated to the strata.
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


IDENTIFY_ROWS: dict[str, tuple[Callable[[], CausalStudy], Any]] = {
    "incremental": (study, TILTS),
    "incremental mean": (study, IncrementalMean((Incremental(2.0, name="two"),))),
    "log-link MSM": (
        lambda: study(positive_strata_frame()),
        MSMProjection(MSM.linear(link="log")),
    ),
    "logit-link MSM": (
        lambda: study(binary_strata_frame()),
        MSMProjection(MSM.linear(link="logit")),
    ),
    "continuous-dose MSM": (dose_study, MSMProjection(MSM.linear(doses=DOSES))),
}

ESTIMATE_ROWS: dict[str, tuple[Any, Any]] = {
    "incremental": (TILTS, "tmle"),
    "DR-TMLE": (ATE(), "drtmle"),
}


def estimate_witness(name: str) -> None:
    estimand, method = ESTIMATE_ROWS[name]
    reaches_a_learner(
        lambda: study().estimate(
            estimand, method, cross_fit=False, simultaneous=False, **never_fit_learners()
        )
    )


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a learner was reached before the configuration refusal")


@pytest.fixture(scope="module")
def unguarded_result() -> Any:
    """A stratified ``DRTMLE(guard=())`` fit, which is the ordinary TMLE."""
    return fit(
        DRTMLE(guard=(), estimands=("ate",), **linear_in_sample()), strata_frame(), treatment="A"
    ).single()


class TestAFormerlyRefusedStratifiedRequestFits:
    @pytest.mark.parametrize("name", list(TMLE_ROWS))
    def test_the_fit_reaches_a_learner(self, name: str) -> None:
        tmle_witness(name)

    @pytest.mark.parametrize("guard", GUARDS)
    def test_the_drtmle_fit_reaches_a_learner(self, guard: tuple[str, ...]) -> None:
        drtmle_witness(guard)

    @pytest.mark.parametrize("name", list(GUARD_INDEPENDENT))
    def test_an_unrelated_refusal_still_holds(self, name: str) -> None:
        build, fragment = GUARD_INDEPENDENT[name]
        assert_refused_before_any_call(build, None, "", fragment)

    @pytest.mark.parametrize("name", list(IDENTIFY_ROWS))
    def test_identify_admits_it(self, name: str) -> None:
        build, estimand = IDENTIFY_ROWS[name]
        build().identify(estimand)

    @pytest.mark.parametrize("name", list(ESTIMATE_ROWS))
    def test_estimate_reaches_a_learner(self, name: str) -> None:
        estimate_witness(name)

    def test_the_refit_slot_admits_a_guarded_drtmle(
        self, unguarded_result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``guard=()`` fit whose copied estimator is set to ``guard=("Q", "g")`` refits."""
        estimator = copy.copy(unguarded_result.estimator)
        estimator.guard = ("Q", "g")
        reconfigured = replace(unguarded_result, estimator=estimator)
        assert estimator._refit_configuration_refusal(reconfigured.data) is None
        assert replayability(reconfigured).refit_nuisances

    def test_the_surface_replays_a_stratum_of_an_unguarded_fit(self) -> None:
        result = study().estimate(
            ATE(),
            DRTMLEMethod(guard=()),
            cross_fit=False,
            simultaneous=False,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.02))
        simulated_confounding(result, estimand="ate[S=1]", grid=grid, random_state=3)


#: Each stratified refusal that remains, its request and the fragments of its reason.
REMAINING: dict[str, tuple[Callable[[], Any], tuple[str, ...], type[Exception]]] = {
    "fold-wise targeting": (
        lambda: fit(
            TMLE(
                **spies(cross_fit=True, n_folds=2, targeting_scheme="fold", q_bounds=(-30.0, 30.0))
            ),
            strata_frame(),
            treatment="A",
        ),
        ("targeting_scheme='fold'", "fold-local update"),
        CapabilityError,
    ),
    "fold evaluation": (
        lambda: fit(
            TMLE(**spies(cross_fit=True, n_folds=2, cv_evaluation=True, q_bounds=(-30.0, 30.0))),
            strata_frame(),
            treatment="A",
        ),
        ("cv_evaluation=True", "stratum-indexed fold average", "docs/roadmap.md X28"),
        CapabilityError,
    ),
    "a DR-TMLE stratum with no trainable rows": (
        lambda: fit(
            DRTMLE(estimands=("ate",), **drtmle_spies()),
            strata_frame().assign(A=lambda f: np.where(f["S"] == 1, 0, f["A"])),
            treatment="A",
        ),
        ("inside baseline stratum S=1", "no trainable rows for a reduced regression"),
        ValueError,
    ),
}


class TestEachRemainingStratifiedRefusal:
    @pytest.mark.parametrize("name", list(REMAINING))
    def test_it_refuses_before_any_learner(self, name: str) -> None:
        build, fragments, error = REMAINING[name]
        NeverFit.calls = 0
        assert_refused_before_any_call(build, None, "", *fragments, error=error)

    def test_removing_the_drtmle_fold_check_lets_a_learner_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(drtmle_module, "refuse_untrainable_stratum_folds", lambda *a, **k: None)
        build, _, _ = REMAINING["a DR-TMLE stratum with no trainable rows"]
        reaches_a_learner(build)

    def test_removing_the_stratified_data_check_lets_a_learner_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An all-control stratum beside an incremental target reaches a learner."""
        monkeypatch.setattr(tmle_module, "check_stratified_targets", lambda *a, **k: None)
        frame = strata_frame().assign(A=lambda f: np.where(f["S"] == 1, 0, f["A"]))
        reaches_a_learner(
            lambda: fit(TMLE(incremental=[Incremental(2.0)], **spies()), frame, treatment="A")
        )


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
    "DRTMLE treatment_probabilities= with treatment_delta=": (
        lambda: _fit_missing_treatment(
            _never_drtmle(estimands=("ate",)), treatment_probabilities=np.full(200, 0.5)
        ),
        "treatment_probabilities= and treatment_delta= are not combined",
    ),
    "DRTMLE missing outcomes, three arms, observational, evaluation=": (
        lambda: DRTMLE(estimands=("ate",), evaluation=_trial(40), **drtmle_spies()).fit(
            _three_arms(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
        ),
        "the composite-indicator DR-TMLE for an observational missing outcome or treatment",
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
        assert "ate[S=1]" in self.names(
            fit(TMLE(**linear_in_sample()), strata_frame(), treatment="A")
        )

    def test_an_identity_link_msm(self) -> None:
        result = fit(TMLE(msm=MSM.linear(), **linear_in_sample()), strata_frame(), treatment="A")
        assert "msm[a][S=1]" in self.names(result)

    def test_regimes(self) -> None:
        estimator = TMLE(interventions=(Static(1), Static(0)), **linear_in_sample())
        assert "ey_regime[always 1][S=1]" in self.names(
            fit(estimator, strata_frame(), treatment="A")
        )

    def test_shifts(self) -> None:
        estimator = TMLE(policies=[Shift(0.5, cap=None)], density_bins=6, **linear_in_sample())
        assert "ey_policy[+0.5][S=1]" in self.names(fit_dose(estimator))

    def test_an_empty_drtmle_guard(self, unguarded_result: Any) -> None:
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


# ------------------------------------------------------ a declared missing treatment

#: What every missing-treatment refusal of the shared gate opens with.
SUPPORTED = "A missing treatment is supported for arm means and their contrasts"
UNIDENTIFIED = "P(A = a | W) is not identified"


def missing_treatment_frame(n: int = 200) -> pd.DataFrame:
    """:func:`strata_frame` with ``A`` missing where ``DeltaA = 0`` and a binary ``Z``."""
    frame = strata_frame(n)
    rng = np.random.default_rng(23)
    recorded = rng.random(n) < 0.8
    return frame.assign(
        A=np.where(recorded, frame["A"].astype(float), np.nan),
        DeltaA=recorded.astype(float),
        Z=rng.integers(0, 2, n).astype(float),
        Ybin=(frame["Y"] > frame["Y"].median()).astype(float),
    )


def _fit_missing_treatment(estimator: Any, **roles: Any) -> Any:
    outcome = roles.pop("outcome", "Y")
    return estimator.fit(
        missing_treatment_frame(),
        outcome=outcome,
        treatment="A",
        covariates=COVARIATES,
        treatment_delta="DeltaA",
        **roles,
    )


class _CustomEstimator(TMLE):
    """A subclass that the gate does not name: it meets the generic reason."""

    _assessment_method = "custom"


def _never(**settings: Any) -> dict[str, Any]:
    return {"cross_fit": False, "simultaneous": False, **never_fit_learners(), **settings}


def _never_drtmle(**settings: Any) -> DRTMLE:
    return DRTMLE(
        reduced_outcome_learner=NeverFit(),
        reduced_treatment_learner=NeverFit(),
        **_never(**settings),
    )


#: Every composition the gate refuses, with a fragment of its reason.  Each is refused
#: before any learner.
MISSING_TREATMENT_ROWS: dict[str, tuple[Callable[[], Any], tuple[str, ...]]] = {
    "att": (lambda: _fit_missing_treatment(TMLE(**_never(estimands=("att",)))), (UNIDENTIFIED,)),
    "atc": (lambda: _fit_missing_treatment(TMLE(**_never(estimands=("atc",)))), (UNIDENTIFIED,)),
    "ey_obs": (
        lambda: _fit_missing_treatment(TMLE(**_never(estimands=("ey_obs",)))),
        (UNIDENTIFIED, "ey_obs read the treatment law of every row"),
    ),
    "par": (lambda: _fit_missing_treatment(TMLE(**_never(estimands=("par",)))), (UNIDENTIFIED,)),
    "paf": (
        lambda: _fit_missing_treatment(TMLE(**_never(estimands=("paf",))), outcome="Ybin"),
        (UNIDENTIFIED,),
    ),
    "incremental": (
        lambda: _fit_missing_treatment(TMLE(**_never(incremental=[Incremental(2.0)]))),
        ("incremental= reads P(A | W)",),
    ),
    "learned_rule": (
        lambda: _fit_missing_treatment(TMLE(**_never(learned_rule=LearnedRule()))),
        ("no composite derivation for a rule learned from the fit", "roadmap F27"),
    ),
    "intermediate": (
        lambda: _fit_missing_treatment(TMLE(**_never()), intermediate="Z"),
        ("No derivation of the composite indicator for a controlled direct effect",),
    ),
    "cross_fit": (
        lambda: _fit_missing_treatment(TMLE(**_never(cross_fit=True))),
        ("roadmap F21", "cross_fit=False"),
    ),
    "randomized": (
        lambda: _fit_missing_treatment(_never_drtmle(randomized=True, estimands=("ate",))),
        ("observes the treatment on every row",),
    ),
    "ctmle": (
        lambda: _fit_missing_treatment(CTMLE(**_never(estimands=("ate",)))),
        ("roadmap F5", "Fit TMLE or DRTMLE"),
    ),
    "evaluation": (
        lambda: _fit_missing_treatment(
            _never_drtmle(estimands=("ate",), evaluation=strata_frame(50))
        ),
        ("evaluation companion",),
    ),
    "unnamed_estimator": (
        lambda: _fit_missing_treatment(_CustomEstimator(**_never(estimands=("ate",)))),
        ("This fit is not one of those compositions",),
    ),
}


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: TMLE(**_never(msm=MSM.linear())), id="msm"),
        pytest.param(
            lambda: TMLE(**_never(interventions=[Static(1.0, name="on"), Static(0.0, name="off")])),
            id="interventions",
        ),
    ],
)
def test_a_missing_treatment_admits_regimes_and_arm_msms(build: Callable[[], Any]) -> None:
    """Identified under the composite conditions, and lifted: the request reaches a learner."""
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        _fit_missing_treatment(build())
    assert NeverFit.calls > 0


@pytest.mark.parametrize("row", sorted(MISSING_TREATMENT_ROWS))
def test_a_missing_treatment_composition_is_refused_before_any_learner(row: str) -> None:
    build, fragments = MISSING_TREATMENT_ROWS[row]
    assert_refused_before_any_call(build, None, "learner", SUPPORTED, *fragments)


def test_a_missing_treatment_with_a_dose_is_refused_by_the_container() -> None:
    """A conditional density has no composite indicator; the container refuses first."""
    assert_refused_before_any_call(
        lambda: _fit_missing_treatment(TMLE(**_never(policies=[Shift(1.0, cap=None)]))),
        None,
        "learner",
        "arm-coded treatment only",
        error=DataError,
    )


def test_a_missing_treatment_beside_strata_at_a_guard_reaches_a_learner() -> None:
    reaches_a_learner(
        lambda: _fit_missing_treatment(_never_drtmle(estimands=("ate",)), strata=["S"])
    )


def test_the_composite_dr_tmle_refuses_the_evaluation_companion() -> None:
    """With an observational ``delta=`` alone the composite route refuses ``evaluation=``."""
    frame = _trial(120)
    assert_refused_before_any_call(
        lambda: _never_drtmle(estimands=("ate",), evaluation=frame).fit(
            frame, outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        ),
        None,
        "learner",
        "composite-indicator DR-TMLE",
        "evaluation companion",
    )


@pytest.mark.parametrize("row", ["att", "cross_fit", "ctmle"])
def test_a_missing_treatment_refusal_needs_the_gate(
    monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    """The mutation control: without the gate the request reaches a learner."""
    monkeypatch.setattr(tmle_module, "missing_treatment_refusal", lambda *args: None)
    build, fragments = MISSING_TREATMENT_ROWS[row]
    with pytest.raises(AssertionError):
        assert_refused_before_any_call(build, None, "learner", SUPPORTED, *fragments)
