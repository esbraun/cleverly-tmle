"""Stacked CV-TMLE of arm-indexed means and contrasts with missing-at-random outcomes.

``docs/technical-reference/point-treatment-tmle.md`` ("Stacked CV-TMLE for arm-indexed
targets") states the contract. This module holds its fast checks, grouped by the witness list
of the retired RM9 roadmap item:

* W12, refusals: every refusal row, before any learner call, through the engine keywords
  and through the public ``TMLEMethod`` route.
* W12, preflight: every sample and training-complement minimum, a ``DataError`` before any
  learner call, and the two-per-class rule for a package ``SuperLearner`` classification
  role.
* W1 to W11, numerical witnesses, in their own section below the refusals. Each one
  builds the correct value from stored arrays, asserts package equality, and asserts a
  stated gap for a deliberate mutation.
* W12, unequal folds: the point is the whole-sample mean, not an equal-fold average.
* W13, Super Learner leakage: a held-out outcome cannot move the inner split.

The refusal tests pin :class:`~tests.unit._natural_course_support.NeverFit` learners, so a
refusal that ran after a learner call would raise ``AssertionError`` instead.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    ATT,
    CapabilityError,
    CausalStudy,
    CollaborativeTMLEMethod,
    CrossFitting,
    DataError,
    Inference,
    MethodConfigurationError,
    ModelSpec,
    PointTreatment,
    SplitPlan,
    SuperLearner,
    Targeting,
    TMLEMethod,
)
from cleverly.data import CausalData
from cleverly.datasets import make_missing_outcome, make_missing_outcome_binary
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators._nuisance import fit_nuisances
from cleverly.learners import make_folds, random_partition
from cleverly.utils.bounds import OutcomeScaler
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The first clause of every contract refusal.
CONTRACT = (
    "Cross-fitted TMLE of arm-indexed means and contrasts with missing outcomes supports "
    "one audited stacked CV-TMLE contract; "
)

#: Rows in the refusal frames, and the outer folds every admitted setting declares.
N = 200
FOLDS = 3

#: A plan the package drew, which is the only kind any layer accepts. The contract refuses
#: it for being supplied, and a plan with no generator record would be refused earlier,
#: for a reason this contract row is not about.
_DRAWN_PLAN = SplitPlan.from_folds([random_partition(N, FOLDS, seed=0)])

#: The admitted engine settings that each refusal row departs from.
ADMITTED: dict[str, Any] = {
    "cross_fit": True,
    "n_folds": FOLDS,
    "repeats": 1,
    "stratify_folds": "none",
    "targeting_scheme": "pooled",
    "cv_evaluation": False,
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "n_bootstrap": 0,
    "simultaneous": False,
    "random_state": 1,
}


def _frame(*, continuous: bool = False) -> pd.DataFrame:
    """A two-arm MAR frame with the extra columns the row-structure refusals declare."""
    make = make_missing_outcome if continuous else make_missing_outcome_binary
    frame = make(n=N, seed=1)[0]
    rows = np.arange(N)
    return frame.assign(
        wt=np.where(rows % 2 == 0, 0.5, 1.5),
        cid=(rows // 2).astype(float),
        S=(rows % 2).astype(float),
    )


COVARIATES = ("W1", "W2", "W3")


# --------------------------------------------------------------------- W12: refusals


@dataclass(frozen=True)
class Row:
    """One refusal row of the contract table, in the order the fit checks it.

    ``engine`` and ``public`` hold the departure from the admitted settings in the two
    spellings. ``role`` names a design column the fit declares, as ``(engine keyword,
    PointTreatment field, value)``.
    """

    id: str
    fragment: str
    engine: dict[str, Any] = field(default_factory=dict)
    public: dict[str, dict[str, Any]] = field(default_factory=dict)
    role: tuple[str, str, Any] | None = None
    continuous: bool = False
    collaborative: bool = False


ROWS: tuple[Row, ...] = (
    Row("ctmle", "C-TMLE (CTMLE, or CollaborativeTMLEMethod)", collaborative=True),
    Row(
        "linear-fluctuation",
        "Set fluctuation='logistic' (Targeting(fluctuation='logistic'))",
        engine={"fluctuation": "linear"},
        public={"targeting": {"fluctuation": "linear"}},
    ),
    Row(
        "one-step",
        "Set targeting='iterative' (Targeting(algorithm='iterative'))",
        engine={"targeting": "one_step"},
        public={"targeting": {"algorithm": "one_step"}},
    ),
    Row(
        "target-weights",
        "Set target_weights=False (Targeting(target_weights=False))",
        engine={"target_weights": True},
        public={"targeting": {"target_weights": True}},
    ),
    Row(
        "fold-evaluation",
        "Set cv_evaluation=False (CrossFitting(fold_evaluation=False))",
        engine={"cv_evaluation": True},
        public={"cross_fitting": {"fold_evaluation": True}},
    ),
    # Reached through the estimand dimension of the parametrization.
    Row("att-atc", "The default estimand list and estimands='all' include att and atc"),
    Row(
        "continuous-without-q-bounds",
        "Declare q_bounds equal to the known outcome support (Targeting(q_bounds=(lower, upper)))",
        continuous=True,
    ),
    # Reached through the fold-strata dimension of the parametrization.
    Row("stratified-folds", "Set stratify_folds='none' (CrossFitting(stratify_by='none'))"),
    Row(
        "split-plan",
        "Leave split_plan=None (CrossFitting(split_plan=None))",
        engine={"split_plan": _DRAWN_PLAN},
        public={"cross_fitting": {"split_plan": _DRAWN_PLAN}},
    ),
    Row(
        "one-fold",
        "Set n_folds of at least 2 (CrossFitting(n_folds=...))",
        engine={"n_folds": 1},
        public={"cross_fitting": {"n_folds": 1}},
    ),
    Row(
        "repeats",
        "Set repeats=1 (CrossFitting(repeats=1))",
        engine={"repeats": 2},
        public={"cross_fitting": {"repeats": 2}},
    ),
    Row(
        "fold-targeting",
        "Set targeting_scheme='pooled' (CrossFitting(targeting_scheme='pooled'))",
        engine={"targeting_scheme": "fold"},
        public={"cross_fitting": {"targeting_scheme": "fold"}},
    ),
    Row(
        "weights",
        "Drop weights= from fit (PointTreatment(weights=None))",
        role=("weights", "weights", "wt"),
    ),
    Row(
        "clusters",
        "Drop id= from fit (PointTreatment(cluster=None))",
        role=("id", "cluster", "cid"),
    ),
    Row(
        "baseline-strata",
        "Drop strata= from fit (PointTreatment(strata=()))",
        role=("strata", "strata", ("S",)),
    ),
    Row(
        "bootstrap",
        "Set n_bootstrap=0 (Inference(n_bootstrap=0))",
        engine={"n_bootstrap": 2},
        public={"inference": {"n_bootstrap": 2}},
    ),
)
BY_ID = {row.id: row for row in ROWS}
ORDER = {row.id: index for index, row in enumerate(ROWS)}
#: The rows the parametrization reaches through a dimension rather than an override.
DIMENSION_ROWS = ("att-atc", "stratified-folds")
OVERRIDE_ROWS = tuple(row for row in ROWS if row.id not in DIMENSION_ROWS)
#: The override rows the contract's own check order still decides between.
#: ``stratify_folds != "none"`` (the "stratified-folds" row) and ``n_folds < 2`` (the
#: "one-fold" row) are refused engine-wide now, before this contract is reached, so
#: neither one's fixed-order position inside the contract can be exercised through it any
#: more. Each keeps its own dedicated test of the global refusal instead.
CONTRACT_ROWS = tuple(row for row in OVERRIDE_ROWS if row.id != "one-fold")


def _expected(row: Row | None, *, conditional: bool, stratify: str) -> Row:
    """The first contract row this composition breaks, in the fit's check order."""
    broken = [] if row is None else [row]
    if conditional:
        broken.append(BY_ID["att-atc"])
    if stratify != "none":
        broken.append(BY_ID["stratified-folds"])
    return min(broken, key=lambda item: ORDER[item.id])


#: The engine's estimand requests: one admitted list and three that include ``att``.
ENGINE_ESTIMANDS: tuple[Any, ...] = (("ate",), None, "all", ("att",))


def _engine_fit(row: Row | None, *, estimands: Any, stratify: str) -> None:
    settings: dict[str, Any] = {
        **ADMITTED,
        **never_fit_learners(),
        "estimands": estimands,
        "stratify_folds": stratify,
    }
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": COVARIATES,
        "delta": "Delta",
    }
    continuous = False
    engine = TMLE
    if row is not None:
        settings.update(row.engine)
        continuous = row.continuous
        if row.collaborative:
            engine = CTMLE
        if row.role is not None:
            keyword, _, value = row.role
            roles[keyword] = value
    frame = _frame(continuous=continuous)
    if row is not None and row.role is not None and row.role[0] == "strata":
        roles["covariates"] = (*COVARIATES, "S")
    engine(**settings).fit(frame, **roles)


@pytest.mark.parametrize("estimands", ENGINE_ESTIMANDS, ids=("ate", "default", "all", "att"))
@pytest.mark.parametrize("row", CONTRACT_ROWS, ids=[row.id for row in CONTRACT_ROWS])
def test_each_engine_refusal_row_fires_before_any_learner(row: Row, estimands: Any) -> None:
    """Each row refuses alone, and the fixed check order decides between several rows.

    ``stratify_folds`` is fixed at ``"none"`` here: ``"treatment"`` no longer reaches this
    contract at all, because :func:`cleverly.learners.crossfit.fold_strata_refusal` now
    refuses it engine-wide, before any capability check below runs (see
    ``test_stratified_folds_are_refused_before_the_contract_is_reached``).
    """
    conditional = estimands != ("ate",)
    if row.id == "ctmle" and conditional:
        # CTMLE's own estimand check now runs before this contract's collaborative-
        # estimator refusal (``CTMLE._resolve_estimands_for_data`` calls
        # ``_check_estimands`` before its superclass's contract check), so a conditional
        # estimand meets that refusal first rather than the contract's.
        with pytest.raises(ValueError, match=r"CTMLE does not support estimand\(s\)") as caught:
            _engine_fit(row, estimands=estimands, stratify="none")
        assert "clever covariates condition on a random event" in str(caught.value)
        assert NeverFit.calls == 0
        return
    expected = _expected(row, conditional=conditional, stratify="none")

    with pytest.raises(CapabilityError) as caught:
        _engine_fit(row, estimands=estimands, stratify="none")
    message = str(caught.value)
    assert message.startswith(CONTRACT)
    assert expected.fragment in message
    assert NeverFit.calls == 0


def test_att_and_atc_are_refused_by_name_in_every_request_form() -> None:
    for estimands in (None, "all", ("att",)):
        with pytest.raises(CapabilityError) as caught:
            _engine_fit(None, estimands=estimands, stratify="none")
        message = str(caught.value)
        assert BY_ID["att-atc"].fragment in message
        assert "Request estimands from ['ate', 'ey', 'ey1', 'ey0', 'rr', 'or']" in message
        requested = "['att']" if estimands == ("att",) else "['att', 'atc']"
        assert f"no audited result covers {requested}" in message
        assert NeverFit.calls == 0


def test_stratified_folds_are_refused_before_the_contract_is_reached() -> None:
    """``stratify_folds='treatment'`` is refused engine-wide now, not by this contract.

    The contract's own copy of this check is gone: cross-fitting with a data-dependent
    fold policy is refused for every estimator by
    :func:`cleverly.learners.crossfit.fold_strata_refusal`, at
    :class:`~cleverly.estimators.TMLE` construction, before the fit -- and this contract
    -- are ever reached.
    """
    with pytest.raises(
        ValueError, match=r"stratify_folds='treatment' requests stratification of the outer folds"
    ) as caught:
        _engine_fit(None, estimands=("ate",), stratify="treatment")
    assert "Set stratify_folds='none'" in str(caught.value)
    assert NeverFit.calls == 0


def test_crossed_folds_are_refused_before_the_contract_is_reached() -> None:
    with pytest.raises(
        ValueError,
        match=r"stratify_folds='treatment\+outcome' requests stratification of the outer folds",
    ) as caught:
        _engine_fit(None, estimands=("ate",), stratify="treatment+outcome")
    message = str(caught.value)
    assert "on the treatment and the outcome" in message
    assert "(docs/technical-reference/cv-tmle.md, fold and outcome-scale rules)" in message
    assert NeverFit.calls == 0


def test_the_att_refusal_names_the_admitted_list_for_three_arms() -> None:
    """``estimands='all'`` on three arms includes ``att``; the admitted list has no ``ey1``."""
    rng = np.random.default_rng(5)
    arms = np.arange(90) % 3
    observed = rng.random(90) < 0.7
    frame = pd.DataFrame(
        {
            "Y": np.where(observed, (rng.random(90) < 0.5).astype(float), np.nan),
            "A": arms.astype(float),
            "W1": rng.normal(size=90),
            "Delta": observed.astype(float),
        }
    )
    estimator = TMLE(**{**ADMITTED, **never_fit_learners(), "estimands": "all"})

    with pytest.raises(CapabilityError) as caught:
        estimator.fit(frame, outcome="Y", treatment="A", covariates=("W1",), delta="Delta")
    assert "Request estimands from ['ate', 'ey', 'rr', 'or']" in str(caught.value)
    assert NeverFit.calls == 0


#: The public route's estimand requests, and whether each includes ``att``.
PUBLIC_ESTIMANDS: tuple[tuple[Any, bool], ...] = ((ATE(), False), (ATT(), True))


def _public_fit(row: Row | None, *, estimand: Any, stratify: str) -> None:
    groups: dict[str, dict[str, Any]] = {
        "cross_fitting": {"enabled": True, "n_folds": FOLDS, "stratify_by": stratify},
        "targeting": {},
        "inference": {"simultaneous": False},
    }
    design: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "adjustment": COVARIATES,
        "missingness": "Delta",
    }
    continuous = False
    method_class: type[TMLEMethod] = TMLEMethod
    if row is not None:
        for group, overrides in row.public.items():
            groups[group] = {**groups[group], **overrides}
        continuous = row.continuous
        if row.collaborative:
            method_class = CollaborativeTMLEMethod
        if row.role is not None:
            _, name, value = row.role
            design[name] = value
            if name == "strata":
                design["adjustment"] = (*COVARIATES, "S")
    method = method_class(
        models=ModelSpec(**never_fit_learners()),
        cross_fitting=CrossFitting(**groups["cross_fitting"]),
        targeting=Targeting(**groups["targeting"]),
        inference=Inference(**groups["inference"]),
    )
    study = CausalStudy(_frame(continuous=continuous), design=PointTreatment(**design))
    study.identify(estimand).estimate(method=method)


@pytest.mark.parametrize(("estimand", "conditional"), PUBLIC_ESTIMANDS, ids=("ATE", "ATT"))
@pytest.mark.parametrize("row", CONTRACT_ROWS, ids=[row.id for row in CONTRACT_ROWS])
def test_each_public_refusal_row_fires_before_any_learner(
    row: Row, estimand: Any, conditional: bool
) -> None:
    """``stratify`` is fixed at ``"none"``; see ``test_each_engine_refusal_row_...``."""
    with pytest.raises(CapabilityError) as caught:
        _public_fit(row, estimand=estimand, stratify="none")
    message = str(caught.value)
    if row.collaborative and conditional:
        # The public route refuses a collaborative ATT before the engine sees it.
        assert message.startswith("method 'collaborative_tmle' cannot estimate ATT")
    else:
        expected = _expected(row, conditional=conditional, stratify="none")
        assert message.startswith(CONTRACT)
        assert expected.fragment in message
    assert NeverFit.calls == 0


def test_the_public_att_request_is_refused_by_name() -> None:
    with pytest.raises(CapabilityError) as caught:
        _public_fit(None, estimand=ATT(), stratify="none")
    assert BY_ID["att-atc"].fragment in str(caught.value)
    assert NeverFit.calls == 0


def test_the_public_stratified_folds_are_refused_before_the_contract_is_reached() -> None:
    """Declaring ``stratify_by='treatment'`` is refused engine-wide, not by the contract.

    ``CrossFitting()`` now defaults to ``stratify_by='none'``, so
    this declares the policy explicitly rather than relying on a default the contract
    used to catch.
    """
    with pytest.raises(
        MethodConfigurationError,
        match=r"stratify_folds='treatment' requests stratification of the outer folds",
    ) as caught:
        _public_fit(None, estimand=ATE(), stratify="treatment")
    assert "Set stratify_folds='none'" in str(caught.value)
    assert NeverFit.calls == 0


def test_a_single_fold_is_refused_before_the_contract_is_reached() -> None:
    """``n_folds=1`` under cross-fitting is refused engine-wide, in both spellings.

    The contract's own copy of this check is gone: the global check in
    ``_cross_fit_policy_refusal`` (:mod:`cleverly.learners.crossfit`) covers it before
    the arm-indexed contract, or any other capability check, is reached.
    """
    one_fold = BY_ID["one-fold"]
    with pytest.raises(ValueError, match=r"cross_fit=True with n_folds=1 leaves one fold"):
        _engine_fit(one_fold, estimands=("ate",), stratify="none")
    assert NeverFit.calls == 0

    with pytest.raises(
        MethodConfigurationError, match=r"enabled=True with n_folds=1 leaves one fold"
    ):
        _public_fit(one_fold, estimand=ATE(), stratify="none")
    assert NeverFit.calls == 0


def test_the_global_fold_policy_refusals_recur_at_fit_time_for_a_bypassed_estimator() -> None:
    """A copy that bypassed ``__init__`` meets the same refusal again, at fit time.

    :meth:`~cleverly.estimators.TMLE.refit` and a result restored from a pickle written
    by an earlier version both reach ``fit`` without running the constructor's checks, so
    ``_resolve_estimands_for_data`` asks ``_cross_fit_policy_reason`` again there.
    """
    roles = {"outcome": "Y", "treatment": "A", "covariates": COVARIATES, "delta": "Delta"}
    admitted = TMLE(**{**ADMITTED, **never_fit_learners(), "estimands": ("ate",)})

    stratified = copy.copy(admitted)
    stratified.stratify_folds = "treatment"
    with pytest.raises(
        ValueError, match=r"stratify_folds='treatment' requests stratification of the outer folds"
    ) as caught:
        stratified.fit(_frame(), **roles)
    assert "restored result or a copied estimator" in str(caught.value)
    assert NeverFit.calls == 0

    single_fold = copy.copy(admitted)
    single_fold.n_folds = 1
    with pytest.raises(
        ValueError, match=r"cross_fit=True with n_folds=1 leaves one fold"
    ) as caught:
        single_fold.fit(_frame(), **roles)
    assert "restored result or a copied estimator" in str(caught.value)
    assert NeverFit.calls == 0


def test_the_admitted_composition_reaches_the_learners() -> None:
    """The control for every refusal above: the admitted settings reach a learner fit.

    Without it, a contract that refused every fit would pass each refusal test.
    """
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _engine_fit(None, estimands=("ate",), stratify="none")
    assert NeverFit.calls == 1


def test_the_admitted_public_composition_reaches_the_learners() -> None:
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _public_fit(None, estimand=ATE(), stratify="none")
    assert NeverFit.calls == 1


def test_a_continuous_outcome_with_fixed_q_bounds_reaches_the_learners() -> None:
    frame = _frame(continuous=True)
    support = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
    estimator = TMLE(
        **{**ADMITTED, **never_fit_learners(), "estimands": ("ate",)}, q_bounds=support
    )

    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
    assert NeverFit.calls == 1


@pytest.mark.parametrize("offset", [0.0, 1e-6], ids=("constant", "nearly-constant"))
def test_a_supplied_weight_column_is_outside_the_unweighted_contract(offset: float) -> None:
    frame = _frame()
    frame["wt"] = 1.0 + offset * np.where(np.arange(N) % 2 == 0, -1.0, 1.0)
    estimator = TMLE(**{**ADMITTED, **never_fit_learners(), "estimands": ("ate",)})

    with pytest.raises(CapabilityError, match="Drop weights= from fit"):
        estimator.fit(
            frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta", weights="wt"
        )
    assert NeverFit.calls == 0


def test_unstratified_folds_are_selectable_for_the_in_sample_fit() -> None:
    """``"none"`` is no longer reserved, so the in-sample fit accepts it and reports.

    The contract governs cross-fitted fits. This one declares the same fold policy with
    ``cross_fit=False``, reaches the learners, and returns a finite estimate. One fold
    balances nothing, so the policy changes no assignment here.
    """
    estimator = TMLE(
        **{**ADMITTED, "estimands": ("ate",), "cross_fit": False},
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=LogisticRegression(),
    )

    result = estimator.fit(
        _frame(), outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta"
    ).single()

    assert np.isfinite(result["ate"].psi)
    assert result.config.crossfit.scheme == "none"


def test_unstratified_folds_are_selectable_for_a_complete_outcome_cross_fit() -> None:
    """Out of sample too: a complete-outcome fit draws an unstratified partition.

    The plan the result records names the unstratified generator, which is the evidence
    that the policy reached fold generation rather than being ignored.
    """
    frame = _frame().drop(columns=["Delta"]).dropna(subset=["Y"])
    estimator = TMLE(
        **{**ADMITTED, "estimands": ("ate",), "n_folds": FOLDS},
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
    )

    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=COVARIATES).single()

    assert np.isfinite(result["ate"].psi)
    assert result.config.crossfit.scheme == "vfold"
    assert result.config.crossfit.stratify_by == ()
    provenance = result.split_plan.provenance
    assert provenance is not None
    assert provenance[0].scheme == "vfold"


# --------------------------------------------------------------------- W12: preflight

#: Rows in the preflight frames. Three folds make each training complement the union of
#: two validation folds, so a check that read a validation fold instead of its complement
#: names a different fold, or refuses a partition that fits.
PREFLIGHT_N = 60
PREFLIGHT_SEED = 17
LAST = FOLDS - 1


def _preflight_folds() -> Any:
    return make_folds(PREFLIGHT_N, FOLDS, stratify=None, random_state=PREFLIGHT_SEED)


def _preflight_estimator(**learners: Any) -> TMLE:
    return TMLE(
        **{
            **ADMITTED,
            **never_fit_learners(),
            **learners,
            "estimands": ("ate",),
            "random_state": PREFLIGHT_SEED,
        }
    )


def _preflight_fit(frame: pd.DataFrame, **learners: Any) -> Any:
    return (
        _preflight_estimator(**learners)
        .fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")
        .single()
    )


def _preflight_frame(case: str) -> pd.DataFrame:
    """A 60-row MAR frame whose only shortfall is ``case``.

    The base frame spreads both arms, both response kinds, and both outcome classes over
    every fold, with at least two of each class in each training complement. Each case
    moves one class into the last validation fold, or down to one row in the sample.
    """
    rows = np.arange(PREFLIGHT_N)
    last = _preflight_folds().test_index(LAST)
    rest = np.setdiff1d(rows, last)
    arm = (rows % 2).astype(float)
    observed = rows % 5 != 0
    outcome = ((rows // 2) % 2).astype(float)

    if case == "sample-respondent":
        observed = np.zeros(PREFLIGHT_N, dtype=bool)
        observed[1] = True
    elif case == "sample-nonrespondent":
        observed = np.ones(PREFLIGHT_N, dtype=bool)
        observed[0] = False
    elif case == "sample-arm-respondent":
        treated = np.flatnonzero(arm == 1.0)
        observed[treated] = False
        observed[treated[0]] = True
    elif case == "complement-respondent":
        observed = np.zeros(PREFLIGHT_N, dtype=bool)
        observed[last[:4]] = True
        arm[last[:4]] = [0.0, 1.0, 0.0, 1.0]
        outcome[last[:4]] = [0.0, 0.0, 1.0, 1.0]
    elif case == "complement-nonrespondent":
        observed = np.ones(PREFLIGHT_N, dtype=bool)
        observed[last[:2]] = False
    elif case == "complement-arm":
        arm = np.zeros(PREFLIGHT_N)
        arm[last[:4]] = 1.0
        observed[last[:4]] = True
        outcome[last[:4]] = [0.0, 1.0, 0.0, 1.0]
    elif case == "complement-arm-respondent":
        observed[(arm == 1.0) & np.isin(rows, rest)] = False
        observed[last[arm[last] == 1.0][:2]] = True
    elif case == "complement-outcome":
        outcome[rest] = 0.0
        outcome[last[:2]] = 1.0
        observed[last[:2]] = True
    elif case == "sl-outcome":
        outcome[:] = 0.0
        outcome[last[:2]] = 1.0
        observed[last[:2]] = True
        outcome[rest[1]] = 1.0
        observed[rest[1]] = True
    elif case == "sl-response":
        observed = np.ones(PREFLIGHT_N, dtype=bool)
        observed[last[:2]] = False
        observed[rest[0]] = False
    elif case == "sl-treatment":
        arm = np.zeros(PREFLIGHT_N)
        arm[last[:2]] = 1.0
        observed[last[:2]] = True
        outcome[last[:2]] = [0.0, 1.0]
        arm[rest[1]] = 1.0
        observed[rest[1]] = True
    elif case == "sample-outcome":
        outcome[:] = 0.0
        outcome[1] = 1.0
    elif case == "sample-outcome-none":
        outcome[:] = 0.0
    elif case == "sl-sample-outcome":
        outcome[:] = 0.0
        outcome[[1, 2]] = 1.0
    elif case == "sl-sample-response":
        observed = np.ones(PREFLIGHT_N, dtype=bool)
        observed[[0, 1]] = False
    elif case == "sl-sample-treatment":
        # One arm-1 respondent inside the last fold and one outside it, so every
        # complement holds one and only the Super Learner rule can refuse.
        arm = np.zeros(PREFLIGHT_N)
        arm[[last[0], rest[0]]] = 1.0
        observed[[last[0], rest[0]]] = True
    elif case != "base":
        raise ValueError(case)
    return pd.DataFrame(
        {
            "Y": np.where(observed, outcome, np.nan),
            "A": arm,
            "W": rows.astype(float),
            "Delta": observed.astype(float),
        }
    )


@pytest.mark.parametrize(
    ("case", "fragment"),
    [
        ("sample-respondent", "the sample has 1 respondent(s)."),
        ("sample-nonrespondent", "the sample has 1 nonrespondent(s)."),
        ("sample-arm-respondent", "the sample has 1 respondent(s) in arm 1."),
    ],
)
def test_a_sample_below_the_minimum_refuses_before_any_learner(case: str, fragment: str) -> None:
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame(case))
    message = str(caught.value)
    assert fragment in message
    assert message.endswith(
        "fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))"
    )
    assert NeverFit.calls == 0


@pytest.mark.parametrize(
    ("case", "absent"),
    [
        ("complement-respondent", "respondent"),
        ("complement-nonrespondent", "nonrespondent"),
        ("complement-arm", "row in arm 1"),
        ("complement-arm-respondent", "respondent in arm 1"),
        ("complement-outcome", "respondent with outcome 1"),
    ],
)
def test_a_training_complement_below_the_minimum_refuses_before_any_learner(
    case: str, absent: str
) -> None:
    """Only the last fold's complement is short, so the message names that fold."""
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame(case))
    message = str(caught.value)
    assert f"repeat 0, fold {LAST}'s training complement contains no {absent}." in message
    assert (
        "trying fold counts or seeds until one fits would choose the partition by the "
        "values it must not read" in message
    )
    assert message.endswith(
        "fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False)), "
        "or collect more observations at the rare level."
    )
    assert NeverFit.calls == 0


def test_one_respondent_with_an_outcome_is_a_sample_shortfall() -> None:
    """One respondent with outcome 1 leaves its fold's complement without it in every partition."""
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame("sample-outcome"))
    message = str(caught.value)
    assert "the sample has 1 respondent(s) with outcome 1." in message
    assert "no fold count or random_state can fit the outcome regression" in message
    assert message.endswith("(CrossFitting(enabled=False))")
    assert NeverFit.calls == 0


def test_one_outcome_class_offers_no_remedy_that_cannot_succeed() -> None:
    """With no respondent at outcome 1, neither a new partition nor an in-sample fit helps."""
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame("sample-outcome-none"))
    message = str(caught.value)
    assert "no respondent has outcome 1. No partition can supply it" in message
    assert "an in-sample fit trains the outcome learner on the same single class." in message
    assert "Increase n_folds" not in message
    assert "cross_fit=False" not in message
    assert NeverFit.calls == 0


@pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
def test_the_in_sample_remedy_fits_one_respondent_with_an_outcome() -> None:
    """The remedy the sample shortfall names is achievable: the in-sample fit succeeds.

    One outcome-1 row separates the logistic fluctuation, and the fit warns about it. The
    test asks only that the named remedy returns an estimate.
    """
    frame = _preflight_frame("sample-outcome")
    estimator = TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=LogisticRegression(),
        cross_fit=False,
        stratify_folds="none",
        estimands=("ate",),
        simultaneous=False,
    )
    result = estimator.fit(
        frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta"
    ).single()
    assert np.isfinite(result["ate"].psi)


def test_the_preflight_folds_are_the_fitted_folds() -> None:
    """The fixtures above place rows by these folds, so the fit must draw the same ones."""
    estimator = _preflight_estimator(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=LogisticRegression(),
    )
    result = estimator.fit(
        _preflight_frame("base"), outcome="Y", treatment="A", covariates=("W",), delta="Delta"
    ).single()
    np.testing.assert_array_equal(result.nuisance.folds.assignment, _preflight_folds().assignment)


def test_the_base_preflight_frame_reaches_the_learners() -> None:
    """The control for every preflight case: the base frame passes every check."""
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _preflight_fit(_preflight_frame("base"))
    assert NeverFit.calls == 1


def _never_fit_super_learner(
    task: Literal["classification"] | None = "classification",
) -> SuperLearner:
    return SuperLearner(library=[("never", NeverFit())], task=task, n_folds=2)


#: Each Super Learner case: its frame, the role under test, and the short class.
SUPER_LEARNER_CASES = (
    ("sl-outcome", "outcome_learner", "outcome", "1 respondent(s) with outcome 1"),
    ("sl-treatment", "treatment_learner", "treatment", "1 row(s) in arm 1"),
    ("sl-response", "missingness_learner", "response", "1 nonrespondent(s)"),
)


@pytest.mark.parametrize(
    ("case", "keyword", "role", "short"),
    SUPER_LEARNER_CASES,
    ids=[case[2] for case in SUPER_LEARNER_CASES],
)
def test_a_super_learner_role_needs_two_rows_per_class_in_each_complement(
    case: str, keyword: str, role: str, short: str
) -> None:
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame(case), **{keyword: _never_fit_super_learner()})
    message = str(caught.value)
    assert f"cannot fit the {role} learner because repeat 0, fold {LAST}'s" in message
    assert f"training complement holds {short}." in message
    assert (
        "is a package SuperLearner with a classification task, and its inner "
        "stratified split needs at least two rows in each class of its target in each "
        "training complement." in message
    )
    assert NeverFit.calls == 0


@pytest.mark.parametrize(
    ("case", "keyword"),
    [(case, keyword) for case, keyword, _, _ in SUPER_LEARNER_CASES],
    ids=[case[2] for case in SUPER_LEARNER_CASES],
)
def test_the_two_per_class_rule_reads_the_resolved_learner(case: str, keyword: str) -> None:
    """The control for the rule above: a learner that is not a Super Learner passes."""
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _preflight_fit(_preflight_frame(case))
    assert NeverFit.calls == 1


#: Each Super Learner sample case: its frame, the role under test, and the short class.
SUPER_LEARNER_SAMPLE_CASES = (
    ("sl-sample-outcome", "outcome_learner", "outcome", "2 respondent(s) with outcome 1"),
    ("sl-sample-treatment", "treatment_learner", "treatment", "2 row(s) in arm 1"),
    ("sl-sample-response", "missingness_learner", "response", "2 nonrespondent(s)"),
)


@pytest.mark.parametrize(
    ("case", "keyword", "role", "short"),
    SUPER_LEARNER_SAMPLE_CASES,
    ids=[case[2] for case in SUPER_LEARNER_SAMPLE_CASES],
)
def test_a_super_learner_class_of_two_is_a_sample_shortfall(
    case: str, keyword: str, role: str, short: str
) -> None:
    """Two rows in a class leave some complement with at most one in every partition.

    The complement cases above hold three rows of the short class in the sample, so they
    reach the per-complement rule instead: three is the sample minimum.
    """
    with pytest.raises(DataError) as caught:
        _preflight_fit(_preflight_frame(case), **{keyword: _never_fit_super_learner()})
    message = str(caught.value)
    assert f"cannot fit the {role} learner: the sample holds {short}." in message
    assert "no fold count or random_state can fit it." in message
    assert "Increase n_folds" not in message
    assert message.endswith(
        f"Replace the {role} learner with one that is not a package classification "
        "SuperLearner, or "
        "fit in sample with cross_fit=False on the engine (CrossFitting(enabled=False))."
    )
    assert NeverFit.calls == 0


@pytest.mark.parametrize(
    ("case", "keyword"),
    [(case, keyword) for case, keyword, _, _ in SUPER_LEARNER_SAMPLE_CASES],
    ids=[case[2] for case in SUPER_LEARNER_SAMPLE_CASES],
)
def test_the_sample_rule_reads_the_resolved_learner(case: str, keyword: str) -> None:
    """The control for the rule above: a learner that is not a Super Learner passes."""
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _preflight_fit(_preflight_frame(case))
    assert NeverFit.calls == 1


def test_the_in_sample_remedy_fits_a_super_learner_class_of_two() -> None:
    """The in-sample remedy is achievable: the Super Learner's own split sees both rows."""
    estimator = TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=SuperLearner(
            library=[("glm", LogisticRegression())], task="classification", n_folds=2
        ),
        cross_fit=False,
        stratify_folds="treatment",
        estimands=("ate",),
        simultaneous=False,
    )
    result = estimator.fit(
        _preflight_frame("sl-sample-response"),
        outcome="Y",
        treatment="A",
        covariates=("W",),
        delta="Delta",
    ).single()
    assert np.isfinite(result["ate"].psi)


def test_a_super_learner_without_a_task_on_a_binary_target_counts() -> None:
    """``task=None`` infers classification from a 0/1 target, so its split stratifies."""
    with pytest.raises(DataError, match="cannot fit the response learner"):
        _preflight_fit(
            _preflight_frame("sl-response"), missingness_learner=_never_fit_super_learner(None)
        )
    assert NeverFit.calls == 0


def test_the_default_learner_is_a_classification_super_learner() -> None:
    """``outcome_learner=None`` resolves to the package Super Learner for a binary outcome."""
    with pytest.raises(DataError, match="cannot fit the outcome learner"):
        _preflight_fit(_preflight_frame("sl-outcome"), outcome_learner=None)
    assert NeverFit.calls == 0


def _two_valued_continuous_fit(q_bounds: tuple[float, float], case: str = "sl-outcome") -> Any:
    """The ``case`` frame with its outcome moved from {0, 1} to {2, 5}."""
    frame = _preflight_frame(case)
    frame["Y"] = 2.0 + 3.0 * frame["Y"]
    estimator = TMLE(
        **{
            **ADMITTED,
            **never_fit_learners(),
            "outcome_learner": _never_fit_super_learner(None),
            "estimands": ("ate",),
            "random_state": PREFLIGHT_SEED,
        },
        family="gaussian",
        q_bounds=q_bounds,
    )
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")


def test_a_task_free_outcome_learner_infers_its_task_from_the_scaled_outcome() -> None:
    """``q_bounds=(2, 5)`` scales the outcome to {0, 1}, so the Super Learner classifies.

    Before the fix the preflight inferred the task from the raw {2, 5} outcome, skipped
    the role, and the fit raised a raw ``ValueError`` inside the Super Learner after the
    treatment and response learners were already fitted.
    """
    with pytest.raises(DataError) as caught:
        _two_valued_continuous_fit((2.0, 5.0))
    message = str(caught.value)
    assert f"cannot fit the outcome learner because repeat 0, fold {LAST}'s" in message
    assert "training complement holds 1 respondent(s) with outcome 5." in message
    assert NeverFit.calls == 0


def test_a_super_learner_class_of_one_offers_no_in_sample_remedy() -> None:
    """An in-sample Super Learner cannot split one row either, so only a learner change helps."""
    with pytest.raises(DataError) as caught:
        _two_valued_continuous_fit((2.0, 5.0), case="sample-outcome")
    message = str(caught.value)
    assert "cannot fit the outcome learner: the sample holds 1 respondent(s) with outcome 5." in (
        message
    )
    assert message.endswith(
        "Replace the outcome learner with one that is not a package classification SuperLearner."
    )
    assert NeverFit.calls == 0


def test_a_task_free_outcome_learner_on_a_wider_support_regresses() -> None:
    """The control: ``q_bounds=(1, 6)`` scales {2, 5} away from {0, 1}, so no rule applies."""
    with pytest.raises(AssertionError, match="must run before any learner is fitted"):
        _two_valued_continuous_fit((1.0, 6.0))
    assert NeverFit.calls == 1


def _fold_dependent_task_frame(*, keep_middle_in_complement: bool) -> pd.DataFrame:
    """Give the last complement {0, 1}, or retain a middle level for regression."""
    frame = _preflight_frame("base")
    last = _preflight_folds().test_index(LAST)
    outside = frame.index[frame["Delta"].eq(1) & ~frame.index.isin(last)]
    inside = frame.index[frame["Delta"].eq(1) & frame.index.isin(last)]
    frame["Y"] = np.where(frame["Delta"].eq(1), 0.0, np.nan)
    frame.loc[outside[0], "Y"] = 1.0
    frame.loc[inside[0], "Y"] = 0.5
    if keep_middle_in_complement:
        frame.loc[outside[1], "Y"] = 0.5
    return frame


@pytest.mark.parametrize("keep_middle_in_complement", [False, True])
def test_task_free_super_learner_uses_each_training_complements_task(
    keep_middle_in_complement: bool,
) -> None:
    """A complement can classify even when the full continuous sample regresses."""
    estimator = TMLE(
        **{
            **ADMITTED,
            **never_fit_learners(),
            "outcome_learner": _never_fit_super_learner(None),
            "estimands": ("ate",),
            "random_state": PREFLIGHT_SEED,
        },
        family="gaussian",
        q_bounds=(0.0, 1.0),
    )
    frame = _fold_dependent_task_frame(keep_middle_in_complement=keep_middle_in_complement)
    if keep_middle_in_complement:
        with pytest.raises(AssertionError, match="must run before any learner is fitted"):
            estimator.fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")
        assert NeverFit.calls == 1
    else:
        with pytest.raises(DataError) as caught:
            estimator.fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")
        assert f"cannot fit the outcome learner because repeat 0, fold {LAST}'s" in str(
            caught.value
        )
        assert "holds 1 respondent(s) with outcome 1" in str(caught.value)
        assert NeverFit.calls == 0


class _RecordingSuperLearner(SuperLearner):
    """A Super Learner that records the classes in each of its inner training sets."""

    classes: ClassVar[list[tuple[float, ...]]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None, groups: Any = None) -> Any:
        fitted = super().fit(X, y, sample_weight=sample_weight, groups=groups)
        target = np.asarray(y, dtype=float)
        for train, _ in self.folds_:
            type(self).classes.append(tuple(np.unique(target[train]).tolist()))
        return fitted


def test_at_two_rows_per_class_every_inner_training_set_holds_both_classes() -> None:
    """The response learner meets exactly two nonrespondents in one complement."""
    frame = _preflight_frame("sl-response")
    # ``sl-response`` leaves one nonrespondent, ``rest[0]``, outside the last fold. A
    # second one outside it brings the last fold's complement to exactly two.
    rest = np.setdiff1d(np.arange(PREFLIGHT_N), _preflight_folds().test_index(LAST))
    frame.loc[rest[2], ["Delta", "Y"]] = [0.0, np.nan]
    _RecordingSuperLearner.classes = []
    learner = _RecordingSuperLearner(
        library=[("glm", LogisticRegression())], task="classification", n_folds=2
    )

    result = _preflight_fit(
        frame,
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=learner,
    )

    assert np.isfinite(result["ate"].psi)
    assert len(_RecordingSuperLearner.classes) == FOLDS * 2
    assert all(classes == (0.0, 1.0) for classes in _RecordingSuperLearner.classes)


def _probe_a_frame() -> pd.DataFrame:
    """RM9 probe A: one nonrespondent, and every missing outcome set to 0."""
    frame = make_missing_outcome_binary(n=200, seed=3)[0]
    delta = np.ones(len(frame))
    delta[0] = 0.0
    return frame.assign(Delta=delta, Y=frame["Y"].fillna(0.0))


def test_probe_a_is_refused_before_any_learner_through_the_engine() -> None:
    """Before the preflight, probe A failed inside scikit-learn after nine learner fits."""
    estimator = TMLE(**{**ADMITTED, **never_fit_learners(), "estimands": ("ate",), "n_folds": 5})

    with pytest.raises(DataError, match=r"the sample has 1 nonrespondent\(s\)"):
        estimator.fit(_probe_a_frame(), outcome="Y", treatment="A", delta="Delta")
    assert NeverFit.calls == 0


def test_probe_a_is_refused_before_any_learner_through_the_public_method() -> None:
    study = CausalStudy(
        _probe_a_frame(),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=COVARIATES, missingness="Delta"
        ),
    )
    method = TMLEMethod(
        models=ModelSpec(**never_fit_learners()),
        cross_fitting=CrossFitting(n_folds=5, stratify_by="none"),
        inference=Inference(simultaneous=False),
    )

    with pytest.raises(DataError, match=r"the sample has 1 nonrespondent\(s\)"):
        study.identify(ATE()).estimate(method=method)
    assert NeverFit.calls == 0


# --------------------------------------------------------------------- W1 to W11
#
# Each witness builds the correct value from stored arrays or from a hand refit, asserts
# package equality, and asserts a stated gap for a deliberate mutation built from the
# same arrays. The frames are seeded, so every gap is a fixed number; each margin sits
# below the measured gap by a factor of at least three.

#: Rows, seed, and covariates of the witness frames. Three unstratified folds of 601
#: rows hold 201, 200, and 200 rows, so the fold weights are unequal.
WITNESS_N = 601
WITNESS_SEED = 11
WITNESS_COVARIATES = ("W1", "W2")
#: The prediction bound of the logistic submodel, ``TMLE(alpha=0.9995)``.
ALPHA = 0.9995
#: The shift of each arm on the logit of the outcome.
ARM_SHIFT = (0.0, 0.5, -0.4)
#: The three-arm estimand request of the contrast witnesses.
ALL_CONTRASTS = ("ey", "ate", "rr", "or")


def _witness_frame(n_arms: int = 3) -> pd.DataFrame:
    """A seeded MAR frame whose outcome regression a main-terms logistic model misses.

    The outcome logit is nonlinear in ``W1`` and every arm shares it. Its steep linear
    part makes the plug-in terms of the arm curves move together, so the arm curves have a
    positive same-row covariance. The response probability depends on ``A`` and on
    ``W``. ``Y`` is ``NaN`` where ``Delta`` is 0.
    """
    rng = np.random.default_rng(WITNESS_SEED)
    w1 = rng.uniform(-2.0, 2.0, size=WITNESS_N)
    w2 = rng.normal(size=WITNESS_N)
    logits = np.column_stack([np.zeros(WITNESS_N), 0.4 * w1 - 0.2, -0.5 * w1 + 0.3 * w2])
    probability = np.exp(logits[:, :n_arms])
    probability /= probability.sum(axis=1, keepdims=True)
    draw = rng.random(WITNESS_N)[:, None]
    arm = (draw > np.cumsum(probability, axis=1)).sum(axis=1).astype(float)
    curve = 2.0 * w1 + 0.5 * (1.5 * np.sin(2.0 * w1) + w1**2 - 0.8)
    outcome_logit = curve + np.asarray(ARM_SHIFT)[arm.astype(int)]
    outcome = (rng.random(WITNESS_N) < expit(outcome_logit)).astype(float)
    respond = rng.random(WITNESS_N) < expit(1.5 + 0.6 * w1 - 0.3 * arm + 0.4 * w2)
    return pd.DataFrame(
        {
            "Y": np.where(respond, outcome, np.nan),
            "A": arm,
            "W1": w1,
            "W2": w2,
            "Delta": respond.astype(float),
        }
    )


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _witness_fit(
    frame: pd.DataFrame, *, estimands: tuple[str, ...] = ("ey",), **overrides: Any
) -> Any:
    """Fit the admitted composition with main-terms logistic learners for each role."""
    settings: dict[str, Any] = {
        **ADMITTED,
        "estimands": estimands,
        "outcome_learner": _logistic(),
        "treatment_learner": _logistic(),
        "missingness_learner": _logistic(),
        **overrides,
    }
    return (
        TMLE(**settings)
        .fit(frame, outcome="Y", treatment="A", covariates=WITNESS_COVARIATES, delta="Delta")
        .single()
    )


def _newton(offset: np.ndarray, covariate: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """The logistic MLE of ``outcome`` on ``covariate`` with a fixed offset and no intercept."""
    epsilon = np.zeros(covariate.shape[1])
    for _ in range(50):
        fitted = expit(offset + covariate @ epsilon)
        gradient = covariate.T @ (outcome - fitted)
        hessian = (covariate * (fitted * (1.0 - fitted))[:, None]).T @ covariate
        step = np.linalg.solve(hessian, gradient)
        epsilon = epsilon + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return epsilon


def _treatment_block(values: np.ndarray, arms: tuple[float, ...]) -> np.ndarray:
    """One indicator column for each non-reference arm, as the outcome design holds."""
    return np.column_stack([values == arm for arm in arms[1:]]).astype(float)


@dataclass(frozen=True)
class HandNuisances:
    """Stitched nuisance predictions refitted by hand, one row per sample row.

    Parameters
    ----------
    outcome : dict of float to ndarray
        ``Q(a, W)`` for each arm.
    realised : ndarray
        ``Q(A, W)`` at each row's own arm.
    propensity : ndarray
        ``g(a | W)``, one column per arm.
    response : ndarray
        ``P(Delta = 1 | A = a, W)``, one column per arm.
    """

    outcome: dict[float, np.ndarray]
    realised: np.ndarray
    propensity: np.ndarray
    response: np.ndarray


def _hand_nuisances(
    frame: pd.DataFrame, arms: tuple[float, ...], splits: list[tuple[np.ndarray, np.ndarray]]
) -> HandNuisances:
    """Refit each role on each training set and predict on its validation rows."""
    n = len(frame)
    arm = frame["A"].to_numpy(dtype=float)
    covariates = frame[list(WITNESS_COVARIATES)].to_numpy(dtype=float)
    observed = frame["Delta"].to_numpy() == 1.0
    outcome = frame["Y"].fillna(0.0).to_numpy(dtype=float)
    design = np.column_stack([_treatment_block(arm, arms), covariates])
    at = {
        level: np.column_stack([_treatment_block(np.full(n, level), arms), covariates])
        for level in arms
    }
    predictions = {level: np.empty(n) for level in arms}
    realised = np.empty(n)
    propensity = np.empty((n, len(arms)))
    response = np.empty((n, len(arms)))
    for train, test in splits:
        rows = train[observed[train]]
        outcome_model = _logistic().fit(design[rows], outcome[rows])
        treatment_model = _logistic().fit(covariates[train], arm[train])
        response_model = _logistic().fit(design[train], observed[train].astype(float))
        realised[test] = outcome_model.predict_proba(design[test])[:, 1]
        propensity[test] = treatment_model.predict_proba(covariates[test])
        for column, level in enumerate(arms):
            predictions[level][test] = outcome_model.predict_proba(at[level][test])[:, 1]
            response[test, column] = response_model.predict_proba(at[level][test])[:, 1]
    return HandNuisances(predictions, realised, propensity, response)


@dataclass(frozen=True)
class HandTarget:
    """The hand-built fluctuation, points, and curves on the ``[0, 1]`` scale.

    Parameters
    ----------
    epsilon : ndarray
        One coefficient per arm.
    targeted : dict of float to ndarray
        ``Q*(a, W)`` for each arm.
    psi : dict of float to float
        The plug-in mean of each arm.
    curve : dict of float to ndarray
        The influence curve of each arm mean.
    """

    epsilon: np.ndarray
    targeted: dict[float, np.ndarray]
    psi: dict[float, float]
    curve: dict[float, np.ndarray]


def _inverse(
    propensity: np.ndarray, response: np.ndarray, g_bounds: tuple[float, float], lower: float
) -> np.ndarray:
    """``1 / (g_a pi_a)`` per arm, after the package's truncation of each factor."""
    return 1.0 / (np.clip(propensity, *g_bounds) * np.clip(response, lower, 1.0))


def _hand_target(
    frame: pd.DataFrame,
    nuisances: HandNuisances,
    arms: tuple[float, ...],
    g_bounds: tuple[float, float],
    lower: float,
) -> HandTarget:
    """Fit the joint fluctuation on the respondent rows and build each arm's curve."""
    arm = frame["A"].to_numpy(dtype=float)
    observed = frame["Delta"].to_numpy() == 1.0
    outcome = frame["Y"].fillna(0.0).to_numpy(dtype=float)
    inverse = _inverse(nuisances.propensity, nuisances.response, g_bounds, lower)
    covariate = (arm[:, None] == np.asarray(arms)[None, :]) * inverse
    epsilon = _newton(logit(nuisances.realised[observed]), covariate[observed], outcome[observed])
    realised = expit(logit(nuisances.realised) + covariate @ epsilon)
    residual = np.where(observed, outcome - realised, 0.0)
    targeted = {
        level: expit(logit(nuisances.outcome[level]) + epsilon[column] * inverse[:, column])
        for column, level in enumerate(arms)
    }
    psi = {level: float(np.mean(values)) for level, values in targeted.items()}
    curve = {
        level: covariate[:, column] * residual + targeted[level] - psi[level]
        for column, level in enumerate(arms)
    }
    return HandTarget(epsilon, targeted, psi, curve)


def _stored_inverse(result: Any) -> np.ndarray:
    """``1 / (g_a pi_a)`` from the stored nuisances, truncated as the fit truncated them."""
    return _inverse(
        np.asarray(result.nuisance.propensity.values, dtype=float),
        np.asarray(result.nuisance.missingness, dtype=float),
        result.config.g_bounds,
        result.config.missingness_bound,
    )


def _arm_index(result: Any) -> np.ndarray:
    """Each row's column in the arm-ordered nuisance arrays."""
    return np.searchsorted(np.asarray(result.nuisance.arms), result.data.treatment)


def _assert_inside_the_logistic_bound(result: Any) -> None:
    """The submodel clips predictions to ``[1 - alpha, alpha]``; the witnesses need no clip."""
    fluctuation = result.fluctuations["mean"]
    for fit in (result.nuisance.outcome, fluctuation.targeted):
        for values in (fit.observed, *fit.arms.values()):
            assert np.all((values > 1.0 - ALPHA) & (values < ALPHA))


def test_w1_the_stitched_nuisances_fluctuation_points_and_curves_rebuild_by_hand() -> None:
    frame = _witness_frame(2)
    result = _witness_fit(frame, estimands=("ey1", "ey0", "ate"))
    arms = (0.0, 1.0)
    splits = list(result.nuisance.folds)
    hand = _hand_nuisances(frame, arms, splits)
    _assert_inside_the_logistic_bound(result)

    nuisance = result.nuisance
    np.testing.assert_allclose(nuisance.outcome.observed, hand.realised, rtol=0.0, atol=1e-8)
    for level in arms:
        np.testing.assert_allclose(
            nuisance.outcome.arms[level], hand.outcome[level], rtol=0.0, atol=1e-8
        )
    np.testing.assert_allclose(nuisance.propensity.values, hand.propensity, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(nuisance.missingness, hand.response, rtol=0.0, atol=1e-8)

    target = _hand_target(
        frame, hand, arms, result.config.g_bounds, result.config.missingness_bound
    )
    fluctuation = result.fluctuations["mean"]
    np.testing.assert_allclose(fluctuation.epsilon, target.epsilon, rtol=0.0, atol=1e-7)
    for level, name in ((0.0, "ey0"), (1.0, "ey1")):
        np.testing.assert_allclose(
            fluctuation.targeted.arms[level], target.targeted[level], rtol=0.0, atol=1e-8
        )
        assert result[name].psi == pytest.approx(target.psi[level], abs=1e-8)
        np.testing.assert_allclose(
            result[name].influence_curve, target.curve[level], rtol=0.0, atol=1e-7
        )
    np.testing.assert_allclose(
        result["ate"].influence_curve, target.curve[1.0] - target.curve[0.0], rtol=0.0, atol=1e-7
    )

    everything = np.arange(len(frame))
    in_sample = _hand_nuisances(frame, arms, [(everything, everything)])
    mutant = _hand_target(
        frame, in_sample, arms, result.config.g_bounds, result.config.missingness_bound
    )
    gap = np.max(np.abs(in_sample.realised - hand.realised))
    assert gap > 0.02, "the in-sample prediction mutation vanished"
    point_gap = max(abs(mutant.psi[level] - target.psi[level]) for level in arms)
    assert point_gap > 1e-3, "the in-sample point mutation vanished"


def test_w2_the_joint_fluctuation_is_one_scalar_per_arm_fitted_on_that_arm() -> None:
    frame = _witness_frame(2)
    result = _witness_fit(frame, estimands=("ey1", "ey0", "ate"))
    epsilon = np.asarray(result.fluctuations["mean"].epsilon, dtype=float)
    inverse = _stored_inverse(result)
    arm = result.data.treatment
    observed = result.data.observed
    outcome = result.data.outcome
    offset = logit(result.nuisance.outcome.observed)
    assert epsilon.shape == (2,)
    assert np.min(np.abs(epsilon)) > 1e-3

    for column, level in enumerate(result.nuisance.arms):
        rows = observed & (arm == level)
        alone = _newton(offset[rows], inverse[rows, column : column + 1], outcome[rows])
        assert epsilon[column] == pytest.approx(alone[0], abs=1e-9)

    own = inverse[np.arange(arm.size), _arm_index(result)]
    shared = _newton(offset[observed], own[observed, None], outcome[observed])[0]
    gap = float(np.max(np.abs(epsilon - shared)))
    assert gap > 0.005, "the shared-scalar mutation vanished"


def test_w3_each_row_fills_one_clever_column_and_the_fit_reads_respondents_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, np.ndarray, np.ndarray, Any]] = []
    real = TMLE._solve_rows

    def spy(
        self: TMLE,
        scaled: Any,
        initial: Any,
        submodel: Any,
        weights: Any,
        observed: Any,
        **kwargs: Any,
    ) -> Any:
        seen.append(
            (submodel, np.asarray(observed, dtype=bool).copy(), np.asarray(scaled).copy(), initial)
        )
        return real(self, scaled, initial, submodel, weights, observed, **kwargs)

    monkeypatch.setattr(TMLE, "_solve_rows", spy)
    result = _witness_fit(_witness_frame(3))
    assert len(seen) == 1
    submodel, mask, scaled, initial = seen[0]
    covariate = np.asarray(submodel.observed, dtype=float)
    rows = np.arange(result.n)
    own = _arm_index(result)
    inverse = _stored_inverse(result)

    assert covariate.shape == (result.n, 3)
    assert [submodel.arm_columns[level] for level in result.nuisance.arms] == [0, 1, 2]
    nonzero = covariate != 0.0
    assert np.all(nonzero.sum(axis=1) == 1), "a row fills more than one clever column"
    assert np.all(nonzero[rows, own])
    np.testing.assert_allclose(covariate[rows, own], inverse[rows, own], rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(mask, result.data.observed)
    assert 0 < mask.sum() < result.n

    offset = logit(np.asarray(initial.observed, dtype=float))
    epsilon = _newton(offset[mask], covariate[mask], scaled[mask])
    stored = np.asarray(result.fluctuations["mean"].epsilon, dtype=float)
    np.testing.assert_allclose(stored, epsilon, rtol=0.0, atol=1e-9)

    second = (own + 1) % 3
    filled = covariate.copy()
    filled[rows, second] = inverse[rows, second]
    mutant = _newton(offset[mask], filled[mask], scaled[mask])
    assert np.max(np.abs(mutant - epsilon)) > 0.01, "the second-column mutation vanished"

    everyone = _newton(offset, covariate, scaled)
    assert np.max(np.abs(everyone - epsilon)) > 0.05, "the nonrespondent-row mutation vanished"


def test_w4_each_contrast_is_built_from_its_arm_and_the_reference_arm() -> None:
    result = _witness_fit(_witness_frame(3), estimands=ALL_CONTRASTS)
    reference = result["ey[0.0]"]
    for level, other in ((1.0, 2.0), (2.0, 1.0)):
        mean = result[f"ey[{level}]"]
        wrong = result[f"ey[{other}]"]
        label = f"[{level} vs 0.0]"

        ate = result["ate" + label]
        assert ate.psi == pytest.approx(mean.psi - reference.psi, abs=1e-15)
        np.testing.assert_allclose(
            ate.influence_curve,
            mean.influence_curve - reference.influence_curve,
            rtol=0.0,
            atol=1e-15,
        )
        wrong_ate = mean.influence_curve - wrong.influence_curve
        assert np.max(np.abs(ate.influence_curve - wrong_ate)) > 0.5, (
            "the wrong-arm ate mutation vanished"
        )

        rr = result["rr" + label]
        assert rr.log_psi == pytest.approx(np.log(mean.psi) - np.log(reference.psi), abs=1e-14)
        rr_curve = mean.influence_curve / mean.psi - reference.influence_curve / reference.psi
        np.testing.assert_allclose(rr.influence_curve, rr_curve, rtol=0.0, atol=1e-13)
        wrong_rr = mean.influence_curve / mean.psi - wrong.influence_curve / wrong.psi
        assert np.max(np.abs(rr.influence_curve - wrong_rr)) > 0.5, (
            "the wrong-arm rr mutation vanished"
        )

        def odds_scale(estimate: Any) -> np.ndarray:
            return np.asarray(
                estimate.influence_curve / (estimate.psi * (1.0 - estimate.psi)), dtype=float
            )

        odds = result["or" + label]
        log_odds = logit(mean.psi) - logit(reference.psi)
        assert odds.log_psi == pytest.approx(log_odds, abs=1e-13)
        np.testing.assert_allclose(
            odds.influence_curve, odds_scale(mean) - odds_scale(reference), rtol=0.0, atol=1e-12
        )
        wrong_or = odds_scale(mean) - odds_scale(wrong)
        assert np.max(np.abs(odds.influence_curve - wrong_or)) > 1.0, (
            "the wrong-arm or mutation vanished"
        )


def _population_mean_tmle(
    initial: np.ndarray, respond: np.ndarray, probability: np.ndarray, outcome: np.ndarray
) -> tuple[float, np.ndarray]:
    """The TMLE of a population mean when ``outcome`` is seen with ``probability``.

    The data are ``(W, U, U Y)`` with ``U`` the indicator ``respond``. The fluctuation
    fits ``1 / probability`` on the rows with ``U = 1``.
    """
    epsilon = _newton(logit(initial[respond]), 1.0 / probability[respond, None], outcome[respond])
    targeted = expit(logit(initial) + epsilon[0] / probability)
    psi = float(np.mean(targeted))
    residual = np.where(respond, outcome - targeted, 0.0)
    return psi, residual / probability + targeted - psi


def test_w5_each_arm_mean_is_the_population_mean_tmle_on_w_t_a_u() -> None:
    """Arm ``a`` is a population mean with ``U = 1{A = a} Delta`` and ``P(U | W) = g_a pi_a``."""
    result = _witness_fit(_witness_frame(3))
    inverse = _stored_inverse(result)
    arm = result.data.treatment
    observed = result.data.observed
    outcome = result.data.outcome
    response = np.clip(
        result.nuisance.missingness_at_realised_arm(arm), result.config.missingness_bound, 1.0
    )
    natural_psi, _ = _population_mean_tmle(
        result.nuisance.outcome.observed, observed, response, outcome
    )

    gaps = []
    for column, level in enumerate(result.nuisance.arms):
        estimate = result[f"ey[{level}]"]
        psi, curve = _population_mean_tmle(
            result.nuisance.outcome.arms[level],
            observed & (arm == level),
            1.0 / inverse[:, column],
            outcome,
        )
        assert estimate.psi == pytest.approx(psi, abs=1e-10)
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-9)
        gaps.append(abs(natural_psi - psi))
    # The natural-course mean drops the arm from the response indicator and reads Q at
    # the realised arm. The measured gaps are 0.015, 0.030, and 0.046.
    assert max(gaps) > 0.015, "the natural-course identity mutation vanished"


def test_w6_counterfactual_rows_move_by_their_own_arm_inverse_under_a_wrong_q() -> None:
    result = _witness_fit(_witness_frame(3))
    fluctuation = result.fluctuations["mean"]
    epsilon = np.asarray(fluctuation.epsilon, dtype=float)
    inverse = _stored_inverse(result)
    own = inverse[np.arange(result.n), _arm_index(result)]
    arm = result.data.treatment
    _assert_inside_the_logistic_bound(result)

    checked = 0
    for column, level in enumerate(result.nuisance.arms):
        if abs(epsilon[column]) <= 1e-3:
            continue
        checked += 1
        others = arm != level
        initial = logit(result.nuisance.outcome.arms[level])
        expected = expit(initial + epsilon[column] * inverse[:, column])
        targeted = np.asarray(fluctuation.targeted.arms[level], dtype=float)
        np.testing.assert_allclose(targeted[others], expected[others], rtol=0.0, atol=1e-12)
        mutant = expit(initial + epsilon[column] * own)
        gap = np.max(np.abs(mutant[others] - targeted[others]))
        assert gap > 0.01, "the realised-arm inverse mutation vanished"
    assert checked >= 2


class _RowRecorder(BaseEstimator):
    """A logistic learner that records the ``W1`` value of every row it trains on."""

    rows: ClassVar[dict[str, list[np.ndarray]]] = {}

    def __init__(self, role: str = "outcome", column: int = 0) -> None:
        self.role = role
        self.column = column

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _RowRecorder:
        design = np.asarray(X, dtype=float)
        type(self).rows.setdefault(self.role, []).append(design[:, self.column].copy())
        self.model_ = _logistic().fit(design, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X), dtype=float)


def _recorded_fit(frame: pd.DataFrame, **overrides: Any) -> tuple[Any, dict[str, list[set[int]]]]:
    """Fit with a recorder in every role, and map each recorded ``W1`` value to its row."""
    _RowRecorder.rows = {}
    result = _witness_fit(
        frame,
        outcome_learner=_RowRecorder("outcome", 2),
        treatment_learner=_RowRecorder("treatment", 0),
        missingness_learner=_RowRecorder("response", 2),
        **overrides,
    )
    index = {value: row for row, value in enumerate(frame["W1"].to_numpy())}
    assert len(index) == len(frame)
    fits = {
        role: [{index[value] for value in values} for values in recorded]
        for role, recorded in _RowRecorder.rows.items()
    }
    return result, fits


def test_w7_no_role_trains_on_a_held_out_row() -> None:
    frame = _witness_frame(3)
    result, fits = _recorded_fit(frame)
    observed = result.data.observed
    folds = list(result.nuisance.folds)
    assert {role: len(rows) for role, rows in fits.items()} == {
        "outcome": 3,
        "treatment": 3,
        "response": 3,
    }
    for fold, (train, test) in enumerate(folds):
        assert fits["outcome"][fold] == set(train[observed[train]].tolist())
        assert fits["treatment"][fold] == set(train.tolist())
        assert fits["response"][fold] == set(train.tolist())
        for role in fits:
            assert not fits[role][fold] & set(test.tolist())

    # An in-sample fit is the leak these checks exist to catch.
    _, leaked = _recorded_fit(frame, cross_fit=False, stratify_folds="treatment")
    held_out = set(folds[0][1].tolist())
    for role in ("outcome", "treatment", "response"):
        assert leaked[role][0] & held_out, "the in-sample leak mutation vanished"


class _TargetRecorder(BaseEstimator):
    """A linear learner that records the scaled outcome of every training fit."""

    targets: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _TargetRecorder:
        type(self).targets.append(np.asarray(y, dtype=float).copy())
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X), dtype=float)


#: The known support of the continuous witness outcome.
CONTINUOUS_BOUNDS = (0.0, 10.0)


def _continuous_nuisances(
    frame: pd.DataFrame, bounds: tuple[float, float] | None, folds: Any
) -> tuple[Any, list[np.ndarray]]:
    """Fit the nuisances with ``fit_nuisances``, as the estimator does after its scaler."""
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=list(WITNESS_COVARIATES), delta="Delta"
    )
    assert data.family == "gaussian"
    _TargetRecorder.targets = []
    nuisance = fit_nuisances(
        data,
        outcome_learner=_TargetRecorder(),
        treatment_learner=_logistic(),
        missingness_learner=_logistic(),
        intermediate_learner=None,
        folds=folds,
        scaler=OutcomeScaler.from_outcome(data.outcome[data.observed], bounds),
    )
    return nuisance, list(_TargetRecorder.targets)


def test_w8_a_fixed_scale_keeps_held_out_outcomes_out_of_the_training_predictions() -> None:
    """``TMLE`` refuses ``q_bounds=None`` here, so this reads ``fit_nuisances`` directly."""
    frame = _witness_frame(2)
    rng = np.random.default_rng(3)
    level = 5.0 + 2.0 * np.tanh(frame["W1"]) + 0.5 * frame["A"] + rng.uniform(-1, 1, len(frame))
    frame["Y"] = np.where(frame["Delta"] == 1.0, level, np.nan)
    folds = make_folds(len(frame), FOLDS, stratify=None, random_state=WITNESS_SEED)
    test = folds.test_index(0)
    row = int(test[frame["Delta"].to_numpy()[test] == 1.0][0])
    moved = frame.copy()
    moved.loc[row, "Y"] = 9.9
    assert moved["Y"].max() > frame["Y"].max()

    fixed, fixed_targets = _continuous_nuisances(frame, CONTINUOUS_BOUNDS, folds)
    fixed_moved, fixed_moved_targets = _continuous_nuisances(moved, CONTINUOUS_BOUNDS, folds)
    np.testing.assert_array_equal(fixed_targets[0], fixed_moved_targets[0])
    for level_code in fixed.arms:
        np.testing.assert_array_equal(
            fixed.outcome.arms[level_code][test], fixed_moved.outcome.arms[level_code][test]
        )

    free, free_targets = _continuous_nuisances(frame, None, folds)
    free_moved, free_moved_targets = _continuous_nuisances(moved, None, folds)
    target_gap = np.max(np.abs(free_targets[0] - free_moved_targets[0]))
    assert target_gap > 0.05, "the data-dependent scale mutation vanished from the targets"
    for level_code in free.arms:
        before = free.outcome.arms[level_code][test]
        after = free_moved.outcome.arms[level_code][test]
        assert np.max(np.abs(before - after)) > 0.05, (
            "the data-dependent scale mutation vanished from the predictions"
        )
        # LinearRegression is affine-equivariant, so original units hide the change.
        np.testing.assert_allclose(
            free.scaler.lower + free.scaler.range * before,
            free_moved.scaler.lower + free_moved.scaler.range * after,
            rtol=0.0,
            atol=1e-9,
        )


def test_w9_the_delta_mask_and_both_inverse_factors_each_move_the_curve() -> None:
    result = _witness_fit(_witness_frame(3))
    fluctuation = result.fluctuations["mean"]
    arm = result.data.treatment
    observed = result.data.observed
    outcome = result.data.outcome
    propensity = np.clip(result.nuisance.propensity.values, *result.config.g_bounds)
    response = np.clip(result.nuisance.missingness, result.config.missingness_bound, 1.0)
    residual = outcome - np.asarray(fluctuation.targeted.observed, dtype=float)
    assert np.all(outcome[~observed] == 0.0)

    for column, level in enumerate(result.nuisance.arms):
        estimate = result[f"ey[{level}]"]
        targeted = np.asarray(fluctuation.targeted.arms[level], dtype=float)
        plug_in = targeted - estimate.psi
        indicator = (arm == level).astype(float)
        g, pi = propensity[:, column], response[:, column]

        hand = np.where(observed, indicator * residual / (g * pi), 0.0) + plug_in
        np.testing.assert_allclose(estimate.influence_curve, hand, rtol=0.0, atol=1e-12)

        mutants = {
            "Delta mask": indicator * residual / (g * pi) + plug_in,
            "1 / pi": np.where(observed, indicator * residual / g, 0.0) + plug_in,
            "1 / g": np.where(observed, indicator * residual / pi, 0.0) + plug_in,
        }
        for label, mutant in mutants.items():
            gap = np.max(np.abs(estimate.influence_curve - mutant))
            assert gap > 0.1, f"the dropped {label} mutation vanished"

        complete_case = float(np.mean(targeted[observed]))
        assert abs(complete_case - estimate.psi) > 0.01, "the complete-case mutation vanished"


def test_w10_the_ate_variance_keeps_the_same_row_covariance() -> None:
    result = _witness_fit(_witness_frame(3), estimands=("ey", "ate"))
    reference = result["ey[0.0]"].influence_curve
    n = result.n
    for level in (1.0, 2.0):
        curve = result[f"ey[{level}]"].influence_curve
        ate = result[f"ate[{level} vs 0.0]"]
        joint = float(np.var(curve - reference, ddof=1) / n)
        assert ate.variance == pytest.approx(joint, rel=1e-12)

        covariance = float(np.cov(curve, reference)[0, 1])
        assert covariance > 0.0
        dropped = float((np.var(curve, ddof=1) + np.var(reference, ddof=1)) / n)
        assert abs(dropped - ate.variance) > 0.05 * ate.variance, (
            "the dropped-covariance mutation vanished"
        )


def _max_t_quantile(
    curves: np.ndarray, std_errors: np.ndarray, *, seed: int, draws: int, diagonal: bool = False
) -> float:
    """The 0.95 quantile of the Rademacher max-t statistic, drawn as the package draws it.

    ``diagonal=True`` gives each column its own multipliers, which is the band a diagonal
    covariance implies.
    """
    rng = np.random.default_rng(seed)
    n, columns = curves.shape
    centred = curves - curves.mean(axis=0, keepdims=True)

    def multipliers() -> np.ndarray:
        packed = rng.integers(0, 256, size=(draws, (n + 7) // 8), dtype=np.uint8)
        return np.unpackbits(packed, axis=1, count=n).astype(float) * 2.0 - 1.0

    if diagonal:
        replicates = np.column_stack(
            [multipliers() @ centred[:, column] / n for column in range(columns)]
        )
    else:
        replicates = multipliers() @ centred / n
    statistics = np.max(np.abs(replicates) / std_errors, axis=1)
    return float(np.quantile(statistics, 0.95))


def test_w11_the_band_critical_value_uses_the_joint_same_row_curves() -> None:
    draws = 1000
    result = _witness_fit(
        _witness_frame(3),
        estimands=ALL_CONTRASTS,
        simultaneous=True,
        n_multiplier=draws,
        multiplier_kind="rademacher",
    )
    estimates = list(result.estimates.values())
    assert len(estimates) == 9
    curves = np.column_stack([estimate.influence_curve for estimate in estimates])
    std_errors = np.array([estimate.std_error for estimate in estimates])
    correlation = np.corrcoef(curves, rowvar=False)
    assert np.max(np.abs(correlation - np.eye(9))) > 0.9

    seed = ADMITTED["random_state"]
    joint = _max_t_quantile(curves, std_errors, seed=seed, draws=draws)
    assert result.simultaneous.critical_value == pytest.approx(joint, abs=1e-12)
    diagonal = _max_t_quantile(curves, std_errors, seed=seed, draws=draws, diagonal=True)
    assert diagonal - joint > 0.05, "the diagonal-covariance mutation vanished"


class _FoldConstant(BaseEstimator):
    """An outcome learner that predicts one constant per outer fold, in fitting order."""

    table: ClassVar[tuple[float, ...]] = (0.25, 0.5, 0.75)
    next_fold: ClassVar[int] = 0

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _FoldConstant:
        self.fold_ = type(self).next_fold
        type(self).next_fold += 1
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = np.full(len(X), type(self).table[self.fold_])
        return np.column_stack([1.0 - probability, probability])


def test_unequal_folds_report_the_whole_sample_mean_not_an_equal_fold_average() -> None:
    _FoldConstant.next_fold = 0
    result = _witness_fit(_witness_frame(3), outcome_learner=_FoldConstant())
    folds = [test for _, test in result.nuisance.folds]
    assert [len(test) for test in folds] == [201, 200, 200]
    for fold, test in enumerate(folds):
        assert np.all(result.nuisance.outcome.observed[test] == _FoldConstant.table[fold])

    for level in result.nuisance.arms:
        targeted = np.asarray(result.fluctuations["mean"].targeted.arms[level], dtype=float)
        whole_sample = float(np.average(targeted, weights=result.data.weights))
        equal_fold = float(np.mean([np.mean(targeted[test]) for test in folds]))
        assert result[f"ey[{level}]"].psi == pytest.approx(whole_sample, abs=1e-15)
        assert abs(whole_sample - equal_fold) > 1e-4, "the equal-fold-average mutation vanished"


class _InnerFoldRecorder(SuperLearner):
    """A Super Learner that records its inner fold assignment at each outer fold."""

    assignments: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None, groups: Any = None) -> Any:
        fitted = super().fit(X, y, sample_weight=sample_weight, groups=groups)
        type(self).assignments.append(np.asarray(self.folds_.assignment).copy())
        return fitted


#: The inner seed and inner folds of the leakage witness.
INNER_SEED = 5
INNER_FOLDS = 2


def _inner_assignments(frame: pd.DataFrame) -> tuple[Any, list[np.ndarray]]:
    _InnerFoldRecorder.assignments = []
    learner = _InnerFoldRecorder(
        library=[("glm", _logistic())],
        task="classification",
        n_folds=INNER_FOLDS,
        random_state=INNER_SEED,
    )
    result = _witness_fit(frame, estimands=("ey1", "ey0", "ate"), outcome_learner=learner)
    return result, list(_InnerFoldRecorder.assignments)


def _all_row_mutant(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """The inner split a leaky learner would draw: stratified on every row's outcome."""
    labels = frame["Y"].fillna(-1.0).to_numpy()
    split = make_folds(len(frame), INNER_FOLDS, stratify=labels, random_state=INNER_SEED)
    return np.asarray(split.assignment)[rows]


def test_a_held_out_outcome_leaves_the_inner_super_learner_split_unchanged() -> None:
    """The inner split stratifies on its training target, so held-out outcomes cannot move it."""
    frame = _witness_frame(2)
    result, before = _inner_assignments(frame)
    folds = list(result.nuisance.folds)
    observed = result.data.observed
    assert len(before) == len(folds)

    for fold, (train, test) in enumerate(folds):
        rows = train[observed[train]]
        target = frame["Y"].to_numpy()[rows]
        expected = make_folds(rows.size, INNER_FOLDS, stratify=target, random_state=INNER_SEED)
        np.testing.assert_array_equal(before[fold], expected.assignment)

        flipped = frame.copy()
        held_out = int(test[observed[test]][0])
        flipped.loc[held_out, "Y"] = 1.0 - flipped.loc[held_out, "Y"]
        moved_result, after = _inner_assignments(flipped)
        np.testing.assert_array_equal(
            moved_result.nuisance.folds.assignment, result.nuisance.folds.assignment
        )
        np.testing.assert_array_equal(before[fold], after[fold])

        leaky_before = _all_row_mutant(frame, rows)
        leaky_after = _all_row_mutant(flipped, rows)
        assert np.any(leaky_before != leaky_after), "the all-row stratification mutation vanished"
