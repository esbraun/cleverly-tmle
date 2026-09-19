"""Stacked CV-TMLE of arm-indexed means and contrasts with missing-at-random outcomes.

``docs/roadmap.md`` RM9 ("Arm-indexed stacked contract") states the contract. This module
holds its fast checks, grouped by the roadmap's witness list:

* W12, refusals: every refusal row, before any learner call, through the engine keywords
  and through the public ``TMLEMethod`` route.
* W12, preflight: every sample and training-complement minimum, a ``DataError`` before any
  learner call, and the two-per-class rule for a package ``SuperLearner`` classification
  role.
* W1 to W11, numerical witnesses: added in their own section below the refusals. Each one
  builds the correct value from stored arrays, asserts package equality, and asserts a
  stated gap for a deliberate mutation.

The refusal tests pin :class:`~tests.unit._natural_course_support.NeverFit` learners, so a
refusal that ran after a learner call would raise ``AssertionError`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import (
    ATE,
    ATT,
    CapabilityError,
    CausalStudy,
    CollaborativeTMLEMethod,
    CrossFitting,
    DataError,
    Inference,
    ModelSpec,
    PointTreatment,
    SplitPlan,
    SuperLearner,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_missing_outcome, make_missing_outcome_binary
from cleverly.estimators import CTMLE, TMLE
from cleverly.learners import make_folds
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The first clause of every contract refusal.
CONTRACT = (
    "Cross-fitted TMLE of arm-indexed means and contrasts with missing outcomes supports "
    "one audited stacked CV-TMLE contract; "
)

#: Rows in the refusal frames, and the outer folds every admitted setting declares.
N = 200
FOLDS = 3

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
        engine={"split_plan": SplitPlan((tuple(int(v) for v in np.arange(N) % FOLDS),))},
        public={
            "cross_fitting": {
                "split_plan": SplitPlan((tuple(int(v) for v in np.arange(N) % FOLDS),))
            }
        },
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


@pytest.mark.parametrize("stratify", ["treatment", "none"])
@pytest.mark.parametrize("estimands", ENGINE_ESTIMANDS, ids=("ate", "default", "all", "att"))
@pytest.mark.parametrize("row", OVERRIDE_ROWS, ids=[row.id for row in OVERRIDE_ROWS])
def test_each_engine_refusal_row_fires_before_any_learner(
    row: Row, estimands: Any, stratify: str
) -> None:
    """Each row refuses alone, and the fixed check order decides between several rows."""
    expected = _expected(row, conditional=estimands != ("ate",), stratify=stratify)

    with pytest.raises(CapabilityError) as caught:
        _engine_fit(row, estimands=estimands, stratify=stratify)
    message = str(caught.value)
    assert message.startswith(CONTRACT)
    assert expected.fragment in message
    assert NeverFit.calls == 0


@pytest.mark.parametrize("estimands", [None, "all", ("att",)], ids=("default", "all", "att"))
@pytest.mark.parametrize("stratify", ["treatment", "none"])
def test_att_and_atc_are_refused_by_name_in_every_request_form(
    estimands: Any, stratify: str
) -> None:
    with pytest.raises(CapabilityError) as caught:
        _engine_fit(None, estimands=estimands, stratify=stratify)
    message = str(caught.value)
    assert BY_ID["att-atc"].fragment in message
    assert "Request estimands from ['ate', 'ey', 'ey1', 'ey0', 'rr', 'or']" in message
    requested = "['att']" if estimands == ("att",) else "['att', 'atc']"
    assert f"no audited result covers {requested}" in message
    assert NeverFit.calls == 0


def test_treatment_strata_are_refused_for_an_admitted_estimand() -> None:
    with pytest.raises(CapabilityError) as caught:
        _engine_fit(None, estimands=("ate",), stratify="treatment")
    assert "no audited result covers stratify_folds='treatment' (RM17)" in str(caught.value)
    assert NeverFit.calls == 0


def test_outcome_strata_are_refused_for_an_admitted_estimand() -> None:
    with pytest.raises(CapabilityError) as caught:
        _engine_fit(None, estimands=("ate",), stratify="treatment+outcome")
    assert "stratify_folds='treatment+outcome' (RM17)" in str(caught.value)
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


@pytest.mark.parametrize("stratify", ["treatment", "none"])
@pytest.mark.parametrize(("estimand", "conditional"), PUBLIC_ESTIMANDS, ids=("ATE", "ATT"))
@pytest.mark.parametrize("row", OVERRIDE_ROWS, ids=[row.id for row in OVERRIDE_ROWS])
def test_each_public_refusal_row_fires_before_any_learner(
    row: Row, estimand: Any, conditional: bool, stratify: str
) -> None:
    with pytest.raises(CapabilityError) as caught:
        _public_fit(row, estimand=estimand, stratify=stratify)
    message = str(caught.value)
    if row.collaborative and conditional:
        # The public route refuses a collaborative ATT before the engine sees it.
        assert message.startswith("method 'collaborative_tmle' cannot estimate ATT")
    else:
        expected = _expected(row, conditional=conditional, stratify=stratify)
        assert message.startswith(CONTRACT)
        assert expected.fragment in message
    assert NeverFit.calls == 0


@pytest.mark.parametrize("stratify", ["treatment", "none"])
def test_the_public_att_request_is_refused_by_name(stratify: str) -> None:
    with pytest.raises(CapabilityError) as caught:
        _public_fit(None, estimand=ATT(), stratify=stratify)
    assert BY_ID["att-atc"].fragment in str(caught.value)
    assert NeverFit.calls == 0


def test_the_public_default_strata_are_refused_for_the_ate() -> None:
    """``CrossFitting()`` defaults to treatment strata, which the contract refuses."""
    with pytest.raises(CapabilityError) as caught:
        _public_fit(None, estimand=ATE(), stratify="treatment")
    assert BY_ID["stratified-folds"].fragment in str(caught.value)
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


def test_unstratified_folds_stay_reserved_for_the_in_sample_fit() -> None:
    """The ``"none"`` widening admits the cross-fitted surface only."""
    estimator = TMLE(
        **{**ADMITTED, **never_fit_learners(), "estimands": ("ate",), "cross_fit": False}
    )
    with pytest.raises(CapabilityError, match="stratify_folds='none' is currently reserved"):
        estimator.fit(_frame(), outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
    assert NeverFit.calls == 0


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
        "fit in sample with cross_fit=False and stratify_folds='treatment' on the engine "
        "(CrossFitting(enabled=False, stratify_by='treatment'))"
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
    assert "Increase n_folds, use a different random_state, or fit in sample" in message
    assert NeverFit.calls == 0


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


def _never_fit_super_learner(task: str | None = "classification") -> SuperLearner:
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
    assert "the Super Learner's inner stratified split" in message
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
# Numerical witnesses land here. Each builds the correct value from stored arrays,
# asserts package equality, and asserts a stated gap for a deliberate mutation.
