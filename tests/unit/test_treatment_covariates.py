"""Treatment-mechanism covariates (X22 part (h)): the treatment mechanism reads a subset ``Wg``.

``PointTreatment(treatment_adjustment=Wg)`` and ``TMLE.fit(treatment_covariates=Wg)`` let the
treatment mechanism read a subset of the adjustment set.  Every other nuisance keeps the full
set.  The declaration restricts the observed-data model and does not change the estimand, so
the design prints a pinned statistical-model restriction beside the assumptions.

The checks fall in four groups.  A spy learner records which covariates every learner role sees.
The text, the data field and the refusals are pinned.  The exact law of the plan's section
I-2.6.2 puts a number on the term that the restriction hides: an instrument ``Z`` in the
mechanism makes the remainder linear in the outcome-regression error, and the arm-1 error
equals the plan's closed formulas to ``1e-10``.  A mutation that fits the mechanism on the full
set makes the same witness fail.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from cleverly import ATE, CausalStudy, CounterfactualMean, PointTreatment
from cleverly.data import CausalData
from cleverly.data.treatment_covariates import (
    RESTRICTION_TEMPLATE,
    treatment_mechanism_restriction,
)
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import Incremental
from cleverly.msm import MSM
from cleverly.study import _matches_registered_point_identification
from cleverly.targets import TARGETS
from tests.unit import _known_mechanism_support as law
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The plan's pinned restriction (X22 I-2.6.3), with ``{Wg}`` still to fill.
PINNED = (
    "statistical-model restriction, treatment-mechanism covariates {Wg}: the treatment "
    "mechanism depends on the adjustment covariates only through {Wg}. It restricts the "
    "observed-data model and does not change the estimand. The fit is consistent when the "
    "treatment mechanism fitted on {Wg} is consistent, which needs the restriction, or when "
    "the outcome regression is consistent. The interval needs the restriction and both "
    "nuisances consistent at the usual product rate. When the restriction fails, the "
    "remainder is first order in the outcome-regression error. To exclude instruments, omit "
    "them from adjustment= instead: the outcome regression and the treatment mechanism then "
    "both use {Wg}."
)

#: The plan's pinned refusal texts (X22 I-2.8), with the argument names of the design.
R_H1 = (
    "treatment_adjustment names {cols} outside adjustment=. The treatment mechanism "
    "conditions on a subset of the adjustment set"
)
R_H4 = (
    "treatment_adjustment= restricts a fitted treatment mechanism, and "
    "treatment_probabilities= supplies a known one. Remove one of them"
)
R_H6 = "treatment_adjustment repeats {cols}"

#: A covariate's id is its column mean over 1000, so a spy learner reads which columns it got.
OFFSET = 1000.0


def make_frame(n: int = 240, seed: int = 3) -> pd.DataFrame:
    """Four covariates ``c1..c4`` whose mean is ``1000 * j``, a binary treatment and an outcome."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(size=(n, 4))
    frame = pd.DataFrame(
        {f"c{j + 1}": OFFSET * (j + 1) + noise[:, j] for j in range(4)},
    )
    logit = 1.2 * (noise[:, 1] - 0.5)
    frame["A"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    frame["Y"] = noise[:, 0] + 0.5 * frame["A"] + 0.3 * noise[:, 2] + rng.normal(0, 0.1, n)
    frame["D"] = (rng.uniform(size=n) < 0.8).astype(float)
    frame["T"] = (rng.uniform(size=n) < 0.8).astype(float)
    return frame


def ids(design: Any) -> frozenset[int]:
    """The covariate ids a design holds: 0 is a treatment, intercept or bin column."""
    means = np.asarray(design, dtype=float).mean(axis=0)
    return frozenset(round(float(value) / OFFSET) for value in means)


class Spy:
    """Records the covariate ids each learner role sees, at fit and at predict."""

    seen: ClassVar[dict[str, list[frozenset[int]]]] = {}

    @classmethod
    def reset(cls) -> None:
        cls.seen = {}

    @classmethod
    def roles(cls) -> dict[str, frozenset[int]]:
        out: dict[str, frozenset[int]] = {}
        for role, widths in cls.seen.items():
            out[role] = frozenset().union(*widths)
        return out


class SpyClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that predicts the training prior and records the ids of its design."""

    def __init__(self, role: str = "g") -> None:
        self.role = role

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SpyClassifier:
        Spy.seen.setdefault(self.role, []).append(ids(X))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        Spy.seen.setdefault(self.role, []).append(ids(X))
        return np.tile(self.prior_, (len(X), 1))


class SpyRegressor(BaseEstimator, RegressorMixin):
    """A regressor that predicts the training mean and records the ids of its design."""

    def __init__(self, role: str = "q") -> None:
        self.role = role

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SpyRegressor:
        Spy.seen.setdefault(self.role, []).append(ids(X))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        Spy.seen.setdefault(self.role, []).append(ids(X))
        return np.full(len(X), self.mean_)


def spied(**overrides: Any) -> TMLE:
    """A TMLE whose learners are spies, one role each."""
    settings: dict[str, Any] = {
        "outcome_learner": SpyRegressor("q"),
        "treatment_learner": SpyClassifier("g"),
        "missingness_learner": SpyClassifier("missing"),
        "cross_fit": False,
        "q_bounds": (-5.0, 5.0),
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return TMLE(**settings)


COVARIATES = ["c1", "c2", "c3", "c4"]
WG = ["c2", "c3"]


# --------------------------------------------------------------------------- the spy learner


@pytest.mark.parametrize("cross_fit", [False, True])
def test_the_mechanism_learner_sees_exactly_wg(cross_fit: bool) -> None:
    Spy.reset()
    kwargs: dict[str, Any] = {"cross_fit": cross_fit}
    if cross_fit:
        kwargs["n_folds"] = 2
    spied(**kwargs).fit(
        make_frame(), outcome="Y", treatment="A", covariates=COVARIATES, treatment_covariates=WG
    )
    roles = Spy.roles()
    assert roles["g"] == {2, 3}
    assert roles["q"] == {0, 1, 2, 3, 4}


def test_the_default_mechanism_learner_sees_every_covariate() -> None:
    Spy.reset()
    spied().fit(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    assert Spy.roles()["g"] == {1, 2, 3, 4}


def test_missingness_and_treatment_observation_learners_see_the_full_set() -> None:
    Spy.reset()
    spied(missingness_learner=SpyClassifier("missing")).fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        delta="D",
        treatment_delta="T",
        treatment_covariates=WG,
    )
    roles = Spy.roles()
    assert roles["g"] == {2, 3}
    assert {1, 2, 3, 4} <= roles["missing"]
    assert roles["q"] >= {1, 2, 3, 4}


def test_a_dose_density_reads_wg() -> None:
    frame = make_frame()
    frame["A"] = frame["c2"] % 1.0 + frame["c3"] % 1.0
    Spy.reset()
    spied(msm=MSM.linear(doses=[0.2, 0.6, 1.0, 1.4, 1.8]), density_bins=4).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_kind="continuous",
        treatment_covariates=WG,
    )
    assert Spy.roles()["g"] - {0} == {2, 3}
    assert Spy.roles()["q"] >= {1, 2, 3, 4}


def test_the_classifier_ratio_of_a_policy_reads_wg() -> None:
    from cleverly.interventions import Shift

    frame = make_frame()
    frame["A"] = frame["c2"] % 1.0 + frame["c3"] % 1.0
    Spy.reset()
    spied(policies=[Shift(0.1, cap=None)], ratio="classifier", n_folds=2, cross_fit=True).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_kind="continuous",
        treatment_covariates=WG,
    )
    assert Spy.roles()["g"] - {0} == {2, 3}


def test_the_nested_inner_designs_of_a_drtmle_fit_read_wg() -> None:
    Spy.reset()
    DRTMLE(
        outcome_learner=SpyRegressor("q"),
        treatment_learner=SpyClassifier("g"),
        reduced_outcome_learner=SpyRegressor("rq"),
        reduced_treatment_learner=SpyClassifier("rg"),
        reduced_crossfit="nested",
        cross_fit=True,
        n_folds=3,
        estimands=("ate",),
        q_bounds=(-5.0, 5.0),
        simultaneous=False,
        random_state=0,
        max_outer=2,
    ).fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    assert Spy.roles()["g"] == {2, 3}
    assert Spy.roles()["q"] >= {0, 1, 2, 3, 4}


def test_the_evaluation_companion_of_a_drtmle_fit_reads_wg() -> None:
    Spy.reset()
    DRTMLE(
        outcome_learner=SpyRegressor("q"),
        treatment_learner=SpyClassifier("g"),
        reduced_outcome_learner=SpyRegressor("rq"),
        reduced_treatment_learner=SpyClassifier("rg"),
        evaluation=make_frame(n=120, seed=9),
        cross_fit=True,
        n_folds=2,
        estimands=("ate",),
        q_bounds=(-5.0, 5.0),
        simultaneous=False,
        random_state=0,
        max_outer=2,
    ).fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    # Every mechanism design, the companion's included, holds Wg and nothing else.
    assert Spy.roles()["g"] == {2, 3}
    assert {len(widths) for widths in Spy.seen["g"]} == {2}


def test_the_screen_reports_only_covariates_of_wg() -> None:
    result = spied(screen_treatment=True, treatment_learner=SpyClassifier("g")).fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    assert set(result.single().nuisance.treatment_covariates) <= set(WG)
    assert Spy.roles()["g"] <= {2, 3}


def test_a_reported_wg_is_the_whole_set_by_default() -> None:
    result = spied().fit(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    assert result.single().nuisance.treatment_covariates == tuple(COVARIATES)
    restricted = spied().fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    assert restricted.single().nuisance.treatment_covariates == tuple(WG)


def test_an_empty_wg_fits_the_marginal_mechanism() -> None:
    Spy.reset()
    result = spied().fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=[],
    )
    assert Spy.roles()["g"] == {0}
    assert result.single().data.mechanism_covariates.shape == (240, 1)
    propensity = np.asarray(result.single().nuisance.propensity.values)
    assert np.ptp(propensity[:, 1]) < 1e-12


# ------------------------------------------------------------------------------ the data field


def test_the_default_accessor_is_the_covariate_matrix_itself() -> None:
    data = CausalData.from_frame(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    assert data.treatment_covariate_names is None
    assert data.mechanism_covariates is data.covariates
    assert data.mechanism_covariate_names == data.covariate_names
    assert data.treatment_adjustment_sources is None


def test_the_accessor_selects_the_declared_columns_in_design_order() -> None:
    data = CausalData.from_frame(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=["c3", "c2"],
    )
    assert data.treatment_covariate_names == ("c2", "c3")
    assert np.array_equal(data.mechanism_covariates, data.covariates[:, [1, 2]])
    assert data.mechanism_covariate_names == ("c2", "c3")
    assert data.treatment_adjustment_sources == ("c2", "c3")


def test_a_declaration_of_the_whole_set_is_no_restriction() -> None:
    data = CausalData.from_frame(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=list(reversed(COVARIATES)),
    )
    assert data.treatment_covariate_names is None


def test_a_categorical_wg_names_its_indicators_together() -> None:
    frame = make_frame()
    frame["site"] = np.where(
        frame["c1"] % 1.0 < 0.5, "a", np.where(frame["c2"] % 1.0 < 0.5, "b", "c")
    )
    data = CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=[*COVARIATES, "site"],
        treatment_covariates=["site", "c1"],
    )
    assert data.treatment_covariate_names == ("c1", "site__b", "site__c")
    assert data.treatment_adjustment_sources == ("c1", "site")
    assert data.mechanism_covariates.shape[1] == 3


def test_a_dropped_constant_covariate_is_in_neither_set() -> None:
    frame = make_frame()
    frame["flat"] = 5.0
    with pytest.warns(UserWarning, match="dropped"):
        data = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="A",
            covariates=[*COVARIATES, "flat"],
            treatment_covariates=[*COVARIATES, "flat"],
        )
    assert data.treatment_covariate_names is None


def test_from_arrays_takes_the_declaration() -> None:
    rng = np.random.default_rng(0)
    data = CausalData.from_arrays(
        rng.normal(size=60),
        rng.integers(0, 2, 60),
        rng.normal(size=(60, 3)),
        covariate_names=["a", "b", "c"],
        treatment_covariates=["b"],
    )
    assert data.treatment_covariate_names == ("b",)
    assert data.mechanism_covariates.shape == (60, 1)


def test_the_copies_keep_or_restrict_the_declaration() -> None:
    data = CausalData.from_frame(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    rows = np.arange(0, 240, 2)
    assert data.subset(rows).treatment_covariate_names == ("c2", "c3")
    assert np.array_equal(data.subset(rows).mechanism_covariates, data.covariates[rows][:, [1, 2]])
    assert data.without_covariates(["c1"]).treatment_covariate_names == ("c2", "c3")
    assert data.without_covariates(["c2"]).treatment_covariate_names == ("c3",)
    # Dropping the covariates outside Wg leaves Wg as the whole design: no restriction.
    assert data.without_covariates(["c1", "c4"]).treatment_covariate_names is None
    # Replacing the values of the design keeps the declaration: the names do not change.
    replaced = data.with_covariates(data.covariates + 0.0)
    assert replaced.treatment_covariate_names == ("c2", "c3")


def test_with_treatment_covariates_attaches_once() -> None:
    data = CausalData.from_frame(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    attached = data.with_treatment_covariates(WG)
    assert attached.treatment_covariate_names == ("c2", "c3")
    assert data.treatment_covariate_names is None
    with pytest.raises(ValueError, match="declared twice"):
        attached.with_treatment_covariates(WG)


# ----------------------------------------------------------------------------- the refusals


def test_r_h1_names_outside_the_adjustment_set() -> None:
    expected = R_H1.format(cols=["z"])
    with pytest.raises(DataError) as raised:
        PointTreatment(
            outcome="Y", treatment="A", adjustment=("c1",), treatment_adjustment=("c1", "z")
        )
    assert str(raised.value) == expected
    with pytest.raises(DataError, match="outside covariates="):
        CausalData.from_frame(
            make_frame(),
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_covariates=["c1", "z"],
        )
    with pytest.raises(DataError, match="outside covariates="):
        CausalData.from_frame(
            make_frame(), outcome="Y", treatment="A", covariates=COVARIATES
        ).with_treatment_covariates(["z"])
    with pytest.raises(DataError, match="outside adjustment="):
        PointTreatment(outcome="Y", treatment="A", randomized=True, treatment_adjustment=("c1",))


def test_r_h6_repeated_names() -> None:
    with pytest.raises(DataError) as raised:
        PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("c1", "c2"),
            treatment_adjustment=("c1", "c1"),
        )
    assert str(raised.value) == R_H6.format(cols=["c1"])
    with pytest.raises(DataError, match=r"treatment_covariates repeats \['c2'\]"):
        spied().fit(
            make_frame(),
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_covariates=["c2", "c2"],
        )


def test_r_h4_a_known_mechanism_beside_wg() -> None:
    with pytest.raises(DataError) as raised:
        PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("c1", "c2"),
            treatment_adjustment=("c1",),
            treatment_probabilities={0: "p0", 1: "p1"},
        )
    assert str(raised.value) == R_H4
    frame = make_frame()
    frame["p1"] = 0.5
    frame["p0"] = 0.5
    learners = never_fit_learners()
    assert_refused_before_any_call(
        lambda: TMLE(cross_fit=False, **learners).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_probabilities={0: "p0", 1: "p1"},
            treatment_covariates=WG,
        ),
        None,
        "learner",
        "treatment_covariates= restricts a fitted treatment mechanism, and "
        "treatment_probabilities= supplies a known one. Remove one of them",
        error=DataError,
    )
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=COVARIATES)
    both = data.with_treatment_covariates(WG)
    with pytest.raises(DataError, match="Remove one of them"):
        both.with_known_mechanism(np.full(240, 0.5))
    known = data.with_known_mechanism(np.full(240, 0.5))
    with pytest.raises(DataError, match="Remove one of them"):
        known.with_treatment_covariates(WG)
    with pytest.raises(DataError, match="Remove one of them"):
        TMLE(cross_fit=False, **never_fit_learners()).fit(
            both, treatment_probabilities=np.full(240, 0.5), treatment_covariates=None
        )


def test_r_h3_incremental_beside_wg() -> None:
    learners = never_fit_learners()
    raised = assert_refused_before_any_call(
        lambda: TMLE(cross_fit=False, incremental=[Incremental(2.0)], **learners).fit(
            make_frame(),
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_covariates=WG,
        ),
        None,
        "learner",
        "incremental= shifts the fitted propensity score, so a mechanism on {c2, c3} defines "
        "the shift on {c2, c3} and changes the estimand.",
        "Omit treatment_adjustment=",
        error=CapabilityError,
    )
    assert isinstance(raised, CapabilityError)
    assert raised.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
    assert NeverFit.calls == 0


def test_r_h3_does_not_refuse_an_incremental_fit_without_wg() -> None:
    # The refusal reads the declaration, so the default path stays on the shipped code.
    data = CausalData.from_frame(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    TMLE(cross_fit=False, incremental=[Incremental(2.0)])._check_incremental(data)


def test_a_study_design_with_a_mismatched_container_is_refused() -> None:
    frame = make_frame()
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=tuple(COVARIATES), treatment_adjustment=("c2",)
    )
    plain = PointTreatment(outcome="Y", treatment="A", adjustment=tuple(COVARIATES))
    data = design.prepare(frame)
    assert data.treatment_covariate_names == ("c2",)
    assert design.prepare(data) is data
    with pytest.raises(DataError, match="treatment_adjustment"):
        plain.prepare(data)
    with pytest.raises(DataError, match="treatment_adjustment"):
        design.prepare(plain.prepare(frame))
    other = dataclasses.replace(design, treatment_adjustment=("c3",))
    with pytest.raises(DataError, match="treatment_adjustment"):
        other.prepare(data)
    # Declaring the whole set is the same design as declaring nothing.
    whole = dataclasses.replace(design, treatment_adjustment=tuple(COVARIATES))
    assert whole.prepare(frame).treatment_covariate_names is None
    assert whole.prepare(plain.prepare(frame)).treatment_covariate_names is None


# ------------------------------------------------------------------------------- the text


def pinned(names: str) -> str:
    return PINNED.replace("{Wg}", names)


def test_the_text_is_the_pinned_one() -> None:
    assert RESTRICTION_TEMPLATE == PINNED
    assert treatment_mechanism_restriction(["age", "bmi"]) == pinned("{age, bmi}")
    assert treatment_mechanism_restriction([]) == pinned("{}")


def study(**design: Any) -> CausalStudy:
    return CausalStudy(
        make_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=tuple(COVARIATES), **design),
    )


def test_a_restricted_design_records_the_text_and_prints_it_after_the_assumptions() -> None:
    effect = study(treatment_adjustment=("c2", "c3")).identify(ATE())
    assert effect.identification.model_restrictions == (pinned("{c2, c3}"),)
    lines = effect.summary().splitlines()
    start = lines.index("model restrictions:")
    assert lines[start + 1] == "  - " + pinned("{c2, c3}")
    assert lines.index("assumptions:") < start
    facts = effect.summary_lines()
    assert "model restrictions: " + pinned("{c2, c3}") in facts


def test_an_unrestricted_design_prints_no_restriction() -> None:
    for design in ({}, {"treatment_adjustment": tuple(COVARIATES)}):
        effect = study(**design).identify(ATE())
        assert effect.identification.model_restrictions == ()
        assert "restriction" not in effect.summary()


def test_an_empty_wg_prints_empty_braces() -> None:
    effect = study(treatment_adjustment=()).identify(ATE())
    assert effect.identification.model_restrictions == (pinned("{}"),)


def test_the_provenance_matcher_reads_the_restriction() -> None:
    effect = study(treatment_adjustment=("c2", "c3")).identify(ATE())
    assert effect._study is not None
    data = effect._study.data
    assert isinstance(data, CausalData)
    registered = TARGETS["ate"].identification
    assert _matches_registered_point_identification(
        effect.identification, registered, effect, data, effect.functional.axis
    )
    dropped = dataclasses.replace(effect.identification, model_restrictions=())
    assert not _matches_registered_point_identification(
        dropped, registered, effect, data, effect.functional.axis
    )


def test_a_fitted_result_prints_the_restriction_of_its_design() -> None:
    result = (
        study(treatment_adjustment=("c2", "c3"))
        .identify(ATE())
        .estimate(
            outcome_learner=SpyRegressor("q"),
            treatment_learner=SpyClassifier("g"),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
    )
    summary = result.summary().splitlines()
    kinds = [line.split(":")[0] for line in summary]
    assert summary[kinds.index("model restrictions")] == "model restrictions: " + pinned("{c2, c3}")
    assert kinds.index("identification assumptions") + 1 == kinds.index("model restrictions")
    assert sum(line.startswith("model restrictions") for line in summary) == 1


def test_a_low_level_fit_prints_the_same_text_from_its_data() -> None:
    result = spied().fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    summary = result.single().summary().splitlines()
    assert "model restrictions: " + pinned("{c2, c3}") in summary
    plain = spied().fit(make_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
    assert "restriction" not in plain.single().summary()


def test_a_study_fit_reads_wg_through_the_design() -> None:
    Spy.reset()
    study(treatment_adjustment=("c2", "c3")).identify(ATE()).estimate(
        outcome_learner=SpyRegressor("q"),
        treatment_learner=SpyClassifier("g"),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert Spy.roles()["g"] == {2, 3}
    assert Spy.roles()["q"] == {0, 1, 2, 3, 4}


# ------------------------------------------------------------------------------ the witnesses

#: Nonzero floor of a witness: the error is a number, not a rounding residue.
FLOOR = 1e-3


def fit_law(
    frame: pd.DataFrame,
    *,
    treatment_covariates: list[str] | None,
    fluctuation: Literal["linear", "logistic"],
    k: str = "Z",
    covariates: tuple[str, ...] = ("W1", "Z"),
) -> Any:
    """The arm-1 fit of the instrument law with the cell-mean mechanism and the raw stub."""
    estimator = TMLE(
        outcome_learner=law.RawScaleOutcome(k=k, width=1 + len(covariates)),
        treatment_learner=law.CellMeanClassifier(),
        cross_fit=False,
        q_bounds=(0.0, 1.0),
        g_bounds=(0.01, 0.99),
        fluctuation=fluctuation,
        simultaneous=False,
    )
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(covariates),
        treatment_covariates=treatment_covariates,
    ).single()


def arm1_error(result: Any) -> float:
    return float(result.estimates["ey1"].psi) - law.INSTRUMENT_TRUTH_ARM1


def gbar(frame: pd.DataFrame) -> np.ndarray:
    return 0.375 + 0.25 * frame["W1"].to_numpy()


def delta_formula(frame: pd.DataFrame, k: str, *, on_instrument: bool = True) -> float:
    """``P_0[(1 - g_0 / gbar) delta k]``, exact because ``P_n = P_0``."""
    g0 = law.instrument_mechanism(frame["W1"], frame["Z"], on_instrument=on_instrument)
    values = frame["Z"].to_numpy() if k == "Z" else frame["W1"].to_numpy()
    return float(np.mean((1.0 - g0 / gbar(frame)) * law.INSTRUMENT_DELTA * values))


def qstar_formula(frame: pd.DataFrame, result: Any, *, on_instrument: bool = True) -> float:
    """``P_0[(1 - g_0 / gbar)(Q* - Q_0)]`` with the fit's own targeted arm-1 outcome."""
    g0 = law.instrument_mechanism(frame["W1"], frame["Z"], on_instrument=on_instrument)
    qstar = np.asarray(result.fluctuations["mean"].targeted.arms[1.0], dtype=float)
    q0 = 0.5 + 0.2 * frame["W1"].to_numpy()
    return float(np.mean((1.0 - g0 / gbar(frame)) * (qstar - q0)))


def check_linear_witness() -> None:
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation="linear")
    expected = delta_formula(frame, "Z")
    assert abs(arm1_error(result) - expected) < 1e-10
    assert abs(arm1_error(result)) > FLOOR


def check_logistic_witness() -> None:
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation="logistic")
    expected = qstar_formula(frame, result)
    assert abs(arm1_error(result) - expected) < 1e-10
    assert abs(arm1_error(result)) > FLOOR


def test_the_instrument_law_is_exact() -> None:
    frame = law.instrument_frame()
    assert len(frame) == 16 * law.INSTRUMENT_M
    # The cell-mean mechanism returns gbar on {W1} and g_0 on {W1, Z}, exactly.
    one = frame[frame["Z"] == 1]
    assert np.isclose(frame.groupby("W1")["A"].mean().loc[0.0], 0.375)
    assert np.isclose(one.groupby("W1")["A"].mean().loc[0.0], 0.5)
    # P_n equals P_0: the arm-1 mean of the true outcome regression is 0.6.
    assert np.isclose((0.5 + 0.2 * frame["W1"]).mean(), law.INSTRUMENT_TRUTH_ARM1)


def test_witness_linear_error_equals_the_delta_formula() -> None:
    check_linear_witness()
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation="linear")
    assert arm1_error(result) == pytest.approx(-1 / 75, abs=1e-10)


def test_witness_logistic_error_equals_the_targeted_formula() -> None:
    check_logistic_witness()
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation="logistic")
    assert arm1_error(result) == pytest.approx(-0.0138815, abs=1e-6)
    # The delta formula holds for the linear fluctuation only: under the logistic update the
    # error differs from it by more than the floor.
    assert abs(arm1_error(result) - delta_formula(frame, "Z")) > 1e-4


@pytest.mark.parametrize("fluctuation", ["linear", "logistic"])
def test_vanishing_control_one_the_error_is_a_function_of_wg(
    fluctuation: Literal["linear", "logistic"],
) -> None:
    """``k = W1``: the outcome error varies with ``Wg`` only, and the factor annihilates it."""
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation=fluctuation, k="W1")
    assert abs(arm1_error(result)) < 1e-10


@pytest.mark.parametrize("fluctuation", ["linear", "logistic"])
def test_vanishing_control_two_the_mechanism_does_not_depend_on_the_instrument(
    fluctuation: Literal["linear", "logistic"],
) -> None:
    """``g_0 = gbar``: the restriction holds, so the same outcome error does no harm."""
    frame = law.instrument_frame(on_instrument=False)
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation=fluctuation)
    assert abs(arm1_error(result)) < 1e-10


@pytest.mark.parametrize("fluctuation", ["linear", "logistic"])
def test_instrument_out_of_the_adjustment_set_is_the_h2_analogue(
    fluctuation: Literal["linear", "logistic"],
) -> None:
    """Dropping ``Z`` from ``adjustment=`` puts both nuisances on ``W1``: the error vanishes."""
    frame = law.instrument_frame()
    result = fit_law(
        frame,
        treatment_covariates=None,
        fluctuation=fluctuation,
        k="W1",
        covariates=("W1",),
    )
    assert abs(arm1_error(result)) < 1e-10


def test_the_full_set_mechanism_removes_the_term() -> None:
    """Without the restriction the cell-mean mechanism returns ``g_0``, and the error is 0."""
    frame = law.instrument_frame()
    for fluctuation in ("linear", "logistic"):
        result = fit_law(frame, treatment_covariates=None, fluctuation=fluctuation)
        assert abs(arm1_error(result)) < 1e-10


def test_mutation_fitting_the_mechanism_on_the_full_set_fails_the_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_linear_witness()
    check_logistic_witness()
    monkeypatch.setattr(CausalData, "mechanism_covariates", property(lambda self: self.covariates))
    # The cell-mean mechanism now returns g_0, the error is 0, and the nonzero floor fails.
    with pytest.raises(AssertionError):
        check_linear_witness()
    with pytest.raises(AssertionError):
        check_logistic_witness()


def test_the_witness_assertion_has_teeth_against_a_swapped_formula() -> None:
    """A formula that drops the factor ``(1 - g_0 / gbar)`` is not the error."""
    frame = law.instrument_frame()
    result = fit_law(frame, treatment_covariates=["W1"], fluctuation="linear")
    wrong = float(np.mean(law.INSTRUMENT_DELTA * frame["Z"].to_numpy()))
    assert abs(arm1_error(result) - wrong) > 1e-3


def test_the_study_route_reproduces_the_witness() -> None:
    frame = law.instrument_frame()
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "Z"), treatment_adjustment=("W1",)
        ),
    ).identify(CounterfactualMean(1))
    result = effect.estimate(
        outcome_learner=law.RawScaleOutcome(),
        treatment_learner=law.CellMeanClassifier(),
        cross_fit=False,
        q_bounds=(0.0, 1.0),
        g_bounds=(0.01, 0.99),
        fluctuation="linear",
        simultaneous=False,
    )
    assert float(result["ey1"].psi) - law.INSTRUMENT_TRUTH_ARM1 == pytest.approx(
        delta_formula(frame, "Z"), abs=1e-10
    )
