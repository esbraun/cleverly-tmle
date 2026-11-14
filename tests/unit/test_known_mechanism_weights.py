"""A declared treatment mechanism beside observation weights: the two readings.

A weighted fit estimates the effect in the tilted law ``dP_w = w dP / E w`` and divides by
that law's mechanism, ``P_w(A = a | W) = g0 m / sum_a' g0 m`` with ``m(a, W) = E[w | A = a, W]``
(Bayes' rule).  A declaration ``g0`` beside non-constant weights therefore needs a reading:
``mechanism_law="target"`` divides by ``g0`` as the tilted law's mechanism, and
``mechanism_law="sample"`` estimates ``g_w`` as ``g0 r / sum g0 r`` with
``r(a | W) = P_v(A = a | W)``, a classification with weights ``v = w / g0(A | W)``.

The exact-law tests run on :mod:`tests.unit._known_mechanism_support`, whose empirical law is
the law, with weights that read ``W``, ``A`` and ``Y``.  The tilt classifier is the trainable
:class:`~tests.unit._known_mechanism_support.WeightedCellClassifier`, whose weighted fit on
the exact law returns ``r`` exactly, so a ``"sample"`` fit divides by ``g_w`` itself and its
estimate equals ``Psi(P_w)`` whatever the outcome regression.  The truth is computed by
enumeration of the tilted cells.  The mutation controls are learners that receive what the
route hands them and fit something else: the classifier the route would fit if it passed
``w``, or no weights, or dropped the ``g0`` factor.  Each must miss.

The test names W1 to W12 follow ``rows/Fix-16/plan.md`` section 5 of the plans repository.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from cleverly import ATE, CausalStudy, DataError, PointTreatment, TMLEMethod, _support
from cleverly._summary import vocabulary
from cleverly.data import CausalData
from cleverly.data.known_mechanism import (
    KNOWN_SAMPLE_ADAPTIVE_IPTW_REFUSAL,
    KNOWN_SAMPLE_DRTMLE_EVALUATION_REFUSAL,
    KNOWN_SAMPLE_DRTMLE_GUARD_REFUSAL,
    KNOWN_WEIGHTED_LAW_REFUSAL,
    divides_by_declaration,
    unweighted_tilt_refusal,
)
from cleverly.estimators import DRTMLE, TMLE, AdaptiveIPTW, tmle
from cleverly.estimators._nuisance import KnownPropensity, TiltedPropensity
from cleverly.exceptions import CapabilityError, MethodConfigurationError
from cleverly.inference.bootstrap import run_bootstrap
from cleverly.interventions import Incremental, ModifiedPolicy
from cleverly.learners import SuperLearner
from cleverly.sensitivity.positivity import truncation_curve
from cleverly.validation import refute
from cleverly.validation._interpretation import fit_configuration
from tests.unit import _known_mechanism_support as law
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit

#: The exact-law tolerance: the fluctuation stops at ``tol = 1e-10`` on its relative score.
EXACT = 1e-9
#: How far a mutated fit must miss.
MISS = 1e-3
ROLES: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}

#: The exact laws carry whole-number weights, which the weight check reads as possible counts.
pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.WeightingWarning")
#: The two-arm targets of W4.
BINARY_TARGETS = ("ey0", "ey1", "ate", "att", "atc", "rr", "or")

#: The weights of the plan's section 1, from the cell weight ``wt`` of the law.  Only the
#: first reads ``W`` alone; there the two readings coincide.
WEIGHTS: dict[str, Any] = {
    "covariates": lambda f: f["wt"],
    "treatment": lambda f: f["wt"] * (1.0 + f["A"]),
    "treatment_outcome": lambda f: f["wt"] * (1.0 + f["A"]) * (1.0 + 2.0 * f["Y"]),
    "outcome": lambda f: f["wt"] * (1.0 + 2.0 * f["Y"]),
}
#: The weights that read the treatment or the outcome, where the readings differ.
TILTING = ("treatment", "treatment_outcome", "outcome")


def _columns(arms: int) -> dict[float, str]:
    return {float(code): f"p{code}" for code in range(arms)}


def _frame(weight: str, arms: int = 2) -> pd.DataFrame:
    frame = law.point_frame(arms)
    frame["w"] = WEIGHTS[weight](frame)
    return frame


def _wrong_outcome() -> LogisticRegression:
    """A main-terms logistic outcome regression, which the law's table is not."""
    return LogisticRegression(max_iter=1000)


def _tmle(treatment_learner: Any = None, **settings: Any) -> TMLE:
    base: dict[str, Any] = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": law.WeightedCellClassifier()
        if treatment_learner is None
        else treatment_learner,
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    base.update(settings)
    return TMLE(**base)


def _drtmle(**settings: Any) -> DRTMLE:
    base: dict[str, Any] = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": law.WeightedCellClassifier(),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_mechanism_residual_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ey0", "ey1", "ate"),
    }
    base.update(settings)
    return DRTMLE(**base)


def _fit(
    frame: pd.DataFrame, mechanism_law: str, estimator: Any = None, arms: int = 2, **roles: Any
) -> Any:
    model = estimator if estimator is not None else _tmle(estimands="all")
    return model.fit(
        frame,
        **ROLES,
        weights="w",
        treatment_probabilities=_columns(arms),
        mechanism_law=mechanism_law,
        **roles,
    ).single()


# ------------------------------------------------------------------------- the tilted law


def _tilted_cells(frame: pd.DataFrame, arms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``P_w(W)``, ``g_w(a | W)`` and ``Qbar_w(a, W)`` per cell of the law, by enumeration."""
    total = float(frame["w"].sum())
    mass = np.zeros(len(law.CELLS))
    mechanism = np.zeros((len(law.CELLS), arms))
    outcome = np.zeros((len(law.CELLS), arms))
    for index, (w1, w2) in enumerate(law.CELLS):
        cell = frame[(frame["W1"] == w1) & (frame["W2"] == w2)]
        weight = float(cell["w"].sum())
        mass[index] = weight / total
        for arm in range(arms):
            rows = cell[cell["A"] == float(arm)]
            mechanism[index, arm] = float(rows["w"].sum()) / weight
            outcome[index, arm] = float((rows["w"] * rows["Y"]).sum() / rows["w"].sum())
    return mass, mechanism, outcome


def _tilted_truth(frame: pd.DataFrame, arms: int = 2) -> dict[str, float]:
    """``Psi(P_w)``: the arm means, and at two arms the contrasts, ATT and ATC."""
    mass, mechanism, outcome = _tilted_cells(frame, arms)
    means = mass @ outcome
    truth = {f"ey[{float(a)}]": float(means[a]) for a in range(arms)}
    if arms == 2:
        blip = outcome[:, 1] - outcome[:, 0]
        treated = mass * mechanism[:, 1]
        control = mass * mechanism[:, 0]
        truth.update(
            ey0=float(means[0]),
            ey1=float(means[1]),
            ate=float(means[1] - means[0]),
            rr=float(means[1] / means[0]),
            att=float(treated @ blip / treated.sum()),
            atc=float(control @ blip / control.sum()),
        )
        truth["or"] = float((means[1] / (1 - means[1])) / (means[0] / (1 - means[0])))
    return truth


def _errors(result: Any, truth: dict[str, float]) -> dict[str, float]:
    compared = [name for name in result.estimates if name in truth]
    assert compared
    return {name: abs(result.estimates[name].psi - truth[name]) for name in compared}


def _assert_exact(result: Any, truth: dict[str, float]) -> None:
    errors = _errors(result, truth)
    assert max(errors.values()) <= EXACT, errors


def _assert_misses(result: Any, truth: dict[str, float]) -> None:
    errors = _errors(result, truth)
    assert max(errors.values()) > MISS, errors


def _design_mechanism(design: Any, arms: int) -> np.ndarray:
    """The law's ``g0`` at each row of a ``(W1, W2)`` design, ``(n, K)``."""
    cells = [law.CELLS.index((float(a), float(b))) for a, b in np.asarray(design, dtype=float)]
    if arms == 2:
        one = np.array([law.BINARY_MECHANISM[cell] for cell in cells])
        return np.column_stack([1.0 - one, one])
    return np.array([law.THREE_ARM_MECHANISM[cell] for cell in cells], dtype=float)


class ObservationWeightedTilt(law.WeightedCellClassifier):
    """Mutation: the tilt classifier fitted with ``w`` in place of ``v = w / g0(A | W)``.

    It multiplies the weights it receives back by ``g0`` at the observed arm, so it fits the
    classifier the route would fit if it handed over the observation weights.
    """

    def __init__(self, arms: int = 2) -> None:
        self.arms = arms

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ObservationWeightedTilt:
        g0 = _design_mechanism(design, self.arms)
        at_arm = g0[np.arange(len(target)), np.asarray(target, dtype=int)]
        return super().fit(design, target, np.asarray(sample_weight) * at_arm)


class UnweightedTilt(law.WeightedCellClassifier):
    """Mutation: the tilt classifier fitted without weights, so ``r`` becomes ``g0``."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> UnweightedTilt:
        return super().fit(design, target, None)


class TiltAlone(law.WeightedCellClassifier):
    """Mutation: predicts ``r / g0`` normalised, so the fit divides by ``r`` alone (W5)."""

    def __init__(self, arms: int = 2) -> None:
        self.arms = arms

    def predict_proba(self, design: Any) -> np.ndarray:
        ratio = super().predict_proba(design) / _design_mechanism(design, self.arms)
        return ratio / ratio.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------ W1


def _refused(build: Any) -> None:
    NeverFit.calls = 0
    assert_refused_before_any_call(
        build, None, "learner", KNOWN_WEIGHTED_LAW_REFUSAL, error=DataError
    )


def test_w1_every_point_entry_refuses_a_declaration_beside_weights_without_a_reading() -> None:
    """W1: each point entry refuses, before any learner, with the pinned text."""
    frame = _frame("treatment_outcome")
    g = law.mechanism(frame)
    learners = {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}
    unweighted = CausalData.from_frame(frame.drop(columns=["w"]), **ROLES)
    weighted = CausalData.from_frame(frame, **ROLES, weights="w")
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        weights="w",
        treatment_probabilities=_columns(2),
    )
    entries = {
        "from_frame": lambda: CausalData.from_frame(
            frame, **ROLES, weights="w", treatment_probabilities=_columns(2)
        ),
        "from_arrays": lambda: CausalData.from_arrays(
            frame["Y"],
            frame["A"],
            frame[["W1", "W2"]],
            weights=frame["w"],
            treatment_probabilities=g,
        ),
        "with_known_mechanism": lambda: weighted.with_known_mechanism(g),
        "fit_frame": lambda: TMLE(**learners).fit(
            frame, **ROLES, weights="w", treatment_probabilities=_columns(2)
        ),
        "fit_container": lambda: TMLE(**learners).fit(weighted, treatment_probabilities=g),
        "tmle": lambda: tmle(
            frame["Y"],
            frame["A"],
            frame[["W1", "W2"]],
            obsWeights=frame["w"],
            g1W=g[:, 1],
            **learners,
        ),
        "design": lambda: (
            CausalStudy(frame, design=design)
            .identify(ATE())
            .estimate(TMLEMethod().with_overrides(**learners))
        ),
        # A container built by hand that carries both reaches the fit's own check.
        "hand_built": lambda: TMLE(**learners).fit(
            dataclasses.replace(unweighted.with_known_mechanism(g), weights=weighted.weights)
        ),
    }
    for build in entries.values():
        _refused(build)


def test_w1_the_keyword_is_validated_and_needs_a_declaration() -> None:
    frame = _frame("treatment")
    with pytest.raises(DataError, match="mechanism_law must be"):
        CausalData.from_frame(
            frame, **ROLES, weights="w", treatment_probabilities=_columns(2), mechanism_law="P_w"
        )
    with pytest.raises(DataError, match="declares no treatment_probabilities"):
        CausalData.from_frame(frame, **ROLES, weights="w", mechanism_law="target")
    with pytest.raises(DataError, match="declares no treatment_probabilities"):
        PointTreatment(outcome="Y", treatment="A", adjustment=("W1",), mechanism_law="sample")
    data = CausalData.from_frame(frame, **ROLES, weights="w")
    with pytest.raises(ValueError, match="passes none"):
        TMLE(outcome_learner=NeverFit(), treatment_learner=NeverFit()).fit(
            data, mechanism_law="sample"
        )


def test_w1b_constant_weights_and_no_weights_need_no_reading() -> None:
    """W1b: constant weights tilt nothing, so both readings coincide and nothing is refused.

    A point declaration lies in ``(0, 1)``, so the exemption of a 0/1 declaration is a
    longitudinal one and is tested with the node carriers.
    """
    frame = law.point_frame(2)
    frame["w"] = 2.0
    data = CausalData.from_frame(frame, **ROLES, weights="w", treatment_probabilities=_columns(2))
    assert data.known_treatment is not None and data.known_treatment.law is None
    assert divides_by_declaration(data)
    plain = CausalData.from_frame(
        frame.drop(columns=["w"]), **ROLES, treatment_probabilities=_columns(2)
    )
    assert divides_by_declaration(plain)
    # "sample" on constant weights is the design mechanism itself: known, and the truth.
    sample = _fit(frame, "sample", _tmle(NeverFit(), estimands=BINARY_TARGETS))
    assert sample.config.treatment_mechanism == "known"
    _assert_exact(sample, law.point_truth(2))


# ------------------------------------------------------------------------------------ W2


def test_w2_target_with_covariate_weights_divides_by_the_declaration() -> None:
    """W2 (formerly E9): ``"target"`` beside weights on ``W`` is the shipped arithmetic.

    The truth is the tilted law's, and the fit equals the fit whose treatment learner returns
    the declared values, to the last bits: the declaration is divided by unchanged.
    """
    frame = _frame("covariates")
    result = _fit(frame, "target", _tmle(NeverFit(), estimands="all"))
    _assert_exact(result, law.point_truth(2, weighted=True))
    assert isinstance(result.nuisance.propensity, KnownPropensity)
    fitted = _tmle(law.FixedMechanism(2), estimands="all").fit(frame, **ROLES, weights="w").single()
    for name, estimate in result.estimates.items():
        np.testing.assert_allclose(estimate.psi, fitted.estimates[name].psi, rtol=1e-12)


# ------------------------------------------------------------------------------------ W3


def _case_cohort() -> pd.DataFrame:
    """A case-cohort sample of the exact population law, its weights ``1 / pi``.

    Every case is kept; controls are kept with probability 1/2 when treated, and 1/5 or 1/10
    by ``W2`` when not.  Each ``(cell, A, Y)`` count of the law is a multiple of ten, so the
    sample holds exactly ``pi`` times each count and the weighted sample is the population.
    """
    population = law.point_frame(2)
    keep = np.where(
        population["Y"] == 1.0,
        1.0,
        np.where(population["A"] == 1.0, 0.5, np.where(population["W2"] == 0.0, 0.2, 0.1)),
    )
    pieces = []
    for _, group in population.assign(pi=keep).groupby(["W1", "W2", "A", "Y"]):
        count = len(group) * float(group["pi"].iloc[0])
        assert abs(count - round(count)) < 1e-9
        pieces.append(group.iloc[: round(count)])
    sample = pd.concat(pieces, ignore_index=True)
    sample["w"] = 1.0 / sample["pi"]
    return sample


def test_w3_target_is_exact_on_a_case_cohort_sample_of_a_trial() -> None:
    """W3: randomization before sampling is the ``"target"`` reading, and it is exact.

    The declared mechanism is the population's, which the weighted sample restores.  The
    mutation declares the arm shares of the rows as drawn under ``"target"``, and misses.

    ``"sample"`` is exact here too.  Its tilted mechanism ``g0 r / sum g0 r`` with
    ``r = P_v(A | W)`` and ``v = w / g0(A | W)`` equals ``P_w(A | W)`` for every positive
    ``g0``, because ``P_v(A = a | W)`` is proportional to ``P_w(A = a | W) / g0(a | W)``.  So
    the reading decides whether the fit divides by the declaration as known or estimates the
    tilted mechanism from it; it does not decide consistency when the classifier is right.
    """
    sample = _case_cohort()
    truth = law.point_truth(2)
    targets = ("ey0", "ey1", "ate", "rr", "or", "att", "atc")
    _assert_exact(_fit(sample, "target", _tmle(NeverFit(), estimands=targets)), truth)
    drawn = sample.groupby(["W1", "W2"])["A"].transform("mean")
    as_drawn = sample.assign(p0=1.0 - drawn, p1=drawn)
    _assert_misses(_fit(as_drawn, "target", _tmle(NeverFit(), estimands=targets)), truth)
    _assert_exact(_fit(sample, "sample", _tmle(estimands=targets)), truth)
    _assert_exact(_fit(as_drawn, "sample", _tmle(estimands=targets)), truth)


# ------------------------------------------------------------------------------------ W4


@pytest.mark.parametrize("weight", list(WEIGHTS))
def test_w4_sample_recovers_the_tilted_truth_with_a_trainable_classifier(weight: str) -> None:
    """W4: ``"sample"`` is exact for every two-arm target, including outcome-reading weights.

    The outcome regression is wrong and the tilt classifier is trained, so the exactness
    comes from ``g_w`` alone.  The mutations: the classifier fitted with ``w`` in place of
    ``v``, fitted unweighted, and the ``"target"`` reading where the weights read ``A`` or
    ``Y``, each miss.
    """
    frame = _frame(weight)
    truth = _tilted_truth(frame)
    result = _fit(frame, "sample", _tmle(estimands=BINARY_TARGETS))
    assert set(BINARY_TARGETS) <= set(result.estimates)
    _assert_exact(result, truth)
    assert isinstance(result.nuisance.propensity, TiltedPropensity)
    for mutation in (ObservationWeightedTilt(), UnweightedTilt()):
        _assert_misses(_fit(frame, "sample", _tmle(mutation, estimands=BINARY_TARGETS)), truth)
    target = _fit(frame, "target", _tmle(NeverFit(), estimands=BINARY_TARGETS))
    if weight in TILTING:
        _assert_misses(target, truth)
    else:
        _assert_exact(target, truth)


# ------------------------------------------------------------------------------------ W5


def test_w5_the_declared_factor_enters_the_tilted_mechanism() -> None:
    """W5: dividing by ``r`` alone, without the ``g0`` factor, misses.

    ``g0`` is not constant across arms in any cell but the two at one half, so a dropped
    factor or a normalisation over the wrong axis moves the mechanism.
    """
    frame = _frame("treatment_outcome")
    truth = _tilted_truth(frame)
    _assert_exact(_fit(frame, "sample", _tmle(estimands=BINARY_TARGETS)), truth)
    _assert_misses(_fit(frame, "sample", _tmle(TiltAlone(), estimands=BINARY_TARGETS)), truth)


# ------------------------------------------------------------------------------------ W6


@pytest.mark.parametrize("weight", ["treatment_outcome", "outcome"])
def test_w6_three_arms_are_exact(weight: str) -> None:
    frame = _frame(weight, arms=3)
    truth = _tilted_truth(frame, arms=3)
    result = _fit(frame, "sample", _tmle(estimands=("ey",)), arms=3)
    _assert_exact(result, truth)
    _assert_misses(_fit(frame, "sample", _tmle(TiltAlone(3), estimands=("ey",)), arms=3), truth)
    _assert_misses(_fit(frame, "target", _tmle(NeverFit(), estimands=("ey",)), arms=3), truth)


# ------------------------------------------------------------------------------------ W7


def test_w7_a_sample_fit_reads_estimated_and_a_target_fit_reads_known() -> None:
    """W7: every reader of status follows the predicate, not the raw declaration."""
    frame = _frame("treatment")
    sample = _fit(frame, "sample", _tmle(estimands=("ate",)))
    target = _fit(frame, "target", _tmle(NeverFit(), estimands=("ate",)))
    assert sample.config.treatment_mechanism == "estimated"
    assert sample.nuisance.treatment_mechanism == "estimated"
    assert target.config.treatment_mechanism == "known"
    header = dict(sample.summary().header)
    assert header[vocabulary.TREATMENT_MODEL] != vocabulary.KNOWN_MECHANISM
    assert dict(target.summary().header)[vocabulary.TREATMENT_MODEL] == vocabulary.KNOWN_MECHANISM
    sample_key, _ = _support.point_key(sample.estimator, sample.data, ("ate",))
    target_key, _ = _support.point_key(target.estimator, target.data, ("ate",))
    assert "known_mechanism" not in sample_key.qualifiers
    assert "known_mechanism" in target_key.qualifiers
    # The refutation fingerprint keeps the treatment learner, which fitted the tilt.
    sample_fit = fit_configuration(sample, "ate")
    target_fit = fit_configuration(target, "ate")
    assert not sample_fit.known_mechanism and len(sample_fit.learners) == 2
    assert target_fit.known_mechanism and len(target_fit.learners) == 1
    assert sample.nuisance.diagnostics["tilt_effective_n"] < frame.shape[0]
    assert "treatment mechanism=known" not in repr(sample.data)
    assert "treatment mechanism=known" in repr(target.data)


def test_w7_the_data_summary_names_a_known_mechanism_for_target_only() -> None:
    frame = _frame("treatment")

    def header(mechanism_law: str) -> dict[str, str]:
        design = PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2"),
            weights="w",
            treatment_probabilities=_columns(2),
            mechanism_law=mechanism_law,  # type: ignore[arg-type]
        )
        return dict(CausalStudy(frame, design=design).summary().header)

    assert header("target")[vocabulary.TREATMENT_MODEL] == vocabulary.KNOWN_MECHANISM
    assert vocabulary.TREATMENT_MODEL not in header("sample")


def test_w7_the_evalue_note_of_a_dr_tmle_fit_follows_the_predicate() -> None:
    """The linear-remainder note marks an estimated mechanism, so ``"sample"`` carries it."""
    from cleverly.sensitivity.evalue import _LINEAR_REMAINDER_NOTE, evalue

    frame = _frame("treatment")
    sample = _fit(frame, "sample", _drtmle(guard=()))
    reading = evalue(sample, "ate").limit_reading
    assert reading is not None and not reading.licensed
    assert _LINEAR_REMAINDER_NOTE in reading.sentence
    target = _fit(frame, "target", _drtmle(guard=(), treatment_learner=NeverFit()))
    target_reading = evalue(target, "ate").limit_reading
    assert target_reading is None or _LINEAR_REMAINDER_NOTE not in target_reading.sentence


# ------------------------------------------------------------------------------------ W8


def test_w8_an_incremental_fit_tilts_the_estimated_mechanism_with_its_term() -> None:
    """W8: the tilt of ``g_w`` is exact, and its curve carries the mechanism term.

    The witness: the same tilt on data that declares ``g_w`` itself as the target mechanism
    is the known stochastic regime, whose curve has no ``(A - g)`` term.  The two agree on
    the point and differ in the curve.
    """
    frame = _frame("treatment_outcome")
    mass, mechanism, outcome = _tilted_cells(frame, 2)
    delta = 2.0
    shifted = delta * mechanism[:, 1] / (delta * mechanism[:, 1] + mechanism[:, 0])
    truth = float(mass @ (shifted * outcome[:, 1] + (1.0 - shifted) * outcome[:, 0]))
    result = _fit(frame, "sample", _tmle(incremental=[Incremental(delta)]))
    (ipsi,) = result.estimates.values()
    assert abs(ipsi.psi - truth) <= EXACT
    assert result.nuisance.incremental is not None
    assert not result.nuisance.incremental.mechanism_known
    cells = [law.CELLS.index((a, b)) for a, b in frame[["W1", "W2"]].itertuples(index=False)]
    declared = frame.assign(q0=mechanism[cells, 0], q1=mechanism[cells, 1])
    known = (
        _tmle(NeverFit(), incremental=[Incremental(delta)])
        .fit(
            declared,
            **ROLES,
            weights="w",
            treatment_probabilities={0.0: "q0", 1.0: "q1"},
            mechanism_law="target",
        )
        .single()
    )
    (regime,) = known.estimates.values()
    assert abs(regime.psi - truth) <= EXACT
    assert np.max(np.abs(regime.influence_curve - ipsi.influence_curve)) > MISS


@pytest.mark.parametrize("arms", [2, 3])
def test_w8_a_discrete_policy_reads_the_estimated_mechanism(arms: int) -> None:
    def apply(a: Any, h: Any) -> Any:
        values = np.asarray(a, dtype=float)
        if arms == 2:
            return np.where(np.asarray(h["W2"], dtype=float) == 1.0, 1.0, values)
        return np.where(values == 2.0, 1.0, values)

    frame = _frame("treatment_outcome", arms=arms)
    mass, mechanism, outcome = _tilted_cells(frame, arms)
    truth = 0.0
    for cell, (_, w2) in enumerate(law.CELLS):
        for arm in range(arms):
            target = int(apply(np.array([float(arm)]), pd.DataFrame({"W2": [w2]}))[0])
            truth += mass[cell] * mechanism[cell, arm] * outcome[cell, target]
    policy = ModifiedPolicy("mtp", apply=apply, policy_kind="known")
    result = _fit(frame, "sample", _tmle(policies=[policy]), arms=arms)
    assert abs(result.estimates["ey_policy[mtp]"].psi - truth) <= EXACT
    target = _fit(frame, "target", _tmle(NeverFit(), policies=[policy]), arms=arms)
    # The two-arm policy moves only the untreated rows with W2 = 1, so the target reading's
    # error is smaller here (7.7e-4) than elsewhere, and still 1e5 times the tolerance.
    assert abs(target.estimates["ey_policy[mtp]"].psi - truth) > 5e-4


# ------------------------------------------------------------------------------------ W9


class ReplicaTMLE(law.ReplicaFolds, TMLE):
    pass


@pytest.mark.parametrize("scheme", ["pooled", "fold"])
def test_w9_a_cross_fitted_sample_fit_on_replica_folds_is_exact(scheme: str) -> None:
    """W9: each training fold is a replica of the weighted law, so ``r`` is exact out of fold.

    A classifier trained on rows outside the training fold, or with the weights of other
    rows, would not return ``r`` on every fold.
    """
    frame = _frame("treatment_outcome")
    estimator = ReplicaTMLE(
        outcome_learner=_wrong_outcome(),
        treatment_learner=law.WeightedCellClassifier(),
        cross_fit=True,
        n_folds=5,
        estimands=BINARY_TARGETS,
        simultaneous=False,
        random_state=0,
        targeting_scheme=scheme,
    )
    _assert_exact(_fit(frame, "sample", estimator), _tilted_truth(frame))


def test_w9_the_tilt_weights_have_mean_one_in_each_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tilt weights ``w / g0`` are rescaled to mean one on each fold's training rows.

    A penalised learner's loss is not invariant to the weight scale, so the raw tilt weights
    would change its regularisation relative to the observation weights.  The spy reads the
    weights each fold's treatment fit receives, the one fit whose design is the covariates
    alone (the outcome fits also read the treatment).
    """
    from cleverly.estimators import _nuisance

    frame = _frame("treatment_outcome")
    seen: list[float] = []
    original = _nuisance.fit_on_rows

    def spy(learner: Any, design: Any, target: Any, weights: Any, rows: Any, *rest: Any) -> Any:
        if np.shape(design)[1] == len(ROLES["covariates"]):
            seen.append(float(np.mean(np.asarray(weights, dtype=float)[rows])))
        return original(learner, design, target, weights, rows, *rest)

    monkeypatch.setattr(_nuisance, "fit_on_rows", spy)
    estimator = TMLE(
        outcome_learner=_wrong_outcome(),
        treatment_learner=LogisticRegression(C=0.1, max_iter=1000),
        cross_fit=True,
        n_folds=5,
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )
    _fit(frame, "sample", estimator)
    assert len(seen) == 5
    np.testing.assert_allclose(seen, 1.0, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------ W11


def test_w11_fixed_mechanism_constructions_refuse_a_sample_declaration() -> None:
    """W11: DR-TMLE with a guard or a companion, and AdaptiveIPTW, redirect before any learner."""
    frame = _frame("treatment")
    never = {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}
    cases = {
        KNOWN_SAMPLE_DRTMLE_GUARD_REFUSAL: _drtmle(guard=("Q",), **never),
        KNOWN_SAMPLE_DRTMLE_EVALUATION_REFUSAL: _drtmle(guard=(), evaluation=frame, **never),
        KNOWN_SAMPLE_ADAPTIVE_IPTW_REFUSAL: AdaptiveIPTW(
            treatment_learner=NeverFit(), reduced_outcome_learner=NeverFit(), cross_fit=False
        ),
    }
    for refusal, estimator in cases.items():
        NeverFit.calls = 0
        assert_refused_before_any_call(
            lambda model=estimator: _fit(frame, "sample", model), None, "learner", refusal
        )
        # The target reading divides by the declaration, which these constructions take:
        # the fit reaches its first learner.
        with pytest.raises(AssertionError, match="before any learner is fitted"):
            _fit(frame, "target", estimator)


def test_w11_an_unguarded_dr_tmle_on_a_sample_declaration_is_the_tmle_fit() -> None:
    frame = _frame("treatment_outcome")
    plain = _fit(frame, "sample", _tmle(estimands=("ey0", "ey1", "ate")))
    unguarded = _fit(frame, "sample", _drtmle(guard=()))
    for name, estimate in plain.estimates.items():
        np.testing.assert_allclose(unguarded.estimates[name].psi, estimate.psi, rtol=1e-10)


@pytest.mark.parametrize(
    "learner, member",
    [
        (KNeighborsClassifier(), "KNeighborsClassifier.fit"),
        (
            SuperLearner([LogisticRegression(), KNeighborsClassifier()], task="classification"),
            "the SuperLearner member KNeighborsClassifier, whose fit",
        ),
    ],
    ids=["learner", "super_learner_member"],
)
def test_w11_a_tilt_classifier_that_ignores_weights_is_refused(learner: Any, member: str) -> None:
    """12.5: without ``sample_weight`` the classifier returns ``g0`` and the fit ``g0**2``."""
    frame = _frame("covariates")
    NeverFit.calls = 0
    assert_refused_before_any_call(
        lambda: _fit(frame, "sample", _tmle(learner, outcome_learner=NeverFit())),
        None,
        "learner",
        unweighted_tilt_refusal(member),
        error=MethodConfigurationError,
    )
    # The target reading fits no treatment learner, so it takes any.
    _fit(frame, "target", _tmle(learner, estimands=("ate",)))


# ------------------------------------------------------------------------------------ W12


def test_w12_a_subset_and_a_bootstrap_replicate_keep_the_reading() -> None:
    frame = _frame("treatment")
    data = CausalData.from_frame(
        frame, **ROLES, weights="w", treatment_probabilities=_columns(2), mechanism_law="sample"
    )
    assert data.subset(np.arange(0, 4800, 3)).known_treatment.law == "sample"
    seen: list[Any] = []

    def record(replicate: CausalData) -> dict[str, tuple[float, float]]:
        seen.append(replicate.known_treatment.law)
        return {"x": (0.0, 1.0)}

    run_bootstrap(data, record, n_replicates=2, resampling="iid", random_state=4)
    assert seen == ["sample", "sample"]


def test_w12_the_placebo_declares_target_and_the_subset_keeps_the_reading() -> None:
    """The placebo treatment is independent of ``(W, A, Y)``, so its shares are ``P_w``'s."""
    frame = _frame("treatment")
    result = _fit(frame, "sample", _tmle(estimands=("ate",)))
    seen: list[CausalData] = []
    real = result.estimator.refit

    def spy(data: CausalData, **options: Any) -> Any:
        seen.append(data)
        return real(data, **options)

    result.estimator.refit = spy  # type: ignore[method-assign]
    refute(result, tests=("placebo", "subset"), n_replicates=2, random_state=1)
    laws = {data.known_treatment.law for data in seen}
    assert laws == {"target", "sample"}


def test_w12_simulated_confounding_carries_the_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding

    frame = _frame("treatment")
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        weights="w",
        treatment_probabilities=_columns(2),
        mechanism_law="sample",
    )
    result = (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            TMLEMethod().with_overrides(
                outcome_learner=_wrong_outcome(),
                treatment_learner=law.WeightedCellClassifier(),
                cross_fit=False,
                random_state=0,
            )
        )
    )
    assert result.config.treatment_mechanism == "estimated"
    readings: list[Any] = []
    original = CausalData.with_known_mechanism

    def spy(self: CausalData, treatment_probabilities: Any, **options: Any) -> CausalData:
        readings.append(options.get("mechanism_law"))
        return original(self, treatment_probabilities, **options)

    monkeypatch.setattr(CausalData, "with_known_mechanism", spy)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
    surface = simulated_confounding(result, "ate", grid=grid, random_state=3)
    assert readings == ["sample", "sample"]
    assert surface.tilt_estimated
    assert any("re-estimates the weight tilt" in line for line in surface.population_lines())


def test_w12_generated_data_validation_carries_the_reading() -> None:
    """Each plasmode draw declares the fit's mechanism with its reading, so its refit runs
    the configuration it validates."""
    from cleverly.validation import GaussianAdjustmentOutcome
    from cleverly.validation._point_generated import PlasmodeLaw, PointTarget

    frame = _frame("treatment")
    result = _fit(frame, "sample", _tmle(estimands=("ate",)))
    plasmode = PlasmodeLaw.fit(result)
    target = PointTarget("difference", "all", 1.0, 0.0)
    draw = plasmode.draw(result.data, GaussianAdjustmentOutcome(), seed=3, target=target)
    assert draw.known_treatment is not None and draw.known_treatment.law == "sample"
    assert not divides_by_declaration(draw)


def test_w12_positivity_reads_the_fitted_tilt_under_sample() -> None:
    """A bound that moves a declared value is refused under ``"target"`` only."""
    frame = _frame("treatment")
    bounds = [0.3]
    with pytest.raises(CapabilityError, match="outside the truncation bounds"):
        truncation_curve(
            _fit(frame, "target", _tmle(NeverFit(), estimands=("ate",))), bounds=bounds
        )
    curve = truncation_curve(_fit(frame, "sample", _tmle(estimands=("ate",))), bounds=bounds)
    assert "known" not in set(curve.get("treatment_mechanism", ["estimated"]))


def test_w12_a_reading_recorded_without_weights_takes_its_route_once_weighted() -> None:
    """A ``"sample"`` reading changes nothing on unweighted data and acts once weights tilt."""
    frame = _frame("treatment_outcome")
    data = CausalData.from_frame(
        frame.drop(columns=["w"]),
        **ROLES,
        treatment_probabilities=_columns(2),
        mechanism_law="sample",
    )
    assert divides_by_declaration(data)
    weighted = CausalData.from_frame(frame, **ROLES, weights="w")
    tilted = dataclasses.replace(data, weights=weighted.weights)
    assert not divides_by_declaration(tilted)
    result = _tmle(estimands=BINARY_TARGETS).fit(tilted).single()
    _assert_exact(result, _tilted_truth(frame))


# ------------------------------------------------------------------- plots and operations


def _sample_study() -> CausalStudy:
    """A study whose design reads the declaration in the sample, with a score column."""
    frame = _frame("treatment")
    frame["S"] = frame["W1"] + 0.5 * frame["W2"]
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        weights="w",
        treatment_probabilities=_columns(2),
        mechanism_law="sample",
    )
    return CausalStudy(frame, design=design)


@pytest.mark.parametrize("weight", ["treatment", "covariates"])
def test_w11_available_methods_reports_the_sample_refusals_of_a_study(weight: str) -> None:
    """A study holds its data, so it reports what the fit would refuse.

    Under ``"sample"`` beside weights that tilt the law, adaptive IPTW and DR-TMLE (whose
    default guard is non-empty) read the estimated mechanism as fixed and refuse.  Weights
    that read the covariates only make the two readings the same law, but a ``"sample"``
    declaration is still estimated, so both stay refused.  A ``"target"`` design divides by
    the declaration and keeps both.
    """
    frame = _frame(weight)

    def methods(mechanism_law: str) -> dict[str, Any]:
        design = PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2"),
            weights="w",
            treatment_probabilities=_columns(2),
            mechanism_law=mechanism_law,
        )
        effect = CausalStudy(frame, design=design).identify(ATE())
        return {record.name: record for record in effect.available_methods()}

    sample = methods("sample")
    assert not sample["adaptive_iptw"].available
    assert sample["adaptive_iptw"].refusal == KNOWN_SAMPLE_ADAPTIVE_IPTW_REFUSAL
    assert not sample["drtmle"].available
    assert sample["drtmle"].refusal == KNOWN_SAMPLE_DRTMLE_GUARD_REFUSAL
    assert sample["tmle"].available
    with pytest.raises(CapabilityError, match="AdaptiveIPTW"):
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W1", "W2"),
                weights="w",
                treatment_probabilities=_columns(2),
                mechanism_law="sample",
            ),
        ).identify(ATE()).estimate("adaptive_iptw")
    target = methods("target")
    assert target["adaptive_iptw"].available and target["drtmle"].available


def _sample_settings(**settings: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "outcome_learner": _wrong_outcome(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    base.update(settings)
    return base


def _sample_coverage(n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rows of the exact law drawn with replacement, with ``Psi(P_w)`` as the truth."""
    frame = _frame("treatment")
    rows = np.random.default_rng(seed).integers(0, len(frame), size=n)
    return frame.iloc[rows].reset_index(drop=True), _tilted_truth(frame)


def test_sample_plots() -> None:
    """Plan section 6: every plot draws on a ``"sample"`` fit, or its input made from one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cleverly import ConditionalContrast, plotting
    from cleverly.bases import Linear, Modifiers
    from cleverly.validation.simulation import CoverageStudy
    from tests.unit._capability_sweep_support import fit_ltmle_known_sample_weights

    study = _sample_study()
    stacked = SuperLearner(
        [LogisticRegression(max_iter=1000), law.WeightedCellClassifier()],
        task="classification",
    )
    result = study.identify(ATE()).estimate(
        **_sample_settings(treatment_learner=stacked, n_bootstrap=20)
    )
    assert result.data.known_treatment.law == "sample"
    repeated = study.identify(ATE()).estimate(
        **_sample_settings(cross_fit=True, n_folds=2, repeats=2)
    )
    longitudinal = fit_ltmle_known_sample_weights()
    contrast = study.identify(ConditionalContrast(Modifiers(["W1"]), model=Linear(["W1"])))
    effect = contrast.estimate(**_sample_settings()).effect_curve()
    curve = contrast.targeting_curve(
        "S",
        computed_from=["W1", "W2"],
        score_kind="external",
        n_half_samples=20,
        **_sample_settings(),
    )
    coverage = CoverageStudy(
        dgp=_sample_coverage,
        estimator=lambda: TMLE(**_sample_settings()),
        n=600,
        n_replicates=3,
        estimands=("ate",),
        fit_kwargs={
            **ROLES,
            "weights": "w",
            "treatment_probabilities": _columns(2),
            "mechanism_law": "sample",
        },
        seed=1,
    ).run()
    figures = {
        "overlap": plotting.overlap(result),
        "balance": plotting.balance(result),
        "influence": plotting.influence(result),
        "learners": plotting.learners(result),
        "calibration": plotting.calibration(result),
        "forest": plotting.forest(result),
        "curve": plotting.curve(effect),
        "bootstrap": plotting.bootstrap(result),
        "repeats": plotting.repeats(repeated),
        "weights": plotting.weights(longitudinal),
        "targeting": plotting.targeting(curve),
        "zipper": plotting.zipper(coverage),
        "confounding": plotting.confounding(result.sensitivity.contour()),
        "sensitivity": plotting.sensitivity(result.diagnostics.truncation_curve()),
    }
    assert isinstance(plotting.notes(figures["forest"]), tuple)
    for name, figure in figures.items():
        assert figure.axes, name
        plt.close(figure)


def test_sample_operations() -> None:
    """Plan section 6: one call for each operation the earlier tests do not reach.

    The benchmark refuses: a refit without the covariates would still divide by the
    declaration that reads them.  ``variable_importance`` takes no declaration, so the
    reading never reaches it.
    """
    from cleverly import ConditionalContrast
    from cleverly.bases import Linear, Modifiers
    from cleverly.effect_curves import score_assessment
    from cleverly.sensitivity import missingness_tilt

    study = _sample_study()
    result = study.identify(ATE()).estimate(**_sample_settings())
    single = result.single() if hasattr(result, "single") else result
    frame = _frame("treatment")
    with pytest.raises(CapabilityError, match="would still divide by the declared mechanism"):
        result.sensitivity.benchmark("W1")
    assert result.diagnostics.refute(
        tests=("random_common_cause",), n_replicates=2, random_state=0
    ).tests
    assert len(result.sensitivity.marginal_bounds().to_frame())
    assert result.sensitivity.evalue() is not None
    scores = {"S": (frame["W1"] + 0.5 * frame["W2"]).to_numpy(dtype=float)}
    assert score_assessment(single, single.data, scores).to_frame() is not None
    curve = (
        study.identify(ConditionalContrast(Modifiers(["W1"]), model=Linear(["W1"])))
        .estimate(**_sample_settings())
        .effect_curve()
    )
    assert np.all(np.isfinite(curve.predict({"W1": np.array([0.0, 1.0])})))
    frame["Delta"] = 1.0
    frame.loc[frame.index[::7], "Delta"] = 0.0
    frame.loc[frame["Delta"] == 0.0, "Y"] = np.nan
    missing = _fit(
        frame,
        "sample",
        _tmle(estimands=("ate",), missingness_learner=LogisticRegression(max_iter=1000)),
        delta="Delta",
    )
    tilt = missingness_tilt(missing, gamma=[0.5, 2.0])
    assert len(tilt.to_frame()) == 2
