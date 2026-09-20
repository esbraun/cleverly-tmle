"""Public, reusable outer-fold assignments for point-treatment fits.

A :class:`~cleverly.SplitPlan` is the *realised* outer split, read off one result and
handed to the next.  Three questions decide whether that is a useful object, and this
module answers each one:

* Does the second fit run the split the first one ran?  The identity tests below compare
  the two fits at every result-determining layer, on a binary, a multi-arm, a clustered,
  a repeated and a continuous-dose fit.
* Does the plan freeze *only* the outer folds?  ``TestASuppliedPlanFreezesTheOuterSplit``
  is the witness.  A plan that also froze the learner folds or the collaborative
  selection folds would pass every identity test above, because those are run at one
  seed, so the witness holds the plan fixed and moves the seed instead.
* Does the plan refuse the rows it cannot label?  A fold label is a position, so a plan
  reused on other rows, on a different row count, or on a refit that drops rows is
  refused, and each refusal is asserted by message rather than only by type.
* Does the fit know where the labels came from?  A fit accepts a plan only when the plan
  records how :func:`~cleverly.learners.random_partition` drew each repeat, and draws each
  repeat again to check it.  ``TestOnlyARecordedDrawIsAccepted`` is the witness: a
  hand-built plan and a forged one are refused before any learner runs.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly import (
    ATE,
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    CollaborativeTMLEMethod,
    CrossFitting,
    DRTMLEMethod,
    Inference,
    LongitudinalTreatment,
    ModelSpec,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    RegimeMean,
    Runtime,
    SplitPlan,
    SuperLearner,
    Targeting,
    TMLEMethod,
)
from cleverly._typing import FoldStrata, IntArray
from cleverly.datasets import (
    make_longitudinal,
    make_nonlinear_bounded,
    make_shift_dose,
    multi_arm_dgp,
)
from cleverly.estimators.serialize import dumps, loads
from cleverly.estimators.tmle import TMLE as TMLEEngine
from cleverly.exceptions import DataError, MethodConfigurationError
from cleverly.interventions import Shift
from cleverly.learners import FoldOrigin, random_partition
from cleverly.learners.crossfit import (
    _UNRECORDED_PLAN_REASON,
    RANDOM_PARTITION_GENERATOR,
)
from cleverly.provenance import data_fingerprint, fold_fingerprint
from cleverly.validation import refute

N_FOLDS = 3
SEED = 17

#: Candidate names the ensemble below scores, in library order.
ENSEMBLE_NAMES = ("LinearRegression", "DecisionTreeRegressor")


def _models() -> ModelSpec:
    """The cheap, deterministic pair every fit uses unless the test is about learning."""
    return ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    )


def _ensemble_models() -> ModelSpec:
    """An outcome learner whose inner folds can actually move the fit.

    Two candidates is the minimum.  With one, the inner split changes no weight and no
    prediction, so the seed witness below would pass against an implementation that froze
    the learner folds along with the outer ones.

    Both candidates are seeded, so the inner fold split is the *only* thing the estimator
    seed can move.  An unseeded tree draws its own feature order from the global numpy
    state, which would make the witness pass on ambient randomness rather than on the
    behavior it is there to observe.
    """
    return ModelSpec(
        outcome_learner=SuperLearner(
            library=[LinearRegression(), DecisionTreeRegressor(max_depth=2, random_state=0)],
            n_folds=2,
        ),
        treatment_learner=LogisticRegression(max_iter=1000),
    )


def _method(
    *,
    split_plan: SplitPlan | None = None,
    repeats: int = 1,
    n_jobs: int = 1,
    n_bootstrap: int = 0,
    fold_evaluation: bool = False,
    enabled: bool = True,
    n_folds: int = N_FOLDS,
    stratify_by: FoldStrata = "none",
    q_bounds: tuple[float, float] | None = (0.0, 1.0),
    random_state: int = SEED,
    models: ModelSpec | None = None,
    kind: type[TMLEMethod] = TMLEMethod,
    **extra: Any,
) -> TMLEMethod:
    """Build one method declaration, with every knob these tests turn.

    ``kind`` and ``extra`` reach the collaborative and reduced-dimension configurations,
    which add fields to the same five groups rather than replacing them.

    ``stratify_by`` defaults to ``"none"``, the only value a cross-fitted fit accepts
    (the fold and outcome-scale rules); ``q_bounds`` defaults to ``(0.0, 1.0)``, this module's frames now being drawn
    from :func:`~cleverly.datasets.make_nonlinear_bounded`, whose proportion outcome has
    that support. Both default to what a cross-fitted fit needs, since ``enabled=True``
    is this helper's own default.
    """
    return kind(
        models=_models() if models is None else models,
        cross_fitting=CrossFitting(
            enabled=enabled,
            n_folds=n_folds,
            learner_folds=2,
            repeats=repeats,
            stratify_by=stratify_by,
            fold_evaluation=fold_evaluation,
            split_plan=split_plan,
        ),
        targeting=Targeting(q_bounds=q_bounds),
        inference=Inference(simultaneous=False, n_bootstrap=n_bootstrap),
        runtime=Runtime(random_state=random_state, n_jobs=n_jobs),
        **extra,
    )


def _study(frame: Any, *, cluster: str | None = None) -> Any:
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            cluster=cluster,
        ),
    )


def _effect(frame: Any, *, cluster: str | None = None) -> Any:
    return _study(frame, cluster=cluster).identify(ATE())


def _backed(frame: pd.DataFrame, backend: str) -> Any:
    """Return one built frame under the requested backend."""
    return frame if backend == "pandas" else pl.from_pandas(frame)


def _never_called(reason: str) -> Callable[..., Any]:
    """Return a stand-in that fails the test when anything calls it."""

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(reason)

    return unexpected


def _package_plan(
    n: int,
    *,
    n_folds: int = N_FOLDS,
    repeats: int = 1,
    cluster: IntArray | None = None,
    seed: int = SEED,
) -> SplitPlan:
    """Return a plan the package drew, which is the only kind a fit accepts.

    ``random_partition`` is the generator every accepted plan records, and
    :meth:`SplitPlan.from_folds` copies that record off the draw.  A fit draws each
    repeat again from the record and compares it label for label, so a plan built any
    other way is refused before its first learner.
    """
    return SplitPlan.from_folds(
        random_partition(n, n_folds, cluster=cluster, seed=seed + repeat)
        for repeat in range(repeats)
    )


def _cluster_codes(frame: Any, *, cluster: str | None) -> IntArray | None:
    """Return the cluster codes a fit on ``frame`` will draw grouped folds from.

    A grouped draw reads the codes the data carry, so the plan has to be drawn from the
    same vector the fit will hand to ``verify``.
    """
    if cluster is None:
        return None
    return _study(frame, cluster=cluster).data.cluster


def _generated_then_supplied(
    frame: Any, *, cluster: str | None = None, plan_seed: int = SEED, **knobs: Any
) -> tuple[Any, Any]:
    """Fit once on a generated draw, then again on the plan that first fit realised.

    ``stratify_by="none"`` is the default here, so the first fit draws its own split
    through the generator a plan has to record.  That is the pair the reuse contract is
    about: nothing is supplied until the second fit reads ``result.split_plan``.  A
    caller that declares a stratified policy supplies the first draw instead, because a
    stratified split records no generator and no fit accepts it back.
    """
    knobs.setdefault("stratify_by", "none")
    effect = _effect(frame, cluster=cluster)
    if knobs["stratify_by"] == "none":
        generated = effect.estimate(method=_method(random_state=plan_seed, **knobs))
        assert generated.split_plan.provenance is not None
    else:
        plan = _package_plan(
            len(frame),
            n_folds=knobs.get("n_folds", N_FOLDS),
            repeats=knobs.get("repeats", 1),
            cluster=_cluster_codes(frame, cluster=cluster),
            seed=plan_seed,
        )
        generated = effect.estimate(method=_method(split_plan=plan, **knobs))
        assert generated.split_plan.provenance == plan.provenance
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan, **knobs))
    return generated, supplied


def _assert_same_fit(generated: Any, supplied: Any) -> None:
    """Assert identity at every result-determining layer of the fit."""
    assert generated.split_plan == supplied.split_plan
    assert generated.split_plan.fingerprint == supplied.split_plan.fingerprint
    assert generated.provenance.fold_fingerprint == supplied.provenance.fold_fingerprint
    assert supplied.split_plan.fingerprint == supplied.provenance.fold_fingerprint

    assert len(generated.repeats) == len(supplied.repeats)
    for expected, actual in zip(generated.repeats, supplied.repeats, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)
        np.testing.assert_array_equal(
            expected.nuisance.propensity.values, actual.nuisance.propensity.values
        )
        np.testing.assert_array_equal(
            expected.nuisance.outcome.observed, actual.nuisance.outcome.observed
        )
        assert expected.nuisance.outcome.arms.keys() == actual.nuisance.outcome.arms.keys()
        for arm, values in expected.nuisance.outcome.arms.items():
            np.testing.assert_array_equal(values, actual.nuisance.outcome.arms[arm])
        assert expected.psi == actual.psi

    assert generated.estimates.keys() == supplied.estimates.keys()
    for name, expected in generated.estimates.items():
        actual = supplied.estimates[name]
        assert expected.psi == actual.psi
        assert expected.variance == actual.variance
        np.testing.assert_array_equal(expected.influence_curve, actual.influence_curve)

    # These are the fitted fold labels that diagnostics and fold-indexed artifacts read.
    for expected, actual in zip(generated.nuisances, supplied.nuisances, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)


class TestSplitPlanRecord:
    def test_sequences_are_normalized_to_an_immutable_value(self) -> None:
        source = [[0, 1, 0, 1], [1, 0, 1, 0]]
        plan = SplitPlan(source)

        assert plan.assignments == ((0, 1, 0, 1), (1, 0, 1, 0))
        assert plan.n == 4
        assert plan.n_folds == 2
        assert plan.n_repeats == 2
        assert len(plan.fingerprint) == 16
        source[0][0] = 1
        assert plan.assignments[0][0] == 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.assignments = ((0, 1),)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "assignments",
        [
            [],
            [[]],
            [[0, 1], [0]],
            [[0, -1]],
            [[0, 2]],
            [[0.0, 1.5]],
        ],
        ids=(
            "no-repeats",
            "no-rows",
            "unequal-row-counts",
            "negative-label",
            "missing-label",
            "non-integral-label",
        ),
    )
    def test_malformed_assignments_are_refused(self, assignments: Any) -> None:
        with pytest.raises(DataError):
            SplitPlan(assignments)

    def test_materialize_returns_independent_internal_arrays(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        first = plan.to_folds()
        second = plan.to_folds()

        assert len(first) == len(second) == plan.n_repeats
        for left, right, expected in zip(first, second, plan.assignments, strict=True):
            np.testing.assert_array_equal(left.assignment, expected)
            np.testing.assert_array_equal(right.assignment, expected)
            assert left.assignment is not right.assignment

    def test_one_fold_is_a_valid_record_for_an_in_sample_result(self) -> None:
        plan = SplitPlan([[0, 0, 0]])

        assert plan.n_folds == 1
        assert plan.to_folds()[0].is_single

    def test_fingerprint_is_canonical_and_sensitive_to_every_repeat(self) -> None:
        first = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        same = SplitPlan(np.asarray(first.assignments, dtype=np.int32))
        reordered = SplitPlan(tuple(reversed(first.assignments)))

        assert same == first
        assert hash(same) == hash(first)
        assert same.fingerprint == first.fingerprint
        assert reordered.fingerprint != first.fingerprint

    def test_one_function_defines_the_fold_digest_for_the_plan_and_for_provenance(self) -> None:
        """Nothing but calling one function twice makes the two digests comparable.

        ``fingerprint_array`` folds in the dtype, so a plan's tuples and a fit's ``int64``
        arrays digest differently unless both go through ``fold_fingerprint``.  A caller
        who cannot compare ``plan.fingerprint`` with ``provenance.fold_fingerprint``
        cannot tell whether the second fit ran the split the first one did.
        """
        plan = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        as_int64 = [np.asarray(draw, dtype=np.int64) for draw in plan.assignments]
        as_int32 = [np.asarray(draw, dtype=np.int32) for draw in plan.assignments]

        assert plan.fingerprint == fold_fingerprint(as_int64) == fold_fingerprint(as_int32)

    def test_the_representation_summarizes_rather_than_prints_every_label(self) -> None:
        """A plan holds one label per row, and a traceback asked for one line."""
        plan = SplitPlan([list(np.arange(5000) % 4)], source_fingerprint="abc123")
        text = repr(plan)

        assert text == (
            "SplitPlan(n=5000, n_folds=4, n_repeats=1, "
            f"fingerprint={plan.fingerprint}, generator=None, source=abc123)"
        )
        assert len(text) < 200

    def test_the_representation_names_the_generator_a_plan_records(self) -> None:
        """``generator=`` is what says whether a fit can accept the plan at all."""
        plan = _package_plan(120, n_folds=4)
        text = repr(plan)

        assert text == (
            "SplitPlan(n=120, n_folds=4, n_repeats=1, "
            f"fingerprint={plan.fingerprint}, generator={RANDOM_PARTITION_GENERATOR})"
        )

    def test_a_hand_built_plan_is_bound_to_no_data(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])

        assert plan.source_fingerprint is None
        # Bound to nothing, so no fingerprint can contradict it.
        assert len(plan.validate(n=4, source_fingerprint="whatever")) == 1

    def test_the_binding_must_be_a_digest_string(self) -> None:
        with pytest.raises(DataError, match="source_fingerprint"):
            SplitPlan([[0, 1, 0, 1]], source_fingerprint=17)  # type: ignore[arg-type]

    def test_the_stratification_argument_is_named_as_make_folds_names_it(self) -> None:
        """``strata`` is the survey design role in this codebase, and this is not it."""
        plan = SplitPlan([[0, 1, 0, 1]])
        balanced = np.array([0.0, 1.0, 1.0, 0.0])

        assert len(plan.validate(n=4, stratify=balanced)) == 1
        with pytest.raises(TypeError, match="strata"):
            plan.validate(n=4, strata=balanced)  # type: ignore[call-arg]


#: Every learner fit these tests observe, in call order. A refusal witness reads it empty.
_LEARNER_CALLS: list[str] = []


class _CountedRegression(LinearRegression):
    """A learner that records each fit, so a refusal can be shown to precede all of them."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        _LEARNER_CALLS.append("outcome")
        return super().fit(X, y, sample_weight)


class _CountedLogistic(LogisticRegression):
    """The propensity half of the same counter."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        _LEARNER_CALLS.append("treatment")
        return super().fit(X, y, sample_weight)


def _counted_models() -> ModelSpec:
    """The cheap pair, counting every fit.

    Cloning keeps the class, so a counter on the class reaches every clone the estimator
    makes. An instance counter would stay at zero however many fits ran.
    """
    return ModelSpec(
        outcome_learner=_CountedRegression(),
        treatment_learner=_CountedLogistic(max_iter=1000),
    )


class TestOnlyARecordedDrawIsAccepted:
    """A plan is accepted for where its labels came from, not only for their shape.

    ``SplitPlan.validate`` reads the labels against the data and cannot see what chose
    them: an assignment that puts every treated unit in one fold passes cluster integrity
    and training support, and labels chosen by reading the outcome pass both as well. What
    rules that out is the generator record, and drawing the split again from it.
    """

    @pytest.fixture(autouse=True)
    def _reset_counter(self) -> Any:
        _LEARNER_CALLS.clear()
        yield
        _LEARNER_CALLS.clear()

    def test_a_package_draw_carries_its_record_through_the_plan_and_back(self) -> None:
        folds = random_partition(120, N_FOLDS, seed=SEED)
        plan = SplitPlan.from_folds([folds])

        assert plan.provenance == (folds.origin,)
        assert plan.provenance[0] == FoldOrigin(
            generator=RANDOM_PARTITION_GENERATOR,
            scheme="vfold",
            requested_n_folds=N_FOLDS,
            seed=SEED,
        )
        # And back out again, so a fit on this plan returns a plan with the same record.
        assert [draw.origin for draw in plan.to_folds()] == [folds.origin]

    def test_a_hand_built_plan_records_nothing(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])

        assert plan.provenance is None
        assert [draw.origin for draw in plan.to_folds()] == [None]

    def test_one_unrecorded_repeat_leaves_the_whole_plan_unrecorded(self) -> None:
        """A plan is accepted repeat by repeat, so a partial record is no record."""
        drawn = random_partition(120, N_FOLDS, seed=SEED)
        hand_built = dataclasses.replace(drawn, origin=None)

        assert SplitPlan.from_folds([drawn, hand_built]).provenance is None

    @pytest.mark.parametrize(
        ("provenance", "match"),
        [
            ((), "one origin per repeat"),
            (
                (
                    FoldOrigin(RANDOM_PARTITION_GENERATOR, "vfold", 2, 0),
                    FoldOrigin(RANDOM_PARTITION_GENERATOR, "vfold", 2, 1),
                ),
                "one origin per repeat",
            ),
            (("cleverly.random_partition/1",), "must be a FoldOrigin"),
        ],
        ids=("no-records", "too-many-records", "not-a-record"),
    )
    def test_a_malformed_record_is_refused_at_construction(
        self, provenance: Any, match: str
    ) -> None:
        with pytest.raises(DataError, match=match):
            SplitPlan([[0, 1, 0, 1]], provenance=provenance)

    def test_a_pickle_that_predates_the_field_restores_unrecorded(self) -> None:
        """An old stored plan reads as what it is: labels with no generator record."""
        plan = _package_plan(120)
        state = {
            "assignments": plan.assignments,
            "source_fingerprint": plan.source_fingerprint,
        }
        restored = SplitPlan.__new__(SplitPlan)
        restored.__setstate__(state)

        assert restored.provenance is None
        assert restored.assignments == plan.assignments

    def test_verify_accepts_the_draw_it_records(self) -> None:
        _package_plan(120).verify(n=120)

    def test_verify_refuses_a_plan_with_no_record(self) -> None:
        with pytest.raises(DataError, match="no generator record"):
            SplitPlan([[0, 1, 0, 1]]).verify(n=4)

    def test_verify_refuses_one_changed_label(self) -> None:
        """The forgery a shape check cannot see: a valid split, drawn by nobody."""
        plan = _package_plan(120)
        forged = list(plan.assignments[0])
        forged[0] = (forged[0] + 1) % N_FOLDS
        plan = SplitPlan([forged], provenance=plan.provenance)

        with pytest.raises(DataError, match=r"repeat 0 differs .* at 1 row"):
            plan.verify(n=120)

    def test_verify_refuses_a_record_naming_another_seed(self) -> None:
        plan = _package_plan(120)
        origin = dataclasses.replace(plan.provenance[0], seed=plan.provenance[0].seed + 1)

        with pytest.raises(DataError, match="repeat 0 differs"):
            SplitPlan(plan.assignments, provenance=(origin,)).verify(n=120)

    def test_verify_refuses_a_record_naming_another_generator(self) -> None:
        plan = _package_plan(120)
        origin = dataclasses.replace(plan.provenance[0], generator="somebody.else/3")

        with pytest.raises(DataError, match=re.escape("records generator 'somebody.else/3'")):
            SplitPlan(plan.assignments, provenance=(origin,)).verify(n=120)

    def test_verify_requires_the_grouped_scheme_exactly_when_clusters_are_declared(self) -> None:
        codes = np.repeat(np.arange(40), 3)
        grouped = _package_plan(120, cluster=codes)
        rows = _package_plan(120)

        grouped.verify(n=120, cluster=codes)
        rows.verify(n=120)
        with pytest.raises(DataError, match="records a 'vfold' draw"):
            rows.verify(n=120, cluster=codes)
        with pytest.raises(DataError, match="records a 'grouped' draw"):
            grouped.verify(n=120)

    def test_verify_names_the_repeat_that_differs(self) -> None:
        plan = _package_plan(120, repeats=3)
        forged = list(plan.assignments[2])
        forged[5] = (forged[5] + 1) % N_FOLDS
        plan = SplitPlan(
            [plan.assignments[0], plan.assignments[1], forged], provenance=plan.provenance
        )

        with pytest.raises(DataError, match="repeat 2 differs"):
            plan.verify(n=120)

    def test_verify_refuses_another_row_count_before_it_draws(self) -> None:
        with pytest.raises(DataError, match="120 rows but the data have 119 rows"):
            _package_plan(120).verify(n=119)

    def test_verify_resolves_no_fold_count_for_the_fit(self) -> None:
        """The cap belongs to the draw that made the plan, not to the check of it."""
        codes = np.repeat(np.arange(4), 30)
        with pytest.warns(UserWarning, match="only 4 clusters"):
            plan = _package_plan(120, n_folds=10, cluster=codes)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan.verify(n=120, cluster=codes)

        assert not caught
        assert plan.n_folds == 4

    def test_unbound_keeps_the_record_and_drops_the_binding(self) -> None:
        bound = SplitPlan.from_folds(
            [random_partition(120, N_FOLDS, seed=SEED)], source_fingerprint="abc123"
        )
        free = bound.unbound()

        assert free.source_fingerprint is None
        assert free.provenance == bound.provenance
        assert free.assignments == bound.assignments

    def test_a_hand_built_plan_is_refused_at_declaration_before_any_learner(self) -> None:
        frame, _ = make_nonlinear_bounded(n=120, seed=61)
        plan = SplitPlan([list(np.arange(120) % N_FOLDS)])

        with pytest.raises(MethodConfigurationError, match="no generator record"):
            _effect(frame).estimate(method=_method(split_plan=plan, models=_counted_models()))

        assert _LEARNER_CALLS == []

    def test_a_forged_plan_is_refused_before_any_learner(self) -> None:
        """Valid record, one label moved: the fit draws the split again and sees it."""
        frame, _ = make_nonlinear_bounded(n=120, seed=61)
        drawn = _package_plan(120)
        forged = list(drawn.assignments[0])
        forged[0] = (forged[0] + 1) % N_FOLDS
        plan = SplitPlan([forged], provenance=drawn.provenance)

        with pytest.raises(DataError, match="repeat 0 differs"):
            _effect(frame).estimate(method=_method(split_plan=plan, models=_counted_models()))

        assert _LEARNER_CALLS == []

    def test_the_counter_is_a_live_witness_under_the_plan_that_was_drawn(self) -> None:
        """The nonzero control: the same fit on the unforged plan fits learners."""
        frame, _ = make_nonlinear_bounded(n=120, seed=61)
        result = _effect(frame).estimate(
            method=_method(split_plan=_package_plan(120), models=_counted_models())
        )

        assert result.split_plan.provenance == _package_plan(120).provenance
        assert _LEARNER_CALLS.count("outcome") >= N_FOLDS
        assert _LEARNER_CALLS.count("treatment") >= N_FOLDS

    def test_neutering_the_record_check_lets_the_forgery_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation that shows the refusal above is ``verify``'s work.

        With the comparison removed, the forged labels pass every remaining check: they
        are contiguous, they carry both arms into every training complement, and they
        fingerprint as a plan.  The fit runs, which is the failure the record prevents.
        """
        frame, _ = make_nonlinear_bounded(n=120, seed=61)
        drawn = _package_plan(120)
        forged = list(drawn.assignments[0])
        forged[0] = (forged[0] + 1) % N_FOLDS
        plan = SplitPlan([forged], provenance=drawn.provenance)

        monkeypatch.setattr(SplitPlan, "verify", lambda self, **kwargs: None)
        result = _effect(frame).estimate(method=_method(split_plan=plan, models=_counted_models()))

        assert result.split_plan.assignments == plan.assignments
        assert _LEARNER_CALLS


class TestCrossFittingConfiguration:
    def test_disabled_cross_fitting_refuses_repeated_splits_at_declaration(self) -> None:
        with pytest.raises(MethodConfigurationError, match="enabled=False makes no split"):
            CrossFitting(enabled=False, repeats=2)

        with pytest.raises(MethodConfigurationError, match="enabled=False makes no split"):
            TMLEMethod().with_overrides(cross_fit=False, repeats=2)

    def test_flat_shortcut_sets_the_plan_on_the_normalized_configuration(self) -> None:
        plan = _package_plan(120, n_folds=2)

        method = TMLEMethod().with_overrides(n_folds=2, repeats=1, split_plan=plan)

        assert method.cross_fitting.split_plan is plan

    def test_more_folds_than_declared_is_refused(self) -> None:
        plan = _package_plan(120, n_folds=3)
        with pytest.raises(MethodConfigurationError, match="n_folds"):
            CrossFitting(n_folds=2, split_plan=plan)

    def test_fewer_folds_than_declared_is_accepted(self) -> None:
        """The direction a cap produces.

        ``resolve_n_folds`` caps the declared count at the rarest stratum and again at the
        cluster count, so a plan this package wrote under a 10-fold declaration can hold
        three.  Refusing it here would refuse the package's own record of its own fit.
        What the fit then needs of the plan is not a count at all: ``SplitPlan.validate``
        checks the labels against the data once they are in hand.
        """
        plan = _package_plan(120, n_folds=3)

        assert CrossFitting(n_folds=10, split_plan=plan).split_plan is plan

    def test_repeat_count_must_match_the_supplied_plan(self) -> None:
        plan = _package_plan(120, n_folds=2, repeats=2)
        with pytest.raises(MethodConfigurationError, match="repeats"):
            CrossFitting(n_folds=2, repeats=1, split_plan=plan)

    def test_a_plan_conflicts_with_disabled_cross_fitting(self) -> None:
        plan = _package_plan(120, n_folds=2)
        with pytest.raises(MethodConfigurationError, match=r"enabled|cross.?fit"):
            CrossFitting(enabled=False, n_folds=2, split_plan=plan)

    def test_one_declared_fold_cannot_carry_a_supplied_plan(self) -> None:
        plan = _package_plan(120, n_folds=2)
        with pytest.raises(MethodConfigurationError, match=r"n_folds|cross.?fit"):
            CrossFitting(enabled=True, n_folds=1, split_plan=plan)

    def test_a_one_fold_plan_records_no_draw_and_is_refused_for_that(self) -> None:
        """``random_partition`` draws two folds at least, so a one-fold plan is hand-built."""
        plan = SplitPlan([[0, 0, 0]])
        with pytest.raises(MethodConfigurationError, match="no generator record"):
            CrossFitting(enabled=True, n_folds=2, split_plan=plan)

    def test_a_plan_without_a_generator_record_is_refused(self) -> None:
        plan = SplitPlan([list(np.arange(120) % N_FOLDS)])
        with pytest.raises(MethodConfigurationError, match="no generator record"):
            CrossFitting(n_folds=N_FOLDS, split_plan=plan)

    def test_only_a_split_plan_is_accepted(self) -> None:
        with pytest.raises(MethodConfigurationError, match="split_plan"):
            CrossFitting(n_folds=2, split_plan=[[0, 1]])  # type: ignore[arg-type]

    def test_a_repeat_count_below_one_is_refused_at_declaration(self) -> None:
        with pytest.raises(MethodConfigurationError, match="repeats must be at least 1; got 0"):
            CrossFitting(repeats=0)


class TestOneInputEarnsOneReasonAtBothLayers:
    """The declaration and the engine read one ordered refusal, so neither can disagree.

    Each case below trips more than one check where it can, so a layer that reordered its
    checks would report a different reason from the other layer.  The engine spells the
    cross-fitting switch ``cross_fit`` and the declaration spells it ``enabled``; the
    reason names the spelling its caller wrote and is otherwise identical.
    """

    @pytest.mark.parametrize(
        ("settings", "expected"),
        [
            (
                {
                    "enabled": False,
                    "repeats": 2,
                    "split_plan": _package_plan(120, n_folds=2, repeats=2),
                },
                "split_plan requires enabled cross-fitting with at least two folds",
            ),
            (
                {
                    "enabled": False,
                    "repeats": 2,
                    "split_plan": SplitPlan([list(np.arange(120) % 2)] * 2),
                },
                _UNRECORDED_PLAN_REASON,
            ),
            (
                {"repeats": 0, "split_plan": [[0, 1, 0, 1]]},
                "repeats must be at least 1; got 0",
            ),
            (
                {"repeats": 1, "split_plan": [[0, 1, 0, 1]]},
                "split_plan must be a SplitPlan",
            ),
            (
                {"split_plan": _package_plan(120, n_folds=2), "n_bootstrap": 4},
                "n_bootstrap cannot be combined with split_plan: targeted bootstrap "
                "replicates duplicate sampled rows, while the supplied assignments "
                "identify only the original row positions",
            ),
            (
                {"enabled": False, "repeats": 2},
                "repeats takes the median over independent cross-fitting splits, and "
                "{switch}=False makes no split to draw or repeat. Enable cross-fitting or "
                "set repeats=1",
            ),
        ],
        ids=(
            "plan-before-repeats",
            "record-before-everything",
            "repeats-first",
            "not-a-plan",
            "bootstrap",
            "no-split",
        ),
    )
    def test_the_declaration_and_the_engine_give_the_same_reason(
        self, settings: dict[str, Any], expected: str
    ) -> None:
        enabled = settings.get("enabled", True)
        repeats = settings.get("repeats", 1)
        split_plan = settings.get("split_plan")
        n_bootstrap = settings.get("n_bootstrap", 0)

        with pytest.raises(MethodConfigurationError) as declared:
            TMLEMethod(
                cross_fitting=CrossFitting(
                    enabled=enabled,
                    n_folds=2,
                    repeats=repeats,
                    split_plan=split_plan,
                ),
                inference=Inference(n_bootstrap=n_bootstrap),
            )
        with pytest.raises(ValueError) as engine:
            TMLEEngine(
                cross_fit=enabled,
                n_folds=2,
                repeats=repeats,
                split_plan=split_plan,
                n_bootstrap=n_bootstrap,
                simultaneous=False,
            )

        assert str(declared.value) == expected.format(switch="enabled")
        assert str(engine.value) == expected.format(switch="cross_fit")


class TestTheEngineRefusesAPlanItCannotServe:
    """The engine's own refusals, which the configuration layer normally reaches first.

    ``CrossFitting.__post_init__`` fires at declaration, so a fit driven from
    :class:`~cleverly.CausalStudy` never reaches these branches.  The engine is public
    enough to be constructed directly, and both callers read one message source
    (``SplitPlan._policy_refusal``) under two exception contracts, so both contracts need
    a test.  The engine raises :class:`ValueError`; the configuration raises
    :class:`~cleverly.exceptions.MethodConfigurationError`.
    """

    @pytest.mark.parametrize(
        ("settings", "match"),
        [
            ({"split_plan": [[0, 1, 0, 1]]}, "split_plan must be a SplitPlan"),
            ({"split_plan": SplitPlan([[0, 1, 0, 1]])}, "no generator record"),
            ({"split_plan": _package_plan(120, n_folds=2), "cross_fit": False}, "cross-fitting"),
            ({"split_plan": _package_plan(120, n_folds=2), "n_folds": 1}, "cross-fitting"),
            (
                {"split_plan": _package_plan(120, n_folds=3), "n_folds": 2},
                "3 folds but n_folds is 2",
            ),
            ({"split_plan": _package_plan(120, n_folds=2, repeats=2), "repeats": 1}, "2 repeats"),
            (
                {"split_plan": _package_plan(120, n_folds=2), "n_bootstrap": 4},
                "n_bootstrap cannot be combined",
            ),
        ],
        ids=(
            "not-a-plan",
            "no-record",
            "cross-fit-disabled",
            "one-fold",
            "more-folds-than-declared",
            "repeat-mismatch",
            "bootstrap",
        ),
    )
    def test_the_engine_refuses_at_construction(self, settings: Any, match: str) -> None:
        keywords = {"n_folds": 2, **settings}
        with pytest.raises(ValueError, match=match):
            TMLEEngine(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                learner_folds=2,
                random_state=SEED,
                simultaneous=False,
                **keywords,
            )

    def test_the_engine_reuses_a_plan_it_accepts(self) -> None:
        frame, _ = make_nonlinear_bounded(n=120, seed=41)
        covariates = ["W1", "W2", "W3"]

        def fit(plan: SplitPlan | None) -> Any:
            return (
                TMLEEngine(
                    outcome_learner=LinearRegression(),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    n_folds=N_FOLDS,
                    learner_folds=2,
                    split_plan=plan,
                    random_state=SEED,
                    simultaneous=False,
                    q_bounds=(0.0, 1.0),
                )
                .fit(frame, outcome="Y", treatment="A", covariates=covariates)
                .single()
            )

        generated = fit(_package_plan(120))
        supplied = fit(generated.split_plan)

        _assert_same_fit(generated, supplied)


class TestDataBoundValidationPrecedesNuisanceFitting:
    """Two checks stand between a supplied plan and the first learner, in this order.

    :meth:`SplitPlan.verify` draws each repeat again from its record, and
    :meth:`SplitPlan.validate` reads the labels against the data.  The fit-level tests
    here trip ``verify``, because a plan that reaches ``validate`` at all is one the
    recorded generator drew, and a draw that reads no column cannot be built to strand an
    arm on demand.  ``validate``'s own refusals are therefore checked on the method, which
    is public and takes the data as arguments.
    """

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(9)
        cells = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (12, 1))
        return pd.DataFrame(
            {
                "Y": cells[:, 1].astype(float),
                "A": cells[:, 0].astype(float),
                "W1": rng.normal(size=len(cells)),
                "W2": rng.normal(size=len(cells)),
                "W3": rng.normal(size=len(cells)),
                "pid": np.repeat(np.arange(len(cells) // 2), 2),
            }
        )

    def _refuse_before_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        frame: pd.DataFrame,
        plan: SplitPlan,
        *,
        match: str,
        stratify_by: FoldStrata = "none",
        cluster: str | None = None,
    ) -> None:
        monkeypatch.setattr(
            TMLEEngine,
            "_nuisances",
            _never_called("nuisance fitting began before the supplied split was checked"),
        )
        method = _method(
            n_folds=plan.n_folds,
            repeats=plan.n_repeats,
            stratify_by=stratify_by,
            split_plan=plan,
            q_bounds=None,  # this class's frame has a binary outcome
        )
        with pytest.raises(DataError, match=match):
            _effect(frame, cluster=cluster).estimate(method=method)

    def test_row_count_is_checked_before_fit(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _package_plan(len(frame) - 1)
        self._refuse_before_fit(monkeypatch, frame, plan, match="rows but the data have")

    def test_a_changed_label_is_caught_before_fit(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = _package_plan(len(frame))
        forged = list(drawn.assignments[0])
        forged[0] = (forged[0] + 1) % N_FOLDS
        plan = SplitPlan([forged], provenance=drawn.provenance)
        self._refuse_before_fit(monkeypatch, frame, plan, match="repeat 0 differs")

    def test_a_declared_cluster_needs_a_grouped_draw(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row-level draw cuts across clusters, and the record says it is one."""
        plan = _package_plan(len(frame), n_folds=2)
        self._refuse_before_fit(
            monkeypatch, frame, plan, cluster="pid", match="records a 'vfold' draw"
        )

    def test_each_training_complement_keeps_every_treatment_arm(self, frame: pd.DataFrame) -> None:
        assignment = np.ones(len(frame), dtype=int)
        assignment[frame["A"].to_numpy() == 1] = 0

        with pytest.raises(DataError, match="treatment arm"):
            SplitPlan([assignment]).validate(
                n=len(frame), treatment=frame["A"].to_numpy(), stratify=frame["A"].to_numpy()
            )

    def test_each_training_complement_keeps_every_requested_stratum(
        self, frame: pd.DataFrame
    ) -> None:
        assignment = np.ones(len(frame), dtype=int)
        rare_cell = (frame["A"].to_numpy() == 1) & (frame["Y"].to_numpy() == 1)
        assignment[rare_cell] = 0
        # Keep both treatment arms in fold 0, so the treatment-support check alone passes.
        assignment[[0, 2]] = 0
        crossed = np.unique(
            np.column_stack([frame["A"].to_numpy(), frame["Y"].to_numpy()]),
            axis=0,
            return_inverse=True,
        )[1].astype(float)

        with pytest.raises(DataError, match="stratum"):
            SplitPlan([assignment]).validate(
                n=len(frame), treatment=frame["A"].to_numpy(), stratify=crossed
            )

    def test_a_declared_cluster_cannot_be_split(self, frame: pd.DataFrame) -> None:
        assignment = np.arange(len(frame)) % 2

        with pytest.raises(DataError, match="more than one fold"):
            SplitPlan([assignment]).validate(n=len(frame), cluster=frame["pid"].to_numpy())


def test_generated_folds_do_not_acquire_supplied_plan_support_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_nonlinear_bounded(n=120, seed=14)

    monkeypatch.setattr(
        SplitPlan,
        "validate",
        _never_called("a generated split entered the supplied-plan validator"),
    )
    result = _effect(frame).estimate(method=_method())

    assert result.split_plan.n_folds == N_FOLDS


def test_in_sample_gaussian_fit_ignores_crossfit_only_outcome_strata() -> None:
    frame, _ = make_nonlinear_bounded(n=120, seed=15)

    result = _effect(frame).estimate(method=_method(enabled=False, stratify_by="treatment+outcome"))

    assert result.split_plan.n_folds == 1


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_binary_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=4, backend=backend)
    generated, supplied = _generated_then_supplied(frame)

    _assert_same_fit(generated, supplied)
    assert supplied.data.backend == backend


@pytest.mark.filterwarnings("error::cleverly.exceptions.PositivityWarning")
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_multi_arm_plans_are_exactly_identical(backend: str) -> None:
    # Split replay is the subject. Mild confounding keeps this fixture inside the
    # automatic bounds instead of coupling the identity check to truncation.
    # The draw is named because the claim below is about this split: at another seed one
    # unit falls outside the automatic bounds and the truncation assertion is not about
    # the plan any more.
    # A binary outcome: this fit is cross-fitted, and Targeting refuses a declared
    # q_bounds on a binary outcome outright rather than merely not needing one.
    frame, _ = multi_arm_dgp(confounding=0.1, family="binomial").sample(
        n=180, seed=7, backend=backend
    )
    generated, supplied = _generated_then_supplied(frame, plan_seed=6, q_bounds=None)

    _assert_same_fit(generated, supplied)
    assert generated.nuisance.propensity.values.shape[1] == 3
    # The warning filter above fires only above the 5% warning threshold. The fixture
    # claims more: no unit is truncated at all in either fit.
    for fit in (generated, supplied):
        assert fit.nuisance.propensity.truncate(fit.config.g_bounds).fraction == 0.0
    assert supplied.data.backend == backend


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_clustered_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=11)
    frame = frame.copy()
    frame["pid"] = np.repeat(np.arange(60), 3)
    generated, supplied = _generated_then_supplied(_backed(frame, backend), cluster="pid")

    _assert_same_fit(generated, supplied)
    cluster = supplied.data.cluster
    assert cluster is not None
    for assignment in supplied.split_plan.assignments:
        labels = np.asarray(assignment)
        assert all(np.unique(labels[cluster == code]).size == 1 for code in np.unique(cluster))


def test_generated_and_supplied_repeated_plans_are_exactly_identical() -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=19)
    generated, supplied = _generated_then_supplied(frame, repeats=3)

    _assert_same_fit(generated, supplied)
    assert supplied.split_plan.n_repeats == supplied.n_repeats == 3
    # The scheme separates the two fits: one generated its own draws, the other reused
    # them. stratify_by carries what the folds were held to, which is nothing under this
    # policy, and the supplied fit records the same answer the generated fit did.
    assert generated.config.crossfit.scheme == "vfold"
    assert supplied.config.crossfit.scheme == "supplied"
    assert supplied.config.crossfit.stratify_by == generated.config.crossfit.stratify_by == ()


def test_generated_and_supplied_fold_diagnostics_are_exactly_identical() -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=21)
    generated, supplied = _generated_then_supplied(frame, fold_evaluation=True)

    _assert_same_fit(generated, supplied)
    expected = generated.cv_targeting
    actual = supplied.cv_targeting
    assert expected is not None
    assert actual is not None
    assert actual.n_folds == expected.n_folds
    assert actual.fold_sizes == expected.fold_sizes
    assert actual.fold_estimates == expected.fold_estimates
    assert actual.epsilon == expected.epsilon
    assert actual.fold_epsilon == expected.fold_epsilon
    assert actual.variance == expected.variance
    assert actual.repeats == expected.repeats
    assert actual.backend == expected.backend
    for report in ("pooled", "canonical"):
        expected_estimates = getattr(expected, report)
        actual_estimates = getattr(actual, report)
        assert actual_estimates.keys() == expected_estimates.keys()
        for name, estimate in expected_estimates.items():
            compared = actual_estimates[name]
            assert compared.psi == estimate.psi
            assert compared.variance == estimate.variance
            np.testing.assert_array_equal(compared.influence_curve, estimate.influence_curve)


def test_collaborative_selection_reuses_a_supplied_plan_exactly() -> None:
    """The collaborative selector adds folds of its own, and reuses the outer ones."""
    frame, _ = make_nonlinear_bounded(n=180, seed=43)
    generated, supplied = _generated_then_supplied(
        frame,
        kind=CollaborativeTMLEMethod,
        strategy="greedy",
        selection_folds=3,
        selection_inner_folds=2,
    )

    _assert_same_fit(generated, supplied)
    assert supplied.ctmle_selection is not None
    assert supplied.ctmle_selection.selected == generated.ctmle_selection.selected
    np.testing.assert_array_equal(
        supplied.ctmle_selection.folds.assignment,
        generated.ctmle_selection.folds.assignment,
    )


def test_reduced_dimension_correction_reuses_a_supplied_plan_exactly() -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=45)
    generated, supplied = _generated_then_supplied(
        frame,
        kind=DRTMLEMethod,
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LinearRegression(),
    )

    _assert_same_fit(generated, supplied)
    assert supplied.nuisance.reduced is not None


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_a_continuous_dose_reuses_a_supplied_plan_exactly(backend: str) -> None:
    """A dose has no arms to balance, so the plan is the only thing holding the split."""
    # A binary outcome: this identity compares the generated and supplied fits to each
    # other, never to a population truth, so replacing Y costs nothing. A cross-fitted
    # continuous outcome now needs a declared q_bounds that make_shift_dose's Gaussian Y
    # does not have.
    frame, _ = make_shift_dose(n=180, seed=47, backend="pandas")
    rng = np.random.default_rng(47)
    w1 = np.asarray(frame["W1"])
    frame = frame.assign(Y=rng.binomial(1, 1.0 / (1.0 + np.exp(-w1))).astype(float))
    frame = _backed(frame, backend)
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    ).identify(
        ModifiedTreatmentPolicyEffect(
            shifts=(
                Shift(0.0, cap=10.0, name="natural course"),
                Shift(0.5, cap=10.0, name="up half"),
            )
        )
    )
    generated = effect.estimate(method=_method(q_bounds=None))
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan, q_bounds=None))

    assert generated.estimates.keys() == supplied.estimates.keys()
    for name, expected in generated.estimates.items():
        assert expected.psi == supplied.estimates[name].psi
        np.testing.assert_array_equal(
            expected.influence_curve, supplied.estimates[name].influence_curve
        )
    assert generated.split_plan == supplied.split_plan
    assert supplied.config.crossfit.scheme == "supplied"
    # Nothing to balance and nothing checked: a dose declares no strata either way.
    assert supplied.config.crossfit.stratify_by == generated.config.crossfit.stratify_by == ()


class TestASuppliedPlanFreezesTheOuterSplit:
    """The witness for what a supplied plan does *not* fix.

    Every identity test above runs both fits at one seed, so an implementation that
    seeded the learner folds and the collaborative selection folds off the plan rather
    than off ``random_state`` would pass all of them.  The component that has to be there
    vanishes at the truth.  These two tests hold the plan fixed and move the seed, which
    is the only arrangement in which that component is observable.
    """

    @pytest.fixture(scope="class")
    def effect(self) -> Any:
        frame, _ = make_nonlinear_bounded(n=180, seed=4)
        return _effect(frame)

    def test_the_learner_folds_still_follow_the_seed(self, effect: Any) -> None:
        generated = effect.estimate(
            method=_method(split_plan=_package_plan(180), models=_ensemble_models())
        )
        plan = generated.split_plan
        eleven = effect.estimate(
            method=_method(split_plan=plan, models=_ensemble_models(), random_state=11)
        )
        seventy_seven = effect.estimate(
            method=_method(split_plan=plan, models=_ensemble_models(), random_state=77)
        )

        # The outer split is frozen, down to the digest a later fit compares against.
        assert eleven.split_plan == seventy_seven.split_plan == plan
        assert eleven.split_plan.fingerprint == seventy_seven.split_plan.fingerprint
        assert eleven.provenance.fold_fingerprint == seventy_seven.provenance.fold_fingerprint
        for left, right in zip(eleven.repeats, seventy_seven.repeats, strict=True):
            np.testing.assert_array_equal(left.folds.assignment, right.folds.assignment)

        # Two candidates were really scored, so the inner split had something to move.
        # With one candidate the ensemble weight is 1.0 whatever the inner folds are, and
        # every assertion below would hold against a plan that froze them.
        scored = [draw.names for draw in eleven.nuisance.diagnostics["outcome"]]
        assert scored
        assert all(names == ENSEMBLE_NAMES for names in scored)

        # And it moved: different inner folds, different ensemble, different estimate.
        weights = [
            (left.weights, right.weights)
            for left, right in zip(
                eleven.nuisance.diagnostics["outcome"],
                seventy_seven.nuisance.diagnostics["outcome"],
                strict=True,
            )
        ]
        assert any(not np.array_equal(left, right) for left, right in weights)
        assert not np.array_equal(
            eleven.nuisance.outcome.observed, seventy_seven.nuisance.outcome.observed
        )
        assert eleven.estimates["ate"].psi != seventy_seven.estimates["ate"].psi

        # And the seed is what moved it. Every candidate here is seeded, so a repeat of
        # the first fit reproduces it bit for bit, and the difference above cannot be read
        # as ambient randomness that the plan was never going to control.
        again = effect.estimate(
            method=_method(split_plan=plan, models=_ensemble_models(), random_state=11)
        )
        assert again.estimates["ate"].psi == eleven.estimates["ate"].psi
        np.testing.assert_array_equal(
            again.nuisance.outcome.observed, eleven.nuisance.outcome.observed
        )

    def test_the_collaborative_selection_folds_still_follow_the_seed(self, effect: Any) -> None:
        knobs = {
            "kind": CollaborativeTMLEMethod,
            "strategy": "greedy",
            "selection_folds": 3,
            "selection_inner_folds": 2,
            # An ensemble outcome learner: on this bounded law the greedy search always
            # selects zero covariates regardless of seed, so a single deterministic
            # learner would make every candidate's fit -- and so the final psi -- tie
            # across seeds too, leaving no witness. A Super Learner's own inner split
            # is still seed-sensitive even at the empty candidate.
            "models": _ensemble_models(),
        }
        generated = effect.estimate(method=_method(split_plan=_package_plan(180), **knobs))
        plan = generated.split_plan
        eleven = effect.estimate(method=_method(split_plan=plan, random_state=11, **knobs))
        seventy_seven = effect.estimate(method=_method(split_plan=plan, random_state=77, **knobs))

        assert eleven.split_plan == seventy_seven.split_plan == plan
        assert eleven.provenance.fold_fingerprint == seventy_seven.provenance.fold_fingerprint
        for left, right in zip(eleven.repeats, seventy_seven.repeats, strict=True):
            np.testing.assert_array_equal(left.folds.assignment, right.folds.assignment)

        selection = eleven.ctmle_selection
        other = seventy_seven.ctmle_selection
        assert selection is not None
        assert other is not None
        # The selection folds are a separate draw, and the plan does not fix them.
        assert not np.array_equal(selection.folds.assignment, other.folds.assignment)
        assert not np.array_equal(selection.cv_risk, other.cv_risk)
        assert eleven.estimates["ate"].psi != seventy_seven.estimates["ate"].psi


class TestAPlanIsBoundToTheRowsThatRealisedIt:
    """A fold label is a position, so the row count alone cannot police reuse."""

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        built, _ = make_nonlinear_bounded(n=180, seed=49)
        return built

    @pytest.fixture(scope="class")
    def fitted(self, frame: pd.DataFrame) -> Any:
        return _effect(frame).estimate(method=_method(split_plan=_package_plan(len(frame))))

    def test_the_result_plan_carries_the_fingerprint_provenance_recorded(self, fitted: Any) -> None:
        assert fitted.split_plan.source_fingerprint == fitted.provenance.data_fingerprint
        assert fitted.split_plan.source_fingerprint == data_fingerprint(fitted.data)

    def test_reordered_rows_of_the_same_size_are_refused(
        self, frame: pd.DataFrame, fitted: Any
    ) -> None:
        order = np.random.default_rng(0).permutation(len(frame))
        shuffled = frame.iloc[order].reset_index(drop=True)

        with pytest.raises(DataError, match=r"labels rows by position|realised on data"):
            _effect(shuffled).estimate(method=_method(split_plan=fitted.split_plan))

    def test_an_unbound_plan_reuses_the_labels_wherever_the_caller_says(
        self, frame: pd.DataFrame, fitted: Any
    ) -> None:
        """The declared way to mean "reuse these labels on other rows".

        The record travels with the labels, so the second fit draws the same split again
        and finds the same assignment: a row-level draw reads the row count and the seed,
        and neither one moved.
        """
        order = np.random.default_rng(0).permutation(len(frame))
        shuffled = frame.iloc[order].reset_index(drop=True)
        unbound = fitted.split_plan.unbound()

        assert unbound.source_fingerprint is None
        assert unbound.provenance == fitted.split_plan.provenance
        assert unbound.assignments == fitted.split_plan.assignments
        result = _effect(shuffled).estimate(method=_method(split_plan=unbound))

        assert result.split_plan.assignments == fitted.split_plan.assignments
        # Same labels, different rows behind them, so a different fit.
        assert result.estimates["ate"].psi != fitted.estimates["ate"].psi

    def test_the_same_rows_are_accepted(self, frame: pd.DataFrame, fitted: Any) -> None:
        again = _effect(frame).estimate(method=_method(split_plan=fitted.split_plan))

        _assert_same_fit(fitted, again)


class TestTheFoldCountIsCheckedAgainstTheDeclarationOnly:
    """Declaration time rules out one direction, and validation decides the rest.

    ``resolve_n_folds`` caps the declared count, and a cap only ever reduces it.  So a
    declaration can refuse a plan that holds *more* folds than it asks for, and nothing
    else.  What a fit needs of a supplied plan is not a count: it needs labels these rows
    can carry, which is what ``SplitPlan.validate`` checks.
    """

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        built, _ = make_nonlinear_bounded(n=180, seed=51)
        built = built.copy()
        built["four"] = np.repeat(np.arange(4), 45)
        built["two"] = np.repeat(np.arange(2), 90)
        return built

    def test_a_capped_plan_round_trips_under_the_declaration_that_capped_it(
        self, frame: pd.DataFrame
    ) -> None:
        with pytest.warns(UserWarning, match="only 4 clusters"):
            drawn = _package_plan(
                len(frame), n_folds=10, cluster=_cluster_codes(frame, cluster="four")
            )
        generated = _effect(frame, cluster="four").estimate(
            method=_method(n_folds=10, split_plan=drawn)
        )
        assert generated.split_plan.n_folds == 4
        # The declaration is recorded as declared, not as resolved.
        assert generated.config.crossfit.n_folds == 10

        # No cap warning here, and none is wanted: the supplied path generates no split,
        # so there is no count to reduce.  ``test_nothing_is_reduced_on_the_supplied_path``
        # is the assertion of that.
        supplied = _effect(frame, cluster="four").estimate(
            method=_method(n_folds=10, split_plan=generated.split_plan)
        )

        _assert_same_fit(generated, supplied)
        assert supplied.config.crossfit.n_folds == 10

    def test_a_plan_these_rows_cannot_carry_is_refused_by_validation(
        self, frame: pd.DataFrame
    ) -> None:
        """The plan and the declaration agree, and the labels still cannot serve.

        Clustering is not part of the data fingerprint, so declaring ``id=`` on the same
        columns leaves the binding satisfied and leaves the labels unchanged.  The labels
        were drawn without the clusters, so they cut across them, and cluster integrity is
        the prohibition cross-fitting exists to enforce.  The refusal names the scheme the
        record holds rather than a fold count, which is what makes it readable.
        """
        unclustered = _effect(frame).estimate(
            method=_method(n_folds=N_FOLDS, split_plan=_package_plan(len(frame)))
        )
        assert unclustered.split_plan.n_folds == N_FOLDS

        with pytest.raises(DataError, match="records a 'vfold' draw"):
            _effect(frame, cluster="two").estimate(
                method=_method(n_folds=N_FOLDS, split_plan=unclustered.split_plan)
            )

    def test_nothing_is_reduced_on_the_supplied_path(self, frame: pd.DataFrame) -> None:
        """A supplied plan generates no split, so there is no count to cap.

        The draw that made this plan warns that four clusters cap ten folds at four.
        Handing the realised plan to a fit must not repeat that warning: it describes a
        resolution step the supplied path does not run, and the fit draws the split again
        only to compare it.
        """
        with pytest.warns(UserWarning, match="only 4 clusters"):
            drawn = _package_plan(
                len(frame), n_folds=10, cluster=_cluster_codes(frame, cluster="four")
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _effect(frame, cluster="four").estimate(method=_method(n_folds=10, split_plan=drawn))

        assert not [one for one in caught if "reducing n_folds" in str(one.message)]


class TestARefitThatMovesTheStratumVectorKeepsTheSuppliedPlan:
    """Retired: ``stratify_by="treatment+outcome"`` no longer reaches a supplied plan.

    This class used to fit a ``stratify_by="treatment+outcome"`` supplied plan, whose
    balancing code crosses the treatment with the outcome, on a sample engineered so a
    placebo permutation's re-count left the rarest crossed cell with two members -- one
    fewer than a *generated* three-fold split would need, while the plan's own labels
    still carried every cell into every training complement.  The package refuses
    ``stratify_by="treatment+outcome"`` under cross-fitting outright now (both under
    ``CrossFitting`` and from the engine), so that scenario cannot be built any more:
    the refusal fires at declaration, before any plan is drawn and before the permutation
    the old tests read.  The permutation is kept below because it is what made the
    scenario the refusal now forecloses; the other two tests assert the refusal itself,
    from the two contracts this module's other refusals are checked under.
    """

    #: The permutation the retired scenario read, and the one the cell count describes.
    PLACEBO_SEED = 24

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        """150 rows whose crossed cells hold 72, 3, 72 and 3."""
        rng = np.random.default_rng(7)
        outcome = np.zeros(150)
        outcome[72:75] = 1.0
        outcome[147:150] = 1.0
        return pd.DataFrame(
            {
                "Y": outcome,
                "A": np.repeat([0.0, 1.0], 75),
                "W1": rng.normal(size=150),
                "W2": rng.normal(size=150),
                "W3": rng.normal(size=150),
            }
        )

    def test_the_permutation_leaves_a_cell_below_the_supplied_fold_count(
        self, frame: pd.DataFrame
    ) -> None:
        """The crossed-cell scenario the retired capability used to be exercised on.

        Kept as a fact about this fixture rather than about any fit: it is what made the
        retired scenario a genuine edge case rather than an ordinary one.
        """
        permuted = np.random.default_rng(self.PLACEBO_SEED).permutation(np.asarray(frame["A"]))
        crossed = np.unique(
            np.column_stack([permuted, np.asarray(frame["Y"])]), axis=0, return_inverse=True
        )[1]

        assert int(np.bincount(crossed).min()) == 2

    def test_the_declaration_is_refused_before_any_plan_is_drawn(self, frame: pd.DataFrame) -> None:
        """The configuration contract: ``MethodConfigurationError`` from ``CrossFitting``."""
        with pytest.raises(MethodConfigurationError, match=r"stratify_folds='treatment\+outcome'"):
            _generated_then_supplied(frame, stratify_by="treatment+outcome", plan_seed=1)

    def test_the_engine_gives_the_same_refusal(self) -> None:
        """The engine contract: ``ValueError`` at the same declaration, with no data at all.

        No frame is needed, which is itself evidence: the refusal is a fact about the
        declared policy, not about anything the retired scenario's sample supplied.
        """
        with pytest.raises(ValueError, match=r"stratify_folds='treatment\+outcome'"):
            TMLEEngine(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                n_folds=N_FOLDS,
                stratify_folds="treatment+outcome",
                random_state=SEED,
                simultaneous=False,
            )


class TestARefutationThatChangesTheRowSetIsRefusedUpFront:
    """Two tests refit on rows the fit never ran, and they fail differently without this.

    ``subset`` keeps a share of the rows, so the count changes and
    :meth:`SplitPlan.validate` raises a row-count :class:`~cleverly.exceptions.DataError`
    deep in a refit the caller has already paid for, naming a plan this call never
    mentioned.  ``bootstrap_measurement_error`` draws n rows with replacement, so the
    count is unchanged and nothing raises at all: the labels land on resampled units and
    the report describes a split the fit never ran.  The silent one is the worse of the
    two, and one guard covers both.
    """

    @pytest.fixture(scope="class")
    def supplied(self) -> Any:
        frame, _ = make_nonlinear_bounded(n=180, seed=53)
        _, result = _generated_then_supplied(frame)
        return result

    @pytest.mark.parametrize("test_name", ["subset", "bootstrap_measurement_error"])
    def test_a_row_set_changing_test_is_refused_before_any_refit(
        self, supplied: Any, monkeypatch: pytest.MonkeyPatch, test_name: str
    ) -> None:
        monkeypatch.setattr(
            TMLEEngine, "refit", _never_called("a refit ran before the composition was refused")
        )
        with pytest.raises(CapabilityError) as raised:
            refute(supplied, tests=("placebo", test_name), n_replicates=1)

        message = str(raised.value)
        assert f"'{test_name}'" in message
        assert "split_plan=SplitPlan(" in message

    def test_the_refusal_precedes_the_declaration_check_it_would_otherwise_hit(
        self, supplied: Any
    ) -> None:
        """The negative control for "up front".

        ``bootstrap_measurement_error`` needs a registered declaration, and that check
        also runs before any refit.  Reaching the plan refusal first is what shows the
        guard is the earliest one, rather than merely early.
        """
        with pytest.raises(CapabilityError, match="split_plan=SplitPlan"):
            refute(supplied, tests=("bootstrap_measurement_error",), n_replicates=1)

    def test_the_column_replacing_tests_still_run_under_a_bound_plan(self, supplied: Any) -> None:
        """Every other refutation keeps each row in its position, so the plan applies.

        ``TMLE.refit`` hands the plan on unbound for exactly this reason: the binding
        exists to catch a plan reused on other rows, and a replaced or perturbed column
        is the same rows.
        """
        report = refute(
            supplied,
            tests=("placebo", "random_common_cause"),
            n_replicates=1,
        )

        assert [test.name for test in report.tests] == ["placebo", "random_common_cause"]
        assert report.passed

    def test_the_battery_reports_the_refusal_and_keeps_running(self, supplied: Any) -> None:
        battery = supplied.assess(include_refits=True)
        rows = {item.name: item for item in battery.diagnostics.items}

        assert rows["refute"].status is AssessmentStatus.UNAVAILABLE
        assert "subset" in rows["refute"].detail
        assert rows["score_equations"].status is AssessmentStatus.PASSED
        assert rows["support"].status is AssessmentStatus.COMPLETED

    def test_an_unsupplied_fit_still_refutes_on_a_subsample(self) -> None:
        """The negative control: the refusal is about the plan, not about the test."""
        frame, _ = make_nonlinear_bounded(n=180, seed=53)
        generated = _effect(frame).estimate(method=_method())

        report = refute(generated, tests=("subset",), n_replicates=1)

        assert [test.name for test in report.tests] == ["subset"]

    def test_an_unsupplied_fit_reaches_the_declaration_check_instead(self) -> None:
        """The other half of that control: no plan, no plan refusal."""
        frame, _ = make_nonlinear_bounded(n=180, seed=53)
        generated = _effect(frame).estimate(method=_method())

        with pytest.raises(CapabilityError, match="requires the exact registered"):
            refute(generated, tests=("bootstrap_measurement_error",), n_replicates=1)


def test_a_refit_keeps_the_generator_record_and_drops_the_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refutation refits on the same rows with one column replaced.

    The binding goes, because the replaced column changes the data fingerprint. The
    record stays, because the refit still has to show that the labels it reuses are the
    labels the recorded generator draws.
    """
    frame, _ = make_nonlinear_bounded(n=120, seed=55)
    plan = _package_plan(120)
    result = _effect(frame).estimate(method=_method(split_plan=plan))
    seen: list[Any] = []
    original = TMLEEngine._fit_single

    def record(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(self.split_plan)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TMLEEngine, "_fit_single", record)
    report = refute(result, tests=("placebo",), n_replicates=1)

    assert [test.name for test in report.tests] == ["placebo"]
    assert seen
    for refitted in seen:
        assert refitted.source_fingerprint is None
        assert refitted.provenance == plan.provenance
        assert refitted.assignments == plan.assignments


def test_the_realized_plan_and_fingerprint_survive_result_persistence() -> None:
    frame, _ = make_nonlinear_bounded(n=120, seed=23)
    generated, supplied = _generated_then_supplied(frame, repeats=2)
    restored = loads(dumps(supplied))

    assert restored.split_plan == supplied.split_plan == generated.split_plan
    assert restored.split_plan.provenance == supplied.split_plan.provenance
    assert restored.method.cross_fitting.split_plan == supplied.split_plan
    assert restored.split_plan.fingerprint == restored.provenance.fold_fingerprint
    for expected, actual in zip(supplied.repeats, restored.repeats, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)


def test_an_in_sample_result_exposes_the_realized_one_fold_plan() -> None:
    frame, _ = make_nonlinear_bounded(n=120, seed=27)
    result = _effect(frame).estimate(method=_method(enabled=False))

    assert result.split_plan.n_folds == 1
    assert result.split_plan.n_repeats == 1
    assert result.split_plan.assignments == ((0,) * len(frame),)
    assert result.split_plan.fingerprint == result.provenance.fold_fingerprint


def test_a_supplied_plan_is_exact_under_serial_and_parallel_scheduling() -> None:
    frame, _ = make_nonlinear_bounded(n=180, seed=29)
    effect = _effect(frame)
    generated = effect.estimate(method=_method(split_plan=_package_plan(180)))
    serial = effect.estimate(method=_method(split_plan=generated.split_plan, n_jobs=1))
    parallel = effect.estimate(method=_method(split_plan=generated.split_plan, n_jobs=2))

    # Against the fit that drew the split, not against the other supplied fit: two
    # supplied fits agreeing says nothing about whether either reproduced the original.
    _assert_same_fit(generated, serial)
    _assert_same_fit(generated, parallel)


def test_a_split_plan_is_refused_for_a_longitudinal_fit_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_longitudinal(n=120, seed=31)
    effect = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    ).identify(RegimeMean({"always": 1}))
    plan = _package_plan(len(frame))

    monkeypatch.setattr(
        "cleverly.study.LTMLE",
        _never_called("the longitudinal engine was constructed before refusing split_plan"),
    )
    with pytest.raises(MethodConfigurationError, match="split_plan"):
        effect.estimate(method=_method(split_plan=plan))


def test_a_split_plan_is_refused_for_bootstrap_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_nonlinear_bounded(n=120, seed=37)
    plan = _package_plan(len(frame))

    monkeypatch.setattr(
        "cleverly.study.TMLE",
        _never_called("the point engine was constructed before refusing split_plan with bootstrap"),
    )
    with pytest.raises(MethodConfigurationError, match=r"split_plan|bootstrap"):
        _effect(frame).estimate(method=_method(split_plan=plan, n_bootstrap=4))


def test_the_bootstrap_refusal_belongs_to_the_declaration_that_holds_both_groups() -> None:
    """``TMLEMethod.__post_init__`` sees the cross-fitting and inference groups at once.

    The combination is therefore refused where it is declared, rather than in the middle
    of translating the declaration for an engine.  ``DRTMLEMethod.__post_init__`` is the
    precedent, and the test above shows the refusal still reaches a study fit.
    """
    plan = _package_plan(120, n_folds=3)

    with pytest.raises(MethodConfigurationError, match="bootstrap"):
        _method(split_plan=plan, n_bootstrap=4)
