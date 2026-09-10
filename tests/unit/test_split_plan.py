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
"""

from __future__ import annotations

import dataclasses
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
    TMLEMethod,
)
from cleverly._typing import FoldStrata
from cleverly.datasets import (
    make_longitudinal,
    make_multi_arm,
    make_nonlinear_ate,
    make_shift_dose,
)
from cleverly.estimators.serialize import dumps, loads
from cleverly.estimators.tmle import TMLE as TMLEEngine
from cleverly.exceptions import DataError, MethodConfigurationError
from cleverly.interventions import Shift
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
    stratify_by: FoldStrata = "treatment",
    random_state: int = SEED,
    models: ModelSpec | None = None,
    kind: type[TMLEMethod] = TMLEMethod,
    **extra: Any,
) -> TMLEMethod:
    """Build one method declaration, with every knob these tests turn.

    ``kind`` and ``extra`` reach the collaborative and reduced-dimension configurations,
    which add fields to the same five groups rather than replacing them.
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
        inference=Inference(simultaneous=False, n_bootstrap=n_bootstrap),
        runtime=Runtime(random_state=random_state, n_jobs=n_jobs),
        **extra,
    )


def _effect(frame: Any, *, cluster: str | None = None) -> Any:
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            cluster=cluster,
        ),
    ).identify(ATE())


def _backed(frame: pd.DataFrame, backend: str) -> Any:
    """Return one built frame under the requested backend."""
    return frame if backend == "pandas" else pl.from_pandas(frame)


def _never_called(reason: str) -> Callable[..., Any]:
    """Return a stand-in that fails the test when anything calls it."""

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(reason)

    return unexpected


def _generated_then_supplied(
    frame: Any, *, cluster: str | None = None, **knobs: Any
) -> tuple[Any, Any]:
    """Fit once to draw a split, then again on the plan that first fit realised."""
    effect = _effect(frame, cluster=cluster)
    generated = effect.estimate(method=_method(**knobs))
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
            f"fingerprint={plan.fingerprint}, source=abc123)"
        )
        assert len(text) < 200

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


class TestCrossFittingConfiguration:
    def test_flat_shortcut_sets_the_plan_on_the_normalized_configuration(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])

        method = TMLEMethod().with_overrides(n_folds=2, repeats=1, split_plan=plan)

        assert method.cross_fitting.split_plan is plan

    def test_more_folds_than_declared_is_refused(self) -> None:
        plan = SplitPlan([[0, 1, 2, 0, 1, 2]])
        with pytest.raises(MethodConfigurationError, match="n_folds"):
            CrossFitting(n_folds=2, split_plan=plan)

    def test_fewer_folds_than_declared_is_accepted(self) -> None:
        """The direction a cap produces.

        ``resolve_n_folds`` caps the declared count at the rarest stratum and again at the
        cluster count, so a plan this package wrote under a 10-fold declaration can hold
        three.  Refusing it here would refuse the package's own record of its own fit.
        The equality the fit needs is against the resolved count, which
        ``TMLE._repeat_draws`` checks once the data are in hand.
        """
        plan = SplitPlan([[0, 1, 2, 0, 1, 2]])

        assert CrossFitting(n_folds=10, split_plan=plan).split_plan is plan

    def test_repeat_count_must_match_the_supplied_plan(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        with pytest.raises(MethodConfigurationError, match="repeats"):
            CrossFitting(n_folds=2, repeats=1, split_plan=plan)

    def test_a_plan_conflicts_with_disabled_cross_fitting(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])
        with pytest.raises(MethodConfigurationError, match=r"enabled|cross.?fit"):
            CrossFitting(enabled=False, n_folds=2, split_plan=plan)

    def test_a_one_fold_plan_cannot_be_supplied_as_cross_fitting(self) -> None:
        plan = SplitPlan([[0, 0, 0]])
        with pytest.raises(MethodConfigurationError, match=r"n_folds|cross.?fit"):
            CrossFitting(enabled=True, n_folds=1, split_plan=plan)

    def test_only_a_split_plan_is_accepted(self) -> None:
        with pytest.raises(MethodConfigurationError, match="split_plan"):
            CrossFitting(n_folds=2, split_plan=[[0, 1]])  # type: ignore[arg-type]


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
            ({"split_plan": SplitPlan([[0, 1, 0, 1]]), "cross_fit": False}, "cross-fitting"),
            ({"split_plan": SplitPlan([[0, 1, 0, 1]]), "n_folds": 1}, "cross-fitting"),
            (
                {"split_plan": SplitPlan([[0, 1, 2, 0, 1, 2]]), "n_folds": 2},
                "3 folds but n_folds is 2",
            ),
            ({"split_plan": SplitPlan([[0, 1, 0, 1]] * 2), "repeats": 1}, "2 repeats"),
            (
                {"split_plan": SplitPlan([[0, 1, 0, 1]]), "n_bootstrap": 4},
                "n_bootstrap cannot be combined",
            ),
        ],
        ids=(
            "not-a-plan",
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
        frame, _ = make_nonlinear_ate(n=120, seed=41)
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
                )
                .fit(frame, outcome="Y", treatment="A", covariates=covariates)
                .single()
            )

        generated = fit(None)
        supplied = fit(generated.split_plan)

        _assert_same_fit(generated, supplied)


class TestDataBoundValidationPrecedesNuisanceFitting:
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
        stratify_by: FoldStrata = "treatment",
        cluster: str | None = None,
    ) -> None:
        monkeypatch.setattr(
            TMLEEngine,
            "_nuisances",
            _never_called("nuisance fitting began before the supplied split was validated"),
        )
        method = _method(
            n_folds=plan.n_folds,
            repeats=plan.n_repeats,
            stratify_by=stratify_by,
            split_plan=plan,
        )
        with pytest.raises(DataError):
            _effect(frame, cluster=cluster).estimate(method=method)

    def test_row_count_is_checked_before_fit(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.arange(len(frame) - 1) % N_FOLDS
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]))

    def test_each_training_complement_keeps_every_treatment_arm(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.ones(len(frame), dtype=int)
        assignment[frame["A"].to_numpy() == 1] = 0
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]))

    def test_each_training_complement_keeps_every_requested_stratum(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.ones(len(frame), dtype=int)
        rare_cell = (frame["A"].to_numpy() == 1) & (frame["Y"].to_numpy() == 1)
        assignment[rare_cell] = 0
        # Keep both treatment arms in fold 0, so the treatment-support check alone passes.
        assignment[[0, 2]] = 0
        self._refuse_before_fit(
            monkeypatch,
            frame,
            SplitPlan([assignment]),
            stratify_by="treatment+outcome",
        )

    def test_a_declared_cluster_cannot_be_split(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.arange(len(frame)) % 2
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]), cluster="pid")


def test_generated_folds_do_not_acquire_supplied_plan_support_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=14)

    monkeypatch.setattr(
        SplitPlan,
        "validate",
        _never_called("a generated split entered the supplied-plan validator"),
    )
    result = _effect(frame).estimate(method=_method())

    assert result.split_plan.n_folds == N_FOLDS


def test_in_sample_gaussian_fit_ignores_crossfit_only_outcome_strata() -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=15)

    result = _effect(frame).estimate(method=_method(enabled=False, stratify_by="treatment+outcome"))

    assert result.split_plan.n_folds == 1


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_binary_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=4, backend=backend)
    generated, supplied = _generated_then_supplied(frame)

    _assert_same_fit(generated, supplied)
    assert supplied.data.backend == backend


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_multi_arm_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_multi_arm(n=180, seed=7, backend=backend)
    generated, supplied = _generated_then_supplied(frame)

    _assert_same_fit(generated, supplied)
    assert generated.nuisance.propensity.values.shape[1] == 3
    assert supplied.data.backend == backend


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_clustered_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=11)
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
    frame, _ = make_nonlinear_ate(n=180, seed=19)
    generated, supplied = _generated_then_supplied(frame, repeats=3)

    _assert_same_fit(generated, supplied)
    assert supplied.split_plan.n_repeats == supplied.n_repeats == 3
    assert supplied.config.crossfit.scheme == "supplied"
    # The scheme carries "nothing was generated". stratify_by carries what the folds were
    # held to, which the supplied path checks rather than balances, so it records the same
    # treatment name the generated fit balanced on.
    assert supplied.config.crossfit.stratify_by == generated.config.crossfit.stratify_by == ("A",)


def test_generated_and_supplied_fold_diagnostics_are_exactly_identical() -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=21)
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
    frame, _ = make_nonlinear_ate(n=180, seed=43)
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
    frame, _ = make_nonlinear_ate(n=180, seed=45)
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
    frame, _ = make_shift_dose(n=180, seed=47, backend=backend)
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
    generated = effect.estimate(method=_method())
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan))

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
        frame, _ = make_nonlinear_ate(n=180, seed=4)
        return _effect(frame)

    def test_the_learner_folds_still_follow_the_seed(self, effect: Any) -> None:
        generated = effect.estimate(method=_method(models=_ensemble_models()))
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
        }
        generated = effect.estimate(method=_method(**knobs))
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
        built, _ = make_nonlinear_ate(n=180, seed=49)
        return built

    @pytest.fixture(scope="class")
    def fitted(self, frame: pd.DataFrame) -> Any:
        return _effect(frame).estimate(method=_method())

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
        """The declared way to mean "reuse these labels on other rows"."""
        order = np.random.default_rng(0).permutation(len(frame))
        shuffled = frame.iloc[order].reset_index(drop=True)
        unbound = SplitPlan(fitted.split_plan.assignments)

        assert unbound.source_fingerprint is None
        assert unbound.assignments == fitted.split_plan.assignments
        result = _effect(shuffled).estimate(method=_method(split_plan=unbound))

        assert result.split_plan.assignments == fitted.split_plan.assignments
        # Same labels, different rows behind them, so a different fit.
        assert result.estimates["ate"].psi != fitted.estimates["ate"].psi

    def test_the_same_rows_are_accepted(self, frame: pd.DataFrame, fitted: Any) -> None:
        again = _effect(frame).estimate(method=_method(split_plan=fitted.split_plan))

        _assert_same_fit(fitted, again)


class TestTheFoldCountIsCheckedAgainstWhatTheDataResolve:
    """Declaration time rules out one direction; fit time checks the equality.

    ``resolve_n_folds`` caps the declared count, and a cap only ever reduces it.  So a
    declaration can refuse a plan that holds *more* folds than it asks for, and nothing
    else, while the count a fit actually runs is not knowable until the data are in hand.
    """

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        built, _ = make_nonlinear_ate(n=180, seed=51)
        built = built.copy()
        built["four"] = np.repeat(np.arange(4), 45)
        built["two"] = np.repeat(np.arange(2), 90)
        return built

    def test_a_capped_plan_round_trips_under_the_declaration_that_capped_it(
        self, frame: pd.DataFrame
    ) -> None:
        with pytest.warns(UserWarning, match="only 4 clusters"):
            generated = _effect(frame, cluster="four").estimate(method=_method(n_folds=10))
        assert generated.split_plan.n_folds == 4
        # The declaration is recorded as declared, not as resolved.
        assert generated.config.crossfit.n_folds == 10

        with pytest.warns(UserWarning, match="only 4 clusters"):
            supplied = _effect(frame, cluster="four").estimate(
                method=_method(n_folds=10, split_plan=generated.split_plan)
            )

        _assert_same_fit(generated, supplied)
        assert supplied.config.crossfit.n_folds == 10

    def test_a_plan_is_refused_when_these_data_resolve_a_different_count(
        self, frame: pd.DataFrame
    ) -> None:
        """The plan and the declaration agree, and the data do not.

        Clustering is not part of the data fingerprint, so declaring ``id=`` on the same
        columns leaves the binding satisfied and moves only the resolved fold count.
        """
        unclustered = _effect(frame).estimate(method=_method(n_folds=N_FOLDS))
        assert unclustered.split_plan.n_folds == N_FOLDS

        with (
            pytest.raises(DataError, match=r"holds 3 folds but these data resolve"),
            pytest.warns(UserWarning, match="only 2 clusters"),
        ):
            _effect(frame, cluster="two").estimate(
                method=_method(n_folds=N_FOLDS, split_plan=unclustered.split_plan)
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
        frame, _ = make_nonlinear_ate(n=180, seed=53)
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
        frame, _ = make_nonlinear_ate(n=180, seed=53)
        generated = _effect(frame).estimate(method=_method())

        report = refute(generated, tests=("subset",), n_replicates=1)

        assert [test.name for test in report.tests] == ["subset"]

    def test_an_unsupplied_fit_reaches_the_declaration_check_instead(self) -> None:
        """The other half of that control: no plan, no plan refusal."""
        frame, _ = make_nonlinear_ate(n=180, seed=53)
        generated = _effect(frame).estimate(method=_method())

        with pytest.raises(CapabilityError, match="requires the exact registered"):
            refute(generated, tests=("bootstrap_measurement_error",), n_replicates=1)


def test_the_realized_plan_and_fingerprint_survive_result_persistence() -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=23)
    generated, supplied = _generated_then_supplied(frame, repeats=2)
    restored = loads(dumps(supplied))

    assert restored.split_plan == supplied.split_plan == generated.split_plan
    assert restored.method.cross_fitting.split_plan == supplied.split_plan
    assert restored.split_plan.fingerprint == restored.provenance.fold_fingerprint
    for expected, actual in zip(supplied.repeats, restored.repeats, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)


def test_an_in_sample_result_exposes_the_realized_one_fold_plan() -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=27)
    result = _effect(frame).estimate(method=_method(enabled=False))

    assert result.split_plan.n_folds == 1
    assert result.split_plan.n_repeats == 1
    assert result.split_plan.assignments == ((0,) * len(frame),)
    assert result.split_plan.fingerprint == result.provenance.fold_fingerprint


def test_a_supplied_plan_is_exact_under_serial_and_parallel_scheduling() -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=29)
    effect = _effect(frame)
    generated = effect.estimate(method=_method())
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
    plan = SplitPlan([np.arange(len(frame)) % N_FOLDS])

    monkeypatch.setattr(
        "cleverly.study.LTMLE",
        _never_called("the longitudinal engine was constructed before refusing split_plan"),
    )
    with pytest.raises(MethodConfigurationError, match="split_plan"):
        effect.estimate(method=_method(split_plan=plan))


def test_a_split_plan_is_refused_for_bootstrap_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=37)
    assignment = np.arange(len(frame)) % N_FOLDS
    plan = SplitPlan([assignment])

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
    plan = SplitPlan([[0, 1, 2, 0, 1, 2]])

    with pytest.raises(MethodConfigurationError, match="bootstrap"):
        _method(split_plan=plan, n_bootstrap=4)
