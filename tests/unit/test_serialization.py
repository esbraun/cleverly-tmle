"""Saving a result, loading it back, and where the boundary actually is.

The claim these tests defend is narrow and checkable: a reloaded result is not an
approximation of the original.  Every estimate, influence curve, fluctuation
coefficient and targeted prediction comes back bit-for-bit, and every analysis that
goes through ``retarget`` produces the identical number afterwards -- which is the
real test, because it exercises the arrays the round trip had to preserve rather
than the ones it happened to.

The two analyses that genuinely refit are the documented exception, and their error
is asserted rather than described.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import TMLE, load
from cleverly.datasets import GENERATORS, make_shift_dose
from cleverly.estimators.recipe import TMLERecipe
from cleverly.estimators.serialize import FORMAT_VERSION, dumps, loads, result_to_dict
from cleverly.interventions import Shift

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _fit(**kwargs):  # type: ignore[no-untyped-def]
    frame, _ = GENERATORS[kwargs.pop("generator", "binary_outcome")](n=300, seed=3)
    covariates = [c for c in frame.columns if c.startswith("W")]
    settings = {
        "outcome_learner": "glm",
        "treatment_learner": "glm",
        "n_folds": 4,
        "random_state": 7,
    }
    settings.update(kwargs)
    return TMLE(**settings).fit(frame, outcome="Y", treatment="A", covariates=covariates).single()


@pytest.fixture(scope="module")
def result():
    return _fit(estimands="all")


@pytest.fixture(scope="module")
def reloaded(result, tmp_path_factory):  # type: ignore[no-untyped-def]
    path = tmp_path_factory.mktemp("fits") / "fit.npz"
    result.save(path)
    return load(path)


class TestRoundTripIsExact:
    def test_every_estimate_returns_bit_for_bit(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        assert set(reloaded.estimates) == set(result.estimates)
        for name in result.estimates:
            before, after = result[name], reloaded[name]
            assert after.psi == before.psi
            assert after.variance == before.variance
            assert after.scale == before.scale
            assert after.log_psi == before.log_psi
            assert after.n == before.n and after.n_clusters == before.n_clusters
            # assert_array_equal, not allclose: a round trip that loses a bit is a bug,
            # not a tolerance question.
            np.testing.assert_array_equal(after.influence_curve, before.influence_curve)

    def test_targeting_detail_returns_bit_for_bit(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        assert set(reloaded.fluctuations) == set(result.fluctuations)
        for group, before in result.fluctuations.items():
            after = reloaded.fluctuations[group]
            np.testing.assert_array_equal(after.epsilon, before.epsilon)
            np.testing.assert_array_equal(after.score, before.score)
            np.testing.assert_array_equal(after.targeted.observed, before.targeted.observed)
            assert after.targeted.arms.keys() == before.targeted.arms.keys()
            for level, values in before.targeted.arms.items():
                np.testing.assert_array_equal(after.targeted.arms[level], values)
            assert after.converged == before.converged
            assert after.names == before.names

    def test_nuisances_and_folds_return_bit_for_bit(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        np.testing.assert_array_equal(
            reloaded.nuisance.propensity.values, result.nuisance.propensity.values
        )
        assert reloaded.nuisance.propensity.arms == result.nuisance.propensity.arms
        np.testing.assert_array_equal(
            reloaded.nuisance.folds.assignment, result.nuisance.folds.assignment
        )
        assert reloaded.nuisance.folds.n_folds == result.nuisance.folds.n_folds
        assert reloaded.nuisance.scaler == result.nuisance.scaler

    def test_config_and_data_survive(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        assert reloaded.config == result.config
        np.testing.assert_array_equal(reloaded.data.outcome, result.data.outcome)
        np.testing.assert_array_equal(reloaded.data.covariates, result.data.covariates)
        assert reloaded.data.covariate_names == result.data.covariate_names
        assert reloaded.data.family == result.data.family
        # Not inferable from the levels: a continuous treatment has none, so a reader
        # that reconstructed the kind from an empty level list would guess "discrete"
        # for a dose and quietly change what the object means.
        assert reloaded.data.treatment_kind == result.data.treatment_kind
        assert reloaded.data.treatment_levels == result.data.treatment_levels

    def test_in_memory_round_trip_agrees_with_the_file(self, result) -> None:  # type: ignore[no-untyped-def]
        back = loads(dumps(result))
        np.testing.assert_array_equal(back["ate"].influence_curve, result["ate"].influence_curve)

    def test_fold_targeting_detail_survives(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = _fit(targeting_scheme="fold")
        path = tmp_path / "cv.npz"
        original.save(path)
        back = load(path)
        for group, before in original.fluctuations.items():
            after = back.fluctuations[group]
            assert len(after.folds) == len(before.folds) > 0
            for fa, fb in zip(after.folds, before.folds, strict=True):
                np.testing.assert_array_equal(fa.index, fb.index)
                np.testing.assert_array_equal(fa.epsilon, fb.epsilon)
                assert fa.converged == fb.converged


class TestRetargetSurvivesTheRoundTrip:
    """The real test: analyses that consume the arrays, not just the arrays."""

    def test_truncation_curve_is_identical(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        before = result.sensitivity.truncation_curve()
        after = reloaded.sensitivity.truncation_curve()
        np.testing.assert_array_equal(np.asarray(before["psi"]), np.asarray(after["psi"]))
        np.testing.assert_array_equal(np.asarray(before["std_err"]), np.asarray(after["std_err"]))

    def test_positivity_is_identical(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        before = result.sensitivity.positivity()
        after = reloaded.sensitivity.positivity()
        # This one used to return nan without a live estimator; it is the reason
        # build_submodel became a free function.
        assert after.clever_covariate_max == before.clever_covariate_max
        assert after.truncated == before.truncated
        assert after.verdict() == before.verdict()

    def test_score_check_is_identical(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        before = result.validation.score_check()
        after = reloaded.validation.score_check()
        assert bool(after) == bool(before)
        assert [row.score for row in after.rows] == [row.score for row in before.rows]

    def test_omitted_variable_is_identical(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        before = result.sensitivity.omitted_variable()
        after = reloaded.sensitivity.omitted_variable()
        assert after.robustness_value == pytest.approx(before.robustness_value, abs=0)

    def test_contrasts_are_identical(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        def difference(p):  # type: ignore[no-untyped-def]
            return p[0] - p[1]

        before = result.contrast(difference, ["ey1", "ey0"])
        after = reloaded.contrast(difference, ["ey1", "ey0"])
        np.testing.assert_array_equal(after.influence_curve, before.influence_curve)
        np.testing.assert_array_equal(reloaded.covariance(), result.covariance())


class TestAShiftFitRoundTrips:
    """A continuous fit carries two nuisances an arm-indexed one does not.

    Given its own class rather than folded into the fixtures above, because the module's
    ``result`` is a binary fit and every assertion there would need a branch.  What is
    checked is the same claim: the arrays come back, and the analyses that consume them
    produce the identical number afterwards.
    """

    @pytest.fixture(scope="class")
    def shift_pair(self, tmp_path_factory):  # type: ignore[no-untyped-def]
        frame, _ = make_shift_dose(n=400, seed=3)
        original = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=4,
                random_state=7,
                shifts=[Shift(0.0, cap=None), Shift(0.5, cap=5.0)],
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )
        path = tmp_path_factory.mktemp("shift") / "fit.npz"
        original.save(path)
        return original, load(path)

    def test_the_density_returns_bit_for_bit(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        before, after = shift_pair
        np.testing.assert_array_equal(
            after.nuisance.density.bin_probabilities, before.nuisance.density.bin_probabilities
        )
        np.testing.assert_array_equal(after.nuisance.density.edges, before.nuisance.density.edges)

    def test_the_shifts_return_bit_for_bit(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        before, after = shift_pair
        assert after.nuisance.shifts.names == before.nuisance.shifts.names
        assert after.nuisance.shifts.deltas == before.nuisance.shifts.deltas
        assert after.nuisance.shifts.reference == before.nuisance.shifts.reference
        # `design` is the array the fluctuation actually consumes, so it is the one worth
        # asserting: it is built from ratio and ratio_at together.
        np.testing.assert_array_equal(after.nuisance.shifts.design, before.nuisance.shifts.design)
        np.testing.assert_array_equal(after.nuisance.shifts.capped, before.nuisance.shifts.capped)

    def test_the_treatment_is_still_continuous(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        _, after = shift_pair
        assert after.data.treatment_kind == "continuous"
        assert after.data.is_continuous_treatment
        assert after.data.n_arms == 0
        assert after.config.parameter_axis == "shift"

    def test_the_score_check_is_identical(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        before, after = shift_pair
        assert [row.score for row in after.validation.score_check().rows] == [
            row.score for row in before.validation.score_check().rows
        ]

    def test_shift_support_is_identical(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        before, after = shift_pair
        for name, report in before.sensitivity.shift_support().items():
            assert after.sensitivity.shift_support()[name].max_ratio == report.max_ratio
            assert (
                after.sensitivity.shift_support()[name].effective_sample_size
                == report.effective_sample_size
            )

    def test_the_shifts_are_reconstructible(self, shift_pair) -> None:  # type: ignore[no-untyped-def]
        """Unlike a rule, a shift is data and so never makes a fit unrebuildable."""
        before, after = shift_pair
        recipe = TMLERecipe.from_estimator(before.estimator)
        assert recipe.learners_reconstructible
        assert [s["delta"] for s in recipe.shifts] == [0.0, 0.5]
        assert [s["cap"] for s in recipe.shifts] == [None, 5.0]
        assert after.estimator.shifts == before.estimator.shifts


class TestTheRefitBoundary:
    def test_library_specifications_rebuild_and_refit(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = _fit(generator="linear_ate")
        path = tmp_path / "spec.npz"
        original.save(path)
        back = load(path)
        # A genuine refit, which needs the estimator rebuilt from the recipe.
        refutation = back.validation.refute(n_replicates=2, tests=["placebo"])
        assert refutation.passed

    def test_a_fitted_estimator_cannot_be_rebuilt_and_says_so(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        frame, _ = GENERATORS["linear_ate"](n=300, seed=1)
        covariates = [c for c in frame.columns if c.startswith("W")]
        original = (
            TMLE(
                outcome_learner="glm",
                treatment_learner=LogisticRegression(),
                n_folds=4,
                random_state=7,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )
        path = tmp_path / "obj.npz"
        original.save(path)
        back = load(path)

        # retarget-based work is unaffected...
        assert back.sensitivity.positivity().clever_covariate_max
        assert bool(back.validation.score_check())

        # ...and the refit-based path explains itself rather than silently
        # substituting the default library.
        with pytest.raises(ValueError, match="cannot be rebuilt from the recipe"):
            back.validation.refute(n_replicates=1, tests=["placebo"])

    def test_recipe_flags_which_slot_was_the_problem(self) -> None:
        estimator = TMLE(outcome_learner="glm", treatment_learner=LogisticRegression())
        recipe = TMLERecipe.from_estimator(estimator)
        assert not recipe.learners_reconstructible
        assert recipe.unreconstructible_slots == ("treatment_learner",)

    def test_a_list_of_library_names_is_still_a_specification(self) -> None:
        estimator = TMLE(outcome_learner=["glm", "mean"], treatment_learner="glm")
        recipe = TMLERecipe.from_estimator(estimator)
        assert recipe.learners_reconstructible
        assert recipe.build().outcome_learner == ["glm", "mean"]


class TestFormat:
    def test_no_pickle_in_the_payload(self, result, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """np.load(allow_pickle=False) is the check that matters."""
        path = tmp_path / "nopickle.npz"
        result.save(path)
        with np.load(path, allow_pickle=False) as archive:
            assert "__manifest__" in archive.files

    def test_a_future_version_is_refused_rather_than_misread(self, result) -> None:  # type: ignore[no-untyped-def]
        manifest, store = result_to_dict(result)
        manifest["format_version"] = FORMAT_VERSION + 1
        from cleverly.estimators.serialize import result_from_dict

        with pytest.raises(ValueError, match="format version"):
            result_from_dict(manifest, store)

    def test_dropped_pieces_are_named_not_hidden(self, result) -> None:  # type: ignore[no-untyped-def]
        manifest, _ = result_to_dict(result)
        # simultaneous bands are computed by default and are not persisted; the
        # manifest must say so rather than leaving the reader to discover it.
        assert "simultaneous" in manifest["dropped"]

    def test_the_declared_fold_policy_survives_the_round_trip(self, result) -> None:  # type: ignore[no-untyped-def]
        # The plan is what says a fit *asked* for 10 folds; losing it would leave only
        # the count it got, which is the question the field exists to answer.
        assert loads(dumps(result)).config.crossfit == result.config.crossfit


class TestTheReducedRegressionsSurvive:
    """Format version 9's arrays, grafted on rather than fitted.

    No estimator produces a :class:`~cleverly.estimators.reduced.ReducedSet` yet -- the
    extra score equations are the commits after this one -- so the round trip is checked
    against a set attached by hand.  It is checked *now* rather than then because
    ``_nuisance_from`` names every field it reconstructs: one left unwritten reloads
    silently as ``None``, and the version comment says what reporting a plain TMLE's
    interval under a doubly-robust name would look like.
    """

    @staticmethod
    def _with_reduced(result):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        from cleverly.estimators.reduced import ReducedSet

        nuisance = result.nuisance
        n, k = nuisance.n, len(nuisance.arms)
        rng = np.random.default_rng(0)
        reduced = ReducedSet(
            qr=rng.normal(size=(n, k)),
            gr1=rng.uniform(0.2, 0.8, size=(n, k)),
            gr2=rng.normal(size=(n, k)),
            arms=nuisance.arms,
            g_bounds=(0.01, 0.99),
        )
        repeat = replace(result.repeats[0], nuisance=replace(nuisance, reduced=reduced))
        return replace(result, repeats=(repeat,)), reduced

    def test_every_array_returns_bit_for_bit(self, result) -> None:  # type: ignore[no-untyped-def]
        grafted, reduced = self._with_reduced(result)
        back = loads(dumps(grafted)).nuisance.reduced
        assert back is not None
        for name in ("qr", "gr1", "gr2"):
            np.testing.assert_array_equal(getattr(back, name), getattr(reduced, name))
        assert back.arms == reduced.arms
        assert back.g_bounds == reduced.g_bounds
        assert back.reduction == reduced.reduction

    def test_a_fit_without_them_still_reloads_as_none(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        assert result.nuisance.reduced is None
        assert reloaded.nuisance.reduced is None


class TestTheTargetingHalvesSurvive:
    """Format version 10: the records a targeting step with more than one equation hangs.

    Grafted rather than fitted, for the reason the class above grafts: a hand-built record
    can carry a value no fit here produces -- a failure, a capped exit, a non-zero
    ill-conditioning count -- so no default can masquerade as a round trip.  Each test
    walks ``dataclasses.fields`` rather than a list written out here, because the failure
    mode is a *field* left unwritten and a hand-written list is exactly as likely to
    forget it as the writer was.

    What makes this worth a test rather than an argument is in
    :data:`~cleverly.estimators.serialize.FORMAT_VERSION`'s note: ``score_check`` reads
    these three, so losing one narrows the check rather than the record.
    """

    @staticmethod
    def _reduced(nuisance):  # type: ignore[no-untyped-def]
        from cleverly.estimators.reduced import ReducedSet

        n, k = nuisance.n, len(nuisance.arms)
        rng = np.random.default_rng(1)
        return ReducedSet(
            qr=rng.normal(size=(n, k)),
            gr1=rng.uniform(0.2, 0.8, size=(n, k)),
            gr2=rng.normal(size=(n, k)),
            arms=nuisance.arms,
            g_bounds=(0.01, 0.99),
        )

    @staticmethod
    def _graft(result, **halves):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        fluctuations = dict(result.repeats[0].fluctuations)
        group = next(iter(fluctuations))
        fluctuations[group] = replace(fluctuations[group], **halves)
        repeat = replace(result.repeats[0], fluctuations=fluctuations)
        return replace(result, repeats=(repeat,)), group

    @staticmethod
    def _assert_fields_match(before, after) -> None:  # type: ignore[no-untyped-def]
        from dataclasses import fields

        assert after is not None
        for spec in fields(before):
            original, restored = getattr(before, spec.name), getattr(after, spec.name)
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(restored, original)
            elif spec.name in {"reduced", "folds"}:
                continue  # checked by their own assertions below
            else:
                assert restored == original, spec.name

    def test_the_mechanism_tilt_returns_every_field(self, result) -> None:  # type: ignore[no-untyped-def]
        from cleverly.fluctuation.mechanism import MechanismFluctuation

        rng = np.random.default_rng(2)
        mechanism = MechanismFluctuation(
            propensity=rng.uniform(0.1, 0.9, size=result.data.n),
            epsilon=rng.normal(size=2),
            score=rng.normal(size=2),
            score_scale=rng.uniform(1.0, 2.0, size=2),
            score_initial=rng.normal(size=2),
            converged=False,
            n_iter=7,
            epsilon_std_error=rng.uniform(size=2),
            hessian_condition=1.5e9,
            loglik=-123.25,
            failure="max_iter_reached",
            trace=((0, 1.0, 2.0, -3.0), (1, 0.5, 1.0, -2.0)),
        )
        grafted, group = self._graft(result, mechanism=mechanism)
        back = loads(dumps(grafted)).repeats[0].fluctuations[group].mechanism
        self._assert_fields_match(mechanism, back)

    def test_the_projection_returns_every_field(self, result) -> None:  # type: ignore[no-untyped-def]
        from cleverly.estimators.targeting import ProjectionFluctuation

        rng = np.random.default_rng(3)
        fold = ProjectionFluctuation(beta=rng.normal(size=3), trace=((0, 1e-3, 1e-4),))
        projection = ProjectionFluctuation(
            beta=rng.normal(size=3),
            trace=((0, 1e-2, 1e-1), (1, 1e-4, 1e-5)),
            converged=False,
            failure="max_iter_reached",
            folds=(fold,),
        )
        grafted, group = self._graft(result, projection=projection)
        back = loads(dumps(grafted)).repeats[0].fluctuations[group].projection
        self._assert_fields_match(projection, back)
        assert back is not None and len(back.folds) == 1
        self._assert_fields_match(fold, back.folds[0])

    def test_the_reduction_record_returns_every_field(self, result) -> None:  # type: ignore[no-untyped-def]
        from cleverly.estimators.targeting import ReductionFluctuation

        rng = np.random.default_rng(4)
        reduced = self._reduced(result.nuisance)
        reduction = ReductionFluctuation(
            reduced=reduced,
            guard=("Q", "g"),
            bounds=(0.02, 0.98),
            epsilon=rng.normal(size=2),
            score=rng.normal(size=2),
            score_scale=rng.uniform(1.0, 2.0, size=2),
            score_initial=rng.normal(size=2),
            names=("h_qr", "h_gr"),
            trace=((0, 1.0, 2.0, 3.0, -4.0),),
            rounds=13,
            converged=False,
            failure="max_iter_reached",
            exit_reason="cap",
            closing_capped=True,
            ill_conditioned=3,
            closing=2,
        )
        grafted, group = self._graft(result, reduction=reduction)
        back = loads(dumps(grafted)).repeats[0].fluctuations[group].reduction
        self._assert_fields_match(reduction, back)
        assert back is not None
        for name in ("qr", "gr1", "gr2"):
            np.testing.assert_array_equal(getattr(back.reduced, name), getattr(reduced, name))
        assert back.reduced.g_bounds == reduced.g_bounds

    def test_the_refit_reductions_are_not_the_nuisances_own(self, result) -> None:  # type: ignore[no-untyped-def]
        """The record carries the alternation's refit, which the nuisances never hold.

        Both slots take a ``ReducedSet`` through one writer, so this is what says the two
        are still distinct payloads rather than one written twice.
        """
        from cleverly.estimators.targeting import ReductionFluctuation

        reduced = self._reduced(result.nuisance)
        reduction = ReductionFluctuation(
            reduced=reduced,
            guard=("Q",),
            bounds=(0.01, 0.99),
            epsilon=np.zeros(1),
            score=np.zeros(1),
            score_scale=np.ones(1),
            score_initial=np.zeros(1),
        )
        grafted, group = self._graft(result, reduction=reduction)
        back = loads(dumps(grafted))
        assert back.nuisance.reduced is None
        assert back.repeats[0].fluctuations[group].reduction is not None

    def test_a_fit_with_none_of_them_still_reloads_as_none(self, result, reloaded) -> None:  # type: ignore[no-untyped-def]
        for group, fluctuation in reloaded.repeats[0].fluctuations.items():
            assert fluctuation.mechanism is None, group
            assert fluctuation.projection is None, group
            assert fluctuation.reduction is None, group


class TestProvenance:
    def test_identical_data_gives_an_identical_fingerprint(self) -> None:
        assert _fit().provenance.data_fingerprint == _fit().provenance.data_fingerprint

    def test_one_perturbed_value_changes_the_fingerprint(self) -> None:
        frame, _ = GENERATORS["linear_ate"](n=200, seed=1)
        covariates = [c for c in frame.columns if c.startswith("W")]
        settings = {
            "outcome_learner": "glm",
            "treatment_learner": "glm",
            "n_folds": 4,
            "random_state": 7,
        }
        first = (
            TMLE(**settings).fit(frame, outcome="Y", treatment="A", covariates=covariates).single()
        )
        moved = frame.copy()
        moved.loc[0, "W1"] = moved.loc[0, "W1"] + 1e-12
        second = (
            TMLE(**settings).fit(moved, outcome="Y", treatment="A", covariates=covariates).single()
        )
        assert first.provenance.data_fingerprint != second.provenance.data_fingerprint

    def test_the_fold_fingerprint_is_recorded_separately_from_the_seed(self) -> None:
        """Folds are not recoverable from a seed alone, so they are hashed."""
        result = _fit()
        assert result.provenance.random_state == 7
        assert result.provenance.fold_fingerprint
        assert result.provenance.fold_fingerprint != result.provenance.data_fingerprint

    def test_run_id_is_carried_through(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _fit(run_id="experiment-42")
        path = tmp_path / "run.npz"
        result.save(path)
        assert load(path).provenance.run_id == "experiment-42"

    def test_provenance_appears_in_the_summary(self) -> None:
        text = _fit(run_id="abc").summary()
        assert "abc" in text
