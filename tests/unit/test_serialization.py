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
from cleverly.datasets import GENERATORS
from cleverly.estimators.recipe import TMLERecipe
from cleverly.estimators.serialize import FORMAT_VERSION, dumps, loads, result_to_dict

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
        np.testing.assert_array_equal(reloaded.nuisance.propensity, result.nuisance.propensity)
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
