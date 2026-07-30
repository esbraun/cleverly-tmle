"""Missing outcomes and controlled direct effects.

These are the two features that change the *estimand*, not just the estimator, so the
tests check that the right quantity is recovered rather than merely that the code runs.

For missing outcomes, note what is and is not being claimed.  Under missingness at
random given ``(A, W)``, a *correctly specified* outcome regression fit on complete cases
already identifies the estimand -- so the missingness model is not what rescues
identification.  What it buys is the other half of double robustness: with the outcome
model misspecified, the clever covariate's ``1 / P(Delta = 1 | A, W)`` factor is what
keeps the estimate consistent.  The tests below check exactly that asymmetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import cde_dgp, make_cde, make_missing_outcome, missing_outcome_dgp
from cleverly.estimators.base import TMLEResultSet
from tests.conftest import OracleTreatment, fast_tmle

COVARIATES = ["W1", "W2", "W3"]


class TestMissingOutcomes:
    @pytest.fixture(scope="class")
    def fit(self) -> tuple[object, dict[str, float]]:
        frame, truth = make_missing_outcome(n=2500, seed=61)
        result = (
            fast_tmle(estimands=("ate", "att", "ey1", "ey0"))
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
            .single()
        )
        return result, truth

    def test_the_estimand_is_recovered(self, fit) -> None:
        result, truth = fit
        for name in ("ate", "att", "ey1", "ey0"):
            low, high = result[name].ci
            assert low <= truth[name] <= high, f"{name}: {result.psi(name)} vs {truth[name]}"

    def test_the_missingness_model_is_fitted_and_diagnosed(self, fit) -> None:
        result, _ = fit
        assert result.nuisance.missingness is not None
        assert result.nuisance.missingness.shape == (result.data.n, 2)
        diagnostics = result.validation.nuisance()
        report = diagnostics["missingness"]
        # The mechanism depends on A and W in this process, so it must be predictable.
        assert report.metrics["auc"] > 0.55

    def test_the_score_equation_accounts_for_missingness(self, fit) -> None:
        result, _ = fit
        assert result.validation.score_check().passed

    def test_only_observed_rows_enter_the_outcome_regression(self, fit) -> None:
        result, _ = fit
        assert result.data.has_missing_outcome
        assert int(result.data.observed.sum()) < result.data.n

    def test_the_missingness_model_rescues_a_misspecified_outcome_model(self) -> None:
        """Double robustness through the missingness mechanism, not the outcome model."""
        dgp = missing_outcome_dgp()
        biases = {"with_delta": [], "ignoring_delta": []}
        for seed in range(12):
            frame, truth = dgp.sample(1500, seed=200 + seed)
            columns = {"outcome": "Y", "treatment": "A", "covariates": COVARIATES}
            # An intercept-only outcome model: badly misspecified on purpose, so only the
            # inverse-probability part of the clever covariate can carry the estimate.
            settings = {
                "outcome_learner": [("mean", _MeanOnly())],
                "treatment_learner": OracleTreatment(dgp),
                "n_folds": 4,
                "learner_folds": 3,
                "estimands": ("ate",),
                "simultaneous": False,
                "random_state": 0,
            }
            with_delta = TMLE(**settings).fit(frame, delta="Delta", **columns).single()
            complete_case = TMLE(**settings).fit(frame.dropna(subset=["Y"]), **columns).single()
            biases["with_delta"].append(with_delta.psi("ate") - truth["ate"])
            biases["ignoring_delta"].append(complete_case.psi("ate") - truth["ate"])

        modelled = abs(float(np.mean(biases["with_delta"])))
        ignored = abs(float(np.mean(biases["ignoring_delta"])))
        # Modelling the missingness must reduce the bias that a complete-case analysis
        # leaves behind when the outcome model cannot help.
        assert modelled < ignored
        assert modelled < 0.1

    def test_the_refutation_tests_run_on_a_missing_outcome_fit(self, fit) -> None:
        """The refuters carry ``Delta`` through, and the placebo still finds nothing.

        Each refuter rebuilds the ``CausalData`` -- a placebo treatment, an extra
        covariate, a row subset -- and a missing-outcome fit is the case where that
        rebuilding could quietly lose the indicator or desynchronise it from the
        outcome.  Nothing exercised this path before.
        """
        result, _ = fit
        refutation = result.validation.refute(estimand="ate", n_replicates=3, random_state=0)
        assert {test.name for test in refutation.tests} == {
            "placebo",
            "random_common_cause",
            "subset",
        }
        assert refutation.passed, refutation.summary()

    def test_the_nuisance_layer_refuses_to_ignore_missingness(self) -> None:
        """The low-level guard against silently dropping the missingness model."""
        from sklearn.linear_model import LinearRegression, LogisticRegression

        from cleverly import CausalData
        from cleverly.estimators._nuisance import fit_nuisances
        from cleverly.learners import Folds
        from cleverly.utils.bounds import OutcomeScaler

        frame, _ = make_missing_outcome(n=300, seed=62)
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta"
        )
        with pytest.raises(ValueError, match="no missingness_learner"):
            fit_nuisances(
                data,
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(),
                missingness_learner=None,
                intermediate_learner=None,
                folds=Folds.single(data.n),
                scaler=OutcomeScaler.from_outcome(data.outcome[data.observed]),
            )


class _MeanOnly:
    """An intercept-only regression: deliberately useless, for the DR test."""

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> _MeanOnly:
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self._mean = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self._mean)


class TestControlledDirectEffect:
    @pytest.fixture(scope="class")
    def fits(self) -> object:
        frame, _ = make_cde(n=2500, seed=63)
        return fast_tmle(estimands=("ate",)).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            intermediate="Z",
        )

    def test_one_result_per_level_of_the_intermediate(self, fits) -> None:
        assert isinstance(fits, TMLEResultSet)
        assert sorted(fits) == [0.0, 1.0]
        assert len(fits) == 2

    @pytest.fixture(scope="class")
    def replicates(self) -> dict[float, list[float]]:
        """One fit per seed, read at *both* levels of the intermediate.

        A single fit carries every level, so the two cases of the test below share these
        eight fits rather than each repeating them -- which is what the parametrize used
        to do, at n=1500 with three nuisances apiece.
        """
        estimates: dict[float, list[float]] = {0.0: [], 1.0: []}
        for seed in range(70, 78):
            frame, _ = make_cde(n=1500, seed=seed)
            result = fast_tmle(estimands=("ate",)).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=COVARIATES,
                intermediate="Z",
            )
            for level in estimates:
                estimates[level].append(result[level].psi("ate"))
        return estimates

    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_each_controlled_direct_effect_is_recovered(
        self, z: float, replicates: dict[float, list[float]]
    ) -> None:
        # Averaged over replications rather than asserted on one fit: a single-sample
        # coverage check fails 5% of the time by construction, which is a coin flip
        # dressed up as a test.
        truth = cde_dgp().truth(z)["ate"]
        estimates = replicates[z]

        mean = float(np.mean(estimates))
        monte_carlo_se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
        assert abs(mean - truth) < max(3.0 * monte_carlo_se, 0.05), (
            f"z={z}: mean {mean:.4f} vs truth {truth:.4f} (mc se {monte_carlo_se:.4f})"
        )

    def test_the_two_effects_differ_by_the_interaction(self, fits) -> None:
        # The process has an A-by-Z interaction of exactly 0.6.
        difference = fits[1.0].psi("ate") - fits[0.0].psi("ate")
        combined_se = np.hypot(fits[1.0]["ate"].std_error, fits[0.0]["ate"].std_error)
        assert difference == pytest.approx(0.6, abs=3.0 * combined_se)

    def test_the_intermediate_mechanism_is_fitted(self, fits) -> None:
        result = fits[0.0]
        assert result.nuisance.intermediate is not None
        assert result.nuisance.intermediate.shape == (result.data.n, 2)
        assert "intermediate" in {model.name for model in result.validation.nuisance().models}

    def test_the_score_equation_is_solved_for_both_levels(self, fits) -> None:
        for z in (0.0, 1.0):
            assert fits[z].validation.score_check().passed

    def test_the_result_set_stacks_into_one_frame(self, fits) -> None:
        frame = fits.to_frame()
        assert len(frame) == 2
        assert "intermediate" in frame.columns
        assert set(frame["intermediate"].to_list()) == {0.0, 1.0}

    def test_the_summary_covers_both_levels(self, fits) -> None:
        text = fits.summary()
        assert "Z = 0" in text
        assert "Z = 1" in text

    def test_an_unknown_level_is_refused(self, fits) -> None:
        with pytest.raises(KeyError, match="no result for"):
            fits[2.0]

    def test_ignoring_the_intermediate_gives_a_different_estimand(self) -> None:
        frame, _ = make_cde(n=2500, seed=64)
        controlled = fast_tmle(estimands=("ate",)).fit(
            frame, outcome="Y", treatment="A", covariates=COVARIATES, intermediate="Z"
        )
        total = (
            fast_tmle(estimands=("ate",))
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)
            .single()
        )
        # The total effect includes the pathway through Z, so it must exceed both
        # controlled direct effects in this process.
        assert total.psi("ate") > controlled[0.0].psi("ate")
        assert total.psi("ate") > controlled[1.0].psi("ate")


class TestCoverageStudiesRunPerLevel:
    """A coverage study has to be told which level it is measuring coverage for.

    Each level of the intermediate is a different parameter, so a study that guessed
    would be comparing an estimate at one level against a truth at another and reporting
    the mismatch as bias.  Until this landed, ``CoverageStudy`` refused a
    controlled-direct-effect fit outright, which left the estimand with no route to
    empirical coverage evidence at all.

    The tests here are about plumbing -- that the level reaches both the process and the
    result set, and that the two agree.  They deliberately do not assert coverage: three
    replications cannot, and a study sized to do so belongs in the nightly slow tier.
    """

    def _study(self, z: float | None, **kwargs: object) -> object:
        from cleverly.validation import CoverageStudy

        fit_kwargs = {
            "outcome": "Y",
            "treatment": "A",
            "covariates": COVARIATES,
            "intermediate": "Z",
        }
        fit_kwargs.update(kwargs.pop("fit_kwargs", {}))  # type: ignore[arg-type]
        return CoverageStudy(
            dgp=cde_dgp(),
            estimator=lambda: TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=3,
                random_state=0,
                simultaneous=False,
                estimands=("ate",),
            ),
            n=400,
            n_replicates=3,
            seed=1,
            intermediate_value=z,
            fit_kwargs=fit_kwargs,
            **kwargs,  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize(("z", "expected"), [(0.0, 0.9), (1.0, 1.5)])
    def test_the_truth_it_compares_against_is_the_effect_at_that_level(
        self, z: float, expected: float
    ) -> None:
        # cde_dgp's outcome mean is linear with a 0.6 * a * z interaction, so the
        # controlled direct effect is 0.9 + 0.6 * z exactly. Getting 0.9 for both levels
        # would be the signature of the level never reaching DGP.sample, which defaults
        # it to zero without saying so.
        study = self._study(z)
        assert study.run().summaries["ate"].truth == pytest.approx(expected, abs=1e-6)

    def test_a_level_is_required_when_the_fit_returns_a_set(self) -> None:
        with pytest.raises(ValueError, match=r"intermediate_value=0\.0 or 1\.0"):
            self._study(None)

    def test_a_level_without_an_intermediate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not name an intermediate"):
            self._study(1.0, fit_kwargs={"intermediate": None})

    def test_an_unrecognised_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"must be 0\.0 or 1\.0"):
            self._study(2.0)
