r"""Does Tier 2's design enter the regime it committed to, and by a learner?

``docs/drtmle/validation-plan.md`` §5 asks Tier 2 for *"a series, spline or histogram
regression with a smoothing sequence chosen in advance, so the rate is analysable and
reproducible"* -- and it is **the demonstration**, unlike Tier 1, whose nuisances are handed
over.  Four things have to hold for a coverage number from it to mean anything, and each
fails against a different mistake.

*The smoothing sequence is declared, not learned.*  A bandwidth chosen by cross-validation
would make the rate neither identified nor reproducible, which is §5's objection to a Super
Learner arriving through a back door.

*The drifting nuisance drifts and the wrong one stays wrong.*  A study reporting only the
first cannot tell a shrinking product from a converging pair, which is what §5 means by
"verifying the regime was entered".

*No drift coefficient vanishes.*  The remainder is an **inner product**, not a norm, so a
nuisance error of the right size can still leave :math:`c_a = 0`.  That is the trap §5 spends
a section on, and here it arrives twice: once in the arms and once in their contrast.

*The nuisances are fitted.*  ``tests/e2e/test_double_robustness.py``'s "correct" cell is an
oracle, which makes :math:`R_2` exactly zero and a plain ``TMLE``'s interval already valid;
the gap this study is about opens only where the good nuisance is estimated.  So the learners
here must learn -- a cell whose settings named an injected instance would be Tier 1 under a
second name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_tier2 as tier2

#: The study's sizes, which every rate below is read across.
SIZES = (600, 1200, 2400)


class TestTheSmoothingSequenceIsTheCommittedOne:
    """``h_n = c_h n^(-beta)``, declared before any fit and read off the study's size."""

    @pytest.mark.parametrize("n", SIZES)
    def test_the_bandwidth_follows_the_declared_law(self, n: int) -> None:
        assert tier2.bandwidth(n) == pytest.approx(tier2.BANDWIDTH_C * n**-tier2.BANDWIDTH_EXPONENT)

    def test_the_exponent_is_half_the_remainder_exponent(self) -> None:
        """A local-constant bias is ``O(h^2)``, which is what makes ``R_2`` drift at alpha."""
        assert pytest.approx(tier2.ALPHA / 2.0) == tier2.BANDWIDTH_EXPONENT

    def test_the_learner_reads_the_studys_size_and_not_its_training_folds(self) -> None:
        """The sequence is indexed by the sample the estimator was handed.

        A learner that took its bandwidth from the rows it happened to be trained on would
        follow ``(4/5)n`` under five-fold cross-fitting, so the realised rate would be the
        declared one times a constant the design never wrote down.
        """
        learner = tier2.KernelOutcome(2400)
        rows = 400
        design = np.column_stack([np.ones(rows), np.random.default_rng(0).normal(size=(rows, 4))])
        learner.fit(design, np.random.default_rng(1).uniform(0.2, 0.8, rows))
        assert learner.states_[1.0]["h"] == pytest.approx(tier2.bandwidth(2400))


class TestTheRegimeIsEnteredAndTheWrongNuisanceStaysWrong:
    """The pair §5 asks for: one norm falling, the other bounded away from zero."""

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_drifting_nuisance_falls_at_twice_the_bandwidth_exponent(self, cell: str) -> None:
        key = "q_error_1" if cell == "q-drift" else "g_error"
        errors = [tier2.nuisance_error(cell, n)[key] for n in SIZES]
        slope = np.polyfit(np.log(SIZES), np.log(errors), 1)[0]
        assert slope == pytest.approx(-2.0 * tier2.BANDWIDTH_EXPONENT, abs=0.01)

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_misspecified_nuisance_does_not_move_with_n(self, cell: str) -> None:
        key = "g_error" if cell == "q-drift" else "q_error_1"
        errors = [tier2.nuisance_error(cell, n)[key] for n in SIZES]
        assert errors[0] == pytest.approx(errors[-1], rel=1e-12)

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_misspecified_nuisance_is_wrong_by_a_wide_margin(self, cell: str) -> None:
        """Measured against the **truth's own spread**, not against an absolute number.

        A propensity that varies over ``[0.40, 0.60]`` has a standard deviation of ``0.09``,
        so an absolute bar borrowed from the outcome scale would call a misspecification that
        removes most of the mechanism's variation "small".  The scale-free statement is the
        one the design means: the wrong nuisance's error is a large share of what there was
        to get right.
        """
        dgp = tier2.base_law()
        if cell == "q-drift":
            spread = np.sqrt(
                dgp.expectation(lambda w: np.asarray(dgp.propensity(w)) ** 2)
                - dgp.expectation(lambda w: np.asarray(dgp.propensity(w))) ** 2
            )
            error = tier2.nuisance_error(cell, SIZES[0])["g_error"]
        else:
            spread = np.sqrt(
                dgp.expectation(lambda w: np.asarray(dgp.outcome_mean(w, 1.0, None)) ** 2)
                - dgp.expectation(lambda w: np.asarray(dgp.outcome_mean(w, 1.0, None))) ** 2
            )
            error = tier2.nuisance_error(cell, SIZES[0])["q_error_1"]
        assert error >= 0.5 * spread, (cell, error, spread)


class TestNoDriftCoefficientVanishes:
    """§5's inner-product trap, measured rather than hoped for.

    Both arms *and* the contrast, because :math:`c_1 - c_0` can vanish with both arm
    coefficients nonzero -- which is the case §5 gives a finite-support example of.
    """

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_every_coefficient_clears_the_floor(self, cell: str) -> None:
        realised = tier2.drift_coefficients(cell)
        for key in ("c1", "c0", "c_ate"):
            assert abs(realised[key]) >= tier2.C_MIN, (cell, key, realised)

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_arms_have_opposite_signs_so_the_contrast_is_a_sum(self, cell: str) -> None:
        """Which is what makes cancellation in the ATE impossible rather than unlikely."""
        realised = tier2.drift_coefficients(cell)
        assert realised["c1"] * realised["c0"] < 0.0
        assert abs(realised["c_ate"]) > max(abs(realised["c1"]), abs(realised["c0"]))

    def test_the_predicted_remainder_scales_at_the_declared_exponent(self) -> None:
        """``n^alpha R_2 -> c`` is what the design predicts; the study measures it."""
        for cell in tier2.CELLS:
            declared = tier2.drift_coefficients(cell)["c_ate"]
            for n in SIZES:
                predicted = tier2.exact_remainder(cell, n)["r2_ate"]
                assert n**tier2.ALPHA * predicted == pytest.approx(declared, rel=1e-12)


class TestBothNuisancesAreFitted:
    """The trap the roadmap records: an oracle makes ``R_2`` zero and the gap disappear."""

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_neither_learner_ignores_its_training_rows(self, cell: str) -> None:
        """A learner that returned the same function whatever it saw would be an injection.

        Fitted on two deliberately different targets and asked for the same design: an
        oracle answers identically, and a learner does not.
        """
        settings = tier2.settings(cell, 600)
        rng = np.random.default_rng(4)
        rows = 300
        covariates = rng.normal(size=(rows, 4))
        arms = (rng.uniform(size=rows) < 0.5).astype(float)
        design = np.column_stack([arms, covariates])

        outcome = settings["outcome_learner"]
        first = outcome.fit(design, rng.uniform(0.2, 0.8, rows)).predict(design)
        second = outcome.fit(design, rng.uniform(0.2, 0.8, rows)).predict(design)
        assert not np.allclose(first, second)

        treatment = settings["treatment_learner"]
        one = treatment.fit(covariates, arms).predict_proba(covariates)[:, 1]
        two = treatment.fit(covariates, 1.0 - arms).predict_proba(covariates)[:, 1]
        assert not np.allclose(one, two)

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_reduced_learners_are_not_named_by_the_cell(self, cell: str) -> None:
        """Tier 1's check, at Tier 2's settings.

        ``DRTMLE``'s reductions fall back to the primary *specification*, so a cell that
        named its smoother here would make :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}`
        smoothers of a smoother rather than the univariate regressions the derivation is
        about.  The harness names ``"glm"`` instead, and this is what keeps that a decision.
        """
        settings = tier2.settings(cell, 600)
        assert "reduced_outcome_learner" not in settings
        assert "reduced_treatment_learner" not in settings

    def test_the_wrong_outcome_models_error_is_the_dropped_terms(self) -> None:
        """Analytic because the covariates are independent standard normals.

        The least-squares limit of a subset model is then the truth with the dropped terms
        deleted -- which is what makes :func:`~benchmarks.drtmle_tier2.outcome_error` exact
        rather than an approximation, and so what makes the drift coefficient a quadrature.
        """
        latent = np.random.default_rng(2).normal(size=(500, 4))
        coefficients = {0: 1.0, 1: 0.5, 2: -0.8, 3: 0.4}
        for arm, kept in tier2.WRONG_OUTCOME_COLUMNS.items():
            dropped = [index for index in coefficients if index not in kept]
            expected = -sum(coefficients[index] * latent[:, index] for index in dropped)
            np.testing.assert_allclose(tier2.outcome_error(latent, arm), expected)

    def test_the_wrong_mechanisms_limit_is_not_the_truth(self) -> None:
        dgp = tier2.base_law()
        latent = np.random.default_rng(3).normal(size=(2000, 4))
        gap = tier2.wrong_mechanism(latent) - np.asarray(dgp.propensity(latent))
        assert float(np.sqrt(np.mean(gap**2))) > 10 * tier2.C_MIN / 10


class TestTheCellsShareTierOnesInterface:
    """One harness reads both tiers, so the two modules supply the same names.

    Not tidiness: ``benchmarks/drtmle_coverage.py`` selects a tier by module and every table
    it prints reads these seven names.  A tier missing one would fail at the table rather
    than at the selection, half way through a dispatch.
    """

    @pytest.mark.parametrize(
        "name",
        (
            "CELLS",
            "ALPHA",
            "base_law",
            "settings",
            "drift_coefficients",
            "exact_remainder",
            "nuisance_error",
            "summary_rows",
            "SUMMARY_HEADERS",
        ),
    )
    def test_the_name_is_present(self, name: str) -> None:
        from benchmarks import drtmle_injection

        assert hasattr(tier2, name)
        assert hasattr(drtmle_injection, name)

    def test_the_cells_are_the_same_two(self) -> None:
        from benchmarks import drtmle_injection

        assert tier2.CELLS == drtmle_injection.CELLS

    def test_the_remainder_exponent_is_the_same(self) -> None:
        """What two tiers must share to be about one regime is the remainder's rate."""
        from benchmarks import drtmle_injection

        assert tier2.ALPHA == drtmle_injection.ALPHA

    def test_an_unknown_cell_is_refused(self) -> None:
        for call in (
            lambda: tier2.settings("nonsense", 600),
            lambda: tier2.drift_coefficients("nonsense"),
        ):
            with pytest.raises(ValueError, match="cell must be one of"):
                call()


class TestTheSettingsAreTheStudysOwn:
    """What a cell hands the estimator, beyond its learners."""

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_scaler_is_declared_rather_than_recovered(self, cell: str) -> None:
        """Tier 1's decision, at Tier 2's sizes and for a sharper reason.

        A recovered scaler carries an ``O(n^(-1/2))`` error from the outcome noise, and here
        that is the same order as the *variance* of the slow nuisance -- so a recovered one
        would put a second sequence inside the one the design committed to.
        """
        assert tier2.settings(cell, 600)["q_bounds"] == tier2.Q_BOUNDS

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_estimands_are_spelled_out(self, cell: str) -> None:
        """``att``/``atc`` are the binary default and ``DRTMLE`` refuses them."""
        assert tier2.settings(cell, 600)["estimands"] == ("ate", "ey1", "ey0")


def _fitted(cell: str, n: int, seed: int) -> Any:
    """One end-to-end fit at a cell's settings, for the claims that need a real one."""
    from cleverly import DRTMLE

    dgp = tier2.base_law()
    frame, _ = dgp.sample(n, seed=seed)
    return (
        DRTMLE(
            **tier2.settings(cell, n),
            reduced_outcome_learner="glm",
            reduced_treatment_learner="glm",
            random_state=seed,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


class TestACellFitsAndStaysInsideTheContract:
    """One fit per cell at the smallest size, which is what the design turns on.

    Item 25: a cell whose truncations are active is evidence about the constrained rendering
    rather than about Theorem 1's estimator, and the base law was chosen for **overlap** so
    that these cells are inside it.  Whether that worked is measured here rather than
    assumed -- as C1 found, on a well-overlapped law it is not automatic.
    """

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_fit_returns_and_solves_what_it_reports(self, cell: str) -> None:
        result = _fitted(cell, 300, seed=8)
        assert set(result.estimates) == {"ate", "ey1", "ey0"}
        assert np.isfinite(result.estimates["ate"].psi)
        assert result.validation.score_check().passed

    @pytest.mark.parametrize("cell", tier2.CELLS)
    def test_the_initial_mechanism_does_not_clip(self, cell: str) -> None:
        """The half of item 25's contract the base law was chosen to secure."""
        check = _fitted(cell, 300, seed=8).validation.correction_check()
        assert check.initial_clip_share == pytest.approx(0.0)
