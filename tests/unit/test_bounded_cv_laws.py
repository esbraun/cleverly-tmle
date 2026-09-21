"""The bounded cross-fitted laws, and the design checks the cells are declared against.

Two kinds of check live here.  The *quadrature* statements are too expensive to run at
every import of a study driver, so :mod:`tests.studies.bounded_cv_laws` states them as
cached measurement functions and this module asserts them once per session.  The
*mutation* statements are the ones that say a design check bites: each one puts a
degenerate law or an interchanged threshold into the declared design and requires the
check to refuse it.

The two halves are the same argument.  An exact-law check is blind to a term that vanishes
at the truth, and a design check whose witness has quietly gone to zero passes every law
including a broken one.  Each check below therefore has a partner that must fail.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.bounds import OutcomeScaler
from tests.conftest import OracleOutcomeUnit
from tests.studies import (
    bounded_cv_laws,
    canonical_properties,
    ctmle_oat_properties,
    ctmle_selector_properties,
    cvtmle_properties,
)
from tests.studies.canonical_cvtmle import G_BOUNDS as CV_G_BOUNDS
from tests.studies.canonical_cvtmle import STUDY as CANONICAL_CVTMLE
from tests.studies.evidence import document, property_verdicts
from tests.studies.evidence.properties import PropertyCell, summarize_cells


@pytest.fixture(autouse=True)
def _clean_design_cache() -> Any:
    """Drop the cached design verdict around any test that mutates a law."""
    bounded_cv_laws.assert_bounded_law_design.cache_clear()
    yield
    bounded_cv_laws.assert_bounded_law_design.cache_clear()


class TestTheLawsStayInsideTheUnitInterval:
    """Every declared law's mean, on the grid its truth is integrated over."""

    @pytest.mark.parametrize("law", bounded_cv_laws.declared_laws(), ids=lambda law: str(law.name))
    def test_the_mean_and_both_beta_shapes_stay_usable(self, law: Any) -> None:
        low, high, shape = bounded_cv_laws.mean_range(law)
        assert 0.0 < low < high < 1.0, (
            f"{law.name} reaches [{low}, {high}] on the quadrature grid, outside the open "
            f"unit interval a beta law needs"
        )
        assert shape >= bounded_cv_laws.MINIMUM_BETA_SHAPE, (
            f"{law.name}'s smallest beta shape is {shape}, below the declared floor; a "
            f"shape near zero puts real probability on a draw that rounds to an endpoint"
        )

    @pytest.mark.parametrize("law", bounded_cv_laws.declared_laws(), ids=lambda law: str(law.name))
    def test_a_draw_lands_strictly_inside_the_unit_interval(self, law: Any) -> None:
        frame, _ = law.sample(2_000, seed=11)
        outcome = frame["Y"].to_numpy()
        assert bool(((outcome > 0.0) & (outcome < 1.0)).all())

    def test_a_law_whose_mean_leaves_the_interval_refuses_to_sample(self) -> None:
        """The package's own refusal, which is what makes the check above a gate.

        The declared laws cannot reach this state, so nothing would exercise it.  A law
        that could is built here instead, and it must be refused at ``sample`` rather than
        clipped, because ``truth`` integrates the unclipped mean.
        """
        broken = replace(
            bounded_cv_laws.linear_dgp(),
            name="bounded_linear_unclamped",
            outcome_mean=lambda w, a, z: 0.5 + w[:, 0],
        )
        with pytest.raises(ValueError, match=r"(?i)beta|unit interval"):
            broken.sample(64, seed=0)


class TestTheDoubleRobustnessDesign:
    """The bounded law's contrast, its range, and the thresholds they are read against."""

    def test_the_declared_design_accepts_the_declared_cells(self) -> None:
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        canonical_properties.assert_double_robustness_design(cells, design=bounded_cv_laws.DESIGN)

    def test_the_gaussian_thresholds_reject_the_bounded_law(self) -> None:
        """Why the design is a parameter and not a constant.

        The bounded outcome lives in ``(0, 1)``, so its contrast is an order of magnitude
        smaller than the Gaussian law's and the Gaussian spread threshold rejects it.  A
        single shared pair of constants could only be the looser of the two, which would
        stop rejecting the law it was written for.
        """
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        with pytest.raises(RuntimeError, match="contrast witness"):
            canonical_properties.assert_double_robustness_design(
                cells, design=canonical_properties.GAUSSIAN_DESIGN
            )

    def test_the_gaussian_row_keeps_its_own_law_and_thresholds(self) -> None:
        """The in-sample row is untouched: same law, same two constants, same witness."""
        assert canonical_properties.GAUSSIAN_DESIGN.contrast_spread == 0.5
        assert canonical_properties.GAUSSIAN_DESIGN.wrong_q_error == 0.25
        cells = canonical_properties.cells()
        law = next(cell.dgp for cell in cells if cell.property == "double_robustness")
        assert law.name == "canonical_double_robustness_bounded_nonlinear"
        assert law.family == "gaussian"
        canonical_properties.assert_double_robustness_design(cells)

    def test_a_flattened_law_fails_the_design_check(self, monkeypatch: Any) -> None:
        """Deliberate mutation: ``kappa -> 0`` collapses the mean to a constant 0.5.

        The law still samples, still has a truth and still has a propensity, so nothing
        upstream refuses it.  What it loses is the only thing the double-robustness family
        measures: a contrast a main-effects regression can be wrong about.  Both clauses of
        the bounded witness must therefore refuse it.
        """
        monkeypatch.setattr(bounded_cv_laws, "DOUBLE_ROBUST_LOGIT_SLOPE", 0.0)
        bounded_cv_laws.assert_bounded_law_design.cache_clear()
        law = bounded_cv_laws.double_robustness_dgp()
        assert float(np.ptp(np.asarray(law.truth()["ate"]))) == 0.0

        with pytest.raises(RuntimeError, match="analytic mean range"):
            bounded_cv_laws.assert_bounded_law_design()
        with pytest.raises(RuntimeError):
            bounded_cv_laws.bounded_cells("cvtmle_properties")

    def test_the_bounded_thresholds_accept_a_flattened_gaussian_law(self) -> None:
        """The other half of the claim above, which nothing used to check.

        The claim is that one shared pair of thresholds cannot read both laws. Half of it is
        that the Gaussian pair rejects the bounded law, which the test above shows. This is
        the other half: a Gaussian law flattened until its own design refuses it still clears
        the bounded pair, so the looser pair would stop rejecting the law it was written for.
        """
        gaussian = canonical_properties.double_robustness_dgp()
        base = gaussian.outcome_mean

        def flattened(w: Any, a: float, z: Any) -> Any:
            reference = np.asarray(base(w, 0.0, z), dtype=float)
            return reference + 0.2 * (np.asarray(base(w, a, z), dtype=float) - reference)

        flat = replace(gaussian, outcome_mean=flattened)
        cells = tuple(
            replace(cell, dgp=flat) if cell.property == "double_robustness" else cell
            for cell in canonical_properties.cells()
        )

        # The witness is the law's own, and this law is no longer the one it witnesses, so
        # both designs carry the bounded module's witness and differ in their thresholds
        # alone.  That is the comparison the claim is about.
        def witness(dgp: Any) -> None:
            return None

        strict = replace(canonical_properties.GAUSSIAN_DESIGN, witness=witness)
        loose = replace(
            strict,
            contrast_spread=bounded_cv_laws.DOUBLE_ROBUST_CONTRAST_SPREAD,
            wrong_q_error=bounded_cv_laws.DOUBLE_ROBUST_WRONG_Q_ERROR,
        )
        with pytest.raises(RuntimeError, match="contrast witness"):
            canonical_properties.assert_double_robustness_design(cells, design=strict)
        canonical_properties.assert_double_robustness_design(cells, design=loose)

    def test_a_recentred_law_fails_the_contrast_witness(self, monkeypatch: Any) -> None:
        """A second mutation, which the range clause alone would not catch.

        Pushing the centre far outside the law's own range leaves the clamp saturated at
        both arms, so the analytic range is still attained and only the contrast vanishes.
        """
        monkeypatch.setattr(bounded_cv_laws, "DOUBLE_ROBUST_CENTRE", 500.0)
        bounded_cv_laws.assert_bounded_law_design.cache_clear()
        with pytest.raises(RuntimeError, match="deterministic outcome contrast"):
            bounded_cv_laws.assert_bounded_law_design()

    def test_the_population_both_wrong_bias_clears_the_margin(self) -> None:
        """The control has to fail, and this says it fails by quadrature rather than luck."""
        bias = bounded_cv_laws.both_wrong_limit_bias(CV_G_BOUNDS)
        assert abs(bias) >= bounded_cv_laws.BOTH_WRONG_LIMIT_BIAS, (
            f"the both-wrong population bias is {bias}, below the declared threshold; the "
            f"control cannot be discriminated from a correct fit"
        )

    def test_a_flattened_law_loses_the_population_bias(self, monkeypatch: Any) -> None:
        """The partner mutation: with no contrast there is nothing to be wrong about."""
        monkeypatch.setattr(bounded_cv_laws, "DOUBLE_ROBUST_LOGIT_SLOPE", 0.0)
        bounded_cv_laws.both_wrong_limit_bias.cache_clear()
        try:
            bias = bounded_cv_laws.both_wrong_limit_bias(CV_G_BOUNDS)
        finally:
            bounded_cv_laws.both_wrong_limit_bias.cache_clear()
        assert abs(bias) < bounded_cv_laws.BOTH_WRONG_LIMIT_BIAS


class TestTheCorrectlySpecifiedLearner:
    """The root-n and calibration cells claim two correct nuisances; this checks one."""

    def test_the_quasibinomial_solver_recovers_the_linear_law(self) -> None:
        error = bounded_cv_laws.quasibinomial_recovery()
        assert error <= bounded_cv_laws.LINEAR_RECOVERY_TOLERANCE, (
            f"the quasibinomial fit leaves {error} on the declared coefficients, so the "
            f"law is not the logistic-linear one those cells call correctly specified"
        )

    def test_a_nonlinear_term_breaks_the_recovery(self, monkeypatch: Any) -> None:
        """Partner mutation: a squared term the learner has no column for."""
        law = bounded_cv_laws.linear_dgp()
        base = law.outcome_mean

        def curved(w: Any, a: float, z: float | None) -> Any:
            return np.clip(np.asarray(base(w, a, z)) + 0.05 * w[:, 0] ** 2, 1e-6, 1 - 1e-6)

        monkeypatch.setattr(
            bounded_cv_laws, "linear_dgp", lambda: replace(law, outcome_mean=curved)
        )
        bounded_cv_laws.quasibinomial_recovery.cache_clear()
        try:
            error = bounded_cv_laws.quasibinomial_recovery(2**12)
        finally:
            bounded_cv_laws.quasibinomial_recovery.cache_clear()
        assert error > bounded_cv_laws.LINEAR_RECOVERY_TOLERANCE


class TestTheNullLaw:
    """The sharp null has to be sharp, and confounded."""

    def test_the_two_arm_means_are_identically_equal(self) -> None:
        law = bounded_cv_laws.null_dgp()
        latent = np.random.default_rng(3).normal(size=(256, law.n_latent))
        assert np.array_equal(
            law.outcome_mean(latent, 1.0, None), law.outcome_mean(latent, 0.0, None)
        )
        assert law.truth()["ate"] == 0.0

    def test_a_law_with_a_treatment_term_at_the_null_is_refused(self, monkeypatch: Any) -> None:
        """Partner mutation: an effect that survives at ``effect = 0``."""
        original = bounded_cv_laws.null_dgp

        def leaky(effect: float = 0.0) -> Any:
            law = original(effect)
            base = law.outcome_mean
            return replace(
                law, outcome_mean=lambda w, a, z: np.clip(base(w, a, z) + 0.01 * a, 1e-6, 1 - 1e-6)
            )

        monkeypatch.setattr(bounded_cv_laws, "null_dgp", leaky)
        bounded_cv_laws.assert_bounded_law_design.cache_clear()
        with pytest.raises(RuntimeError, match="identically equal"):
            bounded_cv_laws.assert_bounded_law_design()

    def test_the_sharp_null_is_confounded_by_quadrature(self) -> None:
        """The declared displacement, read against the law rather than against a pilot note.

        ``NULL_CONFOUNDING_DISPLACEMENT`` was read by nothing at all. A calibrated rejection
        rate on a law with no confounding is evidence about a randomized experiment rather
        than about the estimator, which is exactly what the constant refuses.
        """
        measured = bounded_cv_laws.null_confounding_displacement()
        assert measured >= bounded_cv_laws.NULL_CONFOUNDING_DISPLACEMENT, (
            f"the bounded sharp null displaces an unadjusted contrast by {measured} standard "
            f"errors at n = {bounded_cv_laws.NULL_CONFOUNDING_N}, below the declared floor"
        )

    def test_an_unconfounded_null_loses_that_displacement(self, monkeypatch: Any) -> None:
        """The partner mutation: drop the confounders and the displacement goes to zero."""
        monkeypatch.setattr(bounded_cv_laws, "NULL_WEIGHTS", (0.0, 0.0, 0.0))
        bounded_cv_laws.null_confounding_displacement.cache_clear()
        try:
            measured = bounded_cv_laws.null_confounding_displacement()
        finally:
            bounded_cv_laws.null_confounding_displacement.cache_clear()
        assert measured < bounded_cv_laws.NULL_CONFOUNDING_DISPLACEMENT

    def test_the_two_declared_effects_move_the_contrast_upward(self) -> None:
        for effect in (bounded_cv_laws.ALTERNATIVE_EFFECT, bounded_cv_laws.GENERATED_DESIGN_EFFECT):
            assert bounded_cv_laws.null_dgp(effect).truth()["ate"] > 0.0

    def test_the_gaussian_generated_design_effect_is_mirrored_faithfully(self) -> None:
        """The one constant this module copies rather than imports, checked against its source."""
        assert (
            bounded_cv_laws.GAUSSIAN_GENERATED_DESIGN_EFFECT
            == ctmle_oat_properties.GENERATED_DESIGN_EFFECT
        )


class TestTheGeneratedDesignRule:
    """The rule that sizes the generated-design law, executed rather than only written.

    The rule at ``bounded_cv_laws.GENERATED_DESIGN_EFFECT`` has two clauses.  The floor is a
    design quantity of two laws and is deterministic, so it is checked here.  The
    control-discrimination clause is a Monte Carlo measurement and is recorded in the module
    docstring; the registered run publishes the deficit it produced.
    """

    #: The 0.05 grid the rule declares.
    GRID = 0.05

    @staticmethod
    def _standardized(law: Any) -> float:
        """A law's ATE over the square root of its mean conditional outcome variance."""
        latent = law.quadrature()
        propensity = np.asarray(law.propensity(latent), dtype=float)
        variance = 0.0
        for arm, weight in ((1.0, propensity), (0.0, 1.0 - propensity)):
            mean = np.asarray(law.outcome_mean(latent, arm, None), dtype=float)
            if law.family == "beta":
                within = mean * (1.0 - mean) / (1.0 + float(law.concentration))
            else:
                within = np.full_like(mean, law.noise_scale**2)
            variance += float(np.mean(weight * within))
        return float(law.truth()["ate"] / np.sqrt(variance))

    def _target(self) -> float:
        return self._standardized(
            canonical_properties.null_dgp(bounded_cv_laws.GAUSSIAN_GENERATED_DESIGN_EFFECT)
        )

    def test_the_declared_effect_is_the_smallest_grid_point_above_the_design_floor(self) -> None:
        """The floor clause, and the clause that says the rule took the smallest value.

        The second half is what separates a rule from a search for headroom. Without it any
        coefficient above the floor would pass, and the largest would pass most comfortably.
        """
        target = self._target()
        effect = bounded_cv_laws.GENERATED_DESIGN_EFFECT
        assert self._standardized(bounded_cv_laws.null_dgp(effect)) >= target
        below = round(effect - self.GRID, 10)
        assert below > 0.0
        assert self._standardized(bounded_cv_laws.null_dgp(below)) < target

    def test_the_retired_coefficient_sits_far_above_that_floor(self) -> None:
        """Why the re-derivation moved the number, stated as a measurement.

        The coefficient this law carried before was the Gaussian law's own 0.3, which is a
        shift of an unbounded mean rather than of a logit. On the standardized scale the two
        are not the same statement: 0.3 here is more than twice the Gaussian law's signal.
        """
        retired = self._standardized(bounded_cv_laws.null_dgp(0.3))
        assert retired > 2.0 * self._target()

    def test_the_budget_is_a_doubling_of_the_gaussian_studys(self) -> None:
        """The ladder clause. The budget is a rung, not an arbitrary count."""
        budget = bounded_cv_laws.GENERATED_DESIGN_REPLICATES
        base = canonical_properties.DOUBLE_ROBUST_REPLICATES
        assert budget in {base, 2 * base, 4 * base}


class TestTheInstrumentLaw:
    """``W2`` is the instrument, and the selector-necessity control depends on its absence."""

    def test_the_instrument_is_absent_from_the_outcome(self) -> None:
        law = bounded_cv_laws.instrument_dgp()
        rows = np.zeros((2, 3))
        rows[1, 1] = 5.0
        assert np.array_equal(
            law.outcome_mean(rows, 1.0, None)[0], law.outcome_mean(rows, 1.0, None)[1]
        )

    def test_an_instrument_that_enters_the_outcome_is_refused(self, monkeypatch: Any) -> None:
        law = bounded_cv_laws.instrument_dgp()
        base = law.outcome_mean
        monkeypatch.setattr(
            bounded_cv_laws,
            "instrument_dgp",
            lambda: replace(
                law,
                outcome_mean=lambda w, a, z: np.clip(
                    np.asarray(base(w, a, z)) + 0.02 * w[:, 1], 1e-6, 1 - 1e-6
                ),
            ),
        )
        bounded_cv_laws.assert_bounded_law_design.cache_clear()
        with pytest.raises(RuntimeError, match="no longer an instrument"):
            bounded_cv_laws.assert_bounded_law_design()


class TestTheInheritedCells:
    """The bounded twins carry every budget, seed and role the Gaussian cells declare."""

    @pytest.mark.parametrize(
        ("consumer", "exclude"),
        [
            ("cvtmle_properties", ()),
            ("ctmle_selector_properties", ("double_robustness",)),
            ("ctmle_oat_properties", ("double_robustness",)),
        ],
    )
    def test_every_field_but_the_law_and_the_learners_is_carried_over(
        self, consumer: str, exclude: tuple[str, ...]
    ) -> None:
        offset = bounded_cv_laws.INHERITED_SEED_OFFSETS[consumer]
        gaussian = [cell for cell in canonical_properties.cells() if cell.property not in exclude]
        bounded = bounded_cv_laws.bounded_cells(consumer, exclude=exclude)
        assert len(bounded) == len(gaussian)
        for source, twin in zip(gaussian, bounded, strict=True):
            assert (twin.property, twin.cell, twin.role) == (
                source.property,
                source.cell,
                source.role,
            )
            assert (twin.n, twin.replicates, twin.estimand) == (
                source.n,
                source.replicates,
                source.estimand,
            )
            assert twin.seed == source.seed + offset
            assert twin.dgp.family == "beta"

    def test_the_declared_offsets_are_the_ones_the_modules_apply(self) -> None:
        """Read off the committed Gaussian cells, so a twin cannot quietly change a stream."""
        canonical = {(cell.property, cell.cell): cell.seed for cell in canonical_properties.cells()}
        # Each module *redeclares* ``double_robustness`` on a law of its own rather than
        # inheriting it, so those cells carry the module's own seeds and say nothing about
        # the offset.  They are excluded here and by ``bounded_cells`` for the same reason.
        for cells, consumer, declared in (
            (cvtmle_properties.cells("stacked"), "cvtmle_properties", ()),
            (
                ctmle_selector_properties.cells(),
                "ctmle_selector_properties",
                ("double_robustness",),
            ),
            (ctmle_oat_properties.cells(), "ctmle_oat_properties", ("double_robustness",)),
        ):
            offsets = {
                cell.seed - canonical[cell.property, cell.cell]
                for cell in cells
                if (cell.property, cell.cell) in canonical and cell.property not in declared
            }
            assert offsets == {bounded_cv_laws.INHERITED_SEED_OFFSETS[consumer]}

    def test_a_consumer_without_a_declared_offset_is_refused(self) -> None:
        with pytest.raises(KeyError, match="inherited seed offset"):
            bounded_cv_laws.bounded_cells("some_new_properties")

    def test_a_law_without_a_declared_twin_is_refused(self) -> None:
        cell = canonical_properties.cells()[0]
        stranger = replace(cell, dgp=replace(cell.dgp, name="a_law_nobody_declared"))
        with pytest.raises(KeyError, match="no declared bounded twin"):
            bounded_cv_laws.bounded_twin(stranger)

    def test_an_unmapped_learner_is_refused(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        cell = next(
            cell for cell in canonical_properties.cells() if cell.property == "double_robustness"
        )
        with pytest.raises(TypeError, match="no declared bounded counterpart"):
            bounded_cv_laws.bounded_twin(replace(cell, outcome_learner=RandomForestRegressor))

    def test_the_oracle_outcome_learner_becomes_the_exact_unit_oracle(self) -> None:
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        correct = next(cell for cell in cells if cell.cell == "both_correct")
        assert isinstance(correct.outcome_learner(), OracleOutcomeUnit)

    def test_the_correctly_specified_outcome_learner_is_the_quasibinomial_solver(self) -> None:
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        ladder = next(cell for cell in cells if cell.property == "root_n_and_efficiency")
        assert type(ladder.outcome_learner()).__name__ == "QuasiBinomialGLM"


class TestTheUnitOracle:
    """Exact under two conditions, and each one has its own guard."""

    def test_it_returns_the_true_mean_unchanged(self) -> None:
        law = bounded_cv_laws.double_robustness_dgp()
        frame, _ = law.sample(200, seed=5)
        covariates = frame[["W1", "W2", "W3", "W4"]].to_numpy()
        design = np.column_stack([frame["A"].to_numpy(), covariates])
        oracle = OracleOutcomeUnit(law).fit(design, frame["Y"].to_numpy())
        expected = np.where(
            frame["A"].to_numpy() == 1.0,
            law.outcome_mean(covariates, 1.0, None),
            law.outcome_mean(covariates, 0.0, None),
        )
        assert np.array_equal(oracle.predict(design), expected)

    def test_it_refuses_a_law_whose_mean_leaves_the_unit_interval(self) -> None:
        from cleverly.datasets import nonlinear_dgp

        law = nonlinear_dgp()
        frame, _ = law.sample(100, seed=5)
        design = np.column_stack(
            [frame["A"].to_numpy(), frame[["W1", "W2", "W3", "W4"]].to_numpy()]
        )
        with pytest.raises(ValueError, match="outside the open unit interval"):
            OracleOutcomeUnit(law).fit(design, frame["Y"].to_numpy())

    def test_the_scaler_guard_accepts_a_declared_unit_scale(self) -> None:
        result = _preflight_result()
        bounded_cv_laws.assert_unit_outcome_scaler(result)

    def test_the_scaler_guard_refuses_a_derived_scale(self) -> None:
        """Deliberate mutation: put a derived scaler on an otherwise valid result."""

        class _Derived:
            scaler = OutcomeScaler(-0.05, 1.05)

        class _Result:
            nuisance = _Derived()

        with pytest.raises(RuntimeError, match="identity outcome scaler"):
            bounded_cv_laws.assert_unit_outcome_scaler(_Result())


def _estimator(cell: PropertyCell) -> Any:
    """The cross-fitted construction the bounded double-robustness preflight runs."""
    return lambda: TMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=True,
        n_folds=5,
        estimands=cell.estimand,
        simultaneous=False,
        g_bounds=CV_G_BOUNDS,
        q_bounds=bounded_cv_laws.Q_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def _preflight_result() -> Any:
    cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
    return canonical_properties.run_double_robustness_preflight(
        cells, _estimator, g_bounds=CV_G_BOUNDS, n=400, design=bounded_cv_laws.DESIGN
    )


class TestTheFittedPreflight:
    """One fit, which is where the design check meets the estimator it is about."""

    def test_the_bounded_preflight_passes_its_fitted_controls(self) -> None:
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        cell = next(cell for cell in cells if cell.cell == "treatment_correct")
        result = _preflight_result()
        bounded_cv_laws.assert_bounded_double_robustness_fit(cell, result, g_bounds=CV_G_BOUNDS)

    def test_the_gaussian_thresholds_reject_the_fitted_bounded_witness(self) -> None:
        """The fitted half of the parametrization, mutated the same way as the design half."""
        cells = bounded_cv_laws.bounded_cells("cvtmle_properties")
        cell = next(cell for cell in cells if cell.cell == "treatment_correct")
        result = _preflight_result()
        with pytest.raises(RuntimeError, match="witness vanished"):
            canonical_properties.assert_double_robustness_fit(
                cell,
                result,
                g_bounds=CV_G_BOUNDS,
                design=canonical_properties.GAUSSIAN_DESIGN,
            )


class TestTheFoldPolicySeam:
    """Three policies, one law, and a partition that actually differs between them."""

    def test_every_cell_is_diagnostic_and_shares_one_seed(self) -> None:
        cells = bounded_cv_laws.fold_policy_cells(CANONICAL_CVTMLE)
        assert [cell.cell for cell in cells] == list(bounded_cv_laws.FOLD_POLICIES)
        assert {cell.role for cell in cells} == {"diagnostic"}
        assert {cell.seed for cell in cells} == {bounded_cv_laws.fold_policy_seed(CANONICAL_CVTMLE)}
        assert {cell.dgp.family for cell in cells} == {"binomial"}

    def test_the_family_draws_from_its_own_stream(self) -> None:
        """The literal this seed replaced collided with another study's inherited cell.

        ``ctmle_oat_properties`` inherits ``type_i_error/sharp_null`` at seed ``9_100`` under
        an offset of ``5_000``, which is the ``14_100`` this family used to write out. The
        two cells then drew bit-identical covariates. An offset table cannot separate them,
        because a study-specific family sits outside every offset, so the seed is hashed from
        the registering record instead.
        """
        inherited = {
            cell.seed
            for cell in bounded_cv_laws.bounded_cells(
                "ctmle_oat_properties", exclude=("double_robustness",)
            )
        }
        sharp_null = next(
            cell.seed
            for cell in bounded_cv_laws.bounded_cells(
                "ctmle_oat_properties", exclude=("double_robustness",)
            )
            if cell.property == "type_i_error"
        )
        assert sharp_null == 14_100
        seed = bounded_cv_laws.fold_policy_seed(CANONICAL_CVTMLE)
        assert seed != sharp_null
        assert seed not in inherited

    def test_the_estimator_passes_no_q_bounds_on_the_binary_law(self) -> None:
        """A binary outcome already has the identity scaler, and ``q_bounds`` is refused."""
        cell = bounded_cv_laws.fold_policy_cells(CANONICAL_CVTMLE)[0]
        estimator = bounded_cv_laws.fold_policy_estimator(g_bounds=CV_G_BOUNDS)(cell)()
        assert estimator.q_bounds is None

    @pytest.mark.parametrize("policy", bounded_cv_laws.FOLD_POLICIES)
    def test_each_policy_draws_the_split_it_names(self, policy: str) -> None:
        law = bounded_cv_laws.fold_policy_dgp()
        frame, _ = law.sample(300, seed=2)
        cell = next(
            cell
            for cell in bounded_cv_laws.fold_policy_cells(CANONICAL_CVTMLE)
            if cell.cell == policy
        )
        estimator = bounded_cv_laws.fold_policy_estimator(g_bounds=CV_G_BOUNDS)(cell)()
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
        folds = result.nuisance.folds
        treatment = frame["A"].to_numpy()
        outcome = frame["Y"].to_numpy()
        shares = np.array([treatment[test].mean() for _, test in folds if len(test)], dtype=float)
        crossed = np.array(
            [(treatment[test] * outcome[test]).mean() for _, test in folds if len(test)],
            dtype=float,
        )
        if policy == "unstratified":
            assert float(np.ptp(shares)) > 0.0
        else:
            # A stratified draw balances its strata to within one row per fold, so the
            # realized share moves far less than an unstratified draw's does.
            assert float(np.ptp(shares)) < 0.06
        if policy == "treatment_outcome_stratified":
            assert float(np.ptp(crossed)) < 0.06

    def test_the_three_policies_do_not_draw_the_same_partition(self) -> None:
        law = bounded_cv_laws.fold_policy_dgp()
        frame, _ = law.sample(300, seed=2)
        assignments = {}
        for cell in bounded_cv_laws.fold_policy_cells(CANONICAL_CVTMLE):
            estimator = bounded_cv_laws.fold_policy_estimator(g_bounds=CV_G_BOUNDS)(cell)()
            result = estimator.fit(frame, outcome="Y", treatment="A").single()
            assignments[cell.cell] = result.nuisance.folds.assignment.copy()
        for left in bounded_cv_laws.FOLD_POLICIES:
            for right in bounded_cv_laws.FOLD_POLICIES:
                if left < right:
                    assert not np.array_equal(assignments[left], assignments[right]), (
                        f"{left} and {right} drew the same partition, so the diagnostic "
                        f"would report one policy three times"
                    )

    def test_the_stacked_record_declares_exactly_these_three_cells(self) -> None:
        """The record spells the policies out, because it cannot import them.

        ``canonical_cvtmle`` is imported *by* ``canonical_properties``, which this module
        imports, so the study record cannot read :data:`FOLD_POLICIES` without a cycle. It
        therefore carries the three names as literals, and a literal that drifted from the
        cells the study runs would declare a summary the study never writes.
        """
        assert CANONICAL_CVTMLE.property_cells[property_verdicts.FOLD_POLICY_FAMILY] == tuple(
            bounded_cv_laws.FOLD_POLICIES
        )

    def test_an_unknown_policy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="policy must be one of"):
            bounded_cv_laws.FoldPolicyTMLE(
                "stratified_on_a_hunch",
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(),
            )


def _fold_policy_rows(coverages: dict[str, float], replicates: int = 200) -> pd.DataFrame:
    """Synthetic replication rows for the three policies, paired on ``replicate``."""
    rng = np.random.default_rng(4)
    frames = []
    for cell, coverage in coverages.items():
        covered = (rng.random(replicates) < coverage).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "property": property_verdicts.FOLD_POLICY_FAMILY,
                    "cell": cell,
                    "role": "diagnostic",
                    "replicate": np.arange(replicates),
                    "n": bounded_cv_laws.FOLD_POLICY_N,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.5,
                    "estimate": 0.5 + rng.normal(scale=0.05, size=replicates),
                    "std_error": 0.05,
                    "covered": covered,
                    "rejected": 0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestTheFoldPolicyDiagnostics:
    """A reported family publishes numbers, a role that says so, and no verdict."""

    @staticmethod
    def _summary(rows: pd.DataFrame) -> pd.DataFrame:
        summary = summarize_cells(rows, margin=0.25, confidence_level=0.99, alpha=0.05)
        for column in ("coverage_gain_ci_lower", "coverage_gain_ci_upper"):
            summary[column] = np.nan
        summary["passed"] = False
        summary["property_passed"] = pd.Series(
            [None] * len(summary), dtype=object, index=summary.index
        )
        return summary

    def test_every_arm_publishes_a_coverage_interval_and_no_verdict(self) -> None:
        rows = _fold_policy_rows(
            {
                "unstratified": 0.95,
                "treatment_stratified": 0.94,
                "treatment_outcome_stratified": 0.93,
            }
        )
        summary = self._summary(rows)
        property_verdicts.fold_policy_diagnostics(summary, rows, CANONICAL_CVTMLE)
        assert bool(summary["passed"].all())
        assert bool(summary["property_passed"].all())
        assert summary["coverage_ci_lower"].notna().all()
        reference = summary["cell"] == property_verdicts.FOLD_POLICY_REFERENCE_CELL
        assert bool(summary.loc[reference, "coverage_gain_ci_lower"].isna().all())
        assert bool(summary.loc[~reference, "coverage_gain_ci_lower"].notna().all())

    def test_the_paired_difference_resolves_a_real_coverage_gap(self) -> None:
        """A degraded policy has to show up in the reported difference, or nothing does."""
        rows = _fold_policy_rows(
            {
                "unstratified": 0.95,
                "treatment_stratified": 0.95,
                "treatment_outcome_stratified": 0.60,
            }
        )
        summary = self._summary(rows)
        property_verdicts.fold_policy_diagnostics(summary, rows, CANONICAL_CVTMLE)
        degraded = summary["cell"] == "treatment_outcome_stratified"
        assert float(summary.loc[degraded, "coverage_gain_ci_upper"].iloc[0]) < 0.0

    def test_a_gated_role_on_a_reported_family_is_refused(self) -> None:
        rows = _fold_policy_rows({"unstratified": 0.95, "treatment_stratified": 0.94})
        rows.loc[rows["cell"] == "treatment_stratified", "role"] = "control"
        summary = self._summary(rows)
        with pytest.raises(ValueError, match="reported family"):
            property_verdicts.fold_policy_diagnostics(summary, rows, CANONICAL_CVTMLE)

    def test_a_missing_reference_arm_is_refused(self) -> None:
        rows = _fold_policy_rows({"treatment_stratified": 0.94})
        summary = self._summary(rows)
        with pytest.raises(ValueError, match="nothing to difference"):
            property_verdicts.fold_policy_diagnostics(summary, rows, CANONICAL_CVTMLE)

    def test_the_rendered_row_reads_as_reported_rather_than_pass(self) -> None:
        rows = _fold_policy_rows({"unstratified": 0.95, "treatment_stratified": 0.94})
        summary = self._summary(rows)
        property_verdicts.fold_policy_diagnostics(summary, rows, CANONICAL_CVTMLE)
        rendered = [document._property_verdict(row) for row in summary.itertuples()]
        assert rendered == ["reported", "reported"]
        measured = document._measured(
            next(row for row in summary.itertuples() if row.cell == "treatment_stratified")
        )
        assert "coverage" in measured and "paired coverage difference" in measured
