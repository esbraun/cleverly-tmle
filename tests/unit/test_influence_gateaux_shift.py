"""The shift influence curve against a numerically differentiated one.

The non-circular check: ``src/`` builds the influence curve from its clever covariate and
its plug-in, while ``tests/discrete_law_shift.py`` writes the parameter down longhand and
differentiates it by a complex step.  A wrong clever covariate used *consistently* -- in
targeting and in inference alike -- would solve its own score equation and pass a score
check; it cannot pass this.

Nothing here runs the targeting step.  The law is realised exactly by the sample and the
nuisances are the true ones, so the score is already zero and ``Qbar* == Qbar``; building
the ``InitialFit`` and ``Submodel`` by hand keeps the test about the influence curve rather
than about the Newton solver.  ``tests/unit/test_remainder_regime.py`` uses the same
discipline for the same reason.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tests.discrete_law_shift as law
from cleverly.data import CausalData
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import regime_means, shift_means
from cleverly.interventions import Shift, ShiftSet
from cleverly.learners.density import ConditionalDensity

MEANS = ("ey_shift[natural course]", "ey_shift[+1]", "ey_shift[+1 (cap 2)]")
CONTRASTS = (
    "ate_shift[+1 vs natural course]",
    "ate_shift[+1 (cap 2) vs natural course]",
)

#: The declared policies, in the order their codes run: 0 natural course, 1 the capped
#: shift, 2 the tightly capped one.  Matches ``law.NAMES``.
SHIFTS = (
    Shift(0.0, cap=law.CAP, name="natural course"),
    Shift(law.DELTA, cap=law.CAP, name="+1"),
    Shift(law.DELTA, cap=law.CAP_TIGHT, name="+1 (cap 2)"),
)


def _pieces() -> tuple[np.ndarray, InitialFit, object, ShiftSet]:
    """The law's true nuisances, assembled into what ``shift_means`` consumes."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    dose = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)
    index = np.rint(dose).astype(int)

    density = ConditionalDensity(law.G[covariate], law.EDGES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(
            outcome, dose, covariate.reshape(-1, 1).astype(float), treatment_kind="continuous"
        )
        shifts = ShiftSet.evaluate(SHIFTS, data, density)

    initial = InitialFit(
        law.Q[covariate, index],
        {
            float(code): law.Q[covariate, np.asarray(law.POLICIES[name])[index]]
            for code, name in shifts.labels.items()
        },
    )
    submodel = submodel_for("mtp", dose, np.zeros((dose.size, 0)), arms=(), shifts=shifts.design)
    return outcome, initial, submodel, shifts


def _weighted_pieces(weights: np.ndarray) -> tuple[np.ndarray, InitialFit, object, ShiftSet]:
    """The *tilted* law's nuisances, which is what a weighted fit's converge to.

    A weight tilts the population, so the density a weighted fit learns is
    :math:`g_w(a \\mid W)` and the regression is :math:`\\bar Q_w`; neither is a factor in
    the clever covariate, which is the whole content of the claim being checked.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    dose = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)
    index = np.rint(dose).astype(int)
    g_w, q_w = law.tilted_nuisances(weights)

    density = ConditionalDensity(g_w[covariate], law.EDGES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(
            outcome, dose, covariate.reshape(-1, 1).astype(float), treatment_kind="continuous"
        )
        shifts = ShiftSet.evaluate(SHIFTS, data, density)

    initial = InitialFit(
        q_w[covariate, index],
        {
            float(code): q_w[covariate, np.asarray(law.POLICIES[name])[index]]
            for code, name in shifts.labels.items()
        },
    )
    submodel = submodel_for("mtp", dose, np.zeros((dose.size, 0)), arms=(), shifts=shifts.design)
    return outcome, initial, submodel, shifts


class TestAWeightedShiftFit:
    """``weights=`` on a dose, which used to be refused for a reason that was wrong.

    The refusal said a weight "puts a further per-arm factor in the clever covariate's
    denominator".  It does not: a weight tilts the *population*, so the estimand is the
    shift parameter at ``dP_w``, ``h`` divides by the density ratio and nothing else, and
    putting ``w`` there would divide the estimating equation by the very tilt it applies.
    The roadmap's *Refusals worth lifting* item 3 established that for ``LTMLE``; these are
    the same statements one node down.
    """

    #: A tilt that depends on the covariate and on the dose -- not on ``Y``, which a real
    #: weight cannot see -- and is far from constant, so the two laws genuinely differ.
    WEIGHTS = law.cell_weights(lambda w, a, y: 0.5 + 0.5 * w + 0.25 * a)

    def test_the_weighted_estimand_is_the_tilted_laws(self) -> None:
        weights = law.row_weights(self.WEIGHTS)
        outcome, initial, submodel, _ = _weighted_pieces(self.WEIGHTS)
        means = shift_means(outcome, initial, submodel, weights / weights.mean())
        expected = float(law.weighted_functional(law.PROBS, "ey_shift[+1]", self.WEIGHTS))
        assert means[1.0].psi == pytest.approx(expected, abs=1e-12)
        # ... and it is a different number from the unweighted one, or the tilt is inert.
        assert abs(expected - law.TRUTH["ey_shift[+1]"]) > 1e-2

    @pytest.mark.parametrize("name", MEANS)
    def test_the_curve_is_the_gateaux_derivative_of_the_tilted_parameter(self, name: str) -> None:
        weights = law.row_weights(self.WEIGHTS)
        outcome, initial, submodel, shifts = _weighted_pieces(self.WEIGHTS)
        code = float(list(shifts.labels.values()).index(name[len("ey_shift[") : -1]))
        means = shift_means(outcome, initial, submodel, weights / weights.mean())
        reported = np.asarray(means[code].influence_curve)[law.first_row_of()]
        expected = law.weighted_eif(name, self.WEIGHTS)
        np.testing.assert_allclose(reported, expected, atol=1e-12, rtol=0)

    def test_the_weight_is_not_a_factor_in_the_clever_covariate(self) -> None:
        """The refusal's stated reason, asserted false rather than argued against.

        The covariate is built from the density alone.  Dividing it by the weight -- what
        the old message said the parameter needed -- moves the curve away from the tilted
        law's derivative, so the two statements cannot both be right.
        """
        from dataclasses import replace as dc_replace

        weights = law.row_weights(self.WEIGHTS)
        normalised = weights / weights.mean()
        outcome, initial, submodel, _ = _weighted_pieces(self.WEIGHTS)
        divided = dc_replace(
            submodel,
            observed=submodel.observed / normalised[:, None],
            arms={code: values / normalised[:, None] for code, values in submodel.arms.items()},
        )
        means = shift_means(outcome, initial, divided, normalised)
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.weighted_eif("ey_shift[+1]", self.WEIGHTS)))
        assert gap > 1e-2

    def test_a_constant_weight_is_the_unweighted_fit(self) -> None:
        flat = law.cell_weights(lambda w, a, y: 1.0)
        outcome, initial, submodel, _ = _weighted_pieces(flat)
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        assert means[1.0].psi == pytest.approx(law.TRUTH["ey_shift[+1]"], abs=1e-12)
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif("ey_shift[+1]"), atol=1e-12, rtol=0)


class TestThePremisesHold:
    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        for name in MEANS + CONTRASTS:
            total = float((law.PROBS.reshape(-1) * law.eif(name)).sum())
            assert total == pytest.approx(0.0, abs=1e-12)

    def test_the_sample_realises_the_law(self) -> None:
        frame = law.frame()
        assert len(frame) == law.N
        for w in range(3):
            rows = frame[frame["W"] == float(w)]
            counts = np.array([float((rows["A"] == dose).mean()) for dose in law.DOSES])
            np.testing.assert_allclose(counts, law.G[w], rtol=0, atol=1e-15)


class TestTheInfluenceCurveIsTheEfficientOne:
    @pytest.mark.parametrize("name", MEANS)
    def test_it_matches_the_numerical_gateaux_derivative(self, name: str) -> None:
        outcome, initial, submodel, shifts = _pieces()
        code = float(list(shifts.labels.values()).index(name[len("ey_shift[") : -1]))
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        reported = np.asarray(means[code].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", MEANS)
    def test_the_point_estimate_is_the_functional(self, name: str) -> None:
        outcome, initial, submodel, shifts = _pieces()
        code = float(list(shifts.labels.values()).index(name[len("ey_shift[") : -1]))
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        assert means[code].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    def test_a_contrast_is_the_difference_of_the_two_curves(self) -> None:
        outcome, initial, submodel, _ = _pieces()
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        curve = means[1.0].influence_curve - means[0.0].influence_curve
        reported = curve[law.first_row_of()]
        expected = law.eif("ate_shift[+1 vs natural course]")
        np.testing.assert_allclose(reported, expected, atol=1e-14, rtol=0)

    def test_the_natural_course_reports_the_mean_outcome(self) -> None:
        # h is identically one under d = identity, so the plug-in is E[Qbar(A, W)] = E[Y].
        outcome, initial, submodel, _ = _pieces()
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        assert means[0.0].psi == pytest.approx(float(outcome.mean()), abs=1e-12)


class TestItIsNotTheRegimeThatInducesIt:
    """The correction this estimand exists to get right.

    A shift induces the density ``g^d(b | w) = sum over the preimage``, and the stochastic
    regime at that density has the *same mean* and the *same clever covariate*.  Its
    influence curve is different, because its plug-in term averages over the doses instead
    of reading the one the unit received.  Delegating ``shift_means`` to ``regime_means``
    would therefore report a standard error for a different estimator, and these three
    assertions are what would catch it.
    """

    def test_the_clever_covariates_agree_entry_for_entry(self) -> None:
        _, _, submodel, _ = _pieces()
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        index = np.rint(frame["A"].to_numpy(dtype=float)).astype(int)
        expected = law.clever_covariate(law.CAP)[covariate, index]
        np.testing.assert_allclose(submodel.observed[:, 1], expected, atol=1e-14, rtol=0)

    def test_the_induced_regime_has_the_same_mean(self) -> None:
        assert float(law.induced_regime_functional(law.PROBS, "+1")) == pytest.approx(
            law.TRUTH["ey_shift[+1]"], abs=1e-12
        )

    def test_but_a_different_influence_curve(self) -> None:
        outcome, initial, submodel, _ = _pieces()
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        dose = frame["A"].to_numpy(dtype=float)

        regime_submodel = submodel_for(
            "regime",
            dose,
            law.G[covariate],
            arms=law.DOSES,
            regimes=law.INDUCED[covariate][:, :, None],
        )
        arms = {float(a): law.Q[covariate, int(a)] for a in range(len(law.DOSES))}
        regime_initial = InitialFit(law.Q[covariate, np.rint(dose).astype(int)], arms)
        regime = regime_means(
            outcome,
            regime_initial,
            regime_submodel,
            law.INDUCED[covariate][:, :, None],
            np.ones(outcome.size),
        )[0.0]
        shift = shift_means(outcome, initial, submodel, np.ones(outcome.size))[1.0]

        assert regime.psi == pytest.approx(shift.psi, abs=1e-12), "the means must agree"
        gap = np.max(np.abs(regime.influence_curve - shift.influence_curve))
        assert gap > 1e-2, "the influence curves must not agree"

    def test_the_variance_identity_holds(self) -> None:
        # Var(D_mtp) = Var(D_regime) + Var(Qbar(d(A,W),W) - E[. | W]).  The extra term is
        # the price of an intervention that reads the natural value of treatment.
        outcome, initial, submodel, _ = _pieces()
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        dose = frame["A"].to_numpy(dtype=float)
        index = np.rint(dose).astype(int)

        shift = shift_means(outcome, initial, submodel, np.ones(outcome.size))[1.0]
        shifted_prediction = law.Q[covariate, np.asarray(law.SHIFTED)[index]]
        conditional = (law.INDUCED * law.Q).sum(axis=1)[covariate]
        extra = float(np.var(shifted_prediction - conditional))

        regime_curve = shift.influence_curve - (shifted_prediction - conditional)
        assert float(np.var(shift.influence_curve)) == pytest.approx(
            float(np.var(regime_curve)) + extra, abs=1e-12
        )


class TestTheNegativeControl:
    """A wrong clever covariate must move the influence curve, or the test proves nothing."""

    def test_scaling_the_clever_covariate_breaks_the_match(self) -> None:
        outcome, initial, submodel, _ = _pieces()
        from dataclasses import replace as dc_replace

        scaled = dc_replace(
            submodel,
            observed=submodel.observed * 1.05,
            arms={code: values * 1.05 for code, values in submodel.arms.items()},
        )
        means = shift_means(outcome, initial, scaled, np.ones(outcome.size))
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]")))
        assert gap > 1e-2, "a 5% error in the clever covariate must be visible here"

    def test_the_wrong_shift_map_breaks_the_match(self) -> None:
        # The tightly capped policy is a *different* parameter; reading its curve against
        # the other one's oracle must fail, or the label is not doing any work.
        outcome, initial, submodel, _ = _pieces()
        means = shift_means(outcome, initial, submodel, np.ones(outcome.size))
        reported = np.asarray(means[2.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]")))
        assert gap > 1e-2
