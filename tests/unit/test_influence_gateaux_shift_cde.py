"""A coarsened shift's influence curve against a numerically differentiated one.

The non-circular check, one axis further out than
``tests/unit/test_influence_gateaux_shift.py``: ``src/`` builds the curve from a clever
covariate that now divides by ``pi`` and ``q_z`` as well as by the density ratio, while
``tests/discrete_law_shift_cde.py`` writes the parameter down longhand and differentiates
it by a complex step.  A wrong composition used *consistently* -- in targeting and in
inference alike -- would solve its own score equation and pass a score check; it cannot
pass this.

Nothing here runs the targeting step, for the reason
``tests/unit/test_influence_gateaux_shift.py`` gives: the law is realised exactly by the
sample and the nuisances are the true ones, so the score is already zero and
``Qbar* == Qbar``.  There is a second reason on this axis, and it is worth stating because
it also bounds what this module *can* see.  The estimator's mechanism is a conditional
density fitted by binning, and ``fit_conditional_density`` chooses equal-mass quantile
edges from the sample -- on four tied doses that is not the unit-width partition this law
is stated on, so there is no end-to-end run that is exact.  ``tests/unit/test_shift_fit.py``
pins what ``fit_nuisances`` produces structurally instead.

**What this module cannot see.**  At ``epsilon = 0`` the reported curve reads the *observed*
block of the covariate and the untargeted ``Qbar`` at the shifted dose, so a mechanism
evaluated at the wrong dose in a *counterfactual* block moves nothing here.  That mutation
is checked by ``tests/unit/test_shift_submodel.py`` structurally and by
``tests/unit/test_shift_fit.py`` through a plug-in with ``epsilon != 0``.  Do not add a
control here claiming to catch it without watching it fail first.
"""

from __future__ import annotations

import warnings
from dataclasses import replace as dc_replace

import numpy as np
import pytest

import tests.discrete_law_shift_cde as law
from cleverly.data import CausalData
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import shift_means
from cleverly.interventions import Shift, ShiftSet
from cleverly.learners.density import ConditionalDensity

MEANS = ("ey_shift[natural course]", "ey_shift[+1]", "ey_shift[+1 (cap 2)]")
CONTRASTS = (
    "ate_shift[+1 vs natural course]",
    "ate_shift[+1 (cap 2) vs natural course]",
)

#: ``None`` is the parameter a ``delta=``-only fit reports; ``0`` and ``1`` are the
#: controlled direct effects under the policy.
LEVELS = (None, 0, 1)

#: The declared policies, in the order their codes run: 0 natural course, 1 the capped
#: shift, 2 the tightly capped one.
SHIFTS = (
    Shift(0.0, cap=law.CAP, name="natural course"),
    Shift(law.DELTA, cap=law.CAP, name="+1"),
    Shift(law.DELTA, cap=law.CAP_TIGHT, name="+1 (cap 2)"),
)


class _Pieces:
    """The law's true nuisances, assembled into what ``shift_means`` consumes."""

    def __init__(self, level: int | None) -> None:
        frame = law.frame()
        self.covariate = frame["W"].to_numpy().astype(int)
        self.dose = frame["A"].to_numpy(dtype=float)
        self.intermediate = frame["Z"].to_numpy(dtype=float).astype(int)
        self.observed = frame["Delta"].to_numpy(dtype=float).astype(bool)
        # Left as NaN where the outcome was not recorded: a curve that dropped the Delta
        # mask would come back NaN rather than merely wrong.
        self.outcome = frame["Y"].to_numpy(dtype=float)
        index = np.rint(self.dose).astype(int)

        density = ConditionalDensity(law.G_EXACT[self.covariate], law.EDGES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = CausalData.from_arrays(
                self.outcome,
                self.dose,
                self.covariate.reshape(-1, 1).astype(float),
                treatment_kind="continuous",
                delta=self.observed.astype(float),
            )
            self.shifts = ShiftSet.evaluate(SHIFTS, data, density)

        maps = [np.asarray(law.POLICIES[name]) for name in self.shifts.labels.values()]
        # (n, S + 1): the dose each block is evaluated at -- the row's own, then each
        # policy's. The order is ShiftSet.design's first axis.
        at = np.column_stack([index] + [mapping[index] for mapping in maps])
        self.evaluated_at = at

        self.missingness = law.PI_EXACT[self.covariate[:, None], at]
        if level is None:
            self.density_z = None
            self.selection = None
            qbar = law.QBAR_MARGINAL_EXACT
        else:
            probability = law.QZ_EXACT[self.covariate[:, None], at]
            self.density_z = probability if level == 1 else 1.0 - probability
            self.selection = (self.intermediate == level).astype(float)
            qbar = law.QBAR_EXACT[:, :, level]

        self.initial = InitialFit(
            qbar[self.covariate, index],
            {
                float(code): qbar[self.covariate, mapping[index]]
                for code, mapping in zip(self.shifts.codes, maps, strict=True)
            },
        )
        self.submodel = submodel_for(
            "mtp",
            self.dose,
            np.zeros((self.dose.size, 0)),
            arms=(),
            shifts=self.shifts.design,
            missingness=self.missingness,
            intermediate_density=self.density_z,
            selection=self.selection,
        )

    def means(self) -> dict[float, object]:
        return shift_means(
            self.outcome,
            self.initial,
            self.submodel,
            np.ones(self.outcome.size),
            self.observed,
        )

    def code_of(self, name: str) -> float:
        label = name[len("ey_shift[") : -1]
        return float(list(self.shifts.labels.values()).index(label))


class TestThePremisesHold:
    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        for level in LEVELS:
            for name in MEANS + CONTRASTS:
                total = float((law.PROBS.reshape(-1) * law.eif(name, level)).sum())
                assert total == pytest.approx(0.0, abs=1e-12)

    def test_the_sample_realises_the_law(self) -> None:
        frame = law.frame()
        assert len(frame) == law.N
        for w in range(3):
            rows = frame[frame["W"] == float(w)]
            counts = np.array([float((rows["A"] == dose).mean()) for dose in law.DOSES])
            np.testing.assert_allclose(counts, law.G[w], atol=1e-15)

    def test_the_missingness_is_severe_and_moves_with_the_dose(self) -> None:
        # Both halves are load-bearing. Half the sample has no outcome, so a complete-case
        # analysis has something to get wrong; and pi varies across the doses, so pi at the
        # assigned dose is a different number from pi at the observed one.
        assert float(law.frame()["Delta"].mean()) == pytest.approx(0.5, abs=1e-12)
        spread = law.PI_EXACT.max(axis=1) - law.PI_EXACT.min(axis=1)
        assert float(spread.min()) > 0.25

    def test_the_covariate_is_finite_and_the_mechanisms_are_off_their_bounds(self) -> None:
        pieces = _Pieces(1)
        assert np.all(np.isfinite(pieces.submodel.observed))
        assert float(law.PI_EXACT.min()) > 0.01
        assert float(law.QZ_EXACT.min()) > 0.01
        assert float(1.0 - law.QZ_EXACT.max()) > 0.01


class TestTheInfluenceCurveIsTheEfficientOne:
    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("name", MEANS)
    def test_it_matches_the_numerical_gateaux_derivative(
        self, name: str, level: int | None
    ) -> None:
        pieces = _Pieces(level)
        means = pieces.means()
        reported = np.asarray(means[pieces.code_of(name)].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name, level), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("name", MEANS)
    def test_the_point_estimate_is_the_functional(self, name: str, level: int | None) -> None:
        pieces = _Pieces(level)
        means = pieces.means()
        assert means[pieces.code_of(name)].psi == pytest.approx(law.TRUTH[level][name], abs=1e-12)

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_contrast_is_the_difference_of_the_two_curves(self, level: int | None) -> None:
        pieces = _Pieces(level)
        means = pieces.means()
        curve = means[1.0].influence_curve - means[0.0].influence_curve
        expected = law.eif("ate_shift[+1 vs natural course]", level)
        np.testing.assert_allclose(curve[law.first_row_of()], expected, atol=1e-14, rtol=0)

    @pytest.mark.parametrize("level", LEVELS)
    def test_an_unrecorded_outcome_contributes_the_plug_in_alone(self, level: int | None) -> None:
        """The claim the ``Delta`` factor is entirely responsible for.

        A row with no recorded outcome contributes a genuine zero to the residual term, so
        its influence curve must be exactly ``Qbar(d(A, W), W) - psi``.  Twenty-four of the
        seventy-two support points are such rows, and no shift test in the suite reached
        them before this module.
        """
        pieces = _Pieces(level)
        means = pieces.means()
        code = pieces.code_of("ey_shift[+1]")
        curve = np.asarray(means[code].influence_curve)
        plug_in = pieces.initial.arms[code] - means[code].psi
        missing = ~pieces.observed
        np.testing.assert_allclose(curve[missing], plug_in[missing], atol=1e-14, rtol=0)
        assert int(missing.sum()) == law.N // 2

    def test_the_natural_course_is_the_mar_identified_mean_not_the_complete_case_one(
        self,
    ) -> None:
        """With ``delta=`` the natural course is no longer ``mean(Y)`` over recorded rows.

        Without missing outcomes the identity ``psi(natural course) == mean(Y)`` is the
        canary that the covariate is identically one under the identity policy.  Here the
        right answer is the MAR-identified mean, and the complete-case mean is a different
        number -- so the canary has to be restated rather than kept.
        """
        pieces = _Pieces(None)
        psi = pieces.means()[0.0].psi
        assert psi == pytest.approx(law.TRUTH[None]["ey_shift[natural course]"], abs=1e-12)
        complete_case = float(np.nanmean(pieces.outcome[pieces.observed]))
        assert abs(psi - complete_case) > 1e-2


class TestTheNegativeControls:
    """Each mutation must move the curve, or the match above proves nothing."""

    def test_dropping_the_missingness_factor_breaks_the_match(self) -> None:
        pieces = _Pieces(None)
        naive = submodel_for(
            "mtp",
            pieces.dose,
            np.zeros((pieces.dose.size, 0)),
            arms=(),
            shifts=pieces.shifts.design,
        )
        means = shift_means(
            pieces.outcome, pieces.initial, naive, np.ones(pieces.outcome.size), pieces.observed
        )
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", None)))
        assert gap > 1e-2, "a covariate with no 1/pi must not match the coarsened EIF"

    def test_dropping_the_intermediate_factor_breaks_the_match(self) -> None:
        pieces = _Pieces(1)
        without = submodel_for(
            "mtp",
            pieces.dose,
            np.zeros((pieces.dose.size, 0)),
            arms=(),
            shifts=pieces.shifts.design,
            missingness=pieces.missingness,
            selection=pieces.selection,
        )
        means = shift_means(
            pieces.outcome, pieces.initial, without, np.ones(pieces.outcome.size), pieces.observed
        )
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", 1)))
        assert gap > 1e-2

    def test_dropping_the_selection_indicator_breaks_the_match(self) -> None:
        pieces = _Pieces(0)
        without = submodel_for(
            "mtp",
            pieces.dose,
            np.zeros((pieces.dose.size, 0)),
            arms=(),
            shifts=pieces.shifts.design,
            missingness=pieces.missingness,
            intermediate_density=pieces.density_z,
        )
        means = shift_means(
            pieces.outcome, pieces.initial, without, np.ones(pieces.outcome.size), pieces.observed
        )
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", 0)))
        assert gap > 1e-2

    def test_dropping_the_delta_mask_breaks_the_match(self) -> None:
        pieces = _Pieces(None)
        means = shift_means(
            pieces.outcome, pieces.initial, pieces.submodel, np.ones(pieces.outcome.size), None
        )
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        assert not np.all(np.isfinite(reported)), "an unmasked NaN outcome must show"

    @pytest.mark.parametrize("level", (0, 1))
    def test_the_other_level_is_a_different_parameter(self, level: int) -> None:
        # The controlled direct effect changes sign between the levels on this law, so
        # confusing them does not perturb the answer -- it inverts it.
        other = 1 - level
        pieces = _Pieces(level)
        means = pieces.means()
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", other)))
        assert gap > 1e-2

    def test_leaving_z_alone_is_a_different_parameter_again(self) -> None:
        pieces = _Pieces(None)
        means = pieces.means()
        for level in (0, 1):
            reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
            gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", level)))
            assert gap > 1e-2

    def test_the_complete_case_functional_is_a_different_number(self) -> None:
        for level in LEVELS:
            for name in MEANS:
                gap = abs(
                    float(law.observed_only_functional(law.PROBS, name, level))
                    - law.TRUTH[level][name]
                )
                assert gap > 1e-2, f"{name} at level {level} needs a complete-case gap"

    def test_the_induced_regime_has_the_same_mean_and_must_not_share_the_curve(self) -> None:
        # Carried across from tests/unit/test_influence_gateaux_shift.py: coarsening the
        # outcome does not touch the argument, and the means agreeing is exactly what makes
        # the wrong turn tempting.
        for level in LEVELS:
            regime = float(law.induced_regime_functional(law.PROBS, "+1", level))
            assert regime == pytest.approx(law.TRUTH[level]["ey_shift[+1]"], abs=1e-12)

    def test_scaling_the_clever_covariate_breaks_the_match(self) -> None:
        pieces = _Pieces(1)
        scaled = dc_replace(
            pieces.submodel,
            observed=pieces.submodel.observed * 1.05,
            arms={code: values * 1.05 for code, values in pieces.submodel.arms.items()},
        )
        means = shift_means(
            pieces.outcome, pieces.initial, scaled, np.ones(pieces.outcome.size), pieces.observed
        )
        reported = np.asarray(means[1.0].influence_curve)[law.first_row_of()]
        gap = np.max(np.abs(reported - law.eif("ey_shift[+1]", 1)))
        assert gap > 1e-2
