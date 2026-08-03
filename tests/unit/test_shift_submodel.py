"""The shift clever covariate, checked against values written out by hand.

Everything here is exact.  The clever covariate for a modified treatment policy is a
ratio of one density at two points, so on a law whose density is known it has a closed
form, and the three degenerate cases the module docstring claims (no shift, an
unreachable shift, and a four-dose example) are arithmetic rather than approximation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import DataError, PositivityWarning
from cleverly.fluctuation.submodel import submodel_for
from cleverly.interventions import Shift, ShiftSet, check_shift_support
from cleverly.learners.density import ConditionalDensity

#: Doses 0, 1, 2, 3 with edges at the half-integers, so every bin is one wide and the
#: "density" is the probability mass function itself.  That is what makes the expected
#: clever covariate a ratio of two numbers a reader can check.
EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
G = np.array([0.4, 0.3, 0.2, 0.1])
N = 40


def _setup() -> tuple[CausalData, ConditionalDensity, np.ndarray]:
    rng = np.random.default_rng(0)
    treatment = np.tile(np.arange(4.0), N // 4)
    covariates = rng.normal(size=(N, 1))
    outcome = rng.binomial(1, 0.5, N).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(outcome, treatment, covariates, treatment_kind="continuous")
    return data, ConditionalDensity(np.tile(G, (N, 1)), EDGES), treatment


def _shifts(*shifts: Shift) -> ShiftSet:
    data, density, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ShiftSet.evaluate(shifts, data, density)


class TestTheCleverCovariate:
    def test_the_four_dose_case_matches_the_hand_computation(self) -> None:
        # h(a) = g(a - 1) / g(a) + 1{a > cap - 1}, with cap = 3:
        #   a = 0 -> g(-1)/g(0) = 0, and 0 is not > 2
        #   a = 1 -> g(0)/g(1)
        #   a = 2 -> g(1)/g(2)
        #   a = 3 -> g(2)/g(3), plus 1 because 3 > 2 and the cap holds that row back
        shifts = _shifts(Shift(1.0, cap=3.0))
        expected = [0.0, G[0] / G[1], G[1] / G[2], G[2] / G[3] + 1.0]
        np.testing.assert_allclose(shifts.ratio[:4, 0], expected, rtol=0, atol=1e-14)

    def test_the_natural_course_is_identically_one(self) -> None:
        # delta = 0 makes h = g(a)/g(a) = 1, so the influence curve collapses to Y - psi,
        # which is the influence curve of E[Y]. The estimand really is the mean outcome.
        np.testing.assert_allclose(_shifts(Shift(0.0, cap=3.0)).ratio, 1.0, rtol=0, atol=1e-14)

    def test_a_shift_nobody_can_take_is_identically_one(self) -> None:
        # Every row is capped back to its own dose, so the policy is the natural course
        # under another name and must report the same covariate.
        np.testing.assert_allclose(_shifts(Shift(99.0, cap=3.0)).ratio, 1.0, rtol=0, atol=1e-14)

    def test_a_tight_cap_removes_the_ratio_term_above_it(self) -> None:
        # The regression test for a bug the loose cap could not see. A unit can only have
        # been *shifted* to dose a if the shift from a - delta was not itself held back,
        # which needs a <= cap; above the cap the only way to be at a is to have stayed,
        # so the ratio drops out and the indicator is the whole covariate.
        #
        # With cap = 2 the policy is 0->1, 1->2, 2->2, 3->3, so the induced density is
        # g^d = (0, g0, g1 + g2, g3) and h = g^d / g:
        shifts = _shifts(Shift(1.0, cap=2.0))
        expected = [0.0, G[0] / G[1], (G[1] + G[2]) / G[2], 1.0]
        np.testing.assert_allclose(shifts.ratio[:4, 0], expected, rtol=0, atol=1e-14)

    def test_an_uncapped_shift_drops_the_indicator(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uncapped = _shifts(Shift(1.0, cap=None))
        # Same ratio as the capped version everywhere except the top dose, where the
        # capped one adds 1 and this one does not.
        expected = [0.0, G[0] / G[1], G[1] / G[2], G[2] / G[3]]
        np.testing.assert_allclose(uncapped.ratio[:4, 0], expected, rtol=0, atol=1e-14)

    def test_a_row_with_no_estimated_density_contributes_nothing(self) -> None:
        # A zero denominator is a support failure, not a large weight. Reporting infinity
        # would put a NaN through the Newton solve; zero is the honest value and
        # check_shift_support counts the row.
        data, _, treatment = _setup()
        holed = np.tile(np.array([0.5, 0.5, 0.0, 0.0]), (N, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shifts = ShiftSet.evaluate(
                (Shift(1.0, cap=3.0),), data, ConditionalDensity(holed, EDGES)
            )
            support = check_shift_support(shifts, ConditionalDensity(holed, EDGES), treatment)
        assert np.all(np.isfinite(shifts.ratio))
        assert support["+1"].unsupported == N // 2


class TestTheSubmodel:
    def _submodel(self, *shifts: Shift):  # type: ignore[no-untyped-def]
        data, _, _ = _setup()
        built = _shifts(*shifts)
        return built, submodel_for(
            "mtp", data.treatment, np.zeros((N, 0)), arms=(), shifts=built.design
        )

    def test_a_column_targets_one_shift(self) -> None:
        # Unlike the regime group, where a column targets a distribution over the arms
        # and no column belongs to one parameter, a shift column really does target one
        # mean -- so arm_columns is populated and column_for can answer.
        _, submodel = self._submodel(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        assert submodel.group == "mtp"
        assert submodel.arm_columns == {0.0: 0, 1.0: 1}
        np.testing.assert_array_equal(submodel.column_for(1.0), submodel.observed[:, 1])

    def test_the_natural_course_leaves_every_row_where_it_was(self) -> None:
        # d_0(A, W) = A, so the covariate "at the shifted dose" is the covariate at the
        # observed one -- the same array, not merely a close one.
        _, submodel = self._submodel(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        np.testing.assert_array_equal(submodel.arms[0.0], submodel.observed)

    def test_a_missing_shifts_argument_says_how_to_build_one(self) -> None:
        data, _, _ = _setup()
        with pytest.raises(ValueError, match="needs shifts="):
            submodel_for("mtp", data.treatment, np.zeros((N, 0)), arms=())

    def test_a_mis_shaped_shifts_argument_is_refused(self) -> None:
        data, _, _ = _setup()
        with pytest.raises(ValueError, match=r"shape \(n, S \+ 1, S\)"):
            submodel_for(
                "mtp", data.treatment, np.zeros((N, 0)), arms=(), shifts=np.zeros((N, 2, 2))
            )

    def test_a_fit_with_no_extra_mechanism_is_the_bare_ratio(self) -> None:
        # The regression surface: `_arm_matrix` returns ones when the argument is absent,
        # so a shift fit that declares neither delta= nor intermediate= divides by nothing
        # and every array is the one it was before either was supported.
        built, submodel = self._submodel(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        np.testing.assert_array_equal(submodel.observed, built.design[:, 0, :])
        for index in (0, 1):
            np.testing.assert_array_equal(
                submodel.arms[float(index)], built.design[:, index + 1, :]
            )

    def test_each_block_divides_by_the_mechanism_at_its_own_dose(self) -> None:
        """The claim no Gateaux check on an exact law can make.

        At ``epsilon = 0`` the reported curve reads the *observed* block and the untargeted
        ``Qbar``, so a mechanism evaluated at the wrong dose in a counterfactual block
        moves no reported number there at all --
        ``tests/unit/test_influence_gateaux_shift_cde.py`` says so in its docstring, and the
        mutation was applied and seen to pass.  This is where it is caught instead: block
        ``j`` of the covariate must divide by column ``j`` of each mechanism, longhand.
        """
        data, _, _ = _setup()
        built = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        rng = np.random.default_rng(7)
        pi = rng.uniform(0.2, 0.9, size=(N, 3))
        qz = rng.uniform(0.2, 0.9, size=(N, 3))
        submodel = submodel_for(
            "mtp",
            data.treatment,
            np.zeros((N, 0)),
            arms=(),
            shifts=built.design,
            missingness=pi,
            intermediate_density=qz,
        )
        np.testing.assert_allclose(
            submodel.observed,
            built.design[:, 0, :] / (pi[:, 0] * qz[:, 0])[:, None],
            atol=1e-14,
            rtol=0,
        )
        for index in (0, 1):
            np.testing.assert_allclose(
                submodel.arms[float(index)],
                built.design[:, index + 1, :] / (pi[:, index + 1] * qz[:, index + 1])[:, None],
                atol=1e-14,
                rtol=0,
            )
        # And the mutation that reuses column 0 everywhere is far away, so the assertions
        # above are not passing by coincidence on a mechanism that barely moves.
        naive = built.design[:, 1, :] / (pi[:, 0] * qz[:, 0])[:, None]
        assert np.max(np.abs(submodel.arms[0.0] - naive)) > 1e-2

    def test_the_selection_indicator_stays_off_the_counterfactual_blocks(self) -> None:
        # mean_submodel's rule, and for its reason: the counterfactual blocks are already
        # evaluated at Z = z by construction, so zeroing them would leave every row whose
        # intermediate took the other level with an un-updated prediction in the plug-in.
        data, _, _ = _setup()
        built = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        selection = np.tile([1.0, 0.0], N // 2)
        submodel = submodel_for(
            "mtp",
            data.treatment,
            np.zeros((N, 0)),
            arms=(),
            shifts=built.design,
            selection=selection,
        )
        np.testing.assert_array_equal(submodel.observed, built.design[:, 0, :] * selection[:, None])
        for index in (0, 1):
            np.testing.assert_array_equal(
                submodel.arms[float(index)], built.design[:, index + 1, :]
            )
        assert np.any(submodel.arms[1.0][selection == 0.0] != 0.0)

    def test_a_mis_shaped_mechanism_names_the_block_count(self) -> None:
        data, _, _ = _setup()
        built = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        with pytest.raises(ValueError, match=r"missingness probabilities must have shape"):
            submodel_for(
                "mtp",
                data.treatment,
                np.zeros((N, 0)),
                arms=(),
                shifts=built.design,
                missingness=np.full((N, 2), 0.5),
            )

    def test_the_arm_builders_accept_and_ignore_it(self) -> None:
        # The registry dispatches on the group name alone, so every builder takes the
        # same keyword-only signature. A builder that targets arms must tolerate shifts=.
        rng = np.random.default_rng(1)
        treatment = rng.binomial(1, 0.5, N).astype(float)
        propensity = np.column_stack([np.full(N, 0.5), np.full(N, 0.5)])
        submodel = submodel_for(
            "mean", treatment, propensity, arms=(0.0, 1.0), shifts=np.zeros((N, 2, 1))
        )
        assert submodel.group == "mean"


class TestTheShiftSet:
    def test_it_keys_parameters_by_code_not_by_delta(self) -> None:
        # The regime path's convention: codes 0..S-1, labels carried separately. A float
        # key derived from a user-supplied delta is a worse dictionary key than an ordinal.
        shifts = _shifts(Shift(0.0, cap=3.0), Shift(0.5, cap=3.0))
        assert shifts.codes == (0.0, 1.0)
        assert shifts.labels == {0.0: "natural course", 1.0: "+0.5"}
        assert shifts.label(1.0) == "+0.5"

    def test_a_subset_slices_every_array(self) -> None:
        shifts = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        subset = shifts.subset(np.arange(10))
        assert subset.n == 10
        assert subset.ratio_at.shape == (10, 2, 2)
        np.testing.assert_array_equal(subset.ratio, shifts.ratio[:10])

    def test_the_design_stacks_the_observed_covariate_first(self) -> None:
        shifts = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        assert shifts.design.shape == (N, 3, 2)
        np.testing.assert_array_equal(shifts.design[:, 0, :], shifts.ratio)
        np.testing.assert_array_equal(shifts.design[:, 1:, :], shifts.ratio_at)

    def test_a_reference_names_a_declared_shift(self) -> None:
        shifts = _shifts(Shift(0.0, cap=3.0), Shift(1.0, cap=3.0))
        assert shifts.reference == 0.0
        data, density, _ = _setup()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chosen = ShiftSet.evaluate(
                (Shift(0.0, cap=3.0), Shift(1.0, cap=3.0)), data, density, reference="+1"
            )
        assert chosen.reference == 1.0
        with pytest.raises(DataError, match="is not one of the shifts"):
            ShiftSet.evaluate((Shift(0.0, cap=3.0),), data, density, reference="nope")


class TestTheRefusalsAndWarnings:
    @pytest.mark.parametrize("bad", [np.inf, np.nan])
    def test_a_non_finite_shift_is_refused(self, bad: float) -> None:
        with pytest.raises(DataError, match="must be finite"):
            Shift(bad, cap=1.0)
        with pytest.raises(DataError, match="must be finite or None"):
            Shift(1.0, cap=bad)

    def test_an_uncapped_shift_that_extrapolates_warns(self) -> None:
        data, density, _ = _setup()
        with pytest.warns(PositivityWarning, match="above the largest one observed"):
            ShiftSet.evaluate((Shift(1.0, cap=None),), data, density)

    def test_at_least_one_shift_is_required(self) -> None:
        data, density, _ = _setup()
        with pytest.raises(DataError, match="at least one shift"):
            ShiftSet.evaluate((), data, density)

    def test_a_default_name_says_which_policy_it_is(self) -> None:
        assert Shift(0.0, cap=None).name == "natural course"
        assert Shift(0.5, cap=None).name == "+0.5"
        assert Shift(-1.0, cap=None).name == "-1"
