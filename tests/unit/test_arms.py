"""Counterfactual quantities keyed by treatment arm.

:class:`~cleverly.fluctuation.iterative.InitialFit` and
:class:`~cleverly.fluctuation.submodel.Submodel` used to name their counterfactual
arrays ``at_one`` and ``at_zero``, which made every routine that moves or subsets one
of them a two-arm routine.  They now key those arrays by the treatment level the arm
sets, and the tests here are about the two claims that change buys.

The first is *generality*: a fit or submodel carrying three arms must survive shrinking,
row-slicing, sign-taking and fluctuating with all three intact.  Nothing in the package
builds a three-arm object yet -- the estimand layer is still binary -- so these tests
construct them by hand.  That is the point: they fail the moment a helper reintroduces
an assumption that there are exactly two.

The second is *correctness of the lookup*.  A mapping keyed by arm is only an
improvement over positional fields if the code actually looks arms up rather than
indexing column 0 and column 1 and trusting the order.  The test that matters most here
is :meth:`TestArmColumns.test_the_means_follow_arm_columns_not_the_column_order`: it
permutes the arm-to-column map and checks the reported counterfactual means swap with
it.  Under the old code the columns were indexed literally, so that permutation was
invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import (
    InitialFit,
    apply_logistic,
    check_matching_arms,
    solve_fluctuation,
)
from cleverly.fluctuation.submodel import Submodel, mean_submodel, restrict, weighted_form
from tests.conftest import binary_mean_parts, binary_means

LEVELS = (0.0, 1.0, 2.0)


def _three_arm_fit(n: int = 12) -> InitialFit:
    """An initial fit over three arms, with a distinct constant per arm."""
    return InitialFit(
        np.full(n, 0.5),
        {level: np.full(n, 0.2 + 0.25 * level) for level in LEVELS},
    )


def _three_arm_submodel(n: int = 12) -> Submodel:
    """A three-column submodel, one column per arm."""
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(n, 3))
    return Submodel(
        observed,
        {level: rng.normal(size=(n, 3)) for level in LEVELS},
        ("h0", "h1", "h2"),
        "multi",
        {level: index for index, level in enumerate(LEVELS)},
    )


class TestThreeArms:
    """The containers and their helpers must not count arms."""

    def test_shrinking_preserves_every_arm(self) -> None:
        shrunk = _three_arm_fit().shrunk(0.9995)
        assert shrunk.levels == LEVELS
        for level in LEVELS:
            assert shrunk.arms[level].shape == (12,)

    def test_map_arms_touches_the_observed_array_too(self) -> None:
        """``observed`` is not an arm, but every transformation applies to it as well."""
        fit = _three_arm_fit()
        doubled = fit.map_arms(lambda values: values * 2.0)
        np.testing.assert_allclose(doubled.observed, fit.observed * 2.0)
        for level in LEVELS:
            np.testing.assert_allclose(doubled.arms[level], fit.arms[level] * 2.0)

    def test_row_slicing_keeps_the_arm_keys(self) -> None:
        rows = np.array([0, 3, 7])
        sliced = _three_arm_fit().map_arms(lambda values: values[rows])
        assert sliced.levels == LEVELS
        assert sliced.n == 3

    def test_restrict_and_weighted_form_keep_three_arms(self) -> None:
        submodel = _three_arm_submodel()
        kept = restrict(submodel, np.array([1, 2, 5]))
        assert kept.levels == LEVELS
        assert kept.arm_columns == submodel.arm_columns
        signed, weights = weighted_form(submodel, np.ones(submodel.n))
        assert signed.levels == LEVELS
        assert set(np.unique(signed.arms[2.0])) <= {-1.0, 0.0, 1.0}
        assert weights.shape == (submodel.n,)

    def test_fluctuating_moves_all_three_arms(self) -> None:
        fit, submodel = _three_arm_fit(), _three_arm_submodel()
        moved = apply_logistic(fit, submodel, np.array([0.1, -0.2, 0.05]), 0.9995)
        assert moved.levels == LEVELS
        for level in LEVELS:
            assert not np.allclose(moved.arms[level], fit.arms[level])


class TestValidation:
    def test_an_arm_shaped_unlike_the_observed_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"arm 1\.0 has shape"):
            InitialFit(np.zeros(5), {0.0: np.zeros(5), 1.0: np.zeros(4)})

    def test_an_integer_arm_key_is_refused(self) -> None:
        """``arms[1]`` and ``arms[1.0]`` are different entries, so the type is load-bearing."""
        with pytest.raises(TypeError, match="must be floats"):
            InitialFit(np.zeros(5), {0: np.zeros(5), 1: np.zeros(5)})  # type: ignore[dict-item]

    def test_a_fit_with_no_arms_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one counterfactual arm"):
            InitialFit(np.zeros(5), {})

    def test_arm_columns_must_name_an_arm_the_submodel_has(self) -> None:
        with pytest.raises(ValueError, match="not one of the submodel's arms"):
            Submodel(
                np.zeros((5, 2)),
                {0.0: np.zeros((5, 2)), 1.0: np.zeros((5, 2))},
                ("h0", "h1"),
                "mean",
                {0.0: 0, 2.0: 1},
            )

    def test_arm_columns_must_point_at_a_column_that_exists(self) -> None:
        with pytest.raises(ValueError, match=r"outside the 2 column"):
            Submodel(
                np.zeros((5, 2)),
                {0.0: np.zeros((5, 2)), 1.0: np.zeros((5, 2))},
                ("h0", "h1"),
                "mean",
                {0.0: 0, 1.0: 5},
            )

    def test_fluctuating_across_mismatched_arms_is_refused(self) -> None:
        """The failure this prevents is a silently *shorter* fit, not an exception.

        A dict comprehension over the fit's arms would simply drop an arm the submodel
        lacks, and the missing counterfactual prediction would surface later -- or never.
        """
        fit = InitialFit(np.full(6, 0.5), {0.0: np.full(6, 0.4), 1.0: np.full(6, 0.6)})
        submodel = Submodel(np.zeros((6, 1)), {0.0: np.zeros((6, 1))}, ("h",), "mean", {0.0: 0})
        with pytest.raises(ValueError, match="must describe the same ones"):
            check_matching_arms(fit, submodel)
        with pytest.raises(ValueError, match="must describe the same ones"):
            apply_logistic(fit, submodel, np.zeros(1), 0.9995)


class TestArmColumns:
    def test_the_mean_submodel_maps_each_arm_to_its_own_column(self) -> None:
        rng = np.random.default_rng(1)
        a = (rng.random(40) < 0.5).astype(float)
        submodel = mean_submodel(a, np.full(40, 0.4))
        assert submodel.arm_columns == {0.0: 0, 1.0: 1}
        np.testing.assert_allclose(submodel.column_for(0.0), submodel.observed[:, 0])
        np.testing.assert_allclose(submodel.column_for(1.0), submodel.observed[:, 1])

    def test_a_contrast_submodel_has_no_per_arm_column(self) -> None:
        """The ATT's single column targets a difference, so no column belongs to an arm."""
        from cleverly.fluctuation.submodel import att_submodel

        rng = np.random.default_rng(2)
        a = (rng.random(40) < 0.5).astype(float)
        submodel = att_submodel(a, np.full(40, 0.4), arm_fractions=float(a.mean()))
        assert submodel.arm_columns == {}
        with pytest.raises(KeyError, match="no column dedicated to arm"):
            submodel.column_for(1.0)

    def test_the_means_follow_arm_columns_not_the_column_order(self) -> None:
        """Permute the arm-to-column map and the reported means must permute with it.

        This is the test the old positional code could not have passed: it indexed
        ``observed[:, 1]`` for the treated arm unconditionally, so a submodel that put the
        treated column first was silently mis-read.
        """
        rng = np.random.default_rng(3)
        n = 200
        a = (rng.random(n) < 0.5).astype(float)
        y = rng.random(n)
        submodel = mean_submodel(a, np.full(n, 0.45))
        fit = InitialFit(np.full(n, 0.5), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.6)})

        psi_one, _, psi_zero, _ = binary_means(y, fit, submodel, np.ones(n))

        # Same arrays, but the map now claims h0 targets the treated arm and h1 the
        # control one.  Only the residual weights move, so the plug-in means are
        # unchanged and the influence curves are not.
        swapped = Submodel(
            submodel.observed,
            submodel.arms,
            submodel.names,
            submodel.group,
            {0.0: 1, 1.0: 0},
        )
        swapped_one, swapped_one_ic, swapped_zero, swapped_zero_ic = binary_means(
            y, fit, swapped, np.ones(n)
        )
        assert swapped_one == pytest.approx(psi_one)
        assert swapped_zero == pytest.approx(psi_zero)

        residual = y - fit.observed
        np.testing.assert_allclose(
            swapped_one_ic, submodel.observed[:, 0] * residual + fit.arms[1.0] - psi_one
        )
        np.testing.assert_allclose(
            swapped_zero_ic, submodel.observed[:, 1] * residual + fit.arms[0.0] - psi_zero
        )

    def test_the_decomposed_parts_use_the_same_lookup(self) -> None:
        """``counterfactual_mean_parts`` is a diagnostic and must not drift from the sum."""
        rng = np.random.default_rng(4)
        n = 150
        a = (rng.random(n) < 0.5).astype(float)
        y = rng.random(n)
        submodel = mean_submodel(a, np.full(n, 0.4))
        fit = InitialFit(np.full(n, 0.5), {0.0: np.full(n, 0.45), 1.0: np.full(n, 0.55)})
        psi_one, ic_one, psi_zero, ic_zero = binary_means(y, fit, submodel, np.ones(n))
        parts_one, parts_zero = binary_mean_parts(y, fit, submodel, np.ones(n))
        np.testing.assert_allclose(parts_one.total, ic_one, atol=1e-14, rtol=0)
        np.testing.assert_allclose(parts_zero.total, ic_zero, atol=1e-14, rtol=0)
        assert np.isfinite([psi_one, psi_zero]).all()


class TestTargetingStillSolvesTheScore:
    """The refactor is only safe if the estimating equation is untouched."""

    def test_a_two_arm_fluctuation_still_drives_the_score_to_zero(self) -> None:
        rng = np.random.default_rng(5)
        n = 600
        w = rng.normal(size=n)
        g1 = 1.0 / (1.0 + np.exp(-0.7 * w))
        a = (rng.random(n) < g1).astype(float)
        y = (rng.random(n) < 0.3 + 0.3 * a).astype(float)
        submodel = mean_submodel(a, g1)
        fit = InitialFit(np.full(n, 0.5), {0.0: np.full(n, 0.5), 1.0: np.full(n, 0.5)})
        fluctuation = solve_fluctuation(y, fit, submodel, np.ones(n))
        assert fluctuation.converged
        assert fluctuation.relative_score_norm < 1e-10
        assert fluctuation.targeted.levels == (0.0, 1.0)
