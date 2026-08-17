r"""The two extra clever covariates of doubly-robust inference, in isolation.

:mod:`tests.unit.test_remainder_drtmle` establishes *what* the extra equations remove, at
nuisances that are wrong on purpose and with every term written out longhand.  This module
checks the arrays the estimator will actually fluctuate along, and it takes its numbers from
that one by import rather than deriving them a second time -- the same arrangement
:mod:`tests.unit.test_reduced_regressions` uses, so no two modules here can drift into
agreeing with each other about something wrong.

Nothing here runs a fit.  The covariates are pure functions of a
:class:`~cleverly.estimators.reduced.ReducedSet` and a mechanism, which is what makes them
checkable against arithmetic rather than against another estimator.

**Every test below is at deliberately wrong nuisances, and that is not an accident.**  At the
truth :math:`Q_r` and :math:`g_{r,2}` are identically zero row by row, so both covariates
vanish and every assertion that could distinguish a right implementation from a wrong one
becomes ``0 == 0``.  :class:`TestTheyVanishAtTheTruth` pins that degeneracy deliberately --
it is a real property, and it is why no ``test_influence_gateaux*`` module can serve here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cleverly.estimators.reduced import ReducedSet
from cleverly.fluctuation._score import score_columns
from cleverly.fluctuation.reduced import reduced_mechanism_covariate, reduced_outcome_submodel
from tests import discrete_law as law
from tests.unit.test_remainder_drtmle import WRONG_G, WRONG_Q, _reduced

ARMS = (0.0, 1.0)

#: Wide enough not to bite. A bound that clipped ``gr1`` in the tests *about* something else
#: would make every comparison a comparison against the bound.
INERT = (1e-6, 1.0 - 1e-6)


def realised() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(covariate cell, treatment, outcome)`` of the sample the law realises exactly."""
    frame = law.frame()
    return (
        frame["W"].to_numpy().astype(int),
        frame["A"].to_numpy(dtype=float),
        frame["Y"].to_numpy(dtype=float),
    )


def reduced_at(g_hat: np.ndarray, q_hat: np.ndarray) -> ReducedSet:
    """The three reduced regressions at every *row*, from the per-cell longhand."""
    cell, _, _ = realised()
    columns = {name: [] for name in ("qr", "gr1", "gr2")}
    for arm in ARMS:
        qr, gr1, gr2 = _reduced(g_hat, q_hat, int(arm))
        for name, values in (("qr", qr), ("gr1", gr1), ("gr2", gr2)):
            columns[name].append(values[cell])
    return ReducedSet(
        qr=np.column_stack(columns["qr"]),
        gr1=np.column_stack(columns["gr1"]),
        gr2=np.column_stack(columns["gr2"]),
        arms=ARMS,
        g_bounds=INERT,
    )


class TestTheOutcomeCovariate:
    r"""Equation (10): :math:`1_a\,g_{r,2}(a|W)/g_{r,1}(a|W)`, shaped like ``mean_submodel``."""

    def test_the_observed_column_carries_the_indicator(self) -> None:
        _, treatment, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        submodel = reduced_outcome_submodel(treatment, reduced, bounds=INERT)

        for j, arm in enumerate(ARMS):
            expected = (treatment == arm) * reduced.gr2[:, j] / reduced.gr1[:, j]
            np.testing.assert_allclose(submodel.observed[:, j], expected, rtol=0, atol=1e-14)

    def test_the_counterfactual_columns_do_not(self) -> None:
        """The update is applied at the counterfactual covariate; the score is not.

        ``mean_submodel`` makes exactly this distinction, and reading ``observed`` where
        ``arms[a]`` belongs leaves every row that took the other arm un-updated in the
        plug-in -- a mistake that moves the estimate and no score equation.
        """
        _, treatment, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        submodel = reduced_outcome_submodel(treatment, reduced, bounds=INERT)

        for j, arm in enumerate(ARMS):
            ratio = reduced.gr2[:, j] / reduced.gr1[:, j]
            np.testing.assert_allclose(submodel.arms[arm][:, j], ratio, rtol=0, atol=1e-14)
            other = 1 - j
            np.testing.assert_array_equal(submodel.arms[arm][:, other], np.zeros(law.N))
            assert not np.allclose(submodel.arms[arm][:, j], submodel.observed[:, j])

    def test_it_labels_itself_with_the_group_it_is_solved_beside(self) -> None:
        """The estimand is still ``E[Y^a]``; what changed is which equations it solves."""
        _, treatment, _ = realised()
        submodel = reduced_outcome_submodel(treatment, reduced_at(WRONG_G, WRONG_Q), bounds=INERT)

        assert submodel.group == "mean"
        assert submodel.names == ("h_dr0", "h_dr1")
        assert submodel.dim == 2

    def test_one_column_still_belongs_to_one_arm(self) -> None:
        """Seam 6: ``column_for`` is what ``sensitivity/omitted_variable.py`` reads.

        This is the whole reason equation (10) is a *second* submodel rather than two more
        columns on the first -- ``drtmle``'s ``Qsteps = 2`` backfitting.  One wider submodel
        would have made a column mean something the Riesz representer does not expect.
        """
        _, treatment, _ = realised()
        submodel = reduced_outcome_submodel(treatment, reduced_at(WRONG_G, WRONG_Q), bounds=INERT)

        assert submodel.arm_columns == {0.0: 0, 1.0: 1}
        assert submodel.contrast_columns == {}
        np.testing.assert_array_equal(submodel.column_for(1.0), submodel.observed[:, 1])

    def test_the_gr1_bound_bites_here_rather_than_at_fit_time(self) -> None:
        r""":math:`g_{r,1}` is truncated at *targeting* time, unlike :math:`g_{r,2}`.

        Which matters to a reader of a truncation curve: the sweep moves this denominator
        and does **not** move ``gr2``, whose target was a quotient formed when the
        regression was fitted.  Half the extra equation responds to the sweep and half is
        flat by construction, and neither half is a bug.
        """
        _, treatment, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        loose = reduced_outcome_submodel(treatment, reduced, bounds=INERT)
        binding = reduced_outcome_submodel(treatment, reduced, bounds=(0.45, 0.55))

        assert np.max(np.abs(binding.observed - loose.observed)) > 1e-3
        # The stored array is untouched: the bound is applied on the way out.
        assert np.min(reduced.gr1) < 0.45 or np.max(reduced.gr1) > 0.55


class TestTheBivariateOutcomeCovariate:
    """van der Laan's joint ``(Q, g)`` reduction has a different score direction."""

    def test_it_is_the_pinned_drtmle_h2_at_nonzero_drift(self) -> None:
        cell, treatment, _ = realised()
        mechanism = WRONG_G[cell]
        base = reduced_at(WRONG_G, WRONG_Q)
        gr = np.column_stack([0.72 - 0.08 * cell, 0.31 + 0.09 * cell])
        reduced = replace(base, gr1=gr, gr2=np.full_like(gr, np.nan), reduction="bivariate")

        submodel = reduced_outcome_submodel(treatment, reduced, bounds=INERT, propensity=mechanism)
        armwise_g = np.column_stack([1.0 - mechanism, mechanism])
        ratio = (gr - armwise_g) / (armwise_g * gr)
        expected = np.column_stack([(treatment == arm) * ratio[:, j] for j, arm in enumerate(ARMS)])

        np.testing.assert_allclose(submodel.observed, expected, atol=1e-14, rtol=0)
        assert np.max(np.abs(expected)) > 1e-2, "the formula witness must not vanish"
        # Deliberate mutation: omitting the outer 1/g factor is not algebraically equivalent.
        mutated = np.column_stack(
            [
                (treatment == arm) * (gr[:, j] - armwise_g[:, j]) / gr[:, j]
                for j, arm in enumerate(ARMS)
            ]
        )
        assert np.max(np.abs(submodel.observed - mutated)) > 1e-2

    def test_it_requires_the_current_mechanism(self) -> None:
        _, treatment, _ = realised()
        base = reduced_at(WRONG_G, WRONG_Q)
        reduced = replace(base, gr2=np.full_like(base.gr2, np.nan), reduction="bivariate")

        with pytest.raises(ValueError, match="current propensity"):
            reduced_outcome_submodel(treatment, reduced, bounds=INERT)


class TestTheMechanismCovariate:
    r"""Equation (9): :math:`Q_r(a, W)/g^*(a|W)`, signed so one tilt solves both arms."""

    def test_each_column_is_the_reduced_outcome_regression_over_the_mechanism(self) -> None:
        cell, _, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        mechanism = WRONG_G[cell]
        covariate = reduced_mechanism_covariate(reduced, mechanism, bounds=INERT)

        np.testing.assert_allclose(
            covariate[:, 1], reduced.qr[:, 1] / mechanism, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            covariate[:, 0], -reduced.qr[:, 0] / (1.0 - mechanism), rtol=0, atol=1e-14
        )

    def test_the_signs_make_the_score_the_per_arm_equations(self) -> None:
        r"""The claim the sign convention exists for, checked rather than argued.

        :func:`~cleverly.fluctuation.mechanism.solve_mechanism` forms one residual,
        :math:`A - g^*(a_1|W)`, and equation (9) is stated once per arm against
        :math:`1_a - g^*(a|W)`.  The lower arm's residual is the negation of the upper's,
        so its column carries a minus sign -- and if it does not, that equation is solved
        with the wrong sign, which moves ``ey0`` and ``ate`` and leaves ``ey1`` exactly
        where it was.
        """
        cell, treatment, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        mechanism = WRONG_G[cell]
        weights = np.ones(law.N)
        covariate = reduced_mechanism_covariate(reduced, mechanism, bounds=INERT)

        score = score_columns(treatment, mechanism, covariate, weights, np.ones(law.N, dtype=bool))
        for j, arm in enumerate(ARMS):
            arm_mechanism = mechanism if arm == 1.0 else 1.0 - mechanism
            indicator = (treatment == arm).astype(float)
            longhand = np.mean(reduced.qr[:, j] / arm_mechanism * (indicator - arm_mechanism))
            np.testing.assert_allclose(score[j], longhand, rtol=0, atol=1e-14)

        assert np.max(np.abs(score)) > 1e-3, "a zero score would make the check vacuous"

    def test_the_lower_arm_is_bounded_by_the_complement(self) -> None:
        """``Propensity.bounded``'s two-arm rule, which the binary path is a surface for."""
        cell, _, _ = realised()
        reduced = reduced_at(WRONG_G, WRONG_Q)
        mechanism = WRONG_G[cell]
        bounds = (0.40, 0.50)
        covariate = reduced_mechanism_covariate(reduced, mechanism, bounds=bounds)

        clipped = np.clip(mechanism, *bounds)
        np.testing.assert_allclose(covariate[:, 1], reduced.qr[:, 1] / clipped, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            covariate[:, 0], -reduced.qr[:, 0] / (1.0 - clipped), rtol=0, atol=1e-14
        )

    def test_more_than_two_arms_is_refused_by_name(self) -> None:
        reduced = reduced_at(WRONG_G, WRONG_Q)
        three = ReducedSet(
            qr=np.zeros((law.N, 3)),
            gr1=np.full((law.N, 3), 0.3),
            gr2=np.zeros((law.N, 3)),
            arms=(0.0, 1.0, 2.0),
            g_bounds=INERT,
        )
        with pytest.raises(ValueError, match="two arms"):
            reduced_mechanism_covariate(three, np.full(law.N, 0.5), bounds=INERT)

        with pytest.raises(ValueError, match="rows"):
            reduced_mechanism_covariate(reduced, np.full(law.N + 1, 0.5), bounds=INERT)


class TestTheyVanishAtTheTruth:
    r""":math:`Q_r \equiv g_{r,2} \equiv 0` at exact nuisances, so both covariates are zero.

    Which is the degeneracy that makes every exact-law instrument in this package blind to
    this variant: with both covariates zero, both extra coefficients are zero and the
    estimator reproduces ``TMLE`` array for array.  Pinned here so the blindness is a
    recorded property rather than a surprise to whoever writes the next test.
    """

    def test_both_covariates_are_zero(self) -> None:
        _, treatment, _ = realised()
        cell, _, _ = realised()
        reduced = reduced_at(law.G, law.Q)

        submodel = reduced_outcome_submodel(treatment, reduced, bounds=INERT)
        covariate = reduced_mechanism_covariate(reduced, law.G[cell], bounds=INERT)

        assert np.max(np.abs(submodel.observed)) < 1e-14
        assert np.max(np.abs(covariate)) < 1e-14

    def test_but_gr1_does_not_vanish(self) -> None:
        """The trap: it is a probability sitting in a denominator whose numerator vanishes.

        An implementation that got ``gr1`` wrong -- inverted with ``gr2``, say, which is
        how the R source names them -- would pass the test above and every Gateaux module.
        """
        reduced = reduced_at(law.G, law.Q)
        assert np.min(reduced.gr1) > 0.2
        assert np.max(reduced.gr1) < 0.8
