r"""Theorem 1's objects, in the theorem's notation, against the ones the package reports.

This module resolves the discrepancy in the mechanism correction's *sign*
disagrees with Benkeser, Carone, van der Laan & Gilbert.  It is the one class of error nothing
else here can catch -- all three empirical means are driven to zero, so no reported
:math:`\hat\Psi` moves however the signs go, and what a wrong sign moves is the variance,
which is the only product this variant has.

**The source is** Benkeser, Carone, van der Laan & Gilbert (2016), the Berkeley working paper (UCB
Biostatistics paper 356).  Its display equations and its own appendices do not agree, and that
disagreement is the question under test. Writing :math:`u` and :math:`v` for the two
**positive** quantities the software computes,

.. math::

    u := \frac{\bar Q_r(w)}{g(w)}\{a - g(w)\},
    \qquad
    v := a\,\frac{g_{2,r}(w)}{g_{1,r}(w)}\{y - \bar Q(w)\},

* the display in §3.1 defines :math:`D_A := -u`, with a *leading minus*, and its bivariate
  :math:`D_Y` likewise -- while §3.2's redefinition of :math:`D_Y` for the univariate
  construction, the one this package implements, has **no** leading minus.  The same object
  printed twice with two signs is the first thing that says a display is not the place to
  settle this;
* Theorem 1 reports :math:`D^{*,\#} = D^* - D_A - D_Y` and
  :math:`\sigma_n^2 = P_n\{D^* - D_A - D_Y\}^2`;
* **Appendices A and B derive both terms**, and each derivation reads
  :math:`P_0[\text{term}] = -(P_n - P_0)D + B_n + (\text{second order})` with
  :math:`B_n := P_n D`.  Since :math:`P_0[u] = P_n[u] - (P_n - P_0)[u]` is an identity for any
  :math:`u` whatever, that decomposition is satisfiable only with :math:`D` equal to the
  **positive** term.  So the appendices force :math:`D_A = +u` and :math:`D_Y = +v`, and then
  :math:`D^{*,\#} = D^* - u - v` -- which is exactly what this package and ``drtmle`` compute.

The leading minus in the §3.1 display is therefore not a rival convention to be matched; it is
inconsistent with the derivation printed twenty pages later in the same document, and with
Theorem 1's own :math:`\sigma_n^2`.  **Item 21 resolves in favour of the implementation.**

What is checkable rather than argued is below, and it is the reason this module exists at all:
Appendix A's opening algebraic step, which is what fixes *which* quantity the decomposition is
of; and the consequence for the asymptotic-linearity representation, which holds with the
corrections **subtracted** and fails by exactly twice the correction when they are added.

Everything runs on ``tests/discrete_law.py`` at nuisances that are **wrong on purpose**, for
the reason :mod:`tests.unit.test_remainder_drtmle` sets out: at the truth :math:`Q_r` and
:math:`g_{r,2}` vanish row by row, both readings give the same array, and every check here
would pass against either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from tests import discrete_law as law
from tests.conftest import binary_means
from tests.unit.test_remainder_drtmle import (
    ESTIMANDS,
    WRONG_G,
    WRONG_Q,
    _expansion,
    _extra_curves,
    _reduced,
)

#: The arms, as the oracle indexes them.
ARMS = (0, 1)


def positive_terms(arm: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``(u, v)`` at every row: the theorem's corrections in the orientation it derives.

    Built by :func:`tests.unit.test_remainder_drtmle._extra_curves`, which writes them out
    longhand from the law -- so the mapping between the paper's objects and the package's is
    an equality of arrays rather than a table of names.
    """
    return _extra_curves(WRONG_G, WRONG_Q, arm)


def library_parts() -> Any:
    r"""The **package's** :math:`D^*_g` and :math:`D^*_Q` at the same wrong nuisances.

    Split out of :class:`TestTheImplementationComputesTheTheoremsTerms` so that
    :class:`TestTheReportedVarianceIsTheorem1s` reads the same arrays: the array
    comparison and the interval comparison have to be about one object, or the second is
    checking a second construction rather than the reported one.
    """
    from cleverly.inference.influence import reduced_correction_parts
    from tests.unit.test_reduced_submodel import reduced_at

    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)
    at_one, at_zero = WRONG_Q[covariate, 1], WRONG_Q[covariate, 0]
    targeted = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    return reduced_correction_parts(
        outcome,
        targeted,
        treatment,
        reduced_at(WRONG_G, WRONG_Q),
        WRONG_G[covariate],
        bounds=(1e-6, 1.0 - 1e-6),
        guard=("Q", "g"),
    )


def library_corrections(arm: int) -> np.ndarray:
    """What the reported curve subtracts for one arm, through the production selector.

    :meth:`~cleverly.inference.influence.CorrectionParts.total` rather than
    ``d_g + d_q``, so the partial-guard branch is on the path this reads.
    """
    return np.asarray(library_parts().total()[float(arm)], dtype=float)


def plain_curve(arm: int) -> np.ndarray:
    """:math:`D^*` for one arm at the wrong-on-purpose nuisances, with no correction."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)
    at_one, at_zero = WRONG_Q[covariate, 1], WRONG_Q[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    submodel = submodel_for("mean", treatment, WRONG_G[covariate])
    _, ic_one, _, ic_zero = binary_means(outcome, initial, submodel, np.ones(law.N))
    return np.asarray(ic_one if arm == 1 else ic_zero, dtype=float)


class TestAppendixAOpensByFixingTheOrientation:
    r"""Which quantity the decomposition is *of*, checked on the law rather than read.

    Appendix A begins by rewriting the mechanism-side piece of the remainder as

    .. math::

        -P_0\Bigl\{\frac{\bar Q_{0n,r}}{g_0}(g_n - g_0)\Bigr\}
          = P_0\Bigl\{\frac{\bar Q_{n,r}}{g_0}(A - g_n)\Bigr\},

    which holds because :math:`E_0[A \mid W] = g_0`.  The right-hand side is the **positive**
    term; the decomposition that follows is of *that*, and :math:`B_{A,n} := P_n D_A` is
    subtracted from it.  So :math:`D_A` is the positive term, and the display's :math:`-u` is
    the thing that does not fit.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_two_sides_agree_exactly_on_the_law(self, arm: int) -> None:
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        treatment = frame["A"].to_numpy(dtype=float)
        indicator = (treatment == float(arm)).astype(float)

        qr, _, _ = _reduced(WRONG_G, WRONG_Q, arm)
        truth = law.G if arm == 1 else 1.0 - law.G
        guess = WRONG_G if arm == 1 else 1.0 - WRONG_G
        ratio = (qr / truth)[covariate]

        left = -float(np.mean(ratio * (guess - truth)[covariate]))
        right = float(np.mean(ratio * (indicator - guess[covariate])))

        assert left == pytest.approx(right, abs=1e-14)
        assert abs(right) > 1e-3, "or both sign readings would agree on this fixture"

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_displays_reading_would_need_the_correction_to_vanish(self, arm: int) -> None:
        r"""``P_0[u] = P_n[D_A]`` with ``D_A = -u`` forces ``P_0[u] = 0``, and it is not.

        On this law the sample realises :math:`P_0` exactly, so
        :math:`(P_n - P_0)D_A = 0` and the appendix's decomposition collapses to
        :math:`P_0[u] = P_n[D_A]`.  With the display's :math:`D_A = -u` that reads
        :math:`P_0[u] = -P_0[u]`.  The assertion below is what makes that a contradiction
        rather than a possibility -- and it is exactly the degeneracy that appears **at the
        truth**, where :math:`Q_r` vanishes row by row and the two readings really are the
        same.
        """
        u, v = positive_terms(arm)
        assert abs(float(np.mean(u))) > 1e-3
        assert abs(float(np.mean(v))) > 1e-3


class TestTheRepresentationPicksTheSign:
    r"""The consequence, on the quantity Theorem 1 is a statement about.

    Equation (1) of the paper is
    :math:`\Psi(Q_n) - \Psi(Q_0) = (P_n - P_0)D^* - B_n + M_n + R_n`; the appendices expand
    :math:`R_n`, and once the algorithm has driven the bias terms to :math:`o_P(n^{-1/2})` the
    estimator is asymptotically linear with :math:`D^{*,\#} = D^* - u - v`.  On a law the
    sample realises exactly, with saturated reductions, every term in that expansion is
    computable and the claim becomes an equality that either holds or does not.
    """

    @pytest.mark.parametrize("name", ESTIMANDS)
    @pytest.mark.parametrize("guard", [("Q",), ("g",)])
    def test_subtracting_the_correction_closes_the_expansion(
        self, guard: tuple[str, ...], name: str
    ) -> None:
        """Theorem 1's reading, which is the package's: nothing of first order is left."""
        assert _expansion(WRONG_G, WRONG_Q, guard=guard, sign=-1.0)[name] == pytest.approx(
            0.0, abs=1e-12
        )

    @pytest.mark.parametrize("name", ESTIMANDS)
    @pytest.mark.parametrize("guard", [("Q",), ("g",)])
    def test_adding_it_leaves_exactly_twice_the_correction(
        self, guard: tuple[str, ...], name: str
    ) -> None:
        r"""The display's reading, and it does not fail by a little.

        Stated as an equality rather than as "it is large": the two readings differ by
        :math:`2 P_n[\text{correction}]`, so a check that the wrong one is merely non-zero
        would pass against an unrelated error too.
        """
        wrong = _expansion(WRONG_G, WRONG_Q, guard=guard, sign=+1.0)[name]
        right = _expansion(WRONG_G, WRONG_Q, guard=guard, sign=-1.0)[name]
        term = 0 if guard == ("Q",) else 1
        per_arm = {arm: float(np.mean(positive_terms(arm)[term])) for arm in ARMS}
        doubled = (
            2.0
            * {
                "ey1": per_arm[1],
                "ey0": per_arm[0],
                "ate": per_arm[1] - per_arm[0],
            }[name]
        )

        assert wrong - right == pytest.approx(doubled, abs=1e-12)
        assert abs(wrong) > 1e-3


class TestTheImplementationComputesTheTheoremsTerms:
    r"""The step that makes the adjudication above a statement about *this* package.

    Everything else here is arithmetic on the oracle's longhand terms, which would go on
    holding if the library flipped a sign tomorrow.  This closes the loop: the arrays
    :func:`~cleverly.inference.influence.reduced_correction_parts` builds **are** :math:`u`
    and :math:`v`, so Theorem 1's :math:`D^{*,\#} = D^* - D_A - D_Y` with the appendices'
    reading is the curve the package reports, row for row.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_parts_are_the_theorems_positive_terms(self, arm: int) -> None:
        parts = library_parts()
        u, v = positive_terms(arm)

        np.testing.assert_allclose(parts.d_g[float(arm)], u, rtol=0, atol=1e-14)
        np.testing.assert_allclose(parts.d_q[float(arm)], v, rtol=0, atol=1e-14)


class TestTheTwoSignsAreMateriallyDifferentVariances:
    """Why the sign discrepancy was a correctness question rather than a documentation error.

    The point estimate cannot see the sign, so the whole of the difference lands in
    :math:`\\sigma_n^2` -- the only quantity this variant produces.  A fixture on which the
    two readings agreed would make the item unanswerable rather than answered, so the
    separation is asserted before anything else here is believed.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_reported_variance_moves_with_the_sign(self, arm: int) -> None:
        curve = plain_curve(arm)
        u, v = positive_terms(arm)

        subtracted = float(np.var(curve - u - v))
        added = float(np.var(curve + u + v))

        assert abs(subtracted / added - 1.0) > 0.05, (
            "the fixture must separate the two sign readings"
        )

    @pytest.mark.parametrize("arm", ARMS)
    def test_and_the_only_way_the_mean_can_see_it_is_through_the_corrections(
        self, arm: int
    ) -> None:
        r"""Why nothing that reports only :math:`\hat\Psi` could have caught the sign discrepancy.

        The two readings' means differ by **exactly** :math:`2 P_n[u + v]`, and the targeting
        step's whole job is to drive that to zero -- so on a solved fit the difference the
        test above measures is invisible to every estimate-based assertion in the package,
        while the variance moves by 5% or more.  These nuisances are untargeted, which is the
        only reason the quantity is non-zero here and available to be checked at all.
        """
        curve = plain_curve(arm)
        u, v = positive_terms(arm)

        difference = float(np.mean(curve + u + v)) - float(np.mean(curve - u - v))
        assert difference == pytest.approx(2.0 * float(np.mean(u + v)), abs=1e-14)


class TestTheReportedVarianceIsTheorem1s:
    r"""Theorem 1's :math:`\sigma_n^2 = P_n\{D^* - D_A - D_Y\}^2`, and what is reported.

    Everything above this class is about the *curve*.  The theorem's last line is about
    the **variance**, and until this class it was pinned only through the curve it is
    built from -- which is not the same claim: the theorem's form is an *uncentred* second
    moment, the package reports a centred one, and the interval a reader is shown is the
    composition of the two with a contrast in between.  Three things, each of which can be
    wrong on its own.

    Fit-free, as the rest of this module is: the arrays are the oracle's longhand terms
    and the only library code is the one function that turns a curve into an interval.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_interval_the_package_builds_is_the_theorems(self, arm: int) -> None:
        r"""The variance built from the package's **own** corrections is the theorem's.

        :class:`TestTheImplementationComputesTheTheoremsTerms` closes the loop at the
        arrays; this closes it at the number a reader is shown.  The curve here is
        :func:`~cleverly.inference.influence.reduced_correction_parts`' output subtracted
        from :math:`D^*`, and the claim is that
        :func:`~cleverly.inference.influence.make_estimate` then reports
        :math:`\sigma_n^2/n` for :math:`\sigma_n^2` built out of the theorem's :math:`u`
        and :math:`v` -- so a sign, a dropped term or a re-association in
        :meth:`~cleverly.inference.influence.CorrectionParts.total` moves the interval and
        this goes red.

        The second assertion is the fixture's: an interval that did **not** move would
        make the first one vacuous, and the corrections are the only product this variant
        has.
        """
        from cleverly.inference.influence import make_estimate

        curve = plain_curve(arm)
        u, v = positive_terms(arm)
        name = f"ey{arm}"

        packaged = make_estimate(name, 0.0, curve - library_corrections(arm), n=law.N)
        theorem = make_estimate(name, 0.0, curve - u - v, n=law.N)
        ordinary = make_estimate(name, 0.0, curve, n=law.N)

        assert packaged.variance == pytest.approx(theorem.variance, rel=1e-12)
        assert packaged.ci == pytest.approx(theorem.ci, rel=1e-12)
        assert abs(packaged.std_error / ordinary.std_error - 1.0) > 0.02

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_theorems_uncentred_form_is_the_reported_one_once_the_score_is_zero(
        self, arm: int
    ) -> None:
        r""":math:`P_n\{D\}^2` and :math:`\operatorname{Var}_n(D)` differ by
        :math:`(P_n D)^2` -- **exactly**, and that is the whole of the difference between
        what Theorem 1 writes and what the package reports.

        So the theorem's variance is the reported one *conditional on the score equations
        being solved*, which is the qualification
        :mod:`cleverly.validation.score` exists to check and not an approximation to wave
        through.  These nuisances are untargeted, so the gap is material here -- which is
        what makes the second half of this test a statement rather than a tautology.
        """
        curve = plain_curve(arm)
        u, v = positive_terms(arm)
        corrected = curve - u - v

        uncentred = float(np.mean(corrected**2))
        centred = float(np.var(corrected))
        score = float(np.mean(corrected))

        assert uncentred - centred == pytest.approx(score**2, abs=1e-14)
        assert abs(score) > 1e-3, "an untargeted fixture, or the gap below is not being shown"

        solved = corrected - score  # what the targeting step delivers
        assert float(np.mean(solved**2)) == pytest.approx(float(np.var(solved)), abs=1e-14)

    def test_the_contrast_reads_the_covariance_rather_than_the_sum(self) -> None:
        r"""``ate``'s variance is :math:`\operatorname{Var}(D_1 - D_0)`, not the sum.

        The two arms' corrected curves are strongly dependent -- they share :math:`Y`,
        :math:`A` and the same reduced regressions -- so a contrast that added the
        diagonal would report an interval of the wrong width in a direction nothing else
        here would catch.  Asserted through
        :func:`~cleverly.inference.cluster.influence_covariance`, which is what the
        delta method and the simultaneous bands read.
        """
        from cleverly.inference.cluster import influence_covariance

        curves = []
        for arm in ARMS:
            u, v = positive_terms(arm)
            curves.append(plain_curve(arm) - u - v)
        matrix = influence_covariance(np.column_stack(curves))

        contrast = np.array([-1.0, 1.0])  # ARMS is (0, 1), so ate = column 1 - column 0
        through_covariance = float(contrast @ matrix @ contrast)
        directly = float(np.var(curves[1] - curves[0], ddof=1) / law.N)

        assert through_covariance == pytest.approx(directly, rel=1e-12)
        assert through_covariance != pytest.approx(float(matrix[0, 0] + matrix[1, 1]), rel=1e-3), (
            "the arms must covary here, or this fixture cannot tell a sum from a difference"
        )
