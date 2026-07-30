r"""Invariants that hold by definition, and so must hold exactly.

Most of the estimator's algebraic identities are already asserted where the machinery they
belong to is tested: ``ate == ey1 - ey0`` and the ratio transforms in
:mod:`tests.unit.test_influence_gateaux` and :mod:`tests.e2e.test_oracle`; the
scaled/unscaled affine round trip in :mod:`tests.unit.test_bounds`; invariance to the scale
of the observation weights in :mod:`tests.unit.test_weighted_estimand`; iterative against
one-step targeting and the weighted against the clever-covariate parameterisation in
:mod:`tests.unit.test_fluctuation`.

Two are not asserted anywhere, and they are the two that catch the specific mistake a
statistical test is worst at catching -- a sign or a population swapped somewhere in the
conditional-effect machinery.  Both are checked here on :mod:`tests.discrete_law`, where a
sample realises the law exactly and oracle nuisances leave nothing to estimate, so the
comparisons are bit-for-bit rather than statistical.

**Relabelling the arms.**  Replacing :math:`A` by :math:`1 - A` is a change of notation, not
of data.  Every estimand has to follow it in the way its definition dictates:
:math:`EY_1` and :math:`EY_0` trade places, the average treatment effect and the two log
ratios negate, and -- the interesting one --

.. math:: \mathrm{ATT}' = -\,\mathrm{ATC}, \qquad \mathrm{ATC}' = -\,\mathrm{ATT},

because conditioning on the relabelled treated is conditioning on the original controls.
An implementation that inverted the propensity-odds factor :math:`g_1/g_0` produces
perfectly plausible numbers on any single fit and fails this immediately -- verified by
making that inversion in :func:`~cleverly.fluctuation.submodel.att_submodel` and watching
six of these tests turn red.

It is worth being exact about what this does *not* catch, because the boundary is
instructive.  Swapping the two conditioning populations outright -- computing the ATT with
the control arm and the ATC with the treated one -- leaves every assertion in this class
passing, since a defect symmetric in the arms survives a test of symmetry in the arms
untouched.  That mistake is caught instead by the closed-form comparisons in
:mod:`tests.unit.test_influence_gateaux` and :mod:`tests.unit.test_remainder_cde`, where
the estimate is checked against an independently written functional rather than against
itself relabelled (verified the same way: forty-four failures).  Symmetry tests and
closed-form tests fail on disjoint mistakes, which is the argument for keeping both.

**A null outcome model.**  If :math:`\bar Q(1, w) = \bar Q(0, w)` at every :math:`w` then
the effect is zero at every :math:`w`, hence zero in every population, however the
treatment was assigned.  The law used here confounds hard -- the propensity runs from 0.25
to 0.75 and the unadjusted difference in means is ``-0.175`` -- so the estimator has real
work to do to return the zero, and an estimator that returned zero for the wrong reason
would be caught by the same comparison.

This is deliberately not the null used by the type I error study in
:mod:`tests.e2e.test_coverage_slow`, which achieves a zero effect by leaving :math:`A` out
of the outcome model altogether.  That is a statement about a data-generating process; this
is a statement about arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import TMLE
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment, fast_tmle

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")

#: How a relabelling ``A <- 1 - A`` maps each estimand: the name it becomes, and the sign.
RELABELLED: dict[str, tuple[str, float]] = {
    "ey1": ("ey0", 1.0),
    "ey0": ("ey1", 1.0),
    "ate": ("ate", -1.0),
    "att": ("atc", -1.0),
    "atc": ("att", -1.0),
    "rr": ("rr", -1.0),
    "or": ("or", -1.0),
}


def _oracle_fit(frame: pd.DataFrame, dgp: Any) -> Any:
    """A fit with both nuisances handed to it, so nothing is estimated."""
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"]).single()


def _value(estimate: Any) -> float:
    """The point estimate on the scale its influence curve lives on."""
    return float(estimate.log_psi if estimate.scale == "ratio" else estimate.psi)


class TestRelabellingTheArms:
    """``A <- 1 - A`` is a change of notation; the estimates have to follow it exactly."""

    @pytest.fixture(scope="class")
    def original(self):
        return _oracle_fit(law.frame(), law.DiscreteLaw())

    @pytest.fixture(scope="class")
    def relabelled(self):
        """The same data with the arms swapped, and an oracle that knows it.

        Swapping the ``A`` axis of the cell probabilities gives the law of the relabelled
        data exactly -- ``g' = 1 - g`` and ``Qbar'(a, w) = Qbar(1 - a, w)`` -- so the
        oracle stays an oracle rather than becoming a misspecified model that happens to
        be close.
        """
        frame = law.frame()
        frame["A"] = 1.0 - frame["A"]
        return _oracle_fit(frame, law.DiscreteLaw(law.PROBS[:, ::-1, :]))

    def test_the_relabelled_law_is_the_mirror_image(self) -> None:
        # The premise: if this failed, the fixture's oracle would be wrong and every
        # assertion below would be measuring that instead.
        mirrored = law.DiscreteLaw(law.PROBS[:, ::-1, :])
        np.testing.assert_allclose(mirrored.g, 1.0 - law.G_EXACT, atol=1e-15, rtol=0)
        np.testing.assert_allclose(mirrored.q, law.Q_EXACT[:, ::-1], atol=1e-15, rtol=0)

    def test_the_relabelling_is_not_a_symmetry_of_the_law(self) -> None:
        # Teeth: on a law where the arms were interchangeable every claim below would hold
        # for an estimator that ignored the treatment entirely.
        assert abs(law.TRUTH["ate"]) > 0.2
        assert abs(law.TRUTH["att"] - law.TRUTH["atc"]) > 1e-3
        assert np.max(np.abs(law.G_EXACT - 0.5)) > 0.1

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_follows_the_relabelling(self, original, relabelled, name) -> None:
        partner, sign = RELABELLED[name]
        assert _value(relabelled.estimates[name]) == pytest.approx(
            sign * _value(original.estimates[partner]), abs=1e-12
        )

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_influence_curve_follows_the_relabelling(self, original, relabelled, name) -> None:
        # Row order is untouched by the relabelling, so the two curves are comparable
        # element by element -- which is a far stronger claim than agreement of their
        # summaries, and the one that pins the sign inside the conditional-effect terms.
        partner, sign = RELABELLED[name]
        got = np.asarray(relabelled.estimates[name].influence_curve)
        want = sign * np.asarray(original.estimates[partner].influence_curve)
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_standard_error_is_unchanged(self, original, relabelled, name) -> None:
        # A negation leaves the variance alone, so the interval has the same width whether
        # the question was asked one way round or the other.
        partner, _ = RELABELLED[name]
        assert relabelled.estimates[name].std_error == pytest.approx(
            original.estimates[partner].std_error, abs=1e-12
        )

    @pytest.mark.parametrize("name", ("ate", "att", "atc"))
    def test_it_holds_for_fitted_learners_too(self, name: str) -> None:
        """The same claim without oracles, which is what exercises the design matrices.

        The oracle version cannot see a mistake in how ``[A, W]`` is assembled, because an
        oracle ignores its design.  A fitted GLM cannot be exact here -- the two fits solve
        different optimisation problems and stop at different points -- so this asserts
        agreement at the solver's tolerance rather than at machine precision.
        """
        frame = law.frame()
        swapped = frame.copy()
        swapped["A"] = 1.0 - swapped["A"]
        kwargs = {"outcome": "Y", "treatment": "A", "covariates": ["W"]}
        first = fast_tmle(estimands=("ate", "att", "atc")).fit(frame, **kwargs).single()
        second = fast_tmle(estimands=("ate", "att", "atc")).fit(swapped, **kwargs).single()
        partner, sign = RELABELLED[name]
        assert second.estimates[name].psi == pytest.approx(
            sign * first.estimates[partner].psi, abs=1e-6
        )


#: ``P(W = w)``, ``g(w)`` and an outcome mean that does *not* depend on the arm.
NULL_P_W = np.array([0.50, 0.25, 0.25])
NULL_G = np.array([0.75, 0.50, 0.25])
NULL_Q = np.array([0.25, 0.50, 0.75])
NULL_N = 1024


def _null_law() -> tuple[pd.DataFrame, np.ndarray]:
    """A sample realising ``Qbar(1, w) == Qbar(0, w)`` exactly, and its cell probabilities.

    Built here rather than in :mod:`tests.discrete_law` because that module is one specific
    law with one set of constants; this is a different law that happens to share its shape,
    and :class:`tests.discrete_law.DiscreteLaw` already accepts arbitrary cell
    probabilities for exactly this reason.
    """
    counts = np.empty((3, 2, 2))
    for w in range(3):
        for a in range(2):
            arm = NULL_G[w] if a == 1 else 1.0 - NULL_G[w]
            for y in range(2):
                outcome = NULL_Q[w] if y == 1 else 1.0 - NULL_Q[w]
                counts[w, a, y] = NULL_P_W[w] * arm * outcome * NULL_N
    if np.max(np.abs(counts - np.rint(counts))) > 1e-6:  # pragma: no cover - guards constants
        raise AssertionError("the null law's cell probabilities are not multiples of 1/N")
    counts = np.rint(counts).astype(int)

    support = [(w, a, y) for w in range(3) for a in range(2) for y in range(2)]
    repeats = [counts[cell] for cell in support]
    columns = np.repeat(np.array(support, dtype=float), repeats, axis=0)
    frame = pd.DataFrame({"W": columns[:, 0], "A": columns[:, 1], "Y": columns[:, 2]})
    return frame, counts / NULL_N


class TestAnArmIndependentOutcomeGivesExactlyZero:
    r""":math:`\bar Q(1, \cdot) \equiv \bar Q(0, \cdot)` implies a zero effect, exactly."""

    @pytest.fixture(scope="class")
    def null_fit(self):
        frame, probs = _null_law()
        return _oracle_fit(frame, law.DiscreteLaw(probs))

    def test_the_law_confounds_hard_enough_to_matter(self) -> None:
        # Teeth.  Without this the whole class would be satisfied by an estimator that
        # returned zero unconditionally, or by one that never adjusted for anything.
        frame, _ = _null_law()
        treated = frame.loc[frame["A"] == 1.0, "Y"].mean()
        control = frame.loc[frame["A"] == 0.0, "Y"].mean()
        assert treated - control == pytest.approx(-0.174603, abs=1e-5)
        assert NULL_G.max() - NULL_G.min() >= 0.5

    def test_the_truth_is_zero(self) -> None:
        # The identification formula, evaluated longhand on the null law's own cell
        # probabilities -- not an assumption about what the estimator should return.
        _, probs = _null_law()
        for name in ("ate", "att", "atc", "rr", "or"):
            assert float(law.functional(probs, name)) == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("name", ("ate", "att", "atc"))
    def test_the_estimate_is_zero(self, null_fit, name: str) -> None:
        assert null_fit.estimates[name].psi == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("name", ("rr", "or"))
    def test_the_log_ratios_are_zero(self, null_fit, name: str) -> None:
        assert null_fit.estimates[name].log_psi == pytest.approx(0.0, abs=1e-15)
        assert null_fit.estimates[name].psi == pytest.approx(1.0, abs=1e-15)

    def test_the_two_counterfactual_means_coincide(self, null_fit) -> None:
        assert null_fit.estimates["ey1"].psi == pytest.approx(
            null_fit.estimates["ey0"].psi, abs=1e-15
        )

    def test_the_influence_curve_is_not_trivially_zero(self, null_fit) -> None:
        r"""A zero estimate does not mean a zero influence curve, and should not.

        With :math:`\bar Q(1, w) = \bar Q(0, w) = \bar Q(w)` the two plug-in terms cancel
        but the residual terms do not:

        .. math::

            D^*_{\mathrm{ate}}(O) = \left(\frac{\mathbb 1\{A = 1\}}{g(W)}
                - \frac{\mathbb 1\{A = 0\}}{1 - g(W)}\right)\bigl(Y - \bar Q(W)\bigr).

        So the interval around the zero has positive width, which is the correct answer --
        the effect is zero, and the data still only pin it down to within sampling error.
        Asserting the closed form is what distinguishes "correctly zero" from "zeroed out".
        """
        frame, _ = _null_law()
        w = frame["W"].to_numpy().astype(int)
        a = frame["A"].to_numpy(dtype=float)
        y = frame["Y"].to_numpy(dtype=float)
        g = NULL_G[w]
        longhand = (a / g - (1.0 - a) / (1.0 - g)) * (y - NULL_Q[w])
        np.testing.assert_allclose(
            np.asarray(null_fit.estimates["ate"].influence_curve), longhand, atol=1e-12, rtol=0
        )
        # Positive and not negligible: about 0.031 on this law, so the interval is roughly
        # +/- 0.06 around a point estimate that is exactly zero.
        assert null_fit.estimates["ate"].std_error > 0.02
