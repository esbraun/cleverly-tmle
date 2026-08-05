r"""Is the corrected remainder the quantity item 13 asks about, or a plausible neighbour?

``benchmarks/drtmle_remainder.py`` computes

.. code-block:: text

    R_remaining = psi-hat - psi_0 - (P_n - P_0) D-hat_DR

and the whole of what makes it a measurement rather than an assertion is that
:math:`P_0\hat D` is taken at an **independent draw** through the fit's own nuisance
functions.  ``docs/drtmle/validation-plan.md`` §5 refuses :math:`P_n\hat D` in its place by
name, and the substitution is the plausible shortcut: it is one character away, it produces a
number of the right size, and it would make ``R_remaining`` equal to
:math:`\hat\psi - \psi_0` -- which is the bias, not the remainder, and would report a
first-order quantity as a second-order one.

Four claims here, and the first is the control every module in this variant owes.

*At correct nuisances everything vanishes.*  ``R_2``, both branches and ``R_remaining`` are
zero to quadrature error when the estimator is handed the truth.  That is the non-failing
control: a module whose numbers are large at the truth is measuring its own arithmetic.

*The two renderings differ.*  ``R_remaining`` is not :math:`\hat\psi - \psi_0`, on a real
fit, by a margin no rounding explains.

*The plain remainder agrees with Tier 1's quadrature.*  At Tier 1's injected sequence
:func:`~benchmarks.drtmle_remainder.plain_remainder` and
:func:`~benchmarks.drtmle_injection.exact_remainder` compute the same population integral by
completely different routes -- one over a companion draw at the fitted arrays, one by Sobol
quadrature over prescribed functions.  Two independent renderings of one definition, which
is the strongest check available here.

*The fold weighting is the estimator's.*  A uniform average over the folds is a different
convention, and §5 requires the one in force be documented rather than discovered.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pytest
from benchmarks import drtmle_injection as injection
from benchmarks import drtmle_remainder as remainder

from cleverly import DRTMLE
from cleverly.datasets import binary_outcome_dgp, clustered_dgp

#: ``600`` rather than something smaller, and that is not a typo: a doubly-robust fit at
#: ``400`` rows takes *longer* than one at ``600`` -- noisier nuisances loosen the coupling
#: and lengthen the alternation, which ``tests/unit/test_drtmle_fit.py`` already records and
#: C1 measured at ``16.4s`` against ``5.6s``.
N = 600

#: Sobol points of the deterministic rule, so the companion is ``2 * POINTS`` rows -- and the
#: number the i.i.d. arm is given, so the two rules are compared **at the same row count**
#: rather than at whichever one flatters the newer.
POINTS = 2_048
EVALUATION_N = 2 * POINTS
SEED = 7


def _fit(cell: str, *, oracle: bool = False, rule: str = "sobol", points: int = POINTS) -> Any:
    """One Tier-1 fit, under either evaluation rule.

    ``rule="sobol"`` is :func:`~benchmarks.drtmle_remainder.quadrature_frame` and returns the
    fit with its weight vector; ``rule="draw"`` is the i.i.d. companion, whose weights are
    ``None``.  Both are here because the deterministic rule's whole warrant is that it
    computes the same population integral the i.i.d. one estimates, and a module that only
    ran the new rule could not say so.
    """
    dgp = injection.base_law()
    frame, _ = dgp.sample(N, seed=SEED)
    if rule == "sobol":
        evaluation, weights = remainder.quadrature_frame(dgp, points)
    else:
        evaluation, weights = remainder.evaluation_frame(dgp, 2 * points, seed=99_991), None
    shared = injection.settings(cell, N)
    if oracle:
        # The injected sequence with its drift switched off: the prescribed functions *are*
        # the truth, so the estimator is handed correct nuisances and every remainder below
        # has to vanish. Built by asking for the sequence at a size where n^(-alpha) is
        # numerically zero rather than by writing a second oracle, so the control runs
        # through the same learners the measurement does.
        shared = {**shared, "outcome_learner": injection.InjectedOutcome(cell, _HUGE)}
        if cell == "g-drift":
            shared = {**shared, "treatment_learner": injection.InjectedMechanism(cell, _HUGE)}
    result = (
        DRTMLE(
            **shared,
            reduced_outcome_learner="glm",
            reduced_treatment_learner="glm",
            random_state=3,
            evaluation=evaluation,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    return result, weights


def _rows(fit: Any, weights: Any, *, points: int = POINTS, limit: int | None = None) -> Any:
    """Every remainder column, with the rule's weights and its own ``psi_0`` attached."""
    dgp = injection.base_law()
    truth = remainder.truth_at(dgp, points) if weights is not None else None
    return {
        row.estimand: row
        for row in remainder.remainder_rows(
            fit,
            dgp,
            n=N,
            bounds=fit.config.g_bounds,
            row_weights=weights,
            limit=limit,
            truth=truth,
        )
    }


def _scaled_outcome(fit: Any) -> Any:
    """The companion's own ``Y``, on the scale the curve is built at."""
    evaluation = fit.repeats[0].fluctuations["mean"].reduction.evaluation
    return fit.nuisance.scaler.scale(evaluation.data.outcome)


#: A size at which the injected drift ``n^(-alpha)`` is below floating-point resolution
#: against the outcome, so the "sequence" is the truth exactly.  ``1e64 ** -0.25 = 1e-16``.
_HUGE = 10**64


@pytest.fixture(scope="module")
def fitted() -> Any:
    """The deterministic rule, which is the one the study's remainder columns come off."""
    return _fit("q-drift")


@pytest.fixture(scope="module")
def sampled() -> Any:
    """The i.i.d. rule at the **same row count**, which is what the other is checked against."""
    return _fit("q-drift", rule="draw")


@pytest.fixture(scope="module")
def rows(fitted: Any) -> dict[str, Any]:
    """The remainder columns of one fit, computed once.

    Module-scoped because every branch column is a pair of binned quadratures over the whole
    evaluation draw, per arm and per fold -- recomputing them per test is the commonest
    waste ``CLAUDE.md`` names, and nothing here mutates them.
    """
    return _rows(*fitted)


class TestTheControlThatMustNotFail:
    """At correct nuisances every remainder here is zero.

    The ``q-drift`` cell with its drift switched off: the outcome regression *is*
    :math:`\\bar Q_0` and the mechanism is the cell's wrong limit, so ``R_2`` -- which is an
    inner product of the two errors -- vanishes through the outcome factor alone.  A module
    that reported something large here would be measuring its own arithmetic.
    """

    @pytest.fixture(scope="class")
    def at_truth(self) -> Any:
        return _fit("q-drift", oracle=True)

    def test_the_plain_remainder_vanishes(self, at_truth: Any) -> None:
        fit, weights = at_truth
        values = remainder.plain_remainder(fit, injection.base_law(), fit.config.g_bounds, weights)
        assert abs(values["r2_ate"]) < 1e-9

    def test_the_corrected_remainder_is_small_against_the_bias_it_would_otherwise_carry(
        self, at_truth: Any
    ) -> None:
        """Not exactly zero, and the reason is worth stating rather than absorbing.

        With one nuisance exactly right the *expansion*'s remainder vanishes, but
        ``R_remaining`` is computed at a **finite** evaluation rule, so what is left is that
        rule's own error -- an :math:`O(m^{-1/2})` quantity with no ``n`` in it on the i.i.d.
        draw and a Sobol discretisation on the deterministic grid.  The bar is therefore a
        Monte Carlo one, and it is stated as a share of the estimator's own standard error
        rather than as an absolute number.
        """
        fit, weights = at_truth
        rows = _rows(fit, weights)
        standard_error = fit.estimates["ate"].std_error
        assert abs(rows["ate"].remaining) < 0.5 * standard_error


class TestTheRemainderIsNotTheBias:
    """``P_n D-hat`` in place of ``P_0 D-hat`` is the shortcut §5 refuses by name."""

    def test_the_two_renderings_differ_on_a_real_fit(self, rows: dict[str, Any]) -> None:
        row = rows["ate"]
        bias = row.psi - row.truth
        # The curve accounts for most of the bias, which is what asymptotic linearity means
        # and what makes the remainder a *second-order* quantity.  Reporting `P_n D-hat` in
        # place of `P_0 D-hat` would make these two equal to the digit.
        assert abs(row.remaining) < 0.5 * abs(bias)
        assert abs(bias) > 1e-3

    def test_the_empirical_mean_of_the_reported_curve_is_what_targeting_drove_to_zero(
        self, rows: dict[str, Any]
    ) -> None:
        """Which is exactly why it cannot stand in for the population mean."""
        assert abs(rows["ate"].pn_curve) < 1e-6
        assert abs(rows["ate"].p0_curve) > 1e-3


class TestThePlainRemainderAgreesWithTierOnesQuadrature:
    """Two routes to one population integral, sharing no code.

    :func:`~benchmarks.drtmle_remainder.plain_remainder` averages over a companion draw at
    the arrays the fit produced; :func:`~benchmarks.drtmle_injection.exact_remainder`
    integrates prescribed functions on the Sobol rule the truth uses.  They agree only if
    both are the definition, which is the strongest check this module has -- and it is
    available *because* Tier 1's nuisances are prescribed, which is what that tier is for.
    """

    @pytest.mark.parametrize("cell", injection.CELLS)
    def test_the_two_agree_to_the_evaluation_rules_own_error(self, cell: str, fitted: Any) -> None:
        # `q-drift` is the module's own fit rather than a second one: a doubly-robust fit is
        # the expensive thing in this module and refitting a configuration a fixture already
        # holds is the commonest waste `CLAUDE.md` names.
        result, weights = fitted if cell == "q-drift" else _fit(cell)
        measured = remainder.plain_remainder(
            result, injection.base_law(), result.config.g_bounds, weights
        )
        exact = injection.exact_remainder(cell, N)
        # A Monte Carlo window, not a numerical one: the companion is a rule of
        # EVALUATION_N rows and the quadrature is exact, so what separates them is the
        # rule's own error.  Stated relative to the quantity rather than absolutely, and
        # kept at the width the i.i.d. draw needed -- how much *narrower* the deterministic
        # rule is is the next test's claim, and a window tightened here would hide it.
        for key in ("r2_1", "r2_0", "r2_ate"):
            assert measured[key] == pytest.approx(exact[key], rel=0.15, abs=0.01), (cell, key)

    def test_the_deterministic_rule_is_the_closer_of_the_two(
        self, fitted: Any, sampled: Any
    ) -> None:
        """A **comparative** claim at one row count, so there is no constant to tune.

        Both companions hold ``2 * POINTS`` rows and both are compared against the same
        Sobol quadrature over the prescribed sequence.  If the deterministic rule were merely
        a reshuffling of the draw the two disagreements would be of a size; it removes the
        outcome noise from the integrand outright, so it is not.
        """
        law, exact = injection.base_law(), injection.exact_remainder("q-drift", N)
        gaps = {}
        for tag, (result, weights) in (("sobol", fitted), ("draw", sampled)):
            measured = remainder.plain_remainder(result, law, result.config.g_bounds, weights)
            gaps[tag] = abs(measured["r2_ate"] - exact["r2_ate"])

        assert gaps["sobol"] < 0.25 * gaps["draw"], gaps


class TestTheFoldWeightingIsTheEstimators:
    """Section 5's convention, and it is not the uniform one."""

    def test_the_curve_is_averaged_at_the_folds_own_weights(self, fitted: Any) -> None:
        evaluation = fitted[0].repeats[0].fluctuations["mean"].reduction.evaluation
        weights = evaluation.fold_weights
        assert weights.size == evaluation.n_folds
        np.testing.assert_allclose(weights.sum(), 1.0)
        # Not uniform in general, and this fit's split is not balanced to the row.
        counts = np.asarray(evaluation.fold_sizes, dtype=float)
        np.testing.assert_allclose(weights, counts / counts.sum())

    def test_a_fit_without_a_companion_is_refused_rather_than_approximated(self) -> None:
        # Deliberately the smallest fit that runs: what is being checked is that the two
        # functions refuse, which no sample size makes truer.
        small = 150
        dgp = injection.base_law()
        frame, _ = dgp.sample(small, seed=SEED)
        plain = (
            DRTMLE(
                **injection.settings("q-drift", small),
                reduced_outcome_learner="glm",
                reduced_treatment_learner="glm",
                random_state=3,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        for call in (
            lambda: remainder.corrected_remainder(plain, dgp),
            lambda: remainder.plain_remainder(plain, dgp, plain.config.g_bounds),
        ):
            with pytest.raises(ValueError, match=r"companion|evaluation"):
                call()


class TestTheBinnedLimitsSayHowWellTheyResolved:
    """The branch columns are approximations and the module reports their own error."""

    def test_a_branch_is_dropped_when_it_does_not_separate_from_its_discretisation(
        self, rows: dict[str, Any]
    ) -> None:
        for row in rows.values():
            resolved = np.isfinite(row.branch_q)
            assert resolved == (
                row.branch_error <= max(abs(row.branch_q), abs(row.branch_g)) if resolved else True
            )
            assert np.isfinite(row.branch_error)

    def test_the_conditional_mean_is_a_conditional_mean(self) -> None:
        """The quadrature the ``0n`` limits are estimated by, checked on a law it knows.

        With the target an exact function of the design, a binned average returns it back to
        the bin width -- and with the target independent of the design it returns the
        marginal mean.  Two cases, because a helper that returned the target unchanged would
        pass the first and fail the second.
        """
        rng = np.random.default_rng(11)
        design = rng.normal(size=4_000)
        exact = remainder.conditional_mean(design, design, bins=40)
        assert np.corrcoef(exact, design)[0, 1] > 0.99

        noise = rng.normal(size=4_000)
        flat = remainder.conditional_mean(noise, design, bins=40)
        assert float(np.std(flat)) < 0.2 * float(np.std(noise))

    def test_a_masked_cell_with_no_eligible_row_falls_back_rather_than_returning_nan(
        self,
    ) -> None:
        rng = np.random.default_rng(12)
        design = rng.normal(size=500)
        target = rng.normal(size=500)
        mask = design > np.quantile(design, 0.9)
        values = remainder.conditional_mean(target, design, mask=mask, bins=20)
        assert np.all(np.isfinite(values))

    def test_uniform_weights_are_the_unweighted_average(self) -> None:
        """The i.i.d. path must be bit for bit what it was, since C3c's numbers came off it."""
        rng = np.random.default_rng(13)
        design, target = rng.normal(size=800), rng.normal(size=800)
        plain = remainder.conditional_mean(target, design, bins=16)
        uniform = remainder.conditional_mean(target, design, bins=16, weights=np.ones(800))

        np.testing.assert_array_equal(plain, uniform)

    def test_a_weight_selects_which_rows_the_cell_average_is_taken_over(self) -> None:
        """Two rows per cell, one weighted out: the average is the survivor, exactly.

        The smallest law on which a weighted conditional mean has an answer that can be
        written down -- and the one a ``bincount`` that weights the totals but not the counts
        gets wrong by the weight itself.
        """
        design = np.array([0.0, 0.0, 1.0, 1.0])
        target = np.array([2.0, 6.0, 10.0, 30.0])
        weights = np.array([1.0, 0.0, 3.0, 1.0])

        values = remainder.conditional_mean(target, design, bins=2, weights=weights)

        np.testing.assert_allclose(values, [2.0, 2.0, 15.0, 15.0])


class TestTheDeterministicRuleIsTheSameIntegral:
    """The quadrature companion, checked against the law before any fit is involved.

    ``benchmarks/drtmle_remainder.quadrature_frame`` claims to be a *rule* for the same
    population expectation the i.i.d. draw estimates.  Three of the four claims below need no
    estimator at all, which is the point: an identity that holds of the construction cannot
    be confounded with anything the fit did.
    """

    def test_the_grids_own_truth_is_the_laws_at_the_full_rule(self) -> None:
        """``truth_at`` duplicates three lines of ``_integrate`` and this is what pins them.

        Bit for bit at :math:`2^{18}`, because the duplication exists only so that a
        *coarser* grid can be asked for -- if the two ever disagree at the full rule, the
        coarser one is answering for a different law.
        """
        dgp = injection.base_law()
        measured, law = remainder.truth_at(dgp, 2**18), dgp.truth()

        for key in ("ey1", "ey0", "ate"):
            assert measured[key] == law[key], key

    def test_the_weighted_frame_carries_the_laws_own_mechanism(self) -> None:
        frame, weights = remainder.quadrature_frame(injection.base_law(), 1_024)
        treatment = np.asarray(nw.from_native(frame, eager_only=True)["A"].to_numpy(), float)
        truth_g = injection.base_law().propensity(injection.base_law().quadrature(1_024))

        assert np.average(treatment, weights=weights) == pytest.approx(float(np.mean(truth_g)))

    def test_the_frame_integrates_an_inverse_probability_estimand_exactly(self) -> None:
        """The whole construction in one line, and the one that fails if ``Y`` sits at the
        wrong arm.

        :math:`E[Y \\cdot 1\\{A = 1\\} / g_0(1 \\mid W)]` is :math:`E[\\bar Q_0(1, W)]` for the
        law, and on this frame it is that identity *exactly* rather than to a Monte Carlo
        error -- which is what "the Monte Carlo is out of two of the three coordinates" says
        when it is written as arithmetic.
        """
        dgp, points = injection.base_law(), 1_024
        frame, weights = remainder.quadrature_frame(dgp, points)
        native = nw.from_native(frame, eager_only=True)
        outcome = np.asarray(native["Y"].to_numpy(), dtype=float)
        treatment = np.asarray(native["A"].to_numpy(), dtype=float)
        one = np.asarray(dgp.propensity(dgp.quadrature(points)), dtype=float)
        mechanism = np.stack([one, 1.0 - one], axis=1).reshape(-1)

        estimand = np.average(outcome * (treatment == 1.0) / mechanism, weights=weights)

        assert estimand == pytest.approx(remainder.truth_at(dgp, points)["ey1"], abs=1e-12)

    def test_a_coarser_grid_is_the_finer_ones_first_rows(self) -> None:
        """:func:`quadrature_frame`'s interleaving contract, which the ladder rests on."""
        dgp = injection.base_law()
        coarse, coarse_weights = remainder.quadrature_frame(dgp, 256)
        fine, fine_weights = remainder.quadrature_frame(dgp, 1_024)
        head = nw.from_native(fine, eager_only=True).head(512)

        np.testing.assert_array_equal(coarse_weights, fine_weights[:512])
        for column in nw.from_native(coarse, eager_only=True).columns:
            np.testing.assert_array_equal(
                nw.from_native(coarse, eager_only=True)[column].to_numpy(),
                head[column].to_numpy(),
            )

    def test_a_law_the_rule_cannot_be_built_for_is_refused(self) -> None:
        """Each refusal names what the derivation would need; none returns a plausible frame."""
        with pytest.raises(ValueError, match="binomial"):
            remainder.quadrature_frame(binary_outcome_dgp(), 64)
        with pytest.raises(ValueError, match="hidden variables"):
            remainder.quadrature_frame(clustered_dgp(), 64)

    def test_the_two_rules_agree_inside_the_noisier_ones_error(
        self, fitted: Any, sampled: Any
    ) -> None:
        """Two routes to one population integral, at the same row count and one fitted law.

        The window is the i.i.d. companion's **own** standard error, read off its own rowwise
        curve rather than taken from a constant -- so this is a comparison against a measured
        tolerance and not against a chosen one.
        """
        determined = _rows(*fitted)["ate"].p0_curve
        drawn = _rows(*sampled)["ate"].p0_curve
        error = _rows(*sampled)["ate"].companion_se / float(np.sqrt(N))

        assert abs(determined - drawn) < 1.96 * error, (determined, drawn, error)

    def test_the_rules_own_error_is_orders_below_the_draws(self, fitted: Any, sampled: Any) -> None:
        """E1's whole claim, as a number: the grid's error is not the column's spread.

        ``companion_halving`` is each rule's own witness -- how far ``P_0 D-hat`` moves when
        half the rows are taken away.  On the deterministic grid that is discretisation; on
        the draw it is noise, and the two are not the same size.
        """
        determined = _rows(*fitted)["ate"].companion_halving
        drawn = _rows(*sampled)["ate"].companion_halving

        assert determined < 0.05 * drawn, (determined, drawn)


class TestTheCurveIsAffineInTheOutcome:
    """The premise :func:`~benchmarks.drtmle_remainder.quadrature_frame` rests on.

    Substituting :math:`E_0[Y \\mid A, W]` for :math:`Y` is exact **only** because the curve
    is affine in it.  Put ``Y`` inside any nonlinearity -- a bounded-outcome clip, a squared
    residual, a robust loss -- and :math:`P_0\\hat D` becomes silently wrong while every other
    check in this module goes on passing, because none of them varies the outcome.  This is
    the one instrument that can watch that premise break, and it is why
    :func:`~benchmarks.drtmle_remainder.corrected_curve` takes an ``outcome`` argument at all.
    """

    #: How far the outcome is moved between the three evaluations, on the ``[0, 1]`` scaled
    #: outcome.  **A whole scaled unit, and the size is the test**: a small step probes
    #: *local* linearity only, so a clip or a bound anywhere outside its reach passes.  That
    #: was measured rather than reasoned -- at ``0.013`` a residual clip at ``+/-0.4`` was
    #: applied and this test did not notice.  Anything that puts ``Y`` inside a nonlinearity
    #: has to bite somewhere in ``[Y, Y + 2]`` to be a nonlinearity at all.
    STEP = 1.0

    def test_the_second_difference_in_the_outcome_is_zero(self, fitted: Any) -> None:
        result, _ = fitted
        base = _scaled_outcome(result)

        low = remainder.corrected_curve(result, outcome=base)
        mid = remainder.corrected_curve(result, outcome=base + self.STEP)
        high = remainder.corrected_curve(result, outcome=base + 2 * self.STEP)

        for arm in low:
            for a, b, c in zip(low[arm], mid[arm], high[arm], strict=True):
                np.testing.assert_allclose(a - 2 * b + c, 0.0, rtol=0.0, atol=1e-12)

    def test_the_first_difference_is_not_zero_either(self, fitted: Any) -> None:
        """The required non-vacuous control: a curve ignoring ``Y`` would pass the test above."""
        result, _ = fitted
        base = _scaled_outcome(result)

        low = remainder.corrected_curve(result, outcome=base)
        high = remainder.corrected_curve(result, outcome=base + self.STEP)

        assert (
            max(
                float(np.abs(h - low_).max())
                for arm in low
                for low_, h in zip(low[arm], high[arm], strict=True)
            )
            > 1e-3
        )


class TestAPrefixIsTheSameFitAtACoarserGrid:
    """Why the convergence ladder is one fit rather than one fit per rung.

    Because :func:`~benchmarks.drtmle_remainder.quadrature_frame` interleaves the arms and
    :meth:`~cleverly.datasets.DGP.quadrature`'s grids are nested, the first ``2 * k`` rows of
    a companion **are** the grid at ``k`` points -- so ``limit=`` reads a coarser rung off a
    finer fit.  If that were only approximately true, the movement between two rungs would be
    a difference between two fits and would have to be argued bit-identical instead of being
    the quadrature.
    """

    def test_the_prefix_equals_a_companion_built_at_that_grid(self, fitted: Any) -> None:
        coarse = POINTS // 4
        sliced = _rows(*fitted, points=coarse, limit=2 * coarse)["ate"]
        refitted = _rows(*_fit("q-drift", points=coarse), points=coarse)["ate"]

        for column in ("p0_curve", "remaining", "r2", "branch_q", "branch_g"):
            assert getattr(sliced, column) == getattr(refitted, column), column
