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

import numpy as np
import pytest
from benchmarks import drtmle_injection as injection
from benchmarks import drtmle_remainder as remainder

from cleverly import DRTMLE

#: ``600`` rather than something smaller, and that is not a typo: a doubly-robust fit at
#: ``400`` rows takes *longer* than one at ``600`` -- noisier nuisances loosen the coupling
#: and lengthen the alternation, which ``tests/unit/test_drtmle_fit.py`` already records and
#: C1 measured at ``16.4s`` against ``5.6s``.
N = 600

#: Rows of the evaluation draw, and the one constant here that is a **statistical** choice.
#: ``P_0 D-hat`` is a quadrature, so its error is ``sd(D)/sqrt(m)`` and it lands **directly**
#: in ``R_remaining``: at ``m = 1,500`` that error is ``0.026`` against a remainder of
#: ``0.007``, so a single replicate's column would be mostly noise.  The study averages it
#: down across replicates, drawing an independent evaluation sample per draw; a test cannot,
#: so it pays for the accuracy up front.
EVALUATION_N = 4_000
SEED = 7


def _fit(cell: str, *, oracle: bool = False, evaluation_n: int = EVALUATION_N) -> Any:
    dgp = injection.base_law()
    frame, _ = dgp.sample(N, seed=SEED)
    evaluation = remainder.evaluation_frame(dgp, evaluation_n, seed=99_991)
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
    return (
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


#: A size at which the injected drift ``n^(-alpha)`` is below floating-point resolution
#: against the outcome, so the "sequence" is the truth exactly.  ``1e64 ** -0.25 = 1e-16``.
_HUGE = 10**64


@pytest.fixture(scope="module")
def fitted() -> Any:
    return _fit("q-drift")


@pytest.fixture(scope="module")
def rows(fitted: Any) -> dict[str, Any]:
    """The remainder columns of one fit, computed once.

    Module-scoped because every branch column is a pair of binned quadratures over the whole
    evaluation draw, per arm and per fold -- recomputing them per test is the commonest
    waste ``CLAUDE.md`` names, and nothing here mutates them.
    """
    return {
        row.estimand: row
        for row in remainder.remainder_rows(
            fitted, injection.base_law(), n=N, bounds=fitted.config.g_bounds
        )
    }


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
        values = remainder.plain_remainder(at_truth, injection.base_law(), at_truth.config.g_bounds)
        assert abs(values["r2_ate"]) < 1e-9

    def test_the_corrected_remainder_is_small_against_the_bias_it_would_otherwise_carry(
        self, at_truth: Any
    ) -> None:
        """Not exactly zero, and the reason is worth stating rather than absorbing.

        With one nuisance exactly right the *expansion*'s remainder vanishes, but
        ``R_remaining`` is computed at a **finite** evaluation draw, so what is left is that
        draw's quadrature error -- an :math:`O(m^{-1/2})` quantity with no ``n`` in it.  The
        bar is therefore a Monte Carlo one, and it is stated as a share of the estimator's
        own standard error rather than as an absolute number.
        """
        rows = {
            row.estimand: row
            for row in remainder.remainder_rows(
                at_truth, injection.base_law(), n=N, bounds=at_truth.config.g_bounds
            )
        }
        standard_error = at_truth.estimates["ate"].std_error
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
    def test_the_two_agree_to_the_evaluation_draws_own_error(self, cell: str, fitted: Any) -> None:
        # `q-drift` is the module's own fit rather than a second one: a doubly-robust fit is
        # the expensive thing in this module and refitting a configuration a fixture already
        # holds is the commonest waste `CLAUDE.md` names.
        result = fitted if cell == "q-drift" else _fit(cell)
        measured = remainder.plain_remainder(result, injection.base_law(), result.config.g_bounds)
        exact = injection.exact_remainder(cell, N)
        # A Monte Carlo window, not a numerical one: the companion is a draw of
        # EVALUATION_N rows and the quadrature is exact, so what separates them is the
        # draw's own standard error.  Stated relative to the quantity rather than absolutely.
        for key in ("r2_1", "r2_0", "r2_ate"):
            assert measured[key] == pytest.approx(exact[key], rel=0.15, abs=0.01), (cell, key)


class TestTheFoldWeightingIsTheEstimators:
    """Section 5's convention, and it is not the uniform one."""

    def test_the_curve_is_averaged_at_the_folds_own_weights(self, fitted: Any) -> None:
        evaluation = fitted.repeats[0].fluctuations["mean"].reduction.evaluation
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
