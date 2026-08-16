r"""What does a fit do when the reduced regressions are *exactly* right?

The validation plan asked whether a sweep's failures persist "when the reductions are handed
the **oracle** values", and sized that as free "because the datasets already know their
truth".  They do not.  A reduction is a conditional expectation given a
**fitted** object -- :math:`Q_r(a, W) = E[\bar Q_0 - \bar Q^* \mid \hat g(a|W)]`, and the two
reduced mechanisms given :math:`\hat{\bar Q}(a, W)` -- so its truth is a property of the
estimator's own arrays and not of the process, and no closed form or fresh draw from a
continuous DGP supplies it.  On the continuous processes the sweep therefore carries
``--reduced-learner`` as a labelled proxy, and the full construction remains future work.

**Here it is buildable, and this module builds it.**  On the exact law the conditioning
variables take three values, the law's cell probabilities are known, and the conditional
expectations are finite sums -- which is what
:func:`tests.unit.test_remainder_drtmle._reduced` already computes.  What was missing is the
*end-to-end* half: handing those values to a real alternation and seeing what a fit does with
them.  :attr:`~cleverly.estimators.targeting.ReductionSpec.refit` is the seam, and it is a
callable for exactly this reason -- the reductions arrive as a closure rather than as an
import, so a subclass can supply a different one without the loop knowing.

Three fits, and the third is the one that answers §4's question:

* **oracle** -- the reductions are the law's own conditional expectations at the *current*
  targeted pair, recomputed every round as the real ones are;
* **saturated** -- :class:`~tests.discrete_law_longitudinal.CellMeans`, which on a sample that
  realises the law exactly estimates that same conditional expectation.  It must reproduce
  the oracle array for array, and that equality is the **control**: it is what says the
  injection computes the reduction rather than something else that happens to be smooth;
* **glm** -- a reduction that is genuinely *wrong*, being linear in a design that is not.

The reading, and it is the reason the arm was wanted: under a wrong reduction the fit still
**solves all three equations** -- the score check and the state identities pass -- while
:math:`\hat\Psi` moves.  So a sweep fit that fails its *scores* cannot be blamed on its
reductions being noisy, which is precisely the discrimination §4 asked for, and the
`--reduced-learner` proxy is measuring a real effect on `psi` rather than on convergence.

Two things this deliberately does not do.  It does not run on the continuous processes, where
the oracle does not exist yet. And it does not take the
comparison at *correct* nuisances, where :math:`Q_r` and :math:`g_{r,2}` vanish row by row and
every arm of it would agree for the wrong reason -- lesson 2, the degeneracy this variant's
instruments go blind in.  The nuisances here are wrong on purpose.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly import DRTMLE
from cleverly.estimators.reduced import ReducedSet
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.discrete_law_longitudinal import CellMeans

# Imported rather than re-derived, exactly as `tests/unit/test_reduced_regressions.py`
# imports them: two modules that wrote the same longhand twice would be free to drift, and
# the whole value of an oracle is that it is the *same* statement the arithmetic module
# checks `fit_reduced` against.
from tests.unit.test_remainder_drtmle import WRONG_G, WRONG_Q, _reduced

#: The estimands ``DRTMLE`` reports; ``att``/``atc`` are refused by name.
ESTIMANDS = ("ey1", "ey0", "ate")

#: A truncation that never binds, so nothing here measures bounded-mechanism centring
#: is ``tests/unit/test_drtmle_fit.py``'s.  Copied from
#: :mod:`tests.unit.test_influence_gateaux_drtmle`, as the sibling modules copy their
#: constants, so they disagree about nothing except what they assert.
INERT = (1e-6, 1.0 - 1e-6)


class Misspecified:
    """A ``DGP``-shaped object returning declared constants, for the two oracle learners.

    Copied from :mod:`tests.unit.test_influence_gateaux_drtmle`; only ``propensity`` and
    ``outcome_mean`` are read.
    """

    def __init__(self, g: np.ndarray, q: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=float)
        self.q = np.asarray(q, dtype=float)

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return self.g[self._index(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        return self.q[self._index(covariates), int(arm)]


def _covariate() -> np.ndarray:
    """Which covariate cell each row belongs to."""
    return law.frame()["W"].to_numpy().astype(int)


def _per_cell(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """A row-level array that is constant within each cell, as one value per cell.

    The constancy is **asserted rather than assumed**.  It holds because the primary
    nuisances are oracle learners reading ``W`` alone and the fit is uncross-fitted, so every
    array the alternation produces is a function of ``W``; if a later edit made it false, an
    oracle built by reading one representative row would be quietly conditioning on the wrong
    thing rather than failing.
    """
    out = np.zeros(3)
    for cell in range(3):
        rows = values[covariate == cell]
        assert np.ptp(rows) < 1e-12, f"cell {cell} is not constant: the oracle cannot read it"
        out[cell] = rows[0]
    return out


def _oracle_set(nuisance: Any, g_bounds: tuple[float, float]) -> ReducedSet:
    r"""The law's own reductions at ``nuisance``'s current targeted pair.

    A function of the *current* pair rather than a constant, which is what makes this an
    oracle for the alternation rather than for its starting point:
    :attr:`~cleverly.estimators.targeting.ReductionSpec.refit` is handed the targeted
    nuisances every round precisely because equations (9) and (10) are stated at starred
    reductions, and an oracle that ignored its argument would answer a different question --
    and would still pass a great deal, which is why that is this module's mutation.
    """
    covariate = _covariate()
    arms = nuisance.arms
    g_hat = _per_cell(np.asarray(nuisance.propensity.arm(arms[-1])), covariate)
    q_hat = np.column_stack(
        [_per_cell(np.asarray(nuisance.outcome.arms[arm]), covariate) for arm in arms]
    )
    columns: dict[str, list[np.ndarray]] = {"qr": [], "gr1": [], "gr2": []}
    for index, _ in enumerate(arms):
        for name, values in zip(("qr", "gr1", "gr2"), _reduced(g_hat, q_hat, index), strict=True):
            columns[name].append(values[covariate])
    return ReducedSet(
        qr=np.column_stack(columns["qr"]),
        gr1=np.column_stack(columns["gr1"]),
        gr2=np.column_stack(columns["gr2"]),
        arms=arms,
        g_bounds=(float(g_bounds[0]), float(g_bounds[1])),
    )


class OracleReductionDRTMLE(DRTMLE):
    """``DRTMLE`` with the reduced regressions replaced by the law's own, initial and refit.

    Both hooks, because both matter.  ``_nuisances`` supplies the set the *first* round's
    mechanism covariate reads, and ``_reduction`` supplies the closure every later refit goes
    through -- overriding only the second would leave one fitted set in the fit, which is the
    kind of half-substitution that makes a comparison mean nothing in particular.
    """

    def _nuisances(self, data, folds, scaler, config, intermediate_value, seed=None):  # type: ignore[no-untyped-def]
        base, extra = super()._nuisances(data, folds, scaler, config, intermediate_value, seed)
        if base.reduced is None:
            return base, extra
        return replace(base, reduced=_oracle_set(base, config.g_bounds)), extra

    def _reduction(self, data, nuisance):  # type: ignore[no-untyped-def]
        spec = super()._reduction(data, nuisance)
        if spec is None:
            return None
        bounds = nuisance.reduced.g_bounds
        return replace(spec, refit=lambda current: (_oracle_set(current, bounds), ()))


def _fit(estimator: type[DRTMLE], **overrides: Any) -> Any:
    """One fit on the exact law, at nuisances wrong on purpose.

    ``cross_fit=False`` and oracle primary learners for
    :mod:`tests.unit.test_influence_gateaux_drtmle`'s reason: neither learner learns from the
    data, so out-of-fold prediction would add fold bookkeeping to a fit whose nuisances are
    declared.  It is also what makes every array a function of ``W`` alone, which
    :func:`_per_cell` depends on and asserts.
    """
    dgp = Misspecified(WRONG_G, WRONG_Q)
    settings: dict[str, Any] = {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "reduced_outcome_learner": CellMeans(),
        "reduced_treatment_learner": CellMeans(),
        "estimands": ESTIMANDS,
        "g_bounds": INERT,
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return estimator(**settings).fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    """The three fits, shared: each class below reads a different part of the same set."""
    return {
        "oracle": _fit(OracleReductionDRTMLE),
        "saturated": _fit(DRTMLE),
        "glm": _fit(DRTMLE, reduced_outcome_learner="glm", reduced_treatment_learner="glm"),
    }


def _reduced_of(fit: Any) -> ReducedSet:
    return fit.repeats[0].fluctuations["mean"].reduction.reduced


class TestTheOracleIsWhatTheSaturatedLearnerEstimates:
    """The control, and it comes first because nothing below means anything without it."""

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_each_regression_agrees(self, fits, name: str) -> None:
        """Array for array, at the *exit* state of two independently run alternations.

        This is a stronger statement than ``test_reduced_regressions``' -- that one compares
        one call of ``fit_reduced`` against the longhand at fixed nuisances, where this
        compares the whole alternation's trajectory, since each round's reductions decide the
        next round's covariates.  Agreement here says the two fits took the same path.
        """
        oracle = getattr(_reduced_of(fits["oracle"]), name)
        saturated = getattr(_reduced_of(fits["saturated"]), name)

        np.testing.assert_allclose(saturated, oracle, atol=1e-10, rtol=0)

    def test_and_the_two_fits_report_the_same_estimate(self, fits) -> None:
        """Which follows, and is worth asserting separately: equal inputs, equal answers."""
        for name in ESTIMANDS:
            assert fits["oracle"].estimates[name].psi == pytest.approx(
                fits["saturated"].estimates[name].psi, abs=1e-9
            )


class TestAFitWithOracleReductionsRecoversTheTruth:
    r"""The headline, and it is a stronger claim than the module was written expecting.

    **Both primary nuisances are wrong on purpose**, so a plain TMLE has no guarantee here at
    all -- its remainder is a product of two errors and neither factor is zero.  With the
    reductions exactly right the doubly-robust fit lands on the law's own truth: ``0.66``,
    ``0.38`` and ``0.28``, to ``3.6e-08``.

    That is :mod:`tests.unit.test_remainder_drtmle`'s arithmetic arriving at the other end of
    the estimator.  That module shows on this law that one guard removes the whole first-order
    remainder at exact reductions; here the same law is *fitted*, and what the expansion says
    should happen to :math:`\hat\Psi` happens to it.  The two are independent in the way that
    matters -- one is a hand-written expansion, the other is the production alternation -- so
    this is the end-to-end confirmation the arithmetic could not give itself.

    The residual ``3.6e-08`` is not slack in the claim, it is
    Equation (9) is never solved exactly because its covariate reads the very mechanism it
    tilts, a documented [property of the alternation](../../docs/drtmle.md#the-alternation).  The
    control below is what says so -- under ``guard=("g",)`` there is no mechanism equation and
    the recovery is exact to the bit.
    """

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_estimate_is_the_truth(self, fits, name: str) -> None:
        assert fits["oracle"].estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-6)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_and_exactly_so_when_no_mechanism_equation_is_solved(self, name: str) -> None:
        """``guard=("g",)`` solves equations (8) and (10) only, and both are exact.

        Zero to the bit, which is what pins the ``3.6e-08`` above to equation (9) rather than
        to the oracle being approximate or the law being realised imperfectly.  A fit is cheap
        here -- a ``"g"`` guard refits no reductions at all.
        """
        single = _fit(OracleReductionDRTMLE, guard=("g",))

        assert single.estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    def test_all_three_scores_and_both_identities(self, fits) -> None:
        check = fits["oracle"].validation.correction_check()

        assert check.passed, check.summary()
        assert fits["oracle"].validation.score_check().passed
        for row in check.rows:
            assert abs(row.residual) < 1e-15, row.name

    def test_the_reductions_are_not_trivially_zero(self, fits) -> None:
        """The precondition, because at correct nuisances all of this vanishes row by row.

        ``Q_r`` and ``g_{r,2}`` are identically zero when the nuisances are right, so a fit
        taken there would satisfy every assertion in this module for the wrong reason --
        lesson 2.  These nuisances are wrong on purpose and this is what says so.
        """
        reduced = _reduced_of(fits["oracle"])

        assert np.max(np.abs(reduced.qr)) > 1e-3
        assert np.max(np.abs(reduced.gr2)) > 1e-3


class TestAWrongReductionMovesPsiAndNotTheScores:
    """The discrimination the arm exists to make, and the reason it is worth having.

    A `glm` reduction is linear in a design that is not, so it is wrong in a way no amount of
    data would fix.  What it does **not** do is stop the fit solving its equations: the
    alternation drives the three empirical means to zero at whatever reductions it is given,
    because they are the *directions* of the submodels rather than the targets.

    So a sweep fit whose **scores** fail is not a fit whose reductions were noisy, and
    ``--reduced-learner`` on the continuous processes is measuring a movement in `psi` rather
    than a convergence failure.  That is what a reader would otherwise have to guess.
    """

    def test_the_reduction_really_is_different(self, fits) -> None:
        oracle, glm = _reduced_of(fits["oracle"]), _reduced_of(fits["glm"])

        assert np.max(np.abs(glm.gr2 - oracle.gr2)) > 1e-3, "the glm reduction is not wrong here"

    def test_but_the_equations_are_still_solved(self, fits) -> None:
        check = fits["glm"].validation.correction_check()

        assert check.passed, check.summary()
        assert fits["glm"].validation.score_check().passed

    def test_and_psi_is_biased_against_the_known_truth(self, fits) -> None:
        r"""The size of it, in standard errors, against a truth rather than against a fit.

        Because the oracle arm recovers :math:`\Psi_0` above, this is **bias** and not merely
        a difference between two estimates: measured at ``0.80`` of a standard error on
        ``ey1``, ``0.38`` on ``ey0`` and ``0.36`` on the ``ate``.  A wrong reduction is
        therefore expensive, and expensive in the one direction an interval cannot see --
        the fit still reports every score solved.

        The bar is ``0.1`` rather than the measured value, since what the assertion is for is
        that the effect is *material*; the numbers are in the docstring because a threshold
        that tracked them would fail on the first harmless reseeding.
        """
        shifts = {
            name: abs(fits["glm"].estimates[name].psi - law.TRUTH[name])
            / fits["glm"].estimates[name].std_error
            for name in ESTIMANDS
        }

        assert max(shifts.values()) > 0.1, shifts
