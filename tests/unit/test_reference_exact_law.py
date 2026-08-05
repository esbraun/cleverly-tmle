r"""E2's first fidelity gate: the reference provider against a law that knows the answer.

``benchmarks/drtmle_reference.py`` builds the three reduced regressions at their population
limits on a *continuous* law, where the result is a **numerical reference and not an oracle**
-- a smoothing over one index, at a finite point count, neither of which that module bounds.
``docs/roadmap.md``'s **E2** says what has to bound them before any paired number is read, and
names this one first:

    an exact-law control where the conditioning is discrete and the answer is a finite sum

This is that control.  On :mod:`tests.discrete_law` the conditioning variable takes finitely
many values, so a conditional expectation *is* a finite sum and
:class:`~benchmarks.drtmle_reference.SaturatedCells` computes it **exactly** rather than to a
tolerance.  What the control then says is that the provider around the smoother -- the
weights, the ``| A = a`` mask, the arm columns, the recomputation at the current targeted pair
-- computes the reduced regression and not some other quantity that happens to be smooth.

**Why that is not what the sibling modules already say.**
:mod:`tests.unit.test_oracle_reductions` compares the *library's* saturated learner against
the same longhand, and :mod:`tests.unit.test_drtmle_reference` pins the provider's structure
on the continuous law -- its routing, its weights, its linearity -- against no known answer.
Neither runs the **benchmark** provider against a number the law supplies.  Two of the three
integrals E2 relies on are exact by construction rather than by approximation (``qr``'s
``| A = a`` is a *weight* and ``gr1``'s target integrates to :math:`g_0` through those
weights), and this module is where that construction is checked rather than argued.

**The nuisances are wrong on purpose and they are wrong in two ways.**  At correct nuisances
:math:`Q_r` and :math:`g_{r,2}` vanish row by row -- lesson 2, the degeneracy every instrument
here goes blind in -- so both parametrisations below take a misspecified pair.  The two differ
in something this module needs and the sibling does not: ``distinct`` gives every covariate
cell its own nuisance *value*, which makes each reduction a **relabelling** of :math:`W`, and
``tied`` ties two cells, which makes it a genuine **pooling**.  Only the second can see the
``| A = a`` mask at all, because a singleton cell's weighted mean does not depend on its
weight.  That is :data:`tests.unit.test_remainder_drtmle.TIED_G`'s own argument, transported.

**What this control cannot see**, written down because a suite records what it caught and not
what it is blind to:

* **fold routing.**  The fit is uncross-fitted, so the companion has one outer fold and every
  routing is the same routing.  That half is pinned in
  :class:`tests.unit.test_drtmle_reference.TestTheProviderRoutesFoldsAndArms`, which runs on
  *fitted* primaries for the matching reason -- on injected ones every fold's companion copy
  is identical to the bit and a routing test passes against any routing whatever.
* **the quadrature.**  There is no grid here: the companion enumerates the law's own support.
  A finite point count is what the randomisation budget in ``benchmarks/drtmle_reference_study.py``
  measures, and no exact law can say anything about it.
* **the smoothing bias of the shipped reference.**  :class:`SplineProjection` is *not* exact on
  a discrete index and is shown below not to be.  What bounds it is the held-out risk, which
  needs a law with a continuum in it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks import drtmle_reference as reference
from benchmarks.drtmle_reference import (
    EqualCountBins,
    ReferenceReductionDRTMLE,
    SaturatedCells,
    SplineProjection,
    reference_reductions,
)
from benchmarks.drtmle_remainder import ARMS, Window

from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

# Imported rather than restated, as `test_oracle_reductions` imports its longhand from
# `test_remainder_drtmle`: the whole value of an exact-law control is that it is the *same*
# statement a sibling module already checks the library against, so two spellings of it would
# be free to drift and the agreement below would stop meaning anything.
from tests.unit.test_oracle_reductions import (
    ESTIMANDS,
    INERT,
    Misspecified,
    OracleReductionDRTMLE,
)
from tests.unit.test_remainder_drtmle import TIED_G, TIED_Q, WRONG_G, WRONG_Q

#: The two misspecified pairs, and the difference between them is load-bearing.  Under
#: ``distinct`` the three cells take three nuisance values, so conditioning on a nuisance is
#: conditioning on ``W`` and every reduction is a relabelling; under ``tied`` two cells share a
#: value and the reduction genuinely pools.  A control taken only at the first would be blind
#: to ``qr``'s mask and to the pooling itself.
NUISANCES = {"distinct": (WRONG_G, WRONG_Q), "tied": (TIED_G, TIED_Q)}


class ExactLaw(Misspecified):
    """The law's **own** ``G`` and ``Q``, plus the two fields the provider reads off a ``DGP``.

    :class:`~tests.unit.test_oracle_reductions.Misspecified` is a ``DGP``-shaped holder of
    declared constants and is reused rather than re-written; what it does not carry is
    ``n_latent`` and ``name``, which :func:`benchmarks.drtmle_remainder._latent` checks the
    companion's covariate count against.  Nothing else of a ``DGP`` is reached from here --
    in particular no ``quadrature``, since this law has a support to enumerate rather than a
    grid to integrate.
    """

    name = "discrete-law"
    n_latent = 1

    def __init__(self) -> None:
        super().__init__(law.G, law.Q)


def _interleave(blocks: list[np.ndarray]) -> np.ndarray:
    """One row per ``(support point, arm)`` pair, point-major.

    :func:`benchmarks.drtmle_remainder.quadrature_frame`'s contract, restated here rather than
    imported: this module builds the *discrete* analogue of that frame and the layout is part
    of what makes a :class:`~benchmarks.drtmle_remainder.Window` mean what it means.
    """
    return np.stack(blocks, axis=1).reshape(-1)


def _companion() -> tuple[pd.DataFrame, np.ndarray]:
    r"""The discrete analogue of ``quadrature_frame``: the law's support at both arms.

    Returns ``(frame, weights)``.  Every row of :func:`tests.discrete_law.frame` appears twice,
    once per arm, with ``Y`` set to :math:`\bar Q_0(a, W)` and its weight set to
    :math:`g_0(a \mid W)` -- which is what the quasi-random companion carries and what
    :func:`benchmarks.drtmle_reference._check_the_weights_are_the_laws` refuses anything else
    for.

    **Why that makes a saturated cell mean the law's own conditional expectation, exactly.**
    The sample realises the law exactly, so each support point appears with multiplicity
    proportional to :math:`P(W)`; a cell of the conditioning index is therefore a set of rows
    carrying mass :math:`P(W) g_0(a|W)`, which is the measure
    :func:`tests.unit.test_remainder_drtmle._reduced` sums against.  Reading it off the three
    targets:

    * ``qr``'s ``| A = a`` is the fit mask, so its cell mean is weighted by
      :math:`P(W) g_0(a|W)` -- the mechanism inside the conditioning, not beside it;
    * ``gr1``'s target is the bare indicator, and the two rows of one support point carry
      :math:`g_0(1|W)` and :math:`g_0(0|W)`, so a weighted average over a cell is
      :math:`E[g_0(a|W) \mid \text{cell}]` -- that object's definition and not a proxy;
    * ``gr2``'s is :math:`(1_a - \hat g)/\hat g`, whose weighted average is
      :math:`E[(g_0 - \hat g)/\hat g \mid \text{cell}]` by the same substitution.

    So there is no quadrature here and no sampling error: what a finite point count costs on
    the continuous law costs nothing on this one, which is the whole reason the control exists.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy(dtype=float)
    cells = np.rint(covariate).astype(int)
    payload = pd.DataFrame(
        {
            "W": np.repeat(covariate, len(ARMS)),
            "A": _interleave([np.full(covariate.size, arm) for arm in ARMS]),
            "Y": _interleave([law.Q[cells, int(arm)] for arm in ARMS]),
        }
    )
    weights = _interleave([law.G[cells] if arm == 1.0 else 1.0 - law.G[cells] for arm in ARMS])
    return payload, weights


COMPANION, WEIGHTS = _companion()
WINDOW = Window(0, COMPANION.shape[0])


def _settings(pair: str) -> dict[str, Any]:
    """One fit's configuration at a declared misspecified pair.

    ``cross_fit=False`` and oracle primary learners for
    :mod:`tests.unit.test_oracle_reductions`' reason: neither learner learns from the data, so
    out-of-fold prediction would add fold bookkeeping to a fit whose nuisances are declared,
    and it is what makes every array a function of ``W`` alone.  The reduced *learner* is named
    and immaterial -- ``super()._nuisances`` fits one before either subclass replaces it, and
    naming the same one in both arms is what keeps the provider the only difference between
    them.
    """
    dgp = Misspecified(*NUISANCES[pair])
    return {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "reduced_outcome_learner": "glm",
        "reduced_treatment_learner": "glm",
        "estimands": ESTIMANDS,
        "g_bounds": INERT,
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }


def _reference_fit(pair: str, smoother: Any) -> Any:
    return (
        ReferenceReductionDRTMLE(
            dgp=ExactLaw(),
            reference=smoother,
            window=WINDOW,
            row_weights=WEIGHTS,
            evaluation=COMPANION,
            **_settings(pair),
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module", params=sorted(NUISANCES))
def fits(request: Any) -> dict[str, Any]:
    """The two arms of the control, at one misspecified pair.

    **The oracle arm carries no companion and that is deliberate rather than an oversight.**
    :class:`~tests.unit.test_oracle_reductions.OracleReductionDRTMLE`'s ``refit`` returns an
    empty companion tuple, because nothing there needed one; handing it an ``evaluation=``
    frame would ask it for per-fold companion reductions it does not build.  Nothing is lost,
    because the companion contributes to no fit, no fold and no score -- which is not an
    assumption here but the property ``tests/unit/test_drtmle_companion.py`` pins bit for bit.
    """
    return {
        "oracle": OracleReductionDRTMLE(**_settings(request.param))
        .fit(law.frame(), outcome="Y", treatment="A")
        .single(),
        "reference": _reference_fit(request.param, SaturatedCells()),
        "pair": request.param,
    }


def _reduced_of(fit: Any) -> Any:
    return fit.repeats[0].fluctuations["mean"].reduction.reduced


class TestTheProviderReproducesTheLawsOwnReductions:
    """The control proper, and nothing else in E2 means anything without it."""

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_each_regression_agrees(self, fits: dict[str, Any], name: str) -> None:
        """Array for array, at the **exit** state of two independently run alternations.

        A stronger statement than a single call compared at fixed nuisances, and the reason is
        the alternation: each round's reductions decide the next round's covariates, so
        agreement here says the two providers took the same path and not merely that they
        agree once.  ``atol`` with ``rtol=0`` for the reason every exact-law module in this
        repository states it explicitly -- a relative tolerance would loosen the check to
        roughly ``1e-6`` on the larger entries while still reading as exact.
        """
        np.testing.assert_allclose(
            getattr(_reduced_of(fits["reference"]), name),
            getattr(_reduced_of(fits["oracle"]), name),
            rtol=0,
            atol=1e-10,
        )

    def test_and_the_two_fits_report_the_same_estimate(self, fits: dict[str, Any]) -> None:
        """Which follows from the above, and is worth asserting apart: equal inputs, equal answers."""
        for name in ESTIMANDS:
            assert fits["reference"].estimates[name].psi == pytest.approx(
                fits["oracle"].estimates[name].psi, abs=1e-9
            )

    def test_the_reductions_are_not_trivially_zero(self, fits: dict[str, Any]) -> None:
        """Anti-vacuity, and it is the degeneracy this whole variant's instruments live in.

        At correct nuisances ``qr`` and ``gr2`` vanish row by row and every assertion above
        would pass against a provider that returned zeros.  ``gr1`` is a probability and does
        not vanish, so it is checked away from its own trivial value rather than away from nought.
        """
        produced = _reduced_of(fits["reference"])
        assert float(np.max(np.abs(produced.qr))) > 1e-3
        assert float(np.max(np.abs(produced.gr2))) > 1e-3
        assert float(np.min(produced.gr1)) > 1e-3

    def test_the_alternation_actually_ran(self, fits: dict[str, Any]) -> None:
        """Anti-vacuity for the trajectory claim: at one round it is a single-call comparison."""
        reduction = fits["reference"].repeats[0].fluctuations["mean"].reduction
        assert reduction.rounds > 1
        assert float(np.max(np.abs(np.asarray(reduction.epsilon, dtype=float)))) > 0.0


class TestTheMaskAndTheWeightsAreLoadBearing:
    """Two mutations, **run** rather than described, on the arm that can see them.

    Both are silent failures: every array stays in range, no solver complains, and the limit
    answers for the wrong measure to five decimals.  ``tied`` is the parametrisation that can
    see the first of them at all -- with three distinct nuisance values every conditioning cell
    is a singleton and a weighted mean does not depend on its weight, so a mask dropped there
    changes nothing and a test taken there would pass against the mutation.
    """

    def test_dropping_qrs_arm_mask_moves_qr_and_nothing_else(
        self, fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r""":math:`Q_r` alone carries a mask, and it is the whole of the ``| A = a``.

        Watched to fail rather than argued: the mutation is applied to the shipped
        :func:`benchmarks.drtmle_reference._fit_mask` and the reductions are rebuilt at the same
        state.  ``gr1`` and ``gr2`` must be untouched, which is the half that says the mask is
        in the right place rather than merely present somewhere.
        """
        if fits["pair"] != "tied":
            pytest.skip("a singleton conditioning cell cannot see a weight, let alone a mask")
        state = fits["reference"].nuisance

        def build() -> Any:
            return reference_reductions(
                state,
                dgp=ExactLaw(),
                reference=SaturatedCells(),
                window=WINDOW,
                row_weights=WEIGHTS,
                g_bounds=INERT,
            )[0]

        straight = build()
        monkeypatch.setattr(reference, "_fit_mask", lambda name, indicator: None)
        mutated = build()

        assert float(np.max(np.abs(straight.qr - mutated.qr))) > 1e-3
        np.testing.assert_allclose(straight.gr1, mutated.gr1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(straight.gr2, mutated.gr2, rtol=0, atol=1e-12)

    def test_uniform_weights_are_refused(self, fits: dict[str, Any]) -> None:
        r"""A weight vector that is not :math:`g_0(a|W)` integrates against the wrong measure.

        The guard is the same one a draw block trips on the continuous law
        (:class:`tests.unit.test_drtmle_reference.TestTheWeightsMustBeTheLaws`); what this adds
        is that it fires on the law where the correct weights are the law's own cell
        probabilities rather than a quadrature's, so "the weights are the mechanism" is checked
        against a case where a plausible alternative -- one weight per row -- exists.
        """
        with pytest.raises(ValueError, match="not the law's own"):
            reference_reductions(
                fits["reference"].nuisance,
                dgp=ExactLaw(),
                reference=SaturatedCells(),
                window=WINDOW,
                row_weights=np.ones(WINDOW.rows),
                g_bounds=INERT,
            )


class TestTheControlDiscriminates:
    """A control that passes for every smoother is not a control.

    Each of these is a smoother the exact-law gate has to **reject**, and each fails for a
    different reason: a regressogram pools cells the law separates, and a spline basis is not
    exact on an index with three values however many knots it is given.  The second is why
    :class:`~benchmarks.drtmle_reference.SaturatedCells` exists rather than the control reusing
    the shipped reference -- which would be comparing an approximation against itself.
    """

    @pytest.mark.parametrize("smoother", [EqualCountBins(2), SplineProjection(8)])
    def test_a_smoother_that_is_not_exact_here_does_not_reproduce_the_law(
        self, fits: dict[str, Any], smoother: Any
    ) -> None:
        produced = _reduced_of(_reference_fit(fits["pair"], smoother))
        exact = _reduced_of(fits["oracle"])
        moved = max(
            float(np.max(np.abs(getattr(produced, name) - getattr(exact, name))))
            for name in ("qr", "gr1", "gr2")
        )
        assert moved > 1e-2


class TestTheEstimateWouldNotHaveRejectedIt:
    """The measurement E2's gates are designed around, on a law where the truth is known.

    ``benchmarks/drtmle_reference.py`` records it from the continuous law: an eight-bin
    reference reached a ``psi`` *nearer the truth* than the good one while being a far worse
    estimate of the reduced functions.  Here the same statement is exact -- the truth is
    :data:`tests.discrete_law.TRUTH` and the sample realises the law -- and it is what says a
    fidelity gate must read the **function**.
    """

    def test_a_coarse_reference_moves_the_function_far_and_the_estimate_barely(self) -> None:
        """Read at ``distinct``, where the reduction is a relabelling of ``W``.

        That is the configuration the trap is sharpest in and it is not a convenience: with
        every cell separated, a reduction's *value* barely reaches ``psi`` whatever the
        smoother does, so an estimate-reading gate has nothing to see.  The comparison is
        against the fit's own standard error rather than against an absolute size, because a
        gap far inside one is a gap no study of this estimator could ever resolve.
        """
        exact = _reference_fit("distinct", SaturatedCells())
        coarse = _reference_fit("distinct", EqualCountBins(2))

        good, bad = _reduced_of(exact), _reduced_of(coarse)
        function_gap = float(np.max(np.abs(good.qr - bad.qr))) / float(np.max(np.abs(good.qr)))
        estimate = exact.estimates["ate"]
        estimate_gap = abs(coarse.estimates["ate"].psi - estimate.psi) / float(estimate.std_error)

        assert function_gap > 0.5, "the coarse reference must be a badly wrong function"
        assert estimate_gap < 0.05, "and its estimate must be inside a twentieth of one se"
