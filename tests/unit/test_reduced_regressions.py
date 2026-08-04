r"""Does :func:`~cleverly.estimators.reduced.fit_reduced` fit the regressions it claims to?

:mod:`tests.unit.test_remainder_drtmle` pins the *arithmetic* the reduced-dimension
regressions have to satisfy: what a solved score equation removes from the second-order
remainder, and what it leaves.  It computes the three regressions longhand at the true law
and fits nothing.  This module checks the fitting code against exactly those numbers, by
importing them rather than deriving them a second time -- so the two modules cannot drift
into agreeing with each other about something wrong.

**Why the nuisances handed in are wrong on purpose.**  :math:`Q_r` and :math:`g_{r,2}` are
residual regressions and are identically zero when the nuisance they are residuals of is
right, row by row.  Handed the law's own nuisances, every assertion below would be
comparing zeros -- which is a real property and is asserted as one, but it is not a test of
the fitting.  ``WRONG_G`` and ``WRONG_Q`` are the same constants
:mod:`tests.unit.test_remainder_drtmle` uses, which are the same ones every sibling
remainder module uses.

**Why a tie is what makes most of this non-vacuous.**  A reduced regression conditions on
the *value* of the other nuisance.  With three distinct values on this law it conditions on
``W`` itself and is a relabelling rather than a pooling: every group is a singleton, so any
pooling weight cancels and a mistake in it is invisible.  ``TIED_G`` and ``TIED_Q`` make it
a genuine pooling, and they are what separates an implementation that conditions on the
estimated nuisance from one that quietly conditions on the covariate.

The saturated learner is what makes the comparison exact.  On a one-column design whose
column is a nuisance prediction, the mean of the target within each distinct design value
*is* the conditional expectation being estimated -- and on a law the sample realises
exactly, that sample mean is the population one.  So these are equalities at ``1e-12``,
not statistical claims.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly.data import CausalData
from cleverly.estimators._nuisance import NuisanceEstimates, Propensity
from cleverly.estimators.reduced import ReducedSet, fit_reduced
from cleverly.fluctuation.iterative import InitialFit
from cleverly.learners import make_folds
from cleverly.learners.crossfit import Folds
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law as law
from tests import discrete_law_multi as multi

# The package's one saturated learner. It lives in the longitudinal law only because that
# is where it was first needed -- it reads a design matrix and a target and knows nothing
# about nodes -- and duplicating it here to avoid a cross-module import would be the
# commoner mistake.
from tests.discrete_law_longitudinal import CellMeans
from tests.unit.test_remainder_drtmle import TIED_G, TIED_Q, WRONG_G, WRONG_Q, _reduced

ARMS = (0.0, 1.0)

#: ``WRONG_Q`` with arm 1's prediction tied across covariate cells 0 and 1 and arm 0's
#: left distinct.  Every other outcome regression in play here -- ``WRONG_Q``, ``TIED_Q``,
#: the law's own -- ties the *same* cells in both columns or neither, and then the two
#: reduced mechanisms pool over the same groups and sum to one however they are computed.
#: Only a design that is coarse for one arm and fine for the other shows that these are
#: two separate regressions rather than a probability and its complement.
SPLIT_Q = WRONG_Q.copy()
SPLIT_Q[1, 1] = SPLIT_Q[0, 1]

#: Wide enough never to bind on this law, whose propensities lie in ``[0.25, 0.6]``. The
#: bound has to be inert for the comparisons against ``_reduced`` to hold, since that
#: helper divides by the untruncated mechanism; :class:`TestTheGr2BoundIsFitTime` is where
#: a binding one is the point.
INERT_BOUNDS = (1e-6, 1.0 - 1e-6)


def causal_data() -> CausalData:
    """The law's sample as a :class:`~cleverly.data.CausalData`."""
    return CausalData.from_frame(law.frame(), outcome="Y", treatment="A", covariates=["W"])


def nuisances(g_hat: np.ndarray, q_hat: np.ndarray, *, folds: Folds | None = None):
    """A :class:`NuisanceEstimates` holding the given guesses, cell values expanded to rows.

    The outcome is binary here, so :class:`OutcomeScaler` is the identity and ``q_hat`` is
    already on the ``[0, 1]`` scale the initial fit lives on.
    """
    covariate = law.frame()["W"].to_numpy().astype(int)
    treatment = law.frame()["A"].to_numpy(dtype=float)
    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    mechanism = g_hat[covariate]
    return NuisanceEstimates(
        propensity=Propensity(np.column_stack([1.0 - mechanism, mechanism]), ARMS),
        outcome=InitialFit(
            observed=np.where(treatment == 1.0, at_one, at_zero),
            arms={1.0: at_one, 0.0: at_zero},
        ),
        scaler=OutcomeScaler(0.0, 1.0),
        folds=Folds.single(law.N) if folds is None else folds,
    )


def fitted(
    g_hat: np.ndarray,
    q_hat: np.ndarray,
    *,
    g_bounds: tuple[float, float] = INERT_BOUNDS,
    data: CausalData | None = None,
    folds: Folds | None = None,
) -> ReducedSet:
    reduced, _, _ = fit_reduced(
        causal_data() if data is None else data,
        nuisances(g_hat, q_hat, folds=folds),
        regression_learner=CellMeans(),
        classification_learner=CellMeans(),
        g_bounds=g_bounds,
    )
    return reduced


def longhand(g_hat: np.ndarray, q_hat: np.ndarray, arm: int) -> tuple[np.ndarray, ...]:
    """``_reduced``'s per-cell values, expanded to one value per row."""
    covariate = law.frame()["W"].to_numpy().astype(int)
    return tuple(values[covariate] for values in _reduced(g_hat, q_hat, arm))


class TestTheFittedValuesAreTheLonghandOnes:
    """The claim this module exists for, at both arms and under both kinds of design."""

    @pytest.mark.parametrize("arm", [0, 1])
    @pytest.mark.parametrize(
        ("g_hat", "q_hat", "what"),
        [
            (WRONG_G, WRONG_Q, "untied"),
            (TIED_G, WRONG_Q, "a tied mechanism"),
            (WRONG_G, TIED_Q, "a tied outcome regression"),
        ],
    )
    def test_each_regression_matches(
        self, arm: int, g_hat: np.ndarray, q_hat: np.ndarray, what: str
    ) -> None:
        reduced = fitted(g_hat, q_hat)
        column = reduced.column_for(float(arm))
        qr, gr1, gr2 = longhand(g_hat, q_hat, arm)
        for name, expected in (("qr", qr), ("gr1", gr1), ("gr2", gr2)):
            np.testing.assert_allclose(
                getattr(reduced, name)[:, column], expected, atol=1e-12, rtol=0, err_msg=what
            )

    def test_the_tied_cases_actually_pool(self) -> None:
        """The precondition the two tied cases rest on, asserted rather than trusted.

        Without a tie every group is a singleton, the pooling weight cancels, and the
        parametrisation above would be running the same test three times.
        """
        assert len(set(np.round(TIED_G, 12).tolist())) == 2
        assert len(set(np.round(TIED_Q[:, 1], 12).tolist())) == 2


class TestTheResidualIsOnTheScaledOutcome:
    r""":math:`Q_r` is a residual of :math:`\hat{\bar Q}`, which lives on ``[0, 1]``.

    **Nothing else in this module can see that.**  This law's outcome is binary, so
    :class:`~cleverly.utils.bounds.OutcomeScaler` is the identity and ``scale`` is a no-op:
    taking the residual against the raw outcome instead was applied and passed all 25
    tests here.  It is the same blind spot the working model over regimens found -- an
    exact law with a binary outcome makes a scaling bug invisible -- and it did not
    announce itself the second time either.  An affine relabelling is what does see it.
    """

    LOWER, SPAN = -3.0, 4.0

    def _relabelled(self) -> tuple[CausalData, NuisanceEstimates]:
        frame = law.frame()
        data = CausalData.from_frame(
            frame.assign(Y=self.LOWER + self.SPAN * frame["Y"]),
            outcome="Y",
            treatment="A",
            covariates=["W"],
        )
        scaler = OutcomeScaler(self.LOWER, self.LOWER + self.SPAN)
        return data, replace(nuisances(WRONG_G, WRONG_Q), scaler=scaler)

    def test_relabelling_the_outcome_leaves_qr_exactly_where_it_was(self) -> None:
        data, nuisance = self._relabelled()
        assert not nuisance.scaler.is_identity, "the relabelling must not be a no-op"
        reduced, _, _ = fit_reduced(
            data,
            nuisance,
            regression_learner=CellMeans(),
            classification_learner=CellMeans(),
            g_bounds=INERT_BOUNDS,
        )
        np.testing.assert_allclose(reduced.qr, fitted(WRONG_G, WRONG_Q).qr, atol=1e-12, rtol=0)

    def test_and_the_raw_residual_would_not_have(self) -> None:
        """The negative control: on this relabelling the two scales genuinely disagree."""
        covariate = law.frame()["W"].to_numpy().astype(int)
        outcome = law.frame()["Y"].to_numpy(dtype=float)
        raw = self.LOWER + self.SPAN * outcome - WRONG_Q[covariate, 1]
        scaled = outcome - WRONG_Q[covariate, 1]
        assert np.max(np.abs(raw - scaled)) > 1e-3

    def test_the_two_reduced_mechanisms_do_not_read_the_outcome_at_all(self) -> None:
        """So relabelling cannot move them, which is what says the above is about ``Qr``."""
        data, nuisance = self._relabelled()
        reduced, _, _ = fit_reduced(
            data,
            nuisance,
            regression_learner=CellMeans(),
            classification_learner=CellMeans(),
            g_bounds=INERT_BOUNDS,
        )
        plain = fitted(WRONG_G, WRONG_Q)
        np.testing.assert_allclose(reduced.gr1, plain.gr1, atol=1e-14, rtol=0)
        np.testing.assert_allclose(reduced.gr2, plain.gr2, atol=1e-14, rtol=0)


class TestTheArmMaskBelongsToTheOutcomeResidualOnly:
    """``Qr`` conditions on ``A = a`` and the two reduced mechanisms do not.

    That conditioning is carried entirely by which rows the learner is *trained* on: with
    the mask, cells sharing a value of ``g-hat`` are pooled in proportion to
    :math:`P(W) g_0(a|W)`, and without it in proportion to :math:`P(W)`.  The two agree
    wherever the design is untied, which is why
    :mod:`tests.unit.test_remainder_drtmle` found dropping the weight passed 95 tests
    before it was pinned -- so this is checked at the call site as well as through the
    arithmetic.
    """

    def test_the_masks_are_the_arm_and_then_none(self, monkeypatch: Any) -> None:
        seen: list[Any] = []
        import cleverly.estimators.reduced as module

        original = module.cross_fit_companion

        def record(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["fit_mask"])
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "cross_fit_companion", record)
        fitted(WRONG_G, WRONG_Q)

        treatment = law.frame()["A"].to_numpy(dtype=float)
        # Three calls per arm, in (qr, gr1, gr2) order, arms ascending.
        assert len(seen) == 6
        for position, arm in enumerate(ARMS):
            qr_mask, gr1_mask, gr2_mask = seen[3 * position : 3 * position + 3]
            np.testing.assert_array_equal(qr_mask, treatment == arm)
            assert gr1_mask is None and gr2_mask is None

    def test_the_pooled_value_is_mechanism_weighted(self) -> None:
        """And the arithmetic it produces, against the weight written out longhand.

        ``TIED_G`` pools covariate cells 0 and 2; the value there must be the residual
        averaged by ``P(W) g_0(1|W)`` and not by ``P(W)``.
        """
        reduced = fitted(TIED_G, WRONG_Q)
        covariate = law.frame()["W"].to_numpy().astype(int)
        residual = law.Q[:, 1] - WRONG_Q[:, 1]
        rows = np.array([0, 2])
        weight = law.P_W[rows] * law.G[rows]
        expected = float(np.sum(weight * residual[rows]) / np.sum(weight))
        unweighted = float(np.sum(law.P_W[rows] * residual[rows]) / np.sum(law.P_W[rows]))

        column = reduced.column_for(1.0)
        for cell in rows:
            value = reduced.qr[covariate == cell, column]
            np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
        assert abs(expected - unweighted) > 1e-3, "the two weightings must disagree here"


class TestTheClipIsPerRegression:
    """``gr1`` is a probability; the other two are signed and must not be floored at zero.

    Clipping a residual regression into ``[0, 1]`` returns a perfectly plausible array --
    every negative value becomes zero and nothing raises -- so this is checked against a
    learner that predicts outside the range on purpose rather than hoped for.
    """

    def test_the_two_residual_regressions_keep_their_sign(self) -> None:
        reduced = fitted(WRONG_G, WRONG_Q)
        assert reduced.qr.min() < 0.0
        assert reduced.gr2.min() < 0.0

    def test_the_reduced_mechanism_is_a_probability(self) -> None:
        class OutOfRange(BaseEstimator):
            """Predicts ``1.5`` whatever it is asked, through both prediction paths."""

            def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
                self.classes_ = np.array([0.0, 1.0])
                return self

            def predict(self, x: Any) -> np.ndarray:
                return np.full(np.asarray(x).shape[0], 1.5)

            def predict_proba(self, x: Any) -> np.ndarray:
                p = self.predict(x)
                return np.column_stack([1.0 - p, p])

        reduced, _, _ = fit_reduced(
            causal_data(),
            nuisances(WRONG_G, WRONG_Q),
            regression_learner=OutOfRange(),
            classification_learner=OutOfRange(),
            g_bounds=INERT_BOUNDS,
        )
        assert reduced.gr1.max() == 1.0, "gr1 must be clipped into [0, 1]"
        assert reduced.qr.max() == 1.5 and reduced.gr2.max() == 1.5, "the others must not be"


class TestTheyVanishAtTheTruth:
    """The property that makes every ``test_influence_gateaux*`` module blind to all this.

    Asserted where the arrays are produced rather than only where the arithmetic is
    reasoned about, because it is the reason this module exists at wrong nuisances at all.
    """

    def test_the_two_residual_regressions_are_zero(self) -> None:
        reduced = fitted(law.G, law.Q)
        np.testing.assert_allclose(reduced.qr, 0.0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(reduced.gr2, 0.0, atol=1e-14, rtol=0)

    def test_but_the_reduced_mechanism_does_not(self) -> None:
        """``gr1`` is a probability, and it sits in a denominator whose numerator vanishes.

        So an implementation that got *it* wrong would pass every degeneracy check there
        is, which is the whole reason it is fitted separately rather than folded in.
        """
        reduced = fitted(law.G, law.Q)
        assert np.all(reduced.gr1 > 0.2) and np.all(reduced.gr1 < 0.8)


class TestTheGr2BoundIsFitTime:
    """The one truncation in this package chosen when the array is built, not when it is read.

    ``gr2``'s target is a quotient by the mechanism, so it cannot be stored raw and
    re-truncated later the way the propensity and the missingness mechanism are.  The
    consequence -- that a truncation sweep does not move these arrays -- is only honest if
    the bound that *was* used travels with them.
    """

    def test_a_binding_bound_moves_gr2_and_is_recorded(self) -> None:
        loose = fitted(WRONG_G, WRONG_Q, g_bounds=INERT_BOUNDS)
        tight = fitted(WRONG_G, WRONG_Q, g_bounds=(0.5, 0.5 + 1e-9))
        assert np.max(np.abs(loose.gr2 - tight.gr2)) > 1e-3
        assert tight.g_bounds == (0.5, 0.5 + 1e-9)

    def test_and_it_does_not_move_the_other_two(self) -> None:
        """The design is the *untruncated* mechanism, because it is a conditioning
        variable rather than a denominator: bounding it would collapse the extreme rows
        into ties and coarsen the sigma-algebra the reduction projects onto."""
        loose = fitted(WRONG_G, WRONG_Q, g_bounds=INERT_BOUNDS)
        tight = fitted(WRONG_G, WRONG_Q, g_bounds=(0.5, 0.5 + 1e-9))
        np.testing.assert_allclose(loose.qr, tight.qr, atol=1e-14, rtol=0)
        np.testing.assert_allclose(loose.gr1, tight.gr1, atol=1e-14, rtol=0)

    def test_gr1_is_bounded_at_read_time_instead(self) -> None:
        reduced = fitted(WRONG_G, WRONG_Q)
        bounded = reduced.bounded_gr1((0.45, 0.55))
        assert bounded.min() >= 0.45 and bounded.max() <= 0.55
        assert reduced.gr1.min() < 0.45, "the stored array must be the untruncated one"


class TestTheSplitIsTheOneTheNuisancesUsed:
    """Which folds, and the cluster codes that have to reach them.

    The split is reused deliberately -- ``fit_reduced``'s docstring gives the argument, and
    it is not the reuse-versus-redraw one -- so what is pinned here is that it *is* reused
    and that ``groups`` is forwarded, in :mod:`tests.unit.test_crossfit_leakage`'s idiom.
    """

    def test_the_reduced_regressions_run_on_nuisance_folds(self, monkeypatch: Any) -> None:
        import cleverly.estimators.reduced as module

        seen: list[Any] = []
        original = module.cross_fit_companion

        def record(*args: Any, **kwargs: Any) -> Any:
            seen.append((args[4], kwargs["groups"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "cross_fit_companion", record)
        folds = make_folds(law.N, 3, random_state=0)
        fitted(WRONG_G, WRONG_Q, folds=folds)

        assert seen, "nothing was fitted"
        for used, groups in seen:
            assert used is folds
            assert groups is None  # this law declares no clusters

    def test_cluster_codes_are_forwarded(self, monkeypatch: Any) -> None:
        import cleverly.estimators.reduced as module

        seen: list[Any] = []
        original = module.cross_fit_companion

        def record(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["groups"])
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "cross_fit_companion", record)
        frame = law.frame().assign(pid=np.arange(law.N) // 10)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"], id="pid")
        fitted(WRONG_G, WRONG_Q, data=data)

        assert seen and all(codes is not None for codes in seen)
        for codes in seen:
            np.testing.assert_array_equal(codes, data.cluster)

    def test_a_cross_fitted_run_is_close_but_not_exact(self) -> None:
        """And why the exact comparisons above are taken at ``Folds.single``.

        The sample realises the law exactly; a *fold* of it does not, so an out-of-fold
        cell mean is that cell's mean among two-thirds of the rows and differs from the
        population value by an amount that is real rather than numerical.  Asserting the
        exact values under cross-fitting would be asserting something false, and loosening
        the tolerance until it passed would turn a statement about the regression into a
        statement about a seed.  So the fold structure is pinned structurally above and
        the arithmetic is pinned where it is exact.
        """
        reduced = fitted(WRONG_G, WRONG_Q, folds=make_folds(law.N, 3, random_state=0))
        column = reduced.column_for(1.0)
        _, gr1, _ = longhand(WRONG_G, WRONG_Q, 1)
        error = np.abs(reduced.gr1[:, column] - gr1)
        assert error.max() > 1e-12, "a fold does not realise the law, so this cannot be exact"
        assert error.max() < 0.05, "but it is the same regression, so it cannot be far off"


class TestTheRefusals:
    """Each says what the derivation would need, rather than what has not been typed."""

    def test_the_bivariate_reduction_is_refused_by_name(self) -> None:
        with pytest.raises(NotImplementedError, match="bivariate"):
            fit_reduced(
                causal_data(),
                nuisances(WRONG_G, WRONG_Q),
                regression_learner=CellMeans(),
                classification_learner=CellMeans(),
                g_bounds=INERT_BOUNDS,
                reduction="bivariate",
            )

    def test_an_unknown_reduction_is_refused_too(self) -> None:
        with pytest.raises((NotImplementedError, ValueError)):
            fit_reduced(
                causal_data(),
                nuisances(WRONG_G, WRONG_Q),
                regression_learner=CellMeans(),
                classification_learner=CellMeans(),
                g_bounds=INERT_BOUNDS,
                reduction="trivariate",
            )

    def test_a_multi_arm_fit_is_refused_with_the_reason(self) -> None:
        """On the genuine three-armed law, so the message names real levels.

        The scope is binary because that is what has been *derived*: ``drtmle`` accepts a
        multi-valued treatment and the software paper works an example, but van der Laan
        (2014) states its problem for a binary one.
        """
        data = CausalData.from_frame(multi.frame(), outcome="Y", treatment="A", covariates=["W"])
        mechanism = np.full((data.n, data.n_arms), 1.0 / data.n_arms)
        wider = replace(
            nuisances(WRONG_G, WRONG_Q),
            propensity=Propensity(mechanism, data.arm_codes),
        )
        with pytest.raises(NotImplementedError, match="binary treatment"):
            fit_reduced(
                data,
                wider,
                regression_learner=CellMeans(),
                classification_learner=CellMeans(),
                g_bounds=INERT_BOUNDS,
            )


class TestTheSetItself:
    def test_the_shape_is_validated_against_the_arms(self) -> None:
        with pytest.raises(ValueError, match=r"gr1 must be \(n, 2\)"):
            ReducedSet(
                qr=np.zeros((4, 2)),
                gr1=np.zeros((4, 3)),
                gr2=np.zeros((4, 2)),
                arms=ARMS,
                g_bounds=INERT_BOUNDS,
            )

    def test_an_unknown_reduction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reduction must be one of"):
            ReducedSet(
                qr=np.zeros((4, 2)),
                gr1=np.zeros((4, 2)),
                gr2=np.zeros((4, 2)),
                arms=ARMS,
                g_bounds=INERT_BOUNDS,
                reduction="bivariate ",
            )

    def test_gr1_is_not_complemented_across_the_arms(self) -> None:
        r"""The two columns condition on different designs, so they do not sum to one.

        Which is why :meth:`ReducedSet.bounded_gr1` clips column by column rather than
        following :meth:`~cleverly.estimators._nuisance.Propensity.bounded`'s two-arm
        complement rule -- and it takes :data:`SPLIT_Q` to see, because on every other
        outcome regression here both columns take three distinct values, so both
        sigma-algebras are :math:`\sigma(W)`, both reductions are relabellings, and
        :math:`g_{r,1}(1|w) + g_{r,1}(0|w)` is :math:`g + (1 - g)` after all.
        """
        assert np.max(np.abs(fitted(WRONG_G, WRONG_Q).gr1.sum(axis=1) - 1.0)) < 1e-12
        split = fitted(WRONG_G, SPLIT_Q).gr1
        assert np.max(np.abs(split.sum(axis=1) - 1.0)) > 1e-2
