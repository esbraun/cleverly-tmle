r"""The public ``reduced_crossfit`` contract: where each construction trains, and what moves.

``DRTMLE(reduced_crossfit=...)`` takes ``"foldwise"`` (the default) or ``"pooled"``.
``docs/technical-reference/dr-tmle/targeting.md``, *Reduced-regression cross-fitting*, gives
the argument.  Under ``"pooled"`` (R ``drtmle``'s ``cvFolds`` construction) fold ``k``'s
reduced regressions train on the other folds' cross-fitted designs, and those designs came
from models that trained on fold ``k``.  So fold ``k``'s data reach its own reduced
regressions twice: through the initial designs, and through the tilts the alternation applies
to them.  ``"foldwise"`` closes both.  Fold ``k`` trains on its own **slab**: leave-two-out
designs at the other folds' rows, tilted along the slab's own covariates at the pooled
``epsilon``.

**Six kinds of claim, and they are different kinds.**

* **The witness.**  Given the ``epsilon`` history, fold ``k``'s slab does not move when fold
  ``k``'s treatments and outcomes do, and restoring any one production covariate breaks that
  (:class:`TestAFoldsReducedRegressionsNeverSeeThatFoldsData`).
* **The production identity.**  At fold ``k``'s own rows slab ``k`` is the production state
  at every refit (:class:`TestEachSlabHoldsTheProductionStateAtItsOwnFold`).
* **Routing, structural.**  The training rows of fold ``k``'s reduced regression read the
  arrays that arm says they do -- the slab under ``"foldwise"``, production under
  ``"pooled"`` -- and the row it *predicts* reads the production one either way.  Asserted at
  the call site rather than inferred from a number, because both designs are plausible arrays
  of the right shape and swapping them changes an estimator silently.
* **Routing, longhand.**  With a saturated reduction learner each construction's fitted value
  is a cell mean over a nameable set of rows, and both are written out: ``"foldwise"`` leaves
  two folds out, ``"pooled"`` leaves one.
* **The supplied split is honoured exactly.**  Entry ``k`` of the leave-two-out designs comes
  from models fitted with outer fold ``k`` removed, against the caller's own partition and no
  second split; a set built against another partition is refused.
* **Non-vacuity and invariance.**  Where the initial designs coincide the two constructions
  still **differ** after the first tilt; carried along the production covariates instead,
  they coincide bit for bit.  Reordering the rows reorders the answer and changes nothing
  else.

**One neighbouring claim deliberately lives elsewhere.**  That a companion's fold weights are
the **held-out row counts** -- ``n_k/n``, not equal weights, which coincide only on a balanced
split -- is asserted in :mod:`tests.unit.test_drtmle_companion` beside the rest of the
``evaluation=`` contract.  It is a property of :class:`~cleverly.estimators._nuisance.
CompanionEstimates` rather than of this keyword, and asserting it in two places would let the
two drift.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.base
import sklearn.linear_model

from cleverly.data import CausalData
from cleverly.datasets import make_binary_outcome
from cleverly.estimators import DRTMLE
from cleverly.estimators import _nuisance as nuisance_module
from cleverly.estimators import drtmle as drtmle_module
from cleverly.estimators import reduced as reduced_module
from cleverly.estimators import targeting as targeting_module
from cleverly.estimators._nuisance import (
    InitialFit,
    InnerDesigns,
    NuisanceEstimates,
    Propensity,
    _screened,
    fit_inner_designs,
)
from cleverly.estimators._strata import (
    refuse_untrainable_leave_two_out,
    refuse_untrainable_stratum_folds,
)
from cleverly.estimators.reduced import ReducedSet, fit_reduced
from cleverly.estimators.targeting import ONE_STEP_FOLDWISE_REFUSAL
from cleverly.exceptions import CapabilityError, ConvergenceWarning, RefusalReason
from cleverly.fluctuation.iterative import apply_logistic, split_carry
from cleverly.fluctuation.mechanism import apply_mechanism_tilt, split_mechanism_carry
from cleverly.fluctuation.reduced import reduced_mechanism_covariate
from cleverly.inference import cross_validated_variance
from cleverly.learners import Folds, make_folds
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.discrete_law_longitudinal import CellMeans

# The same constants, the same builders and the same law every sibling reduced-regression
# module uses -- `test_reduced_regressions.py` for the fitting, `test_oracle_reductions.py`
# for the end-to-end alternation. Two modules that built their own would be free to disagree
# about the fixture rather than about the claim.
from tests.unit.test_oracle_reductions import INERT, Misspecified
from tests.unit.test_reduced_regressions import ARMS, causal_data, nuisances
from tests.unit.test_remainder_drtmle import WRONG_G, WRONG_Q

#: Three folds, which is the fewest the construction accepts: it leaves two out at a time,
#: so a fold's reduced regression trains on ``K - 2`` folds and there must be one left.
FOLDS = 3


@pytest.fixture(scope="module")
def source_cv_fit() -> Any:
    """A cheap unequal-fold fit that distinguishes the source's three CV choices.

    The pinned R ``cvFolds`` path uses held-out predictions for both primary and reduced
    regressions, then runs one alternation over the assembled rows, averages ``QnStar`` over
    those rows, and takes ``cov(IC) / n``.  At 101 rows and three folds, an equal ``1/V``
    average and the cross-validated variance are different numbers, so the alternatives are
    nonzero mutations rather than descriptions that happen to coincide on balanced folds.

    The outcome is binary (the fold and outcome-scale rules): a cross-fitted continuous outcome needs a declared
    ``q_bounds``, and this fit's subject is the CV contract itself, not a scale choice, so
    it moves to a law whose family the scaler never asks a bound of.
    """
    frame, _ = make_binary_outcome(n=101, seed=17)
    return (
        DRTMLE(
            estimands=("ey1", "ey0"),
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
            reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
            reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=FOLDS,
            learner_folds=2,
            random_state=0,
            simultaneous=False,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def _folds() -> Any:
    """The outer split every test here shares, drawn once from a fixed seed."""
    return make_folds(law.N, FOLDS, stratify=law.frame()["A"].to_numpy(dtype=float), random_state=0)


def _inner_from(
    nuisance: Any, permute: np.ndarray | None = None, *, spread: float = 0.0
) -> InnerDesigns:
    """Fold-free copies built by declaration rather than by fitting.

    ``permute`` reorders which covariate cell carries which mechanism *value*, so the inner
    design takes exactly the production design's set of values on a different set of rows.
    That is what makes the longhand below exact and non-vacuous at once: a saturated learner
    keyed on design values never meets an unseen one -- so nothing falls back to the training
    mean -- while the rows a given value pools are genuinely different ones.

    ``spread`` shifts each fold's copy by a different amount, which is what makes the folds
    **distinguishable from each other**.  Without it every entry is the same array and a
    construction reading fold ``k'``'s copy where it should read fold ``k``'s passes every
    assertion here -- so the fixture would be blind to the one indexing mistake this
    construction can make.

    Both left off leaves the copies equal to the production arrays, which is the degenerate
    control.
    """
    folds = nuisance.folds
    if permute is None and spread == 0.0:
        return InnerDesigns(
            outcome=tuple(nuisance.outcome for _ in range(folds.n_folds)),
            propensity=tuple(nuisance.propensity for _ in range(folds.n_folds)),
        )
    covariate = law.frame()["W"].to_numpy().astype(int)
    base = WRONG_G if permute is None else WRONG_G[permute]
    return InnerDesigns(
        outcome=tuple(nuisance.outcome for _ in range(folds.n_folds)),
        propensity=tuple(
            Propensity(
                np.column_stack([1.0 - (values := base[covariate] + spread * fold), values]), ARMS
            )
            for fold in range(folds.n_folds)
        ),
    )


def _fitted(inner: InnerDesigns | None, *, folds: Any) -> ReducedSet:
    """``fit_reduced`` at the declared nuisances, pooled when ``inner`` is ``None``."""
    base = nuisances(WRONG_G, WRONG_Q, folds=folds)
    reduced, _, _, _ = fit_reduced(
        causal_data(),
        base if inner is None else replace(base, inner=inner),
        regression_learner=CellMeans(),
        classification_learner=CellMeans(),
        mechanism_residual_learner=CellMeans(),
        g_bounds=INERT,
        crossfit="pooled" if inner is None else "foldwise",
    )
    return reduced


class TestTheTrainingRowsNeverSawTheFoldTheyPredict:
    """The structural pin, in :mod:`tests.unit.test_crossfit_leakage`'s idiom.

    Asserted at the **call site** rather than read off the fitted values, because the two
    designs are arrays of the same shape holding plausible probabilities: handing the
    training rows the production design, or predicting at the inner one, produces a fit that
    is wrong and looks exactly as right as this one.
    """

    def test_no_fold_trains_on_the_rows_it_predicts(self, monkeypatch: Any) -> None:
        folds = _folds()
        calls: list[tuple[np.ndarray, np.ndarray]] = []
        real = reduced_module.fit_on_rows

        def spy(learner: Any, design: Any, target: Any, weights: Any, rows: Any, *args: Any) -> Any:
            calls.append((np.asarray(rows), np.asarray(design, dtype=float).reshape(-1)))
            return real(learner, design, target, weights, rows, *args)

        monkeypatch.setattr(reduced_module, "fit_on_rows", spy)
        _fitted(_inner_from(nuisances(WRONG_G, WRONG_Q, folds=folds)), folds=folds)

        assert calls, "the foldwise path fitted nothing; the spy is on the wrong name"
        # Three regressions per arm, two arms, one model per fold.
        assert len(calls) == 3 * len(ARMS) * FOLDS
        by_fold = [set(folds.test_index(fold).tolist()) for fold in range(FOLDS)]
        for rows, _ in calls:
            held_out = [
                fold for fold, index in enumerate(by_fold) if not set(rows.tolist()) & index
            ]
            assert held_out, f"a model trained on rows from every fold: {sorted(rows)[:5]}..."

    def test_the_training_design_is_the_inner_one_and_not_the_production_one(
        self, monkeypatch: Any
    ) -> None:
        """The anti-vacuity half: without it the pin above passes on a pooled fit too."""
        folds = _folds()
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        inner = _inner_from(base, permute=np.array([1, 2, 0]), spread=0.02)
        seen: list[np.ndarray] = []
        real = reduced_module.fit_on_rows

        def spy(learner: Any, design: Any, target: Any, weights: Any, rows: Any, *args: Any) -> Any:
            seen.append(np.asarray(design, dtype=float).reshape(-1))
            return real(learner, design, target, weights, rows, *args)

        monkeypatch.setattr(reduced_module, "fit_on_rows", spy)
        _fitted(inner, folds=folds)

        # `qr` is the regression whose design is the mechanism, and it is fitted first, once
        # per fold, for the *first* arm -- which is arm 0, since `arms` is ascending and
        # `fit_reduced` iterates it in order. Reading arm 1's array here would compare the
        # complement and pass for the wrong reason.
        production = np.asarray(base.propensity.arm(ARMS[0]))
        assert not np.allclose(production, np.asarray(inner.propensity[0].arm(ARMS[0]))), (
            "the fixture ties the two designs"
        )
        # Fold by fold, and against *that* fold's copy: `spread` makes the entries differ
        # from one another, so reading fold `k'`'s array where fold `k`'s belongs is caught
        # here and nowhere else -- every other test in this module uses copies that are equal
        # across folds and is blind to it by construction.
        for fold, design in enumerate(seen[:FOLDS]):
            assert np.array_equal(design, np.asarray(inner.propensity[fold].arm(ARMS[0])))


class TestTheEvaluationDesignStaysTheProductionOne:
    r"""Fold ``k`` predicts at the design its rows were actually assigned under.

    The longhand: with a saturated reduction learner, row ``i``'s fitted value is the mean of
    the target over the training rows whose **inner** design equals row ``i``'s
    **production** design.  Predicting at the inner design instead would be a different
    estimator and a silent one -- every array stays in range and no score moves -- which is
    why this is written out rather than trusted.
    """

    def test_one_folds_values_are_the_longhand_ones(self) -> None:
        folds = _folds()
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        inner = _inner_from(base, permute=np.array([1, 2, 0]))
        fitted = _fitted(inner, folds=folds)

        scaled = np.asarray(law.frame()["Y"].to_numpy(dtype=float))
        treatment = np.asarray(law.frame()["A"].to_numpy(dtype=float))
        production = np.asarray(base.propensity.arm(1.0))
        training = np.asarray(inner.propensity[0].arm(1.0))
        target = scaled - np.asarray(base.outcome.arms[1.0])
        column = fitted.column_for(1.0)

        for fold, (train, test) in enumerate(folds):
            del fold
            rows = train[treatment[train] == 1.0]
            for row in test[:20]:
                pool = rows[np.isclose(training[rows], production[row])]
                assert pool.size, "the permuted design left a production value unseen"
                assert fitted.qr[row, column] == pytest.approx(
                    float(np.mean(target[pool])), abs=1e-12, rel=0
                )


class TestThePooledConstructionTrainsOnOneFoldsComplement:
    r"""The **default** arm, written out rather than left as the foldwise arm's baseline.

    ``"pooled"`` reuses the primary split as it stands: fold ``k``'s reduced regression trains
    on the complement of fold ``k``, at the **production** designs and targets, and predicts
    there too.  That is one fold left out where
    :class:`TestTheEvaluationDesignStaysTheProductionOne` leaves two, and the difference
    between those two sentences is the whole of what the keyword selects.

    Written as a cell mean for the same reason the foldwise longhand is: a saturated learner
    makes the fitted value an exact finite average over a nameable set of rows, so a routing
    mistake is a wrong number here rather than a plausible one.  Without this class, ``pooled``
    is only ever asserted as *the thing foldwise equals in the degenerate case*, which cannot
    tell a correct default from two matching mistakes.
    """

    def test_one_folds_values_are_the_complements_cell_means(self) -> None:
        folds = _folds()
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        fitted = _fitted(None, folds=folds)

        scaled = np.asarray(law.frame()["Y"].to_numpy(dtype=float))
        treatment = np.asarray(law.frame()["A"].to_numpy(dtype=float))
        production = np.asarray(base.propensity.arm(1.0))
        target = scaled - np.asarray(base.outcome.arms[1.0])
        column = fitted.column_for(1.0)

        for train, test in folds:
            # `qr` carries the `| A = a` of its definition as a fit mask; there is no
            # missingness on this law, so the mask is the arm indicator alone.
            rows = train[treatment[train] == 1.0]
            for row in test[:20]:
                pool = rows[np.isclose(production[rows], production[row])]
                assert pool.size, "a production value appears in no training fold"
                assert fitted.qr[row, column] == pytest.approx(
                    float(np.mean(target[pool])), abs=1e-12, rel=0
                )

    def test_the_pool_is_one_folds_complement_and_not_two(self) -> None:
        """The anti-vacuity half: the two constructions' pools must not coincide.

        If the training set here were also leave-two-out, every assertion above would hold of
        the foldwise arm as well and the class would be checking nothing about the default.
        """
        folds = _folds()
        counts = {len(train) for train, _ in folds}
        assert counts, "the split yielded no folds"
        # Leave-one-out training sets are strictly larger than leave-two-out ones.
        assert min(counts) > law.N - 2 * (law.N // FOLDS)


class TestTheCanonicalSourceCVContract:
    """What the pinned R ``cvFolds`` path maps to in cleverly's separate CV API.

    This is structural implementation provenance, not numerical parity with R and not a
    theorem for a new estimator.  The training-row tests above pin the out-of-fold primary
    and reduced regressions; these checks pin the three choices made after those predictions
    have been assembled.  Each has a neighbouring alternative that is nonzero on this fit.
    """

    def test_it_runs_one_pooled_alternation_and_no_fold_report(self, source_cv_fit: Any) -> None:
        fluctuation = source_cv_fit.fluctuations["mean"]

        assert source_cv_fit.config.cross_fit
        assert source_cv_fit.config.targeting_scheme == "pooled"
        assert not source_cv_fit.config.cv_evaluation
        assert not fluctuation.folds
        assert source_cv_fit.cv_targeting is None

    @pytest.mark.parametrize(("arm", "name"), [(1.0, "ey1"), (0.0, "ey0")])
    def test_the_plugin_is_the_whole_sample_mean_not_an_equal_fold_average(
        self, source_cv_fit: Any, arm: float, name: str
    ) -> None:
        targeted = np.asarray(source_cv_fit.fluctuations["mean"].targeted.arms[arm])
        whole_sample = float(np.average(targeted, weights=source_cv_fit.data.weights))
        equal_fold = float(
            np.mean(
                [
                    np.average(targeted[test], weights=source_cv_fit.data.weights[test])
                    for _, test in source_cv_fit.nuisance.folds
                ]
            )
        )

        assert [len(test) for _, test in source_cv_fit.nuisance.folds] == [34, 34, 33]
        assert abs(whole_sample - equal_fold) > 1e-5, "the fold-aggregation mutation vanished"
        assert source_cv_fit[name].psi == source_cv_fit.nuisance.scaler.unscale_level(whole_sample)

    @pytest.mark.parametrize("name", ["ey1", "ey0"])
    def test_the_variance_is_from_the_whole_corrected_curve(
        self, source_cv_fit: Any, name: str
    ) -> None:
        estimate = source_cv_fit[name]
        indices = [test for _, test in source_cv_fit.nuisance.folds]
        ordinary = float(np.var(estimate.influence_curve, ddof=1) / source_cv_fit.n)
        fold_evaluated = cross_validated_variance(estimate.influence_curve, indices, None)

        assert abs(ordinary - fold_evaluated) > 1e-5, "the variance mutation vanished"
        assert estimate.variance == ordinary


class TestTheSuppliedFoldMappingIsHonoured:
    """Each pair ``{k, m}`` is fitted with both folds removed, against the caller's own split.

    The construction is two keywords of
    :func:`~cleverly.estimators._nuisance.cross_fit_companion`: ``fit_mask`` with fold ``k``'s
    rows dropped, and ``only`` naming the folds ``m > k``.  So there is no second split and no
    new randomness to check.  What there *is* to check is that the mask is fold ``k``'s
    complement, that each pair is fitted once, and that the partition passed through is the
    object the caller supplied.  An off-by-one would nest every fold inside its neighbour and
    hand back arrays of exactly the right shape, holding exactly plausible probabilities,
    which is why this is asserted at the call site.
    """

    def test_entry_k_removes_exactly_fold_k_from_the_supplied_split(self, monkeypatch: Any) -> None:
        folds = _folds()
        base = replace(nuisances(WRONG_G, WRONG_Q, folds=folds), folds=folds)
        seen: list[tuple[Any, np.ndarray, tuple[int, ...]]] = []
        real = nuisance_module.cross_fit_companion

        def spy(learner: Any, design: Any, target: Any, weights: Any, split: Any, **kw: Any) -> Any:
            mask = np.asarray(kw["fit_mask"], dtype=bool).copy()
            seen.append((split, mask, tuple(kw["only"])))
            return real(learner, design, target, weights, split, **kw)

        monkeypatch.setattr(nuisance_module, "cross_fit_companion", spy)
        fit_inner_designs(
            causal_data(),
            base,
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
        )

        # One mechanism and one regression run per fold but the last, which every pair
        # already covers.
        assert len(seen) == 2 * (FOLDS - 1)
        assignment = np.asarray(folds.assignment)
        observed = np.asarray(causal_data().observed, dtype=bool)
        pairs = []
        for fold in range(FOLDS - 1):
            mechanism_split, mechanism_mask, mechanism_only = seen[2 * fold]
            regression_split, regression_mask, regression_only = seen[2 * fold + 1]
            # The *same object*, not merely an equal partition: a copy would be a second
            # split, and the whole claim of the construction is that there is not one.
            assert mechanism_split is folds
            assert regression_split is folds
            assert np.array_equal(mechanism_mask, assignment != fold)
            assert np.array_equal(regression_mask, observed & (assignment != fold))
            assert mechanism_only == regression_only == tuple(range(fold + 1, FOLDS))
            pairs += [(fold, other) for other in mechanism_only]
        assert sorted(pairs) == [
            (first, second) for first in range(FOLDS) for second in range(first + 1, FOLDS)
        ]


class TestIdenticalInnerDesignsReproduceThePooledArrays:
    """Where the fold-free copies *are* the production arrays, so are the reductions.

    The array-level half of the degenerate control below, and the cheaper one: it needs no
    alternation, so a failure here localises to :func:`~cleverly.estimators.reduced.
    _slab_column` rather than to the transfer through the loop.
    """

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_every_reduction_is_bit_for_bit_the_pooled_one(self, name: str) -> None:
        folds = _folds()
        pooled = _fitted(None, folds=folds)
        foldwise = _fitted(_inner_from(nuisances(WRONG_G, WRONG_Q, folds=folds)), folds=folds)
        assert np.array_equal(getattr(pooled, name), getattr(foldwise, name))


class TestTheInnerDesignsLeaveTwoFoldsOut:
    """What :func:`~cleverly.estimators._nuisance.fit_inner_designs` builds.

    Two claims, and the second is what stops the first from being free: entry ``k`` must come
    from models that never saw fold ``k``, and it must actually *differ* from the production
    array, or the whole construction is the pooled one under another name.
    """

    def test_a_learning_primary_nuisance_gives_designs_that_differ(self) -> None:
        folds = _folds()
        data = causal_data()
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        inner = fit_inner_designs(
            data,
            replace(base, folds=folds),
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
        )
        assert inner.n_folds == FOLDS
        production = np.asarray(base.propensity.arm(1.0))
        differs = [
            not np.allclose(np.asarray(inner.propensity[fold].arm(1.0)), production)
            for fold in range(FOLDS)
        ]
        assert all(differs), "every fold's copy equals the production mechanism"

    def test_designs_built_against_another_split_are_refused(self) -> None:
        """The designs are keyed to *a* split; entry ``k`` is what left outer fold ``k`` out.

        Reusing a set built against a different partition would nest every fold inside
        somebody else's, and the arrays would be the right shape and entirely ordinary
        looking.  Unreachable through :class:`~cleverly.DRTMLE`, where one ``_nuisances``
        call builds both, and reachable through :func:`~cleverly.estimators.reduced.
        fit_reduced` directly -- which is where the check lives.
        """
        folds = _folds()
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        wider = make_folds(law.N, FOLDS + 2, random_state=0)
        mismatched = _inner_from(replace(base, folds=wider))
        with pytest.raises(ValueError, match="built against a different split"):
            fit_reduced(
                causal_data(),
                replace(base, inner=mismatched),
                regression_learner=CellMeans(),
                classification_learner=CellMeans(),
                mechanism_residual_learner=CellMeans(),
                g_bounds=INERT,
                crossfit="foldwise",
            )

    def test_it_refuses_a_split_with_nothing_to_leave_out(self) -> None:
        folds = make_folds(law.N, 2, random_state=0)
        base = replace(nuisances(WRONG_G, WRONG_Q, folds=folds), folds=folds)
        with pytest.raises(ValueError, match="at least"):
            fit_inner_designs(
                causal_data(),
                base,
                outcome_learner=CellMeans(),
                treatment_learner=CellMeans(),
            )


def _fit(**overrides: Any) -> Any:
    """One fit on the exact law at nuisances wrong on purpose, cross-fitted.

    ``cross_fit=True`` where :mod:`tests.unit.test_oracle_reductions` has ``False``, and that
    is the whole difference: without a complement there is no fold to leave out and the
    construction has nothing to do.
    """
    dgp = Misspecified(WRONG_G, WRONG_Q)
    settings: dict[str, Any] = {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "reduced_outcome_learner": CellMeans(),
        "reduced_mechanism_residual_learner": CellMeans(),
        "reduced_treatment_learner": CellMeans(),
        "estimands": ("ey1", "ey0", "ate"),
        "g_bounds": INERT,
        "cross_fit": True,
        "n_folds": FOLDS,
        "learner_folds": 3,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return DRTMLE(**settings).fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def oracle_pair() -> dict[str, Any]:
    """The same fit under both constructions, at primary learners that do not learn."""
    return {"pooled": _fit(reduced_crossfit="pooled"), "foldwise": _fit()}


class TestADataIndependentPrimaryLearnerLeavesOnlyTheTiltChannel:
    """The two constructions share every initial array here, and they must still **differ**.

    An oracle primary learner returns the same function whichever rows trained it, so every
    leave-two-out design equals the production one and the initial reduced regressions agree.
    What remains is the tilt: a pooled fit carries each row along its own fold's covariate,
    and a foldwise slab carries every row along the slab fold's covariate.  So the two fits
    part after the first fluctuation, and that difference is the tilt channel alone.  An
    implementation that carried the slabs along the production covariates would make the two
    fits identical here, which is the mutation
    :class:`TestTheSlabsTiltAlongTheirOwnCovariates` pins bit for bit.
    """

    def test_the_initial_designs_are_the_production_ones(self, oracle_pair: dict[str, Any]) -> None:
        nuisance = oracle_pair["foldwise"].nuisance
        assert nuisance.inner is not None and nuisance.inner.n_folds == FOLDS
        production = np.asarray(nuisance.propensity.arm(1.0))
        for fold in range(FOLDS):
            assert np.array_equal(np.asarray(nuisance.inner.propensity[fold].arm(1.0)), production)

    def test_the_estimates_differ(self, oracle_pair: dict[str, Any]) -> None:
        pooled, foldwise = oracle_pair["pooled"]["ate"], oracle_pair["foldwise"]["ate"]
        assert abs(pooled.psi - foldwise.psi) > 1e-9

    def test_both_fits_solve_their_equations(self, oracle_pair: dict[str, Any]) -> None:
        for fit in oracle_pair.values():
            assert fit.diagnostics.score_equations().passed

    def test_the_pooled_fit_builds_no_slabs(self, oracle_pair: dict[str, Any]) -> None:
        assert oracle_pair["pooled"].nuisance.inner is None


def _bare(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """Carry the slabs along the **production** covariate of one equation: the mutation.

    A bare carry item travels along the solver's own covariate
    (:func:`~cleverly.fluctuation.iterative.split_carry`,
    :func:`~cleverly.fluctuation.mechanism.split_mechanism_carry`), so dropping the slab's own
    covariate from ``method``'s items is exactly the construction this row removed.
    """
    original = getattr(targeting_module._Companion, method)

    def production(self: Any, *args: Any, **kwargs: Any) -> Any:
        items = original(self, *args, **kwargs)
        if not self.fitting_rows:
            return items
        return tuple(item[0] for item in items)

    monkeypatch.setattr(targeting_module._Companion, method, production)


#: The three covariates a slab is tilted along: equation (8)'s ``1/g*`` (the priming solve
#: and every later equation (8) solve), equation (10)'s ``H_2`` and equation (9)'s ``Qr/g*``.
CARRIES = ("outcome_carry", "extra_carry", "mechanism_carry")


class TestTheSlabsTiltAlongTheirOwnCovariates:
    """The transfer through the alternation, pinned by its mutation.

    With the slabs' initial arrays equal to the production ones, the only difference between
    a foldwise and a pooled fit is the covariate each slab is tilted along.  Carry the slabs
    along the production covariates instead and a slab stays equal to the production state at
    every row, so the foldwise fit must reproduce the pooled fit **bit for bit**: ``psi``, the
    variance, the curve, every ``epsilon`` and the round count.  That equality pins the
    slab refits, the carries and the take-back in one assertion.
    """

    @pytest.fixture(scope="class")
    def pair(self) -> dict[str, Any]:
        with pytest.MonkeyPatch.context() as patch:
            for method in CARRIES:
                _bare(patch, method)
            mutated = _fit()
        return {"pooled": _fit(reduced_crossfit="pooled"), "foldwise": mutated}

    def test_the_estimate_and_the_curve_are_identical(self, pair: dict[str, Any]) -> None:
        for name in ("ey1", "ey0", "ate"):
            pooled, foldwise = pair["pooled"][name], pair["foldwise"][name]
            assert pooled.psi == foldwise.psi
            assert pooled.variance == foldwise.variance
            assert np.array_equal(pooled.influence_curve, foldwise.influence_curve)

    def test_the_fluctuation_took_the_same_route(self, pair: dict[str, Any]) -> None:
        pooled = pair["pooled"].fluctuations["mean"]
        foldwise = pair["foldwise"].fluctuations["mean"]
        assert np.array_equal(pooled.epsilon, foldwise.epsilon)
        assert pooled.reduction.n_outer == foldwise.reduction.n_outer
        assert pooled.reduction.exit_reason == foldwise.reduction.exit_reason


#: The fold whose data the witness perturbs, and the scripted ``epsilon`` per solve.
WITNESS_FOLD = 0
SCRIPTED_STEP = 0.07


def _scripted(sign: int, dim: int) -> np.ndarray:
    return sign * SCRIPTED_STEP * np.linspace(1.0, 0.5, dim)


class _Script:
    """Replace every alternation solve's ``epsilon`` by a fixed script, and record the slabs.

    The witness asks whether fold ``k``'s reduced regressions depend on fold ``k``'s
    treatment and outcome **given** the ``epsilon`` history.  The pooled ``epsilon`` does
    depend on them, by design (the CV-TMLE structure), so the history has to be held fixed
    for the question to be asked at all.  Each solve still runs, and its production array
    and every carried array are then replaced by one tilt at the scripted ``epsilon`` along
    the covariate each array was handed.  The exit tests are switched off, so both fits run
    exactly ``max_outer`` rounds and their refits align one to one.
    """

    def __init__(self, patch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        self.slabs: list[dict[str, np.ndarray]] = []
        solve = targeting_module.solve_submodel
        mechanism = targeting_module._solve_reduced_mechanism
        inputs = targeting_module._reduction_inputs

        def outcome(scaled: Any, initial: Any, submodel: Any, *args: Any, **kwargs: Any) -> Any:
            real = solve(scaled, initial, submodel, *args, **kwargs)
            self.calls += 1
            epsilon = _scripted((-1) ** self.calls, submodel.dim)
            alpha = args[2].alpha if len(args) > 2 else kwargs["spec"].alpha
            fits, owns = split_carry(kwargs.get("carry", ()), submodel)
            return replace(
                real,
                epsilon=epsilon,
                targeted=apply_logistic(initial, submodel, epsilon, alpha),
                carried=tuple(
                    apply_logistic(fit, own, epsilon, alpha)
                    for fit, own in zip(fits, owns, strict=True)
                ),
            )

        def tilt(response: Any, propensity: Any, reduced: Any, *args: Any, **kwargs: Any) -> Any:
            real = mechanism(response, propensity, reduced, *args, **kwargs)
            self.calls += 1
            bounds = kwargs["bounds"]
            covariate = targeting_module._blocked_columns(
                kwargs.get("blocks"),
                reduced_mechanism_covariate(reduced, propensity, bounds=bounds),
            )
            dim = 1 if np.ndim(covariate) == 1 else np.shape(covariate)[1]
            epsilon = _scripted((-1) ** self.calls, dim)
            bases, owns = split_mechanism_carry(kwargs.get("carry", ()), covariate)
            return replace(
                real,
                epsilon=epsilon,
                propensity=apply_mechanism_tilt(propensity, covariate, epsilon, bounds=bounds),
                carried=tuple(
                    apply_mechanism_tilt(base, own, epsilon, bounds=bounds)
                    for base, own in zip(bases, owns, strict=True)
                ),
            )

        def record(*args: Any, **kwargs: Any) -> Any:
            slabs = args[4] if len(args) > 4 else kwargs.get("slabs")
            if slabs is not None:
                k = WITNESS_FOLD
                reduced = slabs.reduced[k]
                self.slabs.append(
                    {
                        "q1": np.array(slabs.outcome[k].arms[1.0]),
                        "q0": np.array(slabs.outcome[k].arms[0.0]),
                        "g": np.array(slabs.mechanism[k]),
                        **{name: np.array(getattr(reduced, name)) for name in ("qr", "gr1", "gr2")},
                    }
                )
            return inputs(*args, **kwargs)

        patch.setattr(targeting_module, "solve_submodel", outcome)
        patch.setattr(targeting_module, "_solve_reduced_mechanism", tilt)
        patch.setattr(targeting_module, "_reduction_inputs", record)
        patch.setattr(targeting_module, "_solved", lambda *args: False)
        patch.setattr(targeting_module, "_STALL_FACTOR", np.inf)


#: The two update orders and the two reductions, each order once with each reduction.
CONFIGURATIONS = (
    {"update_order": "drtmle", "reduction": "univariate"},
    {"update_order": "benkeser", "reduction": "bivariate"},
)


def _witness_fit(
    frame: Any, settings: dict[str, Any], mutations: tuple[str, ...] = ()
) -> tuple[Any, _Script]:
    """One scripted foldwise fit at learning primaries and ``glm`` reductions."""
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        script = _Script(patch)
        for method in mutations:
            _bare(patch, method)
        fit = (
            DRTMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                estimands=("ey1", "ey0"),
                g_bounds=INERT,
                cross_fit=True,
                n_folds=FOLDS,
                learner_folds=3,
                simultaneous=False,
                random_state=0,
                max_outer=2,
                **settings,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    return fit, script


def _perturbed(frame: Any, assignment: np.ndarray) -> Any:
    """``frame`` with fold ``k``'s outcomes flipped and its treatments permuted within the fold.

    ``W`` is left alone on purpose: fold ``k``'s own rows are predicted at their own ``W``,
    so moving it would move fold ``k``'s slab legitimately.  The permutation keeps each
    fold's arm counts, so a split stratified on treatment draws the same folds.
    """
    rows = np.flatnonzero(assignment == WITNESS_FOLD)
    moved = frame.copy()
    moved.loc[rows, "Y"] = 1 - moved.loc[rows, "Y"].to_numpy()
    order = np.random.default_rng(3).permutation(rows)
    moved.loc[rows, "A"] = frame.loc[order, "A"].to_numpy()
    return moved


@pytest.fixture(scope="module", params=CONFIGURATIONS, ids=lambda each: each["update_order"])
def witness(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The straight and the perturbed scripted fits, and each mutation's pair."""
    settings = request.param
    frame = law.frame()
    straight, script = _witness_fit(frame, settings)
    assignment = np.asarray(straight.nuisance.folds.assignment)
    moved = _perturbed(frame, assignment)
    perturbed, moved_script = _witness_fit(moved, settings)
    pairs: dict[str, Any] = {"correct": (straight, script, perturbed, moved_script)}
    for method in CARRIES:
        a, sa = _witness_fit(frame, settings, (method,))
        b, sb = _witness_fit(moved, settings, (method,))
        pairs[method] = (a, sa, b, sb)
    return {"frame": frame, "moved": moved, "assignment": assignment, "pairs": pairs}


def _largest(gap: np.ndarray) -> float:
    """The largest absolute entry, with an absent family (all ``NaN``, as ``gr2`` under the
    bivariate reduction) counted as no gap; a ``NaN`` in only one array counts as infinite.
    """
    gap = np.asarray(gap, dtype=float)
    if np.all(np.isnan(gap)):
        return 0.0
    if np.any(np.isnan(gap)):
        return np.inf
    return float(np.max(np.abs(gap)))


def _slab_gap(pair: tuple[Any, _Script, Any, _Script]) -> float:
    """The largest difference between the two fits' recorded slab ``k`` states."""
    _, first, _, second = pair
    assert len(first.slabs) == len(second.slabs) > 0
    return max(
        _largest(a[name] - b[name])
        for a, b in zip(first.slabs, second.slabs, strict=True)
        for name in a
    )


class TestAFoldsReducedRegressionsNeverSeeThatFoldsData:
    r"""The fold-perturbation witness: the defect this row removes, asserted directly.

    Fold ``k``'s reduced regressions train on slab ``k``'s rows outside fold ``k``.  Given
    the ``epsilon`` history, slab ``k`` at every row is a function of the leave-two-out
    designs, fold ``k``'s own reduced models and the data outside fold ``k``.  So flipping
    fold ``k``'s outcomes and permuting its treatments must leave slab ``k`` (its outcome
    regression, its mechanism and its three reduced regressions, at every row and at every
    refit) **exactly** where it was.  The pooled construction fails this at the first refit,
    because its designs at other folds' rows came from models that trained on fold ``k``.

    The mutations restore one production covariate each.  A production covariate at a row of
    fold ``m`` is built from fold ``m``'s reduced regressions, which trained on fold ``k``,
    so each mutation lets fold ``k``'s data back in at the first refit it reaches.
    """

    def test_the_perturbation_moves_the_data_and_keeps_the_split(
        self, witness: dict[str, Any]
    ) -> None:
        frame, moved = witness["frame"], witness["moved"]
        assert not frame["Y"].equals(moved["Y"]) and not frame["A"].equals(moved["A"])
        assert frame["W"].equals(moved["W"])
        _, _, perturbed, _ = witness["pairs"]["correct"]
        assert np.array_equal(
            np.asarray(perturbed.nuisance.folds.assignment), witness["assignment"]
        )

    def test_the_perturbation_moves_the_estimate(self, witness: dict[str, Any]) -> None:
        """Without this the witness could pass on data that reach nothing."""
        straight, _, perturbed, _ = witness["pairs"]["correct"]
        assert abs(straight["ey1"].psi - perturbed["ey1"].psi) > 1e-6

    def test_every_refit_was_recorded(self, witness: dict[str, Any]) -> None:
        _, script, _, _ = witness["pairs"]["correct"]
        # Two rounds of the default order: three refits a round (gr1 and gr2 with the
        # mechanism block, then qr), so at least four even if a family is skipped.
        assert len(script.slabs) >= 4

    def test_slab_k_does_not_move(self, witness: dict[str, Any]) -> None:
        assert _slab_gap(witness["pairs"]["correct"]) <= 1e-12

    @pytest.mark.parametrize("method", CARRIES)
    def test_each_production_covariate_lets_the_fold_back_in(
        self, witness: dict[str, Any], method: str
    ) -> None:
        assert _slab_gap(witness["pairs"][method]) > 1e-8


def _learning_pair(**settings: Any) -> dict[str, Any]:
    """The same fit under both constructions, at primary learners that *do* learn."""
    return {
        "pooled": _fit(
            reduced_crossfit="pooled",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            **settings,
        ),
        "foldwise": _fit(
            reduced_crossfit="foldwise",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            **settings,
        ),
    }


class TestWhereTheDesignsDifferTheConstructionsDo:
    """Without this every equality above could be passing for the wrong reason.

    A primary learner that *learns* is what separates the two: an inner model and an outer
    model then disagree, so the reduced regressions are fitted on different designs and, for
    two of the three, different targets.
    """

    @pytest.fixture(scope="class")
    def pair(self) -> dict[str, Any]:
        return _learning_pair(
            reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
            reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
            reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        )

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_the_reductions_move(self, pair: dict[str, Any], name: str) -> None:
        pooled = np.asarray(getattr(pair["pooled"].nuisance.reduced, name))
        foldwise = np.asarray(getattr(pair["foldwise"].nuisance.reduced, name))
        assert float(np.max(np.abs(pooled - foldwise))) > 1e-12

    def test_both_fits_still_solve_their_equations(self, pair: dict[str, Any]) -> None:
        """A construction that moved the arrays and broke the loop would prove nothing."""
        for fit in pair.values():
            assert fit.diagnostics.score_equations().passed

    def test_a_retarget_of_a_foldwise_fit_is_still_foldwise(self, pair: dict[str, Any]) -> None:
        r"""The sensitivity analyses re-enter the alternation, and must not fall back to pooled.

        ``DRTMLE.retarget`` refits the reduced regressions -- they are regressions *of* the
        current pair, so they cannot be cached arithmetic the way an ordinary fit's are --
        and it reads the construction off the estimator while the fold-free designs ride on
        :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.inner`.  If either half went
        missing, a truncation curve on a foldwise fit would silently report a *pooled*
        estimator's sensitivity under the foldwise fit's name.

        The witness is that the retargeted estimate differs from the pooled fit's at the same
        bound: a fallback would make the two identical.
        """
        foldwise, pooled = pair["foldwise"], pair["pooled"]
        estimator = DRTMLE(
            reduced_crossfit="foldwise",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
            reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
            reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            estimands=("ate",),
            cross_fit=True,
            n_folds=FOLDS,
            learner_folds=3,
            simultaneous=False,
            random_state=0,
        )
        data = foldwise.data
        assert foldwise.nuisance.inner is not None
        estimates, _ = estimator.retarget(
            data, foldwise.nuisance, estimands=("ate",), g_bounds=INERT
        )
        assert estimates["ate"].psi != pooled["ate"].psi


class TestEachRepeatedDrawGetsItsOwnFoldFreeDesigns:
    """``repeats=`` redraws the primary split, and the designs are keyed to *a* split.

    This is why the fold-free copies live on
    :class:`~cleverly.estimators._nuisance.NuisanceEstimates` rather than on the estimator:
    ``_nuisances`` runs once per draw, so each draw builds its own against its own folds and
    the two cannot come apart.  Carrying them on the estimator would look identical on an
    ordinary fit and be wrong only here -- one draw's designs keyed to another draw's
    split, which is not a construction at all.
    """

    @pytest.fixture(scope="class")
    def repeated(self) -> Any:
        return DRTMLE(
            reduced_crossfit="foldwise",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            estimands=("ate",),
            cross_fit=True,
            n_folds=FOLDS,
            learner_folds=3,
            simultaneous=False,
            random_state=0,
            repeats=2,
        ).fit(law.frame(), outcome="Y", treatment="A")

    def test_the_draws_differ_and_so_do_their_designs(self, repeated: Any) -> None:
        draws = repeated.single().repeats
        assert len(draws) == 2
        first, second = draws[0].nuisance, draws[1].nuisance
        assert not np.array_equal(first.folds.assignment, second.folds.assignment), (
            "the two draws share a split, so this fixture cannot see the failure"
        )
        for each in (first, second):
            assert each.inner is not None and each.inner.n_folds == FOLDS
        assert not np.allclose(
            first.inner.propensity[0].arm(1.0), second.inner.propensity[0].arm(1.0)
        )


class TestASaturatedReductionOnAFiniteLawCannotSeeItInGr1:
    r"""A blind spot, measured by running it and watching it **pass**.

    :math:`g_{r,1}` is the one reduction whose *target* is data rather than an estimate --
    the arm indicator -- so the only way a construction reaches it is through its design.
    And a **saturated** learner keyed on distinct design values reads a design only through
    the *partition* it induces.  On a law with three covariate cells both constructions'
    :math:`\hat{\bar Q}` take three distinct values partitioning the rows by cell, so the two
    partitions coincide, the cell means coincide, and :math:`g_{r,1}` comes back **bit for
    bit identical however different the two designs are**.

    :math:`Q_r` and :math:`g_{r,2}` still move, because their targets are a residual of
    :math:`\hat{\bar Q}` and a quotient by :math:`\hat g` -- which is the half of the
    contamination that earlier descriptions of ``fit_reduced`` left out.

    So this is a degeneracy of the *fixture*, not of the construction, and it is written down
    rather than left to be rediscovered: a later reader finding ``gr1`` equal across the two
    arms should find this class rather than conclude the foldwise path is inert.  The class
    above uses ``glm`` reductions precisely to leave the degeneracy.
    """

    @pytest.fixture(scope="class")
    def pair(self) -> dict[str, Any]:
        return _learning_pair(
            reduced_outcome_learner=CellMeans(),
            reduced_treatment_learner=CellMeans(),
            reduced_mechanism_residual_learner=CellMeans(),
        )

    def test_gr1_does_not_move_and_that_is_the_partition(self, pair: dict[str, Any]) -> None:
        pooled = np.asarray(pair["pooled"].nuisance.reduced.gr1)
        foldwise = np.asarray(pair["foldwise"].nuisance.reduced.gr1)
        assert np.array_equal(pooled, foldwise)

    @pytest.mark.parametrize("name", ["qr", "gr2"])
    def test_the_two_with_estimated_targets_still_move(
        self, pair: dict[str, Any], name: str
    ) -> None:
        pooled = np.asarray(getattr(pair["pooled"].nuisance.reduced, name))
        foldwise = np.asarray(getattr(pair["foldwise"].nuisance.reduced, name))
        assert float(np.max(np.abs(pooled - foldwise))) > 1e-12


def _permuted(base: Any, inner: InnerDesigns | None, order: np.ndarray, folds: Folds) -> Any:
    """``base`` and its fold-free copies with their rows in ``order``.

    Rebuilt through the same constructors :func:`~tests.unit.test_reduced_regressions.
    nuisances` uses rather than by reaching into the containers, so a permuted fixture is the
    same *kind* of object as the straight one and not a hand-assembled lookalike.
    """

    def move(fit: Any) -> InitialFit:
        return InitialFit(
            observed=np.asarray(fit.observed)[order],
            arms={arm: np.asarray(fit.arms[arm])[order] for arm in ARMS},
        )

    def tilt(propensity: Any) -> Propensity:
        at_one = np.asarray(propensity.arm(1.0))[order]
        return Propensity(np.column_stack([1.0 - at_one, at_one]), ARMS)

    moved = replace(
        base,
        propensity=tilt(base.propensity),
        outcome=move(base.outcome),
        folds=folds,
        inner=None
        if inner is None
        else InnerDesigns(
            outcome=tuple(move(each) for each in inner.outcome),
            propensity=tuple(tilt(each) for each in inner.propensity),
        ),
    )
    return moved


class TestReorderingTheRowsReordersTheAnswer:
    r"""A permutation of the sample permutes the output and changes nothing else.

    Both constructions read row order in two places -- which rows a fold holds, and which rows
    a cell pools -- and neither is allowed to matter.  Permuting the sample **together with
    its fold assignment** holds both fixed as *sets*, so every fitted value must come back at
    its row's new position and nowhere else.

    This is the test that catches a construction indexing by position where it means to index
    by row: such a mistake is invisible on any fixture whose folds happen to be contiguous
    blocks, and this law's are not.

    Checked to ``1e-12`` absolute rather than bit for bit, and the gap is real rather than
    slack: a cell mean sums its pool in array order, so a permuted pool can differ in the last
    unit in the last place.  What is exactly invariant is the *mapping*; what is invariant to
    floating point is the value.
    """

    @pytest.mark.parametrize("crossfit", ["pooled", "foldwise"])
    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_the_arrays_come_back_permuted(self, crossfit: str, name: str) -> None:
        folds = _folds()
        order = np.random.default_rng(7).permutation(law.N)
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        # A non-degenerate inner set, so the foldwise arm is genuinely reading its own arrays
        # rather than reproducing the pooled ones under a keyword.
        inner = None if crossfit == "pooled" else _inner_from(base, permute=np.array([1, 2, 0]))
        straight = _fitted(inner, folds=folds)

        shuffled_folds = Folds(np.asarray(folds.assignment)[order], FOLDS)
        shuffled, _, _, _ = fit_reduced(
            CausalData.from_frame(
                law.frame().iloc[order].reset_index(drop=True),
                outcome="Y",
                treatment="A",
                covariates=["W"],
            ),
            _permuted(base, inner, order, shuffled_folds),
            regression_learner=CellMeans(),
            classification_learner=CellMeans(),
            mechanism_residual_learner=CellMeans(),
            g_bounds=INERT,
            crossfit=crossfit,
        )

        np.testing.assert_allclose(
            np.asarray(getattr(shuffled, name)),
            np.asarray(getattr(straight, name))[order],
            rtol=0,
            atol=1e-12,
        )

    def test_the_permutation_is_not_the_identity_on_the_folds(self) -> None:
        """Without this the class could pass on an order that moved no row between folds."""
        folds = _folds()
        order = np.random.default_rng(7).permutation(law.N)
        assignment = np.asarray(folds.assignment)
        assert not np.array_equal(assignment[order], assignment)


class TestTheRefusalsAndTheDefault:
    """``foldwise`` is the default wherever there are folds to keep apart, and only there."""

    def test_the_default_is_foldwise(self, oracle_pair: dict[str, Any]) -> None:
        assert DRTMLE().reduced_crossfit == "foldwise"
        assert oracle_pair["foldwise"].nuisance.inner is not None
        assert oracle_pair["pooled"].nuisance.inner is None

    def test_an_empty_guard_builds_no_slabs(self) -> None:
        fit = _fit(guard=())
        assert fit.nuisance.inner is None

    def test_an_in_sample_default_fit_builds_no_slabs(self) -> None:
        """In sample there is one fold and no complement, so the two settings coincide."""
        default = _fit(cross_fit=False)
        pooled = _fit(cross_fit=False, reduced_crossfit="pooled")
        assert default.nuisance.inner is None
        assert default["ate"].psi == pooled["ate"].psi
        assert default["ate"].variance == pooled["ate"].variance

    def test_an_in_sample_default_truncation_curve_runs(self) -> None:
        """The retarget check compares constructions only where they can differ."""
        fit = _fit(cross_fit=False)
        curve = fit.diagnostics.truncation_curve(bounds=[0.05])
        assert len(np.asarray(curve["truncated_fraction"])) == 3  # one row per estimand

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"reduced_crossfit": "fold"}, "must be one of"),
        ],
    )
    def test_it_names_the_settings_it_takes(self, settings: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            DRTMLE(**settings)

    def test_two_folds_are_refused_by_reason_naming_pooled(self) -> None:
        with pytest.raises(CapabilityError, match="leaves two folds out") as raised:
            DRTMLE(n_folds=2)
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "reduced_crossfit='pooled'" in str(raised.value.refusal.alternative)

    def test_two_folds_are_accepted_under_pooled_and_in_sample(self) -> None:
        DRTMLE(n_folds=2, reduced_crossfit="pooled")
        DRTMLE(n_folds=2, cross_fit=False)
        DRTMLE(n_folds=2, guard=())

    def test_two_folds_set_after_construction_are_refused_at_the_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fit asks again before any learner, with the declaration's own refusal.

        Without that, the fit reaches the post-draw leave-two-out check, whose message
        blames the split rather than the fold count.  The monkeypatch makes that check
        raise so that the test fails if the fit gets as far as the draw.
        """

        def reached(*_: Any) -> None:
            raise AssertionError("the fit drew folds before refusing n_folds=2")

        monkeypatch.setattr(drtmle_module, "refuse_untrainable_leave_two_out", reached)
        estimator = DRTMLE(n_folds=3, simultaneous=False, random_state=0)
        estimator.n_folds = 2
        with pytest.raises(CapabilityError, match="leaves two folds out") as raised:
            estimator.fit(law.frame(), outcome="Y", treatment="A")
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "reduced_crossfit='pooled'" in str(raised.value.refusal.alternative)

    def test_the_one_step_walk_is_refused_by_reason(self) -> None:
        """A cost decision rather than a derivation, and it names both ways out."""
        with pytest.raises(CapabilityError, match="one_step") as raised:
            _fit(targeting="one_step")
        assert raised.value.refusal.reason is ONE_STEP_FOLDWISE_REFUSAL.reason
        alternative = str(raised.value.refusal.alternative)
        assert "pooled" in alternative and "iterative" in alternative

    def test_one_step_runs_under_pooled(self) -> None:
        fit = _fit(targeting="one_step", reduced_crossfit="pooled")
        assert np.isfinite(fit["ate"].psi)


def test_the_module_under_test_is_the_one_imported() -> None:
    """`fit_inner_designs` is monkeypatched above by name; this fails if it moves."""
    assert drtmle_module.fit_inner_designs is nuisance_module.fit_inner_designs


def _identity_gap(build: Any) -> tuple[int, float]:
    """``(refits, worst gap)`` between each slab's own-fold rows and the production state.

    Slab ``k`` at fold ``k``'s rows holds the production arrays, and it is moved along a
    covariate that equals the production one there.  So at every refit its outcome
    regression, its mechanism and its three reduced regressions must equal the production
    ones at those rows exactly.  A slab moved by a different step, or refitted at a stale
    state, breaks this at the first refit that reads it.
    """
    worst = 0.0
    refits = 0
    inputs = targeting_module._reduction_inputs

    def check(
        nuisance: Any,
        targeted: Any,
        mechanism: Any,
        reduced: Any,
        slabs: Any = None,
        companion: Any = None,
    ) -> Any:
        nonlocal worst, refits
        if slabs is not None:
            refits += 1
            assignment = np.asarray(nuisance.folds.assignment)
            for fold in range(slabs.n_folds):
                rows = assignment == fold
                gaps = [
                    np.asarray(slabs.outcome[fold].arms[arm])[rows]
                    - np.asarray(targeted.arms[arm])[rows]
                    for arm in slabs.arms
                ]
                gaps.append(np.asarray(slabs.mechanism[fold])[rows] - np.asarray(mechanism)[rows])
                gaps.extend(
                    np.asarray(getattr(slabs.reduced[fold], name))[rows]
                    - np.asarray(getattr(reduced, name))[rows]
                    for name in ("qr", "gr1", "gr2")
                )
                worst = max(worst, *(_largest(gap) for gap in gaps))
        return inputs(nuisance, targeted, mechanism, reduced, slabs, companion)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(targeting_module, "_reduction_inputs", check)
        build()
    return refits, worst


class TestEachSlabHoldsTheProductionStateAtItsOwnFold:
    """The production identity, at every refit of a real (unscripted) fit."""

    @pytest.mark.parametrize("settings", CONFIGURATIONS, ids=lambda each: each["update_order"])
    def test_at_learning_primaries(self, settings: dict[str, Any]) -> None:
        refits, worst = _identity_gap(
            lambda: _fit(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                **settings,
            )
        )
        assert refits >= 2
        assert worst == 0.0

    @pytest.mark.parametrize("evaluation", [False, True], ids=["plain", "companion"])
    def test_with_strata(self, evaluation: bool) -> None:
        from tests.unit.test_stratified_drtmle_exact import SEED, rows

        frame = rows(360, SEED + 2)[["V", "W", "A", "Y"]]
        refits, worst = _identity_gap(
            lambda: DRTMLE(
                estimands=("ey", "ate"),
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                simultaneous=False,
                random_state=0,
                evaluation=frame if evaluation else None,
            ).fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        )
        assert refits >= 2
        assert worst == 0.0


class TestALeaveTwoOutComplementMustTrainEveryArm:
    """A pair of folds that holds every row of an arm leaves its slab nothing to fit.

    The leave-two-out primary regressions train on the whole sample outside the pair, so
    the check is per arm over the whole sample.  A stratum's reduced regressions for fold
    ``k`` train on the fold-``k`` complement, which ``refuse_untrainable_stratum_folds``
    checks, so a stratum whose arm sits in two folds is not refused here.
    """

    def test_such_a_split_is_refused_by_name(self) -> None:
        data = causal_data()
        treated = np.asarray(data.treatment, dtype=float) == 1.0
        assignment = np.where(treated, np.arange(data.n) % 2, np.arange(data.n) % FOLDS)
        with pytest.raises(ValueError, match="folds 0 and 1 together hold every trainable row"):
            refuse_untrainable_leave_two_out(data, Folds(assignment, FOLDS))

    def test_a_split_that_spreads_each_arm_passes(self) -> None:
        refuse_untrainable_leave_two_out(causal_data(), _folds())

    def test_a_stratum_whose_arm_sits_in_two_folds_passes(self) -> None:
        """Stratum ``W = 2``'s treated rows are all in folds 0 and 1, and the fit accepts it.

        The arm still has rows in fold 2 in the other strata, so every leave-two-out fit
        trains on it.  Each one-fold complement still holds a treated row of the stratum,
        so the stratum check passes too.  The precondition asserts are the witness: the
        split does put the stratum's arm in two folds, which a per-stratum pair check
        refused.
        """
        frame = law.frame()
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W"], strata=["W"]
        )
        rows = np.arange(data.n)
        corner = (frame["W"].to_numpy() == 2.0) & (frame["A"].to_numpy() == 1.0)
        assignment = np.where(corner, rows % 2, rows % FOLDS)
        folds = Folds(assignment, FOLDS)
        assert set(np.unique(assignment[corner])) == {0, 1}
        assert np.any((assignment == 2) & (frame["A"].to_numpy() == 1.0))

        refuse_untrainable_leave_two_out(data, folds)
        refuse_untrainable_stratum_folds(data, folds)


def _three_arms() -> Any:
    """A 900-row, three-arm draw: one normal covariate and a logistic outcome."""
    rng = np.random.default_rng(1)
    covariate = rng.normal(size=900)
    weights = np.exp(np.column_stack([0.0 * covariate, 0.5 * covariate, -0.5 * covariate]))
    weights /= weights.sum(axis=1, keepdims=True)
    treatment = np.array([rng.choice(3, p=row) for row in weights])
    outcome = rng.random(900) < 1.0 / (1.0 + np.exp(-(covariate + 0.3 * treatment)))
    return pd.DataFrame({"W": covariate, "A": treatment, "Y": outcome.astype(float)})


class TestThreeArms:
    """The identity and the pair cache at three arms, where each mechanism is a matrix."""

    @pytest.mark.parametrize("order", ["drtmle", "benkeser"])
    def test_each_slab_holds_the_production_state_at_its_own_fold(self, order: str) -> None:
        refits, worst = _identity_gap(
            lambda: DRTMLE(
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=FOLDS,
                learner_folds=2,
                simultaneous=False,
                random_state=0,
                update_order=order,
            ).fit(_three_arms(), outcome="Y", treatment="A")
        )
        assert refits >= 2
        assert worst == 0.0

    def test_each_pair_of_folds_is_fitted_once_and_serves_both_slabs(self) -> None:
        """Slab ``k`` at fold ``m`` and slab ``m`` at fold ``k`` read one model, fitted once.

        The longhand fit on the rows outside both folds is the witness: a pair served by
        two different training sets, or a slab handed another pair's model, fails it.
        """
        frame = _three_arms()
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
        folds = make_folds(data.n, 4, random_state=0)
        learner = sklearn.linear_model.LogisticRegression(max_iter=1000)
        fits: list[int] = []
        original = nuisance_module.fit_on_rows

        def counted(*args: Any) -> Any:
            fits.append(1)
            return original(*args)

        base = NuisanceEstimates(
            propensity=Propensity(np.full((data.n, 3), 1.0 / 3.0), data.arm_codes),
            outcome=InitialFit(
                np.full(data.n, 0.5), {arm: np.full(data.n, 0.5) for arm in data.arm_codes}
            ),
            scaler=OutcomeScaler(0.0, 1.0),
            folds=folds,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(nuisance_module, "fit_on_rows", counted)
            inner = fit_inner_designs(
                data, base, outcome_learner=learner, treatment_learner=learner
            )
        assert len(fits) == 2 * 4 * 3 // 2
        assignment = np.asarray(folds.assignment)
        design = data.mechanism_covariates
        for first in range(4):
            for second in range(first + 1, 4):
                outside = (assignment != first) & (assignment != second)
                model = sklearn.base.clone(learner).fit(
                    design[outside], np.asarray(data.treatment)[outside]
                )
                for slab, rows in ((first, second), (second, first)):
                    expected = model.predict_proba(design[assignment == rows])
                    got = np.asarray(inner.propensity[slab].values)[assignment == rows]
                    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        # The own-fold rows are the production arrays, not a refit.
        for fold in range(4):
            own = assignment == fold
            assert np.all(np.asarray(inner.propensity[fold].values)[own] == 1.0 / 3.0)


class TestAScreenedMechanismIsScreenedInEverySlab:
    """``screen_treatment=True`` screens each leave-two-out fit as it screens production."""

    def test_the_slab_mechanism_is_the_screened_fit_outside_both_folds(self) -> None:
        from tests.unit.test_stratified_drtmle_exact import SEED, rows

        frame = rows(360, SEED + 2)[["V", "W", "A", "Y"]]
        frame = frame.assign(N=np.random.default_rng(0).normal(size=len(frame)))
        covariates = ["W", "V", "N"]
        result = (
            DRTMLE(
                estimands=("ate",),
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=FOLDS,
                simultaneous=False,
                random_state=0,
                screen_treatment=True,
                screen_threshold=0.2,
                min_retain=1,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )
        nuisance = result.repeats[0].nuisance
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=covariates)
        assignment = np.asarray(nuisance.folds.assignment)

        def gap(learner: Any) -> float:
            worst = 0.0
            for slab in range(FOLDS):
                for other in range(FOLDS):
                    if other == slab:
                        continue
                    outside = (assignment != slab) & (assignment != other)
                    model = sklearn.base.clone(learner).fit(
                        data.mechanism_covariates[outside], data.treatment[outside]
                    )
                    expected = model.predict_proba(data.mechanism_covariates[assignment == other])
                    got = nuisance.inner.propensity[slab].arm(1.0)[assignment == other]
                    worst = max(worst, _largest(got - expected[:, 1]))
            return worst

        plain = sklearn.linear_model.LogisticRegression(max_iter=1000)
        assert gap(_screened(plain, 0.2, 1)) <= 1e-12
        assert gap(plain) > 1e-3, "the screen dropped nothing, so the check is vacuous"


class TestAnAlternationThatRunsOutOfRoundsIsAFailure:
    """A cap exit whose closing pass misses ``spec.tol`` reports ``max_iter_reached``.

    The degenerate case: the 60-cluster informative law of
    ``tests/unit/test_cluster_ratio_variance.py`` at ``g_bounds=(1e-9, 1 - 1e-9)``.  A
    foldwise slab takes the pooled ``epsilon`` without a solve of its own, and with no real
    truncation its mechanism runs toward 0.  The foldwise fit runs to ``max_outer`` with a
    standard error near ``1.4e5`` while every closing score is inside ``_UNSOLVED``.  So the
    flag has to come from the cap and not from the scores.  The pooled fit of the same draw
    exits on tolerance and is the control: it reports no failure and no warning.
    """

    @pytest.fixture(scope="class")
    def fits(self) -> dict[str, Any]:
        from tests.studies import clustered_unequal_laws as laws
        from tests.unit.test_cluster_ratio_variance import UNIT_LAW, ExactPropensity

        frame = laws.draw_informative(60, np.random.default_rng(20261010), **UNIT_LAW)
        out: dict[str, Any] = {}
        for crossfit in ("foldwise", "pooled"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = (
                    DRTMLE(
                        outcome_learner=sklearn.linear_model.LogisticRegression(
                            C=1e6, max_iter=1000
                        ),
                        treatment_learner=OracleTreatment(ExactPropensity()),
                        reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
                        reduced_mechanism_residual_learner=sklearn.linear_model.LinearRegression(),
                        reduced_treatment_learner=sklearn.linear_model.LogisticRegression(
                            max_iter=1000
                        ),
                        cross_fit=True,
                        n_folds=5,
                        learner_folds=2,
                        estimands=("ey0",),
                        simultaneous=False,
                        random_state=0,
                        g_bounds=(1e-9, 1.0 - 1e-9),
                        reduced_crossfit=crossfit,
                    )
                    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
                    .single()
                )
            out[crossfit] = (result, caught)
        return out

    def test_the_foldwise_fit_runs_to_the_cap_unsettled(self, fits: dict[str, Any]) -> None:
        result, _ = fits["foldwise"]
        reduction = result.fluctuations["mean"].reduction
        assert reduction.exit_reason == "cap"
        assert reduction.rounds == reduction.max_outer
        assert not reduction.converged
        assert result["ey0"].std_error > 1e3

    def test_it_is_recorded_as_a_targeting_failure(self, fits: dict[str, Any]) -> None:
        result, _ = fits["foldwise"]
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.reduction.failure == "max_iter_reached"
        assert fluctuation.mechanism.failure == "max_iter_reached"
        rows = {row.name: row for row in result.diagnostics.score_equations().rows}
        assert rows["mean (reduced)"].failure == "max_iter_reached"

    def test_and_it_warns(self, fits: dict[str, Any]) -> None:
        _, caught = fits["foldwise"]
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert any("ran out of its 50 rounds" in message for message in messages), messages

    def test_the_pooled_control_exits_on_tolerance_unflagged(self, fits: dict[str, Any]) -> None:
        result, caught = fits["pooled"]
        reduction = result.fluctuations["mean"].reduction
        assert reduction.exit_reason == "tolerance"
        assert reduction.failure is None
        assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
