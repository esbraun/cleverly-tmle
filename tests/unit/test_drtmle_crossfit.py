r"""The public ``reduced_crossfit`` contract: where each construction trains, and what moves.

``DRTMLE(reduced_crossfit=...)`` takes ``"pooled"`` (the default) or ``"nested"``, and
``docs/drtmle.md``'s *Reduced-regression cross-fitting* is the argument the pair exists for.
The short version: the cross-fitting argument for the cheap construction splits into a term
the ordinary lemma reaches and a term it does not, and what is left over is
:math:`\Delta_k` -- the difference between fold ``k``'s reduced regression as fitted and the
same regression fitted on designs and targets that never saw fold ``k``.  That second object
*is* ``"nested"``, so :math:`\Delta_k` is the pooled-minus-nested difference and the keyword
is a **diagnostic rather than a tuning one**.  This module is what says each arm computes
what it claims to.

**Five kinds of claim, and they are different kinds.**

* **Routing, structural.**  The training rows of fold ``k``'s reduced regression read the
  arrays that arm says they do -- fold-free under ``"nested"``, production under
  ``"pooled"`` -- and the row it *predicts* reads the production one either way.  Asserted at
  the call site rather than inferred from a number, because both designs are plausible arrays
  of the right shape and swapping them changes an estimator silently.
* **Routing, longhand.**  With a saturated reduction learner each construction's fitted value
  is a cell mean over a nameable set of rows, and both are written out: ``"nested"`` leaves
  two folds out, ``"pooled"`` leaves one.
* **The supplied split is honoured exactly.**  Entry ``k`` of the fold-free designs comes
  from models fitted with outer fold ``k`` removed, against the caller's own partition and no
  second split; a set built against another partition is refused.
* **Degeneracy and non-vacuity together.**  Where the fold-free arrays coincide with the
  production ones the two constructions coincide **bit for bit**, which is the only thing
  pinning the whole transfer through the alternation in one equality; and where they do not
  coincide the two **differ**, without which every equality here would pass for the wrong
  reason.
* **Invariance.**  Reordering the rows reorders the answer and changes nothing else.

Both arms are held to the same score and identity checks, so neither is green by being
exercised more gently than the other.

**One neighbouring claim deliberately lives elsewhere.**  That a companion's fold weights are
the **held-out row counts** -- ``n_k/n``, not equal weights, which coincide only on a balanced
split -- is asserted in :mod:`tests.unit.test_drtmle_companion` beside the rest of the
``evaluation=`` contract.  It is a property of :class:`~cleverly.estimators._nuisance.
CompanionEstimates` rather than of this keyword, and asserting it in two places would let the
two drift.

**What the degeneracy control corrects.** A fixture selected where two expressions cannot differ
:mod:`tests.unit.test_influence_gateaux_drtmle` is silent about this construction "because
every conditioning cell is a singleton at saturated reductions".  That reason is wrong twice
over: on this law the design takes three values over a thousand rows, so the cells are not
singletons; and saturation of the *reduction* has nothing to do with it -- under a primary
learner that learns, an inner model and an outer model disagree and any reduction learner
returns different arrays.  The operative reason is that the module fits at
``cross_fit=False`` and at oracle primary learners: one fold has no complement to nest
inside, and a learner that ignores its training rows returns the same function whichever rows
it saw.  :class:`TestADataIndependentPrimaryLearnerMakesTheTwoConstructionsAgree` is that
corrected statement, asserted rather than described -- and it is deliberately kept as a
mutation watched to **pass**, because it is the shape of agreement that says nothing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.data import CausalData
from cleverly.datasets import make_linear_ate
from cleverly.estimators import DRTMLE
from cleverly.estimators import _nuisance as nuisance_module
from cleverly.estimators import drtmle as drtmle_module
from cleverly.estimators import reduced as reduced_module
from cleverly.estimators._nuisance import (
    InitialFit,
    InnerDesigns,
    Propensity,
    fit_inner_designs,
)
from cleverly.estimators.reduced import ReducedSet, fit_reduced
from cleverly.inference import cross_validated_variance
from cleverly.learners import Folds, make_folds
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
    """
    frame, _ = make_linear_ate(n=101, seed=17)
    return (
        DRTMLE(
            estimands=("ey1", "ey0"),
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
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
    reduced, _, _ = fit_reduced(
        causal_data(),
        base if inner is None else replace(base, inner=inner),
        regression_learner=CellMeans(),
        classification_learner=CellMeans(),
        g_bounds=INERT,
        crossfit="pooled" if inner is None else "nested",
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

        assert calls, "the nested path fitted nothing; the spy is on the wrong name"
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
    r"""The **default** arm, written out rather than left as the nested arm's baseline.

    ``"pooled"`` reuses the primary split as it stands: fold ``k``'s reduced regression trains
    on the complement of fold ``k``, at the **production** designs and targets, and predicts
    there too.  That is one fold left out where
    :class:`TestTheEvaluationDesignStaysTheProductionOne` leaves two, and the difference
    between those two sentences is the whole of what the keyword selects.

    Written as a cell mean for the same reason the nested longhand is: a saturated learner
    makes the fitted value an exact finite average over a nameable set of rows, so a routing
    mistake is a wrong number here rather than a plausible one.  Without this class, ``pooled``
    is only ever asserted as *the thing nested equals in the degenerate case*, which cannot
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
        the nested arm as well and the class would be checking nothing about the default.
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
    """Entry ``k`` is fitted with outer fold ``k`` removed, against the caller's own split.

    The construction is one keyword of
    :func:`~cleverly.learners.crossfit.cross_fit_predictions` -- ``fit_mask`` with fold ``k``'s
    rows dropped -- so there is no second split and no new randomness to check.  What there
    *is* to check is that the mask is fold ``k``'s complement and that the partition passed
    through is the object the caller supplied.  An off-by-one would nest every fold inside its
    neighbour and hand back arrays of exactly the right shape, holding exactly plausible
    probabilities, which is why this is asserted at the call site.
    """

    def test_entry_k_removes_exactly_fold_k_from_the_supplied_split(self, monkeypatch: Any) -> None:
        folds = _folds()
        base = replace(nuisances(WRONG_G, WRONG_Q, folds=folds), folds=folds)
        seen: list[tuple[Any, np.ndarray]] = []
        real = nuisance_module.cross_fit_predictions

        def spy(learner: Any, design: Any, target: Any, weights: Any, split: Any, **kw: Any) -> Any:
            seen.append((split, np.asarray(kw["fit_mask"], dtype=bool).copy()))
            return real(learner, design, target, weights, split, **kw)

        monkeypatch.setattr(nuisance_module, "cross_fit_predictions", spy)
        fit_inner_designs(
            causal_data(),
            base,
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
        )

        assert len(seen) == 2 * FOLDS, "one mechanism and one regression refit per outer fold"
        assignment = np.asarray(folds.assignment)
        observed = np.asarray(causal_data().observed, dtype=bool)
        for fold in range(FOLDS):
            mechanism_split, mechanism_mask = seen[2 * fold]
            regression_split, regression_mask = seen[2 * fold + 1]
            # The *same object*, not merely an equal partition: a copy would be a second
            # split, and the whole claim of the construction is that there is not one.
            assert mechanism_split is folds
            assert regression_split is folds
            assert np.array_equal(mechanism_mask, assignment != fold)
            assert np.array_equal(regression_mask, observed & (assignment != fold))


class TestIdenticalInnerDesignsReproduceThePooledArrays:
    """Where the fold-free copies *are* the production arrays, so are the reductions.

    The array-level half of the degenerate control below, and the cheaper one: it needs no
    alternation, so a failure here localises to :func:`~cleverly.estimators.reduced.
    _nested_column` rather than to the transfer through the loop.
    """

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_every_reduction_is_bit_for_bit_the_pooled_one(self, name: str) -> None:
        folds = _folds()
        pooled = _fitted(None, folds=folds)
        nested = _fitted(_inner_from(nuisances(WRONG_G, WRONG_Q, folds=folds)), folds=folds)
        assert np.array_equal(getattr(pooled, name), getattr(nested, name))


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
                g_bounds=INERT,
                crossfit="nested",
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
    return {"pooled": _fit(), "nested": _fit(reduced_crossfit="nested")}


class TestADataIndependentPrimaryLearnerMakesTheTwoConstructionsAgree:
    """**A mutation watched to pass**, and that is what it is for.

    An oracle primary learner returns the same function whichever rows trained it, so every
    inner model and every outer model coincide and the two constructions must agree to the
    bit.  Asserted because it pins the plumbing -- a nested path that trained on the wrong
    rows would break it -- and labelled because agreement here is evidence about nothing
    else. This precondition keeps the identity check non-degenerate.
    """

    def test_the_estimates_are_identical(self, oracle_pair: dict[str, Any]) -> None:
        pooled, nested = oracle_pair["pooled"]["ate"], oracle_pair["nested"]["ate"]
        assert pooled.psi == nested.psi
        assert pooled.variance == nested.variance

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_the_reductions_are_identical(self, oracle_pair: dict[str, Any], name: str) -> None:
        pooled = getattr(oracle_pair["pooled"].nuisance.reduced, name)
        nested = getattr(oracle_pair["nested"].nuisance.reduced, name)
        assert np.array_equal(pooled, nested)

    def test_and_the_nested_fit_really_did_build_the_designs(
        self, oracle_pair: dict[str, Any]
    ) -> None:
        """Otherwise the agreement above is the pooled path passing under a keyword."""
        assert oracle_pair["pooled"].nuisance.inner is None
        assert oracle_pair["nested"].nuisance.inner is not None
        assert oracle_pair["nested"].nuisance.inner.n_folds == FOLDS


class TestADegenerateInnerDesignReproducesThePooledFit:
    r"""The transfer through the alternation, pinned in one equality.

    Equations (9) and (10) are stated at the *targeted* collection, so every refit inside the
    loop needs the fold-free arrays moved by the same fluctuation the production ones took.
    They are **carried through the solvers** rather than reconstructed from
    ``(initial, epsilon)``: the outcome tilt is applied once per Newton step and shrunk after
    each, and the mechanism tilt is clipped into ``g_bounds``, so a net offset reproduces the
    endpoint only on a fit where nothing touched a bound.

    With the fold-free copies set equal to the production arrays the nested fit must
    therefore be the pooled fit exactly -- ``psi``, the variance, the whole curve, every
    ``epsilon`` and the round count.  Skip the carry in either funnel and the two diverge
    after the first refit.
    """

    @pytest.fixture(scope="class")
    def pair(self, request: Any) -> dict[str, Any]:
        real = drtmle_module.fit_inner_designs

        def degenerate(data: Any, nuisance: Any, **kwargs: Any) -> InnerDesigns:
            return _inner_from(nuisance)

        drtmle_module.fit_inner_designs = degenerate
        try:
            nested = _fit(reduced_crossfit="nested")
        finally:
            drtmle_module.fit_inner_designs = real
        return {"pooled": _fit(), "nested": nested}

    def test_the_estimate_and_the_curve_are_identical(self, pair: dict[str, Any]) -> None:
        for name in ("ey1", "ey0", "ate"):
            pooled, nested = pair["pooled"][name], pair["nested"][name]
            assert pooled.psi == nested.psi
            assert pooled.variance == nested.variance
            assert np.array_equal(pooled.influence_curve, nested.influence_curve)

    def test_the_fluctuation_took_the_same_route(self, pair: dict[str, Any]) -> None:
        pooled = pair["pooled"].fluctuations["mean"]
        nested = pair["nested"].fluctuations["mean"]
        assert np.array_equal(pooled.epsilon, nested.epsilon)
        assert pooled.reduction.n_outer == nested.reduction.n_outer
        assert pooled.reduction.exit_reason == nested.reduction.exit_reason


def _learning_pair(**settings: Any) -> dict[str, Any]:
    """The same fit under both constructions, at primary learners that *do* learn."""
    return {
        "pooled": _fit(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            **settings,
        ),
        "nested": _fit(
            reduced_crossfit="nested",
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
            reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        )

    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_the_reductions_move(self, pair: dict[str, Any], name: str) -> None:
        pooled = np.asarray(getattr(pair["pooled"].nuisance.reduced, name))
        nested = np.asarray(getattr(pair["nested"].nuisance.reduced, name))
        assert float(np.max(np.abs(pooled - nested))) > 1e-12

    def test_both_fits_still_solve_their_equations(self, pair: dict[str, Any]) -> None:
        """A construction that moved the arrays and broke the loop would prove nothing."""
        for fit in pair.values():
            assert fit.validation.score_check().passed

    def test_a_retarget_of_a_nested_fit_is_still_nested(self, pair: dict[str, Any]) -> None:
        r"""The sensitivity analyses re-enter the alternation, and must not fall back to pooled.

        ``DRTMLE.retarget`` refits the reduced regressions -- they are regressions *of* the
        current pair, so they cannot be cached arithmetic the way an ordinary fit's are --
        and it reads the construction off the estimator while the fold-free designs ride on
        :attr:`~cleverly.estimators._nuisance.NuisanceEstimates.inner`.  If either half went
        missing, a truncation curve on a nested fit would silently report a *pooled*
        estimator's sensitivity under the nested fit's name.

        The witness is that the retargeted estimate differs from the pooled fit's at the same
        bound: a fallback would make the two identical.
        """
        nested, pooled = pair["nested"], pair["pooled"]
        estimator = DRTMLE(
            reduced_crossfit="nested",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
            reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            estimands=("ate",),
            cross_fit=True,
            n_folds=FOLDS,
            learner_folds=3,
            simultaneous=False,
            random_state=0,
        )
        data = nested.data
        assert nested.nuisance.inner is not None
        estimates, _ = estimator.retarget(data, nested.nuisance, estimands=("ate",), g_bounds=INERT)
        assert estimates["ate"].psi != pooled["ate"].psi


class TestEachRepeatedDrawGetsItsOwnFoldFreeDesigns:
    """``repeats=`` redraws the primary split, and the designs are keyed to *a* split.

    This is why the fold-free copies live on
    :class:`~cleverly.estimators._nuisance.NuisanceEstimates` rather than on the estimator:
    ``_nuisances`` runs once per draw, so each draw builds its own against its own folds and
    the two cannot come apart.  Carrying them on the estimator would look identical on an
    ordinary fit and be wrong only here -- one draw's designs nested inside another draw's
    split, which is not a construction at all.
    """

    @pytest.fixture(scope="class")
    def repeated(self) -> Any:
        return DRTMLE(
            reduced_crossfit="nested",
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
    arms should find this class rather than conclude the nested path is inert.  The class
    above uses ``glm`` reductions precisely to leave the degeneracy.
    """

    @pytest.fixture(scope="class")
    def pair(self) -> dict[str, Any]:
        return _learning_pair(
            reduced_outcome_learner=CellMeans(), reduced_treatment_learner=CellMeans()
        )

    def test_gr1_does_not_move_and_that_is_the_partition(self, pair: dict[str, Any]) -> None:
        pooled = np.asarray(pair["pooled"].nuisance.reduced.gr1)
        nested = np.asarray(pair["nested"].nuisance.reduced.gr1)
        assert np.array_equal(pooled, nested)

    @pytest.mark.parametrize("name", ["qr", "gr2"])
    def test_the_two_with_estimated_targets_still_move(
        self, pair: dict[str, Any], name: str
    ) -> None:
        pooled = np.asarray(getattr(pair["pooled"].nuisance.reduced, name))
        nested = np.asarray(getattr(pair["nested"].nuisance.reduced, name))
        assert float(np.max(np.abs(pooled - nested))) > 1e-12


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

    @pytest.mark.parametrize("crossfit", ["pooled", "nested"])
    @pytest.mark.parametrize("name", ["qr", "gr1", "gr2"])
    def test_the_arrays_come_back_permuted(self, crossfit: str, name: str) -> None:
        folds = _folds()
        order = np.random.default_rng(7).permutation(law.N)
        base = nuisances(WRONG_G, WRONG_Q, folds=folds)
        # A non-degenerate inner set, so the nested arm is genuinely reading its own arrays
        # rather than reproducing the pooled ones under a keyword.
        inner = None if crossfit == "pooled" else _inner_from(base, permute=np.array([1, 2, 0]))
        straight = _fitted(inner, folds=folds)

        shuffled_folds = Folds(np.asarray(folds.assignment)[order], FOLDS)
        shuffled, _, _ = fit_reduced(
            CausalData.from_frame(
                law.frame().iloc[order].reset_index(drop=True),
                outcome="Y",
                treatment="A",
                covariates=["W"],
            ),
            _permuted(base, inner, order, shuffled_folds),
            regression_learner=CellMeans(),
            classification_learner=CellMeans(),
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
    """``pooled`` is the default and a pooled fit is untouched by any of this."""

    def test_the_default_builds_no_inner_designs(self, oracle_pair: dict[str, Any]) -> None:
        assert DRTMLE().reduced_crossfit == "pooled"
        assert oracle_pair["pooled"].nuisance.inner is None

    def test_an_empty_guard_builds_none_either(self) -> None:
        fit = _fit(guard=(), reduced_crossfit="nested")
        assert fit.nuisance.inner is None

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"reduced_crossfit": "fold"}, "must be one of"),
            ({"reduced_crossfit": "nested", "cross_fit": False}, "needs cross-fitting"),
            ({"reduced_crossfit": "nested", "n_folds": 2}, "leaves two folds out"),
        ],
    )
    def test_it_says_what_the_construction_would_need(
        self, settings: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            DRTMLE(**settings)

    def test_the_one_step_walk_is_refused_by_name(self) -> None:
        """A cost decision rather than a derivation, and it says so."""
        with pytest.raises(NotImplementedError, match="one_step"):
            _fit(reduced_crossfit="nested", targeting="one_step")


def test_the_module_under_test_is_the_one_imported() -> None:
    """`fit_inner_designs` is monkeypatched above by name; this fails if it moves."""
    assert drtmle_module.fit_inner_designs is nuisance_module.fit_inner_designs
