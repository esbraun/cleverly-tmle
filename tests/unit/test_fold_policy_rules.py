"""What the fold policy and the outcome scale are now held to, and the proof of each.

Every rule here has the same shape. A cross-fitted fit conditions on the split it drew, so
the split must not be a function of the values the fit then analyses, and the scale the
nuisances are fitted on must not be either. The rules that follow are the two halves of
that sentence, plus the checks that replace what stratification used to buy.

Each witness comes with a **deliberate-mutation control**: a monkeypatch that puts the
wrong implementation into production and asserts the witness then fails. The precedent is
``unweight`` in ``tests/conftest.py``. Without one, a witness that would pass against any
implementation reads exactly like a witness that pins the right one, and an invariance
claim is the shape most at risk of that: two fits agree for many reasons, and only one of
them is the reason the test is named after.
"""

from __future__ import annotations

import importlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import variable_importance
from cleverly.data import CausalData
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_longitudinal,
    make_nonlinear_bounded,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError, LongitudinalError
from cleverly.interventions import Shift
from cleverly.learners import SuperLearner, random_partition
from cleverly.learners.crossfit import _MAX_SEED
from cleverly.longitudinal import LTMLE
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law
from tests.conftest import linear_in_sample
from tests.studies import (
    multi_arm_common,
    multi_arm_ctmle_selector_properties,
    multi_arm_drtmle_properties,
    multi_arm_properties,
)
from tests.unit._capability_sweep_support import (
    cross_fitted,
    dose_frame,
    fit_shift,
    outcome_bounds,
    reconfigured,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners

BOUNDED_COVARIATES = ["W1", "W2", "W3"]


#: Every nuisance fit the counting learners below have run, since the last reset. A module
#: global rather than a constructor argument because scikit-learn's ``clone`` deep-copies
#: every parameter, so a list handed to ``__init__`` reaches each fold as a fresh copy and
#: counts nothing.
_FIT_COUNTER: list[str] = []


def reset_counter() -> None:
    """Forget every recorded fit, so a case counts only its own."""
    _FIT_COUNTER.clear()


class CountingLogistic(LogisticRegression):
    """A logistic regression that records every fit, so "no learner ran" is checkable.

    A refusal that fires before any nuisance is fitted and a refusal that fires after two
    of them produce the same exception, and only the counter separates them. That matters
    here because the whole claim of a preflight is *when* it refuses.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        _FIT_COUNTER.append("treatment")
        return super().fit(X, y, sample_weight=sample_weight)


class CountingLinear(LinearRegression):
    """The regression counterpart of :class:`CountingLogistic`."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        _FIT_COUNTER.append("outcome")
        return super().fit(X, y, sample_weight=sample_weight)


def bounded_fit(**overrides: Any) -> TMLE:
    """A cross-fitted estimator on a law whose support the fit can declare."""
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "q_bounds": (0.0, 1.0),
        "n_folds": 5,
        "learner_folds": 3,
        "random_state": 0,
        "simultaneous": False,
        "estimands": ["ate"],
    }
    settings.update(overrides)
    return TMLE(**settings)


def permute_arms_and_outcomes(frame: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Reassign ``A`` and ``Y`` across rows, leaving ``n``, the row order and ``W`` alone.

    This is the perturbation an unstratified draw must be blind to. It keeps the two
    columns' marginal distributions and destroys their association with everything else,
    so a split that reads either one moves and a split that reads neither cannot.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    moved = frame.copy()
    moved["A"] = frame["A"].to_numpy()[order]
    moved["Y"] = frame["Y"].to_numpy()[order]
    return moved


# --------------------------------------------------------------- W1: the outer draw


class TestTheOuterSplitReadsNeitherArmNorOutcome:
    """W1. The shipped default draws folds from the seed, and from nothing else."""

    @staticmethod
    def assignment(frame: pd.DataFrame) -> np.ndarray:
        result = (
            bounded_fit()
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        return np.asarray(result.nuisance.folds.assignment)

    def test_the_generated_folds_do_not_move_when_the_arms_and_outcomes_do(self) -> None:
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        np.testing.assert_array_equal(
            self.assignment(frame), self.assignment(permute_arms_and_outcomes(frame))
        )

    @pytest.mark.parametrize("mutant", ["treatment", "treatment+outcome"])
    def test_a_split_that_reads_them_moves(
        self, monkeypatch: pytest.MonkeyPatch, mutant: str
    ) -> None:
        """The control. A split balanced on the arms is a function of the arms.

        ``_fold_strata`` is the one place the estimator decides what the outer split is
        balanced on, and the shipped policy makes it return ``None``. Putting the retired
        policies back through it is the mutation, and the witness above has to fail under
        both.
        """

        def strata(self: TMLE, data: Any) -> np.ndarray:
            arm = np.asarray(data.treatment, dtype=float)
            if mutant == "treatment":
                return arm
            outcome = np.asarray(data.outcome, dtype=float)
            # Two outcome levels rather than every distinct value: a stratum per value
            # caps the fold count at one row and refuses to split at all, which would
            # make the mutant fail for a reason that is not the one under test.
            return arm * 2.0 + (outcome > float(np.median(outcome)))

        monkeypatch.setattr(TMLE, "_fold_strata", strata)
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        assert not np.array_equal(
            self.assignment(frame), self.assignment(permute_arms_and_outcomes(frame))
        ), f"the {mutant!r} split did not move when the arms and outcomes did"


# ------------------------------------------------- W2 and W8: the declaration refusals


class TestAPolicyThisVersionRefusesNeverReachesALearner:
    """W2 and W8. The fold-policy refusals fire at the declaration and again at the fit."""

    @pytest.mark.parametrize("policy", ["treatment", "treatment+outcome"])
    @pytest.mark.parametrize("engine", [TMLE, DRTMLE])
    def test_a_stratified_policy_is_refused_at_construction(
        self, policy: str, engine: type
    ) -> None:
        with pytest.raises(ValueError, match="No shipped result covers that split"):
            engine(stratify_folds=policy, n_folds=5, random_state=0)

    def test_one_fold_under_cross_fitting_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Set n_folds to at least 2"):
            TMLE(cross_fit=True, n_folds=1)

    def test_a_restored_estimator_is_refused_before_its_first_learner(self) -> None:
        """R9. An estimator that skipped ``__init__`` is caught at the fit instead.

        ``refit`` copies an estimator and a pickle restores one without running the
        constructor, so a saved fit can arrive carrying a policy this version refuses.
        The counter is what says the refusal precedes the nuisances rather than following
        them.
        """
        reset_counter()
        estimator = bounded_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
        )
        estimator.stratify_folds = "treatment"
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        with pytest.raises(ValueError, match="fold policy this version refuses"):
            estimator.fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"

    def test_removing_the_refusal_lets_the_stratified_fit_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control for both refusals above, and for the counter.

        With the reason suppressed the same estimator fits, which is what says the
        counter was empty because the refusal fired and not because the fixture never
        reached a learner.
        """
        import cleverly.learners.crossfit as policy

        monkeypatch.setattr(policy, "fold_strata_refusal", lambda *a, **k: None)
        reset_counter()
        estimator = bounded_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
        )
        estimator.stratify_folds = "treatment"
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        estimator.fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
        assert _FIT_COUNTER, "the suppressed refusal did not let a single learner run"

    def test_one_fold_under_cross_fitting_would_be_named_a_cross_fitted_estimator(self) -> None:
        """Why ``n_folds=1`` with ``cross_fit=True`` is refused rather than accepted.

        ``TMLEConfig.estimator_name`` reads ``cross_fit`` alone, so a one-fold fit that
        kept the flag would publish the stacked CV-TMLE name over in-sample nuisances.
        This states the name the refusal prevents, so a future default that let the pair
        through would be visible here rather than in a published row.
        """
        fitted = (
            bounded_fit()
            .fit(
                make_nonlinear_bounded(n=200, seed=3)[0],
                outcome="Y",
                treatment="A",
                covariates=BOUNDED_COVARIATES,
            )
            .single()
        )
        assert fitted.config.estimator_name == "stacked CV-TMLE (Levy)"


# ------------------------------------------------------- W3: the complement preflight


def seed_stranding_an_arm(n: int, n_folds: int, treated: np.ndarray) -> int:
    """A seed whose draw puts every treated row in one fold, so a complement has none.

    Found by searching the generator the fit uses rather than constructed, because the
    property has to hold of a split the package itself would draw.
    """
    for seed in range(5_000):
        folds = random_partition(n, n_folds, seed=seed)
        labels = np.asarray(folds.assignment)[treated]
        if np.unique(labels).size == 1:
            return seed
    raise AssertionError("no seed in the search range stranded the arm")


class TestAStrandedArmIsRefusedBeforeAnyLearner:
    """W3. What stratification used to guarantee is now checked on the realized draw."""

    @staticmethod
    def stranded_frame() -> tuple[pd.DataFrame, int]:
        rng = np.random.default_rng(0)
        n, n_folds = 40, 10
        frame = pd.DataFrame(
            {
                "Y": rng.uniform(0.2, 0.8, size=n),
                "A": np.zeros(n),
                "W1": rng.normal(size=n),
                "W2": rng.normal(size=n),
            }
        )
        treated = np.array([0, 1])
        frame.loc[treated, "A"] = 1.0
        return frame, seed_stranding_an_arm(n, n_folds, treated)

    def test_the_fit_is_refused_with_no_learner_fitted(self) -> None:
        frame, seed = self.stranded_frame()
        reset_counter()
        estimator = bounded_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
            n_folds=10,
            random_state=seed,
        )
        with pytest.raises(DataError, match="training complement contains no row in arm"):
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"

    def test_the_refusal_names_no_redraw(self) -> None:
        """I2. A refusal raised after the draw may not send the reader back to redraw it.

        A fold count or a seed found by trying them until one fits was chosen by looking
        at the treatment, which is exactly the dependence the unstratified draw removes.
        """
        frame, seed = self.stranded_frame()
        with pytest.raises(DataError) as raised:
            bounded_fit(n_folds=10, random_state=seed).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"]
            )
        message = str(raised.value)
        for forbidden in ("Increase n_folds", "different random_state", "reduce n_folds"):
            assert forbidden not in message, f"the refusal offers {forbidden!r}"

    def test_disabling_the_preflight_lets_the_stranded_split_reach_a_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. Without the check the same draw reaches the nuisance fits."""
        monkeypatch.setattr(TMLE, "_preflight_training_support", lambda *a, **k: None)
        frame, seed = self.stranded_frame()
        reset_counter()
        estimator = bounded_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
            n_folds=10,
            random_state=seed,
        )
        with pytest.raises(Exception):  # noqa: B017 - any downstream failure will do
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        assert _FIT_COUNTER, "the disabled preflight still fitted no learner"


# ------------------------------------------------------- W-cluster: the unit is a cluster


def clustered_frame(n_clusters: int = 12, size: int = 5) -> pd.DataFrame:
    """A clustered sample whose treated rows all sit inside one cluster."""
    rng = np.random.default_rng(2)
    n = n_clusters * size
    cluster = np.repeat(np.arange(n_clusters), size)
    arm = np.where(cluster == 0, 1.0, 0.0)
    return pd.DataFrame(
        {
            "Y": rng.uniform(0.2, 0.8, size=n),
            "A": arm,
            "W1": rng.normal(size=n),
            "W2": rng.normal(size=n),
            "cid": cluster,
        }
    )


class TestAnArmInsideOneClusterIsRefused:
    """W-cluster. A grouped draw moves whole clusters, so the unit is the cluster."""

    def test_the_sample_minimum_counts_clusters(self) -> None:
        frame = clustered_frame()
        with pytest.raises(DataError, match="at least two independent units"):
            bounded_fit(n_folds=4).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cid"
            )

    def test_counting_rows_instead_misses_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control. Five treated rows pass a row count and fail a cluster count.

        ``_independent_units`` is the one place the unit is decided, so the mutation is
        to make it answer "row" on data that declared clusters.
        """
        engine = sys.modules["cleverly.estimators.tmle"]

        monkeypatch.setattr(
            engine,
            "_independent_units",
            lambda data: (np.arange(int(np.asarray(data.treatment).size)), "row"),
        )
        frame = clustered_frame()
        with pytest.raises(DataError) as raised:
            bounded_fit(n_folds=4).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cid"
            )
        assert "at least two independent units" not in str(raised.value), (
            "the row count still reached the sample minimum, so the witness proves nothing"
        )


# ----------------------------------------------------------- W4: the outcome scale


class TestADeclaredSupportKeepsTheScaleOutOfTheFolds:
    """W4. With ``q_bounds`` declared, a held-out outcome cannot move a fold's model."""

    @staticmethod
    def fold_zero_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        result = (
            bounded_fit()
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        folds = result.nuisance.folds
        held_out = folds.test_index(0)
        return held_out, np.asarray(result.nuisance.outcome.observed)[held_out]

    @staticmethod
    def with_one_outcome_moved(frame: pd.DataFrame, row: int) -> pd.DataFrame:
        moved = frame.copy()
        moved.loc[row, "Y"] = 0.999
        return moved

    def test_moving_a_held_out_outcome_leaves_its_own_folds_predictions_alone(self) -> None:
        """The prediction for a row comes from a model that never saw that row.

        With ``q_bounds=(0, 1)`` the scale is declared, so the model for fold zero is a
        function of folds one to four alone. Moving a fold-zero outcome therefore cannot
        reach it, and the predictions it makes for the other fold-zero rows stand.
        """
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        held_out, before = self.fold_zero_predictions(frame)
        moved = self.with_one_outcome_moved(frame, int(held_out[0]))
        held_out_again, after = self.fold_zero_predictions(moved)
        np.testing.assert_array_equal(held_out, held_out_again)
        np.testing.assert_allclose(before[1:], after[1:], rtol=0.0, atol=0.0)

    def test_an_undeclared_scale_lets_it_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control, and the reason the refusal exists.

        ``q_bounds=None`` takes the scale from every observed outcome, held-out rows
        included, so one moved outcome reaches every fold's model. The shipped package
        refuses that pair rather than running it, so the mutation is to ignore the
        declaration inside ``_scaler``.
        """

        def unbounded(self: TMLE, data: Any) -> OutcomeScaler:
            return OutcomeScaler.from_outcome(data.outcome[data.observed], None)

        monkeypatch.setattr(TMLE, "_scaler", unbounded)
        frame, _ = make_nonlinear_bounded(n=200, seed=3)
        held_out, before = self.fold_zero_predictions(frame)
        _, after = self.fold_zero_predictions(self.with_one_outcome_moved(frame, int(held_out[0])))
        assert not np.allclose(before[1:], after[1:]), (
            "the undeclared scale left fold zero's predictions untouched, so the witness "
            "above is not evidence about the scale"
        )


# --------------------------------------------- W5: the inner split reads training rows only


_RECORDED_INNER_SPLITS: list[tuple[int, ...]] = []


class RecordingSuperLearner(SuperLearner):
    """A Super Learner that files the inner split it drew, in fit order."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        fitted = super().fit(X, y, sample_weight=sample_weight)
        _RECORDED_INNER_SPLITS.append(tuple(int(value) for value in self.folds_.assignment))
        return fitted


class TestTheInnerSplitDoesNotReadHeldOutRows:
    """W5. A Super Learner's inner split is a function of its own training rows.

    The recorded learner is the *treatment* one, and its inner split is stratified on its
    own target, ``A``. The perturbation is therefore a perturbation of ``A``: flipping
    ``Y`` cannot move this split under any implementation, so a ``Y`` witness would pass
    against a mutation that read every row of ``A``, which is the leak being ruled out.

    A one-row flip of ``A`` moves the target of every outer fold that *trains* on that
    row, so most recorded splits move and must. The claim is about the one fold that
    holds the row out: its Super Learner never saw the row, so its split has to be the
    one it drew before. Both halves are asserted, because the unchanged half alone is
    what a dead perturbation also produces.
    """

    @staticmethod
    def inner_splits(frame: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        _RECORDED_INNER_SPLITS.clear()
        estimator = TMLE(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=RecordingSuperLearner(
                library=[("glm", LogisticRegression(max_iter=1000))],
                task="classification",
                n_folds=3,
                random_state=0,
            ),
            n_folds=4,
            random_state=0,
            simultaneous=False,
            estimands=["ate"],
        )
        result = estimator.fit(
            frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"]
        ).single()
        outer = np.asarray(result.nuisance.folds.assignment)
        return outer, list(_RECORDED_INNER_SPLITS)

    @staticmethod
    def with_one_treatment_flipped(frame: pd.DataFrame, row: int = 0) -> pd.DataFrame:
        moved = frame.copy()
        moved.loc[row, "A"] = 1.0 - float(frame.loc[row, "A"])
        return moved

    def test_flipping_a_held_out_treatment_moves_no_inner_split_of_that_fold(self) -> None:
        frame, _ = make_binary_outcome(n=200, seed=5)
        before_outer, before = self.inner_splits(frame)
        after_outer, after = self.inner_splits(self.with_one_treatment_flipped(frame))

        # The outer draw reads n, the clusters and the seed, so the fold that holds row
        # zero out is the same fold in both fits. W1 pins that property on its own.
        np.testing.assert_array_equal(before_outer, after_outer)
        held_out = int(before_outer[0])
        assert len(before) == len(after) == before_outer.max() + 1
        assert before[held_out] == after[held_out], (
            "the inner split of the fold that holds the perturbed row out moved, so the "
            "Super Learner read a row outside its training complement"
        )
        moved = [fold for fold in range(len(before)) if before[fold] != after[fold]]
        assert moved, (
            "no recorded inner split moved at all, so the perturbation is dead and the "
            "unchanged fold above is evidence about nothing"
        )
        assert held_out not in moved

    def test_an_inner_split_that_reads_every_row_does_move(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. A split stratified on a vector that includes held-out rows moves.

        The mutation is the smallest implementation that reads outside the training rows:
        the inner split is balanced on the first ``n`` entries of the whole sample's
        treatment rather than on the target the learner was handed. Under it the fold
        that holds the perturbed row out moves as well, which is what the witness above
        forbids.
        """
        import cleverly.learners.super_learner as learners

        frame, _ = make_binary_outcome(n=200, seed=5)
        leaked = {"treatment": frame["A"].to_numpy(dtype=float)}
        original = learners.make_folds

        def leaking(n: int, n_folds: int = 10, **kwargs: Any) -> Any:
            if kwargs.get("stratify") is not None:
                kwargs["stratify"] = leaked["treatment"][:n]
            return original(n, n_folds, **kwargs)

        monkeypatch.setattr(learners, "make_folds", leaking)
        before_outer, before = self.inner_splits(frame)
        moved_frame = self.with_one_treatment_flipped(frame)
        leaked["treatment"] = moved_frame["A"].to_numpy(dtype=float)
        _, after = self.inner_splits(moved_frame)
        held_out = int(before_outer[0])
        assert before[held_out] != after[held_out], (
            "the leaking inner split of the holding-out fold did not move, so the witness "
            "above is not evidence about what the inner split reads"
        )


# ------------------------------------------------- the collaborative search's own splits


def instrument_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    frame, _ = make_nonlinear_bounded(n=n, seed=seed)
    return frame


def collaborative(**overrides: Any) -> CTMLE:
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "q_bounds": (0.0, 1.0),
        "strategy": "greedy",
        "selection_folds": 3,
        "n_folds": 4,
        "learner_folds": 3,
        "random_state": 0,
        "simultaneous": False,
        "estimands": ["ate"],
    }
    settings.update(overrides)
    return CTMLE(**settings)


class TestTheSelectionSplitReadsNoArm:
    """The collaborative search draws its own split, under the same rule."""

    @staticmethod
    def selection_assignment(frame: pd.DataFrame, **overrides: Any) -> np.ndarray:
        result = (
            collaborative(**overrides)
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        return np.asarray(result.extra["ctmle"].folds.assignment)

    def test_the_selection_folds_do_not_move_when_the_arms_do(self) -> None:
        frame = instrument_frame()
        np.testing.assert_array_equal(
            self.selection_assignment(frame),
            self.selection_assignment(permute_arms_and_outcomes(frame)),
        )

    def test_a_selection_split_that_reads_the_arm_moves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_fold_strata", lambda self, data: np.asarray(data.treatment))
        frame = instrument_frame()
        assert not np.array_equal(
            self.selection_assignment(frame),
            self.selection_assignment(permute_arms_and_outcomes(frame)),
        )

    def test_the_selection_scheme_is_recorded_even_without_cross_fitting(self) -> None:
        """I3. The one split a ``cross_fit=False`` collaborative fit draws is recorded.

        ``CrossFitPlan`` describes the outer split, and an in-sample fit has none, so a
        reader who stopped there would see a fit that drew nothing. The search drew a
        selection split all the same, and its record is where that fact lives.
        """
        frame = instrument_frame()
        result = (
            collaborative(cross_fit=False, q_bounds=None)
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        assert result.config.crossfit.scheme == "none"
        origin = result.extra["ctmle"].folds.origin
        assert origin is not None, "the selection split recorded no generator"
        assert origin.generator == "cleverly.random_partition/1"
        assert origin.scheme == "vfold"
        assert origin.requested_n_folds == 3
        assert isinstance(origin.seed, int)

    def test_the_selection_split_is_the_one_the_record_redraws(self) -> None:
        """A record that does not reproduce the split describes a different draw."""
        frame = instrument_frame()
        result = (
            collaborative(cross_fit=False, q_bounds=None, random_state=7)
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        folds = result.extra["ctmle"].folds
        redrawn = random_partition(
            len(frame), folds.origin.requested_n_folds, seed=folds.origin.seed
        )
        np.testing.assert_array_equal(folds.assignment, redrawn.assignment)


class TestTheCollaborativeRefusals:
    """What a collaborative fit has no result for, refused before any learner."""

    def test_a_stratified_policy_is_refused_without_cross_fitting(self) -> None:
        with pytest.raises(ValueError, match="draws those folds whether or not cross_fit"):
            collaborative(cross_fit=False, stratify_folds="treatment")

    @pytest.mark.parametrize("policy", ["treatment", "treatment+outcome"])
    def test_in_sample_oat_accepts_an_unused_fold_policy(self, policy: str) -> None:
        from cleverly import CollaborativeTMLEMethod, CrossFitting

        method = CollaborativeTMLEMethod(
            strategy="oat", cross_fitting=CrossFitting(enabled=False, stratify_by=policy)
        )
        assert method.cross_fitting.stratify_by == policy

        frame = instrument_frame()
        result = (
            collaborative(strategy="oat", cross_fit=False, stratify_folds=policy, selection_folds=5)
            .fit(frame, outcome="Y", treatment="A", covariates=BOUNDED_COVARIATES)
            .single()
        )
        assert result.nuisance.folds.is_single
        assert result.extra["ctmle"].strategy == "oat"

    @pytest.mark.parametrize("policy", ["treatment", "treatment+outcome"])
    def test_cross_fitted_oat_still_refuses_stratification(self, policy: str) -> None:
        from cleverly import CollaborativeTMLEMethod, CrossFitting
        from cleverly.exceptions import MethodConfigurationError

        with pytest.raises(MethodConfigurationError, match="No shipped result covers that split"):
            CollaborativeTMLEMethod(
                strategy="oat", cross_fitting=CrossFitting(enabled=True, stratify_by=policy)
            )
        with pytest.raises(ValueError, match="No shipped result covers that split"):
            collaborative(strategy="oat", cross_fit=True, stratify_folds=policy)

    def test_clusters_are_refused_before_a_learner_runs(self) -> None:
        reset_counter()
        frame = clustered_frame(n_clusters=20, size=5)
        estimator = collaborative(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
        )
        with pytest.raises(CapabilityError, match="C-TMLE has no clustered result"):
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cid")
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"

    def test_oat_cluster_refusal_names_its_own_missing_result(self) -> None:
        frame = clustered_frame(n_clusters=20, size=5)
        with pytest.raises(
            CapabilityError, match="clustered inference for the outcome-adaptive"
        ) as exc:
            collaborative(strategy="oat", selection_folds=5, cross_fit=False).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cid"
            )
        assert "selection folds" not in str(exc.value)

    def test_the_method_catalog_says_so_before_anyone_fits(self) -> None:
        """A refusal a caller can read off the design, rather than meet at the fit.

        ``available_methods`` is what a caller asks before choosing, so a method the fit
        will refuse has to be absent from it. The two must agree, which is why the test
        for the refusal and the test for the catalog sit together.
        """
        from cleverly import ATE, CausalStudy, PointTreatment

        frame = clustered_frame(n_clusters=20, size=5)
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), cluster="cid")
        catalog = {
            method.name: method
            for method in CausalStudy(frame, design=design).identify(ATE()).available_methods()
        }
        assert catalog["tmle"].available, "the ordinary TMLE has a clustered result"
        assert not catalog["collaborative_tmle"].available
        assert "no clustered result" in (catalog["collaborative_tmle"].reason or "")

    def test_the_catalog_still_offers_the_search_without_clusters(self) -> None:
        """The control. The same design without ``cluster=`` keeps the method."""
        from cleverly import ATE, CausalStudy, PointTreatment

        frame = clustered_frame(n_clusters=20, size=5)
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
        catalog = {
            method.name: method
            for method in CausalStudy(frame, design=design).identify(ATE()).available_methods()
        }
        assert catalog["collaborative_tmle"].available


class TestContinuousDoseOutcomeSupport:
    """A dose has no arms, but its binary outcome still needs training support."""

    def test_a_stranded_outcome_class_is_refused_before_fitting(self) -> None:
        reset_counter()
        n = 90
        rng = np.random.default_rng(7)
        assignment = random_partition(n, 3, seed=0).assignment
        rare_rows = np.flatnonzero(assignment == 0)[:2]
        outcome = np.zeros(n)
        outcome[rare_rows] = 1.0
        frame = pd.DataFrame({"W": rng.normal(size=n), "A": rng.normal(size=n), "Y": outcome})
        estimator = TMLE(
            outcome_learner=CountingLogistic(max_iter=1000),
            shifts=[Shift(0.0, cap=None), Shift(0.5, cap=4.0)],
            n_folds=3,
            random_state=0,
            simultaneous=False,
        )
        with pytest.raises(DataError, match="training complement contains no observed outcome 1"):
            estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_kind="continuous",
            )
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"


# ------------------------------------------------------------ the longitudinal rules


def longitudinal_frame(n: int = 400, seed: int = 0, **kwargs: Any) -> pd.DataFrame:
    frame, _ = make_longitudinal(n=n, seed=seed, **kwargs)
    return frame


LONGITUDINAL_COLUMNS: dict[str, Any] = {
    "baseline": ["W1", "W2"],
    "treatment": ["A1", "A2"],
    "censoring": ["C1", "C2"],
    "time_varying": [[], ["L2"]],
    "outcome": "Y",
}


#: The settings every sequential fit in this module shares, so a seam that replaces the
#: estimator can be built from exactly the same declaration.
LONGITUDINAL_SETTINGS: dict[str, Any] = {
    "outcome_learner": LogisticRegression(max_iter=1000),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "censoring_learner": LogisticRegression(max_iter=1000),
    "pseudo_learner": LinearRegression(),
    "n_folds": 3,
    "learner_folds": 3,
    "random_state": 0,
}


def sequential(**overrides: Any) -> LTMLE:
    settings: dict[str, Any] = dict(LONGITUDINAL_SETTINGS)
    settings.update(overrides)
    return LTMLE({"always": 1, "never": 0}, **settings)


class TestTheLongitudinalSplitReadsNoTreatment:
    """The first-node arms no longer choose the split, and are checked on it instead."""

    @staticmethod
    def assignment(frame: pd.DataFrame) -> np.ndarray:
        result = sequential().fit(frame, **LONGITUDINAL_COLUMNS)
        return np.asarray(result.folds.assignment)

    @staticmethod
    def permuted(frame: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        moved = frame.copy()
        # The first node only. Every unit is at risk there, so its column has no missing
        # value to preserve, and it is the node the retired policy balanced on.
        moved["A1"] = frame["A1"].to_numpy()[order]
        return moved

    def test_the_folds_do_not_move_when_the_first_node_arms_do(self) -> None:
        frame = longitudinal_frame()
        np.testing.assert_array_equal(self.assignment(frame), self.assignment(self.permuted(frame)))

    def test_a_split_stratified_on_the_first_node_moves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control, and the implementation this replaced.

        The sequential fit used to balance its outer split on the first treatment node,
        which is the variable the first mechanism is then fitted to.
        """
        engine = sys.modules["cleverly.longitudinal.estimator"]

        def stratified(n: int, n_folds: int, *, cluster: Any = None, seed: int) -> Any:
            del cluster
            from cleverly.learners import make_folds

            return make_folds(n, n_folds, stratify=stratified.arm, random_state=seed)

        stratified.arm = None  # type: ignore[attr-defined]

        frame = longitudinal_frame()
        stratified.arm = np.nan_to_num(frame["A1"].to_numpy(dtype=float))  # type: ignore[attr-defined]
        monkeypatch.setattr(engine, "random_partition", stratified)
        first = self.assignment(frame)
        moved = self.permuted(frame)
        stratified.arm = np.nan_to_num(moved["A1"].to_numpy(dtype=float))  # type: ignore[attr-defined]
        assert not np.array_equal(first, self.assignment(moved)), (
            "the stratified split did not move, so the witness above is not evidence "
            "about what the split reads"
        )


class TestTheCommittedFirstNodeStratifiedSeam:
    """The retired longitudinal policy, committed so an attribution can be redrawn.

    ``docs/roadmap.md`` RM18 attributes one red cell to the fold policy alone, and the
    diagnostic behind it needed the retired arm. The arm is
    ``tests.studies.ltmle_crossfit_properties.FirstNodeStratifiedLTMLE`` rather than an edit
    to the shipped estimator, so the run is reproducible from this repository.
    """

    @staticmethod
    def _assignment(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(estimator.fit(frame, **LONGITUDINAL_COLUMNS).folds.assignment)

    def test_the_seam_draws_a_split_the_first_node_moves_and_the_shipped_one_does_not(
        self,
    ) -> None:
        from tests.studies.ltmle_crossfit_properties import FirstNodeStratifiedLTMLE

        frame = longitudinal_frame()
        rng = np.random.default_rng(29)
        moved = frame.copy()
        moved["A1"] = frame["A1"].to_numpy()[rng.permutation(len(frame))]

        settings = dict(LONGITUDINAL_SETTINGS)
        seam = FirstNodeStratifiedLTMLE({"always": 1, "never": 0}, **settings)
        retired = self._assignment(seam, frame)
        assert not np.array_equal(
            retired,
            self._assignment(
                FirstNodeStratifiedLTMLE({"always": 1, "never": 0}, **settings), moved
            ),
        ), "the committed seam does not read the first node, so it is not the retired arm"
        assert not np.array_equal(retired, self._assignment(sequential(), frame)), (
            "the seam and the shipped estimator drew the same split, so the 2x2 would "
            "report one policy twice"
        )
        np.testing.assert_array_equal(
            self._assignment(sequential(), frame), self._assignment(sequential(), moved)
        )


class TestTheLongitudinalRefusals:
    """The two cross-fitted longitudinal designs with no shipped result."""

    def test_a_continuous_outcome_needs_a_declared_support(self) -> None:
        frame = longitudinal_frame()
        frame["Y"] = frame["Y"].to_numpy(dtype=float) + np.linspace(0.0, 3.0, len(frame))
        with pytest.raises(LongitudinalError, match="needs a declared q_bounds"):
            sequential().fit(frame, **LONGITUDINAL_COLUMNS)

    def test_the_same_outcome_fits_in_sample(self) -> None:
        """The control for the refusal above: one fold has no held-out row to leak to."""
        frame = longitudinal_frame()
        frame["Y"] = frame["Y"].to_numpy(dtype=float) + np.linspace(0.0, 3.0, len(frame))
        result = sequential(n_folds=1, outcome_learner=LinearRegression()).fit(
            frame, **LONGITUDINAL_COLUMNS
        )
        assert result.folds.is_single

    def test_clusters_are_refused_under_cross_fitting(self) -> None:
        frame = longitudinal_frame(n=400, cluster_size=5)
        with pytest.raises(LongitudinalError, match="has no clustered result"):
            sequential().fit(frame, id="id", **LONGITUDINAL_COLUMNS)

    def test_the_same_clusters_fit_in_sample(self) -> None:
        frame = longitudinal_frame(n=400, cluster_size=5)
        result = sequential(n_folds=1).fit(frame, id="id", **LONGITUDINAL_COLUMNS)
        assert result.folds.is_single


# ------------------------------------------------------- the bootstrap is conditional


class TestTheBootstrapDropsWhatThePreflightRefuses:
    """S4. A replicate the preflight refuses is dropped, counted, and fits nothing.

    A resample of a rare arm can hold too few of it for any split to carry, and the
    preflight refuses that replicate exactly as it refuses an ordinary fit. The replicate
    is then dropped rather than redrawn, because redrawing until one succeeded would
    select the resamples by the arm count. So the reported interval is conditional on the
    replicates that ran, and ``n_failed`` is how far that is from what was asked for.
    """

    @staticmethod
    def rare_arm_frame(n: int = 120, treated: int = 6) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "Y": rng.uniform(0.2, 0.8, size=n),
                "A": np.zeros(n),
                "W1": rng.normal(size=n),
                "W2": rng.normal(size=n),
            }
        )
        frame.loc[: treated - 1, "A"] = 1.0
        return frame

    def test_some_replicates_are_dropped_and_none_of_them_fitted_a_learner(self) -> None:
        frame = self.rare_arm_frame()
        settings: dict[str, Any] = {
            "outcome_learner": CountingLinear(),
            "treatment_learner": CountingLogistic(max_iter=1000),
            "q_bounds": (0.0, 1.0),
            "n_folds": 4,
            "learner_folds": 3,
            "random_state": 0,
            "simultaneous": False,
            "estimands": ["ate"],
        }
        reset_counter()
        TMLE(**settings).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        per_replicate = len(_FIT_COUNTER)
        assert per_replicate > 0, "the ordinary fit ran no learner, so the count means nothing"

        reset_counter()
        with pytest.warns(UserWarning, match="bootstrap replicates failed and were dropped"):
            result = (
                TMLE(n_bootstrap=40, **settings)
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        bootstrap = result.bootstrap
        assert bootstrap is not None
        assert 0 < bootstrap.n_failed < bootstrap.n_requested, (
            f"{bootstrap.n_failed} of {bootstrap.n_requested} replicates were dropped; "
            "this fixture needs a rare enough arm for some resamples to lose it and "
            "common enough for others to keep it"
        )
        ran = bootstrap.n_requested - bootstrap.n_failed
        assert len(_FIT_COUNTER) == per_replicate * (ran + 1), (
            f"{len(_FIT_COUNTER)} learner fits over {ran} surviving replicates plus the "
            f"original fit, against {per_replicate} per fit. A dropped replicate that "
            "fitted a learner would show up here"
        )


# ------------------------------------------------------------------- the seed range


def test_every_registered_fold_seed_stays_inside_the_generators_range() -> None:
    """The DR-TMLE studies hand ``random_partition`` a seed one above a drawn one.

    ``SeedSequence.generate_state`` returns a full 32-bit word, and
    :func:`~cleverly.learners.random_partition` refuses anything above ``2**32 - 1``, so
    ``seed + 1`` is one draw away from a study that cannot run. The margin is small: the
    largest registered DR-TMLE fold seed leaves under fifteen thousand of it. Gating it
    here turns a latent overflow into a failing test rather than a failed regeneration.
    """
    from tests.studies import canonical_drtmle, canonical_multi_arm_drtmle
    from tests.studies.evidence.seeds import replicate_seed

    margins: list[tuple[str, int]] = []
    for module in (canonical_drtmle, canonical_multi_arm_drtmle):
        record = module.STUDY
        for scenario in record.scenarios:
            for replicate in range(record.replicates):
                seed = replicate_seed(record, scenario, replicate) + 1
                margins.append((f"{record.slug}/{scenario}/{replicate}", _MAX_SEED - seed))
    worst_label, worst = min(margins, key=lambda pair: pair[1])
    assert worst >= 0, (
        f"{worst_label} would hand random_partition a seed {-worst} above its limit; "
        "reduce the stream modulo 2**31 - 1, as CrossFitPlan.seeds does"
    )


# --------------------------------- W9: what a training complement owes beyond the arms


def seed_stranding_rows(n: int, n_folds: int, rows: np.ndarray, limit: int = 20_000) -> int:
    """A seed whose draw puts every row of ``rows`` in one fold.

    The generalisation of :func:`seed_stranding_an_arm`, which asks the same question of
    the treated rows. Searched over the generator the fit uses, so the property holds of
    a split the package itself would draw.
    """
    for seed in range(limit):
        folds = random_partition(n, n_folds, seed=seed)
        if np.unique(np.asarray(folds.assignment)[rows]).size == 1:
            return seed
    raise AssertionError("no seed in the search range stranded those rows")


def respondents_in_one_fold(n: int = 60, n_folds: int = 6) -> tuple[pd.DataFrame, int]:
    """A continuous-outcome sample whose four respondents share one fold.

    The fit that reads it is an in-sample collaborative search, whose selection folds are
    drawn by the same generator from the same seed. That fit is off the arm-indexed stacked
    surface and that contract's own preflight, so the general complement check is what
    has to catch the respondentless training complement.
    """
    rng = np.random.default_rng(0)
    respondents = np.array([0, 1, 2, 3])
    delta = np.zeros(n)
    delta[respondents] = 1.0
    frame = pd.DataFrame(
        {
            "Y": np.where(delta == 1.0, rng.uniform(0.2, 0.8, size=n), np.nan),
            "A": rng.binomial(1, 0.5, size=n).astype(float),
            "W1": rng.normal(size=n),
            "W2": rng.normal(size=n),
            "D": delta,
        }
    )
    frame.loc[respondents, "A"] = np.array([0.0, 1.0, 0.0, 1.0])
    return frame, seed_stranding_rows(n, n_folds, respondents)


class TestARespondentlessComplementIsRefusedOnEveryScale:
    """W9. The outcome regression trains on the respondents, whatever scale they are on.

    The binary case was covered through the outcome classes, which a Gaussian outcome has
    none of, so a continuous outcome with ``delta=`` reached a learner and failed inside
    one. The complement check asks for a respondent on every scale.

    The witness is an in-sample greedy C-TMLE with ``delta=``, whose selection folds ask
    that check before any learner. A cross-fitted regime fit carried this witness until
    the F21 refusal closed that composition. The arm-indexed and natural-course contracts
    run their own preflights, so this admitted fit is the one that reaches the general
    response branch.
    """

    @staticmethod
    def selection_fit(**overrides: Any) -> CTMLE:
        settings: dict[str, Any] = {
            "strategy": "greedy",
            "cross_fit": False,
            "selection_folds": 6,
            "outcome_learner": LinearRegression(),
            "treatment_learner": LogisticRegression(max_iter=1000),
            "q_bounds": (0.0, 1.0),
            "estimands": ["ate"],
            "simultaneous": False,
        }
        settings.update(overrides)
        return CTMLE(**settings)

    @staticmethod
    def columns() -> dict[str, Any]:
        return {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "delta": "D"}

    def test_the_fit_is_refused_with_no_learner_fitted(self) -> None:
        frame, seed = respondents_in_one_fold()
        reset_counter()
        estimator = self.selection_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
            random_state=seed,
        )
        with pytest.raises(
            DataError,
            match=(
                r"C-TMLE selection cannot fit its nuisances because .*training complement "
                "contains no row with an observed outcome"
            ),
        ):
            estimator.fit(frame, **self.columns())
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"

    def test_the_refusal_names_no_redraw(self) -> None:
        frame, seed = respondents_in_one_fold()
        with pytest.raises(DataError) as raised:
            self.selection_fit(random_state=seed).fit(frame, **self.columns())
        message = str(raised.value)
        for forbidden in ("Increase n_folds", "different random_state", "reduce n_folds"):
            assert forbidden not in message, f"the refusal offers {forbidden!r}"

    def test_disabling_the_preflight_lets_the_same_draw_reach_a_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. Without the check the respondentless complement reaches a fit."""
        monkeypatch.setattr(TMLE, "_check_training_support", lambda *a, **k: None)
        frame, seed = respondents_in_one_fold()
        reset_counter()
        estimator = self.selection_fit(
            outcome_learner=CountingLinear(),
            treatment_learner=CountingLogistic(max_iter=1000),
            random_state=seed,
        )
        with pytest.raises(Exception):  # noqa: B017 - any downstream failure will do
            estimator.fit(frame, **self.columns())
        assert _FIT_COUNTER, "the disabled preflight still fitted no learner"


class TestASingleEventOutcomeStatesASampleMinimum:
    """W10. One unit holding an outcome class is a fact about the sample, not the draw.

    Every partition leaves the complement of that unit's own fold without the class, so
    the refusal states the minimum rather than describing one drawn split, exactly as the
    arm minimum beside it does.
    """

    @staticmethod
    def single_event_frame(n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "Y": np.zeros(n),
                "A": rng.binomial(1, 0.5, size=n).astype(float),
                "W1": rng.normal(size=n),
                "W2": rng.normal(size=n),
            }
        )
        frame.loc[0, "Y"] = 1.0
        return frame

    @staticmethod
    def binary_fit(**overrides: Any) -> TMLE:
        settings: dict[str, Any] = {
            "outcome_learner": LogisticRegression(max_iter=1000),
            "treatment_learner": LogisticRegression(max_iter=1000),
            "n_folds": 5,
            "learner_folds": 3,
            "random_state": 0,
            "simultaneous": False,
            "estimands": ["ate"],
        }
        settings.update(overrides)
        return TMLE(**settings)

    def test_the_refusal_states_the_minimum_and_names_no_split(self) -> None:
        reset_counter()
        estimator = self.binary_fit(
            outcome_learner=CountingLogistic(max_iter=1000),
            treatment_learner=CountingLogistic(max_iter=1000),
        )
        with pytest.raises(DataError) as raised:
            estimator.fit(
                self.single_event_frame(), outcome="Y", treatment="A", covariates=["W1", "W2"]
            )
        message = str(raised.value)
        assert "needs each outcome class in at least two independent units" in message
        assert "outcome 1 appears in 1 row(s)" in message
        assert "collect more observations with outcome 1" in message
        for forbidden in ("Increase n_folds", "different random_state", "reduce n_folds"):
            assert forbidden not in message, f"the refusal offers {forbidden!r}"
        assert _FIT_COUNTER == [], f"{len(_FIT_COUNTER)} learner fit(s) ran before the refusal"

    def test_two_events_in_one_cluster_count_as_one_unit(self) -> None:
        """The unit is the cluster, because a grouped draw moves whole clusters."""
        frame = self.single_event_frame(n=60)
        frame["cid"] = np.repeat(np.arange(12), 5)
        frame.loc[1, "Y"] = 1.0  # the same cluster as row zero
        with pytest.raises(DataError, match="outcome 1 appears in 1 cluster"):
            self.binary_fit(n_folds=4).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cid"
            )

    def test_two_events_in_one_fold_get_the_per_draw_message_instead(self) -> None:
        """The control. Two units clear the minimum, and the draw is then what refuses.

        The same sample with one more event is refused for a different reason and in a
        different sentence, so the minimum above is what the first message is about.
        """
        frame = self.single_event_frame()
        frame.loc[30, "Y"] = 1.0
        seed = seed_stranding_rows(len(frame), 5, np.array([0, 30]))
        with pytest.raises(DataError) as raised:
            self.binary_fit(random_state=seed).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"]
            )
        message = str(raised.value)
        assert "needs each outcome class in at least two independent units" not in message
        assert "training complement contains no observed outcome 1" in message


# ------------------------------- I2: no post-draw refusal names a repartition, anywhere


class TestTheReducedRegressionRefusalNamesNoRedraw:
    """The third fold loop. Its message used to end "reduce n_folds".

    The reduced regressions are fitted inside the same drawn split as the primary
    nuisances, so this is a post-draw refusal and may name no redraw either. It is the
    backstop under the complement preflight, which refuses the same draw first: the
    preflight is disabled here so that the message below is the one raised, and a
    treatment learner that tolerates a single class keeps the primary nuisances from
    failing before it.
    """

    def test_it_offers_the_pooled_construction_and_no_new_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.dummy import DummyClassifier

        monkeypatch.setattr(TMLE, "_preflight_training_support", lambda *a, **k: None)
        n, n_folds = 60, 6
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "Y": rng.uniform(0.2, 0.8, size=n),
                "A": np.zeros(n),
                "W1": rng.normal(size=n),
                "W2": rng.normal(size=n),
            }
        )
        treated = np.array([0, 1, 2])
        frame.loc[treated, "A"] = 1.0
        estimator = DRTMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=DummyClassifier(strategy="prior"),
            q_bounds=(0.0, 1.0),
            n_folds=n_folds,
            learner_folds=3,
            random_state=seed_stranding_rows(n, n_folds, treated),
            simultaneous=False,
            estimands=["ate"],
            reduced_crossfit="nested",
        )
        with pytest.raises(ValueError) as raised:
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        message = str(raised.value)
        assert "no trainable rows for a reduced regression" in message
        assert "reduced_crossfit='pooled'" in message
        for forbidden in ("Increase n_folds", "different random_state", "reduce n_folds"):
            assert forbidden not in message, f"the refusal offers {forbidden!r}"


# ------------------------------------------- M7: the balancing policies reach no split


def test_no_fit_reaches_the_strata_decision_under_a_balancing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal precedes every split, so ``_fold_strata`` sees ``"none"`` only.

    This is what lets the two ``"treatment+outcome"`` refusals that stood inside that
    method be deleted rather than kept for a caller who cannot reach them. The policy is
    smuggled onto a constructed estimator as well as declared, because a restored result
    and a copied estimator arrive that way. An in-sample fit accepts the declaration and
    reaches no split: the fit trains on every row.
    """
    seen: list[str] = []
    original = TMLE._fold_strata

    def recording(self: TMLE, data: Any) -> Any:
        seen.append(self.stratify_folds)
        return original(self, data)

    monkeypatch.setattr(TMLE, "_fold_strata", recording)
    frame, _ = make_nonlinear_bounded(n=120, seed=1)
    columns: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": BOUNDED_COVARIATES,
    }

    for policy in ("treatment", "treatment+outcome"):
        named = re.escape(f"stratify_folds={policy!r}")
        with pytest.raises(ValueError, match=named):
            bounded_fit(stratify_folds=policy, n_folds=3)
        for cross_fit in (True, False):
            smuggled = bounded_fit(cross_fit=cross_fit, n_folds=3)
            smuggled.stratify_folds = policy
            if cross_fit:
                with pytest.raises(ValueError, match=named):
                    smuggled.fit(frame, **columns)
            else:
                smuggled.fit(frame, **columns)
        # Selector-based collaborative fits draw selection folds at every setting, so
        # they are refused at both, before the declaration reaches a partition.
        for cross_fit in (True, False):
            with pytest.raises(ValueError, match=named):
                collaborative(stratify_folds=policy, cross_fit=cross_fit)
    assert seen == [], f"a fit reached _fold_strata under {sorted(set(seen))}"

    # The control: the recorder is installed, and an ordinary cross-fitted fit reaches it.
    bounded_fit(n_folds=3).fit(frame, **columns)
    assert seen == ["none"]


# --------------------------------- the study seam: every layer one policy has to reach


#: The two modules that draw a fold partition, and the fold count each of their call sites
#: declares.  A selector-based C-TMLE fit draws three splits and a DR-TMLE fit draws one,
#: and every one of them reads :meth:`~cleverly.estimators.TMLE._fold_strata`, so the count
#: is what names the layer in a recorded draw.  Imported through :mod:`importlib` because
#: ``cleverly.estimators`` binds the name ``tmle`` to a function, which shadows the
#: submodule of that name.
_CTMLE_MODULE = importlib.import_module("cleverly.estimators.ctmle")
_TMLE_MODULE = importlib.import_module("cleverly.estimators.tmle")

MULTI_ARM_COVARIATES = ["W1", "W2", "W3"]
OUTER_FOLDS, SELECTION_FOLDS, NESTED_FOLDS = 5, 3, 2


def multi_arm_frame(n: int = 200, seed: int = 5) -> pd.DataFrame:
    """A draw of the law both study rows sample their fold-policy arms from."""
    frame, _ = multi_arm_common.law().sample(n, seed=seed, backend="pandas")
    return frame


def selector_settings() -> dict[str, Any]:
    """The selector study's own configuration, shrunk to what a unit test can afford.

    The fold counts are the study's, because they are the subject: the seam has to reach
    each of the three layers they declare. The learners are plain regressions rather than
    ``FAST_KWARGS``, whose super learner would fit a library inside every one of those
    folds and buy this test nothing.
    """
    return {
        "outcome_learner": LogisticRegression(max_iter=500),
        "treatment_learner": LogisticRegression(max_iter=500),
        "cross_fit": True,
        "n_folds": OUTER_FOLDS,
        "selection_folds": SELECTION_FOLDS,
        "selection_inner_folds": NESTED_FOLDS,
        "penalty": False,
        "strategy": "discrete",
        "candidates": ((), ("W1",)),
        "estimands": "ate",
        "ctmle_estimand": "ate",
        "reference": multi_arm_common.REFERENCE,
        "simultaneous": False,
        "g_bounds": multi_arm_common.G_BOUNDS,
        "max_iter": 50,
        "tol": 1e-8,
        "random_state": 0,
    }


def drtmle_settings() -> dict[str, Any]:
    """The DR-TMLE study's own configuration, under the same shrinking rule."""
    return {
        "outcome_learner": LogisticRegression(max_iter=500),
        "treatment_learner": LogisticRegression(max_iter=500),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(C=1e6, max_iter=500),
        "cross_fit": True,
        "n_folds": OUTER_FOLDS,
        "estimands": "ate",
        "reference": multi_arm_common.REFERENCE,
        "simultaneous": False,
        "g_bounds": multi_arm_common.G_BOUNDS,
        "max_outer": 20,
        "max_iter": 50,
        "tol": 1e-8,
        "random_state": 0,
        "guard": ("Q", "g"),
        "reduction": "univariate",
        "reduced_crossfit": "pooled",
        "update_order": "drtmle",
    }


def record_draws(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
    """Record ``(requested folds, the draw read strata)`` for every partition a fit draws.

    Both modules are patched, because the outer split is drawn in one and the selection
    and nested splits in the other. Recording the *call* rather than the realized labels
    is what separates "the policy reached this layer" from "this layer's labels moved": a
    nested draw sits inside a selection fold, so its labels move whenever the selection
    split above it does, whatever policy the nested draw itself read.
    """
    drawn: list[tuple[int, bool]] = []

    def recorder(original: Any) -> Any:
        def recording(
            n: int,
            n_folds: int,
            *,
            stratify: Any = None,
            cluster: Any = None,
            random_state: Any = None,
        ) -> Any:
            drawn.append((n_folds, stratify is not None))
            return original(
                n, n_folds, stratify=stratify, cluster=cluster, random_state=random_state
            )

        return recording

    for module in (_CTMLE_MODULE, _TMLE_MODULE):
        monkeypatch.setattr(module, "make_folds", recorder(module.make_folds))
    return drawn


def selector_splits(estimator: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The outer and the selection partition one selector-based C-TMLE fit realized."""
    result = estimator.fit(
        frame, outcome="Y", treatment="A", covariates=MULTI_ARM_COVARIATES
    ).single()
    return (
        np.asarray(result.nuisance.folds.assignment),
        np.asarray(result.extra["ctmle"].folds.assignment),
    )


def outer_split(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    """The one partition a DR-TMLE fit realized."""
    result = estimator.fit(
        frame, outcome="Y", treatment="A", covariates=MULTI_ARM_COVARIATES
    ).single()
    return np.asarray(result.nuisance.folds.assignment)


class TestTheFoldPolicySeamReachesEverySplitLayer:
    """W6. ``FoldPolicyMixin`` varies every split a fit draws, and varies nothing else.

    The seam under test is :class:`tests.studies.multi_arm_properties.FoldPolicyMixin`,
    which the two multi-arm fold-policy diagnostics are fitted through. Its claim has two
    halves: the policy reaches every split the method draws, and the policy is the only
    thing that differs between the two arms of the paired comparison. Neither half is
    visible in a committed row. A diagnostic whose stratified arm quietly varied one layer
    of three would publish a coverage difference under the name of a change it never made.
    """

    @staticmethod
    def _assert_treatment_strata(estimator: Any, frame: pd.DataFrame) -> None:
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=MULTI_ARM_COVARIATES
        )
        np.testing.assert_array_equal(estimator._fold_strata(data), data.treatment)

    def test_the_stratified_policy_reads_the_treatment_vector(self) -> None:
        self._assert_treatment_strata(
            multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
            multi_arm_frame(),
        )

    def test_an_outcome_vector_cannot_pass_as_treatment_strata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mask mutation control for the policy's defining input."""
        monkeypatch.setattr(
            multi_arm_properties.FoldPolicyMixin,
            "_fold_strata",
            lambda self, data: np.asarray(data.outcome, dtype=float),
        )
        with pytest.raises(AssertionError):
            self._assert_treatment_strata(
                multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
                multi_arm_frame(),
            )

    @pytest.mark.parametrize(
        ("study_module", "estimator_type"),
        [
            (multi_arm_ctmle_selector_properties, multi_arm_properties.FoldPolicyCTMLE),
            (multi_arm_drtmle_properties, multi_arm_properties.FoldPolicyDRTMLE),
        ],
    )
    def test_each_study_pairs_the_policies_and_routes_them_through_the_seam(
        self, study_module: Any, estimator_type: type[Any]
    ) -> None:
        cells = study_module.cells()
        diagnostic = [
            cell for cell in cells if cell.property == multi_arm_properties.FOLD_POLICY_FAMILY
        ]
        n_500 = next(
            cell
            for cell in cells
            if cell.property == "root_n_and_efficiency" and cell.cell == "n_500"
        )
        assert [cell.cell for cell in diagnostic] == list(multi_arm_properties.FOLD_POLICIES)
        assert len({cell.seed for cell in diagnostic}) == 1
        assert diagnostic[0].seed != n_500.seed
        for cell in diagnostic:
            estimator = study_module._estimator(cell)()
            assert isinstance(estimator, estimator_type)
            assert estimator._policy == cell.cell

    def test_the_policy_reaches_all_three_selector_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = record_draws(monkeypatch)
        selector_splits(
            multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
            multi_arm_frame(),
        )
        assert {layer for layer, _ in drawn} == {OUTER_FOLDS, SELECTION_FOLDS, NESTED_FOLDS}, (
            f"the fit drew {sorted({layer for layer, _ in drawn})} rather than the outer, "
            f"selection and nested layers this method declares"
        )
        unbalanced = sorted({layer for layer, stratified in drawn if not stratified})
        assert unbalanced == [], (
            f"the {unbalanced} layer(s) drew an unstratified split on an arm named "
            f"treatment_stratified, so the published difference would name a change the "
            f"fit made on a strict sub-part of its splits"
        )

    def test_the_reference_arm_reproduces_the_shipped_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half. An arm named ``unstratified`` has to be the shipped estimator."""
        drawn = record_draws(monkeypatch)
        frame = multi_arm_frame()
        shipped = selector_splits(CTMLE(**selector_settings()), frame)
        reference = selector_splits(
            multi_arm_properties.FoldPolicyCTMLE("unstratified", **selector_settings()), frame
        )
        np.testing.assert_array_equal(shipped[0], reference[0])
        np.testing.assert_array_equal(shipped[1], reference[1])
        assert {stratified for _, stratified in drawn} == {False}

    def test_the_stratified_arm_moves_the_splits_the_reference_arm_leaves_alone(self) -> None:
        frame = multi_arm_frame()
        shipped = selector_splits(CTMLE(**selector_settings()), frame)
        stratified = selector_splits(
            multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
            frame,
        )
        assert not np.array_equal(shipped[0], stratified[0]), "the outer split did not move"
        assert not np.array_equal(shipped[1], stratified[1]), "the selection split did not move"

    def test_an_always_empty_strata_vector_stops_the_selector_arm_differing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate-mutation control, which is what makes the witness nonzero.

        A stratified arm differs from the shipped fit only because ``_fold_strata`` hands
        ``make_folds`` a vector. Emptying it puts the reference arm's behaviour into the
        stratified arm, and the two assertions above then have nothing to find. Without
        this control, a witness comparing two fits that differ for some other reason would
        read exactly like one that pins the policy.
        """
        monkeypatch.setattr(
            multi_arm_properties.FoldPolicyMixin, "_fold_strata", lambda self, data: None
        )
        frame = multi_arm_frame()
        shipped = selector_splits(CTMLE(**selector_settings()), frame)
        mutated = selector_splits(
            multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
            frame,
        )
        np.testing.assert_array_equal(shipped[0], mutated[0])
        np.testing.assert_array_equal(shipped[1], mutated[1])

    def test_the_seam_changes_no_fold_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The policy is the only thing that may differ, so no split size may.

        :class:`tests.studies.bounded_cv_laws.FoldPolicyTMLE` pins ten folds inside its
        own seam. A seam that pinned a count here would hand the diagnostic a second
        difference between its arms, and the paired coverage difference could not say
        which of the two it measured.
        """
        frame = multi_arm_frame()
        counts: dict[str, list[Any]] = {}
        for arm, estimator in (
            ("shipped", CTMLE(**selector_settings())),
            (
                "stratified",
                multi_arm_properties.FoldPolicyCTMLE("treatment_stratified", **selector_settings()),
            ),
        ):
            drawn = record_draws(monkeypatch)
            outer, selection = selector_splits(estimator, frame)
            counts[arm] = [
                sorted(layer for layer, _ in drawn),
                len(np.unique(outer)),
                len(np.unique(selection)),
            ]
        assert counts["shipped"] == counts["stratified"]
        assert counts["shipped"][1:] == [OUTER_FOLDS, SELECTION_FOLDS]

    def test_the_drtmle_seam_moves_its_one_layer_and_its_reference_arm_does_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = record_draws(monkeypatch)
        frame = multi_arm_frame()
        shipped = outer_split(DRTMLE(**drtmle_settings()), frame)
        reference = outer_split(
            multi_arm_properties.FoldPolicyDRTMLE("unstratified", **drtmle_settings()), frame
        )
        stratified = outer_split(
            multi_arm_properties.FoldPolicyDRTMLE("treatment_stratified", **drtmle_settings()),
            frame,
        )
        np.testing.assert_array_equal(shipped, reference)
        assert not np.array_equal(shipped, stratified), "the one DR-TMLE split did not move"
        assert [layer for layer, _ in drawn] == [OUTER_FOLDS] * 3, (
            "this method draws one split per fit, and the record says otherwise"
        )
        assert [balanced for _, balanced in drawn] == [False, False, True]
        assert len(np.unique(stratified)) == OUTER_FOLDS

    def test_an_always_empty_strata_vector_stops_the_drtmle_arm_differing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same control, for the method that draws one split."""
        monkeypatch.setattr(
            multi_arm_properties.FoldPolicyMixin, "_fold_strata", lambda self, data: None
        )
        frame = multi_arm_frame()
        shipped = outer_split(DRTMLE(**drtmle_settings()), frame)
        mutated = outer_split(
            multi_arm_properties.FoldPolicyDRTMLE("treatment_stratified", **drtmle_settings()),
            frame,
        )
        np.testing.assert_array_equal(shipped, mutated)

    def test_the_recorded_plan_still_names_no_strata(self) -> None:
        """The caveat ``FoldPolicyMixin`` records, pinned so that a reader is not surprised.

        :meth:`~cleverly.estimators.TMLE.crossfit_plan` reads ``self.stratify_folds``
        rather than the method this seam replaces, so the plan on the stratified arm's
        result describes a split that arm did not draw. Nothing on the property path reads
        it. The assertion is here so the mismatch is a recorded fact rather than a
        discovery.
        """
        result = (
            multi_arm_properties.FoldPolicyDRTMLE("treatment_stratified", **drtmle_settings())
            .fit(multi_arm_frame(), outcome="Y", treatment="A", covariates=MULTI_ARM_COVARIATES)
            .single()
        )
        assert result.config.crossfit.stratify_by == ()

    @pytest.mark.parametrize("seam", ["FoldPolicyCTMLE", "FoldPolicyDRTMLE"])
    def test_an_undeclared_policy_is_refused_before_the_estimator_is_built(self, seam: str) -> None:
        with pytest.raises(ValueError, match="policy must be one of"):
            getattr(multi_arm_properties, seam)("treatment_outcome_stratified")


# --------------------------------------------- a copied or modified estimator is refused


@pytest.mark.parametrize("policy", ["treatment", "treatment+outcome"])
def test_a_reconfigured_refit_names_the_requested_strata(policy: str) -> None:
    """A dose split reads no treatment, and its refit still refuses the requested policy.

    ``__init__`` refuses both policies, so each result here is a live fit whose estimator
    was given the policy after construction. The discrete fit draws strata under the policy
    and the dose fit draws none, and both refits refuse with the same sentence.
    """
    frame = dose_frame()
    dose = reconfigured(
        fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame)),
        stratify_folds=policy,
    )
    discrete = reconfigured(cross_fitted(), stratify_folds=policy)
    assert dose.estimator._fold_strata(dose.data) is None
    assert discrete.estimator._fold_strata(discrete.data) is not None

    for result in (dose, discrete):
        with pytest.raises(CapabilityError) as raised:
            result.estimator.refit(result.data)
        message = str(raised.value)
        reads = (
            "the treatment and the outcome" if policy == "treatment+outcome" else "the treatment"
        )
        assert f"requests stratification of the outer folds on {reads}" in message
        assert "A split drawn from those strata would make" in message
        assert "balances the outer folds" not in message
        assert "Set stratify_folds='none'" in message
        assert "cross_fit=False" in message


def test_a_live_fit_on_an_undeclared_scale_is_refused() -> None:
    """A cross-fitted dose fit needs a declared scale, and the declared one keeps its interval."""
    frame = dose_frame()
    result = fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame))
    assert result.inference_status == "influence_curve"
    with pytest.raises(CapabilityError, match="needs a declared q_bounds"):
        fit_shift(frame, cross_fit=True, n_folds=2)


class TestVariableImportanceRefusesBeforeItsFirstFit:
    """``variable_importance`` raises the fit-time refusals before any learner."""

    def test_a_modified_fold_policy_meets_the_fold_policy_refusal(self) -> None:
        """The refusal names the remedy, and it arrives before any learner."""
        estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, **never_fit_learners()))
        estimator.stratify_folds = "treatment"
        with pytest.raises(CapabilityError) as raised:
            variable_importance(
                discrete_law.frame(),
                outcome="Y",
                candidates=["A"],
                covariates=["W"],
                estimator=estimator,
            )
        assert str(raised.value) == estimator._fold_policy_refusal()
        assert "Set stratify_folds='none'" in str(raised.value)
        assert NeverFit.calls == 0

    @staticmethod
    def run_undeclared_scale() -> tuple[TMLE, CausalData]:
        frame, _ = make_linear_ate(n=400, seed=2)
        estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, **never_fit_learners()))
        prepared = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W1", "W2"))
        variable_importance(
            frame, outcome="Y", candidates=["A"], covariates=["W1", "W2"], estimator=estimator
        )
        return estimator, prepared

    def test_an_undeclared_scale_meets_the_scale_refusal(self) -> None:
        """The refusal names the remedy, and it arrives before any learner."""
        with pytest.raises(CapabilityError) as raised:
            self.run_undeclared_scale()
        assert "Declare the known outcome support" in str(raised.value)
        assert NeverFit.calls == 0

    def test_without_the_scale_check_a_learner_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control: with the refusal removed, the run reaches its first learner."""
        monkeypatch.setattr(TMLE, "_refuse_unbounded_cross_fitted_scale", lambda self, data: None)
        with pytest.raises(AssertionError, match="a refusal or preflight must run"):
            self.run_undeclared_scale()
        assert NeverFit.calls > 0
