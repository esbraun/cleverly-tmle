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

import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_binary_outcome, make_longitudinal, make_nonlinear_bounded
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError, LongitudinalError
from cleverly.learners import SuperLearner, random_partition
from cleverly.learners.crossfit import _MAX_SEED
from cleverly.longitudinal import LTMLE
from cleverly.utils.bounds import OutcomeScaler

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
    """W5. A Super Learner's inner split is a function of its own training rows."""

    @staticmethod
    def inner_splits(frame: pd.DataFrame) -> list[tuple[int, ...]]:
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
        estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        return list(_RECORDED_INNER_SPLITS)

    @staticmethod
    def with_one_outcome_flipped(frame: pd.DataFrame) -> pd.DataFrame:
        moved = frame.copy()
        moved.loc[0, "Y"] = 1.0 - float(frame.loc[0, "Y"])
        return moved

    def test_flipping_a_held_out_outcome_moves_no_inner_split(self) -> None:
        frame, _ = make_binary_outcome(n=200, seed=5)
        assert self.inner_splits(frame) == self.inner_splits(self.with_one_outcome_flipped(frame))

    def test_an_inner_split_that_reads_every_row_does_move(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. A split stratified on a vector that includes held-out rows moves.

        The mutation is the smallest implementation that reads outside the training rows:
        the inner split is balanced on the first ``n`` entries of the whole sample's
        outcome rather than on the target the learner was handed.
        """
        import cleverly.learners.super_learner as learners

        frame, _ = make_binary_outcome(n=200, seed=5)
        leaked = {"outcome": frame["Y"].to_numpy(dtype=float)}
        original = learners.make_folds

        def leaking(n: int, n_folds: int = 10, **kwargs: Any) -> Any:
            if kwargs.get("stratify") is not None:
                kwargs["stratify"] = leaked["outcome"][:n]
            return original(n, n_folds, **kwargs)

        monkeypatch.setattr(learners, "make_folds", leaking)
        first = self.inner_splits(frame)
        moved = self.with_one_outcome_flipped(frame)
        leaked["outcome"] = moved["Y"].to_numpy(dtype=float)
        assert first != self.inner_splits(moved), (
            "the leaking inner split did not move, so the witness above is not evidence "
            "about what the inner split reads"
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


def sequential(**overrides: Any) -> LTMLE:
    settings: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "n_folds": 3,
        "learner_folds": 3,
        "random_state": 0,
    }
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
