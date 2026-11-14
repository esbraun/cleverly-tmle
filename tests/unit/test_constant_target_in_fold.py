"""A cross-fitted point-treatment fold whose binary outcome takes one value predicts it.

The rule is the point-treatment twin of the longitudinal one
(``docs/technical-reference/longitudinal-tmle.md``, "A fold with one target value"): the
maximum-likelihood probability of the one value is 1, so the fold's regression is that value
(:class:`~cleverly.estimators._nuisance.ConstantFit`) and no learner runs.  It reaches the
per-arm outcome regression of ``TMLE(outcome_by_arm=True)`` and the leave-two-out outcome
regressions of ``DRTMLE(reduced_crossfit="foldwise")``, and in sample a per-arm regression
whose arm holds one value.  A cross-fitted pooled outcome regression never reaches it, because a
training complement without both outcome classes is refused before any learner
(``_preflight_training_support``).

It never reaches a treatment-mechanism regression.  A complement without an arm is a
positivity failure, and predicting a probability of 1 for the other arm would move the
failure into an infinite inverse weight.

Each test fixes the split by replacing the fold draw, so the constant fold is built rather
than searched for.  The mutation controls replace
:func:`~cleverly.estimators._nuisance.cross_fit_companion` with a wrapper that changes how
``constant_folds`` reaches it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators import _nuisance as nuisance_module
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.learners import Folds
from cleverly.sensitivity import marginal_sensitivity_bounds
from cleverly.utils.bounds import shrink_probabilities
from cleverly.validation.nuisance import TMLE_CONSTANT_TARGET_IN_FOLD

tmle_module = importlib.import_module("cleverly.estimators.tmle")
drtmle_module = importlib.import_module("cleverly.estimators.drtmle")
marginal_module = importlib.import_module("cleverly.sensitivity.marginal")

#: The sklearn refusal of a one-class target, which the rule replaces in an outcome fold and
#: which a mechanism fold keeps.
TWO_CLASSES = "at least 2 classes"


def _fixed_split(monkeypatch: pytest.MonkeyPatch, assignment: np.ndarray, n_folds: int) -> None:
    """Every fit in the test draws ``assignment`` as its split."""
    monkeypatch.setattr(
        tmle_module, "make_folds", lambda *args, **kwargs: Folds(assignment, n_folds)
    )


def _wrap_cross_fit(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[dict[str, Any]], None]
) -> None:
    """Route every fold loop through ``change`` applied to its keyword arguments."""
    original = nuisance_module.cross_fit_companion

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        change(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(nuisance_module, "cross_fit_companion", wrapped)


def _without_constant_folds(kwargs: dict[str, Any]) -> None:
    kwargs.pop("constant_folds", None)


def _constant_folds_on_mechanism(kwargs: dict[str, Any]) -> None:
    if kwargs.get("classes") is not None:
        kwargs["constant_folds"] = []


# ------------------------------------------------------------------ TMLE, per-arm outcome

#: Three folds. Arm 1's outcome is 1 only in fold 1 (one-based), so the arm-1 model trained
#: without fold 1 sees only ``Y = 0``. Arm 0 holds both classes in every fold, so every pooled
#: complement passes the training-support preflight.
K_TMLE = 3


def _per_arm_frame(n: int = 180) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    assignment = np.arange(n) % K_TMLE
    treatment = (np.arange(n) // 3) % 2
    outcome = np.where((treatment == 0) | (assignment == 0), rng.binomial(1, 0.5, n), 0).astype(
        float
    )
    frame = pd.DataFrame({"W": rng.normal(size=n), "A": treatment, "Y": outcome})
    return frame, assignment


def _per_arm_fit(frame: pd.DataFrame) -> Any:
    estimator = TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=True,
        n_folds=K_TMLE,
        outcome_by_arm=True,
        random_state=0,
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


class TestPerArmOutcome:
    def test_the_fold_predicts_its_one_value_and_the_report_names_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, assignment = _per_arm_frame()
        _fixed_split(monkeypatch, assignment, K_TMLE)
        result = _per_arm_fit(frame)
        nuisance = result.nuisance
        held = assignment == 0
        # Precondition: arm 1's training rows of fold 1 hold one class, the sample does not.
        arm1 = frame["A"].to_numpy() == 1
        assert set(frame["Y"][arm1 & ~held]) == {0.0}
        assert set(frame["Y"][arm1]) == {0.0, 1.0}
        assert np.all(nuisance.outcome.arms[1.0][held] == 0.0)
        # Nonzero witness: the folds that fitted a learner do not predict 0.
        assert np.all(nuisance.outcome.arms[1.0][~held] > 0.0)
        assert nuisance.outcome_constant_folds == (1,)
        # The targeting step starts a prediction of exactly 0 at its probability bound, the
        # estimator's own ``alpha``, so the logit it fluctuates is finite.
        start = shrink_probabilities(nuisance.outcome.arms[1.0][held], TMLE().alpha)
        assert np.all(start == 1.0 - TMLE().alpha)
        assert np.isfinite(result["ate"].psi) and np.isfinite(result["ate"].std_error)
        report = result.diagnostics.nuisance_models()["outcome"]
        assert report.fold_omission == TMLE_CONSTANT_TARGET_IN_FOLD
        assert report.omitted_folds == (1,)
        text = str(result.diagnostics.nuisance_models().summary("full"))
        assert "fitted no learner in fold 1" in " ".join(text.split())

    def test_without_the_rule_the_fold_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: the fold loop no longer receives ``constant_folds``."""
        frame, assignment = _per_arm_frame()
        _fixed_split(monkeypatch, assignment, K_TMLE)
        _wrap_cross_fit(monkeypatch, _without_constant_folds)
        with pytest.raises(ValueError, match=TWO_CLASSES):
            _per_arm_fit(frame)

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
    def test_a_sample_whose_arm_is_constant_keeps_the_constant_cell_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rule is a training-fold rule: it does not hide a constant arm of the sample.

        Pooled, no fold is constant and the status is the existing one.  Per arm, the rule
        fires in every fold, and the status is the same.
        """
        frame, assignment = _per_arm_frame()
        frame.loc[frame["A"] == 1, "Y"] = 0.0
        _fixed_split(monkeypatch, assignment, K_TMLE)
        pooled = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=True,
            n_folds=K_TMLE,
            random_state=0,
        )
        result = pooled.fit(frame, outcome="Y", treatment="A").single()
        assert result.nuisance.outcome_constant_folds == ()
        assert result["ate"].inference == "constant_arm_cell_plugin"
        by_arm = _per_arm_fit(frame)
        assert by_arm.nuisance.outcome_constant_folds == (1, 2, 3)
        assert by_arm["ate"].inference == "constant_arm_cell_plugin"
        assert by_arm["ey0"].inference != "constant_arm_cell_plugin"


class TestPerArmInSample:
    """In sample, a per-arm regression whose arm holds one outcome value is that value.

    One is the maximum-likelihood probability of the value, as in a training fold.  The
    estimate reads a constant arm cell, so its interval is withheld.
    """

    @staticmethod
    def _fit(frame: pd.DataFrame) -> Any:
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            outcome_by_arm=True,
        )
        return estimator.fit(frame, outcome="Y", treatment="A").single()

    @staticmethod
    def _frame() -> pd.DataFrame:
        frame, _ = _per_arm_frame()
        frame.loc[frame["A"] == 1, "Y"] = 0.0
        return frame

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
    def test_the_arm_predicts_its_one_value(self) -> None:
        result = self._fit(self._frame())
        nuisance = result.nuisance
        assert np.all(nuisance.outcome.arms[1.0] == 0.0)
        # Nonzero witness: the other arm fitted a learner.
        assert np.all(nuisance.outcome.arms[0.0] > 0.0)
        assert nuisance.outcome_constant_folds == ()
        assert result["ate"].inference == "constant_arm_cell_plugin"
        assert result["ey0"].inference != "constant_arm_cell_plugin"
        assert np.isfinite(result["ey0"].std_error)

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
    @pytest.mark.parametrize("estimand", ["ate", "ey1", "att"])
    def test_the_marginal_bounds_withhold_the_interval_of_a_withheld_estimate(
        self, estimand: str
    ) -> None:
        """At gamma = 1 the set interval is a confidence interval for the estimate itself."""
        result = self._fit(self._frame())
        assert result[estimand].inference == "constant_arm_cell_plugin"
        with pytest.raises(CapabilityError, match="constant_arm_cell_plugin") as raised:
            marginal_sensitivity_bounds(result, gamma=(1.0, 2.0), estimand=estimand)
        assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP
        # Nonzero witness: the arm that fitted a learner keeps its bounds and interval.
        bounds = marginal_sensitivity_bounds(result, gamma=(1.0, 2.0), estimand="ey0")
        assert np.isfinite(bounds.rows[0].lower_limit)

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
    def test_an_unreported_estimand_on_the_constant_arm_is_withheld(self) -> None:
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            outcome_by_arm=True,
            estimands=["ey0"],
        )
        result = estimator.fit(self._frame(), outcome="Y", treatment="A").single()
        assert "ate" not in result.estimates
        with pytest.raises(CapabilityError, match="constant_arm_cell_plugin"):
            marginal_sensitivity_bounds(result, gamma=(1.0,), estimand="ate")

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.ConvergenceWarning")
    def test_without_the_status_check_the_interval_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the bounds no longer read the estimate's status."""
        monkeypatch.setattr(marginal_module, "_withheld_status", lambda *args: None)
        result = self._fit(self._frame())
        bounds = marginal_sensitivity_bounds(result, gamma=(1.0,), estimand="ate")
        row = bounds.rows[0]
        assert np.isfinite(row.lower_limit) and row.lower_limit < row.lower

    def test_without_the_rule_the_arm_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: the fold loop no longer receives ``constant_folds``."""
        _wrap_cross_fit(monkeypatch, _without_constant_folds)
        with pytest.raises(ValueError, match=TWO_CLASSES):
            self._fit(self._frame())


# ------------------------------------------------------------------ DR-TMLE, leave two out

#: Four folds. Every ``Y = 0`` row is in folds 1 and 2 (one-based), so each one-fold
#: complement holds both classes and the leave-two-out complement of that pair holds ``Y = 1``
#: only.
K_DRTMLE = 4


def _drtmle(**settings: Any) -> DRTMLE:
    return DRTMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        reduced_outcome_learner=LinearRegression(),
        reduced_mechanism_residual_learner=LinearRegression(),
        n_folds=K_DRTMLE,
        reduced_crossfit="foldwise",
        estimands=["ate"],
        random_state=0,
        **settings,
    )


def _pair_frame(n: int = 160) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    assignment = np.arange(n) % K_DRTMLE
    treatment = (np.arange(n) // 4) % 2
    outcome = np.where(assignment <= 1, rng.binomial(1, 0.5, n), 1).astype(float)
    frame = pd.DataFrame({"W": rng.normal(size=n), "A": treatment, "Y": outcome})
    return frame, assignment


class TestLeaveTwoOutOutcome:
    def test_the_pair_predicts_its_one_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, assignment = _pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        result = _drtmle().fit(frame, outcome="Y", treatment="A").single()
        inner = result.nuisance.inner
        assert inner is not None
        # Slab 1 reads the pair {1, 2} model at fold 2's rows, and slab 2 at fold 1's.
        assert np.all(inner.outcome[0].observed[assignment == 1] == 1.0)
        assert np.all(inner.outcome[1].observed[assignment == 0] == 1.0)
        for arm in (0.0, 1.0):
            assert np.all(inner.outcome[0].arms[arm][assignment == 1] == 1.0)
        # Nonzero witness: a pair whose complement holds both classes fits a learner.
        assert np.all(inner.outcome[0].observed[assignment == 2] < 1.0)
        # The production folds each held both classes.
        assert result.nuisance.outcome_constant_folds == ()
        assert np.isfinite(result["ate"].psi)

    def test_without_the_rule_the_pair_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: the fold loop no longer receives ``constant_folds``."""
        frame, assignment = _pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        _wrap_cross_fit(monkeypatch, _without_constant_folds)
        with pytest.raises(ValueError, match=TWO_CLASSES):
            _drtmle().fit(frame, outcome="Y", treatment="A")


# ------------------------------------------------------------------ never the mechanism


#: Every treated row is in folds 1 and 2 (one-based): the leave-two-out complement of that
#: pair holds no treated row.
def _arm_pair_frame(n: int = 160) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    assignment = np.arange(n) % K_DRTMLE
    treatment = np.where(assignment <= 1, (np.arange(n) // 4) % 2, 0)
    frame = pd.DataFrame(
        {"W": rng.normal(size=n), "A": treatment, "Y": rng.binomial(1, 0.5, n).astype(float)}
    )
    return frame, assignment


def _mechanism_refusal(monkeypatch: pytest.MonkeyPatch, *, preflight: bool) -> str:
    frame, assignment = _arm_pair_frame()
    _fixed_split(monkeypatch, assignment, K_DRTMLE)
    if not preflight:
        monkeypatch.setattr(drtmle_module, "refuse_untrainable_leave_two_out", lambda *a: None)
    with pytest.raises(ValueError) as raised:
        _drtmle().fit(frame, outcome="Y", treatment="A")
    return str(raised.value)


class TestNeverTheMechanism:
    def test_a_complement_without_an_arm_is_refused_before_any_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = _mechanism_refusal(monkeypatch, preflight=True)
        assert message.startswith(
            "folds 0 and 1 together hold every trainable row of arm 1 in the sample, so the "
            "leave-two-out primary regressions of reduced_crossfit='foldwise' cannot be fitted."
        )
        assert "reduced_crossfit='pooled'" in message

    def test_past_the_preflight_the_mechanism_fold_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mechanism regression keeps the classifier's refusal, the second line of defence."""
        assert TWO_CLASSES in _mechanism_refusal(monkeypatch, preflight=False)

    def test_the_rule_on_the_mechanism_changes_the_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: ``constant_folds`` passed to the mechanism regression.

        The guard of :func:`~cleverly.estimators._nuisance.cross_fit_companion` refuses it,
        so the pinned refusal above no longer holds.
        """
        _wrap_cross_fit(monkeypatch, _constant_folds_on_mechanism)
        message = _mechanism_refusal(monkeypatch, preflight=False)
        assert TWO_CLASSES not in message
        assert "constant_folds applies to a binary target with no classes only" in message
