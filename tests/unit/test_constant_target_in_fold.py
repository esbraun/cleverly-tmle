"""A cross-fitted point-treatment fold whose binary outcome takes one value predicts it.

The rule is the point-treatment twin of the longitudinal one
(``docs/technical-reference/longitudinal-tmle.md``, "A node with no event"): the
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
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.learners import Folds, SuperLearner
from cleverly.sensitivity import marginal_sensitivity_bounds
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
        # The targeting step starts a prediction of exactly 0 at its probability bound, so
        # the stored targeted predictions there are finite and inside (0, 1).
        targeted = result.repeats[0].fluctuations["mean"].targeted.arms[1.0][held]
        assert np.all(np.isfinite(targeted)) and np.all((targeted > 0.0) & (targeted < 1.0))
        assert np.isfinite(result["ate"].psi) and np.isfinite(result["ate"].std_error)
        report = result.diagnostics.nuisance_models()["outcome"]
        assert report.fold_omission == TMLE_CONSTANT_TARGET_IN_FOLD
        assert report.omitted_folds == (1,)
        text = str(result.diagnostics.nuisance_models().summary("full"))
        assert "fitted no learner in fold 1" in " ".join(text.split())

    def test_the_targeting_step_starts_the_fold_at_the_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The targeted predictions of the constant fold move with ``alpha``, the bound.

        A targeting path that read the exact 0 instead of its bound would not see ``alpha``.
        The rows outside the fold start inside the bound at both values, so they do not move.
        """
        frame, assignment = _per_arm_frame()
        _fixed_split(monkeypatch, assignment, K_TMLE)
        held = assignment == 0
        targeted = {}
        for alpha in (0.9995, 0.99):
            estimator = TMLE(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                cross_fit=True,
                n_folds=K_TMLE,
                outcome_by_arm=True,
                random_state=0,
                alpha=alpha,
            )
            fit = estimator.fit(frame, outcome="Y", treatment="A").single()
            assert np.all(fit.nuisance.outcome.arms[1.0][held] == 0.0)
            assert np.all(fit.nuisance.outcome.arms[1.0][~held] > 1.0 - 0.99)
            targeted[alpha] = fit.repeats[0].fluctuations["mean"].targeted.arms[1.0]
        assert np.all(targeted[0.99][held] > targeted[0.9995][held])

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


def _super_learner() -> SuperLearner:
    """A small package Super Learner, whose inner split stratifies on its target."""
    return SuperLearner([LogisticRegression(), LogisticRegression(C=0.1)], task="classification")


def _super_learner_fit(frame: pd.DataFrame) -> Any:
    estimator = TMLE(
        outcome_learner=_super_learner(),
        treatment_learner=LogisticRegression(),
        cross_fit=True,
        n_folds=K_TMLE,
        outcome_by_arm=True,
        random_state=0,
    )
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def _shortfall_without_the_exemption(
    self: Any, data: Any, folds: Any, *, fits_treatment: bool = True
) -> Any:
    """``_super_learner_complement_shortfall`` with ``count < 2`` for every class."""
    size = int(np.asarray(data.observed).size)
    for role, _task, rows, target, label in self._super_learner_inner_split_roles(
        data, fits_treatment=fits_treatment
    ):
        for repeat, draw in enumerate(folds):
            for fold, (train, _) in enumerate(draw):
                kept = np.zeros(size, dtype=bool)
                kept[train] = True
                values = target[kept & rows]
                for value in np.unique(target[rows]):
                    count = int(np.count_nonzero(values == value))
                    if count < 2:
                        return role, repeat, fold, label(float(value)), count
    return None


class TestPackageSuperLearner:
    """A package Super Learner needs two rows of each class only where it runs.

    A per-arm complement with no row of one outcome class takes the rule and fits no Super
    Learner, so the inner-split preflight lets it through.  One row of a class is still
    refused, because the learner runs there and its stratified split needs two.
    """

    def test_a_complement_without_a_class_takes_the_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, assignment = _per_arm_frame()
        _fixed_split(monkeypatch, assignment, K_TMLE)
        result = _super_learner_fit(frame)
        assert result.nuisance.outcome_constant_folds == (1,)
        assert np.all(result.nuisance.outcome.arms[1.0][assignment == 0] == 0.0)
        assert np.isfinite(result["ate"].std_error)

    def test_one_row_of_a_class_is_still_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, assignment = _per_arm_frame()
        lone = int(np.flatnonzero((frame["A"].to_numpy() == 1) & (assignment == 1))[0])
        frame.loc[lone, "Y"] = 1.0
        _fixed_split(monkeypatch, assignment, K_TMLE)
        with pytest.raises(DataError, match="holds 1 respondent\\(s\\) in arm 1 with outcome 1"):
            _super_learner_fit(frame)

    def test_without_the_exemption_the_constant_fold_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the preflight asks two rows of a class the fold does not fit."""
        monkeypatch.setattr(
            tmle_module.TMLE,
            "_super_learner_complement_shortfall",
            _shortfall_without_the_exemption,
        )
        frame, assignment = _per_arm_frame()
        _fixed_split(monkeypatch, assignment, K_TMLE)
        with pytest.raises(DataError, match="holds 0 respondent\\(s\\) in arm 1 with outcome 1"):
            _super_learner_fit(frame)


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
    def test_validate_names_the_constant_arm_as_the_cause(self) -> None:
        result = self._fit(self._frame())
        item = next(i for i in result.validate().items if i.name == "score_equations")
        assert item.status.value == "failed"
        assert "constant_arm_cell_plugin" in item.detail and "ate" in item.detail

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
    defaults: dict[str, Any] = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_mechanism_residual_learner": LinearRegression(),
        "n_folds": K_DRTMLE,
        "reduced_crossfit": "foldwise",
        "estimands": ["ate"],
        "random_state": 0,
    }
    return DRTMLE(**{**defaults, **settings})


def _pair_frame(n: int = 160) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    assignment = np.arange(n) % K_DRTMLE
    treatment = (np.arange(n) // 4) % 2
    outcome = np.where(assignment <= 1, rng.binomial(1, 0.5, n), 1).astype(float)
    frame = pd.DataFrame({"W": rng.normal(size=n), "A": treatment, "Y": outcome})
    return frame, assignment


def _per_arm_pair_frame(n: int = 160) -> tuple[pd.DataFrame, np.ndarray]:
    """Arm 1's ``Y = 0`` rows all sit in folds 1 and 2 (one-based); arm 0 is unrestricted."""
    rng = np.random.default_rng(0)
    assignment = np.arange(n) % K_DRTMLE
    treatment = (np.arange(n) // 4) % 2
    outcome = rng.binomial(1, 0.5, n).astype(float)
    outcome[(treatment == 1) & (assignment >= 2)] = 1.0
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

    def test_a_per_arm_pair_predicts_its_one_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``outcome_by_arm=True``: arm 1's rows outside folds 1 and 2 hold ``Y = 1`` only."""
        frame, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        result = _drtmle(outcome_by_arm=True).fit(frame, outcome="Y", treatment="A").single()
        inner = result.nuisance.inner
        assert inner is not None
        assert np.all(inner.outcome[0].arms[1.0][assignment == 1] == 1.0)
        assert np.all(inner.outcome[1].arms[1.0][assignment == 0] == 1.0)
        # Nonzero witnesses: arm 0 of the same pair, and arm 1 of a pair with both classes.
        assert np.all(inner.outcome[0].arms[0.0][assignment == 1] < 1.0)
        assert np.all(inner.outcome[0].arms[1.0][assignment == 2] < 1.0)
        assert np.isfinite(result["ate"].psi)

    def test_without_the_rule_the_per_arm_pair_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the per-arm fold loop no longer receives ``constant_folds``."""
        frame, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        _wrap_cross_fit(monkeypatch, _without_constant_folds)
        with pytest.raises(ValueError, match=TWO_CLASSES):
            _drtmle(outcome_by_arm=True).fit(frame, outcome="Y", treatment="A")

    def test_without_the_rule_the_pair_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: the fold loop no longer receives ``constant_folds``."""
        frame, assignment = _pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        _wrap_cross_fit(monkeypatch, _without_constant_folds)
        with pytest.raises(ValueError, match=TWO_CLASSES):
            _drtmle().fit(frame, outcome="Y", treatment="A")


def _super_learner_drtmle() -> DRTMLE:
    return _drtmle(outcome_learner=_super_learner(), outcome_by_arm=True)


def _lone_pair_frame() -> pd.DataFrame:
    """Arm 1 keeps one ``Y = 0`` row outside folds 1 and 2 (one-based)."""
    frame, assignment = _per_arm_pair_frame()
    lone = int(np.flatnonzero((frame["A"].to_numpy() == 1) & (assignment == 2))[0])
    frame.loc[lone, "Y"] = 0.0
    return frame


#: The leave-two-out refusal of a pair that keeps one row of an outcome class.
LONE_PAIR = (
    r"folds 0 and 1 leave 1 respondent\(s\) in arm 1 with outcome 0 outside them.*"
    r"reduced_crossfit='pooled'"
)


class TestLeaveTwoOutSuperLearner:
    """A package Super Learner on the pairs: no row of a class takes the rule, one refuses."""

    def test_a_pair_without_a_class_takes_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        result = _super_learner_drtmle().fit(frame, outcome="Y", treatment="A").single()
        inner = result.nuisance.inner
        assert inner is not None
        assert np.all(inner.outcome[0].arms[1.0][assignment == 1] == 1.0)
        assert np.all(inner.outcome[0].arms[1.0][assignment == 2] < 1.0)
        assert np.isfinite(result["ate"].psi)

    def test_one_row_of_a_class_is_refused_before_any_learner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        fitted: list[object] = []
        _wrap_cross_fit(monkeypatch, fitted.append)
        with pytest.raises(ValueError, match=LONE_PAIR):
            _super_learner_drtmle().fit(_lone_pair_frame(), outcome="Y", treatment="A")
        assert fitted == []

    def test_without_the_check_the_super_learner_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the leave-two-out check no longer reads the Super Learner roles."""
        original = drtmle_module.refuse_untrainable_leave_two_out
        monkeypatch.setattr(
            drtmle_module,
            "refuse_untrainable_leave_two_out",
            lambda data, folds, **_: original(data, folds),
        )
        _, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        with pytest.raises(ValueError, match="the rarer class has only 1 member"):
            _super_learner_drtmle().fit(_lone_pair_frame(), outcome="Y", treatment="A")

    def test_without_the_exemption_the_constant_pair_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: the outcome role loses the binary count-0 exemption."""
        original = drtmle_module.DRTMLE._leave_two_out_super_learner_roles

        def renamed(self: Any, data: Any) -> Any:
            return tuple(("mutated", *rest) for _, *rest in original(self, data))

        monkeypatch.setattr(drtmle_module.DRTMLE, "_leave_two_out_super_learner_roles", renamed)
        frame, assignment = _per_arm_pair_frame()
        _fixed_split(monkeypatch, assignment, K_DRTMLE)
        with pytest.raises(ValueError, match="folds 0 and 1 leave 0 respondent"):
            _super_learner_drtmle().fit(frame, outcome="Y", treatment="A")


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
        monkeypatch.setattr(drtmle_module, "refuse_untrainable_leave_two_out", lambda *a, **k: None)
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
