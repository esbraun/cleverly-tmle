"""What matrix the sequential regression is actually handed at each node.

``sequential.fit_regimen`` regresses on :meth:`LongitudinalData.covariate_history` --
``[W, L_1, ..., L_t]``, with **no** treatment columns -- while the mechanism models use
:meth:`LongitudinalData.history_design`, which carries them.  Which of the two the
outcome sequence gets is an argument, written out in ``history_design``'s docstring: among
the followers ``A_s = d_s(W, L_1, ..., L_s)`` is a deterministic function of columns the
design already carries, so adding it buys no information.

**This module does not check that argument, and cannot.**  Both designs are consistent,
so there is no second answer to compare against; the measured attempts are recorded in
``tests/e2e/test_ltmle.py``.  What this module checks is the *call site*: that
the matrix ``fit_regimen`` passes is that method's output and not the other one's, bit for
bit, so the decision cannot be reversed silently by an edit that looks like a tidy-up.  A
statistical test would be the wrong instrument for that and a structural one is the right
one -- it fails on the edit, and says nothing about which design is correct.

Verified by mutation: changing ``design = data.covariate_history(time)`` to
``data.history_design(time, treatment=plan.values)`` turns
:func:`test_the_outcome_regression_is_handed_the_covariate_history` red.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_longitudinal_survival
from cleverly.longitudinal import LTMLE, LongitudinalData

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

#: A regimen whose second node is a rule, so the assignment really does vary by row.  A
#: rule that ignored the history would assign a constant, and a constant column is
#: standardised to zeros -- which is one of the two things that hid this from the e2e
#: comparison in the first place, and would hide the mutation from a width check here.
REGIMENS: dict[str, Any] = {
    "never": 0,
    "treat if l2 rises": (1, lambda history: history["L2"] > 0.0),
}


def multivalue_panel(n: int = 600, *, seed: int = 7) -> pd.DataFrame:
    """The same shape, with a **three-level** first treatment node.

    ``P(A2 = 1 | A1)`` is non-monotone in ``A1`` on purpose. That is what separates the
    two candidate encodings of an earlier arm: a single ordinal column forces any model
    linear in its design to order the three fitted probabilities, and drop-first
    indicators do not. Nothing else in the panel depends on ``A1``, so the comparison is
    about the encoding rather than about the rest of the law.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    a1 = rng.integers(0, 3, n).astype(float)
    c1 = (rng.random(n) < 0.9).astype(float)
    l2 = np.where(c1 == 1, rng.standard_normal(n), np.nan)
    middle = a1 == 1.0
    a2 = np.where(c1 == 1, (rng.random(n) < np.where(middle, 0.85, 0.15)).astype(float), np.nan)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    y = np.where((c1 == 1) & (c2 == 1), rng.integers(0, 2, n).astype(float), np.nan)
    return pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})


def panel(n: int = 200, *, seed: int = 5) -> pd.DataFrame:
    """A small two-time-point panel with monotone censoring."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    a1 = rng.integers(0, 2, n).astype(float)
    c1 = (rng.random(n) < 0.85).astype(float)
    l2 = np.where(c1 == 1, rng.standard_normal(n), np.nan)
    a2 = np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    y = np.where((c1 == 1) & (c2 == 1), rng.integers(0, 2, n).astype(float), np.nan)
    return pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})


@pytest.fixture
def designs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, np.ndarray]]:
    """Every design matrix ``sequential`` hands a nuisance fit, in the order it hands it.

    Recorded at the two helpers the module calls, rather than through a learner: a learner
    is cloned per fold and reached only after the call, so spying there would pin what
    scikit-learn received instead of what this module passed.  The mechanism goes through
    ``cross_fit_companion``, because it retains each fold model's predictions on every row.
    The outcome recursion goes through ``cross_fit_predictions`` at every node, on one fold
    or on many -- an outer fold's model is a one-fold split with that fold's training rows
    as the fit mask, which is why the cross-fitted recursion needs no second call site.
    """
    from cleverly.longitudinal import sequential

    original_companion = sequential.cross_fit_companion
    original_predictions = sequential.cross_fit_predictions
    recorded: list[tuple[str, np.ndarray]] = []

    def companion_spy(learner: Any, design: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.append(("mechanism", np.array(design)))
        return original_companion(learner, design, *args, **kwargs)

    def predictions_spy(learner: Any, design: Any, *args: Any, **kwargs: Any) -> Any:
        recorded.append(("outcome", np.array(design)))
        return original_predictions(learner, design, *args, **kwargs)

    monkeypatch.setattr(sequential, "cross_fit_companion", companion_spy)
    monkeypatch.setattr(sequential, "cross_fit_predictions", predictions_spy)
    return recorded


def test_the_outcome_regression_is_handed_the_covariate_history(
    designs: list[tuple[str, np.ndarray]],
) -> None:
    """``covariate_history(t)`` exactly, for every node of every regimen.

    Equality of the whole matrix rather than of its width: a width check would pass an
    edit that swapped a covariate column for a treatment one, and the columns are what the
    argument in ``history_design`` is about.
    """
    frame = panel()
    data = LongitudinalData.from_frame(frame, **COLUMNS)
    LTMLE(
        REGIMENS,
        reference="never",
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        simultaneous=False,
        random_state=0,
    ).fit(data)

    outcome_designs = [matrix for kind, matrix in designs if kind == "outcome"]
    # One per outer fold, node, and regimen.  Each fold completes its recursion backwards.
    expected = [data.covariate_history(time) for time in (2, 1)] * (2 * len(REGIMENS))
    assert len(outcome_designs) == len(expected)
    for seen, want in zip(outcome_designs, expected, strict=True):
        np.testing.assert_array_equal(seen, want)


def test_the_mechanism_is_handed_the_treatment_columns(
    designs: list[tuple[str, np.ndarray]],
) -> None:
    """The other half of the same statement, so the pin above cannot be read as a ban.

    The treatment columns belong in the mechanism's conditioning set -- ``g_2`` is
    ``P(A_2 = 1 | H_2, A_1 = a_1)`` -- and this fails if an edit ever "tidied" the outcome
    design and the mechanism's together.
    """
    frame = panel()
    data = LongitudinalData.from_frame(frame, **COLUMNS)
    LTMLE(
        REGIMENS,
        reference="never",
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        simultaneous=False,
        random_state=0,
    ).fit(data)

    widths = {matrix.shape[1] for kind, matrix in designs if kind == "mechanism"}
    # t=1: [W1] and [W1, A1]; t=2: [W1, L2, A1] and [W1, L2, A1, A2].  Both nodes are
    # two-level here, so each contributes the single 0/1 code column it always did; the
    # ``K - 1`` reading of the same rule is pinned on the three-level panel below.
    assert widths == {1, 2, 3, 4}
    assert widths - {data.covariate_history(time).shape[1] for time in (1, 2)}


def _crossfit_result(frame: pd.DataFrame) -> Any:
    return LTMLE(
        {"never": 0},
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        simultaneous=False,
        random_state=0,
    ).fit(LongitudinalData.from_frame(frame, **COLUMNS))


def test_each_row_uses_one_complete_outer_training_recursion() -> None:
    """A held-out outcome cannot alter any prediction used for its own row."""
    frame = panel()
    row = int(np.flatnonzero(frame["Y"].notna().to_numpy())[0])
    original = _crossfit_result(frame)
    changed = frame.copy()
    changed.loc[row, "Y"] = 1.0 - changed.loc[row, "Y"]
    perturbed = _crossfit_result(changed)

    np.testing.assert_array_equal(original.folds.assignment, perturbed.folds.assignment)
    for left, right in zip(
        original.fits["never"].steps, perturbed.fits["never"].steps, strict=True
    ):
        assert left.initial[row] == right.initial[row]
        assert left.targeted[row] == right.targeted[row]


def test_a_fold_that_does_not_converge_is_reported_once_per_regimen() -> None:
    """The per-fold solves are silenced, so something has to speak for them.

    Each outer solve runs with ``warn=False``: ten folds at ten nodes would otherwise emit
    a hundred warnings for one problem. The cost of silencing them is that a fit could
    fail in three folds of ten and say nothing, and the stitched score cannot betray it --
    that score is a nonzero residual on a perfectly healthy fit, so there is no threshold
    it crosses. The aggregated warning is the only channel left, which is why it is tested
    rather than assumed.

    One warning per regimen and not one per fold, because that is the count a reader can
    act on. The message has to name the nodes, since which node failed is what decides
    whether the fit is salvageable.
    """
    from cleverly.exceptions import ConvergenceWarning
    from cleverly.longitudinal import sequential

    original = sequential.solve_fluctuation

    def never_converges(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(
            original(*args, **kwargs), converged=False, failure="max_iter_reached"
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sequential, "solve_fluctuation", never_converges)
        with pytest.warns(ConvergenceWarning, match="outer-fold targeting solve") as caught:
            _crossfit_result(panel())

    # One regimen, two nodes, two folds: four failed solves, named once.
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "4 outer-fold targeting solve(s) did not converge" in message
    assert "'never'" in message
    assert "[1, 2]" in message
    assert "max_iter_reached" in message


def test_fold_artifacts_cover_each_row_and_match_the_stitched_mechanism() -> None:
    """The public fold record identifies every held-out prediction and update."""
    result = _crossfit_result(panel())
    fit = result.fits["never"]
    for step in fit.steps:
        indices = np.concatenate([record.index for record in step.fluctuation.folds])
        np.testing.assert_array_equal(np.sort(indices), np.arange(result.n))
        assert step.fluctuation.n_solver_calls == result.folds.n_folds

    for time, node in enumerate(result.mechanism.treatment):
        slabs = result.mechanism.treatment_by_fold[time]["never"]
        stitched = slabs[result.folds.assignment, np.arange(result.n)]
        np.testing.assert_allclose(stitched, node["never"], rtol=0.0, atol=0.0)


def test_a_held_out_terminal_event_cannot_enter_its_survival_recursion() -> None:
    """The horizon-two survival recursion has the same whole-fold leakage boundary."""
    frame, _ = make_longitudinal_survival(n=500, seed=41, backend="pandas")
    eligible = (
        (frame["Y1"] == 0.0) & frame["Y2"].notna() & frame["C1"].eq(1.0) & frame["C2"].eq(1.0)
    )
    row = int(np.flatnonzero(eligible.to_numpy())[0])

    def fit(candidate: pd.DataFrame) -> Any:
        return LTMLE(
            {"never": 0},
            outcome_learner=sklearn.linear_model.LinearRegression(),
            pseudo_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            simultaneous=False,
            random_state=0,
        ).fit(
            candidate,
            outcome=("Y1", "Y2"),
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        )

    original = fit(frame)
    changed = frame.copy()
    changed.loc[row, "Y2"] = 1.0 - changed.loc[row, "Y2"]
    perturbed = fit(changed)
    np.testing.assert_array_equal(original.folds.assignment, perturbed.folds.assignment)
    left = original.fits["never @ t=2"]
    right = perturbed.fits["never @ t=2"]
    for left_step, right_step in zip(left.steps, right.steps, strict=True):
        assert left_step.initial[row] == right_step.initial[row]
        assert left_step.targeted[row] == right_step.targeted[row]


class TestAThreeLevelArmEntersAsIndicators:
    """How a categorical node is *coded* into the mechanism's conditioning set.

    The exact-law suites cannot see this. Their learners are saturated and partition by
    distinct design row, and an ordinal code and a drop-first indicator tuple are a
    bijection -- so every prediction, every influence curve and every remainder is
    identical under either encoding. What separates them is a learner that is *linear in
    its design*, which is why the witness below is a ``glm`` on a deliberately
    non-monotone truth rather than another pass over ``discrete_law_longitudinal``.
    """

    def test_the_mechanism_design_carries_k_minus_one_columns(
        self, designs: list[tuple[str, np.ndarray]]
    ) -> None:
        """``A1`` has three levels, so it occupies two columns rather than one."""
        frame = multivalue_panel()
        data = LongitudinalData.from_frame(frame, **COLUMNS)
        LTMLE(
            {"low then off": (0, 0), "mid then on": (1, 1)},
            reference="low then off",
            outcome_learner=sklearn.linear_model.LinearRegression(),
            pseudo_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            simultaneous=False,
            random_state=0,
        ).fit(data)

        widths = {matrix.shape[1] for kind, matrix in designs if kind == "mechanism"}
        # t=1: [W1] and [W1, A1a, A1b]; t=2: [W1, L2, A1a, A1b] and the same plus A2.
        # Under one ordinal column per node these would be {1, 2, 3, 4}.
        assert widths == {1, 3, 4, 5}

    def test_the_two_columns_are_the_drop_first_indicators(self) -> None:
        """Which two columns, and in which order -- a width check cannot say that."""
        data = LongitudinalData.from_frame(multivalue_panel(), **COLUMNS)
        design = data.history_design(2, treatment=(2.0, 1.0), include_current=True)
        # [W1, L2] then A1's indicators for levels 1 and 2, then A2's single 0/1 column.
        np.testing.assert_array_equal(design[:, -3], np.zeros(data.n))
        np.testing.assert_array_equal(design[:, -2], np.ones(data.n))
        np.testing.assert_array_equal(design[:, -1], np.ones(data.n))

        middle = data.history_design(2, treatment=(1.0, 0.0), include_current=True)
        np.testing.assert_array_equal(middle[:, -3], np.ones(data.n))
        np.testing.assert_array_equal(middle[:, -2], np.zeros(data.n))
        np.testing.assert_array_equal(middle[:, -1], np.zeros(data.n))

    def test_an_ordinal_column_could_not_fit_a_non_monotone_mechanism(self) -> None:
        """The witness: the two encodings disagree, and only one can reach the truth.

        ``P(A2 = 1 | A1)`` is ``0.15, 0.85, 0.15`` across ``A1``'s three levels -- a
        deliberately non-monotone law that no model linear in a single ordinal ``A1``
        column can represent, since such a model orders its three fitted values. The
        indicators leave the three arms unconstrained and recover it.

        Asserted as a *gap* rather than as inequality: two encodings would differ in the
        last bit for uninteresting reasons, and what matters is that the ordinal one is
        wrong by more than a rounding.
        """
        frame = multivalue_panel(n=4000, seed=11)
        data = LongitudinalData.from_frame(frame, **COLUMNS)
        at_risk = data.uncensored_through(1)
        a2 = np.nan_to_num(data.treatment[:, 1], nan=0.0)[at_risk]
        codes = data.treatment[:, 0][at_risk]

        indicators = data.history_design(2)[at_risk]
        ordinal = np.column_stack([data.covariate_history(2)[at_risk], codes])

        def fitted_by_arm(design: np.ndarray) -> np.ndarray:
            model = LogisticRegression(max_iter=1000).fit(design, a2)
            predicted = model.predict_proba(design)[:, 1]
            return np.array([predicted[codes == arm].mean() for arm in (0.0, 1.0, 2.0)])

        truth = np.array([0.15, 0.85, 0.15])
        assert np.abs(fitted_by_arm(indicators) - truth).max() < 0.05
        # Ordered fitted values cannot straddle the middle arm, so it is the one that
        # misses, and by most of the distance between the arms.
        assert np.abs(fitted_by_arm(ordinal) - truth).max() > 0.3
