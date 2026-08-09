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

from typing import Any

import numpy as np
import pandas as pd
import pytest

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

    Recorded at :func:`cleverly.estimators._nuisance.cross_fit_predictions` as the
    ``sequential`` module sees it, rather than through a learner: a learner is cloned per
    fold and reached only after the call, so spying there would pin what sklearn received
    instead of what this module passed.  The outcome sequence is the one whose
    ``predict_designs`` is keyed ``"history"``; the mechanism keys by regimen label.
    """
    from cleverly.longitudinal import sequential

    original = sequential.cross_fit_predictions
    recorded: list[tuple[str, np.ndarray]] = []

    def spy(learner: Any, design: Any, *args: Any, **kwargs: Any) -> Any:
        keys = tuple(kwargs["predict_designs"])
        recorded.append(("outcome" if keys == ("history",) else "mechanism", np.array(design)))
        return original(learner, design, *args, **kwargs)

    monkeypatch.setattr(sequential, "cross_fit_predictions", spy)
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
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        simultaneous=False,
        random_state=0,
    ).fit(data)

    outcome_designs = [matrix for kind, matrix in designs if kind == "outcome"]
    # One per node per regimen, and the recursion runs backwards within each regimen.
    expected = [data.covariate_history(time) for time in (2, 1)] * len(REGIMENS)
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
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        simultaneous=False,
        random_state=0,
    ).fit(data)

    widths = {matrix.shape[1] for kind, matrix in designs if kind == "mechanism"}
    # t=1: [W1] and [W1, A1]; t=2: [W1, L2, A1] and [W1, L2, A1, A2].
    assert widths == {1, 2, 3, 4}
    assert widths - {data.covariate_history(time).shape[1] for time in (1, 2)}
