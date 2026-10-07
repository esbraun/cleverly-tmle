"""A censoring node at which nobody was censored has the fixed factor one.

On the default integer grid of a time-to-event input every time is at least one, so the
first censoring column is all ones.  Before the fix the engine handed that constant target
to the censoring learner, and a standard classifier raised ``ValueError: This solver needs
samples of at least 2 classes``, which is not a refusal.  The node now fits no learner, its
factor is exactly one at every plan, and the nuisance report shows an omission.  A
cross-fitted training fold with no censored unit predicts its empirical rate, one.

``survtmle`` behaves the same way: it sets ``G_dC = 1`` at ``t = 1`` and takes a ``noCens``
branch when no censoring is observed.

The mutation controls put the learner back on the constant target and see the crash
return, so the tests fail if the guard is removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import sequential as sequential_module
from cleverly.validation.longitudinal import (
    LONGITUDINAL_NO_CENSORING,
    LONGITUDINAL_NO_CENSORING_IN_FOLD,
)
from tests.discrete_law_longitudinal import CellMeans

EVENTS = ["Y1", "Y2", "Y3"]
TREATMENT = ["A1", "A2", "A3"]
CENSORING = ["C1", "C2", "C3"]


def _panel(*, n: int = 400, seed: int = 0, censored_at_2: int | None = None) -> pd.DataFrame:
    """A three-node survival panel with one baseline arm and no censoring at node 1.

    ``censored_at_2`` keeps exactly that many censored units at node 2, so that a two-fold
    split leaves one training fold with none when it is one.
    """
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n)
    a = rng.binomial(1, 0.3 + 0.4 * w)
    frame = pd.DataFrame({"W": w})
    alive = np.ones(n, dtype=bool)
    observed = np.ones(n, dtype=bool)
    for k in (1, 2, 3):
        in_study = alive & observed
        frame[f"A{k}"] = np.where(in_study, a, np.nan)
        if k == 1:
            stay = np.ones(n)
        elif k == 2 and censored_at_2 is not None:
            stay = np.ones(n)
            stay[np.flatnonzero(in_study)[:censored_at_2]] = 0.0
        else:
            stay = rng.binomial(1, 0.9, n).astype(float)
        observed = observed & (stay == 1.0)
        frame[f"C{k}"] = np.where(in_study, stay, np.nan)
        y = rng.binomial(1, 0.15 + 0.1 * a, n)
        frame[f"Y{k}"] = np.where(alive & observed, y, np.nan)
        alive = alive & observed & (y == 0)
    return frame


def _fit(frame: pd.DataFrame, *, n_folds: int = 1, censoring_learner: object = None) -> object:
    return LTMLE(
        {"t": 1, "c": 0},
        n_folds=n_folds,
        random_state=3,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=(
            LogisticRegression(max_iter=1000) if censoring_learner is None else censoring_learner
        ),
    ).fit(frame, outcome=EVENTS, treatment=TREATMENT, baseline=["W"], censoring=CENSORING)


def _psis(result: object) -> np.ndarray:
    return np.array([estimate.psi for estimate in result.estimates.values()])  # type: ignore[attr-defined]


def _force_learner(monkeypatch: pytest.MonkeyPatch, *, node: bool, fold: bool) -> None:
    """Remove the node guard, the fold guard, or both."""
    if node:
        monkeypatch.setattr(
            sequential_module, "_retains_every_eligible_unit", lambda *args, **kwargs: False
        )
    if fold:
        original = sequential_module.cross_fit_predictions

        def without_fold_guard(*args: object, **kwargs: object) -> object:
            kwargs.pop("constant_folds", None)
            return original(*args, **kwargs)

        monkeypatch.setattr(sequential_module, "cross_fit_predictions", without_fold_guard)


def test_a_node_with_no_censoring_fits_with_a_standard_classifier() -> None:
    result = _fit(_panel())
    mechanism = result.mechanism  # type: ignore[attr-defined]
    assert mechanism.no_censoring_nodes == (1,)
    assert mechanism.no_censoring_folds == {}
    for label in ("t", "c"):
        assert np.array_equal(mechanism.censoring[0][label], np.ones(400))
    assert np.array_equal(mechanism.censoring_observed[0], np.ones(400))
    assert mechanism.censoring_diagnostics[0] == ()
    assert np.all(np.isfinite(_psis(result)))

    report = result.diagnostics.nuisance_models()  # type: ignore[attr-defined]
    censoring_rows = sorted(row.time for row in report.rows if row.role == "censoring")
    assert censoring_rows == [2, 3]
    assert ("censoring", 1, LONGITUDINAL_NO_CENSORING) in [
        (item.role, item.time, item.reason) for item in report.omissions
    ]


def test_the_crash_returns_without_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_learner(monkeypatch, node=True, fold=True)
    with pytest.raises(ValueError, match="at least 2 classes"):
        _fit(_panel())


def test_the_fixed_factor_is_what_a_learner_that_accepts_the_target_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix moves no number where the old engine did not crash.

    A cell-mean learner accepts a constant target and returns exactly one, so the forced
    learner fit and the guarded fit agree bit for bit.
    """
    guarded = _fit(_panel(), censoring_learner=CellMeans())
    _force_learner(monkeypatch, node=True, fold=True)
    forced = _fit(_panel(), censoring_learner=CellMeans())
    assert forced.mechanism.no_censoring_nodes == ()  # type: ignore[attr-defined]
    assert np.array_equal(_psis(guarded), _psis(forced))
    for name in guarded.influence_curves:  # type: ignore[attr-defined]
        assert np.array_equal(
            guarded.influence_curves[name],  # type: ignore[attr-defined]
            forced.influence_curves[name],  # type: ignore[attr-defined]
        )


def test_a_training_fold_with_no_censoring_predicts_one() -> None:
    frame = _panel(censored_at_2=1)
    result = _fit(frame, n_folds=2)
    mechanism = result.mechanism  # type: ignore[attr-defined]
    assert mechanism.no_censoring_nodes == (1,)
    assert set(mechanism.no_censoring_folds) == {2}
    (fold,) = mechanism.no_censoring_folds[2]
    # Training fold ``k`` is every fold but ``k``, so its held-out rows predict one.
    held_out = result.folds.assignment == fold - 1  # type: ignore[attr-defined]
    assert np.array_equal(mechanism.censoring_observed[1][held_out], np.ones(held_out.sum()))
    assert not np.all(mechanism.censoring_observed[1][~held_out] == 1.0)
    assert np.all(np.isfinite(_psis(result)))

    report = result.diagnostics.nuisance_models()  # type: ignore[attr-defined]
    reasons = [(item.role, item.time, item.reason) for item in report.omissions]
    assert ("censoring", 2, LONGITUDINAL_NO_CENSORING_IN_FOLD) in reasons
    assert any(row.role == "censoring" and row.time == 2 for row in report.rows)


def test_the_fold_crash_returns_without_the_fold_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_learner(monkeypatch, node=False, fold=True)
    with pytest.raises(ValueError, match="at least 2 classes"):
        _fit(_panel(censored_at_2=1), n_folds=2)


def test_a_zero_weight_censored_row_does_not_count() -> None:
    """The eligible sample is the at-risk rows with positive weight."""
    frame = _panel()
    frame["w"] = 1.0
    frame.loc[0, "C1"] = 0.0
    frame.loc[0, ["Y1", "A2", "C2", "Y2", "A3", "C3", "Y3"]] = np.nan
    frame.loc[0, "w"] = 0.0
    result = LTMLE(
        {"t": 1, "c": 0},
        n_folds=1,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
    ).fit(
        frame,
        outcome=EVENTS,
        treatment=TREATMENT,
        baseline=["W"],
        censoring=CENSORING,
        weights="w",
    )
    assert result.mechanism.no_censoring_nodes == (1,)  # type: ignore[attr-defined]
