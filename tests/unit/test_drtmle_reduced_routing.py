r"""Which learner slot fits each of ``DRTMLE``'s three reduced regressions.

Benkeser et al. (2017) fit three univariate regressions: :math:`Q_r` (the outcome residual
on :math:`\hat g`), :math:`g_{r,1}` (the arm indicator on :math:`\hat{\bar Q}`) and
:math:`g_{r,2}` (:math:`(1_a - \hat g)/\hat g` on :math:`\hat{\bar Q}`).  R ``drtmle`` fits
both reduced propensities with its ``gr`` learner and only :math:`Q_r` with its ``Qr``
learner.  ``reduced_outcome_learner`` fits :math:`Q_r`, ``reduced_treatment_learner`` fits
:math:`g_{r,1}`, and ``reduced_mechanism_residual_learner`` fits :math:`g_{r,2}`.  A
``None`` regression slot resolves to the default regression learner on its own, so
``reduced_outcome_learner=`` alone does not move :math:`g_{r,2}`.

The two regression slots are told apart by their rows: :math:`Q_r` trains on the rows that
took the arm, and :math:`g_{r,2}` on every row.  The fits here are in sample, so no fold
removes rows from either.

A classifier cannot fit either regression, so a guarded fit refuses one in a regression slot
before any learner fits.  A ``None`` slot takes ``outcome_learner``, which is often a
classifier on a binary outcome.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CapabilityError
from cleverly.estimators import DRTMLE

N = 300

#: ``(tag, rows, columns)`` of every recorded fit, appended by :class:`Recording`.
CALLS: list[tuple[str, int, int]] = []


class Recording(LinearRegression):
    """Least squares that records the tag, rows and columns of each fit it makes."""

    def __init__(self, tag: str = "") -> None:
        super().__init__()
        self.tag = tag

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Recording:
        design = np.asarray(X, dtype=float)
        width = 1 if design.ndim == 1 else design.shape[1]
        CALLS.append((self.tag, design.shape[0], width))
        return super().fit(X, y, sample_weight=sample_weight)


def _frame() -> pd.DataFrame:
    """A draw whose logistic mechanism is wrong, so :math:`g_{r,2}` is not zero."""
    rng = np.random.default_rng(11)
    w1, w2 = rng.normal(size=N), rng.normal(size=N)
    a = rng.binomial(1, expit(1.5 * w1**2 - 1.0))
    y = np.clip(expit(0.8 * w1 + 0.5 * a - 0.4 * w2) + rng.normal(0.0, 0.05, N), 0.0, 1.0)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def _fit(**learners: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
        **learners,
    }
    return DRTMLE(**settings).fit(_frame(), outcome="Y", treatment="A").single()


def _reduced_calls(tag: str) -> list[tuple[int, int]]:
    """The rows and columns of every one-column fit recorded under ``tag``."""
    return [(rows, width) for name, rows, width in CALLS if name == tag and width == 1]


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    CALLS.clear()


class TestEachRegressionSlotFitsItsOwnRegression:
    def test_the_mechanism_residual_slot_fits_gr2_and_the_outcome_slot_fits_qr(self) -> None:
        _fit(
            reduced_outcome_learner=Recording("qr"),
            reduced_mechanism_residual_learner=Recording("gr2"),
        )
        qr, gr2 = _reduced_calls("qr"), _reduced_calls("gr2")
        assert qr and gr2
        # Q_r trains on the arm's rows only; g_{r,2} trains on every row.
        assert all(rows < N for rows, _ in qr)
        assert all(rows == N for rows, _ in gr2)

    def test_the_outcome_slot_alone_leaves_gr2_on_the_default(self) -> None:
        """``None`` resolves to ``outcome_learner``, as on a fit that sets neither slot."""
        _fit(outcome_learner=Recording("primary"), reduced_outcome_learner=Recording("qr"))
        assert all(rows < N for rows, _ in _reduced_calls("qr"))
        assert _reduced_calls("primary")
        assert all(rows == N for rows, _ in _reduced_calls("primary"))

    def test_both_slots_unset_fit_qr_and_gr2_with_the_outcome_learner(self) -> None:
        _fit(outcome_learner=Recording("primary"))
        rows = {count for count, _ in _reduced_calls("primary")}
        assert any(count < N for count in rows)
        assert N in rows


class TestTheGr2LearnerMovesTheEstimate:
    def test_two_fits_differing_only_in_the_gr2_learner_differ(self) -> None:
        """A nonzero witness: a slot that fitted nothing would leave ``psi`` unchanged."""
        linear = _fit(
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=LinearRegression(),
        )
        constant = _fit(
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=DummyRegressor(),
        )
        difference = float(linear.estimates["ate"].psi) - float(constant.estimates["ate"].psi)
        # Measured at -9.3e-4, about 0.15 standard errors.
        assert abs(difference) > 5e-4

    def test_the_bivariate_reduction_does_not_fit_the_mechanism_residual_slot(self) -> None:
        _fit(reduction="bivariate", reduced_mechanism_residual_learner=Recording("gr2"))
        assert _reduced_calls("gr2") == []


def test_the_slot_is_validated_like_the_others() -> None:
    with pytest.raises(TypeError, match="reduced_mechanism_residual_learner"):
        DRTMLE(reduced_mechanism_residual_learner=object())


class TestAClassifierInARegressionSlotIsRefused:
    """A ``None`` regression slot takes ``outcome_learner``, a classifier on a binary outcome."""

    @staticmethod
    def _binary() -> pd.DataFrame:
        frame = _frame()
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
        return frame

    def _estimator(self, **learners: Any) -> DRTMLE:
        settings: dict[str, Any] = {
            "outcome_learner": LogisticRegression(max_iter=1000),
            "treatment_learner": LogisticRegression(max_iter=1000),
            "reduced_outcome_learner": LinearRegression(),
            "reduced_treatment_learner": LogisticRegression(max_iter=1000),
            "cross_fit": False,
            "estimands": ("ate",),
            "simultaneous": False,
            **learners,
        }
        return DRTMLE(**settings)

    def test_the_unset_mechanism_residual_slot_is_refused_by_name(self) -> None:
        with pytest.raises(CapabilityError, match="reduced_mechanism_residual_learner=None"):
            self._estimator().fit(self._binary(), outcome="Y", treatment="A")

    def test_a_regressor_in_the_slot_fits(self) -> None:
        result = (
            self._estimator(reduced_mechanism_residual_learner=LinearRegression())
            .fit(self._binary(), outcome="Y", treatment="A")
            .single()
        )
        assert np.isfinite(float(result.estimates["ate"].psi))

    def test_the_bivariate_reduction_needs_no_mechanism_residual_learner(self) -> None:
        result = (
            self._estimator(reduction="bivariate")
            .fit(self._binary(), outcome="Y", treatment="A")
            .single()
        )
        assert np.isfinite(float(result.estimates["ate"].psi))
