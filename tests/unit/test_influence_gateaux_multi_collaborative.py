r"""Multi-arm collaborative curves against the finite-law Gateaux oracle.

The DR-TMLE comparisons live in the two union-model cells, so exactly one correction is
nonzero.  At the both-correct law every correction disappears and a sign error passes.
OAT is checked separately on the regular exact law where its Qbar vector identifies W.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly import CTMLE, DRTMLE
from tests import discrete_law_multi as law
from tests.discrete_law_longitudinal import CellMeans

ARMS = tuple(range(law.K))
ORACLES = ("ey[0]", "ey[1]", "ey[2]", "ate[1 vs 0]", "ate[2 vs 0]")
RATIO_ORACLES = ("rr[1 vs 0]", "rr[2 vs 0]", "or[1 vs 0]", "or[2 vs 0]")
WRONG_G = np.array(
    [
        [0.40, 0.35, 0.25],
        [0.35, 0.35, 0.30],
        [0.30, 0.30, 0.40],
    ]
)
WRONG_Q = law.Q + np.array(
    [
        [0.12, -0.18, 0.10],
        [-0.10, 0.16, -0.20],
        [0.14, -0.12, -0.15],
    ]
)
CELLS = {
    "g_right": (law.G_EXACT, WRONG_Q, ("Q",)),
    "q_right": (WRONG_G, law.Q_EXACT, ("g",)),
}


class FixedMultiTreatment(BaseEstimator):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FixedMultiTreatment:
        self.classes_ = np.arange(float(law.K))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        covariate = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        return np.asarray(self.values)[covariate]


class FixedMultiOutcome(BaseEstimator):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FixedMultiOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        indicators = design[:, : law.K - 1]
        arm = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
        covariate = np.rint(design[:, law.K - 1]).astype(int)
        return np.asarray(self.values)[covariate, arm]


class SaturatedCategorical(BaseEstimator):
    """Weighted class frequencies within each generated-regressor row."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SaturatedCategorical:
        design = np.round(np.asarray(X, dtype=float), 12)
        target = np.asarray(y, dtype=float)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        self.classes_ = np.unique(target)
        self.keys_, inverse = np.unique(design, axis=0, return_inverse=True)
        totals = np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=weights * (target == level),
                    minlength=len(self.keys_),
                )
                for level in self.classes_
            ]
        )
        self.probabilities_ = totals / totals.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.round(np.asarray(X, dtype=float), 12)
        out = np.empty((design.shape[0], len(self.classes_)))
        for row, value in enumerate(design):
            matches = np.flatnonzero(np.all(self.keys_ == value, axis=1))
            out[row] = self.probabilities_[matches[0]]
        return out


def _targeted(g_hat: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Population equation (8), one independent logistic fluctuation per arm."""
    out = np.asarray(q_hat, dtype=float).copy()
    mass = law.PROBS.sum(axis=2)
    for arm in ARMS:
        covariate = 1.0 / g_hat[:, arm]
        eta = np.log(q_hat[:, arm] / (1.0 - q_hat[:, arm]))
        epsilon = 0.0
        for _ in range(40):
            star = 1.0 / (1.0 + np.exp(-(eta + epsilon * covariate)))
            score = float(np.sum(mass[:, arm] * covariate * (law.Q_EXACT[:, arm] - star)))
            slope = float(np.sum(mass[:, arm] * covariate**2 * star * (1.0 - star)))
            epsilon += score / slope
        out[:, arm] = 1.0 / (1.0 + np.exp(-(eta + epsilon * covariate)))
    return out


def _arm_curve(
    g_hat: np.ndarray, q_star: np.ndarray, arm: int, *, sign: float = -1.0
) -> np.ndarray:
    psi = float(np.sum(law.P_W * q_star[:, arm]))
    qr = law.Q_EXACT[:, arm] - q_star[:, arm]
    gr1 = law.G_EXACT[:, arm]
    gr2 = (law.G_EXACT[:, arm] - g_hat[:, arm]) / g_hat[:, arm]
    values = []
    for w, observed_arm, outcome in law.SUPPORT:
        indicator = float(observed_arm == arm)
        plain = indicator / g_hat[w, arm] * (outcome - q_star[w, arm]) + q_star[w, arm] - psi
        d_g = qr[w] / g_hat[w, arm] * (indicator - g_hat[w, arm])
        d_q = indicator * gr2[w] / gr1[w] * (outcome - q_star[w, arm])
        values.append(plain + sign * (d_g + d_q))
    return np.asarray(values)


def _longhand(cell: str, oracle: str, *, sign: float = -1.0) -> np.ndarray:
    g_hat, q_hat, _ = CELLS[cell]
    q_star = _targeted(g_hat, q_hat)
    if oracle.startswith("ey["):
        return _arm_curve(g_hat, q_star, int(oracle[3:-1]), sign=sign)
    arm, reference = (int(value) for value in oracle[4:-1].split(" vs "))
    return _arm_curve(g_hat, q_star, arm, sign=sign) - _arm_curve(
        g_hat, q_star, reference, sign=sign
    )


def _fit(cell: str) -> Any:
    g_hat, q_hat, guard = CELLS[cell]
    estimator = DRTMLE(
        outcome_learner=FixedMultiOutcome(q_hat),
        treatment_learner=FixedMultiTreatment(g_hat),
        reduced_outcome_learner=CellMeans(),
        reduced_treatment_learner=CellMeans(),
        guard=guard,
        estimands=("ey", "ate"),
        reference=0.0,
        g_bounds=(1e-6, 1.0 - 1e-6),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(labelled=False), outcome="Y", treatment="A").single()


def _numeric_reported_name(oracle: str) -> str:
    stem, _, rest = oracle.partition("[")
    parts = [f"{float(value):.1f}" for value in rest[:-1].split(" vs ")]
    return f"{stem}[{' vs '.join(parts)}]"


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    return {cell: _fit(cell) for cell in CELLS}


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("oracle", ORACLES)
def test_multi_arm_drtmle_curve_is_the_gateaux_derivative(
    fits: dict[str, Any], cell: str, oracle: str
) -> None:
    np.testing.assert_allclose(_longhand(cell, oracle), law.eif(oracle), atol=1e-12, rtol=0)
    estimate = fits[cell].estimates[_numeric_reported_name(oracle)]
    reported = np.asarray(estimate.influence_curve)
    cells = law.cell_of_row()
    per_cell = np.array(
        [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.SUPPORT))]
    )
    np.testing.assert_allclose(per_cell, law.eif(oracle), atol=1e-11, rtol=0)


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("oracle", ORACLES)
def test_the_multi_arm_correction_sign_is_load_bearing(cell: str, oracle: str) -> None:
    wrong = _longhand(cell, oracle, sign=1.0)
    assert np.max(np.abs(wrong - law.eif(oracle))) > 1e-3


def test_each_union_cell_has_a_nonzero_correction(fits: dict[str, Any]) -> None:
    for fit in fits.values():
        reduced = fit.fluctuations["mean"].reduction.reduced
        assert max(float(np.max(np.abs(reduced.qr))), float(np.max(np.abs(reduced.gr2)))) > 1e-3


@pytest.fixture(scope="module")
def oat_fit() -> Any:
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=law.OracleMultiOutcome(),
            treatment_learner=SaturatedCategorical(),
            estimands=("ey", "ate"),
            reference="low",
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


@pytest.mark.parametrize("oracle", ORACLES)
def test_oat_regular_multi_arm_curve_is_the_gateaux_derivative(oat_fit: Any, oracle: str) -> None:
    design = np.column_stack([oat_fit.nuisance.outcome.arms[arm] for arm in oat_fit.nuisance.arms])
    assert np.unique(design, axis=0).shape[0] == 3
    estimate = oat_fit.estimates[law.reported_name(oracle)]
    reported = np.asarray(estimate.influence_curve)
    cells = law.cell_of_row()
    per_cell = np.array(
        [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.SUPPORT))]
    )
    np.testing.assert_allclose(per_cell, law.eif(oracle), atol=1e-12, rtol=0)


@pytest.fixture(scope="module")
def selector_fits() -> dict[str, Any]:
    """A fixed categorical candidate keeps the selector away from a selection tie."""
    return {
        stem: (
            CTMLE(
                strategy="discrete",
                candidates=(("W",),),
                ctmle_estimand=stem,
                outcome_learner=law.OracleMultiOutcome(),
                treatment_learner=law.OracleMultiTreatment(),
                estimands=("ey", "ate", "rr", "or"),
                reference="low",
                n_folds=2,
                learner_folds=2,
                selection_folds=2,
                selection_inner_folds=2,
                simultaneous=False,
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A")
            .single()
        )
        for stem in ("ey", "ate", "rr", "or")
    }


@pytest.mark.parametrize("stem", ["ey", "ate", "rr", "or"])
@pytest.mark.parametrize("oracle", (*ORACLES, *RATIO_ORACLES))
def test_fixed_multinomial_selector_reports_the_finite_law_gateaux_curve(
    selector_fits: dict[str, Any], stem: str, oracle: str
) -> None:
    fit = selector_fits[stem]
    estimate = fit.estimates[law.reported_name(oracle)]
    reported = np.asarray(estimate.influence_curve)
    cells = law.cell_of_row()
    per_cell = np.array(
        [reported[np.flatnonzero(cells == point)[0]] for point in range(len(law.SUPPORT))]
    )
    np.testing.assert_allclose(per_cell, law.eif(oracle), atol=1e-11, rtol=0)
