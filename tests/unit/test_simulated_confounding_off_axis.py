"""X82 part F3b: the off-axis rows of the part F plan, section 6.1.

Each row is a target and axis that the validator refused on main after F1a, or a row that
the plan lists for a witness.  On a saturated finite law, the in-sample TMLE estimate equals
the plug-in g-formula, so a cell can be checked against a hand computation on the table the
cell refits.  The surface hands that table to ``refit``, and the tests record it there.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    IncrementalEffect,
    IncrementalMean,
    MSMProjection,
    PointTreatment,
    RegimeMean,
    TMLEMethod,
)
from cleverly._summary.vocabulary import NOTE_TEXTS
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Stochastic
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.unit._confounding_support import replacement as _replacement

_IN_SAMPLE = TMLEMethod(cross_fitting=CrossFitting(enabled=False))
_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.3))
_FACTORS = {0.0: 1.0, 1.0: 0.5, 2.0: 3.0}


class CellMean(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """A saturated regression: the weighted mean of the target in each covariate cell."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMean:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.means_: dict[tuple[float, ...], float] = {}
        for key in {tuple(row) for row in X}:
            mask = (np.asarray(key) == X).all(axis=1)
            self.means_[key] = float(np.average(y[mask], weights=w[mask]))
        self.fallback_ = float(np.average(y, weights=w))
        return self

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        return np.array([self.means_.get(tuple(row), self.fallback_) for row in np.asarray(X)])


class CellFrequency(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """A saturated classifier: the weighted class frequencies in each covariate cell."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellFrequency:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.unique(y)
        self.freq_: dict[tuple[float, ...], np.ndarray[Any, Any]] = {}
        for key in {tuple(row) for row in X}:
            mask = (np.asarray(key) == X).all(axis=1)
            total = w[mask].sum()
            self.freq_[key] = np.array([w[mask & (y == c)].sum() / total for c in self.classes_])
        return self

    def predict_proba(self, X: Any) -> np.ndarray[Any, Any]:
        return np.array([self.freq_[tuple(row)] for row in np.asarray(X, dtype=float)])

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _law(*, n: int = 900, seed: int = 0, arms: int = 3) -> pd.DataFrame:
    """A finite law: W in three cells, a treatment that reads W, and a Gaussian outcome."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 3, n).astype(float)
    logits = np.column_stack([np.zeros(n)] + [0.4 * k * (w - 1.0) for k in range(1, arms)])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    a = np.array([rng.choice(arms, p=row) for row in p]).astype(float)
    y = 0.5 * w + 0.8 * a + rng.normal(size=n)
    return pd.DataFrame({"W": w, "A": a, "Y": y, "S": (w > 0).astype(float)})


def _record(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the table each cell refits, and run the real refit."""
    seen: list[Any] = []
    original = TMLE.refit

    def refit(self: Any, data: Any, **kwargs: Any) -> Any:
        seen.append(data)
        return original(self, data, **kwargs)

    monkeypatch.setattr(TMLE, "refit", refit)
    return seen


def _cells(data: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The empirical treatment law g(a | w) and outcome means Q(a, w), one row per unit."""
    w = data.covariates[:, data.covariate_names.index("W")]
    a, y = data.treatment, data.outcome
    arms = np.arange(data.n_arms, dtype=float)
    g = np.zeros((data.n, len(arms)))
    q = np.zeros((data.n, len(arms)))
    for cell in np.unique(w):
        rows = w == cell
        for k, arm in enumerate(arms):
            g[rows, k] = np.mean(a[rows] == arm)
            hit = rows & (a == arm)
            q[rows, k] = y[hit].mean()
    return w, g, q


def _incremental_mean(data: Any, factors: dict[float, float], g: Any = None) -> float:
    """The incremental g-formula mean, sum over arms of q_delta(a | W) Q(a, W), by hand."""
    _, fitted, q = _cells(data)
    g = fitted if g is None else g
    delta = np.array([factors.get(float(k), 1.0) for k in range(data.n_arms)])
    tilted = delta * g / (delta * g).sum(axis=1, keepdims=True)
    return float(np.mean((tilted * q).sum(axis=1)))


def _ipsi_fit(frame: pd.DataFrame, *, strata: bool = False) -> Any:
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "S") if strata else ("W",),
            strata=("S",) if strata else (),
        ),
    )
    target = IncrementalEffect(
        (Incremental({0.0: 1.0}, name="natural"), Incremental(_FACTORS, name="tilt")),
        reference="natural",
    )
    return study.identify(target).estimate(
        method=_IN_SAMPLE,
        outcome_learner=CellMean(),
        treatment_learner=CellFrequency(),
        simultaneous=False,
    )


# --------------------------------------------- row 1: an incremental target, multi-arm


def test_a_multi_arm_incremental_cell_equals_the_hand_g_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ipsi_fit(_law())
    alias = "ate_ipsi[tilt vs natural]"
    assert result[alias].psi == pytest.approx(
        _incremental_mean(result.data, _FACTORS) - result.data.outcome.mean(), abs=1e-9
    )
    seen = _record(monkeypatch)
    surface = simulated_confounding(
        result, estimand=alias, grid=_GRID, arm_pair=(2.0, 0.0), random_state=5
    )
    assert surface.complete
    assert len(seen) == len(surface.cells) - 1
    for cell, data in zip(surface.cells[1:], seen, strict=True):
        hand = _incremental_mean(data, _FACTORS) - data.outcome.mean()
        assert cell.estimate == pytest.approx(hand, abs=1e-9)
    moved = surface.cells[2]
    assert moved.displacement is not None and abs(moved.displacement) > 0.05
    # Mutation control: a replay that froze the original mechanism instead of refitting it
    # misses the cell.  The tilt reads g, so the frozen form is a different target.
    _, original_g, _ = _cells(result.data)
    frozen = _incremental_mean(seen[1], _FACTORS, g=original_g) - seen[1].outcome.mean()
    assert abs(frozen - moved.estimate) > 1e-3
    summary = surface.summary()
    assert summary.note("incremental_reading").text == NOTE_TEXTS["incremental_reading"]


def _swapped(data: Any, tail: np.ndarray[Any, Any], pair: tuple[float, float]) -> Any:
    """The table with the two arms of ``pair`` exchanged on the latent-tail rows."""
    a = data.treatment.copy()
    first, second = (a == pair[0]) & tail, (a == pair[1]) & tail
    a[first], a[second] = pair[1], pair[0]
    return data.with_treatment(a)


def test_the_multi_arm_incremental_swap_moves_only_the_declared_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The law is asymmetric across arms, so a swap of the wrong pair moves the g-formula.

    Each arm has its own mechanism (the logits read W with slope 0.4 k) and its own outcome
    mean (0.8 a), so the three pairs give three different tables and three hand values.
    """
    result = _ipsi_fit(_law())
    seen = _record(monkeypatch)
    surface = simulated_confounding(
        result,
        estimand="ate_ipsi[tilt vs natural]",
        grid=_GRID,
        arm_pair=(2.0, 0.0),
        random_state=5,
    )
    cell = surface.cells[2]
    assert (cell.treatment_strength, cell.outcome_strength) == (0.2, 0.0)
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    tail = latent >= NormalDist().inv_cdf(1 - cell.treatment_strength)
    # The table the cell refitted is the declared pair exchanged on the tail rows, and no
    # other arm moves.
    assert np.array_equal(seen[1].treatment, _swapped(result.data, tail, (2.0, 0.0)).treatment)
    before, after = result.data.treatment, seen[1].treatment
    changed = before != after
    assert changed.any()
    assert set(np.unique(before[changed])) <= {0.0, 2.0}
    # Mutation control: a swap of either other pair gives a different hand value.
    for wrong in ((1.0, 0.0), (2.0, 1.0)):
        table = _swapped(result.data, tail, wrong)
        hand = _incremental_mean(table, _FACTORS) - table.outcome.mean()
        assert abs(hand - cell.estimate) > 1e-3, wrong
    skipped = _incremental_mean(result.data, _FACTORS) - result.data.outcome.mean()
    assert abs(skipped - cell.estimate) > 1e-3


def test_a_multi_arm_incremental_target_needs_its_arm_pair() -> None:
    result = _ipsi_fit(_law())
    with pytest.raises(ValueError, match="pass arm_pair="):
        simulated_confounding(result, estimand="ate_ipsi[tilt vs natural]", grid=_GRID)


# ----------------------------------------- row 2: an incremental target in a stratum


def test_a_stratum_incremental_cell_equals_the_hand_g_formula_inside_the_stratum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _law(arms=2)
    frame["A"] = frame["A"].astype(float)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "S"), strata=("S",)),
    )
    result = study.identify(IncrementalMean((Incremental(3.0, name="tilt"),))).estimate(
        method=_IN_SAMPLE,
        outcome_learner=CellMean(),
        treatment_learner=CellFrequency(),
        simultaneous=False,
    )
    alias = next(name for name in result.estimates if name.endswith("[S=1.0]"))
    seen = _record(monkeypatch)
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=5)
    assert surface.stratum == (1.0,)
    for cell, data in zip(surface.cells[1:], seen, strict=True):
        w, g, q = _cells(data)
        tilted = np.column_stack((g[:, 0], 3.0 * g[:, 1]))
        tilted = tilted / tilted.sum(axis=1, keepdims=True)
        inside = w > 0
        assert cell.estimate == pytest.approx(
            float(np.mean((tilted * q).sum(axis=1)[inside])), abs=1e-9
        )
    assert max(abs(cell.displacement or 0.0) for cell in surface.cells) > 0.05


# ----------------------------------------------------- row 3: MSM coefficients, multi-arm


def _arm_design(a: Any, frame: Any) -> np.ndarray[Any, Any]:
    n = len(frame)
    return np.column_stack([np.ones(n), np.full(n, float(a == 1.0)), np.full(n, float(a == 2.0))])


def test_a_multi_arm_msm_equals_the_projection_of_the_perturbed_arm_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = CausalStudy(
        _law(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    model = MSM(design=_arm_design, terms=("intercept", "arm1", "arm2"), design_kind="known")
    result = study.identify(MSMProjection(model)).estimate(
        method=_IN_SAMPLE,
        outcome_learner=CellMean(),
        treatment_learner=CellFrequency(),
        simultaneous=False,
    )
    seen = _record(monkeypatch)
    surface = simulated_confounding(
        result, estimand="msm[arm2]", grid=_GRID, arm_pair=(2.0, 0.0), random_state=5
    )
    for cell, data in zip(surface.cells[1:], seen, strict=True):
        _, _, q = _cells(data)
        # A saturated identity-link MSM over the arms reads each arm's mean of Q(a, W).
        assert cell.estimate == pytest.approx(float(q[:, 2].mean() - q[:, 0].mean()), abs=1e-9)
    assert max(abs(cell.displacement or 0.0) for cell in surface.cells) > 0.05


# ------------------------------------------- row 4: a continuous MSM in a baseline stratum


def _dose_design(dose: Any, frame: Any) -> np.ndarray[Any, Any]:
    n = len(frame)
    values = np.full(n, float(dose)) if np.ndim(dose) == 0 else np.asarray(dose, dtype=float)
    return np.column_stack([np.ones(n), values])


def test_a_stratum_continuous_msm_cell_equals_a_complete_refit() -> None:
    rng = np.random.default_rng(3)
    n = 400
    w = rng.normal(size=n)
    s = (w > 0).astype(float)
    a = 0.6 * w + rng.normal(size=n)
    frame = pd.DataFrame({"W": w, "S": s, "A": a, "Y": w + 0.7 * a + rng.normal(size=n)})
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "S"),
            strata=("S",),
            treatment_kind="continuous",
        ),
    )
    model = MSM(
        design=_dose_design,
        terms=("intercept", "dose"),
        doses=(-1.0, 0.0, 1.0),
        design_kind="known",
    )
    result = study.identify(MSMProjection(model)).estimate(
        method=_IN_SAMPLE,
        outcome_learner=LinearRegression(),
        treatment_learner=LinearRegression(),
        simultaneous=False,
    )
    alias = next(name for name in result.estimates if name.startswith("msm[dose][S="))
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=5)
    assert surface.complete
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            _replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=5,
        )
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
    assert max(abs(cell.displacement or 0.0) for cell in surface.cells) > 1e-3


# --------------------------------- row 5: a known stochastic regime on a continuous dose


def test_a_stochastic_regime_on_a_continuous_dose_is_refused_upstream() -> None:
    """No estimator fits this target (X64), so no fit reaches the surface."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=200)
    frame = pd.DataFrame({"W": w, "A": w + rng.normal(size=200), "Y": rng.normal(size=200)})
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )
    policy = Stochastic(lambda f: np.full((len(f), 2), 0.5), name="half", density_kind="known")
    with pytest.raises(CapabilityError, match="X64"):
        study.identify(RegimeMean((policy,))).estimate(
            method=_IN_SAMPLE,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
        )
