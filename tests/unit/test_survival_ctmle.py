"""Collaborative TMLE of a risk curve (X91 part B): checks 1 to 10 of the design.

``SurvivalCTMLE`` follows Stitelman and van der Laan (2010): for each static arm of a binary
baseline treatment and each horizon, a greedy search grows a main-term logistic treatment
model :math:`g_1` and a pooled main-term logistic censoring model :math:`g_2`, targets the
arm's hazards at each candidate, and picks the path position whose cross-validated
penalized log-likelihood is smallest.

Each component that could vanish at the truth carries a nonzero witness or a mutation:
the censoring model evaluated at :math:`A = a` (check 1), the penalty's survival ratio and
its rows (check 5), the loss's last node and its clip (check 6), the censoring order
(check 7) and the weight mask of the fold solves (check 8).
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly import (
    CausalStudy,
    CollaborativeTMLEMethod,
    Inference,
    ModelSpec,
    RegimeContrast,
    TimeToEvent,
    TMLEMethod,
)
from cleverly.exceptions import CapabilityError, LongitudinalError
from cleverly.learners.crossfit import Folds, make_folds
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    Separable,
    SurvivalCTMLE,
    SurvivalSelection,
)
from cleverly.longitudinal import collaborative as module
from cleverly.longitudinal.estimator import _fit_key, _run_hazard
from cleverly.longitudinal.hazard import (
    HazardState,
    fit_hazards,
    influence,
    level_label,
    level_plans,
    solve_per_horizon,
    walk_loss,
)
from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens
from cleverly.longitudinal.sequential import Mechanism

ARMS = {"a1": 1, "a0": 0}
BOUNDS = (0.01, 1.0)
ALPHA = 0.9995


class Pick(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """A logistic regression on chosen columns of the hazard design ``[W..., A]``.

    It misspecifies the outcome model on purpose, and asserts the design's width so that a
    change of the design layout fails here instead of picking other columns.
    """

    def __init__(self, columns: tuple[int, ...] = (), width: int | None = None) -> None:
        self.columns = columns
        self.width = width

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Pick:
        matrix = np.asarray(X, dtype=float)
        assert self.width is None or matrix.shape[1] == self.width, matrix.shape
        self.model_ = LogisticRegression(max_iter=1000).fit(
            matrix[:, list(self.columns)], y, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(np.asarray(X, dtype=float)[:, list(self.columns)])


def _frame(
    n: int,
    seed: int,
    p: int,
    g: Any,
    hazard: Any,
    censoring: Any,
    n_times: int = 2,
) -> pd.DataFrame:
    """One sample: an event at node ``k`` with ``hazard``, then censoring after it."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, p))
    a = rng.binomial(1, expit(g(w)))
    time = np.full(n, float(n_times))
    event = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for k in range(1, n_times + 1):
        hit = alive & (rng.uniform(size=n) < expit(hazard(w, a, k)))
        time[hit], event[hit] = k, 1
        alive &= ~hit
        if k < n_times:
            gone = alive & (rng.uniform(size=n) < expit(censoring(w, a, k)))
            time[gone] = k
            alive &= ~gone
    frame = pd.DataFrame(w, columns=[f"W{i + 1}" for i in range(p)])
    frame["A"], frame["time"], frame["event"] = a, time, event
    return frame


def _data(frame: pd.DataFrame, **kwargs: Any) -> LongitudinalData:
    baseline = [name for name in frame.columns if name.startswith("W")]
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=baseline, **kwargs
    )


def _fit(data: LongitudinalData, learner: Any, **kwargs: Any) -> Any:
    settings: dict[str, Any] = {"selection_folds": 5, "random_state": 0}
    settings.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SurvivalCTMLE(ARMS, reference="a0", outcome_learner=learner, **settings).fit(data)


def _selector(
    data: LongitudinalData, learner: Any, *, penalty: bool = True, **kwargs: Any
) -> module._HazardSelector:
    """The selector a fit builds, built by hand so a check can reach its positions."""
    plans = resolve_plans(resolve_regimens(ARMS, 1, held_nodes=data.n_times), data)
    outer = Folds.single(data.n)
    initial = fit_hazards(data, level_plans(data), learner=learner, folds=outer)
    return module._HazardSelector(
        data,
        plans,
        initial,
        outer=outer,
        selection=make_folds(data.n, 5, random_state=0),
        bounds=BOUNDS,
        alpha=ALPHA,
        max_iter=20,
        tol=1e-10,
        penalty=penalty,
        **kwargs,
    )


def _plan(selector: module._HazardSelector, label: str) -> Any:
    return next(plan for plan in selector.plans if plan.label == label)


# ------------------------------------------------------------------------ the laws


def _instrument() -> LongitudinalData:
    """The check 2 law: ``W1`` noise, ``W2, W3`` confounders, ``W4`` a strong instrument."""
    return _data(
        _frame(
            800,
            11,
            4,
            lambda w: 0.3 * w[:, 1] - 0.3 * w[:, 2] + 2.0 * w[:, 3],
            lambda w, a, k: -0.5 + 0.8 * w[:, 1] + 0.8 * w[:, 2] - 0.7 * a,
            lambda w, a, k: -3.0 + 0.0 * a,
        )
    )


def _heavy_censoring() -> LongitudinalData:
    """The check 3 law: about a quarter censored per interval, by ``W3``, over three nodes."""
    return _data(
        _frame(
            1500,
            12,
            4,
            lambda w: 0.3 * w[:, 0],
            lambda w, a, k: -0.8 + 0.5 * w[:, 0] + 0.8 * w[:, 2] - 0.5 * a,
            lambda w, a, k: -1.2 + 1.2 * w[:, 2],
            n_times=3,
        )
    )


def _a_censoring() -> LongitudinalData:
    """The check 1 law: censoring depends on ``A`` with a nonzero coefficient."""
    return _data(
        _frame(
            600,
            13,
            2,
            lambda w: 0.4 * w[:, 0],
            lambda w, a, k: -0.6 + 0.5 * w[:, 0] - 0.5 * a,
            lambda w, a, k: -1.5 + 1.5 * a + 0.4 * w[:, 1],
            n_times=3,
        )
    )


@pytest.fixture(scope="module")
def instrument() -> LongitudinalData:
    return _instrument()


@pytest.fixture(scope="module")
def instrument_fit(instrument: LongitudinalData) -> Any:
    return _fit(instrument, Pick((4,), 5))


# ---------------------------------------------------- check 1: the full set is hazard LTMLE


def _full_terms(selector: module._HazardSelector) -> tuple[tuple[str, ...], tuple[str, ...]]:
    covariates = selector.mechanisms.covariates
    return covariates, (module.TREATMENT_TERM, *covariates)


def _reference(
    data: LongitudinalData,
    selector: module._HazardSelector,
    candidate: module.Candidate,
    label: str,
    horizon: int,
) -> Any:
    """Hazard ``LTMLE``'s driver on that arm alone, at a hand-built per-level mechanism."""
    plan = _plan(selector, label)
    level = selector.arm(plan)
    labels = [(label, level)] + [(level_label(l), l) for l in range(2)]
    mechanism = Mechanism(
        treatment=tuple(
            {name: candidate.treatment[:, at] if k == 0 else np.ones(data.n) for name, at in labels}
            for k in range(data.n_times)
        ),
        censoring=tuple(
            {name: candidate.retention[:, k, at] for name, at in labels}
            for k in range(data.n_times)
        ),
    )
    fits, _ = _run_hazard(
        data,
        [plan],
        level_plans(data),
        mechanism,
        selector.initial,
        horizons=(horizon,),
        g_bounds=BOUNDS,
        alpha=ALPHA,
        algorithm="per_horizon",
        max_iter=20,
        tol=1e-10,
        step_size=1e-3,
    )
    return fits[_fit_key(label, None, horizon, True)]


@pytest.mark.parametrize(("label", "horizon"), [("a1", 3), ("a0", 2)])
def test_check_1_the_full_set_candidate_is_hazard_ltmle_at_a_equals_a(
    label: str, horizon: int
) -> None:
    data = _a_censoring()
    learner = LogisticRegression(max_iter=1000)
    selector = _selector(data, learner)
    plan = _plan(selector, label)
    terms = _full_terms(selector)
    position = selector.position(
        plan, "full", None, terms, None, selector.start, horizon, data.weights, 0
    )
    fit = module._regimen_fit(data, plan, position, horizon)
    reference = _reference(data, selector, position.candidate, label, horizon)
    assert fit.psi_scaled == reference.psi_scaled
    np.testing.assert_array_equal(fit.influence_curve_scaled, reference.influence_curve_scaled)
    np.testing.assert_array_equal(fit.cumulative, reference.cumulative)

    # The witness: the censoring model's coefficient on A is far from zero, at nodes with
    # events, so evaluating it at the observed A moves every unit with A != a.
    eligible, censored = module.eligible_censoring(data)
    fitted = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=100)
    rows, nodes = np.nonzero(eligible[:, 1:])
    design = np.column_stack(
        [
            (nodes == 1).astype(float),
            data.treatment[rows, 0],
            data.baseline[rows],
        ]
    )
    fitted.fit(design, censored[:, 1:][rows, nodes])
    assert abs(fitted.coef_[0, 1]) > 1.0
    level = selector.arm(plan)
    k = 2
    at = np.column_stack([np.ones(data.n), np.full(data.n, float(level)), data.baseline])
    np.testing.assert_allclose(
        position.candidate.retention[:, k, level],
        1.0 - fitted.predict_proba(at)[:, 1],
        atol=1e-6,
    )

    # The mutation: the censoring model at the observed treatment misses the reference.
    mutated = _selector(
        data,
        learner,
        mechanisms=module._Mechanisms(
            data, Folds.single(data.n), BOUNDS, evaluate_at_observed=True
        ),
    )
    wrong = mutated.position(
        plan, "full", None, terms, None, mutated.start, horizon, data.weights, 0
    )
    assert module._regimen_fit(data, plan, wrong, horizon).psi_scaled != reference.psi_scaled


# --------------------------------------------------------- check 2: the g1 selection


def test_check_2_the_selection_drops_the_instrument_and_keeps_a_confounder(
    instrument: LongitudinalData, instrument_fit: Any
) -> None:
    record = instrument_fit.selection[("a1", 1)]
    g1, _ = record.selected_terms
    assert "W4" not in g1
    assert {"W2", "W3"} & set(g1)
    assert record.selected not in (0, 1)
    # The plug-in variance of the selected fit is below the full-set fit's.
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    full = selector.position(
        plan, "full", None, _full_terms(selector), None, selector.start, 1, instrument.weights, 0
    )
    full_curve = module._regimen_fit(instrument, plan, full, 1).influence_curve_scaled
    chosen = instrument_fit.fits[_fit_key("a1", None, 1, True)].influence_curve_scaled
    assert np.var(chosen) < np.var(full_curve)


def test_the_hand_built_selector_is_the_fit_s(
    instrument: LongitudinalData, instrument_fit: Any
) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    path = selector.path(plan, 1, "full", None)
    record = instrument_fit.selection[("a1", 1)]
    assert tuple(p.move for p in path) == record.moves
    np.testing.assert_array_equal([p.risk for p in path], record.train_risk)


# ------------------------------------------------------- check 3: the g2 selection


def test_check_3_heavy_censoring_selects_its_covariate_into_g2() -> None:
    data = _heavy_censoring()
    learner = Pick((0, 1, 3, 4), 5)
    result = _fit(data, learner)
    record = result.selection[("a1", 2)]
    g1, g2 = record.selected_terms
    assert "W3" in g2
    # The mutation: the selected candidate without W3 in g2 moves the estimate.
    selector = _selector(data, learner)
    plan = _plan(selector, "a1")
    kept = selector.position(plan, "full", None, (g1, g2), None, selector.start, 2, data.weights, 0)
    dropped = tuple(term for term in g2 if term != "W3")
    without = selector.position(
        plan, "full", None, (g1, dropped), None, selector.start, 2, data.weights, 0
    )
    psi = module._regimen_fit(data, plan, kept, 2).psi_scaled
    assert psi == pytest.approx(result["risk_regimen[a1 @ t=2]"].psi, abs=1e-12)
    assert abs(module._regimen_fit(data, plan, without, 2).psi_scaled - psi) > 0.005


# ------------------------------------------ check 4: the untargeted loss and no penalty


def test_check_4_scoring_the_initial_state_changes_the_selection(
    instrument: LongitudinalData, instrument_fit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = module._HazardSelector.score

    def untargeted(self: Any, plan: Any, candidate: Any, state: Any, *args: Any) -> Any:
        return score(self, plan, candidate, self.start, *args)

    monkeypatch.setattr(module._HazardSelector, "score", untargeted)
    mutated = _fit(instrument, Pick((4,), 5))
    key = ("a1", 1)
    assert mutated.selection[key].selected_terms != instrument_fit.selection[key].selected_terms


def test_check_4_without_the_penalty_the_instrument_enters(
    instrument: LongitudinalData, instrument_fit: Any
) -> None:
    bare = _fit(instrument, Pick((4,), 5), penalty=False)
    key = ("a0", 1)
    assert "W4" in bare.selection[key].selected_terms[0]
    assert "W4" not in instrument_fit.selection[key].selected_terms[0]
    assert not bare.selection[key].penalized
    np.testing.assert_array_equal(bare.selection[key].train_penalty, 0.0)


# ---------------------------------------------------------- check 5: the penalty (24)


def _penalty_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g1 = np.array([0.3, 0.6, 0.8])
    retention = np.array([[1.0, 0.9, 0.7], [1.0, 0.5, 0.8], [1.0, 0.95, 0.6]])
    inverse = 1.0 / (g1[:, None] * np.cumprod(retention, axis=1))
    hazards = np.array([[0.1, 0.2, 0.3], [0.25, 0.15, 0.4], [0.05, 0.5, 0.2]])
    survival = np.cumprod(1.0 - hazards, axis=1)
    return inverse, survival, hazards, np.array([1.0, 2.0, 0.5])


def _hand_penalty(
    inverse: np.ndarray, survival: np.ndarray, hazards: np.ndarray, w: np.ndarray, t: int
) -> float:
    total = 0.0
    for i in range(3):
        for k in range(t):
            ratio = survival[i, t - 1] / survival[i, k]
            total += w[i] * inverse[i, k] * ratio * hazards[i, k] * (1.0 - hazards[i, k])
    return total / w.sum()


def test_check_5_the_penalty_is_the_printed_24() -> None:
    inverse, survival, hazards, w = _penalty_example()
    for t in (1, 2, 3):
        assert module.source_penalty(inverse, survival, hazards, w, t) == pytest.approx(
            _hand_penalty(inverse, survival, hazards, w, t), rel=1e-14
        )
    value = module.source_penalty(inverse, survival, hazards, w, 3)
    mutations = {
        "squared ratio": _hand_penalty(inverse, survival**2, hazards, w, 3),
        "at-risk factor": _hand_penalty(inverse * survival, survival, hazards, w, 3),
        "A = a rows only": _hand_penalty(
            inverse * np.array([[1.0], [0.0], [1.0]]), survival, hazards, w, 3
        ),
    }
    for name, wrong in mutations.items():
        assert abs(wrong - value) > 1e-3, name


def _binding() -> LongitudinalData:
    """A law where ``g_bounds`` binds: a strong instrument and heavy censoring by ``W2``."""
    return _data(
        _frame(
            800,
            14,
            3,
            lambda w: 2.5 * w[:, 2],
            lambda w, a, k: -1.0 + 0.5 * w[:, 0] - 0.5 * a,
            lambda w, a, k: -0.8 + 1.2 * w[:, 1],
            n_times=3,
        )
    )


def _vector_penalty(
    inverse: np.ndarray, survival: np.ndarray, hazards: np.ndarray, w: np.ndarray, t: int
) -> float:
    """The printed (24), written over arrays, from inputs built outside the selector."""
    ratio = survival[:, t - 1 : t] / survival[:, :t]
    terms = inverse[:, :t] * ratio * hazards[:, :t] * (1.0 - hazards[:, :t])
    return float(np.sum(w[:, None] * terms) / w.sum())


def test_check_5_the_penalty_reads_the_bounded_product_and_the_initial_hazards() -> None:
    """A witness where the cumulative bound binds, so each wrong bounding moves the penalty."""
    data = _binding()
    selector = _selector(data, LogisticRegression(max_iter=1000))
    plan = _plan(selector, "a1")
    horizon = data.n_times
    position = selector.position(
        plan, "full", None, _full_terms(selector), None, selector.start, horizon, data.weights, 0
    )
    level = selector.arm(plan)
    candidate = position.candidate
    # Rebuild the cumulative mechanism from the two fitted models, not from the inputs.
    g1 = candidate.treatment[:, level]
    retention = np.cumprod(candidate.retention[:, :, level], axis=1)
    raw = g1[:, None] * retention
    lo, hi = BOUNDS
    binding = raw < lo
    assert binding.sum() >= 20, "the witness needs rows where g_bounds binds"
    # The rows bind through the product, not through either factor alone.
    assert np.any(binding & (g1[:, None] >= lo) & (retention >= lo))
    initial = (selector.start.survival[:, :, level], selector.start.hazards[:, :, level, 0])

    def penalty(inverse: np.ndarray, survival: np.ndarray, hazards: np.ndarray) -> float:
        return _vector_penalty(inverse, survival, hazards, data.weights, horizon)

    right = penalty(1.0 / np.clip(raw, lo, hi), *initial)
    assert position.penalty == pytest.approx(right, rel=1e-12)
    mutations = {
        "g1 times bound(G)": 1.0 / (g1[:, None] * np.clip(retention, lo, hi)),
        "bound(g1) times bound(G)": 1.0
        / (np.clip(g1, lo, hi)[:, None] * np.clip(retention, lo, hi)),
        "unbounded": 1.0 / raw,
    }
    for name, inverse in mutations.items():
        wrong = penalty(inverse, *initial)
        assert abs(wrong - right) > 1e-3 * right, name
    # The penalty reads the initial hazards, not the targeted ones.
    targeted = penalty(
        1.0 / np.clip(raw, lo, hi),
        position.state.survival[:, :, level],
        position.state.hazards[:, :, level, 0],
    )
    assert abs(targeted - right) > 1e-6


def test_check_5_a_cell_without_treated_rows_still_contributes() -> None:
    inverse, survival, hazards, w = _penalty_example()
    # Every row is untreated, so (24)'s A = a = 1 indicator would drop them all; the printed
    # average keeps them, and the penalty grows as g1(1 | W) falls.
    values = []
    for g1 in (0.5, 0.1, 0.01):
        scaled = inverse * (0.3 / g1)
        values.append(module.source_penalty(scaled, survival, hazards, w, 3))
    assert values[0] > 0.0
    assert values[0] < values[1] < values[2]


# ------------------------------------------------------------- check 6: the loss


def test_check_6_the_loss_stops_at_the_horizon(instrument: LongitudinalData) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    inputs = selector.inputs(plan, selector.mechanisms.candidate("full", None, (), ()))
    state = selector.start
    rows, levels = inputs.rows, inputs.observed
    lam = state.hazards[rows, :, levels][:, :, 0]
    events = inputs.events[:, :, 0]
    terms = np.where(inputs.at_risk, events * np.log(lam) + (1.0 - events) * np.log(1.0 - lam), 0.0)

    def hand(last: int) -> float:
        return -float(np.sum(inputs.weights[:, None] * terms[:, :last]))

    for t in (1, 2):
        assert walk_loss(state, inputs, last_node=t) == pytest.approx(hand(t), rel=1e-13)
        assert module.selection_loss(state, inputs, t) == (
            walk_loss(state, inputs, last_node=t),
            0,
        )
    # Off by one either way.
    assert abs(walk_loss(state, inputs, last_node=1) - hand(2)) > 1.0
    assert walk_loss(state, inputs) == pytest.approx(hand(2), rel=1e-13)


def test_check_6_the_clip_keeps_a_zero_hazard_finite(instrument: LongitudinalData) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    inputs = selector.inputs(plan, selector.mechanisms.candidate("full", None, (), ()))
    events = inputs.events[:, 0, 0] * inputs.at_risk[:, 0]
    row = int(np.flatnonzero(events)[0])
    level = int(inputs.observed[row, 0]) if inputs.observed.ndim > 1 else int(inputs.observed[row])
    logits = np.array(selector.start.logits, copy=True)
    logits[row, 0, level, 0] = -np.inf
    state = HazardState(logits, selector.start.full, selector.start.node_bounds)
    loss, clipped = module.selection_loss(state, inputs, 1)
    assert np.isfinite(loss)
    assert clipped == 1
    with np.errstate(divide="ignore"):
        assert not np.isfinite(walk_loss(state, inputs, last_node=1))


# ------------------------------------------------------------- check 7: the orders


def _wide(frame: pd.DataFrame, n_times: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The same units as a wide held panel, each censoring after node ``j`` at ``C_{j+1}``."""
    out = frame[[name for name in frame.columns if name.startswith("W")] + ["A"]].copy()
    time, event = frame["time"].to_numpy(), frame["event"].to_numpy()
    for k in range(1, n_times + 1):
        censored = (event == 0) & (time == k - 1)
        out[f"C{k}"] = np.where(time < k - 1, np.nan, np.where(censored, 0.0, 1.0))
        out[f"Y{k}"] = np.where(time >= k, ((event == 1) & (time == k)).astype(float), np.nan)
    columns = {
        "treatment": "A",
        "baseline": [name for name in frame.columns if name.startswith("W")],
        "censoring": [f"C{k}" for k in range(1, n_times + 1)],
        "outcome": [f"Y{k}" for k in range(1, n_times + 1)],
    }
    return out, columns


def test_check_7_both_orders_give_one_path() -> None:
    frame = _frame(
        500,
        21,
        2,
        lambda w: 0.4 * w[:, 0],
        lambda w, a, k: -0.7 + 0.5 * w[:, 1] - 0.4 * a,
        lambda w, a, k: -1.3 + 0.8 * w[:, 0],
        n_times=3,
    )
    long = _data(frame)
    wide, columns = _wide(frame, 3)
    wide_data = LongitudinalData.from_frame(wide, **columns)
    np.testing.assert_array_equal(long.censoring_target[:, 0], True)
    left = _fit(long, LogisticRegression(max_iter=1000))
    right = _fit(wide_data, LogisticRegression(max_iter=1000))
    for name in left.estimates:
        assert left[name].psi == right[name].psi
    for key, record in left.selection.items():
        assert record.moves == right.selection[key].moves
        np.testing.assert_array_equal(record.cv_risk, right.selection[key].cv_risk)


def _hazards(w: int, a: int) -> list[float]:
    return [0.2 + 0.1 * w - 0.05 * a, 0.3 - 0.1 * a + 0.05 * w, 0.25 + 0.1 * w * a]


def _exact_law() -> tuple[LongitudinalData, pd.DataFrame]:
    """A finite law laid out with probability weights, so the weighted sample is the law.

    ``W`` binary, ``A`` binary, main-term logistic treatment and censoring (so the
    full-set candidate is the true mechanism), censoring after nodes 1 and 2 by ``W``,
    events at three nodes from a hazard table.
    """
    rows = []
    for w in (0, 1):
        pw = 0.5
        pa1 = expit(-0.3 + 0.9 * w)
        for a in (0, 1):
            pa = pa1 if a else 1.0 - pa1
            hazard = _hazards(w, a)
            cens = [expit(-1.0 + 1.2 * w + 0.3 * a), expit(-0.6 + 1.2 * w + 0.3 * a)]
            alive = pw * pa
            for k in range(1, 4):
                rows.append((w, a, float(k), 1, alive * hazard[k - 1]))
                alive *= 1.0 - hazard[k - 1]
                if k < 3:
                    rows.append((w, a, float(k), 0, alive * cens[k - 1]))
                    alive *= 1.0 - cens[k - 1]
            rows.append((w, a, 3.0, 0, alive))
    frame = pd.DataFrame(rows, columns=["W1", "A", "time", "event", "weight"])
    frame = pd.concat([frame] * 2, ignore_index=True)
    return _data(frame, weights="weight"), frame


def _one_step(mutate: bool) -> float:
    """``Psi(Q) + P_0 D(Q, g)`` of arm 1 at horizon 3, at a wrong ``Q`` and the true ``g``.

    With the true mechanism the second-order remainder is zero, so the one-step value is the
    true risk whatever ``Q`` is; a censoring factor read at the wrong node is not.
    """
    data, _ = _exact_law()
    # The hazard model reads A alone, so only the mechanism can correct it.
    selector = _selector(data, Pick((1,), 2))
    plan = _plan(selector, "a1")
    candidate = selector.mechanisms.candidate("full", None, ("W1",), ("A", "W1"))
    if mutate:
        # G_{k-1} for G_k: drop each node's own retention from the cumulative.
        shifted = []
        for level in range(2):
            raw = candidate.treatment[:, level][:, None] * np.cumprod(
                np.concatenate([np.ones((data.n, 1)), candidate.retention[:, :-1, level]], axis=1),
                axis=1,
            )
            shifted.append(np.clip(raw, *BOUNDS))
        candidate = dataclasses.replace(candidate, cumulative=tuple(shifted))
    inputs = selector.inputs(plan, candidate)
    levels, curves = influence(selector.start, inputs, (3,))
    # The curve is already (w / E[w]) D, so its plain mean is the weighted mean P_0 D.
    return float(levels[0, 0, 0] + np.mean(curves[0, 0, 0]))


def test_check_7_the_one_step_value_is_the_truth_and_the_order_mutation_misses_it() -> None:
    truth = 0.0
    for w in (0, 1):
        survival = 1.0
        for hazard in _hazards(w, 1):
            survival *= 1.0 - hazard
        truth += 0.5 * (1.0 - survival)
    assert _one_step(False) == pytest.approx(truth, abs=1e-8)
    assert abs(_one_step(True) - truth) > 1e-3


# -------------------------------------------------------- check 8: the fold solves


def test_check_8_validation_events_never_reach_a_fold_solve(instrument: LongitudinalData) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    fit_rows, test = next(iter(selector.selection))
    train = np.zeros(instrument.n, dtype=bool)
    train[fit_rows] = True
    candidate = selector.mechanisms.candidate("cv1", train, ("W2",), ())
    weights = instrument.weights * train
    inputs = selector.inputs(plan, candidate, weights)
    rng = np.random.default_rng(0)
    order = np.arange(instrument.n)
    order[test] = rng.permutation(np.asarray(test))
    permuted = type(inputs)(
        **{
            **{name: getattr(inputs, name) for name in inputs.__dataclass_fields__},
            "events": inputs.events[order],
            "at_risk": inputs.at_risk[order],
        }
    )
    targets = ((plan.label, None, 2),)
    left, _ = solve_per_horizon(selector.start, inputs, 2, targets, max_iter=20, tol=1e-10)
    right, _ = solve_per_horizon(selector.start, permuted, 2, targets, max_iter=20, tol=1e-10)
    np.testing.assert_array_equal(left.logits, right.logits)
    held_out = instrument.weights * ~train
    loss = module.selection_loss(left, selector.inputs(plan, candidate, held_out), 2)[0]
    moved = module.selection_loss(
        left, type(inputs)(**{**permuted.__dict__, "weights": held_out}), 2
    )[0]
    assert loss != moved


def test_check_8_the_cross_validated_choice_is_not_the_training_choice(instrument_fit: Any) -> None:
    differs = [
        key
        for key, record in instrument_fit.selection.items()
        if int(np.argmin(record.train_risk)) != record.selected
    ]
    assert differs


# --------------------------------------------------------- check 9: the contrasts


def test_check_9_the_risk_difference_and_the_survival_ratio(instrument_fit: Any) -> None:
    one = instrument_fit["risk_regimen[a1 @ t=2]"]
    zero = instrument_fit["risk_regimen[a0 @ t=2]"]
    difference = instrument_fit["ate_regimen[a1 vs a0 @ t=2]"]
    assert difference.psi == pytest.approx(one.psi - zero.psi, abs=1e-15)
    np.testing.assert_allclose(
        difference.influence_curve, one.influence_curve - zero.influence_curve, atol=1e-12
    )
    ratio = instrument_fit.ratio(
        "risk_regimen[a1 @ t=2]", "risk_regimen[a0 @ t=2]", view="survival"
    )
    s1, s0 = 1.0 - one.psi, 1.0 - zero.psi
    assert ratio.psi == pytest.approx(s1 / s0, rel=1e-13)
    # The curve of (6), D1 / psi0 - psi1 D0 / psi0^2, carried on the log scale.
    d1, d0 = -one.influence_curve, -zero.influence_curve
    np.testing.assert_allclose(
        ratio.influence_curve * ratio.psi, d1 / s0 - s1 * d0 / s0**2, rtol=1e-10, atol=1e-12
    )
    assert ratio.inference == "working_mechanism_plugin"


# ------------------------------------- check 10: the curve, the refusals, determinism


def test_check_10_the_plugin_curve_is_the_shipped_hazard_curve(
    instrument: LongitudinalData, instrument_fit: Any
) -> None:
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a0")
    record = instrument_fit.selection[("a0", 2)]
    position = selector.path(plan, 2, "full", None)[record.selected]
    _, curves = influence(position.state, position.inputs, (2,))
    estimate = instrument_fit["risk_regimen[a0 @ t=2]"]
    np.testing.assert_allclose(estimate.influence_curve, curves[0, 0, 0], rtol=0, atol=1e-12)
    assert estimate.inference == "working_mechanism_plugin"
    with pytest.raises(CapabilityError):
        _ = estimate.std_error


def test_check_10_the_status_record_and_summary(instrument_fit: Any) -> None:
    assert instrument_fit.inference_status == "working_mechanism_plugin"
    assert instrument_fit.simultaneous is None
    text = str(instrument_fit.summary())
    assert "survival C-TMLE" in text
    assert "not forced to rise" in text
    record = instrument_fit.selection[("a1", 2)]
    assert isinstance(record, SurvivalSelection)
    assert record.penalty_form == "printed_24"
    assert record.combination == module.COMBINATION
    assert record.diagnostic_horizon == 2
    rows = record.to_frame()
    assert len(rows["move"]) == len(record.moves)
    assert len(record.to_frame(instrument_fit.data)) == len(record.moves)
    assert "moves" in str(record.summary())
    # The assembled mechanism holds ones at every later treatment node, under every label.
    mechanism = instrument_fit.mechanism
    for node in mechanism.treatment[1:]:
        for values in node.values():
            np.testing.assert_array_equal(values, 1.0)
    assert {"a1", "a0", level_label(0), level_label(1)} <= set(mechanism.treatment[0])


def test_check_10_the_fit_is_deterministic_and_round_trips(instrument: LongitudinalData) -> None:
    one = _fit(instrument, Pick((4,), 5), n_jobs=1)
    two = _fit(instrument, Pick((4,), 5), n_jobs=2)
    for name in one.estimates:
        assert one[name].psi == two[name].psi
    for key, record in one.selection.items():
        assert record.moves == two.selection[key].moves
    loaded = pickle.loads(pickle.dumps(one))
    assert loaded.selection[("a1", 1)].moves == one.selection[("a1", 1)].moves
    from cleverly.estimators.serialize import dumps, loads

    restored = loads(dumps(one))
    assert restored.selection[("a1", 1)].selected == one.selection[("a1", 1)].selected


_X, _Y, _A = ("g1", "x"), ("g1", "y"), ("g2", "A")


def _scripted_table() -> dict[Any, float]:
    return {
        ((), None, 0): 10.0,
        # Pass 1: x and A tie below 10, and the tie goes to the try order, so x.
        ((), _X, 0): 8.0,
        ((), _Y, 0): 9.0,
        ((), _A, 0): 8.0,
        # Pass 2: A only equals the current 8.0, so the search restarts ...
        ((_X,), _Y, 0): 8.5,
        ((_X,), _A, 0): 8.0,
        # ... and forces the restarted pass's best move, A, although it is worse.
        ((_X,), _Y, 1): 9.5,
        ((_X,), _A, 1): 9.0,
        # y does not improve on 9.0 either: a second restart forces it.
        ((_X, _A), _Y, 1): 9.2,
        ((_X, _A), _Y, 2): 9.7,
    }


_SCRIPTED_PATH = [(None, 0), (_X, 0), (_A, 1), (_Y, 2)]


def test_check_10_the_greedy_rule_on_a_scripted_table() -> None:
    """A tie, a restart and a forced addition, on a scripted risk table."""
    table = _scripted_table()
    path = module.greedy_rule(((_X, _Y), (_A,)), lambda *key: table[key])
    assert path == _SCRIPTED_PATH
    # A strict decrease is accepted with no restart.
    table[((_X,), _A, 0)] = 7.9
    table[((_X, _A), _Y, 0)] = 7.0
    assert module.greedy_rule(((_X, _Y), (_A,)), lambda *key: table[key]) == [
        (None, 0),
        (_X, 0),
        (_A, 0),
        (_Y, 0),
    ]
    # A non-finite risk never beats a finite one.
    table = _scripted_table()
    table[((), _X, 0)] = float("nan")
    assert module.greedy_rule(((_X, _Y), (_A,)), lambda *key: table.get(key, 50.0))[1] == (_A, 0)


def _mutant(old: str, new: str) -> Any:
    """``greedy_rule`` with one edit to its source, compiled in the module's namespace."""
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(module.greedy_rule))
    assert source.count(old) == 1, old
    namespace = dict(vars(module))
    exec(compile(source.replace(old, new), module.__file__, "exec"), namespace)
    return namespace["greedy_rule"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # A tie accepts the move instead of restarting.
        (
            "if not risks[best] < current and not stepped:",
            "if not risks[best] <= current and not stepped:",
        ),
        # A pass that still finds no improvement stops instead of forcing its best move.
        (
            "        stepped = False\n        move = remaining.pop(best)",
            "        if not risks[best] < current:\n            break\n"
            "        stepped = False\n        move = remaining.pop(best)",
        ),
    ],
    ids=["ties-accepted", "no-forced-addition"],
)
def test_check_10_a_mutated_rule_fails_the_scripted_table(old: str, new: str) -> None:
    rule = _mutant(old, new)
    table = _scripted_table()
    assert rule(((_X, _Y), (_A,)), lambda *key: table.get(key, 50.0)) != _SCRIPTED_PATH
    # The unmutated source compiled the same way passes, so the edit is what fails.
    same = _mutant(old, old)
    assert same(((_X, _Y), (_A,)), lambda *key: table[key]) == _SCRIPTED_PATH


def test_check_10_the_fit_path_is_the_greedy_rule(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selector's path is the rule's: a spy sees every call, and a mutant changes it."""
    selector = _selector(instrument, Pick((4,), 5))
    plan = _plan(selector, "a1")
    calls: list[list[Any]] = []
    rule = module.greedy_rule
    no_forced_addition = _mutant(
        "        stepped = False\n        move = remaining.pop(best)",
        "        if not risks[best] < current:\n            break\n"
        "        stepped = False\n        move = remaining.pop(best)",
    )

    def spy(pools: Any, score: Any) -> Any:
        path = rule(pools, score)
        calls.append(path)
        return path

    monkeypatch.setattr(module, "greedy_rule", spy)
    path = selector.path(plan, 2, "full", None)
    assert len(calls) == 1
    assert [(p.move, p.n_steps) for p in path] == calls[0]
    assert any(p.n_steps > 0 for p in path), "the law needs a restart to witness the rule"
    # The forced-addition mutant stops at the first pass with no improvement.
    monkeypatch.setattr(module, "greedy_rule", no_forced_addition)
    selector = _selector(instrument, Pick((4,), 5))
    assert len(selector.path(plan, 2, "full", None)) < len(path)


def test_a_non_finite_risk_is_never_selected(
    instrument: LongitudinalData, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert module._finite_argmin([np.nan, 2.0, 1.0, np.inf, 1.0]) == 2
    assert module._finite_argmin([np.nan, np.inf]) == -1
    selector = _selector(instrument, Pick((4,), 5))
    real = selector.cross_validate

    def poisoned(plan: Any, horizon: int) -> Any:
        risk, held = real(plan, horizon)
        risk = risk.copy()
        risk[int(np.argmin(risk))] = np.nan  # the old argmin would have picked this
        risk[0] = -np.inf
        return risk, held

    monkeypatch.setattr(selector, "cross_validate", poisoned)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, records, _ = module.run_selection(selector, [2])
    for record in records.values():
        assert np.isfinite(record.cv_risk[record.selected])
        assert record.selected != 0
    monkeypatch.setattr(selector, "cross_validate", lambda plan, horizon: (np.full(10, np.nan), ()))
    with pytest.raises(LongitudinalError, match="not finite"):
        module.run_selection(selector, [2])


def test_a_non_finite_newton_step_keeps_the_last_finite_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(0)
    design = rng.normal(size=(200, 2))
    target = rng.binomial(1, expit(0.5 * design[:, 0])).astype(float)
    weights = np.ones(200)
    plain = module._logistic(design, target, weights)
    assert plain.converged
    solve = np.linalg.solve

    def scripted(first: Any) -> Any:
        calls = iter(range(10**6))

        def fake(information: Any, score: Any) -> Any:
            move = solve(information, score)
            return first(move) if next(calls) == 0 else move

        return fake

    # A step that is not finite ends the fit at the last finite iterate.
    monkeypatch.setattr(np.linalg, "solve", scripted(lambda move: move * np.nan))
    fit = module._logistic(design, target, weights)
    assert not fit.converged
    assert np.array_equal(fit.coef, np.zeros(3))
    # A step whose deviance overflows is halved until it is finite, and the fit goes on.
    monkeypatch.setattr(np.linalg, "solve", scripted(lambda move: move + 1e308))
    fit = module._logistic(design, target, weights)
    assert np.all(np.isfinite(fit.coef))
    x = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])
    assert module._deviance(x, np.array([0.0, 1e308]), np.zeros(3), np.ones(3)) == np.inf


# -------------------------------------------------------------------- refusals


def _survival_frame() -> pd.DataFrame:
    return _frame(
        200,
        3,
        2,
        lambda w: 0.3 * w[:, 0],
        lambda w, a, k: -0.5 + 0.4 * w[:, 1] - 0.3 * a,
        lambda w, a, k: -1.5 + 0.0 * a,
        n_times=2,
    )


class _Refuse(LogisticRegression):  # type: ignore[misc]
    """A learner that fails if any fit reaches it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")


SETTINGS = [
    ({"survival_model": "iterated_mean"}, module.SURVIVAL_MODEL_REFUSAL),
    ({"targeting": "one_step"}, module.ONE_STEP_REFUSAL),
    ({"strategy": "ordered"}, module.STRATEGY_REFUSAL),
    ({"patience": 2}, None),
    ({"candidates": (("W1",),)}, None),
    ({"ordering": ("W1",)}, None),
    ({"preorder": "logistic"}, None),
    ({"lambdas": (1.0, 0.0)}, None),
    ({"oat_design": "per_arm"}, None),
    ({"selection_inner_folds": 3}, module.INNER_FOLDS_REFUSAL),
    ({"selection_estimand": "ate"}, module.ESTIMAND_REFUSAL),
    ({"loss": "squared"}, module.LOSS_REFUSAL),
    ({"simultaneous": True}, module.BAND_REFUSAL),
    ({"n_multiplier": 500}, module.BAND_REFUSAL),
    ({"repeats": 2}, module.BOOTSTRAP_REFUSAL),
    ({"n_bootstrap": 10}, module.BOOTSTRAP_REFUSAL),
    ({"stopping": "absolute"}, module.RULES_REFUSAL),
    ({"iterate": "best"}, module.RULES_REFUSAL),
    ({"q_bounds": (0.0, 1.0)}, module.Q_BOUNDS_REFUSAL),
    ({"msm": "~ 1"}, module.PROJECTION_REFUSAL),
    ({"ratio_trim": 0.1}, module.PROJECTION_REFUSAL),
    ({"isotonic": True}, module.PROJECTION_REFUSAL),
    ({"treatment_learner": LogisticRegression()}, module.LEARNERS_REFUSAL),
    ({"censoring_learner": LogisticRegression()}, module.LEARNERS_REFUSAL),
    ({"density_bins": 5}, module.DOSE_SETTINGS_REFUSAL),
]


@pytest.mark.parametrize(("settings", "refusal"), SETTINGS)
def test_every_refused_setting_raises_before_a_learner(
    settings: dict[str, Any], refusal: Any
) -> None:
    data = _data(_survival_frame())
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(ARMS, outcome_learner=_Refuse(), **settings).fit(data)
    if refusal is not None:
        assert caught.value.refusal.reason == refusal.reason
        assert refusal.condition in str(caught.value)


def test_repeated_splits_are_refused_before_a_learner() -> None:
    """The repeat witness of ``result.selection``: no fit with ``repeats > 1`` exists."""
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(ARMS, outcome_learner=_Refuse(), repeats=3).fit(_data(_survival_frame()))
    assert caught.value.refusal is module.BOOTSTRAP_REFUSAL


def _design_cases() -> list[tuple[str, Any, Any, Any]]:
    frame = _survival_frame()
    two_causes = frame.copy()
    two_causes.loc[two_causes.index[::7], "event"] = 2
    three = frame.copy()
    three.loc[three.index[::5], "A"] = 2
    return [
        ("causes", lambda: _data(two_causes), ARMS, module.CAUSES_REFUSAL),
        (
            "eliminate",
            lambda: _data(two_causes, causes={1: "relapse", 2: "death"}, eliminate=["death"]),
            ARMS,
            module.ELIMINATE_REFUSAL,
        ),
        ("arms", lambda: _data(three), {"a1": 1, "a0": 0}, module.ARMS_REFUSAL),
        (
            "rule",
            lambda: _data(frame),
            {
                "rule": DynamicRegimen(
                    "rule", ((lambda history: (history["W1"] > 0).astype(int)),), rule_kind="known"
                ),
                "a0": 0,
            },
            module.ARMS_REFUSAL,
        ),
        (
            "known",
            lambda: _data(frame.assign(p=0.5, q=0.5), treatment_probabilities={0: "q", 1: "p"}),
            ARMS,
            module.KNOWN_REFUSAL,
        ),
        (
            "cluster",
            lambda: _data(frame.assign(id=np.arange(len(frame)) // 2), id="id"),
            ARMS,
            module.CLUSTER_REFUSAL,
        ),
    ]


@pytest.mark.parametrize(
    ("name", "build", "regimens", "refusal"), _design_cases(), ids=[c[0] for c in _design_cases()]
)
def test_every_refused_design_raises_before_a_learner(
    name: str, build: Any, regimens: Any, refusal: Any
) -> None:
    try:
        data = build()
    except Exception as error:  # a container that refuses first states its own reason
        pytest.skip(f"the container refuses {name} itself: {error}")
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(regimens, outcome_learner=_Refuse()).fit(data)
    assert caught.value.refusal.reason == refusal.reason
    assert refusal.condition in str(caught.value)


def test_a_separable_regimen_is_refused() -> None:
    frame = _survival_frame()
    frame.loc[frame.index[::7], "event"] = 2
    data = _data(frame, causes={1: "relapse", 2: "death"})
    regimens = {"split": Separable(arms={"death": 0, "relapse": 1}, first="death"), "a0": 0}
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(regimens, outcome_learner=_Refuse()).fit(data)
    assert caught.value.refusal is module.SEPARABLE_REFUSAL


def test_a_sequential_design_is_refused() -> None:
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=200, seed=0)
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE({"always": 1, "never": 0}, outcome_learner=_Refuse()).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    assert caught.value.refusal is module.SEQUENTIAL_REFUSAL


def test_a_node_every_row_leaves_at_is_refused() -> None:
    frame = _survival_frame()
    # Every unit event-free after node 1 is censored there, so node 2 holds no one.
    frame.loc[frame["time"] == 2, ["time", "event"]] = (1.0, 0)
    data = _data(frame, grid=(1.0, 2.0))
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(ARMS, outcome_learner=_Refuse()).fit(data)
    assert caught.value.refusal.reason == module.Q_BOUNDS_REFUSAL.reason
    assert "at node 2 (time 2)" in str(caught.value)
    # A horizon before the node fits.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SurvivalCTMLE(
            ARMS, horizons=[1], outcome_learner=LogisticRegression(max_iter=1000)
        ).fit(data)
    assert sorted(result.selection) == [("a0", 1), ("a1", 1)]


def test_the_truncation_curve_and_simulated_confounding_are_refused(instrument_fit: Any) -> None:
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve
    from cleverly.sensitivity._simulated_confounding_longitudinal import longitudinal_fit_refusal

    with pytest.raises(CapabilityError) as caught:
        longitudinal_truncation_curve(instrument_fit, [(0.05, 1.0)])
    assert module.TRUNCATION_REFUSAL.condition in str(caught.value)
    assert longitudinal_fit_refusal(instrument_fit) is module.SIMULATED_CONFOUNDING_REFUSAL


# --------------------------------------------------------------- the study layer


def test_the_study_layer_fits_a_time_to_event_design_by_collaborative_tmle() -> None:
    frame = _survival_frame()
    study = CausalStudy(
        frame,
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )
    effect = study.identify(RegimeContrast(regimens=ARMS, reference="a0"))
    record = next(r for r in effect.available_methods() if r.name == "collaborative_tmle")
    assert record.available
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(
            method=CollaborativeTMLEMethod(selection_folds=3),
            outcome_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    assert result.selection is not None
    assert result.inference_status == "working_mechanism_plugin"
    assert type(result.estimator) is SurvivalCTMLE


def test_the_study_layer_names_the_refusal_of_a_sequential_design() -> None:
    from cleverly import LongitudinalTreatment
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=200, seed=0)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
    )
    effect = study.identify(RegimeContrast(regimens={"always": 1, "never": 0}, reference="never"))
    record = next(r for r in effect.available_methods() if r.name == "collaborative_tmle")
    assert not record.available
    assert record.refusal is not None and record.refusal.roadmap == "F5"


def test_hazard_ltmle_is_unchanged_by_the_hook() -> None:
    """The base estimator still runs its own block: no selection, its usual status."""
    data = _data(_survival_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            ARMS,
            survival_model="hazard",
            n_folds=1,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            simultaneous=False,
        ).fit(data)
    assert result.selection is None
    assert result.inference_status != "working_mechanism_plugin"


def test_the_method_translation_resolves_the_defaults_the_engine_fixes() -> None:
    kwargs = CollaborativeTMLEMethod(selection_folds=3).estimator_kwargs(longitudinal=True)
    assert kwargs["survival_model"] == "hazard"
    assert kwargs["simultaneous"] is False
    assert kwargs["selection_estimand"] is None
    assert kwargs["selection_folds"] == 3
    explicit = CollaborativeTMLEMethod(selection_estimand="ey1").estimator_kwargs(longitudinal=True)
    assert explicit["selection_estimand"] == "ey1"
    # The unset defaults resolve per engine: the same method object fits a band and the
    # iterated-mean model under every other engine.
    assert CollaborativeTMLEMethod().estimator_kwargs()["simultaneous"] is True
    assert TMLEMethod().estimator_kwargs(longitudinal=True)["survival_model"] == "iterated_mean"
    assert TMLEMethod().estimator_kwargs(longitudinal=True)["simultaneous"] is True


@pytest.mark.parametrize(
    ("method", "refusal"),
    [
        (
            CollaborativeTMLEMethod(models=ModelSpec(survival_model="iterated_mean")),
            module.SURVIVAL_MODEL_REFUSAL,
        ),
        (CollaborativeTMLEMethod(inference=Inference(simultaneous=True)), module.BAND_REFUSAL),
    ],
    ids=["iterated_mean", "simultaneous"],
)
def test_an_explicit_value_the_engine_cannot_fit_is_refused_not_remapped(
    method: CollaborativeTMLEMethod, refusal: Any
) -> None:
    kwargs = method.estimator_kwargs(longitudinal=True)
    assert kwargs["survival_model"] == (method.models.survival_model or "hazard")
    with pytest.raises(CapabilityError) as caught:
        SurvivalCTMLE(ARMS, **{**kwargs, "outcome_learner": _Refuse()}).fit(
            _data(_survival_frame())
        )
    assert caught.value.refusal is refusal
