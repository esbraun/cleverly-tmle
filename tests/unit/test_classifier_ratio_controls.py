"""Controls of the classifier ratio route of Díaz et al. (2023), Section 5.4 (X40 part (c)).

The route stacks the observed rows (label 0) and the rows each randomizer value assigns
(label 1, weighted by the value's probability), and reads the ratio as the odds of the label.
These tests check four properties of the shipped construction in
``cleverly.learners.density_ratio.classifier_ratio``:

- T-c1: the outer folds split units, so no unit's copy trains the model that predicts it.
- T-c2: the inner folds are grouped by unit or cluster, through ``groups=`` of ``fit_learner``.
- T-c3: the copy weights are the observation weight times the randomizer probability.
- T-c4: the ratio is normalized, ``sum_a g(a | h) r(a, h) = 1`` for a support-preserving
  policy, on both routes.

M-c1 to M-c5 are deliberate mutations, by monkeypatch or by a mutant copy of the fold loop,
that each witness must catch.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.interventions import ModifiedPolicy, Piece, Randomizer, Shift
from cleverly.interventions.policy import lazy_frame, policy_branches
from cleverly.learners import density_ratio
from cleverly.learners.crossfit import Folds, random_partition

from ..discrete_law_longitudinal import CellMeans

CAUGHT = 1e-4
INF = np.inf

#: ``g(a | w)`` in eighths on the doses ``{0, 1, 2}``, indexed ``[w, a]``.
G = np.array([[1.0, 3.0, 4.0], [4.0, 2.0, 2.0]]) / 8.0
#: Rows per covariate value, so every cell count is an integer.
PER_W = 64


def _law() -> tuple[np.ndarray, np.ndarray]:
    """``(w, a)`` for ``2 * PER_W`` rows that realise ``G`` exactly."""
    w, a = [], []
    for value in (0, 1):
        for dose in range(3):
            count = round(G[value, dose] * PER_W)
            w.extend([value] * count)
            a.extend([dose] * count)
    return np.asarray(w, dtype=float), np.asarray(a, dtype=float)


UP = (
    Piece(-INF, 1.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
    Piece(1.5, INF),
)
#: Up one with probability 0.25, held at the top dose; otherwise stay.  Support-preserving.
RANDOM = ModifiedPolicy(
    "random",
    pieces={"up": UP, "stay": (Piece(-INF, INF),)},
    randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
    policy_kind="known",
)


def _assigned(policy: Any, w: np.ndarray, a: np.ndarray) -> list[tuple[float, np.ndarray]]:
    import pandas as pd

    frame = lazy_frame(lambda: pd.DataFrame({"W": w}))
    return [
        (float(p), np.asarray(branch.assign(a, frame), dtype=float))
        for p, branch in policy_branches(policy)
    ]


def _random_truth(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``g^d(a | w) / g(a | w)`` under :data:`RANDOM`."""
    g = G[w.astype(int)]
    k = a.astype(int)
    rows = np.arange(a.size)
    up = np.where(k >= 1, g[rows, np.maximum(k - 1, 0)], 0.0) + np.where(k == 2, g[:, 2], 0.0)
    return (0.75 * g[rows, k] + 0.25 * up) / g[rows, k]


def _ratio(
    w: np.ndarray,
    a: np.ndarray,
    policy: Any,
    *,
    weights: np.ndarray | None = None,
    learner: Any = None,
    evaluate_at: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    return density_ratio.classifier_ratio(
        CellMeans() if learner is None else learner,
        w.reshape(-1, 1),
        a,
        _assigned(policy, w, a),
        np.ones(a.size) if weights is None else weights,
        Folds.single(a.size),
        evaluate_at=[a] if evaluate_at is None else evaluate_at,
    )


# ------------------------------------------------------------------ T-c1, M-c1


class _Recorder:
    """Instruments ``fit_learner`` and ``predict_mean`` of the ratio module.

    The last covariate column is the unit's id, so a design row names its unit.  Each fitted
    model remembers the ids it trained on, and each prediction records the ids it read.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, drop_groups: bool = False) -> None:
        self.trained: dict[int, set[float]] = {}
        self.models: list[Any] = []  # alive, so no id is reused
        self.leaks = 0
        self.groups: list[tuple[np.ndarray, Any]] = []
        fit, predict = density_ratio.fit_learner, density_ratio.predict_mean

        def recorded_fit(learner: Any, x: Any, y: Any, weight: Any = None, **kwargs: Any) -> Any:
            if drop_groups:
                kwargs["groups"] = None
            self.groups.append((np.asarray(x)[:, -1].copy(), kwargs.get("groups")))
            model = fit(learner, x, y, weight, **kwargs)
            self.trained[id(model)] = set(np.asarray(x)[:, -1].tolist())
            self.models.append(model)
            return model

        def recorded_predict(model: Any, x: Any, task: Any) -> Any:
            seen = self.trained[id(model)]
            self.leaks += sum(unit in seen for unit in np.asarray(x)[:, -1].tolist())
            return predict(model, x, task)

        monkeypatch.setattr(density_ratio, "fit_learner", recorded_fit)
        monkeypatch.setattr(density_ratio, "predict_mean", recorded_predict)


def _sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a = rng.normal(loc=w, scale=1.0)
    return w, a, np.arange(n, dtype=float)


def _shipped(
    w: np.ndarray, a: np.ndarray, ids: np.ndarray, folds: Folds, cluster: Any = None
) -> None:
    density_ratio.classifier_ratio(
        LogisticRegression(),
        np.column_stack([w, ids]),
        a,
        [(1.0, a + 0.5)],
        np.ones(a.size),
        folds,
        evaluate_at=[a],
        groups=cluster,
    )


def _stacked_row_folds(
    w: np.ndarray, a: np.ndarray, ids: np.ndarray, folds: Folds, cluster: Any = None
) -> None:
    """M-c1: the shipped stack, with the outer folds drawn over stacked rows, not units."""
    h = np.column_stack([w, ids])
    n = a.size
    design = np.vstack([np.column_stack([a, h]), np.column_stack([a + 0.5, h])])
    label = np.concatenate([np.zeros(n), np.ones(n)])
    rows = random_partition(2 * n, folds.n_folds, seed=5)
    for train, test in rows:
        model = density_ratio.fit_learner(
            LogisticRegression(), design[train], label[train], np.ones(train.size)
        )
        observed = test[test < n]
        density_ratio.predict_mean(model, design[observed], "classification")


@pytest.mark.parametrize("n_folds", [3, 5])
@pytest.mark.parametrize("clustered", [False, True])
def test_t_c1_no_unit_trains_the_model_that_predicts_it(
    monkeypatch: pytest.MonkeyPatch, n_folds: int, clustered: bool
) -> None:
    w, a, ids = _sample(300, 1)
    cluster = (np.arange(a.size) // 3).astype(np.int64) if clustered else None
    folds = random_partition(a.size, n_folds, cluster=cluster, seed=2)
    recorder = _Recorder(monkeypatch)
    _shipped(w, a, ids, folds, cluster)
    assert len(recorder.trained) == n_folds
    assert recorder.leaks == 0


@pytest.mark.parametrize("n_folds", [3, 5])
def test_m_c1_folds_over_stacked_rows_leak(monkeypatch: pytest.MonkeyPatch, n_folds: int) -> None:
    w, a, ids = _sample(300, 1)
    recorder = _Recorder(monkeypatch)
    _stacked_row_folds(w, a, ids, random_partition(a.size, n_folds, seed=2))
    assert recorder.leaks > 0


# ------------------------------------------------------------------ T-c2, M-c2


def _grouped_by_unit(recorder: _Recorder, units: np.ndarray) -> bool:
    for ids, groups in recorder.groups:
        if groups is None:
            return False
        if not np.array_equal(np.asarray(groups), units[ids.astype(int)]):
            return False
    return True


@pytest.mark.parametrize("clustered", [False, True])
def test_t_c2_the_inner_folds_are_grouped_by_unit(
    monkeypatch: pytest.MonkeyPatch, clustered: bool
) -> None:
    w, a, ids = _sample(300, 3)
    cluster = (np.arange(a.size) // 3).astype(np.int64) if clustered else None
    units = np.arange(a.size) if cluster is None else cluster
    recorder = _Recorder(monkeypatch)
    _shipped(w, a, ids, random_partition(a.size, 3, cluster=cluster, seed=4), cluster)
    assert recorder.groups
    assert _grouped_by_unit(recorder, units)


def test_m_c2_a_call_without_groups_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation drops ``groups=`` at the ``fit_learner`` call itself (review n1)."""
    w, a, ids = _sample(300, 3)
    recorder = _Recorder(monkeypatch, drop_groups=True)
    _shipped(w, a, ids, random_partition(a.size, 3, seed=4))
    assert not _grouped_by_unit(recorder, np.arange(a.size))


# ------------------------------------------------------------------ T-c3, M-c3, M-c4


def test_t_c3_integer_weights_equal_the_expanded_rows() -> None:
    w, a, _ = _sample(400, 5)
    weights = 1.0 + (np.arange(a.size) % 3 == 0)
    learner = LogisticRegression(C=1.0, tol=1e-12, max_iter=10_000)
    shift = Shift(0.5, cap=None)
    by_weight = _ratio(w, a, shift, weights=weights, learner=learner)[0]
    copies = np.flatnonzero(weights == 2.0)
    w2, a2 = np.concatenate([w, w[copies]]), np.concatenate([a, a[copies]])
    expanded = _ratio(w2, a2, shift, learner=learner)[0][: a.size]
    np.testing.assert_allclose(by_weight, expanded, rtol=1e-6, atol=1e-8)


def _weights_by_w(w: np.ndarray) -> np.ndarray:
    """Weights that depend on ``W`` only, so the weighted law has the same ratio."""
    return np.where(w == 1.0, 2.0, 1.0)


def test_t_c3_a_randomizer_with_a_saturated_classifier_is_exact() -> None:
    w, a = _law()
    for weights in (None, _weights_by_w(w)):
        ratio = _ratio(w, a, RANDOM, weights=weights)[0]
        np.testing.assert_allclose(ratio, _random_truth(w, a), rtol=0, atol=1e-12)
    assert np.max(np.abs(_random_truth(w, a) - 1.0)) > 0.1


def _reweight(monkeypatch: pytest.MonkeyPatch, rule: str) -> None:
    """Rewrite the label-1 weights the shipped stack passes to ``fit_learner``."""
    fit = density_ratio.fit_learner

    def mutated(learner: Any, x: Any, y: Any, weight: Any = None, **kwargs: Any) -> Any:
        label = np.asarray(y)
        weight = np.asarray(weight, dtype=float).copy()
        observed = weight[label == 0]
        blocks = int(np.sum(label == 1) // observed.size)
        assigned = weight[label == 1].reshape(blocks, observed.size)
        if rule == "unit":  # M-c3: every label-1 block weighted as the observed row
            assigned = np.tile(observed, (blocks, 1))
        elif rule == "drop":  # M-c4: the observation weight dropped from the label-1 block
            assigned = assigned / observed
        else:  # M-c5: the label-1 blocks carry total weight 2 per unit
            assigned = 2.0 * assigned
        weight[label == 1] = assigned.reshape(-1)
        return fit(learner, x, y, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", mutated)


@pytest.mark.parametrize("rule", ["unit", "drop"], ids=["M-c3", "M-c4"])
def test_m_c3_m_c4_the_copy_weight_mutations_move_the_ratio(
    monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    w, a = _law()
    weights = _weights_by_w(w)
    _reweight(monkeypatch, rule)
    ratio = _ratio(w, a, RANDOM, weights=weights)[0]
    assert np.max(np.abs(ratio - _random_truth(w, a))) > CAUGHT


# ------------------------------------------------------------------ T-c4, M-c5

#: Up one with no cap: the dose 2 leaves the support.
LEAVING = Shift(1.0, cap=None)


def _classifier_normalization(policy: Any) -> np.ndarray:
    w, a = _law()
    doses = [np.full(a.size, float(dose)) for dose in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = _ratio(w, a, policy, evaluate_at=doses)
    g = G[w.astype(int)]
    return sum(g[:, dose] * ratios[dose] for dose in range(3))


def _density_normalization(policy: Any) -> np.ndarray:
    import pandas as pd

    w, _ = _law()
    g = G[w.astype(int)]
    frame = lazy_frame(lambda: pd.DataFrame({"W": w}))
    rows = np.arange(w.size)

    def density(values: Any) -> np.ndarray:
        b = np.asarray(values, dtype=float).reshape(-1)
        index = np.rint(b).astype(int)
        on = (np.abs(b - index) < 1e-12) & (index >= 0) & (index <= 2)
        return np.where(on, g[rows, np.clip(index, 0, 2)], 0.0)

    total = np.zeros(w.size)
    for p, branch in policy_branches(policy):
        for dose in range(3):
            values = np.full(w.size, float(dose))
            total += p * g[:, dose] * branch.ratio(values, frame, density)
    return total


@pytest.mark.parametrize("route", ["classifier", "density"])
def test_t_c4_the_ratio_is_normalized(route: str) -> None:
    normalization = _classifier_normalization if route == "classifier" else _density_normalization
    np.testing.assert_allclose(normalization(RANDOM), 1.0, rtol=0, atol=1e-12)
    w, _ = _law()
    # A policy that leaves the support reads P(d(A, H) in supp) = 1 - g(2 | h) < 1.
    leaving = normalization(LEAVING)
    np.testing.assert_allclose(leaving, 1.0 - G[w.astype(int), 2], rtol=0, atol=1e-12)
    assert np.max(1.0 - leaving) > 0.05


def test_t_c4_the_fold_mean_ratio_is_reported() -> None:
    from cleverly.estimators import TMLE

    rng = np.random.default_rng(6)
    import pandas as pd

    n = 600
    wv = rng.integers(0, 2, n).astype(float)
    av = rng.normal(loc=wv, scale=1.0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.3 * av - 0.2 * wv)))).astype(float)
    frame = pd.DataFrame({"W": wv, "A": av, "Y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TMLE(
            policies=[Shift(0.3, cap=None, name="up")],
            ratio="classifier",
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            n_folds=3,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W"])
    report = result.single().diagnostics.support()["up"]
    assert len(report.fold_mean_ratio) == 3
    assert all(np.isfinite(value) for value in report.fold_mean_ratio)


def test_m_c5_a_label_one_total_weight_of_two_breaks_the_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reweight(monkeypatch, "double")
    assert np.max(np.abs(_classifier_normalization(RANDOM) - 1.0)) > CAUGHT
