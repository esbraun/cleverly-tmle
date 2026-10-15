"""A repeated outcome: each horizon is the end-of-study fit on the truncated panel.

W1 is the reduction.  For every horizon ``h`` the repeated fit's estimate, influence
curve, node regressions and cumulative mechanism equal those of the end-of-study fit on
the panel truncated after ``h``, with ``Y_h`` as its outcome and each earlier ``Y_s`` as
the first covariate of block ``s + 1``.  The test runs over every composition the
repeated outcome supports.

W8 checks that the last horizon is the end-of-study fit with the earlier outcomes declared
as covariates.  W13 checks that the fit never reads the ``outcome`` field: a permuted copy
changes nothing.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.interventions import ModifiedPolicy, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData, Repeated

T = 3
N = 400

#: Equality of two fits that share every input.  Both run the same arithmetic on the same
#: arrays, so the tolerance only absorbs a reordered sum.
RTOL = 1e-12
ATOL = 1e-13


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def panel(
    *, n: int = N, seed: int = 0, continuous: bool = False, held: bool = False
) -> pd.DataFrame:
    """A ``T = 3`` panel whose outcome feeds the next node's covariate, treatment and censoring.

    The frame carries the true mechanism columns ``g{t}_0``, ``g{t}_1`` and ``r{t}``, a
    cluster column and a weight column, so each composition reads the same draw.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    columns: dict[str, Any] = {"W": w, "cluster": rng.integers(0, 40, size=n)}
    columns["weight"] = rng.choice([1.0, 2.0, 3.0], size=n)
    alive = np.ones(n, dtype=bool)
    previous_y = np.zeros(n)
    previous_a = np.zeros(n)
    held_a = rng.binomial(1, _expit(0.3 * w)).astype(float)
    for t in range(1, T + 1):
        lt = rng.normal(size=n) + 0.5 * previous_y - 0.3 * previous_a
        if held:
            p_treat = np.full(n, np.nan)
            at = held_a
        else:
            p_treat = _expit(0.2 * w + 0.4 * lt - 0.8 * previous_y + 0.3 * previous_a)
            at = rng.binomial(1, p_treat).astype(float)
        p_stay = _expit(1.8 + 0.3 * lt - 0.7 * previous_y + 0.2 * at)
        ct = rng.binomial(1, p_stay).astype(float)
        mean = _expit(-0.3 + 0.4 * w + 0.5 * lt + 0.6 * at + 0.9 * previous_y)
        if continuous:
            yt = mean * (1.0 + t) + rng.normal(scale=0.4, size=n)
        else:
            yt = rng.binomial(1, mean).astype(float)
        for name, values in (
            (f"L{t}", lt),
            (f"A{t}", at),
            (f"C{t}", ct),
            (f"Y{t}", yt),
            (f"g{t}_1", p_treat),
            (f"g{t}_0", 1.0 - p_treat),
            (f"r{t}", p_stay),
        ):
            column = np.asarray(values, dtype=float).copy()
            column[~alive] = np.nan
            columns[name] = column
        columns[f"Y{t}"][alive & (ct == 0)] = np.nan
        alive = alive & (ct == 1)
        previous_y = np.where(np.isnan(yt), 0.0, yt)
        previous_a = at
    if held:
        columns["A"] = held_a
        for t in range(1, T + 1):
            del columns[f"A{t}"], columns[f"g{t}_1"], columns[f"g{t}_0"]
    return pd.DataFrame(columns)


def _treat_if_y1(history: Any) -> Any:
    return (np.asarray(history["Y1"]) > 0.5).astype(int)


def _stochastic_node(history: Any) -> np.ndarray:
    p = _expit(0.5 * np.asarray(history["L2"]))
    return np.column_stack([1.0 - p, p])


def _treat_if_y2(a: Any, h: Any) -> Any:
    """Treat the units whose second outcome was high; leave the others' natural arm."""
    return np.where(np.asarray(h["Y2"], dtype=float) > 0.5, 1, np.asarray(a))


#: Each plan as one entry per node, so the truncated fit takes its first ``h`` entries.
PLANS: dict[str, tuple[Any, ...]] = {
    "never": (0, 0, 0),
    "always": (1, 1, 1),
}
RULE: dict[str, tuple[Any, ...]] = {"rule": (1, _treat_if_y1, 1)}
POLICY: dict[str, tuple[Any, ...]] = {
    "policy": (1, Stochastic(_stochastic_node, "q2", density_kind="known"), 0)
}
MTP: dict[str, tuple[Any, ...]] = {
    "mtp": (
        1,
        1,
        ModifiedPolicy("treat if Y2 high", apply=_treat_if_y2, policy_kind="known"),
    )
}


def regimens(plans: dict[str, tuple[Any, ...]], horizon: int, *, held: bool) -> dict[str, Any]:
    """The plans truncated after ``horizon`` nodes, in ``LTMLE``'s declaration."""
    out: dict[str, Any] = {}
    for label, nodes in plans.items():
        if held:
            out[label] = nodes[0]
            continue
        head = nodes[:horizon]
        if all(isinstance(node, int) for node in nodes):
            out[label] = head if len(head) > 1 else head[0]
        elif any(callable(node) and not hasattr(node, "density_kind") for node in head):
            out[label] = DynamicRegimen(label, head, rule_kind="known")
        else:
            out[label] = DynamicRegimen(label, head)
    return out


@dataclasses.dataclass(frozen=True)
class Composition:
    """One supported composition: what the plans are, and what the fit is handed."""

    name: str
    plans: dict[str, tuple[Any, ...]]
    n_folds: int = 1
    continuous: bool = False
    held: bool = False
    cluster: bool = False
    weights: bool = False
    known: bool = False


COMPOSITIONS = (
    Composition("static", PLANS),
    Composition("static_folds", PLANS, n_folds=5),
    Composition("rule_reads_y1", {**PLANS, **RULE}),
    Composition("rule_reads_y1_folds", {**PLANS, **RULE}, n_folds=5),
    Composition("known_stochastic_policy", {**PLANS, **POLICY}),
    Composition("mtp", {**PLANS, **MTP}),
    Composition("continuous", {**PLANS, **RULE}, continuous=True),
    Composition("continuous_folds", PLANS, continuous=True, n_folds=5),
    Composition("clusters", PLANS, cluster=True),
    Composition("clusters_folds", PLANS, cluster=True, n_folds=5),
    Composition("weights", {**PLANS, **RULE}, weights=True),
    Composition("held_treatment", PLANS, held=True),
    Composition("known_mechanisms", {**PLANS, **RULE}, known=True),
)


def estimator(composition: Composition, plans: dict[str, Any]) -> LTMLE:
    regression = LinearRegression() if composition.continuous else LogisticRegression()
    return LTMLE(
        plans,
        reference="never",
        outcome_learner=regression,
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        censoring_learner=LogisticRegression(),
        n_folds=composition.n_folds,
        q_bounds=(-1.0, 6.0) if composition.continuous else None,
        simultaneous=False,
        random_state=3,
    )


def columns(composition: Composition, horizon: int, *, repeated: bool) -> dict[str, Any]:
    """The fit's roles: the whole repeated panel, or the panel truncated after ``horizon``."""
    nodes = range(1, (T if repeated else horizon) + 1)
    blocks = [[f"L{t}"] for t in nodes]
    if not repeated:
        # The reduction: Y_s is the first covariate of block s + 1.
        blocks = [blocks[0], *([f"Y{t - 1}", *blocks[t - 1]] for t in nodes if t > 1)]
    out: dict[str, Any] = {
        "outcome": Repeated([f"Y{t}" for t in nodes]) if repeated else f"Y{horizon}",
        "treatment": "A" if composition.held else [f"A{t}" for t in nodes],
        "baseline": ["W"],
        "time_varying": blocks,
        "censoring": [f"C{t}" for t in nodes],
    }
    if composition.cluster:
        out["id"] = "cluster"
    if composition.weights:
        out["weights"] = "weight"
    if composition.known:
        out["treatment_probabilities"] = {f"A{t}": {0.0: f"g{t}_0", 1.0: f"g{t}_1"} for t in nodes}
        out["censoring_probabilities"] = {f"C{t}": f"r{t}" for t in nodes}
    return out


def fit(composition: Composition, frame: pd.DataFrame, horizon: int, *, repeated: bool) -> Any:
    plans = regimens(composition.plans, T if repeated else horizon, held=composition.held)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(composition, plans).fit(
            frame, **columns(composition, horizon, repeated=repeated)
        )


@pytest.fixture(scope="module", params=COMPOSITIONS, ids=[c.name for c in COMPOSITIONS])
def reduction(request: pytest.FixtureRequest) -> tuple[Composition, pd.DataFrame, Any]:
    composition = request.param
    frame = panel(continuous=composition.continuous, held=composition.held)
    return composition, frame, fit(composition, frame, T, repeated=True)


def _close(left: Any, right: Any, what: str) -> None:
    if left is None or right is None:
        assert left is None and right is None, what
        return
    np.testing.assert_allclose(
        np.asarray(left, dtype=float),
        np.asarray(right, dtype=float),
        rtol=RTOL,
        atol=ATOL,
        err_msg=what,
    )


@pytest.mark.parametrize("horizon", range(1, T + 1))
def test_w1_each_horizon_is_the_truncated_end_of_study_fit(
    reduction: tuple[Composition, pd.DataFrame, Any], horizon: int
) -> None:
    composition, frame, repeated = reduction
    truncated = fit(composition, frame, horizon, repeated=False)
    assert repeated.data.is_repeated and not truncated.data.is_repeated
    for label in composition.plans:
        name = f"ey_regimen[{label} @ t={horizon}]"
        assert repeated.psi(name) == pytest.approx(
            truncated.psi(f"ey_regimen[{label}]"), rel=RTOL, abs=ATOL
        )
        _close(
            repeated.influence_curves[name],
            truncated.influence_curves[f"ey_regimen[{label}]"],
            f"{name} influence curve",
        )
        left = repeated.fits[f"{label} @ t={horizon}"]
        right = truncated.fits[label]
        assert left.horizon == horizon and len(left.steps) == len(right.steps) == horizon
        for field in ("cumulative", "cumulative_unbounded", "assignment", "node_ratio"):
            mine, theirs = getattr(left, field), getattr(right, field)
            if field == "node_ratio" and (mine is None) != (theirs is None):
                # A policy plan records a node ratio at every node.  Its static head,
                # declared alone on the truncated panel, records none.  The cumulative
                # product above is the quantity both fits use.
                continue
            if mine is not None and np.ndim(mine) == 2:
                mine = np.asarray(mine)[:, :horizon]
            if theirs is not None and np.ndim(theirs) == 2:
                theirs = np.asarray(theirs)[:, :horizon]
            _close(mine, theirs, f"{name} {field}")
        for mine, theirs in zip(left.steps, right.steps, strict=True):
            assert mine.time == theirs.time
            np.testing.assert_array_equal(mine.trained_on, theirs.trained_on)
            for field in ("pseudo_outcome", "initial", "targeted", "clever"):
                a, b = getattr(mine, field), getattr(theirs, field)
                _close(a[mine.trained_on], b[theirs.trained_on], f"{name} t={mine.time} {field}")


def test_w8_the_last_horizon_is_the_end_of_study_fit() -> None:
    """At ``horizons=[T]`` the repeated fit is the end-of-study fit, number for number."""
    composition = COMPOSITIONS[0]
    frame = panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repeated = LTMLE(
            regimens(PLANS, T, held=False),
            reference="never",
            horizons=[T],
            outcome_learner=LogisticRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            censoring_learner=LogisticRegression(),
            n_folds=1,
            simultaneous=False,
            random_state=3,
        ).fit(frame, **columns(composition, T, repeated=True))
    end = fit(composition, frame, T, repeated=False)
    assert set(repeated) == {f"ey_regimen[{label} @ t={T}]" for label in PLANS} | {
        f"ate_regimen[always vs never @ t={T}]"
    }
    for label in PLANS:
        assert repeated.psi(f"ey_regimen[{label} @ t={T}]") == pytest.approx(
            end.psi(f"ey_regimen[{label}]"), rel=RTOL, abs=ATOL
        )


def test_w13_the_fit_never_reads_the_outcome_field() -> None:
    """A permuted ``outcome`` field moves no estimate: every horizon reads ``repeated``.

    ``outcome`` holds ``Y_T`` for code that reads an end-of-study outcome.  A reader that
    took it in place of :meth:`LongitudinalData.outcome_at` would change here.
    """
    composition = COMPOSITIONS[2]
    frame = panel()
    data = LongitudinalData.from_frame(frame, **columns(composition, T, repeated=True))
    order = np.random.default_rng(0).permutation(data.n)
    stale = dataclasses.replace(data, outcome=np.asarray(data.outcome)[order])
    plans = regimens(composition.plans, T, held=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = estimator(composition, plans).fit(data)
        second = estimator(composition, plans).fit(stale)
    for name in first:
        assert first.psi(name) == second.psi(name), name
        np.testing.assert_array_equal(first.influence_curves[name], second.influence_curves[name])
