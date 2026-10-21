"""The frequency-weighted bootstrap replicate equals the materialized refit (X39, T1).

Each witness compares :class:`~cleverly.longitudinal.frequency.FrequencyReplicate` with a full
refit of the resample that :func:`~cleverly.inference.bootstrap.run_bootstrap` materializes,
on the same row index.  Each mutation control breaks one of the six cases in the module
docstring and must break the equality.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.inference.bootstrap import _bootstrap_draw, run_bootstrap, run_index_bootstrap
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, SDR, frequency
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal.frequency import FrequencyReplicate, distinct_rows, merge_equal_rows
from cleverly.longitudinal.regimen import resolve_plans
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_

pytestmark = pytest.mark.xdist_group("bootstrap_fast_path")

K = 3
#: A static, a per-node constant, a rule and a policy plan.  The first is the reference.
PLANS = ("low", "high", "switch", "respond", "mix")
SIZES = {2: 1200, 3: 1500}
EXACT = 1e-10


@dataclass
class Fitted:
    """One fit, the estimator that ran it, and the replicate it built."""

    result: Any
    model: Any
    replicate: FrequencyReplicate | None


def _frame(m: int) -> pd.DataFrame:
    return law_.draw(K, m, SIZES[m], 11, law_.L_R)


def _estimator(name: str, m: int, **settings: Any) -> Any:
    cls = {"LTMLE": LTMLE, "SDR": SDR}[name]
    settings.setdefault("n_folds", 1)
    settings.setdefault("n_bootstrap", 2)
    settings.setdefault("outcome_learner", GLM())
    settings.setdefault("pseudo_learner", GLM() if name == "LTMLE" else LinearRegression())
    return cls(
        plans_.regimens(PLANS, K, m),
        reference="low",
        g_bounds=(1e-8, 1.0),
        random_state=7,
        **settings,
    )


def _fit(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    m: int,
    *,
    fast: bool = True,
    frame: pd.DataFrame | None = None,
    **settings: Any,
) -> Fitted:
    """Fit with the replicate captured, or with the fast path switched off."""
    original = longitudinal_estimator._SequentialEstimator._frequency_replicate
    captured: dict[str, Any] = {}

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        replicate = original(self, *args, **kwargs) if fast else None
        captured["replicate"] = replicate
        return replicate

    model = _estimator(name, m, **settings)
    with monkeypatch.context() as patch:
        patch.setattr(longitudinal_estimator._SequentialEstimator, "_frequency_replicate", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                _frame(m) if frame is None else frame,
                **law_.roles(K, m),
                **law_.probability_columns(K, m),
            )
    return Fitted(result, model, captured.get("replicate"))


def _materialized(fitted: Fitted, index: np.ndarray) -> dict[str, tuple[float, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.model._bootstrap_estimates(fitted.result.data.subset(index))


def _fast(fitted: Fitted, index: np.ndarray) -> dict[str, tuple[float, float]]:
    assert fitted.replicate is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.replicate(index)


def _largest_gap(
    left: dict[str, tuple[float, float]], right: dict[str, tuple[float, float]]
) -> float:
    assert set(left) == set(right)
    return max(
        max(abs(left[name][0] - right[name][0]), abs(left[name][1] - right[name][1]))
        for name in left
    )


def _plan(fitted: Fitted, label: str) -> Any:
    data = fitted.result.data
    plans = resolve_plans(fitted.result.config.regimens, data)
    return next(plan for plan in plans if plan.label == label)


@pytest.fixture(scope="module")
def module_patch() -> Iterator[pytest.MonkeyPatch]:
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


@pytest.fixture(scope="module", params=[("LTMLE", 2), ("LTMLE", 3), ("SDR", 2), ("SDR", 3)])
def pair(request: pytest.FixtureRequest, module_patch: pytest.MonkeyPatch) -> tuple[Fitted, Fitted]:
    """The same bootstrap fit on the fast path and on the materialized refit."""
    name, m = request.param
    fast = _fit(module_patch, name, m, n_bootstrap=4, simultaneous=True)
    slow = _fit(module_patch, name, m, fast=False, n_bootstrap=4, simultaneous=True)
    return fast, slow


class TestTheFastPathEqualsTheMaterializedRefit:
    def test_every_draw_and_standard_error_agree(self, pair: tuple[Fitted, Fitted]) -> None:
        fast, slow = pair
        assert fast.replicate is not None, "the fit took the materialized refit"
        assert slow.replicate is None
        # The plans hold rule, per-node and policy nodes, and the data declares censoring.
        kinds = {type(regimen).__name__ for regimen in fast.result.config.regimens}
        assert kinds == {"Regimen", "DynamicRegimen"}
        assert not np.all(fast.result.data.uncensored)
        a, b = fast.result.bootstrap, slow.result.bootstrap
        assert a.n_failed == b.n_failed
        assert set(a.draws) == set(b.draws) == set(fast.result.estimates)
        for name in a.draws:
            np.testing.assert_allclose(a.draws[name], b.draws[name], rtol=0, atol=EXACT)
            np.testing.assert_allclose(a.std_errors[name], b.std_errors[name], rtol=0, atol=EXACT)

    def test_the_resample_holds_duplicates(self, pair: tuple[Fitted, Fitted]) -> None:
        fast, _ = pair
        assert fast.replicate is not None
        index, _ = _bootstrap_draw(fast.result.data.n, None, np.random.default_rng(0))
        counts = fast.replicate.counts(index)
        assert counts.max() >= 3
        assert np.count_nonzero(counts == 0) > 0
        assert fast.replicate.data.n < fast.result.data.n

    def test_a_replicate_that_fails_fails_on_both_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Seed 7 draws one resample of the m = 3 sample that leaves a regimen without a
        # follower at a node.  Both paths drop it.
        fast = _fit(monkeypatch, "LTMLE", 3, n_bootstrap=4)
        slow = _fit(monkeypatch, "LTMLE", 3, fast=False, n_bootstrap=4)
        assert fast.result.bootstrap.n_failed == slow.result.bootstrap.n_failed == 1


class TestTheSameRowIndex:
    def test_the_index_bootstrap_reads_the_rows_run_bootstrap_subsets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fitted = _fit(monkeypatch, "LTMLE", 2)
        data = fitted.result.data
        subsets: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        marker = np.arange(data.n, dtype=float)

        def by_subset(sample: Any) -> dict[str, tuple[float, float]]:
            subsets.append(np.asarray(sample.baseline)[:, -1].copy())
            return {"x": (0.0, 1.0)}

        def by_index(index: np.ndarray) -> dict[str, tuple[float, float]]:
            indices.append(np.asarray(index))
            return {"x": (0.0, 1.0)}

        marked = data.subset(np.arange(data.n))
        object.__setattr__(marked, "baseline", np.column_stack([data.baseline, marker]))
        run_bootstrap(marked, by_subset, n_replicates=3, resampling="iid", random_state=5)
        run_index_bootstrap(marked, by_index, n_replicates=3, random_state=5)
        assert len(subsets) == len(indices) == 3
        for subset, index in zip(subsets, indices, strict=True):
            np.testing.assert_array_equal(subset, marker[index])

    def test_clustered_data_is_refused(self) -> None:
        class Clustered:
            n = 4
            cluster = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="resamples clusters"):
            run_index_bootstrap(Clustered(), lambda index: {}, n_replicates=2)  # type: ignore[arg-type]


class TestEachCase:
    """The witnesses and mutation controls for the six cases of the module docstring."""

    def test_a_constant_node_in_the_resample_takes_the_constant_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fitted = _fit(monkeypatch, "LTMLE", 2)
        data = fitted.result.data
        follows = _plan(fitted, "high").masks(data).following(K)
        outcome = np.asarray(data.outcome, dtype=float)
        # The resample keeps only the final-node followers of ``high`` with Y = 1, so that
        # node's target is constant in the resample and not in the sample.
        assert np.unique(outcome[follows]).size == 2
        keep = np.flatnonzero(~follows | (outcome == 1.0))
        index = np.concatenate([keep, keep[: data.n - keep.size]])
        expected = _materialized(fitted, index)
        assert _largest_gap(_fast(fitted, index), expected) < EXACT

        # Mutation: presence read from the full-sample followers, not the resample's.
        monkeypatch.setattr(
            FrequencyReplicate, "_fit_rows", staticmethod(lambda node, present: node.trained_on)
        )
        try:
            mutated = _fast(fitted, index)
        except Exception:
            return
        assert _largest_gap(mutated, expected) > 1e-6

    def test_separation_fails_both_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fitted = _fit(monkeypatch, "LTMLE", 2)
        data = fitted.result.data
        follows = _plan(fitted, "high").masks(data).following(K)
        outcome = np.asarray(data.outcome, dtype=float)
        w = np.asarray(data.baseline, dtype=float)[:, 0]
        separated = (w >= 2) == (outcome == 1.0)
        keep = np.flatnonzero(~follows | separated)
        assert np.unique(outcome[follows & separated]).size == 2
        index = np.concatenate([keep, keep[: data.n - keep.size]])
        with pytest.raises(Exception) as materialized:
            _materialized(fitted, index)
        with pytest.raises(Exception) as fast:
            _fast(fitted, index)
        assert type(fast.value) is type(materialized.value)

    def test_a_targeting_only_replicate_breaks_the_equality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: freeze every node regression at the full-sample fit."""
        fitted = _fit(monkeypatch, "LTMLE", 3)
        replicate = fitted.replicate
        assert replicate is not None
        index, _ = _bootstrap_draw(fitted.result.data.n, None, np.random.default_rng(3))
        expected = _materialized(fitted, index)
        assert _largest_gap(_fast(fitted, index), expected) < EXACT

        original = FrequencyReplicate._regress
        frozen: dict[tuple[str, int], tuple[np.ndarray, np.ndarray | None]] = {}

        def record(
            self: Any,
            node: Any,
            target: Any,
            counts: Any,
            present: Any,
            rows: Any,
            plan: Any,
            *,
            clip: Any,
        ) -> Any:
            every = np.flatnonzero(node.at_risk & present)
            value, by_arm = original(self, node, target, counts, present, every, plan, clip=clip)
            frozen[(plan.label, node.time)] = (value, by_arm)
            return original(self, node, target, counts, present, rows, plan, clip=clip)

        sample = np.bincount(replicate.inverse, minlength=replicate.data.n).astype(float)
        monkeypatch.setattr(FrequencyReplicate, "_regress", record)
        replicate.from_counts(sample)

        def stale(
            self: Any,
            node: Any,
            target: Any,
            counts: Any,
            present: Any,
            rows: Any,
            plan: Any,
            *,
            clip: Any,
        ) -> Any:
            value, by_arm = frozen[(plan.label, node.time)]
            position = np.cumsum(node.at_risk) - 1
            at = position[rows]
            return value[at], None if by_arm is None else by_arm[at]

        monkeypatch.setattr(FrequencyReplicate, "_regress", stale)
        assert _largest_gap(_fast(fitted, index), expected) > 1e-6

    def test_counts_reach_the_fit_as_frequencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: unit weights on the distinct rows, as if the counts were not passed."""
        fitted = _fit(monkeypatch, "SDR", 2)
        index, _ = _bootstrap_draw(fitted.result.data.n, None, np.random.default_rng(4))
        expected = _materialized(fitted, index)
        assert _largest_gap(_fast(fitted, index), expected) < EXACT
        original = merge_equal_rows
        monkeypatch.setattr(
            frequency,
            "merge_equal_rows",
            lambda design, target, weight: original(design, target, np.minimum(weight, 1.0)),
        )
        assert _largest_gap(_fast(fitted, index), expected) > 1e-6


def _continuous_outcome(m: int) -> pd.DataFrame:
    """The binary-outcome law with Gaussian noise added to every observed outcome."""
    frame = _frame(m).copy()
    noise = np.random.default_rng(1).normal(size=len(frame)) * 0.5
    frame["Y"] = frame["Y"] + noise * frame["Y"].notna()
    return frame


class TestADataDependentScaler:
    """Case 5: a continuous outcome is rescaled by the rows the resample holds."""

    @pytest.mark.parametrize("name", ["LTMLE", "SDR"])
    def test_a_resample_without_the_largest_outcomes_agrees(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        fitted = _fit(monkeypatch, name, 2, frame=_continuous_outcome(2))
        assert fitted.replicate is not None, "a continuous outcome took the materialized refit"
        data = fitted.result.data
        outcome = np.asarray(data.outcome, dtype=float).reshape(data.n, -1)[:, -1]
        order = np.argsort(np.nan_to_num(outcome, nan=-np.inf))[::-1]
        dropped = set(order[:3].tolist())
        index, _ = _bootstrap_draw(data.n, None, np.random.default_rng(5))
        index = np.array([i if i not in dropped else order[10] for i in index])
        assert not dropped & set(index.tolist())
        expected = _materialized(fitted, index)
        assert _largest_gap(_fast(fitted, index), expected) < EXACT

        # Mutation: the scaler read from every distinct row, not from the rows present.
        original = FrequencyReplicate._scaler
        monkeypatch.setattr(
            FrequencyReplicate,
            "_scaler",
            lambda self, present: original(self, np.ones_like(present)),
        )
        assert _largest_gap(_fast(fitted, index), expected) > 1e-6


class TestEligibility:
    def test_a_continuous_or_mixed_treatment_node_takes_the_materialized_refit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = longitudinal_estimator._SequentialEstimator._frequency_replicate

        def kinds(kind: str) -> list[Any]:
            seen: list[Any] = []

            def spy(self: Any, prepared: Any, *args: Any, **kwargs: Any) -> Any:
                with monkeypatch.context() as inner:
                    if kind != "categorical":
                        inner.setattr(type(prepared), "node_kind", lambda _self, _time: kind)
                    seen.append(original(self, prepared, *args, **kwargs))
                return seen[-1]

            with monkeypatch.context() as patch:
                patch.setattr(
                    longitudinal_estimator._SequentialEstimator, "_frequency_replicate", spy
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _estimator("LTMLE", 2).fit(
                        _frame(2), **law_.roles(K, 2), **law_.probability_columns(K, 2)
                    )
            return seen

        (control,) = kinds("categorical")
        assert control is not None
        assert kinds("continuous") == [None]
        assert kinds("mixed") == [None]

    def test_a_penalized_subclass_does_not_inherit_the_declaration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Ridge(GLM):
            """A penalized fit would break frequency exactness; the base class declares it."""

        class DeclaredRidge(Ridge):
            frequency_exact = True

        assert frequency.frequency_exact(GLM())
        assert not frequency.frequency_exact(Ridge())
        assert frequency.frequency_exact(DeclaredRidge())

        original = longitudinal_estimator._SequentialEstimator._frequency_replicate
        seen: list[Any] = []

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(original(self, *args, **kwargs))
            return seen[-1]

        monkeypatch.setattr(
            longitudinal_estimator._SequentialEstimator, "_frequency_replicate", spy
        )
        model = _estimator("LTMLE", 2, outcome_learner=Ridge(), pseudo_learner=Ridge())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_frame(2), **law_.roles(K, 2), **law_.probability_columns(K, 2))
        assert seen == [None]

    def test_a_cross_fitted_fit_takes_the_materialized_refit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fitted = _fit(monkeypatch, "LTMLE", 2, n_folds=2, n_bootstrap=2)
        assert fitted.replicate is None
        assert fitted.result.bootstrap is not None

    def test_a_penalized_learner_takes_the_materialized_refit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = LTMLE(
            plans_.regimens(("low", "high"), K, 2),
            outcome_learner=LogisticRegression(),
            pseudo_learner=GLM(),
            n_folds=1,
            g_bounds=(1e-8, 1.0),
            n_bootstrap=2,
            random_state=0,
        )
        original = longitudinal_estimator._SequentialEstimator._frequency_replicate
        seen: list[Any] = []

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(original(self, *args, **kwargs))
            return seen[-1]

        monkeypatch.setattr(
            longitudinal_estimator._SequentialEstimator, "_frequency_replicate", spy
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_frame(2), **law_.roles(K, 2), **law_.probability_columns(K, 2))
        assert seen == [None]

    def test_distinct_rows_map_back_to_the_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fitted = _fit(monkeypatch, "LTMLE", 2)
        data = fitted.result.data
        unique, inverse = distinct_rows(data)
        np.testing.assert_array_equal(
            np.nan_to_num(np.asarray(unique.outcome)[inverse], nan=-1.0),
            np.nan_to_num(np.asarray(data.outcome), nan=-1.0),
        )
        np.testing.assert_array_equal(
            np.asarray(unique.baseline)[inverse], np.asarray(data.baseline)
        )

    def test_merging_never_joins_rows_that_differ(self) -> None:
        design = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0 + 4e-16], [1.0, 0.0]])
        target = np.array([1.0, 1.0, 1.0, 0.0])
        merged, response, weight = merge_equal_rows(design, target, np.array([1.0, 2.0, 3.0, 4.0]))
        assert merged.shape[0] == 3
        assert sorted(weight.tolist()) == [3.0, 3.0, 4.0]
        assert response.shape == (3,)
