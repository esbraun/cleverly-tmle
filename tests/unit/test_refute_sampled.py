"""Refutation on a simple random sample of the fit's units (X106 part H).

``refute(result, sample_limit=m)`` draws ``m`` of the fit's units without replacement from its
own child of the root seed, refits the stored estimator on them, and runs the tests on that
refit exactly as ``refute()`` runs them on any fit of ``m`` units. The witnesses:

- W1 a sampled call equals ``refute()`` of the same estimator fitted on the sampled frame rows,
  bit for bit, for each family and for weights, a known mechanism, strata, a band-rank placebo,
  a split plan and a band; the reference is built here, not with the implementation's helpers;
- W2 a limit of at least the unit count is the full-data call; W3 no test reads the full fit;
- W3b a sample that loses an arm or stratum refuses, one that makes a column constant withholds;
- W4 the sample law; W5 refusals, all before any refit; W6 the preview; W7 checkpoints;
- W8 the facade; W9 the verdict scope; W10 seeds.
"""

from __future__ import annotations

import functools
import importlib
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment, SplitPlan
from cleverly.datasets import make_longitudinal
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.learners import GLM, random_partition
from cleverly.longitudinal import LTMLE, StructuralNestedMean
from cleverly.validation import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    RefutationCheckpoint,
    RefutationSample,
    planned_refits,
    refute,
)
from cleverly.validation._sampled_refute import resolved_rule, unit_count
from tests.unit._capability_sweep_support import KINDS, sweep_arguments
from tests.unit.test_refuter_draw_replay import _same_bits

refute_module: Any = importlib.import_module("cleverly.validation.refute")

SEED = 11
DRAWS = 3


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _sample_seed(seed: int) -> int:
    """The sample's own seed, written out: child 0 of ``SeedSequence([seed, 5])``."""
    return int(np.random.SeedSequence([seed, 5]).spawn(1)[0].generate_state(1)[0])


def _rows(n: int, m: int, seed: int = SEED) -> np.ndarray:
    return np.sort(np.random.default_rng(_sample_seed(seed)).choice(n, size=m, replace=False))


def _cluster_rows(labels: np.ndarray, m: int, seed: int = SEED) -> np.ndarray:
    unique = np.unique(labels)
    chosen = np.random.default_rng(_sample_seed(seed)).choice(unique, size=m, replace=False)
    return np.flatnonzero(np.isin(labels, chosen))


def _point_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-w[:, 0])))
    y = 0.5 * w[:, 0] - 0.3 * w[:, 1] + 0.5 * a + rng.normal(size=n)
    weight = rng.uniform(0.5, 2.0, size=n)
    stratum = (w[:, 1] > 0).astype(int)
    cluster = np.repeat(np.arange(n // 5), 5)
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "A": a,
            "Y": y,
            "wt": weight,
            "S": stratum,
            "cl": cluster,
            "p1": np.full(n, 0.5),
            "p0": np.full(n, 0.5),
        }
    )


def _tmle(**overrides: Any) -> TMLE:
    options: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": True,
        "n_folds": 2,
        "simultaneous": False,
        "q_bounds": (-10.0, 10.0),
        "random_state": SEED,
    }
    options.update(overrides)
    return TMLE(**options)


#: name -> (estimator settings, fit keywords, clustered)
POINT_CASES: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {
    "plain": ({}, {}, False),
    "clustered": ({}, {"id": "cl"}, True),
    "weighted": ({}, {"weights": "wt"}, False),
    "known mechanism": ({}, {"treatment_probabilities": {1: "p1", 0: "p0"}}, False),
    "strata": ({}, {"strata": ["S"]}, False),
    "band": ({"simultaneous": True}, {}, False),
}


def _fit_point(frame: pd.DataFrame, settings: dict[str, Any], keywords: dict[str, Any]) -> Any:
    covariates = ["W1", "W2", *keywords.get("strata", ())]
    return _quiet(
        lambda: (
            _tmle(**settings)
            .fit(frame, outcome="Y", treatment="A", covariates=covariates, **keywords)
            .single()
        )
    )


def _assert_same_report(sampled: Any, reference: Any, *, rtol: float = 0.0) -> None:
    """Bit for bit, or to ``rtol`` where the two calls normalise weights in different orders."""
    assert [t.name for t in sampled.tests] == [t.name for t in reference.tests]
    for first, second in zip(sampled.tests, reference.tests, strict=True):
        if rtol:
            np.testing.assert_allclose(first.values, second.values, rtol=rtol, atol=0)
            np.testing.assert_allclose(first.original, second.original, rtol=rtol, atol=0)
            np.testing.assert_allclose(first.statistic, second.statistic, rtol=rtol, atol=0)
        else:
            assert _same_bits(first.values, second.values), first.name
            assert _same_bits(first.original, second.original), first.name
            assert _same_bits(first.statistic, second.statistic), first.name
            assert first.detail == second.detail, first.name
        assert first.passed == second.passed, first.name


# ----------------------------------------------------------------- W1 identity


@pytest.mark.parametrize("case", list(POINT_CASES))
def test_w1_a_sampled_point_call_is_the_refutation_of_the_sampled_frame(case: str) -> None:
    settings, keywords, clustered = POINT_CASES[case]
    frame = _point_frame()
    fit = _fit_point(frame, settings, keywords)
    m = 30 if clustered else 150
    rows = _cluster_rows(frame["cl"].to_numpy(), m) if clustered else _rows(len(frame), m)
    tests = ("placebo", "subset", "random_common_cause")
    sampled = _quiet(
        lambda: refute(fit, tests=tests, sample_limit=m, random_state=SEED, n_replicates=DRAWS)
    )
    reference_fit = _fit_point(frame.iloc[rows].reset_index(drop=True), settings, keywords)
    reference = _quiet(
        lambda: refute(reference_fit, tests=tests, random_state=SEED, n_replicates=DRAWS)
    )
    # The sampled call renormalises weights already at mean one on all rows; the reference
    # normalises the raw column once. The two differ in the last bits only.
    _assert_same_report(sampled, reference, rtol=1e-12 if "weights" in keywords else 0.0)
    assert sampled.sample is not None and sampled.sample.applied
    assert sampled.sample.sampled_rows == rows.size
    assert sampled.sample.seed == _sample_seed(SEED)
    assert sampled.sample.rule == ("clusters" if clustered else "units")


def test_w1_a_split_plan_fit_refits_under_its_generator_records_on_the_sample() -> None:
    frame = _point_frame()
    plan = SplitPlan.from_folds([random_partition(len(frame), 2, seed=5)])
    fit = _fit_point(frame, {"split_plan": plan}, {})
    rows = _rows(len(frame), 150)
    sampled = _quiet(
        lambda: refute(
            fit,
            tests=("placebo", "subset"),
            sample_limit=150,
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    own = SplitPlan.from_folds([random_partition(150, 2, seed=5)])
    reference_fit = _fit_point(frame.iloc[rows].reset_index(drop=True), {"split_plan": own}, {})
    reference = _quiet(
        lambda: refute(
            reference_fit, tests=("placebo", "subset"), random_state=SEED, n_replicates=DRAWS
        )
    )
    _assert_same_report(sampled, reference)


def test_w1_a_band_rank_placebo_redraws_its_bands_on_the_sample() -> None:
    from cleverly.interventions import Support

    from .. import dose_declarations as law
    from .test_mtp_declared_support import INSIDE, NATURAL

    frame = law.point_frame(n=400)

    def fit_on(data: pd.DataFrame) -> Any:
        estimator = TMLE(policies=[NATURAL, INSIDE], ratio="density", **law.point_learners())
        return _quiet(
            lambda: estimator.fit(
                data,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_support=Support(0.0, law.upper),
            ).single()
        )

    fit = fit_on(frame)
    estimand = "ate_policy[inside vs natural course]"
    rows = _rows(len(frame), 250)
    sampled = _quiet(
        lambda: refute(
            fit,
            estimand=estimand,
            tests=("placebo",),
            sample_limit=250,
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    reference = _quiet(
        lambda: refute(
            fit_on(frame.iloc[rows].reset_index(drop=True)),
            estimand=estimand,
            tests=("placebo",),
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    _assert_same_report(sampled, reference)


def _panel_frame(n: int = 600) -> pd.DataFrame:
    frame, _ = make_longitudinal(n=n, seed=0, censoring=False)
    return frame


PANEL = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
}


def _ltmle() -> LTMLE:
    return LTMLE(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=SEED,
        simultaneous=False,
    )


def _snm() -> StructuralNestedMean:
    return StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=SEED,
        simultaneous=False,
    )


@pytest.mark.parametrize(
    ("build", "estimand"),
    [(_ltmle, "ate_regimen"), (_snm, "blip[A1:1.0:intercept]")],
    ids=["longitudinal", "structural"],
)
def test_w1_a_sampled_panel_call_is_the_refutation_of_the_sampled_units(
    build: Callable[[], Any], estimand: str
) -> None:
    frame = _panel_frame()
    fit = _quiet(lambda: build().fit(frame, **PANEL))
    if estimand == "ate_regimen":
        estimand = next(name for name in fit if name.startswith("ate_regimen"))
    rows = _rows(len(frame), 400)
    tests = ("subset", "random_common_cause")
    sampled = _quiet(
        lambda: refute(
            fit,
            estimand=estimand,
            tests=tests,
            sample_limit=400,
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    reference_fit = _quiet(lambda: build().fit(frame.iloc[rows].reset_index(drop=True), **PANEL))
    reference = _quiet(
        lambda: refute(
            reference_fit, estimand=estimand, tests=tests, random_state=SEED, n_replicates=DRAWS
        )
    )
    _assert_same_report(sampled, reference)


def test_w1_a_sampled_survival_call_is_the_refutation_of_the_sampled_units() -> None:
    from tests.unit.test_continuous_survival_refute import _fit, _frame

    frame = _frame(n=300)
    fit = _fit(frame)
    estimand = "ate_regimen[1 vs 0, 1 @ t=2]"
    rows = _rows(len(frame), 200)
    tests = ("placebo", "subset")
    sampled = _quiet(
        lambda: refute(
            fit,
            estimand=estimand,
            tests=tests,
            sample_limit=200,
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    reference_fit = _fit(frame.iloc[rows].reset_index(drop=True), random_state=SEED)
    reference = _quiet(
        lambda: refute(
            reference_fit, estimand=estimand, tests=tests, random_state=SEED, n_replicates=DRAWS
        )
    )
    _assert_same_report(sampled, reference)


def test_w1_mutation_the_full_result_or_the_shared_generator_breaks_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _point_frame()
    fit = _fit_point(frame, {}, {})
    tests = ("placebo", "subset")

    def call() -> Any:
        return _quiet(
            lambda: refute(
                fit, tests=tests, sample_limit=150, random_state=SEED, n_replicates=DRAWS
            )
        )

    right = call()
    monkeypatch.setattr(refute_module, "_sampled_baseline", lambda result, sampled, seed: result)
    wrong = call()
    assert not _same_bits(right["placebo"].values, wrong["placebo"].values)
    monkeypatch.undo()
    monkeypatch.setattr(
        refute_module, "_generated_child_seeds", lambda root, name, count: (root,) * count
    )
    moved = call()
    assert not _same_bits(right["subset"].values, moved["subset"].values)


def test_w1_a_sampled_call_leaves_the_full_result_unchanged() -> None:
    fit = _fit_point(_point_frame(), {}, {})
    before = _quiet(lambda: refute(fit, tests=("subset",), random_state=3, n_replicates=DRAWS))
    _quiet(
        lambda: refute(fit, tests=("subset",), sample_limit=100, random_state=3, n_replicates=DRAWS)
    )
    after = _quiet(lambda: refute(fit, tests=("subset",), random_state=3, n_replicates=DRAWS))
    _assert_same_report(before, after)


# ----------------------------------------------------------------- W2, W3


@functools.cache
def _plain() -> Any:
    return _fit_point(_point_frame(), {}, {})


def test_w2_a_limit_of_at_least_the_unit_count_is_the_full_data_call() -> None:
    fit = _plain()
    full = _quiet(lambda: refute(fit, tests=("placebo",), random_state=3, n_replicates=DRAWS))
    capped = _quiet(
        lambda: refute(
            fit, tests=("placebo",), sample_limit=300, random_state=3, n_replicates=DRAWS
        )
    )
    _assert_same_report(full, capped)
    assert full.sample is None
    assert capped.sample is not None and not capped.sample.applied
    assert "at least the fit's 300 units" in capped.sample.sentence()


def test_w3_every_reference_is_the_sampled_baseline() -> None:
    from cleverly.validation._readings import subset_reading

    fit = _plain()
    report = _quiet(
        lambda: refute(
            fit,
            tests=("placebo", "subset"),
            sample_limit=150,
            random_state=SEED,
            n_replicates=DRAWS,
        )
    )
    sample = report.sample
    assert sample is not None
    assert sample.baseline_estimate != sample.full_estimate
    for test in report.tests:
        assert test.original == sample.baseline_estimate
    values = report["subset"].values
    size = int(0.7 * 150)
    own = subset_reading(values, std_error=sample.baseline_std_error, n=150, m=size)
    full = subset_reading(values, std_error=fit["ate"].plugin_std_error, n=300, m=size)
    assert own.rho == pytest.approx(report["subset"].statistic)
    assert not np.isclose(full.rho, report["subset"].statistic)


def test_w3b_a_sample_that_loses_an_arm_refuses_before_any_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _point_frame()
    frame.loc[:, "A"] = 0
    frame.loc[[3, 4, 5], "A"] = 1
    fit = _fit_point(frame, {"cross_fit": False}, {})
    calls = _count_refits(monkeypatch)
    with pytest.raises(DataError, match="no unit with treatment code"):
        refute(fit, tests=("subset",), sample_limit=20, random_state=SEED)
    assert calls == []


def test_w3b_a_column_constant_only_in_the_sample_withholds_every_verdict() -> None:
    frame = _point_frame()
    rows = _rows(len(frame), 150)
    outside = np.setdiff1d(np.arange(len(frame)), rows)[:2]
    frame["B"] = 0.0
    frame.loc[outside, "B"] = 1.0
    fit = _quiet(
        lambda: (
            _tmle().fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "B"]).single()
        )
    )
    report = _quiet(
        lambda: refute(
            fit, tests=("placebo",), sample_limit=150, random_state=SEED, n_replicates=DRAWS
        )
    )
    assert report.sample is not None
    assert report.sample.departures == ("B constant",)
    assert all(test.passed is None for test in report.tests)


def _count_refits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = TMLE.refit

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TMLE, "refit", counting)
    return calls


# ----------------------------------------------------------------- W4 the sample law


def test_w4_whole_clusters_and_every_row_of_them() -> None:
    frame = _point_frame()
    fit = _fit_point(frame, {}, {"id": "cl"})
    seed = refute_module._generated_child_seeds(SEED, "sample", 1)[0]
    rows = refute_module.draw_rows(fit.data, seed, 20)
    labels = frame["cl"].to_numpy()
    chosen = np.unique(labels[rows])
    assert chosen.size == 20
    np.testing.assert_array_equal(rows, np.flatnonzero(np.isin(labels, chosen)))


def test_w4_inclusion_is_uniform_and_reads_no_data() -> None:
    class Data:
        n = 50
        cluster = None

    counts = np.zeros(50)
    for seed in range(2000):
        counts[refute_module.draw_rows(Data(), seed, 10)] += 1
    assert stats.chisquare(counts).pvalue > 0.001
    assert counts.sum() == 2000 * 10


def test_w4_a_negative_control_is_restricted_to_the_sampled_rows() -> None:
    frame = _point_frame()
    fit = _fit_point(frame, {}, {})
    control = 0.5 * frame["W1"].to_numpy() + np.random.default_rng(4).normal(size=len(frame))
    rows = _rows(len(frame), 150)
    sampled = _quiet(
        lambda: refute(
            fit,
            tests=("negative_control_outcome",),
            sample_limit=150,
            random_state=SEED,
            negative_control_outcome=control,
        )
    )
    reference_fit = _fit_point(frame.iloc[rows].reset_index(drop=True), {}, {})
    reference = _quiet(
        lambda: refute(
            reference_fit,
            tests=("negative_control_outcome",),
            random_state=SEED,
            negative_control_outcome=control[rows],
        )
    )
    _assert_same_report(sampled, reference)
    shifted = _quiet(
        lambda: refute(
            reference_fit,
            tests=("negative_control_outcome",),
            random_state=SEED,
            negative_control_outcome=control[:150],
        )
    )
    assert not _same_bits(shifted.tests[0].values, sampled.tests[0].values)


def test_w4_weights_are_renormalised_on_the_sample() -> None:
    fit = _fit_point(_point_frame(), {}, {"weights": "wt"})
    rows = _rows(300, 150)
    sampled = refute_module._sampled_data(fit, rows)
    assert float(np.mean(sampled.weights)) == pytest.approx(1.0)


# ----------------------------------------------------------------- W5 refusals


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"sample_limit": 1}, "sample_limit must be None or an integer of at least 2"),
        ({"sample_limit": True}, "sample_limit must be None"),
        ({"sample_limit": 2.5}, "sample_limit must be None"),
        ({"sample_limit": 50, "sampling": "rows"}, "unknown sampling"),
        ({"sampling": "units"}, "sampling= names how sample_limit="),
    ],
)
def test_w5_malformed_keywords_raise_value_error(keywords: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        refute(_plain(), tests=("subset",), **keywords)


def test_w5_a_malformed_keyword_comes_before_a_refusal() -> None:
    fit = _fit_point(_point_frame(), {}, {"id": "cl"})
    with pytest.raises(ValueError, match="unknown sampling"):
        refute(fit, tests=("subset",), sample_limit=10, sampling="rows")


@pytest.mark.parametrize(
    ("clustered", "keywords", "message"),
    [
        (True, {"sample_limit": 10, "sampling": "units"}, "this fit is clustered"),
        (False, {"sample_limit": 50, "sampling": "clusters"}, "declares no clusters"),
        (False, {"sample_limit": 5}, "fewer units than the 10 rows"),
    ],
)
def test_w5_sampling_refusals_come_before_any_refit_and_agree_everywhere(
    monkeypatch: pytest.MonkeyPatch, clustered: bool, keywords: dict[str, Any], message: str
) -> None:
    from cleverly.validation.refute import refute_refusal

    fit = _fit_point(_point_frame(), {}, {"id": "cl"} if clustered else {})
    calls = _count_refits(monkeypatch)
    with pytest.raises(CapabilityError, match=message) as raised:
        refute(fit, tests=("subset",), **keywords)
    assert calls == []
    record = refute_refusal(fit, estimand="ate", tests=("subset",), **keywords)
    assert record == raised.value.refusal
    with pytest.raises(CapabilityError) as previewed:
        planned_refits(fit, tests=("subset",), **keywords)
    assert previewed.value.refusal == record


def test_w5_a_full_data_refusal_comes_before_the_sampling_rule() -> None:
    fit = _fit_point(_point_frame(), {}, {"id": "cl"})
    with pytest.raises(CapabilityError, match="noise_nodes"):
        refute(
            fit, tests=("random_common_cause",), noise_nodes=True, sample_limit=10, sampling="units"
        )


# ----------------------------------------------------------------- W6 preview


def test_w6_the_preview_counts_the_baseline_and_the_refits_refute_makes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _plain()
    tests = ("placebo", "subset")
    plan = planned_refits(fit, tests=tests, sample_limit=150, random_state=SEED, n_replicates=DRAWS)
    assert [test.name for test in plan.tests] == ["sampled_baseline", *tests]
    assert plan.tests[0].refits == 1
    assert all(test.bound == "upper" for test in plan.tests[1:])
    calls = _count_refits(monkeypatch)
    _quiet(
        lambda: refute(fit, tests=tests, sample_limit=150, random_state=SEED, n_replicates=DRAWS)
    )
    assert len(calls) == plan.total_refits


# ----------------------------------------------------------------- W7 checkpoints


BOOTSTRAP = 5


def _bootstrap(fit: Any, directory: Path, **keywords: Any) -> Any:
    return _quiet(
        lambda: refute(
            fit,
            tests=("bootstrap_stability",),
            n_replicates=BOOTSTRAP,
            checkpoint=RefutationCheckpoint(directory),
            **keywords,
        )
    )


@functools.cache
def _in_sample() -> Any:
    return _fit_point(_point_frame(), {"cross_fit": False}, {})


def test_w7_a_sampled_directory_serves_only_its_own_sample(tmp_path: Path) -> None:
    fit = _in_sample()
    _bootstrap(fit, tmp_path, sample_limit=150, random_state=SEED)
    for keywords, part in (
        ({"random_state": SEED}, "sampling"),
        ({"sample_limit": 140, "random_state": SEED}, "sampling"),
        ({"sample_limit": 150, "random_state": SEED + 1}, ""),
    ):
        with pytest.raises(CapabilityError, match=part):
            _bootstrap(fit, tmp_path, **keywords)


def test_w7_a_resumed_sampled_call_equals_the_uninterrupted_call(tmp_path: Path) -> None:
    fit = _in_sample()
    whole = _bootstrap(fit, tmp_path / "whole", sample_limit=150, random_state=SEED)
    stopped: list[int] = []

    def stop(progress: Any) -> None:
        stopped.append(progress.completed)
        if progress.completed == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _quiet(
            lambda: refute(
                fit,
                tests=("bootstrap_stability",),
                n_replicates=BOOTSTRAP,
                checkpoint=RefutationCheckpoint(tmp_path / "part"),
                progress=stop,
                sample_limit=150,
                random_state=SEED,
            )
        )
    resumed = _bootstrap(fit, tmp_path / "part", sample_limit=150, random_state=SEED)
    _assert_same_report(whole, resumed)


def test_w7_a_resume_whose_baseline_moved_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = _in_sample()
    _bootstrap(fit, tmp_path, sample_limit=150, random_state=SEED)
    original = refute_module._sampled_baseline

    def moved(result: Any, sampled: Any, seed: int) -> Any:
        return original(result, sampled.subset(np.arange(sampled.n - 1)), seed)

    monkeypatch.setattr(refute_module, "_sampled_baseline", moved)
    with pytest.raises(CapabilityError, match="sampled baseline"):
        _bootstrap(fit, tmp_path, sample_limit=150, random_state=SEED)


def test_w7_an_unsampled_request_keeps_its_parts() -> None:
    from cleverly.validation._refute_checkpoint import request_parts

    parts = request_parts(_plain(), estimand="ate", seed=1, repeat_seeds=None)
    assert parts["sampling"] is None


# ----------------------------------------------------------------- W8 facade


def test_w8_the_facade_keys_its_cache_on_the_sample() -> None:
    frame = _point_frame()
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    fit = _quiet(
        lambda: study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            simultaneous=False,
            q_bounds=(-10.0, 10.0),
            random_state=SEED,
        )
    )
    keywords = {"tests": ("placebo",), "n_replicates": DRAWS, "random_state": SEED}
    full = _quiet(lambda: fit.diagnostics.refute(**keywords))
    sampled = _quiet(lambda: fit.diagnostics.refute(sample_limit=150, **keywords))
    assert full.sample is None
    assert sampled.sample is not None and sampled.sample.applied
    direct = _quiet(lambda: refute(fit, sample_limit=150, **keywords))
    _assert_same_report(sampled, direct)
    preview = fit.diagnostics.planned_refits(sample_limit=150, **keywords)
    assert preview.tests[0].name == "sampled_baseline"


# ----------------------------------------------------------------- W9 verdict scope


def test_w9_a_sample_below_the_registered_size_prints_no_verdict() -> None:
    report = _quiet(
        lambda: refute(
            _plain(), tests=("placebo",), sample_limit=150, random_state=SEED, n_replicates=20
        )
    )
    assert report["placebo"].passed is None
    assert "150 rows (fewer than 2,000)" in report["placebo"].evidence


def test_w9_a_sample_on_the_registered_configuration_prints_the_verdict() -> None:
    from tests.studies import refuter_calibration_point as point

    fit = point.fit_cell("T1", point.draw_law("gaussian", 4000, seed=1), seed=1)
    report = _quiet(lambda: refute(fit, tests=("placebo",), sample_limit=2000, random_state=1))
    assert report["placebo"].passed is not None


# ----------------------------------------------------------------- W10 seeds


def test_w10_the_same_seed_repeats_and_the_recorded_seed_regenerates_the_sample() -> None:
    fit = _plain()

    def call(seed: int | None) -> Any:
        return _quiet(
            lambda: refute(
                fit, tests=("subset",), sample_limit=150, random_state=seed, n_replicates=DRAWS
            )
        )

    first, second = call(SEED), call(SEED)
    _assert_same_report(first, second)
    assert first.sample == second.sample


def test_w10_an_unseeded_fit_records_the_drawn_seed() -> None:
    fit = _fit_point(_point_frame(), {"random_state": None}, {})
    first = _quiet(lambda: refute(fit, tests=("subset",), sample_limit=150, n_replicates=DRAWS))
    again = _quiet(
        lambda: refute(
            fit,
            tests=("subset",),
            sample_limit=150,
            random_state=first.random_state,
            n_replicates=DRAWS,
        )
    )
    _assert_same_report(first, again)
    assert isinstance(first.sample, RefutationSample)


def test_the_bootstrap_measurement_error_test_runs_on_the_sample() -> None:
    fit = _in_sample()
    report = _quiet(
        lambda: refute(
            fit,
            tests=("bootstrap_measurement_error",),
            bootstrap_measurement_error=BootstrapMeasurementError(("W1",)),
            measurement_error_rule=EmpiricalInclusionRule(
                alpha=2.0 / BOOTSTRAP, minimum_draws=BOOTSTRAP
            ),
            n_replicates=BOOTSTRAP,
            sample_limit=150,
            random_state=SEED,
        )
    )
    assert report.sample is not None and report.sample.applied


# ----------------------------------------------------------------- the sweep


#: Kinds whose sampled call meets a refusal that depends on the drawn rows, with its message.
SAMPLE_REFUSALS: dict[str, str] = {
    # Few subjects reach the third node, so half of them can hold no unit with code 2 there.
    "separable": "no unit with treatment code 2 at node 3",
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_kind_the_refute_row_admits_runs_a_sampled_call(kind: str) -> None:
    """Each kind ``refute`` admits either runs a sampled call or meets its pinned refusal."""
    result = KINDS[kind].build()
    request = dict(sweep_arguments(result, KINDS[kind].arguments).get("refute", {}))
    request["random_state"] = 0
    row = result.diagnostics._capability_for_arguments("refute", request)
    if not row.available or not set(row.requires_arguments) <= set(request):
        pytest.skip("the refute row does not admit this kind")
    fit = _refuted_fit(result)
    units = unit_count(fit)
    limit = max(units // 2, 2 if resolved_rule(fit) == "clusters" else 10)
    if limit >= units:
        pytest.skip("too few units to sample")
    if kind in SAMPLE_REFUSALS:
        with pytest.raises((CapabilityError, DataError), match=SAMPLE_REFUSALS[kind]):
            _quiet(lambda: result.diagnostics.refute(sample_limit=limit, **request))
        return
    report = _quiet(lambda: result.diagnostics.refute(sample_limit=limit, **request))
    reports = report.values() if hasattr(report, "values") else (report,)
    for each in reports:
        assert each.sample is not None and each.sample.applied, kind


def _refuted_fit(result: Any) -> Any:
    fits = getattr(result, "kept_fits", None)
    return next(iter(fits.values())) if fits else result
