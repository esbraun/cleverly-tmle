"""``LTMLE(stopping=)`` judges every node's score equation by the declared rule.

G4 applies the stopping rule of ``Targeting(stopping=)`` to the sequential engines.  A node's
solver receives all ``n`` rows of the panel, and the rows that are not followers contribute zero
to the score.  So a node's threshold is the standard error of its own score ``P_n S``, with
``n_u`` the distinct clusters (else rows) of the panel and ``span`` the outcome range.

Every test recomputes the threshold **by hand** from the arrays a node kept (its pseudo-outcome,
its targeted prediction, its clever covariate and its follower mask), with ``pandas`` for the
per-unit sums.  The fits cover the engines of plan I-1.3: static and dynamic plans, a continuous
outcome (span 7), clusters, weights, cross-fitting, survival, competing risks, repeated outcomes,
a long panel and a working model over regimens.  A mutation control for each part of the
context (the clusters, the span, the stacked subject) must fail the hand comparison.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    LongitudinalTreatment,
    RegimeMean,
    SDRMethod,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
)
from cleverly.exceptions import ConvergenceWarning, MethodConfigurationError, RefusalReason
from cleverly.fluctuation._score import STOP_CONTEXT_MISSING, StopContext
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR, LongitudinalData
from cleverly.longitudinal.sequential import _fluctuate_node, node_stop_context
from cleverly.methods import IPSI_NO_FLUCTUATION, IPSIMethod, sdr_no_fluctuation_refusal
from tests.unit import test_long_panel as long_panel_module
from tests.unit import test_longitudinal_msm as msm_module
from tests.unit import test_repeated_outcome as repeated_module
from tests.unit._source_mutation import mutate

sequential_module = importlib.import_module("cleverly.longitudinal.sequential")
estimator_module = importlib.import_module("cleverly.longitudinal.estimator")
longitudinal_msm_module = importlib.import_module("cleverly.longitudinal.msm")

RULES = ("scaled_var", "sample_size")

NODES: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
SURVIVAL: dict[str, Any] = {"outcome": ["Y1", "Y2"], **NODES}
COMPETING: dict[str, Any] = {"outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]}, **NODES}
PLANS = {"always": 1, "never": 0}


# ------------------------------------------------------------------------------ hand formulas


def hand_thresholds(
    contribution: np.ndarray, units: np.ndarray | None, rule: str
) -> tuple[np.ndarray, int]:
    """Thresholds of the columns of ``contribution`` from the ``tmle3`` formula, with ``pandas``.

    The per-unit influence is ``(n_u / n)`` times the unit sum of the contribution, and
    ``scaled_var`` is its ``ddof=1`` SD over ``sqrt(n_u) * min(log n_u, 10)``.
    """
    n = contribution.shape[0]
    key = np.arange(n) if units is None else np.asarray(units)
    sums = pd.DataFrame(contribution).groupby(key).sum()
    n_units = sums.shape[0]
    if rule == "sample_size":
        return np.full(contribution.shape[1], 1.0 / n_units), n_units
    influence = sums * (n_units / n)
    sd = influence.std(axis=0, ddof=1).to_numpy()
    return sd / (np.sqrt(n_units) * min(np.log(n_units), 10.0)), n_units


def hand_node(
    data: LongitudinalData, step: Any, span: float, rule: str
) -> tuple[np.ndarray, float, int]:
    """Threshold, rule ratio and unit count of one per-regimen node, from the arrays it kept.

    The node's score column is ``w h (Z - Q*)`` on its followers, ``h`` the clever covariate
    (the cumulative inverse probability, zero off the followers) and ``w`` the observation
    weight.  The contribution is ``span`` times that column, and the score is its mean over
    all ``n`` rows.
    """
    follower = np.asarray(step.trained_on, dtype=bool)
    residual = np.where(follower, step.pseudo_outcome - step.targeted, 0.0)
    part = np.asarray(data.weights) * np.asarray(step.clever) * residual
    thresholds, n_units = hand_thresholds(span * part[:, None], data.cluster, rule)
    ratio = float(abs(part.mean()) * span / thresholds[0])
    return thresholds, ratio, n_units


def every_node(result: Any) -> list[tuple[str, Any]]:
    """Every ``(label, step)`` of a per-regimen result, in report order."""
    return [(key, step) for key, fit in result.fits.items() for step in fit.steps]


def assert_hand_match(result: Any, rule: str, *, units_expected: int | None = None) -> None:
    """Every node of ``result`` carries the rule and the hand thresholds and ratio."""
    span = result.scaler.range
    nodes = every_node(result)
    assert nodes
    for key, step in nodes:
        fluctuation = step.fluctuation
        assert fluctuation.stopping == rule, (key, step.time)
        thresholds, ratio, n_units = hand_node(result.data, step, span, rule)
        np.testing.assert_allclose(fluctuation.thresholds, thresholds, rtol=1e-12, err_msg=key)
        assert fluctuation.rule_ratio == pytest.approx(ratio, abs=1e-9), (key, step.time)
        assert fluctuation.converged == (ratio <= 1.0), (key, step.time)
        if units_expected is not None:
            assert n_units == units_expected


# ------------------------------------------------------------------------------ the fits


def learners(**settings: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000, random_state=0),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000, random_state=0),
        "censoring_learner": LogisticRegression(max_iter=1000, random_state=0),
        "learner_folds": 2,
        "simultaneous": False,
        "random_state": 5,
        **settings,
    }


def end_of_study(
    rule: str, *, n: int = 300, n_folds: int = 1, cluster_size: int | None = None
) -> Any:
    frame, _ = make_longitudinal(n=n, seed=12, cluster_size=cluster_size)
    extra = {} if cluster_size is None else {"id": "id"}
    estimator = LTMLE(PLANS, n_folds=n_folds, stopping=rule, **learners())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", **NODES, **extra)


def survival(rule: str, *, n_folds: int = 1) -> Any:
    frame, _ = make_longitudinal_survival(n=300, seed=1)
    return LTMLE(PLANS, n_folds=n_folds, stopping=rule, **learners()).fit(frame, **SURVIVAL)


def competing(rule: str, *, n_folds: int = 1) -> Any:
    frame, _ = make_longitudinal_competing(n=300, seed=1)
    return LTMLE(PLANS, n_folds=n_folds, stopping=rule, **learners()).fit(frame, **COMPETING)


def repeated_panel(rule: str, composition: repeated_module.Composition, frame: pd.DataFrame) -> Any:
    """The repeated-outcome fit of ``composition``, under ``rule``."""
    plans = repeated_module.regimens(composition.plans, repeated_module.T, held=composition.held)
    regression = LinearRegression() if composition.continuous else LogisticRegression()
    estimator = LTMLE(
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
        stopping=rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            frame, **repeated_module.columns(composition, repeated_module.T, repeated=True)
        )


def end_of_study_panel(
    rule: str, composition: repeated_module.Composition, frame: pd.DataFrame
) -> Any:
    """The end-of-study fit at the last node of ``composition``'s panel, under ``rule``."""
    horizon = repeated_module.T
    plans = repeated_module.regimens(composition.plans, horizon, held=composition.held)
    regression = LinearRegression() if composition.continuous else LogisticRegression()
    estimator = LTMLE(
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
        stopping=rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, **repeated_module.columns(composition, horizon, repeated=False))


def long_panel_fit(rule: str) -> Any:
    frame = long_panel_module.long_panel(n=300, seed=2)
    data = long_panel_module.build(frame)
    regimens = [long_panel_module.Regimen("always", ("offer",) * 3)]
    estimator = LTMLE(regimens, stopping=rule, **long_panel_module.learners())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(data)


def working_model(rule: str, *, n_folds: int = 1) -> Any:
    estimator = LTMLE(
        PLANS,
        msm=msm_module.DOSE,
        n_folds=n_folds,
        simultaneous=False,
        outcome_learner=LogisticRegression(max_iter=1000, random_state=0),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=0),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=0),
        stopping=rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            msm_module.panel(120),
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=(("L1",), ("L2",)),
            censoring=("C1", "C2"),
        )


COMPOSITIONS = {composition.name: composition for composition in repeated_module.COMPOSITIONS}


# ------------------------------------------------------------------------------ every engine


@pytest.mark.parametrize("rule", RULES)
class TestEveryPerRegimenEngine:
    def test_a_single_fold_fit_keeps_the_hand_threshold_at_every_node(self, rule: str) -> None:
        result = end_of_study(rule)
        assert_hand_match(result, rule, units_expected=result.data.n)

    def test_a_cross_fitted_fit_keeps_the_hand_threshold_at_every_pooled_node(
        self, rule: str
    ) -> None:
        result = end_of_study(rule, n_folds=3)
        assert not result.folds.is_single
        assert_hand_match(result, rule, units_expected=result.data.n)

    def test_survival(self, rule: str) -> None:
        result = survival(rule)
        assert result.data.is_survival
        assert_hand_match(result, rule)

    def test_cross_fitted_survival(self, rule: str) -> None:
        assert_hand_match(survival(rule, n_folds=3), rule)

    def test_competing_risks_judge_every_cause(self, rule: str) -> None:
        result = competing(rule)
        assert result.data.is_competing
        assert len(result.fits) >= 4
        assert_hand_match(result, rule)

    def test_cross_fitted_competing_risks(self, rule: str) -> None:
        assert_hand_match(competing(rule, n_folds=3), rule)

    @pytest.mark.parametrize("name", list(COMPOSITIONS))
    def test_every_supported_composition(self, rule: str, name: str) -> None:
        """Static, dynamic, policy, MTP, continuous, clustered, weighted, held, known, folds."""
        composition = COMPOSITIONS[name]
        frame = repeated_module.panel(continuous=composition.continuous, held=composition.held)
        result = end_of_study_panel(rule, composition, frame)
        if composition.continuous:
            assert result.scaler.range == pytest.approx(7.0)
        if composition.weights:
            assert float(np.ptp(result.data.weights)) > 0.1
        if composition.cluster:
            assert result.data.cluster is not None
        assert_hand_match(result, rule)

    def test_a_repeated_outcome_applies_the_rule_at_every_horizon(self, rule: str) -> None:
        frame = repeated_module.panel()
        result = repeated_panel(rule, COMPOSITIONS["static"], frame)
        assert result.data.is_repeated
        horizons = {fit.horizon for fit in result.fits.values()}
        assert horizons == {1, 2, 3}
        assert_hand_match(result, rule)

    def test_a_cross_fitted_repeated_outcome(self, rule: str) -> None:
        frame = repeated_module.panel()
        assert_hand_match(repeated_panel(rule, COMPOSITIONS["static_folds"], frame), rule)

    def test_a_long_panel(self, rule: str) -> None:
        result = long_panel_fit(rule)
        assert result.data.panel is not None
        assert_hand_match(result, rule)


# ------------------------------------------------------------------------------ clusters


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("n_folds", [1, 3])
class TestClusters:
    def test_n_u_counts_the_clusters_of_the_panel(self, rule: str, n_folds: int) -> None:
        result = end_of_study(rule, n_folds=n_folds, cluster_size=10)
        clusters = int(np.unique(result.data.cluster).size)
        assert clusters == 30
        assert_hand_match(result, rule, units_expected=clusters)
        for _, step in every_node(result):
            if rule == "sample_size":
                np.testing.assert_allclose(step.fluctuation.thresholds, 1.0 / clusters)

    def test_clusters_move_the_scaled_var_threshold_off_the_row_threshold(
        self, rule: str, n_folds: int
    ) -> None:
        clustered = end_of_study(rule, n_folds=n_folds, cluster_size=10)
        rows = end_of_study(rule, n_folds=n_folds)
        # 300 rows and 30 clusters: the same rule must not read the same threshold.
        left = clustered.fits["always"].steps[0].fluctuation.thresholds
        right = rows.fits["always"].steps[0].fluctuation.thresholds
        assert abs(float(left[0]) - float(right[0])) > 1e-3 * float(right[0])


class TestClusterMutationControls:
    def test_the_clustered_comparison_fails_when_the_clusters_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clean = end_of_study("scaled_var", cluster_size=10)
        assert_hand_match(clean, "scaled_var")
        mutate(
            monkeypatch,
            sequential_module,
            sequential_module,
            "node_stop_context",
            [("StopContext(data.cluster, scaler.range)", "StopContext(None, scaler.range)")],
        )
        broken = end_of_study("scaled_var", cluster_size=10)
        with pytest.raises(AssertionError):
            assert_hand_match(broken, "scaled_var")

    def test_the_unclustered_comparison_does_not_see_that_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows are the units without clusters, so the mutation is a no-op there."""
        mutate(
            monkeypatch,
            sequential_module,
            sequential_module,
            "node_stop_context",
            [("StopContext(data.cluster, scaler.range)", "StopContext(None, scaler.range)")],
        )
        assert_hand_match(end_of_study("scaled_var"), "scaled_var")

    def test_the_comparison_fails_when_the_span_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = repeated_module.panel(continuous=True)
        assert_hand_match(
            end_of_study_panel("scaled_var", COMPOSITIONS["continuous"], frame), "scaled_var"
        )
        mutate(
            monkeypatch,
            sequential_module,
            sequential_module,
            "node_stop_context",
            [("StopContext(data.cluster, scaler.range)", "StopContext(data.cluster, 1.0)")],
        )
        broken = end_of_study_panel("scaled_var", COMPOSITIONS["continuous"], frame)
        with pytest.raises(AssertionError):
            assert_hand_match(broken, "scaled_var")


# ------------------------------------------------------------------------------ working model


def hand_stacked_node(result: Any, fit: Any, time: int, span: float, rule: str) -> tuple[Any, ...]:
    """The stacked node ``time`` of a working-model fit, from the arrays its cells kept."""
    model = fit.model
    live = [k for k, cell in enumerate(model.cells) if cell.horizon >= time]
    design = fit.fluctuation_design
    data = result.data
    columns = []
    for k in live:
        step = fit.fits[k].steps[time - 1]
        follower = np.asarray(step.trained_on, dtype=bool)
        residual = np.where(follower, step.pseudo_outcome - step.targeted, 0.0)
        loss = np.asarray(data.weights) * model.weights[:, k] * np.asarray(step.clever)
        columns.append(span * (loss * residual)[:, None] * design[:, k, :])
    stacked = np.concatenate(columns)
    subject = np.arange(data.n) if data.cluster is None else np.asarray(data.cluster)
    thresholds, n_units = hand_thresholds(stacked, np.tile(subject, len(live)), rule)
    # The score is the mean of the stacked columns, and the contribution already carries the span.
    score = np.abs(stacked.mean(axis=0))
    return thresholds, float(np.max(score / thresholds)), n_units, stacked.shape[0]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("n_folds", [1, 3])
class TestWorkingModel:
    def test_each_stacked_node_counts_a_subject_once(self, rule: str, n_folds: int) -> None:
        result = working_model(rule, n_folds=n_folds)
        (fit,) = result.msm_fits
        n_times = result.data.n_times
        assert len(fit.nodes) == n_times
        for position, node in enumerate(fit.nodes):
            time = n_times - position
            thresholds, ratio, n_units, rows = hand_stacked_node(
                result, fit, time, result.scaler.range, rule
            )
            assert n_units == result.data.n
            assert rows == len(fit.model.cells) * result.data.n
            assert node.stopping == rule
            np.testing.assert_allclose(node.thresholds, thresholds, rtol=1e-12)
            assert node.rule_ratio == pytest.approx(ratio, abs=1e-9)
            assert node.converged == (ratio <= 1.0)

    def test_the_cells_of_a_working_model_carry_the_rule(self, rule: str, n_folds: int) -> None:
        result = working_model(rule, n_folds=n_folds)
        for fit in result.fits.values():
            assert all(step.fluctuation.stopping == rule for step in fit.steps)


def test_the_stacked_threshold_fails_when_the_copies_count_as_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = working_model("scaled_var")
    (fit,) = clean.msm_fits
    thresholds, *_ = hand_stacked_node(clean, fit, 2, clean.scaler.range, "scaled_var")
    np.testing.assert_allclose(fit.nodes[0].thresholds, thresholds, rtol=1e-12)
    mutate(
        monkeypatch,
        longitudinal_msm_module,
        estimator_module,
        "fit_regimens_msm",
        [("np.tile(subject, len(live))", "np.arange(len(live) * data.n)")],
    )
    broken = working_model("scaled_var")
    (broken_fit,) = broken.msm_fits
    assert not np.allclose(broken_fit.nodes[0].thresholds, thresholds, rtol=1e-6)


def test_a_clustered_working_model_counts_a_cluster_once() -> None:
    frame, _ = make_longitudinal(n=240, seed=3, cluster_size=8)
    estimator = LTMLE(
        PLANS,
        msm=msm_module.DOSE,
        n_folds=1,
        stopping="scaled_var",
        **learners(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(frame, outcome="Y", **NODES, id="id")
    (fit,) = result.msm_fits
    clusters = int(np.unique(result.data.cluster).size)
    for position, node in enumerate(fit.nodes):
        time = result.data.n_times - position
        thresholds, _, n_units, _ = hand_stacked_node(
            result, fit, time, result.scaler.range, "scaled_var"
        )
        assert n_units == clusters
        np.testing.assert_allclose(node.thresholds, thresholds, rtol=1e-12)


# ------------------------------------------------------------------------------ the node solve


def crafted_node(n: int = 400, seed: int = 4) -> dict[str, Any]:
    """A node whose offset is confidently wrong on a third of the rows.

    The inner Newton solve reaches the score equation, and the shrink into ``[1 - alpha, alpha]``
    then moves the confident predictions, so the shipped test needs three outer iterations.  A
    declared rule sees a score far inside its threshold after the first.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    truth = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * z)))
    pseudo = rng.binomial(1, truth).astype(float)
    offset = truth.copy()
    wrong = rng.random(n) < 0.3
    offset[wrong] = np.where(pseudo[wrong] == 1.0, 1e-7, 1.0 - 1e-7)
    return {
        "pseudo_outcome": pseudo,
        "initial": offset,
        "loss_weights": rng.uniform(0.5, 2.0, size=n),
        "fitted_on": np.ones(n, dtype=bool),
        "label": "r",
        "time": 1,
        "alpha": 0.9995,
        "max_iter": 20,
        "tol": 1e-10,
    }


class TestTheNodeSolve:
    def test_a_declared_rule_without_a_context_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match=STOP_CONTEXT_MISSING[:40]):
            _fluctuate_node(**crafted_node(), stopping="sample_size")

    @pytest.mark.parametrize("rule", RULES)
    def test_a_node_stops_at_the_first_iterate_that_meets_the_rule(self, rule: str) -> None:
        node = crafted_node()
        stop = StopContext(None, 1.0)
        shipped = _fluctuate_node(**node)
        ruled = _fluctuate_node(**node, stopping=rule, stop=stop)
        assert shipped.n_iter == 3 and ruled.n_iter == 1
        assert ruled.converged and shipped.converged
        assert ruled.rule_ratio is not None and ruled.rule_ratio <= 1.0
        # The witness: the rule moved the fit, far above the float noise of a reordered sum.
        assert float(np.max(np.abs(ruled.epsilon - shipped.epsilon))) > 1e-4
        # The ruled walk is the shipped walk truncated: its last iterate is the shipped trace's.
        assert ruled.trace == shipped.trace[: ruled.n_iter + 1]
        thresholds, ratio = hand_crafted(node, ruled, rule)
        np.testing.assert_allclose(ruled.thresholds, thresholds, rtol=1e-12)
        assert ruled.rule_ratio == pytest.approx(ratio, rel=1e-6)

    def test_the_default_path_is_the_shipped_solve(self) -> None:
        node = crafted_node()
        plain = _fluctuate_node(**node)
        named = _fluctuate_node(**node, stopping="relative", stop=StopContext(None, 1.0))
        np.testing.assert_array_equal(plain.epsilon, named.epsilon)
        np.testing.assert_array_equal(plain.targeted.observed, named.targeted.observed)
        assert named.stopping == "relative" and named.thresholds is None
        assert named.rule_ratio is None

    @pytest.mark.parametrize("rule", RULES)
    def test_a_constant_node_meets_any_rule_without_a_solve(self, rule: str) -> None:
        n = 50
        node = crafted_node(n)
        node["pseudo_outcome"] = np.zeros(n)
        node["initial"] = np.zeros(n)
        fluctuation = _fluctuate_node(**node, stopping=rule, stop=StopContext(None, 1.0))
        assert fluctuation.converged and fluctuation.n_iter == 0
        assert fluctuation.stopping == rule
        assert fluctuation.rule_ratio == 0.0
        assert fluctuation.thresholds is not None

    def test_a_node_that_misses_the_rule_warns_with_the_pinned_text(self) -> None:
        # A span of 1e4 puts the score, in the outcome's units, far above 1 / n_u.
        node = {**crafted_node(), "max_iter": 1}
        stop = StopContext(None, 1e4)
        with pytest.warns(
            ConvergenceWarning,
            match=r"targeting step did not meet the sample_size stopping rule after 1 iteration\(s\)",
        ):
            ruled = _fluctuate_node(**node, stopping="sample_size", stop=stop)
        assert not ruled.converged
        assert ruled.rule_ratio is not None and ruled.rule_ratio > 1.0

    def test_the_context_a_fit_builds_is_the_clusters_and_the_range(self) -> None:
        frame, _ = make_longitudinal(n=80, seed=2, cluster_size=8)
        data = LongitudinalData.from_frame(frame, outcome="Y", id="id", **NODES)
        from cleverly.utils.bounds import OutcomeScaler

        stop = node_stop_context(data, OutcomeScaler(-1.0, 6.0))
        assert stop.span == 7.0
        np.testing.assert_array_equal(stop.units, data.cluster)


def hand_crafted(node: dict[str, Any], fluctuation: Any, rule: str) -> tuple[np.ndarray, float]:
    """Thresholds and ratio of a crafted node at its reported fit."""
    residual = node["pseudo_outcome"] - fluctuation.targeted.observed
    part = node["loss_weights"] * residual
    thresholds, _ = hand_thresholds(part[:, None], None, rule)
    return thresholds, float(abs(part.mean()) / thresholds[0])


# ------------------------------------------------------------------------------ plumbing


class TestPlumbing:
    def test_a_fit_with_the_shipped_rule_equals_a_fit_without_it(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
        plain = LTMLE(PLANS, n_folds=3, **learners()).fit(frame, outcome="Y", **NODES)
        named = LTMLE(PLANS, n_folds=3, stopping="relative", **learners()).fit(
            frame, outcome="Y", **NODES
        )
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
            np.testing.assert_array_equal(
                estimate.influence_curve, named.estimates[key].influence_curve
            )
        for _, step in every_node(named):
            assert step.fluctuation.stopping == "relative"
            assert step.fluctuation.thresholds is None
            assert step.fluctuation.rule_ratio is None

    def test_the_verdict_follows_the_declared_rule_and_not_the_tolerance(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        settings = {"max_iter": 1, "tol": 1e-30, **learners()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shipped = LTMLE(PLANS, **settings).fit(frame, outcome="Y", **NODES)
            ruled = LTMLE(PLANS, stopping="sample_size", **settings).fit(
                frame, outcome="Y", **NODES
            )
        # One Newton step cannot reach a tolerance of 1e-30, but it meets the rule.
        assert not shipped.converged
        assert ruled.converged

    def test_an_invalid_rule_is_refused_by_name(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"stopping must be 'relative', 'scaled_var' or 'sample_size'; got 'bogus'",
        ):
            LTMLE(PLANS, stopping="bogus")  # type: ignore[arg-type]

    def test_the_engine_records_the_rule_in_its_replay_recipe(self) -> None:
        result = end_of_study("sample_size", n=120)
        assert result.replay_recipe is not None
        assert result.replay_recipe.stopping == "sample_size"
        assert end_of_study("relative", n=120).replay_recipe.stopping == "relative"

    def test_a_truncation_replay_repeats_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = end_of_study("scaled_var", n=200)
        seen: list[tuple[str, Any]] = []
        original = estimator_module._run_recursion

        def spy(*args: Any, **kwargs: Any) -> Any:
            fits, msm_fits = original(*args, **kwargs)
            seen.append((kwargs["stopping"], fits))
            return fits, msm_fits

        monkeypatch.setattr(estimator_module, "_run_recursion", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve = result.diagnostics.truncation_curve([result.config.g_bounds, (0.05, 1.0)])
        assert len(curve) >= 2
        assert seen
        for rule, fits in seen:
            assert rule == "scaled_var"
            for fit in fits.values():
                assert all(step.fluctuation.stopping == "scaled_var" for step in fit.steps)
        # At the fitted bound the replay equals the fit.
        replayed = seen[0][1]
        for key, fit in result.fits.items():
            np.testing.assert_allclose(replayed[key].psi_scaled, fit.psi_scaled, rtol=1e-12)

    def test_a_cluster_bootstrap_refits_under_the_rule(self) -> None:
        frame, _ = make_longitudinal(n=160, seed=4, cluster_size=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                PLANS, n_folds=1, stopping="sample_size", n_bootstrap=4, **learners()
            ).fit(frame, outcome="Y", **NODES, id="id")
        assert result.bootstrap is not None or result.estimates

    def test_the_study_path_carries_the_rule_to_the_engine(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=12)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.identify(RegimeMean(PLANS)).estimate(
                stopping="scaled_var",
                cross_fit=False,
                simultaneous=False,
                random_state=5,
                outcome_learner=LogisticRegression(max_iter=1000),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
            )
        assert result.replay_recipe.stopping == "scaled_var"
        assert_hand_match(result, "scaled_var")

    def test_the_method_object_carries_the_rule_and_the_default_rule(self) -> None:
        method = TMLEMethod(targeting=Targeting(stopping="sample_size"))
        assert method.estimator_kwargs(longitudinal=True)["stopping"] == "sample_size"
        assert TMLEMethod().estimator_kwargs(longitudinal=True)["stopping"] == "relative"
        engine = LTMLE(PLANS, **method.estimator_kwargs(longitudinal=True))
        assert engine.stopping == "sample_size"


# ------------------------------------------------------------------------------ R-abc2, R-abc3


class TestRefusals:
    def test_r_abc2_sdr_refuses_a_stopping_rule_by_name(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            SDR(PLANS, stopping="scaled_var")
        assert str(raised.value) == sdr_no_fluctuation_refusal("stopping").message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_abc2_the_sdr_method_refuses_a_declared_rule(self) -> None:
        method = SDRMethod(targeting=Targeting(stopping="sample_size"))
        with pytest.raises(MethodConfigurationError) as raised:
            method.estimator_kwargs(longitudinal=True)
        assert str(raised.value) == sdr_no_fluctuation_refusal("stopping").message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_abc2_the_sdr_method_by_default_passes_no_rule(self) -> None:
        kwargs = SDRMethod().estimator_kwargs(longitudinal=True)
        assert "stopping" not in kwargs
        SDR(PLANS, **kwargs)

    def test_r_abc3_ipsi_refuses_a_stopping_rule(self) -> None:
        with pytest.raises(MethodConfigurationError) as raised:
            IPSI([Incremental(2.0)], stopping="scaled_var")
        assert str(raised.value) == IPSI_NO_FLUCTUATION.message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_abc3_the_ipsi_method_refuses_a_declared_rule(self) -> None:
        method = IPSIMethod(targeting=Targeting(stopping="sample_size"))
        with pytest.raises(MethodConfigurationError) as raised:
            method.estimator_kwargs(longitudinal=True)
        assert str(raised.value) == IPSI_NO_FLUCTUATION.message()
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_abc3_the_ipsi_method_by_default_passes_no_rule(self) -> None:
        kwargs = IPSIMethod().estimator_kwargs(longitudinal=True)
        assert "stopping" not in kwargs

    def test_the_sdr_and_ipsi_lists_name_the_rule(self) -> None:
        assert "stopping" in estimator_module.SDR_REFUSED_SETTINGS
        assert "stopping" in estimator_module.IPSI_REFUSED_SETTINGS

    def test_ltmle_still_refuses_a_point_option_other_than_the_rule(self) -> None:
        with pytest.raises(MethodConfigurationError, match="point-treatment option"):
            TMLEMethod(targeting=Targeting(step_size=0.1)).estimator_kwargs(longitudinal=True)
