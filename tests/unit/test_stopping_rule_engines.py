"""The stopping context reaches every point solve, and the alternating exits read a current ratio.

``Targeting(stopping=)`` is a rule about the **outcome** score equation.  G3a built the rule and
the solvers.  This module covers what G3b wires around them:

* ``TMLE._retarget_detailed`` builds the :class:`~cleverly.fluctuation._score.StopContext` once,
  and every point group (arm means, ATT, multi-arm, missing outcome, clusters, weights, strata,
  shift, point IPSI, linear and logit MSM, one-step, cross-fitted, fold-targeted, CDE, DRTMLE in
  both orders, the missing-outcome reduction, C-TMLE) fits under a declared rule without the
  loud ``STOP_CONTEXT_MISSING`` failure;
* a fold solve counts the units of its own rows and a C-TMLE candidate counts the units of its
  training rows (``sample_size`` thresholds are ``1 / n_u``);
* the truncation curve and the bootstrap run under ``stopping="scaled_var"``;
* the outer exits of the alternating engines judge the outcome equation by a rule ratio computed
  at the pair they judge (rev 3.1 F4, rev 3.2 F1).  Every test below recomputes that ratio **by
  hand**, with ``pandas`` for the per-unit sums, from the pair of the round it audits.

The shipped ``relative`` ruler can never exit these fits: the one-step walk stops as soon as the
rule is met, which leaves the outcome ``relative_score_norm`` many orders above ``tol``.  The plan
builds that regime with ``step_bound``, which is G6; ``targeting="one_step"`` builds it today.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model as lm

from cleverly.datasets import (
    make_binary_outcome,
    make_clustered,
    make_instrument,
    make_linear_ate,
    make_missing_outcome,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators import targeting as targeting_module
from cleverly.estimators.drtmle import DRTMLE as DRTMLEClass
from cleverly.estimators.targeting import (
    _STALL_FACTOR,
    TargetingSpec,
    _missing_outcome_state,
    _outcome_ruler,
    build_submodel,
)
from cleverly.fluctuation import InitialFit, mean_submodel, solve_fluctuation, solve_one_step
from cleverly.fluctuation._score import StopContext
from cleverly.fluctuation.reduced import missing_outcome_outcome_submodel
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from cleverly.utils.bounds import expit
from tests.conftest import (
    SELECTOR_CONFIGS,
    linear_drtmle,
    linear_in_sample,
    mean_one_weights,
)
from tests.unit._direct_effect_support import COVARIATES as CDE_COVARIATES
from tests.unit._direct_effect_support import cde_frame
from tests.unit._source_mutation import mutate

tmle_module = importlib.import_module("cleverly.estimators.tmle")
ctmle_module = importlib.import_module("cleverly.estimators.ctmle")

RULES = ("scaled_var", "sample_size")


# ------------------------------------------------------------------------------ hand formulas


def hand_state(
    y: np.ndarray,
    q_observed: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray,
    observed: np.ndarray,
    *,
    units: np.ndarray | None,
    span: float,
    rule: str,
    arm_curves: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Thresholds and the rule ratio of one pair, from the ``tmle3`` formula, with ``pandas``.

    The per-column contribution is ``span * w h (Y - Q*)``, zero where the outcome is
    unobserved.  A column listed in ``arm_curves`` takes the full arm curve ``span * (w h (Y -
    Q*) + Q*_a(W) - mean(Q*_a))``.  The per-unit influence is ``(n_u / n)`` times the unit sum,
    and ``scaled_var`` is its ``ddof=1`` SD over ``sqrt(n_u) * min(log n_u, 10)``.
    """
    n = y.shape[0]
    residual = np.where(observed, y - q_observed, 0.0)
    part = (weights * residual)[:, None] * h
    score = part.mean(axis=0)
    contribution = span * part
    for column, q_arm in (arm_curves or {}).items():
        contribution[:, column] = span * (part[:, column] + q_arm - q_arm.mean())
    key = np.arange(n) if units is None else np.asarray(units)
    sums = pd.DataFrame(contribution).groupby(key).sum()
    n_units = sums.shape[0]
    influence = sums * (n_units / n)
    if rule == "sample_size":
        thresholds = np.full(h.shape[1], 1.0 / n_units)
    else:
        sd = influence.std(axis=0, ddof=1).to_numpy()
        thresholds = sd / (np.sqrt(n_units) * min(np.log(n_units), 10.0))
    return thresholds, float(np.max(np.abs(score) * span / thresholds))


def hand_arm_h(treatment: np.ndarray, g1: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """``1(A = a) / g_a`` for a two-arm complete-data fit, ``g`` truncated to ``bounds``."""
    g1_bounded = np.clip(g1, bounds[0], bounds[1])
    return np.column_stack(
        [(treatment == 0.0) / (1.0 - g1_bounded), (treatment == 1.0) / g1_bounded]
    )


class Log:
    """Wrap functions so a test sees every call, in order, with its arguments and its result."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any], Any]] = []
        self._monkeypatch = monkeypatch

    def wrap(self, owner: Any, name: str, tag: str) -> None:
        original = getattr(owner, name)
        events = self.events

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            events.append((tag, args, kwargs, result))
            return result

        self._monkeypatch.setattr(owner, name, wrapper)

    def of(self, tag: str) -> list[tuple[str, tuple[Any, ...], dict[str, Any], Any]]:
        return [event for event in self.events if event[0] == tag]


# ------------------------------------------------------------------- every point group fits


@functools.cache
def _lin() -> Any:
    return make_linear_ate(n=400, seed=2)[0]


@functools.cache
def _missing() -> Any:
    return make_missing_outcome(n=400, seed=4)[0]


@functools.cache
def _clustered() -> Any:
    return make_clustered(n=400, cluster_size=10, seed=7)[0]


@functools.cache
def _dose() -> Any:
    return make_shift_dose(n=300, seed=0)[0]


@functools.cache
def _binary() -> Any:
    return make_binary_outcome(n=400, seed=17)[0]


def _stratified() -> Any:
    frame = _lin().copy()
    frame["V"] = np.where(frame["W1"] >= frame["W1"].median(), "high", "low")
    return frame


def _bounded() -> Any:
    from tests.conftest import bounded_frame

    return bounded_frame(n=300, seed=1)[0]


STRATA_COVARIATES = ("W1", "W2", "W3", "W4", "V")


def _fit(rule: str, frame: Any, *, columns: dict[str, Any] | None = None, **settings: Any) -> Any:
    estimator = TMLE(**linear_in_sample(stopping=rule, **settings))
    result = estimator.fit(frame, outcome="Y", treatment="A", **(columns or {}))
    return result.single()


#: One builder per kind of point fit: ``(frame, columns, settings, units column or None)``.
KINDS: dict[str, Callable[[], tuple[Any, dict[str, Any], dict[str, Any], str | None]]] = {
    "arm-means": lambda: (_lin(), {}, {}, None),
    "multi-arm": lambda: (make_multi_arm(n=600, seed=0)[0], {}, {}, None),
    "missing-outcome": lambda: (_missing(), {"delta": "Delta"}, {}, None),
    "clusters": lambda: (_clustered(), {"id": "cluster"}, {}, "cluster"),
    "weights": lambda: (
        _lin().assign(weight=mean_one_weights(len(_lin()))),
        {"weights": "weight"},
        {},
        None,
    ),
    "strata": lambda: (
        _stratified(),
        {"covariates": STRATA_COVARIATES, "strata": ("V",)},
        {},
        None,
    ),
    "one-step": lambda: (_lin(), {}, {"targeting": "one_step"}, None),
    "shift": lambda: (_dose(), {}, {"policies": [Shift(0.5, cap=5.0)]}, None),
    "ipsi": lambda: (_lin(), {}, {"incremental": [Incremental(2.0)]}, None),
    "ipsi-strata": lambda: (
        _stratified(),
        {"covariates": STRATA_COVARIATES, "strata": ("V",)},
        {"incremental": [Incremental(2.0)]},
        None,
    ),
    "msm-identity": lambda: (_lin(), {}, {"msm": MSM.linear()}, None),
    "msm-logit": lambda: (_binary(), {}, {"msm": MSM.linear(link="logit")}, None),
    "msm-logit-strata": lambda: (
        _binary().assign(V=lambda f: np.where(f["W1"] >= f["W1"].median(), "high", "low")),
        {"covariates": ("W1", "W2", "W3", "V"), "strata": ("V",)},
        {"msm": MSM.linear(link="logit")},
        None,
    ),
    "cross-fitted": lambda: (
        _bounded(),
        {},
        {"cross_fit": True, "n_folds": 3, "learner_folds": 3, "q_bounds": (0.0, 1.0)},
        None,
    ),
    "cv-evaluation": lambda: (
        _bounded(),
        {},
        {
            "cross_fit": True,
            "n_folds": 3,
            "learner_folds": 3,
            "q_bounds": (0.0, 1.0),
            "cv_evaluation": True,
        },
        None,
    ),
}


def _expected_units(frame: Any, units_column: str | None) -> int:
    return len(frame) if units_column is None else int(frame[units_column].nunique())


class TestEveryPointGroupFitsUnderADeclaredRule:
    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("kind", sorted(KINDS))
    def test_the_fit_runs_and_each_threshold_counts_the_units(self, kind: str, rule: str) -> None:
        frame, columns, settings, units_column = KINDS[kind]()
        result = _fit(rule, frame, columns=columns, **settings)
        n_units = _expected_units(frame, units_column)
        assert result.fluctuations
        for group, fluctuation in result.fluctuations.items():
            assert fluctuation.stopping == rule, group
            assert fluctuation.thresholds is not None, group
            assert fluctuation.rule_ratio is not None, group
            assert np.all(np.isfinite(fluctuation.thresholds)) and np.all(
                fluctuation.thresholds > 0.0
            ), group
            if rule == "sample_size":
                np.testing.assert_allclose(fluctuation.thresholds, 1.0 / n_units, rtol=1e-12)
        assert all(np.isfinite(e.psi) for e in result.estimates.values())

    @pytest.mark.parametrize("rule", RULES)
    def test_the_controlled_direct_effect_fits(self, rule: str) -> None:
        estimator = TMLE(**linear_in_sample(stopping=rule))
        results = estimator.fit(
            cde_frame(), outcome="Y", treatment="A", covariates=CDE_COVARIATES, intermediate="Z"
        )
        for level in (0.0, 1.0):
            for fluctuation in results[level].fluctuations.values():
                assert fluctuation.stopping == rule and fluctuation.thresholds is not None

    @pytest.mark.parametrize("rule", RULES)
    def test_the_missing_data_composite_route_fits(self, rule: str) -> None:
        # A missing treatment value routes the fit through the composite data, which replaces
        # the container the solvers see; the context is built on the replacement.
        frame = _lin().copy()
        frame["TD"] = 1.0
        frame.loc[::17, "TD"] = 0.0
        frame.loc[::17, "A"] = np.nan
        estimator = TMLE(**linear_in_sample(stopping=rule))
        result = estimator.fit(frame, outcome="Y", treatment="A", treatment_delta="TD").single()
        assert all(f.stopping == rule for f in result.fluctuations.values())

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("order", ["drtmle", "benkeser"])
    def test_drtmle_fits_in_both_orders(self, rule: str, order: str) -> None:
        estimator = DRTMLE(
            **linear_drtmle(
                stopping=rule,
                cross_fit=True,
                n_folds=3,
                q_bounds=(-30.0, 30.0),
                update_order=order,
                estimands=("ate", "ey1", "ey0"),
            )
        )
        result = estimator.fit(_lin(), outcome="Y", treatment="A").single()
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.stopping == rule and fluctuation.reduction is not None
        assert len(fluctuation.reduction.rule_trace) == len(fluctuation.reduction.trace)

    @pytest.mark.parametrize("rule", RULES)
    def test_drtmle_fits_the_missing_outcome_reduction(self, rule: str) -> None:
        frame = _trial()
        result = _missing_drtmle(rule).fit(
            frame, outcome="Y", treatment="A", delta="Delta", covariates=["W1", "W2"]
        )
        fluctuation = result.single().fluctuations["mean"]
        assert fluctuation.stopping == rule and fluctuation.rule_ratio is not None
        assert fluctuation.reduction is not None and fluctuation.reduction.rule_trace

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("strategy", ["greedy", "ordered", "discrete"])
    def test_ctmle_fits_every_selector(self, rule: str, strategy: str) -> None:
        frame, _ = make_instrument(n=300, seed=44)
        estimator = CTMLE(
            strategy=strategy,  # type: ignore[arg-type]
            **linear_in_sample(stopping=rule, estimands=("ate",)),
            **SELECTOR_CONFIGS[strategy],
        )
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
        assert result.fluctuations["mean"].stopping == rule
        assert result.ctmle_selection is not None

    def test_the_outcome_adaptive_selector_fits(self) -> None:
        frame, _ = make_instrument(n=300, seed=44)
        estimator = CTMLE(
            strategy="oat", **linear_in_sample(stopping="scaled_var", estimands=("ate",))
        )
        assert estimator.fit(frame, outcome="Y", treatment="A").single().fluctuations

    def test_the_context_is_none_under_the_shipped_rule(self) -> None:
        estimator = TMLE(**linear_in_sample())
        data = estimator._prepare(
            _lin(),
            outcome="Y",
            treatment="A",
            covariates=None,
            delta=None,
            weights=None,
            id=None,
            intermediate=None,
        )
        nuisance = estimator.fit(_lin(), outcome="Y", treatment="A").single().nuisance
        assert estimator._stop_context(data, nuisance) is None
        declared = TMLE(**linear_in_sample(stopping="scaled_var"))
        context = declared._stop_context(data, nuisance)
        assert context is not None and context.units is None
        assert context.span == nuisance.scaler.upper - nuisance.scaler.lower


class TestTheContextCarriesTheUnitsAndTheSpan:
    def test_a_clustered_fit_hands_the_solver_its_clusters_and_the_outcome_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = Log(monkeypatch)
        log.wrap(tmle_module, "solve_submodel", "solve")
        frame = _clustered()
        result = _fit("scaled_var", frame, columns={"id": "cluster"})
        # The first solve is the arm-mean group; ATT and ATC follow.
        _, args, kwargs, fluctuation = log.of("solve")[0]
        scaled, _, submodel, weights, observed = args[:5]
        stop = kwargs["stop"]
        assert stop is not None
        codes = pd.factorize(frame["cluster"])[0]
        assert np.array_equal(np.asarray(stop.units), codes)
        # The span is the padded observed range, derived here from the outcome itself.
        spread = float(frame["Y"].max() - frame["Y"].min())
        assert stop.span == pytest.approx(1.2 * spread, rel=1e-12)
        arm_curves = {
            column: fluctuation.targeted.arms[level]
            for level, column in submodel.arm_columns.items()
        }
        thresholds, ratio = hand_state(
            scaled,
            fluctuation.targeted.observed,
            submodel.observed,
            weights,
            observed,
            units=codes,
            span=1.2 * spread,
            rule="scaled_var",
            arm_curves=arm_curves,
        )
        np.testing.assert_allclose(fluctuation.thresholds, thresholds, rtol=1e-12)
        assert fluctuation.rule_ratio == pytest.approx(ratio, rel=1e-12)
        assert result.fluctuations["mean"].thresholds is not None


# ----------------------------------------------------------------------- fold and C-TMLE n_u


def _clustered_units(frame: Any) -> np.ndarray:
    return np.asarray(frame["cluster"])


class TestFoldSolvesCountTheirOwnUnits:
    @pytest.fixture(scope="class")
    def frame(self) -> Any:
        # Bounded outcome, so a cross-fitted fit may declare its support.
        rng = np.random.default_rng(3)
        n_units, per_unit = 48, 8
        n = n_units * per_unit
        cluster = np.repeat(np.arange(n_units), per_unit)
        w = rng.normal(size=n) + np.repeat(rng.normal(size=n_units), per_unit)
        a = rng.binomial(1, expit(0.4 * w)).astype(float)
        y = np.clip(0.4 + 0.1 * a + 0.1 * np.tanh(w) + 0.1 * rng.normal(size=n), 0.01, 0.99)
        return pd.DataFrame({"Y": y, "A": a, "W1": w, "W2": rng.normal(size=n), "cluster": cluster})

    def _fit(self, frame: Any, **settings: Any) -> Any:
        estimator = TMLE(
            **linear_in_sample(
                stopping="sample_size",
                cross_fit=True,
                n_folds=4,
                learner_folds=4,
                q_bounds=(0.0, 1.0),
                **settings,
            )
        )
        return estimator.fit(frame, outcome="Y", treatment="A", id="cluster").single()

    def test_a_fold_targeted_fit_uses_the_units_of_each_validation_fold(self, frame: Any) -> None:
        result = self._fit(frame, targeting_scheme="fold")
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.folds
        cluster = _clustered_units(frame)
        counts = []
        for record in fluctuation.folds:
            n_units = np.unique(cluster[record.index]).size
            counts.append(n_units)
            assert record.thresholds is not None
            np.testing.assert_allclose(record.thresholds, 1.0 / n_units, rtol=1e-12)
        # The fold bound is not the pooled one: every fold holds fewer units than the sample.
        assert max(counts) < frame["cluster"].nunique()
        assert fluctuation.thresholds is None
        assert fluctuation.stopping == "sample_size"
        assert fluctuation.rule_ratio == max(r.rule_ratio for r in fluctuation.folds)
        assert fluctuation.converged == all(r.converged for r in fluctuation.folds)

    def test_a_fold_targeted_projection_uses_the_units_of_each_fold(self, frame: Any) -> None:
        result = self._fit(frame, targeting_scheme="fold", msm=MSM.linear(link="logit"))
        fluctuation = result.fluctuations["msm"]
        cluster = _clustered_units(frame)
        assert fluctuation.folds
        for record in fluctuation.folds:
            np.testing.assert_allclose(
                record.thresholds, 1.0 / np.unique(cluster[record.index]).size, rtol=1e-12
            )

    def test_the_common_update_fit_uses_every_unit(self, frame: Any) -> None:
        result = self._fit(frame, cv_evaluation=True)
        np.testing.assert_allclose(
            result.fluctuations["mean"].thresholds, 1.0 / frame["cluster"].nunique(), rtol=1e-12
        )

    def test_the_shipped_rule_leaves_the_fold_record_unchanged(self, frame: Any) -> None:
        estimator = TMLE(
            **linear_in_sample(
                cross_fit=True,
                n_folds=4,
                learner_folds=4,
                q_bounds=(0.0, 1.0),
                targeting_scheme="fold",
            )
        )
        fluctuation = estimator.fit(frame, outcome="Y", treatment="A", id="cluster").single()
        pooled = fluctuation.fluctuations["mean"]
        assert pooled.stopping == "relative" and pooled.rule_ratio is None
        assert all(r.thresholds is None and r.rule_ratio is None for r in pooled.folds)


class TestCtmleCandidatesCountTheirTrainingUnits:
    def test_each_candidate_solve_counts_the_units_of_its_training_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C-TMLE refuses a clustered fit, so each row is a unit and n_u is the row count.
        log = Log(monkeypatch)
        log.wrap(ctmle_module, "solve_submodel", "solve")
        log.wrap(ctmle_module._Selector, "target", "target")
        rng = np.random.default_rng(8)
        n = 300
        w1 = rng.normal(size=n)
        w2 = rng.normal(size=n)
        a = rng.binomial(1, expit(0.6 * w1)).astype(float)
        y = 1.0 + a + w1 + 0.5 * w2 + rng.normal(size=n)
        frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})
        estimator = CTMLE(
            strategy="discrete",
            candidates=((), ("W1",), ("W1", "W2")),
            selection_folds=3,
            **linear_in_sample(stopping="sample_size", estimands=("ate",)),
        )
        estimator.fit(frame, outcome="Y", treatment="A")
        targets = log.of("target")
        solves = log.of("solve")
        assert len(targets) > 3
        # Every candidate target is one solve, in order; the retarget of the winner follows.
        assert len(solves) >= len(targets)
        sizes = set()
        for (_, t_args, _, _), (_, _, s_kwargs, fluctuation) in zip(targets, solves, strict=False):
            rows = np.asarray(t_args[3])
            sizes.add(rows.size)
            assert s_kwargs["stop"] is not None
            assert s_kwargs["stop"].units is None
            np.testing.assert_allclose(fluctuation.thresholds, 1.0 / rows.size, rtol=1e-12)
        # A training fold is not the sample: some candidate solve counted fewer than every row.
        assert min(sizes) < n


# ------------------------------------------------------- truncation curve and the bootstrap


class TestSensitivityAndBootstrapRunUnderADeclaredRule:
    @pytest.mark.parametrize("rule", RULES)
    def test_a_truncation_curve_retargets_under_the_rule(self, rule: str) -> None:
        result = _fit(rule, _clustered(), columns={"id": "cluster"})
        curve = result.diagnostics.truncation_curve([0.01, 0.05, 0.1])
        assert len(curve) >= 9
        assert np.all(np.isfinite(curve["psi"].to_numpy()))
        # At the fitted bounds the curve re-solves the fit's own targeting step, so it
        # reproduces the fit: a retarget that ran the rule differently would not.
        at_fit = curve["delta_from_fitted"].to_numpy()
        assert np.isclose(at_fit, 0.0, atol=1e-12).any()

    def test_a_retarget_of_a_fitted_result_reproduces_it(self) -> None:
        result = _fit("scaled_var", _clustered(), columns={"id": "cluster"})
        estimates, fluctuations = result.estimator.retarget(
            result.data, result.nuisance, estimands=("ate",)
        )
        assert fluctuations["mean"].stopping == "scaled_var"
        assert estimates["ate"].psi == result.estimates["ate"].psi

    @pytest.mark.parametrize("rule", RULES)
    def test_a_bootstrap_replicates_without_a_failed_draw(self, rule: str) -> None:
        estimator = TMLE(**linear_in_sample(stopping=rule, n_bootstrap=6))
        result = estimator.fit(_clustered(), outcome="Y", treatment="A", id="cluster").single()
        assert result.bootstrap is not None
        assert result.bootstrap.n_requested == 6
        # A replicate that raised is dropped and counted: a missing context would show here.
        assert result.bootstrap.n_failed == 0
        assert all(np.all(np.isfinite(draws)) for draws in result.bootstrap.draws.values())


# ------------------------------------------------------------- Fluctuation.rule_ratio itself


class TestTheSolversRecordTheirRatio:
    @staticmethod
    def _setting() -> dict[str, Any]:
        rng = np.random.default_rng(4)
        n = 240
        units = np.repeat(np.arange(60), 4)
        w = rng.normal(size=n)
        g1 = expit(0.5 * w)
        a = rng.binomial(1, g1).astype(float)
        y = np.clip(0.3 + 0.2 * a + 0.1 * w + 0.15 * rng.normal(size=n), 0.01, 0.99)
        return {
            "n": n,
            "units": units,
            "a": a,
            "g1": g1,
            "y": y,
            "ones": np.ones(n),
            "initial": InitialFit(np.full(n, 0.4), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.4)}),
        }

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("solver", ["iterative", "one_step"])
    def test_the_ratio_is_the_largest_score_over_its_threshold(
        self, rule: str, solver: str
    ) -> None:
        s = self._setting()
        submodel = mean_submodel(s["a"], s["g1"])
        stop = StopContext(s["units"], 2.0)
        kwargs: dict[str, Any] = {"stopping": rule, "stop": stop}
        if solver == "iterative":
            fit = solve_fluctuation(s["y"], s["initial"], submodel, s["ones"], None, **kwargs)
        else:
            fit = solve_one_step(
                s["y"], s["initial"], submodel, s["ones"], None, step_size=0.01, **kwargs
            )
        thresholds, ratio = hand_state(
            s["y"],
            fit.targeted.observed,
            submodel.observed,
            s["ones"],
            np.ones(s["n"], dtype=bool),
            units=s["units"],
            span=2.0,
            rule=rule,
            arm_curves={
                column: fit.targeted.arms[level] for level, column in submodel.arm_columns.items()
            },
        )
        np.testing.assert_allclose(fit.thresholds, thresholds, rtol=1e-12)
        assert fit.rule_ratio == pytest.approx(ratio, rel=1e-10, abs=1e-14)
        assert fit.converged == (fit.rule_ratio <= 1.0)

    def test_the_shipped_rule_records_no_ratio(self) -> None:
        s = self._setting()
        submodel = mean_submodel(s["a"], s["g1"])
        fit = solve_fluctuation(s["y"], s["initial"], submodel, s["ones"], None)
        assert fit.rule_ratio is None and fit.thresholds is None


# ------------------------------------------------------------------- the alternating exits

N_COMPLETE = 400
STEP = 0.01


def _drtmle(order: str, guard: tuple[str, ...], rule: str = "sample_size", **extra: Any) -> DRTMLE:
    return DRTMLE(
        **linear_drtmle(
            stopping=rule,
            cross_fit=False,
            update_order=order,
            guard=guard,
            estimands=("ate",),
            targeting="one_step",
            step_size=STEP,
            learner_folds=2,
            **extra,
        )
    )


@functools.cache
def _complete() -> Any:
    return make_linear_ate(n=N_COMPLETE, seed=5)[0]


def _trial(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 0.8 + 1.1 * a - 0.2 * w2 + rng.normal(scale=0.6, size=n)
    y[observed == 0.0] = np.nan
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": y})


def _missing_drtmle(rule: str, **extra: Any) -> DRTMLE:
    return DRTMLE(
        **linear_drtmle(
            stopping=rule,
            cross_fit=False,
            randomized=True,
            missingness_learner=lm.LogisticRegression(max_iter=1000),
            estimands=("ate",),
            targeting="one_step",
            step_size=STEP,
            max_outer=12,
            **extra,
        )
    )


def drtmle_problems(
    monkeypatch: pytest.MonkeyPatch,
    order: str,
    guard: tuple[str, ...],
    rule: str = "sample_size",
) -> list[str]:
    """Fit a DRTMLE under ``rule`` and audit every round of its outer loop by hand."""
    log = Log(monkeypatch)
    log.wrap(tmle_module, "solve_with_reduction", "reduction")
    log.wrap(targeting_module, "solve_submodel", "solve")
    log.wrap(targeting_module, "_retargeted_mechanism", "mech")
    log.wrap(targeting_module, "_restated_outcome_score", "restate")
    result = _drtmle(order, guard, rule).fit(_complete(), outcome="Y", treatment="A").single()
    fluctuation = result.fluctuations["mean"]
    record = fluctuation.reduction
    assert record is not None
    ((_, args, kwargs, _),) = log.of("reduction")
    data, nuisance, group, spec = args
    bounds, nuisance_bound = kwargs["bounds"], kwargs["nuisance_bound"]
    scaled, weights, observed = kwargs["scaled"], kwargs["weights"], kwargs["observed"]
    span = nuisance.scaler.range
    problems: list[str] = []

    q_now, current_now = nuisance.outcome, nuisance
    hand: list[float] = []
    for tag, call_args, _, out in log.events:
        if tag == "solve":
            q_now = out.targeted
        elif tag == "mech":
            current_now = out
        elif tag == "restate":
            h = build_submodel(
                data, current_now, group, bounds=bounds, nuisance_bound=nuisance_bound
            ).observed
            thresholds, ratio = hand_state(
                scaled,
                q_now.observed,
                h,
                weights,
                observed,
                units=data.cluster,
                span=span,
                rule=rule,
                arm_curves={j: q_now.arms[level] for j, level in enumerate(nuisance.arms)},
            )
            hand.append(ratio)
            k = len(hand) - 1
            read, solver = record.rule_trace[k]
            if not np.isclose(out.rule_ratio, ratio, rtol=1e-12):
                problems.append(
                    f"round {k + 1}: the restated record holds {out.rule_ratio}, hand {ratio}"
                )
            if not np.allclose(out.thresholds, thresholds, rtol=1e-12):
                problems.append(f"round {k + 1}: the restated thresholds are not the hand ones")
            if not np.isclose(read, ratio, rtol=1e-12):
                problems.append(f"round {k + 1}: the exit read {read}, hand {ratio}")
            if not np.isclose(solver, call_args[0].rule_ratio, rtol=1e-12):
                problems.append(
                    f"round {k + 1}: the trace holds a solver ratio the solver did not report"
                )
    rounds = len(hand)
    if rounds != record.rounds:
        problems.append(f"{rounds} restated rounds, record says {record.rounds}")

    # The exit claims.
    if record.exit_reason == "tolerance" and not hand[-1] <= 1.0:
        problems.append(f"a tolerance exit at hand ratio {hand[-1]}")
    if record.exit_reason == "stall":
        problems += _stall_problems(record, hand, spec.tol)
    if guard == ("Q",) and (record.exit_reason != "tolerance" or record.failure is not None):
        problems.append(
            f"the Q-guard fit exited at {record.exit_reason!r}, failure {record.failure}"
        )
    if record.exit_reason == "tolerance" and not record.converged:
        problems.append("a tolerance exit that reports converged=False")

    # The reported pair: every ingredient is on the result.
    h_final = hand_arm_h(result.data.treatment, fluctuation.mechanism.propensity, record.bounds)
    thresholds, ratio = hand_state(
        scaled,
        fluctuation.targeted.observed,
        h_final,
        weights,
        observed,
        units=data.cluster,
        span=span,
        rule=rule,
        arm_curves={j: fluctuation.targeted.arms[level] for j, level in enumerate(nuisance.arms)},
    )
    if not np.isclose(fluctuation.rule_ratio, ratio, rtol=1e-9):
        problems.append(
            f"the reported ratio {fluctuation.rule_ratio} is not the hand ratio {ratio}"
        )
    if not np.allclose(fluctuation.thresholds, thresholds, rtol=1e-12):
        problems.append("the reported thresholds are not the hand thresholds")
    if not np.isclose(record.rule_trace[-1][0], ratio, rtol=1e-9) or len(record.rule_trace) != len(
        record.trace
    ):
        problems.append("the closing row of the rule trace is not the reported ratio")
    if record.converged != (ratio <= 1.0 and record.failure is None) and guard == ("Q",):
        problems.append("converged disagrees with the hand ratio at the reported pair")
    return problems


def _stall_problems(record: Any, hand: list[float], tol: float) -> list[str]:
    """The stall rule replayed from the trace and the hand ratios (``worst`` and ``joint``)."""
    previous: float | None = None
    previous_joint: float | None = None
    stalled: list[bool] = []
    for row, ratio in zip(record.trace[: record.rounds], hand, strict=True):
        _, _, reduced, mechanism, joint = row
        worst = max(tol * ratio, reduced, mechanism)
        climbing = previous_joint is None or joint > previous_joint + tol * (1.0 + abs(joint))
        improving = previous is None or worst <= _STALL_FACTOR * previous
        stalled.append(not climbing and not improving)
        previous = worst if previous is None else min(previous, worst)
        previous_joint = joint
    problems = []
    if not stalled[-1]:
        problems.append("a stall exit where the replayed stall rule does not stall")
    if any(stalled[:-1]):
        problems.append("the replayed stall rule stalls before the round the loop stopped at")
    return problems


ORDERS_AND_GUARDS = [
    ("drtmle", ("Q",)),
    ("benkeser", ("Q",)),
    ("drtmle", ("Q", "g")),
    ("benkeser", ("Q", "g")),
]


class TestDrtmleExitsReadTheRatioAtThePairTheyJudge:
    @pytest.mark.parametrize(("order", "guard"), ORDERS_AND_GUARDS)
    def test_every_round_reads_the_hand_ratio(
        self, monkeypatch: pytest.MonkeyPatch, order: str, guard: tuple[str, ...]
    ) -> None:
        assert drtmle_problems(monkeypatch, order, guard) == []

    def test_the_stale_ratio_witness_and_the_shipped_exit_witness(self) -> None:
        for order, expect_stale in (("benkeser", True), ("drtmle", False)):
            result = _drtmle(order, ("Q",)).fit(_complete(), outcome="Y", treatment="A").single()
            fluctuation = result.fluctuations["mean"]
            record = fluctuation.reduction
            assert record is not None
            gaps = [abs(read - solver) for read, solver in record.rule_trace]
            # (1) The shipped exit could not fire: the outcome score sits far above tol.
            assert fluctuation.relative_score_norm > 1e3 * 1e-10
            if expect_stale:
                # (2) The solver's ratio and the restated ratio differ by more than 1e-2.
                assert max(gaps) > 1e-2
                assert (
                    max(read for read, _ in record.rule_trace)
                    > 1.0
                    >= max(solver for _, solver in record.rule_trace)
                )
            else:
                # Under the shipped order nothing moves after the solve: the restatement
                # is a no-op and the fresh ratio is the solver's.
                assert max(gaps[:-1]) < 1e-12

    def test_the_shipped_rule_records_no_rule_trace(self) -> None:
        result = (
            _drtmle("benkeser", ("Q",), rule="relative")
            .fit(_complete(), outcome="Y", treatment="A")
            .single()
        )
        record = result.fluctuations["mean"].reduction
        assert record is not None and record.rule_trace == ()
        assert result.fluctuations["mean"].rule_ratio is None


def missing_problems(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fit the missing-outcome reduction and rebuild each round's restated pair independently."""
    log = Log(monkeypatch)
    log.wrap(targeting_module, "_solve_missing_outcome_reduction", "missing")
    log.wrap(targeting_module, "solve_submodel", "solve")
    log.wrap(targeting_module, "_solve_missing_observation_mechanism", "observation")
    log.wrap(targeting_module, "_solve_missing_treatment_mechanism", "treatment")
    log.wrap(DRTMLEClass, "_fit_reduced", "refit")
    result = (
        _missing_drtmle("sample_size")
        .fit(_trial(), outcome="Y", treatment="A", delta="Delta", covariates=["W1", "W2"])
        .single()
    )
    fluctuation = result.fluctuations["mean"]
    record = fluctuation.reduction
    assert record is not None
    ((_, args, kwargs, _),) = log.of("missing")
    data, nuisance, group, _spec = args
    bounds, nuisance_bound = kwargs["bounds"], kwargs["nuisance_bound"]
    scaled, weights, observed = kwargs["scaled"], kwargs["weights"], kwargs["observed"]
    span = nuisance.scaler.range
    problems: list[str] = []

    first = next(i for i, e in enumerate(log.events) if e[0] == "solve")
    rounds: list[dict[str, Any]] = []
    for tag, _, _, out in log.events[first:]:
        if tag == "solve":
            rounds.append({"q": out.targeted, "solver": out.rule_ratio})
        elif tag == "observation":
            rounds[-1]["m"] = np.asarray(out.propensity, dtype=float)
        elif tag == "treatment":
            rounds[-1]["g"] = np.asarray(out.propensity, dtype=float)
        elif tag == "refit":
            rounds[-1]["reduced"] = out[0]
    if len(rounds) != record.rounds or len(record.rule_trace) != record.rounds:
        problems.append(f"{len(rounds)} rounds audited, record says {record.rounds}")
    mask = np.asarray(observed, dtype=bool)
    ratios = []
    for k, state in enumerate(rounds):
        current = _missing_outcome_state(nuisance, state["q"], state["g"], state["m"])
        standard = build_submodel(
            data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
        )
        extra = missing_outcome_outcome_submodel(
            data.treatment, state["reduced"], g_bounds=bounds, missingness_bound=nuisance_bound
        )
        _, ratio = hand_state(
            scaled,
            state["q"].observed,
            np.hstack([standard.observed, extra.observed]),
            weights,
            mask,
            units=data.cluster,
            span=span,
            rule="sample_size",
            arm_curves={j: state["q"].arms[level] for j, level in enumerate(nuisance.arms)},
        )
        ratios.append(ratio)
        read, solver = record.rule_trace[k]
        if not np.isclose(read, ratio, rtol=1e-12):
            problems.append(f"round {k + 1}: the exit read {read}, hand {ratio}")
        if not np.isclose(solver, state["solver"], rtol=1e-12):
            problems.append(
                f"round {k + 1}: the trace holds a solver ratio the solver did not report"
            )
    # The reported outcome record holds the stacked ratio at the final pair, the ratio its
    # verdict reads; the reduction record holds the drift block's part of it, at the stacked
    # thresholds.  So the outcome row and the record agree (review finding F3).
    last = rounds[-1]
    current = _missing_outcome_state(nuisance, last["q"], last["g"], last["m"])
    standard = build_submodel(data, current, group, bounds=bounds, nuisance_bound=nuisance_bound)
    drift = missing_outcome_outcome_submodel(
        data.treatment, last["reduced"], g_bounds=bounds, missingness_bound=nuisance_bound
    )
    stacked_thresholds, stacked_ratio = hand_state(
        scaled,
        last["q"].observed,
        np.hstack([standard.observed, drift.observed]),
        weights,
        mask,
        units=data.cluster,
        span=span,
        rule="sample_size",
        arm_curves={j: last["q"].arms[level] for j, level in enumerate(nuisance.arms)},
    )
    width = standard.observed.shape[1]
    drift_score = np.abs(np.asarray(record.score)) * span
    drift_ratio = float(np.max(drift_score / stacked_thresholds[width:]))
    if not np.isclose(fluctuation.rule_ratio, stacked_ratio, rtol=1e-9):
        problems.append("the reported outcome ratio is not the hand stacked ratio")
    if not np.allclose(fluctuation.thresholds, stacked_thresholds[:width], rtol=1e-12):
        problems.append("the reported outcome thresholds are not the stacked ones")
    if record.rule_ratio is None or not np.isclose(record.rule_ratio, drift_ratio, rtol=1e-9):
        problems.append("the reduction record does not hold the drift block's stacked ratio")
    if (record.failure is None) and not fluctuation.rule_ratio <= 1.0:
        problems.append("the record reports no failure where the stacked ratio exceeds one")
    if record.converged and not ratios[-1] <= 1.0:
        problems.append("converged=True where the hand ratio of the last round exceeds one")
    if np.isclose(max(abs(r - s) for r, s in record.rule_trace), 0.0, atol=1e-2):
        problems.append("no round shows a restated ratio that differs from the solver's")
    return problems


class TestMissingOutcomeLoopComputesItsRatioFromTheRestatedScores:
    def test_every_round_reads_the_hand_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert missing_problems(monkeypatch) == []

    def test_the_shipped_exit_could_not_fire(self) -> None:
        result = (
            _missing_drtmle("sample_size")
            .fit(_trial(), outcome="Y", treatment="A", delta="Delta", covariates=["W1", "W2"])
            .single()
        )
        fluctuation = result.fluctuations["mean"]
        assert fluctuation.relative_score_norm > 1e3 * 1e-10
        assert fluctuation.reduction is not None and fluctuation.reduction.rule_trace


@functools.cache
def _ipsi_frame() -> Any:
    return make_linear_ate(n=200, seed=5)[0]


def ipsi_problems(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fit a point IPSI under ``sample_size`` and audit each round's ratio and the exit round."""
    log = Log(monkeypatch)
    log.wrap(tmle_module, "solve_with_mechanism", "mechanism")
    log.wrap(targeting_module, "solve_submodel", "solve")
    estimator = TMLE(
        **linear_in_sample(
            stopping="sample_size",
            incremental=[Incremental(2.0)],
            targeting="one_step",
            step_size=STEP,
        )
    )
    result = estimator.fit(_ipsi_frame(), outcome="Y", treatment="A").single()
    fluctuation = result.fluctuations["ipsi"]
    record = fluctuation.mechanism
    assert record is not None
    ((_, args, _, _),) = log.of("mechanism")
    data, nuisance, _, spec = args
    span = nuisance.scaler.range
    solves = log.of("solve")
    problems: list[str] = []
    # The first solve precedes the loop; each later solve is the last step of a round.
    ratios = []
    for _, s_args, _, out in solves[1:]:
        scaled, _, submodel, weights, observed = s_args[:5]
        _, ratio = hand_state(
            scaled,
            out.targeted.observed,
            submodel.observed,
            weights,
            observed,
            units=data.cluster,
            span=span,
            rule="sample_size",
        )
        ratios.append(ratio)
    if len(ratios) != len(record.trace) or len(record.rule_trace) != len(record.trace):
        problems.append("the rule trace is not one entry per round")
    for k, (read, ratio) in enumerate(zip(record.rule_trace, ratios, strict=False)):
        if not np.isclose(read, ratio, rtol=1e-12):
            problems.append(f"round {k + 1}: the exit read {read}, hand {ratio}")
        if not np.isclose(read, solves[k + 1][3].rule_ratio, rtol=1e-12):
            problems.append(f"round {k + 1}: the exit did not read the solver's ratio")
    # The exit round: the first round whose mechanism score meets tol and whose hand ratio is
    # at most one.  (No stall can precede it: it comes at the second round.)
    settled = [row[2] for row in record.trace]
    expected = next(
        (
            k + 1
            for k, (s, r) in enumerate(zip(settled, ratios, strict=False))
            if s <= spec.tol and r <= 1.0
        ),
        None,
    )
    if expected is None or len(record.trace) != expected:
        problems.append(f"the loop ran {len(record.trace)} rounds, hand exit round {expected}")
    if record.failure is not None:
        problems.append(f"failure {record.failure}")
    if not fluctuation.relative_score_norm > 1e3 * spec.tol:
        problems.append("the shipped exit could have fired here")
    return problems


class TestPointIpsiExitReadsTheSolversRatio:
    def test_every_round_and_the_exit_round_match_the_hand_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert ipsi_problems(monkeypatch) == []

    def test_the_shipped_rule_records_no_rule_trace(self) -> None:
        estimator = TMLE(**linear_in_sample(incremental=[Incremental(2.0)]))
        result = estimator.fit(_ipsi_frame(), outcome="Y", treatment="A").single()
        mechanism = result.fluctuations["ipsi"].mechanism
        assert mechanism is not None and mechanism.rule_trace == ()


# --------------------------------------------------------------------- mutation controls


def _patch_reduction(
    monkeypatch: pytest.MonkeyPatch, name: str, edits: list[tuple[str, str]]
) -> None:
    mutate(monkeypatch, targeting_module, targeting_module, name, edits)
    if name == "solve_with_reduction":
        monkeypatch.setattr(tmle_module, name, getattr(targeting_module, name))
    if name == "solve_with_mechanism":
        monkeypatch.setattr(tmle_module, name, getattr(targeting_module, name))


class TestMutationControls:
    """One broken line each.  The hand audit of the unmutated code is empty; with the line broken
    it reports a problem."""

    def test_the_outcome_ruler_is_the_rule_ratio_on_the_tol_scale(self) -> None:
        shipped = TargetingSpec()
        ruled = TargetingSpec(stopping="sample_size", tol=1e-6)
        assert _outcome_ruler(shipped, 3.0e-4, None) == 3.0e-4
        assert _outcome_ruler(ruled, 3.0e-4, 0.5) == pytest.approx(5.0e-7, rel=1e-14)
        assert _outcome_ruler(ruled, 3.0e-4, 2.0) == pytest.approx(2.0e-6, rel=1e-14)

    def test_the_outer_test_reading_the_relative_score_again_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert drtmle_problems(monkeypatch, "drtmle", ("Q",)) == []
        _patch_reduction(
            monkeypatch,
            "solve_with_reduction",
            [
                (
                    "outcome_solved = fluctuation.rule_ratio <= 1.0",
                    "outcome_solved = _solved(fluctuation.relative_score_norm, "
                    "fluctuation.score_norm, spec.tol, negligible)",
                )
            ],
        )
        assert drtmle_problems(monkeypatch, "drtmle", ("Q",)) != []

    def test_the_outcome_entry_reading_the_relative_score_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert ipsi_problems(monkeypatch) == []
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_outcome_ruler",
            [("return spec.tol * rule_ratio", "return relative")],
        )
        assert ipsi_problems(monkeypatch) != []
        mutated = targeting_module._outcome_ruler
        ruled = TargetingSpec(stopping="sample_size", tol=1e-6)
        assert mutated(ruled, 3.0e-4, 0.5) != pytest.approx(5.0e-7, rel=1e-14)

    def test_the_outcome_entry_dropping_the_ratio_fails_the_ruler_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ruled = TargetingSpec(stopping="sample_size", tol=1e-6)
        assert _outcome_ruler(ruled, 3.0e-4, 2.0) == pytest.approx(2.0e-6, rel=1e-14)
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_outcome_ruler",
            [("return spec.tol * rule_ratio", "return spec.tol")],
        )
        mutated = targeting_module._outcome_ruler
        assert mutated(ruled, 3.0e-4, 2.0) != pytest.approx(2.0e-6, rel=1e-14)

    def test_the_exits_reading_the_solvers_ratio_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert drtmle_problems(monkeypatch, "benkeser", ("Q",)) == []
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_restated_outcome_score_ruled",
            [("rule_ratio=ratio,", "rule_ratio=fluctuation.rule_ratio,")],
        )
        problems = drtmle_problems(monkeypatch, "benkeser", ("Q",))
        assert any("the exit read" in problem for problem in problems)

    def test_the_missing_outcome_loop_reading_the_solvers_ratio_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.MonkeyPatch.context() as audit:
            assert missing_problems(audit) == []
        _patch_reduction(
            monkeypatch,
            "_solve_missing_outcome_reduction",
            [
                (
                    "rule_rows.append((restated_ratio, outcome_fit.rule_ratio))",
                    "rule_rows.append((outcome_fit.rule_ratio, outcome_fit.rule_ratio))",
                )
            ],
        )
        assert any("the exit read" in p for p in missing_problems(monkeypatch))

    def test_restating_with_the_solvers_thresholds_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under sample_size the thresholds are 1 / n_u at every pair, so only scaled_var can
        # tell a refreshed threshold vector from the solver's.
        assert drtmle_problems(monkeypatch, "benkeser", ("Q",), "scaled_var") == []
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_restated_outcome_score_ruled",
            [("thresholds=thresholds,", "thresholds=fluctuation.thresholds,")],
        )
        problems = drtmle_problems(monkeypatch, "benkeser", ("Q",), "scaled_var")
        assert any("thresholds" in problem for problem in problems)


def test_a_declared_rule_without_the_helper_context_still_fails_loudly() -> None:
    """The loud failure of G3a stays for a direct call that passes no context."""
    from cleverly.estimators.targeting import solve_submodel
    from cleverly.fluctuation._score import STOP_CONTEXT_MISSING

    s = TestTheSolversRecordTheirRatio._setting()
    submodel = mean_submodel(s["a"], s["g1"])
    spec = TargetingSpec(stopping="scaled_var")
    with pytest.raises(ValueError, match=STOP_CONTEXT_MISSING[:40]):
        solve_submodel(s["y"], s["initial"], submodel, s["ones"], np.ones(s["n"], bool), spec)
    assert replace(spec, stopping="relative").stopping == "relative"
