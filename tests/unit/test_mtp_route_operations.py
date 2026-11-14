"""Operations on a modified-treatment-policy fit under each density-ratio route (W15, W16).

The capability sweep (:mod:`tests.unit.test_capability_row_sweep`) asks every row about every
kind of :mod:`tests.unit._mtp_route_kinds`, and the plot matrix
(:mod:`tests.unit.test_plotting_result_matrix`) draws every plot on each, so those gates check
that each operation runs or refuses with a stated reason.  This module keeps the checks they do
not make: a refit reads the route recorded on the result, a truncation curve that runs moves the
estimate when its bound binds, the point ``g(W)`` axis of a policy fit on a dose is refused, the
omitted-variable representer is the policy's clever covariate, and ``ratio_trim=`` reaches a
continuous node.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.exceptions import CapabilityError
from cleverly.interventions import PolicySet
from cleverly.longitudinal import LTMLE, SDR
from cleverly.longitudinal.estimator import longitudinal_truncation_curve
from cleverly.longitudinal.sequential import Mechanism
from cleverly.sensitivity import omitted_variable, positivity
from cleverly.sensitivity.positivity import truncation_curve
from cleverly.utils.bounds import bound
from tests.studies import canonical_longitudinal_mtp as panel_study
from tests.studies import longitudinal_mtp_common as panel_law
from tests.unit import _capability_sweep_support as sweep
from tests.unit import _mtp_route_kinds as kinds

sequential = importlib.import_module("cleverly.longitudinal.sequential")
sdr_module = importlib.import_module("cleverly.longitudinal.sdr")

POINT = [name for name in kinds.FITS if not name.startswith("mtp/panel/")]
PANEL = [name for name in kinds.FITS if name.startswith("mtp/panel/")]

_BUILT: dict[str, Any] = {}


def build(name: str) -> Any:
    """Each kind's fit once per process; a test reads it and writes nothing to it."""
    if name not in _BUILT:
        _BUILT[name] = kinds.FITS[name]()
    return _BUILT[name]


# ------------------------------------------------------------------ the refit route


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the route of every policy ratio a point or longitudinal fit estimates."""
    calls: list[str] = []
    for name, route in (("evaluate_by_classifier", "classifier"), ("evaluate", "density")):
        original = vars(PolicySet)[name].__func__

        def point(cls: Any, *args: Any, _f: Any = original, _r: str = route, **kw: Any) -> Any:
            calls.append(_r)
            return _f(cls, *args, **kw)

        monkeypatch.setattr(PolicySet, name, classmethod(point))
    for name, route in (("classifier_ratio", "classifier"), ("fit_conditional_density", "density")):
        original = getattr(sequential, name)

        def panel(*args: Any, _f: Any = original, _r: str = route, **kw: Any) -> Any:
            calls.append(_r)
            return _f(*args, **kw)

        monkeypatch.setattr(sequential, name, panel)
    return calls


def _refute(result: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arguments = sweep.sweep_arguments(dataclasses.replace(result))["refute"]
        dataclasses.replace(result).diagnostics.refute(**arguments, random_state=0)


@pytest.mark.parametrize("name", list(kinds.FITS))
def test_w15_every_refit_reads_the_route_recorded_on_the_result(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = build(name)
    recorded = result.config.ratio_route.route
    assert result.estimator.ratio == recorded
    calls = _spy(monkeypatch)
    _refute(result)
    assert calls and set(calls) == {recorded}, calls


@pytest.mark.parametrize("name", ["mtp/shift/default", "mtp/panel/default"])
def test_w15_a_refit_forced_to_the_other_route_fails_the_spy(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation (i): the refit reads the density route while the fit read the classifier."""
    result = build(name)
    estimator = type(result.estimator)
    original = estimator.refit

    def other(self: Any, *args: Any, **kw: Any) -> Any:
        moved = copy.copy(self)
        moved.ratio = "density"
        return original(moved, *args, **kw)

    monkeypatch.setattr(estimator, "refit", other)
    calls = _spy(monkeypatch)
    _refute(result)
    assert "density" in calls and set(calls) != {result.config.ratio_route.route}


# ------------------------------------------------------------------ the nuisance report


@pytest.mark.parametrize("name", list(kinds.FITS))
def test_w15_the_nuisance_report_states_the_classifier_route(name: str) -> None:
    """A classifier-route fit says that it fitted no treatment model; a density fit does not."""
    from cleverly.validation.longitudinal import LONGITUDINAL_CLASSIFIER_RATIO
    from cleverly.validation.nuisance import NUISANCE_CLASSIFIER_RATIO

    result = build(name)
    classifier = result.config.ratio_route.route == "classifier"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = dataclasses.replace(result).diagnostics.nuisance_models()
    if name in PANEL:
        stated = {o.time for o in report.omissions if o.reason == LONGITUDINAL_CLASSIFIER_RATIO}
        assert stated == ({1, 2} if classifier else set())
        return
    assert report.treatment_omission == (NUISANCE_CLASSIFIER_RATIO if classifier else None)
    codes = [item.code for item in report.summary().notes]
    assert ("classifier_ratio" in codes) is classifier
    assert ("No treatment model was fitted" in str(report.summary("full"))) is classifier


# ------------------------------------------------------------------ truncation


@pytest.mark.parametrize("name", POINT)
def test_w15_the_point_propensity_axis_of_a_policy_fit_is_refused(name: str) -> None:
    with pytest.raises(CapabilityError) as caught:
        truncation_curve(build(name), [0.05, 0.1])
    assert "a point fit fits no propensity g(W)" in str(caught.value)
    assert caught.value.refusal.alternative


def test_w15_without_the_rule_the_point_curve_runs_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation (iv): the policy_ratio rule removed, the curve caps nothing at any bound,
    so it fails the positive-share check a working curve meets."""
    rules = tuple(rule for rule in positivity._TRUNCATION_RULES if rule[0] != "policy_ratio")
    monkeypatch.setattr(positivity, "_TRUNCATION_RULES", rules)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = truncation_curve(build("mtp/shift/default"), [0.05, 0.2, 0.4])
    assert not np.any(np.asarray(frame["truncated_fraction"], dtype=float) > 0.0)


def _sweep(result: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return longitudinal_truncation_curve(result, [None, 0.95, 0.9, 0.8])


@pytest.mark.parametrize("name", PANEL)
def test_w15_the_longitudinal_curve_of_a_policy_sweeps_ratio_trim(name: str) -> None:
    result = build(name)
    frame = _sweep(result)
    for estimand, rows in frame.groupby("estimand"):
        fitted = rows[rows["is_fitted_bound"]]
        assert len(fitted) == 1
        assert abs(float(fitted["psi"].iloc[0]) - result.estimates[estimand].psi) <= 1e-10
        assert fitted["ratio_trim"].isna().all()
    trimmed = frame[frame["trimmed_share"] > 0.0]
    assert len(trimmed)
    for _, row in trimmed.iterrows():
        assert row["inference_status"] == "trimmed_ratio_plugin"
    up = frame[frame["estimand"] == "ey_regimen[up]"]
    moved = up[up["trimmed_share"] > 0.0]
    assert len(moved) and np.all(np.abs(moved["delta_from_fitted"]) > 1e-8)


def _shipped_cumulative(
    self: Mechanism, data: Any, plan: Any, bounds: tuple[float, float]
) -> tuple[Any, Any]:
    """The composition before the fix: ``g_bounds``' upper bound on the implied mechanism."""
    raw, bounded = self._cumulative_with_unbounded(data, plan, bounds)
    trim = self.ratio_trims.get(plan.label)
    if trim is None:
        return raw, bounded
    density = plan.intervention_density(data)
    implied = trim.implied_cumulative(np.cumprod(density, axis=1, dtype=np.float64), raw)
    return implied, bound(implied, *bounds)


def _shipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Mechanism, "cumulative_with_unbounded", _shipped_cumulative)
    original = sdr_module.partial_weights

    def untrimmed(*args: Any, **kw: Any) -> Any:
        return original(*args[:6])

    monkeypatch.setattr(sdr_module, "partial_weights", untrimmed)


def test_w15_the_shipped_trim_composition_runs_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation (v): the trim moves a tenth of the entries and no estimate."""
    _shipped(monkeypatch)
    frame = _sweep(kinds.fit_panel("classifier"))
    up = frame[frame["estimand"] == "ey_regimen[up]"]
    assert (up["trimmed_share"] > 0.0).any()
    assert np.all(np.abs(up["delta_from_fitted"]) <= 1e-12)


def test_w15_an_engine_that_refuses_ratio_trim_refuses_the_curve() -> None:
    from cleverly.longitudinal import SequentialGComputation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SequentialGComputation(
            {"natural": panel_law.NATURAL, "up": panel_law.UP},
            reference="natural",
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=1,
        ).fit(panel_law.sample_continuous(300, 0), **_PANEL_COLUMNS)
    with pytest.raises(CapabilityError, match="ratio_trim"):
        longitudinal_truncation_curve(result, [0.9])


# ------------------------------------------------------------------ the representer


def _representer_gap(name: str) -> float:
    result = build(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elements = dataclasses.replace(result).sensitivity.elements()
    clever = result.nuisance.policies.policy_ratio[:, 0]
    observed = np.asarray(result.data.observed, dtype=float)
    return float(np.max(np.abs(elements.riesz_representer - clever * observed)))


@pytest.mark.parametrize("name", [n for n in kinds.FITS if n.startswith("mtp/atoms/")])
def test_w15_the_representer_is_the_policy_clever_covariate(name: str) -> None:
    assert _representer_gap(name) <= 1e-12


def test_w15_a_constant_representer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation (vi)."""
    monkeypatch.setattr(
        omitted_variable, "_riesz_representer", lambda parameter, submodel, axis: 1.0
    )
    assert _representer_gap("mtp/atoms/default") > 1e-3


# ------------------------------------------------------------------ W16: ratio_trim

_PANEL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}


def _trim_fit(estimator: type, ratio: str, trim: float | None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(
            panel_study.regimens(panel_study.CONTINUOUS),
            reference=panel_study.REFERENCE,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=2000),
            n_folds=1 if estimator is LTMLE else 2,
            ratio=ratio,
            simultaneous=False,
            random_state=0,
            ratio_trim=trim,
        ).fit(panel_law.sample_continuous(600, 3), **_PANEL_COLUMNS)


def _trim_reaches_the_estimate(estimator: type, ratio: str) -> list[str]:
    found = []
    untrimmed = _trim_fit(estimator, ratio, None)
    trimmed = _trim_fit(estimator, ratio, 0.9)
    trim = trimmed.mechanism.ratio_trims["up"]
    if trim.trimmed_share < 0.05:
        found.append(f"share {trim.trimmed_share}")
    gap = abs(trimmed.estimates["ey_regimen[up]"].psi - untrimmed.estimates["ey_regimen[up]"].psi)
    if not gap > 1e-4:
        found.append(f"ey_regimen[up] moved {gap}")
    if estimator is LTMLE:
        # The clever covariate is the cumulative numerator over the bounded mechanism: on a
        # row whose lower bound does not bind it is the running product of min(r, Q_q).
        plan = next(p for p in trimmed.replay_recipe.plans if p.label == "up")
        implied, bounded = trimmed.mechanism.cumulative_with_unbounded(
            trimmed.data, plan, trimmed.config.g_bounds
        )
        numerator = np.cumprod(plan.intervention_density(trimmed.data), axis=1)
        rows = (trim.cumulative > 0.0) & (implied > trimmed.config.g_bounds[0])
        safe = np.where(rows, bounded, 1.0)
        gap = float(np.max(np.abs(np.where(rows, numerator / safe - trim.cumulative, 0.0))))
        if gap > 1e-12:
            found.append(f"clever covariate off the trimmed product by {gap}")
    return found


@pytest.mark.parametrize("ratio", ["density", "classifier"])
@pytest.mark.parametrize("estimator", [LTMLE, SDR], ids=["LTMLE", "SDR"])
def test_w16_ratio_trim_reaches_a_continuous_node(estimator: type, ratio: str) -> None:
    assert _trim_reaches_the_estimate(estimator, ratio) == []


@pytest.mark.parametrize("estimator", [LTMLE, SDR], ids=["LTMLE", "SDR"])
def test_w16_the_shipped_composition_leaves_the_estimate(
    estimator: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: the upper bound of ``g_bounds`` applied to the implied mechanism."""
    _shipped(monkeypatch)
    found = " ".join(_trim_reaches_the_estimate(estimator, "classifier"))
    assert "ey_regimen[up] moved" in found


def test_w16_the_hazard_model_refuses_ratio_trim() -> None:
    with pytest.raises(CapabilityError, match="ratio_trim"):
        LTMLE({"always": 1}, survival_model="hazard", ratio_trim=0.9)


def test_w16_a_static_plan_with_a_binding_trim_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=400, seed=12)

    def fit() -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                {"always": 1, "never": 0},
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
                n_folds=1,
                ratio_trim=0.9,
                simultaneous=False,
            ).fit(
                frame,
                outcome="Y",
                treatment=["A1", "A2"],
                baseline=["W1", "W2"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )
        assert result.mechanism.ratio_trims["always"].trimmed_share > 0.0
        return {name: e.psi for name, e in result.estimates.items()}

    fixed = fit()
    _shipped(monkeypatch)
    assert fit() == fixed
