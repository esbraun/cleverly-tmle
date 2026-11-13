"""The small-sample scale of a bounded mean and the two withheld statuses (Fix-3 PR 2).

An unclustered mean of a yes/no outcome takes the Wald interval on the logit scale with the
normal reference, and every other estimate keeps its interval (T2b, T2c, T2e, T8, T9, T24). A
treatment arm with one observed outcome value in its stratum, or an estimate on its bound,
withholds the interval, the p-value and the band; a contrast inherits the status (T6, T7, T7b,
T7c, T10, T18a, T18b, T21, T23). The summary names the scale (T22), and every builder of an
estimate records its outcome family or is named as leaving it out (T25). The status registry
and the refusal codes are pinned in ``tests/unit/test_inference_status_registry.py`` (T15).
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._inference_status import (
    CONSTANT_ARM_CELL_STATUS,
    NO_CONSTANT_ARM_BANDS,
    OUT_OF_RANGE_STATUS,
    status_record,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference import Transform, make_estimate, simultaneous_bands, wald_ci
from cleverly.inference import influence as influence_module
from cleverly.inference.influence import (
    ClusterReference,
    bounded_status,
    restamp_parameter_statuses,
    with_precedent_status,
)
from cleverly.inference.multiplier import out_of_range
from cleverly.inference.results import ratio_contrast, smooth_contrast
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import stratified_law as law
from tests.studies.evidence.properties import replicate_row

ROOT = Path(__file__).resolve().parents[2]
Z = float(stats.norm.ppf(0.975))
UNIT = (0.0, 1.0)

#: A draw of the stratified boundary law at n = 500 whose treated rows of stratum V = 2 all
#: hold the event, found by scanning seeds 0 to 199 with ``constant_outcome_cells`` alone.
CONSTANT_CELL_SEED = 3


def _curve(se: float, n: int = 400, seed: int = 0) -> np.ndarray:
    """A centred curve whose influence-curve variance is ``se ** 2`` exactly."""
    draw = np.random.default_rng(seed).normal(size=n)
    draw = (draw - draw.mean()) / draw.std(ddof=1)
    return draw * se * math.sqrt(n)


def _mean(psi: float, se: float = 0.02, **kwargs: Any) -> Any:
    return make_estimate("ey1", psi, _curve(se), n=400, scale="level", bounds=UNIT, **kwargs)


def _logit_interval(psi: float, se: float, quantile: float = Z) -> tuple[float, float]:
    half = quantile * se / (psi * (1.0 - psi))
    return float(expit(logit(psi) - half)), float(expit(logit(psi) + half))


def _heavy(nu_target: float, n: int = 400) -> np.ndarray:
    """A centred curve whose moment df is near ``nu_target``: a few large terms on a light base."""
    rng = np.random.default_rng(7)
    curve = 0.02 * rng.normal(size=n)
    k = max(2, int(nu_target / 2))
    curve[:k] = np.tile([1.0, -1.0], k)[:k]
    return curve - curve.mean()


# ----------------------------------------------------------------------------------- T2b, T2c


def test_a_bounded_mean_takes_the_logit_interval_written_out() -> None:
    estimate = _mean(0.93)
    assert estimate.interval_scale == "range_logit"
    low, high = estimate.ci
    assert (low, high) == pytest.approx(_logit_interval(0.93, 0.02), rel=1e-12)
    # Asymmetric about the estimate: the side toward the bound is shorter.
    assert high - 0.93 < 0.93 - low
    assert 0.0 < low < high < 1.0


def test_the_identity_scale_fails_the_bounded_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(influence_module.ParameterEstimate, "_range_scale", lambda self: None)
    estimate = _mean(0.93)
    assert estimate.ci == pytest.approx(wald_ci(0.93, 0.02), rel=1e-12)
    assert estimate.ci != pytest.approx(_logit_interval(0.93, 0.02), rel=1e-6)


def test_a_heavy_curve_on_a_bounded_mean_keeps_the_normal_reference() -> None:
    curve = _heavy(8.0)
    estimate = make_estimate(
        "ey1", 0.93, curve, n=400, scale="level", bounds=UNIT, outcome_family="binomial"
    )
    assert 3.0 < estimate.small_sample_df < 30.0
    se = estimate.std_error
    low, high = estimate.ci
    # Half-width on the logit scale over se f'(psi): z, not t on the curve's df.
    quantile = (logit(high) - logit(low)) / 2.0 / (se / (0.93 * 0.07))
    assert quantile == pytest.approx(Z, rel=1e-9)
    t_quantile = stats.t.ppf(0.975, estimate.small_sample_df)
    assert abs(quantile - t_quantile) > 0.05


# ----------------------------------------------------------------------------------- T2e


def test_no_t_ships_and_a_difference_keeps_its_identity_interval() -> None:
    curve = _heavy(8.0)
    gaussian = make_estimate("ey1", 2.5, curve, n=400, scale="level", outcome_family="gaussian")
    assert gaussian.effective_df is not None and gaussian.effective_df < 30.0
    assert gaussian.interval_scale == "identity"
    assert gaussian.ci == wald_ci(2.5, gaussian.std_error)
    difference = make_estimate(
        "ate", 0.3, _curve(0.04), n=400, scale="difference", bounds=(-1.0, 1.0)
    )
    assert difference.interval_scale == "identity"
    assert difference.ci == wald_ci(0.3, difference.std_error)


def test_reading_the_effective_df_fails_the_normal_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    curve = _heavy(8.0)
    gaussian = make_estimate("ey1", 2.5, curve, n=400, scale="level", outcome_family="gaussian")
    shipped = gaussian.ci

    def with_t(self: Any) -> tuple[float, float]:
        return wald_ci(self.psi, self._plugin_std_error(), self.alpha, df=self.effective_df)

    monkeypatch.setattr(influence_module.ParameterEstimate, "_wald_interval", with_t)
    assert gaussian.ci != shipped


# ----------------------------------------------------------------------------------- T6


@pytest.mark.parametrize("status", [CONSTANT_ARM_CELL_STATUS, OUT_OF_RANGE_STATUS])
def test_a_withheld_estimate_reports_no_interval_test_or_error(status: str) -> None:
    estimate = replace(_mean(0.9995, se=5e-5), inference=status)
    reason = status_record(status).reason
    for read in (
        lambda: estimate.ci,
        lambda: estimate.std_error,
        lambda: estimate.pvalue,
        lambda: estimate.wald_test(),
    ):
        with pytest.raises(CapabilityError) as raised:
            read()
        assert reason in str(raised.value)
    assert estimate.plugin_std_error == pytest.approx(5e-5, rel=1e-9)
    assert all(math.isnan(limit) for limit in estimate.plugin_interval)
    assert math.isnan(estimate._plugin_pvalue())
    row = replicate_row(
        property_name="p",
        cell="c",
        role="positive",
        replicate=0,
        n=400,
        requested=1,
        truth=0.94,
        estimate=estimate,
        alpha=0.05,
    )
    assert (row["covered"], row["rejected"]) == (0, 0)


def test_removing_the_status_shows_an_interval_that_misses_and_rejects() -> None:
    shown = _mean(0.9995, se=5e-5)
    low, high = shown.ci
    assert not low <= 0.94 <= high
    assert shown.wald_test(0.94).pvalue < 0.05


def test_a_constant_cell_member_refuses_the_band() -> None:
    estimates = {
        "ey1": replace(_mean(0.9995, se=5e-5), inference=CONSTANT_ARM_CELL_STATUS),
        "ey0": make_estimate("ey0", 0.6, _curve(0.03, seed=1), n=400, scale="level", bounds=UNIT),
    }
    with pytest.raises(CapabilityError) as raised:
        simultaneous_bands(estimates, n_replicates=500, random_state=0)
    assert NO_CONSTANT_ARM_BANDS.format(names="ey1") in str(raised.value)


# ----------------------------------------------------------------------------------- T7, T7b, T7c


def test_the_df_never_sets_a_status_on_a_bounded_mean() -> None:
    estimate = make_estimate("ey1", 0.6, _heavy(2.1), n=400, scale="level", bounds=UNIT)
    assert estimate.small_sample_df < 5.0
    assert estimate.inference == "influence_curve"


def test_replace_keeps_the_status_and_so_does_an_inheriting_contrast() -> None:
    withheld = replace(_mean(0.9995, se=5e-5), inference=CONSTANT_ARM_CELL_STATUS)
    healed = replace(withheld, influence_curve=_curve(0.03), variance=0.03**2)
    assert healed.inference == CONSTANT_ARM_CELL_STATUS
    other = make_estimate("ey0", 0.6, _curve(0.03, seed=1), n=400, scale="level", bounds=UNIT)
    contrast = smooth_contrast(
        {"ey1": withheld, "ey0": other},
        lambda v: float(v[0] - v[1]),
        ("ey1", "ey0"),
        n=400,
        gradient=lambda v: np.array([1.0, -1.0]),
    )
    assert contrast.inference == CONSTANT_ARM_CELL_STATUS
    assert replace(contrast, influence_curve=_curve(0.01)).inference == CONSTANT_ARM_CELL_STATUS


def test_the_precedence_of_the_two_statuses() -> None:
    assert bounded_status("estimated_weight_plugin", 0.6, UNIT) == "estimated_weight_plugin"
    assert (
        bounded_status("estimated_weight_plugin", 0.6, UNIT, constant_cell=True)
        == CONSTANT_ARM_CELL_STATUS
    )
    assert bounded_status("working_model_projection", 1.0, UNIT) == OUT_OF_RANGE_STATUS
    assert (
        bounded_status("constant_node_plugin", 0.6, UNIT, constant_cell=True)
        == "constant_node_plugin"
    )
    assert bounded_status("influence_curve", 1.0, UNIT, constant_cell=True) == (
        CONSTANT_ARM_CELL_STATUS
    )
    # A labelled shown estimate with a heavy curve keeps its status and its normal interval.
    labelled = make_estimate(
        "ey1",
        2.5,
        _heavy(3.0),
        n=400,
        scale="level",
        inference="estimated_weight_plugin",
        outcome_family="gaussian",
    )
    assert labelled.inference == "estimated_weight_plugin"
    assert labelled.ci == wald_ci(2.5, labelled.std_error)


# ----------------------------------------------------------------------------------- T8


def test_a_clustered_bounded_mean_keeps_its_cluster_reference() -> None:
    curve = _curve(0.03, n=60)
    cluster = np.repeat(np.arange(20), 3)
    reference = ClusterReference(cluster)
    estimate = make_estimate(
        "ey1",
        0.9,
        curve,
        n=60,
        cluster=cluster,
        scale="level",
        bounds=UNIT,
        reference_df=reference.reference_df("ey1"),
    )
    assert estimate.reference_df == reference.reference_df("ey1") == 18
    assert estimate.small_sample_df is None
    assert estimate.interval_scale == "identity"
    assert estimate.ci == wald_ci(0.9, estimate.std_error, df=18)


# ----------------------------------------------------------------------------------- T9


def _fixtures() -> list[Any]:
    rng = np.random.default_rng(20261113)
    out = []
    for k in range(200):
        se = float(rng.uniform(0.005, 0.08))
        curve = _curve(se, seed=k)
        kind = k % 4
        if kind == 0:
            out.append(
                make_estimate(
                    "ey1", float(rng.uniform(0.05, 0.95)), curve, n=400, scale="level", bounds=UNIT
                )
            )
        elif kind == 1:
            out.append(
                make_estimate(
                    "ate",
                    float(rng.uniform(-0.5, 0.5)),
                    curve,
                    n=400,
                    scale="difference",
                    bounds=(-1.0, 1.0),
                )
            )
        elif kind == 2:
            psi = float(rng.uniform(0.5, 2.0))
            out.append(make_estimate("rr", psi, curve, n=400, scale="ratio", log_psi=math.log(psi)))
        else:
            base = make_estimate("p", float(rng.uniform(0.2, 0.8)), curve, n=400, scale="level")
            out.append(replace(base, transform=Transform.logit()))
    return out


def test_each_interval_end_is_the_null_its_test_puts_at_alpha() -> None:
    scales = set()
    for estimate in _fixtures():
        scales.add(estimate.interval_scale)
        for end in estimate.ci:
            assert estimate.wald_test(end).pvalue == pytest.approx(0.05, rel=1e-7)
        assert estimate.wald_test(estimate.psi).pvalue == pytest.approx(1.0)
        test = estimate.wald_test(estimate.psi)
        assert test.interval_scale == estimate.interval_scale
    assert scales == {"range_logit", "identity", "log", "logit"}


def test_the_default_null_of_a_bounded_mean_is_outside_its_range() -> None:
    estimate = _mean(0.3, se=0.2)
    assert estimate.pvalue == 0.0
    assert estimate.wald_test().statistic == math.inf
    assert estimate.ci[0] > 0.0
    assert estimate.wald_test(1.0).pvalue == 0.0
    assert estimate.wald_test(-0.1).pvalue == 0.0


# ----------------------------------------------------------------------------------- T10


def test_an_out_of_range_member_keeps_the_rule_it_had() -> None:
    edge = make_estimate("ey1", 1.0, _curve(0.02), n=400, scale="level", bounds=UNIT)
    assert edge.inference == OUT_OF_RANGE_STATUS
    other = make_estimate("ey0", 0.6, _curve(0.03, seed=1), n=400, scale="level", bounds=UNIT)
    estimates = {"ey1": edge, "ey0": other}
    assert out_of_range(estimates, "logit") == ("ey1",)
    with pytest.raises(CapabilityError):
        simultaneous_bands(estimates, n_replicates=500, random_state=0, band_scale="logit")
    band = simultaneous_bands(estimates, n_replicates=500, random_state=0, band_scale="identity")
    low, high = band.bands["ey1"]
    assert (high - low) / 2.0 == pytest.approx(
        band.critical_value * edge.plugin_std_error, rel=1e-12
    )


# ----------------------------------------------------------------------------------- T18a


@pytest.fixture(scope="module")
def constant_cell_fit() -> Any:
    frame = law.sample(500, CONSTANT_CELL_SEED)
    return stratified_tmle.fit_cleverly(frame, simultaneous=True)


def test_the_constant_cell_witness_withholds_the_mean_and_its_contrast(
    constant_cell_fit: Any,
) -> None:
    estimates = constant_cell_fit.estimates
    assert estimates["ey[1][V=2]"].inference == CONSTANT_ARM_CELL_STATUS
    assert estimates["ate[V=2]"].inference == CONSTANT_ARM_CELL_STATUS
    assert estimates["ey[0][V=2]"].inference == "influence_curve"
    assert estimates["ey[1]"].inference == "influence_curve"
    assert constant_cell_fit.simultaneous is None
    summary = str(constant_cell_fit.summary(detail="full"))
    assert NO_CONSTANT_ARM_BANDS.format(names="ate[V=2], ey[1][V=2]")[:60] in " ".join(
        summary.split()
    )


def test_the_builder_call_patched_out_shows_a_narrow_interval_that_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    tmle_module = importlib.import_module("cleverly.estimators.tmle")
    monkeypatch.setattr(tmle_module, "constant_outcome_cells", lambda *a, **k: frozenset())
    frame = law.sample(500, CONSTANT_CELL_SEED)
    estimate = stratified_tmle.fit_cleverly(frame)["ey[1][V=2]"]
    low, high = estimate.ci
    assert high - low < 1e-3
    assert not low <= law.TRUTH["ey[1][V=2]"] <= high


def test_a_healthy_draw_takes_no_status_and_builds_its_band() -> None:
    frame = law.sample(2_000, 0)
    fit = stratified_tmle.fit_cleverly(frame, simultaneous=True)
    assert all(e.inference == "influence_curve" for e in fit.estimates.values())
    assert fit.simultaneous is not None


# ----------------------------------------------------------------------------------- T18b


@pytest.mark.parametrize("psi", [1.0, 1.0 - 1e-17, 0.0, -0.01])
def test_an_estimate_on_its_bound_takes_the_out_of_range_status(psi: float) -> None:
    estimate = make_estimate("ey1", psi, _curve(0.02), n=400, scale="level", bounds=UNIT)
    assert estimate.inference == OUT_OF_RANGE_STATUS
    with pytest.raises(CapabilityError) as raised:
        _ = estimate.ci
    text = str(raised.value)
    assert status_record(OUT_OF_RANGE_STATUS).reason in text
    assert status_record(CONSTANT_ARM_CELL_STATUS).reason not in text
    other = make_estimate("ey0", 0.6, _curve(0.03, seed=1), n=400, scale="level", bounds=UNIT)
    difference = smooth_contrast(
        {"ey1": estimate, "ey0": other},
        lambda v: float(v[0] - v[1]),
        ("ey1", "ey0"),
        n=400,
        gradient=lambda v: np.array([1.0, -1.0]),
    )
    assert difference.inference == OUT_OF_RANGE_STATUS


def test_a_ratio_inherits_the_status_of_a_level() -> None:
    edge = make_estimate("ey1", 1.0, _curve(0.02), n=400, scale="level", bounds=UNIT)
    other = make_estimate("ey0", 0.6, _curve(0.03, seed=1), n=400, scale="level", bounds=UNIT)
    ratio = ratio_contrast({"ey1": edge, "ey0": other}, "ey1", "ey0", n=400, name="rr")
    assert ratio.inference == OUT_OF_RANGE_STATUS


# ----------------------------------------------------------------------------------- T21


@pytest.mark.parametrize("status", [CONSTANT_ARM_CELL_STATUS, OUT_OF_RANGE_STATUS])
def test_the_joint_heterogeneity_test_refuses_a_withheld_coefficient(status: str) -> None:
    from tests.unit.test_conditional_contrast import law_study, linear_contrast, oracle

    fit = law_study().identify(linear_contrast()).estimate(**oracle())
    name = next(iter(fit.estimates))
    withheld = replace(
        fit, estimates={**fit.estimates, name: replace(fit.estimates[name], inference=status)}
    )
    with pytest.raises(CapabilityError) as raised:
        withheld.heterogeneity_test()
    assert status_record(status).reason in str(raised.value)


# ----------------------------------------------------------------------------------- T22


def test_the_summary_names_the_logit_scale(constant_cell_fit: Any) -> None:
    summary = str(constant_cell_fit.summary())
    assert "log-odds scale" in summary
    record = constant_cell_fit.estimates["ey[1]"].to_dict()
    assert record["interval_scale"] == "range_logit"
    assert constant_cell_fit.estimates["ate"].to_dict()["interval_scale"] == "identity"


# ----------------------------------------------------------------------------------- T23


def test_a_status_in_one_draw_withholds_every_draw() -> None:
    shown = _mean(0.6, se=0.03, inference="estimated_weight_plugin")
    withheld = replace(shown, inference=CONSTANT_ARM_CELL_STATUS)
    draws = [{"ey1": shown}, {"ey1": withheld}, {"ey1": shown}]
    for stamped in (with_precedent_status(draws), restamp_parameter_statuses(draws)):
        assert {draw["ey1"].inference for draw in stamped} == {CONSTANT_ARM_CELL_STATUS}


def test_a_fold_estimate_takes_no_status_from_its_own_rows() -> None:
    from cleverly.estimators.tmle import _carry_cell_status

    fold = make_estimate("ey1", 0.6, _curve(0.03), n=400, scale="level")
    fold = replace(fold, bounds=UNIT)
    pooled = {"ey1": replace(_mean(0.6), outcome_family="binomial")}
    carried = _carry_cell_status({"ey1": fold}, pooled)["ey1"]
    assert carried.inference == "influence_curve"
    assert carried.outcome_family == "binomial"
    pooled = {"ey1": replace(_mean(0.6), inference=CONSTANT_ARM_CELL_STATUS)}
    assert _carry_cell_status({"ey1": fold}, pooled)["ey1"].inference == CONSTANT_ARM_CELL_STATUS


# ----------------------------------------------------------------------------------- T24


def test_the_logit_is_never_read_at_or_beyond_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    original = Transform.range_logit.__func__  # type: ignore[attr-defined]

    def spy(cls: Any, low: float, high: float) -> Transform:
        transform = original(cls, low, high)

        def forward(value: float) -> float:
            seen.append(float(value))
            return transform.forward(value)

        return Transform(transform.name, forward, transform.inverse, transform.derivative)

    monkeypatch.setattr(Transform, "range_logit", classmethod(spy))
    for psi in (0.0, 1.0, 1.0 - 1e-17, 1.02, 0.5, 0.999999):
        estimate = make_estimate("ey1", psi, _curve(0.02), n=400, scale="level", bounds=UNIT)
        shown = replace(estimate, inference="influence_curve")
        _ = shown.ci, shown.plugin_interval, shown._plugin_pvalue(), shown.wald_test(0.5)
        assert all(math.isfinite(limit) for limit in shown.ci)
    assert seen
    assert all(0.0 < value < 1.0 for value in seen)


def test_an_estimate_just_inside_its_bound_shows_the_whole_range() -> None:
    estimate = make_estimate("ey1", 0.999999999999, _curve(0.1), n=400, scale="level", bounds=UNIT)
    assert estimate.inference == "influence_curve"
    assert estimate.ci == (0.0, 1.0)
    assert all(estimate.ci[0] <= truth <= estimate.ci[1] for truth in (1e-9, 0.5, 1 - 1e-9))


# ----------------------------------------------------------------------------------- T25


def _gaussian_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    w = rng.normal(size=n)
    a = rng.binomial(1, expit(0.3 * w))
    return pd.DataFrame({"W": w, "A": a, "Y": 1.0 + a + w + rng.normal(size=n)})


def test_the_tmle_builder_records_the_family_of_the_fit() -> None:
    frame = _gaussian_frame()
    gaussian = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )
    assert {e.outcome_family for e in gaussian.estimates.values()} == {"gaussian"}
    assert {e.interval_scale for e in gaussian.estimates.values()} == {"identity"}
    binary = frame.assign(Y=(frame["Y"] > 1.5).astype(float))
    fit = (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(binary, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )
    assert {e.outcome_family for e in fit.estimates.values()} == {"binomial"}
    assert fit["ey1"].interval_scale == "range_logit"


#: The call sites that record the outcome family, and those that leave it ``None`` by decision
#: (plan 3.1). A derived estimate takes the family its inputs share. Each entry is
#: ``(path, enclosing function)``.
STAMPED = {
    ("src/cleverly/estimators/_strata.py", "stratum_mixture"),
    ("src/cleverly/longitudinal/structural.py", "build"),
    ("src/cleverly/effect_curves.py", "estimate"),
    ("src/cleverly/estimators/iptw.py", "_assemble"),
    ("src/cleverly/estimators/iptw.py", "with_error"),
    ("src/cleverly/estimators/atmle_result.py", "fit_trial_effect"),
    ("src/cleverly/longitudinal/estimator.py", "_estimates"),
    ("src/cleverly/longitudinal/estimator.py", "_msm_estimates"),
    ("src/cleverly/estimators/tmle.py", "_stratum_estimates"),
}
#: Left ``None``: a dose-response point has a continuous treatment; a custom target's
#: functional need not be an outcome mean (the TMLE loop stamps only built-in targets after
#: ``finish``); a sensitivity tilt is a sensitivity quantity; ``ratio_estimates`` has no caller.
LEFT_NONE = {
    ("src/cleverly/estimators/dose_response.py", "_target"),
    ("src/cleverly/targets/base.py", "finish"),
    ("src/cleverly/targets/base.py", "finish_unscaled"),
    ("src/cleverly/sensitivity/_tilt_targeting.py", "finish"),
    ("src/cleverly/inference/influence.py", "ratio_estimates"),
}
#: Builders that carry the family of their inputs.
DERIVED = {
    ("src/cleverly/inference/influence.py", "make_estimate"),
    ("src/cleverly/inference/influence.py", "median_estimates"),
    ("src/cleverly/inference/results.py", "smooth_contrast"),
    ("src/cleverly/inference/results.py", "transform_estimate"),
    ("src/cleverly/inference/results.py", "ratio_contrast"),
    # The fold-evaluated estimate takes the family of the whole-sample estimate of its name
    # (``_carry_cell_status``).
    ("src/cleverly/estimators/tmle.py", "_average_over_folds"),
}


def _call_sites() -> set[tuple[str, str]]:
    sites = set()
    for path in sorted((ROOT / "src" / "cleverly").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(ROOT))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("make_estimate", "ParameterEstimate")
                ):
                    sites.add((relative, function.name))
    return sites


def test_every_estimate_builder_has_a_family_decision() -> None:
    sites = _call_sites()
    # An inner function is reported with its enclosing ones too; each site needs one decision.
    decided = STAMPED | LEFT_NONE | DERIVED
    undecided = {site for site in sites if site not in decided}
    enclosing = {
        site
        for site in undecided
        if any(other[0] == site[0] and other in decided for other in sites)
    }
    assert not undecided - enclosing, sorted(undecided - enclosing)
    assert decided <= sites, sorted(decided - sites)


def test_every_stamped_site_passes_a_family() -> None:
    for path, name in STAMPED:
        source = (ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        )
        text = ast.get_source_segment(source, function) or ""
        assert re.search(r"outcome_family\s*=", text), (path, name)


def test_a_custom_target_records_no_family() -> None:
    from cleverly.targets.builtin import BUILTIN_TARGETS

    assert all(target.name for target in BUILTIN_TARGETS)
    frame = _gaussian_frame()
    fit = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )
    # The built-in targets of a continuous outcome record "gaussian"; a target outside the
    # built-in list keeps None, so the t reference PR 3 reads stays opt-in.
    assert fit["ate"].outcome_family == "gaussian"
    custom = make_estimate("custom", 1.0, _curve(0.1), n=400)
    assert custom.outcome_family is None
    assert custom.effective_df is None
