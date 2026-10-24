"""The score check and the summary read the stopping rule a fit declared.

G5 of the stopping-rule plan reports on what G3 and G4 built.  Under ``Targeting(stopping=)``
set to ``"scaled_var"`` or ``"sample_size"``:

* a fluctuation row is held to the threshold the solver stopped at, so a fit that met its rule
  passes although its score is above the default bar, and a fit that missed its rule fails
  although its score is below the default bar;
* an estimand row is a ``"diagnostic"`` that cannot fail.  It carries ``residual_se``, the mean
  of the influence curve in standard errors, ``coverage_bound``, the coverage a Wald interval
  keeps at that residual, and, under ``scaled_var``, ``rule_bound_se``, the residual the rule
  implies before the fit is read;
* ``summary()`` prints one pinned line, and a second pinned line when the rule was not met.

Every number below is recomputed by hand with ``numpy``, ``pandas`` or ``scipy``.  The plan
fixes the two lines of text, so the tests compare against the literal text and not against the
constants that produce it.
"""

from __future__ import annotations

import functools
import importlib
import math
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cleverly._summary import vocabulary
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import DRTMLE, TMLE
from cleverly.fluctuation._score import (
    RESIDUAL_LINE,
    RULE_FAILURE_LINE,
    residual_line,
    residual_se,
    rule_failure_line,
    wald_coverage_bound,
)
from cleverly.longitudinal import LTMLE
from cleverly.validation.score import DEFAULT_TOLERANCE, score_check, score_threshold
from tests.conftest import linear_drtmle, linear_in_sample, mean_one_weights
from tests.unit import test_stopping_rule_engines as engines
from tests.unit import test_stopping_rule_longitudinal as longitudinal
from tests.unit._source_mutation import mutate

score_module = importlib.import_module("cleverly.validation.score")

RULES = ("scaled_var", "sample_size")


# ------------------------------------------------------------------------------ the fits


@functools.cache
def one_step(rule: str, *, weighted: bool = False) -> Any:
    """A one-step fit that stops at its rule, with a score well above the default bar."""
    frame, _ = make_linear_ate(n=400, seed=1)
    columns: dict[str, Any] = {}
    if weighted:
        frame = frame.assign(weight=mean_one_weights(len(frame)))
        columns = {"weights": "weight"}
    estimator = TMLE(**linear_in_sample(stopping=rule, targeting="one_step", step_size=0.01))
    return estimator.fit(frame, outcome="Y", treatment="A", **columns).single()


@functools.cache
def prognostic(rule: str = "scaled_var") -> Any:
    """``W ~ N(0, 1)``, ``A ~ Bern(1/2)``, ``Y = W + A + 0.1 eps``.

    The outcome is almost determined by ``W``, so the arm curves are large and nearly equal
    and the ATE curve, which cancels them, is small: ``(se_1 + se_0) / se_ATE`` is about ten.
    """
    rng = np.random.default_rng(3)
    n = 2000
    w = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n)
    y = w + a + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({"W": w, "A": a, "Y": y})
    estimator = TMLE(**linear_in_sample(stopping=rule))
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def unit_scale(n_units: int) -> float:
    """``c = 1 / min(log n_u, 10)``, the factor between a threshold and its column's SE."""
    return 1.0 / min(float(np.log(n_units)), 10.0)


def hand_residual(estimate: Any) -> float:
    """``|P_n IC| / se`` from the stored influence curve."""
    return abs(float(np.mean(estimate.influence_curve))) / estimate.plugin_std_error


def hand_coverage(residual: float, alpha: float) -> float:
    """``Phi(z - b) - Phi(-z - b)`` with ``scipy``."""
    z = norm.ppf(1.0 - alpha / 2.0)
    return float(norm.cdf(z - residual) - norm.cdf(-z - residual))


def rule_note(result: Any) -> str:
    """The note a point summary prints under a declared rule, computed by hand.

    The remainder is ``|P_n IC|`` over the standard error each interval uses, the largest over
    the estimates that report one; the coverage is rounded down to two decimals of a percent.
    """
    residuals = {
        name: abs(float(estimate.score)) / float(estimate.std_error)
        for name, estimate in result.estimates.items()
        if estimate.supplies_inference and float(estimate.std_error) > 0.0
    }
    if not residuals:
        return vocabulary.NOTE_TEXTS["stopping_rule_plain"]
    worst = max(residuals, key=lambda name: residuals[name])
    alpha = float(result.estimates[worst].alpha)
    b = residuals[worst]
    coverage = math.floor(hand_coverage(max(b, 0.001), alpha) * 10000) / 100
    return vocabulary.NOTE_TEXTS["stopping_rule"].format(
        b="less than 0.001" if b < 0.001 else f"{b:.3g}",
        name=worst,
        level=f"{(1.0 - alpha) * 100:g}%",
        bound=f"{coverage:.2f}%",
    )


def with_fluctuation(result: Any, group: str, **changes: Any) -> Any:
    """``result`` with one fluctuation changed, without refitting."""
    repeat = result.repeats[0]
    changed = replace(repeat.fluctuations[group], **changes)
    fluctuations = {**repeat.fluctuations, group: changed}
    return replace(result, repeats=(replace(repeat, fluctuations=fluctuations),))


# ------------------------------------------------------------------------------ the text


class TestThePinnedText:
    def test_the_residual_line_is_the_text_of_the_plan(self) -> None:
        assert RESIDUAL_LINE == (
            "stopping rule {rule}: the largest score residual is {b:.3g} standard errors, "
            "on {name}. At that residual a {level:.0%} Wald interval keeps at least "
            "{bound:.4f} coverage."
        )

    def test_the_failure_line_is_the_text_of_the_plan(self) -> None:
        assert RULE_FAILURE_LINE == "the targeting did not meet the {rule} stopping rule; {detail}"
        assert (
            rule_failure_line("sample_size", "mean is 2 times its threshold")
            == "the targeting did not meet the sample_size stopping rule; "
            "mean is 2 times its threshold"
        )

    @pytest.mark.parametrize("residual", [0.0, 0.1, 0.5, 1.009, 3.0])
    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    def test_the_coverage_bound_is_the_normal_formula(self, residual: float, alpha: float) -> None:
        assert wald_coverage_bound(residual, alpha) == pytest.approx(
            hand_coverage(residual, alpha), abs=1e-12
        )
        assert wald_coverage_bound(-residual, alpha) == wald_coverage_bound(residual, alpha)

    def test_the_coverage_bound_at_the_values_the_plan_quotes(self) -> None:
        assert f"{wald_coverage_bound(0.1, 0.05):.4f}" == "0.9489"
        assert f"{wald_coverage_bound(1.009, 0.05):.4f}" == "0.8277"
        assert np.isnan(wald_coverage_bound(float("nan"), 0.05))

    def test_a_residual_line_names_the_worst_estimand_at_the_fit_level(self) -> None:
        estimates = one_step("scaled_var").estimates
        residuals = {name: hand_residual(e) for name, e in estimates.items()}
        worst = max(residuals, key=lambda name: residuals[name])
        expected = (
            f"stopping rule scaled_var: the largest score residual is {residuals[worst]:.3g} "
            f"standard errors, on {worst}. At that residual a 95% Wald interval keeps at least "
            f"{hand_coverage(residuals[worst], 0.05):.4f} coverage."
        )
        assert residual_line(estimates, "scaled_var", 0.05) == expected

    def test_a_residual_line_is_empty_without_a_finite_residual(self) -> None:
        flat = SimpleNamespace(plugin_std_error=0.0, score=1.0)
        assert np.isnan(residual_se(flat))  # type: ignore[arg-type]
        assert np.isnan(residual_se(SimpleNamespace(plugin_std_error=float("nan"), score=1.0)))  # type: ignore[arg-type]
        assert residual_line({}, "scaled_var", 0.05) == ""


# ------------------------------------------------------------------------------ the verdict


class TestTheVerdictReadsTheRule:
    @pytest.mark.parametrize("rule", RULES)
    def test_a_fit_that_met_its_rule_passes_above_the_default_bar(self, rule: str) -> None:
        result = one_step(rule)
        check = score_check(result)
        # The witness: by the default bar, written out, every fluctuation row would fail.
        bar = score_threshold(
            (e.plugin_std_error for e in result.estimates.values()), result.data.n
        )
        scores = [f.score_norm for f in result.fluctuations.values()]
        assert min(scores) > bar
        assert check.passed
        assert check.declared and check.stopping == rule
        rows = [row for row in check.rows if row.kind == "fluctuation"]
        assert {row.name for row in rows} == set(result.fluctuations)
        for row in rows:
            fluctuation = result.fluctuations[row.name]
            assert row.stopping == rule and row.passed
            assert row.ratio <= 1.0
            assert fluctuation.rule_ratio is not None and fluctuation.rule_ratio <= 1.0

    def test_a_fluctuation_row_is_the_column_with_the_largest_ratio_in_outcome_units(self) -> None:
        result = one_step("scaled_var")
        span = result.repeats[0].nuisance.scaler.range
        check = score_check(result)
        for row in (r for r in check.rows if r.kind == "fluctuation"):
            fluctuation = result.fluctuations[row.name]
            ratios = np.abs(fluctuation.score) * span / fluctuation.thresholds
            column = int(np.argmax(ratios))
            assert row.score == pytest.approx(abs(fluctuation.score[column]) * span, rel=1e-12)
            assert row.threshold == pytest.approx(fluctuation.thresholds[column], rel=1e-12)
            assert row.ratio == pytest.approx(ratios[column], rel=1e-12)
            assert row.ratio == pytest.approx(fluctuation.rule_ratio, rel=1e-9)
            assert row.score_initial == pytest.approx(
                fluctuation.initial_score_norm * span, rel=1e-12
            )

    def test_the_mutation_that_reads_the_default_bar_fails_that_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert score_check(one_step("scaled_var")).passed
        monkeypatch.setattr(score_module, "_judged_by_rule", lambda fluctuation: False)
        broken = score_check(one_step("scaled_var"))
        assert not broken.passed

    @pytest.mark.parametrize("rule", RULES)
    def test_a_fit_that_missed_its_rule_fails_below_the_default_bar(self, rule: str) -> None:
        result = one_step(rule)
        group = "att"
        fluctuation = result.fluctuations[group]
        # Score at zero (under any bar), rule ratio above one: the rule was not met.
        zeros = np.zeros_like(fluctuation.score)
        missed = with_fluctuation(result, group, score=zeros, rule_ratio=1.75, converged=False)
        check = score_check(missed)
        assert missed.repeats[0].fluctuations[group].score_norm == 0.0
        row = next(r for r in check.rows if r.name == group and r.kind == "fluctuation")
        assert not row.passed and not check.passed
        assert check.rule_failures == (row,)
        detail = f"{group} is 1.75 times its threshold. See res.diagnostics.score_equations()"
        expected = f"the targeting did not meet the {rule} stopping rule; {detail}"
        assert check.rule_failure_line() == expected
        assert check.one_line() == expected
        # The estimand rows stay diagnostics and cannot fail.
        diagnostics = [r for r in check.rows if r.kind == "diagnostic"]
        assert diagnostics and all(r.passed for r in diagnostics)

    def test_the_failing_fit_prints_both_pinned_lines(self) -> None:
        result = one_step("scaled_var")
        missed = with_fluctuation(
            result,
            "att",
            score=np.zeros_like(result.fluctuations["att"].score),
            rule_ratio=1.75,
            converged=False,
        )
        summary = missed.summary()
        assert summary.note("stopping_rule").text == rule_note(missed)
        unmet = summary.note("score_unsolved")
        assert unmet.kind == "warning"
        assert unmet.text == vocabulary.NOTE_TEXTS["score_rule_unmet"].format(
            names="att", spread="interval"
        )
        assert "FAIL" not in summary
        assert "PASS" not in summary

    def test_the_verdict_text_says_the_rule_and_not_the_machine_precision_root(self) -> None:
        text = score_check(one_step("sample_size")).summary()
        assert "met the sample_size stopping rule" in text
        assert "not to machine precision" in text
        assert "the theory does not license" not in text

    def test_a_default_fit_reads_the_default_bar_and_prints_nothing_new(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=1)
        result = (
            TMLE(**linear_in_sample(targeting="one_step", step_size=0.01))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        check = score_check(result)
        assert not check.declared and check.stopping == "relative"
        assert check.rule_line() == "" and check.rule_failure_line() == ""
        assert [r.kind for r in check.rows] == ["fluctuation"] * 3 + ["influence curve"] * 5
        assert all(r.stopping == "relative" for r in check.rows)
        assert all(np.isnan(r.residual_se) for r in check.rows)
        columns = list(check.to_frame().columns)
        assert not {"stopping", "residual_se", "coverage_bound", "rule_bound_se"} & set(columns)
        assert "stopping rule" not in result.summary()


# ------------------------------------------------------------------------------ the diagnostics


class TestTheEstimandRows:
    @pytest.mark.parametrize("rule", RULES)
    def test_every_estimand_row_is_a_diagnostic_with_the_hand_residual(self, rule: str) -> None:
        result = one_step(rule)
        check = score_check(result)
        rows = {r.name: r for r in check.rows if r.kind == "diagnostic"}
        assert set(rows) == set(result.estimates)
        for name, estimate in result.estimates.items():
            row = rows[name]
            hand = hand_residual(estimate)
            assert row.passed and row.converged
            assert row.residual_se == pytest.approx(hand, rel=1e-12)
            assert row.residual_se == pytest.approx(abs(estimate.score) / estimate.std_error)
            assert row.coverage_bound == pytest.approx(hand_coverage(hand, 0.05), abs=1e-12)
            assert 0.0 < row.coverage_bound < 0.95
        assert max(r.residual_se for r in rows.values()) > 1e-3

    def test_the_coverage_bound_uses_the_level_of_the_fit(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=1)
        estimator = TMLE(
            **linear_in_sample(
                stopping="scaled_var", targeting="one_step", step_size=0.01, alpha_sig=0.1
            )
        )
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
        check = score_check(result)
        for row in (r for r in check.rows if r.kind == "diagnostic"):
            assert row.coverage_bound == pytest.approx(
                hand_coverage(row.residual_se, 0.1), abs=1e-12
            )
        note = result.summary().note("stopping_rule").text
        assert note == rule_note(result)
        assert "a 90% interval" in note

    def test_an_estimand_row_cannot_fail_under_a_rule(self) -> None:
        result = one_step("scaled_var")
        # An influence curve with a mean far from zero is a large residual and not a failure.
        estimate = result.estimates["ey1"]
        shifted = replace(
            estimate, influence_curve=estimate.influence_curve + 5.0 * estimate.std_error
        )
        broken = replace(result, estimates={**result.estimates, "ey1": shifted})
        row = next(
            r for r in score_check(broken).rows if r.name == "ey1" and r.kind == "diagnostic"
        )
        assert row.passed and row.residual_se > 4.0
        assert row.coverage_bound < 0.1
        assert score_check(broken).passed

    def test_the_relative_rule_keeps_the_estimand_bar_and_kind(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=1)
        result = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        check = score_check(result)
        for name, estimate in result.estimates.items():
            row = next(r for r in check.rows if r.name == name and r.kind == "influence curve")
            bar = DEFAULT_TOLERANCE * estimate.plugin_std_error / np.sqrt(result.data.n)
            assert row.threshold == pytest.approx(bar, rel=1e-12)
            assert row.passed == (abs(estimate.score) <= row.threshold)


class TestTheRuleBound:
    def test_a_unit_weight_arm_row_equals_c(self) -> None:
        result = one_step("scaled_var")
        c = unit_scale(result.data.n)
        rows = {r.name: r for r in score_check(result).rows if r.kind == "diagnostic"}
        for name in ("ey1", "ey0"):
            assert rows[name].rule_bound_se == pytest.approx(c, abs=1e-12)

    def test_a_score_column_arm_row_is_about_c(self) -> None:
        result = one_step("scaled_var")
        c = unit_scale(result.data.n)
        rows = {r.name: r for r in score_check(result).rows if r.kind == "diagnostic"}
        for name in ("att", "atc"):
            assert rows[name].rule_bound_se == pytest.approx(c, rel=0.01)

    def test_the_bound_is_not_asserted_for_sample_size(self) -> None:
        rows = score_check(one_step("sample_size")).rows
        assert all(np.isnan(r.rule_bound_se) for r in rows)

    def test_a_weighted_arm_row_is_the_threshold_over_the_weighted_se(self) -> None:
        result = one_step("scaled_var", weighted=True)
        fluctuation = result.fluctuations["mean"]
        rows = {r.name: r for r in score_check(result).rows if r.kind == "diagnostic"}
        for column, name in ((0, "ey0"), (1, "ey1")):
            hand = fluctuation.thresholds[column] / result.estimates[name].plugin_std_error
            assert rows[name].rule_bound_se == pytest.approx(hand, rel=1e-12)
        # Weighted rows are not the unit-weight value, and the plan does not claim they are.
        c = unit_scale(result.data.n)
        assert abs(rows["ey1"].rule_bound_se - c) > 1e-6

    def test_a_contrast_has_the_bound_of_its_arms_and_not_sqrt_2_c(self) -> None:
        result = prognostic()
        n = result.data.n
        c = unit_scale(n)
        se1 = result.estimates["ey1"].plugin_std_error
        se0 = result.estimates["ey0"].plugin_std_error
        se_ate = result.estimates["ate"].plugin_std_error
        ratio = (se1 + se0) / se_ate
        # The witness: the arm curves are large and the contrast cancels them.
        assert ratio > 3.0
        rows = {r.name: r for r in score_check(result).rows if r.kind == "diagnostic"}
        assert rows["ate"].rule_bound_se == pytest.approx(c * ratio, rel=1e-6)
        assert rows["ate"].rule_bound_se > np.sqrt(2.0) * c
        assert rows["ate"].rule_bound_se == pytest.approx(
            (result.fluctuations["mean"].thresholds[1] + result.fluctuations["mean"].thresholds[0])
            / se_ate,
            rel=1e-12,
        )
        for name in ("ey1", "ey0"):
            assert rows[name].rule_bound_se == pytest.approx(c, abs=1e-12)

    def test_the_sqrt_2_c_mutation_fails_the_contrast_test(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = prognostic()
        c = unit_scale(result.data.n)
        ratio = (
            result.estimates["ey1"].plugin_std_error + result.estimates["ey0"].plugin_std_error
        ) / result.estimates["ate"].plugin_std_error
        monkeypatch.setattr(
            score_module, "_rule_bound_se", lambda result, name, estimate: np.sqrt(2.0) * c
        )
        row = next(
            r for r in score_check(result).rows if r.name == "ate" and r.kind == "diagnostic"
        )
        assert row.rule_bound_se != pytest.approx(c * ratio, rel=1e-6)

    def test_the_mutation_that_drops_the_weight_of_the_second_arm_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = prognostic()
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "_rule_bound_se",
            [("abs(g) * float", "abs(max(g, 0.0)) * float")],
        )
        row = next(
            r for r in score_check(result).rows if r.name == "ate" and r.kind == "diagnostic"
        )
        c = unit_scale(result.data.n)
        ratio = (
            result.estimates["ey1"].plugin_std_error + result.estimates["ey0"].plugin_std_error
        ) / result.estimates["ate"].plugin_std_error
        assert row.rule_bound_se != pytest.approx(c * ratio, rel=1e-3)

    @staticmethod
    @functools.cache
    def stratified(walk: bool = False) -> Any:
        frame, columns, settings, _ = engines.KINDS["strata"]()
        extra = {"targeting": "one_step", "step_size": 0.01} if walk else {}
        return engines._fit("scaled_var", frame, columns=columns, **settings, **extra)

    def test_a_stratified_row_sums_the_blocks_of_its_strata(self) -> None:
        # Review finding F2: a stratified group holds one block per stratum and no marginal
        # column.  A marginal row weights each stratum's block by P_n(S = s), times the arm
        # share ratio for a conditional effect; a stratum row reads its own block.
        result = self.stratified()
        data = result.data
        p = [float(np.mean(data.strata == code)) for code in range(data.n_strata)]
        labels = [data.stratum_label(code) for code in range(data.n_strata)]
        rows = {r.name: r for r in score_check(result).rows if r.kind == "diagnostic"}

        def threshold(group: str, column: str) -> float:
            fluctuation = result.fluctuations[group]
            return float(fluctuation.thresholds[fluctuation.names.index(column)])

        def se(name: str) -> float:
            return float(result.estimates[name].plugin_std_error)

        a = np.asarray(data.treatment)
        for name, columns in (("ey1", ("h1",)), ("ey0", ("h0",)), ("ate", ("h1", "h0"))):
            hand = sum(
                p[s] * threshold("mean", f"{c} | {labels[s]}") for s in range(2) for c in columns
            )
            assert rows[name].rule_bound_se == pytest.approx(hand / se(name), rel=1e-12)
            for s in range(2):
                stratum = f"{name}[{labels[s]}]"
                own = sum(threshold("mean", f"{c} | {labels[s]}") for c in columns)
                assert rows[stratum].rule_bound_se == pytest.approx(own / se(stratum), rel=1e-12)
        for name, arm in (("att", 1.0), ("atc", 0.0)):
            share = float(np.mean(a == arm))
            hand = sum(
                p[s]
                * float(np.mean(a[data.strata == s] == arm))
                / share
                * threshold(name, f"h_{name} | {labels[s]}")
                for s in range(2)
            )
            assert rows[name].rule_bound_se == pytest.approx(hand / se(name), rel=1e-12)
        # Mutation: the positional reading of the unstratified fit reads the first stratum's
        # blocks for a marginal row.
        positional = threshold("mean", f"h1 | {labels[0]}") / se("ey1")
        assert rows["ey1"].rule_bound_se != pytest.approx(positional, rel=1e-3)

    def test_the_stratified_weights_reproduce_each_curve_mean(self) -> None:
        # The weights are exact: on a walk that leaves a residual, sum_j g_j P_n S_j equals the
        # mean of each reported curve, marginal and stratum rows alike.
        result = self.stratified(walk=True)
        span = result.nuisance.scaler.range
        checked = 0
        for name, estimate in result.estimates.items():
            mapped = score_module._column_weights(result, name)
            if mapped is None:
                continue
            group, weights = mapped
            score = result.fluctuations[group].score
            implied = span * sum(g * float(score[column]) for column, g in weights.items())
            assert implied == pytest.approx(float(np.mean(estimate.influence_curve)), abs=1e-10)
            checked += 1
        assert checked == len(result.estimates)
        # Witness: the residuals are not at roundoff, so the identity is not vacuous.
        assert max(abs(float(np.mean(e.influence_curve))) for e in result.estimates.values()) > 1e-6

    @pytest.mark.parametrize("kind", ["msm-identity", "msm-logit-strata", "ipsi", "shift"])
    def test_the_rows_without_a_computed_bound_are_nan(self, kind: str) -> None:
        frame, columns, settings, _ = engines.KINDS[kind]()
        result = engines._fit("scaled_var", frame, columns=columns, **settings)
        rows = [r for r in score_check(result).rows if r.kind == "diagnostic"]
        assert rows and all(np.isnan(r.rule_bound_se) for r in rows)

    @pytest.mark.parametrize("name", ["ey1", "ey0", "ate", "att", "atc"])
    def test_the_residual_is_inside_the_bound_for_a_fit_that_met_the_rule(self, name: str) -> None:
        result = one_step("scaled_var")
        assert all(f.rule_ratio <= 1.0 for f in result.fluctuations.values())
        row = next(r for r in score_check(result).rows if r.name == name and r.kind == "diagnostic")
        assert row.residual_se <= row.rule_bound_se

    def test_a_rule_that_was_missed_can_exceed_the_bound_and_is_not_asserted(self) -> None:
        result = one_step("scaled_var")
        shifted = replace(
            result.estimates["ey1"],
            influence_curve=result.estimates["ey1"].influence_curve
            + 5.0 * result.estimates["ey1"].std_error,
        )
        broken = replace(result, estimates={**result.estimates, "ey1": shifted})
        row = next(
            r for r in score_check(broken).rows if r.name == "ey1" and r.kind == "diagnostic"
        )
        assert row.residual_se > row.rule_bound_se


# ------------------------------------------------------------------------------ the frame


class TestTheFrame:
    def test_a_declared_rule_adds_the_report_columns(self) -> None:
        frame = score_check(one_step("scaled_var")).to_frame()
        assert list(frame.columns[-4:]) == [
            "stopping",
            "residual_se",
            "coverage_bound",
            "rule_bound_se",
        ]
        fluctuation = frame[frame["kind"] == "fluctuation"]
        assert fluctuation["residual_se"].isna().all()
        assert (frame["stopping"] == "scaled_var").all()
        diagnostic = frame[frame["kind"] == "diagnostic"]
        assert diagnostic["residual_se"].notna().all()

    def test_the_shipped_rule_keeps_the_columns_it_had(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=1)
        result = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        columns = list(score_check(result).to_frame().columns)
        assert "stopping" not in columns and "residual_se" not in columns
        assert "rule_bound_se" not in columns and "coverage_bound" not in columns


# ------------------------------------------------------------------------------ the summary


class TestThePointSummary:
    @pytest.mark.parametrize("rule", RULES)
    def test_the_summary_ends_with_the_pinned_residual_line(self, rule: str) -> None:
        result = one_step(rule)
        residuals = {name: hand_residual(e) for name, e in result.estimates.items()}
        worst = max(residuals, key=lambda name: residuals[name])
        expected = (
            f"stopping rule {rule}: the largest score residual is {residuals[worst]:.3g} "
            f"standard errors, on {worst}. At that residual a 95% Wald interval keeps at least "
            f"{hand_coverage(residuals[worst], 0.05):.4f} coverage."
        )
        # The check keeps the pinned line; the summary states the same remainder in words.
        assert score_check(result).rule_line() == expected
        summary = result.summary()
        assert summary.note("stopping_rule").text == rule_note(result)
        assert f"(on {worst})" in summary.note("stopping_rule").text
        assert summary.note_codes.count("stopping_rule") == 1

    def test_the_check_summary_repeats_the_line_after_the_verdict(self) -> None:
        check = score_check(one_step("scaled_var"))
        assert check.summary().splitlines()[-1] == check.rule_line()
        assert "residual" in check.summary()

    def test_the_default_summary_has_no_rule_line(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=1)
        result = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        assert "stopping rule" not in result.summary()
        assert "Score-equation" not in result.summary()


# ------------------------------------------------------------------------------ every engine


class TestEveryPointGroupReports:
    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("kind", sorted(engines.KINDS))
    def test_the_check_and_the_summary_run(self, kind: str, rule: str) -> None:
        frame, columns, settings, _ = engines.KINDS[kind]()
        result = engines._fit(rule, frame, columns=columns, **settings)
        check = score_check(result)
        assert check.declared and check.stopping == rule
        # Every one of these fits met its rule, so the check passes (review finding F1).
        assert check.passed, [r.name for r in check.rows if not r.passed]
        for row in check.rows:
            if row.kind == "diagnostic":
                assert row.passed and np.isfinite(row.residual_se)
                assert 0.0 <= row.coverage_bound <= 1.0
            if row.kind == "fluctuation":
                # An outcome equation reads the rule, the nested strata solve included.  A
                # mechanism equation of the same fit keeps the default bar, which the rule does
                # not govern.
                expected = "relative" if row.name.endswith("(mechanism)") else rule
                assert row.stopping == expected, row.name
        assert any(r.stopping == rule for r in check.rows if r.kind == "fluctuation")
        line = check.rule_line()
        assert line.startswith(f"stopping rule {rule}: the largest score residual is ")
        assert result.summary().note("stopping_rule").text == rule_note(result)

    @pytest.mark.parametrize("rule", RULES)
    def test_a_doubly_robust_fit_judges_its_outcome_rows_by_the_rule(self, rule: str) -> None:
        estimator = DRTMLE(
            **linear_drtmle(
                stopping=rule,
                cross_fit=True,
                n_folds=3,
                q_bounds=(-30.0, 30.0),
                estimands=("ate", "ey1", "ey0"),
            )
        )
        result = estimator.fit(engines._lin(), outcome="Y", treatment="A").single()
        check = score_check(result)
        assert check.declared and check.passed
        assert {r.kind for r in check.rows} >= {"fluctuation", "diagnostic"}
        assert result.summary().note("stopping_rule").text == rule_note(result)
        rows = {r.name: r for r in check.rows}
        assert rows["mean (reduced)"].stopping == rule
        assert rows["mean (mechanism)"].stopping == "relative"
        # The bound is the outcome-column bound of an arm row, which a corrected curve changes.
        assert all(np.isnan(r.rule_bound_se) for r in check.rows if r.kind == "diagnostic")


# ------------------------------------------- rows whose equations the rule governs (F1)


def ruled_drtmle(order: str, rule: str) -> Any:
    """A complete-data DRTMLE whose one-step walk stops at ``rule``, both extra equations on."""
    return (
        engines._drtmle(order, ("Q", "g"), rule)
        .fit(engines._complete(), outcome="Y", treatment="A")
        .single()
    )


def ruled_missing(rule: str) -> Any:
    """The missing-outcome DRTMLE of the engine audits, under ``rule``."""
    return (
        engines._missing_drtmle(rule)
        .fit(engines._trial(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def ruled_strata(rule: str) -> Any:
    """A stratified logit MSM whose bounded walks stop at ``rule``."""
    frame, columns, settings, _ = engines.KINDS["msm-logit-strata"]()
    return engines._fit(rule, frame, columns=columns, step_bound=0.01, max_iter=300, **settings)


RULED_FITS: dict[str, Any] = {
    "drtmle": functools.cache(lambda rule: ruled_drtmle("drtmle", rule)),
    "benkeser": functools.cache(lambda rule: ruled_drtmle("benkeser", rule)),
    "missing": functools.cache(ruled_missing),
    "strata": functools.cache(ruled_strata),
}

#: The row each fit's default bar used to fail, and the source edit that restores that bar.
RULED_ROW = {
    "drtmle": ("mean (reduced)", 'if rule != "relative" and reduction.rule_ratio is not None:'),
    "benkeser": ("mean (reduced)", 'if rule != "relative" and reduction.rule_ratio is not None:'),
    "missing": ("mean (reduced)", 'if rule != "relative" and reduction.rule_ratio is not None:'),
    "strata": ("msm (strata)", "if _judged_by_rule(nested):"),
}


class TestRuleGovernedRows:
    """Rows whose equations a declared rule stopped are judged by the rule.

    The ``(reduced)`` row of a doubly-robust fit, its ``D*_Q`` and ``D*_Y`` correction rows,
    and the ``(strata)`` row of a stratified working model.  Each fit below met its rule, and
    the default bar fails each of those rows (the witnesses), so the check reads the rule.
    """

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("kind", list(RULED_FITS))
    def test_a_fit_that_met_its_rule_passes(self, kind: str, rule: str) -> None:
        result = RULED_FITS[kind](rule)
        check = score_check(result)
        assert check.passed, [r.name for r in check.rows if not r.passed]
        name, _ = RULED_ROW[kind]
        row = next(r for r in check.rows if r.name == name)
        assert row.stopping == rule and row.passed and row.ratio <= 1.0
        # Witness: the row's own score in the solver's units is above the default bar.
        default = score_threshold(
            (e.plugin_std_error for e in result.estimates.values()), result.data.n
        )
        fluctuation = result.fluctuations[name.split(" ")[0]]
        nested = fluctuation.stratified if kind == "strata" else fluctuation.reduction
        assert nested is not None
        assert float(np.max(np.abs(nested.score))) > default
        assert "worst score" in check.one_line() and "not solved" not in check.one_line()

    @pytest.mark.parametrize("kind", ["drtmle", "missing"])
    def test_the_outcome_corrections_are_diagnostics_and_the_mechanism_ones_are_not(
        self, kind: str
    ) -> None:
        check = score_check(RULED_FITS[kind]("scaled_var"))
        outcome = [
            r
            for r in check.rows
            if ("(D*_Q)" in r.name or "(D*_Y)" in r.name) and r.kind != "identity"
        ]
        mechanism = [
            r
            for r in check.rows
            if ("(D*_g)" in r.name or "(D*_A)" in r.name or "(D*_M)" in r.name)
            and "identity" not in r.name
        ]
        assert outcome and mechanism
        assert all(r.kind == "diagnostic" and r.passed for r in outcome)
        assert all(r.kind == "correction" for r in mechanism)
        # The identity rows of the outcome corrections are still checked.
        identities = [r for r in check.rows if r.kind == "identity"]
        assert any("(D*_Q)" in r.name or "(D*_Y)" in r.name for r in identities)
        assert all(r.passed and np.isfinite(r.threshold) for r in identities)

    @pytest.mark.parametrize("kind", list(RULED_FITS))
    def test_the_mutation_that_restores_the_default_bar_fails_the_fit(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = RULED_FITS[kind]("sample_size")
        name, guard = RULED_ROW[kind]
        mutate(monkeypatch, score_module, score_module, "score_check", [(guard, "if False:")])
        check = score_module.score_check(result)
        assert not check.passed
        assert name in [r.name for r in check.rows if not r.passed]

    @pytest.mark.parametrize("kind", ["drtmle", "missing"])
    def test_the_mutation_that_holds_the_corrections_to_the_default_bar_fails(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = RULED_FITS[kind]("sample_size")
        monkeypatch.setattr(score_module, "_ruled_reduction", lambda result: False)
        check = score_module.score_check(result)
        failed = [r.name for r in check.rows if not r.passed]
        assert failed and all("(D*_Q)" in n or "(D*_Y)" in n for n in failed)

    @pytest.mark.parametrize("kind", ["drtmle", "missing"])
    def test_a_reduced_ratio_above_one_fails_the_fit(self, kind: str) -> None:
        result = RULED_FITS[kind]("scaled_var")
        reduction = result.fluctuations["mean"].reduction
        assert reduction is not None and reduction.rule_ratio is not None
        missed = with_fluctuation(result, "mean", reduction=replace(reduction, rule_ratio=1.5))
        check = score_check(missed)
        row = next(r for r in check.rows if r.name == "mean (reduced)")
        assert not row.passed and not check.passed
        assert row in check.rule_failures

    def test_a_strata_ratio_above_one_fails_the_fit(self) -> None:
        result = RULED_FITS["strata"]("scaled_var")
        nested = result.fluctuations["msm"].stratified
        missed = with_fluctuation(
            result, "msm", stratified=replace(nested, rule_ratio=1.5, converged=False)
        )
        check = score_check(missed)
        row = next(r for r in check.rows if r.name == "msm (strata)")
        assert not row.passed and not check.passed

    def test_the_missing_outcome_row_reads_the_ratio_its_verdict_reads(self) -> None:
        # Review finding F3: the outcome row reads the stacked ratio of the standard and
        # drift blocks, the ratio the record's verdict reads.
        for rule in RULES:
            result = RULED_FITS["missing"](rule)
            fluctuation = result.fluctuations["mean"]
            reduction = fluctuation.reduction
            span = result.nuisance.scaler.range
            standard = float(np.max(np.abs(fluctuation.score) * span / fluctuation.thresholds))
            assert fluctuation.rule_ratio == pytest.approx(
                max(standard, reduction.rule_ratio), rel=1e-12
            )
            row = next(r for r in score_check(result).rows if r.name == "mean")
            assert row.ratio == pytest.approx(fluctuation.rule_ratio, rel=1e-9)
            assert row.passed == (fluctuation.rule_ratio <= 1.0)


# ------------------------------------------------------------------------------ longitudinal


class TestTheLongitudinalSummary:
    @pytest.mark.parametrize("rule", RULES)
    def test_a_fit_that_met_its_rule_prints_the_residual_line_only(self, rule: str) -> None:
        result = longitudinal.end_of_study(rule)
        assert result.converged
        residuals = {name: hand_residual(e) for name, e in result.estimates.items()}
        worst = max(residuals, key=lambda name: residuals[name])
        summary = result.summary()
        text = summary.note("stopping_rule").text
        assert text == rule_note(result)
        # The remainder of the note is the one the stored influence curve gives by hand.
        shown = "less than 0.001" if residuals[worst] < 0.001 else f"{residuals[worst]:.3g}"
        assert f"{shown} standard errors of bias (on {worst})" in text
        assert "score_unsolved" not in summary.note_codes

    def test_a_working_model_prints_the_line_on_its_regimen_parameters(self) -> None:
        result = longitudinal.working_model("scaled_var")
        assert result.converged
        residuals = {name: hand_residual(e) for name, e in result.estimates.items()}
        worst = max(residuals, key=lambda name: residuals[name])
        assert worst.startswith("msm_regimen[")
        text = result.summary().note("stopping_rule").text
        assert text == rule_note(result)
        shown = "less than 0.001" if residuals[worst] < 0.001 else f"{residuals[worst]:.3g}"
        assert f"{shown} standard errors of bias (on {worst})" in text

    def test_a_fit_that_missed_its_rule_prints_the_failure_line(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                longitudinal.PLANS,
                stopping="scaled_var",
                alpha=0.55,
                **longitudinal.learners(),
            ).fit(frame, outcome="Y", **longitudinal.NODES)
        assert not result.converged
        misses = [
            (key, step)
            for key, step in longitudinal.every_node(result)
            if step.fluctuation.rule_ratio > 1.0
        ]
        assert misses
        key, step = misses[0]
        ratio = step.fluctuation.rule_ratio
        named = f"{key} at time {step.time} stopped at {ratio:.3g} times its threshold"
        summary = result.summary()
        item = summary.note("score_unsolved")
        assert item.kind == "warning"
        assert item.text.startswith("The fit did not meet its declared stopping rule: ")
        assert named in item.text
        assert summary.note("stopping_rule").text == rule_note(result)

    def test_the_shipped_rule_prints_the_shipped_warning(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(longitudinal.PLANS, alpha=0.55, **longitudinal.learners()).fit(
                frame, outcome="Y", **longitudinal.NODES
            )
        summary = result.summary()
        assert "stopping_rule" not in summary.note_codes
        if not result.converged:
            assert summary.note("score_unsolved").text == vocabulary.NOTE_TEXTS[
                "score_unsolved_longitudinal"
            ].format(draw="", what="interval")

    def test_the_default_summary_has_no_rule_line(self) -> None:
        frame, _ = make_longitudinal(n=300, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(longitudinal.PLANS, **longitudinal.learners()).fit(
                frame, outcome="Y", **longitudinal.NODES
            )
        assert "stopping rule" not in result.summary()
        assert "did not meet" not in result.summary()
