"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from cleverly import CapabilityError
from cleverly.datasets import navigation_protocol
from cleverly.validation import BoundedAdjustmentOutcome, GaussianAdjustmentOutcome, refute
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    ROOT,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "interventions.ipynb"

#: The propensity bound the regime fit prints, which Step 6 compares with each plan's ratios.
TRUNCATION = 0.0114

_PROBE = "reviews/notebook-review/probes/interventions-final"
_BINARY = f"{_PROBE}/summary.log, sweep_binary.csv (seeds 9000 to 9119)"
_DOSE = f"{_PROBE}/summary.log, sweep_dose.csv (seeds 9000 to 9119)"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    # The mean density ratio of the shown dose configuration, quad_rboost40.
    "1.05": f"largest mean of the mean density ratio, quad_rboost40 (+1.0, 1.048): {_DOSE}",
    # The density comparison, same quadratic Q.
    "1.01": f"largest mean density ratio, quad_oracle320 (+1.0 uncapped): {_DOSE}",
    "1.00": f"smallest mean density ratio, quad_oracle320 (+0.5 capped, 1.002): {_DOSE}",
    # The cross-fitted shift probe that the technical reference cites.
    "3.91": (
        "cross-fitted +1.0 mean SE over empirical SD, default booster density: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    "0.98": (
        "cross-fitted +1.0 mean SE over empirical SD, exact density at 320 bins: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    "+0.00011": f"incremental mean error, degree-2 logistic g: {_BINARY}",
    "+0.00066": f"incremental mean error, shallow booster: {_BINARY}",
    "0.00008": f"Monte Carlo SE of each incremental mean error: {_BINARY}",
    "0.93": f"median propensity calibration slope, degree-2 logistic g (0.929): {_BINARY}",
    "0.94": f"median propensity calibration slope, shallow booster (0.939): {_BINARY}",
    "3.3": "a section number of Kennedy (2019), docs/references.md#incremental-interventions",
    "4.5": "the capacity of 5 minus the raise of 0.5, the threshold the cap acts on",
    "6.1": "the size of the screen's printed movement of -6.1 standard errors (sensitivity)",
    "6.4": "the size of the nudge's printed movement of -6.4 standard errors (sensitivity-other)",
    "2.664e-06": (
        "the fitted model's smallest offer probability, which the regime support "
        "report holds (min_support_propensity of offer to all); check() asserts it"
    ),
}

#: Where each probe or study decimal of ``UNPRINTED_DECIMALS`` lives: the committed file, a
#: pattern whose first group is the number, and the decimals the prose rounds it to.
_SUMMARY = f"{_PROBE}/summary.log"
_ROW = r"\| {row} \| (\d+)/120 = [\d.]+ \| ([+-][\d.]+) \(([\d.]+)\) \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|"
_SECTIONS = {
    "poly2": r"### treatment learner: poly2",
    "booster": r"### treatment learner: booster",
    "quad_oracle320": r"### configuration: quad_oracle320",
    "quad_rboost40": r"### configuration: quad_rboost40",
}
#: (prose decimal, section, row, column) where column 1 is the coverage count, 2 the mean
#: error, and 3 its Monte Carlo SE.
_PROBE_CELLS = (
    ("+0.00011", "poly2", "ipsi", 2),
    ("0.00008", "poly2", "ipsi", 3),
    ("+0.00066", "booster", "ipsi", 2),
    ("0.00008", "booster", "ipsi", 3),
)
#: Coverage counts the readings quote, by (section, row).
_PROBE_COVERAGE = {
    ("poly2", "all"): 117,
    ("poly2", "screen"): 117,
    ("poly2", "ipsi"): 116,
    ("booster", "ipsi"): 110,
    ("quad_rboost40", "capped"): 117,
    ("quad_rboost40", "uncapped"): 117,
    ("quad_rboost40", "one"): 114,
}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _probe_row(section: str, row: str) -> tuple[str, ...]:
    text = _read(_SUMMARY)
    start = text.index(_SECTIONS[section])
    match = re.search(_ROW.format(row=row), text[start:])
    assert match is not None, (section, row)
    return match.groups()


def _rounds_to(value: float, prose: str) -> bool:
    places = len(prose.split(".")[1])
    return f"{value:.{places}f}" == prose.lstrip("+")


def check_sources() -> None:
    """Each unprinted probe decimal equals its committed source, rounded as written."""
    for prose, section, row, column in _PROBE_CELLS:
        assert prose in UNPRINTED_DECIMALS
        assert _rounds_to(float(_probe_row(section, row)[column - 1]), prose), (prose, section)
    for (section, row), count in _PROBE_COVERAGE.items():
        assert int(_probe_row(section, row)[0]) == count, (section, row)
    # The median calibration slopes of the two treatment models.
    summary = _read(_SUMMARY)
    slopes = re.findall(r"calibration slope 5%, 50%, 95%: [\d.]+, ([\d.]+), ", summary)
    assert [round(float(value), 2) for value in slopes] == [0.94, 0.93]  # booster, poly2
    # Density-ratio means: 1.03 to 1.05 with this density, 1.00 to 1.01 with the exact one.
    for section, low, high in (("quad_rboost40", 1.03, 1.05), ("quad_oracle320", 1.00, 1.01)):
        part = summary[summary.index(_SECTIONS[section]) :]
        means = re.search(r"capped ([\d.]+) \(.*?uncapped ([\d.]+) \(.*?one ([\d.]+) \(", part)
        assert means is not None
        values = [round(float(value), 2) for value in means.groups()]
        assert (min(values), max(values)) == (low, high), section
    truth = _read(f"{_PROBE}/truth.log")
    assert re.search(r"ipsi x2 vs x1\s+0\.024137 ", truth)  # the callback's truth check
    # The cross-fitted shift probe of the technical reference: 3.91 and 0.98.
    reference = _read("docs/technical-reference/point-treatment-tmle.md")
    assert re.search(r"default settings \| 40 \| cross-fitted \| 3\.91 \|", reference)
    assert re.search(r"exact law \| 320 \| cross-fitted \| 0\.98 \|", reference)
    # The Kennedy locators, and the threshold the cap acts on.
    kennedy = _read("docs/references.md")
    assert "| Section 3.3, Theorem 2," in kennedy
    assert 5.0 - 0.5 == 4.5


SCREEN = "ate_regime[screen on risk vs offer to none]"
OFFER_ALL = "ate_regime[offer to all vs offer to none]"
CAPPED = "ate_policy[+0.5 capped at 5 vs current practice]"
UNCAPPED = "ate_policy[+0.5 uncapped vs current practice]"
ONE = "ate_policy[+1.0 uncapped vs current practice]"
IPSI = "ate_ipsi[double odds vs current odds]"


def check(namespace: dict[str, Any]) -> None:
    """The three-proposal tutorial's truths, support figures, and draw-specific claims hold."""
    check_sources()
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    # The incremental truth carries g, so it moved when the law's g was bounded. truth.py
    # recomputes it from the structural equations as 0.024137 (MC SE 2.4e-06).
    double_odds = namespace["incremental_truth"]["ate_ipsi[odds x2 vs natural course]"]
    assert double_odds == pytest.approx(0.024137, abs=2e-5)
    assert "positivity *for the assigned dose*" in namespace["shift_effect"].summary()
    assert "An upper cap alone does not ensure this for negative shifts or support gaps" in (
        namespace["shift_effect"].summary()
    )
    assert "*no positivity assumption*" in namespace["incremental_effect"].summary()

    # Each reading names the fields its protocol changes in the program protocol.
    program = navigation_protocol()
    assert changed_fields(namespace["regime_protocol"], program) == {
        "treatment_strategies",
        "treatment_versions",
        "assumption_rationale",
    }
    assert changed_fields(namespace["dose_protocol"], program) == {
        "outcome",
        "time_zero",
        "treatment_strategies",
        "treatment_versions",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert changed_fields(namespace["incremental_protocol"], program) == {
        "treatment_strategies",
        "treatment_versions",
        "assumption_rationale",
    }
    # The intensity protocol names an index and a score with no fixed range (IV-04, IV-05).
    dose_protocol = namespace["dose_protocol"]
    assert "standardized" not in dose_protocol.outcome
    assert "no fixed range" in dose_protocol.outcome
    assert not any("contact" in text for text in dose_protocol.treatment_versions)
    assert any("index" in text for text in dose_protocol.assumption_rationale)
    # "Offer to none reuses the program's version for usual support."
    assert namespace["regime_protocol"].treatment_versions[0] == program.treatment_versions[1]
    # Each axis has its own protocol, printed in its step and stamped on its result.
    for cell, protocol, result in (
        ("protocol", "regime_protocol", "regime_result"),
        ("dose-identify", "dose_protocol", "shift_result"),
        ("incremental-identify", "incremental_protocol", "incremental_result"),
    ):
        assert_protocol_recorded(NOTEBOOK, cell, namespace[protocol], namespace[result])
    assert (
        len(
            {
                namespace[name].fingerprint
                for name in ("regime_protocol", "dose_protocol", "incremental_protocol")
            }
        )
        == 3
    )

    # "Current practice already offers navigation more often to flagged patients."
    current = namespace["current"]
    assert (
        current.loc["flagged", "share offered now"]
        > current.loc["not flagged", "share offered now"]
    )

    # "Each interval contains its population value on this draw", the screen by a narrow
    # margin: its lower end lies within 0.001 below the population value.
    regimes = namespace["regime_result"]
    assert covers(regimes[OFFER_ALL], offer_all)
    assert covers(regimes[SCREEN], screen)
    assert 0.0 < screen - regimes[SCREEN].ci[0] < 0.001
    # "offer to all minus screen: 0.056 (0.048, 0.065) ... population value 0.057": the gap
    # comes from the joint influence curve, so its interval is narrower than either contrast's.
    given_up = namespace["given_up"]
    assert given_up.psi == pytest.approx(regimes[OFFER_ALL].psi - regimes[SCREEN].psi)
    assert covers(given_up, offer_all - screen)
    assert given_up.std_error < min(regimes[OFFER_ALL].std_error, regimes[SCREEN].std_error)

    # "`needs attention` is empty" on the regime fit.
    assert not namespace["regime_assessment"].attention
    rows = namespace["regime_support"].regimes
    # Offer to all has the smallest ratio effective n and the largest ratio.
    assert rows["offer to all"].effective_sample_size == min(
        row.effective_sample_size for row in rows.values()
    )
    assert rows["offer to all"].max_ratio == max(row.max_ratio for row in rows.values())
    # "The largest ratio of each plan is below that cap ... Truncation therefore changes no
    # weight on this draw, and each score load agrees with its ratio effective n."
    assert f"propensity truncated to [{TRUNCATION}, " in stored_output(NOTEBOOK, "regime-fit")
    for row in rows.values():
        assert row.max_ratio < 1 / TRUNCATION
        assert row.score_load["effective"] == pytest.approx(row.effective_sample_size, rel=1e-3)
    # Nonzero witness: the fitted model does put a probability below the bound and below the
    # law's floor of 0.05, so the bound would bind if that row carried weight. The row is
    # off-plan: every on-plan ratio is at most max_ratio, so every on-plan g is at least
    # 1 / max_ratio, which is far above the smallest fitted g.
    smallest = rows["offer to all"].min_support_propensity
    assert f"{smallest:.3e}" == "2.664e-06"  # the Details of the overlap reading
    assert smallest < TRUNCATION < 0.05
    assert smallest < 0.01 / rows["offer to all"].max_ratio
    # "The screen asks for offers where current practice already offers more often, and its
    # effective sample size is the largest."
    assert rows["screen on risk"].effective_sample_size == max(
        row.effective_sample_size for row in rows.values()
    )

    # Two warnings, for the uncapped policies; the cap of 5 lies below the largest dose, so
    # the capped policy moves no row past it and raises none.
    warned = namespace["positivity_warnings"]
    assert len(warned) == 2
    assert "'+0.5 uncapped'" in warned[0] and "'+1.0 uncapped'" in warned[1]
    assert not any("capped at 5" in message for message in warned)
    dose = np.asarray(namespace["dose_frame"]["assigned_navigation_intensity"], dtype=float)
    assert dose.max() > 5.0
    support = namespace["shift_assessment"].report("support")
    assert support["current practice"].ess_ratio == pytest.approx(1.0)
    capped = support["+0.5 capped at 5"]
    assert 0.015 < capped.capped_fraction < 0.035
    assert 0.60 < capped.ess_ratio < 0.75
    # The printed widths: the +1.0 interval is wider than either +0.5 interval.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    assert width[ONE] > max(width[CAPPED], width[UNCAPPED])
    # "On this draw each interval contains its population value."
    dose_truth = namespace["dose_truth"]
    shift_result = namespace["shift_result"]
    for name in (CAPPED, UNCAPPED, ONE):
        assert covers(shift_result[name], dose_truth[name])
    # The shown fit is the quadratic Q and the regularized density booster of Step 9.
    models = namespace["shift_method"].models
    assert models.density_bins == 40
    assert models.treatment_learner.get_params()["max_depth"] == 2
    assert models.treatment_learner.get_params()["l2_regularization"] == 1.0
    assert models.outcome_learner.get_params()["polynomialfeatures__degree"] == 2
    # On this draw the estimated density keeps less than the true density ratio, for both
    # shifts.
    assert 0.15 < support["+1.0 uncapped"].ess_ratio < np.exp(-1.0)
    assert support["+0.5 uncapped"].ess_ratio < np.exp(-0.25)
    # The histogram reports zeros at assigned doses outside its fitted bin range.
    # These counts describe this seeded output, not population positivity.
    assert {name: row.unsupported for name, row in support.items()} == {
        "current practice": 0,
        "+0.5 capped at 5": 0,
        "+0.5 uncapped": 2,
        "+1.0 uncapped": 3,
    }
    assert tuple(item.name for item in namespace["shift_assessment"].attention) == ("support",)
    # The mean ratio: reference 1 for every shift of a normal intensity. "The estimated ratio
    # runs 3% to 8% high on average, and most for +1.0." Current practice reads exactly 1
    # (control); each moving shift reads above it (nonzero witness). In sample the one
    # per-fold value equals the overall mean.
    assert support["current practice"].mean_ratio == pytest.approx(1.0)
    means = {name: support[name].mean_ratio for name in ("+0.5 capped at 5", "+0.5 uncapped")}
    means["+1.0 uncapped"] = support["+1.0 uncapped"].mean_ratio
    assert all(1.02 < value < 1.09 for value in means.values())
    assert means["+1.0 uncapped"] == max(means.values())
    for name in support:
        assert support[name].fold_mean_ratio == (support[name].mean_ratio,)

    # The cap gap, read with its interval through the joint influence curve.
    population_gap = dose_truth[UNCAPPED] - dose_truth[CAPPED]
    assert 0.02 < population_gap < 0.06
    gap = namespace["gap"]
    assert gap.psi == pytest.approx(shift_result[UNCAPPED].psi - shift_result[CAPPED].psi)
    assert gap.psi > 0.0
    # "The population gap lies within 1.96 standard errors of the estimate"; the interval
    # also excludes zero, so the gap it covers is a nonzero one.
    assert abs(gap.psi - population_gap) < 1.96 * gap.std_error
    assert covers(gap, population_gap)
    assert gap.ci[0] > 0.0
    # "below the standard error of each estimate": the correlation is load-bearing.
    assert gap.std_error < min(shift_result[CAPPED].std_error, shift_result[UNCAPPED].std_error)

    # The incremental interval contains its population value on this draw, 0.9 SE away.
    tilt = namespace["incremental_result"][IPSI]
    assert covers(tilt, double_odds)
    assert 0.85 <= (tilt.psi - double_odds) / tilt.std_error < 0.95
    incremental = namespace["incremental_assessment"]
    assert not incremental.attention
    stored = stored_output(NOTEBOOK, "incremental-fit")
    assert "checks that need attention: none" in stored
    # "The slope, 0.94 (standard error 0.05), is close to 1".
    assert "treatment model calibration slope: 0.94 (standard error 0.05)" in stored
    # "The check marks both rows as solved, with each score below 0.01 of its threshold."
    score_rows = incremental.report("score_equations").rows
    assert {row.name for row in score_rows} >= {"ipsi", "ipsi (mechanism)"}
    assert all(row.passed and row.ratio < 0.01 for row in score_rows)
    # "the clever covariate stays between one half and two for this multiplier".
    tilts = incremental.report("support")
    assert tilts["double odds"].guaranteed == pytest.approx((0.5, 2.0))

    # The grid has no outcome axis because the fit declares q_bounds: the outcome strength of
    # such a fit is a coefficient on the logit scale of the score.
    assert namespace["regime_result"].estimator.q_bounds == (0.0, 1.0)
    _check_refutation_limits(namespace)
    _check_generated_limits(namespace)
    _check_bound_and_surfaces(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """Steps 14 to 16: the refutation, the generated outcomes, and the stress surfaces."""
    # The law that ``check`` rebuilds for its refusal is the notebook's own.
    assert namespace["bounded"] == BoundedAdjustmentOutcome(effect=0.5)
    # The treatment-only stress surface: both cells return, and the flip moves the screen
    # estimate down by 0.029037 (-6.1 standard errors) with a small induced association.
    surface = namespace["surface"]
    assert "bounded outcome law: logit on [0, 1]" in surface.summary()
    cells = sorted(surface.cells, key=lambda cell: cell.treatment_strength)
    assert [cell.treatment_strength for cell in cells] == [0.0, 0.1]
    assert all(cell.outcome_strength == 0.0 and cell.failure is None for cell in cells)
    assert cells[1].estimate < cells[0].estimate
    assert cells[1].displacement == pytest.approx(-0.0290, abs=0.001)
    for cell in cells:
        assert cell.induced_treatment_association is not None
        assert abs(cell.induced_treatment_association) < 0.1
    _check_refutation(namespace)
    _check_generated(namespace)
    _check_benchmarks(namespace)
    # The nudge and raise surfaces: the flip moves each estimate down.
    for key in ("nudge_surface", "raise_surface"):
        flipped = max(namespace[key].cells, key=lambda cell: cell.treatment_strength)
        zero = min(namespace[key].cells, key=lambda cell: cell.treatment_strength)
        assert flipped.estimate < zero.estimate
        assert flipped.outcome_strength == 0.0 and flipped.failure is None


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 14: the placebo and subset checks of the screen, and the stated limit."""
    regimes = namespace["regime_result"]
    screen_report = namespace["screen_refutation"]
    assert screen_report.estimand == SCREEN
    tests = {test.name: test for test in screen_report.tests}
    assert list(tests) == ["placebo", "subset"]
    original = regimes[SCREEN].psi
    for test in tests.values():
        # "No check prints a verdict": no registry row covers a TMLE regime contrast.
        assert test.passed is None and test.rung == "L0" and len(test.values) == 20
    assert screen_report.passed is None
    assert (
        "This fit differs on estimand kind 'ate_regime[screen on risk vs offer to none]'"
        in screen_report.summary()
    )
    placebo = tests["placebo"]
    values = np.asarray(placebo.values)
    assert round(float(values.mean()), 7) == 0.0009427
    assert round(float(placebo.statistic), 4) == 0.7714
    assert round(float(placebo.statistic_pvalue), 2) == 0.45
    assert round(float(np.mean(placebo.standard_errors) / values.std(ddof=1)), 3) == 0.884
    # A nonzero witness: a placebo that kept the observed offers would return the estimate.
    assert abs(values.mean()) < 0.1 * original
    subset = np.asarray(tests["subset"].values)
    reference = float(regimes[SCREEN].std_error * np.sqrt((3000 / 2100 - 1) * 3000 / 2999))
    assert round(float(np.std(subset, ddof=1)), 7) == 0.0026376
    assert round(reference, 7) == 0.0031383


def _check_refutation_limits(namespace: dict[str, Any]) -> None:
    """Step 14: the stored output, and the negative control that the page says it cannot run."""
    regimes = namespace["regime_result"]
    stored = stored_output(NOTEBOOK, "refute-regimes")
    assert stored.count("descriptive") == 2
    # The limit: `prior_utilization` is an adjustment column, so a control that equals it is
    # refused, and the frame holds no other pre-offer measure.
    frame = namespace["frame"]
    with pytest.raises(CapabilityError, match="equals the adjustment column"):
        refute(
            regimes,
            estimand=SCREEN,
            tests=("negative_control_outcome",),
            negative_control_outcome=frame["prior_utilization"].to_numpy(),
            negative_control_kind="pre_treatment",
            n_replicates=5,
        )


def _check_generated(namespace: dict[str, Any]) -> None:
    """Step 15: the generated outcomes, each at 10 draws, and the limits that the page states."""
    expected = (
        ("screen_generated", "simulated_outcome", SCREEN, 0.049828, 0.051083),
        ("nudge_plasmode", "plasmode", "ate_ipsi[double odds vs current odds]", 0.016329, 0.016364),
        ("raise_generated", "simulated_outcome", CAPPED, 0.24367, 0.24717),
    )
    for name, test_name, estimand, truth, mean in expected:
        report = namespace[name]
        test = report[test_name]
        assert report.estimand == estimand
        # "No reading prints a verdict ... 10 draws are below the calibrated minimum of 400."
        assert test.passed is None and test.rung == "L1" and len(test.values) == 10
        assert round(float(test.declared_effect), 5 if truth > 0.1 else 6) == truth
        assert round(float(np.mean(test.values)), 5 if truth > 0.1 else 6) == mean
        # "Each mean of the refits lies close to its truth": a nonzero witness against a mean
        # that sat at the real-data estimate, which differs from every generated truth.
        assert abs(np.mean(test.values) - truth) < 0.05 * abs(truth) + 0.01
        assert (
            abs(
                truth
                - namespace[
                    {
                        "screen_generated": "regime_result",
                        "nudge_plasmode": "incremental_result",
                        "raise_generated": "shift_result",
                    }[name]
                ][estimand].psi
            )
            > 0.005
        )
    # The bias reading is in standard deviations of the draws: (mean - truth) / sd.
    screen = namespace["screen_generated"]["simulated_outcome"]
    assert (
        round((np.mean(screen.values) - screen.declared_effect) / np.std(screen.values, ddof=1), 3)
        == 0.472
    )


def _check_generated_limits(namespace: dict[str, Any]) -> None:
    """Step 15: the stored output, and the refusals that the step table states."""
    assert stored_output(NOTEBOOK, "generated").count("10 refits, verdict descriptive") == 3
    # The limits of the step table.
    regimes, incremental = namespace["regime_result"], namespace["incremental_result"]
    shift = namespace["shift_result"]
    with pytest.raises(CapabilityError, match="registered Gaussian process"):
        refute(
            regimes,
            estimand=SCREEN,
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
            n_replicates=10,
        )
    with pytest.raises(CapabilityError, match="no truth for an incremental tilt"):
        refute(
            incremental,
            estimand="ate_ipsi[double odds vs current odds]",
            tests=("simulated_outcome",),
            simulated_outcome=BoundedAdjustmentOutcome(effect=0.5),
            n_replicates=10,
        )
    with pytest.raises(CapabilityError, match="no law of a continuous treatment"):
        refute(
            shift,
            estimand=CAPPED,
            tests=("plasmode",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
            n_replicates=10,
        )


def _check_benchmarks(namespace: dict[str, Any]) -> None:
    """The bound: robustness values, benchmarks, and the tipping strength the reading compares."""
    regimes = namespace["regime_result"]
    for name, rv, rva in ((SCREEN, 0.3938, 0.3647), (OFFER_ALL, 0.4050, 0.3777)):
        robustness = regimes.sensitivity.robustness_value(estimand=name)
        assert round(robustness["rv"], 4) == rv and round(robustness["rva"], 4) == rva
        assert robustness["limit_licensed"] is False
        assert "No calibration covers this configuration" in robustness["limit_sentence"]
    benchmarks = namespace["benchmarks"]
    assert list(benchmarks) == ["prior_utilization", "medication_burden", "age"]
    strongest, bound = benchmarks["prior_utilization"]
    # "a confounder like it would leave the lower limit at 0.0439": a nonzero witness, the
    # bound moves the estimate, but not to zero.
    assert round(bound.ci_lower, 4) == 0.0439
    assert 0.0 < bound.bias < regimes[SCREEN].psi
    # prior_utilization is the strongest on both sides.
    for record, _ in benchmarks.values():
        assert record.cf_y <= strongest.cf_y and record.cf_d <= strongest.cf_d
    # "an offer-side strength of only about 0.17 to pull the lower limit to zero, below
    # prior_utilization's 0.206".
    tipping = namespace["tipping_cf_d"]
    assert round(tipping, 2) == 0.17 and tipping < strongest.cf_d
    assert (
        regimes.sensitivity.omitted_confounding(
            estimand=SCREEN, cf_y=1.0, cf_d=tipping - 0.01
        ).ci_lower
        > 0
    )
    assert (
        regimes.sensitivity.omitted_confounding(
            estimand=SCREEN, cf_y=1.0, cf_d=tipping + 0.01
        ).ci_lower
        < 0
    )
    # The rule reads discharge_risk, so a benchmark cannot drop it.
    with pytest.raises(CapabilityError):
        regimes.sensitivity.benchmark(covariates=("discharge_risk",), estimand=SCREEN)


def _check_bound_and_surfaces(namespace: dict[str, Any]) -> None:
    """Step 16: the regime bound, its refusals, and the stress surfaces of the other two."""
    regimes = namespace["regime_result"]
    text = stored_output(NOTEBOOK, "bound")
    assert "robustness value, screen: 0.394 (0.365 at the 95% limit)" in text
    assert "robustness value, offer to all: 0.405 (0.378 at the 95% limit)" in text
    assert "every shift above assumes rho = 1" in text
    assert "pulls the screen's lower limit to zero at cf_y = 1: 0.17" in text
    for call, reason in (
        (
            lambda: namespace["incremental_result"].sensitivity.omitted_confounding(
                estimand="ate_ipsi[double odds vs current odds]"
            ),
            "tilts the treatment mechanism",
        ),
        (
            lambda: namespace["shift_result"].sensitivity.omitted_confounding(estimand=CAPPED),
            "reads a fitted conditional density",
        ),
    ):
        with pytest.raises(CapabilityError, match=reason):
            call()
    for call in (
        lambda: regimes.sensitivity.evalue(estimand=SCREEN),
        lambda: namespace["incremental_result"].sensitivity.evalue(
            estimand="ate_ipsi[double odds vs current odds]"
        ),
        lambda: namespace["shift_result"].sensitivity.evalue(estimand=CAPPED),
    ):
        with pytest.raises(Exception, match=r"E-value (needs an|requires a discrete) arm"):
            call()
    # The nudge and raise surfaces: the flip moves each estimate down, by the stored amounts.
    stored = stored_output(NOTEBOOK, "sensitivity-other")
    assert "nudge: movement -6.4 standard errors" in stored
    assert "raise: movement -0.9 standard errors" in stored
