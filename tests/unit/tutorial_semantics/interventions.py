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
from cleverly._summary import vocabulary
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

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = (
    "bound",
    "refute-regimes",
    "sensitivity",
    "sensitivity-other",
)


NOTEBOOK = EXAMPLES / "interventions.ipynb"

#: The propensity bound the regime fit prints, which Step 6 compares with each plan's ratios.
TRUNCATION = 0.0114

_PROBE = "reviews/notebook-review/probes/interventions-final"
_BINARY = f"{_PROBE}/summary.log, sweep_binary.csv (seeds 9000 to 9119)"
_DOSE = f"{_PROBE}/summary.log, sweep_dose.csv (seeds 9000 to 9119)"
_LEARNED = "tests/canonical/point_shift_learned_density/summary.csv (ate_policy rows)"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    # The mean density ratio of the shown dose configuration, quad_rboost40.
    "0.97": f"smallest mean of the mean density ratio, quad_rboost40 (+1.0, 0.966): {_DOSE}",
    "1.00": f"largest mean of the mean density ratio, quad_rboost40 (capped, 1.003): {_DOSE}",
    # The cross-fitted shift probe that the technical reference cites.
    "3.91": (
        "cross-fitted +1.0 mean SE over empirical SD, default booster density: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    "0.98": (
        "cross-fitted +1.0 mean SE over empirical SD, exact density at 320 bins: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    # Coverage of the three effects in the learned-density study (1,200 draws per row).
    "93.3": f"in_sample_learned coverage, +0.5 capped at 5 (0.9325): {_LEARNED}",
    "92.7": f"in_sample_learned coverage, +0.5 uncapped (0.9267): {_LEARNED}",
    "89.7": f"in_sample_learned coverage, +1.0 uncapped (0.8967): {_LEARNED}",
    "94.4": f"smallest in_sample_oracle effect coverage, +1.0 uncapped (0.9442): {_LEARNED}",
    "95.6": f"largest in_sample_oracle effect coverage, +0.5 capped at 5 (0.9558): {_LEARNED}",
    "95.4": f"smallest crossfit_learned effect coverage, +0.5 uncapped (0.9542): {_LEARNED}",
    "96.3": f"largest crossfit_learned effect coverage, +0.5 capped at 5 (0.9625): {_LEARNED}",
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


#: Decimals a reading derives from a printed number: by "cell: literal", the code cell, the
#: printed text, and the derivation (tests.unit.tutorial_semantics.derived_mismatches).
DERIVED_DECIMALS = {
    "sensitivity-other-reading: 1.1": ("sensitivity-other", "-1.1", "magnitude"),
    "sensitivity-reading: 0.2": ("sensitivity", "-0.2", "magnitude"),
    "sensitivity-other-reading: 7.8": ("sensitivity-other", "-7.8", "magnitude"),
    "sensitivity-other-reading: 4.1": ("sensitivity-other", "-4.1", "magnitude"),
}


#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "incremental-identify-reading": {"0.5": "incremental-fit"},
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
    ("quad_rboost40", "capped"): 109,
    ("quad_rboost40", "uncapped"): 110,
    ("quad_rboost40", "one"): 103,
    ("quad_oracle320", "capped"): 112,
    ("quad_oracle320", "uncapped"): 115,
    ("quad_oracle320", "one"): 112,
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
    # Density-ratio means of the shown configuration: "averaged 0.97 to 1.00".
    part = summary[summary.index(_SECTIONS["quad_rboost40"]) :]
    means = re.search(r"capped ([\d.]+) \(.*?uncapped ([\d.]+) \(.*?one ([\d.]+) \(", part)
    assert means is not None
    values = [round(float(value), 2) for value in means.groups()]
    assert (min(values), max(values)) == (0.97, 1.00)
    # "The reported standard errors were 15% to 24% smaller than the spread of the estimates",
    # "with little bias": each mean error within 2 Monte Carlo SEs of zero.
    shortfall = []
    for row in ("capped", "uncapped", "one"):
        _, error, error_se, ratio = _probe_row("quad_rboost40", row)
        shortfall.append(round(100 * (1 - float(ratio))))
        assert abs(float(error)) < 2 * float(error_se), row
    assert (min(shortfall), max(shortfall)) == (15, 24)
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
    # A full summary wraps its lines, so each check reads the words in order.
    shift = " ".join(str(namespace["shift_effect"].summary(detail="full")).split())
    assert "positivity *for the assigned dose*" in shift
    assert "An upper cap alone does not ensure this for negative shifts or support gaps" in shift
    incremental = " ".join(str(namespace["incremental_effect"].summary(detail="full")).split())
    assert "*no positivity assumption*" in incremental

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
    # The score has no worst value, so death cannot be scored as the worst score.
    assert "hypothetical strategy" in dose_protocol.intercurrent_event_handling[2]
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
    assert re.search(
        rf"Probability limits:\s+\[{TRUNCATION}, ", stored_output(NOTEBOOK, "regime-fit")
    )
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
    # The model checks, the overlap figure, and the truncation curve of the Details.
    stored = stored_output(NOTEBOOK, "regime-support")
    assert "treatment model calibration slope: 0.935 (standard error 0.054)" in stored
    assert "outcome model regression slope: 0.954" in stored
    assert "not offered  below 0.05: 2.6%" in stored and "offered      below 0.05: 0.2%" in stored
    assert namespace["overlap_fig"].axes
    assert "truncation curve, screen: largest move 0.18 standard errors" in stored
    assert "truncation curve, offer to all: largest move 0.29 standard errors" in stored
    assert "limits tried: 0.001 to 0.2; rows capped at 0.2: 10.9%" in stored
    # A nonzero witness: the curve's widest limit does move the estimate.
    curve = namespace["curve"]
    assert curve["delta_from_fitted"].abs().max() > 0.0
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
    assert 0.70 < capped.ess_ratio < 0.85
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
    # On this draw the estimated density keeps more than the true density ratio, for both
    # shifts: "43.0% (36.8% with the true density)".
    assert np.exp(-1.0) < support["+1.0 uncapped"].ess_ratio < 0.55
    assert support["+0.5 uncapped"].ess_ratio > np.exp(-0.25)
    # The density reports zeros at assigned doses outside its fitted bin range.
    # These counts describe this seeded output, not population positivity.
    assert {name: row.unsupported for name, row in support.items()} == {
        "current practice": 0,
        "+0.5 capped at 5": 0,
        "+0.5 uncapped": 2,
        "+1.0 uncapped": 3,
    }
    assert tuple(item.name for item in namespace["shift_assessment"].attention) == ("support",)
    # The mean ratio: reference 1 for every shift of a normal intensity. "With a correct density
    # the weights average 1; here they run a little low", most for +1.0. Current practice reads
    # exactly 1 (control); each moving shift reads below it (nonzero witness). In sample the one
    # per-fold value equals the overall mean.
    assert support["current practice"].mean_ratio == pytest.approx(1.0)
    means = {name: support[name].mean_ratio for name in ("+0.5 capped at 5", "+0.5 uncapped")}
    means["+1.0 uncapped"] = support["+1.0 uncapped"].mean_ratio
    assert all(0.93 < value < 0.998 for value in means.values())
    assert means["+1.0 uncapped"] == min(means.values())
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

    # The support labels the readings name: all three experimental. The raise learns its density,
    # and the rows that cover a shift (T21, T22) give the method the exact density.
    assert set(namespace["incremental_result"].to_frame()["support"]) == {"experimental"}
    assert set(namespace["shift_result"].to_frame()["support"]) == {"experimental"}
    (reason,) = set(namespace["shift_result"].to_frame()["support_reason"])
    assert "Nearest row: T21" in reason
    assert set(namespace["regime_result"].to_frame()["support"]) == {"experimental"}
    # The screen's fit declares q_bounds, so its outcome-axis cells move the score on the logit
    # scale of [0, 1], and the reading reads them through the induced correlation.
    assert namespace["regime_result"].estimator.q_bounds == (0.0, 1.0)
    _check_refutation_limits(namespace)
    _check_generated_limits(namespace)
    _check_bound_and_surfaces(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation and the stress surfaces."""
    # The treatment-only stress surface: both cells return, and the flip moves the screen
    # estimate down by 0.029037 (-6.1 standard errors) with a small induced association.
    surface = namespace["surface"]
    surface_summary = surface.summary(detail="full")
    settings = surface_summary.section("Settings").lines
    assert any(line.startswith("Bounded outcome law: logit on [0, 1];") for line in settings)
    law = vocabulary.NOTE_TEXTS["law_outcome_logit"]
    assert law in surface_summary.note("sensitivity_assumption").text
    cells = {(cell.treatment_strength, cell.outcome_strength): cell for cell in surface.cells}
    assert set(cells) == {(0.0, 0.0), (0.0, 0.3), (0.1, 0.0), (0.1, 0.3)}
    assert all(cell.failure is None for cell in cells.values())
    assert cells[0.1, 0.0].displacement == pytest.approx(-0.0290, abs=0.001)
    # "The flip barely ties the hidden variable to the offer": every association is small.
    for cell in cells.values():
        assert cell.induced_treatment_association is not None
        assert abs(cell.induced_treatment_association) < 0.1
    assert cells[0.0, 0.3].induced_outcome_association == pytest.approx(-0.211, abs=5e-4)
    _check_surface_relations(cells, namespace["regime_result"][SCREEN].std_error)
    nudge = {
        (c.treatment_strength, c.outcome_strength): c for c in namespace["nudge_surface"].cells
    }
    _check_surface_relations(nudge, namespace["incremental_result"][IPSI].std_error)
    # The raise: the score strength 0.6 is in score units, and its correlation is computed.
    raise_cells = {
        (c.treatment_strength, c.outcome_strength): c for c in namespace["raise_surface"].cells
    }
    assert set(raise_cells) == {(0.0, 0.0), (0.0, 0.6), (0.1, 0.0), (0.1, 0.6)}
    assert all(cell.induced_outcome_association is None for cell in raise_cells.values())
    variance = float(namespace["dose_frame"]["transition_score"].var(ddof=0))
    assert -0.6 / np.sqrt(variance + 0.36) == pytest.approx(-0.205, abs=5e-4)
    se = namespace["shift_result"][CAPPED].std_error
    moves = {key: cell.displacement / se for key, cell in raise_cells.items()}
    # "That alone moves the raise 0.3 standard errors, but together with the intensity noise
    # 4.1": the joint cell moves more than the sum of its parts.
    assert abs(moves[0.0, 0.6]) < 0.5
    assert moves[0.1, 0.6] < moves[0.1, 0.0] + moves[0.0, 0.6] < 0.0
    _check_refutation(namespace)
    _check_benchmarks(namespace)


def _check_surface_relations(cells: dict[tuple[float, float], Any], se: float) -> None:
    """The outcome-only cell barely moves the estimate; the joint cell moves it a little more."""
    moves = {key: cell.displacement / se for key, cell in cells.items()}
    assert abs(moves[0.0, 0.3]) < 0.5
    assert moves[0.1, 0.0] < -5.0
    # The part a confounder adds, joint minus flip, is negative and small beside the flip.
    added = moves[0.1, 0.3] - moves[0.1, 0.0]
    assert -0.3 * abs(moves[0.1, 0.0]) < added < 0.0


def _check_refutation(namespace: dict[str, Any]) -> None:
    """The refutation of the three fits, and the raise's off-zero placebo."""
    regimes = namespace["regime_result"]
    screen_report = namespace["screen_refutation"]
    assert screen_report.estimand == SCREEN
    every = ["placebo", "subset", "random_common_cause"]
    for key, replicates, names in (
        ("screen_refutation", 20, every),
        ("nudge_refutation", 20, every),
        ("raise_refutation", 10, ["placebo", "subset"]),
    ):
        report = namespace[key]
        assert [test.name for test in report.tests] == names
        for test in report.tests:
            # "No check prints a verdict": no registry row covers these fits.
            assert test.passed is None and test.rung == "L0" and len(test.values) == replicates
        assert report.passed is None
    tests = {test.name: test for test in screen_report.tests}
    original = regimes[SCREEN].psi
    compared = screen_report.summary(detail="full").section(vocabulary.SECTION_COMPARED).lines
    assert any(
        "another estimate (ate_regime[screen on risk vs offer to none])" in line
        for line in compared
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
    # The noise column moves neither the screen nor the nudge by more than 0.1 standard error.
    for key in ("screen_refutation", "nudge_refutation"):
        noise = {test.name: test for test in namespace[key].tests}["random_common_cause"]
        assert abs(noise.statistic) < 0.1
    # "The raise's shuffled-intensity refits sit above zero ... with a t of 3.01, and its 70%
    # refits spread 1.29 times what its standard error predicts." A relation, not a pass.
    raised = {test.name: test for test in namespace["raise_refutation"].tests}
    assert np.mean(raised["placebo"].values) > 0.0 and raised["placebo"].statistic > 2.0
    assert raised["subset"].statistic > 1.0
    stored = stored_output(NOTEBOOK, "refute-regimes")
    assert (
        "raise placebo: t = 3.01" in stored and "raise subset: scatter ratio rho = 1.29" in stored
    )
    # "a standard error of about 16%" (20 refits) and "about 24%" (10): 1 / sqrt(2 (R - 1)).
    assert round(100 / np.sqrt(2 * 19)) == 16 and round(100 / np.sqrt(2 * 9)) == 24
    # The screen and nudge placebos sit near zero, a control for the raise's.
    for key in ("screen_refutation", "nudge_refutation"):
        placebo_i = {test.name: test for test in namespace[key].tests}["placebo"]
        assert abs(placebo_i.statistic) < 2.0


def _check_refutation_limits(namespace: dict[str, Any]) -> None:
    """Step 14: the stored output, and the negative control that the page says it cannot run."""
    regimes = namespace["regime_result"]
    stored = stored_output(NOTEBOOK, "refute-regimes")
    assert stored.count("descriptive") == 8
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


def _check_generated_limits(namespace: dict[str, Any]) -> None:
    """The refusals that the tests on simulated scores meet; the Details name the raise's."""
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
    for name, rv, rva in ((SCREEN, 0.3938, 0.3638), (OFFER_ALL, 0.4050, 0.3766)):
        robustness = regimes.sensitivity.robustness_value(estimand=name)
        assert round(robustness["rv"], 4) == rv and round(robustness["rva"], 4) == rva
        assert robustness["limit_licensed"] is False
        assert "No calibration covers this configuration" in robustness["limit_sentence"]
    benchmarks = namespace["benchmarks"]
    assert list(benchmarks) == ["prior_utilization", "medication_burden", "age"]
    strongest, bound = benchmarks["prior_utilization"]
    # "a confounder like it would leave the lower limit at 0.0438": a nonzero witness, the
    # bound moves the estimate, but not to zero.
    assert round(bound.ci_lower, 4) == 0.0438
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
    assert "robustness value, screen: 0.394 (0.364 at the 95% limit)" in text
    assert "robustness value, offer to all: 0.405 (0.377 at the 95% limit)" in text
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
    for line in (
        "0.10             0.0      -6.4              0.023",
        "0.10             0.3      -7.8             -0.211",
        "0.10             0.0      -1.1              0.000",
        "0.10             0.6      -4.1             -0.205",
    ):
        assert line in stored
