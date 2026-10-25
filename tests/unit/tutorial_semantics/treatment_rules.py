"""The reviewed semantic callback for ``docs/examples/treatment-rules``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`, and the slow tier adds :func:`check_refits`.
Each assertion pins a reading the prose makes about this draw.  A relation on one draw certifies
no method; the repeated-sampling numbers the prose quotes come from the committed review probe,
which :func:`_check_probe_log` reads, and from the registered study pages, which
:func:`_check_study_pages` reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
REFIT_CELLS = ("sensitivity", "refute", "refute-generated")

NOTEBOOK = EXAMPLES / "treatment-rules.ipynb"

_PROBE_NAME = "reviews/notebook-review/probes/treatment-rules/probe.log"
_PROBE = ROOT / _PROBE_NAME
_EVIDENCE = ROOT / "docs" / "technical-reference" / "method-evidence"

#: The probe lines the readings quote, keyed by the decimal they quote.
_PROBE_LINES = {
    "0.0871": "in-sample score minus true value, forest: mean 0.0871",
    "0.0036": "in-sample score minus true value, logistic: mean 0.0036",
    "0.9475": "coverage 0.9475",
    "0.9918": "SE over SD 0.9918",
    "0.9375": "coverage 0.9375",
}

#: The registered-study coverages the readings quote, keyed by the study page that prints them.
_STUDY_COVERAGES = {
    "learned-rule-cvtmle.md": ("0.9465", "0.9413"),
    "learned-rule-cvtmle-boundary.md": ("0.8938", "0.8948"),
}

#: Decimals the readings quote from the committed review probe or a study page, which no cell
#: prints.
UNPRINTED_DECIMALS = {
    **dict.fromkeys(_PROBE_LINES, f"a value of the review probe log, {_PROBE_NAME}"),
    **{
        literal: f"a coverage that the registered study page {page} prints"
        for page, literals in _STUDY_COVERAGES.items()
        for literal in literals
    },
    "0.95": "the nominal coverage of a 95% interval, the target the study coverages are read against",
}


def _check_probe_log() -> None:
    """Each quoted probe number appears in the committed probe log, in its line."""
    text = _PROBE.read_text(encoding="utf-8")
    for literal, line in _PROBE_LINES.items():
        assert literal in UNPRINTED_DECIMALS
        assert line in text, f"{line!r} is not in {_PROBE_NAME}"
    # The learned rule's coverage and SE ratio come from the learned-rule line, and the lottery's
    # coverage from the lottery line.
    rule_line = next(line for line in text.splitlines() if line.startswith("learned rule:"))
    assert "SE over SD 0.9918" in rule_line and "coverage 0.9475" in rule_line
    lottery_line = next(line for line in text.splitlines() if line.startswith("lottery minus"))
    assert "coverage 0.9375" in lottery_line
    # "the forest's in-sample score was too high": in every draw.
    forest_line = next(line for line in text.splitlines() if "forest:" in line)
    assert "share above zero 1.0000" in forest_line
    assert "draws 400" in text


def _check_study_pages() -> None:
    """Each quoted study coverage is the one its registered study page prints."""
    for page, literals in _STUDY_COVERAGES.items():
        text = (_EVIDENCE / page).read_text(encoding="utf-8")
        for literal in literals:
            assert f"| {literal} |" in text, f"{literal} is not a coverage of {page}"


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    assert frame.shape == (2_000, 4)
    assert list(frame.columns) == ["need", "lives_alone", "offer", "home_30"]
    assert frame["need"].between(-1.0, 1.0).all()
    # The data reading: "from 0.6522 to 0.6763"; the best rule "would reach 0.7197".
    assert round(truth["ey_none"], 4) == 0.6522 and round(truth["ey_all"], 4) == 0.6763
    assert round(truth["ey_best_rule"], 4) == 0.7197 and truth["best_rule_cutoff"] == -0.1


def _check_association(namespace: dict[str, Any]) -> None:
    frame, truth, by_half = namespace["frame"], namespace["truth"], namespace["by_half"]
    means = frame.groupby("offer")[["need", "lives_alone"]].mean()
    # "Offered patients had higher need and lived alone more often".
    assert (means.loc[1.0] > means.loc[0.0]).all()
    # "the simple difference ... says navigation hurts, while offering it to everyone would in
    # truth raise the share".
    shares = namespace["shares"]
    assert shares[1.0] - shares[0.0] < 0.0 < truth["ate"]
    # "the simple differences point opposite ways".
    difference = by_half[1.0] - by_half[0.0]
    assert difference["need 0 or below"] < 0.0 < difference["need above 0"]


def _check_protocol(namespace: dict[str, Any]) -> None:
    from cleverly.datasets import navigation_protocol

    protocol = namespace["protocol"]
    assert changed_fields(protocol, navigation_protocol()) == {
        "treatment_strategies",
        "treatment_versions",
        "outcome",
        "intercurrent_event_handling",
    }
    assert_protocol_recorded(
        NOTEBOOK, "protocol", protocol, namespace["learned"], namespace["options"]
    )
    # "Note [1] points to seven more conditions".
    assert "states 7 further conditions" in stored_output(NOTEBOOK, "identify")


def _check_learned(namespace: dict[str, Any]) -> None:
    learned, value, record = namespace["learned"], namespace["value"], namespace["record"]
    truth, true_target = namespace["truth"], namespace["true_target"]
    assert value.support.label == "supported" and value.support.row == "T24"
    assert record.n_folds == 10 and set(record.fold_sizes) == {200}
    # "no patient reached them".
    assert "no row reached a limit" in str(learned.summary(detail="full"))
    # At a glance: "each interval contains its true value"; "the estimate lands on it".
    assert covers(value, true_target)
    assert round(value.psi, 4) == round(true_target, 4)
    # "just below the best rule's 0.7197".
    assert 0.0 < truth["ey_best_rule"] - true_target < 0.005
    # "almost no patient with a need score below -0.1 and ... almost every patient above 0.2".
    frame = namespace["frame"]
    offered = namespace["offered_by_rule"]
    assert offered[frame["need"] < -0.1].mean() < 0.01
    assert offered[frame["need"] > 0.2].mean() > 0.99
    # "the rules miss many patients between -0.1 and 0.2": the best rule offers all of them.
    assert offered[frame["need"].between(-0.1, 0.2)].mean() < 0.5
    # "The share each fold's rule offers varies" (printed as 0.345 to 0.545).
    shares = namespace["fold_shares"]
    assert round(min(shares), 3) == 0.345 and round(max(shares), 3) == 0.545


def _check_assessment(namespace: dict[str, Any]) -> None:
    assessment, overlap = namespace["assessment"], namespace["overlap"]
    assert not assessment.attention
    ledger = assessment.to_frame().set_index("check")["status"]
    assert ledger["score_equations"] == "passed"
    assert round(float(overlap["min_propensity"]), 3) == 0.429
    assert round(float(overlap["max_ratio"]), 2) == 2.33
    assert round(float(overlap["effective_n"])) == 1139
    # "Only the 1152 patients whose offer matched the rule carry weight".
    assert namespace["matched"] == 1152 and float(overlap["effective_n"]) <= namespace["matched"]


def _check_failure(namespace: dict[str, Any]) -> None:
    scored = namespace["scored"].set_index("model")
    logistic, forest = scored.loc["logistic, as above"], scored.loc["random forest"]
    # "Both rules look better on the patients they were learned from than they are".
    assert logistic["in-sample score"] > logistic["true value"]
    assert forest["in-sample score"] > forest["true value"]
    # "The in-sample score even ranks the worse rule first".
    assert forest["in-sample score"] > logistic["in-sample score"]
    assert forest["true value"] < logistic["true value"]
    # "In this sample its true value is 0.7175 ..., close to the fold rules' value".
    assert round(float(logistic["true value"]), 4) == 0.7175
    assert abs(float(logistic["true value"]) - namespace["true_target"]) < 0.001


def _check_lottery(namespace: dict[str, Any]) -> None:
    options, gain = namespace["options"], namespace["gain"]
    for name in options.to_frame()["estimand"]:
        support = options[name].support
        assert support.label == "supported" and support.row == "T20"
    # "With no offers, 0.668 ...; under the lottery, 0.694".
    assert round(options["ey_regime[offer to none]"].psi, 3) == 0.668
    assert round(options["ey_regime[lottery]"].psi, 3) == 0.694
    # At a glance: the gain's interval contains the true gain, and is clear of zero.
    true_gain, truth = namespace["true_gain"], namespace["truth"]
    assert covers(gain, true_gain) and gain.ci[0] > 0.0
    # "so it gains more than offering navigation to everyone would".
    assert true_gain > truth["ate"]
    # The lottery offers navigation to about a quarter of patients.
    frame = namespace["frame"]
    chance = namespace["lottery_chances"](frame)[:, 1]
    assert 0.2 < chance.mean() < 0.3
    assert np.allclose(namespace["lottery_chances"](frame).sum(axis=1), 1.0)
    # "nothing that needs attention, and the largest weight is 3.19".
    assert not namespace["options_assessment"].attention
    weights = namespace["lottery_overlap"]["max_ratio"]
    assert round(float(weights.max()), 2) == 3.19


def check(namespace: dict[str, Any]) -> None:
    """Pin the readings of the learned-rule and lottery fits on their seeded draw."""
    _check_probe_log()
    _check_study_pages()
    _check_data(namespace)
    _check_association(namespace)
    _check_protocol(namespace)
    _check_learned(namespace)
    _check_assessment(namespace)
    _check_failure(namespace)
    _check_lottery(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The sensitivity, refutation and generated-outcome refits of the lottery's gain."""
    lottery_fit, gain = namespace["lottery_fit"], namespace["gain"]
    name = namespace["LOTTERY"]
    # The named contrast that the checks read is the `options.contrast` difference.
    assert lottery_fit[name].psi == pytest.approx(gain.psi, abs=1e-6)
    assert lottery_fit[name].std_error == pytest.approx(gain.std_error, abs=1e-6)

    robustness, bound = namespace["robustness"], namespace["bound"]
    # "The robustness value, 0.076 ... 0.039 on both sides".
    assert round(robustness["rv"], 3) == 0.076 and round(robustness["rva"], 3) == 0.039
    # "would shift the gain by at most 0.0008, leaving one-sided 95% limits of 0.0124 and 0.0400".
    assert round(bound.bias, 4) == 0.0008 and bound.ci_lower > 0.0
    assert round(bound.ci_lower, 4) == 0.0124 and round(bound.ci_upper, 4) == 0.0400
    # "They are one-sided, which is why they sit inside the two-sided interval".
    low, high = lottery_fit[name].ci
    assert low < bound.ci_lower and bound.ci_upper < high
    # "Living alone has strengths of only 0.003 and 0.002".
    benchmark = namespace["benchmark"]
    assert round(benchmark.cf_y, 3) == 0.003 and round(benchmark.cf_d, 3) == 0.002

    reports = namespace["refutations"]
    assert list(reports) == ["lottery's gain", "learned rule's value"]
    scatter = {}
    for label, report in reports.items():
        tests = {test.name: test for test in report.tests}
        assert list(tests) == ["placebo", "subset", "random_common_cause"]
        for test in tests.values():
            assert test.verdict == "descriptive" and test.passed is None
            assert test.n_replicates == 20
        # "The noise column barely moves either estimate".
        assert abs(tests["random_common_cause"].statistic) < 0.05
        scatter[label] = round(tests["subset"].statistic, 2)
    # "the scatter ratio ... is near 1 for both" (0.94 and 0.91).
    assert scatter == {"lottery's gain": 0.94, "learned rule's value": 0.91}
    placebo = {label: report.tests[0] for label, report in reports.items()}
    # "the lottery's gain averages 0.0013, near zero, and the rules are worth 0.6692 on average,
    # near the plain share of 0.6695".
    assert round(placebo["lottery's gain"].mean, 4) == 0.0013
    # "the `t` column ... is below 1 for both".
    assert all(abs(test.statistic) < 1.0 for test in placebo.values())
    share = float(namespace["frame"]["home_30"].mean())
    assert round(share, 4) == 0.6695
    rules = placebo["learned rule's value"]
    assert round(rules.mean, 4) == 0.6692
    assert abs(rules.mean - share) < 2 * rules.spread

    generated = namespace["generated_tests"]
    assert list(generated) == ["dummy_outcome", "simulated_outcome", "plasmode"]
    assert generated["dummy_outcome"].declared_effect == 0.0
    for test in generated.values():
        assert test.verdict == "descriptive" and test.n_replicates == 20
        # "Each mean estimate lands near its true gain"; "At 20 refits the interval is wide".
        bias = test.readings[0]
        assert bias.lower < 0.0 < bias.upper
        assert bias.lower < bias.band[0] and bias.band[1] < bias.upper
        assert abs(test.mean - test.declared_effect) < 0.005
    # "such as 0.0271 against 0.0261 for `simulated_outcome`".
    simulated = generated["simulated_outcome"]
    assert round(simulated.mean, 4) == 0.0271 and round(simulated.declared_effect, 4) == 0.0261
    # "The check is meant to run with 400 refits, the number its simulation study used".
    assert "fewer than the 400 its simulation study needs" in simulated.evidence
