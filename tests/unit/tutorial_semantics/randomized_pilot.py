"""The reviewed semantic callback for ``docs/examples/randomized-pilot``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the refutation cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw or about a
committed probe log, not a statistical guarantee.
"""

from __future__ import annotations

import re
from typing import Any

from cleverly.datasets import PILOT_OFFER_CHANCES, randomized_pilot_truth
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refutation",)

NOTEBOOK = EXAMPLES / "randomized-pilot.ipynb"
_PROBE = "reviews/notebook-review/probes/randomized-pilot"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "-0.0009": f"mean error of TMLE with the recorded chances, 200 draws: {_PROBE}/sweep.log",
    "0.98": f"mean SE over the SD of that fit's estimates, 200 draws: {_PROBE}/sweep.log",
    "-0.0622": f"mean error of TMLE with the fitted chances, 200 draws: {_PROBE}/sweep.log",
    "1.27": f"mean SE over the SD of the fitted-chance estimates: {_PROBE}/sweep.log",
    "4.2": "a section number of Moore and van der Laan (2009), not a measured value",
    "0.5": "the AUC of a model that ranks no better than chance, a definition",
    "0.118": (
        "the size of the -0.118 that draw-check prints, and the threshold of the count in "
        f"{_PROBE}/balance.log"
    ),
    "0.0241": f"SD of the default and known-score ATT estimates, 200 draws: {_PROBE}/sweep.log",
}


#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "identify-reading": {"0.2": "association", "0.8": "association"},
    "draw-check-reading": {
        "0.193": "association",
        "0.795": "association",
        "0.2": "association",
        "0.8": "association",
    },
    "interpret": {"0.119": "sensitivity"},
}


def _assert_sweep_log() -> None:
    """Each repeated-draw number of the readings matches the committed sweep log."""
    log = (ROOT / _PROBE / "sweep.log").read_text(encoding="utf-8")
    assert "n = 3000; draws = 200" in log

    def row(name: str) -> str:
        match = re.search(rf"^{name} .*$", log, flags=re.MULTILINE)
        assert match, name
        return match.group(0)

    known = row("tmle_known_in")
    assert "mean error -0.0009" in known and "SE/SD 0.98" in known
    assert "covered 186 of 200" in known
    fitted = row("tmle_fitted_g_in")
    assert "mean error -0.0622" in fitted and "SE/SD 1.27" in fitted
    assert "covered 58 of 200" in fitted
    # "contained the truth 192 times": the ATT of the page's fit, against its exact truth.
    truth = randomized_pilot_truth()
    att = row("att")
    assert f"true {truth['att']:.4f}" in att and "covered 192 of 200" in att
    # "the same spread as the default form's (a standard deviation of 0.0241)"
    assert "SD 0.0241" in att and "SD 0.0241" in row("att_ks")


def _assert_balance_log() -> None:
    """The balance reading's repeated-draw count matches the committed balance log."""
    log = (ROOT / _PROBE / "balance.log").read_text(encoding="utf-8")
    assert "n = 3000; draws = 200" in log
    # "The largest gap reached 0.118 on 5 of them"
    assert "largest at or above 0.118 on 5 of 200 draws" in log


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    assert frame.shape == (3000, 9)
    assert truth == randomized_pilot_truth()
    # The simple comparison has the wrong sign, and the risk group explains it.
    assert namespace["unadjusted"] < 0.0 < truth["ate"]
    groups = namespace["groups"]
    assert groups["recorded_chance"].to_dict() == {
        float(group): chance for group, chance in PILOT_OFFER_CHANCES.items()
    }
    assert groups.loc[1.0, "success_rate"] < groups.loc[0.0, "success_rate"]
    assert groups.loc[1.0, "share_offered"] > 0.75 > 0.25 > groups.loc[0.0, "share_offered"]


def _check_protocol(namespace: dict[str, Any]) -> None:
    protocol, program = namespace["protocol"], namespace["program"]
    assert changed_fields(protocol, program) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert_protocol_recorded(
        NOTEBOOK, "protocol", protocol, namespace["result"], namespace["fitted"]
    )


def _check_estimate(namespace: dict[str, Any]) -> None:
    estimate, truth = namespace["estimate"], namespace["truth"]
    assert covers(estimate, truth["ate"])
    assert estimate.ci[0] > 0.0
    assert (estimate.support.label, estimate.support.row) == ("supported", "T31")
    assert "Treatment model:      known (declared)" in stored_output(NOTEBOOK, "estimate")


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    truth = namespace["truth"]["ate"]
    recorded, fitted = namespace["estimate"], namespace["fitted"]["ate"]
    # The fitted chances pull the estimate down and further from the truth.
    assert fitted.psi < recorded.psi
    assert abs(fitted.psi - truth) > abs(recorded.psi - truth)
    # "still just reaches the truth": the upper limit is within 0.01 above it.
    assert covers(fitted, truth) and fitted.ci[1] - truth < 0.01
    # The fitted model smooths the jump: chances outside the recorded ones in both groups.
    chances = namespace["chances"]
    assert chances.loc[0.0, "min"] < 0.2 < chances.loc[0.0, "max"]
    assert chances.loc[1.0, "min"] < 0.8 < chances.loc[1.0, "max"]
    _assert_sweep_log()


def _check_assessment(namespace: dict[str, Any]) -> None:
    assert namespace["attention"] == ()
    support = namespace["support"]
    assert abs(support.truncated["most_extreme"] - 0.2) < 1e-12
    assert support.truncated["fraction"] == 0.0
    assert "The models show no problem these metrics can detect." in stored_output(
        NOTEBOOK, "assessment"
    )
    # "The estimating equations passed"
    assert namespace["score"].passed and namespace["worst"] < 1.0
    # A weight of 1 / 0.2 = 5 is the largest weight.
    for size in support.effective_sample_size.values():
        assert 0.6 < size["ratio"] < 0.7


def _check_draw(namespace: dict[str, Any]) -> None:
    table, balance = namespace["table"], namespace["balance"]
    assert balance.threshold == 0.1
    after = table["after weighting"]
    # The weighting removes the discharge-risk gap and leaves medication burden past the line,
    # the only covariate there, moved away from zero by the weights.
    assert (
        table.loc["discharge_risk", "before weighting"] > 1.0 > 0.02 > abs(after["discharge_risk"])
    )
    assert after.abs().idxmax() == "medication_burden" and (after.abs() > 0.1).sum() == 1
    burden = table.loc["medication_burden"]
    assert after["medication_burden"] < burden["before weighting"] < 0.0
    assert abs(after["medication_burden"] - (-0.118)) < 5e-4
    _assert_balance_log()


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    robustness = namespace["robustness"]
    assert 0.0 < robustness["rva"] < robustness["rv"]
    # "would need to add only 0.009": the bisection found a strength where the limit tips.
    tipping = namespace["tipping_cf_d"]
    bound = namespace["sensitivity"].omitted_confounding
    assert (
        bound(cf_y=1.0, cf_d=tipping * 0.9).ci_lower
        > 0
        > bound(cf_y=1.0, cf_d=tipping * 1.1).ci_lower
    )
    evalue = namespace["evalue"]
    assert evalue.point > evalue.limit > 1.0
    assert abs(evalue.risk_ratio - namespace["others"]["risk ratio"].psi) < 1e-9


def _check_other_estimands(namespace: dict[str, Any]) -> None:
    truth, others = namespace["truth"], namespace["others"]
    att, atc, ratio = others["ATT"], others["ATC"], others["risk ratio"]
    # "The ATT interval misses its truth by a little on this draw."
    assert not covers(att, truth["att"]) and att.ci[0] - truth["att"] < 0.005
    assert covers(atc, truth["atc"])
    assert covers(ratio, truth["ey1"] / truth["ey0"])
    # "The true values ... differ": the larger effect is among the offered.
    assert truth["att"] > truth["ate"] > truth["atc"]
    assert all(point.support.label == "supported" for point in others.values())


def _check_comparison(namespace: dict[str, Any]) -> None:
    truth = namespace["truth"]["ate"]
    fits = namespace["fits"]
    rows = {key: fit["ate"].support.row for key, fit in fits.items()}
    assert rows == {
        ("TMLE", "in sample"): "T31",
        ("DR-TMLE", "in sample"): "D06.qg",
        ("TMLE", "out of fold"): "T32",
        ("adaptive IPTW", "out of fold"): "AI.known_arms.targeted",
    }
    points = [fit["ate"] for fit in fits.values()]
    assert all(point.support.label == "supported" for point in points)
    assert all(covers(point, truth) for point in points)
    assert max(point.psi for point in points) - min(point.psi for point in points) < 0.01
    corrections = namespace["corrections"]
    assert corrections.passed and corrections.truncations_active == ()


def _check_step_bound(namespace: dict[str, Any]) -> None:
    walk = namespace["walk"]
    # The page prints whether the walk converged, not how many iterations it took.
    assert walk.converged and walk.n_iter > 1
    point, estimate = namespace["point"], namespace["estimate"]
    assert abs(point.psi - estimate.psi) < 1e-6
    assert all(abs(a - b) < 5e-5 for a, b in zip(point.ci, estimate.ci, strict=True))
    assert namespace["point"].support.label == "supported"


def check(namespace: dict[str, Any]) -> None:
    """Assert each claim the readings make about the stored outputs."""
    _check_data(namespace)
    _check_protocol(namespace)
    _check_estimate(namespace)
    _check_failure_mode(namespace)
    _check_assessment(namespace)
    _check_draw(namespace)
    _check_sensitivity(namespace)
    _check_other_estimands(namespace)
    _check_comparison(namespace)
    _check_step_bound(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """Assert the refutation readings."""
    estimate = namespace["estimate"]
    tests = {test.name: test for test in namespace["refutation"].tests}
    assert list(tests) == ["placebo", "random_common_cause", "subset"]
    for test in tests.values():
        assert test.verdict == "descriptive" and test.passed is None
        assert test.original == estimate.psi
    assert abs(tests["placebo"].mean) < 0.01 * estimate.psi
    assert abs(tests["random_common_cause"].statistic) < 0.1
    # "a little less than the predicted spread": a scatter ratio below one.
    assert 0.7 < tests["subset"].statistic < 0.9
