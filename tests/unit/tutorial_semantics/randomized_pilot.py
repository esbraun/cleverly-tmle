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
    assert "covered 61 of 200" in fitted


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
    assert abs(fitted.psi - truth) > 4 * abs(recorded.psi - truth)
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
    # A weight of 1 / 0.2 = 5 is the largest weight.
    for size in support.effective_sample_size.values():
        assert 0.6 < size["ratio"] < 0.7


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


def _check_step_bound(namespace: dict[str, Any]) -> None:
    walk = namespace["walk"]
    # The iteration count near the tolerance depends on the platform, so the page prints none.
    assert walk.converged and walk.n_iter > 1
    assert abs(namespace["point"].psi - namespace["estimate"].psi) < 1e-6
    assert namespace["point"].support.label == "supported"


def check(namespace: dict[str, Any]) -> None:
    """Assert each claim the readings make about the stored outputs."""
    _check_data(namespace)
    _check_protocol(namespace)
    _check_estimate(namespace)
    _check_failure_mode(namespace)
    _check_assessment(namespace)
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
