"""The reviewed semantic callback for ``docs/examples/point-treatment-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "point-treatment-tmle.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's protocol, failure mode, warnings, and sensitivity claims hold."""
    summary = namespace["effect"].summary()
    assert "Discharge-home order" in summary
    assert "scores death before day 30 as the worst transition score" in summary
    assert "stacked CV-TMLE" in namespace["result"].summary()

    # "The offered patients score ... higher", and the arms differ on discharge risk first.
    by_arm = namespace["by_arm"]
    assert namespace["unadjusted"] > namespace["truth"]["ate"]
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.3

    # The protocol step prints the record, and the fit and the restored artifact carry its digest.
    fingerprint = namespace["protocol"].fingerprint
    assert namespace["protocol"] == navigation_protocol()
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert namespace["result"].provenance.protocol_fingerprint == fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "save-and-restore")
    assert {item.name for item in namespace["replayed"].attention} == {
        item.name for item in namespace["assessment"].attention
    }

    spread_truth = namespace["spread_truth"]
    assert spread_truth["att"] > spread_truth["ate"] > spread_truth["atc"]
    assert spread_truth["atc"] < 0.25 * spread_truth["att"]
    # "misses by several times more than either fit with one flexible learner"; about 6x here.
    truth = namespace["truth"]["ate"]
    errors = {label: abs(point.psi - truth) for label, point in namespace["dr_points"].items()}
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > 3.0 * one_flexible
    # "its interval ... excludes the true value", while the Step 6 interval contains it.
    both_linear = namespace["dr_points"]["both linear"].ci
    assert not both_linear[0] <= truth <= both_linear[1]
    flexible = namespace["estimate"].ci
    assert flexible[0] <= truth <= flexible[1]

    assessment = namespace["assessment"]
    assert "nuisance_models" in {item.name for item in assessment.attention}
    ledger = assessment.to_frame()
    rows = ledger[ledger["check"] == "nuisance_models"]
    assert rows["detail"].str.contains("propensity is poorly calibrated").any()
    assert 0.0 < namespace["support"].truncated["fraction"] < 0.05
    # "more than a quarter of the patients" at the largest bound of the truncation curve.
    assert namespace["curve"]["truncated_fraction"].iloc[-1] > 0.25
    # "about 0.26", at the worst-case alignment rho = 1 the prose names.
    assert 0.2 < namespace["robustness"]["rv"] < 0.33
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("discharge_risk",)
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    assert bounds.ci_lower > 0.0
