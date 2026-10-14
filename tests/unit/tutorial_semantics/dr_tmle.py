"""The reviewed semantic callback for ``docs/examples/dr-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from cleverly.datasets import navigation_protocol, nonlinear_bounded_dgp
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "dr-tmle.ipynb"

COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")

_PROBE = "reviews/notebook-review/probes/dr-tmle-known/summary.log"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "-0.0008": f"ordinary TMLE mean estimate minus the truth, 200 draws: {_PROBE}",
    "-0.0003": f"DR-TMLE mean estimate minus the truth, 200 draws: {_PROBE}",
    "0.0087": f"empirical SD of both estimators, 200 draws: {_PROBE}",
    "0.0093": f"ordinary TMLE mean SE, 200 draws: {_PROBE}",
    "0.0092": f"DR-TMLE mean SE, 200 draws: {_PROBE}",
    "1.06": f"mean SE over empirical SD for both estimators, 200 draws: {_PROBE}",
}


def check(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page declares the law's true propensity as a known mechanism and fits a wrong outcome
    regression in sample.  It claims an exact reduction to the ordinary estimator, a guard table
    in which every extra equation moves the estimate, passing score and correction reports with
    no active truncation, score checks that also pass for constant reductions, a support report
    that reads the declared mechanism, and an approximate E-value on the DR-TMLE fit.
    """
    # The protocol step prints the record, and the guarded fit carries its digest.
    # "The changed field is `assumption rationale`, and only its first entry changes."
    base = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, base) == {"assumption_rationale"}
    assert protocol.assumption_rationale[1:] == base.assumption_rationale[1:]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, namespace["guarded"])

    # The data step adds the law's own propensity as the recorded probabilities.
    frame = namespace["frame"]
    assert list(frame.columns)[:2] == ["transition_score", "transition_navigation"]
    g0 = nonlinear_bounded_dgp().propensity(frame.loc[:, list(COVARIATES)].to_numpy(dtype=float))
    np.testing.assert_array_equal(frame["offer_probability"].to_numpy(), g0)
    np.testing.assert_array_equal(frame["usual_probability"].to_numpy(), 1.0 - g0)
    # "the true propensity lies between 5% and 95%"
    assert g0.min() >= 0.05 and g0.max() <= 0.95

    # Step 3: the unadjusted arm difference exceeds the ATE, and the arms differ on risk.
    by_arm = namespace["by_arm"]
    unadjusted = namespace["unadjusted"]
    assert unadjusted > namespace["truth"]["ate"] + 0.005
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.4

    # Step 5: the design declares the columns, and C-TMLE leaves the catalog.
    design = namespace["study"].design
    assert design.treatment_probabilities == ((0, "usual_probability"), (1, "offer_probability"))
    assert namespace["available"] == ["tmle", "drtmle"]

    ordinary = namespace["ordinary"]["ate"]
    empty_guard = namespace["empty_guard"]["ate"]
    guarded = namespace["guarded"]["ate"]
    assert namespace["drtmle"].guard == ("Q", "g")
    assert namespace["drtmle"].cross_fitting.enabled is False
    assert ordinary.psi == empty_guard.psi
    stored = stored_output(NOTEBOOK, "estimate")
    for line in (
        "DR-TMLE: guard Q, g; univariate reduction",
        "treatment mechanism: known (declared); no truncation bound moved a known value",
        "TMLE (in-sample nuisances)",
    ):
        assert line in stored
    assert "outcome scaled" not in stored
    assert namespace["guarded"].extra["drtmle"].reduction == "univariate"

    # "The 95% interval ... contains the true ATE"; one draw, not a coverage result.
    truth = namespace["truth"]["ate"]
    assert covers(guarded, truth)

    # Step 7: "each guard moves the estimate", and "Every interval contains the true ATE".
    # The nonzero witness for each extra equation: no guard reproduces another's estimate.
    by_guard = namespace["by_guard"]
    estimates = [by_guard[guard]["ate"].psi for guard in [(), ("g",), ("Q",), ("Q", "g")]]
    assert len(set(estimates)) == 4
    assert all(covers(by_guard[guard]["ate"], truth) for guard in by_guard)
    # "the default guard moves it by -0.30 ordinary standard errors, and the standard-error
    # ratio is 0.91."
    shift = (guarded.psi - ordinary.psi) / ordinary.std_error
    assert -0.35 < shift < -0.25
    assert 0.89 < guarded.std_error / ordinary.std_error < 0.93

    # The failure mode: constant reductions pass every score check, as the fitted ones do.
    crude_fit = namespace["crude_fit"]
    assert crude_fit.diagnostics.score_equations().passed
    assert crude_fit.diagnostics.corrections().passed
    # The witness that the two reductions still differ: their estimates are not equal.
    assert crude_fit["ate"].psi != guarded.psi

    assessment = namespace["assessment"]
    assert not assessment.attention  # "no row needs attention"
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "None of the three truncations is active"
    assert assessment.report("score_equations").passed
    stored = stored_output(NOTEBOOK, "assessment")
    # "The fit runs in sample, so no cross-fitting condition applies."
    assert "this fit is Theorem 1's estimator" in stored
    assert "does not cover cross-fitting" not in stored
    # The status table prints only the rows that ran or run on request. The omitted-variable
    # rows are unavailable for a drtmle fit and stay off the page; the E-value ran.
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "unavailable"
        assert not re.search(rf"sensitivity\s+{operation}\b", stored)
    assert "unavailable" not in stored and "not_applicable" not in stored
    assert str(ledger.loc[("sensitivity", "evalue")]) == "completed"
    assert re.search(r"sensitivity\s+evalue\s+completed", stored)
    assert re.search(r"refute\s+deferred", stored)

    # "The support report reads the recorded probabilities": its smallest value is the smallest
    # declared probability, and no row is truncated.
    support = assessment.report("support")
    assert support.truncated["count"] == 0
    assert "treatment mechanism known (declared)" in stored
    assert f"most extreme untruncated g(W) = {g0.min():.6f}" in stored

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    # Step 10: the approximate E-value of the DR-TMLE fit, and the printed conversion chain.
    evalue = namespace["evalue"]
    assert evalue.approximate
    assert evalue.point > evalue.limit > 1.0
    printed = stored_output(NOTEBOOK, "sensitivity")
    for words in (
        "Standardised by sd(Y) = 0.2346",
        "Chinn's log(OR) / 1.81 step",
        "common-outcome square-root conversion",
        "risk ratio 1.897",
    ):
        assert words in printed
    # The chain the reading names: d = psi / sd(Y), log OR = 1.81 d, RR = sqrt(OR).
    sd_y = float(np.std(frame["transition_score"], ddof=1))
    assert np.isclose(np.sqrt(np.exp(1.81 * guarded.psi / sd_y)), evalue.risk_ratio, rtol=1e-3)
