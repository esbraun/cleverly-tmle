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

_PROBE = "reviews/notebook-review/probes/dr-tmle-final"
_STUDY = "tests/canonical/drtmle/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "-0.0020": f"ordinary TMLE mean estimate minus the truth, 200 draws: {_PROBE}/summary.log",
    "-0.0031": f"DR-TMLE mean estimate minus the truth, 200 draws: {_PROBE}/summary.log",
    "0.0070": f"ordinary TMLE empirical SD, 200 draws: {_PROBE}/summary.log",
    "0.0069": f"DR-TMLE empirical SD, 200 draws: {_PROBE}/summary.log",
    "0.97": f"ordinary TMLE mean SE over empirical SD, 200 draws: {_PROBE}/summary.log",
    "0.98": f"DR-TMLE mean SE over empirical SD, 200 draws: {_PROBE}/summary.log",
    "0.0011": f"mean DR-TMLE minus ordinary estimate (-0.00113), 200 draws: {_PROBE}/summary.log",
    "-0.0019": f"mean ATE remainder over 20 draws: {_PROBE}/remainder.log",
    "0.3": f"ordinary bias over mean SE, 0.00201 / 0.00677, 200 draws: {_PROBE}/summary.log",
    "0.0068": f"ordinary TMLE mean SE (0.00677), 200 draws: {_PROBE}/summary.log",
    "0.05": f"median |constant - ordinary| in ordinary SEs, 200 draws: {_PROBE}/summary.log",
    "0.17": f"median |spline - ordinary| in ordinary SEs, 200 draws: {_PROBE}/summary.log",
    "2.75": f"5% quantile of the E-value of the limit, 200 draws: {_PROBE}/summary.log",
    "3.09": f"95% quantile of the E-value of the limit, 200 draws: {_PROBE}/summary.log",
    "0.9475": f"coverage of double_robustness/outcome_correct: {_STUDY}",
    "0.0026": f"lower 99% bias limit (0.002611) of double_robustness/outcome_correct: {_STUDY}",
    "0.0075": f"upper 99% bias limit (0.007540) of double_robustness/outcome_correct: {_STUDY}",
    "0.0067": f"bias margin (0.006749) of double_robustness/outcome_correct: {_STUDY}",
    "0.945": f"coverage of double_robust_contraction/outcome_correct_n1500: {_STUDY}",
    "0.94": f"coverage of double_robust_contraction/outcome_correct_n3000: {_STUDY}",
    "0.94375": f"coverage of double_robust_contraction/outcome_correct_n6000: {_STUDY}",
}


def check(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page claims an exact reduction to the ordinary estimator, a close estimate and standard
    error on this draw, passing score and correction reports with no active truncation, score
    checks that also pass for constant reductions, a support report with no row below the bound,
    a contract line that names the cross-fitting gap, a reduced-regression table in which the
    fitted weights need not follow the lowest risk, and an approximate E-value on the DR-TMLE fit.
    """
    # The protocol step prints the record, and the guarded fit carries its digest.
    # "The changed field is `assumption rationale`, and only its first entry changes."
    base = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, base) == {"assumption_rationale"}
    assert protocol.assumption_rationale[1:] == base.assumption_rationale[1:]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, namespace["guarded"])

    # The data step uses the shared program columns.
    frame = namespace["frame"]
    assert list(frame.columns)[:2] == ["transition_score", "transition_navigation"]

    # "the true propensity lies between 5% and 95%": the law's propensity on this draw's rows.
    g0 = nonlinear_bounded_dgp().propensity(frame.loc[:, list(COVARIATES)].to_numpy(dtype=float))
    assert g0.min() >= 0.05 and g0.max() <= 0.95

    # Step 3: the unadjusted arm difference exceeds the ATE, and the arms differ on risk.
    by_arm = namespace["by_arm"]
    unadjusted = namespace["unadjusted"]
    assert unadjusted > namespace["truth"]["ate"] + 0.005
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.4

    # "The last line lists drtmle among the methods available for this ATE."
    assert "drtmle" in namespace["available"]

    ordinary = namespace["ordinary"]["ate"]
    empty_guard = namespace["empty_guard"]["ate"]
    guarded = namespace["guarded"]["ate"]
    assert namespace["drtmle"].guard == ("Q", "g")
    assert ordinary.psi == empty_guard.psi
    # "The line `DR-TMLE: guard Q, g; univariate reduction` names the method, the default
    # guard, and the reduction that the fit ran."
    assert "DR-TMLE: guard Q, g; univariate reduction" in stored_output(NOTEBOOK, "estimate")
    assert namespace["guarded"].extra["drtmle"].reduction == "univariate"

    # "The 95% interval ... contains the true ATE"; one draw, not a coverage result.
    truth = namespace["truth"]["ate"]
    assert covers(guarded, truth)

    # "On this draw it moves by -0.03 ordinary standard errors, and the standard-error ratio is
    # 1.00."  The nonzero witness: the guarded estimate does move, below the ordinary one.
    shift = (guarded.psi - ordinary.psi) / ordinary.std_error
    assert -0.1 < shift < -0.01
    assert 0.98 < guarded.std_error / ordinary.std_error < 1.02

    # The failure mode: constant reductions pass every score check, as the spline fit does.
    crude_fit = namespace["crude_fit"]
    assert crude_fit.diagnostics.score_equations().passed
    assert crude_fit.diagnostics.corrections().passed
    crude = crude_fit["ate"]
    # "On this draw the three estimates agree to three decimals."  The witness that the two
    # reductions still differ: their estimates are not equal.
    assert {round(x.psi, 3) for x in (crude, guarded, ordinary)} == {round(ordinary.psi, 3)}
    assert crude.psi != guarded.psi

    assessment = namespace["assessment"]
    assert not assessment.attention  # "no row needs attention"
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "no truncation is active"
    assert assessment.report("score_equations").passed
    # "It also states that Theorem 1 does not cover cross-fitting, and that condition (S) is open."
    stored = stored_output(NOTEBOOK, "assessment")
    assert "Theorem 1 does not cover cross-fitting" in stored
    assert "condition (S) is open" in stored
    # The status table prints only the rows that ran or run on request. The omitted-variable
    # rows, which need a consistent assignment model, stay off the page; the E-value ran.
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "unavailable"
        assert not re.search(rf"sensitivity\s+{operation}\b", stored)
    assert "unavailable" not in stored and "not_applicable" not in stored
    assert str(ledger.loc[("sensitivity", "evalue")]) == "completed"
    assert re.search(r"sensitivity\s+evalue\s+completed", stored)
    assert re.search(r"refute\s+deferred", stored)

    # "No row falls below the bound, which agrees with a law whose true propensity lies between
    # 5% and 95%."  The fitted model's own support report, and its smallest fitted value.
    support = assessment.report("support")
    assert support.truncated["count"] == 0
    assert "maximum truncated fraction 0.0%" in stored

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    reduced = namespace["reduced"]
    assert set(reduced) == {"qr", "gr1", "gr2"}
    # "The spline has the lowest cross-validated risk in three of the six gr1 fits, two of the six
    # qr fits, and one of the six gr2 fits."  Fails on a revert to plain linear reducers, which
    # leave these diagnostics empty.
    for family, count in (("gr1", 3), ("qr", 2), ("gr2", 1)):
        assert len(reduced[family]) == 6
        assert sum(fit.best == "spline" for fit in reduced[family]) == count

    # "In the fourth qr fit the linear candidate has the lower risk, and the fitted weight on the
    # spline is 1.0."  The witness that the fitted reduction is not the best candidate.
    fourth = reduced["qr"][3]
    assert fourth.best == "linear"
    assert round(float(fourth.weights[list(fourth.names).index("spline")]), 2) == 1.0

    # Step 10: the approximate E-value of the DR-TMLE fit, and the printed conversion chain.
    evalue = namespace["evalue"]
    assert evalue.approximate
    assert evalue.point > evalue.limit > 1.0
    printed = stored_output(NOTEBOOK, "sensitivity")
    for words in (
        "Standardised by sd(Y) = 0.2346",
        "Chinn's log(OR) / 1.81 step",
        "common-outcome square-root conversion",
        "risk ratio 1.874",
    ):
        assert words in printed
    # The chain the reading names: d = psi / sd(Y), log OR = 1.81 d, RR = sqrt(OR).
    sd_y = float(np.std(frame["transition_score"], ddof=1))
    assert np.isclose(np.sqrt(np.exp(1.81 * guarded.psi / sd_y)), evalue.risk_ratio, rtol=1e-3)
