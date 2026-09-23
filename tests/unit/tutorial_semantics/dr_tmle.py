"""The reviewed semantic callback for ``docs/examples/dr-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cleverly.datasets import navigation_protocol, nonlinear_bounded_dgp
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "dr-tmle.ipynb"

COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")


def check(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page claims an exact reduction to the ordinary estimator, a close estimate and standard
    error on this draw, passing score and correction reports with no active truncation, score
    checks that also pass for constant reductions that land near the ordinary TMLE, a
    reduced-regression table whose g_r1 fits mostly favor the spline candidate, and a
    sensitivity term nu2 that the doubted assignment model understates.
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

    # Step 3: the unadjusted arm difference exceeds the ATE, and the arms differ on risk.
    by_arm = namespace["by_arm"]
    unadjusted = namespace["unadjusted"]
    # The margin is the retired Gaussian law's, scaled by the ratio of the two ATEs.
    assert unadjusted > namespace["truth"]["ate"] + 0.005
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.4

    # "The catalog lists drtmle as available for this ATE. For the ATT it prints False."
    catalog = {method.name: method for method in namespace["effect"].available_methods()}
    assert catalog["drtmle"].available
    att = namespace["att_catalog"]["drtmle"]
    assert not att.available and att.reason

    ordinary = namespace["ordinary"]["ate"]
    empty_guard = namespace["empty_guard"]["ate"]
    guarded = namespace["guarded"]["ate"]
    assert namespace["drtmle"].guard == ("Q", "g")
    assert ordinary.psi == empty_guard.psi

    # "The 95% interval ... contains the true ATE"; one draw, not a coverage result.
    truth = namespace["truth"]["ate"]
    assert covers(guarded, truth)

    # "On this draw it moves by -0.20 ordinary standard errors, less than one. The
    # standard-error ratio is 0.99." The lower bound witnesses that the guarded estimate moves.
    shift = (guarded.psi - ordinary.psi) / ordinary.std_error
    assert -0.9 < shift < -0.1
    assert 0.8 < guarded.std_error / ordinary.std_error < 1.25

    # The failure mode: constant reductions pass every score check.
    crude_fit = namespace["crude_fit"]
    assert crude_fit.diagnostics.score_equations().passed
    assert crude_fit.diagnostics.corrections().passed
    crude = crude_fit["ate"]
    # "The constant fit is close to [the ordinary TMLE] on this draw, so the constant reductions
    # change little." The witness that the reductions matter: the spline fit moves further.
    assert abs(crude.psi - ordinary.psi) < abs(guarded.psi - ordinary.psi)
    assert abs(crude.std_error / ordinary.std_error - 1) < abs(
        guarded.std_error / ordinary.std_error - 1
    )
    assert abs(crude.psi - guarded.psi) > 0.1 * guarded.std_error
    # "On this draw the spline fit lands nearer the true ATE ... than the constant fit does."
    assert abs(guarded.psi - truth) < abs(crude.psi - truth)

    assessment = namespace["assessment"]
    assert not assessment.attention  # "no row needs attention"
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "no truncation was active"
    assert assessment.report("score_equations").passed
    summary = assessment.summary()
    # "Every omitted-variable operation is unavailable for this fit."
    not_run = summary.split("Not run", 1)[1]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert f"sensitivity.{operation}" in not_run
    # "The evalue row of Step 9 still returns."
    assert "sensitivity  evalue" in summary.split("Not run", 1)[0]

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    reduced = namespace["reduced"]
    assert set(reduced) == {"qr", "gr1", "gr2"}
    # "the spline candidate has the lowest cross-validated risk in four of the six gr1 fits".
    # Fails on a revert to plain linear reducers, which leave these diagnostics empty.
    assert len(reduced["gr1"]) == 6
    assert sum(fit.best == "spline" for fit in reduced["gr1"]) == 4
    # "Each of the three families splits between its two candidates."
    assert {fit.best for fit in reduced["gr1"]} == {"logistic", "spline"}
    for family in ("qr", "gr2"):
        assert {fit.best for fit in reduced[family]} == {"linear", "spline"}

    # Step 10 shows the refusal, and the ledger marks every omitted-variable operation.
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    for operation in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", operation)]) == "unavailable"
    # "The refusal names the fitted method, drtmle, and the quantity the package will not
    # estimate for it."
    refusal = namespace["bound_refusal"]
    assert "'drtmle'" in refusal
    assert "nu^2" in refusal
    assert "does not assume a consistent treatment mechanism" in refusal
    # The refusal keys on the estimator, so a term that vanishes needs its own witness: the
    # ordinary fit of Step 7 shares this page's doubted assignment model, and the bound runs
    # there. Its nu^2 is 4.316 where the sample second moment of the untruncated true ATE
    # representer is 7.75. That gap is the optimism the DR-TMLE refusal prevents. The bounded
    # law keeps nonlinear_dgp's propensity, so the representer survives the outcome family.
    elements = namespace["ordinary"].sensitivity.elements(estimand="ate")
    g0 = nonlinear_bounded_dgp().propensity(frame.loc[:, list(COVARIATES)].to_numpy(dtype=float))
    true_nu2 = float(np.mean(1.0 / g0 + 1.0 / (1.0 - g0)))
    assert round(true_nu2, 2) == 7.75
    assert round(elements.nu2, 3) == 4.316
    assert true_nu2 > 1.7 * elements.nu2
