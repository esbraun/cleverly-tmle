"""The reviewed semantic callback for ``docs/examples/dr-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "dr-tmle.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page claims an exact reduction to the ordinary estimator, a close estimate and standard
    error on this draw, passing score and correction reports with no active truncation, score
    checks that also pass for constant reductions, and a reduced-regression table whose g_r1
    fits all favor the spline candidate.
    """
    # The protocol step prints the record, and the guarded fit carries its digest.
    fingerprint = namespace["protocol"].fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert namespace["guarded"].provenance.protocol_fingerprint == fingerprint

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
    assert guarded.ci[0] <= truth <= guarded.ci[1]

    # "moves by less than one standard error, and the standard errors are similar". The lower
    # bound witnesses "the guarded estimate moves": about 0.4 SE on this draw.
    shift = abs(guarded.psi - ordinary.psi) / ordinary.std_error
    assert 0.1 < shift < 0.9
    assert 0.8 < guarded.std_error / ordinary.std_error < 1.25

    # The failure mode: constant reductions pass every score check and move the estimate. The
    # page also narrates that, on this draw, the constant fit lands nearer the truth.
    crude_fit = namespace["crude_fit"]
    assert crude_fit.diagnostics.score_equations().passed
    assert crude_fit.diagnostics.corrections().passed
    crude = crude_fit["ate"]
    assert crude.psi != guarded.psi
    assert abs(crude.psi - truth) < abs(guarded.psi - truth)

    assessment = namespace["assessment"]
    assert not assessment.attention  # "no row needs attention"
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "no truncation was active"
    assert assessment.report("score_equations").passed
    summary = assessment.summary()
    assert "omitted_confounding" in summary

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    reduced = namespace["reduced"]
    assert set(reduced) == {"qr", "gr1", "gr2"}
    # Fails on a revert to plain linear reducers, which leave these diagnostics empty.
    assert reduced["gr1"] and all(fit.best == "spline" for fit in reduced["gr1"])
    # "The Q_r and g_r2 fits split between the two candidates."
    for family in ("qr", "gr2"):
        assert {fit.best for fit in reduced[family]} == {"linear", "spline"}

    # The robustness value is positive and exceeds its confidence-limit value, as narrated.
    robustness = namespace["robustness"]
    assert 0.0 < robustness["rva"] < robustness["rv"]
