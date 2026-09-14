"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import pytest
from scipy.special import expit

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import EXAMPLES, changed_fields, stored_output

NOTEBOOK = EXAMPLES / "survey-nonresponse.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The survey question reports the observation factor its fitted score uses.

    The same documented-size run also witnesses the tutorial's interval and sensitivity claims.
    """
    effect = namespace["effect"]
    fitted = namespace["full"]
    expression = effect.functional.expression
    assumptions = " ".join(effect.identification.assumptions).lower()
    nuisances = effect.identification.required_nuisances
    summary = effect.summary()

    assert "responded=1" in expression
    assert "missingness at random" in assumptions
    assert "response positivity" in assumptions
    assert "missingness_mechanism" in nuisances
    assert fitted.nuisance.missingness is not None
    assert expression in summary
    assert "missingness_mechanism" in summary
    assert "E_W[E(Y | A=a, W)] and the declared smooth contrast" not in summary
    assert "stacked CV-TMLE" in fitted.summary()

    # The protocol step prints the record, and the fit carries its digest.
    # The reading names the fields this page changes in the program protocol.
    assert changed_fields(namespace["protocol"], navigation_protocol()) == {
        "target_population",
        "outcome",
        "assumption_rationale",
    }
    fingerprint = namespace["protocol"].fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert fitted.provenance.protocol_fingerprint == fingerprint
    assert "scores death before day 30 as the worst transition score" in summary

    # "About a quarter of patients never return the 30-day survey."
    frame = namespace["frame"]
    assert 0.70 < float(frame["responded"].mean()) < 0.80
    assert int(frame["transition_score"].isna().sum()) == int((frame["responded"] == 0).sum())

    # "Respondents are a lower-risk group", and the naive respondent contrast overshoots.
    by_response = namespace["by_response"]
    assert by_response.loc[0.0, "discharge_risk"] > by_response.loc[1.0, "discharge_risk"] + 0.5
    truth = namespace["truth"]["ate"]
    assert namespace["unadjusted"] > truth

    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    assert complete_case.psi > truth
    assert not complete_case.ci[0] <= truth <= complete_case.ci[1]
    assert full.ci[0] <= truth <= full.ci[1]

    # "Both laws have the same population ATE", and the mild complete-case interval covers it.
    # The strength-2 interaction averages W1, whose mean is 0, so the laws share the ATE of 1.2;
    # the tolerance absorbs only the numerical integration of the stronger law's truth.
    mild_truth = namespace["mild_truth"]["ate"]
    assert mild_truth == pytest.approx(truth, abs=1e-4)
    mild = namespace["mild"]["ate"]
    assert mild.ci[0] <= mild_truth <= mild.ci[1]

    # "Each interval contains its population value", the odds ratio lies above the risk ratio,
    # and the ratio intervals are asymmetric on the reported scale.
    box_points = namespace["box_points"]
    box_truth = namespace["box_truth"]
    for key, point in box_points.items():
        assert point.ci[0] <= box_truth[key] <= point.ci[1]
    assert box_points["or"].psi > box_points["rr"].psi > 1.0
    for key in ("rr", "or"):
        point = box_points[key]
        assert point.ci[1] - point.psi > point.psi - point.ci[0]

    # The refusal is the attributable fraction's missing-outcome refusal, not another error.
    assert "does not yet support PointTreatment(missingness=...)" in namespace["refusal"]

    # "No row needs attention", and the support report prints the joint mechanism row.
    assert tuple(namespace["assessment"].attention) == ()
    assert "P(A=a,Delta=1|W)" in stored_output(NOTEBOOK, "assessment")

    # "near gamma=1.3", and "at most 0.286 of the score range" is 0.5 - expit(-gamma).
    tipping = namespace["tipping_gamma"]
    assert tipping == pytest.approx(1.30, abs=0.03)
    assert namespace["largest_shift"] == pytest.approx(0.5 - expit(-tipping))

    # The curve's gamma=0 row is the MAR estimate, and its interval reuses the MAR standard error.
    curve = namespace["missingness_curve"]
    mar = curve[curve["gamma"] == 0.0]
    assert float(mar["psi"].iloc[0]) == pytest.approx(full.psi, rel=1e-9)
    assert (curve["std_err"] == full.std_error).all()
    # Positive gamma lowers the navigation-arm mean, so the ATE falls along the grid and
    # changes sign between gamma = 1 and gamma = 2, where the tipping gamma lies.
    assert curve["psi"].is_monotonic_decreasing
    assert float(curve.loc[curve["gamma"] == 1.0, "psi"].iloc[0]) > 0.0
    assert float(curve.loc[curve["gamma"] == 2.0, "psi"].iloc[0]) < 0.0
