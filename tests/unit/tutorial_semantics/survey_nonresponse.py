"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any

import pytest


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

    # "About a quarter of patients never return the 30-day survey."
    assert 0.70 < float(namespace["frame"]["responded"].mean()) < 0.80

    truth = namespace["truth"]["ate"]
    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    assert complete_case.psi > truth
    assert not complete_case.ci[0] <= truth <= complete_case.ci[1]
    assert full.ci[0] <= truth <= full.ci[1]

    mild_truth = namespace["mild_truth"]["ate"]
    mild = namespace["mild"]["ate"]
    assert mild.ci[0] <= mild_truth <= mild.ci[1]

    box_study = namespace["box_study"]
    box_method = namespace["box_method"]
    risk_ratio = box_study.identify(namespace["RiskRatio"](reference=0)).estimate(
        method=box_method
    )["rr"]
    odds_ratio = box_study.identify(namespace["OddsRatio"](reference=0)).estimate(
        method=box_method
    )["or"]
    assert odds_ratio.psi > risk_ratio.psi > 1.0
    # "near gamma=1.3", and "at most about 0.29 of the score range" is expit(gamma) - 0.5.
    assert namespace["tipping_gamma"] == pytest.approx(1.30, abs=0.03)
    # The refusal is the attributable fraction's missing-outcome refusal, not another error.
    assert "does not yet support PointTreatment(missingness=...)" in namespace["refusal"]

    # The curve's gamma=0 row is the MAR estimate, and its interval reuses the MAR standard error.
    curve = namespace["missingness_curve"]
    mar = curve[curve["gamma"] == 0.0]
    assert float(mar["psi"].iloc[0]) == pytest.approx(full.psi, rel=1e-9)
    assert (curve["std_err"] == full.std_error).all()
    # Positive gamma lowers the navigation-arm mean, so the ATE falls along the grid.
    assert curve["psi"].is_monotonic_decreasing
