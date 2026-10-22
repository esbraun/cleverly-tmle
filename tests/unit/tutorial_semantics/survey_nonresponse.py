"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import csv
import re
from typing import Any

import numpy as np
import pytest
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import CrossFitting
from cleverly.datasets import missing_outcome_bounded_dgp, navigation_protocol
from cleverly.sensitivity import missingness_tilt
from cleverly.sensitivity.omitted_variable import OMITTED_VARIABLE_OPERATIONS
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "survey-nonresponse.ipynb"

_PROBE = "reviews/notebook-review/probes/sn-bounded"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.173": "complete-case target E[tau(W) | Delta = 1] of the law (0.172772), by quadrature: "
    f"{_PROBE}/summary.log",
    "0.960": f"complete-case intervals that contain 0.172772 (480 of 500): {_PROBE}/summary.log",
    "0.00043": f"mean complete-case estimate minus 0.172772, 500 draws: {_PROBE}/summary.log",
    "0.00023": f"Monte Carlo SE of the mean complete-case estimate: {_PROBE}/summary.log",
    "3.17": f"5th percentile of the interval tipping gamma (3.1721): {_PROBE}/summary.log",
    "5.48": f"95th percentile of the interval tipping gamma (5.4783): {_PROBE}/summary.log",
}


def _quadrature_targets() -> tuple[float, float]:
    """The ATE and the complete-case target ``E[tau(W) | Delta = 1]`` of the law."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    mass = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), -1), -1)
    mass = mass.reshape(-1)
    law = missing_outcome_bounded_dgp()
    g = law.propensity(grid)
    respond = g * law.missingness(grid, 1.0) + (1 - g) * law.missingness(grid, 0.0)
    effect = law.outcome_mean(grid, 1.0, None) - law.outcome_mean(grid, 0.0, None)
    ate = float((effect * mass).sum())
    return ate, float((effect * respond * mass).sum() / (respond * mass).sum())


def _read(path: str) -> list[dict[str, str]]:
    with (EXAMPLES.parents[1] / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(namespace: dict[str, Any]) -> None:
    """Witness the identification, graph, comparison, diagnostic, and sensitivity claims.

    The committed review probe also backs each quoted number.
    """
    law = missing_outcome_bounded_dgp()
    truth = namespace["truth"]["ate"]
    ate, cc_target = _quadrature_targets()
    # The library truth and an independent quadrature agree, and the complete-case target
    # differs from the ATE by a visible amount.
    assert ate == pytest.approx(truth, abs=1e-5)
    assert cc_target > ate + 0.02

    # The review probe: each quoted value recomputed from its committed rows.
    probe = _read(f"{_PROBE}/sweep.csv")
    assert len(probe) == 500
    ate_hits = sum(row["covers"] == "True" for row in probe)
    cc = np.array([float(row["cc_psi"]) for row in probe])
    assert ate_hits == 478
    assert sum(row["cc_covers_ate"] == "True" for row in probe) == 0
    assert sum(row["cc_covers_target"] == "True" for row in probe) == 480
    assert f"{cc.mean() - cc_target:.5f}" == "0.00043"
    assert f"{cc.std(ddof=1) / np.sqrt(500):.5f}" == "0.00023"
    tips = np.array([float(row["tip_ci"]) for row in probe if row["tip_ci"]])
    assert len(tips) == 485
    assert sum(bool(row["tip_point"]) for row in probe) == 358
    assert (f"{np.percentile(tips, 5):.2f}", f"{np.percentile(tips, 95):.2f}") == ("3.17", "5.48")

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

    # Step 6: a cross-fitted, stacked fit with boosted trees for Q, logistic g and pi, and the
    # known range of the score as q_bounds.
    method = namespace["method"]
    fit_summary = fitted.summary()
    assert "stacked CV-TMLE (Levy): nuisances cross-fitted over 5 folds" in fit_summary
    assert "TMLE (in-sample nuisances)" not in fit_summary
    assert method.cross_fitting == CrossFitting(n_folds=5)
    assert method.targeting.q_bounds == (0.0, 1.0)
    assert isinstance(method.models.outcome_learner, HistGradientBoostingRegressor)
    assert method.models.outcome_learner.early_stopping is True
    assert isinstance(method.models.treatment_learner, LogisticRegression)
    assert isinstance(method.models.missingness_learner, LogisticRegression)

    # The protocol step prints the record, and the fit carries its digest.  The reading names
    # the three fields this page changes; the composite death rule of the program stays.
    program = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "assumption_rationale",
    }
    assert "composite strategy" in protocol.intercurrent_event_handling[2]
    assert "between 0 and 1" in protocol.outcome
    rationale = protocol.assumption_rationale
    assert rationale[0] == program.assumption_rationale[0]
    assert rationale[2:] == program.assumption_rationale[1:]
    assert "missingness at random" in rationale[1]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, fitted)

    # "This synthetic law draws no deaths", and the score lies in (0, 1).
    frame = namespace["frame"]
    assert set(frame.columns) == {
        "transition_score",
        "transition_navigation",
        "discharge_risk",
        "age",
        "prior_utilization",
        "responded",
    }
    observed = frame["transition_score"].dropna()
    assert float(observed.min()) > 0.0 and float(observed.max()) < 1.0
    # "About a quarter of patients never return the survey."
    assert 0.70 < float(frame["responded"].mean()) < 0.80
    assert int(frame["transition_score"].isna().sum()) == int((frame["responded"] == 0).sum())

    # Step 3: respondents have lower discharge risk (W1 -> Delta) and more prior utilization
    # (W3 -> Delta).
    by_response = namespace["by_response"]
    assert by_response.loc[1.0, "discharge_risk"] < 0.0 < by_response.loc[0.0, "discharge_risk"]
    assert by_response.loc[1.0, "prior_utilization"] > by_response.loc[0.0, "prior_utilization"]

    # The causal graph lists every arrow of the law and no other.  Each present arrow moves
    # its child by a visible amount, and each absent arrow moves it by nothing.  Columns are
    # (discharge_risk, age, prior_utilization).
    base = np.zeros((1, 3))

    def moves(function: Any, column: int) -> float:
        shifted = base.copy()
        shifted[0, column] = 1.0
        return abs(float(function(shifted)[0] - function(base)[0]))

    def outcome(w: Any) -> Any:
        return law.outcome_mean(w, 0.0, None)

    def response(w: Any) -> Any:
        return law.missingness(w, 0.0)

    assert [moves(law.propensity, column) > 0.05 for column in range(3)] == [True, True, False]
    assert moves(law.propensity, 2) == 0.0
    assert all(moves(outcome, column) > 0.02 for column in range(3))
    assert [moves(response, column) > 0.02 for column in range(3)] == [True, False, True]
    assert moves(response, 1) == 0.0
    # A -> Delta, and the effect falls as discharge risk rises (W1 modifies A -> Y).
    assert float(law.missingness(base, 1.0)[0]) > float(law.missingness(base, 0.0)[0]) + 0.05
    high_risk = np.array([[1.0, 0.0, 0.0]])

    def effect_at(w: Any) -> float:
        return float(law.outcome_mean(w, 1.0, None)[0] - law.outcome_mean(w, 0.0, None)[0])

    assert effect_at(high_risk) < effect_at(base) - 0.05
    # Step 6: "in this synthetic law the two logistic models have the true form" (their
    # logits are exactly linear in A and W), and the outcome surface is not linear.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(500, 3))
    ones = np.ones(len(w))

    def linear_residual(design: Any, target: Any) -> float:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(np.max(np.abs(design @ coefficients - target)))

    stacked = np.column_stack([np.r_[ones, ones], np.r_[ones * 0, ones], np.vstack([w, w])])
    means = np.r_[law.outcome_mean(w, 0.0, None), law.outcome_mean(w, 1.0, None)]
    assert linear_residual(stacked, means) > 0.05
    assert linear_residual(np.column_stack([ones, w]), logit(law.propensity(w))) < 1e-9
    response_logits = np.r_[logit(law.missingness(w, 0.0)), logit(law.missingness(w, 1.0))]
    assert linear_residual(stacked, response_logits) < 1e-9
    # "The adjustment set must contain W3", and Step 5 declares the response role.
    assert "prior_utilization" in namespace["study"].design.adjustment
    assert namespace["study"].design.missingness == "responded"

    # Step 6 and Step 7 relations on this draw.
    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    respondents = namespace["respondents"]
    assert covers(full, truth)
    assert not covers(complete_case, truth)
    assert complete_case.psi > truth
    assert len(respondents) == int(frame["responded"].sum())
    # The complete-case target for these respondents comes from the law's outcome means, and
    # the complete-case interval contains it.
    respondent_effect = namespace["respondent_effect"]
    w_resp = respondents[["discharge_risk", "age", "prior_utilization"]].to_numpy()
    assert respondent_effect == pytest.approx(
        float(np.mean(law.outcome_mean(w_resp, 1.0, None) - law.outcome_mean(w_resp, 0.0, None))),
        rel=1e-12,
    )
    assert covers(complete_case, respondent_effect)
    assert f"{namespace['distance']:.1f}" == "0.2"
    assert f"{cc_target:.3f}" == "0.173"

    # The support and nuisance reports print what the reading quotes.
    assert tuple(namespace["assessment"].attention) == ()
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "P(A=a,Delta=1|W)  0.0081" in assessment_output
    assert "max |clever covariate| (mean): 81.76" in assessment_output
    assert "nuisance fits look reasonable" in assessment_output
    assert "out-of-fold predictions" in assessment_output
    # The status table prints only the rows that ran or run on request.  The omitted-variable
    # rows do not run for a fit with a response mechanism and stay off the page.
    ledger = namespace["assessment"].to_frame().set_index(["surface", "check"])
    for row in OMITTED_VARIABLE_OPERATIONS:
        assert str(ledger.loc[("sensitivity", row), "status"]) == "unavailable"
        assert not re.search(rf"sensitivity\s+{row}\b", assessment_output)
    # The E-value standardizes by a missing-at-random plug-in of sd(Y), which fits one more
    # outcome regression, so the default assessment defers it.
    assert str(ledger.loc[("sensitivity", "evalue"), "status"]) == "deferred"
    assert re.search(r"sensitivity\s+evalue\s+deferred", assessment_output)
    assert "unavailable" not in assessment_output and "not_applicable" not in assessment_output
    for row in ("missingness", "tipping_gamma"):
        assert str(ledger.loc[("sensitivity", row), "status"]) == "completed"
        assert re.search(rf"sensitivity\s+{row}\s+completed", assessment_output)

    # Step 9: the estimate never reaches 0 under the navigation-arm tilt, and the interval does
    # at 4.921.  Nonzero witness: just before it, the lower limit is still positive.
    assert namespace["tipping_gamma"] is None
    ci_tipping = namespace["ci_tipping_gamma"]
    assert f"{ci_tipping:.3f}" == "4.921"
    arm_gamma = {0: 0.0, 1: -1.0}
    at_tip = missingness_tilt(fitted, [ci_tipping], estimands=["ate"], arm_gamma=arm_gamma)
    before = missingness_tilt(fitted, [ci_tipping - 0.5], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(at_tip["ci_lower"].iloc[0])) < 1e-4
    assert float(before["ci_lower"].iloc[0]) > 0.0
    # The mean shift divides each move by the mean non-response probability under navigation.
    # Nonzero witness: navigation raises response, so the other arm's column would differ.
    missingness = fitted.nuisance.bounded_missingness(0.01)
    navigation_column = fitted.nuisance.arms.index(1)
    nonresponse = namespace["nonresponse"]
    assert nonresponse == pytest.approx(float(np.mean(1.0 - missingness[:, navigation_column])))
    assert nonresponse < float(np.mean(1.0 - missingness[:, 1 - navigation_column])) - 0.05
    # Gamma = 30 has reached the floor: a larger tilt moves the estimate by nothing visible.
    floor = missingness_tilt(fitted, [30.0, 60.0], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(floor["psi"].iloc[0] - floor["psi"].iloc[1])) < 1e-6
    assert float(floor["psi"].iloc[0]) > 0.0
    assert (
        f"{namespace['mar_mean']:.3f}",
        f"{namespace['ci_mean_shift']:.3f}",
    ) == ("0.727", "0.688")
    assert namespace["ci_mean_shift"] < namespace["mar_mean"]

    # The curve's gamma=0 row is the MAR estimate, and its interval reuses the MAR standard error.
    curve = namespace["missingness_curve"]
    mar = curve[curve["gamma"] == 0.0]
    assert float(mar["psi"].iloc[0]) == pytest.approx(full.psi, rel=1e-9)
    assert (curve["std_err"] == full.std_error).all()
    assert curve["psi"].is_monotonic_decreasing
    assert (curve["psi"] > 0.0).all()
