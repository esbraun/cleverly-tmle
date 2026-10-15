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
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting
from cleverly.datasets import missing_outcome_dgp, navigation_protocol
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

_PROBE = "reviews/notebook-review/probes/survey-nonresponse-final"
_STEP6 = "reviews/notebook-review/probes/sn-step6"
_MAR = "tests/canonical/tmle_mar/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "1.428": f"respondent-standardized effect of the strength-2 law (1.427936), by quadrature: "
    f"{_PROBE}/truth.log",
    "0.011": f"mean complete-case estimate minus 1.427936 (+0.0111), 500 draws: "
    f"{_PROBE}/summary.log",
    "0.0028": f"Monte Carlo SE of the mean complete-case estimate, 500 draws: {_PROBE}/summary.log",
    "0.947": f"pooled Step 6 coverage (3314 of 3500), seeds 1000-4999: {_STEP6}/summary.log",
    "0.939": f"lower Wald limit of the pooled Step 6 coverage: {_STEP6}/summary.log",
    "0.954": f"upper Wald limit of the pooled Step 6 coverage: {_STEP6}/summary.log",
    "0.98": f"pooled Step 6 mean SE over empirical SD (0.981): {_STEP6}/summary.log",
    "0.94": f"5th percentile of the tipping gamma (0.9421), 500 draws: {_PROBE}/summary.log",
    "1.45": f"95th percentile of the tipping gamma (1.4534), 500 draws: {_PROBE}/summary.log",
    "-0.0039": f"bias_ci_lower of mar_robustness/mechanisms_correct (-0.003853): {_MAR}",
    "0.0023": f"bias_ci_upper of mar_robustness/mechanisms_correct (0.002294): {_MAR}",
    "0.0103": f"bias_margin of mar_robustness/mechanisms_correct (0.010316): {_MAR}",
    "-0.1229": f"bias_ci_lower of missingness_necessity/ate__complete_case_control "
    f"(-0.122939): {_MAR}",
    "-0.1172": f"bias_ci_upper of missingness_necessity/ate__complete_case_control "
    f"(-0.117198): {_MAR}",
    "0.0096": f"bias_margin of missingness_necessity/ate__complete_case_control (0.009634): {_MAR}",
}


def check(namespace: dict[str, Any]) -> None:
    """Witness the identification, graph, comparison, diagnostic, and sensitivity claims.

    The committed review probes and registered properties also back each quoted number.
    """
    # The 500-draw probe: the Step 6 interval covered on 460 draws, the complete-case interval
    # covered the ATE on 18, and the complete-case estimate averaged 0.011 above 1.428.
    with (EXAMPLES.parents[1] / _PROBE / "sweep.csv").open(encoding="utf-8", newline="") as stream:
        probe_rows = list(csv.DictReader(stream))
    assert len(probe_rows) == 500
    assert sum(row["full_covers"] == "True" for row in probe_rows) == 460
    assert sum(row["cc_covers_truth"] == "True" for row in probe_rows) == 18
    estimates = np.array([float(row["cc_psi"]) for row in probe_rows])
    assert f"{estimates.mean() - 1.427936:.3f}" == "0.011"
    assert f"{estimates.std(ddof=1) / np.sqrt(len(estimates)):.4f}" == "0.0028"
    assert sum(row["cc_covers_population_target"] == "True" for row in probe_rows) == 470
    tips = np.array([float(row["tip_point"]) for row in probe_rows])
    assert tuple(f"{q:.2f}" for q in np.percentile(tips, [5, 95])) == ("0.94", "1.45")

    # The Step 6 coverage probe: 3,314 of 3,500 draws (0.947, Wald 0.939 to 0.954), an SE
    # ratio of 0.98, the first block of 460, and later 500-draw blocks of 468 to 484.
    step6 = []
    for name in ("probe-s1000.csv", "probe-s2000.csv"):
        path = EXAMPLES.parents[1] / _STEP6 / name
        with path.open(encoding="utf-8", newline="") as stream:
            step6 += list(csv.DictReader(stream))
    assert len(step6) == 3500
    covered = np.array([row["lib_covers"] == "True" for row in step6])
    rate = covered.mean()
    half = 1.959964 * np.sqrt(rate * (1 - rate) / len(covered))
    assert (int(covered.sum()), f"{rate:.3f}") == (3314, "0.947")
    assert (f"{rate - half:.3f}", f"{rate + half:.3f}") == ("0.939", "0.954")
    psi = np.array([float(row["lib_psi"]) for row in step6])
    se = np.array([float(row["lib_se"]) for row in step6])
    assert f"{se.mean() / psi.std(ddof=1):.2f}" == "0.98"
    blocks = [int(covered[start : start + 500].sum()) for start in range(0, 3500, 500)]
    assert blocks[0] == 460
    assert (min(blocks[1:]), max(blocks[1:])) == (468, 484)

    # The complete-case target of the strength-2 law, 1.2 - 0.9 E[W1 | Delta = 1] = 1.428,
    # by Gauss-Hermite quadrature over the three independent standard-normal covariates.
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    mass = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), -1), -1)
    grid_law = missing_outcome_dgp(strength=2.0)
    g = grid_law.propensity(grid)
    respond = g * grid_law.missingness(grid, 1.0) + (1 - g) * grid_law.missingness(grid, 0.0)
    respond = respond * mass.reshape(-1)
    effect = grid_law.outcome_mean(grid, 1.0, None) - grid_law.outcome_mean(grid, 0.0, None)
    assert f"{float((effect * respond).sum() / respond.sum()):.3f}" == "1.428"

    # The registered cells of ordinary missing-outcome TMLE that the trust table quotes.
    with (EXAMPLES.parents[1] / _MAR).open(encoding="utf-8", newline="") as stream:
        cells = {(row["property"], row["cell"]): row for row in csv.DictReader(stream)}
    for key, quoted in {
        ("mar_robustness", "mechanisms_correct"): ("-0.0039", "0.0023", "0.0103"),
        ("missingness_necessity", "ate__complete_case_control"): ("-0.1229", "-0.1172", "0.0096"),
    }.items():
        row = cells[key]
        values = (row["bias_ci_lower"], row["bias_ci_upper"], row["bias_margin"])
        assert tuple(f"{float(value):.4f}" for value in values) == quoted

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
    # Step 6 fits in sample: "TMLE (in-sample nuisances)" and no outer folds, with linear and
    # logistic learners of few coefficients.
    method = namespace["method"]
    assert "TMLE (in-sample nuisances)" in fitted.summary()
    assert "stacked CV-TMLE" not in fitted.summary()
    assert method.cross_fitting == CrossFitting(enabled=False)
    assert isinstance(method.models.outcome_learner, LinearRegression)
    assert isinstance(method.models.treatment_learner, LogisticRegression)
    assert isinstance(method.models.missingness_learner, LogisticRegression)

    # The protocol step prints the record, and the fit carries its digest.
    # The reading names the four fields this page changes in the program protocol.
    program = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # The hypothetical strategy replaces the program's composite rule for death; the other two
    # intercurrent-event entries stay.  Nonzero witness: the program carries the composite rule.
    events = protocol.intercurrent_event_handling
    assert events[:2] == program.intercurrent_event_handling[:2]
    assert len(events) == 3
    assert "hypothetical strategy" in events[2]
    assert "composite strategy" in program.intercurrent_event_handling[2]
    assert "worst transition score" not in "\n".join(protocol.summary_lines())
    assert "no documented range" in protocol.outcome
    # "It adds missingness at random for response and for death": every program rationale
    # stays, and two entries are added.
    rationale = protocol.assumption_rationale
    assert rationale[0] == program.assumption_rationale[0]
    assert rationale[3:] == program.assumption_rationale[1:]
    assert len(rationale) == len(program.assumption_rationale) + 2
    assert "missingness at random" in rationale[1]
    assert "death before day 30" in rationale[2]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, fitted)

    # "This synthetic law draws no deaths", and the score has Gaussian noise with no range:
    # the drawn frame has no death column, and the law is a Gaussian family.
    frame = namespace["frame"]
    law = missing_outcome_dgp(strength=2.0)
    assert set(frame.columns) == {
        "transition_score",
        "transition_navigation",
        "discharge_risk",
        "age",
        "prior_utilization",
        "responded",
    }
    assert law.family == "gaussian"
    assert law.noise_scale > 0.0

    # "About a quarter of patients never return the 30-day survey."
    assert 0.70 < float(frame["responded"].mean()) < 0.80
    assert int(frame["transition_score"].isna().sum()) == int((frame["responded"] == 0).sum())

    # Step 3: respondents have lower discharge risk (W1 -> Delta).
    by_response = namespace["by_response"]
    assert by_response.loc[1.0, "discharge_risk"] < 0.0 < by_response.loc[0.0, "discharge_risk"]
    assert round(float(by_response.loc[1.0, "discharge_risk"]), 3) == -0.232
    assert by_response.loc[0.0, "discharge_risk"] > by_response.loc[1.0, "discharge_risk"] + 0.5
    truth = namespace["truth"]["ate"]
    # "Respondents also have more prior utilization", as the arrow W3 -> Delta predicts.
    assert by_response.loc[1.0, "prior_utilization"] > by_response.loc[0.0, "prior_utilization"]

    # The causal graph lists every arrow of the strength-2 law and no other.  Each present arrow
    # moves its child by a visible amount, and each absent arrow moves it by nothing.  Columns
    # are (discharge_risk, age, prior_utilization).
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
    assert all(moves(outcome, column) > 0.3 for column in range(3))
    assert [moves(response, column) > 0.02 for column in range(3)] == [True, False, True]
    assert moves(response, 1) == 0.0
    # A -> Delta, and the effect falls as discharge risk rises (W1 modifies A -> Y).
    assert float(law.missingness(base, 1.0)[0]) > float(law.missingness(base, 0.0)[0]) + 0.05
    high_risk = np.array([[1.0, 0.0, 0.0]])

    def effect_at(w: Any) -> float:
        return float(law.outcome_mean(w, 1.0, None)[0] - law.outcome_mean(w, 0.0, None)[0])

    assert effect_at(high_risk) < effect_at(base) - 0.5
    # Step 6: Q is wrong (curvature and the A-by-W1 interaction leave a main-effects linear
    # residual), while g and pi have the law's form (their logits are exactly linear in A and W).
    rng = np.random.default_rng(0)
    w = rng.normal(size=(500, 3))
    ones = np.ones(len(w))

    def linear_residual(design: Any, target: Any) -> float:
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(np.max(np.abs(design @ coefficients - target)))

    stacked = np.column_stack([np.r_[ones, ones], np.r_[ones * 0, ones], np.vstack([w, w])])
    means = np.r_[law.outcome_mean(w, 0.0, None), law.outcome_mean(w, 1.0, None)]
    assert linear_residual(stacked, means) > 0.1
    assert linear_residual(np.column_stack([ones, w]), logit(law.propensity(w))) < 1e-9
    response_logits = np.r_[logit(law.missingness(w, 0.0)), logit(law.missingness(w, 1.0))]
    assert linear_residual(stacked, response_logits) < 1e-9
    # "The adjustment set must contain W3."
    assert "prior_utilization" in namespace["study"].design.adjustment

    # Step 5 declares the response indicator as a design role.
    assert namespace["study"].design.missingness == "responded"

    # Step 6: the reading quotes the scaling line, and the scaler holds that range.
    estimate_output = stored_output(NOTEBOOK, "estimate")
    assert "outcome scaled from [-6.56, 13.49] to [0, 1]" in estimate_output
    scaler = fitted.nuisance.scaler
    assert (round(scaler.lower, 2), round(scaler.upper, 2)) == (-6.56, 13.49)

    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    respondents = namespace["respondents"]
    assert complete_case.psi > truth
    assert not covers(complete_case, truth)
    assert covers(full, truth)
    assert len(respondents) == int(frame["responded"].sum())

    # The complete-case fit targets the effect standardized to the respondents' covariates,
    # 1.409 for this draw.  Independent witness: the law's effect is 1.2 - 0.9 * W1, so its
    # average over respondents follows from the Step 3 respondent mean of discharge risk.
    respondent_effect = namespace["respondent_effect"]
    assert respondent_effect == pytest.approx(
        1.2 - 0.9 * float(by_response.loc[1.0, "discharge_risk"]), abs=1e-9
    )
    assert respondent_effect > truth + 0.1
    assert covers(complete_case, respondent_effect)
    # The population target, 1.428, lies between the population ATE and the complete-case
    # estimate of this draw.
    assert truth < 1.428 < complete_case.psi
    # "The move from 1.200 to 1.409 is most of the gap", and "1.0 standard error above".
    assert respondent_effect - truth > complete_case.psi - respondent_effect
    distance = namespace["distance"]
    assert distance == pytest.approx(
        (complete_case.psi - respondent_effect) / complete_case.std_error, rel=1e-12
    )
    assert f"{distance:.1f}" == "1.0"

    # The support report prints the joint mechanism row, and the nuisance report prints the
    # verdict and the in-sample calibration footnote that the reading quotes.
    assert tuple(namespace["assessment"].attention) == ()
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "P(A=a,Delta=1|W)  0.0075" in assessment_output
    assert "max |clever covariate| (mean): 90.2" in assessment_output
    assert "nuisance fits look reasonable" in assessment_output
    assert "in-sample slope of 1 by construction" in assessment_output
    # The status table prints only the rows that ran or run on request. The omitted-variable
    # rows and the E-value do not run for a fit with a response mechanism and stay off the page.
    # The two missingness rows that Step 9 reads are `completed`.
    ledger = namespace["assessment"].to_frame().set_index(["surface", "check"])
    for row in (*OMITTED_VARIABLE_OPERATIONS, "evalue"):
        assert str(ledger.loc[("sensitivity", row), "status"]) == "unavailable"
        assert not re.search(rf"sensitivity\s+{row}\b", assessment_output)
    assert "unavailable" not in assessment_output and "not_applicable" not in assessment_output
    for row in ("missingness", "tipping_gamma"):
        assert str(ledger.loc[("sensitivity", row), "status"]) == "completed"
        assert re.search(rf"sensitivity\s+{row}\s+completed", assessment_output)
    assert re.search(r"refute\s+deferred", assessment_output)

    # "tipping gamma is 1.306" and "1.130" for the interval, which lies nearer zero.  Nonzero
    # witness: slightly before the interval tipping gamma the lower limit is still positive.
    tipping = namespace["tipping_gamma"]
    ci_tipping = namespace["ci_tipping_gamma"]
    assert (f"{tipping:.3f}", f"{ci_tipping:.3f}") == ("1.306", "1.130")
    assert 0.0 < ci_tipping < tipping
    arm_gamma = {0: 0.0, 1: -1.0}
    at_tip = missingness_tilt(fitted, [ci_tipping], estimands=["ate"], arm_gamma=arm_gamma)
    before = missingness_tilt(fitted, [ci_tipping - 0.05], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(at_tip["ci_lower"].iloc[0])) < 1e-4
    assert float(before["ci_lower"].iloc[0]) > 0.01

    # The mean shift divides the move of the estimate by the mean non-response probability
    # under navigation.  At the point tipping gamma the estimate has moved by its whole MAR
    # value.  Nonzero witness: navigation raises response, so the other arm's column would give
    # a larger non-response mean and a smaller shift.
    missingness = fitted.nuisance.bounded_missingness(0.01)
    navigation_column = fitted.nuisance.arms.index(1)
    nonresponse = namespace["nonresponse"]
    assert nonresponse == pytest.approx(float(np.mean(1.0 - missingness[:, navigation_column])))
    assert nonresponse < float(np.mean(1.0 - missingness[:, 1 - navigation_column])) - 0.05
    assert namespace["mean_shift"] == pytest.approx(full.psi / nonresponse, rel=1e-4)
    assert (f"{namespace['mean_shift']:.2f}", f"{namespace['ci_mean_shift']:.2f}") == (
        "5.86",
        "5.21",
    )
    # The marginal SD of the navigation-arm respondent scores, 1.66, and the two ratios.
    navigation = frame[(frame["responded"] == 1) & (frame["transition_navigation"] == 1)]
    assert namespace["navigation_sd"] == pytest.approx(
        float(navigation["transition_score"].std()), rel=1e-12
    )
    assert f"{namespace['navigation_sd']:.2f}" == "1.66"
    ratios = (
        namespace["mean_shift"] / namespace["navigation_sd"],
        namespace["ci_mean_shift"] / namespace["navigation_sd"],
    )
    assert tuple(f"{ratio:.1f}" for ratio in ratios) == ("3.5", "3.1")
    # "Against the variation within one profile, the same shift is larger": the law's noise SD
    # lies below the marginal SD.
    assert law.noise_scale < namespace["navigation_sd"] - 0.3

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
