"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CrossFitting
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
_STEP9 = "reviews/notebook-review/probes/sn-step9"
_MAR = "tests/canonical/tmle_mar/properties.csv"
_ARM = "tests/canonical/tmle_mar_arm_indexed_cvtmle/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "1.428": f"respondent-standardized effect of the strength-2 law (1.427936), by quadrature: "
    f"{_PROBE}/truth.log",
    "0.011": f"mean complete-case estimate minus 1.427936 (+0.0111), 500 draws: "
    f"{_PROBE}/summary.log",
    "0.01": f"the same mean shift (+0.0111), rounded: {_PROBE}/summary.log",
    "0.951": f"Step 6 coverage (2854 of 3000), seeds 2000-4999: {_STEP6}/summary.log",
    "0.99": f"Step 6 mean SE over empirical SD (0.989), seeds 2000-4999: {_STEP6}/summary.log",
    "0.946": f"Step 9 coverage of the library fit (2837 of 3000), seeds 1001-4000: "
    f"{_STEP9}/summary.log and {_STEP9}/summary-n4000b.log",
    "0.947": f"oracle TMLE coverage (0.9472), 20000 draws: {_STEP9}/oracle-4000.log",
    "0.94": f"5th percentile of the tipping gamma (0.9421), 500 draws: {_PROBE}/summary.log",
    "1.45": f"95th percentile of the tipping gamma (1.4534), 500 draws: {_PROBE}/summary.log",
    "5.4": f"5th percentile of the largest shift in score units (5.3835): {_PROBE}/summary.log",
    "7.6": f"95th percentile of the largest shift in score units (7.6110): {_PROBE}/summary.log",
    "-0.0039": f"bias_ci_lower of mar_robustness/mechanisms_correct (-0.003853): {_MAR}",
    "0.0023": f"bias_ci_upper of mar_robustness/mechanisms_correct (0.002294): {_MAR}",
    "0.0103": f"bias_margin of mar_robustness/mechanisms_correct (0.010316): {_MAR}",
    "-0.1229": f"bias_ci_lower of missingness_necessity/ate__complete_case_control "
    f"(-0.122939): {_MAR}",
    "-0.1172": f"bias_ci_upper of missingness_necessity/ate__complete_case_control "
    f"(-0.117198): {_MAR}",
    "0.0096": f"bias_margin of missingness_necessity/ate__complete_case_control (0.009634): {_MAR}",
    "0.9361": f"coverage_ci_lower of interval_calibration/l1_ate__learned_nuisances "
    f"(0.936086): {_ARM}",
    "0.9617": f"coverage_ci_upper of interval_calibration/l1_ate__learned_nuisances "
    f"(0.961718): {_ARM}",
    "0.9413": f"se_ratio_ci_lower of interval_calibration/l1_ate__learned_nuisances "
    f"(0.941324): {_ARM}",
    "1.0198": f"se_ratio_ci_upper of interval_calibration/l1_ate__learned_nuisances "
    f"(1.019810): {_ARM}",
}


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
    assert "worst transition score" not in summary
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
    mild_law = missing_outcome_dgp(strength=1.0)
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

    # "Respondents are a lower-risk group", and the naive respondent contrast overshoots.
    by_response = namespace["by_response"]
    assert by_response.loc[1.0, "discharge_risk"] < 0.0 < by_response.loc[0.0, "discharge_risk"]
    assert round(float(by_response.loc[1.0, "discharge_risk"]), 3) == -0.232
    assert by_response.loc[0.0, "discharge_risk"] > by_response.loc[1.0, "discharge_risk"] + 0.5
    truth = namespace["truth"]["ate"]
    assert namespace["unadjusted"] > truth

    # "Discharge risk and age confound the offer": each moves both the propensity and the
    # outcome mean of the strength-2 law.  Columns are (discharge_risk, age, prior_utilization).
    base = np.zeros((1, 3))
    for column in (0, 1):
        shifted = base.copy()
        shifted[0, column] = 1.0
        assert abs(float(law.propensity(shifted)[0] - law.propensity(base)[0])) > 0.05
        assert (
            abs(
                float(
                    law.outcome_mean(shifted, 0.0, None)[0] - law.outcome_mean(base, 0.0, None)[0]
                )
            )
            > 0.3
        )

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

    # "Both laws have the same population ATE", and the mild complete-case interval covers it.
    # The strength-2 interaction averages W1, whose mean is 0, so the laws share the ATE of 1.2;
    # the tolerance absorbs only the numerical integration of the stronger law's truth.
    mild_truth = namespace["mild_truth"]["ate"]
    assert mild_truth == pytest.approx(truth, abs=1e-4)
    mild = namespace["mild"]["ate"]
    assert covers(mild, mild_truth)

    # "On the mild law the effect is the same for every patient", "the linear outcome model is
    # also correct", and the laws differ in how strongly discharge risk drives response.
    # Nonzero witnesses: the strength-2 law fails the first two, so these checks can fail.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(500, 3))
    mild_effect = mild_law.outcome_mean(w, 1.0, None) - mild_law.outcome_mean(w, 0.0, None)
    strong_effect = law.outcome_mean(w, 1.0, None) - law.outcome_mean(w, 0.0, None)
    assert float(np.ptp(mild_effect)) < 1e-12
    assert float(np.ptp(strong_effect)) > 1.0

    def linear_residual(dgp: Any) -> float:
        arms = np.repeat([0.0, 1.0], len(w))
        stacked = np.vstack([w, w])
        target = np.concatenate([dgp.outcome_mean(w, 0.0, None), dgp.outcome_mean(w, 1.0, None)])
        design = np.column_stack([np.ones(len(arms)), arms, stacked])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(np.max(np.abs(design @ coefficients - target)))

    assert linear_residual(mild_law) < 1e-9
    assert linear_residual(law) > 0.1
    high_risk = np.array([[1.0, 0.0, 0.0]])
    for arm in (0.0, 1.0):
        assert (
            float(law.missingness(high_risk, arm)[0])
            < float(mild_law.missingness(high_risk, arm)[0]) - 0.05
        )

    # "Each interval contains its population value on this draw", the odds ratio lies above
    # the risk ratio, and the ratio intervals are asymmetric on the reported scale.
    box_points = namespace["box_points"]
    box_truth = namespace["box_truth"]
    for key, point in box_points.items():
        assert covers(point, box_truth[key])
    assert box_points["or"].psi > box_points["rr"].psi > 1.0
    for key in ("rr", "or"):
        point = box_points[key]
        assert point.ci[1] - point.psi > point.psi - point.ci[0]
    # "The outcome is common": the top-box share is well above a rare-outcome level in each arm.
    assert box_truth["ey0"] > 0.3
    assert box_truth["ey1"] > box_truth["ey0"]

    # Step 9: "The fold policy is the shipped default, stratify_by='none'", with five folds and
    # no declared q_bounds, and the fit is the stacked cross-fitted one.
    box_study = namespace["box_study"]
    box_method = namespace["box_method"]
    assert box_method.cross_fitting == CrossFitting(n_folds=5, stratify_by="none")
    assert CrossFitting(n_folds=5).stratify_by == "none"
    assert box_method.targeting.q_bounds is None
    assert (
        "stacked CV-TMLE"
        in box_study.identify(ATE(reference=0)).estimate(method=box_method).summary()
    )

    # "No row needs attention", the support report prints the joint mechanism row, and the
    # nuisance report prints the verdict the reading quotes.
    assert tuple(namespace["assessment"].attention) == ()
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "P(A=a,Delta=1|W)  0.0075" in assessment_output
    assert "max |clever covariate| (mean): 90.2" in assessment_output
    assert "nuisance fits look reasonable" in assessment_output
    # "The unavailable rows are not implemented for a fit with a response mechanism": the
    # omitted-variable rows and the E-value are unavailable, and no number for one appears in
    # the returned results.  The missingness rows returned.
    ledger = namespace["assessment"].to_frame().set_index(["surface", "check"])
    for row in (*OMITTED_VARIABLE_OPERATIONS, "evalue"):
        assert str(ledger.loc[("sensitivity", row), "status"]) == "unavailable"
        assert f"sensitivity  {row}" not in assessment_output.split("Not run", 1)[0]
    returned = assessment_output.split("Checks", 1)[0]
    assert "sensitivity  missingness" in returned
    assert "sensitivity  tipping_gamma" in returned

    # "tipping gamma is 1.306", and "at most 0.315 of the score range" is the maximum over the
    # fitted [0, 1] mean of the logit move, reached at a mean of 0.658.
    tipping = namespace["tipping_gamma"]
    assert f"{tipping:.3f}" == "1.306"
    grid = np.linspace(1e-6, 1 - 1e-6, 200_001)
    moves = grid - expit(np.log(grid / (1 - grid)) - tipping)
    assert namespace["largest_shift"] == pytest.approx(float(np.max(moves)), abs=1e-6)
    assert namespace["peak_mean"] == pytest.approx(float(grid[np.argmax(moves)]), abs=1e-4)
    # Witness: the mid-range move is strictly smaller, so the mid-range formula would fail.
    assert namespace["largest_shift"] > 0.5 - expit(-tipping) + 0.01
    # "That is 6.32 score units": the shift times the fitted scaling range.
    assert namespace["shift_units"] == pytest.approx(
        namespace["largest_shift"] * (scaler.upper - scaler.lower), rel=1e-12
    )
    assert f"{namespace['shift_units']:.2f}" == "6.32"

    # "The interval reaches 0 earlier": the use_ci tipping gamma lies nearer zero, and the
    # interval's lower limit reaches zero there.  Nonzero witness: slightly before it, the
    # lower limit is still positive, so a wrong gamma fails.
    ci_tipping = namespace["ci_tipping_gamma"]
    assert f"{ci_tipping:.3f}" == "1.130"
    assert 0.0 < ci_tipping < tipping
    arm_gamma = {0: 0.0, 1: -1.0}
    at_tip = missingness_tilt(fitted, [ci_tipping], estimands=["ate"], arm_gamma=arm_gamma)
    before = missingness_tilt(fitted, [ci_tipping - 0.05], estimands=["ate"], arm_gamma=arm_gamma)
    assert abs(float(at_tip["ci_lower"].iloc[0])) < 1e-4
    assert float(before["ci_lower"].iloc[0]) > 0.01
    assert f"{namespace['ci_shift_units']:.2f}" == "5.52"
    # The reference SD is that of the navigation-arm respondent scores, 1.66.
    navigation = frame[(frame["responded"] == 1) & (frame["transition_navigation"] == 1)]
    assert namespace["navigation_sd"] == pytest.approx(
        float(navigation["transition_score"].std()), rel=1e-12
    )
    assert f"{namespace['navigation_sd']:.2f}" == "1.66"
    ratios = (
        namespace["shift_units"] / namespace["navigation_sd"],
        namespace["ci_shift_units"] / namespace["navigation_sd"],
    )
    assert tuple(f"{ratio:.1f}" for ratio in ratios) == ("3.8", "3.3")

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
    # "The ATE moves less than 0.315 of the range": at the tipping gamma the ATE has moved by
    # its full MAR value, which is a smaller fraction of the range.
    assert full.psi / (scaler.upper - scaler.lower) < namespace["largest_shift"]
