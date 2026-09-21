"""The reviewed semantic callback for ``docs/examples/survey-nonresponse``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    DataError,
    MethodConfigurationError,
    NaturalCourseMean,
    PointTreatment,
    PopulationAttributableFraction,
)
from cleverly.datasets import missing_outcome_dgp, navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

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
    # Step 6 fits in sample: "TMLE (in-sample nuisances)" and no outer folds.
    assert "TMLE (in-sample nuisances)" in fitted.summary()
    assert "stacked CV-TMLE" not in fitted.summary()
    assert not namespace["method"].cross_fitting.enabled

    # The protocol step prints the record, and the fit carries its digest.
    # The reading names the fields this page changes in the program protocol.
    program = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "assumption_rationale",
    }
    # "It adds the response rule": every program rationale stays, and two entries are added.
    rationale = protocol.assumption_rationale
    assert rationale[0] == program.assumption_rationale[0]
    assert rationale[3:] == program.assumption_rationale[1:]
    assert len(rationale) == len(program.assumption_rationale) + 2
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, fitted)
    assert "scores death before day 30 as the worst transition score" in summary

    # "About a quarter of patients never return the 30-day survey."
    frame = namespace["frame"]
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
    law = missing_outcome_dgp(strength=2.0)
    mild_law = missing_outcome_dgp(strength=1.0)
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

    # "CausalStudy raises DataError for a missing outcome that carries no response indicator."
    covariates = tuple(namespace["covariates"])
    with pytest.raises(DataError, match="missing value"):
        CausalStudy(
            frame.drop(columns=["responded"]),
            design=PointTreatment(
                outcome="transition_score",
                treatment="transition_navigation",
                adjustment=covariates,
            ),
        )

    # Step 6: the reading quotes the scaling line, and the scaler holds that range.
    estimate_output = stored_output(NOTEBOOK, "estimate")
    assert "outcome scaled from [-6.56, 13.49] to [0, 1]" in estimate_output
    scaler = fitted.nuisance.scaler
    assert (round(scaler.lower, 2), round(scaler.upper, 2)) == (-6.56, 13.49)

    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    assert complete_case.psi > truth
    assert not covers(complete_case, truth)
    assert covers(full, truth)
    assert len(namespace["respondents"]) == int(frame["responded"].sum())

    # "It targets the respondents' average effect", 1.409, which the complete-case interval
    # contains.  Independent witness: the law's effect is 1.2 - 0.9 * W1, so its average over
    # respondents follows from the Step 3 respondent mean of discharge risk.
    respondent_effect = namespace["respondent_effect"]
    assert respondent_effect == pytest.approx(
        1.2 - 0.9 * float(by_response.loc[1.0, "discharge_risk"]), abs=1e-9
    )
    assert respondent_effect > truth + 0.1
    assert covers(complete_case, respondent_effect)
    # "That shift is most of the gap", and "about one standard error above 1.409".
    assert respondent_effect - truth > complete_case.psi - respondent_effect
    distance = (complete_case.psi - respondent_effect) / complete_case.std_error
    assert 0.5 < distance < 1.5

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

    # "Each interval contains its population value", the odds ratio lies above the risk ratio,
    # and the ratio intervals are asymmetric on the reported scale.
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

    # The refusal is the attributable fraction's missing-outcome refusal, not another error,
    # and "identify raises the refusal before any method or learner is chosen".
    assert "does not yet support PointTreatment(missingness=...)" in namespace["refusal"]
    box_study = namespace["box_study"]
    with pytest.raises(CapabilityError, match=r"does not yet support PointTreatment\(missingness"):
        box_study.identify(PopulationAttributableFraction(reference=0))

    # Step 6: "Step 4 recorded a standardized score, which has no such support, so the fit stays
    # in sample". The refusal names q_bounds=None, and it comes from the arm-indexed contract.
    study = namespace["study"]
    method = namespace["method"]
    box_method = namespace["box_method"]
    assert method.cross_fitting == CrossFitting(enabled=False)
    with pytest.raises(CapabilityError, match="q_bounds=None"):
        effect.estimate(method=replace(method, cross_fitting=CrossFitting(n_folds=5)))

    # Step 9: "The fold policy is the shipped default, `stratify_by='none'`", so the binary fit
    # names no policy and gets that one.
    assert box_method.cross_fitting == CrossFitting(n_folds=5, stratify_by="none")
    assert CrossFitting(n_folds=5).stratify_by == "none"
    # "cleverly refuses stratify_by='treatment' on any fit", and it refuses at construction,
    # before a study or a learner exists. The binary outcome is the nonzero witness that the
    # q_bounds rule refuses nothing here: the same five folds fit without a declared support.
    with pytest.raises(MethodConfigurationError, match="balances the outer folds"):
        CrossFitting(n_folds=5, stratify_by="treatment")
    assert box_method.targeting.q_bounds is None
    assert (
        "stacked CV-TMLE"
        in box_study.identify(ATE(reference=0)).estimate(method=box_method).summary()
    )

    # Step 10: "box_method" fits the natural-course mean under the cross-fitted contract, and
    # "method" is refused because the in-sample contract needs q_bounds for a continuous outcome.
    natural = box_study.identify(NaturalCourseMean()).estimate(method=box_method)
    assert list(natural.estimates) == ["ey_obs"]
    assert "stacked CV-TMLE" in natural.summary()
    with pytest.raises(CapabilityError, match="continuous outcomes require fixed"):
        study.identify(NaturalCourseMean()).estimate(method=method)

    # "No row needs attention", the support report prints the joint mechanism row, and the
    # nuisance report prints the verdict the reading quotes.
    assert tuple(namespace["assessment"].attention) == ()
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "P(A=a,Delta=1|W)  0.0075" in assessment_output
    assert "max |clever covariate| (mean): 90.2" in assessment_output
    assert "nuisance fits look reasonable" in assessment_output
    for row in ("omitted_confounding", "robustness_value", "elements", "contour", "evalue"):
        assert f"sensitivity  {row}" in assessment_output

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
    assert namespace["largest_shift"] * (scaler.upper - scaler.lower) == pytest.approx(
        6.32, abs=0.005
    )

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
