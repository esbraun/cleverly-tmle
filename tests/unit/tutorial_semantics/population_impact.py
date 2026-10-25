"""The reviewed semantic callback for ``docs/examples/population-impact``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`, and the slow tier passes it to :func:`check_refits` as
well.  Each assertion witnesses one claim the prose makes about the law or about this draw.  A
relation on this draw certifies no method; the registered studies do that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import CapabilityError, CrossFitting, NaturalCourseMean
from cleverly.datasets import navigation_protocol, population_impact_truth
from cleverly.datasets import population_impact as law
from cleverly.sensitivity import ConfounderStrengthGrid, missingness_tilt, simulated_confounding
from cleverly.validation import BernoulliAdjustmentOutcome
from cleverly.validation._readings import subset_reference_sd
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "population-impact.ipynb"

#: The support row each fit of the page resolves to.
POPULATION_ROW = "T26.binary"
DIRECT_ROW = "T30"


def _moves(function: Any, column: int) -> float:
    """How far one standard deviation of covariate ``column`` moves ``function`` at the mean."""
    base = np.zeros((1, len(law.COVARIATES)))
    shifted = base.copy()
    shifted[0, column] = 1.0
    return abs(float(function(shifted)[0] - function(base)[0]))


def _check_law(namespace: dict[str, Any]) -> None:
    """The data step, the graph, and the truths the readings quote."""
    truth = namespace["truth"]
    assert truth == population_impact_truth()
    frame = namespace["frame"]
    assert list(frame.columns) == [
        "top_box",
        "transition_navigation",
        *law.COVARIATES,
        "pcp_visit_7d",
        "responded",
    ]
    assert frame["top_box"].isna().equals(frame["responded"] == 0)
    # "A quarter of patients never answer."
    assert 0.70 < float(frame["responded"].mean()) < 0.80

    # The graph: each drawn arrow moves its child and each absent arrow moves it by nothing.
    # Columns are (discharge_risk, age, prior_utilization).
    assert all(_moves(law.propensity, column) > 0.04 for column in range(3))
    visit = lambda w: law.visit_probability(w, 0.0)  # noqa: E731
    assert [_moves(visit, column) > 0.04 for column in range(3)] == [True, True, False]
    outcome = lambda w: law.outcome_probability(w, 0.0, 0.0)  # noqa: E731
    assert all(_moves(outcome, column) > 0.04 for column in range(3))
    response = lambda w: law.response_probability(w, 0.0)  # noqa: E731
    assert [_moves(response, column) > 0.02 for column in range(3)] == [True, True, False]
    base = np.zeros((1, 3))
    assert law.visit_probability(base, 1.0)[0] > law.visit_probability(base, 0.0)[0] + 0.2
    assert law.response_probability(base, 1.0)[0] > law.response_probability(base, 0.0)[0] + 0.03
    assert law.outcome_probability(base, 0.0, 1.0)[0] > law.outcome_probability(base, 0.0, 0.0)[0]
    assert law.outcome_probability(base, 1.0, 0.0)[0] > law.outcome_probability(base, 0.0, 0.0)[0]
    # "Discharge teams favor patients with a high discharge risk", "high-risk patients answer
    # less often", and the offer "does more" for high-risk patients.
    high = np.array([[1.0, 0.0, 0.0]])
    assert law.propensity(high)[0] > law.propensity(base)[0]
    assert law.response_probability(high, 0.0)[0] < law.response_probability(base, 0.0)[0]

    def effect(w: Any) -> float:
        return float(law.arm_mean(w, 1.0)[0] - law.arm_mean(w, 0.0)[0])

    assert effect(high) > effect(base) + 0.02
    # The truths the failure mode quotes.
    assert truth["att"] > truth["atc"] + 0.04
    # "In these simulated data the two logistic models have the true form": the treatment and
    # response logits are linear in W.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(400, 3))
    design = np.column_stack([np.ones(len(w)), w])
    for probability in (law.propensity(w), law.response_probability(w, 1.0)):
        target = np.log(probability / (1.0 - probability))
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        assert float(np.max(np.abs(design @ coefficients - target))) < 1e-9


def _check_comparison(namespace: dict[str, Any]) -> None:
    """The simple comparison: offered patients are higher risk and respondents overstate today."""
    truth = namespace["truth"]
    rates = namespace["rates"]
    answered = namespace["answered"]
    by_offer = namespace["frame"].groupby("transition_navigation")["discharge_risk"].mean()
    assert by_offer[1.0] > 0.0 > by_offer[0.0]
    assert float(answered["top_box"].mean()) > truth["ey_obs"] + 0.01
    assert float(rates[1.0] - rates[0.0]) < truth["ate"] - 0.02


def _check_setup(namespace: dict[str, Any]) -> None:
    """The study description, the design, and the conditions the readings name."""
    protocol = namespace["protocol"]
    program = navigation_protocol()
    assert changed_fields(protocol, program) == {
        "target_population",
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert "not top box" in protocol.intercurrent_event_handling[2]
    assert protocol.intercurrent_event_handling[:2] == program.intercurrent_event_handling[:2]
    assert "missingness at random" in protocol.assumption_rationale[1]
    fits = [
        namespace[name]
        for name in ("gain_fit", "current_fit", "achieved_fit", "ate_fit", "rr_fit", "direct_fit")
    ]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, *fits)

    study = namespace["study"]
    assert "pcp_visit_7d" not in study.design.adjustment
    assert study.design.missingness == "responded"
    names = " ".join(namespace["gain_question"].identification.assumptions).lower()
    assert "no unmeasured confounding" in names and "missingness at random" in names
    # "The rate under current practice needs only the two survey conditions."
    current = " ".join(study.identify(NaturalCourseMean()).identification.assumptions).lower()
    assert "missingness at random" in current
    assert "unmeasured confounding" not in current and "consistency" not in current
    printed = stored_output(NOTEBOOK, "identify")
    for condition in ("well-defined treatment", "no interference", "no hidden confounding"):
        assert condition in printed

    method = namespace["method"]
    assert method.cross_fitting == CrossFitting(n_folds=10)
    outcome = method.models.outcome_learner
    assert isinstance(outcome, Pipeline)
    products = outcome.steps[0][1]
    assert isinstance(products, PolynomialFeatures)
    assert products.degree == 2 and products.interaction_only
    assert isinstance(method.models.treatment_learner, LogisticRegression)
    assert isinstance(method.models.missingness_learner, LogisticRegression)
    assert namespace["in_sample"].cross_fitting == CrossFitting(enabled=False)
    assert isinstance(namespace["in_sample"].models.intermediate_learner, LogisticRegression)


def _check_estimates(namespace: dict[str, Any]) -> None:
    """Labels, the intervals that contain their truths, and the failure-mode relations."""
    truth = namespace["truth"]
    gain = namespace["gain_fit"]["par"]
    current = namespace["current_fit"]["ey_obs"]
    achieved = namespace["achieved_fit"]["par"]
    gain_share = namespace["gain_share_fit"]["paf"]
    achieved_share = namespace["achieved_share_fit"]["paf"]
    ate = namespace["ate_fit"]["ate"]
    rr = namespace["rr_fit"]["rr"]
    odds = namespace["or_fit"]["or"]
    direct = namespace["direct_fit"]["ate"]
    for estimate in (gain, current, achieved, gain_share, achieved_share, ate, rr, odds):
        assert (estimate.support.label, estimate.support.row) == ("supported", POPULATION_ROW)
    assert (direct.support.label, direct.support.row) == ("supported", DIRECT_ROW)
    # "Each interval contains its true value."  The gain and its share are minus the PAR and PAF
    # at reference 1.
    assert covers(current, truth["ey_obs"])
    assert covers(gain, truth["par[reference=1]"])
    assert covers(gain_share, truth["paf[reference=1]"])
    assert covers(achieved, truth["par[reference=0]"])
    assert covers(achieved_share, truth["paf[reference=0]"])
    assert gain.psi < 0.0 < achieved.psi
    # The failure mode: the ATE splits into the two parts to the printed digits, it is larger
    # than the gain, and its interval misses the true gain.
    assert abs(namespace["parts"] - ate.psi) < 5e-5
    assert f"{namespace['parts']:.4f}" == f"{ate.psi:.4f}"
    assert ate.psi > -gain.psi
    assert not covers(ate, -truth["par[reference=1]"])
    assert covers(ate, truth["ate"])
    assert f"{namespace['not_offered']:.3f}" == "0.563"
    # The ratios: both above 1, the odds ratio further from it, and each interval holds its truth.
    assert covers(rr, truth["rr"]) and covers(odds, truth["or"])
    assert odds.psi > rr.psi > 1.0
    # The direct effect: positive, its interval holds the truth, and it is wider than the ATE's.
    assert direct.ci[0] > 0.0
    assert covers(direct, truth["cde_ate[visit=1]"])
    assert direct.ci[1] - direct.ci[0] > ate.ci[1] - ate.ci[0]
    # The visit has positivity in both arms of the law.
    frame = namespace["frame"]
    for arm in (0.0, 1.0):
        share = frame.loc[frame["transition_navigation"] == arm, "pcp_visit_7d"].mean()
        assert 0.2 < share < 0.8
    # The E-value reads the risk ratio of the page.
    evalue = namespace["evalue"]
    assert evalue.risk_ratio == pytest.approx(rr.psi)
    assert (f"{evalue.point:.2f}", f"{evalue.limit:.2f}") == ("2.09", "1.89")
    assert not evalue.approximate


def _check_assessment(namespace: dict[str, Any]) -> None:
    """Nothing needs attention, nothing is capped, and the direct-effect fit is thinner."""
    assert tuple(namespace["assessment"].attention) == ()
    assert tuple(namespace["direct_assessment"].attention) == ()
    support = namespace["support"]
    joint = namespace["joint"]
    triple = namespace["triple"]
    assert joint["clipped_fraction"] == 0.0 and triple["clipped_fraction"] == 0.0
    assert support.truncated["fraction"] == 0.0
    assert triple["min"] < 0.5 * joint["min"]
    assert round(support.bounds[0], 4) == 0.0095 and support.nuisance_bound == 0.01
    for name in ("propensity", "missingness", "outcome"):
        assert abs(namespace["models"][name]["calibration_slope"] - 1.0) < 0.1
    assert namespace["scores"].passed
    assert "nuisance fits look reasonable" in namespace["nuisance"].summary()


def _check_tilt(namespace: dict[str, Any]) -> None:
    """The gain falls with gamma, reaches zero at the printed gamma, and one arm alone never does."""
    gain_fit = namespace["gain_fit"]
    curve = namespace["curve"]
    assert float(curve["gain"].iloc[0]) == pytest.approx(-gain_fit["par"].psi, rel=1e-9)
    assert curve["gain"].is_monotonic_decreasing
    tip = namespace["both"]
    assert f"{tip:.3f}" == "1.950" and f"{np.exp(tip):.2f}" == "7.03"
    direction = namespace["direction"]
    near = missingness_tilt(gain_fit, (tip - 0.1, tip, tip + 0.1), arm_gamma=direction).to_frame()
    values = near["psi"].to_numpy()
    assert values[0] < 0.0 < values[2] and abs(values[1]) < 1e-4
    assert namespace["offered_only"] is None


def check(namespace: dict[str, Any]) -> None:
    """Witness the law, the comparison, the setup, the estimates, the checks, and the tilt."""
    _check_law(namespace)
    _check_comparison(namespace)
    _check_setup(namespace)
    _check_estimates(namespace)
    _check_assessment(namespace)
    _check_tilt(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refit cells: the confounder bound, the refutation, and the generated outcomes."""
    truth = namespace["truth"]
    today = namespace["today"]
    everyone = namespace["everyone_fit"]["ey1"]
    assert (everyone.support.label, everyone.support.row) == ("supported", POPULATION_ROW)
    benchmarks = namespace["benchmarks"]
    risk_record, risk_bound = benchmarks["discharge_risk"]
    # "A confounder as strong as discharge_risk, the strongest recorded one, would leave the gain
    # positive"; every recorded outcome-side strength is 0.041 or less.
    assert risk_bound.lower - today > 0.0
    for record, _ in benchmarks.values():
        assert record.cf_y <= risk_record.cf_y + 1e-12 and record.cf_d <= risk_record.cf_d
    assert f"{risk_record.cf_y:.3f}" == "0.041"
    # The tipping strength sits below the weight-side strength of discharge_risk, and the bound
    # there reaches today's rate.  Nonzero witness: at half that strength the gain survives.
    tipping = namespace["tipping_cf_d"]
    assert f"{tipping:.3f}" == "0.016" and tipping < risk_record.cf_d
    sensitivity = namespace["sensitivity"]
    assert sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping).lower == pytest.approx(
        today, abs=1e-6
    )
    assert sensitivity.omitted_confounding(cf_y=1.0, cf_d=tipping / 2).lower > today
    assert truth["ey1"] > truth["ey_obs"]

    # The refutation: descriptive, the placebo near zero, the noise column barely moving it.
    ate = namespace["ate_fit"]["ate"]
    refutation = namespace["refutation"]
    tests = {test.name: test for test in refutation.tests}
    assert set(tests) == {"placebo", "subset", "random_common_cause"}
    for test in tests.values():
        assert test.passed is None and len(test.values) == 20
    assert set(refutation.to_frame()["verdict"]) == {"descriptive"}
    assert abs(float(np.mean(tests["placebo"].values))) < 0.1 * ate.psi
    shift = np.asarray(tests["random_common_cause"].values) - ate.psi
    assert abs(float(shift.mean())) < 0.2 * ate.std_error
    # The 70% refits spread about as much as the standard error implies for such subsamples.
    n = len(namespace["frame"])
    reference = subset_reference_sd(ate.std_error, n, int(0.7 * n))
    assert f"{reference:.4f}" == "0.0126"
    assert 0.5 < float(np.std(tests["subset"].values, ddof=1)) / reference < 2.0

    # The generated outcomes run at level L1, near their true values.
    independent, adjusted = namespace["independent"], namespace["adjusted"]
    assert isinstance(independent, BernoulliAdjustmentOutcome)
    assert (independent.effect, independent.adjustment_scale) == (0.0, 0.0)
    assert isinstance(adjusted, BernoulliAdjustmentOutcome) and adjusted.effect != 0.0
    for name, check_result in namespace["generated_tests"].items():
        assert check_result.rung == "L1" and check_result.n_replicates == 10, name
        assert abs(check_result.mean - check_result.declared_effect) < 0.02, name
    dummy = namespace["generated_tests"]["dummy_outcome"].readings[0]
    assert 0.0 < dummy.lower < 0.01

    # The replay the page skips: the response reads the offer, so a cell that moves the offer is
    # refused and names F12.  Nonzero witness: the law's response differs by arm.
    base = np.zeros((1, 3))
    assert law.response_probability(base, 1.0)[0] > law.response_probability(base, 0.0)[0]
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
    with pytest.raises(CapabilityError, match="observation_reads") as excinfo:
        simulated_confounding(namespace["ate_fit"], grid=grid)
    assert "F12" in str(excinfo.value)
