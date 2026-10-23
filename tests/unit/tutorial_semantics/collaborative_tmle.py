"""The reviewed semantic callback for ``docs/examples/collaborative-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import navigation_protocol
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import evalue_from_rr, omitted_variable_bounds
from tests.studies.oat_per_arm_laws import BINARY_ACTIVE
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    ROOT,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "collaborative-tmle.ipynb"

_PROBE = "reviews/notebook-review/probes/collaborative-tmle-oat/summary.log"
_TRUTH = "reviews/notebook-review/probes/collaborative-tmle-oat/truth.log"
_REFUTE = "reviews/notebook-review/probes/collaborative-tmle-refuters/probe.log"

# Integer counts the readings quote, which the decimal gate does not read:
# 1,857, 1,881 and 1,963 of 2,000 draws come from _PROBE.

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.015": "the gap between the two printed estimates, 0.16289 - 0.1483 = 0.0146",
    "+0.0006": f"mean error of both fits (+0.00055), 2,000 draws: {_PROBE}",
    "0.0358": f"ordinary TMLE empirical SD (0.03583), 2,000 draws: {_PROBE}",
    "0.0297": f"OAT C-TMLE empirical SD (0.02969), 2,000 draws: {_PROBE}",
    "0.93": f"ordinary TMLE mean SE over SD (0.9333), 2,000 draws: {_PROBE}",
    "0.98": f"OAT C-TMLE mean SE over SD (0.9759), 2,000 draws: {_PROBE}",
    "0.0134": f"median |OAT - TMLE| (0.0134), 2,000 draws: {_PROBE}",
    "0.0381": f"95th percentile of |OAT - TMLE| (0.0381), 2,000 draws: {_PROBE}",
    "0.0016": f"twice the Monte Carlo SE of the mean error (2 x 0.0358 / sqrt(2000)): {_PROBE}",
    "10.14": f"population nu2 of the full set (10.136): {_TRUTH}",
    "9.14": f"population nu2 of the full set, g clipped at the page bounds (9.142): {_TRUTH}",
    "4.12": f"population nu2 without staffing (4.122): {_TRUTH}",
    "0.193": f"ordinary TMLE mean share of g outside (0.1, 0.9) (0.1933): {_PROBE}",
    "0.0002": f"OAT C-TMLE mean share of g outside (0.1, 0.9) (0.0002): {_PROBE}",
    "1.01": f"both placebo ratios of mean reported SE over SD (1.012, 1.010), 400 refits: {_REFUTE}",
}


def _probe(pattern: str, path: str = _PROBE) -> list[float]:
    """The numbers of the one line of a committed probe log that ``pattern`` matches."""
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"{path} has no line matching {pattern!r}"
    return [float(value) for value in match.groups()]


def _check_committed_sources() -> None:
    """Each probe and study number of UNPRINTED_DECIMALS, read from its committed file."""
    for name, sd, ratio, covered in (
        ("tmle", 0.0358, 0.93, 1857),
        ("oat", 0.0297, 0.98, 1881),
    ):
        bias, empirical, mean_ratio, count = _probe(
            rf"^{name}\s+bias ([+-][\d.]+)\s+empirical sd ([\d.]+).*mean se / sd ([\d.]+)"
            r"\s+covered (\d+) of 2000$"
        )
        assert round(bias, 4) == 0.0006 and round(empirical, 4) == sd
        assert round(mean_ratio, 2) == ratio and count == covered
    assert round(_probe(r"^gcomp\s+bias [+-][\d.]+\s+empirical sd ([\d.]+)")[0], 4) == 0.0297
    assert _probe(r"^oat se below tmle se on (\d+) of 2000 draws$") == [1963.0]
    tails = (
        _probe(rf"^{name}\s+mean share of g outside \(0\.1, 0\.9\) ([\d.]+);")[0]
        for name in ("tmle", "oat")
    )
    assert [round(value, 4) for value in tails] == [0.1933, 0.0002]
    assert _probe(r"^\|oat - tmle\| median ([\d.]+), 95th percentile ([\d.]+)$") == [0.0134, 0.0381]
    # "about 0.0016": twice the Monte Carlo SE of the larger mean error's SD over 2,000 draws.
    assert round(2 * 0.03583 / np.sqrt(2000), 4) == 0.0016
    for label, value in (
        (r"full set", 10.14),
        (r"full set, clipped at the page bounds", 9.14),
        (r"without staffing", 4.12),
    ):
        assert round(_probe(rf"^{label}: nu2 = ([\d.]+)$", _TRUTH)[0], 2) == value
    # "about 1.01": the 400-refit placebo ratios of both fits, from the refutation probe.
    for name, ratio in (("tmle", 1.012), ("oat", 1.010)):
        (probed,) = _probe(rf"^{name} placebo t -[\d.]+ mean se / sd ([\d.]+);", _REFUTE)
        assert probed == ratio and round(probed, 2) == 1.01


def check(namespace: dict[str, Any]) -> None:
    """Both fits target one ATE; staffing makes only the ordinary g extreme."""
    _check_committed_sources()
    frame, truth = namespace["frame"], namespace["truth"]

    # "This is the law `binary_active` of the registered study ... at the same sample size":
    # the page's sampler reproduces the registered sampler row for row, and its quadrature
    # reproduces the registered truth.
    registered, registered_truth = BINARY_ACTIVE.sample(1_500, 44)
    page = frame[["baseline_readiness", "coordination_need", "navigator_staffing"]].to_numpy()
    np.testing.assert_array_equal(page, registered[["W1", "W2", "W3"]].to_numpy())
    np.testing.assert_array_equal(frame["transition_navigation"], registered["A"])
    np.testing.assert_array_equal(frame["transition_success"], registered["Y"])
    for key in ("ey0", "ey1", "ate"):
        assert abs(truth[key] - registered_truth[key]) < 1e-12

    # Step 3: the crude contrast overstates the effect, and both readiness and staffing differ
    # between arms while need does not.
    by_arm = namespace["by_arm"]
    assert namespace["unadjusted"] > truth["ate"] + 0.03
    gap = by_arm.loc[1.0] - by_arm.loc[0.0]
    assert gap["navigator_staffing"] > gap["baseline_readiness"] > 0.2
    assert abs(gap["coordination_need"]) < 0.1

    # Step 4: three fields change.
    base, protocol = navigation_protocol(), namespace["protocol"]
    assert changed_fields(protocol, base) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert "composite strategy" in protocol.intercurrent_event_handling[2]
    assert protocol.intercurrent_event_handling[1] == base.intercurrent_event_handling[1]
    assert protocol.assumption_rationale[2:] == base.assumption_rationale[1:]
    tmle, collaborative = namespace["tmle"], namespace["collaborative"]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, tmle, collaborative)

    # Both fits read one design, so one estimand; both report an influence-curve interval.
    assert tmle.identified_effect is collaborative.identified_effect
    ordinary, oat = tmle["ate"], collaborative["ate"]
    assert ordinary.inference == "influence_curve"
    assert oat.inference == "influence_curve"
    assert "Its intervals read the stacked parametric curve" in stored_output(
        NOTEBOOK, "collaborative"
    )
    assert not namespace["in_sample"].enabled
    # "The interval contains the true 0.1186 on this draw", for each fit.
    assert covers(ordinary, truth["ate"])
    assert covers(oat, truth["ate"])
    # "The two estimates differ by about 0.015 on this draw."
    assert 0.01 < abs(ordinary.psi - oat.psi) < 0.02

    # Step 8: staffing puts the ordinary g in the tails, and the OAT g has none there.
    weights = {name: fit.diagnostics.support() for name, fit in namespace["fits"].items()}
    ordinary_support, oat_support = weights["TMLE"], weights["OAT C-TMLE"]
    assert ordinary_support.tail_mass[0.1]["below"] > 0.05
    assert ordinary_support.tail_mass[0.1]["above"] > 0.05
    assert ordinary_support.truncated["fraction"] > 0.0
    assert oat_support.tail_mass[0.1]["below"] == oat_support.tail_mass[0.1]["above"] == 0.0
    assert oat_support.truncated["fraction"] == 0.0
    # "The remaining weight variation comes from readiness": the OAT ESS ratios are below 1 in
    # at least one arm, the nonzero witness that the OAT g is not an empty g.
    oat_ess = oat_support.effective_sample_size
    assert min(oat_ess["treated"]["ratio"], oat_ess["control"]["ratio"]) < 0.99
    assert oat_support.clever_covariate_max["mean"] > 2.0
    for arm in ("treated", "control"):
        ordinary_ratio = ordinary_support.effective_sample_size[arm]["ratio"]
        assert ordinary_ratio < 0.75 < oat_ess[arm]["ratio"]
    assert (
        ordinary_support.clever_covariate_max["mean"] > 5 * oat_support.clever_covariate_max["mean"]
    )
    assert namespace["se_ratio"] < 0.9

    # Step 9: the working mechanisms read the outcome prediction only, so they discriminate
    # the offer weakly.  The ordinary g, which holds staffing, discriminates it well.
    assessment = namespace["assessment"]
    assert tuple(assessment.attention) == ()
    nuisance = namespace["nuisance"]
    assert nuisance.treatment_role == "collaborative_working_model"
    for index in (0, 1):
        assert 0.5 < nuisance[f"propensity[{index}]"].metrics["auc"] < 0.6
    ordinary_auc = tmle.diagnostics.nuisance_models()["propensity"].metrics["auc"]
    assert ordinary_auc > 0.8
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    assert ledger[("validation", "score_equations")] == "passed"
    stored = stored_output(NOTEBOOK, "assessment")
    assert "unavailable" not in stored and "not_applicable" not in stored

    # Step 11: the bound on the declared set, with nu2 above the 4 of a g near one half.
    elements, robustness = namespace["elements"], namespace["robustness"]
    assert elements.nu2 > 6.0
    assert 0.0 < robustness["rva"] < robustness["rv"] < 1.0
    _check_refutation(namespace)
    _check_generated_outcome(namespace)
    _check_evalue(namespace)
    _check_stress_surface(namespace)
    _check_bound_limit(namespace)


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 10: three descriptive refits of each fit, and the checks the page says it skips."""
    refutations, fits = namespace["refutations"], namespace["fits"]
    assert list(refutations) == list(fits)
    for name, fit in fits.items():
        report = refutations[name]
        tests = {test.name: test for test in report.tests}
        assert list(tests) == ["placebo", "random_common_cause", "subset"]
        original = fit["ate"].psi
        for test in tests.values():
            # "No check prints a verdict": no registry row covers either fit.
            assert test.passed is None and test.rung == "L0"
            assert len(test.values) == 20 and test.original == pytest.approx(original)
        assert report.passed is None
        # A nonzero witness: a placebo that kept the observed offers would return the original
        # estimate, so the null reading would sit near 0.15 and not near zero.
        values = np.asarray(tests["placebo"].values)
        assert abs(values.mean()) < 0.1 * original
        assert tests["placebo"].statistic_pvalue > 0.05
        # The standard errors of the placebo refits are shorter than the draws' spread, as the
        # reading says of its 20 draws.
        assert np.mean(tests["placebo"].standard_errors) < values.std(ddof=1)
        # "Both reported standard errors are on the scale that the refits show".
        se = fit["ate"].std_error
        reference = float(se * np.sqrt(1 / 0.7 - 1))
        scatter = float(np.std(tests["subset"].values, ddof=1))
        assert 0.8 < scatter / reference < 1.0
        shift = (np.asarray(tests["random_common_cause"].values) - original) / se
        assert 0.0 < abs(shift.mean()) < 0.05
    stored = stored_output(NOTEBOOK, "refutation")
    assert "This fit differs on method 'TMLE'" in stored
    assert "This fit differs on method 'CTMLE[oat:per_arm]'" in stored
    # The page says the registry holds one C-TMLE point row, the greedy placebo cell C1.
    from cleverly.validation import _interpretation as registry

    c_tmle_rows = [
        row
        for row in registry.VERDICT_EVIDENCE
        if any(method.startswith("CTMLE") for method in row.scope.methods or ())
        and row.scope.outcome_families == frozenset({"binomial"})
    ]
    assert [(row.instrument, sorted(row.scope.methods or ())) for row in c_tmle_rows] == [
        ("placebo", ["CTMLE[greedy]"])
    ]
    assert all(row.scope.n_min == 2000 for row in c_tmle_rows)
    # "The data hold the five columns of Step 2 and no other measure".
    assert list(namespace["frame"].columns) == [
        "transition_success",
        "transition_navigation",
        "baseline_readiness",
        "coordination_need",
        "navigator_staffing",
    ]
    with pytest.raises(ValueError, match="negative_control_outcome"):
        namespace["refute"](namespace["collaborative"], tests=("negative_control_outcome",))


def _check_generated_outcome(namespace: dict[str, Any]) -> None:
    """Step 10: the generated Bernoulli law has ATE 0.12, and 400 draws state a pass."""
    law = namespace["generated_law"]
    # An identity link gives an arm contrast that does not vary with W: the truth is the effect.
    assert law.link == "identity" and law.effect == 0.12
    for name, report in namespace["generated"].items():
        (test,) = report.tests
        assert test.name == "simulated_outcome" and test.rung == "L2"
        assert test.passed is True and report.passed is True
        assert len(test.values) == 400
        # "0.12194 and 0.12191": the mean of the generated estimates sits at the law's truth.
        spread = np.std(test.values, ddof=1) / np.sqrt(400)
        assert abs(np.mean(test.values) - 0.12) < 3 * spread
        # The generated truth is not the page's ATE.
        assert abs(namespace["fits"][name]["ate"].psi - 0.12) > 0.02
    stored = stored_output(NOTEBOOK, "generated")
    assert stored.count("VERDICT: every refutation test with a verdict passed it.") == 2


def _check_evalue(namespace: dict[str, Any]) -> None:
    """Step 11: the E-value of each fit is the retargeted risk ratio's, with the page's scope."""
    for name, fit in namespace["fits"].items():
        value = fit.sensitivity.evalue()
        assert value.estimand == "rr" and not value.approximate
        assert value.risk_ratio > 1.0 and 1.0 < value.risk_ratio_ci[0] < value.risk_ratio
        assert value.point == pytest.approx(evalue_from_rr(value.risk_ratio), rel=1e-12)
        # The limit is the E-value of the interval end nearest the null.
        assert value.limit == pytest.approx(evalue_from_rr(value.risk_ratio_ci[0]), rel=1e-12)
        assert 1.0 < value.limit < value.point
        # "Derived ... no nuisance model was refit": the retargeted ratio of two arm means.
        assert value.source_estimand == "ate"
        # The note under the OAT fit only: no registered calibration covers its limit.
        assert (value.limit_reading is not None) == (name == "OAT C-TMLE")
    stored = stored_output(NOTEBOOK, "evalue")
    assert stored.count("No calibration covers this configuration") == 1
    assert stored.index("OAT C-TMLE") < stored.index("No calibration covers this configuration")


def _check_stress_surface(namespace: dict[str, Any]) -> None:
    """Step 11: the stress surface induces no confounding at this nearly balanced offer."""
    surfaces = namespace["surfaces"]
    treated_share = float(namespace["frame"]["transition_navigation"].mean())
    assert 0.45 < treated_share < 0.55
    for name, surface in surfaces.items():
        cells = surface.to_frame()
        assert len(cells) == 16
        zero = cells[(cells["treatment_strength"] == 0.0) & (cells["outcome_strength"] == 0.0)]
        assert float(zero["displacement"].iloc[0]) == 0.0
        # "The printed associations stay between -0.0071 and 0.0122 in every cell."
        association = cells["induced_treatment_association"]
        assert association.min() == pytest.approx(-0.0071, abs=5e-5)
        assert association.max() == pytest.approx(0.0122, abs=5e-5)
        assert abs(association).max() < 0.05
        # "The cells on the diagonal ... return near zero": a nonzero witness against the
        # off-diagonal cells, which move the estimate by several times as much.
        diagonal = cells[
            (cells["treatment_strength"] == cells["outcome_strength"])
            & (cells["treatment_strength"] > 0.0)
        ]["displacement"]
        assert (diagonal > 0).all() and diagonal.max() < 0.04
        # The outcome-only column moves the estimate further as the strength grows.
        column = cells[cells["treatment_strength"] == 0.0].sort_values("outcome_strength")
        movement = column["displacement"].to_numpy()
        assert (np.diff(movement) < 0).all()
        original = namespace["fits"][name]["ate"].psi
        if name == "TMLE":
            assert abs(movement[-1]) > original
        else:
            assert 0.5 * original < abs(movement[-1]) < original
    assert (
        surfaces["TMLE"].to_frame()["induced_treatment_association"].to_numpy()
        == surfaces["OAT C-TMLE"].to_frame()["induced_treatment_association"].to_numpy()
    ).all()
    calibration = surfaces["TMLE"].calibration_frame().set_index(["covariate", "role"])["strength"]
    assert calibration[("baseline_readiness", "outcome")] == pytest.approx(0.4267, abs=5e-5)
    assert calibration[("navigator_staffing", "treatment")] == pytest.approx(0.3727, abs=5e-5)
    assert calibration.max() < 0.5


def _check_bound_limit(namespace: dict[str, Any]) -> None:
    """Step 11: the bound refuses the per-arm OAT fit even under the declaration."""
    collaborative = namespace["collaborative"]
    with pytest.raises(CapabilityError, match="per-arm"):
        omitted_variable_bounds(collaborative, "ate", assume_consistent_mechanism=True)
    with pytest.raises(CapabilityError, match="assume_consistent_mechanism=True"):
        omitted_variable_bounds(collaborative, "ate")
    # The ordinary TMLE fit gives the bound the page reads.
    assert omitted_variable_bounds(namespace["tmle"], "ate").lower < namespace["tmle"]["ate"].psi
