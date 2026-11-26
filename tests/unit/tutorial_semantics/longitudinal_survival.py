"""The reviewed semantic callback for ``docs/examples/longitudinal-survival``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import (
    EXAMPLES,
    ROOT,
    adjustment_table,
    assert_protocol_recorded,
    assert_weights_lesson,
    cap_moves,
    changed_fields,
    covers,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute-checks", "refute-generated")


NOTEBOOK = EXAMPLES / "longitudinal-survival.ipynb"

_PROBE = ROOT / "reviews/notebook-review/probes/longitudinal-survival-refuters/probe.log"
_PROBE_NAME = "reviews/notebook-review/probes/longitudinal-survival-refuters/probe.log"

#: Decimals the refutation Details dropdowns quote from the committed review probe, which no cell prints.
UNPRINTED_DECIMALS = dict.fromkeys(
    (
        "-0.0011",
        "0.0003",
        "0.5401",
        "0.8782",
        "0.9226",
        "1.0698",
        "0.2913",
        "0.9943",
        "-0.0988",
        "0.0972",
        "-0.0580",
        "-0.1568",
        "0.0398",
        "0.0400",
        "0.1380",
    ),
    f"a value of the review probe log, {_PROBE_NAME}",
)


#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "assessment-reading": {"0.01": "estimate"},
}


def _check_probe_log() -> None:
    """Each quoted probe number appears in the committed probe log, at its printed form."""
    text = _PROBE.read_text(encoding="utf-8")
    for literal in UNPRINTED_DECIMALS:
        assert literal in text, f"{literal} is not in {_PROBE_NAME}"


def _death_as_censoring_functional(a: float, *, death_first: bool) -> float:
    """The g-formula with death as a censoring node, from the generator's own equations.

    With death before readmission in each period the readmission hazard among those alive is
    ``h s / (1 - h (1 - s))``; with readmission first it is the cause-specific ``h s``.
    """
    from cleverly.datasets import longitudinal as law

    points, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)
    h1, s1 = law._hazard_one(w1, w2, a), law._relapse_share_one(w1, w2, a)
    l2 = law._L2["w1"] * w1 + law._L2["a1"] * a + noise
    h2, s2 = law._hazard_two(w1, w2, l2, a, a), law._relapse_share_two(w1, l2, a, a)
    if death_first:
        first, second = h1 * s1 / (1 - h1 * (1 - s1)), h2 * s2 / (1 - h2 * (1 - s2))
    else:
        first, second = h1 * s1, h2 * s2
    return float(np.sum(mass * (first + (1 - first) * second)))


def _summed(result: Any, names: list[str], signs: list[float]) -> tuple[float, float]:
    """A signed sum of estimates and its standard error from the summed influence curves."""
    psi = sum(sign * result[name].psi for name, sign in zip(names, signs, strict=True))
    curve = np.sum(
        [sign * result[name].influence_curve for name, sign in zip(names, signs, strict=True)],
        axis=0,
    )
    return psi, float(np.sqrt(np.var(curve, ddof=1) / result.data.n))


def _hazard_truth(process: Any, baseline: Any, doses: tuple[float, float], cause: int) -> float:
    """The risk of ``cause`` by node 2 under a multinomial hazard, from its stated equations.

    The hazard of cause ``j`` is ``exp(eta_j) / (1 + sum_k exp(eta_k))`` with ``eta_j`` the
    intercept, the shared covariate term and the effect times the cumulative dose.  The sum over
    the two nodes is written out, so the check does not call the library's recursion.
    """
    from cleverly.validation.outcome_processes import standardized_sum

    shift = standardized_sum(baseline, process.adjustment_scale)
    risk = np.zeros(len(shift))
    alive = np.ones(len(shift))
    for dose in doses:
        odds = np.exp(
            np.array(process.intercepts)[:, None]
            + shift[None, :]
            + np.outer(process.effects, [dose] * len(shift))
        )
        probability = odds / (1.0 + odds.sum(axis=0))
        risk += alive * probability[cause]
        alive *= 1.0 - probability.sum(axis=0)
    return float(risk.mean())


def _check_contrast_route(namespace: dict[str, Any]) -> None:
    """The headline differences come from a typed contrast fit, which the refutation also reads."""
    from cleverly import RegimeContrast

    result, contrast_fit, target = (
        namespace["result"],
        namespace["contrast_fit"],
        namespace["target"],
    )
    differences = namespace["differences"]
    assert isinstance(contrast_fit.identified_effect.estimand, RegimeContrast)
    assert contrast_fit.data.n == 4_000 and contrast_fit.data.n_times == 2
    assert target == "ate_regimen[always vs never, readmission @ t=2]"
    assert differences["readmission", 2] is contrast_fit[target]
    # "The four differences read `supported`", from the competing-risk study's in-sample row.
    assert len(contrast_fit.estimates) == 4
    for estimate in contrast_fit.estimates.values():
        assert estimate.support.label == "supported" and estimate.support.row == "G13.in_sample"
        # A bounded difference reads its two levels' intervals (correlated MOVER).
        assert estimate.interval_scale == "mover" and estimate.components is not None
    # Nonzero witness that the route sets the label: the same pair through `result.contrast`.
    names = ["cif_regimen[always, readmission @ t=2]", "cif_regimen[never, readmission @ t=2]"]
    derived = result.contrast(lambda psi: psi[0] - psi[1], names, name="derived")
    assert derived.support.label == "experimental"
    assert derived.psi == pytest.approx(contrast_fit[target].psi, abs=1e-12)
    # "the event-free rows, combined by hand, read `experimental`".
    for horizon in (1, 2):
        assert differences["event-free", horizon].support.label == "experimental"
    for estimate in namespace["event_free"].values():
        assert estimate.support.label == "experimental"
    # The silenced warning of this fit names the day-30 death difference as the reader of the
    # flagged share, which the fit does not report (X120 N1).
    (warning,) = [w for w in contrast_fit.fit_warnings if w.category == "SparseInfluenceWarning"]
    assert (
        "cif_regimen[always, death @ t=1], which the fit does not report and whose interval "
        "ate_regimen[always vs never, death @ t=1] reads" in warning.message
    )


def _check_bootstrap_lesson(namespace: dict[str, Any]) -> None:
    """ "Check the flagged interval with the bootstrap" under "Go deeper"."""
    bootstrap_fit, differences = namespace["bootstrap_fit"], namespace["differences"]
    assert bootstrap_fit.bootstrap.n_requested == 199 and bootstrap_fit.bootstrap.n_failed == 0
    assert not [w for w in bootstrap_fit.fit_warnings if w.category == "SparseInfluenceWarning"]
    death = bootstrap_fit["ate_regimen[always vs never, death @ t=1]"]
    readmission = bootstrap_fit["ate_regimen[always vs never, readmission @ t=2]"]
    for estimate, cause, horizon in ((death, "death", 1), (readmission, "readmission", 2)):
        assert estimate.bootstrap.interval == "bca"
        assert estimate.bootstrap.support.label == "experimental"
        # The bootstrap refits the same model, so the point estimate and usual interval match.
        assert estimate.psi == pytest.approx(differences[cause, horizon].psi, abs=1e-12)
        assert estimate.ci == pytest.approx(differences[cause, horizon].ci, abs=1e-12)
    # "keeps its interval well below zero"; "still includes zero".
    assert death.bootstrap.ci[1] < -0.05
    assert covers(readmission.bootstrap.ci, 0.0)
    # The BCa limits differ from the usual ones, so the cell reads a separate interval.
    assert death.bootstrap.ci != pytest.approx(death.ci, abs=1e-4)


def _check_crossfit_lesson(namespace: dict[str, Any]) -> None:
    """ "Flexible outcome models with cross-fitting" under "Go deeper"."""
    crossfit_fit, differences, known = (
        namespace["crossfit_fit"],
        namespace["differences"],
        namespace["known"],
    )
    assert crossfit_fit.data.n == 4_000
    assert crossfit_fit.config.n_folds == 10
    for estimate in crossfit_fit.estimates.values():
        assert estimate.support.label == "supported"
        assert estimate.support.row == "G13.cross_fitted"
    # "goes from 0.0223 to 0.0225, and both intervals contain the true 0.0024".
    flexible = crossfit_fit["ate_regimen[always vs never, readmission @ t=2]"]
    assert round(differences["readmission", 2].psi, 4) == 0.0223
    assert round(flexible.psi, 4) == 0.0225
    assert covers(flexible, known["readmission", 2])
    assert covers(differences["readmission", 2], known["readmission", 2])
    # The printed mechanism checks are out of fold, so they are not 1 by construction.
    mechanisms = namespace["mechanisms"]
    assert len(mechanisms) == 4
    assert set(mechanisms["evaluation"]) == {"out_of_fold"}
    slopes = mechanisms.set_index(["model", "time"])
    assert round(slopes.loc[("offer", 1), "calibration_slope"], 2) == 0.99
    assert round(slopes.loc[("offer", 2), "calibration_slope"], 2) == 0.99
    for time in (1, 2):
        row = slopes.loc[("enrollment", time)]
        # "within about one standard error of 1".
        assert abs(row["calibration_slope"] - 1.0) < 1.1 * row["calibration_slope_se"]
    # Nonzero witness: in sample, the main fit's logistic offer and enrollment models give slopes
    # within 0.015 of 1, which their score equations force (up to the default penalty); out of
    # fold, the day-31 enrollment slope moves away from 1.
    in_sample = namespace["assessment"].report("nuisance_models").to_frame()
    in_sample = in_sample[in_sample["role"].isin(["treatment", "censoring"])]
    assert set(in_sample["evaluation"]) == {"in_sample"}
    assert (in_sample["calibration_slope"] - 1.0).abs().max() < 0.015
    assert slopes.loc[("enrollment", 2), "calibration_slope"] < 0.9


def _check_rule_lesson(namespace: dict[str, Any]) -> None:
    """ "Offer on day 31 only to patients with identified needs" under "Go deeper"."""
    from cleverly.datasets import competing_rule_truth, competing_truth, rule_arm_at_node_two

    rule_fit, contrast_fit = namespace["rule_fit"], namespace["contrast_fit"]
    protocol, rule_protocol = namespace["protocol"], namespace["rule_protocol"]
    assert changed_fields(rule_protocol, protocol) == {"treatment_strategies", "treatment_versions"}
    assert_protocol_recorded(NOTEBOOK, "rule-protocol", rule_protocol, rule_fit)
    assert namespace["needs_rule"].rule_kind == "known"
    # The truths, pinned as literals and recomputed with the dataset's own helper.
    expected = {
        ("readmission", 1): -0.0265092091,
        ("readmission", 2): 0.0079943104,
        ("death", 1): -0.0816978242,
        ("death", 2): -0.1345304816,
    }
    simulated = {"readmission": "relapse", "death": "death"}
    for (cause, horizon), value in expected.items():
        truth = competing_rule_truth(
            1.0, rule_arm_at_node_two, simulated[cause], horizon
        ) - competing_truth(0.0, 0.0, simulated[cause], horizon)
        assert namespace["rule_known"][cause, horizon] == pytest.approx(truth, abs=1e-12)
        assert truth == pytest.approx(value, abs=1e-9)
    always_name = "ate_regimen[always vs never, {} @ t={}]"
    rule_name = "ate_regimen[needs_rule vs never, {} @ t={}]"
    for cause in ("readmission", "death"):
        # "The day-30 rows equal always's".
        rule_t1, always_t1 = (
            rule_fit[rule_name.format(cause, 1)],
            contrast_fit[always_name.format(cause, 1)],
        )
        assert rule_t1.psi == pytest.approx(always_t1.psi, abs=1e-12)
        assert rule_t1.std_error == pytest.approx(always_t1.std_error, abs=1e-12)
        # By day 60 the rule differs from always, but stays "close to always on both".
        rule_t2, always_t2 = (
            rule_fit[rule_name.format(cause, 2)],
            contrast_fit[always_name.format(cause, 2)],
        )
        assert 1e-3 < abs(rule_t2.psi - always_t2.psi) < 0.01
        for horizon in (1, 2):
            estimate = rule_fit[rule_name.format(cause, horizon)]
            assert covers(estimate, expected[cause, horizon])
            assert estimate.support.label == "supported"
            assert estimate.support.row == "G13.in_sample"
    # "78% have needs above zero, so the rule would skip the day-31 call for about a fifth".
    assert round(100 * namespace["share"]) == 78
    assert 0.17 < 1.0 - namespace["share"] < 0.25


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation and generated-outcome refits of the day-60 readmission difference."""
    contrast_fit, target = namespace["contrast_fit"], namespace["target"]
    readmission_t2 = namespace["differences"]["readmission", 2]
    # Each test prints a descriptive reading at level L0: the registry covers a survival outcome
    # and the fit has competing events.
    refutation, stability = namespace["refutation"], namespace["stability"]
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    for test in (*tests.values(), *stability.tests):
        assert test.verdict == "descriptive" and test.passed is None and test.rung == "L0"
        assert test.original == pytest.approx(readmission_t2.psi, abs=1e-12)
    assert refutation.passed is None
    for name in ("placebo", "subset"):
        assert "This fit differs on outcome family 'competing'" in tests[name].scope_sentence
    assert "Covered:" not in tests["random_common_cause"].scope_sentence
    assert tests["placebo"].n_replicates == tests["subset"].n_replicates == 20
    # "all four behave as expected: the random offers give a mean near zero" on this seeded draw.
    assert tests["placebo"].expectation == "~ 0"
    assert tests["placebo"].statistic_pvalue > 0.05 and tests["subset"].statistic_pvalue > 0.05
    assert abs(tests["random_common_cause"].statistic) < 0.05
    spread = stability.tests[0]
    assert spread.n_replicates == 20
    assert 0.8 < spread.spread / contrast_fit[target].std_error < 1.25
    # ``noise_nodes=True`` puts a noise column at every node. A baseline-only column moves the
    # estimate differently under the same seed, so the flag acts on this fit.
    baseline_only = contrast_fit.diagnostics.refute(
        estimand=target, tests=("random_common_cause",), random_state=61
    ).tests[0]
    assert baseline_only.mean != pytest.approx(tests["random_common_cause"].mean, abs=1e-9)

    # Generated outcomes: the dummy law has the exact null, the adjusted laws do not, and the
    # simulated truth equals the hazard equations written out for the static regimens.
    generated = namespace["generated_tests"]
    assert list(generated) == ["dummy_outcome", "simulated_outcome", "plasmode"]
    for test in generated.values():
        bias = test.readings[0]
        assert test.n_replicates == 20 and test.rung == "L1" and test.verdict == "descriptive"
        assert bias.name == "bias" and not bias.inside
        # 20 draws give an interval wider than the band, which is why no verdict prints.
        assert bias.upper - bias.lower > 2 * bias.band[1]
    assert generated["dummy_outcome"].declared_effect == 0.0
    baseline = contrast_fit.data.baseline
    adjusted = namespace["adjusted"]
    expected = _hazard_truth(adjusted, baseline, (1.0, 2.0), 0) - _hazard_truth(
        adjusted, baseline, (0.0, 0.0), 0
    )
    assert generated["simulated_outcome"].declared_effect == pytest.approx(expected, abs=1e-10)
    assert expected > 0.1
    # The plasmode truth is a Monte Carlo value, near the held-history truth.
    assert generated["plasmode"].declared_effect == pytest.approx(0.1602, abs=5e-4)
    assert abs(generated["plasmode"].declared_effect - expected) < 0.03


def check(namespace: dict[str, Any]) -> None:
    """The competing-risk analysis, its failure mode, and the readings' relations hold."""
    _check_probe_log()
    result = namespace["result"]
    population = namespace["population"]
    frame = namespace["frame"]

    # Each protocol step prints its record, and each fit carries the digest of its own protocol.
    protocol, eliminated_protocol = namespace["protocol"], namespace["eliminated_protocol"]
    fingerprints = {
        assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result),
        assert_protocol_recorded(NOTEBOOK, "failure-mode", protocol),
        assert_protocol_recorded(
            NOTEBOOK, "failure-mode", eliminated_protocol, namespace["eliminated"]
        ),
    }
    assert len(fingerprints) == 2
    # The adapted protocol changes seven fields (no longer quoted; the code cell shows them).
    assert changed_fields(protocol, namespace["program"]) == {
        "eligibility",
        "treatment_strategies",
        "treatment_versions",
        "outcome",
        "horizon",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # The protocol names death as competing and disenrollment as censoring.
    handling = protocol.intercurrent_event_handling
    assert "competing event" in handling[0]
    assert "disenrollment" in handling[1] and "stayed enrolled" in handling[1]
    # The death-as-censoring protocol changes three fields (no longer quoted).
    assert changed_fields(eliminated_protocol, protocol) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert eliminated_protocol.intercurrent_event_handling[1] == handling[1]

    # The data step: the counts, the rows the reading names, and the censoring gate.
    assert frame["enrolled_p1"].eq(0).sum() == 451
    assert frame["enrolled_p2"].eq(0).sum() == 233
    disenrolled = frame["enrolled_p1"] == 0
    assert frame.loc[disenrolled, ["readmission_p1", "death_p1"]].isna().all().all()
    assert disenrolled.iloc[1] and frame.iloc[1, 4:].isna().all()
    assert frame.loc[4, "readmission_p1"] == 1 and frame.loc[4, "readmission_p2"] == 1
    assert np.isnan(frame.loc[4, "navigation_p2"])
    assert result.data.censoring_names == ("enrolled_p1", "enrolled_p2")
    assert result.data.cluster is None

    # "about 0.21 of patients die first by day 60 under never against 0.07 under always, while
    # readmission is nearly equal": death falls, day-30 readmission falls, day-60 rises.
    def cif(plan: str, cause: str, horizon: int) -> float:
        return population[f"cif_regimen[{plan}, {cause} @ t={horizon}]"]

    assert cif("always", "death", 2) < cif("never", "death", 2) - 0.1
    assert cif("always", "readmission", 1) < cif("never", "readmission", 1)
    assert cif("always", "readmission", 2) > cif("never", "readmission", 2)
    from cleverly.datasets import longitudinal as law

    for plan, arm in (("always", 1.0), ("never", 0.0)):
        for horizon in (1, 2):
            total = cif(plan, "readmission", horizon) + cif(plan, "death", horizon)
            assert total == pytest.approx(law.survival_truth(arm, arm, horizon), abs=1e-9)
    assert law._relapse_share_one(0.0, 0.0, 1.0) > law._relapse_share_one(0.0, 0.0, 0.0)

    # "Offered patients were older and less ready ... the day-31 offer went to patients with more
    # needs".
    by_offer = frame.groupby("navigation_p1")[["age", "baseline_readiness"]].mean()
    assert by_offer.loc[1.0, "age"] > by_offer.loc[0.0, "age"]
    assert by_offer.loc[1.0, "baseline_readiness"] < by_offer.loc[0.0, "baseline_readiness"]
    at_risk = frame.loc[frame["navigation_p2"].notna()]
    needs = at_risk.groupby("navigation_p2")["identified_needs"].mean()
    assert needs.loc[1.0] > needs.loc[0.0] + 0.5
    # "at day 30 it has the wrong sign" (the simple difference) on this draw.
    naive = namespace["naive_differences"]
    assert (
        naive["readmission", 1]
        > 0.0
        > population["ate_regimen[always vs never, readmission @ t=1]"]
    )

    # "the history lines show what each offer is adjusted for".
    identify_output = stored_output(NOTEBOOK, "identify")
    adjusted = adjustment_table(identify_output)
    assert adjusted["navigation_p1"] == ("age", "baseline_readiness")
    assert adjusted["navigation_p2"] == (
        "age",
        "baseline_readiness",
        "identified_needs",
        "navigation_p1",
    )

    # Estimate readmission and death risk. Eight incidences, each interval covering its population value on this draw, and
    # simultaneous bands wider than the pointwise intervals.
    assert len(result.estimates) == 8
    for name, estimate in result.estimates.items():
        assert covers(estimate, population[name])
    bands = result.simultaneous
    assert bands is not None and len(bands.bands) == 8
    assert bands.critical_value > bands.pointwise_critical_value
    for name, (low, high) in bands.bands.items():
        assert low < result[name].ci[0] and high > result[name].ci[1]
    estimate_output = stored_output(NOTEBOOK, "estimate")
    assert len(re.findall(r"^always +938 of 4,000 ", estimate_output, re.MULTILINE)) == 1
    assert len(re.findall(r"^never +671 of 4,000 ", estimate_output, re.MULTILINE)) == 1

    # Compare the follow-up plans. Each contrast equals the signed sum with the standard error of the summed influence
    # curves, so the joint correlation enters.  Event-free survival is one minus both causes; one
    # minus one cause is a different number, which is the nonzero witness.
    differences, event_free = namespace["differences"], namespace["event_free"]
    incidence = namespace["incidence"]
    for cause in ("readmission", "death"):
        for horizon in (1, 2):
            names = [incidence("always", cause, horizon), incidence("never", cause, horizon)]
            psi, se = _summed(result, names, [1.0, -1.0])
            assert differences[cause, horizon].psi == pytest.approx(psi, abs=1e-12)
            assert differences[cause, horizon].std_error == pytest.approx(se, rel=1e-8)
    for horizon in (1, 2):
        for plan in ("always", "never"):
            names = [incidence(plan, "readmission", horizon), incidence(plan, "death", horizon)]
            psi, se = _summed(result, names, [-1.0, -1.0])
            assert event_free[plan, horizon].psi == pytest.approx(1.0 + psi, abs=1e-12)
            assert event_free[plan, horizon].std_error == pytest.approx(se, rel=1e-8)
            # The two causes of one plan share patients: a patient who dies cannot be
            # readmitted, so their errors are negatively correlated and the joint SE is smaller.
            independent = np.hypot(result[names[0]].std_error, result[names[1]].std_error)
            assert se < independent - 1e-4
            assert (1.0 - result[names[0]].psi) - event_free[plan, horizon].psi > 0.02
        free = differences["event-free", horizon]
        assert free.psi == pytest.approx(
            event_free["always", horizon].psi - event_free["never", horizon].psi, abs=1e-12
        )
    # "every interval contains its true value" on this draw.
    for (cause, horizon), estimate in differences.items():
        if cause == "event-free":
            target = sum(
                cif("never", c, horizon) - cif("always", c, horizon)
                for c in ("readmission", "death")
            )
        else:
            target = population[f"ate_regimen[always vs never, {cause} @ t={horizon}]"]
        assert covers(estimate, target)
    for (plan, horizon), estimate in event_free.items():
        assert covers(
            estimate, 1.0 - cif(plan, "readmission", horizon) - cif(plan, "death", horizon)
        )
    assert len(differences) + len(event_free) == 10
    # "the interval ... includes zero" for readmission; death and event-free "well away from zero".
    for horizon in (1, 2):
        assert covers(differences["readmission", horizon], 0.0)
        assert not covers(differences["death", horizon], 0.0)
        assert not covers(differences["event-free", horizon], 0.0)

    # Death as censoring through eliminate=. Retention has two factors: the censoring step keeps
    # the enrolled, then the death step removes the enrolled who died. Dropping either factor is a
    # mutation that changes the risk set.
    eliminated = namespace["eliminated"]
    data = eliminated.data
    assert data.eliminated == ("death",)
    enrolled_died = ((frame["enrolled_p1"] == 1) & (frame["death_p1"] == 1)).to_numpy()
    assert enrolled_died.sum() == 253
    assert data.retained is not None
    retained_p1, kept_p1 = data.retained[:, 0], data.uncensored[:, 0]
    assert (retained_p1 == (frame["enrolled_p1"] == 1).to_numpy()).all()
    assert (retained_p1 & ~kept_p1 == enrolled_died).all()
    assert not kept_p1[disenrolled.to_numpy()].any() and disenrolled.sum() > 0
    through_p1 = kept_p1 & (frame["readmission_p1"].fillna(0) == 0).to_numpy()
    died_p2 = through_p1 & (frame["death_p2"] == 1).to_numpy()
    left_p2 = through_p1 & (frame["enrolled_p2"] == 0).to_numpy()
    assert died_p2.sum() > 0 and left_p2.sum() > 0
    assert not data.uncensored[died_p2 | left_p2, 1].any()

    # Each constant equals the generator's functional under its order, and the order matters.
    functional = namespace["shortcut_truth"]
    for order, death_first in (("death first", True), ("readmission first", False)):
        for plan, arm in (("never", 0.0), ("always", 1.0)):
            assert functional[order][plan] == pytest.approx(
                _death_as_censoring_functional(arm, death_first=death_first), abs=5e-7
            )
    death_first = functional["death first"]["always"] - functional["death first"]["never"]
    assert round(death_first, 4) == -0.0403

    eliminated_t2 = namespace["eliminated_t2"]
    readmission_t2 = differences["readmission", 2]
    # The difference comes from the typed contrast fit and reads `supported` from the
    # eliminated-events study, as do the risks; `eliminated.contrast` gives the same numbers and
    # reads experimental, the witness that the route sets the label.
    derived = eliminated.contrast(
        lambda psi: psi[0] - psi[1],
        ["risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]"],
        name="derived",
    )
    assert derived.support.label == "experimental"
    assert eliminated_t2.psi == pytest.approx(derived.psi, abs=1e-12)
    assert eliminated_t2.std_error == pytest.approx(derived.std_error, rel=1e-8)
    assert eliminated_t2.support.label == "supported"
    assert eliminated_t2.support.row == "G48.in_sample"
    for estimate in eliminated.estimates.values():
        assert estimate.support.label == "supported"
    # "turns the small positive estimate (0.0223) into a negative one (-0.0157)".
    assert eliminated_t2.psi < 0.0 < readmission_t2.psi
    # "The interval (-0.0596, 0.0267) contains both 0.0024 and -0.0403".
    assert covers(eliminated_t2, death_first)
    assert covers(eliminated_t2, population["ate_regimen[always vs never, readmission @ t=2]"])
    # "mostly by raising the never-schedule risk from 0.2285 to 0.2858, since more patients die
    # under never".
    raised_never = (
        eliminated["risk_regimen[never @ t=2]"].psi
        - result[incidence("never", "readmission", 2)].psi
    )
    raised_always = (
        eliminated["risk_regimen[always @ t=2]"].psi
        - result[incidence("always", "readmission", 2)].psi
    )
    assert raised_never > raised_always + 0.01 and raised_always > 0.0
    assert population["ate_regimen[always vs never, death @ t=2]"] < 0.0

    # Check the fit prints only the operations that ran.
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "unavailable" not in assessment_output and "deferred" not in assessment_output
    ran = namespace["ran"]
    assert list(zip(ran["check"], ran["status"], strict=True)) == [
        ("score_equations", "passed"),
        ("support", "completed"),
        ("nuisance_models", "completed"),
    ]
    assessment = namespace["assessment"]
    assert not assessment.attention
    support = namespace["support"]
    assert support["converged"].all() and (support["share_truncated"] == 0.0).all()
    # "The weakest group is the 671 patients who followed never through day 31", with the largest
    # weight and the minimum effective-sample-size ratio there.
    ratios = support["effective_n"] / support["n_followed"]
    weakest = support.loc[ratios.idxmin()]
    assert (weakest["regimen"], weakest["time"], weakest["n_followed"]) == ("never", 2, 671)
    assert support["n_followed"].min() == 671
    assert support.loc[support["max_weight"].idxmax(), "regimen"] == "never"
    assert ratios.min() == pytest.approx(0.788, abs=0.0005)
    largest = support.loc[support["epsilon"].abs().idxmax()]
    assert (largest["regimen"], largest["cause"], largest["horizon"], largest["time"]) == (
        "never",
        "readmission",
        2,
        2,
    )
    assert largest["epsilon"] == pytest.approx(-0.0679, abs=5e-5)

    # Compare the two schedules: the typed contrast route, which the refutation also reads.
    _check_contrast_route(namespace)

    # How much each recorded variable moves the answer. Each refit drops one recorded covariate from the full design; every move is
    # positive, and the last refit has no time-varying history left.
    benchmark = namespace["benchmark"]
    assert list(benchmark["dropped"].unique()) == ["age", "baseline_readiness", "identified_needs"]
    assert (benchmark["move"] > 0.0).all()
    moves = benchmark.set_index(["dropped", "cause"])["move_in_se"]
    assert moves.tolist() == [1.75, 0.64, 0.60, 0.37, 0.79, 1.12]
    death_moves = benchmark.loc[benchmark["cause"] == "death", "move"]
    assert death_moves.max() == pytest.approx(0.0175, abs=5e-5)
    # "a missing factor that does the same, as strongly as `age`, could put the true difference
    # below zero".
    age_move = benchmark.set_index(["dropped", "cause"]).loc[("age", "readmission"), "move"]
    assert age_move > readmission_t2.psi > 0.0
    assert namespace["reduced"].time_varying == ((), ())
    assert namespace["full_design"].time_varying == ((), ("identified_needs",))
    _check_weights_lesson(namespace)
    _check_bootstrap_lesson(namespace)
    _check_crossfit_lesson(namespace)
    _check_rule_lesson(namespace)


def _check_weights_lesson(namespace: dict[str, Any]) -> None:
    """The "Weights and balance period by period" lesson under "Go deeper"."""
    assert_weights_lesson(namespace, feedback=False)
    balance = namespace["balance_by_check"]
    censoring = balance[balance["diagnostic"].eq("censoring")]
    worst = censoring.loc[censoring["after"].idxmax()]
    # "falls from 0.25 to 0.14, which is identified needs in the second period".
    assert worst["covariate"] == "identified_needs" and worst["exposure"] == "censoring at 2"
    assert censoring.drop(index=worst.name)["after"].max() < worst["after"] - 0.01
    rows = namespace["weight_rows"].set_index(["regimen", "time"])["max_weight"]
    # "At the day-31 offer they reach 28.72 for never and 14.533 for always".
    assert rows["never", 2] > rows["always", 2] > rows["always", 1]
    moves = cap_moves(namespace)
    # "Raising the floor to 0.02 moves nothing. At 0.05 only the two never risks by day 60
    # move", each by under a tenth of a standard error.
    assert moves[0.02].abs().max() < 1e-9
    moved = moves.index[moves[0.05].abs() > 1e-9]
    assert set(moved) <= {
        "cif_regimen[never, readmission @ t=2]",
        "cif_regimen[never, death @ t=2]",
    }
    assert moves[0.05].abs().max() < 0.1
    # "only 72.8 and 77.9 behind the two death risks", fewer than behind the always
    # readmission risk.
    kish = namespace["influence_rows"]["kish"]
    readmission = kish["cif_regimen[always, readmission @ t=2]"]
    assert kish["cif_regimen[always, death @ t=2]"] < readmission
    assert kish["cif_regimen[never, death @ t=2]"] < readmission
