"""The reviewed semantic callback for ``docs/examples/longitudinal-survival``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import (
    EXAMPLES,
    ROOT,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "longitudinal-survival.ipynb"

_PROBE = ROOT / "reviews/notebook-review/probes/longitudinal-survival-refuters/probe.log"
_PROBE_NAME = "reviews/notebook-review/probes/longitudinal-survival-refuters/probe.log"

#: Decimals the Step 10 readings quote from the committed review probe, which no cell prints.
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
    ),
    f"a value of the review probe log, {_PROBE_NAME}",
)


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


def _check_refutation(namespace: dict[str, Any], readmission_t2: Any) -> None:
    """Step 10: the refutation tests of the day-60 readmission difference."""
    contrast_fit, target = namespace["contrast_fit"], namespace["target"]
    # "The two lines agree": the typed contrast and the Step 7 contrast.
    assert contrast_fit[target].psi == pytest.approx(readmission_t2.psi, abs=5e-5)
    assert contrast_fit[target].std_error == pytest.approx(readmission_t2.std_error, abs=5e-5)
    assert contrast_fit.data.n == 4_000 and contrast_fit.data.n_times == 2

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
    # "At 20 draws, none of these numbers shows a defect" on this seeded draw.
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
    # "The code changes seven fields".
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
    # "The death-as-censoring protocol changes three fields".
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

    # "navigation lowers the hazard of any event and raises the share of events that are
    # readmissions": death falls, day-30 readmission falls, day-60 readmission rises.
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

    # "offered patients are older ... less ready", and the day-31 offer goes to more needs.
    by_offer = frame.groupby("navigation_p1")[["age", "baseline_readiness"]].mean()
    assert by_offer.loc[1.0, "age"] > by_offer.loc[0.0, "age"]
    assert by_offer.loc[1.0, "baseline_readiness"] < by_offer.loc[0.0, "baseline_readiness"]
    at_risk = frame.loc[frame["navigation_p2"].notna()]
    needs = at_risk.groupby("navigation_p2")["identified_needs"].mean()
    assert needs.loc[1.0] > needs.loc[0.0] + 0.5
    # "the naive day-30 readmission difference has the wrong sign" on this draw.
    naive = namespace["naive_differences"]
    assert (
        naive["readmission", 1]
        > 0.0
        > population["ate_regimen[always vs never, readmission @ t=1]"]
    )

    # "The first offer has no identified_needs ... The history at the day-31 offer is ...".
    identify_output = stored_output(NOTEBOOK, "identify")
    assert "history at navigation_p1: ['age', 'baseline_readiness']" in identify_output
    assert (
        "history at navigation_p2: ['age', 'baseline_readiness', 'identified_needs', "
        "'navigation_p1']"
    ) in identify_output

    # Step 6. Eight incidences, each interval covering its population value on this draw, and
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
    assert estimate_output.count("always: 938 of 4000 units followed it throughout") == 1
    assert estimate_output.count("never: 671 of 4000 units followed it throughout") == 1

    # Step 7. Each contrast equals the signed sum with the standard error of the summed influence
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
    # "each of the ten printed intervals contains its population value" on this draw.
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
    # "The readmission intervals contain zero ... The death and event-free intervals exclude it."
    for horizon in (1, 2):
        assert covers(differences["readmission", horizon], 0.0)
        assert not covers(differences["death", horizon], 0.0)
        assert not covers(differences["event-free", horizon], 0.0)

    # Step 8. The censoring indicator is K_k = C_k (1 - D_k): both components remove rows, so
    # dropping either one is a mutation that changes the risk set.
    kept_p1 = namespace["kept_p1"]
    enrolled_died = (frame["enrolled_p1"] == 1) & (frame["death_p1"] == 1)
    assert (kept_p1[enrolled_died] == 0).all() and enrolled_died.sum() == 253
    assert (kept_p1[disenrolled] == 0).all() and disenrolled.sum() > 0
    assert ((kept_p1 == 1) == ((frame["enrolled_p1"] == 1) & ~enrolled_died)).all()
    kept_p2 = namespace["kept_p2"]
    died_p2 = namespace["through_p1"] & (frame["death_p2"] == 1)
    left_p2 = namespace["through_p1"] & (frame["enrolled_p2"] == 0)
    assert died_p2.sum() > 0 and left_p2.sum() > 0
    assert (kept_p2[died_p2 | left_p2] == 0).all()
    recoded = namespace["death_as_censoring"]
    assert recoded.loc[kept_p1 == 0, "readmission_p1"].isna().all()
    eliminated = namespace["eliminated"]
    assert eliminated.data.censoring_names == ("enrolled_alive_p1", "enrolled_alive_p2")

    # Each constant equals the generator's functional under its order, and the order matters.
    functional = namespace["functional"]
    for order, death_first in (("death first", True), ("readmission first", False)):
        for plan, arm in (("never", 0.0), ("always", 1.0)):
            assert functional[order][plan] == pytest.approx(
                _death_as_censoring_functional(arm, death_first=death_first), abs=5e-7
            )
    death_first = functional["death first"]["always"] - functional["death first"]["never"]
    assert round(death_first, 4) == -0.0403

    eliminated_t2 = namespace["eliminated_t2"]
    readmission_t2 = differences["readmission", 2]
    # "the shortcut reports a reduction, and the competing-event analysis reports an increase".
    assert eliminated_t2.psi < 0.0 < readmission_t2.psi
    # "The shortcut's interval ... contains both population values."
    assert covers(eliminated_t2, death_first)
    assert covers(eliminated_t2, population["ate_regimen[always vs never, readmission @ t=2]"])
    # "More patients die under the never plan, so the shortcut raises the never-plan risk the
    # most."
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

    # Step 9 prints only the operations that ran.
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
    # "671 patients follow never through the second node, the smallest group", with the largest
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

    # Step 10. The refutation reads a fit of the typed contrast that agrees with Step 7.
    _check_refutation(namespace, readmission_t2)

    # Step 11. Each refit drops one recorded covariate from the full design; every move is
    # positive, and the last refit has no time-varying history left.
    benchmark = namespace["benchmark"]
    assert list(benchmark["dropped"].unique()) == ["age", "baseline_readiness", "identified_needs"]
    assert (benchmark["move"] > 0.0).all()
    moves = benchmark.set_index(["dropped", "cause"])["move_in_se"]
    assert moves.tolist() == [1.75, 0.64, 0.60, 0.37, 0.79, 1.12]
    death_moves = benchmark.loc[benchmark["cause"] == "death", "move"]
    assert death_moves.max() == pytest.approx(0.0175, abs=5e-5)
    # "An unrecorded cause that acts like age could ... change the sign".
    age_move = benchmark.set_index(["dropped", "cause"]).loc[("age", "readmission"), "move"]
    assert age_move > readmission_t2.psi > 0.0
    assert namespace["reduced"].time_varying == ((), ())
    assert namespace["full_design"].time_varying == ((), ("identified_needs",))
