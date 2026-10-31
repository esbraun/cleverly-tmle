"""The pre-run design of ``mtp-boundaries``: truths, design shares, sizing rules and controls.

Every number the study declares before its pilot is recomputed here: the quadrature truths, the
binding, overlap, preimage and atom shares of law B and law B' (``X40-plan.md`` section 5.0's
probe numbers, which were Monte Carlo at 10^6 draws, to their Monte Carlo error), the
arm-pooled pass-probability rule that sets ``R`` from the pilot, the per-arm screen, the power
of each necessity pair as a function of the pilot's standardized bias, the seeds, the oracle
density, the rich first-node basis, and each deliberate defect against its closed form.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies import canonical_mtp_boundaries as study
from tests.studies import mtp_boundaries_common as common
from tests.studies import mtp_boundaries_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import ROOT, registered

pytestmark = pytest.mark.xdist_group("mtp_boundaries_design")

#: The probe numbers of ``reviews/x40/rev2/lawB2.log`` (10^6 draws), with their tolerance: the
#: Monte Carlo SE of a share near 0.25 is 0.0005 at 10^6 draws, and the log rounds to 0.001.
PROBE_TOLERANCE = 0.002


# ---------------------------------------------------------------------------- truths


def test_every_truth_is_stable_and_every_plan_moves_the_mean() -> None:
    b = {label: common.truth(label, "B") for label in common.B_LABELS}
    atom = {label: common.truth(label, "B_atom") for label in common.B_ATOM_LABELS}
    for label in ("p1", "p2", "p3", "p4"):
        assert abs(b[label] - b["natural"]) > 0.01
    assert abs(atom["p5"] - atom["natural"]) > 0.01
    plans = [b[label] for label in ("p1", "p2", "p3", "p4")]
    gaps = [abs(x - y) for i, x in enumerate(plans) for y in plans[i + 1 :]]
    assert min(gaps) > 0.005


def test_the_pinned_truths_match_the_quadrature() -> None:
    assert common.truth("natural", "B") == pytest.approx(0.629508896737884, abs=1e-12)
    assert common.contrast("p3") == pytest.approx(-0.025184762881319434, abs=1e-12)
    assert common.contrast("p4") == pytest.approx(-0.11148632112351742, abs=1e-12)
    assert common.contrast("p5") == pytest.approx(-0.02685794801370245, abs=1e-12)


def test_the_truth_reads_definition_one() -> None:
    """A Monte Carlo of Definition 1 agrees with the quadrature, for every plan."""
    rng = np.random.default_rng(5)
    n = 400_000
    frame = common.sample(n, 17, atom=True)
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    u = common.upper(w2)
    for label in ("p2", "p4", "p5"):
        d1, d2 = common.MAPS[label]
        atom = label == "p5"
        a1 = d1(frame["A1"].to_numpy(), u)
        l2 = rng.binomial(1, common.p_l2(w1, a1)).astype(float)
        a2 = common.tn_draw(rng, common.mean2(w1, w2, a1, l2), u)
        if atom:
            a2 = np.where(rng.random(n) < common.p_atom(w1, l2), common.ATOM, a2)
        mean = float(np.mean(common.outcome_mean(w1, a1, l2, d2(a2, u))))
        law = "B_atom" if atom else "B"
        assert abs(mean - common.truth(label, law)) < 4 * 0.5 / np.sqrt(n)


# ---------------------------------------------------------------------------- design shares


def test_the_binding_share_of_p2() -> None:
    node1 = common.share_at_least(lambda u: u - common.STEP, 1)
    node2 = common.share_at_least(lambda u: u - common.STEP, 2)
    assert node1 >= 0.20 and node2 >= 0.20
    assert node1 == pytest.approx(0.245, abs=PROBE_TOLERANCE)
    assert node2 == pytest.approx(0.255, abs=PROBE_TOLERANCE)


def test_the_two_branch_overlap_of_p4() -> None:
    def overlap(a: np.ndarray, u: float) -> np.ndarray:
        d = common._branches(a, u)
        return (d >= 1.5) & (d < 4.0)

    # The overlap's preimages end at 1.5 - 1 = 0.5 and 4 + 1.5 = 5.5.
    jumps = (1.5 - common.BRANCH_UP, 4.0 + common.BRANCH_DOWN)
    node1 = common.mass_where(overlap, 1, jumps)
    node2 = common.mass_where(overlap, 2, jumps)
    assert node1 >= 0.30 and node2 >= 0.30
    assert node1 == pytest.approx(0.579, abs=PROBE_TOLERANCE)
    assert node2 == pytest.approx(0.571, abs=PROBE_TOLERANCE)


def test_the_bounds_of_the_outcome_and_the_atom_probability() -> None:
    low, high = common.outcome_bounds()
    assert (round(low, 3), round(high, 3)) == (0.060, 0.899)
    assert low >= 0.05 and high <= 0.95
    low, high = common.atom_probability_bounds()
    assert (round(low, 3), round(high, 3)) == (0.130, 0.378)
    assert low >= 0.1 and high <= 0.4


def test_the_preimage_and_the_atom_ratio_of_p5() -> None:
    p = common.atom_history_expectation(lambda p, s: p)
    preimage = common.atom_history_expectation(lambda p, s: (1 - p) * s)
    ratio = common.atom_history_expectation(lambda p, s: common.atom_ratio(p, s))
    above = common.atom_history_expectation(
        lambda p, s: (common.atom_ratio(p, s) > 2.0).astype(float)
    )
    assert p == pytest.approx(0.246, abs=PROBE_TOLERANCE)
    assert preimage >= 0.20
    assert preimage == pytest.approx(0.270, abs=PROBE_TOLERANCE)
    assert p + preimage == pytest.approx(0.515, abs=PROBE_TOLERANCE)
    assert ratio == pytest.approx(2.17, abs=0.01)
    assert above == pytest.approx(0.508, abs=PROBE_TOLERANCE)
    # The probe's 6.43 is a sample maximum, so the supremum is at least that.
    assert common.atom_ratio_supremum() >= 6.43


def test_the_atom_maps_to_itself() -> None:
    """Condition 1 of section 2.3: the declared atom's image is a declared atom."""
    assert common._to_atom(np.array([common.ATOM]), 6.0)[0] == common.ATOM


# ---------------------------------------------------------------------------- sizing


@pytest.mark.parametrize(
    ("arms", "pilot", "expected"),
    [
        (7, 3_000, (6_000, 6_400, 8_000)),
        (7, 1_000, (None, None, None)),
        (7, 2_000, (8_400, 9_200, 12_400)),
        (7, 4_000, (5_200, 5_600, 6_800)),
    ],
)
def test_the_arm_pooled_rule_reproduces_the_rev2_probe(
    arms: int, pilot: int, expected: tuple[int | None, ...]
) -> None:
    assert round(tools.C_FIT, 4) == 1.0692
    found = tuple(tools.smallest_replicates(m, pilot, arms)[0] for m in (1.0, 1.005, 1.01))
    assert found == expected


def test_the_recorded_probabilities_of_s1() -> None:
    assert round(tools.tau_upper(3_000, 7), 4) == 0.0119
    _, cell, family = tools.smallest_replicates(1.0, 3_000, 7)
    assert (round(cell, 3), round(family, 3)) == (0.994, 0.961)
    _, cell, family = tools.smallest_replicates(1.01, 3_000, 7)
    assert (round(cell, 3), round(family, 3)) == (0.993, 0.954)


def test_the_family_wise_rule_shares_the_pilot_mean_error() -> None:
    """With the shared term, the family is less likely to pass than independent cells imply."""
    replicates, _cell, _family = tools.smallest_replicates(1.005, 3_000, 7)
    assert replicates == 6_400
    sigma = tools.pilot_sigma(3_000)
    tau = tools.tau_upper(3_000, 7)
    _, independent = tools.pass_probabilities(replicates, 1.005, 0.0, tau, 7)
    _, shared = tools.pass_probabilities(replicates, 1.005, sigma**2 / 7, tau, 7)
    assert shared < independent


def test_the_per_arm_screen_and_the_contingency() -> None:
    assert round(tools.pilot_sigma(3_000), 4) == 0.0138
    assert round(tools.screen_half_width(3_000), 3) == 0.046
    agreeing = tools.replicates_from_pilot([1.0] * 7, 3_000)
    assert agreeing["R"] == 6_000 and not agreeing["suspect"]
    off = tools.replicates_from_pilot([1.0] * 6 + [1.06], 3_000)
    assert off["screened"] == [True] * 6 + [False]
    shifted = tools.replicates_from_pilot([1.016] * 7, 3_000)
    assert shifted["suspect"]


def test_the_declared_sizes() -> None:
    assert properties.PILOT_DRAWS == 3_000
    assert set(properties.CALIBRATION_REPLICATES) == set(study.CALIBRATION_LABELS)
    for label, replicates in properties.CALIBRATION_REPLICATES.items():
        reporting = label in properties.REPORTING_LABELS
        expected = properties.PILOT_DRAWS if reporting else properties.POOLED_REPLICATES
        assert replicates == expected, label
    assert {label for label, n in properties.CALIBRATION_SIZES.items() if n != 2_000} == {
        "p4",
        "p4_sdr",
    }
    assert properties.NECESSITY_REPLICATES == 1_000
    assert properties.NECESSITY_PILOT_DRAWS >= 100
    assert study.FAILURE_BOUND == 0.01


def test_the_pooled_r_is_the_rule_read_off_the_published_pilots() -> None:
    """Sizing rules 1 to 3 on the declared pilot readings, the reporting arm left out."""
    assert set(properties.PILOT_SE_RATIOS) == set(properties.CELLS) - properties.REPORTING_LABELS
    reading = tools.replicates_from_pilot(
        list(properties.PILOT_SE_RATIOS.values()), properties.PILOT_DRAWS
    )
    assert all(reading["screened"]) and not reading["suspect"]
    assert reading["R"] == properties.POOLED_REPLICATES
    assert reading["family_wise"] >= 0.95 and reading["per_cell"] >= 0.90
    # The reporting arm's two readings lie outside the screen at both sizes.
    half = tools.screen_half_width(properties.PILOT_DRAWS)
    assert abs(0.919 - 1.0) > half and abs(0.926 - 1.0) > half


def test_the_projected_pseudo_learner_moves_an_sdr_fit_on_law_b() -> None:
    """X40-pilot-diagnosis.md fix item 3: the learner choice is visible in the record.

    On pilot draw 0 of ``p4_sdr`` at n = 2,000 the unprojected first node misses the true
    ``m_1`` by up to 6.8 at the assigned dose (E3), so the two fits must differ.
    """
    plan = "p4"
    law = common.LAW_OF[plan]
    seed = properties._pilot_seed("interval_calibration", "p4_sdr", 0)
    frame = common.SAMPLERS[law](2_000, seed)
    name = properties.estimand(plan)
    labels = (study.REFERENCE, plan)
    fits = [
        study.fit(frame, law, labels, estimator="sdr", outcome="correct", pseudo_learner=learner)[
            name
        ].psi
        for learner in (None, common.UnitRangePseudo())
    ]
    assert abs(fits[0] - fits[1]) > 0.01


def test_necessity_power_as_a_function_of_the_pilot_bias() -> None:
    # Power 0.90 at R = 1,000 needs |delta| >= 0.25 + (2.576 + 1.2816) / sqrt(1000).
    needed = 0.25 + (tools.Z + norm.ppf(0.90)) / np.sqrt(1_000)
    assert tools.necessity_power(needed, 1_000) == pytest.approx(0.90, abs=1e-9)
    assert tools.necessity_power(needed - 0.01, 1_000) < 0.90
    assert tools.necessity_power(-needed, 1_000) == pytest.approx(0.90, abs=1e-9)
    assert tools.equivalence_power(0.0, 1_000) > 0.999


# ---------------------------------------------------------------------------- seeds


def test_the_seeds_are_new() -> None:
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
        if seed is not None
    }
    assert study.SEED not in others
    assert study.RESAMPLING_SEED not in others
    assert common.BOUND_SEED not in others | {study.SEED, study.RESAMPLING_SEED}


def test_the_record_compares_lmtp_through_its_runner() -> None:
    from tests.canonical.mtp_boundaries.regenerate import REFERENCE

    record = study.STUDY
    assert record.comparator_state == "compared"
    assert record.reference == "lmtp"
    assert record.extra_artifacts == ("exactness.csv",)
    assert REFERENCE.runner == "mtp_boundaries_runner.R"
    runner = (ROOT / "tests" / "canonical" / REFERENCE.runner).read_text(encoding="utf-8")
    block = re.search(r"estimators <- c\((.*?)\n\)", runner, re.S)
    assert block is not None
    pairs = dict(re.findall(r'(\w+) = "(\w+)"', block.group(1)))
    assert pairs == {scenario: spec[1] for scenario, spec in study.SCENARIOS.items()}


# ---------------------------------------------------------------------------- the fit


@pytest.fixture(scope="module")
def fitted() -> dict[str, tuple[pd.DataFrame, object]]:
    out = {}
    for law, labels in (("B", common.B_LABELS), ("B_atom", common.B_ATOM_LABELS)):
        frame = common.SAMPLERS[law](2_000, 11)
        out[law] = (frame, study.fit(frame, law, labels, outcome="correct"))
    return out


def _closed(frame: pd.DataFrame, label: str, node: int, law: str) -> np.ndarray:
    w1, w2, a1, l2, a2 = (frame[k].to_numpy() for k in ("W1", "W2", "A1", "L2", "A2"))
    u = common.upper(w2)
    if node == 1:
        return common.node_ratio(label, 1, a1, common.mean1(w1, w2), u)
    p = common.p_atom(w1, l2) if law == "B_atom" else None
    return common.node_ratio(label, 2, a2, common.mean2(w1, w2, a1, l2), u, p)


@pytest.mark.parametrize("label", ["p1", "p2", "p3", "p4", "p5"])
def test_the_oracle_ratio_matches_the_closed_form_to_the_binning(fitted, label: str) -> None:
    law = common.LAW_OF[label]
    frame, result = fitted[law]
    ratio = result.fits[label].node_ratio
    for node in (1, 2):
        gap = np.abs(ratio[:, node - 1] - _closed(frame, label, node, law))
        assert np.median(gap) < 5e-3


def test_the_atom_rows_read_the_mixed_ratio(fitted) -> None:
    frame, result = fitted["B_atom"]
    at = np.abs(frame["A2"].to_numpy() - common.ATOM) < 1e-12
    ratio = result.fits["p5"].node_ratio[:, 1]
    np.testing.assert_allclose(ratio[at], _closed(frame, "p5", 2, "B_atom")[at], rtol=0.02)
    assert ratio[at].mean() > 1.5


def _control_ratio(law: str, label: str, defect: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = common.SAMPLERS[law](2_000, 11)
    with properties.ratio_defect(defect):
        result = study.fit(frame, law, ("natural", label), outcome="wrong")
    return frame, result.fits[label].node_ratio


def test_the_jacobian_control_drops_the_slope_only() -> None:
    frame, ratio = _control_ratio("B", "p3", "jacobian")
    w1, w2, a = (frame[k].to_numpy() for k in ("W1", "W2", "A1"))
    u = common.upper(w2)
    mean = common.mean1(w1, w2)
    source = common.power_inverse(a, u)
    expected = common.tn_pdf(source, mean, u) / common.tn_pdf(a, mean, u)
    assert np.median(np.abs(ratio[:, 0] - expected)) < 5e-3
    declared = _closed(frame, "p3", 1, "B")
    assert np.median(np.abs(expected - declared)) > 0.05


def test_the_branch_control_reads_the_first_branch_only() -> None:
    frame, ratio = _control_ratio("B", "p4", "branch")
    w1, w2, a = (frame[k].to_numpy() for k in ("W1", "W2", "A1"))
    u = common.upper(w2)
    mean = common.mean1(w1, w2)
    first = common.tn_pdf(a - 1.0, mean, u) * (a - 1.0 < 3.0) / common.tn_pdf(a, mean, u)
    assert np.median(np.abs(ratio[:, 0] - first)) < 5e-3


def test_the_mass_control_reads_one_at_the_atom() -> None:
    frame, ratio = _control_ratio("B_atom", "p5", "atom")
    at = np.abs(frame["A2"].to_numpy() - common.ATOM) < 1e-12
    np.testing.assert_allclose(ratio[at, 1], 1.0)


def test_no_defect_outlives_its_context() -> None:
    from cleverly.interventions import policy
    from cleverly.learners import density

    with (
        properties.ratio_defect("branch"),
        pytest.raises(RuntimeError),
        properties.ratio_defect("jacobian"),
    ):
        pass
    assert policy._PieceBranch.ratio is properties._PIECE_RATIO
    assert density.MixedDensity.mass_between is properties._MASS_BETWEEN


# ---------------------------------------------------------------------------- learners


@pytest.mark.parametrize("label", ["natural", "p2", "p4", "p5"])
def test_the_rich_first_node_basis_recovers_the_regression(label: str) -> None:
    law = common.LAW_OF.get(label, "B")
    frame = common.SAMPLERS[law](20_000, 23)
    w1, w2, a1 = (frame[k].to_numpy() for k in ("W1", "W2", "A1"))
    exact = common.first_regression(label, law, w1, w2, a1)
    design = np.column_stack([w1, w2, a1])
    model = common.RichPseudo().fit(design, exact)
    error = np.abs(model.predict(design) - exact)
    # Far below the 0.01 sampling SE of the contrasts at n = 2,000.
    assert np.mean(error) < 1e-3
    assert np.quantile(error, 0.99) < 5e-3


@pytest.mark.parametrize("label", ["p1", "p3"])
def test_the_pinned_bounds_recompute(label: str) -> None:
    assert common.contrast_sd(label, 20_000) == pytest.approx(common.BOUNDS[label], rel=0.1)


def test_the_noise_control_reads_the_pinned_bounds() -> None:
    for label, (plan, _) in properties.CELLS.items():
        assert properties.EFFICIENCY_SD[label] == common.BOUNDS[plan]


def test_the_budget_is_recorded() -> None:
    budget = properties.BUDGET
    assert {"seconds", "cpu_hours", "peak_gib"} <= set(budget)
    assert budget["cpu_hours"]["python_total"] > 0


# ---------------------------------------------------------------------------- the gate


def test_the_gating_cells_are_published_and_hold_no_inadmissible_red() -> None:
    """Design review M7: the record publishes under ``reporting``, so this is its merge gate."""
    published = {
        f"{family}/{cell}" for family, cells in study.PROPERTY_CELLS.items() for cell in cells
    }
    assert set(properties.GATING_CELLS) <= published
    assert len(properties.GATING_CELLS) == 6 * 3 + 4 * 2
    assert not any("p4_sdr" in cell for cell in properties.GATING_CELLS)
    assert (study.STUDY.implementation,) == properties.GATING_TRUTH
    assert (
        tools.gating_findings(study.STUDY, properties.GATING_CELLS, properties.GATING_TRUTH) == []
    )


# ---------------------------------------------------------------------------- exactness


def _paired_rows(gap: float = 0.0, se_gap: float = 0.0) -> pd.DataFrame:
    """Two replications of every witness estimand, the reference moved by ``gap``."""
    records = []
    for scenario in sorted(study.EXACT):
        for name in study.names(study.SCENARIOS[scenario][2]):
            for replicate in (0, 1):
                for implementation, shift, se_shift in (
                    (study.STUDY.implementation, 0.0, 0.0),
                    ("lmtp", gap, se_gap),
                ):
                    records.append(
                        {
                            "implementation": implementation,
                            "scenario": scenario,
                            "replicate": replicate,
                            "estimand": name,
                            "estimate": 0.4 + shift,
                            "std_error": 0.01 + se_shift,
                        }
                    )
    return pd.DataFrame.from_records(records)


def test_the_exactness_gate_reads_every_witness_estimand() -> None:
    table = study.extra_artifacts(_paired_rows())["exactness.csv"]
    assert set(table["scenario"]) == set(study.EXACT)
    assert table["passed"].all()
    assert (table["pairs"] == 2).all()
    assert (table["gate"] == study.SMOKE_GATE).all()
    assert not len(study.scientific_failures({"exactness.csv": table})["exactness witness"])


@pytest.mark.parametrize(("gap", "se_gap"), [(2e-6, 0.0), (0.0, 2e-6)])
def test_a_paired_gap_at_the_gate_turns_the_witness_red(gap: float, se_gap: float) -> None:
    # The deliberate mutation: a reference moved by twice the gate in either column.
    table = study.extra_artifacts(_paired_rows(gap, se_gap))["exactness.csv"]
    assert not table["passed"].any()
    failed = study.scientific_failures({"exactness.csv": table})["exactness witness"]
    assert len(failed) == len(table)


def test_a_gap_below_the_gate_passes_and_an_unpaired_witness_does_not() -> None:
    assert study.extra_artifacts(_paired_rows(5e-7, 5e-7))["exactness.csv"]["passed"].all()
    rows = _paired_rows()
    lost = rows.loc[
        ~((rows["implementation"] == "lmtp") & (rows["scenario"] == study.BOUNDARIES_SDR))
    ]
    table = study.extra_artifacts(lost)["exactness.csv"]
    assert not table.loc[table["scenario"] == study.BOUNDARIES_SDR, "passed"].any()
    assert table.loc[table["scenario"] == study.BOUNDARIES_LTMLE, "passed"].all()


def _failure(scenarios: str, replicates: str, implementation: str = "lmtp") -> dict[str, object]:
    """One failure record, in the columns the regeneration writes."""
    return {
        "implementation": implementation,
        "group_id": f"{implementation}:{scenarios}",
        "scenarios": scenarios,
        "replicates": replicates,
        "attempt": 1,
        "class": "error",
        "stage": "fit",
        "call": "lmtp_tmle",
        "message": "synthetic",
        "seed": 0,
    }


def test_a_lost_witness_fit_turns_the_exactness_audit_red() -> None:
    # A real failure frame names its scenarios in the ``scenarios`` column, joined by ``;``.
    table = study.extra_artifacts(_paired_rows())["exactness.csv"]
    artifacts = {"exactness.csv": table}
    failures = pd.DataFrame.from_records(
        [
            _failure(f"{study.BOUNDARIES_LTMLE};{study.ATOM_LTMLE}", "3;4"),
            _failure(study.BOUNDARIES_LTMLE, "3", study.STUDY.implementation),
            _failure(study.ATOM_SDR, "7"),
        ]
    )
    audit = study.account_failures(failures, artifacts).set_index("leg")
    # Replicates 3 and 4 of one witness scenario; the atom fits and the repeated pair do not count.
    assert audit.loc["exactness witness", "failed"] == 2
    assert not audit.loc["exactness witness", "passed"]
    clean = study.account_failures(failures.iloc[:0], artifacts).set_index("leg")
    assert clean.loc["exactness witness", "passed"]


def test_the_atom_scenarios_are_not_witnesses() -> None:
    assert {study.BOUNDARIES_LTMLE, study.BOUNDARIES_SDR} == study.EXACT
    pairing = study.REFERENCE_METADATA["pairing"]
    assert all("default margins" in pairing[s] for s in (study.ATOM_LTMLE, study.ATOM_SDR))


def test_the_runner_reads_lmtp_bound_through_the_gate() -> None:
    runner = (ROOT / "tests" / "canonical" / "mtp_boundaries_runner.R").read_text(encoding="utf-8")
    assert "refuse_bound = FALSE" in runner
