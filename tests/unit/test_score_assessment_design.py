"""The declaration of the ``score-assessment`` study, before its run.

The truths are checked against an independent quasi-Monte Carlo integration and a different
solver, and against the closed form on law ``H``.  The design numbers the declaration fixes are
computed here, not copied: the calibration replication count by the pooled predictive rule at
the pilot's SE ratios, with the log-ratio risk-difference rows in a pool of their own; the pre-run
n rule that read the oracle reference; each bias row's second-order bias, checked against the
oracle, and the bias replication count it sets; each control's SE ratio or standardized
displacement and its detection probability; the paired Wald test's exact-law power and the pass
probabilities of the power and size cells; and the cost of one run (plan rev 2.2).  One replication of every group runs through the
declared property rows.  The run itself is not started here.
"""

from __future__ import annotations

import functools
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.special import expit

from tests.studies import conditional_projection as cp
from tests.studies import score_assessment as study
from tests.studies import score_assessment_pilot as pilot
from tests.studies import score_assessment_properties as properties
from tests.studies.evidence.seeds import stream_seed


@functools.cache
def uniform_w() -> np.ndarray:
    """Scrambled Sobol points on ``U(-1, 1)^3``, a different rule from the quadrature."""
    return 2.0 * stats.qmc.Sobol(3, scramble=True, seed=11).random_base2(18) - 1.0


@functools.cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


# ------------------------------------------------------------------------------ the truths


def test_the_difference_truths_match_an_independent_integration() -> None:
    w = uniform_w()
    tau = expit(cp.outcome_logit(w, 1)) - expit(cp.outcome_logit(w, 0))
    pair = study.SCORES["binary_paired"]
    s1, s2 = pair.first(w), pair.second(w)  # type: ignore[misc]
    truth = study.truth("observational", "difference", "binary_paired")
    phi = np.column_stack([np.ones(len(w)), s2])
    calibration = np.linalg.lstsq(phi, tau, rcond=None)[0]
    expected = {
        "S_risk_difference": float(np.mean(s1**2 - 2 * s1 * tau) + np.mean(tau) ** 2),
        "SB_calibration_intercept": float(calibration[0]),
        "SB_calibration_slope": float(calibration[1]),
        "paired": float(np.mean(s1**2 - s2**2 - 2 * (s1 - s2) * tau)),
    }
    for name, value in expected.items():
        assert truth[name] == pytest.approx(value, abs=2e-5), name


def test_the_log_ratio_truths_match_a_root_and_a_direct_risk() -> None:
    w = uniform_w()
    q_t, q_r = expit(cp.outcome_logit(w, 1)), expit(cp.outcome_logit(w, 0))
    pair = study.SCORES["crr_paired"]
    s1, s2 = pair.first(w), pair.second(w)  # type: ignore[misc]
    phi = np.column_stack([np.ones(len(w)), s1])

    def equation(beta: np.ndarray) -> np.ndarray:
        return phi.T @ ((q_t + q_r) * expit(phi @ beta) - q_t) / len(w)

    root = optimize.root(equation, np.zeros(2), tol=1e-14).x

    def risk(s: np.ndarray) -> float:
        return float(np.mean(q_t * np.log1p(np.exp(-s)) + q_r * np.log1p(np.exp(s))))

    truth = study.truth("observational", "log_ratio", "crr_paired")
    assert truth["S_calibration_intercept"] == pytest.approx(root[0], abs=2e-4)
    assert truth["S_calibration_slope"] == pytest.approx(root[1], abs=2e-4)
    assert truth["paired"] == pytest.approx(risk(s1) - risk(s2), abs=2e-5)
    c = math.log(np.mean(q_t) / np.mean(q_r))
    assert truth["S_risk_difference"] == pytest.approx(risk(s1) - risk(np.full(1, c)), abs=2e-5)


@pytest.mark.parametrize(
    ("kind", "paired"),
    [("cont_paired", -0.03), ("cont_equal", 0.0), ("cont_power", study.POWER_C**2 / 3.0)],
)
def test_the_law_h_paired_truths_have_their_closed_forms(kind: str, paired: float) -> None:
    assert study.truth("H", "difference", kind)["paired"] == pytest.approx(paired, abs=1e-14)


def test_the_equal_scores_have_both_curve_terms() -> None:
    """``S - tau = 0.5 W1`` and ``SB - tau = 0.5 W3``: the paired truth is 0 for a reason."""
    w = uniform_w()
    pair = study.SCORES["cont_equal"]
    tau = study.h_tau(w)
    np.testing.assert_allclose(pair.first(w) - tau, 0.5 * w[:, 0], atol=1e-14)
    np.testing.assert_allclose(pair.second(w) - tau, 0.5 * w[:, 2], atol=1e-14)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("law", "scale", "kind"),
    [
        ("observational", "difference", "binary_paired"),
        ("observational", "log_ratio", "crr_paired"),
        ("noweight_law", "difference", "dr"),
        ("noweight_law", "log_ratio", "crr_dr"),
    ],
)
def test_the_quadrature_error_is_negligible(law: str, scale: str, kind: str) -> None:
    assert study.quadrature_error(law, scale, kind) < 1e-10


def test_the_plan_paired_truths() -> None:
    truths = numbers()["truths"]
    assert truths["paired_2k"] == pytest.approx(0.0022, abs=5e-5)
    assert truths["paired_crr_4k"] == pytest.approx(0.0019, abs=5e-5)


# --------------------------------------------------------------------------- the record


def test_the_record_declares_its_gates() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.failure_bound == {record.implementation: properties.FAILURE_BOUND}
    assert (record.seed, record.resampling_seed) == (20261297, 20261298)
    assert record.property_artifacts == (properties.COMPARISON_FILE,)
    assert record.replicates == study.PRIMARY_REPLICATES >= 1_000


def test_every_group_declares_its_cells_once() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells))
    positives = [cell for cell in cells if cell.endswith(f"__{study.POSITIVE}")]
    assert len(positives) == 32
    assert study.PROPERTY_CELLS[study.TYPE_I] == (study.SIZE_CELL,)
    assert study.PROPERTY_CELLS[study.POWER] == (study.POWER_CELL,)
    assert len(study.PROPERTY_CELLS[study.BIAS]) == 12


def test_the_seed_streams_are_keyed_by_label() -> None:
    seeds = {
        stream_seed(study.STUDY, stream, group.label, 0)
        for group in study.GROUPS
        for stream in ("property_sample", pilot.STREAM)
    }
    assert len(seeds) == 2 * len(study.GROUPS)


def test_the_sizes_of_the_plan() -> None:
    sizes = {group.label: (group.n, group.replicates) for group in study.GROUPS}
    assert sizes["power_cont_2k"] == (2_000, 1_000)
    assert sizes["paired_dr_4k"] == sizes["paired_crr_dr_4k"] == (4_000, study.R_BIAS)
    calibrated = [group for group in study.GROUPS if group.calibration]
    assert {group.replicates for group in calibrated} == {study.R_CAL}
    assert {group.n for group in calibrated if group.scale == "difference"} == {2_000}
    # The n rule moved the log-ratio calibration groups to 4,000 (test_the_n_rule_moves...).
    assert {group.n for group in calibrated if group.scale == "log_ratio"} == {4_000}


# ------------------------------------------------------------------------------- sizing


def test_the_pilot_covers_every_calibration_row_on_its_own_stream() -> None:
    """Every calibration group, the n rule's 2,000-row log-ratio groups included, at one
    pilot length (plan rev 2.2, S3)."""
    rows = study.pilot_rows()
    for label in study.PILOT_GROUPS:
        group = study.PILOT_GROUP_BY_LABEL[label]
        for name in group.calibration:
            row = rows[(label, name)]
            assert int(row["replicates"]) + int(row["failed"]) == pilot.REPLICATES == 2_400
            assert int(row["failed"]) == 0
    assert pilot.STREAM != "property_sample"


def test_the_calibration_count_is_the_pooled_rule() -> None:
    assert numbers()["calibration_replicates"] == study.R_CAL
    assert numbers()["calibration_family"] >= 0.95
    rows = numbers()["log_ratio_row_pass"]
    assert len(rows) == 3 and min(rows.values()) >= study.LOG_RATIO_ROW_PASS


def test_the_log_ratio_risk_differences_pool_apart() -> None:
    """The rows that carry the best-constant term form their own sizing pool (rev 2.2, B2)."""
    pools: dict[str, list[str]] = {}
    for name in study.sizing_ratios():
        cell, pool = name.split("__")
        pools.setdefault(pool, []).append(cell)
    assert sorted(pools[study.LOG_RATIO_POOL]) == sorted(
        f"{group}_{label}" for group, label in study.LOG_RATIO_RD
    )
    assert len(pools[study.POSITIVE]) == 29


def test_the_n_rule_moved_the_log_ratio_groups_to_4k() -> None:
    """At 2,000 rows the pilot agrees with the oracle reference, but no R up to the cap meets
    the calibration rule, so both log-ratio calibration groups moved to 4,000 before the run."""
    rule = study.n_rule()
    assert rule["excess"] <= rule["interval"]
    assert rule["family_at_cap"] < 0.95
    assert rule["decision"] == "move"
    assert {group.label for group in study.N_RULE_GROUPS} == {"single_crr_2k", "paired_crr_2k"}
    assert {group.n for group in study.N_RULE_GROUPS} == {2_000}


def test_the_4k_pilot_agrees_with_the_oracle_reference() -> None:
    """At the moved n the pilot's excess is still the theory term, not a defect."""
    agreement = study.oracle_agreement("single_crr_4k", "paired_crr_4k")
    assert agreement["excess"] <= agreement["interval"]
    assert agreement["oracle"] > 1.0


def test_a_perfectly_calibrated_row_passes_at_the_count() -> None:
    assert numbers()["per_row_pass"] >= 0.999


def test_the_power_cell() -> None:
    assert numbers()["power"] == pytest.approx(0.904, abs=1e-3)
    assert numbers()["power_pass"] > 0.9999


def test_the_size_cell() -> None:
    """At a true size of 0.07 the 99% upper end stays at or below 0.10 almost surely."""
    assert numbers()["size_pass"] > 0.999


def test_each_positive_bias_row_passes_at_its_theory_bias() -> None:
    """Rev 2.2, B1: priced at the larger of the second-order and the oracle bias, each row
    passes with at least 0.99, and ``R_BIAS`` is the smallest multiple of 400 that achieves it."""
    biases = numbers()["theory_bias"]
    assert min(numbers()["bias_pass"].values()) >= 0.99
    assert numbers()["bias_replicates"] == study.R_BIAS == 2_000
    oracle = study.oracle_rows()[("paired_crr_dr_4k", "S_risk_difference")]
    group = study.GROUP_BY_LABEL["paired_crr_dr_4k"]
    priced = study.priced_bias(group, "S_risk_difference")
    assert (
        priced == float(oracle["standardized_bias"]) > biases["paired_crr_dr_4k_S_risk_difference"]
    )
    assert biases["paired_crr_dr_4k_S_risk_difference"] == pytest.approx(0.120, abs=5e-3)
    assert biases["paired_dr_4k_paired"] == biases["paired_crr_dr_4k_paired"] == 0.0
    priced_all = [study.priced_bias(g, label) for g in study.GROUPS for label in g.bias]
    fewer = [study.bias_pass_probability(b, study.R_BIAS - 400, control=False) for b in priced_all]
    assert min(fewer) < 0.99


def _oracle_rows() -> list[tuple[str, str]]:
    return [(group, label) for group, label in study.oracle_rows() if label != "paired"]


@pytest.mark.parametrize(("group", "label"), _oracle_rows())
def test_the_theory_bias_lies_in_the_oracle_interval(group: str, label: str) -> None:
    """The analytic second-order bias inside the oracle simulation's 99% interval."""
    row = study.oracle_rows()[(group, label)]
    predicted = study.theory_bias(study.PILOT_GROUP_BY_LABEL[group], label)
    half = float(stats.norm.ppf(0.995)) * float(row["standardized_bias_se"])
    assert abs(predicted - float(row["standardized_bias"])) <= half


def test_the_oracle_reference_does_not_share_the_estimator_code() -> None:
    from tests.studies import score_assessment_oracle as oracle

    source = Path(oracle.__file__).read_text()
    imported = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    assert "cleverly" not in imported
    assert "_one_step_estimates(" not in source
    row = study.oracle_rows()[("paired_crr_4k", "S_risk_difference")]
    assert int(row["replicates"]) == oracle.REPLICATES


# ----------------------------------------------------------------------------- controls


@pytest.mark.parametrize(
    ("cell", "ratio"),
    [
        ("paired_2k_paired__inflated_se_control", 3.03),
        ("paired_cont_2k_paired__inflated_se_control", 4.40),
        ("equal_cont_2k_paired__term_dropped_control", 0.246),
    ],
)
def test_each_se_control_leaves_the_band(cell: str, ratio: float) -> None:
    control = numbers()["controls"][cell]
    assert control["ratio"] == pytest.approx(ratio, abs=5e-3)
    assert control["detection"] > 0.9999


def test_the_pilot_reads_the_se_controls_where_the_design_puts_them() -> None:
    """The pilot's control ratios sit on the design's side of the band and detect too."""
    rows = study.pilot_rows()
    for label, control in (
        ("paired_2k", "unpaired_se"),
        ("paired_cont_2k", "unpaired_se"),
        ("equal_cont_2k", "no_residual_se"),
    ):
        group = study.GROUP_BY_LABEL[label]
        piloted = float(rows[(label, "paired")][f"{control}_ratio"])
        design = numbers()["controls"][study.control_cell(group, control)]["ratio"]
        assert (piloted - 1.0) * (design - 1.0) > 0.0
        assert piloted == pytest.approx(design, rel=0.15)
        assert study.se_control_detection(piloted, group.replicates) > 0.9999


@pytest.mark.parametrize(("group", "displacement"), [("dr_4k", -7.2), ("crr_dr_4k", -4.1)])
def test_the_untargeted_paired_displacements(group: str, displacement: float) -> None:
    control = numbers()["controls"][f"untargeted_{group}_paired"]
    assert control["standardized"] == pytest.approx(displacement, abs=0.05)
    assert control["detection"] > 0.9999


def test_every_untargeted_row_detects() -> None:
    controls = numbers()["controls"]
    rows = [cell for cell in study.PROPERTY_CELLS[study.BIAS] if cell.startswith("untargeted_")]
    assert len(rows) == 6
    assert min(controls[cell]["detection"] for cell in rows) > 0.9999


def test_the_untargeted_control_reads_the_declared_regression() -> None:
    """Witness: the plug-in at the fit's initial regression is the declared one's plug-in."""
    group = study.GROUP_BY_LABEL["paired_dr_4k"]
    frame, _ = study.draw_from_seed(group.label, group.n, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = study.fit_group(group, frame, random_state=0)
    w = frame.loc[:, ["W1", "W2", "W3"]].to_numpy()
    m_t = expit(cp.outcome_logit(w, 1) + cp.declared_error("noweight", w, 1))
    m_r = expit(cp.outcome_logit(w, 0) + cp.declared_error("noweight", w, 0))
    pair = study.SCORES["dr"]
    s1, s2 = pair.first(w), pair.second(w)  # type: ignore[misc]
    expected = float(np.mean(s1**2 - s2**2 - 2.0 * (s1 - s2) * (m_t - m_r)))
    assert fitted["plug_in"]["paired"] == pytest.approx(expected, abs=1e-6)
    assert abs(fitted["plug_in"]["paired"] - float(fitted["estimates"]["paired"].psi)) > 1e-3


# --------------------------------------------------------------------------------- cost


def test_one_run_fits_the_budget() -> None:
    """Plan 5.1: 8 CPU-hours is twice the estimate; one run must stay under it."""
    cost = numbers()["cost"]
    assert cost["cpu_hours"] < 8.0
    assert cost["wall_hours_4"] < 2.0


def test_the_declared_seconds_are_the_pilot_seconds() -> None:
    rows = study.pilot_rows()
    for group in study.GROUPS:
        if group.calibration:
            measured = float(rows[(group.label, "S_risk_difference")]["seconds"])
            assert group.seconds == pytest.approx(measured, abs=0.01)


# ------------------------------------------------------------------------ one replication


@pytest.mark.parametrize("label", [group.label for group in study.GROUPS])
def test_one_replication_fills_its_declared_cells(label: str) -> None:
    group = study.GROUP_BY_LABEL[label]
    drawn = properties._draw(label, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows, records = properties._replicate(label, 0, drawn)
    found = {f"{row['property']}/{row['cell']}" for row in rows}
    assert found == set(properties._cells_of((label, 0)))
    assert all(np.isfinite(row["estimate"]) for row in rows)
    assert {row["n"] for row in rows} == {group.n}
    table = properties.comparison(pd.DataFrame(records))
    assert set(table["parameter"]) == set(study.parameter_labels(group.paired))
    assert {"reduced_fits", "reduced_coverage"} <= set(table.columns)


def test_the_private_twin_returns_the_public_assessment() -> None:
    group = study.GROUP_BY_LABEL["paired_crr_4k"]
    frame = study.with_scores(group, study.draw_from_seed(group.label, group.n, 3)[0])
    from cleverly import CausalStudy, ConditionalContrast

    effect = CausalStudy(frame, design=study.design(group)).identify(
        ConditionalContrast(study.MODIFIERS, scale="log_ratio", model=study.BASIS)
    )
    settings = {**study.learners(group), "random_state": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        public = effect.assess_score(
            "S",
            computed_from=["W1"],
            score_kind="external",
            compare="SB",
            compare_computed_from=["W1", "W3"],
            **settings,
        )
    fitted = study.fit_group(group, frame.drop(columns=["S", "SB"]), random_state=0)
    for score, name, estimate in public.rows():
        label = study._label(score, name)
        assert float(fitted["estimates"][label].psi) == float(estimate.psi)
        assert float(fitted["estimates"][label].std_error) == float(estimate.std_error)


def test_the_manifest_will_hash_every_study_module() -> None:
    """The provenance rule a registered study meets, checked before registration."""
    from tests.unit.test_study_provenance import _required_from

    record = study.STUDY
    required = _required_from(record.runner_module, record.properties_module)
    assert required <= set(record.modules)


def test_the_cell_descriptions_name_the_study_groups() -> None:
    """The evidence descriptions word each group's law; a renamed or added group must reach them."""
    from tests.studies.evidence.descriptions import _SCORE_GROUPS

    described = {label: paired for label, (_, paired) in _SCORE_GROUPS.items()}
    declared = {
        label: group.paired
        for label, group in study.GROUP_BY_LABEL.items()
        if label != "power_cont_2k"
    }
    assert described == declared
