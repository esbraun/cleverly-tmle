"""The declaration of the ``conditional-projection`` study, before its run.

The truths are checked against an independent quasi-Monte Carlo integration and a different
solver, and against the closed form where one exists.  The design numbers the declaration
fixes are computed here, not copied: each calibration row's pass probability at its ``R``, the
family-wise rate, the heterogeneity power and the pass probabilities of the power and type-I
cells, each control's detection probability, and the cost of one run.  One replication of the
cheap groups runs through the declared property rows.  The run itself is not started here.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest
from scipy import optimize, stats
from scipy.special import expit

from tests.studies import conditional_projection as study
from tests.studies import conditional_projection_properties as properties
from tests.studies.evidence.seeds import stream_seed


@functools.cache
def sobol(dimension: int, power: int = 18) -> np.ndarray:
    """Scrambled Sobol points on ``[0, 1)^dimension``, a different rule from the quadrature."""
    return stats.qmc.Sobol(dimension, scramble=True, seed=7).random_base2(power)


def uniform_w() -> np.ndarray:
    return 2.0 * sobol(3) - 1.0


def least_squares(phi: np.ndarray, values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(phi * np.sqrt(weight)[:, None], np.sqrt(weight) * values)[0]


def coefficients(truth: dict[str, float], stem: str) -> np.ndarray:
    return np.array([truth[f"{stem}[{term}]"] for term in study.TERMS])


@pytest.mark.parametrize("law", ["observational", "weighted"])
def test_the_difference_truths_match_an_independent_integration(law: str) -> None:
    w = uniform_w()
    tau = expit(study.outcome_logit(w, 1)) - expit(study.outcome_logit(w, 0))
    weight = study.sampling_weight(w) if law == "weighted" else np.ones(len(w))
    estimate = least_squares(study.design_matrix(w), tau, weight)
    np.testing.assert_allclose(
        coefficients(study.truth(law), "cate_projection"), estimate, atol=2e-4
    )


def test_the_log_ratio_truth_matches_a_root_of_its_population_equation() -> None:
    w = uniform_w()
    q_t, q_r = expit(study.outcome_logit(w, 1)), expit(study.outcome_logit(w, 0))
    phi = study.design_matrix(w)

    def equation(beta: np.ndarray) -> np.ndarray:
        return phi.T @ ((q_t + q_r) * expit(phi @ beta) - q_t) / len(w)

    root = optimize.root(equation, np.zeros(3), tol=1e-14).x
    np.testing.assert_allclose(
        coefficients(study.truth("crr", "log_ratio"), "log_crr_projection"), root, atol=2e-4
    )


def test_the_clustered_truth_integrates_the_cluster_effect_out() -> None:
    points = sobol(4)
    w, u = 2.0 * points[:, :3] - 1.0, stats.norm.ppf(points[:, 3])
    tau = expit(study.outcome_logit(w, 1) + u) - expit(study.outcome_logit(w, 0) + u)
    estimate = least_squares(study.design_matrix(w), tau, np.ones(len(w)))
    np.testing.assert_allclose(
        coefficients(study.truth("clustered"), "cate_projection"), estimate, atol=5e-4
    )


def test_the_assess_truths_match_an_independent_integration() -> None:
    w = uniform_w()
    s = study.score(w)
    tau = expit(study.outcome_logit(w, 1)) - expit(study.outcome_logit(w, 0))
    risk = float(np.mean(s**2 - 2.0 * s * tau) + np.mean(tau) ** 2)
    calibration = least_squares(np.column_stack([np.ones(len(w)), s]), tau, np.ones(len(w)))
    truth = study.truth("assess")
    np.testing.assert_allclose(
        [truth["risk_difference"], truth["calibration[(intercept)]"], truth["calibration[S]"]],
        [risk, *calibration],
        atol=2e-4,
    )


def test_the_heterogeneity_law_has_its_closed_form_truth() -> None:
    truth = study.truth("hetero_trial", slope=0.10)
    np.testing.assert_allclose(coefficients(truth, "cate_projection"), [0.2, 0.1, 0.0], atol=1e-12)


@pytest.mark.parametrize(("law", "scale"), [("observational", "difference"), ("crr", "log_ratio")])
def test_the_quadrature_error_is_negligible(law: str, scale: str) -> None:
    assert study.quadrature_error(law, scale) < 1e-10


def test_the_projection_is_genuine() -> None:
    """The CATE is not linear in W3, so the W3 slope is a projection, not a coefficient."""
    w, mass = study._grid()
    q_t, q_r = study.arm_means("observational", w)
    phi = study.design_matrix(w)
    beta = study.projection(phi, mass, q_t, q_r, "difference")
    residual = q_t - q_r - phi @ beta
    assert float(np.sqrt(mass @ residual**2)) > 0.01


def test_the_weight_moves_the_estimand() -> None:
    """Design review B4: the weighted truth is about half an SD from the unweighted one."""
    shift = coefficients(study.truth("weighted"), "cate_projection") - coefficients(
        study.truth("observational"), "cate_projection"
    )
    sd = np.sqrt(np.diag(study.efficient_covariance("observational", 2_000)))
    assert abs(shift[0] / sd[0]) > 0.45


def test_the_ranges_of_the_mechanisms() -> None:
    w, _ = study._grid(16)
    assert study.propensity(w).min() > 0.23 and study.propensity(w).max() < 0.77
    pi = np.concatenate([study.observation(w, 0), study.observation(w, 1)])
    assert pi.min() > 0.59 and pi.max() < 0.94
    assert np.ptp(study.observation(w, 0)) > 0.2


# ------------------------------------------------------------------- the declaration


def test_the_record_declares_its_gates() -> None:
    record = study.STUDY
    assert (
        record.failure_bound
        == {record.implementation: properties.FAILURE_BOUND}
        == {"cleverly-projection-tmle": 0.01}
    )
    assert record.publication_policy == "gated"
    assert record.margins.calibration_se_ratio == (0.93, 1.07)
    assert record.margins.standardized_bias == 0.25
    assert record.bias_row_families == {"bias"}
    assert dict(record.property_cells) == study.PROPERTY_CELLS


def test_every_group_declares_its_cells_once() -> None:
    labels = [group.label for group in study.GROUPS]
    assert len(labels) == len(set(labels))
    cells = [(family, cell) for family, names in study.PROPERTY_CELLS.items() for cell in names]
    assert len(cells) == len(set(cells))
    assert {group.law for group in study.GROUPS} <= set(study.LAWS)
    assert set(study.BIAS_ROLES) == set(study.PROPERTY_CELLS["bias"])


def test_the_controls_share_their_groups_fits() -> None:
    """Design review S2: each control reads its group's draws, and names them."""
    shared = {(control, group.label) for group in study.GROUPS for control in group.controls}
    assert shared == {
        ("weights_ignored", "weighted_sl_2k"),
        ("iid_se", "clustered_4k"),
        ("ipw", "slow_1k"),
        ("t_learner", "slow_4k"),
        ("ipw", "slow_4k"),
        ("ipw", "slow_16k"),
        ("no_weight", "noweight_dr_4k"),
        ("missing_ignored", "missing_dr_4k"),
    }
    assert set(study.CONTROL_CLAIMS) == {control for control, _ in shared}


def test_the_seed_streams_are_keyed_by_label() -> None:
    """Reordering the groups leaves every draw where it was."""
    seeds = {
        group.label: stream_seed(study.STUDY, "property_sample", group.label, 0)
        for group in study.GROUPS
    }
    assert len(set(seeds.values())) == len(seeds)
    assert seeds["obs_1k"] == stream_seed(study.STUDY, "property_sample", "obs_1k", 0)


def test_the_sizes_the_review_asked_for() -> None:
    by = study.GROUP_BY_LABEL
    glm = [g for g in study.GROUPS if g.calibration and g.learner in ("spline", "slow")]
    assert {g.replicates for g in glm} == {4_800}
    assert by["weighted_sl_2k"].replicates == 3_600 and by["weighted_sl_2k"].folds == 5
    assert study.learners(by["weighted_sl_2k"])["outcome_learner"].n_folds == 5


# ---------------------------------------------------------------- the design numbers


@pytest.fixture(scope="module")
def numbers() -> dict:
    return study.design_numbers()


def test_each_calibration_row_passes_at_its_size(numbers: dict) -> None:
    for cell, row in numbers["calibration"].items():
        assert row["pass_at_1"] > 0.999, cell
        assert row["pass_at_0.97"] > (0.80 if row["replicates"] == 3_600 else 0.90), cell


def test_the_family_wise_rate_at_calibration(numbers: dict) -> None:
    """Re-review RB2: the ladder carries no calibration rows, so 11 groups give 33."""
    assert len(numbers["calibration"]) == 33
    assert not any(cell.startswith("slow_") for cell in numbers["calibration"])
    assert numbers["calibration_family"] > 0.99


def test_the_family_wise_rate_at_the_piloted_ratios(numbers: dict) -> None:
    """Re-review RB3: the pass probabilities at the pilot's SE ratios, read two ways.

    Plugged in row by row, the pilot's noise (SD about 0.015 per GLM row at 2,400, about 0.023
    per ensemble row at 960) counts as miscalibration.  The rows of each learner spread no
    more than that noise (Cochran's Q), so the pooled reading shrinks each row to its
    learner's common ratio.  Both numbers are recorded here; the declaration is unchanged.

    The ``assess_score_2k`` rows are the TMLE score assessment of X97, piloted at 2,400 draws
    on the same stream for the fresh declaration (they held the one-step's ratios before).  At
    ``R_GLM = 4,800`` the pooled family reads 0.9617 (0.962 with the one-step rows too), so
    ``R_GLM`` stays.
    """
    for cell, row in numbers["calibration"].items():
        assert 0.0 <= row["pass_at_pilot"] <= 1.0, cell
        assert row["pass_at_pooled"] > 0.98, cell
    for learner, test in numbers["pilot_homogeneity"].items():
        assert test["p_value"] > 0.5 and test["tau2"] == 0.0, learner
    assert numbers["pilot_homogeneity"]["spline"]["common_ratio"] == pytest.approx(1.0017, abs=1e-4)
    assert numbers["pilot_homogeneity"]["ensemble"]["common_ratio"] == pytest.approx(
        1.0120, abs=1e-4
    )
    assert numbers["calibration_family_pilot"] == pytest.approx(0.7792, abs=1e-4)
    assert numbers["calibration_family_pooled"] == pytest.approx(0.9617, abs=1e-4)
    assess = {
        cell: row["pilot_ratio"]
        for cell, row in numbers["calibration"].items()
        if cell.startswith("assess_score_2k_")
    }
    assert assess == pytest.approx(
        {
            "assess_score_2k_risk_difference__correctly_specified": 1.00854,
            "assess_score_2k_calibration_intercept__correctly_specified": 1.00609,
            "assess_score_2k_calibration_S__correctly_specified": 1.00494,
        },
        abs=1e-5,
    )
    assert numbers["calibration_family_pooled"] >= 0.95


def test_the_pilot_covers_every_calibration_row_on_its_own_stream() -> None:
    """Re-review RB3: the pilot is complete, and its stream is not the run's."""
    from tests.studies import conditional_projection_pilot as pilot

    rows = study.pilot_rows()
    for group, _, name in study.calibration_parameters():
        row = rows[(group.label, name)]
        size = pilot.ENSEMBLE_REPLICATES if group.learner == "ensemble" else pilot.REPLICATES
        assert row["replicates"] + row["failed"] == size, (group.label, name)
        assert row["failed"] <= 0.01 * size, (group.label, name)
    assert pilot.STREAM != "property_sample"
    assert stream_seed(study.STUDY, pilot.STREAM, "obs_1k", 0) != stream_seed(
        study.STUDY, "property_sample", "obs_1k", 0
    )


def test_the_ladder_bias_rows(numbers: dict) -> None:
    """Design review S1: the pilot's 0.161 passes at R = 4,800, and 0.19 still mostly does."""
    assert numbers["slow_1k_bias"]["pilot"] > 0.999
    assert numbers["slow_1k_bias"]["at_0.19"] > 0.9


def test_the_heterogeneity_power_and_its_pass(numbers: dict) -> None:
    assert numbers["power"] == pytest.approx(0.9326, abs=5e-4)
    assert numbers["noncentrality"] == pytest.approx(14.27, abs=0.01)
    assert numbers["power_pass"] > 0.9999
    assert study.heterogeneity_power(2_000, 0.10)[0] == pytest.approx(0.664, abs=1e-3)
    assert numbers["type_i_pass"] > 0.9999


def test_each_control_detects(numbers: dict) -> None:
    controls = numbers["controls"]
    declared = {
        cell
        for cell in (*study.PROPERTY_CELLS["bias"], *study.PROPERTY_CELLS["interval_calibration"])
        if study.BIAS_ROLES.get(cell) == "control" or cell.endswith(study.ROW_VARIANCE)
    }
    assert set(controls) == declared
    for cell, row in controls.items():
        assert row["detection"] > 0.99, (cell, row)


def test_the_iid_se_control_publishes_only_the_terms_it_can_see() -> None:
    """Re-review RB1: ``W3`` varies within clusters, so its row-level SE would not detect."""
    group = study.GROUP_BY_LABEL["clustered_4k"]
    assert group.controls["iid_se"] == ("(intercept)", "W1")
    rows = study.pilot_rows()
    w3 = rows[("clustered_4k", "cate_projection[W3]")]["iid_se_ratio"]
    assert study.row_variance_detection(w3, group.replicates) < 0.01
    for term in group.controls["iid_se"]:
        ratio = rows[("clustered_4k", f"cate_projection[{term}]")]["iid_se_ratio"]
        assert study.row_variance_detection(ratio, group.replicates) > 0.99


def test_the_t_learner_and_ipw_numbers_of_the_plan(numbers: dict) -> None:
    """The plan's 2.29 for the T-learner, and the IPW reading on its own SD (review S3)."""
    controls = numbers["controls"]
    assert controls["t_learner_4k_W1"]["standardized"] == pytest.approx(2.29, abs=0.01)
    assert 0.41 < abs(controls["ipw_1k_W1"]["standardized"]) < 0.46


def test_one_run_fits_in_a_day_on_sixteen_cores(numbers: dict) -> None:
    assert numbers["cost"]["cpu_hours"] < 60.0
    assert numbers["cost"]["wall_hours_16"] < 24.0
    assert numbers["cost"]["wall_hours_4"] < 16.0


# -------------------------------------------------------------------- one replication


@pytest.mark.parametrize(
    "label", ["obs_1k", "clustered_4k", "slow_4k", "noweight_dr_4k", "missing_dr_4k", "null_2k"]
)
def test_one_replication_fills_its_declared_cells(label: str) -> None:
    group = study.GROUP_BY_LABEL[label]
    drawn = properties._draw(label, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties._replicate(label, 0, drawn)
    found = {f"{row['property']}/{row['cell']}" for row in rows}
    assert found == set(properties._cells_of((label, 0)))
    assert all(np.isfinite(row["estimate"]) for row in rows)
    assert {row["n"] for row in rows} == {group.n}


def test_the_manifest_will_hash_every_study_module() -> None:
    """The provenance rule a registered study meets, checked before registration."""
    from tests.unit.test_study_provenance import _required_from

    record = study.STUDY
    required = _required_from(record.runner_module, record.properties_module)
    assert required <= set(record.modules)
