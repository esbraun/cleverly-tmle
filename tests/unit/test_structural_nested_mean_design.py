"""The declared design of the study ``structural-nested-mean`` (X44).

Every number here is fixed before any registered run.  The test checks eight conditions:

(a) each positive setup satisfies its structural model, by the g-formula, and at the declared
    ``GLM`` order every nuisance regression of every setup returns its exact cell mean;
(b) each negative control misses the regimen truth by a computed amount, and the detection rule
    (the bias interval lies beyond the margin on the side of the gap) fires with chance at
    least 0.95 at the declared replicates;
(c) the replicates of the calibration cells follow the arm-pooled predictive rule: the chance
    that all 61 cells pass, with the SE ratio, the coverage and the bias leg, is at least 0.95
    when the true SE ratio of each arm is drawn from its pilot;
(d) the ladder rungs clear their coverage floor, and each rung stays under the failure bound;
(e) the model-check size cells and the power cell, with the exact-law power from the influence
    curves;
(f) the gated direction of the unclustered standard error;
(g) the declared red set is empty;
(h) the budget in CPU-hours.

The cheap conditions are recomputed here.  The pilots are pinned below.  Each pilot fits the
shipped estimator on draws of the declared laws; the pinned values are the totals the design
reads.  A pilot of one configuration pools the disjoint runs of that configuration.  No test
here reads a run artifact, and no ``passed`` column exists.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import beta, binom, chi2, norm

import cleverly.longitudinal.structural as snmm_module
from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean
from tests import discrete_law_structural as finite
from tests.studies import canonical_structural_nested_mean as study
from tests.studies import structural_nested_mean_laws as laws
from tests.studies import structural_nested_mean_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import descriptions
from tests.studies.evidence.property_verdicts import MINIMUM_POWER, power_cell_pass_probability
from tests.studies.evidence.registry import Margins, registered

#: The tolerance of the exact-law checks.
EXACT = 1e-8
label_of = study.label_of

# ------------------------------------------------------------------------- pinned pilots

#: ``label -> (draws, SE ratio, relative SD of the reported SE, standardized bias, coverage,
#: empirical SD times sqrt(n))`` of the calibration pilot, n = 2,000.  Draws pool the disjoint
#: pilots of one configuration.
PILOT: dict[str, tuple[int, float, float, float, float, float]] = {
    "s2_blip_a1_l1": (9000, 1.0002, 0.0180, -0.0131, 0.9500, 1.3158),
    "s2_blip_a2_l1": (9000, 0.9766, 0.0162, -0.0071, 0.9434, 1.1574),
    "s2_ey0": (9000, 0.9852, 0.0199, 0.0225, 0.9456, 0.9717),
    "s2_ey_r11": (9000, 0.9911, 0.0235, -0.0031, 0.9478, 1.0918),
    "s2_ate_r11": (9000, 0.9959, 0.0186, -0.0162, 0.9470, 1.5459),
    "s2_ey_r10": (9000, 0.9816, 0.0237, 0.0037, 0.9437, 1.2473),
    "s3_blip_a1_l1": (3000, 0.9956, 0.0198, 0.0055, 0.9510, 1.3945),
    "s3_blip_a2_l1": (3000, 1.0037, 0.0206, -0.0148, 0.9517, 1.3242),
    "s3_blip_a3_l1": (3000, 1.0007, 0.0205, 0.0017, 0.9507, 1.2032),
    "s3_ey0": (3000, 0.9972, 0.0210, 0.0170, 0.9490, 0.9960),
    "s3_ate_r111": (3000, 1.0050, 0.0201, -0.0059, 0.9523, 1.6515),
    "s3_ate_r101": (3000, 1.0016, 0.0210, 0.0060, 0.9457, 1.6592),
    "k_blip_a1_l1": (3000, 0.9828, 0.0179, 0.0011, 0.9410, 1.4786),
    "k_blip_a1_l2": (3000, 0.9963, 0.0212, -0.0045, 0.9490, 1.5364),
    "k_blip_a2_l1": (3000, 1.0072, 0.0191, -0.0210, 0.9523, 1.3067),
    "k_blip_a2_l2": (3000, 1.0013, 0.0178, 0.0107, 0.9523, 1.2683),
    "k_ey0": (3000, 0.9759, 0.0224, 0.0202, 0.9417, 1.1339),
    "k_ey_r12": (3000, 0.9913, 0.0255, 0.0260, 0.9457, 1.4463),
    "k_ate_r12": (3000, 0.9948, 0.0202, 0.0079, 0.9467, 1.9280),
    "k_ate_r21": (3000, 0.9912, 0.0217, -0.0177, 0.9493, 1.9646),
    "c_blip_a1": (3000, 0.9990, 0.0202, 0.0144, 0.9487, 0.8128),
    "c_blip_a2": (3000, 0.9813, 0.0179, -0.0192, 0.9467, 0.6931),
    "c_ey0": (3000, 0.9994, 0.0152, 0.0034, 0.9570, 0.6959),
    "c_ey_dose_mid": (3000, 1.0133, 0.0169, 0.0070, 0.9537, 0.7274),
    "c_ate_dose_mid": (3000, 0.9995, 0.0207, 0.0060, 0.9487, 0.4259),
    "c_ate_dose_high": (3000, 0.9936, 0.0214, 0.0199, 0.9483, 0.9143),
    "h_blip_a1_l1": (3000, 1.0140, 0.0291, -0.0119, 0.9530, 2.0530),
    "h_blip_a1_l1_w": (3000, 1.0033, 0.0187, -0.0072, 0.9527, 2.6718),
    "h_blip_a2_l1": (3000, 0.9988, 0.0249, -0.0082, 0.9483, 1.6787),
    "h_blip_a2_l1_w": (3000, 0.9971, 0.0185, -0.0083, 0.9503, 2.3169),
    "h_blip_a2_l1_a1": (3000, 0.9988, 0.0177, -0.0127, 0.9523, 2.2954),
    "h_ey0": (3000, 0.9853, 0.0239, 0.0541, 0.9483, 1.0394),
    "h_ey_r11": (3000, 0.9915, 0.0258, -0.0095, 0.9460, 1.4078),
    "h_ate_r11": (3000, 0.9928, 0.0195, -0.0418, 0.9483, 1.6654),
    "h_ate_r10": (3000, 0.9925, 0.0244, -0.0140, 0.9497, 1.7748),
    "p_blip_a1_l1": (9000, 0.9974, 0.0264, 0.0173, 0.9484, 1.7918),
    "p_blip_a2_l1": (9000, 0.9933, 0.0234, 0.0197, 0.9477, 1.4875),
    "p_blip_a2_l1_a1": (9000, 0.9993, 0.0178, -0.0396, 0.9477, 2.2981),
    "p_ey0": (9000, 0.9887, 0.0232, 0.0077, 0.9474, 1.0318),
    "p_ate_r11": (9000, 0.9947, 0.0192, -0.0196, 0.9486, 1.5542),
    "p_ey_r01": (9000, 0.9934, 0.0300, 0.0294, 0.9456, 1.2680),
    "m_blip_a1_l1": (8995, 0.9999, 0.0175, -0.0001, 0.9492, 1.3032),
    "m_blip_a2_l1": (8995, 0.9818, 0.0250, -0.0050, 0.9437, 1.3563),
    "m_ey0": (8995, 0.9941, 0.0182, 0.0196, 0.9479, 1.3374),
    "m_ate_r11": (8995, 0.9875, 0.0226, -0.0034, 0.9476, 1.7255),
    "l_blip_a1_l1": (9400, 0.9973, 0.0226, -0.0069, 0.9497, 1.4007),
    "l_blip_a2_l1": (9400, 0.9894, 0.0257, 0.0083, 0.9465, 1.5949),
    "l_blip_a2_l1_s2": (9400, 0.9918, 0.0162, -0.0175, 0.9473, 2.2809),
    "l_ey0": (9400, 0.9868, 0.0205, 0.0225, 0.9454, 0.9706),
    "n_blip_a1_l1": (9000, 1.0113, 0.0207, -0.0545, 0.9542, 1.4031),
    "n_blip_a2_l1": (9000, 1.0478, 0.0165, 0.0997, 0.9596, 1.2001),
    "n_ey0": (9000, 1.0190, 0.0192, -0.0016, 0.9524, 1.3383),
    "n_ate_r11": (9000, 1.0400, 0.0212, 0.0249, 0.9567, 1.7325),
    "s2w_blip_a1_l1": (10500, 0.9930, 0.0239, -0.0083, 0.9488, 1.5830),
    "s2w_ey0": (10500, 0.9927, 0.0296, 0.0107, 0.9494, 1.1960),
    "s2w_ey_r11": (10500, 0.9813, 0.0306, -0.0106, 0.9416, 1.2541),
    "s2w_ate_r11": (10500, 0.9921, 0.0253, -0.0141, 0.9493, 1.8567),
    "s2c_blip_a1_l1": (9000, 0.9781, 0.0797, 0.0107, 0.9447, 2.5890),
    "s2c_ey0": (9000, 0.9986, 0.0597, -0.0148, 0.9478, 3.5550),
    "s2c_ey_r11": (9000, 0.9982, 0.0629, 0.0040, 0.9489, 3.6877),
    "s2c_ate_r11": (9000, 0.9808, 0.0808, 0.0218, 0.9433, 3.1175),
}
#: ``setup -> (seconds per fit, draws, recordable failures)`` of the same pilot.
PILOT_COST: dict[str, tuple[float, int, int]] = {
    "S2": (0.0872, 9000, 0),
    "S3": (0.2971, 3000, 0),
    "K": (0.1406, 3000, 0),
    "C": (0.0452, 3000, 0),
    "H": (0.0905, 3000, 0),
    "P": (0.0946, 9000, 0),
    "M": (0.0875, 9000, 5),
    "L": (0.0830, 9400, 0),
    "N": (0.4135, 9000, 0),
    "S2W": (0.0774, 10500, 0),
    "S2C": (0.0810, 9000, 0),
}
#: ``label -> (draws, standardized bias, empirical SD times sqrt(n), SE ratio)`` of the
#: negative controls at n = 2,000 and the order of ``INTERACTIONS``.
PILOT_CONTROLS: dict[str, tuple[int, float, float, float]] = {
    "lu_ey_r11": (600, -0.8862, 1.3457, 1.0076),
    "lu_ate_r10": (600, 4.7763, 1.5411, 1.0438),
    "lu_ate_r01": (600, 4.8778, 1.2506, 0.9865),
    "vb_ey_r11": (600, -11.0708, 1.3064, 1.0147),
    "vb_ate_r11": (600, -2.3001, 1.7717, 0.9971),
    "vb_ate_r10": (600, 21.9385, 1.6066, 1.0143),
    "vb_ate_r01": (600, 18.2406, 1.5220, 0.9551),
}
#: ``setup -> (seconds per fit, draws, recordable failures)`` of the control pilot.
PILOT_CONTROL_COST: dict[str, tuple[float, int, int]] = {
    "LU": (0.0764, 600, 0),
    "VB": (0.0781, 600, 0),
}
#: ``(label, n) -> (draws, failures, seconds per fit, SE ratio, noise, bias, coverage, SD times sqrt(n))`` of the
#: ladder estimands.  A rung at n = 2,000 reads the calibration pilot.
PILOT_LADDER: dict[tuple[str, int], tuple[int, int, float, float, float, float, float, float]] = {
    ("s2_ate_r11", 500): (600, 0, 0.0438, 1.0125, 0.0374, -0.0654, 0.9417, 1.5218),
    ("s2_ate_r11", 2000): (9000, 0, 0.0872, 0.9959, 0.0186, -0.0162, 0.9470, 1.5459),
    ("s2_ate_r11", 8000): (600, 0, 0.1983, 1.0379, 0.0095, -0.0748, 0.9583, 1.4843),
    ("h_ate_r11", 500): (600, 0, 0.0535, 1.0049, 0.0388, -0.0416, 0.9483, 1.6469),
    ("h_ate_r11", 2000): (3000, 0, 0.0905, 0.9928, 0.0195, -0.0418, 0.9483, 1.6654),
    ("h_ate_r11", 8000): (600, 0, 0.2236, 1.0166, 0.0100, 0.0143, 0.9533, 1.6276),
    ("k_ate_r12", 1000): (3000, 0, 0.1012, 0.9833, 0.0285, -0.0217, 0.9403, 1.9489),
    ("k_ate_r12", 4000): (600, 0, 0.2182, 1.0047, 0.0142, -0.0618, 0.9467, 1.9078),
    ("k_ate_r12", 16000): (600, 0, 0.8382, 1.0049, 0.0073, -0.0207, 0.9583, 1.9124),
    ("c_ate_dose_high", 500): (600, 0, 0.0248, 1.0289, 0.0440, -0.0351, 0.9600, 0.8837),
    ("c_ate_dose_high", 2000): (3000, 0, 0.0452, 0.9936, 0.0214, 0.0199, 0.9483, 0.9143),
    ("c_ate_dose_high", 8000): (600, 0, 0.1088, 1.0089, 0.0109, 0.0249, 0.9500, 0.9036),
    ("n_ate_r11", 500): (600, 0, 0.3511, 1.0378, 0.0446, -0.0857, 0.9500, 1.7497),
    ("n_ate_r11", 2000): (9000, 0, 0.4135, 1.0400, 0.0212, 0.0249, 0.9567, 1.7325),
    ("n_ate_r11", 8000): (1500, 0, 0.6702, 1.0365, 0.0109, 0.0826, 0.9520, 1.7353),
    ("n_blip_a1_l1", 500): (600, 0, 0.3511, 1.0153, 0.0429, -0.0474, 0.9617, 1.4084),
    ("n_blip_a1_l1", 2000): (9000, 0, 0.4135, 1.0113, 0.0207, -0.0545, 0.9542, 1.4031),
    ("n_blip_a1_l1", 8000): (1500, 0, 0.6702, 0.9865, 0.0103, -0.0960, 0.9427, 1.4376),
}
#: ``cell -> (draws, failures, tested, rejected, seconds per check, SD of the reported difference, mean SE,
#: median covariance degrees of freedom)``.
PILOT_CHECKS: dict[str, tuple[int, int, int, int, float, float, float, float]] = {
    "s2_f2": (600, 0, 600, 10, 0.2172, 0.01536, 0.01700, 67.6),
    "s2_f10": (600, 0, 600, 11, 0.7993, 0.01317, 0.01466, 155.0),
    "h_f2": (600, 0, 600, 5, 0.2325, 0.01538, 0.01590, 26.3),
    "h_f10": (600, 0, 600, 6, 0.8526, 0.01329, 0.01377, 59.8),
    "s2_n2000_f2": (600, 0, 600, 20, 0.3069, 0.00986, 0.00996, 210.1),
    "s2_n2000_f10": (600, 0, 600, 19, 1.0290, 0.00929, 0.00928, 399.5),
    "h_n2000_f2": (600, 0, 600, 11, 0.2893, 0.00899, 0.00955, 63.1),
    "h_n2000_f10": (600, 0, 600, 15, 1.1025, 0.00847, 0.00886, 164.4),
    "sp_f2": (600, 0, 600, 17, 0.2553, 0.01159, 0.01298, 8.8),
    "sp_f10": (600, 0, 600, 36, 0.8784, 0.01101, 0.01204, 7.6),
    "lu_f2": (800, 0, 800, 800, 0.2774, 0.03312, 0.03322, 344.8),
    "lu_f10": (800, 0, 800, 800, 1.0383, 0.03247, 0.03244, 563.6),
}
#: ``(seconds per fit, draws, recordable failures)`` of the ``SP`` positivity pilot.
PILOT_POSITIVITY: tuple[float, int, int] = (0.0752, 600, 0)
#: ``estimand -> (draws, SE ratio, relative SD of the SE, seconds per fit)`` of the unclustered S2C fit.
PILOT_UNCLUSTERED: dict[str, tuple[int, float, float, float]] = {
    "blip[A1:1.0:intercept]": (2400, 1.0111, 0.0408, 0.0765),
    "ey_reference": (2400, 0.5152, 0.0426, 0.0765),
    "ey_regimen[r11]": (2400, 0.5545, 0.0448, 0.0765),
    "ate_regimen[r11 vs r00]": (2400, 1.0179, 0.0422, 0.0765),
}
PILOT_UNCLUSTERED_SECONDS = 0.0765

#: PINNED-END


# ------------------------------------------------------------------------ declarations

POSITIVE_SETUPS = study.CALIBRATED
#: The AGENT-BRIEF floor on the chance that a cell passes under perfect calibration.
PASS_FLOOR = 0.90
#: The floor on the chance that the 61 calibration cells all pass.
FAMILY_FLOOR = 0.95
#: The calibration replicates are a multiple of this grain.
REPLICATE_GRAIN = 400
#: The floor on the chance that a negative control is detected, and on the chance that a gated
#: unclustered direction holds.
POWER_FLOOR = 0.95
#: The floor on the chance that a fit set stays within the failure bound.
FAILURE_FLOOR = 0.99
#: Every calibration arm is piloted on at least this many draws.
POOLED_DRAWS = 3_000
#: The declared total CPU-hours of a full run, from the timed pilot fits (``cpu_hours``).
BUDGET_CPU_HOURS = 11.97
#: The ceiling on a full run that makes a replicate count feasible: 12 CPU-hours is three hours
#: on four workers.  The parent can raise it; the calibration replicates then follow.
CPU_HOUR_CEILING = 12.0
#: The chance that all 61 calibration cells pass at the declared replicates.
DECLARED_FAMILY_PASS = 0.442

Z99 = float(norm.ppf(0.995))
BIAS_MARGIN = Margins().standardized_bias


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.slug == "structural-nested-mean"
    assert (record.n, record.replicates) == (study.PRIMARY_N, study.PRIMARY_REPLICATES)
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert record.reference is None
    assert record.pending_reference == "econml"
    assert record.pending_reason
    assert record.comparison == "matching implementation"
    assert record.comparator_state == "pending"
    assert record.calibration_efficiency_ratio is False
    assert record.failure_bound == {study.IMPLEMENTATION: 0.01, "econml": 0.01}
    assert properties.FAILURE_BOUND == study.FAILURE_BOUND == 0.01
    assert properties.CALIBRATION_N == study.CALIBRATION_N == 2_000
    assert list(study.LADDER) == [500, 2_000, 8_000]
    assert study.ladder_sizes("K") == (1_000, 4_000, 16_000)
    assert study.ladder_sizes("N") == (500, 2_000, 8_000)
    assert properties.PROPERTY_RULES
    assert properties.PROPERTY_WITNESS is not None


def test_the_seeds_are_the_studys_own() -> None:
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert study.SEED not in others
    assert study.RESAMPLING_SEED not in others
    assert study.SEED != study.RESAMPLING_SEED


def test_the_property_streams_are_distinct() -> None:
    seeds = {properties.sample_seed(name, 0) for name in properties.FIT_SETS}
    assert len(seeds) == len(properties.FIT_SETS)
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


def test_the_declared_cells_are_the_cells_the_properties_draw() -> None:
    declared = {(cell.property, cell.cell) for cell in properties.declared_cells()}
    published = {
        (family, cell)
        for family, cells in study.PROPERTY_CELLS.items()
        if family != "root_n_rate"
        for cell in cells
    }
    assert declared == published
    labels = list(study.CALIBRATION_LABELS)
    assert len(labels) == len(set(labels)) == len(PILOT) == 61
    assert set(PILOT) == set(labels)
    assert set(PILOT_CONTROLS) == set(study.CONTROL_LABELS)
    assert set(PILOT_LADDER) == {
        (label, size)
        for label, (setup, _) in study.LADDER_TARGETS.items()
        for size in study.ladder_sizes(setup)
    }
    for cell in properties.declared_cells():
        assert cell.estimand in cell.dgp.truth()
    sizes = {name for name, (kind, *_) in study.CHECK_CELLS.items() if kind == "size"}
    assert sizes == {
        "s2_f2",
        "s2_f10",
        "h_f2",
        "h_f10",
        "s2_n2000_f2",
        "s2_n2000_f10",
        "h_n2000_f2",
        "h_n2000_f10",
        "sp_f2",
        "sp_f10",
    }
    assert set(PILOT_CHECKS) == set(study.CHECK_CELLS)


def test_every_key_the_study_publishes_has_a_description() -> None:
    descriptions.implementation(study.IMPLEMENTATION)
    for scenario, estimands in study.SCENARIOS.items():
        descriptions.scenario(scenario)
        for name in estimands:
            descriptions.estimand(name)
    for _, estimand in {
        **study.TARGETS,
        **study.CONTROL_TARGETS,
        **study.POSITIVITY_TARGETS,
    }.values():
        descriptions.estimand(estimand)
    for family, cells in study.PROPERTY_CELLS.items():
        descriptions.claim(family)
        for cell in cells:
            control = cell.endswith(
                ("__shrunken_se_control", "__noise_control", "__too_few_features")
            )
            descriptions.cell(family, cell, role="control" if control else "positive")


def test_the_primary_scenarios_are_the_positive_setups() -> None:
    assert set(study.SCENARIOS) == {
        name.lower() for name in laws.setups() if name not in {"SP", "LU", "VB"}
    }
    assert set(study.NEGATIVE_CONTROLS) == {"LU", "VB"}
    assert {scenario: scenario for scenario in study.SCENARIOS} == study.SEED_OWNERS
    assert set(study.CALIBRATED) == set(study.SETUP_OF.values())


def test_no_calibration_estimand_repeats_another_of_its_setup() -> None:
    """Two labels of one setup that read one statistic enter the family product twice.

    With ``phi = 1`` the contrast of ``r10`` is the first blip and that of ``r01`` is the
    second, so a setup that reports both a blip and such a contrast calibrates one number
    twice.  Two coefficients may share a truth, so the check compares the pilot rows: one
    statistic on the same draws gives identical rows, two statistics do not.
    """
    for setup in study.CALIBRATED:
        rows = [PILOT[label_of(setup, e)][1:] for e in laws.setup(setup).estimands]
        assert len(rows) == len({tuple(row) for row in rows}), setup


# ------------------------------------------------------------ (a) the structural model


def _phi_of(features: dict[int, list[str]] | None, node: int) -> Callable[[dict], np.ndarray]:
    terms = [] if not features else list(features.get(node, []))

    def phi(history: dict[str, float]) -> np.ndarray:
        return np.array([1.0, *(history[name] for name in terms)])

    return phi


def _levels_of(law: finite.Law, node: int) -> list[float]:
    treatment = law.treatments[node - 1]
    return sorted({path[treatment] for path, _ in finite.paths(law)})


def _coding_of(law: finite.Law, node: int) -> Callable[[float], np.ndarray]:
    if law.treatments[node - 1] in law.doses:
        return lambda level: np.array([level])
    levels = _levels_of(law, node)
    return lambda level: np.array([1.0 * (level == other) for other in levels[1:]])


@pytest.mark.parametrize("name", [*POSITIVE_SETUPS, "SP"])
def test_each_positive_setup_satisfies_its_structural_model(name: str) -> None:
    """(a) The declared coefficients are the blips of the law, by the g-formula."""
    chosen = laws.setup(name)
    law = chosen.law
    for node in range(1, law.n_nodes + 1):
        gap = finite.structural_blip_errors(
            law,
            node,
            _levels_of(law, node),
            _phi_of(chosen.features, node),
            np.array(chosen.coefficients[node]),
            _coding_of(law, node),
        )
        assert gap < EXACT, (name, node, gap)
    truths = laws.truths(name)
    for key, value in chosen.psi.items():
        assert truths[key] == value


def _exact_fit(name: str, **settings: Any) -> Any:
    chosen = laws.setup(name)
    law = chosen.law
    table = finite.frame(law).drop(columns="E")
    table["wt"] = table["p"] * (laws.weight(table["W"]) if chosen.weighted else 1.0)
    estimator = StructuralNestedMean(
        chosen.regimens or None,
        reference=chosen.reference or None,
        blip_features=chosen.features,
        n_folds=1,
        simultaneous=False,
        **settings,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            table,
            outcome="Y",
            treatment=law.treatments,
            baseline=law.baseline,
            time_varying=law.time_varying,
            weights="wt",
            family="gaussian",
            continuous_treatment=list(law.doses),
        )


def _exact_error(name: str, order: int) -> float:
    """The largest distance of the fit at population weights from the exact truths."""
    result = _exact_fit(name, **study.learners(name, order))
    return max(
        abs(result[estimand].psi - value)
        for estimand, value in laws.truths(name).items()
        if estimand in result
    )


@pytest.mark.parametrize("name", [n for n in POSITIVE_SETUPS if n != "N"])
def test_the_declared_order_is_the_lowest_exact_order(name: str) -> None:
    """(a) The declared ``GLM`` order returns every truth at population weights, and one order
    less does not (except at main terms), so no setup carries a higher order than it needs."""
    order = study.INTERACTIONS.get(name, study.DEFAULT_INTERACTIONS)
    assert _exact_error(name, order) < 1e-6, name
    if order > 1:
        assert _exact_error(name, order - 1) > 1e-6, name


#: The regressions that are not exact at the declared order.  The node-2 outcome regression of
#: ``SP`` needs a three-way cell, and an order of three fails on a separated cell in 17% of the
#: draws at n = 2,000.  Its treatment regressions are exact, so the estimator is unbiased.
INEXACT_REGRESSIONS = frozenset({("SP", "outcome")})


@pytest.mark.parametrize("name", [n for n in laws.setups() if n != "N"])
def test_every_nuisance_regression_is_exact_at_the_declared_order(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) Each regression of a population-weighted fit returns the weighted cell mean of its
    target.  The final truths alone can hide a wrong nuisance, because the estimator is doubly
    robust: a correct treatment side makes psi exact with a wrong outcome regression."""
    captured: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    original = snmm_module._regress

    def recording(
        data: Any,
        learners: Any,
        role: str,
        design: np.ndarray,
        target: np.ndarray,
        folds: Any,
        n_jobs: Any,
        sink: Any,
    ) -> np.ndarray:
        prediction = original(data, learners, role, design, target, folds, n_jobs, sink)
        captured.append(
            (
                role,
                np.array(design, dtype=float),
                np.array(target, dtype=float),
                np.array(data.weights, dtype=float),
                np.array(prediction, dtype=float),
            )
        )
        return prediction

    monkeypatch.setattr(snmm_module, "_regress", recording)
    order = study.INTERACTIONS.get(name, study.DEFAULT_INTERACTIONS)
    _exact_fit(name, **study.learners(name, order))
    assert captured, name
    worst: dict[str, float] = {}
    for role, design, target, weight, prediction in captured:
        key = pd.DataFrame(design).round(10).astype(str).agg("|".join, axis=1)
        frame = pd.DataFrame({"key": key, "target": target * weight, "weight": weight})
        grouped = frame.groupby("key").sum()
        cell_mean = (grouped["target"] / grouped["weight"]).reindex(key).to_numpy()
        gap = float(np.max(np.abs(cell_mean - prediction)))
        worst[role] = max(worst.get(role, 0.0), gap)
        if (name, role) not in INEXACT_REGRESSIONS:
            assert gap < 1e-6, (name, role, gap)
    for pair in INEXACT_REGRESSIONS:
        if pair[0] == name:
            # A deliberate witness: the exemption stays only while the regression is inexact.
            assert 1e-6 < worst[pair[1]] < 0.2, (pair, worst)


def test_the_booster_law_is_nonlinear_in_the_baseline() -> None:
    """``N`` is the adaptive cell: its learners are boosters, and its law has nine baselines."""
    declared = study.learners("N")
    assert declared["outcome_learner"].__class__.__name__ == "HistGradientBoostingRegressor"
    assert declared["treatment_learner"].__class__.__name__ == "HistGradientBoostingClassifier"
    law = laws.setup("N").law
    assert len({path["W"] for path, _ in finite.paths(law)}) == 9


def test_the_outcome_support_holds_the_draws() -> None:
    for name in laws.setups():
        if laws.setup(name).cluster:
            # The cluster intercept widens the outcome; no model check reads a clustered fit.
            continue
        low, high = laws.outcome_support(name)
        frame = laws.sample(name, 4_000, 7)
        assert frame["Y"].min() >= low
        assert frame["Y"].max() <= high


def test_the_samplers_reproduce_the_declared_structure() -> None:
    weighted = laws.sample("S2W", 4, 3)
    assert np.allclose(weighted["w"], laws.weight(weighted["W"]))
    clustered = laws.sample("S2C", 100, 5)
    assert clustered["id"].nunique() == 10
    assert (clustered.groupby("id")["W"].nunique() == 1).all()


# ---------------------------------------------------------------- (b) negative controls


def _inside_probability(coverage: float, replicates: int) -> float:
    """The chance that the exact 99% interval of the covered count lies inside the band."""
    tail = 0.005
    hits = np.arange(0, replicates + 1)
    lower = np.where(hits > 0, beta.ppf(tail, np.maximum(hits, 1), replicates - hits + 1), 0.0)
    upper = np.where(
        hits < replicates, beta.ppf(1.0 - tail, hits + 1, np.maximum(replicates - hits, 1)), 1.0
    )
    low, high = Margins().calibration_coverage
    inside = (lower >= low) & (upper <= high)
    return float(binom.pmf(hits, replicates, coverage)[inside].sum())


def control_detection_probability(label: str) -> float:
    """The chance that a control's 99% bias interval lies beyond the margin on the gap's side.

    The standardized bias of the study is normal about the true value with variance ``1 / R``.
    The true value is normal about the pilot value with variance ``1 / k`` for ``k`` pilot
    draws.  The interval is the estimate plus or minus ``z / sqrt(R)`` in standard deviations.
    """
    draws, bias, _, _ = PILOT_CONTROLS[label]
    replicates = study.CONTROL_REPLICATES
    spread = math.sqrt(1.0 / replicates + 1.0 / draws)
    side = 1.0 if properties.control_gap(label) > 0.0 else -1.0
    threshold = BIAS_MARGIN + Z99 / math.sqrt(replicates)
    return float(norm.sf((threshold - side * bias) / spread))


def test_the_negative_controls_miss_the_regimen_truth() -> None:
    """(b) The limit of the undeclared fit differs from the g-formula value by a computed amount."""
    for setup in study.NEGATIVE_CONTROLS:
        gaps = properties.check_gaps(setup)
        assert max(abs(gap) for gap in gaps.values()) > 0.1, setup
        limit = properties.population_limit(setup)
        for contrast, gap in gaps.items():
            assert abs(limit[contrast] - laws.truth(setup, contrast) - gap) < EXACT
    assert max(abs(properties.control_gap(label)) for label in study.CONTROL_LABELS) > 0.5


def test_each_control_gap_has_the_sign_and_size_the_pilot_reads() -> None:
    """(b) The pilot's standardized bias agrees with the exact gap over the sampling SD."""
    for label, (_, bias, sd_root_n, _) in PILOT_CONTROLS.items():
        sd = sd_root_n / math.sqrt(properties.CALIBRATION_N)
        gap = properties.control_gap(label)
        assert math.copysign(1.0, gap) == math.copysign(1.0, bias), label
        assert gap / sd == pytest.approx(bias, abs=0.15 + 0.05 * abs(bias)), label


def test_each_negative_control_is_detected_with_chance_at_least_the_floor() -> None:
    """(b) The detection rule fires on every control cell with chance >= 0.95.

    A cell passes when its 99% standardized-bias interval lies beyond 0.25 SD on the side of
    the computed gap.  A blind instrument leaves the cell inside the margin, and the row is red.
    """
    for label in study.CONTROL_LABELS:
        assert control_detection_probability(label) >= POWER_FLOOR, label


def test_no_row_is_declared_red() -> None:
    """(g) The controls pass by detection and the sparse-cell size cells are positive cells."""
    assert study.EXPECTED_RED == ()
    assert study.EXPECTED_RED_PROPERTIES == ()
    rule = next(
        rule for rule in properties.PROPERTY_RULES if "blip_feature_control" in rule.families
    )
    assert [clause.name for clause in rule.clauses] == ["bias beyond margin on the side of the gap"]
    assert study.CONFIGURATION["expected_red"] == []


# --------------------------------------------------------------------- (c) calibration

#: The nodes of the probabilists' Gauss-Hermite rule.
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(41)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()
#: The upper quantile of the between-cell spread of an arm.
TAU_QUANTILE = 0.80
#: The cells of one setup share their draws, so their pilot errors in the SE ratio are treated
#: as one error: the shared term has the variance of one cell.
EFFECTIVE_CELLS = 1


def _ratio_leg(rho: Any, noise: float, replicates: int) -> Any:
    """The chance that the 99% SE-ratio interval lies inside the calibration band."""
    low, high = Margins().calibration_se_ratio
    spread = rho * np.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    inside = norm.cdf((high - Z99 * spread - rho) / spread) - norm.cdf(
        (low + Z99 * spread - rho) / spread
    )
    return np.maximum(inside, 0.0)


def _coverage_leg(coverage: Any, replicates: int) -> Any:
    """The chance that the 99% coverage interval lies inside the calibration band."""
    low, high = Margins().calibration_coverage
    width = np.sqrt(coverage * (1.0 - coverage) / replicates)
    inside = norm.cdf((high - Z99 * width - coverage) / width) - norm.cdf(
        (low + Z99 * width - coverage) / width
    )
    return np.maximum(inside, 0.0)


def _bias_leg(bias: float, draws: int, replicates: int) -> float:
    """The chance that the 99% bias interval lies inside the margin.

    The bias estimate is normal about the pilot value with the variance ``1 / R`` of the study
    plus ``1 / k`` of the pilot.
    """
    half = BIAS_MARGIN - Z99 / math.sqrt(replicates)
    spread = math.sqrt(1.0 / replicates + 1.0 / draws)
    return float(norm.cdf((half - bias) / spread) - norm.cdf((-half - bias) / spread))


def _cell_pass(label: str, rho: Any, replicates: int) -> Any:
    """The chance a cell passes at the true SE ratio ``rho``: ratio, coverage and bias legs."""
    draws, _, noise, bias, _, _ = PILOT[label]
    z = float(norm.ppf(0.975))
    coverage = norm.cdf(z * rho - bias) - norm.cdf(-z * rho - bias)
    return (
        _ratio_leg(rho, noise, replicates)
        * _coverage_leg(coverage, replicates)
        * _bias_leg(bias, draws, replicates)
    )


def _pilot_se(label: str) -> float:
    """The sampling SD of a cell's pilot SE ratio."""
    draws, ratio, noise, *_ = PILOT[label]
    return ratio * math.sqrt(1.0 / (2.0 * (draws - 1)) + noise**2 / draws)


def _spread_upper_bound(labels: list[str]) -> float:
    """The upper ``TAU_QUANTILE`` bound of the between-cell SD of an arm's true SE ratios.

    Cochran's ``Q`` of the arm's pilot ratios is chi-squared on ``k - 1`` degrees of freedom
    when the true ratios agree.  The bound is the spread at which the observed ``Q`` sits at the
    lower ``1 - TAU_QUANTILE`` quantile.  It is zero when the observed ``Q`` is that small.
    """
    ratios = np.array([PILOT[label][1] for label in labels])
    errors = np.array([_pilot_se(label) for label in labels])
    if len(labels) < 2:
        return 0.0

    def q(tau: float) -> float:
        weight = 1.0 / (errors**2 + tau**2)
        centre = float((weight * ratios).sum() / weight.sum())
        return float((weight * (ratios - centre) ** 2).sum())

    target = float(chi2.ppf(1.0 - TAU_QUANTILE, len(labels) - 1))
    return 0.0 if q(0.0) <= target else float(brentq(lambda tau: q(tau) - target, 0.0, 1.0))


def _arms() -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {}
    for label in PILOT:
        arms.setdefault(label.split("_")[0], []).append(label)
    return arms


def arm_pass_probability(labels: list[str], replicates: int) -> float:
    """The chance that every cell of one arm passes, with the arm's true SE ratio uncertain.

    The true SE ratio of a cell is ``m + e + u``.  ``m`` is the mean of the arm's pilot ratios,
    ``e`` is the shared pilot error, normal with the variance of one pilot ratio divided by
    :data:`EFFECTIVE_CELLS`, and ``u`` is the cell's own departure, normal with the upper bound
    :func:`_spread_upper_bound`.  The cells share ``e``, so the arm's chance is the mean over
    ``e`` of the product over cells.
    """
    centre = float(np.mean([PILOT[label][1] for label in labels]))
    shared = float(np.median([_pilot_se(label) for label in labels])) / math.sqrt(EFFECTIVE_CELLS)
    own = _spread_upper_bound(labels)
    total = 0.0
    for node, weight in zip(_NODES, _WEIGHTS, strict=True):
        product = 1.0
        for label in labels:
            rho = centre + shared * node + own * _NODES
            product *= float(np.sum(_WEIGHTS * _cell_pass(label, rho, replicates)))
        total += float(weight) * product
    return total


def family_pass_probability(replicates: int) -> float:
    """The chance that all 61 calibration cells pass: the arms are independent."""
    return float(np.prod([arm_pass_probability(labels, replicates) for labels in _arms().values()]))


def perfect_family_pass_probability(replicates: int) -> float:
    """The chance that all 61 cells pass at a true SE ratio of 1, coverage 0.95 and no bias.

    Necessary, not sufficient: the pilots say how far the arms sit from perfect calibration.
    """
    return float(
        np.prod(
            [
                calibration_pass_probability(
                    replicates, true_se_ratio=1.0, coverage=0.95, se_noise=PILOT[label][2]
                )
                for label in PILOT
            ]
        )
    )


def test_the_vectorized_cell_rule_agrees_with_the_shared_function() -> None:
    """The ratio and coverage legs equal :func:`calibration_pass_probability`."""
    for label in ("s2_blip_a1_l1", "n_blip_a2_l1", "s2c_ey0"):
        noise = PILOT[label][2]
        for rho in (0.95, 1.0, 1.04):
            shared = calibration_pass_probability(
                1_200, true_se_ratio=rho, coverage=0.95, se_noise=noise
            )
            mine = float(_ratio_leg(rho, noise, 1_200) * _coverage_leg(0.95, 1_200))
            assert mine == pytest.approx(shared, rel=1e-9)


def test_every_calibration_arm_has_a_pilot_of_the_pooled_size() -> None:
    """(c) MAJOR-1: each arm is piloted on at least 3,000 draws of the declared configuration."""
    for label, row in PILOT.items():
        assert row[0] >= POOLED_DRAWS - 3, (label, row[0])
    for setup, (_, draws, _) in PILOT_COST.items():
        assert draws >= POOLED_DRAWS, setup


def test_the_calibration_replicates_follow_the_sizing_rule() -> None:
    """(c) MAJOR-1: the rule is the smallest multiple of 400 whose family pass probability is at
    least 0.95 under the arm-pooled predictive rule, with the bias leg.  The pilots show that no
    replicate count reaches 0.95: the chance rises with R to a limit set by the pilot's
    resolution of each arm's true SE ratio.  So the rule falls back to the largest multiple of
    400 whose full run fits the declared CPU-hour ceiling, and declares the chance it gives.
    Perfect calibration is the AGENT-BRIEF floor: necessary and not sufficient."""
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert perfect_family_pass_probability(replicates) >= FAMILY_FLOOR
    # The floor is out of reach at any replicate count: the limit of the chance is below it.
    assert family_pass_probability(10**6) < FAMILY_FLOOR
    assert family_pass_probability(replicates) == pytest.approx(DECLARED_FAMILY_PASS, abs=0.005)
    assert cpu_hours() <= CPU_HOUR_CEILING
    assert cpu_hours(replicates + REPLICATE_GRAIN) > CPU_HOUR_CEILING
    smaller = replicates - REPLICATE_GRAIN
    assert family_pass_probability(smaller) < family_pass_probability(replicates)


def test_the_noise_control_scale_is_the_pilot_sampling_sd() -> None:
    """The noise control adds one pilot SD at the cell's size: the SD times root n of the pilot."""
    assert set(properties.NOISE_SD) == set(PILOT)
    for label, value in properties.NOISE_SD.items():
        assert value == pytest.approx(PILOT[label][5], abs=5e-5)


def test_the_calibration_pilot_shows_no_defect() -> None:
    """Each cell's pilot SE ratio, coverage and bias are consistent with a calibrated cell at
    the pilot's resolution (99% two-sided)."""
    z = float(norm.ppf(0.995))
    for label, (draws, ratio, noise, bias, coverage, _) in PILOT.items():
        half = ratio * math.sqrt(1.0 / (2.0 * (draws - 1)) + noise**2 / draws)
        assert ratio + z * half > 0.93, (label, ratio)
        assert ratio - z * half < 1.07, (label, ratio)
        assert abs(bias) < 0.25, (label, bias)
        assert 0.92 - 3 * math.sqrt(0.05 * 0.95 / draws) <= coverage, (label, coverage)


# -------------------------------------------------------------------------- failures


def failure_pass_probability(draws: int, failures: int, replicates: int) -> float:
    """The chance that a fit set loses no more replicates than the declared bound allows.

    The failure rate is Beta(f + 1/2, k - f + 1/2) given ``f`` failures in ``k`` pilot draws
    (Jeffreys), and the count lost in ``R`` replicates is binomial in it.
    """
    allowed = math.floor(study.FAILURE_BOUND * replicates + 1e-9)
    grid = (np.arange(400) + 0.5) / 400
    rate = beta.ppf(grid, failures + 0.5, draws - failures + 0.5)
    return float(np.mean(binom.cdf(allowed, replicates, rate)))


def _pilot_failures() -> dict[str, tuple[int, int]]:
    """``fit set -> (pilot draws, pilot failures)`` for every fit set of the study."""
    out: dict[str, tuple[int, int]] = {}
    for name, spec in properties.FIT_SETS.items():
        if spec.kind == "calibration":
            _, draws, failures = PILOT_COST[spec.setup]
        elif spec.kind == "ladder":
            label = next(
                label for label, (setup, _) in study.LADDER_TARGETS.items() if setup == spec.setup
            )
            draws, failures = PILOT_LADDER[(label, spec.n)][:2]
        elif spec.kind == "control":
            _, draws, failures = PILOT_CONTROL_COST[spec.setup]
        elif spec.kind == "positivity":
            _, draws, failures = PILOT_POSITIVITY
        else:
            draws, failures = PILOT_CHECKS[name.removeprefix("check_")][:2]
        out[name] = (draws, failures)
    return out


def test_the_recordable_failures_stay_under_the_declared_bound() -> None:
    """MAJOR-3: every fit set of the study stays within the 1% bound with chance at least 0.99.

    A pilot that shows failures at the bound would leave the cell red for a fitting reason,
    which the old ``K`` rung at n = 500 did: 6 failures in 600 draws.
    """
    pilots = _pilot_failures()
    assert set(pilots) == set(properties.FIT_SETS)
    for name, (draws, failures) in pilots.items():
        spec = properties.FIT_SETS[name]
        chance = failure_pass_probability(draws, failures, spec.replicates)
        assert chance >= FAILURE_FLOOR, (name, draws, failures, chance)


def test_the_old_k_rung_fails_the_failure_rule() -> None:
    """The rung that was dropped: 16 failures in 1,200 draws at n = 500, from the diagnosis."""
    assert failure_pass_probability(1_200, 16, properties.LADDER_REPLICATES) < 0.5


# ------------------------------------------------------------------------- (d) ladder


def _ladder_pass(replicates: int, coverage: float = 0.95) -> float:
    """The chance a positive rung clears the coverage floor: the exact lower endpoint >= 0.90."""
    tail = 0.005
    hits = np.arange(0, replicates + 1)
    lower = np.where(hits > 0, beta.ppf(tail, np.maximum(hits, 1), replicates - hits + 1), 0.0)
    passing = hits[lower >= 0.90]
    return float(binom.sf(int(passing.min()) - 1, replicates, coverage))


def test_the_ladder_rungs_clear_the_floor_with_the_declared_replicates() -> None:
    assert properties.LADDER_REPLICATES == 1_000
    assert _ladder_pass(properties.LADDER_REPLICATES) >= PASS_FLOOR
    for (label, size), row in PILOT_LADDER.items():
        draws, failures, _, ratio, _, bias, coverage, _ = row
        assert failures / draws <= 0.01, (label, size)
        assert 0.80 < ratio < 1.20, (label, size, ratio)
        assert abs(bias) < 0.25, (label, size, bias)
        assert coverage >= 0.90, (label, size, coverage)


def test_the_ladder_spread_contracts_at_the_root_n_rate() -> None:
    """The empirical SD times the root of n is flat across the rungs."""
    for label, (setup, _) in study.LADDER_TARGETS.items():
        scaled = [PILOT_LADDER[(label, size)][7] for size in study.ladder_sizes(setup)]
        assert max(scaled) / min(scaled) < 1.12, (label, scaled)


def test_the_adaptive_cell_has_a_ladder_that_shows_the_remainder_contract() -> None:
    """MAJOR-4: ``N`` stands on three sizes with a contrast and a blip.  Its bias interval at the
    largest rung lies inside 0.25 SD with chance at least 0.90, and its bias in SD units does not
    grow from n = 2,000 to n = 8,000 by more than the pilot's resolution."""
    assert study.LADDER_ESTIMANDS["N"] == ("ate_regimen[r11 vs r00]", "blip[A1:1.0:intercept]")
    for label in (label_of("N", e) for e in study.LADDER_ESTIMANDS["N"]):
        draws, _, _, _, _, bias, _, _ = PILOT_LADDER[(label, 8_000)]
        assert _bias_leg(bias, draws, properties.LADDER_REPLICATES) >= PASS_FLOOR, label
        mid_draws, *_, mid_bias, _, _ = PILOT_LADDER[(label, 2_000)]
        resolution = Z99 * math.sqrt(1.0 / draws + 1.0 / mid_draws)
        assert abs(bias) <= abs(mid_bias) + resolution, label


# ----------------------------------------------------------------- (e) the model check


def size_pass_probability(rate: float, replicates: int, ceiling: float = 0.10) -> float:
    """The chance the exact upper endpoint of the rejection rate stays at or below the ceiling."""
    tail = 0.005
    hits = np.arange(0, replicates)
    upper = beta.ppf(1.0 - tail, hits + 1, replicates - hits)
    passing = hits[upper <= ceiling]
    if passing.size == 0:
        return 0.0
    return float(binom.cdf(int(passing.max()), replicates, rate))


def predictive_size_pass_probability(
    rejected: int, draws: int, replicates: int, ceiling: float = 0.10
) -> float:
    """The same chance with the rejection rate drawn from the pilot: Beta(r + 1/2, k - r + 1/2)."""
    grid = (np.arange(400) + 0.5) / 400
    rate = beta.ppf(grid, rejected + 0.5, draws - rejected + 0.5)
    return float(np.mean([size_pass_probability(float(r), replicates, ceiling) for r in rate]))


def test_the_size_cells_pass_with_the_declared_replicates() -> None:
    """(e) A correct model rejects at most at the nominal 5%; a cell passes when the exact upper
    endpoint stays at or below 0.10.  The chance of that is at least 0.90 at the nominal rate and
    at the rate the pilot read, and the sparse-positivity cells carry the replicates the
    predictive rule needs."""
    assert study.SIZE_REPLICATES % 200 == 0
    for name, (kind, *_) in study.CHECK_CELLS.items():
        if kind != "size":
            continue
        replicates = study.check_replicates(name)
        draws, failures, _, rejected, *_ = PILOT_CHECKS[name]
        assert size_pass_probability(0.05, replicates) >= PASS_FLOOR, name
        chance = predictive_size_pass_probability(rejected, draws - failures, replicates)
        assert chance >= PASS_FLOOR, (name, replicates, chance)
    assert size_pass_probability(0.05, study.SIZE_REPLICATES - 200) < PASS_FLOOR
    for name in study.CHECK_REPLICATES:
        draws, failures, _, rejected, *_ = PILOT_CHECKS[name]
        smaller = study.check_replicates(name) - 200
        assert predictive_size_pass_probability(rejected, draws - failures, smaller) < PASS_FLOOR


def _exact_difference_moments(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The population gap and the asymptotic covariance of the structural minus the
    nonparametric contrasts, from the influence curves on the exact table.

    The table's weights are the probabilities :math:`p_i`, and a curve carries the weight of
    its row, :math:`m p_i \\psi_i` for :math:`m` rows.  The population covariance of the
    difference is :math:`\\sum_i p_i \\psi_i \\psi_i^\\top = \\sum_i c_i c_i^\\top / (m^2 p_i)`.
    """
    chosen = laws.setup(name)
    law = chosen.law
    table = finite.frame(law).drop(columns="E")
    saturated = len(law.baseline) + 2 * len(law.treatments)
    estimator = StructuralNestedMean(
        chosen.regimens,
        reference=chosen.reference,
        blip_features=chosen.features,
        outcome_learner=GLM(family="gaussian", interactions=saturated, tol=1e-13),
        treatment_learner=GLM(interactions=saturated, tol=1e-13),
        n_folds=1,
        simultaneous=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(
            table,
            outcome="Y",
            treatment=law.treatments,
            baseline=law.baseline,
            time_varying=law.time_varying,
            weights="p",
            family="gaussian",
        )
        check = result.check_model(q_bounds=laws.outcome_support(name))
    curves = np.column_stack(
        [result[n].influence_curve - check.ltmle[n].influence_curve for n in check.names]
    )
    rows, probability = len(table), table["p"].to_numpy()
    covariance = (curves.T / (rows**2 * probability)) @ curves
    return np.asarray(check.difference), covariance


def exact_power(name: str, n: int, alpha: float = 0.05) -> float:
    """The asymptotic power of the joint Wald test at size ``n``: the noncentral chi-squared
    tail at the population gap with the exact difference covariance."""
    gap, covariance = _exact_difference_moments(name)
    noncentrality = n * float(gap @ np.linalg.solve(covariance, gap))
    critical = float(stats.chi2.ppf(1.0 - alpha, len(gap)))
    return float(stats.ncx2.sf(critical, len(gap), noncentrality))


def test_the_exact_difference_covariance_matches_the_monte_carlo_spread() -> None:
    """The influence-curve covariance predicts the sampling spread of the difference: the
    pilot's SD and mean SE of the reported difference are within 15% of the exact value."""
    name, cell = "LU", "lu_f10"
    _, covariance = _exact_difference_moments(name)
    index = list(properties.check_names(name)).index(properties.check_contrast(name))
    exact_sd = math.sqrt(covariance[index, index] / study.POWER_N)
    sd, se = PILOT_CHECKS[cell][5:7]
    assert 0.85 < sd / exact_sd < 1.15, (cell, sd, exact_sd)
    assert 0.85 < se / exact_sd < 1.15, (cell, se, exact_sd)


def test_the_power_cells_reach_their_declared_floor() -> None:
    """(e) The exact-law power at the declared size is at least 0.95, so the exact lower
    endpoint of the rejection rate clears 0.80 with chance at least 0.90, and each pilot rate
    agrees."""
    power = exact_power("LU", study.POWER_N)
    assert power >= POWER_FLOOR
    assert power_cell_pass_probability(power, study.POWER_REPLICATES) >= PASS_FLOOR
    for cell in ("lu_f2", "lu_f10"):
        draws, _, _, rejected, *_ = PILOT_CHECKS[cell]
        assert rejected / draws >= 0.90, (cell, rejected, draws)
    assert MINIMUM_POWER == 0.80


def test_the_sparse_positivity_cells_are_positive_cells_with_low_power() -> None:
    """(e) BLOCKER-1: ``SP`` at n = 2,000 fits in every pilot draw, rejects a correct model near
    or below the nominal rate under the Satterthwaite reference, and reads no red row.  Its
    covariance has few degrees of freedom, so the check has low power there: the cells measure
    size, and the page says so."""
    assert study.SPARSE_N == 2_000
    for cell in ("sp_f2", "sp_f10"):
        draws, failures, tested, rejected, *_, median_df = PILOT_CHECKS[cell]
        assert failures == 0, cell
        assert tested == draws, cell
        assert rejected / tested <= 0.10, (cell, rejected, tested)
        assert 3.0 < median_df < 20.0, (cell, median_df)
        assert f"warning_rate/{cell}__model_check" not in study.EXPECTED_RED
    for cell in ("s2_n2000_f2", "s2_n2000_f10", "h_n2000_f2", "h_n2000_f10"):
        assert PILOT_CHECKS[cell][7] > 50.0, cell


def test_the_check_is_conservative_at_ten_folds_where_positivity_holds() -> None:
    """(e) MINOR-5: on ``S2`` and ``H`` the check rejects a correct model less often than the
    nominal rate, so a non-rejection carries less information; the page states it and
    publishes the rejection rate beside the verdict."""
    for cell in ("s2_f10", "h_f10", "s2_n2000_f10", "h_n2000_f10"):
        _, _, tested, rejected, *_ = PILOT_CHECKS[cell]
        assert rejected / tested < 0.05, (cell, rejected, tested)


# ------------------------------------------------------------------ (f) the cluster cell


def _leaves_below_chance(estimand: str) -> float:
    """The chance that the 99% interval of the unclustered SE ratio lies below the band."""
    z = float(norm.ppf(0.995))
    replicates = properties.CALIBRATION_REPLICATES
    _, ratio, noise, _ = PILOT_UNCLUSTERED[estimand]
    spread = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    return float(norm.cdf((0.93 - z * spread - ratio) / spread))


def test_the_unclustered_standard_error_leaves_the_band_with_power() -> None:
    """(f) Each gated direction holds with chance at least 0.95 at the declared replicates.

    A cell is gated exactly when the pilot gives it that chance; the others report.  The
    level estimands carry the cluster intercept, so ignoring the clusters understates their SE
    by about half.  The blip and the contrast difference the intercept away."""
    gated = {f"{label_of('S2C', e)}__unclustered" for e in properties.GATED_ESTIMANDS}
    assert set(properties.GATED_DIRECTIONS) == gated
    for estimand in laws.setup("S2C").estimands:
        cell = f"{label_of('S2C', estimand)}__unclustered"
        chance = _leaves_below_chance(estimand)
        if cell in gated:
            assert properties.GATED_DIRECTIONS[cell] == "below"
            assert chance >= POWER_FLOOR, (cell, chance)
        else:
            assert chance < POWER_FLOOR, (cell, chance)


# ---------------------------------------------------------------------- (h) the budget


def cpu_hours(calibration_replicates: int | None = None) -> float:
    """Every phase of a full run, in CPU-hours, from the pilot's seconds per fit."""
    replicates = (
        properties.CALIBRATION_REPLICATES
        if calibration_replicates is None
        else calibration_replicates
    )
    seconds = 0.0
    seconds += study.PRIMARY_REPLICATES * sum(
        PILOT_COST[study.SETUP_OF[scenario]][0] for scenario in study.SCENARIOS
    )
    for setup in study.CALIBRATED:
        seconds += replicates * PILOT_COST[setup][0]
    seconds += replicates * PILOT_UNCLUSTERED_SECONDS
    for spec in properties.FIT_SETS.values():
        if spec.kind == "ladder":
            label = next(
                label for label, (setup, _) in study.LADDER_TARGETS.items() if setup == spec.setup
            )
            seconds += spec.replicates * PILOT_LADDER[(label, spec.n)][2]
    for setup in study.NEGATIVE_CONTROLS:
        seconds += study.CONTROL_REPLICATES * PILOT_CONTROL_COST[setup][0]
    seconds += study.POSITIVITY_REPLICATES * PILOT_POSITIVITY[0]
    for name in study.CHECK_CELLS:
        seconds += study.check_replicates(name) * PILOT_CHECKS[name][4]
    return seconds / 3600.0


def test_the_budget_is_declared() -> None:
    """(h) The total CPU-hours of a full run, from timed pilot fits."""
    print(f"\nbudget {cpu_hours():.2f} CPU-hours")
    assert cpu_hours() == pytest.approx(BUDGET_CPU_HOURS, rel=0.05)
