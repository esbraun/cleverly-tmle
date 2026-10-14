"""The declared design of the study ``sequential-baselines`` (X41).

Every number here is fixed before any registered run.  The conditions are those of
``X41-plan.md`` section 5.1: (a) exact specification of each declared GLM, (b) MLE existence
at the ladder's first size, (c) the K1 power, (d) the K2 and K3 predicted ratios and (e) the
calibration replicate count.  The cheap conditions are recomputed here.  The pilots of (b) to
(e) are pinned: their scripts and saved outputs are named beside each table.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from tests.studies import canonical_sequential_baselines as study
from tests.studies import sequential_baselines_laws as laws
from tests.studies import sequential_baselines_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import descriptions
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("sequential_baselines_design")

#: The rows and seed of the Monte Carlo checks of (a), the truths and the bounds.
LARGE_N = 1_000_000
LARGE_SEED = 2026144102
#: (a) for G and GM: the largest mean of a residual times a design column.
RESIDUAL_TOLERANCE = 1e-3

# ------------------------------------------------------------------------- pinned pilots

#: (b) The MLE existence probe: ``(law, n) -> (draws, draws with any failed fit)``.  A draw
#: fails when any declared fit of the law (``FITS`` in ``mle_probe.py``) raises ``DataError``.
MLE_PROBE: dict[tuple[str, int], tuple[int, int]] = {
    ("D2", 1_000): (2_000, 11),
    ("D2", 2_000): (2_000, 0),
    ("D2S", 2_000): (2_000, 0),
    ("D2C", 2_000): (2_000, 227),
    ("D2C", 4_000): (2_000, 13),
    ("D2C", 8_000): (2_000, 0),
    ("G", 2_000): (2_000, 0),
    ("X", 2_000): (2_000, 0),
    ("GM", 2_000): (2_000, 0),
}
#: The size at which each law is fitted in a gated cell.
DECLARED_SIZES = {
    "D2": study.LADDER[0],
    "G": study.LADDER[0],
    "D2S": study.CALIBRATION_N,
    "X": study.CALIBRATION_N,
    "GM": study.CALIBRATION_N,
    "D2C": study.D2C_N,
}

#: The design pilot at ``CALIBRATION_N``: ``cell -> (replicates, mean SE over empirical SD,
#: relative SD of the SE, mean bias over empirical SD, sqrt(n) times the empirical SD)``.  It
#: draws from the ``design_pilot`` stream, which no published cell reads.
PILOT: dict[str, tuple[int, float, float, float, float]] = {
    "first_node_se/ice_g_contrast__first_node": (400, 0.035, 0.4198, 0.1257, 2.7709),
    "first_node_se/ice_g_treat_all__first_node": (400, 0.2418, 0.0847, 0.0818, 2.063),
    "first_node_se/ice_g_treat_none__first_node": (400, 0.2407, 0.087, -0.0863, 2.0801),
    "fixed_mechanism_se/hajek_g_contrast__fixed_mechanism": (400, 1.1766, 0.0632, 0.1343, 3.4405),
    "fixed_mechanism_se/hajek_g_treat_all__fixed_mechanism": (400, 1.0799, 0.0861, 0.1026, 2.6414),
    "fixed_mechanism_se/hajek_g_treat_none__fixed_mechanism": (
        400,
        1.0606,
        0.0925,
        -0.0708,
        2.6979,
    ),
    "fixed_mechanism_se/ht_g_contrast__fixed_mechanism": (400, 1.3778, 0.0774, 0.1072, 4.1558),
    "fixed_mechanism_se/ht_g_treat_all__fixed_mechanism": (400, 1.0867, 0.0992, 0.0947, 2.6486),
    "fixed_mechanism_se/ht_g_treat_none__fixed_mechanism": (400, 1.3787, 0.1, -0.0543, 3.5863),
    "interval_calibration/hajek_d2_contrast__correctly_specified": (
        400,
        0.9979,
        0.023,
        -0.0861,
        1.9551,
    ),
    "interval_calibration/hajek_d2_policy__correctly_specified": (
        400,
        0.9743,
        0.0183,
        -0.0039,
        0.8512,
    ),
    "interval_calibration/hajek_d2_rule__correctly_specified": (
        400,
        1.0388,
        0.0312,
        -0.0652,
        1.3488,
    ),
    "interval_calibration/hajek_d2_static_00__correctly_specified": (
        400,
        0.989,
        0.0351,
        0.0868,
        1.4718,
    ),
    "interval_calibration/hajek_d2_static_12__correctly_specified": (
        400,
        1.0236,
        0.0279,
        -0.0319,
        1.2719,
    ),
    "interval_calibration/hajek_d2c_relapse_h2__correctly_specified": (
        400,
        1.0051,
        0.0158,
        0.0148,
        0.5873,
    ),
    "interval_calibration/hajek_d2s_contrast_h2__correctly_specified": (
        400,
        1.0131,
        0.0238,
        -0.0288,
        1.7783,
    ),
    "interval_calibration/hajek_d2s_risk_h1__correctly_specified": (
        400,
        0.9435,
        0.0384,
        -0.0223,
        0.5524,
    ),
    "interval_calibration/hajek_d2s_risk_h2__correctly_specified": (
        400,
        1.0324,
        0.0302,
        -0.0965,
        1.1889,
    ),
    "interval_calibration/hajek_d2w_static_12__correctly_specified": (
        400,
        1.0311,
        0.0288,
        -0.0164,
        1.2754,
    ),
    "interval_calibration/hajek_g_contrast__correctly_specified": (
        400,
        1.0138,
        0.0617,
        0.1343,
        3.4405,
    ),
    "interval_calibration/hajek_g_rule__correctly_specified": (
        400,
        0.9615,
        0.1084,
        -0.0374,
        3.7606,
    ),
    "interval_calibration/hajek_g_treat_all__correctly_specified": (
        400,
        0.997,
        0.0847,
        0.1026,
        2.6414,
    ),
    "interval_calibration/hajek_g_treat_none__correctly_specified": (
        400,
        0.9782,
        0.0888,
        -0.0708,
        2.6979,
    ),
    "interval_calibration/ht_d2_contrast__correctly_specified": (
        400,
        0.9916,
        0.0265,
        -0.0881,
        2.0162,
    ),
    "interval_calibration/ht_d2_policy__correctly_specified": (
        400,
        0.9851,
        0.0215,
        -0.0137,
        0.8448,
    ),
    "interval_calibration/ht_d2_rule__correctly_specified": (400, 1.009, 0.0411, -0.0191, 1.6249),
    "interval_calibration/ht_d2_static_00__correctly_specified": (
        400,
        0.9809,
        0.0395,
        0.0746,
        1.5426,
    ),
    "interval_calibration/ht_d2_static_12__correctly_specified": (
        400,
        1.0208,
        0.0327,
        -0.0479,
        1.3095,
    ),
    "interval_calibration/ht_d2c_relapse_h2__correctly_specified": (
        400,
        1.0018,
        0.018,
        0.0198,
        0.5993,
    ),
    "interval_calibration/ht_d2s_contrast_h2__correctly_specified": (
        400,
        1.0068,
        0.0267,
        -0.0357,
        1.8326,
    ),
    "interval_calibration/ht_d2s_risk_h1__correctly_specified": (
        400,
        0.9429,
        0.0385,
        -0.0219,
        0.5527,
    ),
    "interval_calibration/ht_d2s_risk_h2__correctly_specified": (
        400,
        1.0257,
        0.0347,
        -0.1025,
        1.2279,
    ),
    "interval_calibration/ht_g_contrast__correctly_specified": (400, 0.98, 0.0805, 0.1072, 4.1558),
    "interval_calibration/ht_g_rule__correctly_specified": (400, 0.9697, 0.1144, -0.065, 3.9228),
    "interval_calibration/ht_g_treat_all__correctly_specified": (
        400,
        0.9979,
        0.0934,
        0.0947,
        2.6486,
    ),
    "interval_calibration/ht_g_treat_none__correctly_specified": (
        400,
        0.966,
        0.1129,
        -0.0543,
        3.5863,
    ),
    "interval_calibration/ice_d2_contrast__correctly_specified": (
        400,
        0.9972,
        0.0252,
        -0.0781,
        1.9432,
    ),
    "interval_calibration/ice_d2_policy__correctly_specified": (
        400,
        0.9722,
        0.0204,
        -0.0012,
        0.8479,
    ),
    "interval_calibration/ice_d2_rule__correctly_specified": (400, 1.0355, 0.0412, -0.0534, 1.3441),
    "interval_calibration/ice_d2_static_00__correctly_specified": (
        400,
        0.989,
        0.0379,
        0.0755,
        1.4568,
    ),
    "interval_calibration/ice_d2_static_12__correctly_specified": (
        400,
        1.0237,
        0.0309,
        -0.033,
        1.2677,
    ),
    "interval_calibration/ice_d2c_relapse_h2__correctly_specified": (
        400,
        1.0014,
        0.0168,
        0.0193,
        0.5887,
    ),
    "interval_calibration/ice_d2s_contrast_h2__correctly_specified": (
        400,
        0.9924,
        0.026,
        -0.0166,
        1.803,
    ),
    "interval_calibration/ice_d2s_risk_h1__correctly_specified": (
        400,
        0.9419,
        0.0385,
        -0.0228,
        0.5532,
    ),
    "interval_calibration/ice_d2s_risk_h2__correctly_specified": (
        400,
        1.0309,
        0.0344,
        -0.0932,
        1.1879,
    ),
    "interval_calibration/ice_d2w_static_12__correctly_specified": (
        400,
        1.0312,
        0.0327,
        -0.0174,
        1.2724,
    ),
    "interval_calibration/ice_g_contrast__correctly_specified": (
        400,
        1.043,
        0.0364,
        0.1257,
        2.7709,
    ),
    "interval_calibration/ice_g_rule__correctly_specified": (400, 1.0478, 0.0458, -0.055, 2.1091),
    "interval_calibration/ice_g_treat_all__correctly_specified": (
        400,
        1.0102,
        0.0496,
        0.0818,
        2.063,
    ),
    "interval_calibration/ice_g_treat_none__correctly_specified": (
        400,
        1.0185,
        0.0537,
        -0.0863,
        2.0801,
    ),
    "interval_calibration/ice_gm_contrast__correctly_specified": (
        400,
        0.9705,
        0.0214,
        -0.0435,
        0.7108,
    ),
    "interval_calibration/ice_gm_down__correctly_specified": (400, 1.0331, 0.0167, 0.0408, 1.268),
    "misspecification_control/hajek_x_contrast__main_terms": (400, 0.9692, 0.0468, 1.8464, 3.2118),
    "misspecification_control/ht_x_contrast__main_terms": (400, 0.994, 0.0542, 1.9784, 3.7288),
    "misspecification_control/ice_x_contrast__main_terms": (400, 0.9636, 0.043, 1.6139, 3.0056),
}

#: The floor on the chance that each K1 cell fails, and that each gated direction holds.
POWER_FLOOR = 0.95
#: The family-wise floor of (e), and the grain of ``R``.
FAMILY_FLOOR = 0.95
REPLICATE_GRAIN = 400
#: The upper confidence level of the between-cell spread of the true SE ratios.
TAU_LEVEL = 0.80


# ------------------------------------------------------------------------- declarations


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.slug == "sequential-baselines"
    assert (record.n, record.replicates) == (study.PRIMARY_N, study.PRIMARY_REPLICATES)
    assert record.n == study.LADDER[-1]
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert record.reference is None
    assert record.calibration_efficiency_ratio is False
    assert record.failure_bound == {study.IMPLEMENTATION: 0.01}
    assert study.G_BOUNDS == (0.01, 1.0)
    assert properties.CALIBRATION_N == study.CALIBRATION_N == 2_000
    assert set(study.EXPECTED_RED) == {
        f"misspecification_control/{estimator}_x_contrast__main_terms"
        for estimator in study.ESTIMATORS
    }
    assert study.CONFIGURATION["ladder"] == list(study.LADDER)


def test_the_seeds_are_the_studys_own() -> None:
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert study.SEED not in others
    assert study.RESAMPLING_SEED not in others
    assert LARGE_SEED not in others | {study.SEED, study.RESAMPLING_SEED}


def test_the_declared_cells_are_the_cells_the_properties_draw() -> None:
    declared = {(cell.property, cell.cell) for cell in properties.declared_cells()}
    published = {
        (family, cell)
        for family, cells in study.PROPERTY_CELLS.items()
        if family not in {"root_n_rate"}
        for cell in cells
        if not cell.endswith(("__shrunken_se_control", "__noise_control"))
    }
    assert declared == published
    calibration = list(study.CALIBRATION_LABELS)
    assert len(calibration) == len(set(calibration)) == 43


def test_every_key_the_study_publishes_has_a_description() -> None:
    descriptions.implementation(study.IMPLEMENTATION)
    for scenario, estimands in study.SCENARIOS.items():
        descriptions.scenario(scenario)
        for name in estimands:
            descriptions.estimand(name)
    for name, _ in properties.TARGETS.values():
        descriptions.estimand(name)
    for family, cells in study.PROPERTY_CELLS.items():
        descriptions.claim(family)
        for cell in cells:
            descriptions.cell(family, cell, role="positive")


# ------------------------------------------------------------------- (a) specification


def _support_fit(estimator: str, law: str, *, weighted: bool = False) -> Any:
    frame = laws.DISCRETE_LAWS[law].support_frame.copy()
    frame["w"] = frame["prob"] * (laws.observation_weight(frame["W"]) if weighted else 1.0)
    return study.fit(estimator, law, frame, weights="w")


@pytest.mark.parametrize("estimator", study.ESTIMATORS)
def test_every_discrete_cell_is_exact_at_population_weights(estimator: str) -> None:
    """(a) D2, D2S, D2C and the weighted D2: each declared GLM reproduces the truth.

    At population weights each fit is the population fit.  A point equal to the truth to
    ``1e-10`` shows that every outcome regression and mechanism the estimator reads is
    correctly specified.
    """
    prefixes = {"d2": "D2", "d2s": "D2S", "d2c": "D2C", "d2w": "D2"}
    fits: dict[str, Any] = {}
    checked = 0
    for label in study.CALIBRATION_LABELS:
        owner, prefix = label.split("_")[:2]
        if owner != estimator or prefix not in prefixes:
            continue
        if prefix not in fits:
            fits[prefix] = _support_fit(estimator, prefixes[prefix], weighted=prefix == "d2w")
        name, truth = properties.TARGETS[label]
        assert abs(fits[prefix][name].psi - truth) < 1e-10, label
        checked += 1
    assert checked == {"ice": 10, "hajek": 10, "ht": 9}[estimator]


def _g_q(law: laws.ContinuousLaw, node: int, frame: pd.DataFrame, arms: tuple) -> np.ndarray:
    """The declared linear form of ``Q_node`` under ``arms`` on ``G``'s coefficients."""
    v = frame["V"].to_numpy()
    realised = [v if arm == "V" else np.full(len(v), float(arm)) for arm in arms]
    c = laws.G_OUTCOME
    w_coef, l_coef = c[1], c[3]
    constant = c[0] + c[2] * v + c[4] * realised[2]
    for later in range(laws.G_TIMES, node, -1):
        constant = constant + l_coef * laws.L_A * realised[later - 2]
        w_coef = w_coef + l_coef * laws.L_W
        l_coef = l_coef * laws.L_LAG
    return constant + w_coef * frame["W"].to_numpy() + l_coef * frame[f"L{node}"].to_numpy()


def _g_residual_moments(law: laws.ContinuousLaw) -> float:
    """The largest mean of residual times design column over every plan and node."""
    frame = law.sample(LARGE_N, LARGE_SEED)
    worst = 0.0
    for arms in laws.G_PLANS.values():
        v = frame["V"].to_numpy()
        followed = np.ones(LARGE_N, dtype=bool)
        for node in range(1, laws.G_TIMES + 1):
            target = v if arms[node - 1] == "V" else float(arms[node - 1])
            followed &= (frame[f"A{node}"].to_numpy() == target) & (frame[f"C{node}"] == 1)
            if node == laws.G_TIMES:
                following = frame["Y"].to_numpy()
            else:
                following = _g_q(law, node + 1, frame, arms)
            residual = np.where(followed, following - _g_q(law, node, frame, arms), 0.0)
            columns = [np.ones(LARGE_N), frame["W"].to_numpy(), v]
            columns += [np.nan_to_num(frame[f"L{t}"].to_numpy()) for t in range(1, node + 1)]
            for column in columns:
                worst = max(worst, abs(float(np.mean(residual * column))))
    return worst


def test_the_g_regressions_are_linear_and_x_is_not() -> None:
    """(a) G: each ICE regression is the declared main-terms linear form.

    The witness: X's outcome carries ``0.4 L_3^2``, so the same check is far outside the
    tolerance there.
    """
    assert _g_residual_moments(laws.G) < RESIDUAL_TOLERANCE
    assert _g_residual_moments(laws.X) > 50 * RESIDUAL_TOLERANCE


def test_the_gm_regressions_are_linear() -> None:
    """(a) GM: the natural-course and lowered-dose ICE regressions are linear."""
    frame = laws.gm_sample(LARGE_N, LARGE_SEED)
    c = laws.GM_OUTCOME
    w, l1, a1, l2, a2, y = (frame[k].to_numpy() for k in ("W", "L1", "A1", "L2", "A2", "Y"))

    def q2(dose: np.ndarray) -> np.ndarray:
        return c[0] + c[1] * w + c[2] * l2 + c[3] * dose

    lowered = np.where(a2 >= laws.gm_boundary(l2), a2 - laws.GM_STEP, a2)
    mean_l2 = laws.GM_L2[0] * w + laws.GM_L2[1] * l1 + laws.GM_L2[2] * a1
    natural_q1 = c[0] + c[1] * w + c[2] * mean_l2 + c[3] * laws.GM_DOSE_L * mean_l2
    down_q1 = natural_q1 - c[3] * laws.GM_STEP * laws.GM_MOVED
    one = np.ones(LARGE_N)
    checks = (
        (y - q2(a2), (one, w, l1, a1, l2, a2)),
        (q2(a2) - natural_q1, (one, w, l1, a1)),
        (q2(lowered) - down_q1, (one, w, l1, a1)),
    )
    for residual, columns in checks:
        for column in columns:
            assert abs(float(np.mean(residual * column))) < RESIDUAL_TOLERANCE


def test_the_closed_form_truths_match_the_samplers() -> None:
    """The G and X truths integrate the sampler; one draw of ``LARGE_N`` rows per plan."""
    rng = np.random.default_rng(LARGE_SEED)
    for law in (laws.G, laws.X):
        truths = laws.g_truths(law)
        for label, arms in laws.G_PLANS.items():
            w = rng.uniform(-1.0, 1.0, LARGE_N)
            v = rng.binomial(1, laws.P_V, LARGE_N).astype(float)
            l, a = np.zeros(LARGE_N), np.zeros(LARGE_N)
            for arm in arms:
                l = laws.L_W * w + laws.L_LAG * l + laws.L_A * a + rng.uniform(-1, 1, LARGE_N)
                a = v if arm == "V" else np.full(LARGE_N, float(arm))
            y = law.outcome_mean(w, v, l, a) + rng.uniform(-1.0, 1.0, LARGE_N)
            error = float(np.std(y) / math.sqrt(LARGE_N))
            assert abs(float(np.mean(y)) - truths[label]) < 4 * error, (law.name, label)


def test_q_bounds_hold_the_outcome_and_g_bounds_never_bind() -> None:
    for law in (laws.G, laws.X):
        frame = law.sample(LARGE_N, LARGE_SEED)
        low, high = law.q_bounds()
        y = frame["Y"].dropna()
        assert low <= y.min() and y.max() <= high
        assert low < law.outcome_support()[0] + 1e-6
        # The mechanisms' smallest probability over a large draw, far above the 0.01 bound.
        for t in range(1, laws.G_TIMES + 1):
            seen = frame[f"L{t}"].notna()
            a_prev = frame[f"A{t - 1}"] if t > 1 else pd.Series(0.0, index=frame.index)
            p = 1.0 / (
                1.0 + np.exp(-law.treatment_logit(frame["W"], frame["V"], frame[f"L{t}"], a_prev))
            )
            r = 1.0 / (
                1.0 + np.exp(-law.retention_logit(frame["W"], frame[f"L{t}"], frame[f"A{t}"]))
            )
            assert min(p[seen].min(), 1 - p[seen].max(), r[seen].min()) > 2 * study.G_BOUNDS[0]
    low, high = laws.gm_outcome_support()
    assert study.CONFIGURATION["q_bounds"]["GM"] == [low, high]  # type: ignore[index]
    y = laws.gm_sample(LARGE_N, LARGE_SEED)["Y"]
    assert low <= y.min() and y.max() <= high
    for discrete in (laws.D2, laws.D2S, laws.D2C):
        smallest = 1.0
        for l1, a1, l2 in itertools.product((0, 1), repeat=3):
            smallest = min(
                smallest,
                discrete.p_a1(l1),
                1 - discrete.p_a1(l1),
                discrete.p_c1(l1, a1),
                float(np.min(discrete.g2(l2, a1))),
            )
            for a2 in range(discrete.arms):
                smallest = min(smallest, discrete.p_c2(l2, a2))
        assert smallest > 0.05, discrete.name


# -------------------------------------------------------------------- (b) MLE existence


def test_every_declared_fit_exists_at_its_declared_size() -> None:
    """(b) At the size each law is fitted in a gated cell, its fits exist on at least 99.9%.

    A failed replicate stops the property phase (``require_complete``), so the sizes are
    chosen where the probe saw no failure in 2,000 draws.  The smaller sizes are the reason.
    """
    assert set(DECLARED_SIZES) == {law for law, _ in MLE_PROBE}
    for law, size in DECLARED_SIZES.items():
        draws, failed = MLE_PROBE[law, size]
        assert draws == 2_000
        assert failed == 0, (law, size)
    assert MLE_PROBE["D2", 1_000][1] / 2_000 > 0.001
    assert MLE_PROBE["D2C", 4_000][1] / 2_000 > 0.001
    # Every GLM fit set reads its law at or above that size.  The adaptive cells fit no GLM.
    for name, (law, n, _) in properties.FIT_SETS.items():
        if not name.startswith("adaptive_"):
            assert n >= DECLARED_SIZES[law], name


# ------------------------------------------------------------------------ the pilot rule


def _pass(replicates: int, ratio: float, noise: float, bias: float = 0.0) -> float:
    """The chance a cell passes the calibration rule at true ratio and standardized bias."""
    z = float(norm.ppf(0.975))
    coverage = float(norm.cdf(z * ratio - bias) - norm.cdf(-z * ratio - bias))
    return calibration_pass_probability(
        replicates, true_se_ratio=ratio, coverage=coverage, se_noise=noise
    )


def _gated_pilot() -> dict[str, tuple[int, float, float, float, float]]:
    return {
        cell: values
        for cell, values in PILOT.items()
        if cell.startswith("interval_calibration/") and cell.endswith("__correctly_specified")
    }


def _law_of(cell: str) -> str:
    prefix = cell.split("/")[1].split("_")[1]
    return "d2" if prefix == "d2w" else prefix


def family_pass_probability(replicates: int) -> float:
    r"""(e) The arm-pooled predictive pass probability of every gated calibration cell.

    The arms are the three estimators.  Arm :math:`a` has pilot cell ratios
    :math:`\hat r_c` with mean :math:`m_a`.  :math:`s^2` is the within-arm variance pooled over
    the arms.  :math:`\tau^2` is the between-cell variance of the true ratios beyond pilot
    noise, at its 80% upper bound from the chi-square law of :math:`s^2`.  The arm's true ratio
    is :math:`\mu_a \sim N(m_a, s^2 / k_a)`, where :math:`k_a` counts the distinct laws of the
    arm (cells of one law share draws), and a cell's true ratio is :math:`\mu_a + \delta_c`
    with independent :math:`\delta_c \sim N(0, \tau^2)`.  So each arm's ratio is
    :math:`N(m_a, s^2 / k_a + \tau^2)`, and the cells of one arm share :math:`\mu_a`.  The
    family-wise chance is the product over the arms of the chance that every cell of the arm
    passes, by nested Gauss-Hermite quadrature.
    """
    gated = _gated_pilot()
    arms: dict[str, list[str]] = {}
    for cell in gated:
        arms.setdefault(cell.split("/")[1].split("_")[0], []).append(cell)
    deviations, degrees = 0.0, 0
    noise_variance = []
    means = {}
    for arm, cells in arms.items():
        ratios = np.array([gated[cell][1] for cell in cells])
        means[arm] = float(ratios.mean())
        deviations += float(((ratios - ratios.mean()) ** 2).sum())
        degrees += len(cells) - 1
        for cell in cells:
            k, ratio, noise = gated[cell][:3]
            noise_variance.append(ratio**2 * (1.0 / (2.0 * (k - 1)) + noise**2 / k))
    pooled = deviations / degrees
    upper = degrees * pooled / float(chi2.ppf(1.0 - TAU_LEVEL, degrees))
    tau_squared = max(0.0, upper - float(np.mean(noise_variance)))
    nodes, weights = np.polynomial.hermite_e.hermegauss(41)
    weights = weights / weights.sum()
    total = 1.0
    for arm, cells in arms.items():
        laws_of_arm = len({_law_of(cell) for cell in cells})
        shared = math.sqrt(pooled / laws_of_arm)
        tau = math.sqrt(tau_squared)
        arm_chance = 0.0
        for node, weight in zip(nodes, weights, strict=True):
            mean = means[arm] + shared * node
            joint = 1.0
            for cell in cells:
                noise = gated[cell][2]
                joint *= (
                    sum(
                        inner_weight * _pass(replicates, mean + tau * inner, noise)
                        for inner, inner_weight in zip(nodes, weights, strict=True)
                    )
                    if tau > 0
                    else _pass(replicates, mean, noise)
                )
            arm_chance += weight * joint
        total *= arm_chance
    return float(total)


def test_the_calibration_replicates_follow_the_sizing_rule() -> None:
    """(e) ``R`` is the smallest multiple of 400 with family-wise pass chance >= 0.95."""
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert family_pass_probability(replicates) >= FAMILY_FLOOR
    assert family_pass_probability(replicates - REPLICATE_GRAIN) < FAMILY_FLOOR
    assert len(_gated_pilot()) == len(study.CALIBRATION_LABELS)


def test_the_k1_cells_fail_with_the_declared_power() -> None:
    """(c) Each K1 cell fails the calibration rule with chance at least 0.95."""
    for estimator in study.ESTIMATORS:
        k, ratio, noise, bias, _ = PILOT[
            f"misspecification_control/{estimator}_x_contrast__main_terms"
        ]
        assert k >= 400
        power = 1.0 - _pass(properties.CALIBRATION_REPLICATES, ratio, noise, bias)
        assert power >= POWER_FLOOR, estimator


def _direction_power(cell: str, side: str) -> float:
    """The chance the whole 99% SE-ratio interval lies on ``side`` of 1 at ``R``."""
    _, ratio, noise = PILOT[cell][:3]
    replicates = properties.CALIBRATION_REPLICATES
    spread = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    z = float(norm.ppf(0.995))
    if side == "above":
        return float(norm.cdf((ratio - z * spread - 1.0) / spread))
    return float(norm.cdf((1.0 - z * spread - ratio) / spread))


def test_the_k2_and_k3_directions_are_gated_only_where_powered() -> None:
    """(d) K2 is predicted above 1 and K3 below 1; a cell is gated when its power >= 0.95."""
    powered = {}
    for family, side in (("fixed_mechanism_se", "above"), ("first_node_se", "below")):
        for cell in study.PROPERTY_CELLS[family]:
            if _direction_power(f"{family}/{cell}", side) >= POWER_FLOOR:
                powered[cell] = side
    assert powered == properties.GATED_DIRECTIONS


def test_the_noise_control_scale_is_the_pilot_spread() -> None:
    for label in study.CALIBRATION_LABELS:
        pinned = PILOT[f"interval_calibration/{label}__correctly_specified"][4]
        assert properties.NOISE_SD[label] == pytest.approx(pinned, rel=1e-3), label
