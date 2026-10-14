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
}

#: The design pilot at ``CALIBRATION_N``: ``cell -> (replicates, mean SE over empirical SD,
#: relative SD of the SE, mean bias over empirical SD, sqrt(n) times the empirical SD)``.  It
#: draws from the ``design_pilot`` stream, which no published cell reads.
PILOT: dict[str, tuple[int, float, float, float, float]] = {}

#: The floor on the chance that each K1 cell fails, and that each gated direction holds.
POWER_FLOOR = 0.95
#: The family-wise floor of (e), and the grain of ``R``.
FAMILY_FLOOR = 0.95
REPLICATE_GRAIN = 400
#: The upper confidence level of the between-cell spread of the true SE ratios.
TAU_LEVEL = 0.80
#: The pilots of (b) to (e) are not all pinned yet; these tests wait for them.
AWAITING_PILOT = pytest.mark.skip(reason="design pilot not pinned yet (X41 handoff)")


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


@AWAITING_PILOT
def test_every_declared_fit_exists_at_the_first_ladder_size() -> None:
    """(b) At ``LADDER[0]`` and at ``CALIBRATION_N``, each law fits on at least 99.9%."""
    for (law, n), (draws, failed) in MLE_PROBE.items():
        if n >= study.LADDER[0]:
            assert draws == 2_000
            assert 1.0 - failed / draws >= 0.999, (law, n)
    laws_probed = {law for law, _ in MLE_PROBE}
    assert laws_probed == {"D2", "D2S", "D2C", "G", "X", "GM"}
    assert all((law, study.LADDER[0]) in MLE_PROBE for law in laws_probed)


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
    noise, at its 80% upper bound from the chi-square law of :math:`s^2`.  Each cell's true
    ratio is :math:`N(m_a, s^2 / k_a + \tau^2)`, where :math:`k_a` counts the distinct laws of
    the arm (cells of one law share draws).  The pass chance of each cell is averaged over that
    law by Gauss-Hermite quadrature, and the family-wise chance is the product.
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
        spread = math.sqrt(pooled / laws_of_arm + tau_squared)
        for cell in cells:
            noise = gated[cell][2]
            chance = sum(
                weight * _pass(replicates, means[arm] + spread * node, noise)
                for node, weight in zip(nodes, weights, strict=True)
            )
            total *= chance
    return float(total)


@AWAITING_PILOT
def test_the_calibration_replicates_follow_the_sizing_rule() -> None:
    """(e) ``R`` is the smallest multiple of 400 with family-wise pass chance >= 0.95."""
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert family_pass_probability(replicates) >= FAMILY_FLOOR
    assert family_pass_probability(replicates - REPLICATE_GRAIN) < FAMILY_FLOOR
    assert len(_gated_pilot()) == len(study.CALIBRATION_LABELS)


@AWAITING_PILOT
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


@AWAITING_PILOT
def test_the_k2_and_k3_directions_are_gated_only_where_powered() -> None:
    """(d) K2 is predicted above 1 and K3 below 1; a cell is gated when its power >= 0.95."""
    powered = {}
    for family, side in (("fixed_mechanism_se", "above"), ("first_node_se", "below")):
        for cell in study.PROPERTY_CELLS[family]:
            if _direction_power(f"{family}/{cell}", side) >= POWER_FLOOR:
                powered[cell] = side
    assert powered == properties.GATED_DIRECTIONS


@AWAITING_PILOT
def test_the_noise_control_scale_is_the_pilot_spread() -> None:
    for label in study.CALIBRATION_LABELS:
        pinned = PILOT[f"interval_calibration/{label}__correctly_specified"][4]
        assert properties.NOISE_SD[label] == pytest.approx(pinned, rel=1e-3), label
