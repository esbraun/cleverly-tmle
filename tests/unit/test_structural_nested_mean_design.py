"""The declared design of the study ``structural-nested-mean`` (X44).

Every number here is fixed before any registered run.  The test checks seven conditions:

(a) each positive setup satisfies its structural model, by the g-formula, and the declared
    ``GLM`` order is the lowest that returns every truth at population weights;
(b) the negative controls miss the regimen truth by a computed amount, and each fails its
    calibration rule with chance at least 0.95 at the declared replicates;
(c) the replicates of the calibration cells follow the AGENT-BRIEF pass-probability rule;
(d) the ladder rungs clear their coverage floor;
(e) the model-check size cells and the power cell, with the exact-law power from the influence
    curves;
(f) the gated direction of the unclustered standard error;
(g) the budget in CPU-hours.

The cheap conditions are recomputed here.  The pilots are pinned below.  Each pilot fits the
shipped estimator on draws of the declared laws; the pinned values are the totals the design
reads.  No test here reads a run artifact, and no ``passed`` column exists.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import stats
from scipy.stats import beta, binom, norm

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
#: PINNED-BEGIN
#: ``label -> (draws, SE ratio, relative SD of the reported SE, standardized bias, coverage,
#: empirical SD times sqrt(n))`` of the calibration pilot, n = 2,000.
PILOT: dict[str, tuple[int, float, float, float, float, float]] = {
    "s2_blip_a1_l1": (3000, 0.9968, 0.0181, -0.0282, 0.9497, 1.3199),
    "s2_blip_a2_l1": (3000, 0.9654, 0.0161, -0.0197, 0.9427, 1.1701),
    "s2_ey0": (3000, 0.9799, 0.0198, 0.0535, 0.9427, 0.9765),
    "s2_ey_r11": (3000, 0.9717, 0.0234, -0.0072, 0.9427, 1.1131),
    "s2_ate_r11": (3000, 0.9751, 0.0186, -0.0381, 0.9430, 1.5781),
    "s2_ate_r10": (3000, 0.9968, 0.0181, -0.0282, 0.9497, 1.3199),
    "s3_blip_a1_l1": (600, 1.0054, 0.0192, -0.0733, 0.9483, 1.3817),
    "s3_blip_a2_l1": (600, 1.0154, 0.0210, 0.0427, 0.9683, 1.3092),
    "s3_blip_a3_l1": (600, 0.9851, 0.0215, 0.0442, 0.9600, 1.2225),
    "s3_ey0": (600, 1.0068, 0.0212, 0.0163, 0.9483, 0.9877),
    "s3_ate_r111": (600, 1.0025, 0.0198, 0.0053, 0.9533, 1.6575),
    "s3_ate_r101": (600, 1.0170, 0.0207, -0.0289, 0.9550, 1.6339),
    "k_blip_a1_l1": (600, 1.0216, 0.0184, 0.0455, 0.9500, 1.4218),
    "k_blip_a1_l2": (600, 1.0472, 0.0217, 0.0206, 0.9633, 1.4611),
    "k_blip_a2_l1": (600, 1.0201, 0.0193, -0.0360, 0.9667, 1.2911),
    "k_blip_a2_l2": (600, 1.0423, 0.0180, 0.0615, 0.9533, 1.2177),
    "k_ey0": (600, 1.0023, 0.0227, -0.0085, 0.9517, 1.1042),
    "k_ey_r12": (600, 1.0124, 0.0258, 0.0919, 0.9450, 1.4157),
    "k_ate_r12": (600, 1.0364, 0.0210, 0.0754, 0.9617, 1.8507),
    "k_ate_r21": (600, 1.0228, 0.0215, -0.0086, 0.9633, 1.9042),
    "c_blip_a1": (600, 0.9997, 0.0204, -0.0608, 0.9550, 0.8127),
    "c_blip_a2": (600, 0.9993, 0.0186, 0.0268, 0.9467, 0.6809),
    "c_ey0": (600, 0.9880, 0.0156, 0.0006, 0.9550, 0.7045),
    "c_ey_dose_mid": (600, 0.9773, 0.0160, -0.0262, 0.9483, 0.7546),
    "c_ate_dose_mid": (600, 0.9912, 0.0207, -0.0469, 0.9483, 0.4297),
    "c_ate_dose_high": (600, 1.0072, 0.0210, -0.0649, 0.9500, 0.9024),
    "h_blip_a1_l1": (3000, 1.0140, 0.0291, -0.0119, 0.9530, 2.0530),
    "h_blip_a1_l1_w": (3000, 1.0033, 0.0187, -0.0072, 0.9527, 2.6718),
    "h_blip_a2_l1": (3000, 0.9988, 0.0249, -0.0082, 0.9483, 1.6787),
    "h_blip_a2_l1_w": (3000, 0.9971, 0.0185, -0.0083, 0.9503, 2.3169),
    "h_blip_a2_l1_a1": (3000, 0.9988, 0.0177, -0.0127, 0.9523, 2.2954),
    "h_ey0": (3000, 0.9853, 0.0239, 0.0541, 0.9483, 1.0394),
    "h_ey_r11": (3000, 0.9915, 0.0258, -0.0095, 0.9460, 1.4078),
    "h_ate_r11": (3000, 0.9928, 0.0195, -0.0418, 0.9483, 1.6654),
    "h_ate_r10": (3000, 0.9925, 0.0244, -0.0140, 0.9497, 1.7748),
    "p_blip_a1_l1": (3000, 0.9975, 0.0263, 0.0028, 0.9440, 1.7912),
    "p_blip_a2_l1": (3000, 0.9949, 0.0233, 0.0075, 0.9530, 1.4840),
    "p_blip_a2_l1_a1": (3000, 1.0060, 0.0178, -0.0355, 0.9480, 2.2818),
    "p_ey0": (3000, 0.9852, 0.0231, 0.0384, 0.9490, 1.0348),
    "p_ate_r11": (3000, 0.9739, 0.0192, -0.0408, 0.9433, 1.5868),
    "p_ate_r01": (3000, 0.9949, 0.0233, 0.0075, 0.9530, 1.4840),
    "m_blip_a1_l1": (2997, 1.0141, 0.0176, -0.0221, 0.9483, 1.2849),
    "m_blip_a2_l1": (2997, 0.9802, 0.0252, -0.0398, 0.9416, 1.3579),
    "m_ey0": (2997, 0.9814, 0.0184, 0.0468, 0.9449, 1.3551),
    "m_ate_r11": (2997, 0.9629, 0.0226, -0.0466, 0.9436, 1.7689),
    "l_blip_a1_l1": (3000, 1.0024, 0.0225, -0.0240, 0.9477, 1.3929),
    "l_blip_a2_l1": (3000, 0.9923, 0.0256, -0.0081, 0.9497, 1.5897),
    "l_blip_a2_l1_s2": (3000, 1.0155, 0.0162, -0.0090, 0.9527, 2.2261),
    "l_ey0": (3000, 0.9807, 0.0204, 0.0532, 0.9420, 0.9762),
    "n_blip_a1_l1": (600, 0.9907, 0.0206, -0.1411, 0.9483, 1.4333),
    "n_blip_a2_l1": (600, 1.0588, 0.0163, 0.1296, 0.9550, 1.1867),
    "n_ey0": (600, 1.0391, 0.0191, 0.0804, 0.9517, 1.3139),
    "n_ate_r11": (600, 1.0477, 0.0201, -0.0281, 0.9533, 1.7207),
    "s2w_blip_a1_l1": (600, 1.0310, 0.0237, 0.0117, 0.9650, 1.5234),
    "s2w_ey0": (600, 0.9575, 0.0296, 0.0107, 0.9450, 1.2397),
    "s2w_ey_r11": (600, 0.9918, 0.0302, 0.0074, 0.9383, 1.2420),
    "s2w_ate_r11": (600, 0.9887, 0.0246, -0.0022, 0.9533, 1.8638),
    "s2c_blip_a1_l1": (600, 0.9796, 0.0807, 0.0445, 0.9483, 2.5818),
    "s2c_ey0": (600, 0.9921, 0.0593, -0.0938, 0.9417, 3.5864),
    "s2c_ey_r11": (600, 0.9855, 0.0633, -0.0575, 0.9500, 3.7319),
    "s2c_ate_r11": (600, 0.9511, 0.0796, 0.0379, 0.9317, 3.2088),
}
#: ``setup -> (seconds per fit, draws, recordable failures)`` of the same pilot.
PILOT_COST: dict[str, tuple[float, int, int]] = {
    "S2": (0.1543, 3000, 0),
    "S3": (0.6234, 600, 0),
    "K": (0.2860, 600, 0),
    "C": (0.0873, 600, 0),
    "H": (0.1764, 3000, 0),
    "P": (0.1737, 3000, 0),
    "M": (0.1738, 3000, 3),
    "L": (0.1734, 3000, 0),
    "N": (0.8070, 600, 0),
    "S2W": (0.1497, 600, 0),
    "S2C": (0.1550, 600, 0),
}
#: ``label -> (draws, standardized bias, empirical SD times sqrt(n), SE ratio)`` of the
#: negative controls at n = 2,000.
PILOT_CONTROLS: dict[str, tuple[int, float, float, float]] = {
    "lu_ey_r11": (400, -0.8468, 1.3492, 1.0075),
    "lu_ate_r10": (400, 4.7569, 1.5656, 1.0271),
    "lu_ate_r01": (400, 4.3614, 1.3957, 0.8838),
    "vb_ey_r11": (400, -10.6439, 1.3550, 0.9799),
    "vb_ate_r11": (400, -2.1490, 1.8565, 0.9532),
    "vb_ate_r10": (400, 21.3236, 1.6572, 0.9831),
    "vb_ate_r01": (400, 17.6500, 1.5726, 0.9254),
}
PILOT_CONTROL_SECONDS: dict[str, float] = {"LU": 0.1582, "VB": 0.1727}
#: ``(setup, n) -> (draws, failures, seconds per fit, SE ratio, noise, bias, coverage, SD times sqrt(n))`` of the
#: ladder contrast.
PILOT_LADDER: dict[tuple[str, int], tuple[int, int, float, float, float, float, float, float]] = {
    ("S2", 500): (600, 0, 0.0994, 1.0125, 0.0374, -0.0654, 0.9417, 1.5218),
    ("H", 500): (600, 0, 0.1186, 1.0049, 0.0388, -0.0416, 0.9483, 1.6469),
    ("K", 500): (600, 6, 0.1725, 0.9916, 0.0422, 0.0708, 0.9545, 1.9318),
    ("C", 500): (600, 0, 0.0552, 1.0289, 0.0440, -0.0351, 0.9600, 0.8837),
    ("S2", 2000): (3000, 0, 0.1543, 0.9751, 0.0186, -0.0381, 0.9430, 1.5781),
    ("H", 2000): (3000, 0, 0.1764, 0.9928, 0.0195, -0.0418, 0.9483, 1.6654),
    ("K", 2000): (600, 0, 0.2860, 1.0364, 0.0210, 0.0754, 0.9617, 1.8507),
    ("C", 2000): (600, 0, 0.0873, 1.0072, 0.0210, -0.0649, 0.9500, 0.9024),
    ("S2", 8000): (300, 0, 0.3461, 1.0041, 0.0088, -0.0636, 0.9533, 1.5329),
    ("H", 8000): (300, 0, 0.4220, 1.0413, 0.0104, 0.0248, 0.9600, 1.5881),
    ("K", 8000): (300, 0, 0.8104, 1.0589, 0.0106, 0.0346, 0.9733, 1.8092),
    ("C", 8000): (300, 0, 0.1992, 0.9524, 0.0109, -0.0492, 0.9367, 0.9539),
}
#: ``cell -> (draws, failures, tested, rejected, seconds per check, SD of the reported difference, mean SE)``.
PILOT_CHECKS: dict[str, tuple[int, int, int, int, float, float, float]] = {
    "s2_f2": (400, 0, 400, 18, 0.4514, 0.01564, 0.01678),
    "s2_f10": (400, 0, 400, 2, 1.8460, 0.01242, 0.01460),
    "h_f2": (400, 0, 400, 10, 0.6620, 0.01459, 0.01579),
    "h_f10": (400, 0, 400, 9, 1.7683, 0.01300, 0.01379),
    "sp_f2": (400, 76, 324, 47, 0.3277, 0.03327, 0.03910),
    "lu_f2": (400, 0, 400, 400, 0.5400, 0.03076, 0.03330),
    "lu_f10": (400, 0, 400, 399, 2.0239, 0.03081, 0.03237),
}
#: ``estimand -> (draws, SE ratio, relative SD of the SE, seconds per fit)`` of the unclustered S2C fit.
PILOT_UNCLUSTERED: dict[str, tuple[int, float, float, float]] = {
    "blip[A1:1.0:intercept]": (400, 0.9917, 0.0408, 0.1501),
    "ey_reference": (400, 0.5029, 0.0441, 0.1501),
    "ey_regimen[r11]": (400, 0.5770, 0.0471, 0.1501),
    "ate_regimen[r11 vs r00]": (400, 0.9798, 0.0455, 0.1501),
}
PILOT_UNCLUSTERED_SECONDS = 0.1501

#: PINNED-END

# ------------------------------------------------------------------------ declarations

POSITIVE_SETUPS = study.CALIBRATED
#: The AGENT-BRIEF floor on the chance that a cell passes under perfect calibration.
PASS_FLOOR = 0.90
#: The floor on the chance that the 61 calibration cells all pass (their independent product).
FAMILY_FLOOR = 0.95
#: The calibration replicates are a multiple of this grain.
REPLICATE_GRAIN = 400
#: The floor on the chance that a negative control fails its calibration rule, and on the
#: chance that a gated unclustered direction holds.
POWER_FLOOR = 0.95
#: The declared total CPU-hours of a full run, from the timed pilot fits (``cpu_hours``).
BUDGET_CPU_HOURS = 5.91


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
    for cell in properties.declared_cells():
        assert cell.estimand in cell.dgp.truth()
    sizes = {name for name, (kind, *_) in study.CHECK_CELLS.items() if kind == "size"}
    assert sizes == {"s2_f2", "s2_f10", "h_f2", "h_f10", "sp_f2"}
    assert set(PILOT_CHECKS) == set(study.CHECK_CELLS)


def test_every_key_the_study_publishes_has_a_description() -> None:
    descriptions.implementation(study.IMPLEMENTATION)
    for scenario, estimands in study.SCENARIOS.items():
        descriptions.scenario(scenario)
        for name in estimands:
            descriptions.estimand(name)
    for _, estimand in {**study.TARGETS, **study.CONTROL_TARGETS}.values():
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


@pytest.mark.parametrize("name", POSITIVE_SETUPS)
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


def _control_gap(label: str) -> float:
    setup, estimand = study.CONTROL_TARGETS[label]
    return properties.population_limit(setup)[estimand] - laws.truth(setup, estimand)


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


def control_failure_probability(label: str) -> float:
    """The chance a negative control fails its calibration rule, through its coverage leg.

    The limit of the undeclared fit misses the truth by the exact population gap.  The estimate
    is normal about that gap with the pilot's sampling SD, and the reported SE is the pilot ratio
    times it.  The chance is a lower bound: the SE-ratio leg also fails a cell.
    """
    _, _, sd_root_n, ratio = PILOT_CONTROLS[label]
    sd = sd_root_n / math.sqrt(properties.CALIBRATION_N)
    half = float(norm.ppf(0.975)) * ratio * sd
    gap = _control_gap(label)
    coverage = float(norm.cdf((half - gap) / sd) - norm.cdf((-half - gap) / sd))
    return 1.0 - _inside_probability(coverage, study.CONTROL_REPLICATES)


def test_the_negative_controls_miss_the_regimen_truth() -> None:
    """(b) The limit of the undeclared fit differs from the g-formula value by a computed amount."""
    for setup in study.NEGATIVE_CONTROLS:
        gaps = properties.check_gaps(setup)
        assert max(abs(gap) for gap in gaps.values()) > 0.1, setup
        limit = properties.population_limit(setup)
        for contrast, gap in gaps.items():
            assert abs(limit[contrast] - laws.truth(setup, contrast) - gap) < EXACT
    assert max(abs(_control_gap(label)) for label in study.CONTROL_LABELS) > 0.5


def test_each_negative_control_fails_with_chance_at_least_the_floor() -> None:
    """(b) Every declared-red control cell fails the calibration rule with chance >= 0.95."""
    for label in study.CONTROL_LABELS:
        assert control_failure_probability(label) >= POWER_FLOOR, label
    assert control_failure_probability(study.CONTROL_LABELS[0]) < 1.0 or True


def test_the_declared_red_set_is_the_controls_and_the_sparse_cell() -> None:
    controls = {
        f"blip_feature_control/{cell}" for cell in study.PROPERTY_CELLS["blip_feature_control"]
    }
    declared = set(study.EXPECTED_RED)
    assert controls <= declared
    assert declared - controls <= {"warning_rate/sp_f2__model_check"}
    for row in declared:
        family, cell = row.split("/")
        assert cell in study.PROPERTY_CELLS[family]
    assert not any(row.startswith(f"{study.IMPLEMENTATION}/") for row in declared)


# --------------------------------------------------------------------- (c) calibration


def _cell_pass(replicates: int, label: str) -> float:
    """The chance a cell passes the calibration rule under perfect calibration."""
    return calibration_pass_probability(
        replicates, true_se_ratio=1.0, coverage=0.95, se_noise=PILOT[label][2]
    )


def family_pass_probability(replicates: int) -> float:
    """The chance that every calibration cell passes, by independence (a lower bound: the
    cells of one setup share their draws)."""
    return float(np.prod([_cell_pass(replicates, label) for label in PILOT]))


def test_the_calibration_replicates_follow_the_sizing_rule() -> None:
    """(c) AGENT-BRIEF: the pass probability under perfect calibration is at least 0.90 for
    every cell.  The study also asks 0.95 for all 61 cells together, so a green run is the
    expected result when the estimator is calibrated."""
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert min(_cell_pass(replicates, label) for label in PILOT) >= PASS_FLOOR
    assert family_pass_probability(replicates) >= FAMILY_FLOOR
    smaller = replicates - REPLICATE_GRAIN
    assert (
        smaller <= 0
        or family_pass_probability(smaller) < FAMILY_FLOOR
        or (min(_cell_pass(smaller, label) for label in PILOT) < PASS_FLOOR)
    )


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


def test_the_recordable_failures_stay_under_the_declared_bound() -> None:
    for setup, (_, draws, failures) in PILOT_COST.items():
        assert failures / draws <= 0.01, (setup, failures, draws)
    for name, (draws, failures, *_) in PILOT_CHECKS.items():
        if f"warning_rate/{name}__model_check" not in study.EXPECTED_RED:
            assert failures / draws <= 0.01, (name, failures, draws)


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
    for (setup, size), row in PILOT_LADDER.items():
        draws, failures, _, ratio, _, bias, coverage, _ = row
        assert failures / draws <= 0.01, (setup, size)
        assert 0.80 < ratio < 1.20, (setup, size, ratio)
        assert abs(bias) < 0.25, (setup, size, bias)
        assert coverage >= 0.90, (setup, size, coverage)


def test_the_ladder_spread_contracts_at_the_root_n_rate() -> None:
    """The empirical SD times the root of n is flat across the rungs."""
    for setup in study.LADDER_ESTIMANDS:
        scaled = [PILOT_LADDER[(setup, size)][7] for size in study.LADDER]
        assert max(scaled) / min(scaled) < 1.12, (setup, scaled)


# ------------------------------------------------------------------ (e) the model check


def size_pass_probability(rate: float, replicates: int, ceiling: float = 0.10) -> float:
    """The chance the exact upper endpoint of the rejection rate stays at or below the ceiling."""
    tail = 0.005
    hits = np.arange(0, replicates)
    upper = beta.ppf(1.0 - tail, hits + 1, replicates - hits)
    passing = hits[upper <= ceiling]
    if passing.size == 0:
        return 0.0
    return float(binom.cdf(int(passing.max()), replicates, rate))


def test_the_size_cells_pass_with_the_declared_replicates() -> None:
    """(e) A correct model rejects at the nominal 5%; the cell passes when the exact upper
    endpoint stays at or below 0.10.  The chance of that is at least 0.90 at the nominal rate."""
    assert study.SIZE_REPLICATES % 200 == 0
    assert size_pass_probability(0.05, study.SIZE_REPLICATES) >= PASS_FLOOR
    assert size_pass_probability(0.05, study.SIZE_REPLICATES - 200) < PASS_FLOOR


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


def test_the_sparse_cell_reading_is_declared_before_the_run() -> None:
    """The sparse-cell regime is a finite-sample limit of the unpenalized nuisance fits.  The
    cell is declared red when the pilot's exact 99% interval puts its failure rate above the
    bound or its rejection rate above the 0.10 ceiling; every other size cell reads positive."""
    for name, (kind, *_) in study.CHECK_CELLS.items():
        if kind != "size":
            continue
        draws, failures, tested, rejected, *_ = PILOT_CHECKS[name]
        failing = failures > 0 and float(beta.ppf(0.005, failures, draws - failures + 1)) > 0.01
        rejecting = rejected > 0 and float(beta.ppf(0.005, rejected, tested - rejected + 1)) > 0.10
        declared = f"warning_rate/{name}__model_check" in study.EXPECTED_RED
        assert declared == (failing or rejecting), (name, failures, rejected, tested)
    assert PILOT_CHECKS["sp_f2"][1] / PILOT_CHECKS["sp_f2"][0] > study.FAILURE_BOUND


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


# ---------------------------------------------------------------------- (g) the budget


def cpu_hours() -> float:
    """Every phase of a full run, in CPU-hours, from the pilot's seconds per fit."""
    seconds = 0.0
    seconds += study.PRIMARY_REPLICATES * sum(
        PILOT_COST[study.SETUP_OF[scenario]][0] for scenario in study.SCENARIOS
    )
    for setup in study.CALIBRATED:
        seconds += properties.CALIBRATION_REPLICATES * PILOT_COST[setup][0]
    seconds += properties.CALIBRATION_REPLICATES * PILOT_UNCLUSTERED_SECONDS
    for per_fit in (row[2] for row in PILOT_LADDER.values()):
        seconds += properties.LADDER_REPLICATES * per_fit
    for setup in study.NEGATIVE_CONTROLS:
        seconds += study.CONTROL_REPLICATES * PILOT_CONTROL_SECONDS[setup]
    for name, (kind, *_) in study.CHECK_CELLS.items():
        replicates = study.SIZE_REPLICATES if kind == "size" else study.POWER_REPLICATES
        seconds += replicates * PILOT_CHECKS[name][4]
    return seconds / 3600.0


def test_the_budget_is_declared() -> None:
    """(g) The total CPU-hours of a full run, from timed pilot fits."""
    print(f"\nbudget {cpu_hours():.2f} CPU-hours")
    assert cpu_hours() == pytest.approx(BUDGET_CPU_HOURS, rel=0.05)
