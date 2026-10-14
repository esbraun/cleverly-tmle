"""The declared design of ``longitudinal-incremental`` and ``-dropout`` (``X19-plan.md`` 3.5).

Every number here was fixed before any registered run.

* Every truth is a finite sum on an exact law.  Law ``C`` is law ``A`` with dropout at random
  given the history, so every plan value of ``C`` with dropout prevented is the value on ``A``.
* The fallback-read bias bound is below 0.1 standard errors in every declared positive cell
  (``X19-plan.md`` 6.2).  The cells whose bound failed at the plan's size were raised: the S1
  primary scenarios to n = 6,000 and the S1 ``single_robustness`` pair to 6,000 and 24,000.
* Both primary studies run at n = 24,000.  The five-fold contrasts carry a second-order bias of
  order 1/n, measured at 0.22 standardized at n = 8,000 on law ``C`` against the 0.25 gate.
* The power cell has exact-law power 0.882 at n = 4,000 (:data:`POWER_SCALE`).
* The band cells' pointwise controls clear the RM36 control-power rule at ``p0 + 0.005``, with
  ``p0`` from the exact efficient influence functions: 0.811 for S1 and 0.695 for S2.
* The mechanism-term control's exact SD ratio is 0.78, below the calibration band.
* The S2 witnesses: the informative weight moves the truth by 3.9 standard errors and the
  first-node propensity by more than 0.05; the independent-unit SE of the clustered cell has
  exact ratio 0.60 and coverage 0.76; the complete-case limit sits 2.5 standard errors from
  the truth.
* The ``npcausal`` runner's plan grid is the Python one.

Probes before declaration.  A 40-replication smoke of the S2 primary scenarios read replicates
0 to 39 of their declared streams.  A 400-replication probe of a one-fold law ``C`` scenario,
since removed, read replicates 0 to 399 of its own stream.  It found a standardized bias of
about 0.24 on both contrasts, and the one-fold fits moved to the reported ``in_sample_fit``
family.  Nothing else changed in response to a probe.

Seeds: the four seeds are new, and no registered study uses any of them.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from tests import discrete_law_longitudinal_incremental as law
from tests.studies import canonical_longitudinal_incremental as s1
from tests.studies import canonical_longitudinal_incremental_dropout as s2
from tests.studies import longitudinal_incremental_common as common
from tests.studies import longitudinal_incremental_dropout_properties as p2
from tests.studies import longitudinal_incremental_properties as p1
from tests.studies.default_band_properties import (
    MINIMUM_CONTROL_POWER,
    control_power,
    design,
)
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("longitudinal_incremental_design")

RUNNER = ROOT / "tests" / "canonical" / "npcausal_longitudinal_incremental" / "run_study.R"


def test_the_declared_numbers() -> None:
    for record in (s1.STUDY, s2.STUDY):
        assert record.publication_policy == "gated"
        assert record.margins == Margins()
    assert (s1.PRIMARY_N, s1.PRIMARY_REPLICATES) == (24_000, 1_000)
    assert (s2.PRIMARY_N, s2.PRIMARY_REPLICATES) == (24_000, 1_000)
    assert (s1.SEED, s1.RESAMPLING_SEED, s2.SEED, s2.RESAMPLING_SEED) == (
        20261241,
        2026124101,
        20261242,
        2026124201,
    )
    assert s1.FOLDS == {"ipsi_folds_2": 2, "ipsi_folds_5": 5}
    assert {scenario: folds for scenario, (_, _, folds) in s2.SCENARIOS.items()} == {
        "dropout_folds_5": 5,
        "survival_folds_5": 5,
    }
    assert s1.PAIRING_GATE == 1e-10
    assert s1.STUDY.reference == "npcausal" and s2.STUDY.reference is None
    assert p1.REQUIREMENT_SIZES == (6_000, 24_000)
    assert p1.EFFICIENCY_RATIO_BAND == p2.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert (p2.N, p2.REPLICATES, p2.CLUSTER_SIZE) == (8_000, 1_000, 10)


def test_the_studies_draw_from_their_own_seeds() -> None:
    mine = {s1.SEED, s1.RESAMPLING_SEED, s2.SEED, s2.RESAMPLING_SEED}
    assert len(mine) == 4
    others = {
        seed
        for record in registered()
        if record.slug not in {s1.STUDY.slug, s2.STUDY.slug}
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


# ------------------------------------------------------------------------------- truths


def test_dropout_at_random_leaves_every_plan_value_of_law_a() -> None:
    """A nonzero witness of the censoring axis: the dropout law has the same targets."""
    for plan in s1.PLANS.values():
        assert law.truth(law.LAW_C, plan) == pytest.approx(law.truth(law.LAW_A, plan), abs=1e-12)
    assert law.LAW_C.probs.shape != law.LAW_A.probs.shape


def test_the_natural_course_is_the_observed_mean() -> None:
    probs = law.LAW_A.probs
    observed = float(np.sum(probs * law.LAW_A.mu))
    assert s1.TRUTH[f"ey_regimen[{s1.REFERENCE}]"] == pytest.approx(observed, abs=1e-12)
    assert abs(s1.TRUTH["ate_regimen[odds x2 vs natural course]"]) > 0.5


def test_the_slope_is_the_projection_of_the_plan_means() -> None:
    log_delta = np.log([law.DELTAS[label] for label in s2.PLANS])
    means = [law.truth(law.LAW_C, plan) for plan in s2.PLANS.values()]
    slope = np.polyfit(log_delta, means, 1)[0]
    assert p2.TRUTH["msm_log_delta"] == pytest.approx(slope, abs=1e-12)


# ------------------------------------------------------------------- the fallback bound

#: Every declared positive cell: ``(name, law, plans, estimands, n, folds, eif_sd options)``.
POSITIVE_CELLS = (
    *(
        ("s1_primary", law.LAW_A, s1.PLANS, s1.ESTIMANDS, s1.PRIMARY_N, folds, {})
        for folds in s1.FOLDS.values()
    ),
    ("mean_x2", law.LAW_A, s1.PLANS, (p1.MEAN_X2,), p1.CALIBRATION_N, 5, {}),
    ("contrast_x2", law.LAW_A, s1.PLANS, (p1.CONTRAST_X2,), p1.CALIBRATION_N, 5, {}),
    ("contrast_x2_in_sample", law.LAW_A, s1.PLANS, (p1.CONTRAST_X2,), p1.CALIBRATION_N, 1, {}),
    ("known_x2", law.LAW_A, s1.PLANS, (p1.MEAN_X2,), p1.CALIBRATION_N, 5, {"known": (1, 2, 3)}),
    ("no_positivity", law.LAW_AX, p1.AX_PLANS, (p1.CONTRAST_X2,), p1.CALIBRATION_N, 5, {}),
    (
        "mixed",
        law.LAW_A,
        p1.MIXED_PLANS,
        ("ey_regimen[per_node]", "ey_regimen[tilt_then_static]"),
        p1.CALIBRATION_N,
        5,
        {},
    ),
    *(
        (f"rate_{n}", law.LAW_A, s1.PLANS, (p1.CONTRAST_X2,), n, 5, {})
        for n in p1.RATE_SIZES
        if n != min(p1.RATE_SIZES)
    ),
    ("sharp_null", law.LAW_A0, s1.PLANS, (p1.CONTRAST_X2,), p1.NULL_N, 5, {}),
    ("power", law.LAW_AW, p1.POWER_PLANS, (p1.POWER_NAME,), p1.NULL_N, 5, {}),
    ("s1_band", law.LAW_A, p1.BAND_PLANS, p1.BAND_NAMES, p1.BAND_N, 5, {}),
    *(
        (f"single_{n}", law.LAW_A, s1.PLANS, (p1.CONTRAST_X2,), n, 5, {})
        for n in p1.REQUIREMENT_SIZES
    ),
    ("dropout", law.LAW_C, s2.PLANS, s2.ESTIMANDS["dropout_folds_5"], s2.PRIMARY_N, 5, {}),
    ("survival", law.LAW_D, s2.PLANS, s2.ESTIMANDS["survival_folds_5"], s2.PRIMARY_N, 5, {}),
    ("competing", law.LAW_DC, s2.PLANS, (p2.COMPETING_X2,), p2.N, 5, {}),
    (
        "weighted",
        law.LAW_C,
        s2.PLANS,
        (p2.CONTRAST_X0_5,),
        p2.N,
        5,
        {"weights": p2.WEIGHTS},
    ),
    ("clustered", law.LAW_CK.marginal, s2.PLANS, (p2.MEAN_X2,), p2.N, 5, {}),
)


@pytest.mark.parametrize(
    ("target", "plans", "wanted", "n", "folds", "extra"),
    [cell[1:] for cell in POSITIVE_CELLS],
    ids=[cell[0] for cell in POSITIVE_CELLS],
)
def test_every_positive_cell_meets_the_fallback_bound(
    target: Any, plans: dict, wanted: tuple, n: int, folds: int, extra: dict
) -> None:
    bound = common.fallback_bound(target, plans, wanted, n=n, n_folds=folds, **extra)
    assert bound < common.FALLBACK_BOUND


@pytest.mark.parametrize(
    ("target", "plans", "wanted"),
    [cell[1:4] for cell in POSITIVE_CELLS],
    ids=[cell[0] for cell in POSITIVE_CELLS],
)
def test_the_lower_bound_never_binds_at_the_truth(target: Any, plans: dict, wanted: tuple) -> None:
    """The smallest true product of the non-tilt factors stays ten times above the bound.

    A tilt factor sits outside the bounded product, so only label nodes and retention count.
    """
    used = {label for name in wanted for label in names_in(name, plans)}
    for label in used:
        product = minimum_retention_product(target, plans[label])
        assert product > 10 * common.G_BOUNDS[0], label


def names_in(estimand: str, plans: dict) -> list[str]:
    return [label for label in plans if label in estimand]


def minimum_retention_product(target: Any, plan: Any) -> float:
    labels = [
        1.0 / common.node_ratio(target, entry, t)
        for t, entry in enumerate(plan, 1)
        if entry[0] != "tilt"
    ]
    return float(np.prod(labels)) * common.minimum_retention(target) ** target.n_times


def test_the_working_model_slope_meets_the_fallback_bound() -> None:
    """The slope is a fixed combination of the plan means, so its bias is too."""
    bias = sum(
        abs(weight) * common.plan_fallback_bias(law.LAW_C, s2.PLANS[label], n=p2.N, n_folds=5)
        for label, weight in p2.SLOPE_WEIGHTS.items()
    )
    assert bias / (p2.EFFICIENCY_SD["msm_log_delta"] / np.sqrt(p2.N)) < common.FALLBACK_BOUND


def test_the_bound_binds_where_the_sizes_were_raised() -> None:
    """The witness: at the plan's sizes the bound fails, so raising them was needed."""
    assert common.fallback_bound(law.LAW_A, s1.PLANS, s1.ESTIMANDS, n=5_000, n_folds=2) > 0.1
    assert common.fallback_bound(law.LAW_A, s1.PLANS, (p1.CONTRAST_X2,), n=2_000, n_folds=5) > 0.1
    survival = s2.ESTIMANDS["survival_folds_5"]
    assert common.fallback_bound(law.LAW_D, s2.PLANS, survival, n=2_000, n_folds=5) > 0.1


# ------------------------------------------------------------------- sized cells


def test_the_power_cell_is_sized_on_the_exact_law() -> None:
    power = design_power(
        p1.POWER_TRUTH, common.eif_sd(law.LAW_AW, p1.POWER_PLANS, p1.POWER_NAME), p1.NULL_N
    )
    assert 0.80 <= power <= 0.90
    assert power == pytest.approx(0.882, abs=0.001)
    assert power_cell_pass_probability(power, p1.NULL_REPLICATES) >= 0.95


def test_the_sharp_null_has_one_mean() -> None:
    values = {law.truth(law.LAW_A0, plan) for plan in s1.PLANS.values()}
    assert max(values) - min(values) < 1e-12
    assert abs(p1.NULL_TRUTH) < 1e-12


def _correlation(target: Any, plans: dict, names: tuple) -> np.ndarray:
    parts = []
    for name in names:
        left, right, cause, horizon = common.parse(name)
        options: dict = {}
        if horizon is not None:
            options.update(horizon=horizon, cause=cause)
        if right is not None:
            options["other"] = plans[right]
        cells, coefficient = law.eif(target, plans[left], **options)
        parts.append((np.nan_to_num(cells), coefficient))
    probs = target.probs
    k = len(parts)
    covariance = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            covariance[i, j] = np.sum(probs * parts[i][0] * parts[j][0])
            covariance[i, j] += law.SIGMA**2 * np.sum(probs * parts[i][1] * parts[j][1])
    sd = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sd, sd)


@pytest.mark.parametrize(
    ("target", "plans", "names", "replicates", "p0"),
    [
        (law.LAW_A, p1.BAND_PLANS, p1.BAND_NAMES, p1.BAND_REPLICATES, 0.811),
        (law.LAW_D, p2.PLANS, p2.BAND_NAMES, p2.REPLICATES, 0.695),
    ],
    ids=["s1_nine_tilts", "s2_survival_grid"],
)
def test_the_band_controls_clear_the_control_power_rule(
    target: Any, plans: dict, names: tuple, replicates: int, p0: float
) -> None:
    designed = design(_correlation(target, plans, names), replicates)
    assert designed.p0 == pytest.approx(p0, abs=0.002)
    assert control_power(designed.p0 + 0.005, replicates) >= MINIMUM_CONTROL_POWER


def test_the_mechanism_term_control_falls_below_the_calibration_band() -> None:
    full = common.eif_sd(law.LAW_A, s1.PLANS, p1.CONTRAST_X2)
    reduced = common.eif_sd(law.LAW_A, s1.PLANS, p1.CONTRAST_X2, known=(1, 2, 3))
    assert reduced / full == pytest.approx(0.78, abs=0.005)
    assert reduced / full < Margins().calibration_se_ratio[0] - 0.1


# ------------------------------------------------------------------- the S2 witnesses


def test_the_weight_is_informative() -> None:
    weighted = p2.TRUTH["weighted_contrast_x0_5"]
    plain = common.truth(law.LAW_C, s2.PLANS, p2.CONTRAST_X0_5)
    error = p2.EFFICIENCY_SD["weighted_contrast_x0_5"] / np.sqrt(p2.N)
    assert abs(weighted - plain) / error == pytest.approx(3.9, abs=0.05)
    probs = law.LAW_C.probs
    first = law.LAW_C.axis_of("treatment", 1)
    axes = tuple(range(first + 1, probs.ndim))
    plain_arm = probs.sum(axis=axes)
    weighted_arm = (p2.WEIGHTS * probs).sum(axis=axes)
    for w in (0, 1):
        pi = plain_arm[w, 1] / plain_arm[w].sum()
        tilted = weighted_arm[w, 1] / weighted_arm[w].sum()
        assert abs(tilted - pi) > 0.05


def test_the_independent_unit_error_under_covers_the_clustered_law() -> None:
    marginal = law.LAW_CK.marginal
    cells, coefficient = law.eif(marginal, s2.PLANS[p2.X2])
    cells = np.nan_to_num(cells)
    keep = marginal.probs > 0
    variance = float(np.sum(marginal.probs[keep] * cells[keep] ** 2))
    variance += law.SIGMA**2 * float(np.sum(marginal.probs[keep] * coefficient[keep] ** 2))
    conditional = [
        float(np.sum(half.probs * (cells + coefficient * (half.mu - marginal.mu))))
        for half in law.LAW_CK.halves
    ]
    within = ((conditional[0] - conditional[1]) / 2.0) ** 2
    ratio = np.sqrt(variance / (variance + (p2.CLUSTER_SIZE - 1) * within))
    coverage = 2.0 * norm.cdf(norm.ppf(0.975) * ratio) - 1.0
    assert ratio == pytest.approx(0.60, abs=0.005)
    assert coverage < 0.90
    assert ratio < 0.80 - 0.1


def test_the_complete_case_limit_misses_the_truth() -> None:
    error = common.eif_sd(law.LAW_C, s2.PLANS, p2.MEAN_X2) / np.sqrt(p2.N)
    gap = (p2.COMPLETE_CASE_LIMIT - p2.COMPLETE_CASE_TRUTH) / error
    assert gap == pytest.approx(2.5, abs=0.05)


def test_the_survival_and_competing_targets_are_risks() -> None:
    for name, value in p2.BAND_TRUTH.items():
        if name.startswith("risk_regimen["):
            assert 0.0 < value < 1.0
    risks = [p2.BAND_TRUTH[f"risk_regimen[{s2.REFERENCE} @ t={h}]"] for h in s2.HORIZONS]
    assert risks == sorted(risks)


# ------------------------------------------------------------------- declared cells


@pytest.mark.parametrize(("study", "properties"), [(s1, p1), (s2, p2)], ids=["s1", "s2"])
def test_the_declared_cells_are_the_cells_a_run_publishes(study: Any, properties: Any) -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=2)
        summary = properties.summarize_properties(rows)
    derived = {
        (family, cell)
        for family, cell in zip(rows["property"], rows["cell"], strict=True)
        if cell.endswith(("__shrunken_se_control", "__noise_control"))
    }
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared) | derived
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        if family != "simultaneous_coverage":
            np.testing.assert_allclose(
                selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0
            )
        assert set(selected["n"]) == {cell.n}
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


def test_the_runner_reads_the_python_plan_grid() -> None:
    source = RUNNER.read_text()
    grid = re.search(r"DELTAS <- c\((?P<body>[^)]*)\)", source)
    assert grid is not None
    pairs = re.findall(r'"([^"]+)" = ([0-9.]+)', grid["body"])
    assert {label: float(value) for label, value in pairs} == law.DELTAS
    scenarios = re.search(r"SCENARIOS <- c\((?P<body>[^)]*)\)", source)
    assert scenarios is not None
    assert re.findall(r'"([^"]+)"', scenarios["body"]) == list(s1.FOLDS)
    assert "N_TIMES <- 3L" in source
