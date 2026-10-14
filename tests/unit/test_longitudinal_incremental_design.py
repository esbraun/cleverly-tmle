"""The declared design of ``longitudinal-incremental`` and ``-dropout`` (``X19-plan.md`` 3.5).

Every number here was fixed before any registered run.

* Every truth is a finite sum on an exact law.  Law ``C`` is law ``A`` with dropout at random
  given the history, so every plan value of ``C`` with dropout prevented is the value on ``A``.
* The fallback-read bias bound is below 0.1 standard errors in every declared positive cell
  (``X19-plan.md`` 6.2).
* The sizing rule.  The five-fold contrasts carry a second-order bias of order 1/n, and the
  SE ratio sits a little below one at moderate n.  The sizing probe
  (``tests/diagnostics/x19_sizing/``) measured both for every gated cell at n = 8,000.
  Projected to the declared size (bias as :math:`1/\\sqrt n` after adding one Monte Carlo
  SE, SE-ratio deficit as 1/n), every primary row and positive root-n rung passes its bias rule
  with probability at least 0.95, and every calibration cell passes with probability at least
  0.90.  A study's calibration cells also pass jointly with probability at least 0.90, which
  needs 3,000 replications: at 2,000 an ideal cell passes with 0.935 and S1's seven with
  0.59.  The rule sets n = 96,000 (S1, which the two-fold rows need), 80,000 (S2) and 48,000
  (S2 calibration, which the joint rule needs).  S1 calibration runs at 48,000; the rule allows 24,000 at 3,000
  replications, and the larger size cuts the projected bias by a factor of 1.4.  The null,
  power, band, robustness and clustered cells, and the competing-risk cell, are read at their
  own sizes rather than projected.
* The power cell has exact-law power 0.882 at n = 4,000 (:data:`POWER_SCALE`).
* The band cells' pointwise controls clear the RM36 control-power rule at ``p0 + 0.005``, with
  ``p0`` from the exact efficient influence functions: 0.811 for S1 and 0.695 for S2.
* The mechanism-term control's exact SD ratio is 0.78, below the calibration band, and the
  probe reads the published control below it too.
* The robustness fits use a linear outcome regression.  Its exact plug-in limit misses the
  truth, so the targeting control has a bias to show and the correction has work to do.
* The S2 witnesses: the informative weight moves the truth by 9.5 standard errors and the
  first-node propensity by more than 0.05; the independent-unit SE of the clustered cell has
  exact ratio 0.60 and coverage 0.76; the complete-case limit sits 6.2 standard errors from
  the truth.
* The ``npcausal`` runner's plan grid is the Python one.

Probes before declaration.  A 40-replication smoke of the S2 primary scenarios read replicates
0 to 39 of their declared streams.  A 400-replication probe of a one-fold law ``C`` scenario
read replicates 0 to 399 of its own stream.  Every other probe reads seeds outside the
declared streams (``tests/diagnostics/x19_sizing/``).  The probes changed sizes and
replication counts only.

Seeds: the four seeds are new, and no registered study uses any of them.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm
from scipy.stats import t as student
from sklearn.dummy import DummyClassifier

from tests import discrete_law_longitudinal_incremental as law
from tests.studies import canonical_longitudinal_incremental as s1
from tests.studies import canonical_longitudinal_incremental_dropout as s2
from tests.studies import longitudinal_incremental_common as common
from tests.studies import longitudinal_incremental_dropout_properties as p2
from tests.studies import longitudinal_incremental_properties as p1
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.default_band_properties import (
    MINIMUM_CONTROL_POWER,
    control_power,
    design,
)
from tests.studies.evidence import clopper_pearson
from tests.studies.evidence.property_verdicts import (
    CLUSTER_ROBUST_CONTROL_SE_CEILING,
    CLUSTERED_COVERAGE_GAIN,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("longitudinal_incremental_design")

RUNNER = ROOT / "tests" / "canonical" / "npcausal_longitudinal_incremental" / "run_study.R"


def test_the_declared_numbers() -> None:
    for record in (s1.STUDY, s2.STUDY):
        assert record.publication_policy == "gated"
        assert record.margins == Margins()
    assert (s1.PRIMARY_N, s1.PRIMARY_REPLICATES) == (96_000, 1_000)
    assert (s2.PRIMARY_N, s2.PRIMARY_REPLICATES) == (80_000, 1_000)
    assert (s1.SEED, s1.RESAMPLING_SEED, s2.SEED, s2.RESAMPLING_SEED) == (
        20261241,
        2026124101,
        20261242,
        2026124201,
    )
    assert s1.FOLDS == {"ipsi_folds_2": 2, "ipsi_folds_5": 5}
    assert {scenario: folds for scenario, (_, _, folds) in s2.SCENARIOS.items()} == {
        "dropout_folds_1": 1,
        "dropout_folds_5": 5,
        "survival_folds_1": 1,
        "survival_folds_5": 5,
    }
    assert s1.PAIRING_GATE == 1e-10
    assert s1.STUDY.reference == "npcausal" and s2.STUDY.reference is None
    assert p1.REQUIREMENT_SIZES == (48_000, 96_000)
    assert p1.EFFICIENCY_RATIO_BAND == p2.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert (p1.CALIBRATION_N, p1.CALIBRATION_REPLICATES) == (48_000, 3_000)
    assert (p1.RATE_SIZES, p1.RATE_REPLICATES) == ((24_000, 64_000, 96_000), 1_000)
    assert (p2.N, p2.REPLICATES, p2.CLUSTER_SIZE) == (48_000, 3_000, 10)
    assert (p2.BAND_N, p1.NULL_N, p1.NULL_REPLICATES) == (48_000, 4_000, 2_000)
    assert (p2.CLUSTER_N, p2.CLUSTER_REPLICATES, p2.COMPETING_N) == (8_000, 3_000, 96_000)


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


def test_the_fitted_natural_course_is_the_sample_mean() -> None:
    """A tilt of 1 sets every weight to 1, so the one-step value is the sample mean exactly."""
    frame = law.sample(law.LAW_A, 2_000, 11)
    result = common.fit(frame, law.LAW_A, s1.PLANS, n_folds=5, reference=s1.REFERENCE)
    natural = result[f"ey_regimen[{s1.REFERENCE}]"]
    assert float(natural.psi) == pytest.approx(float(frame["Y"].mean()), abs=1e-10)
    curve = np.asarray(result.influence_curves[f"ey_regimen[{s1.REFERENCE}]"], dtype=float)
    assert np.max(np.abs(curve - (frame["Y"] - frame["Y"].mean()))) < 1e-10


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
    ("competing", law.LAW_DC, s2.PLANS, (p2.COMPETING_X2,), p2.COMPETING_N, 5, {}),
    (
        "weighted",
        law.LAW_C,
        s2.PLANS,
        (p2.CONTRAST_X0_5,),
        p2.N,
        5,
        {"weights": p2.WEIGHTS},
    ),
    ("clustered", law.LAW_CK.marginal, s2.PLANS, (p2.MEAN_X2,), p2.CLUSTER_N, 5, {}),
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


# ------------------------------------------------------------------- the sizing rule

#: The sizing probe's reading: each gated cell at n = 8,000 on fresh draws.
READING = pd.read_csv(ROOT / "tests" / "diagnostics" / "x19_sizing" / "reading.csv")
PRIMARY_PASS = 0.95
CALIBRATION_PASS = 0.90
X2 = "ate_regimen[odds x2 vs natural course]"


def probe(cell_set: str, cell: str) -> Any:
    (row,) = READING[(READING["set"] == cell_set) & (READING["cell"] == cell)].itertuples()
    return row


def projected(row: Any, n: int) -> tuple[float, float]:
    """The standardized bias and the SE ratio at ``n``, from the probe at ``row.n``.

    The bias is second order, so its standardized value falls as :math:`1/\\sqrt n`; one Monte
    Carlo SE is added first.  The SE-ratio deficit, the cost of estimating the nuisances, falls
    as :math:`1/n`.  A ratio above one is not projected.
    """
    bias = (abs(row.standardized_bias) + row.mcse) * np.sqrt(row.n / n)
    ratio = 1.0 - (1.0 - min(row.se_ratio, 1.0)) * row.n / n
    return float(bias), float(ratio)


def bias_pass_probability(bias: float, replicates: int) -> float:
    """The chance the 99% Student interval of the mean error lies inside the bias margin."""
    margins = Margins()
    half = float(student.ppf(0.5 + margins.confidence_level / 2.0, replicates - 1))
    spread = 1.0 / np.sqrt(replicates)
    room = margins.standardized_bias - half * spread
    return float(norm.cdf((room - bias) / spread) - norm.cdf((-room - bias) / spread))


def calibrated_pass_probability(row: Any, n: int, replicates: int) -> float:
    bias, ratio = projected(row, n)
    critical = float(norm.ppf(0.975)) * ratio
    coverage = float(norm.cdf(critical - bias) - norm.cdf(-critical - bias))
    margins = Margins()
    return calibration_pass_probability(
        replicates,
        true_se_ratio=ratio,
        coverage=coverage,
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )


PRIMARY_ROWS = (
    *(
        (f"s1:{scenario}", name, s1.PRIMARY_N, s1.PRIMARY_REPLICATES)
        for scenario in s1.FOLDS
        for name in s1.ESTIMANDS
    ),
    *(
        (f"s2:{scenario}", name, s2.PRIMARY_N, s2.PRIMARY_REPLICATES)
        for scenario, names in s2.ESTIMANDS.items()
        for name in names
    ),
)


@pytest.mark.parametrize(("cell_set", "cell", "n", "replicates"), PRIMARY_ROWS)
def test_every_primary_row_is_sized_for_its_bias(
    cell_set: str, cell: str, n: int, replicates: int
) -> None:
    bias, _ = projected(probe(cell_set, cell), n)
    assert bias_pass_probability(bias, replicates) >= PRIMARY_PASS


def test_the_sizing_rule_binds() -> None:
    """The witness: at the earlier sizes, or at two folds, a primary row is likely red."""
    worst = probe("s1:ipsi_folds_5", "ate_regimen[odds x0.5 vs natural course]")
    assert bias_pass_probability(projected(worst, 24_000)[0], 1_000) < 0.5
    two_folds = probe("s1:ipsi_folds_2", "ate_regimen[odds x0.5 vs natural course]")
    assert bias_pass_probability(projected(two_folds, 64_000)[0], 1_000) < PRIMARY_PASS
    contrast = probe("s1:law_a", "interval_calibration/contrast_x2__correctly_specified")
    assert calibrated_pass_probability(contrast, 4_000, p1.CALIBRATION_REPLICATES) < 0.5
    one_fold = probe("s1:law_a", "interval_calibration/contrast_x2_in_sample__correctly_specified")
    assert calibrated_pass_probability(one_fold, 8_000, p1.CALIBRATION_REPLICATES) < 0.5
    # At 2,000 replications the seven S1 cells pass jointly with probability below 0.6.
    assert (
        np.prod(
            [
                calibrated_pass_probability(probe(*cell[:2]), cell[2], 2_000)
                for cell in CALIBRATION_CELLS
                if cell[0].startswith("s1:")
            ]
        )
        < 0.6
    )
    weighted = "interval_calibration/weighted_contrast_x0_5__correctly_specified"
    assert calibrated_pass_probability(probe("s2:law_c@16000", weighted), 16_000, 3_000) < 0.85
    competing = "interval_calibration/competing_contrast_x2_h3__correctly_specified"
    replicates = p2.REPLICATES
    assert (
        calibrated_pass_probability(probe("s2:law_dc@32000", competing), 32_000, replicates) < 0.5
    )
    assert (
        calibrated_pass_probability(probe("s2:law_dc@64000", competing), 64_000, replicates) < 0.93
    )
    cluster = probe("s2:law_ck", "clustered_inference/mean_x2__cluster_robust")
    assert calibrated_pass_probability(cluster, p2.CLUSTER_N, 1_000) < 0.5


@pytest.mark.parametrize("size", [size for size in p1.RATE_SIZES if size != min(p1.RATE_SIZES)])
def test_every_positive_rung_is_sized_for_its_bias(size: int) -> None:
    bias, _ = projected(probe("s1:ipsi_folds_5", X2), size)
    assert bias_pass_probability(bias, p1.RATE_REPLICATES) >= PRIMARY_PASS


CALIBRATION_CELLS = (
    *(
        (
            f"s1:{p1.CALIBRATION_STREAM[label]}",
            f"interval_calibration/{label}__correctly_specified",
            p1.CALIBRATION_N,
            p1.CALIBRATION_REPLICATES,
        )
        for label in p1.CALIBRATION_LABELS
    ),
    *(
        (
            # Read at 16,000: at 8,000 the SE ratios exceed 1, which the projection clamps.
            "s2:law_c@16000",
            f"interval_calibration/{label}__correctly_specified",
            p2.N,
            p2.REPLICATES,
        )
        for label in (
            "dropout_contrast_x2",
            "dropout_contrast_x2_in_sample",
            "msm_log_delta",
            "weighted_contrast_x0_5",
        )
    ),
    # The band draws carry both survival labels, read at their declared size.
    *(
        (
            "s2:band",
            f"interval_calibration/{label}__correctly_specified",
            p2.BAND_N,
            p2.REPLICATES,
        )
        for label in ("survival_contrast_x2_h3", "survival_contrast_x2_h3_in_sample")
    ),
    # The competing-risk tail does not shrink as 1/n, so the probe reads the cell above the
    # size where the tail is gone (32,000) and projects from there.
    (
        "s2:law_dc@64000",
        "interval_calibration/competing_contrast_x2_h3__correctly_specified",
        p2.COMPETING_N,
        p2.REPLICATES,
    ),
    (
        "s2:law_ck",
        "clustered_inference/mean_x2__cluster_robust",
        p2.CLUSTER_N,
        p2.CLUSTER_REPLICATES,
    ),
)


@pytest.mark.parametrize(("cell_set", "cell", "n", "replicates"), CALIBRATION_CELLS)
def test_every_calibration_cell_is_sized_for_its_pass_probability(
    cell_set: str, cell: str, n: int, replicates: int
) -> None:
    probability = calibrated_pass_probability(probe(cell_set, cell), n, replicates)
    assert probability >= CALIBRATION_PASS, probability


@pytest.mark.parametrize("study", ["s1", "s2"])
def test_the_calibration_cells_pass_jointly(study: str) -> None:
    """A study's calibration cells all pass with the per-cell probability floor."""
    probabilities = [
        calibrated_pass_probability(probe(cell_set, cell), n, replicates)
        for cell_set, cell, n, replicates in CALIBRATION_CELLS
        if cell_set.startswith(f"{study}:")
    ]
    assert float(np.prod(probabilities)) >= CALIBRATION_PASS


def interval_pass_probability(rate: float, replicates: int, accept: Any) -> float:
    """The chance that the exact 99% interval of a binomial rate satisfies ``accept``."""
    successes = np.arange(replicates + 1)
    level = Margins().confidence_level
    passing = [
        k for k in successes if accept(clopper_pearson(int(k), replicates, confidence_level=level))
    ]
    return float(binom.pmf(passing, replicates, rate).sum())


def test_the_rate_cells_are_sized_at_their_declared_sizes() -> None:
    """The null, power, band and clustered-control cells, read at their own sizes.

    Each rate moves one Monte Carlo SE toward failure before its pass probability is taken.
    """
    margins = Margins()

    null = probe("s1:sharp_null", "type_i_error/contrast_x2__sharp_null")
    rejection = null.rejected + np.sqrt(null.rejected * (1 - null.rejected) / null.replicates)
    ceiling = margins.alpha + margins.type_i_margin
    assert (
        interval_pass_probability(rejection, p1.NULL_REPLICATES, lambda i: i.high <= ceiling)
        >= PRIMARY_PASS
    )
    power = probe("s1:alternative", "power/contrast_x1_5__alternative")
    rate = power.rejected - np.sqrt(power.rejected * (1 - power.rejected) / power.replicates)
    assert power_cell_pass_probability(rate, p1.NULL_REPLICATES) >= PRIMARY_PASS
    for cell_set, label, replicates in (
        ("s1:band", p1.BAND_LABEL, p1.BAND_REPLICATES),
        # Read at n = 16,000; the declared band runs at 48,000, where it is no less calibrated.
        ("s2:band", p2.BAND_LABEL, p2.REPLICATES),
    ):
        band = probe(cell_set, f"simultaneous_coverage/{label}__simultaneous_band")
        spread = np.sqrt(band.covered * (1 - band.covered) / band.replicates)
        for coverage in (band.covered - spread, band.covered + spread):
            assert (
                interval_pass_probability(
                    coverage, replicates, lambda i: i.within(*margins.calibration_coverage)
                )
                >= CALIBRATION_PASS
            )
        control = probe(cell_set, f"simultaneous_coverage/{label}__pointwise_joint_control")
        rate = control.covered + np.sqrt(control.covered * (1 - control.covered) / replicates)
        assert (
            interval_pass_probability(rate, replicates, lambda i: i.high < 1.0 - margins.alpha)
            >= PRIMARY_PASS
        )
    control = probe("s2:law_ck", "clustered_inference/mean_x2__iid_control")
    assert control.se_ratio < CLUSTER_ROBUST_CONTROL_SE_CEILING - 0.1
    robust = probe("s2:law_ck", "clustered_inference/mean_x2__cluster_robust")
    assert robust.covered - control.covered > CLUSTERED_COVERAGE_GAIN + 0.1


#: The sizes the ``single_robustness`` probe read (``tests/diagnostics/x19_sizing``).
REQUIREMENT_PROBE_SIZES = (6_000, 24_000)


def _requirement_probe(size: int, cell: str) -> Any:
    return probe(f"s1:outcome_linear__n_{size}", cell)


def test_the_single_robustness_pair_is_sized_at_its_declared_sizes() -> None:
    """Both sizes pass bias equivalence together, at the larger of the two projections.

    The remainder is second order, so the bias is projected as the other gated cells are.  The
    probe reads 0.20 at n = 6,000 and 0.11 at 24,000, so 24,000 would fail.
    """
    probability = 1.0
    for size in p1.REQUIREMENT_SIZES:
        bias = max(
            projected(
                _requirement_probe(
                    probed, f"single_robustness/contrast_x2__mechanism_correct__n_{probed}"
                ),
                size,
            )[0]
            for probed in REQUIREMENT_PROBE_SIZES
        )
        probability *= bias_pass_probability(bias, p1.REQUIREMENT_REPLICATES)
    assert probability >= PRIMARY_PASS
    probed = _requirement_probe(24_000, "single_robustness/contrast_x2__mechanism_correct__n_24000")
    assert bias_pass_probability(projected(probed, 24_000)[0], 1_000) < PRIMARY_PASS


def linear_plug_in_limit(target: Any, delta: float) -> float:
    """The population limit of the plug-in with linear regressions, by exact projection.

    The recursion of :func:`tests.studies.longitudinal_incremental_common.plug_in` on the law
    itself: at each node a probability-weighted least-squares fit on the history and the arm,
    predicted at both arms and mixed by the tilted propensity.  Linear regression is
    affine-equivariant, so the outcome scale does not change the limit.
    """
    names = [axis.name for axis in target.axes]
    assert names == ["W", "A1", "L2", "A2", "L3", "A3"]
    probs = np.asarray(target.probs, dtype=float)
    cells = np.array(np.unravel_index(np.arange(probs.size), probs.shape), dtype=float).T
    weight = probs.reshape(-1)
    value = np.asarray(target.mu, dtype=float).reshape(-1)
    for node in (3, 2, 1):
        arm = 2 * node - 1
        design = np.column_stack([np.ones(len(cells)), cells[:, : arm + 1]])
        root = np.sqrt(weight)[:, None]
        beta = np.linalg.lstsq(design * root, value * root[:, 0], rcond=None)[0]
        predictions = []
        for level in (0.0, 1.0):
            counterfactual = design.copy()
            counterfactual[:, -1] = level
            predictions.append(counterfactual @ beta)
        upper = probs.sum(axis=tuple(range(arm + 1, probs.ndim)))
        propensity = np.take(upper, 1, axis=arm) / upper.sum(axis=arm)
        pi = propensity[tuple(cells[:, i].astype(int) for i in range(arm))]
        tilted = delta * pi / (delta * pi + 1.0 - pi)
        value = tilted * predictions[1] + (1.0 - tilted) * predictions[0]
    return float(np.sum(weight * value))


def test_the_targeting_control_misses_the_truth_at_its_size() -> None:
    """The witness: a linear outcome regression leaves a plug-in that misses the truth.

    A constant regression makes every plan's regression the same mean, so the plug-in
    contrast is exactly zero.  The linear one is misspecified: its plug-in limit sits 0.0012
    from the truth, and the one-step correction must remove that.  The control's
    standardized bias at n = 96,000 is read from that limit alone; the probe's spread at
    24,000 sets the scale, shrunk as :math:`1/\\sqrt n`.
    """
    limit = linear_plug_in_limit(law.LAW_A, 2.0) - linear_plug_in_limit(law.LAW_A, 1.0)
    bias = limit - p1.CONTRAST_TRUTH
    assert bias == pytest.approx(-0.0012058, abs=2e-7)
    # The probe's plug-in error at 24,000 includes this limit and a finite-sample part.
    plug_in = _requirement_probe(24_000, "targeting_necessity/contrast_x2__plug_in")
    assert plug_in.bias < bias < 0.0
    size = p1.REQUIREMENT_SIZES[1]
    spread = plug_in.bias_se * np.sqrt(plug_in.replicates) * np.sqrt(24_000 / size)
    standardized = abs(bias) / spread
    assert standardized - 1.0 / np.sqrt(p1.REQUIREMENT_REPLICATES) >= p1.TARGETING_DISPLACEMENT
    # The control's 99% interval must lie wholly outside the equivalence margin.
    margins = Margins()
    replicates = p1.REQUIREMENT_REPLICATES
    half = float(student.ppf(0.5 + margins.confidence_level / 2.0, replicates - 1))
    deviation = 1.0 / np.sqrt(replicates)
    room = margins.standardized_bias + half * deviation
    assert float(norm.cdf((standardized - room) / deviation)) >= PRIMARY_PASS


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
    # The statistic the study publishes, on fresh draws: the control reads the reduced curve.
    control = probe("s1:law_a", "mechanism_term_necessity/contrast_x2__mechanism_term_dropped")
    assert control.se_ratio < Margins().calibration_se_ratio[0] - 0.1


# ------------------------------------------------------------------- the S2 witnesses


@pytest.mark.parametrize(
    ("target", "name", "extra"),
    [
        (law.LAW_C, p2.CONTRAST_X2, {}),
        (law.LAW_D, p2.SURVIVAL_X2, {"horizons": [3]}),
    ],
)
def test_the_calibration_cells_see_the_censoring_factor(
    target: Any, name: str, extra: dict
) -> None:
    """A deliberate mutation: no censoring model leaves the estimate and shrinks the SE.

    The censoring factor of the curve barely moves the estimate, so the bias rows cannot see
    it.  Without it the reported SE falls to about 0.86 of the correct one, below the
    calibration band, so the calibration cells fail when the factor is lost.
    """
    frame = law.sample(target, 8_000, 5)
    settings = {"n_folds": 5, "reference": s2.REFERENCE, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correct = common.fit(frame, target, s2.PLANS, **settings)[name]
        dropped = common.fit(
            frame,
            target,
            s2.PLANS,
            censoring_learner=DummyClassifier(strategy="constant", constant=1),
            **settings,
        )[name]
    assert abs(float(dropped.psi) - float(correct.psi)) < 0.1 * float(correct.std_error)
    ratio = float(dropped.std_error) / float(correct.std_error)
    assert ratio < Margins().calibration_se_ratio[0] - 0.03


def test_the_weight_is_informative() -> None:
    weighted = p2.TRUTH["weighted_contrast_x0_5"]
    plain = common.truth(law.LAW_C, s2.PLANS, p2.CONTRAST_X0_5)
    error = p2.EFFICIENCY_SD["weighted_contrast_x0_5"] / np.sqrt(p2.N)
    assert abs(weighted - plain) / error == pytest.approx(9.46, abs=0.05)
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
    assert gap == pytest.approx(6.20, abs=0.05)


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
