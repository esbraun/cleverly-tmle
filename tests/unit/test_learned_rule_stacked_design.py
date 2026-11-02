"""The declared constants of the ``learned-rule-stacked`` study, recomputed without a study run.

Each number here was fixed before any registered run. The law checks call no sampler: they read
the population projection of the declared learner and integrate on the RM30 grid. The ratio-bias
oracle draws cluster sizes and effects only, by the method of
``tests/unit/test_fold_evaluated_ratio_bias.py``, and fits nothing. The declared seeds are only
computed, never drawn from.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_stacked_law as law
from tests.studies import learned_rule_compositions as compositions
from tests.studies import learned_rule_stacked as study
from tests.studies import learned_rule_stacked_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_stacked_design")


@cache
def _moments() -> law.ClusterMoments:
    return law.cluster_moments(law.population_rule("clustered"))


# ----------------------------------------------------------------------------- the law


@pytest.mark.parametrize("scenario", law.SCENARIOS)
def test_the_efficiency_bounds(scenario: str) -> None:
    assert law.efficiency_bound(scenario) == pytest.approx(
        study.EFFICIENCY_BOUNDS[scenario], abs=2e-6
    )


def test_rows_and_weighted_are_the_compositions_laws() -> None:
    """The same laws, learner and limit rules as ``learned-rule-compositions``, so the same
    bounds; only the report differs."""
    assert study.EFFICIENCY_BOUNDS["rows"] == compositions.EFFICIENCY_BOUNDS["fold_targeted"]
    assert study.EFFICIENCY_BOUNDS["weighted"] == compositions.EFFICIENCY_BOUNDS["weighted"]


def test_the_size_dependence_and_the_icc_on_the_curve_scale() -> None:
    """Declared at the limit rule: the correlation of the cluster size with its cluster's mean
    curve is 0.891, the curve's intra-cluster correlation under the row law 0.077, and the row
    rule's SE over the declared one 0.771, which the row-variance control must read."""
    moments = _moments()
    assert moments.size_curve_correlation == pytest.approx(0.891, abs=5e-4)
    assert moments.curve_icc == pytest.approx(0.0772, abs=5e-4)
    ratio = np.sqrt(moments.row_variance / moments.cluster_variance)
    assert ratio == pytest.approx(0.771, abs=5e-4)
    assert (law.SIGMA_U, law.KAPPA) == (1.0, 1.5)


def test_without_a_size_slope_the_size_carries_no_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """The witness: the correlation comes from the size law, and vanishes without its slope."""
    monkeypatch.setattr(law, "KAPPA", 0.0)
    moments = law.cluster_moments(law.population_rule("clustered"))
    assert moments.size_curve_correlation == pytest.approx(0.0, abs=1e-12)
    # The curve still varies by cluster, so the ICC is not what carries the correlation.
    assert moments.curve_icc > 0.05


# ------------------------------------------------------------------------- ratio bias

#: The ratio bias of each report at the limit rule, rebuilt before the run from the law alone
#: (200,000 draws of 200 clusters, label 20267502), in stacked SDs at nominal n = 2,000:
#: fold-evaluated -0.1198, stacked -0.0090, each with Monte Carlo SE 0.0015.
DECLARED_RATIO_BIAS = {"fold_evaluated": -0.1198, "stacked": -0.0090}


def test_the_fold_evaluated_ratio_bias_is_rebuilt_from_the_law() -> None:
    """At J = 200 and V = 10 the fold-evaluated point carries about V times the stacked ratio
    bias. The check reruns the oracle at 50,000 draws and holds each to 4 combined MC SEs."""
    oracle = law.ratio_bias_oracle(50_000, 20267502)
    for report, declared in DECLARED_RATIO_BIAS.items():
        allowed = 4.0 * float(np.hypot(oracle[f"{report}_se"], 0.0015))
        assert abs(oracle[report] - declared) <= allowed, (report, oracle)
    assert oracle["fold_evaluated"] < -10 * oracle["fold_evaluated_se"]
    assert 4 < oracle["fold_evaluated"] / oracle["stacked"] < 25


#: The same oracle at 40 clusters, the smallest count on the normal reference that shares the
#: ``dependence="clustered"`` key (200,000 draws, label 20267504): fold-evaluated -0.2855,
#: stacked -0.0276, each with MC SE 0.0015. Declared before the run, so that any extension of
#: the stacked row below 200 clusters rests on a declared bias (design review finding 2).
DECLARED_RATIO_BIAS_J40 = {"fold_evaluated": -0.2855, "stacked": -0.0276}


def test_the_ratio_bias_at_forty_clusters_is_declared() -> None:
    """At J = 40 the fold-evaluated bias grows about as ``V / sqrt(J)`` in SDs, against
    ``1 / sqrt(J)`` for the stacked point."""
    oracle = law.ratio_bias_oracle(50_000, 20267505, clusters=40)
    for report, declared in DECLARED_RATIO_BIAS_J40.items():
        allowed = 4.0 * float(np.hypot(oracle[f"{report}_se"], 0.0015))
        assert abs(oracle[report] - declared) <= allowed, (report, oracle)
    ratio = DECLARED_RATIO_BIAS_J40["fold_evaluated"] / DECLARED_RATIO_BIAS["fold_evaluated"]
    assert 1.8 < ratio < 2.8  # sqrt(200 / 40) = 2.24


def test_equal_sizes_carry_no_fold_average_bias() -> None:
    """The deliberate-mutation control of the oracle: equal sizes remove the denominator that
    varies, and with it the fold-evaluated bias."""
    oracle = law.ratio_bias_oracle(20_000, 20267503, equal_sizes=True)
    assert abs(oracle["fold_evaluated"]) <= 4 * oracle["fold_evaluated_se"] + 1e-12


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "rows__declared",
            "rows__shrunken_se_control",
            "rows__noise_control",
            "weighted__declared",
            "weighted__shrunken_se_control",
            "weighted__noise_control",
            "clustered__declared",
            "clustered__shrunken_se_control",
            "clustered__noise_control",
            "clustered__row_variance_control",
        ),
        "fold_evaluated_reporting": ("clustered__fold_evaluated",),
    }
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert study.STUDY.truth_varies_by_replicate and not study.STUDY.calibration_efficiency_ratio


def test_the_fit_is_the_declared_one() -> None:
    estimator = law.estimator(base.outcome_learner(), 5)
    assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
    assert estimator.cross_fit and not estimator.cv_evaluation
    assert estimator.n_folds == 10 and estimator.repeats == 1
    assert estimator.targeting_scheme == "pooled" and estimator.g_bounds == "auto"
    assert estimator.random_state == 5
    assert law.estimator(base.outcome_learner(), 5, cv_evaluation=True).cv_evaluation
    assert study.PRIMARY_N == properties.CALIBRATION_N == 2_000


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/inference/cluster.py",
        "tests/studies/_learned_rule_stacked_law.py",
        "tests/studies/_learned_rule_compositions_law.py",
        "tests/studies/_learned_rule_law.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_unequal_clusters_rule_is_declared() -> None:
    """The rule names the clustered cell, its bias verdict, all three of its controls, and the
    reporting row's paired difference, without which the fold-evaluated report would pass it
    too (design review finding 1)."""
    text = " ".join(study.UNEQUAL_CLUSTERS_RULE)
    for cell in (
        "fold_evaluated_reporting/clustered__fold_evaluated",
        "paired_error_difference + 2.576 * paired_error_difference_se < 0",
        "clustered__declared",
        "clustered__row_variance_control",
        "clustered__shrunken_se_control",
        "clustered__noise_control",
        "bias_equivalent",
    ):
        assert cell in text


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_strata as e2

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module in (e1, e2, learners, split_mean):
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others


def test_the_named_streams_are_distinct_from_every_registered_study() -> None:
    labels = [
        ("root_n_rate", "empirical_sd"),
        ("interval_calibration", "correctly_specified"),
        ("performance", "cleverly", "binary", "ate"),
        ("equivalence", "binary", "ate"),
    ]
    records = [*_other_studies(), study.STUDY]
    seeds = [stream_seed(record, *label) for record in records for label in labels]
    assert len(set(seeds)) == len(seeds)


#: The rule-L3 suffix of every property draw label.
DECLARED_SUFFIXES: dict[tuple[str, str], tuple[object, ...]] = {
    ("interval_calibration", "rows__declared"): (),
    ("interval_calibration", "weighted__declared"): (),
    ("interval_calibration", "clustered__declared"): (),
}


def test_rule_l3_keeps_every_label() -> None:
    assert properties.suffixes(study.STUDY) == DECLARED_SUFFIXES


def _own_sample_seeds() -> set[int]:
    return base.declared_sample_seeds(
        study.STUDY,
        study.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(study.STUDY),
    )


def test_the_sample_seeds_are_distinct() -> None:
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert len(_own_sample_seeds()) == total


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_arms as h
    from tests.studies import learned_rule_arms_properties as h_properties
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_contrast as g
    from tests.studies import learned_rule_contrast_properties as g_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_cvtmle_properties as gated_properties
    from tests.studies import learned_rule_in_sample as f2
    from tests.studies import learned_rule_in_sample_properties as f2_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_optimal_value as i
    from tests.studies import learned_rule_optimal_value_properties as i_properties
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_split_mean_properties as split_mean_properties
    from tests.studies import learned_rule_strata as e2
    from tests.studies import learned_rule_strata_properties as e2_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for record, module in (
        (gated, gated_properties),
        (compositions, composition_properties),
        (split_mean, split_mean_properties),
        (learners, learner_properties),
        (e1, e1_properties),
        (e2, e2_properties),
        (f2, f2_properties),
        (g, g_properties),
        (h, h_properties),
        (i, i_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


#: The four sample seeds this study shares with another X11 study: (this study's draw, the other
#: draw). Seeds are 32-bit, and about 32,400 against the other X11 studies' 278,560 expect two
#: or three coincidences by chance. ``rows`` replicate 1,784 draws RM30's law at n = 2,000 and
#: the E2 binary draw at n = 1,000 draws ``W1`` first too, so the two share their first 1,000
#: values of ``W1`` and nothing else (the E1 precedent). A clustered draw spends its first draws
#: on 200 normal cluster effects, not on uniform ``W1``, so it shares no variable with the other
#: draw. One replicate of 8,800 in each cell makes the dependence negligible, and no verdict
#: reads two studies jointly.
SHARED_SEEDS = {
    ("rows__declared", 1_784): ("learned-rule-strata", "interval_calibration/binary_steep", 9_841),
    ("clustered__declared", 2_299): ("learned-rule-cvtmle", "primary/non_exceptional", 1_874),
    ("clustered__declared", 5_240): ("learned-rule-compositions", "primary/weighted", 3_894),
    ("clustered__declared", 7_192): ("learned-rule-strata", "interval_calibration/weighted", 4_518),
}


def test_no_sample_seed_is_another_x11_studys_but_four_declared() -> None:
    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    shared = {
        stream_seed(study.STUDY, "property_sample", "interval_calibration", label, replicate)
        for label, replicate in SHARED_SEEDS
    }
    assert own & others == shared
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_shared_rows_draw_shares_only_a_w1_prefix() -> None:
    from tests.studies import _learned_rule_strata_law as e2_law

    seed = stream_seed(
        study.STUDY, "property_sample", "interval_calibration", "rows__declared", 1_784
    )
    here, there = law.draw("rows", 2_000, seed), e2_law.draw("binary_steep", 1_000, seed)
    np.testing.assert_array_equal(here["W1"].to_numpy()[:1_000], there["W1"].to_numpy())
    assert not np.array_equal(here["W2"].to_numpy()[:1_000], there["W2"].to_numpy())


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, "rows", 0) in declared


# ---------------------------------------------------------------------------- the sizing

#: The pilot (``reviews/x11/scripts/X11-F1-calibration-pilot.py`` in the plans repository,
#: throwaway label 20267501, 4,000 draws of each positive at nominal n = 2,000, V = 10): the mean
#: reported SE over the SD of ``estimate - truth``, in ``CALIBRATION_LABELS`` order.
PILOT_DRAWS = 4_000
PILOT_SE_RATIOS = (0.9936, 0.9758, 0.9882)

#: The pilot readings of each control on the same draws: (SE ratio, coverage). The noise
#: controls add one efficiency-bound unit of noise from ``default_rng(1)``.
PILOT_CONTROLS = {
    "rows__shrunken_se_control": (0.7949, 0.8808),
    "rows__noise_control": (0.7071, 0.8350),
    "weighted__shrunken_se_control": (0.7806, 0.8760),
    "weighted__noise_control": (0.6981, 0.8285),
    "clustered__shrunken_se_control": (0.7906, 0.8755),
    "clustered__noise_control": (0.7060, 0.8345),
    "clustered__row_variance_control": (0.7624, 0.8605),
}


def test_the_calibration_count_is_the_per_arm_rule() -> None:
    """``per_arm_replicates`` over the three positives at 4,000 pilot draws gives 8,800
    (family-wise 0.9528); 8,400 misses (0.9475). ``weighted`` (0.9758) sets it."""
    assert properties.CALIBRATION_REPLICATES == 8_800
    sized = tools.per_arm_replicates(PILOT_SE_RATIOS, PILOT_DRAWS)
    assert sized["R"] == properties.CALIBRATION_REPLICATES
    assert min(sized["per_cell"]) >= tools.CELL_TARGET
    below = tools.per_arm_pass(
        PILOT_SE_RATIOS, PILOT_DRAWS, properties.CALIBRATION_REPLICATES - 400
    )
    assert float(np.prod(below)) < tools.FAMILY_TARGET or min(below) < tools.CELL_TARGET
    half = tools.screen_half_width(PILOT_DRAWS)
    assert all(abs(ratio - 1.0) <= half for ratio in PILOT_SE_RATIOS)


def test_every_control_is_pilot_checked_below_the_band() -> None:
    """Each control's pilot SE ratio sits below the band floor 0.93 by many of its MC SEs at
    the declared count, so it fails as declared."""
    floor = study.STUDY.margins.calibration_se_ratio[0]
    assert set(PILOT_CONTROLS) == {
        cell
        for cell in study.PROPERTY_CELLS["interval_calibration"]
        if not cell.endswith("__declared")
    }
    for cell, (ratio, coverage) in PILOT_CONTROLS.items():
        assert ratio < floor - 0.05, cell
        assert coverage < study.STUDY.margins.calibration_coverage[0], cell
