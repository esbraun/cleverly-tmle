"""The declared constants of the ``learned-rule-in-sample`` study, recomputed without a study run.

Each number here was fixed before any registered run. The law checks call no sampler: they read
the population projection of each declared learner and integrate on the RM30 grid. The pilot
readings and the sizing rules that read them are pinned. The declared seeds are only computed,
never drawn from, except the two shared ones, whose draws show which variables they share.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from scipy.stats import norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_in_sample_law as law
from tests.studies import learned_rule_compositions as compositions
from tests.studies import learned_rule_in_sample as study
from tests.studies import learned_rule_in_sample_properties as properties
from tests.studies import learned_rule_split_mean as split_mean
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_in_sample_design")

base = law.base


def _optimal_rule(name: str) -> dict[float, np.ndarray]:
    return base.fixed_rule_grid(
        lambda w1, w2: base.qbar0(1.0, w1, w2, name) > base.qbar0(0.0, w1, w2, name)
    )


# ----------------------------------------------------------------------------- the law


@pytest.mark.parametrize("scenario", law.SCENARIOS)
def test_the_efficiency_bounds(scenario: str) -> None:
    assert law.efficiency_bound(scenario) == pytest.approx(
        study.EFFICIENCY_BOUNDS[scenario], abs=2e-6
    )


def test_the_bounds_are_the_earlier_studies_of_the_same_laws() -> None:
    """The same laws and learner as ``learned-rule-compositions`` (``fold_targeted``) and
    ``learned-rule-split-mean``; only the sample splitting differs."""
    assert (
        study.EFFICIENCY_BOUNDS["non_exceptional"]
        == compositions.EFFICIENCY_BOUNDS["fold_targeted"]
    )
    assert study.EFFICIENCY_BOUNDS == split_mean.EFFICIENCY_BOUNDS


def test_the_misspecified_limit_rule_is_not_the_optimal_one() -> None:
    """The declared learner cannot represent the ``misspecified_limit`` blip: its limit rule's
    value is 0.554749 against the optimal 0.560251 (``test_learned_rule_study_constants``)."""
    limit, _ = base.rule_value(law.population_rule("misspecified_limit"), "misspecified_limit")
    optimal, _ = base.rule_value(_optimal_rule("misspecified_limit"), "misspecified_limit")
    assert limit == pytest.approx(0.554749, abs=2e-6)
    assert optimal == pytest.approx(0.560251, abs=2e-6)


def test_the_necessity_learner_has_a_strong_limit_blip() -> None:
    """``WithoutW1MainLogistic``'s limit blip on ``non_exceptional`` runs from about -0.31 to
    +0.35 on the probability scale, so its one rule settles at n = 2,000, and its limit rule's
    value is within 2e-5 of the optimal value: the untargeted bias the control must show is in the outcome
    regression, not in the rule."""
    blips = law.limit_blips("non_exceptional", law.without_w1_main_learner())
    every = np.concatenate(list(blips.values()))
    assert every.min() == pytest.approx(-0.31, abs=0.01)
    assert every.max() == pytest.approx(0.35, abs=0.01)
    rule = law.grid_rule(law.projection("non_exceptional", law.without_w1_main_learner()))
    value, _ = base.rule_value(rule, "non_exceptional")
    optimal, _ = base.rule_value(_optimal_rule("non_exceptional"), "non_exceptional")
    assert value == pytest.approx(optimal, abs=2e-5)
    assert optimal == pytest.approx(0.577115, abs=2e-6)


def test_the_weak_limit_learner_has_a_weak_limit_blip() -> None:
    """``DropW1Logistic``'s limit blip is constant in ``W1``: 0.037 at ``W2 = 0`` and 0.049 at
    ``W2 = 1``, small beside its sampling error at n = 2,000 (the design record, D2)."""
    blips = law.limit_blips("non_exceptional", base.drop_w1_learner())
    for w2, declared in ((0.0, 0.037), (1.0, 0.049)):
        assert np.ptp(blips[w2]) < 1e-9
        assert blips[w2][0] == pytest.approx(declared, abs=5e-4)


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "non_exceptional__declared",
            "non_exceptional__shrunken_se_control",
            "non_exceptional__noise_control",
            "misspecified_limit__declared",
            "misspecified_limit__shrunken_se_control",
            "misspecified_limit__noise_control",
            "non_exceptional__interpolating_control",
        ),
        "targeting_necessity": ("non_exceptional__targeted", "non_exceptional__untargeted"),
        "target_distinction": ("non_exceptional__fold_average",),
        "weak_limit_necessity": (
            "non_exceptional_drop_w1__targeted",
            "non_exceptional_drop_w1__untargeted",
        ),
    }
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert properties.TARGETING_DISPLACEMENT == 0.5
    assert study.STUDY.truth_varies_by_replicate and not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == 2_000


def test_the_fit_is_the_declared_one() -> None:
    estimator = law.estimator(law.InteractionLogistic(), 5)
    assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
    assert not estimator.cross_fit and not estimator.cv_evaluation
    assert estimator.g_bounds == "auto" and estimator.random_state == 5
    assert law.InteractionLogistic.cleverly_learner_family == "parametric"
    assert law.WithoutW1MainLogistic.cleverly_learner_family == "parametric"
    assert law.DropW1Logistic.cleverly_learner_family == "parametric"
    assert not hasattr(law.ClippedInterpolatingTree, "cleverly_learner_family")


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/_support.py",
        "tests/studies/_learned_rule_in_sample_law.py",
        "tests/studies/_learned_rule_law.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_support_row_options() -> None:
    """The row the registering change may add names both nuisance families and nothing else:
    no cell fits a rule learner or passes ``treatment_probabilities=``."""
    assert study.SUPPORT_OPTIONS == {
        "outcome_learner_family": "parametric",
        "treatment_learner_family": "parametric",
    }
    assert law.FAMILIES["declared"] == {
        "outcome_learner_family": "parametric",
        "treatment_learner_family": "parametric",
    }


def test_the_declared_gates_are_read() -> None:
    """The calibration rule and the necessity rule are the study's rules; the two diagnostic
    families have none, and their published role is diagnostic."""
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert "interval_calibration" in families and "targeting_necessity" in families
    assert not families & {study.REPORTING_FAMILY, study.WEAK_LIMIT_FAMILY}
    assert properties.TARGETING_NECESSITY.threshold == properties.TARGETING_DISPLACEMENT
    assert properties.TARGETING_NECESSITY.truth_varies
    assert properties.NECESSITY_KINDS[study.WEAK_LIMIT_FAMILY][1] == (
        properties.DIAGNOSTIC_ROLE,
        properties.DIAGNOSTIC_ROLE,
    )


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_stacked as f1
    from tests.studies import learned_rule_strata as e2

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module in (e1, e2, f1, learners, split_mean):
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
    ("interval_calibration", "non_exceptional__declared"): (),
    ("interval_calibration", "misspecified_limit__declared"): (),
    ("targeting_necessity", "non_exceptional"): (),
    ("weak_limit_necessity", "non_exceptional_drop_w1"): (),
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
    assert total == 33_600
    assert len(_own_sample_seeds()) == total


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_cvtmle_properties as gated_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_split_mean_properties as split_mean_properties
    from tests.studies import learned_rule_stacked as f1
    from tests.studies import learned_rule_stacked_properties as f1_properties
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
        (f1, f1_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


#: The two sample seeds this study shares with another X11 study. Seeds are 32-bit, and about
#: 33,600 against the other X11 studies' 310,956 expect two or three coincidences by chance.
#: The primary draw shares ``W1``, ``W2`` and ``A`` on all 2,000 rows with the split-mean
#: calibration draw (the same sampler at the same n, another blip, so not ``Y``). The calibration
#: draw shares the first 1,000 values of ``W1`` with the E2 diagnostic draw at n = 1,000 (the E1
#: precedent). One replicate of thousands in each cell makes the dependence negligible, and no
#: verdict reads two studies jointly.
SHARED_PRIMARY_SEED = ("misspecified_limit", 6_476)
SHARED_PROPERTY_SEED = ("misspecified_limit__declared", 156)
SPLIT_MEAN_PARTNER = ("interval_calibration", "non_exceptional__declared", 593)
E2_PARTNER = ("weak_margin_diagnostic", "binary_weak", 3_312)


def _shared_property_seed() -> int:
    label, replicate = SHARED_PROPERTY_SEED
    return stream_seed(study.STUDY, "property_sample", "interval_calibration", label, replicate)


def _shared_primary_seed() -> int:
    return replicate_seed(study.STUDY, *SHARED_PRIMARY_SEED)


def test_no_sample_seed_is_another_x11_studys_but_two_declared() -> None:
    from tests.studies import learned_rule_strata as e2

    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == {_shared_property_seed(), _shared_primary_seed()}
    partner = stream_seed(split_mean.STUDY, "property_sample", *SPLIT_MEAN_PARTNER)
    assert partner == _shared_primary_seed()
    assert stream_seed(e2.STUDY, "property_sample", *E2_PARTNER) == _shared_property_seed()
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_shared_primary_draw_shares_the_covariates_and_treatment() -> None:
    seed = _shared_primary_seed()
    here = base.draw("misspecified_limit", 2_000, seed)
    there = base.draw("non_exceptional", 2_000, seed)
    for column in ("W1", "W2", "A"):
        np.testing.assert_array_equal(here[column].to_numpy(), there[column].to_numpy())
    assert not np.array_equal(here["Y"].to_numpy(), there["Y"].to_numpy())


def test_the_shared_calibration_draw_shares_only_a_w1_prefix() -> None:
    from tests.studies import _learned_rule_strata_law as e2_law

    seed = _shared_property_seed()
    here = base.draw("misspecified_limit", 2_000, seed)
    there = e2_law.draw("binary_weak", 1_000, seed)
    np.testing.assert_array_equal(here["W1"].to_numpy()[:1_000], there["W1"].to_numpy())
    assert not np.array_equal(here["W2"].to_numpy()[:1_000], there["W2"].to_numpy())


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, "non_exceptional", 0) in declared


# ---------------------------------------------------------------------------- the sizing

#: The pilot (``reviews/x11/scripts/X11-F2-calibration-pilot.py`` in the plans repository,
#: throwaway label 20267601, 4,000 draws of each positive at n = 2,000): the mean reported SE
#: over the SD of ``estimate - truth``, in ``CALIBRATION_LABELS`` order, and the standardized
#: bias of each.
PILOT_DRAWS = 4_000
PILOT_SE_RATIOS = (1.0093, 1.0206)
PILOT_BIASES = (0.1231, 0.1898)

#: The ``WithoutW1MainLogistic`` pilot (4,000 draws on ``non_exceptional``, cell index 3 of the
#: label): the targeted arm's standardized bias, the untargeted arm's bias in its own SDs, and
#: the paired displacement in targeted-arm SDs.
NECESSITY_PILOT = {"targeted_bias": 0.1185, "untargeted_bias": 2.3965, "displacement": 1.8794}

#: The ``DropW1Logistic`` pilot on the same law (the plan's learner, now the weak-limit
#: diagnostic): the same three readings. Its targeted arm would miss the bias leg at any count
#: that resolves it, which is why it carries no verdict (the design record, D2).
WEAK_LIMIT_PILOT = {"targeted_bias": 0.3046, "untargeted_bias": 1.3714, "displacement": 1.0257}

#: The pilot readings of each control on the same draws: (SE ratio, coverage). The noise
#: controls add one efficiency-bound unit of noise from ``default_rng(1)``.
PILOT_CONTROLS = {
    "non_exceptional__shrunken_se_control": (0.8075, 0.8810),
    "non_exceptional__noise_control": (0.7140, 0.8325),
    "misspecified_limit__shrunken_se_control": (0.8165, 0.8842),
    "misspecified_limit__noise_control": (0.7139, 0.8290),
    "non_exceptional__interpolating_control": (0.3868, 0.0000),
}


def bias_pass(
    bias: float, replicates: int, pilot: int = PILOT_DRAWS, margin: float = 0.25
) -> float:
    """The predictive probability that the 99% interval of the mean standardized error falls
    inside ``±margin`` at ``replicates`` draws, given a pilot bias read from ``pilot`` draws.

    The interval's half-width is about ``2.576 / sqrt(R)`` SDs; the future mean is normal about
    the pilot bias with variance ``1/R + 1/pilot`` (pilot error and run error).
    """
    half = norm.ppf(0.995) / np.sqrt(replicates)
    spread = np.sqrt(1.0 / replicates + 1.0 / pilot)
    return float(
        norm.cdf((margin - half - bias) / spread) - norm.cdf((-margin + half - bias) / spread)
    )


def _smallest(biases: Sequence[float]) -> int:
    for replicates in range(400, 40_001, 400):
        if np.prod([bias_pass(b, replicates) for b in biases]) >= tools.FAMILY_TARGET:
            return replicates
    raise AssertionError("no count reaches the target")


def test_the_calibration_count_is_the_per_arm_rule() -> None:
    """``per_arm_replicates`` over the two positives at 4,000 pilot draws gives 7,200
    (family-wise 0.956); 6,800 misses. ``misspecified_limit`` (1.0206) sets it."""
    assert properties.CALIBRATION_REPLICATES == 7_200
    sized = tools.per_arm_replicates(PILOT_SE_RATIOS, PILOT_DRAWS)
    assert sized["R"] == properties.CALIBRATION_REPLICATES
    assert min(sized["per_cell"]) >= tools.CELL_TARGET
    below = tools.per_arm_pass(
        PILOT_SE_RATIOS, PILOT_DRAWS, properties.CALIBRATION_REPLICATES - 400
    )
    assert float(np.prod(below)) < tools.FAMILY_TARGET or min(below) < tools.CELL_TARGET
    half = tools.screen_half_width(PILOT_DRAWS)
    assert all(abs(ratio - 1.0) <= half for ratio in PILOT_SE_RATIOS)


def test_the_primary_count_is_the_bias_leg_rule() -> None:
    """Both primary bias legs at 8,400 pass with probability 0.9526; 8,000 gives 0.9476. The
    per-arm SE rule alone would give 7,200, at which the ``misspecified_limit`` bias leg is far
    likelier to miss (the design record, D3)."""
    assert study.PRIMARY_REPLICATES == _smallest(PILOT_BIASES) == 8_400
    at = np.prod([bias_pass(b, 8_400) for b in PILOT_BIASES])
    below = np.prod([bias_pass(b, 8_000) for b in PILOT_BIASES])
    assert at == pytest.approx(0.9526, abs=1e-4) and below == pytest.approx(0.9476, abs=1e-4)
    # The deliberate mutation: the RM30 primary count of 2,000 would miss.
    assert bias_pass(PILOT_BIASES[1], 2_000) < 0.6


def test_the_necessity_count_is_the_bias_leg_rule() -> None:
    """The positive arm's bias leg sets 1,200 (0.959); the control's bias and the displacement
    are many standard errors past their thresholds there."""
    assert _smallest([NECESSITY_PILOT["targeted_bias"]]) == properties.NECESSITY_REPLICATES
    assert properties.NECESSITY_REPLICATES == 1_200
    assert bias_pass(NECESSITY_PILOT["targeted_bias"], 1_200) == pytest.approx(0.9587, abs=1e-4)
    se = np.sqrt(1.0 / properties.NECESSITY_REPLICATES + 1.0 / PILOT_DRAWS)
    assert NECESSITY_PILOT["untargeted_bias"] - 0.25 > 20 * se
    assert NECESSITY_PILOT["displacement"] - properties.TARGETING_DISPLACEMENT > 20 * se


def test_the_weak_limit_learner_would_miss_its_bias_leg() -> None:
    """The reason D2 replaced the plan's learner: its targeted arm's bias leg passes with
    probability under 0.2 at any count up to 40,000."""
    assert (
        max(bias_pass(WEAK_LIMIT_PILOT["targeted_bias"], r) for r in (1_200, 8_400, 40_000)) < 0.2
    )


def test_every_control_is_pilot_checked_below_the_band() -> None:
    """Each control's pilot SE ratio sits below the band floor 0.93 by more than 0.05, and its
    coverage below the coverage floor, so it fails as declared."""
    floor = study.STUDY.margins.calibration_se_ratio[0]
    assert set(PILOT_CONTROLS) == {
        cell
        for cell in study.PROPERTY_CELLS["interval_calibration"]
        if not cell.endswith("__declared")
    }
    for cell, (ratio, coverage) in PILOT_CONTROLS.items():
        assert ratio < floor - 0.05, cell
        assert coverage < study.STUDY.margins.calibration_coverage[0], cell
