"""The declared constants of the ``learned-rule-in-sample`` study, recomputed without a study run.

Each number here was fixed before any registered run. The law checks call no sampler: they read
the population projection of each declared learner and integrate on the RM30 grid. The pilot
readings and the sizing rules that read them are pinned. The declared seeds are only computed,
never drawn from, except the shared ones, whose draws show which variables they share.
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


def test_the_necessity_learners_rule_moves_continuously() -> None:
    """At each ``W2``, ``WithoutW1MainLogistic``'s limit blip crosses zero once, at an interior
    ``W1`` (about -0.08), with a slope of about 0.37 there. Sampling error moves the rule's
    boundary along ``W1`` and flips no whole level, so the rule settles as n grows (C3a), the
    condition D2 replaced the plan's learner for (the design record)."""
    blips = law.limit_blips("non_exceptional", law.without_w1_main_learner())
    for w2, root in ((0.0, -0.0857), (1.0, -0.0801)):
        blip = blips[w2]
        (crossing,) = np.flatnonzero(np.diff(np.sign(blip)) != 0)
        lower, upper = base.GRID[crossing], base.GRID[crossing + 1]
        found = lower - blip[crossing] * (upper - lower) / (blip[crossing + 1] - blip[crossing])
        assert found == pytest.approx(root, abs=1e-3)
        assert -0.9 < found < 0.9
        slope = (blip[crossing + 1] - blip[crossing]) / (upper - lower)
        assert slope == pytest.approx(0.372, abs=0.005)
    # The deliberate mutation: the plan's learner has no crossing at either level.
    weak = law.limit_blips("non_exceptional", base.drop_w1_learner())
    assert all((np.sign(blip) == 1).all() for blip in weak.values())


def test_the_weak_limit_learner_has_a_weak_limit_blip() -> None:
    """``DropW1Logistic``'s limit blip is constant in ``W1``: 0.037 at ``W2 = 0`` and 0.049 at
    ``W2 = 1``, small beside its sampling error at n = 2,000 (the design record, D2)."""
    blips = law.limit_blips("non_exceptional", base.drop_w1_learner())
    for w2, declared in ((0.0, 0.037), (1.0, 0.049)):
        assert np.ptp(blips[w2]) < 1e-9
        assert blips[w2][0] == pytest.approx(declared, abs=5e-4)


def _cell_moments(w2: float, a: float) -> tuple[float, float]:
    """``P(A = a, W2 = w2)`` and ``P(Y = 1 | A = a, W2 = w2)`` on ``non_exceptional``, by the
    trapezoid rule on the RM30 grid (``W1`` uniform on [-1, 1], ``W2`` a fair coin)."""
    treated = base.g0(base.GRID, w2)
    arm = treated if a else 1.0 - treated
    mass = 0.5 * np.trapezoid(arm, base.GRID) / 2.0
    outcome = 0.5 * np.trapezoid(arm * base.qbar0(a, base.GRID, w2, "non_exceptional"), base.GRID)
    return float(mass), float(outcome / 2.0 / mass)


def _flip_probability(w2: float, n: int) -> tuple[float, float, float]:
    """The limit blip, its SE at ``n`` rows and ``P(rule flips)`` of ``drop_w1`` at ``w2``.

    The learner is saturated in ``(A, W2)``, so its blip at each ``W2`` is the difference of two
    cell means of ``Y``, and its rule treats that level when the difference is positive. The
    probability is the normal one for that difference.
    """
    (control_mass, control), (treated_mass, treated) = (_cell_moments(w2, a) for a in (0, 1))
    blip = treated - control
    se = np.sqrt(
        treated * (1 - treated) / (n * treated_mass) + control * (1 - control) / (n * control_mass)
    )
    return blip, float(se), float(norm.cdf(-blip / se))


#: The exact-law flip probability of the ``drop_w1`` rule at each ``W2`` at n = 2,000, and
#: the pilot's share of draws whose rule flipped at that level (446 and 254 of 4,000).
FLIP_PROBABILITIES = {0.0: 0.1186, 1.0: 0.0632}
PILOT_FLIP_SHARES = {0.0: 446 / 4_000, 1.0: 254 / 4_000}


def test_the_weak_limit_rule_flips_in_one_sample_in_six() -> None:
    """The reason D2 replaced the plan's learner, computed from the law before any draw: at
    n = 2,000 the ``drop_w1`` rule differs from its limit rule at ``W2 = 0`` with probability
    0.119 and at ``W2 = 1`` with probability 0.063, about 17.5% of samples in all. Condition
    C3a (the rule settles) fails at this n, so its in-sample targeted arm measures the winner's
    curse of an unsettled rule, not the fluctuation."""
    limits = law.limit_blips("non_exceptional", base.drop_w1_learner())
    for w2, declared in FLIP_PROBABILITIES.items():
        blip, se, probability = _flip_probability(w2, study.PRIMARY_N)
        assert blip == pytest.approx(limits[w2][0], abs=1e-6)
        assert probability == pytest.approx(declared, abs=2e-3)
        assert probability == pytest.approx(PILOT_FLIP_SHARES[w2], abs=0.015)
        assert blip / se < 2.0
    either = 1.0 - np.prod([1.0 - p for p in FLIP_PROBABILITIES.values()])
    assert either == pytest.approx(0.175, abs=2e-3)
    # The deliberate mutation: at 20 times the rows the same rule has settled.
    assert all(_flip_probability(w2, 40_000)[2] < 1e-5 for w2 in base.W2_LEVELS)


def test_the_weak_limit_rule_value_tells_a_flipped_draw() -> None:
    """``WEAK_LIMIT_RULE_VALUE`` is the value of the ``drop_w1`` limit rule (treat every row),
    and each of the three other rules the learner can learn sits at least 7e-3 from it, so
    ``report_weak_limit`` reads a flip from a draw's truth."""
    limit = law.population_rule("non_exceptional", base.drop_w1_learner())
    assert all(rule.all() for rule in limit.values())
    value, _ = base.rule_value(limit, "non_exceptional")
    assert value == pytest.approx(study.WEAK_LIMIT_RULE_VALUE, abs=1e-12)
    others = [
        base.rule_value(
            {0.0: np.full(base.GRID.shape, low), 1.0: np.full(base.GRID.shape, high)},
            "non_exceptional",
        )[0]
        for low, high in ((False, True), (True, False), (False, False))
    ]
    gap = min(abs(other - value) for other in others)
    assert gap > 7e-3
    assert gap / properties.FLIP_TOLERANCE >= 1e3


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
        "target_distinction": (
            "non_exceptional__fold_average",
            "misspecified_limit__fold_average",
        ),
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


#: The one pair of this study's own draws that share a sample seed. Rule L3 judges a sample
#: seed against the seeds of the same law, and these two draw different laws: primary
#: ``misspecified_limit`` replicate 13,424 and ``interval_calibration``
#: ``non_exceptional__declared`` replicate 834. The sampler is the same at the same n, so the
#: two share ``W1``, ``W2`` and ``A`` on all 2,000 rows, not ``Y`` (another blip). Each is one
#: draw of 14,400 or 7,200, and no verdict reads the two cells jointly. It appeared when D6
#: moved the primary count from 8,400.
OWN_SHARED = (
    ("primary", "misspecified_limit", 13_424),
    ("interval_calibration", "non_exceptional__declared", 834),
)


def _seed_of(family: str, setting: str, replicate: int) -> int:
    if family == "primary":
        return replicate_seed(study.STUDY, setting, replicate)
    resolved = properties.suffixes(study.STUDY)
    return stream_seed(
        study.STUDY, "property_sample", family, setting, replicate, *resolved[family, setting]
    )


def test_the_sample_seeds_are_distinct_but_one_declared_pair() -> None:
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert total == 45_600
    assert len(_own_sample_seeds()) == total - 1
    first, second = OWN_SHARED
    assert _seed_of(*first) == _seed_of(*second)
    here = base.draw("misspecified_limit", 2_000, _seed_of(*first))
    there = base.draw("non_exceptional", 2_000, _seed_of(*second))
    for column in ("W1", "W2", "A"):
        np.testing.assert_array_equal(here[column].to_numpy(), there[column].to_numpy())
    assert not np.array_equal(here["Y"].to_numpy(), there["Y"].to_numpy())


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_contrast as g
    from tests.studies import learned_rule_contrast_properties as g_properties
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
        (g, g_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


#: The two sample seeds this study shares with another X11 study. Seeds are 32-bit, and about
#: 45,600 against the other X11 studies' 310,956 expect about 3.3 coincidences by chance.
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

#: The design reviewer's probe of the declared ``misspecified_limit`` fit (4,000 throwaway draws
#: at n = 2,000 from labels 99117731 and 99117732, which no declared stream reads): its
#: standardized bias, and the reading of the pilot's and the probe's 8,000 draws pooled. The
#: primary count reads the pooled value (the design record, D6).
PROBE_DRAWS = 4_000
PROBE_BIAS = 0.2206
POOLED_DRAWS = PILOT_DRAWS + PROBE_DRAWS
POOLED_BIAS = 0.2052

#: The bias reading of each primary law that sizes the primary count, and the draws it rests on:
#: ``non_exceptional`` from the pilot, ``misspecified_limit`` from the pooled draws.
PRIMARY_BIAS_READINGS = ((PILOT_BIASES[0], PILOT_DRAWS), (POOLED_BIAS, POOLED_DRAWS))

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


def _pass(readings: Sequence[tuple[float, int]], replicates: int) -> float:
    """The product of the bias legs' pass probabilities, each from its own pilot size."""
    return float(np.prod([bias_pass(b, replicates, pilot) for b, pilot in readings]))


def _smallest(readings: Sequence[tuple[float, int]]) -> int:
    for replicates in range(400, 40_001, 400):
        if _pass(readings, replicates) >= tools.FAMILY_TARGET:
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


def test_the_pooled_reading_is_the_two_readings_weighted_by_draws() -> None:
    """The pilot and the probe agree within 1.4 combined SEs, and the pooled reading sits
    between them, weighted by their draws."""
    pooled = (PILOT_BIASES[1] * PILOT_DRAWS + PROBE_BIAS * PROBE_DRAWS) / POOLED_DRAWS
    assert abs(POOLED_BIAS - pooled) < 1e-4
    combined = np.sqrt(1.0 / PILOT_DRAWS + 1.0 / PROBE_DRAWS)
    assert abs(PROBE_BIAS - PILOT_BIASES[1]) / combined < 1.4


def test_the_primary_count_is_the_bias_leg_rule() -> None:
    """Both primary bias legs at 14,400 pass with probability 0.9529 at the pooled reading;
    14,000 gives 0.9498. The per-arm SE rule alone would give 7,200 (the design record, D3).
    The count declared on the pilot reading alone, 8,400, would pass with probability 0.857
    at the pooled reading, which is why it moved before the run (D6)."""
    assert study.PRIMARY_REPLICATES == _smallest(PRIMARY_BIAS_READINGS) == 14_400
    assert _pass(PRIMARY_BIAS_READINGS, 14_400) == pytest.approx(0.9529, abs=1e-4)
    assert _pass(PRIMARY_BIAS_READINGS, 14_000) == pytest.approx(0.9498, abs=1e-4)
    # The first declaration: the pilot reading alone set 8,400 (0.9526), which the pooled
    # reading puts at 0.857.
    pilot_only = tuple((bias, PILOT_DRAWS) for bias in PILOT_BIASES)
    assert _smallest(pilot_only) == 8_400
    assert _pass(pilot_only, 8_400) == pytest.approx(0.9526, abs=1e-4)
    assert _pass(PRIMARY_BIAS_READINGS, 8_400) == pytest.approx(0.857, abs=1e-3)
    # The deliberate mutation: the RM30 primary count of 2,000 would miss.
    assert bias_pass(POOLED_BIAS, 2_000, POOLED_DRAWS) < 0.6


def test_the_necessity_count_is_the_bias_leg_rule() -> None:
    """The positive arm's bias leg sets 1,200 (0.959); the control's bias and the displacement
    are many standard errors past their thresholds there."""
    assert (
        _smallest([(NECESSITY_PILOT["targeted_bias"], PILOT_DRAWS)])
        == properties.NECESSITY_REPLICATES
    )
    assert properties.NECESSITY_REPLICATES == 1_200
    assert bias_pass(NECESSITY_PILOT["targeted_bias"], 1_200) == pytest.approx(0.9587, abs=1e-4)
    se = np.sqrt(1.0 / properties.NECESSITY_REPLICATES + 1.0 / PILOT_DRAWS)
    assert NECESSITY_PILOT["untargeted_bias"] - 0.25 > 20 * se
    assert NECESSITY_PILOT["displacement"] - properties.TARGETING_DISPLACEMENT > 20 * se


def test_a_learner_whose_rule_flips_would_miss_its_bias_leg() -> None:
    """A consequence of the flips, not the reason for D2: a learner whose rule differs from its
    limit rule in 17.5% of samples (``test_the_weak_limit_rule_flips_in_one_sample_in_six``)
    carries a winner's-curse bias, and at the pilot's +0.30 SD its bias leg would pass with
    probability under 0.2 at any count up to 40,000."""
    assert (
        max(bias_pass(WEAK_LIMIT_PILOT["targeted_bias"], r) for r in (1_200, 14_400, 40_000)) < 0.2
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
