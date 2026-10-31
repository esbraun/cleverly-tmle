"""The declared constants of the ``learned-rule-learners`` study, recomputed or pinned.

Each number here was fixed before any registered run. The limit rules, their values and shares
and the efficiency bounds are recomputed by quadrature from the population projections of the
declared learners on the RM30 grid; the pilot readings and the sizing rule that reads them are
pinned; and the study's seeds are checked against every other learned-rule study. No check draws
a sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_learners_law as law
from tests.studies import learned_rule_compositions as compositions
from tests.studies import learned_rule_compositions_properties as composition_properties
from tests.studies import learned_rule_cvtmle as gated
from tests.studies import learned_rule_cvtmle_boundary as boundary
from tests.studies import learned_rule_cvtmle_properties as gated_properties
from tests.studies import learned_rule_learners as study
from tests.studies import learned_rule_learners_properties as properties
from tests.studies import learned_rule_split_mean as split_mean
from tests.studies import learned_rule_split_mean_properties as split_mean_properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.seeds import replicate_seed, stream_seed
from tests.unit import test_learned_rule_learners as witnesses

# ------------------------------------------------------------------------------ the limits

#: The population value of each limit rule at ``misspecified_limit``.
LIMIT_VALUES = {
    "plug_in": 0.5547491,
    "blip": 0.5601869,
    "weighted_classification": 0.5602490,
    "optimal": 0.5602505,
}

#: The population share of ``W`` on which two limit rules differ.
SHARES = {
    ("plug_in", "blip"): 0.14737,
    ("plug_in", "weighted_classification"): 0.13875,
    ("blip", "weighted_classification"): 0.00862,
}

#: The value of ``W1`` at which each limit score changes sign, at ``W2 = 0`` and ``W2 = 1``; one
#: crossing each on ``[-1, 1]``.
THRESHOLDS = {
    "plug_in": (0.005025, 0.001387),
    "blip": (0.317040, 0.278985),
    "weighted_classification": (0.284236, 0.277309),
    "optimal": (0.278233, 0.278233),
}

#: C3: the population share of ``W`` whose limit score lies within ``epsilon`` of its threshold,
#: for ``epsilon`` 0.01 and 0.001 on the score's own scale (a blip difference, or a probability
#: minus one half). Each share is linear in ``epsilon``, so no mass sits at the threshold.
NEAR_THRESHOLD = {
    "blip": (0.03525, 0.00362),
    "weighted_classification": (0.06175, 0.00600),
}


def test_the_blip_limit_is_the_w_g1_projection() -> None:
    """The blip limit from the law of the pseudo-outcome is the projection of the true blip.

    With ``g = g0``, ``E[D | W]`` is the true blip whatever ``Qbar*`` is, so the projection read
    from the law of ``D`` equals W-g1's projection of the true blip, and the declared share and
    values are W-g1's.
    """
    limits = witnesses._projections()
    scores = law.limit_scores()
    for name in ("plug_in", "blip", "optimal"):
        np.testing.assert_array_equal(scores[name] > 0.0, limits[name])
    assert law.disagreement("plug_in", "blip") == pytest.approx(witnesses.DECLARED_SHARE, abs=5e-4)
    for name, value in witnesses.DECLARED_VALUES.items():
        assert law.limit_value(name)[0] == pytest.approx(value, abs=5e-5)


def test_the_limit_values_shares_and_thresholds() -> None:
    for name, value in LIMIT_VALUES.items():
        assert law.limit_value(name)[0] == pytest.approx(value, abs=1e-7)
    for (first, second), share in SHARES.items():
        assert law.disagreement(first, second) == pytest.approx(share, abs=1e-5)
    step = base.GRID[1] - base.GRID[0]
    for name, expected in THRESHOLDS.items():
        found = law.thresholds(name)
        for w2, threshold in zip(base.W2_LEVELS, expected, strict=True):
            (crossing,) = found[w2]
            assert crossing == pytest.approx(threshold, abs=1e-6), (name, w2)
            rule = law.limit_rule(name)[w2]
            first = base.GRID[np.argmax(rule)]
            assert np.all(rule == (first <= base.GRID)) and abs(first - threshold) <= step
    # Both rule learners beat the plug-in rule in the limit, by about 0.0054.
    gain = LIMIT_VALUES["blip"] - LIMIT_VALUES["plug_in"]
    assert 0.0054 < gain < LIMIT_VALUES["weighted_classification"] - LIMIT_VALUES["plug_in"]


def test_no_limit_rule_has_mass_at_its_threshold() -> None:
    """C3 for both losses: the near-threshold share is of order ``epsilon``."""
    for name, (wide, narrow) in NEAR_THRESHOLD.items():
        assert law.near_threshold(name, 0.01) == pytest.approx(wide, abs=1e-5)
        assert law.near_threshold(name, 0.001) == pytest.approx(narrow, abs=1e-5)
        assert 8.0 < wide / narrow < 12.0


def test_the_classification_limit_is_integrated_under_the_inner_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The weighted-classification limit reads the law of ``|D|`` given ``W``, not the blip.

    Two nonzero witnesses. Replacing ``Qbar*`` by the true ``Qbar0`` leaves the blip limit
    unchanged bit for bit and moves the classification threshold at ``W2 = 0`` by about 0.0018.
    Dropping the weights ``|D|`` (the majority sign) moves it far more.
    """
    honest = law.limit_scores()
    declared = law.thresholds("weighted_classification")

    class Truth:
        def predict_proba(self, design: np.ndarray) -> np.ndarray:
            q = base.qbar0(design[:, 0], design[:, 1], design[:, 2], law.LAW)
            return np.column_stack([1.0 - q, q])

    monkeypatch.setattr(law, "outcome_projection", lambda: Truth())
    law.limit_scores.cache_clear()
    try:
        moved = law.limit_scores()
        np.testing.assert_allclose(moved["blip"], honest["blip"], atol=1e-12, rtol=0)
        shift = law.thresholds("weighted_classification")[0.0][0] - declared[0.0][0]
        assert -0.0025 < shift < -0.0012
    finally:
        law.limit_scores.cache_clear()
    monkeypatch.undo()
    index, value, weight = law._pseudo_outcome_law()
    covariates, mass = law._quadrature()
    majority = law.classifier().set_params(fit__C=1e12, fit__tol=1e-13, fit__max_iter=50_000)
    majority.fit(
        covariates[index], (value > 0.0).astype(float), fit__sample_weight=weight / weight.mean()
    )
    score = majority.predict_proba(covariates)[:, 1] - 0.5
    moved_share = float(np.sum(mass * ((score > 0.0) != (honest["weighted_classification"] > 0.0))))
    assert moved_share == pytest.approx(0.0541, abs=5e-4)


def test_the_truncation_does_not_bind_in_the_limit() -> None:
    lower = 5.0 / (np.sqrt(study.PRIMARY_N) * np.log(study.PRIMARY_N))
    assert lower == pytest.approx(0.014709, abs=1e-6)
    g0 = np.concatenate([base.g0(base.GRID, w2) for w2 in base.W2_LEVELS])
    assert (g0.min(), g0.max()) == pytest.approx(law.G0_RANGE, abs=1e-12)
    assert min(g0.min(), 1.0 - g0.max()) > 25.0 * lower


def test_the_efficiency_bounds() -> None:
    for loss in law.LOSSES:
        _, variance = law.limit_value(loss)
        assert np.sqrt(variance) == pytest.approx(study.EFFICIENCY_BOUNDS[loss], abs=2e-6)


# ------------------------------------------------------------------------------- the pilot

#: The calibration pilot at the declared calibration size n = 4,000: 2,000 draws of each
#: positive from ``SeedSequence([20263299, 10 + cell, draw])``, a label no declared stream reads.
#: Each draw fits the declared rule learner, its fold-locality control and the plug-in rule on
#: one sample and fold seed. Each value is the mean reported SE over the SD of the error
#: ``estimate - truth``, in the order of :data:`law.LOSSES`. No draw failed, no rule-L4 refit
#: disagreed on any row, and no solver warned. The script is
#: ``reviews/x11/scripts/X11-C-calibration-pilot.py`` in the plans repository.
CALIBRATION_PILOT = (1.0094, 0.9925)
CALIBRATION_PILOT_DRAWS = 2_000

#: The calibration pilot's coverage of the positives and of the fold-locality controls.
PILOT_COVERAGE = (0.9530, 0.9505)
PILOT_CONTROL_COVERAGE = (0.2620, 0.2735)

#: The pilot at the primary size n = 2,000: 4,000 draws of each positive from
#: ``SeedSequence([20263299, cell, draw])``. SE ratios, coverages and standardized biases, in
#: the order of :data:`law.LOSSES`. The SE ratio sits below one, and the oracle SE's ratio
#: with it (0.951 and 0.967): the error spreads more than the fixed-rule variance of the
#: realized rules, through the covariance between folds whose rules are fitted on each other's
#: validation rows. Two solver warnings in 8,000 fits, no failure.
PRIMARY_PILOT_RATIO = (0.9544, 0.9708)
PRIMARY_PILOT_COVERAGE = (0.94175, 0.94375)
PRIMARY_PILOT_STANDARDIZED_BIAS = (0.0201, 0.0005)
PRIMARY_PILOT_DRAWS = 4_000

#: The design review's paired pilot at three sizes (label 20263297 for n = 1,000 and the extra
#: blip draws at n = 4,000): SE ratios, in the order of :data:`law.LOSSES`, and the paired
#: error-variance ratio of the rule learner to the plug-in rule on the same draws. The excess
#: declines with n and is within about 3% of zero at n = 4,000; the data pin no rate.
REVIEW_PILOT_SE_RATIO = {
    1_000: (0.946, 0.959),
    2_000: (0.955, 0.971),
    4_000: (1.001, 0.993),
}
REVIEW_PILOT_VARIANCE_RATIO = {
    1_000: (1.051, 1.051),
    2_000: (1.059, 1.031),
    4_000: (0.993, 0.973),
}


def test_the_calibration_replicates_follow_the_per_arm_rule() -> None:
    """R is the smallest multiple of 400 whose family-wise pass probability reaches 0.95.

    The two positives are two rule learners, each predicted from its own pilot ratio, as in the
    PR A and PR B studies (``mtp_study_tools.per_arm_replicates``). The arm-pooled rule reads
    their spread as between-arm noise and accepts no R.
    """
    rule = tools.per_arm_replicates(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert rule["R"] == properties.CALIBRATION_REPLICATES == 7_200
    assert min(rule["per_cell"]) >= tools.CELL_TARGET
    assert rule["family_wise"] >= tools.FAMILY_TARGET
    below = tools.per_arm_pass(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS, 6_800)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    assert tools.replicates_from_pilot(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)["R"] is None


def test_no_budget_sizes_the_band_at_the_primary_size() -> None:
    """Why the band is read at n = 4,000: at n = 2,000 no R up to 40,000 passes the rule.

    The blip cell's pass probability stays below 0.88 at every budget, because its pilot ratio
    0.954 sits within the pilot's own noise of the band's lower end 0.93.
    """
    rule = tools.per_arm_replicates(PRIMARY_PILOT_RATIO, PRIMARY_PILOT_DRAWS)
    assert rule["R"] is None
    assert rule["per_cell"][0] < 0.88
    assert properties.CALIBRATION_N == 2 * study.PRIMARY_N == 4_000


def test_the_three_size_pilot_supports_a_declining_excess_and_no_rate() -> None:
    """The record's D1 rationale: an excess at 1,000 and 2,000 that is near zero at 4,000.

    The SE ratio rises with n, the paired error-variance ratio of the rule learner to the
    plug-in rule is above one at 1,000 and 2,000 and within 3% of one at 4,000. The pooled blip
    ratio 1.001 at 4,000 (4,000 draws) sits below the declared pilot's 1.009, which sized R. The
    sizing at the declared n is conservative against it: the pooled blip ratio with the
    classification ratio would allow 4,400 per arm (taking both on 4,000 draws), and the
    declared 7,200 passes per cell with probability 0.998 (blip, 4,000 draws) and 0.979
    (classification, 2,000 draws).
    """
    for loss in range(2):
        ratios = [REVIEW_PILOT_SE_RATIO[n][loss] for n in (1_000, 2_000, 4_000)]
        assert ratios == sorted(ratios)
        assert REVIEW_PILOT_VARIANCE_RATIO[1_000][loss] > 1.03
        assert REVIEW_PILOT_VARIANCE_RATIO[2_000][loss] > 1.03
        assert abs(REVIEW_PILOT_VARIANCE_RATIO[4_000][loss] - 1.0) < 0.03
    pooled = (REVIEW_PILOT_SE_RATIO[4_000][0], CALIBRATION_PILOT[1])
    assert tools.per_arm_replicates(pooled, 4_000)["R"] == 4_400
    assert properties.CALIBRATION_REPLICATES == 7_200
    blip = tools.per_arm_pass((pooled[0],), 4_000, properties.CALIBRATION_REPLICATES)
    classification = tools.per_arm_pass((pooled[1],), 2_000, properties.CALIBRATION_REPLICATES)
    assert (round(float(blip[0]), 3), round(float(classification[0]), 3)) == (0.998, 0.979)


def test_the_diagnostic_cell_is_declared_apart_from_the_gated_cells() -> None:
    """One n = 1,000 cell per loss, 2,000 replicates, its own seed streams, and no verdict.

    A 2,000-draw bootstrap reads the SE ratio with a standard deviation of about
    ``1 / sqrt(2 R)`` = 0.016, so its 99% interval is about 0.04 wide on each side and its 80%
    interval about 0.02. The cell reports the size; no band reads it.
    """
    assert properties.DIAGNOSTIC_N == 1_000
    assert properties.DIAGNOSTIC_REPLICATES == 2_000
    assert study.DIAGNOSTIC_FAMILY not in {"interval_calibration"}
    assert study.PROPERTY_CELLS[study.DIAGNOSTIC_FAMILY] == (
        "blip__n1000",
        "weighted_classification__n1000",
    )
    draws = {(d.family, d.label): d for d in properties.PROPERTY_DRAWS}
    for loss in law.LOSSES:
        draw = draws[study.DIAGNOSTIC_FAMILY, f"{loss}__n1000"]
        assert draw.replicates == properties.DIAGNOSTIC_REPLICATES
        assert properties.size_of(draw.family) == 1_000
    assert properties.size_of("interval_calibration") == properties.CALIBRATION_N
    sd = 1.0 / np.sqrt(2.0 * properties.DIAGNOSTIC_REPLICATES)
    assert 0.015 < sd < 0.017
    assert 0.038 < tools.Z * sd < 0.043
    assert "tests/studies/evidence/property_verdicts.py" in study.STUDY.modules
    assert "src/cleverly/estimators/tmle.py" in study.STUDY.modules


def test_the_coverage_band_is_not_binding_at_the_declared_size() -> None:
    replicates = properties.CALIBRATION_REPLICATES
    low, high = study.STUDY.margins.calibration_coverage
    for coverage in PILOT_COVERAGE:
        half = tools.Z * np.sqrt(coverage * (1.0 - coverage) / replicates)
        assert low < coverage - half and coverage + half < high


def test_the_fold_locality_control_resolves_below_the_band() -> None:
    """The control's 99% coverage interval at the declared size lies far below 0.92."""
    replicates = properties.CALIBRATION_REPLICATES
    floor = study.STUDY.margins.calibration_coverage[0]
    for coverage in PILOT_CONTROL_COVERAGE:
        half = tools.Z * np.sqrt(coverage * (1.0 - coverage) / replicates)
        assert coverage + half < floor - 0.6


def test_the_primary_budget_resolves_the_framework_defaults() -> None:
    """At 2,000 replicates of n = 2,000 the primary verdicts clear their margins.

    The coverage floor 0.90, the SE sanity band 0.80 to 1.20 against the pilot ratios, and the
    bias margin 0.25 SD against the pilot's standardized bias.
    """
    replicates = study.PRIMARY_REPLICATES
    margins = study.STUDY.margins
    for coverage in PRIMARY_PILOT_COVERAGE:
        half = tools.Z * np.sqrt(coverage * (1.0 - coverage) / replicates)
        assert coverage - half > margins.coverage_floor + 0.02
    for ratio in PRIMARY_PILOT_RATIO:
        assert 0.80 + 0.1 < ratio < 1.20 - 0.1
    for bias in PRIMARY_PILOT_STANDARDIZED_BIAS:
        assert tools.equivalence_power(bias, replicates, margins.standardized_bias) > 0.999


# ------------------------------------------------------------------------------- the seeds


def _seeds(record: Any, primary: int, draws: Any, suffixes: Any) -> dict[str, set[int]]:
    """The sample seeds and the fold seeds of a declared run of ``record``."""
    out: dict[str, set[int]] = {
        "sample": {replicate_seed(record, s, r) for s in record.scenarios for r in range(primary)},
        "fold": {
            stream_seed(record, "fold_partition", "primary", s, r)
            for s in record.scenarios
            for r in range(primary)
        },
    }
    for draw in draws:
        suffix = suffixes[draw.family, draw.label]
        for kind, name in (("sample", "property_sample"), ("fold", "fold_partition")):
            out[kind] |= {
                stream_seed(record, name, draw.family, draw.label, r, *suffix)
                for r in range(draw.replicates)
            }
    return out


def test_no_dataset_or_partition_seed_is_shared_with_another_learned_rule_study() -> None:
    """This study's samples and partitions are its own, and distinct within the study.

    Both scenarios draw the one law, so a sample seed shared between them would fit one dataset
    twice; every sample seed of the study is distinct.
    """
    record = study.STUDY
    assert (record.seed, record.resampling_seed) == (20263031, 20263032)
    suffixes = properties.suffixes(record)
    assert all(suffix == () for suffix in suffixes.values())
    mine = _seeds(record, study.PRIMARY_REPLICATES, properties.PROPERTY_DRAWS, suffixes)
    size = (
        2 * study.PRIMARY_REPLICATES
        + 2 * properties.CALIBRATION_REPLICATES
        + 2 * properties.DIAGNOSTIC_REPLICATES
    )
    assert len(mine["sample"]) == len(mine["fold"]) == size
    assert not mine["sample"] & mine["fold"]
    others = [
        _seeds(
            gated.STUDY,
            gated.PRIMARY_REPLICATES,
            gated_properties.PROPERTY_DRAWS,
            gated_properties.suffixes(gated.STUDY),
        ),
        _seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {}),
        _seeds(
            compositions.STUDY,
            compositions.PRIMARY_REPLICATES,
            composition_properties.PROPERTY_DRAWS,
            composition_properties.suffixes(compositions.STUDY),
        ),
        _seeds(
            split_mean.STUDY,
            split_mean.PRIMARY_REPLICATES,
            split_mean_properties.PROPERTY_DRAWS,
            split_mean_properties.suffixes(split_mean.STUDY),
        ),
    ]
    samples = set().union(*(other["sample"] for other in others))
    folds = set().union(*(other["fold"] for other in others))

    def seed(name: str, label: str, replicate: int) -> int:
        return stream_seed(record, name, "interval_calibration", label, replicate)

    wc = "weighted_classification__declared"
    # One shared sample seed: wc replicate 732 at n = 4,000 and compositions' clustered primary
    # replicate 2,444 at n = 2,000. Both draws open with W1 from the same generator, so they share
    # 2,000 values of W1; W2, A and Y are drawn after all n values of W1 and differ.
    assert mine["sample"] & samples == {seed("property_sample", wc, 732)}
    assert seed("property_sample", wc, 732) == replicate_seed(
        compositions.STUDY, "clustered", 2_444
    )
    # Cross-role coincidences, which share no dataset: a sample seed of this study that another
    # study uses to partition, and one partition seed shared with learned-rule-split-mean.
    assert mine["sample"] & folds == {
        seed("property_sample", wc, 699),
        seed("property_sample", "blip__declared", 794),
        seed("property_sample", wc, 5_354),
    }
    assert mine["fold"] & samples == set()
    assert mine["fold"] & folds == {seed("fold_partition", wc, 6_652)}
