"""The declared constants of the ``learned-rule-split-mean`` study, recomputed or pinned.

Each number here was fixed before any registered run. The efficiency bounds are recomputed by
quadrature from the population projection of the declared learner on the RM30 grid; the pilot
ratios and the sizing rule that reads them are pinned; and the study's seeds are checked against
every other learned-rule study. No check draws a sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_split_mean_law as law
from tests.studies import learned_rule_compositions as compositions
from tests.studies import learned_rule_compositions_properties as composition_properties
from tests.studies import learned_rule_cvtmle as gated
from tests.studies import learned_rule_cvtmle_boundary as boundary
from tests.studies import learned_rule_cvtmle_properties as gated_properties
from tests.studies import learned_rule_split_mean as study
from tests.studies import learned_rule_split_mean_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.seeds import replicate_seed, stream_seed


@pytest.fixture(scope="module")
def rules() -> dict[str, dict[float, np.ndarray]]:
    return {name: law.population_rule(name) for name in law.LAWS}


def test_the_efficiency_bounds(rules: dict[str, dict[float, np.ndarray]]) -> None:
    for name, rule in rules.items():
        _, variance = base.rule_value(rule, name)
        assert np.sqrt(variance) == pytest.approx(study.EFFICIENCY_BOUNDS[name], abs=2e-6)


def test_the_limit_rules_are_the_rm30_ones(rules: dict[str, dict[float, np.ndarray]]) -> None:
    """The thresholds of RM30's independent constants table (``d0`` and ``d1``), to one grid step.

    ``non_exceptional`` is correctly specified, so its limit rule is ``d0 = 1{W1 > -0.1}``.
    ``misspecified_limit`` has the learner's limit thresholds 0.005025 at ``W2 = 0`` and
    0.001387 at ``W2 = 1`` (``test_learned_rule_study_constants.py``).
    """
    step = base.GRID[1] - base.GRID[0]
    thresholds = {
        "non_exceptional": {0.0: -0.1, 1.0: -0.1},
        "misspecified_limit": {0.0: 0.005025, 1.0: 0.001387},
    }
    for name, rule in rules.items():
        for w2, threshold in thresholds[name].items():
            first = base.GRID[np.argmax(rule[w2])]
            assert rule[w2][-1] and not rule[w2][0]
            assert np.all(rule[w2] == (first <= base.GRID))
            assert abs(first - threshold) <= 1.01 * step, (name, w2)


#: The calibration pilot: 4,000 draws of each calibration positive at n = 2,000 with
#: ``repeats=5``, from ``SeedSequence([20263199, cell, draw])``, a label no declared stream
#: reads. Each value is the mean reported SE over the SD of the error ``estimate - truth``, in
#: the order of :data:`law.LAWS`. No draw failed, no rule-L4 refit disagreed on any of the
#: 2 x 4,000 x 5 x 2,000 checked rows, and no solver warned. The script is
#: ``reviews/x11/scripts/X11-B-calibration-pilot.py`` in the plans repository.
CALIBRATION_PILOT = (1.0196, 0.9983)
CALIBRATION_PILOT_DRAWS = 4_000

#: The pilot's coverage of the positives, and its SE ratio and coverage of the
#: independent-splits control (the mean single-split SE over ``sqrt(5)``), in the same order.
PILOT_COVERAGE = (0.9555, 0.94775)
PILOT_INDEPENDENT_RATIO = (0.4617, 0.4535)
PILOT_INDEPENDENT_COVERAGE = (0.62225, 0.62775)

#: The pilot's standardized bias of the error, for the primary phase's bias equivalence.
PILOT_STANDARDIZED_BIAS = (-0.0197, 0.0159)


def test_the_calibration_replicates_follow_the_per_arm_rule() -> None:
    """R is the smallest multiple of 400 whose family-wise pass probability reaches 0.95.

    The two positives are two laws, one correctly specified and one whose learner has a
    misspecified limit, so each is predicted from its own pilot ratio. The arm-pooled rule of
    ``mtp_study_tools`` reads their spread (1.0196 against 0.9983) as between-arm noise, flags it
    as suspect and accepts no R up to 40,000.
    """
    rule = tools.per_arm_replicates(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert rule["R"] == properties.CALIBRATION_REPLICATES == 6_400
    assert min(rule["per_cell"]) >= tools.CELL_TARGET
    assert rule["family_wise"] >= tools.FAMILY_TARGET
    below = tools.per_arm_pass(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS, 6_000)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    pooled = tools.replicates_from_pilot(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert pooled["R"] is None


def test_the_coverage_band_is_not_binding_at_the_declared_size() -> None:
    """The 99% exact half-width at 6,400 is about 0.007, and each pilot coverage clears it."""
    replicates = properties.CALIBRATION_REPLICATES
    low, high = study.STUDY.margins.calibration_coverage
    for coverage in PILOT_COVERAGE:
        half = tools.Z * np.sqrt(coverage * (1.0 - coverage) / replicates)
        assert low < coverage - half and coverage + half < high


def test_the_independent_splits_control_resolves_below_the_band() -> None:
    """The control's ratio sits near ``1/sqrt(5)``, far below 0.93 at any budget here.

    Its upper 99% end at 6,400 replicates, with the sizing rule's spread constant, is below
    0.5, so the control cannot pass by luck and a helper that returned the fit's own SE would
    fail it.
    """
    replicates = properties.CALIBRATION_REPLICATES
    band = study.STUDY.margins.calibration_se_ratio
    for ratio in PILOT_INDEPENDENT_RATIO:
        assert abs(ratio - 1.0 / np.sqrt(law.SPLITS)) < 0.02
        upper = ratio * (1.0 + tools.Z * tools.C_FIT / np.sqrt(2 * replicates))
        assert upper < 0.5 < band[0]
    for coverage in PILOT_INDEPENDENT_COVERAGE:
        assert coverage < study.STUDY.margins.calibration_coverage[0] - 0.25


def test_the_primary_budget_resolves_the_framework_defaults() -> None:
    """At 2,000 replicates the primary verdicts clear their margins at the pilot's values.

    The coverage floor 0.90 against the lower 99% end of each pilot coverage, and the bias
    margin 0.25 SD against the pilot's standardized bias plus the 99% half-width.
    """
    replicates = study.PRIMARY_REPLICATES
    margins = study.STUDY.margins
    for coverage in PILOT_COVERAGE:
        half = tools.Z * np.sqrt(coverage * (1.0 - coverage) / replicates)
        assert coverage - half > margins.coverage_floor + 0.02
    for bias in PILOT_STANDARDIZED_BIAS:
        assert tools.equivalence_power(bias, replicates, margins.standardized_bias) > 0.999


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

    The seeds are 32-bit, so about two coincidences across roles are expected among some
    250,000 seeds. One occurs: the fold seed of ``non_exceptional__declared`` replicate 4,780
    equals the RM30 ``n_8000`` property sample seed of replicate 1,370. It partitions a
    different sample of n = 2,000 and draws no data, so no dataset is shared. The pin makes a
    new coincidence visible.
    """
    record = study.STUDY
    assert (record.seed, record.resampling_seed) == (20263021, 20263022)
    suffixes = properties.suffixes(record)
    assert all(suffix == () for suffix in suffixes.values())
    mine = _seeds(record, study.PRIMARY_REPLICATES, properties.PROPERTY_DRAWS, suffixes)
    size = 2 * study.PRIMARY_REPLICATES + 2 * properties.CALIBRATION_REPLICATES
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
    ]
    samples = set().union(*(other["sample"] for other in others))
    folds = set().union(*(other["fold"] for other in others))
    assert not mine["sample"] & samples
    assert not mine["fold"] & folds
    assert not mine["sample"] & folds
    assert mine["fold"] & samples == {
        stream_seed(
            record, "fold_partition", "interval_calibration", "non_exceptional__declared", 4_780
        )
    }
    assert (
        stream_seed(gated.STUDY, "property_sample", "root_n_and_efficiency", "n_8000", 1_370)
        in mine["fold"]
    )
