"""The declared constants of the ``learned-rule-intermediate`` study, recomputed without a run.

Each number here was fixed before any registered run. The law checks call no sampler: they
integrate on the RM30 grid through the study law module. The pilot readings and the sizing rules
that read them are pinned. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_intermediate_law as law
from tests.studies import learned_rule_intermediate as study
from tests.studies import learned_rule_intermediate_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_intermediate_design")

base = law.base


# ----------------------------------------------------------------------------- the law


@pytest.mark.parametrize("z", law.LEVELS)
def test_each_levels_limit_is_the_declared_one(z: float) -> None:
    """The bounds that size the noise controls, the optimal rules and their treated shares."""
    limit = law.level_limit(z)
    assert limit["bound"] == pytest.approx(study.EFFICIENCY_BOUNDS[f"learned_z{int(z)}"], abs=1e-6)
    assert limit["known_bound"] == pytest.approx(
        study.EFFICIENCY_BOUNDS[f"known_z{int(z)}"], abs=1e-6
    )
    assert limit["treated"] == pytest.approx({0.0: 0.55, 1.0: 0.25}[z], abs=2e-4)
    assert limit["truth"] == pytest.approx({0.0: 0.577115, 1.0: 0.597071}[z], abs=1e-6)
    assert limit["known_truth"] == pytest.approx({0.0: 0.576491, 1.0: 0.581904}[z], abs=1e-6)
    assert limit["disagreement"] == pytest.approx(0.30025, abs=1e-6)


def test_level_zero_is_rm30s_value() -> None:
    """At ``z = 0`` the outcome regression is RM30's, so the optimal value is RM30's truth."""
    from tests.studies import _learned_rule_optimal_law as optimal

    assert law.level_limit(0.0)["truth"] == pytest.approx(
        optimal.law_limit("non_exceptional")["truth"], abs=1e-12
    )


def test_the_known_rule_misses_the_level_one_optimum() -> None:
    """``1{W1 > 0}`` is near the level-0 optimum and far from the level-1 one, so the known
    cell is not the learned cell at either level."""
    for z, gap in ((0.0, 0.0006), (1.0, 0.0152)):
        limit = law.level_limit(z)
        assert limit["truth"] - limit["known_truth"] == pytest.approx(gap, abs=1e-4)


def test_positivity_holds() -> None:
    assert law.smallest_weight_product() == pytest.approx(0.0450, abs=1e-4)


@pytest.mark.parametrize(("z", "bias", "units"), [(0.0, 0.098305, 3.27), (1.0, -0.025308, -1.17)])
def test_the_drop_w1_plug_in_is_biased_at_each_level(z: float, bias: float, units: float) -> None:
    """The necessity learner's limit: its untargeted plug-in misses its own rule's value by
    ``bias``, ``units`` SEs of that rule's curve at n = 2,000."""
    limit = law.drop_w1_limit(z)
    assert limit["bias"] == pytest.approx(bias, abs=1e-6)
    assert limit["bias"] / (limit["bound"] / np.sqrt(study.PRIMARY_N)) == pytest.approx(
        units, abs=0.01
    )
    assert limit["treated"] == {0.0: 1.0, 1.0: 0.0}[z]


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.SCENARIOS == {
        "cde_z0": ("ey_learned_rule[learned rule]",),
        "cde_z1": ("ey_learned_rule[learned rule]",),
    }
    assert study.STUDY.scenario_seed_owners == {"cde_z1": "cde_z0"}
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "learned_z0__declared",
            "learned_z0__shrunken_se_control",
            "learned_z0__noise_control",
            "learned_z0__term_dropped_control",
            "learned_z1__declared",
            "learned_z1__shrunken_se_control",
            "learned_z1__noise_control",
            "learned_z1__term_dropped_control",
            "known_z0__declared",
            "known_z0__shrunken_se_control",
            "known_z0__noise_control",
            "known_z1__declared",
            "known_z1__shrunken_se_control",
            "known_z1__noise_control",
        ),
        "targeting_necessity": (
            "drop_w1_z0__targeted",
            "drop_w1_z0__untargeted",
            "drop_w1_z1__targeted",
            "drop_w1_z1__untargeted",
        ),
    }
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert properties.TARGETING_DISPLACEMENT == 0.5
    assert study.STUDY.truth_varies_by_replicate
    assert not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == study.STUDY.n == 2_000
    assert properties.DRAW_SETTINGS == {
        "learned": ("declared", False),
        "known": ("declared", True),
        "drop_w1": ("drop_w1", False),
    }


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/direct_effect.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/_support.py",
        "tests/studies/_learned_rule_intermediate_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/learned_rule_intermediate_properties.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))


def test_the_support_rows_are_declared_before_the_run() -> None:
    """Two rows: the learned rule (P11) and the known regime (P5), each fold-evaluated with an
    intermediate; nothing else."""
    assert study.SUPPORT_ROWS == {
        "learned": {
            "splitting": ("fold_evaluated",),
            "families": ("P11",),
            "missing": ("intermediate",),
        },
        "known": {
            "splitting": ("fold_evaluated",),
            "families": ("P5",),
            "missing": ("intermediate",),
        },
    }


def test_the_declared_gates_are_read() -> None:
    """Every verdict family has a rule; the invalid-inference control is a declared SE control
    kind, read below the band; the exactness gate is the harness's."""
    from tests.studies.evidence import property_verdicts

    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == set(study.PROPERTY_CELLS)
    assert property_verdicts.SE_CONTROL_SIDES[study.TERM_DROPPED] == "below"
    assert law.EXACTNESS_TOLERANCE == 1e-8
    assert properties.TARGETING_NECESSITY.labels == study.NECESSITY_LABELS


def test_the_docstrings_quote_the_law() -> None:
    doc = " ".join((study.__doc__ or "").split())
    for quoted in ("30%", "n = 2,000", "0.098", "-0.025", "0.2 + 0.6 A - 1.2 W1"):
        assert quoted in doc, quoted
    law_doc = " ".join((law.__doc__ or "").split())
    for quoted in ("55% treated", "(25%)", "30%", "0.045"):
        assert quoted in law_doc, quoted
    assert f"{law.level_limit(0.0)['treated']:.0%}" == "55%"
    assert f"{law.level_limit(1.0)['treated']:.0%}" == "25%"
    assert f"{law.level_limit(0.0)['disagreement']:.0%}" == "30%"


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def _x11_modules() -> list[tuple[Any, Any]]:
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_arms as h
    from tests.studies import learned_rule_arms_properties as h_properties
    from tests.studies import learned_rule_compositions as compositions
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_contrast as g
    from tests.studies import learned_rule_contrast_properties as g_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_properties as gated_properties
    from tests.studies import learned_rule_in_sample as f2
    from tests.studies import learned_rule_in_sample_properties as f2_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_optimal_value as i
    from tests.studies import learned_rule_optimal_value_properties as i_properties
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_split_mean_properties as split_mean_properties
    from tests.studies import learned_rule_stacked as f1
    from tests.studies import learned_rule_stacked_properties as f1_properties
    from tests.studies import learned_rule_strata as e2
    from tests.studies import learned_rule_strata_properties as e2_properties

    return [
        (gated, gated_properties),
        (compositions, composition_properties),
        (split_mean, split_mean_properties),
        (learners, learner_properties),
        (e1, e1_properties),
        (e2, e2_properties),
        (f1, f1_properties),
        (f2, f2_properties),
        (g, g_properties),
        (h, h_properties),
        (i, i_properties),
    ]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_cvtmle_boundary as boundary

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module, _ in [*_x11_modules(), (boundary, None)]:
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others


def test_the_named_streams_are_distinct_from_every_registered_study() -> None:
    labels = [
        ("root_n_rate", "empirical_sd"),
        ("interval_calibration", "correctly_specified"),
        ("performance", "cleverly", "binary", "ate"),
        ("equivalence", "binary", "ate"),
    ]
    records = {record.slug: record for record in _other_studies()}
    records.update({module.STUDY.slug: module.STUDY for module, _ in _x11_modules()})
    records[study.STUDY.slug] = study.STUDY
    seeds = [stream_seed(record, *label) for record in records.values() for label in labels]
    assert len(set(seeds)) == len(seeds)


#: The suffix rule L3 adds to each property draw's labels: one retry for the necessity draw,
#: whose first stream seed meets a registered study's.
DECLARED_SUFFIXES = {
    ("interval_calibration", "learned"): (),
    ("interval_calibration", "known"): (),
    ("targeting_necessity", "drop_w1"): ("retry", 1),
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


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import learned_rule_cvtmle_boundary as boundary

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for record, module in _x11_modules():
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


def test_the_sample_seeds_are_distinct() -> None:
    """The two primary scenarios draw one sample stream (``cde_z1`` is owned by ``cde_z0``)."""
    assert replicate_seed(study.STUDY, "cde_z1", 7) == replicate_seed(study.STUDY, "cde_z0", 7)
    total = study.PRIMARY_REPLICATES + sum(draw.replicates for draw in properties.PROPERTY_DRAWS)
    assert total == TOTAL_SEEDS
    assert len(_own_sample_seeds()) == total


def test_no_sample_seed_is_another_x11_studys() -> None:
    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == set()
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, study.PRIMARY_SCENARIO, 0) in declared


def test_no_pilot_seed_is_a_declared_seed() -> None:
    """The pilot's throwaway label draws no declared sample seed, and it is no record or
    resampling seed."""
    declared = learned_rule_run.declared_sample_seeds()
    records = {record.seed for record in registered()} | {
        record.resampling_seed for record in registered() if record.resampling_seed is not None
    }
    for label, cells in PILOT_LABELS.items():
        assert label not in records and label not in (study.SEED, study.RESAMPLING_SEED)
        drawn = {
            int(np.random.SeedSequence([label, k, i]).generate_state(2)[0])
            for k, draws in cells
            for i in range(draws)
        }
        assert not drawn & declared, label


# ---------------------------------------------------------------------------- the sizing

#: The study's floor of draws for a cell, as a multiple of 400 (plan section 11).
FLOOR = 1_200

#: The pilot labels and the (cell index, draws) each drew (``reviews/x11/scripts/
#: X11-J-calibration-pilot.py`` in the plans repository): learned, known, drop_w1, then the probe
#: of the learned cell alone.
PILOT_LABELS = {
    20270811: ((0, 4_000), (1, 4_000), (2, 4_000)),
    20270823: ((0, 4_000),),
}

PILOT_DRAWS = 4_000

#: The pilot readings of each cell and level (no failed fit, no solver warning, rule L4 and the
#: exactness gate on every row): SE ratio, standardized bias and coverage against each row's own
#: truth.
PILOT: dict[str, dict[str, float]] = {
    "learned_z0": {"ratio": 0.977819, "bias": -0.015886, "coverage": 0.94625},
    "learned_z1": {"ratio": 0.958915, "bias": -0.004718, "coverage": 0.936},
    "known_z0": {"ratio": 1.022361, "bias": 0.005434, "coverage": 0.952},
    "known_z1": {"ratio": 1.007526, "bias": -0.007243, "coverage": 0.94775},
}

#: The learned cell on the 8,000 draws of the pilot and the probe pooled.
POOLED: dict[str, dict[str, float]] = {
    "learned_z0": {"ratio": 0.981517, "bias": -0.018756, "coverage": 0.947625},
    "learned_z1": {"ratio": 0.980528, "bias": -0.006695, "coverage": 0.94425},
}

POOLED_DRAWS = 8_000

#: The controls on the pilot, (SE ratio, coverage): the shrunken SE (x0.80), the noise (one bound
#: unit) and, for the learned labels, the curve without ``1{Z = z} / q``.
PILOT_CONTROLS: dict[str, dict[str, tuple[float, float]]] = {
    "learned_z0": {
        "shrunken": (0.782255, 0.87125),
        "noise": (0.707024, 0.83525),
        "term_dropped": (0.578638, 0.7405),
    },
    "learned_z1": {
        "shrunken": (0.767132, 0.86575),
        "noise": (0.694146, 0.82625),
        "term_dropped": (0.712463, 0.837),
    },
    "known_z0": {"shrunken": (0.817889, 0.892), "noise": (0.720621, 0.84425)},
    "known_z1": {"shrunken": (0.806021, 0.88275), "noise": (0.712760, 0.83675)},
}

#: The necessity pilot at each level: the targeted positive's standardized bias, the untargeted
#: control's standardized bias in its own SD, and the signed displacement in positive-arm SDs.
PILOT_NECESSITY: dict[str, dict[str, float]] = {
    "drop_w1_z0": {"targeted": 0.035613, "untargeted": 3.907520, "displacement": 3.042413},
    "drop_w1_z1": {"targeted": 0.025283, "untargeted": -1.223602, "displacement": -1.159039},
}


def bias_pass(
    bias: float, replicates: int, pilot: int = PILOT_DRAWS, margin: float = 0.25
) -> float:
    """The predictive probability that the 99% interval of the mean standardized error falls
    inside ``±margin`` at ``replicates`` draws, given a pilot bias read from ``pilot`` draws
    (the F2 rule)."""
    half = norm.ppf(0.995) / np.sqrt(replicates)
    spread = np.sqrt(1.0 / replicates + 1.0 / pilot)
    return float(
        norm.cdf((margin - half - bias) / spread) - norm.cdf((-margin + half - bias) / spread)
    )


def discrimination_pass(
    bias: float, replicates: int, pilot: int = PILOT_DRAWS, margin: float = 0.25
) -> float:
    """The predictive probability that the 99% interval of a control's standardized bias clears
    ``±margin``, given its pilot bias."""
    half = norm.ppf(0.995) / np.sqrt(replicates)
    spread = np.sqrt(1.0 / replicates + 1.0 / pilot)
    return float(norm.cdf((abs(bias) - margin - half) / spread))


_Z = np.linspace(-6.0, 6.0, 241)
_W = norm.pdf(_Z) / norm.pdf(_Z).sum()


def _predicted(coverage: float, replicates: int, pilot: int | None = PILOT_DRAWS) -> np.ndarray:
    extra = 0.0 if pilot is None else 1 / pilot
    spread = np.sqrt(max(coverage * (1 - coverage), 1e-12) * (1 / replicates + extra))
    grid = np.clip(coverage + spread * _Z, 0.0, 1.0)
    return np.rint(grid * replicates).astype(int)


def floor_pass(
    coverage: float, replicates: int, floor: float, pilot: int | None = PILOT_DRAWS
) -> float:
    """The predictive probability that the exact 99% interval's lower bound is at or above
    ``floor``."""
    hits = _predicted(coverage, replicates, pilot)
    lower = np.where(hits > 0, beta.ppf(0.005, hits, replicates - hits + 1), 0.0)
    return float(np.sum(_W * (lower >= floor)))


def ceiling_pass(
    coverage: float, replicates: int, ceiling: float, pilot: int | None = PILOT_DRAWS
) -> float:
    """The predictive probability that the exact 99% interval's upper bound is below
    ``ceiling``."""
    hits = _predicted(coverage, replicates, pilot)
    upper = np.where(hits < replicates, beta.ppf(0.995, hits + 1, replicates - hits), 1.0)
    return float(np.sum(_W * (upper < ceiling)))


def _smallest(passes: Callable[[int], float], target: float = tools.FAMILY_TARGET) -> int:
    for replicates in range(400, 40_001, 400):
        if replicates >= FLOOR and passes(replicates) >= target:
            return replicates
    raise AssertionError("no count reaches the target")


def primary_pass(
    replicates: int,
    readings: Mapping[str, Mapping[str, float]] | None = None,
    pilot: int = PILOT_DRAWS,
) -> float:
    """Both levels' bias legs and coverage floors (0.90)."""
    readings = PILOT if readings is None else readings
    floor = study.STUDY.margins.coverage_floor
    out = 1.0
    for z in law.LEVELS:
        reading = readings[f"learned_z{int(z)}"]
        out *= bias_pass(reading["bias"], replicates, pilot) * floor_pass(
            reading["coverage"], replicates, floor, pilot
        )
    return out


def calibration_pass(
    replicates: int, reading: Mapping[str, float], pilot: int | None = PILOT_DRAWS
) -> float:
    """One positive's SE-ratio leg (the per-arm model) times its coverage leg (the exact interval
    inside ``[0.92, 0.98]``). ``pilot=None`` reads both as known (the conditional stress test)."""
    low, high = study.STUDY.margins.calibration_coverage
    if pilot is None:
        ratio = float(tools.cell_pass(np.array([reading["ratio"]]), replicates, tools.C_FIT)[0])
    else:
        ratio = tools.per_arm_pass([reading["ratio"]], pilot, replicates)[0]
    coverage = reading["coverage"]
    inside = floor_pass(coverage, replicates, low, pilot) - (
        1.0 - ceiling_pass(coverage, replicates, high, pilot)
    )
    return ratio * inside


def draw_pass(
    labels: tuple[str, ...],
    replicates: int,
    readings: Mapping[str, Mapping[str, float]] | None = None,
    pilot: int | None = PILOT_DRAWS,
) -> float:
    """The product over one draw's two levels."""
    readings = PILOT if readings is None else readings
    return float(
        np.prod([calibration_pass(replicates, readings[label], pilot) for label in labels])
    )


def necessity_pass(replicates: int) -> float:
    """Both positives' bias legs, both controls' discrimination legs, and the smaller absolute
    displacement (``paired_displacement``) at or above the threshold, read as normal with the
    pilot's SE."""
    out = 1.0
    for z in law.LEVELS:
        reading = PILOT_NECESSITY[f"drop_w1_z{int(z)}"]
        out *= bias_pass(reading["targeted"], replicates)
        out *= discrimination_pass(reading["untargeted"], replicates)
    smallest = min(abs(reading["displacement"]) for reading in PILOT_NECESSITY.values())
    spread = np.sqrt(1.0 / replicates + 1.0 / PILOT_DRAWS)
    return out * float(norm.cdf((smallest - properties.TARGETING_DISPLACEMENT) / spread))


def arm_pooled(
    readings: Mapping[str, Mapping[str, float]], labels: tuple[str, ...], pilot: int
) -> dict[str, Any]:
    """The arm-pooled rule over one draw's two levels."""
    return tools.replicates_from_pilot([readings[label]["ratio"] for label in labels], pilot)


def control_pass(ratio: float, replicates: int, pilot: int = PILOT_DRAWS) -> float:
    """The predictive probability that a control's 99% SE-ratio interval lies wholly below the
    band, given its pilot ratio."""
    run = tools.pilot_sigma(replicates)
    spread = np.hypot(tools.pilot_sigma(pilot), run)
    low = study.STUDY.margins.calibration_se_ratio[0]
    return float(norm.cdf((low - norm.ppf(0.995) * run - ratio) / spread))


LEARNED = ("learned_z0", "learned_z1")
KNOWN = ("known_z0", "known_z1")

#: Core-seconds per fit on the pilot (truth and harness included, four jobs on shared cores).
SECONDS = {"learned": 0.42, "known": 0.207, "drop_w1": 0.485}

#: The run-time budget: about an hour at four jobs.
BUDGET = 4 * 3_600


def test_the_primary_count() -> None:
    assert study.PRIMARY_REPLICATES == _smallest(primary_pass) == 1_200
    assert _smallest(lambda r: primary_pass(r, POOLED, POOLED_DRAWS)) == 1_200
    assert primary_pass(1_200) == pytest.approx(0.957, abs=1e-3)


def test_the_learned_count_falls_back_to_the_budget() -> None:
    """The per-cell rule has no count on the pilot and 7,200 on the pooled draws; the arm-pooled
    rule has none on either, so the declared fallback sets the run-time budget."""
    with pytest.raises(AssertionError, match="no count"):
        _smallest(lambda r: draw_pass(LEARNED, r), 0.975)
    assert _smallest(lambda r: draw_pass(LEARNED, r, POOLED, POOLED_DRAWS), 0.975) == 7_200
    first = arm_pooled(PILOT, LEARNED, PILOT_DRAWS)
    assert first["screened"] == [True, False] and first["R"] is None
    pooled = arm_pooled(POOLED, LEARNED, POOLED_DRAWS)
    assert pooled["R"] is None and pooled["family_wise"] == pytest.approx(0.940, abs=1e-3)
    assert properties.LEARNED_REPLICATES == 20_000
    assert draw_pass(LEARNED, 20_000, POOLED, POOLED_DRAWS) > 0.999


def test_the_known_count_falls_back_to_the_budget() -> None:
    assert _smallest(lambda r: draw_pass(KNOWN, r), 0.975) == 10_000
    assert tools.per_arm_replicates([PILOT[label]["ratio"] for label in KNOWN], 4_000)["R"] == 7_600
    pooled = arm_pooled(PILOT, KNOWN, PILOT_DRAWS)
    assert pooled["R"] is None and pooled["family_wise"] == pytest.approx(0.698, abs=1e-3)
    assert properties.KNOWN_REPLICATES == properties.LEARNED_REPLICATES
    assert draw_pass(KNOWN, properties.KNOWN_REPLICATES) > 0.99


def test_the_budget_holds_the_run() -> None:
    """The two calibration draws at one count, the largest multiple of 2,000 inside the budget."""
    fixed = (study.PRIMARY_REPLICATES * SECONDS["learned"]) + (
        properties.NECESSITY_REPLICATES * SECONDS["drop_w1"]
    )
    per_draw = SECONDS["learned"] + SECONDS["known"]

    def cost(replicates: int) -> float:
        return fixed + replicates * per_draw

    assert (
        cost(properties.LEARNED_REPLICATES) <= BUDGET < cost(properties.LEARNED_REPLICATES + 2_000)
    )


def test_the_stress_test() -> None:
    """Conditional on the readings as known, without a second pilot error."""
    assert draw_pass(LEARNED, properties.LEARNED_REPLICATES, PILOT, None) > 0.99
    assert draw_pass(KNOWN, properties.KNOWN_REPLICATES, PILOT, None) > 0.99


def test_the_controls_fail_at_the_declared_counts() -> None:
    counts = {"learned": properties.LEARNED_REPLICATES, "known": properties.KNOWN_REPLICATES}
    for label, controls in PILOT_CONTROLS.items():
        for ratio, coverage in controls.values():
            assert control_pass(ratio, counts[label.split("_")[0]]) >= 0.999, label
            assert ratio < study.STUDY.margins.calibration_se_ratio[0]
            assert coverage < study.STUDY.margins.calibration_coverage[0]
    assert set(PILOT_CONTROLS["learned_z0"]) == {"shrunken", "noise", "term_dropped"}


def test_the_necessity_count() -> None:
    assert properties.NECESSITY_REPLICATES == _smallest(necessity_pass) == 1_200
    # The signed displacement at z = 1 is negative: drop_w1 treats nobody there, so the plug-in
    # sits below its truth. The verdict reads the absolute displacement.
    assert PILOT_NECESSITY["drop_w1_z1"]["displacement"] < -properties.TARGETING_DISPLACEMENT


#: The sample seeds a declared run draws.
TOTAL_SEEDS = 42_400
