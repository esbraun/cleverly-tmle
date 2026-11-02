"""The declared constants of the ``learned-rule-optimal-value`` study, recomputed without a run.

Each number here was fixed before any registered run. The law checks call no sampler except the
``n = 10^6`` cross-check of the wrong-limit learner's limit: they solve the population score
equations of each law's learner and integrate on the RM30 grid, through the study law module that
W-f2 (``tests/unit/test_learned_rule_optimal.py``) reads too. The pilot readings and the sizing
rules that read them are pinned. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_optimal_law as law
from tests.studies import learned_rule_optimal_value as study
from tests.studies import learned_rule_optimal_value_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_optimal_value_design")

base = law.base


# ----------------------------------------------------------------------------- the laws


@pytest.mark.parametrize("scenario", list(study.LIMITS))
def test_each_laws_limit_is_the_declared_one(scenario: str) -> None:
    """Impl review answer 1: the truth, the bound, the shares and the gate's readings."""
    found = law.law_limit(scenario)
    declared = study.LIMITS[scenario]
    assert found["truth"] == pytest.approx(declared["truth"], abs=1e-6)
    assert np.sqrt(found["variance"]) == pytest.approx(declared["bound"], abs=1e-6)
    for key in ("treated", "tie_mass", "disagreement", "limit_value"):
        assert found[key] == pytest.approx(declared[key], abs=1e-6), key
    assert found["margin"] == pytest.approx(declared["margin"], abs=1e-3)


def test_only_the_non_exceptional_law_gates() -> None:
    """The gate reads C7, C3f and C8's margin. Each other law misses exactly one, which is the
    deliberate mutation: ``wrong_limit`` fails C3f and the hinge law fails C7."""
    assert study.gated("non_exceptional")
    assert not study.gated("wrong_limit") and not study.gated("partially_exceptional")
    wrong, hinge = study.LIMITS["wrong_limit"], study.LIMITS["partially_exceptional"]
    assert wrong["tie_mass"] == 0.0 and wrong["margin"] <= study.MAX_MARGIN
    assert wrong["disagreement"] > study.MAX_DISAGREEMENT
    assert hinge["disagreement"] <= study.MAX_DISAGREEMENT and hinge["margin"] <= study.MAX_MARGIN
    assert hinge["tie_mass"] == pytest.approx(0.6, abs=1e-3)
    for limit in study.LIMITS.values():
        assert limit["margin"] <= study.MAX_MARGIN


def test_the_efficiency_bound_is_rm30s() -> None:
    """The bound is RM30's (G's ``TAG_BOUNDS["value"]``), and the study law's quadrature is
    RM30's ``rule_value`` on the same rule."""
    from tests.studies import learned_rule_contrast as g

    assert study.EFFICIENCY_BOUND == g.TAG_BOUNDS["value"] == 0.664438
    d0 = law.optimal_rule(law.law_qbar("non_exceptional"))
    assert law.rule_moments(d0, law.law_qbar("non_exceptional")) == pytest.approx(
        base.rule_value(d0, "non_exceptional"), rel=1e-14
    )
    assert study.EFFICIENCY_BOUNDS == {study.CALIBRATION_LABEL: study.EFFICIENCY_BOUND}


def test_the_control_n_is_the_smallest_multiple_of_500_with_three_ses() -> None:
    """W-f2's gap and the control's n: 3 SEs need 2,085 rows, 2,000 gives 2.94, and the declared
    2,500 gives 3.29. The control's expected coverage of ``Psi_d0`` is 0.0925."""
    gap, variance, share = law.target_gap(law.WRONG_LIMIT_AMPLITUDE)
    assert share == 0.0
    assert gap == pytest.approx(0.045242, abs=1e-6)
    assert variance == pytest.approx(0.474126, abs=1e-6)
    needed = 9.0 * variance / gap**2
    assert needed == pytest.approx(2_085, abs=1)
    assert study.CONTROL_N == 500 * int(np.ceil(needed / 500)) == 2_500
    z = {n: gap / np.sqrt(variance / n) for n in (2_000, study.CONTROL_N)}
    assert z[2_000] == pytest.approx(2.94, abs=0.005) and z[2_000] < 3.0
    assert z[study.CONTROL_N] == pytest.approx(3.29, abs=0.005)
    coverage = norm.cdf(base.CRITICAL - z[2_500]) - norm.cdf(-base.CRITICAL - z[2_500])
    assert coverage == pytest.approx(0.0925, abs=1e-4)
    # The learner's limit value is the limit rule's: d1 treats nobody.
    assert study.LIMITS["wrong_limit"]["truth"] - study.LIMITS["wrong_limit"]["limit_value"] == (
        pytest.approx(gap, abs=1e-6)
    )


def test_the_wrong_limit_matches_a_fit_at_a_million_rows() -> None:
    """The population score equations against the declared learner fitted to 10^6 rows (a
    throwaway seed): its blip is negative at every grid point, so its rule treats nobody, and it
    agrees with the quadrature limit within 0.01."""
    frame = law.draw("wrong_limit", 1_000_000, 99_011)
    model = base.outcome_learner().fit(
        frame[["A", "W1", "W2"]].to_numpy(), frame["Y"].to_numpy().astype(int)
    )
    limit = law.limit_blip(law.law_qbar("wrong_limit"))
    for w2 in base.W2_LEVELS:
        covariates = np.column_stack([base.GRID, np.full_like(base.GRID, w2)])
        fitted = base._blip(model, covariates)
        assert (fitted < 0.0).all()
        np.testing.assert_allclose(fitted, limit[w2], atol=0.01)


def test_the_hinge_law_has_no_regret_off_the_tie_set() -> None:
    """On the hinge law any rule that treats ``W1 > 0.2`` has the optimal value, so the tie set
    changes no value. The mutation: a rule that also leaves ``(0.2, 0.3]`` untreated loses value.
    """
    qbar = law.law_qbar("partially_exceptional")
    truth = law.law_limit("partially_exceptional")["truth"]
    everyone_above = dict.fromkeys(base.W2_LEVELS, base.GRID > -0.5)
    assert law.rule_moments(everyone_above, qbar)[0] == pytest.approx(truth, abs=1e-12)
    late = dict.fromkeys(base.W2_LEVELS, base.GRID > 0.3)
    assert truth - law.rule_moments(late, qbar)[0] > 1e-4


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "non_exceptional_known_g__declared",
            "non_exceptional_known_g__shrunken_se_control",
            "non_exceptional_known_g__noise_control",
        ),
        "optimal_target_control": ("wrong_limit_known_g__target_gap_control",),
        "estimated_g_report": ("non_exceptional_estimated_g__estimated_g",),
        "exceptional_law_report": ("partially_exceptional_known_g__exceptional",),
    }
    assert study.SCENARIOS == {"non_exceptional": ("ey_optimal_rule[learned rule]",)}
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert not study.STUDY.truth_varies_by_replicate
    assert not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == study.STUDY.n == 2_000
    assert properties.SETTINGS == {
        "interval_calibration": ("non_exceptional", True, 2_000, "declared", "positive"),
        "optimal_target_control": ("wrong_limit", True, 2_500, "target_gap_control", "control"),
        "estimated_g_report": ("non_exceptional", False, 2_000, "estimated_g", "diagnostic"),
        "exceptional_law_report": (
            "partially_exceptional",
            True,
            2_000,
            "exceptional",
            "diagnostic",
        ),
    }
    assert law.LEARNERS["partially_exceptional"] is law.hinge_learner
    assert law.LEARNERS["non_exceptional"] is law.LEARNERS["wrong_limit"] is base.outcome_learner
    assert (law.WRONG_LIMIT_AMPLITUDE, law.HINGE_AMPLITUDE, law.HINGE_KNOT) == (3.0, 1.0, 0.2)


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/targets/builtin.py",
        "src/cleverly/_support.py",
        "tests/studies/_learned_rule_optimal_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/learned_rule_optimal_value_properties.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))


def test_the_support_row_declares_p12_with_the_known_g_qualifier() -> None:
    """Impl review answer 6: one row, family P12, known g only; estimated g is held."""
    assert study.SUPPORT_ROWS == {
        "fold_evaluated": {
            "splitting": ("fold_evaluated",),
            "families": ("P12",),
            "qualifiers": ("known_mechanism",),
        },
    }


def test_the_declared_gates_are_read() -> None:
    """Every verdict family has a rule, the diagnostics have none, and the control's threshold
    is the calibration floor the verdict function reads."""
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == {"interval_calibration", "optimal_target_control"}
    assert not families & set(study.DIAGNOSTIC_FAMILIES)
    assert study.STUDY.margins.calibration_coverage[0] == 0.92


def test_the_docstrings_quote_the_law() -> None:
    """The hashed docstrings say what the cells measure (the F2 lesson)."""
    doc = " ".join((study.__doc__ or "").split())
    for quoted in ("29.3%", "0.0452", "3.29", "2,085", "0.6", "2,500", "2,000"):
        assert quoted in doc, quoted
    assert f"{study.LIMITS['wrong_limit']['treated']:.1%}" == "29.3%"
    gap, _, _ = law.target_gap(law.WRONG_LIMIT_AMPLITUDE)
    assert f"{gap:.4f}" == "0.0452"
    law_doc = " ".join((law.__doc__ or "").split())
    for quoted in ("0.0452", "3.29", "0.707", "mass 0.6"):
        assert quoted in law_doc, quoted
    properties_doc = " ".join((properties.__doc__ or "").split())
    assert "n = 2,500" in properties_doc and "n = 2,000" in properties_doc


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
    from tests.studies import learned_rule_intermediate as j
    from tests.studies import learned_rule_intermediate_properties as j_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
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
        (j, j_properties),
    ]


def test_the_record_seeds_are_unused() -> None:
    """The first choice, 20269121, drew five sample seeds of other X11 studies, one of them an
    RM30 ``non_exceptional`` sample at n = 2,000 row for row. It was replaced before any draw by
    the first pair, in steps of 10 from 20269131, that shares no sample seed (20269131 shared
    two and 20269141 four)."""
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


#: The suffix rule L3 adds to each property draw's labels: none, since no two draws share a
#: (family, label) pair and no seed of one draw is another's.
DECLARED_SUFFIXES = {
    ("interval_calibration", "non_exceptional_known_g"): (),
    ("optimal_target_control", "wrong_limit_known_g"): (),
    ("estimated_g_report", "non_exceptional_estimated_g"): (),
    ("exceptional_law_report", "partially_exceptional_known_g"): (),
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
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert total == TOTAL_SEEDS
    assert len(_own_sample_seeds()) == total


def test_no_sample_seed_is_another_x11_studys() -> None:
    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == SHARED_SEEDS
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, study.PRIMARY_LAW, 0) in declared


def test_no_pilot_seed_is_a_declared_seed() -> None:
    """The pilots' throwaway labels draw no declared sample seed, and neither label is a record
    or resampling seed. The first two labels tried, 20269177 and 20269187, each drew one
    declared seed, and the pilot script refused them before any draw."""
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

#: The study's floor of draws for a gated cell (plan section 11).
FLOOR = 1_000

#: The pilot labels and the (cell index, draws) each drew (``reviews/x11/scripts/
#: X11-I-calibration-pilot.py`` in the plans repository): the first pilot and the probe of the
#: gated cell.
PILOT_LABELS = {
    20269193: ((0, 4_000), (1, 4_000), (2, 1_000), (3, 1_000)),
    20269211: ((0, 4_000),),
}

#: The first pilot (label 20269193, no failed fit, no solver warning, rule L4 on every row):
#: 4,000 draws of the gated cell and the control, 1,000 of each reporting cell. Each reading is
#: against ``Psi_d0``: the SE ratio, the standardized bias and the coverage.
PILOT_DRAWS = 4_000
PILOT = {
    "known": {"ratio": 0.9714, "bias": -0.1041, "coverage": 0.9360},
    "wrong_limit": {"ratio": 1.0130, "bias": -3.2946, "coverage": 0.0922},
    "estimated": {"ratio": 0.9807, "bias": -0.1191, "coverage": 0.9370},
    "partial": {"ratio": 0.8824, "bias": -0.0238, "coverage": 0.9080},
}

#: The gated cell's readings against its own data-adaptive target ``psi~0n`` (descriptive), its
#: mean regret, and the mean regret in estimate SDs.
PILOT_ADAPTIVE = {"ratio": 0.9794, "bias": -0.0099, "coverage": 0.9415}
PILOT_REGRET = (0.001445, 0.0943)

#: The calibration controls on the gated pilot: (SE ratio, coverage) of the shrunken-SE control
#: (factor 0.80) and of the noise control (one efficiency-bound unit).
PILOT_SHRUNKEN = (0.7771, 0.8698)
PILOT_NOISE = (0.6971, 0.8297)

#: The record's ``rule_disagreement`` on each pilot cell: mean, smallest, largest.
PILOT_DISAGREEMENT = {
    "known": (0.0774, 0.0265, 0.3045),
    "wrong_limit": (0.0068, 0.0, 0.2688),
    "estimated": (0.0773, 0.0295, 0.2145),
    "partial": (0.2856, 0.0, 0.7095),
}

#: The probe of the gated cell (label 20269211, 4,000 draws, no failed fit, no solver warning,
#: rule L4 on every row) and the pooled 8,000 draws, against ``Psi_d0``. The rule for the probe
#: was fixed before it ran: each count is the larger of the declared rule's counts on the first
#: pilot and on the pooled draws.
PROBE: dict[str, dict[str, float]] = {
    "probe": {"draws": 4_000, "ratio": 0.9976, "bias": -0.1046, "coverage": 0.9495},
    "pooled": {"draws": 8_000, "ratio": 0.9843, "bias": -0.1044, "coverage": 0.9428},
}

#: The probe's readings against ``psi~0n``, its mean regret (and in estimate SDs), and its
#: calibration controls (SE ratio, coverage).
PROBE_ADAPTIVE = {"ratio": 1.0067, "bias": -0.0112, "coverage": 0.9497}
PROBE_REGRET = (0.001396, 0.0935)
PROBE_SHRUNKEN = (0.7981, 0.8832)
PROBE_NOISE = (0.6999, 0.8300)


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


_Z = np.linspace(-6.0, 6.0, 241)
_W = norm.pdf(_Z) / norm.pdf(_Z).sum()


def _predicted(coverage: float, replicates: int, pilot: int | None = PILOT_DRAWS) -> np.ndarray:
    """The run's hit counts on a grid over the coverage's predictive law (pilot and run error).

    ``pilot=None`` reads the coverage as known, for the conditional stress test."""
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


def _smallest(passes: Callable[[int], float]) -> int:
    for replicates in range(400, 40_001, 400):
        if replicates >= FLOOR and passes(replicates) >= tools.FAMILY_TARGET:
            return replicates
    raise AssertionError("no count reaches the target")


def primary_pass(
    replicates: int, reading: Mapping[str, float] = PILOT["known"], draws: int = PILOT_DRAWS
) -> float:
    """The primary gate's legs that can fail at these readings: the bias leg and the coverage
    floor 0.90 (the SE sanity band, 0.80 to 1.20, cannot fail at a ratio of 0.97)."""
    floor = study.STUDY.margins.coverage_floor
    return bias_pass(reading["bias"], replicates, draws) * floor_pass(
        reading["coverage"], replicates, floor, draws
    )


def calibration_pass(
    replicates: int,
    reading: Mapping[str, float] = PILOT["known"],
    draws: int = PILOT_DRAWS,
    known: bool = False,
) -> float:
    """The calibration positive's pass probability: the SE-ratio leg (the per-arm model of
    :func:`~tests.studies.mtp_study_tools.per_arm_pass`) times the coverage leg (the exact
    interval inside ``[0.92, 0.98]``). The two legs move together, so the product is a lower
    bound on the joint probability. ``known`` reads both as known (the conditional stress test).
    """
    low, high = study.STUDY.margins.calibration_coverage
    if known:
        ratio = float(tools.cell_pass(np.array([reading["ratio"]]), replicates, tools.C_FIT)[0])
        pilot: int | None = None
    else:
        ratio = tools.per_arm_pass([reading["ratio"]], draws, replicates)[0]
        pilot = draws
    coverage = reading["coverage"]
    inside = floor_pass(coverage, replicates, low, pilot) - (
        1.0 - ceiling_pass(coverage, replicates, high, pilot)
    )
    return ratio * inside


def _counts(reading: Mapping[str, float], draws: int) -> tuple[int, int]:
    return (
        _smallest(lambda r: primary_pass(r, reading, draws)),
        _smallest(lambda r: calibration_pass(r, reading, draws)),
    )


def test_the_primary_count_is_the_larger_rule_count() -> None:
    """The bias leg and the coverage floor pass together with 0.944 at 1,200 on the first pilot
    and 0.987 at 1,600, so it gives 1,600; the pooled draws give 1,200 (0.987). The bias leg
    alone would give 1,200 (0.985)."""
    assert _smallest(primary_pass) == 1_600
    assert primary_pass(1_200) == pytest.approx(0.9435, abs=1e-3)
    assert primary_pass(1_600) == pytest.approx(0.987, abs=1e-3)
    assert bias_pass(PILOT["known"]["bias"], 1_200) == pytest.approx(0.985, abs=1e-3)
    pooled = PROBE["pooled"]
    assert _smallest(lambda r: primary_pass(r, pooled, int(pooled["draws"]))) == 1_200
    assert study.PRIMARY_REPLICATES == 1_600


def test_the_calibration_count_is_the_larger_product_rule_count() -> None:
    """On the first pilot the SE ratio alone (H's per-arm rule) gives 11,200; the coverage leg
    also binds (coverage 0.936 against the band's floor 0.92), and the product reaches 0.95 at
    13,200. The pooled draws give 4,400, and the probe alone 3,200."""
    reading = PILOT["known"]
    assert tools.per_arm_replicates([reading["ratio"]], PILOT_DRAWS)["R"] == 11_200
    assert _smallest(calibration_pass) == 13_200
    assert calibration_pass(13_200) == pytest.approx(0.9525, abs=1e-3)
    assert calibration_pass(12_800) < tools.FAMILY_TARGET
    probe, pooled = PROBE["probe"], PROBE["pooled"]
    assert _counts(probe, int(probe["draws"])) == (1_200, 3_200)
    assert _counts(pooled, int(pooled["draws"])) == (1_200, 4_400)
    assert calibration_pass(13_200, pooled, int(pooled["draws"])) > 0.999
    assert max(13_200, 4_400) == properties.CALIBRATION_REPLICATES


def test_the_probe_matches_the_first_pilot_where_both_read() -> None:
    """The pooled readings sit at the two pilots' means (the pooled SE ratio within 0.0005, as the
    pooled spread is not the mean of the two), so a mistyped pin fails here. The two SE
    ratios differ by 1.5 combined pilot SDs and the coverages by 2.6 combined SEs; the biases,
    which read the regret, agree within 0.001."""
    probe, pooled = PROBE["probe"], PROBE["pooled"]
    for key in ("ratio", "bias", "coverage"):
        assert pooled[key] == pytest.approx((PILOT["known"][key] + probe[key]) / 2, abs=5e-4), key
    sigma = tools.pilot_sigma(PILOT_DRAWS)
    assert abs(probe["ratio"] - PILOT["known"]["ratio"]) / (np.sqrt(2) * sigma) < 1.6
    assert abs(probe["bias"] - PILOT["known"]["bias"]) < 0.001


def test_the_calibration_count_survives_one_adverse_pilot_se() -> None:
    """The stress test reads the SE ratio and the coverage as known at one pilot SE adverse
    each, without adding the pilot error a second time."""
    reading = PILOT["known"]
    coverage = reading["coverage"]
    shifted = {
        "ratio": reading["ratio"] - tools.pilot_sigma(PILOT_DRAWS),
        "coverage": coverage - np.sqrt(coverage * (1 - coverage) / PILOT_DRAWS),
    }
    found = calibration_pass(properties.CALIBRATION_REPLICATES, shifted, known=True)
    assert found > tools.FAMILY_TARGET


def test_the_target_control_count_is_the_floor() -> None:
    """The control's coverage (0.092) clears the ceiling 0.92 with probability one at 400, so
    the floor sets 1,200."""
    floor = study.STUDY.margins.calibration_coverage[0]
    coverage = PILOT["wrong_limit"]["coverage"]
    assert (
        properties.TARGET_REPLICATES
        == _smallest(lambda r: ceiling_pass(coverage, r, floor))
        == 1_200
    )
    assert ceiling_pass(coverage, 400, floor) > 0.9999
    assert coverage == pytest.approx(0.0925, abs=0.012)


def test_the_reporting_counts_follow_their_rules() -> None:
    assert properties.ESTIMATED_REPLICATES == properties.EXCEPTIONAL_REPLICATES == 1_200
    assert properties.EXCEPTIONAL_REPLICATES == 1_200 == _smallest(lambda r: 1.0)


def test_every_calibration_control_is_pilot_checked_below_the_band() -> None:
    floor = study.STUDY.margins.calibration_se_ratio[0]
    low = study.STUDY.margins.calibration_coverage[0]
    for ratio, coverage in (PILOT_SHRUNKEN, PILOT_NOISE, PROBE_SHRUNKEN, PROBE_NOISE):
        assert ratio < floor - 0.05
        assert ceiling_pass(coverage, properties.CALIBRATION_REPLICATES, low) > 0.999
    for (ratio, _), reading in ((PILOT_SHRUNKEN, PILOT["known"]), (PROBE_SHRUNKEN, PROBE["probe"])):
        assert ratio == pytest.approx(properties.SHRUNKEN_SE_FACTOR * reading["ratio"], abs=1e-4)


def test_the_regret_is_what_separates_the_two_targets() -> None:
    """Against its own ``psi~0n`` the gated cell is centred (bias -0.01 SDs in both pilots);
    against ``Psi_d0`` it reads the mean regret, about 0.094 SDs, as bias. At the control the
    regret is the quadrature gap."""
    for reading, adaptive, (regret, in_sds) in (
        (PILOT["known"], PILOT_ADAPTIVE, PILOT_REGRET),
        (PROBE["probe"], PROBE_ADAPTIVE, PROBE_REGRET),
    ):
        assert reading["bias"] == pytest.approx(adaptive["bias"] - in_sds, abs=0.002)
        assert regret == pytest.approx(in_sds * study.EFFICIENCY_BOUND / np.sqrt(2_000), rel=0.05)
        assert abs(adaptive["bias"]) < 0.05


#: The sample seeds a declared run draws.
TOTAL_SEEDS = 1_600 + 13_200 + 1_200 + 1_200 + 1_200

#: The sample seeds this study shares with the RM30 and the other X11 studies.
SHARED_SEEDS: set[int] = set()
