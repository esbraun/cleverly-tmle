"""The declared constants of the ``learned-regimen-ltmle`` study, recomputed without a run.

Each number here was fixed before any registered run. The law checks call no sampler: they
integrate on the study grid through the study law module. The pilot readings and the sizing rules
that read them are pinned. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_regimen_law as law
from tests.studies import learned_regimen_ltmle as study
from tests.studies import learned_regimen_ltmle_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_regimen_ltmle_design")

base = law.base


# ----------------------------------------------------------------------------- the law


def test_the_law_limit_is_the_declared_one() -> None:
    """The optimal pair's value, the bound that sizes the noise control, and its shares."""
    limit = law.law_limit()
    assert limit["truth"] == pytest.approx(0.557923, abs=1e-6)
    assert limit["bound"] == pytest.approx(study.EFFICIENCY_BOUNDS[law.LAW], abs=1e-6)
    assert limit["treated_1"] == pytest.approx(0.360, abs=1e-3)
    assert limit["treated_2"] == pytest.approx(0.341, abs=1e-3)


def test_the_learned_pair_beats_every_static_regimen() -> None:
    """The best static regimen is worth 0.4600, 0.098 below the optimal pair: a learned rule
    that collapsed to a static one would read far from the truth of the declared limit."""
    values = law.static_values()
    assert values == pytest.approx(
        {(0.0, 0.0): 0.4371, (0.0, 1.0): 0.4586, (1.0, 0.0): 0.4600, (1.0, 1.0): 0.2635},
        abs=1e-4,
    )
    assert law.law_limit()["truth"] - max(values.values()) > 0.09


def test_each_boundary_has_a_margin() -> None:
    """The weak-margin diagnostic: 5.6% of the rows that follow the node-1 rule have a node-2
    blip within 0.1 logit of zero, and 4.0% a node-1 value gap below 0.01. Both are disclosed;
    neither is a gate."""
    margins = law.margins()
    assert margins["node_2"] == pytest.approx(0.05598, abs=1e-4)
    assert margins["node_1"] == pytest.approx(0.0400, abs=1e-3)


def test_the_node_2_rule_depends_on_the_first_arm() -> None:
    """``L2`` is lowered by 0.8 under ``A1 = 1`` and the blip falls by 0.6, so the node-2
    threshold moves from ``L2 > 0.083`` to ``L2 > 0.583``: the backward step is load bearing."""
    thresholds = [(-law.Y_BLIP2[0] - law.Y_BLIP2[2] * a1) / law.Y_BLIP2[1] for a1 in (0, 1)]
    assert thresholds == pytest.approx([0.08333, 0.58333], abs=1e-4)
    assert law.L2_MEAN[2] == -0.8


def test_the_drop_w1_learner_treats_everyone_and_its_plug_in_is_biased() -> None:
    """The ``drop_w1`` limit rule treats everyone at node 1, and its untargeted plug-in misses
    that pair's value by 0.0596, 2.65 SEs of the pair's curve at n = 2,000. Its limit arm
    contrast is only 0.0266 and 0.0145 at the two levels of ``W2``: the weak margin that moved it
    out of ``conservative_coverage``."""
    limit = law.node1_limit("drop_w1")
    assert limit["treated_1"] == pytest.approx(1.0, abs=1e-12)
    assert limit["plug_in"] == pytest.approx(0.529063, abs=1e-6)
    assert limit["rule_value"] == pytest.approx(0.469496, abs=1e-6)
    assert limit["bias"] == pytest.approx(0.059567, abs=1e-6)
    assert limit["bias"] / (limit["bound"] / np.sqrt(study.PRIMARY_N)) == pytest.approx(
        2.65, abs=0.01
    )
    gaps = [float(np.unique(limit["gap"][w2]).item()) for w2 in law.W2_LEVELS]
    assert gaps == pytest.approx([0.026586, 0.014500], abs=1e-6)


def test_the_squared_w1_learner_has_a_margin_and_a_biased_plug_in() -> None:
    """The ``squared_w1`` limit rule treats half the population at node 1 and is worth 0.053
    less than the optimum; its arm contrast crosses zero inside the support, with 7.8% of ``W``
    within 0.01 of the crossing; its plug-in misses its value by 2.17 SEs."""
    limit = law.node1_limit("squared_w1")
    assert limit["treated_1"] == pytest.approx(0.497, abs=1e-3)
    assert limit["rule_value"] == pytest.approx(0.504591, abs=1e-6)
    assert law.law_limit()["truth"] - limit["rule_value"] == pytest.approx(0.0533, abs=1e-4)
    assert limit["near"] == pytest.approx(0.078, abs=1e-3)
    assert limit["bias"] / (limit["bound"] / np.sqrt(study.PRIMARY_N)) == pytest.approx(
        2.17, abs=0.01
    )
    for w2 in law.W2_LEVELS:
        gap = limit["gap"][w2]
        assert gap.min() < -0.05 and gap.max() > 0.15


def test_the_linear_learners_would_learn_a_near_optimal_rule() -> None:
    """Why the redesign took ``squared_w1``: a main-terms linear node-1 learner is misspecified in
    form but its limit rule is worth within 1e-4 of the optimum, with no plug-in bias to speak
    of, so it would test nothing the declared cell does not."""
    law.NODE1_FEATURES["_main"] = lambda w1, w2: np.column_stack(
        [np.ones_like(w1), w1, np.full_like(w1, w2)]
    )
    try:
        limit = law.node1_limit("_main")
    finally:
        del law.NODE1_FEATURES["_main"]
    assert law.law_limit()["truth"] - limit["rule_value"] < 1e-4
    assert abs(limit["bias"] / (limit["bound"] / np.sqrt(study.PRIMARY_N))) < 0.1


def test_positivity_holds() -> None:
    """The smallest product of the four treatment and censoring probabilities on the support of
    ``W1`` (the tails of ``L2`` are unbounded, so this reads its 0.1% quantiles)."""
    w1 = law.GRID
    smallest = np.inf
    for w2 in law.W2_LEVELS:
        for a1 in (0.0, 1.0):
            first = law.g1(w1, w2) if a1 else 1.0 - law.g1(w1, w2)
            for tail in (-3.09, 3.09):
                l2 = law.l2_mean(a1, w1, w2) + tail
                second = np.minimum(law.g2(l2, a1), 1.0 - law.g2(l2, a1))
                product = first * law.c1(w1) * second * law.c2(l2)
                smallest = min(smallest, float(product.min()))
    assert smallest > 0.05


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.SCENARIOS == {"two_node": ("ey_regimen[learned regimen]",)}
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "two_node__declared",
            "two_node__shrunken_se_control",
            "two_node__noise_control",
        ),
        "conservative_coverage": (
            "squared_w1__declared",
            "squared_w1__shrunken_coverage_control",
        ),
        "targeting_necessity": (
            "squared_w1__targeted",
            "squared_w1__untargeted",
            "two_node__targeted",
            "two_node__untargeted",
        ),
        "fold_locality": ("two_node__fold_local", "two_node__validation_rows"),
    }
    assert properties.SHRUNKEN_SE_FACTOR == study.COVERAGE_SHRINK == 0.80
    assert properties.TARGETING_DISPLACEMENT == properties.FOLD_LOCALITY_DISPLACEMENT == 0.5
    assert study.STUDY.truth_varies_by_replicate
    assert not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == study.STUDY.n == 2_000
    assert properties.DRAW_LEARNERS == {
        ("interval_calibration", "two_node"): "declared",
        ("conservative_coverage", "squared_w1"): "squared_w1",
        ("targeting_necessity", "squared_w1"): "squared_w1",
        ("targeting_necessity", "two_node"): "declared",
        ("fold_locality", "two_node"): "declared",
    }
    assert {(draw.family, draw.label) for draw in properties.PROPERTY_DRAWS} == set(
        properties.DRAW_LEARNERS
    )
    # No cell reads the weak-margin learner (design review, finding 4).
    assert "drop_w1" not in properties.DRAW_LEARNERS.values()
    for draw in properties.PROPERTY_DRAWS:
        assert draw.law == study.PRIMARY_SCENARIO


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/longitudinal/estimator.py",
        "src/cleverly/longitudinal/learned.py",
        "src/cleverly/_support.py",
        "tests/studies/_learned_regimen_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/learned_regimen_ltmle_properties.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))


def test_the_support_row_is_declared_before_the_run() -> None:
    """One row, ``G47``: its key is the key of the study's fits, and the registered row is the
    declared one, the only row with that key."""
    from cleverly._support_rows import LONGITUDINAL_ROWS, long_row

    declared = study.SUPPORT_ROWS["learned"]
    assert declared == {
        "row": "G47",
        "regimens": ("learned",),
        "splitting": ("cross_fitted",),
        "missing": ("censoring",),
        "kinds": ("point", "wald"),
    }
    row = long_row(
        declared["row"],
        [study.STUDY.slug],
        regimens=declared["regimens"],
        splitting=declared["splitting"],
        missing=declared["missing"],
    )
    assert set(row.kinds) == set(declared["kinds"])
    fitted = law.fit(law.draw(400, 3), "declared", 4)
    assert row.key == fitted.config.support_key
    (registered,) = [existing for existing in LONGITUDINAL_ROWS if existing.key == row.key]
    assert registered.id == declared["row"]
    assert set(registered.kinds) == set(declared["kinds"])
    assert registered.evidence == (study.STUDY.slug,)


def test_the_declared_gates_are_read() -> None:
    """Every verdict family has a rule; the controls are the declared kinds; the harness's
    exactness gate is the declared tolerance; the fit and harness stop on a mismatch."""
    from tests.studies.evidence import property_verdicts

    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == set(study.PROPERTY_CELLS)
    assert property_verdicts.SE_CONTROL_SIDES["shrunken_se_control"] == "below"
    assert law.EXACTNESS_TOLERANCE == 1e-8
    assert (
        properties.TARGETING_NECESSITY.labels
        == study.NECESSITY_LABELS
        == (
            "squared_w1",
            "two_node",
        )
    )
    assert properties.FOLD_LOCALITY.arms == study.LOCALITY_ARMS
    assert properties.FOLD_LOCALITY.truth_varies and properties.TARGETING_NECESSITY.truth_varies


def test_the_coverage_verdict_reads_the_floor_and_the_control_ceiling() -> None:
    """A deliberate-mutation control of :func:`coverage_verdicts`: a positive whose lower bound
    sits just below the floor fails, a control whose upper bound reaches the floor fails."""
    import pandas as pd

    summary = pd.DataFrame(
        {
            "property": ["conservative_coverage"] * 4,
            "role": ["positive", "positive", "control", "control"],
            "bias_equivalent": [True, True, False, False],
            "coverage_ci_lower": [0.921, 0.919, 0.80, 0.85],
            "coverage_ci_upper": [0.96, 0.96, 0.919, 0.92],
        }
    )
    properties.coverage_verdicts(summary, floor=0.92)
    assert summary["passed"].tolist() == [True, False, True, False]


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
    from tests.studies import learned_rule_ep as ep_rule
    from tests.studies import learned_rule_ep_properties as ep_rule_properties
    from tests.studies import learned_rule_in_sample as f2
    from tests.studies import learned_rule_in_sample_properties as f2_properties
    from tests.studies import learned_rule_intermediate as j
    from tests.studies import learned_rule_intermediate_properties as j_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_optimal_compositions as f3
    from tests.studies import learned_rule_optimal_compositions_properties as f3_properties
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
        (ep_rule, ep_rule_properties),
        (e1, e1_properties),
        (e2, e2_properties),
        (f1, f1_properties),
        (f2, f2_properties),
        (g, g_properties),
        (h, h_properties),
        (i, i_properties),
        (f3, f3_properties),
        (j, j_properties),
    ]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_cvtmle_boundary as boundary

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module, _ in [*_x11_modules(), (boundary, None)]:
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others
    assert study.SEED not in PILOT_LABELS and study.RESAMPLING_SEED not in PILOT_LABELS


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


#: The suffix rule L3 adds to each property draw's labels: none.
DECLARED_SUFFIXES: dict[tuple[str, str], tuple[Any, ...]] = {
    ("interval_calibration", "two_node"): (),
    ("conservative_coverage", "squared_w1"): (),
    ("targeting_necessity", "squared_w1"): (),
    ("targeting_necessity", "two_node"): (),
    ("fold_locality", "two_node"): (),
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


def test_the_first_seed_candidate_was_refused() -> None:
    """20271421, the first candidate, would draw sample seeds of another X11 study."""
    import dataclasses

    record = dataclasses.replace(study.STUDY, seed=20271421, resampling_seed=20271422)
    seeds = base.declared_sample_seeds(
        record, study.PRIMARY_REPLICATES, properties.PROPERTY_DRAWS, properties.suffixes(record)
    )
    assert len(seeds & _other_x11_sample_seeds()) == FIRST_CANDIDATE_CLASHES


# ---------------------------------------------------------------------------- the sizing

#: The study's floor of draws for a cell, as a multiple of 400 (plan section 11).
FLOOR = 1_200

#: The pilot labels and the (seed index, draws) each drew (``reviews/x11/scripts/
#: X11-K-calibration-pilot.py`` in the plans repository): the first pilot (``declared`` and
#: ``validation_rows`` on index 0, paired; ``drop_w1``, then named ``misspecified``, on index 2),
#: the probe of ``declared`` and the redesign pilot of ``squared_w1``.
PILOT_LABELS = {
    20271401: ((0, 4_000), (2, 4_000)),
    20271413: ((0, 4_000),),
    20271437: ((0, 4_000),),
}

PILOT_DRAWS = 4_000
POOLED_DRAWS = 8_000

#: The readings of each pilot cell (no failed fit, no solver warning, rule L4 and the exactness
#: gate on every row): SE ratio, standardized bias and coverage against each row's own truth;
#: for ``squared_w1`` also its control's coverage (SE x 0.80); for the paired families the
#: positive's and the control's standardized bias (each in its own SD) and the displacement in
#: positive-arm SDs.
PILOT: dict[str, dict[str, float]] = {
    "declared": {"ratio": 0.974528, "bias": 0.014822, "coverage": 0.94075},
    "squared_w1": {"bias": -0.023974, "coverage": 0.9455, "control_coverage": 0.88325},
}
#: The probe of ``declared`` (label 20271413) and the 8,000 draws pooled with the first pilot.
PROBE = {"ratio": 0.964592, "bias": -0.005381, "coverage": 0.94325}
POOLED = {"ratio": 0.969535, "bias": 0.004671, "coverage": 0.942}
#: The ``declared`` controls on the first pilot: SE ratio of the shrunken SE (x0.80) and of the
#: noise control (one bound unit), and their coverage.
PILOT_CONTROLS = {"shrunken": (0.779622, 0.8740), "noise": (0.701905, 0.8235)}
#: ``targeting_necessity``: ``squared_w1`` on the redesign pilot and the declared learners on the
#: 8,000 pooled ``declared`` draws; ``fold_locality`` on the paired first-pilot draws. The
#: ``drop_w1`` reading of the first pilot is kept as the record of the cell it no longer is.
PILOT_NECESSITY = {
    "squared_w1": ({"positive": -0.023974, "control": 2.571193, "displacement": 2.646166}, 4_000),
    "two_node": ({"positive": 0.004671, "control": 0.839936, "displacement": 0.76496}, 8_000),
}
DROP_W1_NECESSITY = {"positive": -0.023220, "control": 1.945835, "displacement": 2.063745}
PILOT_LOCALITY = {"positive": 0.014822, "control": 4.288171, "displacement": 4.243259}
#: The first pilot's ``misspecified`` (``drop_w1``) coverage reading, which moved
#: ``conservative_coverage`` to ``squared_w1``.
DROP_W1_COVERAGE = {"coverage": 0.8980, "upper": 0.9100, "ratio": 0.8400}

#: The mean seconds per draw over the five pilot cells (fit and harness, 4 jobs on a machine at
#: load about 8), and the budget of the probe rule: 75 minutes of fitting at 4 jobs.
PILOT_SECONDS = 1.589
BUDGET_MINUTES = 75
BUDGET_FITS = int(BUDGET_MINUTES * 60 * 4 / PILOT_SECONDS)


def bias_pass(
    bias: float, replicates: int, pilot: int | None = PILOT_DRAWS, margin: float = 0.25
) -> float:
    """The predictive probability that the 99% interval of the mean standardized error falls
    inside ``±margin`` at ``replicates`` draws, given a pilot bias read from ``pilot`` draws
    (the F2 rule). ``pilot=None`` reads the bias as known."""
    half = norm.ppf(0.995) / np.sqrt(replicates)
    spread = np.sqrt(1.0 / replicates + (0.0 if pilot is None else 1.0 / pilot))
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


def primary_pass(replicates: int, pilot: int | None = PILOT_DRAWS) -> float:
    """The bias leg times the coverage floor (0.90) of the ``declared`` reading.

    The primary gate's SE sanity leg is left out: it reads the band (0.80, 1.20), which cannot
    fail at a ratio of 0.97. ``pilot=None`` reads both legs as known."""
    reading = PILOT["declared"]
    floor = study.STUDY.margins.coverage_floor
    return bias_pass(reading["bias"], replicates, pilot) * floor_pass(
        reading["coverage"], replicates, floor, pilot
    )


def calibration_pass(
    replicates: int, reading: dict[str, float], pilot: int | None = PILOT_DRAWS
) -> float:
    """The positive's SE-ratio leg (the per-cell model) times its coverage leg (the exact interval
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


def control_pass(ratio: float, replicates: int, pilot: int = PILOT_DRAWS) -> float:
    """The predictive probability that a control's 99% SE-ratio interval lies wholly below the
    band, given its pilot ratio."""
    run = tools.pilot_sigma(replicates)
    spread = np.hypot(tools.pilot_sigma(pilot), run)
    low = study.STUDY.margins.calibration_se_ratio[0]
    return float(norm.cdf((low - norm.ppf(0.995) * run - ratio) / spread))


def coverage_pass(replicates: int, pilot: int | None = PILOT_DRAWS) -> float:
    """The ``squared_w1`` positive's bias leg and coverage floor (0.92), times its control's
    coverage ceiling (below 0.92)."""
    reading = PILOT["squared_w1"]
    floor = study.STUDY.margins.calibration_coverage[0]
    bias = bias_pass(reading["bias"], replicates, pilot)
    return (
        bias
        * floor_pass(reading["coverage"], replicates, floor, pilot)
        * ceiling_pass(reading["control_coverage"], replicates, floor, pilot)
    )


def paired_pass(reading: dict[str, float], replicates: int, pilot: int = PILOT_DRAWS) -> float:
    """One paired label: the positive's bias leg, the control's discrimination leg, and the
    displacement at or above 0.5 positive-arm SDs, read as normal with the pilot's SE."""
    spread = np.sqrt(1.0 / replicates + 1.0 / pilot)
    return (
        bias_pass(reading["positive"], replicates, pilot)
        * discrimination_pass(reading["control"], replicates, pilot)
        * float(norm.cdf((abs(reading["displacement"]) - 0.5) / spread))
    )


def _fits() -> dict[str, int]:
    """The fits of each part of a declared run (``fold_locality`` fits twice per draw)."""
    return {
        "primary": study.PRIMARY_REPLICATES,
        "interval_calibration": properties.CALIBRATION_REPLICATES,
        "conservative_coverage": properties.COVERAGE_REPLICATES,
        "targeting_necessity": len(study.NECESSITY_LABELS) * properties.NECESSITY_REPLICATES,
        "fold_locality": 2 * properties.LOCALITY_REPLICATES,
    }


def test_the_primary_count() -> None:
    assert study.PRIMARY_REPLICATES == _smallest(primary_pass) == 1_200
    assert primary_pass(1_200) == pytest.approx(0.992, abs=1e-3)


def test_the_calibration_rule_needs_9200_draws_on_the_first_pilot_and_the_pooled_draws() -> None:
    """The per-cell rule gives 9,200 on the first pilot and on the 8,000 draws pooled with the
    probe, and 22,000 on the probe alone. Each triggers the probe rule's budget clause."""
    assert _smallest(lambda r: calibration_pass(r, PILOT["declared"])) == 9_200
    assert _smallest(lambda r: calibration_pass(r, POOLED, POOLED_DRAWS)) == 9_200
    assert _smallest(lambda r: calibration_pass(r, PROBE)) == 22_000


def test_the_calibration_count_is_the_per_cell_rules() -> None:
    """The design review's exception to the budget fallback (finding 1): the count is the per-cell
    rule's on the pooled 8,000 draws."""
    assert (
        properties.CALIBRATION_REPLICATES
        == _smallest(lambda r: calibration_pass(r, POOLED, POOLED_DRAWS))
        == 9_200
    )
    assert calibration_pass(9_200, POOLED, POOLED_DRAWS) == pytest.approx(0.952, abs=1e-3)
    assert calibration_pass(9_200, POOLED, None) == pytest.approx(0.993, abs=1e-3)


def test_what_the_budget_fallback_would_have_given() -> None:
    """Pinned, not declared: on the pilot timings (a host at load 8) the fallback would have left
    4,000 draws, a pooled predictive pass of 0.720."""
    others = sum(count for name, count in _fits().items() if name != "interval_calibration")
    assert BUDGET_FITS == 11_327
    assert others == 8_400
    assert calibration_pass(4_000, POOLED, POOLED_DRAWS) == pytest.approx(0.720, abs=1e-3)


def test_the_run_host_timing_condition() -> None:
    """The run keeps 9,200 if the run host's mean seconds per fit, times the declared fits, over
    4 jobs is at most 75 minutes: at most about 1.02 s per fit (the design record states the
    condition before the timing is taken)."""
    fits = sum(_fits().values())
    assert fits == 17_600
    ceiling = BUDGET_MINUTES * 60 * 4 / fits
    assert ceiling == pytest.approx(1.023, abs=1e-3)
    # The pilot host's timing would not meet it; the review's idle-host timings do.
    assert PILOT_SECONDS > ceiling > 0.84


def test_the_controls_fail_at_the_declared_count() -> None:
    count = properties.CALIBRATION_REPLICATES
    for ratio, coverage in PILOT_CONTROLS.values():
        assert control_pass(ratio, count) >= 0.999
        assert ratio < study.STUDY.margins.calibration_se_ratio[0]
        assert coverage < study.STUDY.margins.calibration_coverage[0]


def test_the_coverage_count() -> None:
    assert properties.COVERAGE_REPLICATES == _smallest(coverage_pass) == 2_400
    assert coverage_pass(2_400) == pytest.approx(0.972, abs=1e-3)
    assert coverage_pass(2_000) < tools.FAMILY_TARGET


def test_the_drop_w1_learner_had_no_coverage_count() -> None:
    """The first pilot's reading that moved ``conservative_coverage`` off ``drop_w1``: its exact
    99% coverage interval lies wholly below the 0.92 floor."""
    floor = study.STUDY.margins.calibration_coverage[0]
    assert DROP_W1_COVERAGE["upper"] < floor
    hits = round(DROP_W1_COVERAGE["coverage"] * PILOT_DRAWS)
    assert beta.ppf(0.995, hits + 1, PILOT_DRAWS - hits) == pytest.approx(
        DROP_W1_COVERAGE["upper"], abs=1e-4
    )


def test_the_paired_counts() -> None:
    for label, (reading, pilot) in PILOT_NECESSITY.items():
        assert (
            properties.NECESSITY_REPLICATES
            == _smallest(lambda r, reading=reading, pilot=pilot: paired_pass(reading, r, pilot))
            == 1_200
        ), label
        assert paired_pass(reading, 1_200, pilot) > 0.9999, label
    assert set(PILOT_NECESSITY) == set(study.NECESSITY_LABELS)
    assert (
        properties.LOCALITY_REPLICATES
        == _smallest(lambda r: paired_pass(PILOT_LOCALITY, r))
        == 1_200
    )
    assert paired_pass(PILOT_LOCALITY, 1_200) > 0.9999
    # The withdrawn drop_w1 label would also have passed: the move cannot help a verdict.
    assert paired_pass(DROP_W1_NECESSITY, 1_200) > 0.999


def test_the_stress_test() -> None:
    """Conditional on the readings as known, without a second pilot error."""
    assert coverage_pass(properties.COVERAGE_REPLICATES, None) > 0.99
    assert primary_pass(study.PRIMARY_REPLICATES, None) > 0.99
    assert calibration_pass(properties.CALIBRATION_REPLICATES, POOLED, None) > 0.99


#: The sample seeds a declared run draws.
TOTAL_SEEDS = 16_400

#: The sample seeds of another X11 study (``learned-rule-intermediate`` included) that the refused
#: first candidate 20271421 draws.
FIRST_CANDIDATE_CLASHES = 4
