"""The declared constants of ``learned-rule-optimal-compositions``, recomputed without a run.

Each number here is fixed before any registered run. The truths and bounds are recomputed by
quadrature through the study's law module; the target control's power, the stratum-fold
feasibility and the counts are computed from declared inputs and from the committed sizing
pilot (``tests/studies/learned_rule_optimal_compositions_pilot.csv``); the declared seeds are only
computed, never drawn from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_intermediate_law as zlaw
from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies import learned_rule_optimal_compositions as study
from tests.studies import learned_rule_optimal_compositions_pilot as pilot
from tests.studies import learned_rule_optimal_compositions_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed
from tests.unit.test_learned_rule_optimal_value_design import (
    _x11_modules,
    bias_pass,
    ceiling_pass,
    floor_pass,
)

pytestmark = pytest.mark.xdist_group("learned_rule_optimal_compositions_design")

base = law.base


# ----------------------------------------------------------------------------- the laws


@pytest.mark.parametrize("label", list(study.LIMITS))
def test_each_labels_limit_is_the_declared_one(label: str) -> None:
    limit = law.label_limits()[label]
    assert limit["truth"] == pytest.approx(study.LIMITS[label]["truth"], abs=1e-6)
    assert limit["bound"] == pytest.approx(study.LIMITS[label]["bound"], abs=1e-6)


def test_the_marginal_limits_are_the_optimal_value_studys() -> None:
    """The marginal and in-sample labels read ``learned-rule-optimal-value``'s gated law."""
    from tests.studies import learned_rule_optimal_value as optimal

    for label in ("strata_all", "in_sample"):
        assert study.LIMITS[label]["truth"] == optimal.LIMITS["non_exceptional"]["truth"]
        assert study.LIMITS[label]["bound"] == optimal.LIMITS["non_exceptional"]["bound"]
    assert study.LIMITS["intermediate_z0"]["truth"] == pytest.approx(
        study.LIMITS["strata_all"]["truth"], abs=1e-6
    )


def test_the_share_mixture_of_the_strata_is_the_marginal_value() -> None:
    limits = law.label_limits()
    mixture = 0.5 * limits["strata_s0"]["truth"] + 0.5 * limits["strata_s1"]["truth"]
    assert mixture == pytest.approx(limits["strata_all"]["truth"], abs=1e-12)


def test_the_weight_keeps_the_optimal_rule() -> None:
    """A weight in ``W`` alone leaves ``Qbar_w = Qbar0``, so ``d0^w = d0`` and ``g_w = g0``: the
    known mechanism is the tilt's own (impl review M1)."""
    d0 = law.optimal.optimal_rule(law.optimal.law_qbar(law.LAW))
    assert all(
        np.array_equal(
            d0[w2], base.qbar0(1, base.GRID, w2, law.LAW) > base.qbar0(0, base.GRID, w2, law.LAW)
        )
        for w2 in base.W2_LEVELS
    )
    assert study.LIMITS["weighted"]["truth"] - study.LIMITS["strata_all"]["truth"] == (
        pytest.approx(0.0537, abs=5e-5)
    )


def test_the_intermediate_levels_have_no_tie_set() -> None:
    """The level blips ``0.1 + W1 - 0.6 z`` are linear in ``W1`` with slope 1: no tie mass and a
    bounded density at 0 (margin exponent 1), as the plan requires of the gated cell."""
    for z in zlaw.LEVELS:
        # Linear with slope 1: at most one grid point is a root, a set of measure zero.
        assert int(np.sum(zlaw.blip(z, base.GRID) == 0.0)) <= 1
        assert float(np.ptp(np.diff(zlaw.blip(z, base.GRID)))) < 1e-12
        gap = np.abs(zlaw.blip(z, base.GRID))
        assert float(np.mean(gap <= 0.01)) / 0.01 < 10.0


def test_the_cells() -> None:
    labels = (
        "weighted",
        "strata_all",
        "strata_s0",
        "strata_s1",
        "contrast_1",
        "contrast_0",
        "in_sample",
        "intermediate_z0",
        "intermediate_z1",
    )
    assert labels == study.CALIBRATION_LABELS
    assert {
        "interval_calibration": tuple(
            f"{label}__{kind}"
            for label in labels
            for kind in ("declared", "shrunken_se_control", "noise_control")
        ),
        "weighted_target_control": ("weighted__target_gap_control",),
    } == study.PROPERTY_CELLS
    assert set(study.SCENARIOS) == set(law.SETTINGS)
    assert sum(len(names) for names in study.SCENARIOS.values()) == 9
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert not study.STUDY.truth_varies_by_replicate
    assert not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == study.STUDY.n == 2_000
    assert study.STUDY.reference is None


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_support_rows_name_each_composition_as_the_fits_key_it() -> None:
    """Each declared row's dimensions are the ones a fit of its setting reports (T55)."""
    for setting, row in study.SUPPORT_ROWS.items():
        frame = law.draw(setting, 400, 1)
        result = law.fit(frame, setting, 3)
        result = result[0.0] if setting == "intermediate" else result
        key = result.config.support_key
        assert tuple(key.splitting) == row["splitting"], setting
        assert set(key.family) == set(row["families"]), setting
        assert set(row["qualifiers"]) <= set(key.qualifiers), setting
        if "weighting" in row:
            assert (key.weighting,) == row["weighting"]
        if "missing" in row:
            assert tuple(key.missing) == row["missing"]


def test_the_declared_gates_are_read() -> None:
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == set(study.PROPERTY_CELLS)
    assert study.STUDY.margins.calibration_coverage[0] == 0.92


# ------------------------------------------------------------------------- the controls


def target_control_power(replicates: int, se_ratio: float = 1.07) -> float:
    """The probability that the control's exact 99% coverage interval lies below 0.92.

    The control's interval is centred on the weighted target, 0.0537 from the unweighted truth,
    with SE ``se_ratio`` times the efficient one; the largest SE the calibration band admits
    (1.07) gives the most coverage, so the least power.
    """
    gap = study.LIMITS["weighted"]["truth"] - study.LIMITS["strata_all"]["truth"]
    z = gap / (study.LIMITS["weighted"]["bound"] / np.sqrt(study.PRIMARY_N))
    critical = base.CRITICAL * se_ratio
    covered = norm.cdf(critical - z) - norm.cdf(-critical - z)
    hits = np.arange(replicates + 1)
    upper = np.where(hits < replicates, beta.ppf(0.995, hits + 1, replicates - hits), 1.0)
    floor = study.STUDY.margins.calibration_coverage[0]
    return float(np.sum(binom.pmf(hits, replicates, covered) * (upper < floor)))


def test_the_target_control_has_power() -> None:
    """Plan section 5: the gap is 3.40 SEs at n = 2,000 and the control's power at least 0.90."""
    gap = study.LIMITS["weighted"]["truth"] - study.LIMITS["strata_all"]["truth"]
    se = study.LIMITS["weighted"]["bound"] / np.sqrt(study.PRIMARY_N)
    assert gap / se == pytest.approx(3.40, abs=0.01)
    assert target_control_power(properties.TARGET_REPLICATES) >= 0.90
    assert properties.TARGET_REPLICATES == 1_200


def test_an_empty_stratum_fold_arm_is_negligible() -> None:
    """Plan review finding 10: a stratum-fold arm with no row refuses with a ``DataError``.

    A union bound over 10 validation folds, 2 strata and 2 arms: each stratum-fold holds
    ``Binomial(200, 1/2)`` rows, and each row is in an arm with probability at least
    ``min g0 = expit(-0.5)``. The probability of an empty arm in one draw is below ``10^-6``.
    """
    smallest = float(min(base.g0(base.GRID, w2).min() for w2 in base.W2_LEVELS))
    smallest = min(smallest, 1.0 - float(max(base.g0(base.GRID, w2).max() for w2 in (0.0, 1.0))))
    rows = np.arange(201)
    empty = float(np.sum(binom.pmf(rows, 200, 0.5) * (1.0 - smallest) ** rows))
    assert 10 * 2 * 2 * empty < 1e-6


# --------------------------------------------------------------------------- the sizing


def _pilot() -> pd.DataFrame:
    return pd.read_csv(pilot.PILOT_CSV).set_index("label")


def test_the_pilot_covers_every_label_at_its_declared_size() -> None:
    frame = _pilot()
    assert list(frame.index) == list(study.CALIBRATION_LABELS)
    assert (frame["replicates"] == pilot.PILOT_DRAWS).all()
    assert (frame["n"] == study.PRIMARY_N).all()
    assert int(frame["solver_warnings"].max()) == 0


def test_the_pilot_screens_every_label_inside_the_band() -> None:
    """The arm-pooled rule's per-arm screen: each label's pilot SE ratio within
    ``2.576 sigma_P + 0.01`` of 1, or the label is diagnosed alone before ``R`` is set."""
    reading = tools.replicates_from_pilot(_pilot()["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    assert all(reading["screened"]), reading
    assert not reading["suspect"], reading


def test_the_calibration_count_is_the_arm_pooled_rule_count() -> None:
    """The smallest multiple of 400 with family-wise pass probability at least 0.95 over the nine
    labels (each at least 0.90), on the committed pilot (``tools.replicates_from_pilot``)."""
    reading = tools.replicates_from_pilot(_pilot()["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    assert reading["R"] == properties.CALIBRATION_REPLICATES, reading


def test_every_label_passes_its_coverage_leg_at_the_count() -> None:
    """The coverage leg (exact interval inside ``[0.92, 0.98]``) of each label at the declared
    count, predictive over the pilot's coverage, is at least 0.90."""
    low, high = study.STUDY.margins.calibration_coverage
    replicates = properties.CALIBRATION_REPLICATES
    for label, row in _pilot().iterrows():
        inside = floor_pass(row["coverage"], replicates, low, pilot.PILOT_DRAWS) - (
            1.0 - ceiling_pass(row["coverage"], replicates, high, pilot.PILOT_DRAWS)
        )
        assert inside >= tools.CELL_TARGET, (label, inside)


def primary_pass(replicates: int) -> float:
    """The product over the nine parameters of the bias leg (99% interval inside 0.25 SDs) and
    the coverage floor (0.90), each predictive over the pilot (the SE sanity band, 0.80 to 1.20,
    cannot fail at the pilot's ratios)."""
    floor = study.STUDY.margins.coverage_floor
    family = 1.0
    for _, row in _pilot().iterrows():
        family *= bias_pass(row["standardized_bias"], replicates, pilot.PILOT_DRAWS) * floor_pass(
            row["coverage"], replicates, floor, pilot.PILOT_DRAWS
        )
    return family


def test_the_primary_count_is_the_smallest_passing_multiple_of_400() -> None:
    """The smallest multiple of 400, at least the floor of 1,000, at which the nine parameters
    pass together with probability at least 0.95: the bias legs bind, at standardized biases up
    to 0.145 (0.957 at 3,200; 0.935 at 2,800)."""
    replicates = study.PRIMARY_REPLICATES
    assert replicates >= 1_000 and replicates % 400 == 0
    assert primary_pass(replicates) >= tools.FAMILY_TARGET
    assert primary_pass(replicates - 400) < tools.FAMILY_TARGET


def test_no_label_regret_reaches_the_bias_margin() -> None:
    """Plan section 5: a label whose regret at n = 2,000 reaches the bias margin (0.25 SDs) is a
    design finding before the run. The pilot's mean regret in efficient SEs stays below it."""
    for label, row in _pilot().iterrows():
        se = study.LIMITS[label]["bound"] / np.sqrt(study.PRIMARY_N)
        assert abs(row["mean_regret"]) / se < study.STUDY.margins.standardized_bias, label


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    modules = [m for m, _ in _x11_modules() if m.STUDY.slug != study.STUDY.slug]
    for module in [*modules, boundary, optimal]:
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others
    assert pilot.PILOT_SEED not in others | {study.SEED, study.RESAMPLING_SEED}


def test_rule_l3_keeps_every_label() -> None:
    assert set(properties.suffixes(study.STUDY).values()) == {()}


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
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal
    from tests.studies import learned_rule_optimal_value_properties as optimal_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for module, module_properties in [*_x11_modules(), (optimal, optimal_properties)]:
        if module.STUDY.slug == study.STUDY.slug:
            continue
        seeds |= base.declared_sample_seeds(
            module.STUDY,
            module.PRIMARY_REPLICATES,
            module_properties.PROPERTY_DRAWS,
            module_properties.suffixes(module.STUDY),
        )
    return seeds


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
    assert replicate_seed(study.STUDY, "weighted", 0) in declared


def test_no_pilot_seed_is_a_declared_seed() -> None:
    declared = learned_rule_run.declared_sample_seeds()
    drawn = {
        pilot.sample_seed(setting, r) for setting in law.SETTINGS for r in range(pilot.PILOT_DRAWS)
    }
    assert len(drawn) == len(law.SETTINGS) * pilot.PILOT_DRAWS
    assert not drawn & declared
