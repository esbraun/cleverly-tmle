"""The declared constants of the ``learned-rule-strata`` study, recomputed or pinned.

Each number here was fixed before any registered run. The limit-rule values, the efficiency
bounds and the asymptotic SE ratios of the share-term control are recomputed by quadrature or by
an independent adaptive integral; the pilot readings and the sizing rule that reads them are
pinned; and the study's seeds are checked against every registered study and every other
declared X11 study. No check draws a sample except the two seed-prefix witnesses.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from tests.canonical import learned_rule_run
from tests.studies import _fold_evaluated_strata_law as strata
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_strata_law as law
from tests.studies import fold_evaluated_strata as e1
from tests.studies import learned_rule_strata as study
from tests.studies import learned_rule_strata_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

# ------------------------------------------------------------------------------ the law


def test_the_law_is_e1s_with_a_steeper_binary_blip() -> None:
    """The bounded settings are E1's, drawn by E1's own function; the binary settings repeat
    E1's binary draw with the blip ``-1.2 + 3.0 W1`` and differ from E1's sample only in the
    outcome of treated rows."""
    assert (
        law.SETTINGS
        == tuple(study.SCENARIOS)
        == (
            "unweighted",
            "weighted",
            "binary_steep",
            "binary_steep_weighted",
        )
    )
    assert law.BOUNDED_SETTINGS == strata.BOUNDED_SETTINGS
    assert law.N_FOLDS == strata.N_FOLDS == 5
    assert law.SHARES == strata.SHARES and law.STRATA == strata.STRATA
    for setting in law.BOUNDED_SETTINGS:
        assert law.LAWS[setting] is strata.LAWS[setting]
    frame, truth = study.draw_from_seed("weighted", 300, 17)
    assert truth == {}
    np.testing.assert_array_equal(frame.to_numpy(), strata.draw("weighted", 300, 17).to_numpy())
    assert {s: (v.outcome, v.weighted, v.intercept, v.slope) for s, v in law.LAWS.items()} == {
        "unweighted": ("beta", False, 0.1, 1.5),
        "weighted": ("beta", True, 0.1, 1.5),
        "binary_steep": ("binary", False, -1.2, 3.0),
        "binary_steep_weighted": ("binary", True, -1.2, 3.0),
        "binary_weak": ("binary", False, 0.1, -0.3),
        "binary_moderate": ("binary", False, -0.6, 1.5),
    }
    assert tuple(law.LAWS)[:4] == law.SETTINGS
    assert tuple(law.DIAGNOSTIC_LAWS) == tuple(law.LAWS)[4:]
    ours = law.draw("binary_steep", 2_000, 17)
    e1s = strata.draw("binary_unweighted", 2_000, 17)
    for column in ("W1", "W2", "S", "A", strata.WEIGHT):
        np.testing.assert_array_equal(ours[column], e1s[column])
    control = ours["A"] == 0.0
    np.testing.assert_array_equal(ours.loc[control, "Y"], e1s.loc[control, "Y"])
    assert (ours.loc[~control, "Y"] != e1s.loc[~control, "Y"]).any()
    np.testing.assert_array_equal(
        law.draw("binary_steep_weighted", 300, 5), law.draw("binary_steep", 300, 5)
    )


def test_each_outcome_law_puts_the_rule_boundary_in_its_own_stratum() -> None:
    """The bounded blip ``0.1 + 1.5 W1`` changes sign at ``-1/15``, inside stratum 1; the
    binary blip ``-1.2 + 3.0 W1`` at ``0.4``, inside stratum 2. The slopes, 1.5 and 3.0, are ten
    and twenty times RM30's ``weak_blip``, so the boundary has probability zero and a margin
    (not exceptional)."""
    for setting in law.SETTINGS:
        spec = law.LAWS[setting]
        bounded = setting in law.BOUNDED_SETTINGS
        assert law.rule_root(setting) == pytest.approx(-1.0 / 15.0 if bounded else 0.4)
        boundary = law.boundary_stratum(setting)
        assert boundary == (1 if bounded else 2)
        assert spec.slope == (1.5 if bounded else 3.0)
        low, high = strata.EDGES[boundary], strata.EDGES[boundary + 1]
        assert low + 0.1 < law.rule_root(setting) < high - 0.1


def test_the_binary_stratum_arms_the_rule_reads_are_not_sparse() -> None:
    """Under the limit rule a binary stratum reads the control arm below ``W1 = 0.4`` and the
    treated arm above it, each with a mean between 0.35 and 0.93 at every point, so the rarer
    outcome value is common in every stratum-arm the target reads."""
    for setting in ("binary_steep", "binary_steep_weighted"):
        spec = law.LAWS[setting]
        for code in law.STRATA:
            points = law.grid(code)
            for w2 in base.W2_LEVELS:
                arm = law.limit_rule(setting)(points, w2).astype(float)
                mean = strata.qbar0(arm, points, w2, spec.slope, spec.intercept)
                assert float(np.min(mean)) > 0.35 and float(np.max(mean)) < 0.93


#: The value of the limit rule ``d0 = I(b(W1) > 0)`` in each stratum and marginally.
LIMIT_VALUES = {
    "unweighted": (0.4139218, 0.4887563, 0.7639604, 0.5976313),
    "weighted": (0.3983710, 0.4728142, 0.7529272, 0.5839764),
    "binary_steep": (0.4139218, 0.4814102, 0.6932779, 0.5632530),
    "binary_steep_weighted": (0.3983710, 0.4654316, 0.6808181, 0.5489434),
}


def _adaptive(function: object, low: float, high: float) -> float:
    return float(quad(function, low, high, epsabs=1e-13, epsrel=1e-12, limit=200)[0])


@pytest.mark.parametrize("setting", law.SETTINGS)
def test_the_limit_values_are_the_quadrature_and_an_adaptive_integral(setting: str) -> None:
    """The grid quadrature of a rule that jumps inside a stratum is accurate to about
    ``1e-5``, which an adaptive integral split at the root confirms."""
    spec = law.LAWS[setting]
    found = law.limit_moments(setting)
    assert (*found.stratum_values, found.value) == pytest.approx(LIMIT_VALUES[setting], abs=5e-8)
    weights = law._weights(setting)
    root = law.rule_root(setting)
    for code in law.STRATA:
        low, high = strata.EDGES[code], strata.EDGES[code + 1]
        cuts = [low, *([root] if low < root < high else []), high]
        total = 0.0
        for w2, w in weights.items():
            for a, b in itertools.pairwise(cuts):
                arm = float(strata.blip(0.5 * (a + b), spec.slope, spec.intercept) > 0.0)
                total += w * _adaptive(
                    lambda x, w2=w2, arm=arm: float(
                        strata.qbar0(arm, x, w2, spec.slope, spec.intercept)
                    ),
                    a,
                    b,
                )
        exact = total / ((high - low) * sum(weights.values()))
        assert found.stratum_values[code] == pytest.approx(exact, abs=2e-5)


#: The E1 setting with the same outcome law and weight, whose arm means and bounds this study's
#: limit values equal in a stratum the limit rule treats alike. The control arm does not read
#: the blip, so a binary setting's untreated strata read E1's binary ``ey[0]``.
E1_TWIN = {
    "unweighted": "unweighted",
    "weighted": "weighted",
    "binary_steep": "binary_unweighted",
    "binary_steep_weighted": "binary_weighted",
}


@pytest.mark.parametrize("setting", law.SETTINGS)
def test_away_from_the_boundary_the_limit_value_is_an_arm_mean_of_e1(setting: str) -> None:
    """In a stratum the limit rule treats alike, its value is E1's arm mean there; in the
    boundary stratum it exceeds both arm means, the nonzero witness that the rule is read."""
    truths = strata.truths(E1_TWIN[setting])
    found = law.limit_moments(setting)
    boundary = law.boundary_stratum(setting)
    rule = law.limit_rule(setting)
    for code in law.STRATA:
        arms = (truths[f"ey[0][S={code}]"], truths[f"ey[1][S={code}]"])
        if code == boundary:
            assert found.stratum_values[code] > max(arms) + 0.002
            continue
        treat = bool(rule(np.array([0.5 * (strata.EDGES[code] + strata.EDGES[code + 1])]), 0.0)[0])
        assert found.stratum_values[code] == pytest.approx(arms[int(treat)], abs=1e-12)


@pytest.mark.parametrize("setting", law.SETTINGS)
def test_the_efficiency_bounds_are_the_curve_sds_at_the_limit_rule(setting: str) -> None:
    """The declared bounds by the module's quadrature; in a stratum the limit rule treats
    alike, the bound is E1's bound of that arm, a second route."""
    for label in law.PARAMETER_LABELS:
        bound = study.EFFICIENCY_BOUNDS[f"{setting}_{label}"]
        assert law.efficiency_bound(setting, label) == pytest.approx(bound, abs=5e-7)
    boundary = law.boundary_stratum(setting)
    rule = law.limit_rule(setting)
    for code in law.STRATA:
        if code == boundary:
            continue
        treat = bool(rule(np.array([0.5 * (strata.EDGES[code] + strata.EDGES[code + 1])]), 0.0)[0])
        arm = "ey1" if treat else "ey0"
        assert study.EFFICIENCY_BOUNDS[f"{setting}_s{code}"] == pytest.approx(
            e1.EFFICIENCY_BOUNDS[f"{E1_TWIN[setting]}_{arm}_s{code}"], abs=1e-6
        )


#: The asymptotic SE ratio of the marginal curve without its share term, at the limit rule.
SHARE_TERM_RATIOS = {
    "unweighted": 0.778441,
    "weighted": 0.772545,
    "binary_steep": 0.983345,
    "binary_steep_weighted": 0.982464,
}


def test_the_share_term_ratios_choose_the_declared_controls() -> None:
    """The bounded share terms resolve below the band at the declared count; the binary ones
    are above the floor, so no binary label carries the control (E1's finding, again)."""
    half = tools.Z * tools.C_FIT / np.sqrt(2.0 * properties.CALIBRATION_REPLICATES)
    for setting, ratio in SHARE_TERM_RATIOS.items():
        assert law.share_term_ratio(setting) == pytest.approx(ratio, abs=5e-7)
        declared = f"{setting}_all" in study.SHARE_DROPPED_LABELS
        assert declared == (setting in law.BOUNDED_SETTINGS)
        if declared:
            assert ratio + half < 0.93 - 0.1
        else:
            assert ratio > 0.93
    assert study.SHARE_DROPPED_LABELS == ("unweighted_all", "weighted_all")


# --------------------------------------------------------------------------- the cells


def test_the_cells() -> None:
    calibration = study.PROPERTY_CELLS["interval_calibration"]
    assert len(study.CALIBRATION_LABELS) == 16
    assert len(study.STRATUM_LABELS) == 12
    assert len(calibration) == 16 * 3 + 12 + 2 == len(set(calibration))
    assert set(study.PROPERTY_CELLS) == {
        "interval_calibration",
        "targeting_necessity",
        study.WEAK_FAMILY,
    }
    assert study.PROPERTY_CELLS[study.WEAK_FAMILY] == tuple(
        f"{name}_{label}__diagnostic"
        for name in ("binary_weak", "binary_moderate")
        for label in law.PARAMETER_LABELS
    )
    assert study.NECESSITY_LABELS == ("unweighted_s0", "unweighted_s1", "unweighted_s2")
    assert len(study.PROPERTY_CELLS["targeting_necessity"]) == 6
    assert set(study.EFFICIENCY_BOUNDS) == set(study.CALIBRATION_LABELS)
    from tests.studies.evidence.property_verdicts import SE_CONTROL_SIDES

    kinds = {cell.rsplit("__", 1)[1] for cell in calibration}
    assert kinds == {"declared", "shrunken_se_control", "noise_control", *properties.OWN_CONTROLS}
    assert all(SE_CONTROL_SIDES[kind] == "below" for kind in properties.OWN_CONTROLS)
    assert study.STUDY.truth_varies_by_replicate
    assert study.STUDY.extra_artifacts == (study.HARNESS_ARTIFACT,)
    # PROPERTY_RULES names a rule for every family that forms a verdict.
    families = {family for rule in properties.PROPERTY_RULES for family in rule.families}
    assert families >= set(study.PROPERTY_CELLS) - {study.WEAK_FAMILY}
    assert study.WEAK_FAMILY not in families


def test_the_parameter_names_are_the_fits() -> None:
    assert law.parameter("all") == "ey_learned_rule[learned rule]"
    assert law.parameter("s2") == "ey_learned_rule[learned rule][S=2]"
    assert tuple(study.SCENARIOS["unweighted"]) == law.PARAMETERS


def test_the_fit_is_the_declared_one() -> None:
    for setting in law.SETTINGS:
        estimator = law.estimator(5, setting)
        assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
        assert estimator.cross_fit and estimator.cv_evaluation
        assert estimator.n_folds == 5
        assert estimator.targeting_scheme == "pooled"
        assert estimator.g_bounds == "auto"
        binary = law.LAWS[setting].outcome == "binary"
        assert estimator.q_bounds == (None if binary else (0.0, 1.0))
        assert estimator.simultaneous is False
        assert estimator.repeats == 1
        assert estimator.random_state == 5
    assert study.PRIMARY_N == properties.CALIBRATION_N == 1_000


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/_strata.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/learners/crossfit.py",
        "tests/studies/_learned_rule_strata_law.py",
        "tests/studies/_fold_evaluated_strata_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/fractional_glm.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    """Every registered study but this one, which the register now holds too."""
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_split_mean as split_mean

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module in (e1, learners, split_mean):
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others


def test_the_named_streams_are_distinct_from_every_registered_study() -> None:
    """The labels ``test_every_named_stream_of_every_study_is_distinct`` reads, on this record
    and on every registered study, E1's included."""
    labels = [
        ("root_n_rate", "empirical_sd"),
        ("root_n_rate", "reported_se"),
        ("interval_calibration", "correctly_specified"),
        ("crossfit_overfitting", "in_sample_control"),
        ("crossfit_overfitting", "coverage_gain"),
        ("performance", "cleverly", "binary", "ate"),
        ("equivalence", "binary", "ate"),
    ]
    records = [*_other_studies(), study.STUDY]
    seeds = [stream_seed(record, *label) for record in records for label in labels]
    assert len(set(seeds)) == len(seeds)


#: The rule-L3 suffix of every property draw label (a weighted setting is judged against its
#: rows-only twin's sample law).
DECLARED_SUFFIXES: dict[tuple[str, str], tuple[object, ...]] = {
    ("interval_calibration", "unweighted"): (),
    ("interval_calibration", "weighted"): (),
    ("interval_calibration", "binary_steep"): ("retry", 1),
    ("interval_calibration", "binary_steep_weighted"): ("retry", 1),
    ("targeting_necessity", "unweighted"): (),
    ("weak_margin_diagnostic", "binary_weak"): (),
    ("weak_margin_diagnostic", "binary_moderate"): (),
}


def test_rule_l3_moves_the_two_binary_labels() -> None:
    """Under the bare label, calibration ``binary_steep`` replicate 10,426 would repeat its own
    replicate 5,152, so rule L3 moves the label to its first retry; then the bare
    ``binary_steep_weighted`` replicate 13,742 would repeat the moved ``binary_steep`` replicate
    4,555 (same sample law), so it moves too. The other labels keep theirs."""
    assert properties.SAMPLE_LAW == {
        "unweighted": "unweighted",
        "weighted": "unweighted",
        "binary_steep": "binary_steep",
        "binary_steep_weighted": "binary_steep",
    }
    assert properties.suffixes(study.STUDY) == DECLARED_SUFFIXES
    label = ("property_sample", "interval_calibration")
    assert stream_seed(study.STUDY, *label, "binary_steep", 10_426) == stream_seed(
        study.STUDY, *label, "binary_steep", 5_152
    )
    assert stream_seed(study.STUDY, *label, "binary_steep_weighted", 13_742) == stream_seed(
        study.STUDY, *label, "binary_steep", 4_555, "retry", 1
    )


def _own_sample_seeds() -> set[int]:
    return base.declared_sample_seeds(
        study.STUDY,
        study.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(study.STUDY),
    )


#: The one pair of this study's own draws that share a sample seed: rule L3 judges a primary
#: scenario by its name, and primary ``weighted`` draws the rows-only law, so its replicate 642
#: draws the sample of calibration ``unweighted`` replicate 4,245. The two fits differ (one
#: reads the weight), each is one draw of 2,000 or 14,800, and no verdict reads the two cells
#: jointly.
OWN_SHARED = (("primary", "weighted", 642), ("interval_calibration", "unweighted", 4_245))

#: The sample seeds this study shares with an earlier X11 study: none. (At the first declared
#: necessity count, 8,000, two necessity replicates shared a ``W1`` prefix with
#: learned-rule-cvtmle and learned-rule-learners; at 3,600 neither is drawn.)
#: ``learned-rule-stacked`` (X11 PR F1), declared after this study, shares two: its ``rows``
#: calibration replicate 1,784 (RM30's law at n = 2,000, so the first 1,000 values of ``W1``)
#: and its ``clustered`` calibration replicate 7,192 (whose first draws are cluster effects, so
#: no variable) (``test_learned_rule_stacked_design``).
SHARED_SEEDS: dict[tuple[str, str, int], tuple[str, str, int]] = {
    ("interval_calibration", "binary_steep", 9_841): ("learned-rule-stacked", "rows", 1_784),
    ("interval_calibration", "weighted", 4_518): ("learned-rule-stacked", "clustered", 7_192),
}


def _seed_of(family: str, setting: str, replicate: int) -> int:
    if family == "primary":
        return replicate_seed(study.STUDY, setting, replicate)
    resolved = properties.suffixes(study.STUDY)
    return stream_seed(
        study.STUDY, "property_sample", family, setting, replicate, *resolved[family, setting]
    )


def test_the_sample_seeds_are_distinct_but_one_declared_pair() -> None:
    own = _own_sample_seeds()
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert len(own) == total - 1
    first, second = OWN_SHARED
    assert _seed_of(*first) == _seed_of(*second)
    np.testing.assert_array_equal(
        law.draw("weighted", 1_000, _seed_of(*first)),
        law.draw("unweighted", 1_000, _seed_of(*second)),
    )


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_compositions as compositions
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_cvtmle_properties as gated_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_split_mean_properties as split_mean_properties
    from tests.studies import learned_rule_stacked as f1
    from tests.studies import learned_rule_stacked_properties as f1_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for record, module in (
        (gated, gated_properties),
        (compositions, composition_properties),
        (split_mean, split_mean_properties),
        (learners, learner_properties),
        (e1, e1_properties),
        (f1, f1_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


def test_no_sample_seed_is_another_x11_studys() -> None:
    """The study shares no sample seed with an earlier X11 study, two with the later
    ``learned-rule-stacked``, and the run form lists it.

    ``learned_rule_run.declared_sample_seeds()`` is the union of this study's seeds and every
    other X11 study's, E1's among them (design record, D2), so a smoke run of any of them
    avoids this study's seeds.
    """
    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == {_seed_of(*key) for key in SHARED_SEEDS}
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    assert not seeds & learned_rule_run.declared_sample_seeds()
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, "unweighted", 0) in learned_rule_run.declared_sample_seeds()


# ---------------------------------------------------------------------------- the sizing

#: The pilot's SE ratio of each calibration label, in ``CALIBRATION_LABELS`` order, at
#: n = 1,000 (``reviews/x11/scripts/X11-E2-calibration-pilot.py`` in the plans repository,
#: throwaway label 20266011, 4,000 draws per setting: cells 0 and 1 for the bounded settings,
#: cells 7 and 8 for the declared binary blip). Each ratio is the mean reported SE over the SD
#: of ``estimate - truth``. The superseded binary blips (cells 2, 3, 5 and 6) are not read.
PILOT_DRAWS = 4_000
PILOT_SE_RATIOS = (
    1.0258, 1.0052, 0.9916, 0.9976, 1.0044, 0.9894, 1.0076, 0.9924,
    1.0164, 1.0146, 1.0035, 1.0017, 1.0123, 1.0077, 0.9731, 0.9962,
)  # fmt: skip

#: The pooled necessity pilot's standardized bias of the targeted arm and of the untargeted
#: control (in the control's own SD) for each label: 2,000 draws under label 20266011 (cell 4)
#: and 4,000 under label 20267401 (cell 9, design review finding 3), 6,000 in all.
NECESSITY_PILOT_DRAWS = 6_000
PILOT_NECESSITY = {
    "unweighted_s0": (0.1637, 12.76),
    "unweighted_s1": (0.0244, 8.58),
    "unweighted_s2": (-0.0555, -12.90),
}


def test_the_calibration_count_is_the_per_arm_rule() -> None:
    """``per_arm_replicates`` over the 16 positives at the pilot's size, 4,000 draws, gives
    14,800; 14,400 misses the family target. The largest ratio, 1.0258 (bounded stratum 0),
    sets most of it: without it the rule gives 12,000."""
    assert len(PILOT_SE_RATIOS) == len(study.CALIBRATION_LABELS)
    sized = tools.per_arm_replicates(PILOT_SE_RATIOS, PILOT_DRAWS)
    assert sized["R"] == properties.CALIBRATION_REPLICATES == 14_800
    assert sized["family_wise"] == pytest.approx(0.9525, abs=5e-4)
    assert min(sized["per_cell"]) == pytest.approx(0.9778, abs=5e-4)
    below = tools.per_arm_pass(PILOT_SE_RATIOS, PILOT_DRAWS, 14_400)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    assert tools.per_arm_replicates((1.0, *PILOT_SE_RATIOS[1:]), PILOT_DRAWS)["R"] == 12_000
    # Every arm passes the per-arm screen, so none is diagnosed alone.
    half = tools.screen_half_width(PILOT_DRAWS)
    assert all(abs(ratio - 1.0) <= half for ratio in PILOT_SE_RATIOS)


def _necessity_power(replicates: int, shift: float) -> float:
    """The product of the targeted arms' equivalence power, each bias moved ``shift`` away
    from zero."""
    return float(
        np.prod(
            [
                tools.equivalence_power(bias + np.sign(bias) * shift, replicates)
                for bias, _ in PILOT_NECESSITY.values()
            ]
        )
    )


def test_the_necessity_count_is_the_declared_rule_on_the_pooled_pilot() -> None:
    """The smallest multiple of 400 whose equivalence-power product is at least 0.95 at the
    pooled biases and at one pooled pilot SE further from zero: 3,600. 3,200 misses. Each
    control's bias, at least 8.5 own SDs, is established outside the margin surely."""
    assert set(PILOT_NECESSITY) == set(study.NECESSITY_LABELS)
    shift = 1.0 / np.sqrt(NECESSITY_PILOT_DRAWS)
    replicates = properties.NECESSITY_REPLICATES
    assert replicates == 3_600
    assert _necessity_power(replicates, 0.0) >= 0.95
    assert _necessity_power(replicates, shift) >= 0.95
    assert _necessity_power(replicates - 400, shift) < 0.95
    # The first pilot's stratum-0 bias, one of its SEs higher, would need more than 6,000.
    assert tools.equivalence_power(0.199, 6_000) < 0.95
    assert all(
        tools.necessity_power(control, replicates) > 0.9999
        for _, control in PILOT_NECESSITY.values()
    )
