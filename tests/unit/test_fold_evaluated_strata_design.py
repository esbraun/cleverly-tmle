"""The declared constants of the ``fold-evaluated-strata`` study, recomputed or pinned.

Each number here was fixed before any registered run. The truths, the stratum shares, the
efficiency bounds and the asymptotic SE ratios of the share-term control are recomputed by
quadrature or by an independent adaptive integral; the pilot readings and the sizing rule that
reads them are pinned; and the study's seeds are checked against every registered study and every
other declared X11 study. No check draws a sample.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from tests.canonical import learned_rule_run
from tests.studies import _fold_evaluated_strata_law as law
from tests.studies import _learned_rule_law as base
from tests.studies import fold_evaluated_strata as study
from tests.studies import fold_evaluated_strata_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

# ------------------------------------------------------------------------------ the truth

#: The declared truth of every parameter, by setting (``law.truths``).
TRUTHS = {
    "unweighted": {
        "ate[S=0]": -0.2203454,
        "ate[S=1]": -0.0643461,
        "ate[S=2]": 0.1841802,
        "ate": 0.0162909,
        "ey[0][S=0]": 0.4139218,
        "ey[0][S=1]": 0.4814102,
        "ey[0][S=2]": 0.5797803,
        "ey[0]": 0.5121791,
        "ey[1][S=0]": 0.1935764,
        "ey[1][S=1]": 0.4170641,
        "ey[1][S=2]": 0.7639604,
        "ey[1]": 0.5284699,
    },
    "weighted": {
        "ate[S=0]": -0.2147088,
        "ate[S=1]": -0.0633777,
        "ate[S=2]": 0.1887141,
        "ate": 0.0197974,
        "ey[0][S=0]": 0.3983710,
        "ey[0][S=1]": 0.4654316,
        "ey[0][S=2]": 0.5642131,
        "ey[0]": 0.4964712,
        "ey[1][S=0]": 0.1836621,
        "ey[1][S=1]": 0.4020539,
        "ey[1][S=2]": 0.7529272,
        "ey[1]": 0.5162685,
    },
    # The descriptive sparse-event law: binary, rows only, blip 0.1 + 3.0 W1.
    "binary_sparse": {
        "ate[S=0]": -0.3429242,
        "ate[S=1]": -0.1375448,
        "ate[S=2]": 0.2771145,
        "ate": 0.0079760,
        "ey[0][S=0]": 0.4139218,
        "ey[0][S=1]": 0.4814102,
        "ey[0][S=2]": 0.5797803,
        "ey[0]": 0.5121791,
        "ey[1][S=0]": 0.0709976,
        "ey[1][S=1]": 0.3438654,
        "ey[1][S=2]": 0.8568948,
        "ey[1]": 0.5201551,
    },
}
# The truth reads the mean model only, so each binary setting's truths are its bounded twin's.
TRUTHS["binary_unweighted"] = TRUTHS["unweighted"]
TRUTHS["binary_weighted"] = TRUTHS["weighted"]

#: The asymptotic SE ratio of the marginal curve without its share term, by setting and target.
SHARE_TERM_RATIOS = {
    "unweighted": {"ate": 0.854196, "ey0": 0.947636, "ey1": 0.655980},
    "weighted": {"ate": 0.853715, "ey0": 0.945242, "ey1": 0.660007},
    "binary_unweighted": {"ate": 0.985391, "ey0": 0.995435, "ey1": 0.942633},
    "binary_weighted": {"ate": 0.985314, "ey0": 0.995271, "ey1": 0.944234},
}


def _adaptive(function: object, low: float, high: float) -> float:
    return float(quad(function, low, high, epsabs=1e-13, epsrel=1e-12, limit=200)[0])


def _functional(target: str, w1: float, w2: float, slope: float = law.SLOPE) -> float:
    q1, q0 = float(law.qbar0(1.0, w1, w2, slope)), float(law.qbar0(0.0, w1, w2, slope))
    return {"ate": q1 - q0, "ey0": q0, "ey1": q1}[target]


def test_the_law_is_the_rm30_family_with_a_beta_outcome() -> None:
    """Same covariates, treatment mechanism, outcome-regression form and draw order as RM30;
    the outcome is a beta proportion with that mean, at the bounded laws' concentration."""
    from tests.studies.bounded_cv_laws import CONCENTRATION, MINIMUM_BETA_SHAPE, Q_BOUNDS

    w1 = np.linspace(-1.0, 1.0, 9)
    for w2 in base.W2_LEVELS:
        np.testing.assert_allclose(
            law.qbar0(0.0, w1, w2), base.qbar0(0.0, w1, w2, "non_exceptional"), rtol=1e-15
        )
        expected = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * w1 - 0.3 * w2 + 0.1 + 1.5 * w1)))
        np.testing.assert_allclose(law.qbar0(1.0, w1, w2), expected, rtol=1e-13)
    assert law.PHI == CONCENTRATION and law.OUTCOME_BOUNDS == Q_BOUNDS
    grid = np.linspace(-1.0, 1.0, 4001)
    means = np.concatenate([law.qbar0(a, grid, w2) for a in (0.0, 1.0) for w2 in base.W2_LEVELS])
    smallest = law.PHI * float(np.min(np.minimum(means, 1.0 - means)))
    assert smallest == pytest.approx(1.0930, abs=5e-4) and smallest > MINIMUM_BETA_SHAPE
    rm30 = base.draw("non_exceptional", 500, 17)
    for setting, spec in law.LAWS.items():
        frame = law.draw(setting, 500, 17)
        for column in ("W1", "W2", "A"):
            np.testing.assert_array_equal(frame[column], rm30[column])
        if spec.outcome == "binary":
            assert set(np.unique(frame["Y"])) == {0.0, 1.0}
        else:
            assert frame["Y"].min() > 0.0 and frame["Y"].max() < 1.0
        np.testing.assert_array_equal(frame["S"], np.digitize(frame["W1"], [-0.6, 0.1]))
        np.testing.assert_array_equal(frame[law.WEIGHT], 1.0 + 1.5 * frame["W2"])


def test_the_settings() -> None:
    """Four calibration settings: the outcome law crossed with the weight, at the declared
    mean model; and the sparse-event diagnostic, binary and rows only at the steeper blip."""
    assert law.SETTINGS == ("unweighted", "weighted", "binary_unweighted", "binary_weighted")
    assert law.BOUNDED_SETTINGS == ("unweighted", "weighted")
    assert {s: (v.outcome, v.weighted, v.slope) for s, v in law.LAWS.items()} == {
        "unweighted": ("beta", False, 1.5),
        "weighted": ("beta", True, 1.5),
        "binary_unweighted": ("binary", False, 1.5),
        "binary_weighted": ("binary", True, 1.5),
        "binary_sparse": ("binary", False, 3.0),
    }
    w1 = np.linspace(-1.0, 1.0, 9)
    for w2 in base.W2_LEVELS:
        expected = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * w1 - 0.3 * w2 + 0.1 + 3.0 * w1)))
        np.testing.assert_allclose(law.qbar0(1.0, w1, w2, law.SPARSE_SLOPE), expected, rtol=1e-13)
    assert set(study.SCENARIOS) == set(law.SETTINGS)


def test_the_shares_are_unequal_and_untouched_by_the_weight() -> None:
    lengths = np.diff(law.EDGES) / 2.0
    np.testing.assert_allclose(lengths, law.SHARES, rtol=0, atol=1e-15)
    assert len(set(law.SHARES)) == 3
    # S reads W1 only and the weight W2 only, and they are independent.
    for setting in law.LAWS:
        weights = law._weights(setting)
        tilted = [sum(0.5 * w * share for w in weights.values()) for share in law.SHARES]
        np.testing.assert_allclose(np.array(tilted) / sum(tilted), law.SHARES, atol=1e-15)


@pytest.mark.parametrize("setting", law.LAWS)
def test_the_truths_are_the_quadrature_and_an_adaptive_integral(setting: str) -> None:
    declared = law.truths(setting)
    for name, value in TRUTHS[setting].items():
        assert declared[name] == pytest.approx(value, abs=5e-8), name
    weights = law._weights(setting)
    slope = law.LAWS[setting].slope
    for target, name in law.TARGETS.items():
        parts = []
        for code in law.STRATA:
            low, high = law.EDGES[code], law.EDGES[code + 1]
            numerator = sum(
                w
                * _adaptive(
                    lambda x, w2=w2, target=target: _functional(target, x, w2, slope), low, high
                )
                for w2, w in weights.items()
            )
            exact = numerator / ((high - low) * sum(weights.values()))
            parts.append(exact)
            assert declared[f"{name}[S={code}]"] == pytest.approx(exact, abs=1e-8)
        # The marginal truth is the share mixture, and it is the whole-support integral too.
        whole = sum(
            w
            * _adaptive(
                lambda x, w2=w2, target=target: _functional(target, x, w2, slope), -1.0, 1.0
            )
            for w2, w in weights.items()
        ) / (2.0 * sum(weights.values()))
        mixture = [declared[f"{name}[S={code}]"] for code in law.STRATA]
        assert declared[name] == pytest.approx(float(np.dot(law.SHARES, mixture)), abs=1e-15)
        assert declared[name] == pytest.approx(float(np.dot(law.SHARES, parts)), abs=1e-8)
        assert declared[name] == pytest.approx(whole, abs=1e-8)


def test_the_binary_truths_are_the_bounded_ones() -> None:
    """The two outcome laws share the mean model, so their truths agree exactly."""
    assert law.truths("binary_unweighted") == law.truths("unweighted")
    assert law.truths("binary_weighted") == law.truths("weighted")


def test_the_stratum_effects_are_strongly_heterogeneous() -> None:
    """The stratum ATEs change sign and spread by more than 0.4, which the share term reads."""
    for setting in law.LAWS:
        effects = [TRUTHS[setting][f"ate[S={code}]"] for code in law.STRATA]
        assert effects[0] < effects[1] < 0.0 < effects[2]
        assert effects[2] - effects[0] > 0.4


@pytest.mark.parametrize("setting", law.SETTINGS)
def test_the_efficiency_bounds_are_the_curve_sds(setting: str) -> None:
    """The declared bounds by the module's quadrature, and the marginal ones by a second route:
    the whole-support integral of the unstratified efficient curve."""
    for label in law.PARAMETER_LABELS:
        bound = study.EFFICIENCY_BOUNDS[f"{setting}_{label}"]
        assert law.efficiency_bound(setting, label) == pytest.approx(bound, abs=5e-7)
    weights = law._weights(setting)
    mean_weight = float(np.mean(list(weights.values())))
    for target, name in law.TARGETS.items():
        psi = law.truths(setting)[name]

        def second(x: float, w2: float, target: str = target, psi: float = psi) -> float:
            f, residual = law._target_parts(target, np.array([x]), w2, setting)
            return float(residual[0] + (f[0] - psi) ** 2)

        variance = sum(
            0.25 * (w / mean_weight) ** 2 * _adaptive(lambda x, w2=w2: second(x, w2), -1.0, 1.0)
            for w2, w in weights.items()
        )
        assert np.sqrt(variance) == pytest.approx(
            study.EFFICIENCY_BOUNDS[f"{setting}_{target}_all"], abs=5e-6
        )


@pytest.mark.parametrize("setting", law.BOUNDED_SETTINGS)
def test_the_share_term_ratios_choose_the_declared_controls(setting: str) -> None:
    """The ``ate`` and ``ey[1]`` share terms resolve below the band; ``ey[0]``'s does not.

    The share term carries ``1 - ratio^2`` of the marginal variance: about 27% for ``ate``.
    """
    for target, ratio in SHARE_TERM_RATIOS[setting].items():
        assert law.share_term_ratio(setting, target) == pytest.approx(ratio, abs=5e-7)
    assert 0.25 < 1.0 - SHARE_TERM_RATIOS[setting]["ate"] ** 2 < 0.30
    assert study.SHARE_DROPPED_LABELS == (
        "unweighted_ate_all",
        "unweighted_ey1_all",
        "weighted_ate_all",
        "weighted_ey1_all",
    )
    # A control passes when its 99% SE-ratio interval clears 0.93. At the declared count the
    # half width is 2.576 * 1.069 / sqrt(2 R); each declared control clears it with room for
    # the finite-sample excess the pilot measured, and ey[0] would straddle it.
    half = tools.Z * tools.C_FIT / np.sqrt(2.0 * properties.CALIBRATION_REPLICATES)
    for target in ("ate", "ey1"):
        assert SHARE_TERM_RATIOS[setting][target] + half < 0.93 - 0.04
    assert SHARE_TERM_RATIOS[setting]["ey0"] + half > 0.93


@pytest.mark.parametrize("setting", ("binary_unweighted", "binary_weighted"))
def test_no_binary_label_carries_a_share_dropped_control(setting: str) -> None:
    """Under the Bernoulli outcome the residual variance swamps the share term: every ratio
    without it is above the band floor 0.93, so a share-dropped control could not fail, and the
    binary settings declare none (finding 1 of the design review; its pilot read 0.974 and
    0.941 for ``ate`` and ``ey[1]``). The share term's formula does not read the outcome type,
    so the bounded settings test it for both."""
    for target, ratio in SHARE_TERM_RATIOS[setting].items():
        assert law.share_term_ratio(setting, target) == pytest.approx(ratio, abs=5e-7)
        assert ratio > 0.93
    assert not any(label.startswith(setting) for label in study.SHARE_DROPPED_LABELS)


# --------------------------------------------------------------------------- the cells


def test_the_cells() -> None:
    calibration = study.PROPERTY_CELLS["interval_calibration"]
    assert len(study.CALIBRATION_LABELS) == 48
    assert len(study.STRATUM_LABELS) == 36
    assert len(calibration) == 48 * 3 + 36 + 4 == len(set(calibration))
    assert study.PROPERTY_CELLS[study.SPARSE_FAMILY] == tuple(
        f"binary_sparse_{label}__diagnostic" for label in law.PARAMETER_LABELS
    )
    assert set(study.PROPERTY_CELLS) == {
        "interval_calibration",
        "targeting_necessity",
        study.SPARSE_FAMILY,
    }
    assert (
        tuple(f"unweighted_{target}_s{code}" for target in law.TARGETS for code in law.STRATA)
        == study.NECESSITY_LABELS
    )
    assert len(study.PROPERTY_CELLS["targeting_necessity"]) == 18
    assert set(study.EFFICIENCY_BOUNDS) == set(study.CALIBRATION_LABELS)
    # Every cell kind is a positive, a derived control or a below-band SE control.
    from tests.studies.evidence.property_verdicts import SE_CONTROL_SIDES

    kinds = {cell.rsplit("__", 1)[1] for cell in calibration}
    assert kinds == {"declared", "shrunken_se_control", "noise_control", *properties.OWN_CONTROLS}
    assert all(SE_CONTROL_SIDES[kind] == "below" for kind in properties.OWN_CONTROLS)


def test_the_fit_is_the_declared_one() -> None:
    for setting in law.LAWS:
        estimator = law.estimator(5, setting)
        assert estimator.cross_fit and estimator.cv_evaluation
        assert estimator.n_folds == 5 == law.N_FOLDS
        assert estimator.targeting_scheme == "pooled"
        assert estimator.g_bounds == "auto"
        binary = law.LAWS[setting].outcome == "binary"
        assert estimator.q_bounds == (None if binary else (0.0, 1.0))
        assert estimator.estimands == ("ate", "ey")
        assert estimator.simultaneous is False
        assert estimator.random_state == 5
    assert law.estimator(5, law.SPARSE, cv_evaluation=False).cv_evaluation is False
    assert study.PRIMARY_N == properties.CALIBRATION_N == properties.SPARSE_N == 1_000
    assert properties.SPARSE_REPLICATES == 2_000


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/_strata.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/learners/crossfit.py",
        "tests/studies/_fold_evaluated_strata_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/fractional_glm.py",
    ):
        assert module in study.STUDY.modules


# --------------------------------------------------------------------------- the seeds


def test_the_record_seeds_are_unused() -> None:
    others = {record.seed for record in registered()} | {
        record.resampling_seed for record in registered() if record.resampling_seed is not None
    }
    assert study.SEED not in others and study.RESAMPLING_SEED not in others
    from tests.studies import learned_rule_learners as learners

    assert {study.SEED, study.RESAMPLING_SEED}.isdisjoint({learners.SEED, learners.RESAMPLING_SEED})


def test_the_named_streams_are_distinct_from_every_registered_study() -> None:
    """The labels ``test_every_named_stream_of_every_study_is_distinct`` reads, on this record."""
    labels = [
        ("root_n_rate", "empirical_sd"),
        ("root_n_rate", "reported_se"),
        ("interval_calibration", "correctly_specified"),
        ("crossfit_overfitting", "in_sample_control"),
        ("crossfit_overfitting", "coverage_gain"),
        ("performance", "cleverly", "binary", "ate"),
        ("equivalence", "binary", "ate"),
    ]
    records = [*registered(), study.STUDY]
    seeds = [stream_seed(record, *label) for record in records for label in labels]
    assert len(set(seeds)) == len(seeds)


def test_rule_l3_moves_one_label() -> None:
    """Under the bare label, ``interval_calibration/unweighted`` replicate 4,935 would draw the
    sample seed of primary ``unweighted`` replicate 665, so rule L3 moves the whole label to its
    first retry; every other label keeps its own."""
    bare = stream_seed(study.STUDY, "property_sample", "interval_calibration", "unweighted", 4_935)
    assert bare == replicate_seed(study.STUDY, "unweighted", 665)
    assert properties.suffixes(study.STUDY) == {
        ("interval_calibration", "unweighted"): ("retry", 1),
        ("interval_calibration", "weighted"): (),
        ("interval_calibration", "binary_unweighted"): (),
        ("interval_calibration", "binary_weighted"): (),
        ("targeting_necessity", "unweighted"): (),
        (study.SPARSE_FAMILY, law.SPARSE): (),
    }


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and other X11 studies draws."""
    from tests.studies import learned_rule_compositions as compositions
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_cvtmle_properties as gated_properties
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_learners_properties as learner_properties
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_split_mean_properties as split_mean_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for record, primary, module in (
        (gated, gated.PRIMARY_REPLICATES, gated_properties),
        (compositions, compositions.PRIMARY_REPLICATES, composition_properties),
        (split_mean, split_mean.PRIMARY_REPLICATES, split_mean_properties),
        (learners, learners.PRIMARY_REPLICATES, learner_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY, primary, module.PROPERTY_DRAWS, module.suffixes(record.STUDY)
        )
    return seeds


#: The three sample seeds this study shares with another X11 study, all in ``binary_weighted``
#: draws at n = 1,000, each against an RM30-law draw at n = 2,000 or 4,000: (this study's draw,
#: the other draw). Seeds are 32-bit, and about 43,000 against 152,000 seeds expect one or two
#: coincidences by chance.
SHARED_SEEDS = {
    ("primary", "binary_weighted", 541): ("learned-rule-split-mean", "misspecified_limit", 1_507),
    ("interval_calibration", "binary_weighted", 524): (
        "learned-rule-split-mean",
        "interval_calibration/non_exceptional__declared",
        597,
    ),
    ("interval_calibration", "binary_weighted", 1_711): (
        "learned-rule-learners",
        "interval_calibration/weighted_classification__declared",
        6_697,
    ),
}


def test_no_sample_seed_is_another_x11_studys_but_three_w1_prefixes() -> None:
    """The study's samples share no seed with each other, and three with another X11 study.

    This law draws ``W1`` first, as RM30 does, so a shared seed shares a prefix of ``W1``. Each
    of the three pairs draws 1,000 rows here and 2,000 or 4,000 there, so the two samples share
    their first 1,000 values of ``W1`` and nothing else: ``W2`` starts after the ``W1`` block,
    whose length differs. That is the precedent of ``learned-rule-learners``, whose shared seed
    with ``learned-rule-compositions`` is the same kind (``test_learned_rule_study_harness``).
    One replicate of 2,000 or 7,600 in each cell makes the dependence negligible, and no verdict
    reads the two studies jointly.
    """
    own = base.declared_sample_seeds(
        study.STUDY,
        study.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(study.STUDY),
    )
    assert len(own) == len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    others = _other_x11_sample_seeds()
    resolved = properties.suffixes(study.STUDY)

    def seed_of(family: str, setting: str, replicate: int) -> int:
        if family == "primary":
            return replicate_seed(study.STUDY, setting, replicate)
        return stream_seed(
            study.STUDY, "property_sample", family, setting, replicate, *resolved[family, setting]
        )

    shared = {seed_of(*key) for key in SHARED_SEEDS}
    assert own & others == shared
    assert learned_rule_run.declared_sample_seeds() == own | others
    for key in SHARED_SEEDS:
        frame = law.draw(key[1], study.PRIMARY_N, seed_of(*key))
        rm30 = base.draw("non_exceptional", 2_000, seed_of(*key))
        np.testing.assert_array_equal(frame["W1"], rm30["W1"][: study.PRIMARY_N])
        assert not np.array_equal(frame["W2"], rm30["W2"][: study.PRIMARY_N])


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    assert not seeds & learned_rule_run.declared_sample_seeds()


# ---------------------------------------------------------------------------- the sizing

#: The pooled pilot's SE ratio of each calibration label, in ``CALIBRATION_LABELS`` order, at
#: n = 1,000 (``reviews/x11/scripts/X11-E1-calibration-pilot.py`` in the plans repository). The
#: bounded settings pool the declared pilot (label 20264099, 8,000 draws each) with the design
#: review's 2,000 further rows-only draws of the identical configuration (label 20264731). The
#: binary settings pool the binary pilot (label 20264907, 8,000 draws each) with, rows only, the
#: design review's 4,000 draws of the identical configuration (label 20264731).
PILOT_DRAWS = {
    "unweighted": 10_000,
    "weighted": 8_000,
    "binary_unweighted": 12_000,
    "binary_weighted": 8_000,
}
PILOT_SE_RATIOS = (
    1.0085, 0.9956, 0.9994, 0.9983, 1.0058, 1.0037, 1.0024, 1.0005, 1.0050, 0.9853, 1.0037, 1.0027,
    1.0159, 1.0065, 1.0167, 1.0046, 1.0043, 1.0139, 1.0072, 1.0099, 1.0103, 1.0003, 1.0082, 0.9897,
    1.0058, 1.0042, 1.0110, 1.0051, 1.0091, 0.9999, 1.0160, 1.0073, 1.0044, 1.0047, 1.0159, 1.0036,
    1.0072, 1.0040, 0.9997, 1.0149, 1.0071, 0.9938, 1.0083, 1.0085, 1.0063, 1.0119, 0.9933, 1.0124,
)  # fmt: skip

#: The pilot's standardized bias of the targeted arm and of the untargeted control (in the
#: control's own SD) for each necessity label: 2,000 draws.
PILOT_NECESSITY = {
    "unweighted_ate_s0": (0.091, 18.62),
    "unweighted_ate_s1": (0.013, 7.67),
    "unweighted_ate_s2": (-0.018, -9.82),
    "unweighted_ey0_s0": (-0.002, 12.59),
    "unweighted_ey0_s1": (0.021, 3.51),
    "unweighted_ey0_s2": (-0.056, -10.46),
    "unweighted_ey1_s0": (0.125, 28.51),
    "unweighted_ey1_s1": (0.034, 10.67),
    "unweighted_ey1_s2": (-0.085, -17.05),
}


def test_the_calibration_count_is_the_per_arm_rule() -> None:
    """``per_arm_replicates`` over the 48 positives, every cell read at the smallest pooled pilot
    size (8,000 draws), gives 7,600; 7,200 misses the family target.

    Reading each cell at its own pilot size would give 7,200, so the smallest size errs toward
    more replicates, as the product rule does by treating the 12 correlated labels of one fit as
    independent. The bounded settings alone would give 6,800 (the design review's pooled reading)
    and the binary positives move it to 7,600 (design review finding 2)."""
    assert len(PILOT_SE_RATIOS) == len(study.CALIBRATION_LABELS)
    assert tuple(PILOT_DRAWS) == law.SETTINGS
    smallest = min(PILOT_DRAWS.values())
    sized = tools.per_arm_replicates(PILOT_SE_RATIOS, smallest)
    assert sized["R"] == properties.CALIBRATION_REPLICATES == 7_600
    assert sized["family_wise"] == pytest.approx(0.9560, abs=5e-4)
    assert min(sized["per_cell"]) == pytest.approx(0.9947, abs=5e-4)
    below = tools.per_arm_pass(PILOT_SE_RATIOS, smallest, 7_200)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    own = [
        cell
        for index, draws in enumerate(PILOT_DRAWS.values())
        for cell in tools.per_arm_pass(PILOT_SE_RATIOS[12 * index : 12 * index + 12], draws, 7_600)
    ]
    assert float(np.prod(own)) >= sized["family_wise"]
    bounded = tools.per_arm_replicates(PILOT_SE_RATIOS[:24], smallest)
    assert bounded["R"] == 6_800
    # Every arm passes the per-arm screen, so none is diagnosed alone.
    half = tools.screen_half_width(smallest)
    assert all(abs(ratio - 1.0) <= half for ratio in PILOT_SE_RATIOS)


def test_the_necessity_count_covers_the_pilots_largest_bias() -> None:
    """2,800 draws establish every targeted arm inside the margin with probability above 0.999
    at the pilot's biases, and above 0.95 if ``ey1_s0``'s bias is two pilot SEs larger (0.17).
    Each control's bias, at least 3.5 own SDs, is established outside the margin surely."""
    assert set(PILOT_NECESSITY) == set(study.NECESSITY_LABELS)
    replicates = properties.NECESSITY_REPLICATES
    assert replicates == 2_800
    powers = [tools.equivalence_power(bias, replicates) for bias, _ in PILOT_NECESSITY.values()]
    assert float(np.prod(powers)) > 0.999
    assert tools.equivalence_power(0.17, replicates) > 0.95
    assert all(
        tools.necessity_power(control, replicates) > 0.9999
        for _, control in PILOT_NECESSITY.values()
    )
