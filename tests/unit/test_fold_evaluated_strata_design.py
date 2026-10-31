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
}

#: The asymptotic SE ratio of the marginal curve without its share term, by setting and target.
SHARE_TERM_RATIOS = {
    "unweighted": {"ate": 0.854196, "ey0": 0.947636, "ey1": 0.655980},
    "weighted": {"ate": 0.853715, "ey0": 0.945242, "ey1": 0.660007},
}


def _adaptive(function: object, low: float, high: float) -> float:
    return float(quad(function, low, high, epsabs=1e-13, epsrel=1e-12, limit=200)[0])


def _functional(target: str, w1: float, w2: float) -> float:
    q1, q0 = float(law.qbar0(1.0, w1, w2)), float(law.qbar0(0.0, w1, w2))
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
    frame = law.draw(500, 17)
    rm30 = base.draw("non_exceptional", 500, 17)
    for column in ("W1", "W2", "A"):
        np.testing.assert_array_equal(frame[column], rm30[column])
    assert frame["Y"].min() > 0.0 and frame["Y"].max() < 1.0
    np.testing.assert_array_equal(frame["S"], np.digitize(frame["W1"], [-0.6, 0.1]))
    np.testing.assert_array_equal(frame[law.WEIGHT], 1.0 + 1.5 * frame["W2"])


def test_the_shares_are_unequal_and_untouched_by_the_weight() -> None:
    lengths = np.diff(law.EDGES) / 2.0
    np.testing.assert_allclose(lengths, law.SHARES, rtol=0, atol=1e-15)
    assert len(set(law.SHARES)) == 3
    # S reads W1 only and the weight W2 only, and they are independent.
    for setting in law.SETTINGS:
        weights = law._weights(setting)
        tilted = [sum(0.5 * w * share for w in weights.values()) for share in law.SHARES]
        np.testing.assert_allclose(np.array(tilted) / sum(tilted), law.SHARES, atol=1e-15)


@pytest.mark.parametrize("setting", law.SETTINGS)
def test_the_truths_are_the_quadrature_and_an_adaptive_integral(setting: str) -> None:
    declared = law.truths(setting)
    for name, value in TRUTHS[setting].items():
        assert declared[name] == pytest.approx(value, abs=5e-8), name
    weights = law._weights(setting)
    for target, name in law.TARGETS.items():
        parts = []
        for code in law.STRATA:
            low, high = law.EDGES[code], law.EDGES[code + 1]
            numerator = sum(
                w * _adaptive(lambda x, w2=w2, target=target: _functional(target, x, w2), low, high)
                for w2, w in weights.items()
            )
            exact = numerator / ((high - low) * sum(weights.values()))
            parts.append(exact)
            assert declared[f"{name}[S={code}]"] == pytest.approx(exact, abs=1e-8)
        # The marginal truth is the share mixture, and it is the whole-support integral too.
        whole = sum(
            w * _adaptive(lambda x, w2=w2, target=target: _functional(target, x, w2), -1.0, 1.0)
            for w2, w in weights.items()
        ) / (2.0 * sum(weights.values()))
        mixture = [declared[f"{name}[S={code}]"] for code in law.STRATA]
        assert declared[name] == pytest.approx(float(np.dot(law.SHARES, mixture)), abs=1e-15)
        assert declared[name] == pytest.approx(float(np.dot(law.SHARES, parts)), abs=1e-8)
        assert declared[name] == pytest.approx(whole, abs=1e-8)


def test_the_stratum_effects_are_strongly_heterogeneous() -> None:
    """The stratum ATEs change sign and spread by more than 0.4, which the share term reads."""
    for setting in law.SETTINGS:
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
            f, residual = law._target_parts(target, np.array([x]), w2)
            return float(residual[0] + (f[0] - psi) ** 2)

        variance = sum(
            0.25 * (w / mean_weight) ** 2 * _adaptive(lambda x, w2=w2: second(x, w2), -1.0, 1.0)
            for w2, w in weights.items()
        )
        assert np.sqrt(variance) == pytest.approx(
            study.EFFICIENCY_BOUNDS[f"{setting}_{target}_all"], abs=5e-6
        )


@pytest.mark.parametrize("setting", law.SETTINGS)
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


# --------------------------------------------------------------------------- the cells


def test_the_cells() -> None:
    calibration = study.PROPERTY_CELLS["interval_calibration"]
    assert len(study.CALIBRATION_LABELS) == 24
    assert len(study.STRATUM_LABELS) == 18
    assert len(calibration) == 24 * 3 + 18 + 4 == len(set(calibration))
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
    estimator = law.estimator(5)
    assert estimator.cross_fit and estimator.cv_evaluation
    assert estimator.n_folds == 5 == law.N_FOLDS
    assert estimator.targeting_scheme == "pooled"
    assert estimator.g_bounds == "auto"
    assert estimator.q_bounds == (0.0, 1.0)
    assert estimator.estimands == ("ate", "ey")
    assert estimator.simultaneous is False
    assert estimator.random_state == 5
    assert study.PRIMARY_N == properties.CALIBRATION_N == 1_000


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
    first retry; the other two labels keep theirs."""
    bare = stream_seed(study.STUDY, "property_sample", "interval_calibration", "unweighted", 4_935)
    assert bare == replicate_seed(study.STUDY, "unweighted", 665)
    assert properties.suffixes(study.STUDY) == {
        ("interval_calibration", "unweighted"): ("retry", 1),
        ("interval_calibration", "weighted"): (),
        ("targeting_necessity", "unweighted"): (),
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


def test_no_sample_seed_is_another_x11_studys() -> None:
    """The study's samples share no seed with each other or with another RM30 or X11 study.

    This law draws ``W1`` first, as RM30 does, so a shared seed would share a prefix of ``W1``.
    """
    own = base.declared_sample_seeds(
        study.STUDY,
        study.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(study.STUDY),
    )
    assert len(own) == 2 * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    others = _other_x11_sample_seeds()
    assert not own & others, sorted(own & others)
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    assert not seeds & learned_rule_run.declared_sample_seeds()


# ---------------------------------------------------------------------------- the sizing

#: The pilot's SE ratio of each calibration label, in ``CALIBRATION_LABELS`` order: 8,000 draws
#: per setting at n = 1,000 under the throwaway label 20264099
#: (``reviews/x11/scripts/X11-E1-calibration-pilot.py`` in the plans repository).
PILOT_DRAWS = 8_000
PILOT_SE_RATIOS = (
    1.0053, 1.0000, 0.9950, 0.9951, 1.0045, 1.0057, 1.0021, 1.0021, 0.9954, 0.9875, 1.0038, 0.9973,
    1.0159, 1.0065, 1.0167, 1.0046, 1.0043, 1.0139, 1.0072, 1.0099, 1.0103, 1.0003, 1.0082, 0.9897,
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
    """``per_arm_replicates`` over the 24 positives gives 6,400; 6,000 misses the family target."""
    assert len(PILOT_SE_RATIOS) == len(study.CALIBRATION_LABELS)
    sized = tools.per_arm_replicates(PILOT_SE_RATIOS, PILOT_DRAWS)
    assert sized["R"] == properties.CALIBRATION_REPLICATES == 6_400
    assert sized["family_wise"] == pytest.approx(0.9505, abs=5e-4)
    assert min(sized["per_cell"]) == pytest.approx(0.9888, abs=5e-4)
    below = tools.per_arm_pass(PILOT_SE_RATIOS, PILOT_DRAWS, 6_000)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    # Every arm passes the per-arm screen, so none is diagnosed alone.
    half = tools.screen_half_width(PILOT_DRAWS)
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
