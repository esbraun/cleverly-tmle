"""The declared constants of the ``learned-rule-contrast`` study, recomputed without a study run.

Each number here was fixed before any registered run. The law checks call no sampler: they read
the population projection of the declared learner and integrate on the RM30 grid. The pilot
readings and the sizing rules that read them are pinned. The declared seeds are only computed,
never drawn from, except the shared ones, whose draws show which variables they share.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_contrast_law as law
from tests.studies import _learned_rule_split_mean_law as split_mean_law
from tests.studies import learned_rule_contrast as study
from tests.studies import learned_rule_contrast_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_contrast_design")

base = law.base


def _optimal_rule(name: str) -> dict[float, np.ndarray]:
    return base.fixed_rule_grid(
        lambda w1, w2: base.qbar0(1.0, w1, w2, name) > base.qbar0(0.0, w1, w2, name)
    )


# ----------------------------------------------------------------------------- the law


def test_the_efficiency_bounds() -> None:
    """Each parameter's bound at the limit rule of the declared learner's population
    projection on ``non_exceptional``; the value's is RM30's 0.664438."""
    bounds = law.efficiency_bounds(
        split_mean_law.population_rule("non_exceptional"), "non_exceptional"
    )
    for tag, declared in study.TAG_BOUNDS.items():
        assert bounds[tag] == pytest.approx(declared, abs=2e-6), tag
    assert study.TAG_BOUNDS["value"] == 0.664438
    assert set(study.EFFICIENCY_BOUNDS) == set(study.CALIBRATION_LABELS)


#: The true values at the optimal rule ``1{b(W) > 0}`` of each law, recorded in the G
#: implementation notes, and the two true contrasts (Q6's nonzero witness): the contrast against
#: ``Static(1)`` and against ``Static(0)`` differ on each law.
OPTIMAL = {
    "non_exceptional": (0.57711, 0.53162, 0.51218),
    "misspecified_limit": (0.56025, 0.51083, 0.51218),
}


@pytest.mark.parametrize("scenario", law.SCENARIOS)
def test_the_true_contrasts_are_nonzero_and_differ(scenario: str) -> None:
    optimal, treat_all, treat_none = OPTIMAL[scenario]
    value = base.rule_value(_optimal_rule(scenario), scenario)[0]
    assert value == pytest.approx(optimal, abs=1e-5)
    assert law.static_value(1, scenario) == pytest.approx(treat_all, abs=1e-5)
    assert law.static_value(0, scenario) == pytest.approx(treat_none, abs=1e-5)
    limit = base.rule_value(split_mean_law.population_rule(scenario), scenario)[0]
    against = {arm: limit - law.static_value(arm, scenario) for arm in law.ARMS}
    assert min(against.values()) > 0.04


def test_the_limit_rule_treats_about_half() -> None:
    """The ``non_exceptional`` limit rule is neither static regime, so neither contrast's curve
    is degenerate (impl review open question 6)."""
    rule = split_mean_law.population_rule("non_exceptional")
    share = float(np.mean(np.concatenate(list(rule.values()))))
    assert 0.5 < share < 0.6


# ---------------------------------------------------------------- the necessity learner

#: The ``drop_w1`` learner's limit on ``non_exceptional``, by quadrature on the exact law: the
#: learner is saturated in ``(A, W2)``, so its limit regression is the four cell means of ``Y``.
#: Its limit blip is positive at both values of ``W2``, so its limit rule treats every row, which
#: is ``Static(1)``. The untargeted contrast against ``Static(0)`` at that rule converges to the
#: confounded difference of cell means, 0.023355 above the truth: the bias the control must show.
DROP_W1_LIMIT_BLIPS = {0.0: 0.037065, 1.0: 0.048519}
DROP_W1_CONTRAST_0_BIAS = 0.023355


def _cell_mean(w2: float, a: int) -> float:
    treated = base.g0(base.GRID, w2)
    arm = treated if a else 1.0 - treated
    mass = np.trapezoid(arm, base.GRID)
    outcome = np.trapezoid(arm * base.qbar0(a, base.GRID, w2, "non_exceptional"), base.GRID)
    return float(outcome / mass)


def test_the_necessity_learners_limit_is_static_one_with_a_biased_plug_in() -> None:
    """Why the necessity pair reads the contrast against ``Static(0)``: at the limit rule the
    contrast against ``Static(1)`` is zero, and the one against ``Static(0)`` has a plug-in bias
    of 0.0234 (about 1.6 contrast SEs at n = 2,000)."""
    means = {(w2, a): _cell_mean(w2, a) for w2 in base.W2_LEVELS for a in (0, 1)}
    for w2, declared in DROP_W1_LIMIT_BLIPS.items():
        assert means[w2, 1] - means[w2, 0] == pytest.approx(declared, abs=1e-6)
    plug_in = np.mean([means[w2, 1] - means[w2, 0] for w2 in base.W2_LEVELS])
    truth = law.static_value(1, "non_exceptional") - law.static_value(0, "non_exceptional")
    assert plug_in - truth == pytest.approx(DROP_W1_CONTRAST_0_BIAS, abs=1e-6)
    se = study.TAG_BOUNDS["contrast_0"] / np.sqrt(study.PRIMARY_N)
    assert DROP_W1_CONTRAST_0_BIAS / se > 1.4
    # The deliberate mutation: the declared learner's limit rule is not Static(1).
    rule = split_mean_law.population_rule("non_exceptional")
    assert not all(levels.all() for levels in rule.values())


# --------------------------------------------------------------- the misspecified law

#: Corollary 2's predicted asymptotic SE ratio of each parameter on ``misspecified_limit``, in
#: ``law.TAGS`` order: ``sqrt(Var D / Var(D - Pi(D | T_g)))`` for the curve ``D`` at the declared
#: learner's population limit and the true ``g0``, where ``T_g`` is the span of the logistic
#: treatment model's scores ``(A - g0)(1, W1, W2)``. The reported curve omits the projection
#: ``Pi``, so a ratio above 1 is the over-coverage Corollary 2 predicts. On this law every ratio
#: is at most 1.0005 (design review finding 2), so the ``conservative_coverage`` cells measure
#: coverage under a misspecified treated arm, not conservativeness.
PREDICTED_SE_RATIOS = (1.000150, 1.000094, 1.000000, 1.000462, 1.000150)


def _limit_arms(scenario: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """The declared learner's population limit ``(Qbar(0, W), Qbar(1, W))`` on the RM30 grid."""
    model = split_mean_law.population_model(scenario)
    arms = {}
    for w2 in base.W2_LEVELS:
        rows = [np.column_stack([np.full_like(base.GRID, a), base.GRID, np.full_like(base.GRID, w2)])
                for a in (0.0, 1.0)]  # fmt: skip
        arms[w2] = tuple(model.predict_proba(design)[:, 1] for design in rows)
    return arms


def _curve(
    regime: np.ndarray, a: int, y: int, arms: tuple[np.ndarray, np.ndarray], treated: np.ndarray
) -> np.ndarray:
    """The curve of the regime ``regime`` at ``(A, Y) = (a, y)`` and the outcome limit ``arms``."""
    q = np.where(regime, arms[1], arms[0])
    g = np.where(regime, treated, 1 - treated)
    return np.asarray((a == regime) * (y - q) / g + q, dtype=float)


def _predicted_se_ratios(
    arms: dict[float, tuple[np.ndarray, np.ndarray]], scenario: str
) -> dict[str, float]:
    """Corollary 2's ratio of each tag's curve at the outcome limit ``arms`` and the true ``g0``,
    by quadrature over ``(W1, W2, A, Y)`` on the RM30 grid."""
    mass = np.full(base.GRID.size, base.GRID[1] - base.GRID[0])
    mass[[0, -1]] /= 2.0
    mass *= 0.5 * 0.5  # the density of W1, and the probability of each W2
    weights, scores = [], []
    curves: dict[str, list[np.ndarray]] = {tag: [] for tag in law.TAGS}
    for w2 in base.W2_LEVELS:
        treated = base.g0(base.GRID, w2)
        rule = arms[w2][1] > arms[w2][0]
        tangent = np.column_stack([np.ones_like(base.GRID), base.GRID, np.full_like(base.GRID, w2)])
        for a in (0, 1):
            outcome = base.qbar0(a, base.GRID, w2, scenario)
            for y in (0, 1):
                arm = treated if a else 1 - treated
                weights.append(mass * arm * (outcome if y else 1 - outcome))
                scores.append((a - treated)[:, None] * tangent)
                value = _curve(rule, a, y, arms[w2], treated)
                known = {
                    arm_: _curve(np.full(rule.shape, bool(arm_)), a, y, arms[w2], treated)
                    for arm_ in law.ARMS
                }
                parts = {
                    "value": value,
                    "known_1": known[1],
                    "known_0": known[0],
                    "contrast_1": value - known[1],
                    "contrast_0": value - known[0],
                }
                for tag, part in parts.items():
                    curves[tag].append(part)
    p = np.concatenate(weights)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    score = np.vstack(scores)
    gram = (score * p[:, None]).T @ score
    ratios = {}
    for tag, pieces in curves.items():
        d = np.concatenate(pieces)
        variance = p @ (d - p @ d) ** 2
        cross = (score * p[:, None]).T @ d
        ratios[tag] = float(np.sqrt(variance / (variance - cross @ np.linalg.solve(gram, cross))))
    return ratios


def test_corollary_2s_projection_is_negligible_on_the_misspecified_law() -> None:
    """D8: the analogy predicts SE ratios of at most 1.0005 on ``misspecified_limit``, so the
    law cannot tell the analogy's route from exact calibration. On ``non_exceptional``, where
    the outcome model is correct, the projection vanishes exactly."""
    ratios = _predicted_se_ratios(_limit_arms("misspecified_limit"), "misspecified_limit")
    for (tag, ratio), declared in zip(ratios.items(), PREDICTED_SE_RATIOS, strict=True):
        assert ratio == pytest.approx(declared, abs=2e-6), tag
        assert 1.0 <= ratio <= 1.001, tag
    correct = _predicted_se_ratios(_limit_arms("non_exceptional"), "non_exceptional")
    assert all(ratio == pytest.approx(1.0, abs=1e-6) for ratio in correct.values())
    # The nonzero witness: an outcome limit that misses both arms by 0.15 W1 puts a visible
    # share of every curve in the treatment model's tangent space (every ratio above 1.005).
    crude = {
        w2: (q0 + 0.15 * base.GRID, q1 + 0.15 * base.GRID)
        for w2, (q0, q1) in _limit_arms("misspecified_limit").items()
    }
    assert min(_predicted_se_ratios(crude, "misspecified_limit").values()) > 1.005


def test_the_untreated_arms_limit_is_correctly_specified() -> None:
    """D8: RM30's two laws share ``Qbar0(0, W)``, which the interaction logistic model
    represents up to its shared ``W1 W2`` coefficient. On ``misspecified_limit`` the arm-0 limit
    stays within 0.005 of ``Qbar0(0, .)`` everywhere on the grid, so the ``known_0`` cells read
    a correctly specified arm. The treated arm's limit misses ``Qbar0(1, .)`` by more than 0.1:
    the deliberate-mutation control that the check can fail."""
    for w2, (q0, q1) in _limit_arms("misspecified_limit").items():
        assert np.max(np.abs(q0 - base.qbar0(0, base.GRID, w2, "misspecified_limit"))) < 0.005
        assert np.max(np.abs(q1 - base.qbar0(1, base.GRID, w2, "misspecified_limit"))) > 0.1
    for w2 in base.W2_LEVELS:
        shared = base.qbar0(0, base.GRID, w2, "non_exceptional")
        assert np.array_equal(shared, base.qbar0(0, base.GRID, w2, "misspecified_limit"))


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    tags = ("value", "known_1", "known_0", "contrast_1", "contrast_0")
    assert tuple(law.TAGS) == tags
    assert study.REPORTS == ("fold_evaluated", "stacked")
    calibration = [
        f"non_exceptional_{report}_{tag}__{kind}"
        for report in study.REPORTS
        for tag in tags
        for kind in ("declared", "shrunken_se_control", "noise_control")
    ]
    coverage = [
        f"misspecified_limit_{report}_{tag}__{kind}"
        for report in study.REPORTS
        for tag in tags
        for kind in ("declared", "shrunken_coverage_control")
    ]
    contrasts = [
        f"{scenario}_{report}_{tag}"
        for scenario in law.SCENARIOS
        for report in study.REPORTS
        for tag in ("contrast_1", "contrast_0")
    ]
    assert {
        "interval_calibration": tuple(calibration),
        "conservative_coverage": tuple(coverage),
        "contrast_covariance": tuple(f"{c}__correlation_one_control" for c in contrasts),
        "independent_se_diagnostic": tuple(f"{c}__independent_se" for c in contrasts),
        "targeting_necessity": ("drop_w1_contrast_0__targeted", "drop_w1_contrast_0__untargeted"),
    } == study.PROPERTY_CELLS
    assert {"non_exceptional": tuple(law.TAGS.values())} == study.SCENARIOS
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert properties.TARGETING_DISPLACEMENT == 0.5
    assert study.COVERAGE_SHRINK == 0.60
    assert study.STUDY.truth_varies_by_replicate and not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == 2_000


def test_the_fit_is_the_declared_one() -> None:
    estimator = law.estimator(base.outcome_learner(), 5, law.ARMS)
    assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
    assert [regime.name for regime in estimator.interventions] == ["always 1", "always 0"]
    assert estimator.cv_evaluation and estimator.targeting_scheme == "pooled"
    assert estimator.n_folds == 10 and estimator.g_bounds == "auto"
    assert estimator.simultaneous and estimator.repeats == 1
    assert not law.estimator(base.outcome_learner(), 5, law.ARMS, cv_evaluation=False).cv_evaluation


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/targets/builtin.py",
        "tests/studies/_learned_rule_contrast_law.py",
        "tests/studies/_learned_rule_law.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_support_rows_declare_the_known_regimes_qualifier_for_both_reports() -> None:
    """Impl review open question 2: rows for both reports, each with ``known_regimes``."""
    assert study.SUPPORT_ROWS == {
        "fold_evaluated": {"splitting": ("fold_evaluated",), "qualifiers": ("known_regimes",)},
        "stacked": {"splitting": ("stacked",), "qualifiers": ("known_regimes",)},
    }


def test_the_declared_gates_are_read() -> None:
    """Every verdict family but the diagnostic has a rule, and the thresholds are the ones the
    verdict functions read."""
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == {
        "interval_calibration",
        "conservative_coverage",
        "contrast_covariance",
        "targeting_necessity",
    }
    assert properties.TARGETING_NECESSITY.threshold == properties.TARGETING_DISPLACEMENT
    assert properties.TARGETING_NECESSITY.truth_varies
    assert properties.TARGETING_NECESSITY.labels == (study.NECESSITY_LABEL,)
    assert study.CORRELATION_ONE_CEILING == 0.80


def test_the_coverage_control_fails_at_every_ratio_the_claim_admits() -> None:
    """The shrink was declared before the pilot: at a true SE ratio of 1.30 the control's
    coverage sits below the floor by more than the 99% half-width at the declared count."""
    floor = study.STUDY.margins.calibration_coverage[0]
    coverage = 2 * norm.cdf(base.CRITICAL * study.COVERAGE_SHRINK * 1.30) - 1
    half = norm.ppf(0.995) * np.sqrt(coverage * (1 - coverage) / properties.COVERAGE_REPLICATES)
    assert coverage + half < floor


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import learned_rule_in_sample as f2
    from tests.studies import learned_rule_learners as learners
    from tests.studies import learned_rule_split_mean as split_mean
    from tests.studies import learned_rule_stacked as f1
    from tests.studies import learned_rule_strata as e2

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module in (e1, e2, f1, f2, learners, split_mean):
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


#: The suffix rule L3 adds to each property draw's labels: none, since no two draws share a
#: (family, label) pair or a law at one n.
DECLARED_SUFFIXES = {
    ("interval_calibration", "non_exceptional"): (),
    ("conservative_coverage", "misspecified_limit"): (),
    ("targeting_necessity", "non_exceptional_drop_w1"): (),
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


def _seed_of(family: str, setting: str, replicate: int) -> int:
    if family == "primary":
        return replicate_seed(study.STUDY, setting, replicate)
    resolved = properties.suffixes(study.STUDY)
    return stream_seed(
        study.STUDY, "property_sample", family, setting, replicate, *resolved[family, setting]
    )


def test_the_sample_seeds_are_distinct() -> None:
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert total == 11_200
    assert len(_own_sample_seeds()) == total


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
    from tests.studies import learned_rule_arms as h
    from tests.studies import learned_rule_arms_properties as h_properties
    from tests.studies import learned_rule_compositions as compositions
    from tests.studies import learned_rule_compositions_properties as composition_properties
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
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

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for record, module in (
        (gated, gated_properties),
        (compositions, composition_properties),
        (split_mean, split_mean_properties),
        (learners, learner_properties),
        (e1, e1_properties),
        (e2, e2_properties),
        (f1, f1_properties),
        (f2, f2_properties),
        (h, h_properties),
        (i, i_properties),
    ):
        seeds |= base.declared_sample_seeds(
            record.STUDY,
            record.PRIMARY_REPLICATES,
            module.PROPERTY_DRAWS,
            module.suffixes(record.STUDY),
        )
    return seeds


#: The one sample seed this study shares with another X11 study. Seeds are 32-bit, and 11,200
#: against the other X11 studies' 356,553 expect about 0.93 coincidences by chance. The coverage
#: draw (RM30's ``misspecified_limit`` law at n = 2,000) and the E1 calibration draw at
#: n = 1,000 both draw ``W1`` first, so the two share the first 1,000 values of ``W1`` and
#: nothing else (the E1 precedent). One replicate of 3,200 or 8,800 makes the dependence
#: negligible, and no verdict reads two studies jointly.
SHARED_SEED = ("conservative_coverage", "misspecified_limit", 1_088)
E1_PARTNER = ("interval_calibration", "weighted", 6_514)


def test_no_sample_seed_is_another_x11_studys_but_one_declared() -> None:
    from tests.studies import fold_evaluated_strata as e1

    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == {_seed_of(*SHARED_SEED)}
    assert stream_seed(e1.STUDY, "property_sample", *E1_PARTNER) == _seed_of(*SHARED_SEED)
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_shared_coverage_draw_shares_only_a_w1_prefix() -> None:
    from tests.studies import _fold_evaluated_strata_law as e1_law

    seed = _seed_of(*SHARED_SEED)
    here = base.draw("misspecified_limit", study.PRIMARY_N, seed)
    there = e1_law.draw("weighted", 1_000, seed)
    np.testing.assert_array_equal(here["W1"].to_numpy()[:1_000], there["W1"].to_numpy())
    assert not np.array_equal(here["W2"].to_numpy()[:1_000], there["W2"].to_numpy())


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, study.PRIMARY_LAW, 0) in declared


# ---------------------------------------------------------------------------- the sizing

#: The pilot (``reviews/x11/scripts/X11-G-calibration-pilot.py`` in the plans repository,
#: throwaway label 20267701, 4,000 draws of each cell at n = 2,000, no failed fit). Each reading
#: is of the fold-evaluated report, in ``law.TAGS`` order: the mean reported SE over the SD of
#: ``estimate - truth``, the standardized bias, and the Wald coverage. The stacked report's
#: readings differ from these by at most 0.0003 in ratio and 0.0005 in coverage: at ten equal
#: folds its point is the fold-evaluated one, so its arms are not separate draws (design D4).
PILOT_DRAWS = 4_000
PILOT = {
    "non_exceptional": {
        "ratio": (0.9942, 1.0052, 1.0101, 1.0014, 1.0078),
        "bias": (-0.0092, -0.0230, -0.0285, 0.0152, 0.0188),
        "coverage": (0.9457, 0.9533, 0.9543, 0.9435, 0.9497),
    },
    "misspecified_limit": {
        "ratio": (0.9947, 1.0115, 0.9773, 0.9770, 1.0106),
        "bias": (0.0256, 0.0256, 0.0007, -0.0010, 0.0250),
        "coverage": (0.9495, 0.9547, 0.9423, 0.9477, 0.9533),
    },
}

#: The pilot readings of each control on the same draws, fold-evaluated report, ``law.TAGS``
#: order. ``non_exceptional``: the shrunken (x 0.8) and the noise control's (SE ratio, coverage),
#: the noise from ``default_rng(1)``. ``misspecified_limit``: the coverage control's (x 0.6)
#: coverage. The contrasts' correlation-one coverage on each law, and the independent SE's
#: (SE ratio, coverage) on ``non_exceptional``.
PILOT_SHRUNKEN = ((0.7953, 0.8798), (0.8042, 0.8860), (0.8081, 0.8928), (0.8011, 0.8802),
                  (0.8062, 0.8848))  # fmt: skip
PILOT_NOISE = ((0.7132, 0.8340), (0.7078, 0.8397), (0.7157, 0.8415), (0.7064, 0.8353),
               (0.7128, 0.8297))  # fmt: skip
PILOT_COVERAGE_CONTROL = (0.7630, 0.7673, 0.7510, 0.7450, 0.7625)
PILOT_CORRELATION_ONE = {
    "non_exceptional": (0.0838, 0.0525),
    "misspecified_limit": (0.0853, 0.0517),
}
PILOT_INDEPENDENT = ((1.4879, 0.9960), (1.3594, 0.9915))

#: The ``drop_w1`` pilot (4,000 draws on ``non_exceptional``, fold-evaluated only): the targeted
#: contrast's standardized bias, SE ratio and coverage, the untargeted contrast's bias in its own
#: SDs, and the paired displacement in targeted-arm SDs.
NECESSITY_PILOT = {
    "targeted_bias": 0.0105,
    "ratio": 0.9588,
    "coverage": 0.9298,
    "untargeted_bias": 1.4496,
    "displacement": 1.2142,
}

#: The study's floor of draws for a gated cell (plan section 11).
FLOOR = 1_000


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


def _predicted(coverage: float, replicates: int, pilot: int = PILOT_DRAWS) -> np.ndarray:
    """The run's hit counts on a grid over the coverage's predictive law (pilot and run error)."""
    spread = np.sqrt(max(coverage * (1 - coverage), 1e-12) * (1 / replicates + 1 / pilot))
    grid = np.clip(coverage + spread * _Z, 0.0, 1.0)
    return np.rint(grid * replicates).astype(int)


def floor_pass(coverage: float, replicates: int, floor: float, pilot: int = PILOT_DRAWS) -> float:
    """The predictive probability that the exact 99% interval's lower bound is at or above
    ``floor``."""
    hits = _predicted(coverage, replicates, pilot)
    lower = np.where(hits > 0, beta.ppf(0.005, hits, replicates - hits + 1), 0.0)
    return float(np.sum(_W * (lower >= floor)))


def ceiling_pass(
    coverage: float, replicates: int, ceiling: float, pilot: int = PILOT_DRAWS
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


def _primary_pass(replicates: int) -> float:
    return float(np.prod([bias_pass(b, replicates) for b in PILOT[study.PRIMARY_LAW]["bias"]]))


def test_the_primary_count_is_the_floor() -> None:
    """The five bias legs pass with probability 0.99999 at 1,200, the smallest multiple of 400
    at or above the floor of 1,000; without the floor the rule would give 800 (0.999)."""
    assert study.PRIMARY_REPLICATES == _smallest(_primary_pass) == 1_200
    assert _primary_pass(1_200) > 0.9999
    assert _primary_pass(800) == pytest.approx(0.999, abs=1e-3)
    assert _primary_pass(400) < tools.FAMILY_TARGET


def test_the_calibration_count_is_the_arm_pooled_rule_on_the_five_parameters() -> None:
    """The arm-pooled rule over the five fold-evaluated positives gives 5,600 (family-wise
    0.951) and 5,200 misses. The per-arm rule gives the same count. Counting the stacked
    positives as five more arms would understate the pilot error (design D4)."""
    ratios = PILOT[study.PRIMARY_LAW]["ratio"]
    sized = tools.replicates_from_pilot(ratios, PILOT_DRAWS)
    assert all(sized["screened"]) and not sized["suspect"]
    assert sized["R"] == properties.CALIBRATION_REPLICATES == 5_600
    assert sized["family_wise"] == pytest.approx(0.951, abs=1e-3)
    assert tools.per_arm_replicates(ratios, PILOT_DRAWS)["R"] == 5_600
    below = tools.per_arm_pass(ratios, PILOT_DRAWS, 5_200)
    assert float(np.prod(below)) < tools.FAMILY_TARGET
    # Each positive's coverage band [0.92, 0.98] holds at the count.
    low, high = study.STUDY.margins.calibration_coverage
    for coverage in PILOT[study.PRIMARY_LAW]["coverage"]:
        assert floor_pass(coverage, 5_600, low) > 0.99
        assert 1 - floor_pass(1 - coverage, 5_600, 1 - high) < 0.01


def _coverage_pass(
    replicates: int,
    readings: Mapping[str, Sequence[float]] = PILOT[study.COVERAGE_LAW],
    controls: Sequence[float] = PILOT_COVERAGE_CONTROL,
    pilot: int = PILOT_DRAWS,
) -> float:
    floor = study.STUDY.margins.calibration_coverage[0]
    out = 1.0
    for bias, coverage, control in zip(
        readings["bias"], readings["coverage"], controls, strict=True
    ):
        out *= bias_pass(bias, replicates, pilot=pilot)
        out *= floor_pass(coverage, replicates, floor, pilot)
        out *= ceiling_pass(control, replicates, floor, pilot)
    return out


def test_the_coverage_count_is_the_predictive_rule() -> None:
    """Every ``conservative_coverage`` cell passes with probability 0.964 at 3,200; 2,800 gives
    0.943. ``known_0``'s coverage of 0.942 binds."""
    assert properties.COVERAGE_REPLICATES == _smallest(_coverage_pass) == 3_200
    assert _coverage_pass(3_200) == pytest.approx(0.964, abs=1e-3)
    assert _coverage_pass(2_800) == pytest.approx(0.943, abs=1e-3)


#: The design review's second pilot (``reviews/x11/X11-G-study-design-review.md``, finding 1):
#: the pilot script run unchanged but for the throwaway label 99127741, 4,000 draws each of the
#: two law cells, fold-evaluated report, ``law.TAGS`` order, no failed fit, and 3 solver
#: warnings on ``non_exceptional`` (none on ``misspecified_limit``). ``pooled`` reads the two
#: pilots' 8,000 draws together. ``ratio`` is ``non_exceptional``'s; ``bias``, ``coverage`` and
#: ``control`` (the x 0.6 coverage control) are ``misspecified_limit``'s.
REVIEW_PROBE: dict[str, dict[str, Any]] = {
    "probe": {
        "draws": 4_000,
        "ratio": (0.9980, 1.0101, 1.0050, 1.0082, 0.9951),
        "bias": (0.0082, 0.0025, -0.0188, 0.0054, 0.0268),
        "coverage": (0.9490, 0.9500, 0.9475, 0.9475, 0.9428),
        "control": (0.7638, 0.7645, 0.7612, 0.7580, 0.7562),
    },
    "pooled": {
        "draws": 8_000,
        "ratio": (0.9961, 1.0077, 1.0072, 1.0049, 1.0013),
        "bias": (0.0170, 0.0140, -0.0089, 0.0022, 0.0259),
        "coverage": (0.9492, 0.9524, 0.9449, 0.9476, 0.9480),
        "control": (0.7634, 0.7659, 0.7561, 0.7515, 0.7594),
    },
}


def _probe_counts(reading: dict[str, Any]) -> tuple[int, int]:
    draws = reading["draws"]
    calibration = tools.replicates_from_pilot(reading["ratio"], draws)["R"]
    coverage = _smallest(lambda r: _coverage_pass(r, reading, reading["control"], draws))
    return calibration, coverage


def test_the_review_probe_gives_at_most_the_declared_counts() -> None:
    """The declared rules, unchanged, on the review's second pilot reproduce the declared
    counts, and on the pooled 8,000 draws give 4,400 and 2,400, at which the declared counts
    pass with probability 0.990 (calibration) and 0.995 (coverage). The counts err toward
    more draws, and none was resized (design review finding 1)."""
    declared = (properties.CALIBRATION_REPLICATES, properties.COVERAGE_REPLICATES)
    assert _probe_counts(REVIEW_PROBE["probe"]) == declared == (5_600, 3_200)
    assert _probe_counts(REVIEW_PROBE["pooled"]) == (4_400, 2_400)
    pooled = REVIEW_PROBE["pooled"]
    sized = tools.replicates_from_pilot(pooled["ratio"], pooled["draws"])
    shared = tools.pilot_sigma(pooled["draws"]) ** 2 / len(pooled["ratio"])
    _, calibration = tools.pass_probabilities(
        declared[0], sized["mean"], shared, sized["tau"], len(pooled["ratio"])
    )
    assert calibration == pytest.approx(0.990, abs=1e-3)
    coverage = _coverage_pass(declared[1], pooled, pooled["control"], pooled["draws"])
    assert coverage == pytest.approx(0.995, abs=1e-3)


def test_the_necessity_count_is_the_floor() -> None:
    """The positive arm's bias leg passes with probability 0.977 at 400, so the floor sets
    1,200; the control's bias and the displacement are many standard errors past their
    thresholds there."""
    targeted = NECESSITY_PILOT["targeted_bias"]
    assert _smallest(lambda r: bias_pass(targeted, r)) == properties.NECESSITY_REPLICATES
    assert properties.NECESSITY_REPLICATES == 1_200
    assert bias_pass(targeted, 400) == pytest.approx(0.977, abs=1e-3)
    se = np.sqrt(1.0 / properties.NECESSITY_REPLICATES + 1.0 / PILOT_DRAWS)
    assert NECESSITY_PILOT["untargeted_bias"] - 0.25 > 30 * se
    assert NECESSITY_PILOT["displacement"] - properties.TARGETING_DISPLACEMENT > 20 * se
    assert tools.necessity_power(NECESSITY_PILOT["untargeted_bias"], 1_200) > 0.999


def test_every_calibration_control_is_pilot_checked_below_the_band() -> None:
    floor = study.STUDY.margins.calibration_se_ratio[0]
    for ratio, coverage in (*PILOT_SHRUNKEN, *PILOT_NOISE):
        assert ratio < floor - 0.05
        assert coverage < study.STUDY.margins.calibration_coverage[0]
    for ratio, (shrunken, _) in zip(PILOT[study.PRIMARY_LAW]["ratio"], PILOT_SHRUNKEN, strict=True):
        assert shrunken == pytest.approx(properties.SHRUNKEN_SE_FACTOR * ratio, abs=1e-4)


def test_the_coverage_control_fails_at_the_count() -> None:
    floor = study.STUDY.margins.calibration_coverage[0]
    for coverage in PILOT_COVERAGE_CONTROL:
        assert ceiling_pass(coverage, properties.COVERAGE_REPLICATES, floor) > 0.9999


def test_the_correlation_one_ceiling_was_set_far_above_the_pilot() -> None:
    """The ceiling 0.80 was recorded from the pilot before the run (design D6): the pilot's
    correlation-one coverage is 0.05 to 0.09, and each cell fails its ceiling with predictive
    probability above 0.9999 at its law's count."""
    counts = {
        "non_exceptional": properties.CALIBRATION_REPLICATES,
        "misspecified_limit": properties.COVERAGE_REPLICATES,
    }
    for scenario, readings in PILOT_CORRELATION_ONE.items():
        for coverage in readings:
            assert coverage < 0.10
            assert ceiling_pass(coverage, counts[scenario], study.CORRELATION_ONE_CEILING) > 0.9999


def test_the_independent_se_over_covers() -> None:
    """The diagnostic's pilot reading: an SE ratio above 1.3 and coverage above 0.99 on both
    contrasts. It carries no verdict."""
    for ratio, coverage in PILOT_INDEPENDENT:
        assert ratio > 1.3 and coverage > 0.99
