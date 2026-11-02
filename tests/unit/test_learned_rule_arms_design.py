"""The declared constants of the ``learned-rule-arms`` study, recomputed without a study run.

Each number here was fixed before any registered run. The law checks call no sampler except the
``n = 10^6`` cross-check of the population limit: they solve the population score equations of
each declared learner and integrate on the RM30 grid. The pilot readings and the sizing rules that
read them are pinned. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta, norm

from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_arms_law as law
from tests.studies import learned_rule_arms as study
from tests.studies import learned_rule_arms_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_arms_design")

base = law.base

#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2``, times the density 1/2
#: of ``W1`` and the probability 1/2 of the level.
MASS = np.full(law.GRID.size, law.GRID[1] - law.GRID[0])
MASS[[0, -1]] /= 2.0
MASS *= 0.25


def _limit(terms: str, w2: float) -> np.ndarray:
    return law.limit_predictions(terms, law.GRID, w2)


def _describe(
    predictions: Callable[[float], np.ndarray], reference: Mapping[float, np.ndarray] | None = None
) -> dict[str, Any]:
    """Each arm's share of the argmax rule of ``predictions``, the top-two gap's distribution,
    and the share on which the rule differs from ``reference``, by quadrature."""
    shares = np.zeros(len(law.ARMS))
    gaps, masses = [], []
    disagreement = 0.0
    for w2 in law.W2_LEVELS:
        values = predictions(w2)
        rule = np.argmax(values, axis=0)
        ordered = np.sort(values, axis=0)
        shares += [np.sum(MASS * (rule == arm)) for arm in law.ARMS]
        gaps.append(ordered[-1] - ordered[-2])
        masses.append(MASS)
        if reference is not None:
            disagreement += float(np.sum(MASS * (rule != reference[w2])))
    gap, mass = np.concatenate(gaps), np.concatenate(masses)
    order = np.argsort(gap)
    cumulative = np.cumsum(mass[order])
    return {
        "shares": tuple(float(s) for s in shares),
        "near_tie": {c: float(np.sum(mass * (gap < c))) for c in (0.005, 0.01, 0.02)},
        "quantiles": {
            q: float(gap[order][np.searchsorted(cumulative, q)]) for q in GAP_QUANTILE_LEVELS
        },
        "disagreement": disagreement,
    }


#: The quantile levels of the top-two gap the declaration records.
GAP_QUANTILE_LEVELS = (0.01, 0.05, 0.1, 0.25, 0.5)

#: The top-two gap quantiles of each law cell's learner limit (probability scale).
GAP_QUANTILES = {
    "declared": (0.006248, 0.031185, 0.061944, 0.146003, 0.242488),
    "missing_quadratic": (0.005499, 0.027399, 0.054658, 0.133813, 0.265246),
}


# ----------------------------------------------------------------------------- the law


def test_every_arm_probability_is_at_least_0_15() -> None:
    smallest = min(law.g0(law.GRID, w2).min() for w2 in law.W2_LEVELS)
    assert smallest == pytest.approx(0.166170, abs=1e-6)
    assert smallest >= 0.15


@pytest.mark.parametrize("terms", list(study.LIMITS))
def test_each_law_cells_learner_limit_is_the_declared_one(terms: str) -> None:
    """Impl review answer 1: each cell's per-arm shares, ``P(gap < c)`` and gap quantiles at
    its learner's population limit, and the share on which its rule differs from the optimal
    rule."""
    found = _describe(lambda w2: _limit(terms, w2), law.optimal_rule())
    declared = study.LIMITS[terms]
    assert found["shares"] == pytest.approx(declared["shares"], abs=1e-6)
    assert sum(found["shares"]) == pytest.approx(1.0, abs=1e-9)
    for c, share in declared["near_tie"].items():
        assert found["near_tie"][c] == pytest.approx(share, abs=1e-6), c
    assert tuple(found["quantiles"].values()) == pytest.approx(GAP_QUANTILES[terms], abs=1e-6)
    assert found["disagreement"] == pytest.approx(declared["disagreement"], abs=1e-6)


def test_both_law_cells_meet_the_margin_condition_and_the_drop_w1_limit_does_not() -> None:
    """Both law cells gate. The deliberate mutation: the ``drop_w1`` learner's limit assigns arm
    1 everywhere, so it loses two arms and the condition refuses it."""
    assert study.gated("declared") and study.gated("missing_quadratic")
    assert (study.MIN_ARM_SHARE, study.MAX_NEAR_TIE, law.NEAR_TIE) == (0.10, 0.02, 0.01)
    found = _describe(lambda w2: _limit("drop_w1", w2))
    assert found["shares"] == pytest.approx((0.0, 1.0, 0.0))
    assert min(found["shares"]) < study.MIN_ARM_SHARE


def test_the_missing_quadratic_limit_rule_is_a_different_rule() -> None:
    """Impl review answer 6: the limit keeps every arm at a share of at least 0.10 and differs
    from the optimal rule on a share of at least 0.05, so the cell does not copy the correct
    one."""
    limit = study.LIMITS["missing_quadratic"]
    assert min(limit["shares"]) >= 0.10 and limit["disagreement"] >= 0.05
    assert law.BLIPS[2][3] == 4.0


def test_the_missing_quadratic_limit_matches_a_fit_at_a_million_rows() -> None:
    """Impl review answer 1: the population score equations against the declared learner fitted
    to 10^6 rows (a throwaway seed), as wave-1 W-k1 did. The coefficients agree within 0.06 and
    the rule's shares and disagreement within 0.002."""
    frame = law.draw(1_000_000, 99_001)
    design = np.column_stack(
        [(frame["A"] == 1.0) * 1.0, (frame["A"] == 2.0) * 1.0, frame["W1"], frame["W2"]]
    )
    model = law.outcome_learner("missing_quadratic").fit(design, frame["Y"].to_numpy().astype(int))
    fitted = np.concatenate([model.model_.intercept_, model.model_.coef_[0]])
    population = law.population_coefficients("missing_quadratic")
    assert np.max(np.abs(fitted - population)) < 0.06

    def predictions(w2: float) -> np.ndarray:
        covariates = np.column_stack([law.GRID, np.full_like(law.GRID, w2)])
        return np.vstack(
            [model.predict_proba(law.arm_design(arm, covariates))[:, 1] for arm in law.ARMS]
        )

    found = _describe(predictions, law.optimal_rule())
    declared = study.LIMITS["missing_quadratic"]
    assert found["shares"] == pytest.approx(declared["shares"], abs=2e-3)
    assert found["disagreement"] == pytest.approx(declared["disagreement"], abs=2e-3)


def test_the_efficiency_bound_and_the_limit_values() -> None:
    value, variance = law.rule_value(law.optimal_rule())
    assert value == pytest.approx(0.732720, abs=1e-6)
    assert np.sqrt(variance) == pytest.approx(study.EFFICIENCY_BOUND, abs=2e-6)
    missing, _ = law.rule_value(law.limit_rule("missing_quadratic"))
    assert missing == pytest.approx(0.727895, abs=1e-6)
    assert study.EFFICIENCY_BOUNDS == {study.PRIMARY_LAW: study.EFFICIENCY_BOUND}


#: Corollary 2's predicted asymptotic SE ratio on the ``missing_quadratic`` cell:
#: ``sqrt(Var D / Var(D - Pi(D | T_g)))`` for the curve ``D`` at the learner's population limit
#: and the true ``g0``, where ``T_g`` is the span of the multinomial treatment model's scores
#: ``(1{A = a} - g0(a | W))(1, W1, W2)`` for arms 1 and 2. The reported curve omits ``Pi``.
PREDICTED_SE_RATIO = 1.000445


def _predicted_se_ratio(limit: Callable[[float], np.ndarray]) -> float:
    weights, scores, curves = [], [], []
    for w2 in law.W2_LEVELS:
        g = law.g0(law.GRID, w2)
        values = limit(w2)
        rule = np.argmax(values, axis=0)
        q = np.take_along_axis(values, rule[None, :], axis=0)[0]
        g_rule = np.take_along_axis(g, rule[None, :], axis=0)[0]
        tangent = np.column_stack([np.ones_like(law.GRID), law.GRID, np.full_like(law.GRID, w2)])
        for arm in law.ARMS:
            outcome = law.qbar0(arm, law.GRID, w2)
            for y in (0.0, 1.0):
                weights.append(MASS * g[arm] * (outcome if y else 1.0 - outcome))
                scores.append(np.hstack([((arm == b) - g[b])[:, None] * tangent for b in (1, 2)]))
                curves.append((rule == arm) * (y - q) / g_rule + q)
    p = np.concatenate(weights)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    score, curve = np.vstack(scores), np.concatenate(curves)
    variance = p @ (curve - p @ curve) ** 2
    cross = (score * p[:, None]).T @ curve
    gram = (score * p[:, None]).T @ score
    return float(np.sqrt(variance / (variance - cross @ np.linalg.solve(gram, cross))))


def test_corollary_2s_projection_is_negligible_on_the_missing_quadratic_cell() -> None:
    """The analogy's route predicts an SE ratio of 1.0004 on the ``missing_quadratic`` cell, so
    the cell cannot tell it from exact calibration and measures coverage, not
    conservativeness. At the correct limit the projection vanishes exactly. The nonzero
    witness: an outcome limit that misses every arm by ``0.15 W1`` predicts a ratio above
    1.005."""
    ratio = _predicted_se_ratio(lambda w2: _limit("missing_quadratic", w2))
    assert ratio == pytest.approx(PREDICTED_SE_RATIO, abs=2e-6)
    assert 1.0 <= ratio <= 1.001
    assert _predicted_se_ratio(lambda w2: _limit("declared", w2)) == pytest.approx(1.0, abs=1e-9)
    crude = _predicted_se_ratio(lambda w2: _limit("missing_quadratic", w2) + 0.15 * law.GRID)
    assert crude > 1.005


# ---------------------------------------------------------------- the necessity learner

#: The ``drop_w1`` learner's limit: the six cell means ``E[Y | A, W2]``, in arm-code order for
#: each ``W2``. Its limit rule is arm 1 at both levels.
DROP_W1_CELL_MEANS = {
    0.0: (0.544780, 0.569279, 0.516446),
    1.0: (0.471167, 0.543176, 0.319569),
}

#: The plug-in limit bias of each rule the ``drop_w1`` learner can learn (one arm for each
#: ``W2``), in SEs at n = 2,000 of that rule's value: the bias the untargeted control carries
#: when every fold learns that rule. Only the treat-with-arm-0 rule has no bias, because no arm-0
#: blip exists. The limit rule ``(1, 1)`` sits at 9.39.
DROP_W1_BIAS_SES = {
    (0, 0): -0.21,
    (0, 1): 3.67,
    (0, 2): 1.78,
    (1, 0): 3.55,
    (1, 1): 9.39,
    (1, 2): 7.04,
    (2, 0): 2.46,
    (2, 1): 7.77,
    (2, 2): 5.53,
}


def _cell_mean(arm: int, w2: float) -> float:
    g = law.g0(law.GRID, w2)[arm]
    return float(np.sum(MASS * g * law.qbar0(arm, law.GRID, w2)) / np.sum(MASS * g))


def test_the_drop_w1_limit_is_the_cell_means_and_its_plug_in_is_biased() -> None:
    """Impl review answer 8: the untargeted plug-in's limit bias is several SEs whichever rule
    the folds settle on, except arm 0 everywhere, which no limit is near (its cell means sit at
    least 0.024 and 0.072 below arm 1's)."""
    for w2, means in DROP_W1_CELL_MEANS.items():
        found = tuple(_cell_mean(arm, w2) for arm in law.ARMS)
        assert found == pytest.approx(means, abs=1e-6)
        np.testing.assert_allclose(_limit("drop_w1", w2)[:, 0], means, atol=1e-6)
    for (d0, d1), declared in DROP_W1_BIAS_SES.items():
        rule = {0.0: np.full(law.GRID.size, d0), 1.0: np.full(law.GRID.size, d1)}
        value, variance = law.rule_value(rule)
        plug_in = 0.5 * (DROP_W1_CELL_MEANS[0.0][d0] + DROP_W1_CELL_MEANS[1.0][d1])
        assert (plug_in - value) / np.sqrt(variance / study.PRIMARY_N) == pytest.approx(
            declared, abs=0.01
        )
    assert DROP_W1_BIAS_SES[1, 1] > 9.0
    assert min(b for rule, b in DROP_W1_BIAS_SES.items() if 1 in rule) > 1.7


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert study.PROPERTY_CELLS == {
        "interval_calibration": (
            "three_arms__declared",
            "three_arms__shrunken_se_control",
            "three_arms__noise_control",
        ),
        "conservative_coverage": (
            "missing_quadratic__declared",
            "missing_quadratic__shrunken_coverage_control",
        ),
        "targeting_necessity": ("drop_w1__targeted", "drop_w1__untargeted"),
        "rule_margin_diagnostic": ("three_arms__rule_margin", "missing_quadratic__rule_margin"),
    }
    assert study.SCENARIOS == {"three_arms": ("ey_learned_rule[learned rule]",)}
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert properties.TARGETING_DISPLACEMENT == 0.5
    assert study.COVERAGE_SHRINK == 0.60
    assert study.STUDY.truth_varies_by_replicate and not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == 2_000
    assert properties.FAMILY_LEARNERS == {
        "interval_calibration": "declared",
        "conservative_coverage": "missing_quadratic",
        "targeting_necessity": "drop_w1",
    }


def test_the_fit_is_the_declared_one() -> None:
    estimator = law.estimator(law.outcome_learner(), 5)
    assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
    assert estimator.cv_evaluation and estimator.targeting_scheme == "pooled"
    assert estimator.n_folds == 10 and estimator.g_bounds == "auto"
    assert estimator.repeats == 1 and not estimator.interventions
    assert estimator.treatment_learner is not None
    assert np.inf == estimator.treatment_learner.C


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/targets/builtin.py",
        "tests/studies/_learned_rule_arms_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/learned_rule_arms_properties.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_support_row_declares_three_arms_fold_evaluated() -> None:
    """Impl review answer 5: one row, three arms only; ``arms4`` is another key."""
    assert study.SUPPORT_ROWS == {
        "fold_evaluated": {"splitting": ("fold_evaluated",), "treatment": ("arms3",)},
    }


def test_the_declared_gates_are_read() -> None:
    """Every verdict family but the diagnostic has a rule, and the thresholds are the ones the
    verdict functions read."""
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == {"interval_calibration", "conservative_coverage", "targeting_necessity"}
    assert properties.TARGETING_NECESSITY.threshold == properties.TARGETING_DISPLACEMENT
    assert properties.TARGETING_NECESSITY.truth_varies
    assert properties.TARGETING_NECESSITY.labels == (study.NECESSITY_LABEL,)
    assert study.MARGIN_FAMILY not in families


def test_the_coverage_control_fails_at_every_ratio_the_claim_admits() -> None:
    """The shrink was declared before the pilot: at a true SE ratio of 1.30 the control's
    coverage sits below the floor by more than the 99% half-width at the declared count."""
    floor = study.STUDY.margins.calibration_coverage[0]
    coverage = 2 * norm.cdf(base.CRITICAL * study.COVERAGE_SHRINK * 1.30) - 1
    half = norm.ppf(0.995) * np.sqrt(coverage * (1 - coverage) / properties.COVERAGE_REPLICATES)
    assert coverage + half < floor


def test_the_module_docstring_quotes_the_law() -> None:
    """The hashed docstrings say what the cells measure (the F2 lesson)."""
    doc = " ".join((study.__doc__ or "").split())
    for quoted in ("0.175", "1.6%", "7.0%", "0.245", "1.8%", "1.0004"):
        assert quoted in doc, quoted
    assert f"{min(study.LIMITS['declared']['shares']):.3f}" == "0.175"
    assert f"{study.LIMITS['declared']['near_tie'][0.01]:.1%}" == "1.6%"
    assert f"{study.LIMITS['missing_quadratic']['disagreement']:.1%}" == "7.0%"
    assert f"{min(study.LIMITS['missing_quadratic']['shares']):.3f}" == "0.245"
    assert f"{study.LIMITS['missing_quadratic']['near_tie'][0.01]:.1%}" == "1.8%"
    assert f"{PREDICTED_SE_RATIO:.4f}" == "1.0004"
    law_doc = " ".join((law.__doc__ or "").split())
    assert "0.166" in law_doc and "7.0%" in law_doc


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def _x11_modules() -> list[tuple[Any, Any]]:
    from tests.studies import fold_evaluated_strata as e1
    from tests.studies import fold_evaluated_strata_properties as e1_properties
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
    from tests.studies import learned_rule_optimal_value as optimal
    from tests.studies import learned_rule_optimal_value_properties as optimal_properties
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
        (optimal, optimal_properties),
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
    ("interval_calibration", "three_arms"): (),
    ("conservative_coverage", "missing_quadratic"): (),
    ("targeting_necessity", "three_arms_drop_w1"): (),
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


# ---------------------------------------------------------------------------- the sizing

#: The study's floor of draws for a gated cell (plan section 11).
FLOOR = 1_000

#: The pilot (``reviews/x11/scripts/X11-H-calibration-pilot.py`` in the plans repository,
#: throwaway label 20268801, 4,000 draws of each cell at n = 2,000, no failed fit, rule L4 on
#: every row and every stored fold rule).
PILOT_DRAWS = 4_000
PILOT = {
    "declared": {"ratio": 0.9986, "bias": -0.0108, "coverage": 0.9500},
    "missing_quadratic": {"ratio": 0.9953, "bias": -0.0062, "coverage": 0.9515},
}

#: The calibration controls on the ``declared`` pilot: (SE ratio, coverage) of the shrunken-SE
#: control (factor 0.80) and of the noise control (one efficiency-bound unit).
PILOT_SHRUNKEN = (0.7989, 0.8818)
PILOT_NOISE = (0.7092, 0.8260)

#: The coverage control's coverage on the ``missing_quadratic`` pilot (SE times 0.60).
PILOT_COVERAGE_CONTROL = 0.7508

#: The ``drop_w1`` pilot: the targeted value's standardized bias, SE ratio and coverage, the
#: untargeted fold average's bias in its own SDs, and the paired displacement in targeted-arm
#: SDs. Its fitted rules vary between draws: the limit rule ``(1, 1)`` in 1,702 of the 4,000.
NECESSITY_PILOT = {
    "targeted_bias": -0.0714,
    "ratio": 0.9371,
    "coverage": 0.9347,
    "untargeted_bias": 4.1791,
    "displacement": 6.0530,
}

#: The ``rule_margin_diagnostic`` readings of the pilot: the mean share of rows whose fitted rule
#: differs from the cell's limit rule, the mean near-tie share, the mean smallest arm share and
#: the share of draws in which some arm gets no row.
PILOT_MARGIN = {
    "declared": (0.0361, 0.0165, 0.1768, 0.0),
    "missing_quadratic": (0.0385, 0.0183, 0.2456, 0.0),
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


def test_the_primary_count_is_the_floor() -> None:
    """The bias leg passes with probability 0.976 at 400 and 0.99993 at 800, so the floor of
    1,000 sets 1,200."""
    bias = PILOT["declared"]["bias"]
    assert study.PRIMARY_REPLICATES == _smallest(lambda r: bias_pass(bias, r)) == 1_200
    assert bias_pass(bias, 400) == pytest.approx(0.976, abs=1e-3)


def test_the_calibration_count_is_the_per_arm_rule() -> None:
    """One positive, so the per-arm rule (the arm-pooled rule needs two arms) gives 3,200
    (0.952) and 2,800 misses (0.924). The coverage band holds at the count."""
    ratio = PILOT["declared"]["ratio"]
    sized = tools.per_arm_replicates([ratio], PILOT_DRAWS)
    assert sized["R"] == properties.CALIBRATION_REPLICATES == 3_200
    assert sized["family_wise"] == pytest.approx(0.952, abs=1e-3)
    assert tools.per_arm_pass([ratio], PILOT_DRAWS, 2_800)[0] < tools.FAMILY_TARGET
    low, high = study.STUDY.margins.calibration_coverage
    coverage = PILOT["declared"]["coverage"]
    assert floor_pass(coverage, 3_200, low) > 0.99
    assert 1 - floor_pass(1 - coverage, 3_200, 1 - high) < 0.01
    assert bias_pass(PILOT["declared"]["bias"], 3_200) > 0.9999


def test_the_calibration_count_survives_one_adverse_pilot_se() -> None:
    """The stress test reads the true ratio as known at one pilot SD below the pilot reading,
    without adding the pilot error a second time: the cell still passes with 0.952."""
    shifted = PILOT["declared"]["ratio"] - tools.pilot_sigma(PILOT_DRAWS)
    found = float(tools.cell_pass(np.array([shifted]), 3_200, tools.C_FIT)[0])
    assert found == pytest.approx(0.952, abs=1e-3)


def _coverage_pass(
    replicates: int,
    coverage: float | None = None,
    known: bool = False,
    readings: Mapping[str, float] | None = None,
) -> float:
    """The coverage family's pass probability. ``readings`` defaults to the declared pilot's
    ``missing_quadratic`` row with :data:`PILOT_COVERAGE_CONTROL`; a review reading passes its
    own ``bias``, ``coverage``, ``control`` and ``draws``. ``known`` reads the coverages as known
    (the conditional stress test); the bias leg keeps its predictive pilot error."""
    floor = study.STUDY.margins.calibration_coverage[0]
    if readings is None:
        readings = {
            **PILOT["missing_quadratic"],
            "control": PILOT_COVERAGE_CONTROL,
            "draws": PILOT_DRAWS,
        }
    draws = int(readings["draws"])
    coverage = readings["coverage"] if coverage is None else coverage
    pilot = None if known else draws
    return (
        bias_pass(readings["bias"], replicates, draws)
        * floor_pass(coverage, replicates, floor, pilot)
        * ceiling_pass(readings["control"], replicates, floor, pilot)
    )


def test_the_coverage_count_is_the_predictive_rule() -> None:
    """Both ``conservative_coverage`` cells pass with probability 0.953 at 1,200; 800 gives
    0.809."""
    assert properties.COVERAGE_REPLICATES == _smallest(_coverage_pass) == 1_200
    assert _coverage_pass(1_200) == pytest.approx(0.953, abs=1e-3)
    assert _coverage_pass(800) == pytest.approx(0.809, abs=1e-3)


def test_the_coverage_count_is_thin_under_one_adverse_pilot_se() -> None:
    """On the declared pilot alone, with the true coverage known at one pilot SE below the
    reading (0.948), the cells pass with 0.907 at 1,200 and would with 0.976 at 1,600. The
    review's second pilot superseded this reading (:data:`REVIEW_PROBE`); the count stands."""
    reading = PILOT["missing_quadratic"]["coverage"]
    shifted = reading - np.sqrt(reading * (1 - reading) / PILOT_DRAWS)
    assert _coverage_pass(1_200, shifted, known=True) == pytest.approx(0.907, abs=1e-3)
    assert _coverage_pass(1_600, shifted, known=True) == pytest.approx(0.976, abs=1e-3)


#: The design review's second pilot (``reviews/x11/X11-H-study-design-review.md``, finding 1):
#: the pilot script run unchanged but for the throwaway label 99137761 and its cell loop, 4,000
#: draws each of ``missing_quadratic`` and ``declared``, no failed fit, no solver warning, rule
#: L4 on every row and stored fold rule. ``pooled`` reads the two pilots' 8,000 draws together.
#: ``ratio`` and ``primary_bias`` are ``declared``'s; ``bias``, ``coverage`` and ``control``
#: (the x 0.6 coverage control) are ``missing_quadratic``'s.
REVIEW_PROBE: dict[str, dict[str, float]] = {
    "probe": {
        "draws": 4_000,
        "ratio": 0.9991,
        "primary_bias": 0.0378,
        "bias": 0.0066,
        "coverage": 0.9535,
        "control": 0.7668,
    },
    "pooled": {
        "draws": 8_000,
        "ratio": 0.9988,
        "primary_bias": 0.0135,
        "bias": 0.0002,
        "coverage": 0.9525,
        "control": 0.7588,
    },
}


def _probe_counts(reading: Mapping[str, float]) -> tuple[int, int, int]:
    draws = int(reading["draws"])
    primary = _smallest(lambda r: bias_pass(reading["primary_bias"], r, draws))
    calibration = tools.per_arm_replicates([reading["ratio"]], draws)["R"]
    coverage = _smallest(lambda r: _coverage_pass(r, readings=reading))
    return primary, calibration, coverage


def test_the_review_probe_gives_at_most_the_declared_counts() -> None:
    """The declared rules, unchanged, on the review's second pilot reproduce the declared
    counts (coverage at 0.979), and on the pooled 8,000 draws give 1,200, 2,800 and 1,200, at
    which the declared counts pass with 0.975 (calibration) and 0.976 (coverage). With the
    coverage known at one pooled SE below the pooled reading (0.950), 1,200 passes with 0.953.
    The counts err toward more draws, and none was resized (design review finding 1)."""
    declared = (
        study.PRIMARY_REPLICATES,
        properties.CALIBRATION_REPLICATES,
        properties.COVERAGE_REPLICATES,
    )
    probe, pooled = REVIEW_PROBE["probe"], REVIEW_PROBE["pooled"]
    assert _probe_counts(probe) == declared == (1_200, 3_200, 1_200)
    assert _coverage_pass(1_200, readings=probe) == pytest.approx(0.979, abs=1e-3)
    assert _probe_counts(pooled) == (1_200, 2_800, 1_200)
    draws = int(pooled["draws"])
    calibration = tools.per_arm_pass([pooled["ratio"]], draws, declared[1])[0]
    assert calibration == pytest.approx(0.975, abs=1e-3)
    assert _coverage_pass(declared[2], readings=pooled) == pytest.approx(0.976, abs=1e-3)
    reading = pooled["coverage"]
    shifted = reading - np.sqrt(reading * (1 - reading) / draws)
    assert shifted == pytest.approx(0.950, abs=1e-3)
    assert _coverage_pass(1_200, shifted, known=True, readings=pooled) == pytest.approx(
        0.953, abs=1e-3
    )


def test_the_review_probe_matches_the_declared_pilot_where_both_read() -> None:
    """The pooled readings are the two pilots' means, so a mistyped pin fails here."""
    probe, pooled = REVIEW_PROBE["probe"], REVIEW_PROBE["pooled"]
    pairs = (
        ("ratio", PILOT["declared"]["ratio"]),
        ("primary_bias", PILOT["declared"]["bias"]),
        ("bias", PILOT["missing_quadratic"]["bias"]),
        ("coverage", PILOT["missing_quadratic"]["coverage"]),
        ("control", PILOT_COVERAGE_CONTROL),
    )
    for key, first in pairs:
        assert pooled[key] == pytest.approx((first + probe[key]) / 2, abs=1e-4), key


def test_the_necessity_count_is_the_bias_leg_rule() -> None:
    """The positive arm's bias leg passes with 0.829 at 400, 0.988 at 800 and 0.9992 at 1,200,
    so the floor sets 1,200; the control's bias and the displacement are over 100 standard
    errors past their thresholds there."""
    targeted = NECESSITY_PILOT["targeted_bias"]
    assert _smallest(lambda r: bias_pass(targeted, r)) == properties.NECESSITY_REPLICATES
    assert properties.NECESSITY_REPLICATES == 1_200
    assert bias_pass(targeted, 800) == pytest.approx(0.988, abs=1e-3)
    se = np.sqrt(1.0 / properties.NECESSITY_REPLICATES + 1.0 / PILOT_DRAWS)
    assert NECESSITY_PILOT["untargeted_bias"] - 0.25 > 100 * se
    assert NECESSITY_PILOT["displacement"] - properties.TARGETING_DISPLACEMENT > 100 * se
    assert tools.necessity_power(NECESSITY_PILOT["untargeted_bias"], 1_200) > 0.999


def test_every_calibration_control_is_pilot_checked_below_the_band() -> None:
    floor = study.STUDY.margins.calibration_se_ratio[0]
    low = study.STUDY.margins.calibration_coverage[0]
    for ratio, coverage in (PILOT_SHRUNKEN, PILOT_NOISE):
        assert ratio < floor - 0.05
        assert ceiling_pass(coverage, properties.CALIBRATION_REPLICATES, low) > 0.999
    shrunken = properties.SHRUNKEN_SE_FACTOR * PILOT["declared"]["ratio"]
    assert PILOT_SHRUNKEN[0] == pytest.approx(shrunken, abs=1e-4)


def test_the_coverage_control_fails_at_the_count() -> None:
    floor = study.STUDY.margins.calibration_coverage[0]
    assert ceiling_pass(PILOT_COVERAGE_CONTROL, properties.COVERAGE_REPLICATES, floor) > 0.9999


def test_no_pilot_draw_lost_an_arm_in_a_law_cell() -> None:
    """The margin condition's finite-sample reading: no pilot draw of a law cell left an arm
    without a row, and the mean smallest share stayed near the limit's."""
    for terms, (_, near_tie, smallest, lost) in PILOT_MARGIN.items():
        assert lost == 0.0
        assert smallest == pytest.approx(min(study.LIMITS[terms]["shares"]), abs=0.01)
        assert near_tie < study.MAX_NEAR_TIE


#: The sample seeds a declared run draws: 1,200 primary and 3,200 + 1,200 + 1,200 property.
TOTAL_SEEDS = 6_800

#: The sample seeds this study shares with the RM30 and the other X11 studies.
SHARED_SEEDS: set[int] = set()


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))
