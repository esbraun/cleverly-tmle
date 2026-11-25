"""The declared constants of the ``learned-rule-contrast-comparators`` study (T58), recomputed.

Each number here was fixed before any declared run. The law checks integrate on the RM30 grid
and cross-check by an independent quadrature or by Monte Carlo. The sizing figures are computed
here, never copied. The checks that read the sizing pilot
(``tests/studies/learned_rule_contrast_comparators_pilot.csv``) skip until the pilot has been run;
the calibration count is raised before the declared run if the pilot's per-arm rule asks for
more. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import beta, binom, norm

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_contrast_comparators_law as law
from tests.studies import _learned_rule_split_mean_law as split_mean_law
from tests.studies import learned_rule_contrast_comparators as study
from tests.studies import learned_rule_contrast_comparators_pilot as pilot
from tests.studies import learned_rule_contrast_comparators_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("learned_rule_contrast_comparators_design")

base = law.base

# ----------------------------------------------------------------------------- the law

#: The fixed truths of each law: ``Static(0)``, ``coin``, ``offer`` and the natural course.
TRUTHS = {
    "non_exceptional": {"static_0": 0.51218, "coin": 0.52190, "offer": 0.54434, "natural": 0.52657},
    "misspecified_limit": {
        "static_0": 0.51218,
        "coin": 0.51151,
        "offer": 0.53340,
        "natural": 0.51558,
    },
}


@pytest.mark.parametrize("scenario", law.SCENARIOS)
def test_the_comparator_truths(scenario: str) -> None:
    for comparator, declared in TRUTHS[scenario].items():
        assert law.comparator_value(comparator, scenario) == pytest.approx(declared, abs=1e-5)
    treat_all = base.fixed_rule_grid(lambda w1, _w2: np.ones(w1.shape, dtype=bool))
    psi_1 = base.rule_value(treat_all, scenario)[0]
    psi_0 = law.comparator_value("static_0", scenario)
    assert law.comparator_value("coin", scenario) == pytest.approx(0.5 * (psi_1 + psi_0), abs=1e-10)


def _quad_value(integrand: Callable[[float, float], float], split: bool) -> float:
    """``E`` of ``integrand(W1, W2)`` by adaptive quadrature, split at ``W1 = 0`` if asked."""
    total = 0.0
    for w2 in base.W2_LEVELS:
        pieces = ((-1.0, 0.0), (0.0, 1.0)) if split else ((-1.0, 1.0),)
        for low, high in pieces:
            total += 0.25 * quad(integrand, low, high, args=(w2,), epsabs=1e-13)[0]
    return total


@pytest.mark.parametrize("scenario", law.SCENARIOS)
def test_a_second_quadrature_gives_the_same_truths(scenario: str) -> None:
    """Adaptive quadrature, split at the offer's jump, written apart from the law module."""

    def q(a: float, w1: float, w2: float) -> float:
        return float(base.qbar0(a, np.array([w1]), w2, scenario)[0])

    def g(w1: float, w2: float) -> float:
        return float(base.g0(np.array([w1]), w2)[0])

    def offer(w1: float, w2: float) -> float:
        p = 0.5 if w1 > 0 else 0.0
        return p * q(1, w1, w2) + (1 - p) * q(0, w1, w2)

    def natural(w1: float, w2: float) -> float:
        return g(w1, w2) * q(1, w1, w2) + (1 - g(w1, w2)) * q(0, w1, w2)

    assert law.comparator_value("offer", scenario) == pytest.approx(
        _quad_value(offer, split=True), abs=1e-8
    )
    assert law.comparator_value("natural", scenario) == pytest.approx(
        _quad_value(natural, split=False), abs=1e-8
    )


def test_the_efficiency_bounds() -> None:
    """Each calibration tag's bound at the limit rule of the declared learner's population
    projection; the learned value's is RM30's 0.664438 and the T44 tags' are T44's."""
    bounds = law.efficiency_bounds(
        split_mean_law.population_rule("non_exceptional"), "non_exceptional"
    )
    for tag, declared in study.TAG_BOUNDS.items():
        assert bounds[tag] == pytest.approx(declared, abs=2e-6), tag
    assert bounds["value"] == pytest.approx(0.664438, abs=2e-6)
    assert bounds["known_0"] == pytest.approx(0.688568, abs=2e-6)
    assert set(study.EFFICIENCY_BOUNDS) == set(study.CALIBRATION_LABELS)
    mu = law.comparator_value("natural", "non_exceptional")
    assert mu * (1 - mu) == pytest.approx(0.2493, abs=5e-5)
    assert study.TAG_BOUNDS["natural"] ** 2 == pytest.approx(mu * (1 - mu), abs=1e-6)


#: The Monte Carlo cross-check of the curve variances: draws of ``non_exceptional`` at a fixed
#: seed, used for nothing else.
MONTE_CARLO_DRAWS = 400_000
MONTE_CARLO_SEED = 20265821


def test_monte_carlo_agrees_with_the_quadrature_variances() -> None:
    """``Var(Y) = mu0 (1 - mu0)``, the coin's curve and the natural-course contrast at the limit
    rule, each within four Monte Carlo SEs of the quadrature."""
    scenario = "non_exceptional"
    frame = base.draw(scenario, MONTE_CARLO_DRAWS, MONTE_CARLO_SEED)
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    a, y = frame["A"].to_numpy(), frame["Y"].to_numpy()
    q0, q1 = base.qbar0(0.0, w1, w2, scenario), base.qbar0(1.0, w1, w2, scenario)
    qa = np.where(a == 1, q1, q0)
    treated = base.g0(w1, w2)
    ga = np.where(a == 1, treated, 1 - treated)
    model = split_mean_law.population_model(scenario)
    blip = base._blip(model, np.column_stack([w1, w2]))
    rule = blip > 0.0
    q_d = np.where(rule, q1, q0)
    g_d = np.where(rule, treated, 1 - treated)
    learned = (a == rule) * (y - q_d) / g_d + q_d
    coin = 0.5 / ga * (y - qa) + 0.5 * (q0 + q1)
    curves = {"natural": y, "coin": coin, "contrast_natural": learned - y}
    limit = split_mean_law.population_rule(scenario)
    for tag, curve in curves.items():
        centred = (curve - curve.mean()) ** 2
        estimate = centred.mean()
        mc_se = centred.std(ddof=1) / np.sqrt(MONTE_CARLO_DRAWS)
        assert abs(estimate - law.curve_variance(tag, limit, scenario)) < 4 * mc_se, tag


# ---------------------------------------------------------------- the predicted SE ratios

#: The predicted reported over true SE of each fit-C parameter on ``misspecified_limit``, at the
#: declared learner's population limit and the true ``g0``, with ``T_g`` the span of the
#: logistic treatment model's scores ``(A - g0)(1, W1, W2)``. A column that shares the fitted
#: ``g`` has true curve ``D - Pi(D | T_g)``; the natural course reads no ``g``, so the
#: natural-course contrast's true curve is ``(D_L - Pi(D_L | T_g)) - D_N``. The integrals are
#: exact sums over ``(W1, W2, A, Y)`` on the RM30 grid, not Monte Carlo.
PREDICTED_SE_RATIOS = {
    "value": 1.000150,
    "known_0": 1.000000,
    "coin": 1.000048,
    "offer": 1.000048,
    "natural": 1.000000,
    "contrast_0": 1.000150,
    "contrast_coin": 1.000575,
    "contrast_offer": 1.000149,
    "contrast_natural": 0.997694,
}


def _limit_arms(scenario: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    model = split_mean_law.population_model(scenario)
    arms = {}
    for w2 in base.W2_LEVELS:
        rows = [np.column_stack([np.full_like(base.GRID, a), base.GRID, np.full_like(base.GRID, w2)])
                for a in (0.0, 1.0)]  # fmt: skip
        arms[w2] = tuple(model.predict_proba(design)[:, 1] for design in rows)
    return arms


def _predicted_se_ratios(
    arms: dict[float, tuple[np.ndarray, np.ndarray]], scenario: str, *, shared_natural: bool = False
) -> dict[str, float]:
    """Each fit-C parameter's predicted SE ratio. ``shared_natural`` is the deliberate mutation
    that projects the natural-course contrast as if both columns read ``g``."""
    mass = np.full(base.GRID.size, base.GRID[1] - base.GRID[0])
    mass[[0, -1]] /= 2.0
    mass *= 0.25
    weights, scores = [], []
    curves: dict[str, list[np.ndarray]] = {}
    for w2 in base.W2_LEVELS:
        treated = base.g0(base.GRID, w2)
        q = arms[w2]
        rule = q[1] > q[0]
        tangent = np.column_stack([np.ones_like(base.GRID), base.GRID, np.full_like(base.GRID, w2)])
        offered = {
            "known_0": np.zeros_like(base.GRID),
            "coin": np.full_like(base.GRID, 0.5),
            "offer": law.offer_grid(w2),
        }
        q_d = np.where(rule, q[1], q[0])
        g_d = np.where(rule, treated, 1 - treated)
        for a in (0, 1):
            outcome = base.qbar0(a, base.GRID, w2, scenario)
            g_a = treated if a else 1 - treated
            for y in (0, 1):
                weights.append(mass * g_a * (outcome if y else 1 - outcome))
                scores.append((a - treated)[:, None] * tangent)
                parts = {"value": (a == rule) * (y - q_d) / g_d + q_d}
                for tag, p in offered.items():
                    p_a = p if a else 1 - p
                    parts[tag] = p_a / g_a * (y - q[a]) + p * q[1] + (1 - p) * q[0]
                parts["natural"] = np.full_like(base.GRID, float(y))
                for tag in ("known_0", "coin", "offer", "natural"):
                    stem = "0" if tag == "known_0" else tag
                    parts[f"contrast_{stem}"] = parts["value"] - parts[tag]
                for tag, part in parts.items():
                    curves.setdefault(tag, []).append(part)
    p = np.concatenate(weights)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    score = np.vstack(scores)
    gram = (score * p[:, None]).T @ score

    def residual(d: np.ndarray) -> np.ndarray:
        return d - score @ np.linalg.solve(gram, (score * p[:, None]).T @ d)

    def variance(d: np.ndarray) -> float:
        return float(p @ (d - p @ d) ** 2)

    ratios = {}
    for tag, pieces in curves.items():
        d = np.concatenate(pieces)
        if tag == "natural":
            true = d
        elif tag == "contrast_natural" and not shared_natural:
            true = residual(np.concatenate(curves["value"])) - np.concatenate(curves["natural"])
        else:
            true = residual(d)
        ratios[tag] = float(np.sqrt(variance(d) / variance(true)))
    return ratios


def test_the_predicted_se_ratios_on_the_misspecified_law() -> None:
    """The shared-g columns are conservative by at most 1.0006, too little for any cell to see;
    the natural-course contrast is anti-conservative at about 0.998, which the coverage cells
    cannot tell from 1 either (the T58 limits say so). On ``non_exceptional``, where the outcome
    model is correct, every ratio is 1."""
    ratios = _predicted_se_ratios(_limit_arms("misspecified_limit"), "misspecified_limit")
    assert ratios == pytest.approx(PREDICTED_SE_RATIOS, abs=2e-6)
    for tag, ratio in ratios.items():
        if tag == "contrast_natural":
            assert 0.995 < ratio < 0.999
        else:
            assert 1.0 <= ratio <= 1.001, tag
    correct = _predicted_se_ratios(_limit_arms("non_exceptional"), "non_exceptional")
    assert all(ratio == pytest.approx(1.0, abs=1e-4) for ratio in correct.values())
    # The deliberate mutation: projecting the natural-course contrast as a shared-g column
    # turns its ratio above 1, so the anti-conservative sign comes from the unshared column.
    mutated = _predicted_se_ratios(
        _limit_arms("misspecified_limit"), "misspecified_limit", shared_natural=True
    )
    assert mutated["contrast_natural"] > 1.0


# ---------------------------------------------------------------- the necessity learner

#: The ``drop_w1`` learner's limit treats every row. The untargeted offer contrast at that rule
#: converges to the confounded cell-mean difference, 0.0448 above the truth, about 2.6 of the
#: contrast's SEs at n = 2,000.
DROP_W1_OFFER_BIAS = 0.044815


def _cell_mean(w2: float, a: int) -> float:
    treated = base.g0(base.GRID, w2)
    arm = treated if a else 1.0 - treated
    outcome = np.trapezoid(arm * base.qbar0(a, base.GRID, w2, "non_exceptional"), base.GRID)
    return float(outcome / np.trapezoid(arm, base.GRID))


def test_the_necessity_control_has_a_visible_bias() -> None:
    means = {(w2, a): _cell_mean(w2, a) for w2 in base.W2_LEVELS for a in (0, 1)}
    assert all(means[w2, 1] > means[w2, 0] for w2 in base.W2_LEVELS)
    learned = float(np.mean([means[w2, 1] for w2 in base.W2_LEVELS]))
    offered = sum(
        0.25
        * np.trapezoid(
            law.offer_grid(w2) * means[w2, 1] + (1 - law.offer_grid(w2)) * means[w2, 0], base.GRID
        )
        for w2 in base.W2_LEVELS
    )
    treat_all = base.fixed_rule_grid(lambda w1, _w2: np.ones(w1.shape, dtype=bool))
    truth = base.rule_value(treat_all, "non_exceptional")[0] - law.comparator_value(
        "offer", "non_exceptional"
    )
    bias = learned - offered - truth
    assert bias == pytest.approx(DROP_W1_OFFER_BIAS, abs=1e-6)
    se = np.sqrt(law.curve_variance("contrast_offer", treat_all, "non_exceptional") / 2_000)
    assert bias / se > 2.5


# ------------------------------------------------------------------------ the declaration


def test_the_cells() -> None:
    assert tuple(law.TAGS) == (
        "value",
        "known_0",
        "coin",
        "offer",
        "natural",
        "contrast_0",
        "contrast_coin",
        "contrast_offer",
        "contrast_natural",
        "alone_value",
        "alone_contrast_natural",
    )
    assert study.REPORTS == ("fold_evaluated", "stacked")
    calibration = [
        f"non_exceptional_{report}_{tag}__{kind}"
        for report in study.REPORTS
        for tag in study.CALIBRATION_TAGS
        for kind in ("declared", "shrunken_se_control", "noise_control")
    ]
    coverage = [
        f"misspecified_limit_{report}_{tag}__{kind}"
        for report in study.REPORTS
        for tag in law.CONTRAST_TAGS
        for kind in ("declared", "shrunken_coverage_control")
    ]
    contrasts = [
        f"{scenario}_{report}_{tag}"
        for scenario in law.SCENARIOS
        for report in study.REPORTS
        for tag in law.CONTRAST_TAGS
    ]
    assert {
        "interval_calibration": tuple(calibration),
        "conservative_coverage": tuple(coverage),
        "contrast_covariance": tuple(f"{c}__correlation_one_control" for c in contrasts),
        "independent_se_diagnostic": tuple(f"{c}__independent_se" for c in contrasts),
        "targeting_necessity": (
            "drop_w1_contrast_offer__targeted",
            "drop_w1_contrast_offer__untargeted",
        ),
    } == study.PROPERTY_CELLS
    assert len(calibration) == 48 and len(coverage) == 20 and len(contrasts) == 20
    assert {"non_exceptional": tuple(law.LABELS.values())} == study.SCENARIOS
    assert len(set(law.LABELS.values())) == len(law.TAGS)
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert properties.TARGETING_DISPLACEMENT == 0.5
    assert study.COVERAGE_SHRINK == 0.60
    assert study.CORRELATION_ONE_CEILING == 0.80
    assert study.STUDY.truth_varies_by_replicate and not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == 2_000


def test_the_study_is_declared_and_not_registered() -> None:
    """No declared run has been made, so the record is not in the register and no row is."""
    assert study.STUDY.slug not in {record.slug for record in registered()}
    assert not (study.STUDY.artifacts / "manifest.json").exists()


def test_the_fits_are_the_declared_ones() -> None:
    combined = law.estimator(base.outcome_learner(), 5, "combined")
    assert [type(r).__name__ for r in combined.interventions] == [
        "Static",
        "Stochastic",
        "Stochastic",
    ]
    assert combined.natural_course is not None
    assert combined.natural_course.name == law.NATURAL_COURSE
    alone = law.estimator(base.outcome_learner(), 5, "natural_course")
    assert not alone.interventions and alone.natural_course is not None
    for estimator in (combined, alone):
        assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
        assert estimator.cv_evaluation and estimator.targeting_scheme == "pooled"
        assert estimator.n_folds == 10 and estimator.g_bounds == "auto"
    stacked = law.estimator(base.outcome_learner(), 5, "combined", cv_evaluation=False)
    assert not stacked.cv_evaluation


def test_the_manifest_lists_the_fit_path() -> None:
    for module in (
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/targets/builtin.py",
        "tests/studies/_learned_rule_contrast_comparators_law.py",
        "tests/studies/_learned_rule_law.py",
    ):
        assert module in study.STUDY.modules
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def _key(fit: str, *, cv_evaluation: bool, without_natural_course: bool = False) -> Any:
    estimator = law.estimator(base.outcome_learner(), 7, fit, cv_evaluation=cv_evaluation)
    if without_natural_course:
        estimator = TMLE(
            learned_rule=LearnedRule(),
            interventions=law.regimes(fit)[:-1],
            n_folds=base.N_FOLDS,
            cv_evaluation=cv_evaluation,
            targeting_scheme="pooled",
            g_bounds="auto",
            outcome_learner=base.outcome_learner(),
            treatment_learner=base.treatment_learner(),
            random_state=7,
        )
    frame = base.draw("non_exceptional", 400, 11)
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
    return result.single().config.support_key


@pytest.mark.parametrize("report", study.REPORTS)
def test_each_row_declares_the_key_its_fit_reads(report: str) -> None:
    """The rows' qualifier sets are the package's keys: fit C's, fit N's, and the key of the
    stochastic configuration ``[Static(0), coin, offer]`` that fit C's columns measure."""
    cv = law.REPORTS[report]
    keys = {
        "combined": _key("combined", cv_evaluation=cv),
        "natural_course": _key("natural_course", cv_evaluation=cv),
        "stochastic": _key("combined", cv_evaluation=cv, without_natural_course=True),
    }
    for row, key in keys.items():
        declared = study.SUPPORT_ROWS[f"{row}.{report}"]
        assert key.qualifiers == frozenset(declared["qualifiers"]), row
        assert key.splitting == declared["splitting"], row


def test_every_tag_is_gated_by_a_row() -> None:
    covered = set().union(*(set(tags) for tags in study.ROW_TAGS.values()))
    assert covered == set(law.TAGS)
    combined = {tag for tag, (fit, _) in law.TAGS.items() if fit == "combined"}
    assert set(study.ROW_TAGS["combined"]) == combined
    assert set(study.ROW_TAGS) == {row.split(".")[0] for row in study.SUPPORT_ROWS}


def test_the_declared_gates_are_read() -> None:
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


def test_the_limits_carry_the_natural_course_condition() -> None:
    assert "no conservativeness result" in study.ROW_LIMITS["natural_course"]
    assert "0.998" in study.ROW_LIMITS["natural_course"]
    assert "coverage only" in study.ROW_LIMITS["stochastic"]


# ---------------------------------------------------------------------------- the sizing

#: The SD of an SE-ratio estimate at ``R`` draws and ratio ``r`` is read as ``k r / sqrt(2 R)``,
#: with ``k`` set so that a cell at ratio 1 passes the 99% band 0.93 to 1.07 with probability
#: 0.95 at 2,400 draws, the framework's stated calibration pass probability.
K = 1.069
Z99 = norm.ppf(0.995)


def se_ratio_fail(ratio: float, replicates: int, k: float = K) -> float:
    """The chance that the 99% interval of a cell's SE ratio leaves 0.93 to 1.07."""
    low, high = study.STUDY.margins.calibration_se_ratio
    spread = k * ratio / np.sqrt(2 * replicates)
    return float(
        norm.cdf((low + Z99 * spread - ratio) / spread)
        + norm.sf((high - Z99 * spread - ratio) / spread)
    )


def _cp_bounds(replicates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hits = np.arange(replicates + 1)
    lower = np.where(hits > 0, beta.ppf(0.005, hits, replicates - hits + 1), 0.0)
    upper = np.where(hits < replicates, beta.ppf(0.995, hits + 1, replicates - hits), 1.0)
    return hits, lower, upper


def floor_fail(coverage: float, replicates: int, floor: float) -> float:
    """The exact chance that the Clopper-Pearson 99% lower end falls below ``floor``."""
    hits, lower, _ = _cp_bounds(replicates)
    return float(binom.pmf(hits, replicates, coverage)[lower < floor].sum())


def band_fail(coverage: float, replicates: int, band: tuple[float, float]) -> float:
    """The exact chance that the Clopper-Pearson 99% interval leaves ``band``."""
    hits, lower, upper = _cp_bounds(replicates)
    outside = (lower < band[0]) | (upper > band[1])
    return float(binom.pmf(hits, replicates, coverage)[outside].sum())


def bias_fail(replicates: int, margin: float = 0.25) -> float:
    """The chance that the 99% interval of the mean standardized error leaves ``±margin``."""
    return float(2 * norm.sf((margin - Z99 / np.sqrt(replicates)) * np.sqrt(replicates)))


CELLS = len(study.CALIBRATION_LABELS)


def test_k_matches_the_stated_pass_probability() -> None:
    assert 1 - se_ratio_fail(1.0, 2_400) == pytest.approx(0.95, abs=1e-3)


#: The count at which the planning figures below are read: the declared calibration count
#: before any pilot. A pilot may raise ``CALIBRATION_REPLICATES``; these figures stay pinned here.
PLANNED_CALIBRATION = 6_400


def test_the_calibration_count() -> None:
    """The planned count is the per-arm rule at ideal ratios: 6,400, with family pass about 0.957.
    A pilot may only raise it. The arm-pooled rule, meant for exchangeable arms of one estimator,
    would give 4,800 here."""
    assert CELLS == 16
    ideal = [1.0] * CELLS
    sized = tools.per_arm_replicates(ideal, pilot.PILOT_DRAWS)
    assert sized["R"] == PLANNED_CALIBRATION
    assert PLANNED_CALIBRATION <= properties.CALIBRATION_REPLICATES
    assert sized["family_wise"] == pytest.approx(0.957, abs=1e-3)
    assert tools.replicates_from_pilot(ideal, pilot.PILOT_DRAWS)["R"] == 4_800
    assert properties.CALIBRATION_REPLICATES <= properties.CALIBRATION_REPLICATES_CAP


#: Core-seconds per draw, read from the smoke runs: one draw is two fits under two reports.
SECONDS_PER_DRAW = {"primary": 0.5, "calibration": 0.9, "coverage": 0.9, "necessity": 0.3}
#: The declared budget of the study, the 4,000-draw pilot (about 1 CPU-hour) outside it.
BUDGET_SECONDS = 8 * 3_600


def _study_seconds(calibration: int) -> float:
    return (
        len(study.SCENARIOS) * study.PRIMARY_REPLICATES * SECONDS_PER_DRAW["primary"]
        + calibration * SECONDS_PER_DRAW["calibration"]
        + properties.COVERAGE_REPLICATES * SECONDS_PER_DRAW["coverage"]
        + properties.NECESSITY_REPLICATES * SECONDS_PER_DRAW["necessity"]
    )


def test_the_calibration_cap_is_the_budget() -> None:
    """The run is about 3 CPU-hours at 6,800 calibration draws and about 7.9 at the cap of
    26,400; the budget of 8 CPU-hours, the pilot outside it, holds the cap and not one step
    more."""
    assert _study_seconds(6_800) / 3_600 == pytest.approx(3.0, abs=0.1)
    cap = properties.CALIBRATION_REPLICATES_CAP
    assert _study_seconds(cap) / 3_600 == pytest.approx(7.9, abs=0.1)
    assert _study_seconds(cap) <= BUDGET_SECONDS < _study_seconds(cap + 400)


def test_a_well_behaved_pilot_fits_under_the_cap() -> None:
    """Over 400 simulated pilots of sixteen cells at a true SE ratio of 1, each read with the
    pilot's own error, the per-arm count stays at or under the cap in about 95% of them (a run
    of 2,000 put the 95th percentile at 26,400 and the median at 12,000)."""
    rng = np.random.default_rng(pilot.PILOT_SEED)
    sigma = tools.pilot_sigma(pilot.PILOT_DRAWS)
    counts = []
    for _ in range(400):
        ratios = 1.0 + rng.normal(0.0, sigma, CELLS)
        sized = tools.per_arm_replicates(ratios.tolist(), pilot.PILOT_DRAWS)["R"]
        counts.append(np.inf if sized is None else sized)
    inside = float(np.mean(np.asarray(counts) <= properties.CALIBRATION_REPLICATES_CAP))
    assert 0.92 <= inside <= 0.98, inside


def test_a_cell_at_ratio_098_under_the_per_arm_rule() -> None:
    """A pilot that reads one cell at 0.98 and the rest at 1 raises the count above 6,400, and
    the raised count stays under the cap."""
    sized = tools.per_arm_replicates([0.98] + [1.0] * (CELLS - 1), pilot.PILOT_DRAWS)
    assert sized["R"] is not None
    assert PLANNED_CALIBRATION < sized["R"] <= properties.CALIBRATION_REPLICATES_CAP


def test_the_primary_and_coverage_counts() -> None:
    """Each primary coverage cell at 1,600 and each conservative cell at 4,000 has a false-red
    chance of at most 0.002 at T44-like realized coverage (0.94 and 0.942); 400 fewer draws
    would not."""
    floor = study.STUDY.margins.coverage_floor
    calibration_floor = study.STUDY.margins.calibration_coverage[0]
    assert study.PRIMARY_REPLICATES == 1_600
    assert properties.COVERAGE_REPLICATES == 4_000
    assert floor_fail(0.95, 1_600, floor) == pytest.approx(7.7e-8, rel=0.05)
    assert floor_fail(0.94, 1_600, floor) == pytest.approx(3.7e-4, rel=0.05)
    assert floor_fail(0.95, 1_200, floor) == pytest.approx(1.8e-5, rel=0.05)
    assert floor_fail(0.94, 1_200, floor) == pytest.approx(0.0058, rel=0.05)
    assert floor_fail(0.949, 4_000, calibration_floor) == pytest.approx(3.4e-7, rel=0.05)
    assert floor_fail(0.942, 4_000, calibration_floor) == pytest.approx(0.0017, rel=0.05)
    assert floor_fail(0.942, 3_600, calibration_floor) > 0.002
    assert floor_fail(0.949, 3_200, calibration_floor) == pytest.approx(2.1e-5, rel=0.05)
    assert floor_fail(0.942, 3_200, calibration_floor) == pytest.approx(0.011, rel=0.05)
    assert properties.NECESSITY_REPLICATES == 1_200


def test_the_family_wise_false_red_chance() -> None:
    """Over every gated positive at the planned count and a true SE ratio of 1: about 3e-5 at
    perfect calibration, about 0.021 at T44-like realized coverage (0.94 primary, 0.942
    conservative). The per-arm rule's own figure, about 0.043, is larger because it also
    carries each cell's pilot error."""
    margins = study.STUDY.margins
    primary = len(law.TAGS)
    conservative = len(study.COVERAGE_LABELS)
    calibration = CELLS * (
        se_ratio_fail(1.0, PLANNED_CALIBRATION)
        + band_fail(0.95, PLANNED_CALIBRATION, margins.calibration_coverage)
    )
    assert band_fail(0.95, PLANNED_CALIBRATION, margins.calibration_coverage) < 1e-6
    bias = (primary + conservative) * bias_fail(study.PRIMARY_REPLICATES)
    assert bias < 1e-10

    def total(primary_coverage: float, conservative_coverage: float) -> float:
        return (
            calibration
            + bias
            + primary
            * floor_fail(primary_coverage, study.PRIMARY_REPLICATES, margins.coverage_floor)
            + conservative
            * floor_fail(
                conservative_coverage,
                properties.COVERAGE_REPLICATES,
                margins.calibration_coverage[0],
            )
        )

    assert total(0.95, 0.949) == pytest.approx(2.6e-5, rel=0.1)
    assert total(0.94, 0.942) == pytest.approx(0.021, abs=1e-3)


def test_the_controls_have_power() -> None:
    """The coverage control at SE x 0.60 sits below the floor at every ratio up to 1.30, and the
    correlation-one control at the review's largest SE share (0.36 of the joint SE) covers about
    0.52, far below its ceiling 0.80."""
    floor = study.STUDY.margins.calibration_coverage[0]
    coverage = 2 * norm.cdf(base.CRITICAL * study.COVERAGE_SHRINK * 1.30) - 1
    half = Z99 * np.sqrt(coverage * (1 - coverage) / properties.COVERAGE_REPLICATES)
    assert coverage + half < floor
    correlation_one = 2 * norm.cdf(base.CRITICAL * 0.36) - 1
    assert correlation_one < 0.6 < study.CORRELATION_ONE_CEILING


# ------------------------------------------------------------------------------ the pilot


def _pilot() -> pd.DataFrame:
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the sizing pilot has not been run")
    return pd.read_csv(pilot.PILOT_CSV).set_index("label")


def _calibration_rows() -> pd.DataFrame:
    frame = _pilot()
    return frame.loc[
        [f"{report}_{tag}" for report in study.REPORTS for tag in study.CALIBRATION_TAGS]
    ]


def test_a_resumed_pilot_keeps_the_draw_after_a_cut_line(tmp_path: Any) -> None:
    """A stop mid-write leaves a line with no newline. The resume ends the file first, so the
    next draw keeps its own line and only the cut draw is lost (and run again)."""
    path = tmp_path / pilot.RAW_NAME
    path.write_text('{"replicate": 0}\n{"replicate": 1, "est')
    pilot.terminate(path)
    with path.open("a") as handle:
        handle.write('{"replicate": 2}\n')
    assert [record["replicate"] for record in pilot.records(path)] == [0, 2]
    assert pilot.valid_draws(path) == 2
    pilot.terminate(path)
    assert path.read_text().count("\n") == 3


def test_the_pilot_covers_every_label() -> None:
    frame = _pilot()
    expected = {f"{report}_{tag}" for report in study.REPORTS for tag in law.TAGS}
    assert set(frame.index) == expected
    assert (frame["replicates"] == pilot.PILOT_DRAWS).all()


def test_the_calibration_count_meets_the_pilot_rule() -> None:
    """The per-arm rule on the pilot's sixteen SE ratios, the one rule declared before the
    pilot. A count above the cap stops the run; a count above the declared one is copied into
    ``CALIBRATION_REPLICATES`` before the declared run."""
    sized = tools.per_arm_replicates(_calibration_rows()["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    replicates = sized["R"]
    assert replicates is not None and replicates <= properties.CALIBRATION_REPLICATES_CAP, (
        f"STOP: the per-arm rule asks for {replicates} draws, above the budget cap "
        f"{properties.CALIBRATION_REPLICATES_CAP}"
    )
    assert replicates <= properties.CALIBRATION_REPLICATES, (
        f"raise CALIBRATION_REPLICATES to {replicates} before the declared run"
    )


def plug_in_family(rows: pd.DataFrame, replicates: int) -> float:
    """The family pass probability with each cell's raw pilot ratio, kurtosis and coverage read
    as its truth. Pilot noise biases this figure low, so it is reported, not asserted."""
    family = 1.0
    for _, row in rows.iterrows():
        k = K * np.sqrt((row["excess_kurtosis"] + 2.0) / 2.0)
        family *= 1 - se_ratio_fail(float(row["se_ratio"]), replicates, k=float(k))
        family *= 1 - band_fail(
            float(row["coverage"]), replicates, study.STUDY.margins.calibration_coverage
        )
    return family


def test_the_family_at_the_pilot_values(capsys: pytest.CaptureFixture[str]) -> None:
    """Reports the plug-in family pass figure at the declared count. The count is set by the
    per-arm rule alone, so this figure gates nothing."""
    replicates = properties.CALIBRATION_REPLICATES
    family = plug_in_family(_calibration_rows(), replicates)
    assert 0.0 <= family <= 1.0
    with capsys.disabled():
        print(f"\nplug-in family pass {family:.3f} at {replicates} draws (reported, not gated)")


def test_the_primary_legs_at_the_pilot_values() -> None:
    frame = _pilot()
    floor = study.STUDY.margins.coverage_floor
    for tag in law.TAGS:
        row = frame.loc[f"fold_evaluated_{tag}"]
        assert abs(row["standardized_bias"]) + Z99 / np.sqrt(study.PRIMARY_REPLICATES) < 0.25, tag
        assert floor_fail(float(row["coverage"]), study.PRIMARY_REPLICATES, floor) < 0.01, tag


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def _x11_modules() -> list[tuple[Any, Any]]:
    from tests.unit.test_learned_rule_optimal_value_design import _x11_modules as modules

    return [pair for pair in modules() if pair[0].STUDY.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    for module in [*(m for m, _ in _x11_modules()), boundary, optimal]:
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others
    assert pilot.PILOT_SEED not in others | {study.SEED, study.RESAMPLING_SEED}


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


def test_rule_l3_keeps_every_label() -> None:
    assert properties.suffixes(study.STUDY) == {
        ("interval_calibration", "non_exceptional"): (),
        ("conservative_coverage", "misspecified_limit"): (),
        ("targeting_necessity", "non_exceptional_drop_w1"): (),
    }


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
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal
    from tests.studies import learned_rule_optimal_value_properties as optimal_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for module, module_properties in [*_x11_modules(), (optimal, optimal_properties)]:
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
    assert replicate_seed(study.STUDY, study.PRIMARY_LAW, 0) in declared


def test_no_pilot_seed_is_a_declared_seed() -> None:
    declared = learned_rule_run.declared_sample_seeds()
    drawn = {pilot.sample_seed(r) for r in range(pilot.PILOT_DRAWS)}
    assert len(drawn) == pilot.PILOT_DRAWS
    assert not drawn & declared
