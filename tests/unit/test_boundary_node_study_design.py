"""The design of ``boundary-node-ltmle``: its laws, grids, cells, rules and pilot.

The laws' figures are exact (:mod:`tests.studies.boundary_node_laws`), and each is checked here
against a Monte Carlo draw of the same law, so a wrong formula cannot pass by agreeing with
itself. The pilot (:mod:`tests.studies.boundary_node_pilot`) is read from its committed rows.
The declared rules (:mod:`tests.studies.boundary_node_properties`) are checked on constructed
rows, each against a mutation that must turn it red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies import boundary_node_laws as laws
from tests.studies import boundary_node_ltmle as study
from tests.studies import boundary_node_pilot as pilot
from tests.studies import boundary_node_properties as properties
from tests.studies import fractional_glm
from tests.studies.evidence import property_failures
from tests.studies.evidence.red_cells import owner_ids
from tests.studies.evidence.registry import registered

PILOT = Path(__file__).resolve().parents[1] / "canonical" / "boundary_node_ltmle" / "pilot.csv.gz"


def _draws(law: laws.Law, n: int, replicates: int, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return [law.draw(n, rng) for _ in range(replicates)]


def _follows(frame: pd.DataFrame, node: int, regimen: int) -> np.ndarray:
    decisions = (1, 3, 4)[: laws.DECISION_OF_NODE[node - 1]]
    followed = np.ones(len(frame), dtype=bool)
    for column in decisions:
        followed &= frame[f"A{column}"].to_numpy() == regimen
    return followed & frame[f"Y{node}"].notna().to_numpy()


def _flagged(frame: pd.DataFrame, node: int, regimen: int) -> bool:
    rows = _follows(frame, node, regimen)
    values = frame.loc[rows, f"Y{node}"].to_numpy()
    return bool(rows.any() and (np.all(values == 0) or np.all(values == 1)))


# -------------------------------------------------------------------------- the law


def test_the_mechanism_intercepts_give_the_declared_follow_probabilities() -> None:
    w, weight = np.polynomial.hermite_e.hermegauss(laws.QUADRATURE)
    weight = weight / weight.sum()
    law = laws.law_of("interior")
    for decision, follow in enumerate(laws.FOLLOW, start=1):
        assert float(weight @ law.mechanism(decision, w)) == pytest.approx(follow, abs=1e-10)


def test_the_chosen_point_is_the_declared_rule_on_the_declared_grid() -> None:
    feasible = [a for a in laws.ALPHA_GRID if all(laws.meets_constraints(laws.Law(a)).values())]
    assert laws.chosen_alpha() in feasible
    flags = {a: abs(laws.Law(a).flag_probability(4, 1, laws.DESIGN_N) - 0.5) for a in feasible}
    assert flags[laws.chosen_alpha()] == min(flags.values())


def test_the_interior_law_meets_the_three_constraints() -> None:
    law = laws.law_of("interior")
    assert laws.meets_constraints(law) == {
        "last_share": True,
        "last_flag": True,
        "early_flag": True,
    }


@pytest.mark.parametrize("node", [3, 4])
def test_the_exact_flag_probability_matches_a_draw(node: int) -> None:
    """Witness: the closed form against 3,000 draws at n = 200 (exact 99.9% interval)."""
    law = laws.law_of("interior")
    draws = _draws(law, laws.DESIGN_N, 3_000, seed=node)
    hits = sum(_flagged(frame, node, 1) for frame in draws)
    low, high = stats.binomtest(hits, len(draws)).proportion_ci(0.999)
    assert low <= law.flag_probability(node, 1, laws.DESIGN_N) <= high


def test_the_truth_matches_a_large_draw() -> None:
    law = laws.law_of("interior")
    big = law.draw(400_000, np.random.default_rng(11))
    # Under ``always`` every decision is 1: draw the counterfactual hazards directly.
    w = big["W"].to_numpy()
    survival = np.ones_like(w)
    for node in range(1, laws.NODES + 1):
        survival *= 1.0 - law.hazard(node, w, np.ones_like(w))
    assert float(np.mean(1.0 - survival)) == pytest.approx(law.risk(1), abs=4e-3)


def test_the_zero_variant_flags_node_four_and_keeps_an_interior_risk() -> None:
    law = laws.law_of("zero")
    assert law.flag_probability(4, 1, laws.DESIGN_N) > 0.999
    assert 0.05 < law.risk(1) < 0.95


def test_the_node_share_is_the_variance_of_the_node_term() -> None:
    """Witness: the share's node-4 variance against the sample variance of its term."""
    law = laws.law_of("interior")
    nodes, _ = law.node_variances(1)
    frame = law.draw(200_000, np.random.default_rng(5))
    w = frame["W"].to_numpy()
    one = np.ones_like(w)
    end = np.prod([1.0 - law.hazard(t, w, one) for t in range(1, 5)], axis=0)
    h4 = law.hazard(4, w, one)
    pi = law.mechanism(1, w) * law.mechanism(2, w) * law.mechanism(3, w)
    rows = _follows(frame, 4, 1)
    y = np.nan_to_num(frame["Y4"].to_numpy())
    before = np.prod([1.0 - law.hazard(t, w, one) for t in range(1, 4)], axis=0)
    term = rows / (pi * before) * end / (1.0 - h4) * (y - h4)
    assert float(np.var(term)) == pytest.approx(nodes[3], rel=0.06)


# ----------------------------------------------------------------------- the readings


def test_the_paired_lower_end_is_positive_only_with_gains() -> None:
    assert pilot.mcnemar_lower(40, 0, 1_000) > 0.0
    assert pilot.mcnemar_lower(20, 20, 1_000) < 0.0
    # Mutation control: counting the lost draws as gains would pass a tie.
    assert pilot.mcnemar_lower(40, 0, 1_000) > pilot.mcnemar_lower(20, 20, 1_000)


def test_the_newcombe_interval_contains_the_difference() -> None:
    low, high = pilot.newcombe_interval(900, 1_000, 1_900, 2_000)
    assert low < 0.9 - 0.95 < high
    low, high = pilot.newcombe_interval(800, 1_000, 1_900, 2_000)
    assert high < 0.0


# --------------------------------------------------------------------------- the pilot


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    if not PILOT.exists():
        pytest.skip("the pilot has not been committed")
    frame = pd.read_csv(PILOT, dtype={"lost": str, "withheld": str})
    frame["lost"] = frame["lost"].fillna("")
    frame["withheld"] = frame["withheld"].fillna("")
    for column in ("firing", "supplemented_here", "supplemented_covered", "plugin_covered"):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def test_the_pilot_holds_every_cell_and_draw(rows: pd.DataFrame) -> None:
    counts = rows.groupby(["cell", "target"]).size()
    assert set(counts.index.get_level_values(0)) == {cell.name for cell in pilot.CELLS}
    assert (counts == pilot.PILOT_REPLICATES).all()


def test_the_supplemented_interval_contains_the_plugin_interval(rows: pd.DataFrame) -> None:
    kept = rows[rows["supplemented_here"].astype(bool) & np.isfinite(rows["supplemented_low"])]
    assert len(kept) > 1_000
    assert (kept["supplemented_low"] <= kept["plugin_low"] + 1e-12).all()
    assert (kept["supplemented_high"] >= kept["plugin_high"] - 1e-12).all()


def _control(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[(rows["cell"] == "interior_n200") & (rows["target"] == "always")]


def test_the_paired_control_has_power_at_the_declared_size(rows: pd.DataFrame) -> None:
    """Both clauses of ``paired_attributable`` at n = 200, R = 4,000, on the 2,000-draw pilot.

    The predictive power draws the pilot's rates from their posteriors in each simulation.
    """
    assert len(_control(rows)) == 2_000
    for predictive in (False, True):
        power = pilot.control_power(_control(rows), predictive=predictive)
        assert power["clause_i"] >= 0.90, power
        assert power["clause_ii"] >= 0.90, power


def test_a_plugin_that_loses_nothing_has_no_power(rows: pd.DataFrame) -> None:
    """Mutation control: a plug-in that covers as the supplement does passes neither clause."""
    cell = _control(rows).copy()
    cell["plugin_covered"] = cell["supplemented_covered"]
    power = pilot.control_power(cell, predictive=True, simulations=400)
    assert power["clause_i"] < 0.05 and power["clause_ii"] < 0.05, power


#: Rows whose all-draw G1 pass probability at R = 4,000 is below 0.90 at the pilot rates. G1 on
#: all draws is published, not gated (plan rev 3.1). The 400-draw pilot put ``interior_n400``
#: ``always`` here at 0.918; the 2,000-draw pilot reads 0.941 and puts no row here.
BELOW_ALL_DRAW_G1: set[tuple[str, str]] = set()


def _g1_threshold(replicates: int = 4_000, floor: float = 0.90) -> int:
    hits = np.arange(int(floor * replicates), replicates + 1)
    lower = stats.beta.ppf(0.005, hits, replicates - hits + 1)
    return int(hits[np.argmax(lower >= floor)])


def _pass_probability(coverage: float, trials: int, floor: float) -> float:
    """The chance that the exact 99% lower end of ``trials`` draws reaches ``floor``."""
    hits = np.arange(0, trials + 1)
    lower = stats.beta.ppf(0.005, np.maximum(hits, 1), trials - hits + 1)
    lower[0] = 0.0
    first = int(np.argmax(lower >= floor))
    return float(stats.binom.sf(first - 1, trials, coverage))


def test_the_published_all_draw_shortfall_is_the_declared_one(rows: pd.DataFrame) -> None:
    threshold = _g1_threshold()
    below = set()
    for (cell, target), group in rows.groupby(["cell", "target"]):
        coverage = group["supplemented_covered"].astype(bool).mean()
        if stats.binom.sf(threshold - 1, 4_000, coverage) < 0.90:
            below.add((cell, target))
    assert below == BELOW_ALL_DRAW_G1


def test_every_gated_firing_floor_passes_at_the_pilot_rates(rows: pd.DataFrame) -> None:
    """Each gated row's pass probability over its expected firing draws, at least 0.99."""
    for name in properties.FIRING_CELLS:
        cell_name, target, _ = name.split("__")
        group = rows[(rows["cell"] == cell_name) & (rows["target"] == target)]
        fire = group["firing"].astype(bool)
        coverage = float(group.loc[fire, "supplemented_covered"].astype(bool).mean())
        trials = int(study.expected_firing(study.CELL_BY_NAME[cell_name], target))
        assert _pass_probability(coverage, trials, properties.FIRING_FLOOR) >= 0.99, name


def test_every_zero_cell_floor_passes_at_the_pilot_rates(rows: pd.DataFrame) -> None:
    for name in properties.ZERO_CELLS:
        cell_name, target, _ = name.split("__")
        group = rows[(rows["cell"] == cell_name) & (rows["target"] == target)]
        assert group["firing"].astype(bool).all()
        coverage = float(group["supplemented_covered"].astype(bool).mean())
        assert _pass_probability(coverage, 4_000, properties.ZERO_FLOOR) >= 0.99, name


def test_the_mutation_controls_of_the_pass_probability() -> None:
    """A coverage at the floor passes about half the time; one below it almost never."""
    assert _pass_probability(0.99, 600, 0.90) > 0.999
    assert _pass_probability(0.88, 600, 0.90) < 0.01


def _predictive_lost(lost: int, draws: int, replicates: int = 4_000) -> int:
    """The 95th percentile of the lost keys at ``replicates``, under the Jeffreys posterior."""
    return int(stats.betabinom.ppf(0.95, replicates, lost + 0.5, draws - lost + 0.5))


def _g2_worst_case(errors: np.ndarray, lost: int, margin: float = 0.25) -> bool:
    """G2 with ``lost`` errors filled at the survivor minimum or maximum, the margin held.

    The harness's S1 clause (``bias_clause``): the Student 99% interval must lie inside
    ``margin`` complete-case SDs at every vertex split. The upper end is convex and the lower
    concave in the fills, so the two extreme splits decide.
    """
    bound = margin * float(np.std(errors, ddof=1))
    splits = (0, lost) if lost else (0,)
    for low_fills in splits:
        filled = np.concatenate(
            [errors, np.full(low_fills, errors.min()), np.full(lost - low_fills, errors.max())]
        )
        half = stats.t.ppf(0.995, filled.size - 1) * filled.std(ddof=1) / np.sqrt(filled.size)
        if not (-bound <= filled.mean() - half and filled.mean() + half <= bound):
            return False
    return True


def _g2_pass(errors: np.ndarray, lost: int, simulations: int = 200, seed: int = 0) -> float:
    """The predictive chance that a 4,000-draw row passes G2 with ``lost`` keys lost.

    Each simulation resamples the pilot, then the study's survivors from that resample.
    """
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(simulations):
        posterior = rng.choice(errors, errors.size)
        passes += _g2_worst_case(rng.choice(posterior, 4_000 - lost), lost)
    return passes / simulations


def _floor_pass(hits: int, trials: int, study_trials: int, floor: float, seed: int = 0) -> float:
    """The predictive pass chance of an exact floor: the coverage drawn from its posterior."""
    rng = np.random.default_rng(seed)
    rates = rng.beta(hits + 0.5, trials - hits + 0.5, 200)
    return float(np.mean([_pass_probability(rate, study_trials, floor) for rate in rates]))


def _gated_pass_probabilities(rows: pd.DataFrame) -> dict[str, float]:
    """Every gated clause's predictive pass probability, keyed by clause and row."""
    out: dict[str, float] = {}
    for cell in study.CELLS:
        keys = rows[(rows["cell"] == cell.name) & (rows["target"] == "always")]
        lost = _predictive_lost(int((keys["lost"] != "").sum()), len(keys))
        for target in study.TARGETS:
            group = rows[
                (rows["cell"] == cell.name) & (rows["target"] == target) & (rows["lost"] == "")
            ]
            errors = (group["psi"] - group["truth"]).to_numpy(dtype=float)
            out[f"bias/{cell.name}__{target}"] = _g2_pass(errors, lost)
    for name in properties.FIRING_CELLS:
        cell_name, target, _ = name.split("__")
        group = rows[(rows["cell"] == cell_name) & (rows["target"] == target)]
        fire = group.loc[group["firing"], "supplemented_covered"]
        trials = int(study.expected_firing(study.CELL_BY_NAME[cell_name], target))
        out[f"firing/{name}"] = _floor_pass(
            int(fire.sum()), len(fire), trials, properties.FIRING_FLOOR
        )
    for name in properties.ZERO_CELLS:
        cell_name, target, _ = name.split("__")
        covered = rows[(rows["cell"] == cell_name) & (rows["target"] == target)]
        hits = int(covered["supplemented_covered"].sum())
        out[f"zero/{name}"] = _floor_pass(hits, len(covered), 4_000, properties.ZERO_FLOOR)
    power = pilot.control_power(_control(rows), predictive=True)
    out["paired/clause_i"], out["paired/clause_ii"] = power["clause_i"], power["clause_ii"]
    return out


@pytest.fixture(scope="module")
def chances(rows: pd.DataFrame) -> dict[str, float]:
    return _gated_pass_probabilities(rows)


def test_every_bias_row_passes_g2_at_the_predictive_worst_case(chances: dict[str, float]) -> None:
    """G2 at the 95th percentile of the lost keys, every lost error at its worst vertex."""
    bias = {key: value for key, value in chances.items() if key.startswith("bias/")}
    assert len(bias) == 24
    assert min(bias.values()) >= 0.99, bias


def test_the_gated_family_passes_with_probability_at_least_090(chances: dict[str, float]) -> None:
    """The product of every gated clause's predictive pass probability.

    The clauses are the 24 G2 rows, 19 firing floors, two zero cells and two paired clauses.
    """
    assert len(chances) == 24 + 19 + 2 + 2
    assert float(np.prod(list(chances.values()))) >= 0.90, chances


def test_the_g2_pass_probability_sees_a_bias_and_a_large_loss() -> None:
    """Mutation controls: a bias of 0.3 SD, or losses filled at far extremes, fail G2."""
    errors = np.random.default_rng(1).normal(size=2_000)
    assert _g2_pass(errors, 4, simulations=50) > 0.95
    assert _g2_pass(errors + 0.3, 4, simulations=50) < 0.05
    assert not _g2_worst_case(np.concatenate([errors, [40.0, -40.0]]), 400)


def _primary_legs(group: pd.DataFrame, simulations: int = 200, seed: int = 0) -> np.ndarray:
    """Predictive pass chances of the primary's bias, coverage and SE-ratio legs at R = 1,000."""
    rng = np.random.default_rng(seed)
    replicates = study.PRIMARY_REPLICATES
    psi = group["psi"].to_numpy(dtype=float)
    errors = psi - group["truth"].to_numpy(dtype=float)
    se = group["supplemented_std_error"].to_numpy(dtype=float)
    covered = group["supplemented_covered"].to_numpy(dtype=bool)
    threshold = _g1_threshold(replicates)
    low_band, high_band = study.STUDY.margins.se_ratio_sanity
    legs = np.zeros((simulations, 3), dtype=bool)
    for s in range(simulations):
        pick = rng.choice(rng.integers(0, psi.size, psi.size), replicates)
        e = errors[pick]
        bound = 0.25 * e.std(ddof=1)
        half = stats.t.ppf(0.995, replicates - 1) * e.std(ddof=1) / np.sqrt(replicates)
        legs[s, 0] = -bound <= e.mean() - half and e.mean() + half <= bound
        legs[s, 1] = covered[pick].sum() >= threshold
        boot = rng.integers(0, replicates, (200, replicates))
        ratios = se[pick][boot].mean(axis=1) / psi[pick][boot].std(axis=1, ddof=1)
        low, high = np.quantile(ratios, [0.005, 0.995])
        legs[s, 2] = low_band <= low and high <= high_band
    return legs.mean(axis=0)


def test_the_primary_passes_or_carries_its_declared_reading(rows: pd.DataFrame) -> None:
    """Each leg a primary row's reading does not name passes; each named leg is declared.

    The SE-ratio legs of ``risk[always]`` and the contrast are predicted red under
    ``X54-supplement-width``. The coverage leg of ``risk[never]`` is predicted to pass (0.945)
    and is declared under ``X54-rare-event-coverage`` if it is red. The record's margins are
    the harness defaults; no band is widened for the bound.
    """
    assert study.STUDY.margins.se_ratio_sanity == (0.80, 1.20)
    assert set(study.PRIMARY_RED_READINGS) == set(study.ESTIMANDS)
    assert {reading.owner for reading in study.PRIMARY_RED_READINGS.values()} <= set(owner_ids())
    for target, estimand in study.TARGETS.items():
        group = rows[(rows["cell"] == study.SCENARIO) & (rows["target"] == target)]
        chances = dict(zip(study.PRIMARY_LEGS, _primary_legs(group), strict=True))
        reading = study.PRIMARY_RED_READINGS[estimand]
        assert "bias_equivalent" not in reading.legs
        assert chances["bias_equivalent"] >= 0.99, (target, chances)
        for leg in study.PRIMARY_LEGS:
            if leg not in reading.legs:
                assert chances[leg] >= 0.90, (target, leg, chances)
        if target in ("always", "contrast"):
            assert reading.legs == ("se_calibrated",)
            assert chances["se_calibrated"] < 0.10, (target, chances)
        else:
            assert reading.legs == ("coverage_valid",)
            assert chances["coverage_valid"] >= 0.90, (target, chances)


def _performance(**failed: str) -> pd.DataFrame:
    """A ``performance-tests.csv`` whose every leg passes but the ``estimand=leg`` ones."""
    frame = pd.DataFrame({"estimand": list(study.ESTIMANDS)})
    for leg in study.PRIMARY_LEGS:
        frame[leg] = True
    for target, leg in failed.items():
        frame.loc[frame["estimand"] == study.TARGETS[target], leg] = False
    return frame


def test_an_owned_leg_failure_is_covered_and_any_other_is_not() -> None:
    """c-m1: a width owner claims the SE-ratio leg only, the rare-event owner coverage only."""
    assert study.unowned_primary_failures(_performance()) == []
    assert (
        study.unowned_primary_failures(
            _performance(always="se_calibrated", contrast="se_calibrated", never="coverage_valid")
        )
        == []
    )
    for target, leg in (
        ("always", "bias_equivalent"),
        ("always", "coverage_valid"),
        ("contrast", "coverage_valid"),
        ("never", "se_calibrated"),
        ("never", "bias_equivalent"),
    ):
        (failure,) = study.unowned_primary_failures(_performance(**{target: leg}))
        assert study.TARGETS[target] in failure and leg in failure


def test_a_registered_primary_fails_only_its_owned_legs() -> None:
    """At registration, the committed primary rows fail no leg outside their readings."""
    path = study.STUDY.artifact("performance-tests.csv")
    if not path.exists():
        pytest.skip("the study has not run")
    frame = pd.read_csv(path)
    assert study.unowned_primary_failures(frame[frame["implementation"] == "cleverly"]) == []


def test_the_primary_legs_see_a_short_standard_error(rows: pd.DataFrame) -> None:
    """Mutation control: halving the reported SE fails the SE-ratio leg of ``risk[never]``."""
    group = rows[(rows["cell"] == study.SCENARIO) & (rows["target"] == "never")].copy()
    group["supplemented_std_error"] *= 0.5
    assert _primary_legs(group, simulations=40)[2] < 0.05


# ------------------------------------------------------------------------------- law B


def test_law_b_is_chosen_by_the_declared_rule_on_its_grid() -> None:
    feasible = [
        a
        for a in laws.HELD_ALPHA_GRID
        if all(laws.meets_constraints(laws.Law(a, held=True)).values())
    ]
    chosen = laws.chosen_held_alpha()
    assert chosen in feasible
    flags = {
        a: abs(laws.Law(a, held=True).flag_probability(4, 1, laws.DESIGN_N) - 0.5) for a in feasible
    }
    assert flags[chosen] == min(flags.values())
    law = laws.law_of("held")
    assert laws.meets_constraints(law) == {"last_flag": True, "early_flag": True}
    # Constraint (1) is infeasible under a held treatment (plan 5.5): the share stays small.
    assert law.last_share() < 0.10


def test_law_b_flag_probability_matches_a_draw() -> None:
    """Witness: node 4 of law B flagged under ``always`` against 3,000 draws (99.9%)."""
    law = laws.law_of("held")
    rng = np.random.default_rng(23)
    hits = 0
    for _ in range(3_000):
        frame = law.draw(laws.DESIGN_N, rng)
        # Node 4's risk set under ``always``: treated, and no event before node 4.
        values = frame.loc[(frame["A"] == 1) & (frame["time"] >= 4), "event"].to_numpy()
        hits += bool(len(values) and (np.all(values == 0) or np.all(values == 1)))
    low, high = stats.binomtest(hits, 3_000).proportion_ci(0.999)
    assert low <= law.flag_probability(4, 1, laws.DESIGN_N) <= high


def test_law_b_truth_matches_a_large_draw() -> None:
    law = laws.law_of("held")
    w = np.random.default_rng(29).normal(size=400_000)
    survival = np.ones_like(w)
    for node in range(1, laws.NODES + 1):
        survival *= 1.0 - law.hazard(node, w, np.ones_like(w))
    assert float(np.mean(1.0 - survival)) == pytest.approx(law.risk(1), abs=4e-3)


def test_the_clustered_flag_probability_matches_a_draw() -> None:
    """Witness: the cluster-effect form of the flag probability against 3,000 draws."""
    law = laws.law_of("clustered")
    rng = np.random.default_rng(31)
    hits = sum(_flagged(law.draw(400, rng, cluster_size=10), 4, 1) for _ in range(3_000))
    low, high = stats.binomtest(hits, 3_000).proportion_ci(0.999)
    assert low <= law.flag_probability(4, 1, 400, 10) <= high
    # Mutation control: the independent-row form is outside the same interval or the effect
    # moves it, so a formula that ignored the clusters could not pass by accident.
    assert law.flag_probability(4, 1, 400, 10) != pytest.approx(law.flag_probability(4, 1, 400))


# ------------------------------------------------------------------------------ the cells


def test_the_record_publishes_every_declared_cell() -> None:
    declared = {cell.cell for cell in properties.declared_cells()}
    assert set(study.STUDY.property_cells[study.FAMILY]) == declared
    assert len(declared) == len(study.CELLS) * len(study.TARGETS) * len(study.ARMS) == 72
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    assert {name for name, role in roles.items() if role == "control"} == {properties.CONTROL_CELL}
    assert {name for name, role in roles.items() if role == "positive"} == {
        name for name in declared if name.endswith("__supplemented")
    }


def test_the_firing_floor_is_gated_where_600_firing_draws_are_expected() -> None:
    gated = {
        study.cell_name(cell.name, target, "supplemented")
        for cell in study.CELLS
        for target in study.TARGETS
        if study.expected_firing(cell, target) >= 600
    }
    assert set(properties.FIRING_CELLS) == gated
    assert len(gated) == 19
    # Every law A interior cell gates ``always`` with 920 or more expected firing draws.
    for name in ("interior_n100", "interior_n200", "interior_n400", "interior_n200_crossfit"):
        assert study.expected_firing(study.CELL_BY_NAME[name], "always") >= 920
    assert set(properties.ZERO_CELLS) == {
        "zero_n200__always__supplemented",
        "held_zero_n200__always__supplemented",
    }


def test_the_seeds_are_the_study_s_own() -> None:
    seeds = {s for record in registered() for s in (record.seed, record.resampling_seed)}
    assert not {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED} & seeds
    assert len({study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED}) == 3


def test_declared_and_not_yet_registered() -> None:
    from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

    assert study.STUDY.slug not in {record.slug for record in registered()}
    assert not study.STUDY.artifact("property-replicates.parquet").exists()
    assert properties.__name__ in UNCOMMITTED_MODULES
    assert study.STUDY.publication_policy == "reporting"
    assert properties.FAILURE_BOUND == 0.05
    assert study.STUDY.failure_bound == {"cleverly": 0.05}


def _separated_draw() -> tuple[pd.DataFrame, study.Cell]:
    """Pilot draw 1726 of the cross-fitted cell, whose pseudo-outcome the default GLM raised on."""
    cell = study.CELL_BY_NAME["interior_n200_crossfit"]
    rng = np.random.default_rng(pilot.draw_seed(cell.name, 1726))
    return laws.law_of(cell.variant).draw(cell.n, rng), cell


def test_the_study_glm_fits_a_numerically_separated_draw() -> None:
    """The draw that lost its key in the first pilot fits: no lost-replicate source remains."""
    frame, cell = _separated_draw()
    result = study.fit(frame, cell)
    assert np.isfinite(result[study.TARGETS["always"]].psi)


def test_the_draw_that_stopped_the_first_run_fits() -> None:
    """Property draw 130 of the cross-fitted cell: a target separated once rounded, one value
    ``4.2e-4`` from 1. The flagged learner continues and fits."""
    cell = study.CELL_BY_NAME["interior_n200_crossfit"]
    frame = properties._draw((cell.name, 130))
    result = study.fit(frame, cell)
    assert np.isfinite(result[study.TARGETS["always"]].psi)


def test_a_study_glm_failure_is_a_lost_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit the flagged learner cannot finish costs its key; it does not stop the run."""
    from cleverly.validation._failures import recordable

    def fail(self: object, *args: object, **kwargs: object) -> None:
        raise fractional_glm.QuasiBinomialConvergenceError("quasibinomial IRLS did not converge")

    frame, cell = _separated_draw()
    monkeypatch.setattr(fractional_glm.QuasiBinomialGLM, "fit", fail)
    with pytest.raises(fractional_glm.QuasiBinomialConvergenceError) as caught:
        study.fit(frame, cell)
    assert recordable(caught.value)


def test_without_the_flag_the_separated_draw_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: without the numerical-separation branches the draw raises."""
    frame, cell = _separated_draw()
    monkeypatch.setattr(fractional_glm, "numerically_separated", lambda *args: False)
    real = fractional_glm.completely_separated

    def exact_only(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> bool:
        binary = np.all((target == 0.0) | (target == 1.0))
        return bool(binary and real(design, target, weights))

    monkeypatch.setattr(fractional_glm, "completely_separated", exact_only)
    with pytest.raises(RuntimeError, match="quasibinomial IRLS did not converge"):
        study.fit(frame, cell)


# ------------------------------------------------------------------------------ the rules


def _rows(seed: int = 0, replicates: int = 4_000) -> pd.DataFrame:
    """Rows of every declared cell where each gated rule holds by a wide margin."""
    rng = np.random.default_rng(seed)
    out = []
    for cell in study.CELLS:
        fire = rng.uniform(size=replicates) < 0.5
        if cell.zero:
            fire[:] = True
        supplemented = fire | (rng.uniform(size=replicates) < 0.05)
        errors = rng.normal(size=replicates) * 0.05
        for target in study.TARGETS:
            for arm in study.ARMS:
                name = study.cell_name(cell.name, target, arm)
                covered = rng.uniform(size=replicates) < 0.96
                if arm == "plugin":
                    # The plug-in misses more on firing draws: the control's two clauses hold.
                    covered = np.where(fire, rng.uniform(size=replicates) < 0.80, covered)
                if arm == "supplemented":
                    covered = np.where(fire, True, covered)
                out.append(
                    pd.DataFrame(
                        {
                            "property": study.FAMILY,
                            "cell": name,
                            "role": study.arm_role(cell.name, target, arm),
                            "replicate": np.arange(replicates),
                            "n": cell.n,
                            "requested_replicates": replicates,
                            "failed_replicates": 0,
                            "truth": 0.0,
                            "estimate": errors,
                            "std_error": 0.05 * (2.0 if arm == "supplemented" else 1.0),
                            "covered": covered.astype(int),
                            "rejected": 0,
                            "firing": fire.astype(int),
                            "multi_flag": 0,
                            "supplemented": supplemented.astype(int),
                            "withheld": 0,
                        }
                    )
                )
    return pd.concat(out, ignore_index=True)


def _verdicts(rows: pd.DataFrame) -> pd.DataFrame:
    summary = properties.summarize_properties(rows)
    ctx = property_failures.Context(record=study.STUDY, rows=rows, summary=summary)
    rule = property_failures.rule_verdicts(properties.PROPERTY_RULES[0], summary, ctx)
    merged = summary.merge(rule, on=["property", "cell"], suffixes=("", "_rule"))
    # The declared rule restates the module's verdict on every verdict row.
    assert (merged["passed"] == merged["passed_rule"]).all()
    assert (merged["property_passed"] == merged["property_passed_rule"]).all()
    return summary.set_index("cell")


@pytest.fixture(scope="module")
def passing() -> pd.DataFrame:
    return _rows()


def test_the_rules_pass_rows_that_meet_them(passing: pd.DataFrame) -> None:
    verdicts = _verdicts(passing)
    assert verdicts["passed"].all()
    assert verdicts.loc[properties.CONTROL_CELL, "mcnemar_lower"] > 0.0


def test_a_firing_shortfall_turns_the_firing_floor_red(passing: pd.DataFrame) -> None:
    name = "interior_n200__always__supplemented"
    rows = passing.copy()
    mask = (rows["cell"] == name) & (rows["firing"] == 1)
    rows.loc[mask, "covered"] = (np.arange(mask.sum()) % 10 >= 2).astype(int)
    verdicts = _verdicts(rows)
    assert not verdicts.loc[name, "passed"]
    # The same shortfall on a row whose firing floor is not gated leaves it green.
    free = "interior_n400__never__supplemented"
    assert free not in properties.FIRING_CELLS
    rows = passing.copy()
    mask = (rows["cell"] == free) & (rows["firing"] == 1)
    rows.loc[mask, "covered"] = (np.arange(mask.sum()) % 10 >= 2).astype(int)
    assert _verdicts(rows).loc[free, "passed"]


def test_a_zero_cell_shortfall_turns_its_floor_red(passing: pd.DataFrame) -> None:
    name = "held_zero_n200__always__supplemented"
    rows = passing.copy()
    mask = rows["cell"] == name
    rows.loc[mask, "covered"] = (np.arange(mask.sum()) % 100 >= 9).astype(int)
    verdicts = _verdicts(rows)
    assert verdicts.loc[name, "coverage_ci_lower"] < properties.ZERO_FLOOR
    assert not verdicts.loc[name, "passed"]


def test_a_bias_turns_g2_red(passing: pd.DataFrame) -> None:
    name = "held_n200__never__supplemented"
    rows = passing.copy()
    rows.loc[rows["cell"] == name, "estimate"] += 0.05
    assert not _verdicts(rows).loc[name, "passed"]


def test_a_plugin_that_covers_like_the_supplement_fails_clause_i(passing: pd.DataFrame) -> None:
    rows = passing.copy()
    positive = rows.loc[rows["cell"] == properties.PAIRED_POSITIVE, "covered"].to_numpy()
    rows.loc[rows["cell"] == properties.CONTROL_CELL, "covered"] = positive
    verdicts = _verdicts(rows)
    assert verdicts.loc[properties.CONTROL_CELL, "mcnemar_lower"] <= 0.0
    assert not verdicts.loc[properties.CONTROL_CELL, "passed"]


def test_a_plugin_that_loses_nothing_on_firing_draws_fails_clause_ii(passing: pd.DataFrame) -> None:
    rows = passing.copy()
    mask = rows["cell"] == properties.CONTROL_CELL
    rows.loc[mask, "covered"] = (np.arange(mask.sum()) % 10 >= 2).astype(int)
    verdicts = _verdicts(rows)
    assert verdicts.loc[properties.CONTROL_CELL, "newcombe_upper"] >= 0.0
    assert not verdicts.loc[properties.CONTROL_CELL, "passed"]


def test_a_diagnostic_row_states_no_verdict(passing: pd.DataFrame) -> None:
    name = "interior_n400__always__plugin"
    rows = passing.copy()
    rows.loc[rows["cell"] == name, "covered"] = 0
    assert _verdicts(rows).loc[name, "passed"]


def test_a_lost_draw_is_read_as_a_firing_miss() -> None:
    """The firing floor's worst case adds every lost draw to the firing misses."""
    row = pd.Series({"firing_draws": 600, "firing_coverage": 570 / 600, "failed_replicates": 30})
    ctx = property_failures.Context(record=study.STUDY, rows=pd.DataFrame(), summary=pd.DataFrame())
    candidates = properties._firing_candidates(ctx, row)
    assert len(candidates) == 31
    assert candidates[0][1].passed and not candidates[-1][1].passed


def test_a_quiet_draw_is_neither_firing_nor_supplemented(passing: pd.DataFrame) -> None:
    """Misses on draws supplemented at another node leave the quiet reading alone.

    Mutation control: the same misses on quiet draws lower it and raise ``rare_event_limit``.
    """
    name = "interior_n200__always__supplemented"
    base = _verdicts(passing).loc[name]
    assert not base["rare_event_limit"]
    elsewhere = passing.copy()
    mask = (
        (elsewhere["cell"] == name) & (elsewhere["firing"] == 0) & (elsewhere["supplemented"] == 1)
    )
    assert mask.sum() > 50
    elsewhere.loc[mask, "covered"] = 0
    assert _verdicts(elsewhere).loc[name, "quiet_coverage"] == base["quiet_coverage"]
    quiet = passing.copy()
    mask = (quiet["cell"] == name) & (quiet["firing"] == 0) & (quiet["supplemented"] == 0)
    quiet.loc[mask, "covered"] = (np.arange(mask.sum()) % 5 > 0).astype(int)
    shifted = _verdicts(quiet).loc[name]
    assert shifted["quiet_coverage"] < 0.85
    assert shifted["rare_event_limit"]
    assert shifted["passed"]


def test_the_firing_se_ratio_reads_the_firing_draws(passing: pd.DataFrame) -> None:
    name = "interior_n200__always__supplemented"
    verdicts = _verdicts(passing).loc[name]
    group = passing[passing["cell"] == name]
    fire = group["firing"] == 1
    expected = group.loc[fire, "std_error"].mean() / group.loc[fire, "estimate"].std(ddof=1)
    assert verdicts["firing_se_ratio"] == pytest.approx(expected, rel=1e-12)
    assert verdicts["firing_se_ratio_ci_lower"] < expected < verdicts["firing_se_ratio_ci_upper"]
