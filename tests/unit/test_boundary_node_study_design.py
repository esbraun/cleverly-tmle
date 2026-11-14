"""The design of ``boundary-node-ltmle``: its law, its declared grid, and its pilot.

The law's figures are exact (:mod:`tests.studies.boundary_node_laws`), and each is checked here
against a Monte Carlo draw of the same law, so a wrong formula cannot pass by agreeing with
itself. The pilot (:mod:`tests.studies.boundary_node_pilot`) is read from its committed rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies import boundary_node_laws as laws
from tests.studies import boundary_node_pilot as pilot

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
    frame = pd.read_csv(PILOT)
    frame["lost"] = frame["lost"].fillna("")
    return frame


def test_the_pilot_holds_every_cell_and_draw(rows: pd.DataFrame) -> None:
    counts = rows.groupby(["cell", "target"]).size()
    assert set(counts.index.get_level_values(0)) == {cell[0] for cell in pilot.CELLS}
    assert (counts == pilot.PILOT_REPLICATES).all()


def test_the_supplemented_interval_contains_the_plugin_interval(rows: pd.DataFrame) -> None:
    kept = rows[rows["supplemented_here"].astype(bool) & np.isfinite(rows["supplemented_low"])]
    assert len(kept) > 1_000
    assert (kept["supplemented_low"] <= kept["plugin_low"] + 1e-12).all()
    assert (kept["supplemented_high"] >= kept["plugin_high"] - 1e-12).all()


def test_the_paired_control_has_power_at_the_declared_size(rows: pd.DataFrame) -> None:
    """Both clauses of ``paired_attributable`` at n = 200, R = 4,000, at the pilot rates."""
    cell = rows[(rows["cell"] == "interior_n200") & (rows["target"] == "always")].copy()
    cell["firing"] = cell["firing"].astype(bool)
    for column in ("supplemented_covered", "plugin_covered"):
        cell[column] = cell[column].astype(bool)
    power = pilot.control_power(cell)
    assert power["clause_i"] >= 0.90, power
    assert power["clause_ii"] >= 0.90, power


#: Cells whose plug-in G1 pass probability at R = 4,000 is below 0.90 at the pilot rates,
#: recorded before the declaration as predicted reds (plan section 5). ``interior_n400``
#: ``always``: the non-firing draws cover 0.893 in both the supplemented and the plug-in arm,
#: the Wald interval of a risk whose last node holds about n/16 followers.
PREDICTED_RED = {("interior_n400", "always")}


def _g1_threshold(replicates: int = 4_000, floor: float = 0.90) -> int:
    hits = np.arange(int(floor * replicates), replicates + 1)
    lower = stats.beta.ppf(0.005, hits, replicates - hits + 1)
    return int(hits[np.argmax(lower >= floor)])


def test_the_predicted_reds_are_the_cells_below_the_pass_probability(rows: pd.DataFrame) -> None:
    threshold = _g1_threshold()
    below = set()
    for (cell, target), group in rows.groupby(["cell", "target"]):
        coverage = group["supplemented_covered"].astype(bool).mean()
        if stats.binom.sf(threshold - 1, 4_000, coverage) < 0.90:
            below.add((cell, target))
    assert below == PREDICTED_RED
