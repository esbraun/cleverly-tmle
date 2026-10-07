"""The declared design of the three known-treatment-mechanism studies.

Every number here was fixed before any registered run: the budgets, the seeds, the laws and
their truths, the control's population bias, the cells a run publishes, and the comparator
settings.  None of the three studies declares a power or rejection cell, so the power-sizing
rule has nothing to size.  A replication that raises is never redrawn: the shared harness
refuses a cell that lost one.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tests.studies import canonical_known_mechanism as point
from tests.studies import canonical_known_mechanism_drtmle as dr
from tests.studies import canonical_known_node_mechanisms as nodes
from tests.studies import known_mechanism_drtmle_properties as dr_properties
from tests.studies import known_mechanism_law as law
from tests.studies import known_mechanism_properties as point_properties
from tests.studies import known_node_mechanisms_properties as node_properties
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("known_mechanism_design")

STUDIES = {
    "point": (point, point_properties),
    "drtmle": (dr, dr_properties),
    "nodes": (nodes, node_properties),
}


def test_the_declared_numbers() -> None:
    for module, _ in STUDIES.values():
        record = module.STUDY
        assert record.publication_policy == "reporting"
        assert record.margins == Margins()
        assert (module.PRIMARY_REPLICATES, module.PRIMARY_N) == (1_000, 2_000)
    assert (point.SEED, point.RESAMPLING_SEED) == (20261150, 2026115001)
    assert (dr.SEED, dr.RESAMPLING_SEED) == (20261151, 2026115101)
    assert (nodes.SEED, nodes.RESAMPLING_SEED) == (20261152, 2026115201)
    assert point.G_BOUNDS == (0.1, 0.9)
    assert (point.STUDY.reference, dr.STUDY.reference, nodes.STUDY.reference) == (
        "tmle-r-known-g",
        "drtmle-r-known-g",
        "ltmle-known-gform",
    )
    assert (
        point_properties.ACCURACY_REPLICATES,
        point_properties.BOOTSTRAP_REPLICATES,
        point_properties.BOOTSTRAP_DRAWS,
        point_properties.CALIBRATION_REPLICATES,
        point_properties.RATE_REPLICATES,
    ) == (1_000, 500, 200, 2_000, 1_000)
    assert point_properties.RATE_SIZES == (500, 2_000, 8_000)


def test_every_study_is_registered_and_draws_from_its_own_seeds() -> None:
    slugs = {record.slug for record in registered()}
    mine = {module.STUDY.slug for module, _ in STUDIES.values()}
    assert mine <= slugs
    own = {
        seed
        for module, _ in STUDIES.values()
        for seed in (module.STUDY.seed, module.STUDY.resampling_seed)
    }
    others = {
        seed
        for record in registered()
        if record.slug not in mine
        for seed in (record.seed, record.resampling_seed)
    }
    assert not own & others
    assert len(own) == 6


def test_no_study_declares_a_power_cell() -> None:
    for module, _ in STUDIES.values():
        assert not {"power", "type_i_error"} & set(module.STUDY.property_cells)


def test_the_truths_match_an_independent_monte_carlo() -> None:
    """The quadrature truths against a large draw of the law's own formulas, not of fits."""
    rng = np.random.default_rng(7)
    size = 2_000_000
    w1 = rng.standard_normal(size)
    w2 = rng.binomial(1, 0.5, size).astype(float)
    w3 = rng.uniform(-1.0, 1.0, size)
    truth = law.truth(3)
    for arm in range(3):
        values = law.outcome_probability(arm, w1, w2, w3)
        estimate = float(values.mean())
        error = float(values.std() / math.sqrt(size))
        assert abs(estimate - truth[f"ey[{float(arm)}]"]) < 5 * error
    binary = law.truth(2)
    g1 = law.mechanism(w2)[:, 1]
    blip = law.outcome_probability(1, w1, w2, w3) - law.outcome_probability(0, w1, w2, w3)
    att = float(np.sum(g1 * blip) / np.sum(g1))
    assert abs(att - binary["att"]) < 2e-3
    tilted = law.DELTA * g1 / (law.DELTA * g1 + 1 - g1)
    incremental = tilted * law.outcome_probability(1, w1, w2, w3) + (
        1 - tilted
    ) * law.outcome_probability(0, w1, w2, w3)
    assert abs(float(incremental.mean()) - binary["ey_ipsi"]) < 2e-3


def test_the_control_bias_is_four_margins_or_more() -> None:
    """The plan's necessity rule: the both-wrong control's population bias, recorded before
    any run, is at least four times the accuracy margin at ``n = 2000``."""
    bias = law.control_bias()
    assert bias == pytest.approx(law.CONTROL_BIAS, abs=1e-7)
    margin = Margins().standardized_bias * law.control_sd() / math.sqrt(2_000)
    assert abs(bias) >= 4 * margin
    assert abs(bias) / margin == pytest.approx(15.57, abs=0.01)


def test_the_noise_control_reads_the_exact_wrong_outcome_sd() -> None:
    assert law.wrong_outcome_known_sd() == pytest.approx(point_properties.NOISE_SD, abs=1e-6)


def test_every_declared_mechanism_value_lies_inside_the_bounds() -> None:
    """No bound binds on either side: the runners refuse a sample where one would."""
    frame = law.sample(5_000, 1)
    assert frame[["p0", "p1"]].to_numpy().min() >= 0.25
    three = law.sample(5_000, 2, arms=3)
    assert three[["p0", "p1", "p2"]].to_numpy().min() >= point.G_BOUNDS[0]
    smart = law.smart_sample(5_000, 3)
    product = smart["g1_1"] * smart["r1"] * smart["g2_0"].fillna(1.0)
    assert float(product.min()) > nodes.G_BOUNDS[0]


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_the_declared_cells_are_the_cells_a_run_publishes(name: str) -> None:
    study, properties = STUDIES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=3)
        summary = properties.summarize_properties(rows)
    declared = {(family, cell): truth for family, cell, _, truth in properties.declared_cells()}
    assert {tuple(key) for key in rows.groupby(["property", "cell"]).groups} == set(declared)
    for (family, cell), truth in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        np.testing.assert_allclose(selected["truth"], truth, rtol=1e-12, atol=1e-12)
        assert int(selected["failed_replicates"].max()) == 0
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_the_primary_phase_runs_and_pairs_the_scenarios(name: str) -> None:
    study, _ = STUDIES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, truths, rows = study.draw_and_fit(replicates=2, n=300, n_jobs=1)
    assert set(rows["scenario"]) == set(study.STUDY.scenarios)
    for scenario, estimands in study.STUDY.scenarios.items():
        assert set(rows.loc[rows["scenario"] == scenario, "estimand"]) == set(estimands)
    # Every scenario reads its owner's realized sample.
    owners = study.STUDY.scenario_seed_owners
    for scenario, owner in owners.items():
        mine = samples.loc[samples["scenario"] == scenario].drop(columns=["scenario"])
        theirs = samples.loc[samples["scenario"] == owner].drop(columns=["scenario"])
        shared = [
            column for column in ("W1", "W2", "W3", "A", "Y", "L0", "A1", "L1") if column in mine
        ]
        np.testing.assert_array_equal(
            mine[shared].to_numpy(dtype=float), theirs[shared].to_numpy(dtype=float)
        )
    assert not truths.empty
    # The witness reads this study's own rows, and finds the subject's targeting step.
    displacement = study.targeting_displacement(rows)
    assert any(key.startswith(study.IMPLEMENTATION) for key in displacement)
    assert all(np.isfinite(value) for value in displacement.values())


def test_the_runners_and_their_regenerators_exist() -> None:
    for directory in ("known_mechanism", "known_mechanism_drtmle", "known_node_mechanisms"):
        here = ROOT / "tests" / "canonical" / directory
        assert (here / "run_study.R").exists()
        assert (here / "regenerate.py").exists()


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_the_targeting_witness_floor_is_declared(name: str) -> None:
    """The nonzero-targeting witness reads the committed rows once each study has run."""
    study, _ = STUDIES[name]
    assert study.TARGETING_WITNESS_FLOOR > 0
    path = study.STUDY.artifact("replicates.csv.gz")
    if not path.exists():
        pytest.skip("the registered run has not been committed yet")
    import pandas as pd

    rows = pd.read_csv(path, float_precision="round_trip")
    displacement = study.targeting_displacement(rows)
    subjects = {key.split(" ")[0] for key in displacement}
    assert subjects == {study.IMPLEMENTATION, study.REFERENCE}
    assert min(displacement.values()) > study.TARGETING_WITNESS_FLOOR
