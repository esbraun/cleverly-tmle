"""Design checks of ``few-cluster-floor-stacked`` (X54 part (j), plan rev 2.4, 4 and 5.1).

The declaration against its published cells, seeds and ladders; the pilot reading, recomputed
from the committed pilot tables once they exist; the route, the fixed-bounds reading and the
predicted reds on constructed summaries; and two witnesses with their mutations: the declared
``"auto"`` bounds of an estimated mechanism on a near-separated draw, and the known mechanism's
exact bounds on a draw where ``"auto"`` would move a known propensity.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError
from tests.canonical import property_differential as differential
from tests.studies import few_cluster_arm_means as study_k
from tests.studies import few_cluster_arm_means_ltmle as study_k2
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_pilot as pilot
from tests.studies import few_cluster_floor_stacked as study
from tests.studies import few_cluster_floor_stacked_properties as properties
from tests.studies import few_cluster_floor_tmle as source
from tests.studies.evidence import property_failures
from tests.studies.evidence.registry import registered

# ------------------------------------------------------------------------------ declaration


def test_published_cells_equal_the_declaration():
    declared = properties.declared_cells()
    by_family = {
        family: tuple(cell.cell for cell in declared if cell.property == family)
        for family in (floor.FAMILY, floor.CANDIDATE_FAMILY)
    }
    assert by_family[floor.FAMILY] == study.PUBLISHED_CELLS
    assert by_family[floor.CANDIDATE_FAMILY] == study.CANDIDATE_CELLS
    assert study.STUDY.property_cells == {
        floor.FAMILY: study.PUBLISHED_CELLS,
        floor.CANDIDATE_FAMILY: study.CANDIDATE_CELLS,
        floor.POOLED_FAMILY: study.POOLED_CELLS,
    }
    groups = properties.PROPERTY_GROUPS[floor.FAMILY]
    assert tuple(cell for group in groups for cell in group) == study.PUBLISHED_CELLS


def test_declared_record():
    record = study.STUDY
    assert record.slug == floor.STACKED_REDECLARATION == "few-cluster-floor-stacked"
    assert record.publication_policy == "reporting" and record.reference is None
    assert record.failure_bound == {"cleverly": floor.PRIMARY_FAILURE_BOUND}
    assert properties.FAILURE_BOUND == property_failures.MAXIMUM_BOUND
    assert (record.seed, record.resampling_seed) == (20265431, 20265432)
    assert record.replicates == study.PRIMARY_REPLICATES == 1_000
    assert record.n == 10 * study.PRIMARY_CLUSTERS
    # No study, registered or declared with this one, shares a seed.
    seeds = [study_k.STUDY, study_k2.STUDY, *registered()]
    for other in seeds:
        if other.slug != record.slug:
            assert {other.seed, other.resampling_seed}.isdisjoint(
                {record.seed, record.resampling_seed}
            )


def test_the_study_draws_its_own_streams():
    """A new slug, so no draw is shared with ``few-cluster-floor-tmle`` or with its own pilot."""
    keys = [(law, j, r) for law in study.SIZE_LAWS for j in (4, 10) for r in (0, 1)]
    mine = {study.draw_seed(*key) for key in keys}
    pilots = {study.draw_seed(*key, "pilot") for key in keys}
    theirs = {source.draw_seed(*key) for key in keys}
    assert len(mine) == len(pilots) == len(keys)
    assert mine.isdisjoint(pilots) and mine.isdisjoint(theirs)


def test_the_ladder_arms_and_mechanism_bounds():
    assert study.LADDER == (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39)
    assert study.ANCHOR == 10 and study.FLOOR_FAMILY == "tmle_stacked"
    assert study.MECHANISM_BOUNDS == {"known": source.G_BOUNDS, "estimated": "auto"}
    assert {fit: study.bounds(fit) for fit in study.SOURCES} == {
        "stacked_known": source.G_BOUNDS,
        "stacked_estimated": "auto",
        "stacked_d5_known": source.G_BOUNDS,
        "stacked_d5_estimated": "auto",
        "stacked_estimated_fixed_bounds": (0.01, 0.99),
        "stacked_d5_estimated_fixed_bounds": (0.01, 0.99),
    }
    for fit in study.SOURCES:
        roles = dict(study.arms(fit, 10))
        assert roles[floor.REFERENCE_ARM] == floor.POSITIVE
        # The fixed-bounds arm gates its interval only.
        fixed = fit in study.FIXED_BOUNDS_FITS
        assert roles["iid_t_control"] == (floor.DIAGNOSTIC if fixed else floor.CONTROL)
        assert study.jackknife_counts(fit) == (() if fixed else (10, 15, 20, 30, 39))
        d5 = study.SOURCES[fit] in source.D5_FITS
        assert study.laws_of(fit) == (("unequal_informative",) if d5 else study.SIZE_LAWS)
    assert study.CONFIGURATIONS == (
        "stacked_known",
        "stacked_estimated",
        "stacked_d5_known",
        "stacked_d5_estimated",
    )
    assert properties.pooled_starts() == {
        (fit, floor.JACKKNIFE_ARM): 10 for fit in study.CONFIGURATIONS
    }


def test_the_differential_reads_a_declared_subset_of_whole_groups():
    assert properties.__name__ in differential.SYNTHETIC_MODULES
    assert properties.DIFFERENTIAL_BUDGET == 200
    cells = properties.differential_cells()
    for family in (floor.FAMILY, floor.CANDIDATE_FAMILY):
        for group in properties.PROPERTY_GROUPS[family]:
            assert len({f"{family}/{cell}" in cells for cell in group}) == 1, group
    assert {name.split("/")[0] for name in cells} == {floor.FAMILY, floor.CANDIDATE_FAMILY}
    bound = differential.read_bindings()
    assert (bound["subject"] == properties.__name__).any()


# ------------------------------------------------------------------------------------ pilot


def _committed(name: str):
    path = study.STUDY.artifacts / name
    return path if path.exists() else None


def test_the_declaration_reads_the_committed_pilot():
    """Once the pilot is committed, R, the predicted reds and the shortfalls are those the
    sizing code reads off ``pilot-summary.csv``, and ``pilot-sizing.csv`` is that reading.
    Before it, the declaration holds R = 4,000 and no prediction."""
    path = _committed("pilot-summary.csv")
    if path is None:
        assert study.PROPERTY_REPLICATES == {study.FLOOR_FAMILY: 4_000}
        assert not study.PILOT_PREDICTED_REDS and not study.PILOT_SHORTFALLS
        assert not study.JACKKNIFE_SHORTFALLS
        return
    summary = pd.read_csv(path, keep_default_na=False, na_values=[""])
    sized = properties.sizing(summary)[study.FLOOR_FAMILY]
    assert {study.FLOOR_FAMILY: sized.replicates} == study.PROPERTY_REPLICATES
    short = {study.FLOOR_FAMILY} if sized.plugin < floor.FAMILY_PASS_TARGET else set()
    assert short == study.PILOT_SHORTFALLS
    recomputed = properties.pilot_sizing_table(summary)
    table = pd.read_csv(study.STUDY.artifacts / "pilot-sizing.csv", keep_default_na=False)
    for frame in (recomputed, table):
        reds = frame.loc[frame["kind"] == "predicted red"]
        assert dict(zip(reds["cell"], reds["reading"], strict=True)) == study.PILOT_PREDICTED_REDS
    candidates = pd.read_csv(study.STUDY.artifacts / "pilot-candidates.csv")
    assert set(candidates.loc[candidates["shortfall"].astype(bool), "family"]) == (
        study.JACKKNIFE_SHORTFALLS
    )
    timing = pd.read_csv(study.STUDY.artifacts / "pilot-timing.csv")
    projected = float(
        timing.loc[timing["fit"] == "total at the sized R", "projected_cpu_hours"].iloc[0]
    )
    assert projected <= pilot.BUDGET_TOLERANCE * study.DECLARED_CPU_HOURS


# ------------------------------------------------------------------ constructed summaries


def _row(cell: str, role: str, *, red: bool = False, coverage: float = 0.95) -> dict:
    row = {
        "property": floor.FAMILY,
        "cell": cell,
        "role": role,
        "passed": not red,
        "requested_draws": 400.0,
        "replicates": 400.0,
        "withheld_draws": 0.0,
        "all_draw_coverage": coverage,
        "standardized_bias": 0.0,
        "shown_ci_lower": 0.9,
        "shown_ci_upper": 0.97,
        "se_ratio": 1.0,
        "shown_se_ratio": 1.0,
        "se_ratio_ci_lower": 0.95,
        "se_ratio_ci_upper": 1.05,
    }
    if role == floor.CONTROL:
        row.update(
            paired_se_ratio=0.5,
            paired_se_ratio_ci_lower=0.45,
            paired_se_ratio_ci_upper=0.9 if red else 0.55,
            paired_positive_se_ratio_ci_upper=1.05,
        )
    return row


def _summary(red: set[str] = frozenset(), low: set[str] = frozenset()) -> pd.DataFrame:
    """Every published reference row, passing unless named in ``red``; a positive in ``low``
    reads pilot coverage 0.88."""
    rows = []
    for fit, law, clusters in study.grid():
        for arm, role in study.arms(fit, clusters):
            name = study.cell_name(fit, law, clusters, arm)
            if role == floor.DIAGNOSTIC and arm != "iid_t_control":
                continue
            rows.append(_row(name, role, red=name in red, coverage=0.88 if name in low else 0.95))
    return pd.DataFrame(rows)


def test_the_route_withdraws_support_for_an_uncompared_red():
    """S-j is the redeclaration, so a red at or above 10 with no committed counterpart reads
    ``support withdrawn`` with the floor at 10, and a red below 10 lowers it to the next count."""
    assert properties.route(_summary()).route == floor.LOWERED
    assert properties.route(_summary()).floor == 4
    red = study.cell_name("stacked_d5_estimated", "unequal_informative", 15, floor.REFERENCE_ARM)
    found = properties.route(_summary({red}))
    assert found.route == f"{floor.SUPPORT_WITHDRAWN}: 15" and found.floor == 10
    assert found.owners == {f"{floor.FAMILY}/{red}": None}
    low = study.cell_name("stacked_estimated", "equal10", 6, floor.REFERENCE_ARM)
    found = properties.route(_summary({low}))
    assert found.floor == 7 and found.route == floor.LOWERED
    assert found.owners[f"{floor.FAMILY}/{low}"] == (floor.RED_OWNER, floor.FINITE_SAMPLE_READING)


def test_the_fixed_bounds_arm_scopes_a_lowered_floor():
    """Plan 1.2 item 4: a fixed-bounds positive red at or above a lowered floor scopes it to a
    known mechanism or ``"auto"``; one below it is held. The arm never moves the floor."""
    above = study.cell_name("stacked_estimated_fixed_bounds", "equal10", 8, floor.REFERENCE_ARM)
    below = study.cell_name(
        "stacked_d5_estimated_fixed_bounds", "unequal_informative", 4, floor.REFERENCE_ARM
    )
    summary = _summary({above, below})
    assert properties.route(summary).floor == 4
    red, owners = properties.fixed_bounds_reading(summary, 5)
    assert red == (8,)
    assert owners == {
        f"{floor.FAMILY}/{above}": None,
        f"{floor.FAMILY}/{below}": (floor.RED_OWNER, floor.FINITE_SAMPLE_READING),
    }
    # Mutation: a reading that counts every fixed-bounds red would also take the one below.
    assert properties.fixed_bounds_reading(summary, 0)[0] == (4, 8)


def test_the_predicted_reds_read_their_position():
    """Below 10 a predicted red takes the below-floor reading; at or above, chance only where a
    committed counterpart exists (the two-covariate known configuration at 10, 20 and 30), else
    ``support withdrawn``. A fixed-bounds positive is sized with a control that cannot fail."""
    cells = {
        study.cell_name("stacked_known", "equal10", 5, floor.REFERENCE_ARM),
        study.cell_name("stacked_known", "equal10", 10, floor.REFERENCE_ARM),
        study.cell_name("stacked_estimated", "equal10", 10, floor.REFERENCE_ARM),
        study.cell_name("stacked_estimated_fixed_bounds", "equal10", 15, floor.REFERENCE_ARM),
    }
    reds = properties.predicted_reds(_summary(low=cells), 4_000)
    names = sorted(cells)
    assert reds == {
        f"{floor.FAMILY}/{names[0]}": floor.SUPPORT_WITHDRAWN,
        f"{floor.FAMILY}/{names[1]}": floor.SUPPORT_WITHDRAWN,
        f"{floor.FAMILY}/{names[2]}": floor.CHANCE_READING,
        f"{floor.FAMILY}/{names[3]}": floor.FINITE_SAMPLE_READING,
    }
    assert names[2].startswith("stacked_known__equal10__j10")
    assert properties.predicted_reds(_summary(), 4_000) == {}
    assert properties.positive_inputs(_summary().iloc[0])["ratio_sd"] == 0.0


# ----------------------------------------------------------------------------- witnesses


#: The near-separated draw of plan 4: replicate 4650 of ``few-cluster-floor-tmle``'s property
#: stream at 10 clusters of ``unequal_informative``, with its d5 noise.
WITNESS = ("unequal_informative", 10, 4650)
WITNESS_CELL = "tmle_stacked_d5_estimated__unequal_informative__j10__t_reference"


def _witness_frame() -> pd.DataFrame:
    law, clusters, replicate = WITNESS
    seed = source.draw_seed(law, clusters, replicate)
    return source.with_noise(source.draw(law, clusters, seed), seed)


def _witness_spread() -> float:
    """The committed empirical SD of the estimated d5 configuration's estimate in that cell."""
    table = pd.read_csv(source.STUDY.artifacts / "properties.csv")
    row = table.loc[(table["property"] == floor.FAMILY) & (table["cell"] == WITNESS_CELL)]
    return float(row["empirical_se"].iloc[0])


def _se_over_sd(**overrides) -> float:
    estimate = study.fit_result("stacked_d5_estimated", _witness_frame(), 10, **overrides)
    return float(estimate[study.TARGET].std_error) / _witness_spread()


def test_the_declared_bounds_of_an_estimated_mechanism_tame_the_witness(monkeypatch):
    """At the exact bounds the near-separated draw reads an SE above 10 SD; at the declared
    ``"auto"`` below 3 SD. Mutation: declaring the exact bounds for an estimated mechanism
    brings the declared fit back above 10 SD."""
    spread = _witness_spread()
    assert 0.15 < spread < 0.25
    assert _se_over_sd(g_bounds=source.G_BOUNDS) > 10.0
    assert _se_over_sd() < 3.0
    monkeypatch.setitem(study.MECHANISM_BOUNDS, "estimated", source.G_BOUNDS)
    assert _se_over_sd() > 10.0


#: A draw of ``equal10`` at 4 clusters on a ``"unit"`` stream that no study reads, where
#: ``"auto"`` (about (0.21, 0.79) at 40 rows) would move a known propensity (0.18, 0.82).
GUARD = ("equal10", 4, 0)


def _guard_fit(**overrides):
    law, clusters, replicate = GUARD
    frame = study.draw(law, clusters, study.draw_seed(law, clusters, replicate, "unit"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.fit_result("stacked_known", frame, clusters, **overrides)


def test_a_known_mechanism_keeps_bounds_that_move_no_known_value(monkeypatch):
    """The package refuses to truncate a declared mechanism (``moved_known_values_refusal``).
    On a draw where ``"auto"`` binds, the known configuration's declared bounds fit and
    ``"auto"`` is refused; declaring ``"auto"`` for a known mechanism (the mutation) refuses
    the declared fit."""
    estimate = _guard_fit()[study.TARGET]
    assert math.isfinite(float(estimate.psi))
    with pytest.raises(CapabilityError, match="outside the truncation bounds"):
        _guard_fit(g_bounds="auto")
    monkeypatch.setitem(study.MECHANISM_BOUNDS, "known", "auto")
    with pytest.raises(CapabilityError, match="outside the truncation bounds"):
        _guard_fit()
