"""Design checks of ``few-cluster-arm-means`` (X54 part (k), plan rev 2.4, 4 and 5.2).

The declaration against its published cells, seeds, ladders and gates; the pilot reading,
recomputed from the committed pilot tables once they exist; the primary rule; the floor, the
transfer owners and the sizing on constructed summaries; the bounds witness and the
known-mechanism guard with their mutations; and the law pin: the default latent coefficient
reproduces ``few-cluster-floor-tmle``'s draws, the arm-mean truths match a Monte Carlo of
:math:`10^6` rows, and each arm of an arm-mean law carries the cluster latent, with a mutation
that drops it from the untreated arm.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS, FEW_CLUSTER_THRESHOLD
from cleverly.exceptions import CapabilityError
from tests.canonical import property_differential as differential
from tests.studies import clustered_unequal_laws as laws
from tests.studies import few_cluster_arm_means as study
from tests.studies import few_cluster_arm_means_ltmle as study_k2
from tests.studies import few_cluster_arm_means_properties as properties
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_pilot as pilot
from tests.studies import few_cluster_floor_stacked as study_j
from tests.studies import few_cluster_floor_tmle as source
from tests.studies.evidence import property_failures
from tests.studies.evidence.registry import registered

# ------------------------------------------------------------------------------ declaration


def test_published_cells_equal_the_declaration():
    declared = properties.declared_cells()
    assert {cell.property for cell in declared} == {floor.FAMILY}
    assert tuple(cell.cell for cell in declared) == study.PUBLISHED_CELLS
    assert study.STUDY.property_cells == {floor.FAMILY: study.PUBLISHED_CELLS}
    groups = properties.PROPERTY_GROUPS[floor.FAMILY]
    assert tuple(cell for group in groups for cell in group) == study.PUBLISHED_CELLS
    for cell in declared:
        fit, estimand, law, clusters, arm = study.parse(cell.cell)
        assert study.cell_name(fit, estimand, law, clusters, arm) == cell.cell
        assert cell.estimand == estimand and cell.n == 10 * clusters
        assert cell.dgp.truth()[estimand] == study.law_truth(law)[estimand]


def test_declared_record():
    record = study.STUDY
    assert record.slug == "few-cluster-arm-means"
    assert record.publication_policy == "reporting" and record.reference is None
    assert record.failure_bound == {"cleverly": floor.PRIMARY_FAILURE_BOUND}
    assert properties.FAILURE_BOUND == property_failures.MAXIMUM_BOUND
    assert (record.seed, record.resampling_seed) == (20265441, 20265442)
    assert study.RED_OWNER == study_k2.RED_OWNER == "X54-estimand-floor"
    for other in (study_j.STUDY, study_k2.STUDY, *registered()):
        assert {other.seed, other.resampling_seed}.isdisjoint({record.seed, record.resampling_seed})


def test_the_study_draws_its_own_streams():
    keys = [(law, j, r) for law in study.ATE_LAWS for j in (6, 10) for r in (0, 1)]
    mine = {study.draw_seed(*key) for key in keys}
    pilots = {study.draw_seed(*key, "pilot") for key in keys}
    others = {source.draw_seed(*key) for key in keys} | {study_j.draw_seed(*key) for key in keys}
    assert len(mine) == len(pilots) == len(keys)
    assert mine.isdisjoint(pilots) and mine.isdisjoint(others)


def test_the_ladders_laws_and_gates():
    assert {
        "tmle_in_sample": (6, 7, 8, 9, 10, 15, 20, 30, 39),
        "tmle_stacked": (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39),
        "tmle_fold_evaluated": (8, 9, 10, 15, 20, 30, 39),
        "drtmle_known": (8, 9, 10, 15, 20, 30, 39),
        "ltmle_in_sample": (*range(6, 21), 30, 39),
    } == study.LADDERS
    assert {**dict.fromkeys(study.LADDERS, 10), "ltmle_in_sample": 20} == study.ANCHORS
    assert study.FITS == source.FITS
    assert study.MECHANISM_BOUNDS == {"known": source.G_BOUNDS, "estimated": "auto"}
    for fit in study.FITS:
        family = study.FITS[fit]
        laws_of = study.laws_of(fit)
        if family == "tmle_fold_evaluated":
            assert laws_of == study.EQUAL_LAWS
        elif fit in source.D5_FITS:
            assert laws_of == study.UNEQUAL_LAWS
        else:
            assert laws_of == study.LAWS
        for law in laws_of:
            for estimand in study.estimands_of(fit):
                roles = dict(study.arms(fit, law, 10, estimand))
                contrast = estimand in study.CONTRASTS
                # Arm means are gated on all four laws; a contrast only as the transfer check.
                gated = not contrast or (law in study.ATE_LAWS and fit.endswith("_estimated"))
                assert study.gated(fit, law, estimand) == gated
                assert roles[floor.REFERENCE_ARM] == (floor.POSITIVE if gated else floor.DIAGNOSTIC)
                assert roles["iid_t_control"] == (floor.CONTROL if gated else floor.DIAGNOSTIC)
    assert {study.TRUTH_KEYS[name] for name in study.LTMLE_ESTIMANDS} == {"ey0", "ey1", "ate"}


def test_drtmle_keeps_its_declared_bounds(monkeypatch):
    """The estimated point configurations pass ``"auto"``, the known ones the exact bounds, and
    DR-TMLE nothing, so it keeps ``DRTMLE_G_BOUNDS`` on its reduced regression."""
    seen = {}

    class _Estimate:
        supplies_inference = False

    def fake(fit, frame, clusters, **overrides):
        seen[fit] = overrides
        return dict.fromkeys(study.POINT_ESTIMANDS, _Estimate())

    monkeypatch.setattr(source, "point_result", fake)
    for fit in ("tmle_in_sample_known", "tmle_stacked_estimated", "drtmle_known"):
        study.fit_estimates(fit, pd.DataFrame(), 10)
    assert seen["tmle_in_sample_known"]["g_bounds"] == source.G_BOUNDS
    assert seen["tmle_stacked_estimated"]["g_bounds"] == "auto"
    assert "g_bounds" not in seen["drtmle_known"]
    assert all(value["estimands"] == study.POINT_ESTIMANDS for value in seen.values())


def test_the_differential_reads_a_declared_subset_of_whole_groups():
    assert properties.__name__ in differential.SYNTHETIC_MODULES
    assert properties.DIFFERENTIAL_BUDGET == 200
    cells = properties.differential_cells()
    for group in properties.PROPERTY_GROUPS[floor.FAMILY]:
        assert len({f"{floor.FAMILY}/{cell}" in cells for cell in group}) == 1, group
    bound = differential.read_bindings()
    assert (bound["subject"] == properties.__name__).any()


# ------------------------------------------------------------------------------------ pilot


def _committed(name: str):
    path = study.STUDY.artifacts / name
    return path if path.exists() else None


def test_the_declaration_reads_the_committed_pilot():
    """Once the pilot is committed: R per family, the predicted floors and reds, the
    shortfalls, the hard cell's prediction, the primary and the budget are those the sizing
    code reads off ``pilot-summary.csv``. Before it, R = 4,000, no prediction, and the primary
    is the first declared candidate."""
    path = _committed("pilot-summary.csv")
    if path is None:
        assert set(study.PROPERTY_REPLICATES.values()) == {4_000}
        assert not study.PILOT_PREDICTED_REDS and not study.PILOT_SHORTFALLS
        assert not study.PILOT_PREDICTED_FLOORS and study.PILOT_HARD_CELL is None
        assert study.PRIMARY_CANDIDATES[0] == (study.PRIMARY_ESTIMAND, study.PRIMARY_CLUSTERS)
        return
    summary = pd.read_csv(path, keep_default_na=False, na_values=[""])
    sized = {family: properties.family_sizing(summary, family) for family in study.LADDERS}
    assert {f: int(r["replicates"]) for f, r in sized.items()} == study.PROPERTY_REPLICATES
    assert {f: r["predicted_floor"] for f, r in sized.items()} == study.PILOT_PREDICTED_FLOORS
    assert {f for f, r in sized.items() if not r["target_met"]} == study.PILOT_SHORTFALLS
    reds = {cell: reading for result in sized.values() for cell, reading in result["reds"].items()}
    assert reds == study.PILOT_PREDICTED_REDS
    table = pd.read_csv(study.STUDY.artifacts / "pilot-sizing.csv", keep_default_na=False)
    listed = table.loc[table["kind"] == "predicted red"]
    assert dict(zip(listed["cell"], listed["reading"], strict=True)) == reds
    hard = properties.hard_cell_prediction(summary, sized[study.HARD_CELL[0]]["replicates"])
    assert pytest.approx(hard) == study.PILOT_HARD_CELL
    assert properties.primary_pick(summary) == (study.PRIMARY_ESTIMAND, study.PRIMARY_CLUSTERS)
    timing = pd.read_csv(study.STUDY.artifacts / "pilot-timing.csv")
    total = timing.loc[timing["fit"] == "total at the sized R", "projected_cpu_hours"]
    assert float(total.iloc[0]) <= pilot.BUDGET_TOLERANCE * study.DECLARED_CPU_HOURS


def test_the_primary_rule():
    """The first candidate whose G1 pass probability at 1,000 draws reaches 0.90, else the ATE."""
    good, poor = 0.955, 0.915
    assert floor.g1_pass_probability(good, 1_000) >= 0.90 > floor.g1_pass_probability(poor, 1_000)
    cases = {
        (good, good, good): ("ey1", 10),
        (poor, good, good): ("ey1", 20),
        (poor, poor, good): ("ey1", 39),
        (poor, poor, poor): ("ate", 10),
    }
    for values, chosen in cases.items():
        coverage = dict(zip(study.PRIMARY_CANDIDATES, values, strict=True))
        assert study.primary_choice(coverage) == chosen


def test_the_known_hard_cell_is_hard_at_the_primary_size():
    """Review F3: S-a's primary read 0.914 there, which G1 passes with probability 0.117 at
    1,000 draws and 0.666 at 4,000."""
    committed = properties.source_hard_cell()
    assert (committed.hits, committed.draws) == (914, 1000)
    assert floor.g1_pass_probability(0.914, 1_000) == pytest.approx(0.117, abs=0.002)
    assert floor.g1_pass_probability(0.914, 4_000) == pytest.approx(0.666, abs=0.002)


# ------------------------------------------------------------------ constructed summaries


def _row(cell: str, role: str, *, coverage: float, clause_one: bool, red: bool) -> dict:
    row = {
        "property": floor.FAMILY,
        "cell": cell,
        "role": role,
        "passed": not red,
        "requested_draws": 4000.0,
        "replicates": 4000.0,
        "withheld_draws": 0.0,
        "all_draw_coverage": coverage,
        "standardized_bias": 0.0,
        "shown_ci_lower": coverage - 0.01,
        "shown_ci_upper": coverage + 0.01,
        "se_ratio": 1.0,
        "shown_se_ratio": 1.0,
        "se_ratio_ci_lower": 0.97,
        "se_ratio_ci_upper": 1.03,
    }
    if role == floor.CONTROL:
        high = 0.55 if clause_one else 0.9
        row.update(
            paired_se_ratio=high - 0.05,
            paired_se_ratio_ci_lower=high - 0.1,
            paired_se_ratio_ci_upper=high,
            paired_positive_se_ratio_ci_upper=1.03,
        )
    return row


def _summary(
    red: frozenset[str] = frozenset(),
    weak: frozenset[str] = frozenset(),
    coverage: dict[str, float] | None = None,
) -> pd.DataFrame:
    """The gated rows of every cell, passing unless the cell is ``red``; a control in ``weak``
    fails clause 1 alone (its row's ``passed`` is the registered rule). The hard cell reads
    0.92 unless ``coverage`` says otherwise."""
    coverage = {properties.hard_cell_name(): 0.92, **(coverage or {})}
    rows = []
    for fit, law, clusters in study.grid():
        for estimand in study.estimands_of(fit):
            for arm, role in study.arms(fit, law, clusters, estimand):
                if role == floor.DIAGNOSTIC:
                    continue
                name = study.cell_name(fit, estimand, law, clusters, arm)
                rows.append(
                    _row(
                        name,
                        role,
                        coverage=coverage.get(name, 0.95),
                        clause_one=name not in weak,
                        red=name in red or name in weak,
                    )
                )
    return pd.DataFrame(rows)


def _name(fit, estimand, law, clusters, arm=floor.REFERENCE_ARM) -> str:
    return study.cell_name(fit, estimand, law, clusters, arm)


def test_the_floor_starts_at_the_contrast_floor_and_rises_past_the_default():
    summary = _summary()
    found = properties.routes(summary)
    for family, route in found.items():
        assert route.floor == CLUSTER_INTERVAL_FLOORS[floor.PACKAGE_FAMILIES[family]], family
        assert route.owners == {}
    red = _name("tmle_in_sample_estimated", "ey0", "unequal_arms", 10)
    route = properties.route(_summary(frozenset({red})), "tmle_in_sample")
    assert (route.floor, route.route) == (15, "raised: 15")
    assert route.owners == {f"{floor.FAMILY}/{red}": (study.RED_OWNER, floor.FINITE_SAMPLE_READING)}
    top = _name("ltmle_in_sample", "ey_regimen[always]", "equal10", 39)
    route = properties.route(_summary(frozenset({top})), "ltmle_in_sample")
    assert route.floor == FEW_CLUSTER_THRESHOLD


def test_clause_one_on_an_ate_law_is_unpowered():
    """An arm-mean control on an ATE law failing only clause 1 passes its cell and reads
    ``control unpowered by law``; on an arm-mean law the same control turns its count red."""
    ate = _name("drtmle_known", "ey0", "equal10", 20, "iid_t_control")
    route = properties.route(_summary(weak=frozenset({ate})), "drtmle_known")
    assert route.floor == 8
    assert route.owners == {f"{floor.FAMILY}/{ate}": (study.RED_OWNER, floor.UNPOWERED_READING)}
    arms = _name("drtmle_known", "ey0", "equal10_arms", 20, "iid_t_control")
    route = properties.route(_summary(weak=frozenset({arms})), "drtmle_known")
    assert route.floor == 30


def test_the_hard_cell_condition_feeds_the_contrast_floor():
    """Plan 5.5: in-sample ``LTMLE`` reads its contrast floor 6 only when the hard cell passes
    and agrees with S-a's committed primary (0.914 over 1,000 draws); otherwise 15."""
    hard = properties.hard_cell_name()
    assert properties.contrast_floor("ltmle_in_sample", _summary()) == 6
    assert properties.route(_summary(), "ltmle_in_sample").floor == 6
    disagrees = _summary(coverage={hard: 0.96})
    assert properties.contrast_floor("ltmle_in_sample", disagrees) == 15
    assert properties.contrast_floor("ltmle_in_sample", _summary(frozenset({hard}))) == 15
    assert properties.contrast_floor("ltmle_in_sample") == 15
    assert properties.contrast_floor("tmle_in_sample", disagrees) == 6


def test_the_transfer_check_owners():
    """A red transfer-check contrast below its family's shipped floor is held; at or above it
    is a defect suspect; a contrast that is only reported is never read."""
    shipped = CLUSTER_INTERVAL_FLOORS["tmle_stacked"]
    below = _name("tmle_stacked_estimated", "ate", "equal10", 4)
    above = _name("tmle_stacked_estimated", "ate", "equal10", shipped)
    reported = _name("tmle_stacked_known", "ate", "equal10", 4)
    summary = _summary(frozenset({below, above}))
    summary.loc[summary["cell"] == reported, "passed"] = False
    assert properties.transfer_owners(summary) == {
        f"{floor.FAMILY}/{below}": (study.RED_OWNER, floor.FINITE_SAMPLE_READING),
        f"{floor.FAMILY}/{above}": None,
    }


def test_the_expected_and_predicted_floors_are_not_capped_at_the_anchor():
    ladder = study.LADDERS["tmle_in_sample"]
    passing = {count: [0.95] for count in ladder}
    assert properties.expected_floor(passing, ladder=ladder, contrast=6) == 6
    low = {**passing, 10: [0.91]}
    assert properties.expected_floor(low, ladder=ladder, contrast=6) == 15
    assert properties.expected_floor({**passing, 39: [0.9]}, ladder=ladder, contrast=6) == 40
    assert properties.predicted_floor({10}, ladder=ladder, contrast=6) == 15
    assert properties.predicted_floor({7, 20}, ladder=ladder, contrast=6) == 30
    assert properties.predicted_floor(set(), ladder=ladder, contrast=8) == 8
    assert properties.predicted_floor({39}, ladder=ladder, contrast=6) == 40


def test_the_pilot_sizing_counts_clause_one_on_the_arm_mean_laws_only():
    """A weak ATE-law control is a predicted ``control unpowered by law`` that moves no floor;
    a low arm-mean positive at 10 predicts a floor of 15 with the below-floor reading."""
    weak = _name("tmle_in_sample_known", "ey0", "equal10", 10, "iid_t_control")
    low = _name("tmle_in_sample_known", "ey1", "equal10_arms", 10)
    summary = _summary(weak=frozenset({weak}), coverage={low: 0.905})
    result = properties.family_sizing(summary, "tmle_in_sample")
    assert result["reds"] == {
        f"{floor.FAMILY}/{weak}": floor.UNPOWERED_READING,
        f"{floor.FAMILY}/{low}": floor.FINITE_SAMPLE_READING,
    }
    assert result["predicted_floor"] == 15 and result["expected_floor"] == 15
    assert properties.family_sizing(_summary(), "tmle_in_sample")["reds"] == {}
    table = properties.pilot_sizing_table(summary)
    assert set(table.loc[table["kind"] == "family", "family"]) == set(study.LADDERS)
    # The low cell is the first primary candidate, so the rule moves to the second.
    assert table.loc[table["kind"] == "primary", "cell"].tolist() == [
        _name(study.PRIMARY_FIT, "ey1", study.PRIMARY_LAW, 20)
    ]


# ----------------------------------------------------------------------------- witnesses


def _witness_frame() -> pd.DataFrame:
    seed = source.draw_seed("unequal_informative", 10, 4650)
    return source.with_noise(source.draw("unequal_informative", 10, seed), seed)


def _se_over_sd(**overrides) -> float:
    """The d5 estimated configuration's ATE SE on the S-j witness draw over the committed SD."""
    table = pd.read_csv(source.STUDY.artifacts / "properties.csv")
    cell = "tmle_stacked_d5_estimated__unequal_informative__j10__t_reference"
    spread = float(table.loc[table["cell"] == cell, "empirical_se"].iloc[0])
    fit = "tmle_stacked_d5_estimated"
    if overrides:
        result = source.point_result(fit, _witness_frame(), 10, **overrides)
        return float(result["ate"].std_error) / spread
    return float(study.fit_estimates(fit, _witness_frame(), 10)["ate"].std_error) / spread


def test_the_declared_bounds_of_an_estimated_mechanism_tame_the_witness(monkeypatch):
    assert _se_over_sd(g_bounds=source.G_BOUNDS) > 10.0
    assert _se_over_sd() < 3.0
    monkeypatch.setitem(study.MECHANISM_BOUNDS, "estimated", source.G_BOUNDS)
    assert _se_over_sd() > 10.0


def _guard_frame() -> pd.DataFrame:
    return study.draw("equal10", 4, study.draw_seed("equal10", 4, 0, "unit"))


def test_a_known_mechanism_keeps_bounds_that_move_no_known_value(monkeypatch):
    """On a draw where ``"auto"`` would move a known propensity, the declared known bounds fit
    and ``"auto"`` is refused; declaring ``"auto"`` for a known mechanism refuses the fit."""
    frame = _guard_frame()
    estimates = study.fit_estimates("tmle_stacked_known", frame, 4)
    assert all(math.isfinite(float(e.psi)) for e in estimates.values())
    with pytest.raises(CapabilityError, match="outside the truncation bounds"):
        source.point_result("tmle_stacked_known", frame, 4, g_bounds="auto")
    monkeypatch.setitem(study.MECHANISM_BOUNDS, "known", "auto")
    with pytest.raises(CapabilityError, match="outside the truncation bounds"):
        study.fit_estimates("tmle_stacked_known", frame, 4)


# ------------------------------------------------------------------------------- law pin


def test_the_default_latent_reproduces_the_registered_draws():
    """The ATE laws read the default coefficient: every draw is S-a's, bit for bit, and S-a's
    committed primary replicate 0 refits to its committed estimates from that draw."""
    for law in study.ATE_LAWS:
        seed = source.draw_seed(law, 10, 3)
        pd.testing.assert_frame_equal(study.draw(law, 10, seed), source.draw(law, 10, seed))
        assert study.law_truth(law)["ate"] == source.law_truth(law)
    explicit = laws.informative_truth(latent=laws.LATENT)
    assert explicit == laws.informative_truth() and laws.LATENT == 0.6
    frame, truth = source.draw_scenario(source.SCENARIO, source.PRIMARY_N, 0)
    rows = pd.DataFrame(source.cleverly_rows(frame, truth, source.SCENARIO, 0))
    committed = pd.read_parquet(source.STUDY.artifact("replicates.parquet"))
    committed = committed.loc[committed["replicate"] == 0].set_index("estimand")
    for row in rows.itertuples():
        assert row.truth == pytest.approx(committed.loc[row.estimand, "truth"], abs=1e-12)
        assert row.estimate == pytest.approx(committed.loc[row.estimand, "estimate"], rel=1e-6)


#: Rows of the Monte Carlo check of the arm-mean truths.
MC_ROWS = 1_000_000
#: The outcome-scale ICC each arm of an arm-mean law must reach (each reads above 0.5; the
#: untreated arm of the ATE laws reads 0.07 and 0.18, the F1 finding).
ARM_ICC_FLOOR = 0.30


def _monte_carlo(law: str, seed: int = 20265443) -> dict[str, tuple[float, float, float]]:
    """Each arm's row-weighted mean of the outcome mean over ``MC_ROWS`` rows, its cluster-robust
    MC standard error, and its outcome-scale ICC (between-cluster variance of the cluster mean
    of the outcome mean over :math:`p(1 - p)`)."""
    parameters = study.law_parameters(law)
    rng = np.random.default_rng(seed)
    if law in study.EQUAL_LAWS:
        sizes = np.full(MC_ROWS // 10, 10)
    else:
        count = int(MC_ROWS / laws.SIZES.mean())
        sizes = rng.integers(int(laws.SIZES.min()), int(laws.SIZES.max()) + 1, size=count)
    cluster = np.repeat(np.arange(sizes.size), sizes)
    w1, w2 = rng.normal(size=cluster.size), rng.normal(size=cluster.size)
    w3 = rng.normal(size=sizes.size)[cluster]
    out = {}
    for name, arm in (("ey0", 0.0), ("ey1", 1.0)):
        mean = laws.informative_mean(w1, w2, w3, arm, sizes[cluster].astype(float), **parameters)
        totals = np.bincount(cluster, mean)
        estimate = totals.sum() / sizes.sum()
        error = math.sqrt((totals - estimate * sizes).var(ddof=1) * sizes.size) / sizes.sum()
        between = np.average((totals / sizes - estimate) ** 2, weights=sizes)
        out[name] = (float(estimate), error, float(between / (estimate * (1.0 - estimate))))
    return out


def test_the_arm_mean_truths_match_a_monte_carlo_and_each_arm_carries_the_latent():
    for law in study.ARM_MEAN_LAWS:
        truth = study.law_truth(law)
        for name, (estimate, error, icc) in _monte_carlo(law).items():
            assert abs(estimate - truth[name]) <= 4.0 * error, (law, name)
            assert icc >= ARM_ICC_FLOOR, (law, name)


def test_dropping_the_latent_from_the_untreated_arm_fails_the_icc_check(monkeypatch):
    original = laws.informative_mean

    def untreated_without_latent(w1, w2, w3, a, size, *, latent=laws.LATENT, **rest):
        return original(w1, w2, w3, a, size, latent=latent * np.asarray(a), **rest)

    monkeypatch.setattr(laws, "informative_mean", untreated_without_latent)
    for law in study.ARM_MEAN_LAWS:
        result = _monte_carlo(law)
        assert result["ey0"][2] < ARM_ICC_FLOOR, law
        assert result["ey1"][2] >= ARM_ICC_FLOOR, law


def test_the_untreated_arm_of_an_ate_law_carries_little_latent():
    """The F1 finding the arm-mean laws answer: on both ATE laws the untreated arm's ICC is
    under the floor each arm of an arm-mean law reaches."""
    for law in study.ATE_LAWS:
        assert _monte_carlo(law)["ey0"][2] < ARM_ICC_FLOOR, law


@pytest.mark.parametrize("law", study.ARM_MEAN_LAWS)
def test_an_arm_mean_law_changes_only_the_latent_coefficient(law):
    base = study.law_parameters(study.ARM_LAW_BASE[law])
    mine = study.law_parameters(law)
    assert {key: value for key, value in mine.items() if key != "latent"} == {
        key: value for key, value in base.items() if key != "latent"
    }
    assert mine["latent"] == study.ARM_LATENT[law] and base["latent"] == laws.LATENT
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = study.draw(law, 6, study.draw_seed(law, 6, 0, "unit"))
    assert frame["cluster"].nunique() == 6
