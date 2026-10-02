"""The declared design of the full-refit bootstrap and derived-contrast study.

Every number here was fixed before the registered run: the cells, their sizes and replicate
counts, the bootstrap count, the margins, the seed streams, the failed-replicate cap, the
red-cell policy and the extrapolated budget.  A change to any of them after a verdict is seen
fails this test, which is the point: the brief forbids buying a pass with budget or margins.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.studies import canonical_full_refit_bootstrap as study
from tests.studies import full_refit_bootstrap_properties as props
from tests.studies import survival_grid_law as grid
from tests.studies.evidence.registry import Margins
from tests.studies.evidence.seeds import stream_seed

#: label -> (law, folds, n, replicates, bootstrap replicates, shrunken control).
DECLARED = {
    "rr_end_of_study": ("end_of_study", 1, 2_000, 4_000, 0, True),
    "or_end_of_study": ("end_of_study", 1, 2_000, 4_000, 0, True),
    "rr_survival": ("survival", 1, 2_000, 4_000, 0, True),
    "survival_rr_survival": ("survival", 1, 2_000, 4_000, 0, False),
    "rmst_survival": ("survival", 1, 2_000, 4_000, 0, False),
    "rmst_contrast_survival": ("survival", 1, 2_000, 4_000, 0, False),
    "rmst_crossfit": ("survival", 5, 2_000, 4_000, 0, False),
    "rmst_contrast_crossfit": ("survival", 5, 2_000, 4_000, 0, False),
    "boot_ey_end_of_study": ("end_of_study", 1, 1_000, 4_000, 200, True),
    "boot_ate_end_of_study": ("end_of_study", 1, 1_000, 4_000, 200, True),
    "boot_ey_crossfit": ("end_of_study", 5, 1_000, 4_000, 200, False),
    "boot_ate_crossfit": ("end_of_study", 5, 1_000, 4_000, 200, False),
    "boot_risk2_survival": ("survival", 1, 1_000, 4_000, 200, False),
    "boot_risk4_survival": ("survival", 1, 1_000, 4_000, 200, False),
    "boot_rmst_survival": ("survival", 1, 1_000, 4_000, 200, False),
    "boot_ey_clustered": ("clustered", 1, 1_500, 4_000, 200, False),
    "boot_ate_clustered": ("clustered", 1, 1_500, 4_000, 200, False),
    "boot_ate_point_tmle": ("point", 1, 1_000, 4_000, 200, False),
}

#: Seconds per study replicate of each fit group, from a 20-replicate smoke pass at B = 400, run on one
#: core while another study used the other cores, and the extrapolated total.  Committed
#: before the registered run; only the declared R and B run.
SMOKE_SECONDS_PER_REPLICATE: dict[str, float] = {
    "ratio_end_of_study": 0.0382,
    "derived_survival": 0.0684,
    "derived_crossfit": 0.1663,
    "boot_end_of_study": 13.9679,
    "boot_crossfit": 38.3956,
    "boot_survival": 17.3893,
    "boot_clustered": 10.2176,
    "boot_point_tmle": 1.1586,
}
#: Scaled from the smoke pass at B = 400 to the declared B = 200 and R = 4,000: about
#: 2.8 wall hours on 16 cores.
BUDGET_CORE_SECONDS = 163349.6
CORES = 16


def test_every_cell_is_declared() -> None:
    measured = {
        spec["label"]: (
            spec["law"],
            spec["folds"],
            spec["n"],
            spec["replicates"],
            spec["bootstrap"],
            spec["control"],
        )
        for spec in study.CELLS
    }
    assert measured == DECLARED


def test_the_primary_scenario_is_declared() -> None:
    assert study.STUDY.scenarios == {
        "end_of_study_ratio": ("rr_regimen[always vs never]", "or_regimen[always vs never]")
    }
    assert (study.STUDY.replicates, study.STUDY.n, study.STUDY.seed) == (1_000, 2_000, 20261013)
    assert study.STUDY.reference is None
    assert study.STUDY.publication_policy == "reporting"


def test_the_margins_are_the_shared_ones() -> None:
    assert study.STUDY.margins == Margins()


def test_the_constants() -> None:
    assert study.BOOTSTRAP_REPLICATES == 200
    assert study.PROPERTY_REPLICATES == 4_000
    assert study.FAILED_REPLICATE_CAP == 0.01
    assert study.SHRUNKEN_SE_FACTOR == 0.70
    assert (study.CLUSTERS, study.CLUSTER_SIZE, study.CLUSTER_SHIFT) == (60, 25, 0.10)


def test_the_seed_streams_are_distinct() -> None:
    groups = study.groups()
    seeds = {
        stream_seed(study.STUDY, stream, group, replicate)
        for group in groups
        for stream in ("property_sample", "fit_seed")
        for replicate in range(3)
    }
    assert len(seeds) == 2 * 3 * len(groups)


def test_the_fit_groups_are_declared() -> None:
    """Cells of one group share one sample and one fit per replicate."""
    assert {
        name: tuple(spec["label"] for spec in specs) for name, specs in study.groups().items()
    } == {
        "ratio_end_of_study": ("rr_end_of_study", "or_end_of_study"),
        "derived_survival": (
            "rr_survival",
            "survival_rr_survival",
            "rmst_survival",
            "rmst_contrast_survival",
        ),
        "derived_crossfit": ("rmst_crossfit", "rmst_contrast_crossfit"),
        "boot_end_of_study": ("boot_ey_end_of_study", "boot_ate_end_of_study"),
        "boot_crossfit": ("boot_ey_crossfit", "boot_ate_crossfit"),
        "boot_survival": ("boot_risk2_survival", "boot_risk4_survival", "boot_rmst_survival"),
        "boot_clustered": ("boot_ey_clustered", "boot_ate_clustered"),
        "boot_point_tmle": ("boot_ate_point_tmle",),
    }


def test_the_red_cell_policy_is_declared_before_the_run() -> None:
    policy = " ".join(study.RED_CELL_POLICY)
    for owner in ("X20-derived", "X20-bootstrap", "X20-point-bootstrap"):
        assert owner in policy
    assert study.RED_CELL_POLICY[0].startswith("1. The study policy is reporting.")
    assert "LICENSED_BOOTSTRAP_DESIGNS" in policy
    assert study.CONFIGURATION["red_cell_policy"] == list(study.RED_CELL_POLICY)


def test_the_property_cells_match_the_declaration() -> None:
    cells = set(study.STUDY.property_cells["interval_calibration"])
    expected = {f"{label}__correctly_specified" for label in DECLARED} | {
        f"{label}__shrunken_se_control" for label, spec in DECLARED.items() if spec[5]
    }
    assert cells == expected


def test_the_truths_come_from_the_laws() -> None:
    assert props.TRUTHS["rmst_survival"] == grid.rmst_truth("always", 5)
    assert props.TRUTHS["rr_survival"] == pytest.approx(
        np.log(grid.risk_truth("always", 4) / grid.risk_truth("never", 4)), abs=1e-15
    )


def test_the_clustered_draw_keeps_the_truth_and_clusters() -> None:
    """The latent shift averages to zero and correlates the outcomes of a cluster."""
    frame = props.sample_clustered(11)
    assert frame["cluster"].nunique() == study.CLUSTERS
    observed = frame.dropna(subset=["Y"])
    always = observed.loc[(observed["A1"] == 1) & (observed["A2"] == 1)]
    # Among always-followers the latent sign moves the outcome probability by 0.2.
    gap = (
        always.loc[always["latent"] > 0, "Y"].mean() - always.loc[always["latent"] < 0, "Y"].mean()
    )
    assert gap > 0.05


def test_the_budget_is_committed() -> None:
    if not SMOKE_SECONDS_PER_REPLICATE:
        pytest.fail("the smoke-pass budget has not been committed")
    budget = study.budget_core_seconds(SMOKE_SECONDS_PER_REPLICATE, smoke_bootstrap=400)
    assert sum(budget.values()) == pytest.approx(BUDGET_CORE_SECONDS, rel=1e-9)


def test_a_calibrated_cell_passes_with_high_probability() -> None:
    """The replicate count was chosen from this calculation before the run.

    A cell whose reported standard error is 2% short of the sampling spread, with 5% noise
    in each reported standard error, passes the shared calibration rule with probability at
    least 0.95 at the declared count.  At 1,000 replicates it would pass about 0.3 of the
    time, so a red cell would report the budget rather than the method.
    """
    replicates = study.PROPERTY_REPLICATES
    assert study.calibration_pass_probability(replicates, true_se_ratio=1.0, se_noise=0.05) > 0.99
    assert study.calibration_pass_probability(replicates, true_se_ratio=0.98, se_noise=0.05) >= 0.95
    assert study.calibration_pass_probability(1_000, true_se_ratio=0.98, se_noise=0.05) < 0.5
    # The approximation tracks a direct simulation of the rule (the implementation review's
    # probe): 0.41 and 0.96 at 1,000 and 2,400 replicates for a ratio of one.
    assert study.calibration_pass_probability(1_000, true_se_ratio=1.0) == pytest.approx(
        0.41, abs=0.05
    )
    assert study.calibration_pass_probability(2_400, true_se_ratio=1.0) == pytest.approx(
        0.96, abs=0.03
    )
    margins = Margins()
    assert margins.calibration_se_ratio == (0.93, 1.07)
    assert margins.calibration_coverage == (0.92, 0.98)


def test_the_seed_is_this_study_s_own() -> None:
    from tests.studies.evidence.registry import registered

    used = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
        if seed is not None
    }
    assert study.SEED not in used
