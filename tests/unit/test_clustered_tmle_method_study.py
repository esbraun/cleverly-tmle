"""Focused contract tests for the clustered point-treatment evidence study."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import check_integrity
from tests.studies import canonical_clustered_tmle as study
from tests.studies import clustered_tmle_properties as properties


def test_the_registered_design_matches_the_declared_plan() -> None:
    assert study.STUDY.replicates == 800
    assert study.STUDY.n == 2_000
    assert study.PROPERTY_REPLICATES == 2_400
    assert study.N_FOLDS == 5
    assert study.CLUSTER_SIZE == 10
    assert study.STUDY.scenarios == {study.SCENARIO: ("ey0", "ey1", "ate")}
    assert study.STUDY.publication_policy == "gated"
    assert properties.CONTROL_SE_RATIO_CEILING == 0.80
    assert properties.COVERAGE_GAIN == 0.03


def test_a_primary_fit_keeps_clusters_whole_and_uses_the_exact_mechanism() -> None:
    frame, truth = study.draw_from_seed(study.SCENARIO, 200, 123)
    result = study.fit_cleverly(frame)
    check_integrity(result.nuisance.folds, cluster=frame["cluster"].to_numpy())
    assert set(np.unique(result.nuisance.folds.assignment)) == set(range(study.N_FOLDS))
    assert result.nuisance.propensity.arm(1.0) == pytest.approx(
        study.law().propensity(frame[["W1", "W2"]].to_numpy()), rel=1e-12
    )
    assert set(np.unique(frame["Y"].to_numpy())) == {0.0, 1.0}
    assert truth["ey0"] == pytest.approx(0.4140653, abs=1e-6)
    assert truth["ey1"] == pytest.approx(0.5181150, abs=1e-6)
    assert truth["ate"] == pytest.approx(0.1040497, abs=1e-6)


def test_every_replication_shares_one_grouped_partition() -> None:
    """The witness behind the evidence page's conditional-coverage statement.

    The page says the published coverage is conditional on one fixed external partition.
    Two different draws are what makes that a claim rather than a restatement of the
    seed: the split has to be the same assignment, not merely the same seed.
    """
    first, _ = study.draw_from_seed(study.SCENARIO, 200, 123)
    second, _ = study.draw_from_seed(study.SCENARIO, 200, 456)
    assert not np.array_equal(first["Y"].to_numpy(), second["Y"].to_numpy())
    plans = [study.fit_cleverly(frame) for frame in (first, second)]
    assignments = [np.asarray(plan.nuisance.folds.assignment) for plan in plans]
    np.testing.assert_array_equal(*assignments)
    for plan in plans:
        assert plan.config.crossfit.scheme == "grouped"
        assert plan.config.crossfit.stratify_by == ()
        assert plan.nuisance.folds.origin is not None
        assert plan.nuisance.folds.origin.seed == study.RANDOM_STATE


def test_a_partition_that_is_not_the_declared_one_is_refused() -> None:
    """Deliberate mutation: rotate the realized assignment and require the refusal.

    Rotating by whole clusters keeps every cluster in one fold, so ``check_integrity``
    still passes and only the fixed-partition statement can catch it.
    """
    frame, _ = study.draw_from_seed(study.SCENARIO, 200, 123)
    result = study.fit_cleverly(frame)
    rotated = np.roll(np.asarray(result.nuisance.folds.assignment), study.CLUSTER_SIZE)
    mutated = dataclasses.replace(result.nuisance.folds, assignment=rotated)
    check_integrity(mutated, cluster=frame["cluster"].to_numpy())
    broken = SimpleNamespace(config=result.config, nuisance=SimpleNamespace(folds=mutated))
    with pytest.raises(RuntimeError, match="a different grouped partition"):
        study.assert_the_declared_fixed_partition(broken, frame)


def test_the_iid_property_control_reuses_the_clustered_estimate() -> None:
    robust, iid = properties._fit_replication((0, 321))
    assert robust["cell"] == "cluster_robust"
    assert iid["cell"] == "iid_control"
    assert robust["estimate"] == iid["estimate"]
    assert robust["truth"] == iid["truth"]
    assert robust["std_error"] > iid["std_error"]


def test_inflated_iid_errors_fail_only_the_control_endpoint() -> None:
    rows = pd.read_csv(study.STUDY.artifact("property-replicates.csv.gz"))
    published = properties.summarize_properties(rows).set_index("cell")
    mutated = rows.copy()
    iid = mutated["cell"] == properties.CONTROL
    mutated.loc[iid, "std_error"] *= 2.0
    summary = properties.summarize_properties(mutated).set_index("cell")
    assert bool(published.loc[properties.CONTROL, "passed"])
    assert not bool(summary.loc[properties.CONTROL, "passed"])
    assert bool(summary.loc[properties.POSITIVE, "passed"])


def test_zero_coverage_gain_fails_only_the_joint_endpoint() -> None:
    rows = pd.read_csv(study.STUDY.artifact("property-replicates.csv.gz"))
    robust = rows["cell"] == properties.POSITIVE
    iid = rows["cell"] == properties.CONTROL
    assert rows.loc[robust, "replicate"].tolist() == rows.loc[iid, "replicate"].tolist()
    mutated = rows.copy()
    mutated.loc[iid, "covered"] = rows.loc[robust, "covered"].to_numpy()
    summary = properties.summarize_properties(mutated).set_index("cell")
    assert summary["passed"].all()
    assert not summary["property_passed"].any()
    assert summary["coverage_gain_ci_lower"].eq(0.0).all()


def test_the_r_runner_uses_clustered_ife_arithmetic_for_the_ate() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "tests/canonical/lmtp_clustered_tmle/Dockerfile").read_text()
    runner = (root / "tests/canonical/lmtp_clustered_tmle/run_study.R").read_text()
    adapter = (root / "tests/canonical/lmtp_crossfit_adapter.R").read_text()
    assert 'id = "cluster"' in runner
    assert "one$estimate - zero$estimate" in runner
    assert "fit$estimate@std_error" in runner
    assert "id = id" in adapter
    assert "split across folds" in adapter
    assert study.IFE_VERSION in dockerfile
    assert study.IFE_SHA256 in dockerfile
