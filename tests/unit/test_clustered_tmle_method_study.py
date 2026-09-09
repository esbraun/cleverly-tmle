"""Focused contract tests for the clustered point-treatment evidence study."""

from __future__ import annotations

from pathlib import Path

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
    assert truth["ey0"] == pytest.approx(1.0)
    assert truth["ey1"] == pytest.approx(2.0)
    assert truth["ate"] == pytest.approx(1.0)


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
