"""The two-clause IID control of the ``few_cluster_reference`` family.

The control shows that the law's clustering changes the fit's variance by more than the SE
screen admits. Clause 1 reads the paired ratio of the mean IID SE to the mean cluster SE of the
same fits, below 0.80. Clause 2 requires the paired positive arm's SE-ratio upper bound at most
1.20. The paired ratio alone passes a cluster SE inflated by ``c`` at about ``1/c`` on a law
with no cluster effect; mutation control 2 builds exactly that case and needs clause 2 to fail
it. Mutation control 1 replaces each cluster SE with its IID SE in committed rows, and every
control must fail clause 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import clustered_few_cluster_tmle as few_tmle
from tests.studies import few_cluster_crossfit_ltmle as few_ltmle
from tests.studies import few_cluster_crossfit_properties as few_ltmle_properties
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts as pv
from tests.studies.evidence.inference import Interval, percentile_interval
from tests.studies.evidence.properties import (
    bootstrap_draw_blocks,
    control_row,
    paired_se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    FEW_CLUSTER_COLUMNS,
    FEW_CLUSTER_FAMILY,
    FEW_CLUSTER_IID_CONTROL_CEILING,
    apply_shared_verdicts,
    few_cluster_reference_verdicts,
)

pytestmark = pytest.mark.xdist_group("few_cluster_control")

RECORD = few_tmle.STUDY
POSITIVE, CONTROL, DIAGNOSTIC = "fit__t_reference", "fit__iid_t_control", "fit__normal_reference"
GROUPS = ((POSITIVE, CONTROL, DIAGNOSTIC),)
TRUTH = 0.0
CRITICAL = 2.0


def synthetic_rows(
    replicates: int, *, cluster_scale: float, iid_scale: float, seed: int = 0
) -> pd.DataFrame:
    """One fit per replicate: the estimate, its cluster SE and its IID SE.

    The empirical SD of the estimates is about 1. The IID SE is ``iid_scale`` and the cluster
    SE ``cluster_scale``, each with a little noise shared by the fit, as two SEs of one fit
    share it.
    """
    rng = np.random.default_rng(seed)
    estimates = rng.normal(TRUTH, 1.0, replicates)
    noise = np.exp(rng.normal(0.0, 0.05, replicates))
    rows = []
    for replicate in range(replicates):
        common = {
            "property_name": FEW_CLUSTER_FAMILY,
            "replicate": replicate,
            "n": 100,
            "requested": replicates,
            "truth": TRUTH,
            "estimate": float(estimates[replicate]),
            "critical": CRITICAL,
        }
        cluster = cluster_scale * float(noise[replicate])
        rows.append(control_row(cell=POSITIVE, standard_error=cluster, role="positive", **common))
        rows.append(
            control_row(cell=CONTROL, standard_error=iid_scale * float(noise[replicate]), **common)
        )
        rows.append(
            control_row(cell=DIAGNOSTIC, standard_error=cluster, role=pv.DIAGNOSTIC_ROLE, **common)
        )
    return pd.DataFrame(rows)


def verdicts(rows: pd.DataFrame, groups: Any = GROUPS, record: Any = RECORD) -> pd.DataFrame:
    summary, _ = apply_shared_verdicts(
        rows, record, extra_columns=FEW_CLUSTER_COLUMNS, rate_labels=()
    )
    few_cluster_reference_verdicts(summary, rows, record, groups)
    return summary.set_index("cell")


class TestThePairedRatio:
    def test_it_is_the_ratio_of_mean_standard_errors(self) -> None:
        rows = synthetic_rows(400, cluster_scale=1.0, iid_scale=0.6)
        control = rows[rows["cell"] == CONTROL]
        positive = rows[rows["cell"] == POSITIVE]
        paired = paired_se_ratio_interval(
            control, positive, replicates=2_000, confidence_level=0.99, seed=5
        )
        assert paired.ratio == pytest.approx(
            control["std_error"].mean() / positive["std_error"].mean(), rel=1e-12
        )
        assert paired.interval.low < paired.ratio < paired.interval.high

    def test_the_bootstrap_resamples_the_two_arms_jointly(self) -> None:
        """The fixed-seed check: one shared index draw, and the independent draw is wider."""
        rows = synthetic_rows(400, cluster_scale=1.0, iid_scale=0.6, seed=2)
        values = np.column_stack(
            [
                rows.loc[rows["cell"] == cell, "std_error"].to_numpy(dtype=float)
                for cell in (CONTROL, POSITIVE)
            ]
        )
        rng = np.random.default_rng(9)
        means = np.concatenate(
            [d.mean(axis=1) for d in bootstrap_draw_blocks(values, replicates=2_000, rng=rng)]
        )
        joint = percentile_interval(means[:, 0] / means[:, 1], confidence_level=0.99)
        paired = paired_se_ratio_interval(
            rows[rows["cell"] == CONTROL],
            rows[rows["cell"] == POSITIVE],
            replicates=2_000,
            confidence_level=0.99,
            seed=9,
        )
        assert (paired.interval.low, paired.interval.high) == (joint.low, joint.high)
        rng = np.random.default_rng(9)
        apart = [
            np.concatenate(
                [d.mean(axis=1) for d in bootstrap_draw_blocks(column, replicates=2_000, rng=rng)]
            )
            for column in values.T
        ]
        independent = percentile_interval(apart[0] / apart[1], confidence_level=0.99)
        assert independent.high - independent.low > 1.5 * (joint.high - joint.low)


class TestTheTwoClauses:
    def test_a_clustered_law_with_a_calibrated_cluster_se_passes(self) -> None:
        summary = verdicts(synthetic_rows(400, cluster_scale=1.0, iid_scale=0.6))
        control = summary.loc[CONTROL]
        assert control["paired_se_ratio_ci_upper"] < FEW_CLUSTER_IID_CONTROL_CEILING
        assert control["paired_positive_se_ratio_ci_upper"] <= RECORD.margins.se_ratio_sanity[1]
        assert bool(control["passed"])

    def test_no_cluster_effect_fails_clause_one(self) -> None:
        summary = verdicts(synthetic_rows(400, cluster_scale=1.0, iid_scale=1.0))
        assert summary.loc[CONTROL, "paired_se_ratio"] == pytest.approx(1.0, rel=1e-12)
        assert not bool(summary.loc[CONTROL, "passed"])

    def test_an_inflated_cluster_se_fails_clause_two(self) -> None:
        """Mutation control 2: no cluster effect, a calibrated IID SE, the cluster SE doubled."""
        summary = verdicts(synthetic_rows(400, cluster_scale=2.0, iid_scale=1.0))
        control = summary.loc[CONTROL]
        assert control["paired_se_ratio"] == pytest.approx(0.5, rel=1e-12)
        # A verdict built from clause 1 alone would pass.
        assert control["paired_se_ratio_ci_upper"] < FEW_CLUSTER_IID_CONTROL_CEILING
        assert control["paired_positive_se_ratio_ci_upper"] > 1.8
        assert not bool(control["passed"])
        # The old statistic, IID SE over empirical SD, stays published and reads about 1.
        assert control["se_ratio"] == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize(("high", "passed"), [(1.20, True), (1.2001, False)])
    def test_clause_two_is_at_most_the_ceiling(
        self, high: float, passed: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pv, "se_ratio_interval", lambda group, **_: Interval(0.9, high))
        summary = verdicts(synthetic_rows(400, cluster_scale=1.0, iid_scale=0.6))
        assert summary.loc[CONTROL, "paired_positive_se_ratio_ci_upper"] == high
        assert bool(summary.loc[CONTROL, "passed"]) is passed

    def test_an_undeclared_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match="in no declared group"):
            verdicts(synthetic_rows(50, cluster_scale=1.0, iid_scale=0.6), groups=())


class TestTheCommittedRows:
    """Mutation control 1 on the committed rows of ``few-cluster-cross-fitted-ltmle``."""

    @pytest.fixture(scope="class")
    def rows(self) -> pd.DataFrame:
        return pd.read_parquet(few_ltmle.STUDY.artifact("property-replicates.parquet"))

    def test_every_control_passes_both_clauses(self, rows: pd.DataFrame) -> None:
        groups = few_ltmle_properties.PROPERTY_GROUPS[FEW_CLUSTER_FAMILY]
        summary = verdicts(rows, groups, few_ltmle.STUDY)
        controls = summary[summary["role"] == "control"]
        assert len(controls) == 4
        # The committed reading the regeneration is expected to reproduce.
        assert controls["paired_se_ratio"].between(0.60, 0.68).all()
        assert (controls["paired_positive_se_ratio_ci_upper"] <= 1.05).all()
        assert controls["passed"].all()

    def test_the_cluster_se_swapped_for_the_iid_se_fails_every_control(
        self, rows: pd.DataFrame
    ) -> None:
        groups = few_ltmle_properties.PROPERTY_GROUPS[FEW_CLUSTER_FAMILY]
        mutated = rows.copy()
        for positive, control, _ in groups:
            iid = mutated.loc[mutated["cell"] == control].set_index("replicate")["std_error"]
            mask = mutated["cell"] == positive
            mutated.loc[mask, "std_error"] = (
                mutated.loc[mask, "replicate"].map(iid).to_numpy(dtype=float)
            )
        summary = verdicts(mutated, groups, few_ltmle.STUDY)
        controls = summary[summary["role"] == "control"]
        assert np.allclose(controls["paired_se_ratio"], 1.0, rtol=1e-12)
        assert not controls["passed"].any()


def _context(rows: pd.DataFrame) -> pf.Context:
    summary = pd.DataFrame(
        {
            "property": FEW_CLUSTER_FAMILY,
            "cell": [POSITIVE, CONTROL, DIAGNOSTIC],
            "role": ["positive", "control", pv.DIAGNOSTIC_ROLE],
        }
    )
    return pf.Context(RECORD, rows, summary, groups={FEW_CLUSTER_FAMILY: GROUPS})


class TestLostReplicates:
    """The worst case of clause 1 fills both arms: the IID arm up, the cluster arm down."""

    @pytest.fixture(scope="class")
    def lost(self) -> pd.DataFrame:
        rows = synthetic_rows(400, cluster_scale=1.0, iid_scale=0.6, seed=4)
        drop = ((rows["cell"] == CONTROL) & (rows["replicate"] == 7)) | (
            (rows["cell"] == POSITIVE) & (rows["replicate"] == 11)
        )
        rows = rows.loc[~drop].copy()
        rows.loc[rows["cell"].isin([CONTROL, POSITIVE]), "failed_replicates"] = 1
        return rows

    def _by_hand(self, rows: pd.DataFrame, control_fill: float, positive_fill: float) -> float:
        def complete(cell: str, lost_id: int, fill: float) -> pd.DataFrame:
            group = rows.loc[rows["cell"] == cell, ["replicate", "std_error"]]
            return pd.concat(
                [group, pd.DataFrame({"replicate": [lost_id], "std_error": [fill]})],
                ignore_index=True,
            )

        margins = RECORD.margins
        return paired_se_ratio_interval(
            complete(CONTROL, 7, control_fill),
            complete(POSITIVE, 11, positive_fill),
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=pv.stream_seed(RECORD, FEW_CLUSTER_FAMILY, CONTROL),
        ).interval.high

    def test_the_candidate_fills_both_arms_toward_the_ceiling(self, lost: pd.DataFrame) -> None:
        ctx = _context(lost)
        rule = pv._few_cluster_rule()
        clause = rule.clause("paired iid/cluster se ratio under its ceiling")
        row = pd.Series({"property": FEW_CLUSTER_FAMILY, "cell": CONTROL, "role": "control"})
        ((_, judged),) = clause.candidates(ctx, row)
        control = lost.loc[lost["cell"] == CONTROL, "std_error"]
        positive = lost.loc[lost["cell"] == POSITIVE, "std_error"]
        worst = self._by_hand(lost, float(control.max()), float(positive.min()))
        assert judged.slack == pytest.approx(FEW_CLUSTER_IID_CONTROL_CEILING - worst, abs=1e-15)
        # A version that fills only the control arm, leaving the cluster arm at its mean, gives
        # a lower bound, so it would not be the worst case.
        control_only = self._by_hand(lost, float(control.max()), float(positive.mean()))
        assert control_only < worst

    def test_the_positive_clause_reads_the_positive_cell(self, lost: pd.DataFrame) -> None:
        ctx = _context(lost)
        clause = pv._few_cluster_rule().clause(
            "paired positive se ratio at most the sanity ceiling"
        )
        row = pd.Series({"property": FEW_CLUSTER_FAMILY, "cell": CONTROL, "role": "control"})
        assert clause.reads is not None
        assert (FEW_CLUSTER_FAMILY, POSITIVE) in clause.reads(ctx, row)
        candidates = clause.candidates(ctx, row)
        # The S4 set of a cell that lost one replicate: one SE-at-max set and two vertices.
        assert len(candidates) == 3
        assert all(judged.passed for _, judged in candidates)
