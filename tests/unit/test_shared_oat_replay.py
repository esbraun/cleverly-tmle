"""The shared outcome-adaptive design still produces the committed study rows.

``oat_design="shared"`` is the construction that ``canonical-ctmle-oat`` and
``canonical-multi-arm-ctmle-oat`` published, and their study modules pin it.  The committed
artifacts are read, never refitted, by the rest of the fast tier, so an edit to the shared branch
could move those studies unseen.  This module refits replication 0 of each scenario through the
study's own ``fit_cleverly`` and compares it with the committed row.

The tolerance is ``rtol=1e-10``: the artifacts were written on another machine, and Linux and
Windows differ in the last few bits.  The rows are read with ``float_precision="round_trip"``.

The rows were written under the identity interval.  The package now forms the interval of a
bounded mean (``interval_scale`` ``"range_logit"``) on the logit scale, which moves the endpoints
of the ``ey`` rows and nothing they were computed from.  For those rows the replay checks the
committed endpoints against the identity interval of the refit, and the shipped endpoints
against the logit interval of the committed estimate and standard error
(:func:`~tests.studies.evidence.interval_audit.logit_interval`), so the estimate, the standard
error and both intervals stay pinned.
"""

from __future__ import annotations

from types import ModuleType

import pytest
from scipy import stats

from tests.studies import canonical_ctmle_oat, canonical_multi_arm_ctmle_oat
from tests.studies.evidence.interval_audit import logit_interval
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.schema import reported_inference

pytestmark = pytest.mark.xdist_group("shared_oat_replay")

STUDIES = (canonical_ctmle_oat, canonical_multi_arm_ctmle_oat)


@pytest.mark.parametrize("module", STUDIES, ids=lambda module: module.STUDY.slug)
def test_replication_zero_refits_to_the_committed_rows(module: ModuleType) -> None:
    study = module.STUDY
    committed = read_rows(study.artifact("replicates.parquet"))
    for scenario in study.scenarios:
        rows = committed.loc[
            (committed["implementation"] == study.implementation)
            & (committed["scenario"] == scenario)
            & (committed["replicate"] == 0)
        ].set_index("estimand")
        assert set(rows.index) == set(study.scenarios[scenario])
        frame, _ = module.draw_scenario(scenario, study.n, 0)
        result = module.fit_cleverly(frame, scenario)
        assert result.extra["ctmle"].design == "shared"
        for name in study.scenarios[scenario]:
            estimate = result[name]
            std_error, low, high = reported_inference(estimate)
            row = rows.loc[name]
            assert estimate.psi == pytest.approx(row["estimate"], rel=1e-10, abs=1e-14)
            assert std_error == pytest.approx(row["std_error"], rel=1e-10)
            committed_interval = (row["ci_lower"], row["ci_upper"])
            if estimate.interval_scale == "range_logit":
                assert row["inference_scale"] == "identity"
                half = stats.norm.ppf(1.0 - estimate.alpha / 2.0) * std_error
                low_logit, high_logit = logit_interval(
                    row["estimate"], row["std_error"], alpha=estimate.alpha
                )
                assert (low, high) == pytest.approx((low_logit, high_logit), rel=1e-10, abs=1e-14)
                low, high = estimate.psi - half, estimate.psi + half
            assert low == pytest.approx(committed_interval[0], rel=1e-10, abs=1e-14)
            assert high == pytest.approx(committed_interval[1], rel=1e-10, abs=1e-14)
