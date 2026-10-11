"""The shared outcome-adaptive design still produces the committed study rows.

``oat_design="shared"`` is the construction that ``canonical-ctmle-oat`` and
``canonical-multi-arm-ctmle-oat`` published, and their study modules pin it.  The committed
artifacts are read, never refitted, by the rest of the fast tier, so an edit to the shared branch
could move those studies unseen.  This module refits replication 0 of each scenario through the
study's own ``fit_cleverly`` and compares it with the committed row.

The tolerance is ``rtol=1e-10``: the artifacts were written on another machine, and Linux and
Windows differ in the last few bits.  The rows are read with ``float_precision="round_trip"``.
"""

from __future__ import annotations

from types import ModuleType

import pandas as pd
import pytest

from tests.studies import canonical_ctmle_oat, canonical_multi_arm_ctmle_oat
from tests.studies.evidence.schema import reported_inference

pytestmark = pytest.mark.xdist_group("shared_oat_replay")

STUDIES = (canonical_ctmle_oat, canonical_multi_arm_ctmle_oat)


@pytest.mark.parametrize("module", STUDIES, ids=lambda module: module.STUDY.slug)
def test_replication_zero_refits_to_the_committed_rows(module: ModuleType) -> None:
    study = module.STUDY
    committed = pd.read_csv(study.artifact("replicates.csv.gz"), float_precision="round_trip")
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
            assert low == pytest.approx(row["ci_lower"], rel=1e-10, abs=1e-14)
            assert high == pytest.approx(row["ci_upper"], rel=1e-10, abs=1e-14)
