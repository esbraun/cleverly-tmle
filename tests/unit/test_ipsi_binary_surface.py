r"""Does the categorical machinery leave the two-level longitudinal tilt where it was?

T16: the first five replicates of the committed longitudinal incremental studies, refitted
here, equal the committed ``cleverly`` rows.  The rows were written by the code before the
categorical change, so a two-level node must still run Kennedy's arithmetic bit for bit.

* ``longitudinal-incremental``: the cross-fitted IPSI estimator without dropout.
* ``longitudinal-incremental-dropout``: the same with dropout, and the survival curve.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.studies import canonical_longitudinal_incremental as plain
from tests.studies import canonical_longitudinal_incremental_dropout as dropout

pytestmark = pytest.mark.xdist_group("ipsi_binary_surface")

REPLICATES = 5
ROOT = "tests/canonical/"
COMMITTED = {
    plain.STUDY.implementation: ROOT + "npcausal_longitudinal_incremental/replicates.csv.gz",
    dropout.STUDY.implementation: ROOT + "longitudinal_incremental_dropout/replicates.csv.gz",
}


def _committed(implementation: str, scenario: str, replicate: int) -> pd.DataFrame:
    rows = pd.read_csv(COMMITTED[implementation], float_precision="round_trip")
    keep = (
        (rows["implementation"] == implementation)
        & (rows["scenario"] == scenario)
        & (rows["replicate"] == replicate)
    )
    assert keep.any(), "the committed artifact has no rows for this replicate"
    return rows[keep].set_index("estimand")


def _agree(result: Any, committed: pd.DataFrame) -> None:
    for name, row in committed.iterrows():
        estimate = result[str(name)]
        assert estimate.psi == pytest.approx(row["estimate"], rel=1e-8, abs=1e-10), name
        assert estimate.std_error == pytest.approx(row["std_error"], rel=1e-6, abs=1e-10), name


@pytest.mark.parametrize("replicate", range(REPLICATES))
def test_the_cross_fitted_study_rows_come_back(replicate: int) -> None:
    scenario = "ipsi_folds_5"
    frame, _ = plain.draw_scenario(scenario, plain.PRIMARY_N, replicate)
    _agree(plain.fit(frame, scenario), _committed(plain.STUDY.implementation, scenario, replicate))


@pytest.mark.parametrize("scenario", ["dropout_folds_5", "survival_folds_1"])
@pytest.mark.parametrize("replicate", range(REPLICATES))
def test_the_dropout_study_rows_come_back(scenario: str, replicate: int) -> None:
    frame, _ = dropout.draw_scenario(scenario, dropout.PRIMARY_N, replicate)
    _agree(
        dropout.fit(frame, scenario),
        _committed(dropout.STUDY.implementation, scenario, replicate),
    )
