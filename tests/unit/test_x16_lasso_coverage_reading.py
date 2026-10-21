"""The reading of the red ``binary_lasso`` coverage row of ``canonical-ctmle-sequences``.

The owner is ``X16-plugin-variance``.  Each test rebuilds a number of the reading from committed
rows, so the reading cannot drift from the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.diagnostics.x16_lasso_coverage import run as diagnostic
from tests.studies import canonical_ctmle_sequences as study
from tests.studies.evidence.manifest import read_rows


def _rows() -> pd.DataFrame:
    return pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")


def test_the_lasso_rows_are_the_registered_rows() -> None:
    """The refit reproduces the committed estimates and plug-in standard errors exactly."""
    rows = _rows()
    assert len(rows) == study.PRIMARY_REPLICATES
    committed = read_rows(study.STUDY.artifact("replicates.parquet"))
    committed = committed.loc[
        (committed["implementation"] == study.IMPLEMENTATION)
        & (committed["scenario"] == diagnostic.SCENARIO)
    ].sort_values("replicate", ignore_index=True)
    assert np.array_equal(committed["replicate"], rows["replicate"])
    np.testing.assert_allclose(rows["lasso_estimate"], committed["estimate"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rows["lasso_std_error"], committed["std_error"], rtol=0, atol=1e-12)


def test_plain_tmle_shares_the_shortfall_on_the_same_samples() -> None:
    """The final step reports the penalty 0 on 771 of 800 fits, and TMLE covers alike.

    Both estimators cover about 0.92 with an SE ratio near 0.91.  Coverage falls in the
    lowest quartile of the reported standard error, where the plug-in variance is too small.
    """
    rows = _rows()
    reading = diagnostic.reading(rows).set_index("estimator")
    committed = pd.read_csv(diagnostic.HERE / "reading.csv").set_index("estimator")
    pd.testing.assert_frame_equal(reading, committed, check_exact=False, rtol=1e-5)
    assert int(rows["lasso_unpenalized"].sum()) == 771
    assert round(float(reading.loc["lasso", "coverage"]), 5) == 0.92375
    assert round(float(reading.loc["tmle", "coverage"]), 4) == 0.9225
    assert round(float(reading.loc["lasso", "se_ratio"]), 3) == 0.905
    assert round(float(reading.loc["tmle", "se_ratio"]), 3) == 0.913
    assert float(reading.loc["lasso", "lowest_quartile_coverage"]) < 0.85
    assert float(reading.loc["tmle", "lowest_quartile_coverage"]) < 0.85


def test_the_true_propensity_lies_outside_the_bounds_on_a_tenth_of_rows() -> None:
    reading = pd.read_csv(diagnostic.HERE / "reading.csv").set_index("estimator")
    assert round(float(reading.loc["lasso", "truncated_share"]), 3) == 0.106
