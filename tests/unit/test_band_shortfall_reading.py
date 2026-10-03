"""The finite-sample reading of every red joint-coverage cell, rebuilt from the committed rows.

Each red band cell was read once by the declared diagnostic: the same replications, read again at
the design critical value ``c*`` of ``tests/unit/test_simultaneous_cell_design.py``.  A joint row
records the max-t statistic in ``estimate``, so the oracle band covers a replication exactly when
``estimate <= c*``, and the package band covers it when ``estimate <= std_error``.

The reading is finite-sample when the oracle band also under-covers.  The package critical value
then accounts for only the small gap between the two coverages.  For the strata and survival
cells ``c*`` comes from the exact law.  For a default-band shape it comes from the package's own
influence curves averaged over the design fits, so it is independent of the multiplier but not of
the curves; the source study's pointwise calibration cells are the evidence for those curves.

The owner rows of ``docs/roadmap.md`` and the study pages quote every number below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.studies import default_band_properties, stratified_law
from tests.studies.default_band_properties import design, design_correlation
from tests.studies.evidence.registry import ROOT
from tests.unit.test_simultaneous_cell_design import EXACT


def _exact(cell: str, replicates: int) -> float:
    return design(EXACT[cell][0], replicates).critical


def _shape(label: str) -> float:
    shape = default_band_properties.SHAPE_BY_LABEL[label]
    return design(design_correlation(label), shape.replicates).critical


READINGS = {
    "strata": (
        "tmle3_stratified",
        lambda: _exact("canonical-stratified-tmle/strata", 2400),
        2.846,
        2.816,
        0.9046,
        0.9121,
    ),
    "crossfit_strata": (
        "tmle3_stratified",
        lambda: _exact("canonical-stratified-tmle/crossfit_strata", 2400),
        2.778,
        2.750,
        0.9192,
        0.9258,
    ),
    "all_reported": (
        "ltmle_survival",
        lambda: _exact("canonical-ltmle-survival/all_reported", 9600),
        2.622,
        2.605,
        0.9214,
        0.9241,
    ),
    "categorical_ltmle": (
        "default_bands",
        lambda: _shape("categorical_ltmle"),
        2.689,
        2.680,
        0.9250,
        0.9267,
    ),
    "categorical_ltmle_crossfit": (
        "default_bands",
        lambda: _shape("categorical_ltmle_crossfit"),
        2.689,
        2.680,
        0.9317,
        0.9325,
    ),
    "cde_z0": ("default_bands", lambda: _shape("cde_z0"), 2.331, 2.312, 0.9171, 0.9200),
}


@pytest.mark.parametrize("label", sorted(READINGS))
def test_the_red_band_reading_is_the_committed_rows_own(label: str) -> None:
    directory, critical, quoted_critical, package_critical, package, oracle = READINGS[label]
    rows = pd.read_csv(
        ROOT / "tests" / "canonical" / directory / "property-replicates.csv.gz",
        float_precision="round_trip",
    )
    band = rows.loc[rows["cell"] == f"{label}__simultaneous_band"]
    assert len(band) == int(band["requested_replicates"].iloc[0])
    c_star = critical()
    assert round(c_star, 3) == quoted_critical
    assert round(float(band["std_error"].mean()), 3) == package_critical
    # The package band's own verdict, read back from the statistic it recorded.
    assert (band["covered"] == (band["estimate"] <= band["std_error"]).astype(int)).all()
    assert round(float(band["covered"].mean()), 4) == package
    oracle_coverage = float((band["estimate"] <= c_star).mean())
    assert round(oracle_coverage, 4) == oracle
    # The finite-sample reading: the oracle band under-covers too, and the multiplier's lower
    # critical value explains less of the shortfall than the oracle band shows.
    assert oracle_coverage < 0.95
    assert oracle - package < 0.95 - oracle


def test_the_smallest_stratum_reading() -> None:
    """The ``strata-boundary-mean`` reading: a mean near 0.94 on few treated rows.

    At n = 2,000 the stratum V = 2 holds about 157 treated rows, and about 9 of them are expected
    to be non-events.  A Wald interval of a mean that close to 1 under-covers at that count.
    """
    v, _, a, y = np.asarray(stratified_law.SUPPORT, dtype=float).T
    probs = stratified_law.PROBS
    inside = v == 2
    assert round(stratified_law.TRUTH["ey[1][V=2]"], 4) == 0.9358
    treated_share = probs[inside & (a == 1)].sum() / probs[inside].sum()
    assert round(float(treated_share), 4) == 0.3938
    assert int(2000 * float(probs[inside].sum()) * float(treated_share)) == 157
    assert round(2000 * float(probs[inside & (a == 1) & (y == 0)].sum())) == 9
