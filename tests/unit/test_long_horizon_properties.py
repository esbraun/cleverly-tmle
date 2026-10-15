"""The shared property phase of the long-horizon studies: what its rows and summary record."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from tests import long_horizon_design as D
from tests.studies import long_horizon_properties as L


def _fit_rows(tag: str, family: str, n: int = 2_000) -> pd.DataFrame:
    fit_set = next(f for f in L.fit_sets(tag) if f.family == family)
    key = (tag, fit_set.name, 0, 1, n)
    sample = L.law_.draw(fit_set.K, fit_set.m, n, 5, fit_set.law)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.DataFrame(L.fit_key(key, sample))


@pytest.mark.parametrize("tag", ["S1", "S3"])
def test_the_support_loss_and_learned_rows_record_the_binding_share_per_node(tag: str) -> None:
    """Review minor 5: the share cannot be added after the run without refitting."""
    for family in L.BINDING_FAMILIES:
        rows = _fit_rows(tag, family)
        assert set(L.BINDING_COLUMNS) <= set(rows)
        shares = rows[list(L.BINDING_COLUMNS)].to_numpy(float)
        assert np.isfinite(shares).all()
        assert ((shares >= 0) & (shares <= 1)).all()
    # The default bound moves paths at the later nodes; the unbounded arm moves none.
    rows = _fit_rows(tag, "support_loss", n=4_000)
    bounded = rows.loc[rows["cell"].str.endswith("default_bounds"), list(L.BINDING_COLUMNS)]
    unbounded = rows.loc[rows["cell"].str.endswith("no_bounds"), list(L.BINDING_COLUMNS)]
    assert bounded.to_numpy(float).sum() > 0
    assert unbounded.to_numpy(float).sum() == 0


def test_a_gated_row_records_no_binding_share() -> None:
    rows = _fit_rows("S1", "root_n_and_efficiency")
    assert not set(L.BINDING_COLUMNS) & set(rows)


def test_the_summary_publishes_the_mean_share_of_each_cell() -> None:
    rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "property": "support_loss",
                    "cell": "static__k5__n4000__default_bounds",
                    "binding_share_1": [0.0, 0.0],
                    "binding_share_2": [0.01, 0.03],
                    "binding_share_3": [math.nan, math.nan],
                }
            ),
            pd.DataFrame({"property": "power", "cell": "c", "binding_share_1": [math.nan]}),
        ],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        {"property": ["support_loss", "power"], "cell": ["static__k5__n4000__default_bounds", "c"]}
    )
    L.binding_summary(summary, rows)
    assert summary["binding_share_2"].iloc[0] == pytest.approx(0.02)
    assert math.isnan(summary["binding_share_3"].iloc[0])
    assert math.isnan(summary["binding_share_1"].iloc[1])


def test_the_property_modules_share_the_designs_declarations() -> None:
    """The band, the root-n ladder and the cell names are the design's, declared once."""
    assert L.BAND is D.BAND
    assert L.ROOT_N_SIZES is D.ROOT_N_SIZES
    assert (L.CROSS_FITTED, L.IN_SAMPLE) == (D.CROSS_FITTED, D.IN_SAMPLE)
    for tag in D.STUDIES:
        names = {fit_set.name for fit_set in L.fit_sets(tag)}
        for study, family, cell in D.GATED_CELLS:
            if study == tag and family == "root_n_and_efficiency":
                assert f"root_n:{cell.split('__n_')[0]}:{cell.split('__n_')[1]}" in names
