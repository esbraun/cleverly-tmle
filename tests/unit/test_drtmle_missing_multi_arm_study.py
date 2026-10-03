"""Before any run: the multi-arm missing-outcome study publishes exactly what it declares.

``test_method_evidence.py`` binds every committed property truth to the law of the declared
cell that names it, and it fails when the two cell sets differ. That check can only run on
committed artifacts. This module runs it before the run, on one replication per declared
spec at a small size, so a mismatch costs seconds rather than a regeneration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.studies import multi_arm_mar_drtmle_properties as props


def _published() -> pd.DataFrame:
    rows = []
    for (payload,) in props._payloads(budget=1):
        property_name, cell, replicate, n, _, seed, configuration = payload
        # Cell names do not depend on n; the ladder's smallest rung keeps its size, because
        # its role is read off it.
        small = n if n == min(props.RATE_SIZES) else 300
        rows += props._fit_replication(
            (property_name, cell, replicate, small, 1, seed, configuration)
        )
    frame = pd.DataFrame(rows)
    controls = props.calibration_controls(
        frame,
        props.STUDY,
        labels=("ate",),
        efficiency_bounds={"ate": props.EFFICIENCY_SD},
        calibration_n=props.CALIBRATION_N,
        shrunken_se_factor=props.SHRUNKEN_SE_FACTOR,
        critical=props.CRITICAL,
    )
    return pd.concat([frame, controls], ignore_index=True)


def test_the_declared_cells_are_exactly_the_published_ones() -> None:
    published = _published()
    declared = {(cell.property, cell.cell): cell for cell in props.declared_cells()}
    pairs = set(zip(published["property"], published["cell"], strict=True))
    assert pairs == set(declared), sorted(pairs ^ set(declared))
    for (family, name), cell in declared.items():
        rows = published.loc[(published["property"] == family) & (published["cell"] == name)]
        assert set(rows["role"]) == {cell.role}, (family, name)
        np.testing.assert_allclose(
            rows["truth"].to_numpy(dtype=float),
            float(cell.dgp.truth()[cell.estimand]),
            rtol=1e-12,
            atol=0.0,
            err_msg=f"{family}/{name}",
        )


def test_the_declared_cells_cover_every_registered_summary_cell() -> None:
    """Every summary cell but the two derived rate rows comes from a declared cell."""
    declared = {(cell.property, cell.cell) for cell in props.declared_cells()}
    registered = {
        (family, cell)
        for family, cells in props.STUDY.property_cells.items()
        for cell in cells
        if family != "root_n_rate"
    }
    assert registered == declared


def test_no_two_families_share_a_law_and_a_seed() -> None:
    by_stream: dict[tuple[str, int], set[str]] = {}
    for cell in props.declared_cells():
        by_stream.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in by_stream.values()), by_stream
