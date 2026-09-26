"""Fast path checks for the declared RM18 ratio routing and replay gate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.rm18_comparator_density.replay import (
    CELLS,
    POLICIES,
    SCENARIO,
    TARGET,
    analytic_arrays,
    hazard_quarter,
    reading,
    validate_cells,
)


def test_ratio_routing_preserves_both_shift_axes_and_quarter_source() -> None:
    frame = pd.DataFrame({"A": [2.0, 2.8, 3.2], "W1": [0.0, 0.0, 0.0], "W2": [0.0, 0.0, 0.0]})
    shifted = np.column_stack(
        [frame["A"], frame["A"] + 0.25, np.where(frame["A"] <= 2.5, frame["A"] + 0.5, frame["A"])]
    )
    source = np.array([[1.0, 1.7, 2.7], [1.0, 1.8, 2.8], [1.0, 1.9, 2.9]])
    shifts = SimpleNamespace(
        names=tuple(name for _, _, name in POLICIES), shifted=shifted, ratio=source
    )
    result = SimpleNamespace(nuisance=SimpleNamespace(shifts=shifts))

    observed, at_shifted = analytic_arrays(result, frame)
    assert observed.shape == (3, 3)
    assert at_shifted.shape == (3, 3, 3)
    np.testing.assert_allclose(observed[:, 0], 1.0)
    np.testing.assert_allclose(at_shifted[:, 0, :], observed)
    # Last axis is the ratio policy. Middle axis is the dose at which it is evaluated.
    np.testing.assert_allclose(at_shifted[:, 1, 1], np.exp(0.25 * (shifted[:, 1] - 2.0) - 0.03125))
    np.testing.assert_allclose(
        observed[:, 2],
        np.exp(0.5 * (frame["A"] - 2.0) - 0.125) * (frame["A"] <= 3.0) + (frame["A"] > 2.5),
    )
    np.testing.assert_array_equal(hazard_quarter(result, len(frame)), source[:, 1])

    shifts.names = tuple(reversed(shifts.names))
    with pytest.raises(ValueError, match="policy axes"):
        analytic_arrays(result, frame)
    with pytest.raises(ValueError, match="policy axes"):
        hazard_quarter(result, len(frame))


def _synthetic_cells(count: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for replicate in range(count):
        for cell_index, cell in enumerate(CELLS):
            estimate = 0.1 * cell_index + 0.02 * replicate + 0.001 * replicate**2
            rows.append(
                {
                    "cell": cell,
                    "scenario": SCENARIO,
                    "replicate": replicate,
                    "n": 2000,
                    "estimand": TARGET,
                    "truth": 0.2,
                    "estimate": estimate,
                    "inference_estimate": estimate,
                    "std_error": 0.05 + 0.001 * cell_index,
                    "ci_lower": estimate - 0.1,
                    "ci_upper": estimate + 0.1,
                    "inference_scale": "identity",
                    "covered": int(abs(estimate - 0.2) <= 0.1),
                    "initial_estimate": estimate - 0.01,
                }
            )
    cells = pd.DataFrame(rows)
    anchors = cells.loc[cells["cell"].isin(("C-H", "R-A"))].copy()
    anchors["implementation"] = anchors["cell"].map({"C-H": "cleverly", "R-A": "lmtp"})
    return cells, anchors


def test_pair_gate_rejects_missing_and_swapped_ratio_arms() -> None:
    cells, anchors = _synthetic_cells()
    validate_cells(cells, anchors, count=12)

    missing = cells.loc[~((cells["cell"] == "R-H") & (cells["replicate"] == 4))]
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        validate_cells(missing, anchors, count=12)

    swapped = cells.copy()
    swapped["cell"] = swapped["cell"].replace({"C-H": "R-A", "R-A": "C-H"})
    with pytest.raises(ValueError, match="anchor moved"):
        validate_cells(swapped, anchors, count=12)


def test_reading_uses_paired_rows_and_exact_point_decomposition() -> None:
    cells, _ = _synthetic_cells()
    summary = reading(cells, bootstraps=40).set_index("statistic")
    assert set(summary.index) == {
        *(f"signed_{cell}" for cell in CELLS),
        "observed_gap",
        "C_density",
        "R_density",
        "analytic_engine",
        "hazard_engine",
        "interaction",
    }
    assert summary["reading"].nunique() == 1
    np.testing.assert_allclose(
        summary.loc["observed_gap", "estimate"],
        summary.loc["C_density", "estimate"] + summary.loc["analytic_engine", "estimate"],
    )
    np.testing.assert_allclose(
        summary.loc["observed_gap", "estimate"],
        summary.loc["hazard_engine", "estimate"] + summary.loc["R_density", "estimate"],
    )
    assert np.all(summary["lower"] <= summary["upper"])
