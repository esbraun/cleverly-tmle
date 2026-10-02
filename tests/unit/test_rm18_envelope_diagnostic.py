"""Controls for the finite numerical envelope in RM18 harness validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics import rm18_shared as shared


def _rows() -> pd.DataFrame:
    return pd.DataFrame({"replicate": [0, 1], "estimate": [1.0, 2.0], "std_error": [0.1, 0.2]})


@pytest.mark.parametrize("side", ["refit", "committed"])
@pytest.mark.parametrize("column", ["estimate", "std_error"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_compared_field_fails_the_harness_envelope(
    side: str, column: str, value: float
) -> None:
    refit, committed = _rows(), _rows()
    changed = refit if side == "refit" else committed
    changed.loc[0, column] = value

    reproduced, largest, count = shared.compare_rows(refit, committed, ["replicate"])

    assert not reproduced
    assert largest == np.inf
    assert count == 2


def test_identical_finite_rows_reproduce() -> None:
    assert shared.compare_rows(_rows(), _rows(), ["replicate"]) == (True, 0.0, 2)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("factor", [0.5, 2.0])
@pytest.mark.parametrize("column", ["estimate", "std_error"])
def test_finite_positive_and_negative_differences_keep_the_declared_tolerance(
    direction: int, factor: float, column: str
) -> None:
    committed = _rows()
    refit = committed.copy()
    baseline = committed.loc[1, column]
    perturbation = direction * factor * shared.TOLERANCE * max(1.0, abs(baseline))
    refit.loc[1, column] += perturbation

    reproduced, largest, count = shared.compare_rows(refit, committed, ["replicate"])

    assert reproduced == (factor < 1.0)
    assert largest == pytest.approx(factor * shared.TOLERANCE, rel=1e-6, abs=0.0)
    assert count == 2
