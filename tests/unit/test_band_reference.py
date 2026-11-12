"""The per-coordinate Student t reference of the simultaneous band (Fix-2 plan, section 4).

``band_reference="t"`` keeps the multiplier critical value ``c`` and bands estimate ``j`` at
``c_j = t.isf(norm.sf(c), nu_j)``, with ``nu_j = max(Kish_j, 5)`` read from the centred curve.
These tests pin the definition and its witnesses: the shared Kish count, the mapping on a heavy
column against two Gaussian ones with a mutation control for each way it can go wrong, the
limit, the refusals, the unchanged default band and the one-estimate case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from cleverly.inference import kish_count

# ------------------------------------------------------------------------------ 1. definition


def _mean_zero(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


class TestTheKishCount:
    def test_it_is_the_squared_mass_over_the_fourth_moment_of_the_centred_curve(self) -> None:
        rng = np.random.default_rng(1)
        curve = rng.standard_exponential(500) ** 2 + 3.0
        centred = curve - curve.mean()
        expected = float(np.sum(centred**2) ** 2 / np.sum(centred**4))
        assert kish_count(curve) == pytest.approx(expected, rel=1e-12)
        # A nonzero-mean witness: the uncentred count differs by far more than rounding.
        assert kish_count(curve, centered=False) > 2.0 * kish_count(curve)

    def test_appended_zero_rows_leave_a_mean_zero_curve_unchanged(self) -> None:
        curve = _mean_zero(np.random.default_rng(2).standard_t(3, size=400))
        padded = np.concatenate([curve, np.zeros(600)])
        assert kish_count(padded) == pytest.approx(kish_count(curve), rel=1e-12)

    def test_rows_select_the_terms_read(self) -> None:
        curve = np.array([3.0, -1.0, -1.0, -1.0, 0.0, 0.0])
        rows = curve != 0.0
        assert kish_count(curve, rows=rows) == pytest.approx(144.0 / 84.0, rel=1e-12)
        assert kish_count(curve, rows=np.flatnonzero(rows)) == kish_count(curve, rows=rows)

    def test_a_zero_curve_is_infinite(self) -> None:
        assert kish_count(np.zeros(10)) == math.inf
        assert kish_count(np.full(10, 2.5)) == math.inf

    @pytest.mark.parametrize("factor", [-3.0, 0.01, 7.0])
    def test_rescaling_leaves_it_unchanged(self, factor: float) -> None:
        curve = _mean_zero(np.random.default_rng(3).standard_t(4, size=300))
        assert kish_count(factor * curve) == pytest.approx(kish_count(curve), rel=1e-12)

    def test_the_sparse_report_reads_the_same_count(self) -> None:
        from cleverly.inference.sparse import influence_row

        curve = np.random.default_rng(4).standard_normal(1_000) ** 3
        row = influence_row("x", curve)
        assert row.kish == pytest.approx(kish_count(curve), rel=1e-12)
        # The count the report computed before it called the shared function.
        assert row.kish == pytest.approx(row.n / row.kurtosis, rel=1e-12)

    def test_it_matches_fay_and_graubard_to_order_r_over_n_squared(self) -> None:
        """``R / (1 + (R - 1)/(n - 1)^2)`` with ``R`` the count (plan 2.1 step 2)."""
        curve = _mean_zero(np.random.default_rng(5).standard_normal(2_000))
        count = kish_count(curve)
        realised = count / (1.0 + (count - 1.0) / (curve.size - 1.0) ** 2)
        assert abs(realised / count - 1.0) < 2.0 * count / curve.size**2
        assert stats.kurtosis(curve, fisher=False) == pytest.approx(curve.size / count, rel=1e-9)
