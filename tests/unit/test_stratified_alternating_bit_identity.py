"""The stratified alternating targeting leaves every shipped fit unchanged.

The golden values in :mod:`tests.unit._stratified_alternating_golden` were recorded before the
stratified incremental, link-MSM, continuous-MSM, natural-course and DR-TMLE targeting changed
any source file.  Each fit below is one of those shipped paths: the unstratified fits of the
groups whose solvers gained a stratified form, and the stratified arm, regime, shift and
identity-MSM fits that already used the pooled block fluctuation.  Every value must still
match.

The learners converge to unique optima, so the comparison holds across platforms at a relative
tolerance of ``1e-8``.  On one machine the pre-change and post-change arrays agreed to the bit;
the pull request records that comparison.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit._stratified_alternating_golden import GOLDEN
from tests.unit._stratified_alternating_support import GOLDEN as BUILDERS
from tests.unit._stratified_alternating_support import frame, summary


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return frame()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_each_shipped_fit_is_unchanged(rows: pd.DataFrame, name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = BUILDERS[name](rows)
    actual = summary(result)
    expected = GOLDEN[name]
    assert set(actual) == set(expected)
    keys = sorted(expected)
    np.testing.assert_allclose(
        [actual[key] for key in keys],
        [expected[key] for key in keys],
        rtol=1e-8,
        atol=1e-9,
        err_msg=name,
    )


def test_the_golden_set_covers_every_builder() -> None:
    assert set(GOLDEN) == set(BUILDERS)
