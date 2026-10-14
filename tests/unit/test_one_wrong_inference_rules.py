"""The four rules of the ``one_wrong_inference`` family, on synthetic replicate rows.

Each rule gets a passing table and a deliberate mutation that must fail it: a nominal SE for
the conservative rule, a wide SE for the accuracy rule, a reversed declared sign for the
first-order control, and a shrinking bias for the growth control.  The rows are drawn with a
known bias and SE ratio, so no fit runs.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import drtmle_one_wrong as study
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import (
    ONE_WRONG_COLUMNS,
    ONE_WRONG_FAMILY,
    apply_shared_verdicts,
    one_wrong_inference_verdicts,
    one_wrong_stem,
)

TRUTH = 0.5
SD = 0.01


def _rows(cell: str, *, bias: float, ratio: float, replicates: int, seed: int) -> pd.DataFrame:
    """``replicates`` rows with standardized bias ``bias`` and SE ratio ``ratio``."""
    rng = np.random.default_rng(seed)
    estimate = TRUTH + bias * SD + SD * rng.standard_normal(replicates)
    std_error = np.full(replicates, ratio * SD)
    kind = cell.rsplit("__", 1)[-1]
    built = pd.DataFrame(
        {
            "property": ONE_WRONG_FAMILY,
            "cell": cell,
            "role": "positive" if kind in {"conservative", "valid"} else "control",
            "replicate": np.arange(replicates),
            "n": 4_000,
            "requested_replicates": replicates,
            "failed_replicates": 0,
            "truth": TRUTH,
            "estimate": estimate,
            "std_error": std_error,
            "covered": (np.abs(estimate - TRUTH) <= 1.96 * std_error).astype(int),
            "rejected": 0,
        }
    )
    return built.loc[:, list(REPLICATE_COLUMNS)]


def _verdicts(
    tables: dict[str, dict[str, Any]],
    *,
    signs: dict[str, int] | None = None,
    growth: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Every cell's published verdict, with the growth rows built from their ladders."""
    rows = pd.concat(
        [_rows(cell, seed=index, **spec) for index, (cell, spec) in enumerate(tables.items())],
        ignore_index=True,
    )
    record = dataclasses.replace(study.STUDY, property_cells={ONE_WRONG_FAMILY: (*tables, *growth)})
    summary, _ = apply_shared_verdicts(
        rows, record, extra_columns=ONE_WRONG_COLUMNS, rate_labels=()
    )
    built = one_wrong_inference_verdicts(
        summary,
        rows,
        record,
        predicted_signs=signs or {},
        ladders={"setting": (1_000, 16_000)},
    )
    published = pd.concat([summary, pd.DataFrame(built)], ignore_index=True)
    return published.set_index("cell")


def test_the_conservative_rule_needs_a_ratio_wholly_above_the_band() -> None:
    published = _verdicts(
        {
            "wide__ate__conservative": {"bias": 0.0, "ratio": 1.2, "replicates": 4_000},
            "nominal__ate__conservative": {"bias": 0.0, "ratio": 1.0, "replicates": 4_000},
            "biased__ate__conservative": {"bias": 0.6, "ratio": 1.2, "replicates": 4_000},
        }
    )
    assert bool(published.loc["wide__ate__conservative", "passed"])
    assert published.loc["wide__ate__conservative", "se_ratio_ci_lower"] > 1.07
    # A calibrated SE is exactly what the cell must not show: it witnesses the term's presence.
    assert not bool(published.loc["nominal__ate__conservative", "passed"])
    assert not bool(published.loc["biased__ate__conservative", "passed"])


def test_the_valid_rule_reads_the_sanity_band_and_the_bias() -> None:
    published = _verdicts(
        {
            "calibrated__ey1__valid": {"bias": 0.0, "ratio": 1.0, "replicates": 2_000},
            "wide__ey1__valid": {"bias": 0.0, "ratio": 1.3, "replicates": 2_000},
            "biased__ey1__valid": {"bias": 0.6, "ratio": 1.0, "replicates": 2_000},
            "narrow__ey1__valid": {"bias": 0.0, "ratio": 0.7, "replicates": 2_000},
        }
    )
    assert bool(published.loc["calibrated__ey1__valid", "passed"])
    # The SE-ratio interval is published beside the verdict and decides nothing.
    assert pd.notna(published.loc["calibrated__ey1__valid", "se_ratio_ci_lower"])
    for cell in ("wide__ey1__valid", "biased__ey1__valid", "narrow__ey1__valid"):
        assert not bool(published.loc[cell, "passed"]), cell


def test_the_first_order_rule_is_one_sided_on_the_declared_sign() -> None:
    tables = {
        "below__ate__first_order": {"bias": -0.8, "ratio": 1.0, "replicates": 1_000},
        "small__ate__first_order": {"bias": -0.1, "ratio": 1.0, "replicates": 1_000},
        "collapsed__ate__first_order": {"bias": -0.8, "ratio": 50.0, "replicates": 1_000},
    }
    negative = {one_wrong_stem(cell): -1 for cell in tables}
    published = _verdicts(tables, signs=negative)
    assert bool(published.loc["below__ate__first_order", "passed"])
    assert published.loc["below__ate__first_order", "predicted_sign"] == -1
    assert not bool(published.loc["small__ate__first_order", "passed"])
    assert not bool(published.loc["collapsed__ate__first_order", "passed"])
    # The mutation: the same bias read against the opposite declared sign fails.
    flipped = _verdicts(tables, signs=dict.fromkeys(negative, 1))
    assert not bool(flipped.loc["below__ate__first_order", "passed"])


@pytest.mark.parametrize(
    ("first", "last", "sign", "expected"),
    [
        (-0.6, -1.2, -1, True),
        (-1.2, -0.6, -1, False),
        (-0.6, -1.2, 1, False),
        (-0.8, -0.8, -1, False),
    ],
)
def test_the_growth_rule_reads_the_signed_difference(
    first: float, last: float, sign: int, expected: bool
) -> None:
    tables = {
        "setting_n1000__ate__first_order": {"bias": first, "ratio": 1.0, "replicates": 1_000},
        "setting_n16000__ate__first_order": {"bias": last, "ratio": 1.0, "replicates": 1_000},
    }
    published = _verdicts(tables, signs={"setting__ate": sign}, growth=("setting__ate__growth",))
    row = published.loc["setting__ate__growth"]
    assert row["role"] == "control"
    assert row["growth"] == pytest.approx(sign * (last - first), abs=0.15)
    assert row["growth_ci_lower"] < row["growth"] < row["growth_ci_upper"]
    assert bool(row["passed"]) is expected


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="no one_wrong_inference rule"):
        _verdicts({"cell__ate__mystery": {"bias": 0.0, "ratio": 1.0, "replicates": 200}})
