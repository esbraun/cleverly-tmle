"""The committed interval audits recompute exactly from the committed rows.

``tests/studies/evidence/interval_audit.py`` reads each audited study's bounded-mean rows under
the shipped logit interval.  Each audit file stores the keys, the trigger and the shipped
verdicts; the committed estimates, standard errors and truths are joined back from the study's
row files, so an audit record that drifts from its row fails here.  The numbers the roadmap and
the study pages quote are rebuilt from the same records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies.evidence import interval_audit as audit

SHIPPED = ["trigger", "scale", "covered_shipped", "rejected_shipped"]


@pytest.fixture(scope="module", params=sorted(audit.AUDITS))
def committed(request: pytest.FixtureRequest) -> tuple[str, pd.DataFrame]:
    record = audit.record_for(request.param)
    return request.param, audit.read(record)


def test_every_record_joins_a_committed_row(committed: tuple[str, pd.DataFrame]) -> None:
    _, frame = committed
    assert frame[["estimate", "std_error", "truth", "covered_committed"]].notna().all().all()


def test_the_shipped_columns_are_recomputed_exactly(committed: tuple[str, pd.DataFrame]) -> None:
    slug, frame = committed
    record = audit.record_for(slug)
    # A constant arm cell comes from the data scan; every other trigger is recomputed.
    given = frame.drop(columns=SHIPPED[1:]).assign(
        trigger=np.where(frame["trigger"] == audit.CONSTANT_ARM_CELL, audit.CONSTANT_ARM_CELL, "")
    )
    again = audit.shipped(given, alpha=record.margins.alpha)
    pd.testing.assert_frame_equal(
        again[SHIPPED].reset_index(drop=True), frame[SHIPPED].reset_index(drop=True)
    )


def test_every_declared_bounded_mean_row_is_audited(committed: tuple[str, pd.DataFrame]) -> None:
    slug, frame = committed
    record, declared = audit.record_for(slug), audit.AUDITS[slug]()
    truth = frame.loc[(frame["source"] == "truth") & (frame["trigger"] != audit.CONSTANT_ARM_CELL)]
    for scenario, names in declared.truth.items():
        counts = truth.loc[truth["scenario"] == scenario].groupby("estimand").size()
        assert set(counts.index) >= set(names)
        assert (counts[list(names)] == record.replicates).all()
    cells = (
        frame.loc[frame["source"] == "property", "property"]
        + "/"
        + frame.loc[frame["source"] == "property", "cell"]
    )
    assert set(declared.cells) <= set(cells)


class TestTheRule:
    """The rule the audit applies, with witnesses that fail when one of its parts is wrong."""

    def frame(self, **columns: list[float] | list[str]) -> pd.DataFrame:
        base = {
            "estimate": [0.9],
            "std_error": [0.05],
            "truth": [0.99],
            "rejected_committed": [1.0],
            "trigger": [""],
        }
        base.update(columns)
        return pd.DataFrame(base)

    def test_the_logit_interval_moves_away_from_the_bound(self) -> None:
        # The identity interval of 0.9 with SE 0.05 is 0.802 to 0.998; the logit one is about
        # 0.752 to 0.964.  Each truth below is covered by exactly one of them.
        z = stats.norm.ppf(0.975)
        out = audit.shipped(
            self.frame(
                estimate=[0.9, 0.9],
                std_error=[0.05, 0.05],
                truth=[0.77, 0.98],
                rejected_committed=[1.0, 1.0],
                trigger=["", ""],
            )
        )
        assert [abs(0.9 - t) <= z * 0.05 for t in (0.77, 0.98)] == [False, True]
        assert out["covered_shipped"].tolist() == [1, 0]
        assert out["scale"].tolist() == ["range_logit", "range_logit"]

    def test_the_interval_is_not_symmetric(self) -> None:
        low, high = audit.logit_interval([0.9], [0.05])
        assert 0.9 - low[0] > high[0] - 0.9 > 0.0

    def test_an_estimate_on_its_bound_is_withheld(self) -> None:
        out = audit.shipped(self.frame(estimate=[1.0], truth=[0.99]))
        assert out["trigger"].iloc[0] == audit.OUT_OF_RANGE
        assert (out["covered_shipped"].iloc[0], out["rejected_shipped"].iloc[0]) == (0, 0.0)

    def test_a_constant_cell_withholds_an_estimate_inside_the_range(self) -> None:
        out = audit.shipped(self.frame(trigger=[audit.CONSTANT_ARM_CELL], truth=[0.9]))
        assert out["scale"].iloc[0] == "withheld"
        assert out["covered_shipped"].iloc[0] == 0

    def test_the_default_null_zero_is_always_rejected_inside_the_range(self) -> None:
        # 0.1 / 0.08 = 1.25 standard errors: the identity test does not reject 0.
        out = audit.shipped(self.frame(estimate=[0.1], std_error=[0.08], truth=[0.1]))
        assert out["rejected_shipped"].iloc[0] == 1.0


def _cp_lower(covered: pd.Series) -> float:
    count, total = int(covered.sum()), len(covered)
    return float(stats.beta.ppf(0.005, count, total - count + 1))


def test_the_quoted_stratified_numbers() -> None:
    """`F3-normal-reference`: 0.917 becomes 0.950, and the v2_ey1 cell 0.9208 becomes 0.9496."""
    frame = audit.read(audit.record_for("canonical-stratified-tmle"))
    by = audit.coverage(frame, "covered_committed"), audit.coverage(frame, "covered_shipped")
    row, cell = "stratified_binary/ey[1][V=2]", "interval_calibration/v2_ey1__correctly_specified"
    assert (by[0][row], by[1][row]) == (pytest.approx(0.917), pytest.approx(0.950))
    assert by[0][cell] == pytest.approx(0.92083, abs=5e-6)
    assert by[1][cell] == pytest.approx(0.94958, abs=5e-6)
    shipped = frame.loc[(frame["scenario"] + "/" + frame["estimand"]) == row, "covered_shipped"]
    assert _cp_lower(shipped) >= 0.90
    # One sample of the necessity family holds a constant treated cell in stratum 2.
    withheld = frame.loc[frame["trigger"] == audit.CONSTANT_ARM_CELL]
    assert list(zip(withheld["cell"], withheld["replicate"], strict=True)) == [
        ("v2_ate__stratified", 829)
    ]


def test_the_quoted_composite_numbers() -> None:
    """`composite-high-arm`: 0.9225 becomes 0.926, and its 99% interval still ends below 0.90."""
    frame = audit.read(audit.record_for("composite-missing-drtmle"))
    key = "three_arm_mar_outcome_and_treatment/ey[high]"
    assert audit.coverage(frame, "covered_committed")[key] == pytest.approx(0.9225)
    assert audit.coverage(frame, "covered_shipped")[key] == pytest.approx(0.92625)
    rows = frame.loc[(frame["scenario"] + "/" + frame["estimand"]) == key]
    assert _cp_lower(rows["covered_shipped"]) == pytest.approx(0.8992, abs=5e-5)
