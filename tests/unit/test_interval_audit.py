"""The committed interval audits recompute exactly from the committed rows.

``tests/studies/evidence/interval_audit.py`` reads each audited study's bounded-mean rows under
the shipped logit interval.  Each audit file stores the keys, the trigger and the shipped
verdicts; the committed estimates, standard errors and truths are joined back from the study's
row files, so an audit record that drifts from its row fails here.  The numbers the roadmap and
the study pages quote are rebuilt from the same records.  The package's own ``ci`` and
``wald_test`` are checked against the audit's rule on a sample of every audit, so the audit
cannot certify an interval the package no longer ships, and the screen of every registered study
is recomputed row by row.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly._inference_status import OUT_OF_RANGE_STATUS
from cleverly.inference import make_estimate
from tests.studies.evidence import interval_audit as audit
from tests.studies.evidence.registry import registered

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


def test_the_quoted_pricing_numbers() -> None:
    """`mtp-pricing-calibration`: the five-node `price_down` levels pass their floor on logit."""
    frame = audit.read(audit.record_for("mtp-pricing"))
    for scenario, before, after, lower in (
        ("pricing_ltmle_tau5", 0.917, 0.933, 0.9100),
        ("pricing_sdr_tau5", 0.923, 0.935, 0.9123),
    ):
        key = f"{scenario}/ey_regimen[price_down]"
        assert audit.coverage(frame, "covered_committed")[key] == pytest.approx(before)
        assert audit.coverage(frame, "covered_shipped")[key] == pytest.approx(after)
        rows = frame.loc[(frame["scenario"] + "/" + frame["estimand"]) == key]
        assert _cp_lower(rows["covered_shipped"]) == pytest.approx(lower, abs=5e-5)
        assert _cp_lower(rows["covered_shipped"]) >= 0.90 > _cp_lower(rows["covered_committed"])
    # The binary-outcome calibration cells: the largest move, quoted on the study page.
    cells = frame.loc[frame["source"] == "property"]
    moves = (
        audit.coverage(cells, "covered_shipped") - audit.coverage(cells, "covered_committed")
    ).abs()
    assert len(moves) == 11
    key = "interval_calibration/up_learned_density_tau5__correctly_specified"
    assert moves.idxmax() == key
    assert audit.coverage(cells, "covered_committed")[key] == pytest.approx(0.9610)
    assert audit.coverage(cells, "covered_shipped")[key] == pytest.approx(0.9655)


def test_the_quoted_stratified_drtmle_foldwise_numbers() -> None:
    """`X8-drtmle-small-stratum`: the fold-wise row's 0.907 becomes 0.919, above its floor."""
    frame = audit.read(audit.record_for("stratified-drtmle-foldwise"))
    key = "stratified_both_correct/ey[1][V=2]"
    assert audit.coverage(frame, "covered_committed")[key] == pytest.approx(0.90725)
    assert audit.coverage(frame, "covered_shipped")[key] == pytest.approx(0.91925)
    rows = frame.loc[(frame["scenario"] + "/" + frame["estimand"]) == key]
    assert _cp_lower(rows["covered_shipped"]) == pytest.approx(0.9075, abs=5e-5)
    assert _cp_lower(rows["covered_shipped"]) >= 0.90 > _cp_lower(rows["covered_committed"])
    assert (frame["trigger"] != audit.CONSTANT_ARM_CELL).all()


# ------------------------------------------------------------------ the rule against the package


def _package_estimate(estimate: float, std_error: float, alpha: float) -> Any:
    """The package's estimate of a bounded unclustered mean with this estimate and SE."""
    curve = np.tile([1.0, -1.0], 200)
    unit = make_estimate("ey", 0.5, curve, n=400, scale="level", rule_widened=False)
    scaled = curve * (std_error / unit.std_error) if std_error > 0.0 else np.zeros(400)
    return make_estimate(
        "ey",
        estimate,
        scaled,
        n=400,
        scale="level",
        alpha=alpha,
        bounds=(0.0, 1.0),
        rule_widened=False,
    )


def _sample(frame: pd.DataFrame, size: int = 150) -> pd.DataFrame:
    """A deterministic sample of a study's audit records, every withheld record included."""
    rows = frame.loc[frame["trigger"] == ""]
    picked = rows.iloc[np.linspace(0, len(rows) - 1, min(size, len(rows))).astype(int)]
    return pd.concat([picked, frame.loc[frame["trigger"] == audit.OUT_OF_RANGE]])


def test_the_audit_rule_is_the_package_interval(committed: tuple[str, pd.DataFrame]) -> None:
    """``ci``, coverage and ``wald_test`` of the package agree with the audit on its records.

    The audit recomputes its verdicts with :func:`~interval_audit.logit_interval`.  If the
    package's interval of a bounded mean changed, this is where the two would part.
    """
    slug, frame = committed
    alpha = audit.record_for(slug).margins.alpha
    for row in _sample(frame).itertuples():
        estimate = _package_estimate(float(row.estimate), float(row.std_error), alpha)
        if row.trigger == audit.OUT_OF_RANGE:
            assert estimate.inference == OUT_OF_RANGE_STATUS
            continue
        assert estimate.inference == "influence_curve"
        assert estimate.interval_scale == "range_logit"
        low, high = estimate.ci
        expected = audit.logit_interval([row.estimate], [row.std_error], alpha=alpha)
        assert (low, high) == pytest.approx((expected[0][0], expected[1][0]), abs=1e-12)
        assert int(low <= row.truth <= high) == row.covered_shipped
        assert estimate.wald_test(0.0).pvalue < alpha
        assert bool(estimate.wald_test(row.truth).pvalue < alpha) == bool(
            audit.logit_rejects([row.estimate], [row.std_error], row.truth, alpha=alpha)[0]
        )


@pytest.mark.parametrize(
    ("estimate", "std_error"),
    [(1e-13, 0.01), (1.0 - 1e-12, 0.01), (0.3, 0.0), (0.999, 0.004), (0.002, 0.003)],
)
def test_the_audit_rule_is_the_package_interval_at_the_edges(
    estimate: float, std_error: float
) -> None:
    package = _package_estimate(estimate, std_error, 0.05)
    assert package.inference == "influence_curve"
    low, high = package.ci
    expected = audit.logit_interval([estimate], [std_error])
    assert (low, high) == pytest.approx((expected[0][0], expected[1][0]), abs=1e-12)
    if std_error == 0.0:
        # A zero SE shows the point as its interval and computes no test; no committed row
        # has one.
        assert (low, high) == (estimate, estimate)
        assert math.isnan(package.wald_test(0.25).pvalue)
        return
    for null in (0.0, 0.25, 0.5):
        assert bool(package.wald_test(null).pvalue < 0.05) == bool(
            audit.logit_rejects([estimate], [std_error], null)[0]
        )


# ------------------------------------------------------------------ the stratified scan


def test_the_stratified_scan_reproduces_the_committed_triggers() -> None:
    """Redraw a flagged sample and its neighbours without fitting; the triggers come back."""
    from tests.studies import stratified_tmle_properties as properties

    frame = audit.read(audit.record_for("canonical-stratified-tmle"))
    flagged = {
        (cell, int(replicate))
        for cell, replicate in zip(
            frame.loc[frame["trigger"] == audit.CONSTANT_ARM_CELL, "cell"],
            frame.loc[frame["trigger"] == audit.CONSTANT_ARM_CELL, "replicate"],
            strict=True,
        )
    }
    keys = [
        key
        for key in properties._payloads()
        if (key[0][0], key[0][2])
        in {("stratum_targeting_necessity", r) for r in (828, 829, 830)}
        | {("interval_calibration", r) for r in (0, 1)}
        | {("double_robustness", 0)}
    ]
    assert any(key[0][2] == 829 for key in keys)
    found = {
        (cell, replicate)
        for key in keys
        for _, _, cell, replicate in audit.stratified_property_withheld(key)
    }
    assert found == {("v2_ate__stratified", 829)}
    assert found <= flagged
    assert audit.stratified_truth_withheld("stratified_binary", 0) == []


# ------------------------------------------------------------------ the screen


@pytest.fixture(scope="module")
def screen() -> pd.DataFrame:
    return audit.read_screen()


def test_the_screen_lists_every_registered_study_once(screen: pd.DataFrame) -> None:
    assert list(screen.columns) == list(audit.SCREEN_COLUMNS)
    assert sorted(screen["study"]) == sorted(record.slug for record in registered())
    assert screen["study"].is_unique


@pytest.mark.parametrize("slug", sorted(record.slug for record in registered()))
def test_each_screen_row_recomputes_from_the_committed_rows(
    screen: pd.DataFrame, slug: str
) -> None:
    """A new cell or a regenerated study changes its row; refresh with ``--screen``."""
    committed_row = screen.loc[screen["study"] == slug].iloc[0].to_dict()
    again = audit.screen_row(audit.record_for(slug))
    assert committed_row["max_coverage_change"] == pytest.approx(
        again.pop("max_coverage_change"), abs=1e-6
    )
    committed_row.pop("max_coverage_change")
    assert committed_row == again


def test_every_flip_is_audited_or_explained(screen: pd.DataFrame) -> None:
    flipped = screen.loc[screen["flips"] != ""]
    for row in flipped.itertuples():
        if row.study in audit.AUDITS:
            declared = audit.AUDITS[row.study]()
            rows = {f"{s}/{e}" for s, names in declared.truth.items() for e in names}
            assert set(row.flips.split(";")) <= rows | set(declared.cells), row.study
        else:
            assert row.disposition == "declared" and row.reason, row.study
    explained = set(flipped["study"]) - set(audit.AUDITS)
    assert set(audit.SCREEN_REASONS) == explained
    assert set(screen.loc[screen["disposition"] == "audited", "study"]) == set(audit.AUDITS)
    assert set(screen.loc[screen["disposition"] == "rerun", "study"]) == set(audit.RERUN)


#: The first ``learned-rule-arms`` run's ``conservative_coverage`` rows (commit 8f13a9c), whose
#: ``covered`` reads the identity interval: the input of the screen's mutation control.
IDENTITY_RUN_ROWS = (
    Path(__file__).parent
    / "interval_audit_fixtures"
    / "learned-rule-arms-identity-conservative.parquet"
)


def test_the_screen_finds_the_flip_that_forced_the_learned_rule_arms_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out of :data:`~audit.RERUN`, the first run's rows flip the cell that reads 0.92.

    The cell's lower coverage limit moves from 0.9201 to 0.9173, which crosses the calibration
    band's lower end and not the coverage floor, so a screen that read the floor alone missed it.
    """
    shutil.copy(IDENTITY_RUN_ROWS, tmp_path / "property-replicates.parquet")
    record = replace(audit.record_for("learned-rule-arms"), artifacts=tmp_path)
    monkeypatch.setattr(audit, "RERUN", {})
    row = audit.screen_row(record)
    assert row["flips"].split(";") == ["conservative_coverage/missing_quadratic__declared"]
    assert row["crossings"] == row["flips"]
    assert row["disposition"] == "declared"
    # Mutation: the screen's former proxy, a crossing of the coverage floor alone, sees nothing.
    monkeypatch.setattr(
        audit,
        "_crosses",
        lambda before, after, margins: (
            (before[0] >= margins.coverage_floor) != (after[0] >= margins.coverage_floor)
        ),
    )
    mutated = audit.screen_row(record)
    assert mutated["flips"] == mutated["crossings"] == ""


@pytest.mark.parametrize("slug", ["option-stopping-point", "option-stopping-ltmle"])
def test_the_screen_margin_reproduces_the_committed_widened_coverage(slug: str) -> None:
    """On a stopping-rule study the screen's margin is the one each committed interval carried.

    Read on the identity scale, ``estimate +/- (z se + margin)`` reproduces every committed
    ``covered`` of the bounded groups the screen reads. With the margin dropped, at least one
    group's coverage moves, so the margin is not zero at the truth.
    """
    record = audit.record_for(slug)
    assert audit._stopping_rule_study(record)
    z = float(stats.norm.ppf(1.0 - record.margins.alpha / 2.0))
    moved_without = False
    widened_groups = 0
    for key, group, _ in audit._screen_groups(record, audit.bounded_heads()):
        if key.endswith(audit.WITNESS_SUFFIX):
            continue
        margin = audit._screen_margin(key, group, record.margins.alpha, stopping_rule=True)
        estimate = group["estimate"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        half = z * group["std_error"].to_numpy(dtype=float)
        committed = group["covered"].to_numpy(dtype=int)
        covered = (estimate - half - margin <= truth) & (truth <= estimate + half + margin)
        assert (covered.astype(int) == committed).all(), key
        if (margin > 0).any():
            widened_groups += 1
            plain = (estimate - half <= truth) & (truth <= estimate + half)
            moved_without |= bool((plain.astype(int) != committed).any())
    assert widened_groups > 0
    assert moved_without


def test_the_screen_margin_is_the_package_margin_on_the_logit_scale() -> None:
    """``logit_interval(margin=)`` is ``ci`` of a widened bounded mean of the package."""
    estimate = _package_estimate(0.2, 0.02, 0.05)
    curve = estimate.influence_curve + 0.004
    widened = make_estimate(
        "ey", 0.2, curve, n=400, scale="level", bounds=(0.0, 1.0), rule_widened=True
    )
    assert widened.residual_margin == pytest.approx(0.004, rel=1e-9)
    low, high = audit.logit_interval([0.2], [widened.std_error], margin=[widened.residual_margin])
    assert widened.ci == pytest.approx((low[0], high[0]), abs=1e-12)
    assert widened.ci != pytest.approx(
        tuple(x[0] for x in audit.logit_interval([0.2], [widened.std_error])), abs=1e-6
    )
