"""Mutation controls for the declared reading of ``stratified-drtmle-foldwise``'s ladder.

Each branch of :func:`slope_reading`, :func:`control_reading`, :func:`se_ratio_reading` and
:func:`x8_acceptance` in :mod:`tests.studies.stratified_drtmle_foldwise_ladder` is pinned on a
hand-built interval or summary, at its boundaries.  Synthetic replication rows with a known bias
law then run through :func:`summarize_properties`, so a reading that is computed but not wired
to the row it names fails here too.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies import stratified_drtmle_foldwise as primary
from tests.studies import stratified_drtmle_foldwise_ladder as ladder
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import Margins

MARGINS = Margins()


# ---------------------------------------------------------------------------- slope reading
@pytest.mark.parametrize(
    ("low", "high", "reading"),
    [
        # The escape hatch: a half-width above 0.25 is unresolved wherever the interval sits.
        (-1.61, -1.0, ladder.NOT_RESOLVED),
        (-0.4, 0.2, ladder.NOT_RESOLVED),
        (float("nan"), -1.0, ladder.NOT_RESOLVED),
        # At the target exactly the interval is read.
        (-1.5, -1.0, ladder.CONSISTENT),
        (-0.3, 0.0, ladder.NOT_CONTRACTING),
        (-0.2, 0.1, ladder.NOT_CONTRACTING),
        (-1.2, -0.7, ladder.CONSISTENT),
        (-0.9, -0.5, ladder.CONTRACTING),
        (-0.8, -0.4, ladder.CONTRACTING),
        (-0.5, -0.01, ladder.CONTRACTING),
    ],
)
def test_the_slope_reading_follows_the_declared_table(
    low: float, high: float, reading: str
) -> None:
    assert ladder.slope_reading(Interval(low, high)) == reading


def test_the_escape_hatch_is_read_before_the_position_of_the_interval() -> None:
    # A wide interval wholly below -1/2 would read second order without the hatch.
    wide = Interval(-2.0, -1.0)
    assert wide.high < ladder.FIRST_ORDER_SLOPE
    assert ladder.slope_reading(wide) == ladder.NOT_RESOLVED
    assert ladder.slope_reading(Interval(-1.5, -1.0)) == ladder.CONSISTENT


@pytest.mark.parametrize(
    ("high", "reading"),
    [
        (0.3, ladder.FAILS_TO_CONTRACT),
        (-0.5, ladder.FAILS_TO_CONTRACT),
        (-0.51, ladder.CONTROL_CONTRACTS),
    ],
)
def test_the_control_must_fail_to_contract(high: float, reading: str) -> None:
    assert ladder.control_reading(Interval(high - 0.4, high)) == reading


@pytest.mark.parametrize(
    ("se_ratio", "reading"),
    [
        ((0.9, 1.1), ladder.CLEAN),
        ((0.8, 1.2), ladder.CLEAN),
        # The finite-sample rule no longer rescues a rung: an SE-ratio red is a defect suspect.
        ((0.85, 1.25), ladder.DEFECT_SUSPECT),
        ((0.79, 1.0), ladder.DEFECT_SUSPECT),
        ((0.4, 0.6), ladder.DEFECT_SUSPECT),
    ],
)
def test_a_rung_is_read_on_its_se_ratio_alone(se_ratio: tuple[float, float], reading: str) -> None:
    assert ladder.se_ratio_reading(Interval(*se_ratio), MARGINS) == reading


@pytest.mark.parametrize(
    ("bias", "se_ratio", "coverage"),
    [
        (0.0, 1.0, 0.95),
        (0.5, 1.0, 0.9209),
        (-0.5, 1.0, 0.9209),
        (0.0, 0.9, 0.9223),
        (1.0, 1.1, 0.8754),
    ],
)
def test_the_implied_coverage_is_the_normal_wald_coverage(
    bias: float, se_ratio: float, coverage: float
) -> None:
    assert ladder.implied_coverage(bias, se_ratio) == pytest.approx(coverage, abs=1e-4)
    # Its formula, from an independent normal integral.
    z = norm.ppf(0.975)
    grid = np.linspace(-z * se_ratio, z * se_ratio, 200_001)
    mass = np.trapezoid(norm.pdf(grid - bias), grid)
    assert ladder.implied_coverage(bias, se_ratio) == pytest.approx(mass, abs=1e-8)


# ------------------------------------------------------------------------- the acceptance
def summary_with(
    *, slopes: tuple[str, str], controls: tuple[str, str], level: str = ladder.CLEAN
) -> pd.DataFrame:
    """A summary carrying only the readings :func:`x8_acceptance` reads."""
    rows = []
    for label in ladder.LABELS:
        read = label in ladder.READ_LABELS
        index = ladder.READ_LABELS.index(label) if read else None
        for regime in ladder.REGIMES:
            for size in ladder.SIZES:
                reading = level if regime == ladder.POSITIVE else ladder.NOT_READ
                rows.append((ladder.level_cell(label, regime, size), reading))
            if index is None:
                reading = ladder.NOT_READ
            else:
                reading = (slopes if regime == ladder.POSITIVE else controls)[index]
            rows.append((ladder.rate_cell(label, regime), reading))
    frame = pd.DataFrame(rows, columns=["cell", "reading"])
    frame.insert(0, "property", ladder.FAMILY)
    return frame


FAILS = (ladder.FAILS_TO_CONTRACT, ladder.FAILS_TO_CONTRACT)
SECOND_ORDER = (ladder.CONSISTENT, ladder.CONSISTENT)


@pytest.mark.parametrize(
    ("slopes", "controls", "level", "outcome"),
    [
        (SECOND_ORDER, FAILS, ladder.CLEAN, ladder.ACCEPTED),
        (SECOND_ORDER, FAILS, ladder.DEFECT_SUSPECT, ladder.DEFECT),
        ((ladder.CONSISTENT, ladder.CONTRACTING), FAILS, ladder.CLEAN, ladder.DEFECT),
        ((ladder.NOT_CONTRACTING, ladder.CONSISTENT), FAILS, ladder.CLEAN, ladder.DEFECT),
        ((ladder.CONSISTENT, ladder.NOT_RESOLVED), FAILS, ladder.CLEAN, ladder.DEFECT),
        (
            SECOND_ORDER,
            (ladder.FAILS_TO_CONTRACT, ladder.CONTROL_CONTRACTS),
            ladder.CLEAN,
            ladder.DESIGN_FAILURE,
        ),
        # A contracting control outranks every other finding.
        (
            (ladder.NOT_RESOLVED, ladder.NOT_RESOLVED),
            (ladder.CONTROL_CONTRACTS, ladder.FAILS_TO_CONTRACT),
            ladder.DEFECT_SUSPECT,
            ladder.DESIGN_FAILURE,
        ),
    ],
)
def test_the_x8_acceptance_needs_every_declared_condition(
    slopes: tuple[str, str], controls: tuple[str, str], level: str, outcome: str
) -> None:
    assert ladder.x8_acceptance(summary_with(slopes=slopes, controls=controls, level=level)) == (
        outcome
    )


def test_one_defect_suspect_level_row_blocks_the_acceptance() -> None:
    summary = summary_with(slopes=SECOND_ORDER, controls=FAILS)
    cell = ladder.level_cell("v2_ate", ladder.POSITIVE, 4_000)
    summary.loc[summary["cell"] == cell, "reading"] = ladder.DEFECT_SUSPECT
    assert ladder.x8_acceptance(summary) == ladder.DEFECT


# ------------------------------------------------------------------- end to end, synthetic
#: Replications per rung in the synthetic runs, which is also the budget the level rows are
#: read at.
SYNTHETIC_REPLICATES = 400
#: The standardized bias at n = 2,000 of a label whose bias decays as 1/n.
Z_SECOND_ORDER = 2.4

Bias = Callable[[str, str, int], float]


def synthetic_rows(
    bias: Bias, *, se_scale: Callable[[str, str, int], float] | None = None
) -> pd.DataFrame:
    """Replication rows with unit error SD and the given standardized bias per cell.

    ``covered`` holds on 19 of every 20 replications, so each coverage is 0.95 exactly.
    """
    rng = np.random.default_rng(20261112)
    rows = []
    for regime in ladder.REGIMES:
        for size in ladder.SIZES:
            for label in ladder.LABELS:
                noise = rng.standard_normal(SYNTHETIC_REPLICATES)
                noise = (noise - noise.mean()) / noise.std(ddof=1)
                errors = bias(label, regime, size) + noise
                scale = 1.0 if se_scale is None else se_scale(label, regime, size)
                for r, error in enumerate(errors):
                    rows.append(
                        {
                            "property": ladder.FAMILY,
                            "cell": ladder.level_cell(label, regime, size),
                            "role": "control" if regime == ladder.CONTROL else "positive",
                            "replicate": r,
                            "n": size,
                            "requested_replicates": SYNTHETIC_REPLICATES,
                            "failed_replicates": 0,
                            "truth": 0.0,
                            "estimate": float(error),
                            "std_error": scale,
                            "covered": int(r % 20 != 0),
                            "rejected": 0,
                        }
                    )
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def second_order(z: float = Z_SECOND_ORDER) -> Callable[[int], float]:
    return lambda size: z * 2_000 / size


def first_order(z: float = Z_SECOND_ORDER) -> Callable[[int], float]:
    return lambda size: z * np.sqrt(2_000 / size)


def constant(z: float = 0.6) -> Callable[[int], float]:
    return lambda size: z


@pytest.fixture
def cheap_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The record with fewer bootstrap draws, so a synthetic summary runs in seconds."""
    margins = dataclasses.replace(primary.STUDY.margins, bootstrap_replicates=1_000)
    monkeypatch.setattr(primary, "STUDY", dataclasses.replace(primary.STUDY, margins=margins))


def law(
    positive: dict[str, Callable[[int], float]], control: dict[str, Callable[[int], float]]
) -> Bias:
    def bias(label: str, regime: str, size: int) -> float:
        laws = positive if regime == ladder.POSITIVE else control
        return laws.get(label, constant(0.0))(size)

    return bias


def readings(summary: pd.DataFrame) -> dict[str, str]:
    own = summary.loc[summary["property"] == ladder.FAMILY]
    return dict(zip(own["cell"], own["reading"], strict=True))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return ladder.summarize_properties(rows, verdict_replicates=SYNTHETIC_REPLICATES)


@pytest.mark.usefixtures("cheap_bootstrap")
def test_a_second_order_ladder_with_a_flat_control_is_accepted() -> None:
    rows = synthetic_rows(
        law(
            {label: second_order() for label in ladder.READ_LABELS},
            {label: constant() for label in ladder.READ_LABELS},
        )
    )
    summary = summarize(rows)
    seen = readings(summary)
    for label in ladder.READ_LABELS:
        assert seen[ladder.rate_cell(label, ladder.POSITIVE)] == ladder.CONSISTENT
        assert seen[ladder.rate_cell(label, ladder.CONTROL)] == ladder.FAILS_TO_CONTRACT
    for label in ("v0_ate", "v2_ate"):
        assert seen[ladder.rate_cell(label, ladder.POSITIVE)] == ladder.NOT_READ
    for size in ladder.SIZES:
        assert seen[ladder.level_cell("marginal_ate", ladder.POSITIVE, size)] == ladder.CLEAN
        assert seen[ladder.level_cell("marginal_ate", ladder.CONTROL, size)] == ladder.NOT_READ
    assert ladder.x8_acceptance(summary) == ladder.ACCEPTED


@pytest.mark.usefixtures("cheap_bootstrap")
def test_a_first_order_or_flat_slope_is_not_accepted_and_a_contracting_control_fails_design() -> (
    None
):
    rows = synthetic_rows(
        law(
            {"marginal_ate": first_order(), "v1_ate": constant()},
            {"marginal_ate": constant(), "v1_ate": second_order()},
        )
    )
    summary = summarize(rows)
    seen = readings(summary)
    assert seen[ladder.rate_cell("marginal_ate", ladder.POSITIVE)] == ladder.CONTRACTING
    assert seen[ladder.rate_cell("v1_ate", ladder.POSITIVE)] == ladder.NOT_CONTRACTING
    assert seen[ladder.rate_cell("v1_ate", ladder.CONTROL)] == ladder.CONTROL_CONTRACTS
    assert ladder.x8_acceptance(summary) == ladder.DESIGN_FAILURE


@pytest.mark.usefixtures("cheap_bootstrap")
def test_a_slope_too_noisy_to_resolve_reads_not_resolved_and_blocks_the_acceptance() -> None:
    rows = synthetic_rows(
        law(
            {"marginal_ate": second_order(), "v1_ate": second_order(0.12)},
            {label: constant() for label in ladder.READ_LABELS},
        )
    )
    summary = summarize(rows)
    seen = readings(summary)
    rate = summary.set_index("cell").loc[ladder.rate_cell("v1_ate", ladder.POSITIVE)]
    assert rate["half_width"] > ladder.RESOLUTION_TARGET
    assert seen[ladder.rate_cell("v1_ate", ladder.POSITIVE)] == ladder.NOT_RESOLVED
    assert seen[ladder.rate_cell("marginal_ate", ladder.POSITIVE)] == ladder.CONSISTENT
    assert ladder.x8_acceptance(summary) == ladder.DEFECT


@pytest.mark.usefixtures("cheap_bootstrap")
def test_an_understated_standard_error_on_one_rung_is_a_defect_suspect() -> None:
    def se_scale(label: str, regime: str, size: int) -> float:
        return 0.5 if (label, regime, size) == ("v0_ate", ladder.POSITIVE, 4_000) else 1.0

    rows = synthetic_rows(
        law(
            {label: second_order() for label in ladder.READ_LABELS},
            {label: constant() for label in ladder.READ_LABELS},
        ),
        se_scale=se_scale,
    )
    summary = summarize(rows)
    seen = readings(summary)
    assert seen[ladder.level_cell("v0_ate", ladder.POSITIVE, 4_000)] == ladder.DEFECT_SUSPECT
    assert seen[ladder.level_cell("v0_ate", ladder.POSITIVE, 2_000)] == ladder.CLEAN
    assert ladder.x8_acceptance(summary) == ladder.DEFECT


@pytest.mark.usefixtures("cheap_bootstrap")
def test_a_rung_whose_coverage_alone_is_low_stays_clean_and_publishes_its_implied_coverage() -> (
    None
):
    """Coverage is published, not gated: a rung at 0.85 coverage with a clean SE ratio passes."""
    rows = synthetic_rows(
        law(
            {label: second_order() for label in ladder.READ_LABELS},
            {label: constant() for label in ladder.READ_LABELS},
        )
    )
    cell = ladder.level_cell("marginal_ate", ladder.POSITIVE, 2_000)
    mask = rows["cell"] == cell
    rows.loc[mask, "covered"] = (rows.loc[mask, "replicate"] % 20 >= 3).astype(int)
    summary = summarize(rows)
    row = summary.set_index("cell").loc[cell]
    assert row["coverage"] == pytest.approx(0.85)
    assert row["coverage_ci_upper"] < MARGINS.coverage_floor
    assert row["reading"] == ladder.CLEAN and bool(row["passed"])
    expected = ladder.implied_coverage(float(row["standardized_bias"]), float(row["se_ratio"]))
    assert row["implied_coverage"] == pytest.approx(expected)
    assert row["implied_coverage"] < 0.5  # the synthetic bias is 2.4 SD at n = 2,000
    levels = summary.loc[~summary["cell"].str.contains("__rate_")]
    assert levels["implied_coverage"].notna().all()
    assert ladder.x8_acceptance(summary) == ladder.ACCEPTED
