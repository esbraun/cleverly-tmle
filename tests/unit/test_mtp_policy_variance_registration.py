"""The regime contrast the ``mtp-policy-variance`` page publishes, from the committed rows.

The declaration has the registration compute :func:`regime_contrast` from both phases'
committed rows, because no harness hook sees both.  This test recomputes it so the page's table
cannot drift from the artifacts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.studies.canonical_mtp_policy_variance import regime_contrast
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT

ARTIFACTS = ROOT / "tests" / "canonical" / "mtp_policy_variance"

#: (scenario, estimand) -> (twin coverage, control coverage, difference, lower, upper, reading),
#: as the page prints them (three decimals).
PUBLISHED = {
    ("pricing_ltmle_tau5_tail", "ey_regimen[price_down]"): (
        0.940,
        0.911,
        0.029,
        0.010,
        0.045,
        "gain in regime",
    ),
    ("pricing_ltmle_tau5_tail", "ate_regimen[price_down vs natural_prices]"): (
        0.929,
        0.918,
        0.010,
        -0.009,
        0.027,
        "no change detected",
    ),
    ("pricing_sdr_tau5_tail", "ey_regimen[price_down]"): (
        0.943,
        0.926,
        0.016,
        -0.001,
        0.032,
        "no change detected",
    ),
    ("pricing_sdr_tau5_tail", "ate_regimen[price_down vs natural_prices]"): (
        0.945,
        0.929,
        0.016,
        -0.001,
        0.031,
        "no change detected",
    ),
}


@pytest.fixture(scope="module")
def contrast() -> pd.DataFrame:
    primary = read_rows(ARTIFACTS / "replicates.parquet")
    properties = read_rows(ARTIFACTS / "property-replicates.parquet")
    return regime_contrast(primary, properties)


def test_the_published_regime_contrast_is_the_committed_one(contrast: pd.DataFrame) -> None:
    assert len(contrast) == len(PUBLISHED)
    for row in contrast.itertuples():
        twin, control, difference, low, high, reading = PUBLISHED[(row.scenario, row.estimand)]
        measured = (
            row.twin_coverage,
            row.control_coverage,
            row.difference,
            row.ci_lower,
            row.ci_upper,
        )
        for quoted, value in zip((twin, control, difference, low, high), measured, strict=True):
            assert f"{float(value):.3f}" == f"{quoted:.3f}", (
                row.scenario,
                row.estimand,
            )
        assert row.reading == reading
        assert row.twin_replicates == 1_800
        assert row.control_replicates == 6_000


def test_the_page_prints_the_table(contrast: pd.DataFrame) -> None:
    page = (
        ROOT / "docs" / "technical-reference" / "method-evidence" / "mtp-policy-variance.md"
    ).read_text(encoding="utf-8")
    for row in contrast.itertuples():
        line = (
            f"| `{row.scenario}` | `{row.estimand}` | {row.twin_coverage:.3f} | "
            f"{row.control_coverage:.3f} | {row.difference:.3f} "
            f"({row.ci_lower:.3f}, {row.ci_upper:.3f}) | {row.reading} |"
        )
        assert line in page, line


def _range(values: pd.Series) -> str:
    low, high = f"{values.min():.3f}", f"{values.max():.3f}"
    return low if low == high else f"{low} to {high}"


def test_the_policy_notes_quote_the_committed_option_rows() -> None:
    """Each licence note's SE-ratio and coverage ranges are the committed option rows' own."""
    from cleverly._support_rows import POLICY_NOTES

    published = pd.read_csv(ARTIFACTS / "properties.csv")
    option = published.loc[published["cell"].str.endswith("__option")]

    def rows(*prefixes: str) -> pd.DataFrame:
        return option.loc[option["cell"].str.startswith(prefixes)]

    expected = {
        "G39.ltmle.tau5": rows("pricing_ltmle_tau5__"),
        "G39.sdr.tau5": rows("pricing_sdr_tau5__"),
        "G39.ltmle.tau2": rows("pricing_ltmle_tau2__"),
        "G39.sdr.tau2": rows("pricing_sdr_tau2__"),
        "G10.continuous": rows("p4_ltmle__", "boundaries_ltmle_p4__"),
        "G24.mtp.in_sample": rows("p4_sdr__", "boundaries_sdr_p4__"),
    }
    for row_id, frame in expected.items():
        (note,) = POLICY_NOTES[row_id]
        assert f"SE ratio {_range(frame['se_ratio'])}" in note, (row_id, note)
        assert f"coverage {_range(frame['coverage'])}" in note, (row_id, note)
        assert "conservative" in note if (frame["se_ratio_ci_upper"] > 1.07).any() else True
