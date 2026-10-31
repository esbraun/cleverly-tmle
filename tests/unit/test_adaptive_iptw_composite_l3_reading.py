"""The reading of ``bias/ey_low__composite_l3`` of ``adaptive-iptw``, rebuilt from its rows.

The declared run reads a standardized bias of -0.153 on 400 draws.  The diagnostic of
``tests/diagnostics/adaptive_iptw_composite_l3`` runs the same cell on 4,000 fresh draws.  This
test rebuilds the diagnostic's reading from its committed rows and checks the declared run
against it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.diagnostics.adaptive_iptw_composite_l3 import run
from tests.studies import adaptive_iptw_properties as properties
from tests.studies.evidence.registry import ROOT

ROWS = pd.read_csv(run.HERE / "rows.csv.gz")
CELL = "ey_low__composite_l3"


@pytest.fixture(scope="module")
def declared() -> pd.Series:
    path = ROOT / "tests" / "canonical" / "adaptive_iptw" / "properties.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    return frame[(frame["property"] == "bias") & (frame["cell"] == CELL)].iloc[0]


class TestFreshDraws:
    def test_the_probe_has_the_declared_draw_count(self) -> None:
        assert len(ROWS) == run.PROBE_DRAWS
        assert sorted(ROWS["replicate"]) == list(range(run.PROBE_DRAWS))

    def test_no_probe_seed_is_a_study_seed(self) -> None:
        # The probe labels differ from the study's, so the two seed sets must not meet.
        used = {
            seed
            for replicate in range(study_draws())
            for seed in (
                properties._sample_seed(run.SCENE, replicate),
                properties._fit_seed(run.SCENE, replicate),
            )
        }
        probe = {seed for replicate in range(run.PROBE_DRAWS) for seed in run.seeds(replicate)}
        assert not used & probe

    def test_the_first_probe_draw_refits_to_its_committed_row(self) -> None:
        refit = run.fit_draw(0)
        first = ROWS.iloc[0]
        assert refit["estimate"] == pytest.approx(first["estimate"], rel=1e-12)
        assert refit["truth"] == first["truth"]


def study_draws() -> int:
    return run.study.SCENES[run.SCENE].replicates


class TestTheReading:
    def test_the_interval_contains_zero_and_sits_inside_the_margin(self) -> None:
        values = run.reading(ROWS)
        margin = run.study.STUDY.margins.standardized_bias
        assert values["interval_lower"] < 0.0 < values["interval_upper"]
        assert -margin < values["interval_lower"]
        assert values["interval_upper"] < margin

    def test_the_declared_bias_is_an_excursion_of_a_small_cell(self, declared: pd.Series) -> None:
        values = run.reading(ROWS)
        assert declared["standardized_bias"] < -0.1
        # The 99% interval of the 4,000 draws excludes the declared 400-draw reading.
        assert declared["standardized_bias"] < values["interval_lower"]

    def test_the_probe_is_small_where_the_declared_run_is_large(self) -> None:
        # |std bias| < 0.06: below the 0.05 stop rule up to the Monte Carlo error of 4,000 draws.
        assert abs(run.reading(ROWS)["standardized_bias"]) < 0.06
