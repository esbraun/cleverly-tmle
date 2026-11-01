"""X59 section 4.1: the offline parts of the smoke, which need no Docker.

The identity gate (B4) runs here on the first replicates of the cheapest scenario of each study,
so a change that moves a committed row, a fold or a written ratio fails in the fast tier before
the local queue row spends an hour finding it.  The sizing step is checked on the plan's worked
example, and the smoke declaration on a table it writes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical import x59_smoke
from tests.studies.evidence import parity
from tests.unit import _x59_reference_only_support as synthetic


@pytest.mark.filterwarnings("ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning")
@pytest.mark.parametrize(
    ("study", "scenario"),
    [("ltmle", "binary_k3"), ("sdr", "sdr_binary_k3"), ("categorical", "categorical_k5_switch")],
)
def test_the_identity_gate_refits_the_committed_rows(study: str, scenario: str) -> None:
    report = x59_smoke.identity_report(study, replicates=1, names=[scenario])
    assert not report.empty
    assert x59_smoke.identity_passed(report), report


def test_the_identity_gate_fails_on_a_moved_row() -> None:
    """A deliberate mutation: one estimate off by 1e-9 fails the 1e-12 gate."""
    report = pd.DataFrame(
        {
            "estimate": [0.0, 1e-9],
            "std_error": [0.0, 0.0],
            "initial_estimate": [0.0, 0.0],
            "folds_match": [True, True],
            "ratios_match": [True, True],
        }
    )
    assert not x59_smoke.identity_passed(report)
    assert x59_smoke.identity_passed(report.assign(estimate=0.0))
    assert not x59_smoke.identity_passed(report.assign(estimate=0.0, ratios_match=False))


def test_the_learner_fixture_carries_cleverly_s_predictions() -> None:
    inputs, expected = x59_smoke.learner_cases()
    assert set(inputs["case"]) == {"horizon", "fractional", "linear"}
    for case, block in inputs.groupby("case"):
        fitted = block.loc[block["role"] == "fit"]
        assert 0 < len(fitted) < (block["role"] == "predict").sum(), case
        assert np.isfinite(expected.loc[expected["case"] == case, "python"]).all()
    fractional = inputs.loc[(inputs["case"] == "fractional") & (inputs["role"] == "fit"), "target"]
    assert ((fractional > 0) & (fractional < 1)).all() and fractional.nunique() > 2


def test_a_smoke_table_is_written_at_the_floor_and_keeps_its_maxima() -> None:
    declaration = parity.ParityDeclaration(cells=synthetic.PARITY.cells)
    smoke = parity.smoke_declaration(declaration)
    assert smoke.tolerance["G-P"] == parity.PARITY_FLOOR
    assert smoke.tolerance["G1"] == parity.G1_TOLERANCE
    with pytest.raises(ValueError, match="not declared yet"):
        declaration.tau("G-P")


def test_the_sizing_step_reads_the_timing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pd.DataFrame(
        {"study": ["ltmle", "ltmle"], "scenario": ["exact", "margins"], "tier": [2, 4]}
    ).assign(cpu_hours=[0.91, 0.99]).to_csv(tmp_path / "timing-ltmle.csv", index=False)
    assert (
        x59_smoke.main(
            [
                "sizing",
                "--timings",
                str(tmp_path),
                "--workers",
                "12",
                "--tier1",
                "30",
                "--tier3",
                "17",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "tier 2 at 100, tier 4 at 100" in printed
    assert "ltmle:exact: 100" in printed
