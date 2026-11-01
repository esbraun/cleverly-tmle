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
    assert smoke.tau("G1") == parity.G1_TOLERANCE
    assert "G1" not in smoke.tolerance
    with pytest.raises(ValueError, match="not declared yet"):
        declaration.tau("G-P")


def test_g1_is_fixed_and_no_declared_tolerance_passes_the_ceiling() -> None:
    with pytest.raises(ValueError, match="take no declared tolerance"):
        parity.ParityDeclaration(cells=(), tolerance={"G-P": 1e-6, "G-C": 1e-6, "G1": 1e-3})
    with pytest.raises(ValueError, match="outside"):
        parity.ParityDeclaration(cells=(), tolerance={"G-P": 2 * parity.TOLERANCE_CEILING})
    assert parity.declared_tolerance(5e-7) == pytest.approx(5e-6)
    with pytest.raises(ValueError, match="the smoke fails"):
        parity.declared_tolerance(2 * parity.TOLERANCE_CEILING / parity.SMOKE_SCALE)


def test_the_smoke_merge_keeps_g1_out_of_the_declaration() -> None:
    merged = x59_smoke._declared_only({"G-P": None, "G-C": 3e-7}, {"G-P": 2e-7})
    assert merged == {"G-P": 2e-7, "G-C": 3e-7}
    assert parity.ParityDeclaration(cells=(), tolerance=merged).tau("G1") == parity.G1_TOLERANCE


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


@pytest.mark.filterwarnings("ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning")
def test_the_subject_mutation_moves_g_p_against_an_unmutated_reference(tmp_path: Path) -> None:
    """The committed subject row stands in for an R row that equals it: the mutation must show."""
    from tests.studies import canonical_long_horizon_ltmle as s1
    from tests.studies.evidence.manifest import read_rows, write_rows

    committed = read_rows(s1.STUDY.artifact("replicates.parquet"))
    wald = set(s1.REFERENCE_SCENARIOS["binary_k3"])
    held = committed.loc[
        (committed["scenario"] == "binary_k3")
        & (committed["replicate"] == 0)
        & committed["estimand"].isin(wald)
    ]
    (tmp_path / "none").mkdir()
    write_rows(held.assign(implementation="ltmle"), tmp_path / "none" / "replicates.parquet")
    (tmp_path / "declared-tolerances.json").write_text('{"G-P": 1e-06, "G-C": 1e-06}\n')
    assert x59_smoke.main(["subject-mutation", "ltmle", "--out", str(tmp_path)]) == 0


def test_the_follower_cases_name_declared_plans() -> None:
    for _, study, name, plan in x59_smoke.FOLLOWER_CASES:
        scenario = next(s for s in x59_smoke._module(study).SCENARIOS if s.name == name)
        assert plan in scenario.plans and scenario.n_folds == 1


def test_the_follower_check_fails_a_count_or_a_coefficient_off(tmp_path: Path) -> None:
    coefficients = np.array([0.25, -1.5, 0.75])
    expected = {"case": {"units": {1: 40, 2: 30}, "coefficients": coefficients, "K": 2}}

    def log(units: tuple[int, int], final: np.ndarray) -> None:
        pd.DataFrame(
            {
                "node": [1, 2],
                "family": ["quasibinomial"] * 2,
                "rows": list(units),
                "units": list(units),
                "coefficients": ["0", ";".join(f"{value:.17g}" for value in final)],
            }
        ).to_csv(tmp_path / "followers-case.csv", index=False)

    log((40, 30), coefficients)
    assert all(x59_smoke.follower_checks(tmp_path, expected).values())
    log((40, 29), coefficients)
    assert not all(x59_smoke.follower_checks(tmp_path, expected).values())
    log((40, 30), coefficients + np.array([0.0, 1e-6, 0.0]))
    assert not all(x59_smoke.follower_checks(tmp_path, expected).values())


def test_package_drift_names_every_shared_package_but_lmtp(tmp_path: Path) -> None:
    for version, collapse in (("1.5.4", "2.1.2"), ("1.5.5", "2.0.19")):
        pd.DataFrame(
            {
                "Package": ["collapse", "lmtp", "origami"][: 2 + (version == "1.5.5")],
                "Version": [collapse, version, "1.0.7"][: 2 + (version == "1.5.5")],
            }
        ).to_csv(tmp_path / f"packages-{version}.csv", index=False)
    assert x59_smoke.package_drift(tmp_path) == ["collapse 2.1.2 / 2.0.19"]
