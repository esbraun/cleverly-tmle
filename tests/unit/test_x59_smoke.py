"""X59 section 4.1: the offline parts of the smoke, which need no Docker.

The identity gate (B4) runs here on the first replicates of the cheapest scenario of each study,
so a change that moves a committed row, a fold or a written ratio fails in the fast tier before
the local queue row spends an hour finding it.  The sizing step is checked on the plan's worked
example, and the smoke declaration on a table it writes.
"""

from __future__ import annotations

import argparse
import json
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
    """A deliberate mutation: an estimate off by 1e-9 fails; last-bit noise passes."""
    report = pd.DataFrame(
        {
            "scenario": ["a", "b"],
            "replicate": [0, 0],
            "estimand": ["ey[x]", "msm[(intercept)]"],
            "estimate": [0.0, 1e-9],
            "std_error": [0.0, 0.0],
            "initial_estimate": [0.0, 0.0],
            "committed_estimate": [0.5, 0.53],
            "committed_std_error": [0.01, 0.01],
            "committed_initial_estimate": [0.5, 0.53],
            "folds_match": [True, True],
            "ratios_match": [True, True],
        }
    )
    assert x59_smoke.identity_failures(report) == [
        "b r0 msm[(intercept)] estimate: |refit - committed| 1e-09 > 1.53e-12"
    ]
    # Row 50's cross-platform gap on the MSM intercept is inside the bound.
    assert x59_smoke.identity_passed(report.assign(estimate=[0.0, 1.16e-12]))
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


def _calibration(path: Path, per_worker: float, parent: float, available: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = {
        "id": "r000000000",
        "rss_start": 100.0,
        "hwm_end": 100.0 + per_worker / 2,
        "peak_growth": per_worker / 2,
        "private_after_gc": per_worker / 2,
        "per_worker": per_worker,
    }
    memory = {"total": 24_000.0, "available": available, "limit": None, "current": None}
    path.write_text(json.dumps({"groups": [group], "parent_private_mb": parent, "memory": memory}))


def test_the_memory_readings_plan_each_scenario_from_its_own_calibration(tmp_path: Path) -> None:
    leaf = "run_study__reference-results__{}.calibration.json"
    _calibration(tmp_path / "a" / leaf.format("light"), 300.0, 150.0, 22_000.0)
    _calibration(tmp_path / leaf.format("heavy"), 3_400.0, 550.0, 21_700.0)
    frame = x59_smoke.memory_readings("sdr", tmp_path, cores=12).set_index("scenario")
    assert frame.loc["light", "workers"] == 12 and frame.loc["light", "chunk"] == 12
    assert frame.loc["heavy", "workers"] == 3 and frame.loc["heavy", "chunk"] == 3
    assert frame.loc["heavy", "planned_mb"] >= 3_400.0


def _sizing_inputs(tmp_path: Path, *, memory: bool = True) -> None:
    pd.DataFrame(
        {
            "study": ["sdr", "sdr", "lmtp-parity"],
            "scenario": ["exact", "margins", "parity_tmle_mix"],
            "tier": [2, 4, parity.PARITY_TIER],
            "replicates": [None, None, 500],
            "seconds": [40.0, 20.0, 2.0],
        }
    ).to_csv(tmp_path / "timing-sdr.csv", index=False)
    if memory:
        pd.DataFrame(
            {
                "study": ["sdr", "sdr", "lmtp-parity"],
                "scenario": ["exact", "margins", "parity_tmle_mix"],
                "workers": [3, 6, 12],
                "chunk": [3, 6, 12],
            }
        ).to_csv(tmp_path / "memory-sdr.csv", index=False)


def test_the_sizing_step_reads_the_timing_and_memory_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _sizing_inputs(tmp_path)
    arguments = ["sizing", "--timings", str(tmp_path), "--memory", str(tmp_path)]
    assert x59_smoke.main(arguments) == 0
    printed = capsys.readouterr().out
    expected = parity.reference_sizes(x59_smoke.sizing_costs(tmp_path, tmp_path), tier1_hours=0.0)
    assert f"tier 2 at {expected.tier2}, tier 4 at {expected.tier4}" in printed
    assert "lmtp-parity:parity_tmle_mix: 500 (tier 3" in printed
    assert "sdr:exact" in printed and "3 workers, chunk 3" in printed


def test_the_sizing_step_refuses_a_timed_scenario_without_a_memory_plan(tmp_path: Path) -> None:
    _sizing_inputs(tmp_path, memory=False)
    pd.DataFrame(columns=["study", "scenario", "workers", "chunk"]).to_csv(
        tmp_path / "memory-other.csv", index=False
    )
    with pytest.raises(SystemExit, match="no memory plan for sdr:exact"):
        x59_smoke.sizing_costs(tmp_path, tmp_path)


@pytest.mark.parametrize("study", sorted(x59_smoke.STUDIES))
def test_every_paired_scenario_of_every_record_has_a_tier(study: str) -> None:
    """``timing`` and ``sizing`` read a tier for each of the five records' paired scenarios."""
    from tests.canonical import regenerate

    module = x59_smoke._module(study)
    here = x59_smoke.CANONICAL / x59_smoke.STUDIES[study][1]
    base = regenerate.compared_record(
        module, argparse.Namespace(reference_prefix=None, output=here), here
    )
    tiers = {
        name: x59_smoke.reference_tier(study, name) for name in base.reported(str(base.reference))
    }
    assert tiers and set(tiers.values()) <= {2, 4, parity.PARITY_TIER}
    parity_record = study in x59_smoke.PARITY_STUDIES
    assert (set(tiers.values()) == {parity.PARITY_TIER}) == parity_record


@pytest.mark.parametrize(
    "step", ["run", "mutations", "subject-mutation", "qbound", "memory", "timing"]
)
def test_every_record_reaches_every_smoke_step(step: str) -> None:
    extra = ["--container-mb", "1000"] if step == "memory" else []
    for study in x59_smoke.STUDIES:
        parsed = x59_smoke.build_parser().parse_args([step, study, "--out", "x", *extra])
        assert parsed.study == study


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
