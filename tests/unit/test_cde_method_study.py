"""Focused contracts for the controlled direct-effect implementation study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_cde as cde
from tests.studies import canonical_cde_tmle as primary
from tests.studies import cde_tmle_properties as properties
from tests.studies.evidence.registry import ROOT


def test_each_primary_scenario_selects_its_declared_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = primary.fit_cleverly

    def counted(frame: pd.DataFrame) -> object:
        nonlocal calls
        calls += 1
        return original(frame)

    monkeypatch.setattr(primary, "fit_cleverly", counted)
    samples, _, rows = primary.draw_and_fit(replicates=1, n=500, n_jobs=1)
    assert calls == 1
    effects = rows.loc[rows["estimand"] == "ate"].set_index("scenario")
    assert effects.loc[primary.SCENARIOS[0], "truth"] == pytest.approx(-0.1875)
    assert effects.loc[primary.SCENARIOS[1], "truth"] == pytest.approx(0.3125)
    assert len(samples) == 500
    assert "scenario" not in samples
    first, _ = primary.draw_scenario(primary.SCENARIOS[0], 500, 0)
    second, _ = primary.draw_scenario(primary.SCENARIOS[1], 500, 0)
    pd.testing.assert_frame_equal(first, second)


def test_the_r_matrix_columns_follow_tmle_order_and_duplicate_mar_observation() -> None:
    sample = primary.draw_and_fit(replicates=1, n=200, n_jobs=1)[0]
    law = cde.DiscreteLaw()
    w = np.rint(sample["W"].to_numpy()).astype(int)
    assert sample[["pzn_a0", "pzn_a1"]].to_numpy() == pytest.approx(law.qz[w])
    runner = (ROOT / "tests/canonical/tmle_cde/run_study.R").read_text(encoding="utf-8")
    assert "cbind(frame$pzn_a0, frame$pzn_a1)" in runner
    assert "p_delta1 <- cbind(frame$pin_a0, frame$pin_a1, frame$pin_a0, frame$pin_a1)" in runner
    assert "run_z <- if (level == 0L) frame$Z else 1L - frame$Z" in runner
    assert "q_z0 <- if (level == 0L) original_z0 else original_z1" in runner
    assert "fit[[1L]]" in runner


def test_all_declared_nuisances_leave_the_probability_bounds_inactive() -> None:
    arrays = (
        cde.G,
        cde.QZ,
        cde.PI,
        cde.QBAR,
        properties.WRONG_G,
        properties.WRONG_QZ,
        properties.WRONG_PI,
        properties.WRONG_QBAR,
    )
    for values in arrays:
        assert np.min(values) > primary.NUISANCE_BOUND
        assert np.max(values) < 1.0 - primary.NUISANCE_BOUND


def test_each_single_mechanism_control_has_a_nonzero_exact_witness() -> None:
    configurations = ("treatment_wrong", "intermediate_wrong", "observation_wrong")
    scale = math.sqrt(properties.ROBUSTNESS_N / cde.N)
    separations: dict[tuple[str, int], float] = {}
    for configuration in configurations:
        result = properties._fit(cde.frame(), configuration, cde.PROBS)
        for level in cde.LEVELS:
            estimate = result[float(level)][properties.TARGET]
            bias = abs(float(estimate.psi) - cde.TRUTH[level][properties.TARGET])
            separations[(configuration, level)] = bias / float(estimate.std_error) * scale
            assert bias > 1e-3
    assert min(separations.values()) > properties.EXACT_SEPARATION_FLOOR, separations


def test_native_r_second_result_defect_is_frozen_and_the_adapter_removes_it() -> None:
    fixture = pd.read_csv(ROOT / "tests/canonical/tmle_cde/native-result2-defect.csv").set_index(
        "estimand"
    )
    native = fixture.loc["ate"]
    assert native["truth"] == pytest.approx(0.3125)
    assert abs(native["initial_estimate"] - native["truth"]) < 0.01
    assert abs(native["estimate"] - native["truth"]) > 0.25
    assert not bool(native["covered"])

    published = pd.read_csv(ROOT / "tests/canonical/tmle_cde/replicates.csv.gz")
    adapted = published[
        (published["implementation"] == primary.STUDY.reference)
        & (published["scenario"] == primary.SCENARIOS[1])
        & (published["replicate"] == 0)
    ].set_index("estimand")
    subject = published[
        (published["implementation"] == primary.STUDY.implementation)
        & (published["scenario"] == primary.SCENARIOS[1])
        & (published["replicate"] == 0)
    ].set_index("estimand")
    assert adapted["estimate"].to_numpy() == pytest.approx(
        subject.loc[adapted.index, "estimate"].to_numpy(), abs=1e-8
    )
    probe = (ROOT / "tests/canonical/tmle_cde/probe_native_result2.R").read_text(encoding="utf-8")
    assert "fit[[2L]]" in probe
    assert "Q = q_z0" in probe and "Q.Z1 = q_z1" in probe


def test_native_defect_fixture_has_a_registered_reproduction_hook(tmp_path: Path) -> None:
    from tests.canonical.tmle_cde.regenerate import REFERENCE

    class FakeReference:
        def __init__(self) -> None:
            self.call: dict[str, object] | None = None

        def run(
            self,
            here: object,
            samples: object,
            truths: object,
            output: object,
            **options: object,
        ) -> None:
            self.call = {"runner": options["runner"], "samples": samples, "truths": truths}
            pd.DataFrame({"probe": [1]}).to_csv(output, index=False)

    fake = FakeReference()
    samples = tmp_path / "samples.csv.gz"
    truths_path = tmp_path / "truth.csv"
    frames = primary.reference_artifacts(
        reference=fake,
        here=tmp_path,
        samples=samples,
        truths_path=truths_path,
        output=tmp_path,
        cores=1,
    )
    assert fake.call == {
        "runner": "tmle_cde/probe_native_result2.R",
        "samples": samples,
        "truths": truths_path,
    }
    assert set(frames) == set(primary.STUDY.extra_artifacts)
    assert "tmle_cde/probe_native_result2.R" in REFERENCE.extra_files

    manifest = json.loads(primary.STUDY.artifact("manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["scenario_seed_owners"] == {
        primary.SCENARIOS[1]: primary.SCENARIOS[0]
    }
    assert "native-result2-defect.csv" in manifest["sha256"]
    assert "tests/canonical/tmle_cde/probe_native_result2.R" in manifest["reference_sha256"]
