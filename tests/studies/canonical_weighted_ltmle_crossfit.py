"""Cross-fitted weighted end-of-study longitudinal TMLE evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import weighted_longitudinal_common as common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.weighted_longitudinal_properties_common import property_cells

LMTP_VERSION = "1.5.4"
IFE_VERSION = "0.2.3"
IFE_SHA256 = "b6be1e9ba514db95118e425d2f78deabb2c9f745f44f35a301ff9b5f266d5ed2"
SEED = 20262000
PRIMARY_REPLICATES = common.CROSSFIT_PRIMARY_REPLICATES
PRIMARY_N = common.PRIMARY_N

STUDY = StudyRecord(
    name="cross-fitted weighted end-of-study longitudinal TMLE",
    slug="weighted-ltmle-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "weighted_lmtp_ltmle",
    document=(
        "docs/technical-reference/method-evidence/"
        "cross-fitted-weighted-end-of-study-longitudinal-tmle.md"
    ),
    anchor="cross-fitted-weighted-end-of-study-longitudinal-tmle",
    scenarios={common.SCENARIO: common.ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-weighted-ltmle",
    reference="lmtp-weighted",
    publication_policy="reporting",
    extra_artifacts=("reference-inference.csv.gz",),
    modules=(
        "tests/studies/canonical_weighted_ltmle_crossfit.py",
        "tests/studies/weighted_longitudinal_common.py",
        "tests/studies/weighted_ltmle_crossfit_properties.py",
        "tests/studies/weighted_longitudinal_properties_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/ltmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit/audit.py",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_crossfit/smoke_weighted.R",
        "tests/canonical/lmtp_weighted_glm_adapter.R",
        "tests/canonical/study_harness.R",
        "tests/canonical/ltmle_regimen_adapter.R",
        "tests/canonical/weighted_lmtp_ltmle/run_study.R",
    ),
    runner_module="tests.studies.canonical_weighted_ltmle_crossfit",
    properties_module="tests.studies.weighted_ltmle_crossfit_properties",
    property_cells=property_cells(),
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": "f04a2b47f46debc515ce4ae778e05ebfde922c44",
    "lmtp_tarball_sha256": "fd49d9f291d4ddabb78c36d152b25aaa234a7204b645b9921f998c152e3d2ba5",
    "ife_version": IFE_VERSION,
    "ife_tarball_sha256": IFE_SHA256,
    "r_base_image": (
        "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
    ),
}

CONFIGURATION = {
    "construction": "fold_specific_cross_fit_weighted",
    "selection_probability": {
        "W1_positive": common.SELECTION_LOW,
        "otherwise": common.SELECTION_HIGH,
    },
    "selected_n": common.PRIMARY_N,
    "cross_fit": True,
    "outer_folds": 5,
    "learner_folds": 2,
    "weights": "fixed_inverse_selection_probability",
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
    "g_bounds": list(common.G_BOUNDS),
    "regimens": list(common.REGIMENS),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return common.draw_scenario(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return common.fit_cleverly(frame, cross_fit=True)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return common.rows_from_result(STUDY, fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    return common.replicate(STUDY, payload, cross_fit=True)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return common.draw_and_fit(STUDY, replicates=replicates, n=n, cross_fit=True, n_jobs=n_jobs)


def reference_artifacts(
    *,
    reference: Any,
    here: Path,
    samples: Path,
    truths_path: Path,
    reference_results: Path,
    output: Path,
    cores: int,
) -> dict[str, pd.DataFrame]:
    """Retain both weighted influence-curve conventions from the reference run."""
    del reference, here, samples, truths_path, output, cores
    rows = pd.read_csv(reference_results)
    keys = (
        "implementation",
        "scenario",
        "replicate",
        "n",
        "estimand",
        "truth",
        "estimate",
    )
    if not np.allclose(rows["std_error"], rows["native_std_error"], rtol=0.0, atol=0.0):
        raise RuntimeError("lmtp diagnostic native standard errors differ from the primary rows")
    if not np.allclose(rows["native_std_error"], rows["ht_std_error"], rtol=1e-12, atol=0.0):
        raise RuntimeError("lmtp native standard errors differ from the HT formula")
    if np.allclose(rows["ht_std_error"], rows["hajek_std_error"], rtol=1e-12, atol=0.0):
        raise RuntimeError("lmtp HT and Hajek formulas have no nonzero witness")
    z = 1.959963984540054
    frames = []
    for method, column in (
        ("native", "native_std_error"),
        ("horvitz_thompson", "ht_std_error"),
        ("hajek", "hajek_std_error"),
    ):
        frame = rows.loc[:, list(keys)].copy()
        frame["inference_method"] = method
        frame["std_error"] = rows[column].to_numpy(dtype=float)
        frame["ci_lower"] = frame["estimate"] - z * frame["std_error"]
        frame["ci_upper"] = frame["estimate"] + z * frame["std_error"]
        frame["covered"] = (
            (frame["ci_lower"] <= frame["truth"]) & (frame["truth"] <= frame["ci_upper"])
        ).astype(int)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[["std_error", "ci_lower", "ci_upper"]]).all().all():
        raise RuntimeError("lmtp diagnostic inference contains non-finite values")
    if not (result["std_error"] > 0.0).all():
        raise RuntimeError("lmtp diagnostic inference contains non-positive standard errors")
    return {"reference-inference.csv.gz": result}
