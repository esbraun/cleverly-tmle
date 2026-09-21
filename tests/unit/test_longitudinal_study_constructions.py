"""Construction metadata shared by the pooled longitudinal evidence studies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.studies import (
    canonical_categorical_ltmle_crossfit,
    canonical_ltmle_competing_crossfit,
    canonical_ltmle_crossfit,
    canonical_ltmle_survival_crossfit,
    canonical_weighted_ltmle_crossfit,
)
from tests.studies.evidence.constructions import (
    POOLED_LONGITUDINAL_CROSS_FIT,
    TRAINING_FOLD_FLUCTUATION,
)

ROOT = Path(__file__).resolve().parents[2]

STUDIES = (
    (canonical_ltmle_crossfit, "lmtp_ltmle"),
    (canonical_weighted_ltmle_crossfit, "weighted_lmtp_ltmle"),
    (canonical_categorical_ltmle_crossfit, "categorical_ltmle_crossfit"),
    (canonical_ltmle_survival_crossfit, "lmtp_ltmle_survival"),
    (canonical_ltmle_competing_crossfit, "lmtp_ltmle_competing_crossfit"),
)


@pytest.mark.parametrize(("study", "artifact"), STUDIES)
def test_pooled_longitudinal_studies_name_the_same_constructions(
    study: object, artifact: str
) -> None:
    configuration = study.CONFIGURATION  # type: ignore[attr-defined]
    manifest = json.loads((ROOT / "tests" / "canonical" / artifact / "manifest.json").read_text())

    for recorded in (configuration, manifest["configuration"]):
        assert recorded["construction"] == POOLED_LONGITUDINAL_CROSS_FIT
        assert recorded["reference_construction"] == TRAINING_FOLD_FLUCTUATION
