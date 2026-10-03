"""Regenerate the stratified incremental and MSM evidence against R ``npcausal``."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import (
    canonical_stratified_incremental_msm,
    stratified_incremental_msm_properties,
)
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-npcausal:0.1.0-56a5ac1",
    runner="npcausal_stratified_incremental/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "npcausal_incremental",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_stratified_incremental_msm,
        stratified_incremental_msm_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
