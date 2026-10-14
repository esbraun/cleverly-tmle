"""Regenerate longitudinal IPSI evidence against R ``npcausal`` (``longitudinal-incremental``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_longitudinal_incremental, longitudinal_incremental_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-npcausal:0.1.0-56a5ac1",
    runner="npcausal_longitudinal_incremental/run_study.R",
    mount_runner=True,
    extra_files=("npcausal_ipsi_adapter.R", "npcausal_incremental/Dockerfile", "study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "npcausal_incremental",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_longitudinal_incremental,
        longitudinal_incremental_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
