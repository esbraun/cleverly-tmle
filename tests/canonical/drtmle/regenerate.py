"""Regenerate the canonical DR-TMLE evidence study."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_drtmle, drtmle_properties
from tests.studies.evidence.registry import ROOT

if __name__ == "__main__":
    main(
        canonical_drtmle,
        drtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-drtmle-reference:538a3a2",
            runner="drtmle/run_drtmle.R",
            mount_runner=True,
            build_context=ROOT / "tests" / "canonical" / "drtmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )
