"""Regenerate the canonical DR-TMLE evidence study."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_drtmle, drtmle_properties
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent

#: The pinned R ``drtmle`` runner.  A module constant so that a diagnostic that runs the same
#: comparator, such as ``tests/diagnostics/rm19_one_sided_increment``, calls it with these
#: arguments rather than a copy of them.
REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="drtmle/run_drtmle.R",
    mount_runner=True,
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(canonical_drtmle, drtmle_properties, here=HERE, reference=REFERENCE)
