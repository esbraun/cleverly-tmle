"""Regenerate C-TMLE0 derivative-covariate evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ctmle_derivative, ctmle_derivative_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-ctmle-reference:18de559",
    runner="ctmle_derivative/run_ctmle_derivative.R",
    mount_runner=True,
    extra_files=("study_harness.R", "ctmle_derivative/penalties.csv"),
    build_context=ROOT / "tests" / "canonical" / "ctmle_selector",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_ctmle_derivative,
        ctmle_derivative_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
