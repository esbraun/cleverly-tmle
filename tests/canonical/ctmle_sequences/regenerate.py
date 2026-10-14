"""Regenerate the C-TMLE candidate-sequence evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ctmle_sequences, ctmle_sequence_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-ctmle-reference:18de559",
    runner="ctmle_sequences/run_ctmle_sequences.R",
    mount_runner=True,
    extra_files=(
        "study_harness.R",
        "ctmle_sequences/lambdas.csv",
        "ctmle_sequences/make_r_folds.R",
        "ctmle_sequences/rebuild_type1.R",
        "ctmle_sequences/r_folds.csv.gz",
    ),
    build_context=ROOT / "tests" / "canonical" / "ctmle_selector",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_ctmle_sequences,
        ctmle_sequence_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
