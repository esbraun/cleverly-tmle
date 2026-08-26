"""Regenerate cross-fitted competing-risk LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import (
    canonical_ltmle_competing_crossfit,
    ltmle_competing_crossfit_properties,
)

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_ltmle_competing_crossfit,
        ltmle_competing_crossfit_properties,
        here=Path(__file__).parent,
        reference=Reference(
            image="cleverly-lmtp-crossfit:1.5.4",
            runner="lmtp_ltmle_competing/run_study.R",
            mount_runner=True,
            extra_files=("lmtp_crossfit_adapter.R", "lmtp_competing_adapter.R"),
            build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )
