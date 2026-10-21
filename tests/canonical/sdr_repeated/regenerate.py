"""Regenerate repeated-outcome longitudinal SDR evidence (``longitudinal-repeated-sdr``).

The study pairs with R ``lmtp`` 1.5.4.  The driver runs the Python phase on the declared seeds,
spills each replicate's sample (the draw, the fold assignment and the ``ratio__`` and
``shift__`` columns) to one archive, and hands it to ``run_study.R`` in this directory.
That runner calls ``lmtp_sdr_with_folds`` once per plan and horizon on the panel truncated after
node ``h``.  It sources ``lmtp_sdr_repeated_adapter.R`` (the pairing learner
``SL.cellmeans.outcomes``), ``lmtp_sdr_adapter.R`` and ``lmtp_crossfit_adapter.R`` (both
unchanged, because ``longitudinal-sdr`` gates their hashes) and ``study_harness.R``.

Queue command, on the owner's machine (Docker and the ``lmtp_crossfit`` build context)::

    python -m tests.canonical.sdr_repeated.gates --replicates 5 --timing-only   # P0
    python -m tests.canonical.sdr_repeated.gates                                # G1r
    python -m tests.canonical.sdr_repeated.regenerate --skip-properties --jobs 12

``--skip-properties`` reuses the committed property artifacts, which PR B's declared run wrote.
The Python phase runs again, because the samples come from the fits, on the same seeds.  The
fast test ``test_the_python_rows_of_this_run_equal_the_rows_pr_b_committed`` requires its rows
to equal PR B's to 1e-10.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_sdr_repeated, sdr_repeated_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="sdr_repeated/run_study.R",
    mount_runner=True,
    extra_files=(
        "lmtp_sdr_repeated_adapter.R",
        "lmtp_sdr_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_sdr_repeated,
        sdr_repeated_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
