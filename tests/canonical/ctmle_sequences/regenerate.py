"""Regenerate the C-TMLE candidate-sequence evidence.

A full run, at the declared replicate count and with the property phase, refuses to start
while :data:`~tests.studies.canonical_ctmle_sequences.STAND_INS` names a key.  A probe, with
another ``--replicates`` or with ``--primary-only``, runs regardless.
"""

import argparse
import sys
from collections.abc import Sequence
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


def refuse_stand_ins(arguments: Sequence[str], stand_ins: frozenset[str]) -> None:
    """Refuse a full run while a declared gate holds a stand-in bias.

    Parameters
    ----------
    arguments : sequence of str
        The command-line arguments, without the program name.
    stand_ins : frozenset of str
        The keys that hold a stand-in bias.

    Raises
    ------
    SystemExit
        When the run is full and ``stand_ins`` is not empty.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--replicates", type=int, default=canonical_ctmle_sequences.PRIMARY_REPLICATES
    )
    parser.add_argument("--primary-only", action="store_true")
    known, _ = parser.parse_known_args(list(arguments))
    probe = known.primary_only or known.replicates != canonical_ctmle_sequences.PRIMARY_REPLICATES
    if stand_ins and not probe:
        raise SystemExit(
            "a full run needs a measured bias for every declared gate; these still hold "
            f"stand-ins: {sorted(stand_ins)}.  Run the property pilot "
            "(python -m tests.canonical.ctmle_sequences.power_probe --properties) first"
        )


if __name__ == "__main__":
    refuse_stand_ins(sys.argv[1:], canonical_ctmle_sequences.STAND_INS)
    main(
        canonical_ctmle_sequences,
        ctmle_sequence_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
