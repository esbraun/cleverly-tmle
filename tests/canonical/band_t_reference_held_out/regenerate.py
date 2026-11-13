"""Regenerate the band-t-reference held-out evidence (Fix-2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The study has no comparator. Its pilot,
``python -m tests.studies.band_t_reference_held_out_properties --pilot``, writes
``tests/studies/band_t_reference_held_out_pilot.csv`` on the salted stream
``band-t-reference-held-out/pilot``; it is committed. It fixed only the held-out cell's declared
gain value (``band_t_reference_held_out.REFERENCE_THRESHOLDS``: none, the pilot gain carrying no
witness). The paired gain is reported beside that value and gates nothing (plan rev 3.3). The
cell runs at ``band-t-reference``'s declared R, 5,600.

Red-cell reading (declared before the pilot; plan 5.2). The study publishes under
``reporting``. Green: the rule holds out of sample. Red with the shown-band reading (fits with
no constant cell) inside (0.92, 0.98): the shortfall is the refusal rate, reported to the owner
with the measured constant-cell share. Red with that reading also outside: a finding against the
rule (D5). A red verdict gets a Fix-2 owner entry when it is published. No replicate count,
cell, margin or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import band_t_reference_held_out, band_t_reference_held_out_properties

if __name__ == "__main__":
    run(
        band_t_reference_held_out,
        band_t_reference_held_out_properties,
        here=Path(__file__).resolve().parent,
    )
