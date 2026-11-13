"""Regenerate the band-t-reference evidence (Fix-2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator, so the run starts no container.

Order of work (Fix-2 plan rev 3.1, section 5): first the pilot,
``python -m tests.studies.band_t_reference_properties --pilot``, which writes
``tests/studies/band_t_reference_pilot.csv`` from 800 replicates per positive cell on the salted
stream ``band-t-reference/pilot``. The pilot fixes R (``band_t_reference.declared_replicates``)
and each normal-reference control's clause (``band_t_reference.reference_thresholds``) by the
declared formulas, and nothing else; it is committed before the run. The run then pins a
cleverly-tmle commit with Fix-2's implementation and without Fix-3's PR 2.

Red-cell reading (declared before the pilot; plan 5.5). A passing cell with every clause of its
paired controls licenses ``("mammen", "logit", "t")`` on the rows ``T_BAND_CELLS`` maps it to.
A cell whose coverage passes but whose clause fails licenses nothing. A red positive cell keeps
the option experimental on its rows and is diagnosed (D5). A failing nonzero witness or
magnitude clause is a design or implementation failure, diagnosed before any reading. A
high-Kish cell above 0.98 is a finding against the rule. The held-out ``strata_n1000`` cell is
a reporting row: green, the rule holds out of sample; red with its shown-band reading inside
(0.92, 0.98), the shortfall is the refusal rate; red with that reading outside, a finding
against the rule. No replicate count, cell, margin or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import band_t_reference, band_t_reference_properties

if __name__ == "__main__":
    run(band_t_reference, band_t_reference_properties, here=Path(__file__).resolve().parent)
