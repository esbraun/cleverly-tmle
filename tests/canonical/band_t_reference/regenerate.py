"""Regenerate the band-t-reference evidence (Fix-2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. The study
has no comparator, so the run starts no container.

Order of work (Fix-2 plan rev 3.1, section 5, with rev 3.3): the pilot,
``python -m tests.studies.band_t_reference_properties --pilot``, wrote
``tests/studies/band_t_reference_pilot.csv`` from 800 replicates per positive cell on the salted
stream ``band-t-reference/pilot``, and is committed. It fixed R by the declared coverage-only
rule (``band_t_reference.declared_replicates``: 5,600, the red family deciding at pass 0.957)
and each normal-reference control's declared gain value
(``band_t_reference.REFERENCE_THRESHOLDS``), and nothing else. The run then pins a
cleverly-tmle commit with Fix-2's implementation and without Fix-3's PR 2.

What gates (rev 3.3): each ``t`` band's joint rule (the 99% CP interval inside (0.92, 0.98)),
each pointwise joint control (upper limit below 0.95), and the deterministic checks inside
``t_band_rows`` (the package ``c_j`` agree with the rule, the expansion bound, the widening
witness), which stop the run on any replicate that fails one. What is reported: each
normal-reference control's paired gain interval, published beside its declared value by
``band_t_reference_properties.reference_readings``. It enters no verdict and sized nothing.

Red-cell reading (declared before the pilot; plan 5.5, rev 3.3). A cell whose joint rule and
pointwise control pass licenses ``("mammen", "logit", "t")`` on the rows ``T_BAND_CELLS`` maps
it to. A red positive cell keeps the option experimental on its rows and is diagnosed (D5). A
gain reading below its declared value is reported with the cell and diagnosed; it licenses or
blocks nothing. A high-Kish cell above 0.98 is a finding against the rule. The held-out
``strata_n1000`` cell is its own reporting study, ``band-t-reference-held-out``. No replicate
count, cell, margin or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import band_t_reference, band_t_reference_properties

if __name__ == "__main__":
    run(band_t_reference, band_t_reference_properties, here=Path(__file__).resolve().parent)
