"""The core bin count of an oracle binned density in the modified-treatment-policy studies.

Both ``policy-point-mtp`` and ``longitudinal-mtp`` give the package an oracle hazard: the exact
probability of each bin.  The only error left in the density ratio is then the representation,
the grid and its reading.  A fixed count leaves that error in place as ``n`` grows, so the ratio
is not consistent; the first ``longitudinal-mtp`` run found it with the histogram reading.  The
double-robustness cells with a wrong outcome regression need more: their bias is the ratio
error times the outcome error, so the ratio error must be ``o(n^(-1/2))``.

``oracle_bins(n) = ceil(320 (n / 2,000)^(2/3))`` sets the core count ``K`` of
:func:`cleverly.learners.density.bin_edges`.  The tail refinement adds at most the pieces its
row floor allows, and at 320 core bins on 2,000 rows (six rows per core bin) it adds none, so
``K`` is the grid.  The first run's diagnosis measured the histogram reading's error at 320 as
too small to move the cells, so the cells measure the targeting with a nearly exact ratio, which
is what an oracle hazard is for.  The corrected log-polygon reading at the same count is
measured again when these studies are regenerated.
"""

from __future__ import annotations

import math

#: The bin count at ``n = 2,000``, the size of every primary and calibration cell.
BINS_AT_2000 = 320


def oracle_bins(n: int) -> int:
    """``ceil(320 (n / 2000)^(2/3))``: 127 at 500, 202 at 1,000, 508 at 4,000, 807 at 8,000."""
    return math.ceil(BINS_AT_2000 * (n / 2000.0) ** (2.0 / 3.0))


def fit_bytes(n: int) -> int:
    """A generous peak of one fit, about ``100 n K`` bytes.

    The oracle hazards set ``bin_design = "index"``, so the pooled design holds about
    ``n K / 2`` records of a few columns.  One fit at ``n = 8,000`` and 807 bins peaked at
    0.36 GiB.
    """
    return 100 * n * oracle_bins(n)
