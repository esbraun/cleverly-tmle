"""The bin count of an oracle binned density in the modified-treatment-policy studies.

Both ``policy-point-mtp`` and ``longitudinal-mtp`` give the package an oracle hazard: the exact
probability of each bin.  The only error left in the density ratio is then the binning, which
is of order ``1 / K`` for ``K`` bins.  A fixed ``K`` leaves that error in place as ``n`` grows,
so the ratio is not consistent; the first ``longitudinal-mtp`` run found it.  The
double-robustness cells with a wrong outcome regression need more: their bias is the ratio
error times the outcome error, so the ratio error must be ``o(n^(-1/2))``.  ``K`` therefore
grows as ``n^(2/3)``, faster than ``n^(1/2)``.

The count at ``n = 2,000`` is 320.  The first run's diagnosis measured the oracle's binning
error there: with exact bin probabilities the influence curve's spread was 1.14 times the
efficient one at 80 bins, 1.04 at 160 and 0.99 at 320 on one draw.  At 320 the binning no
longer moves the cells, so the cells measure the targeting with a nearly exact ratio, which is
what an oracle hazard is for.
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
