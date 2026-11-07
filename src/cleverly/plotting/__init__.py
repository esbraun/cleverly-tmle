"""Figures that show what a fitted result's tables cannot: shapes, tails and trade-offs.

Each function draws numbers the object already holds, labels them in words, and never draws
an interval or band the fit did not compute. A result whose inference is withheld draws its
points and says why in the note under the figure; an experimental result says so there.

matplotlib is an optional dependency (``pip install "cleverly[plot]"``). Importing
:mod:`cleverly` or this module does not import it; the first plot does.

The plots:

======================  ============================================================================
function                question
======================  ============================================================================
:func:`forest`          the estimates of one or several fits, with their intervals
:func:`curve`           how a result changes over time, dose, policy or modifier
:func:`sensitivity`     does the estimate depend on a declared sensitivity parameter?
:func:`overlap`         did every kind of unit have a chance of each treatment?
:func:`weights`         at which time point do a longitudinal fit's weights grow too large?
:func:`balance`         did weighting make treated and untreated units alike on each covariate?
:func:`zipper`          do a study's intervals cover the true value, and how do they miss?
======================  ============================================================================
:func:`calibration`     do predicted effects or probabilities match what was measured?
:func:`learners`        which learners carry the fit, and how sharp was the choice?
:func:`repeats`         does the estimate depend on how the data were split?
:func:`bootstrap`       is the bootstrap distribution close to normal?
:func:`influence`       how few rows or clusters carry the uncertainty?
======================  ===========================================================
"""

from __future__ import annotations

from ._core import PALETTE, style
from ._curves import curve
from ._diagnostics import bootstrap, calibration, influence, learners, repeats
from ._forest import forest
from ._overlap import overlap
from ._sensitivity import sensitivity
from ._weights import balance, weights
from ._zipper import zipper

__all__ = [
    "PALETTE",
    "balance",
    "bootstrap",
    "calibration",
    "curve",
    "forest",
    "influence",
    "learners",
    "overlap",
    "repeats",
    "sensitivity",
    "style",
    "weights",
    "zipper",
]
