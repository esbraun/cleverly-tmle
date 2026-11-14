"""Figures that show what a fitted result's tables cannot: shapes, tails and trade-offs.

Each function draws numbers the object already holds, labels them in words, and never draws
an interval or band the fit did not compute. A result whose inference is withheld draws its
points with no interval. The figure carries no notes: :func:`notes` returns what a reader of
one fit may need, such as why an interval is missing or that the fit or its band is
experimental (its interval has no checked coverage), and ``summary()`` prints the same
statements for the result. Check one of them before you publish a figure.

matplotlib is an optional dependency (``pip install "cleverly[plot]"``). Importing
:mod:`cleverly` or this module does not import it; the first plot does.

The plots:

======================  ============================================================================
function                question
======================  ============================================================================
:func:`forest`          how large is each estimate, and how sure is it?
:func:`curve`           how a result changes over time, dose, policy or modifier
:func:`sensitivity`     does the estimate depend on a declared sensitivity parameter?
:func:`confounding`     how strong would hidden confounding need to be?
:func:`overlap`         did every kind of unit have a chance of each treatment?
:func:`targeting`       how much does treating by a score first gain over random?
:func:`weights`         at which time point do a longitudinal fit's weights grow too large?
:func:`balance`         did weighting make treated and untreated units alike on each covariate?
:func:`zipper`          do a study's intervals cover the true value, and how do they miss?
:func:`calibration`     do predicted effects or probabilities match what was measured?
:func:`learners`        which learners carry the fit, and how sharp was the choice?
:func:`repeats`         does the estimate depend on how the data were split?
:func:`bootstrap`       is the bootstrap distribution close to normal?
:func:`influence`       how few rows or clusters carry the uncertainty?
======================  ============================================================================
"""

from __future__ import annotations

from ._confounding import confounding
from ._core import PALETTE, notes, style
from ._curves import curve
from ._diagnostics import bootstrap, calibration, influence, learners, repeats
from ._forest import forest
from ._overlap import overlap
from ._sensitivity import sensitivity
from ._targeting import targeting
from ._weights import balance, weights
from ._zipper import zipper

__all__ = [
    "PALETTE",
    "balance",
    "bootstrap",
    "calibration",
    "confounding",
    "curve",
    "forest",
    "influence",
    "learners",
    "notes",
    "overlap",
    "repeats",
    "sensitivity",
    "style",
    "targeting",
    "weights",
    "zipper",
]
