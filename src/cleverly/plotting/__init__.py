"""Figures that show what a fitted result's tables cannot: shapes, tails and trade-offs.

Each function draws numbers the object already holds, labels them in words, and never draws
an interval or band the fit did not compute. A result whose inference is withheld draws its
points and says why in the note under the figure; an experimental result says so there.

matplotlib is an optional dependency (``pip install "cleverly[plot]"``). Importing
:mod:`cleverly` or this module does not import it; the first plot does.

The plots:

======================  ===========================================================
function                question
======================  ===========================================================
:func:`forest`          the estimates of one or several fits, with their intervals
:func:`curve`           how a result changes over time, dose, policy or modifier
:func:`sensitivity`     does the estimate depend on a declared sensitivity parameter?
:func:`overlap`         did every kind of unit have a chance of each treatment?
======================  ===========================================================
"""

from __future__ import annotations

from ._core import PALETTE, style
from ._curves import curve
from ._forest import forest
from ._overlap import overlap
from ._sensitivity import sensitivity

__all__ = ["PALETTE", "curve", "forest", "overlap", "sensitivity", "style"]
