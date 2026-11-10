"""A feature-sign witness of the adaptive TMLE lasso, roadmap row X90.

One standardized lasso problem ``c'Gc / 2 - b'c + lambda |c|_1`` from the effect-model
selection of study law ``a3`` (``tests.studies.adaptive_tmle_trial``, seed base 771_000_000,
replicate 117, n = 4000; the first outer fold, 3200 rows and 21 penalized columns), at grid
index 53, started from the solution at index 52 as the grid path starts it. A feature-sign
search that admits a new coefficient before the active set meets its optimality conditions
trades columns 12 and 13 in and out of the active set until its iteration cap, and leaves a
Karush-Kuhn-Tucker residual of 0.0104 against a tolerance of 7.3e-09; the fit refused the
replicate.

The data are float64 little-endian, base64: the penalty, then ``b`` (21), the start (21)
and ``G`` (21 x 21, row major).
"""

from __future__ import annotations

import base64

import numpy as np

_DATA = """
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
"""


def witness() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """``(gram, linear, penalty, start)`` of the witness."""
    values = np.frombuffer(base64.b64decode("".join(_DATA.split())), dtype="<f8")
    p = 21
    penalty = float(values[0])
    linear = values[1 : 1 + p].copy()
    start = values[1 + p : 1 + 2 * p].copy()
    gram = values[1 + 2 * p :].reshape(p, p).copy()
    return gram, linear, penalty, start
