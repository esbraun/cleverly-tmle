"""Run the notebook's generated-outcome test of section 9 with 400 draws.

The refits run with three jobs, which changes the speed and not the numbers. Run from the
repository root, with ``TWINS_DATA`` set to a directory that holds the pinned CEVAE files or
left unset to download them::

    python reviews/notebook-review/probes/twins-birth-weight/generated_outcome_400.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common

from cleverly import Runtime
from cleverly.validation import BoundedAdjustmentOutcome

parallel = dataclasses.replace(common.method(), runtime=Runtime(random_state=common.SEED, n_jobs=3))
common.method = lambda: parallel
start = time.time()
result = common.fit()
print(f"fit {time.time() - start:.1f} s, att {result['att'].psi:.4f}", flush=True)
start = time.time()
report = result.diagnostics.refute(
    estimand="att",
    tests=("simulated_outcome",),
    n_replicates=400,
    simulated_outcome=BoundedAdjustmentOutcome(intercept=-1.0, effect=-0.1, adjustment_scale=0.2),
    random_state=common.SEED,
)
print(f"400 draws in {time.time() - start:.0f} s", flush=True)
print(report.summary())
