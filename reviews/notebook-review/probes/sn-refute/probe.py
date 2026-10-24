"""The generated-outcome tests of Step 9 of docs/examples/survey-nonresponse.ipynb at 400 draws.

Run from the repository root:
``python reviews/notebook-review/probes/sn-refute/probe.py <test>``, with ``<test>`` one of
``dummy_outcome``, ``simulated_outcome`` and ``plasmode``.  The script runs the notebook's own
code cells up to the Step 6 fit, then calls ``refute`` with the notebook's declarations and
``n_replicates=400``.  The notebook runs the same call at 10 draws.  The run prints the report and
its wall time, and ``summary.log`` beside this file holds the three runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from cleverly.validation import BoundedAdjustmentOutcome, refute

NOTEBOOK = Path(__file__).resolve().parents[4] / "docs" / "examples" / "survey-nonresponse.ipynb"
UPTO = "estimate"


def _fit() -> dict:
    """Run the notebook's code cells through the Step 6 fit and return their namespace."""
    namespace: dict = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
        if cell["id"] == UPTO:
            return namespace
    raise RuntimeError(f"no cell named {UPTO!r}")


def main(test: str) -> None:
    full = _fit()["full"]
    declarations = {
        "dummy_outcome": {
            "dummy_outcome": BoundedAdjustmentOutcome(effect=0.0, adjustment_scale=0.0)
        },
        "simulated_outcome": {"simulated_outcome": BoundedAdjustmentOutcome()},
        "plasmode": {"simulated_outcome": BoundedAdjustmentOutcome()},
    }
    started = time.time()
    report = refute(full, tests=(test,), n_replicates=400, **declarations[test])
    print(report.summary())
    print(f"SECONDS {time.time() - started:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
