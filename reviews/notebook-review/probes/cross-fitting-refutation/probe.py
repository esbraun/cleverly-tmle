"""Review probe for the refutation step of ``docs/examples/cross-fitting.ipynb``.

Usage: ``python probe.py <part>`` with part in ``team``, ``main_generated``, ``repeated``.
Each part rebuilds the notebook's fits from its code cells and runs the check of the step
at the draw count of the generated-outcome bands (400), which the executed cells do not.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding  # noqa: F401
from cleverly.validation import refute
from cleverly.validation.outcome_processes import BernoulliAdjustmentOutcome, BoundedAdjustmentOutcome

NOTEBOOK = Path(__file__).resolve().parents[4] / "docs" / "examples" / "cross-fitting.ipynb"
LAST_FIT_CELL = "repeated = effect.estimate("


def namespace() -> dict:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    ns: dict = {}
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            exec(source, ns)  # noqa: S102
            if source.startswith(LAST_FIT_CELL):
                break
    return ns


def timed(label, call):
    start = time.time()
    report = call()
    print(f"===== {label}: {time.time() - start:.1f} s")
    print(report.summary(), flush=True)
    return report


if __name__ == "__main__":
    ns = namespace()
    part = sys.argv[1]
    if part == "team":
        process = BernoulliAdjustmentOutcome(effect=0.5, adjustment_scale=0.5)
        for kind in ("plasmode", "simulated_outcome"):
            timed(
                f"team {kind} R=400",
                lambda kind=kind: refute(
                    ns["clustered"], tests=(kind,), simulated_outcome=process, n_replicates=400
                ),
            )
        timed(
            "team subset R=100",
            lambda: refute(ns["clustered"], tests=("subset",), n_replicates=100),
        )
    elif part == "main_generated":
        timed(
            "main bounded simulated_outcome R=400",
            lambda: refute(
                ns["cross_fitted"],
                tests=("simulated_outcome",),
                simulated_outcome=BoundedAdjustmentOutcome(effect=0.6, adjustment_scale=0.5),
                n_replicates=400,
            ),
        )
