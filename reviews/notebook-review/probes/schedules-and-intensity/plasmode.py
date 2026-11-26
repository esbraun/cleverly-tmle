"""Plasmode test of the schedules-and-intensity page's tilt fit, at 400 draws under eight seeds.

Run from the repository root:
    python reviews/notebook-review/probes/schedules-and-intensity/plasmode.py \
        > reviews/notebook-review/probes/schedules-and-intensity/plasmode.log

It rebuilds the page's tilt fit from the notebook's code cells and runs ``plasmode`` on the
"high doubled" estimate with the default ``BoundedAdjustmentOutcome``: each draw resamples the
page's covariates, draws a schedule from the frozen fitted treatment model and a bounded score
with a known effect, and refits the page's method. One 400-draw run reads the standardized bias
with a 95% interval about 0.2 wide, so the probe repeats it under eight seeds and pools them.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

from cleverly.validation import BoundedAdjustmentOutcome, refute

NOTEBOOK = (
    Path(__file__).resolve().parents[4] / "docs" / "examples" / "schedules-and-intensity.ipynb"
)
DRAWS = 400
SEEDS = (24, 7, 1, 2, 3, 4, 5, 6)


def namespace() -> dict:
    """Run the notebook's code cells up to and including the tilt fit."""
    ns: dict = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # the cells' own prints
            exec("".join(cell["source"]), ns)
        if cell.get("id") == "estimate":
            break
    return ns


if __name__ == "__main__":
    fit = namespace()["tilt_result"]
    print(f"plasmode, high doubled; n = {fit.data.n}; draws = {DRAWS} per seed")
    biases = []
    for seed in SEEDS:
        report = refute(
            fit,
            estimand="ate_ipsi[high doubled vs current practice]",
            tests=("plasmode",),
            simulated_outcome=BoundedAdjustmentOutcome(),
            n_replicates=DRAWS,
            random_state=seed,
        )
        row = report.to_frame().iloc[0]
        bias = (row["refuted_mean"] - row["declared_effect"]) / row["refuted_sd"]
        biases.append(bias)
        outside = [
            f"{r.name} {r.estimate:.3f} [{r.lower:.3f}, {r.upper:.3f}] outside {r.band}"
            for r in report.tests[0].readings
            if not r.inside
        ]
        verdict = "; ".join(outside) if outside else "all readings inside their bands"
        print(f"seed {seed}: standardized bias {bias:+.3f}; {verdict}")
    pooled = sum(biases) / len(biases)
    half = 1.96 / math.sqrt(DRAWS * len(biases))
    print(
        f"pooled over {DRAWS * len(biases)} draws: {pooled:+.3f} "
        f"({pooled - half:+.3f}, {pooled + half:+.3f})"
    )
    print(f"largest absolute: {max(abs(b) for b in biases):.3f}")
