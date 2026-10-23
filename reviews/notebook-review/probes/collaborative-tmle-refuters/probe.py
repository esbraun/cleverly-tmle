"""Review probe: the Step 10 refutation checks of collaborative-tmle.ipynb at 400 refits.

Run from the repository root: ``python reviews/notebook-review/probes/collaborative-tmle-refuters/probe.py``.
It executes the notebook's setup, data, protocol, design and two fit cells, then runs ``refute``
with the notebook's seed at ``n_replicates=400`` and prints one line per reading.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import time
from pathlib import Path

NOTEBOOK = Path("docs/examples/collaborative-tmle.ipynb")
FIT_CELLS = ("setup", "data", "association", "protocol", "identify", "tmle", "collaborative")


def main() -> None:
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    namespace: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name in FIT_CELLS:
            exec(cells[name], namespace)  # noqa: S102 (the notebook's own cells)
    from cleverly.validation import refute

    for label, fit in (("tmle", namespace["tmle"]), ("oat", namespace["collaborative"])):
        started = time.time()
        report = refute(fit, estimand="ate", n_replicates=400, random_state=44)
        text = report.summary()
        t = float(re.search(r"^  t = (-?[\d.]+)", text, re.M).group(1))
        se_over_sd = float(re.search(r"mean reported SE over SD of the draws = ([\d.]+)", text).group(1))
        shift = float(re.search(r"paired shift in standard errors = (-?[\d.]+)", text).group(1))
        rho = float(re.search(r"scatter ratio rho = ([\d.]+)", text).group(1))
        print(
            f"{label} placebo t {t:.4f} mean se / sd {se_over_sd:.3f}; "
            f"random_common_cause paired shift {shift:+.4f}; subset rho {rho:.3f}; "
            f"400 refits each in {time.time() - started:.1f} s"
        )


if __name__ == "__main__":
    main()
