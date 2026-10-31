"""Review probe of the placebo in Step 11 of docs/examples/sequential-pricing.ipynb.

The notebook runs the band-rank placebo at 10 draws, because the fast suite runs its cells. This
probe runs every code cell of the notebook before the cell ``refute``, so the fit is the page's
own fit. It then runs the placebo on the ``price_up`` renewal contrast at 400 draws, with the
seed of the page.

Usage: ``python placebo.py > placebo.log`` from the repository root. A review probe is not a
registered study.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np

NOTEBOOK = Path("docs/examples/sequential-pricing.ipynb")
DRAWS = 400

warnings.filterwarnings("ignore")
namespace: dict = {}
for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
    if cell["cell_type"] != "code":
        continue
    if cell.get("id") == "refute":
        break
    exec(compile("".join(cell["source"]), str(cell.get("id")), "exec"), namespace)

from cleverly.validation import refute  # noqa: E402

estimand = "ate_regimen[price_up vs natural_prices]"
start = time.time()
report = refute(
    namespace["result"],
    estimand=estimand,
    tests=("placebo",),
    n_replicates=DRAWS,
    random_state=1000,
)
elapsed = time.time() - start
print(report.summary())
placebo = report["placebo"]
values = np.asarray(placebo.values)
errors = np.asarray(placebo.standard_errors)
sd = float(values.std(ddof=1))
print(f"draws {len(values)}; mean {values.mean():+.6f}; sd {sd:.6f}")
print(f"mean reported SE {errors.mean():.6f}; mean SE over sd {errors.mean() / sd:.3f}")
print(f"share of draws whose 95% Wald interval holds 0: {np.mean(np.abs(values) <= 1.96 * errors):.4f}")
print(f"seconds {elapsed:.0f}")
