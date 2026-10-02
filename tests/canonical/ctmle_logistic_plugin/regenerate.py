"""Regenerate the R ``calc_varIC`` witness for ``cleverly.estimators.logistic_plugin``.

One fit of the registered selector study's first discrete-strategy sample supplies the
targeted outcome regression, the bounded propensity and the selected covariates.  R
``ctmle``'s own ``calc_varIC`` then runs on exactly those inputs in the pinned image, so
the comparison checks the formula and not the selection.  Run from the repository root
with Docker running::

    python -m tests.canonical.ctmle_logistic_plugin.regenerate
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from tests.studies import canonical_ctmle_selector as selector

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-ctmle-reference:18de559"
SCENARIO = "binary_discrete"
REPLICATE = 0


def witness_inputs() -> pd.DataFrame:
    """The inputs of R's ``calc_varIC`` from the cleverly fit of the witness sample."""
    frame, _ = selector.draw_scenario(SCENARIO, selector.PRIMARY_N, REPLICATE)
    result = selector.fit_cleverly(frame, SCENARIO)
    return inputs_of(result)


def inputs_of(result: Any) -> pd.DataFrame:
    """``Y``, ``A``, the targeted ``Q``, the bounded ``g`` and the selected covariates."""
    data = result.data
    targeted = result.fluctuations["mean"].targeted
    g1 = result.nuisance.propensity.bounded(result.config.g_bounds)[:, 1]
    selected = result.ctmle_selection.selected_covariates
    columns: dict[str, Any] = {
        "Y": data.outcome,
        "A": data.treatment,
        "QAW": targeted.observed,
        "Q0W": targeted.arms[0.0],
        "Q1W": targeted.arms[1.0],
        "g1W": g1,
    }
    for name in selected:
        columns[name] = data.covariates[:, data.covariate_names.index(name)]
    return pd.DataFrame(columns)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inputs = witness_inputs()
    if inputs.shape[1] <= 6:
        raise RuntimeError("the witness sample must select a candidate with covariates")
    input_path = HERE / "inputs.csv"
    output_path = HERE / "output.csv"
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        inputs.to_csv(handle, index=False, float_format="%.17g", lineterminator="\n")
    image = subprocess.run(
        ["docker", "inspect", IMAGE, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "Rscript",
            "-v",
            f"{HERE}:/work",
            IMAGE,
            "/work/run_calc_varic.R",
            "/work/inputs.csv",
            "/work/output.csv",
        ],
        check=True,
    )
    text = output_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    output_path.write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "image": IMAGE,
        "image_id": image,
        "ctmle_commit": selector.CTMLE_COMMIT,
        "scenario": SCENARIO,
        "replicate": REPLICATE,
        "n": len(inputs),
        "selected_covariates": [c for c in inputs.columns if c.startswith("W")],
        "sha256": {
            name: _sha256(HERE / name) for name in ("inputs.csv", "output.csv", "run_calc_varic.R")
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
