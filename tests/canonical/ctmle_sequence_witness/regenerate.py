"""Regenerate the R ``ctmle`` candidate-sequence witness.

The seed comes from a declared search over :data:`SEEDS`, in order. Each seed is screened
on the package side first, on the package's own solver arrays. A seed that passes runs R
``ctmle`` and ``glmnet`` in the pinned image, and its R outputs must pass the same screen
plus the conditions on R's selection. The first seed that passes is kept, and the manifest
records every rejected seed with its reason. On every seed where ``ctmleGlmnet`` completed,
``rebuild_check.R`` runs the registered study's type-1 adapter, ``rebuild_type1``, and the
manifest records whether it agrees with ``ctmleGlmnet`` exactly (plan 3.4, S1). Run from the
repository root with Docker running::

    python -m tests.canonical.ctmle_sequence_witness.regenerate
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.learners._l1_logistic import penalty_path
from tests.canonical.ctmle_sequence_witness import harness

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-ctmle-reference:18de559"
CTMLE_COMMIT = "18de559f47dc1286617350a0668391e80e1dbf7c"
SEEDS = range(20261600, 20262100)
#: The final estimate must move by more than this from the selected candidate's (W3).
FINAL_MOVE = 1e-4
#: The R output files, in the order the manifest lists them.
OUTPUTS = (
    "g_tight.csv",
    "coef_tight.csv",
    "g_weighted.csv",
    "coef_weighted.csv",
    "g_multinomial.csv",
    "coef_multinomial.csv",
    "g_internal.csv",
    "g_glm.csv",
    "general_path.csv",
    "glmnet_path.csv",
    "g_final_cols.csv",
    "facts.csv",
)
SOURCES = ("witness.R",)
#: The S1 rebuild cross-check (X16 plan, 3.4): its script, and the shared adapter it sources
#: from ``tests/canonical/ctmle_sequences``.
REBUILD_SOURCES = ("rebuild_check.R",)
SHARED = HERE.parent / "ctmle_sequences"
SHARED_SOURCES = ("rebuild_type1.R",)


def read(path: Path) -> pd.DataFrame:
    """One CSV, parsed with correct rounding.

    The default pandas parser can miss a 17-digit value by about ``1e-13`` relative, so an
    exact comparison with a value the package computes would fail on the parse alone.
    """
    return pd.read_csv(path, float_precision="round_trip")


def read_facts(directory: Path) -> dict[str, str]:
    """``facts.csv`` as a mapping of strings."""
    facts = pd.read_csv(directory / "facts.csv", dtype=str, keep_default_na=False)
    return dict(zip(facts["key"], facts["value"], strict=True))


def matrix(directory: Path, name: str) -> np.ndarray:
    """One wide R output as an ``(n, K)`` array."""
    return read(directory / name).to_numpy(dtype=float)


def _screen_path(chooser: Any, columns: np.ndarray) -> tuple[str | None, harness.Path]:
    path = harness.indexed_path(chooser, columns)
    accepted = path.accepted
    if len(accepted) < 2:
        return f"the path accepts {len(accepted)} candidate(s); W1 needs two", path
    gaps = np.diff((-1, *accepted))
    if not np.any(gaps > 1):
        return "no accepted step skips a candidate (W1)", path
    if not path.stopped or accepted[-1] >= columns.shape[1] - 1:
        return "the stop rule does not bind before the last candidate (W2)", path
    margin = harness.smallest_margin(path)
    if margin <= harness.MARGIN:
        return f"a deciding score gap of {margin:.3g} is not above {harness.MARGIN:g}", path
    if not harness.interior(path):
        return f"a targeted Q leaves {harness.Q_INTERIOR}", path
    return None, path


def screen(frame: pd.DataFrame, grid: np.ndarray) -> str | None:
    """The package-side screen, on the package's own solver arrays."""
    q = frame[["Q0W", "Q1W"]].to_numpy()
    low, high = harness.Q_RANGE
    if q.min() < low or q.max() > high:
        return f"the initial Q leaves {harness.Q_RANGE}: [{q.min():.4f}, {q.max():.4f}]"
    x = frame[list(harness.COVARIATES)].to_numpy(float)
    columns = np.column_stack(
        [fit.predict_proba(x)[:, 1] for fit in penalty_path(x, frame["A"].to_numpy(), None, grid)]
    )
    reason, _ = _screen_path(harness.selector(frame), columns)
    return reason


def confirm(frame: pd.DataFrame, outputs: Path) -> str | None:
    """The conditions on R's outputs: the screen on R's arrays, then R's selection (W3)."""
    chooser = harness.selector(frame)
    internal = matrix(outputs, "g_internal.csv")
    reason, path = _screen_path(chooser, internal)
    if reason is not None:
        return f"on R's internal arrays, {reason}"
    general = read(outputs / "general_path.csv")
    columns = np.column_stack([internal, matrix(outputs, "g_glm.csv")])
    accepted = tuple(int(i) - 1 for i in general.loc[general["accepted"] == 1, "index"])
    distance = harness.separated(columns, accepted)
    if distance < harness.SEPARATION:
        return f"an accepted ctmleGeneral column lies {distance:.3g} from an earlier one"
    facts = read_facts(outputs)
    count = internal.shape[1]
    selected = int(facts["glmnet_best_k"]) - 1
    if selected >= count - 1:
        return "R selects the last penalty"
    lasso = read(outputs / "glmnet_path.csv")
    final_estimate = float(facts["glmnet_est"])
    move = abs(final_estimate - float(lasso["est"].iloc[selected]))
    if move <= FINAL_MOVE:
        return f"the final step moves the estimate by {move:.3g} (W3)"
    final, _ = harness.final_step(
        chooser, path, internal, matrix(outputs, "g_final_cols.csv"), selected
    )
    if final in (selected + 1, count - 1):
        return f"the final step picks index {final}, an end of the remaining penalties (W3)"
    return None


def _write_inputs(frame: pd.DataFrame, grid: np.ndarray, directory: Path) -> None:
    with (directory / "inputs.csv").open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g", lineterminator="\n")
    with (directory / "grid.csv").open("w", encoding="utf-8", newline="\n") as handle:
        pd.DataFrame({"lambda": grid}).to_csv(
            handle, index=False, float_format="%.17g", lineterminator="\n"
        )


def _run_r(inputs: Path, outputs: Path) -> int:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "Rscript",
            "-v",
            f"{HERE}:/fixture:ro",
            "-v",
            f"{inputs}:/inputs:ro",
            "-v",
            f"{outputs}:/outputs",
            IMAGE,
            "/fixture/witness.R",
            "/inputs",
            "/outputs",
        ],
        check=False,
    ).returncode


def _run_rebuild(inputs: Path, outputs: Path) -> dict[str, Any]:
    """Run ``rebuild_check.R`` on one seed's inputs and read its agreement row."""
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "Rscript",
            "-v",
            f"{HERE}:/fixture:ro",
            "-v",
            f"{SHARED}:/shared:ro",
            "-v",
            f"{inputs}:/inputs:ro",
            "-v",
            f"{outputs}:/outputs",
            IMAGE,
            "/fixture/rebuild_check.R",
            "/inputs",
            "/outputs",
        ],
        check=True,
    )
    row = read(outputs / "rebuild.csv").iloc[0]
    best_k, rebuilt = int(row["best_k"]), int(row["rebuild_best_k"])
    return {
        "best_k": best_k,
        "rebuild_best_k": rebuilt,
        "agrees": bool(
            best_k == rebuilt and int(row["est_identical"]) and int(row["lambdas_identical"])
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lf(path: Path) -> None:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8", newline="\n")


def tolerances(facts: dict[str, str]) -> dict[str, float]:
    """Ten times each measured ``glmnet`` self-gap: the prediction tolerances of the test."""
    return {
        name: 10.0 * float(facts[f"self_gap_{name}"])
        for name in ("binary", "weighted", "multinomial")
    }


def main() -> None:
    image = subprocess.run(
        ["docker", "inspect", IMAGE, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    search: list[dict[str, Any]] = []
    rebuild: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as scratch:
        for seed in SEEDS:
            frame, grid = harness.draw_inputs(seed)
            reason = screen(frame, grid)
            if reason is not None:
                search.append({"seed": seed, "stage": "package", "reason": reason})
                continue
            inputs = Path(scratch) / f"{seed}-in"
            outputs = Path(scratch) / f"{seed}-out"
            inputs.mkdir()
            outputs.mkdir()
            _write_inputs(frame, grid, inputs)
            status = _run_r(inputs, outputs)
            if status == 10:
                message = (outputs / "ctmle_glmnet_error.txt").read_text(encoding="utf-8")
                search.append({"seed": seed, "stage": "R", "reason": message.strip()})
                continue
            if status != 0:
                raise RuntimeError(f"witness.R failed with status {status} at seed {seed}")
            check = {"seed": seed, **_run_rebuild(inputs, outputs)}
            rebuild.append(check)
            if not check["agrees"]:
                raise RuntimeError(f"rebuild_type1 disagrees with ctmleGlmnet at seed {seed}")
            reason = confirm(frame, outputs)
            if reason is not None:
                search.append({"seed": seed, "stage": "R", "reason": reason})
                continue
            _write_inputs(frame, grid, HERE)
            for name in OUTPUTS:
                shutil.copyfile(outputs / name, HERE / name)
                _lf(HERE / name)
            break
        else:
            raise RuntimeError(f"no seed in {SEEDS} passes the witness conditions")
    facts = read_facts(HERE)
    manifest = {
        "image": IMAGE,
        "image_id": image,
        "ctmle_commit": CTMLE_COMMIT,
        "glmnet_control": {"fdev": 0, "devmax": 1},
        "seed": seed,
        "search": search,
        "n": harness.N,
        "covariates": list(harness.COVARIATES),
        "grid_size": harness.GRID_SIZE,
        "facts": facts,
        "prediction_tolerance": tolerances(facts),
        "sha256": {
            name: _sha256(HERE / name) for name in ("inputs.csv", "grid.csv", *OUTPUTS, *SOURCES)
        },
        "rebuild": rebuild,
        "rebuild_sha256": {
            **{name: _sha256(HERE / name) for name in REBUILD_SOURCES},
            **{f"../ctmle_sequences/{name}": _sha256(SHARED / name) for name in SHARED_SOURCES},
        },
    }
    if not all(math.isfinite(value) for value in manifest["prediction_tolerance"].values()):
        raise RuntimeError("a glmnet self-gap is not finite")
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
