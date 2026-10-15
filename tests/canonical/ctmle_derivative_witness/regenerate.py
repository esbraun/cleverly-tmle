"""Regenerate the R witness of C-TMLE0, R ``ctmle`` 0.1.2 ``ctmletype = 3``.

The inputs are drawn here from a fixed seed.  The pinned image
``cleverly-ctmle-reference:18de559`` (built from ``tests/canonical/ctmle_selector``) runs
``witness.R`` on them, which fits ``glmnet`` at the two declared penalties and runs a copy
of the type-3 code beside ``ctmleGlmnet`` and ``ctmleGeneral``.  The checks below must hold
before the outputs are written into the manifest.  Run from the repository root with
Docker running::

    python -m tests.canonical.ctmle_derivative_witness.regenerate
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-ctmle-reference:18de559"
CTMLE_COMMIT = "18de559f47dc1286617350a0668391e80e1dbf7c"
SEED = 20263002
N = 1000
#: The two declared penalties.  The partner penalty is the larger one, as in the ``ctmle``
#: README, where the partner sits at ``lambda.min * 1.05``.
LAM = 0.012
LAM_PLUS = 0.018
GBOUND = 0.025
ALPHA = 0.995
#: Block B's partner is ``plogis(CROSS_FACTOR * qlogis(g1WPrev))``, which crosses ``gbound``
#: on rows where ``g1W`` does not.
CROSS_FACTOR = 3.0
#: The least number of rows each crossing block must hold.
MIN_CROSSING = 20
BLOCKS = ("A-binary", "A-continuous", "B", "C")
OUTPUTS = (
    "summary.csv",
    "lasso.csv",
    "versions.csv",
    *(f"rows-{block}.csv" for block in BLOCKS),
)


def draw_inputs() -> pd.DataFrame:
    """The covariates, the treatment, two outcomes, their initial regressions and block C.

    ``W1`` is an instrument, ``W2`` and ``W3`` are confounders, ``W4`` affects the outcome
    only, and ``W5`` and ``W6`` are noise.  The treatment law is main-terms logistic.  Both
    initial regressions are fixed working models that leave ``W3`` out, so the outcome
    regression is wrong and the derivative columns move the fit.
    """
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(N, 6))
    a = rng.binomial(1, expit(0.6 * w[:, 0] + 0.4 * w[:, 1] - 0.3 * w[:, 2])).astype(float)
    y_binary = rng.binomial(
        1, expit(-0.2 + 0.6 * a + 0.7 * w[:, 1] + 0.5 * w[:, 2] + 0.6 * w[:, 3])
    ).astype(float)
    y_continuous = 2.0 + 1.5 * a + w[:, 1] + 0.8 * w[:, 2] + w[:, 3] + rng.normal(size=N)

    def q_binary(level: float) -> np.ndarray:
        values = expit(-0.2 + 0.6 * level + 0.7 * w[:, 1] + 0.6 * w[:, 3])
        return np.asarray(np.clip(values, 0.02, 0.98), dtype=float)

    def q_continuous(level: float) -> np.ndarray:
        return np.asarray(2.0 + 1.5 * level + w[:, 1] + w[:, 3], dtype=float)

    # Block C: a mechanism clipped at both bounds, and a partner that halves its logits.
    # The logits stop at 6.5, so the partner never reaches gbound and no row has a zero
    # finite difference.
    index = np.clip(2.6 * w[:, 0] + 0.8 * w[:, 1], -6.5, 6.5)
    frame = pd.DataFrame(w, columns=[f"W{j}" for j in range(1, 7)])
    frame["A"] = a
    frame["Y_binary"] = y_binary
    frame["Y_continuous"] = y_continuous
    frame["Q0W_binary"] = q_binary(0.0)
    frame["Q1W_binary"] = q_binary(1.0)
    frame["Q0W_continuous"] = q_continuous(0.0)
    frame["Q1W_continuous"] = q_continuous(1.0)
    frame["gM"] = expit(index)
    frame["gMPrev"] = expit(0.5 * index)
    return frame


def declared() -> pd.DataFrame:
    """The settings ``witness.R`` reads."""
    return pd.DataFrame(
        {
            "lam": [LAM],
            "lam_plus": [LAM_PLUS],
            "gbound": [GBOUND],
            "alpha": [ALPHA],
            "cross_factor": [CROSS_FACTOR],
        }
    )


def _bounded(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(values, GBOUND, 1.0 - GBOUND), dtype=float)


def normalized_design(rows: pd.DataFrame) -> np.ndarray:
    """The four type-3 columns, each derivative column divided by its arm's root mean square."""
    a = rows["A"].to_numpy()
    g1, g1_prev = _bounded(rows["g1W"].to_numpy()), _bounded(rows["g1WPrev"].to_numpy())
    columns = [(1 - a) / (1 - g1), a / g1]
    for arm, difference, g in ((0.0, g1_prev - g1, 1 - g1), (1.0, g1 - g1_prev, g1)):
        indicator = a == arm
        column = indicator / g**2 * difference
        columns.append(column / np.sqrt(np.mean(column[indicator] ** 2)))
    return np.column_stack(columns)


def check(directory: Path, inputs: pd.DataFrame) -> dict[str, float]:
    """The conditions under which the comparison is exact.  Raises when one fails."""
    summary = pd.read_csv(directory / "summary.csv", float_precision="round_trip")
    if list(summary["block"]) != list(BLOCKS):
        raise RuntimeError(f"summary.csv holds blocks {list(summary['block'])}")
    if int(summary["zero_ddg"].sum()) != 0:
        raise RuntimeError("a finite difference is zero, so R's 1e-10 substitution acts")
    counts: dict[str, float] = {}
    for block in BLOCKS:
        rows = pd.read_csv(directory / f"rows-{block}.csv", float_precision="round_trip")
        for prefix in ("", "verbatim_"):
            for column in ("Qstar_AW", "Qstar_0W", "Qstar_1W"):
                values = rows[prefix + column].to_numpy()
                if not (values.min() > 1 - ALPHA and values.max() < ALPHA):
                    raise RuntimeError(f"block {block}: {prefix}{column} reaches the shrink bound")
        design = normalized_design(rows)
        smallest = float(np.linalg.svd(design / np.sqrt(len(rows)), compute_uv=False).min())
        if smallest <= 1e-6:
            raise RuntimeError(f"block {block}: the normalized design is near singular")
        counts[f"smallest_singular_value_{block}"] = smallest
    rows_b = pd.read_csv(directory / "rows-B.csv", float_precision="round_trip")
    inside = (rows_b["g1W"] > GBOUND) & (rows_b["g1W"] < 1 - GBOUND)
    crossed = (rows_b["g1WPrev"] < GBOUND) | (rows_b["g1WPrev"] > 1 - GBOUND)
    counts["B_crossing"] = int((inside & crossed).sum())
    mechanism, partner = inputs["gM"].to_numpy(), inputs["gMPrev"].to_numpy()
    for label, clipped in (
        ("C_low", mechanism < GBOUND),
        ("C_high", mechanism > 1 - GBOUND),
    ):
        if np.any(_bounded(partner[clipped]) == _bounded(mechanism[clipped])):
            raise RuntimeError(f"block {label}: the partner equals the mechanism after the bound")
        counts[label] = int(clipped.sum())
    for label in ("B_crossing", "C_low", "C_high"):
        if counts[label] < MIN_CROSSING:
            raise RuntimeError(f"{label} holds {counts[label]} rows, fewer than {MIN_CROSSING}")
    return counts


def _write(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inputs = draw_inputs()
    _write(inputs, HERE / "inputs.csv")
    _write(declared(), HERE / "penalties.csv")
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
            "/work/witness.R",
            "/work/inputs.csv",
            "/work/penalties.csv",
            "/work",
        ],
        check=True,
    )
    for name in OUTPUTS:
        path = HERE / name
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        path.write_text(text, encoding="utf-8", newline="\n")
    counts = check(HERE, inputs)
    versions = pd.read_csv(HERE / "versions.csv").iloc[0].to_dict()
    manifest = {
        "image": IMAGE,
        "image_id": image,
        "ctmle_commit": CTMLE_COMMIT,
        "seed": SEED,
        "n": N,
        "lam": LAM,
        "lam_plus": LAM_PLUS,
        "gbound": GBOUND,
        "alpha": ALPHA,
        "cross_factor": CROSS_FACTOR,
        "checks": counts,
        "versions": {key: str(value) for key, value in versions.items()},
        "sha256": {
            name: _sha256(HERE / name)
            for name in ("inputs.csv", "penalties.csv", "witness.R", *OUTPUTS)
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
