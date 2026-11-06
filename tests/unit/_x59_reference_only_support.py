"""A small synthetic study for the reference-only driver tests (X59 section 3).

Two scenarios draw ``n`` normal values from their own seed stream.  The subject's estimate is the
sample mean plus ``1e-9``, and the fake reference (:func:`fake_reference_run`) reports the plain
mean, so the two agree to well inside every declared parity tolerance.  The reference also
writes an untargeted ``gcomp_estimate``, the mean minus ``0.01``, which is the subject's plug-in
too, so the G-C and witness rows have something to read.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence import replication
from tests.studies.evidence.parity import ParityCell, ParityDeclaration, ReferenceSample
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 12
PRIMARY_N = 60
SUBJECT = "cleverly-synthetic"
REFERENCE = "synthetic-reference"
MOVE = 0.01

STUDY = StudyRecord(
    name="x59 synthetic study",
    slug="x59-synthetic",
    artifacts=ROOT / "tests" / "canonical" / "x59_synthetic",
    document="docs/technical-reference/index.md",
    anchor="x59-synthetic",
    scenarios={"a": ("ey[a]",), "b": ("ey[b]",)},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=59_000_001,
    implementation=SUBJECT,
    pending_reference=REFERENCE,
    pending_reason="X59 driver test: the reference runs reference-only",
    comparison="matching implementation",
    margins=Margins(bootstrap_replicates=200),
    publication_policy="reporting",
    failure_bound={SUBJECT: 0.01, REFERENCE: 0.01},
)

#: The fields the comparator run declares; the registration commit moves them into ``STUDY``.
COMPARATOR_DECLARATION: dict[str, Any] = {
    "reference_replicates": {"a": 10, "b": 8},
    "paired_role": "descriptive",
}

PARITY = ParityDeclaration(
    cells=(
        ParityCell("G-P", "a", "ey[a]"),
        ParityCell("G-C", "a", "ey[a]"),
        ParityCell("G1", "b", "ey[b]"),
    ),
    tolerance={"G-P": 1e-6, "G-C": 1e-6},
)

CONFIGURATION: dict[str, Any] = {}

#: The comparator's declared environment.  A ``shift`` mutation moves every drawn value by
#: :data:`SHIFT`, so a mutated archive differs from the unmutated one.
MUTATION_ENV = "CLEVERLY_X59_MUTATION"
REFERENCE_ENVIRONMENT = {MUTATION_ENV: "none"}
SHIFT = 0.01
#: The environment of every :func:`fake_reference_run` call, in order.
RUNS: list[dict[str, str]] = []

#: Set by a test to make :func:`reference_sample` refuse, proving that a resumed run reused
#: its inputs instead of drawing them again.
DRAWS_REFUSED = False


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n)}), {f"ey[{scenario}]": 0.0}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def reference_sample(scenario: str, n: int, replicate: int) -> ReferenceSample:
    """The comparator input of one replicate, from its seed alone."""
    if DRAWS_REFUSED:
        raise AssertionError("a resumed reference-only run drew its inputs again")
    frame, truths = draw_scenario(scenario, n, replicate)
    if os.environ.get(MUTATION_ENV, "none") == "shift":
        frame["x"] = frame["x"] + SHIFT
    plugin = float(frame["x"].mean()) - MOVE
    return ReferenceSample(frame, truths, {f"ey[{scenario}]": plugin})


def _row(implementation: str, scenario: str, replicate: int, x: np.ndarray, shift: float) -> dict:
    estimate = float(np.mean(x)) + shift
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    low, high = estimate - 1.96 * se, estimate + 1.96 * se
    return {
        "implementation": implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": len(x),
        "estimand": f"ey[{scenario}]",
        "truth": 0.0,
        "estimate": estimate,
        "inference_estimate": estimate,
        "std_error": se,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= 0.0 <= high),
        "initial_estimate": estimate - MOVE,
    }


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = 1) -> replication.PhaseResult:
    """The subject's rows; with a compared record also the reference's inputs."""
    del n_jobs
    rows, samples, truths, plugins = [], [], [], []
    for replicate in range(replicates):
        for scenario in STUDY.scenarios:
            frame, truth = draw_scenario(scenario, n, replicate)
            rows.append(_row(SUBJECT, scenario, replicate, frame["x"].to_numpy(), 1e-9))
            prefix = dict(STUDY.reference_replicates).get(scenario, STUDY.replicates)
            if STUDY.reference is not None and replicate < prefix:
                sample = frame.copy()
                sample.insert(0, "row", range(len(sample)))
                sample.insert(0, "replicate", replicate)
                sample.insert(0, "scenario", scenario)
                samples.append(sample)
                truths += [
                    {"scenario": scenario, "replicate": replicate, "estimand": k, "truth": v}
                    for k, v in truth.items()
                ]
                plugins.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "estimand": f"ey[{scenario}]",
                        "subject_plugin": float(frame["x"].mean()) - MOVE,
                    }
                )
    frame = pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]
    if STUDY.reference is None:
        return replication.PhaseResult(rows=frame)
    return replication.PhaseResult(
        rows=frame,
        samples=pd.concat(samples, ignore_index=True),
        truths=pd.DataFrame(truths),
        plugins=pd.DataFrame(plugins),
    )


def compared() -> StudyRecord:
    """``STUDY`` as the registration commit would leave it."""
    return dataclasses.replace(
        STUDY,
        reference=REFERENCE,
        pending_reference=None,
        pending_reason="",
        **COMPARATOR_DECLARATION,
    )


def fake_reference_run(
    self: Any,
    here: Path,
    samples: Path,
    truths: Path,
    output: Path,
    *,
    cores: int,
    runner: str | None = None,
    env: Any = None,
) -> None:
    """What an R runner would write: the mean per replicate, and its untargeted value."""
    del self, here, cores, runner
    RUNS.append(dict(env or {}))
    frame = pd.read_csv(samples, float_precision="round_trip")
    wanted = pd.read_csv(truths)
    out = []
    for (scenario, replicate), group in frame.groupby(["scenario", "replicate"], sort=True):
        if not ((wanted["scenario"] == scenario) & (wanted["replicate"] == replicate)).any():
            raise AssertionError(f"{scenario} {replicate} has no truth row")
        x = group.sort_values("row")["x"].to_numpy()
        row = _row(REFERENCE, str(scenario), int(replicate), x, 0.0)
        row["gcomp_estimate"] = float(np.mean(x)) - MOVE
        out.append(row)
    pd.DataFrame(out).to_csv(output, index=False)
