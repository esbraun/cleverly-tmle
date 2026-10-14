"""The pre-declaration pairing gates of ``longitudinal-sdr`` (``X4-plan.md`` 6.2).

**G1.**  Three replicates of the primary scenarios at the declared ``n``.  Every row that
:data:`~tests.studies.canonical_longitudinal_sdr.EXACT_ROWS` names (the all-MTP plans of
``sdr_mtp`` and the all-policy plan ``mix`` of ``sdr_policy``) must agree with ``lmtp`` below
:data:`~tests.studies.canonical_longitudinal_sdr.SMOKE_GATE` on the estimate and the standard
error.  The other primary rows are printed, and are read under the default margins.

**G2.**  Saturated cell means on both sides of the discrete end-of-study and survival laws,
with the law's own mechanism and plans ``never``, ``always`` and the law's rule, at one and at
five folds.  Pooled and stratified cell means coincide on a discrete law, so every G2 row must
agree below the same gate.  ``lmtp`` fits them with ``SL.cellmeans``, which
``tests/canonical/lmtp_sdr_adapter.R`` defines.

The run also times each scenario single-process on both sides, for the declared budget.  It
writes ``gates.csv`` (one row per paired estimand) and ``timing.csv`` beside this file, and
exits non-zero when a gate fails.  It needs Docker and the ``cleverly-lmtp-crossfit:1.5.4``
image, so it runs on the local machine.

Usage::

    python -m tests.canonical.longitudinal_sdr.gates [--replicates 3] [--n 2000]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import SDR
from tests import discrete_law_longitudinal as end_law
from tests import discrete_law_survival as survival_law
from tests.studies import canonical_longitudinal_sdr as study
from tests.studies import ltmle_crossfit_properties as end_properties
from tests.studies import ltmle_survival_crossfit_properties as survival_properties
from tests.studies.canonical_ltmle import declared_regimens
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
IMAGE = "cleverly-lmtp-crossfit:1.5.4"
G1_SCENARIOS = (study.MTP, study.POLICY)
#: The paired scenarios read under margins; the smoke times them and prints their gaps.
MARGIN_SCENARIOS = (study.END_OF_STUDY, study.END_OF_STUDY_CROSSFIT, study.SURVIVAL)

END_PLANS = ("never", "always", "treat_if_l2")
SURVIVAL_PLANS = ("never", "always", "continue_if_l2")
G2_SCENARIOS: dict[str, tuple[str, int]] = {
    "gate_end_of_study_1": ("end_of_study", 1),
    "gate_end_of_study_5": ("end_of_study", 5),
    "gate_survival_1": ("survival", 1),
    "gate_survival_5": ("survival", 5),
}


def _g2_regimens(kind: str) -> dict[str, Any]:
    if kind == "end_of_study":
        return declared_regimens({label: end_law.REGIMEN_SPEC[label] for label in END_PLANS})
    return declared_regimens({label: survival_law.REGIMEN_SPEC[label] for label in SURVIVAL_PLANS})


def _g2_truth(kind: str) -> dict[str, float]:
    if kind == "end_of_study":
        names = [f"ey_regimen[{label}]" for label in END_PLANS] + [
            f"ate_regimen[{label} vs never]" for label in END_PLANS[1:]
        ]
        return {name: end_law.TRUTH[name] for name in names}
    return {
        name: value
        for name, value in survival_law.TRUTH.items()
        if any(f"[{label} " in name for label in SURVIVAL_PLANS)
    }


def g2_fit(frame: pd.DataFrame, kind: str, n_folds: int) -> Any:
    """Saturated cell means at every node, and the law's own mechanism, on one discrete law."""
    if kind == "end_of_study":
        mechanism, law, outcome = end_properties.KnownDiscreteMechanism, end_law, "Y"
    else:
        mechanism, law = survival_properties.KnownDiscreteMechanism, survival_law
        outcome = ["Y1", "Y2"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDR(
            _g2_regimens(kind),
            reference="never",
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=mechanism("treatment"),
            censoring_learner=mechanism("censoring"),
            n_folds=n_folds,
            g_bounds=study.G_BOUNDS,
            simultaneous=False,
            random_state=0,
        ).fit(
            frame,
            outcome=outcome,
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def g2_replicate(scenario: str, replicate: int, n: int) -> tuple[pd.DataFrame, list, list]:
    kind, n_folds = G2_SCENARIOS[scenario]
    seed = stream_seed(study.STUDY, "pairing_gate", scenario, replicate)
    if kind == "end_of_study":
        frame = end_properties.sample(end_law.PROBS, n, seed)
    else:
        frame = survival_properties.sample(survival_law.PROBS, n, seed)
    result = g2_fit(frame, kind, n_folds)
    truth = _g2_truth(kind)
    sample = frame.copy()
    for name, values in study.pairing_columns(result, _g2_regimens(kind)).items():
        sample[name] = values
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    rows = []
    for name, value in truth.items():
        estimate = result[name]
        rows.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "estimand": name,
                "truth": value,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
            }
        )
    return sample, truths, rows


def _timed(function: Any, *args: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - start


def collect(replicates: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    samples, truths, rows, timing = [], [], [], []
    for replicate in range(replicates):
        for scenario in (*G1_SCENARIOS, *MARGIN_SCENARIOS):
            (sample, truth, result_rows), seconds = _timed(study._replicate, scenario, replicate, n)
            samples.append(sample)
            truths.extend(truth)
            rows.extend(result_rows)
            timing.append({"side": "cleverly", "scenario": scenario, "seconds": seconds})
        for scenario in G2_SCENARIOS:
            (sample, truth, result_rows), seconds = _timed(g2_replicate, scenario, replicate, n)
            samples.append(sample)
            truths.extend(truth)
            rows.extend(result_rows)
            timing.append({"side": "cleverly", "scenario": scenario, "seconds": seconds})
    return (
        pd.concat(samples, ignore_index=True),
        pd.DataFrame(truths),
        pd.DataFrame(rows),
        timing,
    )


def run_reference(work: Path) -> float:
    start = time.perf_counter()
    subprocess.run(
        [
            "docker", "run", "--rm", "-e", "CLEVERLY_R_CORES=1",
            "-v", f"{work}:/work",
            "-v", f"{ROOT / 'tests' / 'canonical'}:/fixture:ro",
            "--entrypoint", "Rscript", IMAGE,
            "/fixture/longitudinal_sdr_runner.R",
            "/work/samples.csv.gz", "/work/truth.csv", "/work/lmtp.csv",
        ],
        check=True,
    )  # fmt: skip
    return time.perf_counter() - start


def compare(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        reference, on=["scenario", "replicate", "estimand"], suffixes=("_cleverly", "_lmtp")
    )
    if len(merged) != len(rows):
        raise AssertionError(f"{len(rows) - len(merged)} cleverly rows have no lmtp row")
    merged["estimate_gap"] = (merged["estimate_cleverly"] - merged["estimate_lmtp"]).abs()
    merged["std_error_gap"] = (merged["std_error_cleverly"] - merged["std_error_lmtp"]).abs()
    exact = {(scenario, name) for scenario, names in study.EXACT_ROWS.items() for name in names}
    merged["gate"] = [
        "G2" if scenario in G2_SCENARIOS else "G1" if (scenario, name) in exact else "margins"
        for scenario, name in zip(merged["scenario"], merged["estimand"], strict=True)
    ]
    gated = merged["gate"] != "margins"
    merged["passes"] = ~gated | (
        (merged["estimate_gap"] < study.SMOKE_GATE) & (merged["std_error_gap"] < study.SMOKE_GATE)
    )
    return merged.loc[
        :,
        [
            "gate", "scenario", "replicate", "estimand", "estimate_cleverly", "estimate_lmtp",
            "estimate_gap", "std_error_gap", "passes",
        ],
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n", type=int, default=study.PRIMARY_N)
    arguments = parser.parse_args(argv)
    samples, truths, rows, timing = collect(arguments.replicates, arguments.n)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        samples.to_csv(
            work / "samples.csv.gz", index=False, compression="gzip", lineterminator="\n"
        )
        truths.to_csv(work / "truth.csv", index=False, lineterminator="\n")
        seconds = run_reference(work)
        reference = pd.read_csv(work / "lmtp.csv", float_precision="round_trip")
    timing.append({"side": "lmtp", "scenario": "all", "seconds": seconds})
    table = compare(rows, reference)
    table.to_csv(HERE / "gates.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(timing).groupby(["side", "scenario"], as_index=False)["seconds"].mean().to_csv(
        HERE / "timing.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.groupby(["gate", "scenario"])[["estimate_gap", "std_error_gap"]].max())
    failed = table[~table["passes"]]
    if len(failed):
        print(f"{len(failed)} gated rows fail:\n{failed.to_string()}", file=sys.stderr)
        return 1
    print("every G1 and G2 row passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
