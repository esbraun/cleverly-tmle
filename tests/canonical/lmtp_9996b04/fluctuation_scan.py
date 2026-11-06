"""Write the node-K fluctuation scan's inputs, run it, and count the divergences per cell.

Evidence for the plans ``long-horizon-categorical-ltmle`` leaves unpaired (its
``UNPAIRED_PLANS``); no run reads it.  For each one-fold ``lmtp_tmle`` scenario of S2 and of
``long-horizon-lmtp-parity``, each replicate's comparator input (``reference_sample``, the
archive the driver writes) is cut to the columns the scan reads, and ``fluctuation_scan.R``
fits lmtp's node-K fluctuation on raw and on rescaled weights.  Needs ``Rscript`` with
``data.table``; lmtp is not needed, because the scan restates its one ``glm`` call.

    python -m tests.canonical.lmtp_9996b04.fluctuation_scan OUT --replicates 100 --jobs 4
    python -m tests.canonical.lmtp_9996b04.fluctuation_scan OUT --all-plans --replicates 6 \
        --parity-replicates 6
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDIES = {
    "categorical": "tests.studies.canonical_long_horizon_categorical_ltmle",
    "lmtp-parity": "tests.studies.canonical_long_horizon_lmtp_parity",
}


def _cells(study: str, all_plans: bool) -> dict[str, list[str]]:
    """The ``lmtp_tmle`` plans of each scenario: the paired ones, or with ``all_plans`` every one."""
    import importlib
    import re

    module = importlib.import_module(STUDIES[study])
    declared = getattr(module, "REFERENCE_SCENARIOS", None)
    comparators = getattr(module, "COMPARATORS", {})
    out: dict[str, list[str]] = {}
    for scenario in module.SCENARIOS:
        if scenario.n_folds != 1 or comparators.get(scenario.name, "lmtp_tmle") != "lmtp_tmle":
            continue
        if declared is None:  # a parity record pairs every plan of every scenario
            paired = set(scenario.plans)
        else:
            estimands = declared.get(scenario.name, ())
            paired = {p for name in estimands for p in re.split(r"\[| vs |\]", name)[1:-1]}
        plans = [p for p in scenario.plans if all_plans or p in paired]
        if plans:
            out[scenario.name] = plans
    return out


def _write(study: str, scenario: str, plans: list[str], replicate: int, out: Path) -> Path:
    import importlib
    import warnings

    from tests.studies import long_horizon_reference as reference_

    warnings.simplefilter("ignore")
    reference_.plugins = lambda frame, scenario: {}  # the plug-in is not read here
    module = importlib.import_module(STUDIES[study])
    record = module.STUDY
    frame = module.reference_sample(scenario, record.n_of(scenario), replicate).frame
    K = int(frame["design__K"].iloc[0])
    columns = ["design__K", f"C{K}", "Y"] + [
        f"ratio__{plan}__{t}" for plan in plans for t in range(1, K + 1)
    ]
    kept = frame.loc[:, columns].assign(scenario=scenario, replicate=replicate)
    path = out / f"{study}__{scenario}__{replicate:04d}.csv"
    kept.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.canonical.lmtp_9996b04.fluctuation_scan")
    parser.add_argument("out", type=Path)
    parser.add_argument("--replicates", type=int, default=100, help="per S2 scenario")
    parser.add_argument(
        "--parity-replicates", type=int, default=500, help="per parity scenario: it pairs 500"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--all-plans", action="store_true")
    arguments = parser.parse_args(argv)
    inputs = arguments.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    keys = [
        (study, scenario, plans, replicate, inputs)
        for study in STUDIES
        for scenario, plans in _cells(study, arguments.all_plans).items()
        for replicate in range(
            arguments.parity_replicates if study == "lmtp-parity" else arguments.replicates
        )
    ]
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        list(pool.map(_write, *zip(*keys, strict=True)))
    result = arguments.out / "fluctuation-scan.csv"
    subprocess.run(
        ["Rscript", str(HERE / "fluctuation_scan.R"), str(inputs), str(result)], check=True
    )
    table = pd.read_csv(result)
    summary = table.groupby(["scenario", "plan"]).agg(
        replicates=("replicate", "size"),
        diverged=("diverged", "sum"),
        max_weight=("max_weight", "max"),
    )
    print(summary.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
