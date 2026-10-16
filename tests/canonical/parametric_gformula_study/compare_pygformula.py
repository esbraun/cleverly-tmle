"""Run the ``pygformula`` fixture comparison of ``parametric-gformula`` and write its artifacts.

From the repository root, with Docker running::

    python -m tests.canonical.parametric_gformula_study.compare_pygformula

The run has four steps.

1. Draw the fixture panels (``tests/canonical/parametric_gformula_study/fixtures.py``) and write them in
   the long form ``pygformula`` reads.
2. Build the comparator image from ``Dockerfile`` and ``requirements.txt``, then run
   ``run_pygformula.py`` in it on the panels.  The container runs the declared seeds on each
   panel and writes the coefficients and the per-seed estimates.
3. Fit ``cleverly`` on the same panels, one fit per seed, and read its coefficients.
4. Compare coefficients first and estimates second, write the artifacts, and exit with a nonzero
   code when any row fails.

``--reuse-results`` skips steps 1 to 2 and reads the committed ``pygformula-results.csv``, so a
change on the ``cleverly`` side re-runs the comparison without Docker.

The artifacts, beside this file, are ``pygformula-results.csv`` (the container's raw rows),
``pygformula-coefficients.csv``, ``pygformula-estimates.csv``, ``pygformula-seeds.csv`` (both
sides, every seed) and ``pygformula-run.json`` (the sources, the image and the settings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from tests.canonical.parametric_gformula_study import fixtures
from tests.canonical.regenerate import Reference
from tests.studies.canonical_parametric_gformula import PYGFORMULA

HERE = Path(__file__).resolve().parent

#: The comparator image, built from this directory's ``Dockerfile``.
REFERENCE = Reference(
    image=f"cleverly-pygformula-reference:{PYGFORMULA}",
    runner="run_pygformula.py",
    mount_runner=True,
    extra_files=("requirements.txt",),
    interpreter="python",
)

RESULTS = fixtures.PYGFORMULA_RESULTS
COEFFICIENTS = "pygformula-coefficients.csv"
ESTIMATES = "pygformula-estimates.csv"
SEED_TABLE = "pygformula-seeds.csv"
RUN_RECORD = "pygformula-run.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _image_id(image: str) -> str | None:
    found = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return found.stdout.strip() or None if found.returncode == 0 else None


def compare(output: Path, *, reuse_results: bool, names: list[str] | None) -> list[str]:
    """Run the comparison, write the artifacts into ``output``, and return the findings."""
    chosen = list(fixtures.FIXTURES) if names is None else names
    output.mkdir(parents=True, exist_ok=True)
    environment: dict[str, str] = {}
    if reuse_results:
        results = output / RESULTS
        print(f"reading the committed {RESULTS}", flush=True)
        record_path = output / RUN_RECORD
        if record_path.exists():
            environment = json.loads(record_path.read_text(encoding="utf-8"))["environment"]
    else:
        with tempfile.TemporaryDirectory(prefix="pygformula-") as scratch_name:
            scratch = Path(scratch_name)
            samples, spec = fixtures.write_inputs(scratch, chosen)
            results = scratch / RESULTS
            print(f"running pygformula {PYGFORMULA} in {REFERENCE.image}", flush=True)
            REFERENCE.run(HERE, samples, spec, results, cores=1)
            shutil.copyfile(results, output / RESULTS)
            environment = json.loads(
                results.with_name(f"{results.stem}.environment.json").read_text(encoding="utf-8")
            )
        results = output / RESULTS
    pygformula_coefficients, pygformula_estimates = fixtures.read_pygformula(results)
    print("fitting cleverly on the same panels", flush=True)
    cleverly_coefficients, cleverly_estimates = fixtures.cleverly_side(
        chosen, progress=lambda name: print(f"  {name}", flush=True)
    )

    coefficients = fixtures.coefficient_rows(cleverly_coefficients, pygformula_coefficients)
    estimates = fixtures.estimate_rows(cleverly_estimates, pygformula_estimates)
    seeds = pd.concat(
        [
            cleverly_estimates.assign(implementation="cleverly"),
            pygformula_estimates.assign(implementation="pygformula"),
        ],
        ignore_index=True,
    ).sort_values(["fixture", "estimand", "horizon", "implementation", "seed"], ignore_index=True)
    _write(coefficients, output / COEFFICIENTS)
    _write(estimates, output / ESTIMATES)
    _write(
        seeds.loc[:, ["fixture", "implementation", "estimand", "horizon", "seed", "value"]],
        output / SEED_TABLE,
    )
    record = {
        "pygformula": PYGFORMULA,
        "fixtures": chosen,
        "n": fixtures.N,
        "panel_seed": fixtures.PANEL_SEED,
        "seeds": list(fixtures.SEEDS),
        "gate_se": fixtures.GATE_SE,
        "coefficient_rtol": fixtures.COEFFICIENT_RTOL,
        "coefficient_atol": fixtures.COEFFICIENT_ATOL,
        "cleverly_draws": fixtures.CLEVERLY_DRAWS,
        "pygformula_n_simul": "n",
        "pygformula_sim_trunc": False,
        "environment": environment,
        "image": REFERENCE.image,
        "image_id": None if reuse_results else _image_id(REFERENCE.image),
        "sources_sha256": {path.name: _sha256(path) for path in REFERENCE.files(HERE)},
        "results_sha256": _sha256(output / RESULTS),
    }
    (output / RUN_RECORD).write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return fixtures.findings(coefficients, estimates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="read the committed pygformula-results.csv instead of running the container",
    )
    parser.add_argument(
        "--fixtures",
        help="comma-separated fixture names for a disposable smoke (default: all)",
    )
    arguments = parser.parse_args()
    names = arguments.fixtures.split(",") if arguments.fixtures else None
    found = compare(arguments.output, reuse_results=arguments.reuse_results, names=names)
    for line in found:
        print(f"FAILED {line}", flush=True)
    print(f"{len(found)} failed row(s)", flush=True)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
