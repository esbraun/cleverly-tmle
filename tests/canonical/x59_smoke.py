"""The X59 smoke (``X59-plan.md`` section 4.1, local queue row A): one command per step.

Run from the repository root, with Docker for every step but ``identity`` and ``sizing``::

    python -m tests.canonical.x59_smoke g0 --out runs/x59/g0
    python -m tests.canonical.x59_smoke identity ltmle
    python -m tests.canonical.x59_smoke run ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke mutations ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke subject-mutation ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke memory categorical --container-mb 32000
    python -m tests.canonical.x59_smoke timing sdr --out runs/x59/timing
    python -m tests.canonical.x59_smoke sizing --timings runs/x59/timing --workers 12 \\
        --tier1 30 --tier3 17

``STUDY`` is ``ltmle`` (S1), ``categorical`` (S2) or ``sdr`` (S3), and for ``run``,
``mutations`` and ``timing`` also ``ltmle-parity`` or ``lmtp-parity`` (section 5).  A parity
record's subject phase runs first (``python tests/canonical/<dir>/regenerate.py``).  Every step prints what it
read and ends with ``PASS`` or ``FAIL`` (``memory``, ``timing`` and ``sizing`` print numbers
only).  A smoke run writes outside the committed directories, at the smoke prefix of
:data:`SMOKE_PREFIX` replicates, and the driver writes its parity table at the floor tolerance
(:func:`tests.studies.evidence.parity.smoke_declaration`), so its maxima are on disk for the
declaration commit.  No step reads a margins value, and no step changes a committed file.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "tests" / "canonical"

#: Study name -> (study module, regenerate directory, comparator family).
STUDIES = {
    "ltmle": ("tests.studies.canonical_long_horizon_ltmle", "long_horizon_ltmle", "ltmle"),
    "categorical": (
        "tests.studies.canonical_long_horizon_categorical_ltmle",
        "long_horizon_categorical_ltmle",
        "lmtp_tmle",
    ),
    "sdr": ("tests.studies.canonical_long_horizon_sdr", "long_horizon_sdr", "lmtp_sdr"),
    # The section 5 parity records, where the mutations run cheaply.  They commit no subject
    # rows before their first run, so the identity gate does not apply to them.
    "ltmle-parity": (
        "tests.studies.canonical_long_horizon_ltmle_parity",
        "long_horizon_ltmle_parity",
        "ltmle",
    ),
    "lmtp-parity": (
        "tests.studies.canonical_long_horizon_lmtp_parity",
        "long_horizon_lmtp_parity",
        "design",
    ),
}
#: The studies whose committed subject rows the identity gate refits.
IDENTITY_STUDIES = ("ltmle", "categorical", "sdr")
#: The smoke prefix: the first eight replicates of every paired scenario (section 4.1).
SMOKE_PREFIX = 8
#: The images G0 compares.
PINNED_IMAGE = "cleverly-lmtp:9996b04"
SHIPPED_IMAGE = "cleverly-lmtp-crossfit:1.5.4"
#: G0's tolerances: the validation-row score and each R learner against cleverly's fit.
SCORE_TOLERANCE = 1e-8
LEARNER_TOLERANCE = 1e-8
#: The identity gate's tolerance on estimate, standard error and initial estimate (B4).
IDENTITY_TOLERANCE = 1e-12


def _module(study: str) -> Any:
    return importlib.import_module(STUDIES[study][0])


def _docker(*arguments: str) -> None:
    command = ["docker", *arguments]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _verdict(passed: bool) -> int:
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


# ----------------------------------------------------------------------------- G0
def learner_cases(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G0's learner fixture: three node fits of S2's ``respond`` plan and cleverly's predictions.

    A binary horizon target and a fractional earlier target (quasibinomial), and the same
    fractional target by least squares, each on the followers of the plan with cleverly's design
    (:func:`tests.studies.long_horizon_common._design`).
    """
    from sklearn.linear_model import LinearRegression

    from tests import discrete_law_longitudinal_horizon as law_
    from tests.studies import long_horizon_common as common
    from tests.studies import long_horizon_reference as reference_
    from tests.studies.fractional_glm import QuasiBinomialGLM

    s2 = _module("categorical")
    scenario = dataclasses.replace(
        next(s for s in s2.SCENARIOS if s.name == "categorical_k3"), n=4000
    )
    frame = law_.draw(scenario.K, scenario.m, scenario.n, seed, scenario.law)
    K, m = scenario.K, scenario.m
    follow = np.cumprod(reference_.ratios(frame, "respond", K, m) > 0, axis=1).astype(bool)
    X_K = common._design(frame, K, m, ())
    y = np.nan_to_num(frame["Y"].to_numpy(float))
    horizon = QuasiBinomialGLM().fit(X_K[follow[:, K - 1]], y[follow[:, K - 1]])
    carried = horizon.predict(X_K)
    X_prev = common._design(frame, K - 1, m, ())
    rows = follow[:, K - 2]
    fractional = QuasiBinomialGLM().fit(X_prev[rows], carried[rows])
    linear = LinearRegression().fit(X_prev[rows], carried[rows])
    cases = [
        ("horizon", "quasibinomial", X_K, y, follow[:, K - 1], horizon.predict(X_K)),
        ("fractional", "quasibinomial", X_prev, carried, rows, fractional.predict(X_prev)),
        ("linear", "gaussian", X_prev, carried, rows, linear.predict(X_prev)),
    ]
    inputs, expected = [], []
    for name, family, X, target, fit_rows, predicted in cases:
        for role, mask in (("fit", fit_rows), ("predict", np.ones(len(X), dtype=bool))):
            block = pd.DataFrame(X[mask], columns=[f"x{j + 1}" for j in range(X.shape[1])])
            block.insert(0, "target", target[mask])
            block.insert(0, "row", np.flatnonzero(mask))
            block.insert(0, "role", role)
            block.insert(0, "family", family)
            block.insert(0, "case", name)
            inputs.append(block)
        expected.append(pd.DataFrame({"case": name, "row": np.arange(len(X)), "python": predicted}))
    return pd.concat(inputs, ignore_index=True), pd.concat(expected, ignore_index=True)


def g0(out: Path, *, build: bool = True) -> int:
    """G0: the image builds, the adapters agree across the two pins, the learners equal cleverly."""
    out.mkdir(parents=True, exist_ok=True)
    if build:
        _docker("build", "-t", PINNED_IMAGE, str(CANONICAL / "lmtp_9996b04"))
        _docker("build", "-t", SHIPPED_IMAGE, str(CANONICAL / "lmtp_crossfit"))
    inputs, expected = learner_cases()
    inputs.to_csv(out / "learner-input.csv", index=False, float_format="%.17g")
    mounts = ["-v", f"{CANONICAL}:/fixture:ro", "-v", f"{out.resolve()}:/out"]
    for image, modes in (
        (PINNED_IMAGE, ("packages", "image", "adapters", "learners")),
        (SHIPPED_IMAGE, ("packages", "adapters")),
    ):
        for mode in modes:
            _docker("run", "--rm", *mounts, image, "/fixture/lmtp_9996b04/g0.R", mode, "/out")
    checks: dict[str, bool] = {}
    image = pd.read_csv(out / "image.csv")
    score = float(image["max_validation_score"].iloc[0])
    checks[f"five-fold validation-row score {score:.3g} <= {SCORE_TOLERANCE:g}"] = (
        score <= SCORE_TOLERANCE
    )
    checks["cf_tmle fits are the 9996b04 data.table"] = bool(
        image["fits_class"].iloc[0] == "data.table"
    )
    pinned = pd.read_csv(out / "adapters-1.5.5.csv", dtype=str)
    shipped = pd.read_csv(out / "adapters-1.5.4.csv", dtype=str)
    for fit in ("tmle", "policy", "sdr_1", "sdr_5"):
        left = pinned.loc[pinned["fit"] == fit].reset_index(drop=True)
        right = shipped.loc[shipped["fit"] == fit].reset_index(drop=True)
        checks[f"{fit}: byte identity across the two images"] = left.equals(right)
    packages = {
        version: pd.read_csv(out / f"packages-{version}.csv").set_index("Package")["Version"]
        for version in ("1.5.4", "1.5.5")
    }
    common = packages["1.5.4"].index.intersection(packages["1.5.5"].index)
    drift = [
        f"{name} {packages['1.5.4'][name]} / {packages['1.5.5'][name]}"
        for name in common
        if packages["1.5.4"][name] != packages["1.5.5"][name]
    ]
    print("package versions that differ across the images:", drift or "none", flush=True)
    learned = pd.read_csv(out / "learners.csv", float_precision="round_trip")
    merged = expected.merge(learned, on=["case", "row"], how="left")
    worst = merged.assign(gap=(merged["python"] - merged["r_prediction"]).abs()).groupby("case")[
        "gap"
    ]
    for case, gap in worst.max().items():
        checks[f"learner {case}: max |R - cleverly| {gap:.3g} <= {LEARNER_TOLERANCE:g}"] = bool(
            np.isfinite(gap) and gap <= LEARNER_TOLERANCE
        )
    for name, passed in checks.items():
        print(("ok   " if passed else "FAIL ") + name, flush=True)
    return _verdict(all(checks.values()))


# ----------------------------------------------------------------------------- identity (B4)
def identity_report(
    study: str, *, replicates: int = SMOKE_PREFIX, names: Sequence[str] | None = None
) -> pd.DataFrame:
    """B4: refit the first ``replicates`` of each paired scenario Wald-only at HEAD.

    One row per scenario, replicate and Wald estimand: the absolute difference from the committed
    row in estimate, standard error and initial estimate, whether the written fold column is the
    fit's own, and for S2 and S3 whether the written ratios are the weights the fit divides by
    (:func:`tests.studies.canonical_longitudinal_sdr.check_ratios`).
    """
    from tests import discrete_law_longitudinal_horizon as law_
    from tests import longitudinal_horizon_plans as plans_
    from tests.studies import long_horizon_primary as primary
    from tests.studies import long_horizon_reference as reference_
    from tests.studies.canonical_longitudinal_sdr import check_ratios, plans_of
    from tests.studies.evidence.manifest import read_rows
    from tests.studies.evidence.seeds import replicate_seed

    module = _module(study)
    record = module.STUDY
    estimator = "SDR" if study == "sdr" else "LTMLE"
    comparator = STUDIES[study][2]
    committed = read_rows(record.artifact("replicates.parquet"))
    table = {scenario.name: scenario for scenario in module.SCENARIOS}
    out: list[dict[str, Any]] = []
    for name in names or list(module.REFERENCE_SCENARIOS):
        scenario = table[name]
        n = record.n_of(name)
        for replicate in range(replicates):
            seed = replicate_seed(record, name, replicate)
            frame = law_.draw(scenario.K, scenario.m, n, seed, scenario.law)
            result = primary.fit(frame, scenario, estimator, bootstrap=False)
            rows = pd.DataFrame(primary.rows(record, result, scenario, replicate, n))
            held = committed.loc[
                (committed["implementation"] == record.implementation)
                & (committed["scenario"] == name)
                & (committed["replicate"] == replicate)
            ].set_index("estimand")
            folds_match = bool(
                np.array_equal(
                    np.asarray(result.folds.assignment), reference_.folds(n, scenario.n_folds)
                )
            )
            ratios_match = True
            if comparator != "ltmle":
                declared = plans_.regimens(scenario.plans, scenario.K, scenario.m)
                for plan in plans_of(result, declared):
                    written = reference_.ratios(frame, str(plan.label), scenario.K, scenario.m)
                    try:
                        check_ratios(result, plan, written)
                    except AssertionError:
                        ratios_match = False
            for row in rows.itertuples(index=False):
                if row.estimand not in held.index:
                    continue
                kept = held.loc[row.estimand]
                out.append(
                    {
                        "scenario": name,
                        "replicate": replicate,
                        "estimand": row.estimand,
                        "estimate": abs(float(row.estimate) - float(kept["estimate"])),
                        "std_error": abs(float(row.std_error) - float(kept["std_error"])),
                        "initial_estimate": abs(
                            float(row.initial_estimate) - float(kept["initial_estimate"])
                        ),
                        "folds_match": folds_match,
                        "ratios_match": ratios_match,
                    }
                )
    return pd.DataFrame(out)


def identity_passed(report: pd.DataFrame) -> bool:
    """Whether every refit row equals its committed row and every written input matches its fit."""
    if report.empty:
        return False
    worst = float(report[["estimate", "std_error", "initial_estimate"]].to_numpy().max())
    return bool(
        worst <= IDENTITY_TOLERANCE and report["folds_match"].all() and report["ratios_match"].all()
    )


def identity(study: str, replicates: int) -> int:
    report = identity_report(study, replicates=replicates)
    print(
        report.groupby("scenario")[["estimate", "std_error", "initial_estimate"]].max().to_string(),
        flush=True,
    )
    print(
        "folds match:", bool(report["folds_match"].all()),
        "ratios match:", bool(report["ratios_match"].all()),
        flush=True,
    )  # fmt: skip
    return _verdict(identity_passed(report))


# ----------------------------------------------------------------------------- smoke runs
def smoke_run(
    study: str,
    out: Path,
    *,
    prefix: int,
    mutation: str = "none",
    jobs: int = 4,
    target: Path | None = None,
    cache: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """A reference-only run of the first ``prefix`` replicates, into ``out/<mutation>``.

    The mutation enters the driver's resume key (``REFERENCE_ENVIRONMENT``), so each one draws
    its own archive and runs its own comparator; no mutated run reads an unmutated checkpoint.
    """
    target = out / mutation if target is None else target
    variables = {"CLEVERLY_X59_MUTATION": mutation, **dict(env or {})}
    command = [
        sys.executable,
        str(CANONICAL / STUDIES[study][1] / "regenerate.py"),
        "--reference-only",
        "--reference-prefix",
        str(prefix),
        "--reference-jobs",
        str(jobs),
        "--output",
        str(target),
        *(("--cache", str(cache)) if cache is not None else ()),
    ]
    shown = " ".join(f"{name}={value}" for name, value in variables.items())
    print("$", shown, " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, **variables}, cwd=ROOT)
    return target


def _declared_only(
    committed: Mapping[str, float | None], declared: Mapping[str, float]
) -> dict[str, float | None]:
    """The committed G-P and G-C tolerances overridden by the smoke's; G1 is never declared."""
    from tests.studies.evidence import parity

    merged = {**dict(committed), **dict(declared)}
    return {gate: merged.get(gate) for gate in parity.DECLARED_GATES}


def run(study: str, out: Path, prefix: int, jobs: int) -> int:
    from tests.studies.evidence import parity

    target = smoke_run(study, out, prefix=prefix, jobs=jobs)
    table = pd.read_csv(target / "parity.csv", float_precision="round_trip")
    maxima = parity.smoke_maxima(table)
    print("smoke maxima:", json.dumps(maxima), flush=True)
    try:
        declared = {
            gate: parity.declared_tolerance(maxima[gate])
            for gate in parity.DECLARED_GATES
            if gate in maxima
        }
    except ValueError as error:
        print(f"FAIL: {error}", flush=True)
        return 1
    print("tolerances the M1 rule declares from them:", json.dumps(declared), flush=True)
    print("witness reach:", json.dumps(parity.witness_scenarios(table)), flush=True)
    (out / "declared-tolerances.json").write_text(json.dumps(declared, indent=1) + "\n")
    # G1's tolerance is declared already; a smoke G1 over it is a finding before any run.
    g1 = table.loc[table["gate"] == "G1", "max_abs_difference"]
    return _verdict(bool(g1.empty or float(g1.max()) <= parity.G1_TOLERANCE))


def mutations(study: str, out: Path, prefix: int, jobs: int) -> int:
    """M2: every declared mutation moves some gate by at least ten declared tolerances."""
    from tests.studies import long_horizon_reference as reference_
    from tests.studies.evidence import parity

    path = out / "declared-tolerances.json"
    if not path.exists():
        raise SystemExit(f"run `x59_smoke run {study}` first: {path} holds the tolerances")
    declared = json.loads(path.read_text())
    module = _module(study)
    declaration = parity.ParityDeclaration(
        cells=module.PARITY.cells, tolerance=_declared_only(module.PARITY.tolerance, declared)
    )
    detected: dict[str, bool] = {}
    for mutation in reference_.MUTATIONS:
        if mutation == "none":
            continue
        target = smoke_run(study, out, prefix=prefix, mutation=mutation, jobs=jobs)
        table = pd.read_csv(target / "parity.csv", float_precision="round_trip")
        detected[mutation] = parity.mutation_detected(table, declaration)
        print(f"{mutation}: maxima {json.dumps(parity.smoke_maxima(table))}", flush=True)
    print("detected:", json.dumps(detected), flush=True)
    return _verdict(all(detected.values()))


def subject_mutation(study: str, out: Path) -> int:
    """M2's ``cleverly``-side mutation: the node-K fluctuation skipped, against the smoke R rows.

    Refits the subject Wald-only under :func:`tests.studies.long_horizon_mutations.skip_update`
    on every replicate the unmutated smoke run paired, and reads the parity table against that
    run's reference rows at the declared tolerances.  ``LTMLE`` scenarios only: ``SDR`` fits no
    fluctuation, and its G1 cells (``mix``, ``coin``) are plans every observed unit follows, so a
    follower-mask mutation cannot move them; the R-side mutations cover ``SDR``.
    """
    from tests import discrete_law_longitudinal_horizon as law_
    from tests.canonical import regenerate
    from tests.studies import long_horizon_mutations as mutations_
    from tests.studies import long_horizon_primary as primary
    from tests.studies.evidence import parity
    from tests.studies.evidence.manifest import read_rows
    from tests.studies.evidence.seeds import replicate_seed

    module = _module(study)
    declared = json.loads((out / "declared-tolerances.json").read_text())
    declaration = parity.ParityDeclaration(
        cells=module.PARITY.cells, tolerance=_declared_only(module.PARITY.tolerance, declared)
    )
    smoke = out / "none"
    rows = read_rows(smoke / "replicates.parquet")
    here = CANONICAL / STUDIES[study][1]
    record = regenerate.compared_record(
        module, argparse.Namespace(reference_prefix=None, output=here), here
    )
    estimators = getattr(module, "ESTIMATORS", None) or {
        scenario.name: ("SDR" if study == "sdr" else "LTMLE") for scenario in module.SCENARIOS
    }
    table = {scenario.name: scenario for scenario in module.SCENARIOS}
    reference = rows.loc[rows["implementation"] == record.reference]
    refits = []
    for (name, replicate), _ in reference.groupby(["scenario", "replicate"]):
        if estimators[name] != "LTMLE":
            continue
        scenario = table[name]
        n = record.n_of(name)
        frame = law_.draw(
            scenario.K, scenario.m, n, replicate_seed(record, name, int(replicate)), scenario.law
        )
        with mutations_.skip_update({scenario.K}):
            result = primary.fit(frame, scenario, "LTMLE", bootstrap=False)
        refits += primary.rows(record, result, scenario, int(replicate), n)
    if not refits:
        print("no LTMLE scenario is paired; the cleverly-side mutation does not apply", flush=True)
        return 0
    mutated = pd.DataFrame(refits)
    kept = reference.loc[reference["scenario"].isin(set(mutated["scenario"]))]
    paired = pd.concat([mutated.loc[:, kept.columns.intersection(mutated.columns)], kept])
    cells = tuple(cell for cell in declaration.cells if cell.gate != "G-C")
    scoped = parity.ParityDeclaration(
        cells=tuple(c for c in cells if c.scenario in set(mutated["scenario"])),
        tolerance=declaration.tolerance,
    )
    result_table = parity.parity_table(paired, None, record=record, declaration=scoped)
    print(f"skip_update(K): maxima {json.dumps(parity.smoke_maxima(result_table))}", flush=True)
    return _verdict(parity.mutation_detected(result_table, scoped))


def _qbound_one(study: str, name: str, n: int, replicate: int) -> dict[str, Any]:
    from tests import discrete_law_longitudinal_horizon as law_
    from tests.studies import long_horizon_reference as reference_
    from tests.studies.evidence.seeds import replicate_seed

    module = _module(study)
    scenario = next(s for s in module.SCENARIOS if s.name == name)
    frame = law_.draw(
        scenario.K, scenario.m, n, replicate_seed(module.STUDY, name, replicate), scenario.law
    )
    low, high = 1.0, 0.0
    for plan in scenario.plans:
        _, a, b = reference_.plugin(frame, plan, scenario.K, scenario.m)
        low, high = min(low, a), max(high, b)
    bounds = reference_.Q_BOUNDS
    return {
        "study": study,
        "scenario": name,
        "n": n,
        "replicate": replicate,
        "low": low,
        "high": high,
        "outside": bool(low < bounds[0] or high > bounds[1]),
    }


def qbound_replicates(study: str, override: int | None) -> dict[str, int]:
    """The replicates M3 checks per paired scenario: the declared prefix, else its tier's largest.

    Before the declaration commit no prefix is declared, so a tier-2 scenario is checked at the
    largest count of ``TIER2_COUNTS`` and a tier-4 one at the largest of ``TIER4_COUNTS``: every
    prefix the sizing rule can choose is then inside the checked set.
    """
    from tests.studies.evidence import parity

    module = _module(study)
    record = module.STUDY
    declared = {
        **dict(record.reference_replicates),
        **dict(getattr(module, "COMPARATOR_DECLARATION", {}).get("reference_replicates", {})),
    }
    tiers = getattr(module, "REFERENCE_TIERS", {})
    paired = getattr(module, "REFERENCE_SCENARIOS", None) or record.scenarios
    out = {}
    for name in paired:
        if override is not None:
            out[name] = override
        elif name in declared:
            out[name] = int(declared[name])
        elif name in tiers:
            counts = parity.TIER2_COUNTS if tiers[name] == 2 else parity.TIER4_COUNTS
            out[name] = max(counts)
        else:
            out[name] = record.replicates
    return out


def qbound(study: str, replicates: Mapping[str, int], out: Path, jobs: int) -> int:
    """M3 before any run: no paired replicate's regression leaves ``Q_BOUNDS`` on a unit at risk.

    Checks every plan of each named scenario on its first ``replicates[scenario]`` replicates,
    with the arithmetic of :func:`tests.studies.long_horizon_reference.plugin`, restricted as the
    R stops are to the units that follow the plan up to the node.  Writes ``qbound-<study>.csv``.
    """
    from concurrent.futures import ProcessPoolExecutor

    module = _module(study)
    keys = [
        (study, name, module.STUDY.n_of(name), replicate)
        for name, count in replicates.items()
        for replicate in range(count)
    ]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_qbound_one, *zip(*keys, strict=True), chunksize=4))
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"qbound-{study}.csv", index=False)
    summary = frame.groupby("scenario").agg(
        replicates=("replicate", "size"),
        outside=("outside", "sum"),
        low=("low", "min"),
        high=("high", "max"),
    )
    print(summary.to_string(), flush=True)
    return _verdict(not bool(frame["outside"].any()))


# ----------------------------------------------------------------------------- memory and timing
def memory(study: str, container_mb: float, out: Path, prefix: int, jobs: int) -> int:
    """M7: a prefix smoke under ``docker run --memory``, its calibration peaks, and ``w``.

    The run keeps its scratch in ``out/memory-scratch``, where the driver's calibration writes
    one ``*.calibration.json`` per scenario run; the peak is the largest per-worker cost in them.
    """
    from tests.canonical import regenerate
    from tests.studies.evidence import parity

    scratch = out / "memory-scratch"
    smoke_run(
        study,
        out,
        prefix=prefix,
        jobs=jobs,
        target=out / "memory",
        cache=scratch,
        env={regenerate.DOCKER_MEMORY_ENV: f"{container_mb:.0f}m"},
    )
    peaks = []
    for path in sorted(scratch.glob("*.calibration.json")):
        measured = json.loads(path.read_text())
        # study_harness.R's calibration file: one entry per calibrated replicate group, each
        # with its measured per-worker cost in MB (peak growth or private memory after gc).
        groups = [g for g in measured.get("groups", []) if g.get("per_worker") is not None]
        for group in groups:
            print(f"{path.name} {group['id']}: per_worker {group['per_worker']:.0f} MB", flush=True)
        peaks += [float(group["per_worker"]) for group in groups]
    if not peaks:
        print(f"FAIL: no calibration measured a per-worker peak in {scratch}", flush=True)
        return 1
    peak = max(peaks)
    workers = parity.worker_count(container_mb, peak)
    print(f"peak {peak:.0f} MB per worker; w = {workers} in {container_mb:.0f} MB", flush=True)
    return 0


def timing(study: str, out: Path, replicates: int) -> int:
    """R seconds per replicate for each paired scenario, from the runner's ``fit_seconds``.

    One reference run per scenario on one core.  The runner times each replicate's fit alone
    (``horizon_timed``), so neither the container start nor the calibration fork counts.
    """
    from tests.canonical import regenerate

    module = _module(study)
    reference = _reference(study)
    here = CANONICAL / STUDIES[study][1]
    base = regenerate.compared_record(
        module, argparse.Namespace(reference_prefix=None, output=here), here
    )
    paired = base.reported(str(base.reference))
    rows = []
    for name in paired:
        record = dataclasses.replace(
            base,
            reference_scenarios={name: paired[name]},
            reference_replicates={name: replicates},
        )
        scratch = out / study / name
        scratch.mkdir(parents=True, exist_ok=True)
        inputs = regenerate.write_reference_inputs(module, record, base.n, scratch, jobs=4)
        output = scratch / "reference.csv"
        reference.run(
            here,
            inputs.scenarios[name].samples,
            inputs.scenarios[name].truth,
            output,
            cores=1,
            env=regenerate.reference_environment(module),
        )
        fitted = pd.read_csv(output).groupby("replicate")["fit_seconds"].first()
        seconds = float(fitted.mean())
        rows.append(
            {
                "study": study,
                "scenario": name,
                "tier": module.REFERENCE_TIERS[name],
                "cpu_hours": seconds / 3600.0,
            }
        )
        print(f"{name}: {seconds:.1f} s per replicate over {len(fitted)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"timing-{study}.csv", index=False)
    return 0


def _reference(study: str) -> Any:
    """The ``Reference`` the study's ``regenerate.py`` declares."""
    import runpy

    from tests.canonical import regenerate

    captured: dict[str, Any] = {}
    original = regenerate.main

    def capture(*args: Any, reference: Any, **kwargs: Any) -> None:
        captured["reference"] = reference

    regenerate.main = capture  # type: ignore[assignment]
    try:
        runpy.run_path(str(CANONICAL / STUDIES[study][1] / "regenerate.py"), run_name="__main__")
    finally:
        regenerate.main = original
    return captured["reference"]


def declared_sources(studies: Sequence[str]) -> int:
    """Rewrite each pending manifest's ``declared_sources_sha256`` to today's reference bytes.

    A pending study's manifest records the digests of the comparator sources it declares, and
    the fast tier compares them with the files.  An edit of a declared source (an R adapter, a
    runner, or a Dockerfile whose package hashes the smoke fills in) refreshes the digests in
    the same commit; the edit is result-neutral only while no reference has run.
    """
    from tests.studies.evidence.manifest import reference_hashes, write_lines

    for study in studies:
        here = CANONICAL / STUDIES[study][1]
        path = here / "manifest.json"
        if not path.exists():
            print(f"{study}: no committed manifest; nothing declared", flush=True)
            continue
        manifest = json.loads(path.read_text())
        if "comparator" not in manifest:
            raise SystemExit(f"{study} is compared already; its reference_sha256 is a run's")
        before = manifest["comparator"]["declared_sources_sha256"]
        after = reference_hashes(_reference(study).files(here))
        manifest["comparator"]["declared_sources_sha256"] = after
        write_lines(path, json.dumps(manifest, indent=2) + "\n")
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                old, new = before.get(name, "-")[:12], after.get(name, "-")[:12]
                print(f"{study}: {name} {old} -> {new}", flush=True)
    return 0


def sizing(timings: Path, workers: int, tier1: float, tier3: float) -> int:
    """Section 6: every paired scenario's prefix, from the smoke timings and ``w`` alone."""
    from tests.studies.evidence import parity

    frame = pd.concat(
        [pd.read_csv(path) for path in sorted(timings.glob("timing-*.csv"))], ignore_index=True
    )
    keys = [f"{row.study}:{row.scenario}" for row in frame.itertuples(index=False)]
    sized = parity.reference_sizes(
        dict(zip(keys, frame["cpu_hours"], strict=True)),
        dict(zip(keys, frame["tier"].astype(int), strict=True)),
        workers=workers,
        tier1_cpu_hours=tier1,
        tier3_cpu_hours=tier3,
    )
    print(
        f"B = {sized.budget:.0f} CPU-h; tier 2 at {sized.tier2}, tier 4 at {sized.tier4}; "
        f"{sized.cost:.1f} CPU-h, {sized.wall_hours:.1f} h on {sized.workers} workers",
        flush=True,
    )
    for key, count in sized.counts.items():
        print(f"  {key}: {count}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.canonical.x59_smoke")
    steps = parser.add_subparsers(dest="step", required=True)
    step = steps.add_parser("g0")
    step.add_argument("--out", type=Path, default=Path("runs/x59/g0"))
    step.add_argument("--no-build", action="store_true")
    step = steps.add_parser("identity")
    step.add_argument("study", choices=IDENTITY_STUDIES)
    step.add_argument("--replicates", type=int, default=SMOKE_PREFIX)
    for name in ("run", "mutations", "subject-mutation"):
        step = steps.add_parser(name)
        step.add_argument("study", choices=STUDIES)
        step.add_argument("--out", type=Path, required=True)
        step.add_argument("--prefix", type=int, default=SMOKE_PREFIX)
        step.add_argument("--jobs", type=int, default=4)
    step = steps.add_parser("qbound")
    step.add_argument("study", choices=STUDIES)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="per scenario; default: the record's prefix, else the largest the sizing rule picks",
    )
    step.add_argument("--jobs", type=int, default=4)
    step = steps.add_parser("memory")
    step.add_argument("study", choices=STUDIES)
    step.add_argument("--container-mb", type=float, required=True)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--prefix", type=int, default=SMOKE_PREFIX)
    step.add_argument("--jobs", type=int, default=4)
    step = steps.add_parser("timing")
    step.add_argument("study", choices=STUDIES)
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--replicates", type=int, default=2)
    step = steps.add_parser("declared-sources")
    step.add_argument("studies", nargs="+", choices=STUDIES)
    step = steps.add_parser("sizing")
    step.add_argument("--timings", type=Path, required=True)
    step.add_argument("--workers", type=int, required=True)
    step.add_argument("--tier1", type=float, required=True)
    step.add_argument("--tier3", type=float, required=True)
    arguments = parser.parse_args(argv)
    if arguments.step == "g0":
        return g0(arguments.out, build=not arguments.no_build)
    if arguments.step == "identity":
        return identity(arguments.study, arguments.replicates)
    if arguments.step == "run":
        return run(arguments.study, arguments.out, arguments.prefix, arguments.jobs)
    if arguments.step == "mutations":
        return mutations(arguments.study, arguments.out, arguments.prefix, arguments.jobs)
    if arguments.step == "subject-mutation":
        return subject_mutation(arguments.study, arguments.out)
    if arguments.step == "qbound":
        return qbound(
            arguments.study,
            qbound_replicates(arguments.study, arguments.replicates),
            arguments.out,
            arguments.jobs,
        )
    if arguments.step == "memory":
        return memory(
            arguments.study, arguments.container_mb, arguments.out, arguments.prefix, arguments.jobs
        )
    if arguments.step == "timing":
        return timing(arguments.study, arguments.out, arguments.replicates)
    if arguments.step == "declared-sources":
        return declared_sources(arguments.studies)
    return sizing(arguments.timings, arguments.workers, arguments.tier1, arguments.tier3)


if __name__ == "__main__":
    raise SystemExit(main())
