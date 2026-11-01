"""The X59 smoke (``X59-plan.md`` section 4.1, local queue row A): one command per step.

Run from the repository root, with Docker for every step but ``identity``, ``qbound``,
``declared-sources`` and ``sizing``::

    python -m tests.canonical.x59_smoke g0 --out runs/x59/g0
    python -m tests.canonical.x59_smoke identity ltmle
    python -m tests.canonical.x59_smoke qbound ltmle --out runs/x59/qbound --jobs 4
    python -m tests.canonical.x59_smoke run ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke mutations ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke subject-mutation ltmle --out runs/x59/ltmle
    python -m tests.canonical.x59_smoke memory categorical --container-mb 32000 \\
        --out runs/x59/memory
    python -m tests.canonical.x59_smoke timing sdr --out runs/x59/timing
    python -m tests.canonical.x59_smoke sizing --timings runs/x59/timing \\
        --memory runs/x59/memory --contention 1.3
    python -m tests.canonical.x59_smoke declared-sources ltmle categorical sdr

``STUDY`` is ``ltmle`` (S1), ``categorical`` (S2) or ``sdr`` (S3), and for every step but
``identity`` also ``ltmle-parity`` or ``lmtp-parity`` (section 5).  A parity record commits no
subject before its declared run, so its smoke fits the subject on the prefix into the run's
``subject`` directory and pairs the reference with it (``--subject``).  ``memory`` keeps each
study under ``--out/<study>``, so one ``--out`` serves every study.
Every step prints what it read and ends with ``PASS`` or ``FAIL`` (``timing``, ``sizing`` and
``declared-sources`` print numbers only).  A smoke run writes outside the committed
directories, at the smoke prefix of :data:`SMOKE_PREFIX` replicates, and the driver writes its
parity table at the floor tolerance (:func:`tests.studies.evidence.parity.smoke_declaration`),
so its maxima are on disk for the declaration commit; ``run`` fails a maximum that would
declare a tolerance above :data:`tests.studies.evidence.parity.TOLERANCE_CEILING`.  No step
reads a margins value, and no step but ``declared-sources`` changes a committed file: the
commit that fills the Dockerfile hashes, or edits any declared R source, runs it to refresh
the pending manifests' declared digests in the same commit.
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
#: The parity records: no subject rows are committed before their first run, so a smoke fits
#: their subject first (:func:`smoke_subject`) and pairs the reference with it.
PARITY_STUDIES = ("ltmle-parity", "lmtp-parity")
#: The smoke prefix: the first eight replicates of every paired scenario (section 4.1).
SMOKE_PREFIX = 8
#: The images G0 compares.
PINNED_IMAGE = "cleverly-lmtp:9996b04"
SHIPPED_IMAGE = "cleverly-lmtp-crossfit:1.5.4"
#: G0's tolerances: the validation-row score and each R learner against cleverly's fit.
SCORE_TOLERANCE = 1e-8
LEARNER_TOLERANCE = 1e-8
#: The identity gate's bound on estimate, standard error and initial estimate (B4): the
#: re-execution tests' ``|refit - committed| <= atol + rtol |committed|``, because a refit on
#: another platform differs in the last bits, and an MSM term, a projection of several plans'
#: estimates, carries a few of their ULPs (row 50: 1.16e-12 on an intercept of about 0.53).
IDENTITY_RTOL = 1e-12
IDENTITY_ATOL = 1e-12
IDENTITY_QUANTITIES = ("estimate", "std_error", "initial_estimate")


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


#: G0's follower-mapping cases: (case, study, scenario, plan), each fitted through the real
#: lmtp dispatch at its declared size and replicate 0.  ``respond`` is deterministic,
#: ``mixlast`` has a known stochastic node (the replicated branch), ``low`` runs SDR.
FOLLOWER_CASES = (
    ("s2_respond", "categorical", "categorical_k5", "respond"),
    ("s2_mixlast", "categorical", "categorical_k5", "mixlast"),
    ("s3_low", "sdr", "sdr_categorical_k5", "low"),
)


def follower_cases(out: Path) -> dict[str, dict[str, Any]]:
    """Write each case's archive sample and return cleverly's follower counts and node-K fit.

    The counts are :func:`tests.studies.long_horizon_reference.followers` at every node; the
    node-K regression is ``QuasiBinomialGLM`` on the node-K followers with ``cleverly``'s
    design, whose target is the observed outcome in both estimators.
    """
    from sklearn.base import clone

    from tests import discrete_law_longitudinal_horizon as law_
    from tests.studies import long_horizon_common as common
    from tests.studies import long_horizon_reference as reference_
    from tests.studies.fractional_glm import QuasiBinomialGLM

    rows = []
    expected: dict[str, dict[str, Any]] = {}
    for case, study, name, plan in FOLLOWER_CASES:
        module = _module(study)
        scenario = next(s for s in module.SCENARIOS if s.name == name)
        n = module.STUDY.n_of(name)
        frame = module.reference_sample(name, n, 0).frame.copy()
        frame.insert(0, "row", range(len(frame)))
        sample = f"followers-input-{case}.csv"
        frame.to_csv(out / sample, index=False, float_format="%.17g")
        rows.append({"case": case, "sample": sample, "plan": plan, "comparator": STUDIES[study][2]})
        _, fol, _ = reference_.followers(frame, plan, scenario.K, scenario.m)
        policy = law_.policy_nodes(plan, scenario.K)
        design = common._design(frame, scenario.K, scenario.m, policy)
        kept = fol[scenario.K]
        target = np.nan_to_num(frame["Y"].to_numpy(float))
        model = clone(QuasiBinomialGLM()).fit(design[kept], target[kept])
        expected[case] = {
            "units": {t: int(fol[t].sum()) for t in fol},
            "coefficients": np.concatenate([model.intercept_, model.coef_[0]]),
            "K": scenario.K,
        }
    pd.DataFrame(rows).to_csv(out / "followers-cases.csv", index=False)
    return expected


def follower_checks(out: Path, expected: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """Each case's R follower counts at every node, and its node-K coefficients at 1e-8."""
    checks: dict[str, bool] = {}
    for case, held in expected.items():
        log = pd.read_csv(out / f"followers-{case}.csv")
        for node, units in held["units"].items():
            seen = sorted(set(log.loc[log["node"] == node, "units"]))
            checks[f"{case} node {node}: R fits {seen} followers, cleverly {units}"] = seen == [
                units
            ]
        final = log.loc[log["node"] == held["K"], "coefficients"]
        gaps = [
            float(np.max(np.abs(np.array(text.split(";"), dtype=float) - held["coefficients"])))
            if len(text.split(";")) == len(held["coefficients"])
            else np.inf
            for text in final
        ]
        gap = max(gaps, default=np.inf)
        checks[f"{case} node {held['K']}: max |R - cleverly| coefficient {gap:.3g}"] = bool(
            gap <= LEARNER_TOLERANCE
        )
    return checks


def package_drift(out: Path) -> list[str]:
    """The packages both images hold at different versions, ``lmtp`` aside."""
    packages = {
        version: pd.read_csv(out / f"packages-{version}.csv").set_index("Package")["Version"]
        for version in ("1.5.4", "1.5.5")
    }
    common = packages["1.5.4"].index.intersection(packages["1.5.5"].index)
    return [
        f"{name} {packages['1.5.4'][name]} / {packages['1.5.5'][name]}"
        for name in common
        if name != "lmtp" and packages["1.5.4"][name] != packages["1.5.5"][name]
    ]


def g0(out: Path, *, build: bool = True) -> int:
    """G0: the image builds, the adapters agree across the two pins, the learners equal cleverly."""
    out.mkdir(parents=True, exist_ok=True)
    if build:
        # The pinned image is built FROM the shipped one, so the shipped one comes first.
        _docker("build", "-t", SHIPPED_IMAGE, str(CANONICAL / "lmtp_crossfit"))
        _docker("build", "-t", PINNED_IMAGE, str(CANONICAL / "lmtp_9996b04"))
    inputs, expected = learner_cases()
    followed = follower_cases(out)
    inputs.to_csv(out / "learner-input.csv", index=False, float_format="%.17g")
    mounts = ["-v", f"{CANONICAL}:/fixture:ro", "-v", f"{out.resolve()}:/out"]
    for image, modes in (
        (PINNED_IMAGE, ("packages", "image", "adapters", "learners", "followers")),
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
    # The pinned image is the shipped one plus origami and lmtp, so no common package drifts;
    # a drift here explains a byte-identity failure above and fails G0 itself.
    drift = package_drift(out)
    print("package versions that differ across the images:", drift or "none", flush=True)
    checks["no package but lmtp differs across the images"] = not drift
    checks.update(follower_checks(out, followed))
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
                        **{
                            quantity: abs(float(getattr(row, quantity)) - float(kept[quantity]))
                            for quantity in IDENTITY_QUANTITIES
                        },
                        **{
                            f"committed_{quantity}": float(kept[quantity])
                            for quantity in IDENTITY_QUANTITIES
                        },
                        "folds_match": folds_match,
                        "ratios_match": ratios_match,
                    }
                )
    return pd.DataFrame(out)


def identity_failures(report: pd.DataFrame) -> list[str]:
    """Each refit cell outside the bound, and each written input that differs from its fit."""
    failures = []
    for row in report.itertuples(index=False):
        cell = f"{row.scenario} r{row.replicate} {row.estimand}"
        for quantity in IDENTITY_QUANTITIES:
            gap = float(getattr(row, quantity))
            bound = IDENTITY_ATOL + IDENTITY_RTOL * abs(
                float(getattr(row, f"committed_{quantity}"))
            )
            if not gap <= bound:
                failures.append(f"{cell} {quantity}: |refit - committed| {gap:.3g} > {bound:.3g}")
        if not row.folds_match:
            failures.append(f"{cell}: the written folds are not the fit's")
        if not row.ratios_match:
            failures.append(f"{cell}: the written ratios are not the fit's weights")
    return failures


def identity_passed(report: pd.DataFrame) -> bool:
    """Whether every refit row is its committed row and every written input matches its fit."""
    return bool(not report.empty and not identity_failures(report))


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
    for failure in identity_failures(report):
        print("FAIL", failure, flush=True)
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
    subject = (
        smoke_subject(study, out, prefix=prefix, jobs=jobs) if study in PARITY_STUDIES else None
    )
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
        *(("--subject", str(subject), "--replicates", str(prefix)) if subject is not None else ()),
    ]
    shown = " ".join(f"{name}={value}" for name, value in variables.items())
    print("$", shown, " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, **variables}, cwd=ROOT)
    return target


def smoke_subject(study: str, out: Path, *, prefix: int, jobs: int) -> Path:
    """A parity record's subject phase on the smoke prefix, into ``out/subject``, once.

    The record commits no subject rows before its declared run, so the smoke fits them here
    (Python only: the record is pending, so no reference runs) and the reference-only runs pair
    them through ``--subject``.  The fit is deterministic, so a finished one is reused.
    """
    target = out / "subject"
    if (target / "manifest.json").exists():
        print(f"reusing the smoke subject in {target}", flush=True)
        return target
    command = [
        sys.executable,
        str(CANONICAL / STUDIES[study][1] / "regenerate.py"),
        "--replicates",
        str(prefix),
        "--skip-properties",
        "--jobs",
        str(jobs),
        "--output",
        str(target),
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
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
    # Every paired cell's maximum, so the declaration commit reads where each gate's comes from.
    cells = table.loc[table["gate"] != "witness"]
    print(
        cells.loc[:, ["gate", "scenario", "estimand", "paired_replicates", "max_abs_difference"]]
        .sort_values(["gate", "max_abs_difference"], ascending=[True, False])
        .to_string(index=False),
        flush=True,
    )
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
def memory_readings(study: str, scratch: Path, *, cores: int) -> pd.DataFrame:
    """Each scenario's calibration in ``scratch``, and the read chunk and workers it plans.

    The driver writes one ``<leaf>.calibration.json`` per scenario run, its leaf ending in
    ``reference-results__<scenario>``, and plans the real run from it with
    :func:`tests.canonical.regenerate.stream_plan`; this reads the same files the same way.
    """
    from tests.canonical import regenerate

    rows = []
    for path in sorted(scratch.rglob("*.calibration.json")):
        scenario = path.name.removesuffix(".calibration.json").split("reference-results__")[-1]
        measured = json.loads(path.read_text())
        groups = [g for g in measured.get("groups", []) if g.get("per_worker") is not None]
        if not groups:
            continue
        heaviest = max(groups, key=lambda group: float(group["per_worker"]))
        available = regenerate.calibration_available(measured["memory"])
        parent = float(measured["parent_private_mb"])
        chunk, cost = regenerate.stream_plan(
            float(heaviest["per_worker"]), parent, available, cores
        )
        rows.append(
            {
                "study": study,
                "scenario": scenario,
                "per_worker_mb": float(heaviest["per_worker"]),
                "peak_growth_mb": float(heaviest["peak_growth"]),
                "private_after_gc_mb": float(heaviest["private_after_gc"]),
                "parent_mb": parent,
                "available_mb": available,
                "planned_mb": cost,
                "chunk": chunk,
                "workers": regenerate.harness_workers(cores, cost, available),
            }
        )
    return pd.DataFrame(rows)


def memory(study: str, container_mb: float, out: Path, prefix: int, jobs: int) -> int:
    """M7: a prefix smoke under ``docker run --memory``, and each scenario's memory plan.

    Each study keeps its own run and scratch, ``out/<study>/memory`` and
    ``out/<study>/memory-scratch``, where the driver's calibration writes one
    ``*.calibration.json`` per scenario.  Writes ``out/memory-<study>.csv``: per scenario, the
    calibration's readings and the read chunk and workers the run plans from them, which the
    ``sizing`` step reads.
    """
    from tests.canonical import regenerate

    home = out / study
    scratch = home / "memory-scratch"
    smoke_run(
        study,
        home,
        prefix=prefix,
        jobs=jobs,
        target=home / "memory",
        cache=scratch,
        env={regenerate.DOCKER_MEMORY_ENV: f"{container_mb:.0f}m"},
    )
    frame = memory_readings(study, scratch, cores=jobs)
    if frame.empty:
        print(f"FAIL: no calibration measured a per-worker peak in {scratch}", flush=True)
        return 1
    print(frame.to_string(index=False), flush=True)
    frame.to_csv(out / f"memory-{study}.csv", index=False)
    return 0


def reference_tier(study: str, name: str) -> int:
    """A paired scenario's section 6 tier: its study's ``REFERENCE_TIERS``, else tier 3.

    The parity records declare no tiers: every scenario of theirs is tier 3, at the record's
    fixed count.
    """
    from tests.studies.evidence import parity

    module = _module(study)
    tiers = getattr(module, "REFERENCE_TIERS", None)
    if tiers is None:
        return parity.PARITY_TIER
    if name not in tiers:
        raise SystemExit(f"{study}: {name} is paired but has no reference tier")
    return int(tiers[name])


def timing(study: str, out: Path, replicates: int) -> int:
    """R seconds per replicate for each paired scenario, from the runner's ``fit_seconds``.

    One reference run per scenario on one core.  The runner times each replicate's fit alone
    (``horizon_timed``), so neither the container start nor the calibration fork counts.  A
    parity record's scenarios are tier 3 at the record's fixed count.
    """
    from tests.canonical import regenerate
    from tests.studies.evidence import parity

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
        tier = reference_tier(study, name)
        rows.append(
            {
                "study": study,
                "scenario": name,
                "tier": tier,
                "replicates": base.replicates if tier == parity.PARITY_TIER else None,
                "seconds": seconds,
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


def sizing_costs(timings: Path, memory: Path) -> dict[str, Any]:
    """Every timed scenario's :class:`~tests.studies.evidence.parity.ScenarioCost`.

    Joins ``timing-*.csv`` in ``timings`` with ``memory-*.csv`` in ``memory`` on study and
    scenario, and refuses a timed scenario that has no memory plan.
    """
    from tests.studies.evidence import parity

    def read(directory: Path, pattern: str) -> pd.DataFrame:
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise SystemExit(f"no {pattern} in {directory}")
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    timed = read(timings, "timing-*.csv")
    planned = read(memory, "memory-*.csv").loc[:, ["study", "scenario", "workers", "chunk"]]
    joined = timed.merge(planned, on=["study", "scenario"], how="left")
    missing = joined.loc[joined["workers"].isna(), ["study", "scenario"]]
    if not missing.empty:
        names = ", ".join(f"{row.study}:{row.scenario}" for row in missing.itertuples())
        raise SystemExit(f"no memory plan for {names}; run the memory step of its study")
    return {
        f"{row.study}:{row.scenario}": parity.ScenarioCost(
            seconds=float(row.seconds),
            workers=int(row.workers),
            chunk=int(row.chunk),
            tier=int(row.tier),
            replicates=None if pd.isna(row.replicates) else int(row.replicates),
        )
        for row in joined.itertuples(index=False)
    }


def sizing(timings: Path, memory: Path, tier1_hours: float, contention: float) -> int:
    """Section 6: every paired scenario's prefix, from the smoke timings and memory plans."""
    from tests.studies.evidence import parity

    costs = sizing_costs(timings, memory)
    sized = parity.reference_sizes(costs, tier1_hours=tier1_hours, contention=contention)
    hours = ", ".join(f"tier {tier} {value:.1f} h" for tier, value in sorted(sized.hours.items()))
    print(
        f"tier 2 at {sized.tier2}, tier 4 at {sized.tier4}; {hours}; "
        f"{sized.wall_hours:.1f} of {parity.RUN_HOURS:.0f} wall hours",
        flush=True,
    )
    for key, count in sized.counts.items():
        cost = costs[key]
        print(
            f"  {key}: {count} (tier {cost.tier}, {cost.seconds:.1f} s, "
            f"{cost.workers} workers, chunk {cost.chunk})",
            flush=True,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line of every smoke step."""
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
    step.add_argument("--timings", type=Path, required=True, help="the timing-*.csv directory")
    step.add_argument("--memory", type=Path, required=True, help="the memory-*.csv directory")
    step.add_argument("--tier1-hours", type=float, default=0.0)
    step.add_argument(
        "--contention",
        type=float,
        default=1.0,
        help="a smoke run's measured wall over its modelled wall (parity.scenario_wall_seconds)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
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
    return sizing(arguments.timings, arguments.memory, arguments.tier1_hours, arguments.contention)


if __name__ == "__main__":
    raise SystemExit(main())
