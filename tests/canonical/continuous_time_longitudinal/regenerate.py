r"""Regenerate the continuous-time-longitudinal evidence (X6 part (e), E2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs``, ``--fresh``, ``--shard`` and ``--merge``, and nothing else
(:mod:`tests.canonical.declared_run`, which also refuses unless ``HEAD`` equals its upstream on a
clean tree and every thread variable is 1).  The record has no comparator, so the run starts no
container.

Sharded run.  The declared run may split across machines.  ``--shard i/k`` fits shard ``i`` of
``k``: every replicate ``r`` with ``r % k == i - 1`` of both cells, with the secondary fits
inside each ``day_cf`` replicate (``study.shard_keys``), so each of three shards holds a third
of every cell.  A shard keeps the declared guards above and the pilot's, writes its
checkpoints, its rows, ``run.log`` and ``shard.json`` (the shard, its resume key and commit,
each invocation's runtime, whether every key is fitted) into its own ``--output``, and copies
nothing into this directory.  A crashed shard resumes when run again into the same
``--output``.  ``--merge D1 ... Dk`` then refuses unless every shard is complete, every shard
ran at this commit under this run's resume key (one tree, declaration and stack), and the shards
hold every key exactly once, with no overlap and no gap.  It joins their checkpoints in the
scratch the shared driver keys by the run and runs the declared run from them, which fits
nothing and writes the artifacts and the manifest as an unsharded run does; the manifest adds
``generated_with.shards``, each shard's commit and runtime.  A replicate's seed is a function of
its key, so the artifacts are those of an unsharded run on the same draws, save the wall-clock
``seconds`` readings, which no rerun reproduces
(``tests/unit/test_continuous_longitudinal_shards.py``).  ``pilot-diagnosis.md`` gives each
shard's cost.  The laws, cells, learners,
sizes, rules and controls are those of :mod:`tests.studies.continuous_time_longitudinal` and its
readings module, declared from X6 plan rev 7, section 5.7, and checked by
``tests/unit/test_continuous_longitudinal_design.py``.

Pilot.  ``python -m tests.studies.continuous_time_longitudinal_pilot --jobs N --resume`` writes
``continuous_time_longitudinal_pilot.csv.gz`` (:mod:`tests.studies.continuous_time_longitudinal_pilot`
states each reading): the truths, the feasibility readings, the plug-in limit with the IPW-SD
witness, and 400 draws of each of ``day_cf``, ``cont_cf``, ``mechanism_correct`` and
``outcome_correct``.  This script refuses a declared run until that file is committed and agrees
with the declaration (:func:`pilot_refusals`):

* the law module's truths equal the pilot's truth rows (the full risks :math:`\psi_r`, the capped
  risks :math:`\psi^{(K)}` and the panel's risks), and the feasibility readings lie inside
  the plan's bounds (at least 30 expected events of each cause by day 10 among each regime's
  baseline followers, a 99th weight percentile at most 20);
* the plug-in limit separates: some day-30 risk of always or never sits at least 0.5 true-weight
  IPW SDs from its truth (review S8).  If not, :math:`L`'s coefficients in the cause hazards and
  the :math:`L` transition are raised and every truth and pilot reading is read again;
* every pilot cell holds its 400 draws, and the family rule's counts equal ``PRIMARY_REPLICATES``
  and ``MECHANISM_REPLICATES``.  When no pair of rungs carries the family, the rows whose pilot
  coverage is below 0.925 move to reported before the declaration, which is a change to the
  record, not to this script;
* each calibration row's plug-in pass at the primary count is at least 0.90, or its cell holds
  the 800-draw extension; after the extension a row still below 0.90 needs a diagnosis named in a
  committed ``pilot-diagnosis.md`` only when it departs beyond chance (its 99% SE-ratio interval
  excludes 1 or its coverage interval excludes 0.95, ``pilot.calibration_departure``), and is
  otherwise recorded there as a chance reading (review S4: a calibrated row extends with
  probability about 0.69);
* the at-target pass of the calibration pair at the primary count is at least 0.95, the
  plug-in control fires with probability at least 0.99 at ``mechanism_correct``'s count, and a
  shrunk-SE control leaves the band with probability at least 0.999 at a true SE ratio of 1
  (plan 5's preface; 0.99996 at 2,400).  The same power at the pilot's point ratio is a report in
  ``pilot-diagnosis.md``, never a refusal (review B1);
* no gated cell's pilot predicts, with posterior predictive probability above 0.10 (Jeffreys
  prior), that the declared run loses more fits than the failure bound of 0.01 allows;
* the pilot's seconds per fit project the declared run at no more than 200 CPU-h (plan 5.7's
  hold).

Cells (plan 5.7).  ``day_cf`` (CT-day, every regime, 30 gated rows, 12 of them rule rows) and
``cont_cf`` (CT, always and never, 18 gated rows), the default cross-fitted fit at n = 1,000,
under the primary rule.  ``mechanism_correct`` (12 gated rows and the plug-in control),
``outcome_correct`` and ``both_wrong`` (reported) and the X70 pairing ``ltmle_daily`` (reported)
run inside ``day_cf``'s replicates, each on its own stream and at its own count.

Deviation (declared before any run).  The pairing reads the first 400 ``day_cf`` draws on a
9-node panel whose nodes end at days 3, 6, 10, 13, 16, 20, 23, 26 and 30, in place of the plan's
1,000 draws on a 30-node daily panel, which E1's smoke found infeasible
(:mod:`tests.studies.continuous_time_longitudinal` states why).  The target days agree, but the
estimands differ by the coarsening: X70 is read against the panel's risk (``law.PANEL_TRUTHS``)
and the continuous engine against :math:`\psi_r`.

Budget.  ``pilot-diagnosis.md`` holds the smoke's measured seconds per fit and peak memory, the
budget table and the ``--jobs`` rule.

Red-cell readings (declared before any run).  The record is ``gated``: every primary row of
``day_cf`` and ``cont_cf``, every row of ``mechanism_correct``, the plug-in control, both
calibration rows and both shrunk-SE controls are gates, and any red verdict fails the run.  No
cell is re-sized, re-drawn or downgraded after a verdict is seen, and no law is changed to make a
row pass.

* A red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect.  It then
  goes to ``reporting`` with an owner in the roadmap's Red-cell owners table.  Any other red row
  is a defect suspect, and an unfixed defect never ends as an owner row (``guide/studies.md``,
  "Red cells").
* One exception is declared (plan rev 7, review S6).  With :math:`G` right and the continuation
  approximated, the influence-curve variance is expected to be conservative (plan 3.5.7), so an
  SE ratio above 1 is the expected direction.  A calibration row red only because its SE-ratio
  interval lies above 1.07, with its coverage interval at or above 0.95 and no defect found,
  reads as conservative: it goes to ``reporting`` with an owner, and the labels follow the
  primary rows.  A red on the low side stays a defect suspect.
* Diagnosis first reads where the red falls.  Red in ``cont_cf`` only points at the untied path
  (the reduced grid, the continuation held at many visit times).  Red in ``day_cf`` only points at
  the tie rule.  Red at day 30 in both, with day 10 green, points at the visit or covariate term,
  which grows with the visits.
* Red rule rows alone point at the rule's slot, its evaluation or the decision visits.  Red in
  ``mechanism_correct`` points at targeting and the slot weight (plan 3.5.6).  Each diagnosis
  runs paired arms on fresh seeds and reruns at most the affected cell, only for a code defect.
* A red rule row keeps only rules experimental (plan 8.10).  Static regimes are labelled from
  their own rows, so a red rule row holds back no static label.
* The visit cap (review S1).  The gates read the full risk :math:`\psi_r`, and the fit targets
  :math:`\psi^{(K^*_r)}`.  A red bias at a day-30 row of never, or at a difference against never,
  is first read against the capped truth at the fit's :math:`K^*_r` (``visit-cap.csv``): a row
  green against it is a cap red, which the share condition of plan 3.5.9 governs, not an engine
  defect suspect.
* A plug-in control that does not fire is an instrument failure: the run is void and goes back
  to the plan.  A shrunk-SE control passes when its SE-ratio interval lies below 0.93 or wholly
  below its calibration row's own interval.  The second reading carries a row that reads
  conservative (its SE-ratio interval above 1, the direction the bullet above expects), so a
  conservative row never voids the run through its control, and the union leaves no window near
  the switch point (review B1 and its re-check).  On such a row the control checks only that the
  replicates resolve a 15% shortfall from the row.  A shrunk-SE control that passes under neither
  reading is an instrument failure.
* ``outcome_correct``, ``sparse_calibration`` and ``both_wrong`` are reported and start no
  diagnosis.  ``ltmle_daily`` gates nothing either.  A mean paired difference that departs
  from its expected value, the panel's risk minus :math:`\psi_r`, by more than 0.5 of the
  continuous engine's SE (``diagnose`` in ``ltmle-pairing.csv``) starts a diagnosis of both
  engines on the same draws, so the coarsening does not spend the trigger (review S5).
* A replication that raises is never redrawn.  A ``DataError`` is a recordable fit failure
  bounded by 0.01 per cell; a ``day_cf`` replicate that raised also costs each secondary cell
  below its count one fit (``account_failures``).  A secondary fit records only an error
  ``recordable`` judges a fit failure, with its class in ``<cell>_error``; any other error stops
  the run (review S6).
* Part (e)'s support constants are read after the run from ``floors.csv``
  (``readings.floor_values``, plan 3.5.9) and set no verdict.
"""

import argparse
import dataclasses
import datetime
import json
import re
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.canonical import regenerate as driver
from tests.canonical.declared_run import outside, run, thread_refusals
from tests.diagnostics.rm18_shared import refusals, run_log
from tests.parallel import available_cores
from tests.studies import (
    continuous_time_longitudinal,
    continuous_time_longitudinal_law,
    continuous_time_longitudinal_pilot,
    continuous_time_longitudinal_properties,
)
from tests.studies.evidence import replication
from tests.studies.evidence.manifest import merged_shards, write_csv
from tests.studies.evidence.registry import StudyRecord

#: The refusal while the pilot holds no primary draw.  The first pilot's primary draws were
#: dropped when the default treatment model gained its baseline indicator (2026-11-25), so the
#: truth, capped, panel, feasibility and plug-in limit rows stand and the four cells rerun.
FRESH_PILOT = "the pilot holds no primary draws"
#: The plan's hold: a projection above this stops the run before it starts (plan 5.7).
HOLD_CPU_HOURS = 200.0
#: The smallest power of the plug-in control at the pilot's plug-in biases (review m6, re-check
#: R2).  A floor of 0.80 would accept a control that voids the declared run with probability 0.2.
#: At 0.99 the floor needs a pilot bias near 0.32 SD at 1,600 fits, while the separation of 0.5
#: IPW SDs predicts a true bias of 0.5 or more; a 400-draw pilot reads it with an SE near 0.05, so
#: a chance refusal is improbable unless the true bias lies within about 0.1 of the threshold,
#: unlike B1, whose null value sat on its threshold.
CONTROL_POWER = 0.99
#: The smallest power of each shrunk-SE control, at a true SE ratio of 1 (plan 5's preface).
SHRUNK_POWER = 0.999


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    study = continuous_time_longitudinal
    laws = continuous_time_longitudinal_law
    pilot = continuous_time_longitudinal_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    truths = pilot.truth_table(rows)
    if not truths:
        out.append("the pilot holds no truth rows; run --kinds truth")
    elif truths != laws.TRUTH_RISKS:
        out.append("the law module's TRUTH_RISKS differ from the pilot's truth rows")
    capped = pilot.capped_table(rows)
    if not capped:
        out.append("the pilot holds no capped truths; run --kinds truth_capped")
    elif capped != laws.CAPPED_RISKS:
        out.append("the law module's CAPPED_RISKS differ from the pilot's capped rows")
    panel = pilot.panel_table(rows)
    if not panel:
        out.append("the pilot holds no panel truths; run --kinds truth_panel")
    elif panel != laws.PANEL_RISKS:
        out.append("the law module's PANEL_RISKS differ from the pilot's panel rows")
    out.extend(pilot.feasibility_refusals(rows))
    separated = pilot.separated(rows)
    if separated is None:
        out.append("the pilot holds no plug-in limit; run --kinds plugin_limit")
    elif not separated:
        out.append(
            "no day-30 risk of always or never separates its plug-in limit by 0.5 IPW SDs; "
            "raise L's coefficients in the cause hazards and the L transition"
        )
    if not any(pilot.draws(rows, cell) for cell in pilot.CELLS):
        out.append(f"{FRESH_PILOT}; run --kinds primary")
    else:
        for cell in pilot.CELLS:
            if pilot.draws(rows, cell) < pilot.DRAWS:
                out.append(
                    f"the pilot holds {pilot.draws(rows, cell)} draws of {cell}, not {pilot.DRAWS}"
                )
    if out:
        return out
    counts = pilot.counts(rows)
    if counts is None:
        binding = pilot.binding_rows(rows)
        if binding:
            out.append(
                "no pair of rungs carries the family; move these rows to reported before the "
                f"declaration: {binding}"
            )
        else:
            out.append(
                "no pair of rungs carries the family and no row's plug-in coverage is below "
                f"{pilot.REPORTED_COVERAGE}; diagnose the family in {pilot.DIAGNOSIS.name}"
            )
        return out
    primary, mechanism = counts
    if (primary, mechanism) != (study.PRIMARY_REPLICATES, study.MECHANISM_REPLICATES):
        out.append(
            f"the pilot sets {primary} primary and {mechanism} mechanism_correct replicates and "
            f"the record declares {study.PRIMARY_REPLICATES} and {study.MECHANISM_REPLICATES}"
        )
    diagnosed = pilot.DIAGNOSIS.read_text(encoding="utf-8") if pilot.DIAGNOSIS.exists() else ""
    checks = pilot.calibration_checks(rows, primary)
    for cell, passed in zip(checks["cell"], checks["plug_in_pass"], strict=True):
        if passed >= pilot.PLUG_IN_PASS:
            continue
        if pilot.draws(rows, cell) < pilot.EXTENSION_DRAWS:
            out.append(
                f"{cell}'s calibration row passes with plug-in probability {passed:.3f}; extend "
                f"its pilot to {pilot.EXTENSION_DRAWS} draws"
            )
        elif (
            pilot.calibration_departure(rows, cell)
            and f"{cell}/{study.CALIBRATION_ROW}" not in diagnosed
        ):
            # Only a confirmed departure needs a diagnosis; a row below 0.90 whose SE-ratio
            # interval holds 1 and whose coverage interval holds 0.95 is a chance reading,
            # recorded in the diagnosis note (review S4).
            out.append(
                f"{cell}/{study.CALIBRATION_ROW} departs from calibration on the extended "
                f"pilot; commit its diagnosis in {pilot.DIAGNOSIS.name}"
            )
    if pilot.at_target_pass(primary) < 0.95:
        out.append(
            f"the calibration pair passes at target with {pilot.at_target_pass(primary):.3f}"
        )
    power = pilot.control_power(rows, mechanism)
    if power < CONTROL_POWER:
        out.append(f"the plug-in control fires with probability {power:.3f}; back to the plan")
    if pilot.shrunken_power(primary) < SHRUNK_POWER:
        out.append(
            f"a shrunk-SE control leaves the band at target with {pilot.shrunken_power(primary):.4f}"
        )
    out.extend(pilot.failure_refusals(rows, primary, mechanism))
    projected = sum(pilot.budget(rows, primary, mechanism).values())
    if projected > HOLD_CPU_HOURS:
        out.append(f"the declared run projects {projected:.0f} CPU-h, above the plan's hold of 200")
    return out


def refuse_undeclared(argv: Sequence[str]) -> None:
    """Refuse a declared run until the committed pilot agrees with the declaration."""
    if any(argument.startswith("--replicates") for argument in argv):
        return  # a smoke run
    reasons = pilot_refusals()
    if reasons:
        raise SystemExit("refused: " + "; ".join(reasons))


# ------------------------------------------------------------------ the sharded run

HERE = Path(__file__).resolve().parent
#: The flags a declared run takes.
FLAGS = ("--output", "--jobs", "--fresh", "--shard", "--merge")
#: The shards of the declared run: three 4-core cloud machines, each under 24 hours.
SHARDS = 3
#: What a shard writes into its ``--output``, and nothing else.
SHARD_FILE = "shard.json"
SHARD_ROWS = "rows.csv.gz"
SHARD_FAILURES = "failures.csv"
SHARD_CHECKPOINTS = "checkpoints"
SHARD_OUTPUTS = (SHARD_FILE, SHARD_ROWS, SHARD_FAILURES, SHARD_CHECKPOINTS, "run.log")
#: The checkpoint label of the primary loop (``replication.fit_primary``).
PRIMARY_LABEL = "primary"


def parse_shard(text: str) -> tuple[int, int]:
    """``(index, total)`` of ``--shard i/k``, with ``k`` at least 2."""
    match = re.fullmatch(r"([0-9]+)/([0-9]+)", text)
    if match is None:
        raise SystemExit(f"refused: --shard takes i/k, such as 1/{SHARDS}, not {text!r}")
    index, total = int(match.group(1)), int(match.group(2))
    if total < 2 or not 1 <= index <= total:
        raise SystemExit(f"refused: --shard {text} is not one of 1/k to k/k with k of 2 or more")
    return index, total


def run_record(replicates: int, n: int) -> StudyRecord:
    """The record a run at ``replicates`` and ``n`` declares, as the shared driver builds it."""
    base = continuous_time_longitudinal.STUDY
    return dataclasses.replace(
        base, replicates=replicates, n=n, scenario_n=driver._declared_sizes(base, n)
    )


def run_key(replicates: int, n: int) -> dict[str, Any]:
    """The shared driver's resume key of a run: declaration, commit, tree, sizes and versions."""
    return driver.run_key(
        run_record(replicates, n), driver.reference_environment(continuous_time_longitudinal)
    )


def run_scratch(key: dict[str, Any]) -> Path:
    """The scratch directory the shared driver keys by ``key``, where its checkpoints sit."""
    return driver.runs_root() / continuous_time_longitudinal.STUDY.slug / driver.digest(key)[:12]


def checkpoint_files(replicates: int, shard: tuple[int, int] | None) -> list[Path]:
    """The checkpoint of every key of the run or of one shard, under a checkpoint root."""
    implementation = continuous_time_longitudinal.STUDY.implementation
    return [
        replication.checkpoint_file(implementation, PRIMARY_LABEL, cell, replicate)
        for cell, replicate in continuous_time_longitudinal.shard_keys(replicates, shard)
    ]


def _arguments(argv: Sequence[str], smoke: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument("--fresh", action="store_true")
    if smoke:
        parser.add_argument("--replicates", type=int)
        parser.add_argument("--n", type=int, default=continuous_time_longitudinal.PRIMARY_N)
        parser.add_argument("--allow-failures", action="store_true")
    known, unknown = parser.parse_known_args(list(argv))
    if unknown:
        raise SystemExit(f"refused: a declared run takes {', '.join(FLAGS)} only, not {unknown}")
    if known.output is None:
        raise SystemExit("refused: a run needs a scratch --output outside the repository")
    if not smoke:
        known.replicates = continuous_time_longitudinal.PRIMARY_REPLICATES
        known.n = continuous_time_longitudinal.PRIMARY_N
    elif known.replicates == continuous_time_longitudinal.PRIMARY_REPLICATES:
        raise SystemExit(
            "refused: a declared run passes no --replicates, and a smoke run passes a count "
            f"other than the declared {continuous_time_longitudinal.PRIMARY_REPLICATES}"
        )
    known.declared = not smoke
    return known


def _guard(smoke: bool) -> None:
    """The declared run's guards (rule R6): a clean pushed tree and every thread variable 1."""
    if smoke:
        return
    refused = refusals(False) + thread_refusals()
    if refused:
        raise SystemExit("refused:\n  " + "\n  ".join(refused))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def shard_run(shard: tuple[int, int], argv: Sequence[str], *, smoke: bool) -> None:
    """Fit one shard's keys into its own ``--output``; copy nothing into the repository.

    The output holds the shard's checkpoints, its rows (``rows.csv.gz``), its failure records
    when a fit failed, ``run.log`` and ``shard.json``, which names the shard, the resume key it
    ran under (its commit among it) and each invocation's runtime, and says whether every key
    is fitted.  A crashed shard resumes when run again into the same ``--output``; ``--fresh``
    discards what the output holds first.
    """
    known = _arguments(argv, smoke)
    output = outside(known.output, "smoke" if smoke else "declared")
    _guard(smoke)
    key = run_key(known.replicates, known.n)
    label = f"{shard[0]}/{shard[1]}"
    stamp = output / SHARD_FILE
    if known.fresh and output.exists():
        for name in SHARD_OUTPUTS:
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    held = sorted(path.name for path in output.iterdir()) if output.exists() else []
    runs: list[dict[str, Any]] = []
    if held:
        strangers = sorted(set(held) - set(SHARD_OUTPUTS))
        if strangers or not stamp.exists():
            raise SystemExit(f"refused: the shard's --output {output} holds {strangers or held}")
        written = json.loads(stamp.read_text(encoding="utf-8"))
        if written["shard"] != label or written["key"] != driver.digest(key):
            raise SystemExit(
                f"refused: {output} holds shard {written['shard']} at commit {written['head']} "
                f"under another resume key; pass --fresh to discard it"
            )
        runs = list(written["runs"])
    output.mkdir(parents=True, exist_ok=True)
    expected = checkpoint_files(known.replicates, shard)
    record = {
        "slug": continuous_time_longitudinal.STUDY.slug,
        "shard": label,
        "index": shard[0],
        "total": shard[1],
        "replicates": known.replicates,
        "n": known.n,
        "declared": known.declared,
        "head": key["git"]["head"],
        "key": driver.digest(key),
        "run_key": driver.canonical(key),
        "keys": len(expected),
        "complete": False,
        "runs": runs,
        "seconds": sum(entry["seconds"] for entry in runs),
    }
    _write_json(stamp, record)
    started = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    title = f"{continuous_time_longitudinal.STUDY.slug} shard {label}"
    print(f"shard {label}: {len(expected)} keys into {output}", flush=True)
    with (
        run_log(output, title, argv=list(sys.argv)),
        replication.checkpoints(output / SHARD_CHECKPOINTS),
    ):
        result = continuous_time_longitudinal.draw_and_fit(
            replicates=known.replicates, n=known.n, n_jobs=known.jobs, shard=shard
        )
        write_csv(result.rows, output / SHARD_ROWS, compression={"method": "gzip", "mtime": 0})
        (output / SHARD_FAILURES).unlink(missing_ok=True)
        if result.failures is not None:
            write_csv(result.failures, output / SHARD_FAILURES)
    runs.append({"started": started, "seconds": round(time.perf_counter() - clock, 1)})
    missing = [path for path in expected if not (output / SHARD_CHECKPOINTS / path).exists()]
    record.update(
        complete=not missing, runs=runs, seconds=round(sum(entry["seconds"] for entry in runs), 1)
    )
    _write_json(stamp, record)
    print(f"shard {label} wrote {output}; complete: {not missing}", flush=True)


def _shard_record(directory: Path) -> dict[str, Any]:
    stamp = directory / SHARD_FILE
    if not stamp.exists():
        raise SystemExit(f"refused: {directory} holds no {SHARD_FILE}; it is not a shard's output")
    return dict(json.loads(stamp.read_text(encoding="utf-8")))


def merge_refusals(
    directories: Sequence[Path], key: dict[str, Any], replicates: int, *, declared: bool
) -> list[str]:
    """Why the shards in ``directories`` do not make one run under ``key``, if they do not.

    Every shard must be complete, at this run's resume key (one commit, one tree, one
    declaration, one stack), one of the same ``k`` shards, and the shards must hold every key
    of the run exactly once.
    """
    records = [_shard_record(directory) for directory in directories]
    out = []
    heads = sorted({record["head"] for record in records})
    if len(heads) > 1:
        out.append(f"the shards ran at different commits: {heads}")
    if key["git"]["head"] not in heads:
        out.append(f"the shards ran at {heads}, and this merge runs at {key['git']['head']}")
    for directory, record in zip(directories, records, strict=True):
        if record["key"] != driver.digest(key):
            fields = sorted(
                name
                for name in {*record["run_key"], *driver.canonical(key)}
                if record["run_key"].get(name) != driver.canonical(key).get(name)
            )
            out.append(f"shard {record['shard']} in {directory} ran under another key: {fields}")
        if not record["complete"]:
            out.append(f"shard {record['shard']} in {directory} is incomplete")
        if record["declared"] != declared:
            kind = "declared" if record["declared"] else "smoke"
            out.append(f"shard {record['shard']} in {directory} is a {kind} shard")
    totals = sorted({int(record["total"]) for record in records})
    if len(totals) > 1:
        out.append(f"the shards split the run in different ways: {totals}")
    total = totals[-1]
    indexes = [int(record["index"]) for record in records]
    doubled = sorted({index for index in indexes if indexes.count(index) > 1})
    if doubled:
        out.append(f"the shards overlap: shard {doubled} of {total} is given twice")
    gaps = sorted(set(range(1, total + 1)) - set(indexes))
    if gaps:
        out.append(f"the shards leave a gap: shard {gaps} of {total} is missing")
    if out:
        return out
    held: dict[Path, Path] = {}
    for directory, record in zip(directories, records, strict=True):
        shard = (int(record["index"]), total)
        for path in checkpoint_files(replicates, shard):
            source = directory / SHARD_CHECKPOINTS / path
            if not source.exists():
                out.append(f"shard {record['shard']} in {directory} misses the checkpoint {path}")
                break
            held[path] = source
    every = checkpoint_files(replicates, None)
    if not out and set(held) != set(every):
        out.append(f"the shards hold {len(held)} of the run's {len(every)} keys")
    return out


def merge_run(directories: Sequence[Path], argv: Sequence[str], *, smoke: bool) -> None:
    """Build the one record from complete shards, as an unsharded run writes it.

    The shards' checkpoints are joined in the scratch the shared driver keys by this run's
    resume key, and the run is then the unsharded declared run (or smoke run) resumed from
    them: it fits nothing, and reads, gates and publishes as a single run does.  Each
    replicate's seed is a function of its key, so the artifacts are byte-identical to an
    unsharded run on the same draws; the manifest adds ``generated_with.shards``, each shard's
    name, commit and runtime.
    """
    known = _arguments(argv, smoke)
    _guard(smoke)
    key = run_key(known.replicates, known.n)
    reasons = merge_refusals(directories, key, known.replicates, declared=known.declared)
    if reasons:
        raise SystemExit("refused: " + "; ".join(reasons))
    scratch = run_scratch(key)
    if scratch.exists() and any(scratch.iterdir()):
        if not known.fresh:
            raise SystemExit(f"refused: the run's scratch {scratch} is not empty; pass --fresh")
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    driver._claim_scratch(scratch, key, fresh=False)
    root = scratch / driver.CHECKPOINTS
    held = sorted(
        ((_shard_record(path), path) for path in directories), key=lambda pair: pair[0]["index"]
    )
    for record, directory in held:
        shard = (int(record["index"]), int(record["total"]))
        for path in checkpoint_files(known.replicates, shard):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / SHARD_CHECKPOINTS / path, target)
    records = [record for record, _ in held]
    shards = [
        {
            "shard": record["shard"],
            "commit": record["head"],
            "seconds": record["seconds"],
            "invocations": len(record["runs"]),
        }
        for record in records
    ]
    print(f"merging {len(records)} shards through {scratch}", flush=True)
    forwarded = [argument for argument in argv if argument != "--fresh"]
    sys.argv = [sys.argv[0], *forwarded]
    with merged_shards(shards):
        run(
            continuous_time_longitudinal,
            continuous_time_longitudinal_properties,
            here=HERE,
        )


def main(argv: Sequence[str]) -> None:
    """A declared run, one shard of it (``--shard i/k``), or the merge of its shards.

    A run that passes ``--replicates`` is a smoke run and is refused none of the declared
    guards; the shared driver refuses its declared count.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--merge", nargs="+", type=Path, default=None)
    known, rest = parser.parse_known_args(list(argv))
    smoke = any(argument.startswith("--replicates") for argument in rest)
    if known.shard is not None and known.merge is not None:
        raise SystemExit("refused: a run is a shard or a merge, not both")
    if not smoke:
        _arguments(rest, smoke=False)
        refuse_undeclared(rest)
    if known.shard is not None:
        shard_run(parse_shard(known.shard), rest, smoke=smoke)
    elif known.merge is not None:
        merge_run(known.merge, rest, smoke=smoke)
    else:
        sys.argv = [sys.argv[0], *rest]
        run(continuous_time_longitudinal, continuous_time_longitudinal_properties, here=HERE)


if __name__ == "__main__":
    main(sys.argv[1:])
