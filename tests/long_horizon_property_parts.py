"""The part driver of the long-horizon property phase: split it, run it, merge it.

The property phase of each long-horizon study (``S1``, ``S2``, ``S3``;
:mod:`tests.studies.long_horizon_properties`) is a fixed, ordered list of *parts*.  A part is
:data:`BLOCK` replicates of one fit set.  Each key draws from its own seed, so any machine may
fit any part and two machines with the same head and package versions write the same
checkpoints for it.

Commands (run from the repository root with ``PYTHONPATH=.``)::

    python -m tests.long_horizon_property_parts list --study S1
    python -m tests.long_horizon_property_parts split --study S1 --share 0.2
    python -m tests.long_horizon_property_parts run --study S1 [--from I] [--to J] [--jobs N]
    python -m tests.long_horizon_property_parts status --study S1
    python -m tests.long_horizon_property_parts merge --study S1 --source DIR
    python -m tests.long_horizon_property_parts smoke --study S1 [--n 2000]

``list`` prints each part's index, estimated CPU hours and running total.  ``split --share S``
prints the first index at which the running total passes ``S`` of the whole, so one machine
runs ``--to`` that index and the other ``--from`` it.  ``run`` fits the keys of parts
``[I, J)`` into the checkpoints of the study's regeneration scratch: the directory that
``tests/canonical/long_horizon_*/regenerate.py`` computes from the same resume key (head,
declaration and package versions), or ``--cache DIR``.  It claims the scratch as the driver
does, so a scratch of another key is refused.  A part whose keys are all checkpointed is
skipped, and a killed run resumes from the per-replicate checkpoints of
:func:`~tests.studies.evidence.replication.fit_properties`.  ``status`` counts the finished parts.
``merge --source DIR`` copies the property checkpoints of another machine's scratch into this
one; it refuses a scratch whose resume key differs.  The regeneration then fits nothing that a
part finished: its property loops read the same checkpoint identities.  ``smoke`` fits one
replicate of the first fit set of every family at ``min(n, declared n)``, outside the scratch.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from tests import long_horizon_design as D
from tests.studies import long_horizon_properties as shared
from tests.studies.evidence import replication

#: Replicates per part.
BLOCK = 100
#: The n of a smoke fit, where the declared n is larger.
SMOKE_N = 2_000
#: The property module of each study tag.
MODULES = {
    "S1": "tests.studies.long_horizon_ltmle_properties",
    "S2": "tests.studies.long_horizon_categorical_ltmle_properties",
    "S3": "tests.studies.long_horizon_sdr_properties",
}


@dataclass(frozen=True)
class Part:
    """:data:`BLOCK` replicates of one fit set.

    Parameters
    ----------
    index : int
        The part's position in the study's list.
    family : str
        The property family.
    fit_set : str
        The fit set's name.
    start : int
        The first replicate.
    stop : int
        One past the last replicate.
    hours : float
        The estimated CPU hours.
    """

    index: int
    family: str
    fit_set: str
    start: int
    stop: int
    hours: float


def module(tag: str) -> ModuleType:
    """The property module of a study tag."""
    return importlib.import_module(MODULES[tag])


def parts(tag: str) -> list[Part]:
    """Every part of a study's property phase, in the order the phase fits them."""
    replicates = module(tag).REPLICATES
    out: list[Part] = []
    for loop in shared.loops(tag, replicates):
        for fit_set in shared.fit_sets(tag):
            if fit_set.family != loop.family:
                continue
            per = fit_set.cost() / 3600.0
            total = replicates[loop.family]
            for start in range(0, total, BLOCK):
                stop = min(start + BLOCK, total)
                out.append(
                    Part(len(out), loop.family, fit_set.name, start, stop, per * (stop - start))
                )
    return out


def scratch(tag: str, cache: Path | None) -> tuple[Path, dict[str, Any]]:
    """The regeneration scratch and resume key of a study, as its driver computes them."""
    from tests.canonical import regenerate

    study = D.STUDIES[tag]
    declared = dataclasses.replace(
        study.STUDY,
        replicates=study.PRIMARY_REPLICATES,
        n=study.PRIMARY_N,
        scenario_n=regenerate._declared_sizes(study.STUDY, study.PRIMARY_N),
    )
    key = regenerate.run_key(declared)
    root = cache if cache else regenerate.runs_root() / declared.slug / regenerate.digest(key)[:12]
    return root, key


def _implementation(tag: str) -> str:
    return str(D.STUDIES[tag].STUDY.implementation)


def _done(tag: str, root: Path, part: Part) -> int:
    """How many replicates of ``part`` the scratch holds a checkpoint of."""
    label = f"properties/{part.family}"
    directory = root / "checkpoints" / _implementation(tag) / label
    replicates = module(tag).REPLICATES
    n = shared.fit_sets_by_name(tag)[part.fit_set].n
    found = 0
    for replicate in range(part.start, part.stop):
        key = (tag, part.fit_set, replicate, replicates[part.family], n)
        ident = replication._digest((_implementation(tag), label, shared.cell_of(key), replicate))
        found += (directory / f"{ident}.pkl.gz").exists()
    return found


def run_parts(
    tag: str, indices: Sequence[int], root: Path, key: Mapping[str, Any], jobs: int
) -> None:
    """Fit the keys of each part into ``root``'s checkpoints, skipping finished parts."""
    from tests.canonical import regenerate

    every = parts(tag)
    replicates = module(tag).REPLICATES
    root.mkdir(parents=True, exist_ok=True)
    with regenerate.host_lock(root):
        regenerate._claim_scratch(root, key, fresh=False)
        for index in indices:
            part = every[index]
            if _done(tag, root, part) == part.stop - part.start:
                continue
            print(
                f"[{index}] {part.fit_set} replicates {part.start}..{part.stop - 1} "
                f"~{part.hours:.2f} CPU-h",
                flush=True,
            )
            loop = next(
                item for item in shared.loops(tag, replicates) if item.family == part.family
            )
            chosen = [
                k for k in loop.keys if k[1] == part.fit_set and part.start <= k[2] < part.stop
            ]
            # A fresh context per part: a loop's label is used once per run.
            with (
                replication.checkpoints(root / "checkpoints"),
                replication.recording(_implementation(tag)) as recorded,
            ):
                replication.fit_properties(chosen, shared.fit_key, **loop.options(jobs))
            if recorded.failures is not None:
                print(f"[{index}] recorded failures: {len(recorded.failures)}", flush=True)
            print(f"[{index}] done", flush=True)
    print("complete", flush=True)


def merge(tag: str, root: Path, source: Path) -> int:
    """Copy another scratch's property checkpoints into ``root``; refuse another resume key."""
    mine, theirs = root / "_key.json", source / "_key.json"
    if not theirs.exists() or not mine.exists():
        raise SystemExit("both scratches need a resume key: run a part in each first")
    if mine.read_bytes() != theirs.read_bytes():
        raise SystemExit(
            f"{source} was written under another resume key (head, declaration or package "
            "versions differ); its checkpoints cannot be reused here"
        )
    copied = 0
    base = source / "checkpoints" / _implementation(tag)
    for path in sorted(base.glob("properties/*/*.pkl.gz")):
        target = root / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"copied {copied} checkpoints from {source}")
    return copied


def smoke(tag: str, n: int) -> pd.DataFrame:
    """One replicate of the first fit set of every family at ``min(n, declared n)``."""
    out = []
    for family in shared.FAMILIES:
        chosen = [fit_set for fit_set in shared.fit_sets(tag) if fit_set.family == family]
        if not chosen:
            continue
        fit_set = chosen[0]
        key: shared.Key = (tag, fit_set.name, 0, 1, min(n, fit_set.n))
        loop = shared.Loop(family, [key], f"properties/{family}", shared.derived_cells(tag))
        with replication.recording(_implementation(tag)) as recorded:
            values = replication.fit_properties(
                [key], shared.fit_key, **{**loop.options(2), "checkpoint": None}
            )
        if recorded.failures is not None:
            raise RuntimeError(f"{tag} {family}: {recorded.failures['message'].iloc[0]}")
        rows = pd.DataFrame(values[0])
        print(f"{tag} {family} {fit_set.name} n={key[4]}: {len(rows)} rows", flush=True)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.long_horizon_property_parts")
    parser.add_argument("command", choices=("list", "split", "run", "status", "merge", "smoke"))
    parser.add_argument("--study", choices=tuple(MODULES), required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--from", dest="start", type=int, default=0)
    parser.add_argument("--to", type=int)
    parser.add_argument("--share", type=float, default=0.2)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--n", type=int, default=SMOKE_N)
    arguments = parser.parse_args(argv)
    tag = arguments.study
    if arguments.command == "smoke":
        print(smoke(tag, arguments.n).to_string(max_rows=200))
        return
    every = parts(tag)
    if arguments.command == "list":
        total = 0.0
        for part in every:
            total += part.hours
            print(
                f"{part.index:5d} {part.hours:7.3f} {total:8.1f}  {part.fit_set} "
                f"[{part.start}, {part.stop})"
            )
        print(f"{len(every)} parts, {sum(p.hours for p in every):.1f} CPU-h")
        return
    if arguments.command == "split":
        whole = sum(part.hours for part in every)
        running = 0.0
        for part in every:
            running += part.hours
            if running >= arguments.share * whole:
                print(part.index + 1)
                return
        print(len(every))
        return
    root, key = scratch(tag, arguments.cache)
    print(f"scratch {root}", flush=True)
    if arguments.command == "run":
        stop = len(every) if arguments.to is None else arguments.to
        run_parts(tag, range(arguments.start, stop), root, key, arguments.jobs)
    elif arguments.command == "status":
        done = [_done(tag, root, part) for part in every]
        finished = sum(d == p.stop - p.start for d, p in zip(done, every, strict=True))
        print(f"{finished} of {len(every)} parts finished, {sum(done)} replicates checkpointed")
    else:
        if arguments.source is None:
            parser.error("merge takes --source")
        merge(tag, root, arguments.source)


if __name__ == "__main__":
    main()
