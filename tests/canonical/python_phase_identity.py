"""Check that the RM39b migration left every study's Python phase byte for byte unchanged.

RM39 part b moved every primary ``draw_and_fit`` loop onto
:func:`tests.studies.evidence.replication.fit_replicates`.  With no failure, the samples, truth
rows and subject rows must equal the ones the loop wrote before.  This script runs each
registered study's ``draw_and_fit`` twice in fresh interpreters: once in a worktree of the base
commit and once in this checkout.  It writes each side's three frames as CSV and compares the
bytes.

In this checkout the systematic prefix is set to one replicate, so the two-call split of
``fit_replicates`` runs at two replicates.  ``--replicates 21`` with the default prefix runs
the split the way a study runs it.

``--fail R`` checks the failure path.  In this checkout every primary fit of replicate ``R``
raises ``LinAlgError``.  The subject rows must then equal the base's rows without replicate
``R``.  An independent study's samples and truth rows must equal the base's in full, because
its reference input is redrawn without the fit.  A dependent study's samples and truth rows
lose replicate ``R`` as well.

``--phase properties`` checks the property phase the same way (RM39b2).  Each property module
runs in a worktree of the base commit and in this checkout, in the mode ``--mode`` names:

=========  ==================================================================================
mode       what it compares
=========  ==================================================================================
budget     The rows of a module whose ``generate_property_rows`` caps its replicates
           (``budget=``, ``replicates=`` or ``cap=``), at ``--budget`` replicates, byte for
           byte, without the wall-clock columns in :data:`TIMING_COLUMNS`.  The head runs with
           ``--prefix`` (default 1), so the two-call split runs.
batches    The rows of a module that declares ``sampling_batches()``, each cell cut to
           ``--budget`` replicates, byte for byte.
payload    Every property loop of every module.  The base patches ``map_parallel`` and the
           head patches ``fit_properties``.  Each call is captured in order with its
           payloads (or keys), ``n_jobs`` and worker source, and the loop then runs on a
           truncated copy so the code after it still has rows.  A ``run_cells`` call is
           captured through ``CoverageStudy`` on the base and through its keys on the head,
           and returns synthetic rows.  The sequence, payloads and ``n_jobs`` must be equal.
           The worker sources must be equal except in :data:`WORKER_EXCEPTIONS`, and a module
           in :data:`KEY_PROJECTIONS` compares the images of its keys.  The truncated loops'
           rows must be equal where both sides ran them to completion.
labels     The head's ``fit_properties`` labels: unique, ``properties`` or
           ``properties/<name>``, as many calls as the base has pool calls, and equal to the
           labels that the budget run recorded.  Payload and budget runs record them.
=========  ==================================================================================

``--phase properties --fail R`` runs the head only.  Every ``fit_properties`` fit of replicate
``R`` raises ``LinAlgError`` inside the driver's collector.  The stamped rows must equal the rows
of a run with no failure minus each record's cells at ``R`` (so every other ``noise_control`` row
is unchanged), the stamps must equal the records, and ``reconcile`` must pass once
``requested_replicates`` is clipped to the run budget.  The scope is the module's budget or
batches run, or a truncated run for a module that derives controls through
``calibration_controls`` and for a module that has neither a cap nor batches.

Usage::

    git worktree add /tmp/base <base commit>
    python -m tests.canonical.python_phase_identity --base /tmp/base --out /tmp/identity
    python -m tests.canonical.python_phase_identity --phase properties --mode payload \\
        --base /tmp/base --out /tmp/identity
"""

from __future__ import annotations

import argparse
import dataclasses
import filecmp
import functools
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: Runs in either checkout.  It accepts the base's triple or bare frame and the head's
#: ``PhaseResult``, and writes whichever frames exist.
_DUMP = r"""
import sys
from pathlib import Path

import pandas as pd

from tests.studies.evidence.registry import registered

slug, replicates, jobs, prefix, out, fail = sys.argv[1:7]
if prefix != "default":
    from tests.studies.evidence import replication

    replication._PREFIX = int(prefix)
if fail != "none":
    import numpy as np

    from tests.studies.evidence import replication

    class _Raise:
        def __init__(self, fit_one, replicate_of):
            self.fit_one, self.replicate_of = fit_one, replicate_of

        def __call__(self, key, drawn):
            if self.replicate_of(key) == int(fail):
                np.linalg.inv(np.zeros((2, 2)))
            return self.fit_one(key, drawn)

    original = replication.fit_replicates

    def _failing(keys, fit_one, *, label, replicate_of, **options):
        if label == "primary":
            fit_one = _Raise(fit_one, replicate_of)
        return original(keys, fit_one, label=label, replicate_of=replicate_of, **options)

    replication.fit_replicates = _failing
(record,) = [record for record in registered() if record.slug == slug]
module = record.runner()
result = module.draw_and_fit(replicates=int(replicates), n=module.PRIMARY_N, n_jobs=int(jobs))
if isinstance(result, tuple):
    frames = dict(zip(("samples", "truths", "rows"), result))
elif isinstance(result, pd.DataFrame):
    frames = {"rows": result}
else:
    if fail == "none" and result.failures is not None:
        raise SystemExit(f"{slug}: a fit failed: {result.failures.to_dict('records')}")
    if fail != "none" and result.failures is None:
        raise SystemExit(f"{slug}: the forced failure was not recorded")
    frames = {"rows": result.rows, "samples": result.samples, "truths": result.truths}
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
if fail != "none" and result.reference_input_from_fit:
    (out / "dependent").touch()
for name, frame in frames.items():
    if frame is not None:
        frame.to_csv(out / f"{name}.csv", index=False)
"""


def _slugs() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from tests.studies.evidence.registry import registered

    return [record.slug for record in registered()]


def _dump(
    checkout: Path,
    slug: str,
    replicates: int,
    jobs: int,
    prefix: str,
    out: Path,
    fail: int | None = None,
) -> None:
    environment = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}{os.pathsep}{checkout}"}
    failing = "none" if fail is None else str(fail)
    subprocess.run(
        [sys.executable, "-c", _DUMP, slug, str(replicates), str(jobs), prefix, str(out), failing],
        cwd=checkout,
        env=environment,
        check=True,
    )


def compare(base: Path, head: Path, *, fail: int | None = None) -> list[str]:
    """The differences between two dumps, empty when they agree byte for byte.

    With ``fail``, the base frames the failure removes lose replicate ``fail`` first, and both
    sides are compared as text cells.
    """
    names = sorted({path.name for path in (*base.glob("*.csv"), *head.glob("*.csv"))})
    dependent = (head / "dependent").exists()
    out = []
    for name in names:
        left, right = base / name, head / name
        if not left.exists() or not right.exists():
            out.append(f"{name}: present on one side only")
        elif fail is None:
            if not filecmp.cmp(left, right, shallow=False):
                out.append(f"{name}: bytes differ")
        else:
            expected = _cells(left)
            if name == "rows.csv" or dependent:
                expected = expected.loc[expected["replicate"] != str(fail)]
            if not expected.reset_index(drop=True).equals(_cells(right)):
                out.append(f"{name}: cells differ")
    return out


def _cells(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _primary(arguments: argparse.Namespace) -> int:
    failed = []
    for slug in arguments.slugs or _slugs():
        started = time.monotonic()
        suffix = "" if arguments.fail is None else f"-fail{arguments.fail}"
        target = arguments.out / f"{slug}-r{arguments.replicates}{suffix}"
        try:
            _dump(
                arguments.base,
                slug,
                arguments.replicates,
                arguments.jobs,
                "default",
                target / "base",
            )
            _dump(
                ROOT,
                slug,
                arguments.replicates,
                arguments.jobs,
                arguments.prefix,
                target / "head",
                arguments.fail,
            )
        except subprocess.CalledProcessError as error:
            failed.append(slug)
            print(f"{slug}: ERROR ({error})", flush=True)
            continue
        differences = compare(target / "base", target / "head", fail=arguments.fail)
        verdict = "identical" if not differences else "; ".join(differences)
        if differences:
            failed.append(slug)
        print(f"{slug}: {verdict} ({time.monotonic() - started:.0f} s)", flush=True)
    print(f"{len(failed)} studies differ or failed: {failed}", flush=True)
    return 1 if failed else 0


# ---- the property phase (RM39b2) --------------------------------------------------------------

#: Property modules that no registered study names but that ``generate_property_rows`` serves.
_UNREGISTERED_PROPERTY_MODULES = (
    "longitudinal_incremental_properties",
    "longitudinal_incremental_dropout_properties",
)

#: The modules whose worker source differs between the base and this checkout, and why.  The
#: payload mode reports each as ``declared`` and treats any other difference as a failure.  A
#: module is listed only after a payload run showed that its worker differs.  The two E3
#: modules stop counting a failure in their own loop (RM39b2 3.2).  Every other entry splits the
#: draw from the fit (F5), so the head's worker takes the drawn sample and ``draw=`` draws it.
_E3 = "the loop's own failure counting goes, and the draw moves to ``draw=`` (E3)"
_DRAW = "the draw moves to ``draw=`` and the worker takes the drawn sample (F5)"
WORKER_EXCEPTIONS: dict[str, str] = {
    "clustered_few_cluster_properties": _E3,
    "few_cluster_crossfit_properties": _E3,
    "multi_arm_tmle_properties": _DRAW,
    "stratified_tmle_properties": _DRAW,
    "stratified_incremental_msm_properties": _DRAW,
    "stratified_msm_identity_properties": _DRAW,
    "stratified_drtmle_properties": _DRAW,
    "drtmle_properties": _DRAW,
    "mar_tmle_properties": _DRAW,
    "mar_natural_course_properties": _DRAW,
    "mar_natural_course_cvtmle_properties": _DRAW,
    "mar_arm_indexed_cvtmle_properties": _DRAW,
    "mar_attributable_properties": _DRAW,
    "cde_tmle_properties": _DRAW,
    "mar_drtmle_properties": _DRAW,
    "point_msm_properties": _DRAW,
    "learned_rule_cvtmle_properties": _DRAW,
    "policy_point_mtp_properties": _DRAW,
    "incremental_intervention_properties": _DRAW,
    "weighted_ltmle_properties": _DRAW,
    "categorical_ltmle_properties": _DRAW,
    "stochastic_categorical_ltmle_properties": _DRAW,
    "longitudinal_mtp_properties": _DRAW,
    "longitudinal_sdr_properties": _DRAW,
    "longitudinal_msm_properties": _DRAW,
    "crossfit_longitudinal_msm_properties": _DRAW,
    "weighted_ltmle_crossfit_properties": _DRAW,
    "categorical_ltmle_crossfit_properties": _DRAW,
    "point_survival_properties": _DRAW,
    "point_survival_crossfit_properties": _DRAW,
    "point_survival_policies_properties": _DRAW,
    "drtmle_one_wrong_properties": _DRAW,
    "ctmle_oat_per_arm_properties": _DRAW,
    "default_band_properties": _DRAW,
    "full_refit_bootstrap_properties": _DRAW,
    "longitudinal_incremental_properties": _DRAW,
    "longitudinal_incremental_dropout_properties": _DRAW,
}

#: The row columns that hold wall-clock time.  They differ between any two runs of the same
#: code, so the byte comparison leaves them out.  Only ``drtmle_one_wrong`` writes one.
TIMING_COLUMNS = ("seconds",)

#: How many leading replicates of how many cells a truncated loop fits, so that code after a
#: loop still has real rows without a full run.
_TRUNCATE_PER = 1
_TRUNCATE_CELLS = 3

#: Runs a child in a fresh interpreter as ``__main__`` so that joblib workers pickle its
#: classes by value, which the base checkout needs because it has no copy of this module.
_BOOTSTRAP = "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')"

#: The modules whose truncated head and base loops fit different draws, so their rows cannot
#: agree.  ``clustered_few_cluster`` keeps one key for each fit (F6), so the first keys of the
#: head hold other fits than the first payloads of the base.  The key projection compares them.
_TRUNCATION_NOT_ALIGNED = ("clustered_few_cluster_properties",)

_KINDS = {"_fit_set_rows": "fit_set", "_band_rows": "band"}


def _job_payload(item: tuple[Any, ...]) -> Any:
    """A base payload ``((function, args),)`` as the head's key ``(kind, args)``."""
    function, args = item[0]
    return _KINDS[function.__name__], args


#: The modules whose head keys are not the base payloads by design: the reason, the map from a
#: head key to what it stands for, and the map from a base payload to the same thing.  The
#: payload mode compares the ordered, de-duplicated images of the two sides.
#: ``clustered_few_cluster`` fits each estimator of a draw as its own key (RM39b2 F6), so one
#: base payload becomes one key per fit.  The five job modules name the fit set or the band by a
#: kind in place of the function the base passed.
KEY_PROJECTIONS: dict[str, tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = {
    "clustered_few_cluster_properties": (
        (
            "keyed per fit (F6): (sizes, J, replicate, fit, requested) for the base's "
            "((sizes, J, replicate, requested),)"
        ),
        lambda key: (key[0], key[1], key[2], key[4]),
        lambda item: item[0],
    ),
    **{
        name: (
            "a kind names the job: ('fit_set' | 'band', args) for the base's ((function, args),)",
            lambda key: key,
            _job_payload,
        )
        for name in (
            "stochastic_categorical_ltmle_properties",
            "longitudinal_mtp_properties",
            "longitudinal_sdr_properties",
            "longitudinal_incremental_properties",
            "longitudinal_incremental_dropout_properties",
        )
    },
}


def _projected(items: Sequence[Any], mapping: Callable[[Any], Any]) -> tuple[str, int]:
    """The digest and length of the ordered, de-duplicated images of ``items``."""
    images: list[Any] = []
    for item in items:
        image = (mapping(item),)
        if not images or images[-1] != image:
            images.append(image)
    return _digest(images), len(images)


_CAPS = ("budget", "replicates", "cap")


def _canon(value: Any, depth: int = 0) -> str:
    """A text that equals across processes for equal payloads: no addresses, no elided arrays."""
    if depth > 12:
        return "<deep>"
    if isinstance(value, np.ndarray):
        digest = hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest()
        return f"ndarray{value.shape}{value.dtype}:{digest}"
    if isinstance(value, pd.DataFrame):
        hashed = int(pd.util.hash_pandas_object(value, index=True).sum())
        return f"DataFrame{value.shape}{list(map(str, value.columns))}:{hashed}"
    if isinstance(value, pd.Series):
        return f"Series{value.shape}:{int(pd.util.hash_pandas_object(value).sum())}"
    if isinstance(value, tuple | list):
        return f"{type(value).__name__}[" + ",".join(_canon(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted((_canon(k, depth + 1), _canon(v, depth + 1)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, set | frozenset):
        return "set{" + ",".join(sorted(_canon(v, depth + 1) for v in value)) + "}"
    if inspect.isfunction(value) or inspect.isclass(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, functools.partial):
        parts = (value.func, value.args, value.keywords)
        return "partial(" + ",".join(_canon(part, depth + 1) for part in parts) + ")"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        return f"{type(value).__qualname__}({_canon(fields, depth + 1)})"
    return re.sub(r"0x[0-9a-fA-F]+", "0x", repr(value)).replace(os.getcwd(), "<ROOT>")


def _digest(items: Sequence[Any]) -> str:
    text = "\n".join(_canon(item) for item in items)
    return hashlib.sha256(text.encode()).hexdigest()


def _worker_sources(func: Any, seen: set[int] | None = None) -> list[str]:
    """The source of a worker and of every callable it closes over or holds.

    Shared helpers under ``tests.studies.evidence`` are left out: both checkouts call them by
    name, and the adapters (``splat``) differ by construction.
    """
    seen = set() if seen is None else seen
    if func is None or id(func) in seen:
        return []
    seen.add(id(func))
    if isinstance(func, functools.partial):
        return _worker_sources(func.func, seen)
    out: list[str] = []
    if inspect.isfunction(func) or inspect.ismethod(func):
        if not (getattr(func, "__module__", "") or "").startswith("tests.studies.evidence"):
            try:
                out.append(textwrap.dedent(inspect.getsource(func)).strip())
            except (OSError, TypeError):
                out.append(f"<no source: {getattr(func, '__qualname__', func)}>")
        for cell in getattr(func, "__closure__", None) or ():
            try:
                held = cell.cell_contents
            except ValueError:
                continue
            if callable(held) and not isinstance(held, type):
                out.extend(_worker_sources(held, seen))
        return out
    if inspect.isclass(func) or not callable(func):
        return out
    kind = type(func)
    if not (kind.__module__ or "").startswith("tests.studies.evidence"):
        try:
            out.append(textwrap.dedent(inspect.getsource(kind)).strip())
        except (OSError, TypeError):
            out.append(f"<no source: {kind.__qualname__}>")
    for held in getattr(func, "__dict__", {}).values():
        if callable(held) and not isinstance(held, type):
            out.extend(_worker_sources(held, seen))
    return out


def _select(
    keys: Sequence[Any], cell_of: Callable, replicate_of: Callable, per: int, cells: int
) -> list[int]:
    """The indices of the first ``per`` replicates of up to ``cells`` evenly spread cells."""
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(str(cell_of(key)), []).append(index)
    names = list(groups)
    if len(names) > cells:
        step = len(names) / cells
        names = [names[int(i * step)] for i in range(cells)]
    chosen: list[int] = []
    for name in names:
        members = sorted(groups[name], key=lambda i: int(replicate_of(keys[i])))
        chosen.extend(members[:per])
    return sorted(chosen)


class _Raise:
    """A ``fit_one`` that raises ``LinAlgError`` for one replicate and fits the others."""

    def __init__(self, fit_one: Callable, replicate_of: Callable, replicate: int) -> None:
        self.fit_one, self.replicate_of, self.replicate = fit_one, replicate_of, replicate

    def __call__(self, key: Any, drawn: Any) -> Any:
        if int(self.replicate_of(key)) == self.replicate:
            np.linalg.inv(np.zeros((2, 2)))
        return self.fit_one(key, drawn)


@dataclasses.dataclass
class _Fake:
    """One synthetic replicate record, with the fields ``run_cells`` reads."""

    estimand: str
    replicate: int
    truth: float = 0.0
    estimate: float = 0.0
    std_error: float = 1.0
    covered: bool = True
    rejected: bool = False
    alpha: float = 0.05


class _FakeResult:
    def __init__(self, replications: tuple[_Fake, ...]) -> None:
        self.replications, self.n_failed = replications, 0


def _cell_identity(cell: Any) -> str:
    return _canon(
        (
            cell.property,
            cell.cell,
            cell.role,
            cell.n,
            cell.replicates,
            cell.seed,
            cell.estimand,
            dict(cell.fit_kwargs),
            cell.dgp,
        )
    )


def _study_fields(cell: Any) -> str:
    """The fields ``run_cells`` hands to the ``CoverageStudy`` of one cell."""
    return _canon((cell.n, cell.replicates, cell.seed, (cell.estimand,), dict(cell.fit_kwargs)))


class _Capture:
    """Patches the pool of one side, records every property loop, and truncates it."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.side = spec["side"]
        self.calls: list[dict[str, Any]] = []
        self.truncate = spec["mode"] in ("payload", "truncated")
        self.fake_cells = spec["mode"] == "payload"
        self.fail: int | None = spec.get("fail")
        self.per = _TRUNCATE_PER if spec.get("fail") is None else int(spec["per"])
        self.plan: list[dict[str, Any]] = []
        if spec.get("calls") and Path(spec["calls"]).exists():
            self.plan = json.loads(Path(spec["calls"]).read_text())
        self.mismatch = False
        self._cells: list[Any] = []
        self._cell_index = 0
        self._current: dict[str, Any] | None = None

    # -- head: fit_properties ---------------------------------------------------------------

    def _head_pool(self, real: Callable, *, evidence: bool) -> Callable:
        def wrapped(keys: Sequence[Any], fit_one: Callable, **kw: Any) -> list[Any]:
            keys = list(keys)
            splatted = type(fit_one).__name__ == "_Splat"
            arguments = [key if splatted and isinstance(key, tuple) else (key,) for key in keys]
            projection = KEY_PROJECTIONS.get(self.spec["module"])
            record: dict[str, Any] = {
                "kind": "cells" if evidence else "pool",
                "label": kw.get("label", "properties"),
                "n": len(keys),
                "digest": _digest(arguments),
                "raw": _digest(keys),
                "n_jobs": kw.get("n_jobs"),
                "prefer": None,
                "workers": _worker_sources(fit_one),
                "draw": _worker_sources(kw.get("draw")),
            }
            self.calls.append(record)
            if projection is not None and not evidence:
                record["projected"], record["projected_n"] = _projected(keys, projection[1])
            if evidence:
                grouped: list[dict[str, Any]] = []
                for key in keys:
                    cell, _, seed = key
                    identity = _cell_identity(cell)
                    if not grouped or grouped[-1]["id"] != identity:
                        grouped.append({"id": identity, "study": _study_fields(cell), "seeds": []})
                    grouped[-1]["seeds"].append(int(seed))
                record["cells"] = grouped
                record["digest"] = _digest([(g["id"], g["study"], g["seeds"]) for g in grouped])
                record["raw"] = record["digest"]
                if self.fake_cells:
                    return [
                        (
                            {
                                "property": cell.property,
                                "cell": cell.cell,
                                "role": cell.role,
                                "replicate": replicate,
                                "n": cell.n,
                                "requested_replicates": cell.replicates,
                                "failed_replicates": 0,
                                "truth": 0.0,
                                "estimate": 0.0,
                                "std_error": 1.0,
                                "covered": 1,
                                "rejected": 0,
                            },
                            0.05,
                        )
                        for cell, replicate, _ in keys
                    ]
            if self.truncate:
                per = self.per
                limit = 10**9 if evidence else int(self.spec.get("cells", _TRUNCATE_CELLS))
                index = _select(keys, kw["cell_of"], kw["replicate_of"], per, limit)
                record["idx"] = index
                keys = [keys[i] for i in index]
            if self.fail is not None:
                fit_one = _Raise(fit_one, kw["replicate_of"], self.fail)
            return real(keys, fit_one, **kw)

        return wrapped

    # -- base: map_parallel and run_cells ---------------------------------------------------

    def _base_pool(self, real: Callable) -> Callable:
        def wrapped(
            func: Callable, payloads: Any, *, n_jobs: int | None = 1, prefer: str | None = None
        ) -> list[Any]:
            items = list(payloads)
            arguments = [item if isinstance(item, tuple) else (item,) for item in items]
            number = len(self.calls)
            record: dict[str, Any] = {
                "kind": "pool",
                "label": None,
                "n": len(items),
                "digest": _digest(arguments),
                "raw": _digest(items),
                "n_jobs": n_jobs,
                "prefer": prefer,
                "workers": _worker_sources(func),
                "draw": [],
            }
            projection = KEY_PROJECTIONS.get(self.spec["module"])
            if projection is not None:
                record["projected"], record["projected_n"] = _projected(items, projection[2])
            self.calls.append(record)
            planned = self.plan[number] if number < len(self.plan) else {}
            index = planned.get("idx")
            if index is None or planned.get("n") != len(items):
                index = list(range(min(_TRUNCATE_PER * _TRUNCATE_CELLS, len(items))))
                self.mismatch = True
            return real(func, [items[i] for i in index], n_jobs=n_jobs, prefer=prefer)

        return wrapped

    def _base_cells(self, real: Callable) -> Callable:
        def wrapped(cells: Sequence[Any], estimator: Callable, *, n_jobs: int) -> pd.DataFrame:
            self._cells, self._cell_index = list(cells), 0
            self._current = {
                "kind": "cells",
                "label": None,
                "n": 0,
                "n_jobs": n_jobs,
                "prefer": None,
                "workers": [],
                "draw": [],
                "cells": [],
            }
            self.calls.append(self._current)
            frame = real(cells, estimator, n_jobs=n_jobs)
            self._current["digest"] = _digest(
                [(g["id"], g["study"], g["seeds"]) for g in self._current["cells"]]
            )
            self._current["raw"] = self._current["digest"]
            return frame

        return wrapped

    def _spy_study(self, **kw: Any) -> Any:
        assert self._current is not None
        cell = self._cells[self._cell_index]
        self._cell_index += 1
        seeds = [int(s) for s in np.asarray(kw["sample_seeds"]).tolist()]
        study = _canon(
            (
                kw["n"],
                kw["n_replicates"],
                kw["seed"],
                tuple(kw["estimands"]),
                dict(kw["fit_kwargs"]),
            )
        )
        self._current["cells"].append(
            {"id": _cell_identity(cell), "study": study, "seeds": seeds, "n_jobs": kw["n_jobs"]}
        )
        estimand = kw["estimands"][0]
        count = int(kw["n_replicates"])

        class _Study:
            def run(self) -> _FakeResult:
                return _FakeResult(tuple(_Fake(estimand, i) for i in range(count)))

        return _Study()

    # -- installation -----------------------------------------------------------------------

    def _bound(self, name: str, real: Any) -> list[Any]:
        return [
            module
            for key, module in list(sys.modules.items())
            if module is not None
            and key.startswith("tests.studies")
            and getattr(module, name, None) is real
        ]

    def install(self) -> None:
        patched_head = self.side == "head" and (
            self.truncate or self.fail is not None or self.spec["mode"] in ("budget", "batches")
        )
        if patched_head:
            from tests.studies.evidence import replication

            real = replication.fit_properties
            plain = self._head_pool(real, evidence=False)
            cells = self._head_pool(real, evidence=True)
            targets = self._bound("fit_properties", real)
            replication.fit_properties = plain
            for module in targets:
                if module.__name__ == "tests.studies.evidence.properties":
                    module.fit_properties = cells
                elif not module.__name__.startswith("tests.studies.evidence"):
                    module.fit_properties = plain
        if self.side == "base" and self.spec["mode"] == "payload":
            from cleverly.utils import parallel
            from tests.studies.evidence import properties as evidence_properties

            pool = parallel.map_parallel
            wrapper = self._base_pool(pool)
            parallel.map_parallel = wrapper
            for module in self._bound("map_parallel", pool):
                if not module.__name__.startswith("tests.studies.evidence"):
                    module.map_parallel = wrapper
            run = evidence_properties.run_cells
            wrapped_run = self._base_cells(run)
            for module in self._bound("run_cells", run):
                module.run_cells = wrapped_run
            evidence_properties.CoverageStudy = self._spy_study


def _untimed(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(columns=[c for c in TIMING_COLUMNS if c in rows.columns])


def _generate(spec: dict[str, Any], module: Any) -> pd.DataFrame:
    """The module's property rows in the scope the spec names, without wall-clock columns."""
    return _untimed(_generate_timed(spec, module))


def _generate_timed(spec: dict[str, Any], module: Any) -> pd.DataFrame:
    jobs = int(spec["jobs"])
    scope = spec["scope"]
    if scope == "budget":
        return module.generate_property_rows(n_jobs=jobs, **{spec["kw"]: int(spec["budget"])})
    if scope == "batches":
        frames = []
        for batch in module.sampling_batches():
            small = dataclasses.replace(
                batch,
                cells=tuple(
                    dataclasses.replace(cell, replicates=int(spec["budget"]))
                    for cell in batch.cells
                ),
            )
            frames.append(small.run(n_jobs=jobs))
        return pd.concat(frames, ignore_index=True)
    return module.generate_property_rows(n_jobs=jobs)


def _child_properties(spec: dict[str, Any]) -> int:
    """Runs one module on one side and writes what the parent compares."""
    import importlib

    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    if spec["side"] == "head" and spec["prefix"] != "default":
        from tests.studies.evidence import replication

        replication._PREFIX = int(spec["prefix"])
    module = importlib.import_module(f"tests.studies.{spec['module']}")
    capture = _Capture(spec)
    capture.install()
    started = time.monotonic()
    if spec.get("fail") is not None:
        return _child_fail(spec, module, capture, out)
    note = ""
    try:
        rows = _generate(spec, module)
    except Exception as error:
        # The code after a loop can need the loop's full rows, which a truncated loop does not
        # give.  The calls captured so far are what the payload mode compares, and the parent
        # compares the call counts, so a loop the error skipped shows as a count difference.
        if spec["mode"] != "payload" or not capture.calls:
            raise
        rows = pd.DataFrame()
        note = f"{type(error).__name__} after {len(capture.calls)} calls: {error}"
    (out / "note.txt").write_text(note)
    rows.to_csv(out / "rows.csv", index=False)
    (out / "calls.json").write_text(json.dumps(capture.calls))
    print(f"{spec['side']}: {len(rows)} rows in {time.monotonic() - started:.0f} s", flush=True)
    return 0


def _child_fail(spec: dict[str, Any], module: Any, capture: _Capture, out: Path) -> int:
    """Forces a failure of one replicate in every loop and checks the head's accounting of it."""
    from tests.studies.evidence import property_failures, replication

    replicate = int(spec["fail"])
    capture.fail = None
    clean = _generate(spec, module)
    capture.calls.clear()
    capture.fail = replicate
    with replication.recording(
        spec["module"], witness=getattr(module, "PROPERTY_WITNESS", None)
    ) as recorded:
        failed = _generate(spec, module)
    records = property_failures.frame(recorded)
    problems: list[str] = []
    if records is None and not capture.calls and failed.empty:
        (out / "note.txt").write_text("the module declares no property fit\n")
    elif records is None:
        problems.append("the forced failure was not recorded")
    else:
        stamped = property_failures.stamp(failed, records)
        limit = clean.groupby(["property", "cell"]).size()
        for rows in (clean, stamped):
            keys = list(zip(rows["property"], rows["cell"], strict=True))
            rows["requested_replicates"] = [
                min(int(requested), int(limit[key]))
                for requested, key in zip(rows["requested_replicates"], keys, strict=True)
            ]
        try:
            property_failures.reconcile(
                stamped, records, derived=getattr(module, "DERIVED_CELLS", None)
            )
        except ValueError as error:
            problems.append(f"reconcile: {error}")
        lost = property_failures.expand(records)
        lost_keys = {
            (family, cell, int(n))
            for family, cell, n in zip(
                lost["property"], lost["cell"], lost["replicate"], strict=True
            )
        }
        clean_keys = zip(
            clean["property"], clean["cell"], clean["replicate"].astype(int), strict=True
        )
        keep = [key not in lost_keys for key in clean_keys]
        expected = property_failures.stamp(clean.loc[keep].reset_index(drop=True), records)
        actual = stamped.reset_index(drop=True)
        if expected.to_csv(index=False) != actual.to_csv(index=False):
            problems.append(_frame_difference(expected, actual))
        noise = actual["cell"].astype(str).str.endswith("noise_control")
        spec_note = f"{len(records)} records, {len(lost_keys)} cell replicates lost"
        (out / "note.txt").write_text(f"{spec_note}, {int(noise.sum())} noise_control rows\n")
    (out / "problems.json").write_text(json.dumps(problems))
    print(f"head: {len(problems)} problems", flush=True)
    return 0


def _frame_difference(expected: pd.DataFrame, actual: pd.DataFrame) -> str:
    if list(expected.columns) != list(actual.columns):
        return "columns differ"
    if len(expected) != len(actual):
        return f"{len(expected)} expected rows, {len(actual)} actual"
    left = expected.astype(str).reset_index(drop=True)
    right = actual.astype(str).reset_index(drop=True)
    wrong = (left != right).any(axis=0)
    columns = list(wrong[wrong].index)
    rows = int((left != right).any(axis=1).sum())
    first = (left != right).any(axis=1).idxmax()
    was, now = expected.iloc[first].to_dict(), actual.iloc[first].to_dict()
    return f"{rows} rows differ in {columns}; first: {was} vs {now}"


@dataclasses.dataclass(frozen=True)
class _PropertyModule:
    name: str
    slug: str
    kw: str | None
    batches: bool
    calibration: bool


def _property_modules(wanted: Sequence[str] | None) -> list[_PropertyModule]:
    sys.path.insert(0, str(ROOT))
    import importlib

    from tests.studies.evidence.registry import registered

    pairs = [(record.properties_module.rsplit(".", 1)[-1], record.slug) for record in registered()]
    pairs += [(name, name) for name in _UNREGISTERED_PROPERTY_MODULES]
    out = []
    for name, slug in pairs:
        if wanted and name not in wanted and slug not in wanted:
            continue
        module = importlib.import_module(f"tests.studies.{name}")
        parameters = inspect.signature(module.generate_property_rows).parameters
        kw = next((cap for cap in _CAPS if cap in parameters), None)
        source = inspect.getsource(module)
        calibration = "calibration_controls" in source or any(
            shared in source
            for shared in (
                "categorical_longitudinal_properties",
                "weighted_longitudinal_properties_common",
            )
        )
        out.append(
            _PropertyModule(name, slug, kw, hasattr(module, "sampling_batches"), calibration)
        )
    return out


def _spawn(checkout: Path, spec: dict[str, Any], timeout: int) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": f"{checkout / 'src'}{os.pathsep}{checkout}",
        "PYTHONHASHSEED": "0",
    }
    script = str(Path(__file__).resolve())
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stderr.log", "w") as errors:
        subprocess.run(
            [sys.executable, "-c", _BOOTSTRAP, script, "--child-properties", json.dumps(spec)],
            cwd=checkout,
            env=environment,
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )


def _compare_calls(
    base: list[dict[str, Any]], head: list[dict[str, Any]], name: str
) -> tuple[list[str], str]:
    """The differences between two call sequences, and how the worker sources compare."""
    if len(base) != len(head):
        return [f"{len(base)} base calls, {len(head)} head calls"], "n/a"
    out: list[str] = []
    worker = "equal"
    for number, (left, right) in enumerate(zip(base, head, strict=True)):
        where = f"call {number} ({right.get('label')})"
        if left["kind"] != right["kind"]:
            out.append(f"{where}: kind {left['kind']} vs {right['kind']}")
            continue
        if "projected" in right or "projected" in left:
            if left.get("projected") != right.get("projected") or left.get(
                "projected_n"
            ) != right.get("projected_n"):
                out.append(f"{where}: the keys do not project onto the payloads")
            if left["n_jobs"] != right["n_jobs"]:
                out.append(f"{where}: n_jobs {left['n_jobs']} vs {right['n_jobs']}")
            if left["workers"] != right["workers"]:
                worker = "differs"
            continue
        if left["n"] != right["n"] and left["kind"] == "pool":
            out.append(f"{where}: {left['n']} payloads vs {right['n']} keys")
        # A worker takes a key either as its splatted arguments or whole, so the same payloads
        # read equal as arguments or as raw values (see _Capture).
        if left["digest"] != right["digest"] and left.get("raw") != right.get("raw"):
            out.append(f"{where}: payloads differ")
        if left["n_jobs"] != right["n_jobs"]:
            out.append(f"{where}: n_jobs {left['n_jobs']} vs {right['n_jobs']}")
        if left.get("prefer") is not None:
            out.append(f"{where}: the base passes prefer={left['prefer']!r}")
        if left["kind"] == "cells":
            continue
        if left["workers"] != right["workers"]:
            worker = "differs"
    if worker == "differs":
        drawn = any(call["draw"] for call in head)
        if name in WORKER_EXCEPTIONS:
            worker = f"differs, declared: {WORKER_EXCEPTIONS[name]}"
        else:
            worker = "DIFFERS" + (" (the head splits a draw)" if drawn else "")
            out.append("worker source differs and the module is not declared")
    return out, worker


def _check_labels(head: list[dict[str, Any]], base: list[dict[str, Any]] | None) -> list[str]:
    labels = [call["label"] for call in head]
    out = []
    if len(set(labels)) != len(labels):
        out.append(f"labels are not unique: {labels}")
    for label in labels:
        if label != "properties" and not label.startswith("properties/"):
            out.append(f"label {label!r} is neither properties nor properties/<name>")
    if base is not None and len(base) != len(head):
        out.append(f"{len(head)} labels for {len(base)} base calls")
    return out


def _properties(arguments: argparse.Namespace) -> int:
    modules = _property_modules(arguments.slugs)
    failed: list[str] = []
    mode = "fail" if arguments.fail is not None else arguments.mode
    log = None if arguments.log is None else open(arguments.log, "a")  # noqa: SIM115

    def say(line: str) -> None:
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()

    for module in modules:
        started = time.monotonic()
        spec: dict[str, Any] = {
            "module": module.name,
            "kw": module.kw,
            "jobs": arguments.jobs,
            "budget": arguments.budget,
            "prefix": arguments.prefix,
            "per": arguments.per,
            "fail": arguments.fail,
        }
        tag = "payload" if mode == "labels" else mode
        if mode in ("budget", "batches", "fail"):
            tag += str(arguments.budget)
        target = arguments.out / f"{module.name}-{tag}"
        try:
            verdict = _run_property_mode(arguments, module, mode, spec, target)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            tails = []
            for side in ("head", "base"):
                path = target / side / "stderr.log"
                if path.exists():
                    lines = path.read_text().strip().splitlines()
                    if lines and "Error" in lines[-1]:
                        tails.append(f"{side}: {lines[-1]}")
            say(f"{module.slug} [{mode}]: ERROR ({type(error).__name__}) {' | '.join(tails)}")
            failed.append(module.slug)
            continue
        if verdict is None:
            continue
        differences, note = verdict
        text = "identical" if not differences else "; ".join(differences)
        if differences:
            failed.append(module.slug)
        suffix = f" [{note}]" if note else ""
        say(f"{module.slug} [{mode}]: {text}{suffix} ({time.monotonic() - started:.0f} s)")
    say(f"{len(failed)} studies differ or failed: {failed}")
    if log is not None:
        log.close()
    return 1 if failed else 0


def _truncated_head(spec: dict[str, Any], target: Path, timeout: int) -> str:
    """Runs the head on a truncated copy of every loop: three cells first, then every cell.

    A module that checks its declared cells after a loop needs every cell, so the narrower run
    is a speed-up only.  Returns which truncation ran.
    """
    try:
        _spawn(ROOT, {**spec, "side": "head", "out": str(target / "head"), "cells": 3}, timeout)
        return "3 cells per loop"
    except subprocess.CalledProcessError:
        _spawn(ROOT, {**spec, "side": "head", "out": str(target / "head"), "cells": 10**9}, timeout)
        return "every cell per loop"


def _run_property_mode(
    arguments: argparse.Namespace,
    module: _PropertyModule,
    mode: str,
    spec: dict[str, Any],
    target: Path,
) -> tuple[list[str], str] | None:
    """Runs one module in one mode on the needed sides and returns its differences and a note."""
    timeout = arguments.timeout
    if mode in ("budget", "batches"):
        if (mode == "budget" and module.kw is None) or (mode == "batches" and not module.batches):
            return None
        spec = {**spec, "mode": mode, "scope": mode}
        for side, checkout in (("head", ROOT), ("base", arguments.base)):
            _spawn(checkout, {**spec, "side": side, "out": str(target / side)}, timeout)
        same = (target / "base" / "rows.csv").read_bytes() == (
            target / "head" / "rows.csv"
        ).read_bytes()
        head_calls = json.loads((target / "head" / "calls.json").read_text())
        problems = [] if same else ["rows.csv bytes differ"]
        problems += _check_labels(head_calls, None)
        return problems, f"labels {len(head_calls)}"
    if mode in ("payload", "labels"):
        head_calls_path = target / "head" / "calls.json"
        base_calls_path = target / "base" / "calls.json"
        cells = "cached"
        if mode == "payload" or not (head_calls_path.exists() and base_calls_path.exists()):
            spec = {**spec, "mode": "payload", "scope": "truncated"}
            cells = _truncated_head(spec, target, timeout)
            _spawn(
                arguments.base,
                {
                    **spec,
                    "side": "base",
                    "out": str(target / "base"),
                    "calls": str(head_calls_path),
                },
                timeout,
            )
        head = json.loads(head_calls_path.read_text())
        base = json.loads(base_calls_path.read_text())
        if mode == "payload":
            differences, worker = _compare_calls(base, head, module.name)
            # The truncated loops also ran for real on both sides, so their rows must agree.
            left, right = target / "base" / "rows.csv", target / "head" / "rows.csv"
            rows = "rows not compared"
            if module.name in _TRUNCATION_NOT_ALIGNED:
                rows = "rows not compared (the truncations differ by design)"
            elif left.stat().st_size > 1 and right.stat().st_size > 1:
                rows = "rows identical"
                if left.read_bytes() != right.read_bytes():
                    rows = "rows DIFFER"
                    differences.append("the truncated loops' rows differ")
            cells = f"{cells}, {rows}"
            stopped = [
                f"{side} {(target / side / 'note.txt').read_text().split(':')[0]}"
                for side in ("head", "base")
                if (target / side / "note.txt").exists()
                and (target / side / "note.txt").read_text()
            ]
            tail = f", stopped: {'; '.join(stopped)}" if stopped else ""
            return differences, f"{len(head)} calls, workers {worker}, {cells}{tail}"
        problems = _check_labels(head, base)
        budget_labels = (
            target.parent / f"{module.name}-budget{arguments.budget}" / "head" / "calls.json"
        )
        if budget_labels.exists():
            ran = [call["label"] for call in json.loads(budget_labels.read_text())]
            if ran != [call["label"] for call in head]:
                problems.append(f"budget run labels {ran} differ from payload labels")
        return problems, f"labels {[c['label'] for c in head]}"
    # mode == "fail": the head only
    if module.kw is not None:
        scope = "budget"
    elif module.calibration:
        # The batches alone lack the controls that ``calibration_controls`` derives, which the
        # records of a derived module name.
        scope = "truncated"
    elif module.batches:
        scope = "batches"
    else:
        scope = "truncated"
    spec = {**spec, "mode": "budget" if scope != "truncated" else "truncated", "scope": scope}
    if scope == "truncated":
        _truncated_head(spec, target, timeout)
    else:
        _spawn(ROOT, {**spec, "side": "head", "out": str(target / "head")}, timeout)
    problems = json.loads((target / "head" / "problems.json").read_text())
    note = (target / "head" / "note.txt").read_text().strip() if not problems else ""
    return problems, f"{scope} scope" + (f", {note}" if note else "")


def main() -> int:
    if "--child-properties" in sys.argv:
        spec = json.loads(sys.argv[sys.argv.index("--child-properties") + 1])
        return _child_properties(spec)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("primary", "properties"), default="primary")
    parser.add_argument("--base", type=Path, required=True, help="a worktree of the base commit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slugs", nargs="*", help="the studies to check (default: all)")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--prefix", default="1", help="the head's systematic prefix, or 'default' for 20"
    )
    parser.add_argument("--fail", type=int, help="the replicate whose fits raise in this checkout")
    parser.add_argument(
        "--mode",
        choices=("budget", "batches", "payload", "labels"),
        help="the properties phase mode",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2,
        help="properties: the replicates of a budget or batches run",
    )
    parser.add_argument(
        "--per", type=int, default=3, help="properties --fail: replicates a truncated loop fits"
    )
    parser.add_argument("--timeout", type=int, default=7200, help="properties: seconds per child")
    parser.add_argument("--log", type=Path, help="properties: a file to append the report to")
    arguments = parser.parse_args()
    if arguments.phase == "primary":
        return _primary(arguments)
    if arguments.mode is None and arguments.fail is None:
        parser.error("--phase properties needs --mode or --fail")
    return _properties(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
