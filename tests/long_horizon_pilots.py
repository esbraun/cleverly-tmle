"""The pilot driver of the long-horizon design: exact scans and sample pilots, in parts.

The work is a fixed, ordered list of *parts*.  A part is one exact scan of the design (a law of
a candidate scan, one quantity's fallback scan) or one block of :data:`BLOCK` replicates of one
pilot stream (:func:`tests.long_horizon_design.streams`).  Each part writes its own result file
under ``<root>/parts``, and its content depends on the part alone: every replicate draws from
its own seed (:func:`~tests.long_horizon_design.stream_seed`), so any machine may compute any
part and two machines write the same bytes for it.

The root is ``<scratch>/x39-pilots/<key>``.  The key is a digest of the declared parts, the bytes
of the code that runs them and the Python package versions, so a changed declaration, estimator
or package never reuses a stale part.  A root stamped with another key is refused.

Commands (run from the repository root with ``PYTHONPATH=.``)::

    python -m tests.long_horizon_pilots list [--family F]
    python -m tests.long_horizon_pilots split --share S
    python -m tests.long_horizon_pilots run [--jobs N] [--from I] [--to J | --down-to I] [--family F]
    python -m tests.long_horizon_pilots check I
    python -m tests.long_horizon_pilots merge
    python -m tests.long_horizon_pilots scans

``list`` prints each part's index, family, name and estimated CPU hours, with the running total.
``split --share S`` prints the first index at which the running total passes ``S`` of the whole.
``run`` computes the parts ``[I, J)`` in ascending order, or with ``--down-to I`` from the last
part down to ``I``; a part whose file exists is skipped, and a sample part resumes from the
per-replicate checkpoints of :func:`tests.studies.evidence.replication.fit_replicates` under
``<root>/checkpoints``.  ``check I`` recomputes part ``I`` in a temporary root and compares its
bytes with the stored file.  ``merge`` refuses a missing part or a replicate whose seed is not
its declared seed, writes ``<root>/merged/<family>.csv.gz``, and prints each family's reading:
the Monte Carlo check, the F2 and bias floors, the truth-row floor, the overfitting pilot and
every exact scan's pick against the declared one.  ``scans`` prints the exact scans' picks
from the parts that exist.

A pilot above n = 8,000 runs only when no study runs on the same machine, after a one-fit memory
check.  Step 4b's property cells use the same machinery: a property family adds its parts to
:func:`parts` and its reading to :data:`READERS`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tests import discrete_law_longitudinal_horizon as law_
from tests import long_horizon_design as D
from tests import long_horizon_sampling_sd as sampling
from tests.discrete_law_longitudinal_horizon import Law
from tests.studies import long_horizon_primary as primary
from tests.studies.evidence import replication

#: Replicates per sample part.
BLOCK = 100
#: The implementation name of every pilot checkpoint.
IMPLEMENTATION = "x39-pilots"
#: The files whose bytes enter the key, besides ``src/cleverly``.
CODE = (
    "tests/long_horizon_pilots.py",
    "tests/long_horizon_design.py",
    "tests/long_horizon_sampling_sd.py",
    "tests/discrete_law_longitudinal_horizon.py",
    "tests/longitudinal_horizon_plans.py",
    "tests/studies/long_horizon_common.py",
    "tests/studies/long_horizon_primary.py",
    "tests/studies/fractional_glm.py",
    "tests/studies/canonical_long_horizon_ltmle.py",
    "tests/studies/canonical_long_horizon_categorical_ltmle.py",
    "tests/studies/canonical_long_horizon_sdr.py",
    "tests/studies/evidence/replication.py",
)
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FAMILIES = ("mc", "f2", "bias", "truth", "overfit")
#: The bias-family replicate ladder and the truth-row ladder of section 5.3.
BIAS_LADDER = (1_000, 1_200, 2_000, 3_000, 4_000)
TRUTH_LADDER = (1_200, 2_000, 3_000)


# ----------------------------------------------------------------------------- exact scans
@dataclass(frozen=True)
class Candidate:
    """One candidate of a control family on one law.

    Parameters
    ----------
    label : str
        The candidate's name, in the family's declared order.
    positive : Config
        The gated positive the candidate is paired with.
    control : Config
        The control estimator.
    """

    label: str
    positive: D.Config
    control: D.Config


@dataclass(frozen=True)
class Scan:
    """One control family's candidate scan over an ordered list of laws.

    Parameters
    ----------
    name : str
        ``<study>:<family>`` with the arm count and K where a study has several.
    K : int
        The decision count of stages A and B.
    cell_law : Law
        The law of stages A and B.
    cell_n : int
        The cell n of stage A.
    laws : tuple of tuple
        ``(K, law)`` in the order stage C (and stage D) walks.
    cap : int
        The cap of the scan.
    declared : str
        The cell of the declared control in :func:`tests.long_horizon_design.controls`.
    build : Callable
        ``(K, law) -> [Candidate, ...]`` in the declared order.
    """

    name: str
    K: int
    cell_law: Law
    cell_n: int
    laws: tuple[tuple[int, Law], ...]
    cap: int
    declared: str
    build: Callable[[int, Law], list[Candidate]]

    @property
    def study(self) -> str:
        return self.name.split(":")[0]


def _ladder(K: int, m: int, law: Law, plan: str, estimator: str, **control: Any) -> list[Candidate]:
    return [
        Candidate(
            rung,
            D.Config(estimator, K, m, law, plan, learners=D.RUNGS[rung]),
            D.Config(estimator, K, m, law, plan, learners=D.RUNGS[rung], **control),
        )
        for rung in D.RUNGS
    ]


def _targeting(m: int, plan: str) -> Callable[[int, Law], list[Candidate]]:
    def build(K: int, law: Law) -> list[Candidate]:
        return [
            Candidate(
                rung,
                D.Config("LTMLE", K, m, law, plan, learners=D.RUNGS[rung]),
                D.Config("plugin", K, m, law, plan, learners=D.RUNGS[rung]),
            )
            for rung in D.RUNGS
        ]

    return build


def _both_wrong(m: int) -> Callable[[int, Law], list[Candidate]]:
    first, second = ("respond", "never") if m == 2 else ("respond", "low")
    table = (
        ("C1", "constant", "glm"),
        ("C2", "wblind", "glm"),
        ("C3", "wblind", "wblind"),
        ("C4", "wblind_retention_one", "wblind"),
        ("C5", "confounder_blind", "int"),
    )

    def build(K: int, law: Law) -> list[Candidate]:
        positive = D.Config("LTMLE", K, m, law, first, second)
        return [
            Candidate(
                label,
                positive,
                D.Config("LTMLE", K, m, law, first, second, learners=learner, mechanism=mech),
            )
            for label, mech, learner in table
        ]

    return build


def _censoring(m: int, estimator: str) -> Callable[[int, Law], list[Candidate]]:
    plan = "coin" if m == 2 else "mix"

    def build(K: int, law: Law) -> list[Candidate]:
        return _ladder(K, m, law, plan, estimator, mechanism="retention_one")

    return build


def _third(estimator: str) -> Callable[[int, Law], list[Candidate]]:
    def build(K: int, law: Law) -> list[Candidate]:
        out = []
        for where, nodes in (("node K", (K,)), ("every node", tuple(range(1, K + 1)))):
            for rung in D.RUNGS:
                positive = D.Config(estimator, K, 3, law, "mix", learners=D.RUNGS[rung])
                control = D.Config(
                    estimator,
                    K,
                    3,
                    law,
                    "mix",
                    learners=D.RUNGS[rung],
                    mechanism="complement",
                    complement_nodes=nodes,
                )
                out.append(Candidate(f"{where} {rung}", positive, control))
        return out

    return build


def _policy(m: int, estimator: str) -> Callable[[int, Law], list[Candidate]]:
    plan = "coin" if m == 2 else "mix"

    def build(K: int, law: Law) -> list[Candidate]:
        positive = D.Config(estimator, K, m, law, plan)
        return [
            Candidate(name, positive, D.Config(estimator, K, m, law, name, target=plan))
            for name in ("uniform", "mix_node1", "mix_norot", "mix_reversed")
        ]

    return build


def _rule(m: int) -> Callable[[int, Law], list[Candidate]]:
    def build(K: int, law: Law) -> list[Candidate]:
        positive = D.Config("LTMLE", K, m, law, "respond")
        return [
            Candidate(name, positive, D.Config("LTMLE", K, m, law, name, target="respond"))
            for name in ("lagged", "reversed")
        ]

    return build


def _grid(K: int, m: int, gammas: Iterable[float] = (1.0,)) -> tuple[tuple[int, Law], ...]:
    """The grid laws in the fallback order at ``(K, m)``, each crossed with ``gammas``."""
    return tuple(
        (K, Law(law.B, law.s, law.c0, gamma)) for law in D.fallback_order(K, m) for gamma in gammas
    )


@functools.cache
def scans() -> tuple[Scan, ...]:
    """Every control family's candidate scan, in the declared order of section 5.3."""
    out = []
    for m, plan, K, law, *_ in D.TARGETING:
        study = "S1" if m == 2 else "S2"
        cell_n = D.round_n(D.tail(D.Quantity(K, m, plan), law).n_kappa)
        out.append(
            Scan(
                f"{study}:targeting_necessity:{plan}:K{K}",
                K,
                law,
                cell_n,
                (),
                D.NMAX,
                f"{plan}__k{K}__untargeted",
                _targeting(m, plan),
            )
        )
    for study, m, cell_n in (("S1", 2, 40_000), ("S2", 3, 48_000)):
        out.append(
            Scan(
                f"{study}:double_robustness",
                5,
                law_.L_R,
                cell_n,
                (),
                D.NMAX,
                "respond_vs_ref__k5__both_wrong",
                _both_wrong(m),
            )
        )
    for study, m, estimator, cell_law in (
        ("S1", 2, "LTMLE", law_.L_R),
        ("S2", 3, "LTMLE", Law(1.0, 0.05, 3.5)),
        ("S3", 2, "SDR", law_.L_R),
        ("S3", 3, "SDR", Law(1.0, 0.05, 3.5)),
    ):
        plan = "coin" if m == 2 else "mix"
        cell_n = D.round_n(D.tail(D.Quantity(5, m, plan), cell_law).n_kappa)
        out.append(
            Scan(
                f"{study}:censoring_necessity:m{m}",
                5,
                cell_law,
                cell_n,
                _grid(5, m, (1.0, 2.0, 4.0)),
                D.NMAX,
                f"{plan}__k5__ignored",
                _censoring(m, estimator),
            )
        )
    mix_law = Law(1.0, 0.05, 3.5)
    out.append(
        Scan(
            "S2:categorical_probability_necessity",
            5,
            mix_law,
            96_000,
            _grid(5, 3),
            D.NMAX_Q14,
            "third_arm__k5__binary_complement",
            _third("LTMLE"),
        )
    )
    out.append(
        Scan(
            "S3:categorical_probability_necessity",
            5,
            mix_law,
            96_000,
            _grid(5, 3),
            D.NMAX,
            "third_arm__k5__binary_complement",
            _third("SDR"),
        )
    )
    for study, m, estimator, cell_law in (
        ("S1", 2, "LTMLE", law_.L_R),
        ("S2", 3, "LTMLE", mix_law),
        ("S3", 2, "SDR", law_.L_R),
        ("S3", 3, "SDR", mix_law),
    ):
        plan = "coin" if m == 2 else "mix"
        cell_n = D.round_n(D.tail(D.Quantity(5, m, plan), cell_law).n_kappa)
        candidates = ("uniform", "mix_node1", "mix_norot", "mix_reversed")
        declared = next(
            c.cell
            for c in D.controls()
            if c.study == study and c.family == "policy_necessity" and c.config.m == m
        )
        assert declared.split("__")[-1].removesuffix("_control") in candidates
        out.append(
            Scan(
                f"{study}:policy_necessity:m{m}",
                5,
                cell_law,
                cell_n,
                (),
                D.NMAX,
                declared,
                _policy(m, estimator),
            )
        )
    out.append(
        Scan(
            "S1:rule_necessity",
            5,
            law_.L_R,
            40_000,
            _grid(5, 2),
            D.NMAX,
            "respond__k5__reversed_rule",
            _rule(2),
        )
    )
    out.append(
        Scan(
            "S2:rule_necessity",
            5,
            law_.L_R,
            48_000,
            _grid(5, 3) + _grid(4, 3) + _grid(3, 3),
            D.NMAX,
            "respond__k3__reversed_rule",
            _rule(3),
        )
    )
    return tuple(out)


def _reading(candidate: Candidate) -> dict[str, Any]:
    """The positive's ``n'`` and support, and the control's exact reading."""
    control = candidate.control.exact()
    positive = candidate.positive
    plans = [p for p in (positive.target or positive.first, positive.second) if p]
    return {
        "label": candidate.label,
        "positive_n_kappa": positive.positive_tail(),
        "support": min(D.support_minimum(positive.K, positive.m, p, positive.law) for p in plans),
        "bias": control.bias,
        "sigma": control.sigma,
        "sigma_curve": control.sigma_curve,
    }


def _scan_part(scan: Scan, K: int, law: Law, jobs: int) -> dict[str, Any]:
    candidates = scan.build(K, law)
    readings = Parallel(n_jobs=max(1, jobs))(delayed(_reading)(c) for c in candidates)
    return {"scan": scan.name, "K": K, "law": list(law.key), "rows": readings}


def _n_for(row: dict[str, Any], start: int, cap: int) -> int | None:
    if row["bias"] == 0.0:
        return None
    need = D.round_n(max(start, (D.D_FLOOR * row["sigma"] / abs(row["bias"])) ** 2))
    return need if need <= cap else None


def pick(
    scan: Scan, results: dict[tuple[int, tuple[float, ...]], dict[str, Any]]
) -> dict[str, Any]:
    """The declared candidate rule of section 5.3 over a scan's readings.

    Stage A: on the cell law, the first candidate with ``d >= 3`` at
    ``n' = max(cell n, n_kappa+)``.  Stage B: the smallest n at least ``n'`` with ``d >= 3`` on
    the cell law.  Stage C (and the K step): the first law in order on which the positive meets
    the tail criterion and the support floor and some candidate reaches ``d >= 3`` at
    ``n <= cap``; on it the smallest n, ties in list order.
    """
    K0 = scan.K
    cell = results.get((K0, scan.cell_law.key))
    if cell is None:
        return {"stage": "incomplete", "law": scan.cell_law.key, "K": K0}
    if cell is not None:
        best = None
        for index, row in enumerate(cell["rows"]):
            start = D.round_n(max(scan.cell_n, row["positive_n_kappa"]))
            d = abs(row["bias"]) / (row["sigma"] / math.sqrt(start))
            if d >= D.D_FLOOR and start <= D.NMAX:
                return {
                    "stage": "A",
                    "label": row["label"],
                    "n": start,
                    "law": scan.cell_law.key,
                    "K": K0,
                    "positive_n": start,
                }
            need = _n_for(row, start, D.NMAX)
            if need is not None and (best is None or (need, index) < best[:2]):
                best = (need, index, row["label"], start)
        if best is not None:
            return {
                "stage": "B",
                "label": best[2],
                "n": best[0],
                "law": scan.cell_law.key,
                "K": K0,
                "positive_n": best[3],
            }
    for K, law in scan.laws:
        found = results.get((K, law.key))
        if found is None:
            return {"stage": "incomplete", "law": law.key, "K": K}
        best = None
        for index, row in enumerate(found["rows"]):
            start = D.round_n(row["positive_n_kappa"])
            if start > scan.cap or row["support"] < D.SUPPORT_FLOOR:
                continue
            need = _n_for(row, start, scan.cap)
            if need is not None and (best is None or (need, index) < best[:2]):
                best = (need, index, row["label"], start)
        if best is not None:
            stage = "C" if K == K0 else "D"
            return {
                "stage": stage,
                "label": best[2],
                "n": best[0],
                "law": law.key,
                "K": K,
                "positive_n": best[3],
            }
    return {"stage": "stop"}


def _n_min_table(scan: Scan, results: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """Q14: every (law, candidate)'s smallest n at least the positive's ``n'``, with no cap."""
    rows = []
    for (_, key), found in results.items():
        for row in found["rows"]:
            start = D.round_n(row["positive_n_kappa"])
            if row["support"] < D.SUPPORT_FLOOR:
                continue
            need = _n_for(row, start, sys.maxsize)
            rows.append({"law": key, "label": row["label"], "n_min": need})
    return sorted(rows, key=lambda r: math.inf if r["n_min"] is None else r["n_min"])


def _patterns_part(jobs: int) -> dict[str, Any]:
    """The S3 pattern pairs (R, M): both positives and the three controls' readings."""
    pairs = [
        (r, mech)
        for r in ("glm", "wblind", "int", "lblind")
        for mech in ("constant", "wblind", "wblind_retention_one")
    ]

    def one(rung: str, mech: str) -> dict[str, Any]:
        out: dict[str, Any] = {"rung": rung, "mechanism": mech}
        configs = {
            "P_odd": D.pattern("SDR", "respond", True, rung=rung, mechanism=mech),
            "P_even": D.pattern("SDR", "respond", False, rung=rung, mechanism=mech),
            "node3": D.pattern("SDR", "respond", True, (3,), rung=rung, mechanism=mech),
            "odd": D.pattern("SDR", "respond", True, (1, 3, 5), rung=rung, mechanism=mech),
            "ltmle": D.pattern("LTMLE", "respond", True, rung=rung, mechanism=mech),
        }
        for name, config in configs.items():
            reading = config.exact()
            out[name] = {
                "bias": reading.bias,
                "sigma": reading.sigma,
                "n_kappa": sampling.n_kappa(reading.kappa),
            }
        return out

    rows = Parallel(n_jobs=max(1, jobs))(delayed(one)(r, m) for r, m in pairs)
    return {"scan": "S3:sequential_double_robustness", "rows": rows}


def pattern_pick(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stages A and B with ``node3_both_wrong``, then stage D with ``odd_nodes_both_wrong``."""
    for control in ("node3", "odd"):
        best = None
        for index, row in enumerate(rows):
            start = D.round_n(max(row["P_odd"]["n_kappa"], row["P_even"]["n_kappa"]))
            needs = [_n_for(row[name], start, D.NMAX) for name in (control, "ltmle")]
            if None in needs:
                continue
            largest = max(n for n in needs if n is not None)
            if best is None or (largest, index) < best[:2]:
                best = (largest, index, row, start, needs)
        if best is not None:
            row = best[2]
            return {
                "stage": "B" if control == "node3" else "D",
                "rung": row["rung"],
                "mechanism": row["mechanism"],
                "positive_n": best[3],
                "n": best[4],
            }
    return {"stage": "stop"}


# ----------------------------------------------------------------------------- parts
@dataclass(frozen=True)
class Part:
    """One unit of work with its own result file.

    Parameters
    ----------
    family : str
        ``lawscan``, ``scan``, ``patterns`` or a sample family.
    name : str
        The stream, scan or quantity.
    detail : tuple
        The law and K of a scan part, or the replicate block ``(start, stop)`` of a sample part.
    hours : float
        The estimated CPU hours.
    """

    family: str
    name: str
    detail: tuple[Any, ...]
    hours: float

    @property
    def ident(self) -> str:
        return f"{self.family}|{self.name}|{self.detail}"

    @property
    def filename(self) -> str:
        digest = hashlib.sha256(self.ident.encode()).hexdigest()[:16]
        suffix = "csv" if self.family in SAMPLE_FAMILIES else "json"
        return f"{self.family}-{digest}.{suffix}"


def _scan_hours(K: int, m: int, candidates: int) -> float:
    per = {(5, 3): 30.0, (4, 3): 4.0, (3, 3): 0.5}.get((K, m), 1.0)
    return candidates * per / 3600


@functools.cache
def parts() -> tuple[Part, ...]:
    """Every part, in the fixed order: exact scans first, then the sample streams."""
    out = []
    for q in D.FALLBACKS:
        out.append(Part("lawscan", str(q), (q.K, q.m, q.first, q.second), 36 * 2 * 0.5 / 3600))
    for scan in scans():
        laws = ((scan.K, scan.cell_law), *scan.laws)
        seen = set()
        for K, law in laws:
            if (K, law.key) in seen:
                continue
            seen.add((K, law.key))
            m = scan.build(K, law)[0].control.m
            hours = _scan_hours(K, m, len(scan.build(K, law)))
            out.append(Part("scan", scan.name, (K, law.key), hours))
    out.append(Part("patterns", "S3:sequential_double_robustness", (), 12 * 5 * 3 / 3600))
    for stream in D.streams():
        per = stream.cost()
        for start in range(0, stream.replicates, BLOCK):
            stop = min(start + BLOCK, stream.replicates)
            out.append(
                Part(stream.family, stream.ident, (start, stop), per * (stop - start) / 3600)
            )
    return tuple(out)


def _stream(ident: str) -> D.Stream:
    return next(s for s in D.streams() if s.ident == ident)


def _scan(name: str) -> Scan:
    return next(s for s in scans() if s.name == name)


# ----------------------------------------------------------------------------- the key
def _code_digest() -> dict[str, str]:
    files = sorted((ROOT / "src" / "cleverly").rglob("*.py")) + [ROOT / f for f in CODE]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def key_inputs() -> dict[str, Any]:
    from tests.canonical.regenerate import python_versions

    return {
        "parts": [part.ident for part in parts()],
        "block": BLOCK,
        "pilot_seed": D.PILOT_SEED,
        "code": _code_digest(),
        "versions": python_versions(),
    }


def key() -> str:
    text = json.dumps(key_inputs(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def default_scratch() -> Path:
    from tests.canonical.regenerate import runs_root

    return Path(os.environ.get("X39_PILOT_SCRATCH", runs_root().parent))


def claim(root: Path) -> None:
    """Stamp ``root`` with the key, or refuse a root stamped with another."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "KEY.json"
    inputs = key_inputs()
    current = key()
    if stamp.exists():
        stored = json.loads(stamp.read_text(encoding="utf-8"))
        if stored["key"] != current:
            raise SystemExit(
                f"{root} holds parts of key {stored['key']}, not {current}: the declaration, "
                "the code or a package version changed"
            )
        return
    stamp.write_text(
        json.dumps({"key": current, **inputs}, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


# ----------------------------------------------------------------------------- running a part
def _sample_fit(
    stream: D.Stream, key_: tuple[int, int], drawn: pd.DataFrame
) -> list[dict[str, Any]]:
    replicate, seed = key_
    base = {"stream": stream.ident, "replicate": replicate, "seed": seed, "n": stream.n}
    if stream.config is None:
        study, name = stream.scenario
        module = D.STUDIES[study]
        scenario = next(s for s in module.SCENARIOS if s.name == name)
        result = primary.fit(drawn, scenario, module.ESTIMATOR)
        rows = primary.rows(module.STUDY, result, scenario, replicate, stream.n)
        return [
            {
                **base,
                **{
                    k: row[k]
                    for k in (
                        "estimand",
                        "truth",
                        "estimate",
                        "std_error",
                        "ci_lower",
                        "ci_upper",
                        "covered",
                    )
                },
            }
            for row in rows
        ]
    fitted = stream.config.fit_sample(drawn)
    truth = stream.config.truth
    covered = (
        int(fitted["ci_lower"] <= truth <= fitted["ci_upper"])
        if math.isfinite(fitted["ci_lower"])
        else -1
    )
    return [{**base, "estimand": "", "truth": truth, **fitted, "covered": covered}]


def _draw(stream: D.Stream, key_: tuple[int, int]) -> pd.DataFrame:
    _, seed = key_
    if stream.config is None:
        study, name = stream.scenario
        scenario = next(s for s in D.STUDIES[study].SCENARIOS if s.name == name)
        return law_.draw(scenario.K, scenario.m, stream.n, seed, scenario.law)
    c = stream.config
    return law_.draw(c.K, c.m, stream.n, seed, c.law)


def _run_sample(part: Part, root: Path, jobs: int) -> pd.DataFrame:
    stream = _stream(part.name)
    start, stop = part.detail
    keys = [(r, D.stream_seed(stream.ident, r)) for r in range(start, stop)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcomes = replication.fit_replicates(
            keys,
            functools.partial(_sample_fit, stream),
            draw=functools.partial(_draw, stream),
            n_jobs=jobs,
            label=part.family,
            implementation=IMPLEMENTATION,
            cell_of=lambda key_: stream.ident,
            replicate_of=lambda key_: key_[0],
            seed_of=lambda key_: key_[1],
            checkpoint=root / "checkpoints",
        )
    rows = [row for fitted in outcomes.fitted if fitted for row in fitted]
    frame = pd.DataFrame(rows)
    failures = outcomes.failures
    if len(failures):
        failed = pd.DataFrame(
            {
                "stream": stream.ident,
                "replicate": [keys[i][0] for i in outcomes.failed],
                "seed": [keys[i][1] for i in outcomes.failed],
                "n": stream.n,
                "failure": failures["message"].to_numpy(),
            }
        )
        frame = pd.concat([frame, failed], ignore_index=True)
    if "failure" not in frame:
        frame["failure"] = ""
    frame["failure"] = frame["failure"].fillna("")
    return frame.sort_values(["replicate", "estimand"], kind="stable", ignore_index=True)


def compute(part: Part, root: Path, jobs: int) -> bytes:
    """The result file's bytes of ``part``."""
    if part.family in SAMPLE_FAMILIES:
        frame = _run_sample(part, root, jobs)
        return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    if part.family == "lawscan":
        K, m, first, second = part.detail
        pick_, rows = D.fallback_scan(D.Quantity(K, m, first, second))
        payload: dict[str, Any] = {
            "quantity": part.name,
            "pick": None if pick_ is None else list(pick_.key),
            "rows": rows,
        }
    elif part.family == "scan":
        K, law_key = part.detail
        payload = _scan_part(_scan(part.name), K, Law(*law_key), jobs)
    elif part.family == "patterns":
        payload = _patterns_part(jobs)
    else:
        raise ValueError(f"unknown part family {part.family!r}")
    return (json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def run(root: Path, indices: Sequence[int], jobs: int) -> None:
    claim(root)
    every = parts()
    for index in indices:
        part = every[index]
        path = root / "parts" / part.filename
        if path.exists():
            continue
        print(
            f"[{index}] {part.family} {part.name} {part.detail} ~{part.hours:.2f} CPU-h", flush=True
        )
        _write(path, compute(part, root, jobs))
        print(f"[{index}] done", flush=True)
    print("complete", flush=True)


def check(root: Path, index: int, jobs: int) -> bool:
    """Recompute part ``index`` in a temporary root and compare its bytes with the stored file."""
    part = parts()[index]
    stored = (root / "parts" / part.filename).read_bytes()
    with tempfile.TemporaryDirectory() as scratch:
        fresh = compute(part, Path(scratch), jobs)
    same = fresh == stored
    print(f"[{index}] {part.family} {part.name}: {'identical' if same else 'DIFFERENT'}")
    return same


# ----------------------------------------------------------------------------- merging
def _load_json(root: Path, part: Part) -> dict[str, Any]:
    return dict(json.loads((root / "parts" / part.filename).read_text(encoding="utf-8")))


def _read_sample(root: Path, family: str) -> pd.DataFrame:
    frames = []
    for part in parts():
        if part.family != family:
            continue
        frame = pd.read_csv(
            root / "parts" / part.filename,
            float_precision="round_trip",
            dtype={"estimand": str, "failure": str},
            keep_default_na=False,
        )
        for column in ("estimate", "std_error", "ci_lower", "ci_upper", "truth"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        start, stop = part.detail
        replicates = set(frame["replicate"].astype(int))
        if replicates != set(range(start, stop)):
            raise SystemExit(f"{part.name} {part.detail}: replicates {sorted(replicates)[:5]}...")
        expected = {r: D.stream_seed(part.name, r) for r in range(start, stop)}
        wrong = [
            r
            for r, s in zip(frame["replicate"], frame["seed"], strict=True)
            if expected[int(r)] != int(s)
        ]
        if wrong:
            raise SystemExit(f"{part.name}: replicates {wrong[:5]} carry another seed")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bootstrap_ratio(estimates: np.ndarray, errors: np.ndarray, seed: int) -> tuple[float, float]:
    """The SE ratio ``mean(se) / sd(estimate)`` and its bootstrap variance."""
    rng = np.random.default_rng(seed)
    ratio = float(np.mean(errors) / np.std(estimates, ddof=1))
    draws = rng.integers(0, len(estimates), size=(1_000, len(estimates)))
    boot = np.mean(errors[draws], axis=1) / np.std(estimates[draws], axis=1, ddof=1)
    return ratio, float(np.var(boot, ddof=1))


def read_mc(frame: pd.DataFrame) -> list[str]:
    lines = []
    declared = {
        s.ident: c
        for s, c in zip(
            [s for s in D.streams() if s.family == "mc"],
            [c for c in D.controls() if c.monte_carlo],
            strict=True,
        )
    }
    for stream, rows in frame.groupby("stream", sort=False):
        control = declared[str(stream)]
        kept = rows[rows["failure"] == ""]
        errors = kept["estimate"].to_numpy(float) - kept["truth"].to_numpy(float)
        R = len(errors)
        sd = float(np.std(errors, ddof=1))
        sd_error = sd / math.sqrt(2 * (R - 1))
        bias_error = sd / math.sqrt(R)
        exact_sd = control.sigma / math.sqrt(control.n)
        ok = (
            abs(sd - exact_sd) <= 3 * sd_error
            and abs(np.mean(errors) - control.bias) <= 3 * bias_error
        )
        lines.append(
            f"{control.study} {control.family} {control.cell}: R {R} bias {np.mean(errors):+.5f} "
            f"(exact {control.bias:+.5f}, MC error {bias_error:.5f}) SD {sd:.5f} "
            f"(exact {exact_sd:.5f}, MC error {sd_error:.5f}) -> {'pass' if ok else 'STOP'}"
        )
    return lines


def read_f2(frame: pd.DataFrame) -> list[str]:
    lines = []
    by_study: dict[str, list[tuple[float, float]]] = {}
    for index, (stream, rows) in enumerate(frame.groupby("stream", sort=False)):
        kept = rows[rows["failure"] == ""]
        ratio, variance = _bootstrap_ratio(
            kept["estimate"].to_numpy(float), kept["std_error"].to_numpy(float), index
        )
        study = str(stream).split("|")[1].split(":")[0]
        by_study.setdefault(study, []).append((ratio, variance))
        coverage = float(np.mean(kept["covered"]))
        lines.append(
            f"{stream.split('|')[1]}: SE ratio {ratio:.4f} (SD {math.sqrt(variance):.4f}) "
            f"coverage {coverage:.4f}"
        )
    for study, arms in by_study.items():
        ratios, variances = zip(*arms, strict=True)
        chosen = None
        for replicates in range(400, 16_001, 400):
            if D.calibration_pass(1.0, replicates) ** len(arms) < 0.95:
                continue
            if D.f2_family_pass(ratios, variances, replicates) >= 0.95:
                chosen = replicates
                break
        lines.append(f"{study}: k {len(arms)} F2 R {chosen if chosen else 'none -> stop'}")
    return lines


def _ladder_pick(probabilities: Callable[[int], float], ladder: Sequence[int]) -> int | None:
    return next((R for R in ladder if probabilities(R) >= 0.95), None)


def read_bias(frame: pd.DataFrame) -> list[str]:
    lines = []
    biases: dict[str, list[float]] = {}
    for stream, rows in frame.groupby("stream", sort=False):
        kept = rows[rows["failure"] == ""]
        errors = kept["estimate"].to_numpy(float) - kept["truth"].to_numpy(float)
        standardized = float(np.mean(errors) / np.std(errors, ddof=1))
        study = str(stream).split("|")[1].split(":")[0]
        biases.setdefault(study, []).append(standardized)
        lines.append(
            f"{stream.split('|')[1]}: standardized bias {standardized:+.4f} "
            f"(SD {1 / math.sqrt(len(errors)):.4f})"
        )
    for study, values in biases.items():
        chosen = _ladder_pick(
            lambda R, values=values: float(np.prod([D.bias_pass(abs(b), R) for b in values])),
            BIAS_LADDER,
        )
        lines.append(f"{study}: bias R {chosen if chosen else 'none -> stop'}")
    return lines


def read_truth(frame: pd.DataFrame) -> list[str]:
    lines = []
    rows_by_study: dict[str, list[tuple[float, float]]] = {}
    kept = frame[frame["failure"] == ""]
    for (stream, _), rows in kept.groupby(["stream", "estimand"], sort=False):
        errors = rows["estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
        coverage = float(np.mean(rows["covered"]))
        bias = float(np.mean(errors) / np.std(errors, ddof=1))
        study = str(stream).split("|")[1].split(":")[0]
        rows_by_study.setdefault(study, []).append((coverage, bias))
    for study, values in rows_by_study.items():
        chosen = _ladder_pick(
            lambda R, values=values: float(
                np.prod([D.truth_row_pass(min(c, 0.9999), abs(b), R) for c, b in values])
            ),
            TRUTH_LADDER,
        )
        lines.append(f"{study}: {len(values)} truth rows, R {chosen if chosen else 'none -> stop'}")
    return lines


def read_overfit(frame: pd.DataFrame) -> list[str]:
    lines = []
    for index, (stream, rows) in enumerate(frame.groupby("stream", sort=False)):
        kept = rows[rows["failure"] == ""]
        ratio, variance = _bootstrap_ratio(
            kept["estimate"].to_numpy(float), kept["std_error"].to_numpy(float), index
        )
        spread = math.sqrt(variance * 3_200 / 2_000)
        half = 2.576 * spread
        from scipy.stats import norm

        passed = norm.cdf((1.20 - half - ratio) / spread) - norm.cdf((0.85 + half - ratio) / spread)
        lines.append(
            f"{stream.split('|')[1]}: SE ratio {ratio:.4f}, pass probability at "
            f"R = 2,000 {passed:.3f}"
        )
    return lines


READERS: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "mc": read_mc,
    "f2": read_f2,
    "bias": read_bias,
    "truth": read_truth,
    "overfit": read_overfit,
}


def _exists(root: Path, part: Part) -> bool:
    return (root / "parts" / part.filename).exists()


def read_scans(root: Path) -> list[str]:
    """Each exact scan's pick against the declared one; a scan with a missing part says so."""
    lines = []
    declared = {(c.study, c.cell): c for c in D.controls()}
    for scan in scans():
        mine = [p for p in parts() if p.family == "scan" and p.name == scan.name]
        missing = sum(not _exists(root, p) for p in mine)
        results = {}
        for part in mine:
            if _exists(root, part):
                found = _load_json(root, part)
                results[(found["K"], tuple(found["law"]))] = found
        chosen = pick(scan, results) if results else {"stage": "incomplete"}
        control = declared.get((scan.study, scan.declared))
        expected = (control.n, control.config.law.key) if control else None
        lines.append(
            f"{scan.name}: pick {chosen}; declared (n, law) {expected}; "
            f"{missing} of {len(mine)} parts missing"
        )
        if scan.cap == D.NMAX_Q14 and not missing:
            lines.append(f"  smallest n_min: {_n_min_table(scan, results)[:5]}")
    patterns = next(p for p in parts() if p.family == "patterns")
    if _exists(root, patterns):
        lines.append(f"S3 patterns: pick {pattern_pick(_load_json(root, patterns)['rows'])}")
    for part in parts():
        if part.family == "lawscan" and _exists(root, part):
            found = _load_json(root, part)
            q = next(q for q in D.FALLBACKS if str(q) == part.name)
            lines.append(
                f"{part.name}: rule picks {found['pick']}, declared {list(D.FALLBACKS[q].key)}"
            )
    return lines


def merge(root: Path) -> None:
    claim(root)
    missing = [i for i, part in enumerate(parts()) if not (root / "parts" / part.filename).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} parts missing; first {missing[:10]}")
    out = root / "merged"
    out.mkdir(exist_ok=True)
    report = read_scans(root)
    for family in SAMPLE_FAMILIES:
        frame = _read_sample(root, family)
        frame.to_csv(
            out / f"{family}.csv.gz",
            index=False,
            lineterminator="\n",
            float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        failures = int((frame["failure"] != "").sum())
        report.append(f"== {family}: {len(frame)} rows, {failures} recorded failures")
        report += READERS[family](frame)
    text = "\n".join(report) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


# ----------------------------------------------------------------------------- the command line
def _indices(arguments: argparse.Namespace) -> list[int]:
    every = parts()
    chosen = [
        i
        for i, part in enumerate(every)
        if arguments.family is None or part.family == arguments.family
    ]
    if arguments.down_to is not None:
        return [i for i in reversed(chosen) if i >= arguments.down_to]
    stop = len(every) if arguments.to is None else arguments.to
    return [i for i in chosen if arguments.start <= i < stop]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.long_horizon_pilots")
    parser.add_argument(
        "command", choices=("list", "split", "run", "check", "merge", "key", "scans")
    )
    parser.add_argument("index", nargs="?", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--from", dest="start", type=int, default=0)
    parser.add_argument("--to", type=int)
    parser.add_argument("--down-to", type=int)
    parser.add_argument("--family")
    parser.add_argument("--share", type=float, default=0.2)
    arguments = parser.parse_args(argv)
    every = parts()
    if arguments.command == "list":
        total = 0.0
        for index, part in enumerate(every):
            if arguments.family is not None and part.family != arguments.family:
                continue
            total += part.hours
            print(
                f"{index:5d} {part.family:8s} {part.hours:7.3f} {total:8.1f}  {part.name} "
                f"{part.detail}"
            )
        print(f"{len(every)} parts, {sum(p.hours for p in every):.1f} CPU-h")
        return
    if arguments.command == "split":
        whole = sum(p.hours for p in every)
        running = 0.0
        for index, part in enumerate(every):
            running += part.hours
            if running >= arguments.share * whole:
                print(index + 1)
                return
        print(len(every))
        return
    if arguments.command == "key":
        print(key())
        return
    root = arguments.root or default_scratch() / "x39-pilots" / key()
    print(f"root {root}", flush=True)
    if arguments.command == "run":
        run(root, _indices(arguments), arguments.jobs)
    elif arguments.command == "check":
        if arguments.index is None:
            parser.error("check takes a part index")
        if not check(root, arguments.index, arguments.jobs):
            raise SystemExit(1)
    elif arguments.command == "scans":
        print("\n".join(read_scans(root)))
    else:
        merge(root)


if __name__ == "__main__":
    main()
