"""The pilot driver of the long-horizon design: exact scans and sample pilots, in parts.

The work is a fixed, ordered list of *parts*.  A part is one exact scan of the design (a law of
a candidate scan, one quantity's fallback scan), one per-node targeting witness on a fallback
law, or one block of :data:`BLOCK` replicates of one pilot stream
(:func:`tests.long_horizon_design.streams`).  Each part writes its own result file under
``<root>/parts``, and its content depends on the part alone: every replicate draws from its own
seed (:func:`~tests.long_horizon_design.stream_seed`), so any machine may compute any part and
two machines write the same bytes for it.

Rev 4.2 appends parts and moves none: the first :data:`REV_41_PARTS` idents, their order and
their seeds are those of rev 4.1 (``tests/unit/test_long_horizon_pilots.py`` pins them).  The
new parts are the band pilots, the root-n rung pilots and the per-node witness on each fallback
law.

Rev 4.3 appends again after the :data:`REV_42_PARTS` parts of rev 4.2 and moves none of them.
The new parts are four factor-2 fallback scans (the Wald regime ``n >= 2 n_kappa``, decision C1),
the ``truth_boot`` pilots (the refit bootstrap at B = 399 with the BCa interval, the S3 scenarios
first), the ``band_boot`` pilots (the studentized sup-t band at B = 399) and the ``f2_c1`` pilots
(the calibration cells that C1 moved).  ``run`` refuses a ``truth_boot`` part of S1 or S2 until
the S3 ``truth_boot`` parts exist and their reading passes, because the S3 parts are the first
evidence that BCa holds for the SDR estimator (plan rev 4.3, Q-E5).

The root is ``<scratch>/x39-pilots/<key>``.  The key is a digest of the declared parts, the bytes
of the code that runs them and the Python package versions, so a changed declaration, estimator
or package never reuses a stale part.  A root stamped with another key is refused, unless the
caller accepts that key and the equivalence records cover it (below).

Commands (run from the repository root with ``PYTHONPATH=.``)::

    python -m tests.long_horizon_pilots list [--family F]
    python -m tests.long_horizon_pilots split --share S
    python -m tests.long_horizon_pilots run [--jobs N] [--from I] [--to J | --down-to I] [--family F]
    python -m tests.long_horizon_pilots kinds [--from I] [--to J] [--root R]
    python -m tests.long_horizon_pilots check I [I ...] [--root R] [--rtol T]
    python -m tests.long_horizon_pilots merge [--root R] [--source R2 ...] [--accept-key K ...]
    python -m tests.long_horizon_pilots scans

``list`` prints each part's index, family, name and estimated CPU hours, with the running total,
then the part count and CPU hours of each family.
``split --share S`` prints the first index at which the running total passes ``S`` of the whole.
``run`` computes the parts ``[I, J)`` in ascending order, or with ``--down-to I`` from the last
part down to ``I``; a part whose file exists is skipped, and a sample part resumes from the
per-replicate checkpoints of :func:`tests.studies.evidence.replication.fit_replicates` under
``<root>/checkpoints``.

**Equivalence.**  Parts computed at different code states carry different keys.  ``kinds``
prints, for the indices in ``[I, J)``, the cheapest part of each configuration kind
(:data:`KINDS`).  ``check I ...`` recomputes each part in a temporary root and compares it with
the stored file: numeric floating-point columns with ``numpy.isclose(rtol, atol=0)``, and
integer, boolean and text columns exactly.  A JSON part compares the same way, value by value.
Each check writes a record under ``<root>/equivalence/`` with the part index, the stored and the
current key, the kind, the result and the largest relative difference.  ``merge`` takes
``--source R2`` (repeatable) for the parts of other roots, such as an unzipped local share, and
``--accept-key K`` (repeatable) for the stamped keys it may read.  It reads a part from a root
stamped with an accepted key only when that root's records, written at the current key, cover
every kind of the parts it contributes at ``--rtol``.  Otherwise it refuses and names the kinds.

``merge`` refuses a missing part or a replicate whose seed is not its declared seed, writes
``<root>/merged/<family>.csv.gz``, and prints each family's reading: the Monte Carlo check, the
F2 and bias floors, the truth-row floor, the overfitting pilot, the band and root-n readings and
every exact scan's pick against the declared one.  It exits with status 1 after printing when a
reading stops (``-> stop``).  ``scans`` prints the exact scans' picks from the parts that exist.

A pilot above n = 8,000 runs only when no study runs on the same machine, after a one-fit memory
check.  Step 4b's property cells use the same machinery: a property family adds its parts to
:func:`parts` and its reading to :data:`READERS`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
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
from tests.studies import long_horizon_mutations as mutations
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
    "tests/studies/long_horizon_properties.py",
    "tests/studies/fractional_glm.py",
    "tests/studies/canonical_long_horizon_ltmle.py",
    "tests/studies/canonical_long_horizon_categorical_ltmle.py",
    "tests/studies/canonical_long_horizon_sdr.py",
    "tests/studies/evidence/replication.py",
)
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FAMILIES = (
    "mc",
    "f2",
    "bias",
    "truth",
    "overfit",
    "band",
    "rung",
    "truth_boot",
    "band_boot",
    "f2_c1",
)
#: The parts of rev 4.1.  Rev 4.2 appends after them; none of their idents, order or seeds moves.
REV_41_PARTS = 3_288
#: The parts of rev 4.2.  Rev 4.3 appends after them; none of their idents, order or seeds moves.
REV_42_PARTS = 3_449
#: The configuration kinds of a part.  An equivalence record covers the kind of the part it
#: checks, and ``merge`` needs a record for every kind a foreign root contributes.
KINDS = (
    "ltmle",
    "sdr",
    "ltmle_fivefold",
    "sdr_fivefold",
    "trees",
    "msm",
    "plugin",
    "complement",
    "band",
    "ltmle_boot",
    "sdr_boot",
    "band_boot",
    "exact_scan",
    "law_scan",
)
#: The default relative tolerance of ``check`` and ``merge``.
RTOL = 1e-9
#: The columns of a sample part that are text.
TEXT_COLUMNS = ("stream", "estimand", "failure", "interval")
#: The bias-family replicate ladder and the truth-row ladder of section 5.3.
BIAS_LADDER = (1_000, 1_200, 2_000, 3_000, 4_000)
TRUTH_LADDER = (1_200, 2_000, 3_000)
#: The seed of the slope rows' resampling.
SLOPE_SEED = 20263932


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
        ``lawscan``, ``scan``, ``patterns``, ``witness`` or a sample family.
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
    # Rev 4.2: one per-node targeting witness per fallback law, after every earlier part.
    for law, settings in witness_laws():
        out.append(
            Part(
                "witness",
                f"witness:{D.law_name(law)}",
                (law.key, settings),
                _witness_hours(settings),
            )
        )
    # Rev 4.3: the factor-2 fallback scans, then the bootstrap pilots, S3 first.
    for q in D.CALIBRATION_FALLBACKS:
        out.append(
            Part(
                "lawscan",
                str(q),
                (q.K, q.m, q.first, q.second, D.WALD_REGIME_FACTOR),
                36 * 2 * 0.5 / 3600,
            )
        )
    for stream in D.bootstrap_streams():
        per = stream.cost()
        for start in range(0, stream.replicates, BLOCK):
            stop = min(start + BLOCK, stream.replicates)
            out.append(
                Part(stream.family, stream.ident, (start, stop), per * (stop - start) / 3600)
            )
    return tuple(out)


def witness_laws() -> list[tuple[Law, tuple[tuple[int, int], ...]]]:
    """Each distinct fallback law with the ``(K, m)`` settings of the quantities assigned to it."""
    seen: dict[Law, set[tuple[int, int]]] = {}
    for q, law in D.FALLBACKS.items():
        seen.setdefault(law, set()).add((q.K, q.m))
    return [(law, tuple(sorted(settings))) for law, settings in seen.items()]


def _witness_hours(settings: Sequence[tuple[int, int]]) -> float:
    """CPU hours of one witness part, from the measured seconds per plan (K = 5: 9, K = 4: 1.2)."""
    seconds = {(5, 3): 9.0, (4, 3): 1.2, (3, 3): 0.3}
    return (
        sum(len(mutations.WITNESS_PLANS[m]) * seconds.get((K, m), 1.0) for K, m in settings) / 3600
    )


def kind_of(part: Part) -> str:
    """The configuration kind of ``part`` (:data:`KINDS`)."""
    if part.family == "lawscan":
        return "law_scan"
    if part.family in ("scan", "patterns", "witness"):
        return "exact_scan"
    stream = _stream(part.name)
    if stream.family in ("band", "band_boot"):
        return stream.family
    if stream.config is None:
        study, name = stream.scenario
        scenario = next(s for s in D.scenarios(study) if s.name == name)
        if scenario.msm:
            return "msm"
        estimator, folds = ("SDR" if study == "S3" else "LTMLE"), scenario.n_folds
        if stream.family == "truth_boot":
            return f"{estimator.lower()}_boot"
    else:
        config = stream.config
        if config.msm:
            return "msm"
        if config.estimator == "plugin":
            return "plugin"
        if config.trees:
            return "trees"
        if config.complement_nodes:
            return "complement"
        estimator, folds = config.estimator, config.n_folds
    return f"{estimator.lower()}{'_fivefold' if folds > 1 else ''}"


@functools.cache
def _streams_by_ident() -> dict[str, D.Stream]:
    return {s.ident: s for s in (*D.streams(), *D.bootstrap_streams())}


def _stream(ident: str) -> D.Stream:
    return _streams_by_ident()[ident]


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
        boot = stream.family == "truth_boot"
        result = primary.fit(drawn, scenario, module.ESTIMATOR, bootstrap=boot)
        rows = primary.rows(module.STUDY, result, scenario, replicate, stream.n)
        columns = ("estimand", "truth", "estimate", "std_error", "ci_lower", "ci_upper", "covered")
        if boot:
            columns += ("interval", "n_failed", "sparse_warning")
        return [{**base, **{k: row[k] for k in columns}} for row in rows]
    if stream.family == "band":
        return [{**base, **_band_fit(stream, drawn)}]
    if stream.family == "band_boot":
        return [{**base, **_band_boot_fit(stream, drawn)}]
    fitted = stream.config.fit_sample(drawn, sparse=stream.family == "f2_c1")
    if stream.family == "f2_c1":
        fitted = {"sparse_warning": -1, **fitted}
    truth = stream.config.truth
    covered = (
        int(fitted["ci_lower"] <= truth <= fitted["ci_upper"])
        if math.isfinite(fitted["ci_lower"])
        else -1
    )
    return [{**base, "estimand": "", "truth": truth, **fitted, "covered": covered}]


def _band_fit(stream: D.Stream, drawn: pd.DataFrame) -> dict[str, Any]:
    """One replicate of the simultaneous band: the property phase's fit and its joint rows.

    ``covered`` is the band's joint coverage (every truth inside the band) and
    ``pointwise_covered`` the pointwise control's.  ``estimate`` is the max-t statistic and
    ``std_error`` the band's critical value, as in :mod:`tests.studies.evidence.simultaneous`.
    """
    from tests.studies import long_horizon_properties as properties
    from tests.studies.evidence.simultaneous import joint_coverage_rows

    config = stream.config
    assert config is not None
    arm = properties.Arm("k5", "positive", config, read="band", plans=stream.plans)
    result = properties._call(arm, drawn, stream.plans, simultaneous=True, reference=None)
    names = sorted(result.simultaneous.bands)
    band, pointwise = joint_coverage_rows(
        result,
        properties._band_truths(config.K, config.m, config.law, names),
        names,
        label="k5",
        replicate=0,
        n=stream.n,
        requested=0,
        pointwise_critical=properties.CRITICAL,
    )
    return {
        "estimand": "",
        "truth": 0.0,
        "estimate": band["estimate"],
        "std_error": band["std_error"],
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "covered": int(band["covered"]),
        "pointwise_covered": int(pointwise["covered"]),
    }


def _band_boot_fit(stream: D.Stream, drawn: pd.DataFrame) -> dict[str, Any]:
    """One replicate of the studentized bootstrap band, with the IC band and pointwise controls.

    ``covered`` is the sup-t band's joint coverage, ``estimate`` the max-t statistic and
    ``std_error`` the band's critical value.  ``ic_covered`` and ``ic_critical`` are the reporting
    Wald-curve band's, ``pointwise_covered`` the control's, and ``n_failed`` the failed refits of
    the fit's bootstrap.
    """
    from tests.studies import long_horizon_properties as properties

    config = stream.config
    assert config is not None
    arm = properties.Arm("k5", "positive", config, read="band", plans=stream.plans)
    result = properties._call(
        arm,
        drawn,
        stream.plans,
        simultaneous=True,
        reference=None,
        n_bootstrap=D.B_GATE,
    )
    names = sorted(result.simultaneous.bands)
    band, ic, pointwise = properties.band_rows(
        result, config, names, label="k5", replicate=0, n=stream.n, requested=0
    )
    return {
        "estimand": "",
        "truth": 0.0,
        "estimate": band["estimate"],
        "std_error": band["std_error"],
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "covered": int(band["covered"]),
        "ic_covered": int(ic["covered"]),
        "ic_critical": ic["std_error"],
        "pointwise_covered": int(pointwise["covered"]),
        "n_failed": int(result.bootstrap.n_failed),
    }


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
        K, m, first, second, *factor = part.detail
        pick_, rows = D.fallback_scan(
            D.Quantity(K, m, first, second), factor=factor[0] if factor else 1
        )
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
    elif part.family == "witness":
        law_key, settings = part.detail
        law = Law(*law_key)
        rows = [
            reading
            for K, m in settings
            for plan in mutations.WITNESS_PLANS[m]
            for reading in mutations.per_node_witness(K, m, plan, law)
        ]
        payload = {"witness": part.name, "law": list(law_key), "rows": rows}
    else:
        raise ValueError(f"unknown part family {part.family!r}")
    return (json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _is_s3_truth_boot(part: Part) -> bool:
    return part.family == "truth_boot" and part.name.split("|")[1].startswith("S3:")


def s3_gate(root: Path, part: Part) -> None:
    """Refuse a ``truth_boot`` part of S1 or S2 until the S3 ``truth_boot`` reading passes.

    The S3 parts are the first evidence that BCa holds for the SDR estimator (plan rev 4.3,
    Q-E5).  A missing S3 part or a reading that prints ``-> stop`` ends the run.
    """
    if part.family != "truth_boot" or _is_s3_truth_boot(part):
        return
    s3 = [p for p in parts() if _is_s3_truth_boot(p)]
    absent = [p.detail for p in s3 if not (root / "parts" / p.filename).exists()]
    if absent:
        raise SystemExit(
            f"{len(absent)} S3 truth_boot parts are missing: run them first "
            "(`run --family truth_boot --to <first S1 index>`); S1 and S2 follow the S3 reading"
        )
    lines = read_truth_boot(_read_sample(root, "truth_boot", _is_s3_truth_boot))
    stops = [line for line in lines if "-> stop" in line]
    if stops:
        print("\n".join(lines), flush=True)
        raise SystemExit("the S3 truth_boot reading stopped (-> stop): diagnose before S1 and S2")


def run(root: Path, indices: Sequence[int], jobs: int) -> None:
    claim(root)
    every = parts()
    for index in indices:
        part = every[index]
        path = root / "parts" / part.filename
        if path.exists():
            continue
        s3_gate(root, part)
        print(
            f"[{index}] {part.family} {part.name} {part.detail} ~{part.hours:.2f} CPU-h", flush=True
        )
        _write(path, compute(part, root, jobs))
        print(f"[{index}] done", flush=True)
    print("complete", flush=True)


@dataclass(frozen=True)
class Comparison:
    """The result of comparing a recomputed part with its stored file.

    Parameters
    ----------
    same : bool
        Whether every numeric floating-point value agrees at ``rtol`` and everything else agrees
        exactly.
    max_relative_difference : float
        The largest ``|fresh - stored| / |stored|`` over the floating-point values (infinite
        where the stored value is zero and the fresh one is not); 0 when none differs.
    detail : str
        The first difference, or empty.
    """

    same: bool
    max_relative_difference: float
    detail: str = ""


def _close(fresh: np.ndarray, stored: np.ndarray, rtol: float) -> tuple[bool, float]:
    """``isclose(rtol, atol=0)`` with equal NaNs, and the largest relative difference."""
    fresh, stored = np.asarray(fresh, float), np.asarray(stored, float)
    same = np.isclose(fresh, stored, rtol=rtol, atol=0.0, equal_nan=True)
    both_nan = np.isnan(fresh) & np.isnan(stored)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(fresh - stored) / np.abs(stored)
    relative = np.where((fresh == stored) | both_nan, 0.0, relative)
    relative = np.where(np.isnan(relative), np.inf, relative)
    return bool(same.all()), float(np.max(relative)) if relative.size else 0.0


def _compare_json(fresh: Any, stored: Any, rtol: float, where: str = "") -> Comparison:
    """Compare two parsed JSON values: floats at ``rtol``, everything else exactly."""
    if isinstance(fresh, dict) and isinstance(stored, dict):
        if fresh.keys() != stored.keys():
            return Comparison(False, math.inf, f"{where}: keys differ")
        worst = Comparison(True, 0.0)
        for name in stored:
            found = _compare_json(fresh[name], stored[name], rtol, f"{where}/{name}")
            if not found.same:
                return found
            worst = Comparison(
                True, max(worst.max_relative_difference, found.max_relative_difference)
            )
        return worst
    if isinstance(fresh, list) and isinstance(stored, list):
        if len(fresh) != len(stored):
            return Comparison(False, math.inf, f"{where}: lengths differ")
        worst = Comparison(True, 0.0)
        for index, (a, b) in enumerate(zip(fresh, stored, strict=True)):
            found = _compare_json(a, b, rtol, f"{where}[{index}]")
            if not found.same:
                return found
            worst = Comparison(
                True, max(worst.max_relative_difference, found.max_relative_difference)
            )
        return worst
    is_float = isinstance(fresh, float) or isinstance(stored, float)
    numeric = isinstance(fresh, int | float) and isinstance(stored, int | float)
    if is_float and numeric and not isinstance(fresh, bool) and not isinstance(stored, bool):
        same, relative = _close(np.array([fresh]), np.array([stored]), rtol)
        return Comparison(same, relative, "" if same else f"{where}: {fresh!r} against {stored!r}")
    same = type(fresh) is type(stored) and fresh == stored
    return Comparison(
        same, 0.0 if same else math.inf, "" if same else f"{where}: {fresh!r} != {stored!r}"
    )


def _compare_csv(fresh: bytes, stored: bytes, rtol: float) -> Comparison:
    """Compare two sample parts: floating-point columns at ``rtol``, all others exactly."""

    def load(data: bytes) -> pd.DataFrame:
        return pd.read_csv(
            io.BytesIO(data),
            float_precision="round_trip",
            dtype=dict.fromkeys(TEXT_COLUMNS, str),
            keep_default_na=False,
        )

    a, b = load(fresh), load(stored)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return Comparison(False, math.inf, "columns or row count differ")
    worst = 0.0
    for column in b.columns:
        if column in TEXT_COLUMNS:
            if not (a[column] == b[column]).all():
                return Comparison(False, math.inf, f"{column}: text differs")
            continue
        x, y = pd.to_numeric(a[column], errors="coerce"), pd.to_numeric(b[column], errors="coerce")
        if pd.api.types.is_integer_dtype(y) and pd.api.types.is_integer_dtype(x):
            if not (x == y).all():
                return Comparison(False, math.inf, f"{column}: integers differ")
            continue
        same, relative = _close(x.to_numpy(float), y.to_numpy(float), rtol)
        worst = max(worst, relative)
        if not same:
            return Comparison(False, relative, f"{column}: relative difference {relative:.3g}")
    return Comparison(True, worst)


def compare_part(part: Part, fresh: bytes, stored: bytes, rtol: float) -> Comparison:
    """Compare a recomputed part with the stored file.

    Floating-point columns of a sample part, and floating-point values of a JSON part, agree when
    ``numpy.isclose(fresh, stored, rtol=rtol, atol=0)``.  Integer, boolean and text values agree
    only when equal, so a changed seed, replicate number or coverage flag never passes as noise.
    """
    if part.family in SAMPLE_FAMILIES:
        return _compare_csv(fresh, stored, rtol)
    return _compare_json(json.loads(fresh), json.loads(stored), rtol)


def stamped_key(root: Path) -> str | None:
    """The key a root is stamped with, or ``None`` for a root without ``KEY.json``."""
    stamp = root / "KEY.json"
    if not stamp.exists():
        return None
    return str(json.loads(stamp.read_text(encoding="utf-8"))["key"])


def check(root: Path, index: int, jobs: int, rtol: float = RTOL) -> bool:
    """Recompute part ``index`` in a temporary root and compare it with the stored file.

    The comparison is :func:`compare_part` at ``rtol``.  The result is recorded under
    ``<root>/equivalence/`` (:func:`write_record`), whether it passes or not.
    """
    part = parts()[index]
    stored = (root / "parts" / part.filename).read_bytes()
    with tempfile.TemporaryDirectory() as scratch:
        fresh = compute(part, Path(scratch), jobs)
    found = compare_part(part, fresh, stored, rtol)
    byte_identical = fresh == stored
    write_record(root, index, part, found, rtol, stored)
    if byte_identical:
        outcome = "identical"
    elif found.same:
        outcome = f"equivalent at rtol {rtol:g} (max relative difference {found.max_relative_difference:.3g})"
    else:
        outcome = f"DIFFERENT ({found.detail})"
    print(f"[{index}] {part.family} {part.name} [{kind_of(part)}]: {outcome}")
    return found.same


def write_record(
    root: Path, index: int, part: Part, found: Comparison, rtol: float, stored: bytes
) -> Path:
    """Write the equivalence record of one check under ``<root>/equivalence/``."""
    record = {
        "part": index,
        "ident": part.ident,
        "kind": kind_of(part),
        "stored_key": stamped_key(root),
        "current_key": key(),
        "rtol": rtol,
        "result": "pass" if found.same else "fail",
        "max_relative_difference": found.max_relative_difference,
        "detail": found.detail,
        "stored_sha256": hashlib.sha256(stored).hexdigest(),
    }
    path = root / "equivalence" / f"{index:05d}-{record['current_key']}.json"
    _write(path, (json.dumps(record, indent=1, sort_keys=True) + "\n").encode())
    return path


def pick_kinds(start: int = 0, stop: int | None = None, root: Path | None = None) -> dict[str, int]:
    """The cheapest part of each kind in ``[start, stop)``; with ``root``, only stored parts.

    Ties go to the lowest index.  A kind with no part in the range is absent.
    """
    every = parts()
    stop = len(every) if stop is None else stop
    best: dict[str, tuple[float, int]] = {}
    for index in range(start, min(stop, len(every))):
        part = every[index]
        if root is not None and not (root / "parts" / part.filename).exists():
            continue
        kind = kind_of(part)
        if kind not in best or (part.hours, index) < best[kind]:
            best[kind] = (part.hours, index)
    return {kind: best[kind][1] for kind in KINDS if kind in best}


# ----------------------------------------------------------------------------- merging
#: One root, or the roots a merge reads, the first one holding a part winning.
Roots = Path | Sequence[Path]


def _roots(root: Roots) -> tuple[Path, ...]:
    return (root,) if isinstance(root, Path) else tuple(root)


def part_path(root: Roots, part: Part) -> Path | None:
    """The file of ``part`` in the first of ``root``'s roots that holds it, else ``None``."""
    for each in _roots(root):
        path = each / "parts" / part.filename
        if path.exists():
            return path
    return None


def _load_json(root: Roots, part: Part) -> dict[str, Any]:
    path = part_path(root, part)
    assert path is not None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_sample(
    root: Roots, family: str, only: Callable[[Part], bool] | None = None
) -> pd.DataFrame:
    """The parts of ``family`` (those with ``only`` true, when given) read as one frame."""
    frames = []
    for part in parts():
        if part.family != family or (only is not None and not only(part)):
            continue
        path = part_path(root, part)
        assert path is not None, part.ident
        frame = pd.read_csv(
            path,
            float_precision="round_trip",
            dtype={"estimand": str, "failure": str, "interval": str},
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
    by_study: dict[str, list[tuple[float, float, float, float]]] = {}
    for index, (stream, rows) in enumerate(frame.groupby("stream", sort=False)):
        kept = rows[rows["failure"] == ""]
        ratio, variance = _bootstrap_ratio(
            kept["estimate"].to_numpy(float), kept["std_error"].to_numpy(float), index
        )
        study = str(stream).split("|")[1].split(":")[0]
        coverage = float(np.mean(kept["covered"]))
        error = math.sqrt(coverage * (1 - coverage) / len(kept))
        by_study.setdefault(study, []).append((ratio, variance, coverage, error))
        lines.append(
            f"{stream.split('|')[1]}: SE ratio {ratio:.4f} (SD {math.sqrt(variance):.4f}) "
            f"coverage {coverage:.4f} (binomial error {error:.4f})"
        )
    for study, arms in by_study.items():
        ratios, variances, coverages, errors = zip(*arms, strict=True)
        chosen = None
        for replicates in range(400, 16_001, 400):
            if D.calibration_pass(1.0, replicates) ** len(arms) < 0.95:
                continue
            if (
                D.f2_family_pass(
                    ratios,
                    variances,
                    replicates,
                    coverages=coverages,
                    coverage_errors=errors,
                )
                >= 0.95
            ):
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
    """The truth-row floor over the gated rows only (:data:`tests.long_horizon_design.GATED_ROWS`).

    A primary row is gated only in its gating scenario.  A row that reports in another scenario,
    possibly below its tail regime, would under-cover by design and must not set R.
    """
    lines = []
    rows_by_study: dict[str, list[tuple[float, float]]] = {}
    gated = set(D.gated_rows())
    kept = frame[frame["failure"] == ""]
    skipped = 0
    for (stream, estimand), rows in kept.groupby(["stream", "estimand"], sort=False):
        study, scenario = str(stream).split("|")[1].split(":", 1)
        if (study, scenario, str(estimand)) not in gated:
            skipped += 1
            continue
        errors = rows["estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
        coverage = float(np.mean(rows["covered"]))
        bias = float(np.mean(errors) / np.std(errors, ddof=1))
        rows_by_study.setdefault(study, []).append((coverage, bias))
    for study, values in rows_by_study.items():
        chosen = _ladder_pick(
            lambda R, values=values: float(
                np.prod([D.truth_row_pass(min(c, 0.9999), abs(b), R) for c, b in values])
            ),
            TRUTH_LADDER,
        )
        lines.append(
            f"{study}: {len(values)} gated truth rows, R {chosen if chosen else 'none -> stop'}"
        )
    lines.append(f"{skipped} non-gating (stream, estimand) pairs report and set no R")
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


def read_band(frame: pd.DataFrame) -> list[str]:
    """The band's joint coverage and its pass probability under the shipped two-sided rule.

    The rule is ``simultaneous_coverage_verdicts``: the 99 percent exact interval of the joint
    coverage must lie inside ``calibration_coverage`` = (0.92, 0.98).  R is the first rung of
    :data:`tests.long_horizon_design.BAND_LADDER` at which the pass probability, at the measured
    joint coverage, reaches :data:`~tests.long_horizon_design.BAND_TARGET`.  No rung reaching it
    prints ``none -> stop``: the band is diagnosed before the declaration (plan section 5.3).
    """
    from tests.studies.evidence.inference import clopper_pearson

    lines = []
    for stream, rows in frame.groupby("stream", sort=False):
        kept = rows[rows["failure"] == ""]
        study = str(stream).split("|")[1].split(":")[0]
        count, trials = int(kept["covered"].sum()), len(kept)
        coverage = count / trials
        interval = clopper_pearson(count, trials, confidence_level=0.99)
        pointwise = float(np.mean(kept["pointwise_covered"]))
        error = math.sqrt(coverage * (1 - coverage) / trials)
        passes = {R: D.band_pass(coverage, R) for R in D.BAND_LADDER}
        chosen = D.band_replicates(coverage)
        lines.append(
            f"{study} band: joint coverage {coverage:.4f} (99% exact [{interval.low:.4f}, "
            f"{interval.high:.4f}]); pointwise control {pointwise:.4f}"
        )
        lines.append(
            "  pass probability by R: "
            + ", ".join(f"{R:,}: {value:.3f}" for R, value in passes.items())
            + f"; at coverage -/+ one error: R "
            f"{D.band_replicates(coverage - error) or 'none'} / "
            f"{D.band_replicates(coverage + error) or 'none'}"
        )
        lines.append(
            f"{study} band: R {chosen if chosen else 'none -> stop (diagnose per plan section 5.3)'}"
        )
    return lines


def read_rung(frame: pd.DataFrame) -> list[str]:
    """The root-n rungs: each rung's coverage clause and the two slope rows' pass probability.

    The 4,000 rung is the control rung: it carries no claim at R = 1,000, and its row prints the
    shipped rule's pass probability for the record.  The 12,000 and 48,000 rungs are positive:
    their coverage clause reads the exact lower limit at the floor.  The slope rows are fitted
    over the three rungs by ``property_verdicts``; their pass probability resamples the pilots.
    """
    lines = []
    by_study: dict[str, dict[int, pd.DataFrame]] = {}
    for stream, rows in frame.groupby("stream", sort=False):
        kept = rows[rows["failure"] == ""]
        study, rest = str(stream).split("|")[1].split(":", 1)
        n = int(rest.rsplit("_", 1)[1])
        by_study.setdefault(study, {})[n] = kept
        ratio = float(np.mean(kept["std_error"]) / np.std(kept["estimate"], ddof=1))
        coverage = float(np.mean(kept["covered"]))
        errors = kept["estimate"].to_numpy(float) - kept["truth"].to_numpy(float)
        bias = float(np.mean(errors) / np.std(errors, ddof=1))
        if n == min(D.ROOT_N_SIZES):
            reading = (
                f"control rung, no claim at R = {D.ROOT_N_REPLICATES:,}; shipped-rule pass "
                f"probability {D.control_rung_pass(coverage, D.ROOT_N_REPLICATES):.3f}"
            )
        else:
            reading = (
                f"coverage clause pass probability "
                f"{D.positive_rung_pass(coverage, D.ROOT_N_REPLICATES):.3f}"
            )
        lines.append(
            f"{study} {rest}: SE ratio {ratio:.4f}, coverage {coverage:.4f}, standardized bias "
            f"{bias:+.4f}; {reading}"
        )
    for study, rungs in by_study.items():
        for statistic, name in (("spread", "empirical_sd"), ("reported", "reported_se")):
            value = D.slope_pass(rungs, statistic, D.ROOT_N_REPLICATES, seed=SLOPE_SEED)
            verdict = "" if value >= 0.99 else " -> diagnose before declaration"
            lines.append(f"{study} root_n_rate {name}: pass probability {value:.3f}{verdict}")
    return lines


def _failure_line(label: str, kept: pd.DataFrame, rows: pd.DataFrame) -> str:
    """Replicate failures, dropped resamples and the sparse flag of one bootstrap stream."""
    per = kept.drop_duplicates("replicate")
    failed = int((rows["failure"] != "").groupby(rows["replicate"]).any().sum())
    text = (
        f"{label}: {len(per)} fitted replicates, {failed} failed outright; dropped refits "
        f"per replicate mean {float(np.mean(per['n_failed'])):.2f}, "
        f"{float(np.mean(per['n_failed'] > 0)):.3f} of replicates lose at least one"
    )
    if "sparse_warning" in kept:
        flag = kept.groupby("estimand")["sparse_warning"].mean()
        text += f"; SparseInfluenceWarning fires on {float(flag.max()):.3f} of replicates at most"
    return text


def read_truth_boot(frame: pd.DataFrame) -> list[str]:
    """The bootstrapped truth rows: the gated Wald and BCa rows set R, and failures are counted.

    A row is gated as in :func:`read_truth` (``D.gated_rows`` names the Wald rows and the ``__bca``
    rows).  The reading adds, for each scenario, the replicate failures and the dropped refits of
    the bootstrap, the separation record, and the BCa coverage of every gated BCa row.  A ladder
    with no rung reaching 0.95 prints ``none -> stop``.
    """
    lines = []
    gated = set(D.gated_rows())
    kept = frame[frame["failure"] == ""]
    rows_by_study: dict[str, list[tuple[float, float]]] = {}
    for stream, rows in frame.groupby("stream", sort=False):
        label = str(stream).split("|")[1]
        study, scenario = label.split(":", 1)
        fitted = kept[kept["stream"] == stream]
        lines.append(_failure_line(label, fitted, rows))
        for estimand, each in fitted.groupby("estimand", sort=False):
            if (study, scenario, str(estimand)) not in gated:
                continue
            errors = each["estimate"].to_numpy(float) - each["truth"].to_numpy(float)
            coverage = float(np.mean(each["covered"]))
            bias = float(np.mean(errors) / np.std(errors, ddof=1))
            rows_by_study.setdefault(study, []).append((coverage, bias))
            lines.append(
                f"  {estimand}: coverage {coverage:.4f}, standardized bias {bias:+.4f}"
                f", mean SE {float(np.mean(each['std_error'])):.5f}"
            )
    for study, values in rows_by_study.items():
        chosen = _ladder_pick(
            lambda R, values=values: float(
                np.prod([D.truth_row_pass(min(c, 0.9999), abs(b), R) for c, b in values])
            ),
            TRUTH_LADDER,
        )
        lines.append(
            f"{study}: {len(values)} gated bootstrapped truth rows, R "
            f"{chosen if chosen else 'none -> stop'}"
        )
    return lines


def read_band_boot(frame: pd.DataFrame) -> list[str]:
    """The sup-t band's reading (:func:`read_band`) with the IC band and the refit failures.

    A stop here means diagnose the band, not switch its construction (plan rev 4.3).
    """
    lines = read_band(frame)
    kept = frame[frame["failure"] == ""]
    for stream, rows in kept.groupby("stream", sort=False):
        study = str(stream).split("|")[1].split(":")[0]
        lines.append(
            f"{study} band: reporting IC band coverage {float(np.mean(rows['ic_covered'])):.4f}; "
            f"dropped refits per replicate mean {float(np.mean(rows['n_failed'])):.2f}, "
            f"{float(np.mean(rows['n_failed'] > 0)):.3f} of replicates lose at least one"
        )
    return lines


def read_f2_c1(frame: pd.DataFrame) -> list[str]:
    """The moved calibration cells alone: the F2 reading plus the sparse flag of each cell.

    :func:`merge` also prints the family reading over the moved cells and the unmoved ``f2``
    cells together (:func:`read_f2_combined`).
    """
    lines = read_f2(frame)
    kept = frame[frame["failure"] == ""]
    for stream, rows in kept.groupby("stream", sort=False):
        flagged = rows["sparse_warning"]
        known = flagged[flagged >= 0]
        if len(known):
            lines.append(
                f"{str(stream).split('|')[1]}: SparseInfluenceWarning fires on "
                f"{float(known.mean()):.3f} of replicates"
            )
    return lines


def read_f2_combined(f2: pd.DataFrame, c1: pd.DataFrame) -> list[str]:
    """The F2 reading of every calibration cell: the moved ones from ``c1``, the rest from ``f2``."""
    unmoved = set(D.unmoved_f2_idents())
    return read_f2(pd.concat([f2[f2["stream"].isin(unmoved)], c1], ignore_index=True))


READERS: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "mc": read_mc,
    "f2": read_f2,
    "bias": read_bias,
    "truth": read_truth,
    "overfit": read_overfit,
    "band": read_band,
    "rung": read_rung,
    "truth_boot": read_truth_boot,
    "band_boot": read_band_boot,
    "f2_c1": read_f2_c1,
}


def _exists(root: Roots, part: Part) -> bool:
    return part_path(root, part) is not None


def read_witness_text(payload: dict[str, Any]) -> str:
    """One witness part's reading: nodes at five times the floors, nodes at one tenth, invalid."""
    rows = payload["rows"]
    full = sum(bool(r["floors_met"]) for r in rows)
    invalid = [f"K{r['K']} {r['plan']} node {r['node']}" for r in rows if not r["valid"]]
    return (
        f"{payload['witness']}: {len(rows)} (plan, node) witnesses, {full} reach five times every "
        f"floor, {len(rows) - full} fall back to one tenth of the record"
        + (f"; INVALID {invalid} -> stop" if invalid else "")
    )


def read_witness(root: Roots) -> list[str]:
    """The per-node targeting witness on each fallback law."""
    return [
        read_witness_text(_load_json(root, part))
        for part in parts()
        if part.family == "witness" and _exists(root, part)
    ]


def read_scans(root: Roots) -> list[str]:
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
    patterns = next((p for p in parts() if p.family == "patterns"), None)
    if patterns is not None and _exists(root, patterns):
        lines.append(f"S3 patterns: pick {pattern_pick(_load_json(root, patterns)['rows'])}")
    lines += read_witness(root)
    for part in parts():
        if part.family == "lawscan" and _exists(root, part):
            found = _load_json(root, part)
            if len(part.detail) == 5:
                q = next(q for q in D.CALIBRATION_FALLBACKS if str(q) == part.name)
                lines.append(
                    f"{part.name} (n >= {part.detail[4]} n_kappa): rule picks {found['pick']}, "
                    f"declared {list(D.CALIBRATION_FALLBACKS[q].key)}"
                )
                continue
            q = next(q for q in D.FALLBACKS if str(q) == part.name)
            lines.append(
                f"{part.name}: rule picks {found['pick']}, declared {list(D.FALLBACKS[q].key)}"
            )
    return lines


def covered_kinds(root: Path, rtol: float) -> set[str]:
    """The kinds ``root``'s records cover at the current key and ``rtol``.

    A record counts when it passed at a tolerance no looser than ``rtol``, names the current key
    and this root's stamped key, carries the kind of its own part index, and describes the file
    the root still holds.
    """
    folder = root / "equivalence"
    if not folder.is_dir():
        return set()
    current, stamped = key(), stamped_key(root)
    every = parts()
    out: set[str] = set()
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        index = record["part"]
        if (
            record["result"] != "pass"
            or record["rtol"] > rtol
            or record["current_key"] != current
            or record["stored_key"] != stamped
            or not 0 <= index < len(every)
            or record["ident"] != every[index].ident
            or record["kind"] != kind_of(every[index])
        ):
            continue
        held = root / "parts" / every[index].filename
        if (
            held.exists()
            and hashlib.sha256(held.read_bytes()).hexdigest() == record["stored_sha256"]
        ):
            out.add(record["kind"])
    return out


def admit(roots: Sequence[Path], accepted: Sequence[str], rtol: float) -> dict[Path, str]:
    """Refuse a root the merge may not read; return each root's stamped key.

    A root stamped with the current key is read as it is.  A root stamped with another key must
    be named in ``accepted`` and must hold equivalence records, written at the current key, for
    every kind of the parts it contributes.
    """
    current = key()
    every = parts()
    chosen: dict[int, Path] = {}
    for index, part in enumerate(every):
        for root in roots:
            if (root / "parts" / part.filename).exists():
                chosen[index] = root
                break
    missing = [i for i in range(len(every)) if i not in chosen]
    if missing:
        raise SystemExit(f"{len(missing)} parts missing; first {missing[:10]}")
    stamps: dict[Path, str] = {}
    for root in roots:
        stamp = stamped_key(root)
        if stamp is None:
            raise SystemExit(f"{root} holds no KEY.json: it is not a pilot root")
        stamps[root] = stamp
        if stamp == current:
            continue
        needed = {kind_of(every[i]) for i, owner in chosen.items() if owner == root}
        if not needed:
            continue
        if stamp not in accepted:
            raise SystemExit(
                f"{root} holds parts of key {stamp}, not {current}: pass --accept-key {stamp} "
                "after `check` has written equivalence records for it"
            )
        absent = sorted(needed - covered_kinds(root, rtol))
        if absent:
            raise SystemExit(
                f"{root} (key {stamp}) contributes parts of kinds {absent} with no passing "
                f"equivalence record at rtol {rtol:g} for key {current}: run "
                f"`kinds --root {root}` and `check` on the printed indices at this head"
            )
    return stamps


def merge(
    root: Path,
    sources: Sequence[Path] = (),
    accepted: Sequence[str] = (),
    rtol: float = RTOL,
) -> bool:
    """Merge the parts of ``root`` and ``sources`` into ``<root>/merged``; ``False`` on a stop.

    Parts come from ``root`` first, then from each source in order.  A root stamped with another
    key is read only under :func:`admit`.  The merged files and the report are written to
    ``root``, which is stamped when it is new.
    """
    roots = [root, *sources]
    if stamped_key(root) is None:
        claim(root)
    admit(roots, accepted, rtol)
    out = root / "merged"
    out.mkdir(exist_ok=True)
    report = read_scans(roots)
    frames: dict[str, pd.DataFrame] = {}
    for family in SAMPLE_FAMILIES:
        frame = _read_sample(roots, family)
        frames[family] = frame
        if frame.empty:
            report.append(f"== {family}: no parts")
            continue
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
    if not frames.get("f2_c1", pd.DataFrame()).empty and not frames.get("f2", pd.DataFrame()).empty:
        report.append("== f2 with f2_c1: every calibration cell of rev 4.3")
        report += read_f2_combined(frames["f2"], frames["f2_c1"])
    stopped = [line for line in report if "-> stop" in line.lower()]
    text = "\n".join(report) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return not stopped


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
        "command", choices=("list", "split", "run", "check", "kinds", "merge", "key", "scans")
    )
    parser.add_argument("index", nargs="*", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="another root to merge (repeatable)",
    )
    parser.add_argument(
        "--accept-key",
        action="append",
        default=[],
        help="a stamped key merge may read (repeatable)",
    )
    parser.add_argument("--rtol", type=float, default=RTOL)
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
        totals: dict[str, tuple[int, float]] = {}
        for part in every:
            count, hours = totals.get(part.family, (0, 0.0))
            totals[part.family] = (count + 1, hours + part.hours)
        for family, (count, hours) in totals.items():
            print(f"  {family:11s} {count:5d} parts {hours:8.1f} CPU-h")
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
    elif arguments.command == "kinds":
        chosen = pick_kinds(
            arguments.start,
            arguments.to,
            arguments.root if arguments.root is not None else None,
        )
        missing = [kind for kind in KINDS if kind not in chosen]
        for kind, index in chosen.items():
            part = parts()[index]
            print(f"{index:5d} {kind:16s} {part.hours:7.3f} CPU-h  {part.name[:70]}")
        print("check " + " ".join(str(i) for i in chosen.values()))
        if missing:
            print(f"no part in the range for kinds {missing}")
    elif arguments.command == "check":
        if not arguments.index:
            parser.error("check takes one or more part indices")
        results = [check(root, i, arguments.jobs, arguments.rtol) for i in arguments.index]
        if not all(results):
            raise SystemExit(1)
    elif arguments.command == "scans":
        print("\n".join(read_scans([root, *arguments.source])))
    elif not merge(root, arguments.source, arguments.accept_key, arguments.rtol):
        raise SystemExit("a reading stopped (-> stop): diagnose before the declaration")


if __name__ == "__main__":
    main()
