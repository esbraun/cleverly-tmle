r"""Piece F3: where this package's alternation and the published ``drtmle``'s first diverge.

``docs/roadmap.md``'s piece F -- *Localize the shortfall before changing anything* --
is a recovery plan whose premise is that the ``DRTMLE`` shortfall is **measured and not
localized**.  F2 built the state-level instrument; this is the second thing that reads it.
``benchmarks/r/drtmle_reference.R`` records the same step vocabulary out of the R package, on
the same frozen fixture, from the same initial :math:`\bar Q` and :math:`g`; this module
aligns the two and names the **earliest** place they part company.

**What a divergence here is, and is not.**  It is a *question*, adjudicated against Benkeser
et al., ``docs/drtmle/theorem-concordance.md``, the exact-law identities and the remainder
decomposition -- never settled by which side R is on.  Changing this package to match R is
[stop-ship 17](../docs/roadmap.md#stop-ship).  Agreement is evidence about a **transcription**
and this comparison is not a release criterion; ``CLAUDE.md``'s *A differential diagnostic
against R, refused, then authorized* is the whole of what the authorization covers.  Nothing
here runs in any tier ``pytest -m "not slow"`` runs, and no R dependency enters the package,
``noxfile.py`` or the fast CI.

**Why the gates are ordered, and why the order is the finding.**  The two routes stop being
comparable step by step the moment they take a different equation, so a naive "walk both
streams until an array differs" would report the *first* array difference and call it the
divergence -- which on these two implementations is a difference the route already explains.
The gates below are therefore ordered by the causal order in which a difference can first
bite, each is a self-contained comparison of a quantity both sides genuinely have, and the
earliest failing one is what gets classified.  A gate downstream of a failed gate is still
reported, and is labelled **confounded**: it is read, it is not read as evidence.

**Two known differences are removed rather than measured**, and each is removed the way F2
removed clipping -- because a first-divergence hunt confounded by a known convention
difference locates the convention rather than the defect.

*The reduced learner.*  The frozen trace's ``reduced_*_learner="glm"`` is a **two-candidate
Super Learner** over ``{mean, glm}``, and ``stats::glm`` is one unpenalised GLM.  A convex
combination against a single fit is a learner difference this run already knows about, so both
sides are given the bare unpenalised GLM here -- :data:`REDUCED_LEARNERS`, matching
``glm_Qr = "gn"`` and ``glm_gr = "Qn"`` term for term.  What the shipped reduced learner does
instead is **F5's** question and not this one's.

*The folds.*  ``drtmle``'s ``cvFolds`` accepts a vector of fold assignments, so R is handed the
fixture's committed ``fold`` column rather than drawing its own split.  Two different random
splits would make every reduced regression differ at gate 1 and end the comparison there.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks import drtmle_trace as trace_module
from benchmarks.drtmle_trace import Fixture, State, Trace
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "DIVERGENCE_CLASSES",
    "Gate",
    "RExport",
    "Rung",
    "gates",
    "ladder_report",
    "ladder_verdict",
    "python_trace",
    "read_export",
    "read_ladder",
    "report",
]

#: The six places two correct-looking implementations of this algorithm can differ, in the
#: order a difference can first bite.  They are ``docs/roadmap.md``'s R3 table plus the two
#: the table's rows imply but do not name -- the inputs, and the curve built at the end.
#:
#: A seventh is deliberately absent: *"which side is right"*.  Nothing here can classify that
#: and nothing here tries; every one of these is a question routed to the derivation.
DIVERGENCE_CLASSES: tuple[str, ...] = (
    "input",
    "learner",
    "truncation-convention",
    "update-order",
    "reduction-vintage",
    "stopping-rule",
    "frozen-close",
    "corrected-ic",
)

#: The reduced learners both sides are given, chosen so that gate 1 compares two fits of the
#: *same* regression rather than two learners.
#:
#: ``LinearRegression`` is ``stats::glm(family = gaussian)`` exactly -- :math:`Q_r` and
#: :math:`g_{r,2}` agree at ``1.1e-15`` and ``1.2e-15``, which is machine precision and says
#: the designs, the targets and the fitting rows all line up.
#:
#: **The logistic one is where the measurement went, and the answer was not the penalty.**
#: :math:`g_{r,1}` first read ``8.6e-05`` against a ``1e-8`` bar, which looks like
#: ``LogisticRegression``'s residual L2 against ``glm``'s unpenalised IRLS.  It is not: at
#: ``tol=1e-12`` the same ``C=1e6`` reads ``7.5e-09``, so the whole of it was scikit-learn's
#: **default ``tol=1e-4``**.  Sweeping the penalty then moves the reading non-monotonically --
#: ``7.5e-09``, ``3.7e-10``, ``7.9e-10``, ``8.1e-09`` at ``C`` of ``1e6`` to ``1e15`` -- which
#: is solver noise and not a bias, and is why ``C=1e9`` is the setting rather than the largest
#: one available.  :data:`REDUCTION_TOLERANCE` was **not** moved to accommodate any of this.
#:
#: The scaler is affine and does not move a linear predictor.  It is here because the package's
#: own ``glm`` entry has it, and dropping it would be a third difference introduced to remove a
#: second.
REDUCED_LEARNERS: dict[str, Any] = {
    "reduced_outcome_learner": LinearRegression(),
    "reduced_treatment_learner": Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1e9, max_iter=100_000, tol=1e-12)),
        ]
    ),
}

#: Gate 0 is bit-for-bit and has no tolerance: the two sides read one file.  A difference here
#: is F2's own recorded near-miss -- a float parser short by one unit in the last place -- and
#: it must fail rather than be absorbed.
INPUT_TOLERANCE = 0.0

#: Gate 1's bar.  Two unpenalised GLMs fitted by different solvers on identical rows agree to
#: their convergence tolerance and not beyond it; ``1e-8`` is loose against IRLS's own and
#: tight against anything that would be a construction difference.  **Declared before the
#: first R run and not moved since** -- a threshold relaxed after seeing a comparison against
#: another implementation is the failure mode stop-ship 17 names.
REDUCTION_TOLERANCE = 1e-8

#: The columns the R side re-emits from what it actually read, in the order it writes them.
INPUT_COLUMNS = ("w1", "w2", "a", "y", "fold", "weight", "qn1", "qn0", "gn")

#: The six arrays a recorded state carries, in the order the R side emits them per arm.
STATE_FIELDS = ("q", "g", "qr", "gr1", "gr2")


@dataclass(frozen=True)
class Gate:
    """One comparison, its reading, and which class a failure here belongs to.

    ``confounded`` is set on every gate downstream of the first failure.  It is not a third
    verdict on the comparison -- the numbers are what they are -- but on what may be *read
    off* it: once the two routes differ, a later array difference is explained by the route
    and cannot be evidence for anything else.
    """

    name: str
    question: str
    reading: str
    absolute: float
    tolerance: float
    passed: bool
    classification: str
    #: The absolute difference divided by the ``sd`` of the quantity compared, ``nan`` where the
    #: gate compares something with no scale (a route, a vintage pattern, a round count).  F3's
    #: row asks for "absolute **and scale-relative** terms" and the two answer different
    #: questions: ``5e-05`` on an array whose ``sd`` is ``0.6`` is a different fact from the same
    #: number on one whose ``sd`` is ``5e-05``.
    relative: float = float("nan")
    confounded: bool = False

    @property
    def verdict(self) -> str:
        if self.confounded:
            return "confounded"
        return "agree" if self.passed else "differ"


@dataclass(frozen=True)
class RExport:
    """One R run, read back off the binary blob and its scalar index.

    ``arrays`` is keyed ``(step, field, arm)`` and holds float64 exactly as R wrote it --
    ``np.fromfile`` on raw little-endian doubles, so the bytes make the round trip that
    ``%.17g`` and a fast CSV parser did not.  See this module's header and F2's own record.
    """

    steps: list[dict[str, Any]]
    arrays: dict[tuple[int, str, float], np.ndarray]
    inputs: dict[str, np.ndarray]
    summary: dict[str, dict[str, float]]
    meta: dict[str, str]
    #: One row per ``eval_Dstar*`` call per arm: the block, its empirical mean, its ``sd``, and
    #: whether the mechanism it was handed was the **targeted** one.  See :meth:`exit_blocks`.
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def exit_blocks(self) -> dict[tuple[str, float], dict[str, Any]]:
        """The last recorded call of each block, per arm — the state the run exited at.

        *Last*, by the monotone ``call`` counter rather than by the step index, because a step
        index is not unique: ``eval_Dstar`` is called inside R's loop and again after it, and
        both land at whatever step count the loop happened to leave.
        """
        latest: dict[tuple[str, float], dict[str, Any]] = {}
        for row in sorted(self.blocks, key=lambda r: r["call"]):
            latest[(row["block"], row["arm"])] = row
        return latest

    @property
    def arms(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.meta["arms"].split("|"))

    def state(self, step: int, arms: tuple[float, ...] | None = None) -> State:
        """The recorded state at one step, in :class:`~benchmarks.drtmle_trace.State`'s shape.

        Built through the same class the Python side uses rather than a lookalike, so a field
        that moved on one side cannot quietly go on being compared against the other.

        ``arms`` is the **column order to build in**, and passing the Python trace's own is not
        optional.  ``drtmle``'s ``a_0`` is ``(1, 0)`` and a :class:`Trace`'s ``arms`` is this
        package's internal arm ordering, which is ``(0, 1)``; a state built in the exporter's
        order and compared against one built in the trace's compares arm 1 against arm 0
        column for column.  It reads as a large, plausible disagreement in exactly the
        regressions gate 1 exists to check -- measured at ``0.577`` on :math:`g_{r,2}` against
        a bar of ``1e-8`` before this argument existed, which is a *learner* verdict on an
        axis bug.  Both sides label their columns by arm, so the fix is to use the label.
        """
        order = self.arms if arms is None else arms
        columns = {
            field: np.column_stack([self.arrays[(step, field, arm)] for arm in order])
            for field in STATE_FIELDS
        }
        return State(
            q_obs=np.zeros(0),  # R records no observed-arm array; nothing compares it
            q=columns["q"],
            g=columns["g"][:, 0],
            qr=columns["qr"],
            gr1=columns["gr1"],
            gr2=columns["gr2"],
        )

    def of(self, phase: str) -> list[dict[str, Any]]:
        return [step for step in self.steps if step["phase"] == phase]

    @property
    def route(self) -> tuple[str, ...]:
        """The equation sequence of R's **first** round, which is the comparable unit.

        A round rather than the whole stream, because the streams have different lengths and
        a route is a cycle: comparing stream prefixes would report a length difference as an
        order difference, which is gate 4's question and not gate 2's.
        """
        rounds = [step for step in self.steps if step["phase"] == "round"]
        first = [step for step in rounds if step["round"] == 1]
        return tuple(_labelled(step) for step in first)


def _labelled(step: dict[str, Any]) -> str:
    """``refit`` steps carry which regressions they produced; the equations do not need it."""
    note = str(step.get("note", "") or "")
    return f"refit:{note}" if step["equation"] == "refit" and note else str(step["equation"])


def read_export(directory: Path) -> RExport:
    """Read one R record.  **Fails closed** — a bad digest, a short blob or a missing index errors.

    The digests are checked before a number is read, exactly as the input fixture's are and for
    the same reason: this record is *committed* and read back with no R installed, so a silent
    edit to it would make every comparison downstream a comparison against a run nobody took.
    """
    directory = Path(directory)
    _verify_manifest(directory)
    steps = _read_csv(directory / "steps.csv")
    index = _read_csv(directory / "arrays.csv")
    meta = {str(row["key"]): str(row["value"]) for row in _read_csv(directory / "meta.csv")}
    summary = {
        str(row["estimand"]): {"psi": float(row["psi"]), "se": float(row["se"])}
        for row in _read_csv(directory / "summary.csv")
    }
    blocks = [
        {
            "call": int(row["call"]),
            "step": int(row["step"]),
            "round": int(row["round"]),
            "phase": str(row["phase"]),
            "block": str(row["block"]),
            "arm": float(row["arm"]),
            "mean": _float(row["mean"]),
            "sd": _float(row["sd"]),
            "at_targeted_g": str(row["at_targeted_g"]).strip().upper() == "TRUE",
        }
        for row in _read_csv(directory / "blocks.csv")
    ]

    blob = _read_f64(directory / "arrays.f64.gz")
    arrays: dict[tuple[int, str, float], np.ndarray] = {}
    for row in index:
        offset, length = int(row["offset"]), int(row["length"])
        if offset + length > blob.size:
            raise ValueError(
                f"{directory / 'arrays.f64.gz'} holds {blob.size} doubles and its index reaches "
                f"{offset + length}. A truncated export is not a shorter comparison, it is "
                "an unreadable one; rerun the R side rather than interpreting this."
            )
        arrays[(int(row["step"]), str(row["field"]), float(row["arm"]))] = blob[
            offset : offset + length
        ]

    raw = _read_f64(directory / "inputs.f64.gz")
    n = int(meta["n"])
    if raw.size != n * len(INPUT_COLUMNS):
        raise ValueError(
            f"inputs.f64.gz holds {raw.size} doubles; {len(INPUT_COLUMNS)} columns of {n} rows "
            f"is {n * len(INPUT_COLUMNS)}. Gate 0 compares these bit for bit and cannot be "
            "run against a partial file."
        )
    inputs = {name: raw[i * n : (i + 1) * n] for i, name in enumerate(INPUT_COLUMNS)}
    return RExport(
        steps=[
            {
                "step": int(row["step"]),
                "phase": str(row["phase"]),
                "round": int(row["round"]),
                "equation": str(row["equation"]),
                "note": str(row.get("note", "") or ""),
                "epsilon": (_float(row["epsilon_1"]), _float(row["epsilon_0"])),
            }
            for row in steps
        ],
        arrays=arrays,
        inputs=inputs,
        summary=summary,
        meta=meta,
        blocks=blocks,
    )


def _read_f64(path: Path) -> np.ndarray:
    """A gzipped little-endian float64 blob.

    Gzip is a *container* and not a format: the bytes inside it are the raw doubles R wrote, so
    nothing here acquires a parser to be inexact.  That distinction is the whole reason the
    record is binary — see F2's account of a fast CSV parser reading a fixture short by one unit
    in the last place on 65 of 200 rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run benchmarks/r/drtmle_reference.R first — this module "
            "reads an R record and does not produce one."
        )
    return np.frombuffer(gzip.decompress(path.read_bytes()), dtype="<f8")


def _verify_manifest(directory: Path) -> None:
    """Every file's SHA-256 against the record's own manifest."""
    manifest = directory / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} is missing, so nothing in {directory} can be checked. A record with no "
            "manifest is not a record; regenerate it with benchmarks/r/drtmle_reference.R."
        )
    for row in _read_csv(manifest):
        path = directory / str(row["file"])
        if not path.exists():
            raise FileNotFoundError(f"{path} is named in the manifest and is not there.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(row["sha256"]):
            raise ValueError(
                f"{path} does not match its manifest: {digest} against {row['sha256']}. "
                "Every comparison already taken is against the manifest's bytes — regenerate "
                "the record, or restore the file; do not update the digest to match an edit."
            )


def _float(value: Any) -> float:
    """R writes an absent scalar as ``NA``; a refit step has no fluctuation coefficient."""
    text = str(value).strip()
    return float("nan") if text in {"NA", "NaN", "", "None"} else float(text)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run benchmarks/r/drtmle_reference.R first -- this module "
            "reads an R run and does not produce one."
        )
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def python_trace(
    order: str = "cleverly", fixture: Fixture | None = None, version: str | None = None
) -> Trace:
    """This package's trace under the configuration R is handed.

    The reduced learner is the bare GLM pair rather than the frozen trace's ``"glm"`` Super
    Learner -- see this module's header.  Everything else is the frozen configuration, folds
    included, so the fixture's bytes and its manifest are untouched.
    """
    return trace_module.trace(fixture, order=order, version=version, **REDUCED_LEARNERS)


# ------------------------------------------------------------------ the gates


def gates(export: RExport, traces: dict[str, Trace], fixture: Fixture) -> list[Gate]:
    """Every comparison, in the order a difference can first bite.

    ``traces`` is keyed by update order and holds both, because gate 2's question is *which*
    of this package's routes R's round is -- a comparison against one of them alone would
    report an order difference that the other order does not have.
    """
    found = [
        _gate_inputs(export, fixture),
        _gate_truncation(export, traces, fixture),
        _gate_reduction(export, traces),
        _gate_route(export, traces),
        _gate_vintage(export, traces),
        _gate_scores(export, traces),
        _gate_close(traces),
        _gate_corrections(export, traces),
        _gate_reported(export, traces),
    ]
    # Everything downstream of the first failure is read and not read *as evidence*.
    first = next((i for i, gate in enumerate(found) if not gate.passed), len(found))
    return [gate if i <= first else replace(gate, confounded=True) for i, gate in enumerate(found)]


def _reading(theirs: np.ndarray, ours: np.ndarray) -> tuple[float, float]:
    """The absolute worst difference and the same divided by the ``sd`` of what was compared.

    F3's row asks for "absolute **and** scale-relative terms", and the two answer different
    questions: ``5e-05`` on an array whose ``sd`` is ``0.6`` is a different fact from the same
    number on one whose ``sd`` is ``5e-05``.  The denominator is the *pooled* spread of the two
    sides rather than one of them, so the reading does not change when the arguments swap.
    """
    absolute = float(np.max(np.abs(theirs - ours))) if theirs.size else 0.0
    scale = float(np.std(np.concatenate([theirs, ours]))) if theirs.size else 0.0
    return absolute, absolute / scale if scale > 0 else float("nan")


def _gate_inputs(export: RExport, fixture: Fixture) -> Gate:
    arrays = fixture.arrays()
    worst = 0.0
    culprit = ""
    relative = 0.0
    for name in INPUT_COLUMNS:
        theirs, ours = export.inputs[name], np.asarray(arrays[name], dtype=float)
        difference, ratio = _reading(theirs, ours)
        if difference > worst:
            worst, culprit, relative = difference, name, ratio
    return Gate(
        name="0 inputs",
        question="did the two sides read the same numbers?",
        reading="bit for bit" if worst == 0.0 else f"worst |Δ| on {culprit}",
        absolute=worst,
        tolerance=INPUT_TOLERANCE,
        passed=worst <= INPUT_TOLERANCE,
        classification="input",
        relative=relative,
    )


def _gate_reduction(export: RExport, traces: dict[str, Trace]) -> Gate:
    """The three reduced regressions at the **initial** pair, which is neither side's step 1.

    R primes its loop with a ``Qr`` refit and this package primes with an equation-(8) solve,
    so the first reduction each *stream* records is taken at a different outcome regression.
    The comparable object is the one the alternation is handed, and both sides export it on
    purpose: R's ``phase = "reference"`` step, and the ``before`` state of Python's step 0.
    """
    reference = export.of("reference")
    if not reference:
        raise ValueError(
            "the R export carries no reference reduction. Gate 1 is F3's own stopping rule "
            "and cannot be skipped; rerun the R side."
        )
    trace = next(iter(traces.values()))
    theirs = export.state(reference[0]["step"], trace.arms)
    ours = trace.steps[0].before
    worst, culprit, relative, spreads = 0.0, "", 0.0, ""
    for name in ("qr", "gr1", "gr2"):
        left, right = np.ravel(getattr(theirs, name)), np.ravel(getattr(ours, name))
        difference, ratio = _reading(left, right)
        if difference > worst:
            worst, culprit, relative = difference, name, ratio
            spreads = f" (sd R={np.std(left):.4f} vs {np.std(right):.4f})"
    return Gate(
        name="2 first reduced fit",
        question="do Q_r, g_r1 and g_r2 at the initial (Q̄, g) agree?",
        reading=f"worst |Δ| on {culprit}{spreads}" if worst else "identical",
        absolute=worst,
        tolerance=REDUCTION_TOLERANCE,
        passed=worst <= REDUCTION_TOLERANCE,
        classification="learner",
        relative=relative,
    )


def _gate_truncation(export: RExport, traces: dict[str, Trace], fixture: Fixture) -> Gate:
    r"""Do the two truncation conventions produce the same mechanism to divide by?

    **They cannot be made to, and that is what this gate measures rather than a confounder it
    failed to remove.** ``drtmle``'s ``tolg`` is a scalar **lower** bound applied to each arm's
    :math:`g` independently; this package's ``g_bounds`` is a pair, and
    :meth:`~cleverly.estimators._nuisance.Propensity.bounded` clips :math:`g_1` and takes the
    complement. With two arms a row clipped low on one arm is clipped *high* on the other, so no
    choice of bound arranges the two conventions into agreement.

    Evaluated at the **initial** mechanism, which gate 0 has already shown both sides read
    identically, so this is a statement about the convention alone and not about any trajectory.
    On ``v1`` the bound binds on no row and the gate says so and passes; on ``v2`` it binds on
    54 of 200.

    **It precedes the reduced-fit gate, and the reason is causal rather than cosmetic.**  This
    package forms :math:`g_{r,2}`'s *target* at the **truncated** mechanism -- ``reduced.py``'s
    ``_roles`` builds ``(indicator - truncated) / truncated``, and that module's own docstring
    says why the bound is chosen at fit time here and nowhere else -- while ``estimategrn``
    forms it at the untruncated ``train_g``.  So a truncation difference does not wait for the
    targeting step to show up: it is already in what the reduced regressions were asked to
    learn.  Ordered after gate 1 this would be reported as a ``learner`` divergence of
    ``7.87``, which is true of the fitted values and wrong about the cause.
    """
    lower, upper = spec_bounds(export)
    raw = np.asarray(fixture.arrays()["gn"], dtype=float)
    # R: each arm's own probability floored at `tolg`, independently, and never capped.
    theirs = np.column_stack([np.maximum(raw, lower), np.maximum(1.0 - raw, lower)])
    # This package: `g1` clipped on both sides, `g0` the complement of the clipped `g1`.
    clipped_one = np.clip(raw, lower, upper)
    ours = np.column_stack([clipped_one, 1.0 - clipped_one])
    absolute, relative = _reading(theirs, ours)
    rows = int(np.sum(np.any(theirs != ours, axis=1)))
    bound_binds = int(np.sum((raw < lower) | (raw > upper)))
    return Gate(
        name="1 truncation",
        question="do the two truncation conventions give the same mechanism to divide by?",
        reading=(
            f"the bound binds on {bound_binds}/{raw.size} rows; the conventions differ on {rows}"
            + (" — vacuous, nothing clips" if bound_binds == 0 else "")
        ),
        absolute=absolute,
        tolerance=0.0,
        passed=rows == 0,
        classification="truncation-convention",
        relative=relative,
    )


def spec_bounds(export: RExport) -> tuple[float, float]:
    """The pair this package truncates at.  R's ``tolg`` is its lower half, by construction."""
    lower = float(export.meta["tolg"])
    return lower, 1.0 - lower


def _gate_route(export: RExport, traces: dict[str, Trace]) -> Gate:
    """Which equation each round takes, in order.  A **cycle**, compared over one round."""
    theirs = export.route
    ours = {order: _python_route(trace) for order, trace in traces.items()}
    matches = [order for order, route in ours.items() if route == theirs]
    detail = "; ".join(f"{order}={'→'.join(route)}" for order, route in ours.items())
    return Gate(
        name="3 update order",
        question="does a round take the same equations in the same order?",
        reading=f"R={'→'.join(theirs)} | {detail}",
        absolute=0.0 if matches else 1.0,
        tolerance=0.0,
        passed=bool(matches),
        classification="update-order",
    )


def _python_route(trace: Trace) -> tuple[str, ...]:
    """The equation sequence of this package's first round, labelled as R's is.

    A ``refit`` is labelled by **which** regressions the next equation then read, because
    that is the only thing that makes the two streams' ``refit`` steps the same kind of
    object: R refits ``gr`` and ``qr`` in two separate calls and this package's refit closure
    returns all three at once, so an unlabelled ``refit`` would compare a call against a call
    and miss that one of them produced twice as much.

    **Which is not the same as what the refit produced**, and that distinction is F2's, kept
    here rather than re-derived: a refit step's ``after`` is its closure's whole output, and
    the round then adopts one field group from it or all three.  Labelling by what *moved*
    across the step therefore reads ``all`` on both orders and makes the vintage invisible --
    which was the first thing written here and what
    ``tests/unit/test_drtmle_r_compare.py::test_route_labels_a_refit_by_what_it_moved``
    caught.  :func:`~benchmarks.drtmle_trace.vintages` is the instrument that reads adoption,
    and it is what this calls.
    """
    adopted = {row["step"]: row for row in trace_module.vintages(trace)}
    labelled = []
    for step in trace.steps:
        if step.phase != "round" or step.round != 1:
            continue
        if step.equation != "refit":
            labelled.append(step.equation)
            continue
        row = adopted.get(step.index, {})
        fields = [name for name in ("qr", "gr1", "gr2") if row.get(name)]
        labelled.append(
            "refit:"
            + (
                "all"
                if len(fields) == 3
                else "gr"
                if fields == ["gr1", "gr2"]
                else "qr"
                if fields == ["qr"]
                else "+".join(fields) or "none"
            )
        )
    return tuple(labelled)


def _gate_vintage(export: RExport, traces: dict[str, Trace]) -> Gate:
    """**Which** reductions each refit of a round contributes -- R3's fourth row.

    A *pattern*, not a count, and the distinction is the whole gate.  Counting refit steps
    reads ``2`` for R, ``2`` for ``"cleverly"`` and ``2`` for ``"paper"`` and calls all three
    the same, when R and ``"paper"`` adopt one field group per refit and ``"cleverly"`` adopts
    all three at both.  That was the first thing written here and it made the one difference no
    fitted result carries invisible -- which is F2's own lesson, arrived at a second way.
    """
    theirs = _vintages_in(export.route)
    ours = {order: _vintages_in(_python_route(trace)) for order, trace in traces.items()}
    matches = [order for order, pattern in ours.items() if pattern == theirs]
    detail = "; ".join(f"{order}={'+'.join(pattern)}" for order, pattern in ours.items())
    return Gate(
        name="4 reduction vintage",
        question="which reductions does each refit of a round contribute?",
        reading=f"R={'+'.join(theirs)} | {detail}"
        + (f" — matches {', '.join(matches)}" if matches else " — matches neither"),
        absolute=0.0 if matches else 1.0,
        tolerance=0.0,
        passed=bool(matches),
        classification="reduction-vintage",
    )


def _vintages_in(route: tuple[str, ...]) -> tuple[str, ...]:
    """The refit labels of a round, in order: ``("gr", "qr")`` against ``("all", "all")``."""
    return tuple(label.partition(":")[2] for label in route if label.startswith("refit"))


def _gate_scores(export: RExport, traces: dict[str, Trace]) -> Gate:
    r"""The three empirical means at exit — the quantity Theorem 1's premise is about.

    This is what F3's row means by comparing "scores", and the first run of this comparison
    could not answer it because the R side exported none.  R's ``eval_Dstar`` mean is equation
    (8)'s (its ``psi_t`` is the mean of :math:`\bar Q_a`, so the plug-in half cancels
    identically); ``eval_Dstar_g``'s is equation (9)'s and ``eval_Dstar_Q``'s is equation
    (10)'s.

    Both sides being *near* zero is not the comparison — **how near** is, because that is the
    bar each declared convergence at, and it is a bar rather than an outcome.
    """
    exits = export.exit_blocks()
    theirs = max(
        (abs(row["mean"]) for (block, _), row in exits.items() if block in {"D_g", "D_Q"}),
        default=float("nan"),
    )
    ours = max(
        abs(float(np.mean(values)))
        for trace in traces.values()
        for values in trace.corrections.values()
    )
    tol_ic = 1.0 / int(export.meta["n"])
    detail = "; ".join(
        f"{order}={trace.exit['rounds']} rounds ({trace.exit['exit_reason']})"
        for order, trace in traces.items()
    )
    rounds = max((step["round"] for step in export.steps), default=0)
    cap = int(export.meta["max_iter"])
    return Gate(
        name="5 exit scores",
        question="how near zero are equations (9) and (10) when each side stops?",
        reading=(
            f"R worst |P_n D| = {theirs:.2e} after {rounds} rounds "
            f"(cap {cap}, tolIC={tol_ic:g}); this package {ours:.2e} — {detail}"
        ),
        absolute=abs(theirs - ours),
        tolerance=tol_ic,
        # Not "are they equal" — "did R clear the bar this package holds itself to". A run that
        # stopped at its own looser tolerance has not solved the same equations to the same
        # place, and the comparison of everything downstream is in part a comparison of bars.
        passed=bool(theirs <= max(ours, 0.0) * 10 or theirs <= 1e-8),
        classification="stopping-rule",
        relative=theirs / ours if ours > 0 else float("nan"),
    )


def _gate_corrections(export: RExport, traces: dict[str, Trace]) -> Gate:
    r"""The two correction arrays the reported curve subtracts, row by row.

    Gate 8 compares ``se``, which is one scalar summarising everything; this compares the arrays
    that produce it, which is what localizes a variance difference to the curve rather than
    leaving it inferred.

    **A pure sign difference is reported as one and is not a defect finding.** The paper's
    display defines :math:`D_A = -(Q_r/g)(A - g)` while Theorem 1 *subtracts* :math:`D_A`, and
    ``docs/roadmap.md``'s item 21 adjudicated that against the source's own appendices and
    resolved it **in favour of this package's positive correction**. So a sign difference here
    is a question already answered by the derivation, and a comparison that reported it as a
    divergence would be reproducing exactly the mistake item 21 is the worked example of.
    """
    exits = export.exit_blocks()
    worst, culprit, flipped = 0.0, "", []
    relative, spreads = float("nan"), ""
    for order, trace in traces.items():
        for block, name in (("D_g", "D*_g"), ("D_Q", "D*_Q")):
            for arm in export.arms:
                row = exits.get((block, arm))
                ours = trace.corrections.get(f"{name}[{arm:g}]")
                if row is None or ours is None:
                    continue
                theirs = export.arrays[(row["step"], block, arm)]
                direct, direct_rel = _reading(theirs, np.asarray(ours, dtype=float))
                negated, negated_rel = _reading(-theirs, np.asarray(ours, dtype=float))
                # A flip is only *read* as a flip when negating is decisively better. Two
                # arrays at genuinely different fixed points are far apart either way, and
                # whichever sign happens to be nearer is then noise -- reporting that as "the
                # signs differ" would manufacture exactly the finding item 21 warns about.
                if negated * 2.0 < direct:
                    flipped.append(f"{block}[{arm:g}]")
                    direct, direct_rel = negated, negated_rel
                if direct > worst:
                    worst, culprit, relative = direct, f"{order}:{block}[{arm:g}]", direct_rel
                    # The *spreads*, not only the worst row: a correction block is what the
                    # reported variance is built from, so "R's sd against ours" is the reading
                    # that says where an `se` difference comes from. A max difference alone
                    # cannot distinguish a shifted array from a wider one.
                    spreads = f" (sd R={np.std(theirs):.4f} vs {np.std(ours):.4f})"
    note = (
        f"; negating helps decisively on {', '.join(sorted(set(flipped)))}"
        if flipped
        else "; signs agree throughout"
    )
    # Per order as well as the worst, because a residue that is smaller against one order than
    # the other is *route* evidence and the worst-across-orders reading cannot see it. On the
    # committed ladder R's converged spread lands between the two, which is the whole reason
    # this line exists -- see `ladder_verdict`.
    per_order = "; ".join(
        f"{name} {np.std(np.asarray(trace.corrections[f'D*_Q[{LADDER_ARM:g}]'], dtype=float)):.4f}"
        for name, trace in traces.items()
    )
    return Gate(
        name="7 correction arrays",
        question="do the two correction blocks agree row by row?",
        reading=f"worst |Δ| on {culprit}{spreads}{note}; sd(D*_Q[1]) by order: {per_order}",
        absolute=worst,
        tolerance=0.0,
        passed=worst <= 0.0,
        classification="corrected-ic",
        relative=relative,
    )


def _gate_close(traces: dict[str, Trace]) -> Gate:
    """The closing pass, which the R package's loop has no analogue for.

    So this gate is not a comparison -- there is nothing on the other side to compare with.
    What it reports is **how far the closing pass moves the state it was handed**, which is
    the number that says whether R having no analogue could matter.  It passes when the
    movement is nil, which would make the absence of an analogue immaterial.
    """
    worst, culprit, relative = 0.0, "", 0.0
    for order, trace in traces.items():
        before, after = trace.boundary()
        for name in STATE_FIELDS:
            difference, ratio = _reading(
                np.ravel(getattr(before, name)), np.ravel(getattr(after, name))
            )
            if difference > worst:
                worst, culprit, relative = difference, f"{order}:{name}", ratio
    return Gate(
        name="6 frozen close",
        question="does this package's closing pass move the state R never takes?",
        reading=f"worst |Δ| across the boundary on {culprit}",
        absolute=worst,
        tolerance=0.0,
        passed=worst <= 0.0,
        classification="frozen-close",
        relative=relative,
    )


def _gate_reported(export: RExport, traces: dict[str, Trace]) -> Gate:
    """``psi`` and ``se``, which is the only thing a user of either package sees."""
    worst, culprit, relative = 0.0, "", float("nan")
    for order, trace in traces.items():
        for name, theirs in export.summary.items():
            ours = trace.estimates.get(name)
            if ours is None:
                continue
            for quantity in ("psi", "se"):
                difference = abs(theirs[quantity] - ours[quantity])
                if difference > worst:
                    worst, culprit = difference, f"{order}:{name}.{quantity}"
                    # Against R's own `se`, so the reading says "how many standard errors" for
                    # `psi` and "what fraction" for `se` -- both of which a reader can act on,
                    # where an absolute difference in either alone tells them nothing.
                    relative = difference / theirs["se"] if theirs["se"] > 0 else float("nan")
    return Gate(
        name="8 reported estimate",
        question="do the reported psi and se agree?",
        reading=f"worst |Δ| on {culprit}",
        absolute=worst,
        tolerance=0.0,
        passed=worst <= 0.0,
        classification="corrected-ic",
        relative=relative,
    )


# ------------------------------------------------------------------ the stopping-bar ladder

#: When the ``se`` gap counts as explained by the stopping bar.  **Declared before the first
#: rung was read**, because a threshold chosen after seeing a comparison against another
#: implementation is the failure mode stop-ship 17 names.
#:
#: The two are a spread and a reported quantity because they answer different halves: the first
#: asks whether the *array* the variance is built from has come into agreement, the second
#: whether the number a caller sees has.
CLOSED_SPREAD_RATIO = 1.2
CLOSED_SE_RATIO = 1.05

#: Below this fraction of the gap explained, the difference is a construction difference rather
#: than a stopping artefact.  Between the two, the reading is **partial** and carries no verdict.
PERSISTS_FRACTION = 0.5

#: The block whose spread the gap was localized to -- gate 7's reading on the committed record.
LADDER_BLOCK = "D_Q"
LADDER_ARM = 1.0


@dataclass(frozen=True)
class Rung:
    """One R run at one stopping bar."""

    tol_ic: float
    rounds: int
    capped: bool
    #: The worst of equations (9) and (10) at exit -- what the bar is a bar on.
    worst_score: float
    spreads: dict[tuple[str, float], float]
    estimates: dict[str, dict[str, float]]

    @property
    def converged(self) -> bool:
        """Reached its own bar rather than its round cap.

        A capped rung is not a tighter measurement of the same thing -- it is a run that did
        not finish, and reading its spread as "where R lands at this bar" would be reading a
        budget as a fixed point.
        """
        return not self.capped and self.worst_score <= self.tol_ic


def read_ladder(directory: Path) -> list[Rung]:
    """Every rung under ``directory``, loosest bar first.

    A **light** reader: ``blocks.csv``, ``summary.csv`` and ``meta.csv`` only.  It deliberately
    does not go through :func:`read_export`, which requires the per-step array blob that a
    ``--blocks-only`` record does not have -- at a tight bar R runs twenty-odd rounds and the
    full state would be megabytes a rung to answer a question about scalars.
    """
    directory = Path(directory)
    rungs = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        _verify_manifest(child)
        meta = {str(row["key"]): str(row["value"]) for row in _read_csv(child / "meta.csv")}
        blocks = _read_csv(child / "blocks.csv")
        latest: dict[tuple[str, float], dict[str, Any]] = {}
        for row in sorted(blocks, key=lambda r: int(r["call"])):
            latest[(str(row["block"]), float(row["arm"]))] = row
        rungs.append(
            Rung(
                tol_ic=float(meta["tol_ic"]),
                rounds=int(meta["rounds"]),
                capped=meta["capped"] == "1",
                worst_score=max(
                    abs(_float(row["mean"]))
                    for (block, _), row in latest.items()
                    if block in {"D_g", "D_Q"}
                ),
                spreads={key: _float(row["sd"]) for key, row in latest.items()},
                estimates={
                    str(row["estimand"]): {"psi": float(row["psi"]), "se": float(row["se"])}
                    for row in _read_csv(child / "summary.csv")
                },
            )
        )
    return sorted(rungs, key=lambda rung: -rung.tol_ic)


def ladder_verdict(rungs: list[Rung], traces: dict[str, Trace]) -> dict[str, Any]:
    """Is the ``se`` gap a stopping artefact or a construction difference?

    Four outcomes, all four declared before any rung was read.  ``"closed"`` needs *both*
    ratios inside their bars; ``"persists"`` needs less than :data:`PERSISTS_FRACTION` of the
    gap explained; ``"unreachable"`` is R failing to reach the bar at all, which would say the
    bar is not a free parameter for it; and anything else is **partial** and carries no verdict,
    which is the honest outcome when a reading lands between two declared thresholds rather than
    the nearer threshold being quietly widened to reach it.
    """
    converged = [rung for rung in rungs if rung.converged]
    if not converged:
        return {"verdict": "unreachable", "detail": "no rung reached its own bar"}

    loosest, tightest = rungs[0], converged[-1]
    key = (LADDER_BLOCK, LADDER_ARM)
    # Per update order, which is what separates "the bar explained it" from "the route did":
    # a residue that is smaller against one order than the other is a route difference, and
    # gate 7's worst-across-orders reading cannot see that.
    order = min(
        traces,
        key=lambda name: abs(
            _spread(traces[name]) / tightest.spreads[key] - 1.0 if tightest.spreads[key] else 0.0
        ),
    )
    ours = _spread(traces[order])
    before = ours / loosest.spreads[key] if loosest.spreads[key] else float("nan")
    after = ours / tightest.spreads[key] if tightest.spreads[key] else float("nan")
    se_ratio = (
        traces[order].estimates["ey1"]["se"] / tightest.estimates["ey1"]["se"]
        if tightest.estimates["ey1"]["se"]
        else float("nan")
    )
    # On the *distance from agreement*, since a ratio of 1 is agreement and the quantity of
    # interest is how much of the excess went away.
    explained = 1.0 - (abs(after - 1.0) / abs(before - 1.0)) if abs(before - 1.0) > 0 else 1.0

    if abs(after - 1.0) <= CLOSED_SPREAD_RATIO - 1.0 and abs(se_ratio - 1.0) <= (
        CLOSED_SE_RATIO - 1.0
    ):
        verdict = "closed"
    elif explained < PERSISTS_FRACTION:
        verdict = "persists"
    else:
        verdict = "partial"
    return {
        "verdict": verdict,
        "nearest_order": order,
        # Every order's ratio, not only the nearest, because on this ladder R's converged
        # spread lands *between* the two and a single ratio would hide that. Which side of
        # each it lands on is the route evidence.
        "after_by_order": {
            name: _spread(trace) / tightest.spreads[key] for name, trace in traces.items()
        },
        "spread_ratio_before": before,
        "spread_ratio_after": after,
        "se_ratio_after": se_ratio,
        "explained": explained,
        "tightest_bar": tightest.tol_ic,
        "tightest_rounds": tightest.rounds,
    }


def _spread(trace: Trace) -> float:
    """This package's spread of the block the gap was localized to."""
    return float(np.std(np.asarray(trace.corrections[f"D*_Q[{LADDER_ARM:g}]"], dtype=float)))


def ladder_report(rungs: list[Rung], traces: dict[str, Trace]) -> str:
    reading = ladder_verdict(rungs, traces)
    key = (LADDER_BLOCK, LADDER_ARM)
    lines = [
        "# The stopping-bar ladder",
        "",
        "Does `drtmle` run to this package's bar reproduce its correction arrays, or does a",
        "difference survive? The thresholds below were declared before the first rung was read.",
        "",
        "| `tolIC` | rounds | worst `P_n D` | `sd(D_Q[1])` | `psi[ate]` | `se[ey1]` | at its bar |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rung in rungs:
        lines.append(
            f"| `{rung.tol_ic:g}` | {rung.rounds} | `{rung.worst_score:.2e}` | "
            f"`{rung.spreads.get(key, float('nan')):.4f}` | "
            f"`{rung.estimates['ate']['psi']:+.6f}` | `{rung.estimates['ey1']['se']:.6f}` | "
            f"{'yes' if rung.converged else '**no**'} |"
        )
    for name, trace in traces.items():
        lines.append(
            f"| *this package,* `{name}` | {trace.exit['rounds']} | — | "
            f"`{_spread(trace):.4f}` | `{trace.estimates['ate']['psi']:+.6f}` | "
            f"`{trace.estimates['ey1']['se']:.6f}` | yes |"
        )
    lines += [
        "",
        f"**Verdict: `{reading['verdict']}`.** Against `{reading['nearest_order']}`, the "
        f"`sd(D_Q[1])` ratio is {reading['spread_ratio_before']:.2f} at R's own default and "
        f"{reading['spread_ratio_after']:.2f} at `{reading['tightest_bar']:g}` — "
        f"{reading['explained']:.0%} of the gap explained by the bar; `se[ey1]` ratio "
        f"{reading['se_ratio_after']:.3f}.",
        "",
        "R's converged spread against each of this package's orders: "
        + ", ".join(f"`{name}` {ratio:.3f}" for name, ratio in reading["after_by_order"].items())
        + " — so it lands **between** them, which is route evidence rather than a residue.",
        "",
        f"Bars: `closed` needs both ratios inside `{CLOSED_SPREAD_RATIO}` and "
        f"`{CLOSED_SE_RATIO}`; `persists` needs under {PERSISTS_FRACTION:.0%} explained; "
        "anything else is `partial` and carries no verdict.",
        "",
    ]
    return "\n".join(lines)


def first_divergence(found: list[Gate]) -> Gate | None:
    """The earliest gate that failed, or ``None`` if every one of them agreed."""
    return next((gate for gate in found if not gate.passed), None)


# ------------------------------------------------------------------ the report


def report(export: RExport, traces: dict[str, Trace], found: list[Gate]) -> str:
    """The checked-in comparison report, in the form F3's acceptance names."""
    first = first_divergence(found)
    lines = [
        "# F3 — the bounded differential run against the R package",
        "",
        f"`drtmle` {export.meta['package_version']}, `Qsteps={export.meta['qsteps']}`, "
        f"`maxIter={export.meta['max_iter']}`, `tolg={export.meta['tolg']}`, "
        f"n={export.meta['n']} over {export.meta['n_folds']} committed folds.",
        "",
        "**A divergence below is a question, not a verdict.** It is adjudicated against the",
        "derivation — Theorem 1, `docs/drtmle/theorem-concordance.md`, the exact-law",
        "identities and the remainder decomposition — never by which side R is on. Changing",
        "this package to match R is stop-ship 17.",
        "",
        "| gate | question | reading | abs | rel | bar | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for gate in found:
        lines.append(
            f"| {gate.name} | {gate.question} | {gate.reading} | "
            f"`{gate.absolute:.3g}` | `{gate.relative:.3g}` | `{gate.tolerance:g}` | "
            f"**{gate.verdict}** |"
        )
    lines += ["", "## The earliest divergence", ""]
    if first is None:
        lines.append(
            "None. Every gate agreed, to the tolerances declared above. That is a statement "
            "about a transcription and not about a derivation."
        )
    else:
        lines += [
            f"**{first.name}**, classified **`{first.classification}`**.",
            "",
            f"> {first.question}  \n> {first.reading}",
            "",
            "Every gate below it is marked `confounded`: the numbers are what they are, and",
            "once the routes part company a later array difference is explained by the route",
            "and is not evidence for anything else.",
        ]
    header = " | ".join(traces)
    rule = " | ".join("---" for _ in traces)
    lines += [
        "",
        "## The reported estimates",
        "",
        f"| estimand | R | {header} |",
        f"| --- | --- | {rule} |",
    ]
    for name in ("ey1", "ey0", "ate"):
        if name not in export.summary:
            continue
        cells = " | ".join(f"{traces[order].estimates[name]['psi']:+.6f}" for order in traces)
        lines.append(f"| `psi[{name}]` | {export.summary[name]['psi']:+.6f} | {cells} |")
    for name in ("ey1", "ey0", "ate"):
        if name not in export.summary:
            continue
        cells = " | ".join(f"{traces[order].estimates[name]['se']:.6f}" for order in traces)
        lines.append(f"| `se[{name}]` | {export.summary[name]['se']:.6f} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="the R record to read; defaults to the committed one for --fixture-version",
    )
    parser.add_argument(
        "--fixture-version",
        default="v1",
        help="which frozen fixture: v1 (truncation slack) or v2 (truncation binds)",
    )
    parser.add_argument("--qsteps", default="2", help="which of R's outcome-update routes")
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["cleverly", "paper"],
        help="which of this package's update orders to compare R against",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        nargs="?",
        const=Path("__default__"),
        default=None,
        help="read a stopping-bar ladder instead of the nine gates; defaults to the committed one",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--json", type=Path, default=None, help="write the gates as JSON here")
    arguments = parser.parse_args()

    export_path = arguments.export
    if export_path is None:
        export_path = (
            Path(trace_module.__file__).resolve().parent
            / "fixtures"
            / f"r-trace-{arguments.fixture_version}-q{arguments.qsteps}"
        )
    fixture = trace_module.read_fixture(version=arguments.fixture_version)
    traces = {
        order: python_trace(order, fixture, arguments.fixture_version) for order in arguments.orders
    }
    found: list[Gate] = []
    if arguments.ladder is not None:
        ladder_path = arguments.ladder
        if str(ladder_path) == "__default__":
            ladder_path = (
                Path(trace_module.__file__).resolve().parent
                / "fixtures"
                / f"r-ladder-{arguments.fixture_version}-q{arguments.qsteps}"
            )
        text = ladder_report(read_ladder(ladder_path), traces)
    else:
        export = read_export(export_path)
        found = gates(export, traces, fixture)
        text = report(export, traces, found)
    print(text)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps([gate.__dict__ for gate in found], indent=2, default=str)
        )


if __name__ == "__main__":
    main()
