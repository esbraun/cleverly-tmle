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
import json
from dataclasses import dataclass
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
    "gates",
    "python_trace",
    "read_export",
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
    """Read one R run.  **Fails closed** -- a short blob or a missing index is an error."""
    directory = Path(directory)
    steps = _read_csv(directory / "steps.csv")
    index = _read_csv(directory / "arrays.csv")
    meta = {str(row["key"]): str(row["value"]) for row in _read_csv(directory / "meta.csv")}
    summary = {
        str(row["estimand"]): {"psi": float(row["psi"]), "se": float(row["se"])}
        for row in _read_csv(directory / "summary.csv")
    }

    blob = np.fromfile(directory / "arrays.f64", dtype="<f8")
    arrays: dict[tuple[int, str, float], np.ndarray] = {}
    for row in index:
        offset, length = int(row["offset"]), int(row["length"])
        if offset + length > blob.size:
            raise ValueError(
                f"{directory / 'arrays.f64'} holds {blob.size} doubles and its index reaches "
                f"{offset + length}. A truncated export is not a shorter comparison, it is "
                "an unreadable one; rerun the R side rather than interpreting this."
            )
        arrays[(int(row["step"]), str(row["field"]), float(row["arm"]))] = blob[
            offset : offset + length
        ]

    raw = np.fromfile(directory / "inputs.f64", dtype="<f8")
    n = int(meta["n"])
    if raw.size != n * len(INPUT_COLUMNS):
        raise ValueError(
            f"inputs.f64 holds {raw.size} doubles; {len(INPUT_COLUMNS)} columns of {n} rows "
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


def python_trace(order: str = "cleverly", fixture: Fixture | None = None) -> Trace:
    """This package's trace under the configuration R is handed.

    The reduced learner is the bare GLM pair rather than the frozen trace's ``"glm"`` Super
    Learner -- see this module's header.  Everything else is the frozen configuration, folds
    included, so the fixture's bytes and its manifest are untouched.
    """
    return trace_module.trace(fixture, order=order, **REDUCED_LEARNERS)


# ------------------------------------------------------------------ the gates


def gates(export: RExport, traces: dict[str, Trace], fixture: Fixture) -> list[Gate]:
    """Every comparison, in the order a difference can first bite.

    ``traces`` is keyed by update order and holds both, because gate 2's question is *which*
    of this package's routes R's round is -- a comparison against one of them alone would
    report an order difference that the other order does not have.
    """
    found = [
        _gate_inputs(export, fixture),
        _gate_reduction(export, traces),
        _gate_route(export, traces),
        _gate_vintage(export, traces),
        _gate_exit(export, traces),
        _gate_close(traces),
        _gate_reported(export, traces),
    ]
    # Everything downstream of the first failure is read and not read *as evidence*.
    first = next((i for i, gate in enumerate(found) if not gate.passed), len(found))
    return [
        gate if i <= first else Gate(**{**gate.__dict__, "confounded": True})
        for i, gate in enumerate(found)
    ]


def _gate_inputs(export: RExport, fixture: Fixture) -> Gate:
    arrays = fixture.arrays()
    worst = 0.0
    culprit = ""
    for name in INPUT_COLUMNS:
        theirs, ours = export.inputs[name], np.asarray(arrays[name], dtype=float)
        difference = float(np.max(np.abs(theirs - ours))) if ours.size else 0.0
        if difference > worst:
            worst, culprit = difference, name
    return Gate(
        name="0 inputs",
        question="did the two sides read the same numbers?",
        reading="bit for bit" if worst == 0.0 else f"worst |Δ| on {culprit}",
        absolute=worst,
        tolerance=INPUT_TOLERANCE,
        passed=worst <= INPUT_TOLERANCE,
        classification="input",
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
    worst, culprit = 0.0, ""
    for field in ("qr", "gr1", "gr2"):
        difference = float(np.max(np.abs(getattr(theirs, field) - getattr(ours, field))))
        if difference > worst:
            worst, culprit = difference, field
    return Gate(
        name="1 first reduced fit",
        question="do Q_r, g_r1 and g_r2 at the initial (Q̄, g) agree?",
        reading=f"worst |Δ| on {culprit}" if worst else "identical",
        absolute=worst,
        tolerance=REDUCTION_TOLERANCE,
        passed=worst <= REDUCTION_TOLERANCE,
        classification="learner",
    )


def _gate_route(export: RExport, traces: dict[str, Trace]) -> Gate:
    """Which equation each round takes, in order.  A **cycle**, compared over one round."""
    theirs = export.route
    ours = {order: _python_route(trace) for order, trace in traces.items()}
    matches = [order for order, route in ours.items() if route == theirs]
    detail = "; ".join(f"{order}={'→'.join(route)}" for order, route in ours.items())
    return Gate(
        name="2 update order",
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
        name="3 reduction vintage",
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


def _gate_exit(export: RExport, traces: dict[str, Trace]) -> Gate:
    """How many rounds each side ran, and on what test it stopped."""
    theirs = max((step["round"] for step in export.steps), default=0)
    cap = int(export.meta["max_iter"])
    # `drtmle`'s `tolIC` defaults to `1/n`, which is a far looser bar than this package's and
    # is most of why the round counts are not comparable numbers. Named in the reading rather
    # than left for a reader to look up, because "R stopped sooner" and "R stopped at a
    # different tolerance" are different facts and only the second is true.
    tol_ic = 1.0 / int(export.meta["n"])
    detail = "; ".join(
        f"{order}={trace.exit['rounds']} ({trace.exit['exit_reason']})"
        for order, trace in traces.items()
    )
    # R's default `maxIter = 3` is a cap, not a tolerance, and a run that reaches it has not
    # been compared on its stopping *rule* at all -- it has been compared on a budget. That is
    # a fact about the comparison rather than about either implementation, so it fails the
    # gate rather than being reported as agreement.
    return Gate(
        name="4 stopping rule",
        question="did both sides stop on their tolerance rather than on a cap?",
        reading=f"R={theirs} rounds (cap {cap}, tolIC={tol_ic:g}) | {detail}",
        absolute=float(theirs >= cap),
        tolerance=0.0,
        passed=theirs < cap,
        classification="stopping-rule",
    )


def _gate_close(traces: dict[str, Trace]) -> Gate:
    """The closing pass, which the R package's loop has no analogue for.

    So this gate is not a comparison -- there is nothing on the other side to compare with.
    What it reports is **how far the closing pass moves the state it was handed**, which is
    the number that says whether R having no analogue could matter.  It passes when the
    movement is nil, which would make the absence of an analogue immaterial.
    """
    worst, culprit = 0.0, ""
    for order, trace in traces.items():
        before, after = trace.boundary()
        for field in STATE_FIELDS:
            difference = float(np.max(np.abs(getattr(after, field) - getattr(before, field))))
            if difference > worst:
                worst, culprit = difference, f"{order}:{field}"
    return Gate(
        name="5 frozen close",
        question="does this package's closing pass move the state R never takes?",
        reading=f"worst |Δ| across the boundary on {culprit}",
        absolute=worst,
        tolerance=0.0,
        passed=worst <= 0.0,
        classification="frozen-close",
    )


def _gate_reported(export: RExport, traces: dict[str, Trace]) -> Gate:
    """``psi`` and ``se``, which is the only thing a user of either package sees."""
    worst, culprit = 0.0, ""
    for order, trace in traces.items():
        for name, theirs in export.summary.items():
            ours = trace.estimates.get(name)
            if ours is None:
                continue
            for field in ("psi", "se"):
                difference = abs(theirs[field] - ours[field])
                if difference > worst:
                    worst, culprit = difference, f"{order}:{name}.{field}"
    return Gate(
        name="6 reported estimate",
        question="do the reported psi and se agree?",
        reading=f"worst |Δ| on {culprit}",
        absolute=worst,
        tolerance=0.0,
        passed=worst <= 0.0,
        classification="corrected-ic",
    )


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
        "| gate | question | reading | |Δ| | bar | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for gate in found:
        lines.append(
            f"| {gate.name} | {gate.question} | {gate.reading} | "
            f"`{gate.absolute:.3g}` | `{gate.tolerance:g}` | **{gate.verdict}** |"
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
        default=Path("benchmarks/results/r-trace"),
        help="the directory benchmarks/r/drtmle_reference.R wrote",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["cleverly", "paper"],
        help="which of this package's update orders to compare R against",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--json", type=Path, default=None, help="write the gates as JSON here")
    arguments = parser.parse_args()

    fixture = trace_module.read_fixture()
    export = read_export(arguments.export)
    traces = {order: python_trace(order, fixture) for order in arguments.orders}
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
