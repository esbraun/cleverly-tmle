"""What makes the F3 comparison an instrument rather than a printout.

**No R runs here and none is installed.** ``docs/roadmap.md``'s F3 row forbids an R dependency
"in the package, in ``nox``, or in any test tier the fast CI runs", so what this module tests
is the half of F3 that is Python: the reader, the gates, the classification and the report.
The synthetic export below is written from a *real* Python trace, so gates 0 and 1 pass by
construction and a test that wants one to fail has to break something specific -- which is the
same discipline ``tests/unit/test_drtmle_trace.py`` applies to the identities, and for the
same reason: agreement is only evidence if disagreement was reachable.

One module-scoped pair of traces, both update orders, and every test reads it. Two ``DRTMLE``
fits, ~8 s. Building a third would be the commonest waste ``CLAUDE.md`` names.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_r_compare as compare
from benchmarks import drtmle_trace as trace_module

#: R's round, from its own source: `fluctuateG`, refit `gr`, `fluctuateQ2`, `fluctuateQ1`,
#: refit `Qr`. Written here as the shape a synthetic export takes; what the *real* R package
#: does is what `benchmarks/r/drtmle_reference.R` records, and the two are compared by
#: dispatching `.github/workflows/drtmle-r-differential.yml`, never by this file.
R_ROUND = ("9", "refit:gr", "10", "8", "refit:qr")


@pytest.fixture(scope="module")
def fixture() -> Any:
    return trace_module.read_fixture()


@pytest.fixture(scope="module")
def traces(fixture: Any) -> dict[str, Any]:
    return {order: compare.python_trace(order, fixture) for order in ("cleverly", "paper")}


def _write_export(
    directory: Path,
    fixture: Any,
    reference: Any,
    *,
    source_arms: tuple[float, ...] = (0.0, 1.0),
    route: tuple[str, ...] = R_ROUND,
    rounds: int = 2,
    max_iter: int = 3,
    estimates: dict[str, dict[str, float]] | None = None,
) -> Path:
    """A synthetic R export, in the bytes the R script writes.

    The reference reduction is copied off a real Python trace's ``before`` state, so gate 1
    compares a thing against itself and passes -- which is what leaves each test free to break
    exactly one gate and see only that one move.

    ``source_arms`` is the arm order the *state* being copied is in, and the emitted columns are
    **labelled** by arm rather than written in that order.  R's ``a_0`` is ``(1, 0)`` and a
    :class:`~benchmarks.drtmle_trace.Trace`'s ``arms`` is ``(0, 1)``, so a helper that wrote
    column ``i`` under label ``i`` would emit a swapped state -- which reads as a ``0.577``
    disagreement on :math:`g_{r,2}` and a ``learner`` verdict.  That is the axis bug the
    production reader was carrying, and writing it here as well would have hidden it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arms = (1.0, 0.0)
    arrays = fixture.arrays()
    n = len(fixture.frame)

    rows: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    blob: list[np.ndarray] = []
    offset = 0

    def emit(step: int, phase: str, round_no: int, equation: str, note: str, state: Any) -> None:
        nonlocal offset
        rows.append(
            {
                "step": step,
                "phase": phase,
                "round": round_no,
                "equation": equation,
                "note": note,
                "epsilon_1": 0.0,
                "epsilon_0": 0.0,
            }
        )
        for arm in arms:
            for field in compare.STATE_FIELDS:
                values = getattr(state, field)
                column = values if values.ndim == 1 else values[:, source_arms.index(arm)]
                blob.append(np.asarray(column, dtype=float))
                index.append(
                    {
                        "step": step,
                        "field": field,
                        "arm": arm,
                        "offset": offset,
                        "length": len(column),
                    }
                )
                offset += len(column)

    step = 1
    emit(step, "reference", 0, "reference", "initial pair", reference)
    for round_no in range(1, rounds + 1):
        for label in route:
            step += 1
            equation, _, note = label.partition(":")
            emit(step, "round", round_no, equation, note, reference)

    _write_csv(directory / "steps.csv", rows)
    _write_csv(directory / "arrays.csv", index)
    np.concatenate(blob).astype("<f8").tofile(directory / "arrays.f64")

    estimates = estimates or {name: {"psi": 0.0, "se": 0.0} for name in ("ey1", "ey0", "ate")}
    _write_csv(
        directory / "summary.csv",
        [{"estimand": name, **values} for name, values in estimates.items()],
    )
    _write_csv(
        directory / "meta.csv",
        [
            {"key": "qsteps", "value": 2},
            {"key": "max_iter", "value": max_iter},
            {"key": "tolg", "value": 0.01},
            {"key": "n", "value": n},
            {"key": "n_folds", "value": 3},
            {"key": "arms", "value": "1|0"},
            {"key": "verify_residual", "value": 0.0},
            {"key": "package_version", "value": "1.1.2"},
        ],
    )
    np.concatenate(
        [np.asarray(arrays[name], dtype=float) for name in compare.INPUT_COLUMNS]
    ).astype("<f8").tofile(directory / "inputs.f64")
    return directory


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory, fixture: Any, traces: dict[str, Any]) -> Any:
    directory = _write_export(
        tmp_path_factory.mktemp("r-trace"),
        fixture,
        traces["cleverly"].steps[0].before,
        source_arms=traces["cleverly"].arms,
    )
    return compare.read_export(directory)


# ------------------------------------------------------------------ the reader


def test_read_export_round_trips_the_state(export: Any, traces: dict[str, Any]) -> None:
    """The blob comes back as the arrays that went in, bit for bit.

    Raw float64 rather than text is the whole reason this format was chosen -- F2's own
    record has a fast CSV parser reading the fixture short by one unit in the last place on
    65 of 200 rows, which is the size of difference a first-divergence hunt would find and
    mis-classify. ``==`` rather than ``allclose``, so a format that started rounding fails.
    """
    reference = export.of("reference")[0]
    recovered = export.state(reference["step"], traces["cleverly"].arms)
    expected = traces["cleverly"].steps[0].before
    for field in ("qr", "gr1", "gr2"):
        assert np.array_equal(getattr(recovered, field), getattr(expected, field))


def test_read_export_refuses_a_truncated_blob(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """A short export is unreadable, not a shorter comparison."""
    directory = _write_export(
        tmp_path / "short",
        fixture,
        traces["cleverly"].steps[0].before,
        source_arms=traces["cleverly"].arms,
    )
    blob = np.fromfile(directory / "arrays.f64", dtype="<f8")
    blob[:-10].tofile(directory / "arrays.f64")
    with pytest.raises(ValueError, match="truncated export"):
        compare.read_export(directory)


def test_read_export_refuses_partial_inputs(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """Gate 0 is bit-for-bit and cannot be run against a file with a column missing."""
    directory = _write_export(
        tmp_path / "partial",
        fixture,
        traces["cleverly"].steps[0].before,
        source_arms=traces["cleverly"].arms,
    )
    raw = np.fromfile(directory / "inputs.f64", dtype="<f8")
    raw[: -len(fixture.frame)].tofile(directory / "inputs.f64")
    with pytest.raises(ValueError, match="columns of"):
        compare.read_export(directory)


def test_read_export_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"drtmle_reference\.R"):
        compare.read_export(tmp_path / "nothing-here")


# ------------------------------------------------------------------ the gates


def test_every_gate_is_classified(export: Any, traces: dict[str, Any], fixture: Any) -> None:
    """A gate whose failure has no class would be a divergence nothing could act on."""
    for gate in compare.gates(export, traces, fixture):
        assert gate.classification in compare.DIVERGENCE_CLASSES


def test_inputs_and_first_reduction_agree(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """F3's own stopping rule, met: the two sides start from the same numbers.

    Trivially true of a synthetic export copied off the Python trace -- which is the point.
    It is the *arrangement* being checked here, so that the two tests below, where the
    agreement is broken on purpose, are known to be able to move something.
    """
    found = compare.gates(export, traces, fixture)
    assert found[0].passed and found[0].absolute == 0.0
    assert found[1].passed


def test_gate_0_is_bit_for_bit(tmp_path: Path, fixture: Any, traces: dict[str, Any]) -> None:
    """One unit in the last place fails it. It has no tolerance and must not acquire one."""
    directory = _write_export(
        tmp_path / "ulp",
        fixture,
        traces["cleverly"].steps[0].before,
        source_arms=traces["cleverly"].arms,
    )
    raw = np.fromfile(directory / "inputs.f64", dtype="<f8")
    raw[0] = np.nextafter(raw[0], np.inf)
    raw.tofile(directory / "inputs.f64")
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert not found[0].passed
    assert compare.first_divergence(found).classification == "input"


def test_gate_1_fails_on_a_different_reduction(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """A reduced regression that differs past the solver bar is a *learner* difference."""
    reference = traces["cleverly"].steps[0].before
    perturbed = type(reference)(
        q_obs=reference.q_obs,
        q=reference.q,
        g=reference.g,
        qr=reference.qr + 1e-3,
        gr1=reference.gr1,
        gr2=reference.gr2,
    )
    directory = _write_export(
        tmp_path / "learner", fixture, perturbed, source_arms=traces["cleverly"].arms
    )
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert found[0].passed
    assert not found[1].passed
    assert compare.first_divergence(found).classification == "learner"


def test_gate_2_reads_r_s_route_against_both_orders(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The reading names all three routes, so a reader can see *which* order matched."""
    route = compare.gates(export, traces, fixture)[2]
    assert "→".join(R_ROUND) in route.reading
    assert "cleverly=" in route.reading and "paper=" in route.reading


def test_gate_2_passes_when_an_order_has_r_s_route(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """The gate can pass. Without this the two failure tests below prove only that it can
    fail, which every broken gate also does."""
    directory = _write_export(
        tmp_path / "matching",
        fixture,
        traces["cleverly"].steps[0].before,
        route=compare._python_route(traces["cleverly"]),
        source_arms=traces["cleverly"].arms,
    )
    assert compare.gates(compare.read_export(directory), traces, fixture)[2].passed


def test_neither_order_is_r_s_round(traces: dict[str, Any]) -> None:
    """**The finding F3 exists to produce**, pinned so a construction change has to move it.

    R's round is ``9 → refit gr → 10 → 8 → refit Qr``. This package's ``"cleverly"`` takes
    R's *equations* in R's order and adopts one reduction vintage per round; its ``"paper"``
    adopts R's *two* vintages and takes the equations in a different order. So the two
    implementations differ in exactly one respect against each of this package's orders, and
    in different respects — which is a sharper statement than "the routes differ" and is what
    F4's ablation is handed.

    Asserted here rather than left to the report because it is a claim about *this package*,
    checkable with no R installed: the R side of it is the published source, transcribed once
    into :data:`R_ROUND` and read against `drtmle`'s own loop by the F3 workflow.
    """
    routes = {order: compare._python_route(trace) for order, trace in traces.items()}
    equations = tuple(label for label in R_ROUND if not label.startswith("refit"))
    strip = lambda route: tuple(l for l in route if not l.startswith("refit"))  # noqa: E731
    vintages = lambda route: tuple(l for l in route if l.startswith("refit"))  # noqa: E731

    assert strip(routes["cleverly"]) == equations
    assert vintages(routes["cleverly"]) != vintages(R_ROUND)

    assert vintages(routes["paper"]) == vintages(R_ROUND)
    assert strip(routes["paper"]) != equations

    assert routes["cleverly"] != R_ROUND and routes["paper"] != R_ROUND


def test_gate_2_fails_when_no_order_has_r_s_route(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    directory = _write_export(
        tmp_path / "order",
        fixture,
        traces["cleverly"].steps[0].before,
        route=("10", "8", "9", "refit:qr", "refit:gr"),
        source_arms=traces["cleverly"].arms,
    )
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert not found[2].passed
    assert compare.first_divergence(found).classification == "update-order"


def test_gate_4_fails_when_r_reached_its_cap(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """``maxIter`` is a budget, not a stopping rule.

    A run that reached it has not been compared on how it stops, and reporting that as
    agreement would put a fact about the dispatch into a column about the estimator.
    """
    directory = _write_export(
        tmp_path / "capped",
        fixture,
        traces["cleverly"].steps[0].before,
        rounds=3,
        max_iter=3,
        source_arms=traces["cleverly"].arms,
    )
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert not found[4].passed
    assert "cap 3" in found[4].reading


def test_only_gates_after_the_first_failure_are_confounded(
    tmp_path: Path, fixture: Any, traces: dict[str, Any]
) -> None:
    """The failing gate is itself readable; everything downstream of it is not.

    Without this the report would either drop the downstream numbers -- and a reader could
    not see how far apart the two ended up -- or print them as findings, which is the mistake
    the whole ordering exists to prevent.
    """
    directory = _write_export(
        tmp_path / "ulp2",
        fixture,
        traces["cleverly"].steps[0].before,
        source_arms=traces["cleverly"].arms,
    )
    raw = np.fromfile(directory / "inputs.f64", dtype="<f8")
    raw[0] = np.nextafter(raw[0], np.inf)
    raw.tofile(directory / "inputs.f64")
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert [gate.confounded for gate in found] == [False] + [True] * (len(found) - 1)
    assert found[0].verdict == "differ"
    assert found[1].verdict == "confounded"


def test_route_labels_a_refit_by_what_it_moved(traces: dict[str, Any]) -> None:
    """R refits ``gr`` and ``Qr`` in two calls; this package's closure returns all three.

    An unlabelled ``refit`` would compare a call against a call and miss that one of them
    produced twice as much -- which is R3's fourth row, the reduction-refit vintage, and the
    one thing about these two implementations that no fitted result carries.
    """
    routes = {order: compare._python_route(trace) for order, trace in traces.items()}
    assert routes["cleverly"] == ("9", "refit:all", "10", "8", "refit:all")
    assert routes["paper"] == ("8", "refit:gr", "10", "refit:qr", "9")


def test_vintage_gate_compares_a_pattern_not_a_count(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """R adopts ``gr`` then ``Qr``; ``"cleverly"`` adopts all three at both of its refits.

    A gate that counted refit *steps* reads ``2`` for all three and calls them the same, which
    makes the one difference no fitted result carries invisible while looking like a working
    comparison. That was the first thing written here.
    """
    reading = compare.gates(export, traces, fixture)[3].reading
    assert reading.startswith("R=gr+qr")
    assert "cleverly=all+all" in reading and "paper=gr+qr" in reading
    assert "matches paper" in reading


def test_the_state_is_read_in_the_traces_arm_order(export: Any, traces: dict[str, Any]) -> None:
    """``drtmle``'s ``a_0`` is ``(1, 0)`` and a ``Trace``'s ``arms`` is ``(0, 1)``.

    Both sides label their columns by arm, so the alignment is by label. Building the R state
    in the *exporter's* order and comparing it against a trace built in the trace's order
    compares arm 1 against arm 0 column for column -- which reads as a ``0.577`` disagreement
    on :math:`g_{r,2}`, a ``learner`` verdict on an axis bug, and is what gate 1 reported
    before this was fixed. Asserted by requiring the two orders to give *different* answers,
    so the check cannot pass by the two happening to coincide.
    """
    trace = traces["cleverly"]
    assert export.arms != trace.arms
    step = export.of("reference")[0]["step"]
    aligned = export.state(step, trace.arms)
    swapped = export.state(step)
    expected = trace.steps[0].before
    assert np.allclose(aligned.gr2, expected.gr2, rtol=0, atol=1e-12)
    assert not np.allclose(swapped.gr2, expected.gr2, rtol=0, atol=1e-3)


# ------------------------------------------------------------------ the report


def test_report_names_the_first_divergence_and_refuses_to_adjudicate(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The report must carry the refusal, not just the table.

    A reader who arrives at this file from a CI artefact has not read ``CLAUDE.md``, and a
    table of divergences with no statement of what one *is* reads as a list of bugs in
    whichever implementation the reader trusts less.
    """
    text = compare.report(export, traces, compare.gates(export, traces, fixture))
    assert "stop-ship 17" in text
    assert "question, not a verdict" in text
    assert "The earliest divergence" in text
    for gate in compare.gates(export, traces, fixture):
        assert gate.name in text
