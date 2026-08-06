"""What makes the F3 comparison an instrument rather than a printout.

**No R runs here and none is installed.** ``docs/roadmap.md``'s F3 row forbids an R dependency
"in the package, in ``nox``, or in any test tier the fast CI runs", so what this module tests
is the half of F3 that is Python: the reader, the gates, the classification and the report.

**And it reads the *committed* R records rather than a synthetic stand-in**, which is the whole
point of committing them: the reader is exercised against the artefact it will actually meet,
gates 0 to 2 agree because two implementations genuinely agree there rather than because a
helper copied one side onto the other, and a test that wants a gate to fail has to break
something specific. ``CLAUDE.md``'s fence is what bounds that -- a committed R record is a
diagnostic record and not a truth -- so nothing here asserts that this package's ``psi``,
``se`` or curve agrees with R's, only that the *instrument* reads what it claims to read.

Two ``DRTMLE`` fits, module-scoped and shared by every test, ~8 s.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_r_compare as compare
from benchmarks import drtmle_trace as trace_module

from cleverly.estimators import targeting

#: The committed record every test starts from: the canonical fixture under R's own default
#: outcome-update route.
RECORD = Path(trace_module.__file__).resolve().parent / "fixtures" / "r-trace-v1-q2"

#: R's round, from its own source: ``fluctuateG``, refit ``gr``, ``fluctuateQ2``,
#: ``fluctuateQ1``, refit ``Qr``.  Transcribed once so that
#: :func:`test_neither_order_is_r_s_round` can be checked with no R record read at all;
#: :func:`test_the_committed_record_is_r_s_round` is what ties the transcription to a run.
R_ROUND = ("9", "refit:gr", "10", "8", "refit:qr")


@pytest.fixture(scope="module")
def fixture() -> Any:
    return trace_module.read_fixture(version="v1")


@pytest.fixture(scope="module")
def traces(fixture: Any) -> dict[str, Any]:
    return {order: compare.python_trace(order, fixture, "v1") for order in ("cleverly", "paper")}


@pytest.fixture(scope="module")
def export() -> Any:
    return compare.read_export(RECORD)


def mutate(destination: Path, **edits: Callable[[bytes], bytes]) -> Path:
    """A copy of the committed record with named files rewritten, manifest kept honest.

    The manifest is *recomputed* rather than left stale, so a test that breaks a gate breaks
    the gate and not the digest check -- otherwise every one of them would fail inside
    :func:`~benchmarks.drtmle_r_compare._verify_manifest` and prove nothing about the gate.
    :func:`test_the_reader_refuses_a_tampered_record` is the one that leaves it stale, on
    purpose.  Double underscores in a keyword stand for dots, since ``arrays.f64.gz`` is not an
    identifier.
    """
    shutil.copytree(RECORD, destination, dirs_exist_ok=True)
    for name, edit in edits.items():
        path = destination / name.replace("__", ".")
        path.write_bytes(edit(path.read_bytes()))
    rows = [
        {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.iterdir())
        if path.name != "manifest.csv"
    ]
    with (destination / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _gzip_edit(change: Callable[[np.ndarray], np.ndarray]) -> Callable[[bytes], bytes]:
    def edit(raw: bytes) -> bytes:
        values = np.frombuffer(gzip.decompress(raw), dtype="<f8").copy()
        return gzip.compress(change(values).astype("<f8").tobytes())

    return edit


def _nudge(values: np.ndarray) -> np.ndarray:
    values[0] = np.nextafter(values[0], np.inf)
    return values


def _rows_to_csv(rows: list[dict[str, str]]) -> bytes:
    """Rewrite a whole CSV from parsed rows, for edits a byte-level replace cannot express.

    Quoting and column widths need not match what R wrote: :func:`mutate` recomputes the
    manifest, and the reader parses rather than diffs.  The *header order* does have to match,
    which is why it comes off the first row rather than being retyped.
    """
    lines = [",".join(rows[0])]
    lines += [",".join(str(row[name]) for name in rows[0]) for row in rows]
    return ("\n".join(lines) + "\n").encode()


# ------------------------------------------------------------------ the reader


def test_the_record_reads_back_in_the_right_shape(export: Any) -> None:
    """The committed blob comes back as the doubles R wrote.

    Gzip is a *container* and not a format -- the bytes inside it are raw little-endian float64
    -- so nothing here acquires a parser to be inexact.  That is the whole reason the record is
    binary: F2's own account has a fast CSV parser reading a fixture short by one unit in the
    last place on 65 of 200 rows, which is the size of difference a first-divergence hunt would
    find and mis-classify as a learner difference.
    """
    reference = export.of("reference")[0]
    recovered = export.state(reference["step"], (0.0, 1.0))
    n = int(export.meta["n"])
    assert recovered.qr.shape == (n, 2)
    assert all(values.size == n for values in export.inputs.values())
    assert export.arms == (1.0, 0.0)


def test_the_reader_refuses_a_tampered_record(tmp_path: Path) -> None:
    """A file whose digest does not match its manifest is unreadable, not slightly wrong."""
    destination = tmp_path / "tampered"
    shutil.copytree(RECORD, destination)
    raw = (destination / "steps.csv").read_text()
    (destination / "steps.csv").write_text(raw.replace("prime", "primed", 1))
    with pytest.raises(ValueError, match="does not match its manifest"):
        compare.read_export(destination)


def test_the_reader_refuses_a_record_with_no_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "unmanifested"
    shutil.copytree(RECORD, destination)
    (destination / "manifest.csv").unlink()
    with pytest.raises(FileNotFoundError, match="not a record"):
        compare.read_export(destination)


def test_the_reader_refuses_a_truncated_blob(tmp_path: Path) -> None:
    """A short record is unreadable, not a shorter comparison."""
    directory = mutate(tmp_path / "short", arrays__f64__gz=_gzip_edit(lambda v: v[:-10]))
    with pytest.raises(ValueError, match="truncated export"):
        compare.read_export(directory)


def test_the_reader_refuses_partial_inputs(tmp_path: Path) -> None:
    """Gate 0 is bit-for-bit and cannot be run against a file with a column missing."""
    directory = mutate(tmp_path / "partial", inputs__f64__gz=_gzip_edit(lambda v: v[:-200]))
    with pytest.raises(ValueError, match="columns of"):
        compare.read_export(directory)


def test_the_reader_names_the_script_when_a_record_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a record"):
        compare.read_export(tmp_path / "nothing-here")


# ------------------------------------------------------------------ the gates


def test_every_gate_is_classified(export: Any, traces: dict[str, Any], fixture: Any) -> None:
    """A gate whose failure has no class would be a divergence nothing could act on."""
    for gate in compare.gates(export, traces, fixture):
        assert gate.classification in compare.DIVERGENCE_CLASSES


def test_f3s_own_stopping_rule_is_cleared_on_v1(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """*"Stop immediately if the trace inputs or the first reduced fits do not agree."*

    Gates 0 to 2 agreeing is what makes everything below them readable as a **construction**
    difference rather than an input or a learner one.  This is not a synthetic arrangement --
    it is two implementations reading one file and fitting the same three regressions -- so a
    regression here invalidates every reading F3 records.
    """
    found = compare.gates(export, traces, fixture)
    assert found[0].passed and found[0].absolute == 0.0
    assert found[1].passed, found[1].reading
    assert found[2].passed and found[2].absolute <= compare.REDUCTION_TOLERANCE


def test_gate_0_is_bit_for_bit(tmp_path: Path, traces: dict[str, Any], fixture: Any) -> None:
    """One unit in the last place fails it.  It has no tolerance and must not acquire one."""
    directory = mutate(tmp_path / "ulp", inputs__f64__gz=_gzip_edit(_nudge))
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert not found[0].passed
    assert compare.first_divergence(found).classification == "input"


def test_truncation_is_vacuous_on_v1_and_says_so(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """``v1``'s bound binds on no row, so the gate has nothing to compare and reports that.

    A gate that passed silently here would read as "the two conventions agree", which is false:
    they were never exercised.  ``v2`` is the fixture that exercises them, and it is a second
    fixture rather than an edit to this one for the reason F2 gives.
    """
    gate = compare.gates(export, traces, fixture)[1]
    assert gate.passed
    assert "0/200" in gate.reading and "vacuous" in gate.reading


def test_truncation_precedes_the_reduced_fit(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The ordering is causal, and it is the ordering ``v2`` proved necessary.

    This package forms :math:`g_{r,2}`'s *target* at the **truncated** mechanism and
    ``estimategrn`` forms it at the untruncated one, so on a fixture where the bound binds the
    reduced fits differ *because of* the truncation.  Ordered the other way round that reads as
    a ``learner`` divergence of ``7.87``, which is true of the fitted values and wrong about
    the cause.
    """
    found = compare.gates(export, traces, fixture)
    assert found[1].classification == "truncation-convention"
    assert found[2].classification == "learner"


def test_neither_order_is_r_s_round(traces: dict[str, Any]) -> None:
    """**The finding F3 exists to produce**, pinned so a construction change has to move it.

    R's round is ``9 → refit gr → 10 → 8 → refit Qr``.  This package's ``"cleverly"`` takes R's
    *equations* in R's order and adopts one reduction vintage per round; its ``"paper"`` adopts
    R's *two* vintages and takes the equations in a different order.  So the two implementations
    differ in exactly one respect against each of this package's orders, and in different
    respects -- a sharper statement than "the routes differ", and what F4's ablation is handed.

    Checked with **no R record read at all**: the R half is the published source, transcribed
    into :data:`R_ROUND`.
    """
    routes = {order: compare._python_route(trace) for order, trace in traces.items()}
    equations = tuple(label for label in R_ROUND if not label.startswith("refit"))

    def strip(route: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(label for label in route if not label.startswith("refit"))

    def vintages(route: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(label for label in route if label.startswith("refit"))

    assert strip(routes["cleverly"]) == equations
    assert vintages(routes["cleverly"]) != vintages(R_ROUND)

    assert vintages(routes["paper"]) == vintages(R_ROUND)
    assert strip(routes["paper"]) != equations

    assert routes["cleverly"] != R_ROUND and routes["paper"] != R_ROUND


def test_the_committed_record_is_r_s_round(export: Any) -> None:
    """The transcription above against what the package actually did, on the record.

    An **instrument-validity** check and not a correctness one: it asks whether the record says
    what this repository believes ``drtmle``'s loop does, which is the premise every reading
    downstream rests on.  It asserts nothing about ``psi``, ``se`` or any curve, which is the
    line ``CLAUDE.md``'s fence draws.
    """
    assert export.route == R_ROUND


def test_gate_3_fails_when_no_order_has_r_s_route(
    tmp_path: Path, traces: dict[str, Any], fixture: Any
) -> None:
    directory = mutate(
        tmp_path / "order",
        steps__csv=lambda raw: raw.replace(b'"round",1,"9"', b'"round",1,"10"', 1),
    )
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert not found[3].passed
    assert compare.first_divergence(found).classification == "update-order"


def test_gate_4_compares_a_pattern_not_a_count(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """R adopts ``gr`` then ``Qr``; ``"cleverly"`` adopts all three at both of its refits.

    A gate that counted refit *steps* reads ``2`` for all three and calls them the same, which
    makes the one difference no fitted result carries invisible while looking like a working
    comparison.  That was the first thing written here.
    """
    reading = compare.gates(export, traces, fixture)[4].reading
    assert reading.startswith("R=gr+qr")
    assert "cleverly=all+all" in reading and "paper=gr+qr" in reading
    assert "matches paper" in reading


def test_gate_5_reads_the_exit_scores_off_the_record(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The three empirical means at exit -- the quantity Theorem 1's premise is about.

    The first version of this comparison could not answer it, because the R side exported no
    scores.  What this asserts is the *mechanism*: that both sides' worst score reaches the
    reading, and that R's stopping bar is named beside it -- "R stopped sooner" and "R stopped
    at a looser bar" are different facts and only the second is true.
    """
    gate = compare.gates(export, traces, fixture)[5]
    assert gate.classification == "stopping-rule"
    assert "tolIC=" in gate.reading and "rounds" in gate.reading
    exits = export.exit_blocks()
    theirs = max(abs(exits[(b, a)]["mean"]) for b in ("D_g", "D_Q") for a in export.arms)
    assert f"{theirs:.2e}" in gate.reading


def test_gate_5_takes_its_verdict_against_this_packages_own_bar(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The bar is ``_negligible_bar(n)``, not ten times what this package happened to achieve.

    The predicate this replaced was ``theirs <= 10 * ours or theirs <= 1e-8`` with ``ours`` the
    **achieved** worst score -- ``7.94e-11`` here, so an effective bar of ``7.9e-10``, which is
    neither R's ``tolIC``, nor this package's ``5e-6``, nor its predicate.  Asserted as
    ``gate.tolerance`` rather than through the verdict, because on this record both the old and
    the new predicate read ``differ`` and a verdict assertion would pass either way.
    """
    gate = compare.gates(export, traces, fixture)[5]
    n = int(export.meta["n"])
    assert gate.tolerance == targeting._negligible_bar(n) == 1e-3 / n
    assert f"{gate.tolerance:.0e}" in gate.reading and "1e-3/n" in gate.reading
    achieved = max(
        abs(float(np.mean(values))) for t in traces.values() for values in t.corrections.values()
    )
    assert gate.tolerance > achieved * 10, "the old predicate was the tighter one on this record"


@pytest.mark.parametrize(
    ("worst", "expected"),
    [(4.9e-6, True), (5.0e-6, True), (5.1e-6, False)],
    ids=("under", "at", "over"),
)
def test_gate_5s_predicate_is_the_loops_own_either_side_of_the_bar(
    tmp_path: Path, traces: dict[str, Any], fixture: Any, worst: float, expected: bool
) -> None:
    """Both sides of ``5e-6`` at ``n = 200``, including the boundary, which ``<=`` includes.

    Driven by rewriting the record's exit ``D_Q`` means, since that is the quantity the gate
    reads; every other block is left alone so nothing else can move the verdict.
    """

    def rewrite(raw: bytes) -> bytes:
        rows = list(csv.DictReader(raw.decode().splitlines()))
        latest = max(int(row["call"]) for row in rows if row["block"] == "D_Q")
        for row in rows:
            if row["block"] in {"D_g", "D_Q"}:
                row["mean"] = f"{worst if int(row['call']) == latest else 0.0:.17g}"
        return _rows_to_csv(rows)

    directory = mutate(tmp_path / f"bar-{worst:g}", blocks__csv=rewrite)
    gate = compare.gates(compare.read_export(directory), traces, fixture)[5]
    assert gate.passed is expected


def test_the_state_is_read_in_the_traces_arm_order(export: Any, traces: dict[str, Any]) -> None:
    """``drtmle``'s ``a_0`` is ``(1, 0)`` and a ``Trace``'s ``arms`` is ``(0, 1)``.

    Both sides label their columns by arm, so the alignment is by label.  Building the R state
    in the *exporter's* order and comparing it against a trace built in the trace's order
    compares arm 1 against arm 0 column for column -- which reads as a ``0.577`` disagreement
    on :math:`g_{r,2}`, a ``learner`` verdict on an axis bug, and is what gate 2 reported
    before this was fixed.  Asserted by requiring the two orders to give *different* answers,
    so the check cannot pass by the two happening to coincide.
    """
    trace = traces["cleverly"]
    assert export.arms != trace.arms
    step = export.of("reference")[0]["step"]
    aligned = export.state(step, trace.arms)
    swapped = export.state(step)
    expected = trace.steps[0].before
    assert np.allclose(aligned.gr2, expected.gr2, rtol=0, atol=1e-8)
    assert not np.allclose(swapped.gr2, expected.gr2, rtol=0, atol=1e-3)


def test_only_gates_after_the_first_failure_are_confounded(
    tmp_path: Path, traces: dict[str, Any], fixture: Any
) -> None:
    """The failing gate is itself readable; everything downstream of it is not.

    Without this the report would either drop the downstream numbers -- and a reader could not
    see how far apart the two ended up -- or print them as findings, which is the mistake the
    whole ordering exists to prevent.
    """
    directory = mutate(tmp_path / "ulp2", inputs__f64__gz=_gzip_edit(_nudge))
    found = compare.gates(compare.read_export(directory), traces, fixture)
    assert [gate.confounded for gate in found] == [False] + [True] * (len(found) - 1)
    assert found[0].verdict == "differ"
    assert found[1].verdict == "confounded"


def test_every_array_gate_carries_a_scale_relative_reading(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """F3's row asks for "absolute **and** scale-relative terms", and they differ in kind.

    ``5e-05`` on an array whose ``sd`` is ``0.6`` is a different fact from the same number on
    one whose ``sd`` is ``5e-05``.  The gates that compare a route or a vintage pattern have no
    scale and report ``nan`` rather than a misleading ``0``.
    """
    scaled = {"0 inputs", "2 first reduced fit", "6 frozen close", "7 correction arrays"}
    for gate in compare.gates(export, traces, fixture):
        if gate.name in scaled:
            assert not np.isnan(gate.relative), gate.name
        if gate.name.startswith(("3 ", "4 ")):
            assert np.isnan(gate.relative), gate.name


# ------------------------------------------------------------------ the report


def test_report_names_the_first_divergence_and_refuses_to_adjudicate(
    export: Any, traces: dict[str, Any], fixture: Any
) -> None:
    """The report must carry the refusal, not just the table.

    A reader who arrives at this file from a CI artefact has not read ``CLAUDE.md``, and a
    table of divergences with no statement of what one *is* reads as a list of bugs in
    whichever implementation the reader trusts less.
    """
    found = compare.gates(export, traces, fixture)
    text = compare.report(export, traces, found)
    assert "stop-ship 17" in text
    assert "question, not a verdict" in text
    assert "The earliest divergence" in text
    for gate in found:
        assert gate.name in text


# ------------------------------------------------------------------ the stopping-bar ladder


LADDER = Path(trace_module.__file__).resolve().parent / "fixtures" / "r-ladder-v1-q2"


@pytest.fixture(scope="module")
def rungs() -> Any:
    return compare.read_ladder(LADDER)


def test_the_ladder_reads_loosest_first_and_every_rung_converged(rungs: Any) -> None:
    """Order matters — the verdict reads the loosest as "before" and the tightest as "after".

    And every rung reaching its own bar is the precondition for reading any of them: a capped
    run is not a tighter measurement of the same thing, it is a run that did not finish, and
    :attr:`~benchmarks.drtmle_r_compare.Rung.converged` is what keeps the two apart.
    """
    assert [rung.tol_ic for rung in rungs] == sorted((r.tol_ic for r in rungs), reverse=True)
    assert all(rung.converged for rung in rungs)


def test_the_ladder_is_monotone_in_the_bar(rungs: Any) -> None:
    """R's achieved score falls as its bar tightens.

    The sanity check that the ladder is a ladder. A run whose score did not fall would mean the
    knob is not doing what its name says, and every reading off it would be about something
    else.
    """
    scores = [rung.worst_score for rung in rungs]
    assert scores == sorted(scores, reverse=True)


def test_psi_barely_moves_while_se_does(rungs: Any) -> None:
    """Between the converged rungs, which is what the extra equations are supposed to do.

    Deliberately *not* against R's own default rung: that one has not converged, and there
    ``psi[ate]`` moves by a tenth of a standard error — which is a fact about `drtmle`'s
    shipped `maxIter = 3` rather than about the estimand, and is recorded in the document
    rather than smoothed over here.
    """
    converged = [rung for rung in rungs if rung.worst_score < 1e-5]
    assert len(converged) >= 2
    psis = [rung.estimates["ate"]["psi"] for rung in converged]
    assert max(psis) - min(psis) < 1e-4


def test_the_verdict_is_one_of_the_four_declared_outcomes(
    rungs: Any, traces: dict[str, Any]
) -> None:
    reading = compare.ladder_verdict(rungs, traces)
    assert reading["verdict"] in {"closed", "persists", "partial", "unreachable"}


def test_the_verdict_reads_unreachable_when_no_rung_converged(traces: dict[str, Any]) -> None:
    """The third outcome, and the one no run on record produced.

    It has to be reachable or it is not an outcome — a decision rule with a branch nothing can
    take is a branch that will be wrong the first time it matters.
    """
    capped = [
        compare.Rung(
            tol_ic=1e-12,
            rounds=100,
            capped=True,
            worst_score=1e-3,
            spreads={(compare.LADDER_BLOCK, compare.LADDER_ARM): 0.06},
            estimates={"ate": {"psi": 0.2, "se": 0.07}, "ey1": {"psi": 0.7, "se": 0.05}},
        )
    ]
    assert compare.ladder_verdict(capped, traces)["verdict"] == "unreachable"


def test_the_verdict_reads_persists_when_the_bar_explains_little(
    rungs: Any, traces: dict[str, Any]
) -> None:
    """The second outcome, on a synthetic tightest rung that barely moved.

    Built from the real loosest rung so the "before" ratio is the measured one; only the
    tightest is invented, which is the minimum needed to reach the branch.
    """
    loosest = rungs[0]
    stuck = compare.Rung(
        tol_ic=1e-10,
        rounds=21,
        capped=False,
        worst_score=1e-11,
        spreads={**loosest.spreads},
        estimates=loosest.estimates,
    )
    assert compare.ladder_verdict([loosest, stuck], traces)["verdict"] == "persists"


def test_the_ladder_reader_fails_closed_on_a_tampered_rung(tmp_path: Path) -> None:
    """Same discipline as the trace records: the digest is checked before a number is read."""
    destination = tmp_path / "ladder"
    shutil.copytree(LADDER, destination)
    rung = next(child for child in sorted(destination.iterdir()) if child.is_dir())
    raw = (rung / "summary.csv").read_text()
    (rung / "summary.csv").write_text(raw.replace("ate", "ATE", 1))
    with pytest.raises(ValueError, match="does not match its manifest"):
        compare.read_ladder(destination)


def test_the_ladder_report_carries_the_declared_bars(rungs: Any, traces: dict[str, Any]) -> None:
    """A verdict with no thresholds beside it is a claim a reader cannot check."""
    text = compare.ladder_report(rungs, traces)
    assert f"{compare.CLOSED_SPREAD_RATIO}" in text and f"{compare.CLOSED_SE_RATIO}" in text
    assert "declared before the first rung was read" in text
    for name in traces:
        assert name in text


def test_the_explained_fraction_is_reported_in_every_orientation(
    rungs: Any, traces: dict[str, Any]
) -> None:
    """A ratio's distance from agreement is not invariant, so one number is not the answer.

    On the committed rungs the three readings are ``92%``, ``64%`` and ``64%`` -- the ratio as
    computed, the ratio reversed, and the reduction in the raw absolute gap, which has no
    orientation to choose.  Asserted as *disagreeing*, so the test cannot pass by the three being
    computed the same way; and the verdict is asserted to be the same under each, which is what
    makes ``partial`` a robust reading rather than an artefact of the orientation.
    """
    reading = compare.ladder_verdict(rungs, traces)
    orientations = (
        reading["explained"],
        reading["explained_reversed"],
        reading["explained_absolute"],
    )
    assert reading["explained"] != pytest.approx(reading["explained_reversed"])
    assert reading["explained_reversed"] == pytest.approx(reading["explained_absolute"])
    assert all(value >= compare.PERSISTS_FRACTION for value in orientations)
    text = compare.ladder_report(rungs, traces)
    for value in orientations:
        assert f"{value:.0%}" in text


def test_the_ladder_reads_each_order_against_its_own_ratios(
    rungs: Any, traces: dict[str, Any]
) -> None:
    """One order's percentage beside another order's ratios is not a comparison of anything.

    That is how the verdict line was first written -- ``92%`` is ``paper``'s and ``1.266`` /
    ``1.051`` are ``cleverly``'s -- and it is invisible unless the two orders are checked to land
    on *opposite* sides of agreement, which on this ladder they do.
    """
    reading = compare.ladder_verdict(rungs, traces)
    spreads, ses = reading["after_by_order"], reading["se_ratio_by_order"]
    assert set(spreads) == set(ses) == set(traces)
    assert min(spreads.values()) < 1.0 < max(spreads.values()), "R lands between the two orders"
    nearest = str(reading["nearest_order"])
    assert reading["spread_ratio_after"] == pytest.approx(spreads[nearest])
    assert reading["se_ratio_after"] == pytest.approx(ses[nearest])
    text = compare.ladder_report(rungs, traces)
    for name in traces:
        assert f"`{name}` {spreads[name]:.3f}" in text
        assert f"`{name}` {ses[name]:.3f}" in text


# ------------------------------------ the retraction F3-closeout made, and what pins it in place

#: The R script and the report, read as *text*.  Both assertions below are about strings this
#: repository controls; neither runs anything.
R_SCRIPT = Path(trace_module.__file__).resolve().parent / "r" / "drtmle_reference.R"
REPORT = Path(__file__).resolve().parents[2] / "docs" / "drtmle" / "r-differential.md"


def test_every_committed_record_is_a_univariate_run() -> None:
    """What makes the concordance's reading of ``eval_Dstar_Q`` the applicable one.

    ``docs/drtmle/theorem-concordance.md`` §10 says the initial ``g`` enters that function's
    **bivariate** branch and that the argument is unused on the univariate one.  That sentence
    only settles the ``D_Q`` question here if every record was taken under the univariate
    reduction, so this asserts the call sites rather than trusting the prose -- a structural pin,
    exactly as ``tests/unit/test_sequential_design.py`` pins a design matrix it cannot
    statistically adjudicate.  ``meta.csv`` carries no ``reduction`` field, which is why the
    script is what is read.
    """
    source = R_SCRIPT.read_text()
    assert 'reduction = "univariate"' in source
    assert 'reduction = "bivariate"' not in source


def test_the_targeted_g_flag_is_a_call_site_and_the_record_says_which_rows(export: Any) -> None:
    """``at_targeted_g`` records which array was *passed*, and cannot record whether it was read.

    Two things are pinned because the report states both and an earlier revision got the second
    of them wrong.  ``D_g`` is handed the targeted mechanism at every call.  ``D_Q`` is handed
    the initial one from round 1 onward -- but *not* in the ``prime`` phase at round 0, where R
    has just set ``gnStar <- gn`` and the two arrays are the same object, so the flag reads
    ``TRUE`` there.  "FALSE on every ``D_Q``" was two rows too many.
    """
    rows = [row for row in export.blocks if row["block"] in {"D_g", "D_Q"}]
    assert rows, "the committed record carries no correction-block rows"
    for row in rows:
        expected = not (row["block"] == "D_Q" and row["round"] >= 1)
        assert bool(row["at_targeted_g"]) is expected, row


def test_the_report_carries_no_live_univariate_d_q_candidate() -> None:
    """The retraction itself, guarded as a claim rather than left to a reader's memory.

    The flag is a fact and stays on the record; the *inference* that it leaves equation (10)'s
    block uncentred is bivariate-only and was withdrawn, because on the reduction every run here
    uses, the callee does not read the argument.  A future revision that reinstates the
    inference without reinstating the reduction it belongs to fails here.
    """
    text = REPORT.read_text()
    assert "bivariate-only" in text
    assert "the argument is unused" in text
    assert "leaves the curve uncentred exactly where the" not in text
