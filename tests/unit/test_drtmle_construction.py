"""What F4's construction harness has to be true of before a decision cohort reads it.

``benchmarks/drtmle_construction.py`` is piece F4's instrument and
``docs/drtmle/construction-contrasts.md`` is its record.  Four things have to hold before any
verdict taken through it means anything, and each is a class of defect this repository has
already been bitten by once:

* **the instrument does not move what it measures** -- the reference arm through the harness is
  the shipped estimator, bit for bit, and no arm leaves a module patched behind it;
* **the R-style arm is R's round** -- an *instrument-validity* check, and the one thing
  ``CLAUDE.md``'s fence lets a test read a committed R record for.  It asserts nothing about
  ``psi``, ``se`` or any curve;
* **each arm moves its one factor and no other** -- a contrast that moves two things cannot say
  which of them moved the number, which is the whole reason F4 is a matrix rather than an arm;
* **the run fails closed** -- a moved rule, a changed configuration, an overlapping cohort or an
  incomplete draw set is refused *before* anything is fitted.

**Budget.** Nine ``DRTMLE`` fits on 200 rows with frozen closed-form nuisances, shared across the
module.  No fit here uses the tier-2 law: these are questions about the *harness*, and the frozen
fixture answers them deterministically where a simulated draw would answer them on average.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_construction as construction
from benchmarks import drtmle_r_compare as compare
from benchmarks import drtmle_trace as trace_module

from cleverly.estimators import targeting

#: R's round, transcribed from the published source and **not** read off a record -- the same
#: constant ``tests/unit/test_drtmle_r_compare.py`` declares, repeated rather than imported so
#: that a test of the belief and a test of the record cannot drift into one test.
R_ROUND = ("9", "refit:gr", "10", "8", "refit:qr")


@pytest.fixture(scope="module")
def fixture() -> Any:
    return trace_module.read_fixture()


@pytest.fixture(scope="module")
def traces(fixture: Any) -> dict[str, Any]:
    """This package's two routes and the R-style arm, on the one frozen fixture."""
    return {
        order: compare.python_trace(order=order, fixture=fixture)
        for order in ("cleverly", "paper", "r-style")
    }


@pytest.fixture(scope="module")
def export() -> Any:
    """R's committed record at its own default ``Qsteps``.

    Read **only** by the instrument-validity tests below.  ``CLAUDE.md``: a committed R record
    may be read by ``benchmarks/`` and by tests that check an instrument is what it claims to
    be, and never by a test asserting this package's ``psi``, ``se`` or curve agrees with it.
    """
    return compare.read_export(Path("benchmarks/fixtures/r-trace-v1-q2"))


class TestTheInstrumentDoesNotMoveTheFit:
    """The reference arm through the harness is the shipped estimator, bit for bit."""

    def test_the_reference_arm_is_an_ordinary_drtmle(self, fixture: Any) -> None:
        """``arm_estimator("cleverly")`` builds the class a caller would, not a subclass."""
        settings = {"outcome_learner": trace_module.FrozenOutcome("v1")}
        built = construction.arm_estimator("cleverly", settings)
        assert type(built).__name__ == "DRTMLE"
        assert built.update_order == "cleverly"
        assert built.reduced_crossfit == "pooled"

    def test_an_arm_patch_restores_every_name_it_installed(self) -> None:
        originals = {
            name: getattr(targeting, name)
            for name in ("_close_at_frozen_reductions", "_NEGLIGIBLE")
        }
        for arm in construction.ARMS:
            with construction._arm_patch(arm):
                pass
        for name, value in originals.items():
            assert getattr(targeting, name) is value or getattr(targeting, name) == value

    def test_a_raise_inside_a_patched_arm_still_restores(self) -> None:
        """The ``finally`` and not the happy path, since a failing fit is the case that matters."""
        before = targeting._close_at_frozen_reductions
        with pytest.raises(RuntimeError), construction._arm_patch("no-close"):
            raise RuntimeError("the alternation raised")
        assert targeting._close_at_frozen_reductions is before

    def test_the_no_close_patch_is_installed_while_it_is_in_force(self) -> None:
        """The control for the two tests above: they would pass against a patch that never ran."""
        before = targeting._close_at_frozen_reductions
        with construction._arm_patch("no-close"):
            assert targeting._close_at_frozen_reductions is not before
        with construction._arm_patch("loose"):
            assert targeting._NEGLIGIBLE == 1.0
            assert targeting._negligible_bar(200) == pytest.approx(1.0 / 200)


class TestTheRStyleArmIsRsRound:
    """An **instrument-validity** check: does the arm labelled R-style take R's round?

    Not a correctness check and not a parity check.  Every assertion here is about a *route* --
    which equations a round solves, in what order, and which reduced regressions each refit
    contributes -- and none is about an estimate.  Changing this package to match R is
    stop-ship 17; establishing that a benchmark arm is the trajectory it is named for is what
    F4 needs before it reads a contrast against it.
    """

    def test_the_arm_takes_rs_equation_order_and_rs_two_vintages(
        self, traces: dict[str, Any]
    ) -> None:
        assert compare._python_route(traces["r-style"]) == R_ROUND

    def test_it_is_neither_shipped_order_relabelled(self, traces: dict[str, Any]) -> None:
        """F3's finding, pinned: the two factors are **crossed**.

        ``"cleverly"`` takes R's equation order with one reduction vintage and ``"paper"`` takes
        R's two vintages under a different equation order, so a contrast against either alone
        moves two things at once.  If this ever passes against a relabelling, F4's first two
        contrasts have stopped being one factor each.
        """
        routes = {name: compare._python_route(trace) for name, trace in traces.items()}
        assert routes["r-style"] != routes["cleverly"]
        assert routes["r-style"] != routes["paper"]
        vintages = compare._vintages_in
        assert vintages(routes["cleverly"]) == ("all", "all")
        assert vintages(routes["r-style"]) == vintages(routes["paper"]) == ("gr", "qr")
        equations = tuple(label for label in routes["r-style"] if not label.startswith("refit"))
        assert equations == tuple(
            label for label in routes["cleverly"] if not label.startswith("refit")
        )
        assert equations != tuple(
            label for label in routes["paper"] if not label.startswith("refit")
        )

    def test_the_committed_r_record_takes_the_same_round(self, export: Any) -> None:
        """The transcription above against what this repository recorded R doing.

        The premise every F4 contrast against the R-style arm rests on.  It compares two
        *routes* and asserts nothing about any estimate.
        """
        assert export.route == R_ROUND

    def test_the_gates_that_read_the_route_pass_for_this_arm(
        self, export: Any, traces: dict[str, Any]
    ) -> None:
        """F3 recorded gate 3 as matching neither order; the third arm is what it was missing."""
        route_gate = compare._gate_route(export, traces)
        vintage_gate = compare._gate_vintage(export, traces)
        assert route_gate.passed and vintage_gate.passed
        assert "r-style" in route_gate.reading and "r-style" in vintage_gate.reading

    def test_the_arm_refuses_a_shape_its_parity_does_not_fit(self) -> None:
        """Both guards, or the refits do not come in twos and the adoption is not R's."""
        with pytest.raises(ValueError, match="needs both guards"):
            trace_module.RStyleDRTMLE(guard=("Q",))
        with pytest.raises(ValueError, match="cannot also take update_order"):
            trace_module.RStyleDRTMLE(update_order="paper")

    def test_the_tracing_arm_composes_in_the_one_direction_that_reads_adoption(self) -> None:
        """``RStyleDRTMLE`` before ``TracingDRTMLE`` in the MRO, and the order is load-bearing.

        :func:`benchmarks.drtmle_trace.vintages` reads adoption by comparing what a refit
        *produced* against what the next solve *read*, so the adoption has to happen outside the
        recording closure.  Composed the other way the closure returns the already-adopted set,
        both sides compare equal, and the arm silently reads as ``"cleverly"`` -- a right fit
        filed under the wrong route.  Pinned structurally because the failure is invisible in
        every number the fit reports.
        """
        mro = trace_module.TracingRStyleDRTMLE.__mro__
        assert mro.index(trace_module.RStyleDRTMLE) < mro.index(trace_module.TracingDRTMLE)


class TestEachArmMovesOneFactor:
    """Every arm is one construction difference, and the harness is where that is checked."""

    def test_no_close_removes_the_closing_pass_and_nothing_else(self, fixture: Any) -> None:
        plain = trace_module.trace(fixture, order="cleverly")
        with construction._arm_patch("no-close"):
            without = trace_module.trace(fixture, order="cleverly")
        assert plain.exit["closing"] > 0
        assert without.exit["closing"] == 0
        # The rounds are the alternation's and the closing pass does not refit, so removing it
        # must leave the round count exactly where it was. A different count would mean the
        # patch had reached the loop rather than its exit.
        assert without.exit["rounds"] == plain.exit["rounds"]
        assert without.exit["exit_reason"] == plain.exit["exit_reason"]

    def test_the_no_close_state_is_the_pre_close_state_the_trace_records(
        self, fixture: Any
    ) -> None:
        """The identity closing pass hands back what the loop exits at, checked against F2.

        ``Trace.boundary()`` is the pair F2 built for exactly this comparison, so the arm is
        checked against the *recorded* boundary rather than against its own construction.
        """
        traced = trace_module.trace(fixture, order="cleverly")
        pre, _ = traced.boundary()
        with construction._arm_patch("no-close"):
            without = trace_module.trace(fixture, order="cleverly")
        final = without.steps[-1].after
        np.testing.assert_allclose(final.q_obs, pre.q_obs, rtol=0, atol=1e-12)
        np.testing.assert_allclose(final.g, pre.g, rtol=0, atol=1e-12)

    def test_loose_is_rs_bar_and_moves_nothing_else(self) -> None:
        with construction._arm_patch("loose"):
            assert targeting._negligible_bar(600) == pytest.approx(1.0 / 600)
            assert targeting._STALL_FACTOR == 0.95
            assert targeting._UNSOLVED == 1e-6

    def test_raw_moves_the_reduction_target_bound_and_no_other(self, fixture: Any) -> None:
        """``RawTargetDRTMLE`` reaches ``fit_reduced``'s bound and not the covariate bounds."""
        settings = {
            "outcome_learner": trace_module.FrozenOutcome("v2"),
            "treatment_learner": trace_module.FrozenMechanism("v2"),
            "n_folds": 3,
            "learner_folds": 3,
            "g_bounds": (0.15, 0.85),
            "random_state": 0,
            "simultaneous": False,
            "estimands": ["ey1", "ey0", "ate"],
        }
        built = construction.arm_estimator("raw", settings)
        # The constructor bound -- the covariate denominators -- is untouched; only the bound
        # `fit_reduced` forms g_{r,2}'s target at moves.
        assert built.g_bounds == (0.15, 0.85)
        assert construction.INERT_BOUNDS == (0.0, 1.0)

    def test_nested_moves_only_the_crossfit_keyword(self) -> None:
        settings = {"outcome_learner": trace_module.FrozenOutcome("v1")}
        nested = construction.arm_estimator("nested", settings)
        plain = construction.arm_estimator("cleverly", settings)
        assert nested.reduced_crossfit == "nested" and plain.reduced_crossfit == "pooled"
        assert nested.update_order == plain.update_order
        assert nested.guard == plain.guard

    def test_every_declared_contrast_names_an_arm_that_exists(self) -> None:
        for arm, base in construction.CONTRASTS:
            assert arm in construction.ARMS and base in construction.ARMS
            assert construction.ARMS[arm]["factor"]
        factors = [construction.ARMS[arm]["factor"] for arm, _ in construction.CONTRASTS]
        assert len(factors) == len(set(factors)), "one primary contrast per factor"


class TestTheTruncationReadingIsExactAndNotVacuous:
    """F4's sixth factor, and the two fixtures have to be the two regimes it declares."""

    @pytest.fixture(scope="class")
    def rows(self) -> list[Any]:
        return construction.truncation_reading()

    def test_the_two_fixtures_are_the_two_declared_regimes(self, rows: list[Any]) -> None:
        clipped = {row.fixture: (row.clipped, row.rows) for row in rows}
        assert clipped["v1"][0] == 0, "v1 is the bound-inactive control"
        assert clipped["v2"][0] > 0.2 * clipped["v2"][1], "v2 is the bound-active stress design"

    def test_the_convention_is_inert_where_the_bound_is_slack(self, rows: list[Any]) -> None:
        assert all(row.identical for row in rows if row.fixture == "v1")

    def test_it_is_not_inert_where_the_bound_binds(self, rows: list[Any]) -> None:
        """The non-vacuity control: without this the reading above is a test of nothing."""
        initial = [r for r in rows if r.fixture == "v2" and r.stage == "initial"]
        assert any(not row.identical and row.worst > 1e-6 for row in initial)


class TestTheRunFailsClosed:
    """Every reason a run may not be read against a committed manifest, refused before a fit."""

    @pytest.fixture()
    def manifest(self) -> dict[str, Any]:
        return construction.prereg(draws=8)

    def _seeds(self, manifest: dict[str, Any], cohort: str) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in manifest["cohorts"][cohort]]

    def test_a_faithful_run_passes(self, manifest: dict[str, Any]) -> None:
        seeds = self._seeds(manifest, "selection")
        assert (
            construction.validate_prereg(
                manifest, cohort="selection", seeds=seeds, draws_declared=len(seeds)
            )
            == []
        )

    def test_a_rule_that_moved_after_the_freeze_is_refused(self, manifest: dict[str, Any]) -> None:
        manifest["rule"]["negligible_effect"] = 0.5
        complaints = construction.validate_prereg(
            manifest,
            cohort="selection",
            seeds=self._seeds(manifest, "selection"),
            draws_declared=8,
        )
        assert any("negligible_effect" in line for line in complaints)

    def test_a_changed_configuration_is_refused(self, manifest: dict[str, Any]) -> None:
        manifest["configuration"]["reduced_learner"] = "gam"
        complaints = construction.validate_prereg(
            manifest,
            cohort="selection",
            seeds=self._seeds(manifest, "selection"),
            draws_declared=8,
        )
        assert any("reduced_learner" in line for line in complaints)

    def test_a_shared_data_seed_between_the_cohorts_is_refused(
        self, manifest: dict[str, Any]
    ) -> None:
        """On the **data** seed, since two draws sharing one under different splits are the same
        rows twice -- and an effect reproduced on the draws that produced it has not reproduced.
        """
        audit = self._seeds(manifest, "audit")
        contaminated = [(audit[0][0], 999), *self._seeds(manifest, "selection")[1:]]
        complaints = construction.validate_prereg(
            manifest, cohort="selection", seeds=contaminated, draws_declared=8
        )
        assert any("audit draws" in line for line in complaints)

    def test_a_draw_outside_the_committed_cohort_is_refused(self, manifest: dict[str, Any]) -> None:
        complaints = construction.validate_prereg(
            manifest, cohort="selection", seeds=[(12345, 6789)], draws_declared=1
        )
        assert any("not in the committed" in line for line in complaints)

    def test_an_incomplete_draw_set_is_unresolved_by_rule(self, manifest: dict[str, Any]) -> None:
        seeds = self._seeds(manifest, "selection")[:3]
        complaints = construction.validate_prereg(
            manifest, cohort="selection", seeds=seeds, draws_declared=8
        )
        assert any("completeness" in line for line in complaints)

    def test_an_unknown_cohort_is_refused(self, manifest: dict[str, Any]) -> None:
        complaints = construction.validate_prereg(
            manifest, cohort="decision", seeds=[], draws_declared=0
        )
        assert any("not one of" in line for line in complaints)


class TestTheFrozenDesign:
    """The manifest says what was declared, and the sizing says what it can answer."""

    def test_the_cohorts_are_disjoint_on_the_data_seed(self) -> None:
        cohorts = construction.cohort_seeds(construction.COHORT_SEED, 80)
        selection = {data for data, _ in cohorts["selection"]}
        audit = {data for data, _ in cohorts["audit"]}
        assert selection and audit and not (selection & audit)

    def test_reserving_the_sizing_stream_does_not_move_the_cohorts(self) -> None:
        """``spawn`` gives child ``i`` the same state whatever ``n`` is, and this study depends
        on it: :data:`~benchmarks.drtmle_construction.PILOT_PAIRED_SPREAD` was measured on a
        third child, and sizing a study on draws that then move it would be no freeze at all.
        """
        two = np.random.SeedSequence(construction.COHORT_SEED).spawn(2)
        three = np.random.SeedSequence(construction.COHORT_SEED).spawn(3)
        for left, right in zip(two, three[:2], strict=True):
            assert np.array_equal(left.generate_state(16), right.generate_state(16))

    def test_the_smaller_size_takes_a_prefix_of_the_larger(self) -> None:
        manifest = construction.prereg()
        big = construction.seeds_for(manifest, "selection", 2400)
        small = construction.seeds_for(manifest, "selection", 600)
        assert len(small) < len(big)
        assert small == big[: len(small)]

    def test_the_sizing_is_derived_and_says_what_each_size_can_answer(self) -> None:
        sizing = construction.replicate_count()["sizes"]
        assert sizing["2400"]["committed"] >= sizing["2400"]["required"]
        assert sizing["2400"]["powered"] == "moved and flat"
        # n=600's spreads are five to twenty times larger, so the committed count cannot reach
        # the declared half-width and the manifest has to say so rather than imply it.
        assert sizing["600"]["committed"] < sizing["600"]["required"]
        assert sizing["600"]["powered"] == "moved only"
        for block in sizing.values():
            assert set(block["mde"]) == {f"{a}~{b}" for a, b in construction.CONTRASTS}

    def test_the_manifest_round_trips_and_is_reviewable(self, tmp_path: Path) -> None:
        manifest = construction.prereg(draws=8)
        path = construction.write_prereg(manifest, tmp_path / "prereg.json")
        text = path.read_text(encoding="utf-8")
        assert json.loads(text) == manifest
        assert text.startswith("{\n") and text.endswith("\n")


class TestTheContrastArithmetic:
    """The verdict is read off the interval, and the third verdict is not a weak second."""

    def test_an_interval_excluding_zero_is_moved(self) -> None:
        assert construction._verdict(0.2, 0.4, 0.125) == "moved"
        assert construction._verdict(-0.4, -0.2, 0.125) == "moved"

    def test_an_interval_inside_the_margin_is_flat(self) -> None:
        assert construction._verdict(-0.05, 0.05, 0.125) == "flat"

    def test_a_wide_interval_containing_zero_is_unresolved(self) -> None:
        """Neither a pass nor a fail: it says the study cannot tell, which is a recorded
        outcome and not a weak ``flat``."""
        assert construction._verdict(-2.0, 2.0, 0.125) == "unresolved"
        assert construction._verdict(float("nan"), float("nan"), 0.125) == "unresolved"

    def test_a_paired_difference_is_taken_within_a_draw(self) -> None:
        """Two arms on two draws pair by ``data_seed``, never by position."""
        rows = [
            construction.FitRow(
                cohort="selection",
                cell="q-drift",
                n=600,
                data_seed=seed,
                fold_seed=1,
                arm=arm,
                estimand="ate",
                psi=0.0,
                truth=0.0,
                std_error=1.0,
                root_n_remaining=value,
                score_8=0.0,
                score_9=0.0,
                score_10=0.0,
                reduction_drift=0.0,
                identity_failures=0,
                score_failures=0,
                valid=True,
                rounds=1,
                closing=1,
                exit_reason="tolerance",
                failure="",
                bound_active=False,
                initial_clip_share=0.0,
                seconds=0.0,
            )
            for seed, arm, value in (
                (1, "cleverly", 1.0),
                (1, "r-style", 1.5),
                (2, "cleverly", 3.0),
                (2, "r-style", 3.5),
            )
        ]
        contrasts = construction.contrast_rows(rows)
        vintage = [
            row
            for row in contrasts
            if row.contrast == "r-style~cleverly" and row.column == "root_n_remaining"
        ]
        assert len(vintage) == 1
        assert vintage[0].mean == pytest.approx(0.5)
        assert vintage[0].draws == 2
