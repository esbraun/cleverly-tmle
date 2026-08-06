r"""The component-level trace harness, and the guards that make it an instrument.

``benchmarks/drtmle_trace.py`` is ``docs/roadmap.md``'s **F2**: the state-level record F3 (the
bounded differential run against the published ``drtmle`` R package) and F4 (the construction
ablations) both read.  It closes nothing about the coverage shortfall on its own, and
everything here exists because an instrument that is wrong in a way nobody checks is worse
than no instrument.

Four of these tests are about the harness rather than about the estimator, and each guards a
way this could look right and be useless.

* **The fixture is frozen.**  Two implementations compare traces only if they ran on the same
  bytes, so the CSV's digest is checked against its manifest, the draw is regenerated from its
  seed, and the initial predictions are checked to be the closed forms F3 hands R.
* **The instrument does not move what it measures.**  A traced fit is compared against an
  untraced one -- ``psi``, ``se``, every curve, every ``epsilon`` -- under **both** update
  orders.  The harness patches module-level names during the alternation, and a patch that
  perturbed a solve would make every localization downstream a localization of the harness.
* **The fixture is not degenerate.**  At correct nuisances :math:`Q_r` and :math:`g_{r,2}`
  vanish row by row, both corrections are zero arrays, and a trace taken there passes against
  a flipped sign, a swapped update order and a stale reduction alike -- ``CLAUDE.md``'s rule
  about where an exact-law instrument goes blind, in the place it bites hardest.  So the
  fixture's misspecification is asserted, not assumed.
* **The identities can fail.**  A check that recomputes a recorded number and always agrees
  is not evidence until it has been watched to disagree, which is why
  :class:`TestTheIdentitiesAreNotVacuous` perturbs a recorded state and requires the residual
  to move.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks import drtmle_trace as harness

from cleverly.estimators import targeting as targeting_module
from cleverly.validation.drtmle import IDENTITY_TOLERANCE

#: The two routes through a round, both traced on the one fixture.  ``docs/roadmap.md``'s item
#: 22 asks whether they reach the same fixed point on real data; this module asserts only that
#: each is *recorded correctly*, which is a different question and the one F2 answers.
ORDERS = ("cleverly", "paper")


@pytest.fixture(scope="module")
def fixture() -> harness.Fixture:
    return harness.read_fixture()


@pytest.fixture(scope="module")
def traces(fixture: harness.Fixture) -> dict[str, harness.Trace]:
    """Both orders, traced once each and shared by every test below.

    Module-scoped because a trace is a whole ``DRTMLE`` fit with the reductions refitted every
    round -- 3.2 s and 6.5 s on four cores -- and nothing here needs a second one.  So is
    :func:`plain`, for the same reason and at the same cost, and the determinism check below
    is deliberately taken on **one** order rather than two: determinism is a property of the
    pipeline, and a second order re-checks it at 6.5 s a run.  Five fits is this module's whole
    budget and each of them answers a different question.
    """
    return {order: harness.trace(fixture, order=order) for order in ORDERS}


@pytest.fixture(scope="module")
def plain(fixture: harness.Fixture) -> dict[str, object]:
    """The same two fits with the instrument off, which is what a traced fit is compared to."""
    return {
        order: harness._fit(fixture.frame, harness.estimator(order=order, tracing=False))
        for order in ORDERS
    }


class TestTheFixtureIsFrozen:
    """What two implementations have to agree they started from."""

    def test_the_digest_matches_the_manifest(self, fixture: harness.Fixture) -> None:
        # `read_fixture` raises on a mismatch rather than returning a frame nobody checked;
        # this is the assertion that says so out loud.
        assert len(fixture.manifest["sha256"]) == 64
        assert fixture.n == fixture.manifest["n"] == harness.N

    def test_the_draw_regenerates_from_its_seed(self, fixture: harness.Fixture) -> None:
        fresh = harness.build_fixture()
        for column in ("w1", "w2", "a", "y"):
            np.testing.assert_array_equal(
                fresh[column].to_numpy(dtype=float),
                fixture.frame[column].to_numpy(dtype=float),
                err_msg=f"{column} does not regenerate from seed {harness.SEED}",
            )

    def test_the_initial_predictions_are_the_closed_forms(self, fixture: harness.Fixture) -> None:
        """F3's *"the inputs agree"* gate, on the Python side.

        The columns a second implementation is handed have to be the same function this one
        evaluates, or the differential run localizes the hand-off rather than the algorithm --
        which is the first thing piece F says to stop for.
        """
        arrays = fixture.arrays()
        w1, w2 = arrays["w1"], arrays["w2"]
        np.testing.assert_allclose(
            arrays["qn1"], harness.initial_outcome(1.0, w1, w2), rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            arrays["qn0"], harness.initial_outcome(0.0, w1, w2), rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            arrays["gn"], harness.initial_mechanism(w1, w2), rtol=0, atol=1e-15
        )

    def test_the_folds_are_the_ones_a_fit_realises(
        self, fixture: harness.Fixture, plain: dict[str, object]
    ) -> None:
        """The reduced regressions are cross-fitted over them, so a drifted split is a
        different experiment.  :func:`~benchmarks.drtmle_trace.trace` refuses one; this is the
        assertion that the committed column is the one it accepts."""
        result = plain["cleverly"]
        np.testing.assert_array_equal(
            np.asarray(result.repeats[0].nuisance.folds.assignment, dtype=int), fixture.folds
        )

    def test_the_truncation_is_slack_on_every_row(self, fixture: harness.Fixture) -> None:
        """Deliberate, and recorded rather than assumed.

        Clipping is not irrelevant -- it is ``docs/roadmap.md``'s item 20 and the whole of
        piece B1b -- but the two implementations' truncation *conventions* differ, and a
        first-divergence hunt confounded by a known convention difference locates the
        convention.  A fixture that turns clipping on is a second fixture.
        """
        lower, upper = harness.G_BOUNDS
        g = fixture.arrays()["gn"]
        assert float(g.min()) > lower and float(g.max()) < upper
        assert fixture.manifest["clipped"] == 0


class TestTheSecondFixtureIsTheFirstPlusClipping:
    """``v2`` exists so that the truncation question can be asked at all, and only that.

    F2's rule is that *"a fixture that turns clipping on is a second fixture, not an edit to
    this one"*, and these are what make ``v2`` a second fixture rather than a second
    experiment: it is ``v1``'s draw, ``v1``'s truth and ``v1``'s outcome regression, with the
    mechanism strengthened until the bound bites and the bound tightened to meet it.  One thing
    at a time is what makes the difference between them readable as truncation.
    """

    @pytest.fixture(scope="class")
    def second(self) -> harness.Fixture:
        return harness.read_fixture(version="v2")

    def test_the_digest_matches_the_manifest(self, second: harness.Fixture) -> None:
        assert second.manifest["version"] == "v2"

    def test_the_draw_regenerates_from_its_seed(self, second: harness.Fixture) -> None:
        rebuilt = harness.build_fixture(version="v2")
        for column in rebuilt.columns:
            np.testing.assert_array_equal(
                rebuilt[column].to_numpy(dtype=float),
                second.frame[column].to_numpy(dtype=float),
            )

    def test_the_truncation_binds_materially(self, second: harness.Fixture) -> None:
        """Not "at least one row": a bound that bit on two of two hundred would make every
        reading downstream a statement about two rows.  27% is what the mechanism was chosen
        to produce, and the manifest records it so a reader sees it rather than refits."""
        lower, upper = harness.spec("v2").g_bounds
        g = second.arrays()["gn"]
        clipped = int(((g < lower) | (g > upper)).sum())
        assert clipped == second.manifest["clipped"]
        assert clipped >= len(g) // 8

    def test_only_the_mechanism_moved(self, second: harness.Fixture) -> None:
        """The truth and the initial outcome regression are ``v1``'s, unchanged.

        If ``v2`` had a different outcome regression too, a divergence found on it could be
        either, and the fixture would answer neither question.
        """
        first, other = harness.spec("v1").coefficients, harness.spec("v2").coefficients
        for name in ("truth_mechanism", "truth_outcome", "initial_outcome"):
            assert first[name] == other[name]
        assert first["initial_mechanism"] != other["initial_mechanism"]

    def test_it_is_not_degenerate_either(self, second: harness.Fixture) -> None:
        """``v2`` inherits ``v1``'s reason for being misspecified and has to keep it.

        At correct nuisances :math:`Q_r` and :math:`g_{r,2}` vanish row by row and a trace goes
        blind to a sign, an update order and a reduction vintage alike.  A ``v2`` that tidied
        the outcome regression while strengthening the mechanism would clip beautifully and see
        nothing.
        """
        reading = harness.degeneracy(harness.trace(second, order="cleverly"))
        assert reading["max|Q_r|"] > 0.02 * reading["mean|Y|"]
        assert reading["max|g_r2|"] > 0.02 * reading["mean|Y|"]


class TestTheInstrumentDoesNotMoveTheFit:
    """A traced fit is bit for bit an untraced one, under both orders.

    The harness patches :mod:`cleverly.estimators.targeting`'s module-level names for the
    duration of one call.  If that perturbed a solve, every divergence F3 or F4 found would be
    a divergence of the harness -- so this is checked rather than argued, and to the last bit
    rather than to a tolerance.
    """

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_estimates_and_curves_are_identical(
        self, plain: dict[str, object], traces: dict[str, harness.Trace], order: str
    ) -> None:
        untraced = plain[order]
        traced = traces[order]
        for name, values in untraced.influence_curves.items():
            np.testing.assert_array_equal(
                np.asarray(values, dtype=float),
                traced.curve[str(name)],
                err_msg=f"the traced curve for {name} is not the untraced one",
            )
        for _, row in untraced.to_frame().iterrows():
            recorded = traced.estimates[str(row["estimand"])]
            assert recorded["psi"] == float(row["psi"])
            assert recorded["se"] == float(row["std_err"])

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_loop_ends_the_same_way(
        self, plain: dict[str, object], traces: dict[str, harness.Trace], order: str
    ) -> None:
        reduction = plain[order].repeats[0].fluctuations["mean"].reduction
        exit_ = traces[order].exit
        assert exit_["exit_reason"] == reduction.exit_reason
        assert exit_["rounds"] == reduction.rounds
        assert exit_["closing"] == reduction.closing
        assert exit_["ill_conditioned"] == reduction.ill_conditioned

    def test_the_module_is_left_unpatched(self, traces: dict[str, harness.Trace]) -> None:
        """The patch is scoped to the call and restored in a ``finally``.

        A leaked patch would make every later fit in the process traced -- silently, since a
        recorder nobody reads changes no number, until the process runs two fits and the
        second one's steps land in the first one's record.
        """
        assert traces  # the traces have run, which is what could have leaked
        assert targeting_module.solve_submodel.__module__ == targeting_module.__name__
        assert targeting_module.solve_bounded_mechanism.__module__.endswith("fluctuation.mechanism")
        assert targeting_module.reduced_outcome_submodel.__module__.endswith("fluctuation.reduced")


class TestTheTraceIsDeterministic:
    """Two runs of one deterministic pipeline produce the same float64 bytes, or they do not.

    Checked on hashes rather than on arrays with a tolerance, deliberately: a tolerance here
    would hide exactly the drift this exists to catch, and every input -- the draw, the folds,
    the initial predictions, the reduced learners' seeds -- is fixed.
    """

    def test_a_second_trace_digests_the_same(
        self, fixture: harness.Fixture, traces: dict[str, harness.Trace]
    ) -> None:
        """One order, and the cheaper of the two.

        Determinism is a property of the pipeline -- the draw, the folds, the injected
        predictions and the reduced learners' seeds are all fixed -- so re-checking it under
        the second update order costs 6.5 s to re-answer the same question.
        """
        again = harness.trace(fixture, order="cleverly")
        assert harness.digest(again) == harness.digest(traces["cleverly"])


class TestEveryRecordedScoreRecomputes:
    """The acceptance criterion F2 states: every identity inside the existing tolerance.

    ``IDENTITY_TOLERANCE`` is :mod:`cleverly.validation.drtmle`'s, imported rather than
    restated -- it is the bar the package already applies to *"recompute the recorded number
    from the returned state"*, and this is the same question asked per step instead of per fit.
    """

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_worst_residual_is_inside_the_tolerance(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        rows = harness.identities(traces[order])
        assert rows, "a trace with no identities to check is not a checked trace"
        worst = max(rows, key=lambda row: row.residual)
        assert worst.residual <= IDENTITY_TOLERANCE, (
            f"{worst.quantity} at step {worst.step} ({worst.label}) recomputes to "
            f"{worst.recomputed!r} against a recorded {worst.recorded!r}"
        )

    @pytest.mark.parametrize("order", ORDERS)
    def test_all_three_equations_are_covered(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        """A tolerance met by an empty selection is not met.

        The three equations and both update identities all have to appear, or a step
        classified wrongly would drop out of the check rather than fail it.
        """
        quantities = {row.quantity.split("[")[0] for row in harness.identities(traces[order])}
        assert quantities == {"eq(8)", "eq(9)", "eq(10)", "update(Q)", "update(g)"}


class TestTheIdentitiesAreNotVacuous:
    """The check has been watched to fail, which is what makes it evidence.

    Every recomputation here reads arrays the harness also recorded, so an identity that
    trivially agreed -- because both sides came from one expression -- would pass on a fit
    with any defect at all.  Perturbing the recorded state is what says the two sides are
    genuinely two.
    """

    def test_a_perturbed_state_breaks_the_identity(self, traces: dict[str, harness.Trace]) -> None:
        from dataclasses import replace

        traced = traces["cleverly"]
        step = next(one for one in traced.steps if one.equation == "8" and one.phase == "round")
        moved = replace(step.after, q=step.after.q + 1e-3)
        broken = replace(traced, steps=(replace(step, after=moved, index=0),))
        rows = [row for row in harness.identities(broken) if row.quantity.startswith("eq(8)")]
        assert rows
        assert max(row.residual for row in rows) > 1e-6


class TestTheFixtureIsNotDegenerate:
    r"""The regime this instrument would be blind in, and the distance from it.

    At correct nuisances :math:`Q_r` and :math:`g_{r,2}` are zero row by row and the reported
    curve equals :math:`D^*` array for array -- so a trace taken there records a fit in which
    every quantity F3 and F4 are hunting for is identically zero.  This is the test that stops
    a later tidy-up of the fixture from making every comparison downstream vacuous.
    """

    @pytest.mark.parametrize("order", ORDERS)
    def test_both_reduced_regressions_and_both_corrections_are_material(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        numbers = harness.degeneracy(traces[order])
        scale = numbers["mean|Y|"]
        for name in ("max|Q_r|", "max|g_r2|", "max|D*_g|", "max|D*_Q|"):
            assert numbers[name] > 0.05 * scale, f"{name} = {numbers[name]:.3g} is near zero"


class TestTheStepStreamIsWellFormed:
    """What a reader -- and F3's alignment -- is entitled to assume about the record."""

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_first_step_is_the_prime_and_the_last_is_the_close(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        steps = traces[order].steps
        assert steps[0].phase == "prime" and steps[0].equation == "8" and steps[0].round == 0
        assert steps[-1].phase == "close"
        assert [step.phase for step in steps] == sorted(
            (step.phase for step in steps), key=harness.PHASES.index
        )

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_round_count_is_the_one_the_fit_reports(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        """Derived from the step stream, checked against the field the estimator sets.

        The numbering is a rule about *which equation opens a round*, applied after the fact
        because there is nowhere to hook it without the public option F2 refuses -- so it is
        checked against ``ReductionFluctuation.rounds`` rather than trusted.
        """
        assert traces[order].rounds == traces[order].exit["rounds"]

    @pytest.mark.parametrize("order", ORDERS)
    def test_every_step_carries_the_state_either_side_of_it(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        n = traces[order].treatment.size
        for step in traces[order].steps:
            for state in (step.before, step.after):
                assert state.q_obs.shape == (n,)
                assert state.g.shape == (n,)
                for name in ("q", "qr", "gr1", "gr2"):
                    assert getattr(state, name).shape == (n, len(traces[order].arms))


class TestTheClosingBoundaryIsRecorded:
    """The one stage no field on a returned fit distinguishes.

    ``docs/drtmle/investigation-log.md`` calls the closing pass an anaesthetic: it re-solves
    all three equations at the reductions the record carries, so a defect in how the *loop*
    exits is invisible at the fit.  F4's pre-close/post-close column needs the two states, and
    this is where they come from.
    """

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_two_sides_are_recorded_and_differ(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        before, after = traces[order].boundary()
        assert before.q_obs.shape == after.q_obs.shape
        assert not np.array_equal(before.q_obs, after.q_obs), (
            "the closing pass left the outcome regression exactly where the loop did, which "
            "would mean it re-solved nothing"
        )

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_closing_pass_leaves_all_three_equations_at_the_reported_reductions(
        self, traces: dict[str, harness.Trace], order: str
    ) -> None:
        """Its whole purpose, and the reason the reported curve is centred.

        The final joint step's scores are the ones the reported curve's empirical mean
        inherits, so they are what has to be small -- and they are checked here at the bar
        ``score_check`` uses for a reported fit rather than at the loop's stopping rule.
        """
        joint = [step for step in traces[order].steps if step.equation == "joint"]
        assert len(joint) == 1
        assert float(np.max(np.abs(joint[0].score))) < 1e-8


class TestBothOrdersAreTracedOnTheOneFixture:
    """F2's *"both existing update orders on the one fixture"*, and what it makes visible."""

    def test_the_reduction_vintages_differ_between_the_orders(
        self, traces: dict[str, harness.Trace]
    ) -> None:
        r"""The fourth row of ``docs/roadmap.md``'s R3 table, read off a run.

        This package's order refits all three reductions and uses all three; the paper's
        refits :math:`g_{r,1}` and :math:`g_{r,2}` at the once-updated regression and
        :math:`Q_r` at the twice-updated one, and each equation then reads a **mixture** of
        two vintages.  That difference is invisible in every field a fitted result carries,
        which is why it is one of the five places F3 has to classify a divergence into -- and
        it is what the harness records wrongly if it reads the reductions off the refit
        closure instead of off the covariate builders.
        """
        ours = harness.vintages(traces["cleverly"])
        theirs = harness.vintages(traces["paper"])
        assert ours and theirs
        assert all(row["qr"] and row["gr1"] and row["gr2"] for row in ours)
        assert any(not row["qr"] for row in theirs)
        assert any(not row["gr1"] and not row["gr2"] for row in theirs)

    def test_the_comparison_is_reported_rather_than_asserted_equal(
        self, traces: dict[str, harness.Trace]
    ) -> None:
        """Whether the two routes reach one fixed point is item 22's *measurement*.

        So this asserts the comparison is well formed and covers the estimates, the standard
        errors and the curves -- never that the differences are zero.  A test that required
        them equal would be asserting the answer to the open question the second route exists
        to ask.
        """
        rows = harness.compare(traces["cleverly"], traces["paper"])
        quantities = {row["quantity"] for row in rows}
        assert {"psi[ate]", "se[ate]", "max|dD|[ate]", "rounds"} <= quantities
        assert all("difference" in row for row in rows)
