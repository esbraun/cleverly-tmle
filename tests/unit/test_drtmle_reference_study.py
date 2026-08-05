r"""E2's gate harness: the block roles, the pairing, and the two gate statistics.

``benchmarks/drtmle_reference_study.py`` runs the reference against ``glm`` and runs the two
gates that have to be read before that comparison is.  Everything below is arithmetic on
constructed rows or structure on a companion with no fit behind it, because the harness's own
fits are a dispatch rather than a test -- and because the claims that can go wrong here are
about **which rows a number was taken over**, which is exactly what a constructed record can
say and a real one cannot.

Three shapes of mistake this module is written against, each of which produces a plausible
number rather than an error:

* a block used in the wrong role -- scoring a candidate on rows it was fitted on, or fitting
  the reference on the block the remainder is integrated over.  Both make an error look small
  by sharing a randomisation with the thing it is supposed to be independent of;
* a paired difference that is not paired -- averaging the two arms apart, or letting a gate-C
  budget draw contribute its extra scrambles to the comparison and so weigh several times an
  ordinary draw;
* a gate C spread taken *across* draws rather than *within* them, which measures the estimator
  and reports it as the reference's error.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_injection
from benchmarks import drtmle_reference_study as study


def fit_row(**overrides: Any) -> study.FitRow:
    """One :class:`~benchmarks.drtmle_reference_study.FitRow` with everything else declared."""
    fields: dict[str, Any] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "fold_seed": 2,
        "estimator": "glm",
        "scramble": 0,
        "estimand": "ate",
        "psi": 1.5,
        "truth": 1.5,
        "p0_curve": 0.0,
        "pn_curve": 0.0,
        "remaining": 0.0,
        "root_n_remaining": 1.0,
        "companion_replicate_se": 0.01,
        "companion_rows": 40_960,
        "rounds": 3,
        "exit_reason": "tolerance",
        "valid": True,
        "seconds": 10.0,
    }
    fields.update(overrides)
    return study.FitRow(**fields)


def risk_row(**overrides: Any) -> study.RiskRow:
    fields: dict[str, Any] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "candidate": study.REFERENCE.label,
        "reduction": "qr",
        "treatment_arm": 1.0,
        "fold": 0,
        "risk": 1e-3,
        "fitted_rows": 4_096,
        "scored_rows": 8_192,
    }
    fields.update(overrides)
    return study.RiskRow(**fields)


class TestTheBlocksPlayThreeRolesAndDoNotShareOne:
    """The companion's layout, which is what every independence claim here rests on."""

    @pytest.fixture(scope="class")
    def place(self) -> Any:
        # Small point counts: this fixture builds a companion and fits nothing, so what is
        # being checked is the layout rather than any integral taken over it.
        payload = study.Payload(
            cell="q-drift",
            n=600,
            data_seed=11,
            fold_seed=12,
            reference_points=64,
            scoring_points=128,
            evaluation_points=32,
            evaluation_scrambles=2,
            reference_scrambles=3,
        )
        return study.layout(payload, drtmle_injection.base_law())

    def test_every_block_has_its_own_rows(self, place: Any) -> None:
        """Contiguous, non-overlapping, and covering the companion exactly.

        A window that overlapped another's rows would make a held-out risk a fitted risk, or
        put the reference's own randomisation inside the integral its error is supposed to be
        independent of -- and neither would raise.
        """
        blocks = [*place.reference, place.scoring, *place.evaluation]
        edges = [(block.window.start, block.window.stop) for block in blocks]

        assert edges == sorted(edges)
        assert edges[0][0] == 0
        assert edges[-1][1] == place.stack.weights.size
        for (_, stop), (start, _) in pairwise(edges):
            assert stop == start

    def test_the_scoring_block_is_finer_than_the_block_it_scores(self, place: Any) -> None:
        """Declared in the module and checked here, since it is a property of the *roles*.

        A held-out risk carries its own Monte Carlo error and nothing pairs it away -- unlike
        the evaluation rule's, which both arms of the comparison share.
        """
        assert place.scoring.points > place.reference[0].points

    def test_the_three_streams_are_disjoint(self, place: Any) -> None:
        """A shared scramble would make two blocks the same randomisation under two names.

        The reference's is the one that matters most: its error propagates into the fit
        deterministically, so sharing it with the evaluation block would make the fit and the
        integral one random variable with a covariance nobody can sign.
        """
        seeds = [block.seed for block in (*place.reference, place.scoring, *place.evaluation)]

        assert len(set(seeds)) == len(seeds)

    def test_the_reference_scrambles_are_replicates_of_one_resolution(self, place: Any) -> None:
        """Gate C varies the randomisation and nothing else, or it would measure refinement."""
        assert len({block.points for block in place.reference}) == 1
        assert len(place.reference) == 3


class TestThePairingIsPerDraw:
    """The comparison is a paired difference, and these are the ways it stops being one."""

    def test_a_draw_missing_one_arm_is_dropped_rather_than_averaged(self) -> None:
        """An unpaired draw would contribute one arm's level to a column of differences."""
        rows = [
            fit_row(data_seed=1, estimator="glm", root_n_remaining=1.0),
            fit_row(data_seed=1, estimator="reference", scramble=99, root_n_remaining=0.4),
            fit_row(data_seed=2, estimator="glm", root_n_remaining=5.0),
        ]

        differences = study.paired_differences(rows, "q-drift", 600, "ate")

        np.testing.assert_allclose(differences, [-0.6])

    def test_a_budget_draw_contributes_one_difference_and_not_several(self) -> None:
        """Gate C's extra scrambles are a *budget* measurement, not extra comparison draws.

        Folding them in would weigh a budget draw several times an ordinary one -- and would
        do it silently, since every one of those rows is a valid reading of the column.
        """
        rows = [
            fit_row(data_seed=1, estimator="glm", root_n_remaining=1.0),
            *[
                fit_row(
                    data_seed=1,
                    estimator="reference",
                    scramble=seed,
                    root_n_remaining=value,
                )
                for seed, value in ((90, 0.4), (91, 0.9), (92, 0.2))
            ],
        ]

        differences = study.paired_differences(rows, "q-drift", 600, "ate")

        assert differences.size == 1
        # The lowest scramble is the ordinary draw's own block, so the comparison reads the
        # arm every draw has rather than one only the budget draws carry.
        np.testing.assert_allclose(differences, [-0.6])

    def test_a_failed_fit_takes_its_draw_out_of_the_pair(self) -> None:
        rows = [
            fit_row(data_seed=1, estimator="glm", root_n_remaining=1.0),
            fit_row(
                data_seed=1,
                estimator="reference",
                scramble=90,
                root_n_remaining=float("nan"),
                error="ValueError",
            ),
        ]

        assert study.paired_differences(rows, "q-drift", 600, "ate").size == 0


class TestGateCIsAWithinDrawSpread:
    """The reference's own error is conditional on a fit, and that is the whole statistic."""

    def test_the_estimators_own_spread_does_not_reach_the_column(self) -> None:
        """Two draws far apart, each with a tight across-scramble spread.

        A spread taken across all the reference rows at once would read the gap between the
        two draws -- the estimator's own sampling variation -- and report it as the
        reference's randomisation error.  That is E1's mistake in its other form.
        """
        rows = [
            fit_row(data_seed=seed, estimator="reference", scramble=scramble, root_n_remaining=v)
            for seed, values in ((1, (10.0, 10.1, 9.9)), (2, (-10.0, -10.1, -9.9)))
            for scramble, v in enumerate(values)
        ]

        spread, draws = study.budget_spread(rows, "q-drift", 600, "ate")

        assert draws == 2
        assert spread == pytest.approx(0.1, abs=1e-9)

    def test_a_draw_with_one_scramble_is_not_a_spread_of_zero(self) -> None:
        """``nan`` rather than zero: one replicate has no spread, and zero would read as a
        rule with no error rather than as an error nobody measured."""
        rows = [fit_row(estimator="reference", scramble=90, root_n_remaining=1.0)]

        spread, draws = study.budget_spread(rows, "q-drift", 600, "ate")

        assert draws == 0
        assert np.isnan(spread)

    def test_the_glm_arm_is_not_in_it(self) -> None:
        """It has no reference block, so its rows are not replicates of anything here."""
        rows = [
            fit_row(data_seed=1, estimator="reference", scramble=90, root_n_remaining=1.0),
            fit_row(data_seed=1, estimator="reference", scramble=91, root_n_remaining=1.2),
            fit_row(data_seed=1, estimator="glm", root_n_remaining=-40.0),
        ]

        spread, _ = study.budget_spread(rows, "q-drift", 600, "ate")

        assert spread == pytest.approx(0.2 / np.sqrt(2), abs=1e-9)


class TestGateBIsADifferenceOfRisksAndIsPaired:
    """Why a difference and why per reduction, both of which a ratio or a mean would lose."""

    def test_each_candidate_is_paired_against_the_reference_on_the_same_draw(self) -> None:
        rows = [
            risk_row(data_seed=seed, candidate=label, risk=value)
            for seed, base, worse in ((1, 1e-3, 3e-3), (2, 5e-3, 6e-3))
            for label, value in ((study.REFERENCE.label, base), ("bins(8)", worse))
        ]

        gaps = study.risk_gaps(rows, "q-drift", 600, "qr")

        np.testing.assert_allclose(gaps["bins(8)"], [2e-3, 1e-3])
        assert study.REFERENCE.label not in gaps

    def test_arms_and_folds_are_averaged_inside_a_draw(self) -> None:
        """They are the same regression problem at different splits, not replicates of a number.

        Averaging them *before* the difference is what keeps a draw one paired observation; a
        difference taken row by row would give a draw ``2 * n_folds`` of them and an interval
        far too narrow.
        """
        rows = [
            risk_row(candidate=label, treatment_arm=arm, fold=fold, risk=value)
            for label, value in ((study.REFERENCE.label, 1e-3), ("bins(8)", 2e-3))
            for arm in (1.0, 0.0)
            for fold in range(5)
        ]

        gaps = study.risk_gaps(rows, "q-drift", 600, "qr")

        assert gaps["bins(8)"].size == 1
        np.testing.assert_allclose(gaps["bins(8)"], [1e-3])

    def test_the_reductions_are_kept_apart(self) -> None:
        """Their targets are on different scales -- a residual and an indicator -- so a mean
        over them is a number with no units and a gate on it is a gate on whichever is larger."""
        rows = [
            risk_row(candidate=label, reduction=name, risk=value)
            for name, base, worse in (("qr", 1e-4, 2e-4), ("gr1", 0.2, 0.5))
            for label, value in ((study.REFERENCE.label, base), ("bins(8)", worse))
        ]

        np.testing.assert_allclose(study.risk_gaps(rows, "q-drift", 600, "qr")["bins(8)"], [1e-4])
        np.testing.assert_allclose(study.risk_gaps(rows, "q-drift", 600, "gr1")["bins(8)"], [0.3])

    def test_a_refused_candidate_is_a_gap_rather_than_a_zero(self) -> None:
        """A rung whose points-per-parameter budget the block does not meet is recorded with
        its refusal, and must not enter the gate as a risk of ``nan`` turned into agreement."""
        rows = [
            risk_row(candidate=study.REFERENCE.label, risk=1e-3),
            risk_row(candidate="spline(32)", risk=float("nan"), error="ValueError"),
        ]

        assert "spline(32)" not in study.risk_gaps(rows, "q-drift", 600, "qr")


def _passing_gate(**overrides: Any) -> list[study.RiskRow]:
    """Gate rows that pass both of B's clauses: the control rejected, no rung better.

    Built from a helper rather than inline in each test, because what makes a gate *pass* is
    two clauses rather than one and a test that spelled only the first would be exercising a
    gate the module does not have.
    """
    rows: list[study.RiskRow] = []
    for seed in range(8):
        for name in study.REDUCTIONS:
            rows.append(risk_row(data_seed=seed, reduction=name, risk=1e-3, **overrides))
            for rung in study.RUNGS:
                if rung.label == study.REFERENCE.label:
                    continue
                rows.append(
                    risk_row(
                        data_seed=seed,
                        reduction=name,
                        candidate=rung.label,
                        risk=1e-3 + 1e-5 * (seed % 3),
                        **overrides,
                    )
                )
            rows.append(
                risk_row(
                    data_seed=seed,
                    reduction=name,
                    candidate=study.NEGATIVE_CONTROL.label,
                    risk=5e-3 + 1e-5 * (seed % 3),
                    **overrides,
                )
            )
    return rows


def _arms(differences: Sequence[float], *, level: float, scrambles: int) -> list[study.FitRow]:
    """One paired draw per difference, plus a gate-C budget on the first few.

    ``level`` is the ``glm`` arm's own column, which is what the margin is a quarter of, and
    the budget draws carry a spread the caller controls through ``scrambles``.
    """
    rows: list[study.FitRow] = []
    for seed, difference in enumerate(differences):
        rows.append(fit_row(data_seed=seed, estimator="glm", root_n_remaining=level))
        for index in range(scrambles if seed < 3 else 1):
            rows.append(
                fit_row(
                    data_seed=seed,
                    estimator="reference",
                    scramble=90 + index,
                    root_n_remaining=level + difference + 1e-4 * index,
                )
            )
    return rows


class TestTheFrozenRuleHasThreeVerdictsAndReachesEachOne:
    """``unresolved`` is a **third** verdict, and a rule that cannot reach it has two.

    Every case here is constructed to sit in one region of the frozen rule, so what is being
    checked is the rule rather than any measurement: a margin of a quarter of the ``glm`` arm's
    own level, a paired bootstrap interval, and the requirement that the interval lie *wholly*
    on one side of the band.
    """

    def test_a_difference_far_outside_the_band_has_moved(self) -> None:
        fits = _arms([-0.9, -0.95, -1.0, -0.92, -0.98, -0.9, -1.05, -0.94], level=2.0, scrambles=3)

        assert study.comparison_verdict(fits, _passing_gate(), "q-drift", 600, "ate") == "moved"

    def test_a_difference_wholly_inside_the_band_is_equivalent(self) -> None:
        fits = _arms([0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.015, 0.005], level=2.0, scrambles=3)

        assert (
            study.comparison_verdict(fits, _passing_gate(), "q-drift", 600, "ate") == "equivalent"
        )

    def test_an_interval_straddling_the_band_is_unresolved(self) -> None:
        """Not a weak ``equivalent``: the run cannot tell the two apart at this precision."""
        fits = _arms([-0.9, 0.8, -0.7, 0.6, -0.5, 0.9, -0.8, 0.4], level=2.0, scrambles=3)

        assert (
            study.comparison_verdict(fits, _passing_gate(), "q-drift", 600, "ate") == "unresolved"
        )

    def test_a_movement_in_either_direction_counts(self) -> None:
        """A reference that makes the remainder *larger* is still a learner effect.

        E2 asks whether item 13's failure is a learner failure at all, and a rule that only
        counted improvement would report the awkward half of that answer as a null result.
        """
        fits = _arms([0.9, 0.95, 1.0, 0.92, 0.98, 0.9, 1.05, 0.94], level=2.0, scrambles=3)

        assert study.comparison_verdict(fits, _passing_gate(), "q-drift", 600, "ate") == "moved"


class TestAFailedGateMakesItsWholeCellUnresolved:
    """The gates are what say the comparison is about the reduction learner at all."""

    def test_a_negative_control_that_is_not_rejected_fails_gate_b(self) -> None:
        """Without teeth the gate cannot discriminate, whatever the comparison then shows."""
        rows = [
            risk_row(data_seed=seed, reduction=name, candidate=label, risk=1e-3)
            for seed in range(8)
            for name in study.REDUCTIONS
            for label in (study.REFERENCE.label, study.NEGATIVE_CONTROL.label)
        ]
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)

        verdict, reason = study.gate_verdict(fits, rows, "q-drift", 600)

        assert verdict == "fail"
        assert "not rejected" in reason
        assert study.comparison_verdict(fits, rows, "q-drift", 600, "ate") == "unresolved"

    def test_a_rung_that_beats_the_reference_fails_gate_b(self) -> None:
        """The comparison would then answer for a reference another resolution is better than.

        The repair is to move the shipped rung *before* a number is read, which is why this is
        a gate and not a footnote on the comparison.
        """
        rows = _passing_gate()
        better = study.RUNGS[-1].label
        rows = [row for row in rows if row.candidate != better]
        rows += [
            risk_row(data_seed=seed, reduction=name, candidate=better, risk=1e-4)
            for seed in range(8)
            for name in study.REDUCTIONS
        ]
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)

        verdict, reason = study.gate_verdict(fits, rows, "q-drift", 600)

        assert verdict == "fail"
        assert "beats the reference" in reason

    def test_a_reference_noisier_than_its_budget_fails_gate_c(self) -> None:
        """The band is a quarter of the level and the budget is a third of the band.

        A reference whose own scramble-to-scramble spread is that large decides the verdict
        itself, which is the failure this clause exists against.
        """
        level, difference = 2.0, -0.9
        rows = [
            fit_row(data_seed=seed, estimator="glm", root_n_remaining=level) for seed in range(8)
        ]
        for seed in range(8):
            for index in range(3 if seed < 3 else 1):
                rows.append(
                    fit_row(
                        data_seed=seed,
                        estimator="reference",
                        scramble=90 + index,
                        # A spread far past a third of the 0.5 margin.
                        root_n_remaining=level + difference + 0.6 * index,
                    )
                )

        verdict, reason = study.gate_verdict(rows, _passing_gate(), "q-drift", 600)

        assert verdict == "fail"
        assert reason.startswith("C:") or "; C:" in reason
        assert (
            study.comparison_verdict(rows, _passing_gate(), "q-drift", 600, "ate") == "unresolved"
        )

    def test_a_run_with_no_budget_draw_fails_gate_c(self) -> None:
        """Unmeasured and small must not read alike, which is E1's lesson in its other form."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=1)

        verdict, reason = study.gate_verdict(fits, _passing_gate(), "q-drift", 600)

        assert verdict == "fail"
        assert "no budget draw" in reason


class TestTheMarginIsAQuarterOfTheControlArmsLevel:
    """The frozen fraction, and which rows the level it scales is taken over."""

    def test_it_is_the_declared_fraction_of_the_glm_level(self) -> None:
        fits = _arms([-0.5] * 6, level=2.0, scrambles=1)

        margin = study.equivalence_margin(fits, "q-drift", 600, "ate")

        assert margin == pytest.approx(study.EQUIVALENCE_FRACTION * 2.0)

    def test_it_is_taken_over_the_paired_draws_only(self) -> None:
        """A draw the reference arm failed on would otherwise move the band the comparison
        is judged against, while contributing nothing to the comparison itself."""
        fits = _arms([-0.5] * 4, level=2.0, scrambles=1)
        fits += [fit_row(data_seed=99, estimator="glm", root_n_remaining=100.0)]

        assert study.equivalence_margin(fits, "q-drift", 600, "ate") == pytest.approx(
            study.EQUIVALENCE_FRACTION * 2.0
        )


class TestTheTablesAreWhatTheirHeadersSay:
    """The width pin every harness on this page carries, and for the same reason."""

    def test_every_row_matches_its_headers(self) -> None:
        fits = [
            fit_row(data_seed=seed, estimator=arm, scramble=scramble, root_n_remaining=value)
            for seed in (1, 2, 3)
            for arm, scramble, value in (("glm", 0, 1.2), ("reference", 90, 0.5))
        ]
        fits += [
            fit_row(data_seed=1, estimator="reference", scramble=91, root_n_remaining=0.6),
        ]
        risks = [
            risk_row(data_seed=seed, candidate=label, reduction=name, risk=value)
            for seed in (1, 2, 3)
            for name in study.REDUCTIONS
            for label, value in ((study.REFERENCE.label, 1e-3), ("bins(8)", 2e-3))
        ]

        for built, headers in (
            (study.gate_rows(fits, risks), study.GATE_HEADERS),
            (study.comparison_rows(fits, risks), study.COMPARISON_HEADERS),
            (study.cost_rows(fits), study.COST_HEADERS),
        ):
            assert built
            assert all(len(row) == len(headers) for row in built), headers

    def test_the_cost_table_prices_a_fit_and_not_a_draw(self) -> None:
        """A budget draw buys several reference fits, so a table counting draws understates it."""
        fits = [
            fit_row(data_seed=1, estimator="glm", seconds=8.0),
            fit_row(data_seed=1, estimator="reference", scramble=90, seconds=12.0),
            fit_row(data_seed=1, estimator="reference", scramble=91, seconds=12.0),
        ]

        cells = dict(zip(study.COST_HEADERS, study.cost_rows(fits)[0], strict=True))

        assert cells["fits"] == "3"
