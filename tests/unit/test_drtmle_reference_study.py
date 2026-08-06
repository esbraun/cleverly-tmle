r"""E2R's gate harness: the block roles, the selection, the pairing, and the gate statistics.

``benchmarks/drtmle_reference_study.py`` selects the rung each reduced regression is fitted at,
runs the reference against ``glm``, and runs the gates that have to be read before that
comparison is.  Everything below is arithmetic on constructed rows or structure on a companion
with no fit behind it, because the harness's own fits are a dispatch rather than a test -- and
because the claims that can go wrong here are about **which rows a number was taken over**,
which is exactly what a constructed record can say and a real one cannot.

Four shapes of mistake this module is written against, each of which produces a plausible
number rather than an error:

* a block used in the wrong role -- scoring a candidate on rows it was fitted on, or fitting
  the reference on the block the remainder is integrated over.  Both make an error look small
  by sharing a randomisation with the thing it is supposed to be independent of;
* **a rung certified by the block that chose it**, which is E2R's own addition and is the
  reason the selection and the audit are two blocks and two phases.  A gate read on the
  selection block would pass by construction: the rung was chosen there because it won there;
* a paired difference that is not paired -- averaging the two arms apart, or letting a gate-C
  budget draw contribute its extra scrambles to the comparison and so weigh several times an
  ordinary draw;
* a gate C spread taken *across* draws rather than *within* them, which measures the estimator
  and reports it as the reference's error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_injection
from benchmarks import drtmle_reference_study as study


def fit_row(**overrides: Any) -> study.FitRow:
    """One :class:`~benchmarks.drtmle_reference_study.FitRow` with everything else declared."""
    fields: dict[str, Any] = {
        "cohort": "decision",
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


#: The rung most constructed rows below are keyed against, and the one a cell is taken to have
#: selected unless a test says otherwise.  Named here so a test reads as a statement about the
#: gate rather than about which rung happens to be in the middle of the ladder.
SELECTED = study.RUNGS[1].label


def risk_row(**overrides: Any) -> study.RiskRow:
    fields: dict[str, Any] = {
        "cohort": "decision",
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "phase": "audit",
        "candidate": SELECTED,
        "metric": "qr",
        "reduction": "qr",
        "treatment_arm": 1.0,
        "fold": 0,
        "risk": 1e-3,
        "fitted_rows": 4_096,
        "scored_rows": 8_192,
    }
    fields.update(overrides)
    return study.RiskRow(**fields)


def chosen(label: str = SELECTED) -> dict[tuple[str, int], dict[str, str]]:
    """One cell's selection, as :func:`~benchmarks.drtmle_reference_study.gate_rows` takes it."""
    return {("q-drift", 600): dict.fromkeys(study.REDUCTIONS, label)}


def payload(**overrides: Any) -> study.Payload:
    """One draw's declaration, with small point counts and everything else spelled out."""
    fields: dict[str, Any] = {
        "cohort": "decision",
        "cell": "q-drift",
        "n": 600,
        "data_seed": 11,
        "fold_seed": 12,
        "reference_points": 64,
        "selection_points": 128,
        "audit_points": 128,
        "evaluation_points": 32,
        "evaluation_scrambles": 2,
        "reference_scrambles": 3,
    }
    fields.update(overrides)
    return study.Payload(**fields)


class TestTheBlocksPlayFourRolesAndDoNotShareOne:
    """The companion's layout, which is what every independence claim here rests on."""

    @pytest.fixture(scope="class")
    def place(self) -> Any:
        # Small point counts: this fixture builds a companion and fits nothing, so what is
        # being checked is the layout rather than any integral taken over it.
        return study.layout(payload(), drtmle_injection.base_law())

    def test_every_block_has_its_own_rows(self, place: Any) -> None:
        """Contiguous, non-overlapping, and covering the companion exactly.

        A window that overlapped another's rows would make a held-out risk a fitted risk, or
        put the reference's own randomisation inside the integral its error is supposed to be
        independent of -- and neither would raise.
        """
        blocks = [*place.reference, place.selection, place.audit, *place.evaluation]
        edges = [(block.window.start, block.window.stop) for block in blocks]

        assert edges == sorted(edges)
        assert edges[0][0] == 0
        assert edges[-1][1] == place.stack.weights.size
        for (_, stop), (start, _) in pairwise(edges):
            assert stop == start

    def test_the_scored_blocks_are_finer_than_the_block_they_score(self, place: Any) -> None:
        """Declared in the module and checked here, since it is a property of the *roles*.

        A held-out risk carries its own Monte Carlo error and nothing pairs it away -- unlike
        the evaluation rule's, which both arms of the comparison share.
        """
        assert place.selection.points > place.reference[0].points
        assert place.audit.points > place.reference[0].points

    def test_the_four_streams_are_disjoint(self, place: Any) -> None:
        """A shared scramble would make two blocks the same randomisation under two names.

        Two of the pairs matter for their own reason.  The **selection and the audit** must not
        share one, or the block that chose the rung is the block that certifies it -- E2R's
        whole first clause.  And the **reference and the evaluation** must not, for a stronger
        reason still: the reference's error propagates into the fit deterministically, so
        sharing a scramble there would make the fit and the integral one random variable with a
        covariance nobody can sign.
        """
        blocks = (*place.reference, place.selection, place.audit, *place.evaluation)
        seeds = [block.seed for block in blocks]

        assert len(set(seeds)) == len(seeds)
        assert place.selection.seed != place.audit.seed

    def test_the_reference_scrambles_are_replicates_of_one_resolution(self, place: Any) -> None:
        """Gate C varies the randomisation and nothing else, or it would measure refinement."""
        assert len({block.points for block in place.reference}) == 1
        assert len(place.reference) == 3

    def test_both_passes_of_a_draw_see_the_same_rows(self) -> None:
        """The layout is a function of the payload, and the selection is not part of it.

        Pass two fits at the rungs pass one selected, so its payload differs in ``rungs`` and in
        nothing else.  If that field reached the layout, the two arms of the comparison would be
        integrated over different companions and the pairing that removes the evaluation rule's
        error -- most of each level, by E1b's measurement -- would remove nothing.
        """
        law = drtmle_injection.base_law()
        first = study.layout(payload(), law)
        second = study.layout(payload(rungs=(("qr", 8), ("gr1", 32), ("gr2", 16))), law)

        np.testing.assert_array_equal(first.stack.weights, second.stack.weights)
        assert [block.seed for block in first.stack.blocks] == [
            block.seed for block in second.stack.blocks
        ]


class TestASelectionIsARungOrItIsNothing:
    """What the decision cohort is handed, and what it does where nothing was chosen."""

    def test_the_selected_knots_are_routed_per_regression(self) -> None:
        rungs = payload(rungs=(("qr", 8), ("gr1", 32), ("gr2", 16))).references()

        assert [rungs[name].label for name in ("qr", "gr1", "gr2")] == [
            "spline(8)",
            "spline(32)",
            "spline(16)",
        ]

    def test_an_unselected_regression_refuses_to_be_fitted(self) -> None:
        """The fallback this replaced was defended as *visible* -- E2's own shipped rung, named
        in the record -- and it was visible in the record and not in the verdict.  A lost
        selection row produced a reference nobody chose, an audit that could still be scored
        against it, and a cell that printed ``moved`` like any other."""
        with pytest.raises(ValueError, match="no selected rung"):
            payload(rungs=(("qr", 8),)).references()

    def test_the_debug_switch_is_the_only_way_back_to_the_fallback(self) -> None:
        """``--allow-fallback`` exists for a run too thin to rank, and no dispatch passes it."""
        rungs = payload(rungs=(("qr", 8),), allow_fallback=True).references()

        assert rungs["qr"].label == "spline(8)"
        assert rungs["gr1"].label == study.FALLBACK_RUNG.label
        assert rungs["gr2"].label == study.FALLBACK_RUNG.label


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
    """Why a difference and why per metric, both of which a ratio or a mean would lose."""

    def test_each_candidate_is_paired_against_the_selected_rung_on_the_same_draw(self) -> None:
        rows = [
            risk_row(data_seed=seed, candidate=label, risk=value)
            for seed, base, worse in ((1, 1e-3, 3e-3), (2, 5e-3, 6e-3))
            for label, value in ((SELECTED, base), ("bins(8)", worse))
        ]

        gaps = study.risk_gaps(rows, "q-drift", 600, "qr", phase="audit", baseline=SELECTED)

        np.testing.assert_allclose(gaps["bins(8)"], [2e-3, 1e-3])
        assert SELECTED not in gaps

    def test_arms_and_folds_are_averaged_inside_a_draw(self) -> None:
        """They are the same regression problem at different splits, not replicates of a number.

        Averaging them *before* the difference is what keeps a draw one paired observation; a
        difference taken row by row would give a draw ``2 * n_folds`` of them and an interval
        far too narrow.
        """
        rows = [
            risk_row(candidate=label, treatment_arm=arm, fold=fold, risk=value)
            for label, value in ((SELECTED, 1e-3), ("bins(8)", 2e-3))
            for arm in (1.0, 0.0)
            for fold in range(5)
        ]

        gaps = study.risk_gaps(rows, "q-drift", 600, "qr", phase="audit", baseline=SELECTED)

        assert gaps["bins(8)"].size == 1
        np.testing.assert_allclose(gaps["bins(8)"], [1e-3])

    def test_the_metrics_are_kept_apart(self) -> None:
        """Their targets are on different scales -- a residual, an indicator, and either of
        those divided by a mechanism -- so a mean over them is a number with no units and a gate
        on it is a gate on whichever is largest."""
        rows = [
            risk_row(candidate=label, metric=metric, reduction=reduction, risk=value)
            for metric, reduction, base, worse in (
                ("qr", "qr", 1e-4, 2e-4),
                ("gr1", "gr1", 0.2, 0.5),
                ("h3", "qr", 4e-3, 9e-3),
            )
            for label, value in ((SELECTED, base), ("bins(8)", worse))
        ]

        def gap(metric: str) -> np.ndarray:
            return study.risk_gaps(rows, "q-drift", 600, metric, phase="audit", baseline=SELECTED)[
                "bins(8)"
            ]

        np.testing.assert_allclose(gap("qr"), [1e-4])
        np.testing.assert_allclose(gap("gr1"), [0.3])
        np.testing.assert_allclose(gap("h3"), [5e-3])

    def test_the_two_phases_are_kept_apart(self) -> None:
        """The selection block's rows and the audit block's are the same arithmetic on
        different rows, and a gate that pooled them would be certified in part by the block
        that chose."""
        rows = [
            risk_row(phase=phase, candidate=label, risk=value)
            for phase, base, worse in (("select", 1e-3, 5e-3), ("audit", 1e-3, 2e-3))
            for label, value in ((SELECTED, base), ("bins(8)", worse))
        ]

        audit = study.risk_gaps(rows, "q-drift", 600, "qr", phase="audit", baseline=SELECTED)
        select = study.risk_gaps(rows, "q-drift", 600, "qr", phase="select", baseline=SELECTED)

        np.testing.assert_allclose(audit["bins(8)"], [1e-3])
        np.testing.assert_allclose(select["bins(8)"], [4e-3])

    def test_a_refused_candidate_is_a_gap_rather_than_a_zero(self) -> None:
        """A rung whose points-per-parameter budget the block does not meet is recorded with
        its refusal, and must not enter the gate as a risk of ``nan`` turned into agreement."""
        rows = [
            risk_row(candidate=SELECTED, risk=1e-3),
            risk_row(candidate="spline(32)", risk=float("nan"), error="ValueError"),
        ]

        gaps = study.risk_gaps(rows, "q-drift", 600, "qr", phase="audit", baseline=SELECTED)

        assert "spline(32)" not in gaps


def draws_of(values: Sequence[float]) -> dict[int, float]:
    """One candidate's per-draw risks on one metric, as :func:`select_rung` takes them."""
    return dict(enumerate(values))


class TestTheRungIsSelectedOnTheGatesOwnStatistic:
    """E2R's selection rule, and the reason it is not the minimax it first was.

    Gate B's second clause is *no other rung may be strictly better on any metric*, read as a
    bootstrap interval over draws.  So the rule is admissibility against that same statistic on
    the block that chose: a rung is selected when no other rung's interval lies wholly below zero.
    A minimax on the mean risks -- the rule this replaced -- is judged by a quantity the gate does
    not use, and a six-draw pilot showed it buying a *resolved* small loss with an *unresolved*
    larger gain and being rejected for it.
    """

    def test_a_resolved_small_loss_is_not_traded_for_an_unresolved_larger_gain(self) -> None:
        """The pilot's own failure, in constructed rows: on ``qr`` the coarser rung is better by a
        tiny but consistent margin, and on ``h3`` it looks far worse with a spread that swamps
        the difference.  A minimax on the means selects the finer rung, and gate B rejects it."""
        risks = {
            # A consistent 2e-06: every draw the same way, so the interval clears zero.
            "qr": {
                "spline(8)": draws_of([1.0e-3 - 2e-06 * (1 + 0.01 * i) for i in range(8)]),
                "spline(32)": draws_of([1.0e-3 + 1e-09 * i for i in range(8)]),
            },
            # A 1e-05 apparent gain for the finer rung, swamped by an alternating 1e-04 spread.
            "h3": {
                "spline(8)": draws_of([1.0e-3 + 1e-05 + 1e-04 * (-1) ** i for i in range(8)]),
                "spline(32)": draws_of([1.0e-3 - 1e-04 * (-1) ** i for i in range(8)]),
            },
        }

        assert study.select_rung(risks) == "spline(8)"
        assert study.beaten_by(risks, "spline(8)") == []
        assert study.beaten_by(risks, "spline(32)") == [("qr", "spline(8)")]

    def test_an_unresolvable_ladder_selects_the_coarsest(self) -> None:
        """Nothing beats anything, so the ladder cannot tell its rungs apart and the coarsest
        carries the smaller variance and the weaker claim."""
        risks = {
            "gr1": {
                rung.label: draws_of([1e-3 + 1e-04 * (-1) ** i for i in range(8)])
                for rung in study.RUNGS
            }
        }

        assert study.select_rung(risks) == study.RUNGS[0].label

    def test_a_ladder_with_no_admissible_rung_takes_the_fewest_beaten_and_records_it(self) -> None:
        """Two metrics that disagree *resolvably* leave every rung beaten somewhere.

        That is a ladder with no admissible member rather than a tie, and a cell in that state has
        to be visible on the record: the count goes on the row instead of the selection quietly
        reading like an unbeaten winner.
        """
        # `qr` prefers the coarsest and `h3` the middle rung, both by consistent margins, so each
        # of the two beats the other somewhere and the finest is beaten on both.
        best = {
            "qr": {
                study.RUNGS[0].label: 0.0,
                study.RUNGS[1].label: 1e-05,
                study.RUNGS[2].label: 2e-05,
            },
            "h3": {
                study.RUNGS[0].label: 1e-05,
                study.RUNGS[1].label: 0.0,
                study.RUNGS[2].label: 2e-05,
            },
        }
        rows = [
            risk_row(
                data_seed=seed,
                phase="select",
                metric=metric,
                reduction="qr",
                candidate=label,
                risk=1e-3 + offset + 1e-09 * seed,
            )
            for seed in range(8)
            for metric, offsets in best.items()
            for label, offset in offsets.items()
        ]

        # `strict=False` because these rows are one regression's on purpose; the strict path is
        # what `TestSelectionEvidenceCannotBeMissing` exercises.
        picked = [row for row in study.selection_rows(rows, strict=False) if row.reduction == "qr"]

        # A tie on one beat each, broken to the coarsest, with the count on the record.
        assert [row.selected for row in picked] == [study.RUNGS[0].label]
        assert [row.beaten for row in picked] == [1]

    def test_a_metric_with_no_reading_contributes_nothing(self) -> None:
        risks = {
            "qr": {
                "spline(8)": draws_of([1e-3 + 1e-05 + 1e-09 * i for i in range(8)]),
                "spline(16)": draws_of([1e-3 + 1e-09 * i for i in range(8)]),
            },
            "h3": {"spline(16)": draws_of([1e-3] * 8)},
        }

        assert study.select_rung(risks) == "spline(16)"

    def test_a_regression_with_no_reading_at_all_raises(self) -> None:
        """A selection that silently became a default is the mistake this rule replaces."""
        with pytest.raises(ValueError, match="nothing was selected"):
            study.select_rung({"qr": {"bins(4)": draws_of([1e-3] * 8)}})

    def test_the_excess_is_relative_and_over_the_rungs_alone(self) -> None:
        """The control is not a candidate, and letting it into the denominator would make every
        rung's number a statement about how bad the control is."""
        readings = study.relative_excess(
            {"spline(8)": 2e-4, "spline(16)": 1e-4, "spline(32)": 1.5e-4, "bins(4)": 9e-2}
        )

        assert "bins(4)" not in readings
        assert readings == pytest.approx({"spline(8)": 1.0, "spline(16)": 0.0, "spline(32)": 0.5})

    def test_the_selection_reads_the_select_phase_alone(self) -> None:
        """Reading the audit rows here is exactly the self-certification the split prevents."""
        rows = [
            risk_row(
                data_seed=seed,
                phase=phase,
                candidate=rung.label,
                metric=metric,
                reduction=reduction,
                risk=risk + 1e-09 * seed,
            )
            for seed in range(8)
            for metric, reduction in (("qr", "qr"), ("h3", "qr"), ("gr1", "gr1"), ("gr2", "gr2"))
            for rung, base in zip(study.RUNGS, (1e-3, 2e-3, 3e-3), strict=True)
            # The audit block would pick the *finest* rung and the selection block the coarsest.
            for phase, risk in (("select", base), ("audit", 4e-3 - base))
        ]

        picked = study.selection_rows(rows)

        assert {row.selected for row in picked} == {study.RUNGS[0].label}
        assert {row.reduction for row in picked} == set(study.REDUCTIONS)
        assert study.selected_knots(picked)[("q-drift", 600)] == (
            ("gr1", study.RUNGS[0].n_knots),
            ("gr2", study.RUNGS[0].n_knots),
            ("qr", study.RUNGS[0].n_knots),
        )


def _passing_gate(**overrides: Any) -> list[study.RiskRow]:
    """Gate rows that pass both of B's clauses on all five metrics.

    The control rejected, every other rung readable and no better than the selected one, on each
    of ``qr``, ``gr1``, ``gr2``, ``h3`` and ``h2``.  Built from a helper rather than inline in
    each test, because what makes a gate *pass* is three clauses over five metrics and a test
    that spelled one of them would be exercising a gate the module does not have.
    """
    rows: list[study.RiskRow] = []
    for seed in range(8):
        for metric in study.METRICS:
            shared = dict(
                data_seed=seed, metric=metric.name, reduction=metric.reduction, **overrides
            )
            rows.append(risk_row(candidate=SELECTED, risk=1e-3, **shared))
            for rung in study.RUNGS:
                if rung.label == SELECTED:
                    continue
                rows.append(risk_row(candidate=rung.label, risk=1e-3 + 1e-5 * (seed % 3), **shared))
            rows.append(
                risk_row(
                    candidate=study.NEGATIVE_CONTROL.label,
                    risk=5e-3 + 1e-5 * (seed % 3),
                    **shared,
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

    def _verdict(self, fits: Sequence[study.FitRow]) -> str:
        picked = chosen()[("q-drift", 600)]
        return study.comparison_verdict(fits, _passing_gate(), "q-drift", 600, "ate", picked)

    def test_a_difference_far_outside_the_band_has_moved(self) -> None:
        fits = _arms([-0.9, -0.95, -1.0, -0.92, -0.98, -0.9, -1.05, -0.94], level=2.0, scrambles=3)

        assert self._verdict(fits) == "moved"

    def test_a_difference_wholly_inside_the_band_is_equivalent(self) -> None:
        fits = _arms([0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.015, 0.005], level=2.0, scrambles=3)

        assert self._verdict(fits) == "equivalent"

    def test_an_interval_straddling_the_band_is_unresolved(self) -> None:
        """Not a weak ``equivalent``: the run cannot tell the two apart at this precision."""
        fits = _arms([-0.9, 0.8, -0.7, 0.6, -0.5, 0.9, -0.8, 0.4], level=2.0, scrambles=3)

        assert self._verdict(fits) == "unresolved"

    def test_a_movement_in_either_direction_counts(self) -> None:
        """A reference that makes the remainder *larger* is still a learner effect.

        E2 asks whether item 13's failure is a learner failure at all, and a rule that only
        counted improvement would report the awkward half of that answer as a null result.
        """
        fits = _arms([0.9, 0.95, 1.0, 0.92, 0.98, 0.9, 1.05, 0.94], level=2.0, scrambles=3)

        assert self._verdict(fits) == "moved"


class TestAFailedGateMakesItsWholeCellUnresolved:
    """The gates are what say the comparison is about the reduction learner at all."""

    def test_a_negative_control_that_is_not_rejected_fails_gate_b(self) -> None:
        """Without teeth the gate cannot discriminate, whatever the comparison then shows."""
        rows = [
            risk_row(
                data_seed=seed,
                metric=metric.name,
                reduction=metric.reduction,
                candidate=label,
                risk=1e-3,
            )
            for seed in range(8)
            for metric in study.METRICS
            for label in (SELECTED, study.NEGATIVE_CONTROL.label, *[r.label for r in study.RUNGS])
        ]
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        picked = chosen()[("q-drift", 600)]

        verdict, reason = study.gate_verdict(fits, rows, "q-drift", 600, picked)

        assert verdict == "fail"
        assert "not rejected" in reason
        assert study.comparison_verdict(fits, rows, "q-drift", 600, "ate", picked) == "unresolved"

    def test_a_rung_that_beats_the_selected_one_fails_gate_b(self) -> None:
        """The comparison would then answer for a reference another resolution is better than.

        E2's own failing clause, and what E2R changes is which reference it is about: the rung
        pass one selected, held to replicating on a block the selection never saw.
        """
        better = study.RUNGS[-1].label
        rows = [row for row in _passing_gate() if row.candidate != better]
        rows += [
            risk_row(
                data_seed=seed,
                metric=metric.name,
                reduction=metric.reduction,
                candidate=better,
                risk=1e-4,
            )
            for seed in range(8)
            for metric in study.METRICS
        ]
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)

        verdict, reason = study.gate_verdict(fits, rows, "q-drift", 600, chosen()[("q-drift", 600)])

        assert verdict == "fail"
        assert f"beats {SELECTED}" in reason

    def test_a_composite_metric_can_fail_a_gate_the_components_pass(self) -> None:
        """Which is the whole reason the composites were added.

        Componentwise the three regressions are ranked as functions; the fit divides by
        :math:`g^*_b` and :math:`g_{r,1,b}`, so a rung can estimate ``gr2`` better on average and
        worse where the denominator is small.  A gate that read only the three would certify a
        reference the estimator is more sensitive to than the ranking says.
        """
        beaten = study.RUNGS[0].label
        rows = [
            row for row in _passing_gate() if not (row.candidate == beaten and row.metric == "h2")
        ]
        rows += [
            risk_row(data_seed=seed, metric="h2", reduction="gr2", candidate=beaten, risk=1e-5)
            for seed in range(8)
        ]

        verdict, reason = study.gate_verdict(
            _arms([-0.9] * 8, level=2.0, scrambles=3),
            rows,
            "q-drift",
            600,
            chosen()[("q-drift", 600)],
        )

        assert verdict == "fail"
        assert "on h2" in reason
        assert "on gr2" not in reason

    def test_a_rung_that_could_not_be_scored_is_a_gap_and_not_a_pass(self) -> None:
        """The finest rung refused by its own points-per-parameter budget is the usual case, and
        it means the ladder the selection ranged over was shorter than the one declared."""
        missing = study.RUNGS[-1].label
        rows = [row for row in _passing_gate() if row.candidate != missing]

        verdict, reason = study.gate_verdict(
            _arms([-0.9] * 8, level=2.0, scrambles=3),
            rows,
            "q-drift",
            600,
            chosen()[("q-drift", 600)],
        )

        assert verdict == "fail"
        assert f"{missing} has no reading" in reason

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

        picked = chosen()[("q-drift", 600)]
        verdict, reason = study.gate_verdict(rows, _passing_gate(), "q-drift", 600, picked)

        assert verdict == "fail"
        assert reason.startswith("C:") or "; C:" in reason
        assert (
            study.comparison_verdict(rows, _passing_gate(), "q-drift", 600, "ate", picked)
            == "unresolved"
        )

    def test_a_run_with_no_budget_draw_fails_gate_c(self) -> None:
        """Unmeasured and small must not read alike, which is E1's lesson in its other form."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=1)

        verdict, reason = study.gate_verdict(
            fits, _passing_gate(), "q-drift", 600, chosen()[("q-drift", 600)]
        )

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
            risk_row(
                data_seed=seed,
                phase=phase,
                candidate=label,
                metric=metric.name,
                reduction=metric.reduction,
                risk=value,
            )
            for seed in (1, 2, 3)
            for phase in ("select", "audit")
            for metric in study.METRICS
            for label, value in ((SELECTED, 1e-3), ("bins(8)", 2e-3))
        ]
        picks = study.selection_rows(risks)

        for built, headers in (
            (study.selection_table(picks), study.SELECTION_HEADERS),
            (study.gate_rows(fits, risks, chosen()), study.GATE_HEADERS),
            (study.comparison_rows(fits, risks, chosen()), study.COMPARISON_HEADERS),
            (study.cost_rows(fits), study.COST_HEADERS),
            (
                study.integrity_rows(study.run_integrity(fits, risks, chosen(), expected_draws=3)),
                study.INTEGRITY_HEADERS,
            ),
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


# ------------------------------------------------- what the PR #76 review found, as tests
#
# Three defects, each of which let gate B pass without establishing that the selected reference
# is adequate, and each with the acceptance test the review asked for.  They are grouped apart
# from the classes above because the classes above are about the *instrument* and these are about
# the **experiment**: what a rung was chosen on, what it was certified against, and whether a
# missing reading can reach a verdict.


def parsed(argv: Sequence[str] = ()) -> Any:
    """The real parser, so a test cannot pass a configuration the command would reject."""
    return study.build_parser().parse_args(
        ["--phase", "select", "--cells", "q-drift", "--sizes", "600", *argv]
    )


def manifest(**overrides: Any) -> study.SelectionManifest:
    """A complete, valid manifest for one cell -- the thing a decision run is handed."""
    args = parsed(["--selection-draws", "16", "--decision-draws", "32"])
    cohorts = study.cohort_seeds(args.seed, {"selection": 16, "decision": 32})
    fields: dict[str, Any] = {
        "rule": study.frozen_rule(),
        "configuration": study.configuration(args),
        "cohorts": {name: [list(pair) for pair in rows] for name, rows in cohorts.items()},
        "selected": [
            study.SelectionRow(
                cell="q-drift",
                n=600,
                reduction=name,
                selected=SELECTED,
                beaten=0,
                excess=0.0,
                runner_up="-",
                runner_up_excess=float("nan"),
                metrics=" ".join(study.METRICS_OF[name]),
                draws=16,
            )
            for name in study.REDUCTIONS
        ],
    }
    fields.update(overrides)
    return study.SelectionManifest(**fields)


class TestSelectionAndDecisionAreDisjointDraws:
    """The review's first blocker: the audit was split at the QMC row and not at the draw.

    Four disjoint scramble streams split the **integration** noise.  They do not make an audit
    independent of a data-dependent selection, because the quadrature is not what was reused:
    both risk tables were functions of the same fitted samples, the same fold assignments and the
    same draw-specific nuisance states, and the rung was chosen *across those draws*.  The draw is
    the independent unit -- `draw_risks` averages within one and every interval resamples them --
    so the repair is at the draw.
    """

    def test_no_draw_is_in_both_cohorts(self) -> None:
        """On the **data** seed, which is the sample, and not on the `(data, fold)` pair.

        Two draws that share a data seed and differ in their split are the same rows under two
        partitions, and the selection saw those rows. `study-manifest.md` records C3c meeting
        exactly that -- a batch believed fresh shared the pilot's data seeds while drawing its
        own splits, because `SeedSequence.generate_state` is prefix-stable -- so a pair-wise
        check passes there and is the wrong check.
        """
        cohorts = study.cohort_seeds(20250801, {"selection": 16, "decision": 32})

        assert len(cohorts["selection"]) == 16
        assert len(cohorts["decision"]) == 32
        assert not {data for data, _ in cohorts["selection"]} & {
            data for data, _ in cohorts["decision"]
        }

    def test_neither_cohort_is_a_prefix_of_one_stream(self) -> None:
        """Spawned children rather than one stream sliced in two.

        A slice is prefix-stable, so raising one cohort's count would shift which draws the other
        took -- and, with the two halves of one stream read as data and fold seeds, the longer
        cohort's data seeds would *be* the shorter one's two halves.
        """
        small = study.cohort_seeds(20250801, {"selection": 4, "decision": 32})
        large = study.cohort_seeds(20250801, {"selection": 64, "decision": 32})

        assert small["decision"] == large["decision"]
        assert not {data for data, _ in large["selection"]} & {
            data for data, _ in large["decision"]
        }

    def test_both_arms_of_a_decision_draw_carry_one_payload(self) -> None:
        """The pairing is structural rather than recovered by joining on a seed afterwards.

        `decision_draw` fits the control arm and the reference arm from **one** payload, so they
        see the same rows, the same split and the same evaluation windows -- which is what makes
        the difference paired and cancels the evaluation rule's own error.
        """
        args = parsed(["--decision-draws", "3"])
        seeds = study.cohort_seeds(args.seed, {"selection": 2, "decision": 3})["decision"]
        payloads = study._payloads(args, "decision", seeds, {("q-drift", 600): (("qr", 8),)})

        assert [(row.data_seed, row.fold_seed) for row in payloads] == list(seeds)
        assert {row.cohort for row in payloads} == {"decision"}

    def test_a_manifest_whose_draws_are_this_runs_draws_is_refused(self) -> None:
        """The check that makes the disjointness a property of the *pair* of runs.

        A run could otherwise be handed a manifest selected on its own draws -- by a rerun at the
        same seed, or by a hand-edited file -- and nothing inside one process could see it.
        """
        args = parsed(["--decision-draws", "32"])
        seeds = study.cohort_seeds(args.seed, {"selection": 16, "decision": 32})["decision"]
        reused = manifest(
            cohorts={"selection": [list(pair) for pair in seeds], "decision": []},
        )

        complaints = study.validate_selection(
            reused,
            cells=["q-drift"],
            sizes=[600],
            configuration=study.configuration(args),
            decision_seeds=seeds,
        )

        assert any("decision draw(s) are selection draws" in line for line in complaints)


class TestTheDecisionCommandCannotSelect:
    """`--phase decide` reads a mapping and does not choose one, and that is enforced."""

    def test_deciding_never_calls_the_selection(self, tmp_path: Any, monkeypatch: Any) -> None:
        written = study.write_selection_manifest(manifest(), tmp_path / "selection.json")
        args = parsed(
            [
                "--phase",
                "decide",
                "--selection",
                str(written),
                "--selection-draws",
                "16",
                "--decision-draws",
                "32",
                "--out",
                str(tmp_path / "rows"),
            ]
        )

        def refuse(*_: Any, **__: Any) -> None:
            raise AssertionError("a decision run selected a rung")

        monkeypatch.setattr(study, "selection_rows", refuse)
        monkeypatch.setattr(study, "select_rung", refuse)
        monkeypatch.setattr(study, "map_parallel", lambda *_, **__: [])

        assert study.decide(args) == 0

    def test_deciding_without_a_manifest_is_refused(self, tmp_path: Any) -> None:
        args = parsed(["--phase", "decide", "--out", str(tmp_path / "rows")])

        assert study.decide(args) == 2


class TestTheFidelityClauseIsNonInferiorityAndNotSignificance:
    """The review's second blocker: failure to show superiority was read as fidelity.

    E2R's first instrument failed gate B only when a competitor's interval lay **wholly below
    zero**, so an imprecise comparison passed by default -- and the record says so in its own
    words, calling two rungs "genuinely indistinguishable" because a doubled block turned a
    resolved `-7.6e-07` into a `-3.0e-07` straddling zero.  An interval containing zero
    establishes neither equality nor adequate approximation.
    """

    def _gate(self, metric: str, label: str, gaps: Sequence[float]) -> list[study.RiskRow]:
        """`_passing_gate` with one `(metric, rung)` comparison replaced by a declared pattern.

        The baseline's risk is `1e-3` on every metric and draw, so a gap *is* the offset.
        """
        rows = [
            row for row in _passing_gate() if not (row.metric == metric and row.candidate == label)
        ]
        reduction = next(m.reduction for m in study.METRICS if m.name == metric)
        rows += [
            risk_row(
                data_seed=seed,
                metric=metric,
                reduction=reduction,
                candidate=label,
                risk=1e-3 + gap,
            )
            for seed, gap in enumerate(gaps)
        ]
        return rows

    def test_the_composite_margin_is_the_column_in_risk_units(self) -> None:
        r"""`(FIDELITY_FRACTION * delta)^2 / (n * weight_scale)`, and nothing measured after.

        An excess risk `x` on a composite is a mean square perturbation of the correction, so its
        root bounds the mean by Cauchy--Schwarz and `sqrt(n)` times that bounds the column.
        """
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        risks = [risk_row(**vars(row)) for row in _passing_gate()]
        for row in risks:
            if row.metric in ("h3", "h2"):
                row.weight_scale = 4.0

        margins = study.noninferiority_margins(
            fits, risks, "q-drift", 600, chosen()[("q-drift", 600)]
        )

        assert study.equivalence_margin(fits, "q-drift", 600, "ate") == pytest.approx(0.5)
        assert margins["h3"] == pytest.approx((study.FIDELITY_FRACTION * 0.5) ** 2 / (600 * 4.0))
        # And the three componentwise metrics take the control's own measured distance instead.
        assert margins["qr"] == pytest.approx(study.COMPONENT_FRACTION * 4.0e-3, rel=1e-2)

    def test_a_wide_interval_crossing_zero_and_the_margin_is_unresolved(self) -> None:
        """**Not** a pass.  This is the exact reading the repair removes: an interval that
        straddles zero says the run cannot tell two rungs apart, which is a statement about the
        study's precision and not a certification of the reference."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        risks = self._gate("h3", study.RUNGS[0].label, [5e-4 * (-1) ** i for i in range(8)])

        verdict, reason = study.gate_verdict(
            fits, risks, "q-drift", 600, chosen()[("q-drift", 600)]
        )

        assert verdict == "unresolved"
        assert "not shown non-inferior on h3" in reason

    def test_a_non_significant_gap_below_the_margin_does_not_pass(self) -> None:
        """The same clause at a gap that is not wild: mean near zero, spread wide enough that the
        lower bound sits under `-delta_metric`.  Nothing here is significant at zero, and under
        the old rule that alone certified the reference."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        risks = self._gate("h2", study.RUNGS[0].label, [1e-4 * (-1) ** i for i in range(8)])

        verdict, reason = study.gate_verdict(
            fits, risks, "q-drift", 600, chosen()[("q-drift", 600)]
        )

        assert verdict != "pass"
        assert "not shown non-inferior on h2" in reason
        assert "beats" not in reason

    def test_a_difference_inside_the_margin_satisfies_non_inferiority(self) -> None:
        """A consistent gap of `1e-05` against a margin of `4.6e-05`: the selected rung is shown
        to be at most practically worse, which is what the clause asks.

        **The cell still fails**, on the separate and unchanged clause that no competing rung may
        be measurably better at all.  The two are kept apart on purpose -- the review asked for a
        non-inferiority gate and this repository's own rule forbids *loosening* a clause once
        numbers exist, so the fidelity clause was added beside the superiority one rather than in
        place of it.  A cell that fails only on superiority with every bound clear is a real
        finding: a finely resolved ladder whose selected rung is not its best.
        """
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        label = study.RUNGS[0].label
        risks = self._gate("h3", label, [-1e-5 - 1e-9 * i for i in range(8)])
        picked = chosen()[("q-drift", 600)]

        margins = study.noninferiority_margins(fits, risks, "q-drift", 600, picked)
        bounds, _ = study.simultaneous_lower_bounds(
            study.audit_family(risks, "q-drift", 600, picked)
        )

        assert bounds[("h3", label)] > -margins["h3"]
        verdict, reason = study.gate_verdict(fits, risks, "q-drift", 600, picked)
        assert verdict == "fail"
        assert f"{label} beats {SELECTED} on h3" in reason
        assert "not shown non-inferior on h3" not in reason

    def test_the_bound_is_simultaneous_across_comparisons(self) -> None:
        """One resample of *draw indices*, shared by every comparison.

        The comparisons are five metrics of three regressions against one baseline on the same
        fits, so they are strongly dependent; resampling them apart would treat that dependence
        as independence.  Being simultaneous, each bound is at or below its own per-comparison
        one, which is deliberate conservatism -- the conjunction is an intersection--union test
        and needs no correction at all.
        """
        picked = chosen()[("q-drift", 600)]
        risks = _passing_gate()
        family = study.audit_family(risks, "q-drift", 600, picked)
        bounds, shared = study.simultaneous_lower_bounds(family)

        assert shared == 8
        assert set(bounds) == set(family)
        for key, values in family.items():
            alone = study.interval(np.array([values[seed] for seed in sorted(values)]))[0]
            assert bounds[key] <= alone + 1e-12

    def test_too_few_shared_draws_carry_no_bound(self) -> None:
        """An interval that cannot be formed is a gap in the evidence, not a wide one."""
        bounds, shared = study.simultaneous_lower_bounds({("qr", "spline(8)"): {1: 0.5}})

        assert (bounds, shared) == ({}, 1)


class TestMissingEvidenceCannotReachAVerdict:
    """The review's third blocker: a lost reading fell back and the run still decided."""

    def test_a_regression_with_no_reading_raises_in_strict_mode(self) -> None:
        """`select_rung` refused a regression it could not rank and `selection_rows` never
        reached that refusal: it returned no row, `Payload.references` filled the hole with
        `FALLBACK_RUNG`, and the cell went on to a verdict.  Both halves of the fail-closed
        behaviour were written and the path between them was not."""
        rows = [
            risk_row(data_seed=seed, phase="select", metric=metric, reduction=reduction)
            for seed in range(8)
            # `gr1` is wholly absent, which is one lost selection row.
            for metric, reduction in (("qr", "qr"), ("h3", "qr"), ("gr2", "gr2"), ("h2", "gr2"))
        ]

        with pytest.raises(ValueError, match="nothing selected its rung"):
            study.selection_rows(rows)

        assert {row.reduction for row in study.selection_rows(rows, strict=False)} == {"qr", "gr2"}

    def test_a_cell_with_no_selected_rung_cannot_pass(self) -> None:
        """Every decision verdict names an explicit label or the cell is `unresolved`."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        risks = _passing_gate()

        verdict, reason = study.gate_verdict(fits, risks, "q-drift", 600, {"qr": SELECTED})

        assert verdict == "unresolved"
        assert "no selected rung for gr1, gr2" in reason
        assert study.comparison_verdict(fits, risks, "q-drift", 600, "ate", None) == "unresolved"

    def test_two_readable_draws_of_thirty_two_is_unresolved_and_invalid(self) -> None:
        """A bootstrap that quietly shrank from 32 draws to whatever survived would report a
        thinner study as the declared one.  Both halves are checked: the cell cannot carry a
        verdict, and the **run** is marked invalid for branching."""
        fits = _arms([-0.9] * 2, level=2.0, scrambles=3)
        risks = [row for row in _passing_gate() if row.data_seed < 2]
        picked = chosen()

        verdict, reason = study.gate_verdict(
            fits, risks, "q-drift", 600, picked[("q-drift", 600)], expected_draws=32
        )
        integrity = study.run_integrity(fits, risks, picked, expected_draws=32)

        assert verdict == "unresolved"
        assert "of the 32 declared" in reason
        assert [row.valid for row in integrity] == [False]
        assert (
            study.comparison_verdict(
                fits, risks, "q-drift", 600, "ate", picked[("q-drift", 600)], expected_draws=32
            )
            == "unresolved"
        )

    def test_a_recorded_error_invalidates_the_cell(self) -> None:
        """A refused candidate is a gap in the gate, and a gap has to look like one."""
        fits = _arms([-0.9] * 8, level=2.0, scrambles=3)
        risks = [*_passing_gate(), risk_row(data_seed=99, error="ValueError")]

        integrity = study.run_integrity(fits, risks, chosen(), expected_draws=8)

        assert [row.valid for row in integrity] == [False]
        assert [row.risk_errors for row in integrity] == [1]

    def test_a_complete_cell_is_valid(self) -> None:
        """The negative control on the class: everything above has to be reachable *and* the
        ordinary case has to pass, or the integrity row is a constant."""
        integrity = study.run_integrity(
            _arms([-0.9] * 8, level=2.0, scrambles=3),
            _passing_gate(),
            chosen(),
            expected_draws=8,
        )

        assert [row.valid for row in integrity] == [True]


class TestTheManifestIsCheckedBeforeAnythingIsFitted:
    """A run that could not have been certified must not be one that produced numbers first."""

    def _complaints(self, held: study.SelectionManifest, **overrides: Any) -> list[str]:
        args = parsed(["--decision-draws", "32", *overrides.pop("argv", [])])
        return study.validate_selection(
            held,
            cells=["q-drift"],
            sizes=[600],
            configuration=study.configuration(args),
            decision_seeds=study.cohort_seeds(args.seed, {"selection": 16, "decision": 32})[
                "decision"
            ],
        )

    def test_a_complete_manifest_has_nothing_to_say(self) -> None:
        assert self._complaints(manifest()) == []

    def test_a_missing_regression_is_named(self) -> None:
        held = manifest(selected=[row for row in manifest().selected if row.reduction != "gr1"])

        assert any("has no rung for gr1" in line for line in self._complaints(held))

    def test_a_rule_that_moved_after_the_selection_is_refused(self) -> None:
        """The constants may be changed before a dispatch and not after one, and a two-command
        study is where that stops being an honour system: the manifest records the rule it was
        selected under and the decision run compares."""
        held = manifest(rule={**study.frozen_rule(), "equivalence_fraction": 0.5})

        assert any("equivalence_fraction" in line for line in self._complaints(held))

    def test_a_reference_built_at_another_resolution_is_refused(self) -> None:
        """A rung selected at one block size is a statement about that block size."""
        args = parsed()
        held = manifest(configuration={**study.configuration(args), "reference_points": 4_096})

        assert any("reference_points" in line for line in self._complaints(held))

    def test_a_selection_taken_on_too_few_draws_is_refused(self) -> None:
        held = manifest(
            selected=[replace(row, draws=2) for row in manifest().selected],
        )

        assert any("of the 16 declared" in line for line in self._complaints(held))
