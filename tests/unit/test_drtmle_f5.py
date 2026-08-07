"""F5's harness is the instrument its preregistration says it is.

``benchmarks/drtmle_f5.py`` runs the terminal experiment of the ``DRTMLE`` investigation, and
every claim below is one a reader of a verdict has to be able to rely on without rerunning the
study.  Three of them are repairs of defects F4's run surfaced, and they are tested rather than
described because F4's own defects were described accurately in prose while the code did
something else.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from benchmarks import drtmle_construction, drtmle_f5


class TestTheSixArmsAreWhatTheTableSays:
    """Six arms of the roadmap matrix's eight, and every dropped cell has a recorded reason."""

    def test_there_are_six(self) -> None:
        assert len(drtmle_f5.ARMS) == 6

    def test_both_dropped_cells_are_named_with_their_reason(self) -> None:
        # A cell removed from a frozen matrix has to say why in the artefact, not in a comment:
        # the manifest carries `arms_dropped` and this is what keeps it populated.
        assert set(drtmle_f5.ARMS_DROPPED) == {"ceiling-nested", "boost-pooled"}
        for name, why in drtmle_f5.ARMS_DROPPED.items():
            assert len(why) > 40, name
        assert name not in drtmle_f5.ARMS

    def test_the_cross_fitting_axis_survives_in_two_learner_rows(self) -> None:
        """Dropping ``boost-pooled`` costs the interaction at boost and nowhere else.

        F4 measured something on the cross-fitting axis, so it has to stay identifiable
        somewhere; this is the check that it does.
        """
        by_learner: dict[str, set[str]] = {}
        for arm in drtmle_f5.ARMS.values():
            by_learner.setdefault(arm.learner, set()).add(arm.crossfit)
        assert by_learner["glm"] == {"pooled", "nested"}
        assert by_learner["gam"] == {"pooled", "nested"}
        assert by_learner["boost"] == {"nested"}

    def test_the_baseline_is_the_shipped_configuration(self) -> None:
        baseline = drtmle_f5.ARMS[drtmle_f5.BASELINE_ARM]
        assert (baseline.learner, baseline.crossfit) == ("glm", "pooled")
        library = drtmle_f5.reduced_library("glm", 0)
        # C3c's configuration, keyword for keyword: the shipped preset name, not a hand-built
        # library that happens to contain the same candidates.
        assert library["reduced_outcome_learner"] == "glm"
        assert library["reduced_treatment_learner"] == "glm"

    @pytest.mark.parametrize("arm", sorted(drtmle_f5.ARMS))
    def test_every_arm_builds(self, arm: str) -> None:
        settings = {"n_folds": 5, "learner_folds": 3, "estimands": ("ate",)}
        extra = {}
        if drtmle_f5.ARMS[arm].learner == "ceiling":
            # The ceiling needs a law, a window and that window's weights; `arm_estimator`
            # refuses it without them, which is the next test.
            pytest.skip("the ceiling arm is built with a companion in test_the_ceiling_refuses")
        estimator = drtmle_f5.arm_estimator(arm, settings, random_state=0, **extra)
        assert estimator.reduced_crossfit == drtmle_f5.ARMS[arm].crossfit

    def test_the_ceiling_refuses_a_bare_construction(self) -> None:
        with pytest.raises(ValueError, match="needs dgp=, window= and row_weights="):
            drtmle_f5.arm_estimator("ceiling", {"n_folds": 5}, random_state=0)

    def test_nested_arms_really_carry_the_keyword(self) -> None:
        settings = {"n_folds": 5, "learner_folds": 3, "estimands": ("ate",)}
        for arm in ("glm-nested", "gam-nested", "boost-nested"):
            assert (
                drtmle_f5.arm_estimator(arm, settings, random_state=0).reduced_crossfit == "nested"
            )

    def test_the_flexible_libraries_keep_mean(self) -> None:
        """One factor moves, not two.

        The shipped baseline is the ``"glm"`` preset, which is ``mean + glm``.  A
        single-candidate flexible arm would move the function class **and** the ensemble shape,
        which is the bundled-arm mistake F4's whole matrix exists to remove.
        """
        for learner in ("gam", "boost"):
            library = drtmle_f5.reduced_library(learner, 0)
            for slot in ("reduced_outcome_learner", "reduced_treatment_learner"):
                names = [name for name, _ in library[slot]]
                assert names == ["mean", learner], (slot, names)

    def test_the_two_slots_are_tasked_apart(self) -> None:
        """``Q_r`` and ``g_r2`` are regressions and ``g_r1`` is a classification.

        A single object resolved at one task would hand a classifier to ``Q_r``, whose target is
        a signed residual.
        """
        library = drtmle_f5.reduced_library("gam", 0)
        outcome = library["reduced_outcome_learner"][1][1]
        treatment = library["reduced_treatment_learner"][1][1]
        assert "Ridge" in type(outcome.named_steps["model"]).__name__
        assert "Logistic" in type(treatment.named_steps["model"]).__name__

    def test_lightgbm_threads_are_pinned(self) -> None:
        # F5's row requires thread counts pinned and `_boost` sets none; LightGBM's default is
        # -1, which would put a machine's worth of threads inside one of ten workers.
        library = drtmle_f5.reduced_library("boost", 0)
        for slot in ("reduced_outcome_learner", "reduced_treatment_learner"):
            assert library[slot][1][1].get_params()["n_jobs"] == 1

    def test_only_three_arms_are_nominable(self) -> None:
        nominable = sorted(name for name, arm in drtmle_f5.ARMS.items() if arm.nominable)
        assert nominable == ["boost-nested", "gam-nested", "gam-pooled"]

    def test_the_ceiling_is_not_nominable(self) -> None:
        # A ceiling measures an attainable bound and is not a procedure a caller can run.
        assert not drtmle_f5.ARMS["ceiling"].nominable

    def test_boost_only_reaches_a_production_branch_under_nested(self) -> None:
        # A1b's pooled design/target-continuity premise is not closed for a boosted reduction,
        # so the pooled cell could never have been promoted -- which is why dropping it on cost
        # closes no branch.
        boost = [arm for arm in drtmle_f5.ARMS.values() if arm.learner == "boost"]
        assert [arm.crossfit for arm in boost] == ["nested"]
        assert all(arm.nominable for arm in boost)


class TestTheRunRefusesTheFallback:
    """Two function classes under one arm name make the contract's entropy row ambiguous."""

    def test_it_passes_on_this_box(self) -> None:
        assert drtmle_f5.refuse_on_fallback() == []

    def test_it_refuses_when_lightgbm_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.learners import library as library_module

        monkeypatch.setattr(library_module, "_LIGHTGBM", False, raising=False)
        monkeypatch.setattr(library_module, "has_lightgbm", lambda: False)
        monkeypatch.setattr(drtmle_f5, "has_lightgbm", lambda: False)
        complaints = drtmle_f5.refuse_on_fallback()
        assert complaints
        assert any("has_lightgbm() is False" in line for line in complaints)

    def test_it_names_the_fallback_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        monkeypatch.setattr(drtmle_f5, "_boost", lambda task, seed: HistGradientBoostingRegressor())
        complaints = drtmle_f5.refuse_on_fallback()
        assert any("HistGradientBoostingRegressor" in line for line in complaints)

    def test_the_manifest_records_the_resolved_implementation(self) -> None:
        resolved = drtmle_f5.resolved_implementations()
        assert resolved["boost-nested"]["regression"] == "LGBMRegressor"
        assert resolved["boost-nested"]["classification"] == "LGBMClassifier"
        assert resolved["glm-pooled"]["regression"] == "glm"


class TestTheVerdictRuleIsAPartition:
    """F4's two rule defects, made unreachable rather than documented."""

    def test_every_band_is_finite_and_positive(self) -> None:
        for column in drtmle_f5.COLUMNS:
            assert math.isfinite(column.band), column.name
            assert column.band > 0.0, column.name

    def test_f4s_collision_case_is_equivalent_and_not_moved(self) -> None:
        """The interval that broke F4's rule.

        ``[-1e-4, -1e-5]`` excludes zero, so F4's ``moved``-first ladder labelled it a
        localization -- while lying wholly inside its own ``+/- 0.125`` negligible margin.
        """
        assert drtmle_f5.verdict(-1e-4, -1e-5, 0.125, -1) == "equivalent"

    @pytest.mark.parametrize("orientation", (-1, 0, 1))
    def test_all_four_verdicts_are_reachable(self, orientation: int) -> None:
        band = 0.1
        seen = {
            drtmle_f5.verdict(0.0, 0.0, band, orientation),
            drtmle_f5.verdict(2 * band, 3 * band, band, orientation),
            drtmle_f5.verdict(-3 * band, -2 * band, band, orientation),
            drtmle_f5.verdict(-2 * band, 2 * band, band, orientation),
        }
        assert "equivalent" in seen
        assert "unresolved" in seen
        assert {"improved", "worsened"} <= seen

    def test_at_most_one_clause_fires(self) -> None:
        """Exhaustive over a grid: the three named verdicts are mutually exclusive."""
        band = 0.1
        grid = np.linspace(-0.5, 0.5, 41)
        for lower in grid:
            for upper in grid:
                if upper < lower:
                    continue
                got = drtmle_f5.verdict(float(lower), float(upper), band, -1)
                assert got in drtmle_f5.VERDICTS
                equivalent = -band <= lower and upper <= band
                beyond = lower > band or upper < -band
                assert not (equivalent and beyond), (lower, upper)
                if equivalent:
                    assert got == "equivalent"
                elif beyond:
                    assert got in {"improved", "worsened"}
                else:
                    assert got == "unresolved"

    def test_orientation_decides_which_side_is_an_improvement(self) -> None:
        assert drtmle_f5.verdict(0.5, 0.7, 0.1, -1) == "worsened"
        assert drtmle_f5.verdict(0.5, 0.7, 0.1, +1) == "improved"
        assert drtmle_f5.verdict(-0.7, -0.5, 0.1, -1) == "improved"

    def test_a_non_finite_band_is_unresolved_not_a_verdict(self) -> None:
        # F4 passed `float("inf")` for every non-primary column, which made `unresolved`
        # unreachable there. Here an infinite band produces `unresolved` and never a judgement.
        assert drtmle_f5.verdict(-1.0, 1.0, float("inf"), -1) == "unresolved"
        assert drtmle_f5.verdict(-1.0, 1.0, float("nan"), -1) == "unresolved"

    def test_the_remainder_band_is_per_cell(self) -> None:
        """One band across cells whose levels differ threefold is a band for one of them."""
        column = drtmle_f5.COLUMN_BY_NAME["root_n_remaining"]
        q = drtmle_f5.band_for(column, "q-drift")
        g = drtmle_f5.band_for(column, "g-drift")
        assert q == pytest.approx(0.125)
        assert g == pytest.approx(0.413, abs=1e-3)
        assert g > q

    def test_the_q_drift_band_is_f4s_verbatim(self) -> None:
        # Carried so the two studies share a scale.
        expected = drtmle_construction.NEGLIGIBLE_EFFECT * drtmle_construction.C3C_REMAINING_QDRIFT
        assert drtmle_f5.band_for(
            drtmle_f5.COLUMN_BY_NAME["root_n_remaining"], "q-drift"
        ) == pytest.approx(expected)

    def test_a_relative_band_without_a_baseline_is_not_a_verdict(self) -> None:
        column = drtmle_f5.COLUMN_BY_NAME["rmse"]
        assert math.isnan(drtmle_f5.band_for(column, "q-drift", None))
        assert math.isnan(drtmle_f5.band_for(column, "q-drift", 0.0))
        assert drtmle_f5.band_for(column, "q-drift", 2.0) == pytest.approx(0.2)

    def test_the_smallest_decision_margin_comes_off_the_rule(self) -> None:
        # It is what the ceiling's numerical error must be small against, so it is derived
        # from the gating columns rather than chosen beside them.
        assert drtmle_f5.smallest_decision_margin() == pytest.approx(0.02)


class TestIdentityFailuresAreExactAndNotBanded:
    def test_they_are_not_a_declared_column(self) -> None:
        """C3c recorded zero across 6,000 fits, so this quantity has an exact answer.

        Turning it into a paired difference with a tolerance would make the one certainty the
        study has into a statistic.
        """
        assert "identity_failures" not in drtmle_f5.COLUMN_BY_NAME
        assert drtmle_f5.IDENTITY_FAILURES_ARE_EXACT


class TestADeadGateIsRefused:
    """A veto that cannot fire reads exactly like one that fired and found nothing."""

    def _contrast(self, column: str, mean: float) -> drtmle_f5.ContrastRow:
        return drtmle_f5.ContrastRow(
            cohort="selection",
            cell="q-drift",
            n=600,
            arm="gam-pooled",
            role="candidate",
            column=column,
            group="theorem",
            draws=24,
            mean=mean,
            lower=mean,
            upper=mean,
            paired_sd=0.0,
            band=0.1,
            verdict="equivalent",
        )

    def test_a_gating_column_with_no_finite_reading_is_named(self) -> None:
        gating = [c.name for c in drtmle_f5.COLUMNS if c.gates]
        rows = [self._contrast(name, float("nan")) for name in gating]
        complaints = drtmle_f5.refuse_dead_gates(rows)
        assert len(complaints) == len(gating)

    def test_a_missing_gating_column_is_named(self) -> None:
        complaints = drtmle_f5.refuse_dead_gates([])
        assert all("no rows at all" in line for line in complaints)
        assert complaints

    def test_a_live_gate_passes(self) -> None:
        rows = [self._contrast(c.name, 0.01) for c in drtmle_f5.COLUMNS if c.gates]
        assert drtmle_f5.refuse_dead_gates(rows) == []

    def test_nominate_refuses_while_a_gate_is_dead(self) -> None:
        rows = [self._contrast(c.name, float("nan")) for c in drtmle_f5.COLUMNS if c.gates]
        with pytest.raises(ValueError, match="gating column is dead"):
            drtmle_f5.nominate(rows, [])


class TestNothingFromThePilotReachesAVerdict:
    """The pilot measures cost and sizes nothing, enforced structurally rather than remembered."""

    def _fit_row(self, cohort: str) -> drtmle_f5.FitRow:
        return drtmle_f5.FitRow(
            cohort=cohort,
            cell="q-drift",
            n=600,
            data_seed=1,
            fold_seed=2,
            arm="glm-pooled",
            estimand="ate",
            psi=0.1,
            truth=0.1,
            abs_error=0.0,
            std_error=0.01,
            root_n_remaining=1.0,
            score_8=0.0,
            score_9=0.0,
            score_10=0.0,
            identity_failures=0,
            score_failures=0,
            valid=True,
            rounds=3,
            closing=1,
            exit_reason="tolerance",
            failure="",
            bound_active=False,
            initial_clip_share=0.0,
            flex_weight_min=float("nan"),
            risk_qr=float("nan"),
            risk_gr1=float("nan"),
            risk_gr2=float("nan"),
            risk_h3=float("nan"),
            risk_h2=float("nan"),
            impl="glm",
            seconds=5.0,
        )

    def test_contrast_rows_refuses_a_sizing_row(self) -> None:
        with pytest.raises(ValueError, match="measures cost and sizes nothing"):
            drtmle_f5.contrast_rows([self._fit_row(drtmle_f5.SIZING_COHORT)])

    def test_a_cohort_row_is_accepted(self) -> None:
        drtmle_f5.contrast_rows([self._fit_row("selection")])

    def test_the_sizing_stream_is_disjoint_from_both_cohorts(self) -> None:
        cohorts = drtmle_f5.cohort_seeds(drtmle_f5.COHORT_SEED, 80)
        sizing = {a for a, _ in drtmle_f5.sizing_seeds(drtmle_f5.COHORT_SEED, 80)}
        for name, pairs in cohorts.items():
            assert not (sizing & {a for a, _ in pairs}), name


class TestTheFreeze:
    """The seeds, and the properties the whole design rests on."""

    def test_reserving_the_sizing_stream_leaves_the_cohorts_untouched(self) -> None:
        """``spawn`` gives child ``i`` the same state whatever ``n`` is.

        F4 depends on this and so does F5: reserving a third child must not move the two
        cohorts it is reserved beside.
        """
        seed = drtmle_f5.COHORT_SEED
        two = np.random.SeedSequence(seed).spawn(2)
        three = np.random.SeedSequence(seed).spawn(3)
        for left, right in zip(two, three[:2], strict=True):
            assert np.array_equal(left.generate_state(8), right.generate_state(8))

    def test_the_two_cohorts_are_disjoint(self) -> None:
        cohorts = drtmle_f5.cohort_seeds(drtmle_f5.COHORT_SEED, 80)
        selection = {a for a, _ in cohorts["selection"]}
        audit = {a for a, _ in cohorts["audit"]}
        assert not (selection & audit)

    def test_the_confirmation_batches_are_disjoint(self) -> None:
        a = {x for x, _ in drtmle_f5.confirm_seeds(drtmle_f5.CONFIRM_SEED_A, 500)}
        b = {x for x, _ in drtmle_f5.confirm_seeds(drtmle_f5.CONFIRM_SEED_B, 500)}
        assert not (a & b)

    def test_phase_one_does_not_reuse_f4s_draws(self) -> None:
        """Both F4 cohorts have been read, so neither inferential phase may see them again."""
        f4 = drtmle_construction.cohort_seeds(
            drtmle_construction.COHORT_SEED,
            max(drtmle_construction.SIZE_DRAWS.values()),
        )
        f4_data = {a for pairs in f4.values() for a, _ in pairs}
        ours = drtmle_f5.cohort_seeds(drtmle_f5.COHORT_SEED, 80)
        for name, pairs in ours.items():
            assert not (f4_data & {a for a, _ in pairs}), name

    def test_the_seed_family_is_unused(self) -> None:
        # 90-92M is C3c/E1b, 103-106M is E2R, 110M is F4.
        assert drtmle_f5.QUADRATURE_SEED == 120_000_000
        assert drtmle_f5.REFERENCE_SEED == 121_000_000
        assert drtmle_f5.QUADRATURE_SEED != drtmle_construction.QUADRATURE_SEED


class TestTheManifestCoversBothPhases:
    """The confirmation is frozen **before** phase 1 begins, not between the phases."""

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        return drtmle_f5.prereg(draws=8, replicates=16)

    def test_it_carries_both_phases(self, manifest: dict) -> None:
        assert manifest["phase1"]["cohorts"]
        assert manifest["phase2"]["batches"]["A"]["seed"] == drtmle_f5.CONFIRM_SEED_A
        assert manifest["phase2"]["batches"]["B"]["seed"] == drtmle_f5.CONFIRM_SEED_B

    def test_select_refuses_a_manifest_with_no_phase_two(self, manifest: dict) -> None:
        stripped = {k: v for k, v in manifest.items() if k != "phase2"}
        complaints = drtmle_f5.validate_prereg(stripped, phase="select")
        assert any("committed BEFORE phase 1 begins" in line for line in complaints)

    def test_a_moved_band_is_refused(self, manifest: dict) -> None:
        moved = json.loads(json.dumps(manifest))
        moved["rule"]["columns"][0]["band"] = 99.0
        complaints = drtmle_f5.validate_prereg(moved, phase="select")
        assert any("moved after the freeze" in line for line in complaints)

    def test_an_infinite_band_is_refused_by_name(self, manifest: dict) -> None:
        broken = json.loads(json.dumps(manifest))
        broken["rule"]["columns"][0]["band"] = None
        complaints = drtmle_f5.validate_prereg(broken, phase="select")
        assert any("third verdict unreachable" in line for line in complaints)

    def test_clause_three_is_recorded_not_feasible(self, manifest: dict) -> None:
        clause = manifest["phase2"]["clause_3"]
        assert clause["status"] == "not feasible"
        assert "Not read here" in clause["why"]
        assert "drtmle_stress" in clause["instead"]

    def test_the_stress_cells_exclusions_are_declared(self, manifest: dict) -> None:
        joined = " ".join(manifest["phase2"]["exclusions"])
        assert "NOT read against clause 4" in joined
        assert "is not a release number" in joined

    def test_a_foreign_resolved_map_is_refused(self, manifest: dict) -> None:
        foreign = json.loads(json.dumps(manifest))
        foreign["environment"]["resolved"]["boost-nested"]["regression"] = "HistGradientBoosting"
        complaints = drtmle_f5.validate_prereg(foreign, phase="select")
        assert any("resolved learner implementations differ" in line for line in complaints)


class TestTheDispatchIsSchedulingAndNotDesign:
    """Worker count and dispatch order may change throughput and must change no result."""

    def _payloads(self) -> list[drtmle_f5.Payload]:
        return [
            drtmle_f5.Payload(
                cohort="selection",
                cell=cell,
                n=n,
                data_seed=seed,
                fold_seed=seed + 1,
                arms=tuple(drtmle_f5.ARMS),
                quadrature_points=2_048,
                reference_points=8_192,
            )
            for cell in ("q-drift", "g-drift")
            for n in (600, 2_400)
            for seed in (1, 2, 3)
        ]

    def test_longest_first_puts_the_big_draws_in_front(self) -> None:
        """LPT: the long tasks need work to pack around them, so they go in first.

        Built cell-major and size-minor, every ``n = 600`` draw preceded every ``n = 2,400``
        one, so the long tasks were dispatched last and the run ended with a ragged tail.
        """
        payloads = self._payloads()
        assert payloads[0].n == 600, "the natural build order starts short"
        ordered = sorted(
            payloads, key=lambda p: (-drtmle_f5._expected_cost(p), p.cell, p.data_seed)
        )
        sizes = [p.n for p in ordered]
        assert sizes == sorted(sizes, reverse=True)
        assert ordered[0].n == 2_400

    def test_reordering_changes_no_draw(self) -> None:
        # `one_draw` is a pure function of its payload and the artefact is keyed by
        # (cohort, cell, n, data_seed, arm), so order is a hint and never a result.
        payloads = self._payloads()
        ordered = sorted(
            payloads, key=lambda p: (-drtmle_f5._expected_cost(p), p.cell, p.data_seed)
        )
        assert sorted(payloads, key=repr) == sorted(ordered, key=repr)

    def test_the_lean_companion_changes_no_estimate(self) -> None:
        """The ceiling's reference block may be withheld from the arms that never read it.

        A fit predicts at **every** companion row, so carrying the ceiling's 8,192-point
        reference block in one shared companion made all six arms pay for the one arm that uses
        it -- measured at 22.5 s against 16.1 s for a single ``glm-pooled`` fit, and worse on an
        arm whose reduced regressions refit many times.

        Splitting them is only legitimate because the companion contributes to no fit, no fold
        and no score, and because the evaluation blocks are built at the same points and
        scrambles either way.  That is a claim about estimates, so it is checked as one: bit
        for bit, not to a tolerance.
        """
        payload = drtmle_f5.Payload(
            cohort="sizing",
            cell="q-drift",
            n=300,
            data_seed=7,
            fold_seed=8,
            arms=tuple(drtmle_f5.ARMS),
            quadrature_points=512,
            reference_points=2_048,
        )
        dgp, settings = drtmle_f5._law_and_settings("q-drift", 300)
        frame, _ = dgp.sample(300, seed=7)
        lean = drtmle_f5._context(payload, dgp, with_reference=False)
        full = drtmle_f5._context(payload, dgp, with_reference=True)
        assert lean.stack.weights.size < full.stack.weights.size
        assert lean.reference_window is None and full.reference_window is not None

        produced = []
        for context in (lean, full):
            estimator = drtmle_f5.arm_estimator(
                "glm-pooled", settings, random_state=8, evaluation=context.stack.frame
            )
            fit = estimator.fit(frame, outcome="Y", treatment="A").single()
            produced.append(fit.estimates["ate"])
        assert produced[0].psi == produced[1].psi
        assert produced[0].std_error == produced[1].std_error

    def test_both_companions_share_their_evaluation_grid(self) -> None:
        # The remainder is integrated on these, so they have to be the same rows in the same
        # order or the two shapes would answer for different grids.
        payload = drtmle_f5.Payload(
            cohort="sizing",
            cell="q-drift",
            n=300,
            data_seed=7,
            fold_seed=8,
            arms=tuple(drtmle_f5.ARMS),
            quadrature_points=512,
            reference_points=2_048,
        )
        dgp, _ = drtmle_f5._law_and_settings("q-drift", 300)
        lean = drtmle_f5._context(payload, dgp, with_reference=False)
        full = drtmle_f5._context(payload, dgp, with_reference=True)
        assert lean.windows == full.windows
        assert np.array_equal(lean.scoring, full.scoring[: lean.scoring.size])
        shared = lean.stack.weights.size
        assert np.array_equal(lean.stack.weights, full.stack.weights[:shared])

    def test_the_default_worker_count_counts_logical_threads(self) -> None:
        """A worker occupies one hardware thread, so the default is logical and not physical.

        Measured at ``--jobs 9`` on the phase-1 cohort: 8.89 cores busy, 98.8% occupancy per
        worker, two OS threads each -- nothing contended, simply fewer workers than threads.
        """
        jobs = drtmle_f5.default_jobs()
        logical = os.cpu_count() or 2
        assert jobs == max(1, logical - 2)
        assert jobs >= 1

    def test_the_reserve_leaves_the_parent_room(self) -> None:
        # The parent runs the bootstrap, the JSON and the per-draw flush; a default equal to
        # the logical count would put that work on the critical path.
        if (os.cpu_count() or 2) > 2:
            assert drtmle_f5.default_jobs() < (os.cpu_count() or 2)


class TestTheProseAndThePredicateCannotComeApart:
    """The record's rule table **is** the code's, byte for byte.

    F4's other rule defect was that one line of the roadmap named two primary outcomes while the
    code banded one of them -- a disagreement no test could see because the prose and the
    predicate were separate artefacts.  Here the table is generated and the committed block is
    compared to it, so an edit to either side fails.
    """

    RECORD = Path(__file__).resolve().parents[2] / "docs" / "drtmle" / "terminal-experiment.md"

    def _committed_block(self) -> str:
        text = self.RECORD.read_text(encoding="utf-8")
        marker = "<!-- generated by benchmarks.drtmle_f5.format_rule_table(); do not hand-edit -->"
        assert marker in text, "the record lost the generated-table marker"
        after = text.split(marker, 1)[1]
        opening = after.index("```") + 3
        closing = after.index("```", opening)
        return after[opening:closing].strip("\n")

    def test_the_record_carries_the_generated_table(self) -> None:
        assert self._committed_block() == drtmle_f5.format_rule_table().strip("\n")

    def test_the_record_exists_before_phase_one(self) -> None:
        # The rule is frozen before the first fit, so the page that states it has to exist
        # before the first fit too.
        assert self.RECORD.exists()


class TestTheCeilingIsNotCalledAnOracleHere:
    def test_the_label_lives_in_one_place(self) -> None:
        assert drtmle_f5.CEILING_LABEL == "ceiling estimate"

    def test_no_reported_table_says_oracle(self) -> None:
        # On the exact law it is an oracle; at tier 2 it is a ceiling estimate, and F6 decides
        # whether that ever changes.
        assert "oracle" not in drtmle_f5.format_rule_table().lower()

    def test_the_frozen_rung_is_e2s(self) -> None:
        # spline(16): E2's shipped rung, and the only one with a resolved audit result in its
        # favour -- q-drift n=2,400 on q_r, where it measurably beat spline(8).
        assert drtmle_f5.CEILING_KNOTS == 16


class TestTheCeilingsTwoCrossfitsAreTheSameObject:
    """The reading that justifies seven arms instead of eight.

    **In the fast tier and deliberately not behind the ``slow`` marker.**  It fits the reference
    construction twice, which sounds expensive and is 8.6 s measured -- and it is the single
    claim that removes an arm the roadmap's matrix names.  A check that decides the shape of the
    experiment must not be one a default selection skips, because a skipped correctness check
    reads like a passing one.
    """

    def test_a_block_below_the_reference_budget_is_refused(self) -> None:
        # The floor is on *points* and not on rows, because Q_r's fit is masked to one arm.
        with pytest.raises(ValueError, match="masked to one arm"):
            drtmle_f5.ceiling_crossfit_reading(n=400, points=1_024)

    def test_every_reduced_array_is_bit_identical(self) -> None:
        rows = drtmle_f5.ceiling_crossfit_reading(n=400, points=2_048)
        assert rows
        failed = [(r["stage"], r["quantity"], r["worst"]) for r in rows if not r["identical"]]
        assert not failed, (
            f"{failed} differ between pooled and nested -- reduced_crossfit reaches the ceiling "
            "arm after all, and the eighth arm has to run"
        )
