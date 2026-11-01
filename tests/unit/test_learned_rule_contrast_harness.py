"""Unit witnesses for the fits, the truth harness and the controls of ``learned-rule-contrast``.

The study's verdicts read the error ``estimate - truth`` of each row, so a wrong truth, a fit of
another regime, a report on other folds, or a control that does not compute what it names would
fail or pass a cell for the wrong reason. These checks recompute each piece another way. Every
fit draws from a throwaway seed, and no test asserts a bias, a coverage or an SE ratio.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from tests.studies import _learned_rule_contrast_law as law
from tests.studies import learned_rule_contrast as study
from tests.studies import learned_rule_contrast_properties as properties
from tests.studies.evidence import property_failures, replication

base = law.base

pytestmark = pytest.mark.xdist_group("learned_rule_contrast_harness")

#: Throwaway seeds, far from every declared and pilot seed. 403 rows in ten folds gives
#: unequal folds (40 and 41 rows), so the stacked weights ``n_v / n`` differ from ``1/V``.
THROWAWAY = 9_980_000
N = 403
#: The property draws of each cell the end-to-end checks fit.
CAP = 3


@cache
def _frame(scenario: str = "non_exceptional", n: int = N) -> pd.DataFrame:
    return base.draw(scenario, n, THROWAWAY + (scenario == "misspecified_limit"))


@cache
def _measured(scenario: str = "non_exceptional", n: int = N) -> law.Measured:
    return law.measured(_frame(scenario, n), scenario, base.outcome_learner(), THROWAWAY + 7)


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


# ------------------------------------------------------------------------------ the fits


class TestTheFits:
    def test_the_estimator_is_the_declared_one_under_each_report(self) -> None:
        for report, evaluated in law.REPORTS.items():
            estimator = law.estimator(base.outcome_learner(), 5, law.ARMS, cv_evaluation=evaluated)
            assert [regime.name for regime in estimator.interventions] == [
                "always 1",
                "always 0",
            ], report
            assert estimator.cv_evaluation is evaluated
            assert estimator.n_folds == base.N_FOLDS and estimator.g_bounds == "auto"
            assert estimator.targeting_scheme == "pooled" and estimator.simultaneous
            assert estimator.random_state == 5

    def test_each_report_reports_the_five_parameters_on_one_partition(self) -> None:
        fitted = _measured()
        assert set(fitted.results) == set(law.REPORTS)
        stacked, evaluated = fitted.results["stacked"], fitted.results["fold_evaluated"]
        for result in fitted.results.values():
            assert tuple(result.estimates) == tuple(law.TAGS.values())
            assert result.simultaneous is None
        assert law.same_partition(stacked, evaluated)
        assert fitted.refit.rows_checked == N

    def test_the_two_reports_differ_at_unequal_folds(self) -> None:
        """Nonzero witness: at 40 and 41 rows a fold the stacked point weights by ``n_v / n``."""
        fitted = _measured()
        name = law.TAGS["value"]
        gap = abs(
            fitted.results["stacked"].estimates[name].psi
            - fitted.results["fold_evaluated"].estimates[name].psi
        )
        assert gap > 1e-6

    def test_a_fit_on_another_partition_stops(self) -> None:
        first = law.fit(_frame(), base.outcome_learner(), THROWAWAY + 7, law.ARMS)
        other = law.fit(_frame(), base.outcome_learner(), THROWAWAY + 8, law.ARMS)
        assert law.same_partition(first, first)
        assert not law.same_partition(first, other)

    def test_a_fit_of_other_known_regimes_stops(self) -> None:
        """Mutation: the regimes in the other order put ``Static(0)`` in column 1."""
        swapped = law.fit(_frame(), base.outcome_learner(), THROWAWAY + 7, (0, 1))
        with pytest.raises(base.HarnessMismatch, match="not Static"):
            law.check_known_columns(swapped)
        law.check_known_columns(_measured().results["fold_evaluated"])


# ----------------------------------------------------------------------------- the truth


class TestTheTruth:
    @pytest.mark.parametrize("scenario", law.SCENARIOS)
    def test_the_fold_evaluated_truth_is_the_rm30_harness(self, scenario: str) -> None:
        fitted = _measured(scenario)
        result = fitted.results["fold_evaluated"]
        truths = law.truths(result, fitted.refit, scenario)
        rm30 = base.harness(result, base.outcome_learner(), scenario)
        assert truths["value"][0] == pytest.approx(rm30.truth, rel=1e-14)
        assert truths["value"][1] == pytest.approx(rm30.oracle_se, rel=1e-12)
        assert fitted.refit.fold_values == pytest.approx(rm30.fold_values, rel=1e-14)
        for arm in law.ARMS:
            known = law.static_value(arm, scenario)
            assert truths[f"known_{arm}"][0] == known
            assert truths[f"contrast_{arm}"][0] == pytest.approx(rm30.truth - known, rel=1e-13)
            column = 1 + law.ARMS.index(arm)
            legacy = law.contrast_truth(
                result, base.outcome_learner(), scenario, arm, column=column
            )
            assert truths[f"contrast_{arm}"][0] == pytest.approx(legacy.contrast, rel=1e-13)

    def test_the_stacked_truth_weights_the_folds_by_their_rows(self) -> None:
        fitted = _measured()
        result = fitted.results["stacked"]
        sizes = np.array(fitted.refit.sizes, dtype=float)
        assert len(set(sizes)) == 2
        expected = float(np.dot(sizes / sizes.sum(), fitted.refit.fold_values))
        truths = law.truths(result, fitted.refit, "non_exceptional")
        assert truths["value"][0] == pytest.approx(expected, rel=1e-14)
        evaluated = law.truths(fitted.results["fold_evaluated"], fitted.refit, "non_exceptional")
        assert truths["value"][0] != evaluated["value"][0]

    def test_a_record_whose_fold_weights_differ_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the stacked fit read with ``1/V`` folds refuses."""
        result = _measured().results["stacked"]
        (split,) = result.extra["learned_rule"].splits
        equal = tuple(np.full(len(split.fold_weights), 1 / len(split.fold_weights)))
        monkeypatch.setattr(type(split), "fold_weights", property(lambda self: equal))
        with pytest.raises(base.HarnessMismatch, match="fold weights"):
            law.report_weights(result)

    def test_a_rule_that_differs_on_one_row_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _measured().results["fold_evaluated"]
        honest = base.rule_of(result)

        def flipped(fit: object) -> np.ndarray:
            rule = honest.copy()
            rule[11] = ~rule[11]
            return rule

        monkeypatch.setattr(base, "rule_of", flipped)
        with pytest.raises(base.HarnessMismatch, match="1 of"):
            law.harness(result, base.outcome_learner(), "non_exceptional")

    def test_the_curve_variances_match_monte_carlo(self) -> None:
        """``Var D_K`` and ``Var(D_d - D_K)`` against draws of ``(W, A, Y)`` at a fold rule."""
        fitted = _measured()
        result = fitted.results["fold_evaluated"]
        rule = base.fold_rule(result, base.outcome_learner(), 0)
        rng = np.random.default_rng(THROWAWAY + 40)
        draws = 400_000
        w1, w2 = rng.uniform(-1, 1, draws), rng.binomial(1, 0.5, draws).astype(float)
        a = rng.binomial(1, base.g0(w1, w2)).astype(float)
        y = rng.binomial(1, base.qbar0(a, w1, w2, "non_exceptional")).astype(float)

        def curve(arm: np.ndarray) -> np.ndarray:
            q = base.qbar0(arm, w1, w2, "non_exceptional")
            g = np.where(arm > 0.5, base.g0(w1, w2), 1 - base.g0(w1, w2))
            return (a == arm) * (y - q) / g + q

        learned = curve(np.asarray(rule(np.column_stack([w1, w2])), dtype=float))
        grid = {
            level: rule(np.column_stack([base.GRID, np.full_like(base.GRID, level)]))
            for level in base.W2_LEVELS
        }
        for arm in law.ARMS:
            known_curve = curve(np.full(draws, float(arm)))
            known, difference = law.curve_variances(grid, float(arm), "non_exceptional")
            assert np.var(known_curve) == pytest.approx(known, rel=0.02)
            assert np.var(learned - known_curve) == pytest.approx(difference, rel=0.02)

    def test_a_contrast_curve_is_not_the_sum_of_its_parts(self) -> None:
        """Nonzero witness: the contrast's variance is neither the sum nor zero, because the
        residual terms cancel where the rule assigns the known arm."""
        rule = base.fixed_rule_grid(lambda w1, _: w1 > -0.1)
        known, difference = law.curve_variances(rule, 1.0, "non_exceptional")
        value = base.rule_value(rule, "non_exceptional")[1]
        assert 0.1 < difference < value + known - 0.1


# --------------------------------------------------------------------------- the controls


class TestTheControls:
    def test_the_untargeted_value_is_the_rm30_plug_in(self) -> None:
        result = _measured().results["fold_evaluated"]
        assert law.untargeted(result, "value") == pytest.approx(
            base.untargeted_estimate(result), rel=1e-14
        )

    def test_the_untargeted_contrast_is_the_difference_of_the_plug_ins(self) -> None:
        for result in _measured().results.values():
            nuisance = result.nuisance
            control, treated = nuisance.outcome.levels
            arms = {
                1: nuisance.scaler.unscale_levels(nuisance.outcome.arms[treated]),
                0: nuisance.scaler.unscale_levels(nuisance.outcome.arms[control]),
            }
            weights = law.report_weights(result)
            for arm in law.ARMS:
                known = float(
                    np.dot(
                        weights,
                        [np.mean(np.asarray(arms[arm])[test]) for _, test in nuisance.folds],
                    )
                )
                assert law.untargeted(result, f"known_{arm}") == pytest.approx(known, rel=1e-14)
                assert law.untargeted(result, f"contrast_{arm}") == pytest.approx(
                    law.untargeted(result, "value") - known, rel=1e-12
                )
                assert law.untargeted(result, f"contrast_{arm}") != pytest.approx(
                    float(result.estimates[law.TAGS[f"contrast_{arm}"]].psi), abs=1e-6
                )

    @pytest.mark.parametrize("tag", law.CONTRAST_TAGS)
    def test_the_contrast_se_lies_between_the_two_bounds(self, tag: str) -> None:
        """The two control SEs are the correlation-one and independence values, and the joint
        SE differs from both (W-c4 of the implementation)."""
        result = _measured().results["fold_evaluated"]
        learned, known = law.contrast_parts(result, tag)
        assert learned == result.estimates[law.TAGS["value"]].std_error
        assert law.correlation_one_se(result, tag) == abs(learned - known)
        assert law.independent_se(result, tag) == pytest.approx(np.sqrt(learned**2 + known**2))
        joint = float(result.estimates[law.TAGS[tag]].std_error)
        assert law.correlation_one_se(result, tag) < joint / 3
        assert joint < law.independent_se(result, tag) / 1.2

    def test_a_value_is_no_contrast(self) -> None:
        with pytest.raises(ValueError, match="not a contrast"):
            law.contrast_parts(_measured().results["fold_evaluated"], "known_1")


# ------------------------------------------------------------------------ the primary rows


class TestThePrimaryRows:
    def test_one_row_per_parameter_with_its_own_truth(self) -> None:
        record = base.smoke_record(study.STUDY)
        key = (study.PRIMARY_LAW, 0, N)
        frame = study._sample(record, key)
        rows = pd.DataFrame(study._primary(record, key, frame))
        assert list(rows["estimand"]) == list(law.TAGS.values())
        assert rows["truth"].nunique() == len(law.TAGS)
        covered = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
        np.testing.assert_array_equal(rows["covered"], covered.astype(int))
        assert (rows["rule_rows_checked"] == N).all()
        contrast = rows.set_index("estimand")
        for arm in law.ARMS:
            assert contrast.loc[law.TAGS[f"contrast_{arm}"], "truth"] == pytest.approx(
                contrast.loc[law.TAGS["value"], "truth"]
                - contrast.loc[law.TAGS[f"known_{arm}"], "truth"]
            )


# --------------------------------------------------------------------- the property path


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["n"] == properties.CALIBRATION_N).all()
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        summary = properties.summarize_properties(rows)
        assert set(summary["cell"]) == declared

    @pytest.mark.parametrize("scenario", law.SCENARIOS)
    def test_the_controls_rescale_the_positive_of_the_same_fit(self, scenario: str) -> None:
        rows = _property_rows()
        family = properties.LAW_FAMILY[scenario]
        for report in study.REPORTS:
            stem = f"{scenario}_{report}"
            for tag in law.CONTRAST_TAGS:
                positive = rows.loc[rows["cell"] == f"{stem}_{tag}__declared"].set_index(
                    "replicate"
                )
                assert set(positive["property"]) == {family}
                for cell, role in (
                    (f"{stem}_{tag}__{study.CORRELATION_ONE}", "control"),
                    (f"{stem}_{tag}__{study.INDEPENDENT}", "diagnostic"),
                ):
                    other = rows.loc[rows["cell"] == cell].set_index("replicate")
                    assert set(other["role"]) == {role}
                    np.testing.assert_array_equal(other["estimate"], positive["estimate"])
                    np.testing.assert_array_equal(other["truth"], positive["truth"])
                    assert (other["std_error"] != positive["std_error"]).all()
        if scenario == study.COVERAGE_LAW:
            for label in study.COVERAGE_LABELS:
                positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
                control = rows.loc[rows["cell"] == f"{label}__{study.COVERAGE_CONTROL}"]
                control = control.set_index("replicate")
                np.testing.assert_allclose(
                    control["std_error"], study.COVERAGE_SHRINK * positive["std_error"], rtol=1e-15
                )
                np.testing.assert_array_equal(control["estimate"], positive["estimate"])

    def test_the_necessity_control_is_the_untargeted_fit_of_the_same_draw(self) -> None:
        rows = _property_rows()
        rows = rows.loc[rows["property"] == study.NECESSITY_FAMILY]
        targeted, untargeted = (
            rows.loc[rows["cell"] == cell].set_index("replicate") for cell in study.NECESSITY_CELLS
        )
        assert (set(targeted["role"]), set(untargeted["role"])) == ({"positive"}, {"control"})
        np.testing.assert_array_equal(untargeted["truth"], targeted["truth"])
        np.testing.assert_array_equal(untargeted["std_error"], targeted["std_error"])
        assert (untargeted["estimate"] != targeted["estimate"]).all()

    def test_the_necessity_family_fits_its_own_draws_and_learner(self) -> None:
        rows = _property_rows()
        necessity = rows.loc[rows["property"] == study.NECESSITY_FAMILY, "truth"]
        calibration = rows.loc[rows["property"] == "interval_calibration", "truth"]
        assert not set(necessity) & set(calibration)

    def test_the_verdicts_read_the_declared_thresholds(self) -> None:
        """Mutations of the coverage and covariance verdicts on a hand-made summary: each
        passes on one side of its threshold and fails on the other."""
        floor = study.STUDY.margins.calibration_coverage[0]
        summary = pd.DataFrame(
            {
                "property": [study.COVERAGE_FAMILY] * 4 + [study.COVARIANCE_FAMILY] * 2,
                "cell": [f"c{i}" for i in range(6)],
                "role": ["positive", "positive", "control", "control", "control", "control"],
                "bias_equivalent": [True, True, False, False, False, False],
                "coverage_ci_lower": [floor, floor - 1e-9, 0.0, 0.0, 0.0, 0.0],
                "coverage_ci_upper": [
                    1.0,
                    1.0,
                    floor - 1e-9,
                    floor,
                    study.CORRELATION_ONE_CEILING - 1e-9,
                    study.CORRELATION_ONE_CEILING,
                ],
                "passed": [False] * 6,
            }
        )
        properties.coverage_verdicts(summary, floor=floor)
        properties.covariance_verdicts(summary, ceiling=study.CORRELATION_ONE_CEILING)
        assert list(summary["passed"]) == [True, False, True, False, True, False]
        summary.loc[0, "bias_equivalent"] = False
        properties.coverage_verdicts(summary, floor=floor)
        assert not summary.loc[0, "passed"]

    def test_the_declared_rules_reproduce_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        own = summary.set_index("cell")
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        for family in (
            "interval_calibration",
            study.COVERAGE_FAMILY,
            study.COVARIANCE_FAMILY,
            study.NECESSITY_FAMILY,
        ):
            rule = property_failures.rule_for(family, properties)
            assert rule is not None, family
            verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
            assert set(verdicts.index) == set(own.index[own["property"] == family])
            for cell in verdicts.index:
                assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        assert property_failures.rule_for(study.INDEPENDENT_FAMILY, properties) is None
        diagnostic = summary.loc[summary["property"] == study.INDEPENDENT_FAMILY]
        assert set(diagnostic["role"]) == {"diagnostic"}
        assert diagnostic["passed"].all() and diagnostic["property_passed"].all()

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "non_exceptional", 1), 38),
            (("conservative_coverage", "misspecified_limit", 1), 28),
            (("targeting_necessity", "non_exceptional_drop_w1", 1), 2),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost ``non_exceptional`` draw costs its ten positives, their twenty derived
        controls and the eight contrast controls and diagnostics; a ``misspecified_limit`` draw
        its ten positives, their ten coverage controls and its eight contrast cells."""
        cached = _property_rows()
        derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise DataError("a draw with one treated row")
            cells = {
                cell.split("/", 1)[1]
                for cell in properties._cells_of(key)
                if cell.split("/", 1)[1] not in derived
            }
            mine = cached.loc[(cached["replicate"] == replicate) & cached["cell"].isin(cells)]
            return mine.to_dict("records")

        monkeypatch.setattr(properties, "_replication", worker)
        with replication.recording(study.STUDY.implementation) as recorded:
            rows = properties.generate_property_rows(
                n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
            )
        records = property_failures.frame(recorded)
        assert records is not None and len(records) == 1
        rows = property_failures.stamp(rows, records)
        property_failures.reconcile(rows, records, derived=properties.DERIVED_CELLS)
        stamped = rows.groupby("cell")["failed_replicates"].max()
        hit = {
            cell.split("/", 1)[1]
            for cell in properties._cells_of((lost[:2],))  # type: ignore[arg-type]
        }
        assert len(hit) == cost
        assert {cell for cell, count in stamped.items() if count} == hit
        assert (rows.loc[rows["cell"].isin(hit)].groupby("cell").size() == CAP - 1).all()
