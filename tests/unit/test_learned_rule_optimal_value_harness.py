"""Unit witnesses for the laws, fits, truth harness and controls of ``learned-rule-optimal-value``.

The study's verdicts read the error ``estimate - Psi_d0`` of each row, so a wrong truth, a wrong
blip, a fit without the known mechanism, or a control that does not compute what it names would
fail or pass a cell for the wrong reason. These checks recompute each piece another way. Every
fit draws from a throwaway seed, and no test asserts a bias, a coverage or an SE ratio.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from cleverly.exceptions import DataError
from tests.studies import _learned_rule_optimal_law as law
from tests.studies import learned_rule_optimal_value as study
from tests.studies import learned_rule_optimal_value_properties as properties
from tests.studies.evidence import property_failures, replication

base = law.base

pytestmark = pytest.mark.xdist_group("learned_rule_optimal_value_harness")

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_983_000
N = 600
#: The property draws of each cell the end-to-end checks fit.
CAP = 3


@cache
def _frame(scenario: str = "non_exceptional", n: int = N) -> pd.DataFrame:
    return law.draw(scenario, n, THROWAWAY)


@cache
def _measured(scenario: str = "non_exceptional", known_g: bool = True) -> law.Measured:
    return law.measured(
        _frame(scenario), scenario, law.LEARNERS[scenario](), THROWAWAY + 7, known_g=known_g
    )


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


# ------------------------------------------------------------------------------- the law


class TestTheLaws:
    def test_the_blips_are_the_declared_ones(self) -> None:
        w1 = np.array([-1.0, -0.1, 0.2, 0.5, 1.0])
        np.testing.assert_allclose(law.blip("non_exceptional", w1), 0.1 + w1)
        np.testing.assert_allclose(law.blip("wrong_limit", w1), 3.0 * (w1**2 - 0.5))
        np.testing.assert_allclose(
            law.blip("partially_exceptional", w1), [0.0, 0.0, 0.0, 0.3, 0.8], atol=1e-15
        )

    def test_non_exceptional_draws_rm30s_sample(self) -> None:
        """The primary law's draw is RM30's, byte for byte."""
        pd.testing.assert_frame_equal(
            law.draw("non_exceptional", 300, THROWAWAY + 1),
            base.draw("non_exceptional", 300, THROWAWAY + 1),
        )

    @pytest.mark.parametrize("scenario", list(law.BLIPS))
    def test_the_draw_follows_the_outcome_law(self, scenario: str) -> None:
        """A 200,000-row draw against ``Qbar0`` by arm, within 4 MC SEs; a deliberate mutation
        (the blip's sign flipped) misses on the treated arm."""
        frame = law.draw(scenario, 200_000, THROWAWAY + 2)
        w1, w2, a, y = (frame[c].to_numpy() for c in ("W1", "W2", "A", "Y"))
        for arm in (0.0, 1.0):
            hit = y[a == arm]
            q = law.qbar0(arm, w1[a == arm], w2[a == arm], scenario)
            assert abs(hit.mean() - q.mean()) < 4 * hit.std() / np.sqrt(hit.size)
        treated = a == 1.0
        flipped = law.base.qbar0(0.0, w1, w2, "non_exceptional")
        flipped = 1 / (1 + (1 / flipped - 1) * np.exp(law.blip(scenario, w1)))
        if scenario != "partially_exceptional":
            gap = abs(y[treated].mean() - flipped[treated].mean())
            assert gap > 8 * y[treated].std() / np.sqrt(treated.sum())

    def test_the_known_probabilities_are_g0(self) -> None:
        frame = _frame()
        known = law.known_probabilities(frame)
        np.testing.assert_allclose(known.sum(axis=1), 1.0, atol=1e-15)
        np.testing.assert_allclose(
            known[:, 1], base.g0(frame["W1"].to_numpy(), frame["W2"].to_numpy()), rtol=1e-15
        )

    def test_the_optimal_rule_ties_to_arm_0(self) -> None:
        """On the hinge law ``d0`` treats only ``W1 > 0.2``: the tie set goes to arm 0."""
        rule = law.optimal_rule(law.law_qbar("partially_exceptional"))
        for w2 in base.W2_LEVELS:
            np.testing.assert_array_equal(rule[w2], base.GRID > law.HINGE_KNOT)

    def test_the_value_and_variance_match_monte_carlo(self) -> None:
        """``Psi_d0`` and ``Var D*(d0)`` of each law against 400,000 draws with the known g."""
        for scenario in law.BLIPS:
            frame = law.draw(scenario, 400_000, THROWAWAY + 3)
            w1, w2, a, y = (frame[c].to_numpy() for c in ("W1", "W2", "A", "Y"))
            rule = (law.qbar0(1.0, w1, w2, scenario) > law.qbar0(0.0, w1, w2, scenario)) * 1.0
            q = law.qbar0(rule, w1, w2, scenario)
            treated = base.g0(w1, w2)
            g = np.where(rule > 0.5, treated, 1.0 - treated)
            curve = (a == rule) * (y - q) / g + q
            limit = law.law_limit(scenario)
            se = curve.std() / np.sqrt(curve.size)
            assert curve.mean() == pytest.approx(limit["truth"], abs=4 * se), scenario
            assert np.var(curve) == pytest.approx(limit["variance"], rel=0.02), scenario


# ------------------------------------------------------------------------- the learners


class TestTheLearners:
    def test_the_hinge_learner_adds_the_hinge_product(self) -> None:
        design = np.array([[1.0, 0.5, 1.0], [0.0, 0.5, 1.0], [1.0, -0.5, 0.0]])
        features = law.hinge_features(design)
        np.testing.assert_allclose(features[:, :6], law.rm30_features(design))
        np.testing.assert_allclose(features[:, 6], [0.3, 0.0, 0.0], atol=1e-15)

    def test_the_hinge_learner_clones_and_fits(self) -> None:
        frame = _frame("partially_exceptional")
        learner = clone(law.hinge_learner())
        design = frame[["A", "W1", "W2"]].to_numpy()
        model = learner.fit(design, frame["Y"].to_numpy().astype(int))
        assert model.named_steps["model"].coef_.shape == (1, 7)

    def test_each_laws_learner_limit(self) -> None:
        """Correct on ``non_exceptional`` and on the hinge law (C3f); on ``wrong_limit`` the
        declared learner's limit treats nobody. The mutation: the declared learner on the hinge
        law misses ``d0`` off the tie set."""
        for scenario in ("non_exceptional", "partially_exceptional"):
            qbar = law.law_qbar(scenario)
            limit = law.limit_blip(qbar, law.FEATURES[scenario])
            for w2 in base.W2_LEVELS:
                truth = qbar(1.0, base.GRID, w2) - qbar(0.0, base.GRID, w2)
                np.testing.assert_allclose(limit[w2], truth, atol=2e-5)
        limit = law.limit_blip(law.law_qbar("wrong_limit"))
        assert all((limit[w2] < 0).all() for w2 in base.W2_LEVELS)
        misfit = law.limit_blip(law.law_qbar("partially_exceptional"))
        assert max(np.abs(misfit[w2]).max() for w2 in base.W2_LEVELS) > 0.01


# ------------------------------------------------------------------------------- the fits


class TestTheFits:
    def test_the_estimator_is_the_declared_one(self) -> None:
        estimator = law.estimator(base.outcome_learner(), 5)
        assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
        assert estimator.cv_evaluation and estimator.targeting_scheme == "pooled"
        assert estimator.n_folds == 10 and estimator.g_bounds == "auto"
        assert estimator.repeats == 1 and not estimator.interventions
        assert list(estimator.estimands) == ["ey_optimal_rule"]

    def test_the_fit_reports_the_optimal_value_and_keys_p12_known(self) -> None:
        result = _measured().result
        assert tuple(result.estimates) == (law.ESTIMAND,)
        key = result.config.support_key
        assert key.family == frozenset({"P12"})
        assert "known_mechanism" in key.qualifiers
        assert key.splitting == ("fold_evaluated",)
        estimated = _measured(known_g=False).result.config.support_key
        assert "known_mechanism" not in estimated.qualifiers

    def test_the_known_g_fit_uses_the_supplied_probabilities(self) -> None:
        """The deliberate mutation: the estimated-g fit of the same draw and fold seed moves."""
        known, estimated = _measured(), _measured(known_g=False)
        assert known.result.estimates[law.ESTIMAND].psi != pytest.approx(
            estimated.result.estimates[law.ESTIMAND].psi, abs=1e-6
        )

    def test_a_known_g_fit_that_drops_the_mechanism_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: a known-g cell whose fit is handed no probabilities refuses."""
        monkeypatch.setattr(law, "known_probabilities", lambda frame: None)
        with pytest.raises(base.HarnessMismatch, match="known_mechanism"):
            law.measured(
                _frame(), "non_exceptional", base.outcome_learner(), THROWAWAY + 7, known_g=True
            )

    def test_every_row_is_checked(self) -> None:
        assert _measured().refit.rows_checked == N
        assert _measured().solver_warnings == 0


# ------------------------------------------------------------------------------ the truth


class TestTheTruth:
    def test_the_rule_value_is_the_fold_average_and_the_regret_its_gap(self) -> None:
        fitted = _measured()
        refit = fitted.refit
        assert len(refit.fold_values) == 10
        assert refit.rule_value == pytest.approx(np.mean(refit.fold_values), rel=1e-15)
        columns = law.harness_columns(fitted, "non_exceptional")
        truth = law.law_limit("non_exceptional")["truth"]
        assert columns["regret"] == pytest.approx(truth - refit.rule_value, rel=1e-15)
        assert columns["regret"] >= 0.0

    def test_the_regret_is_nonzero_at_the_wrong_limit(self) -> None:
        """The nonzero witness: the fold rules of ``wrong_limit`` treat almost nobody, so the
        regret sits near the quadrature gap."""
        fitted = _measured("wrong_limit")
        columns = law.harness_columns(fitted, "wrong_limit")
        gap, _, _ = law.target_gap(law.WRONG_LIMIT_AMPLITUDE)
        assert columns["regret"] > 0.5 * gap

    def test_the_disagreement_is_the_records(self) -> None:
        fitted = _measured()
        (split,) = fitted.result.extra["learned_rule"].splits
        assert law.harness_columns(fitted, "non_exceptional")["rule_disagreement"] == (
            split.rule_disagreement
        )

    def test_a_rule_that_differs_on_one_validation_row_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _measured().result
        honest = base.rule_of(result)

        def flipped(fit: Any) -> np.ndarray:
            rule = honest.copy()
            rule[11] = ~rule[11]
            return rule

        monkeypatch.setattr(base, "rule_of", flipped)
        with pytest.raises(base.HarnessMismatch, match="validation rows"):
            law.harness(result, base.outcome_learner(), "non_exceptional")

    def test_the_harness_refits_the_learner_it_is_given(self) -> None:
        """Mutation: the hinge fit's rules refit with the declared learner."""
        with pytest.raises(base.HarnessMismatch):
            law.harness(
                _measured("partially_exceptional").result,
                base.outcome_learner(),
                "partially_exceptional",
            )

    def test_the_rule_value_reads_the_law_it_is_given(self) -> None:
        """Mutation: the same fold rules integrated under another law's outcome regression."""
        result = _measured().result
        honest = law.harness(result, base.outcome_learner(), "non_exceptional").rule_value
        other = law.harness(result, base.outcome_learner(), "wrong_limit").rule_value
        assert abs(honest - other) > 0.01


# --------------------------------------------------------------------- the primary rows


class TestThePrimaryRows:
    def test_one_row_with_the_fixed_truth(self) -> None:
        record = base.smoke_record(study.STUDY)
        key = (study.PRIMARY_LAW, 0, N)
        frame = study._sample(record, key)
        rows = pd.DataFrame(study._primary(record, key, frame))
        assert list(rows["estimand"]) == [law.ESTIMAND]
        assert rows["truth"].iloc[0] == law.law_limit(study.PRIMARY_LAW)["truth"]
        covered = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
        np.testing.assert_array_equal(rows["covered"], covered.astype(int))
        assert (rows["rule_rows_checked"] == N).all()
        assert set(rows.columns) >= set(law.HARNESS_COLUMNS)

    def test_the_draw_carries_the_truth(self) -> None:
        _, truth = study.draw_from_seed(study.PRIMARY_LAW, 50, THROWAWAY)
        assert truth == {law.ESTIMAND: law.law_limit(study.PRIMARY_LAW)["truth"]}
        with pytest.raises(KeyError):
            study.draw_from_seed("wrong_limit", 50, THROWAWAY)

    def test_the_harness_artifact_keys_each_replication(self) -> None:
        rows = pd.DataFrame(
            {
                "scenario": [study.PRIMARY_LAW] * 2,
                "replicate": [1, 0],
                "truth": [0.577, 0.577],
                **{column: [0.0, 0.0] for column in law.HARNESS_COLUMNS},
            }
        )
        artifact = study.extra_artifacts(rows)[study.HARNESS_ARTIFACT]
        assert list(artifact["replicate"]) == [0, 1]


# --------------------------------------------------------------------- the property path


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        summary = properties.summarize_properties(rows)
        assert set(summary["cell"]) == declared

    def test_each_cell_draws_its_own_law_at_its_own_n(self) -> None:
        """The control runs at n = 2,500 and every other cell at 2,000, each against its law's
        ``Psi_d0``; the record's n is the primary's."""
        rows = _property_rows()
        for family, (scenario, _, n, suffix, role) in properties.SETTINGS.items():
            cell = rows.loc[rows["cell"] == f"{properties.LABELS[family]}__{suffix}"]
            assert set(cell["n"]) == {n}, family
            assert set(cell["role"]) == {role}, family
            assert set(cell["truth"]) == {law.law_limit(scenario)["truth"]}, family
        assert study.STUDY.n == study.PRIMARY_N == properties.CALIBRATION_N == 2_000
        assert study.CONTROL_N == 2_500

    def test_each_family_draws_its_own_samples(self) -> None:
        rows = _property_rows()
        positive = rows["cell"] == f"{study.CALIBRATION_LABEL}__{study.POSITIVE}"
        estimated = rows["property"] == study.ESTIMATED_FAMILY
        assert not set(rows.loc[positive, "estimate"]) & set(rows.loc[estimated, "estimate"])

    def test_the_target_verdict_reads_the_declared_floor(self) -> None:
        """Mutations on a hand-made summary: the control passes only below the floor."""
        floor = study.STUDY.margins.calibration_coverage[0]
        summary = pd.DataFrame(
            {
                "property": [study.TARGET_FAMILY] * 2,
                "cell": ["c0", "c1"],
                "role": ["control", "control"],
                "coverage_ci_upper": [floor - 1e-9, floor],
                "passed": [False, False],
            }
        )
        properties.target_verdicts(summary, floor=floor)
        assert list(summary["passed"]) == [True, False]
        summary.loc[0, "role"] = "positive"
        with pytest.raises(ValueError, match="not control"):
            properties.target_verdicts(summary, floor=floor)

    def test_the_report_publishes_the_regret_and_the_disagreement(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows).set_index("cell")
        for cell, group in rows.groupby("cell"):
            assert summary.loc[cell, "mean_regret"] == pytest.approx(group["regret"].mean())
            disagreement = group["rule_disagreement"]
            assert summary.loc[cell, "mean_rule_disagreement"] == pytest.approx(disagreement.mean())
            assert summary.loc[cell, "min_rule_disagreement"] == disagreement.min()
            assert summary.loc[cell, "max_rule_disagreement"] == disagreement.max()
        for family in study.DIAGNOSTIC_FAMILIES:
            (cell,) = study.PROPERTY_CELLS[family]
            assert summary.loc[cell, "passed"] and summary.loc[cell, "property_passed"]
            assert summary.loc[cell, "role"] == "diagnostic"

    def test_the_report_publishes_the_readings_against_the_rule_value(self) -> None:
        """Pinned on a fixed frame: the error is ``estimate - rule_value``, not the truth."""
        group = pd.DataFrame(
            {
                "truth": [9.0, 9.0, 9.0, 9.0],
                "estimate": [1.0, 2.0, 3.0, 6.0],
                "rule_value": [0.5, 1.5, 3.5, 4.0],
                "std_error": [0.3, 0.2, 0.3, 0.3],
            }
        )
        # error 0.5, 0.5, -0.5, 2.0: mean 0.625, SD 1.030776; covered while |error| <= 1.96 SE.
        readings = properties.rule_value_readings(group)
        spread = float(np.std([0.5, 0.5, -0.5, 2.0], ddof=1))
        assert spread == pytest.approx(1.0307764)
        assert readings == {
            "rule_value_standardized_bias": pytest.approx(0.625 / spread),
            "rule_value_coverage": 0.5,
            "rule_value_se_ratio": pytest.approx(0.275 / spread),
        }
        # Mutation: reading the error against the truth would move every column.
        shifted = properties.rule_value_readings(group.assign(rule_value=group["truth"]))
        assert shifted["rule_value_coverage"] == 0.0
        assert shifted["rule_value_standardized_bias"] < -1.0

    def test_every_cell_publishes_the_readings_against_the_rule_value(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows).set_index("cell")
        for cell, group in rows.groupby("cell"):
            expected = properties.rule_value_readings(group)
            for column, value in expected.items():
                assert summary.loc[cell, column] == pytest.approx(value, nan_ok=True), cell
        assert set(properties.OPTIMAL_SUMMARY) >= set(expected)

    def test_the_declared_rules_reproduce_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        own = summary.set_index("cell")
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        for family in ("interval_calibration", study.TARGET_FAMILY):
            rule = property_failures.rule_for(family, properties)
            assert rule is not None, family
            verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
            assert set(verdicts.index) == set(own.index[own["property"] == family])
            for cell in verdicts.index:
                assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        for family in study.DIAGNOSTIC_FAMILIES:
            assert property_failures.rule_for(family, properties) is None

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "non_exceptional_known_g", 1), 3),
            (("optimal_target_control", "wrong_limit_known_g", 1), 1),
            (("estimated_g_report", "non_exceptional_estimated_g", 1), 1),
            (("exceptional_law_report", "partially_exceptional_known_g", 1), 1),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost calibration draw costs its positive and its two derived controls."""
        cached = _property_rows()
        derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise DataError("a draw with one row in an arm")
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
