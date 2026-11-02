"""Unit witnesses for the laws, the truth harness and the controls of ``learned-rule-stacked``.

The study's verdicts read the error ``estimate - truth`` of each row, so a wrong truth, a wrong
fold weighting or a wrong row-level SE in the clustered control would fail or pass a cell for
the wrong reason. These checks recompute each piece another way. Every fit draws from a
throwaway seed at a small size, and no test asserts a bias, a coverage or an SE ratio.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from cleverly.exceptions import DataError
from cleverly.inference import influence_variance
from tests.studies import _learned_rule_compositions_law as compositions
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_stacked_law as law
from tests.studies import learned_rule_stacked as study
from tests.studies import learned_rule_stacked_properties as properties
from tests.studies.evidence import property_failures, replication

pytestmark = pytest.mark.xdist_group("learned_rule_stacked_harness")

N = 600


@cache
def _fit(scenario: str, n: int = N, seed: int = 7, cv_evaluation: bool = False) -> Any:
    frame = law.draw(scenario, n, seed)
    return law.fit(frame, scenario, base.outcome_learner(), 3, cv_evaluation=cv_evaluation)


# ----------------------------------------------------------------------------- the laws


class TestTheLaws:
    def test_rows_is_rm30s_draw_and_weighted_is_the_compositions_draw(self) -> None:
        pd.testing.assert_frame_equal(
            law.draw("rows", 400, 5), base.draw("non_exceptional", 400, 5)
        )
        pd.testing.assert_frame_equal(
            law.draw("weighted", 400, 5), compositions.draw("weighted", 400, 5)
        )

    def test_a_clustered_draw_holds_n_over_ten_clusters_of_five_to_fifteen(self) -> None:
        frame = law.draw("clustered", 2_000, 11)
        sizes = frame.groupby("cluster").size()
        assert sizes.size == 200
        assert sizes.min() >= law.SIZE_FLOOR and sizes.max() <= law.SIZE_FLOOR + law.SIZE_TRIALS
        assert (np.diff(frame["cluster"].to_numpy()) >= 0).all()
        with pytest.raises(ValueError, match="multiple"):
            law.draw("clustered", 2_005, 11)

    def test_the_size_increases_with_the_cluster_effect(self) -> None:
        """The nonzero witness of the outcome-dependent size: the size rises with the cluster
        effect that shifts every row's outcome logit, and its mean is 10."""
        rng = np.random.default_rng(3)
        z = rng.normal(size=200_000)
        sizes = law.SIZE_FLOOR + rng.binomial(law.SIZE_TRIALS, law.size_probability(z))
        assert np.corrcoef(z, sizes)[0, 1] > 0.5
        assert sizes.mean() == pytest.approx(law.MEAN_SIZE, abs=0.02)

    def test_the_row_regression_integrates_the_size_biased_effect(self) -> None:
        for a, w1, w2 in ((0.0, -0.7, 0.0), (1.0, 0.2, 1.0), (1.0, 0.9, 0.0)):
            eta = float(law.linear_predictor("clustered", a, np.array([w1]), w2)[0])

            def biased(z: float, eta: float = eta) -> float:
                return float(law.mean_size(np.array([z]))[0] * norm.pdf(z))

            numerator, _ = quad(
                lambda z, eta=eta: expit(eta + law.SIGMA_U * z) * biased(z), -np.inf, np.inf
            )
            denominator, _ = quad(biased, -np.inf, np.inf)
            got = float(law.qbar0("clustered", a, np.array([w1]), w2)[0])
            assert got == pytest.approx(numerator / denominator, abs=1e-10)

    def test_without_size_dependence_the_row_law_is_the_cluster_law(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation: a size slope of zero makes the row masses the Hermite
        masses, and the regression moves, so the size bias is load-bearing."""
        biased = law.qbar0("clustered", 1.0, base.GRID, 0.0)
        monkeypatch.setattr(law, "KAPPA", 0.0)
        _, masses = law._hermite()
        np.testing.assert_allclose(law.row_masses(), masses, rtol=0, atol=1e-15)
        assert np.max(np.abs(law.qbar0("clustered", 1.0, base.GRID, 0.0) - biased)) > 0.01


class TestTheClusterMoments:
    def _rule(self) -> dict[float, np.ndarray]:
        return {w2: 0.1 * (1 + w2) < base.GRID for w2 in base.W2_LEVELS}

    def test_the_value_is_the_row_mean_of_the_cluster_means(self) -> None:
        rule = self._rule()
        moments = law.cluster_moments(rule)
        direct = 0.0
        for w2 in base.W2_LEVELS:
            q = law.qbar0("clustered", rule[w2].astype(float), base.GRID, w2)
            direct += 0.5 * np.trapezoid(q, base.GRID) / 2.0
        assert moments.value == pytest.approx(direct, abs=1e-13)

    def test_the_cluster_variance_rebuilds_from_simulated_cluster_totals(self) -> None:
        """``E(T^2) / E(N)`` against cluster totals of exact cluster means, row noise left out
        on both sides: the between-cluster part of the variance."""
        rule = self._rule()
        moments = law.cluster_moments(rule)
        nodes, masses = law._hermite()
        between = float(
            np.sum(
                masses
                * law.second_size_moment(nodes)
                * (moments.cluster_means - moments.value) ** 2
            )
            / np.sum(masses * law.mean_size(nodes))
        )
        rng = np.random.default_rng(5)
        z = rng.normal(size=400_000)
        sizes = law.SIZE_FLOOR + rng.binomial(law.SIZE_TRIALS, law.size_probability(z))
        fine = np.linspace(-7.0, 7.0, 2_801)
        curve = np.interp(z, fine, law.cluster_mean_curve(rule, fine))
        totals = sizes * (curve - moments.value)
        assert float(np.mean(totals**2) / np.mean(sizes)) == pytest.approx(between, rel=0.01)
        assert moments.cluster_variance > moments.row_variance > between > 0.0


# -------------------------------------------------------------------------- the harness


class TestTheHarness:
    @pytest.mark.parametrize("scenario", law.SCENARIOS)
    def test_every_row_is_checked_and_the_truth_weights_the_fold_masses(
        self, scenario: str
    ) -> None:
        result = _fit(scenario)
        truth = law.harness(result, base.outcome_learner(), scenario)
        assert truth.rows_checked == result.data.n
        assert len(truth.fold_values) == law.N_FOLDS
        weights = np.asarray(result.data.weights, dtype=float)
        masses = np.array([weights[np.asarray(t)].sum() for _, t in result.nuisance.folds])
        np.testing.assert_allclose(truth.fold_weights, masses / masses.sum(), rtol=0, atol=1e-15)
        assert truth.truth == pytest.approx(np.dot(masses / masses.sum(), truth.fold_values))
        (split,) = result.extra["learned_rule"].splits
        assert split.scheme == "stacked"
        np.testing.assert_allclose(split.fold_weights, truth.fold_weights, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("scenario", ["weighted", "clustered"])
    def test_the_fold_weights_are_not_one_over_v(self, scenario: str) -> None:
        """The nonzero witness: under weights or clusters the 1/V truth is another number."""
        truth = law.harness(_fit(scenario), base.outcome_learner(), scenario)
        assert np.max(np.abs(np.asarray(truth.fold_weights) - 1 / law.N_FOLDS)) > 1e-3
        assert abs(truth.truth - float(np.mean(truth.fold_values))) > 1e-6

    def test_record_weights_that_differ_stop_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation: the harness computes ``W_v / W`` itself and checks it."""
        result = _fit("weighted")
        monkeypatch.setattr(law, "fold_masses", lambda result: np.ones(law.N_FOLDS))
        with pytest.raises(base.HarnessMismatch, match="fold weights"):
            law.harness(result, base.outcome_learner(), "weighted")

    def test_an_unweighted_refit_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _fit("weighted", 2_000)
        original = law.fit_on_rows

        def unweighted(learner: Any, design: Any, outcome: Any, weights: Any, *rest: Any) -> Any:
            return original(learner, design, outcome, np.ones_like(weights), *rest)

        monkeypatch.setattr(law, "fit_on_rows", unweighted)
        with pytest.raises(base.HarnessMismatch):
            law.harness(result, base.outcome_learner(), "weighted")

    def test_the_reporting_fit_shares_the_folds_rules_and_fold_values(self) -> None:
        stacked = _fit("clustered")
        evaluated = _fit("clustered", cv_evaluation=True)
        assert law.same_partition(stacked, evaluated)
        first = law.harness(stacked, base.outcome_learner(), "clustered")
        second = law.harness(evaluated, base.outcome_learner(), "clustered")
        assert first.fold_values == second.fold_values
        assert second.fold_weights == (1 / law.N_FOLDS,) * law.N_FOLDS
        assert second.truth == pytest.approx(np.mean(first.fold_values), abs=1e-15)

    def test_another_fold_seed_is_not_the_same_partition(self) -> None:
        frame = law.draw("clustered", N, 7)
        other = law.fit(frame, "clustered", base.outcome_learner(), 4, cv_evaluation=True)
        assert not law.same_partition(_fit("clustered"), other)


# ----------------------------------------------------------- the stored variance rules


class TestTheStoredVariance:
    @pytest.mark.parametrize("scenario", ["rows", "weighted"])
    def test_rows_store_the_centred_rule(self, scenario: str) -> None:
        """``{n (n - 1)}^-1 sum (phi - phibar)^2``, with ``phibar`` zero to solver tolerance."""
        estimate = _fit(scenario).estimates[law.ESTIMAND]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        n = curve.size
        assert abs(float(np.mean(curve))) < 1e-8
        by_hand = float(np.sum((curve - curve.mean()) ** 2)) / (n * (n - 1))
        assert estimate.variance == pytest.approx(by_hand, rel=1e-12)
        assert estimate.variance == pytest.approx(influence_variance(curve), rel=1e-12)
        # The factor n / (n - 1) separates it from the plan's n^-2 sum phi^2 form.
        plan = float(np.sum(curve**2)) / n**2
        assert estimate.variance == pytest.approx(plan * n / (n - 1), rel=1e-8)
        assert estimate.variance != pytest.approx(plan, rel=1e-4)

    def test_clusters_store_the_cluster_total_rule(self) -> None:
        """``J var(T_j; ddof=1) / n^2`` of the cluster totals, normal reference at J = 200."""
        result = _fit("clustered", 2_000)
        estimate = result.estimates[law.ESTIMAND]
        frame = law.draw("clustered", 2_000, 7)
        totals = pd.Series(estimate.influence_curve).groupby(frame["cluster"].to_numpy()).sum()
        clusters, n = totals.size, result.data.n
        assert clusters == 200
        by_hand = clusters * float(np.var(totals.to_numpy(), ddof=1)) / n**2
        assert estimate.variance == pytest.approx(by_hand, rel=1e-12)
        assert estimate.reference_df is None
        assert law.row_variance_se(result) ** 2 != pytest.approx(by_hand, rel=0.05)

    def test_the_row_variance_se_is_the_se_without_clusters(self) -> None:
        """Clusters of one row fold like rows, so the control's SE is the fit without ``id=``.

        At clusters of one row the cluster-total rule is the centred row rule too, so the
        clustered cell's test above is the one that separates the two rules.
        """
        frame = law.draw("rows", N, 11).assign(cluster=np.arange(N))
        clustered = law.fit(frame, "clustered", base.outcome_learner(), 5)
        rows = law.fit(frame, "rows", base.outcome_learner(), 5)
        assert [list(t) for _, t in clustered.nuisance.folds] == [
            list(t) for _, t in rows.nuisance.folds
        ]
        row_se = float(rows.estimates[law.ESTIMAND].std_error)
        assert law.row_variance_se(clustered) == pytest.approx(row_se, rel=1e-10)
        assert float(clustered.estimates[law.ESTIMAND].std_error) == pytest.approx(
            row_se, rel=1e-10
        )


class TestTheVarianceTerms:
    @pytest.mark.parametrize("scenario", ["rows", "weighted"])
    def test_the_terms_rebuild_the_stored_variance(self, scenario: str) -> None:
        result = _fit(scenario)
        estimate = result.estimates[law.ESTIMAND]
        cross, between = law.variance_terms(result)
        curve = np.asarray(estimate.influence_curve, dtype=float)
        weights = np.asarray(result.data.weights, dtype=float)
        (split,) = result.extra["learned_rule"].splits
        plug_ins = np.empty(result.data.n)
        for (_, test), value in zip(result.nuisance.folds, split.fold_estimates, strict=True):
            plug_ins[np.asarray(test)] = value
        within = float(np.sum(weights**2 * (curve / weights + estimate.psi - plug_ins) ** 2))
        n = curve.size
        assert within * (1.0 + cross + between) == pytest.approx(
            estimate.variance * n * (n - 1), rel=1e-8
        )
        assert between > 0.0 and cross != 0.0

    def test_the_fold_evaluated_fit_is_not_read(self) -> None:
        """The terms describe the stacked curve; the reporting row writes none."""
        stacked = law.measured(law.draw("clustered", N, 7), "clustered", base.outcome_learner(), 3)
        row = properties.reporting_row(stacked, law.draw("clustered", N, 7), 3, (0, N, 1))
        assert np.isnan(row["cross_term"]) and np.isnan(row["between_term"])


# ---------------------------------------------------------------------- the property path

CAP = 2


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["n"] == properties.CALIBRATION_N).all()
        assert (rows["rule_rows_checked"] == rows["rows"]).all()
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == declared

    def test_the_controls_share_the_positive_estimate_and_truth(self) -> None:
        rows = _property_rows()
        positive = rows.loc[rows["cell"] == "clustered__declared"].set_index("replicate")
        control = rows.loc[rows["cell"] == f"clustered__{study.ROW_VARIANCE}"].set_index(
            "replicate"
        )
        assert (control["role"] == "control").all()
        np.testing.assert_array_equal(control["estimate"], positive["estimate"])
        np.testing.assert_array_equal(control["truth"], positive["truth"])
        assert (control["std_error"] < positive["std_error"]).all()

    def test_the_reporting_rows_pair_the_stacked_fit_and_state_no_verdict(self) -> None:
        rows = _property_rows()
        reporting = rows.loc[rows["property"] == study.REPORTING_FAMILY].set_index("replicate")
        positive = rows.loc[rows["cell"] == "clustered__declared"].set_index("replicate")
        assert set(reporting["role"]) == {"diagnostic"}
        np.testing.assert_array_equal(reporting["stacked_estimate"], positive["estimate"])
        np.testing.assert_array_equal(reporting["stacked_truth"], positive["truth"])
        assert (reporting["estimate"] != positive["estimate"]).all()
        others = rows.loc[rows["property"] != study.REPORTING_FAMILY]
        assert others[list(properties.PAIRED_COLUMNS)].isna().all().all()
        summary = properties.summarize_properties(rows)
        reported = summary.loc[summary["property"] == study.REPORTING_FAMILY]
        assert len(reported) == 1 and set(reported["role"]) == {"diagnostic"}
        assert reported["passed"].all() and reported["property_passed"].all()
        assert reported[list(properties.REPORTING_SUMMARY)].notna().all().all()
        error = (reporting["estimate"] - reporting["truth"]) - (
            reporting["stacked_estimate"] - reporting["stacked_truth"]
        )
        assert float(reported["paired_error_difference"].iloc[0]) == pytest.approx(error.mean())
        assert property_failures.rule_for(study.REPORTING_FAMILY, properties) is None

    def test_the_shared_rule_reproduces_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        rule = property_failures.rule_for("interval_calibration", properties)
        assert rule is not None
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
        own = summary.set_index("cell")
        calibration = own.loc[own["property"] == "interval_calibration"]
        assert set(verdicts.index) == set(calibration.index)
        for cell in verdicts.index:
            assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        assert property_failures.undeclared_verdict_writers(properties) == []

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "rows__declared", 1), 3),
            (("interval_calibration", "clustered__declared", 1), 5),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost clustered draw costs its positive, its two derived controls, the row-variance
        control and the reporting row."""
        cached = _property_rows()
        derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise DataError("validation fold 2 holds one cluster")
            setting = label.split("__", 1)[0]
            mine = cached.loc[
                (cached["replicate"] == replicate)
                & cached["cell"].str.startswith(f"{setting}__")
                & ~cached["cell"].isin(derived)
            ]
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
