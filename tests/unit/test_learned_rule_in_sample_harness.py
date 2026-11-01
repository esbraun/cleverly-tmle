"""Unit witnesses for the learners, the fits, the truth harness and the controls of
``learned-rule-in-sample``.

The study's verdicts read the error ``estimate - truth`` of each row, so a wrong truth, a learner
that keys another family, or a control that does not fit what it names would fail or pass a cell
for the wrong reason. These checks recompute each piece another way. Every fit draws from a
throwaway seed, and no test asserts a bias, a coverage or an SE ratio.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly._support import learner_family
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.exceptions import DataError
from cleverly.inference import influence_variance
from tests.studies import _learned_rule_in_sample_law as law
from tests.studies import learned_rule_in_sample as study
from tests.studies import learned_rule_in_sample_properties as properties
from tests.studies.evidence import property_failures, replication

base = law.base

pytestmark = pytest.mark.xdist_group("learned_rule_in_sample_harness")

N = 600


@cache
def _frame(scenario: str = "non_exceptional", n: int = N, seed: int = 7) -> pd.DataFrame:
    return base.draw(scenario, n, seed)


@cache
def _measured(
    kind: str = "declared", scenario: str = "non_exceptional", n: int = N, seed: int = 7
) -> tuple[Any, base.Truth, law.Counts]:
    return law.measured(_frame(scenario, n, seed), scenario, kind, 3)


def _design(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = result.data
    return (
        np.asarray(data.treatment_design(), dtype=float),
        np.asarray(data.outcome, dtype=float),
        np.asarray(data.weights, dtype=float),
    )


# ------------------------------------------------------------------------- the learners


class TestTheLearners:
    @pytest.mark.parametrize(
        ("kind", "pipeline"),
        [("declared", base.outcome_learner), ("drop_w1", base.drop_w1_learner)],
    )
    def test_each_marked_class_is_its_rm30_pipeline(self, kind: str, pipeline: Any) -> None:
        """The class changes what ``learner_family`` reads and nothing the fit computes."""
        frame = _frame()
        design = frame[["A", "W1", "W2"]].to_numpy(dtype=float)
        marked = law.LEARNERS[kind]().fit(design, frame["Y"].to_numpy())
        plain = pipeline().fit(design, frame["Y"].to_numpy())
        np.testing.assert_array_equal(marked.predict_proba(design), plain.predict_proba(design))
        assert learner_family(law.LEARNERS[kind]()) == "parametric"
        assert learner_family(pipeline()) == "adaptive"

    def test_the_tree_interpolates_and_is_clipped(self) -> None:
        """In sample the tree's probabilities sit at the clip bounds, on the side of each Y."""
        frame = _frame()
        design = frame[["A", "W1", "W2"]].to_numpy(dtype=float)
        y = frame["Y"].to_numpy()
        tree = law.ClippedInterpolatingTree().fit(design, y)
        treated = tree.predict_proba(design)[:, 1]
        np.testing.assert_array_equal(treated, np.where(y > 0.5, law.CLIP[1], law.CLIP[0]))
        assert learner_family(tree) == "adaptive"
        assert not hasattr(type(tree), "cleverly_learner_family")
        again = law.ClippedInterpolatingTree().fit(design, y)
        np.testing.assert_array_equal(again.predict_proba(design), tree.predict_proba(design))


# ------------------------------------------------------------------------------ the fits


class TestTheFits:
    def test_the_estimator_is_the_declared_in_sample_one(self) -> None:
        estimator = law.estimator(law.InteractionLogistic(), 5)
        assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
        assert not estimator.cross_fit and not estimator.cv_evaluation
        assert estimator.repeats == 1 and estimator.g_bounds == "auto"
        assert estimator.random_state == 5

    @pytest.mark.parametrize("kind", ["declared", "without_w1_main", "drop_w1", "interpolating"])
    def test_each_fit_keys_its_declared_families(self, kind: str) -> None:
        result, _, counts = _measured(kind)
        options = dict(result.config.support_key.options)
        assert result.config.support_key.splitting == ("in_sample",)
        for name, family in law.FAMILIES[kind].items():
            assert options[name] == family
        # The warning follows the key: the interpolating fit warns once, the others never.
        assert counts.in_sample == (1 if kind == "interpolating" else 0)
        (split,) = result.extra["learned_rule"].splits
        assert split.scheme == "in_sample" and split.fold_weights == (1.0,)

    def test_a_fit_that_keys_another_family_stops(self) -> None:
        """The deliberate mutation: the tree's fit read as a declared one is refused."""
        result, _, _ = _measured("interpolating")
        with pytest.raises(RuntimeError, match="keys"):
            law.check_key(result, "declared")

    def test_the_support_row_options_are_the_declared_families(self) -> None:
        assert law.FAMILIES["declared"] == study.SUPPORT_OPTIONS
        assert law.FAMILIES["without_w1_main"] == law.FAMILIES["drop_w1"] == study.SUPPORT_OPTIONS

    def test_the_stored_variance_is_the_centred_rule(self) -> None:
        """``{n (n - 1)}^-1 sum (phi - phibar)^2``, ``phibar`` zero to solver tolerance."""
        result, _, _ = _measured()
        estimate = result.estimates[law.ESTIMAND]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        n = curve.size
        assert abs(float(np.mean(curve))) < 1e-8
        by_hand = float(np.sum((curve - curve.mean()) ** 2)) / (n * (n - 1))
        assert estimate.variance == pytest.approx(by_hand, rel=1e-12)
        assert estimate.variance == pytest.approx(influence_variance(curve), rel=1e-12)
        plan = float(np.sum(curve**2)) / n**2
        assert estimate.variance == pytest.approx(plan * n / (n - 1), rel=1e-8)
        assert estimate.variance != pytest.approx(plan, rel=1e-4)

    def test_the_untargeted_estimate_is_the_initial_plug_in_mean(self) -> None:
        result, _, _ = _measured("without_w1_main")
        nuisance = result.nuisance
        control, treated = nuisance.outcome.levels
        q0 = np.asarray(nuisance.scaler.unscale_levels(nuisance.outcome.arms[control]))
        q1 = np.asarray(nuisance.scaler.unscale_levels(nuisance.outcome.arms[treated]))
        by_hand = float(np.mean(np.where(base.rule_of(result), q1, q0)))
        assert law.untargeted_estimate(result) == pytest.approx(by_hand, abs=1e-15)
        # The nonzero witness: the fluctuation of the W1-dropping model moves the estimate.
        assert abs(float(result.estimates[law.ESTIMAND].psi) - by_hand) > 1e-3


# -------------------------------------------------------------------------- the truth


class TestTheTruth:
    @pytest.mark.parametrize(
        ("kind", "scenario"),
        [
            ("declared", "non_exceptional"),
            ("declared", "misspecified_limit"),
            ("interpolating", "non_exceptional"),
        ],
    )
    def test_the_truth_is_the_value_of_the_one_full_sample_rule(
        self, kind: str, scenario: str
    ) -> None:
        """Refitted here on every row, independently of the harness, and checked on every row."""
        result, truth, _ = _measured(kind, scenario)
        design, outcome, weights = _design(result)
        model = fit_on_rows(
            law.LEARNERS[kind](), design, outcome, weights, np.arange(N), "classification", None
        )
        covariates = np.asarray(result.data.covariates, dtype=float)
        assert np.array_equal(base._blip(model, covariates) > 0.0, base.rule_of(result))
        value, variance = base.rule_value(law.grid_rule(model), scenario)
        assert truth.rows_checked == N
        assert truth.truth == pytest.approx(value, abs=1e-12)
        assert truth.oracle_se == pytest.approx(np.sqrt(variance / N), rel=1e-12)

    def test_a_refit_on_half_the_rows_is_another_rule(self) -> None:
        """The full-sample branch is load-bearing: a fold-like refit misses a row (rule L4)."""
        result, _, _ = _measured()
        design, outcome, weights = _design(result)
        half = fit_on_rows(
            law.InteractionLogistic(),
            design,
            outcome,
            weights,
            np.arange(0, N, 2),
            "classification",
            None,
        )
        covariates = np.asarray(result.data.covariates, dtype=float)
        assert np.any((base._blip(half, covariates) > 0.0) != base.rule_of(result))


# ----------------------------------------------------------------- the target distinction


class TestTheTargetDistinction:
    def test_the_fold_average_rebuilds_from_the_declared_folds(self) -> None:
        result, truth, _ = _measured(n=2_000)
        average = law.fold_average(result, "non_exceptional", 11)
        design, outcome, weights = _design(result)
        full = law.grid_rule(
            fit_on_rows(
                law.InteractionLogistic(),
                design,
                outcome,
                weights,
                np.arange(2_000),
                "classification",
                None,
            )
        )
        values, shares = [], []
        folds = law.diagnostic_folds(2_000, 11)
        assert len(list(folds)) == law.DIAGNOSTIC_FOLDS
        for train, _ in folds:
            model = law.InteractionLogistic().fit(design[train], outcome[train])
            rule = law.grid_rule(model)
            values.append(base.rule_value(rule, "non_exceptional")[0])
            shares.append(np.mean(np.concatenate([rule[w] != full[w] for w in base.W2_LEVELS])))
        assert average.truth == pytest.approx(np.mean(values), abs=1e-12)
        assert average.grid_disagreement == pytest.approx(np.mean(shares), abs=1e-12)
        # The nonzero witness: the fold rules are other rules, with another value.
        assert average.grid_disagreement > 0.0
        assert average.truth != pytest.approx(truth.truth, abs=1e-9)

    def test_another_fold_seed_is_another_partition(self) -> None:
        result, _, _ = _measured(n=2_000)
        first = law.fold_average(result, "non_exceptional", 11)
        second = law.fold_average(result, "non_exceptional", 12)
        assert first.truth != second.truth


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
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == declared

    def test_only_the_interpolating_control_warns(self) -> None:
        rows = _property_rows()
        control = rows["cell"] == f"non_exceptional__{study.INTERPOLATING}"
        assert (rows.loc[control, "in_sample_warnings"] == 1).all()
        assert (rows.loc[~control, "in_sample_warnings"] == 0).all()

    def test_the_interpolating_control_fits_the_same_draw_with_its_own_truth(self) -> None:
        rows = _property_rows()
        positive = rows.loc[rows["cell"] == "non_exceptional__declared"].set_index("replicate")
        control = rows.loc[rows["cell"] == f"non_exceptional__{study.INTERPOLATING}"].set_index(
            "replicate"
        )
        assert (control["role"] == "control").all()
        assert (control["truth"] != positive["truth"]).all()
        assert (control["estimate"] != positive["estimate"]).all()

    def test_the_reporting_rows_read_the_declared_fit_against_the_fold_average(self) -> None:
        rows = _property_rows()
        reporting = rows.loc[rows["property"] == study.REPORTING_FAMILY].set_index("replicate")
        positive = rows.loc[rows["cell"] == "non_exceptional__declared"].set_index("replicate")
        assert set(reporting["role"]) == {"diagnostic"}
        np.testing.assert_array_equal(reporting["estimate"], positive["estimate"])
        np.testing.assert_array_equal(reporting["std_error"], positive["std_error"])
        np.testing.assert_array_equal(reporting["full_sample_truth"], positive["truth"])
        assert (reporting["truth"] != positive["truth"]).all()
        others = rows.loc[rows["property"] != study.REPORTING_FAMILY]
        assert others[list(properties.REPORTING_COLUMNS)].isna().all().all()
        summary = properties.summarize_properties(rows)
        reported = summary.loc[summary["property"] == study.REPORTING_FAMILY]
        assert len(reported) == 1 and set(reported["role"]) == {"diagnostic"}
        assert reported["passed"].all() and reported["property_passed"].all()
        assert reported[list(properties.REPORTING_SUMMARY)].notna().all().all()
        gap = reporting["full_sample_truth"] - reporting["truth"]
        assert float(reported["mean_truth_gap"].iloc[0]) == pytest.approx(gap.mean())
        assert property_failures.rule_for(study.REPORTING_FAMILY, properties) is None

    @pytest.mark.parametrize(
        ("family", "roles"),
        [
            ("targeting_necessity", ("positive", "control")),
            ("weak_limit_necessity", ("diagnostic", "diagnostic")),
        ],
    )
    def test_the_necessity_control_is_the_untargeted_fit_of_the_same_draw(
        self, family: str, roles: tuple[str, str]
    ) -> None:
        rows = _property_rows()
        rows = rows.loc[rows["property"] == family]
        stem = properties.NECESSITY_KINDS[family][2]
        targeted = rows.loc[rows["cell"] == f"{stem}__targeted"].set_index("replicate")
        untargeted = rows.loc[rows["cell"] == f"{stem}__untargeted"].set_index("replicate")
        assert (set(targeted["role"]), set(untargeted["role"])) == ({roles[0]}, {roles[1]})
        np.testing.assert_array_equal(untargeted["truth"], targeted["truth"])
        np.testing.assert_array_equal(untargeted["std_error"], targeted["std_error"])
        assert (untargeted["estimate"] != targeted["estimate"]).all()

    def test_the_two_necessity_families_fit_different_learners(self) -> None:
        """The declared family and the weak-limit diagnostic draw their own samples."""
        rows = _property_rows()
        declared = rows.loc[rows["property"] == "targeting_necessity", "truth"]
        weak = rows.loc[rows["property"] == "weak_limit_necessity", "truth"]
        assert not set(declared) & set(weak)
        summary = properties.summarize_properties(rows)
        weak_summary = summary.loc[summary["property"] == "weak_limit_necessity"]
        assert weak_summary["passed"].all() and weak_summary["property_passed"].all()
        assert weak_summary["weak_limit_displacement"].notna().all()
        assert property_failures.rule_for("weak_limit_necessity", properties) is None

    def test_the_shared_rules_reproduce_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        own = summary.set_index("cell")
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        for family in ("interval_calibration", "targeting_necessity"):
            rule = property_failures.rule_for(family, properties)
            assert rule is not None, family
            verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
            assert set(verdicts.index) == set(own.index[own["property"] == family])
            for cell in verdicts.index:
                assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        assert property_failures.undeclared_verdict_writers(properties) == []

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "non_exceptional__declared", 1), 5),
            (("interval_calibration", "misspecified_limit__declared", 1), 3),
            (("targeting_necessity", "non_exceptional", 1), 2),
            (("weak_limit_necessity", "non_exceptional_drop_w1", 1), 2),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost ``non_exceptional`` draw costs its positive, its two derived controls, the
        interpolating control and the reporting row."""
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
