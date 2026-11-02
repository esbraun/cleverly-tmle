"""Unit witnesses for the law, fits, truth harness and controls of ``learned-rule-intermediate``.

The study's verdicts read the error ``estimate - truth`` of each row at each level, so a wrong
truth, a rule read at the wrong level, a curve that is not the fit's, or a control that does not
compute what it names would fail or pass a cell for the wrong reason. These checks recompute each
piece another way. Every fit draws from a throwaway seed, and no test asserts a bias, a coverage
or an SE ratio.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from tests.studies import _learned_rule_intermediate_law as law
from tests.studies import learned_rule_intermediate as study
from tests.studies import learned_rule_intermediate_properties as properties
from tests.studies.evidence import property_failures, replication

base = law.base

pytestmark = pytest.mark.xdist_group("learned_rule_intermediate_harness")

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_984_000
N = 600
#: The property draws of each cell the end-to-end checks fit.
CAP = 3


@cache
def _frame(n: int = N) -> pd.DataFrame:
    return law.draw(n, THROWAWAY)


@cache
def _measured(learner: str = "declared", known: bool = False) -> law.Measured:
    return law.measured(_frame(), learner, THROWAWAY + 7, known=known)


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


# ------------------------------------------------------------------------------- the law


class TestTheLaw:
    def test_the_first_draws_are_rm30s(self) -> None:
        """``W1, W2, A`` are RM30's draws from the same seed, and ``Z`` is drawn after them."""
        mine = law.draw(400, 123)
        rm30 = base.draw("non_exceptional", 400, 123)
        for column in ("W1", "W2", "A"):
            np.testing.assert_array_equal(mine[column], rm30[column])
        assert list(mine.columns) == ["W1", "W2", "A", "Z", "Y"]

    def test_level_zero_is_rm30s_outcome_regression(self) -> None:
        for w2 in base.W2_LEVELS:
            for a in (0.0, 1.0):
                np.testing.assert_allclose(
                    law.qbar0(a, 0.0, base.GRID, w2),
                    base.qbar0(a, base.GRID, w2, "non_exceptional"),
                    rtol=1e-15,
                )

    def test_the_draw_follows_the_law(self) -> None:
        """A large throwaway draw: the intermediate's and the outcome's frequencies by cell."""
        frame = law.draw(400_000, 77)
        w1, a, z = frame["W1"].to_numpy(), frame["A"].to_numpy(), frame["Z"].to_numpy()
        assert float(np.mean(z)) == pytest.approx(float(np.mean(law.q0(1.0, a, w1))), abs=3e-3)
        expected = np.where(
            z > 0.5,
            law.qbar0(a, 1.0, w1, frame["W2"].to_numpy()),
            law.qbar0(a, 0.0, w1, frame["W2"].to_numpy()),
        )
        other = np.where(
            z > 0.5,
            law.qbar0(a, 0.0, w1, frame["W2"].to_numpy()),
            law.qbar0(a, 1.0, w1, frame["W2"].to_numpy()),
        )
        y = frame["Y"].to_numpy()
        for arm in (0.0, 1.0):
            for level in law.LEVELS:
                cell = (a == arm) & (z == level)
                assert float(y[cell].mean()) == pytest.approx(
                    float(expected[cell].mean()), abs=5e-3
                )
                # Mutation: an outcome drawn at the other level of Z misses the cell mean.
                assert abs(float(y[cell].mean()) - float(other[cell].mean())) > 0.03

    @pytest.mark.parametrize("z", law.LEVELS)
    def test_the_value_and_variance_match_monte_carlo(self, z: float) -> None:
        """``Psi_z(d)`` and ``Var D*_z(d)`` of the known rule against a throwaway Monte Carlo."""
        rng = np.random.default_rng(5)
        m = 400_000
        w1 = rng.uniform(-1.0, 1.0, m)
        w2 = rng.binomial(1, 0.5, m).astype(float)
        a = rng.binomial(1, law.g0(w1, w2)).astype(float)
        zz = rng.binomial(1, law.q0(1.0, a, w1)).astype(float)
        p = np.where(zz > 0.5, law.qbar0(a, 1.0, w1, w2), law.qbar0(a, 0.0, w1, w2))
        y = rng.binomial(1, p).astype(float)
        d = (w1 > 0.0).astype(float)
        g = np.where(d > 0.5, law.g0(w1, w2), 1.0 - law.g0(w1, w2))
        q = law.q0(z, d, w1)
        q_rule = law.qbar0(d, z, w1, w2)
        curve = (a == d) * (zz == z) / (g * q) * (y - law.qbar0(a, z, w1, w2)) + q_rule
        value, variance = law.rule_value(law.known_grid(), z)
        assert float(np.mean(q_rule)) == pytest.approx(value, abs=3e-3)
        assert float(np.var(curve)) == pytest.approx(variance, rel=0.03)

    def test_the_two_levels_have_different_optimal_rules(self) -> None:
        limits = [law.level_limit(z) for z in law.LEVELS]
        assert limits[0]["treated"] == pytest.approx(0.55, abs=1e-3)
        assert limits[1]["treated"] == pytest.approx(0.25, abs=1e-3)
        assert limits[0]["disagreement"] == pytest.approx(0.30, abs=1e-3)


class TestTheLearners:
    def test_the_declared_learner_is_rm30s_on_four_columns(self) -> None:
        model = law.declared_learner().fit(
            _frame()[["A", "W1", "W2", "Z"]].to_numpy(), _frame()["Y"].astype(int)
        )
        assert model.named_steps["model"].coef_.shape == (1, 10)

    def test_the_drop_w1_learner_reads_no_w1(self) -> None:
        frame = _frame()
        design = frame[["A", "W1", "W2", "Z"]].to_numpy()
        model = law.drop_w1_learner().fit(design, frame["Y"].astype(int))
        moved = design.copy()
        moved[:, 1] += 0.5
        np.testing.assert_array_equal(model.predict_proba(design), model.predict_proba(moved))
        assert model.named_steps["model"].coef_.shape == (1, 7)

    def test_the_drop_w1_limit_is_the_cell_means(self) -> None:
        """A large throwaway draw's cell means of ``(A, Z, W2)`` against the quadrature."""
        frame = law.draw(400_000, 78)
        for z in law.LEVELS:
            limit = law.drop_w1_limit(z)
            plug_in = 0.0
            for w2 in base.W2_LEVELS:
                arm = limit["treated"] > 0.0
                cell = frame[(frame["W2"] == w2) & (frame["Z"] == z) & (frame["A"] == float(arm))]
                plug_in += 0.5 * float(cell["Y"].mean())
            assert plug_in == pytest.approx(limit["plug_in"], abs=6e-3)


# ------------------------------------------------------------------------------ the fits


class TestTheFits:
    def test_the_estimator_is_the_declared_one(self) -> None:
        learned = law.estimator(law.declared_learner(), 3)
        assert learned.learned_rule is not None and learned.learned_rule.loss == "plug_in"
        assert (learned.n_folds, learned.cv_evaluation) == (10, True)
        assert (learned.targeting_scheme, learned.g_bounds) == ("pooled", "auto")
        known = law.estimator(law.declared_learner(), 3, known=True)
        assert known.learned_rule is None

    def test_each_level_reports_the_value_and_keys_the_intermediate(self) -> None:
        fitted = _measured()
        assert tuple(fitted.results) == law.LEVELS
        for z in law.LEVELS:
            key = fitted.results[z].config.support_key
            assert key.missing == ("intermediate",) and key.family == frozenset({"P11"})
            assert key.splitting == ("fold_evaluated",)
        known = _measured(known=True)
        assert known.results[0.0].config.support_key.family == frozenset({"P5"})

    def test_every_row_is_checked_at_each_level(self) -> None:
        for fitted in (_measured(), _measured(known=True), _measured("drop_w1")):
            for z in law.LEVELS:
                assert fitted.truths[z].rows_checked == N

    def test_the_levels_learn_different_rules(self) -> None:
        fitted = _measured()
        rules = [base.rule_of(fitted.results[z]) for z in law.LEVELS]
        assert float(np.mean(rules[0] != rules[1])) > 0.03


class TestTheTruth:
    def test_the_truth_is_the_fold_average_at_each_level(self) -> None:
        fitted = _measured()
        for z in law.LEVELS:
            truth = fitted.truths[z]
            assert truth.truth == pytest.approx(np.mean(truth.fold_values), abs=1e-15)
            assert len(truth.fold_values) == law.N_FOLDS

    def test_the_known_rule_truth_is_its_fixed_value(self) -> None:
        fitted = _measured(known=True)
        for z in law.LEVELS:
            assert fitted.truths[z].truth == pytest.approx(
                law.level_limit(z)["known_truth"], abs=1e-12
            )

    def test_the_truth_reads_the_level(self) -> None:
        """Mutation: each level's fold rules integrated at the other level move the truth."""
        fitted = _measured()
        result = fitted.results[1.0]
        refit = law.harness(fitted.results, law.declared_learner())
        assert refit[1.0].truth == fitted.truths[1.0].truth
        rules = law.fold_rules(result.nuisance)
        assert rules is not None
        assert fitted.truths[1.0].truth != fitted.truths[0.0].truth

    def test_a_rule_that_differs_on_one_validation_row_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule L4: a refit rule one validation row off raises."""
        original = law.level_blip

        def off_by_one(model: Any, z: float, covariates: np.ndarray) -> np.ndarray:
            out = original(model, z, covariates).copy()
            if covariates.shape[0] == N:
                out[0] = -out[0] if out[0] != 0.0 else 1.0
            return out

        monkeypatch.setattr(law, "level_blip", off_by_one)
        with pytest.raises(law.HarnessMismatch):
            law.harness(_measured().results, law.declared_learner())

    def test_the_curve_is_the_fits_and_the_mutation_drops_the_factor(self) -> None:
        """The exactness gate holds at each level, and dropping ``1{Z = z} / q`` shrinks the SE."""
        fitted = _measured()
        for z in law.LEVELS:
            result = fitted.results[z]
            with_factor, without = law.curve_variances(result, z)
            assert with_factor == pytest.approx(
                float(result.estimates[law.ESTIMAND].variance), rel=law.EXACTNESS_TOLERANCE
            )
            assert np.sqrt(without) == fitted.truths[z].dropped_std_error
            assert np.sqrt(without) < 0.9 * float(result.estimates[law.ESTIMAND].std_error)

    def test_a_curve_that_is_not_the_fits_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The exactness gate is read: a longhand variance off by 1e-6 raises."""
        original = law.curve_variances

        def off(result: Any, z: float) -> tuple[float, float]:
            with_factor, without = original(result, z)
            return with_factor * (1.0 + 1e-6), without

        monkeypatch.setattr(law, "curve_variances", off)
        with pytest.raises(law.HarnessMismatch, match="longhand variance"):
            law.harness(_measured().results, law.declared_learner())


# ----------------------------------------------------------------------- the primary rows


class TestThePrimaryRows:
    def test_one_row_per_level_from_one_fit(self) -> None:
        record = base.smoke_record(study.STUDY)
        key = (tuple(record.scenarios), 0, N)
        rows = study._primary(record, key, _frame())
        assert [row["scenario"] for row in rows] == ["cde_z0", "cde_z1"]
        assert {row["estimand"] for row in rows} == {law.ESTIMAND}
        assert rows[0]["estimate"] != rows[1]["estimate"]
        for row in rows:
            assert row["covered"] == int(row["ci_lower"] <= row["truth"] <= row["ci_upper"])

    def test_both_scenarios_draw_one_sample(self) -> None:
        a, _ = study.draw_scenario("cde_z0", 50, 3)
        b, _ = study.draw_scenario("cde_z1", 50, 3)
        pd.testing.assert_frame_equal(a, b)

    def test_the_harness_artifact_keys_each_replication_and_level(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "scenario": s,
                    "replicate": r,
                    "truth": 0.5,
                    **dict.fromkeys(law.HARNESS_COLUMNS, 1),
                }
                for r in (1, 0)
                for s in ("cde_z1", "cde_z0")
            ]
        )
        (frame,) = study.extra_artifacts(rows).values()
        assert list(frame["scenario"]) == ["cde_z0", "cde_z0", "cde_z1", "cde_z1"]
        assert list(frame["replicate"]) == [0, 1, 0, 1]


# ------------------------------------------------------------------------ the property path


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        assert set(rows["n"]) == {2_000}
        summary = properties.summarize_properties(rows)
        assert set(summary["cell"]) == declared

    def test_the_term_dropped_control_is_the_positives_estimate_with_the_dropped_se(self) -> None:
        rows = _property_rows().set_index(["cell", "replicate"])
        for label in study.LEARNED_LABELS:
            positive = rows.loc[f"{label}__{study.POSITIVE}"]
            control = rows.loc[f"{label}__{study.TERM_DROPPED}"]
            np.testing.assert_array_equal(control["estimate"], positive["estimate"])
            np.testing.assert_array_equal(control["truth"], positive["truth"])
            np.testing.assert_array_equal(control["std_error"], positive["dropped_std_error"])
            assert (control["std_error"] < positive["std_error"]).all()
            assert set(control["role"]) == {"control"}

    def test_the_known_cells_read_the_fixed_truth(self) -> None:
        rows = _property_rows()
        for z, label in zip(law.LEVELS, study.KNOWN_LABELS, strict=True):
            cell = rows.loc[rows["cell"] == f"{label}__{study.POSITIVE}"]
            np.testing.assert_allclose(cell["truth"], law.level_limit(z)["known_truth"], rtol=1e-14)

    def test_the_necessity_control_is_the_untargeted_plug_in(self) -> None:
        rows = _property_rows().set_index(["cell", "replicate"])
        for label in study.NECESSITY_LABELS:
            targeted = rows.loc[f"{label}__targeted"]
            untargeted = rows.loc[f"{label}__untargeted"]
            np.testing.assert_array_equal(targeted["truth"], untargeted["truth"])
            np.testing.assert_array_equal(targeted["std_error"], untargeted["std_error"])
            assert (targeted["estimate"] != untargeted["estimate"]).all()

    def test_each_draw_label_draws_its_own_samples(self) -> None:
        rows = _property_rows()
        learned = rows["cell"] == f"learned_z0__{study.POSITIVE}"
        known = rows["cell"] == f"known_z0__{study.POSITIVE}"
        assert not set(rows.loc[learned, "estimate"]) & set(rows.loc[known, "estimate"])

    def test_the_declared_rules_reproduce_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        own = summary.set_index("cell")
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        for family in study.PROPERTY_CELLS:
            rule = property_failures.rule_for(family, properties)
            assert rule is not None, family
            verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
            assert set(verdicts.index) == set(own.index[own["property"] == family])
            for cell in verdicts.index:
                assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "learned", 1), 8),
            (("interval_calibration", "known", 1), 8),
            (("targeting_necessity", "drop_w1", 1), 4),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost calibration draw costs both levels' positives, derived and direct controls."""
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
