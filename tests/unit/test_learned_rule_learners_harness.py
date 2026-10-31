"""Unit witnesses for the fits, the truth harness and the controls of ``learned-rule-learners``.

The study's verdicts read the error ``estimate - truth`` of each row, its fold-locality control
reads a rule learned on the validation rows, and its value gain reads a paired plug-in fit. A
control whose rule came from the training rows, a plug-in fit on other folds, or a value gain
read off the wrong fit would pass or fail a cell, or report a gain, for the wrong reason. These
checks rebuild each piece another way, and deliberate mutations must stop the replication.

Every fit draws from a throwaway seed at a small size, and no test asserts a bias, a coverage or
an SE ratio of the study.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.interventions._rule_learners import FoldRule, pseudo_outcome
from cleverly.learners._fitting import as_target, fit_learner
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_learners_law as law
from tests.studies import learned_rule_learners as study
from tests.studies import learned_rule_learners_properties as properties
from tests.studies.evidence import property_failures, replication

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_993_000
N = 600


@cache
def _measured(loss: str) -> tuple[pd.DataFrame, law.Measured]:
    frame = base.draw(law.LAW, N, THROWAWAY + law.LOSSES.index(loss))
    return frame, law.measured(frame, loss, THROWAWAY + 10, control=True)


LOSSES = pytest.mark.parametrize("loss", law.LOSSES)


class TestThePairedFits:
    @LOSSES
    def test_the_declared_fit_is_the_declared_rule_learner(self, loss: str) -> None:
        _, fitted = _measured(loss)
        rule = fitted.result.estimator.learned_rule
        assert rule.loss == loss
        assert fitted.control.estimator.learned_rule.loss == loss
        assert fitted.truth.rows_checked == N
        assert fitted.control_truth.rows_checked == N

    @LOSSES
    def test_the_value_gain_reads_the_plug_in_fit_of_the_same_folds(self, loss: str) -> None:
        frame, fitted = _measured(loss)
        plug_in, truth, _ = base.measured(frame, base.outcome_learner(), law.LAW, THROWAWAY + 10)
        assert law._same_folds(fitted.result, plug_in)
        assert law.value_gain_columns(fitted) == {
            "plug_in_estimate": float(plug_in.estimates[law.ESTIMAND].psi),
            "plug_in_truth": truth.truth,
        }
        assert tuple(law.value_gain_columns(fitted)) == law.VALUE_GAIN_COLUMNS
        # Nonzero: the rule learner's basis holds W1**2, so its rules and their value differ
        # from the plug-in's on the same folds.
        assert np.any(base.rule_of(fitted.result) != base.rule_of(plug_in))
        assert fitted.truth.truth != fitted.plug_in_truth

    def test_a_plug_in_fit_on_other_folds_stops_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation: the plug-in comparison fitted with another fold seed."""
        frame, _ = _measured("blip")
        honest = base.measured

        def shifted(frame: Any, learner: Any, scenario: str, fold_seed: int, **kw: Any) -> Any:
            if kw.get("rule") is None and not kw.get("mutation"):
                fold_seed += 1
            return honest(frame, learner, scenario, fold_seed, **kw)

        monkeypatch.setattr(base, "measured", shifted)
        with pytest.raises(RuntimeError, match="other outer folds"):
            law.measured(frame, "blip", THROWAWAY + 10)


class TestTheFoldLocalityControl:
    @LOSSES
    def test_each_control_rule_is_learned_on_its_validation_rows(self, loss: str) -> None:
        """Rebuild each fold's control rule from scikit-learn on the validation rows alone."""
        _, fitted = _measured(loss)
        control = fitted.control
        data, nuisance = control.data, control.nuisance
        target = pseudo_outcome(
            nuisance,
            nuisance.scaler.scale(np.asarray(data.outcome, dtype=float)),
            np.asarray(data.treatment, dtype=float),
            control.config.g_bounds,
        )
        covariates = np.asarray(data.covariates, dtype=float)
        expected = np.zeros(data.n, dtype=bool)
        for _, test in nuisance.folds:
            rows = np.asarray(test)
            if loss == "blip":
                model = law.blip_learner().fit(covariates[rows], target[rows])
                expected[rows] = model.predict(covariates[rows]) > 0.0
            else:
                model = law.classifier().fit(
                    covariates[rows],
                    (target[rows] > 0.0).astype(int),
                    fit__sample_weight=np.abs(target[rows]),
                )
                expected[rows] = model.predict_proba(covariates[rows])[:, 1] > 0.5
        np.testing.assert_array_equal(base.rule_of(control), expected)
        # Nonzero: the control learns other rules than the declared fit on the same folds.
        assert np.any(base.rule_of(control) != base.rule_of(fitted.result))
        assert fitted.control_truth.truth != fitted.truth.truth

    @LOSSES
    def test_the_control_read_as_a_training_rule_stops_the_replication(self, loss: str) -> None:
        _, fitted = _measured(loss)
        with pytest.raises(base.HarnessMismatch):
            base.harness(fitted.control, base.outcome_learner(), law.LAW)

    @LOSSES
    def test_the_declared_fit_read_as_a_validation_rule_stops_the_replication(
        self, loss: str
    ) -> None:
        _, fitted = _measured(loss)
        with pytest.raises(base.HarnessMismatch):
            base.harness(fitted.result, base.outcome_learner(), law.LAW, source="validation")

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_a_one_sign_validation_fold_gives_the_constant_rule(
        self, sign: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A weighted classifier on rows of one sign: the package's constant rule, no error."""
        _, fitted = _measured("weighted_classification")
        control = fitted.control
        honest = base.pseudo_outcome
        monkeypatch.setattr(
            base,
            "pseudo_outcome",
            lambda *args: sign * (np.abs(honest(*args)) + 1e-3),
        )
        _, test = next(iter(control.nuisance.folds))
        rows = np.asarray(test)
        rule = base.validation_rule(control.estimator, control.data, control.nuisance, rows)
        covariates = np.asarray(control.data.covariates, dtype=float)
        assert np.all(rule(covariates) == (sign > 0.0))
        # The mutation: the classifier fitted to one class raises, which would cost the
        # replication every cell it writes.
        target = base.pseudo_outcome(
            control.nuisance,
            control.nuisance.scaler.scale(np.asarray(control.data.outcome, dtype=float)),
            np.asarray(control.data.treatment, dtype=float),
            control.config.g_bounds,
        )[rows]
        with pytest.raises(ValueError, match="class"):
            fit_learner(
                law.classifier(),
                covariates[rows],
                as_target((target > 0.0).astype(float), "classification"),
                np.abs(target),
            )
        assert (
            FoldRule("weighted_classification", None, sign > 0.0, 0, target)
            .treat(covariates[:3])
            .tolist()
            == [sign > 0.0] * 3
        )


class TestTheTruncation:
    @LOSSES
    def test_the_auto_bound_does_not_bind(self, loss: str) -> None:
        """``g_bounds="auto"`` resolves to ``5 / (sqrt(n) log n)``, far below every fitted ``g``."""
        _, fitted = _measured(loss)
        lower, upper = fitted.result.config.g_bounds
        assert lower == pytest.approx(5.0 / (np.sqrt(N) * np.log(N)), rel=1e-12)
        assert upper == pytest.approx(1.0 - lower, rel=1e-12)
        assert law.propensity_floor(fitted.result) > 5.0 * lower


#: The property draws of the end-to-end check: three per cell of the throwaway record.
CAP = 3


@cache
def _property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws.

    At three draws a bootstrap resample often repeats one row, whose spread is zero, so the
    SE-ratio bootstrap divides by zero there. That is the size of the check, not the study's.
    """

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        assert set(rows["cell"]) == set(study.PROPERTY_CELLS["interval_calibration"])
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == set(study.PROPERTY_CELLS["interval_calibration"])
        positives = [f"{label}__declared" for label in properties.CALIBRATION_LABELS]
        for column in properties.VALUE_GAIN_SUMMARY:
            assert summary.loc[positives, column].notna().all()
            assert summary.loc[~summary.index.isin(positives), column].isna().all()

    def test_the_control_is_its_own_fit_with_its_own_truth(self) -> None:
        rows = _property_rows()
        for label in properties.CALIBRATION_LABELS:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            control = rows.loc[rows["cell"] == f"{label}__validation_rule_control"].set_index(
                "replicate"
            )
            assert (control["role"] == "control").all()
            assert (control["truth"] != positive["truth"]).all()
            assert (control["estimate"] != positive["estimate"]).all()
            for column in law.VALUE_GAIN_COLUMNS:
                np.testing.assert_array_equal(control[column], positive[column])

    def test_the_value_gain_summary_is_the_paired_difference(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows).set_index("cell")
        for label in properties.CALIBRATION_LABELS:
            group = rows.loc[rows["cell"] == f"{label}__declared"]
            gain = group["truth"] - group["plug_in_truth"]
            cell = f"{label}__declared"
            assert summary.loc[cell, "value_gain_mean"] == pytest.approx(gain.mean(), rel=1e-12)
            assert summary.loc[cell, "value_gain_sd"] == pytest.approx(gain.std(ddof=1), rel=1e-12)
            assert summary.loc[cell, "value_gain_positive_share"] == float(np.mean(gain > 0))

    def test_the_shared_rule_reproduces_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        rule = property_failures.rule_for("interval_calibration", properties)
        assert rule is not None
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
        own = summary.set_index("cell")
        assert set(verdicts.index) == set(own.index)
        for cell in verdicts.index:
            assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell

    def test_a_lost_fit_costs_every_cell_it_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One failed fit is stamped on its positive, its control and both derived controls."""
        cached = _property_rows()
        lost = ("interval_calibration", "blip__declared", 1)

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise np.linalg.LinAlgError("singular matrix")
            stem = label.removesuffix("__declared")
            mine = cached.loc[
                (cached["replicate"] == replicate)
                & cached["cell"].isin([label, f"{stem}__validation_rule_control"])
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
            f"blip__{kind}"
            for kind in (
                "declared",
                "validation_rule_control",
                "shrunken_se_control",
                "noise_control",
            )
        }
        assert {cell for cell, count in stamped.items() if count} == hit
        assert (rows.loc[rows["cell"].isin(hit)].groupby("cell").size() == CAP - 1).all()
        summary = properties.summarize_properties(rows)
        accounting = property_failures.property_accounting(
            rows, records, summary, record=study.STUDY, module=properties, n_jobs=1
        )
        assert property_failures.allowed(properties.FAILURE_BOUND, CAP) == 0
        read = set(accounting.loc[accounting["reason"] == property_failures.BOUND_EXCEEDED, "cell"])
        assert read == hit


def test_a_primary_row_carries_the_value_gain_columns() -> None:
    frame, _ = _measured("weighted_classification")
    record = base.smoke_record(study.STUDY)
    (row,) = study._primary(record, ("weighted_classification", 0, N), frame)
    assert set(law.VALUE_GAIN_COLUMNS) <= set(row)
    assert np.isfinite([row[column] for column in law.VALUE_GAIN_COLUMNS]).all()
    assert row["plug_in_truth"] != row["truth"]
    assert row["rule_rows_checked"] == N
