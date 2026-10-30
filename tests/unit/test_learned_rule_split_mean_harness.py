"""Unit witnesses for the truth harness and the controls of the ``learned-rule-split-mean`` study.

The study's verdicts read the error ``estimate - truth`` of each row, and its control reads a
standard error the harness builds, so a truth that reads the wrong split, or a control SE built
from the wrong pieces, would fail or pass a cell for the wrong reason. These checks rebuild each
piece from single-split fits at the split seeds (the W-i1 construction), and one deliberate
mutation, a split's folds read beside another split's rule, must stop the replication.

Every fit draws from a throwaway seed at a small size, and no test asserts a bias, a coverage or
an SE ratio of the study.
"""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_split_mean_law as law
from tests.studies import learned_rule_split_mean as study
from tests.studies import learned_rule_split_mean_properties as properties
from tests.studies.evidence import property_failures, replication

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_991_000
N = 400


@cache
def _repeated() -> tuple[pd.DataFrame, Any]:
    frame = base.draw("misspecified_limit", N, THROWAWAY)
    return frame, law.fit(frame, base.outcome_learner(), THROWAWAY + 1)


@cache
def _singles() -> tuple[Any, ...]:
    """One ``repeats=1`` fit at each split seed of the repeated fit, as W-i1 builds them."""
    frame, result = _repeated()
    out = []
    for repeat in result.repeats:
        single = TMLE(
            learned_rule=LearnedRule(),
            n_folds=law.N_FOLDS,
            cv_evaluation=True,
            targeting_scheme="pooled",
            g_bounds="auto",
            outcome_learner=base.outcome_learner(),
            treatment_learner=base.treatment_learner(),
            random_state=repeat.seed,
        )
        out.append(
            single.fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES)).single()
        )
    return tuple(out)


class TestTheTruth:
    def test_each_split_value_is_the_single_split_fits_truth(self) -> None:
        _, result = _repeated()
        truth = law.harness(result, base.outcome_learner(), "misspecified_limit")
        singles = [
            base.harness(single, base.outcome_learner(), "misspecified_limit")
            for single in _singles()
        ]
        assert len(truth.split_values) == law.SPLITS == len(singles)
        assert truth.split_values == tuple(single.truth for single in singles)
        assert truth.truth == float(np.mean(truth.split_values))
        assert truth.rows_checked == law.SPLITS * N
        # Nonzero: the splits learn different rules, so their targets differ, and a harness
        # that read one split for all of them would give one value five times.
        assert len(set(truth.split_values)) == law.SPLITS

    def test_a_split_read_beside_another_splits_rule_stops_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation: split ``r``'s folds paired with split ``r + 1``'s rule."""
        _, result = _repeated()
        honest = law.split_view

        def crossed(result: Any, split: int) -> SimpleNamespace:
            view = honest(result, split)
            other = result.repeats[(split + 1) % len(result.repeats)].nuisance
            return SimpleNamespace(
                data=view.data,
                nuisance=SimpleNamespace(folds=view.nuisance.folds, regimes=other.regimes),
            )

        monkeypatch.setattr(law, "split_view", crossed)
        with pytest.raises(base.HarnessMismatch):
            law.harness(result, base.outcome_learner(), "misspecified_limit")

    def test_a_point_that_is_not_the_mean_stops_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard of :func:`law.measured`: a median in place of the mean is refused."""
        frame, _ = _repeated()
        monkeypatch.setattr(
            law, "split_estimates", lambda result: np.full(law.SPLITS, -1.0, dtype=float)
        )
        with pytest.raises(RuntimeError, match="not the mean"):
            law.measured(frame, "misspecified_limit", base.outcome_learner(), THROWAWAY + 1)


class TestTheSplitQuantities:
    def test_the_split_estimates_and_errors_are_the_single_split_fits(self) -> None:
        _, result = _repeated()
        singles = _singles()
        np.testing.assert_array_equal(
            law.split_estimates(result),
            [float(single.estimates[law.ESTIMAND].psi) for single in singles],
        )
        np.testing.assert_allclose(
            law.split_std_errors(result),
            [float(single.estimates[law.ESTIMAND].std_error) for single in singles],
            rtol=1e-12,
        )

    def test_the_independent_splits_se_is_the_mean_single_se_over_root_r(self) -> None:
        _, result = _repeated()
        singles = [float(fit.estimates[law.ESTIMAND].std_error) for fit in _singles()]
        expected = float(np.mean(singles)) / np.sqrt(law.SPLITS)
        assert law.independent_splits_se(result) == pytest.approx(expected, rel=1e-12)
        # Nonzero: it is about 1/sqrt(5) of the reported SE, so the control must sit far
        # below the band, and a helper that returned the fit's own SE would not.
        ratio = law.independent_splits_se(result) / float(result.estimates[law.ESTIMAND].std_error)
        assert 0.40 < ratio < 0.50

    def test_the_median_is_the_median_of_the_single_split_points(self) -> None:
        _, result = _repeated()
        points = [float(fit.estimates[law.ESTIMAND].psi) for fit in _singles()]
        assert law.median_estimate(result) == float(np.median(points))
        assert law.median_estimate(result) != float(result.estimates[law.ESTIMAND].psi)

    def test_the_initial_estimate_is_the_mean_of_the_single_split_plug_ins(self) -> None:
        _, result = _repeated()
        expected = np.mean([base.untargeted_estimate(fit) for fit in _singles()])
        assert law.untargeted_estimate(result) == pytest.approx(expected, rel=1e-14)


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
    SE-ratio bootstrap divides by zero there and subtracts infinities. That is the size of the check, not the study's.
    """

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        assert set(rows["cell"]) == set(study.PROPERTY_CELLS["interval_calibration"])
        assert (rows.groupby("cell").size() == CAP).all()
        fitted = rows.loc[rows["cell"].str.endswith(("__declared", "__independent_splits_control"))]
        assert (fitted["rule_rows_checked"] == properties.RULE_CHECKS_PER_ROW * fitted["n"]).all()
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == set(study.PROPERTY_CELLS["interval_calibration"])
        positives = [f"{label}__declared" for label in properties.CALIBRATION_LABELS]
        for column in properties.MEDIAN_COLUMNS:
            assert summary.loc[positives, column].notna().all()
            assert summary.loc[~summary.index.isin(positives), column].isna().all()

    def test_the_control_reads_the_positive_estimate_with_its_own_se(self) -> None:
        rows = _property_rows()
        for label in properties.CALIBRATION_LABELS:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            control = rows.loc[rows["cell"] == f"{label}__independent_splits_control"].set_index(
                "replicate"
            )
            np.testing.assert_array_equal(control["estimate"], positive["estimate"])
            np.testing.assert_array_equal(control["truth"], positive["truth"])
            assert (control["std_error"] < 0.5 * positive["std_error"]).all()
            assert (control["role"] == "control").all()

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
        lost = ("interval_calibration", "non_exceptional__declared", 1)

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise np.linalg.LinAlgError("singular matrix")
            stem = label.removesuffix("__declared")
            mine = cached.loc[
                (cached["replicate"] == replicate)
                & cached["cell"].isin([label, f"{stem}__independent_splits_control"])
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
            f"non_exceptional__{kind}"
            for kind in (
                "declared",
                "independent_splits_control",
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
        # floor(0.01 * 3) = 0, so the loss exceeds the bound in every cell that read it.
        assert property_failures.allowed(properties.FAILURE_BOUND, CAP) == 0
        read = set(accounting.loc[accounting["reason"] == property_failures.BOUND_EXCEEDED, "cell"])
        assert read == hit
