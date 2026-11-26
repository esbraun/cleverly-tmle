"""The review findings of the RM42b support label, each with a test that fails without its fix.

Six surfaces: the bootstrap below the cluster floor, the ``derived`` kind of a derived estimate,
the family scope of a bootstrap licence, the band of a fit with a shown status, the group-A
conditions a retarget keeps, and the default label of a simulation record.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS, FEW_CLUSTER_ESTIMAND_STATUS
from cleverly._support import Support, combine_support, registered_rows, support_for
from cleverly.datasets import make_clustered, make_longitudinal
from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import results as inference_results
from cleverly.inference.influence import bootstrap_is_diagnostic, bootstrap_line
from cleverly.longitudinal import SequentialGComputation
from cleverly.validation.simulation import EstimandSummary, ReplicationRecord
from tests.conftest import linear_in_sample
from tests.studies.support_cases import CASES, classification
from tests.unit.test_group_a_experimental_fits import N_LABEL, SITES, draw
from tests.unit.test_group_a_experimental_fits import fit as group_a_fit
from tests.unit.test_longitudinal_cluster_status import fit_one_node
from tests.unit.test_ltmle_bootstrap import _end_columns, _estimator
from tests.unit.test_sequential_baselines import CENSORED, TIGHT, _draw_binary

ROWS = {row.id: row for row in registered_rows()}


def drtmle_estimated_weights(clusters: int) -> Any:
    """A DR-TMLE fit with weights declared estimated, on ``clusters`` clusters of 40 rows."""
    data = make_clustered(n=40 * clusters, cluster_size=40, seed=7)[0]
    data = data.assign(w=np.random.default_rng(3).uniform(0.5, 2.0, len(data)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            DRTMLE(**linear_in_sample(estimands=("ate", "ey0"), n_bootstrap=8))
            .fit(
                data,
                outcome="Y",
                treatment="A",
                id="cluster",
                weights="w",
                weights_estimated=True,
            )
            .single()
        )


class TestTheBootstrapBelowTheClusterFloor:
    """The cluster count decides the diagnostic names, not the name of the status."""

    def test_estimated_weights_with_seven_clusters_print_a_range(self) -> None:
        estimate = drtmle_estimated_weights(7)["ate"]
        assert estimate.inference == "estimated_weight_plugin"
        assert estimate.supplies_inference
        assert bootstrap_is_diagnostic(estimate)
        assert estimate.bootstrap.below_floor
        assert "percentile range" in bootstrap_line(estimate)
        keys = set(estimate.to_dict())
        assert {"bootstrap_sd", "bootstrap_range_lower", "bootstrap_range_upper"} <= keys
        assert "bootstrap_ci_lower" not in keys

    def test_the_same_fit_at_twelve_clusters_keeps_the_interval_names(self) -> None:
        """The control: the status is the same, so the cluster count is what moved."""
        estimate = drtmle_estimated_weights(12)["ate"]
        assert estimate.inference == "estimated_weight_plugin"
        assert not estimate.bootstrap.below_floor
        assert not bootstrap_is_diagnostic(estimate)
        assert "percentile CI" in bootstrap_line(estimate)
        assert {"bootstrap_ci_lower", "bootstrap_ci_upper"} <= set(estimate.to_dict())

    def test_an_unmeasured_longitudinal_design_has_the_floor_of_twenty(self) -> None:
        """A two-node censored panel is not the in-sample floor's measured design, so every
        estimate's bootstrap, the always-versus-never contrast included, reads the default of
        20 (the ``unmeasured`` branch of ``attach_bootstrap``)."""
        columns = {"id": "id", **_end_columns()}
        flags = {}
        for clusters in (15, 20):
            data, _ = make_longitudinal(n=10 * clusters, seed=1, cluster_size=10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _estimator(n_bootstrap=3, simultaneous=False).fit(data, **columns)
            assert result.config.support_key is not None
            flags[clusters] = {
                name: (
                    estimate.bootstrap.below_floor,
                    bootstrap_is_diagnostic(estimate),
                    estimate.bootstrap.cluster_floor,
                )
                for name, estimate in result.estimates.items()
            }
        assert CLUSTER_INTERVAL_FLOORS["ltmle_in_sample"] < 15
        assert set(flags[15].values()) == {(True, True, 20)}
        assert set(flags[20].values()) == {(False, False, None)}

    @staticmethod
    def point_at_seven(**settings: Any) -> Any:
        frame = make_clustered(n=7 * 30, cluster_size=30, seed=7)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                TMLE(**linear_in_sample(estimands=("ate", "ey1"), n_bootstrap=20, **settings))
                .fit(frame, outcome="Y", treatment="A", id="cluster")
                .single()
            )

    def test_an_unmeasured_point_estimate_reads_the_default_floor(self) -> None:
        """B2: in-sample TMLE at 7 clusters is above its family's floor of 6. Its ATE keeps a
        percentile CI; ``ey1`` keeps the default floor of 10, so its bootstrap is a range."""
        result = self.point_at_seven()
        assert result.data.n_clusters == 7 >= CLUSTER_INTERVAL_FLOORS["tmle_in_sample"]
        ey1, ate = result["ey1"], result["ate"]
        assert ey1.inference == FEW_CLUSTER_ESTIMAND_STATUS
        assert (ey1.bootstrap.below_floor, ey1.bootstrap.cluster_floor) == (True, 10)
        assert bootstrap_is_diagnostic(ey1)
        assert "percentile range" in bootstrap_line(ey1)
        assert ate.inference == "influence_curve"
        assert (ate.bootstrap.below_floor, ate.bootstrap.cluster_floor) == (False, None)
        assert "percentile CI" in bootstrap_line(ate)

    def test_a_guard_blind_to_the_estimand_status_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation: the guard no longer recognizes the status, and ``ey1`` prints a CI."""
        monkeypatch.setattr(inference_results, "FEW_CLUSTER_ESTIMAND_STATUS", "not a status")
        ey1 = self.point_at_seven()["ey1"]
        assert not ey1.bootstrap.below_floor
        assert "percentile CI" in bootstrap_line(ey1)

    def test_a_measured_ltmle_design_splits_the_bootstrap_floor(self) -> None:
        """B2, longitudinal: on the measured one-node design at the in-sample floor, the means'
        bootstraps read the default of 20 and the always-versus-never contrast's does not."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_one_node(
                CLUSTER_INTERVAL_FLOORS["ltmle_in_sample"], n_bootstrap=20, simultaneous=False
            )
        for name in ("ey_regimen[always]", "ey_regimen[never]"):
            summary = result[name].bootstrap
            assert (summary.below_floor, summary.cluster_floor) == (True, 20)
        contrast = result["ate_regimen[always vs never]"]
        assert contrast.inference == "influence_curve"
        assert (contrast.bootstrap.below_floor, contrast.bootstrap.cluster_floor) == (False, None)

    def test_a_derived_bootstrap_keeps_the_floor(self) -> None:
        result = drtmle_estimated_weights(7)
        derived = result.contrast(lambda p: p[0] - p[1], ("ate", "ey0"))
        assert derived.bootstrap is not None and derived.bootstrap.below_floor


class TestTheDerivedKind:
    """A derived estimate needs the ``derived`` kind of its row, beside its parents' support."""

    @staticmethod
    def point_fit() -> Any:
        estimator = TMLE(
            **linear_in_sample(estimands=("ey1", "ey0", "ate"), n_bootstrap=3),
        )
        estimator.outcome_learner = LogisticRegression(max_iter=500)
        case = next(c for c in CASES if c.row == "T01.binary")
        _, data, _ = case.build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return estimator.fit(data).single()

    def test_a_ratio_of_arm_means_reads_the_rows_derived_kind(self) -> None:
        """T01.binary covers P1 and P2, so a ratio or difference of two arm means is studied.

        The ratio of ``ey1`` to ``ey0`` is the ``rr`` estimand that ``canonical-tmle`` measured,
        and their difference is the ``ate``.
        """
        result = self.point_fit()
        assert result["ey1"].support.label == "supported"
        assert "derived" in ROWS["T01.binary"].kinds
        assert result.ratio("ey1", "ey0").support.label == "supported"
        contrast = result.contrast(lambda p: p[0] - p[1], ("ey1", "ey0"))
        assert contrast.support.label == "supported"

    def test_a_row_without_the_kind_labels_a_derived_estimate_experimental(self) -> None:
        """The control: with the kind dropped from the row the same ratio reads experimental."""
        listed = ROWS["T01.binary"]
        support = Support("supported", "Row T01.binary covers this configuration.", row=listed.id)
        assert combine_support([support, support], kind="derived").label == "supported"
        rows = {**ROWS, listed.id: replace(listed, kinds=listed.kinds - {"derived"})}
        original = registered_rows
        import cleverly._support as module

        module.registered_rows = lambda: tuple(rows.values())  # type: ignore[assignment]
        try:
            dropped = combine_support([support, support], kind="derived")
        finally:
            module.registered_rows = original  # type: ignore[assignment]
        assert dropped.label == "experimental"
        assert "licenses no derived estimate" in dropped.reason

    def test_a_point_row_without_p2_licenses_no_derived_estimate(self) -> None:
        """Only a point row that covers both P1 and P2 gains the derived kind."""
        from cleverly._support_rows import point_row

        assert "derived" in point_row("X", ["s"], families=("P1", "P2")).kinds
        assert "derived" not in point_row("X", ["s"], families=("P1",)).kinds

    def test_a_longitudinal_row_that_lists_the_kind_stays_supported(self) -> None:
        for row_id in ("G01", "G02", "G11.nodes4", "G12.nodes4"):
            assert "derived" in ROWS[row_id].kinds
            support = Support("supported", "Covered.", row=row_id)
            assert combine_support([support], kind="derived").label == "supported"

    def test_no_row_lists_a_kind_that_is_never_checked(self) -> None:
        listed = {kind for row in ROWS.values() for kind in row.kinds}
        assert listed <= {
            "point",
            "wald",
            "wald_t",
            "band",
            "bootstrap",
            "derived",
            "bootstrap_bca",
            "bootstrap_band",
            # Checked when variance="policy_weighted" (test_policy_variance.py).
            "wald_policy",
            # Checked by LongitudinalResult.rmst and rmtl (test_restricted_mean_estimand.py).
            "restricted_mean",
            "restricted_mean_difference",
        }


class TestTheBootstrapFamilyScope:
    """The T01 bootstrap cell fits the ATE, so only the ``P1`` family licenses it."""

    @staticmethod
    def bootstrap_support(estimands: tuple[str, ...]) -> Support:
        case = next(c for c in CASES if c.row == "T01.binary")
        estimator, data, _ = case.build()
        found = classification(replace(case, build=lambda: (estimator, data, estimands)))
        return support_for(found, "influence_curve", None, kind="bootstrap")

    def test_the_ate_bootstrap_is_supported(self) -> None:
        support = self.bootstrap_support(("ate",))
        assert support.label == "supported"
        assert support.row == "T01.binary"

    @pytest.mark.parametrize("estimand", ["rr", "or", "att", "atc", "par", "paf"])
    def test_every_other_family_reads_experimental(self, estimand: str) -> None:
        support = self.bootstrap_support(("ate", estimand))
        assert support.label == "experimental"
        assert "P1 only" in support.reason

    def test_the_scope_is_what_moves_the_label(self) -> None:
        """The mutation: a row without the family scope licenses the bootstrap of a ratio."""
        row = ROWS["T01.binary"]
        assert dict(row.kind_families) == {"bootstrap": frozenset({"P1"})}
        assert row.licenses("bootstrap", frozenset({"P1"}))
        assert not row.licenses("bootstrap", frozenset({"P1", "P2"}))
        assert replace(row, kind_families=()).licenses("bootstrap", frozenset({"P1", "P2"}))


class TestTheBandOfAShownStatus:
    def test_a_sequential_fit_with_estimated_weights_builds_a_labelled_band(self) -> None:
        data = _draw_binary(400, 2)
        data["wt"] = np.random.default_rng(0).uniform(0.5, 2.0, len(data))
        columns = {**CENSORED, "baseline": ["W"], "weights": "wt", "weights_estimated": True}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SequentialGComputation(
                {"always": 1, "never": 0},
                outcome_learner=TIGHT,
                treatment_learner=TIGHT,
                censoring_learner=TIGHT,
                simultaneous=True,
            ).fit(data, **columns)
        assert result.inference_status == "estimated_weight_plugin"
        assert result.simultaneous is not None
        assert result.simultaneous.support.label == "experimental"
        assert set(result.simultaneous.bands) == set(result.estimates)


class TestTheRetargetKeepsTheConditions:
    """A fit records its group-A conditions, and a retarget reuses them instead of preflighting."""

    @staticmethod
    def labelled_fit() -> tuple[Any, Any]:
        site = next(s for s in SITES if s.id == "arm.linear_fluctuation")
        data, _ = draw(N_LABEL, 12, site.law)
        return site, group_a_fit(site, data)

    def test_the_fit_saves_the_conditions_it_met(self) -> None:
        site, result = self.labelled_fit()
        conditions = result.config.support_conditions
        assert len(conditions) == 1
        assert site.fragment in conditions[0].condition
        assert conditions[0].roadmap == site.roadmap
        assert pickle.loads(pickle.dumps(result)).config.support_conditions == conditions

    def test_a_retarget_with_the_saved_conditions_keeps_the_label(self) -> None:
        site, result = self.labelled_fit()
        kept, _ = result.estimator.retarget(
            result.data,
            result.nuisance,
            estimands=("ate",),
            conditions=result.config.support_conditions,
        )
        assert kept["ate"].support.label == "experimental"
        assert site.fragment in kept["ate"].support.reason
        assert site.roadmap in kept["ate"].support.evidence

    def test_a_retarget_without_them_is_the_plain_classification(self) -> None:
        """The control: the condition reaches the label only through the saved conditions."""
        site, result = self.labelled_fit()
        plain, _ = result.estimator.retarget(result.data, result.nuisance, estimands=("ate",))
        assert site.fragment not in plain["ate"].support.reason

    def test_a_fit_without_a_condition_saves_none(self) -> None:
        data = make_clustered(n=240, cluster_size=40, seed=7)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(**linear_in_sample(estimands=("ate",)))
                .fit(data, outcome="Y", treatment="A", id="cluster")
                .single()
            )
        assert result.config.support_conditions == ()


class TestTheSimulationDefaults:
    def test_an_unlabelled_record_is_experimental(self) -> None:
        record = ReplicationRecord(
            replicate=0,
            seed=0,
            estimand="ate",
            truth=0.0,
            estimate=0.1,
            std_error=0.1,
            lower=-0.1,
            upper=0.3,
            covered=True,
            rejected=False,
            inference_estimate=0.1,
            alpha=0.05,
        )
        assert record.support == "experimental"
        assert EstimandSummary.__dataclass_fields__["support"].default == "experimental"
