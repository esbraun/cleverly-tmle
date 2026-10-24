"""The support label and the shown statuses, at the surfaces a reader meets.

Each class names a component that can vanish and a witness that fails when it does.

* ``TestAGoldenCopy`` pins the numbers and the text of ``TMLE`` fits against copies made on
  ``origin/main`` before the label existed, so a supported fit stays unmarked and no cluster
  count from 10 to 40 moves a number. The copies live in ``tests/unit/golden/rm42b``. The
  script that made them fits the configuration in :func:`golden_fit` on that commit and dumps
  ``summary()``, ``repr``, ``to_dict()`` and ``to_frame()`` of the ``ate`` estimate. The
  summary has its version and time stamp replaced. That commit's default summary is today's
  ``summary(detail="full")``; the short default leaves out its protocol line only.
* ``TestTheLabelAtEverySurface`` fits one experimental point fit and one experimental
  longitudinal fit and reads the label from every report that prints an estimate.
* ``TestAWithheldStatusWinsOverAShownOne`` pins the two-tier precedence.
* ``TestAShownStatus`` and ``TestFewClusterT`` pin the shown statuses and the t reference.
* ``TestRepeats`` pins the label of a median over fold draws.
"""

from __future__ import annotations

import importlib
import json
import re
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import _support
from cleverly._inference_status import STATUS_RECORDS, precedent_status
from cleverly._support import Support, combine_support
from cleverly.data.causal_data import CausalData
from cleverly.datasets import make_binary_outcome, make_clustered, make_linear_ate
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference import results as results_module
from cleverly.inference.delta import wald_ci
from cleverly.longitudinal import LTMLE, SequentialGComputation
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal.data import LongitudinalData
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import assert_shows_interval
from tests.unit.test_event_free_node import CONSTANT_NODE
from tests.unit.test_longitudinal_cluster_status import fit_competing

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "rm42b" / "origin_main.json").read_text(encoding="utf-8")
)

#: The columns and text that only the label adds.
LABEL_KEYS = ("support", "support_reason", "support_evidence", "support_limits")


def golden_fit(name: str) -> Any:
    """The fit behind a golden copy: ``canonical``, or ``clusters_<J>`` for ``J`` clusters."""
    if name == "canonical":
        frame = make_linear_ate(n=200, seed=1)[0]
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
    else:
        clusters = int(name.split("_")[1])
        frame = make_clustered(n=4 * clusters, cluster_size=4, seed=5)[0]
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", id="cluster")
    return TMLE(**linear_in_sample()).fit(data).single()


def stamped(text: str) -> str:
    """``text`` with its version and time stamp replaced, as the golden copies have it."""
    return re.sub(r"cleverly [\d.]+ \| \S+", "cleverly VERSION | TIMESTAMP", text)


def without_protocol_line(summary: str) -> str:
    """``summary`` without the protocol line, which only a full summary prints."""
    removed = summary.replace("\ncausal study protocol: absent\n", "\n", 1)
    assert removed != summary, "the golden summary no longer holds its protocol line"
    return removed


def unlabelled(summary: str) -> str:
    """``summary`` without the support column and its note, which the label adds."""
    lines = []
    for line in stamped(summary).splitlines():
        if line.startswith("experimental:"):
            continue
        lines.append(re.sub(r"\s+(support|experimental|-{12})\s*$", "", line).rstrip())
    return "\n".join(line for line in lines if line)


def assert_numbers_equal(actual: dict[str, Any], golden: dict[str, Any]) -> None:
    """Every number of ``actual`` that is not a label key equals the golden one, ``rtol=1e-10``."""
    kept = {key: value for key, value in actual.items() if not key.startswith(LABEL_KEYS)}
    assert set(kept) == set(golden)
    for key, value in kept.items():
        if isinstance(value, float):
            assert value == pytest.approx(golden[key], rel=1e-10), key
        else:
            assert value == golden[key], key


class TestAGoldenCopy:
    def test_a_supported_fit_is_unmarked(self) -> None:
        result = golden_fit("canonical")
        golden = GOLDEN["canonical"]
        estimate = result["ate"]
        assert estimate.support.label == "supported"
        assert estimate.support.row == "T01.continuous"
        assert stamped(result.summary(detail="full")) == golden["summary"]
        assert stamped(result.summary()) == without_protocol_line(golden["summary"])
        assert repr(estimate) == golden["repr"]
        assert_numbers_equal(estimate.to_dict(), golden["to_dict"])
        frame = result.to_frame()
        assert list(frame["support"]) == ["supported"] * len(frame)
        for column, values in golden["frame"].items():
            if column == "support":
                continue
            assert list(frame[column]) == pytest.approx(values, rel=1e-10)

    def test_deleting_the_row_turns_the_label_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: with no row the label and the summary change."""
        monkeypatch.setattr(_support, "registered_rows", lambda: ())
        result = golden_fit("canonical")
        assert result["ate"].support.label == "experimental"
        assert stamped(result.summary(detail="full")) != GOLDEN["canonical"]["summary"]
        assert repr(result["ate"]) != GOLDEN["canonical"]["repr"]

    @pytest.mark.parametrize("clusters", [10, 20, 39, 40])
    def test_cluster_counts_from_ten_move_no_number(self, clusters: int) -> None:
        """The few-cluster change touches no fit with 10 or more clusters, to the digit."""
        result = golden_fit(f"clusters_{clusters}")
        golden = GOLDEN[f"clusters_{clusters}"]
        estimate = result["ate"]
        assert result.inference_status == "influence_curve"
        assert unlabelled(result.summary(detail="full")) == unlabelled(golden["summary"])
        assert repr(estimate).removesuffix(" [experimental]") == golden["repr"]
        assert_numbers_equal(estimate.to_dict(), golden["to_dict"])


class TestTheLabelAtEverySurface:
    """An experimental point fit and an experimental longitudinal fit show the label."""

    @pytest.fixture(scope="class")
    def point(self) -> Any:
        return golden_fit("clusters_40")

    @pytest.fixture(scope="class")
    def longitudinal(self) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_competing(40)

    def test_the_point_fit_shows_the_marker(self, point: Any) -> None:
        estimate = point["ate"]
        assert estimate.support.label == "experimental"
        assert repr(estimate).endswith(" [experimental]")
        text = point.summary()
        assert "support" in text
        assert f"experimental: {estimate.support.reason}" in text
        row = estimate.to_dict()
        assert row["support"] == "experimental"
        assert row["support_reason"] == estimate.support.reason
        frame = point.to_frame()
        assert set(frame["support"]) == {"experimental"}
        assert "support_reason" in frame.columns

    def test_the_longitudinal_fit_shows_the_marker(self, longitudinal: Any) -> None:
        for estimate in longitudinal.estimates.values():
            assert estimate.support.label == "experimental"
            assert repr(estimate).endswith(" [experimental]")
            assert estimate.to_dict()["support"] == "experimental"
        assert "experimental" in longitudinal.summary()
        assert set(longitudinal.to_frame()["support"]) == {"experimental"}
        assert set(longitudinal.curve()["support"]) == {"experimental"}
        assert set(longitudinal.incidence_total()["support"]) == {"experimental"}

    def test_a_total_is_experimental_when_one_input_is(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One cause of every total reads a supported record, and the other does not.

        The nonzero witness for ``combine_support``: the right answer is experimental, and a
        combination that returns its first input labels the total supported.
        """
        first_cause = longitudinal.config.causes[0]
        probe = replace(
            longitudinal,
            estimates={
                name: replace_support(estimate, "supported")
                if longitudinal.parameter_index[name][1] == first_cause
                else estimate
                for name, estimate in longitudinal.estimates.items()
            },
        )
        assert set(probe.incidence_total()["support"]) == {"experimental"}
        monkeypatch.setattr(
            longitudinal_estimator, "combine_support", lambda supports, **_: supports[0]
        )
        assert set(probe.incidence_total()["support"]) == {"supported"}


def replace_support(estimate: Any, label: str) -> Any:
    return replace(estimate, support=Support(label, "A study covers it.", ("study-a",), row="R1"))


def single_tier(statuses: Any) -> str:
    """The mutant precedence: the first applied status in table order, shown or withheld."""
    applied = {status for status in statuses if status != "influence_curve"}
    return next((name for name in STATUS_RECORDS if name in applied), "influence_curve")


def constant_node_fit(clusters: int) -> Any:
    """The event-free-node law of ``test_event_free_node`` with ``clusters`` clusters."""
    rng = np.random.default_rng(1)
    n = 600
    w = rng.integers(0, 2, n)
    a = rng.binomial(1, 0.5, n)
    time = rng.integers(1, 5, n).astype(float)
    event = rng.binomial(1, 0.6, n)
    event[(a == 0) & (time <= 2)] = 0
    frame = pd.DataFrame(
        {"time": time, "event": event, "A": a, "W": w, "cluster": np.arange(n) % clusters}
    )
    data = LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=[1, 2, 3, 4],
        id="cluster",
    )
    estimator = LTMLE(
        {"treated": 1, "control": 0},
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        random_state=0,
        n_folds=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(data)


def adaptive_weighted_fit() -> Any:
    """A sequential g-computation with a data-adaptive learner and estimated weights."""
    from sklearn.tree import DecisionTreeRegressor

    from tests.unit.test_sequential_baselines import _small

    frame, columns = _small()
    frame["wt"] = np.random.default_rng(0).uniform(0.5, 2.0, len(frame))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SequentialGComputation(
            {"always": 1},
            outcome_learner=DecisionTreeRegressor(max_depth=2, random_state=0),
            simultaneous=False,
        ).fit(frame, **columns, weights="wt", weights_estimated=True)


class TestAWithheldStatusWinsOverAShownOne:
    """B1: a fit that loses an interval for one reason does not regain it for another.

    ``estimated_weight_plugin`` and ``few_cluster_plugin`` show their intervals. Each witness
    below applies one of them beside a status that withholds, and the withheld one wins. The
    mutant takes the first applied status in table order, which is the shown one in each case.
    """

    def test_a_data_adaptive_learner_beside_estimated_weights(self) -> None:
        result = adaptive_weighted_fit()
        assert result.inference_status == "adaptive_baseline_plugin"
        with pytest.raises(CapabilityError):
            _ = result["ey_regimen[always]"].ci

    def test_a_constant_node_at_fifteen_clusters(self) -> None:
        result = constant_node_fit(15)
        assert result.inference_status == "few_cluster_plugin"
        for name in CONSTANT_NODE:
            assert result[name].inference == "constant_node_plugin"
            with pytest.raises(CapabilityError):
                _ = result[name].ci
        shown = result["risk_regimen[control @ t=3]"]
        assert shown.inference == "few_cluster_plugin"
        low, high = shown.ci
        quantile = stats.t.ppf(0.975, 13)
        assert low == pytest.approx(shown.psi - quantile * shown.plugin_std_error)
        assert high == pytest.approx(shown.psi + quantile * shown.plugin_std_error)

    def test_a_contrast_of_the_constant_node_parameter(self) -> None:
        result = constant_node_fit(15)
        contrast = result.contrast(
            lambda psi: psi[0] - psi[1],
            ["risk_regimen[control @ t=1]", "risk_regimen[control @ t=3]"],
        )
        assert contrast.inference == "constant_node_plugin"
        with pytest.raises(CapabilityError):
            _ = contrast.ci

    def test_the_single_tier_mutant_fails_each_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(longitudinal_estimator, "precedent_status", single_tier)
        monkeypatch.setattr(results_module, "precedent_status", single_tier)
        assert adaptive_weighted_fit().inference_status == "estimated_weight_plugin"
        result = constant_node_fit(15)
        assert result["risk_regimen[control @ t=1]"].inference == "few_cluster_plugin"
        contrast = result.contrast(
            lambda psi: psi[0] - psi[1],
            ["risk_regimen[control @ t=1]", "risk_regimen[control @ t=3]"],
        )
        assert contrast.inference == "few_cluster_plugin"

    def test_the_real_precedence_is_the_registry_precedence(self) -> None:
        assert precedent_status(["estimated_weight_plugin", "adaptive_baseline_plugin"]) == (
            "adaptive_baseline_plugin"
        )
        assert single_tier(["estimated_weight_plugin", "adaptive_baseline_plugin"]) == (
            "estimated_weight_plugin"
        )


class TestAShownStatus:
    """A shown status answers ``ci`` and labels the fit with the record's reason."""

    @pytest.fixture(scope="class")
    def weighted(self) -> Any:
        frame = make_binary_outcome(n=500, seed=3)[0]
        frame["w"] = np.random.default_rng(3).uniform(0.5, 2.0, len(frame))
        return (
            DRTMLE(**linear_in_sample(estimands=("ate", "ey0")))
            .fit(frame, outcome="Y", treatment="A", weights="w", weights_estimated=True)
            .single()
        )

    def test_the_interval_answers_and_the_label_gives_the_reason(self, weighted: Any) -> None:
        assert_shows_interval(weighted, "estimated_weight_plugin", None)
        support = weighted["ate"].support
        assert support.label == "experimental"
        assert STATUS_RECORDS["estimated_weight_plugin"].shown_reason in support.reason

    def test_a_withheld_record_makes_the_same_fit_refuse(self, weighted: Any) -> None:
        """The mutation control: ``interval="withheld"`` makes ``ci`` raise."""
        record = STATUS_RECORDS["estimated_weight_plugin"]
        object.__setattr__(record, "interval", "withheld")
        try:
            with pytest.raises(CapabilityError):
                _ = weighted["ate"].ci
        finally:
            object.__setattr__(record, "interval", "shown")
        assert weighted["ate"].ci

    def test_two_shown_statuses_name_both_in_the_reason(self) -> None:
        """Estimated weights at 7 clusters apply two shown statuses, and the label names both."""
        frame = make_clustered(n=360, cluster_size=40, seed=7)[0].iloc[:280]
        frame = frame.assign(w=np.random.default_rng(3).uniform(0.5, 2.0, len(frame)))
        result = (
            DRTMLE(**linear_in_sample(estimands=("ate", "ey0")))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                id="cluster",
                weights="w",
                weights_estimated=True,
            )
            .single()
        )
        assert result.data.n_clusters == 7
        reason = result["ate"].support.reason
        assert STATUS_RECORDS["estimated_weight_plugin"].shown_reason in reason
        assert STATUS_RECORDS["few_cluster_plugin"].shown_reason in reason
        assert {"X50", "X54"} <= set(result["ate"].support.evidence)


class TestFewClusterT:
    """B2: three to nine clusters read t(J - 2), and fewer than three withhold."""

    @staticmethod
    def fit(clusters: int, *, size: int = 4, **settings: Any) -> Any:
        frame = make_clustered(n=size * clusters, cluster_size=size, seed=5)[0]
        return (
            TMLE(**linear_in_sample(**settings))
            .fit(frame, outcome="Y", treatment="A", id="cluster")
            .single()
        )

    def test_five_clusters_read_t_with_three_degrees_of_freedom(self) -> None:
        result = self.fit(5)
        estimate = result["ate"]
        assert result.inference_status == "few_cluster_plugin"
        assert estimate.reference_df == 3
        low, high = estimate.ci
        normal = wald_ci(estimate.psi, estimate.plugin_std_error, df=None)
        assert low < normal[0] and high > normal[1]
        assert (high - low) / 2 == pytest.approx(
            stats.t.ppf(0.975, 3) * estimate.plugin_std_error, rel=1e-9
        )

    def test_two_clusters_withhold_the_interval(self) -> None:
        result = self.fit(2, size=6)
        assert result.inference_status == "no_cluster_variance_plugin"
        with pytest.raises(CapabilityError):
            _ = result["ate"].ci

    def test_fold_evaluated_targeting_at_five_clusters_keeps_its_refusal(self) -> None:
        """Five whole-cluster folds leave one cluster per fold, which the preflight refuses."""
        with pytest.raises((CapabilityError, DataError, ValueError)):
            self.fit(5, cross_fit=True, n_folds=5, cv_evaluation=True, q_bounds=(-15.0, 15.0))

    def test_a_floor_of_ten_fails_the_five_cluster_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: the point-treatment floor of 10 as the t floor withholds J=5."""
        from cleverly.inference import cluster as cluster_module

        monkeypatch.setattr(cluster_module, "MINIMUM_T_REFERENCE_CLUSTERS", 10)
        result = self.fit(5)
        assert result.inference_status == "no_cluster_variance_plugin"
        with pytest.raises(CapabilityError):
            _ = result["ate"].ci


class TestRepeats:
    """M1: a median over fold draws keeps its row's label."""

    @staticmethod
    def fit_repeats(**settings: Any) -> Any:
        frame = make_binary_outcome(n=300, seed=2)[0]
        return (
            TMLE(
                **linear_in_sample(
                    cross_fit=True,
                    n_folds=3,
                    repeats=3,
                    estimands=("ey1", "ey0", "ate"),
                    **settings,
                )
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_every_estimate_keeps_the_label_of_its_row(self) -> None:
        result = self.fit_repeats()
        assert {estimate.support.row for estimate in result.estimates.values()} == {"T05.binary"}
        assert {estimate.support.label for estimate in result.estimates.values()} == {"supported"}

    def test_dropping_the_support_copy_leaves_the_label_unclassified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: a median that drops ``support`` leaves ``UNCLASSIFIED``."""
        from cleverly._support import UNCLASSIFIED

        tmle_module = importlib.import_module("cleverly.estimators.tmle")
        original = tmle_module.median_estimates

        def without_support(reports: Any) -> Any:
            return {
                name: replace(estimate, support=UNCLASSIFIED)
                for name, estimate in original(reports).items()
            }

        monkeypatch.setattr(tmle_module, "median_estimates", without_support)
        result = self.fit_repeats()
        assert {estimate.support for estimate in result.estimates.values()} == {UNCLASSIFIED}


class TestADerivedEstimate:
    """M5: a ratio takes the combination of its parents' labels."""

    def test_a_ratio_of_a_supported_fit_is_supported(self) -> None:
        frame = make_binary_outcome(n=300, seed=2)[0]
        result = (
            TMLE(**linear_in_sample(estimands=("ey1", "ey0")))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        ratio = result.ratio("ey1", "ey0")
        assert ratio.support == result["ey1"].support
        assert ratio.support.label == "supported"

    def test_a_ratio_of_an_experimental_fit_is_experimental(self) -> None:
        frame = make_clustered(n=160, cluster_size=4, seed=5, family="binomial")[0]
        result = (
            TMLE(**linear_in_sample(estimands=("ey1", "ey0")))
            .fit(frame, outcome="Y", treatment="A", id="cluster")
            .single()
        )
        ratio = result.ratio("ey1", "ey0")
        assert ratio.support.label == "experimental"
        assert ratio.support == combine_support([result["ey1"].support, result["ey0"].support])
