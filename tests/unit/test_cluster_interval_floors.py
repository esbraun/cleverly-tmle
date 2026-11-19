"""The interval floor of a clustered fit is its fit family's (X54 parts (a) and (b)).

:data:`~cleverly._inference_status.CLUSTER_INTERVAL_FLOORS` holds one floor per measured fit
family: in-sample, stacked and fold-evaluated ``TMLE``, cross-fitted foldwise ``DRTMLE`` with a
declared treatment mechanism, and in-sample and cross-fitted ``LTMLE``. Every other fit keeps
the default floor of its kind. A fit from its floor up to 39 positive-mass clusters reports a t
interval under ``"influence_curve"``; from 3 clusters up to one below its floor it takes
``"few_cluster_plugin"``.

Until the registered result of ``few-cluster-floor-tmle`` and
``few-cluster-floor-cross-fitted-ltmle`` lowers a floor, every family holds its kind's default,
so the floor of one family cannot be told from another's on the shipped values. The witnesses
below therefore lower one family's floor and show that exactly that family's fits move: the
estimator hook, the longitudinal status and the support key each read the family. A hook that
reads the old single floor (the mutation controls) fails them.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import _inference_status, _support
from cleverly._inference_status import (
    CLUSTER_INTERVAL_FLOORS,
    DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS,
    DEFAULT_POINT_INTERVAL_CLUSTERS,
)
from cleverly.datasets import make_clustered
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.longitudinal import LTMLE, SDR
from tests.conftest import linear_drtmle, linear_in_sample
from tests.unit._inference_status_support import assert_keeps_inference
from tests.unit._inference_status_support import cluster_labels as labels
from tests.unit._inference_status_support import longitudinal_learners as learners
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("cluster_interval_floors")

FEW = "few_cluster_plugin"
#: The lowered floor of the witnesses: a count every family keeps labelled on its kind's default
#: floor, which the ``lowered`` fixture starts from.
LOWERED = 6
#: Rows per cluster of the point frames.
SIZE = 30
#: The longitudinal panel's rows, spread over the requested cluster count.
PANEL_ROWS = 600

support_module = importlib.import_module("cleverly._support")


def point_frame(clusters: int) -> Any:
    """``clusters`` clusters of 30 rows of ``make_clustered``, with a declared 0.5 mechanism."""
    frame = make_clustered(n=clusters * SIZE, cluster_size=SIZE, seed=11)[0]
    return frame.assign(p1=0.5, p0=0.5)


def fit_point(
    family: str, clusters: int, *, known: bool = False, estimator: type | None = None
) -> Any:
    """One ATE fit of ``family`` on ``clusters`` clusters, with an estimated or declared mechanism."""
    roles: dict[str, Any] = {"outcome": "Y", "treatment": "A", "id": "cluster"}
    if known:
        roles["treatment_probabilities"] = {1: "p1", 0: "p0"}
    bounds = {"q_bounds": (-25.0, 25.0)}
    folds = min(5, clusters)
    if family == "tmle_in_sample":
        settings = linear_in_sample(estimands=("ate",))
        cls = estimator or TMLE
    elif family == "tmle_stacked":
        settings = linear_in_sample(estimands=("ate",), cross_fit=True, n_folds=folds, **bounds)
        cls = estimator or TMLE
    elif family == "tmle_fold_evaluated":
        settings = linear_in_sample(
            estimands=("ate",),
            cross_fit=True,
            cv_evaluation=True,
            n_folds=fold_evaluated_folds(clusters),
            **bounds,
        )
        cls = estimator or TMLE
    elif family == "drtmle_cross_fitted_known_mechanism":
        settings = linear_drtmle(
            estimands=("ate",), n_folds=folds, reduced_crossfit="foldwise", **bounds
        )
        cls = estimator or DRTMLE
    else:  # pragma: no cover - a new family needs its fit here
        raise KeyError(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**settings).fit(point_frame(clusters), **roles).single()


def fold_evaluated_folds(clusters: int) -> int:
    """Five folds, fewer where a validation fold would hold fewer than two clusters."""
    return min(5, clusters // 2)


def expected_df(family: str, clusters: int) -> int:
    """``J - 2``, or ``J - V`` for a fold-evaluated estimate."""
    if family == "tmle_fold_evaluated":
        return clusters - fold_evaluated_folds(clusters)
    return clusters - 2


def fit_longitudinal(family: str, clusters: int, *, estimator: type = LTMLE) -> Any:
    """An end-of-study fit of ``family`` on ``clusters`` clusters of the RM26 panel."""
    frame = multivalue_panel(n=PANEL_ROWS, seed=43).assign(cluster=labels(PANEL_ROWS, clusters))
    n_folds = 1 if family == "ltmle_in_sample" else 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(
            {"never": 0, "always": 1}, reference="never", **learners(n_folds=n_folds)
        ).fit(frame, **COLUMNS, id="cluster")


POINT = (
    "tmle_in_sample",
    "tmle_stacked",
    "tmle_fold_evaluated",
    "drtmle_cross_fitted_known_mechanism",
)
LONGITUDINAL = ("ltmle_in_sample", "ltmle_cross_fitted")


def fit_family(family: str, clusters: int) -> Any:
    """A fit of ``family``; DR-TMLE declares its mechanism, which its family needs."""
    if family in LONGITUDINAL:
        return fit_longitudinal(family, clusters)
    return fit_point(family, clusters, known=family.startswith("drtmle"))


@pytest.fixture
def lowered(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Lower named families' floors from their kinds' defaults for one test.

    Every other family keeps its default floor (10 or 20), so it is labelled at
    :data:`LOWERED` whatever the shipped floors are.
    """
    defaults = {
        family: DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS
        if family in LONGITUDINAL
        else DEFAULT_POINT_INTERVAL_CLUSTERS
        for family in CLUSTER_INTERVAL_FLOORS
    }

    def apply(**floors: int) -> None:
        monkeypatch.setattr(_inference_status, "CLUSTER_INTERVAL_FLOORS", {**defaults, **floors})

    yield apply


def test_every_family_is_named_and_starts_at_its_kinds_default() -> None:
    assert set(CLUSTER_INTERVAL_FLOORS) == {*POINT, *LONGITUDINAL}
    for family in POINT:
        assert CLUSTER_INTERVAL_FLOORS[family] <= DEFAULT_POINT_INTERVAL_CLUSTERS
    for family in LONGITUDINAL:
        assert CLUSTER_INTERVAL_FLOORS[family] <= DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS


@pytest.mark.parametrize("family", [*POINT, *LONGITUDINAL])
def test_a_fit_at_its_floor_reports_the_t_interval_and_one_below_is_labelled(family: str) -> None:
    floor = CLUSTER_INTERVAL_FLOORS[family]
    at = fit_family(family, floor)
    assert_keeps_inference(at)
    for estimate in at.estimates.values():
        assert estimate.reference_df == (
            expected_df(family, floor) if family in POINT else floor - 2
        )
    below = fit_family(family, floor - 1)
    assert below.inference_status == FEW


@pytest.mark.parametrize("family", [*POINT, *LONGITUDINAL])
def test_a_lowered_floor_moves_its_family_and_no_other(
    family: str, lowered: Callable[..., None]
) -> None:
    """The nonzero witness: only the lowered family reads ``influence_curve`` at 6 clusters."""
    lowered(**{family: LOWERED})
    assert_keeps_inference(fit_family(family, LOWERED))
    other = next(name for name in (*POINT, *LONGITUDINAL) if name != family)
    assert fit_family(other, LOWERED).inference_status == FEW
    assert fit_family(family, LOWERED - 1).inference_status == FEW


def test_the_single_floor_mutation_fails_the_point_witness(
    lowered: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hook that reads the old single point floor leaves a lowered family labelled."""
    lowered(tmle_in_sample=LOWERED)
    tmle_module = importlib.import_module("cleverly.estimators.tmle")
    monkeypatch.setattr(
        tmle_module, "point_interval_floor", lambda *_: DEFAULT_POINT_INTERVAL_CLUSTERS
    )
    with pytest.raises(AssertionError):
        assert_keeps_inference(fit_point("tmle_in_sample", LOWERED))


def test_the_single_floor_mutation_fails_the_longitudinal_witness(
    lowered: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    lowered(ltmle_in_sample=LOWERED)
    estimator_module = importlib.import_module("cleverly.longitudinal.estimator")
    monkeypatch.setattr(estimator_module, "longitudinal_floor_family", lambda *_, **__: None)
    with pytest.raises(AssertionError):
        assert_keeps_inference(fit_longitudinal("ltmle_in_sample", LOWERED))


class TestTheDRTMLEFloorNeedsADeclaredMechanism:
    """The DR-TMLE linear remainder vanishes only under a declared mechanism (plan 2.1, H2)."""

    def test_a_declared_mechanism_reads_the_family_and_a_learned_one_the_default(
        self, lowered: Callable[..., None]
    ) -> None:
        lowered(drtmle_cross_fitted_known_mechanism=LOWERED)
        family = "drtmle_cross_fitted_known_mechanism"
        assert_keeps_inference(fit_point(family, LOWERED, known=True))
        assert fit_point(family, LOWERED, known=False).inference_status == FEW

    def test_a_hook_that_ignores_the_declaration_fails(
        self, lowered: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lowered(drtmle_cross_fitted_known_mechanism=LOWERED)
        monkeypatch.setattr(support_module, "divides_by_declaration", lambda _: True)
        with pytest.raises(AssertionError):
            assert (
                fit_point(
                    "drtmle_cross_fitted_known_mechanism", LOWERED, known=False
                ).inference_status
                == FEW
            )

    def test_the_pooled_reduction_keeps_the_default(self, lowered: Callable[..., None]) -> None:
        lowered(drtmle_cross_fitted_known_mechanism=LOWERED)
        settings = linear_drtmle(
            estimands=("ate",), n_folds=5, reduced_crossfit="pooled", q_bounds=(-25.0, 25.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                DRTMLE(**settings)
                .fit(
                    point_frame(LOWERED),
                    outcome="Y",
                    treatment="A",
                    id="cluster",
                    treatment_probabilities={1: "p1", 0: "p0"},
                )
                .single()
            )
        assert result.inference_status == FEW


@pytest.mark.parametrize("family", ["tmle_in_sample", "tmle_stacked", "tmle_fold_evaluated"])
def test_a_point_tmle_floor_holds_for_both_mechanisms(
    family: str, lowered: Callable[..., None]
) -> None:
    """The floor is mechanism-blind because the floor study measures both (plan B4)."""
    lowered(**{family: LOWERED})
    assert_keeps_inference(fit_point(family, LOWERED, known=False))
    assert_keeps_inference(fit_point(family, LOWERED, known=True))


class TestUnmeasuredFitsKeepTheDefault:
    """Every family lowered: a fit outside every family still reads the default floor."""

    @pytest.fixture
    def all_lowered(self, lowered: Callable[..., None]) -> None:
        lowered(**dict.fromkeys(CLUSTER_INTERVAL_FLOORS, LOWERED))

    def test_ctmle0(self, all_lowered: None) -> None:
        """C-TMLE0 (``strategy="derivative"``), the one clustered C-TMLE path."""
        settings = linear_in_sample(estimands=("ate",), strategy="derivative")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                CTMLE(**settings)
                .fit(point_frame(LOWERED), outcome="Y", treatment="A", id="cluster")
                .single()
            )
        assert result.inference_status == FEW
        assert _support.interval_floor(result.config.support_key) == (
            DEFAULT_POINT_INTERVAL_CLUSTERS
        )

    def test_sdr(self, all_lowered: None) -> None:
        result = fit_longitudinal("ltmle_in_sample", LOWERED, estimator=SDR)
        assert result.inference_status == FEW

    def test_a_repeated_split(self, all_lowered: None) -> None:
        settings = linear_in_sample(
            estimands=("ate",), cross_fit=True, n_folds=3, repeats=3, q_bounds=(-25.0, 25.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(**settings)
                .fit(point_frame(LOWERED), outcome="Y", treatment="A", id="cluster")
                .single()
            )
        assert result.inference_status == FEW


class TestTheSupportKeyReadsTheFamily:
    """``interval_floor`` and the dependence tier read the same family as the status."""

    @pytest.mark.parametrize("family", [*POINT, *LONGITUDINAL])
    def test_interval_floor_is_the_family_floor(
        self, family: str, lowered: Callable[..., None]
    ) -> None:
        lowered(**{family: LOWERED})
        result = fit_family(family, LOWERED)
        key = result.config.support_key
        assert _support.floor_family(key) == family
        assert _support.interval_floor(key) == LOWERED
        assert key.dependence == "t"

    def test_a_key_that_keeps_the_single_floor_reads_below_floor(
        self, lowered: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lowered(tmle_in_sample=LOWERED)
        monkeypatch.setattr(support_module, "floor_family", lambda _: None)
        key = fit_point("tmle_in_sample", LOWERED).config.support_key
        assert key.dependence == "below_floor"

    def test_a_t_tier_row_covers_a_fit_at_a_lowered_floor(
        self, lowered: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A binary in-sample fit at a lowered floor reads supported through row T11; the
        mutation that keeps the single floor reads it below the floor, so experimental."""
        lowered(tmle_in_sample=LOWERED)

        def label() -> str:
            frame = make_clustered(n=LOWERED * SIZE, cluster_size=SIZE, seed=11, family="binomial")[
                0
            ]
            settings = linear_in_sample(
                estimands=("ate",), outcome_learner=LogisticRegression(max_iter=1000)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = TMLE(**settings).fit(frame, outcome="Y", treatment="A", id="cluster")
            return str(result.single().estimates["ate"].support.label)

        assert label() == "supported"
        monkeypatch.setattr(support_module, "floor_family", lambda _: None)
        assert label() == "experimental"


def test_the_status_record_names_every_floor() -> None:
    record = _inference_status.STATUS_RECORDS[FEW]
    for family, floor in CLUSTER_INTERVAL_FLOORS.items():
        assert f"{_inference_status.FLOOR_FAMILY_NAMES[family]} {floor}" in record.reason
        assert f"{_inference_status.FLOOR_FAMILY_NAMES[family]} {floor}" in record.plain_reason


def test_cluster_labels_are_the_requested_count() -> None:
    assert np.unique(labels(PANEL_ROWS, LOWERED)).size == LOWERED
    assert point_frame(LOWERED)["cluster"].nunique() == LOWERED
