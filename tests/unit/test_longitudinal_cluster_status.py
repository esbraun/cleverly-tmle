"""A clustered ``LTMLE`` fit shows a labelled t interval below its family's floor.

The point-treatment few-cluster status applies to the longitudinal path too, with the floors of
:data:`~cleverly._inference_status.CLUSTER_INTERVAL_FLOORS`: 6 in sample and 4 cross-fitted. A
fit with 3 positive-mass clusters up to one below its floor stamps ``"few_cluster_plugin"`` on
every mean, contrast and MSM coefficient. Each estimate keeps its interval on a Student t
reference with ``J - 2`` degrees of freedom, under the experimental label, and F28 records the
measured limits below the floor. The floor holds for the always-versus-never contrast its study
measured: from it to 39 clusters that contrast keeps the same reference and the ordinary status,
and every other estimate does so from the default longitudinal floor of 20
(:class:`TestTheFloorHoldsForTheMeasuredContrast`). The witness and
the control run in sample and at five whole-cluster folds, for every kind but the working model,
which ``LTMLE`` refuses above one fold for every fit.

Each kind of fit draws 400 rows from its law and passes the labels
``np.arange(400) * k // 400`` as ``id=``. Only the labels differ between the witness one below
the floor and the control at it. In sample the point estimates are the same numbers, and only
the status moves. A grouped split reads the labels, so at five folds the estimates move too.
Each mutation in :class:`TestTheMutationsFailTheWitness` must fail the check
its surface passes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean, _inference_status
from cleverly._inference_status import (
    CLUSTER_INTERVAL_FLOORS,
    DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS,
    FEW_CLUSTER_ESTIMAND_STATUS,
    FEW_CLUSTER_THRESHOLD,
    NO_SIMULTANEOUS_BANDS,
    NO_T_REFERENCE_BANDS,
    STATUS_RECORDS,
)
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
)
from cleverly.exceptions import CapabilityError, DataWarning
from cleverly.inference.cluster import cluster_inference_status
from cleverly.longitudinal import LTMLE, DynamicRegimen, Regimen
from cleverly.longitudinal import estimator as longitudinal_estimator_module
from tests.unit._inference_status_support import (
    INFERENTIAL_COLUMNS,
    ROUTES,
    assert_assessment_note,
    assert_keeps_inference,
    assert_round_trips,
    assert_shows_interval,
    at_or_below,
    longitudinal_estimator,
    restore,
)
from tests.unit._inference_status_support import LONGITUDINAL_N as N
from tests.unit._inference_status_support import LONGITUDINAL_NODES as NODES
from tests.unit._inference_status_support import cluster_labels as labels
from tests.unit._inference_status_support import longitudinal_learners as learners
from tests.unit.test_longitudinal_msm import DOSE
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("longitudinal_cluster_status")

#: The in-sample LTMLE floor, the floor of every in-sample fit below.
IN_SAMPLE_FLOOR = CLUSTER_INTERVAL_FLOORS["ltmle_in_sample"]


FEW = "few_cluster_plugin"

COMPETING: dict[str, Any] = {
    "outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
    **NODES,
}
SURVIVAL: dict[str, Any] = {"outcome": ["Y1", "Y2"], **NODES}

#: The kinds of fit whose report adds a curve over the horizons.
CURVES = ("survival", "competing risks")


def fit_end_of_study(k: int | None, *, weights: Any = None, **settings: Any) -> Any:
    """The RM26 roadmap law, with ``k`` clusters: two means and one contrast.

    ``k=None`` fits the same rows with no cluster labels.
    """
    frame = multivalue_panel(n=N, seed=43)
    roles: dict[str, Any] = {}
    if k is not None:
        frame = frame.assign(cluster=labels(N, k))
        roles["id"] = "cluster"
    if weights is not None:
        frame = frame.assign(w=weights(frame["cluster"].to_numpy()))
        roles["weights"] = "w"
    return LTMLE({"never": 0, "always": 1}, reference="never", **learners(**settings)).fit(
        frame, **COLUMNS, **roles
    )


def fit_survival(k: int, **settings: Any) -> Any:
    """One event at two horizons: four risks and two contrasts."""
    frame, _ = make_longitudinal_survival(n=N, seed=2)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, reference="never", **learners(**settings)).fit(
        frame, **SURVIVAL, id="cluster"
    )


def fit_competing(k: int, **settings: Any) -> Any:
    """Two causes at two horizons: eight incidences and four contrasts."""
    frame, _ = make_longitudinal_competing(n=N, seed=3)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, reference="never", **learners(**settings)).fit(
        frame, **COMPETING, id="cluster"
    )


def fit_msm(k: int, **settings: Any) -> Any:
    """A working model over the regimens: an intercept and a duration coefficient."""
    frame, _ = make_longitudinal(n=N, seed=0)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, msm=DOSE, **learners(**settings)).fit(
        frame, outcome="Y", **NODES, id="cluster"
    )


FITS: dict[str, Callable[..., Any]] = {
    "end of study": fit_end_of_study,
    "survival": fit_survival,
    "competing risks": fit_competing,
    "msm": fit_msm,
}

ONE_ZERO_MASS_CLUSTER = {"weights": lambda cluster: (cluster >= 1).astype(float)}


#: The outer fold counts of the witness and the control: in sample, and five whole-cluster
#: folds. The working model is refused above one fold, so it runs in sample only.
DESIGNS = [(kind, 1) for kind in FITS] + [(kind, 5) for kind in FITS if kind != "msm"]


def floor(n_folds: int) -> int:
    """The interval floor of a fit: its measured family floor, in sample or cross-fitted."""
    return CLUSTER_INTERVAL_FLOORS["ltmle_in_sample" if n_folds == 1 else "ltmle_cross_fitted"]


def control_count(design: tuple[str, int]) -> int:
    """The fewest clusters at which every estimate of a kind keeps the ordinary status.

    The family floor holds for the always-versus-never contrast of the measured design alone
    (:func:`~cleverly.longitudinal.estimator.measured_design`). None of these fits has it: the
    end-of-study panel has a three-level node, and a curve or a working model reports no such
    contrast. So every control sits at the default floor, and below it every estimate takes
    :data:`~cleverly._inference_status.FEW_CLUSTER_ESTIMAND_STATUS`.
    """
    del design
    return DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS


@pytest.fixture(scope="module", params=DESIGNS, ids=[f"{k}-{f} fold(s)" for k, f in DESIGNS])
def design(request: pytest.FixtureRequest) -> tuple[str, int]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def kind(design: tuple[str, int]) -> str:
    return design[0]


@pytest.fixture(scope="module")
def few_result(design: tuple[str, int]) -> Any:
    kind, n_folds = design
    with warnings.catch_warnings():
        # The filter keeps a learner warning quiet at so few clusters.
        warnings.simplefilter("ignore")
        return FITS[kind](floor(n_folds) - 1, n_folds=n_folds)


@pytest.fixture(scope="module")
def control_result(design: tuple[str, int]) -> Any:
    kind, n_folds = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FITS[kind](control_count(design), n_folds=n_folds)


def assert_reports_show(result: Any, kind: str) -> None:
    """The report each kind of fit adds to ``to_frame`` keeps its spread names and the label."""
    if kind in CURVES:
        views = ("risk", "survival") if kind == "survival" else ("risk",)
        for curve in (result.curve(view) for view in views):
            assert set(curve.columns) >= INFERENTIAL_COLUMNS - {"p_value"}
            assert_no_diagnostic_name(curve.columns)
            assert (curve["inference"] == FEW).all()
            assert (curve["support"] == "experimental").all()
            for row in curve.itertuples():
                estimate = result[row.parameter]
                assert row.std_err == estimate.std_error
        if kind == "competing risks":
            total = result.incidence_total()
            assert "std_err" in total.columns
            assert_no_diagnostic_name(total.columns)
            assert (total["inference"] == FEW).all()
            assert (total["support"] == "experimental").all()
            # The same spread through the joint covariance, a second path to the number.
            for row in total.itertuples():
                names = [
                    name
                    for name, (regimen, _, horizon) in result.parameter_index.items()
                    if name.startswith("cif_regimen[")
                    and (regimen, horizon) == (row.regimen, row.time)
                ]
                assert len(names) == len(result.config.causes)
                expected = float(np.sqrt(result.covariance(names).sum()))
                assert row.std_err == pytest.approx(expected, rel=1e-12)
    elif kind == "msm":
        columns = result.coefficients().columns
        assert set(columns) >= INFERENTIAL_COLUMNS
        assert_no_diagnostic_name(columns)
    else:
        contrast = result.contrast(
            lambda psi: psi[0] - psi[1], ["ey_regimen[always]", "ey_regimen[never]"]
        )
        assert contrast.inference == FEW
        assert contrast.reference_df == result["ey_regimen[always]"].reference_df
        assert contrast.support.label == "experimental"
        low, high = contrast.ci
        assert low < high


def assert_no_diagnostic_name(names: Any) -> None:
    """A shown status keeps the inferential names, so none is renamed to a plug-in one."""
    assert not set(names) & {"plugin_std_err", "plugin_interval_lower", "plugin_interval_upper"}


class TestFewClustersShowALabelledLongitudinalInterval:
    """The first witness: one below the floor shows t(J - 2) under the label, on every kind."""

    def test_the_estimates_show_the_plug_in_t_interval(
        self, few_result: Any, design: tuple[str, int]
    ) -> None:
        assert_shows_interval(few_result, FEW, floor(design[1]) - 3)

    def test_each_report_keeps_its_spread_names_and_the_label(
        self, few_result: Any, kind: str
    ) -> None:
        assert_reports_show(few_result, kind)

    def test_an_explicit_band_request_is_skipped_and_named(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_end_of_study(IN_SAMPLE_FLOOR - 1, simultaneous=True)
        assert not caught
        assert result.simultaneous is None
        assert NO_T_REFERENCE_BANDS in result.summary(detail="full").section("Settings").lines

    def test_the_nuisance_item_and_the_assessment_carry_the_note(self) -> None:
        assert_assessment_note(fit_end_of_study(IN_SAMPLE_FLOOR - 1), FEW)

    def test_the_truncation_curve_replays_under_the_status(self) -> None:
        result = fit_end_of_study(IN_SAMPLE_FLOOR - 1)
        curve = result.diagnostics.truncation_curve(bounds=[0.01, 0.05])
        fitted = curve[curve["is_fitted_bound"]]
        assert len(fitted) == len(result.estimates)
        for row in fitted.itertuples():
            assert row.psi == result[row.estimand].psi

    def test_zero_mass_clusters_do_not_count(self) -> None:
        result = fit_end_of_study(IN_SAMPLE_FLOOR, **ONE_ZERO_MASS_CLUSTER)
        # The witness that the labels alone would keep the interval.
        assert result.data.n_clusters == IN_SAMPLE_FLOOR
        assert_shows_interval(result, FEW, IN_SAMPLE_FLOOR - 3)
        assert dict(result.summary().header)["Clusters"] == (
            f"{IN_SAMPLE_FLOOR}, positive weight mass in {IN_SAMPLE_FLOOR - 1}"
        )

    def test_the_study_workflow_reads_a_t_reference(self) -> None:
        """39 clusters through ``CausalStudy``: a t interval with 37 df, and no default band."""
        frame, _ = make_longitudinal(n=390, seed=0, cluster_size=10)
        result = (
            CausalStudy(
                frame,
                design=LongitudinalTreatment(
                    outcome="Y",
                    treatment=("A1", "A2"),
                    baseline=("W1", "W2"),
                    time_varying=((), ("L2",)),
                    censoring=("C1", "C2"),
                    cluster="id",
                ),
            )
            .identify(RegimeMean({"always": 1, "never": 0}, reference="always"))
            .estimate(
                outcome_learner=LinearRegression(),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
                simultaneous=True,
            )
        )
        assert result.data.n_clusters == FEW_CLUSTER_THRESHOLD - 1
        assert_keeps_inference(result)
        assert {estimate.reference_df for estimate in result.estimates.values()} == {37}
        assert result.simultaneous is None
        assert NO_T_REFERENCE_BANDS in result.summary(detail="full").section("Settings").lines


class TestTwentyClustersKeepTheLongitudinalInterval:
    """The second witness: the same rows at the floor keep every interval, with t(J - 2)."""

    def test_the_estimates_keep_their_interval(
        self, control_result: Any, kind: str, design: tuple[str, int]
    ) -> None:
        assert_keeps_inference(control_result)
        if kind in CURVES:
            assert set(control_result.curve().columns) >= INFERENTIAL_COLUMNS - {"p_value"}
            assert "inference" not in control_result.curve().columns
        if kind == "competing risks":
            assert "std_err" in control_result.incidence_total().columns
        assert STATUS_RECORDS[FEW].shown_reason not in control_result.summary()
        assert {e.reference_df for e in control_result.estimates.values()} == {
            control_count(design) - 2
        }

    def test_the_default_bands_are_built_at_forty_clusters(self) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        assert result.simultaneous is not None
        assert {e.reference_df for e in result.estimates.values()} == {None}
        assert NO_SIMULTANEOUS_BANDS not in str(result.summary(detail="full"))
        assert NO_T_REFERENCE_BANDS not in str(result.summary(detail="full"))

    def test_a_t_reference_skips_the_default_band(self) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1, simultaneous=True)
        assert_keeps_inference(result)
        assert result.simultaneous is None
        assert NO_T_REFERENCE_BANDS in result.summary(detail="full").section("Settings").lines
        assert {e.reference_df for e in result.estimates.values()} == {37}

    def test_the_cluster_count_moves_no_point_estimate(
        self, few_result: Any, control_result: Any, design: tuple[str, int]
    ) -> None:
        assert list(few_result.estimates) == list(control_result.estimates)
        if design[1] > 1:
            # A grouped split reads the labels, so other labels draw other folds and move
            # the point estimates. Only the in-sample fit isolates the status.
            assert few_result.folds.origin.scheme == control_result.folds.origin.scheme
            assert few_result.folds.origin.scheme == "grouped"
            return
        for name, estimate in few_result.estimates.items():
            assert estimate.psi == control_result[name].psi

    def test_all_positive_weights_keep_the_interval(self) -> None:
        result = fit_end_of_study(
            DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS, weights=lambda cluster: np.ones(cluster.size)
        )
        assert_keeps_inference(result)
        assert "positive weight mass in" not in result.summary()


MEANS = ("ey_regimen[always]", "ey_regimen[never]")
CONTRAST = "ate_regimen[always vs never]"


def one_node_panel(n: int = N, *, seed: int = 11, levels: int = 2) -> pd.DataFrame:
    """The design the in-sample floor rests on: one binary treatment node, no censoring.

    ``levels=3`` draws a three-level node instead, a design no floor cell measured.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    if levels == 2:
        a = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.4 * w1))).astype(float)
    else:
        a = rng.integers(0, levels, n).astype(float)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.3 * w1 + 0.5 * (a > 0))))).astype(float)
    return pd.DataFrame({"W1": w1, "A": a, "Y": y})


def fit_one_node(k: int, *, levels: int = 2, **settings: Any) -> Any:
    """In-sample ``LTMLE`` on :func:`one_node_panel` in ``k`` clusters."""
    frame = one_node_panel(levels=levels).assign(cluster=lambda f: labels(len(f), k))
    return LTMLE({"never": 0, "always": 1}, reference="never", **learners(**settings)).fit(
        frame, outcome="Y", treatment=["A"], baseline=["W1"], id="cluster"
    )


def assert_floor_holds_for_the_contrast(result: Any) -> None:
    """At the family floor the contrast is ordinary and each mean shows a labelled interval."""
    assert result.data.n_clusters == IN_SAMPLE_FLOOR
    assert result[CONTRAST].inference == "influence_curve"
    for name in MEANS:
        estimate = result[name]
        assert estimate.inference == FEW_CLUSTER_ESTIMAND_STATUS
        assert estimate.support.label == "experimental"
        assert "X54" in estimate.support.evidence
        assert estimate.reference_df == IN_SAMPLE_FLOOR - 2
        low, high = estimate.ci
        assert low < high
    assert result.inference_status == "influence_curve"


class TestTheFloorHoldsForTheMeasuredContrast:
    """R1 of the X54 post-run ruling: a family floor certifies the contrast its study measured.

    The in-sample and cross-fitted LTMLE floors rest on cells that read the always-versus-never
    contrast alone. Every other estimate keeps the default longitudinal floor: below it, and at
    or above its family's floor, it shows its interval under the parameter status
    :data:`~cleverly._inference_status.FEW_CLUSTER_ESTIMAND_STATUS`.
    """

    def test_at_the_floor_only_the_contrast_is_ordinary(self) -> None:
        assert_floor_holds_for_the_contrast(fit_one_node(IN_SAMPLE_FLOOR))

    @pytest.mark.parametrize(
        "fit",
        [
            pytest.param(lambda k: fit_one_node(k, levels=3), id="three-level node"),
            pytest.param(fit_end_of_study, id="two nodes with censoring"),
        ],
    )
    def test_an_unmeasured_design_keeps_the_default_floor(self, fit: Any) -> None:
        """m1: the in-sample floor rests on one binary node without censoring. A fit of
        another design reads every estimate, the contrast included, at the default floor."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit(IN_SAMPLE_FLOOR)
        assert {e.inference for e in result.estimates.values()} == {FEW_CLUSTER_ESTIMAND_STATUS}

    def test_a_design_read_as_measured_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mutation that drops the design check certifies the three-level contrast."""
        monkeypatch.setattr(
            longitudinal_estimator_module, "measured_design", lambda data, family: True
        )
        result = fit_one_node(IN_SAMPLE_FLOOR, levels=3)
        assert result[CONTRAST].inference == "influence_curve"

    def test_a_shown_parameter_status_warns_nothing(self) -> None:
        """m8: the means' shown parameter status labels their interval and is not a lost
        interval, so neither the fit nor ``_warn_parameter_statuses`` warns about it. A
        withheld parameter status (``constant_node_plugin``) still warns."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_one_node(IN_SAMPLE_FLOOR)
        assert not [w for w in caught if "reported parameter(s) take the" in str(w.message)]
        warn = longitudinal_estimator_module._warn_parameter_statuses
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn(result.estimates)
        assert not caught
        withheld = dict(result.estimates)
        withheld[CONTRAST] = replace(withheld[CONTRAST], inference="constant_node_plugin")
        with pytest.warns(DataWarning, match="'constant_node_plugin' status"):
            warn(withheld)

    def test_the_measured_designs(self) -> None:
        design = longitudinal_estimator_module.measured_design
        one = fit_one_node(IN_SAMPLE_FLOOR).data
        assert design(one, "ltmle_in_sample")
        assert not design(one, "ltmle_cross_fitted")
        assert not design(one, None)
        assert not design(fit_one_node(IN_SAMPLE_FLOOR, levels=3).data, "ltmle_in_sample")
        two = fit_end_of_study(IN_SAMPLE_FLOOR).data
        assert not design(two, "ltmle_in_sample")

    @pytest.mark.parametrize("kind", ["survival", "competing risks", "msm"])
    def test_a_curve_or_a_coefficient_keeps_the_default_floor(self, kind: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = FITS[kind](IN_SAMPLE_FLOOR)
        assert {e.inference for e in result.estimates.values()} == {FEW_CLUSTER_ESTIMAND_STATUS}
        assert {e.support.label for e in result.estimates.values()} == {"experimental"}

    def test_the_default_floor_clears_the_label(self) -> None:
        below = fit_one_node(DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS - 1)
        assert {below[name].inference for name in MEANS} == {FEW_CLUSTER_ESTIMAND_STATUS}
        at = fit_one_node(DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS)
        assert {e.inference for e in at.estimates.values()} == {"influence_curve"}

    def test_the_measured_contrast_is_always_against_never(self) -> None:
        measured = longitudinal_estimator_module.measured_contrast
        always, never = Regimen("always", (1, 1)), Regimen("never", (0, 0))
        assert measured(always, never)
        assert measured(never, always)
        assert not measured(Regimen("once", (1, 0)), never)
        assert not measured(always, Regimen("never too", (1, 1)))
        rule = DynamicRegimen("rule", (1, lambda history: history["L2"] > 0), rule_kind="known")
        assert not measured(rule, never)

    def test_a_contrast_read_as_unmeasured_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            longitudinal_estimator_module, "measured_contrast", lambda *regimens: False
        )
        with pytest.raises(AssertionError):
            assert_floor_holds_for_the_contrast(fit_one_node(IN_SAMPLE_FLOOR))

    def test_means_read_at_the_contrast_floor_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            longitudinal_estimator_module,
            "unmeasured_estimand_status",
            lambda status, *args, **kwargs: status,
        )
        with pytest.raises(AssertionError):
            assert_floor_holds_for_the_contrast(fit_one_node(IN_SAMPLE_FLOOR))


class TestTheMutationsFailTheWitness:
    """Each wrong rule, or a stamp in the wrong place, fails the check its surface passes."""

    def test_restoring_the_influence_curve_status_fails(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row's third witness: the fit stamps what the rule returns."""
        monkeypatch.setattr(
            longitudinal_estimator,
            "_inference_status",
            lambda data, folds, **_: "influence_curve",
        )
        with pytest.raises(AssertionError):
            assert_shows_interval(
                FITS[kind](IN_SAMPLE_FLOOR - 1),
                FEW,
                IN_SAMPLE_FLOOR - 3,
            )

    def test_a_threshold_of_zero_fails_the_witness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The longitudinal path reads its family's floor: both LTMLE floors read as zero."""
        monkeypatch.setattr(
            _inference_status,
            "CLUSTER_INTERVAL_FLOORS",
            {
                **_inference_status.CLUSTER_INTERVAL_FLOORS,
                "ltmle_in_sample": 0,
                "ltmle_cross_fitted": 0,
            },
        )
        result = fit_end_of_study(IN_SAMPLE_FLOOR - 1)
        with pytest.raises(AssertionError):
            assert_shows_interval(result, FEW, IN_SAMPLE_FLOOR - 3)
        with pytest.raises(AssertionError):
            assert_assessment_note(result, FEW)

    def test_an_at_or_below_comparison_fails_the_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(longitudinal_estimator, "cluster_inference_status", at_or_below)
        # The mutant still passes the witness, so only the control can catch it.
        assert_shows_interval(
            fit_end_of_study(IN_SAMPLE_FLOOR - 1),
            FEW,
            IN_SAMPLE_FLOOR - 3,
        )
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit_end_of_study(IN_SAMPLE_FLOOR))

    def test_a_status_that_ignores_the_weights_fails_the_zero_mass_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            longitudinal_estimator,
            "_inference_status",
            lambda data, folds, **_: cluster_inference_status(
                data.cluster, minimum=IN_SAMPLE_FLOOR
            ),
        )
        with pytest.raises((AssertionError, ValueError)):
            assert_shows_interval(
                fit_end_of_study(IN_SAMPLE_FLOOR, **ONE_ZERO_MASS_CLUSTER),
                FEW,
                IN_SAMPLE_FLOOR - 3,
            )

    def test_a_replay_without_the_status_fails_the_truncation_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stamp belongs in the builders the replay shares, and not in ``fit`` alone."""
        result = fit_end_of_study(IN_SAMPLE_FLOOR - 1)
        refit = longitudinal_estimator._refit_bound

        def unstamped(*args: Any, **kwargs: Any) -> Any:
            replay = refit(*args, **kwargs)
            estimates = {
                name: replace(estimate, inference="influence_curve")
                for name, estimate in replay.estimates.items()
            }
            return replace(replay, estimates=estimates)

        monkeypatch.setattr(longitudinal_estimator, "_refit_bound", unstamped)
        with pytest.raises(CapabilityError, match="longitudinal_replay_fitted_bound_mismatch"):
            result.diagnostics.truncation_curve(bounds=[0.05])


class TestASavedLongitudinalResult:
    """A saved result loads with the status its fit stamped, and replays."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_few_cluster_fit_keeps_its_status(self, route: str) -> None:
        result = fit_end_of_study(IN_SAMPLE_FLOOR - 1, simultaneous=True)
        restored = assert_round_trips(result, FEW, route)
        # The status and the replay read the same data and folds, so the replay matches.
        curve = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(curve) == len(result.estimates)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_forty_cluster_fit_keeps_its_interval(self, route: str) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        restored = assert_round_trips(result, "influence_curve", route)
        assert restored.simultaneous is not None
        for name, estimate in restored.estimates.items():
            assert estimate.ci == result[name].ci

    def test_supported_fits_keep_their_intervals(self) -> None:
        in_sample = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        assert_keeps_inference(restore(in_sample, "pickle"))
        frame, _ = make_longitudinal(n=N, seed=0)
        unclustered = LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            **learners(n_folds=5, simultaneous=True),
        ).fit(frame, outcome="Y", **NODES)
        assert_keeps_inference(restore(unclustered, "pickle"))
