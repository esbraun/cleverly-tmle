"""What a fold-evaluated fit with baseline strata (X28) admits, and what it refuses.

``cv_evaluation=True`` with ``strata=`` reports the arm means, their linear contrasts and the
regime means and their contrasts, on complete data with rows as the unit. The construction and
its exact-law witnesses are in ``test_fold_evaluated_strata_exact.py``. This module pins:

* every other target, a missing outcome or treatment, an intermediate variable and clusters
  refuse before any learner, citing X28, and a direct ``retarget`` that names another target
  refuses before any targeting step;
* fold targeting with strata keeps its X61 (b) refusal, ratios and MSM coefficients keep the
  X61 refusal of ``cv_evaluation=True``, and a learned rule with strata keeps its X11 (m)
  refusal;
* the X28 refusal function names a missing treatment, which the fit's X53 refusal reaches first;
* a drawn split whose validation fold holds no row of a stratum, or no row of an arm inside a
  stratum, refuses before any learner with a data error;
* removing the target check lets a refused request reach a learner, so the check is what
  refuses;
* the fit carries the experimental label until a registered study covers it, and the
  stratified stacked report is unchanged by the fold-evaluated weights;
* the marginal is each draw's share mixture, and under ``repeats=3`` the median of those
  mixtures departs from the mixture of the stratum medians;
* the omitted-variable bound, the robustness value, the truncation curve and the default
  simultaneous band run on such a fit, and the bound's limits read descriptive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.data import CausalData
from cleverly.datasets import make_cde, make_linear_ate
from cleverly.estimators import TMLE
from cleverly.estimators._strata import fold_evaluated_strata_refusal
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, LearnedRule, Shift, Static
from cleverly.msm import MSM
from tests.unit._declaration_support import assert_refused_before_any_call, tmle_module
from tests.unit._natural_course_support import NeverFit, never_fit_learners

COVARIATES = ["W1", "W2", "W3", "W4", "S"]
X28 = "X28 in docs/roadmap.md"
REACHED = "before any learner is fitted"


def strata_frame(n: int = 400) -> pd.DataFrame:
    """``make_linear_ate`` with three baseline strata cut from ``W1``."""
    frame, _ = make_linear_ate(n=n, seed=2)
    return frame.assign(S=np.digitize(frame["W1"], [-0.5, 0.5]))


def settings(**extra: Any) -> dict[str, Any]:
    """Fold-evaluated settings whose every learner is a ``NeverFit``."""
    return {
        **never_fit_learners(),
        "cross_fit": True,
        "n_folds": 3,
        "cv_evaluation": True,
        "q_bounds": (-30.0, 30.0),
        "simultaneous": False,
        "random_state": 0,
        **extra,
    }


def fitted(**extra: Any) -> dict[str, Any]:
    """:func:`settings` with linear learners that fit."""
    return settings(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        **extra,
    )


def fit(estimator: TMLE, frame: pd.DataFrame | None = None, **columns: Any) -> Any:
    return estimator.fit(
        strata_frame() if frame is None else frame,
        outcome="Y",
        treatment=columns.pop("treatment", "A"),
        covariates=columns.pop("covariates", COVARIATES),
        strata=["S"],
        **columns,
    )


def missing_outcome_frame() -> pd.DataFrame:
    frame = strata_frame()
    observed = np.random.default_rng(1).random(len(frame)) < 0.8
    return frame.assign(Y=np.where(observed, frame["Y"], np.nan), Delta=observed.astype(float))


def missing_treatment_frame() -> pd.DataFrame:
    frame = strata_frame()
    recorded = np.random.default_rng(1).random(len(frame)) < 0.8
    return frame.assign(
        A=np.where(recorded, frame["A"].astype(float), np.nan), DeltaA=recorded.astype(float)
    )


def binary_frame() -> pd.DataFrame:
    frame = strata_frame()
    return frame.assign(Y=(frame["Y"] > frame["Y"].median()).astype(float))


def cde_frame() -> pd.DataFrame:
    frame, _ = make_cde(n=400, seed=3)
    return frame.assign(S=(frame["W1"] > 0).astype(int))


#: Each refused request, the fragments of its message, and its class.
REFUSED: dict[str, tuple[Callable[[], Any], tuple[str, ...], type[Exception]]] = {
    "att and atc": (
        lambda: fit(TMLE(**settings(estimands=("ate", "att", "atc")))),
        ("cv_evaluation=True with baseline strata", "Fold evaluation of ['atc', 'att']", X28),
        CapabilityError,
    ),
    "the default target list": (
        lambda: fit(TMLE(**settings())),
        ("['atc', 'att']", "estimands=('ate', 'ey')", X28),
        CapabilityError,
    ),
    "a natural-course target": (
        lambda: fit(TMLE(**settings(estimands=("ate", "par")))),
        ("['par']", X28),
        CapabilityError,
    ),
    "an incremental target": (
        lambda: fit(TMLE(**settings(incremental=[Incremental(2.0)]))),
        ("ey_ipsi", X28),
        CapabilityError,
    ),
    "a shift": (
        lambda: fit(
            TMLE(**settings(policies=[Shift(0.5, cap=None)])),
            frame=strata_frame().assign(D=lambda f: f["A"] + f["W2"]),
            treatment="D",
            treatment_kind="continuous",
        ),
        ("ey_policy", X28),
        CapabilityError,
    ),
    # A nonlinear aggregate has no fold-evaluated construction with or without strata, so
    # it meets the X61 refusal of cv_evaluation=True first.
    "an MSM": (
        lambda: fit(TMLE(**settings(msm=MSM.linear()))),
        ("cv_evaluation=True does not yet support", "msm", "X61"),
        CapabilityError,
    ),
    "a risk ratio": (
        lambda: fit(TMLE(**settings(estimands=("ate", "rr"))), frame=binary_frame()),
        ("cv_evaluation=True does not yet support ['rr']", "X61"),
        CapabilityError,
    ),
    "a missing outcome": (
        lambda: fit(
            TMLE(**settings(estimands=("ate",))),
            frame=missing_outcome_frame(),
            delta="Delta",
        ),
        ("A missing outcome (delta=)", X28),
        CapabilityError,
    ),
    # The cross-fitted composite construction refuses first (X53), whatever the report.
    "a missing treatment": (
        lambda: fit(
            TMLE(**settings(estimands=("ate",))),
            frame=missing_treatment_frame(),
            treatment_delta="DeltaA",
        ),
        ("The cross-fitted composite construction is open (roadmap X53)",),
        CapabilityError,
    ),
    "an intermediate variable": (
        lambda: fit(
            TMLE(**settings(estimands=("ate",))),
            frame=cde_frame(),
            covariates=["W1", "W2", "W3", "S"],
            intermediate="Z",
        ),
        ("intermediate=", X28),
        CapabilityError,
    ),
    "clusters": (
        lambda: fit(
            TMLE(**settings(estimands=("ate",))),
            frame=strata_frame().assign(cluster=np.arange(400) // 4),
            id="cluster",
        ),
        ("Clusters (id=)", "Student t reference", X28),
        CapabilityError,
    ),
    "fold targeting": (
        lambda: fit(TMLE(**settings(estimands=("ate",), targeting_scheme="fold"))),
        ("targeting_scheme='fold'", "fold-local update", "X61"),
        CapabilityError,
    ),
    "a learned rule": (
        lambda: fit(TMLE(**settings(learned_rule=LearnedRule()))),
        ("a learned rule with baseline strata", "X11 (m)"),
        CapabilityError,
    ),
}


class TestEachRefusal:
    @pytest.mark.parametrize("name", list(REFUSED))
    def test_it_refuses_before_any_learner(self, name: str) -> None:
        build, fragments, error = REFUSED[name]
        NeverFit.calls = 0
        assert_refused_before_any_call(build, None, "", *fragments, error=error)

    @pytest.mark.parametrize("name", ["att and atc", "a missing outcome", "clusters"])
    def test_removing_the_check_lets_a_learner_fit(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setattr(tmle_module, "fold_evaluated_strata_refusal", lambda *a, **k: None)
        build, _, _ = REFUSED[name]
        NeverFit.calls = 0
        with pytest.raises(AssertionError, match=REACHED):
            build()
        assert NeverFit.calls > 0

    @pytest.mark.parametrize("name", ["an MSM", "a risk ratio"])
    def test_a_nonlinear_aggregate_does_not_cite_x28(self, name: str) -> None:
        build, _, _ = REFUSED[name]
        with pytest.raises(CapabilityError) as caught:
            build()
        assert "X28" not in str(caught.value)
        assert "with baseline strata" not in str(caught.value)

    def test_a_missing_treatment_is_refused_by_the_x28_check_itself(self) -> None:
        """The fit's X53 refusal comes first, so the branch is read directly."""
        data = CausalData.from_frame(
            missing_treatment_frame(),
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            strata=["S"],
            treatment_delta="DeltaA",
        )
        assert data.has_missing_treatment
        refusal = fold_evaluated_strata_refusal(data, ("ate",))
        assert refusal is not None
        assert "A missing treatment beside them" in str(refusal)
        assert X28 in str(refusal)
        complete = CausalData.from_frame(
            strata_frame(), outcome="Y", treatment="A", covariates=COVARIATES, strata=["S"]
        )
        assert fold_evaluated_strata_refusal(complete, ("ate",)) is None

    def test_a_retarget_of_another_target_refuses(self) -> None:
        result = fit(TMLE(**fitted(estimands=("ate",)))).single()
        with pytest.raises(CapabilityError, match=r"\['att'\]"):
            result.estimator.retarget(result.data, result.nuisance, estimands=("ate", "att"))


def _with_rare_stratum(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    """``frame`` with a fourth stratum on its first ``rows`` rows."""
    strata = frame["S"].to_numpy().copy()
    strata[:rows] = 3
    return frame.assign(S=strata)


def _with_rare_arm(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` whose stratum 2 holds exactly one treated row."""
    treatment = frame["A"].to_numpy().copy()
    inside = np.flatnonzero(frame["S"].to_numpy() == 2)
    treatment[inside] = 0
    treatment[inside[0]] = 1
    return frame.assign(A=treatment)


class TestTheSplitRefusals:
    def test_a_fold_without_a_stratum_refuses(self) -> None:
        frame = _with_rare_stratum(strata_frame(), 1)
        assert_refused_before_any_call(
            lambda: fit(TMLE(**settings(estimands=("ate",))), frame=frame),
            None,
            "",
            "holds no row of baseline stratum S=3",
            "cv_evaluation=False",
            "declare fewer baseline strata",
            error=DataError,
        )

    def test_a_fold_without_an_arm_inside_a_stratum_refuses(self) -> None:
        frame = _with_rare_arm(strata_frame())
        assert_refused_before_any_call(
            lambda: fit(TMLE(**settings(estimands=("ate",))), frame=frame),
            None,
            "",
            "holds no row of treatment arm(s) [1] inside baseline stratum S=2",
            error=DataError,
        )

    def test_the_stacked_report_takes_the_same_rows(self) -> None:
        """The refusal is the fold-evaluated report's: the stacked fit reaches a learner."""
        frame = _with_rare_arm(strata_frame())
        NeverFit.calls = 0
        with pytest.raises(AssertionError, match=REACHED):
            fit(TMLE(**settings(estimands=("ate",), cv_evaluation=False)), frame=frame)
        assert NeverFit.calls > 0


class TestWhatTheFitAdmits:
    @pytest.mark.parametrize(
        "extra",
        [
            {"estimands": ("ate", "ey")},
            {"estimands": ("ey1", "ey0")},
            {"interventions": [Static(1.0, name="treat"), Static(0.0, name="control")]},
        ],
        ids=["arm means and ate", "binary arm means", "regimes"],
    )
    def test_the_whitelisted_targets_fit(self, extra: dict[str, Any]) -> None:
        result = fit(TMLE(**fitted(**extra))).single()
        stratum = [name for name in result.estimates if name.endswith("[S=0]")]
        assert stratum
        for estimate in result.estimates.values():
            assert abs(float(np.mean(estimate.influence_curve))) < 1e-8
            assert np.isfinite(estimate.std_error)

    def test_weights_fit(self) -> None:
        frame = strata_frame().assign(weight=lambda f: 1.0 + (f["W2"] > 0))
        result = fit(TMLE(**fitted(estimands=("ate",))), frame=frame, weights="weight").single()
        assert abs(float(np.mean(result.estimates["ate[S=1]"].influence_curve))) < 1e-8

    @pytest.mark.parametrize("repeats", [1, 3])
    def test_each_draw_reports_the_share_mixture(self, repeats: int) -> None:
        # At this seed the three draws order the strata differently, so the medians come from
        # different draws (seed 0 orders them alike, and the gap is then zero).
        result = fit(TMLE(**fitted(estimands=("ate",), repeats=repeats, random_state=1))).single()
        assert {"ate", "ate[S=0]", "ate[S=1]", "ate[S=2]"} <= set(result.estimates)
        codes = np.asarray(result.data.strata)
        shares = np.bincount(codes, minlength=3) / result.data.n
        strata = [f"ate[{result.data.stratum_label(code)}]" for code in range(3)]

        def mixture(psi: Any) -> float:
            return float(shares @ np.array([psi[name] for name in strata]))

        assert len(result.repeats) == repeats
        for repeat in result.repeats:
            assert repeat.psi["ate"] == pytest.approx(mixture(repeat.psi), abs=1e-12)
        reported = {name: estimate.psi for name, estimate in result.estimates.items()}
        gap = abs(reported["ate"] - mixture(reported))
        if repeats == 1:
            assert gap < 1e-12
        else:
            # The median does not commute with the mixture (docs/technical-reference/cv-tmle.md).
            assert gap > 1e-3, gap
            assert reported["ate"] == pytest.approx(
                float(np.median([repeat.psi["ate"] for repeat in result.repeats])), abs=1e-12
            )

    def test_the_fit_is_experimental(self) -> None:
        result = fit(TMLE(**fitted(estimands=("ate",)))).single()
        for estimate in result.estimates.values():
            assert estimate.support.label == "experimental"
            assert "No registered study covers this configuration" in estimate.support.reason

    def test_the_stacked_stratified_report_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stacked fit never reads the fold-evaluated weights."""

        def never(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the stacked report read the fold-evaluated stratum weights")

        monkeypatch.setattr(tmle_module, "fold_evaluated_stratum_weights", never)
        result = fit(TMLE(**fitted(estimands=("ate",), cv_evaluation=False))).single()
        assert "ate[S=0]" in result.estimates


class TestTheDownstreamSurfaces:
    """The reports that read a fold-evaluated stratified fit's estimates and curves."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return fit(TMLE(**fitted(estimands=("ate", "ey"), simultaneous=True))).single()

    def test_the_default_simultaneous_band_is_built(self, result: Any) -> None:
        band = result.simultaneous
        assert band is not None
        assert band.support.label == "experimental"
        assert 1.959 < float(band.critical_value) < 10.0
        for name, estimate in result.estimates.items():
            lower, upper = band.bands[name]
            pointwise = 1.959 * estimate.std_error
            assert lower < estimate.psi - pointwise and estimate.psi + pointwise < upper

    def test_the_omitted_variable_bound_reads_descriptive(self, result: Any) -> None:
        bound = result.sensitivity.omitted_confounding("ate", cf_y=0.05, cf_d=0.05)
        assert bound.features == ("fold_evaluated_strata",)
        assert bound.psi == result.estimates["ate"].psi
        assert bound.lower < bound.psi < bound.upper
        assert np.isfinite([bound.ci_lower, bound.ci_upper]).all()
        assert bound.ci_lower < bound.lower and bound.upper < bound.ci_upper
        assert bound.limit_reading is not None and not bound.limit_reading.licensed
        assert bound.limit_descriptive

    def test_the_plug_in_limits_refuse(self, result: Any) -> None:
        bound = result.sensitivity.omitted_confounding(
            "ate", cf_y=0.05, cf_d=0.05, nu2_estimator="plugin"
        )
        assert bound.limit_refusal is not None
        assert "fold-evaluated stratum estimates" in bound.limit_refusal.text

    def test_the_robustness_value_reads_descriptive(self, result: Any) -> None:
        value = result.sensitivity.robustness_value("ate")
        assert 0.0 < value["rv"] < 1.0
        assert 0.0 <= value["rva"] <= value["rv"]
        assert value["limit_licensed"] is False

    def test_the_truncation_curve_runs(self, result: Any) -> None:
        frame = result.diagnostics.truncation_curve().to_frame()
        assert len(frame) > 0
        numeric = frame.select_dtypes("number").to_numpy(dtype=float)
        assert np.isfinite(numeric[~np.isnan(numeric)]).all()
