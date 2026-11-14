"""``cv_evaluation=True`` refuses clusters whose fold denominators are random.

The fold-evaluated report averages the validation-fold estimates with equal weight. Each fold
estimate divides by the mass of its clusters that the estimand reads: the total weight (the row
count when unweighted) for the means and the ATE, the treated arm's mass for the ATT, and the
reference arm's mass for the ATC. When that mass differs across clusters, each fold estimate
carries a ratio bias and the equal average multiplies it by about the fold count
(``test_fold_evaluated_ratio_bias.py`` rebuilds that from the law alone). The fit refuses before
any learner and redirects to the stacked report, which estimates the same parameter.

Each branch of the mass has a nonzero witness: a frame where only that branch differs refuses,
and its twin where it is equal fits. The mutation controls swap the mass for row counts, drop the
arm mask, or count a zero-weight cluster, and a witness fails each time.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, CrossFitting, PointTreatment, TMLEMethod
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference import cluster as cluster_module
from cleverly.interventions import Static
from tests.unit import _learned_rule_support as learned
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("fold_evaluated_cluster_refusal")

tmle_module = importlib.import_module("cleverly.estimators.tmle")

FIT = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "id": "cluster"}

#: The redirect every refusal ends with (Eric's wording, 2026-11-12).
ALTERNATIVE = (
    "Use the default, cv_evaluation=False (CrossFitting(fold_evaluation=False) on a method). "
    "It estimates the same effect from all folds at once, avoids this bias, and its interval "
    "still accounts for the clustering."
)
SIZES = (
    "cv_evaluation=True cannot be used with these clusters: their sizes differ (from 10 to 20 "
    "rows across 40 clusters). With clusters of different sizes, giving every fold the same "
    "weight can bias the estimate, and the bias is largest when bigger clusters have different "
    f"outcomes. {ALTERNATIVE}"
)
WEIGHT = (
    "cv_evaluation=True cannot be used with these clusters: their total weight differs (from 10 "
    "to 30 across 40 clusters). With clusters of different total weight, giving every fold the "
    "same weight can bias the estimate, and the bias is largest when heavier clusters have "
    f"different outcomes. {ALTERNATIVE}"
)
TREATED = (
    "cv_evaluation=True cannot be used with these clusters for att: the number of treated rows "
    "per cluster differs (from 3 to 7 across 40 clusters). With clusters that differ in their "
    "number of treated rows, giving every fold the same weight can bias the estimate, and the "
    "bias is largest when clusters with more treated rows have different outcomes. "
    f'{ALTERNATIVE} Or leave out att: estimands=("ate", "ey1", "ey0") can use '
    "cv_evaluation=True with these clusters."
)
CONTROL = (
    "cv_evaluation=True cannot be used with these clusters for atc: the number of control rows "
    "per cluster differs (from 5 to 9 across 40 clusters). With clusters that differ in their "
    "number of control rows, giving every fold the same weight can bias the estimate, and the "
    f"bias is largest when clusters with more control rows have different outcomes. {ALTERNATIVE}"
)
ARM_TWO = (
    "cv_evaluation=True cannot be used with these clusters for att[c vs a]: the number of rows "
    "in arm c per cluster differs (from 2 to 6 across 40 clusters). With clusters that differ in "
    "their number of rows in arm c, giving every fold the same weight can bias the estimate, and "
    "the bias is largest when clusters with more rows in arm c have different outcomes. "
    f'{ALTERNATIVE} Or leave out att: estimands=("ate", "ey") can use cv_evaluation=True with '
    "these clusters."
)


def law(arm_counts: Any, *, seed: int = 0, continuous: bool = False) -> pd.DataFrame:
    """Clusters whose rows take each arm a fixed number of times, one row of counts per cluster.

    ``arm_counts[j][a]`` rows of cluster ``j`` take arm ``a``. The cluster size is the row sum.
    A cluster-level effect makes the rows of one cluster correlated.
    """
    rng = np.random.default_rng(seed)
    counts = np.asarray(arm_counts, dtype=int)
    sizes = counts.sum(axis=1)
    cluster = np.repeat(np.arange(counts.shape[0]), sizes)
    treatment = np.concatenate([np.repeat(np.arange(counts.shape[1]), row) for row in counts])
    n = cluster.size
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    u = rng.normal(0.0, 0.5, counts.shape[0])[cluster]
    signal = 0.2 + 0.4 * w1 - 0.3 * w2 + 0.3 * treatment + u
    outcome = signal + rng.normal(size=n) if continuous else rng.binomial(1, expit(signal))
    # Three arms take the string levels a, b and c, so the refusal names the user's own labels.
    arms = treatment if counts.shape[1] == 2 else np.array(["a", "b", "c"])[treatment]
    return pd.DataFrame(
        {"W1": w1, "W2": w2, "A": arms, "Y": outcome.astype(float), "cluster": cluster}
    )


def alternate(first: list[int], second: list[int], clusters: int = 40) -> list[list[int]]:
    """``clusters`` rows of arm counts, alternating between ``first`` and ``second``."""
    return [first if j % 2 == 0 else second for j in range(clusters)]


def estimator(**settings: Any) -> TMLE:
    regression = settings.pop("continuous", False)
    if regression:
        settings.setdefault("q_bounds", (-20.0, 20.0))
    defaults: dict[str, Any] = {
        "outcome_learner": LinearRegression() if regression else LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": True,
        "n_folds": 5,
        "cv_evaluation": True,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
    }
    return TMLE(**{**defaults, **settings})


def fit(frame: pd.DataFrame, **settings: Any) -> Any:
    weights = {"weights": "w"} if "w" in frame else {}
    return estimator(**settings).fit(frame, **FIT, **weights).single()


def refusal(frame: pd.DataFrame, **settings: Any) -> CapabilityError:
    with pytest.raises(CapabilityError) as raised:
        fit(frame, **settings)
    return raised.value


#: 40 clusters of 10 and 20 rows, half of each treated.
UNEQUAL_SIZES = alternate([5, 5], [10, 10])
#: 40 clusters of 10 rows with 3 or 7 treated.
UNEQUAL_TREATED = alternate([7, 3], [3, 7])
#: 40 clusters of 10 rows with 5 treated.
EQUAL = alternate([5, 5], [5, 5])


class TestTheRefusalText:
    """The exact text of each branch, filled from the data."""

    def test_unequal_sizes(self) -> None:
        raised = refusal(law(UNEQUAL_SIZES))
        assert raised.refusal.reason is RefusalReason.API_REDIRECTION
        assert str(raised) == SIZES
        assert raised.refusal.alternative == ALTERNATIVE

    def test_unequal_weight_at_equal_sizes(self) -> None:
        frame = law(EQUAL)
        frame = frame.assign(w=np.where(frame["cluster"] % 2 == 0, 1.0, 3.0))
        assert str(refusal(frame)) == WEIGHT

    def test_unequal_treated_counts_refuse_att(self) -> None:
        assert str(refusal(law(UNEQUAL_TREATED), estimands=("att",))) == TREATED

    def test_unequal_control_counts_refuse_atc(self) -> None:
        assert str(refusal(law(alternate([5, 5], [9, 5])), estimands=("atc",))) == CONTROL

    def test_the_hint_names_every_arm_estimand_that_differs(self) -> None:
        raised = refusal(law(UNEQUAL_TREATED), estimands=("att", "atc"))
        assert str(raised).startswith(TREATED.split(" Or leave out")[0])
        assert str(raised).endswith(
            ' Or leave out att and atc: estimands=("ate", "ey1", "ey0") can use '
            "cv_evaluation=True with these clusters."
        )

    def test_weights_print_on_the_supplied_scale(self) -> None:
        frame = law(EQUAL)
        # The same relative weights, ten times larger: the stored weights are equal, the
        # message is not.
        frame = frame.assign(w=np.where(frame["cluster"] % 2 == 0, 10.0, 30.0))
        assert "(from 100 to 300 across 40 clusters)" in str(refusal(frame))

    def test_a_three_arm_att_reads_each_arm(self) -> None:
        frame = law(alternate([6, 4, 2], [2, 4, 6]))
        assert str(refusal(frame, estimands=("att",))) == ARM_TWO

    def test_the_refusal_runs_before_any_learner(self) -> None:
        stubs = never_fit_learners()
        with pytest.raises(CapabilityError) as raised:
            estimator(**stubs).fit(law(UNEQUAL_SIZES), **FIT)
        assert str(raised.value) == SIZES
        assert NeverFit.calls == 0


class TestTheTotalMass:
    """Total weight decides the means and the ATE; row counts do not enter."""

    def test_equal_sizes_fit(self) -> None:
        result = fit(law(EQUAL))
        assert result["ate"].reference_df == 35

    def test_unequal_rows_with_equal_weight_fit(self) -> None:
        frame = law(UNEQUAL_SIZES)
        frame = frame.assign(w=np.where(frame["cluster"] % 2 == 0, 2.0, 1.0))
        assert np.isfinite(fit(frame)["ate"].psi)

    def test_a_zero_weight_cluster_does_not_make_masses_unequal(self) -> None:
        counts = [[3, 2], *EQUAL[1:]]
        frame = law(counts)
        frame = frame.assign(w=np.where(frame["cluster"] == 0, 0.0, 1.0))
        result = fit(frame)
        assert result["ate"].reference_df == 34

    def test_reading_row_counts_fails_the_equal_weight_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def rows(cluster: Any, mass: Any, total: Any) -> Any:
            ones = np.ones(len(mass))
            return cluster_module.unequal_positive_cluster_mass(
                cluster, ones * (np.asarray(mass) > 0), ones
            )

        monkeypatch.setattr(tmle_module, "unequal_positive_cluster_mass", rows)
        with pytest.raises(CapabilityError):
            self.test_unequal_rows_with_equal_weight_fit()

    def test_counting_a_zero_weight_cluster_fails_its_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def every_cluster(cluster: Any, mass: Any, total: Any) -> Any:
            return cluster_module.unequal_positive_cluster_mass(
                cluster, np.asarray(mass) + (np.asarray(total) == 0), np.ones(len(mass))
            )

        monkeypatch.setattr(tmle_module, "unequal_positive_cluster_mass", every_cluster)
        with pytest.raises(CapabilityError):
            self.test_a_zero_weight_cluster_does_not_make_masses_unequal()


class TestTheArmMass:
    """The ATT reads the treated arm's mass and the ATC the reference arm's, zeros included."""

    def test_unequal_treated_counts_leave_the_ate_alone(self) -> None:
        assert np.isfinite(fit(law(UNEQUAL_TREATED), estimands=("ate",))["ate"].psi)

    def test_equal_treated_counts_fit_att(self) -> None:
        result = fit(law(EQUAL), estimands=("att", "atc"))
        assert {"att", "atc"} <= set(result.estimates)

    def test_a_cluster_randomized_trial_refuses_att_and_atc(self) -> None:
        frame = law(alternate([10, 0], [0, 10]))
        for name, rows in (("att", "treated"), ("atc", "control")):
            text = str(refusal(frame, estimands=(name,)))
            assert (
                f"for {name}: the number of {rows} rows per cluster differs (from 0 to 10" in text
            )
        assert np.isfinite(fit(frame, estimands=("ate",))["ate"].psi)

    def test_unequal_control_counts_leave_att_alone(self) -> None:
        frame = law(alternate([5, 5], [9, 5]))
        assert np.isfinite(fit(frame, estimands=("att",))["att"].psi)

    def test_dropping_the_arm_mask_fails_the_att_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = cluster_module.unequal_positive_cluster_mass

        def unmasked(cluster: Any, mass: Any, total: Any) -> Any:
            return original(cluster, total, total)

        monkeypatch.setattr(tmle_module, "unequal_positive_cluster_mass", unmasked)
        # The mutant fits the frame that TestTheRefusalText needs refused.
        assert np.isfinite(fit(law(UNEQUAL_TREATED), estimands=("att",))["att"].psi)


class TestDefaultAndAllRequests:
    """A default request refuses on the ATT; ``estimands="all"`` drops it."""

    def test_the_default_request_refuses_on_att(self) -> None:
        frame = law(UNEQUAL_TREATED, continuous=True)
        with pytest.raises(CapabilityError) as raised:
            estimator(continuous=True, estimands=None).fit(frame, **FIT)
        # The default request holds atc too, whose control counts also differ.
        assert str(raised.value) == TREATED.replace("leave out att:", "leave out att and atc:")

    def test_all_drops_att_and_atc(self) -> None:
        frame = law(UNEQUAL_TREATED, continuous=True)
        result = estimator(continuous=True, estimands="all").fit(frame, **FIT).single()
        assert "ate" in result.estimates
        assert not {"att", "atc"} & set(result.estimates)

    def test_all_keeps_att_at_equal_treated_counts(self) -> None:
        frame = law(EQUAL, continuous=True)
        result = estimator(continuous=True, estimands="all").fit(frame, **FIT).single()
        assert {"att", "atc"} <= set(result.estimates)


class TestEveryRoute:
    def test_the_method_layer_refuses(self) -> None:
        study = CausalStudy(
            law(UNEQUAL_SIZES),
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=["W1", "W2"], cluster="cluster"
            ),
        )
        method = TMLEMethod(cross_fitting=CrossFitting(fold_evaluation=True))
        with pytest.raises(CapabilityError) as raised:
            study.identify(ATE()).estimate(
                method,
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                simultaneous=False,
                random_state=0,
            )
        assert str(raised.value) == SIZES

    def test_retarget_refuses_new_weights_and_a_new_estimand(self) -> None:
        result = fit(law(UNEQUAL_TREATED))
        # Supplied weights 1 and 3: stored at mean one, with the scale that recovers them.
        weights = np.where(result.data.cluster % 2 == 0, 0.5, 1.5)
        reweighted = dataclasses.replace(
            result.data, weights=weights, weight_spec=result.data.weight_spec.rescaled(2.0)
        )
        with pytest.raises(CapabilityError) as raised:
            result.estimator.retarget(reweighted, result.nuisance, estimands=("ate",))
        assert str(raised.value) == WEIGHT
        with pytest.raises(CapabilityError) as raised:
            result.estimator.retarget(result.data, result.nuisance, estimands=("att",))
        assert str(raised.value) == TREATED
        # The control: the fit's own request still retargets.
        estimates, _ = result.estimator.retarget(result.data, result.nuisance, estimands=("ate",))
        assert estimates["ate"].psi == pytest.approx(result["ate"].psi, rel=1e-12)

    def test_a_learned_rule_fit_keeps_its_experimental_label(self) -> None:
        """Every learned-rule target carries the ratio term at unequal sizes and is admitted
        experimental, the known-regime value and the contrast as well as the learned value."""
        frame = learned.law_frame(600, 3)
        sizes = np.tile([10, 20], 20)
        frame = frame.assign(cluster=np.repeat(np.arange(40), sizes))
        result = learned.learned_tmle(interventions=[Static(1)]).fit(
            frame, **{**FIT, "covariates": learned.COVARIATES}
        )
        estimates = result.single().estimates
        assert {name.split("[")[0] for name in estimates} == {
            "ey_learned_rule",
            "ey_known_regime",
            "ate_learned_rule",
        }
        for estimate in estimates.values():
            assert estimate.support.label == "experimental"
            assert "unequal_clusters" in estimate.support.reason
            assert estimate.reference_df == 35

    def test_unclustered_weighted_and_att_fits_keep_the_equal_average(self) -> None:
        frame = law(UNEQUAL_SIZES).drop(columns="cluster")
        frame = frame.assign(w=np.linspace(0.5, 2.0, len(frame)))
        rows = {key: value for key, value in FIT.items() if key != "id"}
        result = estimator(estimands=("ate", "att", "atc")).fit(frame, **rows, weights="w")
        assert {"ate", "att", "atc"} <= set(result.single().estimates)


class TestTheRedirectTarget:
    """The fold-mass-weighted average of the fold plug-ins is the stacked point.

    This pins that the redirect target is the stacked point. It is algebra on the same targeted
    predictions and cannot fail unless the stacked point stops being that weighted mean, so it is
    not evidence for the refusal; the law-only ratio-bias tests and the refusal pins are.
    """

    def test_the_mass_weighted_fold_average_is_the_stacked_point(self) -> None:
        # Cluster sizes from 2 to 40, a covariate that tracks the size, and an effect that
        # rises with it, so the fold plug-ins differ and so do the fold masses.
        rng = np.random.default_rng(4)
        counts = [[size // 2, size - size // 2] for size in rng.integers(2, 41, 40)]
        frame = law(counts, seed=4)
        size = frame.groupby("cluster")["Y"].transform("size").to_numpy()
        w1 = (size - 20.0) / 10.0 + rng.normal(0.0, 0.1, size.size)
        signal = -0.3 + 0.5 * w1 + 1.5 * frame["A"].to_numpy() * w1
        frame = frame.assign(W1=w1, Y=rng.binomial(1, expit(signal)).astype(float))
        result = fit(frame, cv_evaluation=False)
        targeted = result.fluctuations["mean"].targeted
        contrast = np.asarray(targeted.arms[1.0]) - np.asarray(targeted.arms[0.0])
        tests = [np.asarray(test) for _, test in result.nuisance.folds]
        masses = np.array([test.size for test in tests], dtype=float)
        fold_psi = np.array([contrast[test].mean() for test in tests])
        weighted = float(np.sum(masses / masses.sum() * fold_psi))
        equal = float(fold_psi.mean())
        assert weighted == pytest.approx(result["ate"].psi, abs=1e-12)
        # The witness: the equal average of the same fold plug-ins differs.
        assert abs(equal - result["ate"].psi) > 1e-3
