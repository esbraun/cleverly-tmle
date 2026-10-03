"""At unequal cluster sizes, the cluster-summed curve is the ratio delta-method curve.

The point estimate is a row mean, :math:`\\hat\\psi = \\sum_j A_j / \\sum_j N_j` with
:math:`A_j` the cluster total of :math:`\\phi_i` and :math:`N_j` the cluster size. It converges
to the row-weighted mean :math:`\\mu_I` of Wang, Park, Small and Li (2024). Every target
builder writes the curve as :math:`D_i = \\phi_i - \\hat\\psi`, so the cluster total is
:math:`S_j = A_j - N_j\\hat\\psi`, and ``influence_variance`` gives
:math:`J\\,\\widehat{\\mathrm{var}}(S_j)/n^2`: the delta-method variance of the ratio of two
cluster means. R ``ltmle`` 1.3-0 computes the same number (``HouseholdIC``, ``R/ltmle.R``
lines 1025 to 1032). R ``tmle`` 2.1.1, ``tmle3`` and ``ife`` average the curve within a
cluster instead, which is a different variance at unequal sizes.

At equal sizes the term :math:`-N_j\\hat\\psi` shifts every total by one constant, and the
centered variance removes it. An equal-size check is therefore blind to the term, and the
witnesses here use sizes that differ.

Mutation controls, each proved to fail the named test:

* ``cluster_sums`` patched to cluster means: :class:`TestTheClusterSumIsTheRatioDeltaMethod`;
* the level builder patched to add :math:`\\hat\\psi` back to the curve:
  :class:`TestTheFitCurveCarriesTheSizeTerm`;
* the weights mutated to ones: :class:`TestTheRowWeightedFitTargetsTheIndividualAverage`;
* the deleted size branch restored: ``tests/unit/test_cluster_status.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import cluster as cluster_module
from cleverly.inference import influence_variance
from tests.conftest import OracleTreatment
from tests.studies import clustered_unequal_laws as laws

pytestmark = pytest.mark.xdist_group("cluster_ratio_variance")

#: The informative law the unit tests read. The size enters the outcome strongly, and no
#: shared latent modifies the effect, so the two estimands lie far apart in standard errors.
UNIT_LAW: dict[str, float] = {"delta": 1.0, "gamma": -2.0, "effect_modifier": 0.0}


class ExactPropensity:
    """The propensity of the informative law, as ``OracleTreatment`` reads it."""

    @staticmethod
    def propensity(covariates: Any) -> Any:
        x = np.asarray(covariates, dtype=float)
        return laws.informative_propensity(x[:, 0], x[:, 1])


def ratio_table(sizes: Any, seed: int = 20261009) -> tuple[Any, Any]:
    """A curve ``phi - mean(phi)`` over clusters of the given sizes."""
    cluster = np.repeat(np.arange(len(sizes)), sizes)
    phi = np.random.default_rng(seed).normal(loc=0.7, scale=1.0, size=cluster.size)
    return phi - phi.mean(), cluster


def ratio_rule(curve: Any, cluster: Any) -> float:
    """``J var(A_j - N_j psi) / n**2``, the delta method for a ratio of two cluster means."""
    n = curve.size
    psi = 0.0  # The curve is already centered at the row mean.
    totals = np.array([curve[cluster == j].sum() for j in np.unique(cluster)])
    sizes = np.bincount(cluster)
    return float(totals.size * np.var(totals - sizes * psi, ddof=1) / n**2)


def household_ic(curve: Any, cluster: Any) -> float:
    """R ``ltmle`` 1.3-0 ``HouseholdIC``: cluster sums times ``J / n``, variance over ``J``."""
    j = np.unique(cluster).size
    ic = np.array([curve[cluster == c].sum() for c in np.unique(cluster)]) * j / curve.size
    return float(np.var(ic, ddof=1) / j)


def cluster_mean_rule(curve: Any, cluster: Any) -> float:
    """R ``tmle`` 2.1.1 and ``tmle3``: ``var(by(IC, id, mean)) / J``."""
    means = np.array([curve[cluster == c].mean() for c in np.unique(cluster)])
    return float(np.var(means, ddof=1) / means.size)


def uncentered_rule(curve: Any, cluster: Any, psi: float) -> float:
    """``J var(A_j) / n**2``: the cluster totals of the curve without its ``-psi`` term."""
    totals = np.array([(curve[cluster == c] + psi).sum() for c in np.unique(cluster)])
    return float(totals.size * np.var(totals, ddof=1) / curve.size**2)


class TestTheClusterSumIsTheRatioDeltaMethod:
    SIZES = (1, 2, 3, 4, 5, 6)

    def test_it_equals_the_ratio_rule_and_ltmle(self) -> None:
        curve, cluster = ratio_table(self.SIZES)
        value = influence_variance(curve, cluster)
        assert value == pytest.approx(ratio_rule(curve, cluster), rel=1e-12)
        assert value == pytest.approx(household_ic(curve, cluster), rel=1e-12)

    def test_it_differs_from_the_cluster_mean_and_uncentered_rules(self) -> None:
        """Nonzero witness: at unequal sizes the three rules give different numbers."""
        curve, cluster = ratio_table(self.SIZES)
        value = influence_variance(curve, cluster)
        assert abs(cluster_mean_rule(curve, cluster) / value - 1.0) > 0.10
        assert abs(uncentered_rule(curve, cluster, 0.7) / value - 1.0) > 0.10

    def test_all_four_rules_agree_at_equal_sizes(self) -> None:
        curve, cluster = ratio_table((4,) * 6)
        value = influence_variance(curve, cluster)
        for rule in (ratio_rule, household_ic, cluster_mean_rule):
            assert rule(curve, cluster) == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert uncentered_rule(curve, cluster, 0.7) == pytest.approx(value, rel=1e-12)


def fit_unit_law(
    estimator: type, frame: Any, *, cross_fit: bool, estimands: tuple[str, ...], **roles: Any
) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": LogisticRegression(C=1e6, max_iter=1000),
        "treatment_learner": OracleTreatment(ExactPropensity()),
        "cross_fit": cross_fit,
        "estimands": estimands,
        "simultaneous": False,
        "random_state": 0,
        "g_bounds": (1e-9, 1.0 - 1e-9),
    }
    if cross_fit:
        settings["n_folds"] = 5
    if estimator is DRTMLE:
        settings["reduced_outcome_learner"] = LinearRegression()
        settings["reduced_treatment_learner"] = LogisticRegression(max_iter=1000)
        settings["learner_folds"] = 2
    return (
        estimator(**settings)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster", **roles)
        .single()
    )


@pytest.fixture(scope="module")
def sixty_clusters() -> Any:
    frame = laws.draw_informative(60, np.random.default_rng(20261010), **UNIT_LAW)
    assert frame.groupby("cluster").size().nunique() > 1
    return frame


FITS = {
    "in sample": (TMLE, False),
    "stacked": (TMLE, True),
    "DRTMLE cross-fitted": (DRTMLE, True),
}


class TestTheFitCurveCarriesTheSizeTerm:
    @pytest.mark.parametrize("kind", ["in sample", "stacked"])
    def test_the_variance_is_the_ratio_variance_of_an_independent_curve(
        self, sixty_clusters: Any, kind: str
    ) -> None:
        """The curve rebuilt from the stored targeted predictions gives the reported variance."""
        estimator, cross_fit = FITS[kind]
        result = fit_unit_law(estimator, sixty_clusters, cross_fit=cross_fit, estimands=("ey0",))
        targeted = result.fluctuations["mean"].targeted
        y = sixty_clusters["Y"].to_numpy()
        a = sixty_clusters["A"].to_numpy()
        g = laws.informative_propensity(
            sixty_clusters["W1"].to_numpy(), sixty_clusters["W2"].to_numpy()
        )
        psi = float(np.mean(targeted.arms[0.0]))
        curve = (1.0 - a) / (1.0 - g) * (y - targeted.observed) + targeted.arms[0.0] - psi
        cluster = sixty_clusters["cluster"].to_numpy()
        assert result["ey0"].psi == pytest.approx(psi, rel=1e-12)
        assert result["ey0"].std_error ** 2 == pytest.approx(ratio_rule(curve, cluster), rel=1e-10)

    @pytest.mark.parametrize("kind", list(FITS))
    def test_the_size_term_is_a_large_share_of_the_variance(
        self, sixty_clusters: Any, kind: str
    ) -> None:
        """Nonzero witness: the totals with and without ``-N_j psi`` differ by far more than 5%."""
        estimator, cross_fit = FITS[kind]
        estimate = fit_unit_law(estimator, sixty_clusters, cross_fit=cross_fit, estimands=("ey0",))[
            "ey0"
        ]
        cluster = sixty_clusters["cluster"].to_numpy()
        curve = np.asarray(estimate.influence_curve)
        assert estimate.std_error**2 == pytest.approx(ratio_rule(curve, cluster), rel=1e-10)
        assert uncentered_rule(curve, cluster, estimate.psi) > 1.05 * estimate.std_error**2

    def test_a_builder_that_adds_psi_back_fails(
        self, sixty_clusters: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cleverly.targets.base as target_base

        original = target_base.make_estimate

        def without_the_size_term(name: str, psi: float, ic: Any, **settings: Any) -> Any:
            return original(name, psi, np.asarray(ic) + psi, **settings)

        monkeypatch.setattr(target_base, "make_estimate", without_the_size_term)
        with pytest.raises(AssertionError):
            self.test_the_variance_is_the_ratio_variance_of_an_independent_curve(
                sixty_clusters, "in sample"
            )


class TestTheRowWeightedFitTargetsTheIndividualAverage:
    """J = 1000 clusters, seed 20261011, the exact propensity, target ``ate``.

    The truths differ by 0.189. When this test was written the unweighted fit read -1.1
    standard errors from the row-weighted truth and -14.7 from the cluster-average truth.
    The fit with weights ``1 / N_j`` read 10.2 and -1.5.
    """

    @pytest.fixture(scope="class")
    def frame(self) -> Any:
        return laws.draw_informative(1000, np.random.default_rng(20261011), **UNIT_LAW)

    @pytest.fixture(scope="class")
    def truth(self) -> dict[str, float]:
        return laws.informative_truth(**UNIT_LAW)

    def test_the_truths_differ_by_design(self, truth: dict[str, float]) -> None:
        assert abs(truth["ate"] - truth["ate_cluster"]) > 0.15

    def test_the_unweighted_fit_targets_the_row_weighted_mean(
        self, frame: Any, truth: dict[str, float]
    ) -> None:
        estimate = fit_unit_law(TMLE, frame, cross_fit=False, estimands=("ate",))["ate"]
        assert abs(estimate.psi - truth["ate"]) < 3.0 * estimate.std_error
        assert abs(estimate.psi - truth["ate_cluster"]) > 6.0 * estimate.std_error

    def test_one_over_the_size_targets_the_cluster_average(
        self, frame: Any, truth: dict[str, float]
    ) -> None:
        weighted = frame.assign(w=1.0 / frame["size"])
        estimate = fit_unit_law(TMLE, weighted, cross_fit=False, estimands=("ate",), weights="w")[
            "ate"
        ]
        assert abs(estimate.psi - truth["ate_cluster"]) < 3.0 * estimate.std_error
        assert abs(estimate.psi - truth["ate"]) > 6.0 * estimate.std_error

    def test_weights_of_one_fail_the_cluster_average_check(
        self, frame: Any, truth: dict[str, float]
    ) -> None:
        estimate = fit_unit_law(
            TMLE, frame.assign(w=1.0), cross_fit=False, estimands=("ate",), weights="w"
        )["ate"]
        assert not abs(estimate.psi - truth["ate_cluster"]) < 3.0 * estimate.std_error


class TestUnequalSizesKeepTheInterval:
    @pytest.mark.parametrize(
        "settings",
        [
            pytest.param({}, id="stacked"),
            pytest.param({"cv_evaluation": True}, id="cv_evaluation"),
            pytest.param({"targeting_scheme": "fold"}, id="fold targeting"),
            pytest.param({"repeats": 2}, id="repeats"),
        ],
    )
    def test_a_cross_fitted_tmle_fit_keeps_the_interval(
        self, sixty_clusters: Any, settings: dict[str, Any]
    ) -> None:
        result = (
            TMLE(
                outcome_learner=LogisticRegression(C=1e6, max_iter=1000),
                treatment_learner=OracleTreatment(ExactPropensity()),
                n_folds=5,
                estimands=("ey0", "ey1", "ate"),
                simultaneous=False,
                random_state=0,
                **settings,
            )
            .fit(sixty_clusters, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
            .single()
        )
        assert result.inference_status == "influence_curve"
        for estimate in result.estimates.values():
            low, high = estimate.ci
            assert low < estimate.psi < high or estimate.scale == "ratio"
            assert estimate.reference_df is None

    def test_unequal_weight_mass_and_a_drtmle_fit_keep_the_interval(
        self, sixty_clusters: Any
    ) -> None:
        weighted = sixty_clusters.assign(w=np.where(sixty_clusters["cluster"] % 2 == 0, 0.5, 2.0))
        result = fit_unit_law(DRTMLE, weighted, cross_fit=True, estimands=("ate",), weights="w")
        assert result.inference_status == "influence_curve"
        assert result["ate"].ci[0] < result["ate"].ci[1]

    def test_a_cluster_mean_variance_fails_the_ratio_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control of :class:`TestTheClusterSumIsTheRatioDeltaMethod`."""

        def cluster_means(curve: Any, cluster: Any) -> Any:
            codes = np.asarray(cluster)
            return np.array([np.asarray(curve)[codes == c].mean() for c in np.unique(codes)])

        monkeypatch.setattr(cluster_module, "cluster_sums", cluster_means)
        with pytest.raises(AssertionError):
            TestTheClusterSumIsTheRatioDeltaMethod().test_it_equals_the_ratio_rule_and_ltmle()
