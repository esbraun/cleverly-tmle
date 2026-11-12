"""Pre-run guards of ``clustered-unequal-cvtmle`` and ``clustered-few-cluster-tmle``.

The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs each study's published
``(property, cell)`` set to equal its declared set, and each published truth to be the declared
law's own. A two-replication run of every cell shows both before the declared run. The law's
truths, its declared parameters and the nominal row count are checked here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import beta, binom, kurtosis, norm
from scipy.stats import t as t_dist

from cleverly.datasets import clustered_dgp
from cleverly.exceptions import CapabilityError, RefusalReason
from tests.studies import clustered_few_cluster_properties as few_properties
from tests.studies import clustered_few_cluster_tmle as few_study
from tests.studies import clustered_unequal_cvtmle as unequal_study
from tests.studies import clustered_unequal_laws as laws
from tests.studies import clustered_unequal_properties as unequal_properties
from tests.studies.evidence.properties import TRUTH_ABSOLUTE_FLOOR

pytestmark = pytest.mark.xdist_group("clustered_unequal_study")

STUDIES = {
    "clustered-unequal-cvtmle": (unequal_study, unequal_properties),
    "clustered-few-cluster-tmle": (few_study, few_properties),
}


@pytest.fixture(scope="module", params=sorted(STUDIES))
def pair(request: pytest.FixtureRequest) -> tuple[object, object]:
    return STUDIES[request.param]


def test_the_declared_cells_are_the_cells_a_run_publishes(pair: tuple[object, object]) -> None:
    study, properties = pair
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    # One replicate per fit set suffices: the cells a run publishes are fixed by the fit set.
    rows = properties.generate_property_rows(n_jobs=1, budget=1)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert set(declared) == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        truth = rows.loc[(rows["property"] == family) & (rows["cell"] == name), "truth"]
        np.testing.assert_allclose(
            truth, cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=TRUTH_ABSOLUTE_FLOOR
        )


def test_no_two_families_share_a_declared_stream(pair: tuple[object, object]) -> None:
    _, properties = pair
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


def test_the_two_studies_draw_from_different_seeds() -> None:
    seeds = {unequal_study.STUDY.seed, unequal_study.STUDY.resampling_seed}
    assert not seeds & {few_study.STUDY.seed, few_study.STUDY.resampling_seed}


class TestTheDeclaredLaw:
    def test_the_declared_parameters(self) -> None:
        """The orchestrator's pre-run decision: no effect modifier, delta 1, gamma -3, 4 SD."""
        assert (laws.DELTA, laws.GAMMA, laws.EFFECT_MODIFIER) == (1.0, -3.0, 0.0)
        assert laws.REQUIRED_SEPARATION == 4.0
        assert "6.42 SD" in laws.GAMMA_PILOT

    def test_the_two_estimands_lie_apart(self) -> None:
        truth = laws.informative_truth()
        assert truth["ate"] - truth["ate_cluster"] == pytest.approx(-0.24798, abs=1e-5)
        # The pilot's spread, 0.0386, times the required separation.
        assert abs(truth["ate"] - truth["ate_cluster"]) >= laws.REQUIRED_SEPARATION * 0.0386

    def test_the_non_informative_estimands_coincide(self) -> None:
        truth = laws.informative_truth(0.0, 0.0)
        assert truth["ate"] == pytest.approx(truth["ate_cluster"], abs=1e-15)

    def test_two_integration_tolerances_agree(self) -> None:
        for arguments in ((laws.DELTA, laws.GAMMA, laws.EFFECT_MODIFIER), (0.0, 0.0, 8.0)):
            loose = laws.informative_truth(*arguments, tolerance=1e-10)
            tight = laws.informative_truth(*arguments)
            for name, value in tight.items():
                assert loose[name] == pytest.approx(value, abs=1e-10)

    def test_a_tensor_quadrature_agrees_on_the_smooth_law(self) -> None:
        """Without the effect modifier the integrand is smooth, so Gauss-Hermite is exact enough."""
        nodes, weights = np.polynomial.hermite_e.hermegauss(40)
        weights = weights / np.sqrt(2.0 * np.pi)
        w1, w2, w3 = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        mass = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        m0, m1 = laws.size_means(laws.DELTA, laws.GAMMA, laws.EFFECT_MODIFIER)
        for arm, expected in ((0.0, m0), (1.0, m1)):
            for index, size in enumerate(laws.SIZES):
                mean = laws.informative_mean(
                    w1,
                    w2,
                    w3,
                    arm,
                    np.full(w1.shape, float(size)),
                    delta=laws.DELTA,
                    gamma=laws.GAMMA,
                    effect_modifier=laws.EFFECT_MODIFIER,
                )
                assert float(np.sum(mass * mean)) == pytest.approx(expected[index], abs=1e-9)

    def test_the_equal_size_law_is_clustered_dgp(self) -> None:
        """``equal10`` reproduces the package law's own quadrature truth."""
        mine = laws.informative_truth(0.0, 0.0, 8.0)
        package = clustered_dgp(10, family="binomial").truth()
        for name in ("ey0", "ey1", "ate"):
            assert mine[name] == pytest.approx(package[name], abs=1e-6)


class TestTheDraws:
    def test_a_draw_holds_n_over_ten_clusters_and_publishes_the_nominal_n(self) -> None:
        frame, truth = unequal_study.draw_scenario(unequal_study.INFORMATIVE, 2_000, 0)
        assert frame["cluster"].nunique() == 200
        assert frame.groupby("cluster").size().between(2, 18).all()
        rows = unequal_study.cleverly_rows(frame, truth, unequal_study.INFORMATIVE, 0)
        assert {row["n"] for row in rows} == {unequal_study.PRIMARY_N}

    def test_the_cluster_average_scenario_reads_the_informative_samples(self) -> None:
        informative, _ = unequal_study.draw_scenario(unequal_study.INFORMATIVE, 2_000, 3)
        average, truth = unequal_study.draw_scenario(unequal_study.CLUSTER_AVERAGE, 2_000, 3)
        assert informative.equals(average)
        assert truth["ate"] == pytest.approx(laws.informative_truth()["ate_cluster"])

    @pytest.mark.parametrize("sizes", ["equal10", "unequal_informative"])
    def test_every_few_cluster_fit_reads_its_t_reference(self, sizes: str) -> None:
        frame = few_properties.draw(sizes, 10, 11)
        assert "ltmle_in_sample" not in few_study.fits_at(10, sizes)
        for fit in few_study.fits_at(10, sizes):
            estimate = few_properties.fit_estimate(fit, frame, 10)
            expected = 5 if fit == "tmle_cv_evaluation" else 8
            assert estimate.reference_df == expected == few_study.expected_reference_df(fit, 10)

    def test_the_fold_evaluated_fit_at_informative_sizes_is_refused_and_absent(self) -> None:
        fit, sizes = few_study.REFUSED
        assert fit not in few_study.fits_at(10, sizes)
        assert fit in few_study.fits_at(10, "equal10")
        assert not any(f"{fit}__{sizes}__" in cell for cell in few_study.PUBLISHED_CELLS)
        assert not any(f"{fit}__{sizes}__" in cell.cell for cell in few_properties.declared_cells())
        frame = few_properties.draw(sizes, 10, 11)
        with pytest.raises(CapabilityError) as raised:
            few_properties.fit_estimate(fit, frame, 10)
        assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
        assert "their sizes differ" in str(raised.value)

    def test_the_descriptions_mirror_the_grid(self) -> None:
        from tests.studies.evidence import descriptions

        assert descriptions._FEW_CLUSTER_COUNTS == few_study.CLUSTER_COUNTS


def _se_ratio_pass(ratio: float, replicates: int, band: tuple[float, float]) -> float:
    """P(the 99% SE-ratio interval lies in ``band``): normal estimate, SD ``0.707 / sqrt(R)``."""
    sd = 0.707 / np.sqrt(replicates)
    half = norm.ppf(0.995) * sd
    low, high = band[0] + half, band[1] - half
    return float(norm.cdf((high - ratio) / sd) - norm.cdf((low - ratio) / sd))


def _coverage_pass(coverage: float, replicates: int, band: tuple[float, float]) -> float:
    """P(the 99% Clopper-Pearson coverage interval lies in ``band``)."""
    k = np.arange(replicates + 1)
    lower = np.where(k > 0, beta.ppf(0.005, k, replicates - k + 1), 0.0)
    upper = np.where(k < replicates, beta.ppf(0.995, k + 1, replicates - k), 1.0)
    inside = (lower >= band[0]) & (upper <= band[1])
    return float(binom.pmf(k[inside], replicates, coverage).sum())


def covariate_pass_probability(ratio: float, replicates: int) -> float:
    """The covariate positive cell's joint pass probability at SE ratio ``ratio``.

    The rule is the SE-ratio and coverage calibration bands
    (``property_verdicts.clustered_inference_verdicts``). Coverage 0.945 is a conservative
    reading under ``t_30``.
    """
    margins = unequal_study.STUDY.margins
    return _se_ratio_pass(ratio, replicates, margins.calibration_se_ratio) * _coverage_pass(
        0.945, replicates, margins.calibration_coverage
    )


#: ``E sqrt(chi2_k / k)`` at ``k = J - V = 30``: the ratio of a perfectly calibrated SE from a
#: 30-df variance to the true SD, under normal cluster totals.
JENSEN_RATIO = 0.9917
#: The pre-declaration re-score of the committed covariate positive cell (Fix-4 plan 2.1).
#: Recorded, not a forecast: the registered pair runs on a fresh stream.
COMMITTED_COVERAGE = {"normal": 0.9325, "t30": 0.9425}


class TestTheCovariatePairDesign:
    """``clustered-unequal-cvtmle``'s covariate pair: R, its stream, and the planning ratio."""

    def test_the_declared_replications_and_stream(self) -> None:
        assert unequal_study.COVARIATE_REPLICATES == 4_000
        cells = {cell.cell: cell for cell in unequal_properties.declared_cells()}
        for name in (
            "cluster_robust_fold_evaluated_cluster_covariate",
            "iid_control_fold_evaluated_cluster_covariate",
        ):
            assert cells[name].replicates == 4_000
            assert cells[name].seed == unequal_properties._seed(
                unequal_properties.CLUSTERED, unequal_properties.COVARIATE_STREAM, 0
            )
        assert "cluster_robust_fold_evaluated" not in cells
        assert "iid_control_fold_evaluated" not in cells
        # The other pairs keep their streams and R.
        assert cells["cluster_robust"].replicates == unequal_study.PROPERTY_REPLICATES
        assert cells["cluster_robust"].seed == unequal_properties._seed(
            unequal_properties.CLUSTERED, "stacked", 0
        )

    def test_the_jensen_ratio(self) -> None:
        k = unequal_properties.COVARIATE_CLUSTERS - unequal_properties.COVARIATE_FOLDS
        exact = float(np.sqrt(2.0 / k) * np.exp(gammaln((k + 1) / 2) - gammaln(k / 2)))
        assert exact == pytest.approx(JENSEN_RATIO, abs=5e-5)

    @pytest.mark.parametrize("ratio", [1.0, JENSEN_RATIO, 0.98])
    def test_r_4000_passes_with_probability_at_least_095(self, ratio: float) -> None:
        assert covariate_pass_probability(ratio, unequal_study.COVARIATE_REPLICATES) >= 0.95

    def test_the_planning_table(self) -> None:
        """The plan's table, so a change of model or margin shows here."""
        expected = {(2_400, 1.0): 0.967, (2_400, 0.98): 0.804, (4_000, 0.98): 0.971}
        for (replicates, ratio), value in expected.items():
            assert covariate_pass_probability(ratio, replicates) == pytest.approx(value, abs=2e-3)

    def test_the_law_s_own_ratio_passes(self) -> None:
        """The kurtosis of the law's cluster totals lowers the ratio; R still clears 0.95.

        No fit: each row carries the exact efficient curve of ``ey1``,
        ``A / g (Y - Q1) + Q1 - psi``, and a cluster its total. With fold-centred totals,
        ``rho = 1 - (2 / k + kappa / J) / 8``.
        """
        rng = np.random.default_rng(20261112)
        frame = laws.draw_cluster_covariate(40_000, rng)
        w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
        a, y = frame["A"].to_numpy(), frame["Y"].to_numpy()
        q1 = laws.covariate_mean(w1, w2, 1.0)
        g = laws.covariate_propensity(w1, w2)
        curve = a / g * (y - q1) + q1 - laws.covariate_truth()["ey1"]
        totals = np.bincount(frame["cluster"].to_numpy(), weights=curve)
        kappa = float(kurtosis(totals, fisher=True))
        clusters = unequal_properties.COVARIATE_CLUSTERS
        k = clusters - unequal_properties.COVARIATE_FOLDS
        ratio = 1.0 - (2.0 / k + kappa / clusters) / 8.0
        assert ratio > 0.98
        assert covariate_pass_probability(ratio, unequal_study.COVARIATE_REPLICATES) >= 0.95

    def test_the_committed_rescore(self) -> None:
        rows = pd.read_parquet(unequal_study.STUDY.artifact("property-replicates.parquet"))
        cell = rows[rows["cell"] == "cluster_robust_fold_evaluated_cluster_covariate"]
        error = (cell["estimate"] - cell["truth"]).abs() / cell["std_error"]
        for name, critical in (("normal", norm.ppf(0.975)), ("t30", t_dist.ppf(0.975, 30))):
            assert float((error <= critical).mean()) == pytest.approx(
                COMMITTED_COVERAGE[name], abs=1e-12
            )


#: The S2 pilot of ``clustered-few-cluster-tmle`` (Fix-4 plan 5.2), run before the declaration
#: under the shipped code: 400 fits of each cross-fitted DR-TMLE cell, on the stream
#: ``stream_seed(STUDY, "pilot", f"{sizes}/J{J}", r)``, with the study's own fit
#: (``clustered_few_cluster_properties._cell_rows``), and no failed fit. Pass probabilities at
#: R = 4,000 come from 400 resamples of 4,000 pilot fits each, scored with the study's rules,
#: margins and 10,000 inner bootstrap replicates. The scripts are ``rows/Fix-4/diag/s2_pilot.py``
#: and ``s2_pilot_power.py`` in the plans repository. Per cell: the pass probabilities of the
#: positive arm (coverage floor and bias margin), of clause 1 (paired IID/cluster ratio upper
#: endpoint below 0.80) and of clause 2 (positive SE-ratio upper endpoint at most 1.20); then
#: the pilot's coverage, paired ratio, SE ratio and largest SE over the empirical SD.
FEW_CLUSTER_PILOT: dict[tuple[str, int], tuple[float, ...]] = {
    ("equal10", 10): (0.9075, 1.0, 1.0, 0.9175, 0.7776, 0.9941, 7.23),
    ("equal10", 20): (1.0, 1.0, 1.0, 0.9550, 0.7318, 1.0617, 3.60),
    ("equal10", 30): (1.0, 1.0, 1.0, 0.9675, 0.7295, 1.0391, 1.63),
    ("unequal_informative", 10): (1.0, 1.0, 1.0, 0.9300, 0.6748, 0.9758, 3.55),
    ("unequal_informative", 20): (1.0, 1.0, 1.0, 0.9500, 0.6246, 0.9540, 1.78),
    ("unequal_informative", 30): (1.0, 1.0, 1.0, 0.9525, 0.6112, 0.9942, 2.61),
}
PILOT_DRAWS = 400
#: The plan's bar for every pass probability of a DR-TMLE cell.
PILOT_FLOOR = 0.90
#: The cells whose positive arm clears the bar by resampling but not by the exact binomial
#: probability at the pilot coverage.
MARGINAL_PILOT_CELLS = {("equal10", 10)}


class TestTheFewClusterPilot:
    """``clustered-few-cluster-tmle``'s DR-TMLE cells clear the plan's pilot bar at R = 4,000."""

    def test_the_pilot_covers_every_dr_tmle_cell(self) -> None:
        declared = {
            (sizes, clusters)
            for sizes in few_study.SIZE_LAWS
            for clusters in few_study.CLUSTER_COUNTS
            if "drtmle_crossfit" in few_study.fits_at(clusters, sizes)
        }
        assert set(FEW_CLUSTER_PILOT) == declared
        assert few_study.PROPERTY_REPLICATES == 4_000

    def test_the_pilot_stream_is_not_the_study_s(self) -> None:
        from tests.studies.evidence.seeds import stream_seed

        for sizes, clusters in FEW_CLUSTER_PILOT:
            pilot = {
                stream_seed(few_study.STUDY, "pilot", f"{sizes}/J{clusters}", r)
                for r in range(PILOT_DRAWS)
            }
            study = {few_properties._seed(sizes, clusters, r) for r in range(4_000)}
            assert not pilot & study

    @pytest.mark.parametrize("cell", sorted(FEW_CLUSTER_PILOT))
    def test_each_pass_probability_clears_the_bar(self, cell: tuple[str, int]) -> None:
        positive, clause1, clause2 = FEW_CLUSTER_PILOT[cell][:3]
        assert min(positive, clause1, clause2) >= PILOT_FLOOR

    @pytest.mark.parametrize("cell", sorted(FEW_CLUSTER_PILOT))
    def test_the_coverage_clause_agrees_with_the_resampling(self, cell: tuple[str, int]) -> None:
        """The exact binomial pass probability of the coverage floor at the pilot coverage.

        The positive arm also needs its bias inside 0.25 SD, which every pilot cell meets by a
        wide margin, so the resampled pass probability sits near this one. At equal sizes and
        J = 10 the two straddle the bar: 0.9075 resampled (Monte Carlo SE about 0.015) against
        0.891 here, at a pilot coverage of 0.9175 whose own SE is about 0.014. That cell is
        marginal (:data:`MARGINAL_PILOT_CELLS`), and the plan does not change R, a margin or the
        law after a pilot result.
        """
        positive, coverage = FEW_CLUSTER_PILOT[cell][0], FEW_CLUSTER_PILOT[cell][3]
        replicates = few_study.PROPERTY_REPLICATES
        floor = few_study.STUDY.margins.coverage_floor
        k = np.arange(replicates + 1)
        lower = np.where(k > 0, beta.ppf(0.005, k, replicates - k + 1), 0.0)
        exact = float(binom.pmf(k[lower >= floor], replicates, coverage).sum())
        assert exact == pytest.approx(positive, abs=0.05)
        assert (exact < PILOT_FLOOR) == (cell in MARGINAL_PILOT_CELLS)
