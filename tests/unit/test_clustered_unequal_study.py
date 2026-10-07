"""Pre-run guards of ``clustered-unequal-cvtmle`` and ``clustered-few-cluster-tmle``.

The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs each study's published
``(property, cell)`` set to equal its declared set, and each published truth to be the declared
law's own. A two-replication run of every cell shows both before the declared run. The law's
truths, its declared parameters and the nominal row count are checked here too.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.datasets import clustered_dgp
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
    rows = properties.generate_property_rows(n_jobs=1, budget=2)
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

    def test_every_few_cluster_fit_reads_its_t_reference(self) -> None:
        frame = few_properties.draw("unequal_informative", 10, 11)
        assert "ltmle_in_sample" not in few_study.fits_at(10)
        for fit in few_study.fits_at(10):
            estimate = few_properties.fit_estimate(fit, frame, 10)
            expected = 5 if fit == "tmle_cv_evaluation" else 8
            assert estimate.reference_df == expected == few_study.expected_reference_df(fit, 10)

    def test_the_descriptions_mirror_the_grid(self) -> None:
        from tests.studies.evidence import descriptions

        assert descriptions._FEW_CLUSTER_COUNTS == few_study.CLUSTER_COUNTS
