"""The declared design of ``clustered-cross-fitted-ltmle`` and ``few-cluster-cross-fitted-ltmle``.

Every number here was fixed before any registered run of either study. The truth-binding gate
of ``tests/unit/test_method_evidence.py`` needs each study's published ``(property, cell)`` set
to equal its declared set, and each published truth to be the declared law's own. A
two-replication run of every cell shows both before the declared run.

Two sizing rules are pinned. A ``clustered_inference`` positive arm passes when the 99%
Clopper-Pearson coverage interval lies inside (0.92, 0.98) and the SE-ratio interval inside
(0.93, 1.07). At the probed coverage 0.935 the first holds with probability 0.59 at 2,400
replications and 0.97 at 6,000, so every pair has 6,000. The band cell's pointwise control
needs a power of at least 0.99 at ``p0 + 0.005`` (the RM36 rule of
``tests/unit/test_simultaneous_cell_design.py``); 2,400 replications give 1.0.

``few-cluster-cross-fitted-ltmle`` refuses a failed replication, so its grid rests on a
failure-only probe of 2,000 draws per cell, which records no estimate. A cell with any failure
is dropped before the run. :data:`FAILURE_PROBE` is that record.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta, binom, norm

from cleverly._inference_status import MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS
from tests.studies import clustered_crossfit_ltmle as gated
from tests.studies import clustered_crossfit_ltmle_properties as gated_properties
from tests.studies import clustered_few_cluster_tmle as x24_few
from tests.studies import clustered_longitudinal_laws as law
from tests.studies import clustered_unequal_cvtmle as x24_unequal
from tests.studies import few_cluster_crossfit_ltmle as few
from tests.studies import few_cluster_crossfit_properties as few_properties
from tests.studies.default_band_properties import (
    MINIMUM_CONTROL_POWER,
    _curves,
    control_power,
    design,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins
from tests.studies.evidence.seeds import stream_seed

pytestmark = pytest.mark.xdist_group("clustered_crossfit_ltmle_design")

STUDIES = {
    "clustered-cross-fitted-ltmle": (gated, gated_properties),
    "few-cluster-cross-fitted-ltmle": (few, few_properties),
}

#: The declared failure-only probe: failed fits out of 2,000 draws per cell, on the cell's own
#: first 2,000 streams, recorded by ``few_cluster_crossfit_properties.failure_probe(2_000)`` before
#: the declaration commit.
FAILURE_PROBE = {
    "equal40/J10": 3,
    "equal40/J20": 0,
    "equal40/J30": 0,
    "unequal40/J10": 8,
    "unequal40/J20": 0,
    "unequal40/J30": 0,
}
#: The band cell's design, from ten fits of the primary subject: ``p0`` and the oracle critical.
BAND_DESIGN = (0.8590, 2.409)
#: The positive-arm pass probability of the coverage clause, at coverage 0.935, 0.940, 0.945.
COVERAGE_POWER = {
    2_400: (0.5865, 0.8925, 0.9897),
    4_000: (0.8548, 0.9915, 0.9999),
    6_000: (0.9708, 0.9998, 1.0),
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
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0
        )
        assert set(selected["n"]) == {cell.n}


def test_no_two_families_share_a_declared_stream(pair: tuple[object, object]) -> None:
    _, properties = pair
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


def test_the_studies_draw_from_their_own_seeds() -> None:
    mine = [
        gated.STUDY.seed,
        gated.STUDY.resampling_seed,
        few.STUDY.seed,
        few.STUDY.resampling_seed,
    ]
    assert len(set(mine)) == 4
    others = {
        x24_unequal.STUDY.seed,
        x24_unequal.STUDY.resampling_seed,
        x24_few.STUDY.seed,
        x24_few.STUDY.resampling_seed,
    }
    assert not set(mine) & others


class TestTheGatedStudy:
    def test_the_declared_numbers(self) -> None:
        assert gated.STUDY.publication_policy == "gated"
        assert gated.STUDY.margins == Margins()
        assert (gated.PRIMARY_REPLICATES, gated.CLUSTERS, gated.CLUSTER_SIZE) == (1_600, 100, 40)
        assert gated.PRIMARY_N == 4_000
        assert (gated.N_FOLDS, gated.LEARNER_FOLDS, gated.RANDOM_STATE) == (5, 2, 0)
        assert gated_properties.CLUSTERED_REPLICATES == 6_000
        assert gated_properties.BAND_REPLICATES == 2_400
        assert property_verdicts.CLUSTER_ROBUST_CONTROL_SE_CEILING == 0.80
        assert property_verdicts.CLUSTERED_COVERAGE_GAIN == 0.03
        assert law.DELTA == 0.95

    def test_the_primary_subject_is_cross_fitted_and_clustered(self) -> None:
        frame, truth = gated.draw_scenario(gated.SCENARIO, gated.PRIMARY_N, 0)
        assert frame["id"].nunique() == gated.CLUSTERS
        result = gated.fit_cleverly(frame)
        assert result.folds.origin.scheme == "grouped"
        assert result.data.n_clusters == gated.CLUSTERS
        assert {e.reference_df for e in result.estimates.values()} == {None}
        rows = gated.cleverly_rows(frame, truth, gated.SCENARIO, 0)
        assert {row["n"] for row in rows} == {gated.PRIMARY_N}

    @pytest.mark.parametrize("replicates", sorted(COVERAGE_POWER))
    def test_the_positive_arm_power_table(self, replicates: int) -> None:
        low, high = Margins().calibration_coverage
        observed = tuple(
            round(_coverage_power(coverage, replicates, low, high), 4)
            for coverage in (0.935, 0.940, 0.945)
        )
        assert observed == pytest.approx(COVERAGE_POWER[replicates], abs=1e-4)
        if replicates == gated_properties.CLUSTERED_REPLICATES:
            assert observed[0] >= 0.95

    def test_the_se_ratio_band_is_narrow_enough_at_the_budget(self) -> None:
        """A normal approximation: the 99% half-width of the empirical-SD ratio."""
        half_width = norm.ppf(0.995) / np.sqrt(2 * gated_properties.CLUSTERED_REPLICATES)
        assert half_width < 0.025
        # The probed unequal-size SE ratio, 1.027, keeps its interval under 1.07.
        assert 1.027 + half_width < Margins().calibration_se_ratio[1]

    def test_the_band_cell_has_a_discriminating_control(self) -> None:
        total = None
        for index in range(10):
            frame, _ = law.draw_end_of_study(
                gated.CLUSTERS,
                f"equal{gated.CLUSTER_SIZE}",
                stream_seed(gated.STUDY, "design", gated.BAND_LABEL, index),
            )
            curves = _curves(gated.fit_cleverly(frame, simultaneous=True))
            centred = curves - curves.mean(axis=0)
            covariance = centred.T @ centred / len(centred)
            total = covariance if total is None else total + covariance
        assert total is not None
        sd = np.sqrt(np.diag(total))
        numbers = design(total / np.outer(sd, sd), gated_properties.BAND_REPLICATES)
        assert numbers.p0 == pytest.approx(BAND_DESIGN[0], abs=5e-4)
        assert numbers.critical == pytest.approx(BAND_DESIGN[1], abs=2e-3)
        power = control_power(numbers.p0 + 0.005, gated_properties.BAND_REPLICATES)
        assert power >= MINIMUM_CONTROL_POWER


def _coverage_power(coverage: float, replicates: int, low: float, high: float) -> float:
    """The probability that the 99% Clopper-Pearson interval lies inside ``(low, high)``."""
    k = np.arange(replicates + 1)
    lower = np.where(k > 0, beta.ppf(0.005, k, replicates - k + 1), 0.0)
    upper = np.where(k < replicates, beta.ppf(0.995, k + 1, replicates - k), 1.0)
    inside = (lower >= low) & (upper <= high)
    return float(binom.pmf(k[inside], replicates, coverage).sum())


class TestTheFewClusterStudy:
    def test_the_declared_numbers(self) -> None:
        assert few.STUDY.publication_policy == "reporting"
        assert few.STUDY.reference is None
        assert few.LATENT == "scaled"
        assert (few.PRIMARY_REPLICATES, few.PRIMARY_CLUSTERS) == (1_000, 20)
        assert few.PROPERTY_REPLICATES == 4_000
        assert few.FAILURE_PROBE_DRAWS == 2_000
        assert few_properties.IID_CONTROL_CEILING == 0.80
        # The declared probe dropped both 10-cluster cells.
        assert few.CLUSTER_COUNTS == (20, 30)
        # The package floor of a clustered LTMLE is the smallest measured count.
        assert min(few.CLUSTER_COUNTS) == MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS

    def test_the_grid_is_every_cell_the_probe_found_without_failure(self) -> None:
        assert few.FAILURE_PROBE == FAILURE_PROBE
        kept = {
            f"{sizes}/J{clusters}" for sizes in few.SIZE_LAWS for clusters in few.CLUSTER_COUNTS
        }
        assert kept == {cell for cell, failures in FAILURE_PROBE.items() if failures == 0}

    def test_every_declared_stream_was_probed_without_failure(self) -> None:
        kept = {
            f"{sizes}/J{clusters}" for sizes in few.SIZE_LAWS for clusters in few.CLUSTER_COUNTS
        }
        assert set(few.FAILURE_PROBE_UPPER_STREAMS) == kept
        assert set(few.FAILURE_PROBE_UPPER_STREAMS.values()) == {0}
        assert few.FAILURE_PROBE_DRAWS + 2_000 == few.PROPERTY_REPLICATES
        assert few.PRIMARY_FAILURE_PROBE == 0

    def test_the_control_has_design_power(self) -> None:
        """The IID control's 99% SE-ratio upper bound sits well under its 0.80 ceiling.

        A normal approximation to the bootstrap interval of a ratio of a mean SE to an empirical
        SD at R replications: half-width ``z_0.995 * ratio / sqrt(2 R)``.
        """
        z = norm.ppf(0.995)
        for cell, (cluster_ratio, iid_ratio) in few.CONTROL_DESIGN.items():
            upper = iid_ratio * (1.0 + z / np.sqrt(2 * few.PROPERTY_REPLICATES))
            assert upper < few_properties.IID_CONTROL_CEILING - 0.05, cell
            # The positive arm reads a calibrated ratio on the same stream.
            assert 0.90 < cluster_ratio < 1.10, cell

    def test_every_cell_reads_t_with_j_minus_two(self) -> None:
        for clusters in few.CLUSTER_COUNTS:
            estimate, _ = few_properties.fit_cell("unequal40", clusters, 1)
            assert estimate.reference_df == clusters - 2

    def test_the_descriptions_mirror_the_grid(self) -> None:
        from tests.studies.evidence import descriptions

        assert descriptions._CROSSFIT_FEW_CLUSTER_COUNTS == few.CLUSTER_COUNTS
        assert tuple(descriptions._CROSSFIT_FEW_CLUSTER_SIZES) == few.SIZE_LAWS
