"""The declared design of ``cross-fitted-longitudinal-msm``.

Every number here was fixed before any registered run.  The truth-binding gate of
``tests/unit/test_method_evidence.py`` needs the published ``(property, cell)`` set to equal the
declared set, and each published truth to be the declared law's own.  A two-replication run of
every fit set shows both before the declared run.

Three sizing rules are pinned.  A positive ``interval_calibration`` arm passes when the 99%
Clopper-Pearson coverage interval lies inside (0.92, 0.98) and the SE-ratio interval inside
(0.93, 1.07).  At the true coverage 0.935, 0.940 and 0.945 the coverage clause holds with
probability 0.85, 0.99 and 1.00 at 4,000 replications.  The band cell's pointwise control needs
a power of at least 0.99 at ``p0 + 0.005`` (the RM36 rule): ten fits of the primary subject give
the coefficient correlation -0.838 and ``p0 = 0.9245``, where 2,400 replications give 0.952 and
4,000 give 0.998.  The overfitting control's SE-ratio ceiling is 0.75, and the in-sample control
of ``canonical-ltmle-crossfit`` measured 0.353 (99% interval 0.346 to 0.360) at the same size.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` of the properties
module counts failed fits per fit set and records no estimate.  On streams 0 to 199 of every fit
set it found no failure.  The declared probe on streams 0 to 1,999 found 5 failures in the
``n_500`` rung and none elsewhere (:data:`FAILURE_PROBE_2000`), so that rung was dropped.  A
rate needs three sizes, so the n = 1,000 rung of ``canonical-ltmle-crossfit`` replaced it, after
its own failure-only probe found no failure.  Before
the run, a fit set with any failure is dropped before the run with its gap stated as a
page limit.  The shared harness refuses a cell that lost a replication, so a failure in the run
itself stops the run.  That cell drops to its red-cell owner with its failure count published,
and the run repeats without it, with no other change.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.  The
primary draws come from ``draw_replicate`` under ``SEED``, not from ``longitudinal-msm``'s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import beta, binom

from tests.studies import canonical_crossfit_longitudinal_msm as study
from tests.studies import crossfit_longitudinal_msm_properties as properties
from tests.studies.default_band_properties import MINIMUM_CONTROL_POWER, control_power
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("crossfit_longitudinal_msm_design")

#: The declared failure-only probe on streams 0 to 1,999, recorded before the run.  The
#: ``root_n_and_efficiency/n_500`` fit set failed 5 times and was dropped before the run.
FAILURE_PROBE_2000 = {
    **dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0),
    "root_n_and_efficiency/n_500": 5,
}
#: The replacement rung's own failure-only probe on streams 0 to 1,999: no failure.
FAILURE_PROBE_N1000 = 0
#: The band cell's design, from ten fits of the primary subject: ``p0`` and the correlation.
BAND_DESIGN = (0.9245, -0.838)
#: The positive-arm pass probability of the coverage clause at 4,000 replications, at true
#: coverage 0.935, 0.940 and 0.945.
COVERAGE_POWER_4000 = (0.8548, 0.9915, 0.9999)
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_000),
    ("double_robustness", "outcome_correct"): (2_000, 1_000),
    ("double_robustness", "mechanism_correct"): (2_000, 1_000),
    ("double_robustness", "both_wrong"): (2_000, 1_000),
    ("root_n_and_efficiency", "n_1000"): (1_000, 700),
    ("root_n_and_efficiency", "n_2000"): (2_000, 700),
    ("root_n_and_efficiency", "n_8000"): (8_000, 700),
    ("interval_calibration", "correctly_specified"): (2_000, 4_000),
    ("interval_calibration", "duration_logit"): (2_000, 4_000),
    ("type_i_error", "sharp_null"): (4_000, 800),
    ("power", "alternative"): (4_000, 800),
    ("targeting_necessity", "targeted"): (2_000, 1_000),
    ("projection_necessity", "declared_weights"): (2_000, 1_000),
    ("crossfit_overfitting", "paired"): (1_000, 8_000),
}


def _coverage_power(coverage: float, replicates: int, low: float, high: float) -> float:
    successes = np.arange(replicates + 1)
    tail = (1.0 - Margins().confidence_level) / 2.0
    lower = np.where(successes > 0, beta.ppf(tail, successes, replicates - successes + 1), 0.0)
    upper = np.where(
        successes < replicates, beta.ppf(1.0 - tail, successes + 1, replicates - successes), 1.0
    )
    inside = (lower >= low) & (upper <= high)
    return float(binom.pmf(successes[inside], replicates, coverage).sum())


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (800, 2_500)
    assert (study.N_FOLDS, study.LEARNER_FOLDS, study.RANDOM_STATE) == (5, 2, 0)
    assert record.reference == "lmtp projected regimen fits"
    assert properties.BAND_REPLICATES == 4_000
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert property_verdicts.OVERFIT_SE_CONTROL_CEILING == 0.75
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    rows = properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0
        )
        assert set(selected["n"]) == {cell.n}
    # The summary adds the fitted rate rows, and then equals the registered cell set.  Two
    # replications make several statistics degenerate, which is not what this reads.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = properties.summarize_properties(rows)
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


def test_the_agreement_rows_read_the_ladder_draws() -> None:
    """``in_sample_agreement`` is paired with the n = 8,000 rung by declaration."""
    cells = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    ladder = cells[("root_n_and_efficiency", "duration__n_8000")]
    agreement = cells[("in_sample_agreement", "duration__in_sample_agreement")]
    assert agreement.seed == ladder.seed
    assert agreement.role == property_verdicts.DIAGNOSTIC_ROLE


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.STUDY.seed, study.STUDY.resampling_seed}
    assert len(mine) == 2
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


def test_the_band_budget_meets_the_control_power_rule() -> None:
    p0, _ = BAND_DESIGN
    assert control_power(p0 + 0.005, 2_400) < MINIMUM_CONTROL_POWER
    assert control_power(p0 + 0.005, properties.BAND_REPLICATES) >= MINIMUM_CONTROL_POWER


def test_the_calibration_power_at_four_thousand() -> None:
    low, high = Margins().calibration_coverage
    observed = tuple(
        round(_coverage_power(coverage, 4_000, low, high), 4) for coverage in (0.935, 0.940, 0.945)
    )
    assert observed == pytest.approx(COVERAGE_POWER_4000, abs=1e-4)


def test_the_logit_truth_solves_the_population_projection() -> None:
    """The longhand Newton truth is the root of the weighted logistic normal equations."""
    coefficients = properties.logit_coefficients(properties.law.PROBS)
    means = properties.regimen_means(properties.law.PROBS)
    _, _, _, score, _ = properties._logit_parts(coefficients, means)
    assert float(np.max(np.abs(score))) < 1e-13
    assert float(coefficients[1]) == properties.LOGIT_TRUTH
