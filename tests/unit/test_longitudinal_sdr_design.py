"""The declared design of ``longitudinal-sdr`` (T11 of ``X4-plan.md``).

Every number here was fixed before any registered run, with one dated exception.  On 2026-11-11
the study was re-declared under the red-cell rule of 2026-10-07 (the study docstring says why):
the tail-size rule, :data:`~tests.studies.longitudinal_sdr_properties.BELOW_TAIL_SIZE`, and three
new gated fit sets, ``both_correct__n_4800``, ``P_C__n_4800`` and
``weighted_calibration/selection_weighted_n2400``, with their sizes, replications, design probe,
failure probe and pass probabilities.  Those numbers were fixed after the first run and before
the second.  No committed fit set changed.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The two ``sequential_double_robustness`` controls clear the floor of 3 on the exact law of
  ``tests/discrete_law_longitudinal_sdr.py``, with the harness's denominator (6.8).  Every
  positive pattern is exact there.
* The two-size rule of ``P_A``, ``P_B`` and ``P_D`` passes with probability 0.9934 per cell
  when the bias is zero at both sizes (:data:`TWO_SIZE_PASS`).
* A Wald cell gates only at n of at least twice its exact-law tail size.  The guard
  :func:`test_each_wald_cell_gates_only_at_twice_its_tail_size` derives the classification and
  each gated twin's size from the exact laws, and its witnesses fail under each mutation the
  re-declaration guards against.
* Each new gated cell's replication count is the smallest multiple of 400 at which the three
  pass together with probability at least 0.95, and one step of 400 less on either count falls
  short (:func:`test_the_new_gated_replications_follow_the_sizing_rule`).  The pass
  probabilities are computed here from the pilot counts of :data:`COVERAGE_PILOT`, and the
  weighted cell's SE clause passes with probability at least 0.90 under perfect calibration.
* The power cell is sized on the exact law: the contrast 0.15625, the efficiency bound 2.5749
  and n = 4,000 give power 0.970 (:data:`DESIGN_POWER`).
* The band cell's pointwise control needs a power of at least 0.99 at ``p0 + 0.005`` (the RM36
  rule).  A design probe of 400 band fits gives ``p0 = 0.685``.
* The comparator's scenario designs, transcribed by hand into
  ``tests/canonical/longitudinal_sdr_runner.R``, are the Python ones.

Design probes ran on streams 100,000 and above, which no declared cell reads
(:data:`DESIGN_PROBE`).  They found one engine defect before declaration: a cross-fitted fold
density binned its doses on the fold's training rows, not on the out-of-fold grid.  The
``mtp_up`` spread then read 3.6 times its bound.  ``tests/unit/test_sdr_construction.py`` pins
the fix.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` of the properties
module counts failed fits per fit set and records no estimate.  On streams 0 to 1,999 it found
none (:data:`FAILURE_PROBE`).  The 2026-10 probe covered every fit set declared then, and the
2026-11 probe covered the three new fit sets.  A failure in the run itself stops the run.  That
cell drops to its red-cell owner with its failure count published, and the run repeats without
it, with no other change.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.longitudinal import LTMLE, SDR
from tests import discrete_law_competing as competing_law
from tests import discrete_law_longitudinal as law
from tests import discrete_law_longitudinal_policy as policy_law
from tests import discrete_law_longitudinal_sdr as sdr_law
from tests import discrete_law_survival as survival_law
from tests import long_horizon_design
from tests.canonical.longitudinal_sdr import gates
from tests.long_horizon_sampling_sd import kappa, n_kappa
from tests.longitudinal_sdr import COLUMNS
from tests.studies import canonical_longitudinal_sdr as study
from tests.studies import crossfit_longitudinal_msm_properties as msm
from tests.studies import longitudinal_msm_properties as identity_msm
from tests.studies import longitudinal_sdr_properties as properties
from tests.studies import ltmle_crossfit_properties as end_properties
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_ltmle import declared_regimens
from tests.studies.default_band_properties import MINIMUM_CONTROL_POWER, control_power
from tests.studies.evidence.constructions import SDR_TRAINING_FOLD_RECURSION
from tests.studies.evidence.property_verdicts import (
    UNION_MODEL_SE_BAND,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.red_cells import REPORTING_ROWS
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("longitudinal_sdr_design")

RUNNER = ROOT / "tests" / "canonical" / "longitudinal_sdr_runner.R"
#: The fit sets the re-declaration of 2026-11 added.
NEW_FIT_SETS = (
    "sequential_double_robustness/both_correct__n_4800",
    "sequential_double_robustness/P_C__n_4800",
    "weighted_calibration/selection_weighted_n2400",
)
#: The pre-declaration failure-only probe on streams 0 to 1,999: in 2026-10 of every fit set
#: declared then, and in 2026-11 of the three new fit sets.  Failed fits per fit set.  The
#: 2026-11 probe ran ``failure_probe(2_000, n_jobs=4, fit_sets=NEW_FIT_SETS)`` on 2026-11-11 at
#: 58bba782 (6,000 fits, 130 s wall on 4 cores) and printed 0 for each fit set
#: (``_x4_failure_probe_2026-11.log`` beside the plan).  It reads each fit set's n and streams,
#: not its replication count, so the later move of those counts leaves it current.
FAILURE_PROBE = {
    **{
        f"{family}/{label}": 0
        for family, label, *_ in properties.FIT_SETS
        if f"{family}/{label}" not in NEW_FIT_SETS
    },
    **dict.fromkeys(NEW_FIT_SETS, 0),
}
#: ``p0`` of the band cell, from 400 band fits on the design-probe streams.
BAND_P0 = 0.685
#: The exact-law power of the power cell (:func:`design_power`).
DESIGN_POWER = 0.9698
#: The pass probability of the two-size rule per cell when the bias is zero at both sizes.
TWO_SIZE_PASS = 0.9934
#: The coverage pilots of the new gated cells, ``label -> (covered, draws)``, on the design-probe
#: streams of :data:`DESIGN_PROBE` (raw sampler seeds 300,000 + r and 400,000 + r).
COVERAGE_PILOT = {
    "both_correct__n_4800": (1_884, 2_000),
    "P_C__n_4800": (1_895, 2_000),
    "selection_weighted_n2400": (7_496, 8_000),
}
#: The project's family-wise floor for the gated cells a declaration adds, and the grain of R.
FAMILY_FLOOR = 0.95
REPLICATE_GRAIN = 400
#: The AGENT-BRIEF floor of a calibration cell's SE clause under perfect calibration.
PERFECT_CALIBRATION_FLOOR = 0.90
#: Each new gated cell's pass probability at its declared replications, and their product, as
#: computed by :func:`gated_pass` (checked equal to it).
GATED_PASS = {
    "both_correct__n_4800": 0.9967,
    "P_C__n_4800": 0.9997,
    "selection_weighted_n2400": 0.9594,
}
FAMILY_WISE_PASS = 0.956
#: A record, not a check: each control's exact-law limit of the standardized bias at n = 2,000.
CONTROL_LIMITS = {"node2_both_wrong_control": 4.535, "ltmle_P_D_control": 4.578}
#: A record, not a check: the design probe, on streams 100,000 and above.  Each value is
#: ``(draws, SE ratio, coverage, empirical root-n spread over the bound)``; the clustered labels
#: have no bound and record their root-n spread instead.
DESIGN_PROBE = {
    "static": (2_000, 0.989, 0.944, 1.030),
    "dynamic": (2_000, 1.022, 0.949, 0.985),
    "survival_h2": (400, 1.011, 0.963, 1.024),
    "competing_h2": (400, 1.023, 0.965, 1.011),
    "policy_mix": (400, 0.987, 0.950, 1.022),
    "mtp_up": (120, 0.957, 0.925, 1.065),
    "msm_identity": (400, 1.014, 0.950, 1.006),
    "msm_logit": (400, 1.049, 0.960, 0.980),
    "selection_weighted": (1_500, 0.964, 0.933, 1.080),
    # 2026-11, on raw sampler seeds 300,000 + r (SDR) and 400,000 + r (weighted).  ``P_C``'s
    # spread is over the EIF bound, which its misspecified regressions do not attain.
    "both_correct__n_4800": (2_000, 1.005, 0.942, 1.026),
    "P_C__n_4800": (2_000, 0.986, 0.9475, 1.381),
    "selection_weighted_n2400": (8_000, 0.985, 0.937, 1.044),
    "clusters_200": (400, 1.016, 0.950, 1.727),
    "clusters_20": (400, 1.043, 0.958, 3.590),
}
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    **{
        ("sequential_double_robustness", f"{pattern}__n_{size}"): (size, 1_000)
        for pattern in ("both_correct", "P_A", "P_B", "P_C", "P_D")
        for size in (2_000, 8_000)
    },
    ("sequential_double_robustness", "both_correct__n_4800"): (4_800, 1_600),
    ("sequential_double_robustness", "P_C__n_4800"): (4_800, 1_600),
    ("sequential_double_robustness", "controls"): (2_000, 1_000),
    ("root_n_and_efficiency", "n_1000"): (1_000, 700),
    ("root_n_and_efficiency", "n_2000"): (2_000, 700),
    ("root_n_and_efficiency", "n_8000"): (8_000, 700),
    **{
        ("interval_calibration", stream): (2_000, 4_000)
        for stream in (
            "discrete",
            "survival",
            "competing",
            "policy",
            "mtp",
            "msm",
            "msm_logit",
        )
    },
    ("weighted_calibration", "selection_weighted"): (2_000, 4_000),
    ("weighted_calibration", "selection_weighted_n2400"): (2_400, 5_600),
    ("clustered_calibration", "clusters_200"): (2_000, 4_000),
    ("clustered_calibration", "clusters_20"): (2_000, 4_000),
    ("type_i_error", "sharp_null"): (4_000, 800),
    ("power", "alternative"): (4_000, 800),
    ("crossfit_overfitting", "paired"): (1_000, 8_000),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261081, 2026108101)
    assert (study.SMOKE_GATE, study.LEARNER_FOLDS, study.N_FOLDS) == (1e-6, 2, 5)
    assert study.G_BOUNDS == (1e-8, 1.0)
    assert study.CONFIGURATION["construction"] == SDR_TRAINING_FOLD_RECURSION
    assert record.reference == "lmtp"
    assert properties.BAND_REPLICATES == 2_000
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.SDR_TOLERANCE, properties.CONTROL_FLOOR) == (0.01, 3.0)
    assert {
        "clusters_200": 1.75,
        "clusters_20": 3.6,
        "selection_weighted": weighted.EFFICIENCY_SD["static"],
        "selection_weighted_n2400": weighted.EFFICIENCY_SD["static"],
    } == properties.NOISE_SD
    assert (properties.SDR_GATED_N, properties.SDR_GATED_REPLICATES) == (4_800, 1_600)
    assert properties.WEIGHTED_GATED_REPLICATES == 5_600
    assert properties.WALD_GATED_STEP == 400
    assert properties.WALD_REGIME_FACTOR == long_horizon_design.WALD_REGIME_FACTOR == 2
    assert set(properties.SEPARATE_SIZES) == set(properties.NOISE_SD)
    assert set(properties.CALIBRATION_ESTIMANDS) == set(study.CALIBRATION_LABELS)
    assert set(properties.EFFICIENCY_SD) == set(study.CALIBRATION_LABELS)
    assert tuple(properties.CLUSTER_LAYOUT) == study.CLUSTERED_LABELS
    assert set(properties.NOISE_SD) == {
        label for labels, _ in study.SEPARATE_CALIBRATION.values() for label in labels
    }


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # One replicate per fit set suffices: the cells a run publishes are fixed by the fit set,
        # and test_method_evidence.py checks the summary, which needs two, on the committed rows.
        rows = properties.generate_property_rows(n_jobs=1, budget=1)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        if family not in {"simultaneous_coverage"}:
            np.testing.assert_allclose(
                selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0
            )
        assert set(selected["n"]) == {cell.n}


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


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
    assert control_power(BAND_P0 + 0.005, properties.BAND_REPLICATES) >= MINIMUM_CONTROL_POWER


def test_the_power_cell_reaches_its_floor_on_the_exact_law() -> None:
    contrast = law.TRUTH[properties.STATIC]
    assert contrast == pytest.approx(0.15625, abs=1e-15)
    power = design_power(contrast, properties.EFFICIENCY_SD["static"], properties.NULL_N)
    assert power == pytest.approx(DESIGN_POWER, abs=1e-4)
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= 0.999


def test_the_sharp_null_is_zero() -> None:
    assert properties.end_properties.NULL_TRUTH == 0.0


def test_the_efficiency_bounds_are_the_complex_step() -> None:
    assert properties.EFFICIENCY_SD["static"] == pytest.approx(2.5749, abs=5e-5)
    expected = float(np.sqrt(np.sum(law.PROBS * law.eif(properties.DYNAMIC_MEAN) ** 2)))
    assert properties.EFFICIENCY_SD["dynamic"] == expected


def test_the_two_size_rule_passes_at_zero_bias() -> None:
    """``|b| + 2 MCSE`` at 8,000 below the same at 2,000, when both biases are zero."""
    rng = np.random.default_rng(20261081)
    small = rng.normal(size=1_000_000)
    large = 0.5 * rng.normal(size=1_000_000)
    rate = float(np.mean(np.abs(large) + 1.0 < np.abs(small) + 2.0))
    assert rate == pytest.approx(TWO_SIZE_PASS, abs=5e-4)


# ------------------------------------------------------------------ the SDR family law


def _exact_fit(engine: type, nuisances: tuple[str, str]) -> object:
    settings: dict[str, object] = {
        "reference": sdr_law.REFERENCE,
        "n_folds": 1,
        "g_bounds": study.G_BOUNDS,
        "simultaneous": False,
        **properties.sdr_learners(*nuisances),
    }
    if engine is LTMLE:
        settings.update(max_iter=100, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine(declared_regimens(sdr_law.REGIMEN_SPEC), **settings).fit(
            sdr_law.frame(), **COLUMNS
        )


def test_the_family_law_is_a_law() -> None:
    assert sdr_law.PROBS.sum() == pytest.approx(1.0, abs=1e-15)
    assert sdr_law.TRUTH[sdr_law.ESTIMAND] == pytest.approx(-0.0625, abs=1e-15)
    for table in (sdr_law.G1, sdr_law.C1, sdr_law.P_L2, sdr_law.G2, sdr_law.C2, sdr_law.Q):
        assert np.all((table >= 0.25) & (table <= 0.75))


@pytest.mark.parametrize("pattern", sorted(properties.PATTERN_NUISANCES))
def test_every_positive_pattern_is_exact_on_the_family_law(pattern: str) -> None:
    fit = _exact_fit(SDR, properties.PATTERN_NUISANCES[pattern])
    error = fit[sdr_law.ESTIMAND].psi - sdr_law.TRUTH[sdr_law.ESTIMAND]  # type: ignore[index]
    assert abs(error) < 1e-12


@pytest.mark.parametrize("control", sorted(properties.CONTROLS))
def test_each_control_clears_the_floor_on_the_family_law(control: str) -> None:
    """The exact-law error over the fit's own curve SD, at n = 2,000."""
    engine, nuisances = properties.CONTROLS[control]
    fit = _exact_fit(engine, nuisances)
    name = sdr_law.ESTIMAND
    error = fit[name].psi - sdr_law.TRUTH[name]  # type: ignore[index]
    spread = float(np.std(fit.influence_curves[name]))  # type: ignore[attr-defined]
    limit = abs(error) * np.sqrt(properties.CONTROL_N) / spread
    assert limit >= properties.CONTROL_FLOOR
    assert limit == pytest.approx(CONTROL_LIMITS[control], abs=5e-4)


# ------------------------------------------------------------------ construction diagnostics


def test_the_training_longhand_reproduces_the_fit() -> None:
    """The longhand that builds the stitched cell rebuilds the shipped estimate first."""
    frame, _ = properties.overfit_draw(7, n=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = properties.overfit_fit(frame, n_folds=5)
    transforms = properties.longhand_transforms(frame, fit, construction="training")
    contrast = transforms["always"] - transforms["never"]
    assert float(np.mean(contrast)) == pytest.approx(fit[properties.OVERFIT_NAME].psi, abs=1e-12)
    stitched, _ = properties.stitched_estimate(frame, fit)
    assert abs(stitched - fit[properties.OVERFIT_NAME].psi) > 1e-4


# ------------------------------------------------------------------ the comparator


def test_the_runner_transcribes_the_python_designs() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    block = text[text.index("designs <- list(") :]
    block = block[: block.index("\n)\n")]
    named = dict(re.findall(r"^\s+(\w+) = (\w+)", block, flags=re.MULTILINE))
    assert set(named) == set(study.ESTIMANDS) | set(gates.G2_SCENARIOS)
    assert named[study.END_OF_STUDY] == named[study.END_OF_STUDY_CROSSFIT] == "censored"
    assert named[study.SURVIVAL] == "survival"
    assert named[study.MTP] == "uncensored"
    assert f"copies = {study.POLICY_COPIES}L" in text
    assert 'baseline = c("W1", "W2")' in text
    assert 'learners = "SL.cellmeans"' in text


def test_the_declared_red_cell_route_holds_on_the_committed_rows() -> None:
    """The rows the study docstring says block the merge, read from the committed artifacts.

    The study publishes under ``reporting``, because a margins row of the GLM pairing may be
    re-registered after a diagnosis (``X4-plan.md`` 6.2).  The harness reads one policy per
    study, so this test states the rows that stay blocking: every truth row of ``cleverly``,
    every G1 row of the declared run, and every G1 and G2 row of the pre-declaration gates.
    """
    import pandas as pd

    record = study.STUDY
    gates_file = record.artifact("gates.csv")
    if not gates_file.exists():
        pytest.skip("the pairing gates have not run")
    gate_rows = pd.read_csv(gates_file)
    assert set(gate_rows["gate"]) >= {"G1", "G2"}
    assert gate_rows["passes"].all(), gate_rows.loc[~gate_rows["passes"]].to_string()
    if not record.artifact("performance-tests.csv").exists():
        pytest.skip("the declared run has not committed its artifacts")
    performance = pd.read_csv(record.artifact("performance-tests.csv"))
    subject = performance.loc[performance["implementation"] == record.implementation]
    assert not subject.empty
    assert subject["passed"].all(), subject.loc[~subject["passed"]].to_string()
    equivalence = pd.read_csv(record.artifact("equivalence.csv"))
    exact = {(scenario, name) for scenario, names in study.EXACT_ROWS.items() for name in names}
    g1 = equivalence.loc[
        [
            (scenario, name) in exact
            for scenario, name in zip(equivalence["scenario"], equivalence["estimand"], strict=True)
        ]
    ]
    assert len(g1) == len(exact)
    assert g1["passed"].all(), g1.loc[~g1["passed"]].to_string()


def test_the_failure_probe_covered_every_fit_set_and_found_no_failure() -> None:
    assert set(FAILURE_PROBE) == {f"{family}/{label}" for family, label, *_ in properties.FIT_SETS}
    assert not any(FAILURE_PROBE.values())


def test_the_failure_probe_runs_only_the_named_fit_sets() -> None:
    """The filter names its fit sets, and an unknown name is refused rather than skipped."""
    with pytest.raises(KeyError, match="no fit set"):
        properties.failure_probe(1, fit_sets=["weighted_calibration/no_such_label"])
    assert properties.failure_probe(0, fit_sets=NEW_FIT_SETS) == dict.fromkeys(NEW_FIT_SETS, 0)


# ------------------------------------------------------- sizing of the new gated cells

SDR_GATED_LABELS = ("both_correct__n_4800", "P_C__n_4800")
WEIGHTED_GATED_LABEL = "selection_weighted_n2400"
#: Gauss-Hermite nodes and weights for averaging over a normal law of a true SE ratio.
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(41)
_GH_W = _GH_W / _GH_W.sum()


def _coverage_law(label: str, points: int = 400) -> np.ndarray:
    """The pilot coverage's Jeffreys posterior, as ``points`` equal-mass quantiles."""
    covered, draws = COVERAGE_PILOT[label]
    mass = (np.arange(points) + 0.5) / points
    return np.asarray(stats.beta.ppf(mass, covered + 0.5, draws - covered + 0.5))


@cache
def _accepted(replicates: int, clause: str) -> np.ndarray:
    """Whether each count's exact Clopper-Pearson interval at ``replicates`` passes ``clause``.

    ``floor`` is the coverage floor of an exact-ratio pattern; ``band`` is the coverage band of
    a calibration cell.  Both read the endpoints inclusively, as the verdicts do.
    """
    margins = Margins()
    counts = np.arange(replicates + 1)
    tail = (1.0 - margins.confidence_level) / 2.0
    low = np.where(counts > 0, stats.beta.ppf(tail, counts, replicates - counts + 1), 0.0)
    high = np.where(
        counts < replicates, stats.beta.ppf(1.0 - tail, counts + 1, replicates - counts), 1.0
    )
    if clause == "floor":
        return np.asarray(low >= margins.coverage_floor)
    band = margins.calibration_coverage
    return np.asarray((low >= band[0]) & (high <= band[1]))


def _coverage_pass(coverage: np.ndarray, replicates: int, clause: str) -> float:
    """The coverage clause's pass probability at ``replicates``, averaged over ``coverage``."""
    pmf = stats.binom.pmf(
        np.arange(replicates + 1)[None, :], replicates, np.atleast_1d(coverage)[:, None]
    )
    return float(np.mean(pmf @ _accepted(replicates, clause).astype(float)))


def _ratio_spread(ratio: Any, replicates: int) -> Any:
    """The normal approximation of an SE-ratio estimate's SD from the empirical spread alone."""
    return ratio * np.sqrt(1.0 / (2.0 * (replicates - 1)))


def _se_clause_pass(ratio: Any, replicates: int) -> Any:
    """The chance the 99% interval of the SE ratio lies inside the calibration band."""
    margins = Margins()
    z = float(stats.norm.ppf(0.5 + margins.confidence_level / 2.0))
    low, high = margins.calibration_se_ratio
    spread = _ratio_spread(ratio, replicates)
    return stats.norm.cdf((high - z * spread - ratio) / spread) - stats.norm.cdf(
        (low + z * spread - ratio) / spread
    )


def _control_pass(ratio: float, replicates: int) -> float:
    """The chance a derived control's SE-ratio interval lies wholly below the band."""
    margins = Margins()
    z = float(stats.norm.ppf(0.5 + margins.confidence_level / 2.0))
    spread = float(_ratio_spread(ratio, replicates))
    return float(stats.norm.cdf((margins.calibration_se_ratio[0] - z * spread - ratio) / spread))


def weighted_ratio_clauses(replicates: int) -> dict[str, float]:
    """The weighted 2,400 cell's SE clause and its two controls, by a normal approximation.

    The true SE ratio is normal about the probe's 0.985 with the probe's own sampling SD
    (:func:`_ratio_spread` at 8,000 draws, 0.0078), so the pool's ratio is not taken as the
    truth.  The shrunken control's ratio is :data:`~properties.SHRUNKEN_SE_FACTOR` times the
    positive's; the noise control's is the positive's over ``sqrt(1 + (bound / spread)^2)``,
    with the probe's root-n spread over the bound.
    """
    draws, ratio, _, spread = DESIGN_PROBE[WEIGHTED_GATED_LABEL]
    sd = float(_ratio_spread(ratio, draws))
    return {
        "se_ratio": float(np.sum(_GH_W * _se_clause_pass(ratio + sd * _GH_X, replicates))),
        "shrunken": _control_pass(properties.SHRUNKEN_SE_FACTOR * ratio, replicates),
        "noise": _control_pass(ratio / math.sqrt(1.0 + spread**-2), replicates),
    }


@cache
def gated_pass(sdr_replicates: int, weighted_replicates: int) -> dict[str, float]:
    """Each new gated cell's pass probability at the given replication counts.

    An SDR cell's is its coverage floor averaged over the Jeffreys posterior of its pilot; its
    bias clause and SE screen pass with probability above 0.9999 at the pilot's bias of about
    0.01 SD.  The weighted cell's is its coverage band, averaged the same way, times the SE
    clause and both controls of :func:`weighted_ratio_clauses`.  The clauses are read as
    independent.
    """
    passes = {
        label: _coverage_pass(_coverage_law(label), sdr_replicates, "floor")
        for label in SDR_GATED_LABELS
    }
    passes[WEIGHTED_GATED_LABEL] = _coverage_pass(
        _coverage_law(WEIGHTED_GATED_LABEL), weighted_replicates, "band"
    ) * math.prod(weighted_ratio_clauses(weighted_replicates).values())
    return passes


def family_pass(sdr_replicates: int, weighted_replicates: int) -> float:
    """The chance that the three new gated cells pass together."""
    return math.prod(gated_pass(sdr_replicates, weighted_replicates).values())


def test_the_coverage_pilot_is_the_design_probe() -> None:
    for label, (covered, draws) in COVERAGE_PILOT.items():
        assert draws == DESIGN_PROBE[label][0]
        assert covered / draws == pytest.approx(DESIGN_PROBE[label][2], abs=5e-5)


def test_the_new_gated_replications_follow_the_sizing_rule() -> None:
    """Each count is the smallest multiple of 400 with family-wise pass at least 0.95.

    One step of 400 less on either count, with the other held, falls short.
    """
    sdr, weighted_r = properties.SDR_GATED_REPLICATES, properties.WEIGHTED_GATED_REPLICATES
    assert sdr % REPLICATE_GRAIN == 0 and weighted_r % REPLICATE_GRAIN == 0
    assert family_pass(sdr, weighted_r) >= FAMILY_FLOOR
    assert family_pass(sdr - REPLICATE_GRAIN, weighted_r) < FAMILY_FLOOR
    assert family_pass(sdr, weighted_r - REPLICATE_GRAIN) < FAMILY_FLOOR


def test_the_declared_pass_probabilities_are_the_computed_ones() -> None:
    computed = gated_pass(properties.SDR_GATED_REPLICATES, properties.WEIGHTED_GATED_REPLICATES)
    assert computed == pytest.approx(GATED_PASS, abs=5e-4)
    assert math.prod(computed.values()) == pytest.approx(FAMILY_WISE_PASS, abs=5e-4)


def test_the_weighted_se_clause_passes_under_perfect_calibration() -> None:
    """The AGENT-BRIEF rule: at a true SE ratio of 1, the SE clause passes with chance >= 0.90."""
    replicates = properties.WEIGHTED_GATED_REPLICATES
    assert float(_se_clause_pass(1.0, replicates)) >= PERFECT_CALIBRATION_FLOOR


def test_every_new_gated_cell_passes_at_nominal_coverage() -> None:
    """At a true coverage of 0.95 each coverage clause passes with chance above 0.999."""
    nominal = np.array([0.95])
    assert _coverage_pass(nominal, properties.SDR_GATED_REPLICATES, "floor") > 0.999
    assert _coverage_pass(nominal, properties.WEIGHTED_GATED_REPLICATES, "band") > 0.999


def test_every_fit_set_replicate_reads_its_own_stream() -> None:
    """No two replicates of any two fit sets, of one family or of two, share a sample seed."""
    seeds = [
        properties._seed(family, label, r)
        for family, label, _, _, replicates, _ in properties.FIT_SETS
        for r in range(replicates)
    ]
    assert len(set(seeds)) == len(seeds)


# ------------------------------------------------------------------ the tail-size rule

#: A record of :math:`n_\kappa` of every exact-law curve the study reads: the gated cells' curves
#: (checked by the guard below) and the curves of the bias-only patterns and the controls, which
#: gate no Wald interval (a record, not a check).
TAIL_SIZES = {
    "sdr/both_correct": 2_326,
    "sdr/P_C": 2_326,
    "sdr/P_C own curve": 1_572,
    "sdr/P_A own curve": 84,
    "sdr/P_B own curve": 150,
    "sdr/P_D own curve": 730,
    "node2_both_wrong_control own curve": 190,
    "ltmle_P_D_control own curve": 623,
    "static": 346,
    "dynamic": 144,
    "survival_h2": 612,
    "competing_h2": 600,
    "policy_mix": 213,
    "msm_identity": 379,
    "msm_logit": 394,
    "weighted": 1_159,
    "null": 385,
    "band": 649,
}


@dataclass(frozen=True)
class WaldCell:
    """One verdicted cell that reads a Wald interval, with the curve its tail size is read from.

    Parameters
    ----------
    name : str
        ``family/cell``.
    curve : str or None
        The key of its tail size in :func:`_tail_sizes`; ``None`` for a cell of
        :data:`~tests.studies.longitudinal_sdr_properties.TAIL_SIZE_NOT_COMPUTED`.
    n : int
        Its declared size.
    """

    name: str
    curve: str | None
    n: int


def _tail(probs: np.ndarray, curve: np.ndarray) -> int:
    return n_kappa(kappa(probs, curve))


def _own_curve(engine: type, nuisances: tuple[str, str]) -> np.ndarray:
    """The exact-law influence curve of a fit, one value per row of ``sdr_law.frame()``."""
    fit = _exact_fit(engine, nuisances)
    return np.asarray(fit.influence_curves[sdr_law.ESTIMAND])  # type: ignore[attr-defined]


@cache
def _tail_sizes() -> dict[str, int]:
    """:math:`n_\\kappa` of each curve key: the larger of the quantity's EIF and the cell's own."""
    contrast = _tail(sdr_law.PROBS, sdr_law.eif(sdr_law.ESTIMAND))
    rows = np.ones(len(sdr_law.frame()))
    own = {
        pattern: _tail(rows, _own_curve(SDR, properties.PATTERN_NUISANCES[pattern]))
        for pattern in study.EXACT_RATIO_PATTERNS
    }
    null = law.weighted_eif(
        properties.STATIC, np.ones(len(law.SUPPORT)), base=end_properties.NULL_PROBS
    )
    selected = law.weighted_eif(
        properties.STATIC, weighted.OBS_WEIGHTS, base=weighted.SELECTED_PROBS
    )
    return {
        **{f"sdr/{pattern}": max(contrast, own[pattern]) for pattern in own},
        "static": _tail(law.PROBS, law.eif(properties.STATIC)),
        "dynamic": _tail(law.PROBS, law.eif(properties.DYNAMIC_MEAN)),
        "survival_h2": _tail(survival_law.PROBS, survival_law.eif(properties.SURVIVAL_NAME)),
        "competing_h2": _tail(competing_law.PROBS, competing_law.eif(properties.COMPETING_NAME)),
        "policy_mix": _tail(
            policy_law.PROBS, policy_law.eif_policy(policy_law.PROBS, properties.MIX)
        ),
        "msm_identity": _tail(law.PROBS, identity_msm.influence_curves()[:, 1]),
        "msm_logit": _tail(law.PROBS, msm.logit_influence_curves()[:, msm.TERMS.index("duration")]),
        "weighted": _tail(weighted.SELECTED_PROBS, selected),
        "null": _tail(end_properties.NULL_PROBS, null),
        "band": max(_tail(law.PROBS, law.eif(name)) for name in properties.BAND_NAMES),
    }


def _curve_of(family: str, cell: str) -> str:
    """The curve key of a computable Wald cell; a cell the map does not know raises."""
    if family == "sequential_double_robustness":
        return f"sdr/{cell.rsplit('__n_', 1)[0]}"
    if family in {"root_n_and_efficiency", "power"}:
        return "static"
    if family == "interval_calibration":
        return cell.rsplit("__", 1)[0]
    if family == "weighted_calibration":
        return "weighted"
    if family == "type_i_error":
        return "null"
    if family == "simultaneous_coverage":
        return "band"
    raise KeyError(f"no exact curve is declared for {family}/{cell}")


def _wald_cells() -> list[WaldCell]:
    """Every declared verdicted cell that reads a Wald interval, from ``declared_cells()``.

    A positive cell reads one, except the bias-only ``P_A``, ``P_B`` and ``P_D`` cells.  A control
    or a diagnostic is not a Wald validity claim.
    """
    out = []
    for cell in properties.declared_cells():
        if cell.role != "positive":
            continue
        family, name = cell.property, cell.cell
        pattern = name.rsplit("__n_", 1)[0]
        if family == "sequential_double_robustness" and pattern not in study.EXACT_RATIO_PATTERNS:
            continue
        key = f"{family}/{name}"
        curve = None if key in properties.TAIL_SIZE_NOT_COMPUTED else _curve_of(family, name)
        out.append(WaldCell(key, curve, cell.n))
    return out


def _least_gated_n(tail: int) -> int:
    factor, step = properties.WALD_REGIME_FACTOR, properties.WALD_GATED_STEP
    return step * math.ceil(factor * tail / step)


def tail_size_problems(
    cells: Sequence[WaldCell],
    tails: Mapping[str, int],
    below: Collection[str],
    gated: Collection[str],
    not_computed: Collection[str],
) -> list[str]:
    """What breaks checks 1 and 2 of the tail-size rule; empty when the declaration holds.

    Check 1: the three classes partition the Wald cells, and a computable cell is gated exactly
    when its n is at least ``WALD_REGIME_FACTOR`` times its tail size.  Check 2: every
    below-tail-size cell has a gated twin on the same curve at the least multiple of
    ``WALD_GATED_STEP`` at or above that size.
    """
    problems: list[str] = []
    names = {cell.name for cell in cells}
    classes = (set(below), set(gated), set(not_computed))
    if sum(len(group) for group in classes) != len(set().union(*classes)):
        problems.append("check 1: a cell sits in two classes")
    if set().union(*classes) != names:
        missing = sorted(names - set().union(*classes))
        extra = sorted(set().union(*classes) - names)
        problems.append(f"check 1: the classes miss {missing} and hold {extra}")
    for cell in cells:
        if cell.curve is None:
            continue
        need = properties.WALD_REGIME_FACTOR * tails[cell.curve]
        if cell.n >= need and cell.name not in gated:
            problems.append(f"check 1: {cell.name} has n {cell.n} >= {need} and is not gated")
        if cell.n < need and cell.name not in below:
            problems.append(f"check 1: {cell.name} has n {cell.n} < {need} and is not reporting")
    for cell in cells:
        if cell.name not in below or cell.curve is None:
            continue
        least = _least_gated_n(tails[cell.curve])
        twins = [
            other
            for other in cells
            if other.curve == cell.curve and other.name in gated and other.n == least
        ]
        if not twins:
            problems.append(f"check 2: {cell.name} has no gated twin at n = {least}")
    return problems


def _declared_problems(**changes: object) -> list[str]:
    arguments: dict[str, object] = {
        "cells": _wald_cells(),
        "tails": _tail_sizes(),
        "below": properties.BELOW_TAIL_SIZE,
        "gated": properties.WALD_GATED_CELLS,
        "not_computed": properties.TAIL_SIZE_NOT_COMPUTED,
    }
    arguments.update(changes)
    return tail_size_problems(**arguments)  # type: ignore[arg-type]


def test_each_wald_cell_gates_only_at_twice_its_tail_size() -> None:
    assert _declared_problems() == []
    assert len(properties.TAIL_SIZE_NOT_COMPUTED) == 4
    tails = _tail_sizes()
    assert {key: TAIL_SIZES[key] for key in tails} == tails


def test_the_rule_binds_on_these_laws() -> None:
    """A vacuous rule, or a weighted size that does not move, fails here."""
    tails = _tail_sizes()
    factor = properties.WALD_REGIME_FACTOR
    for pattern in study.EXACT_RATIO_PATTERNS:
        assert 2_000 < factor * tails[f"sdr/{pattern}"] == 4_652 <= properties.SDR_GATED_N
    assert 2_000 < factor * tails["weighted"] == 2_318 <= 2_400
    assert _least_gated_n(tails["sdr/P_C"]) == 4_800
    assert _least_gated_n(tails["weighted"]) == 2_400


def test_the_tail_sizes_are_the_exact_law_values() -> None:
    """A swapped curve, a wrong base law or a moved ``C_KAPPA`` fails here."""
    contrast = kappa(sdr_law.PROBS, sdr_law.eif(sdr_law.ESTIMAND))
    own = _own_curve(SDR, properties.PATTERN_NUISANCES["P_C"])
    p_c = kappa(np.ones(len(own)), own)
    selected = law.weighted_eif(
        properties.STATIC, weighted.OBS_WEIGHTS, base=weighted.SELECTED_PROBS
    )
    selection = kappa(weighted.SELECTED_PROBS, selected)
    np.testing.assert_allclose(
        [contrast, p_c, selection],
        [146.3602447110022, 99.2494917495194, 73.38883639268528],
        rtol=1e-9,
    )
    assert (n_kappa(contrast), n_kappa(p_c), n_kappa(selection)) == (2_326, 1_572, 1_159)


def test_an_own_curve_only_tail_size_moves_the_p_c_twin() -> None:
    """Without the EIF in the maximum, ``P_C`` needs 3,200, so its 4,800 twin fails check 2."""
    tails = {**_tail_sizes(), "sdr/P_C": TAIL_SIZES["sdr/P_C own curve"]}
    assert _least_gated_n(tails["sdr/P_C"]) == 3_200
    problems = _declared_problems(tails=tails)
    assert (
        "check 2: sequential_double_robustness/P_C__n_2000 has no gated twin at n = 3200"
        in problems
    )


def test_gating_the_n_2000_coverage_again_fails() -> None:
    cell = "sequential_double_robustness/both_correct__n_2000"
    problems = _declared_problems(
        below=properties.BELOW_TAIL_SIZE - {cell}, gated=properties.WALD_GATED_CELLS | {cell}
    )
    assert any(problem.startswith(f"check 1: {cell} has n 2000 <") for problem in problems)


def test_dropping_or_shrinking_the_weighted_twin_fails() -> None:
    """Selective application of the rule: the weighted 2,000 cell then has no twin."""
    twin = "weighted_calibration/selection_weighted_n2400__correctly_specified"
    message = (
        "check 2: weighted_calibration/selection_weighted__correctly_specified has no gated twin "
        "at n = 2400"
    )
    cells = _wald_cells()
    dropped = _declared_problems(
        cells=[cell for cell in cells if cell.name != twin],
        gated=properties.WALD_GATED_CELLS - {twin},
    )
    assert message in dropped
    shrunk = _declared_problems(
        cells=[replace(cell, n=2_000) if cell.name == twin else cell for cell in cells]
    )
    assert message in shrunk


def test_an_undeclared_wald_cell_fails() -> None:
    extra = WaldCell("power/static__alternative_at_800", "static", 800)
    problems = _declared_problems(cells=[*_wald_cells(), extra])
    assert any("power/static__alternative_at_800" in problem for problem in problems)


def reporting_problems(
    keys: Collection[str], below: Collection[str], summary: pd.DataFrame
) -> list[str]:
    """Check 3: each reporting cell of the ledger is below its tail size and red on coverage only."""
    problems = []
    floor = study.STUDY.margins.coverage_floor
    for key in keys:
        if key not in below:
            problems.append(f"check 3: {key} is not in BELOW_TAIL_SIZE")
        row = summary.loc[tuple(key.split("/", 1))]
        low, high = UNION_MODEL_SE_BAND
        if not (
            not bool(row["passed"])
            and float(row["coverage_ci_lower"]) < floor
            and bool(row["bias_equivalent"])
            and low <= float(row["se_ratio"]) <= high
        ):
            problems.append(f"check 3: {key} does not fail on its coverage floor alone")
    return problems


def test_each_reporting_cell_of_the_ledger_is_below_its_tail_size() -> None:
    keys = [
        key.key for key in REPORTING_ROWS if key.slug == study.STUDY.slug and key.kind == "property"
    ]
    assert keys == ["sequential_double_robustness/both_correct__n_2000"]
    summary = pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])
    assert reporting_problems(keys, properties.BELOW_TAIL_SIZE, summary) == []
    # A gated, passing cell named reporting fails both halves of the check.
    wrong = "sequential_double_robustness/both_correct__n_8000"
    assert reporting_problems([wrong], properties.BELOW_TAIL_SIZE, summary) == [
        f"check 3: {wrong} is not in BELOW_TAIL_SIZE",
        f"check 3: {wrong} does not fail on its coverage floor alone",
    ]
