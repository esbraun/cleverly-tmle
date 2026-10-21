"""The declared design of ``longitudinal-sdr`` (T11 of ``X4-plan.md``).

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The two ``sequential_double_robustness`` controls clear the floor of 3 on the exact law of
  ``tests/discrete_law_longitudinal_sdr.py``, with the harness's denominator (6.8).  Every
  positive pattern is exact there.
* The two-size rule of ``P_A``, ``P_B`` and ``P_D`` passes with probability 0.9934 per cell
  when the bias is zero at both sizes (:data:`TWO_SIZE_PASS`).
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
module counts failed fits per fit set and records no estimate.  On streams 0 to 1,999 of every
fit set it found none (:data:`FAILURE_PROBE`).  A failure in the run itself stops the run.  That
cell drops to its red-cell owner with its failure count published, and the run repeats without
it, with no other change.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE, SDR
from tests import discrete_law_longitudinal as law
from tests import discrete_law_longitudinal_sdr as sdr_law
from tests.canonical.longitudinal_sdr import gates
from tests.longitudinal_sdr import COLUMNS
from tests.studies import canonical_longitudinal_sdr as study
from tests.studies import longitudinal_sdr_properties as properties
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_ltmle import declared_regimens
from tests.studies.default_band_properties import MINIMUM_CONTROL_POWER, control_power
from tests.studies.evidence.constructions import SDR_TRAINING_FOLD_RECURSION
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("longitudinal_sdr_design")

RUNNER = ROOT / "tests" / "canonical" / "longitudinal_sdr_runner.R"
#: A record, not a check: the pre-declaration failure-only probe on streams 0 to 1,999 of every
#: fit set found zero failures.
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
#: ``p0`` of the band cell, from 400 band fits on the design-probe streams.
BAND_P0 = 0.685
#: The exact-law power of the power cell (:func:`design_power`).
DESIGN_POWER = 0.9698
#: The pass probability of the two-size rule per cell when the bias is zero at both sizes.
TWO_SIZE_PASS = 0.9934
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
    } == properties.NOISE_SD
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
