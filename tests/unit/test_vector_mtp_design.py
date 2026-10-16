"""The declared design of ``vector-mtp``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* Each truth is the quadrature of :mod:`tests.studies.vector_mtp_common`, and a simulation of
  Definition 1 of Díaz et al. (2023) agrees with it.  Each efficiency bound is the pinned
  closed-form value.  The ordered law's induced density ratio satisfies its defining identity
  ``E[r(A, H) phi(A, H)] = E[phi(d(A, H), H)]`` for several test functions on every plan.
* Each calibration positive's replications make the product of the six positives' pass
  probabilities, at fresh-stream probe estimates, at least 0.90.  The product stays at least
  0.90 with every probed number moved one Monte Carlo SE toward failing.  Each pass probability
  is also given at a calibrated ratio of 1.  Each control and each bias-rule positive is sized from its
  measured or exact-law distance at the declared margin.  No margin or size was chosen to save
  time.
* Each necessity control removes exactly its term: the package's defective ratio tracks the
  closed form with the same defect.  The exact-law shift of every defect the study can and
  cannot see is recorded (:data:`EXACT_LAW_SHIFTS`).
* The comparator's ``lmtp`` calls on the vector path pass ``mtp = TRUE``, and its hand
  transcriptions (plan labels and slugs, the unsupplied pair's plans and learners) are the
  Python ones.

**The probes.**  The probe records are not checks.  They hold fits on fresh streams at
n = 2,000, never a declared stream, with no failed fit unless stated.

=============================================  ===============================================
record                                         source
=============================================  ===============================================
density calibration positives                  1,200 draws each (2,400 for the ordered law),
                                               ``stream_seed(STUDY,
                                               "design_remediation_probe_calibration", label,
                                               r)``
classifier calibration positives               2,400 draws each, the pre-queue design review's
                                               ``"design_review_probe_calibration"`` streams
the unsupplied pair (:data:`PROBE_PAIR`)       1,000 draws,
                                               ``"design_remediation_probe_pair"``
new double-robustness legs                     400 draws each (399 for the ordered law),
                                               ``"design_remediation_probe"``
the new necessity pairs                        100 paired draws each,
                                               ``"design_remediation_probe"``; a check of the
                                               reduction below, not the sizing record
``LTMLE`` positives with the outcome wrong     the exact reduction below, 8,000 to 20,000
                                               replications of the closed-form ratio
the ordered density positive's diagnosis       the pre-run re-check: 40,000 closed-form
(:data:`ORDERED_DENSITY_DIAGNOSIS`)            replications and 240 paired package draws,
                                               ``recheck_*`` streams
=============================================  ===============================================

**The Hajek reduction.**  With a constant initial outcome regression and one fold, each node's
intercept fluctuation solves ``sum(R (Y - Q*)) = 0``.  So an ``LTMLE`` estimate with the
outcome wrong is ``sum(R Y) / sum(R)``, with ``R`` the cumulative ratio.  The pre-queue
review confirmed the equality to every printed digit on three package fits, and the binned
oracle ratio moves the Hajek value by at most 0.03 SD.  The reduction sizes those cells far
more precisely than package fits can.  The point ``TMLE`` positive does not reduce this way and
reads package fits.

The Monte Carlo standard error of a standardized mean is ``1 / sqrt(R)``, and a ratio's is
``ratio / sqrt(2 R)`` for a normal estimate.  Each record gives the kurtosis, so the normal
approximation can be checked.  Every probed cell reads 2.5 to 3.3, except the declared arm of
``jacobian_necessity`` (4.6 over 100 package fits).  That arm is sized by the Hajek reduction,
and its pass probability is above 0.999 at its measured distance.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student

from cleverly.interventions import joint_density
from tests.canonical import vector_mtp_parity as parity
from tests.studies import canonical_vector_mtp as study
from tests.studies import vector_mtp_common as common
from tests.studies import vector_mtp_properties as properties
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("vector_mtp_design")

CANONICAL = ROOT / "tests" / "canonical"
RUNNER = CANONICAL / "vector_mtp_runner.R"
LOGRATIO = CANONICAL / "vector_mtp_logratio.R"
#: Every R source on the vector path.
VECTOR_R = (
    "lmtp_vector_adapter.R",
    "vector_mtp_logratio.R",
    "vector_mtp_parity.R",
    "vector_mtp_runner.R",
)

#: A record, not a check: ``(probe draws, mean error / SD, SE ratio, coverage, efficiency
#: empirical, efficiency reported, kurtosis)`` of each calibration positive.  The classifier
#: rows were fitted before the remediation, when ``OracleLogOdds`` was sklearn's unpenalized
#: ``LogisticRegression`` (lbfgs, ``tol=1e-4``).  It is now ``QuasiBinomialGLM`` by IRLS: the
#: same maximum-likelihood fit, to a tighter tolerance.
PROBE_CALIBRATION: dict[str, tuple[int, float, float, float, float, float, float]] = {
    "point_vector_density": (1_200, -0.0389, 1.0283, 0.9550, 0.9718, 0.9993, 2.95),
    "point_vector_classifier": (2_400, 0.0182, 1.0058, 0.9517, 0.9941, 0.9999, 3.08),
    "longitudinal_vector_density": (1_200, -0.0688, 0.9828, 0.9483, 1.0115, 0.9941, 3.00),
    "longitudinal_vector_classifier": (2_400, 0.0098, 0.9913, 0.9429, 0.9962, 0.9876, 2.96),
    "longitudinal_vector_ordered_density": (
        2_400,
        -0.0246,
        0.9676,
        0.9404,
        1.0316,
        0.9982,
        2.97,
    ),
    "longitudinal_vector_ordered_classifier": (
        2_400,
        -0.0347,
        0.9898,
        0.9475,
        1.0033,
        0.9931,
        2.92,
    ),
}
#: A record, not a check: ``(mean error / SD, its Monte Carlo SE, SE ratio, coverage)`` of each
#: bias-rule cell, with its source.  ``nan`` marks a number its source does not give.
PROBE_BIAS: dict[str, tuple[float, float, float, float]] = {
    # 400 package fits (the pre-queue design review).
    "point_vector_ordered_density__outcome_wrong": (-0.023, 0.050, 1.047, 0.973),
    # The Hajek reduction: 20,000 and 8,000 replications (the pre-queue design review).
    "longitudinal_vector_density__outcome_wrong": (-0.013, 0.007, math.nan, math.nan),
    "longitudinal_vector_ordered_density__outcome_wrong": (-0.008, 0.011, math.nan, math.nan),
    # The same fit as longitudinal_vector_density__outcome_wrong on its own stream.
    "flag_scaled__declared_preimage": (-0.013, 0.007, math.nan, math.nan),
    # 80 package fits (the scaffold's design probe).
    "point_vector_ordered_density__both_wrong": (2.075, 0.112, 1.776, 0.900),
    "longitudinal_vector_density__both_wrong": (3.057, 0.112, 1.794, 0.550),
    "longitudinal_vector_ordered_density__both_wrong": (3.297, 0.112, 1.056, 0.100),
    "flag_scaled__preimage_cut_control": (2.450, 0.112, 0.389, 0.062),
    # The Hajek reduction, 8,000 replications each.
    "ordered_full__declared_jacobian": (-0.0142, 0.0112, math.nan, math.nan),
    "ordered_full__jacobian_drop_control": (0.4689, 0.0112, math.nan, math.nan),
    "ordered_crossed__declared_image": (-0.0085, 0.0112, math.nan, math.nan),
    "ordered_crossed__shifted_read_control": (1.0200, 0.0112, math.nan, math.nan),
    # The both-correct LTMLE legs are the density calibration positives' fit.
    "longitudinal_vector_density__both_correct": (-0.0688, 0.0289, 0.9828, 0.9483),
    "longitudinal_vector_ordered_density__both_correct": (-0.0246, 0.0204, 0.9676, 0.9404),
    # 400 package fits each (399 for the ordered law).
    "point_vector_ordered_density__both_correct": (0.0474, 0.0500, 1.051, 0.958),
    "point_vector_ordered_density__outcome_correct": (-0.0823, 0.0500, 1.755, 0.995),
    "longitudinal_vector_density__outcome_correct": (0.0252, 0.0500, 1.332, 0.983),
    "longitudinal_vector_ordered_density__outcome_correct": (-0.0530, 0.0501, 1.438, 1.000),
}
#: Each necessity pair's displacement, in the declared arm's SD: the preimage pair over 80
#: package draws (the scaffold's probe), the others by the Hajek reduction over 8,000.  Over 100
#: paired package draws each, the displacement reads 0.538 (Jacobian) and 1.359 (image).
PROBE_DISPLACEMENT = {
    "preimage_necessity": 3.549,
    "jacobian_necessity": 0.500,
    "image_necessity": 1.232,
}
#: A record, not a check: the unsupplied pair's package side over 1,000 fresh draws:
#: ``(mean error / SD, SE ratio, coverage, kurtosis)`` of each estimand.  The largest fitted
#: ratio over the probe was 49.4, a classifier probability of 0.980.
PROBE_PAIR: dict[str, tuple[float, float, float, float]] = {
    "ey_regimen[natural course]": (-0.0232, 1.0033, 0.9520, 3.09),
    "ey_regimen[ordered]": (-0.0044, 0.9800, 0.9370, 3.05),
    "ey_regimen[full]": (-0.0115, 1.0010, 0.9550, 3.04),
    "ate_regimen[ordered vs natural course]": (0.0166, 0.9991, 0.9500, 2.73),
    "ate_regimen[full vs natural course]": (0.0067, 0.9903, 0.9570, 2.81),
}
PROBE_PAIR_LARGEST_RATIO = 49.4
#: A record, not a check: the pre-run re-check's diagnosis of the ordered density positive,
#: whose remediation probe read an SE ratio of 0.968 (:data:`PROBE_CALIBRATION`).  Both probes
#: use fresh ``recheck_*`` streams, at n = 2,000 under ``full``.
#:
#: * ``closed_form``: an independent re-implementation of the one-node ``LTMLE`` with the
#:   closed-form ordered ratio, 40,000 replications.  ``(mean error / SD, SE ratio, its Monte
#:   Carlo SE, coverage, efficiency empirical, efficiency reported)``.  With the package's own
#:   ``node_ratio`` the same code equals the package to every printed digit, and its influence
#:   curve to 1.8e-15, so the package's curve has no missing term.
#: * ``paired``: 240 draws, each fitted by the package's density route and by the closed-form
#:   ratio.  ``(draws, correlation, SD of the difference in bound SD, the package's variance
#:   excess in bound SD squared, its Monte Carlo SE, the package's mean SE over the closed
#:   form's)``.
ORDERED_DENSITY_DIAGNOSIS = {
    "closed_form": (0.006, 0.9927, 0.0035, 0.9474, 0.9989, 0.9916),
    "paired": (240, 0.9982, 0.059, 0.012, 0.008, 1.0048),
}
#: The exact-law shift of each ratio defect in the outcome-wrong ``LTMLE`` estimate, by the
#: Hajek reduction over 2,000,000 draws (the defective limit minus the declared one, on the
#: same draws), in the contrast's SD at n = 2,000.  The two ordered-law
#: defects of :data:`~tests.studies.vector_mtp_common.ORDERED_DEFECTS` are recomputed below.
#: The rest are the pre-queue design review's, kept as the record of what this study cannot see:
#: the ``D2`` density conditioned on the shifted ``D1`` moves ``full`` by 0.42 SD and the ``D2``
#: map reading the shifted ``D1`` by -0.19 SD, and the fast tier witnesses both.
EXACT_LAW_SHIFTS: dict[tuple[str, str], float] = {
    ("full", "jacobian"): 0.499,
    ("full", "image"): 0.000,
    ("crossed", "jacobian"): -0.687,
    ("crossed", "image"): 1.243,
    ("full", "preimage_cut"): -1.06,
    ("full", "d2_density_reads_shifted"): 0.42,
    ("full", "d2_map_reads_shifted"): -0.19,
    ("flag scaled", "preimage_cut"): 3.60,
    ("flag scaled", "jacobian"): 0.31,
}
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    **{
        ("interval_calibration", label): (2_000, properties.CALIBRATION_REPLICATES[label])
        for label in properties.CELLS
    },
    **{
        ("double_robustness", f"{label}__{configuration}"): (2_000, 2_000)
        for label in properties.ROBUSTNESS_CELLS
        for configuration in properties.ROBUSTNESS_CONFIGURATIONS
    },
    **{(family, "paired"): (2_000, 1_000) for family in properties.NECESSITY},
}
#: The declared replications of each calibration positive.
DECLARED_CALIBRATION = {
    "point_vector_density": 24_000,
    "point_vector_classifier": 10_000,
    "longitudinal_vector_density": 9_600,
    "longitudinal_vector_classifier": 10_000,
    "longitudinal_vector_ordered_density": 19_200,
    "longitudinal_vector_ordered_classifier": 10_000,
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20263101, 2026310101)
    assert study.SMOKE_GATE == 1e-6
    assert parity.PARITY_TOLERANCE == 1e-4
    assert study.REFERENCE == "natural course"
    assert study.CLASSIFIER_PLANS == ("ordered", "full")
    assert (common.CAP, common.STEP, common.FACTOR, common.KNEE) == (5.5, 0.5, 1.25, 3.0)
    # The owner's rule: the method merges on its Python studies, and the lmtp runs follow.
    assert record.comparator_state == "pending"
    assert (record.reference, record.pending_reference) == (None, "lmtp")
    assert "Study queue" in record.pending_reason
    assert properties.CALIBRATION_REPLICATES == DECLARED_CALIBRATION
    assert {
        (family, label): (n, replicates) for family, label, n, replicates in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert properties.PREIMAGE_DISPLACEMENT == properties.NECESSITY_DISPLACEMENT == 0.10
    assert set(record.scenarios) == set(study.LAWS)
    assert set(record.property_cells) == {
        "interval_calibration",
        "double_robustness",
        *properties.NECESSITY,
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


# ------------------------------------------------------------------ truths and bounds


def test_every_contrast_is_away_from_zero() -> None:
    """Each truth self-checks at twice its quadrature order when computed."""
    for scenario in study.LAWS:
        for name, value in study.truth_of(scenario).items():
            if name.startswith("ate_"):
                assert abs(value) > 0.01, (scenario, name)
    for law, label in (("ordered", "crossed"),):
        assert abs(properties.truth(law, label)) > 0.01


def _simulated_mean(law: str, label: str, n: int, seed: int) -> tuple[float, float]:
    """The mean outcome regression of Definition 1, by simulating the intervened law."""
    rng = np.random.default_rng(seed)
    if law == "point":
        w1, w2 = rng.uniform(-1.0, 1.0, n), rng.integers(0, 2, n).astype(float)
        f = rng.binomial(1, common.point_flag(w1, w2)).astype(float)
        d = common.truncated_draw(rng, common.point_dose_mean(f, w1, w2))
        q = common.point_outcome(*common.flag_dose_map(label, f, d), w1, w2)
    elif law == "ordered":
        w1, w2 = rng.uniform(-1.0, 1.0, n), rng.integers(0, 2, n).astype(float)
        f = rng.binomial(1, common.point_flag(w1, w2)).astype(float)
        d1 = common.truncated_draw(rng, common.point_dose_mean(f, w1, w2))
        d2 = common.truncated_draw(rng, common.ordered_d2_mean(d1, f, w1))
        q = common.ordered_outcome(*common.ordered_map(label, f, d1, d2), w1, w2)
    else:
        w = rng.integers(0, 2, n).astype(float)
        f1 = rng.binomial(1, common.long_flag1(w)).astype(float)
        f1, d1 = common.flag_dose_map(
            label, f1, common.truncated_draw(rng, common.long_dose1_mean(f1, w))
        )
        l2 = rng.binomial(1, common.long_l2(w, f1, d1)).astype(float)
        f2 = rng.binomial(1, common.long_flag2(w, f1, d1, l2)).astype(float)
        d2 = common.truncated_draw(rng, common.long_dose2_mean(w, f1, d1, l2, f2))
        q = common.long_outcome(w, f1, d1, l2, *common.flag_dose_map(label, f2, d2))
    return float(np.mean(q)), float(np.std(q)) / math.sqrt(n)


def test_the_quadrature_truths_match_a_simulation_of_definition_one() -> None:
    """Each node's natural value is drawn given the intervened past, then the plan moves it."""
    for law, labels in (
        ("point", common.POINT_LABELS),
        ("ordered", (*common.ORDERED_LABELS, "crossed")),
        ("long", common.LONG_LABELS),
    ):
        for index, label in enumerate(labels):
            mean, error = _simulated_mean(law, label, 400_000, 20263111 + index)
            assert abs(mean - common.TRUTHS[law](label)) < 4.0 * error, (law, label)


def test_the_categorical_truths_are_the_discrete_sum() -> None:
    frame = common.sample_categorical(400_000, 20263118)
    for label in common.CATEGORICAL_LABELS:
        a1, a2 = common.categorical_map(label, frame["A1"], frame["A2"])
        q = common.cat_outcome(a1, a2, frame["W1"], frame["W2"])
        error = float(np.std(q)) / math.sqrt(len(q))
        assert abs(float(np.mean(q)) - common.categorical_truth(label)) < 4.0 * error, label


def test_the_pinned_efficiency_bounds_are_the_closed_form() -> None:
    for law, label in (("point", "flag scaled"), ("long", "flag scaled"), ("ordered", "full")):
        assert common.contrast_sd(law, label) == pytest.approx(properties.BOUNDS[law], rel=1e-12)


def test_the_efficient_influence_functions_are_centred() -> None:
    for law, label in (("point", "flag scaled"), ("long", "flag scaled"), ("ordered", "full")):
        frame = common.SAMPLERS[law](20_000, 20263119)
        curve = common.EIFS[law](label, frame)
        assert abs(float(np.mean(curve))) < 4.0 * float(np.std(curve)) / math.sqrt(20_000), law


def test_the_ordered_ratio_is_the_induced_density_ratio() -> None:
    """``E[r(A, H) phi(A, H)] = E[phi(d(A, H), H)]`` for test functions around every break.

    This is the defining identity of the induced density ratio (Díaz et al. 2023, Equation (3)),
    checked by simulation and independent of the formula's case analysis.  Each test function
    is an indicator of a box that straddles the knee, the knee's image 3.75, the scale's hold
    4.4, or the shift's hold 5.0, in each flag cell, or a smooth moment.  Each difference must
    lie within 4.5 Monte Carlo SEs.  The R transcription is checked against this function on a
    grid by the parity check.
    """
    frame = common.sample_ordered(400_000, 20263121)
    w1, w2, f, d1, d2 = (frame[c].to_numpy(dtype=float) for c in ("W1", "W2", "F", "D1", "D2"))
    tests = {
        "flag": lambda a, b1, b2: a,
        "moment": lambda a, b1, b2: b1 * b2 * (1.0 + a),
        "knee 0": lambda a, b1, b2: (a == 0) & (b1 > 2.6) & (b1 < 3.9),
        "knee 1": lambda a, b1, b2: (a == 1) & (b1 > 2.6) & (b1 < 3.9),
        "hold 1": lambda a, b1, b2: (a == 1) & (b1 > 4.2) & (b2 > 2.0),
        "shift": lambda a, b1, b2: (b1 < 3.2) & (b2 > 4.6),
    }
    for label in (*common.ORDERED_LABELS[1:], "crossed"):
        ratio = common.ordered_ratio(label, f, d1, d2, w1, w2)
        moved = common.ordered_map(label, f, d1, d2)
        for name, phi in tests.items():
            difference = ratio * phi(f, d1, d2) - np.asarray(phi(*moved), dtype=float)
            z = float(np.mean(difference)) / (float(np.std(difference)) / math.sqrt(len(frame)))
            assert abs(z) < 4.5, (label, name, z)


def test_the_rich_first_regression_is_close_to_the_exact_one() -> None:
    """The two-node calibration cells read :class:`RichPseudo` at the first node.

    Fitted to the exact first-node regression on 8,000 rows, its largest error is far below
    the standard error of a contrast at n = 2,000.
    """
    frame = common.sample_long(8_000, 20263120)
    w, f1, d1 = (frame[c].to_numpy() for c in ("W", "F1", "D1"))
    for label in common.LONG_LABELS:
        f1_d, d1_d = common.flag_dose_map(label, f1, d1)
        exact = common.long_first_regression(label, w, f1_d, d1_d)
        design = np.column_stack([w, f1_d, d1_d])
        fitted = common.RichPseudo().fit(design, exact).predict(design)
        assert float(np.sqrt(np.mean((fitted - exact) ** 2))) < 0.002, label


# ------------------------------------------------------------------ the necessity controls


def test_the_preimage_cut_removes_exactly_the_flag_moving_preimage() -> None:
    """The cut fit's node ratios track the closed form without the ``c = 0`` term.

    The package reads binned densities at 202 oracle bins here, so the two agree to the
    binning: a correlation above 0.99 and a median gap below ``1e-3``.  The declared fit tracks
    the full closed form the same way, and the cut removes about half the ratio's mass.
    """
    frame = common.sample_long(1_000, 4)
    means = {}
    for cut in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = properties.preimage_fit(frame, cut=cut).fits["flag scaled"].node_ratio
        for node, exact in enumerate(common.long_ratios("flag scaled", frame, cut=cut)):
            assert np.corrcoef(fitted[:, node], exact)[0, 1] > 0.99, (cut, node)
            assert float(np.median(np.abs(fitted[:, node] - exact))) < 1e-3, (cut, node)
        means[cut] = float(np.mean(fitted[:, 0]))
    assert means[True] < 0.6 * means[False]


@pytest.mark.parametrize("family", ["jacobian_necessity", "image_necessity"])
def test_each_ordered_defect_is_the_closed_form_defect(family: str) -> None:
    """The package's defective ratio tracks the closed form with the same defect.

    On 1,000 rows the oracle reads 202 bins per dose, so the two agree to the binning of a
    two-dose density: a correlation above 0.99 and mean ratios within 0.01.  The defect moves
    the mean ratio by more than 0.03 on both sides, so the control is not inert.
    """
    law, plan, _, _, defect = properties.NECESSITY[family]
    frame = common.SAMPLERS[law](1_000, 4)
    columns = (frame[c].to_numpy(dtype=float) for c in ("F", "D1", "D2", "W1", "W2"))
    f, d1, d2, w1, w2 = columns
    means = {}
    for control in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = properties.necessity_fit(family, frame, control=control).fits[plan]
        exact = common.ordered_ratio(plan, f, d1, d2, w1, w2, defect=defect if control else None)
        ratio = fitted.node_ratio[:, 0]
        assert np.corrcoef(ratio, exact)[0, 1] > 0.99, control
        assert abs(float(np.mean(ratio)) - float(np.mean(exact))) < 0.01, control
        means[control] = (float(np.mean(ratio)), float(np.mean(exact)))
    assert abs(means[True][0] - means[False][0]) > 0.03
    assert abs(means[True][1] - means[False][1]) > 0.03


def test_a_defect_refuses_to_stack_on_another_patch() -> None:
    """The defects patch the module, so a second patch while one is active is refused."""
    with properties.density_defect("jacobian"):
        assert joint_density._enumerate is not properties._ORIGINAL["_enumerate"]
        with (
            pytest.raises(RuntimeError, match="already patched"),
            properties.density_defect("image"),
        ):
            pass
    assert all(
        getattr(joint_density, name) is function for name, function in properties._ORIGINAL.items()
    )


def _hajek_shift(plan: str, defect: str, frame: pd.DataFrame) -> float:
    """The defect's shift of the outcome-wrong ``LTMLE`` limit, in the contrast's SD at 2,000."""
    columns = [frame[c].to_numpy(dtype=float) for c in ("F", "D1", "D2", "W1", "W2")]
    y = frame["Y"].to_numpy(dtype=float)
    declared = common.ordered_ratio(plan, *columns)
    defective = common.ordered_ratio(plan, *columns, defect=defect)
    reference = common.ordered_truth(study.REFERENCE)
    curve = declared * (y - common.ordered_truth(plan)) / np.mean(declared) - (y - reference)
    sd = float(np.std(curve)) / math.sqrt(2_000)
    shift = np.sum(defective * y) / np.sum(defective) - np.sum(declared * y) / np.sum(declared)
    return float(shift) / sd


def test_the_exact_law_shifts_of_the_ordered_defects() -> None:
    """Each recorded shift, recomputed on 400,000 paired draws, to 0.1 SD.

    ``full`` cannot see the flag map's read of ``D1``: ``D1`` moves only where the natural
    flag is 1, which the flag map sends to 1 whichever ``D1`` it reads.  ``crossed`` can.
    """
    frame = common.sample_ordered(400_000, 7)
    for plan in ("full", "crossed"):
        for defect in common.ORDERED_DEFECTS:
            shift = _hajek_shift(plan, defect, frame)
            assert shift == pytest.approx(EXACT_LAW_SHIFTS[plan, defect], abs=0.1), (plan, defect)
    assert abs(EXACT_LAW_SHIFTS["full", "image"]) < 0.01
    assert EXACT_LAW_SHIFTS["crossed", "image"] > 1.0
    assert EXACT_LAW_SHIFTS["full", "jacobian"] > 0.45


# ------------------------------------------------------------------ the comparator


def _code(path: Path) -> str:
    """An R source with its comments removed."""
    text = path.read_text(encoding="utf-8")
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _call_arguments(code: str, name: str) -> list[list[str]]:
    """The top-level arguments of every call of ``name``, direct or via ``lmtp_internal``."""
    calls = []
    for match in re.finditer(rf"\b{name}\b(\"\))?\s*\(", code):
        depth, start, arguments = 0, match.end(), []
        for index in range(start, len(code)):
            character = code[index]
            if character in "([{":
                depth += 1
            elif character in ")]}":
                if depth == 0:
                    arguments.append(code[start:index].strip())
                    break
                depth -= 1
            elif character == "," and depth == 0:
                arguments.append(code[start:index].strip())
                start = index + 1
        calls.append(arguments)
    return calls


def test_the_argument_reader_sees_the_mtp_flag() -> None:
    """The reader below is not blind: it reads ``FALSE`` where the shared adapter writes it."""
    shared = _call_arguments(_code(CANONICAL / "lmtp_crossfit_adapter.R"), "cf_density_ratios")
    assert shared and all(arguments[2] == "FALSE" for arguments in shared)
    call = 'lmtp_internal("cf_density_ratios")(task, x, TRUE, c, p)'
    assert _call_arguments(call, "cf_density_ratios") == [["task", "x", "TRUE", "c", "p"]]


def test_every_vector_path_ratio_fit_passes_mtp_true() -> None:
    """``cf_density_ratios``'s third argument is ``mtp``, and the vector path passes ``TRUE``.

    With ``mtp = FALSE`` ``lmtp`` 1.5.4 multiplies each ratio by the indicator that no
    component moved (``R/density_ratios.R:58-61``).  The shared cross-fit adapter passes
    ``FALSE`` for its static and dynamic regimes, so no vector-path source may reach it.
    """
    calls = [
        (name, arguments)
        for name in VECTOR_R
        for arguments in _call_arguments(_code(CANONICAL / name), "cf_density_ratios")
    ]
    assert calls, "the vector path fits no ratio"
    assert all(arguments[2] == "TRUE" for _, arguments in calls), calls
    for name in VECTOR_R:
        assert "lmtp_tmle_with_folds" not in _code(CANONICAL / name), name


def test_the_runner_names_the_python_plans_and_slugs() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r"^plans <- list\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert match is not None
    block = match.group(1)
    for scenario, labels in study.LABELS.items():
        entry = re.search(rf"{scenario} = c\((.*?)\)", block, re.DOTALL)
        assert entry is not None, scenario
        pairs = re.findall(r'(?:"([^"]+)"|(\w[\w ]*?)) = "([^"]+)"', entry.group(1))
        transcribed = {(quoted or bare).strip(): slug for quoted, bare, slug in pairs}
        assert transcribed == {label: study.slug(label) for label in labels}, scenario
    assert f'reference <- "{study.REFERENCE}"' in text


def test_the_runner_fits_each_classifier_plan_with_its_own_learner() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r"^classifier_plans <- c\((.*?)\)$", text, re.MULTILINE | re.DOTALL)
    assert match is not None
    pairs = re.findall(r'(\w+) = "([^"]+)"', match.group(1))
    assert pairs == [(plan, f"SL.logratio.{plan}") for plan in study.CLASSIFIER_PLANS]
    learners = _code(LOGRATIO)
    for plan in study.CLASSIFIER_PLANS:
        assert re.search(rf'^SL\.logratio\.{plan} <- logratio_learner\("{plan}"\)$', learners, re.M)


def test_the_parity_grid_straddles_every_break() -> None:
    """The R transcription is checked on both sides of every point where a plan jumps."""
    grid = parity.grid()
    assert set(grid["plan"]) == set(study.CLASSIFIER_PLANS)
    for column, breaks in (("D1", (3.0, 3.75, 4.4, 5.5)), ("D2", (0.5, 5.0, 5.5))):
        values = grid[column].to_numpy()
        for value in breaks:
            assert (values < value).any() and (values == value).any() and (values > value).any()
    assert set(grid["F"]) == {0.0, 1.0} and set(grid["W2"]) == {0.0, 1.0}
    assert (grid["ratio"] == 0.0).any() and (grid["ratio"] > 1.0).any()


def test_the_classifier_screen_never_binds() -> None:
    """The closed-form ratio of each plan stays far below a probability of 0.999."""
    frame = common.sample_ordered(400_000, 11)
    columns = [frame[c].to_numpy(dtype=float) for c in ("F", "D1", "D2", "W1", "W2")]
    for plan in study.CLASSIFIER_PLANS:
        assert float(np.max(common.ordered_ratio(plan, *columns))) < 60.0, plan
    assert PROBE_PAIR_LARGEST_RATIO < study.CLASSIFIER_RATIO_SCREEN / 10.0


# ------------------------------------------------------------------ sizes


def _band_pass(point: float, spread: float, band: tuple[float, float]) -> float:
    """The probability that a 99% normal interval around an estimate lies inside ``band``."""
    half = float(norm.ppf(0.995)) * spread
    upper = (band[1] - half - point) / spread
    lower = (band[0] + half - point) / spread
    return float(max(0.0, norm.cdf(upper) - norm.cdf(lower)))


def calibration_pass_probability(
    se_ratio: float, coverage: float, empirical: float, reported: float, replicates: int
) -> float:
    """A Bonferroni lower bound on the calibration rule's pass probability.

    The SE ratio and the empirical efficiency ratio each have spread ``ratio / sqrt(2 R)`` for
    a normal estimate, and coverage ``sqrt(c (1 - c) / R)``.  The reported efficiency ratio is
    a mean of ``R`` standard errors and its spread is negligible.
    """
    margins = Margins()
    spread = 1.0 / math.sqrt(2.0 * replicates)
    binomial = math.sqrt(coverage * (1.0 - coverage) / replicates)
    fails = (
        (1.0 - _band_pass(se_ratio, se_ratio * spread, margins.calibration_se_ratio))
        + (1.0 - _band_pass(coverage, binomial, margins.calibration_coverage))
        + (1.0 - _band_pass(empirical, empirical * spread, properties.EFFICIENCY_RATIO_BAND))
    )
    if not properties.EFFICIENCY_RATIO_BAND[0] < reported < properties.EFFICIENCY_RATIO_BAND[1]:
        return 0.0
    return max(0.0, 1.0 - fails)


def control_pass_probabilities(se_ratio: float, empirical: float, replicates: int) -> float:
    """The probability that both derived controls of a positive pass.

    The shrunken arm's SE ratio must leave the band below.  The noise arm's efficiency ratio
    must leave the efficiency band above; its noise has the efficient SD.
    """
    margins = Margins()
    half = float(norm.ppf(0.995))
    spread = 1.0 / math.sqrt(2.0 * replicates)
    shrunken = properties.SHRUNKEN_SE_FACTOR * se_ratio
    noisy = math.sqrt(empirical**2 + 1.0)
    ceiling = margins.calibration_se_ratio[0] / (1.0 + half * spread)
    floor = properties.EFFICIENCY_RATIO_BAND[1] / (1.0 - half * spread)
    below = norm.cdf((ceiling - shrunken) / (shrunken * spread))
    above = norm.cdf((noisy - floor) / (noisy * spread))
    return float(below * above)


def bias_pass_probability(mean: float, replicates: int, *, control: bool) -> float:
    """The bias rule's pass probability at a standardized mean error ``mean``.

    A positive needs the 99% Student interval of the mean inside the margin, and a control
    needs it wholly outside.
    """
    half = float(student.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    spread = 1.0 / math.sqrt(replicates)
    margin = Margins().standardized_bias
    if control:
        distance = abs(mean)
        return float(norm.cdf((distance - margin - half) / spread))
    limit = margin - half
    return float(norm.cdf((limit - abs(mean)) / spread) - norm.cdf((-limit - abs(mean)) / spread))


def test_the_se_ratio_rule_reproduces_the_brief_anchors() -> None:
    """The SE-ratio leg alone, at a ratio of exactly one: about 0.16, 0.80 and 0.95."""
    band = Margins().calibration_se_ratio
    for replicates, anchor in ((800, 0.16), (1_600, 0.80), (2_400, 0.95)):
        spread = 1.0 / math.sqrt(2.0 * replicates)
        assert _band_pass(1.0, spread, band) == pytest.approx(anchor, abs=0.05)


def _toward_failing(record: tuple[int, float, float, float, float, float, float]) -> tuple:
    """A probe record with its SE ratio, coverage and efficiency ratio one Monte Carlo SE
    toward the nearer edge of each band."""
    draws, _, se_ratio, coverage, empirical, reported, _ = record
    spread = 1.0 / math.sqrt(2.0 * draws)
    binomial = math.sqrt(coverage * (1.0 - coverage) / draws)
    return (
        se_ratio * (1.0 - spread if se_ratio < 1.0 else 1.0 + spread),
        coverage - binomial if coverage < 0.95 else coverage + binomial,
        empirical * (1.0 + spread if empirical > 1.0 else 1.0 - spread),
        reported,
    )


def test_the_calibration_family_survives_its_probe_error() -> None:
    """With every probed number one Monte Carlo SE toward failing, the six positives and their
    controls still pass together with probability at least 0.90."""
    joint = 1.0
    for label, record in PROBE_CALIBRATION.items():
        se_ratio, coverage, empirical, reported = _toward_failing(record)
        replicates = properties.CALIBRATION_REPLICATES[label]
        joint *= calibration_pass_probability(
            se_ratio, coverage, empirical, reported, replicates
        ) * control_pass_probabilities(se_ratio, empirical, replicates)
    assert joint >= 0.90, joint


def calibration_family() -> dict[str, tuple[float, float, float]]:
    """Each positive's pass probability at the probe and at 1, and its controls' together."""
    out = {}
    for label, record in PROBE_CALIBRATION.items():
        _, _, se_ratio, coverage, empirical, reported, _ = record
        replicates = properties.CALIBRATION_REPLICATES[label]
        out[label] = (
            calibration_pass_probability(se_ratio, coverage, empirical, reported, replicates),
            calibration_pass_probability(1.0, 0.95, 1.0, reported, replicates),
            control_pass_probabilities(se_ratio, empirical, replicates),
        )
    return out


def test_the_calibration_family_passes_together_with_high_probability() -> None:
    """At the probes' SE ratios, coverages and efficiency ratios, all six positives and their
    controls pass together with probability at least 0.90."""
    family = calibration_family()
    joint = math.prod(probe * controls for probe, _, controls in family.values())
    assert joint >= 0.90, (joint, family)
    for label, (_, calibrated, _) in family.items():
        assert calibrated >= 0.95, label


def _bias_replicates(cell: str) -> int:
    return (
        properties.NECESSITY_REPLICATES
        if cell.split("__", 1)[0] in {stem for _, _, stem, _, _ in properties.NECESSITY.values()}
        else properties.DOUBLE_ROBUST_REPLICATES
    )


def bias_family() -> dict[str, float]:
    """Each bias-rule cell's pass probability one Monte Carlo SE toward failing."""
    out = {}
    for cell, (mean, error, _, _) in PROBE_BIAS.items():
        control = cell.endswith(("both_wrong", "_control"))
        toward = abs(mean) - error if control else abs(mean) + error
        out[cell] = bias_pass_probability(toward, _bias_replicates(cell), control=control)
    return out


def test_every_bias_rule_cell_is_recorded() -> None:
    declared = {
        cell
        for family, cells in study.STUDY.property_cells.items()
        if family != "interval_calibration"
        for cell in cells
    }
    assert set(PROBE_BIAS) == declared


def test_the_bias_rule_cells_pass_with_high_probability() -> None:
    """Each cell at least 0.90, and each family together at least 0.90."""
    family = bias_family()
    for cell, probability in family.items():
        assert probability >= 0.90, (cell, probability)
    for name, cells in study.STUDY.property_cells.items():
        if name == "interval_calibration":
            continue
        assert math.prod(family[cell] for cell in cells) >= 0.90, name
    for cell, (_, _, se_ratio, _) in PROBE_BIAS.items():
        assert math.isnan(se_ratio) or 0.1 < se_ratio < 10.0, cell
    for family_name, displacement in PROBE_DISPLACEMENT.items():
        column, threshold = properties.NECESSITY_COLUMNS[family_name]
        del column
        assert displacement > threshold, family_name


def test_the_ordered_density_positive_is_expected_to_pass() -> None:
    """The remediation probe's 0.968 is Monte Carlo noise around about 0.991, not a defect.

    The closed-form estimator's SE ratio is the expected one less the in-sample deflation.
    The binned oracle ratio scales the mean SE by the paired ratio and the empirical variance by
    one plus the paired excess.  The declared cell passes with high probability there, and the
    probe lies within two of its own Monte Carlo SEs of it.
    """
    _, se_ratio, error, coverage, empirical, reported = ORDERED_DENSITY_DIAGNOSIS["closed_form"]
    _, _, _, excess, excess_error, se_scale = ORDERED_DENSITY_DIAGNOSIS["paired"]
    low, high = Margins().calibration_se_ratio
    assert low + 3 * error < se_ratio < high - 3 * error
    variance = 1.0 / empirical
    expected = se_ratio * se_scale / math.sqrt(1.0 + excess / variance)
    assert expected == pytest.approx(0.991, abs=0.002)
    assert excess < 2.0 * excess_error
    label = "longitudinal_vector_ordered_density"
    replicates = properties.CALIBRATION_REPLICATES[label]
    assert calibration_pass_probability(expected, coverage, empirical, reported, replicates) > 0.99
    draws, _, probe, *_ = PROBE_CALIBRATION[label]
    assert abs(probe - expected) < 2.0 * probe / math.sqrt(2.0 * draws)


def test_the_unsupplied_pair_is_calibrated_on_the_package_side() -> None:
    """A record of the redesign: the correctly specified classifier's interval is valid.

    The main-terms classifier it replaced read SE ratios of 0.68 and 0.74 on 60 draws.
    """
    spread = 1.0 / math.sqrt(2.0 * 1_000)
    for name, (bias, se_ratio, coverage, _) in PROBE_PAIR.items():
        assert abs(bias) < 0.1, name
        assert abs(se_ratio - 1.0) < float(norm.ppf(0.995)) * spread, name
        assert 0.92 < coverage < 0.98, name
