"""The declared design of ``vector-mtp``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* Each truth is the quadrature of :mod:`tests.studies.vector_mtp_common`, and a simulation of
  Definition 1 of Díaz et al. (2023) agrees with it.  Each efficiency bound is the pinned
  closed-form value.
* Each calibration positive passes its three bands with a probability of at least 0.90 at its
  declared 2,400 replications if it is calibrated, and its reported efficiency ratio is the
  probe's.  The probe cannot resolve a ratio to the 0.03 the band needs, so it shows only that
  no positive is detectably off one (:data:`PROBE_CALIBRATION`).  Each control and each
  bias-rule positive is sized from its measured distance at the declared margin
  (:data:`PROBE_BIAS`).  No margin or size was chosen to save time.
* The preimage-cut control removes exactly the flag-moving preimage: the package's cut ratio
  tracks the closed-form ratio without that term.
* The comparator's plan labels and column slugs, transcribed by hand into
  ``tests/canonical/vector_mtp_runner.R``, are the Python ones.

**The design probe.**  The probe records are not checks.  They hold fits on fresh streams
(``stream_seed(STUDY, "design_probe", family, label, r)``) at n = 2,000: 150 draws of each
density-route calibration positive, 1,000 of each classifier-route one
(``"design_probe_wide"``), and 80 of each double-robustness and preimage cell.  No fit raised.
The Monte Carlo standard error of a standardized mean is ``1 / sqrt(R)``: 0.082 at 150 and
0.112 at 80.  A ratio's is ``1 / sqrt(2 R)``: 0.058 at 150 and 0.022 at 1,000.  The double-
robustness probe ran under the label ``point_vector_density`` for the cell now named
``point_vector_ordered_density``, with the same fit.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student

from tests.studies import canonical_vector_mtp as study
from tests.studies import vector_mtp_common as common
from tests.studies import vector_mtp_properties as properties
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("vector_mtp_design")

RUNNER = ROOT / "tests" / "canonical" / "vector_mtp_runner.R"

#: A record, not a check: ``(probe draws, mean error / SD, SE ratio, coverage, efficiency
#: empirical, efficiency reported)`` of each calibration positive.
PROBE_CALIBRATION: dict[str, tuple[int, float, float, float, float, float]] = {
    "point_vector_density": (150, 0.057, 1.023, 0.973, 0.979, 1.001),
    "point_vector_classifier": (1_000, -0.029, 1.043, 0.954, 0.960, 1.001),
    "longitudinal_vector_density": (150, -0.014, 1.023, 0.927, 0.969, 0.992),
    "longitudinal_vector_classifier": (1_000, 0.012, 0.977, 0.936, 1.011, 0.988),
    "longitudinal_vector_ordered_density": (150, 0.020, 0.996, 0.933, 1.001, 0.997),
    "longitudinal_vector_ordered_classifier": (1_000, 0.010, 1.011, 0.940, 0.983, 0.995),
}
#: A record, not a check: ``(mean error / SD, SE ratio, coverage)`` of each bias-rule cell over
#: 80 probe draws.
PROBE_BIAS: dict[str, tuple[float, float, float]] = {
    "point_vector_ordered_density__outcome_wrong": (-0.089, 1.048, 0.975),
    "point_vector_ordered_density__both_wrong": (2.075, 1.776, 0.900),
    "longitudinal_vector_density__outcome_wrong": (0.080, 0.982, 0.938),
    "longitudinal_vector_density__both_wrong": (3.057, 1.794, 0.550),
    "longitudinal_vector_ordered_density__outcome_wrong": (-0.072, 1.037, 0.938),
    "longitudinal_vector_ordered_density__both_wrong": (3.297, 1.056, 0.100),
    "flag_scaled__declared_preimage": (0.071, 1.014, 0.950),
    "flag_scaled__preimage_cut_control": (2.450, 0.389, 0.062),
}
#: The preimage displacement over the same 80 paired draws, in the declared arm's SD.
PROBE_PREIMAGE_DISPLACEMENT = 3.549
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    **{("interval_calibration", label): (2_000, 2_400) for label in properties.CELLS},
    **{
        ("double_robustness", f"{label}__{configuration}"): (2_000, 1_000)
        for label in properties.ROBUSTNESS_CELLS
        for configuration in ("outcome_wrong", "both_wrong")
    },
    ("preimage_necessity", "paired"): (2_000, 1_000),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20263101, 2026310101)
    assert study.SMOKE_GATE == 1e-6
    assert (common.CAP, common.STEP, common.FACTOR, common.KNEE) == (5.5, 0.5, 1.25, 3.0)
    assert record.reference == "lmtp"
    assert {
        (family, label): (n, replicates) for family, label, n, replicates in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert properties.PREIMAGE_DISPLACEMENT == 0.10
    assert set(record.scenarios) == set(study.LAWS)


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = properties.summarize_properties(rows)
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


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
        ("ordered", common.ORDERED_LABELS),
        ("long", common.LONG_LABELS),
    ):
        for index, label in enumerate(labels):
            mean, error = _simulated_mean(law, label, 400_000, 20263111 + index)
            assert abs(mean - common.TRUTHS[law](label)) < 4.0 * error, (law, label)


def test_the_categorical_truths_are_the_discrete_sum() -> None:
    rng = np.random.default_rng(20263117)
    frame = common.sample_categorical(400_000, 20263118)
    del rng
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


# ------------------------------------------------------------------ the preimage control


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


# ------------------------------------------------------------------ the comparator's table


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


# ------------------------------------------------------------------ sizes


def _band_pass(point: float, spread: float, band: tuple[float, float]) -> float:
    """The probability that a 99% normal interval around an estimate lies inside ``band``."""
    half = float(norm.ppf(0.995)) * spread
    upper = (band[1] - half - point) / spread
    lower = (band[0] + half - point) / spread
    return float(max(0.0, norm.cdf(upper) - norm.cdf(lower)))


def calibration_pass_probability(
    se_ratio: float, empirical: float, reported: float, replicates: int
) -> float:
    """A Bonferroni lower bound on the calibration rule's pass probability.

    The SE ratio and the empirical efficiency ratio each have spread ``ratio / sqrt(2 R)`` for
    a normal estimate.  The reported efficiency ratio is a mean of ``R`` standard errors and its
    spread is negligible.  Coverage at 0.95 lies inside ``[0.92, 0.98]`` with probability
    above ``1 - 1e-6`` at 2,400 replications.
    """
    margins = Margins()
    spread = 1.0 / math.sqrt(2.0 * replicates)
    fails = (1.0 - _band_pass(se_ratio, se_ratio * spread, margins.calibration_se_ratio)) + (
        1.0 - _band_pass(empirical, empirical * spread, properties.EFFICIENCY_RATIO_BAND)
    )
    if not properties.EFFICIENCY_RATIO_BAND[0] < reported < properties.EFFICIENCY_RATIO_BAND[1]:
        return 0.0
    return 1.0 - fails


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


def test_the_calibration_positives_pass_with_high_probability() -> None:
    """At a calibrated ratio and the probe's reported efficiency, each positive passes.

    The probe shows no positive detectably off one: each ratio's 99% interval holds it.
    """
    half = float(norm.ppf(0.995))
    for label, (draws, _, se_ratio, _, empirical, reported) in PROBE_CALIBRATION.items():
        probability = calibration_pass_probability(
            1.0, 1.0, reported, properties.CALIBRATION_REPLICATES
        )
        assert probability >= 0.90, (label, probability)
        spread = 1.0 / math.sqrt(2.0 * draws)
        for ratio in (se_ratio, empirical):
            assert abs(ratio - 1.0) < half * spread, label


def test_the_calibration_controls_leave_their_bands() -> None:
    """The shrunken arm's SE ratio and the noise arm's empirical efficiency ratio."""
    margins = Margins()
    spread = 1.0 / math.sqrt(2.0 * properties.CALIBRATION_REPLICATES)
    half = float(norm.ppf(0.995))
    for label, (_, _, se_ratio, _, empirical, _) in PROBE_CALIBRATION.items():
        shrunken = properties.SHRUNKEN_SE_FACTOR * se_ratio
        assert shrunken * (1 + half * spread) < margins.calibration_se_ratio[0] - 0.1, label
        noisy = math.sqrt(empirical**2 + 1.0)
        assert noisy * (1 - half * spread) > properties.EFFICIENCY_RATIO_BAND[1] + 0.2, label


def test_the_bias_rule_cells_are_sized_at_their_declared_replications() -> None:
    for cell, (mean, se_ratio, _) in PROBE_BIAS.items():
        control = cell.endswith(("both_wrong", "cut_control"))
        replicates = (
            properties.PREIMAGE_REPLICATES
            if cell.startswith(properties.PREIMAGE_LABEL)
            else properties.DOUBLE_ROBUST_REPLICATES
        )
        probability = bias_pass_probability(mean, replicates, control=control)
        assert probability >= 0.90, (cell, probability)
        assert 0.1 < se_ratio < 10.0, cell
    assert PROBE_PREIMAGE_DISPLACEMENT > properties.PREIMAGE_DISPLACEMENT
