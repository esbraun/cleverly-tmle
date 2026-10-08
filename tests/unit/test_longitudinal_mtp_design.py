"""The declared design of ``longitudinal-mtp``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The band cell's pointwise control needs a power of at least 0.99 at ``p0 + 0.005`` (the RM36
  rule).  Ten fits of the continuous primary subject give ``p0 = 0.8222``, where 2,000
  replications give power 1.0.
* The controls whose verdict is a bias outside the margin clear it, so a green control is
  evidence rather than luck.  The untargeted plug-in of the intercept-only regressions is the
  same number under every plan, so its contrast is exactly zero.  The other two are means over
  400 draws of 2,000 rows (:data:`CONTROL_LIMITS`), against the margin 0.25 plus five Monte
  Carlo standard errors of the cell's mean.  The inverse-dropped control is also checked on the
  exact law of ``tests/discrete_law_longitudinal_mtp.py``.
* The continuous nodes read :func:`~tests.studies.oracle_density_bins.oracle_bins` bins: 320 at
  n = 2,000, growing as ``n^(2/3)``.  The positive cells at n = 2,000
  (:data:`POSITIVE_AT_DECLARED_N`) carry a finite-sample bias of about 0.1 empirical standard
  deviations, the both-correct arm included.  ``history__declared_inverse`` reads 0.148, so it
  passes its bias rule with a probability of about 0.74, recorded here before the run.
* The power cell is sized on the exact law (:data:`DESIGN_POWER`): the declared contrast,
  the efficiency bound and ``NULL_N`` give a probability of at least 0.99 that its exact lower
  rejection endpoint clears ``MINIMUM_POWER`` over ``NULL_REPLICATES`` replications.
* The comparator's plan labels, column slugs and copy counts, transcribed by hand into
  ``tests/canonical/longitudinal_mtp_runner.R``, are the Python ones.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` of the properties
module counts failed fits per fit set and records no estimate; on streams 0 to 19 of every fit
set it found none (:data:`FAILURE_PROBE`).  The shared harness refuses a cell that lost a
replication, so a failure in the run itself stops the run.  That cell drops to its red-cell owner
with its failure count published, and the run repeats without it, with no other change.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from tests import discrete_law_longitudinal_mtp as exact_law
from tests import longitudinal_mtp as exact_plans
from tests.studies import canonical_longitudinal_mtp as study
from tests.studies import longitudinal_mtp_common as common
from tests.studies import longitudinal_mtp_properties as properties
from tests.studies.default_band_properties import MINIMUM_CONTROL_POWER, control_power
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("longitudinal_mtp_design")

RUNNER = ROOT / "tests" / "canonical" / "longitudinal_mtp_runner.R"
#: A record, not a check: the pre-declaration failure-only probe on streams 0 to 19 under the
#: declared seeds found zero failures in every fit set, at the oracle bins of the second
#: declaration (``_x12_probe2.log`` beside the plan).
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
#: ``p0`` of the band cell, from ten fits of the continuous primary subject.
BAND_P0 = 0.8222
#: The exact-law power of the power cell: the declared contrast against its efficiency-bound
#: standard error at ``NULL_N`` (:func:`design_power`).
DESIGN_POWER = 1.0
#: A record, not a check: each control's mean distance from the truth over 400 draws of 2,000
#: rows at 320 oracle bins, in empirical standard deviations (``_x12_s2_decl3.log``).  Monte
#: Carlo standard error 0.05.
CONTROL_LIMITS = {"both_wrong": 3.083, "inverse_dropped": 2.655}
#: A record, not a check: the positive cells at their declared n = 2,000, over the same draws.
#: Each value is the mean error in empirical standard deviations; the Monte Carlo standard
#: error is 0.05.
POSITIVE_AT_DECLARED_N = {
    "up__both_correct": 0.110,
    "up__outcome_correct": 0.096,
    "up__mechanism_correct": 0.114,
    "history__declared_inverse": 0.148,
}
#: A record, not a check: the efficiency ratios of the continuous calibration contrasts over the
#: same draws, empirical and reported standard deviation over the bound.
EFFICIENCY_AT_DECLARED_N = {
    "up": (1.039, 1.012),
    "randomized_mtp": (1.017, 1.005),
    "msm_mtp": (1.044, 1.013),
}
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_000),
    ("double_robustness", "outcome_correct"): (2_000, 1_000),
    ("double_robustness", "mechanism_correct"): (2_000, 1_000),
    ("double_robustness", "both_wrong"): (2_000, 1_000),
    ("root_n_and_efficiency", "n_500"): (500, 600),
    ("root_n_and_efficiency", "n_2000"): (2_000, 600),
    ("root_n_and_efficiency", "n_8000"): (8_000, 600),
    ("interval_calibration", "correctly_specified"): (2_000, 2_000),
    ("interval_calibration", "classifier_route"): (2_000, 2_000),
    ("interval_calibration", "categorical_mtp"): (2_000, 2_000),
    ("interval_calibration", "vector_node"): (2_000, 2_000),
    ("interval_calibration", "randomized_mtp"): (2_000, 2_000),
    ("interval_calibration", "survival_mtp_h2"): (2_000, 2_000),
    ("interval_calibration", "msm_mtp"): (2_000, 2_000),
    ("type_i_error", "sharp_null"): (4_000, 600),
    ("power", "alternative"): (4_000, 600),
    ("targeting_necessity", "targeted"): (2_000, 1_000),
    ("inverse_necessity", "paired"): (2_000, 1_000),
    ("crossfit_overfitting", "paired"): (1_000, 10_000),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261041, 2026104101)
    assert (study.SMOKE_GATE, study.LEARNER_FOLDS, study.N_FOLDS) == (1e-6, 2, 5)
    from tests.studies.oracle_density_bins import oracle_bins

    assert (study.TILT_COPIES, study.G_BOUNDS) == (4, (0.001, 1.0))
    assert [oracle_bins(n) for n in (500, 1_000, 2_000, 4_000, 8_000)] == [127, 202, 320, 508, 807]
    assert record.reference == "lmtp"
    assert properties.BAND_REPLICATES == 2_000
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.TARGETING_DISPLACEMENT, properties.INVERSE_DISPLACEMENT) == (0.10, 0.10)
    assert set(study.STUDY.scenarios) == {
        study.CONTINUOUS,
        study.CROSSFIT,
        study.CATEGORICAL,
        study.TILT,
    }


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


def test_the_band_budget_meets_the_control_power_rule() -> None:
    assert control_power(BAND_P0 + 0.005, properties.BAND_REPLICATES) >= MINIMUM_CONTROL_POWER


# ------------------------------------------------------------------ truths


def test_the_continuous_truths_are_stable_under_a_finer_quadrature() -> None:
    for label in common.CONTINUOUS_LABELS:
        assert common._continuous_mean(label, 96) == pytest.approx(
            common._continuous_mean(label, 192), abs=1e-10
        )
    natural = common.continuous_truth("natural")
    for label in common.CONTINUOUS_LABELS[1:]:
        assert abs(common.continuous_truth(label) - natural) > 0.03, label


def test_the_finite_truths_are_the_oracle() -> None:
    law = common.CATEGORICAL_LAW
    for plan in common.CATEGORICAL_PLANS.values():
        assert common.finite_functional(law, law.probs, plan) == pytest.approx(
            law.mean(plan), abs=1e-14
        )
    vector = exact_law.truth("ey_regimen[vector]") - exact_law.truth("ey_regimen[natural]")
    assert vector == properties.VECTOR_TRUTH
    assert abs(study.TRUTH[study.TILT]["ate_regimen[rr 0.25 vs natural]"]) > 0.05
    assert abs(study.TRUTH[study.CATEGORICAL]["ate_regimen[minus one vs natural]"]) > 0.05
    msm = properties.PROJECTION @ np.array(
        [common.continuous_truth(label) for label in properties.MSM_CELLS]
    )
    assert msm[1] == pytest.approx(properties.MSM_TRUTH, abs=1e-15)


def test_the_finite_efficiency_bounds_are_the_complex_step() -> None:
    law = common.CATEGORICAL_LAW
    bound = common.finite_contrast_sd(
        law, common.CATEGORICAL_PLANS["minus one"], common.CATEGORICAL_PLANS["natural"]
    )
    assert bound == pytest.approx(properties.EFFICIENCY_SD["categorical_mtp"], rel=1e-12)
    vector = np.sqrt(
        np.sum(
            exact_law.PROBS
            * (exact_law.eif("ey_regimen[vector]") - exact_law.eif("ey_regimen[natural]")) ** 2
        )
    )
    assert vector == pytest.approx(properties.EFFICIENCY_SD["vector_node"], rel=1e-12)
    from functools import partial

    from tests import discrete_law_longitudinal_policy as policy_law
    from tests import discrete_law_survival as survival_law

    function = partial(exact_law.functional_mtp_survival, horizon=2)
    curve = policy_law.eif(function, survival_law.PROBS)
    survival = float(np.sqrt(np.sum(survival_law.PROBS * curve**2)))
    assert survival == pytest.approx(properties.EFFICIENCY_SD["survival_mtp_h2"], rel=1e-12)


def test_the_continuous_efficiency_bound_has_mean_zero_influence() -> None:
    """The closed-form influence function is centred, and the pinned bound is its own draw.

    The ratio is heavy tailed, so the standard deviation of 20,000 draws moves by several
    percent between seeds.  The pinned value is therefore checked against the declared
    400,000-draw computation, which is deterministic, and the centring on a separate sample.
    """
    frame = common.sample_continuous(20_000, 11)
    curve = common.continuous_eif("up", frame) - common.continuous_eif("natural", frame)
    assert abs(float(np.mean(curve))) < 4.0 * float(np.std(curve)) / np.sqrt(len(curve))
    assert common.continuous_contrast_sd("up") == pytest.approx(
        properties.EFFICIENCY_SD["up"], rel=1e-12
    )


# ------------------------------------------------------------------ controls


def _floor(replicates: int) -> float:
    """The margin, plus five Monte Carlo standard errors of a mean over ``replicates``."""
    return Margins().standardized_bias + 5.0 / np.sqrt(replicates)


def test_the_untargeted_control_is_zero_under_intercept_regressions() -> None:
    frame = common.sample_continuous(500, 3)
    assert properties.untargeted(frame, "mechanism_correct") == pytest.approx(0.0, abs=1e-12)
    scale = properties.EFFICIENCY_SD["up"] / np.sqrt(properties.TARGETING_N)
    assert abs(study.TRUTH[study.CONTINUOUS][properties.UP]) / scale > _floor(
        properties.TARGETING_REPLICATES
    )


def test_the_recorded_control_limits_clear_their_margins() -> None:
    assert CONTROL_LIMITS["both_wrong"] > _floor(properties.DOUBLE_ROBUST_REPLICATES)
    assert CONTROL_LIMITS["inverse_dropped"] > _floor(properties.INVERSE_REPLICATES)
    for empirical, reported in EFFICIENCY_AT_DECLARED_N.values():
        assert 0.9 < empirical < 1.1 and 0.9 < reported < 1.1


def test_the_inverse_dropped_control_moves_the_exact_law() -> None:
    """On the exact law, reading the history piece at the dose itself misses the truth."""
    from cleverly.longitudinal import LTMLE
    from tests import discrete_law_longitudinal as binary_law
    from tests import discrete_law_longitudinal_multivalue as multivalue

    def fit(drop: bool) -> float:
        with warnings.catch_warnings(), exact_plans.exact_bins(), properties._inverse_dropped(drop):
            warnings.simplefilter("ignore")
            result = LTMLE(
                exact_plans.regimens(("natural", "up then history")),
                reference="natural",
                outcome_learner=binary_law.CellMeans(),
                pseudo_learner=binary_law.CellMeans(),
                treatment_learner=multivalue.CellProbabilities(),
                n_folds=1,
                g_bounds=(1e-8, 1.0),
                simultaneous=False,
            ).fit(
                exact_law.frame(),
                outcome="Y",
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                continuous_treatment=["A1", "A2"],
            )
        return float(
            np.max(
                np.abs(
                    result.influence_curves["ey_regimen[up then history]"][exact_law.first_row_of()]
                    - exact_law.eif("ey_regimen[up then history]")
                )
            )
        )

    assert fit(drop=False) < 1e-10
    assert fit(drop=True) > 1e-4


# ------------------------------------------------------------------ the comparator's tables


def test_the_runner_names_the_python_plans_slugs_and_copies() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for scenario, labels in study.LABELS.items():
        match = re.search(rf"^  {scenario} = c\((.*?)\)", text, re.MULTILINE | re.DOTALL)
        assert match is not None, scenario
        pairs = re.findall(r'(?:"([^"]+)"|(\w[\w ]*?)) = "([^"]+)"', match.group(1))
        transcribed = {(quoted or bare).strip(): slug for quoted, bare, slug in pairs}
        assert transcribed == {label: study._slug(label) for label in labels}, scenario
    assert f"rr_tilt = {study.TILT_COPIES}L" in text
    assert "mtp_categorical = TRUE" in text


def test_the_power_cell_reaches_its_floor_on_the_exact_law() -> None:
    """The declared contrast, efficiency bound and size give the power cell its pass rate."""
    power = design_power(
        study.TRUTH[study.CONTINUOUS][properties.UP],
        properties.EFFICIENCY_SD["up"],
        properties.NULL_N,
    )
    assert power == pytest.approx(DESIGN_POWER, abs=1e-6)
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= 0.99


def test_the_quadrature_truths_match_a_simulation_of_definition_one() -> None:
    """Plan 6.12: an independent check of each continuous truth, by simulating the law.

    Each unit draws its natural dose, the policy moves it, the next covariate and the next
    natural dose are drawn given the intervened past, and the policy moves that dose too
    (Díaz et al. 2023, Definition 1).  The mean of the outcome regression over 400,000 such
    units must lie within four Monte Carlo standard errors of the quadrature.
    """
    rng = np.random.default_rng(20261061)
    n = 400_000
    w = rng.integers(0, 2, n).astype(float)
    for label, (d1, d2) in common.CONTINUOUS_MAPS.items():
        a1d = d1(common.truncated_draw(rng, common.mean1(w)))
        l2 = rng.binomial(1, common.p_l2(w, a1d)).astype(float)
        a2d = d2(common.truncated_draw(rng, common.mean2(w, a1d, l2)), a1d)
        q = common.outcome_mean(w, a1d, l2, a2d)
        se = float(np.std(q)) / np.sqrt(n)
        assert abs(float(np.mean(q)) - common.continuous_truth(label)) < 4.0 * se, label


def test_the_positive_cells_pass_with_high_probability_at_their_declared_size() -> None:
    """The bias rule's pass probability at each recorded mean.

    Every cell passes with a probability of at least 0.7.  ``history__declared_inverse`` is the
    one at risk, at about 0.74; if it reads red, the red-cell rule routes it.
    """
    from scipy.stats import norm
    from scipy.stats import t as student

    replicates = 1_000
    half = float(student.ppf(0.995, replicates - 1)) / np.sqrt(replicates)
    limit = Margins().standardized_bias - half
    spread = 1.0 / np.sqrt(replicates)
    probabilities = {
        cell: norm.cdf((limit - abs(mean)) / spread) - norm.cdf((-limit - abs(mean)) / spread)
        for cell, mean in POSITIVE_AT_DECLARED_N.items()
    }
    assert min(probabilities.values()) > 0.7, probabilities
    assert probabilities["history__declared_inverse"] == pytest.approx(0.74, abs=0.02)


def test_the_redesigned_overfitting_control_discriminates_before_the_run() -> None:
    """The pre-run probe of the third declaration: the pair can pass its fixed margins.

    ``tests/diagnostics/longitudinal_mtp_overfit_design`` fits both arms of the redesigned pair
    on 400 fresh draws.  The control's SE ratio clears its 0.75 ceiling at the bootstrap 99%
    upper end, and the paired coverage gain clears its 0.15 floor at the lower end of a 99%
    normal interval.  Every standard error of both arms is finite.  The cross-fitted arm reads
    about 1.20, at the upper end of its sanity band, as the single-tree design of run 2 did
    (1.167): the declaration states that risk rather than sizing it away.  Its pass probability
    at the declared 10,000 replications is about 0.23 from this probe, or about 0.4 pooled with
    the screen.  More replications cannot raise it: the SE ratio is structurally conservative,
    and a narrower interval around it does not move it under 1.2.  The range asserted below
    records the value and does not size the cell.
    """
    import pandas as pd

    from tests.diagnostics.longitudinal_mtp_overfit_design import run as probe
    from tests.studies.evidence.property_verdicts import (
        OVERFIT_COVERAGE_GAIN,
        OVERFIT_SE_CONTROL_CEILING,
    )

    rows = pd.read_csv(probe.HERE / "rows.csv.gz", float_precision="round_trip")
    assert len(rows) == probe.REPLICATES
    truth = float(properties.TRUTH[properties.CONTINUOUS][properties.UP])
    critical = 1.959963984540054
    covered = {}
    for arm in ("cross_fitted", "in_sample"):
        estimate = rows[f"{arm}_estimate"].to_numpy()
        error = rows[f"{arm}_std_error"].to_numpy()
        assert np.isfinite(error).all() and np.isfinite(estimate).all()
        covered[arm] = np.abs(estimate - truth) <= critical * error
    control = rows[["in_sample_estimate", "in_sample_std_error"]].to_numpy()
    draws = np.random.default_rng(20261008).integers(0, len(rows), size=(2_000, len(rows)))
    ratios = control[draws, 1].mean(axis=1) / control[draws, 0].std(axis=1, ddof=1)
    assert float(np.quantile(ratios, 0.995)) < OVERFIT_SE_CONTROL_CEILING
    gain = covered["cross_fitted"].astype(float) - covered["in_sample"].astype(float)
    lower = gain.mean() - 2.5758293035489004 * gain.std(ddof=1) / np.sqrt(len(gain))
    assert lower > OVERFIT_COVERAGE_GAIN
    positive = rows["cross_fitted_std_error"].mean() / rows["cross_fitted_estimate"].std(ddof=1)
    assert 1.15 < positive < 1.25
