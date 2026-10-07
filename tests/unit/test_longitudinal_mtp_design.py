"""The declared design of ``longitudinal-mtp``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The band cell's pointwise control needs a power of at least 0.99 at ``p0 + 0.005`` (the RM36
  rule).  Ten fits of the continuous primary subject give ``p0 = 0.8235``, where 2,000
  replications give power 1.0.
* The controls whose verdict is a bias outside the margin clear it, so a green control is
  evidence rather than luck.  The untargeted plug-in of the intercept-only regressions is the
  same number under every plan, so its contrast is exactly zero.  The other two are means over
  200 draws of 4,000 rows (:data:`CONTROL_LIMITS`), against the margin 0.25 plus five Monte
  Carlo standard errors of the cell's mean.  The inverse-dropped control is also checked on the
  exact law of ``tests/discrete_law_longitudinal_mtp.py``.
* The positive arms that rely on the mechanism alone stay inside the margin
  (:data:`POSITIVE_LIMITS`).  Their bins were chosen before any run by the rule "the smallest
  count whose mean standardized bias plus its Monte Carlo standard error is below 0.25".  At 80
  bins the arms read 0.54 and 0.60, and at 160 bins 0.21 and 0.24, each over 100 draws of
  8,000 rows.  ``MECHANISM_ONLY_BINS = 320`` meets the rule.
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
#: declared seeds found zero failures in every fit set, with the mechanism-only fit sets at 320
#: bins (``_x12_s2_probe.log`` beside the plan).
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
#: ``p0`` of the band cell, from ten fits of the continuous primary subject.
BAND_P0 = 0.8235
#: The exact-law power of the power cell: the declared contrast against its efficiency-bound
#: standard error at ``NULL_N`` (:func:`design_power`).
DESIGN_POWER = 1.0
#: A record, not a check: each control's mean distance from the truth over 200 draws of 4,000
#: rows, in efficiency-bound standard errors of a cell replicate at n = 2,000
#: (``_x12_s2_bp320.log``).  Monte Carlo standard errors 0.082 and 0.037.
CONTROL_LIMITS = {"both_wrong": 4.899, "inverse_dropped": 2.103}
#: A record, not a check: each mechanism-only positive arm's mean standardized bias at 320 bins
#: and its Monte Carlo standard error, from the same draws.  ``both_correct`` at 80 bins is the
#: reference: it reads the same finite-sample bias.
POSITIVE_LIMITS = {
    "up, mechanism correct": (0.110, 0.053),
    "up then history, declared inverse": (0.127, 0.053),
    "up, both correct": (0.112, 0.055),
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
    assert record.publication_policy == "gated"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261041, 2026104101)
    assert (study.SMOKE_GATE, study.LEARNER_FOLDS, study.N_FOLDS) == (1e-6, 2, 5)
    assert (study.DENSITY_BINS, study.TILT_COPIES, study.G_BOUNDS) == (80, 4, (0.001, 1.0))
    assert study.MECHANISM_ONLY_BINS == 320
    assert study.density_bins("mechanism_correct") == 320
    assert {study.density_bins(c) for c in ("primary", "both_correct", "both_wrong")} == {80}
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
    for bias, monte_carlo in POSITIVE_LIMITS.values():
        assert abs(bias) + monte_carlo < Margins().standardized_bias


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
