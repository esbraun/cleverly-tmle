"""The declared design of ``stochastic-categorical-ltmle``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The band cell's pointwise control needs a power of at least 0.99 at ``p0 + 0.005`` (the
  RM36 rule).  Ten fits of the primary subject give ``p0 = 0.8560``, where 2,000 replications
  give power 1.0.
* The two controls whose verdict is a bias outside the margin clear it on the exact law, so a
  green control is evidence rather than luck: the uniform-policy control sits 0.522 and the
  both-wrong control 1.002 efficiency-scaled standard deviations from the truth, against the
  margin 0.25 plus five Monte Carlo standard errors of a 1,200-replication mean.
* The comparator's policy and mechanism tables, transcribed by hand into
  ``tests/canonical/stochastic_categorical_ltmle_runner.R``, are the Python tables by label,
  and every history cell's allocation of four copies reproduces ``4 q`` exactly.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` of the properties
module counts failed fits per fit set and records no estimate; on streams 0 to 199 of every fit
set it found none (:data:`FAILURE_PROBE_200`).  The shared harness refuses a cell that lost a
replication, so a failure in the run itself stops the run.  That cell drops to its red-cell owner
with its failure count published, and the run repeats without it, with no other change.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_longitudinal_multivalue as law
from tests import discrete_law_longitudinal_policy as policy
from tests.studies import canonical_stochastic_categorical_ltmle as study
from tests.studies import stochastic_categorical_ltmle_properties as properties
from tests.studies.default_band_properties import MINIMUM_CONTROL_POWER, control_power
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("stochastic_categorical_ltmle_design")

RUNNER = ROOT / "tests" / "canonical" / "stochastic_categorical_ltmle_runner.R"
#: A record, not a check: the pre-declaration failure-only probe on streams 0 to 199 under the
#: declared seeds found zero failures in every fit set (``_f1_failure200b.log`` beside the plan).
FAILURE_PROBE_200 = dict.fromkeys(
    (f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0
)
#: ``p0`` of the band cell, from ten fits of the primary subject.
BAND_P0 = 0.8560
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_200),
    ("double_robustness", "outcome_correct"): (2_000, 1_200),
    ("double_robustness", "mechanism_correct"): (2_000, 1_200),
    ("double_robustness", "both_wrong"): (2_000, 1_200),
    ("root_n_and_efficiency", "n_500"): (500, 800),
    ("root_n_and_efficiency", "n_2000"): (2_000, 800),
    ("root_n_and_efficiency", "n_8000"): (8_000, 800),
    ("interval_calibration", "correctly_specified"): (2_000, 4_000),
    ("interval_calibration", "policy_risk_h2"): (2_000, 4_000),
    ("interval_calibration", "msm_policy"): (2_000, 4_000),
    ("type_i_error", "sharp_null"): (4_000, 800),
    ("power", "alternative"): (4_000, 800),
    ("targeting_necessity", "targeted"): (2_000, 1_200),
    ("policy_necessity", "paired"): (2_000, 1_200),
    ("randomizer_projection", "paired"): (2_000, 1_200),
    ("crossfit_overfitting", "paired"): (1_000, 40_000),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (2_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261033, 2026103301)
    assert (study.COPIES, study.SMOKE_GATE, study.LEARNER_FOLDS) == (4, 1e-6, 2)
    assert record.reference == "lmtp"
    assert properties.BAND_REPLICATES == 2_000
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.TARGETING_DISPLACEMENT, properties.POLICY_DISPLACEMENT) == (0.10, 0.10)
    assert properties.PROJECTION_LEVEL == 0.99
    assert study.ESTIMANDS == (
        "ey_regimen[low]",
        "ey_regimen[mix]",
        "ey_regimen[taper]",
        "ate_regimen[mix vs low]",
        "ate_regimen[taper vs low]",
    )


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


# ------------------------------------------------------------------ exact-law separations


def _floor(replicates: int) -> float:
    """The margin, plus five Monte Carlo standard errors of a mean over ``replicates``."""
    return Margins().standardized_bias + 5.0 / np.sqrt(replicates)


def test_the_uniform_policy_control_clears_its_margin_on_the_exact_law() -> None:
    flat = (("policy", np.full((2, 3), 1.0 / 3.0)), ("policy", np.full((2, 3, 2, 3), 1.0 / 3.0)))
    control = policy.functional_policy(policy.PROBS, flat) - policy.TRUTH["ey_regimen[low]"]
    scale = properties.EFFICIENCY_SD["mix"] / np.sqrt(properties.POLICY_N)
    separation = abs(control - study.TRUTH[properties.MIX]) / scale
    assert separation == pytest.approx(0.5222, abs=5e-5)
    assert separation > _floor(properties.POLICY_REPLICATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = study.fit(law.frame(), configuration="both_correct", uniform=True)
    assert fitted.psi(properties.MIX) == pytest.approx(control, abs=1e-12)


def test_the_both_wrong_control_clears_its_margin_on_the_exact_law() -> None:
    """On the exact 512-row law the fit is its own population limit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        limit = study.fit(law.frame(), configuration="both_wrong").psi(properties.MIX)
    scale = properties.EFFICIENCY_SD["mix"] / np.sqrt(properties.DOUBLE_ROBUST_N)
    separation = abs(limit - study.TRUTH[properties.MIX]) / scale
    assert separation == pytest.approx(1.0017, abs=5e-5)
    assert separation > _floor(properties.DOUBLE_ROBUST_REPLICATES)


def test_the_longhand_plug_in_is_the_g_formula_on_the_exact_law() -> None:
    """The untargeted control's recursion is the estimator's, less the fluctuation."""
    frame = law.frame()
    for label in ("low", "mix"):
        value = properties.untargeted(frame, label, "both_correct")
        assert value == pytest.approx(study.TRUTH[f"ey_regimen[{label}]"], abs=1e-12)


def test_the_untargeted_control_differs_from_the_targeted_fit() -> None:
    """A witness that the control removes a step: one draw, a correct mechanism only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties._fit_set_rows(
            ("targeting_necessity", "targeted", "mechanism_correct", 0, 2_000, 2, "policy")
        )
    targeted, untargeted = (row["estimate"] for row in rows)
    assert abs(targeted - untargeted) > 0.01


def test_the_truths_are_the_oracle() -> None:
    for name in study.ESTIMANDS:
        assert study.TRUTH[name] == policy.TRUTH[name]
    assert pytest.approx(0.0, abs=1e-15) == properties.NULL_TRUTH
    msm = properties.PROJECTION @ np.array(
        [policy.TRUTH[f"ey_regimen[{label}]"] for label in study.LABELS]
    )
    assert msm[1] == pytest.approx(properties.MSM_TRUTH, abs=1e-15)
    assert abs(properties.MSM_TRUTH) > 0.005


# ------------------------------------------------------------------ the projection verdict


def _projection_rows(integrated_sd: float, recorded_sd: float, replicates: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for replicate in range(replicates):
        common = rng.normal()
        for cell, sd in (
            ("mix__integrated", integrated_sd),
            ("mix__recorded_randomizer", recorded_sd),
        ):
            rows.append(
                {
                    "property": "randomizer_projection",
                    "cell": cell,
                    "replicate": replicate,
                    "estimate": sd * (0.8 * common + 0.6 * rng.normal()),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("integrated", "recorded", "below"), [(1.0, 1.3, True), (1.0, 1.0, False), (1.3, 1.0, False)]
)
def test_the_projection_bound_reads_the_paired_spread(
    integrated: float, recorded: float, below: bool
) -> None:
    point, upper = properties.projection_sd_ratio(
        _projection_rows(integrated, recorded, 1_200), replicates=2_000, level=0.99, seed=1
    )
    assert point == pytest.approx(integrated / recorded, rel=0.1)
    assert (upper < 1.0) is below


# ------------------------------------------------------------------ the comparator's tables


def _vector(text: str, name: str) -> list[float]:
    match = re.search(rf"^\s*{name} = c\(([^)]*)\)", text, re.MULTILINE)
    assert match is not None, f"the runner no longer writes {name} as one flat c(...) vector"
    return [float(value) for value in match.group(1).replace("\n", " ").split(",")]


ARMS = ("standard", "high", "low")


def test_the_runners_policy_tables_are_the_python_policies_by_label() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for layout in (
        "W = c(0, 1)",
        "W = rep(c(0, 1), each = 6)",
        'A1 = rep(rep(c("standard", "high", "low"), each = 2), 2)',
        "L2 = rep(c(0, 1), 6)",
        'arms <- c("standard", "high", "low")',
    ):
        assert layout in text, f"the runner no longer lays its tables out as {layout}"
    for arm in ARMS:
        first = _vector(text, f"q_{arm}")
        expected = [policy.POLICY1[w, policy.ARM_LABELS.index(arm)] for w in range(2)]
        assert first == expected, arm
    second = {
        arm: [float(value) for value in re.findall(rf"q_{arm} = c\(([^)]*)\)", text)[1].split(",")]
        for arm in ARMS
    }
    row = 0
    for w in range(2):
        for first_arm in ARMS:
            for l2 in range(2):
                for arm in ARMS:
                    expected = policy.POLICY2[
                        w, policy.ARM_LABELS.index(first_arm), l2, policy.ARM_LABELS.index(arm)
                    ]
                    assert second[arm][row] == expected, (w, first_arm, l2, arm)
                row += 1
    assert row == 12


def test_the_runners_mechanism_tables_are_the_law() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    first = np.asarray(_vector(text, "probability"), dtype=float).reshape(2, 3)
    np.testing.assert_array_equal(first, law.G1[:, [law.ARM_LABELS.index(arm) for arm in ARMS]])
    columns = {arm: _vector(text, arm) for arm in ARMS}
    row = 0
    for w in range(2):
        for first_arm in ARMS:
            for l2 in range(2):
                for arm in ARMS:
                    expected = law.G2[
                        w, law.ARM_LABELS.index(first_arm), l2, law.ARM_LABELS.index(arm)
                    ]
                    assert columns[arm][row] == expected
                row += 1


def test_every_policy_row_is_realised_exactly_by_the_declared_copies() -> None:
    """``allocate_copies`` refuses a row off the 1 / copies grid; none of the study's is."""
    rows = [*policy.POLICY1, *policy.POLICY2.reshape(-1, 3)]
    for row in rows:
        counts = row * study.COPIES
        np.testing.assert_array_equal(counts, np.rint(counts))
        assert counts.sum() == study.COPIES
    text = RUNNER.read_text(encoding="utf-8")
    assert f"copies <- {study.COPIES}L" in text
    assert re.search(r'^plans <- c\("low", "mix", "taper"\)', text, re.MULTILINE)
