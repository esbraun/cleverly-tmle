"""The declared design of ``policy-point-mtp``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The controls whose verdict is a bias outside the margin clear it, so a green control is
  evidence rather than luck.  The untargeted plug-in of the prior-only outcome regression is
  the same number under both policies, so its contrast is exactly zero.  The inverse-dropped
  control is exactly zero for the same reason: the dropped ratio takes the same value at a dose
  and at its image, so the joint fluctuation returns the sample mean under both policies.  The
  both-wrong control is measured as a mean over eight draws of 8,000 rows
  (:data:`CONTROL_LIMITS`).  Each limit is compared with the margin 0.25 plus five Monte Carlo
  standard errors of the cell's mean.
* The positive arms that read the binned oracle density stay inside the margin.  The mean over
  40 draws of 8,000 rows is recorded in :data:`POSITIVE_LIMITS`, with its Monte Carlo
  standard error.
* The power cell is sized on the exact law (:data:`DESIGN_POWER`): the declared contrast,
  the efficiency bound and ``NULL_N`` give a probability of at least 0.99 that its exact lower
  rejection endpoint clears ``MINIMUM_POWER`` over ``NULL_REPLICATES`` replications.
* The comparator's policy labels and column slugs, transcribed by hand into
  ``tests/canonical/policy_point_mtp_runner.R``, are the Python ones.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` of the properties
module counts failed fits per fit set and records no estimate.  On streams 0 to 19 of every fit
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

from tests.studies import canonical_policy_point_mtp as study
from tests.studies import policy_point_mtp_properties as properties
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("policy_point_mtp_design")

RUNNER = ROOT / "tests" / "canonical" / "policy_point_mtp_runner.R"
#: A record, not a check: the pre-declaration failure-only probe on streams 0 to 19 under the
#: declared seeds found zero failures in every fit set (``_x12_s1_probe.log`` beside the plan).
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
#: The exact-law power of the power cell: the declared contrast against its efficiency-bound
#: standard error at ``NULL_N`` (:func:`design_power`).
DESIGN_POWER = 0.942235
#: A record, not a check: each control's distance from the truth, in efficiency-bound standard
#: errors of a cell replicate at n = 2,000 (``_x12_s1_limits.log``).
CONTROL_LIMITS = {"both_wrong": 2.959, "inverse_dropped": 2.808}
#: A record, not a check: the standardized bias of each positive arm read with the binned
#: oracle density and a prior-only outcome, and its Monte Carlo standard error
#: (``_x12_s1_positives.log``).
POSITIVE_LIMITS = {
    "x1.25": (0.076, 0.095),
    "piecewise": (-0.077, 0.077),
    "halve below 3": (0.024, 0.075),
}
#: The declared size of every fit set: ``(family, label) -> (n, replications)``.
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_000),
    ("double_robustness", "outcome_correct"): (2_000, 1_000),
    ("double_robustness", "density_correct"): (2_000, 1_000),
    ("double_robustness", "both_wrong"): (2_000, 1_000),
    ("root_n_and_efficiency", "n_500"): (500, 600),
    ("root_n_and_efficiency", "n_2000"): (2_000, 600),
    ("root_n_and_efficiency", "n_8000"): (8_000, 600),
    ("interval_calibration", "correctly_specified"): (2_000, 2_000),
    ("interval_calibration", "piecewise"): (2_000, 2_000),
    ("interval_calibration", "halve"): (2_000, 2_000),
    ("interval_calibration", "classifier_route"): (2_000, 2_000),
    ("type_i_error", "sharp_null"): (4_000, 600),
    ("power", "alternative"): (4_000, 600),
    ("targeting_necessity", "targeted"): (2_000, 1_000),
    ("inverse_necessity", "paired"): (2_000, 1_000),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "gated"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261043, 2026104301)
    assert study.DENSITY_BINS == 160
    assert (study.FACTOR, study.CAP, study.KNEE, study.DROP, study.HALVE_BELOW) == (
        1.25,
        5.5,
        3.0,
        0.5,
        3.0,
    )
    assert record.reference == "lmtp"
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.TARGETING_DISPLACEMENT, properties.INVERSE_DISPLACEMENT) == (0.10, 0.10)
    assert set(record.scenarios) == {study.SCENARIO}


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


def test_the_truths_are_stable_under_a_finer_quadrature() -> None:
    for label in study.LABELS:
        assert study._mean(label, 96, 60) == pytest.approx(study._mean(label, 192, 90), abs=1e-10)
    for label in study.LABELS[1:]:
        name = f"ate_policy[{label} vs {study.REFERENCE}]"
        assert abs(study.TRUTH[name]) > 0.009, label


def test_the_pinned_efficiency_bounds_are_the_closed_form() -> None:
    for key, label in (
        ("x1_25", "x1.25"),
        ("piecewise", "piecewise"),
        ("halve", "halve below 3"),
    ):
        assert properties.contrast_sd(label) == pytest.approx(
            properties.EFFICIENCY_SD[key], rel=1e-12
        )
    assert properties.EFFICIENCY_SD["classifier_route"] == properties.EFFICIENCY_SD["x1_25"]


def test_the_closed_form_ratio_is_the_package_ratio_under_the_true_density() -> None:
    """Equation (3) as the bound reads it agrees with the policy classes on the same density."""
    from cleverly.interventions.policy import lazy_frame, policy_branches

    frame = study.sample(400, 5)
    a = frame["A"].to_numpy()
    mean = study.dose_mean(frame["W1"].to_numpy(), frame["W2"].to_numpy())
    rows = lazy_frame(lambda: frame)
    for policy in study.policies()[1:]:
        ((_, branch),) = policy_branches(policy)
        package = branch.ratio(a, rows, lambda b: study.truncated_pdf(b, mean))
        closed = properties._true_ratio(policy.name, a, mean)
        np.testing.assert_allclose(package, closed, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ controls


def _floor(replicates: int) -> float:
    """The margin, plus five Monte Carlo standard errors of a mean over ``replicates``."""
    return Margins().standardized_bias + 5.0 / np.sqrt(replicates)


def test_the_untargeted_control_is_zero_under_a_prior_outcome() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.fit(
            study.sample(500, 3), configuration="density_correct", chosen=properties._pair("x1.25")
        )
    assert properties.initial_contrast(result, "x1.25") == pytest.approx(0.0, abs=1e-12)
    scale = properties.EFFICIENCY_SD["x1_25"] / np.sqrt(properties.TARGETING_N)
    assert abs(study.TRUTH[properties.X125]) / scale > _floor(properties.TARGETING_REPLICATES)


def test_the_inverse_dropped_control_returns_the_natural_course() -> None:
    """The dropped ratio is invariant under the map, so the contrast is exactly zero."""
    with warnings.catch_warnings(), properties._inverse_dropped(True):
        warnings.simplefilter("ignore")
        result = study.fit(
            study.sample(500, 4),
            configuration="density_correct",
            chosen=properties._pair("halve below 3"),
        )
    assert result[properties.HALVE].psi == pytest.approx(0.0, abs=1e-10)
    scale = properties.EFFICIENCY_SD["halve"] / np.sqrt(properties.INVERSE_N)
    assert abs(study.TRUTH[properties.HALVE]) / scale > _floor(properties.INVERSE_REPLICATES)


def test_the_recorded_limits_clear_their_margins() -> None:
    assert CONTROL_LIMITS["both_wrong"] > _floor(properties.DOUBLE_ROBUST_REPLICATES)
    assert CONTROL_LIMITS["inverse_dropped"] > _floor(properties.INVERSE_REPLICATES)
    for bias, monte_carlo in POSITIVE_LIMITS.values():
        assert abs(bias) + monte_carlo < Margins().standardized_bias


# ------------------------------------------------------------------ the comparator's table


def test_the_runner_names_the_python_policies_and_slugs() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r"^policies <- c\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    assert match is not None
    pairs = re.findall(r'(?:"([^"]+)"|(\w[\w ]*?)) = "([^"]+)"', match.group(1))
    transcribed = {(quoted or bare).strip(): slug for quoted, bare, slug in pairs}
    assert transcribed == {label: study.slug(label) for label in study.LABELS}
    assert f'reference <- "{study.REFERENCE}"' in text


def test_the_power_cell_reaches_its_floor_on_the_exact_law() -> None:
    """The declared contrast, efficiency bound and size give the power cell its pass rate."""
    power = design_power(
        study.TRUTH[properties.X125], properties.EFFICIENCY_SD["x1_25"], properties.NULL_N
    )
    assert power == pytest.approx(DESIGN_POWER, abs=1e-6)
    assert power_cell_pass_probability(power, properties.NULL_REPLICATES) >= 0.99
