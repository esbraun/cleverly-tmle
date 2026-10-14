"""The declared design of the registered study ``ctmle-oat-per-arm-strata-missing``.

Every number here was fixed before any registered run: the single primary ``n`` and budget,
the seeds, the laws and their truths, conditions a, b, d, e, f, g, r and s of ``X29-plan.md``
section 3.4, the cells a run publishes, and the reading rule.  The ladder and its condition t
were re-declared after the first run, before the rerun of the property phase
(:mod:`tests.studies.ctmle_oat_per_arm_strata_missing_properties` gives the reason).  The post-run gate runs once the
committed artifacts exist: every red truth row meets the F19 route, and no paired, property
or control row is red.  Before the run it is skipped.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_ctmle_oat_per_arm_strata_missing as study
from tests.studies import ctmle_oat_per_arm_strata_missing_properties as properties
from tests.studies import oat_per_arm_extension_laws as laws
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("ctmle_oat_per_arm_strata_missing_design")

RUNNER = study.STUDY.artifacts / "run_drtmle_oat_per_arm_strata_missing.R"

#: The Monte Carlo size of the fast-tier recomputation.  ``_x29_law_design.log`` holds the
#: declared values at ``10^6`` draws; at this size a gap or a distance moves by well under the
#: distance from its floor, which the declared values below show.
CONDITION_DRAWS = 40_000
CONDITION_SEED = 20261029


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.slug == "ctmle-oat-per-arm-strata-missing"
    assert (record.n, record.replicates) == (2_250, 800)
    assert (record.n, record.replicates) == (study.PRIMARY_N, study.PRIMARY_REPLICATES)
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert record.implementation == "cleverly-ctmle-oat-per-arm"
    assert record.reference == "drtmle-r-oat-per-arm-strata-missing"
    assert record.anchor == "outcome-adaptive-per-arm-c-tmle-with-strata-and-missing-data"
    assert record.document.endswith(
        "method-evidence/outcome-adaptive-per-arm-c-tmle-strata-missing.md"
    )
    assert set(record.extra_artifacts) == {"design-gap.csv", "fit-diagnostics.csv"}
    assert study.RED_CELL_OWNER == "F19"
    assert study.ROUTED_SE_RATIO_CEILING == 1.07
    assert study.ROUTED_SE_RATIO_FLOOR == 0.70
    assert study.PAIRED_GAP_CEILING == 0.005
    assert laws.G_BOUNDS == (0.01, 0.99)
    assert study.NUISANCE_BOUND == 0.01
    assert study.LADDER == (2_000, 4_000, 8_000)
    assert study.CONFIGURATION["ladder"] == study.LADDER_DECLARATION
    for number in ("1.575", "0.0884", "0.0561", "117.1", "ae53bbb", "39.5", "9.9"):
        assert number in study.LADDER_DECLARATION, number
    assert study.CONFIGURATION["property_summary_revision"] == study.PROPERTY_SUMMARY_REVISION
    assert properties.PROPERTY_REPLICATES == properties.LADDER_REPLICATES == 800
    for configuration in properties.CONFIGURATIONS:
        assert configuration.replicates == 800, configuration.name
    sizes = {configuration.name: configuration.n for configuration in properties.CONFIGURATIONS}
    assert sizes == {
        "natural_course_insample": 2_250,
        "missing_outcome_crossfit": 2_000,
        "strata_missing": 3_000,
        "weighted_strata_missing": 3_000,
        "strata_missing_outcome_wrong": 3_000,
        "n_2000": 2_000,
        "n_4000": 4_000,
        "n_8000": 8_000,
    }


def test_the_study_is_registered_and_draws_from_its_own_seeds() -> None:
    assert study.STUDY.slug in {record.slug for record in registered()}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not {study.SEED, study.RESAMPLING_SEED} & others
    seeds = [cell.seed for cell in properties.declared_cells()]
    roots = {
        configuration.name: configuration.seed(0) for configuration in properties.CONFIGURATIONS
    }
    assert len(set(roots.values())) == len(roots)
    assert len(set(seeds)) == len(properties.CONFIGURATIONS)


def test_the_declared_cells_are_the_published_cells() -> None:
    declared = {(cell.property, cell.cell) for cell in properties.declared_cells()}
    rates = {("root_n_rate", "empirical_sd"), ("root_n_rate", "reported_se")}
    published = {(family, cell) for family, cells in study.PROPERTY_CELLS.items() for cell in cells}
    assert declared | rates == published
    assert len(properties.declared_cells()) == len(declared)
    roles = {cell.cell: cell.role for cell in properties.declared_cells()}
    assert [cell for cell, role in roles.items() if role == "control"] == [
        "ate__strata_missing_outcome_wrong"
    ]


def test_every_cell_reads_an_estimand_its_law_reports() -> None:
    for cell in properties.declared_cells():
        assert cell.dgp in laws.LAWS.values()
        assert cell.estimand in cell.dgp.truth(), (cell.cell, cell.estimand)
    for scenario, estimands in study.SCENARIO_ESTIMANDS.items():
        assert set(estimands) <= set(study.SCENARIO_LAWS[scenario].truth()), scenario


def _reference_names(scenario: str, strata: tuple[int, ...]) -> set[str]:
    """The estimand names ``fit_one`` of the R runner writes, by its naming rule."""
    arms = 3 if scenario == "three_arm_missing" else 2

    def mean_name(a: int, tail: str) -> str:
        return f"ey{a}{tail}" if arms == 2 else f"ey[{a}]"

    def pair(stem: str, a: int, tail: str) -> str:
        return f"{stem}{tail}" if arms == 2 else f"{stem}[{a} vs 0]"

    def contrasts(tail: str, ratios: bool) -> set[str]:
        out = set()
        for a in range(1, arms):
            out.add(pair("ate", a, tail))
            if ratios:
                out |= {pair("rr", a, tail), pair("or", a, tail)}
        return out

    names = {mean_name(a, "") for a in range(arms)}
    if scenario != "strata":
        return names | contrasts("", ratios=True)
    names |= contrasts("", ratios=False)
    for s in strata:
        tail = f"[S={s}]"
        names |= {mean_name(a, tail) for a in range(arms)} | contrasts(tail, ratios=False)
    return names


def test_the_reference_runner_reports_every_primary_estimand() -> None:
    """Every primary estimand is paired: the R runner writes each name the record declares.

    The runner builds its names with the format strings checked here, so the mirror of its
    rule in :func:`_reference_names` and the source cannot drift apart unnoticed.
    """
    source = RUNNER.read_text(encoding="utf-8")
    for fragment in (
        'sprintf("ey%d%s", a, tail %||% "")',
        'sprintf("ey[%d]", a)',
        'sprintf("%s%s", stem, tail %||% "")',
        'sprintf("%s[%d vs 0]", stem, a)',
        'sprintf("[S=%d]", as.integer(s))',
        'contrast_rows(frame, add, totals, arms, "", pair)',
        "contrast_rows(frame, add, totals, arms, NULL, pair)",
        "if (is.null(tail)) {",
        'implementation <- "drtmle-r-oat-per-arm-strata-missing"',
        'if (scenario == "strata")',
        'if (scenario == "three_arm_missing")',
    ):
        assert fragment in source, fragment
    strata = tuple(range(len(laws.STRATA_COMPLETE.strata)))
    for scenario, estimands in study.SCENARIO_ESTIMANDS.items():
        assert set(estimands) == _reference_names(scenario, strata), scenario
    assert set(study.STUDY.scenarios) == set(study.SCENARIO_LAWS)


def test_each_configuration_writes_one_row_per_estimand() -> None:
    """One replication of every configuration, at its declared size: the row count per cell."""
    for configuration in properties.CONFIGURATIONS:
        rows = properties.configuration_rows(configuration, n_jobs=1, replicates=1)
        assert list(rows["cell"]) == list(configuration.cells), configuration.name
        assert set(rows["failed_replicates"]) == {0}
        assert set(rows["role"]) == {configuration.role}
        assert set(rows["n"]) == {configuration.n}
        truths = configuration.law.truth()
        expected = [truths[name] for name in configuration.estimands]
        assert rows["truth"].to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize("law", list(laws.LAWS.values()), ids=lambda law: law.name)
def test_each_truth_matches_a_monte_carlo(law: laws.ExtensionLaw) -> None:
    """The quadrature arm means and ``ey_obs`` against ``10^6`` draws, to three MC errors."""
    sample = laws.draw(law, 1_000_000, 20261029)
    weights = laws.weight(sample.w1) if law.weighted else np.ones_like(sample.w1)
    truth = law.truth()
    scale = float(np.max(weights))

    def check(values: np.ndarray, share: np.ndarray, target: float) -> None:
        mean = np.average(values, weights=share)
        error = np.sqrt(np.average((values - mean) ** 2, weights=share) / values.size)
        assert abs(mean - target) < 3.0 * error * scale / np.sqrt(np.mean(share > 0)) + 1e-12

    q = np.column_stack(
        [law.outcome_probability(a, sample.w1, sample.w2, sample.s) for a in range(law.k)]
    )
    g = law.mechanism(sample.w1, sample.w3, sample.s)
    names = [f"ey{a}" for a in range(law.k)] if law.k == 2 else [f"ey[{a}]" for a in range(law.k)]
    for a, name in enumerate(names):
        check(q[:, a], weights, truth[name])
    check(np.sum(g * q, axis=1), weights, truth["ey_obs"])
    for s, label in enumerate(law.stratum_labels):
        share = weights * (sample.s == s)
        for a in range(law.k):
            check(q[:, a], share, truth[f"ey{a}[{label}]"])


# ------------------------------------------------------------------------- the conditions


@pytest.fixture(scope="module")
def draws() -> dict[str, laws.Draws]:
    return {law.name: laws.draw(law, CONDITION_DRAWS, CONDITION_SEED) for law in laws.DESIGN_LAWS}


def test_condition_a_the_strata_design_differs_from_the_pooled_design(draws) -> None:
    for law in (laws.STRATA_MISSING, laws.STRATA_COMPLETE):
        gaps = laws.strata_design_gap(law, draws[law.name])
        assert min(gaps) >= laws.DESIGN_GAP_FLOOR, (law.name, gaps)


def test_condition_b_the_missing_designs_differ_from_the_shipped_products(draws) -> None:
    for law in laws.DESIGN_LAWS:
        sample = draws[law.name]
        if law.missing_outcome:
            gaps = laws.missing_design_gap(law, sample)
            assert min(gaps) >= laws.DESIGN_GAP_FLOOR, (law.name, gaps)
            natural = laws.natural_course_design_gap(law, sample)
            assert natural >= laws.DESIGN_GAP_FLOOR, (law.name, natural)
        if law.missing_treatment:
            gaps = laws.recording_design_gap(law, sample)
            assert min(gaps) >= laws.DESIGN_GAP_FLOOR, (law.name, gaps)


def test_condition_d_every_factor_projection_is_inside_its_bound(draws) -> None:
    for law in laws.DESIGN_LAWS:
        ranges = laws.factor_ranges(law, draws[law.name])
        assert laws.factors_inside_bounds(ranges), (law.name, ranges)


def test_condition_e_the_learner_limit_is_close_to_each_projection(draws) -> None:
    for law in laws.DESIGN_LAWS:
        distances = laws.learner_distances(law, draws[law.name])
        assert max(distances.values()) <= laws.LEARNER_DISTANCE_CEILING, (law.name, distances)


def test_condition_f_the_means_and_the_stratum_effects_are_separated() -> None:
    for law in laws.LAWS.values():
        assert laws.arm_means_inside(law), law.name
    for law in (laws.STRATA_MISSING, laws.STRATA_COMPLETE):
        assert laws.stratum_ate_spacing(law) >= laws.STRATUM_ATE_SPACING


def test_condition_g_the_smallest_stratum_is_large_enough() -> None:
    share = min(laws.STRATA_COMPLETE.strata)
    probability = laws.smallest_stratum_probability(study.PRIMARY_N, share)
    assert probability >= laws.SMALLEST_STRATUM_PROBABILITY


def test_condition_r_each_positive_cell_is_sized_for_its_pass_probability() -> None:
    """A cell whose true standardized bias is 0.10 passes the bias rule with probability 0.90."""
    budgets = {cell.replicates for cell in properties.declared_cells() if cell.role == "positive"}
    budgets.add(study.PRIMARY_REPLICATES)
    for replicates in budgets:
        probability = laws.equivalence_pass_probability(replicates, laws.DESIGN_BIAS)
        assert probability >= laws.PASS_PROBABILITY_FLOOR, (replicates, probability)
    # The witness: the plan's R = 500 rung budget would not have met the condition.
    assert laws.equivalence_pass_probability(500, laws.DESIGN_BIAS) < laws.PASS_PROBABILITY_FLOOR


#: Condition (t) per configuration: the expected rows in the smallest class of any factor's
#: training target, recomputed from :func:`laws.smallest_minority_class`.  The binding class is
#: the nonrespondents of arm 0 in ``S = 2`` on the strata law, the nonrespondents of arm 1 (arm
#: 0 on three arms) on the missing-data laws, and arm 0 in ``S = 2`` on ``strata_complete``.
MINORITY_CLASS = {
    "primary/strata": 164.7,
    "primary/missing_outcome": 150.5,
    "primary/three_arm_missing": 116.4,
    "primary/missing_treatment": 110.5,
    "natural_course_insample": 150.5,
    "missing_outcome_crossfit": 107.0,
    "strata_missing": 59.2,
    "weighted_strata_missing": 59.2,
    "strata_missing_outcome_wrong": 59.2,
    "n_2000": 39.5,
    "n_4000": 79.0,
    "n_8000": 157.9,
}


def test_condition_t_the_smallest_minority_class_of_each_configuration() -> None:
    """The expected smallest class of each factor's training target, per configuration.

    Condition (t) as first declared counted the rows of the smallest factor training cell
    (117.1 at ``n = 2,000``).  A logistic factor fit is bound by its smaller class, and on the
    strata law that is the nonrespondents among those rows.  The condition sets no floor.  The
    witness is the first run's ``n = 500`` rung, at about 10, where 40 of 800 fits had a
    standard error above twice the empirical SD; ``test_the_tail_fits_are_rare`` reads the
    declared configurations.
    """
    measured = {
        f"primary/{scenario}": laws.smallest_minority_class(law, study.PRIMARY_N, cross_fit=False)
        for scenario, law in study.SCENARIO_LAWS.items()
    }
    measured |= {
        item.name: laws.smallest_minority_class(item.law, item.n, cross_fit=item.cross_fit)
        for item in properties.CONFIGURATIONS
    }
    assert measured == pytest.approx(MINORITY_CLASS, abs=0.05)
    shares = laws.minority_class_shares(laws.STRATA_MISSING)
    assert min(shares, key=shares.__getitem__) == ("outcome_observation", 2, 0)
    dropped = laws.smallest_minority_class(laws.STRATA_MISSING, 500, cross_fit=True)
    assert dropped == pytest.approx(9.9, abs=0.05)
    assert dropped < min(measured.values()) / 3


@pytest.mark.parametrize(
    "law", [laws.STRATA_MISSING, laws.MISSING_TREATMENT], ids=lambda law: law.name
)
def test_the_cell_shares_match_a_monte_carlo(law: laws.ExtensionLaw) -> None:
    """The quadrature cell shares against ``2 x 10^5`` draws, to four MC errors.

    ``missing_treatment`` is the nonzero witness of the recording factor, which is one on
    ``strata_missing``.
    """
    size = 200_000
    frame, _ = law.sample(size, 20261029)
    stratum = frame["S"].to_numpy() if law.stratified else np.zeros(size, dtype=int)
    recorded = frame["DA"].to_numpy() == 1 if law.missing_treatment else np.ones(size, bool)
    arm = pd.to_numeric(frame["A"], errors="coerce").to_numpy()
    shares = laws.stratum_arm_shares(law)
    for s in range(shares.shape[0]):
        for a in range(shares.shape[1]):
            hit = (stratum == s) & recorded & (arm == a)
            error = np.sqrt(shares[s, a] * (1.0 - shares[s, a]) / size)
            assert abs(hit.mean() - shares[s, a]) < 4.0 * error, (s, a)
    if law.missing_treatment:
        assert shares.sum() < 0.9
    observed = frame["Delta"].to_numpy() == 1
    minority = laws.minority_class_shares(law)
    for (factor, s, a), share in minority.items():
        inside = (stratum == s) & recorded & (arm == a)
        if factor == "treatment":
            classes = (inside, (stratum == s) & recorded & (arm != a))
        elif factor == "outcome_observation":
            classes = (inside & observed, inside & ~observed)
        else:
            classes = ((stratum == s) & recorded, (stratum == s) & ~recorded)
        hit = min(float(np.mean(item)) for item in classes)
        error = np.sqrt(share * (1.0 - share) / size)
        assert abs(hit - share) < 4.0 * error, (factor, s, a)


def test_condition_s_the_control_clears_the_margin() -> None:
    """The control's population bias, from two fits at ``n = 200,000``, is discriminated."""
    assert np.isfinite(properties.CONTROL_BIAS) and np.isfinite(properties.CONTROL_SD)
    probability = laws.discrimination_pass_probability(
        properties.PROPERTY_REPLICATES, properties.control_standardized_bias()
    )
    assert probability >= laws.PASS_PROBABILITY_FLOOR
    # The witness: a control with the positive cells' design bias would not be discriminated.
    assert (
        laws.discrimination_pass_probability(properties.PROPERTY_REPLICATES, laws.DESIGN_BIAS)
        < laws.PASS_PROBABILITY_FLOOR
    )


# ------------------------------------------------------------------------- the reading rule


def _truth_row(**changes: object) -> pd.Series:
    row = {
        "implementation": study.IMPLEMENTATION,
        "scenario": "missing_outcome",
        "estimand": "ate",
        "bias_equivalent": True,
        "se_ratio": 0.88,
        "se_ratio_ci_upper": 0.95,
        "passed": False,
    }
    row.update(changes)
    return pd.Series(row)


def test_the_reading_rule_routes_a_short_se_and_flags_everything_else() -> None:
    assert study.routed_truth_row(_truth_row())
    assert study.routed_truth_row(_truth_row(implementation=study.REFERENCE))
    assert not study.routed_truth_row(_truth_row(bias_equivalent=False))
    assert not study.routed_truth_row(_truth_row(se_ratio=1.02))
    # The floor: a standard error far shorter than the X17 term makes is a defect suspect.
    assert study.routed_truth_row(_truth_row(se_ratio=0.70))
    assert not study.routed_truth_row(_truth_row(se_ratio=0.69, se_ratio_ci_upper=0.75))
    assert not study.routed_truth_row(_truth_row(se_ratio_ci_upper=1.08))
    assert not study.routed_truth_row(_truth_row(implementation="some-other-implementation"))
    rung = pd.Series({"bias_equivalent": True, "se_ratio": 0.86})
    assert study.routed_ladder_row(rung, 0.95)
    assert not study.routed_ladder_row(rung, 1.08)
    assert not study.routed_ladder_row(pd.Series({"bias_equivalent": True, "se_ratio": 0.69}), 0.8)
    assert not study.routed_ladder_row(pd.Series({"bias_equivalent": False, "se_ratio": 0.86}), 0.9)
    performance = pd.DataFrame([_truth_row(), _truth_row(estimand="rr", bias_equivalent=False)])
    equivalence = pd.DataFrame(
        [{"scenario": "strata", "estimand": "ate", "passed": True}],
    )
    columns = ["property", "cell", "passed", "property_passed", "failed_replicates"]
    props = pd.DataFrame(
        [
            ["root_n_and_efficiency", "n_2000", False, False, 0],
            ["robustness_contract", "ate__strata_missing", True, True, 0],
            ["robustness_contract", "par__strata_missing", True, True, 1],
        ],
        columns=columns,
    ).assign(bias_equivalent=[True, True, True], se_ratio=[0.86, 0.9, 0.9])
    suspects = study.defect_suspects(performance, equivalence, props, {"n_2000": 0.95})
    assert suspects == [
        f"truth row {study.IMPLEMENTATION}/missing_outcome/rr is red outside the F19 route",
        "property row robustness_contract/par__strata_missing lost a replication",
    ]
    truth, ladder = study.routed_rows(performance.iloc[:1], props)
    assert truth == [f"{study.IMPLEMENTATION}/missing_outcome/ate"]
    assert ladder == ["root_n_and_efficiency/n_2000"]


def test_the_reading_rule_is_declared_in_the_reference_metadata() -> None:
    assert study.REFERENCE_METADATA["reading_rule"] == study.READING_RULE
    assert study.REFERENCE_METADATA["red_cell_owner"] == "F19"
    assert study.CONFIGURATION["reading_rule"] == study.READING_RULE
    assert re.search(r"below 1\.07", study.READING_RULE)
    assert re.search(r"at least 0\.70", study.READING_RULE)
    assert "Added after the run" in study.READING_RULE


def _committed(name: str) -> pd.DataFrame:
    path = study.STUDY.artifacts / name
    if not path.exists():
        pytest.skip("the registered run has not been committed yet")
    return pd.read_csv(path)


def test_the_pairing_reads_the_design_change() -> None:
    """Every paired row agrees to far less than the shipped design moves it.

    The paired margin of 0.15 SD passes the shipped design as well as this one, and the RMSE
    ratio of the two is about 1 on every row, so neither carries a verdict on the design.  This
    reads the gap itself: ``mean |psi_cleverly - psi_drtmle| / SD`` at most
    ``study.PAIRED_GAP_CEILING`` on every paired row (0.0008 at most, measured).  The
    witness is ``design-gap.csv``: on every reported row, the shipped design sits at least 0.02
    SD from this one on average, so it would fail the same ceiling.
    """
    gaps = study.paired_gaps(_committed("replicates.csv.gz"))
    paired = _committed("equivalence.csv")
    assert set(zip(gaps["scenario"], gaps["estimand"], strict=True)) == set(
        zip(paired["scenario"], paired["estimand"], strict=True)
    )
    assert set(gaps["replicates"]) == {study.PRIMARY_REPLICATES}
    assert gaps["mean_paired_gap"].max() <= study.PAIRED_GAP_CEILING
    design_gap = _committed("design-gap.csv")
    assert len(design_gap) == 17
    assert design_gap["mean_design_gap"].min() >= laws.DESIGN_GAP_FLOOR
    assert design_gap["mean_design_gap"].min() > 4 * study.PAIRED_GAP_CEILING


def test_the_tail_fits_are_rare() -> None:
    """The measured consequence of condition (t): tail fits and the median beside the mean.

    A tail fit has a standard error above twice the empirical SD of its cell.  The ladder's
    smallest rung holds 3 of 800, and the strata cells, whose smallest class is about 59
    nonrespondents, hold 2 of 800 (replications 271 and 605).  They raise the mean SE ratio of
    the ``S = 2`` contrasts 5% to 7% above its median.
    """
    spread = properties.se_ratio_spread(_committed("property-replicates.csv.gz"))
    spread = spread.set_index("cell")
    tails = spread["tail_fits"].to_dict()
    assert tails["n_2000"] == 3
    assert tails["n_4000"] == tails["n_8000"] == 0
    assert tails["ate__strata_missing"] == tails["ate_s2__strata_missing"] == 2
    assert tails["ate_s2__weighted_strata_missing"] == 1
    assert tails["ate__strata_missing_outcome_wrong"] == 5
    elsewhere = spread.loc[
        ~spread.index.str.contains("strata_missing") & ~spread.index.str.startswith("n_")
    ]
    assert elsewhere["tail_fits"].sum() == 0
    s2 = spread.loc["ate_s2__strata_missing"]
    assert s2["mean_se_ratio"] == pytest.approx(0.999, abs=5e-4)
    assert s2["median_se_ratio"] == pytest.approx(0.946, abs=5e-4)
    summary = _committed("properties.csv").set_index("cell")
    assert np.allclose(spread["mean_se_ratio"], summary.loc[spread.index, "se_ratio"], rtol=1e-12)


def test_the_ratio_cells_compare_one_scale() -> None:
    """A ratio cell's SE ratio reads ``psi_n SE(log psi_n)`` against the SD of ``psi_n``.

    The deliberate mutation: without :func:`properties.on_estimate_scale`, the log-scale
    standard error against the SD of ``psi_n`` reads 0.559 for ``rr`` and 0.322 for ``or``.
    """
    assert properties.ratio_cells() == {
        "rr__missing_outcome_crossfit",
        "or__missing_outcome_crossfit",
    }
    rows = _committed("property-replicates.csv.gz")
    published = _committed("properties.csv").set_index("cell")["se_ratio"]
    for cell, unscaled in (
        ("rr__missing_outcome_crossfit", 0.559),
        ("or__missing_outcome_crossfit", 0.322),
    ):
        group = rows.loc[rows["cell"] == cell]
        spread = float(np.std(group["estimate"], ddof=1))
        assert float(group["std_error"].mean()) / spread == pytest.approx(unscaled, abs=5e-4)
        logged = float(np.std(np.log(group["estimate"]), ddof=1))
        assert published[cell] == pytest.approx(group["std_error"].mean() / logged, rel=0.01)
        assert 0.8 < published[cell] < 0.9


def test_the_post_run_gate() -> None:
    """Once the run is committed: no defect suspect, and every routed row is owned by F19."""
    artifacts = study.STUDY.artifacts
    names = ("performance-tests.csv", "equivalence.csv", "properties.csv")
    if not all((artifacts / name).exists() for name in names):
        pytest.skip("the registered run has not been committed yet")
    from tests.studies.evidence import red_cells

    performance = pd.read_csv(artifacts / "performance-tests.csv")
    equivalence = pd.read_csv(artifacts / "equivalence.csv")
    summary = pd.read_csv(artifacts / "properties.csv")
    replicates = pd.read_csv(artifacts / "property-replicates.csv.gz")
    upper = properties.ladder_se_ratio_upper(replicates)
    assert study.defect_suspects(performance, equivalence, summary, upper) == []
    truth, ladder = study.routed_rows(performance, summary)
    owned = set(red_cells.CLAIMS.get(study.RED_CELL_OWNER, ()))
    slug = study.STUDY.slug
    routed = [red_cells.RedKey(slug, "truth", key) for key in truth]
    routed += [red_cells.RedKey(slug, "property", key) for key in ladder]
    for key in routed:
        assert key in owned, key
