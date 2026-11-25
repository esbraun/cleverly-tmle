"""The design of ``projection-methods`` (P1 PR 3 plan rev 2, section 5, and rev 3), without a run.

Every number plan 5.1 computed for law P is recomputed here by quadrature, and the arms, cells,
controls and readings are checked against the plan's tables.  The pilot rule, the point-reading
fallback and the ``pilot_2`` selection are checked on stated ratios.  The rev 3 sizing reads the
committed probe artifacts of :mod:`tests.studies.projection_methods_limit` (and refits one
replicate of each probe against its row).  The summary step is run on synthetic rows of every
declared cell, and the cost gate reads the committed smoke timings.
"""

from __future__ import annotations

import math
import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logit

from tests.canonical.projection_methods import first_run as first_run_check
from tests.studies import point_reading, repeated_sizing
from tests.studies import projection_methods as study
from tests.studies import projection_methods_law as law
from tests.studies import projection_methods_limit as limit
from tests.studies import projection_methods_pilot as pilot
from tests.studies import projection_methods_properties as properties
from tests.studies.evidence.property_verdicts import DIAGNOSTIC_ROLE, ONE_WRONG_FAMILY
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.evidence.seeds import stream_seed

MARGINS = study.STUDY.margins


@cache
def numbers() -> dict[str, Any]:
    return study.design_numbers()


def table() -> dict[str, Any]:
    return numbers()["law"]


@cache
def limit_rows() -> pd.DataFrame:
    return pd.read_csv(limit.LIMIT)


@cache
def probe_rows() -> pd.DataFrame:
    return pd.read_parquet(limit.PROBES)


def reading(block: str, cell: str, quantity: str, term: str | None = None, **where: Any) -> Any:
    """The values of one quantity in the limit artifact (a float when one row matches)."""
    rows = limit_rows()
    chosen = rows.loc[
        (rows["block"] == block) & (rows["cell"] == cell) & (rows["quantity"] == quantity)
    ]
    if term is not None:
        chosen = chosen.loc[chosen["term"] == term]
    for column, value in where.items():
        chosen = chosen.loc[chosen[column] == value]
    values = chosen["value"].astype(float).to_numpy()
    assert len(values), (block, cell, quantity, term, where)
    return float(values[0]) if len(values) == 1 else values


def first_run(cell: str, quantity: str) -> float:
    return float(reading("first_run", cell, quantity))


def pooled_ratio() -> float:
    """The pooled finite-n SE ratio of the ``dr_glm`` intercept at 4,000 (rev 3 R3.1)."""
    cell = "dr_glm__intercept__conservative"
    n1 = first_run(cell, "replicates")
    sd1, se1 = first_run(cell, "empirical_se"), first_run(cell, "mean_std_error")
    probe = "glm4k_probe"
    n2 = reading(probe, probe, "replicates", "(intercept)")
    sd2, se2 = (
        reading(probe, probe, "sd", "(intercept)"),
        reading(probe, probe, "mean_se", "(intercept)"),
    )
    variance = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    return float((n1 * se1 + n2 * se2) / (n1 + n2) / math.sqrt(variance))


# ------------------------------------------------------------------------ seeds and record


def test_the_study_is_declared_and_not_registered() -> None:
    assert study.SLUG not in {record.slug for record in registered()}


def test_the_seeds_are_unique_against_every_registered_study_and_pilot() -> None:
    records = registered()
    taken = {r.seed for r in records} | {r.resampling_seed for r in records}
    pilots = set()
    for path in (ROOT / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^PILOT_SEED\s*=\s*([0-9_]+)", text, flags=re.MULTILINE):
            pilots.add(int(match.group(1).replace("_", "")))
    assert len(pilots) > 5
    for seed in (study.SEED, study.RESAMPLING_SEED):
        assert seed not in taken
        assert seed not in pilots
    assert study.SEED == 20261531
    assert study.SEED != study.RESAMPLING_SEED


def test_each_stream_has_its_own_samples_and_folds_per_group() -> None:
    """Every stream's labels draw their own seeds, and ``property`` routes by the group's stream."""
    groups = [group.label for group in study.GROUPS]
    seen: set[int] = set()
    for sample_label, fold_label in properties.STREAMS.values():
        samples = {stream_seed(study.STUDY, sample_label, g, r) for g in groups for r in range(100)}
        folds = {stream_seed(study.STUDY, fold_label, g, r) for g in groups for r in range(100)}
        assert len(samples) == len(folds) == 100 * len(groups)
        assert not (samples | folds) & seen
        assert not samples & folds
        seen |= samples | folds
    assert set(properties.STREAMS) == {
        "property",
        "property_r3",
        "pilot",
        "pilot_2",
        "smoke",
        "glm4k_probe",
        "ladder_probe",
        "limit_probe",
    }
    redesigned = [g for g in study.GROUPS if g.stream == "property_r3"]
    assert {g.label for g in redesigned} == {
        "dr_glm_4k",
        "dr_slow_4k",
        "dr_slow_16k",
        "dr_slow_64k",
    }
    raw = properties.STREAMS["property"]
    for group in study.GROUPS:
        for r in range(10):
            sample = properties.sample_seed(group.label, r)
            fold = properties.fold_seed(group.label, r)
            assert sample == properties.sample_seed(group.label, r, group.stream)
            assert fold == properties.fold_seed(group.label, r, group.stream)
            first = (
                stream_seed(study.STUDY, raw[0], group.label, r),
                stream_seed(study.STUDY, raw[1], group.label, r),
            )
            assert ((sample, fold) == first) is (group.stream == "property")
            if group.stream != "property":
                assert sample != first[0] and fold != first[1]
            # A smoke or a pilot keeps its own stream for every group.
            assert properties.sample_seed(group.label, r, "smoke") == stream_seed(
                study.STUDY, "smoke_sample", group.label, r
            )


def test_the_draw_and_the_failure_accounting_read_the_routed_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost replicate records the seed it was drawn with (rev 3 R3.6)."""
    drawn: list[int] = []
    monkeypatch.setattr(
        study, "draw_from_seed", lambda label, n, seed: drawn.append(seed) or (None, {})
    )
    properties._draw("dr_glm_4k", 3, "property")
    assert drawn == [properties.sample_seed("dr_glm_4k", 3, "property_r3")]
    captured: dict[str, Any] = {}

    def fit_properties(keys: Any, *args: Any, seed_of: Any, **kwargs: Any) -> list[Any]:
        captured["seeds"] = {key: seed_of(key) for key in keys}
        return []

    monkeypatch.setattr(properties, "fit_properties", fit_properties)
    with pytest.raises((KeyError, RuntimeError)):
        properties.generate_property_rows(budget=2, groups=("dr_glm_4k", "dr_known_4k"))
    seeds = captured["seeds"]
    assert seeds[("dr_glm_4k", 1, "property")] == stream_seed(
        study.STUDY, "property_r3_sample", "dr_glm_4k", 1
    )
    assert seeds[("dr_known_4k", 1, "property")] == stream_seed(
        study.STUDY, "property_sample", "dr_known_4k", 1
    )


def test_the_primary_is_the_dr_known_intercept_at_2400() -> None:
    assert study.STUDY.scenarios == {"dr_known_4k": ("cate_projection[(intercept)]",)}
    cell = study.calibration_cell(study.GROUP_BY_LABEL["dr_known_4k"], "(intercept)")
    summary = pd.DataFrame(
        {"cell": [cell], "covered": [int(0.95 * pilot.REPLICATES)], "replicates": pilot.REPLICATES}
    )
    assert pilot.primary_replicates(summary) <= study.PRIMARY_REPLICATES == 2_400
    assert study.PRIMARY_COVERAGE_FLOOR == 0.90


# --------------------------------------------------------------------------- arms and cells


def test_the_groups_are_the_plans() -> None:
    sizes = {g.label: (g.n, g.replicates, g.method, g.outcome, g.mechanism) for g in study.GROUPS}
    assert sizes == {
        "dr_known_4k": (4_000, study.DECLARED_R, "drtmle", "wrong", "known"),
        "dr_both_4k": (4_000, 2_400, "drtmle", "correct", "glm"),
        "dr_slow_4k": (4_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_slow_16k": (16_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_slow_64k": (64_000, 1_000, "drtmle", "wrong", "slow"),
        "dr_glm_4k": (4_000, 17_600, "drtmle", "wrong", "glm"),
        "ctmle_points_4k": (4_000, 2_400, "ctmle", "correct", "per_arm"),
    }
    assert study.DECLARED_R >= study.BUDGETED_R == 3_600
    streams = {g.label: g.stream for g in study.GROUPS}
    assert streams == {
        "dr_known_4k": "property",
        "dr_both_4k": "property",
        "dr_slow_4k": "property_r3",
        "dr_slow_16k": "property_r3",
        "dr_slow_64k": "property_r3",
        "dr_glm_4k": "property_r3",
        "ctmle_points_4k": "property",
    }
    assert law.LADDER == (4_000, 16_000, 64_000)


def test_the_arms_hold_one_coefficient_each() -> None:
    """Plan 6.1: the intercept and the slope can sit at different SE ratios."""
    assert numbers()["calibration_arms"] == {"__dr_known_intercept": 1, "__dr_known_slope": 1}
    assert study.pilot_groups() == ("dr_known_4k",)


def test_every_cell_is_declared_once_and_the_record_reads_them() -> None:
    cells = [cell for names in study.PROPERTY_CELLS.values() for cell in names]
    assert len(cells) == len(set(cells))
    assert study.STUDY.property_cells == study.PROPERTY_CELLS
    assert study.STUDY.publication_policy == "reporting"
    assert any("n64000" in c for c in cells)
    assert not any("n1000_" in c or c.startswith("dr_slow_n1000") for c in cells)
    assert properties.PROPERTY_CELLS is study.PROPERTY_CELLS
    assert set(study.PROPERTY_CELLS) == {study.CALIBRATION, study.BIAS, ONE_WRONG_FAMILY, "spread"}
    positives = [c for c in study.PROPERTY_CELLS[study.BIAS] if not c.startswith("wrong_q")]
    assert positives == [
        "dr_known_4k_intercept",
        "dr_known_4k_W1",
        "dr_both_4k_intercept",
        "dr_both_4k_W1",
        "ctmle_points_4k_intercept",
        "ctmle_points_4k_W1",
    ]
    # Eight gated bias rows (plan 5.2): six bias cells and the two dr_glm cells' S1 clause.
    one_wrong = study.PROPERTY_CELLS[ONE_WRONG_FAMILY]
    assert len(positives) + sum(c.startswith("dr_glm__") for c in one_wrong) == 8
    assert sorted(one_wrong) == sorted(
        [
            *(f"dr_slow_n{n}__{t}__first_order" for n in law.LADDER for t in ("intercept", "W1")),
            "dr_slow__intercept__growth",
            "dr_slow__W1__growth",
            "dr_glm__intercept__conservative",
            "dr_glm__W1__valid",
        ]
    )
    for cell in one_wrong:
        assert len(cell.split("__")) == 3, cell
    assert study.PROPERTY_CELLS["spread"] == (
        "dr_known_4k_intercept__d_star_reading",
        "dr_known_4k_W1__d_star_reading",
        "dr_both_4k_intercept__plugin_reading",
        "dr_both_4k_W1__plugin_reading",
        "ctmle_points_4k_intercept__plugin_reading",
        "ctmle_points_4k_W1__plugin_reading",
    )
    assert "dr_known_4k_intercept__shrunken_se_control" in study.PROPERTY_CELLS[study.CALIBRATION]


def test_the_truths_name_every_reported_coefficient() -> None:
    for group in study.GROUPS:
        assert {group.name(t) for t in group.terms} == set(study.truth(group))


# ------------------------------------------------------------------------ plan 5.1 numbers


def test_the_law_p_readings_are_the_plans() -> None:
    """The quadrature pins; the SDs and the conservative ratio are the (8)+(9) estimator's."""
    t = table()
    assert t["beta"] == pytest.approx((0.07302, 0.08141), abs=1e-5)
    np.testing.assert_allclose(t["gram"], np.eye(2), atol=1e-10)
    assert t["wrong_limit"] == pytest.approx((-0.2473, 0.6678, -0.4731), abs=1e-4)
    assert t["efficient_sd"] == pytest.approx((0.0128, 0.0136), abs=1e-4)
    assert t["dr_known_sd_8_9"] == pytest.approx((0.0157, 0.0162), abs=1e-4)
    assert t["d_star_sd"] == pytest.approx((0.0163, 0.0171), abs=1e-4)
    assert t["plugin_displacement"] == pytest.approx((5.35, -4.86), abs=0.01)
    assert t["conditioning_norms"][1] == pytest.approx((0.272, 0.252), abs=1e-3)
    assert t["conditioning_norms"][0] == pytest.approx((0.268, 0.223), abs=1e-3)
    assert t["conservative_ratio_8_9"] == pytest.approx((1.189, 1.000), abs=0.001)
    assert t["correlation"] == pytest.approx(-0.029, abs=0.001)
    assert law.quadrature_error() < 1e-8


def test_the_quadrature_is_the_guard_q_estimator() -> None:
    """At n = 200,000 the fit under ``guard=("Q",)`` reproduces the quadrature (rev 3 R3.8)."""
    t = table()
    for replicate in (0, 1):
        ratio = reading(
            "limit", "dr_glm_4k", "ratio", "(intercept)", guard="Q", replicate=replicate
        )
        assert ratio == pytest.approx(t["conservative_ratio_8_9"][0], abs=0.005)
        for j, term in enumerate(study.TERMS):
            for label in limit.LIMIT_LABELS:
                reported = reading(
                    "limit", label, "reported_sd_root_n", term, guard="Q", replicate=replicate
                )
                quadrature = t["dr_known_sd_8_9"][j] * math.sqrt(law.N)
                assert reported == pytest.approx(quadrature, rel=0.015)


def test_equation_10_moves_the_conservative_ratio() -> None:
    """The declared guard's limit lies below the quadrature's, and R follows the pooled 4k ratio."""
    declared = reading("limit", "dr_glm_4k", "ratio", "(intercept)", guard="Q,g")
    assert len(declared) == 2
    assert 1.12 < float(np.mean(declared)) < 1.16
    pooled = pooled_ratio()
    assert pooled == pytest.approx(1.108, abs=0.001)
    planning = min(1.12, math.floor(pooled / 0.005) * 0.005)
    assert study.PLANNING_RATIO["(intercept)"] == pytest.approx(planning, abs=1e-12)
    assert study.PLANNING_RATIO["(intercept)"] == 1.105


def test_the_readings_are_taken_at_the_targeted_limit() -> None:
    """The projection's fluctuation moves the wrong GLM: its population score is nonzero there.

    At the targeted limit ``Qbar*`` the score ``sum_a s(a) E[phi (Qbar_0 - Qbar*)(a)]`` is zero,
    and at the untargeted GLM it is not, so the SDs of plan 5.1 rev 2 (taken at the GLM) were
    the wrong estimator's.
    """
    p = law.law()
    assert p.epsilon == pytest.approx((-0.0795, 0.0700), abs=1e-4)

    def score(q: dict[int, np.ndarray]) -> np.ndarray:
        return np.array(
            [
                sum(p.grid.mean((1.0 if a else -1.0) * f * (p.q0[a] - q[a])) for a in (0, 1))
                for f in p.grid.phi
            ]
        )

    np.testing.assert_allclose(score(p.q_target), 0.0, atol=1e-12)
    assert np.abs(score(p.q_wrong)).min() > 0.01
    # The untargeted readings, which a fit that skipped the fluctuation would have.
    untargeted = p.sd(p.covariance(p.q_wrong, subtract=False))
    assert untargeted == pytest.approx((0.0167, 0.0177), abs=1e-4)


def test_the_ladder_is_the_plans_and_each_other_offset_vanishes_on_a_term() -> None:
    """The ``R_3`` proxy at 4k / 16k / 64k (rev 3 R3.2): signs and floor only."""
    ladder = table()["ladder"]
    np.testing.assert_allclose(
        ladder["product"], [[-0.887, -0.659], [-1.122, -0.835], [-1.418, -1.056]], atol=0.002
    )
    # The near-symmetries of plan 5.1: each single offset barely moves one term.
    assert np.abs(ladder["w2"][:, 1]).max() == pytest.approx(0.035, abs=0.001)
    assert np.abs(ladder["w1w2"][:, 0]).max() == pytest.approx(0.034, abs=0.001)
    assert np.abs(ladder["w1"][:, 1]).max() == pytest.approx(0.017, abs=0.001)


def test_every_declared_witness_clears_its_floor_before_the_pilot() -> None:
    """Plan 6.2: 0.4 SD for the ladder at its smallest rung (4,000 under rev 3) and 3 SD for the
    wrong-Qbar witness."""
    assert np.abs(table()["ladder"]["product"][0]).min() > 0.4
    assert np.abs(table()["plugin_displacement"]).min() > 3.0


def test_the_declared_signs_are_the_conditioning_terms() -> None:
    ladder = table()["ladder"]["product"]
    for j, term in enumerate(("intercept", "W1")):
        signs = set(np.sign(ladder[:, j]).astype(int))
        assert signs == {study.PREDICTED_SIGNS[f"dr_slow__{term}"]}
        # Growth: the magnitude rises along the ladder.
        assert np.all(np.diff(np.abs(ladder[:, j])) > 0)


# ------------------------------------------------------------- the gates and their powers


def planning_rungs() -> np.ndarray:
    """The planning biases ``(rung, term)`` in SD units and their SEs (rev 3 R3.2).

    4k and 16k are the first run's rungs; 64k is the declared-stream ``ladder_probe``.
    """
    values, errors = [], []
    for n in (4_000, 16_000):
        values.append(
            [
                first_run(f"dr_slow_n{n}__{t}__first_order", "standardized_bias")
                for t in ("intercept", "W1")
            ]
        )
        errors.append(
            [1.0 / math.sqrt(first_run(f"dr_slow_n{n}__intercept__first_order", "replicates"))] * 2
        )
    probe = "ladder_probe"
    values.append([reading(probe, probe, "standardized_bias", t) for t in study.TERMS])
    errors.append([reading(probe, probe, "standardized_bias_se", t) for t in study.TERMS])
    return np.array(values), np.array(errors)


def test_every_ladder_rung_and_growth_cell_detects() -> None:
    """Rev 3 R3.2: each rung fails the +-0.25 bound with probability above ``1 - 1e-9``, and each
    growth cell passes with predictive probability above 0.95."""
    values, errors = planning_rungs()
    np.testing.assert_allclose(
        values, [[-0.634, -0.531], [-0.961, -0.800], [-1.401, -0.898]], atol=0.002
    )
    assert errors[-1] == pytest.approx((0.141, 0.118), abs=0.002)
    replicates = study.LADDER_R
    smallest = float(np.abs(values).min())
    assert smallest == pytest.approx(0.531, abs=0.001)
    assert repeated_sizing.bias_discriminated_probability(MARGINS, smallest, replicates) > 1 - 1e-9
    spread = float(np.sqrt(2.0 / replicates))  # two independent standardized biases
    z = float(stats.norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    growth = np.abs(values[-1]) - np.abs(values[0])
    assert growth == pytest.approx((0.766, 0.367), abs=0.002)
    assert spread == pytest.approx(0.045, abs=0.001)
    for j, value in enumerate(growth):
        assert float(stats.norm.sf((z * spread - value) / spread)) > 0.999
        # Predictive: the verdict SE, the 64k probe's SE and the 4k planning SE together.
        total = math.sqrt(spread**2 + errors[-1][j] ** 2 + errors[0][j] ** 2)
        assert float(stats.norm.cdf((value - z * spread) / total)) > 0.95


def test_the_wrong_outcome_witness_fails_bias_on_both_terms() -> None:
    for gap in table()["plugin_displacement"]:
        probability = repeated_sizing.bias_discriminated_probability(
            MARGINS, abs(float(gap)), study.BUDGETED_R
        )
        assert probability > 1 - 1e-9


def test_the_bias_rows_pass_at_the_design_bias() -> None:
    """Plan 5.2: 0.999999 per row at 0.10 and 0.990 at 0.15; over 8 rows 0.99999 and 0.922."""
    at = np.asarray(
        repeated_sizing.bias_equivalent_probability(MARGINS, np.array([0.10, 0.15]), study.BIAS_R)
    )
    assert at[0] > 0.999999
    assert at[1] == pytest.approx(0.990, abs=0.001)
    assert at[0] ** 8 > 0.99999
    assert at[1] ** 8 == pytest.approx(0.922, abs=0.002)


def test_the_shrunken_control_lies_below_the_band() -> None:
    for replicates in (500, study.BUDGETED_R):
        probability = repeated_sizing.calibration_control_probability(
            MARGINS, replicates, study.SHRUNKEN_FACTOR
        )
        assert probability > 0.999999


def test_the_dr_glm_clauses_pass_at_their_planning_figures() -> None:
    """Rev 3 R3.1: the intercept conservative and the slope valid at R = 17,600."""
    replicates = study.GROUP_BY_LABEL["dr_glm_4k"].replicates
    assert replicates == study.DR_GLM_R == 17_600
    assert study.PLANNING_RATIO == {"(intercept)": 1.105, "W1": 1.00}
    for term in study.TERMS:
        coverage = 2.0 * stats.norm.cdf(1.959964 * study.PLANNING_RATIO[term]) - 1.0
        assert study.PLANNING_COVERAGE[term] == pytest.approx(coverage, abs=0.001)
        floor = repeated_sizing.coverage_floor_pass_probability(
            MARGINS, replicates, study.PLANNING_COVERAGE[term]
        )
        assert floor > 0.9999
    assert study.DR_GLM_KINDS == {"(intercept)": "conservative", "W1": "valid"}
    # The spread constant: at least the first run's recorded bootstrap and the probe's own.
    recorded = first_run("dr_glm__intercept__conservative", "spread_constant")
    assert recorded == pytest.approx(0.736, abs=0.001)
    rows = probe_rows()
    probe = rows.loc[(rows["mode"] == "glm4k_probe") & (rows["term"] == "(intercept)")]
    error = (probe["estimate"] - probe["truth"]).to_numpy()
    measured = limit.spread_constant(
        error, probe["plugin_std_error"].to_numpy(), limit.ratio_spread_seed()
    )
    assert measured == pytest.approx(0.681, abs=0.02)
    assert study.RATIO_SPREAD == 0.74
    assert max(recorded, measured) <= study.RATIO_SPREAD
    # R is the smallest multiple of 400 at pass probability 0.999, found by search.
    planning = study.PLANNING_RATIO["(intercept)"]
    found = next(
        r
        for r in range(400, 50_001, 400)
        if study.conservative_pass_probability(planning, r) >= 0.999
    )
    assert found == replicates
    assert study.sanity_pass_probability(study.PLANNING_RATIO["W1"], replicates) > 0.9999
    # The honest pass probability over the planning uncertainty r ~ N(1.108, 0.014).
    grid = np.linspace(1.108 - 6 * 0.014, 1.108 + 6 * 0.014, 2_001)
    weights = stats.norm.pdf(grid, 1.108, 0.014)
    passes = np.array([study.conservative_pass_probability(r, replicates) for r in grid])
    assert float((weights * passes).sum() / weights.sum()) == pytest.approx(0.928, abs=0.01)


def test_the_d_star_reading_is_published_not_gated() -> None:
    """1.039 and 1.057 at the (8)+(9) limit, 1.033 and 1.008 in the first run: no ``D_A`` witness.

    The study therefore has no witness of the mechanism correction; the exact-law curve test of
    ``tests/unit/test_drtmle_projection.py`` is the witness, and the study docstring says so.
    """
    ratios = table()["d_star_ratio"]
    assert ratios == pytest.approx((1.039, 1.057), abs=0.001)
    low, high = MARGINS.calibration_se_ratio
    assert low < ratios[0] < ratios[1] < high
    measured = [
        first_run(f"dr_known_4k_{t}__d_star_reading", "se_ratio") for t in ("intercept", "W1")
    ]
    assert measured == pytest.approx([1.033, 1.008], abs=0.001)
    assert all(low < r < high for r in measured)
    assert "d_star_reading" in study.READING_CLAIMS
    assert "not a D_A witness" in study.READING_CLAIMS["d_star_reading"]
    assert "no witness of the mechanism correction" in " ".join(str(study.__doc__).split())


def test_the_family_passes_at_the_budget_with_correct_intervals() -> None:
    single = float(repeated_sizing.calibration_pass(MARGINS, study.BUDGETED_R, np.array(1.0)))
    assert single ** sum(numbers()["calibration_arms"].values()) > 0.95


# --------------------------------------------------------------------------- the sizing


def _ratios(intercept: float, slope: float, *, one_arm: bool = False) -> dict[str, float]:
    group = study.GROUP_BY_LABEL["dr_known_4k"]
    cells = [study.calibration_cell(group, t) for t in study.TERMS]
    if one_arm:
        cells = [cell.rsplit("__", 1)[0] + "__dr_known" for cell in cells]
    return dict(zip(cells, (intercept, slope), strict=True))


def _correlations(rho: float) -> repeated_sizing.Correlations:
    def correlations(labels: Any) -> np.ndarray:
        k = len(labels)
        return np.full((k, k), rho) + (1.0 - rho) * np.eye(k)

    return correlations


@pytest.mark.parametrize(
    ("intercept", "slope", "expected"),
    [(1.00, 1.00, 3_600), (1.00, 0.98, 5_600), (1.00, 0.97, 10_800)],
)
def test_one_row_per_arm_sizes_the_plans_table(
    intercept: float, slope: float, expected: int
) -> None:
    replicates = repeated_sizing.calibration_replicates(
        MARGINS, _ratios(intercept, slope), pilot.REPLICATES, _correlations(0.002)
    )
    assert replicates == expected


def test_both_rows_in_one_arm_would_leave_no_budget() -> None:
    """Plan 6.1: ratios 1.00 and 0.98 in one arm read a spread of 0.054 and no budget."""
    ratios = _ratios(1.00, 0.98, one_arm=True)
    groups = repeated_sizing.calibration_groups(ratios, pilot.REPLICATES, _correlations(0.002))
    assert repeated_sizing.heterogeneity_bound(groups) == pytest.approx(0.054, abs=0.001)
    with pytest.raises(ValueError, match="no budget"):
        repeated_sizing.calibration_replicates(
            MARGINS, ratios, pilot.REPLICATES, _correlations(0.002)
        )


# ------------------------------------------------------------------- the pilot and fallback


def _pilot_frame(ratios: Any, se: float = 0.01, replicates: int = pilot.REPLICATES) -> pd.DataFrame:
    cells = pilot.positive_cells()
    return pd.DataFrame(
        {
            "cell": list(cells),
            "group": [study.group_of_cell(cell) for cell in cells],
            "arm": [pilot.arm_of(cell) for cell in cells],
            "replicates": replicates,
            "se_ratio": ratios,
            "se_ratio_se": se,
            "covered": int(0.95 * pilot.REPLICATES),
        }
    )


NO_PAIRS = pd.DataFrame(columns=["first", "second", "correlation"])

#: Both pilots' draws: the fallback runs only after ``pilot_2`` (plan 5.0).
BOTH = pilot.REPLICATES + pilot.EXTENSION


def test_the_pilot_rule_sizes_r_by_x32a_on_correct_ratios() -> None:
    assert pilot.declared_replicates(_pilot_frame(1.0), NO_PAIRS) == (study.BUDGETED_R, "x32a")
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the pilot has not run; the declaration commit adds it (plan 5.0)")
    summary = pd.read_csv(pilot.PILOT_CSV)
    computed, _ = pilot.declared_replicates(summary, pd.read_csv(pilot.CORRELATIONS_CSV))
    assert computed == study.DECLARED_R == study.R


def test_inside_the_absorb_band_x32a_finds_a_budget_at_the_declared_pilot() -> None:
    """With one row per arm, no pilot point the fallback may absorb leaves X32a without a budget.

    The fallback therefore answers only a pilot whose ratios a diagnosis has to explain first.
    """
    low, high = point_reading.ABSORB_BAND
    for ratios in ([low, low], [low, high], [high, high]):
        replicates, rule = pilot.declared_replicates(_pilot_frame(ratios), NO_PAIRS)
        assert rule == "x32a"
        assert replicates <= repeated_sizing.MAXIMUM_REPLICATES


def _no_budget(monkeypatch: pytest.MonkeyPatch, message: str = pilot.NO_BUDGET) -> None:
    def refuse(summary: pd.DataFrame, pairs: pd.DataFrame) -> int:
        raise ValueError(message)

    monkeypatch.setattr(pilot, "x32a_replicates", refuse)


def test_the_point_reading_fallback_runs_where_x32a_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_budget(monkeypatch)
    frame = _pilot_frame([0.975, 1.025], replicates=BOTH)
    replicates, rule = pilot.declared_replicates(frame, NO_PAIRS)
    assert rule == "point_reading"
    ratios = dict(zip(frame["cell"], frame["se_ratio"], strict=True))
    errors = dict(zip(frame["cell"], frame["se_ratio_se"], strict=True))
    found = point_reading.point_reading_replicates(
        MARGINS, ratios, errors, target=study.POINT_READING_TARGET
    )
    assert replicates == max(found, study.BUDGETED_R)
    assert found % repeated_sizing.BUDGET_STEP == 0
    assert point_reading.point_reading_pass(MARGINS, ratios, found) >= 0.95
    below = found - repeated_sizing.BUDGET_STEP
    if below > 0:
        assert point_reading.point_reading_pass(MARGINS, ratios, below) < 0.95


def test_the_point_reading_fallback_refuses_a_defect_suspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_budget(monkeypatch)
    frame = _pilot_frame([0.95, 1.05], replicates=BOTH)
    with pytest.raises(ValueError, match="outside"):
        pilot.declared_replicates(frame, NO_PAIRS, diagnosed=pilot.positive_cells())
    tight = _pilot_frame([0.975, 1.025], se=0.002, replicates=BOTH)
    with pytest.raises(ValueError, match="no diagnosis"):
        pilot.declared_replicates(tight, NO_PAIRS)
    diagnosed = pilot.declared_replicates(tight, NO_PAIRS, diagnosed=pilot.positive_cells())
    assert diagnosed[1] == "point_reading"
    assert point_reading.excludes_one(1.02, 0.002)
    assert point_reading.excludes_one(0.98, 0.002)


def test_pilot_2_draws_each_arm_whose_plug_in_pass_is_low() -> None:
    assert pilot.pilot_2_arms(_pilot_frame(1.0)) == ()
    assert pilot.pilot_2_arms(_pilot_frame([1.0, 0.95])) == ("__dr_known_slope",)


def test_the_fallback_answers_only_the_no_budget_case_after_pilot_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing extension stops the declaration, and so does another sizing error."""
    _no_budget(monkeypatch)
    low = _pilot_frame([1.0, 0.945])
    assert pilot.pilot_2_arms(low) == ("__dr_known_slope",)
    with pytest.raises(ValueError, match="only after pilot_2"):
        pilot.declared_replicates(low, NO_PAIRS)
    _no_budget(monkeypatch, "the pilot correlations are not finite")
    with pytest.raises(ValueError) as caught:
        pilot.declared_replicates(_pilot_frame([0.975, 1.025], replicates=BOTH), NO_PAIRS)
    assert pilot.NO_BUDGET not in str(caught.value)
    assert "pilot_2" not in str(caught.value)


# --------------------------------------------------------------------------- the summary


def synthetic_rows(seed: int = 0, replicates: int = 400) -> pd.DataFrame:
    """Rows of every declared cell: correct positives, detecting controls, a growing ladder."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, cells in study.PROPERTY_CELLS.items():
        for cell in cells:
            if cell in study.GROWTH_CELLS:
                continue
            role, shift, se = "positive", 0.0, 1.0
            if family == study.BIAS and cell.startswith("wrong_q"):
                role, shift = "control", 3.0
            if cell.endswith("__shrunken_se_control"):
                role, se = "control", 0.7
            if family == "spread":
                role = DIAGNOSTIC_ROLE
            if cell.endswith("__first_order"):
                size = int(cell.split("__")[0].rsplit("_n", 1)[1])
                role, shift = "control", -0.5 * (size / 1_000) ** (1.0 / 3.0)
            if cell.endswith("__conservative"):
                se = 1.2
            estimate = rng.normal(shift, 1.0, replicates)
            covered = np.abs(estimate) <= 1.96 * se
            rows += [
                {
                    "property": family,
                    "cell": cell,
                    "role": role,
                    "replicate": r,
                    "n": 4_000,
                    "requested_replicates": replicates,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": float(estimate[r]),
                    "std_error": se,
                    "covered": int(covered[r]),
                    "rejected": int(not covered[r]),
                }
                for r in range(replicates)
            ]
    return pd.DataFrame(rows).loc[:, list(properties.REPLICATE_COLUMNS)]


def test_the_summary_reads_every_declared_cell() -> None:
    summary = properties.summarize_properties(synthetic_rows())
    assert set(summary["cell"]) == {c for cells in study.PROPERTY_CELLS.values() for c in cells}
    diagnostic = summary.loc[summary["role"] == DIAGNOSTIC_ROLE]
    assert set(diagnostic["cell"]) == set(study.PROPERTY_CELLS["spread"])
    assert diagnostic["passed"].all()
    controls = summary.loc[summary["role"] == "control"]
    assert controls["passed"].all(), controls.loc[~controls["passed"].astype(bool), "cell"].tolist()
    conservative = summary.loc[summary["cell"] == "dr_glm__intercept__conservative"]
    assert bool(conservative["passed"].iloc[0])


# ---------------------------------------------------------------------- the cost and files


def test_the_smoke_cost_stays_inside_the_cost_gate() -> None:
    timings = pd.read_csv(study.SMOKE_TIMINGS)
    assert set(timings["group"]) == {group.label for group in study.GROUPS}
    for row in timings.itertuples(index=False):
        declared = study.GROUP_BY_LABEL[row.group].seconds
        assert row.seconds_per_fit <= study.COST_GATE * declared, row.group
    assert numbers()["cpu_hours"] == pytest.approx(study.run_cpu_hours())
    # Rev 3: dr_glm_4k at 17,600 and the ladder at 4k / 16k / 64k, the 64k seconds the smoke's.
    assert study.LADDER_SECONDS[64_000] == pytest.approx(
        float(timings.set_index("group").loc["dr_slow_64k", "seconds_per_fit"]), abs=0.01
    )
    assert study.run_cpu_hours() == pytest.approx(6.19, abs=0.05)  # at the declared R = 5,600
    assert study.run_cpu_hours(10_800) == pytest.approx(6.98, abs=0.05)
    assert study.pilot_cpu_hours() == pytest.approx(0.49, abs=0.02)


def test_the_declaration_names_every_file_a_run_writes() -> None:
    from tests.canonical.declared_run import RECORD_FILES
    from tests.canonical.regenerate import ARTIFACT_NAMES, PROPERTY_FAILURE_FILES

    assert set(study.RUN_FILES) == {*ARTIFACT_NAMES, *RECORD_FILES, *study.STUDY.property_artifacts}
    assert set(study.CONDITIONAL_RUN_FILES) == set(PROPERTY_FAILURE_FILES)
    assert (Path(__file__).parents[1] / "canonical/projection_methods/regenerate.py").exists()


# ------------------------------------------------------------------------- the fitted pieces

#: The status each group's coefficients must carry: the interval ships only with the mechanism
#: declared (plan 1.5, 5.2).
STATUSES = {
    "dr_known_4k": "influence_curve",
    "dr_both_4k": "reduced_regression_plugin",
    "dr_slow_4k": "reduced_regression_plugin",
    "dr_glm_4k": "reduced_regression_plugin",
    "ctmle_points_4k": "generated_design_plugin",
}


@pytest.mark.parametrize("label", sorted(STATUSES))
def test_each_group_fits_with_its_declared_status(label: str) -> None:
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(label, 1_000, 3)
    result = study.fit(group, frame, 3)
    assert {e.inference for e in result.estimates.values()} == {STATUSES[label]}
    assert list(result.estimates) == [group.name(t) for t in group.terms]


@cache
def _known_fit() -> Any:
    group = study.GROUP_BY_LABEL["dr_known_4k"]
    frame, _ = study.draw_from_seed(group.label, 2_000, 7)
    return frame, study.fit(group, frame, 7)


def test_the_d_star_reading_adds_back_a_live_correction() -> None:
    """``D*`` alone differs from the reported curve by the mechanism correction, row by row."""
    _, result = _known_fit()
    reported = np.column_stack(
        [np.asarray(result.estimates[f"cate_projection[{t}]"].influence_curve) for t in study.TERMS]
    )
    added = study.d_star_curves(result) - reported
    assert np.abs(added).max(axis=0).min() > 1e-3
    np.testing.assert_allclose(study.d_star_curves(result).mean(axis=0), 0.0, atol=1e-6)
    ses = study.d_star_std_errors(result)
    assert np.all(ses > 0) and np.all(np.isfinite(ses))


def test_the_wrong_q_witness_reads_the_initial_wrong_regression() -> None:
    frame, result = _known_fit()
    learner = study.outcome_learner(study.GROUP_BY_LABEL["dr_known_4k"])
    learner.fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"].to_numpy())
    arms = {
        a: learner.predict_proba(frame[["W1", "W2"]].assign(A=a)[["A", "W1", "W2"]].to_numpy())[
            :, 1
        ]
        for a in (0.0, 1.0)
    }
    phi = np.column_stack([np.ones(len(frame)), frame["W1"].to_numpy()])
    expected = np.linalg.lstsq(phi, arms[1.0] - arms[0.0], rcond=None)[0]
    np.testing.assert_allclose(study.plugin_initial(result), expected, atol=1e-8)


def test_the_slow_mechanism_is_the_glm_plus_the_declared_offset() -> None:
    frame, _ = study.draw_from_seed("dr_slow_4k", 1_000, 5)
    design = frame[["W1", "W2"]].to_numpy()
    slow = study.SlowMechanism(4_000).fit(design, frame["A"].to_numpy())
    glm = study.law_l.OffsetLogistic(columns=(0, 1)).fit(design, frame["A"].to_numpy())
    offset = 4_000 ** (-1.0 / 3.0) * design[:, 1] * (1.0 + design[:, 0])
    np.testing.assert_allclose(
        logit(slow.predict_proba(design)[:, 1]),
        logit(glm.predict_proba(design)[:, 1]) + offset,
        atol=1e-9,
    )
    assert study.treatment_learner(study.GROUP_BY_LABEL["dr_slow_16k"]).n == 16_000


def test_the_correct_outcome_learner_holds_law_p() -> None:
    frame, _ = study.draw_from_seed("dr_both_4k", 20_000, 9)
    learner = study.CorrectOutcome().fit(frame[["A", "W1", "W2"]].to_numpy(), frame["Y"])
    truth = law.qbar0(frame["A"], frame["W1"], frame["W2"])
    fitted = learner.predict_proba(frame[["A", "W1", "W2"]].to_numpy())[:, 1]
    assert float(np.max(np.abs(fitted - truth))) < 0.05


# ------------------------------------------------------------------ the rev 3 probes and check


@pytest.mark.parametrize("mode", ["glm4k_probe", "ladder_probe", "slow_1k_x2"])
def test_a_probe_replicate_refits_to_its_committed_row(mode: str) -> None:
    """The probes are the study's own draw and fit on their streams (rev 3 R3.8)."""
    rows = probe_rows()
    committed = rows.loc[(rows["mode"] == mode) & (rows["replicate"] == 0)].reset_index(drop=True)
    refit = pd.DataFrame(limit.probe_rows(mode, 0)).loc[:, list(limit.PROBE_COLUMNS)]
    pd.testing.assert_frame_equal(refit, committed.loc[:, list(limit.PROBE_COLUMNS)], rtol=1e-9)


def test_the_probe_artifacts_hold_every_declared_probe() -> None:
    counts = probe_rows().groupby("mode")["replicate"].nunique().to_dict()
    assert counts == limit.probe_modes()
    fits = limit_rows().loc[limit_rows()["block"] == "limit"]
    assert set(zip(fits["guard"], fits["replicate"], strict=True)) == {
        (limit.guard_name(guard), r) for guard, r in limit.LIMIT_FITS
    }
    commits = set(limit_rows().loc[limit_rows()["block"] == "first_run", "commit"])
    assert commits == {first_run_check.FIRST_RUN}


def _first_run_fixture() -> dict[str, pd.DataFrame]:
    cell = study.bias_cell(study.GROUP_BY_LABEL["dr_known_4k"], "(intercept)")
    return {
        "replicates.parquet": pd.DataFrame(
            {
                "implementation": "x",
                "scenario": "dr_known_4k",
                "replicate": [0, 1],
                "estimand": "e",
                "estimate": [0.1, 0.2],
                "std_error": [0.01, 0.02],
                "ci_lower": [0.08, 0.16],
                "ci_upper": [0.12, 0.24],
                "covered": [1, 0],
            }
        ),
        "property-replicates.parquet": pd.DataFrame(
            {
                "property": ["bias", "bias", "one_wrong_inference"],
                "cell": [cell, cell, "dr_glm__W1__valid"],
                "replicate": [0, 1, 0],
                "estimate": [0.3, 0.4, 9.0],
                "std_error": [0.03, 0.04, 9.0],
                "covered": [1, 1, 0],
                "rejected": [0, 0, 1],
            }
        ),
        "summary.csv": pd.DataFrame(
            {
                "implementation": ["x"],
                "scenario": ["dr_known_4k"],
                "estimand": ["e"],
                "mean_estimate": [0.15],
                "empirical_se": [0.07],
                "mean_std_error": [0.015],
                "se_ratio": [0.2],
                "coverage": [0.5],
            }
        ),
        "properties.csv": pd.DataFrame(
            {
                "property": ["bias"],
                "cell": [cell],
                **{c: [0.5] for c in first_run_check.FRAMES["properties.csv"][1]},
                "passed": [True],
            }
        ),
    }


def _check(second: dict[str, pd.DataFrame]) -> list[str]:
    return first_run_check.check(
        second["replicates.parquet"],
        second["property-replicates.parquet"],
        summary=second["summary.csv"],
        property_summary=second["properties.csv"],
        first=_first_run_fixture(),
        stacks=({"numpy": "2.5.5"}, {"numpy": "2.5.4"}),
    )


def test_the_first_run_check_passes_equal_rows_and_notes_drift_within_tolerance() -> None:
    lines = _check(_first_run_fixture())
    assert "exactly" in lines[-1]
    drifted = _first_run_fixture()
    drifted["property-replicates.parquet"]["estimate"] *= 1.0 + 1e-12
    # A re-declared group's rows never enter the comparison.
    drifted["property-replicates.parquet"].loc[2, "estimate"] = -1.0
    lines = _check(drifted)
    assert "within relative" in lines[-1]
    assert "2.5.5 -> 2.5.4" in lines[-1]


@pytest.mark.parametrize(
    ("name", "column", "change"),
    [
        ("property-replicates.parquet", "estimate", 1e-6),
        ("replicates.parquet", "std_error", 1e-6),
        ("properties.csv", "passed", None),
        ("replicates.parquet", "covered", None),
    ],
)
def test_the_first_run_check_stops_a_real_change_or_a_flipped_verdict(
    name: str, column: str, change: float | None
) -> None:
    second = _first_run_fixture()
    frame = second[name]
    if change is None:
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = not bool(frame.loc[0, column])
    else:
        frame.loc[0, column] *= 1.0 + change
    with pytest.raises(RuntimeError, match="do not reproduce"):
        _check(second)
    missing = _first_run_fixture()
    missing["property-replicates.parquet"] = missing["property-replicates.parquet"].iloc[1:]
    with pytest.raises(RuntimeError, match="one run only"):
        _check(missing)


def test_the_study_hands_the_check_to_the_shared_driver() -> None:
    import inspect

    assert inspect.signature(study.check_rows).parameters.keys() == {
        "rows",
        "property_rows",
        "summary",
        "property_summary",
    }
    assert "tests/canonical/projection_methods/first_run.py" in study.STUDY.modules
    assert "tests/studies/projection_methods_limit.py" in study.STUDY.modules
    assert first_run_check.unchanged_groups() == ("dr_known_4k", "dr_both_4k", "ctmle_points_4k")


def test_the_study_page_quotes_the_first_run_and_the_probes() -> None:
    """ "The re-declared cells" quotes the committed artifacts at the precision it prints.

    The study is not registered, so :mod:`tests.studies.evidence.claims` does not resolve its
    page yet; this test ties each printed figure to its row instead.
    """
    page = " ".join((ROOT / study.STUDY.document).read_text(encoding="utf-8").split())
    assert "## The re-declared cells" in page
    glm = "dr_glm__intercept__conservative"
    one_k = "dr_slow_n1000__{}__first_order"

    def limit_ratio(guard: str, replicate: int) -> float:
        return reading(
            "limit", "dr_glm_4k", "ratio", "(intercept)", guard=guard, replicate=replicate
        )

    def slow(scale: int, term: str) -> float:
        mode = f"slow_1k_x{scale}"
        return reading(mode, mode, "standardized_bias", term)

    quoted = [
        (
            f"{first_run(glm, 'se_ratio'):.4f} ({first_run(glm, 'se_ratio_ci_lower'):.4f} to "
            f"{first_run(glm, 'se_ratio_ci_upper'):.4f})"
        ),
        (
            f"{first_run(glm, 'coverage'):.4f} ({first_run(glm, 'coverage_ci_lower'):.3f} to "
            f"{first_run(glm, 'coverage_ci_upper'):.3f})"
        ),
        f"| bias | {first_run(glm, 'bias'):.5f} |",
        f"| {table()['conservative_ratio_8_9'][0]:.3f} |",
        f"| {limit_ratio('Q', 0):.3f} and {limit_ratio('Q', 1):.3f} |",
        f"| {limit_ratio('Q,g', 0):.3f} and {limit_ratio('Q,g', 1):.3f} |",
        f"| {pooled_ratio():.3f} |",
        (
            f"| {first_run(one_k.format('intercept'), 'standardized_bias'):.2f} | "
            f"{first_run(one_k.format('W1'), 'standardized_bias'):.2f} |"
        ),
        *(
            f"| {slow(c, '(intercept)'):+.2f} | {slow(c, 'W1'):.2f} |".replace("| +-", "| -")
            for c in (1, 2, 3)
        ),
        f"| {slow(0, '(intercept)'):+.2f} | {slow(0, 'W1'):.2f} |",
        f"{reading('slow_1k_x0', 'slow_1k_x0', 'bias', '(intercept)'):+.4f} (SE "
        f"{reading('slow_1k_x0', 'slow_1k_x0', 'bias_se', '(intercept)'):.4f})",
        f"R = {study.DR_GLM_R:,} replicates",
    ]
    for text in quoted:
        assert text in page, text
