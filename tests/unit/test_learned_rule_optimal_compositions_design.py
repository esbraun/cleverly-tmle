"""The declared constants of ``learned-rule-optimal-compositions``, recomputed without a run.

Each number here is fixed before any registered run. The truths and bounds are recomputed by
quadrature through the study's law module, and every truth again by an independent quadrature
written here from the laws' formulas; the target controls' power, the stratum-fold
feasibility and the counts are computed from declared inputs and from the committed sizing
pilot (``tests/studies/learned_rule_optimal_compositions_pilot.csv``); the declared seeds are only
computed, never drawn from.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import beta, binom, norm

from cleverly._support_rows import point_row
from tests.canonical import learned_rule_run
from tests.studies import _learned_rule_intermediate_law as zlaw
from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies import learned_rule_optimal_compositions as study
from tests.studies import learned_rule_optimal_compositions_pilot as pilot
from tests.studies import learned_rule_optimal_compositions_properties as properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed
from tests.unit.test_learned_rule_optimal_value_design import (
    _x11_modules,
    bias_pass,
    ceiling_pass,
    floor_pass,
)

pytestmark = pytest.mark.xdist_group("learned_rule_optimal_compositions_design")

base = law.base


# ----------------------------------------------------------------------------- the laws


@pytest.mark.parametrize("label", list(study.LIMITS))
def test_each_labels_limit_is_the_declared_one(label: str) -> None:
    limit = law.label_limits()[label]
    assert limit["truth"] == pytest.approx(study.LIMITS[label]["truth"], abs=1e-6)
    assert limit["bound"] == pytest.approx(study.LIMITS[label]["bound"], abs=1e-6)


def test_the_marginal_limits_are_the_optimal_value_studys() -> None:
    """The marginal and in-sample labels read ``learned-rule-optimal-value``'s gated law. That
    study's bound is the trapezoid rule's, which smears the jump of the variance integrand at the
    rule's threshold (1e-5 relative); this study's is split there (design review m1)."""
    from tests.studies import learned_rule_optimal_value as optimal

    for label in ("strata_all", "in_sample"):
        assert study.LIMITS[label]["truth"] == optimal.LIMITS["non_exceptional"]["truth"]
        assert study.LIMITS[label]["bound"] == pytest.approx(
            optimal.LIMITS["non_exceptional"]["bound"], abs=1e-5
        )
    assert study.LIMITS["intermediate_z0"]["truth"] == pytest.approx(
        study.LIMITS["strata_all"]["truth"], abs=1e-6
    )


def test_the_share_mixture_of_the_strata_is_the_marginal_value() -> None:
    limits = law.label_limits()
    mixture = 0.5 * limits["strata_s0"]["truth"] + 0.5 * limits["strata_s1"]["truth"]
    assert mixture == pytest.approx(limits["strata_all"]["truth"], abs=1e-12)


def test_the_weight_moves_the_rule_and_the_mechanism() -> None:
    """Design review M3: under ``w = 1 + A Y`` the weighted rule treats where ``d0`` does not
    (W1 in (-0.793, -0.1), 34.7% of the population), the weighted mechanism is not ``g0``, and
    the target sits 0.0277 above the weighted value of the unweighted rule."""
    w1 = base.GRID
    for w2 in base.W2_LEVELS:
        blip_w = law.weighted_qbar(1.0, w1, w2) - law.weighted_qbar(0.0, w1, w2)
        moved = (w1 > law.weighted_threshold()) & (w1 <= law.THRESHOLD)
        assert (blip_w[moved] > 0).all() and (blip_w[w1 < law.weighted_threshold()] < 0).all()
        assert (law.weighted_mechanism(w1, w2) > base.g0(w1, w2) + 0.02).all()
    assert (law.THRESHOLD - law.weighted_threshold()) / 2 == pytest.approx(0.347, abs=5e-4)
    gap = study.LIMITS["weighted"]["truth"] - study.UNWEIGHTED_RULE_TRUTH
    assert gap == pytest.approx(0.0277, abs=5e-5)
    assert law.unweighted_rule_truth() == pytest.approx(study.UNWEIGHTED_RULE_TRUTH, abs=1e-6)


def test_the_weighted_mechanism_is_the_weighted_populations() -> None:
    """``g_w(1 | W) = E[w 1{A = 1} | W] / E[w | W]``, summed over the cells of ``(A, Y)``."""
    w1 = base.GRID
    for w2 in base.W2_LEVELS:
        treated = base.g0(w1, w2)
        q1 = base.qbar0(1.0, w1, w2, law.LAW)
        q0 = base.qbar0(0.0, w1, w2, law.LAW)
        cells = {
            (a, y): (treated if a else 1 - treated) * (q if y else 1 - q) * law.weight(a, y)
            for a, q in ((1.0, q1), (0.0, q0))
            for y in (0.0, 1.0)
        }
        mass = sum(cells.values())
        np.testing.assert_allclose(
            law.weighted_mechanism(w1, w2), (cells[1.0, 0.0] + cells[1.0, 1.0]) / mass, rtol=1e-12
        )
        np.testing.assert_allclose(
            law.weighted_qbar(1.0, w1, w2),
            cells[1.0, 1.0] / (cells[1.0, 0.0] + cells[1.0, 1.0]),
            rtol=1e-12,
        )


def _independent_truths() -> dict[str, float]:
    """Every label's truth from the laws' formulas, with :func:`scipy.integrate.quad` and the
    weighted rule's threshold found by root search, sharing no code with the law module."""

    def expit(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def q(a: float, w1: float, w2: float, z: float = 0.0) -> float:
        return expit(0.2 + 0.5 * w1 - 0.3 * w2 + 0.3 * z + a * (0.1 + w1 - 0.6 * z))

    def g(w1: float, w2: float) -> float:
        return expit(0.3 * w1 - 0.2 * w2)

    def mean(f: Callable[[float, float], float], cut: float, levels: tuple[float, ...]) -> float:
        out = 0.0
        for w2 in levels:
            for low, high in ((-1.0, cut), (cut, 1.0)):
                out += quad(lambda w1, w2=w2: f(w1, w2), low, high, epsabs=1e-13, epsrel=1e-12)[0]
        return out * 0.5 / len(levels)

    def ew(w1: float, w2: float) -> float:
        return 1.0 + g(w1, w2) * q(1, w1, w2)

    def qw(a: float, w1: float, w2: float) -> float:
        return (1.0 + a) * q(a, w1, w2) / (1.0 + a * q(a, w1, w2))

    cut_w = brentq(lambda w1: qw(1, w1, 0.0) - qw(0, w1, 0.0), -1.0, 1.0, xtol=1e-14)
    both = (0.0, 1.0)

    def at(cut: float, f: Callable[[float, float, float], float]) -> Callable[..., float]:
        return lambda w1, w2: f(float(w1 > cut), w1, w2)

    value = mean(at(-0.1, q), -0.1, both)
    out = {
        "weighted": mean(at(cut_w, lambda a, w1, w2: ew(w1, w2) * qw(a, w1, w2)), cut_w, both)
        / mean(ew, cut_w, both),
        "strata_all": value,
        "in_sample": value,
        "strata_s0": mean(at(-0.1, q), -0.1, (0.0,)),
        "strata_s1": mean(at(-0.1, q), -0.1, (1.0,)),
    }
    for arm in (1, 0):
        out[f"contrast_{arm}"] = value - mean(lambda w1, w2, a=arm: q(a, w1, w2), -0.1, both)
    for z in (0.0, 1.0):
        cut = -0.1 + 0.6 * z
        out[f"intermediate_z{int(z)}"] = mean(
            lambda w1, w2, z=z, cut=cut: q(float(w1 > cut), w1, w2, z), cut, both
        )
    out["unweighted_rule"] = mean(
        at(-0.1, lambda a, w1, w2: ew(w1, w2) * qw(a, w1, w2)), -0.1, both
    ) / mean(ew, -0.1, both)
    return out


def test_every_truth_agrees_with_an_independent_quadrature() -> None:
    """Design review m6: the declared truths against a second quadrature written here."""
    truths = _independent_truths()
    for label, limit in study.LIMITS.items():
        assert limit["truth"] == pytest.approx(truths[label], abs=1e-6), label
        assert law.truth(label) == pytest.approx(truths[label], abs=1e-10), label
    assert pytest.approx(truths["unweighted_rule"], abs=1e-6) == study.UNWEIGHTED_RULE_TRUTH


def test_the_contrast_bounds_agree_with_the_trapezoid() -> None:
    """The split quadrature of the contrast curves against the RM30 trapezoid of
    ``_learned_rule_contrast_law.curve_variances``, which smears the variance integrand's jump at
    the rule threshold (2e-4 relative, design review m1)."""
    d0 = law.optimal.optimal_rule(law.optimal.law_qbar(law.LAW))
    for arm in (1, 0):
        _, difference = law.contrast.curve_variances(d0, float(arm), law.LAW)
        assert study.LIMITS[f"contrast_{arm}"]["bound"] == pytest.approx(
            np.sqrt(difference), rel=2e-4
        )


def test_the_intermediate_levels_have_no_tie_set() -> None:
    """The level blips ``0.1 + W1 - 0.6 z`` are linear in ``W1`` with slope 1: no tie mass and a
    bounded density at 0 (margin exponent 1), as the plan requires of the gated cell."""
    for z in zlaw.LEVELS:
        # Linear with slope 1: at most one grid point is a root, a set of measure zero.
        assert int(np.sum(zlaw.blip(z, base.GRID) == 0.0)) <= 1
        assert float(np.ptp(np.diff(zlaw.blip(z, base.GRID)))) < 1e-12
        gap = np.abs(zlaw.blip(z, base.GRID))
        assert float(np.mean(gap <= 0.01)) / 0.01 < 10.0


def test_the_cells() -> None:
    labels = (
        "weighted",
        "strata_all",
        "strata_s0",
        "strata_s1",
        "contrast_1",
        "contrast_0",
        "in_sample",
        "intermediate_z0",
        "intermediate_z1",
    )
    assert labels == study.CALIBRATION_LABELS
    calibration = []
    for label in labels:
        calibration += [
            f"{label}__{kind}" for kind in ("declared", "shrunken_se_control", "noise_control")
        ]
        if label in ("strata_s0", "strata_s1"):
            calibration.append(f"{label}__unnormalised_curve_control")
    assert {
        "interval_calibration": tuple(calibration),
        "weighted_target_control": (
            "weighted__target_gap_control",
            "weighted__unweighted_rule_control",
        ),
    } == study.PROPERTY_CELLS
    assert set(study.SCENARIOS) == set(law.SETTINGS)
    assert sum(len(names) for names in study.SCENARIOS.values()) == 9
    assert properties.SHRUNKEN_SE_FACTOR == 0.80
    assert not study.STUDY.truth_varies_by_replicate
    assert not study.STUDY.calibration_efficiency_ratio
    assert study.STUDY.publication_policy == "gated"
    assert study.PRIMARY_N == properties.CALIBRATION_N == study.STUDY.n == 2_000
    assert study.STUDY.reference is None


def test_the_module_list_holds_every_study_specific_import() -> None:
    from tests.unit.test_study_provenance import _required_from

    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules), sorted(required - set(study.STUDY.modules))
    assert len(set(study.STUDY.modules)) == len(study.STUDY.modules)


def test_the_support_rows_cover_every_fitted_key() -> None:
    """Design review M1: each declared row, built as the registering change will build it,
    covers every fitted key of its setting, both intermediate levels included, with options and
    qualifiers equal, not merely contained (T55)."""
    assert set(study.SUPPORT_ROWS) == set(law.SETTINGS)
    for setting, declared in study.SUPPORT_ROWS.items():
        row = point_row(f"declared.{setting}", (study.STUDY.slug,), **declared)
        results = law.fit(law.draw(setting, 400, 1), setting, 3)
        fitted = [results[z] for z in zlaw.LEVELS] if setting == "intermediate" else [results]
        for result in fitted:
            key = result.config.support_key
            assert key.is_covered_by(row.key), (setting, key.differing_axes(row.key))
            assert key.options == row.key.options, setting
            assert key.qualifiers == row.key.qualifiers, setting
    # The deliberate mutation: the in-sample row without its learner families covers nothing.
    bare = {k: v for k, v in study.SUPPORT_ROWS["in_sample"].items() if k != "options"}
    in_sample = law.fit(law.draw("in_sample", 400, 1), "in_sample", 3)
    assert in_sample.config.support_key.differing_axes(
        point_row("bare", (study.STUDY.slug,), **bare).key
    ) == ("options",)


def test_the_weighted_row_states_its_limit() -> None:
    (limit,) = study.SUPPORT_ROWS["weighted"]["limits"]
    assert "weighted population's own treatment mechanism" in limit
    assert "Fix-16" in limit


def test_the_declared_gates_are_read() -> None:
    families = set().union(*(rule.families for rule in properties.PROPERTY_RULES))
    assert families == set(study.PROPERTY_CELLS)
    assert study.STUDY.margins.calibration_coverage[0] == 0.92


# ------------------------------------------------------------------------- the controls


def _control_truth(control: str) -> float:
    return {
        "target_gap_control": study.LIMITS["strata_all"]["truth"],
        "unweighted_rule_control": study.UNWEIGHTED_RULE_TRUTH,
    }[control]


def target_control_power(
    replicates: int, control: str, se_ratio: float = 1.07, bias: float = 0.0
) -> float:
    """The probability that the control's exact 99% coverage interval lies below 0.92.

    The control's interval is centred on the weighted target, read against ``control``'s truth
    below it, with SE ``se_ratio`` times the efficient one; the largest SE the calibration band
    admits (1.07) gives the most coverage, so the least power. ``bias`` (in SEs, negative toward
    the control's truth) stresses it further.
    """
    gap = study.LIMITS["weighted"]["truth"] - _control_truth(control)
    z = gap / (study.LIMITS["weighted"]["bound"] / np.sqrt(study.PRIMARY_N)) + bias
    critical = base.CRITICAL * se_ratio
    covered = norm.cdf(critical - z) - norm.cdf(-critical - z)
    hits = np.arange(replicates + 1)
    upper = np.where(hits < replicates, beta.ppf(0.995, hits + 1, replicates - hits), 1.0)
    floor = study.STUDY.margins.calibration_coverage[0]
    return float(np.sum(binom.pmf(hits, replicates, covered) * (upper < floor)))


@pytest.mark.parametrize(
    ("control", "gap_in_se"), [("target_gap_control", 8.36), ("unweighted_rule_control", 2.05)]
)
def test_each_target_control_has_power(control: str, gap_in_se: float) -> None:
    """Plan section 5: each control's gap at n = 2,000 and its power at least 0.90, also with
    the regret's bias of half an SE toward the control's truth."""
    gap = study.LIMITS["weighted"]["truth"] - _control_truth(control)
    se = study.LIMITS["weighted"]["bound"] / np.sqrt(study.PRIMARY_N)
    assert gap / se == pytest.approx(gap_in_se, abs=0.01)
    assert target_control_power(properties.TARGET_REPLICATES, control, bias=-0.5) >= 0.90
    assert properties.TARGET_REPLICATES == 1_200


def test_the_unnormalised_control_falls_below_the_band() -> None:
    """The stratum curve without its fold share has SE about ``P(W2 = s) = 1/2`` times the
    reported one, so its SE ratio is at most ``0.5 * 1.07 + 0.05`` at the band's largest
    positive, far below the floor 0.93, and it fails at any count (a fold share deviates from
    1/2 by about ``sqrt(0.25 / 200) = 0.035``)."""
    share_bound = 0.5 + 4 * np.sqrt(0.25 / (study.PRIMARY_N / base.N_FOLDS))
    assert share_bound * 1.07 < study.STUDY.margins.calibration_se_ratio[0] - 0.2


def test_an_empty_stratum_fold_arm_is_negligible() -> None:
    """Plan review finding 10: a stratum-fold arm with no row refuses with a ``DataError``.

    A union bound over 10 validation folds, 2 strata and 2 arms: each stratum-fold holds
    ``Binomial(200, 1/2)`` rows, and each row is in an arm with probability at least
    ``min g0 = expit(-0.5)``. The probability of an empty arm in one draw is below ``10^-6``.
    """
    smallest = float(min(base.g0(base.GRID, w2).min() for w2 in base.W2_LEVELS))
    smallest = min(smallest, 1.0 - float(max(base.g0(base.GRID, w2).max() for w2 in (0.0, 1.0))))
    rows = np.arange(201)
    empty = float(np.sum(binom.pmf(rows, 200, 0.5) * (1.0 - smallest) ** rows))
    assert 10 * 2 * 2 * empty < 1e-6


# --------------------------------------------------------------------------- the sizing


def _pilot() -> pd.DataFrame:
    return pd.read_csv(pilot.PILOT_CSV).set_index("label")


def test_the_pilot_covers_every_label_at_its_declared_size() -> None:
    frame = _pilot()
    assert list(frame.index) == list(study.CALIBRATION_LABELS)
    assert (frame["replicates"] == pilot.PILOT_DRAWS).all()
    assert (frame["n"] == study.PRIMARY_N).all()
    assert int(frame["solver_warnings"].max()) == 0


def test_the_pilot_screens_every_label_inside_the_band() -> None:
    """The arm-pooled rule's per-arm screen: each label's pilot SE ratio within
    ``2.576 sigma_P + 0.01`` of 1, or the label is diagnosed alone before ``R`` is set."""
    reading = tools.replicates_from_pilot(_pilot()["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    assert all(reading["screened"]), reading
    assert not reading["suspect"], reading


def _coverage_leg(coverage: float, replicates: int, pilot_draws: int | None) -> float:
    """The coverage leg: the exact 99% interval inside ``[0.92, 0.98]``, predictive over the
    pilot coverage (``pilot_draws=None`` reads it as known)."""
    low, high = study.STUDY.margins.calibration_coverage
    return floor_pass(coverage, replicates, low, pilot_draws) - (
        1.0 - ceiling_pass(coverage, replicates, high, pilot_draws)
    )


def calibration_joint(replicates: int, *, stressed: bool = False) -> float:
    """The joint pass probability of the nine positives at ``replicates`` (design review M2).

    The arm-pooled SE-ratio family (pooled mean and ``tau`` of
    :func:`~tests.studies.mtp_study_tools.replicates_from_pilot`) times the product of the nine
    coverage legs. ``stressed`` is the T46-style stress: the pooled mean one pooled pilot SE
    further from 1 and each coverage one pilot SE toward its nearer edge, both read as known.
    """
    frame = _pilot()
    reading = tools.replicates_from_pilot(frame["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    arms = len(frame)
    sigma = tools.pilot_sigma(pilot.PILOT_DRAWS)
    low, high = study.STUDY.margins.calibration_coverage
    if not stressed:
        _, family = tools.pass_probabilities(
            replicates, reading["mean"], sigma**2 / arms, reading["tau"], arms
        )
        legs = [_coverage_leg(c, replicates, pilot.PILOT_DRAWS) for c in frame["coverage"]]
        return float(family * np.prod(legs))
    mean = reading["mean"] + np.sign(reading["mean"] - 1.0) * sigma / np.sqrt(arms)
    _, family = tools.pass_probabilities(replicates, mean, 0.0, reading["tau"], arms)
    legs = []
    for coverage in frame["coverage"]:
        step = np.sqrt(coverage * (1.0 - coverage) / pilot.PILOT_DRAWS)
        shifted = coverage - step if coverage - low < high - coverage else coverage + step
        legs.append(_coverage_leg(shifted, replicates, None))
    return float(family * np.prod(legs))


def test_the_se_ratio_rule_alone_would_give_8400() -> None:
    """The arm-pooled SE-ratio rule alone (H's rule) gives 8,400; it leaves the coverage legs
    out, so it is not the declared count."""
    reading = tools.replicates_from_pilot(_pilot()["se_ratio"].tolist(), pilot.PILOT_DRAWS)
    assert reading["R"] == 8_400, reading
    assert calibration_joint(8_400) < tools.FAMILY_TARGET


def test_the_calibration_count_is_the_joint_rule_count_with_stress_margin() -> None:
    """Design review M2: the joint family first reaches 0.95 at 9,600, marginal under the stress
    (0.949), so the declared count is the next multiple of 400, 10,000, where both clear it."""
    joint = {r: calibration_joint(r) for r in range(8_000, 10_401, 400)}
    smallest = min(r for r, p in joint.items() if p >= tools.FAMILY_TARGET)
    assert smallest == 9_600
    assert joint[9_600] == pytest.approx(0.952, abs=1e-3)
    assert calibration_joint(9_600, stressed=True) < tools.FAMILY_TARGET
    replicates = properties.CALIBRATION_REPLICATES
    assert replicates == smallest + 400 == 10_000
    assert joint[replicates] == pytest.approx(0.957, abs=1e-3)
    assert calibration_joint(replicates, stressed=True) >= tools.FAMILY_TARGET


def test_every_label_passes_its_coverage_leg_at_the_count() -> None:
    """Each label's coverage leg at the declared count, predictive over its pilot coverage, is
    at least 0.90."""
    for label, row in _pilot().iterrows():
        inside = _coverage_leg(
            row["coverage"], properties.CALIBRATION_REPLICATES, pilot.PILOT_DRAWS
        )
        assert inside >= tools.CELL_TARGET, (label, inside)


def test_the_pooling_caveat_is_the_stated_one() -> None:
    """Design review m7: the pooled rule is optimistic for ``strata_s1``. At one pilot SD above
    its pilot ratio, read as known, it passes its SE-ratio leg with about 0.79 at the count."""
    ratio = float(_pilot().loc["strata_s1", "se_ratio"]) + tools.pilot_sigma(pilot.PILOT_DRAWS)
    found = float(
        tools.cell_pass(np.array([ratio]), properties.CALIBRATION_REPLICATES, tools.C_FIT)[0]
    )
    assert found == pytest.approx(0.79, abs=0.02)


def primary_pass(replicates: int) -> float:
    """The product over the nine parameters of the bias leg (99% interval inside 0.25 SDs) and
    the coverage floor (0.90), each predictive over the pilot (the SE sanity band, 0.80 to 1.20,
    cannot fail at the pilot's ratios)."""
    floor = study.STUDY.margins.coverage_floor
    family = 1.0
    for _, row in _pilot().iterrows():
        family *= bias_pass(row["standardized_bias"], replicates, pilot.PILOT_DRAWS) * floor_pass(
            row["coverage"], replicates, floor, pilot.PILOT_DRAWS
        )
    return family


def test_the_primary_count_is_the_smallest_passing_multiple_of_400() -> None:
    """The smallest multiple of 400, at least the floor of 1,000, at which the nine parameters
    pass together with probability at least 0.95: the bias legs bind, at standardized biases up
    to 0.160 (0.960 at 4,400; 0.949 at 4,000)."""
    replicates = study.PRIMARY_REPLICATES
    assert replicates >= 1_000 and replicates % 400 == 0
    assert primary_pass(replicates) >= tools.FAMILY_TARGET
    assert primary_pass(replicates - 400) < tools.FAMILY_TARGET


def test_no_label_regret_reaches_the_bias_margin() -> None:
    """Plan section 5: a label whose regret at n = 2,000 reaches the bias margin (0.25 SDs) is a
    design finding before the run. The pilot's mean regret in efficient SEs stays below it."""
    for label, row in _pilot().iterrows():
        se = study.LIMITS[label]["bound"] / np.sqrt(study.PRIMARY_N)
        assert abs(row["mean_regret"]) / se < study.STUDY.margins.standardized_bias, label


# --------------------------------------------------------------------------- the seeds


def _other_studies() -> list[StudyRecord]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal

    others = {record.seed for record in _other_studies()} | {
        record.resampling_seed for record in _other_studies() if record.resampling_seed is not None
    }
    modules = [m for m, _ in _x11_modules() if m.STUDY.slug != study.STUDY.slug]
    for module in [*modules, boundary, optimal]:
        others |= {module.SEED, module.RESAMPLING_SEED}
    assert study.SEED not in others and study.RESAMPLING_SEED not in others
    assert pilot.PILOT_SEED not in others | {study.SEED, study.RESAMPLING_SEED}


def test_rule_l3_keeps_every_label() -> None:
    assert set(properties.suffixes(study.STUDY).values()) == {()}


def _own_sample_seeds() -> set[int]:
    return base.declared_sample_seeds(
        study.STUDY,
        study.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(study.STUDY),
    )


def test_the_sample_seeds_are_distinct() -> None:
    total = len(study.SCENARIOS) * study.PRIMARY_REPLICATES + sum(
        draw.replicates for draw in properties.PROPERTY_DRAWS
    )
    assert len(_own_sample_seeds()) == total


def _other_x11_sample_seeds() -> set[int]:
    """Every sample seed a declared run of the RM30 and the other X11 studies draws."""
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_optimal_value as optimal
    from tests.studies import learned_rule_optimal_value_properties as optimal_properties

    seeds = base.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})
    for module, module_properties in [*_x11_modules(), (optimal, optimal_properties)]:
        if module.STUDY.slug == study.STUDY.slug:
            continue
        seeds |= base.declared_sample_seeds(
            module.STUDY,
            module.PRIMARY_REPLICATES,
            module_properties.PROPERTY_DRAWS,
            module_properties.suffixes(module.STUDY),
        )
    return seeds


def test_no_sample_seed_is_another_x11_studys() -> None:
    own = _own_sample_seeds()
    others = _other_x11_sample_seeds()
    assert own & others == set()
    assert learned_rule_run.declared_sample_seeds() == own | others


def test_the_smoke_record_draws_no_declared_seed() -> None:
    smoke = base.smoke_record(study.STUDY)
    seeds = {replicate_seed(smoke, s, r) for s in smoke.scenarios for r in range(32)}
    declared = learned_rule_run.declared_sample_seeds()
    assert not seeds & declared
    # The deliberate mutation: the declared record run as a smoke would draw a declared seed.
    assert replicate_seed(study.STUDY, "weighted", 0) in declared


def test_no_pilot_seed_is_a_declared_seed() -> None:
    declared = learned_rule_run.declared_sample_seeds()
    drawn = {
        pilot.sample_seed(setting, r) for setting in law.SETTINGS for r in range(pilot.PILOT_DRAWS)
    }
    assert len(drawn) == len(law.SETTINGS) * pilot.PILOT_DRAWS
    assert not drawn & declared
