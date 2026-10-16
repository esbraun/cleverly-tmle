r"""The pre-run design of the plug-in omitted-variable bound study (X26 plan 3.5, F9, R5, R7, 9.10).

Every number here is a population quantity of a declared law, computed by quadrature in a few
seconds, or a number read from the committed diagnostic ``tests/diagnostics/x26_crossfit_bias``.
Nothing reads a study replicate, and nothing calls the package's curve.  The population curve of
:math:`\hat\nu^2` is built here from the law: the second-moment term :math:`\alpha^2 - \nu^2`,
the share term of the ATT and the ATC, and the mechanism term :math:`D J^{-1} s` of a correctly
specified logistic model, with :math:`s` the score, :math:`J` the information and
:math:`D = E[2 \alpha\, \partial_\theta \alpha]`.  A cross-fitted fit has the same population
curve as an in-sample one.  Each bound's curve is
:math:`\alpha \varepsilon \mp \kappa (\psi_{\nu^2} + \nu^2 (\varepsilon^2 - 1)) / (2 \nu)` at
:math:`\sigma^2 = 1`, with :math:`\kappa` the strength.

The claim cells are the lower and the upper bound.  The corrected :math:`\nu^2` rows are
measured and unjudged (plan 9.10), so no check here sizes them.

The checks, in the plan's letters:

* (a) the truths agree at 60 and 80 nodes, and with a 2e5-draw Monte Carlo;
* (b) no law truncates a cell with probability above 1e-6 per row, and the package's
  ``truncation_active`` check, which refuses a fit with a fitted logit within three standard
  errors of a truncation bound, acts on the default bounds of the three-arm law and of the
  unbalanced law and on no declared bound;
* (c) every control leaves the band with power of at least 0.9 at the declared ``R``, from the
  population ratio of its standard error to the positive's.  The package reports the plain
  influence variance, so the ratio is a population quantity of the law;
* (d) every positive cell's 99% SE-ratio half-width is at most 0.035 at ``R``;
* (f) the family-wise pass probability under perfect calibration reaches 0.95 at ``R``, and
  ``R`` is at least the smallest multiple of 400 that reaches it;
* (g) on ``binary_known``, adding the logistic mechanism term puts the ATT and ATC ratios
  below 0.93 with power of at least 0.9, so the known-mechanism positives have teeth;
* (h) the refusal rates the diagnostic measured on the balanced law sit below the failure bound;
* (i) no declared seed reads the pilot stream ``x26_pilot2_sample``.

The SE-ratio noise of a cell follows ``calibration_pass_probability``: the empirical spread
contributes ``1 / (2 (R - 1))`` and the reported standard errors ``se_noise**2 / R``.  A
reported standard error is the root of an average of :math:`n` squared curve values, so its
relative spread is :math:`\tfrac12 \sqrt{(\kappa_4 - 1)/n}` with :math:`\kappa_4` the curve's
fourth standardized moment.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import binom, norm

from cleverly._refusal import RefusalReason
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import _mechanism_stack as ms
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from tests.diagnostics.x26_crossfit_bias import run as diagnostic
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies import omitted_variable_bound_plugin_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import descriptions
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The family-wise pass probability ``R`` must reach, and the grain of ``R``.
FAMILY_FLOOR = 0.95
REPLICATE_GRAIN = 400

#: The floors of a control's power and of a positive cell's SE-ratio resolution.
CONTROL_POWER_FLOOR = 0.9
HALF_WIDTH_CEILING = 0.035

#: The calibration band, the interval level, and the per-row truncation floor.
BAND = (0.93, 1.07)
LEVEL = 0.99
CLIP_FLOOR = 1e-6

#: The default ATT and ATC truncation bound, the larger of the two automatic bounds at n = 1000.
CONDITIONAL_BOUND = 0.025

#: Gauss-Hermite nodes per axis of the population curves, and of the outcome noise.
LAW_NODES = {"balanced": 10, "unbalanced": 12, "multinomial": 12}
NOISE_NODES = 12

#: The failure bound of every cell, and the replicates it lets a cell lose at the declared R.
FAILURE_BOUND = 0.01

#: The family-wise pass probability at a true ratio of 0.98 and of 1.02, reported without a
#: gate (review R7).  A small common miscalibration costs most of the margin at this ``R``.
FAMILY_OFF_TARGET = {0.98: 0.9999, 1.02: 0.9997}

#: The smallest multiple of 400 whose family-wise pass probability reaches 0.95.
MINIMAL_REPLICATES = 3_600


def _hermite(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / w.sum()


@dataclass(frozen=True)
class Law:
    """A law on quadrature nodes: design, node weights and the true mechanism."""

    design: np.ndarray
    weights: np.ndarray
    mechanism: np.ndarray


@cache
def binary_law(intercept: float = 0.0, nodes: int = LAW_NODES["balanced"]) -> Law:
    x, w = _hermite(nodes)
    grid = np.array(list(itertools.product(x, repeat=4)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=4))), axis=1)
    one = expit(intercept + 0.3 * grid[:, 0] - 0.2 * grid[:, 1] + 0.1 * grid[:, 2])
    return Law(
        design=np.column_stack([np.ones(len(grid)), grid]),
        weights=weights,
        mechanism=np.column_stack([1.0 - one, one]),
    )


@cache
def multinomial_law() -> Law:
    x, w = _hermite(LAW_NODES["multinomial"])
    nodes = np.array(list(itertools.product(x, repeat=3)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)
    return Law(
        design=np.column_stack([np.ones(len(nodes)), nodes]),
        weights=weights,
        mechanism=study.multinomial_mechanism(nodes),
    )


def _law(scenario: str) -> Law:
    kind = study.TRAITS[scenario][0]
    if kind == "multinomial":
        return multinomial_law()
    if kind == "unbalanced":
        return binary_law(study.UNBALANCED_INTERCEPT, LAW_NODES["unbalanced"])
    return binary_law()


@dataclass(frozen=True)
class Population:
    """The population curves of one parameter, as functions of the observed arm and W.

    Each array is ``(nodes, K)``: the value at an observed arm ``a`` and covariate node.
    """

    nu2: float
    alpha: np.ndarray
    second_moment: np.ndarray
    share_term: np.ndarray
    mechanism_term: np.ndarray


def population(scenario: str, token: str) -> Population:
    """The population curve of ``nu^2`` of one parameter of one scenario."""
    law = _law(scenario)
    g, x, omega = law.mechanism, law.design, law.weights
    nodes, arms = g.shape
    d = x.shape[1]
    _, group, arm_code, versus_code = study.PARAMETERS[scenario][token]
    arm, versus = int(arm_code), int(versus_code)
    conditioning = {"att": arm, "atc": versus}.get(group)
    share = None if conditioning is None else float(omega @ g[:, conditioning])
    alpha = np.zeros((nodes, arms))
    slope = np.zeros((nodes, arms, arms))  # d alpha(a, W) / d g_c
    if group == "mean":
        alpha[:, arm] = 1.0 / g[:, arm]
        alpha[:, versus] = -1.0 / g[:, versus]
        slope[:, arm, arm] = -1.0 / g[:, arm] ** 2
        slope[:, versus, versus] = 1.0 / g[:, versus] ** 2
    elif group == "att":
        assert share is not None
        alpha[:, arm] = 1.0 / share
        alpha[:, versus] = -g[:, arm] / g[:, versus] / share
        slope[:, versus, arm] = -1.0 / (g[:, versus] * share)
        slope[:, versus, versus] = g[:, arm] / (g[:, versus] ** 2 * share)
    else:
        assert share is not None
        alpha[:, arm] = g[:, versus] / g[:, arm] / share
        alpha[:, versus] = -1.0 / share
        slope[:, arm, versus] = 1.0 / (g[:, arm] * share)
        slope[:, arm, arm] = -g[:, versus] / (g[:, arm] ** 2 * share)
    nu2 = float(omega @ (g * alpha**2).sum(axis=1))
    reduced = g[:, 1:]
    information = np.zeros(((arms - 1) * d, (arms - 1) * d))
    for m in range(arms - 1):
        for q in range(arms - 1):
            factor = reduced[:, m] * (float(m == q) - reduced[:, q])
            information[m * d : (m + 1) * d, q * d : (q + 1) * d] = (
                x * (omega * factor)[:, None]
            ).T @ x
    penalty = np.zeros_like(information)
    if study.TRAITS[scenario][1] == "ridge":
        # sklearn's C = 1 penalty on the slopes, per training row: all rows in sample, and four
        # fifths of them in a cross-fitted fold.
        rows = study.PRIMARY_N * (1.0 - 1.0 / study.CROSSFIT_FOLDS if _crossfit(scenario) else 1.0)
        penalty = np.diag([0.0] + [1.0] * (d - 1)) / rows
    inner = (slope * g[:, None, :]).sum(axis=2)
    chain = g[:, None, 1:] * (slope[:, :, 1:] - inner[:, :, None])
    gradient = np.einsum(
        "n,na,nam,nd->md", omega, g * 2.0 * alpha, chain, x, optimize=True
    ).reshape(-1)
    direction = np.linalg.solve(information + penalty, gradient)
    residual = np.eye(arms)[None, :, :] - g[:, None, :]
    score = np.einsum("namd->namd", np.einsum("nam,nd->namd", residual[:, :, 1:], x)).reshape(
        nodes, arms, -1
    )
    share_term = np.zeros((nodes, arms))
    if conditioning is not None:
        assert share is not None
        indicator = (np.arange(arms) == conditioning).astype(float)
        share_term = np.broadcast_to(-2.0 * nu2 * (indicator - share) / share, (nodes, arms))
    return Population(
        nu2=nu2,
        alpha=alpha,
        second_moment=alpha**2 - nu2,
        share_term=np.array(share_term),
        mechanism_term=score @ direction,
    )


def _crossfit(scenario: str) -> bool:
    return study.TRAITS[scenario][2]


def moments(scenario: str, curve: np.ndarray, pop: Population, row: str) -> tuple[float, float]:
    """``E[psi^2]`` and ``E[psi^4]`` of one bound's curve, over arm, W and the outcome noise."""
    law = _law(scenario)
    noise, noise_weights = _hermite(NOISE_NODES)
    sign = -1.0 if row == "lower" else 1.0
    bias = (curve[:, :, None] + pop.nu2 * (noise**2 - 1.0)) / (2.0 * math.sqrt(pop.nu2))
    psi = pop.alpha[:, :, None] * noise + sign * study.strength() * bias
    mass = law.weights[:, None, None] * law.mechanism[:, :, None] * noise_weights
    return float((mass * psi**2).sum()), float((mass * psi**4).sum())


def se_noise(second: float, fourth: float) -> float:
    """The relative spread of one reported standard error at ``n``, from the curve's moments."""
    return 0.5 * math.sqrt((fourth / second**2 - 1.0) / study.PRIMARY_N)


@dataclass(frozen=True)
class Cell:
    """A property cell's predicted SE ratio and the relative noise of its standard error."""

    name: str
    ratio: float
    noise: float


def _curves(scenario: str, token: str) -> tuple[Population, np.ndarray, np.ndarray | None]:
    pop = population(scenario, token)
    known = pop.second_moment + pop.share_term
    stacked = known + pop.mechanism_term
    if study.TRAITS[scenario][1] == "known":
        return pop, known, stacked if token in study.CONTROLLED else None
    return pop, stacked, known if token in study.CONTROLLED else None


@cache
def cells() -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Every positive cell, and every control cell, with its predicted ratio and noise."""
    positives: list[Cell] = []
    controls: list[Cell] = []
    for scenario, parameters in study.PARAMETERS.items():
        for token in parameters:
            pop, positive, control = _curves(scenario, token)
            for row in study.CLAIM_ROWS:
                label = study.cell_label(scenario, token, row)
                second, fourth = moments(scenario, positive, pop, row)
                positives.append(
                    Cell(f"{label}__correctly_specified", 1.0, se_noise(second, fourth))
                )
                if control is None:
                    continue
                other, other_fourth = moments(scenario, control, pop, row)
                controls.append(
                    Cell(
                        f"{label}__{study.control_kind(scenario)}",
                        math.sqrt(other / second),
                        se_noise(other, other_fourth),
                    )
                )
    return tuple(positives), tuple(controls)


def _spread(replicates: int, ratio: float, noise: float) -> float:
    return ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)


def control_power(replicates: int, cell: Cell) -> float:
    """The chance that a control's 99% SE-ratio interval leaves the band on its side."""
    ratio = cell.ratio
    z = float(norm.ppf(0.5 + LEVEL / 2.0))
    spread = _spread(replicates, ratio, cell.noise)
    if cell.name.endswith("__inflated_se_control"):
        return float(norm.cdf((ratio - BAND[1]) / spread - z))
    return float(norm.cdf((BAND[0] - ratio) / spread - z))


def family_pass_probability(replicates: int, ratio: float = 1.0) -> float:
    """The chance that every positive cell passes, at a common true ratio, cells independent.

    The cells of one scenario share their draws and pass together more often than
    independent cells would, so the product is the conservative reading.
    """
    z = float(norm.ppf(0.975))
    coverage = float(norm.cdf(z * ratio) - norm.cdf(-z * ratio))
    positives, _ = cells()
    return float(
        np.prod(
            [
                calibration_pass_probability(
                    replicates, true_se_ratio=ratio, coverage=coverage, se_noise=cell.noise
                )
                for cell in positives
            ]
        )
    )


# ------------------------------------------------------------------------ the declaration


def test_the_declared_cells_are_the_design() -> None:
    """48 positive cells, 28 inflated controls and 4 shrunken controls, all described."""
    declared = study.PROPERTY_CELLS["interval_calibration"]
    kinds = [cell.rsplit("__", 1)[1] for cell in declared]
    assert kinds.count("correctly_specified") == 48
    assert kinds.count("inflated_se_control") == 28
    assert kinds.count("shrunken_se_control") == 4
    assert not any("_nu2_" in cell for cell in declared)
    positives, controls = cells()
    assert {cell.name for cell in (*positives, *controls)} == set(declared)
    for cell in declared:
        descriptions.cell("interval_calibration", cell, slug=study.STUDY.slug)
    for scenario, estimands in study.SCENARIOS.items():
        descriptions.scenario(scenario)
        for estimand in estimands:
            descriptions.estimand(estimand)
    descriptions.implementation(study.STUDY.implementation)
    for scenario in study.PARAMETERS:
        assert properties.scenario_cells(scenario)
        for token in study.PARAMETERS[scenario]:
            assert properties.parameter_cells(scenario, token)
    assert sum(len(properties.scenario_cells(s)) for s in study.PARAMETERS) == len(declared)


def test_each_parameter_is_its_own_primary_scenario() -> None:
    """A refusal of one parameter loses that parameter's primary rows only.

    The failure accounting costs every estimand of a failed scenario, so the primary scenarios
    are the 24 parameters, and the three parameters of a scenario read one sample.
    """
    assert len(study.SCENARIOS) == 24
    assert all(len(estimands) == 2 for estimands in study.SCENARIOS.values())
    for scenario, parameters in study.PARAMETERS.items():
        tokens = list(parameters)
        owner = study.primary_name(scenario, tokens[0])
        assert owner not in study.SEED_OWNERS
        frames = [study.draw_scenario(study.primary_name(scenario, t), 50, 0)[0] for t in tokens]
        for other in frames[1:]:
            pd.testing.assert_frame_equal(frames[0], other)
        assert {study.SEED_OWNERS[study.primary_name(scenario, t)] for t in tokens[1:]} == {owner}
    first = {
        scenario: study.draw_scenario(study.primary_name(scenario, next(iter(parameters))), 50, 0)[
            0
        ]
        for scenario, parameters in study.PARAMETERS.items()
    }
    for left, right in itertools.combinations(first, 2):
        assert not first[left].equals(first[right]), (left, right)


# ------------------------------------------------------------------------------ (a) truths


def test_the_balanced_truths_are_the_closed_forms() -> None:
    """The closed forms equal the one-dimensional quadrature over the index ``eta``."""
    truth = study.truths("binary_logistic")
    moments_, share = study.binary_second_moments(0.0)
    assert share == pytest.approx(0.5, abs=1e-14)
    for token in study.BINARY_PARAMETERS:
        assert moments_[token] == pytest.approx(truth[f"plugin_{token}_nu2"], rel=1e-12)
    assert truth["plugin_atc_nu2"] == truth["plugin_att_nu2"]


def test_the_unbalanced_truths_converge_and_the_share_is_a_fifth() -> None:
    """(a) The 60-node and 80-node rules agree to 1e-10, and a 2e5-draw Monte Carlo agrees.

    The treated share is 0.1889, the expected value of ``expit`` over the shifted index.
    """
    coarse, share_coarse = study.binary_second_moments(study.UNBALANCED_INTERCEPT, 60)
    fine, share = study.binary_second_moments(study.UNBALANCED_INTERCEPT, 80)
    for token in fine:
        assert coarse[token] == pytest.approx(fine[token], rel=1e-10, abs=0.0)
    assert share_coarse == pytest.approx(share, rel=1e-12)
    assert 0.18 < share < 0.19
    assert share == pytest.approx(0.1889, abs=1e-4)
    rng = np.random.default_rng(20262604)
    g = expit(
        study.UNBALANCED_INTERCEPT + math.sqrt(study.INDEX_VARIANCE) * rng.standard_normal(200_000)
    )
    p = g.mean()
    terms = {
        "ate": 1.0 / g + 1.0 / (1.0 - g),
        "att": g**2 / (1.0 - g),
        "atc": (1.0 - g) ** 2 / g,
    }
    for name, values in terms.items():
        error = values.std(ddof=1) / math.sqrt(values.size)
        if name == "ate":
            assert abs(values.mean() - fine["ate"]) < 4.5 * error
        elif name == "att":
            expected = fine["att"] * share**2 - share
            assert abs(values.mean() - expected) < 4.5 * error
        else:
            expected = fine["atc"] * (1.0 - share) ** 2 - (1.0 - share)
            assert abs(values.mean() - expected) < 4.5 * error
    assert abs(p - share) < 4.5 * g.std(ddof=1) / math.sqrt(g.size)
    truth = study.truths("binary_unbalanced")
    for token in study.BINARY_PARAMETERS:
        assert truth[f"plugin_{token}_nu2"] == pytest.approx(fine[token], rel=1e-14)
    assert truth["plugin_att_nu2"] != truth["plugin_atc_nu2"]


def test_the_quadrature_laws_reproduce_the_truths() -> None:
    """The nodes of the population curves return each law's ``nu^2`` to 1e-5 relative."""
    for scenario, parameters in study.PARAMETERS.items():
        truth = study.truths(scenario)
        for token in parameters:
            assert population(scenario, token).nu2 == pytest.approx(
                truth[study.estimand_key(token, "nu2")], rel=1e-5
            ), (scenario, token)


def test_the_three_arm_truths_converge() -> None:
    """(a) The 60-node and 80-node rules agree to 1e-10, and a 2e5-draw Monte Carlo agrees."""
    coarse = study.multinomial_second_moments(60)
    fine = study.multinomial_second_moments(80)
    for token in fine:
        assert coarse[token] == pytest.approx(fine[token], rel=1e-10, abs=0.0)
    rng = np.random.default_rng(20262603)
    g = study.multinomial_mechanism(rng.standard_normal((200_000, 2)))
    terms = {
        "ate_2v0": 1.0 / g[:, 2] + 1.0 / g[:, 0],
        "share_2": g[:, 2],
        "share_0": g[:, 0],
        "att_2v0": g[:, 2] ** 2 / g[:, 0],
        "atc_1v0": g[:, 0] ** 2 / g[:, 1],
    }
    x, w = _hermite(80)
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    omega = np.outer(w, w).reshape(-1)
    exact_g = study.multinomial_mechanism(np.column_stack([w1.reshape(-1), w2.reshape(-1)]))
    exact = {
        "ate_2v0": omega @ (1.0 / exact_g[:, 2] + 1.0 / exact_g[:, 0]),
        "share_2": omega @ exact_g[:, 2],
        "share_0": omega @ exact_g[:, 0],
        "att_2v0": omega @ (exact_g[:, 2] ** 2 / exact_g[:, 0]),
        "atc_1v0": omega @ (exact_g[:, 0] ** 2 / exact_g[:, 1]),
    }
    for name, values in terms.items():
        error = values.std(ddof=1) / math.sqrt(values.size)
        assert abs(values.mean() - exact[name]) < 4.5 * error, name
    assert exact["ate_2v0"] == pytest.approx(fine["ate_2v0"], rel=1e-14)
    p2, p0 = exact["share_2"], exact["share_0"]
    assert (p2 + exact["att_2v0"]) / p2**2 == pytest.approx(fine["att_2v0"], rel=1e-14)
    assert (p0 + exact["atc_1v0"]) / p0**2 == pytest.approx(fine["atc_1v0"], rel=1e-14)


# ------------------------------------------------------------------------- (b) no clipping


def _three_arm_clip_probability(bound: float, angles: int = 4096) -> float:
    r"""``P(min_a g_a(W) < bound)`` per row of the three-arm law.

    Each :math:`\log g_a` is concave in :math:`(W_1, W_2)`, so the set where every arm is at
    least ``bound`` is convex and contains the origin.  At three arms an arm above
    ``1 - 2 bound`` forces the other two below ``bound``, so this set is the whole untruncated
    region.  Its complement has probability :math:`(2\pi)^{-1} \int e^{-r(\phi)^2/2} d\phi`,
    with :math:`r(\phi)` the radius at which the ray at angle :math:`\phi` leaves the set.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    direction = np.column_stack([np.cos(phi), np.sin(phi)])
    low = np.zeros(angles)
    high = np.full(angles, 60.0)
    for _ in range(80):
        middle = (low + high) / 2.0
        inside = study.multinomial_mechanism(direction * middle[:, None]).min(axis=1) >= bound
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return float(np.mean(np.exp(-(low**2) / 2.0)))


def test_no_law_truncates_a_cell() -> None:
    """(b) Every law moves a cell with probability below 1e-6 per row (plan F1).

    The automatic ATE bound at n = 1000 is ``5 / (sqrt(n) log n)``, about 0.0229, and the ATT
    and ATC bound is 0.025, so the larger bound is checked on the two binary laws.  The
    unbalanced law moves a cell with probability 3.7e-9 per row at 0.025, and the three-arm law
    sits at about 7.3e-7 per row.  The scenarios that declare ``WIDE_G_BOUNDS`` truncate nothing
    at 0.005 on either law.  The rebuild applies the package's truncation rule, so a replicate
    that moves a cell still reads its own curve.
    """
    automatic = 5.0 / (math.sqrt(study.PRIMARY_N) * math.log(study.PRIMARY_N))
    assert automatic < CONDITIONAL_BOUND
    index_sd = math.sqrt(study.INDEX_VARIANCE)
    balanced = 2.0 * float(norm.cdf(logit(CONDITIONAL_BOUND) / index_sd))
    assert balanced < 1e-21
    unbalanced = float(norm.cdf((logit(CONDITIONAL_BOUND) - study.UNBALANCED_INTERCEPT) / index_sd))
    assert unbalanced < CLIP_FLOOR
    assert unbalanced == pytest.approx(3.7e-9, rel=0.1)
    lower = study.WIDE_G_BOUNDS[0]
    assert float(norm.cdf((logit(lower) - study.UNBALANCED_INTERCEPT) / index_sd)) < 1e-21
    three_arm = _three_arm_clip_probability(CONDITIONAL_BOUND)
    assert three_arm < CLIP_FLOOR
    assert three_arm == pytest.approx(7.3e-7, rel=0.05)
    assert _three_arm_clip_probability(lower) < 1e-12


def _margin_hit_probability(
    scenario: str, bounds: tuple[float, float], batches: int = 100, n: int = study.PRIMARY_N
) -> float:
    """The chance that a sample has a fitted logit within three standard errors of a bound.

    The package refuses the plug-in limits then (``truncation_active``).  The population
    mechanism stands in for the fit, and the sandwich of a correct logistic model gives the
    standard error of each logit: the gradient of :math:`\\mathrm{logit}\\, g_c` is
    :math:`(e_c - g)_{-r} / (1 - g_c) \\otimes x`, and the covariance is :math:`J^{-1} / n`.
    The check reads the two arms of the representer.  The estimate ignores the spread of the
    fitted coefficients, so it is a prediction of the order of magnitude and not of the rate.
    """
    law = _law(scenario)
    g, x, omega = law.mechanism, law.design, law.weights
    arms, d = g.shape[1], x.shape[1]
    reduced = g[:, 1:]
    information = np.zeros(((arms - 1) * d, (arms - 1) * d))
    for m in range(arms - 1):
        for q in range(arms - 1):
            factor = reduced[:, m] * (float(m == q) - reduced[:, q])
            information[m * d : (m + 1) * d, q * d : (q + 1) * d] = (
                x * (omega * factor)[:, None]
            ).T @ x
    covariance = np.linalg.inv(information) / n
    edges = [math.log(b / (1.0 - b)) for b in bounds]
    columns = [1] if arms == 2 else [0, 2]
    # A multinomial representer of an ATE or ATT of arm 2 versus 0 reads arms 0 and 2; the
    # ATC of arm 1 versus 0 reads arms 0 and 1.  The check below takes the larger set.
    if arms == 3:
        columns = [0, 1, 2]
    rng = np.random.default_rng(20262605)
    hits = 0
    for _ in range(batches):
        rows = rng.choice(len(omega), size=n, p=omega / omega.sum())
        gm, xm = g[rows], x[rows]
        best = math.inf
        for c in columns:
            coefficient = (np.eye(arms)[c][None, 1:] - gm[:, 1:]) / (1.0 - gm[:, [c]])
            gradient = (coefficient[:, :, None] * xm[:, None, :]).reshape(n, -1)
            se = np.sqrt(np.einsum("ij,jk,ik->i", gradient, covariance, gradient))
            fitted = np.log(gm[:, c] / (1.0 - gm[:, c]))
            for edge in edges:
                best = min(best, float(np.min(np.abs(fitted - edge) / se)))
        hits += best <= 3.0
    return hits / batches


@pytest.mark.parametrize(
    ("scenario", "low", "high"),
    [
        ("binary_logistic", 0.0, 0.0),
        ("binary_unbalanced", 0.3, 1.0),
        ("multinomial_logistic", 0.2, 1.0),
    ],
)
def test_the_truncation_check_acts_on_the_default_bounds_of_two_laws(
    scenario: str, low: float, high: float
) -> None:
    """(b) The default bounds put an extreme row within three standard errors, in some laws.

    The prediction is 0.0 on the balanced law, near 0.6 on the unbalanced law and near 0.4 on
    the three-arm law.  A 30-draw probe of the package measured 0.52 and 0.27.  The declared
    bounds predict 0.0 on both.  The probe is why these scenarios declare ``WIDE_G_BOUNDS``.
    """
    default = (CONDITIONAL_BOUND, 1.0 - CONDITIONAL_BOUND)
    share = _margin_hit_probability(scenario, default)
    assert low <= share <= high
    declared = study.WIDE_G_BOUNDS
    if scenario in study.WIDE_G_SCENARIOS:
        assert _margin_hit_probability(scenario, declared) == 0.0
    else:
        assert scenario not in study.WIDE_G_SCENARIOS


def test_the_wide_bounds_reach_the_fit() -> None:
    """The three scenarios that declare ``WIDE_G_BOUNDS`` fit with them, and no other does.

    The ATT and the ATC read the conditional bound, so it is pinned beside the marginal one.
    The manifest is written from ``CONFIGURATION``, which must state the bounds of exactly
    those three scenarios.
    """
    for scenario in study.PARAMETERS:
        frame, _ = study.draw_from_seed(scenario, 120, 7)
        result = study.fit(frame, scenario)
        wide = scenario in study.WIDE_G_SCENARIOS
        assert (tuple(result.config.g_bounds) == study.WIDE_G_BOUNDS) is wide, scenario
        if wide:
            assert tuple(result.config.g_bounds_conditional) == study.WIDE_G_BOUNDS, scenario
        declared = f"g_bounds={study.WIDE_G_BOUNDS}" in study.CONFIGURATION["fits"][scenario]
        assert declared is wide, scenario


# -------------------------------------------------------------------- (c) and (g) controls


def test_the_controls_leave_the_band_with_power() -> None:
    """(c) Every control leaves the band with power of at least 0.99 at the declared R.

    The ratio of a control is the root of the ratio of two population variances of the changed
    curve, the plain influence variance of each.  No sample quantity enters it.
    """
    _, controls = cells()
    assert len(controls) == 32
    inflated = [c for c in controls if c.name.endswith("inflated_se_control")]
    assert len(inflated) == 28
    for cell in controls:
        assert control_power(properties.PROPERTY_REPLICATES, cell) >= 0.99, cell
    assert min(c.ratio for c in inflated) > BAND[1] + 0.04


def test_the_known_mechanism_positives_have_teeth() -> None:
    """(g) The logistic term added to a known-mechanism curve reads below 0.93 with power.

    This is mutation M12 seen through the study: a package that stacked the logistic score
    on a declared mechanism would report these ratios on the ATT and ATC positives.
    """
    _, controls = cells()
    shrunken = [c for c in controls if c.name.startswith("binary_known_")]
    assert len(shrunken) == 4
    for cell in shrunken:
        assert cell.ratio < BAND[0], cell
        assert control_power(properties.PROPERTY_REPLICATES, cell) >= CONTROL_POWER_FLOOR, cell
    assert max(c.ratio for c in shrunken) == pytest.approx(0.8533, abs=1e-3)


# ---------------------------------------------------------------------- (d) the positives


def test_every_positive_cell_resolves_the_band() -> None:
    """(d) Each positive cell's 99% SE-ratio half-width is at most 0.035 at the declared R."""
    positives, _ = cells()
    assert len(positives) == 48
    z = float(norm.ppf(0.5 + LEVEL / 2.0))
    widths = {
        cell.name: z * _spread(properties.PROPERTY_REPLICATES, 1.0, cell.noise)
        for cell in positives
    }
    assert max(widths.values()) <= HALF_WIDTH_CEILING


# ------------------------------------------------------------------------- (f) the budget


def test_the_property_replicates_follow_the_sizing_rule() -> None:
    """(f) R reaches family-wise pass probability 0.95, and exceeds the minimal R.

    The minimal multiple of 400 is recorded in ``MINIMAL_REPLICATES``.  The study declares the
    plan's 10,000 instead, which keeps the chance above 0.99 when the common ratio is 0.98.
    """
    replicates = properties.PROPERTY_REPLICATES
    assert replicates % REPLICATE_GRAIN == 0
    assert family_pass_probability(replicates) >= FAMILY_FLOOR
    assert family_pass_probability(MINIMAL_REPLICATES) >= FAMILY_FLOOR
    assert family_pass_probability(MINIMAL_REPLICATES - REPLICATE_GRAIN) < FAMILY_FLOOR
    assert replicates >= MINIMAL_REPLICATES


def test_the_pass_probability_off_target_is_recorded() -> None:
    """R7: the family-wise chance at true ratios 0.98 and 1.02, reported without a gate."""
    replicates = properties.PROPERTY_REPLICATES
    for ratio, chance in FAMILY_OFF_TARGET.items():
        assert family_pass_probability(replicates, ratio) == pytest.approx(chance, abs=5e-4)


# --------------------------------------------------------------------------- (h) refusals


def test_the_measured_refusal_rates_sit_below_the_failure_bound() -> None:
    """(h) The diagnostic's refusal rate of every parameter is below 0.01, with a wide margin.

    On the balanced law at n = 1000 the package refused 1 of 2,000 cross-fitted draws for each
    of the ATT and the ATC, and 1 of 2,000 in-sample draws for each of them (0.05 percent), all
    with the key ``truncation_active``. The ATE lost none. A cell may lose ``floor(0.01 R)``
    replicates, 100 at R = 10,000. At the measured rate the chance of exceeding that is below
    1e-12. The 99% upper endpoint of the rate, 0.2 percent, still sits below the bound.
    """
    rows = pd.read_csv(diagnostic.HERE / "rows.csv.gz")
    rows = rows.loc[rows["n"] == study.PRIMARY_N]
    refused = rows["refusal"].fillna("").astype(str).ne("")
    counts = rows.loc[refused].groupby(["fitting", "parameter"]).size()
    allowed = math.floor(FAILURE_BOUND * properties.PROPERTY_REPLICATES)
    assert allowed == 100
    arms = diagnostic.ARMS[study.PRIMARY_N]
    for (fitting, token), count in counts.items():
        rate = count / arms
        assert rate < FAILURE_BOUND, (fitting, token, rate)
        exceed = float(binom.sf(allowed, properties.PROPERTY_REPLICATES, rate))
        assert exceed < 1e-12, (fitting, token, exceed)
        upper = float(binom.ppf(0.995, arms, rate)) / arms
        assert upper < FAILURE_BOUND, (fitting, token, upper)
    assert counts.to_dict() == {
        ("crossfit", "atc"): 1,
        ("crossfit", "att"): 1,
        ("insample", "atc"): 1,
        ("insample", "att"): 1,
    }
    assert set(rows.loc[refused, "refusal"]) == {"truncation_active"}


# ---------------------------------------------------------------- the unjudged nu^2 reading


def _nu2_rows(count: int = 40) -> pd.DataFrame:
    """Primary rows of one parameter with the extra columns the reading reads."""
    rng = np.random.default_rng(5)
    rows = []
    for replicate in range(count):
        truth, raw = 4.0, 4.0 + 0.3 + rng.normal(scale=0.5)
        correction = 0.3
        for estimand in ("plugin_att_lower", "plugin_att_upper"):
            rows.append(
                {
                    "scenario": "binary_logistic_att",
                    "replicate": replicate,
                    "n": 1000,
                    "estimand": estimand,
                    "requested_replicates": count + 2,
                    "nu2_truth": truth,
                    "nu2_estimate": raw - correction,
                    "nu2_std_error": 0.4,
                    "nu2_raw_estimate": raw,
                    "nu2_correction": correction,
                }
            )
    return pd.DataFrame(rows)


def test_the_nu2_reading_is_unjudged_and_counts_the_refusals() -> None:
    """The reading has no verdict column, and ``refused`` is the requested less the survivors."""
    frames = study.extra_artifacts(_nu2_rows(40))
    assert set(frames) == set(study.STUDY.extra_artifacts)
    reading = frames[study.NU2_READING]
    assert list(reading.columns) == list(study.NU2_READING_COLUMNS)
    assert not {"passed", "property_passed", "verdict"} & set(reading.columns)
    (row,) = reading.itertuples(index=False)
    assert (row.scenario, row.parameter) == ("binary_logistic", "att")
    assert (row.requested, row.replicates, row.refused) == (42, 40, 2)
    assert row.mean_correction == pytest.approx(0.3)
    per_draw = frames[study.NU2_ROWS]
    assert list(per_draw.columns) == list(study.NU2_ROW_COLUMNS)
    assert len(per_draw) == 40
    estimate = _nu2_rows(40).loc[lambda f: f["estimand"].eq("plugin_att_lower"), "nu2_estimate"]
    spread = float(estimate.std(ddof=1))
    assert row.bias_sd == pytest.approx((float(estimate.mean()) - 4.0) / spread)
    assert row.se_ratio == pytest.approx(0.4 / spread)
    assert row.raw_bias_sd == pytest.approx(row.bias_sd + 0.3 / spread)
    covered = int((np.abs(estimate.to_numpy() - 4.0) <= study.TWO_SIDED * 0.4).sum())
    assert row.coverage == pytest.approx(covered / 40)
    assert row.coverage_lower99 < row.coverage < row.coverage_upper99


def test_the_nu2_audit_passes_only_when_the_reading_counts_every_lost_fit() -> None:
    """The failure audit compares ``refused`` with the records, parameter by parameter."""
    reading = study.extra_artifacts(_nu2_rows(40))
    record = {
        "implementation": study.STUDY.implementation,
        "group_id": "primary/binary_logistic_att/r000000041",
        "scenarios": "binary_logistic_att",
        "replicates": "40;41",
        "attempt": 1,
        "class": "fitting_failure",
        "stage": "cleverly fit",
        "call": "",
        "message": "DataError: refused",
        "seed": "NA",
    }
    failures = pd.DataFrame([record])
    audit = study.account_failures(failures, reading)
    assert audit["passed"].tolist() == [True]
    one = pd.DataFrame([{**record, "replicates": "40"}])
    audit = study.account_failures(one, reading)
    assert audit["passed"].tolist() == [False]
    other = pd.DataFrame([{**record, "scenarios": "binary_logistic_atc"}])
    audit = study.account_failures(other, reading)
    assert audit["passed"].tolist() == [False]
    assert audit["failed"].tolist() == [2]


# ------------------------------------------- the pre-queue design review (C3, C6, C7)

#: The pilot stream of the redesigned study, which the declared run does not use.
PILOT_STREAM = "x26_pilot2_sample"


def test_the_refusal_reading_counts_each_key_of_each_parameter() -> None:
    """The refusals by key, per scenario and parameter, with the rate against the request."""
    scenario = "multinomial_logistic_crossfit"
    cell = f"interval_calibration/{study.cell_label(scenario, 'att_2v0', 'lower')}__correctly_specified"

    def record(replicate: int, key: str) -> dict[str, object]:
        return {
            "group_id": f"properties/interval_calibration/{scenario}/{replicate}",
            "scenarios": cell,
            "message": f"DataError: the plug-in bound of 'att[2.0 vs 0.0]' refused: {key}",
        }

    failures = pd.DataFrame(
        [
            record(1, "corrected_nonpositive"),
            record(2, "corrected_nonpositive"),
            record(3, "truncation_active"),
            {**record(4, "x"), "message": "DataError: the fit failed"},
            record(5, "score_not_zero"),
        ]
    )
    reading = properties.refusal_reading(failures, 100)
    assert list(reading.columns) == list(properties.REFUSAL_READING_COLUMNS)
    assert len(reading) == 24 * (len(study.REFUSAL_KEYS) + 3)
    lost = reading.loc[reading["lost"] > 0].set_index("key")
    assert lost["lost"].to_dict() == {
        "corrected_nonpositive": 2,
        "truncation_active": 1,
        "score_not_zero": 1,
        "other": 1,
        "all": 5,
    }
    assert set(lost["parameter"]) == {"att_2v0"}
    assert lost.loc["all", "rate"] == pytest.approx(0.05)
    quiet = properties.refusal_reading(None, 100)
    assert len(quiet) == 24 * (len(study.REFUSAL_KEYS) + 2)
    assert quiet["lost"].sum() == 0


def test_the_refusal_keys_are_read_from_the_study_messages() -> None:
    """``refusal_key`` reads every key the message names and calls a non-refusal ``other``."""
    for key in (*study.REFUSAL_KEYS, "score_not_zero", "missing_arm", "known_truncated"):
        assert study.refusal_key(f"DataError: the plug-in bound of 'att' refused: {key}") == key
    assert study.refusal_key("DataError: GLM: the information matrix is singular") == "other"


def _fit(scenario: str, replicate: int) -> Any:
    seed = stream_seed(study.STUDY, "design_test_sample", scenario, replicate)
    frame, _ = study.draw_from_seed(scenario, study.PRIMARY_N, seed)
    return study.fit(frame, scenario)


def test_the_reported_variance_is_the_plain_influence_variance() -> None:
    """The package's limits equal the study's rebuild from the plain influence variance.

    A nonzero witness: the second-order correction of the point is nonzero on this fit, so a
    variance that carried a correction of its own would differ from the plain influence
    variance of the rebuilt curve.  ``parameter_rows`` raises ``RuntimeError`` on a departure
    beyond the rebuild tolerance, so a return is the witness.  The control is the deliberate
    mutation: shrinking the package's variance by 1 percent breaks the rebuild.
    """
    result = _fit("binary_logistic", 3)
    for token in study.PARAMETERS["binary_logistic"]:
        values = study.parameter_rows(result, "binary_logistic", token)
        assert values.nu2.correction > 0.0
        assert values.nu2.std_error == pytest.approx(
            math.sqrt(
                float(
                    influence_variance(
                        omitted_variable_module.sensitivity_elements(
                            result,
                            study.PARAMETERS["binary_logistic"][token][0],
                            nu2_estimator="plugin",
                        ).psi_nu2,
                        None,
                    )
                )
            ),
            rel=1e-12,
        )
    original = omitted_variable_module.influence_variance

    def corrected(curve: Any, cluster: Any) -> float:
        return 0.99 * float(original(curve, cluster))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(omitted_variable_module, "influence_variance", corrected)
        with pytest.raises(RuntimeError, match="standard error"):
            study.parameter_rows(result, "binary_logistic", "att")


def test_a_refusal_of_the_bound_is_recorded_and_a_capability_error_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7: a refusal in ``limit_refusal`` is a recorded ``DataError``; nothing else is.

    The package no longer raises a refusal from the bound for a nonpositive variance, so
    ``parameter_rows`` converts no ``CapabilityError``.  One from the bound is a defect of the
    study and stops the run.
    """
    result = _fit("binary_logistic", 3)
    monkeypatch.setattr(ms, "CORRECTION_STABILITY", 1e-9)
    with pytest.raises(DataError, match=r"refused: correction_unstable$") as recorded:
        study.parameter_rows(result, "binary_logistic", "att")
    assert study.refusal_key(str(recorded.value)) == "correction_unstable"
    monkeypatch.undo()

    def bounds(*args: object, **kwargs: object) -> object:
        raise CapabilityError("a study bug", reason=RefusalReason.INFERENCE_GAP, alternative="none")

    monkeypatch.setattr(omitted_variable_module, "omitted_variable_bounds", bounds)
    with pytest.raises(CapabilityError, match="a study bug"):
        study.parameter_rows(result, "binary_logistic", "att")


def test_the_red_cell_policy_declares_no_red_before_the_run() -> None:
    """The policy names the finite-sample, conservative and refusal-limited readings, and no red.

    The variance of the bound changed after the first pilot, and the second pilot added the
    conservative reading.  No cell, scenario or control is declared red or underpowered here.
    """
    assert properties.FAILURE_BOUND == 0.01
    assert study.STUDY.failure_bound == {study.STUDY.implementation: 0.01}
    policy = " ".join(study.RED_CELL_POLICY)
    owners = (study.RED_CELL_OWNER, study.CONSERVATIVE_OWNER, study.REFUSAL_LIMITED_OWNER)
    assert study.CONFIGURATION["red_cell_owners"] == {
        "finite_sample": study.RED_CELL_OWNER,
        "conservative": study.CONSERVATIVE_OWNER,
        "refusal_limited": study.REFUSAL_LIMITED_OWNER,
    }
    for owner in owners:
        assert owner in policy
    assert "refusal-limited" in policy
    assert "conservative" in policy
    for gone in ("bound exceeded", "variance-correction", "declared so before the run"):
        assert gone not in policy
    assert not hasattr(study, "EXPECTED_BOUND_EXCEEDED")
    assert not hasattr(study, "UNDERPOWERED_CONTROLS")
    assert "declared_red_cells" not in study.CONFIGURATION
    assert set(study.STUDY.property_artifacts) == {study.REFUSAL_READING}


def test_no_declared_seed_reads_the_pilot_stream() -> None:
    """(i) The pilot stream lies outside the study: no seed or label of the study names it."""
    primary = {
        replicate_seed(study.STUDY, name, k) for name in study.STUDY.scenarios for k in range(40)
    }
    declared = {
        stream_seed(study.STUDY, "property_sample", "interval_calibration", scenario, k)
        for scenario in study.PARAMETERS
        for k in range(40)
    }
    pilot = {
        stream_seed(study.STUDY, PILOT_STREAM, "interval_calibration", scenario, k)
        for scenario in study.PARAMETERS
        for k in range(40)
    }
    assert len(pilot) == len(study.PARAMETERS) * 40
    assert not pilot & (primary | declared)
    for module in (study, properties):
        source = inspect.getsource(module)
        assert PILOT_STREAM not in source
    assert PILOT_STREAM not in json.dumps(study.CONFIGURATION)
    assert PILOT_STREAM not in " ".join(study.RED_CELL_POLICY)
