r"""The pre-run design of the plug-in omitted-variable bound study (X26 plan 3.5, F9, R5, R7).

Every number here is a population quantity of a declared law, computed by quadrature in a few
seconds.  Nothing reads a study replicate, and nothing calls the package's curve.  The
population curve of :math:`\hat\nu^2` is built here from the law: the second-moment term
:math:`\alpha^2 - \nu^2`, the share term of the ATT and the ATC, and the mechanism term
:math:`D J^{-1} s` of a correctly specified logistic model, with :math:`s` the score,
:math:`J` the information and :math:`D = E[2 \alpha\, \partial_\theta \alpha]`.  Each bound's curve is
:math:`\alpha \varepsilon \mp \kappa (\psi_{\nu^2} + \nu^2 (\varepsilon^2 - 1)) / (2 \nu)` at
:math:`\sigma^2 = 1`, with :math:`\kappa` the strength.

The checks, in the plan's letters:

* (a) the three-arm truths agree at 60 and 80 nodes, and with a 2e5-draw Monte Carlo;
* (b) no law truncates a cell with probability above 1e-6 per row;
* (c) every control leaves the band with power of at least 0.9 at the declared ``R``;
* (d) every positive cell's 99% SE-ratio half-width is at most 0.035 at ``R``;
* (f) the family-wise pass probability under perfect calibration reaches 0.95 at ``R``, and
  ``R`` is at least the smallest multiple of 400 that reaches it;
* (g) on ``binary_known``, adding the logistic mechanism term puts the ATT and ATC ratios
  below 0.93 with power of at least 0.9, so the known-mechanism positives have teeth.

The SE-ratio noise of a cell follows ``calibration_pass_probability``: the empirical spread
contributes ``1 / (2 (R - 1))`` and the reported standard errors ``se_noise**2 / R``.  A
reported standard error is the root of an average of :math:`n` squared curve values, so its
relative spread is :math:`\tfrac12 \sqrt{(\kappa_4 - 1)/n}` with :math:`\kappa_4` the curve's
fourth standardized moment.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

from tests.studies import omitted_variable_bound_plugin as study
from tests.studies import omitted_variable_bound_plugin_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import descriptions

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

#: The ATT and ATC truncation bound, the larger of the two automatic bounds at n = 1000.
CONDITIONAL_BOUND = 0.025

#: Gauss-Hermite nodes per axis of the population curves, and of the outcome noise.
LAW_NODES = {"binary": 10, "multinomial": 12}
NOISE_NODES = 12

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
def binary_law() -> Law:
    x, w = _hermite(LAW_NODES["binary"])
    nodes = np.array(list(itertools.product(x, repeat=4)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=4))), axis=1)
    one = expit(0.3 * nodes[:, 0] - 0.2 * nodes[:, 1] + 0.1 * nodes[:, 2])
    return Law(
        design=np.column_stack([np.ones(len(nodes)), nodes]),
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
    return multinomial_law() if scenario == "multinomial_logistic" else binary_law()


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
    if scenario == "binary_logistic_ridge":
        # sklearn's C = 1 penalty on the slopes, per observation at n = 1000.
        penalty = np.diag([0.0] + [1.0] * (d - 1)) / study.PRIMARY_N
    inner = (slope * g[:, None, :]).sum(axis=2)
    chain = g[:, None, 1:] * (slope[:, :, 1:] - inner[:, :, None])
    gradient = np.einsum(
        "n,na,nam,nd->md", omega, g * 2.0 * alpha, chain, x, optimize=True
    ).reshape(-1)
    direction = np.linalg.solve(information + penalty, gradient)
    residual = np.eye(arms)[None, :, :] - g[:, None, :]
    score = np.einsum("nam,nd->namd", residual[:, :, 1:], x).reshape(nodes, arms, -1)
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


def moments(scenario: str, curve: np.ndarray, pop: Population, row: str) -> tuple[float, float]:
    """``E[psi^2]`` and ``E[psi^4]`` of one row's curve, over arm, W and the outcome noise."""
    law = _law(scenario)
    noise, noise_weights = _hermite(NOISE_NODES)
    if row == "nu2":
        psi = np.broadcast_to(curve[:, :, None], (*curve.shape, noise.size))
    else:
        sign = -1.0 if row == "lower" else 1.0
        bias = (curve[:, :, None] + pop.nu2 * (noise**2 - 1.0)) / (2.0 * math.sqrt(pop.nu2))
        psi = pop.alpha[:, :, None] * noise + sign * study.strength() * bias
    mass = law.weights[:, None, None] * law.mechanism[:, :, None] * noise_weights
    return float((mass * psi**2).sum()), float((mass * psi**4).sum())


def se_noise(second: float, fourth: float) -> float:
    """The relative spread of one reported standard error at ``n``."""
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
    if scenario == "binary_known":
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
            for row in study.ROWS:
                label = study.cell_label(scenario, token, row)
                second, fourth = moments(scenario, positive, pop, row)
                positives.append(
                    Cell(f"{label}__correctly_specified", 1.0, se_noise(second, fourth))
                )
                if control is not None:
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
    z = float(norm.ppf(0.5 + LEVEL / 2.0))
    spread = _spread(replicates, cell.ratio, cell.noise)
    if cell.name.endswith("__inflated_se_control"):
        return float(norm.cdf((cell.ratio - BAND[1]) / spread - z))
    return float(norm.cdf((BAND[0] - cell.ratio) / spread - z))


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
    """36 positive cells, 18 inflated controls and 6 shrunken controls, all described."""
    declared = study.PROPERTY_CELLS["interval_calibration"]
    kinds = [cell.rsplit("__", 1)[1] for cell in declared]
    assert kinds.count("correctly_specified") == 36
    assert kinds.count("inflated_se_control") == 18
    assert kinds.count("shrunken_se_control") == 6
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
    assert sum(len(properties.scenario_cells(s)) for s in study.PARAMETERS) == len(declared)


# ------------------------------------------------------------------------------ (a) truths


def test_the_binary_truths_are_the_closed_forms() -> None:
    """The closed forms equal a one-dimensional quadrature over the index ``eta``."""
    x, w = _hermite(80)
    eta = math.sqrt(study.INDEX_VARIANCE) * x
    g = expit(eta)
    truth = study.truths("binary_logistic")
    assert w @ (1.0 / g + 1.0 / (1.0 - g)) == pytest.approx(truth["plugin_ate_nu2"], rel=1e-12)
    share = float(w @ g)
    assert share == pytest.approx(0.5, abs=1e-14)
    att = (share + w @ (g**2 / (1.0 - g))) / share**2
    assert att == pytest.approx(truth["plugin_att_nu2"], rel=1e-12)
    assert truth["plugin_atc_nu2"] == truth["plugin_att_nu2"]


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
    and ATC bound is 0.025, so the larger bound is checked.  The three-arm law sits at about
    7.3e-7 per row, so about 7e-4 of its replicates move one cell.  The rebuild applies the
    package's truncation rule, so such a replicate still reads its own curve.
    """
    automatic = 5.0 / (math.sqrt(study.PRIMARY_N) * math.log(study.PRIMARY_N))
    assert automatic < CONDITIONAL_BOUND
    index_sd = math.sqrt(study.INDEX_VARIANCE)
    binary = 2.0 * float(norm.cdf(logit(CONDITIONAL_BOUND) / index_sd))
    assert binary < 1e-21
    three_arm = _three_arm_clip_probability(CONDITIONAL_BOUND)
    assert three_arm < CLIP_FLOOR
    assert three_arm == pytest.approx(7.3e-7, rel=0.05)


# -------------------------------------------------------------------- (c) and (g) controls


def test_the_controls_leave_the_band_with_power() -> None:
    """(c) Every control leaves the band with power of at least 0.9 at the declared R."""
    _, controls = cells()
    assert len(controls) == 24
    for cell in controls:
        assert control_power(properties.PROPERTY_REPLICATES, cell) >= CONTROL_POWER_FLOOR, cell
    inflated = [c.ratio for c in controls if c.name.endswith("inflated_se_control")]
    assert min(inflated) == pytest.approx(1.1719, abs=1e-3)
    assert max(inflated) == pytest.approx(2.6170, abs=1e-3)


def test_the_known_mechanism_positives_have_teeth() -> None:
    """(g) The logistic term added to a known-mechanism curve reads below 0.93 with power.

    This is mutation M12 seen through the study: a package that stacked the logistic score
    on a declared mechanism would report these ratios on the ATT and ATC positives.
    """
    _, controls = cells()
    shrunken = [c for c in controls if c.name.startswith("binary_known_")]
    assert len(shrunken) == 6
    for cell in shrunken:
        assert cell.ratio < BAND[0], cell
        assert control_power(properties.PROPERTY_REPLICATES, cell) >= CONTROL_POWER_FLOOR, cell
    bounds = [c.ratio for c in shrunken if "_nu2__" not in c.name]
    assert max(bounds) == pytest.approx(0.8533, abs=1e-3)


# ---------------------------------------------------------------------- (d) the positives


def test_every_positive_cell_resolves_the_band() -> None:
    """(d) Each positive cell's 99% SE-ratio half-width is at most 0.035 at the declared R."""
    positives, _ = cells()
    z = float(norm.ppf(0.5 + LEVEL / 2.0))
    widths = {
        cell.name: z * _spread(properties.PROPERTY_REPLICATES, 1.0, cell.noise)
        for cell in positives
    }
    assert max(widths.values()) <= HALF_WIDTH_CEILING
    # The three-arm ATE's nu^2 carries the heaviest curve: E[alpha^8] grows like E[1/g^7].
    widest = max(widths, key=widths.__getitem__)
    assert widest == "multinomial_logistic_ate_2v0_nu2__correctly_specified"


# ------------------------------------------------------------------------- (f) the budget


def test_the_property_replicates_follow_the_sizing_rule() -> None:
    """(f) R reaches family-wise pass probability 0.95, and exceeds the minimal R.

    The minimal multiple of 400 is 3,600 (0.971), and 3,200 gives 0.909.  The study declares
    the plan's 10,000 instead.  A miscalibration of 2 % drops the chance at 3,600 to between
    0.12 and 0.22, and at 10,000 it stays above 0.999.  The run costs about one CPU-hour.
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
