"""The ``mtp-density-tails`` declaration: truths, routes, pass probabilities, controls and seeds.

Each truth is checked to 1e-5 by an adaptive quadrature written here.  Each declared pass
probability is recomputed from its stated inputs: the at-target calibration rule and its
extension rule, the Jeffreys beta-binomial of the probe counts (the booster's tail table and
its frozen ceilings, the 20,000 tail pilot, the cluster dose pilots), and the whole-study
product against its floors.  Each control is shown to fail its named clause on draws of the
exact law, and the route every cell's fit records is resolved without a fit.

The pilot readings (``TestThePilot``) read ``tests/canonical/mtp_density_tails/pilot.csv.gz``:
each gated calibration arm's plug-in pass probability under the extension rule, the noise
control's power, the ``ic_gap`` clauses with the tail pilot's exceedances, the cluster clauses
from the dose pilots, the booster's counts against its ceilings, the whole-study pass, each
cloud part's budget from the pilot's per-fit seconds, and the committed reported readings.
Until the pilots run, those tests skip, and the declaration commit is the first in which they
do not.  Nothing here reads a study run except ``readings.json``'s check, which skips until
the declared run has written ``properties.csv``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import beta, betabinom, binom, norm, poisson

from tests import mtp_density_tails_law as law
from tests.studies import canonical_mtp_density_tails as study
from tests.studies import mtp_density_tails_fits as fits
from tests.studies import mtp_density_tails_pilot as pilot
from tests.studies import mtp_density_tails_properties as properties
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins
Z99 = float(norm.ppf(0.995))
#: The declared floors: each gated ``ic_gap`` law and each calibration arm, the ``ic_gap``
#: family, and the whole study.  A pilot below a floor is diagnosed before the declaration.
FLOOR_LAW = FLOOR_ARM = 0.90
FLOOR_GAP_FAMILY = 0.85
FLOOR_STUDY = 0.90
#: The gated calibration pilots: nine arms, one estimand each (S's classifier arm is reported).
GATED_CALIBRATION = (
    *(f"calibration_density_{name}" for name in study.DENSITY_GATED),
    *(f"calibration_dr_{name}" for name in study.ROBUST_GATED),
    *(f"calibration_classifier_{name}" for name in study.CLASSIFIER_GATED),
    *(f"calibration_{estimator}" for estimator in study.LONGITUDINAL),
)
#: Gated calibration pilots that ended at the extension cap with no confirmed departure and
#: whose diagnosis found no defect: ``label -> the diagnosis``.  Their plug-in pass enters the
#: family and whole-study figures.  Empty until a pilot ends at the cap.
DIAGNOSED_AT_CAP: dict[str, str] = {}


def jeffreys_at_most(k: float, m_probe: int, c: int, m: int) -> float:
    """P(at most ``c`` of ``m`` draws exceed) under the Jeffreys posterior of ``k`` in ``m_probe``."""
    return float(betabinom.cdf(c, m, k + 0.5, m_probe - k + 0.5))


def arm_pass(replicates: int, band: float = 0.07) -> float:
    """An arm's SE-ratio clause at target: ``2 Phi(band sqrt(2R) - 2.576) - 1``."""
    return float(2.0 * norm.cdf(band * math.sqrt(2.0 * replicates) - Z99) - 1.0)


# ---------------------------------------------------------------------------- truths


def _w1_nodes(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of ``W1``'s law: Gauss-Hermite, or Gauss-Legendre on B and U's range."""
    if name in law.SUPPORTED:
        x, w = np.polynomial.legendre.leggauss(80)
        nodes = law.W1_BOUND * x
        weights = law.W1_BOUND * w * norm.pdf(nodes)
        return nodes, weights / weights.sum()
    x, w = np.polynomial.hermite_e.hermegauss(60)
    return x, w / w.sum()


def _inner(name: str, w1: float, w2: float) -> float:
    """``E[Q0(d(A), W) | W]`` by adaptive quadrature over the dose."""

    def integrand(a: float) -> float:
        aa = np.array([a])
        dose = law.assign(name, aa, np.array([w1]), np.array([w2]))
        q = law.q0(dose, w1, w2)[0]
        return float(q * law.pdf(name, aa, np.array([w1]), np.array([w2]))[0])

    c = float(law.centre(w1, w2))
    if name in law.SUPPORTED:
        u = float(law.upper(name, w1, w2))
        return quad(integrand, 0.0, u, points=[u - law.DELTA], limit=200)[0]
    if name == "T":
        return quad(integrand, law.LOW, law.HIGH, points=[law.HIGH - law.DELTA], limit=200)[0]
    if name == "C":
        points = [c, law.FAR]
        return quad(
            integrand, c - 15 * law.SIGMA, law.FAR + 15 * law.FAR_SD, points=points, limit=400
        )[0]
    if name == "S":
        return quad(integrand, -np.inf, np.inf, limit=400)[0]
    return quad(integrand, c - 15 * law.SIGMA, c + 15 * law.SIGMA, limit=200)[0]


@pytest.mark.parametrize("name", law.LAWS)
def test_each_truth_matches_an_independent_quadrature(name: str) -> None:
    """The law module's ``psi0`` is this quadrature's value, to 1e-5."""
    nodes, weights = _w1_nodes(name)
    value = sum(
        0.5 * weight * _inner(name, float(w1), w2)
        for w1, weight in zip(nodes, weights, strict=True)
        for w2 in (0.0, 1.0)
    )
    assert abs(value - law.TRUTH[name][0]) <= 1e-5, (value, law.TRUTH[name][0])


def test_the_efficiency_bound_of_s_matches_its_quadrature() -> None:
    """S's ``SD(D0)`` is the quadrature of ``E[r0^2] + Var(Q0(d(A), W))``, unit outcome noise."""
    x, w = np.polynomial.hermite_e.hermegauss(30)
    w = w / w.sum()
    first = second = ratio = 0.0
    for w2 in (0.0, 1.0):
        for w1, weight in zip(x, w, strict=True):
            c = float(law.centre(w1, w2))

            def pdf(a: float, c: float = c) -> float:
                return float(law.student.pdf((a - c) / law.SIGMA, law.DEGREES) / law.SIGMA)

            def q(a: float, w1: float = float(w1), w2: float = w2) -> float:
                return float(law.q0(a + law.DELTA, w1, w2))

            first += 0.5 * weight * quad(lambda a: q(a) * pdf(a), -np.inf, np.inf, limit=400)[0]
            second += (
                0.5 * weight * quad(lambda a: q(a) ** 2 * pdf(a), -np.inf, np.inf, limit=400)[0]
            )
            ratio += (
                0.5
                * weight
                * quad(lambda a: pdf(a - law.DELTA) ** 2 / pdf(a), -40, 40, limit=400)[0]
            )
    sd = math.sqrt(ratio + second - first**2)
    assert abs(sd - law.EFFICIENCY_SD["S"]) <= 1e-3, sd


def test_the_scaled_variant_truth_matches_its_closed_form() -> None:
    """``Gscale``: ``E[Q0(1.25 A, W)]`` with ``A | W ~ N(c, 1.5^2)`` in closed form."""
    x, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()
    total = 0.0
    for w2 in (0.0, 1.0):
        c = law.centre(x, w2)
        mean, second = fits.SCALE_G * c, fits.SCALE_G**2 * (c**2 + law.SIGMA**2)
        q = 0.5 * mean - 0.1 * second + x + 0.5 * w2 + 0.3 * mean * x
        total += 0.5 * float(np.sum(w * q))
    error = fits.efficiency_sd("Gscale") / math.sqrt(fits.BOUND_DRAWS)
    assert abs(fits.truth("Gscale") - total) <= 5.0 * error


def test_the_efficient_influence_function_is_centred() -> None:
    """``D0`` has mean ``psi0`` on a large draw of each law (the ratio and the map agree)."""
    for name in law.LAWS:
        frame = law.draw(name, 400_000, stream_seed(study.STUDY, "design_eif", name))
        curve = fits.eif(name, frame)
        error = law.EFFICIENCY_SD[name] / math.sqrt(len(frame))
        assert abs(curve.mean() - law.TRUTH[name][0]) <= 5.0 * error, name


# ---------------------------------------------------------------------------- routes


def test_every_cell_records_the_route_its_row_names() -> None:
    """The route guard: each fit set resolves to the route :data:`ROUTES` declares.

    The density arms request the density route; the classifier arms and the longitudinal
    arms leave ``ratio=None``, which must resolve to the classifier.  Read from each
    estimator on a draw of its own law, without a fit.
    """
    seen = set()
    for family, stream, configuration, _, _, label in properties.FIT_SETS:
        if (family, configuration, label) in seen:
            continue
        seen.add((family, configuration, label))
        payload = (family, stream, configuration, 0, 400, 400, label)
        if family == "cluster_warning":
            assert properties.ROUTES[(family, configuration)] is None
            continue
        if configuration.startswith("longitudinal"):
            frame = fits.draw_longitudinal(400, 0)
        else:
            frame = law.draw(fits.law_of(label), 400, 0)
        assert (
            properties.resolved_route(payload, frame) == properties.ROUTES[(family, configuration)]
        ), stream


def test_the_route_guard_fails_when_the_default_resolves_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: a resolution that always reads the density breaks the classifier rows."""
    import importlib

    from cleverly import _ratio_route as route_module

    def always(ratio: Any, **_: Any) -> Any:
        return route_module.RatioRoute(ratio or "density", route_module.NO_DOSE)

    for name in ("cleverly.estimators.tmle", "cleverly.longitudinal.estimator"):
        monkeypatch.setattr(importlib.import_module(name), "resolve_ratio", always)
    frame = law.draw("G", 400, 0)
    payload = ("interval_calibration", "classifier_G", "classifier", 0, 400, 400, "G")
    assert properties.resolved_route(payload, frame) == "density"
    payload = ("interval_calibration", "longitudinal_SDR", "longitudinal", 0, 400, 400, "SDR")
    assert properties.resolved_route(payload, fits.draw_longitudinal(400, 0)) == "density"


# ---------------------------------------------------------------------------- pass probabilities


def simulate_extensions(rho: float, runs: int, seed: int) -> dict[str, float]:
    """The extension rule on simulated pilots whose SE ratio reads ``rho`` with SD ``1/sqrt(2M)``.

    Extend by :data:`~tests.studies.mtp_density_tails_pilot.EXTENSION_DRAWS` while the plug-in
    is below 0.90; stop when it reaches 0.90 (clear), when the pooled 99% interval excludes 1
    (departure), or after :data:`~tests.studies.mtp_density_tails_pilot.MAX_EXTENSIONS` (cap).
    """
    rng = np.random.default_rng(seed)
    step = pilot.EXTENSION_DRAWS
    blocks = rng.normal(rho, 1.0 / math.sqrt(2.0 * step), (runs, pilot.MAX_EXTENSIONS + 1))
    outcome = np.full(runs, "", dtype=object)
    extensions = np.zeros(runs, dtype=int)
    for e in range(pilot.MAX_EXTENSIONS + 1):
        draws = pilot.CALIBRATION_DRAWS + e * step
        ratio = blocks[:, : e + 1].mean(axis=1)
        open_ = outcome == ""
        clear = open_ & (plug_in_ratio_array(ratio, properties.CALIBRATION_REPLICATES) >= FLOOR_ARM)
        departure = open_ & ~clear & (np.abs(ratio - 1.0) > Z99 / math.sqrt(2.0 * draws))
        outcome[clear] = "clear"
        outcome[departure] = "departure"
        extensions[open_] = e
        if e == pilot.MAX_EXTENSIONS:
            outcome[outcome == ""] = "cap"
    return {
        "mean": float(extensions.mean()),
        **{kind: float(np.mean(outcome == kind)) for kind in ("clear", "departure", "cap")},
    }


def plug_in_ratio_array(se_ratio: np.ndarray, replicates: int) -> np.ndarray:
    """The SE-ratio clause's plug-in pass probability at each pilot SE ratio, at ``replicates``."""
    low, high = MARGINS.calibration_se_ratio
    root = math.sqrt(2.0 * replicates)
    return np.asarray(
        norm.cdf((high - se_ratio) * root - Z99) - norm.cdf((low - se_ratio) * root + Z99)
    )


def tail_clause(exceedances: int, draws: int) -> float:
    """The 20,000 clause: no exceedance in 30 draws, from ``exceedances`` of ``draws``."""
    return jeffreys_at_most(exceedances, draws, 0, properties.GAP_DRAWS[20_000])


def c_cluster_clause(warned: int, draws: int) -> float:
    """At least 26 of 30 C draws warn, from ``warned`` of ``draws`` dose-pilot draws."""
    floor = properties.CLUSTER_C_FLOOR
    return 1.0 - float(betabinom.cdf(floor - 1, 30, warned + 0.5, draws - warned + 0.5))


def gt_cluster_clause(warned: float, draws: int) -> float:
    """At most 6 of 220 G and T draws warn, from ``warned`` of ``draws`` dose-pilot draws."""
    return jeffreys_at_most(warned, draws, properties.CLUSTER_GT_CEILING, 220)


class TestThePassProbabilities:
    def test_each_calibration_arm_is_sized_at_target(self) -> None:
        """Nine arms, one estimand each: the SE-ratio clauses pass with 0.978 at R = 3,200."""
        assert properties.CALIBRATION_REPLICATES == properties.ROBUST_REPLICATES == 3_200
        assert arm_pass(2_800) == pytest.approx(0.9922, abs=5e-4)
        assert arm_pass(2_800) ** 9 == pytest.approx(0.932, abs=5e-4)
        assert arm_pass(3_200) == pytest.approx(0.9975, abs=5e-4)
        assert arm_pass(3_200) ** 9 == pytest.approx(0.978, abs=5e-4)
        assert arm_pass(3_200) ** 9 >= 0.95
        # Had each longitudinal arm gated all seven estimands: 21 SE-ratio clauses.
        assert arm_pass(3_200) ** 21 == pytest.approx(0.949, abs=1e-3)
        assert arm_pass(3_600) ** 21 == pytest.approx(0.984, abs=5e-4)

    def test_the_nine_arms_are_the_declared_gated_arms(self) -> None:
        gated = {
            cell.cell
            for cell in properties.declared_cells()
            if cell.property == "interval_calibration" and cell.role == "positive"
        }
        assert len(gated) == len(GATED_CALIBRATION) == 9
        assert {cell.split("__")[1] for cell in gated} == set(properties.POSITIVE_SUFFIXES)
        assert "S_classifier__reported" in study.STUDY.property_cells["interval_calibration"]
        roles = {cell.cell: cell.role for cell in properties.declared_cells()}
        assert roles["S_classifier__reported"] != "positive"

    def test_a_coverage_clause_cannot_bind_at_target(self) -> None:
        """At R = 3,200 coverage has SD 0.0039 inside a band of plus or minus 0.03."""
        sd = math.sqrt(0.95 * 0.05 / 3_200)
        assert sd == pytest.approx(0.0039, abs=5e-5)
        assert 2.0 * norm.sf(0.03 / sd) < 1e-13

    def test_the_extension_rule(self) -> None:
        """The plug-in reaches 0.90 for a pilot ratio in [0.978, 1.022]; a calibrated arm
        extends 0.91 times on average and ends clear, at a departure or at the cap with 0.897,
        0.026 and 0.076."""
        grid = np.linspace(0.95, 1.05, 2001)
        clear = grid[plug_in_ratio_array(grid, properties.CALIBRATION_REPLICATES) >= FLOOR_ARM]
        assert clear.min() == pytest.approx(0.978, abs=5e-4)
        assert clear.max() == pytest.approx(1.022, abs=5e-4)
        outcome = simulate_extensions(1.0, 40_000, stream_seed(study.STUDY, "design_extend"))
        assert outcome["mean"] == pytest.approx(0.91, abs=0.02)
        assert outcome["clear"] == pytest.approx(0.897, abs=0.006)
        assert outcome["departure"] == pytest.approx(0.026, abs=0.004)
        assert outcome["cap"] == pytest.approx(0.076, abs=0.005)
        # The reserve is the cap on the four costly arms, at the pilots' smoke seconds.
        seconds = {"classifier_G": 53.4, "classifier_T": 58.2, "LTMLE": 120.8, "SDR": 252.0}
        reserve = sum(
            pilot.MAX_EXTENSIONS * pilot.EXTENSION_DRAWS * s / 3600 for s in seconds.values()
        )
        assert reserve == pytest.approx(161.4, abs=0.5)

    def test_the_tail_pilot_clause(self) -> None:
        """No exceedance in 30 draws, from ``k`` of 3,000 tail-pilot draws, k = 0 to 3."""
        draws = pilot.TAIL_DRAWS
        expected = (0.995, 0.985, 0.975, 0.966)
        for k, value in enumerate(expected):
            assert tail_clause(k, draws) == pytest.approx(value, abs=5e-4), k
        assert set(pilot.PILOTS) >= {f"ic_gap_tail_{name}_n20000" for name in study.GAP_GATED}

    def test_the_cluster_clauses_at_the_dose_probe_rates(self) -> None:
        """C warned in 299 of 300 probe draws at 20,000; G and T in none of 2,600."""
        assert c_cluster_clause(299, 300) == pytest.approx(0.99999, abs=5e-6)
        assert gt_cluster_clause(0, 2_600) >= 1.0 - 5e-7
        assert properties.CLUSTER_GT_DRAWS == 220
        # The eager control warned on 21 of 80 probe draws: it passes the clause with 4e-9.
        assert gt_cluster_clause(21, 80) < 1e-8
        assert {f"cluster_{name}_n{n}" for name, n in pilot.DOSE_SOURCES} <= set(pilot.PILOTS)

    def test_the_whole_study_passes_with_at_least_090(self) -> None:
        """0.978 (nine arms) x 0.975 (five laws, no tail exceedance) x 1.0 x 1.0 = 0.954."""
        calibration = arm_pass(properties.CALIBRATION_REPLICATES) ** len(GATED_CALIBRATION)
        gap = tail_clause(0, pilot.TAIL_DRAWS) ** len(study.GAP_GATED)
        clusters = c_cluster_clause(299, 300) * gt_cluster_clause(0, 2_600)
        whole = calibration * gap * clusters
        assert gap == pytest.approx(0.975, abs=1e-3)
        assert gap >= FLOOR_GAP_FAMILY
        assert whole == pytest.approx(0.954, abs=1e-3)
        assert whole >= FLOOR_STUDY
        # One exceedance on each of two laws.
        two = tail_clause(1, pilot.TAIL_DRAWS) ** 2 * tail_clause(0, pilot.TAIL_DRAWS) ** 3
        assert calibration * two * clusters == pytest.approx(0.935, abs=1e-3)
        text = " ".join((study.STUDY.artifacts / "regenerate.py").read_text().split())
        assert "**0.954** predictive" in text
        assert "0.85 for the ``ic_gap`` family, and 0.90 for the whole study" in text

    #: The boosting tail table: ``law -> (k above 1.0 of M, pass, plug-in, pass of none above 3.0)``.
    TABLE_5B: ClassVar[dict[str, tuple[int, int, float, float, float]]] = {
        "G": (2, 60, 0.931, 0.972, 0.866),
        "T": (2, 20, 0.597, 0.677, 0.709),
        "C": (0, 20, 0.955, 1.0, 0.709),
        "S": (0, 30, 0.979, 1.0, 0.776),
        "B": (1, 10, 0.555, 0.677, 0.582),
        "U": (0, 10, 0.878, 1.0, 0.582),
    }

    def test_table_5b_and_the_frozen_ceilings(self) -> None:
        """The ceilings the pilot module freezes are the probe's 0.95 predictive quantiles."""
        for name, (k, m_probe, value, plug, none) in self.TABLE_5B.items():
            assert jeffreys_at_most(k, m_probe, 2, 20) == pytest.approx(value, abs=5e-4), name
            assert binom.cdf(2, 20, k / m_probe) == pytest.approx(plug, abs=5e-4), name
            assert jeffreys_at_most(0, m_probe, 0, 20) == pytest.approx(none, abs=5e-4), name
            ceiling = int(betabinom.ppf(0.95, 20, k + 0.5, m_probe - k + 0.5))
            assert ceiling == pilot.BOOSTER_CEILINGS[name], name
        for name in ("G", "T", "C", "S"):
            assert jeffreys_at_most(0, 2, 1, 10) == pytest.approx(0.617, abs=5e-4), name

    def test_t_and_b_are_declared_at_risk(self) -> None:
        """The two laws whose 2,000 tail clause passes with less than 0.6 are named at risk."""
        at_risk = {
            name
            for name, (k, m, *_) in self.TABLE_5B.items()
            if jeffreys_at_most(k, m, 2, 20) < 0.6
        }
        assert at_risk == {"T", "B"}
        text = (study.STUDY.artifacts / "regenerate.py").read_text()
        assert "T and B are declared at risk" in text
        assert "three-mark misfit signature" in text


# ---------------------------------------------------------------------------- controls


def _gaps(case: str, n: int, draws: int, kind: str = "shipped") -> np.ndarray:
    out = []
    for draw in range(draws):
        frame = law.draw(
            fits.law_of(case), n, stream_seed(study.STUDY, "design_controls", case, kind, n, draw)
        )
        out.append(fits.ic_gap(case, frame, fits.fit_density(case, frame, reading_kind=kind)))
    return np.asarray(out)


def _marginal_density(name: str, w2: np.ndarray) -> Any:
    """The density at ``W1 = 0``: the both-wrong control's density that ignores ``W1``."""
    zero = np.zeros_like(w2)
    return lambda values: law.pdf(name, values, zero, w2)


class TestTheControls:
    def test_the_shrunken_control_leaves_the_band_with_any_passing_positive(self) -> None:
        """A positive whose ratio interval sits inside [0.93, 1.07] lands below 0.93 at 0.70."""
        low, high = MARGINS.calibration_se_ratio
        assert properties.SHRUNKEN_SE_FACTOR * high < low

    def test_the_dr_both_wrong_control_fails_coverage(self) -> None:
        """The density without ``W1`` and the wrong Q: the limit's bias at n = 8,000.

        The one-step limit, from the exact law on 1,000,000 draws, sits 1.6 to 2 of its own SEs
        from the truth on G and T, so coverage is near 0.55 and the clause ``coverage upper
        endpoint below 0.92`` holds at R = 1,000 with probability above 0.999.
        """
        for name in study.ROBUST_GATED:
            frame = law.draw(name, 1_000_000, stream_seed(study.STUDY, "design_dr", name))
            w1, w2, a, y = law.columns(frame)
            zero = np.zeros_like(w1)
            ratio = law.ratio_from(name, _marginal_density(name, w2), a, zero, w2)
            curve = law.wrong_q(fits.assign(name, a, w1, w2), w1, w2, frame) + ratio * (
                y - law.wrong_q(a, w1, w2, frame)
            )
            se = float(np.std(curve) / math.sqrt(properties.ROBUST_N))
            bias = abs(float(curve.mean()) - law.TRUTH[name][0]) / se
            coverage = norm.cdf(1.96 - bias) - norm.cdf(-1.96 - bias)
            assert bias > 1.5, (name, bias)
            assert _coverage_control_power(coverage, properties.CONTROL_REPLICATES) > 0.999

    def test_the_pre_fix_dr_control_fails_coverage_on_g(self) -> None:
        """The pre-fix one-step reading with the wrong Q undercovers on G at n = 8,000.

        As built its mean z is about -1.6 (coverage near 0.62); the bias's lower 95% limit over
        100 draws keeps clear of the 0.85 floor at which the control's power reaches 0.999.
        """
        z = []
        for draw in range(100):
            frame = law.draw(
                "G", properties.ROBUST_N, stream_seed(study.STUDY, "design_prefix", draw)
            )
            estimate, se = fits.pre_fix_one_step("G", frame)
            z.append((estimate - law.TRUTH["G"][0]) / se)
        # The bias's lower 95% limit over the 100 draws, read as the control's coverage.
        bias = abs(float(np.mean(z))) - 2.0 * float(np.std(z, ddof=1)) / math.sqrt(len(z))
        coverage = norm.cdf(1.96 - bias) - norm.cdf(-1.96 - bias)
        assert bias > 0.85, z
        assert _coverage_control_power(coverage, properties.CONTROL_REPLICATES) > 0.999

    def test_the_pre_fix_reading_fails_the_small_sample_bound(self) -> None:
        for name in study.GAP_GATED:
            gaps = _gaps(name, 2_000, 3, "pre_fix")
            assert np.median(gaps) > properties.GAP_BOUND_SMALL[name], (name, gaps)

    def test_the_pre_fix_reading_fails_the_fall_on_g_and_the_large_bound_on_c(self) -> None:
        small = _gaps("G", 2_000, 3, "pre_fix")
        large = _gaps("G", 20_000, 1, "pre_fix")
        assert np.median(large) > properties.GAP_FALL * np.median(small), (small, large)
        assert _gaps("C", 20_000, 1, "pre_fix").max() > properties.GAP_BOUND_LARGE["C"]

    @pytest.mark.parametrize("name", law.SUPPORTED)
    def test_the_support_controls_fail_their_clauses(self, name: str) -> None:
        ignored = _gaps(name, 20_000, 1, "support_ignored")
        assert ignored.max() > properties.GAP_BOUND_LARGE[name], ignored
        broken = _gaps(name, 2_000, 3, "no_row_break")
        assert np.median(broken) > properties.GAP_BOUND_SMALL[name], broken

    def test_the_cluster_controls_fail_their_clauses(self) -> None:
        """The gap-disabled helper never warns on C; the eager one warns on G at a rate whose
        Jeffreys posterior passes the G and T clause with probability below 0.01."""
        for draw in range(5):
            doses = law.draw("C", 20_000, stream_seed(study.STUDY, "design_cluster", "C", draw))
            assert fits.warns(doses["A"])
            assert not fits.warns_gap_disabled(doses["A"])
        warned = 0
        total = 0
        for name, n in properties.CLUSTER_SOURCES:
            for draw in range(10):
                frame = law.draw(name, n, stream_seed(study.STUDY, "design_cluster", name, n, draw))
                warned += fits.warns_eager(frame["A"])
                total += 1
        assert jeffreys_at_most(warned, total, properties.CLUSTER_GT_CEILING, 220) < 0.01

    #: The signature control's fixed window of draws, declared before any fit is read.
    SIGNATURE_WINDOW = range(12)
    #: The fewest breaches the broken reading must make in the window.
    SIGNATURE_BREACHES = 3

    def test_the_signature_fails_a_broken_reading(self) -> None:
        """On draws 0 to 11 of the control's stream, the booster read with occupancy from the
        wrong sample breaches (ratio error above 1.0) at least 3 times, and every breach fails
        at least one mark.

        The claim is about breaches, so the window may hold non-breaching draws; they are not
        read.  At the declaration the window held 4 breaches (draws 0, 4, 8 and 10), each failing
        the exact-probability mark.
        """
        breaches = []
        for draw in self.SIGNATURE_WINDOW:
            frame = law.draw("G", 2_000, stream_seed(study.STUDY, "design_signature", draw))
            other = law.draw("G", 2_000, stream_seed(study.STUDY, "design_signature_other", draw))
            fitted = fits.learned_density("G", frame, "boosting", occupancy_from=other)
            if fitted.error > pilot.TAIL_SMALL:
                breaches.append(draw)
                assert not fits.signature("G", frame, fitted).learner_limitation, draw
        assert len(breaches) >= self.SIGNATURE_BREACHES, breaches

    def test_the_noise_control_power_bound(self) -> None:
        """At 1,000 control draws the noise control fails the band with power 0.99 while the
        arm's SD is below about 1.52 efficiency-bound SEs, and with power 0.5 below 1.85."""
        assert noise_detection_bound(properties.CONTROL_REPLICATES) == pytest.approx(
            1.523, abs=2e-3
        )
        assert noise_detection_bound(properties.CONTROL_REPLICATES, power=0.5) == pytest.approx(
            1.847, abs=2e-3
        )
        assert noise_detection_bound(properties.CALIBRATION_REPLICATES, power=0.5) == pytest.approx(
            2.077, abs=2e-3
        )


#: The noise control's declared detection power.
NOISE_POWER = 0.99


def noise_detection_bound(replicates: int, power: float = NOISE_POWER) -> float:
    """The arm SD, in efficiency-bound SEs, below which the noise control fails the SE-ratio
    band with probability ``power`` at ``replicates`` control draws.

    With a calibrated arm of SD ``t`` bound SEs, the noise control's SE ratio is
    ``rho = t / sqrt(1 + t^2)``.  Its estimate reads ``rho`` with SD about ``rho / sqrt(2R)``,
    and its 99% interval leaves the band below 0.93 when the estimate ``r`` has
    ``r (1 + 2.576 / sqrt(2R)) < 0.93``.  That happens with probability ``power`` when
    ``rho (1 + z_power / sqrt(2R)) (1 + 2.576 / sqrt(2R)) <= 0.93``.  At ``power = 0.5`` this is
    the half-power bound, 1.85 at R = 1,000; at 0.99 it is 1.52.
    """
    low, _ = MARGINS.calibration_se_ratio
    root = math.sqrt(2.0 * replicates)
    z = float(norm.ppf(power))
    ratio = low / ((1.0 + Z99 / root) * (1.0 + z / root))
    return float(ratio / math.sqrt(1.0 - ratio**2))


def _coverage_control_power(coverage: float, replicates: int) -> float:
    """P(the exact 99% upper endpoint of coverage lies below 0.92) at ``replicates`` draws."""
    floor = MARGINS.calibration_coverage[0]
    alpha = 1.0 - MARGINS.confidence_level
    counts = np.arange(replicates + 1)
    upper = np.where(
        counts == replicates, 1.0, beta.ppf(1 - alpha / 2, counts + 1, replicates - counts)
    )
    return float(binom.pmf(counts, replicates, coverage)[upper < floor].sum())


# ---------------------------------------------------------------------------- cells and seeds


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }
        written = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
        filled = {
            name
            for key in properties._payloads(properties.FIT_SETS, 1)
            for name in properties.cells_of(key)
        }
        assert written == declared
        assert filled == declared

    def test_the_cluster_cells_read_the_ic_gap_draws(self) -> None:
        for replicate in (0, 79, 80, 159, 160, 189, 190, 219):
            name, n, source = properties.cluster_source(replicate)
            key = (("cluster_warning", "GT_pooled", "cluster", replicate, 2_000, 220, "GT"),)
            gap = (("ic_gap", f"{name}_n_{n}", "shipped", source, n, 0, name),)
            pd.testing.assert_frame_equal(properties._draw(key), properties._draw(gap))

    #: The smoke's per-fit seconds of each stream a cloud part runs.
    SMOKE_SECONDS: ClassVar[dict[str, float]] = {
        "classifier_G": 53.09,
        "classifier_T": 60.91,
        "classifier_S": 49.98,
        "classifier_dr_G": 54.15,
        "classifier_dr_T": 61.09,
        "classifier_dr_G_both_wrong": 79.90,
        "control_density_G": 0.12,
        "control_classifier_G": 53.09,
        "longitudinal_LTMLE": 121.33,
        "control_LTMLE": 121.33,
        "longitudinal_SDR": 244.49,
    }

    def test_the_eleven_cloud_parts(self) -> None:
        """Each part is at most 55 CPU-h at the smoke's times, and the parts cover each heavy
        stream's replicates once."""
        assert len(properties.RUNNER_PARTS) == 11
        hours = {}
        for part, (streams, span) in properties.RUNNER_PARTS.items():
            hours[part] = sum(
                len(range(properties.stream_replicates(stream)) if span is None else span)
                * self.SMOKE_SECONDS[stream]
                / 3600
                for stream in streams
            )
            assert hours[part] <= properties.PART_HOURS, (part, hours[part])
        assert max(hours.values()) == pytest.approx(54.3, abs=0.05)
        assert sum(hours.values()) == pytest.approx(541.0, abs=0.5)
        for stream in self.SMOKE_SECONDS:
            covered = sorted(r for span in properties.part_ranges(stream) for r in span)
            assert covered == list(range(properties.stream_replicates(stream))), stream

    def test_the_seeds_are_distinct(self) -> None:
        """Every declared seed at its full count, each calibration pilot with its three
        extensions, draws a distinct sample, and 32-bit coincidences are no more than chance.

        Within the study no seed repeats on the same law and size, so no two cells or pilots
        share a sample; a coincidence across laws or sizes draws a different sample.  Against
        every other registered study's primary seeds at their full counts, and within the study,
        the number of coincident values is below the Poisson 0.9999 quantile of chance.
        """
        records = {record.slug: record for record in registered()}
        others = [record for slug, record in records.items() if slug != study.STUDY.slug]
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}
        drawn: list[tuple[str, int, int]] = [
            (scenario, study.PRIMARY_N, replicate_seed(study.STUDY, scenario, r))
            for scenario in study.SCENARIOS
            for r in range(study.PRIMARY_REPLICATES)
        ]
        for family, stream, configuration, n, replicates, label in properties.FIT_SETS:
            if family == "cluster_warning":
                continue  # reads the ic_gap draws by design
            sample = "longitudinal" if configuration.startswith("longitudinal") else label
            drawn += [
                (fits.law_of(sample) if sample != "longitudinal" else sample, n, seed)
                for seed in (
                    stream_seed(study.STUDY, "property_sample", family, stream, r)
                    for r in range(replicates)
                )
            ]
        reserve = pilot.MAX_EXTENSIONS * pilot.EXTENSION_DRAWS
        for label, (kind, case, n, draws) in pilot.PILOTS.items():
            sample = "longitudinal" if kind == "longitudinal" else fits.law_of(case)
            count = draws + (reserve if label.startswith("calibration_") else 0)
            drawn += [(sample, n, pilot.seed(label, r)) for r in range(count)]
        assert len(set(drawn)) == len(drawn)
        seeds = [seed for _, _, seed in drawn]
        within = len(seeds) - len(set(seeds))
        assert within <= poisson.ppf(0.9999, len(seeds) ** 2 / 2.0**33), within
        noise = [
            stream_seed(study.STUDY, "efficiency_noise", prefix)
            for prefix in properties.efficiency_bounds()
        ]
        assert not set(noise) & set(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(record.replicates)
        }
        expected = len(set(seeds)) * len(foreign) / 2.0**32
        assert len(foreign & set(seeds)) <= poisson.ppf(0.9999, expected)


# ---------------------------------------------------------------------------- the pilot


@pytest.fixture(scope="module")
def pilot_rows() -> pd.DataFrame:
    if not pilot.OUTPUT.exists():
        pytest.skip("the pilots have not run; the declaration commit runs them first")
    return pilot.load()


def calibration_plug_in(se_ratio: float, coverage: float, replicates: int) -> float:
    """The plug-in pass probability of one arm at its pilot SE ratio and coverage."""
    low, high = MARGINS.calibration_se_ratio
    sd = se_ratio / math.sqrt(2.0 * replicates)
    ratio = norm.cdf((high - se_ratio) / sd - Z99) - norm.cdf((low - se_ratio) / sd + Z99)
    c_low, c_high = MARGINS.calibration_coverage
    c_sd = math.sqrt(coverage * (1 - coverage) / replicates)
    cover = norm.cdf((c_high - coverage) / c_sd - Z99) - norm.cdf((c_low - coverage) / c_sd + Z99)
    return float(max(ratio, 0.0) * max(cover, 0.0))


def pooled_calibration(rows: pd.DataFrame, truth: float) -> tuple[float, float, int]:
    """A pilot's pooled SE ratio, coverage and draw count; coverage at the symmetric interval.

    Coverage is recomputed from each row's estimate and SE against the law module's truth, so
    it does not depend on the truth a checkpoint was written with.
    """
    estimate = rows["estimate"].to_numpy(dtype=float)
    se = rows["std_error"].to_numpy(dtype=float)
    ratio = float(se.mean() / estimate.std(ddof=1))
    coverage = float(np.mean(np.abs(estimate - truth) <= properties.CRITICAL * se))
    return ratio, coverage, len(rows)


def departure(rows: pd.DataFrame, truth: float, label: str) -> bool:
    """A confirmed departure: the pooled 99% bootstrap interval of the SE ratio excludes 1, or
    the exact 99% interval of coverage excludes 0.95."""
    estimate = rows["estimate"].to_numpy(dtype=float)
    se = rows["std_error"].to_numpy(dtype=float)
    rng = np.random.default_rng(stream_seed(study.STUDY, "design_departure", label))
    index = rng.integers(0, len(rows), (2_000, len(rows)))
    ratios = se[index].mean(axis=1) / estimate[index].std(axis=1, ddof=1)
    low, high = np.quantile(ratios, [0.005, 0.995])
    covered = int(np.sum(np.abs(estimate - truth) <= properties.CRITICAL * se))
    c_low = beta.ppf(0.005, covered, len(rows) - covered + 1) if covered else 0.0
    c_high = beta.ppf(0.995, covered + 1, len(rows) - covered) if covered < len(rows) else 1.0
    return bool(low > 1.0 or high < 1.0 or c_low > 0.95 or c_high < 0.95)


def _pilot_truth(label: str) -> float:
    kind, case, _, _ = pilot.PILOTS[label]
    if kind == "longitudinal":
        return float(fits.LONGITUDINAL_TRUTH[fits.LONGITUDINAL_GATED])
    return float(fits.truth(case))


def _calibration_rows(pilot_rows: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = pilot_rows.loc[(pilot_rows["label"] == label) & (pilot_rows["refusal"].isna())]
    if pilot.PILOTS[label][0] == "longitudinal":
        rows = rows.loc[rows["reading"] == fits.LONGITUDINAL_GATED]
    return rows


def arm_reading(pilot_rows: pd.DataFrame, label: str) -> float:
    """The arm's pass probability under the extension rule, or a failed assertion.

    An arm whose plug-in reaches 0.90 enters at target; an arm at the cap enters at its plug-in
    once :data:`DIAGNOSED_AT_CAP` records a diagnosis with no defect.  A confirmed departure, or
    an arm below 0.90 with an extension left, stops the declaration.
    """
    rows = _calibration_rows(pilot_rows, label)
    truth = _pilot_truth(label)
    ratio, coverage, draws = pooled_calibration(rows, truth)
    extensions = (draws - pilot.CALIBRATION_DRAWS) // pilot.EXTENSION_DRAWS
    assert draws == pilot.CALIBRATION_DRAWS + extensions * pilot.EXTENSION_DRAWS, (label, draws)
    assert 0 <= extensions <= pilot.MAX_EXTENSIONS, (label, draws)
    plug = calibration_plug_in(ratio, coverage, properties.CALIBRATION_REPLICATES)
    if plug >= FLOOR_ARM:
        return arm_pass(properties.CALIBRATION_REPLICATES)
    assert not departure(rows, truth, label), f"{label}: a confirmed departure; diagnose it"
    assert extensions == pilot.MAX_EXTENSIONS, f"{label}: plug-in {plug:.3f}; extend by 400"
    assert label in DIAGNOSED_AT_CAP, f"{label}: at the cap; diagnose it before the declaration"
    return plug


def _pilot_gaps(pilot_rows: pd.DataFrame, label: str) -> np.ndarray:
    rows = pilot_rows.loc[(pilot_rows["label"] == label) & (pilot_rows["reading"] == "ic_gap")]
    return rows["ratio_error"].to_numpy(dtype=float)


def gap_pass(pilot_rows: pd.DataFrame, name: str) -> float:
    """A gated law's three ``ic_gap`` clauses at the planned counts, from its pilots.

    The fall and the 2,000 bound resample the 80 and 30 pilot gaps; the 20,000 clause is the
    Jeffreys beta-binomial of the tail pilot's exceedances (:func:`tail_clause`).
    """
    small = _pilot_gaps(pilot_rows, f"ic_gap_{name}_n2000")
    large = _pilot_gaps(pilot_rows, f"ic_gap_{name}_n20000")
    tail = _pilot_gaps(pilot_rows, f"ic_gap_tail_{name}_n20000")
    assert small.size == pilot.GAP_DRAWS[2_000] and large.size == pilot.GAP_DRAWS[20_000], name
    assert tail.size == pilot.TAIL_DRAWS, name
    rng = np.random.default_rng(stream_seed(study.STUDY, "design_gap_pass", name))
    draws = 4_000
    m_small = np.median(rng.choice(small, (draws, properties.GAP_DRAWS[2_000])), axis=1)
    m_large = np.median(rng.choice(large, (draws, properties.GAP_DRAWS[20_000])), axis=1)
    fall = float(np.mean(m_large <= properties.GAP_FALL * m_small))
    bound = float(np.mean(m_small <= properties.GAP_BOUND_SMALL[name]))
    exceedances = int(np.sum(tail > properties.GAP_BOUND_LARGE[name]))
    return fall * bound * tail_clause(exceedances, tail.size)


def cluster_pass(pilot_rows: pd.DataFrame) -> float:
    """The two cluster clauses from the dose pilots; each control is read on the same doses."""

    def count(label: str, reading: str) -> tuple[int, int]:
        rows = pilot_rows.loc[(pilot_rows["label"] == label) & (pilot_rows["reading"] == reading)]
        return int(rows["estimate"].sum()), len(rows)

    c_warned, c_total = count("cluster_C_n20000", "cluster")
    assert c_total == pilot.DOSE_DRAWS
    c_pass = c_cluster_clause(c_warned, c_total)

    def pooled(reading: str) -> tuple[float, int]:
        """The G and T warning count, each dose pilot weighted by its declared share of the
        220 draws (80, 80, 30 and 30), on the pilots' pooled draw count."""
        rate, total = 0.0, 0
        for name, n in properties.CLUSTER_SOURCES:
            warned, draws = count(f"cluster_{name}_n{n}", reading)
            assert draws == pilot.DOSE_DRAWS, (name, n)
            rate += properties.GAP_DRAWS[n] / properties.CLUSTER_GT_DRAWS * warned / draws
            total += draws
        return rate * total, total

    gt_pass = gt_cluster_clause(*pooled("cluster"))
    disabled, _ = count("cluster_C_n20000", "cluster_gap_disabled")
    assert c_cluster_clause(disabled, c_total) < 0.01
    assert gt_cluster_clause(*pooled("cluster_eager")) < 0.01
    return c_pass * gt_pass


class TestThePilot:
    def test_each_calibration_arm_passes_with_090_or_is_extended(
        self, pilot_rows: pd.DataFrame
    ) -> None:
        family = 1.0
        for label in GATED_CALIBRATION:
            rows = _calibration_rows(pilot_rows, label)
            truth = _pilot_truth(label)
            spread = rows["estimate"].std() / math.sqrt(len(rows))
            assert abs(rows["estimate"].mean() - truth) < 5 * spread, label
            family *= arm_reading(pilot_rows, label)
        assert family >= 0.95 or DIAGNOSED_AT_CAP, family

    def test_the_noise_control_has_power_on_each_controlled_arm(
        self, pilot_rows: pd.DataFrame
    ) -> None:
        """Each controlled arm's pilot SD is below the noise control's detection bound."""
        bound = noise_detection_bound(properties.CONTROL_REPLICATES)
        arms = {
            "calibration_density_G": "G",
            "calibration_classifier_G": "G_classifier",
            "calibration_LTMLE": properties.LTMLE_GATED,
        }
        for label, prefix in arms.items():
            rows = _calibration_rows(pilot_rows, label)
            unit = properties.efficiency_bounds()[prefix] / math.sqrt(properties.CALIBRATION_N)
            spread = float(rows["estimate"].std(ddof=1)) / unit
            assert spread < bound, (label, spread, bound)

    def test_the_ic_gap_clauses_pass_with_090(self, pilot_rows: pd.DataFrame) -> None:
        family = 1.0
        for name in study.GAP_GATED:
            value = gap_pass(pilot_rows, name)
            assert value >= FLOOR_LAW, (name, value)
            family *= value
        assert family >= FLOOR_GAP_FAMILY

    def test_the_cluster_clauses_from_the_dose_pilots(self, pilot_rows: pd.DataFrame) -> None:
        assert cluster_pass(pilot_rows) >= FLOOR_LAW

    def test_the_whole_study_passes_with_090(self, pilot_rows: pd.DataFrame) -> None:
        calibration = 1.0
        for label in GATED_CALIBRATION:
            calibration *= arm_reading(pilot_rows, label)
        gap = 1.0
        for name in study.GAP_GATED:
            gap *= gap_pass(pilot_rows, name)
        whole = calibration * gap * cluster_pass(pilot_rows)
        assert whole >= FLOOR_STUDY, (calibration, gap, whole)

    def test_the_booster_counts_against_the_frozen_ceilings(self, pilot_rows: pd.DataFrame) -> None:
        """The ceilings are read, not recomputed: a count above one reopens that law's grid."""
        readings = pilot.reported_readings(pilot_rows, pd.DataFrame(columns=["property", "cell"]))
        for name in law.LAWS:
            reading = readings["boosting"][name]
            assert reading["tail_2000"]["draws"] == 20, name
            assert reading["ceiling"] == pilot.BOOSTER_CEILINGS[name]
            assert not reading["reopened"], (name, reading["above_1"])
        for name in law.SUPPORTED:
            assert not readings["learned"][name]["reopened"], name

    def test_each_cloud_part_is_rebudgeted_from_the_pilot(self, pilot_rows: pd.DataFrame) -> None:
        """Each cloud part stays at most 55 CPU-h at the pilot's median per-fit seconds."""
        pilot_of = {
            "classifier_G": "calibration_classifier_G",
            "classifier_T": "calibration_classifier_T",
            "classifier_S": "calibration_classifier_S",
            "control_classifier_G": "calibration_classifier_G",
            "longitudinal_LTMLE": "calibration_LTMLE",
            "control_LTMLE": "calibration_LTMLE",
            "longitudinal_SDR": "calibration_SDR",
        }
        seconds = dict(TestTheCells.SMOKE_SECONDS)
        for stream, label in pilot_of.items():
            rows = pilot_rows.loc[pilot_rows["label"] == label]
            seconds[stream] = float(rows.drop_duplicates("replicate")["seconds"].median())
        for part, (streams, span) in properties.RUNNER_PARTS.items():
            hours = sum(
                len(range(properties.stream_replicates(stream)) if span is None else span)
                * seconds[stream]
                / 3600
                for stream in streams
            )
            assert hours <= properties.PART_HOURS, (part, hours)


def test_the_committed_readings_match_their_recomputation() -> None:
    """``readings.json`` is :func:`reported_readings` of the committed pilot and properties."""
    if not (pilot.READINGS.exists() and pilot.OUTPUT.exists() and pilot.PROPERTIES.exists()):
        pytest.skip("the readings are written after the declared run")
    recomputed = pilot.reported_readings(pilot.load(), pd.read_csv(pilot.PROPERTIES))
    assert json.loads(json.dumps(recomputed)) == json.loads(pilot.READINGS.read_text())


def test_a_later_pilot_set_supersedes_a_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draw a later set repeats keeps its earlier rows, marked, and reads only the new ones."""
    first, later = tmp_path / "pilot-checkpoints", tmp_path / "pilot-checkpoints-c"
    for directory, estimates in ((first, (1.0, 2.0)), (later, (3.0,))):
        directory.mkdir()
        rows = [
            dict.fromkeys(pilot.COLUMNS)
            | {"label": "x", "replicate": r, "reading": "a", "estimate": e}
            for r, e in enumerate(estimates)
        ]
        (directory / "x.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(pilot, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(pilot, "CHECKPOINTS", first)
    folded = pilot.collect(first, tmp_path / "pilot.csv.gz")
    assert folded["superseded"].tolist() == [True, False, False]
    current = pilot.load(tmp_path / "pilot.csv.gz")
    assert current.sort_values("replicate")["estimate"].tolist() == [3.0, 2.0]


def _synthetic_pilot() -> pd.DataFrame:
    """Pilot rows built to give known readings: G breaches twice, one breach without a mark."""
    rows: list[dict[str, Any]] = []

    def add(label: str, n: int, reading: str, errors: list[float], marks: int = 1) -> None:
        for r, error in enumerate(errors):
            rows.append(
                {
                    "label": label,
                    "replicate": r,
                    "n": n,
                    "reading": reading,
                    "ratio_error": error,
                    "mark_exact": marks,
                    "mark_top5": 1,
                    "mark_logp": 1,
                }
            )

    quiet = [0.2] * 20
    for name in law.LAWS:
        errors = [1.5, 1.2, *[0.3] * 18] if name == "G" else [0.3] * 20
        add(f"boosting_{name}_n2000", 2_000, "boosting", errors)
        add(f"boosting_{name}_n2000", 2_000, "boosting_pre_fix", [0.9] * 20)
    for name in pilot.LEARNED_LAWS:
        add(f"boosting_{name}_n20000", 20_000, "boosting", [0.15] * 10)
        add(f"stacked_{name}", 2_000, "stacked", [0.3] * 5)
    for name in law.SUPPORTED:
        label = f"learned_{name}"
        errors = [0.3] * 20 if name == "B" else [*[1.1] * 5, *[0.3] * 15]
        add(label, 2_000, "boosting", errors, marks=0 if name == "U" else 1)
        add(label, 2_000, "boosting_pre_fix", [0.9] * 20)
        add(label, 2_000, "boosting_support_ignored", [0.31 if name == "B" else 0.2] * 20)
        add(label, 2_000, "pooled_logistic", quiet)
        add(label, 2_000, "pooled_logistic_pre_fix", [0.5] * 20)
        add(label, 2_000, "pooled_logistic_support_ignored", quiet)
    for case in pilot.CLASSIFIER_CASES:
        add(
            f"classifier_{case}_n2000",
            2_000,
            "with_steps",
            [0.5 if case in ("Gcap", "Bp2") else 1.0],
        )
        add(f"classifier_{case}_n2000", 2_000, "without_steps", [1.0])
    add("classifier_T_n20000", 20_000, "with_steps", [0.7])
    add("classifier_T_n20000", 20_000, "without_steps", [1.0])
    return pd.DataFrame(rows)


def test_reported_readings_on_synthetic_rows() -> None:
    """Every reading on rows built to give it: the fall, the ceilings, the routing, the
    support readings, the steps readings and the reported calibration rows."""
    summary = pd.DataFrame(
        [
            {
                "property": "interval_calibration",
                "cell": "S__reported",
                "se_ratio": 1.0,
                "se_ratio_ci_lower": 0.96,
                "se_ratio_ci_upper": 1.04,
                "coverage": 0.95,
                "coverage_ci_lower": 0.94,
                "coverage_ci_upper": 0.96,
            },
            {
                "property": "interval_calibration",
                "cell": "S_classifier__reported",
                "se_ratio": 0.9,
                "se_ratio_ci_lower": 0.86,
                "se_ratio_ci_upper": 0.94,
                "coverage": 0.93,
                "coverage_ci_lower": 0.92,
                "coverage_ci_upper": 0.94,
            },
            {
                "property": "interval_calibration",
                "cell": "T_classifier_dr__reported",
                "bias": -0.4,
                "empirical_se": 1.0,
                "remainder_proxy": -0.4,
                "coverage": 0.88,
            },
        ]
    )
    readings = pilot.reported_readings(_synthetic_pilot(), summary)
    boosting = readings["boosting"]
    assert boosting["G"]["above_1"] == 2 and boosting["G"]["tail_2000"]["holds"]
    assert not boosting["G"]["reopened"]
    assert boosting["G"]["fall"]["holds"]  # 0.15 / 0.3 = 0.5 <= 2/3
    assert boosting["C"]["fall"]["ratio"] == pytest.approx(0.5)
    learned = readings["learned"]
    assert learned["U"]["above_1"] == 5 and learned["U"]["reopened"]  # ceiling 4
    assert not learned["B"]["reopened"]
    assert learned["B"]["boosting"]["holds"]  # 0.3 < 0.9 and 0.3 <= 1.1 x 0.31
    assert not learned["U"]["boosting"]["holds"]  # 0.3 > 1.1 x 0.2
    routes = {(b["label"], b["replicate"]): b["route"] for b in readings["breaches"]}
    assert routes[("boosting_G_n2000", 0)] == "learner_limitation"
    assert routes[("learned_U", 0)] == "defect_suspect"  # mark_exact is 0
    assert len(routes) == 2 + 5
    steps = readings["classifier_steps"]
    assert steps["T"]["holds"]  # 0.7 <= 0.8 and 1.0 >= 0.9
    assert steps["Gcap"]["holds"] and steps["Bp2"]["holds"] and steps["G"]["holds"]
    calibration = readings["reported_calibration"]
    assert calibration["S__reported"]["inside_band"]
    assert not calibration["S_classifier__reported"]["inside_band"]
    dr = readings["classifier_dr"]["T_classifier_dr__reported"]
    assert dr["holds"] and dr["coverage_below_090"]
    assert readings["stacked"]["G"]["holds"]
    assert json.loads(json.dumps(readings)) == readings
