"""The ``mtp-density-tails`` declaration: truths, routes, pass probabilities, controls and seeds.

Each truth is checked by an adaptive quadrature written here, against the Monte Carlo value the
law module carries.  Each declared pass probability, the boosting tail table among them, is
recomputed from its stated inputs: the at-target calibration rule, the Jeffreys beta-binomial
of the probe counts, and the cluster rates.  Each control is shown to fail its named clause on
draws of the exact law, and the route every cell's fit records is resolved without a fit.

The pilot readings (``TestThePilot``) read ``tests/canonical/mtp_density_tails/pilot.csv.gz``:
the calibration arms' plug-in pass probabilities and their extensions, the ``ic_gap`` clauses'
plug-in and predictive pass probabilities, the cluster clauses from the pilot's doses, the tail table
and the booster's ceilings from the pilot's draws, and each cell's budget from the pilot's
per-fit seconds.  Until the pilots run, those tests skip, and the declaration commit is the
first in which they do not.  Nothing here reads a study run.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import beta, betabinom, binom, chi2, norm

from tests import mtp_density_tails_law as law
from tests.studies import canonical_mtp_density_tails as study
from tests.studies import mtp_density_tails_fits as fits
from tests.studies import mtp_density_tails_pilot as pilot
from tests.studies import mtp_density_tails_properties as properties
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins
Z99 = float(norm.ppf(0.995))


def jeffreys_at_most(k: int, m_probe: int, c: int, m: int) -> float:
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
    """The law module's ``psi0`` is a 4,000,000-draw mean: within 5 Monte Carlo SEs of it."""
    nodes, weights = _w1_nodes(name)
    value = sum(
        0.5 * weight * _inner(name, float(w1), w2)
        for w1, weight in zip(nodes, weights, strict=True)
        for w2 in (0.0, 1.0)
    )
    error = law.EFFICIENCY_SD[name] / math.sqrt(4_000_000)
    assert abs(value - law.TRUTH[name][0]) <= 5.0 * error, (value, law.TRUTH[name][0])


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
        if configuration == "longitudinal":
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


class TestThePassProbabilities:
    def test_each_calibration_arm_is_sized_at_target(self) -> None:
        """Ten arms, one estimand each: the SE-ratio clauses pass with 0.975 at R = 3,200."""
        assert properties.CALIBRATION_REPLICATES == properties.ROBUST_REPLICATES == 3_200
        assert arm_pass(2_800) == pytest.approx(0.9922, abs=5e-4)
        assert arm_pass(2_800) ** 10 == pytest.approx(0.925, abs=5e-4)
        assert arm_pass(3_200) == pytest.approx(0.9975, abs=5e-4)
        assert arm_pass(3_200) ** 10 == pytest.approx(0.975, abs=5e-4)
        assert arm_pass(3_200) ** 10 >= 0.95
        # Had each longitudinal arm gated all seven estimands: 22 SE-ratio clauses.
        assert arm_pass(3_200) ** 22 == pytest.approx(0.946, abs=1e-3)
        assert arm_pass(3_600) ** 22 == pytest.approx(0.983, abs=5e-4)

    def test_the_ten_arms_are_the_declared_gated_arms(self) -> None:
        gated = {
            cell.cell
            for cell in properties.declared_cells()
            if cell.property == "interval_calibration" and cell.role == "positive"
        }
        assert len(gated) == 10
        assert {cell.split("__")[1] for cell in gated} == set(properties.POSITIVE_SUFFIXES)

    def test_a_coverage_clause_cannot_bind_at_target(self) -> None:
        """At R = 3,200 coverage has SD 0.0039 inside a band of plus or minus 0.03."""
        sd = math.sqrt(0.95 * 0.05 / 3_200)
        assert sd == pytest.approx(0.0039, abs=5e-5)
        assert 2.0 * norm.sf(0.03 / sd) < 1e-13

    def test_the_cluster_clauses_at_the_probe_rates(self) -> None:
        """C warned in 20 of 20 probe draws at 20,000; G and T in none of 80."""
        c_pass = 1.0 - float(betabinom.cdf(properties.CLUSTER_C_FLOOR - 1, 30, 20.5, 0.5))
        gt_pass = jeffreys_at_most(0, 80, properties.CLUSTER_GT_CEILING, 220)
        assert c_pass == pytest.approx(0.977, abs=5e-4)
        assert gt_pass == pytest.approx(0.961, abs=5e-4)
        assert c_pass * gt_pass == pytest.approx(0.939, abs=1e-3)
        assert properties.CLUSTER_GT_DRAWS == 220
        # The eager control warned on 21 of 80 probe draws: it passes the clause with 4e-9.
        assert jeffreys_at_most(21, 80, properties.CLUSTER_GT_CEILING, 220) < 1e-8

    #: The boosting tail table: ``law -> (k above 1.0 of M, pass, plug-in, pass of none above 3.0)``.
    TABLE_5B: ClassVar[dict[str, tuple[int, int, float, float, float]]] = {
        "G": (2, 60, 0.931, 0.972, 0.866),
        "T": (2, 20, 0.597, 0.677, 0.709),
        "C": (0, 20, 0.955, 1.0, 0.709),
        "S": (0, 30, 0.979, 1.0, 0.776),
        "B": (1, 10, 0.555, 0.677, 0.582),
        "U": (0, 10, 0.878, 1.0, 0.582),
    }
    #: The 0.95 predictive ceilings of the 2,000 tail clause, out of 20 draws.
    CEILINGS: ClassVar[dict[str, int]] = {"G": 3, "T": 6, "C": 2, "S": 2, "B": 8, "U": 4}

    def test_table_5b(self) -> None:
        for name, (k, m_probe, value, plug, none) in self.TABLE_5B.items():
            assert jeffreys_at_most(k, m_probe, 2, 20) == pytest.approx(value, abs=5e-4), name
            assert binom.cdf(2, 20, k / m_probe) == pytest.approx(plug, abs=5e-4), name
            assert jeffreys_at_most(0, m_probe, 0, 20) == pytest.approx(none, abs=5e-4), name
            ceiling = int(betabinom.ppf(0.95, 20, k + 0.5, m_probe - k + 0.5))
            assert ceiling == self.CEILINGS[name], name
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
        """The pre-fix one-step reading with the wrong Q undercovers on G at n = 8,000."""
        z = []
        for draw in range(20):
            frame = law.draw(
                "G", properties.ROBUST_N, stream_seed(study.STUDY, "design_prefix", draw)
            )
            estimate, se = fits.pre_fix_one_step("G", frame)
            z.append((estimate - law.TRUTH["G"][0]) / se)
        # The bias's lower 95% limit over the 20 draws, read as the control's coverage.
        bias = abs(float(np.mean(z))) - 2.0 * float(np.std(z, ddof=1)) / math.sqrt(len(z))
        coverage = norm.cdf(1.96 - bias) - norm.cdf(-1.96 - bias)
        assert bias > 0.5, z
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

    def test_the_signature_fails_a_broken_reading(self) -> None:
        """Occupancy from the wrong sample fails at least one mark on every draw."""
        for draw in range(3):
            frame = law.draw("G", 2_000, stream_seed(study.STUDY, "design_signature", draw))
            other = law.draw("G", 2_000, stream_seed(study.STUDY, "design_signature_other", draw))
            fitted = fits.learned_density("G", frame, "pooled_logistic", occupancy_from=other)
            assert not fits.signature("G", frame, fitted).learner_limitation, draw


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

    def test_each_large_stream_fits_its_cloud_parts(self) -> None:
        """At the declared per-fit times (181.8 s and 368.9 s), each part is under 80 CPU-h."""
        seconds = {"longitudinal_LTMLE": 181.8, "longitudinal_SDR": 368.9}
        for stream, per_fit in seconds.items():
            for span in properties.part_ranges(stream):
                assert len(span) * per_fit / 3600 <= properties.PART_HOURS, stream
            covered = [r for span in properties.part_ranges(stream) for r in span]
            assert covered == list(range(properties.CALIBRATION_REPLICATES))

    def test_the_seeds_are_distinct(self) -> None:
        records = {record.slug: record for record in registered()}
        others = [record for slug, record in records.items() if slug != study.STUDY.slug]
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, stream, r)
            for family, stream, *_ in properties.FIT_SETS
            for r in range(50)
        ]
        seeds += [pilot.seed(label, r) for label in pilot.PILOTS for r in range(50)]
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)


# ---------------------------------------------------------------------------- the pilot


@pytest.fixture(scope="module")
def pilot_rows() -> pd.DataFrame:
    if not pilot.OUTPUT.exists():
        pytest.skip("the pilots have not run; the declaration commit runs them first")
    return pd.read_csv(pilot.OUTPUT)


def calibration_plug_in(se_ratio: float, coverage: float, replicates: int) -> float:
    """The plug-in pass probability of one arm at its pilot SE ratio and coverage."""
    low, high = MARGINS.calibration_se_ratio
    sd = se_ratio / math.sqrt(2.0 * replicates)
    ratio = norm.cdf((high - se_ratio) / sd - Z99) - norm.cdf((low - se_ratio) / sd + Z99)
    c_low, c_high = MARGINS.calibration_coverage
    c_sd = math.sqrt(coverage * (1 - coverage) / replicates)
    cover = norm.cdf((c_high - coverage) / c_sd - Z99) - norm.cdf((c_low - coverage) / c_sd + Z99)
    return float(max(ratio, 0.0) * max(cover, 0.0))


def gap_pass(small: np.ndarray, large: np.ndarray, name: str, *, predictive: bool) -> float:
    """The three ``ic_gap`` clauses' pass probability at the planned counts, from pilot gaps.

    The median clauses resample the pilot's gaps; the 20,000 clause reads a lognormal fitted to
    them, its parameters fixed (plug-in) or drawn from their posterior (predictive).
    """
    rng = np.random.default_rng(stream_seed(study.STUDY, "design_gap_pass", name, predictive))
    draws = 4_000
    m_small = np.median(rng.choice(small, (draws, properties.GAP_DRAWS[2_000])), axis=1)
    m_large = np.median(rng.choice(large, (draws, properties.GAP_DRAWS[20_000])), axis=1)
    fall = np.mean(m_large <= properties.GAP_FALL * m_small)
    bound = np.mean(m_small <= properties.GAP_BOUND_SMALL[name])
    logs = np.log(large)
    k = logs.size
    mu, s = float(logs.mean()), float(logs.std(ddof=1))
    if predictive:
        sigma = s * np.sqrt((k - 1) / chi2.rvs(k - 1, size=draws, random_state=rng))
        mean = mu + sigma / np.sqrt(k) * rng.standard_normal(draws)
    else:
        sigma, mean = np.full(draws, s), np.full(draws, mu)
    each = norm.cdf((np.log(properties.GAP_BOUND_LARGE[name]) - mean) / sigma)
    large_clause = float(np.mean(each ** properties.GAP_DRAWS[20_000]))
    return float(fall * bound * large_clause)


class TestThePilot:
    def test_each_calibration_arm_passes_with_090_or_is_extended(
        self, pilot_rows: pd.DataFrame
    ) -> None:
        for label, (kind, case, _, draws) in pilot.PILOTS.items():
            if not label.startswith("calibration_"):
                continue
            rows = pilot_rows.loc[pilot_rows["label"] == label]
            if kind == "longitudinal":
                rows = rows.loc[rows["reading"] == fits.LONGITUDINAL_GATED]
            truth = (
                fits.LONGITUDINAL_TRUTH[fits.LONGITUDINAL_GATED]
                if kind == "longitudinal"
                else fits.truth(case)
            )
            assert len(rows) >= draws, label
            ratio = float(rows["std_error"].mean() / rows["estimate"].std(ddof=1))
            coverage = float(rows["covered"].mean())
            assert abs(rows["estimate"].mean() - truth) < 5 * rows["estimate"].std() / math.sqrt(
                len(rows)
            )
            assert (
                calibration_plug_in(ratio, coverage, properties.CALIBRATION_REPLICATES) >= 0.90
            ), (
                label,
                ratio,
                coverage,
            )

    def test_the_ic_gap_clauses_pass_with_090_predictive(self, pilot_rows: pd.DataFrame) -> None:
        family = 1.0
        for name in study.GAP_GATED:
            gaps = {}
            for n in study.GAP_SIZES:
                rows = pilot_rows.loc[
                    (pilot_rows["label"] == f"ic_gap_{name}_n{n}")
                    & (pilot_rows["reading"] == "ic_gap")
                ]
                assert len(rows) == pilot.GAP_DRAWS[n]
                gaps[n] = rows["ratio_error"].to_numpy(dtype=float)
            value = gap_pass(gaps[2_000], gaps[20_000], name, predictive=True)
            assert value >= 0.90, (name, value)
            family *= value
        assert family >= 0.85

    def test_the_cluster_clauses_from_the_pilot_doses(self, pilot_rows: pd.DataFrame) -> None:
        def count(label: str, reading: str) -> tuple[int, int]:
            rows = pilot_rows.loc[
                (pilot_rows["label"] == label) & (pilot_rows["reading"] == reading)
            ]
            return int(rows["estimate"].sum()), len(rows)

        c_warned, c_total = count("ic_gap_C_n20000", "cluster")
        c_pass = 1.0 - float(
            betabinom.cdf(
                properties.CLUSTER_C_FLOOR - 1, 30, c_warned + 0.5, c_total - c_warned + 0.5
            )
        )
        assert c_pass >= 0.90
        gt = [count(f"ic_gap_{name}_n{n}", "cluster") for name, n in properties.CLUSTER_SOURCES]
        eager = [
            count(f"ic_gap_{name}_n{n}", "cluster_eager") for name, n in properties.CLUSTER_SOURCES
        ]
        warned, total = sum(w for w, _ in gt), sum(t for _, t in gt)
        assert jeffreys_at_most(warned, total, properties.CLUSTER_GT_CEILING, 220) >= 0.90
        warned, total = sum(w for w, _ in eager), sum(t for _, t in eager)
        assert jeffreys_at_most(warned, total, properties.CLUSTER_GT_CEILING, 220) < 0.01
        disabled, _ = count("ic_gap_C_n20000", "cluster_gap_disabled")
        assert disabled == 0

    def test_table_5b_and_the_ceilings_from_the_pilot(self, pilot_rows: pd.DataFrame) -> None:
        """Recomputed from the pilot's booster draws: the at-risk laws stay named."""
        for name in law.LAWS:
            rows = pilot_rows.loc[
                (pilot_rows["label"] == f"boosting_{name}_n2000")
                & (pilot_rows["reading"] == "boosting")
            ]
            errors = rows["ratio_error"].to_numpy(dtype=float)
            above = int(np.sum(errors > 1.0))
            value = jeffreys_at_most(above, errors.size, 2, 20)
            if name not in ("T", "B"):
                assert value >= 0.5, (name, above, errors.size)
            assert int(betabinom.ppf(0.95, 20, above + 0.5, errors.size - above + 0.5)) <= 20

    def test_each_cell_is_rebudgeted_from_the_pilot(self, pilot_rows: pd.DataFrame) -> None:
        """Each longitudinal part stays under 80 CPU-h at the pilot's median per-fit seconds."""
        for estimator in study.LONGITUDINAL:
            rows = pilot_rows.loc[pilot_rows["label"] == f"calibration_{estimator}"]
            per_fit = float(rows.drop_duplicates("replicate")["seconds"].median())
            for span in properties.part_ranges(f"longitudinal_{estimator}"):
                assert len(span) * per_fit / 3600 <= properties.PART_HOURS, (estimator, per_fit)
