"""The ``hazard-mtp`` declaration: truths, bounds, limits, sizes, pilot readings and seeds.

Each truth is checked by an independent route (a backward recursion written here, an adaptive
quadrature over a Gaussian dose, and a fit on the ``levels`` law's exact sample), each
efficiency bound against the cause-specific hazard form of the influence function written here
(summed, or integrated by adaptive quadrature) and a large-sample fit, the robustness limits and
the wrong learner against the exact sample and an independent score solve, and the replicate
counts against the shared sizing rules.  The pilot readings come from the committed pilot
artifact.  Nothing here reads a study run.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import root
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.interventions import ModifiedPolicy, Randomizer
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_mtp as study
from tests.studies import hazard_mtp_law as law
from tests.studies import hazard_mtp_pilot as pilot
from tests.studies import hazard_mtp_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed
from tests.studies.oracle_density_bins import oracle_bins

MARGINS = study.STUDY.margins
CONTRAST = study.CONTRAST_LABEL
#: How close the exact-sample fit with the correct mechanism comes to the truth.  The walk
#: stops at ``tol=1e-10`` on the relative score and the logistic mechanisms are solved to
#: ``1e-10``, which leaves about ``2e-8`` on the risks.
EXACT = 1e-7


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _exact_fit(configuration: str = "both_correct") -> Any:
    """The ``levels`` fit on its exact sample, each outcome weighted by its mass."""
    current = law.LEVELS_LAW
    frame = current.exact_frame()
    return _quiet(
        LTMLE(
            study.regimens("levels"),
            reference=study.NATURAL,
            n_folds=1,
            survival_model="hazard",
            targeting=study.ALGORITHM,
            tol=study.TOL,
            random_state=0,
            **study.learners("levels", frame, configuration),
        ).fit,
        current.container(frame, weights="wt"),
    )


@pytest.fixture(scope="module")
def exact_fits() -> dict[str, Any]:
    """The ``levels`` exact-sample fit of every robustness configuration."""
    return {configuration: _exact_fit(configuration) for configuration in properties.CONFIGURATIONS}


# ---------------------------------------------------------------------------- independent routes


def _backward(current: law.Law, f: float, a: float, w1: int, w2: int, horizon: int) -> float:
    """:math:`F(t \\mid f, a, W)` by a backward recursion, written apart from the law module."""
    c = current.hazard
    later = 0.0
    for node in range(horizon, 0, -1):
        eta = c.intercept + c.trend * (node - 1) + c.dose * a + c.w1 * w1 + c.w2 * w2 + c.flag * f
        lam = 1.0 / (1.0 + math.exp(-eta))
        later = lam + (1.0 - lam) * later
    return later


def _location(current: law.Law, f: float, w1: int, w2: int) -> float:
    c0, c1, c2 = current.centre
    return c0 + c1 * w1 + c2 * w2 + current.flag_shift * f


def _arms(current: law.Law, w1: int, w2: int) -> list[tuple[int, float]]:
    if current.kind != "vector":
        return [(0, 1.0)]
    a0, a1, a2 = current.arm
    p = 1.0 / (1.0 + math.exp(-(a0 + a1 * w1 + a2 * w2)))
    return [(0, 1.0 - p), (1, p)]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pdf(current: law.Law, mean: float, a: float) -> float:
    """The dose density at ``a`` given its location, truncated when the law is."""
    value = math.exp(-0.5 * ((a - mean) / current.scale) ** 2) / (
        current.scale * math.sqrt(2.0 * math.pi)
    )
    if current.bounds is None:
        return value
    low, high = current.bounds
    if not low <= a <= high:
        return 0.0
    return value / (_phi((high - mean) / current.scale) - _phi((low - mean) / current.scale))


def _support(current: law.Law, mean: float, span: float = 14.0) -> tuple[float, float, list]:
    """The integration range of a dose and the points where the policy or the ratio jump.

    The normal mass beyond 14 scales is below 1e-40.
    """
    if current.bounds is None:
        return mean - span * current.scale, mean + span * current.scale, []
    low, high = current.bounds
    cuts = [low + current.shift]
    if current.cap is not None:
        cuts += [current.cap - current.shift]
    return low, high, sorted(c for c in cuts if low < c < high)


def _mapped(current: law.Law, a: float) -> float:
    """The policy's dose, written apart from the law module: shifted, held past the cap."""
    if not current.continuous:
        return min(a + 1.0, 5.0)
    moved = a + current.shift
    return a if current.cap is not None and moved > current.cap else moved


def _quadrature_truth(current: law.Law, horizon: int, natural: bool) -> float:
    """``E F(t | d(F, A), W)`` by adaptive quadrature over the Gaussian dose."""
    total = 0.0
    for w1, w2 in law.STRATA:
        for f, share in _arms(current, w1, w2):
            mean = _location(current, f, w1, w2)

            def integrand(
                a: float, f: int = f, mean: float = mean, w1: int = w1, w2: int = w2
            ) -> float:
                dose = a if natural else _mapped(current, a)
                return _pdf(current, mean, a) * _backward(current, f, dose, w1, w2, horizon)

            low, high, cuts = _support(current, mean)
            value, _ = quad(
                integrand,
                low,
                high,
                points=cuts or None,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=200,
            )
            total += law.MASS * share * value
    return total


def _closed_ratio(current: law.Law, f: int, a: float, w1: int, w2: int) -> float:
    """The policy ratio in closed form: a level ratio, or the Gaussian shift's exponential."""
    if current.kind == "levels":
        weights = np.exp(
            -((np.arange(6) - _location(current, 0, w1, w2)) ** 2) / (2 * current.scale**2)
        )
        g = weights / weights.sum()
        index = round(a)
        if index == 0:
            return 0.0
        return float((g[index - 1] + (g[5] if index == 5 else 0.0)) / g[index])
    mean, delta = _location(current, f, w1, w2), current.shift
    smooth = math.exp((delta * (a - mean) - delta**2 / 2.0) / current.scale**2)
    if current.bounds is None:
        return smooth
    # Truncated: the normalizers of g(a - delta) and g(a) cancel; the pre-image must lie in the
    # support, and past the cap the smooth piece is empty and the unit keeps its dose.
    low, _ = current.bounds
    cap = math.inf if current.cap is None else current.cap
    moved = smooth if (a - delta >= low and a <= cap) else 0.0
    return moved + (1.0 if a > cap - delta else 0.0)


def _hazard_form(
    current: law.Law,
    horizon: int,
    natural: bool,
    f: int,
    a: float,
    w1: int,
    w2: int,
    time: float,
    event: int,
) -> float:
    """The uncentred curve in the cause-specific hazard form, apart from the law module.

    :math:`F(t \\mid d(f, a), W) + \\sum_{k \\le t} \\frac{r R_k}{G_k} H_k (dN(k) - \\lambda(k))`
    with :math:`H_k = 1 - (F(t) - F(k)) / S(k)`.
    """
    mapped = a if natural else _mapped(current, a)
    total = _backward(current, f, mapped, w1, w2, horizon)
    ratio = 1.0 if natural else _closed_ratio(current, f, a, w1, w2)
    if ratio == 0.0:
        return total
    incidence = [_backward(current, f, a, w1, w2, k) for k in range(1, horizon + 1)]
    survival, retained = 1.0, 1.0
    last = min(int(time), horizon)
    for node in range(1, last + 1):
        lam = float(current.hazard(node, f, a, w1, w2))
        survival *= 1.0 - lam
        retained *= float(current.retained(node, f, a, w1, w2))
        clever = 1.0 - (incidence[horizon - 1] - incidence[node - 1]) / survival
        jump = 1.0 if (event == 1 and int(time) == node) else 0.0
        total += ratio * clever * (jump - lam) / retained
    return total


# ---------------------------------------------------------------------------- truths


class TestTheTruths:
    @pytest.mark.parametrize("law_name", ["levels"])
    def test_the_levels_truth_is_the_backward_recursion(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        for horizon in range(1, current.n_times + 1):
            for natural in (False, True):
                expected = 0.0
                for w1, w2 in law.STRATA:
                    g = current.level_probabilities(w1, w2)[0]
                    for level in range(6):
                        moved = level if natural else min(level + 1, 5)
                        expected += (
                            law.MASS * g[level] * _backward(current, 0, moved, w1, w2, horizon)
                        )
                assert current.truth(horizon, natural=natural) == pytest.approx(expected, abs=1e-14)

    @pytest.mark.parametrize("law_name", ["gaussian", "truncated", "vector"])
    def test_the_quadrature_reaches_1e_10(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        for horizon in range(1, current.n_times + 1):
            for natural in (False, True):
                assert current.truth(horizon, natural=natural) == pytest.approx(
                    _quadrature_truth(current, horizon, natural), abs=1e-10
                )

    def test_the_exact_sample_fit_returns_every_truth(self, exact_fits: dict[str, Any]) -> None:
        result = exact_fits["both_correct"]
        for name, value in study.truths("levels").items():
            assert result[name].psi == pytest.approx(value, abs=EXACT), name

    def test_the_declared_truths(self) -> None:
        np.testing.assert_allclose(
            [
                properties.TRUTHS[f"{fit}_{arm}"]
                for fit in ("levels", "gaussian_density", "gaussian_classifier", "vector")
                for arm in ("contrast", "risk")
            ],
            [-0.062287, 0.443993, -0.063804, 0.432631, -0.071498, 0.424736, -0.040366, 0.199694],
            atol=5e-6,
        )
        for fit in ("gaussian_crossfit", "untruncated_density"):
            for arm in ("contrast", "risk"):
                assert (
                    properties.TRUTHS[f"{fit}_{arm}"]
                    == properties.TRUTHS[f"gaussian_classifier_{arm}"]
                )
        for scenario in study.SCENARIOS:
            assert study.TRUTH[scenario] == study.truths(study.FITS[scenario][0])

    def test_the_curves_rise(self) -> None:
        for current in law.LAWS.values():
            values = [current.truth(t) for t in range(1, current.n_times + 1)]
            assert np.all(np.diff(values) > 0.0)

    @staticmethod
    def _smallest_cumulative(current: law.Law) -> float:
        smallest = 1.0
        for w1, w2 in law.STRATA:
            g = current.level_probabilities(w1, w2)[0]
            for level in law.LEVELS:
                kept = np.prod(
                    [current.retained(k, 0, level, w1, w2) for k in range(1, current.n_times + 1)]
                )
                smallest = min(smallest, g[level] * kept)
        return float(smallest)

    def test_the_bounds_never_bind(self) -> None:
        """Positivity over every covariate value: ``W1`` in 0..1 and ``W2`` in 0..3.

        On ``levels`` the cumulative bound reads :math:`g(a \\mid W) G_k`, whose smallest value
        over all eight strata, six levels and five visits is 0.028 (``A = 0`` at ``W1 = 1,
        W2 = 3``), above twice the default bound.  On the continuous laws the treatment factor
        of the cumulative is one, so the bound reads the retention alone, whose smallest value
        is 0.64.  On ``truncated`` the ratio is bounded over the whole support (5.74 at the
        cap, the smooth part plus the held unit's one).
        """
        smallest = self._smallest_cumulative(law.LEVELS_LAW)
        assert smallest == pytest.approx(0.0281, abs=1e-4)
        assert smallest > 2.0 * DEFAULT_LTMLE_G_BOUNDS[0]
        for name, expected in {
            "levels": {"ratio_max": 3.117, "g_min": 0.0353, "retained_min": 0.716},
            "gaussian": {"ratio_max": 7.181, "retained_min": 0.637},
            "truncated": {"ratio_max": 5.738, "retained_min": 0.707},
            "vector": {"ratio_max": 7.181, "retained_min": 0.708, "arm_min": 0.310},
        }.items():
            reading = law.positivity(law.LAWS[name])
            for key, value in expected.items():
                assert reading[key] == pytest.approx(value, abs=1e-3), (name, key)

    def test_the_strata_cover_every_covariate_value(self) -> None:
        assert set(law.STRATA) == {(w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3)}
        frame = law.LEVELS_LAW.draw(4_000, 1)
        assert set(frame["W2"].unique()) == {0.0, 1.0, 2.0, 3.0}


# ---------------------------------------------------------------------------- bounds


class TestTheEfficiencyBounds:
    @pytest.mark.parametrize("law_name", list(law.LAWS))
    def test_the_iterated_mean_and_hazard_forms_agree(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        horizon = current.n_times
        for w1, w2 in law.STRATA:
            points = current.points(w1, w2)
            if current.continuous:
                # Beyond eight scales the quadrature weights are below 1e-14 and the hazard
                # form's (F(t) - F(k)) / S(k) loses digits to cancellation, so the two forms are
                # compared within eight scales, on every third node: the forms are identities in
                # the dose, so a node's agreement is exact or not, and a third of them covers
                # every piece of the truncated law.
                points = tuple(
                    point
                    for point in points
                    if abs(point.a - _location(current, point.f, w1, w2)) <= 8.0 * current.scale
                )[::3]
            for point in points:
                for _, time, event in current.outcomes(point.f, point.a, w1, w2):
                    for natural in (False, True):
                        assert current.unit_curve(
                            horizon, natural, point.f, point.a, w1, w2, time, event
                        ) == pytest.approx(
                            _hazard_form(
                                current, horizon, natural, point.f, point.a, w1, w2, time, event
                            ),
                            rel=1e-9,
                            abs=1e-11,
                        )

    def test_the_ratio_is_the_closed_form(self) -> None:
        for current in law.LAWS.values():
            for w1, w2 in law.STRATA:
                for point in current.points(w1, w2)[::7]:
                    assert current.ratio(point.f, point.a, w1, w2) == pytest.approx(
                        _closed_ratio(current, point.f, point.a, w1, w2), rel=1e-12
                    )

    @pytest.mark.parametrize("arm", ["contrast", "risk"])
    @pytest.mark.parametrize(
        ("law_name", "fit_label"),
        [
            ("gaussian", "gaussian_classifier"),
            ("truncated", "gaussian_density"),
            ("vector", "vector"),
        ],
    )
    def test_the_quadrature_bound_is_the_adaptive_bound(
        self, law_name: str, fit_label: str, arm: str
    ) -> None:
        """Each bound by adaptive quadrature of the hazard form's square, risk and contrast."""
        current = law.LAWS[law_name]
        label = f"{fit_label}_{arm}"
        horizon = current.n_times
        psi = properties.TRUTHS[label]
        total = 0.0
        for w1, w2 in law.STRATA:
            for f, share in _arms(current, w1, w2):
                mean = _location(current, f, w1, w2)

                def integrand(
                    a: float, f: int = f, mean: float = mean, w1: int = w1, w2: int = w2
                ) -> float:
                    second = 0.0
                    for mass, time, event in current.outcomes(f, a, w1, w2):
                        curve = _hazard_form(current, horizon, False, f, a, w1, w2, time, event)
                        if arm == "contrast":
                            curve -= _hazard_form(current, horizon, True, f, a, w1, w2, time, event)
                        second += mass * (curve - psi) ** 2
                    return _pdf(current, mean, a) * second

                low, high, cuts = _support(current, mean, span=12.0)
                value, _ = quad(
                    integrand,
                    low,
                    high,
                    points=cuts or None,
                    epsabs=1e-12,
                    epsrel=1e-11,
                    limit=200,
                )
                total += law.MASS * share * value
        assert math.sqrt(total) == pytest.approx(properties.EFFICIENCY_SD[label], rel=1e-8)

    def test_the_declared_bounds(self) -> None:
        np.testing.assert_allclose(
            [
                properties.EFFICIENCY_SD[f"{fit}_{arm}"]
                for fit in ("levels", "gaussian_density", "gaussian_classifier", "vector")
                for arm in ("contrast", "risk")
            ],
            [0.32477, 0.62191, 0.39649, 0.66004, 0.35776, 0.63649, 0.29602, 0.50372],
            atol=5e-5,
        )


# ---------------------------------------------------------------------------- the IC gap


def _ic_rows() -> pd.DataFrame:
    """The committed ``ic_gap`` pilot rows; the test skips while they are missing."""
    rows = _pilot("ic_gap")
    if rows.empty:
        pytest.skip("the ic_gap pilot has not been committed")
    return rows


def _ic_gaps(fit_label: str) -> dict[str, tuple[list[float], float]]:
    """Each arm's IC gap from the ``ic_gap`` pilot: the small-size draws and the large draw.

    The gap is SD(IC - EIF) / bound on the same draw, with the exact per-unit EIF of
    :meth:`~tests.studies.hazard_mtp_law.Law.frame_curves`.
    """
    small_n, large_n = properties.IC_GAP_SIZES
    rows = _ic_rows()
    rows = rows.loc[rows["fit_label"] == fit_label]
    out: dict[str, tuple[list[float], float]] = {}
    for label, _ in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == label]
        small = arm.loc[arm["n"] == small_n, "estimate"].to_numpy(dtype=float)
        large = arm.loc[arm["n"] == large_n, "estimate"].to_numpy(dtype=float)
        assert small.size == properties.IC_GAP_DRAWS and large.size == 1, (fit_label, label)
        out[label.rsplit("_", 1)[1]] = (list(small), float(large[0]))
    return out


class TestTheInfluenceCurveGap:
    """Each fit's influence curve approaches the exact efficient influence function.

    The fits are the ``ic_gap`` pilot's (:data:`~tests.studies.hazard_mtp_pilot.IC_GAP_JOBS`),
    read from the committed artifact.  The density route on the untruncated Gaussian law is the witness: its quantile tail bins
    are wider than the shift, so its gap stays large at n = 20,000; that law is reported, not
    gated.  Every gated fit label meets the declared bounds of
    :data:`~tests.studies.hazard_mtp_properties.IC_GAP_BOUND`.
    """

    def test_the_vectorized_curve_is_the_unit_curve(self) -> None:
        for current in law.LAWS.values():
            frame = current.draw(300, 11)
            horizon = current.n_times
            aim = law.Target(((1.0, horizon, False), (-1.0, horizon, True)))
            vector = current.frame_curves(aim, frame)
            if current.kind == "vector":
                units = [
                    (int(f), a, int(w1), int(w2), float(c == 1), int(c == 1 and y == 1))
                    for w1, w2, f, a, c, y in frame[["W1", "W2", "F", "A", "C1", "Y1"]].itertuples(
                        index=False
                    )
                ]
            else:
                units = [
                    (0, a, int(w1), int(w2), t, int(e))
                    for a, w1, w2, t, e in frame[["A", "W1", "W2", "time", "event"]].itertuples(
                        index=False
                    )
                ]
            scalar = np.array(
                [
                    current.unit_curve(horizon, False, *unit)
                    - current.unit_curve(horizon, True, *unit)
                    for unit in units
                ]
            )
            np.testing.assert_allclose(vector, scalar - aim.value(current), atol=1e-12)

    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_gated_fit_meets_the_declared_bound(self, fit_label: str) -> None:
        small_n, large_n = properties.IC_GAP_SIZES
        for arm, (small, large) in _ic_gaps(fit_label).items():
            median = float(np.median(small))
            assert median <= properties.IC_GAP_BOUND[small_n], (arm, small)
            assert large <= properties.IC_GAP_BOUND[large_n], (arm, large)
            assert large <= properties.IC_GAP_FALL * median, (arm, median, large)

    @pytest.mark.parametrize("fit_label", study.REPORTED_FITS)
    def test_the_untruncated_density_route_does_not(self, fit_label: str) -> None:
        _, large_n = properties.IC_GAP_SIZES
        small, large = _ic_gaps(fit_label)["contrast"]
        assert large > properties.IC_GAP_BOUND[large_n]
        assert large > properties.IC_GAP_FALL * float(np.median(small))

    def test_the_pilot_lost_no_fit(self) -> None:
        rows = _ic_rows()
        assert rows["refusal"].fillna("").eq("").all()
        assert sorted(rows["replicate"].unique()) == list(range(len(pilot.IC_GAP_JOBS)))
        assert (rows["converged"] == 1).all() and (rows["unexpected_warnings"] == 0).all()

    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_large_fit_reports_the_bound(self, fit_label: str) -> None:
        """At n = 40,000 each gated fit's influence-curve SD is within 5% of the bound."""
        rows = _ic_rows()
        rows = rows.loc[(rows["fit_label"] == fit_label) & (rows["n"] == pilot.BOUND_N)]
        assert len(rows) == len(properties.COEFFICIENTS[fit_label])
        for label, ratio in zip(rows["estimand"], rows["estimate"], strict=True):
            assert ratio == pytest.approx(1.0, abs=0.05), label

    def test_no_gated_dose_lands_where_the_bins_are_wider_than_the_shift(self) -> None:
        """On ``truncated`` every quantile bin the ratio reads, at the oracle count, is narrower
        than the shift; on ``gaussian`` the top bins are wider."""
        for law_name, wider in (("truncated", False), ("gaussian", True)):
            current = law.LAWS[law_name]
            frame = current.draw(2_000, stream_seed(study.STUDY, "design_bins", law_name, 0))
            dose = frame["A"].to_numpy(dtype=float)
            edges = bin_edges(dose, oracle_bins(dose.size))
            widest = float(np.max(np.diff(edges)))
            assert (widest > current.shift) is wider, (law_name, widest)


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(self, exact_fits: dict[str, Any]) -> None:
        for configuration, result in exact_fits.items():
            if configuration == "both_wrong":
                continue
            for label, name in properties.COEFFICIENTS["levels"]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[label], abs=EXACT), (
                    configuration,
                    label,
                )

    def test_the_randomizer_mixes_its_branches_exactly(self) -> None:
        """A two-branch randomizer on the exact sample, with the wrong hazard learner.

        The policy moves a unit one level up with probability 0.25 and keeps its level
        otherwise, so its risk is ``0.25 psi_up + 0.75 psi_natural``.  The hazard learner is
        wrong and the mechanism correct, so only the weighted walk can reach it: the initial
        hazards miss it, which is the nonzero witness.
        """
        current = law.LEVELS_LAW
        frame = current.exact_frame()

        def apply(a: Any, h: Any, e: Any) -> Any:
            del h
            return study._up(a, None) if e == "up" else np.asarray(a, dtype=float)

        plans = {
            "random": ModifiedPolicy(
                "random",
                apply=apply,
                randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
                policy_kind="known",
            ),
            study.NATURAL: study.regimens("levels")[study.NATURAL],
        }
        result = _quiet(
            LTMLE(
                plans,
                reference=study.NATURAL,
                n_folds=1,
                survival_model="hazard",
                targeting=study.ALGORITHM,
                tol=study.TOL,
                random_state=0,
                **study.learners("levels", frame, "mechanism_correct"),
            ).fit,
            current.container(frame, weights="wt"),
        )
        initial = {}
        for regimen_fit in result.fits.values():
            record = regimen_fit.hazard
            for (label, _, horizon), value in zip(
                record.targets, record.initial_levels, strict=True
            ):
                initial[(label, horizon)] = float(value)
        for horizon in range(1, current.n_times + 1):
            truth = 0.25 * current.truth(horizon) + 0.75 * current.truth(horizon, natural=True)
            assert result[study.risk_name("random", horizon)].psi == pytest.approx(truth, abs=EXACT)
        last = current.n_times
        truth = 0.25 * current.truth(last) + 0.75 * current.truth(last, natural=True)
        assert abs(initial[("random", last)] - truth) > 1e-3

    def test_the_both_wrong_control_is_discriminated(self, exact_fits: dict[str, Any]) -> None:
        result = exact_fits["both_wrong"]
        gap = result[properties.ESTIMAND[CONTRAST]].psi - properties.TRUTHS[CONTRAST]
        n = properties.DOUBLE_ROBUST_N
        assert gap / _scale(CONTRAST, n) == pytest.approx(3.437, abs=1e-3)
        for spread in (_scale(CONTRAST, n), 1.5 * _scale(CONTRAST, n)):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.DOUBLE_ROBUST_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)


# ---------------------------------------------------------------------------- wrong learner


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


def _pilot(label: str) -> pd.DataFrame:
    return PILOT.loc[PILOT["label"] == label] if not PILOT.empty else PILOT


CALIBRATION_PILOTED = not PILOT.empty and PILOT["label"].str.startswith("calibration_").any()

#: The pass probability each positive bias arm needs (design review d).
POSITIVE_PASS = 0.99
#: Every positive arm whose verdict reads a bias interval, with its replicates: the
#: ``double_robustness`` positives, the targeted necessity arm and the claimed root-n rungs.
POSITIVE_ARMS: dict[str, int] = {
    cell.cell: cell.replicates
    for cell in properties.declared_cells()
    if cell.role == "positive"
    and (
        cell.property in ("double_robustness", "targeting_necessity")
        or (cell.property == "root_n_and_efficiency" and cell.n != min(properties.RATE_SIZES))
    )
}
#: The wrong hazard learner's plug-in contrast gap, in efficiency SEs at n = 2,000, by fit.
WRONG_GAPS = {
    "levels": 3.473,
    "gaussian_density": 4.786,
    "gaussian_classifier": 5.458,
    "gaussian_crossfit": 5.458,
    "vector": 3.529,
}


def _tolerated_bias(replicates: int) -> float:
    """The largest |standardized bias| at which a positive arm passes with ``POSITIVE_PASS``."""
    z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    return MARGINS.standardized_bias - (z + float(norm.ppf(POSITIVE_PASS))) / np.sqrt(replicates)


def _independent_wrong_limit(current: law.Law, aim: law.Target) -> float:
    """The wrong learner's plug-in by a score solve with adaptive quadrature over the dose."""

    fitted: list[np.ndarray] = []
    for node in range(1, current.n_times + 1):

        def score(beta: np.ndarray, node: int = node) -> np.ndarray:
            out = np.zeros(2)
            for w1, w2 in law.STRATA:
                mean = _location(current, 0, w1, w2)

                def integrand(
                    a: float, j: int, w1: int = w1, w2: int = w2, mean: float = mean
                ) -> float:
                    entering = 1.0
                    for k in range(1, node):
                        entering *= float(current.retained(k, 0, a, w1, w2)) * (
                            1.0 - float(current.hazard(k, 0, a, w1, w2))
                        )
                    entering *= float(current.retained(node, 0, a, w1, w2))
                    lam = float(current.hazard(node, 0, a, w1, w2))
                    p = 1.0 / (1.0 + math.exp(-(beta[0] + beta[1] * a)))
                    return _pdf(current, mean, a) * entering * (lam - p) * (1.0, a)[j]

                low, high, cuts = _support(current, mean, span=12.0)
                for j in range(2):
                    out[j] += (
                        law.MASS
                        * quad(
                            integrand,
                            low,
                            high,
                            args=(j,),
                            points=cuts or None,
                            epsabs=1e-13,
                            limit=200,
                        )[0]
                    )
            return out

        solved = root(score, np.zeros(2), tol=1e-13)
        assert np.max(np.abs(score(solved.x))) < 1e-12
        fitted.append(solved.x)
    total = 0.0
    for coefficient, horizon, natural in aim.terms:
        for w1, w2 in law.STRATA:
            mean = _location(current, 0, w1, w2)

            def plug(
                a: float, horizon: int = horizon, natural: bool = natural, mean: float = mean
            ) -> float:
                dose = a if natural else _mapped(current, a)
                survival, running = 1.0, 0.0
                for node in range(horizon):
                    lam = 1.0 / (1.0 + math.exp(-(fitted[node][0] + fitted[node][1] * dose)))
                    running += survival * lam
                    survival *= 1.0 - lam
                return _pdf(current, mean, a) * running

            low, high, cuts = _support(current, mean, span=12.0)
            total += (
                coefficient
                * law.MASS
                * quad(plug, low, high, points=cuts or None, epsabs=1e-13, limit=200)[0]
            )
    return total


class TestTheWrongHazardLearner:
    @pytest.mark.parametrize("fit_label", list(WRONG_GAPS))
    def test_the_wrong_hazard_learner_rule(self, fit_label: str) -> None:
        """The plug-in contrast limit lies at least 3 efficiency SEs from the truth at n = 2,000."""
        label = f"{fit_label}_contrast"
        limit = law.wrong_hazard_limit(study.law_of(fit_label), properties.target(label))
        bias = (limit - properties.TRUTHS[label]) / _scale(label, 2_000)
        assert bias >= 3.0
        assert bias == pytest.approx(WRONG_GAPS[fit_label], abs=1e-3)

    def test_the_estimator_plug_in_is_the_limit(self, exact_fits: dict[str, Any]) -> None:
        result = exact_fits["mechanism_correct"]
        for label, name in properties.COEFFICIENTS["levels"]:
            limit = law.wrong_hazard_limit(law.LEVELS_LAW, properties.target(label))
            assert properties.plug_in(result, name) == pytest.approx(limit, abs=1e-7)

    def test_the_gaussian_limit_is_the_independent_solve(self) -> None:
        """On the truncated law, whose capped policy exercises every branch of the limit.

        The untruncated law runs the same code with no bounds and no cap; its solve takes
        three times as long in adaptive quadrature over the unbounded dose, so it is not
        repeated here.
        """
        current = law.TRUNCATED_LAW
        aim = properties.target("gaussian_density_contrast")
        assert law.wrong_hazard_limit(current, aim) == pytest.approx(
            _independent_wrong_limit(current, aim), abs=1e-9
        )

    @pytest.mark.parametrize("fit_label", pilot.WRONG_LEARNER_FITS)
    def test_the_pilot_plug_in_varies(self, fit_label: str) -> None:
        rows = _pilot(f"wrong_learner_{fit_label}")
        if rows.empty:
            pytest.skip("the pilot has not been committed")
        label = f"{fit_label}_contrast"
        rows = rows.loc[rows["estimand"] == properties.ESTIMAND[label]]
        assert rows["refusal"].fillna("").eq("").all()
        assert len(rows) == pilot.WRONG_LEARNER_DRAWS
        values = rows["wrong_plug_in"].to_numpy(dtype=float)
        assert np.var(values, ddof=1) > 0.0
        limit = law.wrong_hazard_limit(study.law_of(fit_label), properties.target(label))
        error = np.std(values, ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - limit) < 3.0 * error


class SaturatedShare(BaseEstimator, ClassifierMixin):
    """The empirical class shares in each distinct row of the design."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SaturatedShare:
        rows = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        self.classes_ = np.unique(target)
        self.keys_, index = np.unique(rows, axis=0, return_inverse=True)
        counts = np.zeros((len(self.keys_), self.classes_.size))
        np.add.at(counts, (index.reshape(-1), np.searchsorted(self.classes_, target)), 1.0)
        self.shares_ = counts / counts.sum(axis=1, keepdims=True)
        self.default_ = counts.sum(axis=0) / counts.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        rows = np.asarray(X, dtype=float)
        lookup = {tuple(key): i for i, key in enumerate(self.keys_)}
        found = np.array([lookup.get(tuple(row), -1) for row in rows])
        out = np.tile(self.default_, (rows.shape[0], 1))
        out[found >= 0] = self.shares_[found[found >= 0]]
        return out


def test_a_saturated_mechanism_hides_the_hazard_learner() -> None:
    """Why the mechanisms are parametric and not saturated (plan X70 rev 2.4, lesson 8).

    With the empirical level and retention shares of every stratum, the targeted contrast of the
    right and the wrong hazard learner coincide, so no cell could see the hazard fit.  The
    declared parametric mechanisms separate them.
    """
    frame = law.LEVELS_LAW.draw(2_000, 20261120)
    name = properties.ESTIMAND[CONTRAST]

    def fit(hazard: Any, saturated: bool) -> float:
        mechanism = (
            {"treatment_learner": SaturatedShare(), "censoring_learner": SaturatedShare()}
            if saturated
            else {
                "treatment_learner": law.LevelsTreatment(),
                "censoring_learner": law.CensoringLogistic("levels"),
            }
        )
        result = _quiet(
            LTMLE(
                study.regimens("levels"),
                reference=study.NATURAL,
                n_folds=1,
                survival_model="hazard",
                targeting=study.ALGORITHM,
                tol=study.TOL,
                outcome_learner=hazard,
                **mechanism,
            ).fit,
            law.LEVELS_LAW.container(frame),
        )
        return float(result[name].psi)

    right, wrong = fit(law.HazardLogistic("levels"), True), fit(law.NoWLogistic("levels"), True)
    assert right == pytest.approx(wrong, abs=1e-6)
    right, wrong = fit(law.HazardLogistic("levels"), False), fit(law.NoWLogistic("levels"), False)
    assert abs(right - wrong) > 1e-3


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        power = design_power(
            properties.TRUTHS[CONTRAST], properties.EFFICIENCY_SD[CONTRAST], properties.POWER_N
        )
        assert power >= 0.80
        assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99

    def test_the_derived_controls_leave_the_band(self) -> None:
        """The shrunken-SE arm leaves the SE-ratio band and the noise arm the efficiency band."""
        replicates = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > 0.999
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > 0.999

    def test_the_untargeted_control_is_discriminated(self) -> None:
        """The untargeted arm misses by the wrong learner's plug-in gap, at 1.0 and 1.5 spreads."""
        gap = WRONG_GAPS["levels"]
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.NECESSITY_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)

    @pytest.mark.skipif(CALIBRATION_PILOTED, reason="the committed calibration pilot sizes R")
    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The arm-pooled rule at a pilot of the declared size that reads a ratio of one.

        Every calibration positive arm is one arm (two per fit label).  The committed pilot
        replaces this reading once it exists, and this test then skips.
        """
        arms = sum(len(properties.COEFFICIENTS[label]) for label in properties.CALIBRATION_FITS)
        assert arms == 10
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, arms
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET

    @staticmethod
    def _calibration_rows(fit_label: str) -> pd.DataFrame:
        rows = _pilot(f"calibration_{fit_label}")
        if rows.empty:
            pytest.skip(f"the calibration pilot of {fit_label} has not been committed")
        return rows

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        ratios = []
        for fit_label in properties.CALIBRATION_FITS:
            rows = self._calibration_rows(fit_label)
            assert rows["refusal"].fillna("").eq("").all()
            for _, name in properties.COEFFICIENTS[fit_label]:
                arm = rows.loc[rows["estimand"] == name]
                assert len(arm) == sizing.CALIBRATION_PILOT_DRAWS
                ratios.append(float(arm["std_error"].mean() / arm["estimate"].std(ddof=1)))
        reading = mtp_study_tools.replicates_from_pilot(ratios, sizing.CALIBRATION_PILOT_DRAWS)
        assert not reading["suspect"]
        assert all(reading["screened"])
        assert reading["R"] == properties.CALIBRATION_REPLICATES

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0.

        The ``levels`` positive limits are the truth (:class:`TestTheRobustnessCells`).  The
        continuous fit labels' limits are the truth because their ratio is consistent: a
        correctly specified classifier, or the truncated law's density at a growing bin count
        (:class:`TestTheInfluenceCurveGap`).  Their finite-sample bias is read from the
        wrong-learner pilots.  The largest |bias| each
        count tolerates at 0.99 is :func:`_tolerated_bias`.
        """
        for cell, replicates in POSITIVE_ARMS.items():
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, replicates)
            assert probability >= POSITIVE_PASS, (cell, probability)
        assert _tolerated_bias(properties.DOUBLE_ROBUST_REPLICATES) == pytest.approx(
            0.1085, abs=1e-3
        )
        assert _tolerated_bias(properties.RATE_REPLICATES) == pytest.approx(0.0767, abs=1e-3)

    @pytest.mark.parametrize("fit_label", pilot.WRONG_LEARNER_FITS)
    def test_the_wrong_learner_arms_pass_at_the_pilot_bias(self, fit_label: str) -> None:
        """Each positive arm that fits the wrong hazard learner, read at its pilot.

        The arm is read at the pilot's standardized bias plus one of its standard errors.
        """
        rows = _pilot(f"wrong_learner_{fit_label}")
        if rows.empty:
            pytest.skip("the pilot has not been committed")
        label = f"{fit_label}_contrast"
        rows = rows.loc[rows["estimand"] == properties.ESTIMAND[label]]
        error = rows["wrong_targeted"].to_numpy(dtype=float) - properties.TRUTHS[label]
        standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
        cells = [
            cell
            for cell in POSITIVE_ARMS
            if cell.startswith(f"{label}__") and "mechanism_correct" in cell
        ]
        if fit_label == "levels":
            cells.append(f"{CONTRAST}__targeted")
        assert cells
        for cell in cells:
            probability = sizing.bias_equivalent_probability(
                MARGINS, standardized, POSITIVE_ARMS[cell]
            )
            assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` and ``outcome_correct`` arms and the claimed rungs, at the pilot."""
        rows = self._calibration_rows("levels")
        for cell_label, name in properties.COEFFICIENTS["levels"]:
            arm = rows.loc[rows["estimand"] == name]
            error = (arm["estimate"] - arm["truth"]).to_numpy(dtype=float)
            standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
            for cell, replicates in POSITIVE_ARMS.items():
                if cell.startswith(f"{cell_label}__") and cell.endswith(
                    ("__both_correct", "__outcome_correct", "__n_2000", "__n_8000")
                ):
                    probability = sizing.bias_equivalent_probability(
                        MARGINS, standardized, replicates
                    )
                    assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the calibration pilots.

        The calibration pilot of each primary fit label draws its scenario at the primary n
        with the primary fit.  One implementation, clauses (a) to (c) on the covered counts and
        clause (d) on the standardized biases, one group per scenario.
        """
        covered: dict[str, int] = {}
        groups = []
        for scenario in study.SCENARIOS:
            rows = self._calibration_rows(scenario)
            assert int(rows["n"].iloc[0]) == study.PRIMARY_N
            names = list(study.ESTIMANDS[scenario])
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
                )
            )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                sizing.CALIBRATION_PILOT_DRAWS,
                implementations=1,
                coverage_pass=share,
                bias_groups=groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )


# ---------------------------------------------------------------------------- pilot readings


def _paired_ratios(
    rows: pd.DataFrame, fit_label: str, column: str = "iterated_estimate"
) -> dict[str, float]:
    """The SD of hazard minus iterated-mean over the efficiency-bound SE, by cell label."""
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm[column]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


@pytest.mark.parametrize("fit_label", study.DIFFERENCE_FITS)
class TestTheFitsDifferPilot:
    """The ``iterated_mean_difference`` readings of plan X70 section 5, reported and not gated.

    :func:`~tests.studies.hazard_mtp_properties.fits_differ_reading` states the paired ratios and
    the decreasing share with its interval beside the plan's floors; no test asserts a floor,
    because the owner's decision on the requirement is pending.  Each fit's coverage and SE
    ratio are reported side by side.
    """

    @staticmethod
    def _rows(fit_label: str) -> pd.DataFrame:
        rows = _pilot(f"fits_differ_{fit_label}")
        if rows.empty:
            pytest.skip("the pilot has not been committed")
        return rows

    def test_the_pilot_lost_no_fit(self, fit_label: str) -> None:
        rows = self._rows(fit_label)
        assert rows["refusal"].fillna("").eq("").all()
        assert rows["replicate"].nunique() == pilot.FITS_DIFFER_DRAWS

    def test_the_reading_is_complete_and_finite(self, fit_label: str) -> None:
        rows = self._rows(fit_label)
        decreases = rows.loc[rows["estimand"] == "largest_decrease", "estimate"]
        assert len(decreases) == pilot.FITS_DIFFER_DRAWS
        assert np.isfinite(decreases.to_numpy(dtype=float)).all()
        for column in ("estimate", "iterated_estimate", "isotonic_estimate"):
            values = rows.loc[rows["estimand"] != "largest_decrease", column]
            assert np.isfinite(values.to_numpy(dtype=float)).all(), column
        ratios = _paired_ratios(rows, fit_label)
        ratios |= {
            f"{label} (isotonic)": value
            for label, value in _paired_ratios(rows, fit_label, "isotonic_estimate").items()
        }
        reading = properties.fits_differ_reading(
            ratios, int((decreases > 0.0).sum()), len(decreases)
        )
        assert all(np.isfinite(value) and value > 0.0 for value in reading["ratios"].values())
        low, high = reading["share_interval"]
        assert 0.0 <= low <= reading["share"] <= high <= 1.0
        print(reading["text"])


@pytest.mark.skipif(
    _pilot("fits_differ_mechanism_correct").empty, reason="the pilot has not been committed"
)
def test_the_mechanism_correct_ratios_are_reported() -> None:
    rows = _pilot("fits_differ_mechanism_correct")
    assert rows["refusal"].fillna("").eq("").all()
    ratios = _paired_ratios(rows, "levels")
    assert all(np.isfinite(value) and value > 0.0 for value in ratios.values()), ratios


def test_the_reading_is_reported_with_the_floors_in_its_text() -> None:
    """The floors are named constants that the reading's text quotes and no verdict reads."""
    assert properties.PAIRED_DIFFERENCE_FLOOR == 0.05
    assert properties.NON_MONOTONE_FLOOR == 0.10
    reading = properties.fits_differ_reading({"levels_contrast": 0.03, "levels_risk": 0.02}, 0, 200)
    assert not {key for key in reading if key.endswith("passes")}
    assert reading["largest_ratio"] == 0.03 and reading["contrast_ratio"] == 0.03
    assert reading["share_interval"] == (0.0, pytest.approx(0.0261, abs=1e-4))
    assert "0.05" in reading["text"] and "0.1" in reading["text"]
    assert "Reported, not gated" in reading["text"]


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
        rate = {name for name in declared if name.startswith("root_n_rate/")}
        assert written | rate == declared
        assert filled | rate == declared

    def test_the_study_is_not_yet_registered(self) -> None:
        # Registration comes with the artifacts, the evidence page and the grid row.
        assert study.STUDY.slug not in {record.slug for record in registered()}

    def test_the_seeds_are_distinct(self) -> None:
        from tests.studies import canonical_hazard_msm

        # hazard-msm (X70 part (d)) is declared but not registered, so it is named here.
        others = [*registered(), canonical_hazard_msm.STUDY]
        assert canonical_hazard_msm.STUDY.slug not in {r.slug for r in registered()}
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
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)
