"""The kernel-smoothed dose-response curve: weights, quadrature, influence curve and fit.

The exact law has a three-level covariate, a dose whose conditional density is linear on
``[-1, 6]``, and a binary outcome with a nonlinear regression.  Every expectation below is an
exact sum over ``W`` and ``Y`` and a high-order Gauss-Legendre integral over the dose, computed
apart from the package's composite rule.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly._support import UNCLASSIFIED
from cleverly.dose_response import (
    DoseResponse,
    equivalent_kernel,
    influence_curve,
    kernel,
    pair_name,
)
from cleverly.estimators.dose_response import (
    DOSE_RESPONSE_BAND_DRAWS,
    DOSE_RESPONSE_BAND_MEASURED,
    DOSE_RESPONSE_MEASURED,
    DOSE_RESPONSE_STUDY,
    DoseResponseTMLE,
    dose_response_support,
)
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError

SUPPORT = (0.0, 5.0)
A_RANGE = (-1.0, 6.0)
P_W = np.array([0.3, 0.45, 0.25])
SLOPES = np.array([-0.08, 0.02, 0.1])
PAIRS = ((0.8, 0.0), (0.8, 0.3), (0.8, 2.5), (0.8, 5.0), (1.5, 1.2))


def _g(a: np.ndarray, w: int, slopes: np.ndarray = SLOPES) -> np.ndarray:
    width = A_RANGE[1] - A_RANGE[0]
    return (1.0 + slopes[w] * (a - 2.5)) / width


def _q(a: np.ndarray, w: int, shift: float | np.ndarray = 0.0) -> np.ndarray:
    return expit(-1.0 + 0.45 * a - 0.08 * a * a + 0.3 * w + shift)


def _grid(low: float, high: float, breaks: tuple[float, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """An independent composite rule: 40 Gauss-Legendre nodes per piece."""
    points = np.unique(np.array([low, high, *[b for b in breaks if low < b < high]]))
    x, wq = np.polynomial.legendre.leggauss(40)
    nodes, mass = [], []
    for left, right in itertools.pairwise(points):
        nodes.append(0.5 * (right - left) * x + 0.5 * (right + left))
        mass.append(0.5 * (right - left) * wq)
    return np.concatenate(nodes), np.concatenate(mass)


def _breaks(h: float, a0: float) -> tuple[float, ...]:
    return (a0 - h, a0 + h, *SUPPORT)


def _theta(curve: DoseResponse, h: float, a0: float, q=_q, p_w=P_W) -> float:  # type: ignore[no-untyped-def]
    """The parameter by the package's composite rule."""
    rule = curve.quadrature()
    index = curve.pairs.index((h, a0))
    return float(sum(p_w[w] * rule.weights[index] @ q(rule.nodes, w) for w in range(3)))


def _expect_d_times(  # type: ignore[no-untyped-def]
    curve: DoseResponse, h: float, a0: float, score, *, q=_q, g=_g, mutate=None
) -> float:
    """E_P[D(P_nuisance) s] by the independent rule, with D from the package's function."""
    psi = _theta(curve, h, a0, q=q)
    rule = curve.quadrature()
    index = curve.pairs.index((h, a0))
    nodes, mass = _grid(*A_RANGE, _breaks(h, a0))
    total = 0.0
    for w in range(3):
        weight = equivalent_kernel(nodes, a0, h, curve.support, curve.kernel)
        density = g(nodes, w)
        h_obs = weight / density
        if mutate == "drop_inverse_density":
            h_obs = weight
        if mutate == "density_at_dose":
            h_obs = weight / g(np.full_like(nodes, a0), w)
        integral = float(rule.weights[index] @ q(rule.nodes, w))
        if mutate == "omit_integral":
            integral = psi
        for y in (0.0, 1.0):
            prob_y = np.where(y == 1.0, _q(nodes, w), 1.0 - _q(nodes, w))
            d = influence_curve(
                np.full_like(nodes, y), q(nodes, w), h_obs, np.full_like(nodes, integral), psi
            )
            s = score(np.full_like(nodes, y), nodes, w)
            total += P_W[w] * float(np.sum(mass * _g(nodes, w) * prob_y * d * s))
    return total


CURVE = DoseResponse(
    doses=(0.0, 0.3, 1.2, 2.5, 5.0), bandwidths=(0.8, 1.5), support=SUPPORT, nodes_per_piece=8
)


def _outcome_score(y: np.ndarray, a: np.ndarray, w: int) -> np.ndarray:
    return (y - _q(a, w)) * (np.sin(a) + 0.5 * w)


def _outcome_derivative(h: float, a0: float) -> float:
    def tilted(a: np.ndarray, w: int, eps: float) -> np.ndarray:
        q = _q(a, w)
        return q + eps * q * (1.0 - q) * (np.sin(a) + 0.5 * w)

    step = 1e-4
    up = _theta(CURVE, h, a0, q=lambda a, w: tilted(a, w, step))
    down = _theta(CURVE, h, a0, q=lambda a, w: tilted(a, w, -step))
    return (up - down) / (2.0 * step)


S_W = np.array([1.0, -0.4, 0.0])
S_W = S_W - (P_W @ S_W)


def _covariate_score(y: np.ndarray, a: np.ndarray, w: int) -> np.ndarray:
    del y
    return np.full_like(a, S_W[w])


def _dose_score(y: np.ndarray, a: np.ndarray, w: int) -> np.ndarray:
    """A mean-zero score of A | W: the centred dose."""
    del y
    nodes, mass = _grid(*A_RANGE)
    mean = float(np.sum(mass * _g(nodes, w) * nodes))
    return a - mean


@pytest.mark.parametrize(("h", "a0"), PAIRS)
def test_outcome_score_derivative_equals_the_influence_curve(h: float, a0: float) -> None:
    lhs = _outcome_derivative(h, a0)
    rhs = _expect_d_times(CURVE, h, a0, _outcome_score)
    assert abs(lhs) > 1e-3
    assert rhs == pytest.approx(lhs, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(("h", "a0"), PAIRS)
def test_covariate_score_derivative_equals_the_influence_curve(h: float, a0: float) -> None:
    lhs = sum(
        S_W[w] * _theta(CURVE, h, a0, q=lambda a, v, w=w: _q(a, v) * (v == w)) for w in range(3)
    )
    rhs = _expect_d_times(CURVE, h, a0, _covariate_score)
    assert abs(lhs) > 1e-3
    assert rhs == pytest.approx(lhs, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(("h", "a0"), PAIRS)
def test_a_score_of_the_dose_law_has_zero_derivative_and_zero_inner_product(
    h: float, a0: float
) -> None:
    # The parameter reads (P_W, Qbar) only, so a score of A | W leaves it unchanged; the
    # influence curve must be orthogonal to that score (the canonical-gradient claim).
    rhs = _expect_d_times(CURVE, h, a0, _dose_score)
    assert abs(rhs) < 1e-10


@pytest.mark.parametrize("mutation", ["drop_inverse_density", "density_at_dose", "omit_integral"])
def test_each_mutation_breaks_a_derivative_check(mutation: str) -> None:
    failures = 0
    for h, a0 in PAIRS:
        outcome = _expect_d_times(CURVE, h, a0, _outcome_score, mutate=mutation)
        covariate = _expect_d_times(CURVE, h, a0, _covariate_score, mutate=mutation)
        covariate_lhs = sum(
            S_W[w] * _theta(CURVE, h, a0, q=lambda a, v, w=w: _q(a, v) * (v == w)) for w in range(3)
        )
        bad_outcome = abs(outcome - _outcome_derivative(h, a0)) > 1e-8
        bad_covariate = abs(covariate - covariate_lhs) > 1e-8
        failures += int(bad_outcome or bad_covariate)
    assert failures == len(PAIRS)


def _remainder(h: float, a0: float, q, g) -> float:  # type: ignore[no-untyped-def]
    """theta(P~) - theta(P0) + P0 D(P~) by the package's functions."""
    return (
        _theta(CURVE, h, a0, q=q)
        - _theta(CURVE, h, a0)
        + _expect_d_times(CURVE, h, a0, lambda y, a, w: np.ones_like(a), q=q, g=g)
    )


def _product(h: float, a0: float, q, g) -> float:  # type: ignore[no-untyped-def]
    nodes, mass = _grid(*SUPPORT, _breaks(h, a0))
    weight = equivalent_kernel(nodes, a0, h, CURVE.support, CURVE.kernel)
    return float(
        sum(
            P_W[w]
            * np.sum(
                mass
                * weight
                * (g(nodes, w) - _g(nodes, w))
                / g(nodes, w)
                * (q(nodes, w) - _q(nodes, w))
            )
            for w in range(3)
        )
    )


WRONG_SLOPES = np.array([0.09, -0.05, 0.0])


def _wrong_g(a: np.ndarray, w: int) -> np.ndarray:
    return _g(a, w, WRONG_SLOPES)


def _wrong_q(a: np.ndarray, w: int) -> np.ndarray:
    return _q(a, w, shift=0.4 * np.cos(a))


@pytest.mark.parametrize(("h", "a0"), PAIRS)
def test_the_exact_remainder_is_the_product_and_vanishes_with_one_correct_nuisance(
    h: float, a0: float
) -> None:
    both = _remainder(h, a0, _wrong_q, _wrong_g)
    assert abs(both) > 1e-6
    assert both == pytest.approx(_product(h, a0, _wrong_q, _wrong_g), rel=1e-6, abs=1e-11)
    assert abs(_remainder(h, a0, _q, _wrong_g)) < 1e-11
    assert abs(_remainder(h, a0, _wrong_q, _g)) < 1e-11


@pytest.mark.parametrize("name", ["epanechnikov", "triweight", "uniform"])
@pytest.mark.parametrize("a0", [0.0, 0.3, 2.5, 4.9, 5.0])
def test_equivalent_kernel_moments_and_sign(name: str, a0: float) -> None:
    h = 1.0
    nodes, mass = _grid(*SUPPORT, (a0 - h, a0 + h))
    w = equivalent_kernel(nodes, a0, h, SUPPORT, name)  # type: ignore[arg-type]
    assert float(np.sum(mass * w)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.sum(mass * w * (nodes - a0))) == pytest.approx(0.0, abs=1e-12)
    interior = SUPPORT[0] + h <= a0 <= SUPPORT[1] - h
    if interior:
        assert np.all(w >= 0.0)
        np.testing.assert_allclose(w, kernel(name, (nodes - a0) / h) / h, atol=1e-12)  # type: ignore[arg-type]
    else:
        assert w.min() < 0.0
        # The plain kernel average, the mutation, misses the first-moment identity.
        plain = kernel(name, (nodes - a0) / h) * (nodes >= 0) / h  # type: ignore[arg-type]
        plain = plain / np.sum(mass * plain)
        assert abs(float(np.sum(mass * plain * (nodes - a0)))) > 1e-2


def test_the_composite_rule_is_exact_on_a_piecewise_polynomial_and_needs_its_breaks() -> None:
    curve = DoseResponse(doses=(1.0, 3.7), bandwidths=(0.9,), support=SUPPORT)
    breaks = (0.7, 2.2, 4.1)

    def f(a: np.ndarray) -> np.ndarray:
        # A jump at each break: what 1 / g of a binned density looks like.
        return 1.0 + a * a / 10.0 + np.searchsorted(np.array(breaks), a)

    for p, (h, a0) in enumerate(curve.pairs):
        nodes, mass = _grid(*SUPPORT, (*breaks, a0 - h, a0 + h))
        exact = float(np.sum(mass * equivalent_kernel(nodes, a0, h, SUPPORT) * f(nodes)))
        with_breaks = curve.quadrature(breaks)
        without = curve.quadrature()
        assert float(with_breaks.weights[p] @ f(with_breaks.nodes)) == pytest.approx(
            exact, abs=1e-12
        )
        if any(a0 - h < b < a0 + h for b in breaks):
            assert abs(float(without.weights[p] @ f(without.nodes)) - exact) > 1e-4


def test_declaration_refuses_bad_values() -> None:
    with pytest.raises(DataError, match="inside the support"):
        DoseResponse(doses=(6.0,), bandwidths=(1.0,), support=SUPPORT)
    with pytest.raises(DataError, match="positive"):
        DoseResponse(doses=(1.0,), bandwidths=(0.0,), support=SUPPORT)
    with pytest.raises(DataError, match="distinct"):
        DoseResponse(doses=(1.0,), bandwidths=(1.0, 1.0), support=SUPPORT)
    with pytest.raises(DataError, match="at most 200"):
        DoseResponse(doses=tuple(np.linspace(0, 5, 101)), bandwidths=(0.5, 1.0), support=SUPPORT)
    with pytest.raises(DataError, match="10 significant digits"):
        DoseResponse(doses=(1.0, 1.0 + 1e-12), bandwidths=(1.0,), support=SUPPORT)
    with pytest.raises(DataError, match="unknown kernel"):
        DoseResponse(doses=(1.0,), bandwidths=(1.0,), support=SUPPORT, kernel="gaussian")  # type: ignore[arg-type]


def test_names_and_spanning_read_the_declaration_only() -> None:
    curve = DoseResponse.spanning((0.0, 4.0), 3, fractions=(0.25,))
    assert curve.doses == (0.0, 2.0, 4.0)
    assert curve.bandwidths == (1.0,)
    assert curve.names[1] == pair_name(1.0, 2.0) == "dose_response[h=1,a=2]"


def _sample(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.normal(2.5 + 0.6 * w, 1.2)
    y = rng.binomial(1, expit(-1.0 + 0.5 * a - 0.08 * a * a + 0.4 * w))
    return pd.DataFrame({"Y": y, "A": a, "W": w})


def _fit(frame: pd.DataFrame, curve: DoseResponse, **kwargs: object):  # type: ignore[no-untyped-def]
    return DoseResponseTMLE(
        curve,
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=0,
        n_folds=3,
        **kwargs,  # type: ignore[arg-type]
    ).fit(frame, outcome="Y", treatment="A", covariates=["W"])


FIT_CURVE = DoseResponse(doses=(1.0, 2.5, 4.0), bandwidths=(0.6, 1.0), support=(0.5, 4.5))


@pytest.fixture(scope="module")
def fitted():  # type: ignore[no-untyped-def]
    return _fit(_sample(1200, 3), FIT_CURVE)


def test_every_pair_solves_its_score_and_reports_a_band(fitted) -> None:  # type: ignore[no-untyped-def]
    assert set(fitted.estimates) == set(FIT_CURVE.names)
    for name, score in fitted.scores.items():
        assert abs(score) <= 1e-5 * fitted[name].std_error
    for name, estimate in fitted.estimates.items():
        lower, upper = fitted.band.bands[name]
        assert lower <= estimate.ci[0] <= estimate.psi <= estimate.ci[1] <= upper
    assert fitted.quadrature_shift < 0.05
    frame = fitted.to_frame()
    assert list(frame["bandwidth"]) == [h for h, _ in FIT_CURVE.pairs]


def test_round_trip_keeps_the_estimates(fitted) -> None:  # type: ignore[no-untyped-def]
    back = loads(dumps(fitted))
    for name in FIT_CURVE.names:
        assert back[name].psi == fitted[name].psi


def test_linear_fluctuation_solves_its_score() -> None:
    result = _fit(_sample(800, 5), FIT_CURVE, fluctuation="linear")
    assert max(abs(score) for score in result.scores.values()) < 1e-9
    # A binary outcome with the linear fluctuation is no configuration the study measures.
    assert {e.support.label for e in result.estimates.values()} == {"experimental"}
    assert "linear fluctuation" in result[FIT_CURVE.names[0]].support.reason


def test_a_measured_configuration_reads_supported(fitted) -> None:  # type: ignore[no-untyped-def]
    assert not any(cell.below_floor for cell in fitted.pair_support)
    for estimate in fitted.estimates.values():
        assert estimate.support.label == "supported"
        assert estimate.support.evidence == (DOSE_RESPONSE_STUDY,)
    # The band cell measured the Mammen multiplier at 10,000 draws on the identity scale.
    assert fitted.band.support.label == "supported"
    assert fitted.band.support.evidence == (DOSE_RESPONSE_STUDY,)
    assert "identity scale" in fitted.band.support.reason
    assert fitted.band.kind == "mammen"
    assert fitted.band.n_replicates == DOSE_RESPONSE_BAND_DRAWS == 10_000


def test_the_band_is_not_range_bounded(fitted) -> None:  # type: ignore[no-untyped-def]
    """The default ``band_scale="logit"`` maps nothing here: the estimates carry no range.

    Each band is the identity-scale ``psi +/- c * se``, so a limit can leave [0, 1]; a band
    built on the logit of a bounded estimate would be asymmetric around ``psi``.
    """
    band = fitted.band
    assert band.band_scale == "logit"
    for name, estimate in fitted.estimates.items():
        assert estimate.bounds is None
        assert band.scales[name] == "identity"
        low, high = band.bands[name]
        half = band.critical_value * estimate.std_error
        assert (low, high) == pytest.approx((estimate.psi - half, estimate.psi + half), abs=1e-12)


def test_a_band_with_fewer_draws_than_its_cell_reads_experimental() -> None:
    measured = {"outcome": "binary", "fluctuation": "logistic", "strata": False}
    common = {"cross_fitted": True, "weighted": False}
    pairs, band = dose_response_support(**measured, **common, n_replicates=1_000)
    assert (pairs.label, band.label) == ("supported", "experimental")
    assert "10,000 multiplier draws" in band.reason and "1,000" in band.reason
    # The control: the measured draw count is supported.
    assert dose_response_support(**measured, **common)[1].label == "supported"


def test_the_band_reads_its_measured_cell() -> None:
    """The committed band cell is green and read the estimator's default band."""
    import pandas as pd

    from tests.studies import dose_response as study
    from tests.studies.evidence.registry import ROOT

    properties = pd.read_csv(ROOT / "tests" / "canonical" / "dose_response" / "properties.csv")
    cell = properties.set_index("cell").loc["binary_curve__simultaneous_band"]
    assert bool(cell["passed"])
    band = study.estimator("binary")
    assert band.n_replicates == DOSE_RESPONSE_BAND_DRAWS


def test_an_in_sample_fit_reads_experimental() -> None:
    result = _fit(_sample(600, 4), FIT_CURVE, cross_fit=False)
    assert {e.support.label for e in result.estimates.values()} == {"experimental"}
    assert result.band.support.label == "experimental"
    assert "in-sample" in result[FIT_CURVE.names[0]].support.reason


@pytest.mark.parametrize(
    ("outcome", "fluctuation", "strata", "pairs", "band"),
    [
        ("binary", "logistic", False, "supported", "supported"),
        ("binary", "logistic", True, "supported", "experimental"),
        ("continuous", "linear", False, "supported", "experimental"),
        ("continuous", "logistic", False, "experimental", "experimental"),
        ("binary", "linear", False, "experimental", "experimental"),
        ("continuous", "linear", True, "experimental", "experimental"),
    ],
)
def test_the_label_reads_the_measured_table(
    outcome: str, fluctuation: str, strata: bool, pairs: str, band: str
) -> None:
    configured, banded = dose_response_support(
        outcome=outcome, fluctuation=fluctuation, strata=strata, cross_fitted=True, weighted=False
    )
    assert (configured.label, banded.label) == (pairs, band)
    weighted, _ = dose_response_support(
        outcome=outcome, fluctuation=fluctuation, strata=strata, cross_fitted=True, weighted=True
    )
    assert weighted.label == "experimental"
    assert "weighted rows" in weighted.reason


def test_the_measuring_study_is_registered_and_exempt_from_support_rows() -> None:
    from cleverly._support_rows import EXEMPT_STUDIES
    from tests.studies.evidence.registry import registered

    assert DOSE_RESPONSE_STUDY in {record.slug for record in registered()} & set(EXEMPT_STUDIES)
    assert DOSE_RESPONSE_BAND_MEASURED in DOSE_RESPONSE_MEASURED


def test_low_support_pairs_carry_the_label() -> None:
    curve = DoseResponse(doses=(1.0, 2.5), bandwidths=(0.6,), support=(0.5, 4.5), support_floor=1e9)
    result = _fit(_sample(600, 7), curve)
    assert all(s.below_floor for s in result.pair_support)
    assert all(e.inference == "low_dose_support_plugin" for e in result.estimates.values())
    # The configuration is measured, so the floor alone makes each pair and the band
    # experimental, with X95 as the evidence that would lift it.
    for estimate in result.estimates.values():
        assert estimate.support.label == "experimental"
        assert estimate.support.evidence == ("X95",)
    assert result.band.support.label == "experimental"
    assert result.warnings


def test_an_interval_outside_the_observed_doses_is_refused() -> None:
    curve = DoseResponse(doses=(1.0,), bandwidths=(0.6,), support=(-20.0, 4.5))
    with pytest.raises(DataError, match="inside the observed dose"):
        _fit(_sample(300, 1), curve)


def test_a_discrete_treatment_is_refused() -> None:
    frame = _sample(200, 2).assign(A=lambda d: (d["A"] > 2.5).astype(int))
    with pytest.raises(CapabilityError, match="distinct values"):
        DoseResponseTMLE(FIT_CURVE).fit(frame, outcome="Y", treatment="A", covariates=["W"])


def test_baseline_strata_get_their_own_curves_from_one_nuisance_fit() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    v = rng.binomial(1, 0.5, size=n)
    w = rng.normal(size=n)
    a = rng.normal(2.5 + 0.5 * w, 1.2)
    y = rng.binomial(1, expit(-1.0 + (0.2 + 0.4 * v) * a - 0.05 * a * a + 0.3 * w))
    frame = pd.DataFrame({"Y": y, "A": a, "W": w, "V": v})
    curve = DoseResponse(doses=(1.5, 3.5), bandwidths=(1.0,), support=(0.5, 4.5))
    result = DoseResponseTMLE(
        curve,
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=0,
        n_folds=3,
    ).fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
    assert len(result.estimates) == 4
    slope = {
        s: result.estimate(1.0, 3.5, f"V={s}").psi - result.estimate(1.0, 1.5, f"V={s}").psi
        for s in (0, 1)
    }
    # The nonzero witness: V modifies the dose response.
    assert slope[1] - slope[0] > 0.1
    for cell in result.pair_support:
        curve_values = result.estimates[cell.name].influence_curve
        outside = frame["V"].to_numpy() != int(cell.stratum.split("=")[1])  # type: ignore[union-attr]
        assert np.all(curve_values[outside] == 0.0)
        assert abs(float(curve_values.mean())) < 1e-8
    assert set(result.to_frame()["stratum"]) == {"V=0", "V=1"}
    # The strata scenario measures the pointwise intervals; the band cell is unstratified.
    for cell in result.pair_support:
        label = result.estimates[cell.name].support.label
        assert label == ("experimental" if cell.below_floor else "supported")
    assert result.band.support.label == "experimental"


def test_the_curve_plot_draws_one_series_per_bandwidth(fitted) -> None:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cleverly.plotting import curve

    fig = curve(fitted)
    assert fig.axes[0].get_title(loc="left") == "Smoothed dose-response curve"
    guides = [line for line in fig.axes[0].get_lines() if line.get_gid() == "cleverly-guide"]
    assert len(guides) == len(FIT_CURVE.bandwidths)
    plt.close(fig)


# ------------------------------------------------- the estimator's own targeting, exactly

#: A dose density constant on each bin, so the fitted density's form holds exactly.
EXACT_EDGES = np.array([-1.0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0])
EXACT_MASS = np.array(
    [
        [0.10, 0.15, 0.25, 0.25, 0.15, 0.10],
        [0.20, 0.20, 0.20, 0.15, 0.15, 0.10],
        [0.05, 0.10, 0.20, 0.25, 0.20, 0.20],
    ]
)
EXACT_CURVE = DoseResponse(doses=(0.0, 0.4, 2.5, 4.6, 5.0), bandwidths=(0.7, 1.3), support=SUPPORT)


def _g_exact(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    widths = np.diff(EXACT_EDGES)
    index = np.clip(np.searchsorted(EXACT_EDGES, a, side="right") - 1, 0, widths.size - 1)
    return EXACT_MASS[w.astype(int), index] / widths[index]


def _q_exact(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 0.2 + 0.08 * a - 0.006 * a * a + 0.05 * w


def _q_wrong(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _q_exact(a, w) + 0.04 - 0.02 * a + 0.01 * w * a


def _exact_rows() -> pd.DataFrame:
    """The law's support as weighted rows: P_n is then the law's expectation, exactly."""
    breaks = {*EXACT_EDGES.tolist(), *SUPPORT}
    for h, a0 in EXACT_CURVE.pairs:
        breaks |= {a0 - h, a0 + h}
    points = np.array(sorted(b for b in breaks if A_RANGE[0] <= b <= A_RANGE[1]))
    x, wq = np.polynomial.legendre.leggauss(8)
    records = []
    for w in range(3):
        for left, right in itertools.pairwise(points):
            a = 0.5 * (right - left) * x + 0.5 * (right + left)
            mass = 0.5 * (right - left) * wq * P_W[w] * _g_exact(a, np.full(a.size, w))
            q = _q_exact(a, np.full(a.size, w))
            for y, p in ((1.0, q), (0.0, 1.0 - q)):
                records.append(pd.DataFrame({"A": a, "W": float(w), "Y": y, "p": mass * p}))
    return pd.concat(records, ignore_index=True)


class _Density:
    def __init__(self, w: np.ndarray) -> None:
        self.w = w

    def density_at(self, a: np.ndarray) -> np.ndarray:
        return _g_exact(np.asarray(a, dtype=float), self.w)


class _Outcome:
    def __init__(self, observed: np.ndarray, arms: dict[float, np.ndarray]) -> None:
        self.observed = observed
        self.arms = arms


class _Nuisance:
    def __init__(self, data, rule) -> None:  # type: ignore[no-untyped-def]
        w = data.covariates[:, 0]
        a = data.treatment
        self.density = _Density(w)
        self.outcome = _Outcome(
            _q_wrong(a, w),
            {float(k): _q_wrong(np.full(a.size, dose), w) for k, dose in enumerate(rule.nodes)}
            | {float(rule.n_nodes): _q_wrong(np.full(a.size, rule.nodes[-1]), w)},
        )
        from cleverly.estimators.tmle import TMLE

        self.scaler = TMLE(cross_fit=False)._scaler(data)


def _exact_truth(h: float, a0: float) -> float:
    nodes, mass = _grid(*SUPPORT, (*_breaks(h, a0), *EXACT_EDGES))
    weight = equivalent_kernel(nodes, a0, h, SUPPORT, EXACT_CURVE.kernel)
    return float(
        sum(
            P_W[w] * np.sum(mass * weight * _q_exact(nodes, np.full(nodes.size, w)))
            for w in range(3)
        )
    )


def _exact_target(offset: int = 0):  # type: ignore[no-untyped-def]
    estimator = DoseResponseTMLE(EXACT_CURVE, cross_fit=False, fluctuation="linear")
    data = estimator._engine()._prepare(
        _exact_rows(),
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta=None,
        weights="p",
        id=None,
        intermediate=None,
        strata=None,
        treatment_kind="continuous",
    )
    rule = EXACT_CURVE.quadrature(EXACT_EDGES)
    return estimator._target(data, _Nuisance(data, rule), rule, offset, UNCLASSIFIED)  # type: ignore[arg-type]


def test_the_estimators_targeting_recovers_the_truth_from_a_wrong_outcome_regression() -> None:
    """With the true density, the targeted plug-in is the truth exactly, however wrong Q is.

    The remainder is a product with the density error, so it is zero; the score equation then
    makes the plug-in exact.  The untargeted plug-in misses, so the fluctuation is a nonzero
    witness, and this runs through ``DoseResponseTMLE._target`` and ``target_pair``.
    """
    estimates, _, scores = _exact_target()
    for h, a0 in EXACT_CURVE.pairs:
        name = pair_name(h, a0)
        truth = _exact_truth(h, a0)
        assert abs(estimates[name].psi - truth) < 1e-10
        untargeted = sum(
            P_W[w]
            * float(
                np.sum(
                    (rule := EXACT_CURVE.quadrature(EXACT_EDGES)).weights[
                        EXACT_CURVE.pairs.index((h, a0))
                    ]
                    * _q_wrong(rule.nodes, np.full(rule.n_nodes, w))
                )
            )
            for w in range(3)
        )
        assert abs(untargeted - truth) > 1e-3
        assert abs(scores[name]) < 1e-10


def _mutated(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    import cleverly.estimators.dose_response as module

    original = module.target_pair

    def wrapped(y, q_obs, q_nodes, h_obs, h_nodes, *args, **kwargs):  # type: ignore[no-untyped-def]
        if mutation == "drop_inverse_density":
            h_obs = np.sign(h_obs) * np.minimum(np.abs(h_obs), 1.0)
            h_nodes = np.sign(h_nodes) * np.minimum(np.abs(h_nodes), 1.0)
        elif mutation == "untargeted_nodes":
            h_nodes = np.zeros_like(h_nodes)
        return original(y, q_obs, q_nodes, h_obs, h_nodes, *args, **kwargs)

    monkeypatch.setattr(module, "target_pair", wrapped)


@pytest.mark.parametrize("mutation", ["drop_inverse_density", "untargeted_nodes", "node_offset"])
def test_each_targeting_mutation_misses_the_truth(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    if mutation != "node_offset":
        _mutated(monkeypatch, mutation)
    estimates, _, _ = _exact_target(offset=1 if mutation == "node_offset" else 0)
    for h, a0 in EXACT_CURVE.pairs:
        assert abs(estimates[pair_name(h, a0)].psi - _exact_truth(h, a0)) > 1e-6, (h, a0)


def test_an_unsolved_fluctuation_is_a_data_error_a_study_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pair whose Newton steps run out raises ``DataError``, which a study records as a draw.

    A plain ``RuntimeError`` raised in ``cleverly`` stops a repeated-sampling run, and a resumed
    run reaches the same seed and stops again.
    """
    from dataclasses import replace

    import cleverly.estimators.dose_response as module
    from cleverly.validation._failures import recordable

    original = module.target_pair

    def unsolved(*args, **kwargs):  # type: ignore[no-untyped-def]
        return replace(original(*args, **kwargs), converged=False)

    monkeypatch.setattr(module, "target_pair", unsolved)
    with pytest.raises(DataError, match="did not solve its score") as raised:
        _exact_target()
    assert recordable(raised.value)


# ------------------------------------------------------------------ refusals and reports


class _Refusing(LogisticRegression):
    """A learner whose fit raises, so a refusal that fires first is a pre-fit refusal."""

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("a learner was fitted before the refusal")


def _refusing(curve: DoseResponse = FIT_CURVE, **kwargs: object) -> DoseResponseTMLE:
    return DoseResponseTMLE(
        curve,
        outcome_learner=_Refusing(),
        treatment_learner=_Refusing(),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("change", "fit_kwargs", "error", "match"),
    [
        ("missing_outcome", {}, DataError, "missing value"),
        ("discrete", {}, CapabilityError, "distinct values"),
        ("cluster", {"id": "cluster"}, CapabilityError, "clustered rows"),
        ("gaussian", {}, CapabilityError, ""),
    ],
)
def test_refusals_raise_before_any_learner_fits(
    change: str, fit_kwargs: dict[str, str], error: type[Exception], match: str
) -> None:
    frame = _sample(400, 4).assign(cluster=lambda d: np.arange(len(d)) % 20)
    if change == "missing_outcome":
        frame.loc[frame.index[:10], "Y"] = np.nan
    if change == "discrete":
        frame["A"] = np.round(frame["A"]).clip(0, 6)
    if change == "gaussian":
        frame["Y"] = frame["A"] + np.random.default_rng(0).normal(size=len(frame))
    with pytest.raises(error, match=match):
        _refusing().fit(frame, outcome="Y", treatment="A", covariates=["W"], **fit_kwargs)


def test_the_memory_preflight_refuses_before_any_learner_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cleverly.estimators.dose_response as module

    monkeypatch.setattr(module, "NODE_MEMORY_BUDGET", 1024)
    with pytest.raises(DataError, match="GB"):
        _refusing().fit(_sample(400, 4), outcome="Y", treatment="A", covariates=["W"])


def test_the_support_report_reads_the_clever_covariate(fitted) -> None:  # type: ignore[no-untyped-def]
    """Kish size and largest covariate from H = w/g, recomputed from the fit's own rows."""
    frame = _sample(1200, 3)
    for cell in fitted.pair_support:
        weight = FIT_CURVE.weight(cell.bandwidth, cell.dose, frame["A"].to_numpy())
        assert cell.rows == int(np.count_nonzero(weight))
        assert cell.min_density > 0.0
        # |H| = |w| / g, so the largest covariate is at least the largest weight over the
        # largest density and does not scale with the outcome.
        assert cell.max_covariate >= np.abs(weight).max() / 1.0
        assert 1.0 <= cell.kish <= cell.rows


def test_a_quadrature_or_support_note_is_also_a_warning() -> None:
    from cleverly.exceptions import DataWarning

    curve = DoseResponse(doses=(1.0, 2.5), bandwidths=(0.6,), support=(0.5, 4.5), support_floor=1e9)
    with pytest.warns(DataWarning, match="Kish"):
        _fit(_sample(600, 7), curve)


def test_a_two_hundred_pair_declaration_solves_every_score() -> None:
    curve = DoseResponse(
        doses=tuple(np.linspace(0.6, 4.4, 40)),
        bandwidths=(0.5, 0.7, 0.9, 1.1, 1.3),
        support=(0.5, 4.5),
        nodes_per_piece=2,
    )
    result = _fit(_sample(400, 9), curve, density_bins=10)
    assert len(result.estimates) == 200
    for name, score in result.scores.items():
        assert abs(score) <= 1e-5 * result[name].std_error


def test_a_polars_fit_returns_its_frame_in_polars() -> None:
    import polars as pl

    result = _fit(pl.from_pandas(_sample(600, 5)), FIT_CURVE)
    assert result.backend == "polars"
    assert isinstance(result.to_frame(), pl.DataFrame)
    assert isinstance(result.to_frame("pandas"), pd.DataFrame)
