r"""The omitted-variable bound configurations of X82 part G (G2a).

Part G lifts three refusals of the bound and adds the stratum-conditional parameter.  Each
configuration supplies a representer that also reads a second mechanism or a stratum share:

=================  ==============================================================
configuration      representer
=================  ==============================================================
response           :math:`\Delta 1\{A = a\} / (g_a \pi_a)`
missing treatment  :math:`C_a / g_{c,a}`, with :math:`C_a = \Delta_A \Delta 1\{A = a\}`
intermediate       :math:`\Delta 1\{A = a\} 1\{Z = z\} / (g_a q_z \pi_a)`
stratum            the marginal representer times :math:`1\{S = s\}` over the share
=================  ==============================================================

* :class:`TestTheRieszIdentityOnAnExactLaw` checks
  :math:`\hat\nu^2 = \nu_0^2 - E[(\hat\alpha - \alpha_0)^2]` to ``rtol=1e-12`` on laws that
  a weighted sample realises exactly.  The fitted mechanisms are wrong on purpose, so the
  error term is a nonzero witness.  The identity holds when :math:`m(O, \alpha)` and the
  representer read the same mechanisms.  It is blind to a representer value off the support
  of :math:`\alpha_0`, so the next class checks the support.
* :class:`TestTheOracleRepresenter` fits the law's own mechanisms, where the representer is
  :math:`\alpha_0` row by row and :math:`\hat\nu^2 = \nu_0^2`.  Each weight
  (:math:`\Delta`, :math:`1\{Z = z\}`, :math:`\Delta_A`, :math:`1\{S = s\}`) has a mutation
  that drops it and moves the representer off :math:`\alpha_0`.
* :class:`TestTheResidualVariance` checks that :math:`\sigma^2` averages the masked residual
  over every row, with a mutation that averages the respondents instead.
* :class:`TestTheStratumShareCurve` compares the curve of :math:`\hat\nu^2` with a
  finite-difference Gateaux derivative in the row weights, and a mutation drops the share
  term.  The marginal ATT, whose term shipped before part G, is the control.
* :class:`TestOneStratumIsTheMarginalBound` checks that a stratum holding every row gives
  the marginal bound.
* :class:`TestTheLimitReading` checks that every part G limit resolves through the
  ``ovb_limit`` registry rows and prints as descriptive on the shipped registry.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.sensitivity.omitted_variable as ov
from cleverly import ATE, ATT, CausalStudy, PointTreatment
from cleverly.datasets import make_missing_outcome
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity._parameters import ArmParameter
from cleverly.validation import _interpretation as interpretation
from cleverly.validation._interpretation import (
    DESCRIPTIVE_SENTENCE,
    CalibrationScope,
    VerdictEvidence,
)
from tests import discrete_law_composite as composite_law
from tests.studies.mar_arm_indexed_laws import LawOutcome, LawResponse, LawTreatment
from tests.unit._direct_effect_support import cde_frame, fit_cde

pytestmark = pytest.mark.xdist_group("omitted_variable_part_g")

#: The relative tolerance of every exact-law identity.  The probe measured at most 5e-16.
RTOL = 1e-12

#: ``P(W = w)`` of the weighted exact law.
P_W = np.array([0.5, 0.3, 0.2])

#: ``P(A = 1 | W = w)``.
G1 = np.array([0.4, 0.6, 0.25])

#: ``P(Delta = 1 | A = a, W = w)``, indexed ``[w, a]``.  It does not read ``Z``.
PI = np.array([[0.7, 0.9], [0.5, 0.8], [0.85, 0.6]])

#: ``P(Z = 1 | A = a, W = w)``, indexed ``[w, a]``.
QZ = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.2]])


def outcome_risk(a: int, z: int, w: int) -> float:
    """``P(Y = 1 | A = a, Z = z, W = w, Delta = 1)``, with an ``A`` by ``W`` interaction."""
    return float(expit(-0.5 + 0.8 * a + 0.6 * z + 0.4 * w - 0.3 * a * w))


def exact_frame(*, intermediate: bool = False) -> pd.DataFrame:
    """One row per support point, weighted by its probability.

    The weighted empirical law *is* the law, so every weighted mean below is an
    expectation under it.  ``V`` marks ``W = 2`` and serves as a baseline stratum.
    """
    rows = []
    levels = range(2) if intermediate else range(1)
    for w, a, z, d, y in itertools.product(range(3), range(2), levels, range(2), range(2)):
        if d == 0 and y == 1:
            continue
        p = P_W[w] * (G1[w] if a else 1.0 - G1[w])
        if intermediate:
            p *= QZ[w, a] if z else 1.0 - QZ[w, a]
        p *= PI[w, a] if d else 1.0 - PI[w, a]
        if d:
            risk = outcome_risk(a, z, w)
            p *= risk if y else 1.0 - risk
        rows.append(
            {
                "W": float(w),
                "V": float(w == 2),
                "A": float(a),
                "Z": float(z),
                "D": float(d),
                "Y": float(y) if d else np.nan,
                "weight": p,
            }
        )
    frame = pd.DataFrame(rows)
    return frame if intermediate else frame.drop(columns="Z")


class TableProbability(BaseEstimator):
    """A binary classifier that reads ``P(label = 1)`` from a table of the law.

    ``treatment`` reads ``P(A = 1 | W)`` from the first design column.  Otherwise the design
    is ``[A, W, ...]`` and the table is indexed ``[w, a]``.
    """

    def __init__(self, table: Any, treatment: bool = False) -> None:
        self.table = table
        self.treatment = treatment

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TableProbability:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        table = np.asarray(self.table, dtype=float)
        if self.treatment:
            p = table[np.rint(design[:, 0]).astype(int)]
        else:
            a = np.rint(design[:, 0]).astype(int)
            p = table[np.rint(design[:, 1]).astype(int), a]
        return np.column_stack([1.0 - p, p])


def oracle_learners() -> dict[str, Any]:
    """In-sample learners that return the law's own mechanisms."""
    return {
        "outcome_learner": LogisticRegression(max_iter=2000),
        "treatment_learner": TableProbability(G1, treatment=True),
        "missingness_learner": TableProbability(PI),
        "intermediate_learner": TableProbability(QZ),
        "g_bounds": (0.01, 0.99),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }


def learners() -> dict[str, Any]:
    """In-sample learners whose mechanisms are wrong on the law: a strong ridge penalty."""
    return {
        "outcome_learner": LogisticRegression(max_iter=2000),
        "treatment_learner": LogisticRegression(C=0.05, max_iter=2000),
        "missingness_learner": LogisticRegression(C=0.05, max_iter=2000),
        "intermediate_learner": LogisticRegression(C=0.05, max_iter=2000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }


def response_fit(*, oracle: bool = False) -> Any:
    settings = oracle_learners() if oracle else learners()
    return (
        TMLE(estimands=("ey1", "ey0", "ate", "att", "atc"), **settings)
        .fit(
            exact_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="D",
            weights="weight",
        )
        .single()
    )


def intermediate_fits(*, oracle: bool = False) -> Any:
    settings = oracle_learners() if oracle else learners()
    return TMLE(estimands=("ey1", "ey0", "ate", "att", "atc"), **settings).fit(
        exact_frame(intermediate=True),
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="D",
        intermediate="Z",
        weights="weight",
    )


def composite_fit(*, oracle: bool = False) -> Any:
    law = composite_law.TWO
    tables = law.law()
    return (
        TMLE(
            estimands=("ey0", "ey1", "ate"),
            outcome_learner=LawOutcome(tables) if oracle else LogisticRegression(max_iter=2000),
            treatment_learner=LawTreatment(tables) if oracle else LogisticRegression(max_iter=2000),
            missingness_learner=(
                LawResponse(tables) if oracle else LogisticRegression(max_iter=2000)
            ),
            g_bounds=(0.01, 0.99),
            nuisance_bound=0.01,
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        .fit(
            law.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
        .single()
    )


def stratum_fit(target: Any, *, missing: bool = True, oracle: bool = False) -> Any:
    frame = exact_frame()
    if not missing:
        frame = frame[frame["D"] == 1.0].drop(columns="D")
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W", "V"),
        strata=("V",),
        missingness="D" if missing else None,
        weights="weight",
    )
    settings = oracle_learners() if oracle else learners()
    settings.pop("intermediate_learner")
    if not missing:
        settings.pop("missingness_learner")
    return CausalStudy(frame, design=design).identify(target).estimate(**settings)


@pytest.fixture(scope="module")
def response() -> Any:
    return response_fit()


@pytest.fixture(scope="module")
def intermediate() -> Any:
    return intermediate_fits()


@pytest.fixture(scope="module")
def composite() -> Any:
    return composite_fit()


@pytest.fixture(scope="module")
def stratified() -> dict[str, Any]:
    return {"ate": stratum_fit(ATE()), "att": stratum_fit(ATT())}


@pytest.fixture(scope="module")
def oracle() -> dict[str, Any]:
    """The four configurations fitted with the law's own mechanisms."""
    return {
        "response": response_fit(oracle=True),
        "intermediate": intermediate_fits(oracle=True),
        "composite": composite_fit(oracle=True),
        "stratum": stratum_fit(ATT(), oracle=True),
    }


# ------------------------------------------------------------------ the true representers


def _columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    w = frame["W"].to_numpy().astype(int)
    g1 = G1[w]
    return {
        "w": w,
        "A": frame["A"].to_numpy(),
        "D": frame["D"].to_numpy(),
        "g": np.column_stack([1.0 - g1, g1]),
        "pi": PI[w],
    }


def response_representers(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, np.ndarray]:
    """:math:`\\alpha_0` of each parameter of :func:`response_fit`, from the law's tables."""
    c = _columns(frame)
    a, d, g, pi = c["A"], c["D"], c["g"], c["pi"]
    arm = {k: d * (a == k) / (g[:, k] * pi[:, k]) for k in (0, 1)}
    p1 = float(np.average(a, weights=weights))
    p0 = 1.0 - p1
    return {
        "ey1": arm[1],
        "ey0": arm[0],
        "ate": arm[1] - arm[0],
        "att": d * ((a == 1) / (p1 * pi[:, 1]) - (a == 0) * g[:, 1] / (p1 * g[:, 0] * pi[:, 0])),
        "atc": d * ((a == 1) * g[:, 0] / (p0 * g[:, 1] * pi[:, 1]) - (a == 0) / (p0 * pi[:, 0])),
    }


def intermediate_representers(
    frame: pd.DataFrame, level: float, weights: np.ndarray
) -> dict[str, np.ndarray]:
    c = _columns(frame)
    a, g = c["A"], c["g"]
    z = frame["Z"].to_numpy()
    qz = QZ[c["w"]] if level == 1.0 else 1.0 - QZ[c["w"]]
    kept = (c["D"] * (z == level))[:, None] / (c["pi"] * qz)
    arm = {k: (a == k) * kept[:, k] / g[:, k] for k in (0, 1)}
    p1 = float(np.average(a, weights=weights))
    p0 = 1.0 - p1
    return {
        "ey1": arm[1],
        "ey0": arm[0],
        "ate": arm[1] - arm[0],
        "att": ((a == 1) * kept[:, 1] - (a == 0) * kept[:, 0] * g[:, 1] / g[:, 0]) / p1,
        "atc": ((a == 1) * kept[:, 1] * g[:, 0] / g[:, 1] - (a == 0) * kept[:, 0]) / p0,
    }


def composite_representers() -> dict[str, np.ndarray]:
    law = composite_law.TWO
    frame = law.frame()
    w = frame["W"].to_numpy().astype(int)
    recorded = (frame["DeltaA"].to_numpy() == 1) & (frame["Delta"].to_numpy() == 1)
    arm = {k: (recorded & (frame["A"].to_numpy() == k)) / law.composite[w, k] for k in (0, 1)}
    return {"ey1": arm[1], "ey0": arm[0], "ate": arm[1] - arm[0]}


def stratum_representer(frame: pd.DataFrame, weights: np.ndarray, key: Any) -> np.ndarray:
    c = _columns(frame)
    a, d, g, pi = c["A"], c["D"], c["g"], c["pi"]
    s = (frame["V"].to_numpy() == key.stratum[0]).astype(float)
    if key.estimand == "ate":
        share = float(np.average(s, weights=weights))
        return (
            s / share * (d * (a == 1) / (g[:, 1] * pi[:, 1]) - d * (a == 0) / (g[:, 0] * pi[:, 0]))
        )
    share = float(np.average(s * (a == 1), weights=weights))
    return (
        s * d * ((a == 1) / (share * pi[:, 1]) - (a == 0) * g[:, 1] / (share * g[:, 0] * pi[:, 0]))
    )


def identity_gap(result: Any, estimand: str, alpha0: np.ndarray) -> tuple[float, float]:
    """The relative gap in :math:`\\hat\\nu^2 = \\nu_0^2 - E[(\\hat\\alpha - \\alpha_0)^2]`, and the error."""
    weights = result.data.weights
    elements = ov.sensitivity_elements(result, estimand)
    error = float(np.average((elements.riesz_representer - alpha0) ** 2, weights=weights))
    truth = float(np.average(alpha0**2, weights=weights))
    return abs(elements.nu2 - (truth - error)) / elements.nu2, error


def stratum_names(result: Any, stem: str) -> list[str]:
    return [
        name
        for name, key in result.parameter_keys.items()
        if key.stratum is not None and key.estimand == stem
    ]


# ------------------------------------------------------------------------------- tests


class TestTheRieszIdentityOnAnExactLaw:
    r"""The doubly robust :math:`\hat\nu^2` is :math:`\nu_0^2` less the representer's error.

    The identity holds exactly only when :math:`m(O, \alpha)` and the representer use the
    same rows and the same mechanisms, so it checks each configuration's weights.
    """

    @pytest.mark.parametrize("estimand", ["ey1", "ey0", "ate", "att", "atc"])
    def test_a_response_mechanism(self, response: Any, estimand: str) -> None:
        alpha0 = response_representers(exact_frame(), response.data.weights)[estimand]
        gap, error = identity_gap(response, estimand, alpha0)
        assert error > 0.1  # the nonzero witness: the fitted mechanisms are wrong
        assert gap < RTOL

    @pytest.mark.parametrize("level", [0.0, 1.0])
    @pytest.mark.parametrize("estimand", ["ey1", "ey0", "ate", "att", "atc"])
    def test_an_intermediate_at_each_level(
        self, intermediate: Any, level: float, estimand: str
    ) -> None:
        result = intermediate[level]
        frame = exact_frame(intermediate=True)
        alpha0 = intermediate_representers(frame, level, result.data.weights)[estimand]
        gap, error = identity_gap(result, estimand, alpha0)
        assert error > 0.1
        assert gap < RTOL
        assert ov.sensitivity_elements(result, estimand).features == ("response", "intermediate")

    @pytest.mark.parametrize("estimand", ["ey1", "ey0", "ate"])
    def test_a_missing_treatment(self, composite: Any, estimand: str) -> None:
        gap, error = identity_gap(composite, estimand, composite_representers()[estimand])
        assert error > 0.01
        assert gap < RTOL
        assert ov.sensitivity_elements(composite, estimand).features == (
            "response",
            "missing_treatment",
        )

    @pytest.mark.parametrize("stem", ["ate", "att"])
    def test_a_stratum(self, stratified: dict[str, Any], stem: str) -> None:
        result = stratified[stem]
        names = stratum_names(result, stem)
        assert len(names) == 2
        for name in names:
            key = result.parameter_keys[name]
            alpha0 = stratum_representer(exact_frame(), result.data.weights, key)
            gap, error = identity_gap(result, name, alpha0)
            assert error > 0.1, name
            assert gap < RTOL, name
            assert ov.sensitivity_elements(result, name).features == ("response", "stratum")


def oracle_cases(oracle: dict[str, Any]) -> list[tuple[str, Any, str, np.ndarray]]:
    """``(label, fit, estimand, alpha_0)`` for every configuration at the oracle."""
    response = oracle["response"]
    cases = [
        (f"response-{name}", response, name, alpha0)
        for name, alpha0 in response_representers(exact_frame(), response.data.weights).items()
    ]
    for level in (0.0, 1.0):
        frame = exact_frame(intermediate=True)
        cases += [
            (f"intermediate-{level:g}-{name}", oracle["intermediate"][level], name, alpha0)
            for name, alpha0 in intermediate_representers(
                frame, level, oracle["intermediate"][level].data.weights
            ).items()
        ]
    cases += [
        (f"composite-{name}", oracle["composite"], name, alpha0)
        for name, alpha0 in composite_representers().items()
    ]
    stratum = oracle["stratum"]
    cases += [
        (
            f"stratum-{name}",
            stratum,
            name,
            stratum_representer(exact_frame(), stratum.data.weights, stratum.parameter_keys[name]),
        )
        for name in stratum_names(stratum, "att")
    ]
    return cases


class TestTheOracleRepresenter:
    """At the law's own mechanisms the representer is :math:`\alpha_0` on every row."""

    def test_every_configuration_reads_the_true_representer(self, oracle: dict[str, Any]) -> None:
        for label, result, estimand, alpha0 in oracle_cases(oracle):
            elements = ov.sensitivity_elements(result, estimand)
            np.testing.assert_allclose(
                elements.riesz_representer, alpha0, rtol=RTOL, atol=0.0, err_msg=label
            )
            truth = float(np.average(alpha0**2, weights=result.data.weights))
            assert elements.nu2 == pytest.approx(truth, rel=RTOL), label
            # Nonzero witness: each representer is zero on some rows and not on others.
            assert 0 < np.count_nonzero(alpha0) < alpha0.size, label

    def _departure(self, result: Any, estimand: str, alpha0: np.ndarray) -> float:
        """The largest row distance from :math:`\alpha_0`.

        The plug-in estimator reads the same representer and never refuses on its sign,
        so a mutation that makes the doubly robust estimate negative still reports here.
        """
        elements = ov.sensitivity_elements(result, estimand, nu2_estimator="plugin")
        return float(np.max(np.abs(elements.riesz_representer - alpha0)))

    def test_dropping_the_response_indicator(
        self, oracle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = oracle["response"]
        alpha0 = response_representers(exact_frame(), result.data.weights)["ate"]
        monkeypatch.setattr(ov, "_representer_rows", lambda data: np.ones(data.n))
        assert self._departure(result, "ate", alpha0) > 0.5

    def test_dropping_the_level_of_the_intermediate(
        self, oracle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The level enters the shipped covariate through ``selection``."""
        import cleverly.estimators.targeting as targeting

        result = oracle["intermediate"][1.0]
        frame = exact_frame(intermediate=True)
        alpha0 = intermediate_representers(frame, 1.0, result.data.weights)["ate"]
        original = targeting.clever_covariate_inputs

        def unselected(*args: Any, **kwargs: Any) -> Any:
            density, _ = original(*args, **kwargs)
            return density, None

        monkeypatch.setattr(targeting, "clever_covariate_inputs", unselected)
        assert self._departure(result, "ate", alpha0) > 0.5

    def test_reading_the_raw_data_on_a_composite_fit(self, oracle: dict[str, Any]) -> None:
        """The raw data drop the treatment observation factor and read unrecorded rows."""
        result = oracle["composite"]
        mutated = replace(result, extra={**result.extra, "missing_data": "complete"})
        assert self._departure(mutated, "ate", composite_representers()["ate"]) > 0.5

    def test_dropping_the_stratum_indicator(
        self, oracle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = oracle["stratum"]
        name = stratum_names(result, "att")[0]
        key = result.parameter_keys[name]
        alpha0 = stratum_representer(exact_frame(), result.data.weights, key)
        monkeypatch.setattr(ov, "_stratum_rows", lambda result, parameter: None)
        assert self._departure(result, name, alpha0) > 0.5


class TestTheResidualVariance:
    r""":math:`\sigma^2` is :math:`E[R (Y - \bar Q^*)^2]` over every row, ``R`` the support."""

    @staticmethod
    def _masked(result: Any, group: str, rows: np.ndarray) -> float:
        data = result.data
        targeted = result.repeats[0].fluctuations[group].targeted.observed
        outcome = np.where(rows, np.nan_to_num(data.outcome), 0.0)
        residual = np.where(rows, outcome - targeted, 0.0)
        return float(np.average(residual**2, weights=data.weights))

    def test_a_response_mechanism_averages_over_every_row(self, response: Any) -> None:
        rows = np.asarray(response.data.observed, dtype=bool)
        sigma2 = ov.sensitivity_elements(response, "ate").sigma2
        assert sigma2 == pytest.approx(self._masked(response, "mean", rows), rel=RTOL)
        # Mutation: the respondents' mean is larger by 1 / P(Delta = 1).
        share = float(np.average(rows, weights=response.data.weights))
        assert share < 0.8
        assert abs(sigma2 / share - sigma2) > 0.2 * sigma2

    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_an_intermediate_masks_the_level(
        self, intermediate: Any, level: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = intermediate[level]
        data = result.data
        rows = np.asarray(data.observed, dtype=bool) & (data.intermediate == level)
        sigma2 = ov.sensitivity_elements(result, "ate").sigma2
        assert sigma2 == pytest.approx(self._masked(result, "mean", rows), rel=RTOL)
        # Mutation: a mask without the level of Z averages the other level's residuals too.
        monkeypatch.setattr(ov, "targeted_rows", lambda data, value: data.observed)
        assert abs(ov.sensitivity_elements(result, "ate").sigma2 - sigma2) > 0.05 * sigma2


def _gateaux(result: Any, parameter: ArmParameter, step: float = 1e-6) -> np.ndarray:
    """Central differences of :math:`\\hat\\nu^2` under a contamination of each row.

    The nuisances stay fixed.  The contamination :math:`(1 - t) P_n + t \\delta_j` moves the
    normalised row weights, so its derivative is the curve divided by the row's weight.
    """
    data = result.data
    base = np.asarray(data.weights, dtype=float) / float(np.sum(data.weights))
    repeat = result.repeats[0]

    def nu2(weights: np.ndarray) -> float:
        moved = replace(result, data=replace(data, weights=weights * data.n))
        return ov._elements_for(moved, repeat, parameter, "doubly_robust").nu2

    derivative = np.empty(data.n)
    for row in range(data.n):
        point = np.zeros(data.n)
        point[row] = 1.0
        up = nu2((1.0 - step) * base + step * point)
        down = nu2((1.0 + step) * base - step * point)
        derivative[row] = (up - down) / (2.0 * step)
    return derivative


class TestTheStratumShareCurve:
    """The curve of :math:`\\hat\\nu^2` carries the influence of the stratum's share."""

    @pytest.mark.parametrize("stem", ["ate", "att"])
    def test_the_curve_is_the_gateaux_derivative(
        self, stratified: dict[str, Any], stem: str
    ) -> None:
        result = stratified[stem]
        for name in [stem, *stratum_names(result, stem)]:
            parameter = ov.resolve_parameter(result, name)
            elements = ov.sensitivity_elements(result, name)
            assert elements.psi_nu2 is not None
            expected = _gateaux(result, parameter) * result.data.weights
            np.testing.assert_allclose(elements.psi_nu2, expected, atol=1e-5, err_msg=name)

    def test_dropping_the_share_term_departs_from_the_derivative(
        self, stratified: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = stratified["ate"]
        name = stratum_names(result, "ate")[0]
        expected = _gateaux(result, ov.resolve_parameter(result, name)) * result.data.weights
        monkeypatch.setattr(
            ov, "_conditioning_share_influence", lambda nu2, indicator, share, weights: 0.0
        )
        psi = ov.sensitivity_elements(result, name).psi_nu2
        assert psi is not None
        assert float(np.max(np.abs(psi - expected))) > 1.0


class TestOneStratumIsTheMarginalBound:
    """A stratum that holds every row is the marginal parameter, bit for bit up to rounding."""

    @pytest.mark.parametrize("stem", ["ate", "att"])
    def test_every_piece_matches(
        self, stratified: dict[str, Any], stem: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = stratified[stem]
        marginal = ov.sensitivity_elements(result, stem)
        name = stratum_names(result, stem)[0]
        monkeypatch.setattr(ov, "_stratum_rows", lambda result, parameter: np.ones(result.data.n))
        whole = ov.sensitivity_elements(result, name)
        assert whole.sigma2 == pytest.approx(marginal.sigma2, rel=RTOL)
        assert whole.nu2 == pytest.approx(marginal.nu2, rel=RTOL)
        np.testing.assert_allclose(whole.riesz_representer, marginal.riesz_representer, rtol=RTOL)
        assert whole.psi_nu2 is not None and marginal.psi_nu2 is not None
        np.testing.assert_allclose(whole.psi_nu2, marginal.psi_nu2, rtol=1e-10, atol=1e-12)


class TestThePluginEstimatorRefusesTheLimits:
    """The stack carries the treatment mechanism's score alone, so the limits refuse."""

    def test_a_response_bound_reports_points_and_refuses_limits(self, response: Any) -> None:
        bound = ov.omitted_variable_bounds(response, "ate", nu2_estimator="plugin")
        assert bound.limit_refusal is not None
        assert bound.limit_refusal.key == "configuration"
        assert "also divides by the response mechanism" in bound.limit_refusal.text
        assert np.isfinite(bound.lower) and np.isfinite(bound.upper)
        with pytest.raises(CapabilityError):
            _ = bound.ci_lower

    def test_a_stratum_bound_names_its_share(self, stratified: dict[str, Any]) -> None:
        name = stratum_names(stratified["ate"], "ate")[0]
        bound = ov.omitted_variable_bounds(stratified["ate"], name, nu2_estimator="plugin")
        assert bound.limit_refusal is not None
        assert "restricted to a baseline stratum" in bound.limit_refusal.text


class TestTheJointStrengthSentence:
    """The summary says which mechanisms ``cf_d`` measures a joint strength over."""

    def test_each_configuration_names_its_mechanisms(
        self, response: Any, composite: Any, intermediate: Any
    ) -> None:
        expected = {
            "the response mechanism.": response,
            "the response and recording mechanisms.": composite,
            "the response and intermediate mechanisms.": intermediate[0.0],
        }
        for tail, result in expected.items():
            summary = ov.omitted_variable_bounds(result, "ate").summary()
            assert f"joint strength over the treatment mechanism and {tail}" in summary

    def test_the_shipped_bound_prints_no_sentence(self) -> None:
        frame, _ = make_missing_outcome(n=300, seed=4)
        frame = frame.drop(columns="Delta").dropna()
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        bound = ov.omitted_variable_bounds(result, "ate")
        assert bound.features == ()
        assert bound.limit_reading is None
        assert "joint strength" not in bound.summary()
        assert "confidence limits:" not in bound.summary()
        assert "features" not in bound.to_dict()
        assert "limit_licensed" not in ov.robustness_value(result, "ate")


def every_configuration(
    response: Any, composite: Any, intermediate: Any, stratified: dict[str, Any]
) -> dict[str, Any]:
    """One call per ``ovb_limit`` rule this sub-PR adds, keyed by the rule."""
    cde = fit_cde(cde_frame(), ("ate",))[0.0]
    complete_stratum = stratum_fit(ATE(), missing=False)
    stratum_name = stratum_names(stratified["ate"], "ate")[0]
    complete_name = stratum_names(complete_stratum, "ate")[0]
    return {
        "response": lambda: ov.omitted_variable_bounds(response, "ate"),
        "response+missing_treatment": lambda: ov.omitted_variable_bounds(composite, "ate"),
        "response+intermediate": lambda: ov.omitted_variable_bounds(intermediate[1.0], "ate"),
        "intermediate": lambda: ov.omitted_variable_bounds(cde, "ate"),
        "response+stratum": lambda: ov.omitted_variable_bounds(stratified["ate"], stratum_name),
        "stratum": lambda: ov.omitted_variable_bounds(complete_stratum, complete_name),
    }


def covering_row(rule: str) -> VerdictEvidence:
    """A registry row that licenses one part G configuration on every fit."""
    return VerdictEvidence(
        instrument=ov.OVB_LIMIT_INSTRUMENT,
        rule=rule,
        rung="L2",
        theory="test row",
        scope=CalibrationScope.unrestricted(),
        sentence="covered by the test row",
        exact=True,
    )


class TestTheLimitReading:
    """Every part G limit resolves through the registry, one rule per configuration."""

    @pytest.fixture(scope="class")
    def calls(
        self, response: Any, composite: Any, intermediate: Any, stratified: dict[str, Any]
    ) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return every_configuration(response, composite, intermediate, stratified)

    def test_every_configuration_is_descriptive_on_the_shipped_registry(
        self, calls: dict[str, Any]
    ) -> None:
        assert not [
            row
            for row in interpretation.VERDICT_EVIDENCE
            if row.instrument == ov.OVB_LIMIT_INSTRUMENT
        ]
        for rule, call in calls.items():
            bound = call()
            assert "+".join(bound.features) == rule
            assert bound.limit_reading is not None, rule
            assert not bound.limit_reading.licensed, rule
            assert bound.limit_reading.sentence.startswith(DESCRIPTIVE_SENTENCE), rule
            assert bound.limit_descriptive, rule
            assert bound.to_dict()["limit_licensed"] is False, rule
            assert f"confidence limits: {DESCRIPTIVE_SENTENCE}" in bound.summary(), rule

    def test_a_registered_row_licenses_exactly_its_configuration(
        self, calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: each row flips its own rule and leaves the others descriptive."""
        for rule in calls:
            monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (covering_row(rule),))
            readings = {name: call() for name, call in calls.items()}
            licensed = readings.pop(rule)
            assert licensed.limit_reading.licensed, rule
            assert not licensed.limit_descriptive, rule
            assert "confidence limits: covered by the test row" in licensed.summary()
            assert all(bound.limit_descriptive for bound in readings.values()), rule


def test_the_benchmark_refits_a_response_fit() -> None:
    """The short refit drops the covariate from every mechanism the representer reads."""
    frame, _ = make_missing_outcome(n=400, seed=5)
    result = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )
    record = ov.benchmark(result, ["W1"], estimand="ate")
    assert 0.0 <= record.cf_y <= 1.0
    assert 0.0 <= record.cf_d <= 1.0
    assert record.psi_short != record.psi_long
    bound = ov.omitted_variable_bounds(result, "ate", benchmark=record)
    assert bound.features == ("response",)
    assert record.features == ("response",)
    assert "joint strength over the treatment mechanism and the response" in record.summary()
    assert record.to_dict()["features"] == "response"
