r"""The omitted-variable bound on the regime, shift and MSM axes (X82 part G, G2c).

Each parameter is a linear functional :math:`E[\sum_a c(a, W) \bar Q(a, W)]` or
:math:`E[\bar Q(d(A, W), W)]` of the outcome regression, and the fit's own clever covariate is
its Riesz representer.

=================  =====================================================================
axis               representer
=================  =====================================================================
regime ``r``       :math:`\Delta g^\star_r(A \mid W) / (g(A \mid W) \pi(A, W))`
policy ``s``       :math:`\Delta h_s(A, W) / \pi(A, W)`, the shift's density ratio
MSM ``j``          :math:`e_j^\top M^{-1} \Delta h(A, V) \varphi(A, V) / (g \pi)`
=================  =====================================================================

* :class:`TestTheRieszIdentityOnAnExactLaw` checks
  :math:`\hat\nu^2 = \nu_0^2 - E[(\hat\alpha - \alpha_0)^2]` to ``rtol=1e-12`` on laws that
  a sample realises exactly.  The fitted mechanisms are wrong, so the error is a nonzero
  witness.  The identity holds only when :math:`m(O, \alpha)` reads the same weights as the
  representer.
* :class:`TestTheArmSpecialCases` checks that a static regime and a saturated MSM give the
  arm bound's :math:`\nu^2`.
* :class:`TestTheGramMatrixCurve` compares the curve of :math:`\hat\nu^2` with its Gateaux
  derivative on an MSM whose design reads ``V``.  The Gram matrix term is zero on a marginal
  MSM, and a mutation that drops it departs from the derivative.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import cleverly.sensitivity.omitted_variable as ov
import tests.discrete_law_shift as shift_law
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Rule, Shift, Static, Stochastic
from cleverly.msm import MSM
from tests.unit.test_omitted_variable_part_g import (
    G1,
    PI,
    RTOL,
    _gateaux,
    exact_frame,
    identity_gap,
    learners,
)

pytestmark = pytest.mark.xdist_group("omitted_variable_axes")

#: ``g*(a)`` of the stochastic regime, for ``a = 0`` and ``a = 1``.
COIN = (0.3, 0.7)

#: The regimes of the exact law: two static ones, a rule that treats ``W >= 1``, and a
#: coin, whose density is the one weight a deterministic regime's indicator cannot stand in for.
REGIMES = (
    Static(1, name="always"),
    Static(0, name="never"),
    Rule(lambda w: (w["W"] >= 1).astype(int), name="w1", rule_kind="known"),
    Stochastic(
        lambda w: np.tile([COIN[0], COIN[1]], (len(w), 1)), name="coin", density_kind="known"
    ),
)

#: The shifts of :mod:`tests.discrete_law_shift`, in the order their codes run.
SHIFTS = [
    Shift(0.0, cap=shift_law.CAP, name="natural course"),
    Shift(shift_law.DELTA, cap=shift_law.CAP, name="+1"),
    Shift(shift_law.DELTA, cap=shift_law.CAP_TIGHT, name="+1 (cap 2)"),
]


def _fit(frame: Any, **settings: Any) -> Any:
    return (
        TMLE(**learners(), **settings)
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="D", weights="weight")
        .single()
    )


@pytest.fixture(scope="module")
def regime_fit() -> Any:
    return _fit(exact_frame(), interventions=list(REGIMES))


@pytest.fixture(scope="module")
def arm_fit() -> Any:
    return _fit(exact_frame(), estimands=("ey1", "ey0", "ate"))


@pytest.fixture(scope="module")
def msm_fits() -> dict[str, Any]:
    """A marginal MSM and one whose design reads ``W``."""
    frame = exact_frame()
    return {
        "marginal": _fit(frame, msm=MSM.linear()),
        "modified": _fit(frame, msm=MSM.linear(modifiers=("W",))),
    }


@pytest.fixture(scope="module")
def shift_fit() -> Any:
    """The exact shift law with a binned density, which is wrong at the law's doses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                policies=SHIFTS,
                outcome_learner=LinearRegression(),
                cross_fit=False,
                simultaneous=False,
                density_bins=4,
                random_state=0,
            )
            .fit(
                shift_law.frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                dose_atoms=shift_law.DOSES,
            )
            .single()
        )


# ------------------------------------------------------------------ the true representers


def _law_columns() -> dict[str, np.ndarray]:
    frame = exact_frame()
    w = frame["W"].to_numpy().astype(int)
    a = frame["A"].to_numpy().astype(int)
    g1 = G1[w]
    g = np.where(a == 1, g1, 1.0 - g1)
    return {"w": w, "a": a, "denominator": g * PI[w, a], "D": frame["D"].to_numpy()}


def regime_representers() -> dict[str, np.ndarray]:
    r""":math:`\alpha_0 = \Delta g^\star(A \mid W) / (g_0 \pi_0)` of each regime parameter."""
    c = _law_columns()
    assigned = {"always": np.ones_like(c["a"]), "never": np.zeros_like(c["a"])}
    assigned["w1"] = (c["w"] >= 1).astype(int)
    level = {
        name: c["D"] * (c["a"] == arm).astype(float) / c["denominator"]
        for name, arm in assigned.items()
    }
    level["coin"] = c["D"] * np.where(c["a"] == 1, COIN[1], COIN[0]) / c["denominator"]
    out = {f"ey_regime[{name}]": alpha for name, alpha in level.items()}
    for name in ("never", "w1", "coin"):
        out[f"ate_regime[{name} vs always]"] = level[name] - level["always"]
    return out


def msm_representers(design: str) -> dict[str, np.ndarray]:
    r""":math:`\alpha_0 = e_j^\top M^{-1} \Delta \varphi(A, W) / (g_0 \pi_0)`, with ``h = 1``."""
    c = _law_columns()
    weights = exact_frame()["weight"].to_numpy()
    w, a = c["w"].astype(float), c["a"].astype(float)
    if design == "marginal":
        names = ("(intercept)", "a")

        def phi(arm: np.ndarray) -> np.ndarray:
            return np.column_stack([np.ones_like(arm), arm])
    else:
        names = ("(intercept)", "a", "W", "a:W")

        def phi(arm: np.ndarray) -> np.ndarray:
            return np.column_stack([np.ones_like(arm), arm, w, arm * w])

    gram = sum(
        np.einsum("ip,iq,i->pq", phi(np.full_like(a, arm)), phi(np.full_like(a, arm)), weights)
        for arm in (0.0, 1.0)
    ) / float(weights.sum())
    rows = phi(a) @ np.linalg.inv(gram) * (c["D"] / c["denominator"])[:, None]
    return {f"msm[{name}]": rows[:, j] for j, name in enumerate(names)}


def shift_representers() -> dict[str, np.ndarray]:
    r""":math:`\alpha_0 = g^d(A \mid W) / g(A \mid W)`, the density the policy induces over ``g``."""
    frame = shift_law.frame()
    w = frame["W"].to_numpy().astype(int)
    a = np.rint(frame["A"].to_numpy()).astype(int)

    def ratio(label: str) -> np.ndarray:
        induced = np.zeros_like(shift_law.G)
        for source, target in enumerate(shift_law.POLICIES[label]):
            induced[:, target] += shift_law.G[:, source]
        return np.asarray((induced / shift_law.G)[w, a], dtype=float)

    level = {label: ratio(label) for label in ("natural course", "+1", "+1 (cap 2)")}
    out = {f"ey_policy[{label}]": alpha for label, alpha in level.items()}
    for label in ("+1", "+1 (cap 2)"):
        out[f"ate_policy[{label} vs natural course]"] = level[label] - level["natural course"]
    return out


# ------------------------------------------------------------------------------- tests


class TestTheRieszIdentityOnAnExactLaw:
    """The doubly robust estimate is the truth less the representer's error, on each axis."""

    @pytest.mark.parametrize("estimand", sorted(regime_representers()))
    def test_a_regime(self, regime_fit: Any, estimand: str) -> None:
        gap, error = identity_gap(regime_fit, estimand, regime_representers()[estimand])
        assert error > 0.01  # the nonzero witness: the fitted mechanisms are wrong
        assert gap < RTOL
        assert ov.sensitivity_elements(regime_fit, estimand).features == ("response", "regime")

    @pytest.mark.parametrize("design", ["marginal", "modified"])
    def test_an_msm_coefficient(self, msm_fits: dict[str, Any], design: str) -> None:
        result = msm_fits[design]
        truths = msm_representers(design)
        assert set(truths) == set(result.estimates)
        for estimand, alpha0 in truths.items():
            gap, error = identity_gap(result, estimand, alpha0)
            assert error > 0.01, estimand
            assert gap < RTOL, estimand

    @pytest.mark.parametrize("estimand", sorted(shift_representers()))
    def test_a_shift(self, shift_fit: Any, estimand: str) -> None:
        gap, error = identity_gap(shift_fit, estimand, shift_representers()[estimand])
        if "natural course]" not in estimand:
            assert error > 1e-4  # the binned density is wrong at the law's doses
        assert gap < RTOL
        assert ov.sensitivity_elements(shift_fit, estimand).features == ("shift",)

    def test_the_regime_weight_matters(
        self, regime_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: an ``m`` that drops the regime density breaks the identity."""
        estimand = "ey_regime[coin]"
        alpha0 = regime_representers()[estimand]
        original = ov._axis_m_alpha

        def unweighted(parameter: Any, submodel: Any, axis: Any) -> np.ndarray:
            flat = ov._AxisInputs(regimes=np.ones_like(axis.regimes))
            return original(parameter, submodel, flat)

        monkeypatch.setattr(ov, "_axis_m_alpha", unweighted)
        gap, _ = identity_gap(regime_fit, estimand, alpha0)
        assert gap > 1e-3

    def test_the_shift_reads_the_shifted_dose(
        self, shift_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: an ``m`` read at the observed dose, not at ``d(A, W)``, fails."""
        estimand = "ey_policy[+1]"
        alpha0 = shift_representers()[estimand]
        original = ov._m_alpha

        def at_observed(parameter: Any, submodel: Any, *args: Any, **kwargs: Any) -> np.ndarray:
            if parameter.group == "mtp":
                observed = np.asarray(submodel.observed, dtype=float)
                submodel = replace(submodel, arms=dict.fromkeys(submodel.arms, observed))
            return original(parameter, submodel, *args, **kwargs)

        monkeypatch.setattr(ov, "_m_alpha", at_observed)
        gap, _ = identity_gap(shift_fit, estimand, alpha0)
        assert gap > 1e-3


class TestTheArmSpecialCases:
    """A static regime and a saturated MSM are arm parameters under other names."""

    @pytest.mark.parametrize(
        ("regime", "arm"),
        [
            ("ey_regime[always]", "ey1"),
            ("ey_regime[never]", "ey0"),
            ("ate_regime[never vs always]", "ate"),
        ],
    )
    def test_a_static_regime(self, regime_fit: Any, arm_fit: Any, regime: str, arm: str) -> None:
        assert ov.sensitivity_elements(regime_fit, regime).nu2 == pytest.approx(
            ov.sensitivity_elements(arm_fit, arm).nu2, rel=RTOL
        )

    @pytest.mark.parametrize(
        ("coefficient", "arm"), [("msm[(intercept)]", "ey0"), ("msm[a]", "ate")]
    )
    def test_a_saturated_msm(
        self, msm_fits: dict[str, Any], arm_fit: Any, coefficient: str, arm: str
    ) -> None:
        assert ov.sensitivity_elements(msm_fits["marginal"], coefficient).nu2 == pytest.approx(
            ov.sensitivity_elements(arm_fit, arm).nu2, rel=RTOL
        )


class TestTheGramMatrixCurve:
    """The curve of the MSM's :math:`\\hat\\nu^2` carries the influence of :math:`\\hat M`."""

    @pytest.mark.parametrize("design", ["marginal", "modified"])
    def test_the_curve_is_the_gateaux_derivative(
        self, msm_fits: dict[str, Any], design: str
    ) -> None:
        result = msm_fits[design]
        for name in result.estimates:
            parameter = ov.resolve_parameter(result, name)
            elements = ov.sensitivity_elements(result, name)
            assert elements.psi_nu2 is not None
            expected = _gateaux(result, parameter) * result.data.weights
            np.testing.assert_allclose(elements.psi_nu2, expected, atol=1e-5, err_msg=name)

    def test_the_term_vanishes_on_a_marginal_msm(self, msm_fits: dict[str, Any]) -> None:
        """Each row's Gram contribution is ``M`` when neither ``h`` nor ``phi`` reads ``V``."""
        result = msm_fits["marginal"]
        repeat = result.repeats[0]
        for name in result.estimates:
            parameter = ov.resolve_parameter(result, name)
            elements = ov._elements_for(result, repeat, parameter, "doubly_robust")
            submodel, axis = self._pieces(result, parameter)
            term = ov._gram_influence(
                parameter,
                submodel,
                axis,
                elements.riesz_representer,
                elements.nu2,
                result.data.weights,
            )
            np.testing.assert_allclose(term, 0.0, atol=1e-12)

    def test_dropping_the_term_departs_from_the_derivative(
        self, msm_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = msm_fits["modified"]
        name = "msm[a]"
        expected = _gateaux(result, ov.resolve_parameter(result, name)) * result.data.weights
        monkeypatch.setattr(ov, "_gram_influence", lambda *args: 0.0)
        psi = ov.sensitivity_elements(result, name).psi_nu2
        assert psi is not None
        assert float(np.max(np.abs(psi - expected))) > 1.0

    @staticmethod
    def _pieces(result: Any, parameter: Any) -> tuple[Any, Any]:
        repeat = result.repeats[0]
        data, nuisance, bounds = ov._bound_inputs(result, repeat, parameter.group)
        submodel = ov.build_submodel(
            data,
            nuisance,
            parameter.group,
            bounds=bounds,
            nuisance_bound=result.config.missingness_bound,
        )
        return submodel, ov._AxisInputs.build(parameter, nuisance, data.weights)


class TestTheLimitsStayDescriptive:
    def test_each_axis_is_a_feature(
        self, regime_fit: Any, msm_fits: dict[str, Any], shift_fit: Any
    ) -> None:
        cases = [
            (regime_fit, "ate_regime[w1 vs always]", ("response", "regime")),
            (msm_fits["modified"], "msm[a]", ("response", "msm")),
            (shift_fit, "ey_policy[+1]", ("shift",)),
        ]
        for result, name, features in cases:
            bound = ov.omitted_variable_bounds(result, name, cf_y=0.02, cf_d=0.02)
            assert bound.features == features, name
            assert bound.limit_descriptive, name

    def test_the_plugin_limits_refuse_by_configuration(self, shift_fit: Any) -> None:
        elements = ov.sensitivity_elements(shift_fit, "ey_policy[+1]", nu2_estimator="plugin")
        assert elements.limit_refusal is not None
        assert elements.limit_refusal.key == "configuration"
        assert "modified treatment policy" in elements.limit_refusal.text


class TestTheBenchmarkKeepsTheTarget:
    def test_dropping_a_covariate_the_rule_reads_is_refused(self, regime_fit: Any) -> None:
        """The exact law has one covariate, so the rule's refusal comes before the refit."""
        with pytest.raises(CapabilityError, match="cannot drop every covariate"):
            ov.benchmark(regime_fit, ["W"], estimand="ey_regime[w1]")

    def test_a_refit_that_moves_the_regime_is_refused(self) -> None:
        from cleverly.datasets import make_binary_outcome
        from tests.conftest import IN_SAMPLE, fast_tmle

        frame, _ = make_binary_outcome(n=300, seed=17)
        rule = Rule(lambda w: (w["W1"] > 0).astype(int), name="w1", rule_kind="known")
        result = (
            fast_tmle(**IN_SAMPLE, interventions=[Static(1, name="always"), rule])
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        with pytest.raises(CapabilityError, match="declared regime reads them"):
            ov.benchmark(result, ["W1"], estimand="ey_regime[w1]")
        record = ov.benchmark(result, ["W2"], estimand="ey_regime[w1]")
        assert record.features == ("regime",)
