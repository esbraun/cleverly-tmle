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
from sklearn.linear_model import LinearRegression, LogisticRegression

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


# --------------------------------------------------- a conditional contrast's projection


def _projection(basis: Any, modifiers: tuple[str, ...], scale: str = "difference") -> Any:
    from cleverly.bases import Modifiers
    from cleverly.effect_modification import EffectModification

    return EffectModification(basis, Modifiers(list(modifiers)), scale)  # type: ignore[arg-type]


def _linear(scale: str = "difference") -> Any:
    from cleverly.bases import Linear

    return _projection(Linear(["W"]), ("W",), scale)


def _indicators(*, intercept: bool) -> Any:
    from cleverly.bases import Indicators

    return _projection(Indicators("V", levels=(0.0, 1.0), intercept=intercept), ("V",))


def projection_fit(
    projection: Any, *, oracle: bool = False, covariates: tuple[str, ...] = ("W",), **extra: Any
) -> Any:
    from tests.unit.test_omitted_variable_part_g import oracle_learners

    settings = oracle_learners() if oracle else learners()
    settings.pop("intermediate_learner")
    return (
        TMLE(**settings, effect_modification=projection, **extra)
        .fit(
            exact_frame(),
            outcome="Y",
            treatment="A",
            covariates=list(covariates),
            delta="D",
            weights="weight",
        )
        .single()
    )


def projection_intermediate_fits(*, oracle: bool = False) -> Any:
    from tests.unit.test_omitted_variable_part_g import oracle_learners

    settings = oracle_learners() if oracle else learners()
    return TMLE(**settings, effect_modification=_linear()).fit(
        exact_frame(intermediate=True),
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="D",
        intermediate="Z",
        weights="weight",
    )


def projection_composite_fit() -> Any:
    from tests.unit.test_omitted_variable_part_g import composite_law

    return (
        TMLE(
            outcome_learner=LogisticRegression(max_iter=2000),
            treatment_learner=LogisticRegression(max_iter=2000),
            missingness_learner=LogisticRegression(max_iter=2000),
            g_bounds=(0.01, 0.99),
            nuisance_bound=0.01,
            cross_fit=False,
            simultaneous=False,
            random_state=0,
            effect_modification=_linear(),
        )
        .fit(
            composite_law.TWO.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
        .single()
    )


@pytest.fixture(scope="module")
def projections() -> dict[str, Any]:
    """The difference-scale projection on ``Linear(W)``: complete, CDE at each level, composite."""
    levels = projection_intermediate_fits()
    return {
        "response": projection_fit(_linear()),
        "intermediate_0": levels[0.0],
        "intermediate_1": levels[1.0],
        "composite": projection_composite_fit(),
    }


def _gram_rows(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``M^{-1} phi(V)`` at each row, with ``M = P_w[phi phi']`` over every row."""
    gram = np.einsum("ip,iq,i->pq", basis, basis, weights) / float(weights.sum())
    return np.asarray(basis @ np.linalg.inv(gram))


def projection_representers(name: str) -> dict[str, np.ndarray]:
    r""":math:`\alpha_0 = e_j^\top M^{-1} \varphi(W)\, \alpha_{0,\mathrm{ate}}` of each coefficient."""
    from tests.unit.test_omitted_variable_part_g import (
        composite_law,
        composite_representers,
        intermediate_representers,
        response_representers,
    )

    if name == "composite":
        frame = composite_law.TWO.frame()
        weights = np.ones(len(frame))
        ate = composite_representers()["ate"]
    elif name.startswith("intermediate"):
        frame = exact_frame(intermediate=True)
        weights = frame["weight"].to_numpy()
        ate = intermediate_representers(frame, float(name[-1]), weights)["ate"]
    else:
        frame = exact_frame()
        weights = frame["weight"].to_numpy()
        ate = response_representers(frame, weights)["ate"]
    w = frame["W"].to_numpy().astype(float)
    rows = _gram_rows(np.column_stack([np.ones_like(w), w]), weights)
    return {
        "cate_projection[(intercept)]": rows[:, 0] * ate,
        "cate_projection[W]": rows[:, 1] * ate,
    }


#: The features of each projection fit's bound.
PROJECTION_FEATURES = {
    "response": ("response", "projection"),
    "intermediate_0": ("response", "intermediate", "projection"),
    "intermediate_1": ("response", "intermediate", "projection"),
    "composite": ("response", "missing_treatment", "projection"),
}


class TestTheProjectionAxis:
    r"""The bound of a difference-scale projection coefficient (P1 PR 3, d3).

    The coefficient :math:`e_j^\top M^{-1} E[\varphi(V)\{\bar Q(t, W) - \bar Q(r, W)\}]` is
    linear in the outcome regression for the fixed :math:`M = E[\varphi\varphi^\top]`, and
    its representer is :math:`e_j^\top M^{-1} \varphi(V) s(A) / g`, read off the fit's own
    clever covariate.
    """

    @pytest.mark.parametrize("name", sorted(PROJECTION_FEATURES))
    def test_the_riesz_identity(self, projections: dict[str, Any], name: str) -> None:
        result = projections[name]
        truths = projection_representers(name)
        assert set(truths) == set(result.estimates)
        for estimand, alpha0 in truths.items():
            gap, error = identity_gap(result, estimand, alpha0)
            assert error > 1e-3, (name, estimand)  # the nonzero witness: wrong mechanisms
            assert gap < RTOL, (name, estimand)
            assert ov.sensitivity_elements(result, estimand).features == PROJECTION_FEATURES[name]

    def test_the_oracle_representer_and_its_mutants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At the law's mechanisms the representer is the truth; each mutant departs."""
        result = projection_fit(_linear(), oracle=True)
        truths = projection_representers("response")
        for estimand, alpha0 in truths.items():
            alpha = ov.sensitivity_elements(result, estimand).riesz_representer
            np.testing.assert_allclose(alpha, alpha0, atol=1e-10, err_msg=estimand)
        build = ov._AxisInputs.build.__func__  # type: ignore[attr-defined]

        def mutant(change: Any) -> Any:
            def built(cls: Any, parameter: Any, nuisance: Any, weights: Any) -> Any:
                axis = build(cls, parameter, nuisance, weights)
                if axis is None or axis.basis is None:
                    return axis
                return replace(axis, gram_inverse=change(axis.gram_inverse))

            return classmethod(built)

        mutants = {
            # M^{-1} dropped: the coefficient reads e_j.
            "no_inverse": lambda inverse: np.eye(inverse.shape[0]),
            # M doubled: the MSM Gram path fed the arm numerators sums phi phi' twice.
            "doubled": lambda inverse: inverse / 2.0,
            # the other coefficient's row: a representer that reads the wrong phi_j.
            "other_term": lambda inverse: inverse[:, ::-1],
        }
        for label, change in mutants.items():
            with monkeypatch.context() as patch:
                patch.setattr(ov._AxisInputs, "build", mutant(change))
                for estimand, alpha0 in truths.items():
                    alpha = ov.sensitivity_elements(result, estimand).riesz_representer
                    departure = float(np.max(np.abs(alpha - alpha0)))
                    assert departure > 0.1, (label, estimand, departure)

    def test_the_gram_term_is_the_gateaux_derivative(self, projections: dict[str, Any]) -> None:
        result = projections["response"]
        for name in result.estimates:
            parameter = ov.resolve_parameter(result, name)
            elements = ov.sensitivity_elements(result, name)
            assert elements.psi_nu2 is not None
            expected = _gateaux(result, parameter) * result.data.weights
            np.testing.assert_allclose(elements.psi_nu2, expected, atol=1e-5, err_msg=name)

    def test_dropping_the_gram_term_departs(
        self, projections: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nonzero because each row's ``phi phi'`` differs from ``M``."""
        result = projections["response"]
        name = "cate_projection[W]"
        expected = _gateaux(result, ov.resolve_parameter(result, name)) * result.data.weights
        monkeypatch.setattr(ov, "_gram_influence", lambda *args: 0.0)
        psi = ov.sensitivity_elements(result, name).psi_nu2
        assert psi is not None
        assert float(np.max(np.abs(psi - expected))) > 0.1

    def test_the_residual_is_masked_to_the_coefficient_s_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = projection_fit(_indicators(intercept=False), covariates=("W", "V"))
        data, repeat = result.data, result.repeats[0]
        residual = np.where(
            np.asarray(data.observed, dtype=bool),
            data.outcome - repeat.fluctuations["effect_modification"].targeted.observed,
            0.0,
        )
        v = exact_frame()["V"].to_numpy()
        names = sorted(result.estimates)
        assert len(names) == 2
        for name, level in zip(names, (0.0, 1.0), strict=True):
            masked = np.where(v == level, residual, 0.0)
            expected = float(np.average(masked**2, weights=data.weights))
            assert ov.sensitivity_elements(result, name).sigma2 == pytest.approx(
                expected, rel=1e-12
            )
            unmasked = float(np.average(residual**2, weights=data.weights))
            assert abs(unmasked - expected) > 0.01, name

    def test_the_stratum_bridge(self) -> None:
        """``Indicators(V, intercept=False)`` reads the stratum ATE's representer and curve.

        Its ``nu^2`` and the curve of ``nu^2`` equal those of ``ATE()`` with ``strata=V``:
        the projection's Gram term is the stratum share's term. Under ``intercept=True``
        the intercept is the reference stratum's.
        """
        from cleverly import ATE
        from tests.unit.test_omitted_variable_part_g import stratum_fit, stratum_names

        stratum = stratum_fit(ATE())
        by_level = {
            stratum.parameter_keys[name].stratum[0]: name for name in stratum_names(stratum, "ate")
        }
        free = projection_fit(_indicators(intercept=False), covariates=("W", "V"))
        anchored = projection_fit(_indicators(intercept=True), covariates=("W", "V"))
        cases = [
            (free, sorted(free.estimates)[0], by_level[0.0]),
            (free, sorted(free.estimates)[1], by_level[1.0]),
            (anchored, "cate_projection[(intercept)]", by_level[0.0]),
        ]
        for result, coefficient, ate in cases:
            mine = ov.sensitivity_elements(result, coefficient)
            theirs = ov.sensitivity_elements(stratum, ate)
            assert mine.nu2 == pytest.approx(theirs.nu2, rel=1e-10), coefficient
            assert mine.psi_nu2 is not None and theirs.psi_nu2 is not None
            np.testing.assert_allclose(mine.psi_nu2, theirs.psi_nu2, atol=1e-10)

    def test_the_log_ratio_and_strata_refuse(self) -> None:
        from cleverly.exceptions import RefusalReason

        ratio = projection_fit(_linear("log_ratio"))
        refusal = ov.fit_wide_bound_refusal(ratio)
        assert refusal is not None
        assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP and refusal.roadmap == "X82"
        assert refusal.condition == (
            "the omitted-variable bound covers a projection coefficient only in its basic "
            "form, and a log-ratio coefficient is not linear in the outcome regression, so "
            "the linearized bound it would need is not derived."
        )
        with pytest.raises(CapabilityError, match="not linear in the outcome regression"):
            ov.omitted_variable_bounds(ratio, "log_crr_projection[W]", cf_y=0.02, cf_d=0.02)

    def test_the_evalue_refuses(self) -> None:
        from cleverly.assessment import (
            EVALUE_PROJECTION_CONSTANT_REFUSAL,
            EVALUE_PROJECTION_REFUSAL,
        )
        from cleverly.bases import Constant

        for scale in ("difference", "log_ratio"):
            row = projection_fit(_linear(scale)).sensitivity.capability("evalue")
            assert row.refusal == EVALUE_PROJECTION_REFUSAL, scale
        constant = projection_fit(_projection(Constant(), ("W",), "log_ratio"))
        assert list(constant.estimates) == ["log_crr_projection[(intercept)]"]
        row = constant.sensitivity.capability("evalue")
        assert row.refusal == EVALUE_PROJECTION_CONSTANT_REFUSAL
        with pytest.raises(CapabilityError, match="marginal log risk ratio"):
            constant.sensitivity.evalue()

    def test_target_weights_on_disjoint_indicators_reads_the_same_bound(self) -> None:
        """``Indicators(V, intercept=False)`` loads one column per row, so it admits
        ``target_weights=True``. The weighted form solves the same score equations along
        another submodel, so the estimates and ``sigma^2`` agree to second order. The bound
        reads the same representer, so ``nu^2`` and its curve are equal.
        """
        plain = projection_fit(_indicators(intercept=False), covariates=("W", "V"))
        weighted = projection_fit(
            _indicators(intercept=False), covariates=("W", "V"), target_weights=True
        )
        assert set(plain.estimates) == set(weighted.estimates)
        for name in plain.estimates:
            assert weighted.estimates[name].psi == pytest.approx(
                plain.estimates[name].psi, abs=1e-3
            )
            mine = ov.sensitivity_elements(weighted, name)
            theirs = ov.sensitivity_elements(plain, name)
            assert mine.nu2 == pytest.approx(theirs.nu2, rel=1e-12), name
            assert mine.psi_nu2 is not None and theirs.psi_nu2 is not None
            np.testing.assert_allclose(mine.psi_nu2, theirs.psi_nu2, atol=1e-10)
            assert mine.sigma2 == pytest.approx(theirs.sigma2, rel=1e-3), name

    def test_target_weights_refuses_on_an_overlapping_basis(self) -> None:
        """A ``Linear`` basis loads both columns on every row, which the weighted form refuses."""
        with pytest.raises(ValueError, match="at most one clever-covariate column"):
            projection_fit(_linear(), target_weights=True)

    def test_the_benchmark_keeps_the_basis(self) -> None:
        result = projection_fit(_indicators(intercept=False), covariates=("W", "V"))
        with pytest.raises(CapabilityError, match="basis of the modifiers"):
            ov.benchmark(result, ["V"], estimand=sorted(result.estimates)[0])

    def test_the_target_moved_branch(self, projections: dict[str, Any]) -> None:
        result = projections["response"]
        assert not ov._axis_target_moved(result, result)
        nuisance = result.repeats[0].nuisance
        spec = nuisance.effect_modification
        moved_spec = replace(spec, design=spec.design * 2.0)
        moved_repeat = replace(
            result.repeats[0], nuisance=replace(nuisance, effect_modification=moved_spec)
        )
        moved = replace(result, repeats=(moved_repeat, *result.repeats[1:]))
        assert ov._axis_target_moved(result, moved)
