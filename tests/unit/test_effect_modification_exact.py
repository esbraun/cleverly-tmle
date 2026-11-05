r"""Is the projection TMLE of a conditional contrast exact, and is its curve the EIF?

The projection of the conditional contrast onto a declared basis :math:`\varphi(V)` is

.. math::

    \beta_{\mathrm{diff}} = M^{-1} E[\varphi(V)\,\tau(W)], \qquad
    E\big[\varphi\{(\bar Q_t + \bar Q_r)\operatorname{expit}(\varphi^\top\beta) - \bar Q_t\}\big]
    = 0 \ \text{(log ratio)},

with :math:`M = E[\varphi\varphi^\top]` and :math:`\tau = \bar Q_t - \bar Q_r` (P1 plan,
2.1 and 2.3).  Each law of :mod:`tests.discrete_law` and its missing-outcome and three-arm
siblings is realised exactly by its sample, so with oracle nuisances the point estimate is
the functional and the influence curve is the Gateaux derivative of a longhand statement
of :math:`\beta`, taken by complex step.  The longhand statement shares no code with the
library: the log ratio is solved by a fixed number of Newton steps so that the complex step
differentiates through the solve.

``Linear(["W"])`` has two terms against three ``W`` cells, so the projection is not
saturated and the residual term :math:`\varphi(\tau - \varphi^\top\beta)` of the curve is
nonzero.  A saturated basis would make that term vanish and hide an estimator that
dropped it; ``TestTheWitnessesHaveTeeth`` shows each term the checks rely on is nonzero
here, and that a mutated clever covariate fails the double-robustness witness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.bases import Constant, Linear, Modifiers
from cleverly.effect_modification import EffectModification, EffectModificationSet
from cleverly.estimators.tmle import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.msm import MSM
from tests import discrete_law as law
from tests import discrete_law_mar as mar
from tests import discrete_law_multi as multi
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment

#: The basis ``(1, W)`` at ``W = 0, 1, 2``, as the oracle declares it.
PHI = law.EFFECT_MODIFICATION_DESIGN

SCALES = ("difference", "log_ratio")

STEM = {"difference": "cate_projection", "log_ratio": "log_crr_projection"}

#: A wrong outcome regression, indexed ``[w, a]``.  Not in the span of the basis's
#: contrast, so the plug-in projection at it is off and only the targeting can fix it.
Q_ERROR = np.array([[0.10, -0.05], [0.15, 0.10], [-0.20, 0.05]])

projection = law.effect_modification_beta


def binary_functional(scale: str) -> Callable[[Any], Any]:
    """``beta`` of the law of :mod:`tests.discrete_law`, from its ``(3, 2, 2)`` cells."""

    stem = STEM[scale]

    def functional(probs: Any) -> Any:
        return np.array(
            [law.functional(probs, f"{stem}[{term}]") for term in law.EFFECT_MODIFICATION_TERMS]
        )

    return functional


def mar_functional(scale: str) -> Callable[[Any], Any]:
    """``beta`` of the missing-at-random law, from its ``(3, 2, 3)`` cells."""

    def functional(probs: Any) -> Any:
        observed = probs[:, :, mar.OBSERVED_ZERO] + probs[:, :, mar.OBSERVED_ONE]
        q = probs[:, :, mar.OBSERVED_ONE] / observed
        return projection(probs.sum(axis=(1, 2)), q[:, 1], q[:, 0], scale)

    return functional


def multi_functional(scale: str) -> Callable[[Any], Any]:
    """``beta`` of the pair ``("high", "low")`` of the three-arm law."""
    high, low = multi.LABELS.index("high"), multi.LABELS.index("low")

    def functional(probs: Any) -> Any:
        q = probs[:, :, 1] / probs.sum(axis=2)
        return projection(probs.sum(axis=(1, 2)), q[:, high], q[:, low], scale)

    return functional


def declared(scale: str, basis: Any = None, **pair: Any) -> EffectModification:
    return EffectModification(
        Linear(["W"]) if basis is None else basis, Modifiers(["W"]), scale, **pair
    )


def fit_binary(scale: str, *, outcome_law: Any = None, basis: Any = None, **roles: Any) -> Any:
    dgp = law.DiscreteLaw() if outcome_law is None else outcome_law
    frame = roles.pop("frame", law.frame())
    treatment_law = roles.pop("treatment_law", law.DiscreteLaw())
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(treatment_law),
        cross_fit=False,
        effect_modification=declared(scale, basis),
        random_state=0,
        **roles.pop("settings", {}),
    )
    return estimator.fit(frame, outcome="Y", treatment="A", **roles).single()


def fit_mar(scale: str, *, outcome_law: Any = None) -> Any:
    dgp = mar.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp if outcome_law is None else outcome_law),
        treatment_learner=OracleTreatment(dgp),
        missingness_learner=OracleMissingness(dgp),
        cross_fit=False,
        effect_modification=declared(scale),
        random_state=0,
    )
    return estimator.fit(
        mar.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta"
    ).single()


def fit_multi(scale: str) -> Any:
    estimator = TMLE(
        outcome_learner=multi.OracleMultiOutcome(),
        treatment_learner=multi.OracleMultiTreatment(),
        cross_fit=False,
        effect_modification=declared(scale, treatment="high", reference="low"),
        random_state=0,
    )
    return estimator.fit(multi.frame(), outcome="Y", treatment="A").single()


def multi_first_rows() -> np.ndarray:
    counts = np.array([multi.COUNTS[cell] for cell in multi.SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])


def coefficients(fit: Any) -> np.ndarray:
    return np.array([estimate.psi for estimate in fit.estimates.values()])


def curves_at(fit: Any, rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.asarray(estimate.influence_curve)[rows] for estimate in fit.estimates.values()]
    )


def wrong_outcome_law(base: Any) -> Any:
    wrong = type(base)()
    wrong.q = np.clip(base.q + Q_ERROR, 0.05, 0.95)
    return wrong


#: Each law with its sample, fit and longhand functional.
LAWS: dict[str, tuple[Callable[[str], Any], Callable[[str], Any], Any, Any, Any]] = {
    "binary": (fit_binary, binary_functional, law.PROBS, law.SUPPORT, law.first_row_of),
    "missing outcomes": (fit_mar, mar_functional, mar.PROBS, mar.SUPPORT, mar.first_row_of),
    "three arms": (fit_multi, multi_functional, multi.PROBS, multi.SUPPORT, multi_first_rows),
}

#: Absolute window on the curve.  The log ratio runs two independent Newton solves.
CURVE_TOLERANCE = {"difference": 1e-12, "log_ratio": 1e-11}


@pytest.fixture(scope="module", params=[(name, scale) for name in LAWS for scale in SCALES])
def exact(request: Any) -> tuple[str, str, Any]:
    name, scale = request.param
    return name, scale, LAWS[name][0](scale)


class TestTheProjectionIsTheFunctional:
    def test_the_reported_terms_are_the_basis_terms(self, exact: Any) -> None:
        _, scale, fit = exact
        assert list(fit.estimates) == [f"{STEM[scale]}[(intercept)]", f"{STEM[scale]}[W]"]

    def test_the_point_estimate_is_the_projection(self, exact: Any) -> None:
        name, scale, fit = exact
        _, functional, probs, _, _ = LAWS[name]
        np.testing.assert_allclose(coefficients(fit), functional(scale)(probs), atol=1e-12)

    def test_the_curve_is_the_gateaux_derivative(self, exact: Any) -> None:
        name, scale, fit = exact
        _, functional, probs, support, first_rows = LAWS[name]
        expected = law.contamination_eif(functional(scale), probs, support)
        np.testing.assert_allclose(
            curves_at(fit, first_rows()), expected, atol=CURVE_TOLERANCE[scale], rtol=0
        )

    def test_the_projection_is_not_saturated(self, exact: Any) -> None:
        """Otherwise the residual term the curve carries would vanish."""
        name, scale, _ = exact
        _, functional, probs, _, _ = LAWS[name]
        beta = functional(scale)(probs)
        if name == "binary" and scale == "difference":
            q = law.Q_EXACT
            residual = q[:, 1] - q[:, 0] - PHI @ beta
            assert np.max(np.abs(residual)) > 0.02
        assert abs(beta[1]) > 1e-3


class TestWeightsDefineATiltedLaw:
    @pytest.mark.parametrize("scale", SCALES)
    def test_the_weighted_curve_is_the_derivative_of_the_tilted_projection(
        self, scale: str
    ) -> None:
        cells = law.cell_weights(law.WEIGHT_FUNCTIONS["treatment_and_outcome"])
        tilted = law.DiscreteLaw(law.tilt(law.PROBS, cells))
        fit = fit_binary(
            scale,
            outcome_law=tilted,
            treatment_law=tilted,
            frame=law.frame().assign(w=law.row_weights(cells)),
            weights="w",
        )

        def functional(probs: Any) -> Any:
            return binary_functional(scale)(law.tilt(probs, cells))

        np.testing.assert_allclose(coefficients(fit), functional(law.PROBS), atol=1e-12)
        np.testing.assert_allclose(
            curves_at(fit, law.first_row_of()),
            law.gateaux_eif(functional),
            atol=1e-12,
            rtol=0,
        )
        assert np.max(np.abs(coefficients(fit) - binary_functional(scale)(law.PROBS))) > 1e-3


class TestClustersAreTheUnitOfTheCurve:
    def test_the_standard_error_sums_the_curve_within_clusters(self) -> None:
        frame = law.frame()
        frame = frame.assign(cluster=np.arange(len(frame)) // 4)
        clustered = fit_binary("difference", frame=frame, id="cluster")
        iid = fit_binary("difference")
        for name, estimate in clustered.estimates.items():
            curve = np.asarray(iid.estimates[name].influence_curve)
            sums = np.bincount(frame["cluster"].to_numpy(), weights=curve)
            n, clusters = curve.size, sums.size
            # The cluster-robust variance with its G / (G - 1) small-sample factor.
            expected = np.sqrt(np.sum(sums**2) / n**2 * clusters / (clusters - 1))
            assert estimate.psi == pytest.approx(iid.estimates[name].psi, abs=1e-14)
            assert estimate.std_error == pytest.approx(expected, rel=1e-6)
            assert estimate.std_error != pytest.approx(iid.estimates[name].std_error, rel=1e-3)


class TestDoubleRobustness:
    """With a wrong outcome regression and the exact propensity the projection is exact."""

    @pytest.mark.parametrize("scale", SCALES)
    def test_targeting_repairs_a_wrong_outcome_regression(self, scale: str) -> None:
        fit = fit_binary(scale, outcome_law=wrong_outcome_law(law.DiscreteLaw()))
        truth = binary_functional(scale)(law.PROBS)
        np.testing.assert_allclose(coefficients(fit), truth, atol=1e-10)

    @pytest.mark.parametrize("scale", SCALES)
    def test_targeting_repairs_it_under_missing_outcomes(self, scale: str) -> None:
        fit = fit_mar(scale, outcome_law=wrong_outcome_law(mar.DiscreteLaw()))
        np.testing.assert_allclose(coefficients(fit), mar_functional(scale)(mar.PROBS), atol=1e-10)

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_plug_in_at_the_wrong_regression_is_off(self, scale: str) -> None:
        """The witness has something to repair: the T-learner projection is wrong."""
        wrong = wrong_outcome_law(law.DiscreteLaw()).q
        plug_in = projection(law.P_W, wrong[:, 1], wrong[:, 0], scale)
        assert np.max(np.abs(plug_in - binary_functional(scale)(law.PROBS))) > 0.02


class TestCrossChecks:
    def test_a_constant_basis_is_the_ate(self) -> None:
        projected = fit_binary("difference", basis=Constant())
        ate = fit_binary_estimands(("ate",))
        estimate = projected.estimates["cate_projection[(intercept)]"]
        assert estimate.psi == pytest.approx(ate["ate"].psi, abs=1e-12)
        np.testing.assert_allclose(
            estimate.influence_curve, ate["ate"].influence_curve, atol=1e-12, rtol=0
        )

    def test_a_constant_basis_on_the_log_ratio_is_the_log_risk_ratio(self) -> None:
        projected = fit_binary("log_ratio", basis=Constant())
        rr = fit_binary_estimands(("rr",))
        estimate = projected.estimates["log_crr_projection[(intercept)]"]
        assert estimate.psi == pytest.approx(np.log(rr["rr"].psi), abs=1e-12)

    def test_the_cate_projection_is_the_effect_block_of_the_identity_msm(self) -> None:
        """An independent route: the MSM ``[1, a, W, a W]`` with unit weights."""
        dgp = law.DiscreteLaw()
        msm_fit = (
            TMLE(
                outcome_learner=OracleOutcome(dgp),
                treatment_learner=OracleTreatment(dgp),
                cross_fit=False,
                msm=MSM.linear(modifiers=("W",)),
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A")
            .single()
        )
        projected = fit_binary("difference")
        effect_block = [name for name in msm_fit.estimates if "a" in name.split("[")[1]]
        assert len(effect_block) == 2
        np.testing.assert_allclose(
            [msm_fit.estimates[name].psi for name in effect_block],
            coefficients(projected),
            atol=1e-10,
        )


def fit_binary_estimands(estimands: tuple[str, ...]) -> Any:
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        estimands=estimands,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single().estimates


class ScaledLaw(law.DiscreteLaw):
    """The binary law's regression on an outcome ``scale * Y + shift``."""

    def __init__(self, scale: float, shift: float) -> None:
        super().__init__()
        self.q = scale * self.q + shift


def continuous_fit(scale_name: str, factor: float, shift: float, bounds: Any) -> Any:
    frame = law.frame()
    frame["Y"] = factor * frame["Y"] + shift
    return fit_binary(
        scale_name,
        outcome_law=ScaledLaw(factor, shift),
        frame=frame,
        settings={"q_bounds": bounds},
    )


class TestTheOutcomeScale:
    def test_the_difference_is_reported_on_the_outcome_scale(self) -> None:
        """``q_bounds=(-2, 5)``: the projection is 7 times the binary projection."""
        fit = continuous_fit("difference", 7.0, -2.0, (-2.0, 5.0))
        np.testing.assert_allclose(
            coefficients(fit), 7.0 * binary_functional("difference")(law.PROBS), atol=1e-11
        )

    def test_the_log_ratio_is_free_of_the_factor_when_the_scale_starts_at_zero(self) -> None:
        fit = continuous_fit("log_ratio", 5.0, 0.0, (0.0, 5.0))
        np.testing.assert_allclose(
            coefficients(fit), binary_functional("log_ratio")(law.PROBS), atol=1e-11
        )

    @pytest.mark.parametrize(
        ("bounds", "start"),
        [((1.0, 6.0), "These q_bounds start at 1."), (None, "the observed minimum, 1.")],
    )
    def test_a_log_ratio_on_a_shifted_scale_is_refused(self, bounds: Any, start: str) -> None:
        with pytest.raises(CapabilityError, match="q_bounds=\\(0, upper\\)") as caught:
            continuous_fit("log_ratio", 5.0, 1.0, bounds)
        assert start in str(caught.value)


class TestTheWitnessesHaveTeeth:
    """Each component the exact checks rest on is nonzero here, and a mutation fails them."""

    @pytest.mark.parametrize("scale", SCALES)
    def test_dropping_the_projection_residual_term_misses_the_derivative(self, scale: str) -> None:
        """The term :math:`\\varphi(\\tau - \\varphi^\\top\\beta)` of the curve is not zero."""
        fit = fit_binary(scale)
        q = law.Q_EXACT
        beta = binary_functional(scale)(law.PROBS)
        if scale == "difference":
            gram = PHI.T @ (law.P_W[:, None] * PHI)
            plugin = PHI * (q[:, 1] - q[:, 0] - PHI @ beta)[:, None]
            term = np.linalg.solve(gram, plugin.T).T
        else:
            e = 1.0 / (1.0 + np.exp(-(PHI @ beta)))
            weight = law.P_W * (q[:, 1] + q[:, 0]) * e * (1.0 - e)
            jacobian = PHI.T @ (weight[:, None] * PHI)
            plugin = PHI * ((q[:, 1] + q[:, 0]) * e - q[:, 1])[:, None]
            term = -np.linalg.solve(jacobian, plugin.T).T
        w_of_point = np.array([w for w, _, _ in law.SUPPORT])
        mutated = curves_at(fit, law.first_row_of()) - term[w_of_point]
        gap = np.max(np.abs(mutated - law.gateaux_eif(binary_functional(scale))))
        assert gap > 0.05

    @pytest.mark.parametrize("scale", SCALES)
    def test_dropping_the_residual_term_misses_the_derivative_under_missing_outcomes(
        self, scale: str
    ) -> None:
        r"""The :math:`H\,\Delta/\Pi\,(Y - \bar Q^*)` term carries the missingness."""
        fit = fit_mar(scale)
        curve = curves_at(fit, mar.first_row_of())
        unobserved = np.array([k == mar.UNOBSERVED for _, _, k in mar.SUPPORT])
        expected = law.contamination_eif(mar_functional(scale), mar.PROBS, mar.SUPPORT)
        # At an unobserved point the residual term is zero; elsewhere it is what moves the
        # curve off the plug-in part, which is the same at every k of a (w, a) cell.
        plugin_only = curve.copy()
        for index, (w, a, _) in enumerate(mar.SUPPORT):
            plugin_only[index] = curve[mar.SUPPORT.index((w, a, mar.UNOBSERVED))]
        assert np.max(np.abs(plugin_only[~unobserved] - expected[~unobserved])) > 0.05

    def test_a_contrast_without_the_reference_block_fails_double_robustness(
        self, monkeypatch: Any
    ) -> None:
        """Mutation: the covariate ``phi`` in place of ``(2A - 1) phi``."""
        original = EffectModificationSet.coefficients_at

        def one_signed(self: EffectModificationSet, beta: Any) -> Any:
            out = original(self, beta)
            return np.abs(out)

        monkeypatch.setattr(EffectModificationSet, "coefficients_at", one_signed)
        fit = fit_binary("difference", outcome_law=wrong_outcome_law(law.DiscreteLaw()))
        gap = np.max(np.abs(coefficients(fit) - binary_functional("difference")(law.PROBS)))
        assert gap > 1e-3

    def test_a_pair_covariate_that_loads_the_third_arm_fails_the_projection(
        self, monkeypatch: Any
    ) -> None:
        """Mutation: ``1{A = t} - 1{A != t}``, so the third arm's rows carry ``-phi``."""
        original = EffectModificationSet.coefficients_at

        def against_the_rest(self: EffectModificationSet, beta: Any) -> Any:
            out = original(self, beta)
            t = self.arms.index(self.pair[0])
            for index in range(len(self.arms)):
                if index != t:
                    out[:, index, :] = -self.design
            return out

        monkeypatch.setattr(EffectModificationSet, "coefficients_at", against_the_rest)
        estimator = TMLE(
            outcome_learner=multi.OracleMultiOutcome(),
            treatment_learner=multi.OracleMultiTreatment(),
            cross_fit=False,
            effect_modification=declared("difference", treatment="high", reference="low"),
            random_state=0,
        )
        frame = multi.frame()
        # A wrong regression gives the mutated covariate something to get wrong.
        frame_fit = estimator.fit(frame, outcome="Y", treatment="A").single()
        # At the oracle the fluctuation has nothing to do, so the curve is where it shows.
        expected = law.contamination_eif(multi_functional("difference"), multi.PROBS, multi.SUPPORT)
        gap = np.max(np.abs(curves_at(frame_fit, multi_first_rows()) - expected))
        assert gap > 0.05

    def test_a_ratio_covariate_without_expit_fails_double_robustness(
        self, monkeypatch: Any
    ) -> None:
        """Mutation: the covariate ``-1{A = t} phi`` in place of ``(expit(eta) - A) phi``."""

        def arm_only(self: EffectModificationSet, beta: Any) -> Any:
            out = np.zeros((self.n, len(self.arms), self.n_terms))
            out[:, self.arms.index(self.pair[0]), :] = -self.design
            return out

        monkeypatch.setattr(EffectModificationSet, "coefficients_at", arm_only)
        fit = fit_binary("log_ratio", outcome_law=wrong_outcome_law(law.DiscreteLaw()))
        gap = np.max(np.abs(coefficients(fit) - binary_functional("log_ratio")(law.PROBS)))
        assert gap > 1e-3
