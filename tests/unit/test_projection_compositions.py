r"""The projection of a conditional contrast composed with a missing treatment and strata.

P1 plan, PR 1, sections 2.1, 2.2 and 4.1 to 4.2.  Each check runs on an exact law whose
sample is the law, so at oracle nuisances the point estimate is the functional and the
influence curve is the Gateaux derivative, taken by complex step on a longhand statement of
the functional that shares no code with the library.

* **A missing treatment** (MT1 to MT7), on :data:`tests.discrete_law_composite.TWO`: the
  projection reads :math:`\bar Q(a, W) = E[Y \mid A = a, \Delta_A = 1, \Delta = 1, W]` and
  the law of :math:`W` on every row, and fluctuates along
  :math:`c_a(V) C_a / g_{c,a}(W)`.  The treatment-observation factor depends on ``W``, so
  the checks that it enters can fail.
* **Baseline strata** (ST1 to ST6), on :class:`StrataLaw`: the stratum coefficients are the
  projection of the law given :math:`S = s`.  On the difference scale the stratified
  fluctuation also solves the marginal score.  On the log ratio the marginal names come from
  their own unstratified fluctuation (``effect_modification_ratio``) and the stratum names
  from the stratified one (``effect_modification_ratio_strata``).

Every mutation is a monkeypatch or an in-test recomputation; no source file is edited.
"""

from __future__ import annotations

import importlib
import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import cleverly.estimators.targeting as targeting
from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Indicators, Linear, Modifiers, Tensor
from cleverly.effect_modification import EffectModification
from cleverly.estimators import TMLE
from cleverly.estimators.targeting import STRATIFIED_RATIO_GROUP
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.inference.results import estimate_covariance
from tests import discrete_law as law
from tests import discrete_law_composite as dl
from tests.conftest import OracleOutcome, OracleTreatment
from tests.studies.mar_arm_indexed_laws import LawOutcome, LawResponse, LawTreatment
from tests.unit._natural_course_support import NeverFit

#: The module, not the ``cleverly.estimators.tmle`` attribute, which a function shadows.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

SCALES = ("difference", "log_ratio")
STEM = {"difference": "cate_projection", "log_ratio": "log_crr_projection"}
TERMS = law.EFFECT_MODIFICATION_TERMS
PHI = law.EFFECT_MODIFICATION_DESIGN
beta_of = law.effect_modification_beta

#: Absolute window on a curve; the log ratio runs two independent Newton solves.
CURVE_TOLERANCE = {"difference": 1e-12, "log_ratio": 1e-11}


def names(scale: str, label: str | None = None) -> list[str]:
    stem = STEM[scale]
    suffix = "" if label is None else f"[{label}]"
    return [f"{stem}[{term}]{suffix}" for term in TERMS]


def points(fit: Any, keys: list[str]) -> np.ndarray:
    return np.array([fit.estimates[key].psi for key in keys])


def curves(fit: Any, keys: list[str], rows: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(fit.estimates[key].influence_curve)[rows] for key in keys])


def declared(scale: str, basis: Any = None, modifiers: Any = None) -> EffectModification:
    return EffectModification(
        Linear(["W"]) if basis is None else basis,
        Modifiers(["W"]) if modifiers is None else modifiers,
        scale,  # type: ignore[arg-type]
    )


# =============================================================== a missing treatment

COMPOSITE = dl.TWO
ARM_T, ARM_R = COMPOSITE.labels.index(1.0), COMPOSITE.labels.index(0.0)

#: The outcome drift of the composite law (its regression reflected), the wrong regression
#: every MT witness reads.
WRONG_MU = 1.0 - COMPOSITE.q


def composite_functional(scale: str, k: int = COMPOSITE.k) -> Any:
    """``beta`` from the ``(3, K + 1, kinds)`` cell probabilities of a composite law."""

    def functional(probs: Any) -> Any:
        p_w = probs.sum(axis=(1, 2))
        q = probs[:, :k, dl.OBSERVED_ONE] / (
            probs[:, :k, dl.OBSERVED_ZERO] + probs[:, :k, dl.OBSERVED_ONE]
        )
        return beta_of(p_w, q[:, ARM_T], q[:, ARM_R], scale)

    return functional


def first_rows(cells: np.ndarray, n_cells: int) -> np.ndarray:
    return np.array([int(np.flatnonzero(cells == cell)[0]) for cell in range(n_cells)])


def fit_composite(
    scale: str,
    *,
    mu: np.ndarray | None = None,
    recorded: np.ndarray | None = None,
    g: np.ndarray | None = None,
    frame: pd.DataFrame | None = None,
    delta: bool = True,
    oracle: Any = None,
    **settings: Any,
) -> Any:
    oracle = COMPOSITE.law() if oracle is None else oracle
    estimator = TMLE(
        outcome_learner=LawOutcome(oracle, mu=mu),
        treatment_learner=LawTreatment(oracle, g=g),
        missingness_learner=LawResponse(oracle, recorded=recorded),
        cross_fit=False,
        effect_modification=declared(scale),
        random_state=0,
        tol=1e-12,
        **settings,
    )
    return estimator.fit(
        COMPOSITE.frame() if frame is None else frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        delta="Delta" if delta else None,
        treatment_delta="DeltaA",
    ).single()


def recorded_only_law() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """The composite law's rows with an observed outcome: a missing treatment and no ``delta=``.

    It is the empirical law of its own rows, so its tables are the composite law's
    conditioned on :math:`\\Delta = 1`.  Returns the frame, its ``(3, K + 1, 2)`` cell
    probabilities, the oracle tables and each row's cell.
    """
    counts = COMPOSITE.counts[:, :, :2]
    probs = counts / counts.sum()
    frame = COMPOSITE.frame()
    keep = frame["Delta"].to_numpy() == 1.0
    cells = COMPOSITE.cell_of_row()[keep]
    support = list(itertools.product(range(3), range(COMPOSITE.k + 1), range(2)))
    index = {cell: j for j, cell in enumerate(support)}
    reindexed = np.array([index[COMPOSITE.support[c]] for c in cells])
    joint = probs[:, : COMPOSITE.k, :].sum(axis=2)
    recorded = joint.sum(axis=1) / probs.sum(axis=(1, 2))
    tables = {"g": joint / joint.sum(axis=1, keepdims=True), "recorded": recorded}
    return frame[keep].reset_index(drop=True), probs, tables, reindexed


class TestMT1TheCompositeProjectionIsExact:
    @pytest.mark.parametrize("scale", SCALES)
    def test_with_a_missing_outcome(self, scale: str) -> None:
        fit = fit_composite(scale)
        assert fit.extra["missing_data"] == "composite"
        functional = composite_functional(scale)
        np.testing.assert_allclose(
            points(fit, names(scale)), functional(COMPOSITE.probs), atol=1e-12
        )
        rows = first_rows(COMPOSITE.cell_of_row(), len(COMPOSITE.support))
        expected = law.contamination_eif(functional, COMPOSITE.probs, COMPOSITE.support)
        np.testing.assert_allclose(
            curves(fit, names(scale), rows), expected, atol=CURVE_TOLERANCE[scale], rtol=0
        )

    @pytest.mark.parametrize("scale", SCALES)
    def test_without_delta(self, scale: str) -> None:
        frame, probs, tables, cells = recorded_only_law()
        fit = fit_composite(
            scale,
            frame=frame.drop(columns=["Delta"]),
            delta=False,
            g=tables["g"],
            recorded=tables["recorded"],
        )
        functional = composite_functional(scale)
        np.testing.assert_allclose(points(fit, names(scale)), functional(probs), atol=1e-12)
        support = list(itertools.product(range(3), range(COMPOSITE.k + 1), range(2)))
        expected = law.contamination_eif(functional, probs, support)
        np.testing.assert_allclose(
            curves(fit, names(scale), first_rows(cells, len(support))),
            expected,
            atol=CURVE_TOLERANCE[scale],
            rtol=0,
        )


def limit_without_recording(mu: np.ndarray) -> np.ndarray:
    r"""The difference-scale projection when :math:`\pi_A` is replaced by 1 in :math:`g_{c,a}`.

    The fluctuation fits :math:`\operatorname{logit}\bar Q^* = \operatorname{logit}\mu +
    \varepsilon^\top c_a(w) / \tilde g_{c,a}(w)` on the rows with :math:`C_a = 1`, of mass
    :math:`P(w) g_{c,a}(w)`, along the working denominator
    :math:`\tilde g_{c,a} = g(a \mid \Delta_A = 1, w)\,\pi(a, w)`.  On the exact frame the
    empirical score is this population score, so the fit lands on this value.
    """
    from scipy.optimize import root

    working = COMPOSITE.conditional_g * COMPOSITE.pi
    mass = dl.P_W[:, None] * COMPOSITE.composite
    sign = np.zeros(COMPOSITE.k)
    sign[ARM_T], sign[ARM_R] = 1.0, -1.0
    offset = np.log(mu / (1.0 - mu))

    def targeted(eps: np.ndarray) -> np.ndarray:
        clever = sign[None, :, None] * PHI[:, None, :] / working[:, :, None]
        return np.asarray(expit(offset + clever @ eps), dtype=float)

    def score(eps: np.ndarray) -> np.ndarray:
        clever = sign[None, :, None] * PHI[:, None, :] / working[:, :, None]
        residual = mass * (COMPOSITE.q - targeted(eps))
        return np.asarray(np.einsum("wa,wap->p", residual, clever), dtype=float)

    solved = root(score, np.zeros(2), tol=1e-14)
    assert solved.success
    q = targeted(solved.x)
    return np.asarray(beta_of(dl.P_W, q[:, ARM_T], q[:, ARM_R], "difference"), dtype=float)


#: MT2's limit at the wrong regression, computed at import, and its distance from the truth.
NO_RECORDING_LIMIT = limit_without_recording(WRONG_MU)
COMPOSITE_TRUTH = {scale: composite_functional(scale)(COMPOSITE.probs) for scale in SCALES}
assert np.max(np.abs(NO_RECORDING_LIMIT - COMPOSITE_TRUTH["difference"])) >= 1e-2
#: MT3's limit: the plug-in and the Gram averaged over the recorded rows only.
_RECORDED_W = COMPOSITE.probs[:, : COMPOSITE.k, :].sum(axis=(1, 2))
RECORDED_ONLY_LIMIT = beta_of(
    _RECORDED_W / _RECORDED_W.sum(), COMPOSITE.q[:, ARM_T], COMPOSITE.q[:, ARM_R], "difference"
)
assert np.max(np.abs(RECORDED_ONLY_LIMIT - COMPOSITE_TRUTH["difference"])) >= 1e-2


class TestMT2ToMT6TheWitnessesHaveTeeth:
    @pytest.mark.parametrize("scale", SCALES)
    def test_mt2_a_wrong_regression_is_repaired(self, scale: str) -> None:
        fit = fit_composite(scale, mu=WRONG_MU)
        np.testing.assert_allclose(points(fit, names(scale)), COMPOSITE_TRUTH[scale], atol=1e-10)

    def test_mt2_the_recording_factor_replaced_by_one_lands_on_its_limit(self) -> None:
        """The treatment-observation factor depends on W, and the fit reads it."""
        ones = np.ones(3)
        fit = fit_composite("difference", mu=WRONG_MU, recorded=ones)
        np.testing.assert_allclose(points(fit, names("difference")), NO_RECORDING_LIMIT, atol=1e-9)

    def test_mt3_the_law_of_w_on_every_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: the projection averaged over the recorded rows misses by its limit."""
        from cleverly.effect_modification import EffectModificationSet

        original = EffectModificationSet.project
        recorded = COMPOSITE.frame()["DeltaA"].to_numpy() == 1.0

        def recorded_rows(self: Any, predictions: Any, weights: Any) -> Any:
            w = np.asarray(weights, dtype=float) * (
                recorded if len(weights) == recorded.size else 1
            )
            return original(self, predictions, w)

        monkeypatch.setattr(EffectModificationSet, "project", recorded_rows)
        fit = fit_composite("difference")
        np.testing.assert_allclose(points(fit, names("difference")), RECORDED_ONLY_LIMIT, atol=1e-9)

    def test_mt4_a_view_masked_by_delta_alone_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: an unrecorded row's coded treatment reaches the arm indicators."""
        import cleverly.estimators.composite as composite

        original = composite.composite_view

        def delta_only(data: Any) -> Any:
            from dataclasses import replace

            observed = np.asarray(data.observed, dtype=bool)
            raw = np.asarray(data.treatment, dtype=float)
            # The stale code an unrecorded row would carry, here the reference arm's.
            codes = np.where(observed, np.where(np.isfinite(raw), raw, 0.0), np.nan)
            return replace(
                original(data),
                treatment=codes,
                observed=observed & np.isfinite(codes),
                outcome=np.where(observed, np.asarray(data.outcome, dtype=float), 0.0),
            )

        monkeypatch.setattr(composite, "composite_view", delta_only)
        fit = fit_composite("difference", frame=COMPOSITE.frame(coded_unrecorded=True))
        rows = first_rows(COMPOSITE.cell_of_row(), len(COMPOSITE.support))
        expected = law.contamination_eif(
            composite_functional("difference"), COMPOSITE.probs, COMPOSITE.support
        )
        gap = np.max(np.abs(curves(fit, names("difference"), rows) - expected))
        assert gap > 1e-3

    @pytest.mark.parametrize("scale", SCALES)
    def test_mt5_the_forced_composite_is_the_missing_outcome_projection(
        self, scale: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no treatment-observation factor the composite is the shipped ``delta=`` route.

        The missing-at-random law's products ``g pi`` are at least 1/16, so no bound binds.
        """
        from tests import discrete_law_mar as mar
        from tests.conftest import OracleMissingness

        def fit() -> Any:
            dgp = mar.DiscreteLaw()
            return (
                TMLE(
                    outcome_learner=OracleOutcome(dgp),
                    treatment_learner=OracleTreatment(dgp),
                    missingness_learner=OracleMissingness(dgp),
                    cross_fit=False,
                    effect_modification=declared(scale),
                    g_bounds=(0.01, 0.99),
                    nuisance_bound=0.01,
                    random_state=0,
                    tol=1e-12,
                )
                .fit(mar.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")
                .single()
            )

        shipped = fit()
        with monkeypatch.context() as patch:
            patch.setattr(tmle_module, "missing_data_route", lambda estimator, data: "composite")
            forced = fit()
        assert forced.extra["missing_data"] == "composite"
        assert shipped.extra["missing_data"] != "composite"
        keys = names(scale)
        np.testing.assert_allclose(points(forced, keys), points(shipped, keys), atol=1e-12)
        rows = np.arange(shipped.data.n)
        np.testing.assert_allclose(
            curves(forced, keys, rows), curves(shipped, keys, rows), atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize("mutation", ["reference_block", "expit"])
    def test_mt6_log_ratio_numerators(self, mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.effect_modification import EffectModificationSet

        original = EffectModificationSet.coefficients_at

        def mutated(self: EffectModificationSet, beta: Any) -> Any:
            out = original(self, beta)
            if mutation == "reference_block":
                out[:, self.arms.index(self.pair[1]), :] = 0.0
            else:
                out = np.zeros_like(out)
                out[:, self.arms.index(self.pair[0]), :] = -self.design
            return out

        monkeypatch.setattr(EffectModificationSet, "coefficients_at", mutated)
        fit = fit_composite("log_ratio", mu=WRONG_MU)
        gap = np.max(np.abs(points(fit, names("log_ratio")) - COMPOSITE_TRUTH["log_ratio"]))
        assert gap > 1e-3


def composite_study(**design: Any) -> CausalStudy:
    frame = COMPOSITE.frame()
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            treatment_missingness="DeltaA",
            **design,
        ),
    )


def never() -> dict[str, Any]:
    NeverFit.calls = 0
    return {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


CONTRAST = ConditionalContrast(Modifiers(["W"]), model=Linear(["W"]))


class TestMT7RefusalsBeforeAnyLearner:
    def test_cross_fitted(self) -> None:
        effect = composite_study().identify(CONTRAST)
        with pytest.raises(CapabilityError, match="open \\(roadmap X53\\)") as caught:
            effect.estimate(cross_fit=True, **never())
        assert NeverFit.calls == 0
        assert caught.value.refusal.roadmap == "X53"
        assert "projection coefficients of a ConditionalContrast" in str(caught.value)
        assert caught.value.refusal.alternative.startswith("Pass cross_fit=False")

    def test_method_ep(self) -> None:
        effect = composite_study().identify(ConditionalContrast(Modifiers(["W"])))
        record = next(r for r in effect.available_methods() if r.name == "ep")
        assert not record.available
        assert record.refusal.roadmap == "X53"
        assert record.refusal.condition.startswith("The EP learner cross-fits its nuisances")
        with pytest.raises(CapabilityError, match="The EP learner cross-fits"):
            effect.estimate(method="ep", **never())
        assert NeverFit.calls == 0

    def test_assess_score(self) -> None:
        frame = COMPOSITE.frame().assign(score=lambda f: 0.1 * f["W"])
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W",),
                missingness="Delta",
                treatment_missingness="DeltaA",
            ),
        )
        effect = study.identify(ConditionalContrast(Modifiers(["W"])))
        with pytest.raises(CapabilityError, match="its composite form is not built") as caught:
            effect.assess_score(
                "score", computed_from=["W"], score_kind="external", cross_fit=False, **never()
            )
        assert caught.value.refusal.roadmap == "X98"
        assert NeverFit.calls == 0


# ======================================================================= strata

#: ``P(S = s)``, ``P(W = w | S = s)`` (it differs by stratum, so ``M_s`` differs from ``M``),
#: ``g(1 | s, w)`` and ``P(Y = 1 | A = a, S = s, W = w)`` indexed ``[s, w, a]``.
P_S = np.array([5 / 8, 3 / 8])
P_W_S = np.array([[1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 2]])
G_SW = np.array([[3 / 8, 5 / 8, 4 / 8], [4 / 8, 2 / 8, 6 / 8]])
Q_SWA = np.array(
    [
        [[1 / 4, 2 / 4], [2 / 4, 3 / 4], [1 / 4, 3 / 4]],
        [[1 / 4, 3 / 4], [3 / 4, 3 / 4], [2 / 4, 1 / 4]],
    ]
)
#: A wrong outcome regression ``[s, w, a]``, so only the targeting can reach the truth.
Q_SWA_ERROR = np.array(
    [
        [[0.10, -0.05], [0.15, 0.10], [-0.20, 0.05]],
        [[-0.10, 0.15], [0.05, -0.10], [0.10, 0.20]],
    ]
)
N_STRATA_LAW = 4096
STRATA_SUPPORT = tuple(itertools.product(range(2), range(3), range(2), range(2)))


def strata_probs() -> np.ndarray:
    a1 = G_SW[:, :, None] * np.array([0.0, 1.0]) + (1.0 - G_SW[:, :, None]) * np.array([1.0, 0.0])
    joint = P_S[:, None, None] * P_W_S[:, :, None] * a1
    y1 = Q_SWA
    return np.stack([joint * (1.0 - y1), joint * y1], axis=3)


STRATA_PROBS = strata_probs()
_COUNTS = STRATA_PROBS * N_STRATA_LAW
assert np.allclose(_COUNTS, np.rint(_COUNTS)) and _COUNTS.min() >= 2


class StrataLaw:
    """The oracle of the two-stratum law, read from covariates ``[W, S]``."""

    def __init__(self, q: np.ndarray = Q_SWA) -> None:
        self.q = q

    def outcome_mean(self, w: Any, a: float, _: Any) -> Any:
        cov = np.asarray(w, dtype=float)
        return self.q[cov[:, 1].astype(int), cov[:, 0].astype(int), int(a)]

    def propensity(self, x: Any) -> Any:
        cov = np.asarray(x, dtype=float)
        return G_SW[cov[:, 1].astype(int), cov[:, 0].astype(int)]


WRONG_STRATA_LAW = StrataLaw(np.clip(Q_SWA + Q_SWA_ERROR, 0.05, 0.95))


def strata_frame() -> tuple[pd.DataFrame, np.ndarray]:
    counts = np.rint(_COUNTS).astype(int)
    cells = np.repeat(np.arange(len(STRATA_SUPPORT)), [counts[cell] for cell in STRATA_SUPPORT])
    values = np.array(STRATA_SUPPORT, dtype=float)[cells]
    frame = pd.DataFrame(
        {"S": values[:, 0], "W": values[:, 1], "A": values[:, 2], "Y": values[:, 3]}
    )
    return frame, cells


def stratum_functional(scale: str, s: int | None) -> Any:
    """``beta`` of stratum ``s`` (``None``: the marginal), from ``[s, w, a, y]`` probabilities."""

    def functional(probs: Any) -> Any:
        cells = probs.sum(axis=3)
        q = probs[:, :, :, 1] / cells
        if s is None:
            p_sw = probs.sum(axis=(2, 3))
            p_w = p_sw.sum(axis=0)
            qt = (p_sw * q[:, :, 1]).sum(axis=0) / p_w
            qr = (p_sw * q[:, :, 0]).sum(axis=0) / p_w
            return beta_of(p_w, qt, qr, scale)
        p_w = probs[s].sum(axis=(1, 2))
        return beta_of(p_w / p_w.sum(), q[s, :, 1], q[s, :, 0], scale)

    return functional


STRATUM_LABELS = ("S=0.0", "S=1.0")
STRATA_TRUTH = {
    (scale, s): stratum_functional(scale, s)(STRATA_PROBS) for scale in SCALES for s in (None, 0, 1)
}
# ST3's nonzero witness: each stratum's truth differs from the marginal truth.
for _scale in SCALES:
    for _s in (0, 1):
        assert np.max(np.abs(STRATA_TRUTH[(_scale, _s)] - STRATA_TRUTH[(_scale, None)])) >= 0.05


def fit_strata(scale: str, outcome: StrataLaw | None = None, **settings: Any) -> Any:
    frame, _ = strata_frame()
    estimator = TMLE(
        outcome_learner=OracleOutcome(StrataLaw() if outcome is None else outcome),
        treatment_learner=OracleTreatment(StrataLaw()),
        cross_fit=False,
        effect_modification=settings.pop("effect_modification", declared(scale)),
        random_state=0,
        tol=1e-12,
        **settings,
    )
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W", "S"],
        strata=None if settings.get("_no_strata") else ["S"],
    ).single()


def fit_unstratified(
    scale: str, outcome: StrataLaw, basis: Any = None, modifiers: Any = None
) -> Any:
    frame, _ = strata_frame()
    return (
        TMLE(
            outcome_learner=OracleOutcome(outcome),
            treatment_learner=OracleTreatment(StrataLaw()),
            cross_fit=False,
            effect_modification=declared(scale, basis, modifiers),
            random_state=0,
            tol=1e-12,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W", "S"])
        .single()
    )


@pytest.fixture(scope="module", params=SCALES)
def wrong_strata_fit(request: Any) -> tuple[str, Any]:
    return request.param, fit_strata(request.param, WRONG_STRATA_LAW)


class TestST1ToST3TheStratumCoefficients:
    def test_st1_st3_each_stratum_is_exact_at_a_wrong_regression(
        self, wrong_strata_fit: Any
    ) -> None:
        scale, fit = wrong_strata_fit
        for s, label in enumerate(STRATUM_LABELS):
            np.testing.assert_allclose(
                points(fit, names(scale, label)), STRATA_TRUTH[(scale, s)], atol=1e-10
            )

    def test_st1_the_stratum_blocks_are_the_interacted_projection(
        self, wrong_strata_fit: Any
    ) -> None:
        """``Tensor(Indicators(S), Linear(W))`` spans ``{1{S = s} phi(W)}_s``."""
        scale, fit = wrong_strata_fit
        interacted = fit_unstratified(
            scale,
            WRONG_STRATA_LAW,
            basis=Tensor(Indicators("S"), Linear(["W"])),
            modifiers=Modifiers(["W", "S"], levels={"S": (0.0, 1.0)}),
        )
        terms = interacted.nuisance.effect_modification.terms
        coefficient = {
            term: estimate.psi
            for term, estimate in zip(terms, interacted.estimates.values(), strict=True)
        }
        assert len(terms) == 4
        # The intercept, the S = 1 indicator, W and their product.
        b0, b_w = coefficient["(intercept)"], coefficient["W"]
        (s_term,) = [t for t in terms if "S" in t and "W" not in t]
        (sw_term,) = [t for t in terms if "S" in t and "W" in t]
        b_s, b_sw = coefficient[s_term], coefficient[sw_term]
        np.testing.assert_allclose(points(fit, names(scale, "S=0.0")), [b0, b_w], atol=1e-10)
        np.testing.assert_allclose(
            points(fit, names(scale, "S=1.0")), [b0 + b_s, b_w + b_sw], atol=1e-10
        )

    def test_st1_mutation_blocks_read_at_one_common_beta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = targeting._stratified_ratio_submodel

        def common(data: Any, nuisance: Any, *, bounds: Any, nuisance_bound: Any, beta: Any) -> Any:
            stacked = np.asarray(beta, dtype=float).reshape(data.n_strata, -1)
            shared = np.tile(P_S @ stacked, data.n_strata)
            return original(
                data, nuisance, bounds=bounds, nuisance_bound=nuisance_bound, beta=shared
            )

        monkeypatch.setattr(targeting, "_stratified_ratio_submodel", common)
        fit = fit_strata("log_ratio", WRONG_STRATA_LAW)
        gap = max(
            np.max(np.abs(points(fit, names("log_ratio", label)) - STRATA_TRUTH[("log_ratio", s)]))
            for s, label in enumerate(STRATUM_LABELS)
        )
        assert gap >= 1e-3


class TestST2TheMarginalCoefficients:
    def test_st2_difference_marginal_is_exact_at_a_wrong_regression(self) -> None:
        fit = fit_strata("difference", WRONG_STRATA_LAW)
        np.testing.assert_allclose(
            points(fit, names("difference")), STRATA_TRUTH[("difference", None)], atol=1e-10
        )
        assert list(fit.fluctuations) == ["effect_modification"]

    @pytest.mark.parametrize("scale", SCALES)
    def test_st2_the_curves_are_the_gateaux_derivatives(self, scale: str) -> None:
        fit = fit_strata(scale)
        _, cells = strata_frame()
        rows = first_rows(cells, len(STRATA_SUPPORT))
        for s, label in ((None, None), (0, "S=0.0"), (1, "S=1.0")):
            expected = law.contamination_eif(
                stratum_functional(scale, s), STRATA_PROBS, STRATA_SUPPORT
            )
            np.testing.assert_allclose(
                curves(fit, names(scale, label), rows),
                expected,
                atol=CURVE_TOLERANCE[scale],
                rtol=0,
            )

    def test_st2b_the_log_ratio_marginal_names_are_the_unstratified_fit(self) -> None:
        fit = fit_strata("log_ratio", WRONG_STRATA_LAW)
        plain = fit_unstratified("log_ratio", WRONG_STRATA_LAW)
        marginal = names("log_ratio")
        np.testing.assert_allclose(points(fit, marginal), points(plain, marginal), atol=1e-12)
        rows = np.arange(N_STRATA_LAW)
        np.testing.assert_allclose(
            curves(fit, marginal, rows), curves(plain, marginal, rows), atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            points(fit, marginal), STRATA_TRUTH[("log_ratio", None)], atol=1e-10
        )
        # Each name comes from exactly one fluctuation.
        assert list(fit.fluctuations) == ["effect_modification_ratio", STRATIFIED_RATIO_GROUP]
        assert len(fit.estimates) == 6
        # The joint curve spans the two groups.
        pair = [marginal[1], names("log_ratio", "S=1.0")[1]]
        np.testing.assert_allclose(
            fit.covariance(pair), estimate_covariance(fit.estimates, pair), atol=1e-15
        )
        assert abs(fit.covariance(pair)[0, 1]) > 0
        assert fit.heterogeneity_test().statistic == pytest.approx(
            plain.heterogeneity_test().statistic, rel=1e-10
        )
        grid = {"W": [0.0, 1.0, 2.0]}
        np.testing.assert_allclose(
            fit.effect_curve().predict(grid), plain.effect_curve().predict(grid), atol=1e-12
        )

    def test_st2c_the_marginal_read_after_the_stratified_fluctuation_misses(self) -> None:
        """Rev 1's construction: the marginal plug-in of the stratified targeted fit."""
        fit = fit_strata("log_ratio", WRONG_STRATA_LAW)
        targeted = fit.fluctuations[STRATIFIED_RATIO_GROUP].targeted
        contrast = fit.nuisance.effect_modification
        predictions = targeting._raw_arms(targeted, contrast.arms, fit.nuisance.scaler)
        beta = contrast.project(predictions, fit.data.weights).beta
        assert np.max(np.abs(beta - STRATA_TRUTH[("log_ratio", None)])) >= 1e-2

    def test_a_second_group_over_a_claimed_name_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the unstratified group also emits the stratum names."""
        monkeypatch.setattr(
            tmle_module.TMLE,
            "_groups",
            staticmethod(
                lambda estimands, data=None: [
                    "effect_modification_ratio",
                    STRATIFIED_RATIO_GROUP,
                    STRATIFIED_RATIO_GROUP,
                ]
            ),
        )
        with pytest.raises(RuntimeError, match="each name must come from exactly one fluctuation"):
            fit_strata("log_ratio")


class TestST4TheStratumGramAndScale:
    def test_the_stratum_gram_differs_from_the_marginal(self) -> None:
        m = PHI.T @ (P_S @ P_W_S)[:, None] * 0 + PHI.T @ ((P_S @ P_W_S)[:, None] * PHI)
        for s in (0, 1):
            m_s = PHI.T @ (P_W_S[s][:, None] * PHI)
            assert np.max(np.abs(m_s - m)) >= 0.1
            q = Q_SWA[s]
            moment = PHI.T @ (P_W_S[s] * (q[:, 1] - q[:, 0]))
            mutated = np.linalg.solve(m, moment)
            assert np.max(np.abs(mutated - STRATA_TRUTH[("difference", s)])) >= 1e-2

    def test_the_curve_without_the_scale_fails_its_derivative(self) -> None:
        fit = fit_strata("difference")
        _, cells = strata_frame()
        rows = first_rows(cells, len(STRATA_SUPPORT))
        expected = law.contamination_eif(
            stratum_functional("difference", 1), STRATA_PROBS, STRATA_SUPPORT
        )
        in_stratum = np.flatnonzero(fit.data.strata == 1).size
        unscaled = curves(fit, names("difference", "S=1.0"), rows) * in_stratum / N_STRATA_LAW
        assert np.max(np.abs(unscaled - expected)) > 0.05


class TestST5TheStratumTest:
    def test_the_stratum_heterogeneity_test_reads_its_cluster_sums(self) -> None:
        frame, _ = strata_frame()
        frame = frame.assign(cluster=np.arange(len(frame)) // 4)
        fit = (
            TMLE(
                outcome_learner=OracleOutcome(StrataLaw()),
                treatment_learner=OracleTreatment(StrataLaw()),
                cross_fit=False,
                effect_modification=declared("difference"),
                random_state=0,
            )
            .fit(
                frame, outcome="Y", treatment="A", covariates=["W", "S"], strata=["S"], id="cluster"
            )
            .single()
        )
        name = names("difference", "S=1.0")[1]
        curve = np.asarray(fit.estimates[name].influence_curve)
        sums = np.bincount(frame["cluster"].to_numpy(), weights=curve)
        n, clusters = curve.size, sums.size
        variance = np.sum(sums**2) / n**2 * clusters / (clusters - 1)
        test = fit.heterogeneity_test(stratum="S=1.0")
        assert test.statistic == pytest.approx(fit.estimates[name].psi ** 2 / variance, rel=1e-8)
        assert test.terms == ("W",)
        with pytest.raises(DataError, match="is not a baseline stratum"):
            fit.heterogeneity_test(stratum="S=2.0")


class TestST6Refusals:
    def test_strata_rank_before_any_learner(self) -> None:
        frame, _ = strata_frame()
        frame = frame.assign(L=frame["S"] * frame["W"])
        estimator = TMLE(
            **never(),
            cross_fit=False,
            effect_modification=declared("difference", Linear(["W", "L"]), Modifiers(["W", "L"])),
        )
        with pytest.raises(DataError, match=r"inside baseline stratum S=0\.0: the working model"):
            estimator.fit(
                frame, outcome="Y", treatment="A", covariates=["W", "S", "L"], strata=["S"]
            )
        assert NeverFit.calls == 0

    def test_ep_with_strata(self) -> None:
        frame, _ = strata_frame()
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "S"), strata=("S",)),
        )
        effect = study.identify(ConditionalContrast(Modifiers(["W"])))
        record = next(r for r in effect.available_methods() if r.name == "ep")
        assert not record.available
        assert record.refusal.reason is RefusalReason.API_REDIRECTION
        assert record.refusal.alternative.startswith("Declare the stratum column as a modifier")
        with pytest.raises(CapabilityError, match="a baseline stratum is one more modifier"):
            effect.estimate(method="ep", **never())
        assert NeverFit.calls == 0


# ================================================================ repeated splits

from sklearn.linear_model import LogisticRegression  # noqa: E402

from cleverly.inference.multiplier import multiplier_critical_value  # noqa: E402


def binary_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1, w2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    a = rng.binomial(1, expit(0.4 * w1 - 0.3 * w2))
    y = rng.binomial(1, expit(-0.2 + 0.5 * w1 - 0.4 * w2 + a * (0.3 + 0.6 * w1)))
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a.astype(float), "Y": y.astype(float)})


W1_MODIFIERS = Modifiers(["W1"], domains={"W1": (-1.0, 1.0)})


def w1_names(scale: str) -> list[str]:
    return [f"{STEM[scale]}[{term}]" for term in ("(intercept)", "W1")]


def fit_repeated(scale: str = "difference", repeats: int = 3, **settings: Any) -> Any:
    estimator = TMLE(
        outcome_learner=LogisticRegression(max_iter=2000),
        treatment_learner=LogisticRegression(max_iter=2000),
        cross_fit=True,
        n_folds=3,
        repeats=repeats,
        effect_modification=declared(scale, Linear(["W1"]), W1_MODIFIERS),
        random_state=4,
        **settings,
    )
    return estimator.fit(binary_frame(), outcome="Y", treatment="A").single()


def matrix_rule(
    draws: list[dict[str, Any]], keys: list[str], displacement: bool = True
) -> np.ndarray:
    """The matrix rule written out: the candidate of median operator norm (odd draw count)."""
    values = np.array([[draw[key].psi for key in keys] for draw in draws])
    offsets = values - np.median(values, axis=0)
    candidates = []
    for draw, offset in zip(draws, offsets, strict=True):
        curves = np.column_stack([np.asarray(draw[key].influence_curve) for key in keys])
        sigma = np.cov(curves, rowvar=False) / curves.shape[0]
        candidates.append(sigma + (np.outer(offset, offset) if displacement else 0.0))
    norms = [np.linalg.eigvalsh(c)[-1] for c in candidates]
    return candidates[int(np.argsort(norms, kind="stable")[len(candidates) // 2])]


@pytest.fixture(scope="module")
def repeated() -> Any:
    return fit_repeated()


class TestRepeatedSplits:
    def test_rp1_one_split_is_the_shipped_path(self) -> None:
        fit = fit_repeated(repeats=1)
        assert fit.se_without_displacement == {}
        keys = list(fit.estimates)
        np.testing.assert_array_equal(
            fit.covariance(keys), estimate_covariance(fit.estimates, keys)
        )
        assert fit.repeats[0].projection == dict(fit.estimates)
        assert not fit.effect_curve().model.repeated

    def test_rp2_the_matrix_rule(self, repeated: Any) -> None:
        draws = [repeat.projection for repeat in repeated.repeats]
        keys = w1_names("difference")
        assert len({round(draw[keys[1]].psi, 12) for draw in draws}) == 3
        np.testing.assert_allclose(repeated.covariance(keys), matrix_rule(draws, keys), rtol=1e-10)
        one = repeated.covariance([keys[1]])
        assert one[0, 0] == repeated.estimates[keys[1]].variance

    def test_rp3_the_displacement_has_teeth(self, repeated: Any) -> None:
        draws = [repeat.projection for repeat in repeated.repeats]
        keys = w1_names("difference")
        with_d, without_d = matrix_rule(draws, keys), matrix_rule(draws, keys, displacement=False)
        assert np.max(np.abs(with_d - without_d)) / np.max(np.abs(with_d)) >= 1e-8
        for key in keys:
            expected = float(np.median([draw[key].std_error for draw in draws]))
            assert repeated.se_without_displacement[key] == pytest.approx(expected, rel=1e-14)
            # Adding d_r^2 to every draw cannot lower the median, so the reading is at most the
            # reported SE; it equals it when the median draw also has the median variance.
            assert repeated.se_without_displacement[key] <= repeated.estimates[key].std_error
        assert any(
            repeated.se_without_displacement[key] < repeated.estimates[key].std_error
            for key in keys
        )

    def test_rp4_the_test_reads_the_median_and_the_matrix_rule(self, repeated: Any) -> None:
        key = w1_names("difference")[1]
        beta = repeated.estimates[key].psi
        expected = beta**2 / repeated.covariance([key])[0, 0]
        assert repeated.heterogeneity_test().statistic == pytest.approx(expected, rel=1e-12)
        central = beta**2 / estimate_covariance(repeated.estimates, [key])[0, 0]
        assert abs(central - expected) / expected > 1e-6

    def test_rp5_the_curve_is_the_median_rule_per_point(self, repeated: Any) -> None:
        draws = [repeat.projection for repeat in repeated.repeats]
        keys = w1_names("difference")
        grid = {"W1": [-0.8, 0.1, 0.9]}
        phi = np.column_stack([np.ones(3), np.array(grid["W1"])])
        values = np.column_stack([phi @ [d[k].psi for k in keys] for d in draws])
        variances = np.column_stack(
            [np.einsum("ip,pq,iq->i", phi, estimate_covariance(d, keys), phi) for d in draws]
        )
        median = np.median(values, axis=1)
        se = np.sqrt(np.median(variances + (values - median[:, None]) ** 2, axis=1))
        frame = repeated.effect_curve().predict(grid, interval=True)
        np.testing.assert_allclose(frame["estimate"], median, atol=1e-14)
        np.testing.assert_allclose(frame["std_err"], se, rtol=1e-12)
        # The witness: the curve at the median coefficients is another number.
        at_median = phi @ points(repeated, keys)
        assert np.max(np.abs(at_median - median)) > 1e-8

    def test_rp6_the_band_takes_the_median_critical_value(self, repeated: Any) -> None:
        draws = [repeat.projection for repeat in repeated.repeats]
        keys = w1_names("difference")
        grid = {"W1": [-0.8, 0.1, 0.9]}
        phi = np.column_stack([np.ones(3), np.array(grid["W1"])])
        critical = []
        for draw in draws:
            influence = np.column_stack([np.asarray(draw[k].influence_curve) for k in keys])
            se = np.sqrt(np.einsum("ip,pq,iq->i", phi, estimate_covariance(draw, keys), phi))
            critical.append(
                multiplier_critical_value(
                    influence @ phi.T,
                    se,
                    n=influence.shape[0],
                    alpha=0.05,
                    n_replicates=500,
                    kind="rademacher",
                    random_state=3,
                )
            )
        curve = repeated.effect_curve(grid, n_replicates=500, random_state=3)
        assert curve.band.critical_value == pytest.approx(float(np.median(critical)), rel=1e-12)
        assert curve.status == "experimental"

    def test_rp7_a_draw_that_withholds_withholds_the_report(self, repeated: Any) -> None:
        from dataclasses import replace

        keys = w1_names("difference")
        withheld = dict(repeated.repeats[1].projection)
        withheld[keys[1]] = replace(withheld[keys[1]], inference="no_inference")
        repeats = list(repeated.repeats)
        repeats[1] = replace(repeats[1], projection=withheld)
        mutated = replace(repeated, repeats=tuple(repeats))
        with pytest.raises(Exception, match=r"inference|interval"):
            mutated.covariance(keys).sum()
            mutated.effect_curve().predict({"W1": [0.0]}, interval=True)

    def test_rp8_the_curve_round_trips(self, repeated: Any, tmp_path: Any) -> None:
        from cleverly import load

        curve = repeated.effect_curve()
        path = curve.save(tmp_path / "curve.joblib")
        loaded = load(path)
        assert len(loaded.model.draws) == 3
        grid = {"W1": [-0.5, 0.5]}
        np.testing.assert_array_equal(
            loaded.predict(grid, interval=True).to_numpy(),
            curve.predict(grid, interval=True).to_numpy(),
        )

    def test_rp9_another_repeated_point_fit_keeps_the_refusal(self) -> None:
        fit = (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=2000),
                treatment_learner=LogisticRegression(max_iter=2000),
                n_folds=3,
                repeats=3,
                estimands=("ate",),
                random_state=4,
            )
            .fit(binary_frame(), outcome="Y", treatment="A")
            .single()
        )
        assert fit.repeats[0].projection is None
        with pytest.raises(CapabilityError, match="covariance\\(\\)"):
            fit.covariance(["ate"])


# ===================================================================== bootstrap

import cleverly.inference.bootstrap as bootstrap_module  # noqa: E402


def fit_bootstrapped(scale: str = "difference", n_bootstrap: int = 6, **settings: Any) -> Any:
    settings = {"cross_fit": False, **settings}
    estimator = TMLE(
        outcome_learner=LogisticRegression(max_iter=2000),
        treatment_learner=LogisticRegression(max_iter=2000),
        n_bootstrap=n_bootstrap,
        effect_modification=declared(scale, Linear(["W1"]), W1_MODIFIERS),
        random_state=4,
        **settings,
    )
    frame = settings.pop("frame", None)
    return estimator, estimator.fit(
        binary_frame() if frame is None else frame, outcome="Y", treatment="A"
    ).single()


class TestTheBootstrap:
    @pytest.mark.parametrize("scale", SCALES)
    def test_bt1_bt2_each_replicate_is_a_refit(
        self, scale: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        original = tmle_module.TMLE._bootstrap_point_estimates

        def spy(self: Any, replicate: Any, *rest: Any) -> Any:
            seen.append(replicate)
            return original(self, replicate, *rest)

        monkeypatch.setattr(tmle_module.TMLE, "_bootstrap_point_estimates", spy)
        _, fit = fit_bootstrapped(scale)
        keys = w1_names(scale)
        for index in (0, 3):
            fresh = TMLE(
                outcome_learner=LogisticRegression(max_iter=2000),
                treatment_learner=LogisticRegression(max_iter=2000),
                cross_fit=False,
                effect_modification=declared(scale, Linear(["W1"]), W1_MODIFIERS),
                random_state=4,
            )._fit_single(seen[index], intermediate_value=None)
            for key in keys:
                assert fit.estimates[key].bootstrap.draws[index] == pytest.approx(
                    fresh.estimates[key].psi, abs=1e-10
                )
        for key in keys:
            summary = fit.estimates[key].bootstrap
            assert summary.ci == fit.bootstrap.interval(key, 0.05).ci
            assert min(summary.draws) <= summary.ci[0] <= summary.ci[1] <= max(summary.draws)

    def test_bt3_bt7_the_reason_of_a_cross_fitted_bootstrap(self) -> None:
        _, stacked = fit_bootstrapped(cross_fit=True, n_folds=3)
        _, in_sample = fit_bootstrapped()
        for key in w1_names("difference"):
            reason = stacked.estimates[key].bootstrap.support.reason
            assert stacked.estimates[key].bootstrap.support.label == "experimental"
            assert reason == tmle_module.CROSS_FITTED_BOOTSTRAP_REASON
            # No row measures the in-sample projection bootstrap until its study registers.
            assert in_sample.estimates[key].bootstrap.support.reason != reason

    def test_bt7_attach_bootstrap_keeps_the_generic_reason_by_default(self) -> None:
        from cleverly.inference.results import attach_bootstrap

        _, stacked = fit_bootstrapped(cross_fit=True, n_folds=3)
        estimator = stacked.estimator
        preflight = estimator._preflight_fit_configuration(stacked.data)
        classification = tmle_module.classify_point_fit(
            estimator, stacked.data, preflight.estimands, source="test"
        )
        plain = attach_bootstrap(stacked, stacked.bootstrap, classification=classification)
        key = w1_names("difference")[0]
        assert (
            plain.estimates[key].bootstrap.support.reason
            != tmle_module.CROSS_FITTED_BOOTSTRAP_REASON
        )

    def test_bt4_the_test_and_the_curve_ignore_the_bootstrap(self) -> None:
        _, with_boot = fit_bootstrapped()
        _, without = fit_bootstrapped(n_bootstrap=0)
        assert with_boot.heterogeneity_test().statistic == without.heterogeneity_test().statistic
        grid = {"W1": [-0.5, 0.5]}
        np.testing.assert_array_equal(
            with_boot.effect_curve().predict(grid, interval=True).to_numpy(),
            without.effect_curve().predict(grid, interval=True).to_numpy(),
        )

    def test_bt5_a_cluster_bootstrap_resamples_whole_clusters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        original = tmle_module.TMLE._bootstrap_point_estimates

        def spy(self: Any, replicate: Any, *rest: Any) -> Any:
            seen.append(replicate)
            return original(self, replicate, *rest)

        monkeypatch.setattr(tmle_module.TMLE, "_bootstrap_point_estimates", spy)
        frame = binary_frame().assign(cluster=lambda f: np.arange(len(f)) // 3)
        estimator = TMLE(
            outcome_learner=LogisticRegression(max_iter=2000),
            treatment_learner=LogisticRegression(max_iter=2000),
            cross_fit=False,
            n_bootstrap=4,
            effect_modification=declared("difference", Linear(["W1"]), W1_MODIFIERS),
            random_state=4,
        )
        fit = estimator.fit(frame, outcome="Y", treatment="A", id="cluster").single()
        assert fit.bootstrap.resampling == "cluster"
        for replicate in seen:
            sizes = np.bincount(np.unique(replicate.cluster, return_inverse=True)[1])
            assert set(sizes.tolist()) == {3}

    def test_bt6_a_singular_stratum_replicate_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = binary_frame(300, 5)
        frame["S"] = (frame["W2"] > 0).astype(float)
        frame["L"] = (frame["W1"] > 0.6).astype(float)
        rows = np.arange(len(frame))
        # Replicate 1 holds no row with L = 1 inside stratum S = 1, so its basis is singular there.
        singular = rows[~((frame["S"] == 1.0) & (frame["L"] == 1.0)).to_numpy()]
        supplied = iter([rows, np.resize(singular, rows.size), rows, rows])

        def draw(n: int, cluster: Any, rng: Any, members: Any = None) -> Any:
            return next(supplied), None

        monkeypatch.setattr(bootstrap_module, "_bootstrap_draw", draw)
        estimator = TMLE(
            outcome_learner=LogisticRegression(max_iter=2000),
            treatment_learner=LogisticRegression(max_iter=2000),
            cross_fit=False,
            n_bootstrap=4,
            effect_modification=declared(
                "difference",
                Linear(["W1", "L"]),
                Modifiers(["W1", "L"], domains={"W1": (-1.0, 1.0)}),
            ),
            random_state=4,
        )
        with pytest.warns(UserWarning, match="1 of 4 bootstrap replicates failed"):
            fit = estimator.fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2", "S", "L"], strata=["S"]
            ).single()
        assert fit.bootstrap.n_failed == 1
        # The rule itself, on that replicate's rows.
        replicate = fit.data.subset(np.resize(singular, rows.size))
        from cleverly.effect_modification import EffectModificationSet
        from cleverly.estimators._strata import check_stratified_projection_rank

        with pytest.raises(DataError, match=r"inside baseline stratum S=1\.0"):
            check_stratified_projection_rank(
                EffectModificationSet.evaluate(estimator.effect_modification, replicate), replicate
            )
