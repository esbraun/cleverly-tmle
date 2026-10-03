"""DR-TMLE with baseline strata: each stratum is the DR-TMLE of the law given the stratum.

A stratified ``DRTMLE`` fit puts every targeting equation in stratum form, one block
``I(S = s) H / P_n(S = s)`` per stratum, and fits every reduced regression inside each
stratum.  The equations of different strata then share no row, so the stratum estimates must
equal those of separate unstratified fits on the stratum subsets, which fit the same reduced
regressions on the same rows.  The nuisances are fixed functions, so the subset fits see the
same initial arrays, and the reduced learner is the saturated cell mean, which is exact on this
finite support.  Equality holds to the alternation's tolerance.

The routes are the complete-data construction at two and three arms and at every guard, under
both reductions; Díaz and van der Laan's missing-outcome construction at two arms with a
randomized treatment and at three arms with known probabilities; and the composite indicator,
with a missing outcome and with a missing treatment.  The marginal estimate is the stratum
mixture, and its curve is the linearity curve.

The treatment mechanism the fit uses depends on ``W`` alone, so a value of ``g_n`` falls in
every stratum.  A reduced regression fitted on all rows would then pool strata, and the
mutation that fits them so must fail the subset comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from tests.discrete_law_longitudinal import CellMeans
from tests.studies import stratified_law as law
from tests.unit._declaration_support import assert_every_witness_fails

N = 1_500
SEED = 20261104
#: Each stratum estimate equals its subset fit to the closing pass's tolerance.  The pass
#: stops equation (9) on the worst stratum, so a stratum can take one step more than its
#: subset fit.  Measured: ``psi`` within ``1e-10`` and curves within ``1.1e-7``.
TOLERANCE = 1e-9
CURVE_TOLERANCE = 1e-6


class WOnlyTreatment(BaseEstimator, ClassifierMixin):
    """A treatment mechanism that reads ``W`` alone, at two or three arms.

    ``W`` is the first design column in the fit and in a subset fit alike.
    """

    def __init__(self, arms: int = 2) -> None:
        self.arms = arms

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> WOnlyTreatment:
        self.classes_ = np.arange(self.arms, dtype=float)
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        w = np.asarray(design, dtype=float)[:, 0]
        if self.arms == 2:
            p = expit(-0.2 + 0.45 * w)
            return np.column_stack([1.0 - p, p])
        scores = np.column_stack([np.zeros_like(w), -0.3 + 0.4 * w, -0.8 + 0.3 * w])
        weights = np.exp(scores)
        return weights / weights.sum(axis=1, keepdims=True)


def _decode(values: np.ndarray, arms: int) -> tuple[np.ndarray, np.ndarray]:
    """The arm code and the covariates of ``[A, W]``, whose arm block has ``arms - 1`` columns.

    A binary arm is one column of codes, and ``K`` arms are ``K - 1`` indicators.
    """
    width = arms - 1
    block = values[:, :width]
    codes = block[:, 0] if arms == 2 else block @ np.arange(1, arms, dtype=float)
    return codes, values[:, width:]


class ArmOutcome(BaseEstimator, ClassifierMixin):
    """The law's outcome regression at any arm code, shifted on the logit scale.

    The shift is that of
    :class:`~tests.unit._stratified_alternating_support.PerturbedOutcome`.  ``stratum``
    replaces ``V`` in a subset fit, whose design drops the constant ``V``.
    """

    def __init__(self, arms: int = 2, stratum: int | None = None) -> None:
        self.arms = arms
        self.stratum = stratum

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ArmOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        a, covariates = _decode(np.asarray(design, dtype=float), self.arms)
        w = covariates[:, 0]
        v = covariates[:, 1] if self.stratum is None else np.full(len(w), float(self.stratum))
        p = expit(logit(law.outcome(a, w, v)) + 0.35 + 0.3 * a - 0.25 * v)
        return np.column_stack([1.0 - p, p])


class FixedResponse(BaseEstimator, ClassifierMixin):
    """A response mechanism of the outcome, ``P(Delta = 1 | A, W, V)``, or of the treatment.

    ``treatment=True`` reads ``(W, V)``, the design of the treatment's response.  Otherwise
    the design is ``(A, W, V)``.  ``stratum`` replaces ``V`` in a subset fit.
    """

    def __init__(self, treatment: bool = False, arms: int = 2, stratum: int | None = None) -> None:
        self.treatment = treatment
        self.arms = arms
        self.stratum = stratum

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> FixedResponse:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if self.treatment:
            w = values[:, 0]
            v = values[:, 1] if self.stratum is None else np.full(len(w), float(self.stratum))
            p = expit(1.6 + 0.2 * w - 0.3 * v)
        else:
            a, covariates = _decode(values, self.arms)
            w = covariates[:, 0]
            v = covariates[:, 1] if self.stratum is None else np.full(len(w), float(self.stratum))
            p = expit(1.1 - 0.3 * (a > 0) + 0.2 * w - 0.25 * v)
        return np.column_stack([1.0 - p, p])


def rows(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """The baseline-strata law with a third arm, a missing outcome and a missing treatment."""
    frame = law.sample(n, seed)
    rng = np.random.default_rng(seed + 1)
    a, w, v = (frame[name].to_numpy(dtype=float) for name in ("A", "W", "V"))
    observed = rng.random(n) < expit(1.1 - 0.3 * a + 0.2 * w - 0.25 * v)
    recorded = rng.random(n) < expit(1.6 + 0.2 * w - 0.3 * v)
    third = rng.random(n) < 0.3
    a3 = np.where(third, 2.0, a)
    y3 = (rng.random(n) < law.outcome(a3, w, v)).astype(float)
    return frame.assign(
        Delta=observed.astype(float),
        Yobs=np.where(observed, frame["Y"].to_numpy(dtype=float), np.nan),
        A3=a3,
        Y3=y3,
        Y3obs=np.where(observed, y3, np.nan),
        DeltaA=recorded.astype(float),
        Amiss=np.where(recorded, a, np.nan),
    )


#: Each route: its roles, the treatment arms, and its estimator settings.
ROUTES: dict[str, tuple[dict[str, Any], int, dict[str, Any]]] = {
    "complete": ({}, 2, {}),
    "complete_three_arm": ({"treatment": "A3", "outcome": "Y3"}, 3, {}),
    "missing_randomized": ({"outcome": "Yobs", "delta": "Delta"}, 2, {"randomized": True}),
    "missing_known_three_arm": (
        {"outcome": "Y3obs", "treatment": "A3", "delta": "Delta"},
        3,
        {"known": True},
    ),
    "composite_missing_outcome": ({"outcome": "Yobs", "delta": "Delta"}, 2, {}),
    "composite_missing_treatment": (
        {"treatment": "Amiss", "treatment_delta": "DeltaA"},
        2,
        {},
    ),
}


def known_probabilities(frame: pd.DataFrame) -> np.ndarray:
    return WOnlyTreatment(3).fit(None, None).predict_proba(frame[["W"]].to_numpy())


def fit(
    frame: pd.DataFrame,
    route: str,
    *,
    guard: tuple[str, ...] = ("Q", "g"),
    reduction: str = "univariate",
    stratum: int | None = None,
    max_outer: int = 1,
    weights: str | None = None,
) -> Any:
    """The stratified fit, or with ``stratum`` the unstratified fit of that stratum's rows.

    ``max_outer=1`` runs one round and the closing pass in every fit.  Each solve inside a
    round has a unique root along its covariate and each reduced regression is a cell mean,
    so a stratum's round is the subset fit's round.  More rounds would let the alternation's
    global stop rule end the stratified fit and a subset fit at different rounds.
    """
    roles, arms, settings = ROUTES[route]
    settings = dict(settings)
    known = settings.pop("known", False)
    estimator = DRTMLE(
        guard=guard,
        reduction=reduction,
        estimands=("ey", "ate"),
        outcome_learner=ArmOutcome(arms, stratum),
        treatment_learner=WOnlyTreatment(arms),
        missingness_learner=FixedResponse(
            treatment="treatment_delta" in roles, arms=arms, stratum=stratum
        ),
        reduced_outcome_learner=CellMeans(),
        reduced_treatment_learner=CellMeans(),
        cross_fit=False,
        simultaneous=False,
        g_bounds=(1e-6, 1.0 - 1e-6),
        max_iter=100,
        tol=1e-10,
        max_outer=max_outer,
        random_state=0,
        **settings,
    )
    extra: dict[str, Any] = {}
    if known:
        extra["treatment_probabilities"] = known_probabilities(frame)
    if stratum is None:
        extra["strata"] = ["V"]
    if weights is not None:
        extra["weights"] = weights
    return estimator.fit(
        frame,
        **{"outcome": "Y", "treatment": "A", "covariates": ["W", "V"], **roles, **extra},
    ).single()


def subset(frame: pd.DataFrame, stratum: int) -> pd.DataFrame:
    return frame[frame["V"] == stratum].reset_index(drop=True)


def check_subsets(result: Any, frame: pd.DataFrame, route: str, **settings: Any) -> None:
    """Each stratum estimate and curve equals the subset fit's."""
    v = frame["V"].to_numpy()
    for s in law.STRATA:
        alone = fit(subset(frame, s), route, stratum=s, **settings)
        inside = v == s
        share = float(inside.mean())
        observation = alone.fluctuations["mean"].reduction.observation
        if observation is not None:
            # The observation-drift scores, one per arm and stratum, arm-major.  A stratum's
            # block is the subset fit's score: both average over the stratum's rows.
            pooled_scores = result.fluctuations["mean"].reduction.observation.score
            arms = len(observation.score)
            data = result.data
            code = [data.stratum_label(c) for c in range(data.n_strata)].index(f"V={s}")
            blocks = np.asarray(pooled_scores).reshape(arms, len(law.STRATA))[:, code]
            np.testing.assert_allclose(blocks, observation.score, rtol=0.0, atol=TOLERANCE)
        for name, estimate in alone.estimates.items():
            pooled = result[f"{name}[V={s}]"]
            assert pooled.psi == pytest.approx(estimate.psi, abs=TOLERANCE), (route, name, s)
            np.testing.assert_allclose(
                pooled.influence_curve[inside] * share,
                estimate.influence_curve,
                rtol=0.0,
                atol=CURVE_TOLERANCE,
                err_msg=f"{route} {name} V={s}",
            )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return rows()


CASES = [
    pytest.param("complete", ("Q", "g"), "univariate", id="complete-Qg"),
    pytest.param("complete", ("Q",), "univariate", id="complete-Q"),
    pytest.param("complete", ("g",), "univariate", id="complete-g"),
    pytest.param("complete", ("Q", "g"), "bivariate", id="complete-Qg-bivariate"),
    pytest.param("complete_three_arm", ("Q", "g"), "univariate", id="three-arm-Qg"),
    pytest.param("missing_randomized", ("Q", "g"), "univariate", id="missing-randomized"),
    pytest.param("missing_known_three_arm", ("Q", "g"), "univariate", id="missing-known-3"),
    pytest.param("composite_missing_outcome", ("Q", "g"), "univariate", id="composite-outcome"),
    pytest.param("composite_missing_outcome", ("Q",), "univariate", id="composite-outcome-Q"),
    pytest.param("composite_missing_treatment", ("Q", "g"), "univariate", id="composite-treatment"),
]


@pytest.mark.parametrize(("route", "guard", "reduction"), CASES)
def test_each_stratum_is_its_subset_fit(
    frame: pd.DataFrame, route: str, guard: tuple[str, ...], reduction: str
) -> None:
    result = fit(frame, route, guard=guard, reduction=reduction)
    check_subsets(result, frame, route, guard=guard, reduction=reduction)


@pytest.mark.parametrize(("route", "guard", "reduction"), CASES)
def test_the_marginal_is_the_stratum_mixture(
    frame: pd.DataFrame, route: str, guard: tuple[str, ...], reduction: str
) -> None:
    result = fit(frame, route, guard=guard, reduction=reduction)
    v = frame["V"].to_numpy()
    marginal_names = [
        name for name in result.estimates if not name.endswith("]") or "[V=" not in name
    ]
    for name in marginal_names:
        marginal = result[name]
        mixture = 0.0
        linear = np.zeros(len(frame))
        for s in law.STRATA:
            stratum = result[f"{name}[V={s}]"]
            inside = v == s
            mixture += float(inside.mean()) * stratum.psi
            linear[inside] = stratum.influence_curve[inside] * float(inside.mean()) + stratum.psi
        assert marginal.psi == pytest.approx(mixture, abs=1e-12)
        np.testing.assert_allclose(
            marginal.influence_curve, linear - marginal.psi, rtol=0.0, atol=1e-10
        )


def test_every_block_of_every_equation_is_solved(frame: pd.DataFrame) -> None:
    """The solver's own scores, one per arm and stratum, at the state the fit reports."""
    result = fit(frame, "complete")
    fluctuation = result.fluctuations["mean"]
    assert fluctuation.epsilon.shape == (2 * len(law.STRATA),)
    assert fluctuation.mechanism.score.shape == (2 * len(law.STRATA),)
    assert fluctuation.reduction.score.shape == (2 * len(law.STRATA),)
    for score in (fluctuation.score, fluctuation.mechanism.score, fluctuation.reduction.score):
        assert float(np.max(np.abs(score))) < 1e-8


def test_reductions_pooled_across_strata_fail(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: the reduced regressions fitted on every row, as an unstratified fit does."""
    import cleverly.estimators.reduced as reduced_module

    original = reduced_module._pooled_column

    def pooled(learner: Any, **kwargs: Any) -> Any:
        return original(learner, **kwargs)

    def ignore_strata(learner: Any, **kwargs: Any) -> Any:
        kwargs.pop("companion_strata")
        return pooled(learner, **kwargs)

    monkeypatch.setattr(reduced_module, "_stratified_column", ignore_strata)
    mutated = fit(frame, "complete")
    monkeypatch.undo()
    assert_every_witness_fails([lambda: check_subsets(mutated, frame, "complete")])


def test_a_nested_reduction_solves_every_block(frame: pd.DataFrame) -> None:
    """The nested construction fits each stratum's reductions on the nested designs."""
    result = (
        DRTMLE(
            reduced_crossfit="nested",
            estimands=("ey", "ate"),
            outcome_learner=ArmOutcome(),
            treatment_learner=WOnlyTreatment(2),
            reduced_outcome_learner=CellMeans(),
            reduced_treatment_learner=CellMeans(),
            n_folds=3,
            simultaneous=False,
            g_bounds=(1e-6, 1.0 - 1e-6),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        .single()
    )
    fluctuation = result.fluctuations["mean"]
    assert fluctuation.reduction.converged
    for score in (fluctuation.score, fluctuation.mechanism.score, fluctuation.reduction.score):
        assert score.shape[0] % len(law.STRATA) == 0
        assert float(np.max(np.abs(score))) < 1e-6


class TestTheStratifiedCompanionIsTheFit:
    """The evaluation companion of a stratified fit moves with each row's own stratum.

    Handed the fitting rows back as the companion, fold ``k``'s slab at the rows fold ``k``
    holds out must be the production array: the reduced regressions predict there with the
    stratum's own model, and every carried covariate takes the row's stratum block.  The
    companion changes no number of the fit.
    """

    @staticmethod
    def build(frame: pd.DataFrame, evaluation: Any = None) -> Any:
        return (
            DRTMLE(
                estimands=("ey", "ate"),
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                reduced_outcome_learner=LinearRegression(),
                reduced_treatment_learner=LogisticRegression(max_iter=1000),
                n_folds=3,
                simultaneous=False,
                random_state=0,
                evaluation=evaluation,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
            .single()
        )

    @pytest.fixture(scope="class")
    def pair(self) -> tuple[Any, Any]:
        frame = rows(360, SEED + 2)[["V", "W", "A", "Y"]]
        return self.build(frame), self.build(frame, evaluation=frame)

    def test_the_companion_changes_nothing(self, pair: tuple[Any, Any]) -> None:
        plain, paired = pair
        for name, estimate in plain.estimates.items():
            assert paired[name].psi == estimate.psi

    def test_every_moved_array_matches_fold_by_fold(self, pair: tuple[Any, Any]) -> None:
        _, paired = pair
        fluctuation = paired.fluctuations["mean"]
        record = fluctuation.reduction
        assignment = np.asarray(paired.nuisance.folds.assignment)
        for fold in range(record.evaluation.n_folds):
            held = np.flatnonzero(assignment == fold)
            moved = record.evaluation.outcome[fold]
            np.testing.assert_allclose(
                moved.observed[held], fluctuation.targeted.observed[held], rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                record.evaluation.propensity[fold].arm(1.0)[held],
                fluctuation.mechanism.propensity[held],
                rtol=0.0,
                atol=1e-12,
            )
            for name in ("qr", "gr1", "gr2"):
                np.testing.assert_allclose(
                    getattr(record.evaluation.reduced[fold], name)[held],
                    getattr(record.reduced, name)[held],
                    rtol=0.0,
                    atol=1e-12,
                )


class TestACompanionInAnotherStratumOrder:
    """A companion whose rows meet the strata in another order, and hold only two of them.

    Stratum codes follow first appearance, so the companion's own codes name different strata
    from the fit's.  Each companion row must still take its own stratum's reduced regressions
    and blocks: at the companion rows that fold ``k`` holds out, fold ``k``'s slab equals the
    production array at the matching fitting rows.
    """

    @pytest.fixture(scope="class")
    def frame(self) -> pd.DataFrame:
        frame = rows(360, SEED + 2)[["V", "W", "A", "Y"]]
        assert frame["V"].iloc[0] != 2
        return frame

    @staticmethod
    def companion_order(frame: pd.DataFrame) -> np.ndarray:
        """The fitting rows of strata 2 and 1, stratum 2 first."""
        kept = np.flatnonzero(frame["V"].to_numpy() > 0)
        return kept[np.argsort(-frame["V"].to_numpy()[kept], kind="stable")]

    def paired(self, frame: pd.DataFrame) -> Any:
        companion = frame.iloc[self.companion_order(frame)].reset_index(drop=True)
        return TestTheStratifiedCompanionIsTheFit.build(frame, evaluation=companion)

    def check(self, paired: Any, frame: pd.DataFrame) -> None:
        order = self.companion_order(frame)
        fluctuation = paired.fluctuations["mean"]
        record = fluctuation.reduction
        assignment = np.asarray(paired.nuisance.folds.assignment)
        for fold in range(record.evaluation.n_folds):
            at = np.flatnonzero(assignment[order] == fold)
            rows_ = order[at]
            for name in ("qr", "gr1", "gr2"):
                np.testing.assert_allclose(
                    getattr(record.evaluation.reduced[fold], name)[at],
                    getattr(record.reduced, name)[rows_],
                    rtol=0.0,
                    atol=1e-12,
                    err_msg=f"{name} fold {fold}",
                )
            np.testing.assert_allclose(
                record.evaluation.outcome[fold].observed[at],
                fluctuation.targeted.observed[rows_],
                rtol=0.0,
                atol=1e-12,
            )

    def test_each_companion_row_reads_its_own_stratum(self, frame: pd.DataFrame) -> None:
        self.check(self.paired(frame), frame)

    def test_the_companion_s_own_codes_fail(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: the companion's first-appearance codes used as the fit's codes."""
        import cleverly.estimators.reduced as reduced_module
        import cleverly.estimators.targeting as targeting_module

        def raw(data: Any, companion: Any) -> np.ndarray:
            return np.asarray(companion.strata, dtype=np.int64)

        monkeypatch.setattr(reduced_module, "companion_stratum_codes", raw)
        monkeypatch.setattr(targeting_module, "companion_stratum_codes", raw)
        mutated = self.paired(frame)
        monkeypatch.undo()
        assert_every_witness_fails([lambda: self.check(mutated, frame)])


class TestAWeightedStratumIsTheWeightedSubsetFit:
    """The complete-data route under fixed analysis weights.

    The blocks of every equation read the weighted share ``P_n(S = s)`` in
    ``targeting.stratum_probabilities``, and the un-scaling reads it in
    ``tmle.stratum_probabilities``.  A weighted subset fit sees neither.
    """

    @pytest.fixture(scope="class")
    def weighted(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(wt=np.random.default_rng(3).uniform(0.3, 2.5, len(frame)))

    def test_each_weighted_stratum_is_its_weighted_subset_fit(self, weighted: pd.DataFrame) -> None:
        result = fit(weighted, "complete", weights="wt")
        check_subsets(result, weighted, "complete", weights="wt")

    @pytest.mark.parametrize("module", ["targeting", "tmle"])
    def test_an_unweighted_share_fails(
        self, weighted: pd.DataFrame, module: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        def unweighted(data: Any) -> list[float]:
            return [float(np.mean(data.strata == code)) for code in range(data.n_strata)]

        target = importlib.import_module(f"cleverly.estimators.{module}")
        monkeypatch.setattr(target, "stratum_probabilities", unweighted)
        mutated = fit(weighted, "complete", weights="wt")
        monkeypatch.undo()
        assert_every_witness_fails(
            [lambda: check_subsets(mutated, weighted, "complete", weights="wt")]
        )
