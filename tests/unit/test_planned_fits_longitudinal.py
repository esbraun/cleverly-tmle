"""The planned-fit preview of the longitudinal, survival and continuous-time engines.

Each route previews a fit with counting learners, runs it, and compares the counts
(:func:`tests.unit._planned_fits_support.check`).  A mechanism row is exact.  The regression
at a horizon node reads data, so it is exact too; a node below it regresses the later node's
prediction and is an upper row whose minimum is zero.  The fixtures here have no constant
pseudo-outcome, so every upper row of a draw is also reached.  The mutation controls patch
one term out of the preview and show the comparison then fails.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cleverly import (
    CausalStudy,
    IncrementalEffect,
    LongitudinalTreatment,
    ParametricGFormulaMethod,
    PlannedFits,
    RegimeContrast,
    RegimeMean,
    TimeToEvent,
)
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
    make_point_survival,
)
from cleverly.interventions import Incremental, LinearQuantileRegression, Shift, SupportBoundary
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import sequential as sequential_module
from cleverly.survival import CoxHazard
from cleverly.survival import cox as cox_module
from cleverly.survival import engine as engine_module
from tests import discrete_law_longitudinal_horizon as law_
from tests import law_q_tuned as lawq
from tests import longitudinal_horizon_plans as plans_
from tests.unit import _planned_fits_support as support

ROLES = (
    "outcome_learner",
    "pseudo_learner",
    "treatment_learner",
    "censoring_learner",
    "competing_learner",
)
PLAIN: dict[str, str | None] = dict.fromkeys(ROLES) | {
    "covariate_learner": None,
    "hazard_learner": None,
}
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _learners(**overrides: Any) -> dict[str, Any]:
    """Counting learners on every role, three outer folds, seeded."""
    return {
        "outcome_learner": support.logistic("outcome_learner"),
        "pseudo_learner": support.linear("pseudo_learner"),
        "treatment_learner": support.logistic("treatment_learner"),
        "censoring_learner": support.logistic("censoring_learner"),
        "learner_folds": 2,
        "n_folds": 3,
        "random_state": 5,
        "simultaneous": False,
        **overrides,
    }


def _hazard(**overrides: Any) -> dict[str, Any]:
    """:func:`_learners` for a hazard fit of baseline covariates, which fits no pseudo-outcome."""
    settings = _learners(**{"survival_model": "hazard", "targeting": "one_step", **overrides})
    del settings["pseudo_learner"]
    return settings


def _mechanism_only(**overrides: Any) -> dict[str, Any]:
    """:func:`_learners` without the regressions, for an IPW fit."""
    settings = _learners(**overrides)
    del settings["outcome_learner"], settings["pseudo_learner"]
    return settings


def _ensembles(**overrides: Any) -> dict[str, Any]:
    """Counting Super Learners on the mechanism and the regressions."""
    return _learners(
        outcome_learner=support.ensemble("outcome_learner", n_folds=3),
        pseudo_learner=support.ensemble(
            "pseudo_learner", base=LinearRegression, n_folds=3, task="regression"
        ),
        treatment_learner=support.ensemble("treatment_learner", n_folds=3),
        censoring_learner=support.ensemble("censoring_learner", n_folds=3),
        **overrides,
    )


def _two_node() -> CausalStudy:
    frame, _ = make_longitudinal(n=300, seed=12)
    return CausalStudy(frame, design=LongitudinalTreatment(**COLUMNS))


def _survival(n: int = 400, seed: int = 12) -> CausalStudy:
    frame, _ = make_longitudinal_survival(n=n, seed=seed)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
    )


def _point_survival(n_times: int = 4, **options: Any) -> CausalStudy:
    frame, _ = make_point_survival(n=600, seed=3, n_times=n_times, **options)
    return CausalStudy(
        frame,
        design=TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"]),
    )


def _time_varying() -> CausalStudy:
    frame, _ = make_point_survival(n=600, seed=3, n_times=3, layout="wide", time_varying=True)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome=["Y1", "Y2", "Y3"],
            treatment="A",
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"], ["L3"]],
            censoring=["C1", "C2", "C3"],
        ),
    )


def _competing(eliminate: bool = False) -> CausalStudy:
    frame, _ = make_longitudinal_competing(n=400, seed=5)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")},
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
            eliminate=["death"] if eliminate else (),
        ),
    )


def _dose_frame(n: int = 400, seed: int = 4) -> pd.DataFrame:
    """Two continuous doses, a covariate between them, a binary outcome, no censoring."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = 0.5 * w + rng.normal(size=n)
    l2 = rng.binomial(1, 1 / (1 + np.exp(-0.3 * a1))).astype(float)
    a2 = 0.3 * a1 + 0.3 * l2 + rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * a2 + 0.2 * w)))).astype(float)
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})


def _dose_study() -> CausalStudy:
    return CausalStudy(
        _dose_frame(),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        ),
    )


def _dose_regimens() -> dict[str, Any]:
    """The natural course and a shift of every dose."""
    return {"natural": Shift(0.0, cap=None), "up": Shift(0.5, cap=None)}


def _binary_covariate() -> CausalStudy:
    """:func:`_two_node` with ``L2`` cut at its median, so the g-formula models it as binary."""
    frame, _ = make_longitudinal(n=300, seed=12)
    frame["L2"] = (frame["L2"] > frame["L2"].median()).astype(float).where(frame["L2"].notna())
    return CausalStudy(frame, design=LongitudinalTreatment(**COLUMNS))


def _gformula() -> dict[str, Any]:
    return {
        "outcome_learner": support.logistic("outcome_learner"),
        "treatment_learner": support.logistic("treatment_learner"),
        "censoring_learner": support.logistic("censoring_learner"),
        "n_folds": 1,
        "cross_fit": False,
        "random_state": 5,
        "simultaneous": False,
    }


STUDY_ROUTES: dict[str, tuple[Callable[[], CausalStudy], Any, Any, Callable[[], dict]]] = {
    "ltmle_cross_fitted": (_two_node, RegimeMean({"always": 1, "never": 0}), None, _learners),
    "ltmle_in_sample": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        None,
        lambda: _learners(cross_fit=False),
    ),
    "ltmle_repeats": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        None,
        lambda: _learners(repeats=2),
    ),
    "ltmle_ensembles": (_two_node, RegimeMean({"always": 1, "never": 0}), None, _ensembles),
    "ltmle_bootstrap": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        None,
        lambda: _learners(n_bootstrap=2, cross_fit=False),
    ),
    "sdr_cross_fitted": (_two_node, RegimeMean({"always": 1, "never": 0}), "sdr", _learners),
    "sdr_in_sample": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        "sdr",
        lambda: _learners(cross_fit=False),
    ),
    "ipsi_two_deltas": (
        _two_node,
        IncrementalEffect([Incremental(1.0), Incremental(2.0)]),
        "ipsi",
        _learners,
    ),
    "survival_two_horizons": (
        _survival,
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        None,
        _learners,
    ),
    "survival_pooled": (
        lambda: _point_survival(),
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        None,
        lambda: _learners(hazard_pooling="pooled"),
    ),
    "competing": (
        _competing,
        RegimeContrast(regimens={"always": 1, "never": 0}, reference="never"),
        None,
        _learners,
    ),
    "hazard_node": (
        lambda: _point_survival(),
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        None,
        _hazard,
    ),
    "hazard_pooled": (
        lambda: _point_survival(),
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        None,
        lambda: _hazard(hazard_pooling="pooled"),
    ),
    "hazard_time_varying": (
        _time_varying,
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        None,
        lambda: _learners(survival_model="hazard", targeting="one_step"),
    ),
    "survival_ctmle": (
        lambda: _point_survival(),
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        "collaborative_tmle",
        lambda: {
            "outcome_learner": support.logistic("outcome_learner"),
            "n_folds": 3,
            "random_state": 5,
            "simultaneous": False,
        },
    ),
    "survival_drtmle": (
        lambda: _point_survival(),
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
        "drtmle",
        lambda: _hazard(targeting="iterative"),
    ),
    "gcomp": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        "gcomp",
        lambda: _learners(n_folds=1, cross_fit=False),
    ),
    "ipw": (
        _two_node,
        RegimeMean({"always": 1, "never": 0}),
        "ipw",
        lambda: _mechanism_only(n_folds=1, cross_fit=False),
    ),
    "parametric_gformula": (
        _binary_covariate,
        RegimeMean({"always": 1, "never": 0}),
        ParametricGFormulaMethod(
            n_draws=2, covariate_learner=support.logistic("covariate_learner")
        ),
        _gformula,
    ),
    "eliminate": (
        lambda: _competing(eliminate=True),
        RegimeContrast(regimens={"always": 1, "never": 0}, reference="never"),
        None,
        lambda: _learners(competing_learner=support.logistic("competing_learner")),
    ),
    "mtp_density": (_dose_study, RegimeMean(_dose_regimens()), None, _learners),
    "mtp_classifier": (
        _dose_study,
        RegimeMean(_dose_regimens()),
        None,
        lambda: _learners(ratio="classifier"),
    ),
    "sdr_mtp_density": (_dose_study, RegimeMean(_dose_regimens()), "sdr", _learners),
}


def _study_route(
    name: str, mutate: Callable[[pytest.MonkeyPatch], None] | None = None
) -> support.Witness:
    factory, estimand, method, settings = STUDY_ROUTES[name]
    effect = factory().identify(estimand)

    def preview() -> PlannedFits:
        if mutate is None:
            return effect.planned_fits(method, **settings())
        with pytest.MonkeyPatch.context() as patch:
            mutate(patch)
            return effect.planned_fits(method, **settings())

    return support.witness(preview, lambda: effect.estimate(method, **settings()))


@pytest.mark.parametrize("name", sorted(STUDY_ROUTES))
def test_the_preview_counts_the_fits_of_each_longitudinal_route(name: str) -> None:
    found = _study_route(name)
    support.check(found, plain=PLAIN)
    if not any("bootstrap" in row.scope for row in found.plan.rows):
        # No pseudo-outcome of these fixtures turns constant, so every upper row is reached.
        for role in {row.role for row in found.plan.rows}:
            planned = sum(row.fits for row in found.plan.rows if row.role == role)
            assert support.realized_fits(found.counts, role) == planned, role


# ------------------------------------------------------------------ frequency bootstrap


def _frequency_exact(label: str) -> GLM:
    """A counting GLM that declares itself exact under frequency weights on its own class."""
    return type("GLM", (support.counting(GLM, label),), {"frequency_exact": True})()


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_the_frequency_weighted_bootstrap_refits_the_regressions_only(estimator: type) -> None:
    k, m = 3, 2
    frame = law_.draw(k, m, 1200, 11, law_.L_R)
    roles = {**law_.roles(k, m), **law_.probability_columns(k, m)}

    def model() -> Any:
        return estimator(
            plans_.regimens(("low", "high", "switch", "respond", "mix"), k, m),
            reference="low",
            g_bounds=(1e-8, 1.0),
            random_state=7,
            n_folds=1,
            n_bootstrap=3,
            outcome_learner=_frequency_exact("outcome_learner"),
            pseudo_learner=_frequency_exact("pseudo_learner"),
            simultaneous=False,
        )

    found = support.witness(
        lambda: model()._planned_fits(frame, **roles), lambda: model().fit(frame, **roles)
    )
    scopes = {row.scope for row in found.plan.rows}
    assert any(scope.startswith("frequency-weighted regression") for scope in scopes), scopes
    # The declared mechanism fits nothing, so only the regressions are priced.
    assert {row.role for row in found.plan.rows} == {"outcome_learner", "pseudo_learner"}
    support.check(
        found,
        plain={"outcome_learner": None, "pseudo_learner": None},
        outside=("tuner", "outcome_learner", "pseudo_learner"),
    )


# ------------------------------------------------------------------ tuned caps


@pytest.mark.parametrize("method", [None, "sdr"])
@pytest.mark.parametrize("target", ["population", "fold_average"])
def test_a_tuned_cap_is_priced_and_its_plan_checks_are_deferred(
    target: Any, method: str | None
) -> None:
    tuner: Any = (
        LinearQuantileRegression(("W",))
        if target == "population"
        else support.counting(GradientBoostingRegressor, "tuner")(
            loss="quantile", alpha=0.5, n_estimators=20
        )
    )
    cap = SupportBoundary(tuner, 0.5, features=() if target == "population" else ("W",))
    regimen = DynamicRegimen("t", (Shift(1.0, cap=cap, target=target), Shift(0.0, cap=None)))
    study = CausalStudy(
        lawq.sample(600, 11),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        ),
    )
    effect = study.identify(RegimeMean({"t": regimen, "natural": Shift(0.0, cap=None)}))

    def settings() -> dict[str, Any]:
        return _learners(
            outcome_learner=support.linear("outcome_learner"), density_bins=10, q_bounds=(0, 1)
        )

    found = support.witness(
        lambda: effect.planned_fits(method, **settings()),
        lambda: effect.estimate(method, **settings()),
    )
    support.check(found, plain=PLAIN | {"tuner": None})
    assert "fit_tuner" in found.plan.unpreviewed
    assert "_require_regimen_followers" in found.plan.unpreviewed
    tuner_rows = [row for row in found.plan.rows if row.role == "tuner"]
    # A linear quantile regression is a linear program, which fits no learner.
    assert bool(tuner_rows) == (target == "fold_average")


# ------------------------------------------------------------------ continuous time

_COX: dict[str, type] = {}
HORIZONS: Any = [0.5, 1.0]


def _cox(role: str) -> type:
    """A ``CoxHazard`` whose fits count under ``role:CoxHazard``.

    ``CoxHazard.label`` is a property the selection reads, so this is not
    :func:`support.counting`, which sets a ``label`` class attribute.
    """
    if role not in _COX:

        def fit(self: Any, X: Any, time: Any, event: Any) -> Any:
            support.COUNTS[f"{role}:CoxHazard"] += 1
            return CoxHazard.fit(self, X, time, event)

        _COX[role] = type("CoxHazard", (CoxHazard,), {"fit": fit})
    return _COX[role]


def _continuous_frame(causes: int, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    h1 = np.exp(-1.5 + 0.5 * w - a)
    h2 = np.exp(-2.0 - 0.5 * w - 0.5 * a) if causes == 2 else np.zeros(n)
    t = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    c = rng.exponential(1 / 0.15, n)
    return pd.DataFrame(
        {"time": np.minimum(t, c), "event": np.where(t <= c, cause, 0), "A": a, "W": w}
    )


@pytest.mark.parametrize(
    ("causes", "cross_fit", "user_learner"),
    [(2, True, False), (1, True, True), (2, False, False)],
)
def test_the_continuous_time_preview_counts_every_selection(
    causes: int, cross_fit: bool, user_learner: bool
) -> None:
    """``(J + 1)`` Cox selections of ``M * V + 1`` fits per fold, and the propensity."""
    order = iter(("hazard_library", "censoring_library") * 4)

    def library() -> tuple[CoxHazard, ...]:
        made = _cox(next(order))
        return tuple(made(terms) for terms in ("treatment", "main", "interaction"))

    def keep_type(learner: Any) -> Any:
        # The selection rebuilds a plain CoxHazard; the counting subclass must survive it.
        return type(learner)(learner.terms, learner.penalty)

    study = CausalStudy(
        _continuous_frame(causes),
        design=TimeToEvent(
            time="time", event="event", treatment="A", baseline=["W"], grid="continuous"
        ),
    )
    effect = study.identify(
        RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=HORIZONS)
    )
    settings: dict[str, Any] = {
        "random_state": 0,
        "cross_fit": cross_fit,
        "n_folds": 2 if cross_fit else 1,
        "learner_folds": 3,
    }
    if user_learner:
        settings["treatment_learner"] = support.logistic("treatment_learner")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine_module, "default_library", library)
        patch.setattr(cox_module, "_clone", keep_type)
        found = support.witness(
            lambda: effect.planned_fits(**settings), lambda: effect.estimate(**settings)
        )
    plan = found.plan
    assert plan.exact

    def planned(role: str) -> dict[str, int]:
        total: dict[str, int] = {}
        for row in plan.rows:
            if row.role == role:
                for name, value in dict(row.by_candidate).items():
                    total[name] = total.get(name, 0) + value
        return total

    folds = 2 if cross_fit else 1
    for role in ("hazard_library", "censoring_library"):
        realized = found.counts[f"{role}:CoxHazard"]
        assert planned(role) == {"CoxHazard": realized}, role
    assert planned("hazard_library")["CoxHazard"] == causes * folds * (3 * 3 + 1)
    # The default propensity is four logistic models in pipelines, which the class-wide
    # guard sees: 4 * V + 1 per fold.  A user learner is one fit per fold.
    treatment = (
        found.counts["treatment_learner"] if user_learner else sum(found.class_wide.values())
    )
    assert planned("treatment_learner") == {"LogisticRegression": treatment}
    assert treatment == folds * (1 if user_learner else 4 * 3 + 1)
    labelled = sum(v for label, v in found.counts.items() if label != "treatment_learner")
    assert plan.total_candidate_fits == labelled + treatment


# ------------------------------------------------------------------ mutation controls


def test_mutation_a_preview_that_skips_every_retention_node_fails() -> None:
    """The no-departure skip is load-bearing only where nobody left.

    Node 1 of this fixture retains every unit and node 2 does not.  Applying the skip at every
    node drops node 2's censoring fits, and the comparison fails.  (Removing the skip at node 1
    is masked: the counter's constant-fold skip finds the same one-valued target.)
    """
    frame, _ = make_longitudinal(n=300, seed=12)
    frame = frame[frame["C1"] == 1].reset_index(drop=True)
    effect = CausalStudy(frame, design=LongitudinalTreatment(**COLUMNS)).identify(
        RegimeMean({"always": 1, "never": 0})
    )
    honest = support.witness(
        lambda: effect.planned_fits(None, **_learners()),
        lambda: effect.estimate(None, **_learners()),
    )
    support.check(honest, plain=PLAIN)
    assert support.realized_fits(honest.counts, "censoring_learner") == 3

    def preview() -> PlannedFits:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sequential_module, "_retains_every_eligible_unit", lambda *a: True)
            return effect.planned_fits(None, **_learners())

    mutated = support.witness(preview, lambda: effect.estimate(None, **_learners()))
    with pytest.raises(AssertionError):
        support.check(mutated, plain=PLAIN)


def test_mutation_a_preview_without_the_sdr_training_fold_mechanism_fails() -> None:
    original = estimator_module._SequentialEstimator._plan_mechanism

    def first_pass_only(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("fit_rows") is not None:
            return None
        return original(self, *args, **kwargs)

    def mutate(patch: pytest.MonkeyPatch) -> None:
        patch.setattr(estimator_module._SequentialEstimator, "_plan_mechanism", first_pass_only)

    support.check(_study_route("sdr_cross_fitted"), plain=PLAIN)
    with pytest.raises(AssertionError):
        support.check(_study_route("sdr_cross_fitted", mutate), plain=PLAIN)
