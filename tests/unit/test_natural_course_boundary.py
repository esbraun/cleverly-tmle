"""The implementation boundary for ``ey_obs``, PAR and PAF with missing outcomes.

The scalar natural-course mean and the joint fit that stacks it with arm targets share
one resolver, so every refusal here runs for both.  ``REQUESTS`` holds the scalar request
and two joint ones.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CapabilityError, PositivityWarning
from cleverly.data import CausalData
from cleverly.estimators import CTMLE, DRTMLE
from tests.conftest import OracleMissingness, OracleOutcome, fast_tmle
from tests.unit._natural_course_support import (
    NeverFit,
    in_sample_tmle,
    never_fit_learners,
    stacked_tmle,
)


class _RealizedResponseSupport:
    """A law with safe realized response scores and impossible off-course cells."""

    @staticmethod
    def outcome_mean(w: Any, a: Any, z: Any = None) -> np.ndarray:
        levels = np.asarray(w, dtype=float).reshape(-1)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return 0.25 + 0.15 * arms + 0.10 * levels

    @staticmethod
    def missingness(w: Any, a: Any) -> np.ndarray:
        levels = np.rint(np.asarray(w, dtype=float).reshape(-1))
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return np.where(arms == levels, 0.8, 0.001)


def _frame(*, continuous: bool = False, multi_arm: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(310)
    n = 150
    w = rng.normal(size=n)
    a = rng.integers(0, 3, size=n) if multi_arm else rng.binomial(1, 0.5, size=n)
    mean = 0.2 + 0.15 * a + 0.1 * w
    y = (
        mean + rng.normal(scale=0.1, size=n)
        if continuous
        else rng.binomial(1, 1 / (1 + np.exp(-mean)))
    )
    y = np.asarray(y, dtype=float)
    delta = rng.binomial(1, 0.8, size=n)
    y[delta == 0] = np.nan
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W": w,
            "Delta": delta,
            "weight": np.linspace(0.5, 1.5, n),
            "id": np.arange(n) // 2,
            "stratum": (w > 0).astype(int),
            "Z": (w > -0.2).astype(int),
        }
    )


def _fit(frame: pd.DataFrame, **estimator_overrides: Any) -> Any:
    estimator = in_sample_tmle(**estimator_overrides)
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=("W",),
        delta="Delta",
    )


#: The scalar natural-course request and two joint requests. Every refusal below runs for
#: each, because one resolver spells all three.
REQUESTS = [
    pytest.param(("ey_obs",), id="ey_obs"),
    pytest.param(("par",), id="par"),
    pytest.param(("ey_obs", "ey0", "paf"), id="joint"),
]

#: The roles of a fit with a baseline stratum.
STRATA_ROLES = {"outcome": "Y", "treatment": "A", "covariates": ("W", "stratum"), "delta": "Delta"}

#: The subject every natural-course contract refusal opens with.
SUBJECT = (
    "NaturalCourseMean, PAR and PAF with missing outcomes currently support ordinary TMLE "
    "and, for point estimates, the per-arm outcome-adaptive CTMLE under their audited "
    "implementation contracts. "
)


def _conditions(estimator: Any, frame: pd.DataFrame, **roles: Any) -> list[str]:
    """The experimental conditions that the preflight collects, before any learner."""
    declared = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ("W",),
        "delta": "Delta",
        **roles,
    }
    data = CausalData.from_frame(frame, **declared)
    conditions = estimator._preflight_fit_configuration(data).conditions
    assert NeverFit.calls == 0
    return [condition.message() for condition in conditions]


@pytest.mark.parametrize("estimands", REQUESTS)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"fluctuation": "linear"},
            "No audited result covers a fluctuation other than the logistic one here. "
            "X60 in docs/roadmap.md tracks this. Set fluctuation='logistic'.",
        ),
        (
            {"targeting": "one_step"},
            "No audited result covers a targeting algorithm other than the iterative one "
            "here. X60 in docs/roadmap.md tracks this. Set targeting='iterative'.",
        ),
        (
            {"target_weights": True},
            "No audited result covers the weighted fluctuation here. X60 in docs/roadmap.md "
            "tracks this. Set target_weights=False.",
        ),
        (
            {"n_bootstrap": 2},
            "No audited bootstrap result covers this fit. X55 in docs/roadmap.md tracks the "
            "in-sample bootstrap with finite-dimensional nuisances, and F2 the bootstrap with "
            "data-adaptive or cross-fitted nuisances. Set n_bootstrap=0.",
        ),
    ],
)
def test_unsupported_method_settings_are_experimental_conditions(
    estimands: tuple[str, ...], overrides: dict[str, Any], message: str
) -> None:
    estimator = in_sample_tmle(**never_fit_learners(), **{"estimands": estimands, **overrides})
    assert SUBJECT + message in _conditions(estimator, _frame())
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        _fit(_frame(), **never_fit_learners(), **{"estimands": estimands, **overrides})
    assert NeverFit.calls > 0


def test_cross_fitting_with_stratified_folds_refuses_before_the_natural_course_contract() -> None:
    """Fold-strata enforcement for a cross-fitted fit moved out of this contract.

    ``stratify_folds != 'none'`` used to be one of this contract's own clauses, reached
    because the package default for it was ``'treatment'``. The default flipped to
    ``'none'`` (the fold and outcome-scale rules), and the check itself moved to the engine-wide
    fold-strata refusal that now fires at construction, before ``fit`` -- and this
    contract -- are ever reached, so ``cross_fit=True`` alone no longer breaks anything
    here. This declares the old default explicitly to still exercise the boundary.
    """
    with pytest.raises(ValueError, match="stratify_folds='none'"):
        _fit(_frame(), **never_fit_learners(), cross_fit=True, stratify_folds="treatment")
    assert NeverFit.calls == 0


def test_repeated_splits_refuse_at_the_existing_method_boundary() -> None:
    with pytest.raises(ValueError, match="cross_fit=False makes no split"):
        _fit(_frame(), repeats=2)


#: The estimator-variant refusal, after the subject.
VARIANT_REASON = (
    "The C-TMLE selector strategies, the shared outcome-adaptive design and DR-TMLE need "
    "separate targeting and inference results. X51 in docs/roadmap.md tracks the C-TMLE "
    "point estimates, and X50 the DR-TMLE route. "
    "Use ordinary TMLE, or CTMLE(strategy='oat', oat_design='per_arm') for point estimates."
)


@pytest.mark.parametrize("estimands", REQUESTS)
@pytest.mark.parametrize(
    ("build", "tail"),
    [
        pytest.param(
            lambda estimands: CTMLE(
                strategy="oat",
                oat_design="shared",
                estimands=estimands,
                cross_fit=False,
                **never_fit_learners(),
            ),
            "",
            id="ctmle-oat-shared",
        ),
        pytest.param(
            lambda estimands: DRTMLE(estimands=estimands, cross_fit=False, **never_fit_learners()),
            "",
            id="drtmle-guarded",
        ),
        pytest.param(
            lambda estimands: DRTMLE(
                estimands=estimands, guard=(), cross_fit=False, **never_fit_learners()
            ),
            " guard=() is the ordinary TMLE, so use TMLE.",
            id="drtmle-unguarded",
        ),
    ],
)
def test_estimator_variants_refuse_before_nuisance_fitting(
    estimands: tuple[str, ...], build: Any, tail: str
) -> None:
    estimator = build(estimands)
    with pytest.raises(CapabilityError) as caught:
        estimator.fit(
            _frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
    assert str(caught.value) == SUBJECT + VARIANT_REASON + tail
    assert NeverFit.calls == 0


@pytest.mark.parametrize(
    ("estimands", "ctmle_estimand"),
    [
        pytest.param(("ey_obs", "ey0", "ey1", "ate"), "ate", id="joint-ate"),
        pytest.param(("ey_obs", "ey0", "paf"), "ey0", id="joint-ey0"),
    ],
)
def test_a_selector_strategy_refuses_before_nuisance_fitting(
    estimands: tuple[str, ...], ctmle_estimand: str
) -> None:
    """The greedy selector reaches the method clause, with its selection estimand requested."""
    estimator = CTMLE(
        strategy="greedy",
        estimands=estimands,
        ctmle_estimand=ctmle_estimand,
        cross_fit=False,
        **never_fit_learners(),
    )
    with pytest.raises(CapabilityError) as caught:
        estimator.fit(_frame(), outcome="Y", treatment="A", covariates=("W",), delta="Delta")
    assert str(caught.value) == SUBJECT + VARIANT_REASON
    assert NeverFit.calls == 0


@pytest.mark.parametrize("estimands", REQUESTS)
def test_the_per_arm_design_runs_in_sample(estimands: tuple[str, ...]) -> None:
    """The per-arm outcome-adaptive CTMLE passes the method clause and reports points."""
    from sklearn.linear_model import LogisticRegression

    result = (
        CTMLE(
            strategy="oat",
            estimands=estimands,
            cross_fit=False,
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            random_state=0,
        )
        .fit(_frame(), outcome="Y", treatment="A", covariates=("W",), delta="Delta")
        .single()
    )
    assert result.inference_status == "generated_design_plugin"
    for name in estimands:
        assert np.isfinite(result[name].psi), name


#: The in-sample TMLE and the stacked CV-TMLE share every data-axis refusal. Each one
#: builds its learners from :func:`never_fit_learners`, so a refusal that let any learner
#: fit first fails on the assertion in the learner or on its call count.
ESTIMATORS = [
    pytest.param(in_sample_tmle, id="in-sample"),
    pytest.param(stacked_tmle, id="stacked"),
]


@pytest.mark.parametrize("estimands", REQUESTS)
@pytest.mark.parametrize("build", ESTIMATORS)
@pytest.mark.parametrize(
    ("fit_roles", "message"),
    [
        (
            {"intermediate": "Z"},
            "intermediate= is not implemented for these targets. X60 in docs/roadmap.md "
            "tracks this. Fit the natural-course target without intermediate=.",
        )
    ],
    ids=("intermediate",),
)
def test_unsupported_data_compositions_refuse_before_fitting(
    estimands: tuple[str, ...], build: Any, fit_roles: dict[str, Any], message: str
) -> None:
    estimator = build(**never_fit_learners(), estimands=estimands)
    covariates = ("W", "stratum") if "strata" in fit_roles else ("W",)
    with pytest.raises(CapabilityError) as caught:
        estimator.fit(
            _frame(),
            outcome="Y",
            treatment="A",
            covariates=covariates,
            delta="Delta",
            **fit_roles,
        )
    assert str(caught.value) == SUBJECT + message
    assert NeverFit.calls == 0


@pytest.mark.parametrize("estimands", REQUESTS)
def test_strata_fit_in_sample(estimands: tuple[str, ...]) -> None:
    """The in-sample fit targets one block per stratum, so it reaches a learner."""
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        in_sample_tmle(**never_fit_learners(), estimands=estimands).fit(
            _frame(), strata=("stratum",), **STRATA_ROLES
        )
    assert NeverFit.calls > 0


def test_the_stacked_natural_course_mean_fits_strata() -> None:
    """The stacked scalar fit solves one block per stratum on the out-of-fold predictions."""
    with pytest.raises(AssertionError, match="before any learner is fitted"):
        stacked_tmle(**never_fit_learners(), estimands=("ey_obs",)).fit(
            _frame(), strata=("stratum",), **STRATA_ROLES
        )
    assert NeverFit.calls > 0


@pytest.mark.parametrize("estimands", [request for request in REQUESTS if request.id != "ey_obs"])
def test_a_stacked_attributable_fit_meets_the_arm_indexed_strata_refusal(
    estimands: tuple[str, ...],
) -> None:
    """PAR, PAF and a joint fit read the arm-indexed contract, which labels strata."""
    estimator = stacked_tmle(**never_fit_learners(), estimands=estimands)
    texts = _conditions(
        estimator, _frame(), strata=("stratum",), **{**STRATA_ROLES, "covariates": ("W", "stratum")}
    )
    assert any(
        text.endswith(
            "No audited result covers baseline strata. X53 in docs/roadmap.md tracks this. "
            "Drop strata= from fit (PointTreatment(strata=()))."
        )
        for text in texts
    )


@pytest.mark.parametrize("estimands", REQUESTS)
@pytest.mark.parametrize(
    ("fit_roles", "message"),
    [
        (
            {"weights": "weight"},
            "The stacked contract covers unweighted iid rows. X60 in docs/roadmap.md tracks "
            "this. Drop weights= from fit, or fit in sample.",
        ),
        (
            {"id": "id"},
            "The stacked contract covers unweighted iid rows. X60 in docs/roadmap.md tracks "
            "this. Drop id= from fit, or fit in sample.",
        ),
    ],
    ids=("weighted", "clustered"),
)
def test_stacked_weights_and_clusters_are_experimental_conditions(
    estimands: tuple[str, ...], fit_roles: dict[str, Any], message: str
) -> None:
    estimator = stacked_tmle(**never_fit_learners(), estimands=estimands)
    assert SUBJECT + message in _conditions(estimator, _frame(), **fit_roles)


@pytest.mark.parametrize("estimands", REQUESTS)
@pytest.mark.parametrize(
    "fit_roles", [{"weights": "weight"}, {"id": "id"}], ids=("weighted", "clustered")
)
def test_in_sample_weights_and_clusters_are_admitted(
    estimands: tuple[str, ...], fit_roles: dict[str, Any]
) -> None:
    """A fixed weight defines a tilted law, and a cluster is the unit of the same curve."""
    frame = _frame()
    result = (
        in_sample_tmle(estimands=estimands)
        .fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta", **fit_roles)
        .single()
    )
    for name in estimands:
        assert np.isfinite(result.psi(name))
        assert np.isfinite(result[name].std_error)


@pytest.mark.parametrize(
    "estimands",
    [
        pytest.param(("ey_obs",), id="ey_obs"),
        pytest.param(("par",), id="par"),
        pytest.param(("ey_obs", "ey", "paf"), id="joint"),
    ],
)
@pytest.mark.parametrize("cross_fit", [False, True], ids=("in-sample", "stacked"))
def test_a_multi_arm_treatment_is_admitted(estimands: tuple[str, ...], cross_fit: bool) -> None:
    """The natural-course covariate is (A, W), so A may have three arms (decision D3)."""
    result = (
        fast_tmle(
            estimands=estimands,
            cross_fit=cross_fit,
            n_folds=2,
            stratify_folds="none",
            reference=0,
        )
        .fit(_frame(multi_arm=True), outcome="Y", treatment="A", covariates=("W",), delta="Delta")
        .single()
    )
    for estimate in result.estimates.values():
        assert np.isfinite(estimate.psi)
    if "par" in estimands:
        assert "par[0.0]" in result.estimates


@pytest.mark.parametrize("build", [in_sample_tmle, stacked_tmle], ids=("in-sample", "stacked"))
def test_a_continuous_outcome_without_q_bounds_refuses_before_fitting(build: Any) -> None:
    """The refusal comes after the stacked condition, which does not mask it."""
    estimator = build(**never_fit_learners())
    with pytest.raises(CapabilityError, match="analyst-declared q_bounds"):
        estimator.fit(
            _frame(continuous=True), outcome="Y", treatment="A", covariates=("W",), delta="Delta"
        )
    assert NeverFit.calls == 0


def test_a_stacked_continuous_outcome_with_q_bounds_is_an_experimental_condition() -> None:
    estimator = stacked_tmle(**never_fit_learners(), q_bounds=(-1.0, 1.0))
    texts = _conditions(estimator, _frame(continuous=True))
    assert any("requires a binary outcome" in text for text in texts)


def test_bounded_continuous_outcome_is_supported() -> None:
    result = _fit(_frame(continuous=True), q_bounds=(-1.0, 1.0)).single()
    assert np.isfinite(result.psi("ey_obs"))


def _response_frame(law: Any) -> pd.DataFrame:
    w = np.tile(np.array([0.0, 1.0]), 100)
    a = w.copy()
    delta = (np.arange(w.size) % 5 != 0).astype(float)
    mean = law.outcome_mean(w, a)
    y = (np.arange(w.size) % 3 < np.rint(3 * mean)).astype(float)
    return pd.DataFrame({"Y": np.where(delta == 1.0, y, np.nan), "A": a, "W": w, "Delta": delta})


def _fit_for_warnings(law: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fast_tmle(
            estimands=("ey_obs",),
            outcome_learner=OracleOutcome(law),
            missingness_learner=OracleMissingness(law),
            cross_fit=False,
            nuisance_bound=0.01,
        ).fit(_response_frame(law), outcome="Y", treatment="A", covariates=("W",), delta="Delta")
    return list(caught)


def test_response_warning_reads_only_the_realized_natural_course() -> None:
    """The negative half: dangerous *off-course* cells must not raise the warning.

    Paired with the positive half below, which uses the same frame and the mirrored
    law. One alone would pass on a construction that always warns or never does.
    """
    caught = _fit_for_warnings(_RealizedResponseSupport())
    assert not any(issubclass(item.category, PositivityWarning) for item in caught)


class _LowRealizedResponse:
    """The mirror of :class:`_RealizedResponseSupport`: the realized cells are the bad ones."""

    @staticmethod
    def outcome_mean(w: Any, a: Any, z: Any = None) -> np.ndarray:
        levels = np.asarray(w, dtype=float).reshape(-1)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return 0.25 + 0.15 * arms + 0.10 * levels

    @staticmethod
    def missingness(w: Any, a: Any) -> np.ndarray:
        levels = np.rint(np.asarray(w, dtype=float).reshape(-1))
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return np.where(arms == levels, 0.002, 0.9)


def test_a_low_realized_response_warns_with_the_natural_course_guidance() -> None:
    """The positive half of the realized-arm reduction.

    Its negative half above shows no warning when the *off-course* cells are the
    dangerous ones. Without this case the reduction would also pass if it silenced the
    warning outright, so the two together pin which column is read rather than only
    that some column is.
    """
    positivity = [
        item for item in _fit_for_warnings(_LowRealizedResponse()) if _is_positivity(item)
    ]
    assert positivity, "a response probability of 0.002 under a 0.01 bound must warn"
    message = str(positivity[0].message)
    assert "P(Delta = 1 | A, W)" in message
    # The natural-course sentence, not the arm-indexed one: this fit has no g(W) for
    # the clever covariate to divide by, so guidance naming g would be false.
    assert "natural-course response-residual covariate" in message
    assert "res.diagnostics.nuisance_models()" in message
    assert "just as g(W) does" not in message


def _is_positivity(item: warnings.WarningMessage) -> bool:
    return issubclass(item.category, PositivityWarning) and "P(Delta = 1" in str(item.message)


def test_ey_obs_beside_an_intermediate_refuses_on_a_complete_outcome() -> None:
    """The complete-outcome composition the natural-course boundary cannot see.

    ``_resolve_estimands_for_data`` returns at its first branch when no outcome is
    missing, so its ``intermediate=`` refusal never runs here. The guard that does run
    keyed on ``POPULATION_INTERVENTION_TARGETS``, which ``ey_obs`` left when its
    missing-outcome score equation landed. Without a replacement the fit succeeded and
    published the plain empirical mean of ``Y`` once per level of ``Z``, labelling each
    copy a controlled direct effect.
    """
    frame = _frame()
    frame = frame.assign(Y=frame["Y"].fillna(0.0), Z=(np.arange(len(frame)) % 2).astype(float))
    estimator = fast_tmle(estimands=("ey_obs",), cross_fit=False)
    with pytest.raises(CapabilityError, match=r"\['ey_obs'\] do not yet support intermediate="):
        estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W",),
            intermediate="Z",
        )


def test_all_estimands_beside_an_intermediate_drops_ey_obs_rather_than_refusing() -> None:
    """``estimands="all"`` asks for what the data supports, so it drops rather than fails.

    The refusal above must not reach this path. ``all`` named no target, so a
    composition it cannot express is a target to leave out, exactly as ``par`` and
    ``paf`` are left out. Refusing here would break a design that worked before
    ``ey_obs`` left ``POPULATION_INTERVENTION_TARGETS``, which is the shape of the
    regression the refusal above repairs.
    """
    frame = _frame()
    frame = frame.assign(Y=frame["Y"].fillna(0.0), Z=(np.arange(len(frame)) % 2).astype(float))
    result = (
        fast_tmle(estimands="all", cross_fit=False)
        .fit(frame, outcome="Y", treatment="A", covariates=("W",), intermediate="Z")
        .get(0.0)
    )
    assert result is not None
    reported = set(result.estimates)
    assert "ey_obs" not in reported
    assert not reported & {"par", "paf"}
    # It still reported the targets the composition does express, so the drop is a
    # narrowing and not a silent empty fit.
    assert {"ate", "ey1", "ey0"} <= reported


def test_the_summary_names_the_propensity_bound_only_when_a_propensity_was_fitted() -> None:
    """The witness for ``TMLEConfig.fits_treatment``.

    Deleting the guard leaves every other test green, because the line it suppresses
    is only wrong on the one fit that estimates no treatment mechanism. Both halves
    are asserted here: an ordinary fit must keep the line, and this one must not,
    so neither a missing guard nor an always-on one passes.
    """
    missing = _fit(_frame()).single()
    assert not missing.config.fits_treatment
    settings = missing.summary(detail="full").section("Settings").lines
    assert not any(line.startswith("Estimated treatment probabilities") for line in settings)
    # It still names the mechanism it did bound, so the suppression is narrow.
    assert any("that the outcome is recorded raised to" in line for line in settings)

    complete = _frame()
    complete = complete.assign(Y=complete["Y"].fillna(0.0))
    ordinary = (
        fast_tmle(cross_fit=False)
        .fit(complete, outcome="Y", treatment="A", covariates=("W",))
        .single()
    )
    assert ordinary.config.fits_treatment
    settings = ordinary.summary(detail="full").section("Settings").lines
    assert any(line.startswith("Estimated treatment probabilities") for line in settings)
