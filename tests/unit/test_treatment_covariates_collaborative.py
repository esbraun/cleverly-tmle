"""Treatment-mechanism covariates (part (h)) beside C-TMLE, DR-TMLE and the sensitivity bound.

``tests/unit/test_treatment_covariates.py`` pins the design, the data field, the plain TMLE and
the exact-law witnesses.  This file pins the compositions:

* The C-TMLE candidate pool, the preorder, the explicit ordering and the discrete candidates
  all read ``Wg``, and the outcome regression keeps the full set.
* The X16 strategies ``"lasso"`` and ``"learners"`` read ``Wg`` too: the penalty path, its
  default grid and the declared learners.
* ``declares_no_selection`` compares a ``"discrete"`` candidate with ``Wg``, so a fit
  whose one candidate is ``Wg`` selects nothing and takes the TMLE status.  A ``"learners"``
  fit with an empty list is admitted the same way and carries the restriction text.
* The selection summary names the pool, and every C-TMLE fit on a restricted design prints the
  pinned restriction text.
* DR-TMLE prints its pinned pointer to the one-wrong table of the DR-TMLE theorem page.
* The omitted-variable bound admits a plain TMLE fit with ``Wg`` and reads the restriction.
* ``strategy="oat"`` and candidate or ordering names outside ``Wg`` are refused (R-h2, R-h5).
* ``CausalData.with_extra_covariate`` adds the new column to ``Wg``.
"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import ATE, CausalStudy, CollaborativeTMLEMethod, PointTreatment
from cleverly._summary import vocabulary
from cleverly.data import CausalData
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.ctmle import CTMLESelection, declares_no_selection
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.sensitivity import omitted_variable_bounds
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit._source_mutation import mutate
from tests.unit.test_treatment_covariates import (
    COVARIATES,
    WG,
    Spy,
    SpyClassifier,
    SpyRegressor,
    make_frame,
    pinned,
)

ctmle_module = importlib.import_module("cleverly.estimators.ctmle")

R_H2 = (
    "strategy='oat' fits the treatment mechanism on the outcome predictions, so "
    "treatment_adjustment= has no covariates to restrict. Use another strategy, or omit "
    "treatment_adjustment="
)

#: The plan's DR-TMLE line (plan I-2.6.3) in plain words, with ``{names}`` still to fill.
DRTMLE_PINNED = (
    "The treatment model reads only {names}, as the design declares. If treatment depends "
    "on other covariates too, the treatment model is wrong, and no result shows that the "
    "interval keeps its coverage."
)


def restrictions(result: Any) -> list[str]:
    """The model restrictions a full summary lists under its assumptions."""
    lead = "- model restriction: "
    lines = result.summary(detail="full").section("Assumptions").lines
    return [line[len(lead) :] for line in lines if line.startswith(lead)]


def method_lines(result: Any) -> tuple[str, ...]:
    """The Method details lines of a full summary, or none."""
    summary = result.summary(detail="full")
    if "Method details" not in [section.title for section in summary.sections]:
        return ()
    return summary.section("Method details").lines


def collaborative(**overrides: Any) -> CTMLE:
    """A C-TMLE whose learners are spies, fitted in sample on two selection folds."""
    settings: dict[str, Any] = {
        "outcome_learner": SpyRegressor("q"),
        "treatment_learner": SpyClassifier("g"),
        "missingness_learner": SpyClassifier("missing"),
        "strategy": "greedy",
        "cross_fit": False,
        "selection_folds": 2,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return CTMLE(**settings)


def fit(estimator: Any, wg: list[str] | None = WG, **kwargs: Any) -> Any:
    return estimator.fit(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=wg,
        **kwargs,
    ).single()


# ------------------------------------------------------------------------ the candidate pool

STRATEGIES: list[dict[str, Any]] = [
    {"strategy": "greedy"},
    {"strategy": "ordered", "preorder": "logistic"},
    {"strategy": "ordered", "preorder": "partial_correlation"},
    {"strategy": "ordered", "ordering": ["c3", "c2"]},
    {"strategy": "discrete", "candidates": [[], ["c2"], ["c2", "c3"]]},
]


@pytest.mark.parametrize("settings", STRATEGIES, ids=lambda s: s.get("preorder", s["strategy"]))
def test_the_candidate_pool_reads_wg(settings: dict[str, Any]) -> None:
    Spy.reset()
    result = fit(collaborative(**settings))
    roles = Spy.roles()
    assert roles["g"] <= {2, 3}
    assert roles["g"] >= {2}
    assert roles["q"] == {0, 1, 2, 3, 4}
    selection = result.extra["ctmle"]
    assert isinstance(selection, CTMLESelection)
    assert selection.covariates == ("c2", "c3")
    assert all(set(names) <= {"c2", "c3"} for names in selection.path)
    assert set(result.nuisance.treatment_covariates) <= {"c2", "c3"}


def test_the_default_pool_is_the_whole_set() -> None:
    Spy.reset()
    result = fit(collaborative(), wg=None)
    selection = result.extra["ctmle"]
    assert selection.covariates == tuple(COVARIATES)
    assert Spy.roles()["g"] == {1, 2, 3, 4}
    searched = selection.summary().facts[vocabulary.COVARIATES_SEARCHED]
    assert searched == ", ".join(COVARIATES)
    assert "declared" not in searched


def test_a_greedy_path_ends_on_wg_and_never_on_the_full_set() -> None:
    result = fit(collaborative())
    selection = result.extra["ctmle"]
    assert [len(names) for names in selection.path] == [0, 1, 2]


def test_mutation_a_pool_of_the_full_set_fails_the_pool_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def witness() -> None:
        Spy.reset()
        fit(collaborative())
        assert Spy.roles()["g"] <= {2, 3}

    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        ctmle_module._Selector,
        "_forward_path",
        [("list(self.data.mechanism_covariate_names)", "list(self.data.covariate_names)")],
    )
    with pytest.raises(AssertionError):
        witness()


def test_mutation_an_ordering_over_the_full_set_fails_the_pool_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def witness() -> None:
        Spy.reset()
        fit(collaborative(strategy="ordered", preorder="logistic"))
        assert Spy.roles()["g"] <= {2, 3}

    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        ctmle_module._Selector,
        "_ordering",
        [
            ("for name in pool:", "for name in self.data.covariate_names:"),
            (
                "return tuple(pool[j] for j in order)",
                "return tuple(self.data.covariate_names[j] for j in order)",
            ),
        ],
    )
    with pytest.raises(AssertionError):
        witness()


def test_the_partial_correlation_order_ranks_wg_only() -> None:
    Spy.reset()
    result = fit(collaborative(strategy="ordered", preorder="partial_correlation"))
    selection = result.extra["ctmle"]
    assert set(selection.path[-1]) == {"c2", "c3"}
    assert Spy.roles()["g"] <= {2, 3}


# --------------------------------------------------------------------------- the refusals


def test_r_h5_candidates_outside_wg() -> None:
    learners = never_fit_learners()
    estimator = CTMLE(
        strategy="discrete",
        candidates=[["c2"], ["c2", "c1"]],
        cross_fit=False,
        estimands=("ate",),
        **learners,
    )
    raised = assert_refused_before_any_call(
        lambda: fit(estimator),
        None,
        "learner",
        "candidates names ['c1'] outside treatment_adjustment=",
        error=ValueError,
    )
    assert NeverFit.calls == 0
    assert not isinstance(raised, CapabilityError)


def test_r_h5_ordering_outside_wg() -> None:
    learners = never_fit_learners()
    estimator = CTMLE(
        strategy="ordered",
        ordering=["c2", "c3", "c4"],
        cross_fit=False,
        estimands=("ate",),
        **learners,
    )
    assert_refused_before_any_call(
        lambda: fit(estimator),
        None,
        "learner",
        "ordering names ['c4'] outside treatment_adjustment=",
        error=ValueError,
    )


def test_an_ordering_inside_wg_must_cover_wg_and_not_the_full_set() -> None:
    Spy.reset()
    fit(collaborative(strategy="ordered", ordering=["c3", "c2"]))
    with pytest.raises(ValueError, match="ordering must cover every covariate"):
        fit(collaborative(strategy="ordered", ordering=["c2"]))


def test_r_h5_does_not_change_the_default_refusals() -> None:
    with pytest.raises(ValueError, match="ordering names unknown covariate"):
        fit(collaborative(strategy="ordered", ordering=["c1", "c2", "zz"]), wg=None)


def test_r_h2_oat_beside_wg() -> None:
    learners = never_fit_learners()
    raised = assert_refused_before_any_call(
        lambda: fit(CTMLE(strategy="oat", cross_fit=False, estimands=("ate",), **learners)),
        None,
        "learner",
        R_H2,
        error=CapabilityError,
    )
    assert isinstance(raised, CapabilityError)
    assert raised.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_oat_without_wg_is_not_refused_by_r_h2() -> None:
    estimator = CTMLE(strategy="oat", cross_fit=False, estimands=("ate",), **never_fit_learners())
    with pytest.raises(AssertionError):
        # The fit reaches the first learner, which is the failure a refusal must precede.
        fit(estimator, wg=None)
    assert NeverFit.calls > 0


# ------------------------------------------------------------- the X16 strategies

LASSO_GRID = (0.2, 0.05, 0.02, 0.005)


def lasso_frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Four covariates. ``c1`` drives the treatment strongly and lies outside ``Wg = {c2, c3}``."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 4))
    logit = 1.6 * c[:, 0] + 0.8 * c[:, 1]
    frame = pd.DataFrame(c, columns=COVARIATES)
    frame["A"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    frame["Y"] = c[:, 0] + 0.5 * frame["A"] + 0.3 * c[:, 2] + rng.normal(0, 0.5, n)
    return frame


def x16_fit(covariates: list[str], wg: list[str] | None, **settings: Any) -> Any:
    """A C-TMLE whose outcome regression ignores the covariates, so only ``g`` can differ."""
    options: dict[str, Any] = {
        "outcome_learner": DummyRegressor(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate",),
        "selection_folds": 3,
        "q_bounds": (-20.0, 20.0),
    }
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**options)
            .fit(
                lasso_frame(),
                outcome="Y",
                treatment="A",
                covariates=covariates,
                treatment_covariates=wg,
            )
            .single()
        )


X16_STRATEGIES: list[dict[str, Any]] = [
    {"strategy": "lasso", "lambdas": LASSO_GRID},
    {
        "strategy": "learners",
        "treatment_candidates": [LogisticRegression(C=0.001), LogisticRegression(C=0.05)],
    },
    {"strategy": "learners", "treatment_candidates": []},
]


@pytest.mark.parametrize("settings", X16_STRATEGIES, ids=lambda s: str(s["strategy"]))
def test_an_x16_fit_on_wg_equals_the_fit_that_omits_the_other_covariates(
    settings: dict[str, Any],
) -> None:
    """The outcome regression ignores the covariates, so only the pool of ``g`` can differ."""
    restricted = x16_fit(COVARIATES, ["c2", "c3"], **settings)
    omitted = x16_fit(["c2", "c3"], None, **settings)
    full = x16_fit(COVARIATES, None, **settings)
    np.testing.assert_allclose(
        restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
    )
    assert restricted.estimates["ate"].psi == pytest.approx(omitted.estimates["ate"].psi)
    selection = restricted.extra["ctmle"]
    assert selection.covariates == ("c2", "c3")
    assert selection.restricted_pool == ("c2", "c3")
    assert set(selection.selected_covariates) <= {"c2", "c3"}
    assert all(set(names) <= {"c2", "c3"} for names in selection.path)
    # The control: c1 drives the treatment, so a pool that reads it moves the mechanism.
    assert not np.allclose(
        restricted.nuisance.propensity.values, full.nuisance.propensity.values, atol=1e-3
    )


@pytest.mark.parametrize("settings", X16_STRATEGIES[:2], ids=lambda s: str(s["strategy"]))
def test_an_x16_selection_summary_names_the_pool(settings: dict[str, Any]) -> None:
    summary = x16_fit(COVARIATES, ["c2", "c3"], **settings).extra["ctmle"].summary()
    assert summary.facts[vocabulary.COVARIATES_SEARCHED] == "c2, c3 (declared)"


def test_the_lasso_default_grid_reads_wg() -> None:
    restricted = x16_fit(COVARIATES, ["c2", "c3"], strategy="lasso")
    omitted = x16_fit(["c2", "c3"], None, strategy="lasso")
    assert restricted.extra["ctmle"].covariates == ("c2", "c3")
    np.testing.assert_allclose(
        restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
    )
    assert restricted.estimates["ate"].psi == pytest.approx(omitted.estimates["ate"].psi)


def test_mutation_a_lasso_path_on_the_full_set_fails_the_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = X16_STRATEGIES[0]

    def witness() -> None:
        restricted = x16_fit(COVARIATES, ["c2", "c3"], **settings)
        omitted = x16_fit(["c2", "c3"], None, **settings)
        np.testing.assert_allclose(
            restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
        )

    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        ctmle_module._Selector,
        "_lasso_propensities",
        [("design = data.mechanism_covariates", "design = data.covariates")],
    )
    with pytest.raises((AssertionError, ValueError, IndexError)):
        witness()


def test_mutation_a_learners_pool_on_the_full_set_fails_the_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = X16_STRATEGIES[1]

    def witness() -> None:
        restricted = x16_fit(COVARIATES, ["c2", "c3"], **settings)
        omitted = x16_fit(["c2", "c3"], None, **settings)
        np.testing.assert_allclose(
            restricted.nuisance.propensity.values, omitted.nuisance.propensity.values, atol=1e-12
        )

    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        ctmle_module._Selector,
        "indexed_propensities",
        [
            (
                "names = tuple(self.data.mechanism_covariate_names)",
                "names = tuple(self.data.covariate_names)",
            )
        ],
    )
    with pytest.raises(AssertionError):
        witness()


def test_the_learners_spy_sees_wg_only() -> None:
    Spy.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(
            collaborative(
                strategy="learners",
                treatment_candidates=[SpyClassifier("g")],
            )
        )
    assert Spy.roles()["g"] == {2, 3}


def test_an_empty_learners_fit_is_admitted_and_carries_the_restriction() -> None:
    result = x16_fit(COVARIATES, ["c2", "c3"], strategy="learners", treatment_candidates=[])
    tmle = fit_tmle_like()
    assert result.inference_status == tmle.inference_status
    assert pinned("{c2, c3}") in restrictions(result)
    nonempty = x16_fit(
        COVARIATES,
        ["c2", "c3"],
        strategy="learners",
        treatment_candidates=[LogisticRegression(C=0.001)],
    )
    assert nonempty.inference_status == "working_mechanism_plugin"
    assert pinned("{c2, c3}") in restrictions(nonempty)


def fit_tmle_like() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=DummyRegressor(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
                estimands=("ate",),
                q_bounds=(-20.0, 20.0),
            )
            .fit(
                lasso_frame(),
                outcome="Y",
                treatment="A",
                covariates=COVARIATES,
                treatment_covariates=["c2", "c3"],
            )
            .single()
        )


def test_an_empty_learners_fit_equals_the_tmle_fit_on_wg() -> None:
    result = x16_fit(COVARIATES, ["c2", "c3"], strategy="learners", treatment_candidates=[])
    tmle = fit_tmle_like()
    assert result.estimates["ate"].psi == pytest.approx(tmle.estimates["ate"].psi)


def test_every_x16_fit_prints_the_restriction() -> None:
    for settings in X16_STRATEGIES:
        result = x16_fit(COVARIATES, ["c2", "c3"], **settings)
        assert pinned("{c2, c3}") in restrictions(result)


def test_a_declared_empty_wg_gives_an_intercept_pool_to_the_x16_strategies() -> None:
    for settings in X16_STRATEGIES[:2]:
        result = x16_fit(COVARIATES, [], **settings)
        selection = result.extra["ctmle"]
        assert selection.covariates == ()
        assert selection.selected_covariates == ()


# ----------------------------------------------------------------------------- the status


def test_declares_no_selection_compares_with_the_pool() -> None:
    assert declares_no_selection("discrete", [["c2", "c3"]], None, ("c2", "c3"))
    assert not declares_no_selection("discrete", [["c2", "c3"]], None, tuple(COVARIATES))
    assert declares_no_selection("learners", None, [], ("c2", "c3"))


def test_a_discrete_candidate_equal_to_wg_selects_nothing_and_takes_the_tmle_status() -> None:
    tmle = fit(TMLE(**_learners(), estimands=("ate",), cross_fit=False, simultaneous=False))
    admitted = fit(collaborative(strategy="discrete", candidates=[["c2", "c3"]]))
    assert admitted.inference_status == tmle.inference_status
    selecting = fit(collaborative(strategy="discrete", candidates=[[], ["c2", "c3"]]))
    assert selecting.inference_status == "working_mechanism_plugin"
    assert admitted.inference_status != selecting.inference_status


def test_a_discrete_candidate_equal_to_the_full_set_is_refused_beside_wg() -> None:
    with pytest.raises(ValueError, match="outside treatment_adjustment="):
        fit(collaborative(strategy="discrete", candidates=[COVARIATES]))


def test_the_status_of_a_full_candidate_is_unchanged_without_wg() -> None:
    admitted = fit(collaborative(strategy="discrete", candidates=[COVARIATES]), wg=None)
    assert admitted.inference_status != "working_mechanism_plugin"


# ------------------------------------------------------------------------------ the text


def test_the_selection_summary_names_the_pool() -> None:
    result = fit(collaborative())
    selection = result.extra["ctmle"]
    assert selection.summary().facts[vocabulary.COVARIATES_SEARCHED] == "c2, c3 (declared)"
    assert vocabulary.SEARCH_POOL.format(names="c2, c3") in method_lines(result)
    unrestricted = fit(collaborative(), wg=None)
    assert not any("drew only from" in line for line in method_lines(unrestricted))


def test_every_collaborative_fit_prints_the_restriction() -> None:
    for settings in (
        {"strategy": "greedy"},
        {"strategy": "discrete", "candidates": [["c2", "c3"]]},
        {"strategy": "discrete", "candidates": [[], ["c2", "c3"]]},
    ):
        result = fit(collaborative(**settings))
        assert pinned("{c2, c3}") in restrictions(result)


def test_a_collaborative_study_prints_the_restriction_of_its_design() -> None:
    effect = CausalStudy(
        make_frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=tuple(COVARIATES),
            treatment_adjustment=("c2", "c3"),
        ),
    ).identify(ATE())
    result = effect.estimate(
        CollaborativeTMLEMethod(strategy="discrete", candidates=(("c2", "c3"),), selection_folds=2),
        outcome_learner=SpyRegressor("q"),
        treatment_learner=SpyClassifier("g"),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert pinned("{c2, c3}") in restrictions(result)


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": SpyRegressor("q"),
        "treatment_learner": SpyClassifier("g"),
        "random_state": 0,
    }


def drtmle(**overrides: Any) -> DRTMLE:
    settings: dict[str, Any] = {
        "outcome_learner": SpyRegressor("q"),
        "treatment_learner": SpyClassifier("g"),
        "reduced_outcome_learner": SpyRegressor("rq"),
        "reduced_treatment_learner": SpyClassifier("rg"),
        "cross_fit": False,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
        "max_outer": 2,
    }
    settings.update(overrides)
    return DRTMLE(**settings)


def test_the_drtmle_line_is_pinned() -> None:
    assert vocabulary.DRTMLE_RESTRICTED == DRTMLE_PINNED
    result = fit(drtmle())
    expected = DRTMLE_PINNED.format(names="c2, c3")
    assert expected in method_lines(result)
    assert pinned("{c2, c3}") in restrictions(result)


def test_the_drtmle_line_names_only_a_drtmle_fit_with_wg() -> None:
    marker = "If treatment depends on other covariates too"
    assert any(marker in line for line in method_lines(fit(drtmle())))
    assert not any(marker in line for line in method_lines(fit(drtmle(), wg=None)))
    tmle = fit(TMLE(**_learners(), estimands=("ate",), cross_fit=False, simultaneous=False))
    assert marker not in tmle.summary(detail="full")
    assert marker not in fit(collaborative()).summary(detail="full")


def test_the_theorem_page_has_the_one_wrong_table_behind_the_line() -> None:
    from pathlib import Path

    page = Path(__file__).resolve().parents[2] / "docs/technical-reference/dr-tmle/theorem.md"
    assert page.exists()
    assert re.search(r"one-wrong|one wrong", page.read_text(encoding="utf-8"), re.IGNORECASE)


# ------------------------------------------------------------- the omitted-variable bound


def test_the_omitted_variable_bound_reads_the_restriction_of_a_plain_tmle_fit() -> None:
    result = fit(
        TMLE(
            outcome_learner=SpyRegressor("q"),
            treatment_learner=SpyClassifier("g"),
            estimands=("ate",),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
    )
    bounds = omitted_variable_bounds(result, "ate")
    assert np.isfinite(bounds.max_bias)
    # The bound is admitted, and the fit it reads states the restriction it needs.
    assert pinned("{c2, c3}") in restrictions(result)
    doc = omitted_variable_bounds.__doc__ or ""
    assert "treatment_adjustment" in doc
    assert "restriction" in doc


def test_the_omitted_variable_bound_still_refuses_a_collaborative_fit_with_wg() -> None:
    result = fit(collaborative(strategy="discrete", candidates=[["c2", "c3"]]))
    with pytest.raises(CapabilityError):
        omitted_variable_bounds(result, "ate")


# ------------------------------------------------------------------------ the data copies


def test_with_extra_covariate_adds_the_column_to_wg() -> None:
    data = CausalData.from_frame(
        make_frame(),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        treatment_covariates=WG,
    )
    extra = data.with_extra_covariate(np.zeros(data.n), "_noise_0")
    assert extra.treatment_covariate_names == ("c2", "c3", "_noise_0")
    assert extra.mechanism_covariates.shape == (data.n, 3)
    assert extra.covariates.shape == (data.n, 5)
    assert extra.added_covariates == ("_noise_0",)
    plain = CausalData.from_frame(
        make_frame(), outcome="Y", treatment="A", covariates=COVARIATES
    ).with_extra_covariate(np.zeros(data.n), "_noise_0")
    assert plain.treatment_covariate_names is None


def test_a_refit_with_an_extra_covariate_keeps_the_restriction() -> None:
    result = fit(
        TMLE(
            outcome_learner=SpyRegressor("q"),
            treatment_learner=SpyClassifier("g"),
            estimands=("ate",),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
    )
    Spy.reset()
    noisy = result.data.with_extra_covariate(np.random.default_rng(0).normal(size=240), "_noise_0")
    result.estimator.refit(noisy, intermediate_value=None, random_state=0)
    # Wg and the new column, and none of c1 or c4.
    assert Spy.roles()["g"] == {0, 2, 3}
    assert Spy.roles()["q"] == {0, 1, 2, 3, 4}


def test_a_collaborative_refit_extends_the_pool_with_the_added_column() -> None:
    estimator = collaborative(strategy="ordered", ordering=["c3", "c2"])
    result = fit(estimator)
    noisy = result.data.with_extra_covariate(np.random.default_rng(0).normal(size=240), "_noise_0")
    Spy.reset()
    refit = result.estimator.refit(noisy, intermediate_value=None, random_state=0)
    assert Spy.roles()["g"] <= {0, 2, 3}
    assert set(refit.extra["ctmle"].covariates) == {"c2", "c3", "_noise_0"}
