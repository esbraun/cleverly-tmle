"""The parity comparisons of the option fixtures: the R side's JSON against ``cleverly``.

Each ``compare_*`` function reads one fixture payload (the schema of the R script that writes
it), repeats the R run with ``cleverly`` on the exported arrays, and returns the findings.  An empty
list is parity.  The functions use only public ``cleverly`` calls, except that the stopping rule
thresholds come from :mod:`cleverly.fluctuation._score`, whose functions are the single place the
thresholds are computed.

Tolerances (targeting-options plan, section I-3.3):

==================  =====================================================  =============
fixture             quantity                                               tolerance
==================  =====================================================  =============
tmle3_updater       step counts, best step                                 equal
tmle3_updater       cumulative epsilon, final estimates                    1e-6
tmle3_updater       thresholds (relative), final ED                        1e-9, 1e-8
drtmle_stratify     outcome regression ``Qn``, propensity                  1e-8
drtmle_stratify     TMLE estimate                                          1e-6
ctmle_*             selected index                                         equal
ctmle_*             cross-validated score (relative), estimate             1e-6
==================  =====================================================  =============

The ``ctmle`` rows use the X16 fixture tolerance ``GLM_TOLERANCE = 1e-6``
(``tests/canonical/ctmle_sequence_witness/harness.py``).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly.estimators import CTMLE, DRTMLE
from cleverly.fluctuation import InitialFit, mean_submodel, solve_fluctuation
from cleverly.fluctuation._score import (
    StopContext,
    rule_contributions,
    rule_thresholds,
    score_columns,
)
from cleverly.learners import GLM

__all__ = [
    "COMPARATORS",
    "CTMLE_ESTIMATE",
    "CTMLE_INITIAL",
    "CTMLE_RTOL",
    "DRTMLE_ESTIMATE",
    "DRTMLE_QN",
    "TMLE3_ED",
    "TMLE3_EPSILON",
    "TMLE3_ESTIMATE",
    "TMLE3_THRESHOLD_RTOL",
    "compare_ctmle_patience",
    "compare_ctmle_wg",
    "compare_drtmle_stratify",
    "compare_tmle3_updater",
    "ctmle_fit",
]

TMLE3_EPSILON = 1e-6
TMLE3_ESTIMATE = 1e-6
TMLE3_ED = 1e-8
TMLE3_THRESHOLD_RTOL = 1e-9
#: The replay of the walk from its epsilons must reproduce the targeted prediction.
TMLE3_REPLAY = 1e-8
DRTMLE_QN = 1e-8
DRTMLE_ESTIMATE = 1e-6
CTMLE_RTOL = 1e-6
CTMLE_ESTIMATE = 1e-6
#: The initial GLM of the outcome regression: both sides solve the same score equations.
CTMLE_INITIAL = 1e-8

Payload = Mapping[str, Any]


def _vector(values: Any) -> np.ndarray:
    """A JSON number or array as a float vector; ``null`` becomes NaN."""
    return np.atleast_1d(
        np.asarray([np.nan if v is None else v for v in np.atleast_1d(values)], float)
    )


def _gap(found: np.ndarray, expected: np.ndarray) -> float:
    """The largest absolute difference, or infinity when the shapes differ."""
    if found.shape != expected.shape:
        return float("inf")
    return float(np.max(np.abs(found - expected))) if found.size else 0.0


# ---------------------------------------------------------------------------- tmle3_updater


def tmle3_arrays(payload: Payload) -> tuple[np.ndarray, InitialFit, Any]:
    """The outcome, the initial fit and the per-arm submodel of the ``tmle3`` sample."""
    data, initial = payload["data"], payload["initial"]
    y, a = _vector(data["Y"]), _vector(data["A"])
    p_observed = _vector(initial["propensity_observed"])
    propensity = np.column_stack(
        [
            np.where(a == 0.0, p_observed, 1.0 - p_observed),
            np.where(a == 1.0, p_observed, 1.0 - p_observed),
        ]
    )
    fit = InitialFit(
        _vector(initial["q_observed"]),
        {0.0: _vector(initial["q_arm0"]), 1.0: _vector(initial["q_arm1"])},
    )
    return y, fit, mean_submodel(a, propensity)


def _cumulative_epsilon(
    steps: Sequence[np.ndarray], eds: Sequence[np.ndarray], ed0: np.ndarray, upto: int
) -> np.ndarray:
    """The epsilon of the collapsed covariate walk after ``upto`` steps, in the columns of ``H``.

    Step ``k`` moves by ``t_k * ED_{k-1} / ||ED_{k-1}||`` (``tmle3_Update$collapse_covariates``),
    with ``ED_0`` the empirical mean of the influence curve before any step.
    """
    total = np.zeros_like(ed0, dtype=float)
    previous = ed0
    for k in range(upto):
        total = total + float(steps[k][0]) * previous / np.linalg.norm(previous)
        previous = eds[k]
    return total


def compare_tmle3_updater(payload: Payload) -> list[str]:
    """Repeat each ``tmle3_Update`` run of the payload with :func:`solve_fluctuation`.

    A constrained updater is a walk of bounded steps, and the comparison is step for step: the
    step count, the reported iterate, the cumulative epsilon and the final estimates.  An
    unconstrained updater takes one full maximum-likelihood step, which :func:`solve_fluctuation`
    reaches by Newton steps without a declared rule, so the comparison is the epsilon and the
    estimates of that solution.  Every updater is also checked at its final fit: the score, the
    thresholds of the rule, and the bound that ``tmle3`` applies at each step.

    Parameters
    ----------
    payload : mapping
        The parsed ``tmle3_updater.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    y, fit, submodel = tmle3_arrays(payload)
    ones = np.ones(y.shape[0])
    ed0 = _vector(payload["initial"]["ed"])
    findings: list[str] = []
    for result in payload["updaters"]:
        name = result["name"]
        if "error" in result:
            findings.append(f"{name}: R raised an error: {result['error']}")
            continue
        findings.extend(
            f"{name}: {text}" for text in _compare_updater(result, y, fit, submodel, ones, ed0)
        )
    if len(payload["updaters"]) == 0:
        findings.append("the payload holds no updater")
    return findings


def _compare_updater(
    result: Payload,
    y: np.ndarray,
    fit: InitialFit,
    submodel: Any,
    ones: np.ndarray,
    ed0: np.ndarray,
) -> list[str]:
    options = result["options"]
    rule = options["convergence_type"]
    constrained = bool(options["constrain_step"])
    use_best = bool(options["use_best"])
    steps = int(result["steps"])
    epsilons = [_vector(e) for e in result["epsilons"]]
    eds = [_vector(e) for e in result["eds"]]
    final = result["final"]
    findings: list[str] = []
    if len(epsilons) != steps or len(eds) != steps or steps < 1:
        return [f"the payload lists {len(epsilons)} epsilons and {len(eds)} EDs for {steps} steps"]

    # The R side: the bound never binds, the replay reproduces the targeted prediction, and the
    # best step is the first smallest ED norm.
    if result["replay"]["bound_binds"]:
        findings.append(
            "bound(., 0.005) bound along the walk, so the unbounded walk is not the one compared"
        )
    replay_gap = result["replay"]["max_abs_difference"]
    if replay_gap is None or not replay_gap <= TMLE3_REPLAY:
        findings.append(
            f"the replay of the walk differs from the targeted prediction by {replay_gap}"
        )
    norms = [float(np.linalg.norm(ed)) for ed in eds]
    if use_best:
        expected_best = 1 + int(np.argmin(norms))
        if result["best_step"] != expected_best:
            findings.append(
                f"best_step {result['best_step']} is not the first smallest ED norm ({expected_best})"
            )
        reported = expected_best
    else:
        reported = steps

    # The final fit: score, estimates and thresholds, computed from the R arrays alone.
    mask = np.ones(y.shape[0], dtype=bool)
    q_observed = _vector(final["q_observed"])
    arms = {0.0: _vector(final["q_arm0"]), 1.0: _vector(final["q_arm1"])}
    score = score_columns(y, q_observed, submodel.observed, ones, mask)
    if _gap(score, _vector(final["ed"])) > TMLE3_ED:
        findings.append(
            f"the score at the final fit differs from the ED by {_gap(score, _vector(final['ed'])):.3g}"
        )
    if _gap(_vector([arms[0.0].mean(), arms[1.0].mean()]), _vector(final["psi"])) > 1e-12:
        findings.append("psi is not the mean of the final arm predictions")
    curves = {submodel.arm_columns[level]: arms[level] for level in (0.0, 1.0)}
    contributions = rule_contributions(y, q_observed, submodel.observed, ones, mask, 1.0, curves)
    thresholds = rule_thresholds(contributions, StopContext(None, 1.0), rule)
    found = _vector(final["thresholds"])
    if found.shape != thresholds.shape or not np.allclose(
        found, thresholds, rtol=TMLE3_THRESHOLD_RTOL, atol=0.0
    ):
        findings.append(f"thresholds {found} differ from the rule's {thresholds}")

    # The initial ED is the score of the initial fit.
    initial_score = score_columns(y, fit.observed, submodel.observed, ones, mask)
    if _gap(initial_score, ed0) > TMLE3_ED:
        findings.append(f"the initial ED differs from the score by {_gap(initial_score, ed0):.3g}")
    initial_curves = {submodel.arm_columns[level]: fit.arms[level] for level in (0.0, 1.0)}
    initial_thresholds = rule_thresholds(
        rule_contributions(y, fit.observed, submodel.observed, ones, mask, 1.0, initial_curves),
        StopContext(None, 1.0),
        rule,
    )
    initial_met = bool(np.all(np.abs(initial_score) <= initial_thresholds))

    # The walk: cleverly repeats it from the initial arrays.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if constrained:
            findings.extend(
                _compare_walk(
                    result, y, fit, submodel, ones, ed0, epsilons, eds, reported, forced=initial_met
                )
            )
        else:
            findings.extend(_compare_full_step(result, y, fit, submodel, ones, epsilons))
    return findings


def _compare_walk(
    result: Payload,
    y: np.ndarray,
    fit: InitialFit,
    submodel: Any,
    ones: np.ndarray,
    ed0: np.ndarray,
    epsilons: Sequence[np.ndarray],
    eds: Sequence[np.ndarray],
    reported: int,
    *,
    forced: bool,
) -> list[str]:
    """Compare a constrained walk; ``forced`` when the initial fit already meets the rule.

    ``tmle3_Update$update`` takes a step before it first tests the rule, so it takes at least one.
    :func:`solve_fluctuation` tests first and takes none (declared difference).  A fit whose
    initial score meets its rule is therefore compared on its first step, which one iteration
    with the relative rule at ``tol=0`` takes.
    """
    options = result["options"]
    steps = int(result["steps"])
    findings: list[str] = []
    if forced:
        if steps != 1:
            findings.append(
                f"tmle3 took {steps} steps from an initial fit that already meets the rule; "
                "the comparison expects one"
            )
        flu = solve_fluctuation(
            y,
            fit,
            submodel,
            ones,
            stopping="relative",
            tol=0.0,
            step_bound=float(options["delta_epsilon"]),
            step_search=bool(options["optim_delta_epsilon"]),
            max_iter=1,
            warn=False,
        )
        steps = 1
    else:
        flu = solve_fluctuation(
            y,
            fit,
            submodel,
            ones,
            stopping=options["convergence_type"],
            stop=StopContext(None, 1.0),
            step_bound=float(options["delta_epsilon"]),
            step_search=bool(options["optim_delta_epsilon"]),
            iterate="best" if options["use_best"] else "last",
            max_iter=int(options["maxit"]),
            warn=False,
        )
    if flu.n_iter != steps:
        findings.append(f"cleverly walked {flu.n_iter} steps and tmle3 {steps}")
    if options["use_best"] and not forced and flu.best_iterate != reported:
        findings.append(f"cleverly reports iterate {flu.best_iterate} and tmle3 {reported}")
    expected = _cumulative_epsilon(epsilons, eds, ed0, 1 if forced else reported)
    if _gap(np.asarray(flu.epsilon), expected) > TMLE3_EPSILON:
        findings.append(
            f"epsilon {np.asarray(flu.epsilon)} differs from tmle3's cumulative {expected}"
        )
    psi = _vector([flu.targeted.arms[0.0].mean(), flu.targeted.arms[1.0].mean()])
    if _gap(psi, _vector(result["final"]["psi"])) > TMLE3_ESTIMATE:
        findings.append(f"estimates {psi} differ from tmle3's {_vector(result['final']['psi'])}")
    longest = max((abs(float(e[0])) for e in epsilons), default=0.0)
    if longest > float(options["delta_epsilon"]) * (1 + 1e-9):
        findings.append(f"a tmle3 step of length {longest:.6g} exceeds delta_epsilon")
    return findings


def _compare_full_step(
    result: Payload,
    y: np.ndarray,
    fit: InitialFit,
    submodel: Any,
    ones: np.ndarray,
    epsilons: Sequence[np.ndarray],
) -> list[str]:
    steps = int(result["steps"])
    if steps != 1:
        return [f"tmle3 took {steps} steps; an unconstrained updater should reach its rule in one"]
    flu = solve_fluctuation(y, fit, submodel, ones, tol=1e-10, warn=False)
    findings: list[str] = []
    if _gap(np.asarray(flu.epsilon), epsilons[0]) > TMLE3_EPSILON:
        findings.append(f"epsilon {np.asarray(flu.epsilon)} differs from tmle3's {epsilons[0]}")
    psi = _vector([flu.targeted.arms[0.0].mean(), flu.targeted.arms[1.0].mean()])
    if _gap(psi, _vector(result["final"]["psi"])) > TMLE3_ESTIMATE:
        findings.append(f"estimates {psi} differ from tmle3's {_vector(result['final']['psi'])}")
    return findings


# ------------------------------------------------------------------------- drtmle_stratify


def sample_frame(payload: Payload, columns: Sequence[str]) -> pd.DataFrame:
    """The payload's sample as a frame with the outcome, the treatment and ``columns``."""
    data = payload["data"]
    return pd.DataFrame({name: _vector(data[name]) for name in ("Y", "A", *columns)})


def drtmle_fit(payload: Payload, *, stratify: bool) -> Any:
    """The ``cleverly`` counterpart of ``drtmle(guard = NULL, cvFolds = 1)``, in sample."""
    tolg = float(payload["tolg"])
    estimator = DRTMLE(
        guard=(),
        outcome_by_arm=stratify,
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        cross_fit=False,
        estimands=("ey0", "ey1", "ate"),
        g_bounds=(tolg, 1.0 - tolg),
        simultaneous=False,
        random_state=0,
    )
    frame = sample_frame(payload, ("W1", "W2"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()


def compare_drtmle_stratify(payload: Payload) -> list[str]:
    """Repeat ``drtmle(stratify = TRUE)`` and ``drtmle(stratify = FALSE)`` with ``DRTMLE``.

    The comparison reads the initial outcome regression at both arms (``Qn``), the propensity at
    both arms and the TMLE estimates.  It also requires the two R settings to differ, so that a
    stratified fit is not compared with a pooled one that happens to equal it.

    Parameters
    ----------
    payload : mapping
        The parsed ``drtmle_stratify.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    findings: list[str] = []
    settings = {bool(s["stratify"]): s for s in payload["settings"]}
    if set(settings) != {True, False}:
        return ["the payload needs one stratified and one pooled setting"]
    for stratify, setting in settings.items():
        label = "stratify=TRUE" if stratify else "stratify=FALSE"
        result = drtmle_fit(payload, stratify=stratify)
        nuisance = result.repeats[0].nuisance
        for level, key in ((0.0, "qn_arm0"), (1.0, "qn_arm1")):
            gap = _gap(np.asarray(nuisance.outcome.arms[level]), _vector(setting[key]))
            if gap > DRTMLE_QN:
                findings.append(f"{label}: {key} differs from cleverly's by {gap:.3g}")
        propensity = np.asarray(nuisance.propensity.values)
        for column, key in ((0, "gn_arm0"), (1, "gn_arm1")):
            gap = _gap(propensity[:, column], _vector(setting[key]))
            if gap > DRTMLE_QN:
                findings.append(f"{label}: {key} differs from cleverly's by {gap:.3g}")
        found = _vector([result["ey0"].psi, result["ey1"].psi])
        gap = _gap(found, _vector(setting["tmle"]))
        if gap > DRTMLE_ESTIMATE:
            findings.append(f"{label}: the TMLE estimates {found} differ from R's by {gap:.3g}")
    spread = _gap(_vector(settings[True]["qn_arm1"]), _vector(settings[False]["qn_arm1"]))
    if not spread > 1e-3:
        findings.append(
            f"the stratified and pooled R fits differ by {spread:.3g}; the law does not separate them"
        )
    return findings


# ------------------------------------------------------------------------------ ctmle_*


class ColumnGLM(BaseEstimator):
    """A binomial :class:`~cleverly.learners.GLM` on a fixed subset of the design columns.

    The outcome design of a point-treatment fit is ``[A, W1, ..., Wp]``, so ``columns=(0, 2)``
    regresses on the treatment and ``W2``.  It stands for R's ``glm(Y ~ A + W2)``.
    """

    def __init__(self, columns: Sequence[int] | None = None) -> None:
        self.columns = columns

    def _select(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        return values if self.columns is None else values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnGLM:
        self.model_ = GLM(family="binomial").fit(
            self._select(design), np.asarray(target, dtype=float), sample_weight=sample_weight
        )
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._select(design)), dtype=float)


def outcome_columns(payload: Payload) -> list[int]:
    """The design columns of the R outcome regression: the treatment, then ``q_covariates``."""
    covariates = list(payload["covariates"])
    return [0, *(1 + covariates.index(name) for name in np.atleast_1d(payload["q_covariates"]))]


def ctmle_fit(
    payload: Payload,
    *,
    ordering: Sequence[str],
    patience: int | None = None,
    pool: Sequence[str] | None = None,
) -> Any:
    """The ``cleverly`` counterpart of ``ctmleDiscrete(preOrder = TRUE)`` on the payload's sample.

    The selection folds are the payload's, the initial outcome regression is a binomial GLM on
    every covariate, the treatment mechanism is a binomial GLM, and the bound is R's ``gbound``.
    """
    covariates = list(payload["covariates"])
    frame = sample_frame(payload, covariates)
    kwargs: dict[str, Any] = {}
    if pool is not None:
        kwargs["treatment_covariates"] = list(pool)
    estimator = CTMLE(
        strategy="ordered",
        ordering=list(ordering),
        patience=patience,
        selection_split=np.asarray(payload["fold"], dtype=int),
        penalty=False,
        loss="loglik",
        ctmle_estimand="ate",
        cross_fit=False,
        outcome_learner=ColumnGLM(outcome_columns(payload)),
        treatment_learner=GLM(family="binomial"),
        g_bounds=(0.025, 0.975),
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            frame, outcome="Y", treatment="A", covariates=covariates, **kwargs
        ).single()


def _compare_initial(payload: Payload, result: Any) -> list[str]:
    """The initial outcome regression of the fit against the ``Q`` that R was given."""
    arms = result.nuisance.outcome.arms
    findings = []
    for level, key in ((0.0, "q_arm0"), (1.0, "q_arm1")):
        gap = _gap(np.asarray(arms[level]), _vector(payload[key]))
        if gap > CTMLE_INITIAL:
            findings.append(f"the initial {key} differs from R's by {gap:.3g}")
    return findings


def _compare_run(label: str, run: Payload, result: Any, *, expect_stop: bool | None) -> list[str]:
    """The selected index, the scored prefix and the estimate of one ``ctmleDiscrete`` run."""
    findings: list[str] = []
    selection = result.extra["ctmle"]
    score = _vector(run["cv_score"])
    finite = np.isfinite(score)
    scored = int(finite.sum())
    if not finite[:scored].all() or finite[scored:].any():
        findings.append(f"{label}: the unscored candidates are not a suffix of the path")
    if int(run["best_k"]) != selection.selected + 1:
        findings.append(
            f"{label}: R selects k={run['best_k']} and cleverly index {selection.selected} (k={selection.selected + 1})"
        )
    risk = np.asarray(selection.cv_risk, dtype=float)
    if risk.shape != (scored,):
        findings.append(f"{label}: R scored {scored} candidates and cleverly {risk.size}")
    elif not np.allclose(risk, score[:scored], rtol=CTMLE_RTOL, atol=0.0):
        findings.append(
            f"{label}: the cross-validated scores differ by up to {_gap(risk, score[:scored]):.3g}"
        )
    estimate = float(result["ate"].psi)
    if abs(estimate - float(run["est"])) > CTMLE_ESTIMATE:
        findings.append(
            f"{label}: the estimate {estimate:.8g} differs from R's {float(run['est']):.8g}"
        )
    stopped = selection.stopped_at is not None
    if expect_stop is not None and stopped != expect_stop:
        findings.append(
            f"{label}: stopped_at is {selection.stopped_at} but R stopped early is {expect_stop}"
        )
    if bool(run.get("stopped_early", False)) != stopped:
        findings.append(
            f"{label}: R stopped early is {run.get('stopped_early')} and cleverly stopped_at is {selection.stopped_at}"
        )
    return findings


def compare_ctmle_patience(payload: Payload) -> list[str]:
    """Repeat ``ctmleDiscrete(patience = 2)`` and ``patience = FALSE`` with ``CTMLE(patience=)``.

    The comparison reads the selected index, the cross-validated score of each scored candidate and
    the estimate.  The patient run must stop early, or the fixture does not exercise the counter.

    Parameters
    ----------
    payload : mapping
        The parsed ``ctmle_patience.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    covariates = list(payload["covariates"])
    findings: list[str] = []
    whole = payload["runs"]["whole_path"]
    patient = payload["runs"]["patient"]
    if not patient["stopped_early"]:
        findings.append(
            "the patient R run did not stop early, so the fixture does not exercise patience"
        )
    if whole["stopped_early"]:
        findings.append("the whole-path R run stopped early")
    full = ctmle_fit(payload, ordering=covariates)
    findings.extend(_compare_initial(payload, full))
    findings.extend(_compare_run("whole path", whole, full, expect_stop=False))
    findings.extend(
        _compare_run(
            "patience",
            patient,
            ctmle_fit(payload, ordering=covariates, patience=int(payload["patience"])),
            expect_stop=True,
        )
    )
    return findings


def compare_ctmle_wg(payload: Payload) -> list[str]:
    """Repeat ``ctmleDiscrete(Wg = W[, c(1, 3)])`` and ``Wg = W`` with ``treatment_covariates=``.

    Beside the selected index, the scores and the estimate, the comparison requires the restricted
    path to stay inside the pool and the two R runs to give different estimates, so that the
    restriction has an effect to compare.

    Parameters
    ----------
    payload : mapping
        The parsed ``ctmle_wg.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    covariates = list(payload["covariates"])
    pool = [covariates[int(i) - 1] for i in np.atleast_1d(payload["wg_columns"])]
    restricted = payload["runs"]["restricted"]
    whole = payload["runs"]["whole_pool"]
    findings: list[str] = []
    if list(np.atleast_1d(restricted["pool"])) != pool:
        findings.append(f"R's restricted pool {restricted['pool']} is not {pool}")
    result = ctmle_fit(payload, ordering=pool, pool=pool)
    selection = result.extra["ctmle"]
    findings.extend(_compare_initial(payload, result))
    findings.extend(_compare_run("restricted", restricted, result, expect_stop=False))
    if selection.restricted_pool != tuple(pool):
        findings.append(f"cleverly's restricted pool is {selection.restricted_pool}")
    if any(not set(names) <= set(pool) for names in selection.path):
        findings.append("a cleverly candidate leaves the pool")
    if not set(np.atleast_1d(restricted["terms"])) <= set(pool):
        findings.append(f"R's terms {restricted['terms']} leave the pool")
    full = ctmle_fit(payload, ordering=covariates)
    findings.extend(_compare_run("whole pool", whole, full, expect_stop=False))
    if full.extra["ctmle"].restricted_pool is not None:
        findings.append("the whole-pool run reports a restricted pool")
    if not abs(float(restricted["est"]) - float(whole["est"])) > 1e-4:
        findings.append(
            "the restricted and whole-pool R estimates agree to 1e-4; Wg has no effect to compare"
        )
    return findings


COMPARATORS: dict[str, Callable[[Payload], list[str]]] = {
    "tmle3_updater": compare_tmle3_updater,
    "drtmle_stratify": compare_drtmle_stratify,
    "ctmle_patience": compare_ctmle_patience,
    "ctmle_wg": compare_ctmle_wg,
}
