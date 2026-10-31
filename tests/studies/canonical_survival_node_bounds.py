r"""Registered evidence for node bounds on survival fits (``survival-node-bounds``, X33 part (i)).

The subject is ``LTMLE(node_bounds=..., hazard_pooling="pooled")`` on a point-treatment
survival law with five integer visits (:mod:`tests.studies.survival_node_bounds_law`).  Two
models are fitted on every sample:

=================  ==============================================================================
model              construction
=================  ==============================================================================
``mean_model``     ``survival_model="iterated_mean"``.  Every node is bounded by
                   :data:`MEAN_BOUNDS`, so every node regression is the study-local
                   :class:`~tests.studies.fractional_glm.QuasiBinomialGLM` on
                   :math:`(Y - l_t)/(u_t - l_t)` over the arm's followers, on ``W1, W2``.  That
                   per-arm fit is ``survtmle``'s ``glm.ftime = "trt*(W1 + W2)"``
                   (``X22-plan-review.md`` finding 6a)
``hazard_model``   ``survival_model="hazard"``, per-horizon targeting.  Every node's hazard is
                   bounded by :data:`HAZARD_BOUNDS`, and the pooled hazard is
                   :class:`PooledHazardGLM`: ``trt*(W1 + W2) + t``
=================  ==============================================================================

Both models fit the treatment mechanism by an unpenalized logistic regression on ``W1 + W2``
and the censoring mechanism by one unpenalized logistic regression pooled over nodes 2 to 5, on
``trt + W1 + W2 + t`` (``hazard_pooling="pooled"``).  No unit is censored before node 1, so the
container declares node 1's retention known, with probability one
(:func:`~tests.studies.survival_node_bounds_law.container`).  A declared-known node stays out of
the pooled fit, which is ``survtmle``'s ``g_dC = 1`` at the first time.  Every fit is in sample
(``n_folds=1``).  The declared law is in each family, so every nuisance model is correctly
specified, except that the additive logit of the scaled iterated mean is not saturated.  Its
exact-law standardized bias is below ``1e-3`` SD at n = 2,000
(``tests/unit/test_survival_node_bounds_design.py``).

**Valid bounds.**  ``l_t = 0`` and ``u_t = f`` times the largest at-risk iterated mean at node
``t`` over ``(a, W)`` and the horizons 3 and 5 (mean model), or ``f`` times the largest true
hazard at node ``t`` (hazard model).  The declaration sets ``f = 1.5`` and drops it to 1.1 when
the design test's expected exceedance rate of the unbounded fit is below one half.  The design
test computed the rate, and :data:`BOUND_FACTOR` is its outcome.  Both factors hold the truth.

**The comparator** is ``survtmle`` 1.1.1 (``method="mean"`` for P1 and ``method="hazard"`` for
P2) on the same samples, in the image ``cleverly-survtmle:1.1.1`` (``tests/canonical/survtmle``,
base image pinned by digest, the CRAN tarball by checksum), through
``tests/canonical/survival_node_bounds/run_study.R``.  Both calls use the declared formulas.  The
censoring formula ``trt + W1 + W2 + t`` is fitted once per sample by the runner on the
subject's row set (visits 1 to 4, at risk after the event check) and passed to survtmle as a
fitted glm, because survtmle's own row rule keeps the study's end at visit 5 as a censoring
when ``t0 = 3``.  P1 is a matching implementation.  P2 is a different estimator: survtmle
fluctuates every visit's hazard jointly, and the subject targets each horizon on its own.

**Failure bound.**  ``failure_bound`` is 0.01 for each implementation (``FAILURE_BOUND``).  A
cell may lose ``floor(0.01 R)`` replicates, and the accounting fills each lost value at its
adverse support bound (``docs/development/method-benchmarking.md``).  More lost replicates turn
the cell red.  A fit whose arm has no event at some visit reports ``constant_node_plugin``,
and :func:`refuse_constant_nodes` records it as a lost replicate.  The design test bounds that
chance by 5.3e-7 per draw at n = 2,000.

**Publication policy** is ``reporting``.  The red-cell route is declared in the declaration
commit, before any run.  More than 1 per cent of unconverged replicates in a cell turns the cell
red; N's control converges by design.  A red cell goes to the red-cell ledger, and an owner row
enters the roadmap only for a red cell.  No re-run uses a new budget, margin or law.

**Registered.**  The Python run finished with 7,200 replicates and its artifacts are committed,
so the record is in ``registered()``.  The survtmle phase runs from the declared commit with
``--skip-properties`` on the local Study queue (TRACKER row 45).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import DataError
from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import survival_node_bounds_law as law
from tests.studies.canonical_point_survival import initial_estimates as mean_initial_estimates
from tests.studies.canonical_point_survival_hazard import (
    initial_estimates as hazard_initial_estimates,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The smallest n in {1,000, 2,000, 4,000} with at least 40 expected events per arm by visit 5
#: and a union bound of at most 1e-6 on the chance that some arm has no event at some visit.
PRIMARY_N = 2_000
#: The smallest multiple of 400 at which the family-wise product over every gated cell (K by the
#: arm-pooled predictive rule, W and N) is at least 0.95
#: (``tests/unit/test_survival_node_bounds_design.py``).
REPLICATES = 7_200
PRIMARY_REPLICATES = REPLICATES
SEED = 20263391
RESAMPLING_SEED = 2026339101
FAILURE_BOUND = 0.01

#: The factor of the valid bounds, decided by the design test's exceedance rule.
BOUND_FACTOR = law.REDUCED_FACTOR
MEAN_BOUNDS: dict[int, tuple[float, float]] = law.mean_bounds(BOUND_FACTOR)
HAZARD_BOUNDS: dict[int, tuple[float, float]] = law.hazard_bounds(BOUND_FACTOR)

#: The targeting settings of every fit.
MAX_ITER = 100
TOL = 1e-10
#: The cumulative bound the fits declare: wide enough never to bind on this law, whose
#: smallest product of treatment and retention probabilities is above 0.3.
G_BOUNDS = (1e-8, 1.0)

MEAN_MODEL = "mean_model"
HAZARD_MODEL = "hazard_model"
MODELS = (MEAN_MODEL, HAZARD_MODEL)


def _names() -> tuple[str, ...]:
    levels = tuple(
        law.risk_name(label, horizon) for label in law.REGIMENS for horizon in law.HORIZONS
    )
    return levels + tuple(law.contrast_name(horizon) for horizon in law.HORIZONS)


ESTIMANDS: dict[str, tuple[str, ...]] = {model: _names() for model in MODELS}
TRUTH: dict[str, dict[str, float]] = {
    model: {name: law.TRUTH[name] for name in ESTIMANDS[model]} for model in MODELS
}


class PooledHazardGLM(BaseEstimator, ClassifierMixin):
    """The unpenalized logistic hazard ``trt*(W1 + W2) + t`` on the pooled hazard design.

    The pooled design of this law is ``[W1, W2, A, t]`` (``hazard_pooling="pooled"``); the
    learner adds ``A W1`` and ``A W2``.  That is ``survtmle``'s ``glm.ftime``
    ``"trt*(W1 + W2) + t"`` (``X22-plan.md`` II-3.4, P2).  Any other width raises.
    """

    def __init__(self, *, max_iter: int = 1000) -> None:
        self.max_iter = max_iter

    @staticmethod
    def _expand(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 4:
            raise ValueError(f"the pooled hazard design has 4 columns, not {matrix.shape[1]}")
        w1, w2, a, t = matrix.T
        return np.column_stack([w1, w2, a, a * w1, a * w2, t])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> PooledHazardGLM:
        self.model_ = LogisticRegression(C=np.inf, max_iter=self.max_iter)
        self.model_.fit(self._expand(X), np.asarray(y).reshape(-1), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._expand(X)))


def logistic() -> LogisticRegression:
    """The unpenalized logistic regression of both mechanisms."""
    return LogisticRegression(C=np.inf, max_iter=1000)


def mean_learners() -> dict[str, Any]:
    """The learners of the iterated-mean fit."""
    return {
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": QuasiBinomialGLM(),
        "treatment_learner": logistic(),
        "censoring_learner": logistic(),
    }


def hazard_learners() -> dict[str, Any]:
    """The learners of the hazard fit."""
    return {
        "outcome_learner": PooledHazardGLM(),
        "treatment_learner": logistic(),
        "censoring_learner": logistic(),
    }


def _estimator(
    model: str, bounds: Mapping[int, tuple[float, float]] | None, **learners: Any
) -> Any:
    hazard = model == HAZARD_MODEL
    chosen = {**(hazard_learners() if hazard else mean_learners()), **learners}
    return LTMLE(
        law.REGIMENS,
        reference=law.REFERENCE,
        horizons=list(law.HORIZONS),
        n_folds=1,
        simultaneous=False,
        survival_model="hazard" if hazard else "iterated_mean",
        targeting="iterative",
        hazard_pooling="pooled",
        node_bounds=None if bounds is None else law.as_mapping(bounds),
        g_bounds=G_BOUNDS,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
        **chosen,
    )


def declared_bounds(model: str, factor: float = BOUND_FACTOR) -> dict[int, tuple[float, float]]:
    """The valid bounds of one model at ``factor``."""
    return law.mean_bounds(factor) if model == MEAN_MODEL else law.hazard_bounds(factor)


def fit(
    model: str,
    frame: pd.DataFrame,
    *,
    bounds: Mapping[int, tuple[float, float]] | str | None = "declared",
    weights: str | None = None,
    **learners: Any,
) -> Any:
    """One fit of ``model``: with the declared bounds, other bounds, or ``bounds=None``."""
    chosen = declared_bounds(model) if bounds == "declared" else bounds
    assert not isinstance(chosen, str)
    result = _estimator(model, chosen, **learners).fit(law.container(frame, weights=weights))
    refuse_constant_nodes(model, result)
    return result


def refuse_constant_nodes(model: str, result: Any) -> None:
    """Turn a ``constant_node_plugin`` estimate into a recorded failure of the replicate.

    A node where an arm has no event has a constant regression target, and the shipped fit then
    reports the plug-in without an interval.  The study measures the influence-curve interval,
    so such a draw is a lost replicate, recorded by the driver at the shipped worst-case fill
    (a study-local translation, as ``canonical_ctmle_oat_per_arm_parametric`` does for its
    stacked curve).  The design test bounds the chance of the draw
    (:func:`~tests.studies.survival_node_bounds_law.no_event_bound`).
    """
    for name in ESTIMANDS[model]:
        status = result[name].inference
        if status != "influence_curve":
            raise DataError(
                f"{name} reports the inference status {status!r}, not the influence curve"
            )


def initial_estimates(model: str, result: Any, names: Sequence[str]) -> dict[str, float]:
    """The untargeted plug-in of every reported parameter of one fit."""
    if model == HAZARD_MODEL:
        return hazard_initial_estimates(result, names)
    return mean_initial_estimates(result, names)


def mean_exceedance(result: Any, bounds: Mapping[int, tuple[float, float]]) -> bool:
    """Whether an iterated-mean fit's targeted prediction leaves ``bounds`` on a node-t follower.

    Reads every regimen's and horizon's recursion at every node ``t``.
    """
    for regimen_fit in result.fits.values():
        for step in regimen_fit.steps:
            lower, upper = bounds[step.time]
            followers = np.asarray(step.trained_on, dtype=bool)
            values = np.asarray(step.targeted, dtype=float)[followers]
            if values.size and (values.max() > upper or values.min() < lower):
                return True
    return False


def hazard_exceedance(
    result: Any, frame: pd.DataFrame, bounds: Mapping[int, tuple[float, float]]
) -> bool:
    """Whether a hazard fit's initial hazard leaves ``bounds`` on a unit at risk at node ``t``.

    The unit is read at its own arm, so it is a node-``t`` follower of that arm.  This reads
    the declared interval :math:`[l, u]`.  The bounded fit clips into the narrower interval
    :math:`[l + (1 - \alpha) s, u - (1 - \alpha) s]` (``share_clipped``), so a hazard can
    stay inside the bounds and still be clipped.
    """
    hazards = np.asarray(result.initial_hazards.hazards, dtype=float)
    arm = np.asarray(frame["A"], dtype=int)
    time = np.asarray(frame["time"], dtype=float)
    for t, (lower, upper) in bounds.items():
        at_risk = time >= float(t)
        values = hazards[at_risk, t - 1, arm[at_risk], 0]
        if values.size and (values.max() > upper or values.min() < lower):
            return True
    return False


# ---------------------------------------------------------------------------- record

_GATED_KEYS = ("f5_arm1", "rd5")
_REPORTED_KEYS = ("f3_arm1", "f3_arm0", "f5_arm0")

#: The cells the committed property summary must contain.  The properties module builds the
#: same names from its fit sets, and the design test checks the two agree.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{model}_{key}__{kind}"
        for model in MODELS
        for key in _GATED_KEYS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "reported_calibration": tuple(
        f"{model}_{key}__reported" for model in MODELS for key in _REPORTED_KEYS
    ),
    "binding_witness": tuple(f"{model}__binding" for model in MODELS),
    "unbounded_fit": tuple(f"{model}_{key}__unbounded" for model in MODELS for key in _GATED_KEYS),
    "bound_necessity": ("f5_arm1__valid_bounds", "f5_arm1__excluding_bound"),
}

STUDY = StudyRecord(
    name="node bounds on survival fits",
    slug="survival-node-bounds",
    artifacts=ROOT / "tests" / "canonical" / "survival_node_bounds",
    document="docs/technical-reference/method-evidence/survival-node-bounds.md",
    anchor="survival-node-bounds",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    nuisance_count=3,
    margins=Margins(),
    implementation="cleverly",
    reference="survtmle",
    # The two hazard fluctuations differ: survtmle fluctuates every visit's hazard jointly until
    # each |meanIC| <= tol, and the subject targets each horizon on its own walk, as the
    # unmatched settings of ``canonical_point_survival_hazard`` state.
    comparison="matching implementation",
    comparison_overrides={f"{HAZARD_MODEL}:*": "different estimator"},
    failure_bound={"cleverly": FAILURE_BOUND, "survtmle": FAILURE_BOUND},
    efficiency_bounds={name: law.EFFICIENCY_SD[name] for name in ESTIMANDS[MEAN_MODEL]},
    modules=(
        "tests/studies/canonical_survival_node_bounds.py",
        "tests/studies/survival_node_bounds_properties.py",
        "tests/studies/survival_node_bounds_law.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/point_survival_properties.py",
        "tests/studies/canonical_ltmle.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/longitudinal/pooled_hazards.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/fluctuation/iterative.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_survival_node_bounds",
    properties_module="tests.studies.survival_node_bounds_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "survtmle_version": "1.1.1",
    "reference_construction": (
        "P1: survtmle(method='mean', bounds=valid, glm.trt='W1 + W2', glm.ftime='trt*(W1 + W2)', "
        "glm.ctime='trt + W1 + W2 + t'); P2: survtmle(method='hazard', bounds=hazard-valid, "
        "glm.ftime='trt*(W1 + W2) + t'), one t0 per call, bounds at every visit; the censoring "
        "glm 'trt + W1 + W2 + t' is fitted on the subject's visit 1-4 rows and passed as a fit"
    ),
    "pairing": (
        "the shipped default margins (paired difference 0.15 SD, RMSE ratio 1.10, coverage "
        "-0.025, calibration 0.05), declared a priori; D10 (node-1 pooled censoring) is handled "
        "by declaring node 1 known, and none is routed"
    ),
}

CONFIGURATION = {
    "construction": "held point-treatment survival with node bounds, in sample",
    "models": list(MODELS),
    "bound_factor": BOUND_FACTOR,
    "mean_bounds": {str(t): list(b) for t, b in MEAN_BOUNDS.items()},
    "hazard_bounds": {str(t): list(b) for t, b in HAZARD_BOUNDS.items()},
    "hazard_pooling": "pooled",
    "outcome_learner": "quasibinomial IRLS on W1, W2 per arm (mean); logistic trt*(W1+W2)+t "
    "pooled (hazard)",
    "treatment_learner": "unpenalized logistic regression on W1, W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A, t pooled over nodes 2 to 5; "
    "node 1 declared known (probability one)",
    "regimens": list(law.REGIMENS),
    "horizons": list(law.HORIZONS),
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    del scenario
    return law.draw(n, seed), dict(law.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def regimen_rows(
    result: Any,
    truth: Mapping[str, float],
    initials: Mapping[str, float],
    scenario: str,
    replicate: int,
    n: int,
) -> list[dict[str, Any]]:
    """One replication's rows on the identity scale."""
    rows = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = frame.loc[:, ["W1", "W2", "A", "time", "event"]].copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": truth[name]}
        for name in ESTIMANDS[scenario]
    ]
    return sample, truth_rows, []


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result = fit(scenario, frame)
    rows = regimen_rows(
        result,
        truth,
        initial_estimates(scenario, result, ESTIMANDS[scenario]),
        scenario,
        replicate,
        n,
    )
    return sample, truth_rows, rows


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of both models, its truth rows and the subject's rows."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in MODELS],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by model."""
    return TRUTH


__all__ = [
    "BOUND_FACTOR",
    "HAZARD_BOUNDS",
    "MEAN_BOUNDS",
    "STUDY",
    "TRUTH",
    "draw_and_fit",
    "fit",
]
