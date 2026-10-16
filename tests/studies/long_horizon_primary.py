"""The primary phase the three long-horizon studies share: scenarios, fits and rows.

A :class:`Scenario` names one law, one sample size, one decision count and arm count, one fold
count and its plans, reference first.  Each scenario fits with its law's mechanism passed as
probability columns, the declared quasibinomial GLM for the outcome regressions (least squares on
the transform for ``SDR`` before the horizon) and ``g_bounds=(1e-8, 1)``.

A scenario reports every plan mean, every contrast against the reference, and each declared
paired contrast.  A paired contrast reads the joint curve of its two means through
:meth:`~cleverly.longitudinal.LongitudinalResult.contrast`.  The MSM scenario reports the two
coefficients of its working model only.

**Intervals** (plan rev 4.3).  A single-fold scenario that is not an MSM scenario fits with
``n_bootstrap=B_GATE`` and ``bootstrap_interval="bca"``.  It writes two rows per estimand: the
Wald row under the estimand's own name, and the BCa row under the name plus :data:`BCA_SUFFIX`.
The BCa row carries the BCa limits as its interval and the bootstrap SD of the replicate
estimates as its standard error.  The shared replicate schema has no interval column, so the
suffix is the encoding.  A cross-fitted scenario and an MSM scenario write Wald rows only: a
cross-fitted bootstrap interval is a diagnostic (X20), and the MSM fit runs no bootstrap.

No comparator runs while the comparator is pending, so the primary phase returns no sample and no
truth frame: :func:`run_primary` returns ``PhaseResult(samples=None, truths=None)``.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.msm import MSM
from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law
from tests.studies import long_horizon_common as common
from tests.studies.evidence import replication
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

#: The two terms of the duration working model.
MSM_TERMS = ("(intercept)", "duration")
#: The bootstrap replicates of the gated BCa interval.
B_GATE = 399
#: The suffix of a BCa row's estimand.
BCA_SUFFIX = "__bca"


@dataclass(frozen=True)
class Scenario:
    """One primary scenario.

    Parameters
    ----------
    name : str
        The scenario name.
    K : int
        The number of decisions.
    m : int
        The number of arms at each decision.
    law : Law
        The law the scenario draws from.
    n : int
        The declared sample size.
    plans : tuple of str
        The plans, the reference first.
    n_folds : int
        The outer fold count.
    paired : tuple of tuple of str
        The paired contrasts ``(a, b)`` reported beside the contrasts against the reference.
    msm : bool
        Whether the scenario reports the duration working model instead of the plan means.
    """

    name: str
    K: int
    m: int
    law: Law
    n: int
    plans: tuple[str, ...]
    n_folds: int = 1
    paired: tuple[tuple[str, str], ...] = ()
    msm: bool = False

    @property
    def reference(self) -> str:
        """The reference plan."""
        return self.plans[0]


def mean_name(plan: str) -> str:
    """The reported name of a plan mean."""
    return f"ey_regimen[{plan}]"


def contrast_name(first: str, second: str) -> str:
    """The reported name of ``psi(first) - psi(second)``."""
    return f"ate_regimen[{first} vs {second}]"


def bootstrapped(scenario: Scenario) -> bool:
    """Whether ``scenario`` fits with the refit bootstrap: single fold, no working model."""
    return scenario.n_folds == 1 and not scenario.msm


def bca_name(estimand: str) -> str:
    """The estimand of the BCa row that goes with the Wald row ``estimand``."""
    return estimand + BCA_SUFFIX


def wald_estimands(scenario: Scenario) -> tuple[str, ...]:
    """The estimands whose Wald row ``scenario`` reports, in a fixed order."""
    if scenario.msm:
        return tuple(f"msm_regimen[{term}]" for term in MSM_TERMS)
    means = tuple(mean_name(plan) for plan in scenario.plans)
    against = tuple(contrast_name(plan, scenario.reference) for plan in scenario.plans[1:])
    paired = tuple(contrast_name(first, second) for first, second in scenario.paired)
    return means + against + paired


def estimands(scenario: Scenario) -> tuple[str, ...]:
    """The estimands ``scenario`` reports: the Wald rows, then the BCa row of each of them."""
    wald = wald_estimands(scenario)
    if not bootstrapped(scenario):
        return wald
    return wald + tuple(bca_name(name) for name in wald)


def truths(scenario: Scenario) -> dict[str, float]:
    """The exact value of every estimand of ``scenario``, from the oracle.

    A BCa row has the truth of its Wald row.
    """
    K, m, law = scenario.K, scenario.m, scenario.law
    if scenario.msm:
        coefficients = law_.msm_coefficients(K, law)
        return {
            f"msm_regimen[{term}]": coefficients[key]["psi"]
            for term, key in zip(MSM_TERMS, ("msm_intercept", "msm_duration"), strict=True)
        }
    level = {plan: law_.truth(K, m, plan, law) for plan in scenario.plans}
    out = {mean_name(plan): value for plan, value in level.items()}
    for plan in scenario.plans[1:]:
        out[contrast_name(plan, scenario.reference)] = level[plan] - level[scenario.reference]
    for first, second in scenario.paired:
        out[contrast_name(first, second)] = level[first] - level[second]
    if bootstrapped(scenario):
        out |= {bca_name(name): value for name, value in list(out.items())}
    return out


def duration_msm() -> MSM:
    """The working model ``E[Y_dj] = b0 + b1 j`` over ``dur0..durK``, identity link, unit weights."""

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.column_stack(
            [np.ones(len(frame)), np.full(len(frame), float(str(label).removeprefix("dur")))]
        )

    def weight(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del label, horizon
        return np.ones(len(frame))

    return MSM(
        design=design,
        terms=MSM_TERMS,
        weights=weight,
        weights_kind="known",
        design_kind="known",
    )


def fit(
    frame: pd.DataFrame,
    scenario: Scenario,
    estimator: str,
    *,
    bootstrap: bool = True,
    **extra: Any,
) -> Any:
    """One primary fit of ``scenario`` with ``estimator`` (``"LTMLE"`` or ``"SDR"``).

    A bootstrapped scenario (:func:`bootstrapped`) fits with ``n_bootstrap=B_GATE`` and
    ``bootstrap_interval="bca"``; ``bootstrap=False`` fits the Wald rows alone (the rev 4.2
    ``truth`` pilot).  ``extra`` reaches the fit.
    """
    if scenario.msm:
        extra["msm"] = duration_msm()
    elif bootstrap and bootstrapped(scenario):
        extra.setdefault("n_bootstrap", B_GATE)
        extra.setdefault("bootstrap_interval", "bca")
    return common.fit(
        frame,
        scenario.K,
        scenario.m,
        scenario.plans,
        estimator=estimator,
        n_folds=scenario.n_folds,
        g_bounds=common.G_BOUNDS,
        **extra,
    )


def _initial_mean(regimen_fit: Any) -> float:
    """The plug-in at the first node, policy-weighted where the first node is a policy."""
    first = regimen_fit.steps[0]
    policy = getattr(regimen_fit, "policy", None)
    if first.initial_by_arm is None or not policy or policy[0] is None:
        return float(np.mean(first.initial))
    return float(np.mean(np.sum(policy[0] * first.initial_by_arm, axis=1)))


def _initials(result: Any, scenario: Scenario) -> dict[str, float]:
    if scenario.msm:
        labels = scenario.plans
        means = np.array([_initial_mean(result.fits[label]) for label in labels])
        design = np.column_stack(
            [np.ones(len(labels)), [float(label.removeprefix("dur")) for label in labels]]
        )
        beta = np.linalg.solve(design.T @ design, design.T @ means)
        return {f"msm_regimen[{term}]": float(b) for term, b in zip(MSM_TERMS, beta, strict=True)}
    level = {plan: _initial_mean(result.fits[plan]) for plan in scenario.plans}
    out = {mean_name(plan): value for plan, value in level.items()}
    for plan in scenario.plans[1:]:
        out[contrast_name(plan, scenario.reference)] = level[plan] - level[scenario.reference]
    for first, second in scenario.paired:
        out[contrast_name(first, second)] = level[first] - level[second]
    return out


def _estimates(result: Any, scenario: Scenario) -> dict[str, Any]:
    """Every reported Wald-named estimate, the paired contrasts derived from the joint curve."""
    paired = {contrast_name(first, second) for first, second in scenario.paired}
    out = {name: result[name] for name in wald_estimands(scenario) if name not in paired}
    for first, second in scenario.paired:
        out[contrast_name(first, second)] = result.contrast(
            lambda values: float(values[0] - values[1]),
            [mean_name(first), mean_name(second)],
            name=contrast_name(first, second),
            gradient=lambda values: np.array([1.0, -1.0]),
        )
    return out


def sparse_warnings(estimates: Mapping[str, Any], alpha: float = 0.05) -> dict[str, bool]:
    """Whether ``SparseInfluenceWarning`` fires for each estimate: its variance SD is above 0.125.

    A bootstrap fit raises no such warning (it is a warning of the Wald fit), so the flag is the
    decision the warning would take: :func:`cleverly.inference.sparse.influence_report` on the
    estimates that supply a Wald interval.
    """
    from cleverly.inference.sparse import influence_report

    report = influence_report(
        {name: e for name, e in estimates.items() if e.supplies_inference}, alpha=alpha
    )
    return {row.name: bool(row.sparse) for row in report.rows}


def rows(
    record: StudyRecord, result: Any, scenario: Scenario, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's published rows of ``scenario``.

    Each row also carries three keys that the shared schema does not publish: ``interval``
    (``wald`` or ``bca``), ``n_failed`` (the failed refits of the fit's bootstrap, 0 without
    one) and ``sparse_warning`` (whether ``SparseInfluenceWarning`` fires for the estimand).
    The primary loop drops them; the ``truth_boot`` pilot keeps them.
    """
    truth = truths(scenario)
    initials = _initials(result, scenario)
    estimates = _estimates(result, scenario)
    sparse = sparse_warnings(estimates, result.config.alpha_sig)
    failed = 0 if result.bootstrap is None else int(result.bootstrap.n_failed)
    out: list[dict[str, Any]] = []
    wald_only = result.bootstrap is None
    for name in wald_estimands(scenario) if wald_only else estimands(scenario):
        base = name.removesuffix(BCA_SUFFIX)
        estimate = estimates[base]
        is_bca = name != base
        if is_bca:
            summary = estimate.bootstrap
            if summary is None:
                raise ValueError(f"{name}: the fit attached no bootstrap summary")
            low, high = summary.ci
            std_error = summary.std_error
        else:
            low, high = estimate.ci
            std_error = estimate.std_error
        reference = float(truth[name])
        out.append(
            {
                "implementation": record.implementation,
                "scenario": scenario.name,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": initials[base],
                "interval": "bca" if is_bca else "wald",
                "n_failed": failed if is_bca else 0,
                "sparse_warning": int(sparse.get(base, False)),
            }
        )
    return out


# ----------------------------------------------------------------------------- the primary loop
def _draw(record: StudyRecord, scenarios: Mapping[str, Scenario], key: Any) -> pd.DataFrame:
    name, replicate, n = key
    scenario = scenarios[name]
    seed = replicate_seed(record, name, replicate)
    return law_.draw(scenario.K, scenario.m, n, seed, scenario.law)


def _fit_one(
    record: StudyRecord,
    scenarios: Mapping[str, Scenario],
    estimator: str,
    key: Any,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    name, replicate, n = key
    scenario = scenarios[name]
    return rows(record, fit(frame, scenario, estimator), scenario, replicate, n)


def run_primary(
    record: StudyRecord,
    scenarios: Sequence[Scenario],
    estimator: str,
    *,
    replicates: int,
    n: int,
    n_jobs: int,
) -> replication.PhaseResult:
    """The primary loop of a long-horizon study.

    ``n`` is the driver's ``--n``: at the record's own ``n`` each scenario runs at its declared
    size, and any other ``n`` runs every scenario at ``n`` (``StudyRecord.sizes``).
    """
    table = {scenario.name: scenario for scenario in scenarios}
    sizes = record.sizes(n)
    keys = [
        (name, replicate, sizes[name])
        for replicate in range(replicates)
        for name in record.scenarios
    ]
    outcomes = replication.fit_primary(
        keys,
        functools.partial(_fit_one, record, table, estimator),
        draw=functools.partial(_draw, record, table),
        n_jobs=n_jobs,
        implementation=record.implementation,
        rows_at=None,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    frame = pd.DataFrame([row for kept in outcomes.kept for row in kept])
    if frame.empty:
        frame = pd.DataFrame(columns=list(REPLICATE_COLUMNS))
    return replication.result(outcomes, frame.loc[:, list(REPLICATE_COLUMNS)])


def declared(scenarios: Sequence[Scenario]) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """The record's ``scenarios`` and ``scenario_n`` maps."""
    return (
        {scenario.name: estimands(scenario) for scenario in scenarios},
        {scenario.name: scenario.n for scenario in scenarios},
    )
