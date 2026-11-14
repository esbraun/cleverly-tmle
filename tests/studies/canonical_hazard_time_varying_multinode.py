"""Declared evidence for hazard targeting with covariates after baseline beside a treatment at
several nodes (``hazard-time-varying-multinode``).

The subject is ``LTMLE(survival_model="hazard", targeting="one_step")`` on a design with a
binary treatment at each of three nodes and a binary covariate :math:`L_k` measured before the
treatment at every node after the first (X70 part (b2)).  One backward pass per regimen targets,
at each node, the node's event hazards and the continuation, a soft-label hazard fit of the
later events with the later covariates integrated out, as one node panel.  Each node is
evaluated at the observed history: its slots are the regimen's values of :math:`A_k` there and
its hazards are the learner's predictions at the observed earlier treatments with :math:`A_k`
set (``docs/technical-reference/longitudinal-tmle.md``, "A treatment at several nodes with
covariates after baseline").

**The law** is law (M) of :mod:`tests.studies.hazard_time_varying_multinode_law`: one cause,
three nodes, a hazard that reads :math:`A_{k-1}` and :math:`A_k`, a treatment that reads
:math:`(W_1, A_{k-1}, L_k)` and a covariate that reads :math:`(W_1, A_{k-1}, L_{k-1})`.

**Regimens and estimates.**  Always, never, the rule (treat when :math:`W_1 = 1` at node 1 and
when :math:`L_k = +1` after) and the known policy (:func:`regimens`).  21 estimates: each
regimen's risk at each horizon (12) and each of always, the rule and the policy against never at
each horizon (9).

**Fit labels** (:data:`FITS`): ``multinode`` in sample, the primary scenario, and ``crossfit``,
the same law at five outer folds, a calibration fit label only.

**Held cells.**  As ``hazard-time-varying``: :func:`refuse_held_cells` makes a fit with an
estimate outside the influence-curve interval a recorded lost replicate on every path.

**Truths and bounds** are exact sums over the finite support, computed by code that never calls
the estimator; the design test checks each by an independent forward sum.

**No comparator.**  R ``ltmle`` and ``lmtp`` reach this estimand by the iterated-mean algorithm
only, which ``canonical-ltmle-survival`` already ties to the in-package fit, so they would add
transitive evidence only.  The record names no reference.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/hazard_time_varying_multinode/regenerate.py`` before any run.  The record is
not registered until its declared run passes; its support rows ``G63.multinode.*`` land with the
registration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.longitudinal.hazard_backward import untargeted
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_time_varying as held
from tests.studies import hazard_time_varying_multinode_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The shared primary rule (``repeated_sizing.primary_replicates``), provisional until the
#: ``multinode`` calibration pilot is committed (the design test).
PRIMARY_REPLICATES = 2_400
PRIMARY_N = 2_000
SEED = 20261902
RESAMPLING_SEED = 2026190201

#: The targeting settings of every hazard fit of this study.
ALGORITHM = "one_step"
TOL = 1e-10
#: The exact-sample gate: a fit of the law as data returns every truth within this.
EXACT = 1e-8
#: The smoke replicates per fit label and configuration, on the pilot stream.
SMOKE_REPLICATES = 2

#: Fit label -> outer folds.
FITS: dict[str, int] = {"multinode": 1, "crossfit": 5}
#: The primary scenario.
SCENARIOS = ("multinode",)
#: The regimens in the order a fit reports them, and the reference of every contrast.
LABELS = tuple(law.REGIMENS)
REFERENCE = law.REFERENCE
CONTRASTED = tuple(label for label in LABELS if label != REFERENCE)
LAST = law.LAW.n_times


def _flag(values: Any) -> np.ndarray:
    return (np.nan_to_num(np.asarray(values, dtype=float), nan=0.0) > 0.0).astype(float)


def _rule_node(node: int) -> Any:
    """The rule at ``node`` on a rule's history frame ``[W, L_2, ..., L_k]``."""
    if node == 1:
        return lambda h: np.asarray(h["W1"], dtype=float).astype(int)
    return lambda h: _flag(h[f"L{node}"]).astype(int)


def _policy_node(node: int) -> Stochastic:
    """The known policy at ``node`` on a policy frame, which carries the earlier treatments."""

    def density(frame: Any) -> np.ndarray:
        if node == 1:
            p = 0.3 + 0.4 * np.asarray(frame["W1"], dtype=float)
        else:
            previous = np.nan_to_num(np.asarray(frame[f"A{node - 1}"], dtype=float), nan=0.0)
            p = 0.2 + 0.4 * _flag(frame[f"L{node}"]) + 0.2 * previous
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, "q", density_kind="known")


def regimens() -> dict[str, Any]:
    """Always, never, the rule and the known policy, in :data:`LABELS` order."""
    nodes = range(1, LAST + 1)
    return {
        "always": 1,
        "never": 0,
        "rule": DynamicRegimen("rule", tuple(_rule_node(k) for k in nodes), rule_kind="known"),
        "policy": DynamicRegimen("policy", tuple(_policy_node(k) for k in nodes)),
    }


def risk_name(label: str, horizon: int) -> str:
    """The reported name of a regimen's risk by ``horizon``."""
    return f"risk_regimen[{label} @ t={horizon}]"


def contrast_name(label: str, horizon: int) -> str:
    """The reported name of a regimen against never by ``horizon``."""
    return f"ate_regimen[{label} vs {REFERENCE} @ t={horizon}]"


def estimand_names() -> tuple[str, ...]:
    """Every estimate a fit reports, in the fit's order: 12 risks, then 9 contrasts."""
    times = range(1, LAST + 1)
    return (
        *(risk_name(label, t) for label in LABELS for t in times),
        *(contrast_name(label, t) for label in CONTRASTED for t in times),
    )


def target_of(name: str) -> law.Target:
    """The linear target of one reported name."""
    for t in range(1, LAST + 1):
        for label in LABELS:
            if name == risk_name(label, t):
                return law.Target(((1.0, label, t),))
            if label != REFERENCE and name == contrast_name(label, t):
                return law.Target(((1.0, label, t), (-1.0, REFERENCE, t)))
    raise KeyError(name)


def truths() -> dict[str, float]:
    """The exact value of every estimate, by reported name."""
    return {name: target_of(name).value(law.LAW) for name in estimand_names()}


ESTIMANDS = {"multinode": estimand_names()}
TRUTH = {"multinode": truths()}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


#: Cell label -> the reported name it reads (plan X70 section 5): the always, rule and policy
#: contrasts against never and always's risk at the last node.
ARMS: dict[str, str] = {
    "contrast": contrast_name("always", LAST),
    "rule": contrast_name("rule", LAST),
    "policy": contrast_name("policy", LAST),
    "risk": risk_name("always", LAST),
}
#: The six calibration arms, each with the shrunken-SE and noise controls.
CALIBRATION_LABELS = (
    "multinode_contrast",
    "multinode_rule",
    "multinode_policy",
    "multinode_risk",
    "crossfit_contrast",
    "crossfit_policy",
)
CONTROLLED_LABELS = CALIBRATION_LABELS
CONTRAST_LABEL = "multinode_contrast"
JOINT_LABELS = ("multinode_curve",)
RATE_SIZES = (1_000, 2_000, 8_000)
LADDER_LABELS = ("multinode_contrast", "multinode_policy")
DIFFERENCE_FITS = ("multinode",)
DIFFERENCE_LABELS = ("multinode_contrast", "multinode_policy")
DIFFERENCE_ARMS = ("paired_difference", "hazard_fit", "iterated_fit", "isotonic_difference")

#: ``(cell label, configuration)`` of every robustness arm: eleven arms on the five
#: configurations' fits.  The rule and policy ``outcome_correct`` arms are where a defect of
#: the slot values shows: a correct mechanism repairs it, and on the static contrast the held
#: slot, the current level and the level plan coincide for every follower.
DOUBLE_ROBUST_ARMS: tuple[tuple[str, str], ...] = (
    *(
        ("multinode_contrast", configuration)
        for configuration in (
            "both_correct",
            "outcome_correct",
            "mechanism_correct",
            "continuation_wrong",
            "both_wrong",
        )
    ),
    ("multinode_rule", "both_correct"),
    ("multinode_rule", "outcome_correct"),
    *(
        ("multinode_policy", configuration)
        for configuration in (
            "both_correct",
            "outcome_correct",
            "mechanism_correct",
            "continuation_wrong",
        )
    ),
)

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        cell
        for label in CALIBRATION_LABELS
        for cell in _cells(label, "correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "double_robustness": tuple(
        f"{label}__{configuration}" for label, configuration in DOUBLE_ROBUST_ARMS
    ),
    "targeting_necessity": _cells(CONTRAST_LABEL, "targeted", "untargeted"),
    "root_n_and_efficiency": tuple(
        cell
        for label in LADDER_LABELS
        for cell in _cells(label, *(f"n_{size}" for size in RATE_SIZES))
    ),
    "root_n_rate": tuple(
        cell for label in LADDER_LABELS for cell in _cells(label, "empirical_sd", "reported_se")
    ),
    "power": _cells(CONTRAST_LABEL, "alternative"),
    "simultaneous_coverage": tuple(
        cell
        for label in JOINT_LABELS
        for cell in _cells(label, "simultaneous_band", "pointwise_joint_control")
    ),
    "iterated_mean_difference": (
        *(cell for label in DIFFERENCE_LABELS for cell in _cells(label, *DIFFERENCE_ARMS)),
        *(f"{fit}__iterated_curve_decrease" for fit in DIFFERENCE_FITS),
    ),
}

STUDY = StudyRecord(
    name="covariates after baseline beside a treatment at several nodes on the hazard path",
    slug="hazard-time-varying-multinode",
    artifacts=ROOT / "tests" / "canonical" / "hazard_time_varying_multinode",
    document="docs/technical-reference/method-evidence/hazard-time-varying-multinode.md",
    anchor="a-treatment-at-several-nodes-with-covariates-after-baseline",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_hazard_time_varying_multinode.py",
        "tests/studies/hazard_time_varying_multinode_properties.py",
        "tests/studies/hazard_time_varying_multinode_law.py",
        "tests/studies/canonical_hazard_time_varying.py",
        "tests/studies/hazard_time_varying_properties.py",
        "tests/studies/hazard_time_varying_law.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/longitudinal/hazard_backward.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_hazard_time_varying_multinode",
    properties_module="tests.studies.hazard_time_varying_multinode_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "a binary treatment at three nodes and a binary covariate before the "
    "treatment at nodes 2 and 3; one backward pass per regimen, each node evaluated at the "
    "observed history (slots: the regimen's values of the current treatment; hazards predicted "
    "at the observed earlier treatments with the current one set)",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "tol": TOL,
    "fits": dict(FITS),
    "outcome_learner": "unpenalized logistic regression on W1, W2, L_k, A_{k-1} and A_k",
    "pseudo_learner": "the weighted cell mean over every continuation column but the earlier L "
    "columns",
    "treatment_learner": "unpenalized logistic regression on W1, W2, L_k and A_{k-1}",
    "censoring_learner": "unpenalized logistic regression on W1, W2, L_k and A_k",
    "regimens": {
        "always / never": "every treatment set to 1 or 0",
        "rule": "treat when W1 = 1 at node 1 and when L_k = +1 after",
        "policy": "P(A_1 = 1) = 0.3 + 0.4 W1; P(A_k = 1) = 0.2 + 0.4 [L_k = +1] + 0.2 A_{k-1}",
    },
}

#: The support rows the registration adds (plan X70 section 5), each with the cells that
#: license it.  They are not in :data:`cleverly._support_rows.LONGITUDINAL_ROWS` until the
#: declared run passes, so a (b2) fit reads ``experimental`` until then.
G63_MULTINODE_CELLS: dict[str, tuple[str, ...]] = {
    "G63.multinode.in_sample": (
        "multinode_contrast",
        "multinode_risk",
        "multinode_policy",
        "multinode_curve",
    ),
    "G63.multinode.cross_fitted": ("crossfit_contrast", "crossfit_policy"),
}

#: Every nuisance configuration: ``(hazard correct, continuation correct, mechanism correct)``.
#: The wrong mechanism drops ``L_k`` from the treatment and the retention models.
CONFIGURATIONS: dict[str, tuple[bool, bool, bool]] = {
    "both_correct": (True, True, True),
    "outcome_correct": (True, True, False),
    "mechanism_correct": (False, False, True),
    "continuation_wrong": (True, False, True),
    "both_wrong": (False, False, False),
}


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The learners of one nuisance configuration (:data:`CONFIGURATIONS`)."""
    hazard, continuation, mechanism = CONFIGURATIONS[configuration]
    return {
        "outcome_learner": law.hazard_learner(hazard),
        "pseudo_learner": law.continuation_learner(continuation),
        "treatment_learner": law.treatment_learner(mechanism),
        "censoring_learner": law.censoring_learner(mechanism),
    }


FitRecord = held.FitRecord
HeldCellError = held.HeldCellError
refuse_held_cells = held.refuse_held_cells
recorded = held.recorded


def _estimator(fit_label: str, configuration: str, simultaneous: bool) -> LTMLE:
    return LTMLE(
        regimens(),
        reference=REFERENCE,
        n_folds=FITS[fit_label],
        simultaneous=simultaneous,
        survival_model="hazard",
        targeting=ALGORITHM,
        tol=TOL,
        random_state=0,
        **learners(configuration),
    )


def fit_recorded(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    weights: str | None = None,
) -> tuple[Any, FitRecord]:
    """The subject's fit of one fit label's sample, with its :class:`FitRecord`.

    A fit with a held cell raises :class:`HeldCellError` (:func:`refuse_held_cells`).
    """
    result, record = recorded(
        lambda: _estimator(fit_label, configuration, simultaneous).fit(
            frame, **law.LAW.columns(weights=weights)
        )
    )
    refuse_held_cells(result)
    return result, record


def fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    weights: str | None = None,
) -> Any:
    """The subject's fit of one fit label's sample."""
    return fit_recorded(
        fit_label, frame, configuration=configuration, simultaneous=simultaneous, weights=weights
    )[0]


def untargeted_fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "mechanism_correct",
    weights: str | None = None,
) -> Any:
    """The untargeted backward pass of the same fit: every walk skipped."""
    with untargeted():
        return fit(fit_label, frame, configuration=configuration, weights=weights)


def exact_fit(configuration: str = "both_correct", *, targeted: bool = True) -> Any:
    """The in-sample fit of the law as data: every support point once, weighted by its mass."""
    frame = law.LAW.exact_frame()
    if targeted:
        return fit("multinode", frame, configuration=configuration, weights="wt")
    return untargeted_fit("multinode", frame, configuration=configuration, weights="wt")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    del scenario
    return law.LAW.draw(n, seed), dict(TRUTH["multinode"])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, []


def initial_estimates(result: Any) -> dict[str, float]:
    """Node 1's initial plug-in of every reported name, from each regimen's walk record."""
    levels: dict[tuple[str, int], float] = {}
    for regimen_fit in result.fits.values():
        record = regimen_fit.hazard
        for (label, _, horizon), value in zip(record.targets, record.initial_levels, strict=True):
            levels[(label, horizon)] = float(value)
    out: dict[str, float] = {}
    for t in range(1, LAST + 1):
        for label in LABELS:
            out[risk_name(label, t)] = levels[(label, t)]
        for label in CONTRASTED:
            out[contrast_name(label, t)] = levels[(label, t)] - levels[(REFERENCE, t)]
    return out


def walk_steps(result: Any) -> int:
    """The accepted steps over every node walk of every regimen of a fit."""
    records = {id(regimen_fit.hazard): regimen_fit.hazard for regimen_fit in result.fits.values()}
    return sum(int(record.n_iter) for record in records.values())


def subject_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's primary rows, each estimate on the identity scale."""
    initials = initial_estimates(result)
    rows: list[dict[str, Any]] = []
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


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result = fit(scenario, frame)
    return sample, truth_rows, subject_rows(result, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of every scenario, its truth rows and the subject's rows."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    frames = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=frames
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by scenario."""
    return TRUTH


# ------------------------------------------------------------------ the smoke gate

#: Where the smoke gate writes its record.
SMOKE_OUTPUT = STUDY.artifacts / "smoke.json"
#: The cost table's seconds per fit at n = 2,000 on one thread (plan X70, "(b2) cost inputs"),
#: estimated from (b1)'s fits.  A smoke fit slower than :data:`COST_FACTOR` times its label's
#: entry fails the gate, so the cost is re-declared before the runner brief.
COST_SECONDS: dict[str, float] = {"multinode": 1.3, "crossfit": 1.8}
COST_FACTOR = 2.0


def _smoke_cell(fit_label: str, configuration: str, replicate: int) -> dict[str, Any]:
    """One smoke fit on the pilot stream: its wall time, steps and the gate's readings."""
    seed = stream_seed(STUDY, "pilot", "smoke", fit_label, configuration, replicate)
    frame = law.LAW.draw(PRIMARY_N, seed)
    start = time.perf_counter()
    try:
        result, record = fit_recorded(fit_label, frame, configuration=configuration)
    except HeldCellError:
        return {"held": True, "failure": "", "seconds": time.perf_counter() - start}
    except Exception as error:  # a failure is a smoke finding, recorded with its class
        return {"held": False, "failure": type(error).__name__, "seconds": np.nan}
    seconds = time.perf_counter() - start
    scores = [
        float(node.relative_score)
        for regimen_fit in result.fits.values()
        for node in regimen_fit.hazard.nodes
    ]
    return {
        "held": False,
        "failure": "",
        "seconds": seconds,
        "converged": bool(record.converged),
        "largest_node_score": max(scores),
        "walk_steps": walk_steps(result),
        "unexpected_warnings": list(record.unexpected_warnings),
    }


def SMOKE_GATE(
    *,
    replicates: int = SMOKE_REPLICATES,
    cells: tuple[tuple[str, str], ...] | None = None,
    output: Path | None = SMOKE_OUTPUT,
) -> dict[str, Any]:
    """Fit ``replicates`` draws of each fit label and configuration and gate them.

    Every smoke fit must converge with every node walk's relative score at most :data:`TOL`,
    hold no cell, raise nothing and take at most :data:`COST_FACTOR` times its fit label's
    :data:`COST_SECONDS`.  The record holds each cell's wall times and node-walk step
    counts and is written to ``output`` (``smoke.json`` beside the artifacts by default).

    Parameters
    ----------
    replicates : int
        The draws per cell.
    cells : tuple of tuple of str, optional
        The ``(fit label, configuration)`` cells; every in-sample configuration and the
        cross-fitted ``both_correct`` by default.
    output : Path, optional
        Where the record is written; ``None`` writes nothing.

    Returns
    -------
    dict
        The record, with ``cleared``: whether every smoke fit met the gate.
    """
    chosen = cells or (
        *(("multinode", configuration) for configuration in CONFIGURATIONS),
        ("crossfit", "both_correct"),
    )
    record: dict[str, Any] = {"replicates": replicates, "cells": {}}
    cleared = True
    for fit_label, configuration in chosen:
        runs = [_smoke_cell(fit_label, configuration, r) for r in range(replicates)]
        ok = all(
            not run["held"]
            and not run["failure"]
            and run.get("converged", False)
            and run.get("largest_node_score", np.inf) <= TOL
            and run["seconds"] <= COST_FACTOR * COST_SECONDS[fit_label]
            for run in runs
        )
        cleared = cleared and ok
        record["cells"][f"{fit_label}/{configuration}"] = {"cleared": ok, "runs": runs}
    record["cleared"] = cleared
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    return record


__all__ = ["SMOKE_GATE", "STUDY", "TRUTH", "draw_and_fit", "fit"]
