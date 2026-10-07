"""Registered evidence for known stochastic categorical policies at longitudinal nodes.

The subject is ordinary (one-fold) longitudinal TMLE on the two-node three-level law of
:mod:`tests.discrete_law_longitudinal_multivalue`, with the policies of
:mod:`tests.discrete_law_longitudinal_policy` fixed before any run.  Three plans are fitted:
``low`` (the label at both nodes, the reference), ``mix`` (a policy at both nodes) and
``taper`` (``high`` at the first node, a policy at the second).  The truth is the policy
g-formula on the generating masses, constant across replications.

**Learners.**  The node regressions match the comparator's ``SL.glm``: a quasibinomial GLM
at the last node and least squares at the first, where the target is fractional.  The
mechanism is the law's own (:class:`~tests.studies.categorical_longitudinal_common.
KnownCategoricalMechanism`), handed to both implementations.  ``learner_folds=2`` has no
effect on these single learners and is stated for the record.

**Primary.** 2,000 replications at n = 2,000, paired with R ``lmtp`` 1.5.4.  ``lmtp`` takes
one shifted value per unit, so ``tests/canonical/lmtp_policy_adapter.R`` realises each policy
exactly by replicating every unit four times (every policy probability is a multiple of one
quarter) and supplies the exact per-node ratio ``q / g`` at the natural arm.  For ``mix`` both
designs span the same columns and the learners match, so the paired difference is an
exactness witness: a pre-declaration smoke run requires it below ``1e-6``.  For ``taper`` this
package stratifies the second node on the first-node label and ``lmtp`` pools over it, so
that pair is read under the default margins.  ``low`` differs the same way, so neither ``low``
nor either contrast is an exactness witness.  In the committed run their mean absolute paired
difference is about ``9e-3`` with a largest of 0.042, and ``taper``'s is ``2.3e-3`` with a
largest of 0.017.  The study page states this limit.  ``low`` runs through the shared
``lmtp_tmle_with_folds`` exactly as ``canonical-categorical-ltmle`` runs it.

**Properties.** ``tests.studies.stochastic_categorical_ltmle_properties`` declares every
family.

Publication policy is ``gated``.  The red-cell route is declared before any run: a red primary
or paired cell blocks the merge and gets a diagnosis.  A red property cell that reads as a
finite-sample limit is re-registered under ``publication_policy="reporting"`` with an owner row
in "Red-cell owners" before any re-run.  A replication that raises is never redrawn.  The
shared harness refuses a cell that lost a replication, so a raise stops the run.  The failing
cell then drops to its red-cell owner with its failure count stated as a page limit, and the
run repeats without it with no other change.  This is the rule ``clustered-cross-fitted-ltmle``
and ``cross-fitted-longitudinal-msm`` declare.  No budget, margin, law, learner, fold count or
seed changes after a verdict is seen.

**Re-registration.**  The declared run (HEAD ``da2dbf92``) passed every truth, paired and
property gate but one: ``power/mix__alternative`` rejected at 0.5575 (99% interval 0.511 to
0.603) against a floor of 0.80.  Its bias and coverage passed.  The contrast is -0.0488 and
its reported standard error at n = 4,000 is about 0.0237.  The cell is underpowered at its
declared size: n = 4,000 was copied from the categorical study, whose contrast is 0.125, and
the exact power here is 0.5365.  The cell reads as a finite-sample limit of that size, so by
the declared route the study is re-registered under ``reporting`` with the owner row
``F1-power-design``, and the run repeats with no other change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_multivalue as law
from tests import discrete_law_longitudinal_policy as policy
from tests import longitudinal_policies as policies
from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_common as categorical
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 2_000
SEED = 20261033
RESAMPLING_SEED = 2026103301

SCENARIO = "categorical_policy_end_of_study"
LABELS = ("low", "mix", "taper")
REFERENCE = "low"
ESTIMANDS = policy.names(LABELS)
TRUTH = {name: policy.TRUTH[name] for name in ESTIMANDS}
G_BOUNDS = categorical.G_BOUNDS
LEARNER_FOLDS = 2
#: The replication factor of the comparator: the least common denominator of the policies.
COPIES = 4
#: The pre-declaration smoke gate on the exactness pair.
SMOKE_GATE = 1e-6

#: The joint label of the band cell.
BAND_LABEL = "primary"
#: The labels of ``interval_calibration``: the policy contrast, the survival policy's risk at
#: t = 2, and the ``dose`` coefficient of a working model over the three plans.
CALIBRATION_LABELS = ("mix", "policy_risk_h2", "msm_policy")

#: Every published property cell, by family.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"mix__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "root_n_and_efficiency": tuple(f"mix__n_{size}" for size in (500, 2_000, 8_000)),
    "root_n_rate": ("mix__empirical_sd", "mix__reported_se"),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("mix__sharp_null",),
    "power": ("mix__alternative",),
    "targeting_necessity": ("mix__targeted", "mix__untargeted"),
    "policy_necessity": ("mix__declared_policy", "mix__uniform_control"),
    "randomizer_projection": ("mix__integrated", "mix__recorded_randomizer"),
    "crossfit_overfitting": ("cross_fitted_policy_ltmle", "in_sample_control"),
    "simultaneous_coverage": (
        f"{BAND_LABEL}__simultaneous_band",
        f"{BAND_LABEL}__pointwise_joint_control",
    ),
}


STUDY = StudyRecord(
    name="ordinary known stochastic categorical longitudinal TMLE",
    slug="stochastic-categorical-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "stochastic_categorical_ltmle",
    document=(
        "docs/technical-reference/method-evidence/stochastic-categorical-longitudinal-tmle.md"
    ),
    anchor="stochastic-categorical-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-stochastic-categorical-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_stochastic_categorical_ltmle.py",
        "tests/studies/stochastic_categorical_ltmle_properties.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal_policy.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_survival.py",
        "tests/discrete_law_competing.py",
        "tests/longitudinal_policies.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_policy_adapter.R",
        "tests/canonical/stochastic_categorical_ltmle_runner.R",
    ),
    runner_module="tests.studies.canonical_stochastic_categorical_ltmle",
    properties_module="tests.studies.stochastic_categorical_ltmle_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "integrated policy estimator by proportional replication: each unit copied "
        f"{COPIES} times, copy c taking its shifted arm from row c of the node's allocation "
        "table, exact per-node ratio q/g at the natural arm, unit-mean eif"
    ),
    "reference_adapter": "tests/canonical/lmtp_policy_adapter.R",
}

CONFIGURATION = {
    "construction": "ordinary",
    "outcome_kind": "end_of_study",
    "cross_fit": False,
    "n_folds": 1,
    "learner_folds": LEARNER_FOLDS,
    "learner_folds_effect": "none for single learners",
    "treatment_levels": list(categorical.LEVELS),
    "regimens": list(LABELS),
    "policies": {
        "mix": "POLICY1 at node 1, POLICY2 at node 2",
        "taper": "label 'high' at node 1, POLICY2 at node 2",
    },
    "reference": REFERENCE,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": "LinearRegression",
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node_policy_ratio",
    "reference_copies": COPIES,
}


def regimens(*, uniform: bool = False) -> dict[str, Any]:
    """The declared plans, or the ``uniform_control`` that draws every arm at one third."""
    if not uniform:
        return policies.regimens(LABELS)
    flat = np.full((2, 3), 1.0 / 3.0)
    flat2 = np.full((2, 3, 2, 3), 1.0 / 3.0)
    return {
        "low": "low",
        "mix": DynamicRegimen(
            "mix",
            (policies.policy_node(flat, 1, "uniform1"), policies.policy_node(flat2, 2, "uniform2")),
        ),
        "taper": DynamicRegimen("taper", ("high", policies.policy_node(flat2, 2, "uniform2"))),
    }


def _learners(configuration: str) -> tuple[Any, Any, Any]:
    if configuration == "primary":
        return QuasiBinomialGLM(), LinearRegression(), categorical.KnownCategoricalMechanism()
    return categorical._learners(configuration)


def fit(
    frame: pd.DataFrame,
    *,
    configuration: str = "primary",
    n_folds: int = 1,
    uniform: bool = False,
    simultaneous: bool = False,
) -> Any:
    """The one fit every primary and property row reads."""
    outcome, pseudo, treatment = _learners(configuration)
    baseline = ["W", *(("U",) if "U" in frame else ())]
    return LTMLE(
        regimens(uniform=uniform),
        reference=REFERENCE,
        outcome_learner=clone(outcome),
        pseudo_learner=clone(pseudo),
        treatment_learner=clone(treatment),
        n_folds=n_folds,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=baseline,
        time_varying=[[], ["L2"]],
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return law.sample(law.PROBS, n, seed), dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, *, simultaneous: bool = False) -> Any:
    return fit(frame, simultaneous=simultaneous)


def initial_mean(regimen_fit: Any) -> float:
    """The untargeted plug-in: the first node's initial value, policy-weighted where it is drawn."""
    first = regimen_fit.steps[0]
    if first.initial_by_arm is None:
        return float(np.mean(first.initial))
    return float(np.mean(np.sum(regimen_fit.policy[0] * first.initial_by_arm, axis=1)))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    initials = {f"ey_regimen[{label}]": initial_mean(result.fits[label]) for label in LABELS}
    for label in LABELS:
        if label != REFERENCE:
            initials[f"ate_regimen[{label} vs {REFERENCE}]"] = (
                initials[f"ey_regimen[{label}]"] - initials[f"ey_regimen[{REFERENCE}]"]
            )
    rows = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
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


def cleverly_rows(
    frame: pd.DataFrame, truth: dict[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return result_rows(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    scenario: str, replicate: int, n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, result_rows(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        (scenario, replicate, n) for scenario in STUDY.scenarios for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
