"""Registered evidence for policies at a held baseline treatment (``point-treatment-survival-policies``).

The subject is ``LTMLE`` on the held survival design with a known stochastic policy (F1) or a
modified treatment policy (X12) at the one treatment decision, and identity nodes after it.
That is Díaz, Williams, Hoffman and Schenck (2023), Definition 1 with
:math:`d_t(a_t, h_t) = a_t` for ``t >= 2``.  In ``lmtp`` it is ``trt`` of length K, K copied
treatment columns with the shifted value at the first and the observed value after it.

==========  ========================================================================  ========
scenario    law and plans                                                             visits
==========  ========================================================================  ========
``policy``  the five-visit binary law; ``q(1 | W) = 0.5 + 0.25 W1 + 0.25 1{W2 >= 2}``   1, 3, 5
            against the natural course
``mtp``     the dose law on ``0..5``; ``lmtp``'s Example 2.1 policy                  1, 3, 5
            ``a - 1 where a - 1 >= 1`` against the natural course
==========  ========================================================================  ========

Each scenario reports both risks and their difference at visits 1, 3 and 5, at n = 2,000 with
1,600 replications, on the wide layout.

**Both sides receive the law's own mechanism.**  This package fits the known treatment and
retention probabilities (:class:`~tests.studies.point_survival_common.KnownPointMechanism`).
``lmtp`` 1.5.4 at ``f04a2b47`` receives the shifted treatment of each copy and the per-node
density ratio computed from the same probabilities, written beside the replicate data, through
``tests/canonical/lmtp_held_survival_adapter.R``; it fits no density ratio.

**The comparator's treatment is K copied columns**, the shifted value at the first and the
observed value at every later one: the identity policy at ``t >= 2``.  ``lmtp``'s own form with
``trt`` of length one evaluates every node's regression at the shifted treatment, which applies
a policy that is not idempotent once per node.  The pre-declaration smoke showed it: with ``trt``
of length one, ``lmtp`` sat about 0.02 above the truth and above this package for the
``minus one`` risk at visit 5, and with the copied columns the two agree to about ``1e-4``.  The policy's
probabilities are multiples of one quarter, so four copies of each unit realise it exactly.  The
sequential regressions are the logistic regression on ``W1``, ``W2`` and the treatment on both
sides (the dose as a factor).  The fluctuations differ by construction, so the paired rows are
read under the default margins.

**The initial estimate is one quantity on the two sides.**  Here ``initial_estimate`` is the
mean of the node-1 regression at the plan's treatment, fitted on pseudo-outcomes that the
later nodes already targeted.  The ``lmtp`` value is its first regression predicted at the
shifted treatment, which ``lmtp`` also fits on the targeted later nodes.  It enters no verdict.

**A separated fit keeps its last iterate.**  ``QuasiBinomialGLM`` keeps its last iterate,
with ``QuasiBinomialSeparationWarning``, when its coefficients diverge while its deviance
settles by R's ``glm.control`` rule.  R's ``glm`` returns such a fit with a warning, and the
comparator fits ``glm``.  A fit whose deviance has not settled still raises and fails the run.
A failure-only scan of every declared primary replication, which read no estimate, found
two such fits, both in replication 1,003 of ``mtp``, and no failed fit.  Its rows are kept and
read as every other row.

Publication policy is ``reporting``.  The red-cell route was declared before any run.
Every red cell is diagnosed before it is routed.  A genuine defect (an algorithm
defect, an inconsistent estimator, or a test or design bug that makes the cell measure the
wrong thing) is fixed and re-run under a fresh declaration commit that records the change.
Nothing changes to make a cell easier to pass: no margin, budget, law or cell moves.
A red paired ``lmtp`` row with no defect stays published red under an owner row named
for its diagnosed reading, as ``RM18-comparator-density`` is.  Any other red cell with no
defect stays published red under ``X13-finite-sample``.  A replication that raises is never
redrawn: a failed replication fails the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import survival_point as law_module
from cleverly.interventions import ModifiedPolicy, Shift, Stochastic
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.canonical_ltmle import regimen_rows
from tests.studies.canonical_point_survival import initial_estimates
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261062
RESAMPLING_SEED = 2026106201
#: Copies of a unit that realise the policy exactly.
COPIES = 4
REPORTED = (1, 3, 5)

POLICY_LAW = common.Scenario("policy", 5, wide=True)
DOSE_LAW = common.Scenario("mtp", 5, dose="discrete", wide=True)
SCENARIOS = {"policy": POLICY_LAW, "mtp": DOSE_LAW}


def _policy_density(frame: Any) -> np.ndarray:
    q = common.policy_probability(frame["W1"], frame["W2"])
    return np.column_stack([1.0 - q, q])


def _minus_one(a: Any, h: Any) -> Any:
    return common.minus_one(a)


NATURAL = Shift(0.0, cap=None)
POLICY = Stochastic(_policy_density, "q", density_kind="known")
MTP = ModifiedPolicy("minus one", apply=_minus_one, policy_kind="known")
REFERENCE = "natural course"
PLAN = {"policy": "baseline policy", "mtp": "baseline minus one"}
REGIMENS: dict[str, dict[str, Any]] = {
    "policy": {REFERENCE: NATURAL, PLAN["policy"]: POLICY},
    "mtp": {REFERENCE: NATURAL, PLAN["mtp"]: MTP},
}


def slug(label: str) -> str:
    """A plan label as it appears in a pairing column name."""
    return label.replace(" ", "_")


def _names(scenario: str) -> tuple[str, ...]:
    plan = PLAN[scenario]
    return (
        *(common.risk_name(label, t) for label in (REFERENCE, plan) for t in REPORTED),
        *(common.contrast_name(plan, REFERENCE, t) for t in REPORTED),
    )


ESTIMANDS = {scenario: _names(scenario) for scenario in SCENARIOS}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in ("policy_t5", "mtp_t5")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
}

STUDY = StudyRecord(
    name="policies at a held baseline treatment",
    slug="point-treatment-survival-policies",
    artifacts=ROOT / "tests" / "canonical" / "point_survival_policies",
    document="docs/technical-reference/method-evidence/point-treatment-survival-policies.md",
    anchor="point-treatment-survival-policies",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-held-policy-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_point_survival_policies.py",
        "tests/studies/point_survival_policies_properties.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/survival_point.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_policy_adapter.R",
        "tests/canonical/lmtp_held_survival_adapter.R",
        "tests/canonical/point_survival_policies_runner.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_point_survival_policies",
    properties_module="tests.studies.point_survival_policies_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_tmle and theta_dr with trt of length K: K copied treatment columns with the "
        "shifted value at the first and the observed value after it (the identity policy at "
        "t >= 2), K event and censoring nodes, the shifted first treatment of four copies "
        "(policy) or one copy (MTP, natural course), and the law's own per-node density ratio "
        "written beside the replicate data"
    ),
    "reference_adapter": "tests/canonical/lmtp_held_survival_adapter.R",
    "pairing": "read under the default margins; the fluctuations differ by construction",
}

CONFIGURATION = {
    "construction": "held point-treatment survival with a policy at the decision",
    "cross_fit": False,
    "outcome_learner": "QuasiBinomialGLM on W1, W2 and the treatment block",
    "mechanism": "the law's own treatment and retention probabilities",
    "copies": COPIES,
    "reported_nodes": list(REPORTED),
}


def _assign(scenario: str, label: str) -> Any:
    if label == REFERENCE:
        if scenario == "mtp":
            return common.shifted(lambda a: a, DOSE_LAW.levels)
        return lambda w1, w2: tuple(law_module.treatment_probabilities(w1, w2)[0])
    if scenario == "policy":
        return common.known_policy
    return common.shifted(common.minus_one, DOSE_LAW.levels)


def _truth(scenario: str) -> dict[str, float]:
    law = SCENARIOS[scenario]
    plans = {label: _assign(scenario, label) for label in REGIMENS[scenario]}
    values = common.truths(law, plans, REFERENCE)
    return {name: values[name] for name in ESTIMANDS[scenario]}


TRUTH = {scenario: _truth(scenario) for scenario in SCENARIOS}


def fit(scenario: str, frame: pd.DataFrame) -> Any:
    """The subject's primary fit of one scenario's sample."""
    law = SCENARIOS[scenario]
    return LTMLE(
        REGIMENS[scenario],
        reference=REFERENCE,
        n_folds=1,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=common.KnownPointMechanism("treatment", law.dose),
        censoring_learner=common.KnownPointMechanism("censoring", law.dose),
    ).fit(law.container(frame))


def pairing_columns(scenario: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The shifted treatment of every copy and the per-node ratio, for the comparator."""
    law = SCENARIOS[scenario]
    name = law.treatment
    a = frame[name].to_numpy(dtype=float)
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    g = law_module.treatment_probabilities(w1, w2, dose=law.dose)
    levels = np.asarray(law.levels, dtype=float)
    observed = np.searchsorted(levels, a)
    rows = np.arange(len(frame))
    stay = 1.0 - np.asarray(law_module._dropout(a, w1, w2, 0.0, law.dose), dtype=float)
    out: dict[str, np.ndarray] = {}
    for label in REGIMENS[scenario]:
        if label == REFERENCE:
            first = np.ones(len(frame))
            copies = [a]
        elif scenario == "policy":
            q = common.policy_probability(w1, w2)
            first = np.where(a == 1.0, q, 1.0 - q) / g[rows, observed]
            ones = np.rint(COPIES * q).astype(int)
            copies = [np.where(c >= COPIES - ones, 1.0, 0.0) for c in range(COPIES)]
        else:
            moved = common.minus_one(levels)
            induced = np.zeros_like(g)
            for code, target in enumerate(moved):
                induced[:, int(np.searchsorted(levels, target))] += g[:, code]
            first = induced[rows, observed] / g[rows, observed]
            copies = [common.minus_one(a)]
        for copy, values in enumerate(copies, start=1):
            out[f"shift__{slug(label)}__{copy}"] = np.asarray(values, dtype=float)
        out[f"ratio__{slug(label)}__1"] = first
        for node in range(2, law.n_times + 1):
            retained = frame[f"C{node}"].to_numpy(dtype=float)
            out[f"ratio__{slug(label)}__{node}"] = np.where(retained == 1.0, 1.0 / stay, 0.0)
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return SCENARIOS[scenario].draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _replicate(
    payload: tuple[int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    replicate, n = payload
    samples: list[pd.DataFrame] = []
    truth_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        frame, truth = draw_scenario(scenario, n, replicate)
        result = fit(scenario, frame)
        sample = frame.rename(columns={"D": "A"}).copy()
        for column, values in pairing_columns(scenario, frame).items():
            sample[column] = values
        sample.insert(0, "row", np.arange(len(sample)))
        sample.insert(0, "replicate", replicate)
        sample.insert(0, "scenario", scenario)
        samples.append(sample)
        truth_rows.extend(
            {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
            for name, value in truth.items()
        )
        rows.extend(
            regimen_rows(
                STUDY,
                result,
                truth,
                initial_estimates(result, ESTIMANDS[scenario]),
                ESTIMANDS[scenario],
                scenario,
                replicate,
                n=len(frame),
            )
        )
    return pd.concat(samples, ignore_index=True), truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every replication of both scenarios, ordered by replication and then by scenario."""
    outcomes = map_parallel(
        _replicate, [((replicate, n),) for replicate in range(replicates)], n_jobs=n_jobs
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit", "pairing_columns"]
