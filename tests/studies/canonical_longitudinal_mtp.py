"""Registered evidence for longitudinal modified treatment policies (``longitudinal-mtp``).

The subject is longitudinal TMLE with a modified treatment policy at every node, on the
laws of :mod:`tests.studies.longitudinal_mtp_common`, paired with R ``lmtp`` 1.5.4.  Four
primary scenarios:

==============================  ===========================================================
scenario                        subject and pairing
==============================  ===========================================================
``mtp_continuous``              the continuous law, one fold.  Plans ``natural``, ``up``
                                (``Shift(0.5, cap=5.5)`` at both nodes), ``scale at 2``
                                (``Scale(1.25, cap=5.5)`` at node 2) and ``up then history``
                                (a node-2 shift that reads the earlier dose).  Paired exactly
``mtp_continuous_crossfit``     the same plans at five folds.  A reporting comparison
``mtp_categorical``             integer levels ``0..5``, the policy of ``lmtp``'s
                                ``man/lmtp_tmle.Rd`` Example 2.1 at both nodes and its
                                ``L_t``-gated variant (Example 2.3 shape).  Paired exactly
``rr_tilt``                     binary nodes, ``RiskRatioTilt(0.25)`` at both nodes against
                                ``lmtp::ipsi(0.25)`` by proportional replication over four
                                copies.  Asymmetric, so the pair sees the branch order.  Paired
                                exactly
==============================  ===========================================================

**The same ratio array on both sides.**  ``lmtp`` is handed the per-node ratio this package
computed, :attr:`~cleverly.longitudinal.RegimenFit.node_ratio`, and the shifted value of each
node, both written beside the replicate data (columns ``ratio__<plan>__<node>`` and
``shift__<plan>__<node>__<copy>``).  No side estimates a ratio the other does not see, so the
paired difference measures the recursion and the fluctuation alone.  The continuous nodes read
an oracle binned density (:class:`~tests.studies.longitudinal_mtp_common.OracleDoseHazard`,
``DENSITY_BINS`` bins, or ``MECHANISM_ONLY_BINS`` where the outcome regressions are wrong); the
categorical and binary nodes read a saturated multinomial.

**Learners.**  The node regressions match the comparator's ``SL.glm``: a quasibinomial GLM at
the last node and least squares at the first.  ``lmtp`` sees the same columns: the history, the
earlier treatment and the current one (factor-coded on the categorical law, so its dummies
span the drop-first indicators this package uses).

**Pairing reading.**  At one fold the designs, the ratios, the learners and the intercept
fluctuation with the cumulative ratio in the loss weight match, so ``mtp_continuous``,
``mtp_categorical`` and ``rr_tilt`` are exactness witnesses: a pre-declaration smoke run
requires every paired difference below :data:`SMOKE_GATE`, with no prediction at ``lmtp``'s
bound of ``1e-5``.  At five folds ``lmtp`` fits each node's fluctuation on the training rows of
each fold (``R/tmle.R``, ``estimate_tmle``) and this package solves one pooled fluctuation per
node over every row (Díaz et al. 2023, Section 5.2, Step 3; ``lmtp`` upstream 9996b04 records
the training-fold update as a defect), so the ``mtp_continuous_crossfit`` pairs compare two
constructions.  They are declared ``reporting`` before any run: a red cross-fitted paired row
is read as that construction difference and gets the owner row
``mtp-crossfit-construction`` in "Red-cell owners", with no budget, margin or law change, and
it does not block the merge.  A red primary truth row, or a red one-fold paired row, blocks
the merge and gets a diagnosis.

**Properties.** :mod:`tests.studies.longitudinal_mtp_properties` declares every family.

Publication policy was ``gated`` and is ``reporting`` after the re-registration below.  The
red-cell route is declared before any run: a red
property cell that reads as a finite-sample limit is re-registered under
``publication_policy="reporting"`` with an owner row in "Red-cell owners" before any re-run.  A
replication that raises is never redrawn.  The shared harness refuses a cell that lost a
replication, so a raise stops the run; the failing cell then drops to its red-cell owner with
its failure count stated as a page limit, and the run repeats without it with no other change.
This is the rule ``clustered-cross-fitted-ltmle``, ``cross-fitted-longitudinal-msm`` and
``stochastic-categorical-ltmle`` declare.  No budget, margin, law, learner, fold count or seed
changes after a verdict is seen.

**Re-registration.**  The declared run (HEAD ``c094fffd``, run 2026-10-07) passed
44 of 44 truth tests and 22 of 22 paired tests, and 33 of 40 property cells.  The seven red
cells and their readings:

- ``interval_calibration`` of ``up``, ``msm_mtp`` and ``randomized_mtp``, and both
  ``crossfit_overfitting`` cells, read the structural limit of a fixed-bin density ratio.  The
  quantile edges leave wide tail bins, where the binned ratio errs by up to 10 on a probe of
  4,000 rows, so the influence curve's spread is inflated (0.523 against 0.413 for the
  efficient curve there).  The intervals are conservative (coverage 0.9565 to 0.981), and the
  point estimates pass their bias rule.  Owner ``binned-density-tail``.
- ``interval_calibration`` of ``categorical_mtp`` and the n = 500 control rung read as finite
  sample: the categorical efficiency ratio falls from 1.07 at n = 2,000 to 1.02 at n = 8,000
  with saturated learners, and the rung's coverage interval ends at 0.8988 against 0.90.
  Owner ``mtp-longitudinal-finite-sample``.

The design does not change after the verdict.  The study is re-registered under ``reporting``,
and the run repeats with the same seeds and no other change.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.interventions.policy import lazy_frame, policy_branches
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.discrete_law_longitudinal_multivalue import CellProbabilities
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_mtp_common as common
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

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261041
RESAMPLING_SEED = 2026104101

CONTINUOUS = "mtp_continuous"
CROSSFIT = "mtp_continuous_crossfit"
CATEGORICAL = "mtp_categorical"
TILT = "rr_tilt"
N_FOLDS = 5
LEARNER_FOLDS = 2
DENSITY_BINS = 80
#: The bins of a fit whose outcome regressions are wrong, so the mechanism alone carries the
#: estimate.  The binned density's error then enters the bias at first order: at 80 bins the
#: mean standardized bias of the mechanism-only arms was 0.54 to 0.60, and at 160 bins 0.21 to
#: 0.24 (100 draws of 8,000 rows each).  ``tests/unit/test_longitudinal_mtp_design.py``
#: records the 320-bin measurement.  Where the outcome is also right, the density error enters
#: only through a product, and 80 bins suffice.
MECHANISM_ONLY_BINS = 320
#: Bounds wide enough that no cumulative probability of these laws reaches them.
G_BOUNDS = (0.001, 1.0)
#: The pre-declaration smoke gate on the one-fold pairs.
SMOKE_GATE = 1e-6
#: The replication factor of the risk-ratio tilt's pairing.  ``RiskRatioTilt(0.25)`` keeps the
#: treatment on one copy in four and sets it to zero on the other three.  At 0.5 the two
#: branches weigh the same, so a pairing there cannot see a swap of the branch weights.
TILT_COPIES = 4
REFERENCE = "natural"

LABELS: dict[str, tuple[str, ...]] = {
    CONTINUOUS: common.CONTINUOUS_LABELS,
    CROSSFIT: common.CONTINUOUS_LABELS,
    CATEGORICAL: common.CATEGORICAL_LABELS,
    TILT: common.TILT_LABELS,
}


def names(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {REFERENCE}]" for label in labels if label != REFERENCE
    )


ESTIMANDS: dict[str, tuple[str, ...]] = {scenario: names(LABELS[scenario]) for scenario in LABELS}


def _truth_of(scenario: str) -> dict[str, float]:
    if scenario in {CONTINUOUS, CROSSFIT}:
        means = {label: common.continuous_truth(label) for label in common.CONTINUOUS_LABELS}
    elif scenario == CATEGORICAL:
        means = {
            label: common.CATEGORICAL_LAW.mean(common.CATEGORICAL_PLANS[label])
            for label in common.CATEGORICAL_LABELS
        }
    else:
        means = {
            label: common.BINARY_LAW.mean(common.TILT_PLANS[label]) for label in common.TILT_LABELS
        }
    out = {f"ey_regimen[{label}]": value for label, value in means.items()}
    for label, value in means.items():
        if label != REFERENCE:
            out[f"ate_regimen[{label} vs {REFERENCE}]"] = value - means[REFERENCE]
    return out


TRUTH: dict[str, dict[str, float]] = {scenario: _truth_of(scenario) for scenario in LABELS}

#: The joint label of the band cell, over the continuous scenario's estimands.
BAND_LABEL = "primary"
#: The labels of ``interval_calibration``: the continuous ``up`` contrast on the density route
#: and on the classifier route, the categorical and vector policies, a randomized policy, the
#: cumulative risk at t = 2 of a survival plan, and the ``dose`` coefficient of a working
#: model over the continuous plans.
CALIBRATION_LABELS = (
    "up",
    "classifier_route",
    "categorical_mtp",
    "vector_node",
    "randomized_mtp",
    "survival_mtp_h2",
    "msm_mtp",
)

#: Every published property cell, by family.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"up__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "root_n_and_efficiency": tuple(f"up__n_{size}" for size in (500, 2_000, 8_000)),
    "root_n_rate": ("up__empirical_sd", "up__reported_se"),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("up__sharp_null",),
    "power": ("up__alternative",),
    "targeting_necessity": ("up__targeted", "up__untargeted"),
    "inverse_necessity": ("history__declared_inverse", "history__inverse_dropped_control"),
    "crossfit_overfitting": ("cross_fitted_mtp_ltmle", "in_sample_control"),
    "simultaneous_coverage": (
        f"{BAND_LABEL}__simultaneous_band",
        f"{BAND_LABEL}__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="longitudinal modified treatment policies",
    slug="longitudinal-mtp",
    artifacts=ROOT / "tests" / "canonical" / "longitudinal_mtp",
    document="docs/technical-reference/method-evidence/longitudinal-modified-treatment-policies.md",
    anchor="longitudinal-modified-treatment-policies",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-mtp-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_longitudinal_mtp.py",
        "tests/studies/longitudinal_mtp_properties.py",
        "tests/studies/longitudinal_mtp_common.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_survival.py",
        "tests/discrete_law_longitudinal_mtp.py",
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
        "tests/canonical/lmtp_mtp_adapter.R",
        "tests/canonical/longitudinal_mtp_runner.R",
    ),
    runner_module="tests.studies.canonical_longitudinal_mtp",
    properties_module="tests.studies.longitudinal_mtp_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_tmle and theta_dr on an LmtpTask with the shifted values and the per-node "
        "density ratios this package computed (node_ratio), written beside the replicate "
        "data; no lmtp density ratio is fitted. The risk-ratio tilt is realised by copying "
        f"each unit {TILT_COPIES} times (one copy keeps, three set to zero), with the unit's "
        "ratio on every copy and the unit-mean eif"
    ),
    "reference_adapter": "tests/canonical/lmtp_mtp_adapter.R",
    "crossfit_pairing": "reporting: per-fold training fluctuation against the pooled one",
}

CONFIGURATION = {
    "construction": "ordinary and cross-fitted",
    "outcome_kind": "end_of_study",
    "n_folds": {CONTINUOUS: 1, CROSSFIT: N_FOLDS, CATEGORICAL: 1, TILT: 1},
    "learner_folds": LEARNER_FOLDS,
    "density_bins": DENSITY_BINS,
    "density_bins_mechanism_only": MECHANISM_ONLY_BINS,
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": "LinearRegression",
    "continuous_mechanism": "oracle binned density of the truncated normal dose",
    "categorical_mechanism": "saturated multinomial",
    "plans": {scenario: list(labels) for scenario, labels in LABELS.items()},
    "reference": REFERENCE,
    "reference_density_ratios": "this package's node_ratio, supplied to both",
    "tilt_copies": TILT_COPIES,
}


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def regimens(scenario: str) -> dict[str, Any]:
    if scenario in {CONTINUOUS, CROSSFIT}:
        return common.continuous_regimens()
    if scenario == CATEGORICAL:
        return common.categorical_regimens()
    return common.tilt_regimens()


def density_bins(configuration: str) -> int:
    """The bin count of a configuration: finer where the mechanism alone carries the fit."""
    return MECHANISM_ONLY_BINS if configuration == "mechanism_correct" else DENSITY_BINS


def edges_of(frame: pd.DataFrame, bins: int = DENSITY_BINS) -> tuple[np.ndarray, np.ndarray]:
    """The whole-sample bin edges each continuous node's density reads."""
    return (
        bin_edges(frame["A1"].to_numpy(dtype=float), bins),
        bin_edges(frame["A2"].to_numpy(dtype=float), bins),
    )


def _learners(scenario: str, frame: pd.DataFrame, configuration: str) -> tuple[Any, Any, Any]:
    if scenario in {CONTINUOUS, CROSSFIT}:
        extra = 1 if "U" in frame else 0
        return common.continuous_learners(
            configuration, edges_of(frame, density_bins(configuration)), extra=extra
        )
    if configuration != "primary":
        raise ValueError(f"no {configuration!r} configuration on {scenario!r}")
    return QuasiBinomialGLM(), LinearRegression(), CellProbabilities()


def fit(
    frame: pd.DataFrame,
    scenario: str = CONTINUOUS,
    *,
    configuration: str = "primary",
    n_folds: int | None = None,
    plans: Mapping[str, Any] | None = None,
    reference: str = REFERENCE,
    simultaneous: bool = False,
    ratio: str = "density",
    treatment_learner: Any = None,
) -> Any:
    """The one fit every primary and property row of the continuous law reads."""
    outcome, pseudo, treatment = _learners(scenario, frame, configuration)
    folds = n_folds if n_folds is not None else (N_FOLDS if scenario == CROSSFIT else 1)
    baseline = ["W", *(("U",) if "U" in frame else ())]
    continuous = ["A1", "A2"] if scenario in {CONTINUOUS, CROSSFIT} else []
    return LTMLE(
        regimens(scenario) if plans is None else dict(plans),
        reference=reference,
        outcome_learner=clone(outcome),
        pseudo_learner=clone(pseudo),
        treatment_learner=clone(treatment if treatment_learner is None else treatment_learner),
        n_folds=folds,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        density_bins=density_bins(configuration),
        ratio=ratio,
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
        continuous_treatment=continuous,
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario in {CONTINUOUS, CROSSFIT}:
        return common.sample_continuous(n, seed), dict(TRUTH[scenario])
    if scenario == CATEGORICAL:
        return common.CATEGORICAL_LAW.sample(n, seed), dict(TRUTH[scenario])
    if scenario == TILT:
        return common.BINARY_LAW.sample(n, seed), dict(TRUTH[scenario])
    raise KeyError(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initial_mean(regimen_fit: Any) -> float:
    """The untargeted plug-in: the first node's initial carried value."""
    first = regimen_fit.steps[0]
    if first.initial_by_arm is None:
        return float(np.mean(first.initial))
    return float(np.mean(np.sum(regimen_fit.policy[0] * first.initial_by_arm, axis=1)))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    labels = LABELS[scenario]
    initials = {f"ey_regimen[{label}]": initial_mean(result.fits[label]) for label in labels}
    for label in labels:
        if label != REFERENCE:
            initials[f"ate_regimen[{label} vs {REFERENCE}]"] = (
                initials[f"ey_regimen[{label}]"] - initials[f"ey_regimen[{REFERENCE}]"]
            )
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


def pairing_columns(result: Any, frame: pd.DataFrame, scenario: str) -> dict[str, np.ndarray]:
    """The shifted values and per-node ratios this fit used, for the comparator to read.

    ``shift__<plan>__<node>__<copy>`` holds the value each copy of a unit takes at a node: one
    copy for a deterministic policy, and at the risk-ratio tilt one copy per randomizer branch
    (keep, set to zero).  ``ratio__<plan>__<node>`` is the fit's
    :attr:`~cleverly.longitudinal.RegimenFit.node_ratio`.
    """
    data = result.data
    out: dict[str, np.ndarray] = {}
    for label, plan in regimens(scenario).items():
        fitted = result.fits[label]
        assert fitted.node_ratio is not None
        nodes = plan.plan if hasattr(plan, "plan") else (plan, plan)
        for time, policy in enumerate(nodes, start=1):
            out[f"ratio__{_slug(label)}__{time}"] = fitted.node_ratio[:, time - 1]
            source = frame[f"A{time}"].to_numpy()
            frame_at = lazy_frame(lambda time=time: data.policy_frame(time))
            branches = policy_branches(policy)
            copies = TILT_COPIES if scenario == TILT else 1
            allocation = _allocate(branches, copies, f"plan {label!r} at node {time}")
            for copy in range(copies):
                branch = allocation[copy]
                values = branch.assign(
                    source.astype(float),
                    frame_at,
                )
                out[f"shift__{_slug(label)}__{time}__{copy + 1}"] = np.asarray(values, dtype=float)
    return out


def _allocate(branches: Any, copies: int, where: str) -> list[Any]:
    """The branch each copy of a unit takes: ``p * copies`` copies per branch, in order."""
    out: list[Any] = []
    for probability, branch in branches:
        count = probability * copies
        if abs(count - round(count)) > 1e-12:
            raise AssertionError(f"{where} has a branch weight {probability} off 1/{copies}")
        out.extend([branch] * round(count))
    if len(out) != copies:
        raise AssertionError(f"{where} allocates {len(out)} copies, not {copies}")
    return out


def _replicate(
    scenario: str, replicate: int, n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit(frame, scenario)
    sample = frame.copy()
    for name, values in pairing_columns(result, frame, scenario).items():
        sample[name] = values
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
    # Replicate-major, so every scenario of one replicate is contiguous in the sample archive
    # and the comparator's runner reads them as one group.
    payloads = [
        (scenario, replicate, n) for replicate in range(replicates) for scenario in STUDY.scenarios
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
