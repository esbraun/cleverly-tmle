r"""Registered evidence for the sequentially doubly robust estimator (``longitudinal-sdr``).

The subject is :class:`~cleverly.longitudinal.SDR`, the estimator of Díaz, Williams, Hoffman
and Schenck (2023, Section 5.3), paired with R ``lmtp`` 1.5.4's ``cf_sdr`` and
``theta_dr(is_sdr = TRUE)``.  Five primary scenarios:

===============================  ==========================================================
scenario                         subject and pairing
===============================  ==========================================================
``sdr_end_of_study``             the censored law of ``make_longitudinal``, plans ``never``,
                                 ``always`` and the dynamic rule, one fold.  Margins
``sdr_end_of_study_crossfit``    the same plans at five folds.  Margins
``sdr_survival``                 the censored survival law of
                                 ``make_longitudinal_survival``, risks at horizons 1 and 2,
                                 one fold.  Margins
``sdr_mtp``                      the continuous-dose law of ``longitudinal-mtp`` and its four
                                 plans, one fold.  Gate G1 (exact)
``sdr_policy``                   the three-level law of ``stochastic-categorical-ltmle`` and
                                 its plans ``low``, ``mix`` and ``taper``, one fold.  ``mix``
                                 is gate G1 (exact); ``low``, ``taper`` and the contrasts read
                                 margins
===============================  ==========================================================

**The same ratio array on both sides.**  ``lmtp`` is handed the per-node ratio this package
divides by, and the value each copy of a unit takes at each node, both written beside the
replicate data (columns ``ratio__<plan>__<node>`` and ``shift__<plan>__<node>__<copy>``).
Column :math:`t` of the ratio is :math:`\pi_t\,1\{C_t = 1\} / (g_t c_t)` at the observed arm,
zero where the unit leaves the plan at node :math:`t`.  :func:`supplied_ratios` builds it from
the fit's own plans and mechanism, and :func:`check_ratios` asserts that its running products
are the partial weights :math:`W_{1,s}` the engine reads.  Every mechanism here is known
(:class:`~tests.studies.canonical_ltmle.KnownLongitudinalMechanism`,
:class:`~tests.studies.categorical_longitudinal_common.KnownCategoricalMechanism`, or the
oracle binned density of ``longitudinal-mtp``), so the ratio does not depend on the fold.
With fold-free ratios the training-fold recursion equals ``lmtp::cf_sdr`` (``docs/
technical-reference/longitudinal-tmle.md``, "Cross-fitting"), and ``lmtp`` never fits a
density ratio (``cf_density_ratios`` is not called).

**Learners.**  The regression at the horizon is a quasibinomial GLM and every earlier one is
least squares, which is ``lmtp``'s ``SL.glm`` with the binomial family at the outcome and the
gaussian family on the transformed outcome.  A transform can leave :math:`[0, 1]`, so the
pseudo learner is unclipped on both sides.

**Pairing reading** (``X4-plan.md`` 6.2, declared before any run).  ``cleverly`` fits a label or
rule node on the units that follow the plan, with no arm column.  ``lmtp`` fits every
uncensored unit with the natural :math:`A_t` in the design.  On a continuous history the two
GLM fits differ, so the paired rows of ``sdr_end_of_study``, ``sdr_end_of_study_crossfit``,
``sdr_survival`` and the label and partial-policy rows of ``sdr_policy`` read the default
``Margins()``.  At an all-MTP or all-policy plan both designs span the same columns, so the
``sdr_mtp`` rows and ``ey_regimen[mix]`` are exactness witnesses (gate G1: below
:data:`SMOKE_GATE` on estimate and standard error in a pre-declaration smoke of three
replicates).  Gate G2 fits saturated cell means on both sides of a discrete law, where the
pooled and stratified fits coincide, at one and five folds
(``tests/canonical/longitudinal_sdr/gates.py``).

**Properties.** :mod:`tests.studies.longitudinal_sdr_properties` declares every family.

**Red-cell route** (declared before any run).  A red primary truth row, a red G1 row or a
failed G2 gate blocks the merge and gets a diagnosis.  A red margins row gets a diagnosis; if
it reads as the stratified-versus-pooled design difference, it is re-registered ``reporting``
with an owner row, declared before any re-run.  A red property cell that reads as a
finite-sample limit is re-registered the same way.  A replication that raises is never
redrawn; the failing cell drops to its owner with its failure count stated as a page limit,
and the run repeats without it.  No budget, margin, law, learner, fold count or seed moves
after a verdict is seen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.datasets import (
    RULE_LABEL,
    make_longitudinal,
    make_longitudinal_survival,
    rule_arm_at_node_two,
)
from cleverly.longitudinal import SDR, DynamicRegimen
from cleverly.longitudinal.regimen import (
    Plan,
    attach_policy_numerators,
    resolve_plans,
    resolve_regimens,
)
from cleverly.longitudinal.sdr import node_factors, partial_weights
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_multivalue as multivalue
from tests import discrete_law_longitudinal_policy as policy_law
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_longitudinal_mtp as mtp_study
from tests.studies import canonical_ltmle_survival as survival_study
from tests.studies import canonical_stochastic_categorical_ltmle as policy_study
from tests.studies import longitudinal_mtp_common as mtp
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.categorical_longitudinal_common import KnownCategoricalMechanism
from tests.studies.evidence.constructions import SDR_TRAINING_FOLD_RECURSION
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.oracle_density_bins import oracle_bins

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261081
RESAMPLING_SEED = 2026108101

END_OF_STUDY = "sdr_end_of_study"
END_OF_STUDY_CROSSFIT = "sdr_end_of_study_crossfit"
SURVIVAL = "sdr_survival"
MTP = "sdr_mtp"
POLICY = "sdr_policy"
N_FOLDS = 5
LEARNER_FOLDS = 2
#: Bounds no partial path probability of these laws reaches, so every weight is a product of
#: per-node ratios and the pairing smoke can require a binding share of zero.
G_BOUNDS = (1e-8, 1.0)
#: The pre-declaration smoke gate on the exact pairs (G1 and G2).
SMOKE_GATE = 1e-6
#: The replication factor of the policy pairing: every policy probability of the law is a
#: multiple of one quarter.
POLICY_COPIES = policy_study.COPIES

#: The rows of each scenario that are exactness witnesses (G1).  Every other paired row reads
#: the default margins.
EXACT_ROWS: dict[str, tuple[str, ...]] = {
    MTP: mtp_study.names(mtp.CONTINUOUS_LABELS),
    POLICY: ("ey_regimen[mix]",),
}

END_OF_STUDY_REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    RULE_LABEL: DynamicRegimen(
        RULE_LABEL,
        (1, lambda history: rule_arm_at_node_two(history["L2"])),
        rule_kind="known",
    ),
}
END_OF_STUDY_ESTIMANDS = (
    *(f"ey_regimen[{label}]" for label in END_OF_STUDY_REGIMENS),
    "ate_regimen[always vs never]",
    f"ate_regimen[{RULE_LABEL} vs never]",
)

ESTIMANDS: dict[str, tuple[str, ...]] = {
    END_OF_STUDY: END_OF_STUDY_ESTIMANDS,
    END_OF_STUDY_CROSSFIT: END_OF_STUDY_ESTIMANDS,
    SURVIVAL: survival_study.ESTIMANDS,
    MTP: mtp_study.names(mtp.CONTINUOUS_LABELS),
    POLICY: policy_study.ESTIMANDS,
}
REFERENCES = {
    END_OF_STUDY: "never",
    END_OF_STUDY_CROSSFIT: "never",
    SURVIVAL: "never",
    MTP: mtp_study.REFERENCE,
    POLICY: policy_study.REFERENCE,
}
N_FOLDS_OF = {
    scenario: N_FOLDS if scenario == END_OF_STUDY_CROSSFIT else 1 for scenario in ESTIMANDS
}
COPIES_OF = {scenario: POLICY_COPIES if scenario == POLICY else 1 for scenario in ESTIMANDS}

#: The labels of ``interval_calibration`` (see :mod:`tests.studies.longitudinal_sdr_properties`).
CALIBRATION_LABELS = (
    "static",
    "dynamic",
    "survival_h2",
    "competing_h2",
    "policy_mix",
    "mtp_up",
    "msm_identity",
    "weighted",
    "clustered",
    "clustered_few",
)
#: The joint label of the band cell.
BAND_LABEL = "six_regimens"
#: The patterns of ``sequential_double_robustness`` and their sizes (``X4-plan.md`` 6.8).
SDR_PATTERNS = ("both_correct", "P_A", "P_B", "P_C", "P_D")
SDR_SIZES = (2_000, 8_000)

#: Every published property cell, by family.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "sequential_double_robustness": (
        *(f"{pattern}__n_{size}" for pattern in SDR_PATTERNS for size in SDR_SIZES),
        "node2_both_wrong_control",
        "ltmle_P_D_control",
    ),
    "root_n_and_efficiency": tuple(f"static__n_{size}" for size in (1_000, 2_000, 8_000)),
    "root_n_rate": ("static__empirical_sd", "static__reported_se"),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("static__sharp_null",),
    "power": ("static__alternative",),
    "crossfit_overfitting": ("cross_fitted_sdr", "in_sample_control"),
    "simultaneous_coverage": (
        f"{BAND_LABEL}__simultaneous_band",
        f"{BAND_LABEL}__pointwise_joint_control",
    ),
}
#: Cells that are published without a verdict (``X4-plan.md`` 3.5 and 6.11).
DIAGNOSTIC_CELLS: dict[str, tuple[str, ...]] = {
    "construction_diagnostic": ("stitched_cross_fit", "estimated_flexible_mechanism"),
    "sdr_tmle_agreement": ("static",),
}

STUDY = StudyRecord(
    name="sequentially doubly robust longitudinal estimation",
    slug="longitudinal-sdr",
    artifacts=ROOT / "tests" / "canonical" / "longitudinal_sdr",
    document="docs/technical-reference/method-evidence/sequentially-doubly-robust-estimation.md",
    anchor="sequentially-doubly-robust-estimation",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-sdr",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_longitudinal_sdr.py",
        "tests/studies/longitudinal_sdr_properties.py",
        "tests/studies/canonical_longitudinal_mtp.py",
        "tests/studies/canonical_ltmle_survival.py",
        "tests/studies/canonical_stochastic_categorical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/longitudinal_mtp_common.py",
        "tests/studies/oracle_density_bins.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_sdr.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/discrete_law_longitudinal_policy.py",
        "tests/discrete_law_survival.py",
        "tests/discrete_law_competing.py",
        "tests/longitudinal_policies.py",
        "tests/longitudinal_sdr.py",
        "src/cleverly/datasets/longitudinal.py",
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
        "tests/canonical/lmtp_sdr_adapter.R",
        "tests/canonical/longitudinal_sdr_runner.R",
    ),
    runner_module="tests.studies.canonical_longitudinal_sdr",
    properties_module="tests.studies.longitudinal_sdr_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_sdr and theta_dr(is_sdr = TRUE) on an LmtpTask with the shifted values and "
        "the per-node ratios this package divides by, written beside the replicate data; no "
        "lmtp density ratio is fitted. A policy is realised by copying each unit "
        f"{POLICY_COPIES} times, with the unit's ratio on every copy and the unit-mean eif"
    ),
    "reference_adapter": "tests/canonical/lmtp_sdr_adapter.R",
}

CONFIGURATION = {
    "construction": SDR_TRAINING_FOLD_RECURSION,
    "reference_construction": "lmtp cf_sdr with supplied fold-free ratios",
    "n_folds": dict(N_FOLDS_OF),
    "learner_folds": LEARNER_FOLDS,
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": "LinearRegression (unclipped)",
    "mechanism": "known, supplied to both",
    "density_bins": "the longitudinal-mtp oracle bins, ceil(320 (n / 2000)^(2/3))",
    "reference": dict(REFERENCES),
    "policy_copies": POLICY_COPIES,
    "exact_rows": {scenario: list(names) for scenario, names in EXACT_ROWS.items()},
}


def slug(label: str) -> str:
    """The column-name form of a plan label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def regimens(scenario: str) -> dict[str, Any]:
    if scenario in {END_OF_STUDY, END_OF_STUDY_CROSSFIT, SURVIVAL}:
        return dict(END_OF_STUDY_REGIMENS)
    if scenario == MTP:
        return mtp.continuous_regimens()
    if scenario == POLICY:
        return policy_study.regimens()
    raise KeyError(scenario)


def fit(frame: pd.DataFrame, scenario: str, *, n_folds: int | None = None) -> Any:
    """The one fit of each primary scenario."""
    folds = N_FOLDS_OF[scenario] if n_folds is None else n_folds
    settings: dict[str, Any] = {
        "reference": REFERENCES[scenario],
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": LinearRegression(),
        "n_folds": folds,
        "learner_folds": LEARNER_FOLDS,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "random_state": 0,
    }
    if scenario in {END_OF_STUDY, END_OF_STUDY_CROSSFIT, SURVIVAL}:
        settings["treatment_learner"] = KnownLongitudinalMechanism("treatment")
        settings["censoring_learner"] = KnownLongitudinalMechanism("censoring")
        outcome: Any = ["Y1", "Y2"] if scenario == SURVIVAL else "Y"
        return SDR(regimens(scenario), **settings).fit(
            frame,
            outcome=outcome,
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    if scenario == MTP:
        _, _, hazard = mtp.continuous_learners("primary", mtp_study.edges_of(frame))
        settings["treatment_learner"] = clone(hazard)
        settings["density_bins"] = oracle_bins(len(frame))
        return SDR(regimens(scenario), **settings).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        )
    settings["treatment_learner"] = KnownCategoricalMechanism()
    return SDR(regimens(scenario), **settings).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario in {END_OF_STUDY, END_OF_STUDY_CROSSFIT}:
        frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
        return frame, {name: float(truth[name]) for name in END_OF_STUDY_ESTIMANDS}
    if scenario == SURVIVAL:
        frame, _ = make_longitudinal_survival(n=n, seed=seed, censoring=True, backend="pandas")
        return frame, dict(survival_study.TRUTH)
    if scenario == MTP:
        return mtp.sample_continuous(n, seed), dict(mtp_study.TRUTH[mtp_study.CONTINUOUS])
    if scenario == POLICY:
        return multivalue.sample(multivalue.PROBS, n, seed), {
            name: policy_law.TRUTH[name] for name in policy_study.ESTIMANDS
        }
    raise KeyError(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ------------------------------------------------------------------------------------ pairing


def plans_of(result: Any, declared: Mapping[str, Any]) -> tuple[Plan, ...]:
    """The fit's plans, resolved again and with its mechanism's policy numerators attached."""
    data = result.data
    specs = resolve_regimens(
        dict(declared),
        data.n_decisions,
        held_nodes=data.n_times if data.treatment_held else None,
    )
    return attach_policy_numerators(resolve_plans(specs, data), result.mechanism.policy_numerators)


def supplied_ratios(result: Any, plan: Plan) -> np.ndarray:
    r"""``(n, T)`` per-node ratio :math:`\pi_t 1\{C_t = 1\} / (g_t c_t)` at the observed arm.

    Zero where the unit leaves the plan, or the observation, at node :math:`t`.  The mechanism
    is the fit's own, which is fold-free here because every scenario's mechanism is known.
    """
    data = result.data
    numerator = plan.intervention_density(data)
    factors = node_factors(data, plan, result.mechanism)
    observed = np.asarray(data.uncensored, dtype=bool) & (numerator > 0.0)
    safe = np.where(factors > 0.0, factors, 1.0)
    return np.where(observed & (factors > 0.0), numerator / safe, 0.0)


def check_ratios(result: Any, plan: Plan, ratios: np.ndarray) -> None:
    """The running products of ``ratios`` are the unbounded weights :math:`W_{1,s}`."""
    data = result.data
    numerator = plan.intervention_density(data)
    factors = node_factors(data, plan, result.mechanism)
    masks = plan.masks(data)
    weights, _, binding = partial_weights(
        numerator, factors, masks, 1, data.n_times, (np.finfo(float).tiny, 1.0)
    )
    if binding:
        raise AssertionError(f"plan {plan.label!r}: a partial path probability is zero")
    running = np.ones(data.n)
    for node, weight in enumerate(weights, start=1):
        running = running * ratios[:, node - 1]
        following = masks.following(node)
        if not np.allclose(running[following], weight[following], rtol=1e-12, atol=0.0):
            raise AssertionError(
                f"plan {plan.label!r} at node {node}: the supplied ratios are not the weights"
            )


def _allocate(probabilities: np.ndarray, copies: int, where: str) -> np.ndarray:
    """``(n, copies)`` level code each copy of a unit takes: ``p * copies`` copies per level."""
    counts = probabilities * copies
    if np.max(np.abs(counts - np.round(counts))) > 1e-12:
        raise AssertionError(f"{where} has a probability off a multiple of 1/{copies}")
    cumulative = np.cumsum(np.round(counts), axis=1)
    if np.any(cumulative[:, -1] != copies):
        raise AssertionError(f"{where} allocates a unit other than {copies} copies")
    slots = np.arange(copies)
    return np.argmax(cumulative[:, None, :] > slots[None, :, None], axis=2)


def shifted_values(result: Any, plan: Plan, node: int, copies: int) -> list[np.ndarray]:
    """The value each copy of a unit takes at ``node`` under ``plan``, one array per copy."""
    data = result.data
    where = f"plan {plan.label!r} at node {node}"
    targets = plan.policy_targets[node - 1] if plan.policy_targets else None
    if targets is not None:
        if len(targets) != 1 or copies != 1:
            raise AssertionError(f"{where}: only a deterministic continuous policy is paired")
        return [np.asarray(targets[0], dtype=float)]
    if plan.mtp_assignments and plan.mtp_assignments[node - 1] is not None:
        raise AssertionError(f"{where}: a categorical modified treatment policy is not paired")
    levels = np.asarray(data.treatment_levels[node - 1], dtype=object)
    density = plan.policy[node - 1] if plan.policy else None
    if density is None:
        codes = np.nan_to_num(np.asarray(plan.values[:, node - 1], dtype=float)).astype(int)
        density = np.eye(len(levels))[codes]
    allocation = _allocate(np.asarray(density, dtype=float), copies, where)
    return [levels[allocation[:, copy]] for copy in range(copies)]


def pairing_columns(
    result: Any, declared: Mapping[str, Any], copies: int = 1
) -> dict[str, np.ndarray]:
    """The ``ratio__`` and ``shift__`` columns of every plan, for the comparator to read."""
    out: dict[str, np.ndarray] = {}
    for plan in plans_of(result, declared):
        ratios = supplied_ratios(result, plan)
        check_ratios(result, plan, ratios)
        name = slug(plan.label)
        for node in range(1, result.data.n_times + 1):
            out[f"ratio__{name}__{node}"] = ratios[:, node - 1]
            for copy, values in enumerate(shifted_values(result, plan, node, copies), start=1):
                out[f"shift__{name}__{node}__{copy}"] = values
    return out


# --------------------------------------------------------------------------------------- rows


def initial_mean(regimen_fit: Any) -> float:
    """The plug-in at the first node, policy-weighted where the first node is a policy."""
    first = regimen_fit.steps[0]
    if first.initial_by_arm is None or not regimen_fit.policy or regimen_fit.policy[0] is None:
        return float(np.mean(first.initial))
    return float(np.mean(np.sum(regimen_fit.policy[0] * first.initial_by_arm, axis=1)))


def initials(result: Any, scenario: str) -> dict[str, float]:
    if scenario == SURVIVAL:
        return survival_study.horizon_initials(
            result, survival_study.MEAN_NAMES, survival_study.CONTRAST_NAMES
        )
    means = {f"ey_regimen[{label}]": initial_mean(fit) for label, fit in result.fits.items()}
    out = {}
    for name in ESTIMANDS[scenario]:
        if name.startswith("ey_regimen["):
            out[name] = means[name]
        else:
            left, right = name[len("ate_regimen[") : -1].split(" vs ")
            out[name] = means[f"ey_regimen[{left}]"] - means[f"ey_regimen[{right}]"]
    return out


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    plug_in = initials(result, scenario)
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
                "initial_estimate": plug_in[name],
            }
        )
    return rows


def binding_share(result: Any) -> float:
    """The largest share of followed paths whose partial product a bound moved."""
    return max(step.bound_binding for fit in result.fits.values() for step in fit.steps)


def _replicate(
    scenario: str, replicate: int, n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit(frame, scenario)
    if binding_share(result) > 0.0:
        raise AssertionError(f"{scenario} replicate {replicate}: a g_bounds bound binds")
    sample = frame.copy()
    for name, values in pairing_columns(result, regimens(scenario), COPIES_OF[scenario]).items():
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
