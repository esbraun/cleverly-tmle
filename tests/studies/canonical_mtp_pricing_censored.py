"""Registered evidence for censored boundary-binding price plans (``mtp-pricing-censored``).

The subject is a modified treatment policy on a price with dropout after each price node: the
``price_up`` plan of ``mtp-pricing`` at three nodes on law P-C
(:mod:`tests.studies.mtp_pricing_censored_common`), with the censoring factor learned by a
main-terms logistic regression beside the policy ratio.  The parameter is the mean purchase
with the price on the plan and nobody lost to follow-up; law P-C changes no structural equation
of law P, so its truths are law P's (:data:`~tests.studies.mtp_pricing_common.TRUTHS`).  The
joint ratio of a node is the policy's price ratio times ``1{C_t = 1} / P(C_t = 1 | A_t, H_t)``
at the **observed** price (Díaz, Williams, Hoffman and Schenck 2023, arXiv v4, Section 3, the
paragraph after Theorem 1, with the censoring component of each intervention set to one).

Two primary scenarios, :data:`PRIMARY_REPLICATES` replications at :data:`PRIMARY_N`, on the
same draws (``scenario_seed_owners``):

=========================  ==================================================================
scenario                   subject
=========================  ==================================================================
``censored_ltmle_tau3``    ``LTMLE``, plans ``natural_prices`` and ``price_up``, one fold, the
                           exact price ratio on the classifier route
                           (:class:`~tests.studies.mtp_pricing_common.OracleRatio`) and the
                           primary learners of ``mtp-pricing``, with the logistic censoring
                           factor.  The censored recursion with the ratio held exact, so a red
                           row isolates the censoring path
``censored_sdr_tau3``      the same draws, ``SDR``
=========================  ==================================================================

The property families are in :mod:`tests.studies.mtp_pricing_censored_properties`: the
cross-fitted learned-classifier calibration cell of each engine and a censoring-necessity pair
per engine.  The support rows ``G39.<engine>.tau3_cross_fitted_censored`` cite those cells.

**Which cell carries which claim.**  The primary rows check that the censored recursion, with
the price ratio exact, is unbiased and covers at n = 2,400; on this law they cannot see a
censoring factor that never reaches the weights, because the primary outcome learners nearly
correct for the dropout on their own (a 160-draw design-review probe read the bias of a fit with
a constant per-node retention within 0.07 SD of the learned one).  That the factor reaches the
weights, at the observed dose, is carried by the necessity pairs (a wrong outcome model, where
only the weights can correct) and by the exact-law witnesses with their mutation controls
(``tests/unit/test_longitudinal_mtp_censored.py``).  The calibration cells carry the interval
under a learned classifier and a learned censoring factor.

**Comparator.**  R ``lmtp`` 1.5.4 is pending (:attr:`StudyRecord.pending_reference`): its runs
take the per-node joint ratio this package computed (``ratio__<plan>__<node>``, written beside
each sample by :func:`pairing_columns`) and the censoring columns, an exactness witness, on the
local study queue after the merge.  Because ``lmtp`` would read
this package's own joint ratio, the paired rows check the recursion given the ratio and cannot
catch a wrong retention model; the necessity pairs and the exact-law witnesses do.  A missing comparator blocks nothing
(``docs/architecture-invariants.md``, "Theory and comparators").

**Red-cell reading, declared before any run.**  Publication policy ``reporting``, with the
gated verdicts read by code (:data:`~tests.studies.mtp_pricing_censored_properties.GATING_CELLS`
and the primary truth rows).

* A red primary truth row or calibration positive is a defect suspect unless the finite-sample
  route holds: its 99% SE-ratio interval contains 1, its coverage interval
  contains 0.95, and a paired-arm diagnosis on fresh seeds against the complete-data sibling of
  ``mtp-pricing`` finds no defect.  Only then does it report under a named owner in the
  roadmap's Red-cell owners table.
* A red necessity positive, or a necessity control that is not displaced, is a defect suspect
  and never takes an owner.
* A calibration control that does not fail is a red of the study: the cell it guards loses its
  claim until the control is repaired under a fresh declaration.
* No replicate count, law, learner, margin or seed moves after a verdict.

**The tail-size rule** (the X39 rule, as :mod:`tests.studies.canonical_mtp_pricing` states it):
a Wald reading gates only at ``n >= 2 n_kappa``, with ``n_kappa`` read on law P-C's own limit
nuisances (:data:`TAIL_SIZES`).  n = 2,400 puts every primary row in the gated regime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, SDR
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_pricing_censored_common as law
from tests.studies import mtp_pricing_common as common
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

TAU = law.TAU
PRIMARY_REPLICATES = 1_000
#: One n for every row of the record: ``gated_size`` of the largest tail size (1,098, the
#: contrast's EIF proxy), the least multiple of 400 at or above twice it (:data:`TAIL_SIZES`).
#: ``mtp-pricing``'s three-node siblings run at 2,000, so these cells differ from them in the
#: dropout and in n.
PRIMARY_N = 2_400
SEED = 20267701
RESAMPLING_SEED = 2026770101
G_BOUNDS = pricing.G_BOUNDS
PRIMARY_ALPHA = pricing.PRIMARY_ALPHA
REFERENCE = common.REFERENCE
PLAN = "price_up"
LABELS = (REFERENCE, PLAN)
FAILURE_BOUND = 0.01

#: Each scenario: its estimator.
SCENARIOS: dict[str, str] = {f"censored_{e}_tau{TAU}": e for e in ("ltmle", "sdr")}


def names(labels: tuple[str, ...] = LABELS) -> tuple[str, ...]:
    """Each plan's mean and its contrast with the natural course."""
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {REFERENCE}]" for label in labels if label != REFERENCE
    )


ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, names())


def truth_of(labels: tuple[str, ...] = LABELS) -> dict[str, float]:
    """Law P's truths at three nodes, which law P-C keeps."""
    return pricing.truth_of(TAU, labels)


#: Every published property cell (:mod:`tests.studies.mtp_pricing_censored_properties`).
CALIBRATION_LABELS = tuple(f"censored_crossfit_{e}_tau{TAU}" for e in ("ltmle", "sdr"))
#: ``family: ((positive cell, control cell), ...)``.
PAIRED_CELLS: dict[str, tuple[tuple[str, str], ...]] = {
    "censoring_necessity": tuple(
        (f"up_{e}_tau{TAU}__learned_censoring", f"up_{e}_tau{TAU}__prior_censoring_control")
        for e in ("ltmle", "sdr")
    ),
}
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    **{family: tuple(c for pair in pairs for c in pair) for family, pairs in PAIRED_CELLS.items()},
}

STUDY = StudyRecord(
    name="censored boundary-binding price plans",
    slug="mtp-pricing-censored",
    artifacts=ROOT / "tests" / "canonical" / "mtp_pricing_censored",
    document="docs/technical-reference/method-evidence/mtp-pricing-censored.md",
    anchor="mtp-pricing-censored",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={f"censored_sdr_tau{TAU}": f"censored_ltmle_tau{TAU}"},
    margins=Margins(),
    implementation="cleverly-mtp-pricing-censored",
    pending_reference="lmtp",
    pending_reason=(
        "the lmtp 1.5.4 runs read the joint ratio this package computed and the censoring "
        "columns; they run on the local study queue after the merge (X77 plan section 5), and "
        "the method merges on its Python evidence"
    ),
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_mtp_pricing_censored.py",
        "tests/studies/mtp_pricing_censored_properties.py",
        "tests/studies/mtp_pricing_censored_common.py",
        "tests/studies/canonical_mtp_pricing.py",
        "tests/studies/mtp_pricing_properties.py",
        "tests/studies/mtp_pricing_common.py",
        "tests/studies/mtp_study_tools.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/law_q_tuned.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_mtp_tuning.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/oracle_density_bins.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mtp_pricing_censored",
    properties_module="tests.studies.mtp_pricing_censored_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    failure_bound={"cleverly-mtp-pricing-censored": FAILURE_BOUND, "lmtp": FAILURE_BOUND},
)

CONFIGURATION = {
    "construction": "ordinary, one fold",
    "g_bounds": list(G_BOUNDS),
    "support": "Support(4 + 2 L_t, 8 + 2 L_t) at every A_t",
    "outcome_learner": "QuasiBinomialGLM at the last node; LinearRegression at earlier nodes",
    "ltmle_alpha": PRIMARY_ALPHA,
    "treatment_learner": "OracleRatio (the exact ratio on the classifier route)",
    "censoring_learner": "LogisticRegression(C=1e6), main terms of the node's history design",
    "censoring": (
        f"C_t ~ Bernoulli(expit({law.ALPHA} + {law.BETA_W} W1 + {law.BETA_L} L_t "
        f"{law.BETA_GAP:+} (A_t - c_t))), monotone"
    ),
    "plans": list(LABELS),
    "horizon": TAU,
    "reference": REFERENCE,
}

#: Each primary estimand's tail size on law P-C: the larger of the property learners' curve
#: (the EIF proxy) and the primary fit's own, at the 99% chunk-bootstrap bound of the kurtosis
#: on 10^8 draws with 10^7-row limits (``tests/diagnostics/x77_censored_tail/``,
#: ``law_pc_moments.csv``).  The proxy is the larger for every estimand and engine.
TAIL_SIZES: dict[str, int] = {
    "ey_regimen[natural_prices]": 24,
    f"ey_regimen[{PLAN}]": 687,
    f"ate_regimen[{PLAN} vs {REFERENCE}]": 1_098,
}
#: The primary truth rows whose n is below twice their tail size, as ``scenario/estimand``.
BELOW_TAIL_SIZE: frozenset[str] = frozenset()
#: The primary truth rows whose n is at least twice their tail size.
WALD_GATED_CELLS = frozenset(
    f"{scenario}/{name}"
    for scenario, names_of in ESTIMANDS.items()
    for name in names_of
    if f"{scenario}/{name}" not in BELOW_TAIL_SIZE
)
#: The primary truth rows with no computed tail size.
TAIL_SIZE_NOT_COMPUTED: dict[str, str] = {}


def fit(
    frame: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    *,
    estimator: str = "ltmle",
    outcome_learner: Any = None,
    pseudo_learner: Any = None,
    treatment_learner: Any = None,
    censoring_learner: Any = None,
    ratio: Literal["density", "classifier"] = "classifier",
    n_folds: int = 1,
    alpha: float | None = None,
    random_state: int = 0,
) -> Any:
    """``LTMLE`` or ``SDR`` over ``labels`` on law P-C; the primary learners by default."""
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": clone(
            QuasiBinomialGLM() if outcome_learner is None else outcome_learner
        ),
        "pseudo_learner": clone(LinearRegression() if pseudo_learner is None else pseudo_learner),
        "treatment_learner": clone(
            common.OracleRatio() if treatment_learner is None else treatment_learner
        ),
        "censoring_learner": clone(
            law.censoring_learner() if censoring_learner is None else censoring_learner
        ),
        "n_folds": n_folds,
        "g_bounds": G_BOUNDS,
        "ratio": ratio,
        "simultaneous": False,
        "random_state": random_state,
    }
    regimens = common.regimens(labels, TAU)
    if estimator == "ltmle":
        targeting = {} if alpha is None else {"alpha": alpha}
        model: Any = LTMLE(regimens, max_iter=100, tol=1e-10, **targeting, **settings)
    elif estimator == "sdr":
        model = SDR(regimens, **settings)
    else:
        raise ValueError(estimator)
    return model.fit(frame, outcome="Y", **law.roles(TAU))


# ---------------------------------------------------------------------------- rows


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    return law.sample(n, seed), truth_of()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand.  ``initial_estimate`` is the first node's untargeted plug-in."""
    initials = {
        f"ey_regimen[{label}]": pricing.initial_mean(result.fits[label]) for label in LABELS
    }
    initials[f"ate_regimen[{PLAN} vs {REFERENCE}]"] = (
        initials[f"ey_regimen[{PLAN}]"] - initials[f"ey_regimen[{REFERENCE}]"]
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


def pairing_columns(result: Any) -> dict[str, np.ndarray]:
    """The per-node joint ratio this fit used, for the pending comparator to read.

    ``ratio__<plan>__<t>`` is the policy's price ratio at the observed dose (the fit's
    ``node_ratio``) times ``1{C_t = 1}`` over the learned retention at the observed history and
    dose, zero once a unit has left: the joint ratio of Díaz et al. (2023) Section 3.  The
    retention is the step of the fit's unbounded cumulative mechanism, whose treatment factor is
    one at a continuous node.
    """
    out: dict[str, np.ndarray] = {}
    for label in LABELS:
        fitted = result.fits[label]
        assert fitted.node_ratio is not None
        cumulative = np.asarray(fitted.cumulative_unbounded, dtype=float)
        previous = np.ones(cumulative.shape[0])
        for time in range(1, TAU + 1):
            current = cumulative[:, time - 1]
            stayed = result.data.uncensored_through(time) & (previous > 0.0) & (current > 0.0)
            retention = np.where(stayed, current / np.where(previous > 0.0, previous, 1.0), 1.0)
            policy = np.asarray(fitted.node_ratio[:, time - 1], dtype=float)
            out[f"ratio__{pricing.slug(label)}__{time}"] = np.where(stayed, policy / retention, 0.0)
            previous = current
    return out


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    estimator = SCENARIOS[scenario]
    frame, truth = drawn
    result = fit(frame, estimator=estimator, alpha=PRIMARY_ALPHA if estimator == "ltmle" else None)
    sample = frame.copy()
    for name, values in pairing_columns(result).items():
        sample[name] = values
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", np.zeros(len(sample), dtype=int))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, result_rows(result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    # Replicate-major, so both scenarios of one replicate are contiguous in the sample archive
    # and the pending comparator's runner reads them as one group.
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
        spill_samples_at=0,
    )
    samples = replication.assembled_samples(outcomes, at=0)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


__all__ = ["STUDY", "draw_and_fit", "fit"]
