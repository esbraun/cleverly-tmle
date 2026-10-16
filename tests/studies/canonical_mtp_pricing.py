"""Registered evidence for boundary-binding price plans over follow-up lengths (``mtp-pricing``).

The subject is a modified treatment policy on a price whose declared support depends on the
node's engagement, ``continuous_treatment={"A_t": Support(4 + 2 L_t, 8 + 2 L_t)}``, at
``tau`` in ``{2, 3, 5}`` nodes.  ``price_up`` scales the price by 1.12 and binds at the
upper boundary; ``price_down`` scales it by 0.85 and binds at the lower boundary, with two pieces.  The
law, the plans, the truths and the learners live in :mod:`tests.studies.mtp_pricing_common`.
The comparator is R ``lmtp`` 1.5.4.  Six primary scenarios, 1,000 replications at
n = 5,000:

=======================  ================================================================
scenario                 subject and pairing
=======================  ================================================================
``pricing_ltmle_tau2``   law P at ``tau = 2``, ``LTMLE``, plans ``natural_prices``,
                         ``price_up`` and ``price_down``.  Every node reads the exact ratio on the classifier
                         route (:class:`~tests.studies.mtp_pricing_common.OracleRatio`).
                         The shifted values and the per-node ratios are written beside the
                         sample.  Exactness witness
``pricing_sdr_tau2``     the same draw (``scenario_seed_owners``), ``SDR``.  Exactness
                         witness
``pricing_*_tau3``       the same at ``tau = 3``
``pricing_*_tau5``       the same at ``tau = 5``
=======================  ================================================================

**Learners.**  The primary rows match the comparator's ``SL.glm`` and its prediction bound: a
quasibinomial GLM on the outcome design at the last node, and least squares clipped to
``[1e-5, 1 - 1e-5]`` at every earlier node (:data:`PRIMARY_PSEUDO_LEARNER`).  The property
rows read the declared learners of :mod:`tests.studies.mtp_pricing_properties`.

**The comparator.**  ``tests/canonical/mtp_pricing_runner.R`` calls ``lmtp_mtp_tmle``
(``tests/canonical/lmtp_mtp_adapter.R``) for the ``LTMLE`` scenarios and
``lmtp_sdr_with_folds`` (``tests/canonical/lmtp_sdr_adapter.R``) for the ``SDR`` scenarios.
Both read ``shift__<plan>__<node>__1`` and ``ratio__<plan>__<node>``; the adapters loop over
``length(trt)``, so ``tau = 5`` needs no new adapter.  Every primary row is an exactness
witness: ``exactness.csv`` (:func:`extra_artifacts`) records each estimand's largest paired gap
in estimate and standard error, and a gap of :data:`SMOKE_GATE` or more is a red row.  A
pre-declaration R smoke of 3 replications reads the same table.  These rows are comparator
evidence.

**Second declaration (2026-10-16).**  The first run (cleverly-tmle fb70e6ef, n = 2,000) read
the four ``tau = 5`` ``price_down`` rows red: coverage 0.903 to 0.912, SE ratio 0.93, bias
under 0.04 SD, on the exact ratio.  A probe on fresh draws read SE ratio 0.887 at n = 2,000 and
0.968 at n = 4,000, a finite-sample shortfall of the influence-curve variance.  The primary
moves to n = 5,000 on Eric's decision (2026-10-16); the property cells keep their declared
sizes and their committed rows.  The n = 2,000 reading stays in the method-evidence page.

**Third declaration (2026-10-16, before any comparator run).**  The first R smoke stopped:
``lmtp`` bounds every sequential prediction to ``[1e-5, 1 - 1e-5]``, and the least-squares
fits of nodes 1 to 4 predicted at or above ``1 - 1e-5`` in 13 to 20 of 20 probe samples of
every scenario.  The last node, a quasibinomial GLM, never did.  ``lmtp`` clipped those
predictions and the package did not, so the pairs could not agree.  The primary learner before
the last node becomes least squares clipped at ``lmtp``'s bound
(:class:`~tests.studies.mtp_pricing_common.BoundedLeastSquares`), so both sides read one
clipped regression.  The true regression at each node is a probability, so the clip projects
onto the parameter space.  The primary rows refit.  The property cells keep their learners and
their committed rows.

**Red-cell reading, declared before any run.**  As ``mtp-boundaries``
(:mod:`tests.studies.canonical_mtp_boundaries`), with the owner rows
``mtp-pricing-calibration`` and ``mtp-pricing-pairing``.  Publication policy is
``reporting``, and that reading is the merge gate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.interventions.policy import lazy_frame, policy_branches
from cleverly.longitudinal import LTMLE, SDR
from tests.parallel import STUDY_JOBS
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_study_tools as tools
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 5_000
SEED = 20264021
RESAMPLING_SEED = 2026402101
#: The pre-declaration smoke gate on the exactness witnesses.
SMOKE_GATE = 1e-6
#: Bounds wide enough that no cumulative probability of this law reaches them.
G_BOUNDS = (0.001, 1.0)
REFERENCE = common.REFERENCE
LABELS = common.LABELS
FAILURE_BOUND = 0.01
#: The primary rows' learner at every node before the last: the comparator's ``SL.glm`` with
#: ``lmtp``'s prediction bound.  Plain least squares predicts at or above ``1 - 1e-5`` at
#: nodes 1 to 4 in most samples of every scenario, where ``lmtp`` bounds and the package does
#: not, so the two sides would read two regularisations of one regression.  The property cells
#: keep their own learners.
PRIMARY_PSEUDO_LEARNER = common.BoundedLeastSquares()

#: Each scenario: ``(tau, estimator)``.
SCENARIOS: dict[str, tuple[int, str]] = {
    f"pricing_{estimator}_tau{tau}": (tau, estimator)
    for tau in common.HORIZONS
    for estimator in ("ltmle", "sdr")
}


def names(labels: tuple[str, ...] = LABELS) -> tuple[str, ...]:
    """Each plan's mean and its contrast with ``natural``."""
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {REFERENCE}]" for label in labels if label != REFERENCE
    )


ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, names())


def truth_of(tau: int, labels: tuple[str, ...] = LABELS, revenue: bool = False) -> dict[str, float]:
    """The value of every estimand at ``tau``: quadrature at 2, the pinned Monte Carlo above."""
    kind = "er" if revenue else "ey"
    out = {f"ey_regimen[{label}]": common.truth(tau, f"{kind}[{label}]") for label in labels}
    for label in labels:
        if label != REFERENCE:
            out[f"ate_regimen[{label} vs {REFERENCE}]"] = (
                out[f"ey_regimen[{label}]"] - out[f"ey_regimen[{REFERENCE}]"]
            )
    return out


#: Every published property cell (:mod:`tests.studies.mtp_pricing_properties`).
CALIBRATION_LABELS = (
    *(f"up_oracle_tau{tau}" for tau in common.HORIZONS),
    *(f"up_learned_density_tau{tau}" for tau in common.HORIZONS),
    *(f"up_learned_classifier_tau{tau}" for tau in common.HORIZONS),
    "revenue_learned_classifier_tau3",
    "crossfit_ltmle_learned_classifier_tau3",
    "crossfit_sdr_learned_classifier_tau3",
)
#: ``family: ((positive cell, control cell), ...)``.
PAIRED_CELLS: dict[str, tuple[tuple[str, str], ...]] = {
    "double_robustness": (
        ("up_tau3__outcome_wrong_learned_classifier", "up_tau3__both_wrong_control"),
    ),
    "ratio_necessity": tuple(
        (f"up_tau{tau}__learned_classifier", f"up_tau{tau}__unit_ratio_control") for tau in (3, 5)
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
    name="boundary-binding price plans over follow-up lengths",
    slug="mtp-pricing",
    artifacts=ROOT / "tests" / "canonical" / "mtp_pricing",
    document="docs/technical-reference/method-evidence/mtp-pricing.md",
    anchor="mtp-pricing",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={
        f"pricing_sdr_tau{tau}": f"pricing_ltmle_tau{tau}" for tau in common.HORIZONS
    },
    margins=Margins(),
    implementation="cleverly-mtp-pricing",
    reference="lmtp",
    comparison="matching implementation",
    modules=(
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
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_policy_adapter.R",
        "tests/canonical/lmtp_mtp_adapter.R",
        "tests/canonical/lmtp_sdr_adapter.R",
        "tests/canonical/mtp_pricing_runner.R",
    ),
    runner_module="tests.studies.canonical_mtp_pricing",
    properties_module="tests.studies.mtp_pricing_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    failure_bound={"cleverly-mtp-pricing": FAILURE_BOUND, "lmtp": FAILURE_BOUND},
    extra_artifacts=("exactness.csv",),
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_tmle and theta_dr (LTMLE scenarios) and cf_sdr and theta_dr (SDR scenarios) "
        "on an LmtpTask with the shifted values and the per-node ratios this package "
        "computed written beside the replicate data; no lmtp density ratio is fitted"
    ),
    "reference_adapter": "tests/canonical/lmtp_mtp_adapter.R, tests/canonical/lmtp_sdr_adapter.R",
    "pairing": dict.fromkeys(SCENARIOS, "exact under SMOKE_GATE"),
}

CONFIGURATION = {
    "construction": "ordinary, one fold",
    "g_bounds": list(G_BOUNDS),
    "support": "Support(4 + 2 L_t, 8 + 2 L_t) at every A_t",
    "outcome_learner": (
        "QuasiBinomialGLM at the last node; least squares clipped to [1e-5, 1 - 1e-5] at "
        "earlier nodes (lmtp: SL.glm and its prediction bound)"
    ),
    "treatment_learner": "OracleRatio (the exact ratio on the classifier route)",
    "plans": list(LABELS),
    "horizons": list(common.HORIZONS),
    "reference": REFERENCE,
}


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def roles(tau: int) -> dict[str, Any]:
    """The node roles of law P at ``tau``."""
    return {
        "treatment": [f"A{t}" for t in range(1, tau + 1)],
        "baseline": ["W1"],
        "time_varying": [[f"L{t}"] for t in range(1, tau + 1)],
        "continuous_treatment": {f"A{t}": common.support(t) for t in range(1, tau + 1)},
    }


def fit(
    frame: pd.DataFrame,
    tau: int,
    labels: tuple[str, ...] = LABELS,
    *,
    estimator: str = "ltmle",
    outcome_learner: Any = None,
    pseudo_learner: Any = None,
    treatment_learner: Any = None,
    ratio: Literal["density", "classifier"] = "classifier",
    n_folds: int = 1,
    revenue: bool = False,
    density_bins: Any = None,
    random_state: int = 0,
) -> Any:
    """``LTMLE`` or ``SDR`` over ``labels`` on law P at ``tau``; the primary learners by default."""
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": clone(
            QuasiBinomialGLM() if outcome_learner is None else outcome_learner
        ),
        "pseudo_learner": clone(LinearRegression() if pseudo_learner is None else pseudo_learner),
        "treatment_learner": clone(
            common.OracleRatio() if treatment_learner is None else treatment_learner
        ),
        "n_folds": n_folds,
        "g_bounds": G_BOUNDS,
        "ratio": ratio,
        "simultaneous": False,
        "random_state": random_state,
    }
    if density_bins is not None:
        settings["density_bins"] = density_bins
    if revenue:
        settings["q_bounds"] = common.REVENUE_BOUNDS
    regimens = common.regimens(labels, tau)
    if estimator == "ltmle":
        model: Any = LTMLE(regimens, max_iter=100, tol=1e-10, **settings)
    elif estimator == "sdr":
        model = SDR(regimens, **settings)
    else:
        raise ValueError(estimator)
    return model.fit(frame, outcome="R" if revenue else "Y", **roles(tau))


# ---------------------------------------------------------------------------- rows


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    tau = SCENARIOS[scenario][0]
    return common.sample(n, seed, tau), truth_of(tau)


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
    initials = {f"ey_regimen[{label}]": initial_mean(result.fits[label]) for label in LABELS}
    for label in LABELS:
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


def pairing_columns(result: Any, frame: pd.DataFrame, tau: int) -> dict[str, np.ndarray]:
    """The shifted values and per-node ratios this fit used, for the comparator to read."""
    data = result.data
    out: dict[str, np.ndarray] = {}
    for label, plan in common.regimens(LABELS, tau).items():
        fitted = result.fits[label]
        assert fitted.node_ratio is not None
        for time, policy in enumerate(plan.plan, start=1):
            out[f"ratio__{slug(label)}__{time}"] = np.asarray(
                fitted.node_ratio[:, time - 1], dtype=float
            )
            frame_at = lazy_frame(lambda time=time: data.policy_frame(time))
            ((probability, branch),) = policy_branches(policy)
            assert probability == 1.0
            values = branch.assign(frame[f"A{time}"].to_numpy(dtype=float), frame_at)
            out[f"shift__{slug(label)}__{time}__1"] = np.asarray(values, dtype=float)
    return out


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    tau, estimator = SCENARIOS[scenario]
    frame, truth = drawn
    result = fit(frame, tau, estimator=estimator, pseudo_learner=PRIMARY_PSEUDO_LEARNER)
    sample = frame.copy()
    for name, values in pairing_columns(result, frame, tau).items():
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
    # Replicate-major, so every scenario of one replicate is contiguous in the sample archive
    # and the comparator's runner reads them as one group.  The horizons' columns differ, so
    # the columns of a longer horizon are empty in a shorter one's rows.
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
        # Each fit writes its sample to disk, so the run holds one sample per worker.
        spill_samples_at=0,
    )
    samples = replication.assembled_samples(outcomes, at=0)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


# ---------------------------------------------------------------------------- exactness


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``exactness.csv``: each primary estimand's largest paired gap against :data:`SMOKE_GATE`."""
    return {
        "exactness.csv": tools.exactness_table(
            rows,
            subject=STUDY.implementation,
            reference=str(STUDY.reference),
            scenarios=sorted(SCENARIOS),
            gate=SMOKE_GATE,
        )
    }


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The exactness witnesses whose paired gap reached :data:`SMOKE_GATE`."""
    return tools.exactness_failures(artifacts["exactness.csv"])


def account_failures(failures: pd.DataFrame, artifacts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """A replication that either side lost leaves its witness rows unpaired, so it counts."""
    return evidence_failures.audits(
        scientific_failures(artifacts),
        missing={tools.EXACTNESS_AUDIT: evidence_failures.missing_fits(failures, STUDY)},
    )


__all__ = ["STUDY", "draw_and_fit", "fit"]
