"""Registered evidence for declared supports, branches and atoms of an MTP (``mtp-boundaries``).

The subject is a modified treatment policy on a continuous dose whose declared support depends
on the history (``continuous_treatment={"A1": Support(0, u), "A2": Support(0, u)}``), with
maps that bind at that support, change their Jacobian with the dose, have two inverse branches,
or send an interval to a declared atom (``dose_atoms={"A2": (3.0,)}``).  ``LTMLE`` and ``SDR``
read it at one fold on the density route.  The laws, the plans, the truths and the oracle
learners live in :mod:`tests.studies.mtp_boundaries_common`.  The comparator is R ``lmtp``
1.5.4.  Four primary scenarios, 1,000 replications at n = 5,000:

=====================  =================================================================
scenario               subject and pairing
=====================  =================================================================
``boundaries_ltmle``   law B, ``LTMLE``, plans ``natural`` and ``p1`` to ``p4``.  Each
                       node reads the oracle binned density
                       (:class:`~tests.studies.mtp_boundaries_common.OracleHazard`,
                       :func:`~tests.studies.oracle_density_bins.oracle_bins` bins).  The
                       per-node ratio and the shifted values are written beside the sample.
                       Exactness witness
``boundaries_sdr``     the same draw (``scenario_seed_owners``), ``SDR``.  Exactness witness
``atom_ltmle``         law B', ``LTMLE``, plans ``natural`` and ``p5``.  Node 2 reads the
                       oracle mixed density: the exact atom probability and the oracle
                       hazard of the rows with no atom.  Margin-read: the package's node-2
                       outcome design holds the atom indicator ``1{A2 = 3}``, which
                       ``lmtp``'s shifted frame cannot carry
``atom_sdr``           the same draw as ``atom_ltmle``, ``SDR``.  Margin-read, as above
=====================  =================================================================

**Learners.**  The primary rows match the comparator's ``SL.glm``: a quasibinomial GLM on the
node-2 history and dose at the last node, and least squares at the first node.  The property
rows read the correctly specified final model
(:class:`~tests.studies.mtp_boundaries_common.CorrectOutcome`) and a rich first-node basis
(:class:`~tests.studies.mtp_boundaries_common.RichPseudo`).

**The comparator.**  ``tests/canonical/mtp_boundaries_runner.R`` calls ``lmtp_mtp_tmle``
(``tests/canonical/lmtp_mtp_adapter.R``) for the ``LTMLE`` scenarios and
``lmtp_sdr_with_folds`` (``tests/canonical/lmtp_sdr_adapter.R``) for the ``SDR`` scenarios.
Both read the columns ``shift__<plan>__<node>__1`` and ``ratio__<plan>__<node>``, so neither
side estimates a ratio the other does not see.  ``exactness.csv`` (:func:`extra_artifacts`)
records the largest paired gap in estimate and standard error of each estimand of
:data:`EXACT`, and a gap of :data:`SMOKE_GATE` or more is a red row.  The runner passes
``refuse_bound = FALSE``, so a prediction at ``lmtp``'s bound moves a pair and that table reads
it.  The pre-declaration R smoke of 3 replications (2026-10-16) read the witnesses at most
2.3e-10 apart in estimate and 8.5e-12 in standard error.  The two atom scenarios are not
witnesses.  Both sides target the same parameter, but the package's node-2 regression holds the
regressor ``1{A2 = 3}``.  ``lmtp`` substitutes the shifted dose into its treatment column and
carries every other column unchanged, so it cannot recompute that regressor at the shifted dose.
Two different working models give two different initial fits and so two different targeted
estimates, which agree only to the paired margins.  The smoke read the ``p5`` rows 6.3e-04
apart in estimate and 8.7e-05 in standard error, and the ``natural`` rows, where no dose moves,
1.4e-15 apart.  These rows are comparator evidence.

**Properties.**  :mod:`tests.studies.mtp_boundaries_properties` declares every family, its
sizes and the pilots behind them.

**Second declaration (2026-10-16).**  The first run (cleverly-tmle ea94ee79, n = 2,000) read
the two ``boundaries_ltmle`` ``p4`` rows red: coverage 0.916 and 0.922, SE ratio 0.925 (99%
interval 0.872 to 0.982), bias under 0.03 SD.  The ``p4`` calibration cell shows the same
finite-sample shortfall of the influence-curve variance at n = 2,000 and passes at n = 4,000
(SE ratio 0.971).  As for ``mtp-pricing`` (Eric's decision, 2026-10-16), the primary moves to
n = 5,000; the property cells keep their declared sizes and their committed rows.  The
n = 2,000 reading stays in the method-evidence page.

**Red-cell reading, declared before any run.**  Publication policy is ``reporting``, and the
reading below is the merge gate, as in ``vector-mtp``.  The fast tier gates the artefacts'
integrity, and the pull-request review gates the reading.  Every red cell is diagnosed first.
A genuine defect is fixed, and the study re-runs under a fresh declaration that records the
change.  A red ``interval_calibration`` positive is a finite-sample red only if its SE-ratio
99% interval contains 1, its coverage interval is consistent with 0.95, and the diagnosis
finds no defect.  It is then published red with the owner row
``mtp-boundaries-calibration``, declared before any repeat.  Any other red calibration
positive, a red necessity row, a red exactness-witness row and a red primary truth row block
the merge.  A red margin-read paired row is the construction difference above and gets the
owner row ``mtp-boundaries-pairing``.  Nothing that makes a cell easier to pass moves after a
verdict: no margin, no extra replications, no weaker law, no relabelling, no oracle
substitution.  A replication that raises is never redrawn.

**Budget.**  :data:`~tests.studies.mtp_boundaries_properties.BUDGET` states the per-fit
seconds the one-replicate smoke measured and the CPU-hours they imply.
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
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE, SDR
from tests.parallel import STUDY_JOBS
from tests.studies import mtp_boundaries_common as common
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
from tests.studies.oracle_density_bins import oracle_bins

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 5_000
SEED = 20264011
RESAMPLING_SEED = 2026401101
#: The pre-declaration smoke gate on the exactness witnesses.
SMOKE_GATE = 1e-6
#: Bounds wide enough that no cumulative probability of these laws reaches them.
G_BOUNDS = (0.001, 1.0)
REFERENCE = common.REFERENCE
#: The share of one cell's replications that may fail before the cell is red.
FAILURE_BOUND = 0.01

BOUNDARIES_LTMLE = "boundaries_ltmle"
BOUNDARIES_SDR = "boundaries_sdr"
ATOM_LTMLE = "atom_ltmle"
ATOM_SDR = "atom_sdr"

#: Each scenario: ``(law, estimator, plans)``.
SCENARIOS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    BOUNDARIES_LTMLE: ("B", "ltmle", common.B_LABELS),
    BOUNDARIES_SDR: ("B", "sdr", common.B_LABELS),
    ATOM_LTMLE: ("B_atom", "ltmle", common.B_ATOM_LABELS),
    ATOM_SDR: ("B_atom", "sdr", common.B_ATOM_LABELS),
}
EXACT = frozenset({BOUNDARIES_LTMLE, BOUNDARIES_SDR})


def names(labels: tuple[str, ...]) -> tuple[str, ...]:
    """Each plan's mean and its contrast with ``natural``."""
    return tuple(f"ey_regimen[{label}]" for label in labels) + tuple(
        f"ate_regimen[{label} vs {REFERENCE}]" for label in labels if label != REFERENCE
    )


ESTIMANDS: dict[str, tuple[str, ...]] = {
    scenario: names(labels) for scenario, (_, _, labels) in SCENARIOS.items()
}


def truth_of(scenario: str) -> dict[str, float]:
    """The exact value of every estimand of a scenario."""
    law, _, labels = SCENARIOS[scenario]
    means = {label: common.truth(label, law) for label in labels}
    out = {f"ey_regimen[{label}]": value for label, value in means.items()}
    for label, value in means.items():
        if label != REFERENCE:
            out[f"ate_regimen[{label} vs {REFERENCE}]"] = value - means[REFERENCE]
    return out


#: Every published property cell (:mod:`tests.studies.mtp_boundaries_properties`).
CALIBRATION_LABELS = ("p1", "p2", "p3", "p4", "p5", "p4_sdr", "p5_sdr")
NECESSITY_CELLS: dict[str, tuple[str, str, str]] = {
    "jacobian_necessity": ("p3", "declared", "jacobian_dropped_control"),
    "branch_necessity": ("p4", "declared", "one_branch_control"),
    "atom_necessity": ("p5", "declared", "mass_dropped_control"),
    "ratio_necessity": ("p2", "declared", "unit_ratio_control"),
}
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    **{
        family: (f"{label}__{positive}", f"{label}__{control}")
        for family, (label, positive, control) in NECESSITY_CELLS.items()
    },
}

STUDY = StudyRecord(
    name="modified treatment policies at declared boundaries",
    slug="mtp-boundaries",
    artifacts=ROOT / "tests" / "canonical" / "mtp_boundaries",
    document="docs/technical-reference/method-evidence/mtp-boundaries.md",
    anchor="mtp-boundaries",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={BOUNDARIES_SDR: BOUNDARIES_LTMLE, ATOM_SDR: ATOM_LTMLE},
    margins=Margins(),
    implementation="cleverly-mtp-boundaries",
    reference="lmtp",
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_mtp_boundaries.py",
        "tests/studies/mtp_boundaries_properties.py",
        "tests/studies/mtp_boundaries_common.py",
        "tests/studies/mtp_study_tools.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/canonical_ltmle.py",
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
        "tests/canonical/mtp_boundaries_runner.R",
    ),
    runner_module="tests.studies.canonical_mtp_boundaries",
    properties_module="tests.studies.mtp_boundaries_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    failure_bound={"cleverly-mtp-boundaries": FAILURE_BOUND, "lmtp": FAILURE_BOUND},
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
    "pairing": {
        BOUNDARIES_LTMLE: "exact under SMOKE_GATE",
        BOUNDARIES_SDR: "exact under SMOKE_GATE",
        ATOM_LTMLE: "default margins: the package's outcome design holds the atom indicator",
        ATOM_SDR: "default margins: the package's outcome design holds the atom indicator",
    },
}

CONFIGURATION = {
    "construction": "ordinary, one fold",
    "density_bins": "ceil(320 (n / 2000)^(2/3)): 320 at n = 2,000, index-only oracle design",
    "g_bounds": list(G_BOUNDS),
    "support": "Support(0, 6 + 2 W2) at A1 and A2",
    "dose_atoms": {"B_atom": {"A2": [common.ATOM]}},
    "outcome_learner": "QuasiBinomialGLM at node 2; LinearRegression at node 1",
    "treatment_learner": "OracleHazard (exact pooled-hazard bins; exact atom probability)",
    "plans": {scenario: list(labels) for scenario, (_, _, labels) in SCENARIOS.items()},
    "reference": REFERENCE,
}


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def edges_of(frame: pd.DataFrame) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The bin edges each node's density reads: node 2 of law B' reads the rows off the atom."""
    bins = oracle_bins(len(frame))
    a1 = frame["A1"].to_numpy(dtype=float)
    a2 = frame["A2"].to_numpy(dtype=float)
    off_atom = np.abs(a2 - common.ATOM) > 1e-12 * common.ATOM
    return (
        tuple(float(v) for v in bin_edges(a1, bins)),
        tuple(float(v) for v in bin_edges(a2[off_atom], bins)),
    )


Outcome = Literal["primary", "correct", "wrong", "dose_blind"]


def learners(frame: pd.DataFrame, outcome: Outcome) -> tuple[Any, Any, Any]:
    """``(outcome, pseudo, treatment)`` learners of one configuration."""
    if outcome == "primary":
        q: Any = QuasiBinomialGLM()
        first: Any = LinearRegression()
    elif outcome == "correct":
        q, first = common.CorrectOutcome(), common.RichPseudo()
    elif outcome == "wrong":
        q, first = common.WrongOutcome(), LinearRegression()
    elif outcome == "dose_blind":
        q, first = common.DoseBlindOutcome(), LinearRegression()
    else:
        raise ValueError(outcome)
    return q, first, common.OracleHazard(*edges_of(frame))


def fit(
    frame: pd.DataFrame,
    law: str,
    labels: tuple[str, ...],
    *,
    estimator: str = "ltmle",
    outcome: Outcome = "primary",
    ratio: Literal["density", "classifier"] = "density",
    treatment_learner: Any = None,
    pseudo_learner: Any = None,
) -> Any:
    """One fold of ``LTMLE`` or ``SDR`` over ``labels`` on law B or B'."""
    q, first, g = learners(frame, outcome)
    if pseudo_learner is not None:
        first = pseudo_learner
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": clone(q),
        "pseudo_learner": clone(first),
        "treatment_learner": clone(g if treatment_learner is None else treatment_learner),
        "n_folds": 1,
        "g_bounds": G_BOUNDS,
        "density_bins": oracle_bins(len(frame)),
        "ratio": ratio,
        "simultaneous": False,
        "random_state": 0,
    }
    if estimator == "ltmle":
        model: Any = LTMLE(common.regimens(labels), max_iter=100, tol=1e-10, **settings)
    elif estimator == "sdr":
        model = SDR(common.regimens(labels), **settings)
    else:
        raise ValueError(estimator)
    return model.fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        continuous_treatment={"A1": common.SUPPORT, "A2": common.SUPPORT},
        dose_atoms={"A2": (common.ATOM,)} if law == "B_atom" else None,
    )


def fit_scenario(frame: pd.DataFrame, scenario: str) -> Any:
    law, estimator, labels = SCENARIOS[scenario]
    return fit(frame, law, labels, estimator=estimator)


# ---------------------------------------------------------------------------- rows


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    law = SCENARIOS[scenario][0]
    return common.SAMPLERS[law](n, seed), truth_of(scenario)


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
    labels = SCENARIOS[scenario][2]
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

    ``shift__<plan>__<node>__1`` holds the value the plan assigns at a node, and
    ``ratio__<plan>__<node>`` the fit's :attr:`~cleverly.longitudinal.RegimenFit.node_ratio`.
    """
    data = result.data
    out: dict[str, np.ndarray] = {}
    for label, plan in common.regimens(SCENARIOS[scenario][2]).items():
        fitted = result.fits[label]
        assert fitted.node_ratio is not None
        nodes = plan.plan if hasattr(plan, "plan") else (plan, plan)
        for time, policy in enumerate(nodes, start=1):
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
    frame, truth = drawn
    result = fit_scenario(frame, scenario)
    sample = frame.copy()
    for name, values in pairing_columns(result, frame, scenario).items():
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
    # and the comparator's runner reads them as one group.  The two laws share their columns,
    # and the pairing columns of a scenario's plans are empty in the others' rows.
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


__all__ = ["STUDY", "draw_and_fit", "fit"]


# ---------------------------------------------------------------------------- exactness


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``exactness.csv``: each witness estimand's largest paired gap against :data:`SMOKE_GATE`."""
    return {
        "exactness.csv": tools.exactness_table(
            rows,
            subject=STUDY.implementation,
            reference=str(STUDY.reference),
            scenarios=sorted(EXACT),
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
        missing={
            tools.EXACTNESS_AUDIT: evidence_failures.missing_fits(
                failures.loc[failures["scenario"].isin(EXACT)], STUDY
            )
        },
    )
