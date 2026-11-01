"""Registered evidence for point-treatment survival (``point-treatment-survival``).

The subject is ``LTMLE`` on a held design: one baseline treatment held over every node, read
from one time and one event column per unit (``TimeToEvent``).  The law is
``make_point_survival``: ``W1`` Bernoulli(0.5), ``W2`` uniform on ``0..3``, a binary arm,
discrete-time hazards from ``(k, A, W)``, and dropout at the visits from ``(A, W)``.  Every
truth is an exact finite sum, and every efficiency bound is the exact standard deviation of
the efficient influence function over the law's support
(:mod:`tests.studies.point_survival_common`).

**The comparator** is ``survtmle`` 1.1.1 with ``method="mean"`` (Benkeser, Carone and Gilbert
2018), on the same realized samples, through ``tests/canonical/point_survival_runner.R``:

* the treatment model is the logistic regression on ``W1 + W2`` on both sides;
* the censoring model is the logistic regression on ``trt + W1 + W2`` at each visit: one fit
  per node here, and one pooled ``glm`` on numeric time indicators with no shared term there,
  whose likelihood factorises by visit, so the two are one model;
* the sequential regressions are the quasibinomial regression on ``W1 + W2`` per arm, which
  ``survtmle`` writes ``trt * (W1 + W2)``.  The hazard is logistic in ``(k, A, W1, W2)``, so
  the earlier iterated means are misspecified on purpose, and the targeting moves the estimate
  on both sides.

The two estimators differ in their targeting submodel: this package fluctuates each node's
intercept with the clever covariate in the loss weight, and ``survtmle`` fits a logistic
fluctuation per cause on the unweighted clever covariate.  Both solve the same equations,
along different paths.  The paired rows are read under the default margins.

**One survtmle call per arm.**  ``survtmle`` 1.1.1's mean method builds its observed wide frame
from the counterfactual copy of the first arm of ``trtOfInterest`` (``makeWideDataList.R``
reads ``dataList[[2]]``), so in a call with both arms, arm 1's clever covariate divides by the
censoring survival predicted at ``trt = 0``.  Dropout here depends on ``A``, so arm 1's
targeting and influence curve used the wrong weights after visit 1: the first registered run
made that call, and its arm-1 standard errors ran 1 to 5 percent below this package's while
arm 0 agreed within 0.04 percent.  The runner now calls once per arm with ``trtOfInterest``.  The
both-arm call stays as a witness, and :func:`reference_artifacts` writes
``reference-witness.csv``: per estimand, how far the both-arm call's estimate and standard
error sit from the per-arm call's.  Arm 0 and visit 1 must agree to fitting tolerance, and arm
1 after visit 1 must not, which shows the per-arm call changed the weights it was meant to.

**The scenarios.**  ``survival`` (five integer visits; the risks of both arms and their
difference at visits 1, 3 and 5) and ``competing`` (four visits, two causes; the incidences
and their differences at visits 2 and 4), each at n = 2,000 with 1,600 replications.  RMST, the
three-arm law with a time-varying covariate, the continuous event time, the held end-of-study
outcome, the weighted law and the clustered law are property cells, because ``survtmle`` has
none of them (``point_survival_properties``).

**The initial estimate is not one quantity on the two sides.**  Here ``initial_estimate`` is
the mean of the node-1 regression, which is fitted on pseudo-outcomes that the later nodes
already targeted.  The ``survtmle`` value is a second call with ``Gcomp=TRUE``, the fully
untargeted recursion.  The two coincide at visit 1, which has no later node, and differ at
later visits.  The paired rows compare the targeted estimates only, so the initial columns are
context and enter no verdict.

**A separated fit keeps its last iterate.**  ``QuasiBinomialGLM`` keeps its last iterate,
with ``QuasiBinomialSeparationWarning``, when its coefficients diverge while its deviance
settles by R's ``glm.control`` rule.  R's ``glm`` returns such a fit with a warning, and the
comparator fits ``glm``.  A fit whose deviance has not settled still raises and fails the run.
The first declared run stopped on replication 189 of ``competing``, where the coefficient
of ``W1`` diverged at a node with no event of one cause in one ``W1`` cell, and it published
nothing.  A failure-only scan of every declared primary replication, which read no estimate,
then found such a fit in replications 189, 1,008, 1,302 and 1,398 of ``competing`` and no
failed fit.  Their rows are kept and read as every other row.

Publication policy is ``reporting``.  The red-cell route was declared before any run.
Every red cell is diagnosed before it is routed.  A genuine defect (an algorithm
defect, an inconsistent estimator, or a test or design bug that makes the cell measure the
wrong thing) is fixed and re-run under a fresh declaration commit that records the change.
Nothing changes to make a cell easier to pass: no margin, budget, law or cell moves.
A finite-sample red with no defect stays published red: a band cell under
``band-finite-sample``, a clustered cell under the X24/X25 cluster owner that applies, and any
other cell under ``X13-finite-sample``, which closes when a re-declared cell passes.  A
replication that raises is never redrawn: a failed replication fails the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_ltmle import R_BASE_IMAGE, regimen_rows
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261060
RESAMPLING_SEED = 2026106001

SURVTMLE_VERSION = "1.1.1"
SURVTMLE_TARBALL_SHA256 = "d286799265f35f2ae7631f2f1714936ff2b1866cdd95a2848c252482f9ce23d0"

REGIMENS = {"arm0": 0, "arm1": 1}
REFERENCE = "arm0"

SURVIVAL = "survival"
COMPETING = "competing"
SCENARIOS = {SURVIVAL: common.SURVIVAL, COMPETING: common.COMPETING}

#: The reported horizons of each scenario, as node indices.
REPORTED = {SURVIVAL: (1, 3, 5), COMPETING: (2, 4)}


def _names(scenario: str) -> tuple[str, ...]:
    law = SCENARIOS[scenario]
    causes: tuple[str | None, ...] = (
        tuple(common.CAUSES[c] for c in (1, 2)) if law.causes == 2 else (None,)
    )
    levels = tuple(
        common.risk_name(label, horizon, cause)
        for label in REGIMENS
        for horizon in REPORTED[scenario]
        for cause in causes
    )
    contrasts = tuple(
        common.contrast_name("arm1", REFERENCE, horizon, cause)
        for horizon in REPORTED[scenario]
        for cause in causes
    )
    return levels + contrasts


ESTIMANDS = {scenario: _names(scenario) for scenario in SCENARIOS}

#: The per-arm workaround's witness, written by :func:`reference_artifacts`.
WITNESS_FILE = "reference-witness.csv"
#: The columns the runner's rows carry beside the published schema: the both-arm call's values.
WITNESS_COLUMNS = ("joint_estimate", "joint_std_error")

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"survival_t5__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "root_n_and_efficiency": tuple(f"survival_t5__n_{size}" for size in (1_000, 2_000, 8_000)),
    "root_n_rate": ("survival_t5__empirical_sd", "survival_t5__reported_se"),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in (
            "survival_t5",
            "rmst_5",
            "competing_t4",
            "three_arm_t3",
            "continuous_t4",
            "end_of_study",
            "weighted_t5",
        )
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "power": ("survival_t5__alternative",),
    "clustered_inference": ("clustered_t5__cluster_robust", "clustered_t5__iid_control"),
    "simultaneous_coverage": (
        "all_reported__simultaneous_band",
        "all_reported__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="point-treatment survival",
    slug="point-treatment-survival",
    artifacts=ROOT / "tests" / "canonical" / "point_survival",
    document="docs/technical-reference/method-evidence/point-treatment-survival.md",
    anchor="point-treatment-survival",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="survtmle",
    failure_bound={"cleverly": 0.01, "survtmle": 0.01},
    comparison="different estimator",
    modules=(
        "tests/studies/canonical_point_survival.py",
        "tests/studies/point_survival_properties.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/survival_point.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/survtmle/Dockerfile",
        "tests/canonical/point_survival_runner.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_point_survival",
    properties_module="tests.studies.point_survival_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    extra_artifacts=(WITNESS_FILE,),
)

REFERENCE_METADATA = {
    "survtmle_version": SURVTMLE_VERSION,
    "survtmle_tarball_sha256": SURVTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "survtmle(method='mean') per horizon t0, glm.trt='W1 + W2', glm.ftime='trt*(W1 + W2)', "
        "glm.ctime with one intercept and three slopes per visit and no shared term; the "
        "initial estimate is a second call with Gcomp=TRUE; one targeted call per arm "
        "(trtOfInterest), because survtmle 1.1.1's both-arm mean call weights every arm by the "
        "first arm's censoring survival"
    ),
    "pairing": "read under the default margins; the targeting submodels differ by construction",
}

CONFIGURATION = {
    "construction": "held point-treatment survival",
    "cross_fit": False,
    "outcome_learner": "QuasiBinomialGLM on W1, W2 per arm (misspecified on purpose)",
    "treatment_learner": "unpenalized logistic regression on W1, W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A per node",
    "regimens": list(REGIMENS),
    "reported_nodes": {scenario: list(nodes) for scenario, nodes in REPORTED.items()},
}


def _plans(scenario: str) -> dict[str, Any]:
    levels = SCENARIOS[scenario].levels
    return {label: common.static(arm, levels) for label, arm in REGIMENS.items()}


TRUTH = {
    scenario: {
        name: value
        for name, value in common.truths(law, _plans(scenario), REFERENCE).items()
        if name in ESTIMANDS[scenario]
    }
    for scenario, law in SCENARIOS.items()
}


def primary_learners() -> dict[str, Any]:
    """The glm learners both sides fit."""
    return {
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": QuasiBinomialGLM(),
        "treatment_learner": LogisticRegression(penalty=None, max_iter=1000),
        "censoring_learner": LogisticRegression(penalty=None, max_iter=1000),
    }


def fit(scenario: str, frame: pd.DataFrame) -> Any:
    """The subject's primary fit of one scenario's sample."""
    law = SCENARIOS[scenario]
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        n_folds=1,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **primary_learners(),
    ).fit(law.container(frame))


def initial_estimates(result: Any, names: Sequence[str]) -> dict[str, float]:
    """The untargeted plug-in of every reported level and contrast."""
    index = result.parameter_index or {}
    plug_in: dict[tuple[str, str | None, int], float] = {}
    for fit_record in result.fits.values():
        key = (fit_record.regimen.label, fit_record.cause, fit_record.horizon)
        plug_in[key] = float(np.mean(fit_record.steps[0].initial))
    out: dict[str, float] = {}
    for name in names:
        label, cause, horizon = index[name]
        if " vs " in label:
            left, right = label.split(" vs ")
            out[name] = plug_in[(left, cause, horizon)] - plug_in[(right, cause, horizon)]
        else:
            out[name] = plug_in[(label, cause, horizon)]
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return SCENARIOS[scenario].draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[tuple[str, ...], int, int]) -> list[tuple[pd.DataFrame, dict[str, float]]]:
    scenarios, replicate, n = key
    return [draw_scenario(scenario, n, replicate) for scenario in scenarios]


def _reference_input(
    key: tuple[tuple[str, ...], int, int], drawn: list[tuple[pd.DataFrame, dict[str, float]]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's samples and truth rows, which need no fit, and no subject rows."""
    scenarios, replicate, _ = key
    samples: list[pd.DataFrame] = []
    truth_rows: list[dict[str, Any]] = []
    for scenario, (frame, truth) in zip(scenarios, drawn, strict=True):
        sample = frame.loc[:, ["W1", "W2", "A", "time", "event"]].copy()
        sample.insert(0, "row", np.arange(len(sample)))
        sample.insert(0, "replicate", replicate)
        sample.insert(0, "scenario", scenario)
        samples.append(sample)
        truth_rows.extend(
            {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
            for name, value in truth.items()
        )
    return pd.concat(samples, ignore_index=True), truth_rows, []


def _replicate(
    key: tuple[tuple[str, ...], int, int], drawn: list[tuple[pd.DataFrame, dict[str, float]]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios, replicate, _ = key
    samples, truth_rows, _ = _reference_input(key, drawn)
    rows: list[dict[str, Any]] = []
    for scenario, (frame, truth) in zip(scenarios, drawn, strict=True):
        result = fit(scenario, frame)
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
    return samples, truth_rows, rows


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of both scenarios, its truth rows and the subject's rows.

    The samples are ordered by replication and then by scenario, so the reference reads one
    replication's two samples as one contiguous group and splits them by scenario.  A key
    spans both scenarios, so its cell is their ``;``-joined names and it records no one seed.
    """
    outcomes = replication.fit_primary(
        [(tuple(SCENARIOS), replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        cell_of=lambda key: ";".join(key[0]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes,
        estimates.loc[:, list(REPLICATE_COLUMNS)],
        samples=samples,
        truths=truths,
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by scenario."""
    return TRUTH


def reference_artifacts(
    *,
    reference: Any,
    here: Path,
    samples: Path,
    truths_path: Path,
    reference_results: Path,
    output: Path,
    cores: int,
) -> dict[str, pd.DataFrame]:
    """Summarise the both-arm survtmle call against the per-arm calls, per estimand.

    ``max_abs_estimate_gap`` is the largest absolute gap between the two calls' estimates over
    the replications, ``max_abs_se_log_ratio`` the largest absolute log ratio of their
    standard errors, and ``mean_se_ratio`` the mean of the both-arm standard error over the
    per-arm one.  ``tests/unit/test_point_survival_method_study.py`` reads the file.
    """
    del reference, here, samples, truths_path, output, cores
    rows = pd.read_csv(reference_results, float_precision="round_trip")
    missing = [column for column in WITNESS_COLUMNS if column not in rows.columns]
    if missing:
        raise RuntimeError(f"survtmle rows lack the witness columns {missing}")
    rows["horizon"] = rows["estimand"].str.extract(r"@ t=(\d+)\]", expand=False).astype(int)
    rows["arm"] = np.where(
        rows["estimand"].str.startswith("ate_regimen"),
        "contrast",
        rows["estimand"].str.extract(r"\[(arm\d)", expand=False),
    )
    rows["estimate_gap"] = (rows["joint_estimate"] - rows["estimate"]).abs()
    rows["se_ratio"] = rows["joint_std_error"] / rows["std_error"]
    rows["se_log_ratio"] = np.log(rows["se_ratio"]).abs()
    summary = (
        rows.groupby(["scenario", "estimand", "arm", "horizon"], sort=True)
        .agg(
            replicates=("replicate", "size"),
            max_abs_estimate_gap=("estimate_gap", "max"),
            max_abs_se_log_ratio=("se_log_ratio", "max"),
            mean_se_ratio=("se_ratio", "mean"),
        )
        .reset_index()
    )
    return {WITNESS_FILE: summary}


def account_failures(failures: pd.DataFrame, artifacts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The witness reads the surviving reference rows, so a failed fit only drops its rows."""
    del failures, artifacts
    return evidence_failures.audits(
        {"per-arm witness (surviving rows)": pd.DataFrame()}, missing={}
    )


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]
