r"""Declaration and driver of ``score-assessment-compositions`` (P1 PR 2 plan rev 3, section 5B).

The study measures the score assessment of :meth:`cleverly.study.IdentifiedEffect.assess_score`
in the five compositions PR 2 labels experimental until a study measures them
(:data:`cleverly.effect_curves.SCORE_COMPOSITIONS_MEASURED`): sampling weights, 800 clusters of
5, a missing-at-random outcome, the pair ``(2, 0)`` of a three-arm treatment, and an ATE fit that
is not cross-fitted.  Each composition is the shipped ``score-assessment`` paired design: an
estimated mechanism, a difference-scale ATE fit, a binary outcome, and the scores
``S = 0.2 + 0.3 W1`` and ``SB = S + 0.15 sin(pi W3)`` (:data:`tests.studies.score_assessment.SCORES`,
``binary_paired``).  The laws are the ``conditional-projection`` laws
(:func:`tests.studies.conditional_projection.sample`).

Each configuration is two calibration arms (plan 5.0 rule 1): ``risk`` (the S and SB risk
differences and the paired row) and ``cal`` (the S and SB calibration intercepts and slopes).
Two controls fail where the study's machinery would hide a defect: a shrunken SE on the weighted
S risk difference, and the row-level SE that ignores the clusters on the clustered S risk
difference and S calibration slope.

The calibration ``R`` comes from the 3,200-draw pilot
(:mod:`tests.studies.score_assessment_compositions_pilot`), which has not run.  :data:`BUDGETED_R`
is the floor at which every row at a ratio of 1 passes; the declaration replaces it by the
pilot's ``R`` when larger.  The point-reading fallback of :mod:`tests.studies.point_reading`
applies at :data:`POINT_READING_TARGET` (0.95, chosen on 2026-11-20).  The study is not
registered: it joins :func:`tests.studies.evidence.registry.registered` with the artifacts of
its declared run.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as cp
from tests.studies import score_assessment as sa
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "score-assessment-compositions"

#: The record seeds.  The plan named 20261502, which ``calibration-slope-warning`` already holds
#: as its resampling seed, so the study takes the next free pair after ``projection-evidence``.
SEED = 20261521
RESAMPLING_SEED = 20261522

CRITICAL = float(stats.norm.ppf(0.975))

#: The score kind of every group: the shipped paired binary design.
KIND = "binary_paired"
FIRST, SECOND = sa.FIRST, sa.SECOND

#: The floor of the calibration ``R``: X32a's rule with every row at 1 and the pilot's 3,200
#: draws (plan 5B.3).  The pilot's ``R`` replaces it when larger.
BUDGETED_R = 4_800
#: The ``R`` the plan expects from the shipped pilot ratios (plan 5B.3), which prices the run.
PLANNED_R = 13_200
#: The family target of the point-reading fallback (plan 5.0 rule 4; chosen 2026-11-20).
POINT_READING_TARGET = 0.95
#: Pilot draws per gated cell, and the ``pilot_2`` extension (plan 5.0 rule 3).
PILOT_DRAWS = 3_200
PILOT_2_DRAWS = 3_200
#: An arm whose plug-in pass reads below this after the pilot runs ``pilot_2``.
PILOT_2_BELOW = 0.90
#: The shrunken control's factor.
SHRUNKEN_FACTOR = 0.70
#: The clustered law is strong enough for the row-level control only when the pilot's row-level
#: SE ratio of each control row is at most this (plan 5B.2); otherwise the cluster effect is
#: raised before the declaration.
ROW_VARIANCE_CEILING = 0.85


# ------------------------------------------------------------------------------ the laws

#: Every law of the study, from ``conditional-projection``.
LAWS: tuple[str, ...] = ("weighted", "clustered", "missing", "multi_arm", "observational")


def sample(law: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``, with the two score columns."""
    if law not in LAWS:
        raise KeyError(law)
    frame = cp.sample(law, n, rng)
    w = frame.loc[:, ["W1", "W2", "W3"]].to_numpy(dtype=float)
    pair = sa.SCORES[KIND]
    frame[FIRST] = pair.first(w)
    assert pair.second is not None
    frame[SECOND] = pair.second(w)
    return frame


def truth(law: str, *, nodes: int = sa.QUADRATURE_NODES) -> dict[str, float]:
    """The exact value of every parameter on ``law``, keyed by its row label.

    The arm regressions of the contrasted pair are the ``observational`` ones on every law but
    ``clustered``, whose arm regressions are marginal over the cluster effect: the ``multi_arm``
    law's pair ``(2, 0)`` reads ``expit(logit(a / 2))`` at ``a = 2`` and ``0``, and the
    ``missing`` and ``weighted`` laws keep the observational outcome.  The ``weighted`` truth is
    the target population's, whose ``W`` law is tilted by the sampling weight
    (:func:`tests.studies.conditional_projection.law_mass`).
    """
    w, mass = cp._grid(nodes)
    mass = cp.law_mass(law, w, mass)
    q_t, q_r = cp.arm_means(law, w)
    return sa._parameters("difference", q_t, q_r, sa.SCORES[KIND], w, mass)


# ------------------------------------------------------------------------- the fit groups


CALIBRATION = "interval_calibration"

RISK_ROWS: tuple[str, ...] = (f"{FIRST}_risk_difference", f"{SECOND}_risk_difference", "paired")
CALIBRATION_ROWS: tuple[str, ...] = (
    f"{FIRST}_calibration_intercept",
    f"{FIRST}_calibration_slope",
    f"{SECOND}_calibration_intercept",
    f"{SECOND}_calibration_slope",
)

#: The shared calibration kind each control's cells end in.
CONTROL_KINDS: Mapping[str, str] = {
    "shrunken_se": "shrunken_se_control",
    "row_variance": "row_variance_control",
}

#: What each control computes from its group's fits, and the rule it must fail (plan 5B.2).
CONTROL_CLAIMS: Mapping[str, str] = {
    "shrunken_se": "the reported SE times 0.70: its SE-ratio interval must fall below the band",
    "row_variance": "the row-level SE that ignores the clusters, sd(IC) / sqrt(n): its "
    "SE-ratio interval must fall below the band (about 0.79 on one 4,000-row draw)",
}


@dataclass(frozen=True)
class Group:
    """One declared fit group: a law, a size, its replications, and what its fits fill.

    Parameters
    ----------
    label : str
        The group key, which the seed stream and the cell names read.
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows per replication.
    replicates : int
        Replications in the declared run.
    code : str
        The composition code the group measures, a key of
        :data:`cleverly.effect_curves.SCORE_COMPOSITIONS_MEASURED`.
    risk_arm : str
        The calibration arm of the group's risk rows (:data:`RISK_ROWS`).
    calibration_arm : str
        The calibration arm of the group's calibration rows (:data:`CALIBRATION_ROWS`).
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the rows it publishes.
    cross_fit : bool
        Whether the ATE fit is cross-fitted.
    seconds : float
        Single-thread seconds per replication, measured by the 20-replicate smoke on the
        declaration's container (:data:`SMOKE_TIMINGS`).
    """

    label: str
    law: str
    n: int
    replicates: int
    code: str
    risk_arm: str
    calibration_arm: str
    controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cross_fit: bool = True
    seconds: float = 0.3

    @property
    def arms(self) -> tuple[str, str]:
        """The group's two calibration arms."""
        return (self.risk_arm, self.calibration_arm)

    def rows(self, arm: str) -> tuple[str, ...]:
        """The row labels of ``arm``."""
        return RISK_ROWS if arm == self.risk_arm else CALIBRATION_ROWS


R = BUDGETED_R
GROUPS: tuple[Group, ...] = (
    Group(
        "assess_weighted_2k",
        "weighted",
        2_000,
        R,
        "weights",
        "__w_risk",
        "__w_cal",
        controls={"shrunken_se": (f"{FIRST}_risk_difference",)},
        seconds=0.12,
    ),
    Group(
        "assess_clustered_4k",
        "clustered",
        4_000,
        R,
        "clusters",
        "__cl_risk",
        "__cl_cal",
        controls={"row_variance": (f"{FIRST}_risk_difference", f"{FIRST}_calibration_slope")},
        seconds=0.17,
    ),
    Group(
        "assess_missing_4k", "missing", 4_000, R, "missing", "__mis_risk", "__mis_cal", seconds=0.17
    ),
    Group(
        "assess_arms3_4k", "multi_arm", 4_000, R, "multi_arm", "__k3_risk", "__k3_cal", seconds=0.35
    ),
    Group(
        "assess_insample_2k",
        "observational",
        2_000,
        R,
        "not_cross_fitted",
        "__in_risk",
        "__in_cal",
        cross_fit=False,
        seconds=0.03,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}


def calibration_cell(group: Group, arm: str, label: str) -> str:
    """The ``interval_calibration`` cell of ``label`` in ``arm`` (``<group>_<label><arm>``)."""
    return f"{group.label}_{label}{arm}"


def control_cell(group: Group, control: str, label: str) -> str:
    """The cell of ``control``'s row for ``label`` (``<group>_<label>__<kind>``)."""
    return f"{group.label}_{label}__{CONTROL_KINDS[control]}"


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells = [calibration_cell(group, arm, label) for arm in group.arms for label in group.rows(arm)]
    cells += [
        control_cell(group, control, label)
        for control, labels in group.controls.items()
        for label in labels
    ]
    return {CALIBRATION: tuple(cells)}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order."""
    return {CALIBRATION: tuple(c for g in GROUPS for c in group_cells(g)[CALIBRATION])}


PROPERTY_CELLS = property_cells()


def calibration_arms() -> dict[str, tuple[str, ...]]:
    """The gated calibration arms and their rows."""
    return {
        arm: tuple(calibration_cell(group, arm, label) for label in group.rows(arm))
        for group in GROUPS
        for arm in group.arms
    }


CALIBRATION_ARMS = calibration_arms()


# ---------------------------------------------------------------------------------- the fits


def learners(group: Group) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``: the shipped spline per arm."""
    settings: dict[str, Any] = {
        "outcome_learner": cp.spline_glm(),
        "treatment_learner": LogisticRegression(max_iter=2_000),
        "outcome_by_arm": True,
    }
    if group.law == "missing":
        settings["missingness_learner"] = LogisticRegression(max_iter=2_000)
    if not group.cross_fit:
        settings["cross_fit"] = False
    return settings


def design(group: Group) -> PointTreatment:
    """The study design of ``group``'s law."""
    roles: dict[str, Any] = {}
    if group.law == "missing":
        roles["missingness"] = "Delta"
    if group.law == "weighted":
        roles["weights"] = "weight"
    if group.law == "clustered":
        roles["cluster"] = "cluster"
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles)


def estimand(group: Group) -> ConditionalContrast:
    """The ATE fit's estimand: the shipped ``Linear(W1, W3)`` contrast, ``(2, 0)`` on three arms."""
    pair = {"treatment": 2, "reference": 0} if group.law == "multi_arm" else {}
    return ConditionalContrast(sa.MODIFIERS, model=sa.BASIS, **pair)  # type: ignore[arg-type]


def assess(group: Group, frame: pd.DataFrame, random_state: int) -> Any:
    """The paired score assessment of one replication."""
    pair = sa.SCORES[KIND]
    effect = CausalStudy(frame, design=design(group)).identify(estimand(group))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.assess_score(
            FIRST,
            computed_from=list(pair.first_inputs),
            score_kind="external",
            compare=SECOND,
            compare_computed_from=list(pair.second_inputs),
            random_state=random_state,
            **learners(group),
        )


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate of each row label; ``"controls"``: each
        control's standard error by row label; ``"status"``: the assessment's support label.
    """
    assessment = assess(group, frame, random_state)
    estimates = {sa._label(score, name): est for score, name, est in assessment.rows()}
    controls: dict[str, dict[str, float]] = {}
    for control, labels in group.controls.items():
        values = {}
        for label in labels:
            estimate = estimates[label]
            if control == "shrunken_se":
                values[label] = SHRUNKEN_FACTOR * reported_inference(estimate)[0]
            elif control == "row_variance":
                curve = np.asarray(estimate.influence_curve, dtype=float)
                values[label] = float(np.std(curve, ddof=1) / math.sqrt(len(curve)))
            else:
                raise KeyError(control)
        controls[control] = values
    return {"estimates": estimates, "controls": controls, "status": assessment.status}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = GROUP_BY_LABEL[group_label]
    return sample(group.law, n, np.random.default_rng(seed)), truth(group.law)


# ----------------------------------------------------------------------- the primary phase

#: The primary phase (plan 5B.4): the weighted group's paired row on the primary stream.
PRIMARY_GROUP = "assess_weighted_2k"
PRIMARY_REPLICATES = 2_400
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n
PRIMARY_NAME = f"risk_difference[{FIRST} - {SECOND}]"
#: The coverage floor the primary reads (plan 5B.4).
PRIMARY_COVERAGE_FLOOR = 0.90

SCENARIOS = {PRIMARY_GROUP: (PRIMARY_NAME,)}

STUDY = StudyRecord(
    name="Score assessment compositions",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "score_assessment_compositions",
    document="docs/technical-reference/method-evidence/score-assessment-compositions.md",
    anchor="score-assessment-compositions",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-score-assessment",
    reference=None,
    failure_bound={"cleverly-score-assessment": 0.01},
    modules=(
        "tests/studies/score_assessment_compositions.py",
        "tests/studies/score_assessment_compositions_pilot.py",
        "tests/studies/score_assessment_compositions_properties.py",
        "tests/studies/score_assessment.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/point_reading.py",
        "tests/studies/repeated_sizing.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "src/cleverly/effect_curves.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/inference/influence.py",
        "src/cleverly/_inference_status.py",
    ),
    runner_module="tests.studies.score_assessment_compositions",
    properties_module="tests.studies.score_assessment_compositions_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, composition {group.code}"
        f"{'' if group.cross_fit else ', not cross-fitted'}"
    )
    for group in GROUPS
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary row of one replication: the paired risk difference."""
    group = GROUP_BY_LABEL[scenario]
    fold_seed = stream_seed(STUDY, "fold_partition", "primary", scenario, replicate)
    estimate = fit_group(group, frame, random_state=fold_seed)["estimates"]["paired"]
    reference = float(truths["paired"])
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": len(frame),
            "estimand": PRIMARY_NAME,
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": float("nan"),
        }
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truths = drawn
    return primary_rows(frame, truths, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ---------------------------------------------------------------------- the design numbers


def design_numbers() -> dict[str, Any]:
    """The numbers the design test recomputes."""
    return {
        "truths": {law: truth(law) for law in LAWS},
        "calibration_arms": {arm: len(rows) for arm, rows in CALIBRATION_ARMS.items()},
        "cpu_hours": run_cpu_hours(),
        "planned_cpu_hours": run_cpu_hours(PLANNED_R),
        "pilot_cpu_hours": pilot_cpu_hours(),
    }


def run_cpu_hours(replicates: int | None = None) -> float:
    """Single-thread CPU-hours of the property run at ``replicates`` per group, and the primary."""
    primary = GROUP_BY_LABEL[PRIMARY_GROUP].seconds * PRIMARY_REPLICATES
    run = sum(g.seconds * (g.replicates if replicates is None else replicates) for g in GROUPS)
    return (run + primary) / 3_600.0


def pilot_cpu_hours(draws: int = PILOT_DRAWS) -> float:
    """Single-thread CPU-hours of the pilot: ``draws`` per group."""
    return sum(g.seconds * draws for g in GROUPS) / 3_600.0


def smoke_cpu_hours(replicates: int = 20) -> float:
    """Single-thread CPU-hours of a ``replicates``-draw smoke of every group."""
    return sum(g.seconds * replicates for g in GROUPS) / 3_600.0


#: Plan 5.0's cost rule: a group whose smoke cost exceeds this multiple of its declared
#: ``seconds`` runs as a second declared run.
COST_GATE = 1.5

#: The smoke timings (``group``, ``fits``, ``seconds_per_fit``), written by
#: ``tests/canonical/score_assessment_compositions/regenerate.py --property-smoke 20
#: --write-timings``.
SMOKE_TIMINGS = ROOT / "tests" / "studies" / "score_assessment_compositions_smoke.csv"

#: Every file the declared run writes into :attr:`StudyRecord.artifacts`.
RUN_FILES = (
    "replicates.parquet",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.parquet",
    "properties.csv",
    "manifest.json",
    "run.log",
)
CONDITIONAL_RUN_FILES = (
    "property-failures.csv",
    "property-failure-accounting.csv",
    "property-failure-witnesses.parquet",
)
