r"""Declaration and driver of the ``ep-assessment`` study (P1 plan, section 5.4).

The study measures the assessment readings of an EP curve
(:class:`cleverly.effect_curves.CurveDiagnostics`); each gated reading has a control that can
fail, and the stability reading is reported only.  Both
laws are ``ep-cate``'s hte3 code law at d = 3, moderate overlap, n = 2,000, with the ensemble
nuisances and the schedule size ``k(2000) = 4`` cosines per modifier:

- the **strong** law, ``d3_complex_moderate`` (CATE ``1 + sum W + sum sin 5W``);
- the **weak** law, the same baseline and propensity with the constant CATE
  :data:`tests.studies.ep_cate.WEAK_CATE` = 0.1.

**Calibration** (strong, :data:`CALIBRATION_REPLICATES`).  The out-of-fold calibration slope
of the EP curve (5 folds, the producing procedure refitted on each training fold): the 99%
two-sided Student interval of its mean lies inside ``[0.85, 1.15]``.  Control: the stretched
curve ``m + 1.5 (p - m)`` (``p`` the out-of-fold predictions, ``m`` their mean) read on the
same folds and pseudo-outcomes.  A least-squares slope scales inversely with its predictor,
so its slope is exactly ``1/1.5`` of the curve's on every replicate, and its 99% upper bound
must lie below 0.85.  It is a deterministic resolution check of the reading, a mutation rather
than an independent failure (plan nit n2); it fails whenever the EP claim's upper bound is
below 1.275.

**Stability** (strong and weak, :data:`STABILITY_REPLICATES` each, 5 split seeds; reported,
not gated: plan amendment rev 2.4).  The reading is
:meth:`~cleverly.effect_curves.CurveDiagnostics.stability`'s ``relative_sd`` and whether it
exceeds the shipped split-sensitivity warning 0.5.  Plan 5.4 claimed that the weak law reads
above 0.5; the pilot reads 0.088 (SD 0.027) there and 0.009 at the strong law.  The refits
share the draw and differ only in the cross-fitting split, and a curve of noise at a constant
effect varies with the draw, so the reading measures split sensitivity, not sampling noise, and
no control could make its warning fail.  The strong-law replicates are the first
:data:`STABILITY_REPLICATES` calibration replicates, which share their EP fit.

**Risk comparison** (finding 26).  On the same out-of-fold predictions, the reported
EP-minus-constant DR risk difference against its exact value at the true CATE,
``mean_i (p_i - tau_0(W_i))^2 - (c_i - tau_0(W_i))^2``: at the strong law, the share of
replicates whose reported difference has the exact value's sign has a 99% Clopper-Pearson
lower bound of at least 0.80.  The claim is near certain at the strong law (the pilot's signed
difference is 2.00 with SD 0.091), so it detects only a gross defect, such as a sign error in
the risk's cross term; the accuracy of the reading is reported by the ``risk_difference`` and
``exact_difference`` rows, not gated (design review S5 of PR-b).  At the weak law the true
difference is of the order of the estimation SD, so the share is reported only.

The run is ``python -m tests.canonical.ep_assessment.regenerate``.  The replicate counts are
the plan's; :func:`design_numbers` computes every claim's pass probability from the committed
pilot (:mod:`tests.studies.ep_assessment_pilot`), and the design test refuses a declaration
below 0.90.  The primary phase is the projection TMLE of ``obs_4k`` on this study's seed, as
in ``ep-cate``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import ep_cate
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

SLUG = "ep-assessment"
SEED = 20261125
RESAMPLING_SEED = 20261126

N = 2_000
N_FOLDS = 5
N_SEEDS = 5
CONFIDENCE = 0.99
TARGET_PASS = 0.90

CALIBRATION_BAND = (0.85, 1.15)
STRETCH = 1.5
STABILITY_THRESHOLD = 0.5
SHARE_FLOOR = 0.80

CALIBRATION_REPLICATES = 300
STABILITY_REPLICATES = 100

CALIBRATION = "calibration"
STABILITY = "stability"
RISK = "risk_comparison"
DIAGNOSTIC = "diagnostic"

STRONG = ep_cate.LAW_BY_LABEL["d3_complex_moderate"]
WEAK = ep_cate.Law("d3_weak_moderate", 3, "constant", "moderate", (N,))


@dataclass(frozen=True)
class Group:
    label: str
    law: ep_cate.Law
    replicates: int


GROUPS: tuple[Group, ...] = (
    Group("strong_2k", STRONG, CALIBRATION_REPLICATES),
    Group("weak_2k", WEAK, STABILITY_REPLICATES),
)
GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}


# ---------------------------------------------------------------- one replicate


def calibration_reading(diagnostics: Any, seed: int) -> dict[str, Any]:
    """The out-of-fold slope, the stretched control's slope and the DR risk difference.

    The slope is :meth:`CurveDiagnostics.calibration`'s on the difference scale, from the
    same refits (:func:`tests.unit.test_ep_assessment_design` pins the equality).
    """
    predictions, folds = diagnostics._out_of_fold(N_FOLDS, seed)
    _, _, _, pseudo = diagnostics._pieces()
    _, _, d_t, d_r = pseudo
    target = d_t - d_r

    def slope(values: np.ndarray) -> float:
        design = np.column_stack([np.ones(values.size), values])
        return float(np.linalg.lstsq(design, target, rcond=None)[0][1])

    centre = float(np.mean(predictions))
    stretched = centre + STRETCH * (predictions - centre)
    constant = np.empty_like(predictions)
    for k in range(N_FOLDS):
        constant[folds == k] = np.mean(target[folds != k])
    reported = diagnostics._risk(predictions, "difference", pseudo) - diagnostics._risk(
        constant, "difference", pseudo
    )
    return {
        "slope": slope(predictions),
        "stretched_slope": slope(stretched),
        "predictions": predictions,
        "constant": constant,
        "risk_difference": float(reported),
    }


def exact_difference(law: ep_cate.Law, frame: pd.DataFrame, reading: dict[str, Any]) -> float:
    """``mean (p - tau_0)^2 - (c - tau_0)^2`` over the fit rows at the true CATE."""
    tau = ep_cate.cate(law, frame.loc[:, list(law.columns)].to_numpy())
    return float(np.mean((reading["predictions"] - tau) ** 2 - (reading["constant"] - tau) ** 2))


def fit_replicate(group: Group, frame: pd.DataFrame, replicate: int) -> dict[str, float]:
    """Every reading one replicate of ``group`` fills, and its CPU seconds by phase."""
    start = time.process_time()
    curve = ep_cate.fit_ep(group.law, frame, "ensemble", replicate, size=ep_cate.schedule)
    diagnostics = curve.diagnostics
    reading = calibration_reading(diagnostics, replicate)
    exact = exact_difference(group.law, frame, reading)
    out = {
        "slope": reading["slope"],
        "stretched_slope": reading["stretched_slope"],
        "risk_difference": reading["risk_difference"],
        "exact_difference": exact,
        "sign_agrees": float(np.sign(reading["risk_difference"]) == np.sign(exact)),
        "calibration_seconds": time.process_time() - start,
    }
    if replicate < STABILITY_REPLICATES:
        stability = diagnostics.stability(n_seeds=N_SEEDS, random_state=replicate)
        # The default sieve is additive over modifiers, not one Cosine or BSpline, so the
        # reading has no alternative basis family (``alternative_relative_sd`` is None).
        out["relative_sd"] = stability.relative_sd
    return out


# ---------------------------------------------------------------- cells


#: ``(family, cell suffix, role, group, requested)`` of every property cell.
def cell_table() -> tuple[tuple[str, str, str, str, int], ...]:
    c, s = CALIBRATION_REPLICATES, STABILITY_REPLICATES
    return (
        (CALIBRATION, "slope", "positive", "strong_2k", c),
        (CALIBRATION, "stretched_slope", "control", "strong_2k", c),
        (STABILITY, "warns", DIAGNOSTIC, "strong_2k", s),
        (STABILITY, "relative_sd", DIAGNOSTIC, "strong_2k", s),
        (STABILITY, "warns", DIAGNOSTIC, "weak_2k", s),
        (STABILITY, "relative_sd", DIAGNOSTIC, "weak_2k", s),
        (RISK, "sign_agrees", "positive", "strong_2k", c),
        (RISK, "risk_difference", DIAGNOSTIC, "strong_2k", c),
        (RISK, "exact_difference", DIAGNOSTIC, "strong_2k", c),
        (RISK, "sign_agrees", DIAGNOSTIC, "weak_2k", s),
        (RISK, "risk_difference", DIAGNOSTIC, "weak_2k", s),
        (RISK, "exact_difference", DIAGNOSTIC, "weak_2k", s),
        (CALIBRATION, "slope", DIAGNOSTIC, "weak_2k", s),
    )


CELLS = cell_table()


def cell_name(group: str, suffix: str) -> str:
    return f"{group}__{suffix}"


def property_cells() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for family, suffix, _, group, _ in CELLS:
        out.setdefault(family, []).append(cell_name(group, suffix))
    return {family: tuple(cells) for family, cells in out.items()}


PROPERTY_CELLS = property_cells()


def row_values(group: Group, fitted: Mapping[str, float]) -> dict[tuple[str, str], float]:
    """``(family, cell) -> value`` of one replicate."""
    derived = dict(fitted)
    if "relative_sd" in fitted:
        derived["warns"] = float(fitted["relative_sd"] > STABILITY_THRESHOLD)
    out = {}
    for family, suffix, _, label, _ in CELLS:
        if label == group.label and suffix in derived:
            out[(family, cell_name(label, suffix))] = float(derived[suffix])
    return out


# ---------------------------------------------------------------- the study record

PRIMARY_GROUP = ep_cate.PRIMARY_GROUP
PRIMARY_REPLICATES = ep_cate.PRIMARY_REPLICATES
PRIMARY_N = ep_cate.PRIMARY_N
SCENARIOS = ep_cate.SCENARIOS
IMPLEMENTATION = "cleverly-ep-assessment"
FAILURE_BOUND = 0.01

STUDY = StudyRecord(
    name="Assessment readings of an EP curve",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "ep_assessment",
    document="docs/technical-reference/method-evidence/ep-assessment.md",
    anchor="assessment-readings-of-an-ep-curve",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/ep_assessment.py",
        "tests/studies/ep_assessment_properties.py",
        "tests/studies/ep_cate.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.ep_assessment",
    properties_module="tests.studies.ep_assessment_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset(),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law.label} law, n={N}, R={group.replicates}, {N_FOLDS} out-of-fold refits, "
        f"{N_SEEDS} split seeds on the first {STABILITY_REPLICATES}"
    )
    for group in GROUPS
}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if group_label == PRIMARY_GROUP:
        return ep_cate.draw_from_seed(group_label, n, seed)
    law = GROUP_BY_LABEL[group_label].law
    return ep_cate.sample(law, n, np.random.default_rng(seed)), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw under this study's seed, as :func:`draw_and_fit` draws it."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The shared primary phase under this study's seed (:func:`ep_cate.primary_phase`)."""
    return ep_cate.primary_phase(__name__, replicates=replicates, n=n, n_jobs=n_jobs)


# ---------------------------------------------------------------- the verdict rules


def mean_interval(values: np.ndarray) -> tuple[float, float]:
    """The two-sided :data:`CONFIDENCE` Student interval of a mean."""
    r = values.size
    half = float(stats.t.ppf(0.5 + CONFIDENCE / 2.0, r - 1)) * float(np.std(values, ddof=1))
    centre = float(np.mean(values))
    return centre - half / math.sqrt(r), centre + half / math.sqrt(r)


def share_lower(successes: int, trials: int) -> float:
    """The one-sided :data:`CONFIDENCE` Clopper-Pearson lower bound of a share."""
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - CONFIDENCE, successes, trials - successes + 1))


# ---------------------------------------------------------------- the design numbers

PILOT_CSV = ROOT / "tests" / "studies" / "ep_assessment_pilot.csv"


def pilot_rows() -> dict[tuple[str, str], dict[str, float]]:
    frame = pd.read_csv(PILOT_CSV)
    return {
        (str(row["group"]), str(row["statistic"])): {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "replicates": int(row["replicates"]),
        }
        for _, row in frame.iterrows()
    }


def calibration_pass(mean: float, sd: float, replicates: int) -> float:
    """P(the 99% interval of the mean slope lies inside the band) at ``R``."""
    spread = sd / math.sqrt(replicates)
    half = float(stats.t.ppf(0.5 + CONFIDENCE / 2.0, replicates - 1)) * spread
    low, high = CALIBRATION_BAND
    return float(
        max(
            0.0,
            stats.norm.cdf((high - half - mean) / spread)
            - stats.norm.cdf((low + half - mean) / spread),
        )
    )


def control_detects(mean: float, sd: float, replicates: int) -> float:
    """P(the stretched slope's 99% upper bound lies below the band) at ``R``."""
    spread = sd / math.sqrt(replicates)
    half = float(stats.t.ppf(0.5 + CONFIDENCE / 2.0, replicates - 1)) * spread
    return float(stats.norm.cdf((CALIBRATION_BAND[0] - half - mean) / spread))


def share_pass(probability: float, replicates: int) -> float:
    """P(the 99% Clopper-Pearson lower bound of a share reaches :data:`SHARE_FLOOR`)."""
    successes = np.arange(replicates + 1)
    lower = np.array([share_lower(int(k), replicates) for k in successes])
    return float(stats.binom.pmf(successes, replicates, probability)[lower >= SHARE_FLOOR].sum())


def design_numbers() -> dict[str, Any]:
    """Every claim's pass probability and the control's power at the declared ``R``.

    The sign claim reads a normal model of the per-replicate statistic: the probability that
    the reported risk difference has the exact sign is ``Phi(mean / sd)`` of the pilot's signed
    z, the reported difference times the exact sign over its across-replicate SD.  The
    stability reading is reported only; ``warning_share`` is the same model's expected share
    of replicates above 0.5 at each law.
    """
    pilot = pilot_rows()

    def at(group: str, statistic: str) -> tuple[float, float]:
        row = pilot[(group, statistic)]
        return row["mean"], row["sd"]

    slope = at("strong_2k", "slope")
    stretched = at("strong_2k", "stretched_slope")
    strong_sd = at("strong_2k", "relative_sd")
    weak_sd = at("weak_2k", "relative_sd")
    signed = at("strong_2k", "signed_risk_difference")
    warns_strong = float(stats.norm.sf((STABILITY_THRESHOLD - strong_sd[0]) / strong_sd[1]))
    warns_weak = float(stats.norm.sf((STABILITY_THRESHOLD - weak_sd[0]) / weak_sd[1]))
    p_sign = float(stats.norm.cdf(signed[0] / signed[1]))
    seconds = {group.label: pilot[(group.label, "seconds")]["mean"] for group in GROUPS}
    calibration_seconds = pilot.get(("strong_2k", "calibration_seconds"), {"mean": 0.0})["mean"]
    cpu = (
        CALIBRATION_REPLICATES * calibration_seconds
        + STABILITY_REPLICATES * (seconds["strong_2k"] - calibration_seconds)
        + STABILITY_REPLICATES * seconds["weak_2k"]
        + PRIMARY_REPLICATES * 0.31
    ) / 3600.0
    return {
        "calibration": calibration_pass(*slope, CALIBRATION_REPLICATES),
        "control": control_detects(*stretched, CALIBRATION_REPLICATES),
        "sign": share_pass(p_sign, CALIBRATION_REPLICATES),
        "probabilities": {"sign": p_sign},
        "warning_share": {"strong_2k": warns_strong, "weak_2k": warns_weak},
        "cost": {"cpu_hours": cpu, "wall_hours_4": cpu / 4.0, "wall_hours_16": cpu / 16.0},
    }
