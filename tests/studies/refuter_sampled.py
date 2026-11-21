"""Registered evidence for sampled refutation: size and power by ``m`` and ``m/n`` (X106 part H).

``refute(result, sample_limit=m)`` draws ``m`` of the fit's ``n`` independent units without
replacement, refits the stored estimator on them, and runs each test on that sampled baseline
as on any ``m``-unit fit. Under i.i.d. units its verdict has exactly the distribution of the
full-data verdict at ``n = m``, whatever ``m/n`` (support decision D3, plans repository
``rows/X106/H/support-decision.md``). This study reads that claim by ``m`` and ``m/n``, i.i.d.
and in whole clusters, beside the full-data rule at ``n = m`` and at moderate ``n``, and its
mutation control shows that it detects the rejected full-data reference. The design is the
comparison plan, section 6 (``rows/X106/H/comparison-plan.md``).

**The law and the rules** are those of ``refuter-calibration-point`` cell T1
(:mod:`tests.studies.refuter_calibration_point`): twenty standard normal covariates, an
unmeasured ``U`` in the treatment and the confounded control only, the Gaussian outcome
``Y = A + 0.5 W1 + 0.4 W2 + 0.05 (W3 + ... + W20) + e``, and the valid and confounded negative
controls ``NC`` and ``NCU``. ``TMLE`` ATE with linear and logistic nuisances, five folds,
``R = 20`` refits, the placebo ``t``, the subset ``rho`` and the negative-control ``z``, each
two-sided at 0.05. A full cell at ``n = 2,000`` is the registered T1 configuration.

**The clustered law** (cells ``C-*``) groups rows in clusters of five. It adds ``b_Y``, drawn
once per cluster from ``N(0, 0.5^2)``, to the outcome and to both controls, and an independent
``b_A`` from the same law to the treatment logit. ``b_A`` and ``b_Y`` are independent, so the
clusters correlate the treatment and the outcome within a cluster and confound nothing: the
ATE stays 1 and the valid control's adjusted contrast stays 0. The fit declares the clusters
(``id=``), so its inference is cluster-robust and its folds keep clusters whole.

**The cells.** Each trial draws ``n`` units, fits once at ``n`` with the trial's fit seed, and
calls ``refute(fit, tests=("placebo", "subset", "negative_control_outcome"),
sample_limit=m)`` with ``random_state=None``, so the root seed is the fit's (review H5). A full
cell (``-F``, ``P-*``) calls it with ``sample_limit=None``.

=====  =============  =========  ========  =====  ======  ==========================================
cell   units          ``m``      ``n``     m/n    trials  role
=====  =============  =========  ========  =====  ======  ==========================================
I2-F   subjects       2,000      2,000     1      1,000   full rule at ``n = m`` (registered T1)
I2-H   subjects       2,000      4,000     0.5    1,000   sampled
I2-T   subjects       2,000      20,000    0.1    1,000   sampled
I1-F   subjects       1,000      1,000     1      1,000   full rule below the registry's ``n_min``
I1-T   subjects       1,000      10,000    0.1    1,000   sampled
C-F    clusters of 5  400        400       1      1,000   full rule, clustered
C-H    clusters of 5  400        800       0.5    1,000   sampled
C-T    clusters of 5  400        4,000     0.1    1,000   sampled
P-4k   subjects       full       4,000     1      50      full rule at moderate ``n`` (power)
P-20k  subjects       full       20,000    1      50      full rule at moderate ``n`` (power)
=====  =============  =========  ========  =====  ======  ==========================================

**Defects.** Trials 0 to 49 of every cell also run each defect at two levels. The strong level
is X82's (stale share 0.05 for the placebo, the reported SE halved and doubled for the subset,
``delta = 0.45`` for the confounded control), sized for per-trial power at least 0.95 at
``m = 2,000``. The weak level (stale share :data:`WEAK_STALE_SHARE`, SE factors
:data:`WEAK_SE_FACTORS`, ``delta =`` :data:`WEAK_CONTROL_DELTA`) is sized for per-trial power
near 0.5 at ``m = 2,000``. ``tests/unit/test_refuter_sampled_design.py`` computes both sizings
from the law. A sampled cell runs each defect through the sampled call. The SE rescaling reads
the trial's own subset ``rho`` and costs no fit.

**Gates** (declared before the run; :mod:`tests.studies.refuter_sampled_properties` computes
them and the design test checks their operating characteristics):

- ``refuter_size`` (I2-F, I2-H, I2-T; 9 readings): at most :data:`SIZE_MAX_REJECTIONS` of 1,000.
- ``refuter_size_relative`` (I2-H, I2-T, I1-T, C-H, C-T against their ``-F`` cell; 15
  readings): the 90% Wald interval of the difference of rejection rates lies inside
  ``±`` :data:`RELATIVE_MARGIN`, the two one-sided tests at 0.05 each.
- ``refuter_power`` (the I2 cells' strong defects; 12 readings): at least
  :data:`POWER_MIN_DETECTIONS` of 50.
- ``refuter_mutation`` (I2-T, I1-T, C-T): the subset ``rho`` recomputed with the full fit's SE
  and unit count, what an implementation that sliced the full fit would read, must fail both the
  absolute size count and the relative test against its ``-F`` cell.

The I1 and clustered absolute sizes, the weak defects, the P cells, the printed-verdict scope
and the sampled baseline beside the full fit are reported in :data:`REPORTED_FILE`, not gated.

**Red reading** (declared before the run). A red absolute size or power cell of a full cell is
a finding on the X82 rule, routed to the refutation owner, not to this row. A red relative or
absolute cell of a sampled cell whose full partner is green is a defect suspect in the sampled
path and is diagnosed first (W1's identity on the failing trials' seeds). A green mutation cell
that should be red means the study cannot see the reference it is built to detect, and the
sampled claim is withheld until the study is repaired under a fresh declaration. No trial count,
law, threshold, defect size or seed moves after a verdict is seen. No rerun buys a pass.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from cleverly.validation import refute
from cleverly.validation._readings import subset_reference_sd
from tests.parallel import STUDY_JOBS
from tests.studies import refuter_calibration_point as point
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SEED = 20261063
RESAMPLING_SEED = 20261064

#: The rules' constants, read from the X82 study so the full cells sit on its configuration.
REFITS = point.REFITS
N_FOLDS = point.N_FOLDS
LEVEL = point.LEVEL
COVARIATES = point.COVARIATES
SUBSET_FRACTION = 0.7

SIZE_TRIALS = 1_000
POWER_TRIALS = 50
#: An absolute size reading passes at no more than this many rejections of :data:`SIZE_TRIALS`.
SIZE_MAX_REJECTIONS = 73
#: A power reading passes at no fewer than this many detections of :data:`POWER_TRIALS`.
POWER_MIN_DETECTIONS = 38
#: The equivalence margin of the relative size test, on the rejection-rate scale.
RELATIVE_MARGIN = 0.05
#: Each one-sided test of the relative size test runs at this level.
RELATIVE_LEVEL = 0.05

#: The strong defects: X82's levels for the Gaussian law.
STRONG_STALE_SHARE = point.STALE_SHARE["gaussian"]
STRONG_SE_FACTORS = (0.5, 2.0)
STRONG_CONTROL_DELTA = point.CONTROL_DELTA["gaussian"]
#: The weak defects, each sized for per-trial power near 0.5 at ``m = 2,000``.
WEAK_STALE_SHARE = 0.025
WEAK_SE_FACTORS = (0.75, 1.43)
WEAK_CONTROL_DELTA = 0.20

#: Members per cluster, and the SD of each cluster effect.
CLUSTER_SIZE = 5
CLUSTER_SD = 0.5

SIZE_FAMILY = "refuter_size"
RELATIVE_FAMILY = "refuter_size_relative"
POWER_FAMILY = "refuter_power"
MUTATION_FAMILY = "refuter_mutation"

#: The size readings every cell reads.
READINGS = ("placebo", "subset", "negative_control")
#: The defects, by level.
STRONG_DEFECTS = ("placebo_stale_rows", "subset_half_se", "subset_double_se", "control_confounded")
WEAK_DEFECTS = tuple(f"{name}_weak" for name in STRONG_DEFECTS)

#: The committed file of reported readings, and the file of fit configurations.
REPORTED_FILE = "reported.csv"
CELLS_FILE = "cells.csv"


@dataclass(frozen=True)
class Cell:
    """One cell of the comparison plan, section 6.

    Parameters
    ----------
    n : int
        Units the trial draws: subjects, or clusters on a clustered cell.
    m : int or None
        The sample limit, in the same units; ``None`` for a full cell.
    clustered : bool
        Whether the units are clusters of :data:`CLUSTER_SIZE`.
    trials : int
        Trials the cell runs.
    partner : str or None
        The full cell at ``n = m`` that a sampled cell is read against.
    """

    n: int
    m: int | None
    clustered: bool
    trials: int
    partner: str | None = None

    @property
    def rows(self) -> int:
        """Rows of one trial's data set."""
        return self.n * (CLUSTER_SIZE if self.clustered else 1)

    @property
    def sampled(self) -> bool:
        """Whether the cell calls the sampled path."""
        return self.m is not None


CELLS: dict[str, Cell] = {
    "I2-F": Cell(2_000, None, False, SIZE_TRIALS),
    "I2-H": Cell(4_000, 2_000, False, SIZE_TRIALS, "I2-F"),
    "I2-T": Cell(20_000, 2_000, False, SIZE_TRIALS, "I2-F"),
    "I1-F": Cell(1_000, None, False, SIZE_TRIALS),
    "I1-T": Cell(10_000, 1_000, False, SIZE_TRIALS, "I1-F"),
    "C-F": Cell(400, None, True, SIZE_TRIALS),
    "C-H": Cell(800, 400, True, SIZE_TRIALS, "C-F"),
    "C-T": Cell(4_000, 400, True, SIZE_TRIALS, "C-F"),
    "P-4k": Cell(4_000, None, False, POWER_TRIALS),
    "P-20k": Cell(20_000, None, False, POWER_TRIALS),
}

#: The cells whose absolute size and strong-defect power are gated: the registered ``m``.
GATED_CELLS = ("I2-F", "I2-H", "I2-T")
#: The sampled cells read against their full partner.
RELATIVE_CELLS = tuple(name for name, cell in CELLS.items() if cell.partner is not None)
#: The sampled cells at ``m/n = 0.1``, where the mutation control is read.
MUTATION_CELLS = ("I2-T", "I1-T", "C-T")

PROPERTY_CELLS = {
    SIZE_FAMILY: tuple(f"{name}__{reading}" for name in GATED_CELLS for reading in READINGS),
    RELATIVE_FAMILY: tuple(f"{name}__{reading}" for name in RELATIVE_CELLS for reading in READINGS),
    POWER_FAMILY: tuple(f"{name}__{defect}" for name in GATED_CELLS for defect in STRONG_DEFECTS),
    MUTATION_FAMILY: tuple(f"{name}__subset_full_reference" for name in MUTATION_CELLS),
}

#: The primary phase: the I2-F fit's ATE against its exact truth, a sanity row on the law that
#: no verdict of this study reads.
SCENARIOS = {"gaussian": ("ate",)}
PRIMARY_REPLICATES = 200
N = 2_000
PRIMARY_N = N

STUDY = StudyRecord(
    name="sampled refutation",
    slug="refuter-sampled",
    artifacts=ROOT / "tests" / "canonical" / "refuter_sampled",
    document="docs/technical-reference/method-evidence/refuter-sampled.md",
    anchor="sampled-refutation",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-refute-sampled",
    reference=None,
    failure_bound={"cleverly-refute-sampled": 0.01},
    modules=(
        "tests/studies/refuter_sampled.py",
        "tests/studies/refuter_sampled_properties.py",
        "tests/studies/refuter_calibration_point.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.refuter_sampled",
    properties_module="tests.studies.refuter_sampled_properties",
    property_cells=PROPERTY_CELLS,
    property_artifacts=(CELLS_FILE, REPORTED_FILE),
    publication_policy="reporting",
)


# ---- the law ------------------------------------------------------------------------------


def draw_law(cell: Cell, seed: int) -> pd.DataFrame:
    """One trial's data: X82's Gaussian law, in clusters of five on a clustered cell.

    The i.i.d. law is :func:`tests.studies.refuter_calibration_point.draw_law` unchanged. A
    clustered draw adds ``b_Y`` to ``Y``, ``NC`` and ``NCU`` and an independent ``b_A`` to the
    treatment logit, both shared by a cluster's members, and records the cluster in ``cl``.
    """
    if not cell.clustered:
        return point.draw_law("gaussian", cell.rows, seed)
    rng = np.random.default_rng(seed)
    n = cell.rows
    cluster = np.repeat(np.arange(cell.n), CLUSTER_SIZE)
    effect_y = rng.normal(scale=CLUSTER_SD, size=cell.n)[cluster]
    effect_a = rng.normal(scale=CLUSTER_SD, size=cell.n)[cluster]
    covariates = rng.standard_normal((n, len(COVARIATES)))
    latent = rng.standard_normal(n)
    eta = point.treatment_index(covariates)
    treatment = rng.binomial(1, expit(eta + point.GAMMA_U * latent + effect_a)).astype(float)
    index = point.outcome_index(covariates)
    outcome = point.GAUSSIAN_EFFECT * treatment + index + effect_y + rng.standard_normal(n)
    control = index + effect_y + rng.standard_normal(n)
    confounded = (
        index + effect_y + point.CONTROL_DELTA["gaussian"] * latent + rng.standard_normal(n)
    )
    low, high = point.GAUSSIAN_Q_BOUNDS
    if outcome.min() <= low or outcome.max() >= high:
        raise RuntimeError("a Gaussian draw left the declared q_bounds")
    frame = pd.DataFrame(covariates, columns=list(COVARIATES))
    frame["A"] = treatment
    frame["Y"] = outcome
    frame["NC"] = control
    frame["NCU"] = confounded
    frame["cl"] = cluster
    return frame


def confounded_control(frame: pd.DataFrame, delta: float) -> np.ndarray:
    """``NCU`` at strength ``delta`` instead of the drawn :data:`STRONG_CONTROL_DELTA`.

    ``NCU - NC`` is ``delta_0 U + e' - e``, so the control at ``delta`` is rebuilt as
    ``NC + (delta / delta_0) (NCU - NC)``. That scales the noise difference too, which keeps
    the control's law Gaussian with the same mean shift ``delta U``; the design test sizes the
    weak level on that law.
    """
    nc = frame["NC"].to_numpy(dtype=float)
    ncu = frame["NCU"].to_numpy(dtype=float)
    return nc + (delta / STRONG_CONTROL_DELTA) * (ncu - nc)


# ---- the fits -----------------------------------------------------------------------------


def fit_cell(cell: Cell, frame: pd.DataFrame, seed: int) -> Any:
    """The T1 fit of one trial's data, with its clusters on a clustered cell."""
    keywords: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": list(COVARIATES)}
    if cell.clustered:
        keywords["id"] = "cl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return point.estimator(point.CELLS["T1"], seed).fit(frame, **keywords).single()


@dataclass(frozen=True)
class Reading:
    """One rule's statistic on one trial, whether it rejects at 0.05, and its printed verdict."""

    statistic: float
    pvalue: float
    rejected: bool
    verdict: bool | None


def _reading(test: Any) -> Reading:
    base = point._reading(test)
    return Reading(base.statistic, base.pvalue, base.rejected, test.passed)


def _rescaled(rho: float, factor: float) -> Reading:
    base = point.rescaled_subset(rho, factor)
    return Reading(base.statistic, base.pvalue, base.rejected, None)


def full_reference_subset(report: Any, full: Any) -> Reading:
    """The subset ``rho`` read against the full fit's SE and unit count (the mutation).

    A sampled call's subset draws ``k`` of the ``m`` sampled units. An implementation that
    sliced the full fit would read the same refits against ``SE_n sqrt((n/k - 1) n/(n - 1))``
    with ``n`` the full unit count.
    """
    sample = report.sample
    values = np.asarray(report["subset"].values, dtype=float)
    spread = float(values.std(ddof=1))
    k = min(sample.sampled_units, max(20, math.floor(SUBSET_FRACTION * sample.sampled_units)))
    reference = subset_reference_sd(float(full["ate"].plugin_std_error), sample.units, k)
    return _rescaled(spread / reference, 1.0)


def _refute(result: Any, cell: Cell, **keywords: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(result, n_replicates=REFITS, sample_limit=cell.m, **keywords)


def trial_seed(stream: str, name: str, trial: int) -> int:
    """The seed of one named stream of one trial.

    The streams are ``"property_sample"`` (the draw), ``"property_fit"`` (the fit, whose seed
    is also the size call's root seed), ``"refute_stale"`` and ``"refute_control"`` (the
    defect calls), and ``"pilot_sample"`` and ``"pilot_fit"`` for the pilot.
    """
    return stream_seed(STUDY, stream, name, trial)


@contextlib.contextmanager
def _stale(share: float) -> Iterator[None]:
    with point.stale_rows(share):
        yield


@dataclass(frozen=True)
class Trial:
    """One trial's readings by name, its reported numbers, and its fit configuration."""

    readings: dict[str, Reading]
    reported: dict[str, Any]
    configuration: dict[str, Any]


def trial_readings(
    name: str, frame: pd.DataFrame, trial: int, *, power: bool, fit_seed: int
) -> Trial:
    """Every reading of one trial: size, the mutation where declared, and the defects."""
    cell = CELLS[name]
    result = fit_cell(cell, frame, fit_seed)
    report = _refute(
        result,
        cell,
        tests=("placebo", "subset", "negative_control_outcome"),
        negative_control_outcome=frame["NC"].to_numpy(dtype=float),
    )
    readings = {
        "placebo": _reading(report["placebo"]),
        "subset": _reading(report["subset"]),
        "negative_control": _reading(report["negative_control_outcome"]),
    }
    baseline = report.sample.baseline_estimate if report.sample else float(result["ate"].psi)
    baseline_se = (
        report.sample.baseline_std_error if report.sample else float(result["ate"].std_error)
    )
    reported: dict[str, Any] = {
        "full_estimate": float(result["ate"].psi),
        "full_std_error": float(result["ate"].std_error),
        "baseline_estimate": float(baseline),
        "baseline_std_error": float(baseline_se),
        "sample_applied": bool(report.sample is not None and report.sample.applied),
    }
    if name in MUTATION_CELLS:
        readings["subset_full_reference"] = full_reference_subset(report, result)
    if power:
        for level, share, factors, delta in (
            ("", STRONG_STALE_SHARE, STRONG_SE_FACTORS, STRONG_CONTROL_DELTA),
            ("_weak", WEAK_STALE_SHARE, WEAK_SE_FACTORS, WEAK_CONTROL_DELTA),
        ):
            with _stale(share):
                stale = _refute(
                    result,
                    cell,
                    tests=("placebo",),
                    random_state=trial_seed("refute_stale" + level, name, trial),
                )
            readings[f"placebo_stale_rows{level}"] = _reading(stale["placebo"])
            rho = readings["subset"].statistic
            readings[f"subset_half_se{level}"] = _rescaled(rho, factors[0])
            readings[f"subset_double_se{level}"] = _rescaled(rho, factors[1])
            control = _refute(
                result,
                cell,
                tests=("negative_control_outcome",),
                negative_control_outcome=confounded_control(frame, delta),
                random_state=trial_seed("refute_control" + level, name, trial),
            )
            readings[f"control_confounded{level}"] = _reading(control["negative_control_outcome"])
    return Trial(readings, reported, point.trial_configuration(result))


# ---- the primary phase --------------------------------------------------------------------


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return point.draw_law("gaussian", n, seed), {"ate": point.truth("gaussian")}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth_map: Mapping[str, float], scenario: str, replicate: int, seed: int
) -> list[dict[str, Any]]:
    """The I2-F fit's ATE as one primary row."""
    result = fit_cell(CELLS["I2-F"], frame, seed)
    estimate = result["ate"]
    low, high = (float(value) for value in estimate.ci)
    reference = float(truth_map["ate"])
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": result.n,
            "estimand": "ate",
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    seed = replicate_seed(STUDY, scenario, index)
    frame, truth_map = drawn
    return cleverly_rows(frame, truth_map, scenario, index, seed)


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
