"""Declared evidence for the corrected policy-ratio routes of a point dose (``mtp-density-tails``).

The subject is the point TMLE of a modified treatment policy on a continuous dose, through both
of its ratio routes: the binned density route, whose grid and reading the study's fix rebuilt
(per-row segments read through the bin masses, a declared support read row by row, a refined
grid with a relative gap test), and the classifier route that ``ratio=None`` resolves to, with
the policy's step columns.  The longitudinal default route is read on ``longitudinal-mtp``'s
continuous law.

**The laws** are those of :mod:`tests.mtp_density_tails_law`: G (a Gaussian dose and an
unbounded ratio), T (G truncated to ``[-2, 4.5]`` with the shift capped there), C (a 0.15% far
component beyond a gap), S (Student-t tails), and B and U (a covariate-dependent upper end of
the support, declared as ``Support(0, u(W))``, under the policy that adds one below ``u - 1`` and
holds the dose above it).  Each law's truth ``psi0`` and efficiency bound ``SD(D0)`` are Monte
Carlo values over 4,000,000 draws, computed from the closed-form density and outcome regression
and never from the estimator.  The fits and the study-local controls are in
:mod:`tests.studies.mtp_density_tails_fits`.

**The primary rows** fit the density route with the oracle hazard (each row's exact bin masses,
so the grid and the reading are what is left) and the correct outcome regression, in sample, at
n = 2,000 on all six laws, and are reported.  The gated claims are in the property families of
:mod:`tests.studies.mtp_density_tails_properties`.

**No comparator.**  R ``lmtp`` fits the classifier ratio with its own learners and has no binned
density route; ``txshift`` fits a parametric density.  Neither reads the representation this
study tests.  The independent evidence is the closed-form laws, the exact influence-curve gap,
and the unit witnesses of ``tests/unit/test_density_reconstruction.py`` and
``tests/unit/test_classifier_policy_steps.py``.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/mtp_density_tails/regenerate.py`` before any run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tests import mtp_density_tails_law as law
from tests.parallel import STUDY_JOBS
from tests.studies import mtp_density_tails_fits as fits
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261163
RESAMPLING_SEED = 2026116301

SCENARIOS = law.LAWS
#: Each law's one estimand: the mean outcome under its policy.
ESTIMAND = "policy_mean"
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, (ESTIMAND,))
TRUTH: dict[str, dict[str, float]] = {
    name: {ESTIMAND: float(law.TRUTH[name][0])} for name in SCENARIOS
}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


#: Laws whose density-route calibration is gated; the others are reported.
DENSITY_GATED = ("G", "T", "C")
DENSITY_REPORTED = ("S", "B", "U")
#: Laws whose density-route double robustness is gated (n = 8,000).
ROBUST_GATED = ("G", "T")
#: Laws whose classifier-route calibration is gated.
CLASSIFIER_GATED = ("G", "T", "S")
#: The longitudinal default-route estimators.
LONGITUDINAL = ("LTMLE", "SDR")
#: Laws whose influence-curve gap is gated; S and ``Gscale`` are reported.
GAP_GATED = ("G", "T", "C", "B", "U")
GAP_REPORTED = ("S", "Gscale")
GAP_SIZES = (2_000, 20_000)


def _longitudinal_label(estimator: str, estimand: str) -> str:
    """A cell label of one longitudinal estimand: ``LTMLE_ate_up_vs_natural`` and the like."""
    inner = estimand.split("[", 1)[1].rstrip("]").replace(" vs ", "_vs_").replace(" ", "_")
    kind = estimand.split("[", 1)[0].replace("_regimen", "")
    return f"{estimator}_{kind}_{inner}"


LONGITUDINAL_LABELS: dict[str, dict[str, str]] = {
    estimator: {
        estimand: _longitudinal_label(estimator, estimand)
        for estimand in fits.LONGITUDINAL_ESTIMANDS
    }
    for estimator in LONGITUDINAL
}
LONGITUDINAL_GATED_LABELS = {
    estimator: labels[fits.LONGITUDINAL_GATED] for estimator, labels in LONGITUDINAL_LABELS.items()
}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        *_cells("G", "correctly_specified", "shrunken_se_control", "noise_control"),
        *(cell for name in DENSITY_GATED[1:] for cell in _cells(name, "correctly_specified")),
        *(cell for name in DENSITY_REPORTED for cell in _cells(name, "reported")),
        *_cells("G_dr", "wrong_outcome", "both_wrong_control", "pre_fix_control"),
        *_cells("T_dr", "wrong_outcome", "both_wrong_control"),
        *_cells("G_classifier", "correctly_specified", "shrunken_se_control", "noise_control"),
        *(
            cell
            for name in CLASSIFIER_GATED[1:]
            for cell in _cells(f"{name}_classifier", "correctly_specified")
        ),
        *_cells("G_classifier_dr", "reported"),
        *_cells("T_classifier_dr", "reported"),
        *_cells("G_classifier_dr_both_wrong", "reported"),
        *_cells(
            LONGITUDINAL_GATED_LABELS["LTMLE"],
            "correctly_specified",
            "shrunken_se_control",
            "noise_control",
        ),
        *_cells(LONGITUDINAL_GATED_LABELS["SDR"], "correctly_specified"),
        *(
            cell
            for estimator in LONGITUDINAL
            for estimand, label in LONGITUDINAL_LABELS[estimator].items()
            if estimand != fits.LONGITUDINAL_GATED
            for cell in _cells(label, "reported")
        ),
    ),
    "ic_gap": (
        *(
            cell
            for name in (*GAP_GATED, *GAP_REPORTED)
            for size in GAP_SIZES
            for cell in _cells(name, f"n_{size}")
        ),
        *(
            cell
            for name in GAP_GATED
            for size in GAP_SIZES
            for cell in _cells(name, f"pre_fix_n_{size}")
        ),
        *(
            cell
            for name in law.SUPPORTED
            for size in GAP_SIZES
            for cell in _cells(name, f"support_ignored_n_{size}")
        ),
        *(cell for name in law.SUPPORTED for cell in _cells(name, "no_row_break_n_2000")),
    ),
    "cluster_warning": (
        *_cells("C", "n_20000", "gap_disabled_control"),
        *_cells("GT", "pooled", "eager_control"),
    ),
}

STUDY = StudyRecord(
    name="modified treatment policy on a point dose, corrected density and classifier ratios",
    slug="mtp-density-tails",
    artifacts=ROOT / "tests" / "canonical" / "mtp_density_tails",
    document="docs/technical-reference/method-evidence/mtp-density-tails.md",
    anchor="modified-treatment-policy-density-tails",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_mtp_density_tails.py",
        "tests/studies/mtp_density_tails_properties.py",
        "tests/studies/mtp_density_tails_fits.py",
        "tests/mtp_density_tails_law.py",
        "tests/studies/canonical_longitudinal_mtp.py",
        "tests/studies/longitudinal_mtp_common.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/mtp_study_tools.py",
        "tests/studies/oracle_density_bins.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/learners/density.py",
        "src/cleverly/learners/density_ratio.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mtp_density_tails",
    properties_module="tests.studies.mtp_density_tails_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "point TMLE of a modified treatment policy on a continuous dose",
    "primary_route": "density, oracle hazard on the default grid, correct outcome regression, "
    "in sample",
    "density_bins": "default_density_bins(n)",
    "classifier_route": "ratio=None (resolved to the classifier), default stack, "
    f"{fits.POINT_FOLDS} outer folds, policy steps, q_bounds={list(fits.Q_BOUNDS)}",
    "longitudinal_route": "ratio=None at both continuous nodes, default stack, "
    f"{fits.LONGITUDINAL_FOLDS} outer folds",
    "tol": fits.TOL,
    "max_iter": fits.MAX_ITER,
    "laws": list(SCENARIOS),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law.draw(scenario, n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def fit(scenario: str, frame: pd.DataFrame) -> fits.PointFit:
    """The primary fit: the density route with the oracle hazard and the correct regression."""
    return fits.fit_density(scenario, frame)


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, []


def subject_rows(
    fitted: fits.PointFit, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's primary row, on the identity scale."""
    low, high = fitted.ci
    reference = float(truth[ESTIMAND])
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": ESTIMAND,
            "truth": reference,
            "estimate": fitted.estimate,
            "inference_estimate": fitted.estimate,
            "std_error": fitted.std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": fitted.initial,
        }
    ]


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    fitted = fit(scenario, frame)
    return sample, truth_rows, subject_rows(fitted, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of every law, its truth rows and the subject's rows."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    frames = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=frames
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by law."""
    return TRUTH


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]
