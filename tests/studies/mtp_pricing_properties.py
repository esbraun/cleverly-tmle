"""Property families of ``mtp-pricing``.

Every cell reads the mean purchase under ``price_up``, ``ey_regimen[price_up]`` (the mean
revenue for the revenue cell, on ``q_bounds = (0, 10)``), on law P (:mod:`tests.studies.mtp_pricing_common`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    twelve positives at n = 2,000 with the two derived controls of
                            each.  ``LTMLE`` at one fold with the exact ratio
                            (``up_oracle_tau{2,3,5}``), the declared learned density
                            (``up_learned_density_tau{2,3,5}``) and the screened classifier
                            (``up_learned_classifier_tau{2,3,5}``); the revenue outcome at
                            ``tau = 3`` (``revenue_learned_classifier_tau3``); and ``LTMLE`` and
                            ``SDR`` cross-fitted over five folds at ``tau = 3``
                            (``crossfit_{ltmle,sdr}_learned_classifier_tau3``).  Each positive
                            has its own replications, :data:`CALIBRATION_REPLICATES`
``double_robustness``       ``up_tau3``: the screened classifier with the wrong outcome GLM
                            (positive) against the wrong GLM with a ratio of one
                            (``both_wrong_control``); 1,000 at n = 2,000
``ratio_necessity``         ``up_tau3`` and ``up_tau5``, wrong GLM: the screened classifier
                            against a ratio of one; 1,000 at n = 2,000, on a stream of
                            its own
``classifier_split``        ``up_tau3``, wrong GLM, five folds: the shipped unit folds
                            against outer folds drawn over stacked rows
                            (``row_folds_control``, :func:`row_folds`); 1,000 at n = 2,000.
                            Registered only if its pilot reaches power 0.90
==========================  =====================================================================

**Learners, declared before any pilot** (X40 plan section 5.3, M8.1).  The final outcome
regression is :class:`~tests.studies.mtp_pricing_common.FinalOutcome`, correctly specified;
the pseudo-outcome regressions are :class:`~tests.studies.mtp_pricing_common.SplinePseudo`.
The learned density is :class:`~tests.studies.mtp_pricing_common.SplineHazard` at 320 bins, on
the position relative to each row's declared support.  Every calibration cell reads the
correct outcome model, so a learned-density or classifier cell measures interval calibration
with a learned ratio under a correct outcome.  It is no evidence that the ratio is consistent
(design review m3); the double-robustness pair reads the classifier's ratio alone.
The classifier candidates are
:data:`~tests.studies.mtp_pricing_common.CLASSIFIER_CANDIDATES`.  The wrong outcome GLM of
the pairs is :class:`~tests.studies.mtp_pricing_common.WrongOutcome` at every node.

**The learner screen** (:func:`learner_screen`) fits :data:`SCREEN_DRAWS` draws of each
candidate at each ``tau`` on the ``diagnostic`` stream.  Rule, declared before the screen:
take the candidate with the smallest ``|bias| / SD`` among those whose pilot SE ratio lies in
:data:`~tests.studies.mtp_study_tools.LEARNER_BAND`.  The density is a positive only if its
pilot SE ratio lies in the band and its ``|bias|`` is at most 0.1 SD.  If no candidate
qualifies at some ``tau``, that ``tau``'s learned cells are declared ``reporting`` with an
owner before the run; no oracle is substituted.  The winner's screen draws are its
calibration pilot.  :data:`SCREENED_CLASSIFIER` records the winner per ``tau``; until the
screen runs it holds the provisional first candidate.

**Sizes.**  As ``mtp-boundaries`` (:mod:`tests.studies.mtp_boundaries_properties`), with
:data:`PILOT_DRAWS` draws per arm.  ``R`` is read by the arm-pooled rule over all 18
calibration arms of the X40 S2 plan, the six of ``mtp-tuning`` included
(:func:`pooled_replicates`).  Expected: 6,000 at a pilot mean of 1.000, 6,400 at 1.005,
8,000 at 1.010.

**The declared hazard at the default bins** (:func:`default_density_rows`) is a ``reporting``
diagnostic, not a registered family: ``price_up`` at ``tau = 2`` with the declared hazard at
the default bin count (``density_bins=None``), on the ``diagnostic`` stream.  It is not the
package's default learner (design review m4).  Its rows are written beside the pilots and published by the
method-evidence page as the measurement behind the sentence on the default density.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.learners import density_ratio
from cleverly.learners.crossfit import random_partition
from cleverly.learners.density import default_density_bins
from cleverly.longitudinal import sequential
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_mtp_pricing as study
from tests.studies import canonical_mtp_tuning as tuning_study
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    Necessity,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
CALIBRATION_N = 2_000
PAIR_REPLICATES = 1_000
PAIR_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
NECESSITY_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The declared bin count of the learned density (``oracle_bins`` at n = 2,000).
DENSITY_BINS = 320
CROSSFIT_FOLDS = 5
PLAN = "price_up"

#: Each calibration label: ``(tau, ratio, estimator, folds, revenue)``; ``ratio`` is
#: ``"oracle"``, ``"density"`` or ``"classifier"``.
CELLS: dict[str, tuple[int, str, str, int, bool]] = {
    **{f"up_oracle_tau{t}": (t, "oracle", "ltmle", 1, False) for t in common.HORIZONS},
    **{f"up_learned_density_tau{t}": (t, "density", "ltmle", 1, False) for t in common.HORIZONS},
    **{
        f"up_learned_classifier_tau{t}": (t, "classifier", "ltmle", 1, False)
        for t in common.HORIZONS
    },
    "revenue_learned_classifier_tau3": (3, "classifier", "ltmle", 1, True),
    "crossfit_ltmle_learned_classifier_tau3": (3, "classifier", "ltmle", CROSSFIT_FOLDS, False),
    "crossfit_sdr_learned_classifier_tau3": (3, "classifier", "sdr", CROSSFIT_FOLDS, False),
}
assert tuple(CELLS) == study.CALIBRATION_LABELS

PILOT_DRAWS = 2_000
SCREEN_DRAWS = 2_000
#: Calibration arms the screen declares ``reporting`` before the run.  Such an arm leaves the
#: pooled rule, and it runs at its pilot's :data:`PILOT_DRAWS` replications, which are then its
#: published rows (design review m2, m6).  The screen (2,000 draws per arm, 2026-10-15) read
#: the declared 320-bin density outside :data:`~tests.studies.mtp_study_tools.LEARNER_BAND` at
#: every ``tau``, on the conservative side: SE ratio 1.052 (``tau = 2``), 1.105 (3) and 1.079
#: (5), coverage 0.96 to 0.965.  Owner: the learned-density sentence in
#: ``docs/technical-reference/longitudinal-tmle.md``, which states these readings.
REPORTING_LABELS: frozenset[str] = frozenset(f"up_learned_density_tau{t}" for t in common.HORIZONS)
#: The pooled rule's arms: every gating calibration positive of both S2 records.
POOLED_LABELS: tuple[str, ...] = (
    *(label for label in CELLS if label not in REPORTING_LABELS),
    *tuning_study.CALIBRATION_LABELS,
)
POOLED_ARMS = len(POOLED_LABELS)
#: Each pooled arm's pilot SE ratio (2,000 draws on the ``diagnostic`` stream, 2026-10-15):
#: the screen's parts for the learned classifiers, the local queue's for the rest (plans
#: ``reviews/x40/pilots/s2-pricing-calibration-pilot.json`` and
#: ``s2-tuning-calibration-pilot.json``).
PILOT_SE_RATIOS: dict[str, float] = {
    "crossfit_ltmle_learned_classifier_tau3": 0.984,
    "crossfit_sdr_learned_classifier_tau3": 1.003,
    "revenue_learned_classifier_tau3": 0.9918,
    "up_learned_classifier_tau2": 0.9713,
    "up_learned_classifier_tau3": 0.9965,
    "up_learned_classifier_tau5": 0.9679,
    "up_oracle_tau2": 1.0144,
    "up_oracle_tau3": 1.0072,
    "up_oracle_tau5": 0.9762,
    "fold_average_dose_quantile_tau2": 0.9972,
    "fold_average_quantile_regression_tau2": 0.9912,
    "population_dose_quantile_crossfit_classifier_tau2": 0.9986,
    "population_dose_quantile_crossfit_tau2": 0.9978,
    "population_dose_quantile_tau2": 1.0001,
    "population_quantile_regression_tau2": 1.0044,
}
#: The one ``R`` of both S2 records (design review m2): ``mtp_tuning_properties`` reads it
#: from here.  :func:`pooled_replicates` over :data:`PILOT_SE_RATIOS`: every arm inside the
#: screen, mean 0.9934, tau 0.0108, per-cell 0.997, family-wise 0.956.
S2_REPLICATES = 7_200
CALIBRATION_REPLICATES: dict[str, int] = {
    label: PILOT_DRAWS if label in REPORTING_LABELS else S2_REPLICATES for label in CELLS
}
#: The noise of each ``noise_control`` is ``NOISE_FACTOR`` times the arm's *own* SD: its
#: empirical SD in the calibration pilot times sqrt(n) (:func:`arm_summary`, ``noise_sd``).
#: The control's SE ratio is then ``rho / sqrt(1 + NOISE_FACTOR^2)``, 0.707 at ``rho = 1``,
#: whose upper 99% end at R = 6,400 lies far below the band's 0.93.  An arm scaled to another
#: arm's SD can sit at ``k = 0.45`` of its own and stay inside the band (design review B2).
NOISE_FACTOR = 1.0
#: Each arm's ``noise_sd`` from the calibration pilot (2,000 draws, 2026-10-15): its empirical
#: SD times sqrt(2,000).
NOISE_SD: dict[str, float] = {
    "up_oracle_tau2": 0.712,
    "up_oracle_tau3": 0.88,
    "up_oracle_tau5": 1.341,
    "up_learned_density_tau2": 0.891,
    "up_learned_density_tau3": 1.169,
    "up_learned_density_tau5": 2.162,
    "up_learned_classifier_tau2": 0.745,
    "up_learned_classifier_tau3": 0.886,
    "up_learned_classifier_tau5": 1.335,
    "revenue_learned_classifier_tau3": 7.624,
    "crossfit_ltmle_learned_classifier_tau3": 0.923,
    "crossfit_sdr_learned_classifier_tau3": 0.915,
}
assert set(NOISE_SD) == set(CELLS)
#: The screen's winner per ``tau`` (2,000 draws per candidate, 2026-10-15).  The quadratic
#: logistic has the smaller ``|bias| / SD`` of the candidates inside the band at every ``tau``:
#: SE ratio 0.971, 0.997 and 0.968 at ``tau = 2, 3, 5``.  The depth-2 boosted classifier read
#: 0.929 (outside the band), 0.960 and 0.933 (outside).
SCREENED_CLASSIFIER: dict[int, str] = dict.fromkeys(common.HORIZONS, "quadratic_logistic")
#: The declared more local alternative of ``classifier_split`` (M8.5).
SPLIT_ALTERNATIVE = "boosted_depth4"
#: The ``classifier_split`` pair's own classifier, passed as an argument and never through
#: :data:`SCREENED_CLASSIFIER`, so the pair cannot switch the other ``tau = 3`` cells.  The
#: depth-4 alternative is declared first: a parametric classifier of about 20 coefficients
#: cannot memorize a unit's other block (``s2_pairs.log``: displacement 0.026 SE with the
#: quadratic logistic).  If its 400-draw pilot gives power below 0.90, the pair is not
#: registered, and the method-evidence limitations cell says that the effect of a cross-fold
#: leak is shown in the fast tier only (T-c1, M-c1).
SPLIT_CLASSIFIER = SPLIT_ALTERNATIVE
CLASSIFIER_SPLIT_PILOT_DRAWS = 400
#: The paired draws of the pair pilot (:func:`pair_pilot`), design review M5.
PAIR_PILOT_DRAWS = 200

#: Each fit set of pairs: ``(tau, families)``.  Each family draws its own stream, so
#: ``double_robustness`` and ``ratio_necessity`` at ``tau = 3`` fit the same pair on
#: different samples.
PAIR_SETS: dict[str, tuple[int, tuple[str, ...]]] = {
    "dr_tau3": (3, ("double_robustness",)),
    "wrong_tau3": (3, ("ratio_necessity",)),
    "wrong_tau5": (5, ("ratio_necessity",)),
}
#: The ``classifier_split`` pair, fitted by its pilot only.  Its 400-draw pilot (2026-10-15)
#: read necessity power 0.0002 (control bias -0.22 SD, displacement 0.13), so under the declared
#: exit the pair is not registered: the method-evidence limitations cell says that the effect
#: of a cross-fold leak is shown in the fast tier only (T-c1, M-c1).
SPLIT_PILOT_SET: dict[str, tuple[int, tuple[str, ...]]] = {
    "split_tau3": (3, ("classifier_split",)),
}
_ALL_PAIR_SETS = {**PAIR_SETS, **SPLIT_PILOT_SET}
#: Each registered pair set's sample size.  ``wrong_tau5`` moved to n = 4,000 before the
#: declaration (plan sizing rule 5): its 200-draw pilot at n = 2,000 read the positive's bias at
#: 0.208 own SD (equivalence power 0.11).  On 1,400 fresh draws at n = 2,000 the positive read
#: 0.068 and the oracle-ratio fit 0.044, a finite-sample bias of the wrong outcome under the
#: tau = 5 ratios; pooled with the pilot, 0.086 (upper 95% bound 0.136, power 0.85).  At
#: n = 4,000 the positive reads 0.032 on 800 draws (power 0.98 at the upper bound) and the
#: control 15.3 own SD (``X40-progress-studies.md``).
PAIR_SIZES: dict[str, int] = dict.fromkeys(PAIR_SETS, PAIR_N) | {"wrong_tau5": 4_000}
#: Each family's cell suffixes at a ``tau``: ``(positive, control)``.
PAIR_ARMS: dict[str, tuple[str, str]] = {
    "double_robustness": ("outcome_wrong_learned_classifier", "both_wrong_control"),
    "ratio_necessity": ("learned_classifier", "unit_ratio_control"),
    "classifier_split": ("unit_folds", "row_folds_control"),
}
#: The cells that gate the merge though the record publishes under ``reporting`` (design
#: review M7): each calibration positive outside :data:`REPORTING_LABELS` with its two derived
#: controls (the density cells only once the screen qualifies them), and both arms of the DR
#: pair and of ``ratio_necessity`` at tau = 3 and 5.  ``classifier_split`` is not registered
#: (:data:`SPLIT_PILOT_SET`).  The default-density row reports.
GATING_CELLS: tuple[str, ...] = (
    *(
        f"interval_calibration/{label}__{cell}"
        for label in CELLS
        if label not in REPORTING_LABELS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    *(
        f"{family}/{cell}"
        for family, pairs in study.PAIRED_CELLS.items()
        for pair in pairs
        for cell in pair
    ),
)


def estimand(revenue: bool = False) -> str:
    return f"ey_regimen[{PLAN}]"


def truth(tau: int, revenue: bool = False) -> float:
    return common.truth(tau, f"{'er' if revenue else 'ey'}[{PLAN}]")


# ---------------------------------------------------------------------------- learners


def classifier(name: str) -> Any:
    if name == SPLIT_ALTERNATIVE:
        return common.boosted(4)
    return common.CLASSIFIER_CANDIDATES[name]()


def declared_learners(revenue: bool = False) -> dict[str, Any]:
    return {
        "outcome_learner": common.FinalOutcome(revenue=revenue),
        "pseudo_learner": common.SplinePseudo(),
    }


def wrong_learners() -> dict[str, Any]:
    return {"outcome_learner": common.WrongOutcome(), "pseudo_learner": common.WrongOutcome()}


def ratio_settings(
    route: str, tau: int, name: str | None = None, frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """The ratio route of a cell: the exact ratio, the declared density, or a classifier."""
    if route == "oracle":
        return {"ratio": "classifier", "treatment_learner": common.OracleRatio()}
    if route == "density":
        return {
            "ratio": "density",
            "treatment_learner": common.spline_hazard(frame, tau, DENSITY_BINS),
            "density_bins": DENSITY_BINS,
        }
    if route == "classifier":
        return {
            "ratio": "classifier",
            "treatment_learner": classifier(name or SCREENED_CLASSIFIER[tau]),
        }
    if route == "unit":
        return {"ratio": "classifier", "treatment_learner": DummyClassifier(strategy="prior")}
    raise ValueError(route)


# ---------------------------------------------------------------------------- the leak


def _row_fold_ratio(
    learner: Any,
    covariates: Any,
    observed: Any,
    assigned: Any,
    weights: Any,
    folds: Any,
    *,
    evaluate_at: Any,
    fit_mask: Any = None,
    groups: Any = None,
    n_jobs: int = 1,
    atoms: Any = (),
    assigned_by_fold: Any = None,
) -> list[np.ndarray]:
    """``classifier_ratio`` with its outer folds drawn over the ``2 n`` stacked rows.

    A unit's natural row and its assigned copy then fall in different folds about half the
    time, so the classifier that predicts a unit was trained on that unit's other block: the
    cross-fold copy leak of M-c1 (``tests/unit/test_classifier_ratio_controls.py``).
    """
    if folds.is_single or assigned_by_fold is not None or atoms:
        raise ValueError("the row-fold control reads a plain cross-fitted stack")
    h = np.asarray(covariates, dtype=float).reshape(len(observed), -1)
    n = h.shape[0]
    a = np.asarray(observed, dtype=float).reshape(n, 1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    blocks = [(0.0, 1.0, a)] + [
        (1.0, float(p), np.asarray(d, dtype=float).reshape(n, 1)) for p, d in assigned
    ]
    unit = np.tile(np.arange(n), len(blocks))
    design = np.vstack([np.column_stack([dose, h]) for _, _, dose in blocks])
    target = np.concatenate([np.full(n, label) for label, _, _ in blocks])
    weight = np.concatenate([w * share for _, share, _ in blocks])
    keep = np.tile(mask, len(blocks))
    requests = [np.asarray(dose, dtype=float).reshape(n, 1) for dose in evaluate_at]
    results = [np.zeros(n) for _ in requests]
    for train, test in random_partition(design.shape[0], folds.n_folds, seed=5):
        train = train[keep[train]]
        model = density_ratio.fit_learner(
            learner,
            design[train],
            target[train],
            weight[train],
            groups=unit[train],
            warn_unweighted=False,
        )
        rows = test[test < n]
        for out, dose in zip(results, requests, strict=True):
            u = density_ratio.predict_mean(
                model, np.column_stack([dose[rows], h[rows]]), "classification"
            )
            u = np.minimum(np.clip(u, 0.0, 1.0), density_ratio._ODDS_GUARD)
            out[rows] = u / (1.0 - u)
    return results


@contextmanager
def row_folds() -> Iterator[None]:
    """Route the longitudinal classifier ratio through :func:`_row_fold_ratio`.

    Safe only because each property fit runs alone in its process, serially, as in
    :func:`tests.studies.mtp_boundaries_properties.ratio_defect`.
    """
    if sequential.classifier_ratio is not density_ratio.classifier_ratio:
        raise RuntimeError("a classifier-ratio patch is already active")
    sequential.classifier_ratio = _row_fold_ratio  # type: ignore[assignment]
    try:
        yield
    finally:
        sequential.classifier_ratio = density_ratio.classifier_ratio


# ---------------------------------------------------------------------------- fits


def calibration_fit(label: str, frame: pd.DataFrame, classifier_name: str | None = None) -> Any:
    tau, route, estimator, folds, revenue = CELLS[label]
    return study.fit(
        frame,
        tau,
        (study.REFERENCE, PLAN),
        estimator=estimator,
        n_folds=folds,
        revenue=revenue,
        **declared_learners(revenue),
        **ratio_settings(route, tau, classifier_name, frame),
    )


def pair_fits(
    name: str, frame: pd.DataFrame, classifier_name: str | None = None
) -> tuple[Any, Any]:
    """The declared and the control arm of one pair fit set, on one draw.

    ``classifier_name`` overrides the pair's declared classifier: :data:`SPLIT_CLASSIFIER`
    for ``classifier_split``, the screened classifier of its ``tau`` otherwise.
    """
    tau, families = _ALL_PAIR_SETS[name]
    labels = (study.REFERENCE, PLAN)
    if families == ("classifier_split",):
        settings = {
            **wrong_learners(),
            **ratio_settings("classifier", tau, classifier_name or SPLIT_CLASSIFIER),
        }
        declared = study.fit(frame, tau, labels, n_folds=CROSSFIT_FOLDS, **settings)
        with row_folds():
            control = study.fit(frame, tau, labels, n_folds=CROSSFIT_FOLDS, **settings)
        return declared, control
    declared = study.fit(
        frame, tau, labels, **wrong_learners(), **ratio_settings("classifier", tau, classifier_name)
    )
    control = study.fit(frame, tau, labels, **wrong_learners(), **ratio_settings("unit", tau))
    return declared, control


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """The law a declared cell reads, with the truth of its estimand.

    Parameters
    ----------
    tau : int
        The number of price nodes.
    revenue : bool
        Whether the cell reads the revenue.
    """

    def __init__(self, tau: int, revenue: bool = False) -> None:
        self.tau = tau
        self.revenue = revenue
        self.name = f"P_tau{tau}"

    def truth(self) -> dict[str, float]:
        return {estimand(self.revenue): truth(self.tau, self.revenue)}


#: Every fit set: ``(family, stream label, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    *(
        ("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES[label])
        for label in CELLS
    ),
    *(("pairs", name, PAIR_SIZES[name], PAIR_REPLICATES) for name in PAIR_SETS),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _pilot_seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str, cell: str, role: str, law: DeclaredLaw, n: int, reps: int, stream: str
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=law,
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(
                    "interval_calibration" if family == "interval_calibration" else "pairs",
                    stream,
                    0,
                ),
                role=role,
                estimand=estimand(law.revenue),
            )
        )

    for label, (tau, _, _, _, revenue) in CELLS.items():
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                DeclaredLaw(tau, revenue),
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                label,
            )
    for name, (tau, families) in PAIR_SETS.items():
        for family in families:
            for arm, role in zip(PAIR_ARMS[family], ("positive", "control"), strict=True):
                add(
                    family,
                    f"up_tau{tau}__{arm}",
                    role,
                    DeclaredLaw(tau),
                    PAIR_SIZES[name],
                    PAIR_REPLICATES,
                    name,
                )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _tau_of(family: str, label: str) -> int:
    return CELLS[label][0] if family == "interval_calibration" else _ALL_PAIR_SETS[label][0]


def _fit_rows(
    family: str,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    frame: pd.DataFrame,
    classifier_name: str | None = None,
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, fitted on its drawn ``frame``."""
    tau = _tau_of(family, label)
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}
    if family == "interval_calibration":
        revenue = CELLS[label][4]
        result = calibration_fit(label, frame, classifier_name)
        return [
            replicate_row(
                property_name=family,
                cell=f"{label}__correctly_specified",
                role="positive",
                truth=truth(tau, revenue),
                estimate=result[estimand(revenue)],
                alpha=STUDY.margins.alpha,
                **shared,
            )
        ]
    value = truth(tau)
    name = estimand()
    declared, control = pair_fits(label, frame, classifier_name)
    rows = []
    for each in _ALL_PAIR_SETS[label][1]:
        positive, negative = PAIR_ARMS[each]
        rows.append(
            replicate_row(
                property_name=each,
                cell=f"up_tau{tau}__{positive}",
                role="positive",
                truth=value,
                estimate=declared[name],
                alpha=STUDY.margins.alpha,
                **shared,
            )
        )
        if each == "double_robustness":
            rows.append(
                replicate_row(
                    property_name=each,
                    cell=f"up_tau{tau}__{negative}",
                    role="control",
                    truth=value,
                    estimate=control[name],
                    alpha=STUDY.margins.alpha,
                    **shared,
                )
            )
        else:
            rows.append(
                control_row(
                    property_name=each,
                    cell=f"up_tau{tau}__{negative}",
                    truth=value,
                    estimate=float(control[name].psi),
                    standard_error=float(control[name].std_error),
                    critical=CRITICAL,
                    **shared,
                )
            )
    return rows


def fit_set_rows(
    family: str,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    seed: int | None = None,
    classifier_name: str | None = None,
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, from its declared stream or ``seed``."""
    frame = common.sample(
        n, _seed(family, label, replicate) if seed is None else seed, _tau_of(family, label)
    )
    return _fit_rows(family, label, replicate, n, requested, frame, classifier_name)


#: A property key: ``(family, stream label, replicate, n, requested)``.
_Key = tuple[str, str, int, int, int]

#: Each calibration positive's derived controls, which the keys name and :func:`calibration_controls`
#: writes after the loop.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(tuple(CELLS))


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, label, replicate, n, _ = key
    return common.sample(n, _seed(family, label, replicate), _tau_of(family, label))


def _fit(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _fit_rows(*key, frame)


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    family, label, *_ = key
    if family == "interval_calibration":
        positive = f"{family}/{label}__correctly_specified"
        return [positive, *DERIVED_CELLS[positive]]
    tau, families = PAIR_SETS[label]
    return [f"{each}/up_tau{tau}__{arm}" for each in families for arm in PAIR_ARMS[each]]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def _seed_of(key: _Key) -> int:
    family, label, replicate, *_ = key
    return _seed(family, label, replicate)


#: A generous peak of one property worker: the 320-bin hazard design at ``tau = 5``.
FIT_BYTES = 2 * 2**30


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6).  Up to ``floor(0.01 R)`` lost
#: replicates are judged by the worst case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set.

    The keys of every fit set run in one :func:`~tests.studies.evidence.replication.
    fit_properties` loop.  A finished key is kept under
    :func:`~tests.studies.mtp_study_tools.property_checkpoint` (the driver's checkpoint
    scratch, else ``parts``), so a stopped run resumes.  A ``budget`` run keeps nothing,
    because a key's row carries ``requested``.
    """
    jobs = memory_capped_workers(n_jobs, FIT_BYTES)
    keys: list[_Key] = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        keys += [(family, label, r, n, requested) for r in range(requested)]
    outcomes = fit_properties(
        keys,
        _fit,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=_seed_of,
        checkpoint=tools.property_checkpoint(parts) if budget is None else None,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
                efficiency_bounds=NOISE_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


#: The necessity families' declarations, read by the verdict and by its rule.  Each family's
#: labels are the ``tau`` of the pair sets that fit it.
NECESSITIES: tuple[Necessity, ...] = tuple(
    Necessity(
        family=family,
        labels=tuple(f"up_tau{tau}" for tau, families in PAIR_SETS.values() if family in families),
        arms=PAIR_ARMS[family],
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    for family in ("ratio_necessity",)
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("necessity_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    for necessity in NECESSITIES:
        necessity_verdicts(summary, rows, necessity)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes no
#: efficiency band, because the study's noise control answers to the SE-ratio rule.
#: ``double_robustness`` keeps the shared rule of :func:`apply_shared_verdicts`.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(*NECESSITIES),
)


# ---------------------------------------------------------------------------- pilots


def _pilot_rows(
    family: str,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    classifier_name: str | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    rows = fit_set_rows(
        family,
        label,
        replicate,
        n,
        requested,
        seed=_pilot_seed(family, label, replicate),
        classifier_name=classifier_name,
    )
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds / len(rows)
    return rows


def arm_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """An arm's pilot SE ratio, standardized bias, coverage, noise scale and seconds a fit.

    ``noise_sd`` is the arm's own noise-control scale: ``NOISE_FACTOR`` times its empirical SD
    times sqrt(n).
    """
    error = rows["estimate"] - rows["truth"]
    spread = float(rows["estimate"].std(ddof=1))
    return {
        "draws": len(rows),
        "noise_sd": float(NOISE_FACTOR * spread * np.sqrt(rows["n"].iloc[0])),
        "se_ratio": float(rows["std_error"].mean() / spread),
        "standardized_bias": float(error.mean() / spread),
        "coverage": float(rows["covered"].mean()),
        "seconds": float(rows["seconds"].mean()),
    }


def screen_choice(arms: dict[str, dict[str, Any]]) -> str | None:
    """The declared rule: the smallest ``|bias| / SD`` among the candidates in the band."""
    low, high = tools.LEARNER_BAND
    inside = {k: v for k, v in arms.items() if low <= v["se_ratio"] <= high}
    if not inside:
        return None
    return min(inside, key=lambda k: abs(inside[k]["standardized_bias"]))


def learner_screen(
    parts: Path, *, draws: int = SCREEN_DRAWS, n_jobs: int = 1, taus: tuple[int, ...] = ()
) -> dict[str, Any]:
    """Each classifier candidate and the declared density at each ``tau``: the M8.2 screen.

    The candidates' draws share their seeds within a ``tau``, so the candidates are compared
    on the same samples.  The declared density's draws are its calibration pilot.
    """
    out: dict[str, Any] = {}
    for tau in taus or common.HORIZONS:
        label = f"up_learned_classifier_tau{tau}"
        arms = {}
        for name in common.CLASSIFIER_CANDIDATES:
            keys = [
                ("interval_calibration", label, r, CALIBRATION_N, draws, name) for r in range(draws)
            ]
            rows = tools.run_in_parts(
                _pilot_rows, keys, label=f"screen__{label}__{name}", root=parts, n_jobs=n_jobs
            )
            arms[name] = arm_summary(rows)
        density_label = f"up_learned_density_tau{tau}"
        keys = [
            ("interval_calibration", density_label, r, CALIBRATION_N, draws) for r in range(draws)
        ]
        density = arm_summary(
            tools.run_in_parts(
                _pilot_rows, keys, label=f"pilot__{density_label}", root=parts, n_jobs=n_jobs
            )
        )
        low, high = tools.LEARNER_BAND
        out[f"tau{tau}"] = {
            "classifiers": arms,
            "chosen": screen_choice(arms),
            "density": density,
            "density_qualifies": bool(
                low <= density["se_ratio"] <= high and abs(density["standardized_bias"]) <= 0.1
            ),
        }
    return out


def calibration_pilot(
    parts: Path, *, draws: int = PILOT_DRAWS, n_jobs: int = 1, labels: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The calibration pilot of the arms the screen does not cover, on the ``diagnostic`` stream.

    The screened arms reuse the screen's parts: their labels name the winner, so a call after
    :func:`learner_screen` with the winners in :data:`SCREENED_CLASSIFIER` reads them back.
    """
    out: dict[str, Any] = {"arms": {}}
    for label in labels or tuple(CELLS):
        tau = CELLS[label][0]
        if label == f"up_learned_classifier_tau{tau}":
            name = SCREENED_CLASSIFIER[tau]
            keys = [
                ("interval_calibration", label, r, CALIBRATION_N, draws, name) for r in range(draws)
            ]
            part = f"screen__{label}__{name}"
        else:
            keys = [("interval_calibration", label, r, CALIBRATION_N, draws) for r in range(draws)]
            part = f"pilot__{label}"
        rows = tools.run_in_parts(_pilot_rows, keys, label=part, root=parts, n_jobs=n_jobs)
        out["arms"][label] = arm_summary(rows)
    return out


def pooled_replicates(ratios: dict[str, float], draws: int = PILOT_DRAWS) -> dict[str, Any]:
    """The arm-pooled rule over the :data:`POOLED_LABELS` arms of both S2 records.

    ``ratios`` maps each pooled arm to its pilot SE ratio; any other key set is refused.
    """
    if set(ratios) != set(POOLED_LABELS):
        raise ValueError(
            f"the pooled rule reads the {POOLED_ARMS} arms of POOLED_LABELS, not {sorted(ratios)}"
        )
    return tools.replicates_from_pilot([ratios[label] for label in POOLED_LABELS], draws)


def _pair_pilot_reading(
    name: str, draws: int, parts: Path, n_jobs: int, classifier_name: str | None
) -> dict[str, Any]:
    tau, families = _ALL_PAIR_SETS[name]
    keys = [("pairs", name, r, PAIR_N, draws, classifier_name) for r in range(draws)]
    label = f"pair__{name}__{classifier_name or 'declared'}"
    rows = tools.run_in_parts(_pilot_rows, keys, label=label, root=parts, n_jobs=n_jobs)
    out: dict[str, Any] = {"classifier": classifier_name}
    for family in families:
        positive, control = PAIR_ARMS[family]
        declared = rows.loc[rows["cell"] == f"up_tau{tau}__{positive}"].set_index("replicate")
        broken = rows.loc[rows["cell"] == f"up_tau{tau}__{control}"].set_index("replicate")
        broken = broken.loc[declared.index]
        out[family] = tools.pair_reading(
            declared["estimate"],
            broken["estimate"],
            declared["truth"],
            PAIR_REPLICATES,
            NECESSITY_DISPLACEMENT,
        )
    out["seconds"] = float(rows["seconds"].sum() / draws)
    return out


def pair_pilot(
    parts: Path,
    *,
    draws: int = PAIR_PILOT_DRAWS,
    n_jobs: int = 1,
    names: tuple[str, ...] = ("dr_tau3", "wrong_tau3", "wrong_tau5"),
) -> dict[str, Any]:
    """Both arms of each pair on paired draws, before declaration (design review M5).

    Rule, declared before the pilot: each positive needs ``equivalence_power`` at least 0.90
    and each control ``necessity_power`` at least 0.90 at :data:`PAIR_REPLICATES`, each read
    on the arm's own SD.  A positive that fails is a design or learner finding, diagnosed
    before declaration.  Run it after :func:`learner_screen`, with the winners recorded.
    """
    return {name: _pair_pilot_reading(name, draws, parts, n_jobs, None) for name in names}


def classifier_split_pilot(
    parts: Path,
    *,
    draws: int = CLASSIFIER_SPLIT_PILOT_DRAWS,
    n_jobs: int = 1,
    classifier_name: str = SPLIT_CLASSIFIER,
) -> dict[str, Any]:
    """The M8.5 pilot: paired draws of ``classifier_split`` with the pair's own classifier.

    The classifier is an argument of every fit, so ``--jobs`` workers fit the same one.
    """
    return _pair_pilot_reading("split_tau3", draws, parts, n_jobs, classifier_name)


def _default_density_row(replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    started = time.perf_counter()
    tau = 2
    # The diagnostic stream (design review m4): the row is no registered family's sample.
    seed = stream_seed(STUDY, "diagnostic", "reporting", "declared_hazard_default_bins", replicate)
    frame = common.sample(n, seed, tau)
    result = study.fit(
        frame,
        tau,
        (study.REFERENCE, PLAN),
        ratio="density",
        treatment_learner=common.spline_hazard(frame, tau, default_density_bins(len(frame))),
        **declared_learners(),
    )
    row = replicate_row(
        property_name="reporting",
        cell="declared_hazard_default_bins_tau2__declared",
        role="positive",
        truth=truth(tau),
        estimate=result[estimand()],
        alpha=STUDY.margins.alpha,
        replicate=replicate,
        n=n,
        requested=requested,
    )
    row["seconds"] = time.perf_counter() - started
    return [row]


def default_density_rows(
    parts: Path, *, replicates: int = PAIR_REPLICATES, n_jobs: int = 1
) -> dict[str, Any]:
    """The ``reporting`` row of the declared hazard at the default bins, at ``tau = 2``."""
    keys = [(r, PAIR_N, replicates) for r in range(replicates)]
    rows = tools.run_in_parts(
        _default_density_row,
        keys,
        label="reporting__declared_hazard_default_bins_tau2",
        root=parts,
        n_jobs=n_jobs,
    )
    return arm_summary(rows)


#: Per-fit seconds and peak memory from the one-replicate smoke (warm process, one core,
#: n = 2,000, 2026-10-15), and the CPU-hours they imply.  The pilots replace the calibration
#: figures with their own mean seconds a fit.
_CALIBRATION_SECONDS = {
    "up_oracle_tau2": 0.06,
    "up_oracle_tau3": 0.09,
    "up_oracle_tau5": 0.16,
    "up_learned_density_tau2": 4.3,
    "up_learned_density_tau3": 10.0,
    "up_learned_density_tau5": 20.2,
    "up_learned_classifier_tau2": 0.55,
    "up_learned_classifier_tau3": 1.35,
    "up_learned_classifier_tau5": 2.78,
    "revenue_learned_classifier_tau3": 1.39,
    "crossfit_ltmle_learned_classifier_tau3": 6.2,
    "crossfit_sdr_learned_classifier_tau3": 11.8,
}
#: The two candidates at each ``tau``: the quadratic logistic as the classifier cell, the
#: boosted model 1.3 times it (1.61 s against 2.07 s at ``tau = 3``).
_SCREEN_SECONDS = {
    tau: 2.3 * _CALIBRATION_SECONDS[f"up_learned_classifier_tau{tau}"] for tau in common.HORIZONS
}
_PAIR_SECONDS = {"dr_tau3": 1.52, "wrong_tau3": 1.52, "wrong_tau5": 2.74, "split_tau3": 13.6}
_DEFAULT_DENSITY_SECONDS = 0.6
_PRIMARY_SECONDS = 0.71  # the six scenarios of one replicate
_HOUR = 3600.0
_UNSCREENED = [
    label
    for label in CELLS
    if not label.startswith(("up_learned_classifier", "up_learned_density"))
]


def _calibration_hours(replicates: int) -> float:
    return sum(_CALIBRATION_SECONDS.values()) * replicates / _HOUR


BUDGET: dict[str, Any] = {
    "seconds": {
        "primary_replicate": _PRIMARY_SECONDS,
        "calibration_fit": _CALIBRATION_SECONDS,
        "pair": _PAIR_SECONDS,
        "default_density": _DEFAULT_DENSITY_SECONDS,
    },
    "peak_gib": 0.72,
    "cpu_hours": {
        "learner_screen": SCREEN_DRAWS
        * (
            sum(_SCREEN_SECONDS.values())
            + sum(_CALIBRATION_SECONDS[f"up_learned_density_tau{t}"] for t in common.HORIZONS)
        )
        / _HOUR,
        "calibration_pilot": PILOT_DRAWS
        * sum(_CALIBRATION_SECONDS[label] for label in _UNSCREENED)
        / _HOUR,
        "classifier_split_pilot": CLASSIFIER_SPLIT_PILOT_DRAWS
        * _PAIR_SECONDS["split_tau3"]
        / _HOUR,
        "calibration": {r: _calibration_hours(r) for r in (6_000, 6_400, 8_000)},
        "pairs": PAIR_REPLICATES * sum(_PAIR_SECONDS.values()) / _HOUR,
        "default_density": PAIR_REPLICATES * _DEFAULT_DENSITY_SECONDS / _HOUR,
        "primary": study.PRIMARY_REPLICATES * _PRIMARY_SECONDS / _HOUR,
    },
}
BUDGET["cpu_hours"]["python_total"] = sum(
    value if not isinstance(value, dict) else value[6_400] for value in BUDGET["cpu_hours"].values()
)

PILOTS = ("screen", "calibration", "pairs", "classifier_split", "default_density")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", choices=PILOTS)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--labels", nargs="*", default=())
    parser.add_argument("--taus", nargs="*", type=int, default=())
    parser.add_argument("--classifier", default=None)
    arguments = parser.parse_args(argv)
    jobs = arguments.jobs
    if arguments.pilot == "screen":
        result = learner_screen(
            arguments.parts,
            draws=arguments.draws or SCREEN_DRAWS,
            n_jobs=jobs,
            taus=tuple(arguments.taus),
        )
    elif arguments.pilot == "calibration":
        result = calibration_pilot(
            arguments.parts,
            draws=arguments.draws or PILOT_DRAWS,
            n_jobs=jobs,
            labels=tuple(arguments.labels),
        )
    elif arguments.pilot == "pairs":
        result = pair_pilot(arguments.parts, draws=arguments.draws or PAIR_PILOT_DRAWS, n_jobs=jobs)
    elif arguments.pilot == "classifier_split":
        result = classifier_split_pilot(
            arguments.parts,
            draws=arguments.draws or CLASSIFIER_SPLIT_PILOT_DRAWS,
            n_jobs=jobs,
            classifier_name=arguments.classifier or SPLIT_CLASSIFIER,
        )
    else:
        result = default_density_rows(
            arguments.parts, replicates=arguments.draws or PAIR_REPLICATES, n_jobs=jobs
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    (arguments.parts / f"{arguments.pilot}-pilot.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
