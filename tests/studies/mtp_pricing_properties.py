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
                            against a ratio of one; 1,000 at n = 2,000.  At ``tau = 3`` the
                            pair reads the same fits as ``double_robustness``
``classifier_split``        ``up_tau3``, wrong GLM, five folds: the shipped unit folds
                            against outer folds drawn over stacked rows
                            (``row_folds_control``, :func:`row_folds`); 1,000 at n = 2,000.
                            Registered only if its pilot reaches power 0.90
==========================  =====================================================================

**Learners, declared before any pilot** (X40 plan section 5.3, M8.1).  The final outcome
regression is :class:`~tests.studies.mtp_pricing_common.FinalOutcome`, correctly specified;
the pseudo-outcome regressions are :class:`~tests.studies.mtp_pricing_common.SplinePseudo`.
The learned density is :class:`~tests.studies.mtp_pricing_common.SplineHazard` at 320 bins.
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

**The default density row** (:func:`default_density_rows`) is a ``reporting`` diagnostic, not
a registered family: ``price_up`` at ``tau = 2`` with the declared hazard at the default bin count
(``density_bins=None``).  Its rows are written beside the pilots and published by the
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
from cleverly.longitudinal import sequential
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_mtp_pricing as study
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
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
#: The arms of the pooled rule: these twelve and the six of ``mtp-tuning``.
POOLED_ARMS = 18
#: Provisional: the expected ``R`` at a pilot mean of 1.005 for 18 arms at 2,000 draws.
CALIBRATION_REPLICATES: dict[str, int] = dict.fromkeys(CELLS, 6_400)
#: The screen's winner per ``tau``.  Provisional until :func:`learner_screen` runs; the
#: declaration commit records the winner and the screen's numbers.
SCREENED_CLASSIFIER: dict[int, str] = dict.fromkeys(common.HORIZONS, "quadratic_logistic")
#: The declared more local alternative of ``classifier_split`` (M8.5).
SPLIT_ALTERNATIVE = "boosted_depth4"
CLASSIFIER_SPLIT_PILOT_DRAWS = 400

#: Each fit set of pairs: ``(tau, families)``.  One fit of the declared and the control arm
#: serves every family listed.
PAIR_SETS: dict[str, tuple[int, tuple[str, ...]]] = {
    "wrong_tau3": (3, ("double_robustness", "ratio_necessity")),
    "wrong_tau5": (5, ("ratio_necessity",)),
    "split_tau3": (3, ("classifier_split",)),
}
#: Each family's cell suffixes at a ``tau``: ``(positive, control)``.
PAIR_ARMS: dict[str, tuple[str, str]] = {
    "double_robustness": ("outcome_wrong_learned_classifier", "both_wrong_control"),
    "ratio_necessity": ("learned_classifier", "unit_ratio_control"),
    "classifier_split": ("unit_folds", "row_folds_control"),
}


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


def ratio_settings(route: str, tau: int, name: str | None = None) -> dict[str, Any]:
    """The ratio route of a cell: the exact ratio, the declared density, or a classifier."""
    if route == "oracle":
        return {"ratio": "classifier", "treatment_learner": common.OracleRatio()}
    if route == "density":
        return {
            "ratio": "density",
            "treatment_learner": common.SplineHazard(n_bins=DENSITY_BINS),
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
            learner, design[train], target[train], weight[train], groups=unit[train],
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
        **ratio_settings(route, tau, classifier_name),
    )


def pair_fits(name: str, frame: pd.DataFrame) -> tuple[Any, Any]:
    """The declared and the control arm of one pair fit set, on one draw."""
    tau, families = PAIR_SETS[name]
    labels = (study.REFERENCE, PLAN)
    if families == ("classifier_split",):
        settings = {**wrong_learners(), **ratio_settings("classifier", tau)}
        declared = study.fit(frame, tau, labels, n_folds=CROSSFIT_FOLDS, **settings)
        with row_folds():
            control = study.fit(frame, tau, labels, n_folds=CROSSFIT_FOLDS, **settings)
        return declared, control
    declared = study.fit(
        frame, tau, labels, **wrong_learners(), **ratio_settings("classifier", tau)
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
    *(("pairs", name, PAIR_N, PAIR_REPLICATES) for name in PAIR_SETS),
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
                seed=_seed("interval_calibration" if family == "interval_calibration" else "pairs", stream, 0),
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
                    PAIR_N,
                    PAIR_REPLICATES,
                    name,
                )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _tau_of(family: str, label: str) -> int:
    return CELLS[label][0] if family == "interval_calibration" else PAIR_SETS[label][0]


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
    tau = _tau_of(family, label)
    frame = common.sample(n, _seed(family, label, replicate) if seed is None else seed, tau)
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
    declared, control = pair_fits(label, frame)
    rows = []
    for each in PAIR_SETS[label][1]:
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


#: A generous peak of one property worker: the 320-bin hazard design at ``tau = 5``.
FIT_BYTES = 2 * 2**30


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication in parts; ``budget`` caps each fit set."""
    jobs = memory_capped_workers(n_jobs, FIT_BYTES)
    root = tools.parts_root(parts)
    frames = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        keys = [(family, label, r, n, requested) for r in range(requested)]
        frames.append(
            tools.run_in_parts(
                fit_set_rows, keys, label=f"{family}__{label}__{requested}", root=root,
                n_jobs=jobs,
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
                efficiency_bounds=None,
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


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("necessity_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    for family in ("ratio_necessity", "classifier_split"):
        labels = tuple(
            f"up_tau{tau}" for tau, families in PAIR_SETS.values() if family in families
        )
        necessity_verdicts(
            summary,
            rows,
            family=family,
            labels=labels,
            arms=PAIR_ARMS[family],
            column="necessity_displacement",
            threshold=NECESSITY_DISPLACEMENT,
        )
    return finish(summary, rates)


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
    """An arm's pilot SE ratio, standardized bias, coverage and seconds a fit."""
    error = rows["estimate"] - rows["truth"]
    spread = float(rows["estimate"].std(ddof=1))
    return {
        "draws": int(len(rows)),
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
                ("interval_calibration", label, r, CALIBRATION_N, draws, name)
                for r in range(draws)
            ]
            rows = tools.run_in_parts(
                _pilot_rows, keys, label=f"screen__{label}__{name}", root=parts, n_jobs=n_jobs
            )
            arms[name] = arm_summary(rows)
        density_label = f"up_learned_density_tau{tau}"
        keys = [("interval_calibration", density_label, r, CALIBRATION_N, draws) for r in range(draws)]
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
                ("interval_calibration", label, r, CALIBRATION_N, draws, name)
                for r in range(draws)
            ]
            part = f"screen__{label}__{name}"
        else:
            keys = [("interval_calibration", label, r, CALIBRATION_N, draws) for r in range(draws)]
            part = f"pilot__{label}"
        rows = tools.run_in_parts(_pilot_rows, keys, label=part, root=parts, n_jobs=n_jobs)
        out["arms"][label] = arm_summary(rows)
    return out


def pooled_replicates(ratios: list[float], draws: int = PILOT_DRAWS) -> dict[str, Any]:
    """The arm-pooled rule over all :data:`POOLED_ARMS` arms of S2, this study's and law Q's."""
    if len(ratios) != POOLED_ARMS:
        raise ValueError(f"the pooled rule reads {POOLED_ARMS} arms, not {len(ratios)}")
    return tools.replicates_from_pilot(ratios, draws)


def classifier_split_pilot(
    parts: Path,
    *,
    draws: int = CLASSIFIER_SPLIT_PILOT_DRAWS,
    n_jobs: int = 1,
    alternative: bool = False,
) -> dict[str, Any]:
    """The M8.5 pilot: paired draws of ``classifier_split`` with the screened classifier.

    ``alternative=True`` reads the declared more local alternative
    (:data:`SPLIT_ALTERNATIVE`), the re-pilot the rule names if power is below 0.90.
    """
    tau = PAIR_SETS["split_tau3"][0]
    if alternative:
        SCREENED_CLASSIFIER[tau] = SPLIT_ALTERNATIVE
    keys = [("pairs", "split_tau3", r, PAIR_N, draws) for r in range(draws)]
    rows = tools.run_in_parts(
        _pilot_rows,
        keys,
        label=f"split__{SCREENED_CLASSIFIER[tau]}",
        root=parts,
        n_jobs=n_jobs,
    )
    positive, control = PAIR_ARMS["classifier_split"]
    declared = rows.loc[rows["cell"] == f"up_tau{tau}__{positive}"].set_index("replicate")
    broken = rows.loc[rows["cell"] == f"up_tau{tau}__{control}"].set_index("replicate")
    sd = float(declared["estimate"].std(ddof=1))
    bias = broken["estimate"] - broken["truth"]
    moved = broken["estimate"] - declared["estimate"]
    delta = float(bias.mean() / sd)
    return {
        "classifier": SCREENED_CLASSIFIER[tau],
        "standardized_bias": delta,
        "standardized_bias_mc_se": float(bias.std(ddof=1) / np.sqrt(len(bias)) / sd),
        "displacement": float(abs(moved.mean()) / sd),
        "displacement_mc_se": float(moved.std(ddof=1) / np.sqrt(len(moved)) / sd),
        "power": tools.necessity_power(delta, PAIR_REPLICATES),
        "seconds": float(rows["seconds"].mean()),
    }


def _default_density_row(replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    started = time.perf_counter()
    tau = 2
    frame = common.sample(n, _seed("reporting", "default_density_tau2", replicate), tau)
    result = study.fit(
        frame,
        tau,
        (study.REFERENCE, PLAN),
        ratio="density",
        treatment_learner=common.SplineHazard(n_bins=DENSITY_BINS),
        **declared_learners(),
    )
    row = replicate_row(
        property_name="reporting",
        cell="default_density_tau2__declared",
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
    """The ``reporting`` row of the default density (``density_bins=None``) at ``tau = 2``."""
    keys = [(r, PAIR_N, replicates) for r in range(replicates)]
    rows = tools.run_in_parts(
        _default_density_row, keys, label="reporting__default_density_tau2", root=parts,
        n_jobs=n_jobs,
    )
    return arm_summary(rows)


#: Per-fit seconds and peak memory from the one-replicate smoke, and the CPU-hours they
#: imply.  Filled by the smoke; the pilots replace the calibration figures.
BUDGET: dict[str, Any] = {}

PILOTS = ("screen", "calibration", "classifier_split", "default_density")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", choices=PILOTS)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--labels", nargs="*", default=())
    parser.add_argument("--taus", nargs="*", type=int, default=())
    parser.add_argument("--alternative", action="store_true")
    arguments = parser.parse_args(argv)
    jobs = arguments.jobs
    if arguments.pilot == "screen":
        result = learner_screen(
            arguments.parts, draws=arguments.draws or SCREEN_DRAWS, n_jobs=jobs,
            taus=tuple(arguments.taus),
        )
    elif arguments.pilot == "calibration":
        result = calibration_pilot(
            arguments.parts, draws=arguments.draws or PILOT_DRAWS, n_jobs=jobs,
            labels=tuple(arguments.labels),
        )
    elif arguments.pilot == "classifier_split":
        result = classifier_split_pilot(
            arguments.parts, draws=arguments.draws or CLASSIFIER_SPLIT_PILOT_DRAWS, n_jobs=jobs,
            alternative=arguments.alternative,
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
