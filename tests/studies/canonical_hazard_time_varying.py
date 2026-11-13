"""Declared evidence for hazard targeting with covariates after baseline (``hazard-time-varying``).

The subject is ``LTMLE(survival_model="hazard", targeting="one_step")`` on a design with one
baseline treatment held over the nodes and a covariate :math:`L_k` measured at every node after
the first (X70 part (b1)).  One backward pass per regimen targets, at each node, the node's event
hazards and the continuation, a soft-label hazard fit of the later events with the later
covariates integrated out, as one node panel; each node's walk runs after every later node is
final (``docs/technical-reference/longitudinal-tmle.md``, "Covariates after baseline on the
hazard path").

**The laws** are those of :mod:`tests.studies.hazard_time_varying_law`: ``one_cause`` (binary
treatment, one cause, five nodes), ``competing`` (two causes, four nodes) and ``three_arm`` (a
three-level treatment, five nodes, the modified treatment policy "one level up" at node 1).
Plan X70 section 5 names the structure of ``make_point_survival`` for law (i) and expects its
fallback.  Its drop-out is linear on the probability scale and does not read :math:`L_k`, so the
both-wrong cell could not show a bias.  The study's laws keep that structure (logistic hazard in
:math:`(k, A, W, L_k)`, binary :math:`L_k` from :math:`(W_1, A, L_{k-1})`) and draw the drop-out
logistic in :math:`(A, W, L_k)`, with an :math:`L_k` coefficient large enough that the plug-in
whose regressions omit :math:`L` misses the ``one_cause`` contrast by 4.2 efficiency standard
errors at n = 2,000 (``tests/unit/test_hazard_time_varying_study_design.py``).

**Fit labels** (:data:`FITS`): ``one_cause``, ``competing`` and ``three_arm`` in sample, and
``crossfit``, the ``one_cause`` law at five outer folds.  Plan X70 section 5 lists no
cross-fitted law for (b1).  The cross-fitted fit label is added so that the declared support row
``G63.held.cross_fitted`` has a cell; it is a calibration cell only.

**Truths and bounds** are exact sums over the finite support (:math:`W`, the treatment, the
covariate paths and the outcomes), computed by code that never calls the estimator.  The design
test checks each by an independent route.

**No comparator.**  R ``ltmle`` and ``lmtp`` compute the same estimand on the same data
structure by the iterated-mean algorithm, and ``canonical-ltmle-survival`` already ties the
in-package iterated-mean fit to ``ltmle``; the ``iterated_mean_difference`` cells compare the two
fits on the same draws.  No software for the hazard algorithm with covariates after baseline is
known (plan X70, open question 8).  The record names no reference.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/hazard_time_varying/regenerate.py`` before any run.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly._support_rows import long_row
from cleverly.interventions import ModifiedPolicy, Piece
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.hazard_backward import untargeted
from tests.parallel import STUDY_JOBS
from tests.studies import hazard_time_varying_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: Provisional until the calibration pilots of the primary fit labels are committed; the shared
#: primary rule (``repeated_sizing.primary_replicates``, read by the design test) then sets it.
PRIMARY_REPLICATES = 2_400
PRIMARY_N = 2_000
SEED = 20261901
RESAMPLING_SEED = 2026190101

#: The targeting settings of every hazard fit of this study.
ALGORITHM = "one_step"
TOL = 1e-10

#: Fit label -> ``(law, folds)``.
FITS: dict[str, tuple[str, int]] = {
    "one_cause": ("one_cause", 1),
    "competing": ("competing", 1),
    "three_arm": ("three_arm", 1),
    "crossfit": ("one_cause", 5),
}
#: The primary scenarios: every fit label but the cross-fitted variant, which calibration reads.
SCENARIOS = ("one_cause", "competing", "three_arm")
#: The cause every single-estimate cell reads.
FIRST_CAUSE = 1

_INF = np.inf


def labels(law_name: str) -> tuple[str, str]:
    """The regimen and the reference of one law."""
    return ("up", "natural") if law.LAWS[law_name].arms == 3 else ("always", "never")


def regimens(law_name: str) -> dict[str, Any]:
    """The regimens of one law: always and never, or one level up and the natural course."""
    if law.LAWS[law_name].arms == 3:
        return {
            "up": ModifiedPolicy("up", apply=_up, policy_kind="known"),
            "natural": ModifiedPolicy("natural", pieces=(Piece(-_INF, _INF),), policy_kind="known"),
        }
    return {"always": 1, "never": 0}


def _up(a: Any, h: Any) -> Any:
    """One level up, the top stays."""
    del h
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


def maps(law_name: str) -> dict[str, law.Regimen]:
    """Each regimen of one law as a map of the natural treatment, for the exact truths."""
    if law.LAWS[law_name].arms == 3:
        return {"up": law.up, "natural": law.natural}
    return {"always": law.static(1), "never": law.static(0)}


def _cause(law_name: str, cause: int) -> str:
    return "" if law.LAWS[law_name].causes == 1 else f", {law.CAUSES[cause]}"


def risk_name(law_name: str, label: str, horizon: int, cause: int = FIRST_CAUSE) -> str:
    """The reported name of a regimen's risk, or cumulative incidence, by ``horizon``."""
    if law.LAWS[law_name].causes == 1:
        return f"risk_regimen[{label} @ t={horizon}]"
    return f"cif_regimen[{label}{_cause(law_name, cause)} @ t={horizon}]"


def contrast_name(law_name: str, horizon: int, cause: int = FIRST_CAUSE) -> str:
    """The reported name of the contrast of a law's regimen against its reference."""
    first, second = labels(law_name)
    return f"ate_regimen[{first} vs {second}{_cause(law_name, cause)} @ t={horizon}]"


def estimand_names(law_name: str) -> tuple[str, ...]:
    """Every estimate a fit of one law reports, in the fit's order."""
    current = law.LAWS[law_name]
    causes = range(1, current.causes + 1)
    times = range(1, current.n_times + 1)
    return (
        *(
            risk_name(law_name, label, t, j)
            for label in labels(law_name)
            for j in causes
            for t in times
        ),
        *(contrast_name(law_name, t, j) for j in causes for t in times),
    )


def target_of(law_name: str, name: str) -> law.Target:
    """The linear target of one reported name."""
    current = law.LAWS[law_name]
    mapped = maps(law_name)
    for j in range(1, current.causes + 1):
        for t in range(1, current.n_times + 1):
            if name == contrast_name(law_name, t, j):
                first, second = labels(law_name)
                return law.Target(((1.0, mapped[first], t, j), (-1.0, mapped[second], t, j)))
            for label in labels(law_name):
                if name == risk_name(law_name, label, t, j):
                    return law.Target(((1.0, mapped[label], t, j),))
    raise KeyError(name)


def truths(law_name: str) -> dict[str, float]:
    """The exact value of every estimate of one law, by reported name."""
    current = law.LAWS[law_name]
    return {name: target_of(law_name, name).value(current) for name in estimand_names(law_name)}


def law_of(fit_label: str) -> law.Law:
    return law.LAWS[FITS[fit_label][0]]


ESTIMANDS = {scenario: estimand_names(FITS[scenario][0]) for scenario in SCENARIOS}
TRUTH = {scenario: truths(FITS[scenario][0]) for scenario in SCENARIOS}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


#: The calibration fit labels and their two arms: the contrast and the regimen's risk (of the
#: first cause) at the last node.
CALIBRATION_FITS = tuple(FITS)
CALIBRATION_LABELS = tuple(
    f"{fit}_{arm}" for fit in CALIBRATION_FITS for arm in ("contrast", "risk")
)
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = ("one_cause_contrast",)
#: The contrast every robustness, necessity and power cell reads.
CONTRAST_LABEL = "one_cause_contrast"
#: The joint labels: the band over every estimate of one fit.
JOINT_LABELS = ("one_cause_curve", "competing_curve")
RATE_SIZES = (1_000, 2_000, 8_000)
LADDER_LABELS = ("one_cause_contrast", "one_cause_risk")
#: The fits whose draws carry the reported comparison with the iterated-mean fit.
DIFFERENCE_FITS = ("one_cause", "competing")
DIFFERENCE_ARMS = ("paired_difference", "hazard_fit", "iterated_fit", "isotonic_difference")
#: The per-fit coherence reading of the two-cause law: the largest sum of the cumulative
#: incidences over the causes, minus one, over the horizons and both regimens.
COHERENCE_FITS = ("competing",)
COHERENCE_ARMS = ("hazard_excess", "iterated_excess", "isotonic_excess")

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        cell
        for label in CALIBRATION_LABELS
        for cell in (
            _cells(label, "correctly_specified", "shrunken_se_control", "noise_control")
            if label in CONTROLLED_LABELS
            else _cells(label, "correctly_specified")
        )
    ),
    "double_robustness": _cells(
        CONTRAST_LABEL, "both_correct", "outcome_correct", "mechanism_correct", "both_wrong"
    ),
    "targeting_necessity": _cells(CONTRAST_LABEL, "targeted", "untargeted"),
    "root_n_and_efficiency": tuple(
        cell
        for label in LADDER_LABELS
        for cell in _cells(label, *(f"n_{size}" for size in RATE_SIZES))
    ),
    "root_n_rate": tuple(
        cell for label in LADDER_LABELS for cell in _cells(label, "empirical_sd", "reported_se")
    ),
    "power": _cells(CONTRAST_LABEL, "alternative"),
    "simultaneous_coverage": tuple(
        cell
        for label in JOINT_LABELS
        for cell in _cells(label, "simultaneous_band", "pointwise_joint_control")
    ),
    "iterated_mean_difference": (
        *(
            cell
            for fit in DIFFERENCE_FITS
            for arm in ("contrast", "risk")
            for cell in _cells(f"{fit}_{arm}", *DIFFERENCE_ARMS)
        ),
        *(f"{fit}__iterated_curve_decrease" for fit in DIFFERENCE_FITS),
        *(cell for fit in COHERENCE_FITS for cell in _cells(fit, *COHERENCE_ARMS)),
        *_cells(CONTRAST_LABEL, "mechanism_correct_difference"),
        *_cells("one_cause_risk", "mechanism_correct_difference"),
    ),
}

STUDY = StudyRecord(
    name="covariates after baseline on the hazard path, one-step targeting",
    slug="hazard-time-varying",
    artifacts=ROOT / "tests" / "canonical" / "hazard_time_varying",
    document="docs/technical-reference/method-evidence/hazard-time-varying.md",
    anchor="hazard-time-varying",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_hazard_time_varying.py",
        "tests/studies/hazard_time_varying_properties.py",
        "tests/studies/hazard_time_varying_law.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/longitudinal/hazard_backward.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_hazard_time_varying",
    properties_module="tests.studies.hazard_time_varying_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "held baseline treatment, a binary covariate at every node after the first, "
    "one backward pass per regimen; at each node one one-step walk over the node hazards and the "
    "continuation, after every later node is final",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "tol": TOL,
    "fits": {label: list(spec) for label, spec in FITS.items()},
    "outcome_learner": "unpenalized logistic regression on W1, W2, L_k and the treatment at each "
    "node (the earlier L columns dropped)",
    "pseudo_learner": "the weighted cell mean over W1, W2, L_k, the treatment and the later-node "
    "indicators",
    "treatment_learner": "unpenalized logistic (multinomial on three levels) on W1, W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, L_k and the treatment at each "
    "node",
    "regimens": {
        "always / never": "the treatment set to 1 or 0 (one_cause, competing, crossfit)",
        "up / natural": "one level up, the top stays, against the identity (three_arm)",
    },
}

#: The support rows this study will license once it passes and is registered (plan X70 3.4:
#: ``G63.held.*``, the ``G50.in_sample`` pattern).  Until then every fit with covariates after
#: baseline on the hazard path is labelled ``experimental`` by the X70 condition of
#: :mod:`cleverly.longitudinal.hazard_backward`, and the rows stay out of
#: :data:`cleverly._support_rows.LONGITUDINAL_ROWS`, whose evidence must be registered.  The
#: support key has no axis for covariates after baseline, so a held hazard fit without them on
#: the same design has the same key (``tests/unit/test_hazard_time_varying_study_design.py``).
#: Registration therefore adds that axis to ``longitudinal_key`` with these rows.
G63_CELLS: dict[str, tuple[str, ...]] = {
    "G63.held.in_sample": ("one_cause_contrast", "one_cause_risk"),
    "G63.held.competing": ("competing_contrast", "competing_risk"),
    "G63.held.mtp": ("three_arm_contrast", "three_arm_risk"),
    "G63.held.cross_fitted": ("crossfit_contrast", "crossfit_risk"),
}
#: Row id -> the fit label whose fit the row's key is read from.
G63_FITS = {
    "G63.held.in_sample": "one_cause",
    "G63.held.competing": "competing",
    "G63.held.mtp": "three_arm",
    "G63.held.cross_fitted": "crossfit",
}


def declared_support_rows() -> tuple[Any, ...]:
    """The ``G63.held.*`` rows as :func:`~cleverly._support_rows.long_row` would build them.

    Each row's key is that of its fit label's fit; the design test checks it against a fit.
    """
    common: dict[str, Any] = {
        "held": True,
        "survival_model": "hazard",
        "targeting": ALGORITHM,
        "cells": (),
    }
    rows = []
    for row_id, fit_label in G63_FITS.items():
        current = law_of(fit_label)
        _, folds = FITS[fit_label]
        rows.append(
            long_row(
                row_id,
                [STUDY.slug],
                outcome_kind="survival" if current.causes == 1 else "competing",
                regimens=("mtp",) if current.arms == 3 else ("static",),
                treatment=(str(current.arms),) * current.n_times,
                splitting=("cross_fitted",) if folds > 1 else ("in_sample",),
                **{**common, "cells": G63_CELLS[row_id]},
            )
        )
    return tuple(rows)


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The learners of one nuisance configuration.

    The wrong outcome side is the hazard and the continuation without ``L``; the wrong mechanism
    is the censoring model without ``L_k``.  The treatment is baseline and reads no ``L``, so its
    model is correct in every configuration.
    """
    outcome = configuration in {"both_correct", "outcome_correct"}
    mechanism = configuration in {"both_correct", "mechanism_correct"}
    return {
        "outcome_learner": law.hazard_learner(outcome),
        "pseudo_learner": law.continuation_learner(outcome),
        "treatment_learner": law.Logistic(),
        "censoring_learner": law.censoring_learner(mechanism),
    }


@dataclass(frozen=True)
class FitRecord:
    """What a fit's warnings and walks say about it, recorded beside its rows.

    Parameters
    ----------
    converged : bool
        Whether every hazard walk of the fit met its tolerance.
    unexpected_warnings : tuple of str
        The class names of the warnings outside :data:`EXPECTED_WARNINGS`, one entry per
        warning.
    final_score : float
        The largest final relative score over the fit's hazard walks,
        ``max |score| / score_scale``, at the reported hazards.
    """

    converged: bool
    unexpected_warnings: tuple[str, ...]
    final_score: float = 0.0


def final_score(walk: Any) -> float:
    """``max |score| / score_scale`` of one hazard walk at its reported hazards."""
    score = np.abs(np.asarray(walk.score, dtype=float)).reshape(-1)
    if score.size == 0:
        return 0.0
    scale = np.asarray(walk.score_scale, dtype=float).reshape(-1)
    return float(np.max(score / np.maximum(scale, 1e-300)))


#: The warning a fit is expected to raise, with why.  The risks at the first node are near 0.02,
#: so their influence curves read few events and ``SparseInfluenceWarning`` names them.  The
#: warning is unexpected, and counted, when it names a gated estimate: a contrast or a risk at
#: the last node (:func:`unexpected`).
EXPECTED_WARNINGS = {
    "SparseInfluenceWarning": "the first nodes' risks are near 0.02, so their influence curves "
    "read few events; the gated estimates at the last node are not expected in it",
}


def gated_names() -> tuple[str, ...]:
    """The reported names a calibration cell reads: the contrast and risk at each last node."""
    names: list[str] = []
    for law_name in {spec[0] for spec in FITS.values()}:
        last = law.LAWS[law_name].n_times
        names += [contrast_name(law_name, last), risk_name(law_name, labels(law_name)[0], last)]
    return tuple(sorted(names))


def unexpected(caught: Sequence[Any]) -> tuple[str, ...]:
    """The class names of the caught warnings outside :data:`EXPECTED_WARNINGS`.

    A ``SparseInfluenceWarning`` that names a gated estimate counts as unexpected.
    """
    gated = gated_names()
    out = []
    for item in caught:
        name = item.category.__name__
        if name in EXPECTED_WARNINGS and not any(
            f"{label} (" in str(item.message) for label in gated
        ):
            continue
        out.append(name)
    return tuple(out)


def recorded(call: Any) -> tuple[Any, FitRecord]:
    """Run ``call()`` and record its unexpected warnings and the convergence of its walks."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    unexpected_names = unexpected(caught)
    found = (getattr(regimen_fit, "hazard", None) for regimen_fit in result.fits.values())
    walks = [walk for walk in found if walk is not None]
    converged = all(bool(walk.converged) for walk in walks)
    score = max((final_score(walk) for walk in walks), default=0.0)
    return result, FitRecord(converged, unexpected_names, score)


def _estimator(fit_label: str, configuration: str, simultaneous: bool) -> LTMLE:
    law_name, folds = FITS[fit_label]
    return LTMLE(
        regimens(law_name),
        reference=labels(law_name)[1],
        n_folds=folds,
        simultaneous=simultaneous,
        survival_model="hazard",
        targeting=ALGORITHM,
        tol=TOL,
        random_state=0,
        **learners(configuration),
    )


def fit_recorded(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    weights: str | None = None,
) -> tuple[Any, FitRecord]:
    """The subject's fit of one fit label's sample, with its :class:`FitRecord`."""
    current = law_of(fit_label)
    return recorded(
        lambda: _estimator(fit_label, configuration, simultaneous).fit(
            frame, **current.columns(weights=weights)
        )
    )


def fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    weights: str | None = None,
) -> Any:
    """The subject's fit of one fit label's sample."""
    return fit_recorded(
        fit_label,
        frame,
        configuration=configuration,
        simultaneous=simultaneous,
        weights=weights,
    )[0]


def untargeted_fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "mechanism_correct",
    weights: str | None = None,
) -> Any:
    """The untargeted backward pass of the same fit: every walk skipped (plan X70, 3.4).

    Each continuation is fitted on the untargeted pseudo-outcomes of the later nodes, so its
    estimates are the plug-in the targeting-necessity family reads.
    """
    with untargeted():
        return fit(fit_label, frame, configuration=configuration, weights=weights)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law_of(scenario).draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


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


def initial_estimates(result: Any, law_name: str) -> dict[str, float]:
    """Node 1's initial plug-in of every reported name, from each regimen's walk record.

    On a backward pass that plug-in's continuation was fitted on the targeted later nodes, so
    ``estimate - initial_estimate`` measures node 1's fluctuation, not the whole targeting step
    (:class:`cleverly.longitudinal.hazard.HazardFit`); the untargeted plug-in is the targeting
    necessity family's control.
    """
    levels: dict[tuple[str, int, int], float] = {}
    for regimen_fit in result.fits.values():
        record = regimen_fit.hazard
        causes = list(dict.fromkeys(cause for _, cause, _ in record.targets))
        for (label, cause, horizon), value in zip(
            record.targets, record.initial_levels, strict=True
        ):
            levels[(label, causes.index(cause) + 1, horizon)] = float(value)
    current = law.LAWS[law_name]
    first, second = labels(law_name)
    out: dict[str, float] = {}
    for j in range(1, current.causes + 1):
        for t in range(1, current.n_times + 1):
            for label in (first, second):
                out[risk_name(law_name, label, t, j)] = levels[(label, j, t)]
            out[contrast_name(law_name, t, j)] = levels[(first, j, t)] - levels[(second, j, t)]
    return out


def subject_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's primary rows, each estimate on the identity scale."""
    initials = initial_estimates(result, FITS[scenario][0])
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
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


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result = fit(scenario, frame)
    return sample, truth_rows, subject_rows(result, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of every scenario, its truth rows and the subject's rows."""
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
    """The declared truth of every primary estimand, by scenario."""
    return TRUTH


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]
