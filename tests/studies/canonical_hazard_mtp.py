"""Declared evidence for a modified treatment policy on the hazard path (``hazard-mtp``).

The subject is ``LTMLE(survival_model="hazard", targeting="one_step")`` with a modified treatment
policy at the baseline decision, held over every visit.  Each policy is targeted by its own walk,
whose loss is weighted by :math:`\\omega_k = w\\, r(A, W) / G_k(A, W)` at the observed dose, and
the risk curve is the plug-in average of the targeted hazards at each unit's shifted dose
(:mod:`cleverly.longitudinal.hazard`; ``docs/technical-reference/longitudinal-tmle.md``, "A
modified treatment policy on the hazard path").

**The laws** are those of :mod:`tests.studies.hazard_mtp_law`: ``levels`` (a dose on ``0..5``
read as six levels, the policy "one level up, the top stays", five visits), ``gaussian`` (a
Gaussian dose read as a continuous node, the additive shift :math:`\\delta = 1`, five visits),
``truncated`` (the same dose truncated to ``[0, 5.5]``, the shift capped at 5.5: a unit whose
shifted dose would pass the cap keeps its own) and ``vector`` (a binary arm beside a Gaussian
dose, a joint policy that shifts the dose, one visit).
The reference regimen of every law is the natural course, the identity policy, which takes a
walk of its own.  Plan X70 section 5 names the shared ``dose`` scenario of
:mod:`tests.studies.point_survival_common` for ``levels``.  Its drop-out is linear on the
probability scale, so no logistic learner is correct for it; and its hazard leaves the wrong
hazard learner's plug-in contrast about one efficiency SE from the truth.  The study's laws keep
its dose mechanism family and replace the drop-out by a logistic one and the hazard by one with
larger ``W`` and dose coefficients.

**Fit labels** (:data:`FITS`).  ``levels``; ``gaussian_density`` (the binned density route on
``truncated``, with a fitted truncated-normal location density,
:class:`~tests.studies.hazard_mtp_law.GaussianDoseHazard`, at
:func:`~tests.studies.oracle_density_bins.oracle_bins` bins); ``gaussian_classifier`` (the
classifier route on ``gaussian``, a logistic classifier on ``(W1, W2, A)``); ``gaussian_crossfit``
(the same classifier at five folds, the cross-fitted variant); and ``vector`` (a mixed node, the
classifier route).  The cross-fitted variant exercises the cross-fitting of the shifted-dose
predictions (``cross_fit_predictions(..., predict_designs=)``), which does not depend on the
ratio route, so one route covers it; the study's density learner reads the sample it is handed
through its constructor, so the classifier carries the cross-fitted cell.  A vector held over
several visits cannot be declared in the data container (plan X70 rev 2.5), so ``vector`` has
one visit.

**Why the density route is gated on the truncated law.**  On the untruncated Gaussian law the
policy ratio grows exponentially in the dose, and the top quantile bins at ``oracle_bins(n)``
are wider than the shift, so a dose and its pre-image share a bin and the binned ratio reads one
where the true ratio is 5 to 8.  The influence curve of that fit then stays away from the
efficient influence function as n grows (``tests/unit/test_hazard_mtp_study_design.py``,
``TestTheInfluenceCurveGap``).  The reported fit label ``untruncated_density``
(:data:`REPORTED_FITS`) keeps that fit as an ungated row, so the shipped route's tail binning is
disclosed and not hidden.

**Truths and bounds** are exact sums over the dose levels, or Gauss-Hermite quadratures over a
Gaussian dose that reach ``1e-10``, computed by code that never calls the estimator.  The design
test ``tests/unit/test_hazard_mtp_study_design.py`` checks each by an independent route.

**No comparator.**  R ``lmtp`` and ``ltmle`` fit the iterated-mean recursion and ``survtmle`` a
baseline treatment held at a level; none targets discrete-time hazards under a modified
treatment policy.  The record names no reference.  The independent evidence is the exact truths
and bounds, the property families, and the exact-law tests of ``tests/unit/test_hazard_mtp.py``.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/hazard_mtp/regenerate.py`` before any run.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Shift
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import hazard_mtp_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.oracle_density_bins import oracle_bins

#: PROVISIONAL.  A placeholder until the calibration pilot is committed: the shared primary rule
#: (``repeated_sizing.primary_replicates``, read by the design test) then sets this count from
#: the calibration pilots of the primary fit labels, in a declaration commit with no skipped
#: design test, before any declared run.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261119
RESAMPLING_SEED = 2026111901

#: The targeting settings of every hazard fit of this study.
ALGORITHM = "one_step"
TOL = 1e-10
#: The policy's regimen label and the reference's.
SHIFT = "shift"
NATURAL = "natural"

#: Fit label -> ``(law, ratio route, folds)``.
FITS: dict[str, tuple[str, str, int]] = {
    "levels": ("levels", "density", 1),
    "gaussian_density": ("truncated", "density", 1),
    "gaussian_classifier": ("gaussian", "classifier", 1),
    "gaussian_crossfit": ("gaussian", "classifier", 5),
    "vector": ("vector", "classifier", 1),
}
#: The reported, ungated fit label: the density route on the untruncated Gaussian law, whose
#: quantile tail bins are wider than the shift.
REPORTED_FITS: dict[str, tuple[str, str, int]] = {
    "untruncated_density": ("gaussian", "density", 1),
}
ALL_FITS = {**FITS, **REPORTED_FITS}
#: The replicates of the reported, ungated rows.
REPORTED_REPLICATES = 800
#: The primary scenarios: every fit label but the cross-fitted variant, which calibration reads.
SCENARIOS = ("levels", "gaussian_density", "gaussian_classifier", "vector")

_INF = np.inf


def _up(a: Any, h: Any) -> Any:
    """One level up, the top stays."""
    return np.minimum(np.asarray(a, dtype=float) + 1.0, float(law.LEVELS[-1]))


def regimens(law_name: str) -> dict[str, Any]:
    """The policy and the natural course of one law."""
    current = law.LAWS[law_name]
    if current.kind == "levels":
        return {
            SHIFT: ModifiedPolicy("up", apply=_up, policy_kind="known"),
            NATURAL: ModifiedPolicy("natural", pieces=(Piece(-_INF, _INF),), policy_kind="known"),
        }
    if current.kind == "gaussian":
        return {
            SHIFT: Shift(current.shift, cap=current.cap),
            NATURAL: ModifiedPolicy("natural", pieces=(Piece(-_INF, _INF),), policy_kind="known"),
        }
    return {
        SHIFT: JointPolicy(
            SHIFT, components={"A": Shift(current.shift, cap=None)}, policy_kind="known"
        ),
        NATURAL: JointPolicy(NATURAL, components={}, policy_kind="known"),
    }


def risk_name(label: str, horizon: int) -> str:
    return f"risk_regimen[{label} @ t={horizon}]"


def contrast_name(horizon: int) -> str:
    return f"ate_regimen[{SHIFT} vs {NATURAL} @ t={horizon}]"


def estimand_names(law_name: str) -> tuple[str, ...]:
    """Every estimate a fit of one law reports, in the fit's order."""
    n_times = law.LAWS[law_name].n_times
    return (
        *(risk_name(SHIFT, t) for t in range(1, n_times + 1)),
        *(risk_name(NATURAL, t) for t in range(1, n_times + 1)),
        *(contrast_name(t) for t in range(1, n_times + 1)),
    )


def truths(law_name: str) -> dict[str, float]:
    """The exact value of every estimate of one law, by reported name."""
    current = law.LAWS[law_name]
    out: dict[str, float] = {}
    for t in range(1, current.n_times + 1):
        shifted, natural = current.truth(t), current.truth(t, natural=True)
        out[risk_name(SHIFT, t)] = shifted
        out[risk_name(NATURAL, t)] = natural
        out[contrast_name(t)] = shifted - natural
    return {name: out[name] for name in estimand_names(law_name)}


def law_of(fit_label: str) -> law.Law:
    return law.LAWS[ALL_FITS[fit_label][0]]


ESTIMANDS = {scenario: estimand_names(FITS[scenario][0]) for scenario in SCENARIOS}
TRUTH = {scenario: truths(FITS[scenario][0]) for scenario in SCENARIOS}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


#: The calibration fit labels and their two arms: the contrast and the policy's risk at the
#: last visit.
CALIBRATION_FITS = tuple(FITS)
CALIBRATION_LABELS = tuple(
    f"{fit}_{arm}" for fit in CALIBRATION_FITS for arm in ("contrast", "risk")
)
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = ("levels_contrast", "gaussian_density_contrast")
#: The contrast every robustness, necessity and power cell reads.
CONTRAST_LABEL = "levels_contrast"
#: The ratio-route robustness cells: the wrong hazard learner and the correct mechanism, on each
#: continuous fit label, so the weighted loss is read on the mixed node and under cross-fitting.
ROUTE_DR_LABELS = (
    "gaussian_density_contrast",
    "gaussian_classifier_contrast",
    "gaussian_crossfit_contrast",
    "vector_contrast",
)
#: The reported, ungated cells: the calibration of the density route on the untruncated law.
REPORTED_LABELS = tuple(f"{fit}_{arm}" for fit in REPORTED_FITS for arm in ("contrast", "risk"))
#: The joint labels: the band over every estimate of one fit.
JOINT_LABELS = ("levels_curve", "gaussian_density_curve")
RATE_SIZES = (1_000, 2_000, 8_000)
LADDER_LABELS = ("levels_contrast", "levels_risk")
#: The fits whose draws carry the reported comparison with the iterated-mean fit.
DIFFERENCE_FITS = ("levels", "gaussian_density")
DIFFERENCE_ARMS = ("paired_difference", "hazard_fit", "iterated_fit", "isotonic_difference")

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        *(
            cell
            for label in CALIBRATION_LABELS
            for cell in (
                _cells(label, "correctly_specified", "shrunken_se_control", "noise_control")
                if label in CONTROLLED_LABELS
                else _cells(label, "correctly_specified")
            )
        ),
        *(cell for label in REPORTED_LABELS for cell in _cells(label, "reported")),
    ),
    "double_robustness": (
        *_cells(
            CONTRAST_LABEL, "both_correct", "outcome_correct", "mechanism_correct", "both_wrong"
        ),
        *(cell for label in ROUTE_DR_LABELS for cell in _cells(label, "mechanism_correct")),
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
        *_cells(CONTRAST_LABEL, "mechanism_correct_difference"),
        *_cells("levels_risk", "mechanism_correct_difference"),
    ),
}

STUDY = StudyRecord(
    name="modified treatment policy on the hazard path, one-step targeting",
    slug="hazard-mtp",
    artifacts=ROOT / "tests" / "canonical" / "hazard_mtp",
    document="docs/technical-reference/method-evidence/hazard-mtp.md",
    anchor="hazard-mtp",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_hazard_mtp.py",
        "tests/studies/hazard_mtp_properties.py",
        "tests/studies/hazard_mtp_law.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/oracle_density_bins.py",
        "src/cleverly/longitudinal/hazard.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_hazard_mtp",
    properties_module="tests.studies.hazard_mtp_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "held baseline treatment, a modified treatment policy, one-step hazard walk "
    "per policy with the loss weighted by w r / G_k",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "tol": TOL,
    "fits": {label: list(spec) for label, spec in FITS.items()},
    "outcome_learner": "unpenalized logistic regression on W1, W2, (F,) A per visit",
    "treatment_learner": "multinomial logistic on W1, W2 (levels); a truncated-normal location "
    "density by maximum likelihood, read through the binned hazards (density route, truncated "
    "law; least squares on the reported untruncated row); logistic classifier on W1, W2, (F,) A "
    "(classifier route)",
    "censoring_learner": "unpenalized logistic regression on W1, W2, (F,) A per visit",
    "regimens": {
        "shift": "one level up (levels); A + 1 (gaussian, vector); A + 1 capped at 5.5 (truncated)",
        "natural": "identity",
    },
}


def learners(fit_label: str, frame: pd.DataFrame, configuration: str = "both_correct") -> dict:
    """The hazard and mechanism learners of one fit label and nuisance configuration.

    The wrong hazard learner is :class:`~tests.studies.hazard_mtp_law.NoWLogistic`.  The wrong
    mechanism is the prior share for the levels and the arm, a Gaussian with no covariates for
    the density route, and a classifier on the dose alone for the classifier route; the wrong
    censoring learner is the prior share.
    """
    from sklearn.dummy import DummyClassifier

    current = law_of(fit_label)
    _, ratio, _ = ALL_FITS[fit_label]
    hazard_correct = configuration in {"both_correct", "outcome_correct"}
    mechanism_correct = configuration in {"both_correct", "mechanism_correct"}
    out: dict[str, Any] = {
        "outcome_learner": (
            law.HazardLogistic(current.kind) if hazard_correct else law.NoWLogistic(current.kind)
        ),
        "censoring_learner": (
            law.CensoringLogistic(current.kind)
            if mechanism_correct
            else DummyClassifier(strategy="prior")
        ),
    }
    if current.kind == "levels":
        out["treatment_learner"] = (
            law.LevelsTreatment() if mechanism_correct else DummyClassifier(strategy="prior")
        )
    elif ratio == "classifier":
        out["treatment_learner"] = law.RatioLogistic() if mechanism_correct else law.DoseOnlyRatio()
    else:
        bins = oracle_bins(len(frame))
        covariates = ("W1", "W2", "F") if current.kind == "vector" else ("W1", "W2")
        out["treatment_learner"] = law.GaussianDoseHazard(
            tuple(float(v) for v in bin_edges(frame["A"].to_numpy(dtype=float), bins)),
            covariates,
            frame,
            marginal=not mechanism_correct,
            bounds=current.bounds,
        )
    return out


#: The warning classes a fit label is expected to raise, filtered by name, with why.  Every
#: other warning a fit raises is counted in its rows (``unexpected_warnings``); on a gated fit
#: label a ``SparseInfluenceWarning`` is unexpected and counted.
EXPECTED_WARNINGS: dict[str, dict[str, str]] = {
    "untruncated_density": {
        "SparseInfluenceWarning": "the uncapped Gaussian shift's ratio is heavy in the top dose "
        "ranks, so a few rows carry the influence curve's variance; the reported row reads it",
    },
}
#: The support warning of a shift, filtered by its category and message: the uncapped shift
#: moves the largest observed dose past the observed range by construction.
SUPPORT_WARNING = r"policy .* (has cap=|assigns )"


@dataclass(frozen=True)
class FitRecord:
    """What a fit's warnings and walks say about it, recorded beside its rows.

    Parameters
    ----------
    converged : bool
        Whether every hazard walk of the fit met its tolerance.
    unexpected_warnings : tuple of str
        The class names of the warnings outside the fit label's :data:`EXPECTED_WARNINGS` and
        the support warning, one entry per warning.
    final_score : float
        The largest final relative score over the fit's hazard walks: the largest score
        component over its largest possible magnitude, ``max |score| / score_scale``, at the
        reported hazards.
    """

    converged: bool
    unexpected_warnings: tuple[str, ...]
    final_score: float = 0.0

    def columns(self) -> dict[str, float]:
        """The per-row columns: ``converged`` (0 or 1), ``final_score`` and the warning count."""
        return {
            "converged": int(self.converged),
            "final_score": self.final_score,
            "unexpected_warnings": len(self.unexpected_warnings),
        }


def final_score(walk: Any) -> float:
    """``max |score| / score_scale`` of one hazard walk at its reported hazards."""
    score = np.abs(np.asarray(walk.score, dtype=float)).reshape(-1)
    if score.size == 0:
        return 0.0
    scale = np.asarray(walk.score_scale, dtype=float).reshape(-1)
    return float(np.max(score / np.maximum(scale, 1e-300)))


def recorded(call: Any, expected: Sequence[str] = ()) -> tuple[Any, FitRecord]:
    """Run ``call()``, filtering the ``expected`` warning classes and recording every other one."""
    from cleverly.exceptions import PositivityWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=SUPPORT_WARNING, category=PositivityWarning)
        result = call()
    unexpected = tuple(
        item.category.__name__ for item in caught if item.category.__name__ not in expected
    )
    found = (getattr(regimen_fit, "hazard", None) for regimen_fit in result.fits.values())
    walks = [walk for walk in found if walk is not None]
    converged = all(bool(walk.converged) for walk in walks)
    score = max((final_score(walk) for walk in walks), default=0.0)
    return result, FitRecord(converged, unexpected, score)


def fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
) -> Any:
    """The subject's fit of one fit label's sample."""
    return fit_recorded(fit_label, frame, configuration=configuration, simultaneous=simultaneous)[0]


def fit_recorded(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
) -> tuple[Any, FitRecord]:
    """The subject's fit of one fit label's sample, with its :class:`FitRecord`."""
    law_name, ratio, folds = ALL_FITS[fit_label]
    settings: dict[str, Any] = {"ratio": ratio}
    if ratio == "density" and law.LAWS[law_name].continuous:
        settings["density_bins"] = oracle_bins(len(frame))
    return recorded(
        lambda: LTMLE(
            regimens(law_name),
            reference=NATURAL,
            n_folds=folds,
            simultaneous=simultaneous,
            survival_model="hazard",
            targeting=ALGORITHM,
            tol=TOL,
            random_state=0,
            **settings,
            **learners(fit_label, frame, configuration),
        ).fit(law.LAWS[law_name].container(frame)),
        expected=tuple(EXPECTED_WARNINGS.get(fit_label, {})),
    )


def initial_estimates(result: Any, names: Sequence[str]) -> dict[str, float]:
    """The plug-in of the initial hazards of each name, from each policy's walk record."""
    levels: dict[tuple[str, int], float] = {}
    for key, regimen_fit in result.fits.items():
        record = regimen_fit.hazard
        for (label, _, horizon), value in zip(record.targets, record.initial_levels, strict=True):
            levels[(label, horizon)] = float(value)
        del key
    out: dict[str, float] = {}
    for name in names:
        horizon = int(name.rsplit("t=", 1)[1].rstrip("]"))
        if name.startswith("ate_regimen"):
            out[name] = levels[(SHIFT, horizon)] - levels[(NATURAL, horizon)]
        else:
            label = SHIFT if name.startswith(f"risk_regimen[{SHIFT} ") else NATURAL
            out[name] = levels[(label, horizon)]
    return out


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


def subject_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's primary rows, each estimate on the identity scale."""
    initials = initial_estimates(result, ESTIMANDS[scenario])
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
