"""Declared evidence for a treatment at several nodes on the hazard path (``hazard-multinode``).

The subject is ``LTMLE(survival_model="hazard")`` on a wide design whose binary treatment is
decided again at every node, with the per-horizon walk (``targeting="iterative"``, the default)
for the regimen fits and the one-step walk for the working model.  The slots of the walk are full treatment
paths: every regimen contributes the paths it can follow, a policy node enumerates both arms
with its own probabilities, and the copies of one path prefix share one update.  The risk of
each regimen is the plug-in sum over its paths of the path mass times the targeted path risk
(:mod:`cleverly.longitudinal.hazard`; ``docs/technical-reference/longitudinal-tmle.md``, "A
treatment at several nodes on the hazard path").

**The laws** are those of :mod:`tests.studies.hazard_multinode_law`: ``base`` (plan X70
section 4.3, three nodes) and ``four`` (the same form at four nodes with its own coefficients,
so the policy has 16 paths).  The treatment depends on ``W`` and the previous arm, dropout on
``W`` and the last two arms, and the hazard on ``W`` and the last two arms.  Plan X70 asked for
a five-node variant; at five nodes the policy reads 32 prefix cells and a held cell is expected
in about one fit in 500 at n = 2,000, so the variant has four nodes (design review a,
finding 1).

**Fit labels** (:data:`FITS`).  ``base`` fits the five regimens of section 4.3 (``always``,
``never``, the known rule ``rule``, the known policy ``policy`` and ``stay``, which keeps the
previous arm) with ``never`` as reference.  ``four`` fits ``always``, ``never`` and ``policy``
on the four-node law.  ``msm`` fits the logit working model :math:`\\operatorname{logit} m =
\\beta_0 + \\beta_1 d + \\beta_2 t + \\beta_3 d W_1` over the eight static paths of the base law,
with :math:`d` the cumulative dose of the path through :math:`t`.  These three are the primary
scenarios (:data:`SCENARIOS`).  ``onestep`` fits ``always``, ``never`` and ``policy`` on
``base`` with the one-step walk; it carries only the reported ``one_step_walk`` family.

**Held prefix cells.**  When no unit at risk with one treatment prefix has the event (or every
one does), the shipped fit holds that cell and every regimen that reads it reports the plug-in
with no interval (status ``constant_node_plugin``).  :func:`refuse_held_cells` turns such a fit
into a recorded lost replicate on every path, primary, property and pilot, and the design test
bounds the expected count of such fits over the whole study.

**Truths and bounds** are exact sums over the paths and the support points of the observed law,
computed by code that never calls the estimator, and the efficient influence function there is
written in its iterated-mean form, which shares nothing with the estimator's hazard form.  The
design test ``tests/unit/test_hazard_multinode_study_design.py`` checks each against an
independent route and against a fit on the law's exact sample.

**No comparator.**  R ``ltmle`` and ``lmtp`` fit the iterated-mean recursion, and ``survtmle``
a baseline treatment held at a level; none targets discrete-time hazards over treatment paths.
The record names no reference.  The independent evidence is the exact truths and bounds, the
property families, and the exact-law tests of ``tests/unit/test_hazard_multinode.py``.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/hazard_multinode/regenerate.py`` before any run.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.exceptions import DataError
from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import hazard_multinode_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: Set by the shared primary rule (``repeated_sizing.primary_replicates``, read by the design
#: test) from the committed calibration pilots of the primary fit labels.  Provisional until
#: those pilots are committed.
PRIMARY_REPLICATES = 2_400
PRIMARY_N = 2_000
SEED = 20261115
RESAMPLING_SEED = 2026111501

#: The targeting algorithm of the regimen fits: the per-horizon walk, the shipped default, at
#: about 1.5 seconds a ``base`` fit at n = 2,000.  On one ``base`` draw the one-step walk took
#: 0.6 seconds over always and never, 33.5 over the policy and "stay", and 96 over always,
#: never and the policy: the enumerated policy slows the one-step walk, not the rule.  The
#: one-step walk on regimen curves is read by the reported ``onestep`` fit label.
ALGORITHM = "iterative"
#: The fit labels that take the one-step walk: the working model, the only fit whose scores
#: the per-horizon walk does not target, and the reported ``onestep`` label.
ONE_STEP_FITS = ("msm", "onestep")
TOL = 1e-10
#: The lower bound on the cumulative treatment and retention probability of every fit.  The
#: shipped default, 0.01, lies within a factor of two of the smallest cumulative probability of
#: ``base`` (0.0196; 0.042 on ``four``), so the declared bound is 1e-3, ten times below both,
#: and no exact fit is clipped.  The pilot records each fit's smallest estimated cumulative
#: probability, and so how often the default would have clipped.
G_BOUNDS = 1e-3
#: The reference regimen of the regimen fits.
REFERENCE = "never"

#: Fit label -> ``(law, regimen labels, working model)``.
FITS: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "base": ("base", law.LABELS, False),
    "four": ("four", ("always", "never", "policy"), False),
    "msm": ("base", law.static_paths(3), True),
    "onestep": ("base", ("always", "never", "policy"), False),
}
#: The primary scenarios: every fit label but the reported ``onestep``.
SCENARIOS = ("base", "four", "msm")


def law_of(fit_label: str) -> law.Law:
    return law.LAWS[FITS[fit_label][0]]


def algorithm(fit_label: str) -> str:
    """The targeting algorithm of one fit label."""
    return "one_step" if fit_label in ONE_STEP_FITS else ALGORITHM


def regimens(fit_label: str) -> dict[str, Any]:
    """The fit-side regimens of one fit label."""
    _, labels, _ = FITS[fit_label]
    return law.regimens(labels, law_of(fit_label).n_times)


def risk_name(label: str, horizon: int) -> str:
    return f"risk_regimen[{label} @ t={horizon}]"


def contrast_name(label: str, horizon: int) -> str:
    return f"ate_regimen[{label} vs {REFERENCE} @ t={horizon}]"


def estimand_names(fit_label: str) -> tuple[str, ...]:
    """Every estimate a fit of one fit label reports, in the fit's order."""
    _, labels, msm = FITS[fit_label]
    if msm:
        return tuple(law.coefficient_name(term) for term in law.TERMS)
    horizons = range(1, law_of(fit_label).n_times + 1)
    return (
        *(risk_name(label, t) for label in labels for t in horizons),
        *(contrast_name(label, t) for label in labels if label != REFERENCE for t in horizons),
    )


def truths(fit_label: str) -> dict[str, float]:
    """The exact value of every estimate of one fit label, by reported name."""
    current = law_of(fit_label)
    _, labels, msm = FITS[fit_label]
    if msm:
        beta = law.exact_beta(current)
        return {
            law.coefficient_name(term): float(b) for term, b in zip(law.TERMS, beta, strict=True)
        }
    out: dict[str, float] = {}
    for t in range(1, current.n_times + 1):
        reference = current.risk(REFERENCE, t)
        for label in labels:
            value = current.risk(label, t)
            out[risk_name(label, t)] = value
            if label != REFERENCE:
                out[contrast_name(label, t)] = value - reference
    return {name: out[name] for name in estimand_names(fit_label)}


ESTIMANDS = {scenario: estimand_names(scenario) for scenario in SCENARIOS}
TRUTH = {fit_label: truths(fit_label) for fit_label in FITS}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


#: The calibration fit labels and their arms: on ``base`` the contrast of always with never by
#: the last node and the risks of the rule, the policy and "stay"; on ``four`` the contrast and
#: the policy's risk; on ``msm`` the dose and the dose-by-modifier coefficients.  The reported
#: ``onestep`` label reads the contrast and the policy's risk of ``base``.
CALIBRATION_FITS = SCENARIOS
ARMS: dict[str, tuple[str, ...]] = {
    "base": ("contrast", "rule", "policy", "stay"),
    "four": ("contrast", "policy"),
    "msm": ("dose", "dose_W1"),
    "onestep": ("contrast", "policy"),
}
CALIBRATION_LABELS = tuple(f"{fit}_{arm}" for fit in CALIBRATION_FITS for arm in ARMS[fit])
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = ("base_contrast",)
#: The contrast every robustness, necessity and power cell reads.
CONTRAST_LABEL = "base_contrast"
#: The ``mechanism_correct`` robustness cells beyond the four configurations of the contrast:
#: every other calibration arm, each read from the ``mechanism_correct`` fit of its fit label.
#: Where the wrong hazard learner's plug-in lies near the truth (``base_rule`` +0.46,
#: ``base_stay`` -0.96, ``base_policy`` -1.01, ``four_contrast`` +0.76 and ``four_policy``
#: -0.16 efficiency SEs at n = 2,000) a cell is a consistency check: it passes whether or not
#: the targeting moves the estimate.  The weighting over the policy paths rests on the exact-law
#: witnesses of ``tests/unit/test_hazard_multinode.py`` and on :data:`NODE_DR_LABEL`.
ROUTE_DR_LABELS = tuple(label for label in CALIBRATION_LABELS if label != CONTRAST_LABEL)
#: The robustness cell of the node-only hazard learner (the at-risk event share at each node,
#: no arm and no ``W``) with the correct mechanism, on the ``four`` contrast.  Its plug-in
#: contrast is zero in every sample, 2.20 efficiency SEs from the truth at n = 2,000, so the
#: targeting alone moves the estimate to the truth.
NODE_DR_LABEL = "four_contrast"
NODE_CONFIGURATION = "node_hazard"
#: The joint labels: the band over every estimate of one fit.
JOINT_LABELS = ("base_curve", "four_curve")
#: The ladder.  Its smallest rung is a rate rung that carries no claim; at 1,600 rows a ``base``
#: fit holds a prefix cell with probability 1e-3 (0.017 at 1,000).
RATE_SIZES = (1_600, 2_000, 8_000)
LADDER_LABELS = ("base_contrast", "base_policy")
#: The fits whose draws carry the reported comparison with the iterated-mean fit.
DIFFERENCE_FITS = ("base", "four")
DIFFERENCE_ARMS = ("paired_difference", "hazard_fit", "iterated_fit", "isotonic_difference")
#: The reported fit label of the one-step walk on regimen curves, its family and its arms: the
#: one-step fit's own row, and its paired difference against the per-horizon fit of the same
#: regimens on the same draw.
ONE_STEP_FIT = "onestep"
ONE_STEP_FAMILY = "one_step_walk"
ONE_STEP_ARMS = ("one_step_fit", "paired_difference")

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
    "double_robustness": (
        *_cells(
            CONTRAST_LABEL, "both_correct", "outcome_correct", "mechanism_correct", "both_wrong"
        ),
        *(cell for label in ROUTE_DR_LABELS for cell in _cells(label, "mechanism_correct")),
        *_cells(NODE_DR_LABEL, NODE_CONFIGURATION),
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
            for arm in ARMS[fit]
            for cell in _cells(f"{fit}_{arm}", *DIFFERENCE_ARMS)
        ),
        *(f"{fit}__iterated_curve_decrease" for fit in DIFFERENCE_FITS),
        *(
            cell
            for arm in ARMS["base"]
            for cell in _cells(f"base_{arm}", "mechanism_correct_difference")
        ),
    ),
    ONE_STEP_FAMILY: tuple(
        cell
        for arm in ARMS[ONE_STEP_FIT]
        for cell in _cells(f"{ONE_STEP_FIT}_{arm}", *ONE_STEP_ARMS)
    ),
}

STUDY = StudyRecord(
    name="treatment at several nodes on the hazard path",
    slug="hazard-multinode",
    artifacts=ROOT / "tests" / "canonical" / "hazard_multinode",
    document="docs/technical-reference/method-evidence/hazard-multinode.md",
    anchor="hazard-multinode",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_hazard_multinode.py",
        "tests/studies/hazard_multinode_properties.py",
        "tests/studies/hazard_multinode_law.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/msm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_hazard_multinode",
    properties_module="tests.studies.hazard_multinode_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "a binary treatment at every node, slots over full treatment paths with "
    "shared prefix updates; the per-horizon walk over every regimen, the one-step walk for the "
    "working model and the reported onestep fit label",
    "survival_model": "hazard",
    "targeting": {label: algorithm(label) for label in FITS},
    "tol": TOL,
    "g_bounds": G_BOUNDS,
    "fits": {label: [spec[0], list(spec[1]), spec[2]] for label, spec in FITS.items()},
    "working_model": "logit m = b0 + b1 d + b2 t + b3 d W1 over (static path, node), uniform h",
    "outcome_learner": "unpenalized logistic regression on W1, W2, A1..Ak per node",
    "treatment_learner": "unpenalized logistic regression on W1, W2, A1..A(k-1) per node",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A1..Ak per node",
    "regimens": {
        "always": "treat at every node",
        "never": "treat at no node",
        "rule": "treat at every node when W2 >= 2 (known rule)",
        "policy": "treat with probability 0.3 + 0.4 A(k-1) + 0.1 W1 (known policy, A0 = 0)",
        "stay": "treat at node 1 with probability 0.3 + 0.2 W1, then keep the previous arm",
        "pXYZ": "the static path X, Y, Z (msm)",
    },
}


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The hazard and mechanism learners of one nuisance configuration.

    The wrong hazard learner is :class:`~tests.studies.hazard_multinode_law.LastArmLogistic`;
    the wrong treatment and censoring learners are the prior share at each node.  The
    ``node_hazard`` configuration fits the node-only hazard learner, the prior share of the
    event at each node, with the correct mechanism.
    """
    from sklearn.dummy import DummyClassifier

    hazard_correct = configuration in {"both_correct", "outcome_correct"}
    mechanism_correct = configuration in {"both_correct", "mechanism_correct", NODE_CONFIGURATION}

    def mechanism() -> Any:
        return law.HistoryLogistic() if mechanism_correct else DummyClassifier(strategy="prior")

    def hazard() -> Any:
        if hazard_correct:
            return law.HistoryLogistic()
        if configuration == NODE_CONFIGURATION:
            return DummyClassifier(strategy="prior")
        return law.LastArmLogistic()

    return {
        "outcome_learner": hazard(),
        "treatment_learner": mechanism(),
        "censoring_learner": mechanism(),
    }


_SPARSE = (
    "the influence curve of a static arm's risk by the last node is carried by the few units "
    "that followed the arm to an event, whose weight is the inverse of a cumulative treatment "
    "and retention probability as small as 0.02, so the variance estimate's relative SD passes "
    "0.125 on every draw at n = 2,000 (Kish size about 15 for always at node 3 on base); the "
    "calibration rows read the resulting intervals"
)
#: The warning classes a fit label is expected to raise, filtered by name, with why.  Every
#: other warning a fit raises is counted in its rows (``unexpected_warnings``).
EXPECTED_WARNINGS: dict[str, dict[str, str]] = {
    "base": {"SparseInfluenceWarning": _SPARSE},
    "four": {"SparseInfluenceWarning": _SPARSE},
    "onestep": {"SparseInfluenceWarning": _SPARSE},
}


@dataclass(frozen=True)
class FitRecord:
    """What a fit's warnings and walk say about it, recorded beside its rows.

    Parameters
    ----------
    converged : bool
        Whether every hazard walk of the fit met its tolerance.
    unexpected_warnings : tuple of str
        The class names of the warnings outside the fit label's :data:`EXPECTED_WARNINGS`,
        one entry per warning.
    final_score : float
        The largest final relative score over the fit's hazard walks: the largest score
        component over its largest possible magnitude, ``max |score| / score_scale``, at the
        reported hazards.
    n_iter : int
        The largest iteration count over the fit's hazard walks.
    smallest_cumulative : float
        The smallest estimated cumulative treatment and retention probability of the fit,
        before ``g_bounds``, over every unit and node (``nan`` when the fit reports none).
    """

    converged: bool
    unexpected_warnings: tuple[str, ...]
    final_score: float = 0.0
    n_iter: int = 0
    smallest_cumulative: float = float("nan")

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


def walks(result: Any) -> list[Any]:
    """The distinct hazard walk records of a fit: one shared walk, or the working model's."""
    found = [getattr(item, "hazard", None) for item in result.fits.values()]
    found += [msm_fit.hazard for msm_fit in getattr(result, "msm_fits", None) or ()]
    out: list[Any] = []
    for walk in found:
        if walk is not None and not any(walk is seen for seen in out):
            out.append(walk)
    return out


def smallest_cumulative(result: Any) -> float:
    """The smallest unbounded cumulative treatment and retention probability of a fit.

    Every regimen fit of the hazard path carries the same ``(n, T)`` matrix at the observed
    histories, before ``g_bounds``; the working model's fit reads the same matrix.
    """
    found = [getattr(item, "cumulative_unbounded", None) for item in result.fits.values()]
    for msm_fit in getattr(result, "msm_fits", None) or ():
        found += [getattr(item, "cumulative_unbounded", None) for item in msm_fit.fits]
    values = [float(np.nanmin(np.asarray(m, dtype=float))) for m in found if m is not None]
    return min(values, default=float("nan"))


class HeldCellError(DataError):  # numpydoc ignore=PR01
    """A fit reported an estimate without the influence-curve interval (a held prefix cell)."""


def refuse_held_cells(fit_label: str, result: Any) -> None:
    """Turn a fit with a non-inferential estimate into a recorded lost replicate.

    A treatment prefix cell where no unit at risk has the event (or every one does) is held by
    the shipped fit, and every regimen that reads it reports the plug-in with no interval
    (status ``constant_node_plugin``).  The study measures the influence-curve interval, so the
    draw is a lost replicate: :class:`HeldCellError` is a ``DataError``, which the harness
    records under the failure bound, as ``canonical_survival_node_bounds.refuse_constant_nodes``
    does.  It is checked on every estimate of the fit before any row reads one, so the primary
    rows, the property rows (the joint band among them) and the pilot treat the draw alike.
    The design test bounds the expected count of such draws over the study.
    """
    for name in estimand_names(fit_label):
        status = result[name].inference
        if status != "influence_curve":
            raise HeldCellError(
                f"{name} reports the inference status {status!r}, not the influence curve"
            )


def recorded(call: Any, expected: Sequence[str] = ()) -> tuple[Any, FitRecord]:
    """Run ``call()``, filtering the ``expected`` warning classes and recording every other one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    unexpected = tuple(
        item.category.__name__ for item in caught if item.category.__name__ not in expected
    )
    found = walks(result)
    converged = all(bool(walk.converged) for walk in found)
    score = max((final_score(walk) for walk in found), default=0.0)
    n_iter = max((int(walk.n_iter) for walk in found), default=0)
    return result, FitRecord(converged, unexpected, score, n_iter, smallest_cumulative(result))


def fit(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    targeting: str | None = None,
) -> Any:
    """The subject's fit of one fit label's sample."""
    return fit_recorded(
        fit_label,
        frame,
        configuration=configuration,
        simultaneous=simultaneous,
        targeting=targeting,
    )[0]


def fit_recorded(
    fit_label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    targeting: str | None = None,
) -> tuple[Any, FitRecord]:
    """The subject's fit of one fit label's sample, with its :class:`FitRecord`.

    ``targeting`` overrides the fit label's algorithm (the ``onestep`` label's paired
    per-horizon fit).  A fit with a held prefix cell raises :class:`HeldCellError`
    (:func:`refuse_held_cells`).
    """
    current = law_of(fit_label)
    _, _, msm = FITS[fit_label]
    target: dict[str, Any] = {"msm": law.working_model()} if msm else {"reference": REFERENCE}
    result, record = recorded(
        lambda: LTMLE(
            regimens(fit_label),
            n_folds=1,
            simultaneous=simultaneous,
            survival_model="hazard",
            targeting=targeting or algorithm(fit_label),
            tol=TOL,
            g_bounds=G_BOUNDS,
            random_state=0,
            **target,
            **learners(configuration),
        ).fit(frame, **current.fit_columns()),
        expected=tuple(EXPECTED_WARNINGS.get(fit_label, {})),
    )
    refuse_held_cells(fit_label, result)
    return result, record


def initial_estimates(result: Any, names: Sequence[str]) -> dict[str, float]:
    """The plug-in of the initial hazards of each name, from the walk record.

    A regimen fit's walk holds the initial risks in ``(label, cause, horizon)`` order; the
    working model's walk holds the coefficients projected from the initial hazards in
    ``(term, cause)`` order.
    """
    out: dict[str, float] = {}
    levels: dict[tuple[str, int], float] = {}
    for msm_fit in getattr(result, "msm_fits", None) or ():
        record = msm_fit.hazard
        for (term, _), value in zip(record.targets, record.initial_levels, strict=True):
            out[law.coefficient_name(term)] = float(value)
    for regimen_fit in result.fits.values():
        record = getattr(regimen_fit, "hazard", None)
        if record is None or out:
            continue
        for (label, _, horizon), value in zip(record.targets, record.initial_levels, strict=True):
            levels[(label, horizon)] = float(value)
    for name in names:
        if name in out:
            continue
        inner = name.split("[", 1)[1].rstrip("]")
        regimen, horizon_text = inner.rsplit(" @ t=", 1)
        horizon = int(horizon_text)
        if name.startswith("ate_regimen"):
            label = regimen.split(" vs ", 1)[0]
            out[name] = levels[(label, horizon)] - levels[(REFERENCE, horizon)]
        else:
            out[name] = levels[(regimen, horizon)]
    return {name: out[name] for name in names}


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
