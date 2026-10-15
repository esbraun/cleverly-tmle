"""Canonical repeated-outcome ``IPSI`` evidence study (X32b, PR A).

``IPSI`` fits the repeated outcome ``Y_1, ..., Y_T`` of :func:`cleverly.datasets.make_longitudinal_repeated`
once, with ``outcome=Repeated(["Y1", "Y2", "Y3"])``, and reports the mean of ``Y_h`` under the
odds tilts :math:`\\delta \\in \\{0.5, 1, 2\\}` at ``h = 1, 2, 3``.  Each horizon is the
end-of-study fit of the panel truncated after node ``h`` (Kim, Kennedy and Naimi 2021,
Theorem 3.1 at ``t = h``; Theorem 4.1 is the influence function on the prefix).  The
truth-calibrated property study lives in :mod:`tests.studies.ipsi_repeated_properties`.

This study has **no comparator**.  Kim's research code has no pinned release, and
``npcausal::ipsi`` has no dropout, reads an end-of-study outcome and pools its treatment fit over
``(id, time)``.  A dropout-free pairing at each horizon would repeat what the X19 pairing already
shows at 1e-10.  The dropout claim rests on exact witnesses (``E5``, ``E6``, mutation ``N9`` in
``tests/unit/test_influence_gateaux_repeated.py``).  So the record has ``reference=None`` and
no ``pending_reference``.

Declared before any run
-----------------------

* Scenarios ``repeated_folds_5`` (Algorithm 1 of Kim et al.: five folds) and
  ``repeated_folds_1`` (in sample).  Both draw independent samples, so there is no seed owner.
* Plans ``Incremental(0.5)``, ``Incremental(1.0)`` (the natural course, the reference) and
  ``Incremental(2.0)``.  Horizons 1 to 3.  Fifteen estimands: nine means and six contrasts
  against the natural course.
* Outcome regressions are :class:`CellMeansBackoff`.  Treatment and censoring are the main-terms
  quasibinomial GLM, which is correctly specified on this law.  The study declares no mechanism.
  Kim's condition 4 then holds with any consistent outcome regression.
* Truths come from the forward composition of the law's conditionals under the tilt
  (:meth:`~tests.studies.ltmle_repeated_law.RepeatedOutcomeLaw.tilt_counterfactual`).  A fast
  test shows that the observed-law functional (Theorem 3.1) gives the same numbers to 1e-14.
* ``PRIMARY_N = 8,000``.  X19 S2 needed 24,000 because its propensity was saturated cell means.
  Here the propensity and the censoring mechanism are parametric and correct, so the
  :math:`(\\hat\\pi - \\pi)^2` remainder is of order ``p / n`` with ``p`` near 6.  The ``n`` rule
  of :mod:`tests.studies.repeated_sizing` (``n_rule_fires``) may double it, twice at most.
* ``PRIMARY_REPLICATES`` is :func:`tests.studies.ipsi_repeated_properties.primary_replicates`
  applied to pilot P4: the smallest multiple of 400 that meets clauses (a) to (d) of
  :func:`tests.studies.repeated_sizing.primary_replicates` at the share
  :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`, with one implementation.
* The cleverly rows publish ``initial_estimate = nan``.  ``IPSI`` fits no fluctuation, and its
  one-step estimate is the only estimate it reports.
* The registered :class:`Margins` defaults apply.  Publication policy ``gated``.  A red cell is
  diagnosed and fixed before a fresh declaration on fresh seeds.  No red cell is deferred.

Pilot state
-----------

:data:`PILOT_STATE` is ``"placeholder"`` until pilots P4, P5 and P6 are run
(``tests/canonical/ipsi_repeated/README.md`` gives the commands).  While it is a placeholder, the
pilot tables in :mod:`tests.studies.ipsi_repeated_properties` hold declared uniform planning
values, the sizes in this module are the rules applied to them, and a declared run is refused
(:func:`require_recorded_pilots`).  The disposable smoke run with explicit ``--replicates`` and
``--n`` is not a declared run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from cleverly.datasets import make_longitudinal_repeated
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, Repeated
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.canonical_ltmle import G_BOUNDS, regimen_rows
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.ltmle_repeated_properties import CellMeansBackoff

#: ``"placeholder"`` while the sizes below are the rules applied to declared uniform planning
#: values, and ``"recorded"`` once pilots P4, P5 and P6 are pinned in the properties module.
PILOT_STATE: Literal["placeholder", "recorded"] = "placeholder"

PRIMARY_N = 8_000
#: From ``ipsi_repeated_properties.primary_replicates`` on the primary pilot (see the module
#: docstring, Pilot state).  A fast test re-derives it.
PRIMARY_REPLICATES = 2_400
SEED = 20263301
RESAMPLING_SEED = 2026330101
SCENARIOS = ("repeated_folds_5", "repeated_folds_1")
#: The folds of each scenario's fit.
SCENARIO_FOLDS = {"repeated_folds_5": 5, "repeated_folds_1": 1}
HORIZONS = (1, 2, 3)
TIMES = 3
DELTAS = (0.5, 1.0, 2.0)
#: The tilt every contrast is read against: the natural course.
REFERENCE_DELTA = 1.0
REFERENCE = "natural course"

#: The fit's roles.  ``Y_s`` is prepended to block ``s + 1`` by the container.
COLUMNS: dict[str, Any] = {
    "outcome": Repeated([f"Y{t}" for t in range(1, TIMES + 1)]),
    "treatment": [f"A{t}" for t in range(1, TIMES + 1)],
    "baseline": ["W"],
    "time_varying": [[], ["L2"], ["L3"]],
    "censoring": [f"C{t}" for t in range(1, TIMES + 1)],
}


def plans() -> list[Incremental]:
    """The three tilts, in report order."""
    return [Incremental(delta) for delta in DELTAS]


def tilt_name(delta: float) -> str:
    """The name a fit reports for the tilt ``delta``: ``odds x0.5``, ``natural course``, ..."""
    return str(Incremental(delta).name)


def mean_name(delta: float, horizon: int) -> str:
    """The estimand of the mean of ``Y_h`` under the tilt ``delta``."""
    return f"ey_regimen[{tilt_name(delta)} @ t={horizon}]"


def contrast_name(delta: float, horizon: int, reference: float = REFERENCE_DELTA) -> str:
    """The estimand of the contrast of the tilt ``delta`` with ``reference`` at ``horizon``."""
    return f"ate_regimen[{tilt_name(delta)} vs {tilt_name(reference)} @ t={horizon}]"


MEAN_NAMES = tuple(mean_name(delta, horizon) for delta in DELTAS for horizon in HORIZONS)
CONTRAST_NAMES = tuple(
    contrast_name(delta, horizon)
    for delta in DELTAS
    if delta != REFERENCE_DELTA
    for horizon in HORIZONS
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)

# ---------------------------------------------------------------------------- labels

_LABEL = re.compile(r"ipsi_([\d.]+?)(?:_vs_([\d.]+?))?_t(\d)$")


def label(delta: float, horizon: int, reference: float | None = None) -> str:
    """The property label of a mean (``ipsi_0.5_t3``) or a contrast (``ipsi_0.5_vs_1.0_t3``)."""
    if reference is None:
        return f"ipsi_{delta}_t{horizon}"
    return f"ipsi_{delta}_vs_{reference}_t{horizon}"


def label_parts(name: str) -> tuple[float, float | None, int]:
    """``(delta, reference, horizon)`` of a property label."""
    match = _LABEL.match(name)
    if match is None:
        raise ValueError(f"not an IPSI property label: {name!r}")
    delta, reference, horizon = match.groups()
    return float(delta), None if reference is None else float(reference), int(horizon)


def label_estimand(name: str) -> str:
    """The estimand a property label names."""
    delta, reference, horizon = label_parts(name)
    if reference is None:
        return mean_name(delta, horizon)
    return contrast_name(delta, horizon, reference)


#: The fifteen reported labels, in the order of :data:`ESTIMANDS`.
LABELS = (
    *(label(delta, horizon) for delta in DELTAS for horizon in HORIZONS),
    *(
        label(delta, horizon, REFERENCE_DELTA)
        for delta in DELTAS
        if delta != REFERENCE_DELTA
        for horizon in HORIZONS
    ),
)
#: The labels at the last horizon: three means and two contrasts.  C3, C5 and C6 gate these.
GATED_LABELS = tuple(name for name in LABELS if name.endswith("_t3"))
#: The ten labels at the earlier horizons, which the C1 arm gates beside :data:`GATED_LABELS`.
EARLIER_LABELS = tuple(name for name in LABELS if not name.endswith("_t3"))

# ---------------------------------------------------------------------------- truths


def truths() -> dict[str, float]:
    """Every primary parameter: the forward composition of the law under the tilt."""
    mean = {
        (delta, horizon): law.BINARY.tilt_counterfactual(delta, horizon)
        for delta in DELTAS
        for horizon in HORIZONS
    }
    out = {mean_name(delta, horizon): value for (delta, horizon), value in mean.items()}
    for delta in DELTAS:
        if delta == REFERENCE_DELTA:
            continue
        for horizon in HORIZONS:
            out[contrast_name(delta, horizon)] = (
                mean[delta, horizon] - mean[REFERENCE_DELTA, horizon]
            )
    return out


TRUTH = truths()

# ---------------------------------------------------------------------------- the record

#: The property families and their arms.  Each arm is a suffix of ``label__arm``.
CALIBRATION_ARMS = ("in_sample", "crossfit", "continuous")
BOOSTER_ARM = "crossfit_boosted"
SMALL_SAMPLE_ARMS = ("in_sample", "continuous")
#: ``single_robustness``: the positive arm and its control, a plug-in without cleverly.
SINGLE_ROBUSTNESS_ARMS = ("w_only_m", "one_step_necessity")
SINGLE_ROBUSTNESS_LABELS = (
    label(0.5, 3),
    label(2.0, 3),
    label(0.5, 3, REFERENCE_DELTA),
    label(2.0, 3, REFERENCE_DELTA),
)
ABSORBING_ARMS = ("repeated", "absorbing_reading")
ABSORBING_LABELS = tuple(label(delta, horizon) for delta in DELTAS for horizon in (2, 3))
HISTORY_FREE_ARMS = ("repeated", "history_free")
HISTORY_FREE_LABELS = (label(0.5, 3), label(2.0, 3))
JOINT_LABEL = "all_reported"


def _property_cells() -> dict[str, tuple[str, ...]]:
    """The property cells ``ipsi_repeated_properties`` publishes, by family."""
    return {
        "interval_calibration": (
            *(f"{name}__in_sample" for name in LABELS),
            *(f"{name}__crossfit" for name in GATED_LABELS),
            *(f"{name}__continuous" for name in GATED_LABELS),
            *(
                f"{name}__{control}"
                for name in LABELS
                for control in ("shrunken_se_control", "noise_control")
            ),
        ),
        "booster_calibration": tuple(f"{name}__{BOOSTER_ARM}" for name in GATED_LABELS),
        "small_sample_calibration": tuple(
            f"{name}__{arm}" for name in GATED_LABELS for arm in SMALL_SAMPLE_ARMS
        ),
        "simultaneous_coverage": tuple(
            f"{JOINT_LABEL}__{kind}" for kind in ("simultaneous_band", "pointwise_joint_control")
        ),
        "single_robustness": tuple(
            f"{name}__{arm}" for name in SINGLE_ROBUSTNESS_LABELS for arm in SINGLE_ROBUSTNESS_ARMS
        ),
        "repeated_outcome_necessity": tuple(
            f"{name}__{arm}" for name in ABSORBING_LABELS for arm in ABSORBING_ARMS
        ),
        "outcome_history_necessity": tuple(
            f"{name}__{arm}" for name in HISTORY_FREE_LABELS for arm in HISTORY_FREE_ARMS
        ),
    }


STUDY = StudyRecord(
    name="repeated-outcome longitudinal IPSI",
    slug="longitudinal-repeated-ipsi",
    outcome_kind="repeated",
    artifacts=ROOT / "tests" / "canonical" / "ipsi_repeated",
    document="docs/technical-reference/method-evidence/repeated-outcome-ipsi.md",
    anchor="repeated-outcome-longitudinal-ipsi",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    modules=(
        "tests/studies/canonical_ipsi_repeated.py",
        "tests/studies/ipsi_repeated_properties.py",
        "tests/studies/repeated_sizing.py",
        # The law, the learners and the helpers the two modules import.
        "tests/studies/ltmle_repeated_law.py",
        "tests/studies/ltmle_repeated_properties.py",
        "tests/studies/canonical_ltmle_repeated.py",
        "tests/studies/canonical_ltmle.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_ipsi_repeated",
    properties_module="tests.studies.ipsi_repeated_properties",
    property_cells=_property_cells(),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)


class PlaceholderPilotsError(RuntimeError):
    """A declared run was requested while the pilot tables are placeholders."""


def require_recorded_pilots(what: str) -> None:
    """Refuse ``what`` while :data:`PILOT_STATE` is ``"placeholder"``.

    A declared run draws the sizes this module and the properties module state.  While the pilot
    tables are declared uniform planning values, those sizes are not the rules applied to a
    pilot, and a run under them would publish verdicts that no pilot sized.
    """
    if PILOT_STATE != "recorded":
        raise PlaceholderPilotsError(
            f"{what} is refused: pilots P4, P5 and P6 have not been run, so "
            f"{STUDY.slug} holds placeholder pilot tables "
            "(tests/canonical/ipsi_repeated/README.md gives the pilot commands)"
        )


# ---------------------------------------------------------------------------- the fit


def fit_cleverly(frame: pd.DataFrame, *, folds: int, simultaneous: bool = False) -> Any:
    """The primary fit: ``IPSI`` with the declared learners on the repeated outcome."""
    return IPSI(
        plans(),
        reference=REFERENCE,
        outcome_learner=CellMeansBackoff(),
        pseudo_learner=CellMeansBackoff(),
        treatment_learner=QuasiBinomialGLM(),
        censoring_learner=QuasiBinomialGLM(),
        n_folds=folds,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        random_state=0,
    ).fit(frame, **COLUMNS)


#: The primary fit, recorded in the manifest.
CONFIGURATION = {
    "construction": "ipsi",
    "outcome_kind": "repeated",
    "horizon_mode": "all_prefixes",
    "estimator": "one_step",
    "deltas": list(DELTAS),
    "reference": REFERENCE,
    "scenario_folds": dict(SCENARIO_FOLDS),
    "outcome_learner": "CellMeansBackoff",
    "pseudo_learner": "CellMeansBackoff",
    "mechanism_learner": "QuasiBinomialGLM",
    "g_bounds": list(G_BOUNDS),
    "horizons": list(HORIZONS),
    "labels": list(LABELS),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIO_FOLDS:
        raise KeyError(scenario)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario not in SCENARIO_FOLDS:
        raise KeyError(scenario)
    result = fit_cleverly(frame, folds=SCENARIO_FOLDS[scenario])
    return regimen_rows(
        STUDY,
        result,
        truth,
        dict.fromkeys(ESTIMANDS, math.nan),
        ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication.

    The declared ``(replicates, n)`` is refused while the pilot tables are placeholders.  An
    explicit smaller run is a smoke run and is not refused.
    """
    if (replicates, n) == (PRIMARY_REPLICATES, PRIMARY_N):
        require_recorded_pilots("the declared primary run")
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )
