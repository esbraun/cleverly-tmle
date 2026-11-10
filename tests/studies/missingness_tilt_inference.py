r"""Registered evidence for the targeted missingness tilt's intervals (A2, CDR-001).

The subject is :func:`cleverly.sensitivity.missingness_tilt` off :math:`\gamma = 0`: at each
declared :math:`\gamma` the package re-targets the tilted functional
:math:`\psi_\gamma = E[\pi_a \bar Q_a + (1 - \pi_a)\operatorname{expit}(\operatorname{logit}
\bar Q_a + \gamma)]` of every reported arm mean, contrast, ``att``, ratio, ``ey_obs`` and ``par``
from the fit's own nuisances (:mod:`cleverly.sensitivity._tilt_targeting`), and reports the
targeted estimate with a Wald interval from the tilted influence curve.  Each replicate fits
once and reads ``missingness_tilt(result, GAMMAS, estimands=[...])``.  The law is
:mod:`tests.studies.missingness_tilt_law`: a binary outcome missing at random given
``(A, W)``, with truths computed by quadrature from the observed-data law alone.

**Cells.**  Every scenario reports each of its targets at :math:`\gamma \in \{-1, 1, 2\}`; an
estimand is named ``<target>[gamma=<g>]``.  The rule column is the
:data:`~cleverly.sensitivity.missingness.TILT_INTERVAL_INSTRUMENT` rule that
``_tilt_reading`` computes for the fit, which a later change binds to these cells.

=================  ======  =====================================  ============================
scenario           n       construction                           rules it licenses
=================  ======  =====================================  ============================
``in_sample``      1,000   ``TMLE(cross_fit=False)``;             ``level``, ``contrast``,
                           ey1, ey0, ate, att, rr                 ``conditional``, ``ratio``
``in_sample_4k``   4,000   the same fit, the n ladder             the same rules
``cross_fit``      1,000   stacked CV-TMLE at the package's       ``<kind>+cross_fit`` for
                           default 10 folds; ey1, ey0, ate,       the four kinds
                           att, rr
``cv_evaluation``  1,000   ``cv_evaluation=True``, 10 folds;      ``level+cv_evaluation``,
                           ey1, ey0, ate, att                     ``contrast+cv_evaluation``,
                                                                  ``conditional+cv_evaluation``
``weighted``       1,000   in sample, ``weights="wt"`` with       ``<kind>+weighted`` for the
                           ``wt = exp(0.5 W1)``; ey1, ey0, ate,   four kinds
                           att, rr
``natural_course`` 1,000   in sample; ey_obs, ey0, par            ``natural_course``, ``par``
                                                                  (and ``level`` again)
=================  ======  =====================================  ============================

``ratio`` is read on the log scale, where the package forms its interval.  ``att`` under
``cv_evaluation`` is covered; ``rr`` is not, because the fold-evaluated report forms no ratio.
The weight of ``weighted`` reads ``W1``, so its truths are those of the weight-tilted covariate
law (:func:`tests.studies.missingness_tilt_law.truths` with ``weighted=True``), which differ
from the unweighted truths by more than half an SD at n = 1,000 for ``ey1`` and ``ate``: a fit
that dropped its weights would miss them.

**What the cells test.**  At :math:`\gamma = 2` every target's truth moves from its MAR value by
at least half its n = 1,000 SD (the design test checks it against the pilot spread), so those
cells test that the tilt is applied and targeted.  At :math:`\gamma = -1` the movement is small
for the contrasts (``ate`` moves 0.07 SD), so those cells test the validity of the interval
off :math:`\gamma = 0`, not that the tilt was applied.

**License scope.**  A licensed rule covers the configurations measured here: parametric GLM
nuisance learners (the declared logistic regressions, correctly specified), n of at least
1,000, a binary treatment and outcome, one repeat, and the package's default fold count where
the fit is cross-fitted.  It says nothing about flexible outcome learners, whose remainder the
tilted parameter does not protect: it is not doubly robust in the outcome regression.

**Learners** (declared here, not the package defaults).  Outcome: the degree-two
interaction expansion of the design (which holds the law's ``A W1`` term, so the model is
correctly specified), ``StandardScaler`` and ``LogisticRegression(C=1e4)``.  Treatment and
response mechanisms: main-terms ``LogisticRegression(C=1e4)``, correctly specified.

**Initial estimate.**  ``initial_estimate`` is the untargeted tilted plug-in of the same fit,
``cleverly.sensitivity.missingness._tilted_value`` on the MAR-targeted regression, combined over
the fit's repeats by the median on the inference scale (one repeat here), as the A1-era tilt
reported it.

**Size** (declared before any run, from :mod:`tests.studies.missingness_tilt_pilot`).
:data:`PRIMARY_REPLICATES` is :func:`~tests.studies.repeated_sizing.primary_replicates` on the
full pilot's 2,000 draws of every scenario, with one implementation, one bias group per scenario
from the standardized errors on the inference scale, and both shares at
:data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`: 8,400.  No fit was lost in the
pilot's 12,000 fits.  The lowest pilot coverages are all in ``weighted`` at
:math:`\gamma = -1` (``ey0`` 0.9215, one-sided 80% lower bound 0.916; ``att`` 0.926; ``rr``
0.927), with SE ratios down to 0.94.  Every other scenario covers between 0.932 and 0.965, with
SE ratios between 0.95 and 1.05 and no standardized bias above 0.11, so the weighted cells bind
the size.  The design test recomputes it.

**Properties.** :mod:`tests.studies.missingness_tilt_properties` declares the calibration
family at :math:`\gamma = 2` with its two derived controls and the A1-era interval as a third
control, and the targeting-necessity family with a main-terms outcome regression.

**Comparator.** None.  No published implementation reports this targeted tilt.

**Not covered.** Flexible outcome learners, repeated cross-fitting (``+repeats``), clustered
rows, a declared ``arm_gamma`` direction, multi-arm and continuous outcomes, ``or`` and
``paf``, regime means and MSM coefficients (which keep the descriptive plug-in spread), and
collaborative fits.  A fit whose rule no registered study covers reads descriptive.

The red-cell readings are declared in ``tests/canonical/missingness_tilt_inference/regenerate.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cleverly.estimators import TMLE
from cleverly.sensitivity.missingness import (
    _tilt_direction,
    _tilt_targets,
    _tilted_value,
    missingness_tilt,
)
from tests.parallel import STUDY_JOBS
from tests.studies import missingness_tilt_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: The shared primary rule on the full pilot (see the module docstring).
PRIMARY_REPLICATES = 8_400
PRIMARY_N = 1_000
LADDER_N = 4_000
FAILURE_BOUND = 0.01
SEED = 2026111001
RESAMPLING_SEED = 2026111002

GAMMAS = (-1.0, 1.0, 2.0)
#: The large C every parametric learner declares: an unpenalized fit in all but name.
UNPENALIZED_C = 1e4

LINEAR_TARGETS = ("ey1", "ey0", "ate", "att")
RATIO_TARGETS = (*LINEAR_TARGETS, "rr")
NATURAL_TARGETS = ("ey_obs", "ey0", "par")

IN_SAMPLE = "in_sample"
LADDER = "in_sample_4k"
CROSS_FIT = "cross_fit"
CV_EVALUATION = "cv_evaluation"
WEIGHTED = "weighted"
NATURAL_COURSE = "natural_course"


def estimand_name(target: str, gamma: float) -> str:
    """The published name of one target at one tilt."""
    return f"{target}[gamma={gamma:g}]"


# ---------------------------------------------------------------------------- learners


def outcome_learner() -> Pipeline:
    """The declared outcome regression: logistic on the design's pairwise interactions."""
    return make_pipeline(
        PolynomialFeatures(2, interaction_only=True, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=UNPENALIZED_C, max_iter=5000),
    )


def mechanism_learner() -> LogisticRegression:
    """The declared treatment and response mechanisms: main-terms logistic regression."""
    return LogisticRegression(C=UNPENALIZED_C, max_iter=5000)


# ---------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    """One primary construction.

    Parameters
    ----------
    targets : tuple of str
        The parameters the fit reports and the tilt reads.
    cross_fit : bool
        Whether the fit is cross-fitted (the package's default fold count).
    cv_evaluation : bool
        Whether the cross-fitted report is fold-evaluated.
    weighted : bool
        Whether the fit reads the observation weights ``wt``.
    """

    targets: tuple[str, ...]
    cross_fit: bool = False
    cv_evaluation: bool = False
    weighted: bool = False

    @property
    def estimands(self) -> tuple[str, ...]:
        """Every published name, gamma-major as the tilt frame orders its rows."""
        return tuple(estimand_name(t, g) for g in GAMMAS for t in self.targets)


SCENARIOS: dict[str, Scenario] = {
    IN_SAMPLE: Scenario(RATIO_TARGETS),
    LADDER: Scenario(RATIO_TARGETS),
    CROSS_FIT: Scenario(RATIO_TARGETS, cross_fit=True),
    CV_EVALUATION: Scenario(LINEAR_TARGETS, cross_fit=True, cv_evaluation=True),
    WEIGHTED: Scenario(RATIO_TARGETS, weighted=True),
    NATURAL_COURSE: Scenario(NATURAL_TARGETS),
}
SCENARIO_N = {LADDER: LADDER_N}
ESTIMANDS = {scenario: spec.estimands for scenario, spec in SCENARIOS.items()}
TRUTHS_BY_GAMMA = {gamma: law.truths(gamma) for gamma in GAMMAS}
WEIGHTED_TRUTHS_BY_GAMMA = {gamma: law.truths(gamma, weighted=True) for gamma in GAMMAS}
TRUTH: dict[str, dict[str, float]] = {
    scenario: {
        estimand_name(t, g): (WEIGHTED_TRUTHS_BY_GAMMA if spec.weighted else TRUTHS_BY_GAMMA)[g][t]
        for g in GAMMAS
        for t in spec.targets
    }
    for scenario, spec in SCENARIOS.items()
}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "ate_gamma2__correctly_specified",
        "ate_gamma2__shrunken_se_control",
        "ate_gamma2__noise_control",
        "ate_gamma2__mar_variance_control",
    ),
    "targeting_necessity": ("ate_gamma2__targeted", "ate_gamma2__untargeted"),
}

STUDY = StudyRecord(
    name="targeted missingness tilt intervals",
    slug="missingness-tilt-inference",
    artifacts=ROOT / "tests" / "canonical" / "missingness_tilt_inference",
    document="docs/technical-reference/method-evidence/missingness-tilt.md",
    anchor="targeted-missingness-tilt-intervals",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    scenario_n=dict(SCENARIO_N),
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-missingness-tilt",
    reference=None,
    failure_bound={"cleverly-missingness-tilt": FAILURE_BOUND},
    modules=(
        "tests/studies/missingness_tilt_inference.py",
        "tests/studies/missingness_tilt_properties.py",
        "tests/studies/missingness_tilt_law.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.missingness_tilt_inference",
    properties_module="tests.studies.missingness_tilt_properties",
    property_cells=PROPERTY_CELLS,
)

CONFIGURATION = {
    "construction": "targeted missingness tilt off gamma = 0, read by missingness_tilt",
    "gammas": list(GAMMAS),
    "n_folds": "package default (10)",
    "scenario_settings": {
        scenario: {
            "targets": list(spec.targets),
            "n": STUDY.sizes(PRIMARY_N)[scenario],
            "cross_fit": spec.cross_fit,
            "cv_evaluation": spec.cv_evaluation,
            "weighted": spec.weighted,
        }
        for scenario, spec in SCENARIOS.items()
    },
    "learners": (
        "outcome: PolynomialFeatures(2, interaction_only) + StandardScaler + "
        "LogisticRegression(C=1e4); treatment and response: LogisticRegression(C=1e4) on "
        "main terms"
    ),
    "weights": f"exp({law.WEIGHT_SLOPE} W1); truths of the weight-tilted covariate law",
    "initial_estimate": "the untargeted tilted plug-in (_tilted_value) of the same fit",
    "variance_method": "ic",
}


def estimator(scenario: str, **overrides: Any) -> TMLE:
    """The declared estimator of ``scenario``; ``overrides`` replace a setting."""
    spec = SCENARIOS[scenario]
    settings: dict[str, Any] = {
        "estimands": spec.targets,
        "outcome_learner": outcome_learner(),
        "treatment_learner": mechanism_learner(),
        "missingness_learner": mechanism_learner(),
        "cross_fit": spec.cross_fit,
        "random_state": 0,
    }
    if spec.cross_fit:
        settings["cv_evaluation"] = spec.cv_evaluation
    settings.update(overrides)
    return TMLE(**settings)


def fit(frame: pd.DataFrame, scenario: str, **overrides: Any) -> Any:
    """The one fit of a primary scenario."""
    spec = SCENARIOS[scenario]
    return (
        estimator(scenario, **overrides)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            weights="wt" if spec.weighted else None,
        )
        .single()
    )


def tilt(result: Any, targets: tuple[str, ...], gammas: tuple[float, ...] = GAMMAS) -> Any:
    """The tilt every replicate reads."""
    return missingness_tilt(result, list(gammas), estimands=list(targets))


def initial_estimates(
    result: Any, targets: tuple[str, ...], gammas: tuple[float, ...] = GAMMAS
) -> dict[str, float]:
    """Each estimand's untargeted tilted plug-in on its natural scale (the A1-era point).

    The median over the fit's repeats on the inference scale, as the A1-era tilt combined them.
    """
    tiltable = _tilt_targets(result)
    direction = _tilt_direction(result.data, None)
    out = {}
    for gamma in gammas:
        for name in targets:
            target = tiltable[name]
            center = float(
                np.median(
                    [
                        _tilted_value(result, repeat, target, gamma, direction)
                        for repeat in result.repeats
                    ]
                )
            )
            out[estimand_name(name, gamma)] = (
                float(np.exp(center)) if target.kind == "ratio" else center
            )
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law.sample(n, seed, weighted=SCENARIOS[scenario].weighted), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def tilt_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    *,
    scenario: str,
    replicate: int,
    n: int,
    initial: Mapping[str, float],
    implementation: str = STUDY.implementation,
) -> list[dict[str, Any]]:
    """One row per ``(target, gamma)`` of a tilt frame, in the replicate schema.

    A ratio's interval is formed on the log scale, so its ``inference_estimate`` and
    ``std_error`` are log-scale and its endpoints are on the ratio scale.
    """
    rows = []
    for record in frame.to_dict("records"):
        if bool(record["is_mar"]):
            continue
        name = estimand_name(str(record["estimand"]), float(record["gamma"]))
        std_error = float(record.get("std_err", float("nan")))
        if not np.isfinite(std_error):
            raise AssertionError(f"{name} carries no finite targeted standard error")
        psi = float(record["psi"])
        ratio = str(record["estimand"]) == "rr"
        low, high = float(record["ci_lower"]), float(record["ci_upper"])
        reference = float(truth[name])
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": psi,
                "inference_estimate": float(np.log(psi)) if ratio else psi,
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial[name]),
            }
        )
    return rows


def replicate_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """Fit one sample and read its tilt."""
    targets = SCENARIOS[scenario].targets
    result = fit(frame, scenario)
    table = tilt(result, targets).to_frame()
    return tilt_rows(
        table,
        truth,
        scenario=scenario,
        replicate=replicate,
        n=n,
        initial=initial_estimates(result, targets),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    frame, truth = drawn
    return replicate_rows(frame, truth, scenario, replicate, n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row, each scenario at its declared size.  No sample is kept."""
    sizes = STUDY.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in SCENARIOS
            for replicate in range(replicates)
        ],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
