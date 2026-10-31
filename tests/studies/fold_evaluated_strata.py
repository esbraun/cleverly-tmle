"""Registered evidence for fold-evaluated CV-TMLE with baseline strata (X28, X11 PR E1).

The subject is ``TMLE(cross_fit=True, n_folds=5, cv_evaluation=True, estimands=("ate", "ey"),
g_bounds="auto", q_bounds=(0, 1), simultaneous=False)`` fitted with ``strata=["S"]`` on complete data with rows
as the unit, and ``repeats=1``. Each stratum estimate is the ``1/V`` average of the fold
plug-ins over the rows of each validation fold in the stratum, each fold's stratum curve divides
by the fold's own stratum share, and the marginal estimate is the share mixture of the stratum
estimates with its delta-method curve. Each fit reports 12 parameters: ``ate``, ``ey[0]`` and
``ey[1]``, in each of the three strata and marginally.

``unweighted``
    Rows only.
``weighted``
    The known observation weight ``w = 1 + 1.5 W2``, which varies inside every stratum. The
    target is the parameter of the weight-tilted law.

Both settings draw the law of :mod:`tests.studies._fold_evaluated_strata_law`: the RM30 family
with the blip ``0.1 + 1.5 W1``, a beta outcome of concentration 12 around the RM30 outcome
regression, and a stratum cut from ``W1`` at shares 0.2, 0.35 and 0.45. The
target is fixed, so every row's truth is the quadrature value of its parameter. The X11 plan
(``X11-plan.md`` rev 2.1 in the plans repository, sections 2.5 and 3), the study design inputs of
``reviews/x11/X11-E1-impl-review.md`` and the study design record
``reviews/x11/X11-E1-study-design.md`` declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _fold_evaluated_strata_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 1_000
SEED = 20263051
RESAMPLING_SEED = 20263052

SCENARIOS = dict.fromkeys(law.SETTINGS, law.PARAMETERS)

#: The calibration labels: one per setting and parameter, ``<setting>_<target>_<stratum>``.
CALIBRATION_LABELS = tuple(
    f"{setting}_{label}" for setting in law.SETTINGS for label in law.PARAMETER_LABELS
)

#: The SD of ``sqrt(n)`` times each calibration label's estimate: the efficient influence curve
#: SD of its parameter by the quadrature of :func:`~tests.studies._fold_evaluated_strata_law.
#: moments`. They size the noise controls only, at one unit each, and no verdict reads an
#: efficiency ratio (``calibration_efficiency_ratio=False``). ``test_fold_evaluated_strata_design``
#: recomputes them.
EFFICIENCY_BOUNDS = {
    "unweighted_ate_s0": 0.551891,
    "unweighted_ate_s1": 0.476225,
    "unweighted_ate_s2": 0.388963,
    "unweighted_ate_all": 0.311064,
    "unweighted_ey0_s0": 0.408456,
    "unweighted_ey0_s1": 0.325921,
    "unweighted_ey0_s2": 0.300958,
    "unweighted_ey0_all": 0.205782,
    "unweighted_ey1_s0": 0.388732,
    "unweighted_ey1_s1": 0.376414,
    "unweighted_ey1_s2": 0.278199,
    "unweighted_ey1_all": 0.301336,
    "weighted_ate_s0": 0.588550,
    "weighted_ate_s1": 0.515968,
    "weighted_ate_s2": 0.429197,
    "weighted_ate_all": 0.339026,
    "weighted_ey0_s0": 0.429930,
    "weighted_ey0_s1": 0.345356,
    "weighted_ey0_s2": 0.322141,
    "weighted_ey0_all": 0.219425,
    "weighted_ey1_s0": 0.415369,
    "weighted_ey1_s1": 0.409567,
    "weighted_ey1_s2": 0.314792,
    "weighted_ey1_all": 0.330274,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The invalid-inference control of every stratum label: the stratum curve without its
#: ``1 / P_{n,v}(S = s)`` normalisation.
UNNORMALISED = "unnormalised_curve_control"

#: The invalid-inference control of the marginal ``ate`` and ``ey[1]`` labels: the marginal
#: curve without its share term. Its asymptotic SE ratio is 0.854 and 0.656 rows only, 0.854 and
#: 0.660 weighted. ``ey[0]`` carries none: its share term moves the ratio to 0.948 and 0.945 only,
#: which the declared count cannot resolve below the band floor 0.93.
SHARE_DROPPED = "share_dropped_control"

#: The marginal labels whose share-term control is declared.
SHARE_DROPPED_LABELS = tuple(
    f"{setting}_{target}_all" for setting in law.SETTINGS for target in ("ate", "ey1")
)

#: The labels of the stratum parameters, which carry :data:`UNNORMALISED`.
STRATUM_LABELS = tuple(label for label in CALIBRATION_LABELS if not label.endswith("_all"))

#: The targeting-control labels: every stratum parameter of the rows-only setting.
NECESSITY_LABELS = tuple(label for label in STRATUM_LABELS if label.startswith("unweighted_"))


def calibration_cells() -> tuple[str, ...]:
    """Every ``interval_calibration`` cell, in label order."""
    cells: list[str] = []
    for label in CALIBRATION_LABELS:
        cells.extend(
            f"{label}__{kind}" for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        )
        if label in STRATUM_LABELS:
            cells.append(f"{label}__{UNNORMALISED}")
        if label in SHARE_DROPPED_LABELS:
            cells.append(f"{label}__{SHARE_DROPPED}")
    return tuple(cells)


PROPERTY_CELLS = {
    "interval_calibration": calibration_cells(),
    "targeting_necessity": tuple(
        f"{label}__{arm}" for label in NECESSITY_LABELS for arm in ("targeted", "untargeted")
    ),
}

STUDY = StudyRecord(
    name="Fold-evaluated CV-TMLE with baseline strata",
    slug="fold-evaluated-strata",
    artifacts=ROOT / "tests" / "canonical" / "fold_evaluated_strata",
    document="docs/technical-reference/method-evidence/fold-evaluated-strata.md",
    anchor="fold-evaluated-cv-tmle-with-baseline-strata",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-fold-evaluated-strata",
    reference=None,
    failure_bound={"cleverly-fold-evaluated-strata": 0.01},
    modules=(
        "tests/studies/fold_evaluated_strata.py",
        "tests/studies/fold_evaluated_strata_properties.py",
        "tests/studies/_fold_evaluated_strata_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/_strata.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/learners/crossfit.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.fold_evaluated_strata",
    properties_module="tests.studies.fold_evaluated_strata_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    "fit": (
        "TMLE(outcome_learner=<outcome learner>, treatment_learner=<treatment learner>, "
        "cross_fit=True, n_folds=5, cv_evaluation=True, targeting_scheme='pooled', "
        "g_bounds='auto', q_bounds=(0.0, 1.0), estimands=('ate', 'ey'), simultaneous=False, "
        "random_state=<fold seed>).fit(frame, outcome='Y', treatment='A', "
        "covariates=['W1', 'W2', 'S'], strata=['S'], weights=<'w' in the weighted setting>); "
        "n = 1000, rows only, complete data, repeats=1"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "QuasiBinomialGLM(tol=1e-10)) on (A, W1, W2, S)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2, S)",
    "targeting_control_learner": (
        "RM30's drop_w1_learner() with the quasibinomial solver: columns 0 and 2 of "
        "(A, W1, W2, S), so A and W2 only, and their product"
    ),
    "law": (
        "W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, Y | A, W ~ "
        "Beta(12 m, 12 (1 - m)) with m = Qbar0(A, W), logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + "
        "a (0.1 + 1.5 W1); S = 0, 1, 2 on W1 in [-1, -0.6), [-0.6, 0.1), "
        "[0.1, 1] (shares 0.2, 0.35, 0.45); weighted setting w = 1 + 1.5 W2"
    ),
    "truth": (
        "fixed: each stratum mean E_w[f(W) | S = s] by the trapezoid rule on 4001 points of W1 "
        "per stratum for each W2, weighted by w(W2); the marginal truth is the share mixture"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, with the truth of every parameter."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), law.truths(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """One primary replication: one row per reported parameter."""
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, fold_seed)
    truth = law.truths(scenario)
    rows = []
    for label in law.PARAMETER_LABELS:
        name = law.parameter(label)
        estimate = fitted.result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": truth[name],
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= truth[name] <= high),
                "initial_estimate": law.untargeted(fitted.result, label),
            }
        )
    return rows


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``."""
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for scenario in record.scenarios
            for replicate in range(replicates)
        ],
        partial(primary_rows, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
