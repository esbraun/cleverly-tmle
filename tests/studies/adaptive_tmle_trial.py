r"""Registered evidence for adaptive TMLE of a trial with external data (X90 T3, study A).

The subject is ``CausalStudy(..., TrialWithExternalData).identify(TrialATE).estimate(
method=AdaptiveTMLEMethod(...))`` with explicit, fast learners (additive spline regressions,
:mod:`tests.studies._atmle_laws`), five outer folds and five inner penalty folds. Every
scenario draws ``n = 4,000`` pooled rows: the plan's ``n_rct = 1,000`` and ``n_rwd = 3,000``
("3x"), with ``S ~ Bernoulli(0.25)`` independent of ``W`` on the App B laws, which the paper
omits.

Each scenario is one law, form and population, and one *gated* estimand:

``psi``
    The trial effect :math:`\Psi`. The interval is for the working-model projection, so this
    gate also reads the oracle-approximation condition (arXiv:2405.07186v2, Theorem 1) on laws
    inside the dictionary's span: ``a0``, ``a1a``, ``a1b`` (bias linear in ``W``), ``a3``,
    ``a3_pi``, ``a4``, ``a5`` and the pooled-form cells ``f1a``, ``f1b``.
``projection``
    The working-model projection :math:`\Psi_w(P_0)` of the fit's own stored dictionary, by
    quadrature on the stored fold transformers. ``a2c``, ``a2d`` and ``e1`` gate this; their
    :math:`\Psi` coverage is reported with the projection bias beside it (plan B1).
``psi_2``
    The trial's own effect :math:`\Psi_2`, ``h2_trial``.
``b1y``
    A reported stress row, declared in :data:`REPORTED` and not gated. The method's guarantee
    is pointwise (arXiv:2405.07186v2, Theorem 1 and Appendix C Theorem 2; arXiv:2307.12544v2,
    Theorems 5 to 7): the interval covers :math:`\Psi` only under the oracle-approximation
    condition, and a bias that is not in the span of the linear-plus-hinge dictionary (the
    external bias on this law passes through ``expit``) leaves the condition unmet. The two
    external arms' biases differ by about one standard error of that difference
    (:math:`\Psi^\# = -0.0148`, about 1.3 standard errors), so at ``n = 4,000`` the lasso
    usually keeps only the constant and the projection misses :math:`\Psi` by about
    :math:`\Psi^\#`. No uniformly valid interval can be much shorter than the robust one at
    zero difference (Hodges and Le Cam super-efficiency; Leeb and Potscher on post-selection
    inference; Armstrong and Kolesar, 2018, on adaptive intervals), so the row measures the
    cost and gates neither the coverage of :math:`\Psi`, nor the bias, nor the standard error.
    It reports :math:`\Psi` coverage, projection coverage, bias and the SE ratio.

Every row also carries both other truths and the coverage of each, so the reported rows are
computed from the fit that gated. The record declares ``truth_varies_by_replicate``, because
the projection truth moves with each replication's selection.

Controls ride on the host scenarios' draws and fit the same rows again: the trial-only TMLE
beside ``a0``, ``a1a``, ``a1b`` (the efficiency contrast), the naive pooled TMLE on ``a1b``
(K1), the unpenalized main-terms working model on ``e1`` (K2), the Pi-fluctuation mutation on
``a3_pi`` (K3, plan fallback), and the pooled-population fit on ``h2_pooled`` read against
:math:`\Psi_2` (K5). K4 is the property family of :mod:`tests.studies.adaptive_tmle_trial_properties`.

The R ``atmle`` comparator (``845280889d2fb7d929adb494759b1588ab776f43``) is declared in
:data:`COMPARATOR` and not registered here: a record whose truth varies by replication cannot
declare a pending reference, so the comparison runs as its own pairing record on constant truths
(the ``adaptive-iptw-pairing`` pattern), locally, and never blocks this study.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cache, partial
from typing import Any

import numpy as np
import pandas as pd

from cleverly.exceptions import CapabilityError, DataError
from tests.parallel import STUDY_JOBS
from tests.studies import _atmle_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.inference import bootstrap, clopper_pearson, percentile_interval
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The record seeds. Neither equals a seed of any registered or declared study, checked on the
#: seeds alone in ``tests/unit/test_adaptive_tmle_trial_design.py``.
SEED = 20290311
RESAMPLING_SEED = 20290312

PRIMARY_N = 4_000
#: Replications of every scenario: ``repeated_sizing.primary_replicates`` at the pilot that
#: ``test_adaptive_tmle_trial_design`` pins (``tests/canonical/adaptive_tmle_trial/pilot.py``).
PRIMARY_REPLICATES = 2_800

IMPLEMENTATION = "cleverly-adaptive-tmle"

COMPARATOR = {
    "name": "atmle",
    "commit": "845280889d2fb7d929adb494759b1588ab776f43",
    "scope": "form='pooled' on the constant-truth cells a0, a1a, a1b, a3, f1a, f1b",
    "comparison": "matching implementation",
    "state": (
        "pending: a pairing record on constant truths, run locally; the runner writes its basis "
        "matrix, folds and nuisance predictions and the Python side refits from them"
    ),
    "pins": {
        "target_method": "relaxed",
        "n_lambda": 1,
        "force_A": False,
        "folds": "passed from Python",
        "nuisance_predictions": "passed in (atmle predicts in sample with one learner)",
    },
    "declared_differences": (
        "two-epsilon weighted Pi step (target_gwt = TRUE) against one epsilon; denominator "
        "bound and Pi clip; dropped basis columns with support below 1/sqrt(n); stopping rule "
        "1e-4 sd/(sqrt(n) log n) against sd/(sqrt(n) log n)"
    ),
}


@dataclass(frozen=True)
class Cell:
    """One scenario: a law, the engine settings and the estimand that gates."""

    law: str
    estimand: str
    form: str = "cv"
    population: str = "pooled"
    learners: str = "spline"
    controls: tuple[str, ...] = ()
    #: ``"default"`` is the shipped dictionary and lasso; ``"k2"`` is the unpenalized
    #: main-terms working model of control K2 and of the K4 ladder.
    fit: str = "default"


CELLS: dict[str, Cell] = {
    "a0": Cell("a0", "psi", controls=("comparator",)),
    "a1a": Cell("a1a", "psi", controls=("comparator",)),
    "a1b": Cell("a1b", "psi", controls=("comparator", "k1")),
    "a2c": Cell("a2c", "projection"),
    "a2d": Cell("a2d", "projection"),
    "e1": Cell("e1", "projection", controls=("k2",)),
    "a3": Cell("a3", "psi"),
    "a3_pi": Cell("a3", "psi", learners="pi_misspecified", controls=("k3",)),
    "a4": Cell("a4", "psi"),
    "a5": Cell("a5", "psi"),
    "b1y": Cell("b1y", "psi"),
    "e1_k2_n1000": Cell("e1", "projection", fit="k2"),
    "e1_k2_n2000": Cell("e1", "projection", fit="k2"),
    "h2_pooled": Cell("h2", "psi", controls=("k5",)),
    "h2_trial": Cell("h2", "psi_2", population="trial"),
    "f1a": Cell("a1a", "psi", form="pooled"),
    "f1b": Cell("a1b", "psi", form="pooled"),
}

SCENARIOS = {name: (cell.estimand,) for name, cell in CELLS.items()}

#: Reported stress rows (``StudyRecord.reported_scenarios``): published, never gated.
REPORTED = frozenset({"b1y"})

#: K4, the ladder on E1: the K2 working model at three sizes ending at the study's ``n``
#: (the last is the ``k2`` control columns of ``e1``). The only scenarios off ``PRIMARY_N``.
LADDER_N = {"e1_k2_n1000": 1_000, "e1_k2_n2000": 2_000}

#: The efficiency contrast of plan section 5 (m7): ``RMSE(A-TMLE) / RMSE(trial-only TMLE) < 1``
#: at the 99% bootstrap upper bound. It gates on ``a0``, where the external rows carry no bias
#: and the bias model's oracle selection is its constant: at that selection the influence
#: function's SD is 0.48 of the trial-only TMLE's at ``P_0`` (the design test computes it). On
#: ``a1a`` and ``a1b`` a gain needs the lasso to leave out every column that gives the external
#: arms a free mean shift (``A`` and the ``A`` hinges, whose means are not zero). Nothing
#: guarantees that selection at this ``n``, and the T3 review's 200-replicate draw read ratios
#: 1.05 and 1.14, so those two rows report the ratio and do not gate.
EFFICIENCY_CELLS = ("a0",)
EFFICIENCY_REPORTED = ("a1a", "a1b")

#: The extra columns of every primary row.
EXTRA_COLUMNS = (
    "psi_truth",
    "psi_2_truth",
    "projection_truth",
    "covered_psi",
    "covered_psi_2",
    "covered_projection",
    "converged",
    "passes",
    "fit_seconds",
    "truth_seconds",
    "cell_seconds",
    "comparator_estimate",
    "comparator_std_error",
    "control_estimate",
    "control_std_error",
    "control_truth",
    "control_ci_lower",
    "control_ci_upper",
)

CONTROL_COLUMNS = EXTRA_COLUMNS[-7:]  # comparator and control columns

REPORTED_ARTIFACT = "reported.csv"
CONTROLS_ARTIFACT = "controls.csv"

STUDY = StudyRecord(
    name="Adaptive TMLE with external data",
    slug="adaptive-tmle-trial",
    artifacts=ROOT / "tests" / "canonical" / "adaptive_tmle_trial",
    document="docs/technical-reference/method-evidence/adaptive-tmle-trial.md",
    anchor="adaptive-tmle-with-external-data",
    scenarios=SCENARIOS,
    scenario_n=LADDER_N,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    modules=(
        "tests/studies/adaptive_tmle_trial.py",
        "tests/studies/adaptive_tmle_trial_properties.py",
        "tests/studies/_atmle_laws.py",
        "src/cleverly/estimators/atmle.py",
        "src/cleverly/estimators/atmle_result.py",
        "src/cleverly/estimators/_working_basis.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.adaptive_tmle_trial",
    properties_module="tests.studies.adaptive_tmle_trial_properties",
    property_cells={},
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    reported_scenarios=REPORTED,
    extra_artifacts=(REPORTED_ARTIFACT, CONTROLS_ARTIFACT),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "CausalStudy(frame, TrialWithExternalData(outcome='Y', treatment='A', adjustment=W, "
        "source='S')).identify(TrialATE(population)).estimate(method=AdaptiveTMLEMethod(form, "
        "models=ModelSpec(spline learners), cross_fitting=CrossFitting(n_folds=5, "
        "learner_folds=5), targeting=Targeting(g_bounds=(0.01, 0.99)))) at n = 4000; default "
        "linear-plus-hinge dictionary; only the constant of each working model is unpenalized"
    ),
    "learners": (
        "SplineRegressor/SplineClassifier: additive cubic splines (5 knots) of W, pairwise "
        "products z_i z_j, z_i^2 z_j, z_i z_j^2, fully interacted with the arm (and trial "
        "indicator); ridge (alpha = 1) least squares or logistic (C = 10) on the standardized "
        "features. With missing outcomes (a5): Q(S, W, A) by least squares and the observation "
        "model by logistic regression (C = 1e4), each linear in W within the trial-by-arm cells"
    ),
    "cells": {name: vars(cell) for name, cell in CELLS.items()},
    "truth": (
        "psi and psi_2: tensor Gauss-Legendre panels (5 nodes) broken at law thresholds; "
        "projection: the fold's stored transformer applied to nodes broken at its knots, "
        "weighted Gram systems with the true g, Pi, tau; the CV form averages folds"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', scenario, replicate)",
}


@cache
def exact(law_name: str) -> dict[str, float]:
    """``psi``, ``psi_2`` and ``p`` of a law, by quadrature."""
    return laws.exact_targets(laws.LAWS[law_name])


def _learners(cell: Cell, law: laws.Law) -> dict[str, Any]:
    if cell.learners == "flexible":
        return laws.flexible_learners(law)
    if cell.learners == "main_terms":
        return laws.main_terms_learners(law)
    learners = laws.spline_learners(law)
    if cell.learners == "pi_misspecified":
        # K3 fallback (plan): only the Pi learner is misspecified; g and Qt stay correct.
        from sklearn.linear_model import LogisticRegression

        learners["source"] = LogisticRegression(max_iter=2000)
    return learners


def _covered(low: float, high: float, truth: float) -> int:
    return int(low <= truth <= high)


def fit_cell(
    frame: pd.DataFrame, scenario: str, seed: int, cells: dict[str, Cell] | None = None
) -> dict[str, Any]:
    """Fit one scenario's sample and return the row values (every column of the row)."""
    cell = (CELLS if cells is None else cells)[scenario]
    law = laws.LAWS[cell.law]
    learners = _learners(cell, law)
    begin = time.perf_counter()
    start = begin
    k2 = cell.fit == "k2"
    try:
        result = laws.fit_trial(
            frame,
            law,
            form=cell.form,
            population=cell.population,
            seed=seed,
            learners=learners,
            basis=laws.MainTerms() if k2 else None,
            mutation=laws.Mutation(unpenalized=True) if k2 else None,
        )
    except CapabilityError as error:
        # A replicate the method refuses (the training-fold rank check of plan 2.5 step 3, the
        # lasso's optimality tolerance) counts as failed and non-covering, and is reported
        # (plan M2). The driver records a DataError; a CapabilityError would stop the run.
        raise DataError(f"adaptive TMLE refused this replicate: {error}") from error
    fit_seconds = time.perf_counter() - start
    estimate = result.estimate
    std_error, low, high = reported_inference(estimate)
    start = time.perf_counter()
    projection = laws.projection_truth(law, result.working_models, cell.population)
    truth_seconds = time.perf_counter() - start
    targets = exact(cell.law)
    truths = {"psi": targets["psi"], "psi_2": targets["psi_2"], "projection": projection}
    out: dict[str, Any] = {
        "estimand": cell.estimand,
        "truth": truths[cell.estimand],
        "estimate": float(estimate.psi),
        "std_error": std_error,
        "ci_lower": low,
        "ci_upper": high,
        "psi_truth": truths["psi"],
        "psi_2_truth": truths["psi_2"],
        "projection_truth": projection,
        "covered_psi": _covered(low, high, truths["psi"]),
        "covered_psi_2": _covered(low, high, truths["psi_2"]),
        "covered_projection": _covered(low, high, projection),
        "converged": int(result.fit.converged),
        "passes": int(result.fit.passes),
        "fit_seconds": fit_seconds,
        "truth_seconds": truth_seconds,
        "cell_seconds": np.nan,
    }
    out.update(dict.fromkeys(CONTROL_COLUMNS, np.nan))
    if "comparator" in cell.controls:
        estimate_t, se_t = laws.fit_point(frame, law, seed=seed, trial_only=True)
        out["comparator_estimate"], out["comparator_std_error"] = estimate_t, se_t
    control = _control(cell, law, frame, seed, targets, result, std_error, low, high)
    out.update(control)
    out["cell_seconds"] = time.perf_counter() - begin
    return out


def _control(
    cell: Cell,
    law: laws.Law,
    frame: pd.DataFrame,
    seed: int,
    targets: dict[str, float],
    result: Any,
    std_error: float,
    low: float,
    high: float,
) -> dict[str, float]:
    """The control columns of a host scenario (K1, K2, K3, K5)."""
    if "k1" in cell.controls:
        point, se = laws.fit_point(frame, law, seed=seed)
        return _control_columns(point, se, targets["psi"])
    if "k2" in cell.controls:
        fitted = laws.fit_trial(
            frame,
            law,
            seed=seed,
            learners=laws.spline_learners(law),
            basis=laws.MainTerms(),
            mutation=laws.Mutation(unpenalized=True),
        )
        se, lo, hi = reported_inference(fitted.estimate)
        truth = laws.projection_truth(law, fitted.working_models, "pooled")
        return _control_columns(float(fitted.estimate.psi), se, truth, lo, hi)
    if "k3" in cell.controls:
        fitted = laws.fit_trial(
            frame,
            law,
            seed=seed,
            learners=_learners(cell, law),
            mutation=laws.Mutation(no_source_step=True),
        )
        se, lo, hi = reported_inference(fitted.estimate)
        return _control_columns(float(fitted.estimate.psi), se, targets["psi"], lo, hi)
    if "k6" in cell.controls:
        # K6: the same law and form with main-terms GLM nuisances (Karim 2026's default).
        main = Cell(
            cell.law,
            cell.estimand,
            form=cell.form,
            population=cell.population,
            learners="main_terms",
        )
        fitted = laws.fit_trial(
            frame,
            law,
            form=cell.form,
            population=cell.population,
            seed=seed,
            learners=_learners(main, law),
        )
        se, lo, hi = reported_inference(fitted.estimate)
        return _control_columns(float(fitted.estimate.psi), se, targets["psi"], lo, hi)
    if "k5" in cell.controls:
        # K5: the pooled-population fit (S/p replaced by 1) read against Psi_2.
        return _control_columns(float(result.estimate.psi), std_error, targets["psi_2"], low, high)
    return {}


def _control_columns(
    estimate: float,
    std_error: float,
    truth: float,
    low: float | None = None,
    high: float | None = None,
) -> dict[str, float]:
    z = 1.959963984540054
    return {
        "control_estimate": estimate,
        "control_std_error": std_error,
        "control_truth": truth,
        "control_ci_lower": estimate - z * std_error if low is None else low,
        "control_ci_upper": estimate + z * std_error if high is None else high,
    }


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truths depend on the fit, so only the
    constant ones are returned."""
    if scenario not in CELLS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    law = laws.LAWS[CELLS[scenario].law]
    targets = exact(CELLS[scenario].law)
    return laws.draw(law, n, seed), {"psi": targets["psi"], "psi_2": targets["psi_2"]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord,
    key: tuple[str, int, int],
    frame: pd.DataFrame,
    cells: dict[str, Cell] | None = None,
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    seed = stream_seed(record, "fold_partition", scenario, replicate)
    values = fit_cell(frame, scenario, seed, cells)
    row = {
        "implementation": record.implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": n,
        "inference_estimate": values["estimate"],
        "inference_scale": "identity",
        "covered": _covered(values["ci_lower"], values["ci_upper"], values["truth"]),
        "initial_estimate": values["estimate"],
        **values,
    }
    return [row]


def _sample(
    record: StudyRecord, key: tuple[str, int, int], cells: dict[str, Cell] | None = None
) -> pd.DataFrame:
    scenario, replicate, size = key
    law = laws.LAWS[(CELLS if cells is None else cells)[scenario].law]
    return laws.draw(law, size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    record: StudyRecord = STUDY,
    cells: dict[str, Cell] | None = None,
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``.

    ``cells`` is the scenario table of the record: this study's, or the sibling selective
    study's, which shares every function here.
    """
    sizes = record.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for replicate in range(replicates)
            for scenario in record.scenarios
        ],
        partial(_primary, record, cells=cells),
        draw=partial(_sample, record, cells=cells),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
        cell_of=lambda key: key[0],
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]])


# ------------------------------------------------------------------------------------------
# reported and control artifacts, and the gates that read them


def _coverage(values: pd.Series, level: float) -> dict[str, float]:
    count, total = int(values.sum()), len(values)
    interval = clopper_pearson(count, total, confidence_level=level)
    return {
        "coverage": count / total,
        "coverage_lower": interval.low,
        "coverage_upper": interval.high,
    }


def reported_table(
    rows: pd.DataFrame, record: StudyRecord = STUDY, cells: dict[str, Cell] | None = None
) -> pd.DataFrame:
    """Per scenario: the coverage of each of the three targets, beside the gated one.

    ``psi`` coverage on a scenario that gates the projection is a *reported* row, labelled by
    ``projection_bias`` (the mean of ``psi_truth - projection_truth``, in the scenario's
    empirical SDs of the estimate).
    """
    level = record.margins.confidence_level
    records = []
    for scenario, group in rows.groupby("scenario", sort=True):
        cell = (CELLS if cells is None else cells)[str(scenario)]
        sd = float(group["estimate"].std(ddof=1))
        bias = float((group["psi_truth"] - group["projection_truth"]).mean())
        entry: dict[str, Any] = {
            "scenario": scenario,
            "gated": cell.estimand,
            "replicates": len(group),
            "mean_psi_truth": float(group["psi_truth"].mean()),
            "mean_projection_truth": float(group["projection_truth"].mean()),
            "projection_bias_sd": bias / sd,
            "converged_share": float(group["converged"].mean()),
            "mean_fit_seconds": float(group["fit_seconds"].mean()),
            "mean_truth_seconds": float(group["truth_seconds"].mean()),
        }
        for target in ("psi", "psi_2", "projection"):
            summary = _coverage(group[f"covered_{target}"], level)
            entry.update({f"{target}_{key}": value for key, value in summary.items()})
        records.append(entry)
    return pd.DataFrame.from_records(records)


def efficiency_row(
    group: pd.DataFrame, record: StudyRecord, seed: int, role: str = "positive"
) -> dict[str, Any]:
    """RMSE(A-TMLE) / RMSE(trial-only TMLE) against ``psi``, with its bootstrap bounds.

    ``role`` ``"positive"`` passes when the upper bound is below one; ``"reporting"`` always
    passes.
    """
    truth = group["psi_truth"].to_numpy()
    arrays = {
        "a": group["estimate"].to_numpy() - truth,
        "t": group["comparator_estimate"].to_numpy() - truth,
    }
    samples = bootstrap(
        arrays,
        {
            "ratio": lambda draw: np.sqrt(
                (draw["a"] ** 2).mean(axis=1) / (draw["t"] ** 2).mean(axis=1)
            )
        },
        replicates=record.margins.bootstrap_replicates,
        seed=seed,
    )["ratio"]
    interval = percentile_interval(samples, confidence_level=record.margins.confidence_level)
    ratio = float(np.sqrt((arrays["a"] ** 2).mean() / (arrays["t"] ** 2).mean()))
    return {
        "control": "efficiency",
        "scenario": str(group["scenario"].iloc[0]),
        "role": role,
        "statistic": "rmse_ratio",
        "value": ratio,
        "lower": interval.low,
        "upper": interval.high,
        "passed": bool(role == "reporting" or interval.high < 1.0),
    }


def coverage_row(
    group: pd.DataFrame, record: StudyRecord, control: str, role: str, *, target: str
) -> dict[str, Any]:
    """Coverage of ``control_truth`` by the control's own interval, with its exact interval.

    ``role`` ``"control"`` passes when the interval lies wholly below the coverage floor;
    ``"positive"`` when its lower bound is at or above it; ``"reporting"`` always passes (the row
    is a measurement whose power the pilot could not establish).
    """
    covered = (
        (group["control_ci_lower"] <= group["control_truth"])
        & (group["control_truth"] <= group["control_ci_upper"])
        if target == "control"
        else group[f"covered_{target}"].astype(bool)
    )
    summary = _coverage(covered.astype(int), record.margins.confidence_level)
    floor = record.margins.coverage_floor
    if role == "reporting":
        passed = True
    elif role == "control":
        passed = summary["coverage_upper"] < floor
    else:
        passed = summary["coverage_lower"] >= floor
    return {
        "control": control,
        "scenario": str(group["scenario"].iloc[0]),
        "role": role,
        "statistic": "coverage",
        "value": summary["coverage"],
        "lower": summary["coverage_lower"],
        "upper": summary["coverage_upper"],
        "passed": bool(passed),
    }


def controls_table(rows: pd.DataFrame, record: StudyRecord = STUDY) -> pd.DataFrame:
    """The verdict of every control and the efficiency contrast, from the primary rows."""
    by = dict(iter(rows.groupby("scenario")))
    out = [
        efficiency_row(
            by[name],
            record,
            stream_seed(record, "efficiency_bootstrap", name),
            "positive" if name in EFFICIENCY_CELLS else "reporting",
        )
        for name in (*EFFICIENCY_CELLS, *EFFICIENCY_REPORTED)
        if name in by
    ]
    if "a1b" in by:  # K1: the naive pooled TMLE must fail Psi coverage.
        out.append(coverage_row(by["a1b"], record, "k1_naive_pooled", "control", target="control"))
    if "e1" in by:  # K2: its own projection passes; Psi fails.
        out.append(
            coverage_row(by["e1"], record, "k2_main_terms_projection", "positive", target="control")
        )
        k2 = by["e1"].assign(control_truth=by["e1"]["psi_truth"])
        out.append(coverage_row(k2, record, "k2_main_terms_psi", "control", target="control"))
    if "a3_pi" in by:  # K3 fallback: the targeted fit passes, the mutation fails.
        out.append(coverage_row(by["a3_pi"], record, "k3_targeted", "positive", target="psi"))
        out.append(coverage_row(by["a3_pi"], record, "k3_no_pi_step", "control", target="control"))
    if "h2_pooled" in by:  # K5: the pooled-population interval must miss Psi_2.
        out.append(
            coverage_row(
                by["h2_pooled"], record, "k5_pooled_for_psi_2", "control", target="control"
            )
        )
    out.extend(ladder_rows(rows))
    return pd.DataFrame.from_records(out)


def ladder_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """K4: ``|Psi bias| / SE`` and ``|projection bias| / SE`` of the K2 model at three sizes.

    A reporting row (``passed`` is true by declaration): the claim is that the first grows like
    ``sqrt(n)`` while the second stays bounded, and the table prints both beside each ``n``.
    """
    out = []
    sizes = [("e1_k2_n1000", 1_000), ("e1_k2_n2000", 2_000)]
    for name, n in sizes:
        group = rows.loc[rows["scenario"] == name]
        if group.empty:
            continue
        out.append(_ladder(group, n, group["estimate"], group["truth"]))
    host = rows.loc[rows["scenario"] == "e1"]
    if not host.empty:
        out.append(
            _ladder(
                host,
                PRIMARY_N,
                host["control_estimate"],
                host["control_truth"],
                host["control_std_error"],
            )
        )
    return out


def _ladder(
    group: pd.DataFrame,
    n: int,
    estimate: pd.Series,
    projection: pd.Series,
    std_error: pd.Series | None = None,
) -> dict[str, Any]:
    se = float((group["std_error"] if std_error is None else std_error).mean())
    return {
        "control": "k4_ladder",
        "scenario": f"e1_k2_n{n}",
        "role": "reporting",
        "statistic": "psi_bias_over_se",
        "value": float(abs((estimate - group["psi_truth"]).mean()) / se),
        "lower": float(abs((estimate - projection).mean()) / se),
        "upper": float(n),
        "passed": True,
    }


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The reported coverage table and the control verdicts."""
    return {
        REPORTED_ARTIFACT: reported_table(rows),
        CONTROLS_ARTIFACT: controls_table(rows),
    }


def scientific_failures(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The gates the driver reads: every control and the efficiency contrast must pass."""
    controls = frames[CONTROLS_ARTIFACT]
    failed = controls.loc[~controls["passed"].astype(bool)]
    return {"control and efficiency verdicts": failed}
