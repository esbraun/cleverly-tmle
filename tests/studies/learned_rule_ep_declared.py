"""Declaration and driver of the ``learned-rule-ep-declared`` study (P1 plan rev 2.1, section 5.2).

The subject is the learned-rule CV-TMLE of ``learned-rule-ep`` with the EP rule fitted from a
declared sieve or a declared final learner, the two settings that read experimental before P1
PR 1:

- ``S``, the declared sieve: ``LearnedRule(loss="ep", modifiers=MODIFIERS,
  sieve=Sieve(Additive(BSpline(["W1"], 4), Indicators("W2")), size=4))``, a family other than
  the default and a fixed size, so no size is chosen by cross-validation.  Its EP limit is the
  projection of the effect onto that member (:func:`limit_scores`);
- ``L``, the declared final learner: the default sieve with its size chosen by cross-validation
  and ``learner=HistGradientBoostingRegressor(max_depth=2, learning_rate=0.1, max_iter=100,
  early_stopping=False)``, a greedy learner inside the size step.  No limit is claimed.

The laws are the RM30 laws ``non_exceptional`` and ``misspecified_limit``, and the optimal value
at ``non_exceptional`` with the known mechanism (``optimal_known_g``).  Outcome and treatment
learners, ten outer folds, ``cv_evaluation=True``, pooled targeting and ``g_bounds="auto"`` are
the shipped study's; truth per replicate is rule L4 (each fold's rule refitted through
``fit_fold_rule``), so ``truth_varies_by_replicate``.

The calibration ``R`` needs the 1,400-draw pilots per cell, which have not run;
:data:`BUDGETED_R` is the plan's budget, which the declaration commit replaces by the computed
``R`` when larger.  The study runs after Fix-3 PR 2 merges (plan section 7).  It is not
registered: it joins :func:`tests.studies.evidence.registry.registered` with its declared run.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cleverly.bases import Additive, BSpline, Indicators, Sieve
from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_ep_law as ep_law
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_optimal_law as optimal_law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "learned-rule-ep-declared"
#: The record seeds, beside ``learned-rule-ep``'s and on no other study's stream.
SEED = 20261491
RESAMPLING_SEED = 20261492

MODIFIERS = ep_law.MODIFIERS
ESTIMAND = ep_law.ESTIMAND
OPTIMAL_ESTIMAND = ep_law.OPTIMAL_ESTIMAND

#: The declared sieve's family and its fixed size (its largest member).
SIEVE_FAMILY = Additive(BSpline(["W1"], 4), Indicators("W2"))
SIEVE_SIZE = 4

#: The two declared curves.
CURVES = ("S", "L")

#: The laws of the calibration cells, and the label of the optimal-value cell.
LAWS = ep_law.LAWS
OPTIMAL_LABEL = "optimal_known_g"
CELL_LAWS = {"nonexc": "non_exceptional", "misspec": "misspecified_limit", "optimal": OPTIMAL_LABEL}

CALIBRATION_N = 16_000
PRIMARY_N = 2_000
PRIMARY_REPLICATES = 1_000
CONTROL_REPLICATES = 500
#: The plan's budgeted calibration ``R`` and pilot draws per cell (5.2).
BUDGETED_R = 4_800
PILOT_DRAWS = 1_400
SHRUNKEN_SE_FACTOR = 0.70


def declared_rule(curve: str, *, random_state: int = 0) -> LearnedRule:
    """The EP rule of ``curve``: ``"S"`` (declared sieve) or ``"L"`` (declared learner)."""
    if curve == "S":
        return LearnedRule(
            loss="ep", modifiers=MODIFIERS, sieve=Sieve(SIEVE_FAMILY, size=SIEVE_SIZE)
        )
    if curve == "L":
        learner = HistGradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.1,
            max_iter=100,
            early_stopping=False,
            random_state=random_state,
        )
        return LearnedRule(loss="ep", modifiers=MODIFIERS, learner=learner)
    raise KeyError(curve)


# ---------------------------------------------------------------------------- the cells


def calibration_cell(cell_law: str, curve: str) -> str:
    """The ``interval_calibration`` positive cell of one law and curve (``label__arm``)."""
    return f"{cell_law}__{curve}"


PRIMARY_CELLS = ("S_nonexc_2k", "S_misspec_2k", "L_nonexc_2k", "L_misspec_2k")
CONTROL_CELLS = ("S_shrunk", "L_fold_leak")

PROPERTY_CELLS = {
    "interval_calibration": (
        *(calibration_cell(cell_law, curve) for curve in CURVES for cell_law in CELL_LAWS),
        *CONTROL_CELLS,
    ),
}

#: The calibration arms: one per declared curve, three laws each (plan 5.2).
CALIBRATION_ARMS = {
    f"__{curve}": tuple(calibration_cell(cell_law, curve) for cell_law in CELL_LAWS)
    for curve in CURVES
}

SCENARIOS = dict.fromkeys(PRIMARY_CELLS, (ESTIMAND,))

STUDY = StudyRecord(
    name="Learned-rule value, EP rule with a declared sieve or learner",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_ep_declared",
    document="docs/technical-reference/method-evidence/learned-rule-ep-declared.md",
    anchor="learned-rule-value-ep-rule-with-a-declared-sieve-or-learner",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-ep",
    reference=None,
    failure_bound={"cleverly-learned-rule-ep": 0.01},
    modules=(
        "tests/studies/learned_rule_ep_declared.py",
        "tests/studies/_learned_rule_ep_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/_learned_rule_optimal_law.py",
        "src/cleverly/interventions/_rule_learners.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/estimators/ep.py",
        "src/cleverly/effect_modification.py",
        "src/cleverly/bases.py",
        "src/cleverly/estimators/tmle.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_ep_declared",
    properties_module="tests.studies.learned_rule_ep_declared",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    truth_varies_by_replicate=True,
)


# ----------------------------------------------------------------------------- the fits


def scenario_of(cell: str) -> tuple[str, str]:
    """``(curve, law)`` of a primary cell such as ``"S_nonexc_2k"``."""
    curve, cell_law, _ = cell.split("_")
    return curve, CELL_LAWS[cell_law]


def measured(
    frame: pd.DataFrame, curve: str, law: str, fold_seed: int, *, mutation: bool = False
) -> tuple[Any, base.Truth, int]:
    """Fit ``curve`` at ``law`` on one sample and run the shipped harness (rule L4)."""
    rule = declared_rule(curve, random_state=fold_seed)
    return base.measured(
        frame, base.outcome_learner(), law, fold_seed, mutation=mutation, rule=rule
    )


def optimal_measured(frame: pd.DataFrame, curve: str, fold_seed: int) -> tuple[Any, float, Any]:
    """The optimal value at ``non_exceptional`` with ``g0`` supplied, fitted with ``curve``."""
    estimator = TMLE(
        learned_rule=declared_rule(curve, random_state=fold_seed),
        estimands=[optimal_law.TARGET],
        n_folds=base.N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=base.outcome_learner(),
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(base.COVARIATES),
        treatment_probabilities=optimal_law.known_probabilities(frame),
    ).single()
    refit = optimal_law.harness(result, base.outcome_learner(), "non_exceptional")
    return result, optimal_law.law_limit("non_exceptional")["truth"], refit


def _row(cell: str, replicate: int, n: int, truth: float, estimate: Any) -> dict[str, Any]:
    std_error, low, high = reported_inference(estimate)
    return {
        "implementation": STUDY.implementation,
        "scenario": cell,
        "replicate": replicate,
        "n": n,
        "estimand": ESTIMAND,
        "truth": truth,
        "estimate": float(estimate.psi),
        "inference_estimate": float(estimate.psi),
        "std_error": std_error,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": float("nan"),
    }


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    _, law = scenario_of(scenario)
    return base.draw(law, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    cell, replicate, n = key
    curve, law = scenario_of(cell)
    fold_seed = stream_seed(STUDY, "fold_partition", "primary", cell, replicate)
    result, truth, _ = measured(frame, curve, law, fold_seed)
    return [_row(cell, replicate, n, truth.truth, result.estimates[ESTIMAND])]


def _sample(key: tuple[str, int, int]) -> pd.DataFrame:
    cell, replicate, n = key
    return base.draw(scenario_of(cell)[1], n, replicate_seed(STUDY, cell, replicate))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replication."""
    outcomes = replication.fit_primary(
        [(cell, replicate, n) for cell in STUDY.scenarios for replicate in range(replicates)],
        _primary,
        draw=_sample,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])


def calibration_row(cell: str, replicate: int, *, n: int = CALIBRATION_N) -> list[dict[str, Any]]:
    """One ``interval_calibration`` replication of a calibration cell, with its control.

    ``nonexc__S`` also writes the ``S_shrunk`` control (the SE times 0.70 on the same fit).
    """
    cell_law, curve = cell.split("__")
    law = CELL_LAWS[cell_law]
    seed = stream_seed(STUDY, "property_sample", cell, replicate)
    fold_seed = stream_seed(STUDY, "fold_partition", "property", cell, replicate)
    if law == OPTIMAL_LABEL:
        frame = base.draw("non_exceptional", n, seed)
        result, truth, _ = optimal_measured(frame, curve, fold_seed)
        estimate = result.estimates[OPTIMAL_ESTIMAND]
    else:
        frame = base.draw(law, n, seed)
        result, harness_truth, _ = measured(frame, curve, law, fold_seed)
        estimate, truth = result.estimates[ESTIMAND], harness_truth.truth
    std_error, low, high = reported_inference(estimate)
    out = [
        {
            "property": "interval_calibration",
            "cell": cell,
            "role": "positive",
            "replicate": replicate,
            "n": n,
            "requested_replicates": BUDGETED_R,
            "failed_replicates": 0,
            "truth": truth,
            "estimate": float(estimate.psi),
            "std_error": std_error,
            "covered": int(low <= truth <= high),
            "rejected": int(not low <= 0.0 <= high),
        }
    ]
    if cell == calibration_cell("nonexc", "S"):
        half = SHRUNKEN_SE_FACTOR * (high - low) / 2.0
        middle = (high + low) / 2.0
        out.append(
            {
                **out[0],
                "cell": "S_shrunk",
                "role": "control",
                "std_error": std_error * SHRUNKEN_SE_FACTOR,
                "covered": int(middle - half <= truth <= middle + half),
            }
        )
    return out


# --------------------------------------------------------------------------- the limits


def _member_design(w2: float) -> np.ndarray:
    """The declared sieve's member on the grid at ``W2 = w2``."""
    from cleverly.estimators.ep import sieve_member

    member = sieve_member(SIEVE_FAMILY, SIEVE_SIZE, MODIFIERS)
    values = {"W1": base.GRID, "W2": np.full(base.GRID.size, w2)}
    return np.asarray(member.evaluate(values, MODIFIERS), dtype=float)


@cache
def limit_scores(law: str) -> dict[float, np.ndarray]:
    """The ``S`` limit curve on the grid: ``tau`` projected onto the declared member under ``P_0``.

    ``optimal_known_g`` reads ``non_exceptional``'s effect.
    """
    law = "non_exceptional" if law == OPTIMAL_LABEL else law
    tau = ep_law._effect(law)
    mass = ep_law._mass() / len(base.W2_LEVELS)
    design = np.vstack([_member_design(w2) for w2 in base.W2_LEVELS])
    target = np.concatenate([tau[w2] for w2 in base.W2_LEVELS])
    weight = np.sqrt(np.concatenate([mass for _ in base.W2_LEVELS]))
    beta = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
    fitted = design @ beta
    size = base.GRID.size
    return {w2: fitted[i * size : (i + 1) * size] for i, w2 in enumerate(base.W2_LEVELS)}


def limit_value(law: str) -> tuple[float, float]:
    """The value and the row-curve variance of the ``S`` limit rule at ``law``."""
    law = "non_exceptional" if law == OPTIMAL_LABEL else law
    return base.rule_value({w2: s > 0.0 for w2, s in limit_scores(law).items()}, law)


def regret(law: str) -> float:
    """``Psi_{d0} - Psi_{d_S}``: what the ``S`` limit rule gives up against ``d0``."""
    law = "non_exceptional" if law == OPTIMAL_LABEL else law
    optimal = {w2: effect > 0.0 for w2, effect in ep_law._effect(law).items()}
    return base.rule_value(optimal, law)[0] - limit_value(law)[0]


def thresholds(law: str) -> dict[float, tuple[float, ...]]:
    """The values of ``W1`` at which the ``S`` limit score changes sign, for each ``W2``."""
    out = {}
    for w2, half in limit_scores(law).items():
        crossings = np.flatnonzero(np.diff(np.sign(half)) != 0)
        out[w2] = tuple(
            float(
                base.GRID[i] - half[i] * (base.GRID[i + 1] - base.GRID[i]) / (half[i + 1] - half[i])
            )
            for i in crossings
        )
    return out


def design_numbers() -> dict[str, Any]:
    """The design numbers that need no pilot: the ``S`` limits, regrets and efficiency bounds."""
    out: dict[str, Any] = {"arms": {arm: len(cells) for arm, cells in CALIBRATION_ARMS.items()}}
    for law in (*LAWS, OPTIMAL_LABEL):
        value, variance = limit_value(law)
        out[law] = {
            "limit_value": value,
            "efficiency_sd": float(np.sqrt(variance)),
            "regret": regret(law),
            "thresholds": thresholds(law),
        }
    return out


# ------------------------------------------------------------- the properties interface

FAILURE_BOUND = 0.01


def fold_leak_row(replicate: int) -> list[dict[str, Any]]:
    """One replication of the ``L_fold_leak`` control at ``n = 2,000`` (must fail coverage).

    Each fold's ``L`` curve is fitted on its own validation rows, the shipped
    ``validation_row_rule`` mutation, at ``non_exceptional``.
    """
    seed = stream_seed(STUDY, "property_sample", "L_fold_leak", replicate)
    fold_seed = stream_seed(STUDY, "fold_partition", "property", "L_fold_leak", replicate)
    frame = base.draw("non_exceptional", PRIMARY_N, seed)
    result, truth, _ = measured(frame, "L", "non_exceptional", fold_seed, mutation=True)
    std_error, low, high = reported_inference(result.estimates[ESTIMAND])
    return [
        {
            "property": "interval_calibration",
            "cell": "L_fold_leak",
            "role": "control",
            "replicate": replicate,
            "n": PRIMARY_N,
            "requested_replicates": CONTROL_REPLICATES,
            "failed_replicates": 0,
            "truth": truth.truth,
            "estimate": float(result.estimates[ESTIMAND].psi),
            "std_error": std_error,
            "covered": int(low <= truth.truth <= high),
            "rejected": int(not low <= 0.0 <= high),
        }
    ]


def _property_replicate(key: tuple[str, int], _drawn: Any) -> list[dict[str, Any]]:
    cell, replicate = key
    return fold_leak_row(replicate) if cell == "L_fold_leak" else calibration_row(cell, replicate)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every calibration and control replication; ``budget`` caps each cell."""
    replicates = BUDGETED_R if budget is None else min(budget, BUDGETED_R)
    leak = CONTROL_REPLICATES if budget is None else min(budget, CONTROL_REPLICATES)
    keys = [
        *(
            (cell, r)
            for cells in CALIBRATION_ARMS.values()
            for cell in cells
            for r in range(replicates)
        ),
        *(("L_fold_leak", r) for r in range(leak)),
    ]
    outcomes = replication.fit_properties(
        keys,
        _property_replicate,
        draw=lambda key: None,
        cells_of=lambda key: [f"interval_calibration/{key[0]}"],
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        n_jobs=n_jobs,
        seed_of=lambda key: stream_seed(STUDY, "property_sample", key[0], key[1]),
    )
    return pd.DataFrame([row for records in outcomes for row in records])


__all__ = [
    "CALIBRATION_ARMS",
    "PROPERTY_CELLS",
    "STUDY",
    "declared_rule",
    "design_numbers",
    "draw_and_fit",
    "generate_property_rows",
]
