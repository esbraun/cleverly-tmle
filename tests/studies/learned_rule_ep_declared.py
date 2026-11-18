"""Declaration and driver of the ``learned-rule-ep-declared`` study (P1 plan rev 2.2, section 5.2).

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

The calibration ``R`` needs the 1,400-draw pilots per cell
(:mod:`tests.studies.learned_rule_ep_declared_pilot`), which have not run;
:data:`BUDGETED_R` is the plan's budget, which the declaration commit replaces by the computed
``R`` when larger.  The study runs after Fix-3 PR 2 merges (plan section 7).  It is not
registered: it joins :func:`tests.studies.evidence.registry.registered` with its declared run.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import cleverly.interventions._rule_learners as rule_learners
from cleverly.bases import Additive, BSpline, Indicators, Sieve
from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from cleverly.interventions.base import RegimeSet
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
#: Design review S6: 2,000 per primary cell, so a correct estimator at the shipped pilot's
#: coverage 0.930 passes each cell's coverage floor with probability 0.986 and all four with
#: 0.945 (``tests/unit/test_learned_rule_ep_declared_design.py``).
PRIMARY_REPLICATES = 2_000
CONTROL_REPLICATES = 500
#: The diagnostic ``optimal__L`` cell's draws: it states no verdict, so it is sized as the
#: shipped study's reporting cells are, not as a gated arm.
DIAGNOSTIC_REPLICATES = 2_400
#: The budgeted calibration ``R`` of each arm (plan 5.2): design review S3 raised ``L`` to the
#: shipped sibling's assured count, 7,200, and design review 2 S-b raised ``S`` to match, where a
#: cell at a true SE ratio of 0.97 passes with probability 0.991 rather than 0.928.  The declaration commit replaces either by the
#: pilot's computed ``R`` when larger.
BUDGETED_R = {"S": 7_200, "L": 7_200}
PILOT_DRAWS = 1_400
SHRUNKEN_SE_FACTOR = 0.70
#: The harness refits each fold's rule through ``fit_fold_rule`` and compares it with the rule
#: the fit recorded on every grid point and validation row on the pilot draws and on every
#: replicate whose index is a multiple of this (design review B4); every other replicate
#: integrates the recorded rules.
VERIFY_EVERY = 25


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

#: The SE times 0.70 on the ``nonexc__S`` fits, the shared shrunken control.
SHRUNKEN_CONTROL = "nonexc_S__shrunken_se_control"
#: Each fold's ``L`` curve fitted on its own validation rows at n = 2,000, the shared
#: validation-rule coverage control the shipped study used.
LEAK_CONTROL = "nonexc_L__validation_rule_control"
CONTROL_CELLS = (SHRUNKEN_CONTROL, LEAK_CONTROL)

#: Design review S4: ``optimal__L``'s truth is the value of ``d0``, so it passes only if the
#: greedy ``L`` fold rules have negligible regret, which no claim covers ("No limit is claimed",
#: plan 5.2).  It publishes its coverage, SE ratio and per-replicate regret under the diagnostic
#: role and states no verdict, as the shipped study reports a regret-driven ``optimal_known_g``
#: red.
DIAGNOSTIC_CELLS = (calibration_cell("optimal", "L"),)

POSITIVE_CELLS = tuple(
    calibration_cell(cell_law, curve)
    for curve in CURVES
    for cell_law in CELL_LAWS
    if calibration_cell(cell_law, curve) not in DIAGNOSTIC_CELLS
)

PROPERTY_CELLS = {
    "interval_calibration": (
        *(calibration_cell(cell_law, curve) for curve in CURVES for cell_law in CELL_LAWS),
        *CONTROL_CELLS,
    ),
}

#: The gated calibration arms: one per declared curve (plan 5.2).
CALIBRATION_ARMS = {
    f"__{curve}": tuple(cell for cell in POSITIVE_CELLS if cell.endswith(f"__{curve}"))
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
        "tests/studies/learned_rule_ep_declared_properties.py",
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
    properties_module="tests.studies.learned_rule_ep_declared_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    truth_varies_by_replicate=True,
)

#: The sample and fold labels of each stream.  ``property`` is the declared run.
STREAMS: dict[str, tuple[str, str]] = {
    "property": ("property_sample", "fold_partition"),
    "pilot": ("pilot_sample", "pilot_fold_partition"),
    "smoke": ("smoke_sample", "smoke_fold_partition"),
}


def sample_seed(cell: str, replicate: int, stream: str = "property") -> int:
    """The sample seed of replication ``replicate`` of a property cell on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][0], cell, replicate)


def fold_seed(cell: str, replicate: int, stream: str = "property") -> int:
    """The fold seed of replication ``replicate`` of a property cell on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][1], cell, replicate)


def verified(replicate: int, stream: str) -> bool:
    """Whether the harness refits and compares this replicate's fold rules (B4)."""
    return stream == "pilot" or replicate % VERIFY_EVERY == 0


# ------------------------------------------------------------------------- the harness


@contextlib.contextmanager
def recorded_fold_rules() -> Iterator[dict[int, Any]]:
    """Record each outer fold's fitted rule as the fit learns it (design review B4).

    The fit learns fold ``v``'s rule through ``fit_fold_rule`` with the fold's inner seed and
    keeps only its values at the rows.  The recorder keeps the fitted rule itself, so the
    harness integrates the rule the fit used without refitting it.  It patches the module the
    fit calls, in this process only.
    """
    original = rule_learners.fit_fold_rule
    records: dict[int, Any] = {}

    def recorder(*args: Any, **kwargs: Any) -> Any:
        rule = original(*args, **kwargs)
        fold = kwargs.get("fold")
        if fold is None or fold in records:
            raise base.HarnessMismatch(f"fit_fold_rule called for fold {fold} twice or unnamed")
        records[int(fold)] = rule
        return rule

    rule_learners.fit_fold_rule = recorder
    try:
        yield records
    finally:
        rule_learners.fit_fold_rule = original


def _treat(rule: Any) -> Callable[[np.ndarray], np.ndarray]:
    return lambda covariates: np.asarray(rule.treat(covariates), dtype=bool)


def integrate(
    result: Any,
    rules: Sequence[Callable[[np.ndarray], np.ndarray]],
    law: str,
) -> base.Truth:
    """Check each fold's rule against the fit's rule and integrate its value (rule L4).

    ``rules[v]`` is fold ``v``'s rule as a function of the covariates.  Every validation row of
    fold ``v`` must carry the fit's rule; the value and the row-curve variance are
    :func:`tests.studies._learned_rule_law.rule_value` on the grid.
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    fitted = base.rule_of(result)
    folds = list(result.nuisance.folds)
    if len(rules) != len(folds):
        raise base.HarnessMismatch(f"{len(rules)} recorded rules for {len(folds)} folds")
    values, variances, sizes = [], [], []
    checked = 0
    for fold, ((_, test), rule) in enumerate(zip(folds, rules, strict=True)):
        disagree = int(np.sum(rule(covariates[test]) != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the recorded rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        grid = {
            w2: rule(np.column_stack([base.GRID, np.full_like(base.GRID, w2)]))
            for w2 in base.W2_LEVELS
        }
        value, variance = base.rule_value(grid, law)
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))) / len(values))
    return base.Truth(
        truth=float(np.mean(values)),
        oracle_se=oracle,
        fold_values=tuple(values),
        rows_checked=checked,
    )


def verify(result: Any, rules: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
    """Refit every fold's rule (:func:`~tests.studies._learned_rule_law.fold_rule`) and compare.

    The refit reads the fold's recorded inner seed through the package's own code, so a seeded
    learner reproduces the recorded rule bit for bit: every grid point and every row must
    agree, or :class:`~tests.studies._learned_rule_law.HarnessMismatch` stops the replicate.
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    points = np.vstack(
        [
            covariates,
            *(np.column_stack([base.GRID, np.full_like(base.GRID, w2)]) for w2 in base.W2_LEVELS),
        ]
    )
    for fold, rule in enumerate(rules):
        refit = base.fold_rule(result, base.outcome_learner(), fold)
        disagree = int(np.sum(np.asarray(refit(points), dtype=bool) != rule(points)))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the recorded rule at {disagree} of "
                f"{len(points)} rows and grid points"
            )


@dataclass(frozen=True)
class Measured:
    """One declared fit with its harness.

    Parameters
    ----------
    result : object
        The fit.
    truth : tests.studies._learned_rule_law.Truth
        The value of the fit's fold rules at the cell's law, their oracle SE and the rows
        checked.
    warnings : int
        The solver warnings of the fit and its harness.
    verified : bool
        Whether the harness refitted every fold's rule and found it equal (:func:`verify`).
    """

    result: Any
    truth: base.Truth
    warnings: int
    verified: bool


def measured(
    frame: pd.DataFrame,
    curve: str,
    law: str,
    fold_seed: int,
    *,
    check: bool = False,
    optimal: bool = False,
) -> Measured:
    """Fit ``curve`` on one sample, integrate the fit's own fold rules, and on ``check`` refit.

    ``optimal`` fits the optimal value at ``non_exceptional`` with ``g0`` supplied, as
    :func:`tests.studies._learned_rule_ep_law.optimal_measured` does.  Rule L4 stops the run on
    a non-finite estimate or standard error and on a harness mismatch.
    """
    rule = declared_rule(curve, random_state=fold_seed)
    with base.solver_warning_count() as count:
        with recorded_fold_rules() as records:
            if optimal:
                result = (
                    optimal_estimator(curve, fold_seed)
                    .fit(
                        frame,
                        outcome="Y",
                        treatment="A",
                        covariates=list(base.COVARIATES),
                        treatment_probabilities=optimal_law.known_probabilities(frame),
                    )
                    .single()
                )
            else:
                result = base.fit(frame, base.outcome_learner(), fold_seed, rule=rule)
        rules = [_treat(records[fold]) for fold in sorted(records)]
        if check:
            verify(result, rules)
        truth = integrate(result, rules, "non_exceptional" if optimal else law)
    estimate = result.estimates[OPTIMAL_ESTIMAND if optimal else ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return Measured(result=result, truth=truth, warnings=count[0], verified=check)


def optimal_estimator(curve: str, fold_seed: int) -> TMLE:
    """The optimal value at ``non_exceptional`` with ``g0`` supplied, fitted with ``curve``."""
    return TMLE(
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


def control(frame: pd.DataFrame, fold_seed: int) -> tuple[Any, base.Truth, int]:
    """The validation-rule control: each fold's ``L`` curve fitted on its own validation rows.

    The shipped harness refits it (``source="validation"``); it runs at n = 2,000 only.
    """
    rule = declared_rule("L", random_state=fold_seed)
    return base.measured(
        frame, base.outcome_learner(), "non_exceptional", fold_seed, mutation=True, rule=rule
    )


class DeclaredLimitRule(ep_law.FixedLimitRule):
    """Study-only: every row's rule fixed at the ``S`` limit rule of :attr:`LAW`."""

    def _regimes(self, data: Any, estimates: Any, *, seed: int | None = None) -> Any:
        assert self.learned_rule is not None
        treat = ep_law.rule_at(limit_scores(self.LAW), np.asarray(data.covariates, dtype=float))
        values = np.zeros((data.n, 2, 1), dtype=float)
        values[:, 1, 0] = treat
        values[:, 0, 0] = ~treat
        return RegimeSet((self.learned_rule.name,), values, 0.0)


@cache
def _limit_kind(law: str) -> type[DeclaredLimitRule]:
    return type(f"DeclaredLimitRule_{law}", (DeclaredLimitRule,), {"LAW": law})


def paired_fit(
    frame: pd.DataFrame, law: str, fold_seed: int, result: Any, *, optimal: bool
) -> ep_law.Paired:
    """The ``S`` cell's paired fit with its rule fixed at the ``S`` limit rule (B3).

    The same estimator, sample and outer folds as the declared fit; a paired fit whose folds
    differ stops the replication.  Its truth is the limit rule's value.
    """
    extra: dict[str, Any] = {"estimands": [optimal_law.TARGET]} if optimal else {}
    arguments: dict[str, Any] = {"covariates": list(base.COVARIATES)}
    if optimal:
        arguments["treatment_probabilities"] = optimal_law.known_probabilities(frame)
    fitted = (
        _limit_kind("non_exceptional" if optimal else law)(
            learned_rule=ep_law.rule(),
            n_folds=base.N_FOLDS,
            cv_evaluation=True,
            targeting_scheme="pooled",
            g_bounds="auto",
            outcome_learner=base.outcome_learner(),
            treatment_learner=base.treatment_learner(),
            random_state=fold_seed,
            **extra,
        )
        .fit(frame, outcome="Y", treatment="A", **arguments)
        .single()
    )
    if not ep_law._same_folds(result, fitted):
        raise RuntimeError("the limit-rule fit drew other outer folds than the declared fit")
    estimate = fitted.estimates[OPTIMAL_ESTIMAND if optimal else ESTIMAND]
    return ep_law.Paired(float(estimate.psi), float(estimate.std_error), float(limit_value(law)[0]))


def d0_value(law: str) -> float:
    """``Psi_{d0}(P0)`` at ``law``: the value of the optimal rule, which regret reads."""
    law = "non_exceptional" if law == OPTIMAL_LABEL else law
    optimal = {w2: effect > 0.0 for w2, effect in ep_law._effect(law).items()}
    return base.rule_value(optimal, law)[0]


# ----------------------------------------------------------------------------- the primary


def scenario_of(cell: str) -> tuple[str, str]:
    """``(curve, law)`` of a primary cell such as ``"S_nonexc_2k"``."""
    curve, cell_law, _ = cell.split("_")
    return curve, CELL_LAWS[cell_law]


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


def primary_rows(frame: pd.DataFrame, cell: str, replicate: int, seed: int) -> list[dict[str, Any]]:
    """The primary row of one draw of ``cell``, fitted with fold seed ``seed``."""
    curve, law = scenario_of(cell)
    fitted = measured(frame, curve, law, seed, check=replicate % VERIFY_EVERY == 0)
    return [
        _row(cell, replicate, len(frame), fitted.truth.truth, fitted.result.estimates[ESTIMAND])
    ]


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    cell, replicate, _ = key
    seed = stream_seed(record, "fold_partition", "primary", cell, replicate)
    return primary_rows(frame, cell, replicate, seed)


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    cell, replicate, n = key
    return base.draw(scenario_of(cell)[1], n, replicate_seed(record, cell, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record`` (a smoke passes its throwaway one)."""
    outcomes = replication.fit_primary(
        [(cell, replicate, n) for cell in record.scenarios for replicate in range(replicates)],
        partial(_primary, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])


# ------------------------------------------------------------------------- the run's cost

#: Single-thread seconds per replicate of each cell kind, measured on this container: the
#: property cells by the property smoke (``SMOKE_TIMINGS``, three draws a cell, the first one
#: verified), the harness integrating the fit's own fold rules (design review B4) and an ``S``
#: replicate including its paired limit-rule fit; the primaries by two unverified fits a cell
#: on the ``smoke_primary`` stream.  The ``S`` and ``L`` entries are the larger of their two
#: laws.
SECONDS: dict[str, float] = {
    "S": 2.8,
    "optimal__S": 2.3,
    "L": 38.5,
    "optimal__L": 34.0,
    "control": 0.85,
    "primary_S": 0.7,
    "primary_L": 12.5,
}

#: Plan 5.0's cost rule: a cell kind whose smoke cost exceeds this multiple of its declared
#: :data:`SECONDS` runs as a second declared run.
COST_GATE = 1.5

#: The smoke timings, one row per cell (``group``, ``fits``, ``seconds_per_fit``), written by
#: ``tests/canonical/learned_rule_ep_declared/regenerate.py --property-smoke`` on the smoke
#: stream.
SMOKE_TIMINGS = ROOT / "tests" / "studies" / "learned_rule_ep_declared_smoke.csv"


def seconds_of(cell: str) -> float:
    """The declared seconds per replicate of a property cell."""
    if cell == LEAK_CONTROL:
        return SECONDS["control"]
    if cell in SECONDS:
        return SECONDS[cell]
    return SECONDS[cell.rsplit("__", 1)[-1]]


def verified_share() -> float:
    """The share of declared replicates whose harness refits every fold's rule."""
    return 1.0 / VERIFY_EVERY


def run_cpu_hours() -> float:
    """Single-thread CPU-hours of the declared run: properties, verification and primaries."""
    replicates = {cell: BUDGETED_R[cell.rsplit("__", 1)[-1]] for cell in POSITIVE_CELLS}
    replicates.update(dict.fromkeys(DIAGNOSTIC_CELLS, DIAGNOSTIC_REPLICATES))
    replicates[LEAK_CONTROL] = CONTROL_REPLICATES
    # A verified replicate refits each fold's rule once more: about one more fit of the curve.
    refit = {
        cell: 1.0 + (0.5 if cell.endswith("__L") else 0.3) * verified_share() for cell in replicates
    }
    seconds = sum(seconds_of(c) * r * refit[c] for c, r in replicates.items())
    primary = PRIMARY_REPLICATES * sum(
        SECONDS[f"primary_{scenario_of(cell)[0]}"] for cell in PRIMARY_CELLS
    )
    return (seconds + primary) / 3_600.0


def pilot_cpu_hours() -> float:
    """Single-thread CPU-hours of the 1,400-draw pilot per gated cell, every draw verified."""
    return (
        sum(
            seconds_of(cell) * PILOT_DRAWS * (1.5 if cell.endswith("__L") else 1.3)
            for cell in POSITIVE_CELLS
        )
        / 3_600.0
    )


#: Every file the declared run writes into :attr:`StudyRecord.artifacts`, which the
#: registration commit commits; the property failure files only when a property fit recorded a
#: failure.  The pilot summary (:data:`tests.studies.learned_rule_ep_declared_pilot.PILOT_CSV`)
#: is committed with the declaration, before the run.
RUN_FILES = (
    "replicates.parquet",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.parquet",
    "properties.csv",
    "manifest.json",
    "run.log",
)
CONDITIONAL_RUN_FILES = (
    "property-failures.csv",
    "property-failure-accounting.csv",
    "property-failure-witnesses.parquet",
)


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
    out: dict[str, Any] = {
        "arms": {arm: len(cells) for arm, cells in CALIBRATION_ARMS.items()},
        "cpu_hours": run_cpu_hours(),
        "pilot_cpu_hours": pilot_cpu_hours(),
    }
    for law in (*LAWS, OPTIMAL_LABEL):
        value, variance = limit_value(law)
        out[law] = {
            "limit_value": value,
            "efficiency_sd": float(np.sqrt(variance)),
            "regret": regret(law),
            "thresholds": thresholds(law),
        }
    return out


__all__ = [
    "CALIBRATION_ARMS",
    "PROPERTY_CELLS",
    "STUDY",
    "declared_rule",
    "design_numbers",
    "draw_and_fit",
    "measured",
]
