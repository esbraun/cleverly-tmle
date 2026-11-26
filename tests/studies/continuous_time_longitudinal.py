r"""Registered evidence for the continuous-time longitudinal TMLE (X6 part (e), E2).

The subject is :class:`~cleverly.survival.ContinuousLongitudinalTMLE` in its shipped default:
the default libraries, cross-fitted with 10 outer folds and selection inside 5 inner folds, and
the influence-curve variance.  The laws are those of
:mod:`tests.studies.continuous_time_longitudinal_law` (X6 plan rev 7, section 5.7): law CT and its
whole-day version CT-day, each at :math:`n = 1{,}000`, with two causes and target times at days
10, 20 and 30.

**Primary cells** (:data:`CELLS`, the study's scenarios), each gated by the primary rule of
:data:`MARGINS`: the coverage interval above 0.90 (above 0.99 reads conservative, not failed),
the SE ratio's 99% interval inside (0.8, 1.2) and the bias equivalent to zero within 0.2 SD.

* ``day_cf``: CT-day, every regime (always, never and the rule "treat when :math:`L(s) = 1`"):
  every risk (18) and every risk difference against never (12), 30 rows, of which 12 are rule
  rows (:data:`DAY_ROWS`);
* ``cont_cf``: CT, always and never: 12 risks and 6 differences, 18 rows (:data:`CONT_ROWS`).

**Secondary fits** (:data:`SECONDARY`), each on its own stream
(``stream_seed(STUDY, cell, r)``) and run inside replicate ``r`` of ``day_cf`` for ``r`` below its
count, so each keeps the plan's count while the record holds one replicate count:

* ``mechanism_correct`` (:data:`MECHANISM_REPLICATES`, gated): CT-day with the cause hazards, the
  visit intensity and the continuation omitting :math:`L` and the default treatment and
  censoring models, at the 12 risks and differences of always and never at days 20 and 30
  (:data:`ROBUST_ROWS`).  Each row takes the primary rule (double robustness: targeting repairs
  :math:`Q`), which needs the mechanism right: the default treatment library's main-terms member,
  with its baseline indicator interacted with every column, contains law CT's baseline and visit
  treatment models, and the default censoring library's Cox main-terms member contains its
  censoring hazard (the law module's "What the defaults contain"; a logistic fit on 40,000 units
  of each law recovered every true treatment coefficient within 2 SE).  The untargeted plug-in of the same fits is the control ``plug_in_control``: its
  bias must lie beyond 0.2 SD at day 30 for always or never (:data:`PLUG_IN_ROWS`, any one).
* ``outcome_correct`` and ``both_wrong`` (:data:`REPORTED_REPLICATES` each, reported): the
  treatment and censoring models omit :math:`L` (the treatment model is the default's stratified
  main-terms member without :math:`L`), with the default intensities and continuation;
  and both sides omit :math:`L`.  The rows publish bias, coverage and the SE ratio with no verdict
  (plan 5.7, review B3).
* ``ltmle_daily`` (:data:`PAIRED_REPLICATES`, reported): the same draw as ``day_cf`` replicate
  ``r``, refitted by the discrete-time hazard LTMLE of X70 (b2) on a coarsened panel
  (:func:`panel`), against the continuous engine's estimate of the same draw.

**Deviation from plan 5.7 (the X70 pairing).**  The plan pairs the first 1,000 ``day_cf`` draws
on a 30-node daily panel.  E1's smoke measured X70's ``LTMLE(survival_model="hazard")`` at 124 s
on a 10-node panel of 3-day nodes and found that a 30-node daily panel did not finish in 3,000 s,
which projects past the plan's 200 CPU-h hold by itself.  The pairing therefore reads the first
:data:`PAIRED_REPLICATES` (400) draws on a 9-node panel whose nodes end at days
:data:`NODE_ENDS` (3, 6, 10, 13, 16, 20, 23, 26, 30).  Nine nodes cost less than the ten 3-day
nodes the brief named, and unlike them they end at every target day, so each of the 18 static
rows pairs at its own target day.  The estimands differ by the coarsening: a visit inside a node
changes :math:`L` and, by the law, :math:`A`, which the panel sees only at the next node start,
where its regime resets :math:`A`.  X70 is therefore read against the panel's risk
(``law.PANEL_TRUTHS``, simulated, review S5), the continuous engine against :math:`\psi_r`, and
the diagnosis trigger against the expected paired difference, the panel's risk minus
:math:`\psi_r` (at most 0.0085 in absolute value, at the cause-1 day-30 difference).  The rest is
the plan's: covariates carried forward, a visit indicator and its product with the previous
treatment among them, ``markov_order=0``, and the reading of the paired difference and the SE
ratio.  The rule's rows are not paired.  X70 withholds the interval of a row whose recursion read
a constant node (``constant_node_plugin``, at day 30 of always in one of the smoke's two draws):
the point estimate still pairs, and X70's SE ratio and coverage read only the draws with an
interval, whose share is reported (``x70_interval_refused_share``).  The row lead accepted this
redesign in place of the plan's hold.

**The visit cap** (plan 3.5.3, review S1).  Every gate compares with the full risk
:math:`\psi_r`, which stays the target of inference; the fit targets :math:`\psi^{(K^*_r)}`, the
regime through its first :math:`K^*_r` visits.  Each row records its regime's own cap (``cap``,
and ``reference_cap`` for a difference) and the capped truth :math:`\psi^{(K^*_r)}`
(``capped_truth``, from ``law.CAPPED_RISKS``), and ``visit-cap.csv`` reports the bias against
both truths with the distribution of the caps, so a cap gap can be told from an estimator bias.
On CT-day the gap at :math:`K = 4` is at most 0.0013 in absolute value (never, cause 1, day 30)
and 0.0036 at :math:`K = 3`.

**Support floors** (plan 3.5.9, review S7).  ``floors.csv`` (``readings.floors``) holds, per
gated primary row, its verdict, its median ``variance_rel_sd`` and event share, its fewest
followers at the last target time, its largest and 99th-percentile share past the cap and its
largest weight-floor and jump-bound shares.  ``readings.floor_values`` reads part (e)'s constants
from it after the run: ``LONGITUDINAL_REL_SD`` as the largest median rung at which every static
row at or below it passes, the event-share window as the passing rows' range cut short of any red
row, and the fewest followers and largest share past the cap over the passing static rows.  They
set no verdict.

**Calibration** (:mod:`tests.studies.continuous_time_longitudinal_readings`).  The risk difference
always minus never of cause 1 at day 20 of ``day_cf`` and of ``cont_cf`` takes the calibration
rule (the 99% interval of the SE ratio inside (0.93, 1.07) and the coverage interval inside
(0.92, 0.98)); the shrunk-SE control multiplies their SEs by 0.85 and must fall below the band
or wholly below the row's own interval, which carries a row that reads conservative (review B1).
``sparse_calibration`` reads ``day_cf``'s difference of cause 2 at day 10 under the same rule and
gates nothing (review S5).  A gate reads code: :func:`scientific_failures`.

**Size and seeds.**  :data:`PRIMARY_REPLICATES` is 3,200 and :data:`MECHANISM_REPLICATES`
2,400, fixed by Eric's choice on 2026-11-25 (plan 5.7's design values were 2,400 and 1,600;
2,400 is the ``mechanism_correct`` rung paired with 3,200, 1.4 CPU-h dearer than 1,600).  All 60
primary rows are gated, the 12 rule rows among them; ``mechanism_correct`` runs inside
``day_cf``'s replicates, so its count never exceeds the primary count (review S2).  The pilot
(:mod:`tests.studies.continuous_time_longitudinal_pilot`, :data:`PILOT_REPLICATES` draws of each
of ``day_cf``, ``cont_cf``, ``mechanism_correct`` and ``outcome_correct`` on its own stream) no
longer chooses a rung.  Each gated row is read at its group's pooled coverage and SE ratio and
at its own pilot standardized bias (``pilot.own_passes``).  The five rows whose own pass is below
0.96 (:data:`EXPOSED_ROWS`) stay gated with a declared route, by Eric's choice on 2026-11-25: a
red on the bias clause alone, with coverage and the SE ratio passing, keeps the row's claim
experimental and goes to an owner row, and any other red fails the record
(``readings.routed``, :func:`routed_performance`).  ``regenerate.py`` refuses a declared run
unless the product of the other 55 rows' pass probabilities reaches 0.95 at the declared counts
(``pilot.remainder_family_pass``, 0.988 on the second pilot).  The plan's move of rows with pilot coverage below 0.925 to reported is not
applied, by the same choice.  The record holds one count, so ``day_cf`` and ``cont_cf`` share
the primary count.  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study;
replicate ``r`` of a cell draws from ``replicate_seed(STUDY, cell, r)``, and the named streams
are ``pilot/<cell>``, ``plugin_limit`` and ``truth`` (``stream_seed``).

**Budget** (the second pilot's median seconds per fit, from 400 draws of each cell on the
stratified treatment default, and the smoke's for ``both_wrong`` and ``ltmle_daily``, which the
pilot does not run; peak memory is the largest a pilot worker recorded for ``day_cf`` and
``cont_cf`` and the smoke's process peak for the rest, whose pilot workers had also run
``cont_cf``; ``tests/canonical/continuous_time_longitudinal/pilot-diagnosis.md`` holds both).

======================  =======  ===============  =====  ===========
item                    fits     seconds per fit  CPU-h  peak memory
======================  =======  ===============  =====  ===========
``day_cf``              3,200    21.5             19.1   278 MB
``cont_cf``             3,200    143.1            127.2  2,172 MB
``mechanism_correct``   2,400    6.1              4.0    280 MB
``outcome_correct``     800      12.9             2.9    280 MB
``both_wrong``          800      2.4              0.5    281 MB
``ltmle_daily``         400      51.0             5.7    252 MB
total                                             159.5
======================  =======  ===============  =====  ===========

The projection, 159.5 CPU-h, is below the plan's hold of 200.  The declared run splits into
three shards (``regenerate.py --shard i/3``) of 53.2 CPU-h each, about 13.3 hours on a 4-core
cloud machine at ``--jobs 4``, which ``cont_cf``'s 2.2 GB per worker allows; ``--merge`` joins
them into the record.

**Registration** (plan 7.3) lands with the declared run's artifacts, in the change that ships
E1 and E2 together, and touches what a pending study cannot: the registry import and the
scenario, estimand, arm and cell descriptions (``descriptions.py``); the ``EXEMPT_STUDIES`` entry
(:data:`EXEMPT_REASON`); the clause bindings (``tests/canonical/clause-bindings.csv``); the
interval-audit screen row (``interval-audit-screen.csv``), the asymmetric screen row
(``asymmetric-audit-screen.csv``) and the asymmetric refit capture
(``asymmetric-refit/continuous-time-longitudinal/capture.json``); and the evidence page's tables,
grid row and toctree entry.  The study declares no property cells, so it needs no
``UNCOMMITTED_MODULES`` entry.  Part (e)'s support constants follow from ``floors.csv``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.exceptions import CapabilityError
from cleverly.inference.sparse import influence_row
from cleverly.survival import ContinuousLongitudinalTMLE
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_time_longitudinal_law as laws
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: ``day_cf`` and ``cont_cf``'s count, fixed by Eric's choice on 2026-11-25; the pilot checks
#: the anchored family pass at it (``pilot.anchored_family_pass``) and chooses no rung.
PRIMARY_REPLICATES = 3_200
#: ``mechanism_correct``'s count, the rung paired with 3,200 in the same choice.
MECHANISM_REPLICATES = 2_400
#: ``outcome_correct`` and ``both_wrong`` (plan 5.7).
REPORTED_REPLICATES = 800
#: The X70 pairing's draws: the first 400 ``day_cf`` draws (the module docstring's deviation).
PAIRED_REPLICATES = 400
#: The rungs of plan 5.7's family rule, the primary cells' and ``mechanism_correct``'s; the pilot
#: reports the family pass at each.
PRIMARY_RUNGS = (2_400, 3_200, 4_800)
MECHANISM_RUNGS = (1_600, 2_400, 3_200)
#: ``guide/studies.md``: primaries run at 1,000 replications or more.
MINIMUM_REPLICATES = 1_000
#: The plan's pilot of each of the four cells it names.
PILOT_REPLICATES = 400
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026112571
RESAMPLING_SEED = 2026112572
N_FOLDS = 10
LEARNER_FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.85

SUBJECT = "cleverly-continuous-time-longitudinal"

#: ``cell -> (law, regimes)``: every primary cell is the default cross-fitted fit at n = 1,000.
CELLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "day_cf": ("ct_day", tuple(laws.REGIMES)),
    "cont_cf": ("ct", tuple(laws.STATIC)),
}
#: Every risk and difference of ``day_cf`` (30) and of ``cont_cf`` (18).
DAY_ROWS: tuple[str, ...] = laws.targets(CELLS["day_cf"][1])
CONT_ROWS: tuple[str, ...] = laws.targets(CELLS["cont_cf"][1])
ESTIMANDS: dict[str, tuple[str, ...]] = {"day_cf": DAY_ROWS, "cont_cf": CONT_ROWS}
#: The 12 rows of ``mechanism_correct`` and the reported misspecified cells.
ROBUST_ROWS: tuple[str, ...] = laws.targets(tuple(laws.STATIC), times=laws.TIMES[1:])
#: The rows the plug-in control reads: day-30 risks of always and never; any one must fire.
PLUG_IN_ROWS: tuple[str, ...] = tuple(
    laws.level_name(regime, cause, laws.TIMES[-1])
    for regime in laws.STATIC
    for cause in laws.CAUSES
)
#: The calibration rows (plan 5.7) and the reported sparse row (review S5).
CALIBRATION_ROW = laws.difference_name("always", 1, 20.0)
CALIBRATION_CELLS: tuple[str, ...] = ("day_cf", "cont_cf")
SPARSE_ROW = laws.difference_name("always", 2, 10.0)
#: The 18 static rows the X70 pairing reads.
PAIRED_ROWS: tuple[str, ...] = laws.targets(tuple(laws.STATIC))
MARGINS = Margins(standardized_bias=0.2)

#: ``secondary cell -> (configuration, count, rows)``: each run inside ``day_cf``'s replicates.
SECONDARY: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "mechanism_correct": ("mechanism_correct", MECHANISM_REPLICATES, ROBUST_ROWS),
    "outcome_correct": ("outcome_correct", REPORTED_REPLICATES, ROBUST_ROWS),
    "both_wrong": ("both_wrong", REPORTED_REPLICATES, ROBUST_ROWS),
    "ltmle_daily": ("x70", PAIRED_REPLICATES, PAIRED_ROWS),
}
#: The gated secondary cell; the others are reported.
GATED_SECONDARY = ("mechanism_correct",)

#: A gated row whose own pass at the declared counts is below this is exposed (Eric's choice,
#: 2026-11-25): read at its own pilot standardized bias with its group's pooled coverage and SE
#: ratio (``pilot.own_passes``).
EXPOSED_PASS = 0.96
RULE_CLAIM = "rule regimes"
ROBUST_CLAIM = "double robustness (mechanism_correct)"
RULE_MODE = (
    "the rule levels' positive cause-1 bias at n = 1,000 (+0.12 to +0.13 SD in the second pilot, "
    "+0.11 to +0.13 in the first), already in the initial estimate and not removed by targeting; "
    "the first diagnosis excluded the visit cap, the residual score and the treatment misfit, and "
    "it is unexplained beyond that: CT-day's tie or day-grid approximations are suspected, not "
    "shown"
)
#: ``(cell, row) -> (claim, foreseen mode)`` of every exposed row (Eric's choice, 2026-11-25).
#: An exposed row stays gated; a red on its bias clause alone, with its coverage and SE ratio
#: passing, keeps its claim experimental and goes to an owner row instead of failing the record
#: (``readings.exposed``, ``routed_performance``).  The design test pins the list against the
#: committed pilot.
EXPOSED_ROWS: dict[tuple[str, str], tuple[str, str]] = {
    ("day_cf", laws.level_name("rule", 1, 10.0)): (RULE_CLAIM, RULE_MODE),
    ("day_cf", laws.level_name("rule", 1, 20.0)): (RULE_CLAIM, RULE_MODE),
    ("day_cf", laws.level_name("rule", 1, 30.0)): (RULE_CLAIM, RULE_MODE),
    ("mechanism_correct", laws.level_name("never", 1, 30.0)): (
        ROBUST_CLAIM,
        "the visit cap on never at day 30: the fit targets the capped risk and the gate reads the "
        "full one; the pilot bias is -0.112 SD against the full risk and -0.076 against the "
        "capped truth, so about 0.04 SD is the cap's gap and the rest lies within about 1.5 "
        "pilot SEs of zero",
    ),
    ("mechanism_correct", laws.difference_name("always", 1, 30.0)): (
        ROBUST_CLAIM,
        "partly the visit cap on its never arm: the pilot bias is +0.134 SD against the full "
        "difference and +0.094 against the capped one; the remaining +0.09 SD has no identified "
        "cause and lies within about 2 pilot SEs of zero, so it may be pilot noise",
    ),
}

#: The X70 panel's node ends, in days: three nodes per 10 days, of 3, 3 and 4 days.
NODE_ENDS: tuple[int, ...] = (3, 6, 10, 13, 16, 20, 23, 26, 30)

ROWS_FILE = "longitudinal-rows.csv.gz"
SECONDARY_FILE = "secondary-rows.csv.gz"
CALIBRATION_FILE = "calibration.csv"
MECHANISM_FILE = "mechanism-correct.csv"
MISSPECIFIED_FILE = "misspecified.csv"
PAIRING_FILE = "ltmle-pairing.csv"
READINGS_FILE = "targeting-readings.csv"
VISIT_CAP_FILE = "visit-cap.csv"
FLOORS_FILE = "floors.csv"
EXPOSED_FILE = "exposed-rows.csv"

STUDY = StudyRecord(
    name="continuous-time longitudinal TMLE",
    slug="continuous-time-longitudinal",
    artifacts=ROOT / "tests" / "canonical" / "continuous_time_longitudinal",
    document="docs/technical-reference/method-evidence/continuous-time-longitudinal.md",
    anchor="continuous-time-longitudinal-tmle",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=SUBJECT,
    reference=None,
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=5,
    extra_artifacts=(
        ROWS_FILE,
        SECONDARY_FILE,
        CALIBRATION_FILE,
        MECHANISM_FILE,
        MISSPECIFIED_FILE,
        PAIRING_FILE,
        READINGS_FILE,
        VISIT_CAP_FILE,
        FLOORS_FILE,
        EXPOSED_FILE,
    ),
    modules=(
        "tests/studies/continuous_time_longitudinal.py",
        "tests/studies/continuous_time_longitudinal_law.py",
        "tests/studies/continuous_time_longitudinal_readings.py",
        "tests/studies/continuous_time_longitudinal_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.continuous_time_longitudinal",
    properties_module="tests.studies.continuous_time_longitudinal_properties",
    property_cells={},
)

#: The slug's ``EXEMPT_STUDIES`` reason, entered in ``cleverly._support_rows`` in the change that
#: registers the record (plan 3.5.11 and 7.3; a pending study is not registered).
EXEMPT_REASON = (
    "It measures ContinuousLongitudinalTMLE, which no support key reads: the continuous-time fit "
    "has no node structure to classify. "
    "cleverly.survival.longitudinal_result.continuous_longitudinal_support, not a support row, "
    "records the configurations the study licenses."
)

CONFIGURATION = {
    "construction": "Rytgaard, Gerds and van der Laan (2022) TMLE, one backward pass per regime, "
    "influence-curve variance",
    "laws": {cell: law for cell, (law, _) in CELLS.items()},
    "regimes": {cell: list(regimes) for cell, (_, regimes) in CELLS.items()},
    "times": list(laws.TIMES),
    "causes": dict(laws.CAUSES),
    "coefficients": repr(laws.COEFFICIENTS),
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "secondary": {cell: [config, count] for cell, (config, count, _) in SECONDARY.items()},
    "x70_nodes": list(NODE_ENDS),
    "variance_method": "influence curve",
}


# -------------------------------------------------------------------------- the draws


def law_of(scenario: str) -> str:
    """The law a cell draws from."""
    if scenario in CELLS:
        return CELLS[scenario][0]
    return "ct_day"


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    law = law_of(scenario)
    return laws.SAMPLERS[law](n, seed), dict(laws.TRUTHS[law])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def secondary_seed(cell: str, replicate: int) -> int:
    """The seed of a secondary cell's draw ``replicate`` (the X70 pairing reads ``day_cf``'s)."""
    if cell == "ltmle_daily":
        return replicate_seed(STUDY, "day_cf", replicate)
    return stream_seed(STUDY, cell, replicate)


# --------------------------------------------------------------------------- the fits


def engine(regimes: tuple[str, ...], configuration: str = "default") -> ContinuousLongitudinalTMLE:
    """The cross-fitted fit of ``regimes``, with a configuration's misspecified learners."""
    return ContinuousLongitudinalTMLE(
        list(laws.TIMES),
        regimes={name: laws.REGIMES[name] for name in regimes},
        reference=laws.REFERENCE,
        causes=laws.CAUSES,
        cross_fit=True,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        random_state=0,
        **laws.CONFIGURATIONS[configuration](),
    )


def fit_with(frame: pd.DataFrame, subject: ContinuousLongitudinalTMLE) -> Any:
    """One fit of a law's frame."""
    return subject.fit(frame, **laws.COLUMNS)


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The subject's fit of one replicate of a primary cell."""
    return fit_with(frame, engine(CELLS[scenario][1]))


def readings(result: Any) -> dict[str, float]:
    """The targeting record of one fit."""
    record = result.record
    return {
        "steps": float(sum(sum(node) for node in record.steps)),
        "converged": float(record.converged),
        "bounded_share": float(record.bounded_share),
        "jump_bound_share": float(record.jump_bound_share),
        "visit_bound_share": float(record.visit_bound_share),
        "grid_size": float(record.grid_size),
        "dropped_visits": float(record.dropped_visits),
        "largest_cap": float(max(record.caps)),
    }


def cap_readings(result: Any, name: str, law: str) -> dict[str, float]:
    r"""The regime's own cap :math:`K^*_r`, the reference's for a difference, and the capped truth
    :math:`\psi^{(K^*_r)}` at those caps (review S1), a reading beside the gate's :math:`\psi_r`."""
    regime, _, _ = laws.parse(name)
    record = result.record
    cap = float(record.caps[record.regimes.index(regime)])
    reference = (
        float(record.caps[record.regimes.index(laws.REFERENCE)])
        if name.startswith("ate_regimen")
        else np.nan
    )
    return {
        "cap": cap,
        "reference_cap": reference,
        "capped_truth": laws.capped_truth(law, name, cap, reference),
    }


def _regime_readings(result: Any, name: str) -> dict[str, float]:
    regime, _, t = laws.parse(name)
    record = result.record
    position = record.regimes.index(regime)
    reference = record.regimes.index(laws.REFERENCE)
    time_index = laws.TIMES.index(t)
    followers = record.followers_at_last[position]
    share = record.past_cap_share[position][time_index]
    if name.startswith("ate_regimen"):
        followers = min(followers, record.followers_at_last[reference])
        share = max(share, record.past_cap_share[reference][time_index])
    return {"followers_at_last": float(followers), "past_cap_share": float(share)}


def event_share(frame: pd.DataFrame, name: str) -> float:
    """The observed share of units with an event of the row's cause by its time."""
    _, cause, t = laws.parse(name)
    last = frame.groupby("id", sort=False).tail(1)
    return float(np.mean((last["event"] == cause) & (last["stop"] <= t)))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """One row per gated target, on its reported scale, with the support readings."""
    rows = []
    initial = result.initial_estimates
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": SUBJECT,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "interval_scale": estimate.interval_scale,
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial[name]),
                "variance_rel_sd": float(
                    influence_row(name, estimate.influence_curve).variance_rel_sd
                ),
                "event_share": event_share(frame, name),
                **_regime_readings(result, name),
                **cap_readings(result, name, law_of(scenario)),
            }
        )
    return rows


# ------------------------------------------------------------------- the X70 pairing


def panel(frame: pd.DataFrame) -> pd.DataFrame:
    """The wide panel of a law CT-day frame on the nodes ending at :data:`NODE_ENDS`.

    Node ``k`` covers days :math:`(b_{k-1}, b_k]`.  Its covariates ``L{k}`` and treatment
    ``A{k}`` are the state after every visit on or before day :math:`b_{k-1}` (carried forward),
    ``V{k}`` indicates a visit in node ``k - 1``'s days and ``VA{k}`` is ``V{k}`` times the
    previous node's treatment.  ``C{k}`` is 0 when the unit is censored in node ``k``'s days;
    the event indicators ``R{k}`` (cause 1) and ``D{k}`` (cause 2) are then missing, as X70's
    node order (covariates, treatment, censoring, events) needs, and every later node is
    missing.
    """
    ends = np.asarray(NODE_ENDS, dtype=float)
    starts = np.r_[0.0, ends[:-1]]
    units = frame["id"].unique()
    n = units.size
    wide: dict[str, np.ndarray] = {"W1": np.zeros(n), "W2": np.zeros(n)}
    for k in range(1, ends.size + 1):
        for column in ("L", "V", "VA", "A", "C", "R", "D"):
            wide[f"{column}{k}"] = np.full(n, np.nan)
    for position, (_, rows) in enumerate(frame.groupby("id", sort=True)):
        visits = rows["start"].to_numpy(dtype=float)
        ls, treatments = rows["L"].to_numpy(dtype=float), rows["A"].to_numpy(dtype=float)
        end, event = float(rows["stop"].iloc[-1]), int(rows["event"].iloc[-1])
        wide["W1"][position], wide["W2"][position] = rows["W1"].iloc[0], rows["W2"].iloc[0]
        previous = 0.0
        for k, (low, high) in enumerate(zip(starts, ends, strict=True), start=1):
            if end <= low:
                break
            at = int(np.searchsorted(visits, low, side="right")) - 1
            earlier = starts[k - 2] if k > 1 else 0.0
            visit = float(np.any((visits > earlier) & (visits <= low) & (visits > 0)))
            wide[f"L{k}"][position], wide[f"A{k}"][position] = ls[at], treatments[at]
            wide[f"V{k}"][position], wide[f"VA{k}"][position] = visit, visit * previous
            previous = treatments[at]
            died = end <= high and event > 0
            censored = end <= high and event == 0 and end < laws.END
            wide[f"C{k}"][position] = 0.0 if censored else 1.0
            if censored:
                break
            wide[f"R{k}"][position] = float(died and event == 1)
            wide[f"D{k}"][position] = float(died and event == 2)
            if died:
                break
    return pd.DataFrame(wide)


def x70_node(t: float) -> int:
    """The X70 node that ends on target day ``t``."""
    return NODE_ENDS.index(int(t)) + 1


def x70_name(name: str) -> str:
    """X70's name of a continuous engine's static row."""
    regime, cause, t = laws.parse(name)
    label = laws.CAUSES[cause]
    if name.startswith("ate_regimen"):
        return f"ate_regimen[{regime} vs {laws.REFERENCE}, {label} @ t={x70_node(t)}]"
    return f"cif_regimen[{regime}, {label} @ t={x70_node(t)}]"


def x70_fit(frame: pd.DataFrame) -> Any:
    """X70 (b2)'s hazard LTMLE of a CT-day frame's panel, as E1's smoke fitted it."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly.longitudinal import LTMLE
    from cleverly.longitudinal.data import LongitudinalData

    wide = panel(frame)
    nodes = range(1, len(NODE_ENDS) + 1)
    data = LongitudinalData.from_frame(
        wide,
        treatment=[f"A{k}" for k in nodes],
        baseline=["W1", "W2"],
        censoring=[f"C{k}" for k in nodes],
        time_varying=[[f"L{k}", f"V{k}", f"VA{k}"] for k in nodes],
        outcome={
            laws.CAUSES[1]: [f"R{k}" for k in nodes],
            laws.CAUSES[2]: [f"D{k}" for k in nodes],
        },
    )
    data = data.with_markov_order(0)
    return LTMLE(
        dict(laws.STATIC),
        reference=laws.REFERENCE,
        survival_model="hazard",
        targeting="one_step",
        outcome_learner=LogisticRegression(max_iter=2000),
        treatment_learner=LogisticRegression(max_iter=2000),
        censoring_learner=LogisticRegression(max_iter=2000),
        pseudo_learner=LinearRegression(),
        random_state=0,
    ).fit(data)


# ----------------------------------------------------------------- secondary fits


#: The numeric fields a secondary fit writes per row.
SECONDARY_FIELDS = (
    "estimate",
    "std_error",
    "covered",
    "initial",
    "failed",
    "interval_refused",
    "seconds",
    "cap",
    "reference_cap",
    "capped_truth",
)
#: The text field: the error class of a recorded failure, empty otherwise.
SECONDARY_TEXT = ("error",)
#: The cap readings of a fit that has no visit cap (X70).
_NO_CAP = {"cap": np.nan, "reference_cap": np.nan, "capped_truth": np.nan}


def secondary_columns() -> tuple[str, ...]:
    """The extra columns that carry the secondary fits on ``day_cf``'s rows."""
    return tuple(
        f"{cell}_{field}" for cell in SECONDARY for field in (*SECONDARY_FIELDS, *SECONDARY_TEXT)
    )


def _secondary(cell: str, replicate: int, primary_frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    r"""``row -> field -> value`` of one secondary fit, or every row failed with its error class.

    Only an error :func:`~cleverly.validation._failures.recordable` judges a fit failure (a
    ``DataError`` refusal of the draw, a numerical failure) is recorded; any other error, a
    configuration refusal or a defect, stops the run as the shared harness does (review S6).
    X70's rows read the panel's truth (:data:`~tests.studies.continuous_time_longitudinal_law.PANEL_TRUTHS`),
    every other cell's the full risk :math:`\psi_r` and, beside it, the capped truth at the fit's
    own caps.
    """
    configuration, count, names = SECONDARY[cell]
    out: dict[str, dict[str, Any]] = {}
    if replicate >= count:
        return out
    paired = cell == "ltmle_daily"
    truth = laws.PANEL_TRUTHS if paired else laws.TRUTHS["ct_day"]
    frame = primary_frame if paired else laws.sample_day(PRIMARY_N, secondary_seed(cell, replicate))
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if paired:
                result = x70_fit(frame)
                values = {name: result.estimates[x70_name(name)] for name in names}
                initial = dict.fromkeys(names, np.nan)
                caps = dict.fromkeys(names, _NO_CAP)
            else:
                result = fit_with(frame, engine(tuple(laws.STATIC), configuration))
                values = {name: result[name] for name in names}
                initial = {name: float(result.initial_estimates[name]) for name in names}
                caps = {name: cap_readings(result, name, "ct_day") for name in names}
    except Exception as error:
        if not recordable(error):
            raise
        return {name: {"failed": 1.0, "error": type(error).__name__} for name in names}
    seconds = time.perf_counter() - start
    for name, estimate in values.items():
        try:
            low, high = estimate.ci
            spread, covered, refused = (
                float(estimate.std_error),
                float(low <= truth[name] <= high),
                0.0,
            )
        except CapabilityError:
            # X70 withholds the interval of a row whose recursion read a constant node
            # (``constant_node_plugin``); the point estimate stands and pairs.
            spread, covered, refused = np.nan, np.nan, 1.0
        out[name] = {
            "estimate": float(estimate.psi),
            "std_error": spread,
            "covered": covered,
            "initial": initial[name],
            "failed": 0.0,
            "interval_refused": refused,
            "seconds": seconds,
            **caps[name],
            "error": "",
        }
    return out


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
    "interval_scale",
    "seconds",
    "steps",
    "converged",
    "bounded_share",
    "jump_bound_share",
    "visit_bound_share",
    "grid_size",
    "dropped_visits",
    "largest_cap",
    "variance_rel_sd",
    "event_share",
    "followers_at_last",
    "past_cap_share",
    "cap",
    "reference_cap",
    "capped_truth",
    *secondary_columns(),
)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    secondary: dict[str, dict[str, dict[str, Any]]] = {}
    if scenario == "day_cf":
        secondary = {cell: _secondary(cell, replicate, frame) for cell in SECONDARY}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = fit(frame, scenario)
        seconds = time.perf_counter() - start
    extra = {"seconds": seconds, **readings(result)}
    rows = []
    for row in result_rows(result, truth, scenario, replicate, frame):
        values: dict[str, Any] = dict.fromkeys(secondary_columns(), np.nan)
        values.update({f"{cell}_{field}": "" for cell in SECONDARY for field in SECONDARY_TEXT})
        for cell, table in secondary.items():
            for field, value in table.get(row["estimand"], {}).items():
                values[f"{cell}_{field}"] = value
        rows.append({**row, **extra, **values})
    return rows


def shard_keys(
    replicates: int,
    shard: tuple[int, int] | None = None,
    scenarios: tuple[str, ...] | None = None,
) -> list[tuple[str, int]]:
    """The ``(cell, replicate)`` keys of a run, or of shard ``(index, total)`` of it.

    Shard ``i`` of ``k`` holds every replicate ``r`` with ``r % k == i - 1`` in every cell, so
    the shards are disjoint, cover the run, and each holds a third of the ``cont_cf`` fits and
    of the secondary fits inside ``day_cf``'s replicates when ``k`` is 3.  A replicate's seed is
    a function of its key alone, so a shard draws and fits what the unsharded run does.
    """
    chosen = tuple(STUDY.scenarios) if scenarios is None else scenarios
    index, total = (1, 1) if shard is None else shard
    if not 1 <= index <= total:
        raise ValueError(f"shard {index}/{total} is not one of 1/{total} to {total}/{total}")
    return [
        (scenario, replicate)
        for scenario in chosen
        for replicate in range(replicates)
        if replicate % total == index - 1
    ]


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
    shard: tuple[int, int] | None = None,
) -> replication.PhaseResult:
    """Every cell of ``scenarios`` (all by default), each at its declared size.

    ``shard`` fits only that shard's keys (:func:`shard_keys`), for a declared run split across
    machines (``regenerate.py --shard``).
    """
    sizes = STUDY.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario, replicate in shard_keys(replicates, shard, scenarios)
        ],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]])


# --------------------------------------------------------------- reported artifacts


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The per-row readings, the secondary fits and the tables read from them."""
    from tests.studies import continuous_time_longitudinal_readings as readings_module

    return readings_module.artifacts(rows)


def routed_performance(failures: pd.DataFrame) -> pd.DataFrame:
    """The independent-performance failures a declared route carries (the driver's hook).

    The rows of ``failures`` that are exposed (:data:`EXPOSED_ROWS`) and red on the bias clause
    alone, with coverage and the SE ratio passing.  The shared driver drops them from its gate,
    and ``exposed-rows.csv`` records each one as routed, its claim experimental.
    """
    from tests.studies import continuous_time_longitudinal_readings as readings_module

    keep = [
        readings_module.routed(
            str(row["scenario"]),
            str(row["estimand"]),
            bias_equivalent=bool(row["bias_equivalent"]),
            coverage_valid=bool(row["coverage_valid"]),
            se_calibrated=bool(row["se_calibrated"]),
        )
        for _, row in failures.iterrows()
    ]
    return failures.loc[keep] if len(failures) else failures


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The gates read by code beside the primary rule (plan 5.7)."""
    from tests.studies import continuous_time_longitudinal_readings as readings_module

    return readings_module.gates(artifacts)


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """The gates with each lost ``day_cf`` replicate counted against the secondary cells it held.

    A ``day_cf`` replicate that raised carries no secondary fit, so each lost replicate below a
    gated secondary cell's count fails that cell's audits once (decision Q1); a lost ``cont_cf``
    replicate fails its calibration row.
    """
    from tests.studies import continuous_time_longitudinal_readings as readings_module

    lost = evidence_failures.expand(failures, STUDY)
    day = lost.loc[lost["scenario"] == "day_cf", "replicate"].astype(int)
    cont = lost.loc[lost["scenario"] == "cont_cf", "replicate"].astype(int)
    missing = {
        readings_module.MECHANISM_GATE: int(np.sum(day < MECHANISM_REPLICATES)),
        readings_module.PLUG_IN_GATE: int(np.sum(day < MECHANISM_REPLICATES)),
        readings_module.CALIBRATION_GATE: int(day.size + cont.size),
        readings_module.SHRUNK_GATE: int(day.size + cont.size),
    }
    return evidence_failures.audits(readings_module.gates(artifacts), missing=missing)
