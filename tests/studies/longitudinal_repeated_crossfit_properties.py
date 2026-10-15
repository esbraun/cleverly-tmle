r"""Property families of ``longitudinal-repeated-crossfit`` (``X46-plan.md`` 5.2-5.6).

Every fit is the study's declared repeated fit
(:func:`tests.studies.longitudinal_repeated_crossfit.fit_scenario`): three draws of five outer
folds, saturated cell means, ``g_bounds = (0.01, 1)``, ``random_state = 0``.  The labels are
those of :data:`tests.studies.longitudinal_repeated_crossfit.CALIBRATION_LABELS`.

==========================  ===================================================================
family                      cells, replications and size
==========================  ===================================================================
``interval_calibration``    nine labels on laws D (``LTMLE`` and ``SDR``, static and dynamic
                            contrasts), I (odds x2 and x0.5 contrasts, n = 80,000), S (risk
                            difference at t = 2), C (static contrast) and M (``duration``);
                            24,000 each at n = 8,000 unless named, with a derived
                            ``shrunken_se_control`` (reported SE times 0.70) per label
``simultaneous_coverage``   the shipped band of each calibration fit and its pointwise
                            control: D-``LTMLE`` and D-``SDR`` (five names each), I (five),
                            S (nine), C (three) and M (the two centred coefficients)
``repeat_stability``        one fixed law-D sample (n = 8,000) and one fixed law-I sample
                            (n = 80,000), 400 paired fold-seed trials: the three-draw median
                            and the first draw it derives, alone
``repeat_increment``        the S calibration fits at horizons 1 to 3: median SE ratio minus
                            draw-0 SE ratio on the same fits; horizon 3 is gated, horizons 1
                            and 2 are reported; ``inflated_se_control`` multiplies the median
                            SE by 1.05
``single_draw_reading``     draw 0 of every calibration label, read by the calibration
                            statistics without a verdict (the red-cell diagnosis reads it)
``split_displacement``      the share :math:`\operatorname{median}_r(\hat\psi_r -
                            \tilde\psi)^2 / \tilde\sigma^2` of every calibration label,
                            reported only
``root_n_and_efficiency``   the law-D static contrast of ``LTMLE`` and ``SDR`` at n = 2,000
                            (control), 8,000 and 32,000; 800 each
``root_n_rate``             the two rate rows of each ladder
``ipsi_primary``            law I's primary rows at n = 80,000: the three plan means and two
                            contrasts, 1,600 replications, read by the primary rules
==========================  ===================================================================

**Rules.**  ``interval_calibration``, ``simultaneous_coverage``, ``root_n_and_efficiency`` and
``root_n_rate`` read the shared verdicts.  ``repeat_stability`` passes when the 99% upper
endpoint of the paired spread ratio is below 1 (:mod:`tests.studies.repeated_crossfit_properties`
reads the same rule).  ``repeat_increment`` passes when the 99% upper endpoint of the paired
increment is below :data:`INCREMENT_CEILING`; its control passes when its own endpoint is at or
above it.  ``ipsi_primary`` reads the three primary rules: the bias interval inside the margin,
the coverage lower endpoint above the floor, and the SE-ratio interval inside the sanity band.

**What each family can see.**  The split adjustment of (3.14) moves a reported SE by at most
half of the median displacement share, which the pilot reads at 1.2e-4 to 0.0050 on the
calibration labels (``tests/diagnostics/x46_sizing/reading.csv``, column ``share``).  So
``interval_calibration``, ``ipsi_primary`` and ``simultaneous_coverage`` would pass with draw
0's variance alone, or with (3.14) less its displacement term.  They are evidence for the
median's inference, not for the displacement term.  Only ``repeat_stability`` (spread ratios
0.62 to 0.73 against exactly 1 when every draw reuses one split) and ``repeat_increment`` at
horizon 3 (share 0.0077, increment +0.0165) read the repeat layer.  The unit witnesses of
``X46-plan.md`` 4.2 and 4.3 carry the exact check of the displacement term.

**The ladder's 2,000 rung.**  The rung keeps the shared ``control`` role and rule, but at 800
replications the rule's two regions cover every count, so the rung cannot fail.  It is not a
slow-convergence control and is never counted as evidence.  The ``root_n_rate`` slope rows
carry that check: the slope interval must lie within 0.125 of -0.5 and exclude -0.25.

**Failures.**  A fit that raises a recordable error, or returns a row with a non-finite
estimate or a non-positive standard error, is recorded by
:func:`tests.studies.evidence.replication.fit_replicates` and costs its replicate; the run goes
on.  Every row of the fit set publishes the count in ``failed_replicates``.  The summary reads
each cell's survivors, and a cell whose fit set lost more than
``floor(FAILURE_BOUND * replicates)`` replicates is red (:func:`summarize_properties`).  A
checkpointed failure replays on resume, so the summary can be re-read without refitting.

**Declared red-cell reading.**  :data:`RED_CELL_POLICY`, declared before the run; the owner of
every reported red cell is :data:`RED_CELL_OWNER`.  :data:`PASS_PROBABILITIES` states the
declared pass probability that a verdict is read beside.

**Resuming.**  Every fit set goes through
:func:`tests.studies.evidence.replication.fit_replicates`, which checkpoints each replicate
under the regeneration driver's scratch.  A stopped run resumes at the replicates it had not
finished.  :func:`fit_part` fits one fit set, or its first replicates, into the same
checkpoints, so ``tests/canonical/longitudinal_repeated_crossfit/part.py`` can run the study in
parts before the declared run collects them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.learners.crossfit import CrossFitPlan
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies.evidence import replication
from tests.studies.evidence.inference import percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    bootstrap_draw_blocks,
    paired_spread_ratio_interval,
    ratio_intervals,
    replicate_row,
    require_complete,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows

STUDY = study.STUDY

#: The share of a fit set's replicates that may record a fitting failure.  A cell whose fit
#: set lost more than ``floor(FAILURE_BOUND * replicates)`` is red (RM39 part a2's count).
FAILURE_BOUND = 0.01
#: Rounding slack on ``FAILURE_BOUND * replicates``, as ``tests.studies.evidence.failures``.
_FLOOR_SLACK = 1e-9

#: The owner of every red cell that the declared reading reports (``X46-plan.md`` 5).
RED_CELL_OWNER = "X46-finite-sample"

#: The red-cell reading, declared before the run (AGENT-BRIEF "Reading a red calibration
#: row"; ``X46-plan.md`` 5.6).  ``tests/unit/test_longitudinal_repeated_crossfit_design.py``
#: checks the numbers it states.
RED_CELL_POLICY = (
    "1. The policy is reporting. No size, margin, law, learner or rule changes after a verdict "
    "is seen.",
    "2. Every red row is first a defect suspect. The diagnosis below runs; a defect found is "
    "fixed and the study re-runs under a fresh declaration. A red row is reported with owner "
    f"{RED_CELL_OWNER} only when the diagnosis finds no defect. An unfixed defect never ends "
    "as an owner row.",
    "3. Calibration and band diagnosis: compare the label's single_draw_reading row with its "
    "median row, read its split_displacement share, and compare both with the pilot reading in "
    "tests/diagnostics/x46_sizing/reading.csv. A red median row whose draw-0 row is green is a "
    "repeat-layer defect suspect. A red median row whose draw-0 row is off by the same amount, "
    "with a share below 0.01, is a single-draw reading: the repeat layer did not cause it.",
    "4. A red calibration or band row whose SE-ratio 99% interval contains 1, whose coverage "
    "interval contains 0.95, and whose diagnosis finds no defect is a finite-sample red, "
    f"reported with owner {RED_CELL_OWNER}.",
    "5. A red calibration row whose SE-ratio 99% interval excludes 1 is the most probable red "
    "mode: a true ratio near 1.05 reads an interval near [1.03, 1.07]. It stays a defect "
    "suspect until rule 3 reads a single-draw reading that agrees with the pilot within its "
    "Monte Carlo error, and the unit witnesses of X46-plan.md 4.2 and 4.3 pass at the run's "
    f"commit. It is then reported with owner {RED_CELL_OWNER} as a single-draw finite-sample "
    "shortfall, and the evidence page states the ratio. A disagreement is a defect suspect.",
    "6. A red ipsi_primary row is diagnosed against the calibration_i median and draw-0 rows of "
    "the same law and size, and against the pilot bias (+0.067 and -0.077 SD on the contrasts). "
    "A bias that the draw-0 rows share and that agrees with the pilot is a second-order "
    f"finite-sample bias, reported with owner {RED_CELL_OWNER}. Otherwise it is a defect "
    "suspect.",
    "7. A red repeat_stability or repeat_increment row is a repeat-layer defect suspect and is "
    "never reported on an owner row. A stability row whose one-draw split variance is at or "
    "below STABILITY_SPREAD_FLOOR reads rounding noise; that is a design defect.",
    "8. A red root_n_rate row is diagnosed against the per-rung rows and the pilot segment slope "
    "of -0.587 between 2,000 and 8,000. A slope that the positive rungs and the pilot both "
    f"explain is a finite-sample curvature, reported with owner {RED_CELL_OWNER}. The 2,000 "
    "rung cannot fail at 800 replications and is not evidence.",
    "9. A recorded fitting failure is a defect suspect. The failure records in run.log are "
    "diagnosed and the summary is re-read from the checkpoints without refitting. A cell whose "
    "fit set lost at most floor(0.01 * replicates) replicates keeps the verdict of its "
    "survivors. More than that is red; with no defect found it is reported with owner "
    f"{RED_CELL_OWNER}.",
    "10. A control that does not fail where it is declared to have power is a design defect "
    "and is fixed before any rerun.",
    "11. Each verdict is read beside its declared pass probability (PASS_PROBABILITIES). The "
    "nine calibration cells pass jointly with probability 1.00 at the pilot reading and 0.89 "
    "predictively, with the pilot's correlated errors. A red calibration row is therefore "
    "likely and is read by rules 3 to 5.",
)

#: The declared predictive pass probability of each gated cell, rounded to two decimals:
#: the pilot reading's uncertainty is integrated out.  ``calibration_family`` is the joint
#: probability of the nine calibration cells, by a bootstrap of the pilot samples.
PASS_PROBABILITIES: dict[str, float] = {
    "interval_calibration/d_ltmle_static": 0.99,
    "interval_calibration/d_ltmle_dynamic": 1.00,
    "interval_calibration/d_sdr_static": 0.99,
    "interval_calibration/d_sdr_dynamic": 1.00,
    "interval_calibration/i_contrast_x2": 0.97,
    "interval_calibration/i_contrast_x0_5": 0.96,
    "interval_calibration/s_contrast_h2": 0.98,
    "interval_calibration/c_static": 1.00,
    "interval_calibration/m_duration": 1.00,
    "interval_calibration/calibration_family": 0.89,
    "simultaneous_coverage/d_ltmle": 1.00,
    "simultaneous_coverage/d_sdr": 1.00,
    "simultaneous_coverage/i_ipsi": 1.00,
    "simultaneous_coverage/s_ltmle": 1.00,
    "simultaneous_coverage/c_ltmle": 1.00,
    "simultaneous_coverage/m_ltmle": 1.00,
    "repeat_increment/h3": 1.00,
    "repeat_stability/d_ltmle_static": 1.00,
    "repeat_stability/d_ltmle_dynamic": 1.00,
    "repeat_stability/d_sdr_static": 1.00,
    "repeat_stability/d_sdr_dynamic": 1.00,
    "repeat_stability/i_contrast_x2": 1.00,
    "root_n_rate/ltmle_static__empirical_sd": 0.87,
    "root_n_rate/sdr_static__empirical_sd": 0.86,
    "root_n_rate/ltmle_static__reported_se": 1.00,
    "root_n_rate/sdr_static__reported_se": 1.00,
    "ipsi_primary/mean_x0_5": 1.00,
    "ipsi_primary/mean_x1": 1.00,
    "ipsi_primary/mean_x2": 1.00,
    "ipsi_primary/contrast_x0_5": 1.00,
    "ipsi_primary/contrast_x2": 1.00,
}

#: The smallest one-draw split variance, as a share of the mean reported variance, that a
#: stability label must show.  On the pilot's fixed samples the share is 4.1e-26 for the
#: natural course at 80,000, which no draw moves (rounding), and at least 3.5e-4 for the
#: declared names.  1e-8 is far above the first and more than four orders below the second.
STABILITY_SPREAD_FLOOR = 1e-8

CALIBRATION_REPLICATES = 24_000
CALIBRATION_N = study.PRIMARY_N
IPSI_N = study.IPSI_N
STABILITY_TRIALS = 400
STABILITY_N = {"d": study.PRIMARY_N, "i": IPSI_N}
RATE_REPLICATES = 800
RATE_SIZES = study.RATE_SIZES
IPSI_PRIMARY_REPLICATES = study.PRIMARY_REPLICATES
SHRUNKEN_SE_FACTOR = 0.70
INFLATED_SE_FACTOR = 1.05
INCREMENT_CEILING = 0.03
INCREMENT_HORIZON = 3
MAX_REPEAT_SPREAD_RATIO = 1.0
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: ``label: (fit set, scenario, estimand)`` of each calibration label.
CALIBRATION: dict[str, tuple[str, str, str]] = {
    "d_ltmle_static": ("calibration_d", "law_d_ltmle", study.D_CONTRASTS["static"]),
    "d_ltmle_dynamic": ("calibration_d", "law_d_ltmle", study.D_CONTRASTS["dynamic"]),
    "d_sdr_static": ("calibration_d", "law_d_sdr", study.D_CONTRASTS["static"]),
    "d_sdr_dynamic": ("calibration_d", "law_d_sdr", study.D_CONTRASTS["dynamic"]),
    "i_contrast_x2": ("calibration_i", "law_i", study.I_CONTRASTS["contrast_x2"]),
    "i_contrast_x0_5": ("calibration_i", "law_i", study.I_CONTRASTS["contrast_x0_5"]),
    "s_contrast_h2": ("calibration_s", "law_s", study.s_contrast(2)),
    "c_static": ("calibration_c", "law_c", "ate_regimen[always vs never]"),
    "m_duration": ("calibration_m", "law_m", "msm_regimen[duration]"),
}
if tuple(CALIBRATION) != study.CALIBRATION_LABELS:  # pragma: no cover - a declaration guard
    raise AssertionError("the calibration labels drifted from the study module")

#: ``band label: (fit set, scenario, the names the band covers)``.
BANDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "d_ltmle": ("calibration_d", "law_d_ltmle", (*study.D_MEANS, *study.D_CONTRASTS.values())),
    "d_sdr": ("calibration_d", "law_d_sdr", (*study.D_MEANS, *study.D_CONTRASTS.values())),
    "i_ipsi": ("calibration_i", "law_i", study.I_NAMES),
    "s_ltmle": ("calibration_s", "law_s", study.S_NAMES),
    "c_ltmle": ("calibration_c", "law_c", study.C_NAMES),
    "m_ltmle": ("calibration_m", "law_m", study.M_NAMES),
}

#: ``label: (law, scenario, estimand)`` of each stability label.
STABILITY: dict[str, tuple[str, str, str]] = {
    "d_ltmle_static": ("d", "law_d_ltmle", study.D_CONTRASTS["static"]),
    "d_ltmle_dynamic": ("d", "law_d_ltmle", study.D_CONTRASTS["dynamic"]),
    "d_sdr_static": ("d", "law_d_sdr", study.D_CONTRASTS["static"]),
    "d_sdr_dynamic": ("d", "law_d_sdr", study.D_CONTRASTS["dynamic"]),
    "i_contrast_x2": ("i", "law_i", study.I_CONTRASTS["contrast_x2"]),
}

#: ``fit set: (law, scenarios, n, replicates)``.  The fit set names the sample stream.
FIT_SETS: dict[str, tuple[str, tuple[str, ...], int, int]] = {
    "calibration_d": ("d", ("law_d_ltmle", "law_d_sdr"), CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_i": ("i", ("law_i",), IPSI_N, CALIBRATION_REPLICATES),
    "calibration_s": ("s", ("law_s",), CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_c": ("c", ("law_c",), CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_m": ("m", ("law_m",), CALIBRATION_N, CALIBRATION_REPLICATES),
    **{
        f"rate_{size}": ("d", ("law_d_ltmle", "law_d_sdr"), size, RATE_REPLICATES)
        for size in RATE_SIZES
    },
    "ipsi_primary": ("i", ("law_i",), IPSI_N, IPSI_PRIMARY_REPLICATES),
}

INCREMENT_COLUMNS = (
    "increment",
    "increment_ci_lower",
    "increment_ci_upper",
    "increment_ceiling",
    "increment_se_part",
    "increment_spread_part",
)
STABILITY_COLUMNS = (
    "spread_ratio",
    "spread_ratio_ci_lower",
    "spread_ratio_ci_upper",
    "spread_ratio_boundary",
)
SHARE_COLUMN = "displacement_share"


def sample_seed(fit_set: str, replicate: int) -> int:
    """The seed replicate ``replicate`` of ``fit_set`` is drawn with."""
    return stream_seed(STUDY, "property_sample", fit_set, replicate)


# ---------------------------------------------------------------------------------- rows


def _calibration_rows(
    fit_set: str, scenario: str, result: Any, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    truth = study.I_TRUTH if scenario == "law_i" else study.TRUTHS[scenario]
    first = result.repeats[0].estimates

    def row(family: str, cell: str, role: str, estimate: Any, name: str) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=float(truth[name]),
            estimate=estimate,
            alpha=STUDY.margins.alpha,
        )

    rows: list[dict[str, Any]] = []
    for label, (owner, owner_scenario, name) in CALIBRATION.items():
        if (owner, owner_scenario) != (fit_set, scenario):
            continue
        median = result[name]
        rows.append(
            row("interval_calibration", f"{label}__correctly_specified", "positive", median, name)
        )
        rows.append(
            row("single_draw_reading", f"{label}__draw_0", DIAGNOSTIC_ROLE, first[name], name)
        )
        draws = np.array([float(draw.estimates[name].psi) for draw in result.repeats])
        share = float(np.median((draws - median.psi) ** 2) / median.std_error**2)
        rows.append(
            {
                "property": "split_displacement",
                "cell": f"{label}__share",
                "role": DIAGNOSTIC_ROLE,
                "replicate": replicate,
                "n": n,
                "requested_replicates": requested,
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": share,
                "std_error": float(median.std_error),
                "covered": 0,
                "rejected": 0,
            }
        )
    if scenario == "law_s":
        for horizon in study.HORIZONS:
            name = study.s_contrast(horizon)
            role = "positive" if horizon == INCREMENT_HORIZON else DIAGNOSTIC_ROLE
            rows.append(
                row("repeat_increment", f"h{horizon}__repeat_median", role, result[name], name)
            )
            rows.append(
                row("repeat_increment", f"h{horizon}__single_draw", role, first[name], name)
            )
    for label, (owner, owner_scenario, names) in BANDS.items():
        if (owner, owner_scenario) != (fit_set, scenario):
            continue
        rows.extend(
            joint_coverage_rows(
                result,
                truth,
                names,
                label=label,
                replicate=replicate,
                n=n,
                requested=requested,
                pointwise_critical=CRITICAL,
            )
        )
    return rows


def _rate_rows(
    scenario: str, result: Any, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    label = "ltmle_static" if scenario == "law_d_ltmle" else "sdr_static"
    name = study.D_CONTRASTS["static"]
    return [
        replicate_row(
            property_name="root_n_and_efficiency",
            cell=f"{label}__n_{n}",
            role="control" if n == min(RATE_SIZES) else "positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=study.D_TRUTH[name],
            estimate=result[name],
            alpha=STUDY.margins.alpha,
        )
    ]


def _primary_rows(result: Any, replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    return [
        replicate_row(
            property_name="ipsi_primary",
            cell=f"{label}__repeated",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=study.I_TRUTH[name],
            estimate=result[name],
            alpha=STUDY.margins.alpha,
        )
        for label, name in study.IPSI_PRIMARY_LABELS.items()
    ]


def fit_set_rows(key: tuple[str, int, int], drawn: Any = None) -> list[dict[str, Any]]:
    """Every row one replicate of one fit set publishes: ``key = (fit set, replicate, R)``."""
    del drawn
    fit_set, replicate, requested = key
    law, scenarios, n, _ = FIT_SETS[fit_set]
    frame = study.sample(law, n, sample_seed(fit_set, replicate))
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        calibration = fit_set.startswith("calibration_")
        result = study.fit_scenario(scenario, frame.copy(), simultaneous=calibration)
        if result.n_repeats != study.REPEATS:
            raise RuntimeError(f"{fit_set}: the fit drew {result.n_repeats} splits")
        if calibration:
            rows.extend(_calibration_rows(fit_set, scenario, result, replicate, n, requested))
        elif fit_set.startswith("rate_"):
            rows.extend(_rate_rows(scenario, result, replicate, n, requested))
        else:
            rows.extend(_primary_rows(result, replicate, n, requested))
    return rows


def first_draw_seed(base_seed: int) -> int:
    """The first fold seed that a three-draw fit at ``base_seed`` derives."""
    seed = CrossFitPlan(
        n_folds=study.N_FOLDS, random_state=base_seed, repeats=study.REPEATS
    ).seeds()[0]
    if seed is None:  # pragma: no cover - a cross-fitted plan always derives a seed
        raise RuntimeError("the first repeat seed unexpectedly derived None")
    return seed


def stability_sample(law: str) -> pd.DataFrame:
    """The one fixed sample a law's stability trials hold."""
    return study.sample(
        law, STABILITY_N[law], stream_seed(STUDY, "repeat_stability", "sample", law)
    )


def stability_rows(key: tuple[str, int, int], drawn: Any = None) -> list[dict[str, Any]]:
    """One paired trial on law ``key[0]``'s fixed sample: three draws, and the first alone."""
    del drawn
    law, trial, requested = key
    frame = stability_sample(law)
    base = stream_seed(STUDY, "repeat_stability", "fold_seed", law, trial)
    first = first_draw_seed(base)
    rows: list[dict[str, Any]] = []
    for scenario in dict.fromkeys(s for (lw, s, _) in STABILITY.values() if lw == law):
        repeated = study.fit_scenario(scenario, frame.copy(), random_state=base)
        single = study.fit_scenario(scenario, frame.copy(), repeats=1, random_state=first)
        if not np.array_equal(repeated.repeats[0].folds.assignment, single.folds.assignment):
            raise RuntimeError(f"{scenario}: the one-draw control is not the first fold draw")
        for label, (owner, owner_scenario, name) in STABILITY.items():
            if (owner, owner_scenario) != (law, scenario):
                continue
            if float(repeated.repeats[0].estimates[name].psi) != float(single[name].psi):
                raise RuntimeError(f"{label}: the one-draw control does not reproduce draw 0")
            truth = study.I_TRUTH[name] if law == "i" else study.D_TRUTH[name]
            for cell, role, estimate in (
                ("three_repeats", "positive", repeated[name]),
                ("one_repeat_control", "control", single[name]),
            ):
                rows.append(
                    replicate_row(
                        property_name="repeat_stability",
                        cell=f"{label}__{cell}",
                        role=role,
                        replicate=trial,
                        n=STABILITY_N[law],
                        requested=requested,
                        truth=truth,
                        estimate=estimate,
                        alpha=STUDY.margins.alpha,
                    )
                )
    return rows


# ------------------------------------------------------------------------------ the run


def allowed_failures(replicates: int) -> int:
    """How many replicates of one fit set may fail before every cell it feeds is red."""
    return math.floor(FAILURE_BOUND * replicates + _FLOOR_SLACK)


def invalid_rows(rows: list[dict[str, Any]]) -> str | None:
    """The first row with a non-finite estimate or a non-positive standard error, if any."""
    for row in rows:
        estimate, std_error = float(row["estimate"]), float(row["std_error"])
        if not (np.isfinite(estimate) and np.isfinite(std_error) and std_error > 0):
            return (
                f"{row['property']}/{row['cell']} replicate {row['replicate']}: estimate "
                f"{estimate}, std_error {std_error}"
            )
    return None


def _fit(
    label: str,
    worker: Callable[[tuple[str, int, int], Any], list[dict[str, Any]]],
    keys: list[tuple[str, int, int]],
    n_jobs: int,
) -> list[dict[str, Any]]:
    """The rows of every replicate that succeeded; each row carries the fit set's failures."""
    outcomes = replication.fit_replicates(
        keys,
        worker,
        n_jobs=n_jobs,
        # The requested count joins the label, so a run at another budget never reuses these.
        label=f"properties/{label}/{keys[0][2]}",
        implementation=STUDY.implementation,
        cell_of=lambda key: str(key[0]),
        replicate_of=lambda key: int(key[1]),
        validate=invalid_rows,
    )
    rows = [row for value in outcomes.values for row in value]
    failed = len(outcomes.failed)
    if failed:
        print(
            f"{label}: {failed} of {len(keys)} replicates recorded a fitting failure; "
            f"{allowed_failures(keys[0][2])} may fail before its cells are red:\n"
            f"{outcomes.failures.to_string(index=False)}",
            flush=True,
        )
        for row in rows:
            row["failed_replicates"] = failed
    return rows


def plan(budget: int | None = None) -> dict[str, list[tuple[str, int, int]]]:
    """The keys of every fit set and stability law, in run order; ``budget`` caps each."""
    out: dict[str, list[tuple[str, int, int]]] = {}
    for fit_set, (_, _, _, replicates) in FIT_SETS.items():
        requested = replicates if budget is None else min(budget, replicates)
        out[fit_set] = [(fit_set, r, requested) for r in range(requested)]
    for law in ("d", "i"):
        requested = STABILITY_TRIALS if budget is None else min(budget, STABILITY_TRIALS)
        out[f"stability_{law}"] = [(law, r, requested) for r in range(requested)]
    return out


def fit_part(label: str, *, n_jobs: int, first: int | None = None) -> list[dict[str, Any]]:
    """Fit the first ``first`` replicates of one declared fit set or stability law.

    The keys are the declared run's own, so the checkpoints this writes are the ones
    :func:`generate_property_rows` reuses when the declared run reaches ``label``.
    """
    keys = plan()[label]
    if first is not None:
        if first < 1:
            raise ValueError(f"{label}: --first must be at least 1, not {first}")
        keys = keys[:first]
    worker = stability_rows if label.startswith("stability_") else fit_set_rows
    return _fit(label, worker, keys, n_jobs)


def shrunken_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """Each calibration label's positive rows with the reported SE times 0.70."""
    source = rows.loc[
        (rows["property"] == "interval_calibration")
        & rows["cell"].str.endswith("__correctly_specified")
    ].copy()
    source["cell"] = source["cell"].str.replace(
        "__correctly_specified", "__shrunken_se_control", regex=False
    )
    source["role"] = "control"
    source["std_error"] *= SHRUNKEN_SE_FACTOR
    return _recover(source)


def inflated_control(rows: pd.DataFrame) -> pd.DataFrame:
    """The horizon-3 median rows with the reported SE times 1.05."""
    source = rows.loc[
        (rows["property"] == "repeat_increment")
        & (rows["cell"] == f"h{INCREMENT_HORIZON}__repeat_median")
    ].copy()
    source["cell"] = f"h{INCREMENT_HORIZON}__inflated_se_control"
    source["role"] = "control"
    source["std_error"] *= INFLATED_SE_FACTOR
    return _recover(source)


def _recover(rows: pd.DataFrame) -> pd.DataFrame:
    half = CRITICAL * rows["std_error"]
    rows["covered"] = (
        (rows["estimate"] - half <= rows["truth"]) & (rows["truth"] <= rows["estimate"] + half)
    ).astype(int)
    rows["rejected"] = (np.abs(rows["estimate"] / rows["std_error"]) > CRITICAL).astype(int)
    return rows


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, only: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each set and ``only`` selects sets."""
    rows: list[dict[str, Any]] = []
    for label, keys in plan(budget).items():
        if only and label not in only:
            continue
        worker = stability_rows if label.startswith("stability_") else fit_set_rows
        rows.extend(_fit(label, worker, keys, n_jobs))
    frame = pd.DataFrame(rows)
    derived = [
        derive(frame)
        for derive, family in (
            (shrunken_controls, "interval_calibration"),
            (inflated_control, "repeat_increment"),
        )
        if (frame["property"] == family).any()
    ]
    frame = pd.concat([frame, *derived], ignore_index=True)
    return frame.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ----------------------------------------------------------------------------- verdicts


def increment_statistic(median: np.ndarray, single: np.ndarray) -> float:
    """Median SE ratio minus single-draw SE ratio; each array holds ``(estimate, SE)`` rows."""
    return float(
        median[:, 1].mean() / median[:, 0].std(ddof=1)
        - single[:, 1].mean() / single[:, 0].std(ddof=1)
    )


def increment_interval(
    median: pd.DataFrame, single: pd.DataFrame, *, seed: int
) -> tuple[float, float, float]:
    """The paired increment and its 99% bootstrap interval, resampling replications jointly."""
    margins = STUDY.margins
    left = median.sort_values("replicate")
    right = single.sort_values("replicate")
    if not np.array_equal(left["replicate"].to_numpy(), right["replicate"].to_numpy()):
        raise ValueError("the increment arms are not paired on replication")
    values = np.column_stack(
        [
            left[["estimate", "std_error"]].to_numpy(float),
            right[["estimate", "std_error"]].to_numpy(float),
        ]
    )
    point = increment_statistic(values[:, :2], values[:, 2:])
    rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [
            block[:, :, 1].mean(axis=1) / block[:, :, 0].std(axis=1, ddof=1)
            - block[:, :, 3].mean(axis=1) / block[:, :, 2].std(axis=1, ddof=1)
            for block in bootstrap_draw_blocks(
                values, replicates=margins.bootstrap_replicates, rng=rng
            )
        ]
    )
    interval = percentile_interval(draws, confidence_level=margins.confidence_level)
    return point, interval.low, interval.high


def increment_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The gated horizon-3 pair, its control, and the reported horizons."""
    family = summary["property"] == "repeat_increment"
    selected = rows.loc[rows["property"] == "repeat_increment"]

    def arm(cell: str) -> pd.DataFrame:
        return selected.loc[selected["cell"] == cell]

    gate: bool | None = None
    for horizon in study.HORIZONS:
        median, single = arm(f"h{horizon}__repeat_median"), arm(f"h{horizon}__single_draw")
        cells = [f"h{horizon}__repeat_median", f"h{horizon}__single_draw"]
        arms = [(cells, median)]
        if horizon == INCREMENT_HORIZON:
            control = f"h{horizon}__inflated_se_control"
            arms.append(([control], arm(control)))
        for names, numerator in arms:
            point, low, high = increment_interval(
                numerator, single, seed=stream_seed(STUDY, "repeat_increment", names[0])
            )
            mask = family & summary["cell"].isin(names)
            summary.loc[mask, "increment"] = point
            summary.loc[mask, "increment_ci_lower"] = low
            summary.loc[mask, "increment_ci_upper"] = high
            summary.loc[mask, "increment_ceiling"] = INCREMENT_CEILING
            summary.loc[mask, "increment_se_part"] = float(
                numerator["std_error"].mean() / single["std_error"].mean() - 1.0
            )
            summary.loc[mask, "increment_spread_part"] = float(
                single["estimate"].std(ddof=1) / numerator["estimate"].std(ddof=1) - 1.0
            )
            if horizon != INCREMENT_HORIZON:
                summary.loc[mask, "passed"] = True
            elif names[0].endswith("inflated_se_control"):
                summary.loc[mask, "passed"] = bool(high >= INCREMENT_CEILING)
            else:
                gate = bool(high < INCREMENT_CEILING)
                summary.loc[mask, "passed"] = gate
    reported = family & (summary["role"] == DIAGNOSTIC_ROLE)
    summary.loc[reported, "property_passed"] = True
    gated = family & (summary["role"] != DIAGNOSTIC_ROLE)
    summary.loc[gated, "property_passed"] = bool(gate) and bool(summary.loc[gated, "passed"].all())


def stability_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The paired spread-ratio rule of each stability label."""
    margins = STUDY.margins
    family = summary["property"] == "repeat_stability"
    selected = rows.loc[rows["property"] == "repeat_stability"]
    for label in STABILITY:
        three = selected.loc[selected["cell"] == f"{label}__three_repeats"]
        one = selected.loc[selected["cell"] == f"{label}__one_repeat_control"]
        result = paired_spread_ratio_interval(
            three,
            one,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "repeat_stability", "bootstrap", label),
        )
        mask = family & summary["cell"].str.startswith(f"{label}__")
        # A one-draw spread at rounding level makes the ratio a ratio of noise (rule 7).
        spread = float(one["estimate"].var(ddof=1) / np.mean(one["std_error"] ** 2))
        passed = bool(
            result.interval.high < MAX_REPEAT_SPREAD_RATIO and spread > STABILITY_SPREAD_FLOOR
        )
        if spread <= STABILITY_SPREAD_FLOOR:
            print(f"{label}: one-draw split variance share {spread:.3g} is rounding", flush=True)
        summary.loc[mask, "spread_ratio"] = result.ratio
        summary.loc[mask, "spread_ratio_ci_lower"] = result.interval.low
        summary.loc[mask, "spread_ratio_ci_upper"] = result.interval.high
        summary.loc[mask, "spread_ratio_boundary"] = MAX_REPEAT_SPREAD_RATIO
        summary.loc[mask, "passed"] = passed
        summary.loc[mask, "property_passed"] = passed


def diagnostic_readings(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The draw-0 calibration statistics and the displacement shares, reported only."""
    margins = STUDY.margins
    for family in ("single_draw_reading", "split_displacement"):
        mask = summary["property"] == family
        for index in summary.index[mask]:
            cell = str(summary.loc[index, "cell"])
            group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
            if family == "single_draw_reading":
                ratio = ratio_intervals(
                    group,
                    replicates=margins.bootstrap_replicates,
                    confidence_level=margins.confidence_level,
                    seed=stream_seed(STUDY, family, cell),
                )["se_ratio"]
                summary.loc[index, "se_ratio_ci_lower"] = ratio.low
                summary.loc[index, "se_ratio_ci_upper"] = ratio.high
            else:
                summary.loc[index, SHARE_COLUMN] = float(group["estimate"].mean())
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


def ipsi_primary_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Law I's primary rows, read by the three rules every primary row answers to."""
    margins = STUDY.margins
    family = summary["property"] == "ipsi_primary"
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == "ipsi_primary") & (rows["cell"] == cell)]
        ratio = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "ipsi_primary", cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        summary.loc[index, "passed"] = bool(
            summary.loc[index, "bias_equivalent"]
            and summary.loc[index, "coverage_ci_lower"] >= margins.coverage_floor
            and ratio.within(*margins.se_ratio_sanity)
        )


def failure_counts(rows: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    """``(requested, failed)`` of each ``(property, cell)``, checked against its rows.

    A cell whose rows and failures do not add up to its requested count lost a replicate that
    no failure record accounts for, which is a defect of the run, not a failure.
    """
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for (family, cell), group in rows.groupby(["property", "cell"], sort=True):
        requested = group["requested_replicates"].unique()
        failed = group["failed_replicates"].unique()
        if len(requested) != 1 or len(failed) != 1:
            raise ValueError(f"{family}/{cell} has inconsistent replication counts")
        total, lost = int(requested[0]), int(failed[0])
        if len(group) + lost != total:
            raise ValueError(
                f"{family}/{cell} has {len(group)} rows and {lost} recorded failures of {total} "
                f"replicates; the difference is not accounted for"
            )
        counts[(str(family), str(cell))] = (total, lost)
    return counts


def survivors(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows a summary reads: each cell's survivors as a complete table of their own.

    Every cell of one fit set loses the same replicates, so renumbering each cell's survivors
    in replicate order keeps every pairing.  With no failure the rows come back unchanged.
    """
    out = rows.copy()
    keys = ["property", "cell"]
    out["replicate"] = (
        out.groupby(keys)["replicate"].rank(method="first").astype(rows["replicate"].dtype) - 1
    )
    out["requested_replicates"] = (
        out.groupby(keys)["replicate"].transform("size").astype(rows["requested_replicates"].dtype)
    )
    out["failed_replicates"] = 0
    out["failed_replicates"] = out["failed_replicates"].astype(rows["failed_replicates"].dtype)
    return out


def apply_failure_bound(
    summary: pd.DataFrame, counts: dict[tuple[str, str], tuple[int, int]]
) -> pd.DataFrame:
    """Publish each cell's failures, and turn red every cell whose fit set lost too many.

    A ``root_n_rate`` row reads every rung of its ladder, so it takes the rungs' failures and
    is red when any rung is.
    """
    summary = summary.copy()
    over: set[tuple[str, str]] = set()
    for (family, cell), (requested, failed) in counts.items():
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "failed_replicates"] = failed
        if failed > allowed_failures(requested):
            over.add((family, cell))
            summary.loc[mask, ["passed", "property_passed"]] = False
    for index in summary.index[summary["property"] == "root_n_rate"]:
        label = str(summary.loc[index, "cell"]).split("__", 1)[0]
        rungs = [
            key
            for key in counts
            if key[0] == "root_n_and_efficiency" and key[1].startswith(f"{label}__")
        ]
        summary.loc[index, "failed_replicates"] = sum(counts[key][1] for key in rungs)
        if any(key in over for key in rungs):
            summary.loc[index, ["passed", "property_passed"]] = False
    return summary


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules, with failures accounted.

    Each cell is summarized on its survivors (:func:`survivors`), and
    :func:`apply_failure_bound` then publishes the failures and applies
    :data:`FAILURE_BOUND`.
    """
    counts = failure_counts(rows)
    return apply_failure_bound(_summarize(survivors(rows)), counts)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    require_complete(rows)
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(*INCREMENT_COLUMNS, *STABILITY_COLUMNS, SHARE_COLUMN),
        rate_labels=study.RATE_LABELS,
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    stability_verdicts(summary, rows)
    increment_verdicts(summary, rows)
    diagnostic_readings(summary, rows)
    ipsi_primary_verdicts(summary, rows)
    return finish(summary, rates)


__all__ = [
    "BANDS",
    "CALIBRATION",
    "FIT_SETS",
    "PASS_PROBABILITIES",
    "RED_CELL_POLICY",
    "apply_failure_bound",
    "failure_counts",
    "generate_property_rows",
    "plan",
    "summarize_properties",
    "survivors",
]
