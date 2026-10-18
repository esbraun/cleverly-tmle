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
``simultaneous_coverage``   the shipped band of its own fits and their pointwise control,
                            2,400 replications each: D-``LTMLE`` and D-``SDR`` (five names
                            each, ``band_d``), I (five, ``band_i``, n = 80,000), S (nine),
                            C (three) and M (the two centred coefficients)
``repeat_stability``        one fixed law-D sample (n = 8,000) and one fixed law-I sample
                            (n = 80,000), 400 paired fold-seed trials: the three-draw median
                            and the first draw it derives, alone
``repeat_increment``        its own law-S fits (``increment_s``, 2,800 replications) at
                            horizons 1 to 3: median SE ratio minus draw-0 SE ratio on the
                            same fits; horizon 3 is gated, horizons 1 and 2 are reported;
                            ``inflated_se_control`` multiplies the median SE by 1.05
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

**Sample streams.**  A gated family draws its own samples.  ``interval_calibration`` reads the
``calibration_*`` fits; ``single_draw_reading`` and ``split_displacement`` read the same fits,
because they are reported without a verdict.  ``simultaneous_coverage`` reads the ``band_*``
fits and ``repeat_increment`` reads ``increment_s``, each on a stream of its own
(:func:`sample_seed` names the fit set).  The first declared run read the band and the
increment off the calibration fits.  This declaration replaces them, and the committed rows of
that run size the new replication counts (:data:`BAND_REPLICATES`,
:data:`INCREMENT_REPLICATES`; ``tests/unit/test_longitudinal_repeated_crossfit_design.py``).

**Rules.**  ``interval_calibration``, ``simultaneous_coverage``, ``root_n_and_efficiency`` and
``root_n_rate`` read the shared verdicts.  ``repeat_stability`` passes when the 99% upper
endpoint of the paired spread ratio is below 1 (:mod:`tests.studies.repeated_crossfit_properties`
reads the same rule) and the one-draw split variance is above :data:`STABILITY_SPREAD_FLOOR`.
``repeat_increment`` passes when the 99% upper endpoint of the paired increment is below
:data:`INCREMENT_CEILING`; its control passes when its own endpoint is at or above it.
``ipsi_primary`` reads the three primary rules: the bias interval inside the margin, the coverage
lower endpoint above the floor, and the SE-ratio interval inside the sanity band.

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

**Failures.**  Every loop fits through
:func:`tests.studies.evidence.replication.fit_properties` (RM39b2).  A fit that raises a
recordable error, or returns a row with a non-finite estimate or a non-positive standard error,
is recorded and costs every cell of :func:`cells_of` one replicate; the run goes on.  The
regeneration driver stamps ``failed_replicates`` from the records, summarizes the survivors and
reads each cell under :data:`FAILURE_BOUND` by the worst case of its clauses in
:data:`PROPERTY_RULES` or in the shared rule of its family
(``property-failure-accounting.csv``).  A checkpointed failure replays on
resume, so the summary can be re-read without refitting.

**Declared red-cell reading.**  :data:`RED_CELL_POLICY`, declared before the run; the owner of
every reported red cell is :data:`RED_CELL_OWNER`.  :data:`PASS_PROBABILITIES` states the
declared pass probability that a verdict is read beside.

**Resuming.**  Every fit set goes through
:func:`tests.studies.evidence.replication.fit_properties`, which checkpoints each replicate
under the regeneration driver's scratch.  A stopped run resumes at the replicates it had not
finished.  :func:`fit_part` fits one fit set, or its first replicates, into the same
checkpoints, so ``tests/canonical/longitudinal_repeated_crossfit/part.py`` can run the study in
parts before the declared run collects them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.learners.crossfit import CrossFitPlan
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies.evidence import property_failures, property_verdicts, replication
from tests.studies.evidence.failures import filled, vertex
from tests.studies.evidence.inference import Interval, percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    bootstrap_draw_blocks,
    paired,
    paired_spread_ratio_interval,
    ratio_intervals,
    replicate_row,
    require_accounted,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_cells, joint_coverage_rows

STUDY = study.STUDY

#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

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
    "3. Calibration diagnosis: compare the label's single_draw_reading row with its "
    "median row, read its split_displacement share, and compare both with the pilot reading in "
    "tests/diagnostics/x46_sizing/reading.csv. A red band row is compared with the calibration "
    "median rows of its law and with the pilot rates in tests/diagnostics/x46_sizing/"
    "rerun_pilot.csv; the band reads its own samples, so no draw-0 row shares them. A red median row whose draw-0 row is green is a "
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
    "9. A recorded fitting failure costs every cell its fit writes one replicate, and the run "
    "goes on. It is a defect suspect: the records in property-failures.csv are diagnosed, and "
    "the summary is re-read from the checkpoints without refitting. A cell that lost at most "
    "floor(0.01 * replicates) replicates (FAILURE_BOUND) keeps its verdict only when every "
    "declared clause (PROPERTY_RULES or the shared rule of its family) holds at its worst "
    "case in property-failure-accounting.csv; one more is bound exceeded and red. The primary "
    "phase writes failures.csv and failure-accounting.csv under STUDY.failure_bound 0.01 "
    "(RM39 a2). A red cell with no defect found is reported with owner "
    f"{RED_CELL_OWNER}.",
    "10. A control that does not fail where it is declared to have power is a design defect "
    "and is fixed before any rerun.",
    "11. Each verdict is read beside its declared pass probability (PASS_PROBABILITIES). The "
    "nine calibration cells pass jointly with probability 1.00 at the pilot reading and 0.89 "
    "predictively, with the pilot's correlated errors. A red calibration row therefore has "
    "probability about 0.11 and is read by rules 3 to 5. The five band arms and their controls "
    "pass jointly with probability 0.98 predictively at 2,400 replications, and the "
    "repeat_increment gate with 0.96 at 2,800, by the arm-pooled predictive rule.",
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
    "simultaneous_coverage/band_family": 0.98,
    "simultaneous_coverage/d_ltmle": 1.00,
    "simultaneous_coverage/d_sdr": 1.00,
    "simultaneous_coverage/i_ipsi": 1.00,
    "simultaneous_coverage/s_ltmle": 1.00,
    "simultaneous_coverage/c_ltmle": 1.00,
    "simultaneous_coverage/m_ltmle": 1.00,
    "repeat_increment/h3": 0.96,
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
#: The replications of each ``simultaneous_coverage`` fit set and of the ``repeat_increment`` fit
#: set.  Each family draws its own samples, so each is sized by its own rule
#: (``tests/unit/test_longitudinal_repeated_crossfit_design.py``).
BAND_REPLICATES = 2_400
INCREMENT_REPLICATES = 2_800
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
    "d_ltmle": ("band_d", "law_d_ltmle", (*study.D_MEANS, *study.D_CONTRASTS.values())),
    "d_sdr": ("band_d", "law_d_sdr", (*study.D_MEANS, *study.D_CONTRASTS.values())),
    "i_ipsi": ("band_i", "law_i", study.I_NAMES),
    "s_ltmle": ("band_s", "law_s", study.S_NAMES),
    "c_ltmle": ("band_c", "law_c", study.C_NAMES),
    "m_ltmle": ("band_m", "law_m", study.M_NAMES),
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
    "band_d": ("d", ("law_d_ltmle", "law_d_sdr"), CALIBRATION_N, BAND_REPLICATES),
    "band_i": ("i", ("law_i",), IPSI_N, BAND_REPLICATES),
    "band_s": ("s", ("law_s",), CALIBRATION_N, BAND_REPLICATES),
    "band_c": ("c", ("law_c",), CALIBRATION_N, BAND_REPLICATES),
    "band_m": ("m", ("law_m",), CALIBRATION_N, BAND_REPLICATES),
    "increment_s": ("s", ("law_s",), CALIBRATION_N, INCREMENT_REPLICATES),
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
    "one_draw_spread_share",
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
    return rows


def _band_rows(
    fit_set: str, scenario: str, result: Any, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    """The joint rows of every band label this fit set owns: the shipped band and its control."""
    truth = study.I_TRUTH if scenario == "law_i" else study.TRUTHS[scenario]
    rows: list[dict[str, Any]] = []
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


def _increment_rows(
    scenario: str, result: Any, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    """The median and the first draw of each horizon's risk difference on law S."""
    first = result.repeats[0].estimates
    truth = study.TRUTHS[scenario]
    rows: list[dict[str, Any]] = []
    for horizon in study.HORIZONS:
        name = study.s_contrast(horizon)
        role = "positive" if horizon == INCREMENT_HORIZON else DIAGNOSTIC_ROLE
        for arm, estimate in (("repeat_median", result[name]), ("single_draw", first[name])):
            rows.append(
                replicate_row(
                    property_name="repeat_increment",
                    cell=f"h{horizon}__{arm}",
                    role=role,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=float(truth[name]),
                    estimate=estimate,
                    alpha=STUDY.margins.alpha,
                )
            )
    return rows


def _rate_label(scenario: str) -> str:
    return "ltmle_static" if scenario == "law_d_ltmle" else "sdr_static"


def _rate_rows(
    scenario: str, result: Any, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    label = _rate_label(scenario)
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


def fit_set_rows(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one replicate of one fit set publishes: ``key = (fit set, replicate, R)``."""
    fit_set, replicate, requested = key
    _, scenarios, n, _ = FIT_SETS[fit_set]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = study.fit_scenario(
            scenario, frame.copy(), simultaneous=fit_set.startswith("band_")
        )
        if result.n_repeats != study.REPEATS:
            raise RuntimeError(f"{fit_set}: the fit drew {result.n_repeats} splits")
        if fit_set.startswith("calibration_"):
            rows.extend(_calibration_rows(fit_set, scenario, result, replicate, n, requested))
        elif fit_set.startswith("band_"):
            rows.extend(_band_rows(fit_set, scenario, result, replicate, n, requested))
        elif fit_set.startswith("increment_"):
            rows.extend(_increment_rows(scenario, result, replicate, n, requested))
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


def stability_rows(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One paired trial on law ``key[0]``'s fixed sample: three draws, and the first alone."""
    law, trial, requested = key
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

#: Each base cell to the controls derived from its rows, which a lost base replicate also
#: loses: the shrunken-SE control of each calibration label and the inflated-SE control of the
#: gated increment horizon.
DERIVED_CELLS: dict[str, tuple[str, ...]] = {
    **{
        f"interval_calibration/{label}__correctly_specified": (
            f"interval_calibration/{label}__shrunken_se_control",
        )
        for label in CALIBRATION
    },
    f"repeat_increment/h{INCREMENT_HORIZON}__repeat_median": (
        f"repeat_increment/h{INCREMENT_HORIZON}__inflated_se_control",
    ),
}


def draw(key: tuple[str, int, int]) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone.

    A fit-set key draws its replicate's sample.  A stability key draws its law's one fixed
    sample, which every trial of the law holds.
    """
    name, replicate, _ = key
    if name in FIT_SETS:
        law, _, n, _ = FIT_SETS[name]
        return study.sample(law, n, sample_seed(name, replicate))
    return stability_sample(name)


def fit_rows(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of one key: a fit-set replicate, or a stability trial."""
    if key[0] in FIT_SETS:
        return fit_set_rows(key, frame)
    return stability_rows(key, frame)


def cells_of(key: tuple[str, int, int]) -> list[str]:
    """The ``property/cell`` names one key's rows and derived controls fill."""
    name = key[0]
    if name not in FIT_SETS:
        return [
            f"repeat_stability/{label}__{cell}"
            for label, (law, _, _) in STABILITY.items()
            if law == name
            for cell in ("three_repeats", "one_repeat_control")
        ]
    _, scenarios, n, _ = FIT_SETS[name]
    names: list[str] = []
    for scenario in scenarios:
        if name.startswith("calibration_"):
            for label, (owner, owner_scenario, _) in CALIBRATION.items():
                if (owner, owner_scenario) == (name, scenario):
                    names += [
                        f"interval_calibration/{label}__correctly_specified",
                        f"single_draw_reading/{label}__draw_0",
                        f"split_displacement/{label}__share",
                    ]
        elif name.startswith("band_"):
            for label, (owner, owner_scenario, _) in BANDS.items():
                if (owner, owner_scenario) == (name, scenario):
                    names += [f"{JOINT}/{cell}" for cell in joint_cells(label)]
        elif name.startswith("increment_"):
            names += [
                f"repeat_increment/h{horizon}__{arm}"
                for horizon in study.HORIZONS
                for arm in ("repeat_median", "single_draw")
            ]
        elif name.startswith("rate_"):
            names.append(f"root_n_and_efficiency/{_rate_label(scenario)}__n_{n}")
        else:
            names += [f"ipsi_primary/{label}__repeated" for label in study.IPSI_PRIMARY_LABELS]
    return [*names, *(control for base in names for control in DERIVED_CELLS.get(base, ()))]


def _cell_of(key: tuple[str, int, int]) -> str:
    return str(key[0])


def _replicate_of(key: tuple[str, int, int]) -> int:
    return int(key[1])


def _fit(label: str, keys: list[tuple[str, int, int]], n_jobs: int) -> list[dict[str, Any]]:
    """The rows of every key that succeeded; a recorded failure costs its key's cells."""
    values = fit_properties(
        keys,
        fit_rows,
        draw=draw,
        cells_of=cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        # The requested count joins the label, so a run at another budget never reuses these.
        label=f"properties/{label}_r{keys[0][2]}",
        derived=DERIVED_CELLS,
    )
    return [row for value in values for row in value]


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

    The keys, the subject and the checkpoint label are the declared run's own, so the
    checkpoints this writes are the ones :func:`generate_property_rows` reuses when the
    declared run reaches ``label``.  The part opens its own record collector, so a recorded
    failure is printed, checkpointed and replayed by the declared run, and does not stop the
    part.
    """
    keys = plan()[label]
    if first is not None:
        if first < 1:
            raise ValueError(f"{label}: --first must be at least 1, not {first}")
        keys = keys[:first]
    with replication.recording(STUDY.implementation, witness=PROPERTY_WITNESS) as recorded:
        rows = _fit(label, keys, n_jobs)
    failures = recorded.failures
    if failures is not None:
        print(
            f"{label}: {len(failures)} recorded property failures; the declared run replays "
            f"them from the checkpoints:\n{failures.to_string(index=False)}",
            flush=True,
        )
    return rows


class _CellLaw:
    """The exact truth of one declared cell, read from the study's own law tables.

    The estimand of a declared cell is the cell's own name, so :meth:`truth` answers for it.
    """

    def __init__(self, name: str, estimand: str, value: float) -> None:
        self.name = name
        self._truth = {estimand: value}

    def truth(self) -> dict[str, float]:
        """The exact value of the one estimand this cell reads."""
        return dict(self._truth)


def exact_truth(family: str, cell: str) -> float:
    """The exact-law value a cell's rows must publish as their truth.

    A joint band and a displacement share have no scalar estimand, so their rows publish 0.
    """
    label = cell.split("__", 1)[0]
    if family in {JOINT, "split_displacement"}:
        return 0.0
    if family == "ipsi_primary":
        return float(study.I_TRUTH[study.IPSI_PRIMARY_LABELS[label]])
    if family == "repeat_increment":
        return float(study.TRUTHS["law_s"][study.s_contrast(int(label.removeprefix("h")))])
    if family == "root_n_and_efficiency":
        return float(study.TRUTHS["law_d_ltmle"][study.D_CONTRASTS["static"]])
    _, scenario, estimand = {**CALIBRATION, **STABILITY}[label]
    truth = study.I_TRUTH if scenario == "law_i" else study.TRUTHS[scenario]
    return float(truth[estimand])


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its role, size, replications and sample stream.

    The derived controls (:data:`DERIVED_CELLS`) are here, because their rows are published.
    The ``root_n_rate`` rows are computed from the root-n cells' rows, so they are not.
    """
    out: list[PropertyCell] = []
    for keys in plan().values():
        key = keys[0]
        law = FIT_SETS[key[0]][0] if key[0] in FIT_SETS else key[0]
        n = FIT_SETS[key[0]][2] if key[0] in FIT_SETS else STABILITY_N[key[0]]
        seed = (
            sample_seed(key[0], 0)
            if key[0] in FIT_SETS
            else stream_seed(STUDY, "repeat_stability", "sample", key[0])
        )
        for name in cells_of(key):
            family, cell = name.split("/", 1)
            out.append(
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=_CellLaw(f"law_{law}", cell, exact_truth(family, cell)),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=n,
                    replicates=key[2],
                    seed=seed,
                    role=_role(family, cell, n),
                    estimand=cell,
                )
            )
    return tuple(out)


def _role(family: str, cell: str, n: int) -> str:
    """The role a cell's rows carry, as the row builders above write it."""
    if family in {"single_draw_reading", "split_displacement"}:
        return DIAGNOSTIC_ROLE
    if family == "repeat_increment":
        return "positive" if cell.startswith(f"h{INCREMENT_HORIZON}__") else DIAGNOSTIC_ROLE
    if family == "root_n_and_efficiency":
        return "control" if n == min(RATE_SIZES) else "positive"
    if cell.endswith(("__one_repeat_control", "__pointwise_joint_control")):
        return "control"
    return "positive"


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
        rows.extend(_fit(label, keys, n_jobs))
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
    """The paired increment and its 99% bootstrap interval, resampling replications jointly.

    The two arms read the replications both kept (:func:`~tests.studies.evidence.properties.paired`).
    """
    margins = STUDY.margins
    median, single = paired(median, single)
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


def _increment_seed(record: Any, numerator: str) -> int:
    return stream_seed(record, "repeat_increment", numerator)


def increment_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The gated horizon-3 pair, its control, and the reported horizons."""
    family = summary["property"] == "repeat_increment"
    selected = rows.loc[rows["property"] == "repeat_increment"]
    if selected.empty:
        return

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
                numerator, single, seed=_increment_seed(STUDY, names[0])
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


def one_draw_spread_share(one: pd.DataFrame) -> float:
    """The one-draw split variance as a share of the mean reported variance (rule 7)."""
    return float(one["estimate"].var(ddof=1) / np.mean(one["std_error"] ** 2))


def _stability_seed(record: Any, label: str) -> int:
    return stream_seed(record, "repeat_stability", "bootstrap", label)


def stability_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The paired spread-ratio rule of each stability label, above the spread floor."""
    margins = STUDY.margins
    family = summary["property"] == "repeat_stability"
    selected = rows.loc[rows["property"] == "repeat_stability"]
    if selected.empty:
        return
    # One fixed sample per label: the rows answer the spread rule only, so they publish no
    # population-sampling endpoint (bias, coverage, SE ratio) beside it.
    kept = {
        "property",
        "cell",
        "role",
        "n",
        "replicates",
        "failed_replicates",
        *STABILITY_COLUMNS,
        "passed",
        "property_passed",
    }
    for column in summary.columns:
        if column not in kept:
            if summary[column].dtype == bool:
                summary[column] = summary[column].astype(object)
            summary.loc[family, column] = np.nan
    for label in STABILITY:
        three = selected.loc[selected["cell"] == f"{label}__three_repeats"]
        one = selected.loc[selected["cell"] == f"{label}__one_repeat_control"]
        result = paired_spread_ratio_interval(
            three,
            one,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=_stability_seed(STUDY, label),
        )
        mask = family & summary["cell"].str.startswith(f"{label}__")
        # A one-draw spread at rounding level makes the ratio a ratio of noise (rule 7).
        spread = one_draw_spread_share(one)
        passed = bool(
            result.interval.high < MAX_REPEAT_SPREAD_RATIO and spread > STABILITY_SPREAD_FLOOR
        )
        if spread <= STABILITY_SPREAD_FLOOR:
            print(f"{label}: one-draw split variance share {spread:.3g} is rounding", flush=True)
        summary.loc[mask, "spread_ratio"] = result.ratio
        summary.loc[mask, "spread_ratio_ci_lower"] = result.interval.low
        summary.loc[mask, "spread_ratio_ci_upper"] = result.interval.high
        summary.loc[mask, "spread_ratio_boundary"] = MAX_REPEAT_SPREAD_RATIO
        summary.loc[mask, "one_draw_spread_share"] = spread
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


def _ipsi_seed(record: Any, cell: str) -> int:
    return stream_seed(record, "ipsi_primary", cell)


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
            seed=_ipsi_seed(STUDY, cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        summary.loc[index, "passed"] = bool(
            summary.loc[index, "bias_equivalent"]
            and summary.loc[index, "coverage_ci_lower"] >= margins.coverage_floor
            and ratio.within(*margins.se_ratio_sanity)
        )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules, on the surviving rows.

    A cell may be short only by its stamped ``failed_replicates``
    (:func:`~tests.studies.evidence.properties.require_accounted`).  The regeneration driver
    then reads each cell that lost a replicate under :data:`FAILURE_BOUND` and
    :data:`PROPERTY_RULES`.
    """
    require_accounted(rows)
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


# ---------------------------------------------------------------- the declared rules (RM39b2)


def _complete_arm(
    group: pd.DataFrame, estimates: np.ndarray, std_errors: np.ndarray
) -> pd.DataFrame:
    """One arm with every requested replicate: survivors real, each lost id at its fill.

    ``estimates`` and ``std_errors`` hold the survivors in the group's order, then one value per
    lost id in ascending id order.  The frame carries complete stamps, so the paired helpers
    read it as a complete arm.
    """
    requested = int(group["requested_replicates"].iloc[0])
    survived = [int(value) for value in group["replicate"]]
    missing = sorted(set(range(requested)) - set(survived))
    truth = group["truth"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "property": group["property"].iloc[0],
            "cell": group["cell"].iloc[0],
            "role": group["role"].iloc[0],
            "replicate": survived + missing,
            "n": int(group["n"].iloc[0]),
            "requested_replicates": requested,
            "failed_replicates": 0,
            "truth": np.concatenate([truth, np.full(len(missing), float(np.mean(truth)))]),
            "estimate": estimates,
            "std_error": std_errors,
        }
    )
    return frame.sort_values("replicate", ignore_index=True)


def _fills(group: pd.DataFrame, *, spread: bool, se_at: str) -> list[tuple[str, pd.DataFrame]]:
    """An arm's candidate completions for a statistic of ``mean(SE) / sd(estimate)`` type.

    ``spread`` puts the lost estimates at each vertex split between the survivor minimum and
    maximum, which a standard deviation, convex in its fills, reaches its maximum at.  Without
    it they sit at the survivor mean, where the standard deviation is smallest.  The lost
    standard errors sit at the survivor ``min`` or ``max``.
    """
    k = int(group["requested_replicates"].iloc[0]) - len(group)
    estimates = group["estimate"].to_numpy(dtype=float)
    std_errors = group["std_error"].to_numpy(dtype=float)
    se_fill = float(np.max(std_errors) if se_at == "max" else np.min(std_errors))
    errors = filled(std_errors, k, se_fill)
    if spread:
        return [
            (
                f"SE at {se_at}, estimates j={j} at min",
                _complete_arm(group, vertex(estimates, k, j), errors),
            )
            for j in range(k + 1)
        ]
    return [
        (
            f"SE at {se_at}, estimates at mean",
            _complete_arm(group, filled(estimates, k, float(np.mean(estimates))), errors),
        )
    ]


def _stability_cells(row: pd.Series) -> tuple[str, str]:
    label = str(row["cell"]).split("__", 1)[0]
    return f"{label}__three_repeats", f"{label}__one_repeat_control"


def _stability_judged(interval: Interval, share: float) -> property_failures.Judged:
    """The spread-ratio upper endpoint below 1 and the one-draw share above the floor."""
    ratio_slack = MAX_REPEAT_SPREAD_RATIO - interval.high
    share_slack = math.log10(share / STABILITY_SPREAD_FLOOR) if share > 0 else -math.inf
    return property_failures.Judged(
        min(ratio_slack, share_slack),
        bool(interval.high < MAX_REPEAT_SPREAD_RATIO and share > STABILITY_SPREAD_FLOOR),
    )


def _stability_rule() -> property_failures.FamilyRule:
    """``repeat_stability``: S9 on each label's paired spread ratio, with the spread floor.

    A replicate is a fold-seed trial on one fixed sample, so the survivor range is the range
    over fold seeds.  The adverse completion puts the three-draw arm's lost estimates at each
    vertex split, and the one-draw arm's at the survivor mean with its SEs at the survivor
    maximum: that both raises the ratio and lowers the one-draw share toward the floor.  Both
    rows of a label publish the one verdict.
    """

    def read(ctx: property_failures.Context, row: pd.Series) -> property_failures.Judged:
        interval = Interval(
            float(row["spread_ratio_ci_lower"]), float(row["spread_ratio_ci_upper"])
        )
        return _stability_judged(interval, float(row["one_draw_spread_share"]))

    def reads(ctx: property_failures.Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(str(row["property"]), cell) for cell in _stability_cells(row)]

    def candidates(
        ctx: property_failures.Context, row: pd.Series
    ) -> list[tuple[str, property_failures.Judged]]:
        family = str(row["property"])
        three_cell, one_cell = _stability_cells(row)
        label = three_cell.split("__", 1)[0]
        ((one_label, one),) = _fills(ctx.cell(family, one_cell), spread=False, se_at="max")
        out = []
        for three_label, three in _fills(ctx.cell(family, three_cell), spread=True, se_at="max"):
            result = paired_spread_ratio_interval(
                three,
                one,
                replicates=ctx.margins.bootstrap_replicates,
                confidence_level=ctx.margins.confidence_level,
                seed=_stability_seed(ctx.record, label),
            )
            out.append(
                (
                    f"three repeats {three_label}; one repeat {one_label}",
                    _stability_judged(result.interval, one_draw_spread_share(one)),
                )
            )
        return out

    return property_failures.FamilyRule(
        families=frozenset({"repeat_stability"}),
        shared="stability_verdicts",
        clauses=(
            property_failures.Clause(
                "repeat spread ratio under its boundary above the spread floor",
                "S9",
                property_failures.every,
                read,
                candidates,
                reads=reads,
            ),
        ),
    )


def _increment_clause(name: str, *, control: bool) -> property_failures.Clause:
    """The gated increment (``control=False``) or its inflated-SE control, over S5-type fills.

    The increment is the numerator arm's SE ratio less the single-draw arm's.  The gated pair
    passes below :data:`INCREMENT_CEILING`, so its adverse completion raises the increment: the
    numerator's lost SEs at the survivor maximum with its estimates at the mean, and the single
    draw's SEs at the minimum with its estimates at each vertex split.  The control passes at
    or above the ceiling, and its adverse completion is the reverse.  The point increment is
    exact over the set; the bootstrap interval is enumerated.
    """
    horizon = f"h{INCREMENT_HORIZON}"
    numerator = f"{horizon}__inflated_se_control" if control else f"{horizon}__repeat_median"
    single = f"{horizon}__single_draw"
    applies = (
        property_failures.cells(numerator)
        if control
        else property_failures.cells(f"{horizon}__repeat_median", single)
    )

    def judge(high: float) -> property_failures.Judged:
        if control:
            return property_failures.Judged(
                high - INCREMENT_CEILING, bool(high >= INCREMENT_CEILING)
            )
        return property_failures.Judged(INCREMENT_CEILING - high, bool(high < INCREMENT_CEILING))

    def read(ctx: property_failures.Context, row: pd.Series) -> property_failures.Judged:
        return judge(float(row["increment_ci_upper"]))

    def reads(ctx: property_failures.Context, row: pd.Series) -> list[tuple[str, str]]:
        return [("repeat_increment", numerator), ("repeat_increment", single)]

    def candidates(
        ctx: property_failures.Context, row: pd.Series
    ) -> list[tuple[str, property_failures.Judged]]:
        top = _fills(
            ctx.cell("repeat_increment", numerator),
            spread=control,
            se_at="min" if control else "max",
        )
        bottom = _fills(
            ctx.cell("repeat_increment", single),
            spread=not control,
            se_at="max" if control else "min",
        )
        out = []
        for top_label, top_frame in top:
            for bottom_label, bottom_frame in bottom:
                _, _, high = increment_interval(
                    top_frame, bottom_frame, seed=_increment_seed(ctx.record, numerator)
                )
                out.append((f"numerator {top_label}; single draw {bottom_label}", judge(high)))
        return out

    return property_failures.Clause(name, "S5", applies, read, candidates, reads=reads)


def _increment_rule() -> property_failures.FamilyRule:
    """``repeat_increment``: the gated horizon-3 pair and its control share one verdict.

    Horizons 1 and 2 are reported rows and read no clause.
    """
    return property_failures.FamilyRule(
        families=frozenset({"repeat_increment"}),
        shared="increment_verdicts",
        clauses=(
            _increment_clause("increment under its ceiling", control=False),
            _increment_clause("control increment at or above the ceiling", control=True),
        ),
        group="family",
    )


def _ipsi_primary_rule() -> property_failures.FamilyRule:
    """``ipsi_primary``: the three primary rules, S1 bias, S2 coverage floor and S5 SE ratio."""

    def floor(ctx: property_failures.Context) -> float:
        return ctx.margins.coverage_floor

    def sanity(ctx: property_failures.Context) -> tuple[float, float]:
        return ctx.margins.se_ratio_sanity

    return property_failures.FamilyRule(
        families=frozenset({"ipsi_primary"}),
        shared="ipsi_primary_verdicts",
        clauses=(
            property_failures.bias_clause(
                "bias equivalent", kind="equivalent", applies=property_failures.every
            ),
            property_failures.proportion_clause(
                "coverage floor",
                statistic="S2",
                judge=property_failures.floor_judge(floor),
                applies=property_failures.every,
            ),
            property_failures.se_interval_clause(
                "se ratio in sanity band",
                which="se_ratio",
                judge=property_failures.band_judge(sanity),
                seed=lambda ctx, row: _ipsi_seed(ctx.record, str(row["cell"])),
                applies=property_failures.every,
            ),
        ),
    )


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  ``interval_calibration``,
#: ``simultaneous_coverage``, ``root_n_and_efficiency`` and ``root_n_rate`` read the shared
#: rules; ``single_draw_reading`` and ``split_displacement`` are reported rows with no verdict.
PROPERTY_RULES = (_stability_rule(), _increment_rule(), _ipsi_primary_rule())


__all__ = [
    "BANDS",
    "CALIBRATION",
    "DERIVED_CELLS",
    "FIT_SETS",
    "PASS_PROBABILITIES",
    "PROPERTY_RULES",
    "RED_CELL_POLICY",
    "cells_of",
    "declared_cells",
    "fit_part",
    "generate_property_rows",
    "plan",
    "summarize_properties",
]
