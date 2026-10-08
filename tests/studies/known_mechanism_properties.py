"""Repeated-sampling properties of TMLE on a declared known treatment mechanism.

Every number below was fixed before any registered run.

==============================  ============================================================
family                          cells, replications and size
==============================  ============================================================
``known_mechanism_accuracy``    On one two-arm draw per replication, n = 2,000 and 1,000
                                replications: the ATE with the correct and the wrong outcome
                                regression, the ATT and ATC with the wrong one, the stacked
                                CV-TMLE ATE with the wrong one, the incremental mean at
                                ``delta = 2``, and the control, the ATE of an intercept-only
                                estimated mechanism with the wrong outcome regression.  On a
                                separate three-arm draw, 1,000 replications: the three arm
                                means and both contrasts with the wrong outcome regression
``bootstrap_coverage``          The percentile interval of a 200-replicate full-refit bootstrap
                                of the ATE, wrong outcome regression, n = 500, 500 replications
``interval_calibration``        The ATE with the wrong outcome regression, n = 2,000, 2,000
                                replications, and its two derived controls
``variance_direction``          Reported, not gated: on the calibration draws, the same fit
                                with the mechanism estimated by a logistic regression on
                                ``W2``.  The row publishes the ratio of the two empirical
                                standard deviations with a 99% bootstrap interval
``root_n_and_efficiency``       The ATE with the wrong outcome regression at n = 500 (the
                                control rung), 2,000 and 8,000, 1,000 replications each
==============================  ============================================================

The accuracy rule of a positive cell is the bias margin, the coverage floor and the SE-ratio
sanity band together.  The control must establish its bias outside the margin; its
population bias is 0.0790, 15.6 times the margin (``known_mechanism_law.CONTROL_BIAS`` and
``control_sd``).  The
failure rule is the registry's: a replication that raises is never redrawn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_known_mechanism import STUDY, fit_cleverly
from tests.studies.evidence.inference import percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    bootstrap_draw_blocks,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

ACCURACY = "known_mechanism_accuracy"
BOOTSTRAP = "bootstrap_coverage"
CALIBRATION = "interval_calibration"
DIRECTION = "variance_direction"
RATE = "root_n_and_efficiency"

ACCURACY_N = 2_000
ACCURACY_REPLICATES = 1_000
BOOTSTRAP_N = 500
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_DRAWS = 200
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 2_000
RATE_SIZES = (500, 2_000, 8_000)
RATE_REPLICATES = 1_000
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The calibration cell's positive arm.  Not ``correctly_specified``: its outcome regression
#: is wrong on purpose, and the mechanism is what makes the interval valid.
CALIBRATION_ARM = "known_g_q_wrong"
#: The exact SD of ``D*(Qbar_inf, g0)`` at the wrong outcome regression's limit, which only
#: sizes the noise control (the record publishes no efficiency ratio).
NOISE_SD = 1.0732255

TWO_ARM = law.truth(2)
THREE_ARM = law.truth(3)
INCREMENTAL_NAME = f"ey_ipsi[odds x{law.DELTA:g}]"

#: ``cell -> (estimand reported by the fit, truth, role)`` on the two-arm draw.
TWO_ARM_CELLS: dict[str, tuple[str, float, str]] = {
    "ate__known_g__q_correct": ("ate", TWO_ARM["ate"], "positive"),
    "ate__known_g__q_wrong": ("ate", TWO_ARM["ate"], "positive"),
    "att__known_g__q_wrong": ("att", TWO_ARM["att"], "positive"),
    "atc__known_g__q_wrong": ("atc", TWO_ARM["atc"], "positive"),
    "ate__estimated_g_wrong__q_wrong": ("ate", TWO_ARM["ate"], "control"),
    "ate__known_g__q_wrong__cv": ("ate", TWO_ARM["ate"], "positive"),
    "ey_ipsi__known_g__incremental": (INCREMENTAL_NAME, TWO_ARM["ey_ipsi"], "positive"),
}
#: ``cell -> (estimand, truth)`` on the three-arm draw.
THREE_ARM_CELLS: dict[str, tuple[str, float]] = {
    "ey0__known_g__multi_arm__q_wrong": ("ey[0.0]", THREE_ARM["ey[0.0]"]),
    "ey1__known_g__multi_arm__q_wrong": ("ey[1.0]", THREE_ARM["ey[1.0]"]),
    "ey2__known_g__multi_arm__q_wrong": ("ey[2.0]", THREE_ARM["ey[2.0]"]),
    "ate_1_vs_0__known_g__multi_arm__q_wrong": ("ate[1.0 vs 0.0]", THREE_ARM["ate[1.0 vs 0.0]"]),
    "ate_2_vs_0__known_g__multi_arm__q_wrong": ("ate[2.0 vs 0.0]", THREE_ARM["ate[2.0 vs 0.0]"]),
}


def _seed(family: str, stream: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, stream, replicate)


def _row(
    cell: str, role: str, replicate: int, n: int, requested: int, truth: float, estimate: Any
) -> dict[str, Any]:
    return replicate_row(
        property_name=ACCURACY,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def _two_arm(replicate: int) -> list[dict[str, Any]]:
    frame = law.sample(ACCURACY_N, _seed(ACCURACY, "two_arm", replicate))
    fits = {
        "q_correct": fit_cleverly(frame, outcome="correct", estimands=("ate",)),
        "q_wrong": fit_cleverly(frame, outcome="wrong", estimands=("ate", "att", "atc")),
        "control": fit_cleverly(frame, outcome="wrong", mechanism="intercept", estimands=("ate",)),
        "cv": fit_cleverly(frame, outcome="wrong", estimands=("ate",), cross_fit=True),
        "incremental": fit_cleverly(frame, outcome="wrong", estimands=None, incremental=True),
    }
    source = {
        "ate__known_g__q_correct": "q_correct",
        "ate__known_g__q_wrong": "q_wrong",
        "att__known_g__q_wrong": "q_wrong",
        "atc__known_g__q_wrong": "q_wrong",
        "ate__estimated_g_wrong__q_wrong": "control",
        "ate__known_g__q_wrong__cv": "cv",
        "ey_ipsi__known_g__incremental": "incremental",
    }
    return [
        _row(
            cell, role, replicate, ACCURACY_N, ACCURACY_REPLICATES, truth, fits[source[cell]][name]
        )
        for cell, (name, truth, role) in TWO_ARM_CELLS.items()
    ]


def _three_arm(replicate: int) -> list[dict[str, Any]]:
    frame = law.sample(ACCURACY_N, _seed(ACCURACY, "three_arm", replicate), arms=3)
    result = fit_cleverly(frame, outcome="wrong", estimands=("ey", "ate"), arms=3)
    return [
        _row(cell, "positive", replicate, ACCURACY_N, ACCURACY_REPLICATES, truth, result[name])
        for cell, (name, truth) in THREE_ARM_CELLS.items()
    ]


def _bootstrap(replicate: int) -> list[dict[str, Any]]:
    seed = _seed(BOOTSTRAP, "percentile", replicate)
    frame = law.sample(BOOTSTRAP_N, seed)
    result = fit_cleverly(
        frame,
        outcome="wrong",
        estimands=("ate",),
        n_bootstrap=BOOTSTRAP_DRAWS,
        random_state=seed % (2**31),
    )
    summary = result.bootstrap.summary("ate", alpha=STUDY.margins.alpha)
    low, high = summary.ci
    truth = TWO_ARM["ate"]
    return [
        {
            "property": BOOTSTRAP,
            "cell": "ate__known_g__q_wrong__percentile",
            "role": "positive",
            "replicate": replicate,
            "n": BOOTSTRAP_N,
            "requested_replicates": BOOTSTRAP_REPLICATES,
            "failed_replicates": int(result.bootstrap.n_failed),
            "truth": truth,
            "estimate": float(result["ate"].psi),
            "std_error": float(summary.std_error),
            "covered": int(low <= truth <= high),
            "rejected": int(not (low <= 0.0 <= high)),
        }
    ]


def _calibration(replicate: int) -> list[dict[str, Any]]:
    frame = law.sample(CALIBRATION_N, _seed(CALIBRATION, "paired", replicate))
    truth = TWO_ARM["ate"]
    known = fit_cleverly(frame, outcome="wrong", estimands=("ate",))
    parametric = fit_cleverly(frame, outcome="wrong", mechanism="correct", estimands=("ate",))
    rows = []
    for family, cell, role, result in (
        (CALIBRATION, f"ate__{CALIBRATION_ARM}", "positive", known),
        (DIRECTION, "ate__known_mechanism", DIAGNOSTIC_ROLE, known),
        (DIRECTION, "ate__parametric_mechanism", DIAGNOSTIC_ROLE, parametric),
    ):
        row = replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            replicate=replicate,
            n=CALIBRATION_N,
            requested=CALIBRATION_REPLICATES,
            truth=truth,
            estimate=result["ate"],
            alpha=STUDY.margins.alpha,
        )
        rows.append(row)
    return rows


def _rate(payload: tuple[int, int]) -> list[dict[str, Any]]:
    size, replicate = payload
    frame = law.sample(size, _seed(RATE, f"n_{size}", replicate))
    result = fit_cleverly(frame, outcome="wrong", estimands=("ate",))
    return [
        replicate_row(
            property_name=RATE,
            cell=f"n_{size}",
            role="control" if size == min(RATE_SIZES) else "positive",
            replicate=replicate,
            n=size,
            requested=RATE_REPLICATES,
            truth=TWO_ARM["ate"],
            estimate=result["ate"],
            alpha=STUDY.margins.alpha,
        )
    ]


def _dispatch(payload: tuple[str, Any]) -> list[dict[str, Any]]:
    kind, argument = payload
    if kind == "two_arm":
        return _two_arm(argument)
    if kind == "three_arm":
        return _three_arm(argument)
    if kind == "bootstrap":
        return _bootstrap(argument)
    if kind == "calibration":
        return _calibration(argument)
    if kind == "rate":
        return _rate(argument)
    raise KeyError(kind)


def _payloads(budget: int | None = None) -> list[tuple[tuple[str, Any]]]:
    """Every replication, in a fixed order.  ``budget`` caps each family for a smoke run."""

    def count(declared: int) -> int:
        return declared if budget is None else min(declared, budget)

    payloads: list[tuple[str, Any]] = []
    payloads += [("two_arm", r) for r in range(count(ACCURACY_REPLICATES))]
    payloads += [("three_arm", r) for r in range(count(ACCURACY_REPLICATES))]
    payloads += [("bootstrap", r) for r in range(count(BOOTSTRAP_REPLICATES))]
    payloads += [("calibration", r) for r in range(count(CALIBRATION_REPLICATES))]
    payloads += [("rate", (size, r)) for size in RATE_SIZES for r in range(count(RATE_REPLICATES))]
    return [(payload,) for payload in payloads]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries.

    Not named ``declared_cells``: the shared evidence tests read that name as a tuple of
    :class:`~tests.studies.evidence.properties.PropertyCell` objects with a ``dgp``.
    """
    cells = [(ACCURACY, cell, name, truth) for cell, (name, truth, _) in TWO_ARM_CELLS.items()]
    cells += [(ACCURACY, cell, name, truth) for cell, (name, truth) in THREE_ARM_CELLS.items()]
    cells.append((BOOTSTRAP, "ate__known_g__q_wrong__percentile", "ate", TWO_ARM["ate"]))
    for kind in (CALIBRATION_ARM, "shrunken_se_control", "noise_control"):
        cells.append((CALIBRATION, f"ate__{kind}", "ate", TWO_ARM["ate"]))
    for cell in ("ate__known_mechanism", "ate__parametric_mechanism"):
        cells.append((DIRECTION, cell, "ate", TWO_ARM["ate"]))
    cells += [(RATE, f"n_{size}", "ate", TWO_ARM["ate"]) for size in RATE_SIZES]
    return tuple(cells)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Run every declared replication and derive the calibration controls from the same rows."""
    outcomes = map_parallel(_dispatch, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    if budget is not None:
        # A capped smoke run requests what it runs, so the summary reads it as complete.
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    controls = calibration_controls(
        rows,
        STUDY,
        labels=("ate",),
        efficiency_bounds={"ate": NOISE_SD},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
        positive_suffix=CALIBRATION_ARM,
    )
    return pd.concat([rows, controls], ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def _accuracy_verdicts(summary: pd.DataFrame) -> None:
    margins = STUDY.margins
    family = summary["property"] == ACCURACY
    positive = family & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = (
        summary.loc[positive, "bias_equivalent"].astype(bool)
        & (summary.loc[positive, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[positive, "se_ratio"].between(*margins.se_ratio_sanity)
    )
    control = family & (summary["role"] == "control")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"].astype(bool)
    bootstrap = summary["property"] == BOOTSTRAP
    summary.loc[bootstrap, "passed"] = (
        summary.loc[bootstrap, "coverage_ci_lower"] >= margins.coverage_floor
    )


def _direction_report(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The ratio of the estimated-mechanism spread to the known-mechanism spread.

    Reported with a 99% bootstrap interval over replications, paired on ``replicate``.  No
    margin is read: the row is published under the diagnostic role, and Moore and van der
    Laan (2009, Section 7.3) and Petersen et al. (2014, Section 3.7) predict a ratio below
    one with a reported standard error above the spread.
    """
    selected = rows.loc[rows["property"] == DIRECTION]
    known = selected.loc[selected["cell"] == "ate__known_mechanism"].sort_values("replicate")
    estimated = selected.loc[selected["cell"] == "ate__parametric_mechanism"].sort_values(
        "replicate"
    )
    pairs = np.column_stack(
        [known["estimate"].to_numpy(dtype=float), estimated["estimate"].to_numpy(dtype=float)]
    )
    rng = np.random.default_rng(stream_seed(STUDY, DIRECTION, "sd_ratio"))
    ratios = np.concatenate(
        [
            block[:, :, 1].std(axis=1, ddof=1) / block[:, :, 0].std(axis=1, ddof=1)
            for block in bootstrap_draw_blocks(
                pairs, replicates=STUDY.margins.bootstrap_replicates, rng=rng
            )
        ]
    )
    interval = percentile_interval(ratios, confidence_level=STUDY.margins.confidence_level)
    point = float(pairs[:, 1].std(ddof=1) / pairs[:, 0].std(ddof=1))
    mask = summary["property"] == DIRECTION
    summary.loc[mask, "sd_ratio_estimated_over_known"] = point
    summary.loc[mask, "sd_ratio_ci_lower"] = interval.low
    summary.loc[mask, "sd_ratio_ci_upper"] = interval.high
    summary.loc[mask, "passed"] = True
    summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared verdicts, the accuracy and bootstrap rules, and the reported ratio."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "sd_ratio_estimated_over_known",
            "sd_ratio_ci_lower",
            "sd_ratio_ci_upper",
        ),
    )
    _accuracy_verdicts(summary)
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=CALIBRATION_ARM)
    _direction_report(summary, rows)
    return finish(summary, rates)
