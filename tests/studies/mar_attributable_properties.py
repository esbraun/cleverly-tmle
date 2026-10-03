r"""Repeated-sampling properties of PAR and PAF with outcomes missing at random.

Every family reads L1 in sample with the law's own tables, at n = 2,000, unless it says
otherwise.  The design numbers below are exact large-sample limits of the separately
targeted stack (:func:`tests.studies.canonical_mar_attributable.stack_limit`), and
``tests/unit/test_mar_attributable_design.py`` recomputes each one.

``mar_robustness``
    The MAR parent's family and keys.  ``both_correct``, ``outcome_correct`` (wrong ``g`` and
    ``pi``) and ``mechanisms_correct`` (wrong ``Q``) are positive.  ``treatment_wrong``
    (``Q`` and ``g`` wrong) is the reference-path control.  ``observation_wrong`` (``Q`` and
    ``pi`` wrong) is both-wrong for the natural-course coordinate, but that coordinate moves
    by only -0.014 at its limit; the cell's displacement is carried by the reference arm.  ``product_only`` (``Q``
    wrong; ``g pi`` correct at the reference arm with ``pi`` wrong) shows that a correct
    product rescues the reference arm and not the natural course.  Exact targeted limits of
    the PAR bias: -0.1354, -0.1968 and -0.1243, that is 9.0, 13.1 and 8.3 per-replication
    SDs.  (The plan's first-order numbers, -0.0371, -0.195 and -0.177, read the remainder at
    the untargeted regression.)  A control passes when its bias is discriminated and its SE
    ratio lies in ``UNION_MODEL_SE_BAND``.
``root_n_and_efficiency``
    ``par`` at n = 500, 2,000 and 8,000; exact EIF SD 0.67359.
``interval_calibration``
    ``par`` and ``paf`` from the same 4,000 fits.  The shrunken and noise controls of ``par``
    come from :func:`calibration_controls`.  The inflated-SE controls drop the cross-covariance
    of the two parent curves; their exact limiting SE ratios are 1.951 (PAR) and 1.715 (PAF),
    so each must rise above the calibration band.
``targeting_necessity``
    ``par`` targeted against its untargeted plug-in, with a wrong outcome regression.  The
    untargeted limit is displaced by -0.228, 15.1 per-replication SDs.
``missingness_necessity``
    ``par`` with the response declared, against complete-case fits of ``par`` and ``paf``.
    Limiting complete-case displacements: -0.0295 (1.96 per-replication SDs) and -0.0574
    (1.79 SDs).
``simultaneous_coverage``
    The default band of five joint fits at 2,400 replications each: in-sample L1 (``ey_obs``,
    ``ey0``, ``par``, ``paf``), in-sample L3 (``ey_obs``, three arm means, ``par[low]``,
    ``paf[low]``), stacked L1, and the weighted and clustered in-sample L1 fits.
    ``tests/unit/test_simultaneous_cell_design.py`` holds each cell's design from the exact
    covariance, and maps the stacked L3 shape to the L3 design.

Declared before any run: the study is ``gated``.  A red cell is diagnosed from the committed
rows; a defect in ``cleverly`` is fixed and the study regenerated, and otherwise a
declaration switches the study to ``reporting`` with a red-cell owner row (id ``PAR-PAF-MAR``)
before one re-run with the same seeds.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.canonical_mar_attributable import (
    BINARY,
    BINARY_CV,
    CLUSTERED,
    EFFICIENCY_SD,
    NAMES,
    STUDY,
    THREE_ARM,
    TRUTHS,
    WEIGHTED,
    fit_cleverly,
    initial_estimates,
    oracle_learners,
)
from tests.studies.evidence.properties import (
    PropertyCell,
    ReplicationSpec,
    control_row,
    property_role,
    replicate_row,
    replication_payloads,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    robustness_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells
from tests.studies.missing_outcome_study_helpers import WRONG_PI, WRONG_Q

ROBUSTNESS_REPLICATES = 1_200
ROBUSTNESS_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
BAND_REPLICATES = 2_400
BAND_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
MISSINGNESS_DISPLACEMENT = 0.25
TARGET = "par"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The misspecified treatment mechanism of the MAR parent study, ``P(A = 1 | W)``.
WRONG_G = laws._two_arm(np.array([0.70, 0.30, 0.70]))
#: ``P'(A = 0 | W)`` of the product-only cell.  The response mechanism at arm 0 restores the
#: true product ``g(0 | W) pi(0, W)``, and at arm 1 it is the wrong one.
PRODUCT_G0 = np.array([0.80, 0.55, 0.95])
PRODUCT_G = np.column_stack([PRODUCT_G0, 1.0 - PRODUCT_G0])
PRODUCT_PI = np.column_stack([(1.0 - mar.G) * mar.PI[:, 0] / PRODUCT_G0, WRONG_PI[:, 1]])

#: Each ``mar_robustness`` configuration's working tables, as law-table replacements.
CONFIGURATIONS: dict[str, dict[str, np.ndarray]] = {
    "both_correct": {},
    "outcome_correct": {"g": WRONG_G, "pi": WRONG_PI},
    "mechanisms_correct": {"mu": WRONG_Q},
    "treatment_wrong": {"mu": WRONG_Q, "g": WRONG_G},
    "observation_wrong": {"mu": WRONG_Q, "pi": WRONG_PI},
    "product_only": {"mu": WRONG_Q, "g": PRODUCT_G, "pi": PRODUCT_PI},
}
CONTROLS = frozenset({"treatment_wrong", "observation_wrong", "product_only"})

#: Each band label's scenario.  The stacked label refits with the stacked learners.
BAND_SCENARIOS = {
    "attributable_binary": BINARY,
    "attributable_three_arm": THREE_ARM,
    "attributable_binary_cvtmle": BINARY_CV,
    "attributable_weighted": WEIGHTED,
    "attributable_clustered": CLUSTERED,
}

Payload = tuple[str, str, int, int, int, int, str]


def _fit(frame: pd.DataFrame, configuration: str, **extra: Any) -> Any:
    learners = oracle_learners(BINARY, **CONFIGURATIONS[configuration])
    return fit_cleverly(frame, BINARY, learners=learners, **extra)


def _row(
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    name: str,
    result: Any,
    truth: float | None = None,
) -> dict[str, Any]:
    return replicate_row(
        property_name=property_name,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=TRUTHS[BINARY][name] if truth is None else truth,
        estimate=result[name],
        alpha=STUDY.margins.alpha,
    )


def independent_se(result: Any, name: str) -> float:
    """The SE that drops the cross-covariance of the two parent curves (M1's control)."""
    observed, reference = result["ey_obs"], result["ey0"]
    if name == "par":
        return float(np.sqrt(observed.variance + reference.variance))
    psi_obs, psi_ref = observed.psi, reference.psi
    return float(
        np.sqrt(reference.variance / psi_obs**2 + observed.variance * psi_ref**2 / psi_obs**4)
    )


def _fit_replication(payload: Payload) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    if property_name == "simultaneous_coverage":
        scenario = BAND_SCENARIOS[cell]
        frame, truth = _draw(scenario, n, seed)
        result = fit_cleverly(frame, scenario, simultaneous=True)
        return joint_coverage_rows(
            result,
            truth,
            NAMES[scenario],
            label=cell,
            replicate=replicate,
            n=n,
            requested=requested,
            pointwise_critical=CRITICAL,
        )
    frame, _ = _draw(BINARY, n, seed)
    result = _fit(frame, configuration)
    if property_name == "interval_calibration":
        rows = [
            _row(
                property_name,
                f"{name}__correctly_specified",
                "positive",
                replicate,
                n,
                requested,
                name,
                result,
            )
            for name in ("par", "paf")
        ]
        for name in ("par", "paf"):
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{name}__inflated_se_control",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=TRUTHS[BINARY][name],
                    estimate=float(result[name].psi),
                    standard_error=independent_se(result, name),
                    critical=CRITICAL,
                )
            )
        return rows
    role = property_role(
        configuration,
        controls=CONTROLS,
        property_name=property_name,
        n=n,
        rate_sizes=RATE_SIZES,
    )
    rows = [_row(property_name, cell, role, replicate, n, requested, TARGET, result)]
    if property_name == "targeting_necessity":
        rows[0]["cell"] = "par__targeted"
        rows.append(
            control_row(
                property_name=property_name,
                cell="par__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=TRUTHS[BINARY][TARGET],
                estimate=initial_estimates(result, BINARY)[TARGET],
                standard_error=float(result[TARGET].std_error),
                critical=CRITICAL,
            )
        )
    if property_name == "missingness_necessity":
        rows[0]["cell"] = "par__declared"
        complete = frame.loc[frame["Delta"] == 1.0, ["W", "A", "Y"]].reset_index(drop=True)
        learners = oracle_learners(BINARY, **CONFIGURATIONS[configuration])
        ignored = fit_cleverly(
            complete, BINARY, learners=learners, request=("par", "paf"), delta=False
        )
        for name in ("par", "paf"):
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{name}__complete_case_control",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=TRUTHS[BINARY][name],
                    estimate=float(ignored[name].psi),
                    standard_error=float(ignored[name].std_error),
                    critical=CRITICAL,
                )
            )
    return rows


def _draw(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    from tests.studies.canonical_mar_attributable import draw_from_seed

    return draw_from_seed(scenario, n, seed)


def _specs() -> list[ReplicationSpec]:
    specs = [
        ReplicationSpec(
            "mar_robustness", configuration, ROBUSTNESS_N, ROBUSTNESS_REPLICATES, configuration
        )
        for configuration in CONFIGURATIONS
    ]
    specs.extend(
        ReplicationSpec("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct")
        for size in RATE_SIZES
    )
    specs.extend(
        [
            ReplicationSpec(
                "interval_calibration",
                "par__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ReplicationSpec(
                "targeting_necessity",
                "targeted",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "mechanisms_correct",
            ),
            ReplicationSpec(
                "missingness_necessity",
                "declared",
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "both_correct",
            ),
        ]
    )
    specs.extend(
        ReplicationSpec("simultaneous_coverage", label, BAND_N, BAND_REPLICATES, "both_correct")
        for label in BAND_SCENARIOS
    )
    return specs


def _payloads(budget: int | None = None) -> list[tuple[Payload]]:
    """Every replication's payload.  ``budget`` caps each cell, for a smoke run only."""
    payloads = replication_payloads(STUDY, _specs())
    if budget is None:
        return payloads
    return [payload for payload in payloads if payload[0][2] < budget]


@dataclasses.dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads: its name, and its exact truth for every estimand.

    Every non-band family samples L1 in sample, so each one reads the binary scenario's
    truths.  The name carries the nuisance configuration, so two families never share a
    ``(name, seed)`` stream in the collision gate.

    Parameters
    ----------
    name : str
        The law and configuration, such as ``"L1:both_correct"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        """The exact value of every name the binary scenario reports."""
        return dict(TRUTHS[BINARY])


#: The published cells of each non-band family that one replication spec emits, with the
#: estimand each one reports and its role.  The spec's own cell name is the stream key.
_EMITTED: dict[str, tuple[tuple[str, str, str], ...]] = {
    "interval_calibration": (
        ("par__correctly_specified", "par", "positive"),
        ("paf__correctly_specified", "paf", "positive"),
        ("par__inflated_se_control", "par", "control"),
        ("paf__inflated_se_control", "paf", "control"),
        ("par__shrunken_se_control", "par", "control"),
        ("par__noise_control", "par", "control"),
    ),
    "targeting_necessity": (
        ("par__targeted", "par", "positive"),
        ("par__untargeted", "par", "control"),
    ),
    "missingness_necessity": (
        ("par__declared", "par", "positive"),
        ("par__complete_case_control", "par", "control"),
        ("paf__complete_case_control", "paf", "control"),
    ),
}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its law, its estimand and the root of its stream."""
    out: list[PropertyCell] = []
    for spec in _specs():
        seed = replication_payloads(STUDY, [dataclasses.replace(spec, replicates=1)])[0][0][5]
        if spec.property == "simultaneous_coverage":
            out.extend(
                joint_property_cells(spec.cell, n=spec.n, replicates=spec.replicates, seed=seed)
            )
            continue
        role = property_role(
            spec.configuration,
            controls=CONTROLS,
            property_name=spec.property,
            n=spec.n,
            rate_sizes=RATE_SIZES,
        )
        emitted = _EMITTED.get(spec.property, ((spec.cell, TARGET, role),))
        for cell, estimand, kind in emitted:
            out.append(
                PropertyCell(
                    property=spec.property,
                    cell=cell,
                    dgp=DeclaredLaw(f"L1:{spec.configuration}"),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=spec.n,
                    replicates=spec.replicates,
                    seed=seed,
                    role=kind,
                    estimand=estimand,
                )
            )
    return tuple(out)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    controls = calibration_controls(
        rows,
        STUDY,
        labels=(TARGET,),
        efficiency_bounds={TARGET: EFFICIENCY_SD[TARGET]},
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "missingness_displacement"),
        efficiency_bounds=EFFICIENCY_SD,
    )
    robustness_verdicts(summary, family="mar_robustness")
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=(TARGET,),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="missingness_necessity",
        labels=(TARGET,),
        arms=("declared", "complete_case_control"),
        column="missingness_displacement",
        threshold=MISSINGNESS_DISPLACEMENT,
    )
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
