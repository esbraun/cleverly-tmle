"""Repeated-sampling properties of the per-arm outcome-adaptive C-TMLE.

Every cell reads the laws of :mod:`tests.studies.oat_per_arm_laws`, cross-fits with five
unstratified folds unless it says otherwise, and fits the per-arm design with the study's
learners: the correctly specified ``A x W`` logistic GLM for the outcome and the cubic in
``logit q`` for each arm's mechanism.

=========================  ===============================================================
family                     cells
=========================  ===============================================================
``interval_calibration``   ``<law>__correctly_specified`` on both laws, with the derived
                           ``shrunken_se_control`` and ``noise_control``;
                           ``binary_weighted__correctly_specified`` under the declared known
                           weight; ``binary_repeats__correctly_specified`` at
                           ``repeats=3``, reading the median estimate
``generated_design``       ``<law>__oracle_design`` (``Qbar`` pinned to the truth) against
                           ``<law>__estimated``, one seed per pair
``type_i_error``           ``sharp_null`` on the sharp-null twin of the binary law, at
                           ``n = 1,000``
``power``                  ``alternative`` on the binary law, at ``n = 2,000``
``root_n_and_efficiency``  ``n_500``, ``n_2000`` and ``n_8000`` on the binary law
``robustness_contract``    ``outcome_correct`` against the ``outcome_wrong`` control (the
                           main-terms GLM without ``W1``, the confounder): OAT has no
                           treatment-only leg
``crossfit_overfitting``   ``cross_fitted_oat`` against ``in_sample_control``, with the
                           parent study's trees and budget
``simultaneous_coverage``  the default band over the three arm means and two contrasts of
                           the three-arm law, and its pointwise control
=========================  ===============================================================

A replicate that raises is not redrawn.  The summary refuses a cell that lost a replicate, so
a failure stops the publishing run.

One quantity of the plan is not fitted.  ``root_n_and_efficiency`` was to report the SE ratio to
the ordinary ``TMLE`` on the same draws.  The harness has no reported-only family, so the
evidence page reads the superefficiency off the ladder instead: the reported SE times
``sqrt(n)`` against :data:`EIF_CURVE_SD`, the law-level efficient SD.  The number is reported,
not gated, as the plan declared.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

from cleverly.estimators import CTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import cvtmle_properties
from tests.studies import oat_per_arm_laws as laws
from tests.studies.bounded_cv_laws import GENERATED_DESIGN_REPLICATES
from tests.studies.canonical_ctmle_oat_per_arm import STRATIFY_FOLDS, STUDY
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyBatch,
    PropertyCell,
    run_cells,
    se_ratio_deficit_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    robustness_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    FAMILY as JOINT_FAMILY,
)
from tests.studies.evidence.simultaneous import (
    joint_coverage_rows,
    joint_property_cells,
)

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: Sizes and budgets, declared before any run.
CALIBRATION_N = 2_000
#: Every positive calibration cell's budget.  The SE-ratio rule needs the 99% bootstrap interval
#: inside (0.93, 1.07), and a perfectly calibrated cell passes it with probability 0.39 at
#: R = 1,000 and 0.14 at R = 800.  At R = 2,000 it passes with probability 0.935, above the 0.90
#: floor of the sizing rule, so the weighted and repeated-split cells take it too.
CALIBRATION_REPLICATES = 2_000
REPEATS_REPLICATES = 2_000
GENERATED_DESIGN_N = 2_000
NULL_N = 1_000
#: The power cell's size.  At ``n = 1,000`` it clears ``MINIMUM_POWER`` only on the
#: superefficient SD (0.977) and not on the efficient one (0.735), so the cell would rest on the
#: very gain the source warns about.  At ``n = 2,000`` it clears both (0.9999 and 0.955).
POWER_N = 2_000
NULL_REPLICATES = 800
LADDER = (500, 2_000, 8_000)
LADDER_REPLICATES = 700
ROBUSTNESS_N = 2_000
ROBUSTNESS_REPLICATES = 1_000
JOINT_N = 2_000
#: The joint cell's budget, by the RM36 power rule of :mod:`tests.studies.default_band_properties`:
#: the smallest of 2,400, 4,000 and 9,600 at which the pointwise control fails its rule with
#: probability at least 0.99.  Fixed by the design test before any run.
JOINT_REPLICATES = 2_400
#: The factor the shrunken control multiplies the reported standard error by.
SHRUNKEN_SE_FACTOR = 0.70
#: The law-level standard deviation of each calibration estimand's per-arm curve, which scales
#: the noise control.  :func:`~tests.studies.oat_per_arm_laws.design_numbers` recomputes it.
CURVE_SD = {"binary_active": 0.9493, "three_arm_active": 1.2221}

#: The design numbers of the gated rejection and control cells, computed before any run.
#: ``POWER_CURVE_SD`` is the law-level standard deviation of the binary ``ate`` per-arm curve,
#: and ``EIF_CURVE_SD`` that of its efficient influence function: ``POWER_CURVE_SD`` over the
#: square root of the efficiency ratio 0.4279.  The power cell must clear the floor of
#: :func:`~tests.studies.evidence.property_verdicts.design_power` under both.
#: ``CONTROL_BIAS`` and ``CONTROL_SD`` are the robustness control's population bias on ``ate``
#: and its standard deviation at :data:`ROBUSTNESS_N`, from two cross-fitted fits at
#: ``n = 200,000`` (bias 0.0738 and 0.0773, curve standard deviation 1.42 and 1.40).
#: ``JOINT_P0`` is the pointwise joint coverage of the band family, from the curve correlation
#: averaged over ten fits at :data:`JOINT_N`, read off two million normal draws.
POWER_CURVE_SD = 0.9493
EIF_CURVE_SD = 1.4513
CONTROL_BIAS = 0.0755
CONTROL_SD = 0.0316
JOINT_P0 = 0.8267

#: The root seed of each family's sample stream.  Controls reuse their positive cell's seed.
SEEDS = {
    "calibration_binary": 71_701,
    "calibration_three_arm": 71_702,
    "calibration_weighted": 71_703,
    "calibration_repeats": 71_704,
    "generated_binary": 71_705,
    "generated_three_arm": 71_706,
    "null": 71_707,
    "power": 71_708,
    "ladder": 71_709,
    "robustness": 71_710,
    "overfit": 71_711,
}

#: The estimand each law's calibration and generated-design cells read.
ESTIMAND = {"binary_active": "ate", "three_arm_active": "ate[1 vs 0]"}
#: The label of each cell family's law, in :data:`~tests.studies.oat_per_arm_laws.LAWS`.
CALIBRATION_LABELS = ("binary_active", "three_arm_active")

#: The joint cell's label and the family it bands.
JOINT_LABEL = "three_arm_active"
JOINT_NAMES = ("ey[0]", "ey[1]", "ey[2]", "ate[1 vs 0]", "ate[2 vs 0]")


def _learners(law: laws.OatLaw, outcome: str) -> tuple[Callable[[], Any], Callable[[], Any]]:
    """The outcome and treatment learner factories of one cell."""
    k = law.k
    if outcome == "oracle":
        name = law.name
        return (lambda: laws.OracleOutcome(name)), laws.cubic_logit_learner
    if outcome == "wrong":
        return (lambda: laws.confounder_omitted_outcome(k)), laws.cubic_logit_learner
    if outcome == "tree":
        return (
            lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=0)
        ), laws.cubic_logit_learner
    return (lambda: laws.interaction_outcome(k)), laws.cubic_logit_learner


def _cell(
    family: str,
    cell: str,
    law: laws.OatLaw,
    n: int,
    replicates: int,
    seed: int,
    *,
    outcome: str = "correct",
    role: str = "positive",
    estimand: str = "ate",
) -> PropertyCell:
    outcome_learner, treatment_learner = _learners(law, outcome)
    fit_kwargs: dict[str, Any] = {"outcome": "Y", "treatment": "A"}
    if law.weighted:
        fit_kwargs["weights"] = "wt"
    return PropertyCell(
        family,
        cell,
        law,
        outcome_learner,
        treatment_learner,
        n,
        replicates,
        seed,
        role=role,
        estimand=estimand,
        fit_kwargs=fit_kwargs,
    )


def cells() -> tuple[PropertyCell, ...]:
    """Every cell :func:`~tests.studies.evidence.properties.run_cells` fits."""
    binary = laws.BINARY_ACTIVE
    three = laws.THREE_ARM_ACTIVE
    out: list[PropertyCell] = [
        _cell(
            "interval_calibration",
            "binary_active__correctly_specified",
            binary,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            SEEDS["calibration_binary"],
        ),
        _cell(
            "interval_calibration",
            "three_arm_active__correctly_specified",
            three,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            SEEDS["calibration_three_arm"],
            estimand=ESTIMAND["three_arm_active"],
        ),
        _cell(
            "interval_calibration",
            "binary_weighted__correctly_specified",
            laws.BINARY_WEIGHTED,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            SEEDS["calibration_weighted"],
        ),
        _cell(
            "interval_calibration",
            "binary_repeats__correctly_specified",
            binary,
            CALIBRATION_N,
            REPEATS_REPLICATES,
            SEEDS["calibration_repeats"],
        ),
    ]
    for law, key in ((binary, "generated_binary"), (three, "generated_three_arm")):
        for cell, outcome in (("oracle_design", "oracle"), ("estimated", "correct")):
            out.append(
                _cell(
                    "generated_design",
                    f"{law.name}__{cell}",
                    law,
                    GENERATED_DESIGN_N,
                    GENERATED_DESIGN_REPLICATES,
                    SEEDS[key],
                    outcome=outcome,
                    estimand=ESTIMAND[law.name],
                )
            )
    out += [
        _cell(
            "type_i_error", "sharp_null", laws.BINARY_NULL, NULL_N, NULL_REPLICATES, SEEDS["null"]
        ),
        _cell("power", "alternative", binary, POWER_N, NULL_REPLICATES, SEEDS["power"]),
    ]
    out += [
        _cell(
            "root_n_and_efficiency",
            f"n_{n}",
            binary,
            n,
            LADDER_REPLICATES,
            SEEDS["ladder"],
        )
        for n in LADDER
    ]
    out += [
        _cell(
            "robustness_contract",
            "outcome_correct",
            binary,
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            SEEDS["robustness"],
        ),
        _cell(
            "robustness_contract",
            "outcome_wrong",
            binary,
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            SEEDS["robustness"],
            outcome="wrong",
            role="control",
        ),
        _cell(
            "crossfit_overfitting",
            "cross_fitted_oat",
            binary,
            cvtmle_properties.OVERFIT_N,
            cvtmle_properties.OVERFIT_REPLICATES,
            SEEDS["overfit"],
            outcome="tree",
        ),
        _cell(
            "crossfit_overfitting",
            "in_sample_control",
            binary,
            cvtmle_properties.OVERFIT_N,
            cvtmle_properties.OVERFIT_REPLICATES,
            SEEDS["overfit"],
            outcome="tree",
            role="control",
        ),
    ]
    return tuple(out)


def _joint_seed(replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", JOINT_FAMILY, JOINT_LABEL, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study runs, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        The fitted cells, then the joint pair.
    """
    return (
        *cells(),
        *joint_property_cells(
            JOINT_LABEL, n=JOINT_N, replicates=JOINT_REPLICATES, seed=_joint_seed(0)
        ),
    )


def _estimator(cell: PropertyCell) -> Callable[[], CTMLE]:
    """The subject of one cell."""
    in_sample = cell.property == "crossfit_overfitting" and cell.cell == "in_sample_control"
    three = cell.dgp.k > 2
    repeats = 3 if cell.cell.startswith("binary_repeats") else 1
    return lambda: CTMLE(
        strategy="oat",
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=not in_sample,
        n_folds=5,
        repeats=repeats,
        estimands=("ate",),
        reference="0" if three else None,
        simultaneous=False,
        g_bounds=laws.G_BOUNDS,
        stratify_folds=STRATIFY_FOLDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def fit_joint(frame: pd.DataFrame) -> Any:
    """The joint cell's fit: the three-arm law, cross-fitted, with the default band on."""
    return (
        CTMLE(
            strategy="oat",
            outcome_learner=laws.interaction_outcome(3),
            treatment_learner=laws.cubic_logit_learner(),
            cross_fit=True,
            n_folds=5,
            estimands=("ey", "ate"),
            reference="0",
            simultaneous=True,
            g_bounds=laws.G_BOUNDS,
            stratify_folds=STRATIFY_FOLDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def _joint_replication(payload: tuple[int, int, int]) -> list[dict[str, Any]]:
    replicate, n, requested = payload
    frame, truth = laws.THREE_ARM_ACTIVE.sample(n, _joint_seed(replicate))
    result = fit_joint(frame)
    return joint_coverage_rows(
        result,
        truth,
        JOINT_NAMES,
        label=JOINT_LABEL,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


def joint_rows(*, n_jobs: int, replicates: int = JOINT_REPLICATES) -> pd.DataFrame:
    """Every replication of the joint pair."""
    outcomes = map_parallel(
        _joint_replication,
        [((replicate, JOINT_N, replicates),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    return pd.DataFrame([row for result in outcomes for row in result])


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The actual sampling calls, shared by complete and targeted regeneration."""
    return (PropertyBatch("properties", cells(), _estimator),)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    rows = sampling_batches()[0].run(n_jobs=n_jobs)
    controls = calibration_controls(
        rows,
        STUDY,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=CURVE_SD,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    joint = joint_rows(n_jobs=n_jobs)
    return pd.concat([rows, controls, joint], ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 20) -> pd.DataFrame:
    """The first ``replicates`` replications of every cell, for a disposable smoke run."""
    smoke = tuple(
        PropertyCell(
            cell.property,
            cell.cell,
            cell.dgp,
            cell.outcome_learner,
            cell.treatment_learner,
            cell.n,
            min(replicates, cell.replicates),
            cell.seed,
            role=cell.role,
            estimand=cell.estimand,
            fit_kwargs=cell.fit_kwargs,
        )
        for cell in cells()
    )
    rows = run_cells(smoke, _estimator, n_jobs=n_jobs)
    joint = joint_rows(n_jobs=n_jobs, replicates=replicates)
    return pd.concat([rows, joint], ignore_index=True)


def _generated_design_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Each law's oracle and estimated arms must calibrate; the paired deficit is reported.

    The verdict logic of :func:`tests.studies.ctmle_oat_properties.summarize_properties`, per
    law.  The pair is the empirical test of "no generated-design term": the estimated design
    must be calibrated against its own sampling spread, as the oracle design is.
    """
    generated = rows.loc[rows["property"] == "generated_design"]
    margins = STUDY.margins
    for label in CALIBRATION_LABELS:
        oracle = generated.loc[generated["cell"] == f"{label}__oracle_design"].sort_values(
            "replicate"
        )
        estimated = generated.loc[generated["cell"] == f"{label}__estimated"].sort_values(
            "replicate"
        )
        intervals = {
            "oracle_design": se_ratio_interval(
                oracle,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "generated_design", label, "oracle_design"),
            ),
            "estimated": se_ratio_interval(
                estimated,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "generated_design", label, "estimated"),
            ),
        }
        deficit = se_ratio_deficit_interval(
            estimated,
            oracle,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "generated_design", label, "deficit"),
        )
        verdicts = {}
        for kind, interval in intervals.items():
            mask = (summary["property"] == "generated_design") & (
                summary["cell"] == f"{label}__{kind}"
            )
            row = summary.loc[mask].iloc[0]
            coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
            verdicts[kind] = bool(
                interval.within(*margins.calibration_se_ratio)
                and coverage.within(*margins.calibration_coverage)
            )
            summary.loc[mask, "se_ratio_ci_lower"] = interval.low
            summary.loc[mask, "se_ratio_ci_upper"] = interval.high
            summary.loc[mask, "se_ratio_deficit_lower"] = deficit.low
            summary.loc[mask, "se_ratio_deficit_upper"] = deficit.high
            summary.loc[mask, "passed"] = verdicts[kind]
        joint = all(verdicts.values())
        pair = (summary["property"] == "generated_design") & summary["cell"].str.startswith(
            f"{label}__"
        )
        summary.loc[pair, "property_passed"] = joint


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            "se_ratio_deficit_lower",
            "se_ratio_deficit_upper",
        ),
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    robustness_verdicts(summary, family="robustness_contract")
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_oat")
    _generated_design_verdicts(summary, rows)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


def revert_flag_red_cells(summary: pd.DataFrame, primary: pd.DataFrame) -> list[str]:
    """The declared revert-flag cells that are red, by name.

    Parameters
    ----------
    summary : DataFrame
        The property summary.
    primary : DataFrame
        The primary performance table, one row per implementation, scenario and estimand.

    Returns
    -------
    list of str
        ``family/cell`` for each red property cell of
        :data:`~tests.studies.canonical_ctmle_oat_per_arm.REVERT_CELLS`, and
        ``primary/<scenario>/<estimand>`` for each red primary coverage row of the subject.
    """
    from tests.studies.canonical_ctmle_oat_per_arm import IMPLEMENTATION, REVERT_CELLS

    red: list[str] = []
    for family, cell in REVERT_CELLS:
        row = summary.loc[(summary["property"] == family) & (summary["cell"] == cell)]
        if row.empty or not bool(row["passed"].iloc[0]):
            red.append(f"{family}/{cell}")
    subject = primary.loc[primary["implementation"] == IMPLEMENTATION]
    for row in subject.itertuples(index=False):
        if float(row.coverage_ci_lower) < STUDY.margins.coverage_floor:
            red.append(f"primary/{row.scenario}/{row.estimand}")
    return red
