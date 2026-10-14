"""Repeated-sampling properties of C-TMLE0, ``CTMLE(strategy="derivative")``.

Every cell reads a law of :mod:`tests.studies.ctmle_derivative_laws`, fits the internal penalty
cross-validation (``lambdas=None``) with :data:`~tests.studies.canonical_ctmle_derivative.LEARNER_FOLDS`
folds, and runs in sample unless it says otherwise.

=========================  ===============================================================
family                     cells
=========================  ===============================================================
``interval_calibration``   ``in_sample``; ``cross_fitted`` (five folds); ``multi_arm`` (the
                           three-arm law, multinomial lasso, ``ate[1.0 vs 0.0]``);
                           ``missing_outcome`` (``delta=`` with a correct response learner).
                           Each cell also publishes its SDs against the law's efficiency
                           bound (:data:`EFFICIENCY_BOUNDS`)
``clustered_inference``    ``cluster_robust`` (the clustered law, 100 clusters of 20, ``id=``,
                           cross-fitted) and its ``iid_control``: the same fits, with the IID
                           variance of the same influence curve
``type_i_error``           ``sharp_null`` on the sharp-null law
``power``                  ``alternative`` on the alternative law
``root_n_and_efficiency``  ``n_500``, ``n_2000`` and ``n_8000`` on the primary law
``double_robustness``      ``treatment_correct`` (a ``DummyRegressor`` outcome on the primary
                           law); ``outcome_correct`` (the correct outcome on the
                           ``outcome_correct`` law, whose treatment logit adds
                           ``W2 |W2|``); the ``both_wrong`` control (both on that law); and
                           three wrong-outcome variants of the calibration cells:
                           ``multi_arm_treatment_correct`` (a ``DummyRegressor``),
                           ``missing_outcome_treatment_correct`` and
                           ``cross_fitted_treatment_correct`` (the GLM without ``W2``)
``targeting_necessity``    ``ate__targeted`` against ``ate__untargeted``: one fit with the
                           GLM without ``W2`` as the outcome, and its initial plug-in
=========================  ===============================================================

**Why the wrong-outcome cells.**  With the correct outcome GLM the update barely moves the
estimate, so a correct-outcome cell tests the influence-curve variance of a correct plug-in.
The wrong-outcome cells make the update carry the estimate: the GLM without ``W2`` has a
plug-in limit bias of about 2.6 SD on the primary law, and the ``DummyRegressor`` plug-in is
0.  Their theory is the union model, with a correct treatment law and a wrong outcome, so they
answer to the double-robustness rule (the bias margin and the reported-error screen).  They
make no calibration claim.

**The paired ordinary TMLE.**  Each ``interval_calibration``, ``clustered_inference``
positive, ``type_i_error``, ``power`` and ``double_robustness`` row also carries the ordinary ``TMLE`` on the same initial fit, mechanism and bounds, with only
the derivative columns removed (:func:`~tests.studies.canonical_ctmle_derivative.paired_tmle`).
``tmle_estimate``, ``tmle_std_error`` and ``tmle_covered`` are descriptive.  No verdict reads
them.  The declared reading of a red cell reads them.

**The declared reading** (before the run).  The study publishes under ``reporting``, and
:data:`~tests.studies.canonical_ctmle_derivative.RED_CELL_OWNER` owns every red cell.  For each
red interval, type-I or power cell, :func:`red_cell_reading` compares the C-TMLE0 rows with the
paired TMLE rows of the same cell: McNemar on coverage at 1%, and a paired replicate bootstrap
of the SE-ratio difference at 1%.

* (a) **defect found**: the diagnosis finds a code defect.  It is fixed and the study
  regenerates.
* (b) **shared with TMLE**: TMLE is red too, or no paired difference reaches 1%.  The cell is a
  standing record.
* (c) **derivative-specific**: C-TMLE0 is red, TMLE is green, a paired difference reaches 1%,
  and no defect is found.  The revert rule fires: every ``derivative`` fit takes the
  ``derivative_plugin`` status.

A red ``iid_control`` did not separate, which is a design failure: it is diagnosed and the
law is fixed.  A red ``double_robustness`` cell takes :func:`_robustness_reading`.  A red ``both_wrong``
control whose paired TMLE bias lies outside the margin is a standing record of collaborative
bias reduction.  Any other red cell of the family is a defect suspect and is diagnosed.

**The second declaration.**  The first run's ``outcome_correct`` law put the misspecified term
on ``W1 W2``.  ``W1`` is an instrument, so neither estimator's both-wrong limit bias cleared the
margin, and the control could not separate.  :mod:`tests.studies.ctmle_derivative_dr_pilot`
sized the new law before this declaration.

**The third declaration.**  A design review before the second run found three gaps.  The
clustered law correlated only the outcome residuals, so the influence-curve design effect was
1.01, and the cell could not fail if ``id=`` were ignored.  The treatment is now correlated
within a cluster too (design effect about 2.5), and the cell moves to ``clustered_inference``
with an IID control.  Every calibration cell fitted the correct outcome, so the wrong-outcome
cells above were added.  The necessity control's constant plug-in could not fail, so the
necessity cells now fit the GLM without ``W2``.  The pilot also changed its selection rule
(see :mod:`tests.studies.ctmle_derivative_dr_pilot`).  No margin moved.

A replicate that raises is not redrawn.  The exception stops the run, as the plan's failure
rule requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly.inference import influence_variance
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import ctmle_derivative_laws as laws
from tests.studies.canonical_ctmle_derivative import (
    CRITICAL,
    STUDY,
    estimator,
    omitted_confounder_learner,
    paired_tmle,
)
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    clustered_inference_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

#: Sizes and budgets, declared before any run.
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = laws.CALIBRATION_REPLICATES
NULL_REPLICATES = 2_400
POWER_N = laws.POWER_N
POWER_REPLICATES = laws.POWER_REPLICATES
LADDER = (500, 2_000, 8_000)
LADDER_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
DOUBLE_ROBUST_REPLICATES = 1_200
#: The three-arm wrong-outcome cell's budget.  A 150-replicate pilot gave a pass probability of
#: 0.88 at 1,200, because the three-arm estimate is the noisiest of the family.
MULTI_ARM_ROBUST_REPLICATES = 1_800
NECESSITY_N = 2_000
NECESSITY_REPLICATES = 1_200
#: The declared minimum displacement of the untargeted plug-in from the targeted estimate, in
#: the targeted arm's empirical spread.  The design test checks it against the exact limit
#: bias of the GLM without ``W2`` on the primary law, about 2.5 spreads.
TARGETING_DISPLACEMENT = 1.5
#: The efficiency bound of each calibration cell's law and contrast.
EFFICIENCY_BOUNDS = {
    "in_sample": laws.PRIMARY_EIF_SD,
    "cross_fitted": laws.PRIMARY_EIF_SD,
    "multi_arm": laws.THREE_ARM_EIF_SD,
    "missing_outcome": laws.MISSING_OUTCOME_EIF_SD,
}
#: The significance level of the declared reading's paired tests.
READING_LEVEL = 0.01

#: The columns the calibration-family rows carry beside the shared schema.
PAIRED_COLUMNS = ("tmle_estimate", "tmle_std_error", "tmle_covered")
#: The families whose rows carry :data:`PAIRED_COLUMNS`.  The double-robustness rows carry
#: them too, so each cell shows what its law does to the ordinary TMLE.  No verdict reads them.
PAIRED_FAMILIES = (
    "interval_calibration",
    "clustered_inference",
    "type_i_error",
    "power",
    "double_robustness",
)
#: The families :func:`red_cell_reading` reads with McNemar and the SE-ratio bootstrap.
INTERVAL_FAMILIES = ("interval_calibration", "clustered_inference", "type_i_error", "power")


@dataclass(frozen=True)
class Cell:
    """One declared cell: its family, name, law, size, budget and configuration."""

    family: str
    cell: str
    law: laws.DerivativeLaw
    n: int
    replicates: int
    role: str = "positive"
    outcome: str = "correct"
    cross_fit: bool = False

    @property
    def estimand(self) -> str:
        return "ate[1.0 vs 0.0]" if self.law.k > 2 else "ate"


def cells() -> tuple[Cell, ...]:
    """Every cell the study fits, in run order."""
    out = [
        Cell(
            "interval_calibration", "in_sample", laws.PRIMARY, CALIBRATION_N, CALIBRATION_REPLICATES
        ),
        Cell(
            "interval_calibration",
            "cross_fitted",
            laws.PRIMARY,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            cross_fit=True,
        ),
        Cell(
            "interval_calibration",
            "multi_arm",
            laws.THREE_ARM,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
        ),
        Cell(
            "interval_calibration",
            "missing_outcome",
            laws.MISSING_OUTCOME,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
        ),
        Cell(
            "clustered_inference",
            "cluster_robust",
            laws.CLUSTERED,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            cross_fit=True,
        ),
        Cell("type_i_error", "sharp_null", laws.SHARP_NULL, CALIBRATION_N, NULL_REPLICATES),
        Cell("power", "alternative", laws.ALTERNATIVE, POWER_N, POWER_REPLICATES),
    ]
    out += [
        Cell("root_n_and_efficiency", f"n_{n}", laws.PRIMARY, n, LADDER_REPLICATES) for n in LADDER
    ]
    out += [
        Cell(
            "double_robustness",
            "treatment_correct",
            laws.PRIMARY,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            outcome="dummy",
        ),
        Cell(
            "double_robustness",
            "outcome_correct",
            laws.OUTCOME_CORRECT,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
        ),
        Cell(
            "double_robustness",
            "both_wrong",
            laws.OUTCOME_CORRECT,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            role="control",
            outcome="dummy",
        ),
        Cell(
            "double_robustness",
            "multi_arm_treatment_correct",
            laws.THREE_ARM,
            DOUBLE_ROBUST_N,
            MULTI_ARM_ROBUST_REPLICATES,
            outcome="dummy",
        ),
        Cell(
            "double_robustness",
            "missing_outcome_treatment_correct",
            laws.MISSING_OUTCOME,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            outcome="omit_w2",
        ),
        Cell(
            "double_robustness",
            "cross_fitted_treatment_correct",
            laws.PRIMARY,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            outcome="omit_w2",
            cross_fit=True,
        ),
        Cell(
            "targeting_necessity",
            "ate__targeted",
            laws.PRIMARY,
            NECESSITY_N,
            NECESSITY_REPLICATES,
            outcome="omit_w2",
        ),
    ]
    return tuple(out)


def _root_seed(cell: Cell) -> int:
    """The cell's own stream.  The ladder's rungs share one stream, so they share the seeds."""
    label = "ladder" if cell.family == "root_n_and_efficiency" else cell.cell
    return stream_seed(STUDY, "property_sample", cell.family, label, 0)


def _sample_seed(cell: Cell, replicate: int) -> int:
    label = "ladder" if cell.family == "root_n_and_efficiency" else cell.cell
    return stream_seed(STUDY, "property_sample", cell.family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study publishes, so each committed truth is read back against its law.

    Returns
    -------
    tuple of PropertyCell
        One record per published cell, the untargeted arm included.  Its seed is the cell's
        root stream seed.
    """
    out: list[PropertyCell] = []
    for cell in cells():
        names = {
            "targeting_necessity": ("ate__targeted", "ate__untargeted"),
            "clustered_inference": ("cluster_robust", "iid_control"),
        }.get(cell.family, (cell.cell,))
        for name in names:
            out.append(
                PropertyCell(
                    cell.family,
                    name,
                    cell.law,
                    lambda: None,
                    lambda: None,
                    cell.n,
                    cell.replicates,
                    _root_seed(cell),
                    role="control" if name in ("ate__untargeted", "iid_control") else cell.role,
                    estimand=cell.estimand,
                )
            )
    return tuple(out)


def _outcome(cell: Cell) -> Any:
    if cell.outcome == "dummy":
        return DummyRegressor()
    if cell.outcome == "omit_w2":
        return omitted_confounder_learner()
    return None


def fit_cell(cell: Cell, frame: pd.DataFrame) -> Any:
    """Fit the subject of ``cell`` on one sample."""
    kwargs: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": list(laws.COVARIATES),
    }
    if cell.law.response is not None:
        kwargs["delta"] = "Delta"
    if cell.law.clusters is not None:
        kwargs["id"] = "cluster"
    model = estimator(
        lambdas=None,
        outcome=_outcome(cell),
        cross_fit=cell.cross_fit,
        three_arm=cell.law.k > 2,
        missingness=None if cell.law.response is None else LogisticRegression(max_iter=1000),
    )
    return model.fit(frame, **kwargs).single()


def _replication(payload: tuple[Cell, int]) -> list[dict[str, Any]]:
    cell, replicate = payload
    frame = cell.law.sample(cell.n, _sample_seed(cell, replicate))
    truth = float(cell.law.truth()[cell.estimand])
    result = fit_cell(cell, frame)
    estimate = result[cell.estimand]
    row = replicate_row(
        property_name=cell.family,
        cell=cell.cell,
        role=cell.role,
        replicate=replicate,
        n=cell.n,
        requested=cell.replicates,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    rows = [row]
    if cell.family in PAIRED_FAMILIES:
        plain = paired_tmle(result, (cell.estimand,))[cell.estimand]
        low, high = plain.ci
        row["tmle_estimate"] = float(plain.psi)
        row["tmle_std_error"] = float(plain.std_error)
        row["tmle_covered"] = int(low <= truth <= high)
    if cell.family == "clustered_inference":
        iid = control_row(
            property_name=cell.family,
            cell="iid_control",
            replicate=replicate,
            n=cell.n,
            requested=cell.replicates,
            truth=truth,
            estimate=float(estimate.psi),
            standard_error=float(np.sqrt(influence_variance(estimate.influence_curve))),
            critical=CRITICAL,
        )
        if iid["estimate"] != row["estimate"]:
            raise AssertionError("the IID control changed the point estimate")
        rows.append(iid)
    if cell.family == "targeting_necessity":
        arms = result.nuisance.outcome.arms
        codes = list(result.data.arm_codes)
        initial = float(np.mean(arms[codes[1]]) - np.mean(arms[codes[0]]))
        rows.append(
            control_row(
                property_name=cell.family,
                cell="ate__untargeted",
                replicate=replicate,
                n=cell.n,
                requested=cell.replicates,
                truth=truth,
                estimate=initial,
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
            )
        )
    return rows


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int | None = None
) -> pd.DataFrame:
    """Every replication of every cell.

    Parameters
    ----------
    n_jobs : int, optional
        Worker processes.
    replicates : int or None, optional
        A cap on each cell's budget, for a disposable smoke run.  ``None`` runs the declared
        budgets.

    Returns
    -------
    DataFrame
        The shared schema, then :data:`PAIRED_COLUMNS` (empty off the paired families).
    """
    payloads = [
        ((cell, replicate),)
        for cell in cells()
        for replicate in range(
            cell.replicates if replicates is None else min(replicates, cell.replicates)
        )
    ]
    outcomes = map_parallel(_replication, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    if replicates is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=replicates)
    for column in PAIRED_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    return rows.loc[:, [*REPLICATE_COLUMNS, *PAIRED_COLUMNS]]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    shared = rows.loc[:, list(REPLICATE_COLUMNS)]
    summary, rates = apply_shared_verdicts(
        shared,
        STUDY,
        extra_columns=("targeting_displacement",),
        efficiency_bounds=EFFICIENCY_BOUNDS,
    )
    clustered_inference_verdicts(summary, shared, STUDY)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("ate",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    return finish(summary, rates)


def _paired_se_ratio_difference(
    group: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[float, float]:
    """A paired replicate bootstrap of the C-TMLE0 minus TMLE SE ratio, two-sided at 1%."""
    rng = np.random.default_rng(seed)
    derivative = group[["estimate", "std_error"]].to_numpy(float)
    plain = group[["tmle_estimate", "tmle_std_error"]].to_numpy(float)
    size = len(group)
    draws = np.empty(replicates)
    for index in range(replicates):
        pick = rng.integers(0, size, size)
        draws[index] = derivative[pick, 1].mean() / derivative[pick, 0].std(ddof=1) - plain[
            pick, 1
        ].mean() / plain[pick, 0].std(ddof=1)
    low, high = np.quantile(draws, [READING_LEVEL / 2.0, 1.0 - READING_LEVEL / 2.0])
    return float(low), float(high)


def _robustness_reading(family: str, name: str, group: pd.DataFrame) -> dict[str, Any]:
    """The declared reading of a red double-robustness cell.

    A red ``both_wrong`` control whose paired TMLE bias lies outside the margin is
    ``"collaborative-reduction"``: the law biases the ordinary TMLE, and C-TMLE0 removes part
    of that bias on the same draws.  It is a standing record.  Any other red cell of the
    family is ``"defect-suspect"`` and is diagnosed.
    """
    truth = float(group["truth"].iloc[0])
    plain = group["tmle_estimate"].to_numpy(float)
    derivative = group["estimate"].to_numpy(float)
    size = plain.size
    z = float(norm.ppf(1.0 - (1.0 - STUDY.margins.confidence_level) / 2.0))
    margin = STUDY.margins.standardized_bias * float(derivative.std(ddof=1))
    tmle_bias = float(plain.mean() - truth)
    half = z * float(plain.std(ddof=1)) / np.sqrt(size)
    tmle_discriminated = abs(tmle_bias) - half > margin
    reduction = name == "both_wrong" and tmle_discriminated
    return {
        "property": family,
        "cell": name,
        "derivative_bias": float(derivative.mean() - truth),
        "tmle_bias": tmle_bias,
        "tmle_bias_half_width": half,
        "margin": margin,
        "reading": "collaborative-reduction" if reduction else "defect-suspect",
    }


def red_cell_reading(rows: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """The declared reading of each red interval, type-I or power cell.

    Reads committed columns only.  ``reading`` is ``"shared"`` when the paired TMLE's point
    coverage or SE ratio leaves its calibration band, or no paired difference reaches 1%, and ``"derivative-specific"`` otherwise.  A
    ``"derivative-specific"`` cell fires the revert rule unless the diagnosis finds a defect,
    which this function cannot see.

    Parameters
    ----------
    rows : DataFrame
        The property replication rows, with :data:`PAIRED_COLUMNS`.
    summary : DataFrame
        The property summary.

    Returns
    -------
    DataFrame
        One row per red cell of :data:`PAIRED_FAMILIES`.
    """
    margins = STUDY.margins
    out: list[dict[str, Any]] = []
    red = summary.loc[summary["property"].isin(PAIRED_FAMILIES) & ~summary["passed"].astype(bool)]
    for family, name in red[["property", "cell"]].itertuples(index=False):
        group = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        if name == "iid_control":
            out.append({"property": family, "cell": name, "reading": "design-failure"})
            continue
        if family not in INTERVAL_FAMILIES:
            out.append(_robustness_reading(family, name, group))
            continue
        derivative_only = int(((group["covered"] == 1) & (group["tmle_covered"] == 0)).sum())
        tmle_only = int(((group["covered"] == 0) & (group["tmle_covered"] == 1)).sum())
        discordant = derivative_only + tmle_only
        mcnemar = (
            1.0 if discordant == 0 else float(binomtest(derivative_only, discordant, 0.5).pvalue)
        )
        low, high = _paired_se_ratio_difference(
            group,
            replicates=margins.bootstrap_replicates,
            seed=stream_seed(STUDY, "red_cell_reading", family, name),
        )
        tmle_coverage = float(group["tmle_covered"].mean())
        tmle_se_ratio = float(group["tmle_std_error"].mean() / group["tmle_estimate"].std(ddof=1))
        low_band, high_band = margins.calibration_se_ratio
        tmle_red = tmle_coverage < margins.calibration_coverage[0] or not (
            low_band <= tmle_se_ratio <= high_band
        )
        differs = mcnemar < READING_LEVEL or not (low <= 0.0 <= high)
        out.append(
            {
                "property": family,
                "cell": name,
                "derivative_coverage": float(group["covered"].mean()),
                "tmle_coverage": tmle_coverage,
                "tmle_se_ratio": tmle_se_ratio,
                "mcnemar_pvalue": mcnemar,
                "se_ratio_difference_lower": low,
                "se_ratio_difference_upper": high,
                "reading": "derivative-specific" if differs and not tmle_red else "shared",
            }
        )
    return pd.DataFrame(out)
