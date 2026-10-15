"""The declaration-step pilot of ``canonical-ctmle-sequences`` (X16 plan, 3.4.1 and 3.4.3).

The pilot reruns the Monte Carlo probe of the plan on this record's ``pilot`` stream, which
shares no seed with a replicate.  Each law gets :data:`PILOT_REPLICATES` fits at the primary
``n`` with the study's own estimator settings.  Each mutation is computed on the same fit from
the captured analysis-sample selector:

==================  =====================================================================
``ctmle``           the package's C-TMLE, as the study fits it
``no_final_step``   the selected candidate's estimate, without the lasso final step
``stuck_at_0``      ``selected = 0``: the final step from index 0 for lasso, and index 0's
                    estimate for learners
``no_targeting``    the plug-in of the initial outcome fit
``no_sequence``     plain TMLE with the terminal, ``treatment_learner``
==================  =====================================================================

The output, ``power-probe.csv``, holds one row per scenario and estimator: the bias, the
empirical SD and the bias in that SD.  ``tests/unit/test_ctmle_sequences_design.py`` reads
the file and recomputes the pass probabilities and powers of plan 3.4.3.  The binary_learners
pilot draws its selection split from the replicate's seed, because ``r_folds.csv.gz`` covers
the study's own replicates only.

The property pilot measures the bias of each cell of :data:`PROPERTY_PILOT_CELLS`, which no
other study measures.  It fits each cell :data:`PROPERTY_PILOT_REPLICATES` times with the
study's own estimator and law, on seeds of the ``pilot`` stream keyed by the cell's seed root,
so the paired multi-arm cells share their samples as they do in the study.
:func:`property_pilot_seeds` refuses a seed that a registered replicate draws.  The output,
``property-pilot.csv``, holds one row per cell: the bias, the empirical SD, the bias in that SD
and its Monte Carlo SE (:func:`standardized_bias_se`).  ``DECLARED_GATES`` in the design test
holds these biases.  Run from the repository root::

    python -m tests.canonical.ctmle_sequences.power_probe
    python -m tests.canonical.ctmle_sequences.power_probe --properties
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.utils.parallel import map_parallel
from cleverly.validation import CoverageStudy
from cleverly.validation._seeds import sample_seed_streams
from tests.canonical.ctmle_sequence_witness import harness
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ctmle_sequences as study
from tests.studies import ctmle_sequence_properties as properties
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, summarize_cells
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The pilot's replicates per law, as the plan's probe ran them.
PILOT_REPLICATES = 1_200
#: The seed stream's label.
PILOT_STREAM = "pilot"
PROBE_FILE = study.ARTIFACTS / "power-probe.csv"
ESTIMATORS = ("ctmle", "no_final_step", "stuck_at_0", "no_targeting", "no_sequence")
COLUMNS = ("scenario", "estimator", "replicates", "truth", "bias", "sd", "bias_sd")

#: The property cells whose bias no other study measures: the treatment-correct cells of
#: both strategies on the bounded linear law, the multi-arm pair, and every other cell that
#: fits the declared learners of ``LEARNER_COLUMNS``.  The learners cells were measured again
#: when the declared candidates became nested.
PROPERTY_PILOT_CELLS = (
    "double_robustness/lasso_treatment_correct",
    "double_robustness/learners_treatment_correct",
    "multi_arm/lasso_treatment_correct",
    "multi_arm/empty_control",
    "double_robustness/learners_outcome_correct",
    "double_robustness/both_wrong_learners",
    "selector_necessity/learners_collaborative",
)
#: The pilot cells that fit the declared learners.  ``--learners`` runs these alone and keeps
#: the committed rows of the other cells, whose fits do not read ``LEARNER_COLUMNS``.
LEARNER_PILOT_CELLS = tuple(key for key in PROPERTY_PILOT_CELLS if "learners" in key)
#: The property pilot's fits per cell.  The Monte Carlo SE of a bias in its own SD is then
#: about ``1 / sqrt(4,800) = 0.014``, a seventeenth of the margin.
PROPERTY_PILOT_REPLICATES = 4_800
PROPERTY_PILOT_FILE = study.ARTIFACTS / "property-pilot.csv"
PROPERTY_COLUMNS = (
    "property",
    "cell",
    "role",
    "n",
    "replicates",
    "failed_replicates",
    "truth",
    "bias",
    "sd",
    "bias_sd",
    "bias_sd_mc_se",
)


def pilot_seed(scenario: str, replicate: int) -> int:
    """The sample seed of one pilot replicate.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.
    replicate : int
        The pilot replicate index.

    Returns
    -------
    int
        A seed from this record's ``pilot`` stream.
    """
    return stream_seed(study.STUDY, PILOT_STREAM, scenario, study.PRIMARY_N, replicate)


def _split(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % study.SELECTION_FOLDS) + 1


def probe_one(scenario: str, replicate: int) -> dict[str, float]:
    """Every estimator of the pilot on one replicate.

    Parameters
    ----------
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.
    replicate : int
        The pilot replicate index.

    Returns
    -------
    dict
        The truth and one estimate per entry of :data:`ESTIMATORS`.
    """
    seed = pilot_seed(scenario, replicate)
    frame, truth = study.draw_from_seed(scenario, study.PRIMARY_N, seed)
    split = _split(seed, len(frame)) if scenario == "binary_learners" else None
    covariates = list(study.COVARIATES)
    with warnings.catch_warnings():
        # Both laws put 8 to 11 percent of g outside the bounds by design (plan 3.4.1).
        warnings.simplefilter("ignore", PositivityWarning)
        with harness._captured(trace=True) as capture:
            result = (
                study.estimator(scenario, frame, split)
                .fit(frame, outcome="Y", treatment="A", covariates=covariates)
                .single()
            )
        study.check_structure(result, scenario)
        chooser, path = capture.selectors[0], capture.paths[0]
        selection = result.extra["ctmle"]
        selected = int(selection.selected)
        if scenario == "binary_lasso":
            stuck = chooser.final_step(path.candidates, 0)
            stuck_estimate = harness.ate(
                (path.candidates[0] if stuck is None else stuck[1]).targeted
            )
        else:
            stuck_estimate = harness.ate(path.candidates[0].targeted)
        plain = (
            TMLE(
                outcome_learner=study._logistic(),
                treatment_learner=study._logistic(),
                cross_fit=False,
                estimands=("ate",),
                g_bounds=study.G_BOUNDS,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()["ate"]
        )
    return {
        "truth": float(truth["ate"]),
        "ctmle": float(result["ate"].psi),
        "no_final_step": harness.ate(path.candidates[selected].targeted),
        "stuck_at_0": stuck_estimate,
        "no_targeting": harness.ate(chooser.base.outcome),
        "no_sequence": float(plain.psi),
    }


def summarize(estimates: pd.DataFrame) -> pd.DataFrame:
    """One :data:`COLUMNS` row per scenario and estimator.

    Parameters
    ----------
    estimates : DataFrame
        ``scenario``, ``truth`` and one column per estimator, one row per pilot replicate.

    Returns
    -------
    DataFrame
        The bias, the empirical SD and the bias in that SD.
    """
    out = []
    for scenario, group in estimates.groupby("scenario", sort=False):
        truth = float(group["truth"].iloc[0])
        for name in ESTIMATORS:
            values = group[name].to_numpy(dtype=float)
            bias = float(np.mean(values) - truth)
            sd = float(np.std(values, ddof=1))
            out.append(
                {
                    "scenario": scenario,
                    "estimator": name,
                    "replicates": int(values.size),
                    "truth": truth,
                    "bias": bias,
                    "sd": sd,
                    "bias_sd": bias / sd if sd > 0 else math.nan,
                }
            )
    return pd.DataFrame(out, columns=list(COLUMNS))


def run(*, replicates: int, n_jobs: int) -> pd.DataFrame:
    """Run the pilot on both laws.

    Parameters
    ----------
    replicates : int
        Pilot replicates per law.
    n_jobs : int
        Worker processes.

    Returns
    -------
    DataFrame
        :func:`summarize` of the pilot.
    """
    payloads = [
        (scenario, replicate) for scenario in study.SCENARIOS for replicate in range(replicates)
    ]
    rows: list[dict[str, Any]] = map_parallel(probe_one, payloads, n_jobs=n_jobs)
    estimates = pd.DataFrame(
        [{"scenario": scenario, **row} for (scenario, _), row in zip(payloads, rows, strict=True)]
    )
    return summarize(estimates)


# --------------------------------------------------------------- property pilot


def pilot_cells() -> tuple[PropertyCell, ...]:
    """The cells of :data:`PROPERTY_PILOT_CELLS`, as the study declares them.

    Returns
    -------
    tuple of PropertyCell
        One cell per key, in the order of :data:`PROPERTY_PILOT_CELLS`.
    """
    declared = {f"{cell.property}/{cell.cell}": cell for cell in properties.cells()}
    return tuple(declared[key] for key in PROPERTY_PILOT_CELLS)


def registered_seeds() -> frozenset[int]:
    """Every sample seed a registered replicate of the study draws.

    Returns
    -------
    frozenset of int
        The property cells' streams at their declared budgets, and the primary replicates'
        seeds.
    """
    budgets: dict[int, int] = {}
    for cell in properties.cells():
        budgets[cell.seed] = max(budgets.get(cell.seed, 0), cell.replicates)
    seeds = {int(seed) for stream in sample_seed_streams(budgets).values() for seed in stream}
    seeds.update(
        replicate_seed(study.STUDY, scenario, replicate)
        for scenario in study.SCENARIOS
        for replicate in range(study.PRIMARY_REPLICATES)
    )
    return frozenset(seeds)


def property_pilot_seeds(cell: PropertyCell, replicates: int) -> tuple[int, ...]:
    """The pilot's sample seeds of one property cell.

    Parameters
    ----------
    cell : PropertyCell
        A cell of :func:`pilot_cells`.
    replicates : int
        The pilot replicates.

    Returns
    -------
    tuple of int
        Distinct seeds of the ``pilot`` stream, keyed by the cell's seed root.

    Raises
    ------
    RuntimeError
        When a seed repeats or a registered replicate draws it.
    """
    seeds = tuple(
        stream_seed(study.STUDY, PILOT_STREAM, "property", cell.seed, cell.n, replicate)
        for replicate in range(replicates)
    )
    if len(set(seeds)) != len(seeds) or not registered_seeds().isdisjoint(seeds):
        raise RuntimeError(f"the pilot seeds of {cell.property}/{cell.cell} are not disjoint")
    return seeds


def _pilot_replication(coverage: CoverageStudy, index: int, seed: int) -> Any:
    return coverage.run_replication(index, seed)


def property_rows(cell: PropertyCell, replicates: int, *, n_jobs: int) -> pd.DataFrame:
    """The pilot's replicate rows of one cell, in the study's property schema.

    Parameters
    ----------
    cell : PropertyCell
        A cell of :func:`pilot_cells`.
    replicates : int
        The pilot replicates.
    n_jobs : int
        Worker processes.

    Returns
    -------
    DataFrame
        One :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS` row per fit.
    """
    seeds = property_pilot_seeds(cell, replicates)
    coverage = CoverageStudy(
        dgp=cell.dgp,
        estimator=properties._estimator(cell),
        n=cell.n,
        n_replicates=replicates,
        estimands=(cell.estimand,),
        fit_kwargs=dict(cell.fit_kwargs),
        seed=cell.seed,
        sample_seeds=seeds,
        n_jobs=n_jobs,
    )
    outcomes = map_parallel(
        _pilot_replication,
        [(coverage, index, seed) for index, seed in enumerate(seeds)],
        n_jobs=n_jobs,
    )
    records = [
        record
        for outcome in outcomes
        if isinstance(outcome, tuple)
        for record in outcome
        if record.estimand == cell.estimand
    ]
    failed = sum(not isinstance(outcome, tuple) for outcome in outcomes)
    return pd.DataFrame(
        {
            "property": cell.property,
            "cell": cell.cell,
            "role": cell.role,
            "replicate": [record.replicate for record in records],
            "n": cell.n,
            "requested_replicates": replicates,
            "failed_replicates": failed,
            "truth": [record.truth for record in records],
            "estimate": [record.estimate for record in records],
            "std_error": [record.std_error for record in records],
            "covered": [int(record.covered) for record in records],
            "rejected": [int(record.rejected) for record in records],
        }
    ).loc[:, list(REPLICATE_COLUMNS)]


def standardized_bias_se(errors: np.ndarray) -> float:
    """The delta-method Monte Carlo SE of ``mean(errors) / sd(errors)``.

    With ``b`` the bias in its SD, ``gamma`` the skewness and ``kappa`` the kurtosis of the
    errors, the variance is ``(1 - b gamma + b^2 (kappa - 1) / 4) / R``.  For normal errors it
    is ``(1 + b^2 / 2) / R``.

    Parameters
    ----------
    errors : ndarray of float
        The estimate minus the truth, one per replicate.

    Returns
    -------
    float
        The Monte Carlo SE of the standardized bias.
    """
    errors = np.asarray(errors, dtype=float)
    centred = errors - errors.mean()
    sd = float(np.sqrt(np.mean(centred**2)))
    b = float(errors.mean()) / sd
    skewness = float(np.mean(centred**3)) / sd**3
    kurtosis = float(np.mean(centred**4)) / sd**4
    return math.sqrt((1.0 - b * skewness + b**2 * (kurtosis - 1.0) / 4.0) / errors.size)


def summarize_property_pilot(rows: pd.DataFrame) -> pd.DataFrame:
    """One :data:`PROPERTY_COLUMNS` row per cell, standardized as the study's verdict is.

    Parameters
    ----------
    rows : DataFrame
        The pilot's replicate rows.

    Returns
    -------
    DataFrame
        The bias, the empirical SD, the bias in that SD and its Monte Carlo SE.
    """
    margins = study.STUDY.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
        truth_varies=study.STUDY.truth_varies_by_replicate,
    ).set_index(["property", "cell"])
    out = []
    present = set(rows["property"] + "/" + rows["cell"])
    for key in (key for key in PROPERTY_PILOT_CELLS if key in present):
        property_name, cell = key.split("/")
        group = rows.loc[(rows["property"] == property_name) & (rows["cell"] == cell)]
        row = summary.loc[(property_name, cell)]
        errors = group["estimate"].to_numpy(dtype=float) - group["truth"].to_numpy(dtype=float)
        out.append(
            {
                "property": property_name,
                "cell": cell,
                "role": row["role"],
                "n": int(row["n"]),
                "replicates": int(row["replicates"]),
                "failed_replicates": int(row["failed_replicates"]),
                "truth": float(row["truth"]),
                "bias": float(row["bias"]),
                "sd": float(row["empirical_se"]),
                "bias_sd": float(row["standardized_bias"]),
                "bias_sd_mc_se": standardized_bias_se(errors),
            }
        )
    return pd.DataFrame(out, columns=list(PROPERTY_COLUMNS))


def run_properties(
    *, replicates: int, n_jobs: int, keys: tuple[str, ...] = PROPERTY_PILOT_CELLS
) -> pd.DataFrame:
    """Run the property pilot on the cells ``keys`` of :data:`PROPERTY_PILOT_CELLS`.

    Parameters
    ----------
    replicates : int
        Pilot replicates per cell.
    n_jobs : int
        Worker processes.
    keys : tuple of str
        The ``property/cell`` keys to run, every pilot cell by default.

    Returns
    -------
    DataFrame
        :func:`summarize_property_pilot` of the pilot.
    """
    cells = [cell for cell in pilot_cells() if f"{cell.property}/{cell.cell}" in keys]
    rows = pd.concat(
        [property_rows(cell, replicates, n_jobs=n_jobs) for cell in cells],
        ignore_index=True,
    )
    return summarize_property_pilot(rows)


def merge_pilot(committed: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """The committed pilot rows with the rows of ``fresh`` in their place, in declared order.

    Parameters
    ----------
    committed : DataFrame
        The committed ``property-pilot.csv``.
    fresh : DataFrame
        The rows of the cells that ran again.

    Returns
    -------
    DataFrame
        One row per cell of :data:`PROPERTY_PILOT_CELLS`.
    """
    frames = {f"{row.property}/{row.cell}": row for row in committed.itertuples(index=False)}
    frames.update({f"{row.property}/{row.cell}": row for row in fresh.itertuples(index=False)})
    missing = [key for key in PROPERTY_PILOT_CELLS if key not in frames]
    if missing:
        raise RuntimeError(f"the pilot has no row for {missing}")
    return pd.DataFrame(
        [frames[key] for key in PROPERTY_PILOT_CELLS], columns=list(PROPERTY_COLUMNS)
    )


def main() -> None:
    """Write ``power-probe.csv``, or ``property-pilot.csv`` with ``--properties``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--properties", action="store_true", help="run the property pilot")
    parser.add_argument(
        "--learners",
        action="store_true",
        help="with --properties, run the learners cells and keep the committed other rows",
    )
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.properties:
        replicates = arguments.replicates or PROPERTY_PILOT_REPLICATES
        output = arguments.output or PROPERTY_PILOT_FILE
        if arguments.learners:
            fresh = run_properties(
                replicates=replicates, n_jobs=arguments.jobs, keys=LEARNER_PILOT_CELLS
            )
            table = merge_pilot(
                pd.read_csv(PROPERTY_PILOT_FILE, float_precision="round_trip"), fresh
            )
        else:
            table = run_properties(replicates=replicates, n_jobs=arguments.jobs)
    else:
        replicates = arguments.replicates or PILOT_REPLICATES
        output = arguments.output or PROBE_FILE
        table = run(replicates=replicates, n_jobs=arguments.jobs)
    write_csv(table, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
