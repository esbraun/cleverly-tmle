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
the study's own replicates only.  Run from the repository root::

    python -m tests.canonical.ctmle_sequences.power_probe
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
from tests.canonical.ctmle_sequence_witness import harness
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ctmle_sequences as study
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.seeds import stream_seed

#: The pilot's replicates per law, as the plan's probe ran them.
PILOT_REPLICATES = 1_200
#: The seed stream's label.
PILOT_STREAM = "pilot"
PROBE_FILE = study.ARTIFACTS / "power-probe.csv"
ESTIMATORS = ("ctmle", "no_final_step", "stuck_at_0", "no_targeting", "no_sequence")
COLUMNS = ("scenario", "estimator", "replicates", "truth", "bias", "sd", "bias_sd")


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


def main() -> None:
    """Write ``power-probe.csv``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=PILOT_REPLICATES)
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--output", type=Path, default=PROBE_FILE)
    arguments = parser.parse_args()
    write_csv(run(replicates=arguments.replicates, n_jobs=arguments.jobs), arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
