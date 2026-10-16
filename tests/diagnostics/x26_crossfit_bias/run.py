r"""The bias of the cross-fitted plug-in omitted-variable bound, on the binary law of X26.

The package refuses the plug-in limits on a cross-fitted mechanism (``cross_fit`` clause of
``cleverly.sensitivity._mechanism_stack``, constant ``OUT_OF_FOLD_REFUSED``).  A design-review
pilot found the cross-fitted ATE :math:`\nu^2` limit covering 0.887 against a band of 0.92 to
0.98.  This diagnostic holds the evidence for the refusal.  The removed registered scenario
``binary_logistic_crossfit`` declared the fit below.

On the binary law of ``tests.studies.omitted_variable_bound_plugin`` (``make_linear_ate``) at
strength ``cf_y = cf_d = 0.5`` and ``rho = 1``, each draw is fitted twice.

``crossfit``
    ``TMLE(cross_fit=True, n_folds=5, q_bounds=(-40.0, 45.0))`` with a ``LinearRegression``
    outcome and ``LogisticRegression(C=numpy.inf, max_iter=2000)``, estimands ATE, ATT and ATC,
    ``random_state=0``.  The worker sets ``OUT_OF_FOLD_REFUSED`` to False for the length of the
    fit and restores it afterwards, so the plug-in limits are computed instead of refused.
    The assignment is made in the process that fits, which also holds under a process pool.
``insample``
    The registered ``binary_logistic`` fit, ``cross_fit=False``, on the same draw.

Both fits go through the study's ``parameter_rows``, which rebuilds the curve of
:math:`\hat\nu^2` from the captured coefficients and stops the run on any departure from the
package curve.  Each row records :math:`\hat\nu^2` with ``sqrt(influence_variance(psi_nu2))``,
and the lower and upper bound with the standard error read back off the reported limit.

**Seeds.**  The draw of replicate ``k`` at sample size ``n`` uses
``stream_seed(STUDY, "x26-crossfit-bias", n, k)``.  That stream hashes a label into the
spawn key of the study's resampling seed.  The registered study draws with
``replicate_seed``, which spawns on the study seed and an integer scenario index, so the two
streams are disjoint.  Each ``n`` has its own stream.

**The n ladder.**  2,000 replicates at ``n = 1000`` and 1,000 at ``n = 4000``.  The ladder
shows whether the bias, in empirical standard deviations, shrinks as ``n`` grows.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_x26_crossfit_bias_reading.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.x26_crossfit_bias.run --jobs 3
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.sensitivity import _mechanism_stack
from cleverly.utils.parallel import map_parallel
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent

#: The scenario whose law, truths and in-sample fit the diagnostic reuses.
SCENARIO = "binary_logistic"
#: The replicates at each sample size: the n ladder.
ARMS: dict[int, int] = {1_000: 2_000, 4_000: 1_000}
#: The declared outcome support of the cross-fitted fit, as the removed scenario declared it.
Q_BOUNDS = (-40.0, 45.0)
FITTINGS = ("crossfit", "insample")
PARAMETERS = tuple(study.BINARY_PARAMETERS)
ROWS = study.ROWS
CRITICAL = 1.959964
LABEL = "x26-crossfit-bias"


def seed_of(n: int, replicate: int) -> int:
    """The seed of replicate ``replicate`` at sample size ``n``, distinct from the study's."""
    return stream_seed(study.STUDY, LABEL, n, replicate)


@contextmanager
def out_of_fold_allowed() -> Iterator[None]:
    """Let the plug-in limits read an out-of-fold mechanism, and restore the refusal after."""
    previous = _mechanism_stack.OUT_OF_FOLD_REFUSED
    _mechanism_stack.OUT_OF_FOLD_REFUSED = False
    try:
        yield
    finally:
        _mechanism_stack.OUT_OF_FOLD_REFUSED = previous


def fit_crossfit(frame: pd.DataFrame) -> object:
    """The cross-fitted fit that the removed ``binary_logistic_crossfit`` scenario declared."""
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(C=np.inf, max_iter=2000),
            cross_fit=True,
            n_folds=5,
            q_bounds=Q_BOUNDS,
            simultaneous=False,
            estimands=("ate", "att", "atc"),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def replicate_rows(n: int, replicate: int) -> list[dict[str, float | int | str]]:
    """The rows of one draw: both fittings, each parameter."""
    frame, truth = study.draw_from_seed(SCENARIO, n, seed_of(n, replicate))
    out: list[dict[str, float | int | str]] = []
    with out_of_fold_allowed():
        results = {"crossfit": fit_crossfit(frame), "insample": study.fit(frame, SCENARIO)}
        for fitting, result in results.items():
            for token in PARAMETERS:
                values = study.parameter_rows(result, SCENARIO, token)
                row: dict[str, float | int | str] = {
                    "n": n,
                    "replicate": replicate,
                    "fitting": fitting,
                    "parameter": token,
                }
                for name in ROWS:
                    estimate, std_error, _ = values[name]
                    row[name] = estimate
                    row[f"{name}_se"] = std_error
                    row[f"{name}_truth"] = float(truth[study.estimand_key(token, name)])
                out.append(row)
    return out


def _replicate(n: int, replicate: int) -> list[dict[str, float | int | str]]:
    return replicate_rows(n, replicate)


def wilson(covered: int, total: int, level: float = 0.99) -> tuple[float, float]:
    """The Wilson score interval of a proportion."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = covered / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """Per n, fitting, parameter and row: coverage, its 99% Wilson interval, bias and SE ratio.

    ``bias_sd`` is the mean estimate minus the truth over the empirical standard deviation of
    the estimate.  ``se_ratio`` is the mean reported standard error over that deviation.
    """
    out = []
    for n in ARMS:
        for fitting in FITTINGS:
            for token in PARAMETERS:
                cell = rows.loc[
                    (rows["n"] == n) & (rows["fitting"] == fitting) & (rows["parameter"] == token)
                ]
                for name in ROWS:
                    estimate = cell[name].to_numpy()
                    error = cell[f"{name}_se"].to_numpy()
                    truth = float(cell[f"{name}_truth"].iloc[0])
                    covered = int(np.sum(np.abs(estimate - truth) <= CRITICAL * error))
                    low, high = wilson(covered, len(cell))
                    spread = float(estimate.std(ddof=1))
                    out.append(
                        {
                            "n": n,
                            "fitting": fitting,
                            "parameter": token,
                            "row": name,
                            "replicates": len(cell),
                            "coverage": covered / len(cell),
                            "coverage_lower99": low,
                            "coverage_upper99": high,
                            "bias_sd": (float(estimate.mean()) - truth) / spread,
                            "se_ratio": float(error.mean()) / spread,
                        }
                    )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    payloads = [(n, k) for n, count in ARMS.items() for k in range(count)]
    records = map_parallel(_replicate, payloads, n_jobs=arguments.jobs)
    rows = pd.DataFrame([row for batch in records for row in batch])
    rows["fitting"] = pd.Categorical(rows["fitting"], FITTINGS, ordered=True)
    rows = rows.sort_values(["n", "replicate", "fitting", "parameter"], ignore_index=True)
    rows["fitting"] = rows["fitting"].astype(str)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g", lineterminator="\n"
    )
    reading(rows).to_csv(
        arguments.output / "reading.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
