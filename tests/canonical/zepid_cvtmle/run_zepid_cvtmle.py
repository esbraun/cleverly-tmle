"""Run zEpid's native single-crossfit TMLE on the study's exact rows and folds."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assert_native_split_identity(
    frame: pd.DataFrame, native_splits: Sequence[pd.DataFrame]
) -> None:
    """Abort unless every native split contains the declared row identities."""
    labels = sorted(frame["fold"].unique().tolist())
    if labels != list(range(len(native_splits))):
        raise RuntimeError(
            f"declared fold labels {labels} do not identify {len(native_splits)} native splits"
        )
    for fold, split in enumerate(native_splits):
        expected = set(frame.loc[frame["fold"] == fold, "row_id"].astype(int))
        observed = set(split["row_id"].astype(int))
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            unexpected = sorted(observed - expected)[:5]
            raise RuntimeError(
                "zEpid native split changed the declared row identities for "
                f"fold {fold}: missing={missing}, unexpected={unexpected}"
            )


def _fit_one(payload: tuple[pd.DataFrame, float]) -> dict[str, Any]:
    frame, truth = payload
    from sklearn.linear_model import LogisticRegression
    from zepid.causal.doublyrobust import SingleCrossfitTMLE
    from zepid.causal.doublyrobust import crossfit as crossfit_module

    scenario = str(frame["scenario"].iloc[0])
    replicate = int(frame["replicate"].iloc[0])
    partition_random_state = int(frame["partition_random_state"].iloc[0])
    native_split = crossfit_module._sample_split_
    native_targeting = crossfit_module.targeting_step
    initial_estimates: list[float] = []

    def checked_split(data: pd.DataFrame, n_splits: int, random_state: int) -> Any:
        splits = native_split(data, n_splits=n_splits, random_state=random_state)
        assert_native_split_identity(data, splits)
        return splits

    def recorded_targeting(*args: Any, **kwargs: Any) -> Any:
        py_a = kwargs["py_a"] if "py_a" in kwargs else args[2]
        py_n = kwargs["py_n"] if "py_n" in kwargs else args[3]
        initial_estimates.append(float(np.mean(np.asarray(py_a) - np.asarray(py_n))))
        return native_targeting(*args, **kwargs)

    crossfit_module._sample_split_ = checked_split
    crossfit_module.targeting_step = recorded_targeting
    try:
        fit = SingleCrossfitTMLE(frame.copy(), exposure="A", outcome="Y", alpha=0.05)
        learner = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
        covariates = " + ".join(column for column in frame if column.startswith("W"))
        fit.exposure_model(covariates, estimator=learner, bound=[0.025, 0.975])
        fit.outcome_model(f"A + {covariates}", estimator=learner)
        fit.fit(
            n_splits=2,
            n_partitions=1,
            method="median",
            random_state=partition_random_state,
        )
    finally:
        crossfit_module._sample_split_ = native_split
        crossfit_module.targeting_step = native_targeting

    if len(initial_estimates) != 1:
        raise RuntimeError(
            "the one-partition zEpid fit did not expose exactly one pre-targeting plug-in"
        )

    estimate = float(fit.risk_difference)
    std_error = float(fit.risk_difference_se)
    low, high = (float(value) for value in fit.risk_difference_ci)
    return {
        "implementation": "zepid-single-crossfit-tmle",
        "scenario": scenario,
        "replicate": replicate,
        "n": len(frame),
        "estimand": "ate",
        "truth": truth,
        "estimate": estimate,
        "inference_estimate": estimate,
        "std_error": std_error,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": initial_estimates[0],
    }


def _cores(groups: int) -> int:
    raw = os.getenv("CLEVERLY_REFERENCE_CORES", os.getenv("CLEVERLY_R_CORES", "1"))
    try:
        requested = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid reference core count {raw!r}") from error
    return max(1, min(groups, requested))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_zepid_cvtmle.py SAMPLES.csv.gz TRUTH.csv OUTPUT.csv")
    samples_path, truths_path, output_path = map(Path, sys.argv[1:])
    samples = pd.read_csv(samples_path)
    truths = pd.read_csv(truths_path)
    truth_lookup = {
        (str(row.scenario), int(row.replicate)): float(row.truth_ate) for row in truths.itertuples()
    }
    groups = [
        (frame.copy(), truth_lookup[(str(scenario), int(replicate))])
        for (scenario, replicate), frame in samples.groupby(["scenario", "replicate"], sort=True)
    ]
    with ProcessPoolExecutor(max_workers=_cores(len(groups))) as pool:
        rows = list(pool.map(_fit_one, groups))
    if len(rows) != len(groups):
        raise RuntimeError(f"zEpid returned {len(rows)} rows for {len(groups)} replications")
    pd.DataFrame(rows).to_csv(output_path, index=False, na_rep="NA")


if __name__ == "__main__":
    main()
