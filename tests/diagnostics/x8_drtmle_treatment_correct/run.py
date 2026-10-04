"""Where the ``treatment_correct`` bias of the stratified DR-TMLE study comes from.

``canonical-stratified-drtmle`` published its four ``double_robustness/*__treatment_correct``
cells red: the marginal ATE and each stratum ATE carry a positive bias of 0.34 to 0.60 of their
spread, with a correct treatment GLM and an outcome GLM that omits ``W1 W2``.  This diagnostic
refits the first ``REPLICATES`` declared draws of that cell, at the declared seeds, in four arms:

=========  =========================================================================================
arm        fit
=========  =========================================================================================
``C``      the declared stratified ``DRTMLE`` fit.  It must reproduce the committed estimates
``S``      the shipped unstratified ``DRTMLE`` on each stratum's rows, with the rows' own folds and
           the same two GLMs without ``V``, fitted on the stratum
``M``      the shipped unstratified ``DRTMLE`` on every row, with ``V`` a covariate: the marginal ATE
``R``      R ``drtmle`` 1.1.2 on each stratum's rows, handed arm ``C``'s initial arrays and folds
=========  =========================================================================================

A bias that ``S``, ``M`` and ``R`` share is a property of the estimator on this law, not of the
stratified construction.

``--part contraction`` refits arms ``C`` and ``M`` at ``CONTRACTION_N`` rows on fresh draws, seeded
apart from every declared stream, into ``contraction-rows.csv.gz``.  A bias that shrinks faster
than the spread is a finite-sample bias.  The rows go to ``rows.csv.gz`` and the validation of arm ``C`` against
the committed rows to ``validation.csv``.  ``tests/unit/test_band_shortfall_reading.py`` rebuilds
the reading from both.

    python -m tests.diagnostics.x8_drtmle_treatment_correct.run --jobs 16 --scratch <dir>
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.utils.parallel import map_parallel
from tests.canonical.drtmle_stratified.regenerate import REFERENCE
from tests.studies import canonical_stratified_drtmle as study
from tests.studies import stratified_drtmle_properties as properties
from tests.studies.canonical_drtmle import ColumnLogistic, FixedFoldDRTMLE
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
CONFIGURATION = "treatment_correct"
REPLICATES = 400
CONTRACTION_N = 8_000
CONTRACTION_REPLICATES = 200
STRATA = (0, 1, 2)


class SubsetOutcome(BaseEstimator, ClassifierMixin):
    """The wrong outcome GLM inside one stratum: ``(A, W1, W2)`` without ``W1 W2``.

    The design is ``[A, W1, W2, W12]``.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> SubsetOutcome:
        values = np.asarray(design, dtype=float)[:, :3]
        self.model_ = LogisticRegression(
            C=np.inf, max_iter=5000, solver="newton-cholesky", tol=1e-10, random_state=0
        ).fit(values, target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)[:, :3]
        return np.asarray(self.model_.predict_proba(values), dtype=float)


def _settings() -> dict[str, Any]:
    return {
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": ColumnLogistic(),
        "cross_fit": True,
        "n_folds": study.N_FOLDS,
        "stratify_folds": "none",
        "estimands": ("ey", "ate"),
        "simultaneous": False,
        "g_bounds": study.G_BOUNDS,
        "max_outer": study.MAX_OUTER,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
        "guard": ("Q", "g"),
    }


def _contiguous(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def _fit(payload: tuple[int]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    (replicate,) = payload
    frame, truth = properties.draw_from_seed(
        study.SCENARIO, properties.PROPERTY_N, properties._seed(CONFIGURATION, replicate)
    )
    rows: list[dict[str, Any]] = []

    def record(arm: str, name: str, estimate: Any) -> None:
        rows.append(
            {
                "arm": arm,
                "replicate": replicate,
                "estimand": name,
                "truth": float(truth[name]),
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
            }
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stratified = study.fit_cleverly(frame, CONFIGURATION)
        for name in ("ate", *(f"ate[V={s}]" for s in STRATA)):
            record("C", name, stratified[name])
        outcome, treatment = study.learners(CONFIGURATION)
        marginal = (
            FixedFoldDRTMLE(
                frame["fold"].to_numpy(dtype=np.int64),
                outcome_learner=outcome,
                treatment_learner=treatment,
                **_settings(),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=list(study.COVARIATES))
            .single()
        )
        record("M", "ate", marginal["ate"])
        for s in STRATA:
            inside = frame[frame["V"] == s].reset_index(drop=True)
            alone = (
                FixedFoldDRTMLE(
                    _contiguous(inside["fold"].to_numpy(dtype=np.int64)),
                    outcome_learner=SubsetOutcome(),
                    treatment_learner=ColumnLogistic(),
                    **_settings(),
                )
                .fit(inside, outcome="Y", treatment="A", covariates=["W1", "W2", "W12"])
                .single()
            )
            record("S", f"ate[V={s}]", alone["ate"])
    nuisance = stratified.repeats[0].nuisance
    sample = frame.assign(
        qn0=nuisance.outcome.arms[0.0],
        qn1=nuisance.outcome.arms[1.0],
        gn1=nuisance.propensity.arm(1.0),
    )
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", study.SCENARIO)
    return rows, sample


def _contraction(payload: tuple[int]) -> list[dict[str, Any]]:
    """Arms ``C`` and ``M`` at ``CONTRACTION_N`` rows, on a fresh draw."""
    (replicate,) = payload
    seed = stream_seed(study.STUDY, "diagnostic", "x8_drtmle_treatment_correct", replicate)
    frame, truth = properties.draw_from_seed(study.SCENARIO, CONTRACTION_N, seed)
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stratified = study.fit_cleverly(frame, CONFIGURATION)
        outcome, treatment = study.learners(CONFIGURATION)
        marginal = (
            FixedFoldDRTMLE(
                frame["fold"].to_numpy(dtype=np.int64),
                outcome_learner=outcome,
                treatment_learner=treatment,
                **_settings(),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=list(study.COVARIATES))
            .single()
        )
    for arm, result, names in (
        ("C", stratified, ("ate", *(f"ate[V={s}]" for s in STRATA))),
        ("M", marginal, ("ate",)),
    ):
        for name in names:
            rows.append(
                {
                    "arm": arm,
                    "replicate": replicate,
                    "estimand": name,
                    "truth": float(truth[name]),
                    "estimate": float(result[name].psi),
                    "std_error": float(result[name].std_error),
                }
            )
    return rows


def _reference(samples: pd.DataFrame, scratch: Path) -> pd.DataFrame:
    """Arm ``R``: the S2 runner on arm ``C``'s arrays, through the study's own container."""
    scratch.mkdir(parents=True, exist_ok=True)
    samples_path = scratch / "samples.csv.gz"
    truths_path = scratch / "truths.csv"
    output_path = scratch / "reference.csv"
    samples.to_csv(samples_path, index=False, compression="gzip")
    truth = study.law.paper_truths()
    pd.DataFrame(
        [
            {"scenario": study.SCENARIO, "replicate": r, "estimand": name, "truth": truth[name]}
            for r in sorted(samples["replicate"].unique())
            for name in study.ESTIMANDS
        ]
    ).to_csv(truths_path, index=False)
    REFERENCE.run(
        ROOT / "tests" / "canonical" / "drtmle_stratified",
        samples_path,
        truths_path,
        output_path,
        cores=16,
    )
    reference = pd.read_csv(output_path, float_precision="round_trip")
    reference = reference.loc[reference["estimand"].str.startswith("ate")]
    return pd.DataFrame(
        {
            "arm": "R",
            "replicate": reference["replicate"].astype(int),
            "estimand": reference["estimand"],
            "truth": reference["truth"],
            "estimate": reference["estimate"],
            "std_error": reference["std_error"],
        }
    )


def _validation(rows: pd.DataFrame) -> pd.DataFrame:
    """Arm ``C`` against the committed rows of the published cells."""
    committed = pd.read_csv(
        ROOT / "tests" / "canonical" / "drtmle_stratified" / "property-replicates.csv.gz",
        float_precision="round_trip",
    )
    out = []
    for name in ("ate", *(f"ate[V={s}]" for s in STRATA)):
        label = "marginal_ate" if name == "ate" else f"v{name[-2]}_ate"
        published = committed.loc[
            (committed["property"] == "double_robustness")
            & (committed["cell"] == f"{label}__{CONFIGURATION}")
            & (committed["replicate"] < REPLICATES)
        ].set_index("replicate")
        refit = rows.loc[(rows["arm"] == "C") & (rows["estimand"] == name)].set_index("replicate")
        joined = published.join(refit, rsuffix="_refit", how="inner")
        out.append(
            {
                "estimand": name,
                "replicates": len(joined),
                "max_abs_estimate_difference": float(
                    (joined["estimate"] - joined["estimate_refit"]).abs().max()
                ),
            }
        )
    return pd.DataFrame(out)


def _gzip(frame: pd.DataFrame, path: Path) -> None:
    with open(path, "wb") as handle:
        frame.to_csv(
            handle, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--part", choices=("arms", "contraction"), default="arms")
    arguments = parser.parse_args()
    if arguments.part == "contraction":
        payloads = [((r,),) for r in range(CONTRACTION_REPLICATES)]
        found = map_parallel(_contraction, payloads, n_jobs=arguments.jobs)
        rows = pd.DataFrame([row for result in found for row in result])
        rows = rows.sort_values(["arm", "estimand", "replicate"], ignore_index=True)
        _gzip(rows, HERE / "contraction-rows.csv.gz")
        print(rows.groupby(["estimand", "arm"]).apply(_summary).round(4).to_string())
        return
    if arguments.scratch is None:
        parser.error("--scratch is required for the arms part")
    outcomes = map_parallel(_fit, [((r,),) for r in range(REPLICATES)], n_jobs=arguments.jobs)
    rows = pd.DataFrame([row for result, _ in outcomes for row in result])
    samples = pd.concat([sample for _, sample in outcomes], ignore_index=True)
    rows = pd.concat([rows, _reference(samples, arguments.scratch)], ignore_index=True)
    rows = rows.sort_values(["arm", "estimand", "replicate"], ignore_index=True)
    _gzip(rows, HERE / "rows.csv.gz")
    validation = _validation(rows)
    validation.to_csv(HERE / "validation.csv", index=False, lineterminator="\n")
    shutil.rmtree(arguments.scratch, ignore_errors=True)
    print(validation.to_string())
    print(rows.groupby(["estimand", "arm"]).apply(_summary).round(4).to_string())


def _summary(group: pd.DataFrame) -> pd.Series:
    spread = float(group["estimate"].std(ddof=1))
    error = group["estimate"] - group["truth"]
    return pd.Series(
        {
            "replicates": len(group),
            "standardized_bias": float(error.mean()) / spread,
            "se_ratio": float(group["std_error"].mean()) / spread,
            "coverage": float((error.abs() <= 1.959964 * group["std_error"]).mean()),
        }
    )


if __name__ == "__main__":
    main()
