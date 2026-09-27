"""Parts BD-P and FW-B: the three weighted mean rows of `weighted-ltmle-crossfit`, compared again.

Both implementations fit each primary draw.  The paired comparison then runs twice through the
framework: once with the ``lmtp`` rows as committed (``native``) and once with the ``lmtp``
standard error replaced by its ``hajek`` convention.  Step 1 does this on the committed 800
draws.  The pilot validates the harness on the registered draws, fits 800 fresh draws on a
throwaway seed and sets the step 2 budget ``R_p``.  Step 2 fits ``R_p`` fresh draws.
"""

from __future__ import annotations

import gzip
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.canonical.weighted_lmtp_ltmle.regenerate import REFERENCE
from tests.diagnostics import rm18_shared as shared
from tests.studies import canonical_weighted_ltmle_crossfit as study
from tests.studies import weighted_longitudinal_common as primary
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.red_cells import paired_legs
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed, stream_seed

CROSSFIT = study.STUDY
SUBJECT = str(CROSSFIT.implementation)
COMPARATOR = str(CROSSFIT.reference)
#: The three weighted mean rows.  FW-B reads ``ey_regimen[never]``, BD-P the other two.
MEANS = primary.MEAN_NAMES
FW_ROW = "ey_regimen[never]"
BD_ROWS = tuple(name for name in MEANS if name != FW_ROW)
CONVENTIONS = ("native", "hajek")
#: The two-sided 95% critical value the ``lmtp`` rows and ``reference_artifacts`` use.
CRITICAL = 1.959963984540054

PILOT_REPLICATES = 800
TARGET_RESOLUTION = 0.025
BUDGET_FLOOR = 800
BUDGET_CAP = 20_000
PILOT_RECORD = replace(CROSSFIT, seed=stream_seed(CROSSFIT, "rm18", "boundary-pilot"))
PAIRED_RECORD = replace(CROSSFIT, seed=stream_seed(CROSSFIT, "rm18", "boundary-paired"))
#: Python fits per batch before the samples reach the gzip stream.
BATCH = 500
#: The equivalence and performance bootstraps gather ``1,000 x R`` draws per column, so the
#: comparison step runs at most this many cells at once.
COMPARISON_JOBS = 3
#: The committed row columns, and the one ``hajek`` column the paired rows add.
PAIRED_COLUMNS = (
    "implementation",
    "replicate",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "hajek_std_error",
)

EQUIVALENT = "equivalent at the declared budget"
CONVENTION = "comparator SE convention"
DEFICIT = "deficit in cleverly"
UNRESOLVED = "unresolved"
PASSING = ("equivalent", "superior")


# ------------------------------------------------------------------------ the paired rows


def paired_rows(rows: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    """The three mean rows of both implementations, with the ``hajek`` standard error of each
    ``lmtp`` row beside its native one."""
    means = rows.loc[rows["estimand"].isin(MEANS)]
    hajek = inference.loc[
        inference["inference_method"] == "hajek", ["replicate", "estimand", "std_error"]
    ].rename(columns={"std_error": "hajek_std_error"})
    merged = means.merge(hajek.assign(implementation=COMPARATOR), how="left", validate="one_to_one")
    comparator = merged["implementation"] == COMPARATOR
    if merged.loc[comparator, "hajek_std_error"].isna().any():
        raise RuntimeError("an lmtp row has no hajek standard error")
    return merged.loc[:, list(PAIRED_COLUMNS)].sort_values(
        ["replicate", "estimand", "implementation"], ignore_index=True
    )


def with_convention(paired: pd.DataFrame, convention: str) -> pd.DataFrame:
    """The registered replicate schema, with the ``lmtp`` inference of one convention."""
    rows = paired.copy()
    if convention == "hajek":
        comparator = rows["implementation"] == COMPARATOR
        rows.loc[comparator, "std_error"] = rows.loc[comparator, "hajek_std_error"]
        low = rows["estimate"] - CRITICAL * rows["std_error"]
        high = rows["estimate"] + CRITICAL * rows["std_error"]
        rows.loc[comparator, "covered"] = ((low <= rows["truth"]) & (rows["truth"] <= high)).astype(
            int
        )[comparator]
    elif convention != "native":
        raise ValueError(f"unknown convention {convention!r}")
    rows["scenario"] = primary.SCENARIO
    rows["n"] = primary.PRIMARY_N
    rows["inference_estimate"] = rows["estimate"]
    rows["ci_lower"] = rows["estimate"] - CRITICAL * rows["std_error"]
    rows["ci_upper"] = rows["estimate"] + CRITICAL * rows["std_error"]
    rows["inference_scale"] = "identity"
    rows["initial_estimate"] = math.nan
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def comparisons(paired: pd.DataFrame, record: Any, jobs: int) -> pd.DataFrame:
    """The framework comparison of each mean row under each convention, with the subject's
    own coverage interval and SE ratio beside it."""
    frames = []
    for convention in CONVENTIONS:
        rows = with_convention(paired, convention)
        workers = min(jobs, COMPARISON_JOBS)
        performance = independent_performance_tests(rows, record=record, n_jobs=workers)
        compared = equivalence(rows, summarize(rows), performance, record=record, n_jobs=workers)
        own = performance.loc[
            performance["implementation"] == SUBJECT,
            ["estimand", "coverage_ci_lower", "coverage_ci_upper", "se_ratio"],
        ].rename(
            columns={
                "coverage_ci_lower": "subject_coverage_ci_lower",
                "coverage_ci_upper": "subject_coverage_ci_upper",
                "se_ratio": "subject_se_ratio",
            }
        )
        frames.append(compared.merge(own, on="estimand").assign(convention=convention))
    return pd.concat(frames, ignore_index=True)


def committed_paired() -> pd.DataFrame:
    """The committed 800 draws as paired rows (step 1)."""
    return paired_rows(
        shared.read_rows(CROSSFIT.artifact("replicates.csv.gz")),
        shared.read_rows(CROSSFIT.artifact("reference-inference.csv.gz")),
    )


EQUIVALENCE_LEGS = (
    "paired_ci_lower",
    "paired_ci_upper",
    "mean_margin",
    "rmse_ratio_upper",
    "coverage_difference_lower",
    "calibration_excess_upper",
    "calibration_excess_resolution",
)


def reproduces_committed(compared: pd.DataFrame) -> tuple[bool, float, int]:
    """Whether the ``native`` comparison of the committed rows reproduces ``equivalence.csv``."""
    committed = shared.read_rows(CROSSFIT.artifact("equivalence.csv")).set_index("estimand")
    native = compared.loc[compared["convention"] == "native"].set_index("estimand")
    largest = 0.0
    same = True
    for estimand in MEANS:
        for column in EQUIVALENCE_LEGS:
            largest = max(
                largest,
                float(
                    shared.scaled_difference(
                        native.loc[estimand, column], committed.loc[estimand, column]
                    )
                ),
            )
        same = (
            same
            and native.loc[estimand, "comparison_conclusion"]
            == committed.loc[estimand, "comparison_conclusion"]
        )
    return bool(same and largest <= shared.TOLERANCE), largest, len(MEANS)


# --------------------------------------------------------------------- both implementations


def draw_both(record: Any, replicates: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ``replicates`` primary draws of ``record`` with both implementations.

    The Python phase runs in batches that stream the samples into one gzip file, then the
    registered ``lmtp`` runner fits that file.  The two phases never overlap.  A replicate or
    an estimand missing from either implementation stops the part: ``mclapply`` drops a
    killed worker without an error, so the count is checked here as well as in R.
    """
    with tempfile.TemporaryDirectory(prefix="cleverly-rm18-paired-") as raw:
        scratch = Path(raw)
        samples = scratch / "samples.csv.gz"
        truths: list[dict[str, Any]] = []
        estimates: list[dict[str, Any]] = []
        header = True
        with gzip.open(samples, "wt", encoding="utf-8", newline="") as handle:
            for start in range(0, replicates, BATCH):
                stop = min(start + BATCH, replicates)
                outcomes = map_parallel(
                    primary._replicate_dispatch,
                    [
                        (record, (primary.SCENARIO, index, primary.PRIMARY_N), True)
                        for index in range(start, stop)
                    ],
                    n_jobs=jobs,
                )
                for sample, truth_rows, rows in outcomes:
                    sample.to_csv(handle, header=header, index=False, lineterminator="\n")
                    header = False
                    truths.extend(truth_rows)
                    estimates.extend(rows)
        write_csv(pd.DataFrame(truths), scratch / "truth.csv")
        results = scratch / "reference-results.csv"
        REFERENCE.run(study.STUDY.artifacts, samples, scratch / "truth.csv", results, cores=jobs)
        reference = pd.read_csv(results)
        inference = study.reference_artifacts(
            reference=REFERENCE,
            here=CROSSFIT.artifacts,
            samples=samples,
            truths_path=scratch / "truth.csv",
            reference_results=results,
            output=scratch,
            cores=jobs,
        )["reference-inference.csv.gz"]
    rows = pd.concat(
        [pd.DataFrame(estimates), reference.loc[:, list(REPLICATE_COLUMNS)]], ignore_index=True
    )
    expected = {
        (implementation, index, estimand)
        for implementation in (SUBJECT, COMPARATOR)
        for index in range(replicates)
        for estimand in primary.ESTIMANDS
    }
    observed = list(zip(rows["implementation"], rows["replicate"], rows["estimand"], strict=True))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        missing = sorted(expected - set(observed))[:10]
        raise RuntimeError(
            f"{len(expected - set(observed))} implementation-replicate-estimand rows are missing "
            f"and {len(observed) - len(set(observed))} repeat; first missing {missing}. A killed "
            f"worker returns nothing, and a shorter comparison is not the declared one"
        )
    return rows, inference


def registered_draws_reproduce(jobs: int, cap: int | None) -> list[dict[str, Any]]:
    """The BD-P harness validation: the registered draws, through the step 2 path."""
    replicates = shared.budget(CROSSFIT.replicates, cap)
    rows, inference = draw_both(CROSSFIT, replicates, jobs)
    committed = shared.read_rows(CROSSFIT.artifact("replicates.csv.gz"))
    committed = committed.loc[committed["replicate"] < replicates]
    keys = ["implementation", "replicate", "estimand"]
    out = [
        shared.validation_row(
            "BD-P-pilot",
            "registered rows, both implementations",
            shared.compare_rows(rows, committed, keys),
        )
    ]
    stored = shared.read_rows(CROSSFIT.artifact("reference-inference.csv.gz"))
    stored = stored.loc[stored["replicate"] < replicates]
    out.append(
        shared.validation_row(
            "BD-P-pilot",
            "registered lmtp conventions",
            shared.compare_rows(
                inference,
                stored,
                ["inference_method", "replicate", "estimand"],
                ("estimate", "std_error"),
            ),
        )
    )
    return out


# ------------------------------------------------------------------------ budget and readings


def paired_budget(pilot: pd.DataFrame) -> tuple[float, int]:
    """``r_pilot`` and ``R_p = min(20,000, max(800, ceil(800 * (r_pilot / 0.025)^2)))``."""
    means = pilot.loc[pilot["estimand"].isin(MEANS) & pilot["convention"].isin(CONVENTIONS)]
    if len(means) != len(MEANS) * len(CONVENTIONS):
        raise RuntimeError("the pilot does not hold every mean row under both conventions")
    resolution = float(means["calibration_excess_resolution"].max())
    if not math.isfinite(resolution):
        return resolution, BUDGET_CAP
    wanted = math.ceil(PILOT_REPLICATES * (resolution / TARGET_RESOLUTION) ** 2)
    return resolution, min(BUDGET_CAP, max(BUDGET_FLOOR, wanted))


def _verdict(row: Any) -> Any:
    return paired_legs(row, CROSSFIT.margins, bool(row.subject_valid))


def bd_p_label(native: Any, hajek: Any) -> str:
    """The declared BD-P reading of one row."""
    native_conclusion = _verdict(native).conclusion
    hajek_conclusion = _verdict(hajek).conclusion
    if native_conclusion in PASSING:
        return EQUIVALENT
    if hajek_conclusion in PASSING:
        return CONVENTION
    return f"native {native_conclusion}; hajek {hajek_conclusion}"


def fw_b_label(native: Any, hajek: Any) -> str:
    """The declared FW-B reading of ``ey_regimen[never]``."""
    first, second = _verdict(native), _verdict(hajek)
    if first.conclusion in PASSING:
        return EQUIVALENT
    if (not first.coverage or not first.calibration) and (
        second.similar and second.rmse and second.coverage and second.calibration
    ):
        return CONVENTION
    if (
        not second.coverage
        and float(hajek.calibration_excess_resolution)
        <= CROSSFIT.margins.calibration_noninferiority
        and float(native.subject_coverage_ci_upper) < CROSSFIT.margins.calibration_coverage[0]
    ):
        return DEFICIT
    return UNRESOLVED


def comparison_readings(
    part: str, compared: pd.DataFrame, *, labelled: bool
) -> list[dict[str, Any]]:
    """Every leg of every row under both conventions, and the readings when ``labelled``."""
    out = []
    for row in compared.itertuples(index=False):
        scope = f"{row.estimand}, {row.convention}"
        out += [
            shared.reading(
                part,
                scope,
                "paired difference",
                interval=Interval(row.paired_ci_lower, row.paired_ci_upper),
                value=float(row.mean_margin),
                result="value is the margin",
            ),
            shared.reading(part, scope, "RMSE ratio bound", value=float(row.rmse_ratio_upper)),
            shared.reading(
                part, scope, "coverage difference bound", value=float(row.coverage_difference_lower)
            ),
            shared.reading(
                part, scope, "calibration excess bound", value=float(row.calibration_excess_upper)
            ),
            shared.reading(
                part,
                scope,
                "calibration excess resolution",
                value=float(row.calibration_excess_resolution),
            ),
            shared.reading(part, scope, "conclusion", result=str(_verdict(row).conclusion)),
        ]
    for estimand in MEANS:
        native = compared.loc[
            (compared["estimand"] == estimand) & (compared["convention"] == "native")
        ].iloc[0]
        out += [
            shared.reading(
                part,
                estimand,
                "cleverly coverage",
                interval=Interval(
                    float(native.subject_coverage_ci_lower), float(native.subject_coverage_ci_upper)
                ),
            ),
            shared.reading(
                part, estimand, "cleverly SE ratio", value=float(native.subject_se_ratio)
            ),
        ]
    if labelled:
        for estimand in MEANS:
            pair = {
                convention: next(
                    compared.loc[
                        (compared["estimand"] == estimand) & (compared["convention"] == convention)
                    ].itertuples(index=False)
                )
                for convention in CONVENTIONS
            }
            if estimand == FW_ROW:
                out.append(
                    shared.reading(
                        "FW-B",
                        estimand,
                        "reading",
                        result=fw_b_label(pair["native"], pair["hajek"]),
                    )
                )
            else:
                out.append(
                    shared.reading(
                        "BD-P",
                        estimand,
                        "reading",
                        result=bd_p_label(pair["native"], pair["hajek"]),
                    )
                )
    return out


def pilot_table(compared: pd.DataFrame) -> pd.DataFrame:
    """``pilot.csv``: the resolution of each mean row under each convention."""
    return compared.loc[
        :, ["convention", "estimand", "calibration_excess_upper", "calibration_excess_resolution"]
    ]


def seeds(record: Any, replicates: int) -> set[int]:
    """The primary sample seeds of ``replicates`` draws of ``record``."""
    return {replicate_seed(record, primary.SCENARIO, index) for index in range(replicates)}


def registered_seed_set() -> set[int]:
    """Every registered seed of the cross-fitted weighted study."""
    return shared.weighted_registered_seeds(CROSSFIT)


def disjoint() -> bool:
    """The pilot and every step 2 draw up to the cap miss the registered seeds and each other."""
    pilot, paired = seeds(PILOT_RECORD, PILOT_REPLICATES), seeds(PAIRED_RECORD, BUDGET_CAP)
    registered = registered_seed_set()
    return bool(
        pilot.isdisjoint(registered)
        and paired.isdisjoint(registered)
        and pilot.isdisjoint(paired)
        and len(pilot) == PILOT_REPLICATES
        and len(paired) == BUDGET_CAP
    )
