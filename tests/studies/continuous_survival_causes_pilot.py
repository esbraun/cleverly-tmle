"""The pilot of the ``continuous-survival-causes`` study, on its own seed streams.

It draws no declared sample.  Draw ``r`` of a primary cell takes
``stream_seed(STUDY, "pilot", cell, r)`` and draw ``r`` of a law's controls takes
``stream_seed(STUDY, "pilot", f"{law}_controls", r)``, streams no declared cell reads.  The driver
writes one file beside this module, ``continuous_survival_causes_pilot.csv.gz``, with a ``kind``
column:

* ``primary``: one fit of a cell's declared configuration
  (:func:`tests.studies.continuous_survival_causes.engine`), one row per target (first quartile
  included), or one row with the error class when the fit raised;
* ``controls``: on one draw of a law, the double-robustness configuration (each cause's hazard
  from the Cox model on the treatment alone, the default propensity and censoring, which hold the
  truth) fitted twice, with the shipped tilt and with ``tilt="factual"``; one row per declared
  double-robustness label (:data:`~tests.studies.continuous_survival_causes_properties.ROBUSTNESS_LABELS`)
  with the targeted estimate, its standard error and the initial plug-in of both fits.

The committed rows give these readings of X6 plan rev 5, section 5.3, which the regeneration script
checks before a declared run (``regenerate.pilot_refusals``):

* **the primary count** (:func:`primary_count`): the shared primary rule on the 24 gated rows'
  covered counts and one bias group per cell, at least 1,000 (``guide/studies.md``);
* **the calibration count** (:func:`calibration_count`): the arm-pooled predictive rule of
  ``guide/studies.md`` (:func:`~tests.studies.repeated_sizing.calibration_replicates`), the smallest
  multiple of 400 at which the two calibration cells, one per law, pass together with probability
  at least 0.95.  Their pilot SE ratios form one arm; the two laws share no draw, so their noise
  is independent.  Beside it the at-target count (:func:`at_target_count`) and each cell's
  plug-in pass probability at the declared count (:func:`calibration_checks`): a cell below 0.90
  is a departure that asks for a committed diagnosis before the declaration.  The rule reads
  every primary draw the file holds.  On the declared 800 draws it could not size the family at
  any count up to 50,000 (pilot ratios 0.970 and 0.969, predictive SD near half the band), so,
  as ``guide/studies.md`` directs for that case, both cells were extended to
  :data:`EXTENSION_DRAWS` on the same streams (``--kinds primary --draws 2000 --resume``).  The
  primary count and the seconds per fit stay read on the first 800 draws;
* **the control readings** (:func:`control_readings`): each double-robustness cell's pilot
  standardized bias and its pass probability at
  :data:`~tests.studies.continuous_survival_causes_properties.ROBUSTNESS_REPLICATES`: a positive
  cell must establish a bias inside 0.2 SD, a control one beyond it.  A control whose pass
  probability is below 0.80 (``property_verdicts.MINIMUM_POWER``) is uninformative at this law,
  and the declaration says so before the run;
* **the failure share and the predicted failure bound** (:func:`failure_refusals`): the posterior
  predictive probability, under a Jeffreys prior on a cell's failure rate, that the declared run
  loses more fits than the failure accounting allows; above :data:`FAILURE_EXCEEDANCE` the
  declaration is refused;
* **the shrunk-SE control's power** (:func:`shrunken_power`), at the declared calibration count;
* **seconds per fit** (:func:`seconds`), for the budget.

Run it as ``python -m tests.studies.continuous_survival_causes_pilot --jobs N [--kinds ...]``.  It
writes the file after every block of :data:`BLOCK` draws, so ``--resume`` continues each cell from
its first draw the file does not hold, on the same streams.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_causes as study
from tests.studies import continuous_survival_causes_law as laws
from tests.studies import continuous_survival_causes_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.failures import allowed
from tests.studies.evidence.property_verdicts import MINIMUM_POWER
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_survival_causes_pilot.csv.gz")
DRAWS = study.PILOT_REPLICATES
#: The primary cells' extension, which sets the calibration count (the module docstring says why).
EXTENSION_DRAWS = 2_000
#: The control pilot's draws per law: enough to read a standardized bias to about 0.05 SD.
CONTROL_DRAWS = 400
BLOCK = 100
#: The plug-in pass probability below which ``guide/studies.md`` asks for a longer pilot or a
#: committed diagnosis.
PLUG_IN_PASS = 0.90
#: The largest posterior predictive probability that a cell's declared run exceeds its failure
#: allowance at which the declaration may proceed (as ``continuous-survival-variance``'s).
FAILURE_EXCEEDANCE = 0.10
#: The committed diagnosis a departure needs before the declaration.
DIAGNOSIS = study.STUDY.artifacts / "pilot-diagnosis.md"


def pilot_seed(label: str, replicate: int) -> int:
    """The seed of pilot draw ``replicate`` of a cell, or of ``f"{law}_controls"``."""
    return stream_seed(study.STUDY, "pilot", label, replicate)


def _primary(cell: str, replicate: int) -> list[dict[str, Any]]:
    law = study.law_of(cell)
    truth = laws.truth(study.CELLS[cell])
    frame = law.sample(study.PRIMARY_N, pilot_seed(cell, replicate))
    head = {"kind": "primary", "label": cell, "replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = registered.fit_with(frame, study.engine(law))
            seconds = time.perf_counter() - start
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**head, "refusal": type(error).__name__}]
    initial = result.initial_estimates
    early = set(laws.first_quartile(study.CELLS[cell]))
    rows = []
    for estimand, value in truth.items():
        estimate = result[estimand]
        low, high = estimate.ci
        record = estimate.switched
        rows.append(
            {
                **head,
                "estimand": estimand,
                "early": estimand in early,
                "truth": float(value),
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ic_std_error": float(np.sqrt(record.influence_variance)),
                "covered": int(low <= value <= high),
                "initial_estimate": float(initial[estimand]),
                "refusal": "",
                "seconds": seconds,
                "steps": float(result.record.steps),
                "converged": float(result.record.converged),
            }
        )
    return rows


def _controls(name: str, replicate: int) -> list[dict[str, Any]]:
    law = laws.LAWS[name]
    truth = laws.truth(name)
    label = f"{name}_controls"
    frame = law.sample(properties.ROBUSTNESS_N, pilot_seed(label, replicate))
    head = {
        "kind": "controls",
        "label": label,
        "replicate": replicate,
        "n": properties.ROBUSTNESS_N,
    }
    fits: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tilt in ("all", "factual"):
                start = time.perf_counter()
                fits[tilt] = registered.fit_with(
                    frame, study.engine(law, hazard_wrong=True, tilt=tilt)
                )
                seconds[tilt] = time.perf_counter() - start
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**head, "refusal": type(error).__name__}]
    rows = []
    for robustness_label, estimand in properties.ROBUSTNESS_LABELS[name].items():
        targeted, factual = fits["all"][estimand], fits["factual"][estimand]
        rows.append(
            {
                **head,
                "estimand": estimand,
                "robustness_label": robustness_label,
                "truth": float(truth[estimand]),
                "estimate": float(targeted.psi),
                "std_error": float(targeted.std_error),
                "initial_estimate": float(fits["all"].initial_estimates[estimand]),
                "factual_estimate": float(factual.psi),
                "factual_std_error": float(factual.std_error),
                "refusal": "",
                "seconds": seconds["all"],
                "factual_seconds": seconds["factual"],
            }
        )
    return rows


def _one(kind: str, label: str, replicate: int) -> list[dict[str, Any]]:
    # ``map_parallel`` splats a tuple payload.
    return _primary(label, replicate) if kind == "primary" else _controls(label, replicate)


#: ``kind -> labels``: the primary cells, and the laws whose controls the pilot reads.
KINDS: dict[str, tuple[str, ...]] = {
    "primary": tuple(study.CELLS),
    "controls": tuple(laws.LAWS),
}


def _label(kind: str, label: str) -> str:
    return label if kind == "primary" else f"{label}_controls"


def run(
    n_jobs: int,
    kinds: Sequence[str] = tuple(KINDS),
    *,
    resume: bool = False,
    draws: int | None = None,
) -> pd.DataFrame:
    """Run each kind's draws, writing the file after every block.

    With ``resume`` each label starts at its first draw the file does not hold and every other
    row is kept; without it, the file is replaced.  ``draws`` replaces the kind's declared count
    (``--draws 2000 --kinds primary --resume`` is the declared extension).
    """
    frame = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()
    for kind in kinds:
        count = draws if draws is not None else DRAWS if kind == "primary" else CONTROL_DRAWS
        for label in KINDS[kind]:
            held = frame.loc[frame["label"] == _label(kind, label)] if not frame.empty else frame
            first = int(held["replicate"].max()) + 1 if not held.empty else 0
            if not held.empty and set(held["replicate"].astype(int)) != set(range(first)):
                raise SystemExit(f"{OUTPUT.name} does not hold draws 0 to {first - 1} of {label}")
            for start in range(first, count, BLOCK):
                keys = [(kind, label, r) for r in range(start, min(start + BLOCK, count))]
                outcomes = map_parallel(_one, keys, n_jobs=n_jobs)
                block = pd.DataFrame([row for rows in outcomes for row in rows])
                frame = pd.concat([frame, block], ignore_index=True)
                frame.to_csv(OUTPUT, index=False)
                print(f"{kind} {label}: {min(start + BLOCK, count)} of {count}", flush=True)
    return frame


# -------------------------------------------------------------------------- readings


def read() -> pd.DataFrame:
    """The committed pilot rows, or an empty frame before the pilot is committed."""
    return pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()


def of(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """The rows of one primary cell or of one law's controls (``f"{law}_controls"``)."""
    return rows.loc[rows["label"] == label]


def kept_rows(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """The rows of ``label``'s fits that did not raise."""
    chosen = of(rows, label)
    return chosen.loc[chosen["refusal"].fillna("").eq("")]


def draws(rows: pd.DataFrame, label: str) -> int:
    """The draws ``label`` holds, raised or not."""
    return int(of(rows, label)["replicate"].nunique())


def failed_draws(rows: pd.DataFrame, label: str) -> int:
    """The draws of ``label`` whose fit raised."""
    chosen = of(rows, label)
    return int(chosen.loc[chosen["refusal"].fillna("").ne(""), "replicate"].nunique())


def first_draws(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows of the declared pilot's draws, before any extension."""
    return rows.loc[rows["replicate"] < study.PILOT_REPLICATES]


def primary_count(rows: pd.DataFrame) -> int:
    """The shared primary rule on the gated rows of both cells, at least 1,000.

    Read on each cell's first :data:`~tests.studies.continuous_survival_causes.PILOT_REPLICATES`
    draws: an extension of the pilot for the calibration cells does not resize the primary.

    Clauses (a) to (c) on the covered counts of the shipped interval, clause (d) on one bias group
    per cell, both at :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.  The two cells
    hold the same number of pilot draws, so the rule reads one pilot count.
    """
    covered: dict[str, int] = {}
    groups = []
    counts = set()
    for cell in study.CELLS:
        kept = first_draws(kept_rows(rows, cell))
        names = list(study.ESTIMANDS[cell])
        gated = kept.loc[kept["estimand"].isin(names)]
        hits = gated.pivot(index="replicate", columns="estimand", values="covered")[names]
        counts.add(len(hits))
        covered.update({f"{cell}/{name}": int(hits[name].sum()) for name in names})
        error = gated.assign(error=gated["estimate"] - gated["truth"]).pivot(
            index="replicate", columns="estimand", values="error"
        )[names]
        standardized = error / error.std(ddof=1)
        groups.append(
            sizing.PilotGroup(
                standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(error)
            )
        )
    share = sizing.PRIMARY_CLAUSE_SHARE
    rule = sizing.primary_replicates(
        study.MARGINS,
        covered,
        min(counts),
        implementations=1,
        coverage_pass=share,
        bias_groups=groups,
        bias_pass=share,
    )
    return max(rule, study.MINIMUM_REPLICATES)


def calibration_ratios(rows: pd.DataFrame) -> dict[str, float]:
    """Each calibration cell's pilot SE ratio: mean reported SE over the empirical SD."""
    out = {}
    for label, (cell, estimand) in properties.CALIBRATION_TARGETS.items():
        kept = kept_rows(rows, cell)
        chosen = kept.loc[kept["estimand"] == estimand]
        out[f"{label}__correctly_specified"] = float(
            chosen["std_error"].mean() / chosen["estimate"].std(ddof=1)
        )
    return out


def _independent(labels: Sequence[str]) -> np.ndarray:
    """The two laws share no draw: their cells' noise is independent."""
    return np.eye(len(labels))


def calibration_count(rows: pd.DataFrame) -> int:
    """The arm-pooled predictive rule on the calibration cells' pilot SE ratios.

    Above :data:`~tests.studies.repeated_sizing.MAXIMUM_REPLICATES` when no count meets it.
    """
    ratios = calibration_ratios(rows)
    pilot = min(len(kept_rows(rows, cell)["replicate"].unique()) for cell in study.CELLS)
    try:
        return sizing.calibration_replicates(study.MARGINS, ratios, pilot, _independent)
    except ValueError:
        # No count up to 50,000 meets the rule: a count past every cap, which stops the
        # declaration.
        return sizing.MAXIMUM_REPLICATES + sizing.BUDGET_STEP


def at_target_count() -> int:
    """The smallest multiple of 400 at which both calibration cells pass with 0.95 at ratio 1."""
    cells = len(properties.CALIBRATION_LABELS)
    replicates = sizing.BUDGET_STEP
    while (
        float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0]) ** cells
        < sizing.FAMILY_WISE_PASS
    ):
        replicates += sizing.BUDGET_STEP
    return replicates


def calibration_checks(rows: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Each calibration cell's pilot ratio and its plug-in pass probability at ``replicates``."""
    records = []
    for name, ratio in calibration_ratios(rows).items():
        passed = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([ratio]))[0])
        records.append({"cell": name, "se_ratio": ratio, "plug_in_pass": passed})
    return pd.DataFrame(records)


def shrunken_power(rows: pd.DataFrame, replicates: int) -> dict[str, float]:
    """The probability that each shrunk-SE control leaves the band at ``replicates``."""
    return {
        name.replace("correctly_specified", "shrunken_se_control"): (
            sizing.calibration_control_probability(
                study.MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR * ratio
            )
        )
        for name, ratio in calibration_ratios(rows).items()
    }


def control_readings(rows: pd.DataFrame) -> pd.DataFrame:
    """Each double-robustness cell's pilot standardized bias and pass probability at 1,000.

    The positive cell reads the targeted estimate, the plug-in control the initial estimate of
    the same fit, and the factual-tilt control the ``tilt="factual"`` fit of the same draw; each
    is standardized by its own empirical SD, as the bias verdict reads it.
    """
    records = []
    replicates = properties.ROBUSTNESS_REPLICATES
    for name in laws.LAWS:
        kept = kept_rows(rows, f"{name}_controls")
        for label, group in kept.groupby("robustness_label", sort=False):
            truth = float(group["truth"].iloc[0])
            for cell, column, role in (
                ("mechanism_correct", "estimate", "positive"),
                ("plug_in_control", "initial_estimate", "control"),
                ("factual_tilt_control", "factual_estimate", "control"),
            ):
                error = group[column].to_numpy(dtype=float) - truth
                bias = float(np.mean(error) / np.std(error, ddof=1))
                if role == "positive":
                    probability = float(
                        sizing.bias_equivalent_probability(study.MARGINS, bias, replicates)
                    )
                else:
                    probability = sizing.bias_discriminated_probability(
                        study.MARGINS, bias, replicates
                    )
                se = group["factual_std_error" if cell == "factual_tilt_control" else "std_error"]
                records.append(
                    {
                        "cell": f"{label}__{cell}",
                        "role": role,
                        "draws": len(group),
                        "standardized_bias": bias,
                        "se_ratio": float(se.mean() / np.std(error, ddof=1)),
                        "pass_probability": probability,
                        "informative": role == "positive" or probability >= MINIMUM_POWER,
                    }
                )
    return pd.DataFrame(records)


def predicted_exceedance(failed: int, pilot_draws: int, replicates: int, allowance: int) -> float:
    """The posterior predictive probability that more than ``allowance`` of ``replicates`` fail.

    Jeffreys prior on the failure rate, so the count of the declared run is beta-binomial.
    """
    law = betabinom(replicates, failed + 0.5, pilot_draws - failed + 0.5)
    return float(law.sf(allowance))


def failure_refusals(rows: pd.DataFrame, replicates: int) -> list[str]:
    """One refusal per cell whose predicted failure bound exceeds :data:`FAILURE_EXCEEDANCE`."""
    record = study.STUDY
    out = []
    for cell in study.CELLS:
        held = draws(rows, cell)
        if not held:
            out.append(f"the pilot holds no draws of {cell}; run its pilot")
            continue
        allowance = int(np.floor(study.FAILURE_BOUND * replicates))
        probability = predicted_exceedance(failed_draws(rows, cell), held, replicates, allowance)
        if probability > FAILURE_EXCEEDANCE:
            out.append(
                f"{cell} lost {failed_draws(rows, cell)} of {held:,} pilot fits, and the declared "
                f"run of {replicates:,} loses more than the {allowance} its failure bound allows "
                f"with predictive probability {probability:.3f}, above {FAILURE_EXCEEDANCE:.2f}; "
                "diagnose and fix the refusal and rerun the cell's pilot"
            )
    del record
    return out


def failure_exceedance(rows: pd.DataFrame, cell: str, replicates: int) -> float:
    """:func:`predicted_exceedance` of ``cell`` at ``replicates`` and the record's bound."""
    allowance = int(np.floor(study.FAILURE_BOUND * replicates))
    return predicted_exceedance(failed_draws(rows, cell), draws(rows, cell), replicates, allowance)


def accounting_allowance(cell: str) -> int:
    """The failure accounting's own allowance for ``cell`` at the record's count."""
    record = study.STUDY
    return int(allowed(record, record.implementation, cell))


def seconds(rows: pd.DataFrame) -> dict[str, float]:
    """The median seconds per fit of each primary cell and of each law's control fits."""
    out: dict[str, float] = {}
    for cell in study.CELLS:
        # The declared pilot's draws: the extension ran beside other work.
        fits = first_draws(kept_rows(rows, cell)).drop_duplicates("replicate")
        out[cell] = float(fits["seconds"].median())
    for name in laws.LAWS:
        fits = kept_rows(rows, f"{name}_controls").drop_duplicates("replicate")
        if not fits.empty:
            out[f"{name}_wrong_hazard"] = float(fits["seconds"].median())
            out[f"{name}_factual_tilt"] = float(fits["factual_seconds"].median())
    return out


def coverage_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Each target's pilot coverage, SE ratio and standardized bias, first quartile included."""
    records = []
    for cell in study.CELLS:
        kept = kept_rows(rows, cell)
        for estimand, group in kept.groupby("estimand", sort=False):
            error = group["estimate"] - group["truth"]
            spread = float(error.std(ddof=1))
            initial = group["initial_estimate"] - group["truth"]
            records.append(
                {
                    "cell": cell,
                    "estimand": estimand,
                    "early": bool(group["early"].iloc[0]),
                    "fits": len(group),
                    "coverage": float(group["covered"].mean()),
                    "se_ratio": float(group["std_error"].mean() / spread),
                    "ic_se_ratio": float(group["ic_std_error"].mean() / spread),
                    "standardized_bias": float(error.mean() / spread),
                    "plug_in_standardized_bias": float(initial.mean() / initial.std(ddof=1)),
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--kinds", nargs="*", default=list(KINDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--draws", type=int, default=None)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.kinds, resume=arguments.resume, draws=arguments.draws)
