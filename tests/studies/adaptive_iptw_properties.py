r"""Property families of ``adaptive-iptw`` (X21, ``X21-plan.md`` 3.5 and 3.6).

Every fit is one of :data:`~tests.studies.canonical_adaptive_iptw.FITS`.  The tables
:data:`~tests.studies.canonical_adaptive_iptw.SCENES` and
:data:`~tests.studies.canonical_adaptive_iptw.READERS` are the one list of what a stream draws,
what it fits and what each cell reads.  The module adds the row writers, the verdict functions
and the declared rules.

==============================  ===============================================================
family                          cells and rule
==============================  ===============================================================
``interval_calibration``        the known-mechanism fits of laws PD, WT, CL and ST: the shared
                                calibration rule on the 99% interval of the SE ratio and of the
                                coverage.  A ratio key (``rr``, ``or``) is read on the log scale
``simultaneous_coverage``       the simultaneous band of the known-mechanism arm fit of law PD
``bias``                        the estimated-mechanism cells: the 99% interval of the
                                standardized bias inside the margin (positive), or beyond it
                                (control).  An estimated mechanism reports no interval, so no
                                coverage or SE ratio is read
``root_n_points``               the bias of the ``rate`` fit of law P at three sizes, and
``root_n_rate``                 the fitted slope of its empirical SD on the size
``targeting_necessity``         the targeted estimate beside the untargeted Horvitz-Thompson mean
                                of the same fit, paired by replicate
``spread``                      the plug-in SE over the empirical SD of the parametric fit of
                                law VL1 (reported, no verdict)
==============================  ===============================================================

**Rows.**  An estimate row keeps the shared ``REPLICATE_COLUMNS`` schema.  On a non-inferential
status ``std_error`` and ``covered`` hold the plug-in diagnostic
(:func:`~tests.studies.evidence.schema.reported_inference`), and no verdict outside ``spread``
reads them.  A ratio key (``rr``, ``or``) writes the estimate and the truth on the log scale, the
scale of its standard error.  An untargeted row writes the Horvitz-Thompson mean of the initial
mechanism and the targeted estimate's standard error as a placeholder
(:func:`~tests.studies.evidence.properties.control_row`).

**Streams.**  Every stream is a scene.  Its replicate ``r`` is
``stream_seed(STUDY, "property_sample", scene, r)`` and its folds are
``stream_seed(STUDY, "property_fit", scene, r)``.  A scene's fits read one draw.  A known-
mechanism scene of law WT, CL or ST also fits the estimated mechanism on its first
``R_BIAS`` draws (``extra_fits``).

**Failures.**  A replicate that raises is not redrawn.  It costs each cell of its scene one
replicate, and the accounting judges each cell at :data:`FAILURE_BOUND`
(``docs/development/method-benchmarking.md``).

**Convergence.**  Every row also records the fit it reads: ``rounds``, the largest number of
tilt rounds over the fit's keys, and ``converged``, 1 when every key met ``tol``.  A fit that
stops at ``max_iter`` is not dropped.  It keeps its rows, and :func:`convergence_reading`
reports ``nonconverged_fits`` and ``max_rounds`` per cell.  No verdict reads either column, so
a cell can pass with a nonconverged fit, and the summary table shows it.  Rows written before
these columns existed give a summary without them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_adaptive_iptw as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    Rate,
    control_row,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    EXCLUDED_SLOPE,
    ROOT_N_SLOPE,
    ROOT_N_SLOPE_MARGIN,
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
    fitted_rate_row,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
SCENES = study.SCENES
READERS = study.READERS
PROPERTY_CELLS = study.PROPERTY_CELLS
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The necessity family: the targeted estimate beside the untargeted mean of the same fit.
NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=tuple(study.NECESSITY_LABELS),
    arms=("targeted", "untargeted"),
    column="untargeted_displacement",
    threshold=study.NECESSITY_THRESHOLD,
)

#: The label of the one joint cell, and the keys its band covers.
BAND_LABEL = "known_band"
BAND_KEYS = study.PD_ARM_KEYS
LADDER_FAMILY = "root_n_points"
RATE_FAMILY = "root_n_rate"
RATE_CELL = "empirical_sd"
SPREAD_FAMILY = "spread"
BIAS_FAMILIES = ("bias", LADDER_FAMILY)

_Key = tuple[str, int, int]

#: The per-row convergence columns, and the reported summary columns they give.
CONVERGENCE_COLUMNS = ("rounds", "converged")
CONVERGENCE_SUMMARY = ("nonconverged_fits", "max_rounds")


# ------------------------------------------------------------------------------------ streams


def _sample_seed(scene: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", scene, replicate)


def _fit_seed(scene: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_fit", scene, replicate)


def counts(scene: study.Scene, budget: int | None) -> tuple[int, int]:
    """The replicates the scene's cells and its extra fits request.

    Parameters
    ----------
    scene : Scene
        The stream.
    budget : int or None
        A pre-run cap on every cell; ``None`` for the declared counts.

    Returns
    -------
    tuple of int
        The replicates of :attr:`Scene.fits` and of :attr:`Scene.extra_fits`.
    """
    if budget is None:
        return scene.replicates, scene.extra_replicates
    return budget, min(budget, scene.extra_replicates)


def _active(scene: study.Scene, replicate: int, budget: int | None) -> list[tuple[str, int]]:
    """The ``(fit, requested)`` pairs the scene runs on draw ``replicate``."""
    main, extra = counts(scene, budget)
    out = [(fit, main) for fit in scene.fits]
    if replicate < extra:
        out += [(fit, extra) for fit in scene.extra_fits]
    return out


_BY_FIT: dict[tuple[str, str], list[study.Reader]] = {}
for _reader in READERS:
    _BY_FIT.setdefault((_reader.scene, _reader.fit), []).append(_reader)


def _name(reader: study.Reader) -> str:
    return f"{reader.family}/{reader.cell}"


# -------------------------------------------------------------------------------------- rows


def _estimate_row(
    reader: study.Reader,
    result: Any,
    truth: dict[str, float],
    *,
    replicate: int,
    n: int,
    requested: int,
) -> dict[str, Any]:
    estimate = result[reader.key]
    target = float(truth[reader.key])
    row = replicate_row(
        property_name=reader.family,
        cell=reader.cell,
        role=reader.role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=target,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    if reader.key in study.RATIO_KEYS:
        row["estimate"] = float(estimate.inference_value)
        row["truth"] = math.log(target)
    return row


def _untargeted_row(
    reader: study.Reader,
    result: Any,
    truth: dict[str, float],
    *,
    replicate: int,
    n: int,
    requested: int,
) -> dict[str, Any]:
    standard_error, _, _ = reported_inference(result[reader.key])
    return control_row(
        property_name=reader.family,
        cell=reader.cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=float(truth[reader.key]),
        estimate=study.untargeted_value(result, reader.key),
        standard_error=standard_error,
        critical=CRITICAL,
        role=reader.role,
    )


def _band_row(
    reader: study.Reader,
    result: Any,
    truth: dict[str, float],
    *,
    replicate: int,
    n: int,
    requested: int,
) -> dict[str, Any]:
    rows = joint_coverage_rows(
        result,
        truth,
        BAND_KEYS,
        label=BAND_LABEL,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )
    return rows[0]


_WRITERS = {
    "estimate": _estimate_row,
    "spread": _estimate_row,
    "untargeted": _untargeted_row,
    "band": _band_row,
}


def convergence(result: Any) -> dict[str, int]:
    """The :data:`CONVERGENCE_COLUMNS` of one fit, over every fluctuation it ran.

    Parameters
    ----------
    result : CausalResult
        One fit of :data:`~tests.studies.canonical_adaptive_iptw.FITS`.

    Returns
    -------
    dict
        ``rounds``, the largest ``n_iter``, and ``converged``, 1 when every fluctuation
        converged.
    """
    fluctuations = [
        fluctuation for repeat in result.repeats for fluctuation in repeat.fluctuations.values()
    ]
    return {
        "rounds": max((int(item.n_iter) for item in fluctuations), default=0),
        "converged": int(all(bool(item.converged) for item in fluctuations)),
    }


def fit_drawn(key: _Key, frame: pd.DataFrame, *, budget: int | None = None) -> list[dict[str, Any]]:
    """The rows of one scene draw: every fit the scene runs on it, and every cell they fill.

    Parameters
    ----------
    key : tuple
        ``(scene, replicate, budget)``; a budget of 0 is the declared counts.
    frame : pandas.DataFrame
        The draw of the scene's law.
    budget : int or None
        Overrides the key's budget when given.

    Returns
    -------
    list of dict
        One row per cell the draw fills.
    """
    name, replicate, keyed = key
    scene = SCENES[name]
    cap = (keyed or None) if budget is None else budget
    truth = study.truths(scene.law)
    seed = _fit_seed(name, replicate)
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fit, requested in _active(scene, replicate, cap):
            result = study.FITS[fit](frame, scene.n, seed)
            diagnostics = convergence(result)
            for reader in _BY_FIT[(name, fit)]:
                row = _WRITERS[reader.kind](
                    reader,
                    result,
                    truth,
                    replicate=replicate,
                    n=scene.n,
                    requested=requested,
                )
                rows.append({**row, **diagnostics})
    return rows


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_drawn(key, frame)


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    name, replicate, _ = key
    scene = SCENES[name]
    return study.draw_frame(scene.law, scene.n, _sample_seed(name, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows fill."""
    name, replicate, budget = key
    scene = SCENES[name]
    return [
        _name(reader)
        for fit, _ in _active(scene, replicate, budget or None)
        for reader in _BY_FIT[(name, fit)]
    ]


def cell_of(key: _Key) -> str:
    """The fit set whose leading replicates the systematic-failure prefix reads."""
    return f"scene/{key[0]}"


def replicate_of(key: _Key) -> int:
    return key[1]


def payloads(budget: int | None, start: int = 0) -> list[_Key]:
    """Every key of the study, in scene order; ``budget`` caps each cell for a pre-run check."""
    out: list[_Key] = []
    for scene in SCENES.values():
        main, _ = counts(scene, budget)
        out.extend((scene.name, replicate, budget or 0) for replicate in range(start, start + main))
    return out


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pre-run check."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.reindex(columns=list(REPLICATE_COLUMNS)).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def smoke_keys(draws: int, start: int = 0) -> list[_Key]:
    """Keys of a smoke on replicate ids ``start`` to ``start + draws - 1`` of every scene."""
    return payloads(draws, start)


def fit_keys(keys: Sequence[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [fit_drawn(key, _draw(key)) for key in keys]


# ------------------------------------------------------------------------------ declaration


class DeclaredLaw:
    """The law a declared cell reads: one scene's law, with its exact truths.

    A ratio key carries a ``log_`` twin, the value its rows record.

    Parameters
    ----------
    scene : str
        The scene the cell draws from.
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.name = f"adaptive_iptw__{scene}"

    def truth(self) -> dict[str, float]:
        """The exact value of every key of the scene's law, and the log of each ratio."""
        values = dict(study.truths(SCENES[self.scene].law))
        for key in study.RATIO_KEYS & set(values):
            values[f"log_{key}"] = math.log(values[key])
        return values


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, estimand, size and replicate count."""
    cells: list[PropertyCell] = []
    for reader in READERS:
        scene = SCENES[reader.scene]
        seed = stream_seed(STUDY, "property_sample", scene.name, 0)
        replicates = scene.extra_replicates if reader.fit in scene.extra_fits else scene.replicates
        if reader.kind == "band":
            cells += joint_property_cells(
                BAND_LABEL, n=scene.n, replicates=replicates, seed=seed, control=False
            )
            continue
        estimand = f"log_{reader.key}" if reader.key in study.RATIO_KEYS else reader.key
        cells.append(
            PropertyCell(
                property=reader.family,
                cell=reader.cell,
                dgp=DeclaredLaw(scene.name),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=scene.n,
                replicates=replicates,
                seed=seed,
                role=reader.role,
                estimand=estimand,
            )
        )
    return tuple(cells)


# ------------------------------------------------------------------------------------ verdicts


def bias_verdicts(summary: pd.DataFrame) -> None:
    """A positive row passes on its equivalence endpoint and a control on its discrimination one."""
    for family in BIAS_FAMILIES:
        member = summary["property"] == family
        positive = member & (summary["role"] == "positive")
        control = member & (summary["role"] == "control")
        summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
        summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]
        unknown = member & ~summary["role"].isin(("positive", "control"))
        if unknown.any():
            raise ValueError(f"{family} cells need a positive or control role")


def spread_reading(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The spread family reports its SE ratio with the 99% interval and states no verdict."""
    margins = STUDY.margins
    for index in summary.index[summary["property"] == SPREAD_FAMILY]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == SPREAD_FAMILY) & (rows["cell"] == cell)]
        interval = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, SPREAD_FAMILY, cell),
            bound=None,
            truth_varies=STUDY.truth_varies_by_replicate,
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
        summary.loc[index, "passed"] = True
        summary.loc[index, "property_passed"] = True


def convergence_reading(summary: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Report the nonconverged fits and the largest round count of each cell.

    The columns are reported and decide no verdict.  Rows without :data:`CONVERGENCE_COLUMNS`,
    such as those of run 1, leave the summary unchanged.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary.
    rows : pandas.DataFrame
        The replication rows.

    Returns
    -------
    pandas.DataFrame
        ``summary`` with :data:`CONVERGENCE_SUMMARY` filled where the rows carry them.
    """
    if not set(CONVERGENCE_COLUMNS) <= set(rows.columns):
        return summary
    out = summary.copy()
    for column in CONVERGENCE_SUMMARY:
        out[column] = math.nan
    for index in out.index:
        group = rows.loc[
            (rows["property"] == out.loc[index, "property"])
            & (rows["cell"] == out.loc[index, "cell"])
        ]
        kept = group.loc[group["converged"].notna()]
        if kept.empty:
            continue
        out.loc[index, "nonconverged_fits"] = float((kept["converged"] == 0).sum())
        out.loc[index, "max_rounds"] = float(kept["rounds"].max())
    return out


def ladder_rate(rows: pd.DataFrame, columns: Any) -> dict[str, Any]:
    """The fitted slope of the empirical SD of the ``rate`` fit on the size, as a summary row."""
    return fitted_rate_row(
        rows.loc[rows["property"] == LADDER_FAMILY],
        STUDY,
        columns,
        ladder_property=LADDER_FAMILY,
        property_name=RATE_FAMILY,
        cell=RATE_CELL,
        role="positive",
        statistic="spread",
        seed_labels=(RATE_FAMILY, RATE_CELL),
        verdict=_slope_passes,
    )


def _slope_passes(fitted: Rate) -> bool:
    return bool(
        fitted.equivalent_to(ROOT_N_SLOPE, ROOT_N_SLOPE_MARGIN) and fitted.excludes(EXCLUDED_SLOPE)
    )


SUMMARY_COLUMNS = ("untargeted_displacement",)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    necessity_verdicts(summary, rows, NECESSITY)
    bias_verdicts(summary)
    spread_reading(summary, rows)
    rates.append(ladder_rate(rows, summary.columns))
    return convergence_reading(finish(summary, rates), rows)


# --------------------------------------------------------------------------------- the rules


def _ladder(row: pd.Series) -> tuple[str, str, str, tuple[str, ...]]:
    """The ladder, cell prefix, statistic and seed labels of the ``root_n_rate`` row."""
    return LADDER_FAMILY, "", "spread", (RATE_FAMILY, RATE_CELL)


def _root_n_slope(ctx: pf.Context, row: pd.Series, interval: Any) -> pf.Judged:
    fitted = Rate(float(row["slope"]), interval)
    low, high = ROOT_N_SLOPE - ROOT_N_SLOPE_MARGIN, ROOT_N_SLOPE + ROOT_N_SLOPE_MARGIN
    return pf.Judged(min(interval.low - low, high - interval.high), _slope_passes(fitted))


def _bias_rule() -> pf.FamilyRule:
    return pf.FamilyRule(
        families=frozenset(BIAS_FAMILIES),
        shared="bias_verdicts",
        clauses=(
            pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
            pf.bias_clause("bias discriminated", kind="beyond", applies=pf.roles("control")),
        ),
    )


def _rate_rule() -> pf.FamilyRule:
    return pf.FamilyRule(
        families=frozenset({RATE_FAMILY}),
        shared="summarize_properties",
        clauses=(
            pf.rate_slope_clause(
                "slope in root-n band",
                judge=_root_n_slope,
                ladder=_ladder,
                applies=lambda ctx, row: True,
            ),
        ),
    )


#: The failure accounting's declared rules.  The calibration and simultaneous families read the
#: shared rules (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (
    _bias_rule(),
    _rate_rule(),
    *property_verdicts.necessity_rules(NECESSITY),
)

__all__ = [
    "FAILURE_BOUND",
    "NECESSITY",
    "PROPERTY_CELLS",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "convergence",
    "convergence_reading",
    "declared_cells",
    "fit_keys",
    "generate_property_rows",
    "smoke_keys",
    "summarize_properties",
]
