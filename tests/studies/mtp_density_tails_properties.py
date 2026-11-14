"""The property families of the ``mtp-density-tails`` study.

Three families carry verdicts, each declared before any run.

``interval_calibration``
    One arm per route and law, each read on the shared calibration rule: the SE ratio's 99%
    bootstrap interval inside [0.93, 1.07] and the coverage interval inside [0.92, 0.98].

    - The density route with the oracle hazard and the correct outcome regression at n = 2,000,
      3,200 draws per law: gated on G, T and C (``<law>__correctly_specified``), reported on S,
      B and U (``<law>__reported``).
    - The density route's double robustness at n = 8,000, 3,200 draws on G and T: the oracle
      density and the wrong outcome regression (OLS on ``[1, A, W1, W2]``), gated
      (``<law>_dr__wrong_outcome``).  Its rule is the calibration rule, so the arms are filed
      here rather than under ``double_robustness``, whose shared rule reads a bias margin.  Two
      coverage controls on their own seeds, 1,000 draws each, must read coverage below the band:
      ``both_wrong_control`` (the density that ignores ``W1``, on G and T) and
      ``pre_fix_control`` (the pre-fix one-step reading with the wrong regression, on G).
    - The classifier route that ``ratio=None`` resolves to (the default stack, five outer
      folds, the policy steps), with the correct regression at n = 2,000, 3,200 draws, gated on
      G, T and S (``<law>_classifier__correctly_specified``).  Its DR cells (the wrong
      regression, 400 draws on G and T, and the both-wrong contrast on G, the classifier without
      ``W1``) are reported, each row with the remainder proxy
      ``sqrt(n) E_n[(r_n - r0)(Q0 - Q_ols)] / SD(D0)``.
    - The longitudinal default route on ``longitudinal-mtp``'s continuous law at n = 2,000, five
      folds, 3,200 draws on ``LTMLE`` and ``SDR``: ``ate_regimen[up vs natural]`` gated, the six
      other estimands reported.

    The shrunken-SE and noise controls on G (density), G (classifier) and ``LTMLE`` are derived
    from the positive arm's own rows (:func:`~tests.studies.evidence.property_verdicts.calibration_controls`),
    so they cost no fit and read every positive draw.

``ic_gap``
    ``SD(IC - D0) / SD(D0)`` of the oracle-hazard density fit, against the exact efficient
    influence function, at n = 2,000 (80 draws) and n = 20,000 (30 draws).  On G, T, C, B and U
    the 2,000 cell's median is at most :data:`GAP_BOUND_SMALL`, each 20,000 gap is at most
    :data:`GAP_BOUND_LARGE`, and the 20,000 median is at most :data:`GAP_FALL` times the 2,000
    median (the fall clause, on the 20,000 row).  S and ``Gscale`` (law G under
    ``Scale(1.25)``) are reported.  Each control fails its named clause: the pre-fix reading
    (the 2,000 bound on every law; also the fall clause on G and the 20,000 clause on C, B and
    U), the declared support ignored on B and U (the 20,000 clause), and no row-wise break on B
    and U (the 2,000 bound).

``cluster_warning``
    ``isolated_dose_clusters(A, min_rows=3)`` on the doses of the ``ic_gap`` draws, with no fit:
    at least 26 of the 30 C draws at n = 20,000 warn, and at most 6 of the 220 G and T draws (80
    and 30 at each size) do.  The gap-disabled helper must fail the C clause and rev 2's eager
    helper the G and T clause.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests import mtp_density_tails_law as law
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_mtp_density_tails as study
from tests.studies import longitudinal_mtp_common as longitudinal
from tests.studies import mtp_density_tails_fits as fits
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 3_200
ROBUST_N = 8_000
ROBUST_REPLICATES = 3_200
CONTROL_REPLICATES = 1_000
CLASSIFIER_DR_REPLICATES = 400
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
GAP_SIZES = study.GAP_SIZES
GAP_DRAWS = {2_000: 80, 20_000: 30}
#: The 2,000 median bound and the 20,000 per-draw bound of each gated law.
GAP_BOUND_SMALL = {"G": 0.10, "C": 0.10, "T": 0.05, "B": 0.04, "U": 0.04}
GAP_BOUND_LARGE = {"G": 0.10, "C": 0.10, "T": 0.04, "B": 0.03, "U": 0.03}
GAP_FALL = 0.6
#: The cluster clauses: at least this many of the C draws warn, at most this many G and T do.
CLUSTER_C_FLOOR = 26
CLUSTER_GT_CEILING = 6
CLUSTER_SOURCES: tuple[tuple[str, int], ...] = (
    ("G", 2_000),
    ("T", 2_000),
    ("G", 20_000),
    ("T", 20_000),
)
CLUSTER_GT_DRAWS = sum(GAP_DRAWS[n] for _, n in CLUSTER_SOURCES)

#: The named clause each ``ic_gap`` control must fail, by control cell; a control cell not
#: listed is reported.
GAP_CONTROLS: dict[str, str] = {
    **{f"{name}__pre_fix_n_2000": "small" for name in study.GAP_GATED},
    "G__pre_fix_n_20000": "fall",
    **{f"{name}__pre_fix_n_20000": "large" for name in ("C", "B", "U")},
    **{f"{name}__support_ignored_n_20000": "large" for name in law.SUPPORTED},
    **{f"{name}__no_row_break_n_2000": "small" for name in law.SUPPORTED},
}

LONGITUDINAL_LABELS = study.LONGITUDINAL_LABELS
LTMLE_GATED = study.LONGITUDINAL_GATED_LABELS["LTMLE"]
#: The labels whose calibration controls are derived from their positive arm.
CONTROLLED_LABELS = ("G", "G_classifier", LTMLE_GATED)
POSITIVE_SUFFIXES = ("correctly_specified", "wrong_outcome")
#: The DR coverage controls: the cell kind and the clause it publishes.
COVERAGE_CONTROLS = {
    "both_wrong_control": "both-wrong coverage below band",
    "pre_fix_control": "pre-fix coverage below band",
}
EXTRA_COLUMNS = (
    "remainder_proxy",
    "median_gap",
    "max_gap",
    "gap_fall",
    "warned_draws",
)

# ------------------------------------------------------------------ fit sets

#: ``(family, stream, configuration, n, replicates, label)``.  A stream is one fit set: its
#: draws come from ``stream_seed(STUDY, "property_sample", family, stream, replicate)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "interval_calibration",
            f"density_{name}",
            "density",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            name,
        )
        for name in (*study.DENSITY_GATED, *study.DENSITY_REPORTED)
    ),
    *(
        ("interval_calibration", f"dr_{name}", "wrong_outcome", ROBUST_N, ROBUST_REPLICATES, name)
        for name in study.ROBUST_GATED
    ),
    *(
        (
            "interval_calibration",
            f"dr_{name}_both_wrong",
            "both_wrong",
            ROBUST_N,
            CONTROL_REPLICATES,
            name,
        )
        for name in study.ROBUST_GATED
    ),
    ("interval_calibration", "dr_G_pre_fix", "pre_fix", ROBUST_N, CONTROL_REPLICATES, "G"),
    *(
        (
            "interval_calibration",
            f"classifier_{name}",
            "classifier",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            name,
        )
        for name in study.CLASSIFIER_GATED
    ),
    *(
        (
            "interval_calibration",
            f"classifier_dr_{name}",
            "classifier_wrong_outcome",
            CALIBRATION_N,
            CLASSIFIER_DR_REPLICATES,
            name,
        )
        for name in study.ROBUST_GATED
    ),
    (
        "interval_calibration",
        "classifier_dr_G_both_wrong",
        "classifier_both_wrong",
        CALIBRATION_N,
        CLASSIFIER_DR_REPLICATES,
        "G",
    ),
    *(
        (
            "interval_calibration",
            f"longitudinal_{estimator}",
            "longitudinal",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            estimator,
        )
        for estimator in study.LONGITUDINAL
    ),
    *(
        ("ic_gap", f"{name}_n_{n}", "shipped", n, GAP_DRAWS[n], name)
        for name in (*study.GAP_GATED, *study.GAP_REPORTED)
        for n in GAP_SIZES
    ),
    *(
        ("ic_gap", f"{name}_pre_fix_n_{n}", "pre_fix", n, GAP_DRAWS[n], name)
        for name in study.GAP_GATED
        for n in GAP_SIZES
    ),
    *(
        ("ic_gap", f"{name}_support_ignored_n_{n}", "support_ignored", n, GAP_DRAWS[n], name)
        for name in law.SUPPORTED
        for n in GAP_SIZES
    ),
    *(
        ("ic_gap", f"{name}_no_row_break_n_2000", "no_row_break", 2_000, GAP_DRAWS[2_000], name)
        for name in law.SUPPORTED
    ),
    ("cluster_warning", "C_n_20000", "cluster", 20_000, GAP_DRAWS[20_000], "C"),
    # The pooled cell reads draws at both sizes; each row records its own n.
    ("cluster_warning", "GT_pooled", "cluster", 2_000, CLUSTER_GT_DRAWS, "GT"),
)

_Payload = tuple[str, str, str, int, int, int, str]
_Key = tuple[_Payload]


def _seed(family: str, stream: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, stream, replicate)


def cluster_source(replicate: int) -> tuple[str, int, int]:
    """The ``ic_gap`` draw a ``GT_pooled`` replicate reads: ``(law, n, ic_gap replicate)``."""
    offset = replicate
    for name, n in CLUSTER_SOURCES:
        if offset < GAP_DRAWS[n]:
            return name, n, offset
        offset -= GAP_DRAWS[n]
    raise IndexError(f"GT_pooled has {CLUSTER_GT_DRAWS} replicates; got {replicate}")


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, stream, configuration, replicate, n, _, label = key[0]
    if family == "cluster_warning":
        if label == "GT":
            name, size, source = cluster_source(replicate)
        else:
            name, size, source = label, n, replicate
        return law.draw(name, size, _seed("ic_gap", f"{name}_n_{size}", source))
    if configuration == "longitudinal":
        return fits.draw_longitudinal(n, _seed(family, stream, replicate))
    return law.draw(fits.law_of(label), n, _seed(family, stream, replicate))


def _calibration_label(stream: str, configuration: str, label: str) -> str:
    if configuration == "density":
        return label
    if configuration in {"wrong_outcome", "both_wrong", "pre_fix"}:
        return f"{label}_dr"
    if configuration == "classifier":
        return f"{label}_classifier"
    if configuration == "classifier_wrong_outcome":
        return f"{label}_classifier_dr"
    return f"{label}_classifier_dr_both_wrong"


def _calibration_suffix(configuration: str, label: str) -> str:
    if configuration == "density":
        return "correctly_specified" if label in study.DENSITY_GATED else "reported"
    if configuration == "wrong_outcome":
        return "wrong_outcome"
    if configuration == "both_wrong":
        return "both_wrong_control"
    if configuration == "pre_fix":
        return "pre_fix_control"
    if configuration == "classifier":
        return "correctly_specified"
    return "reported"


def _gap_cell(stream: str) -> str:
    name, rest = stream.split("_", 1)
    return f"{name}__{rest}"


def _base_cells(key: _Key) -> list[str]:
    """The cells a key's rows fill, without the derived controls."""
    family, stream, configuration, _, _, _, label = key[0]
    if family == "ic_gap":
        return [f"ic_gap/{_gap_cell(stream)}"]
    if family == "cluster_warning":
        if label == "GT":
            return ["cluster_warning/GT__pooled", "cluster_warning/GT__eager_control"]
        return ["cluster_warning/C__n_20000", "cluster_warning/C__gap_disabled_control"]
    if configuration == "longitudinal":
        return [
            f"interval_calibration/{cell_label}__"
            + ("correctly_specified" if estimand == fits.LONGITUDINAL_GATED else "reported")
            for estimand, cell_label in LONGITUDINAL_LABELS[label].items()
        ]
    prefix = _calibration_label(stream, configuration, label)
    return [f"interval_calibration/{prefix}__{_calibration_suffix(configuration, label)}"]


#: The calibration positive arm of each controlled label to the controls derived from it.
DERIVED_CELLS = calibration_derived(CONTROLLED_LABELS)


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    names: list[str] = []
    for name in _base_cells(key):
        names += [name, *DERIVED_CELLS.get(name, ())]
    return names


def cell_of(key: _Key) -> str:
    return f"{key[0][0]}/{key[0][1]}"


def replicate_of(key: _Key) -> int:
    return key[0][3]


# ------------------------------------------------------------------ rows


def _row(
    cell_name: str,
    role: str,
    payload: _Payload,
    *,
    n: int,
    truth: float,
    estimate: float,
    std_error: float,
    covered: bool,
    **extra: Any,
) -> dict[str, Any]:
    family, cell = cell_name.split("/", 1)
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": payload[3],
        "n": n,
        "requested_replicates": payload[5],
        "failed_replicates": 0,
        "truth": float(truth),
        "estimate": float(estimate),
        "std_error": float(std_error),
        "covered": int(covered),
        "rejected": int(not covered),
        **extra,
    }


def _role(cell_name: str) -> str:
    cell = cell_name.split("/", 1)[1]
    family = cell_name.split("/", 1)[0]
    if family == "ic_gap":
        if cell in GAP_CONTROLS:
            return "control"
        name, rest = cell.split("__", 1)
        if rest.startswith("n_") and name in study.GAP_GATED:
            return "positive"
        return DIAGNOSTIC_ROLE
    if family == "cluster_warning":
        return "control" if cell.endswith("_control") else "positive"
    kind = cell.rsplit("__", 1)[1]
    if kind == "reported":
        return DIAGNOSTIC_ROLE
    return "positive" if kind in POSITIVE_SUFFIXES else "control"


def _covered(fitted: fits.PointFit, truth: float) -> bool:
    low, high = fitted.ci
    return bool(low <= truth <= high)


def fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of one key: one fit (or none, for the cluster cells) on its drawn sample."""
    family, _, configuration, _, n, _, label = payload
    base = _base_cells((payload,))
    if family == "cluster_warning":
        doses = frame["A"].to_numpy(dtype=float)
        if label == "GT":
            flags = (fits.warns(doses), fits.warns_eager(doses))
        else:
            flags = (fits.warns(doses), fits.warns_gap_disabled(doses))
        return [
            _row(
                name,
                _role(name),
                payload,
                n=len(frame),
                truth=0.0,
                estimate=float(flag),
                std_error=1.0,
                covered=not flag,
            )
            for name, flag in zip(base, flags, strict=True)
        ]
    if family == "ic_gap":
        fitted = fits.fit_density(label, frame, reading_kind=configuration)
        truth = fits.truth(label)
        (name,) = base
        return [
            _row(
                name,
                _role(name),
                payload,
                n=n,
                truth=0.0,
                estimate=fits.ic_gap(label, frame, fitted),
                std_error=fitted.std_error,
                covered=_covered(fitted, truth),
            )
        ]
    if configuration == "longitudinal":
        result = fits.fit_longitudinal(label, frame)
        rows = []
        for (estimand, _), name in zip(LONGITUDINAL_LABELS[label].items(), base, strict=True):
            estimate = result[estimand]
            low, high = estimate.ci
            truth = fits.LONGITUDINAL_TRUTH[estimand]
            rows.append(
                _row(
                    name,
                    _role(name),
                    payload,
                    n=n,
                    truth=truth,
                    estimate=float(estimate.psi),
                    std_error=float(estimate.std_error),
                    covered=bool(low <= truth <= high),
                )
            )
        return rows
    (name,) = base
    truth = fits.truth(label)
    if configuration == "pre_fix":
        estimate, std_error = fits.pre_fix_one_step(label, frame)
        half = CRITICAL * std_error
        return [
            _row(
                name,
                _role(name),
                payload,
                n=n,
                truth=truth,
                estimate=estimate,
                std_error=std_error,
                covered=bool(estimate - half <= truth <= estimate + half),
            )
        ]
    extra: dict[str, Any] = {}
    if configuration == "density":
        fitted = fits.fit_density(label, frame)
    elif configuration == "wrong_outcome":
        fitted = fits.fit_density(label, frame, outcome="wrong")
    elif configuration == "both_wrong":
        fitted = fits.fit_density(label, frame, outcome="wrong", hazard="without_w1")
    elif configuration == "classifier":
        fitted = fits.fit_classifier(label, frame)
    else:
        fitted = fits.fit_classifier(
            label,
            frame,
            outcome="wrong",
            without_w1=configuration == "classifier_both_wrong",
        )
        extra["remainder_proxy"] = fits.remainder_proxy(label, frame, fitted.ratio)
    return [
        _row(
            name,
            _role(name),
            payload,
            n=n,
            truth=truth,
            estimate=fitted.estimate,
            std_error=fitted.std_error,
            covered=_covered(fitted, truth),
            **extra,
        )
    ]


def fit_set_rows(payload: _Payload) -> list[dict[str, Any]]:
    """The replicate rows of one key, drawn and fitted in one call."""
    return fit_drawn(payload, _draw((payload,)))


def _payloads(
    fit_sets: Sequence[tuple[str, str, str, int, int, str]], budget: int | None
) -> list[_Key]:
    out: list[_Key] = []
    for family, stream, configuration, n, replicates, label in fit_sets:
        requested = replicates if budget is None else min(budget, replicates)
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_drawn(key[0], frame)


@cache
def efficiency_bounds() -> dict[str, float]:
    """``SD(D0)`` of every calibration label a derived control reads."""
    return {
        "G": float(law.EFFICIENCY_SD["G"]),
        "G_classifier": float(law.EFFICIENCY_SD["G"]),
        LTMLE_GATED: float(longitudinal.continuous_contrast_sd("up")),
    }


#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    parts: Path | None = None,
    streams: Sequence[str] | None = None,
    replicates: range | None = None,
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check.

    A finished key is kept under :func:`~tests.studies.mtp_study_tools.property_checkpoint`
    (the driver's checkpoint scratch, else ``parts``), so a stopped run resumes.  ``streams``
    and ``replicates`` select one cloud part: those fit sets' keys in that replicate range,
    with the full requested count, so the part's checkpoints are the declared run's own and a
    declared run whose scratch holds them reuses them.  A ``budget`` run keeps nothing.
    """
    keys = [
        key
        for key in _payloads(FIT_SETS, budget)
        if (streams is None or key[0][1] in streams)
        and (replicates is None or key[0][3] in replicates)
    ]
    outcomes = fit_properties(
        keys,
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        checkpoint=tools.property_checkpoint(parts) if budget is None else None,
    )
    if streams is not None or replicates is not None:
        # A part's rows live in its checkpoints; the declared run assembles the families.
        return pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CONTROLLED_LABELS,
                efficiency_bounds=efficiency_bounds(),
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.reindex(columns=[*REPLICATE_COLUMNS, "remainder_proxy"]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


#: Cloud parts of the largest streams, each planned at 80 CPU-h (20 h on 4 cores) or less:
#: ``stream -> parts``, split into equal replicate ranges.  Every other stream runs whole.
RUNNER_PARTS: dict[str, int] = {"longitudinal_SDR": 5, "longitudinal_LTMLE": 3}
#: The CPU-hours a cloud part is planned at, at most.
PART_HOURS = 80.0


def part_ranges(stream: str) -> list[range]:
    """The replicate ranges of a stream's cloud parts."""
    replicates = next(fit_set[4] for fit_set in FIT_SETS if fit_set[1] == stream)
    parts = RUNNER_PARTS.get(stream, 1)
    bounds = np.linspace(0, replicates, parts + 1).round().astype(int)
    return [range(int(a), int(b)) for a, b in itertools.pairwise(bounds)]


# ------------------------------------------------------------------ routes

#: The ratio route each configuration's fit records; ``None`` where the key fits no policy
#: ratio of the package (the study-local pre-fix one-step estimate, the cluster readings).
ROUTES: dict[tuple[str, str], str | None] = {
    ("interval_calibration", "density"): "density",
    ("interval_calibration", "wrong_outcome"): "density",
    ("interval_calibration", "both_wrong"): "density",
    ("interval_calibration", "pre_fix"): None,
    ("interval_calibration", "classifier"): "classifier",
    ("interval_calibration", "classifier_wrong_outcome"): "classifier",
    ("interval_calibration", "classifier_both_wrong"): "classifier",
    ("interval_calibration", "longitudinal"): "classifier",
    ("ic_gap", "shipped"): "density",
    ("ic_gap", "pre_fix"): "density",
    ("ic_gap", "support_ignored"): "density",
    ("ic_gap", "no_row_break"): "density",
    ("cluster_warning", "cluster"): None,
}


def resolved_route(payload: _Payload, frame: pd.DataFrame) -> str | None:
    """The route a key's fit resolves to on ``frame``, read from its estimator without a fit."""
    family, _, configuration, _, _, _, label = payload
    if ROUTES[(family, configuration)] is None:
        return None
    if configuration == "longitudinal":
        estimator = fits.longitudinal_estimator(label)
        return estimator.ratio_route(fits.longitudinal_data(frame)).route
    if configuration.startswith("classifier"):
        estimator = fits.classifier_estimator(
            label,
            outcome="correct" if configuration == "classifier" else "wrong",
            without_w1=configuration == "classifier_both_wrong",
        )
        return estimator.ratio_route(fits.point_data(label, frame, route="classifier")).route
    kind = configuration if family == "ic_gap" else "shipped"
    with fits.reading(kind):
        estimator = fits.density_estimator(label, frame)
    with_support = kind not in {"support_ignored", "no_row_break"}
    data = fits.point_data(label, frame, route="density", with_support=with_support)
    return estimator.ratio_route(data).route


# ------------------------------------------------------------------ declared cells


class Declared:
    """The law a cell reads, with its truth, for the declaration table.

    Parameters
    ----------
    name : str
        The cell label.
    estimand : str
        The estimand the cell's rows read.
    value : float
        Its truth: the policy mean, or 0 for an ``ic_gap`` or ``cluster_warning`` row.
    """

    def __init__(self, name: str, estimand: str, value: float) -> None:
        self.name = f"mtp_density_tails__{name}"
        self.estimand = estimand
        self.value = value

    def truth(self) -> dict[str, float]:
        return {self.estimand: self.value}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []
    for key in _payloads(FIT_SETS, 1):
        family, stream, configuration, _, n, _, label = key[0]
        replicates = next(
            fit_set[4] for fit_set in FIT_SETS if fit_set[0] == family and fit_set[1] == stream
        )
        for name in cells_of(key):
            cell = name.split("/", 1)[1]
            prefix = cell.split("__", 1)[0]
            if family == "interval_calibration" and configuration == "longitudinal":
                estimand = next(e for e, lab in LONGITUDINAL_LABELS[label].items() if lab == prefix)
                dgp = Declared(prefix, estimand, fits.LONGITUDINAL_TRUTH[estimand])
            elif family == "interval_calibration":
                dgp = Declared(prefix, study.ESTIMAND, float(law.TRUTH[label][0]))
            else:
                estimand = "ic_gap" if family == "ic_gap" else "warned"
                dgp = Declared(prefix, estimand, 0.0)
            cells.append(
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=dgp,
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=n,
                    replicates=replicates,
                    seed=_seed(family, stream, 0),
                    role=_role(name),
                    estimand=dgp.estimand,
                )
            )
    return tuple(cells)


def synthetic_scale(cell: PropertyCell) -> float:
    """The standard deviation of a synthetic cell's estimates for the property differential.

    A calibration cell spreads at its efficiency bound over ``sqrt(n)`` (a longitudinal one at
    the bound of the gated contrast), an ``ic_gap`` cell at
    a tenth of its 2,000 bound, and a ``cluster_warning`` indicator at 0.01.
    """
    if cell.property == "ic_gap":
        name = cell.cell.split("__", 1)[0]
        return 0.1 * GAP_BOUND_SMALL.get(name, 0.1)
    if cell.property == "cluster_warning":
        return 0.01
    prefix = cell.cell.split("__", 1)[0]
    name = prefix.split("_", 1)[0]
    if name in study.LONGITUDINAL:
        sd = efficiency_bounds()[LTMLE_GATED]
    else:
        sd = float(law.EFFICIENCY_SD.get(name, 1.0))
    return float(sd / np.sqrt(cell.n))


def _synthetic_centres() -> dict[str, float]:
    """Where the property differential centres each ``ic_gap`` and ``cluster_warning`` cell.

    A positive gap sits at half its 2,000 bound, and at a quarter of it at 20,000, so that each
    clause holds with room; a control at twice the bound of its named clause, and the fall
    control at its 2,000 centre.  A cluster indicator sits at 1 where the cell warns.
    """
    centres: dict[str, float] = {}
    for name in (*study.GAP_GATED, *study.GAP_REPORTED):
        small = GAP_BOUND_SMALL.get(name, 0.1)
        centres[f"ic_gap/{name}__n_2000"] = 0.5 * small
        centres[f"ic_gap/{name}__n_20000"] = 0.25 * small
    for cell, clause in GAP_CONTROLS.items():
        name = cell.split("__", 1)[0]
        bound = GAP_BOUND_LARGE[name] if clause == "large" else GAP_BOUND_SMALL[name]
        centres[f"ic_gap/{cell}"] = 2.0 * bound
    centres["cluster_warning/C__n_20000"] = 1.0
    centres["cluster_warning/C__gap_disabled_control"] = 0.0
    centres["cluster_warning/GT__pooled"] = 0.0
    centres["cluster_warning/GT__eager_control"] = 1.0
    return centres


#: :func:`_synthetic_centres`, read by the property differential.
SYNTHETIC_CENTRES = _synthetic_centres()


# ------------------------------------------------------------------ verdicts


def _gap_values(rows: pd.DataFrame, cell: str) -> np.ndarray:
    selected = rows.loc[(rows["property"] == "ic_gap") & (rows["cell"] == cell), "estimate"]
    return selected.to_numpy(dtype=float)


def _small_cell(cell: str) -> str:
    return cell.replace("n_20000", "n_2000")


def _clause_of(cell: str) -> tuple[str, ...]:
    """The ``ic_gap`` clauses a row reads: its own named control clause, or a positive's."""
    if cell in GAP_CONTROLS:
        return (GAP_CONTROLS[cell],)
    if cell.endswith("n_2000"):
        return ("small",)
    return ("large", "fall")


def _bound(cell: str, clause: str) -> float:
    name = cell.split("__", 1)[0]
    return {"small": GAP_BOUND_SMALL, "large": GAP_BOUND_LARGE}[clause][name]


def gap_clause_holds(
    clause: str, cell: str, values: np.ndarray, small: np.ndarray | None = None
) -> tuple[float, bool]:
    """``(slack, holds)`` of one positive ``ic_gap`` clause on the gaps of ``cell``.

    ``small`` is the 2,000 cell's gaps, which the fall clause reads.
    """
    if clause == "small":
        median = float(np.median(values))
        return _bound(cell, clause) - median, median <= _bound(cell, clause)
    if clause == "large":
        largest = float(np.max(values))
        return _bound(cell, clause) - largest, largest <= _bound(cell, clause)
    assert small is not None
    ratio = float(np.median(values) / np.median(small))
    return GAP_FALL - ratio, ratio <= GAP_FALL


def gap_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Read every ``ic_gap`` row: a positive meets its clauses, a control fails its named one.

    Publishes ``median_gap`` and ``max_gap`` on every row and ``gap_fall`` (the 20,000 median
    over the 2,000 median of the same arm) on each 20,000 row.
    """
    mask = summary["property"] == "ic_gap"
    for index in summary.index[mask]:
        cell = str(summary.loc[index, "cell"])
        values = _gap_values(rows, cell)
        summary.loc[index, "median_gap"] = float(np.median(values))
        summary.loc[index, "max_gap"] = float(np.max(values))
        small = None
        if cell.endswith("n_20000"):
            small = _gap_values(rows, _small_cell(cell))
            if small.size:
                summary.loc[index, "gap_fall"] = float(np.median(values) / np.median(small))
        role = str(summary.loc[index, "role"])
        if role == DIAGNOSTIC_ROLE:
            summary.loc[index, "passed"] = True
            summary.loc[index, "property_passed"] = True
            continue
        holds = [gap_clause_holds(c, cell, values, small)[1] for c in _clause_of(cell)]
        summary.loc[index, "passed"] = bool(all(holds) if role == "positive" else not holds[0])


#: A ``cluster_warning`` row's ``estimate`` is its draw's indicator, 1 when the helper warns;
#: an estimate above this threshold reads as a warning.
WARNED = 0.5


def warned_count(estimates: Any) -> int:
    """The draws whose indicator reads as a warning."""
    return int(np.sum(np.asarray(estimates, dtype=float) > WARNED))


def cluster_clause(cell: str, warned: int, lost: int) -> tuple[float, bool]:
    """``(slack, passed)`` of a ``cluster_warning`` row with ``lost`` draws at their worst."""
    if cell == "C__n_20000":
        return float(warned - CLUSTER_C_FLOOR), warned >= CLUSTER_C_FLOOR
    if cell == "GT__pooled":
        total = warned + lost
        return float(CLUSTER_GT_CEILING - total), total <= CLUSTER_GT_CEILING
    if cell == "C__gap_disabled_control":
        total = warned + lost
        return float(CLUSTER_C_FLOOR - 1 - total), total < CLUSTER_C_FLOOR
    if cell == "GT__eager_control":
        return float(warned - CLUSTER_GT_CEILING - 1), warned > CLUSTER_GT_CEILING
    raise ValueError(f"unknown cluster_warning cell {cell!r}")


def cluster_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Read the two cluster clauses and their controls; publish ``warned_draws``."""
    mask = summary["property"] == "cluster_warning"
    for index in summary.index[mask]:
        cell = str(summary.loc[index, "cell"])
        flags = rows.loc[
            (rows["property"] == "cluster_warning") & (rows["cell"] == cell), "estimate"
        ].to_numpy(dtype=float)
        warned = warned_count(flags)
        summary.loc[index, "warned_draws"] = float(warned)
        summary.loc[index, "passed"] = cluster_clause(cell, warned, 0)[1]


def _calibration_kind(cell: str) -> str:
    return cell.rsplit("__", 1)[-1]


def remainder_reading(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish the classifier DR rows' mean remainder proxy, in efficiency SEs."""
    if "remainder_proxy" not in rows:
        return
    for index in summary.index[summary["property"] == "interval_calibration"]:
        cell = str(summary.loc[index, "cell"])
        values = rows.loc[
            (rows["property"] == "interval_calibration") & (rows["cell"] == cell),
            "remainder_proxy",
        ].dropna()
        if len(values):
            summary.loc[index, "remainder_proxy"] = float(values.mean())


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration, ``ic_gap`` and ``cluster_warning`` verdicts, and the reported rows."""
    summary, rates = apply_shared_verdicts(
        rows.loc[:, list(REPLICATE_COLUMNS)], STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=()
    )
    calibration = summary["property"] == "interval_calibration"
    kinds = summary["cell"].astype(str).map(_calibration_kind)
    reported = calibration & (summary["role"] == DIAGNOSTIC_ROLE)
    coverage_controls = calibration & kinds.isin(list(COVERAGE_CONTROLS))
    gated = summary.loc[calibration & ~reported & ~coverage_controls].copy()
    calibration_verdicts(
        gated, margins=STUDY.margins, efficiency_band=None, positive_suffix=POSITIVE_SUFFIXES
    )
    summary.loc[gated.index, "passed"] = gated["passed"]
    floor = STUDY.margins.calibration_coverage[0]
    summary.loc[coverage_controls, "passed"] = (
        summary.loc[coverage_controls, "coverage_ci_upper"] < floor
    )
    summary.loc[reported, "passed"] = True
    summary.loc[reported, "property_passed"] = True
    remainder_reading(summary, rows)
    gap_verdicts(summary, rows)
    cluster_verdicts(summary, rows)
    return finish(summary, rates)


# ------------------------------------------------------------------ declared rules


def _gap_clause(name: str, clause: str, role: str) -> pf.Clause:
    """An ``ic_gap`` clause with its adverse fills: lost gaps at the survivor extreme.

    A positive's lost gaps sit at the survivor maximum for a cap, and the fall clause's 2,000
    gaps at the survivor minimum; a control's at the opposite ends.  ``range``: a lost gap lies
    in the survivor range.
    """
    statistic = {"small": "S10m", "large": "S10x", "fall": "S10r"}[clause]
    positive = role == "positive"

    def applies(ctx: pf.Context, row: pd.Series) -> bool:
        cell = str(row["cell"])
        return str(row["role"]) == role and clause in _clause_of(cell)

    def judge(cell: str, values: np.ndarray, small: np.ndarray | None) -> pf.Judged:
        slack, holds = gap_clause_holds(clause, cell, values, small)
        return pf.Judged(slack, holds) if positive else pf.Judged(-slack, not holds)

    def small_of(ctx: pf.Context, cell: str) -> np.ndarray | None:
        if clause != "fall":
            return None
        return ctx.cell("ic_gap", _small_cell(cell))["estimate"].to_numpy(dtype=float)

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        cell = str(row["cell"])
        values = ctx.cell("ic_gap", cell)["estimate"].to_numpy(dtype=float)
        return judge(cell, values, small_of(ctx, cell))

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        cell = str(row["cell"])
        values = ctx.cell("ic_gap", cell)["estimate"].to_numpy(dtype=float)
        k = ctx.k(row)
        fill = float(values.max() if positive else values.min())
        small = small_of(ctx, cell)
        if small is not None and k:
            # The fall's denominator moves the other way.
            small_fill = float(small.min() if positive else small.max())
            small = np.concatenate(
                [small, np.full(ctx.lost_in("ic_gap", _small_cell(cell)), small_fill)]
            )
        filled = np.concatenate([values, np.full(k, fill)])
        side = "max" if positive else "min"
        return [(f"lost gaps at survivor {side}", judge(cell, filled, small))]

    def reads(ctx: pf.Context, row: pd.Series) -> list[tuple[str, str]]:
        cell = str(row["cell"])
        return [("ic_gap", _small_cell(cell))] if clause == "fall" else []

    return pf.Clause(
        name,
        statistic,
        applies,
        read,
        candidates,
        imputation=pf.RANGE,
        reads=reads if clause == "fall" else None,
    )


def _cluster_clause(name: str, cell: str) -> pf.Clause:
    """A ``cluster_warning`` count clause: lost draws at their adverse indicator (support)."""

    def applies(ctx: pf.Context, row: pd.Series) -> bool:
        return str(row["cell"]) == cell

    def warned(ctx: pf.Context) -> int:
        return warned_count(ctx.cell("cluster_warning", cell)["estimate"])

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        return pf.Judged(*cluster_clause(cell, warned(ctx), 0))

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        return [
            (
                "lost draws at the adverse indicator",
                pf.Judged(*cluster_clause(cell, warned(ctx), ctx.k(row))),
            )
        ]

    return pf.Clause(name, "S3c", applies, read, candidates, imputation=pf.SUPPORT)


def _calibration_rule() -> pf.FamilyRule:
    """The shared calibration rule with the two DR coverage controls added."""
    shared = property_verdicts.calibration_rule(efficiency_band=None)

    def kind(name: str) -> pf.Applies:
        return lambda ctx, row: (
            str(row["role"]) == "control" and _calibration_kind(str(row["cell"])) == name
        )

    extra = tuple(
        pf.proportion_clause(
            clause,
            statistic="S2",
            judge=pf.below_judge(lambda ctx: ctx.margins.calibration_coverage[0]),
            applies=kind(control),
        )
        for control, clause in COVERAGE_CONTROLS.items()
    )
    return pf.FamilyRule(
        families=frozenset({"interval_calibration"}),
        shared="summarize_properties",
        clauses=(*shared.clauses, *extra),
    )


#: The failure accounting's declared rules.
PROPERTY_RULES = (
    _calibration_rule(),
    pf.FamilyRule(
        families=frozenset({"ic_gap"}),
        shared="gap_verdicts",
        clauses=(
            _gap_clause("2,000 median at or below its bound", "small", "positive"),
            _gap_clause("every 20,000 gap at or below its bound", "large", "positive"),
            _gap_clause("20,000 median at most 0.6 of the 2,000 median", "fall", "positive"),
            _gap_clause("control 2,000 median above its bound", "small", "control"),
            _gap_clause("control 20,000 gap above its bound", "large", "control"),
            _gap_clause("control median falls less than 0.6", "fall", "control"),
        ),
    ),
    pf.FamilyRule(
        families=frozenset({"cluster_warning"}),
        shared="cluster_verdicts",
        clauses=(
            _cluster_clause("at least 26 of 30 C draws warn", "C__n_20000"),
            _cluster_clause("at most 6 of 220 G and T draws warn", "GT__pooled"),
            _cluster_clause("gap-disabled helper misses the C clause", "C__gap_disabled_control"),
            _cluster_clause("eager helper breaks the G and T clause", "GT__eager_control"),
        ),
    ),
)


def gated_cells() -> tuple[str, ...]:
    """The ``family/cell`` names whose red verdict gates the study."""
    return tuple(
        f"{cell.property}/{cell.cell}" for cell in declared_cells() if cell.role != DIAGNOSTIC_ROLE
    )


__all__ = [
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "FIT_SETS",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "gated_cells",
    "generate_property_rows",
    "summarize_properties",
]


def _range(text: str | None) -> range | None:
    if text is None:
        return None
    start, stop = (int(part) for part in text.split(":"))
    return range(start, stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One cloud part of the declared property run.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--streams", nargs="+", required=True)
    parser.add_argument("--replicates", default=None, help="START:STOP")
    arguments = parser.parse_args()
    generate_property_rows(
        n_jobs=arguments.jobs,
        parts=arguments.parts,
        streams=arguments.streams,
        replicates=_range(arguments.replicates),
    )
