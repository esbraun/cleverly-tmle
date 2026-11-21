"""Refutation of a fit on a simple random sample of its units (X106 part H).

``refute(result, sample_limit=m)`` draws ``m`` of the fit's ``N`` independent units without
replacement (whole clusters on a clustered fit, each unit with its whole trajectory on a panel),
refits the stored estimator on them, and runs the requested tests on that refit, the *sampled
baseline*, exactly as :func:`~cleverly.validation.refute` runs them on any fit of ``m`` units.
Every reference a test reads comes from the sampled baseline and the sampled rows; nothing of the
full-data fit enters a test.

The units of an i.i.d. sample, chosen by a generator that never reads the data, are themselves
an i.i.d. sample of size ``m``. So a sampled verdict has, unconditionally on the data, the
distribution of the full-data verdict on ``m`` units, for every ``N >= m``, and the verdict
registry reads the sampled baseline's own configuration. A sampled verdict is therefore the
verdict of the analysis on ``m`` units, never a check of the ``N``-unit fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._refusal import Refusal, RefusalReason
from ..data.validate import MIN_OBSERVATIONS
from ..exceptions import DataError

__all__ = ["RefutationSample"]

#: The inclusion rules ``sampling=`` names.
SAMPLING_RULES = ("units", "clusters")


@dataclass(frozen=True)
class RefutationSample:
    """How a sampled :func:`~cleverly.validation.refute` call drew the units it refuted.

    :attr:`RefutationResult.sample <cleverly.validation.RefutationResult>` holds one when the
    call was given ``sample_limit=``.

    Parameters
    ----------
    limit : int
        The ``sample_limit`` the call was given.
    rule : {"units", "clusters"}
        The inclusion rule: a simple random sample without replacement of subjects, or of whole
        clusters.
    applied : bool
        ``False`` when the limit is at least the fit's unit count, so the full data ran.
    units : int
        The fit's independent units: subjects, or clusters on a clustered fit.
    sampled_units : int
        The units the tests read.
    rows : int
        The fit's rows.
    sampled_rows : int
        The rows of the sampled units.
    seed : int or None
        The seed the sample was drawn from, a child of the call's root seed. ``None`` when
        nothing was sampled.
    full_estimate : float
        The full-data fit's estimate. It is shown for information and is not the reference of
        any test.
    baseline_estimate : float
        The estimate of the fit refitted on the sampled units, which every test compares with.
        Equal to :attr:`full_estimate` when nothing was sampled.
    baseline_std_error : float
        Its plug-in standard error on the inference scale.
    departures : tuple of str
        Where the sampled data differ from data prepared from the sampled rows alone: a
        covariate column constant or duplicated in the sample but kept from the full data. A
        fresh fit on those rows would have dropped it, so the sampled analysis is not exactly
        the analysis of that many units, and every verdict is withheld. Empty in the usual
        case.

    Attributes
    ----------
    inclusion_probability : float
    """

    limit: int
    rule: Literal["units", "clusters"]
    applied: bool
    units: int
    sampled_units: int
    rows: int
    sampled_rows: int
    seed: int | None
    full_estimate: float
    baseline_estimate: float
    baseline_std_error: float
    departures: tuple[str, ...] = ()

    @property
    def inclusion_probability(self) -> float:
        """The probability each unit is in the sample: ``sampled_units / units``."""
        return self.sampled_units / self.units

    def sentence(self) -> str:
        """The line a report leads with.

        Returns
        -------
        str
            One or two plain sentences.
        """
        noun = "clusters" if self.rule == "clusters" else "units"
        if not self.applied:
            return (
                f"sample_limit={self.limit:,} is at least the fit's {self.units:,} {noun}, so "
                "the full data was refuted."
            )
        text = (
            f"Refuted on a simple random sample of {self.sampled_units:,} of {self.units:,} "
            f"{noun} ({self.sampled_rows:,} of {self.rows:,} rows, seed {self.seed}). Each test "
            "compares with the fit refitted on that sample "
            f"(estimate {self.baseline_estimate:.5g}); the full-data estimate "
            f"{self.full_estimate:.5g} is shown for information only."
        )
        if self.departures:
            text += f" {_sentence_case(withheld_clause(self.departures))}."
        return text

    def limits(self) -> tuple[str, ...]:
        """The two limits a full report states beside a sampled verdict.

        Returns
        -------
        tuple of str
            Plain sentences.
        """
        if not self.applied:
            return ()
        return (
            "A verdict here is the verdict of the analysis refitted on the sampled units, with "
            "the evidence the same test has on a fit of that many units; it is not a check of "
            "the full-data fit.",
            "Another seed draws another sample and can give another verdict, and a test reads "
            "with the power it has at the sampled size, so a defect that shows only with more "
            "units can pass here.",
        )


def validated_sampling(sample_limit: Any, sampling: Any) -> None:
    """Refuse malformed sampling keywords, as the other malformed arguments are refused.

    Parameters
    ----------
    sample_limit : Any
        The ``sample_limit`` argument.
    sampling : Any
        The ``sampling`` argument.

    Raises
    ------
    ValueError
        If ``sample_limit`` is not ``None`` or an integer of at least 2, or ``sampling`` is not
        ``None``, ``"units"`` or ``"clusters"``, or ``sampling`` is given without
        ``sample_limit``.
    """
    if sample_limit is not None and (
        isinstance(sample_limit, bool)
        or not isinstance(sample_limit, (int, np.integer))
        or sample_limit < 2
    ):
        raise ValueError("sample_limit must be None or an integer of at least 2")
    if sampling is not None and sampling not in SAMPLING_RULES:
        raise ValueError(f"unknown sampling {sampling!r}; choose None, 'units' or 'clusters'")
    if sampling is not None and sample_limit is None:
        raise ValueError(
            "sampling= names how sample_limit= draws its units; pass sample_limit=<units> "
            "with it, or drop sampling="
        )


def _cluster(result: Any) -> Any:
    data = getattr(result, "data", None)
    return None if data is None else getattr(data, "cluster", None)


def sampling_refusal(result: Any, sample_limit: int | None, sampling: str | None) -> Refusal | None:
    """The refusal a sampled request meets before any refit, or ``None``.

    Parameters
    ----------
    result : Any
        The fit.
    sample_limit : int or None
        The validated ``sample_limit``.
    sampling : str or None
        The validated ``sampling``.

    Returns
    -------
    Refusal or None
        The record, or ``None`` when the request is admitted or not sampled.
    """
    if sample_limit is None:
        return None
    clustered = _cluster(result) is not None
    if not clustered and sample_limit < MIN_OBSERVATIONS:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"sample_limit={sample_limit} draws fewer units than the {MIN_OBSERVATIONS} rows "
            f"a fit needs. Pass sample_limit={MIN_OBSERVATIONS} or more.",
        )
    if sampling == "units" and clustered:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "sampling='units' draws subjects one at a time, and this fit is clustered: its "
            "inference treats each cluster as one independent unit, so a sample of subjects "
            "would split clusters and the sampled analysis would not be an analysis of "
            "independent units. Pass sampling='clusters' or sampling=None to draw whole "
            "clusters.",
        )
    if sampling == "clusters" and not clustered:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "sampling='clusters' draws whole clusters, and this fit declares no clusters: its "
            "independent unit is the subject. Pass sampling='units' or sampling=None.",
        )
    return None


def resolved_rule(result: Any) -> Literal["units", "clusters"]:
    """The inclusion rule of a sampled call on ``result``: its independent unit."""
    return "clusters" if _cluster(result) is not None else "units"


def unit_count(result: Any) -> int:
    """The fit's independent units: its clusters on a clustered fit, else its subjects."""
    cluster = _cluster(result)
    if cluster is not None:
        return int(np.unique(np.asarray(cluster)).size)
    return int(result.data.n)


def draw_rows(data: Any, seed: int, size: int) -> np.ndarray:
    """The sorted rows of a simple random sample of ``size`` independent units.

    Parameters
    ----------
    data : Any
        The fit's data.
    seed : int
        The sample's own seed.
    size : int
        The units to draw; at most the unit count.

    Returns
    -------
    numpy.ndarray
        The row positions, increasing. On a clustered fit, every row of each drawn cluster.
    """
    rng = np.random.default_rng(seed)
    cluster = getattr(data, "cluster", None)
    if cluster is None:
        return np.sort(rng.choice(int(data.n), size=size, replace=False))
    labels = np.unique(np.asarray(cluster))
    chosen = rng.choice(labels, size=size, replace=False)
    return np.flatnonzero(np.isin(np.asarray(cluster), chosen))


def row_aligned(values: Any, rows: np.ndarray, n: int) -> Any:
    """``values`` restricted to ``rows`` when it holds one entry per row, else unchanged.

    The full request's checks have already read ``values`` against the fit's rows, so an array
    of another length was refused before this point; anything else passes through for the
    sampled preflight to check.
    """
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim >= 1 and array.shape[0] == n:
        return array[rows]
    return values


def _sentence_case(text: str) -> str:
    """``text`` with its first letter upper case and the rest, column names included, kept."""
    return text[:1].upper() + text[1:]


def withheld_clause(departures: tuple[str, ...]) -> str:
    """The clause a test prints when a sample departure withholds its verdict."""
    return (
        f"the sample makes {', '.join(departures)}, which a fit on these rows alone would drop, "
        "so the sampled analysis is not exactly the analysis of this many units and gives no "
        "pass or fail"
    )


Block = tuple[list[tuple[str, np.ndarray]], np.ndarray | None]


def _blocks(data: Any) -> list[Block]:
    """The covariate blocks a fresh preparation screens, each with the rows it screens on.

    A point fit's preparation screens its covariates together on every row. A longitudinal
    fit's screens the baseline block on every row and each time-varying block alone on the
    rows still observed at its node (``LongitudinalData.from_frame``), whose values are
    finite. A continuous-time survival fit's preparation drops no column, so it has none.
    """
    if hasattr(data, "covariates") and hasattr(data, "covariate_names"):
        matrix = np.asarray(data.covariates, dtype=float)
        return [([(str(name), matrix[:, j]) for j, name in enumerate(data.covariate_names)], None)]
    if not hasattr(data, "node_kind"):
        return []
    out: list[Block] = []
    baseline = np.asarray(data.baseline, dtype=float).reshape(data.n, -1)
    names = list(data.baseline_names)
    out.append(([(str(names[j]), baseline[:, j]) for j in range(baseline.shape[1])], None))
    for block, labels in zip(data.time_varying, data.time_varying_names, strict=True):
        matrix = np.asarray(block, dtype=float)
        if matrix.ndim != 2 or not matrix.shape[1]:
            continue
        present = np.isfinite(matrix).all(axis=1)
        out.append(([(str(labels[j]), matrix[:, j]) for j in range(matrix.shape[1])], present))
    return out


def _degenerate(blocks: list[Block], rows: np.ndarray | None) -> list[str]:
    """Columns constant, or exact copies of an earlier column of their block, on ``rows``.

    Each block is read on its own screened rows, restricted to ``rows``.
    """
    found: list[str] = []
    for columns, present in blocks:
        keep = np.ones(columns[0][1].size, dtype=bool) if present is None else present.copy()
        if rows is not None:
            chosen = np.zeros(keep.size, dtype=bool)
            chosen[rows] = True
            keep &= chosen
        seen: dict[bytes, str] = {}
        for name, values in columns:
            part = values[keep]
            finite = part[np.isfinite(part)]
            if finite.size and float(np.ptp(finite)) == 0.0:
                found.append(f"{name} constant")
                continue
            key = np.ascontiguousarray(np.nan_to_num(part, nan=np.inf)).tobytes()
            if key in seen:
                found.append(f"{name} a copy of {seen[key]}")
            else:
                seen[key] = name
    return found


def _levels(values: Any, rows: np.ndarray | None) -> list[set[float]]:
    """The observed levels of each column of a discrete array, on ``rows``."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if rows is not None:
        array = array[rows]
    return [set(np.unique(column[np.isfinite(column)]).tolist()) for column in array.T]


def _discrete_columns(data: Any) -> list[tuple[str, np.ndarray]]:
    """Each treatment column that has arm levels, with where it sits; none for a dose.

    A point fit's discrete treatment is one column and a continuous one has none. A vector
    treatment reads its categorical components, and a longitudinal fit its categorical nodes
    and the categorical components of its mixed nodes.
    """
    kind = getattr(data, "treatment_kind", None)
    if kind == "continuous":
        return []
    if kind == "vector":
        components = np.asarray(data.treatment_components, dtype=float)
        return [
            (f" at component {index + 1}", components[:, index])
            for index, levels in enumerate(data.component_levels)
            if levels
        ]
    if not hasattr(data, "node_kind"):
        return [("", np.asarray(data.treatment, dtype=float))]
    treatment = np.asarray(data.treatment, dtype=float).reshape(data.n, -1)
    columns = []
    for time in range(1, treatment.shape[1] + 1):
        node = data.node_kind(time)
        if node == "categorical":
            columns.append((f" at node {time}", treatment[:, time - 1]))
        elif node == "mixed":
            values = np.asarray(data.node_values(time), dtype=float)
            columns += [
                (f" at node {time}, component {index + 1}", values[:, index])
                for index, levels in enumerate(data.component_levels[time - 1])
                if levels
            ]
    return columns


def check_sample(data: Any, rows: np.ndarray) -> tuple[str, ...]:
    """Refuse a sample that loses a treatment level or stratum; list softer departures.

    A sample that loses an arm or a stratum is a sample on which the fit's estimand is not
    defined, so it raises before the baseline refit. A covariate that is constant, or a copy
    of another, in the sample but not in the full data is kept by the subset and would be
    dropped by a fresh preparation of the sampled rows; it is returned so the verdicts are
    withheld.

    Parameters
    ----------
    data : Any
        The fit's data.
    rows : numpy.ndarray
        The sampled rows.

    Returns
    -------
    tuple of str
        The departures, in column order.

    Raises
    ------
    DataError
        When the sample holds no unit of a treatment level or stratum the fit holds.
    """
    for where, column in _discrete_columns(data):
        (full,), (part,) = _levels(column, None), _levels(column, rows)
        lost = sorted(full - part)
        if lost:
            raise DataError(
                f"the sample holds no unit with treatment code {lost[0]:g}{where}, which "
                "the fit holds, so the fit's estimand is not defined on it. Raise "
                "sample_limit, or refute the full data."
            )
    strata = getattr(data, "strata", None)
    if strata is not None:
        for full, part in zip(_levels(strata, None), _levels(strata, rows), strict=True):
            if full - part:
                raise DataError(
                    "the sample holds no unit of a stratum the fit holds, so that stratum's "
                    "estimate is not defined on it. Raise sample_limit, or refute the full data."
                )
    blocks = _blocks(data)
    before = set(_degenerate(blocks, None))
    return tuple(item for item in _degenerate(blocks, rows) if item not in before)


def sampled_plan(plan: Any, sampled: Any) -> Any:
    """The split plan of the sampled rows, drawn from ``plan``'s generator records.

    Every plan a fit accepts records how :func:`~cleverly.learners.random_partition` drew each
    repeat (scheme, requested folds, seed), and that draw reads no data. The same records drawn
    on the sampled units are the plan a caller would build for a fit on those rows, so the
    sampled baseline runs the split the full fit's plan prescribes for them.
    """
    from ..learners.crossfit import SplitPlan, random_partition
    from ..provenance import data_fingerprint

    folds = [
        random_partition(
            int(sampled.n),
            int(origin.requested_n_folds),
            cluster=None if origin.scheme != "grouped" else sampled.cluster,
            seed=int(origin.seed),
        )
        for origin in plan.provenance
    ]
    try:
        fingerprint = data_fingerprint(sampled)
    except (AttributeError, TypeError):
        fingerprint = None
    return SplitPlan.from_folds(folds, source_fingerprint=fingerprint)
