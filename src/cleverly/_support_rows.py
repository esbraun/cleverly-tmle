"""The support rows of the point-treatment estimators.

Each row names one configuration that a registered study covers, the kinds of inference the
study licenses there, and the study cells that license it. The rows are explicit tuples: a
row covers a set of values on one axis only where :func:`rows_over` expands it into one row
per value, and each expanded row has a case in ``tests/studies/support_cases.py``.

The row ids follow ``docs/roadmap.md`` and the support inventory: ``T`` for ordinary TMLE,
``D`` for DR-TMLE and ``C`` for C-TMLE. A suffix names the expanded value. The longitudinal
rows (``G``) are a later addition.

A leaf module: it imports :mod:`cleverly._support` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from ._support import KEY_AXES, Kind, SupportKey, SupportRow

__all__ = ["POINT_ROWS", "default_options", "merge_rows", "point_row", "rows_over"]

BINARY: Final[tuple[str, ...]] = ("binomial",)
CONTINUOUS: Final[tuple[str, ...]] = ("gaussian", "unbounded")
BOUNDED: Final[tuple[str, ...]] = ("gaussian", "bounded")

_WALD: Final[frozenset[Kind]] = frozenset({"point", "wald"})
_WALD_T: Final[frozenset[Kind]] = frozenset({"point", "wald_t"})
_POINT: Final[frozenset[Kind]] = frozenset({"point"})


def default_options(engine: str) -> dict[str, Any]:
    """The options a default fit of ``engine`` carries in its key.

    Parameters
    ----------
    engine : str
        ``"tmle"``, ``"drtmle"`` or ``"collaborative_tmle"``.

    Returns
    -------
    dict of str to Any
        The option names and values of :func:`cleverly._support.point_key`.
    """
    options: dict[str, Any] = {
        "fluctuation": "logistic",
        "targeting": "iterative",
        "target_weights": False,
        "multiplier_kind": "rademacher",
    }
    if engine == "drtmle":
        options |= {
            "guard": ("Q", "g"),
            "reduction": "univariate",
            "reduced_crossfit": "pooled",
            "update_order": "drtmle",
            "randomized": False,
            "evaluation": False,
        }
    return options


def point_row(
    row_id: str,
    evidence: Sequence[str],
    *,
    families: Iterable[str],
    outcome: tuple[str, ...] = BINARY,
    missing: tuple[str, ...] = ("complete",),
    treatment: tuple[str, ...] = ("binary",),
    weighting: str = "none",
    splitting: tuple[str, ...] = ("in_sample",),
    dependence: str = "iid",
    qualifiers: Iterable[str] = (),
    options: Mapping[str, Any] | None = None,
    engine: tuple[str, ...] = ("tmle",),
    kinds: frozenset[Kind] = _WALD,
    cells: Sequence[str] = (),
    limits: Sequence[str] = (),
) -> SupportRow:
    """One support row of a point-treatment estimator, with the defaults of an ordinary fit.

    Parameters
    ----------
    row_id : str
        The row id.
    evidence : sequence of str
        The registered study slugs.
    families : iterable of str
        The target families the row covers, as ``P1`` to ``P11``.
    outcome, missing, treatment, splitting : tuple of str
        The axis values of :class:`~cleverly._support.SupportKey`.
    weighting, dependence : str
        The axis values.
    qualifiers : iterable of str, default ()
        ``strata`` and ``known_mechanism``.
    options : mapping of str to Any or None, default None
        Overrides of :func:`default_options`.
    engine : tuple of str, default ("tmle",)
        The engine axis.
    kinds : frozenset of str, default Wald
        What the row licenses.
    cells : sequence of str, default ()
        The licensing study cells.
    limits : sequence of str, default ()
        The stated limits of the row.

    Returns
    -------
    SupportRow
        The row.
    """
    merged = default_options(engine[0]) | dict(options or {})
    key = SupportKey(
        kind="point",
        engine=engine,
        family=frozenset(families),
        outcome=outcome,
        missing=missing,
        treatment=treatment,
        weighting=weighting,
        splitting=splitting,
        dependence=dependence,
        qualifiers=frozenset(qualifiers),
        options=tuple(sorted(merged.items())),
    )
    return SupportRow(row_id, key, kinds, tuple(evidence), tuple(cells), None, tuple(limits))


def rows_over(
    base: SupportRow, axis: str, values: Mapping[str, Any], *, evidence: Sequence[str] | None = None
) -> tuple[SupportRow, ...]:
    """Expand one row over the values of one axis.

    Each value becomes its own row, so a case has to exist for each one.

    Parameters
    ----------
    base : SupportRow
        The row to expand.
    axis : str
        An axis of :class:`~cleverly._support.SupportKey`, other than ``family``.
    values : mapping of str to Any
        The row-id suffix and the axis value of each expanded row.
    evidence : sequence of str or None, default None
        Evidence of the expanded rows. ``None`` keeps the base evidence.

    Returns
    -------
    tuple of SupportRow
        One row per value, with the id ``<base id>.<suffix>``.

    Raises
    ------
    ValueError
        When ``axis`` is not an axis of the key, or is ``family``.
    """
    if axis not in KEY_AXES or axis == "family":
        raise ValueError(f"cannot expand a row over {axis!r}")
    return tuple(
        replace(
            base,
            id=f"{base.id}.{suffix}",
            key=replace(base.key, **{axis: value}),
            evidence=base.evidence if evidence is None else tuple(evidence),
        )
        for suffix, value in values.items()
    )


def merge_rows(row_id: str, *rows: SupportRow) -> SupportRow:
    """Merge rows whose keys differ only in family into one row.

    Two rows that cover the same fit would both match it, so rows of one configuration are
    one row. The merged row covers the union of the families, and names every study and cell.

    Parameters
    ----------
    row_id : str
        The id of the merged row.
    *rows : SupportRow
        The rows to merge. They agree on every axis except the family.

    Returns
    -------
    SupportRow
        The merged row.

    Raises
    ------
    ValueError
        When the rows differ on an axis other than the family.
    """
    first = rows[0]
    for other in rows[1:]:
        if replace(other.key, family=first.key.family) != first.key:
            raise ValueError(f"rows {first.id} and {other.id} differ on more than the family")

    def joined(field: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for row in rows for item in getattr(row, field)))

    return SupportRow(
        row_id,
        replace(first.key, family=frozenset().union(*(row.key.family for row in rows))),
        frozenset().union(*(row.kinds for row in rows)),
        joined("evidence"),
        joined("cells"),
        None,
        joined("limits"),
    )


def _merged(
    rows: Sequence[SupportRow], groups: Mapping[str, Sequence[str]]
) -> tuple[SupportRow, ...]:
    """Replace each group of rows by its merge, at the place of the group's first row."""
    by_id = {row.id: row for row in rows}
    grouped = {member for members in groups.values() for member in members}
    merged = {
        members[0]: merge_rows(new_id, *(by_id[m] for m in members))
        for new_id, members in groups.items()
    }
    return tuple(merged.get(row.id, row) for row in rows if row.id not in grouped - merged.keys())


_ALL_ARM = ("P1", "P2", "P3", "P4")


def _ctmle(strategy: str, design: str = "None") -> tuple[str, ...]:
    return ("collaborative_tmle", strategy, design)


def _guard(*names: str) -> dict[str, Any]:
    return {"guard": tuple(names)}


def _t_rows() -> tuple[SupportRow, ...]:
    """Ordinary TMLE (``T`` rows)."""
    boot: frozenset[Kind] = frozenset({"point", "wald", "bootstrap"})
    rows: list[SupportRow] = []
    rows += rows_over(
        point_row("T01", ["canonical-tmle"], families=_ALL_ARM, cells=["binary", "continuous"]),
        "outcome",
        {"binary": BINARY, "continuous": CONTINUOUS},
    )
    # The full-refit bootstrap cell is the binary ATE of the in-sample ordinary fit.
    rows[0] = replace(
        rows[0],
        kinds=boot,
        evidence=("canonical-tmle", "full-refit-bootstrap-and-derived-contrasts"),
        cells=("binary", "boot_ate_point_tmle"),
    )
    rows += rows_over(
        point_row(
            "T02",
            ["canonical-cvtmle"],
            families=_ALL_ARM,
            splitting=("stacked",),
            cells=["binary", "continuous"],
        ),
        "outcome",
        {"binary": BINARY, "continuous": BOUNDED},
    )
    rows += rows_over(
        point_row(
            "T03",
            ["fold-evaluated-cvtmle"],
            families=("P1", "P3"),
            splitting=("fold_evaluated",),
            cells=["binary", "continuous"],
        ),
        "outcome",
        {"binary": BINARY, "continuous": BOUNDED},
    )
    # The fold-targeted study reads the fold-evaluated ATE on the binary law as its other leg.
    rows[-2] = replace(
        rows[-2],
        evidence=("fold-evaluated-cvtmle", "fold-targeted-cvtmle"),
        cells=("binary", "fold_evaluated_leg"),
    )
    rows.append(
        point_row(
            "T04",
            ["fold-targeted-cvtmle"],
            families=("P1",),
            splitting=("fold_targeted_evaluated",),
            cells=["binary"],
        )
    )
    rows += rows_over(
        point_row(
            "T05",
            ["repeated-crossfit-tmle"],
            families=_ALL_ARM,
            splitting=("stacked", "repeated"),
            cells=["binary", "continuous"],
        ),
        "outcome",
        {"binary": BINARY, "continuous": BOUNDED},
    )
    rows.append(
        point_row(
            "T06",
            ["weighted-tmle"],
            families=("P1", "P2"),
            weighting="fixed",
            cells=["binary_biased_sample"],
        )
    )
    rows.append(
        point_row(
            "T07",
            ["learned-weighted-tmle"],
            families=("P1",),
            outcome=CONTINUOUS,
            weighting="fixed",
            options={"fluctuation": "linear"},
            cells=["continuous_selected_weighted_nuisances"],
        )
    )
    rows.append(
        point_row(
            "T08",
            ["canonical-multi-arm-tmle"],
            families=("P1", "P2"),
            treatment=("arms3",),
            cells=["multi_arm_binary"],
        )
    )
    rows.append(
        point_row(
            "T09",
            ["clustered-tmle"],
            families=("P1",),
            splitting=("stacked",),
            dependence="clustered",
            cells=["clustered_binary"],
        )
    )
    rows += rows_over(
        point_row(
            "T10",
            ["clustered-unequal-cvtmle"],
            families=("P1",),
            splitting=("stacked",),
            dependence="clustered",
            cells=["unequal_noninformative", "unequal_informative"],
            limits=["The fold-evaluated cluster-covariate cell is red (F28)."],
        ),
        "weighting",
        {"unweighted": "none", "cluster_average": "fixed"},
    )
    rows += rows_over(
        point_row(
            "T10",
            ["clustered-unequal-cvtmle"],
            families=("P1",),
            splitting=("fold_evaluated",),
            dependence="clustered",
            cells=["fold_evaluated_property"],
            limits=["The cluster-covariate cell is red (F28)."],
        ),
        "weighting",
        {"fold_evaluated": "none"},
    )
    rows += rows_over(
        point_row(
            "T11",
            ["clustered-few-cluster-tmle"],
            families=("P1",),
            dependence="t",
            kinds=_WALD_T,
            cells=["few_cluster_reference"],
            limits=["The informative-size cells at 10 to 30 clusters include red cells (F28)."],
        ),
        "splitting",
        {
            "in_sample": ("in_sample",),
            "stacked": ("stacked",),
            "fold_evaluated": ("fold_evaluated",),
        },
    )
    rows.append(
        point_row(
            "T13",
            ["canonical-stratified-tmle"],
            families=("P1", "P3", "P4"),
            qualifiers=["strata"],
            cells=["stratified_binary"],
            limits=["Two truth cells at the stratum boundary are red."],
        )
    )
    rows.append(
        point_row(
            "T14",
            ["canonical-stratified-tmle"],
            families=("P1",),
            splitting=("stacked",),
            qualifiers=["strata"],
            cells=["stratified_binary"],
        )
    )
    rows.append(
        point_row(
            "T15",
            ["canonical-stratified-incremental-msm"],
            families=("P9",),
            qualifiers=["strata"],
            cells=["stratified_incremental"],
        )
    )
    rows += rows_over(
        point_row(
            "T16",
            ["canonical-stratified-incremental-msm"],
            families=("P10",),
            qualifiers=["strata"],
            cells=["stratified_l1_logit"],
            limits=["The logit slope at the smallest stratum is red (X8)."],
        ),
        "treatment",
        {"logit_binary": ("binary",), "dose_logit": ("continuous",)},
    )
    rows.append(
        point_row(
            "T16.dose_identity",
            ["canonical-stratified-incremental-msm"],
            families=("P10",),
            outcome=CONTINUOUS,
            treatment=("continuous",),
            qualifiers=["strata"],
            cells=["stratified_l2_identity"],
        )
    )
    rows.append(
        point_row(
            "T16.natural_course",
            ["canonical-stratified-incremental-msm"],
            families=("P4",),
            missing=("outcome",),
            splitting=("stacked",),
            qualifiers=["strata"],
            cells=["stratified_natural_course"],
        )
    )
    rows.append(
        point_row(
            "T17",
            ["canonical-stratified-msm-identity"],
            families=("P10",),
            outcome=CONTINUOUS,
            qualifiers=["strata"],
            cells=["stratified_msm_identity"],
            limits=["One paired cell at the smallest stratum is red (X8)."],
        )
    )
    rows.append(
        point_row(
            "T18",
            ["point-msm"],
            families=("P10",),
            outcome=CONTINUOUS,
            cells=["bounded_continuous_projection"],
        )
    )
    rows.append(
        point_row("T19", ["deterministic-regimes"], families=("P5",), cells=["binary_dynamic_rule"])
    )
    rows.append(
        point_row(
            "T20", ["stochastic-regimes"], families=("P6",), cells=["binary_known_stochastic"]
        )
    )
    rows.append(
        point_row(
            "T21",
            ["shift-policies"],
            families=("P7",),
            outcome=CONTINUOUS,
            treatment=("continuous",),
            cells=["continuous_modified_policy"],
            limits=["One paired calibration leg is red (RM18)."],
        )
    )
    rows.append(
        point_row(
            "T22",
            ["policy-point-mtp"],
            families=("P7",),
            treatment=("continuous",),
            cells=["policy_point"],
            limits=["One calibration cell is red."],
        )
    )
    rows.append(
        point_row(
            "T23",
            ["incremental-interventions"],
            families=("P9",),
            cells=["binary_incremental_odds"],
        )
    )
    rows.append(
        point_row(
            "T24",
            ["learned-rule-cvtmle"],
            families=("P11",),
            splitting=("fold_evaluated",),
            cells=["non_exceptional", "misspecified_limit"],
            limits=["The exceptional and weak-blip laws are red (F27)."],
        )
    )
    rows.append(
        point_row(
            "T25",
            ["mar-tmle"],
            families=("P1",),
            missing=("outcome",),
            cells=["binary_mar_observational"],
        )
    )
    for suffix, outcome, treatment, families, cell in (
        ("binary", BINARY, ("binary",), ("P1", "P2"), "binary_mar_two_arm_stacked"),
        ("continuous", BOUNDED, ("binary",), ("P1",), "continuous_mar_two_arm_stacked"),
        ("three_arm", BINARY, ("arms3",), ("P1", "P2"), "binary_mar_three_arm_stacked"),
        ("three_arm_continuous", BOUNDED, ("arms3",), ("P1",), "continuous_mar_three_arm_stacked"),
    ):
        rows.append(
            point_row(
                f"T26.{suffix}",
                ["stacked-mar-arm-indexed-cvtmle"],
                families=families,
                outcome=outcome,
                missing=("outcome",),
                treatment=treatment,
                splitting=("stacked",),
                kinds=frozenset({"point", "wald", "band"}),
                cells=[cell],
            )
        )
    rows += rows_over(
        point_row(
            "T27",
            ["mar-natural-course-tmle"],
            families=("P4",),
            missing=("outcome",),
            cells=["binary_mar_natural_course", "continuous_mar_natural_course"],
        ),
        "outcome",
        {"binary": BINARY, "continuous": BOUNDED},
    )
    rows.append(
        point_row(
            "T28",
            ["stacked-mar-natural-course-cvtmle"],
            families=("P4",),
            missing=("outcome",),
            splitting=("stacked",),
            cells=["binary_mar_natural_course_flexible"],
        )
    )
    attributable = ("P1", "P4")
    for suffix, kwargs, cell in (
        ("binary", {}, "binary_mar_attributable"),
        ("continuous", {"outcome": BOUNDED}, "continuous_mar_attributable"),
        ("three_arm", {"treatment": ("arms3",)}, "three_arm_mar_attributable"),
        ("binary_stacked", {"splitting": ("stacked",)}, "binary_mar_attributable_cvtmle"),
        (
            "three_arm_stacked",
            {"treatment": ("arms3",), "splitting": ("stacked",)},
            "three_arm_mar_attributable_cvtmle",
        ),
        ("weighted", {"weighting": "fixed"}, "binary_mar_attributable_weighted"),
        ("clustered", {"dependence": "clustered"}, "binary_mar_attributable_clustered"),
    ):
        rows.append(
            point_row(
                f"T29.{suffix}",
                ["mar-attributable-tmle"],
                families=attributable,
                missing=("outcome",),
                kinds=frozenset({"point", "wald", "band"}),
                cells=[cell],
                **kwargs,
            )
        )
    rows.append(
        point_row(
            "T30",
            ["cde-tmle"],
            families=("P1", "P2"),
            missing=("outcome", "intermediate"),
            cells=["binary_cde_z0_mar", "binary_cde_z1_mar"],
            limits=["The simultaneous band at z = 0 is red."],
        )
    )
    rows.append(
        point_row(
            "T31",
            ["known-treatment-mechanism"],
            families=("P1", "P2", "P3"),
            qualifiers=["known_mechanism"],
            kinds=frozenset({"point", "wald", "bootstrap"}),
            cells=["binary_q_correct", "binary_q_wrong", "ate__known_g__q_wrong__percentile"],
        )
    )
    rows.append(
        point_row(
            "T31.incremental",
            ["known-treatment-mechanism"],
            families=("P9",),
            qualifiers=["known_mechanism"],
            cells=["known_g__incremental"],
        )
    )
    rows.append(
        point_row(
            "T32",
            ["known-treatment-mechanism"],
            families=("P1",),
            splitting=("stacked",),
            qualifiers=["known_mechanism"],
            cells=["known_g__cvtmle"],
        )
    )
    return _merged(
        rows,
        {
            "T10.unweighted": ["T09", "T10.unweighted"],
            "T25": ["T25", "T27.binary", "T29.binary"],
            "T26.binary": ["T26.binary", "T28", "T29.binary_stacked"],
            "T26.three_arm": ["T26.three_arm", "T29.three_arm_stacked"],
            "T27.continuous": ["T27.continuous", "T29.continuous"],
        },
    )


def _d_rows() -> tuple[SupportRow, ...]:
    """DR-TMLE (``D`` rows)."""
    dr = ("drtmle",)
    rows: list[SupportRow] = [
        point_row(
            "D01",
            ["canonical-drtmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            cells=["both_correct", "outcome_correct", "treatment_correct"],
            limits=["The one-inconsistent-nuisance cells are red (RM18)."],
        ),
        point_row(
            "D02",
            ["canonical-multi-arm-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            treatment=("arms3",),
            splitting=("stacked",),
            cells=["multi_arm_binary_drtmle"],
            limits=["The boundary and one-sided-bias cells are red (RM18)."],
        ),
        point_row(
            "D03",
            ["canonical-stratified-drtmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            qualifiers=["strata"],
            cells=["stratified_both_correct"],
            limits=["Two truth cells and the small-stratum cell are red (X8)."],
        ),
        point_row(
            "D04",
            ["clustered-unequal-cvtmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            dependence="clustered",
            cells=["unequal_informative"],
        ),
        point_row(
            "D05",
            ["clustered-few-cluster-tmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            dependence="t",
            kinds=_WALD_T,
            cells=["few_cluster_reference"],
        ),
    ]
    for suffix, guard, treatment in (
        ("none", (), ("binary",)),
        ("q", ("Q",), ("binary",)),
        ("g", ("g",), ("binary",)),
        ("qg", ("Q", "g"), ("binary",)),
        ("none_three_arm", (), ("arms3",)),
    ):
        rows.append(
            point_row(
                f"D06.{suffix}",
                ["known-treatment-mechanism-drtmle"],
                families=("P1",),
                engine=dr,
                treatment=treatment,
                qualifiers=["known_mechanism"],
                options=_guard(*guard),
                cells=[f"guard_{suffix}"],
            )
        )
    rows += [
        point_row(
            "D07.estimated_g",
            ["drtmle-one-wrong"],
            families=("P1",),
            engine=dr,
            cells=["q_wrong_g_glm"],
        ),
        point_row(
            "D07.known_g",
            ["drtmle-one-wrong"],
            families=("P1",),
            engine=dr,
            qualifiers=["known_mechanism"],
            cells=["q_wrong_g_known"],
        ),
        point_row(
            "D07.estimated_g_stacked",
            ["drtmle-one-wrong"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            cells=["q_wrong_g_glm_cf"],
        ),
        point_row(
            "D08",
            ["mar-drtmle", "drtmle-one-wrong"],
            families=("P1",),
            engine=dr,
            missing=("outcome",),
            qualifiers=["known_mechanism"],
            options={"randomized": True},
            kinds=frozenset({"point", "wald", "band"}),
            cells=["binary_mar_randomized", "mar_q_wrong_pi_slow"],
        ),
        point_row(
            "D09",
            ["multi-arm-mar-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            missing=("outcome",),
            treatment=("arms3",),
            qualifiers=["known_mechanism"],
            options={"randomized": True},
            kinds=frozenset({"point", "wald", "band"}),
            cells=["three_arm_mar_randomized"],
            limits=["The F4 calibration-draw cell is red."],
        ),
        point_row(
            "D10.outcome",
            ["composite-missing-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            missing=("outcome",),
            cells=["binary_observational_mar"],
        ),
        point_row(
            "D10.outcome_treatment",
            ["composite-missing-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            missing=("outcome", "treatment"),
            cells=["binary_mar_outcome_and_treatment"],
        ),
        point_row(
            "D10.three_arm",
            ["composite-missing-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            missing=("outcome", "treatment"),
            treatment=("arms3",),
            cells=["three_arm_mar_outcome_and_treatment"],
            limits=["The high-arm truth row is red (composite-high-arm)."],
        ),
        point_row(
            "D10.tmle_outcome_treatment",
            ["composite-missing-drtmle"],
            families=("P1", "P2"),
            missing=("outcome", "treatment"),
            cells=["binary_mar_outcome_and_treatment"],
        ),
        point_row(
            "D10.tmle_three_arm",
            ["composite-missing-drtmle"],
            families=("P1", "P2"),
            missing=("outcome", "treatment"),
            treatment=("arms3",),
            cells=["three_arm_mar_outcome_and_treatment"],
        ),
    ]
    return _merged(
        rows, {"D01": ["D01", "D07.estimated_g_stacked"], "D06.qg": ["D06.qg", "D07.known_g"]}
    )


def _c_rows() -> tuple[SupportRow, ...]:
    """C-TMLE (``C`` rows). Every path withholds the interval, so each row licenses the point."""
    rows: list[SupportRow] = []
    for strategy in ("greedy", "ordered", "discrete"):
        rows.append(
            point_row(
                f"C01.{strategy}",
                ["canonical-ctmle-selector"],
                families=("P1",),
                engine=_ctmle(strategy),
                kinds=_POINT,
                cells=[f"binary_{strategy}"],
                limits=["The selector plug-in interval is withheld (F18)."],
            )
        )
        rows.append(
            point_row(
                f"C02.{strategy}",
                ["canonical-multi-arm-ctmle-selector"],
                families=("P1", "P2"),
                engine=_ctmle(strategy),
                treatment=("arms3",),
                kinds=_POINT,
                cells=[f"multi_arm_selector_{strategy}"],
                limits=["The selector plug-in interval is withheld (F18)."],
            )
        )
    shared = _ctmle("oat", "shared")
    per_arm = _ctmle("oat", "per_arm")
    rows += [
        point_row(
            "C03",
            ["canonical-ctmle-oat"],
            families=("P1", "P2"),
            engine=shared,
            kinds=_POINT,
            cells=["binary"],
        ),
        point_row(
            "C03.stacked",
            ["canonical-ctmle-oat"],
            families=("P1",),
            engine=shared,
            outcome=BOUNDED,
            splitting=("stacked",),
            kinds=_POINT,
            cells=["crossfit_overfitting"],
        ),
        point_row(
            "C04",
            ["canonical-multi-arm-ctmle-oat"],
            families=("P1", "P2"),
            engine=shared,
            treatment=("arms3",),
            kinds=_POINT,
            cells=["multi_arm_binary_oat"],
            limits=["The generated-design interval is withheld (F19)."],
        ),
        point_row(
            "C05.binary",
            ["ctmle-oat-per-arm"],
            families=("P1", "P2", "P4"),
            engine=per_arm,
            kinds=_POINT,
            cells=["binary_active"],
        ),
        point_row(
            "C05.three_arm",
            ["ctmle-oat-per-arm"],
            families=("P1", "P2"),
            engine=per_arm,
            treatment=("arms3",),
            kinds=_POINT,
            cells=["three_arm_active"],
        ),
        point_row(
            "C05.binary_stacked",
            ["ctmle-oat-per-arm"],
            families=("P1",),
            engine=per_arm,
            splitting=("stacked",),
            kinds=_POINT,
            cells=["interval_calibration"],
        ),
        point_row(
            "C05.three_arm_stacked",
            ["ctmle-oat-per-arm"],
            families=("P1",),
            engine=per_arm,
            treatment=("arms3",),
            splitting=("stacked",),
            kinds=_POINT,
            cells=["simultaneous_coverage"],
        ),
        point_row(
            "C05.weighted_stacked",
            ["ctmle-oat-per-arm"],
            families=("P1",),
            engine=per_arm,
            weighting="fixed",
            splitting=("stacked",),
            kinds=_POINT,
            cells=["binary_weighted__correctly_specified"],
        ),
        point_row(
            "C05.repeated",
            ["ctmle-oat-per-arm"],
            families=("P1",),
            engine=per_arm,
            splitting=("stacked", "repeated"),
            kinds=_POINT,
            cells=["binary_repeats__correctly_specified"],
        ),
        point_row(
            "C06.strata",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1",),
            engine=per_arm,
            qualifiers=["strata"],
            kinds=_POINT,
            cells=["strata"],
        ),
        point_row(
            "C06.missing_outcome",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1", "P2"),
            engine=per_arm,
            missing=("outcome",),
            kinds=_POINT,
            cells=["missing_outcome"],
        ),
        point_row(
            "C06.three_arm_missing",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1", "P2"),
            engine=per_arm,
            missing=("outcome",),
            treatment=("arms3",),
            kinds=_POINT,
            cells=["three_arm_missing"],
        ),
        point_row(
            "C06.missing_treatment",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1", "P2"),
            engine=per_arm,
            missing=("outcome", "treatment"),
            kinds=_POINT,
            cells=["missing_treatment"],
        ),
    ]
    return tuple(rows)


#: The point-treatment rows: ordinary TMLE (``T``), DR-TMLE (``D``) and C-TMLE (``C``).
POINT_ROWS: Final[tuple[SupportRow, ...]] = (*_t_rows(), *_d_rows(), *_c_rows())
