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
from dataclasses import dataclass, replace
from typing import Any, Final

from ._support import KEY_AXES, Kind, SupportKey, SupportRow
from .learners.glm import GLM

__all__ = [
    "EXEMPT_STUDIES",
    "FAMILY_THEORY",
    "LONGITUDINAL_ROWS",
    "MEASURED_LIMITS",
    "OPTION_ALLOWLIST",
    "POINT_ROWS",
    "SHORTFALLS",
    "TheoryAnchor",
    "default_options",
    "long_row",
    "merge_rows",
    "point_row",
    "rows_over",
]

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
    outcome : tuple of str, default BINARY
        The outcome axis of :class:`~cleverly._support.SupportKey`.
    missing : tuple of str, default ("complete",)
        The missing-data axis.
    treatment : tuple of str, default ("binary",)
        The treatment axis.
    weighting : str, default "none"
        The weighting axis.
    splitting : tuple of str, default ("in_sample",)
        The splitting axis.
    dependence : str, default "iid"
        The dependence axis.
    qualifiers : iterable of str, default ()
        ``strata`` and ``known_mechanism``.
    options : mapping of str to Any or None, default None
        Overrides of :func:`default_options`.
    engine : tuple of str, default ("tmle",)
        The engine axis.
    kinds : frozenset of str, default Wald
        What the row licenses. A row with a Wald kind that covers ``P1`` and ``P2`` also
        licenses ``derived``: a difference or a ratio of two arm means is the ``P1`` or ``P2``
        estimand that the row's study measured.
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
    if {"P1", "P2"} <= key.family and kinds & {"wald", "wald_t"}:
        kinds = kinds | {"derived"}
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

    families = frozenset().union(*(row.key.family for row in rows))
    kinds = frozenset().union(*(row.kinds for row in rows))
    # A kind that only some rows list covers the families of those rows alone.
    kind_families = []
    for kind in sorted(kinds):
        covered: frozenset[str] = frozenset().union(
            *(
                dict(row.kind_families).get(kind, row.key.family)
                for row in rows
                if kind in row.kinds
            )
        )
        if covered != families:
            kind_families.append((kind, covered))
    return SupportRow(
        row_id,
        replace(first.key, family=families),
        kinds,
        joined("evidence"),
        joined("cells"),
        None,
        joined("limits"),
        kind_families=tuple(kind_families),
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


_PLUGIN_STUDY = "omitted-variable-bound-plugin-se"
_PLUGIN_LIMIT = (
    "The cells calibrate the interval of the plug-in omitted-variable limits, for a treatment "
    "mechanism that is a LogisticRegression or is declared known (X26). They do not calibrate "
    "the Wald interval of the effect."
)


def _plugin_cells(scenario: str, parameters: Sequence[str]) -> list[str]:
    """The positive calibration cells of one scenario of ``omitted-variable-bound-plugin-se``."""
    return [
        f"interval_calibration/{scenario}_{parameter}_{bound}__correctly_specified"
        for parameter in parameters
        for bound in ("lower", "upper")
    ]


def _plugin_limit_rows() -> list[SupportRow]:
    """The rows of the plug-in omitted-variable limit study (X26), one per key it fits.

    The study fits an ordinary TMLE with a ``LinearRegression`` outcome and the ATE, the ATT and
    the ATC, in sample and cross-fitted, at two arms and at three. The treatment mechanism is a
    ``LogisticRegression`` or a declared known mechanism. The ridge, the unbalanced and the
    cross-fitted ridge scenarios share the key of a balanced logistic scenario. The two-arm
    in-sample and cross-fitted keys therefore add the study to ``T01.continuous`` and
    ``T02.continuous`` (:func:`_plugin_evidence`), and the three keys that no earlier row covers
    are the rows below. The key reads neither the learner class, the penalty nor the truncation
    bounds of the treatment mechanism, so ``_PLUGIN_LIMIT`` states what the cells cover.
    """
    families = ("P1", "P3")
    three = ("ate_2v0", "att_2v0", "atc_1v0")
    return [
        point_row(
            "T36.three_arm",
            [_PLUGIN_STUDY],
            families=families,
            outcome=CONTINUOUS,
            treatment=("arms3",),
            cells=_plugin_cells("multinomial_logistic", three),
            limits=[_PLUGIN_LIMIT],
        ),
        point_row(
            "T36.three_arm_stacked",
            [_PLUGIN_STUDY],
            families=families,
            outcome=BOUNDED,
            treatment=("arms3",),
            splitting=("stacked",),
            cells=_plugin_cells("multinomial_logistic_crossfit", three),
            limits=[_PLUGIN_LIMIT],
        ),
        point_row(
            "T36.known",
            [_PLUGIN_STUDY],
            families=families,
            outcome=CONTINUOUS,
            qualifiers=["known_mechanism"],
            cells=_plugin_cells("binary_known", ("ate", "att", "atc")),
            limits=[_PLUGIN_LIMIT],
        ),
    ]


def _plugin_evidence(rows: Iterable[SupportRow]) -> list[SupportRow]:
    """Add ``omitted-variable-bound-plugin-se`` to the two-arm rows that its fits land on."""
    scenarios = {
        "T01.continuous": ("binary_logistic", "binary_logistic_ridge", "binary_unbalanced"),
        "T02.continuous": ("binary_logistic_crossfit", "binary_logistic_ridge_crossfit"),
    }
    out = []
    for row in rows:
        if row.id in scenarios:
            cells = [
                cell
                for scenario in scenarios[row.id]
                for cell in _plugin_cells(scenario, ("ate", "att", "atc"))
            ]
            row = replace(row, evidence=(*row.evidence, _PLUGIN_STUDY), cells=(*row.cells, *cells))
        out.append(row)
    return out


def _t_rows() -> tuple[SupportRow, ...]:
    """Ordinary TMLE (``T`` rows)."""
    boot: frozenset[Kind] = frozenset({"point", "wald", "bootstrap", "derived"})
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
        # The bootstrap cell fits the ATE, which is the P1 family. A risk ratio, an odds ratio,
        # an ATT or ATC and a PAR or PAF fit read the bootstrap as experimental.
        kind_families=(("bootstrap", frozenset({"P1"})),),
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
    # The default-band study fits ``TMLE()`` with its default splitting on the binary law, and
    # reads the band that fit publishes.
    rows[-2] = replace(
        rows[-2],
        evidence=(*rows[-2].evidence, "default-simultaneous-bands"),
        kinds=rows[-2].kinds | {"band"},
        cells=(*rows[-2].cells, "default_fit"),
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
            options={"ratio": "density"},
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
            options={"ratio": "density"},
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
    # X31: joint policies at a vector treatment. A vector with a dose keys as ``vector``; an
    # all-categorical vector is a categorical treatment over its observed tuples (four here).
    for suffix, cells in (
        ("density", ["point_vector_mixed", "point_vector_density__correctly_specified"]),
        ("classifier", ["point_vector_classifier__correctly_specified"]),
    ):
        rows.append(
            point_row(
                f"T33.{suffix}",
                ["vector-mtp"],
                families=("P7",),
                treatment=("vector",),
                options={"ratio": suffix},
                cells=cells,
            )
        )
    rows.append(
        point_row(
            "T34",
            ["vector-mtp"],
            families=("P7",),
            treatment=("arms4",),
            options={"ratio": "density"},
            cells=["point_vector_categorical"],
        )
    )
    rows += _plugin_limit_rows()
    rows = _plugin_evidence(rows)
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
            families=("P1", "P2", "P4"),
            engine=per_arm,
            missing=("outcome",),
            kinds=_POINT,
            cells=["missing_outcome", "natural_course_insample"],
        ),
        point_row(
            "C06.missing_outcome_stacked",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1", "P2", "P4"),
            engine=per_arm,
            missing=("outcome",),
            splitting=("stacked",),
            kinds=_POINT,
            cells=["missing_outcome_crossfit"],
        ),
        point_row(
            "C06.strata_missing_stacked",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1", "P4"),
            engine=per_arm,
            missing=("outcome",),
            splitting=("stacked",),
            qualifiers=["strata"],
            kinds=_POINT,
            cells=["strata_missing"],
        ),
        point_row(
            "C06.weighted_strata_missing_stacked",
            ["ctmle-oat-per-arm-strata-missing"],
            families=("P1",),
            engine=per_arm,
            missing=("outcome",),
            weighting="fixed",
            splitting=("stacked",),
            qualifiers=["strata"],
            kinds=_POINT,
            cells=["weighted_strata_missing"],
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
    derivative = _ctmle("derivative")
    study = "ctmle-derivative"
    rows += [
        point_row(
            "C10.in_sample",
            [study],
            families=("P1",),
            engine=derivative,
            cells=["in_sample"],
        ),
        point_row(
            "C10.stacked",
            [study],
            families=("P1",),
            engine=derivative,
            splitting=("stacked",),
            cells=["cross_fitted"],
        ),
        point_row(
            "C10.multi_arm",
            [study],
            families=("P1", "P2"),
            engine=derivative,
            treatment=("arms3",),
            cells=["multi_arm"],
        ),
        point_row(
            "C10.missing_outcome",
            [study],
            families=("P1",),
            engine=derivative,
            missing=("outcome",),
            cells=["missing_outcome"],
        ),
        point_row(
            "C10.clustered",
            [study],
            families=("P1",),
            engine=derivative,
            splitting=("stacked",),
            dependence="clustered",
            cells=["cluster_robust"],
        ),
    ]
    stack = (*_ctmle("oat", "per_arm"), "parametric_stack")
    parametric = "ctmle-oat-per-arm-parametric"
    extensions = "ctmle-oat-per-arm-parametric-extensions"
    # The binary in-sample row also carries the population targets and the band of X58 part (b):
    # the study ctmle-oat-per-arm-parametric-extensions gates ey_obs, par and paf, and the band over
    # the arm means, the ATE and those targets, on the same law.
    rows += [
        replace(
            point_row(
                "C11.binary",
                [parametric, extensions],
                families=("P1", "P2", "P4"),
                engine=stack,
                kinds=_WALD | {"band"},
                cells=[
                    "binary_active__ate__corrected",
                    "binary_population__par__corrected",
                    "binary_population__paf__corrected",
                    "binary_population__ey_obs__corrected",
                    "binary_population__simultaneous_band",
                ],
                limits=[
                    "The band covers the arm means, the ATE and the population targets (cell "
                    "binary_population). A band over rr or or (family P2) is experimental.",
                    "The study measures the band over ey0, ey1, ate, ey_obs, par and paf. A band "
                    "over another set of these families rests on the same multiplier argument "
                    "(plan 13.2).",
                    "The curve of ey_obs is the shipped empirical curve. The cell "
                    "binary_population__ey_obs__corrected measures its interval.",
                ],
            ),
            kind_families=(("band", frozenset({"P1", "P4"})),),
        ),
        point_row(
            "C11.bounded",
            [parametric],
            families=("P1",),
            engine=stack,
            outcome=BOUNDED,
            cells=["bounded_outcome__ate__corrected"],
        ),
    ]
    # The cross-fitted stack (X58 part (c)). The study fits a binary outcome at n_folds=5, at
    # one draw and at repeats=3. The key holds neither the fold count nor the repeat count, so
    # the CTMLE preflight adds a group-A condition for another count, and that fit is
    # experimental although it matches a row here.
    crossfit = "ctmle-oat-per-arm-parametric-crossfit"
    measured = (
        "The study ctmle-oat-per-arm-parametric-crossfit measures n_folds=5 only. A fit at "
        "another fold count inside the cap of 20 folds is experimental."
    )
    rows += [
        point_row(
            "C11.stacked",
            [crossfit],
            families=("P1", "P2"),
            engine=stack,
            splitting=("stacked",),
            cells=[
                *(
                    f"crossfit_1500__{estimand}__corrected"
                    for estimand in ("ey0", "ey1", "ate", "log_rr", "log_or")
                ),
                *(
                    f"{group}__{estimand}__corrected"
                    for group in ("crossfit_500", "crossfit_8000", "crossfit_pseudo_true")
                    for estimand in ("ey0", "ey1", "ate")
                ),
                "binary_active__estimated",
                "sharp_null",
            ],
            limits=[
                measured,
                "The interval needs at least 30 rows in each held-out fold.",
                "Under a wrong outcome model the interval covers the limit psi*, not the arm "
                "mean (cells crossfit_pseudo_true).",
            ],
        ),
        point_row(
            "C11.repeated",
            [crossfit],
            families=("P1",),
            engine=stack,
            splitting=("stacked", "repeated"),
            cells=[f"repeats__{estimand}__corrected" for estimand in ("ey0", "ey1", "ate")],
            limits=[
                measured,
                "The study measures repeats=3 only. A fit at another repeat count is experimental.",
                "The repeats cells cannot see the split-adjustment term of the median rule, about "
                "0.2% of the variance. A unit identity test pins that term.",
            ],
        ),
    ]
    # The K-arm, weighted and population-target stack (X58 part (b) and (d), X58b2). The study
    # fits three arms, a known observation weight, and ey_obs, par and paf, in sample and
    # cross-fitted at n_folds=5. The key holds neither the arm count beyond the axis nor the fold
    # count, so the CTMLE preflight adds the group-A condition for another fold count, and a
    # fit at four arms or with estimated weights matches no row.
    folds = (
        "The study ctmle-oat-per-arm-parametric-extensions measures n_folds=5 only. A fit at "
        "another fold count inside the cap of 20 folds is experimental."
    )
    three_arms = (
        "The study measures three arms only. A fit at four arms matches no row and is experimental."
    )
    known_weight = (
        "The study measures a known observation weight only. A fit with estimated weights is "
        "experimental."
    )
    rows += [
        replace(
            point_row(
                "C11.k_arms",
                [extensions],
                families=("P1", "P2", "P4"),
                engine=stack,
                treatment=("arms3",),
                kinds=_WALD | {"band"},
                cells=[
                    *(
                        f"three_arm__{estimand}__corrected"
                        for estimand in (
                            "ey_0",
                            "ey_1",
                            "ey_2",
                            "ate_0_vs_1",
                            "ate_2_vs_1",
                            "log_rr_0_vs_1",
                            "log_rr_2_vs_1",
                            "log_or_0_vs_1",
                            "log_or_2_vs_1",
                            "par_1",
                            "paf_1",
                            "ey_obs",
                        )
                    ),
                    "three_arm__simultaneous_band",
                    *(
                        f"three_arm_shared__{estimand}__corrected"
                        for estimand in ("ey_0", "ey_1", "ey_2", "ate_0_vs_1", "ate_2_vs_1")
                    ),
                    "three_arm_null__ate_0_vs_1",
                    "three_arm_null__ate_2_vs_1",
                ],
                limits=[
                    three_arms,
                    "The band covers the three arm means and the two contrasts against the "
                    "reference arm (cell three_arm). A band over rr or or (family P2) or over a "
                    "population target (family P4) is experimental.",
                    "The study measures the band over ey[0], ey[1], ey[2] and the two ate "
                    "contrasts against the reference arm. A band over another set of arm means and "
                    "ate contrasts rests on the same multiplier argument (plan 13.2).",
                    "The study fits the reference arm 1 in sample. The arm curves of the law "
                    "three_arm_active are nearly uncorrelated, and the law three_arm_shared "
                    "(correlation 0.25 to 0.26) measures the cross-arm covariance.",
                ],
            ),
            kind_families=(("band", frozenset({"P1"})),),
        ),
        point_row(
            "C11.weighted",
            [extensions],
            families=("P1", "P2", "P4"),
            engine=stack,
            weighting="fixed",
            cells=[
                f"weighted__{estimand}__corrected"
                for estimand in ("ey0", "ey1", "ate", "log_rr", "log_or", "par", "paf", "ey_obs")
            ],
            limits=[
                known_weight,
                "The curve of ey_obs is the shipped weighted empirical curve. The cell "
                "weighted__ey_obs__corrected measures its interval.",
            ],
        ),
        point_row(
            "C11.weighted_stacked",
            [extensions],
            families=("P1",),
            engine=stack,
            weighting="fixed",
            splitting=("stacked",),
            cells=[
                f"weighted_crossfit__{estimand}__corrected" for estimand in ("ey0", "ey1", "ate")
            ],
            limits=[
                folds,
                known_weight,
                "The study measures one cross-fitting draw. A fit with repeats and weights "
                "matches no row and is experimental.",
            ],
        ),
        point_row(
            "C11.stacked_k_arms",
            [extensions],
            families=("P1",),
            engine=stack,
            treatment=("arms3",),
            splitting=("stacked",),
            kinds=_WALD | {"band"},
            cells=[
                *(
                    f"three_arm_crossfit__{estimand}__corrected"
                    for estimand in ("ey_0", "ey_1", "ey_2", "ate_1_vs_0", "ate_2_vs_0")
                ),
                "three_arm_crossfit__simultaneous_band",
                "three_arm_active__estimated__ate_1_vs_0",
                "three_arm_active__estimated__ate_2_vs_0",
            ],
            limits=[
                folds,
                three_arms,
                "The study measures one cross-fitting draw and the reference arm 0. A fit with "
                "repeats at three arms matches no row and is experimental.",
                "The study measures the band over ey[0], ey[1], ey[2] and the two ate contrasts "
                "against the reference arm. A band over another set of arm means and ate "
                "contrasts rests on the same multiplier argument (plan 13.2).",
            ],
        ),
        point_row(
            "C11.stacked_k_arms_weighted",
            [extensions],
            families=("P1",),
            engine=stack,
            treatment=("arms3",),
            weighting="fixed",
            splitting=("stacked",),
            cells=[
                f"composition__{estimand}__corrected"
                for estimand in ("ey_0", "ey_1", "ey_2", "ate_0_vs_1", "ate_2_vs_1")
            ],
            limits=[folds, three_arms, known_weight],
        ),
    ]
    rows += [
        point_row(
            f"C09.{strategy}",
            ["canonical-ctmle-sequences"],
            families=("P1",),
            engine=_ctmle(strategy),
            kinds=_POINT,
            cells=[f"binary_{strategy}"],
            limits=["The C-TMLE plug-in interval is withheld."],
        )
        for strategy in ("lasso", "learners")
    ]
    return tuple(rows)


#: The red cells that lie inside the configuration of one row, as ``(row id, kind) -> sentence``.
#: A fit that reads the kind from the row is experimental, and the sentence gives the measured
#: number (``tests/studies/evidence/support_dispositions.py`` holds the decision of each cell).
SHORTFALLS: Final[Mapping[tuple[str, str], str]] = {
    ("T10.fold_evaluated", "wald"): (
        "Row T10.fold_evaluated is red for the fold-evaluated cluster-covariate interval: "
        "coverage 0.9182 to 0.9450 and SE ratio 0.9453 to 1.0187 (F28)."
    ),
    ("T11.fold_evaluated", "wald_t"): (
        "Row T11.fold_evaluated is red when cluster sizes are informative at 10 to 30 clusters: "
        "at 10 clusters bias 0.1146 to 0.1310, coverage 0.8940 to 0.9180 and SE ratio "
        "0.8297 to 0.8849 (F28)."
    ),
    ("T16.logit_binary", "wald"): (
        "Row T16.logit_binary is red for the logit slope at the smallest stratum: coverage "
        "0.9289 to 0.9560 and SE ratio 0.7818 to 0.9387 (X8)."
    ),
    ("T16.dose_logit", "wald"): (
        "Row T16.dose_logit is red for the logit slope at the smallest stratum: coverage "
        "0.9289 to 0.9560 and SE ratio 0.7818 to 0.9387 (X8)."
    ),
    # X40: the owner chose this label for the four red price_down cells of the five-node price
    # study (2026-10-16). To flip it to a stated limit, move each sentence to MEASURED_LIMITS
    # under the same row id, and change the four dispositions of ``mtp-pricing`` in
    # ``tests/studies/evidence/support_dispositions.py`` (see ``_X40_PRICE_DOWN`` there).
    ("G39.ltmle.tau5", "wald"): (
        "Row G39.ltmle.tau5 is red for the price_down plan at five nodes and n = 5,000: "
        "coverage 0.9160 to 0.9170 and SE ratio 0.9310 to 0.9479 (mtp-pricing-calibration)."
    ),
    ("G39.sdr.tau5", "wald"): (
        "Row G39.sdr.tau5 is red for the price_down plan at five nodes and n = 5,000: "
        "coverage 0.9190 to 0.9230 and SE ratio 0.9339 to 0.9518 (mtp-pricing-calibration)."
    ),
}


#: The measured numbers of the red cells that leave a row supported with a stated limit, by row id
#: (``tests/studies/evidence/support_dispositions.py`` holds the decision of each cell). The
#: row note prints each sentence after the limits of the row itself.
MEASURED_LIMITS: Final[Mapping[str, tuple[str, ...]]] = {
    "T36.three_arm_stacked": (
        "X26-plugin-conservative: coverage 0.9556 to 0.9657, SE ratio 1.0332 to 1.0723, "
        "failures 32 of 10000 (complete case fails)",
        "X26-refusal-limited: coverage 0.9568 to 0.9667, SE ratio 1.0288 to 1.0683, "
        "failures 32 of 10000 (worst case fails)",
    ),
    "C02.greedy": ("RM18-boundary: bias 0.0010, coverage 0.8965 to 0.9627, SE ratio 0.9892",),
    "D01": ("RM18-boundary: coverage 0.8864 to 0.9385, bias 0.0115",),
    "D02": (
        "RM18-boundary: coverage 0.8988 to 0.9542, bias 0.000900",
        "RM18-boundary: coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123",
        "RM18-boundary: bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888",
    ),
    "D03": (
        "X8-drtmle-small-stratum: bias 0.0023 to 0.0112, coverage 0.9000, SE ratio 0.9388",
        (
            "X8-drtmle-small-stratum: coverage 0.9069 to 0.9380, SE ratio 0.9294 to 1.0112, "
            "empirical efficiency ratio 1.0348 to 1.1237, reported efficiency ratio 1.0296 to"
            " 1.0653"
        ),
    ),
    "D09": (
        (
            "F4-calibration-draws: coverage 0.9287 to 0.9537, SE ratio 0.9294 to 1.0000, "
            "empirical efficiency ratio 0.9994 to 1.0751, reported efficiency ratio 0.9980 to"
            " 1.0008"
        ),
        "F4-calibration-draws: joint coverage 0.9196 to 0.9462",
    ),
    "D10.outcome": (
        "composite-high-arm: bias -0.0054 to 0.0013, coverage 0.9225, SE ratio 0.9342",
    ),
    "G03": (
        (
            "RM18-ordinary-weighted: coverage 0.9187 to 0.9454, SE ratio 0.9263 to 0.9985, "
            "empirical efficiency ratio 0.9855 to 1.0615, reported efficiency ratio 0.9783 to"
            " 0.9899"
        ),
        "RM18-ordinary-weighted: bias -0.0025 to 0.0019, margin 0.0075",
        "RM18-ordinary-weighted: bias -0.0027 to 0.0075, margin 0.0172",
    ),
    "G04": (
        (
            "RM18-boundary: difference -0.000150 to 0.000202 within 0.0031, RMSE ratio bound "
            "1.0162 vs 1.1000, coverage difference bound -0.0475 vs -0.0250, calibration "
            "excess bound 0.1050 vs 0.0500, resolution 0.1121"
        ),
        (
            "RM18-fixed-weights: difference -0.000350 to 0.000263 within 0.0047, RMSE ratio "
            "bound 1.0245 vs 1.1000, coverage difference bound -0.0300 vs -0.0250, "
            "calibration excess bound 0.0465 vs 0.0500, resolution 0.0662"
        ),
        (
            "RM18-boundary: difference -0.000118 to 0.000400 within 0.0032, RMSE ratio bound "
            "1.0175 vs 1.1000, coverage difference bound -0.0425 vs -0.0250, calibration "
            "excess bound 0.0382 vs 0.0500, resolution 0.1119"
        ),
        (
            "RM18-fixed-weights: coverage 0.9232 to 0.9492, SE ratio 0.9430 to 1.0203, "
            "empirical efficiency ratio 1.0195 to 1.1006, reported efficiency ratio 1.0333 to"
            " 1.0442"
        ),
    ),
    "G05.cross_fitted": ("band-finite-sample: joint coverage 0.9191 to 0.9458",),
    "G10.continuous": (
        (
            "mtp-longitudinal-limits: coverage 0.9289 to 0.9560, SE ratio 0.9432 to 1.0214, "
            "empirical efficiency ratio 1.0746 to 1.1643, reported efficiency ratio 1.0948 to"
            " 1.1006"
        ),
        "mtp-pricing-learned-density: coverage 0.9530 to 0.9747, SE ratio 1.0265 to 1.1162",
    ),
    "G11": ("band-finite-sample: joint coverage 0.9140 to 0.9283",),
    "G14.survival": (
        (
            "X13-finite-sample: coverage 0.9421 to 0.9689, SE ratio 0.9807 to 1.0762, "
            "empirical efficiency ratio 0.9300 to 1.0204, reported efficiency ratio 1.0000 to"
            " 1.0015"
        ),
    ),
    "G17.survival": (
        (
            "X13-finite-sample: coverage 0.9421 to 0.9689, SE ratio 0.9650 to 1.0560, "
            "empirical efficiency ratio 1.0101 to 1.1085, reported efficiency ratio 1.0592 to"
            " 1.0764"
        ),
    ),
    "G23.in_sample": (
        (
            "X4-finite-sample: bias 0.000011 to 0.0112, margin 0.0172, coverage 0.8853 to "
            "0.9327, SE ratio 0.9459"
        ),
        "band-finite-sample: joint coverage 0.9085 to 0.9394",
    ),
    "G29.g1": ("X45-finite-sample: coverage 0.9278 to 0.9529, SE ratio 0.9259 to 1.0000",),
    "G40.tau3": (
        "mtp-pricing-learned-density: coverage 0.9473 to 0.9704, SE ratio 1.0493 to 1.1408",
    ),
    "G40.tau5": (
        "mtp-pricing-learned-density: coverage 0.9485 to 0.9713, SE ratio 1.0428 to 1.1323",
    ),
    "G34.competing": (
        (
            "X19-competing-calibration: coverage 0.9491 to 0.9681, SE ratio 1.0124 to 1.0830, "
            "empirical efficiency ratio 0.9239 to 0.9884, reported efficiency ratio 1.0005 to"
            " 1.0007"
        ),
    ),
    "T02.binary": (
        "band-finite-sample: joint coverage 0.9101 to 0.9382",
        "band-finite-sample: joint coverage 0.9173 to 0.9443",
        "band-finite-sample: joint coverage 0.9015 to 0.9309",
    ),
    "T13": (
        "strata-boundary-mean: bias -0.0027 to 0.000813, coverage 0.9170, SE ratio 0.9850",
        (
            "strata-boundary-mean: coverage 0.9056 to 0.9344, SE ratio 0.9315 to 1.0094, "
            "empirical efficiency ratio 0.9753 to 1.0552, reported efficiency ratio 0.9742 to"
            " 0.9931"
        ),
    ),
    "T17": (
        (
            "X8-identity-small-stratum: difference -0.000049 to 0.000037 within 0.000694, "
            "RMSE ratio bound 1.0047 vs 1.1000, coverage difference bound -0.0220 vs -0.0250,"
            " calibration excess bound 0.0523 vs 0.0500, resolution 0.0476"
        ),
        (
            "X8-identity-small-stratum: coverage 0.9233 to 0.9515, SE ratio 0.9264 to 1.0042,"
            " empirical efficiency ratio 0.9925 to 1.0766, reported efficiency ratio 0.9937 "
            "to 1.0005"
        ),
    ),
    "T21": (
        (
            "RM18-comparator-density: difference 0.000061 to 0.000361 within 0.000770, RMSE "
            "ratio bound 1.0587 vs 1.1000, coverage difference bound -0.0050 vs -0.0250, "
            "calibration excess bound 0.0648 vs 0.0500, resolution 0.0439"
        ),
    ),
    "T22": (
        (
            "mtp-point-calibration: coverage 0.9479 to 0.9709, SE ratio 1.0011 to 1.0881, "
            "empirical efficiency ratio 0.9208 to 1.0003, reported efficiency ratio 1.0003 to"
            " 1.0034"
        ),
    ),
}


def _with_shortfalls(rows: Iterable[SupportRow]) -> tuple[SupportRow, ...]:
    """Attach the :data:`SHORTFALLS` and :data:`MEASURED_LIMITS` sentences to their rows."""
    out = []
    for row in rows:
        pairs = tuple(
            (kind, text) for (row_id, kind), text in SHORTFALLS.items() if row_id == row.id
        )
        limits = tuple(dict.fromkeys((*row.limits, *MEASURED_LIMITS.get(row.id, ()))))
        out.append(replace(row, shortfalls=pairs, limits=limits))
    return tuple(out)


def _categorical_point_rows() -> tuple[SupportRow, ...]:
    """The point rows of the study ``categorical-incremental`` (X43): ``T35``.

    A categorical tilt fit on one of these configurations carries the evidence of the study.
    Any other categorical tilt fit stays labelled ``experimental``.
    """
    slug = ["categorical-incremental"]
    point: dict[str, Any] = {
        "families": ("P9",),
        "outcome": ("gaussian", "bounded"),
        "treatment": ("arms3",),
    }
    return (
        point_row("T35.in_sample", slug, **point, cells=["point_in_sample"]),
        point_row(
            "T35.cross_fitted",
            slug,
            **point,
            splitting=("stacked",),
            cells=[
                "point_crossfit",
                "point_contrast_t_a",
                "point_contrast_t_b",
                "point_contrast_t_a__full_eif",
                "point_contrast_t_a__sharp_null",
                "point_contrast_t_a__alternative",
            ],
        ),
        point_row(
            "T35.known",
            slug,
            **point,
            splitting=("stacked",),
            qualifiers=("known_mechanism",),
            cells=["point_known_t_a"],
        ),
        point_row(
            "T35.strata",
            slug,
            **point,
            splitting=("stacked",),
            qualifiers=("strata",),
            cells=["point_stratified_t_a_v1"],
        ),
    )


#: The point-treatment rows: ordinary TMLE (``T``), DR-TMLE (``D``), C-TMLE (``C``) and the
#: categorical tilt (``T35``).
POINT_ROWS: Final[tuple[SupportRow, ...]] = _with_shortfalls(
    (
        *_t_rows(),
        *_d_rows(),
        *_c_rows(),
        *_categorical_point_rows(),
    )
)


# --------------------------------------------------------------------------- longitudinal

_LONG_WALD: Final[frozenset[Kind]] = frozenset({"point", "wald"})
_LONG_BAND: Final[frozenset[Kind]] = frozenset({"point", "wald", "band"})
_LONG_T: Final[frozenset[Kind]] = frozenset({"point", "wald_t"})
_LONG_DERIVED: Final[frozenset[Kind]] = frozenset({"point", "wald", "band", "derived"})
_LONG_BOOTSTRAP: Final[frozenset[Kind]] = frozenset(
    {"point", "wald", "band", "bootstrap", "derived"}
)
_TWO: Final[tuple[str, ...]] = ("2", "2")
_OUTCOME_KINDS: Final[frozenset[str]] = frozenset(
    {"end_of_study", "survival", "competing", "repeated"}
)


def _x33_options(
    *, node_bounds: bool = False, hazard_pooling: str = "node"
) -> tuple[tuple[str, Any], ...]:
    """The fitting options of a longitudinal key that change the fit's arithmetic.

    No row declares ``ratio_trim=`` or a Markov order: no registered study fits either one,
    so a fit that declares one matches no row.
    """
    return (
        ("node_bounds", node_bounds),
        ("hazard_pooling", hazard_pooling),
        ("ratio_trim", False),
        ("markov_order", None),
    )


def long_row(
    row_id: str,
    evidence: Sequence[str],
    *,
    outcome_kind: str = "end_of_study",
    regimens: Iterable[str] = ("static",),
    msm: bool = False,
    held: bool = False,
    outcome: tuple[str, ...] = BINARY,
    missing: tuple[str, ...] = ("censoring",),
    treatment: tuple[str, ...] = _TWO,
    weighting: str = "none",
    splitting: tuple[str, ...] = ("in_sample",),
    dependence: str = "iid",
    qualifiers: Iterable[str] = (),
    engine: str = "tmle",
    normalization: str | None = None,
    survival_model: str = "iterated_mean",
    targeting: str = "iterative",
    isotonic: bool = False,
    ratio: str = "density",
    node_bounds: bool = False,
    hazard_pooling: str = "node",
    kinds: frozenset[Kind] = _LONG_WALD,
    cells: Sequence[str] = (),
    limits: Sequence[str] = (),
    learners: Mapping[str, type] | None = None,
) -> SupportRow:
    """One support row of a longitudinal estimator, with the defaults of a censored two-node fit.

    Parameters
    ----------
    row_id : str
        The row id.
    evidence : sequence of str
        The registered study slugs.
    outcome_kind : str, default "end_of_study"
        ``end_of_study``, ``survival``, ``competing`` or ``repeated``.
    regimens : iterable of str, default ("static",)
        The regimen kinds of :data:`cleverly._support.REGIMEN_KINDS` the row covers.
    msm : bool, default False
        Whether the row reports working-model coefficients.
    held : bool, default False
        Whether one baseline treatment is held over every node.
    outcome : tuple of str, default BINARY
        The outcome axis of :class:`~cleverly._support.SupportKey`.
    missing : tuple of str, default ("censoring",)
        The missing-data axis.
    treatment : tuple of str, default two nodes
        The treatment axis. A node count is its length.
    weighting : str, default "none"
        The weighting axis.
    splitting : tuple of str, default ("in_sample",)
        The splitting axis.
    dependence : str, default "iid"
        The dependence axis.
    qualifiers : iterable of str, default ()
        The known treatment and censoring factors.
    engine : str, default "tmle"
        ``tmle``, ``sdr``, ``ipsi``, ``gcomp`` or ``ipw``.
    normalization : str or None, default None
        The inverse-probability normalization of an ``ipw`` row.
    survival_model : {"iterated_mean", "hazard"}, default "iterated_mean"
        How a survival curve is targeted.
    targeting : str, default "iterative"
        The hazard targeting algorithm of a ``hazard`` row.
    isotonic : bool, default False
        Whether the row projects the curve onto a monotone one.
    ratio : {"density", "classifier"}, default "density"
        The policy density ratio of a modified treatment policy row.
    node_bounds : bool, default False
        Whether the row's fits declare ``node_bounds=``.
    hazard_pooling : {"node", "pooled"}, default "node"
        How the row's fits pool the censoring and hazard regressions over the nodes.
    kinds : frozenset of str, default Wald
        What the row licenses.
    cells : sequence of str, default ()
        The licensing study cells.
    limits : sequence of str, default ()
        The stated limits of the row.
    learners : mapping of str to type or None, default None
        The exact learner class per nuisance role, where the study fixes one.

    Returns
    -------
    SupportRow
        The row.
    """
    key = SupportKey(
        kind="longitudinal",
        engine=(engine,),
        family=frozenset(
            (
                outcome_kind,
                "msm" if msm else "regimens",
                *regimens,
            )
        ),
        outcome=outcome,
        missing=missing,
        treatment=("held", *treatment) if held else treatment,
        weighting=weighting,
        splitting=splitting,
        dependence=dependence,
        qualifiers=frozenset(qualifiers),
        options=(
            ("multiplier_kind", "rademacher"),
            ("normalization", normalization),
            ("ratio", ratio),
            ("survival_model", survival_model),
            ("targeting", targeting),
            ("isotonic", isotonic),
            *_x33_options(node_bounds=node_bounds, hazard_pooling=hazard_pooling),
        ),
    )
    return SupportRow(row_id, key, kinds, tuple(evidence), tuple(cells), learners, tuple(limits))


def _long_rows() -> tuple[SupportRow, ...]:
    """The longitudinal rows (``G``): ``LTMLE``, ``SDR`` and the sequential baselines."""
    xf = ("cross_fitted",)
    boot = "full-refit-bootstrap-and-derived-contrasts"
    rows: list[SupportRow] = [
        long_row(
            "G01",
            ["canonical-ltmle", boot],
            kinds=_LONG_BOOTSTRAP,
            cells=["interval_calibration", "simultaneous_coverage", "boot_ey_end_of_study"],
        ),
        long_row(
            "G02",
            ["canonical-ltmle-crossfit", boot],
            splitting=xf,
            kinds=_LONG_DERIVED,
            cells=["interval_calibration", "simultaneous_coverage"],
            limits=["The full-refit bootstrap reads an SE ratio of 1.17 to 1.33 here (X20)."],
        ),
        long_row(
            "G03",
            ["weighted-ltmle"],
            weighting="fixed",
            kinds=_LONG_BAND,
            cells=["interval_calibration"],
        ),
        long_row(
            "G04",
            ["weighted-ltmle-crossfit"],
            weighting="fixed",
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["interval_calibration"],
        ),
        long_row(
            "G05.cross_fitted",
            ["clustered-cross-fitted-ltmle"],
            splitting=xf,
            dependence="clustered",
            kinds=_LONG_BAND,
            cells=["interval_calibration", "simultaneous_coverage"],
        ),
        long_row(
            "G05.in_sample",
            [boot],
            dependence="clustered",
            kinds=_POINT,
            cells=["boot_ey_clustered"],
            limits=["Only the full-refit bootstrap measures this configuration (X20)."],
        ),
        long_row(
            "G06",
            ["clustered-few-cluster-tmle"],
            missing=("complete",),
            treatment=("2",),
            dependence="t",
            kinds=_LONG_T,
            cells=["interval_calibration"],
        ),
        long_row(
            "G07",
            ["few-cluster-cross-fitted-ltmle"],
            splitting=xf,
            dependence="t",
            kinds=_LONG_T,
            cells=["interval_calibration"],
        ),
    ]
    # Static and known stochastic categorical regimens share one key, so they are one row.
    for suffix, splitting in (("in_sample", ("in_sample",)), ("cross_fitted", xf)):
        categorical = long_row(
            f"G08.{suffix}",
            ["canonical-categorical-ltmle" + ("-crossfit" if suffix == "cross_fitted" else "")],
            missing=("complete",),
            treatment=("3", "3"),
            splitting=splitting,
            cells=["interval_calibration"],
        )
        policy = long_row(
            f"G09.{suffix}",
            ["stochastic-categorical-ltmle"],
            regimens=("static", "stochastic"),
            missing=("complete",),
            treatment=("3", "3"),
            splitting=splitting,
            kinds=_LONG_BAND if suffix == "in_sample" else _LONG_WALD,
            cells=["interval_calibration"],
        )
        rows.append(merge_rows(categorical.id, categorical, policy))
    # X40: the two-node continuous fit on the density route also stands for the declared supports,
    # branches and atoms of ``mtp-boundaries`` and the learned density of ``mtp-pricing`` at
    # two nodes. The key reads none of them, so they share this row.
    for suffix, treatment, splitting, studies in (
        (
            "continuous",
            ("continuous", "continuous"),
            ("in_sample",),
            ["longitudinal-mtp", "mtp-boundaries", "mtp-pricing"],
        ),
        ("continuous_cross_fitted", ("continuous", "continuous"), xf, ["longitudinal-mtp"]),
        ("categorical", ("6", "6"), ("in_sample",), ["longitudinal-mtp"]),
        ("rr_tilt", _TWO, ("in_sample",), ["longitudinal-mtp"]),
    ):
        rows.append(
            long_row(
                f"G10.{suffix}",
                studies,
                regimens=("mtp",),
                missing=("complete",),
                treatment=treatment,
                splitting=splitting,
                kinds=_LONG_BAND,
                cells=["interval_calibration"],
            )
        )
    rows += [
        long_row(
            "G11",
            ["canonical-ltmle-survival"],
            outcome_kind="survival",
            kinds=_LONG_BAND,
            cells=["interval_calibration", "simultaneous_coverage"],
        ),
        long_row(
            "G11.nodes4",
            [boot],
            outcome_kind="survival",
            treatment=("2",) * 4,
            kinds=_LONG_BOOTSTRAP,
            cells=["boot_risk2_survival", "boot_risk4_survival", "rmst_survival"],
        ),
        long_row(
            "G12",
            ["canonical-ltmle-survival-crossfit"],
            outcome_kind="survival",
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["interval_calibration"],
        ),
        long_row(
            "G12.nodes4",
            [boot],
            outcome_kind="survival",
            treatment=("2",) * 4,
            splitting=xf,
            kinds=_LONG_DERIVED,
            cells=["rmst_crossfit"],
        ),
        long_row(
            "G13.in_sample",
            ["canonical-ltmle-competing"],
            outcome_kind="competing",
            kinds=_LONG_BAND,
            cells=["interval_calibration"],
        ),
        long_row(
            "G13.cross_fitted",
            ["canonical-ltmle-competing-crossfit"],
            outcome_kind="competing",
            splitting=xf,
            cells=["interval_calibration"],
        ),
    ]
    survival = "point-treatment-survival"
    rows += [
        long_row(
            "G14.survival",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            kinds=_LONG_BAND,
            cells=["survival_t5", "rmst_5"],
        ),
        long_row(
            "G14.competing",
            [survival],
            outcome_kind="competing",
            held=True,
            treatment=("2",) * 4,
            kinds=_LONG_BAND,
            cells=["competing_t4"],
        ),
        long_row(
            "G14.three_arm",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("3",) * 3,
            kinds=_LONG_BAND,
            cells=["three_arm_t3"],
        ),
        long_row(
            "G14.continuous_time",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 4,
            kinds=_LONG_BAND,
            cells=["continuous_t4"],
        ),
        long_row(
            "G14.end_of_study",
            [survival],
            held=True,
            kinds=_LONG_BAND,
            cells=["end_of_study"],
        ),
        long_row(
            "G15",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            weighting="fixed",
            cells=["weighted_t5"],
            limits=["The fixed-weight cell reads a finite-sample shortfall (X13)."],
        ),
        long_row(
            "G16",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            dependence="t",
            kinds=_LONG_T,
            cells=["clustered_t5"],
        ),
        long_row(
            "G17.survival",
            ["point-treatment-survival-crossfit"],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["survival_t5"],
        ),
        long_row(
            "G17.three_arm",
            ["point-treatment-survival-crossfit"],
            outcome_kind="survival",
            held=True,
            treatment=("3",) * 3,
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["three_arm_t3"],
            limits=["The three-arm cell reads a finite-sample shortfall (X13)."],
        ),
        long_row(
            "G18.stochastic",
            ["point-treatment-survival-policies"],
            outcome_kind="survival",
            regimens=("stochastic", "mtp"),
            held=True,
            treatment=("2",) * 5,
            cells=["policy_t5"],
        ),
        long_row(
            "G18.policy",
            ["point-treatment-survival-policies"],
            outcome_kind="survival",
            regimens=("mtp",),
            held=True,
            treatment=("6",) * 5,
            cells=["mtp_t5"],
        ),
    ]
    # Per-horizon and one-step hazard targeting of a held baseline treatment.
    hazard = "point-survival-hazard"
    curve = "point-survival-curve"
    curve_xf = "point-survival-curve-crossfit"
    held5 = ("2",) * 5
    held4 = ("2",) * 4
    rows += [
        long_row(
            "G30.survival",
            [hazard, curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            survival_model="hazard",
            kinds=_LONG_BAND,
            cells=["survival_t5", "survival_t5_per_horizon"],
        ),
        long_row(
            "G30.competing",
            [hazard],
            outcome_kind="competing",
            held=True,
            treatment=held4,
            survival_model="hazard",
            kinds=_LONG_BAND,
            cells=["competing_t4"],
        ),
        long_row(
            "G31.survival",
            [curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["survival_t5"],
        ),
        long_row(
            "G31.competing",
            [curve],
            outcome_kind="competing",
            held=True,
            treatment=held4,
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["competing_t4"],
        ),
        long_row(
            "G31.three_arm",
            [curve],
            outcome_kind="survival",
            regimens=("static", "stochastic"),
            held=True,
            treatment=("3",) * 3,
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["rule_t3", "policy_t3"],
        ),
        long_row(
            "G31.weighted",
            [curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            weighting="fixed",
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["weighted_t5"],
        ),
        long_row(
            "G31.weighted_per_horizon",
            [curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            weighting="fixed",
            survival_model="hazard",
            kinds=_LONG_BAND,
            cells=["weighted_t5_per_horizon"],
        ),
        long_row(
            "G31.clustered",
            [curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            dependence="t",
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_T,
            cells=["clustered_t5"],
        ),
        long_row(
            "G31.clustered_per_horizon",
            [curve],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            dependence="t",
            survival_model="hazard",
            kinds=_LONG_T,
            cells=["clustered_t5_per_horizon"],
        ),
        long_row(
            "G31.isotonic",
            [curve],
            outcome_kind="survival",
            treatment=("2", "2"),
            isotonic=True,
            kinds=_LONG_BAND,
            cells=["isotonic_t2"],
        ),
        long_row(
            "G32.survival",
            [curve_xf],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            splitting=xf,
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["survival_t5"],
        ),
        long_row(
            "G32.survival_per_horizon",
            [curve_xf],
            outcome_kind="survival",
            held=True,
            treatment=held5,
            splitting=xf,
            survival_model="hazard",
            kinds=_LONG_BAND,
            cells=["survival_t5_per_horizon"],
        ),
        long_row(
            "G32.competing",
            [curve_xf],
            outcome_kind="competing",
            held=True,
            treatment=held4,
            splitting=xf,
            survival_model="hazard",
            targeting="one_step",
            kinds=_LONG_BAND,
            cells=["competing_t4"],
        ),
    ]
    rows += [
        long_row(
            "G19",
            ["longitudinal-msm"],
            msm=True,
            cells=["interval_calibration"],
        ),
        long_row(
            "G20",
            ["cross-fitted-longitudinal-msm"],
            msm=True,
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["interval_calibration"],
        ),
        long_row(
            "G21",
            ["known-node-mechanisms"],
            qualifiers=("A1", "A2", "C1", "C2"),
            cells=["smart_q_correct"],
        ),
        long_row(
            "G27",
            ["longitudinal-repeated-outcome"],
            outcome_kind="repeated",
            treatment=("2",) * 3,
            cells=["interval_calibration"],
        ),
    ]
    # The repeated-outcome ``IPSI`` study (X32b): three odds tilts at three nodes.  The band is
    # the default simultaneous band of the in-sample fits over all fifteen reported parameters.
    # A booster fit and the small-sample cells are diagnostic there, so they license nothing.
    ipsi_study = ["longitudinal-repeated-ipsi"]
    ipsi_limits = [
        "The gated cells draw n = 8,000. At n = 2,000 the continuous arm reads an SE ratio down "
        "to 0.92, below the calibration band, and the study reports that cell without a verdict.",
        "The estimator is singly robust: a wrong treatment or censoring mechanism leaves a bias.",
    ]

    def ipsi_row(
        row_id: str,
        *,
        kinds: frozenset[Kind] = _LONG_WALD,
        splitting: tuple[str, ...] = ("in_sample",),
        outcome: tuple[str, ...] = BINARY,
        cells: Sequence[str] = ("interval_calibration",),
    ) -> SupportRow:
        return long_row(
            row_id,
            ipsi_study,
            outcome_kind="repeated",
            regimens=("tilt",),
            treatment=("2",) * 3,
            engine="ipsi",
            kinds=kinds,
            splitting=splitting,
            outcome=outcome,
            cells=cells,
            limits=ipsi_limits,
        )

    rows += [
        ipsi_row("G37.in_sample", kinds=_LONG_BAND, cells=["interval_calibration", "all_reported"]),
        ipsi_row("G37.cross_fitted", splitting=xf),
        ipsi_row("G37.bounded", outcome=BOUNDED),
    ]
    sdr = ["longitudinal-sdr"]
    rows += [
        long_row("G23.in_sample", sdr, engine="sdr", kinds=_LONG_BAND, cells=["static"]),
        long_row(
            "G23.cross_fitted", sdr, engine="sdr", splitting=xf, kinds=_LONG_BAND, cells=["static"]
        ),
        long_row(
            "G24.survival.in_sample",
            sdr,
            outcome_kind="survival",
            engine="sdr",
            cells=["survival"],
        ),
        long_row(
            "G24.survival.cross_fitted",
            sdr,
            outcome_kind="survival",
            engine="sdr",
            splitting=xf,
            cells=["survival"],
        ),
        long_row(
            "G24.competing",
            sdr,
            outcome_kind="competing",
            engine="sdr",
            splitting=xf,
            cells=["competing"],
        ),
        long_row(
            "G24.mtp.in_sample",
            [*sdr, "mtp-boundaries"],
            regimens=("mtp",),
            engine="sdr",
            missing=("complete",),
            treatment=("continuous", "continuous"),
            cells=["mtp"],
        ),
        long_row(
            "G24.mtp.cross_fitted",
            sdr,
            regimens=("mtp",),
            engine="sdr",
            missing=("complete",),
            treatment=("continuous", "continuous"),
            splitting=xf,
            cells=["mtp"],
        ),
        long_row(
            "G24.policy.in_sample",
            sdr,
            regimens=("static", "stochastic"),
            engine="sdr",
            missing=("complete",),
            treatment=("3", "3"),
            cells=["policy"],
        ),
        long_row(
            "G24.policy.cross_fitted",
            sdr,
            regimens=("static", "stochastic"),
            engine="sdr",
            missing=("complete",),
            treatment=("3", "3"),
            splitting=xf,
            cells=["policy"],
        ),
        long_row("G24.msm", sdr, msm=True, engine="sdr", splitting=xf, cells=["msm"]),
        long_row(
            "G25", sdr, engine="sdr", weighting="fixed", splitting=xf, cells=["selection_weighted"]
        ),
        long_row(
            "G26.clusters_200",
            sdr,
            engine="sdr",
            splitting=xf,
            dependence="clustered",
            cells=["clusters_200"],
        ),
        long_row(
            "G26.clusters_20",
            sdr,
            engine="sdr",
            splitting=xf,
            dependence="t",
            kinds=_LONG_T,
            cells=["clusters_20"],
        ),
    ]
    # ``IPSI``: the incremental tilt of every node, without dropout (X19 S1) and with it (S2).
    s1 = ["longitudinal-incremental"]
    s2 = ["longitudinal-incremental-dropout"]
    ipsi: dict[str, Any] = {
        "engine": "ipsi",
        "regimens": ("tilt",),
        "outcome": ("gaussian", "bounded"),
        "treatment": ("2", "2", "2"),
    }
    ipsi_survival = {**ipsi, "outcome": BINARY}
    rows += [
        long_row(
            "G33.cross_fitted",
            s1,
            **ipsi,
            missing=("complete",),
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["contrast_x2", "mean_x2", "per_node", "tilt_then_static", "nine_tilts"],
        ),
        long_row(
            "G33.in_sample",
            s1,
            **ipsi,
            missing=("complete",),
            cells=["contrast_x2_in_sample"],
        ),
        long_row(
            "G33.known",
            s1,
            **ipsi,
            missing=("complete",),
            splitting=xf,
            qualifiers=("A1", "A2", "A3"),
            cells=["known_x2"],
        ),
        long_row(
            "G33.two_nodes",
            s1,
            **{**ipsi, "treatment": ("2", "2")},
            missing=("complete",),
            splitting=xf,
            cells=["no_positivity"],
        ),
        long_row(
            "G34.in_sample", s2, **ipsi, cells=["dropout_folds_1", "dropout_contrast_x2_in_sample"]
        ),
        long_row(
            "G34.cross_fitted",
            s2,
            **ipsi,
            splitting=xf,
            cells=["dropout_folds_5", "dropout_contrast_x2"],
        ),
        long_row(
            "G34.survival.in_sample",
            s2,
            **ipsi_survival,
            outcome_kind="survival",
            kinds=_LONG_BAND,
            cells=["survival_folds_1", "survival_contrast_x2_h3_in_sample", "survival_grid"],
        ),
        long_row(
            "G34.survival.cross_fitted",
            s2,
            **ipsi_survival,
            outcome_kind="survival",
            splitting=xf,
            kinds=_LONG_BAND,
            cells=["survival_folds_5", "survival_contrast_x2_h3", "survival_grid"],
        ),
        long_row(
            "G34.competing",
            s2,
            **ipsi_survival,
            outcome_kind="competing",
            splitting=xf,
            cells=["competing_contrast_x2_h3"],
        ),
        long_row("G34.msm", s2, **ipsi, msm=True, splitting=xf, cells=["msm_log_delta"]),
        long_row(
            "G34.weighted",
            s2,
            **ipsi,
            weighting="fixed",
            splitting=xf,
            cells=["weighted_contrast_x0_5"],
        ),
        long_row(
            "G34.clustered",
            s2,
            **ipsi,
            dependence="clustered",
            splitting=xf,
            cells=["mean_x2__cluster_robust"],
        ),
    ]
    # ``SDR`` on a repeated outcome (X32b): the study fits three nodes with the three plans.
    # The key carries no learner class, so the cell-means, quasi-binomial and booster arms of
    # the study share the row of their splitting and outcome scale.
    sdr_repeated = ["longitudinal-repeated-sdr"]
    sdr_repeated_limit = (
        "The study's lmtp comparator runs are pending, so the evidence is Python property "
        "evidence only."
    )
    rows += [
        long_row(
            "G38.in_sample",
            sdr_repeated,
            outcome_kind="repeated",
            engine="sdr",
            treatment=("2",) * 3,
            kinds=_LONG_BAND,
            cells=["interval_calibration", "simultaneous_coverage"],
            limits=[
                "Calibrated at n = 32,000 with cell-means regressions and quasi-binomial "
                "mechanisms. " + sdr_repeated_limit
            ],
        ),
        long_row(
            "G38.cross_fitted",
            sdr_repeated,
            outcome_kind="repeated",
            engine="sdr",
            treatment=("2",) * 3,
            splitting=xf,
            cells=["interval_calibration"],
            limits=[
                "Calibrated at five folds and n = 8,000, with cell-means regressions and with "
                "a shallow regularized booster, at horizon 3. " + sdr_repeated_limit
            ],
        ),
        long_row(
            "G38.continuous",
            sdr_repeated,
            outcome_kind="repeated",
            engine="sdr",
            outcome=BOUNDED,
            treatment=("2",) * 3,
            cells=["interval_calibration"],
            limits=[
                "Calibrated at n = 32,000, at horizon 3, with q_bounds=(0, 1). "
                + sdr_repeated_limit
            ],
        ),
    ]
    baselines = ["sequential-baselines"]
    for law, regimens, treatment, outcome in (
        ("d2", ("static", "stochastic"), ("2", "3"), BINARY),
        ("g", ("static",), ("2", "2", "2"), BOUNDED),
    ):
        rows.append(
            long_row(
                f"G28.{law}.gcomp",
                baselines,
                regimens=regimens,
                treatment=treatment,
                outcome=outcome,
                engine="gcomp",
                cells=[f"{law}_ice"],
            )
        )
        ipw = long_row(
            f"G28.{law}.ipw",
            baselines,
            regimens=regimens,
            treatment=treatment,
            outcome=outcome,
            engine="ipw",
        )
        rows += rows_over(
            ipw,
            "options",
            {
                "hajek": _long_options("hajek"),
                "horvitz_thompson": _long_options("horvitz_thompson"),
            },
        )
        rows[-2] = replace(rows[-2], cells=(f"{law}_hajek",))
        rows[-1] = replace(rows[-1], cells=(f"{law}_ht",))
    rows += _gformula_rows()
    rows += _structural_rows()
    # X31: a mixed node of one categorical and one or two dose components. G35 has two such
    # nodes; G36 has one node whose second dose reads the first (an ordered map).
    for row_id, treatment, ratio, cells in (
        (
            "G35.density",
            ("mixed", "mixed"),
            "density",
            ["longitudinal_vector_mixed", "longitudinal_vector_density__correctly_specified"],
        ),
        (
            "G35.classifier",
            ("mixed", "mixed"),
            "classifier",
            ["longitudinal_vector_classifier__correctly_specified"],
        ),
        (
            "G36.density",
            ("mixed",),
            "density",
            ["longitudinal_vector_ordered_density__correctly_specified"],
        ),
        (
            "G36.classifier",
            ("mixed",),
            "classifier",
            [
                "longitudinal_vector_ordered_classifier",
                "longitudinal_vector_ordered_classifier__correctly_specified",
            ],
        ),
    ):
        rows.append(
            long_row(
                row_id,
                ["vector-mtp"],
                regimens=("mtp",),
                missing=("complete",),
                treatment=treatment,
                ratio=ratio,
                cells=cells,
            )
        )
    rows += _continuous_dose_rows()
    # Repeated cross-fitting (X46): the cells of laws D, S, C, M at n = 8,000 and law I at
    # n = 80,000, each with three draws of five folds.  ``repeated`` in the splitting axis keeps
    # every single-draw row from covering a repeated fit.
    xfr = ("cross_fitted", "repeated")
    rep = ["longitudinal-repeated-crossfit"]
    rep_cells = ["interval_calibration", "simultaneous_coverage"]
    rows += [
        long_row("G44.law_d.tmle", rep, splitting=xfr, kinds=_LONG_BAND, cells=rep_cells),
        long_row(
            "G44.law_d.sdr", rep, engine="sdr", splitting=xfr, kinds=_LONG_BAND, cells=rep_cells
        ),
        long_row(
            "G44.law_s",
            rep,
            outcome_kind="survival",
            treatment=("2",) * 3,
            splitting=xfr,
            kinds=_LONG_BAND,
            cells=rep_cells,
        ),
        long_row(
            "G44.law_c",
            rep,
            outcome=("gaussian", "bounded"),
            treatment=("2",) * 3,
            splitting=xfr,
            dependence="clustered",
            kinds=_LONG_BAND,
            cells=rep_cells,
        ),
        long_row("G44.law_m", rep, msm=True, splitting=xfr, kinds=_LONG_BAND, cells=rep_cells),
        long_row(
            "G44.law_i",
            rep,
            regimens=("tilt",),
            engine="ipsi",
            outcome=("gaussian", "bounded"),
            missing=("complete",),
            treatment=("2",) * 3,
            splitting=xfr,
            kinds=_LONG_BAND,
            cells=["ipsi_primary", *rep_cells],
            limits=("law I is measured at n = 80,000, where its second-order bias is small",),
        ),
    ]
    # Node bounds on a held survival fit (X33): every node bounded, the censoring model pooled
    # over the nodes, and node 1's retention declared known (the qualifier ``time>=1``), in
    # sample with GLM learners on one five-visit law at n = 2,000.
    bounds = ["survival-node-bounds"]
    bounds_limit = (
        "The study's survtmle comparator runs are pending, so the evidence is Python property "
        "evidence only."
    )
    rows += [
        long_row(
            "G45.mean_model",
            bounds,
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            qualifiers=("time>=1",),
            node_bounds=True,
            hazard_pooling="pooled",
            cells=["interval_calibration", "binding_witness", "bound_necessity"],
            limits=[
                "Calibrated in sample at n = 2,000 with valid bounds at 1.1 times the largest "
                "at-risk iterated mean, quasibinomial node regressions and logistic mechanisms. "
                + bounds_limit
            ],
        ),
        long_row(
            "G45.hazard_model",
            bounds,
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 5,
            qualifiers=("time>=1",),
            survival_model="hazard",
            node_bounds=True,
            hazard_pooling="pooled",
            cells=["interval_calibration"],
            limits=[
                "Calibrated in sample at n = 2,000 with valid bounds at 1.1 times the largest "
                "true hazard and one pooled logistic hazard. The bounds moved the estimate on "
                "one draw in ten, so the study reports the binding witness of this fit and does "
                "not gate it. " + bounds_limit
            ],
        ),
    ]
    return tuple(rows)


def _continuous_dose_rows() -> list[SupportRow]:
    """The rows of the continuous-policy studies (X40): ``G39`` to ``G41``.

    The key reads the node count, the engine, the splitting, the outcome scale and the ratio
    route. It does not read a declared support, an atom, a branch or a tuned cap. The two-node
    density rows of ``mtp-boundaries`` therefore stand on ``G10.continuous`` and
    ``G24.mtp.in_sample``. These rows hold the keys that no earlier row covers.

    ``G39`` is the classifier ratio route on a price with a declared support
    (``mtp-pricing``), by engine and horizon. ``G40`` is the learned density at three and five
    nodes. ``G41`` is a cap tuned on the data over a bounded outcome (``mtp-tuning``).
    """
    xf = ("cross_fitted",)
    pricing, tuning = ["mtp-pricing"], ["mtp-tuning"]
    rows: list[SupportRow] = []
    for suffix, engine, nodes, splitting, outcome, cells in (
        (
            "ltmle.tau2",
            "tmle",
            2,
            ("in_sample",),
            BINARY,
            [
                "pricing_ltmle_tau2",
                "interval_calibration/up_oracle_tau2__correctly_specified",
                "interval_calibration/up_learned_classifier_tau2__correctly_specified",
            ],
        ),
        ("sdr.tau2", "sdr", 2, ("in_sample",), BINARY, ["pricing_sdr_tau2"]),
        (
            "ltmle.tau3",
            "tmle",
            3,
            ("in_sample",),
            BINARY,
            [
                "pricing_ltmle_tau3",
                "interval_calibration/up_oracle_tau3__correctly_specified",
                "interval_calibration/up_learned_classifier_tau3__correctly_specified",
                "double_robustness/up_tau3__outcome_wrong_learned_classifier",
                "ratio_necessity/up_tau3__learned_classifier",
            ],
        ),
        ("sdr.tau3", "sdr", 3, ("in_sample",), BINARY, ["pricing_sdr_tau3"]),
        (
            "ltmle.tau3_cross_fitted",
            "tmle",
            3,
            xf,
            BINARY,
            ["interval_calibration/crossfit_ltmle_learned_classifier_tau3__correctly_specified"],
        ),
        (
            "sdr.tau3_cross_fitted",
            "sdr",
            3,
            xf,
            BINARY,
            ["interval_calibration/crossfit_sdr_learned_classifier_tau3__correctly_specified"],
        ),
        (
            "ltmle.tau3_revenue",
            "tmle",
            3,
            ("in_sample",),
            BOUNDED,
            ["interval_calibration/revenue_learned_classifier_tau3__correctly_specified"],
        ),
        (
            "ltmle.tau5",
            "tmle",
            5,
            ("in_sample",),
            BINARY,
            [
                "pricing_ltmle_tau5",
                "interval_calibration/up_oracle_tau5__correctly_specified",
                "interval_calibration/up_learned_classifier_tau5__correctly_specified",
                "ratio_necessity/up_tau5__learned_classifier",
            ],
        ),
        ("sdr.tau5", "sdr", 5, ("in_sample",), BINARY, ["pricing_sdr_tau5"]),
    ):
        rows.append(
            long_row(
                f"G39.{suffix}",
                pricing,
                regimens=("mtp",),
                engine=engine,
                missing=("complete",),
                treatment=("continuous",) * nodes,
                splitting=splitting,
                outcome=outcome,
                ratio="classifier",
                cells=cells,
            )
        )
    for nodes in (3, 5):
        rows.append(
            long_row(
                f"G40.tau{nodes}",
                pricing,
                regimens=("mtp",),
                missing=("complete",),
                treatment=("continuous",) * nodes,
                cells=[f"interval_calibration/up_learned_density_tau{nodes}__correctly_specified"],
            )
        )
    tuned = "interval_calibration/{}__correctly_specified".format
    for suffix, engine, splitting, ratio, cells in (
        (
            "ltmle",
            "tmle",
            ("in_sample",),
            "density",
            [
                "population_quantile_ltmle",
                tuned("population_dose_quantile_tau2"),
                tuned("population_quantile_regression_tau2"),
            ],
        ),
        ("sdr", "sdr", ("in_sample",), "density", ["population_quantile_sdr"]),
        (
            "ltmle_cross_fitted",
            "tmle",
            xf,
            "density",
            [
                tuned("population_dose_quantile_crossfit_tau2"),
                tuned("fold_average_dose_quantile_tau2"),
                tuned("fold_average_quantile_regression_tau2"),
            ],
        ),
        (
            "ltmle_cross_fitted_classifier",
            "tmle",
            xf,
            "classifier",
            [tuned("population_dose_quantile_crossfit_classifier_tau2")],
        ),
    ):
        rows.append(
            long_row(
                f"G41.{suffix}",
                tuning,
                regimens=("mtp",),
                engine=engine,
                missing=("complete",),
                treatment=("continuous", "continuous"),
                splitting=splitting,
                outcome=BOUNDED,
                ratio=ratio,
                cells=cells,
            )
        )
    return rows


def _gformula_cells(
    law: str, labels: Sequence[str], *, contrast: str | None = "contrast"
) -> list[str]:
    """The gated positive cells of ``parametric-gformula`` that license one law.

    The controls (shrunken standard error, added noise) and the Monte Carlo accounting cells at
    one simulated history carry no row. The three-size and root-n cells read ``contrast``, and a
    law with no such cell passes ``None``.
    """
    cells = [f"interval_calibration/nice_{law}_{label}__correctly_specified" for label in labels]
    if contrast is not None:
        cells += [f"root_n_and_efficiency/nice_{law}_{contrast}__n_{n}" for n in (2000, 4000, 8000)]
        cells += [
            f"root_n_rate/nice_{law}_{contrast}__{leg}" for leg in ("empirical_sd", "reported_se")
        ]
    return cells


def _gformula_rows() -> list[SupportRow]:
    """The rows of the parametric g-formula (``G29``): one per key the study fits.

    Every fit of the study is an in-sample main-terms ``GLM`` fit with no clusters and no
    weights. ``G2`` and ``G5`` share one key, because both fit a Gaussian outcome with complete
    data and two arms at three nodes.
    """
    study = ["parametric-gformula"]
    glm = {"outcome": GLM, "treatment": GLM, "censoring": GLM}
    three = ("2", "2", "2")
    return [
        long_row(
            "G29.g1",
            study,
            regimens=("static", "tilt"),
            engine="parametric_gformula",
            treatment=three,
            learners=glm,
            cells=_gformula_cells("g1", ("contrast", "odds_x2", "rule", "treat_all", "treat_none")),
        ),
        long_row(
            "G29.g2_g5",
            study,
            regimens=("static", "stochastic", "tilt"),
            outcome=CONTINUOUS,
            missing=("complete",),
            engine="parametric_gformula",
            treatment=three,
            learners=glm,
            cells=[
                *_gformula_cells("g2", ("contrast", "policy_b", "treat_all", "treat_none")),
                *_gformula_cells(
                    "g5", ("contrast", "odds_x2", "treat_all", "treat_none"), contrast=None
                ),
                *(
                    f"ice_necessity/pair_g2_{plan}__nice"
                    for plan in ("contrast", "treat_all", "treat_none")
                ),
            ],
        ),
        long_row(
            "G29.g3",
            study,
            outcome_kind="competing",
            engine="parametric_gformula",
            treatment=("2",) * 4,
            learners=glm,
            cells=_gformula_cells(
                "g3",
                ("contrast_h4", "death_h4", "relapse_h2", "relapse_h4"),
                contrast="contrast_h4",
            ),
        ),
        long_row(
            "G29.g4",
            study,
            regimens=("static", "mtp"),
            outcome=CONTINUOUS,
            missing=("complete",),
            engine="parametric_gformula",
            treatment=("continuous",) * 3,
            learners=glm,
            cells=_gformula_cells("g4", ("contrast", "dose", "rule_dose", "shift")),
        ),
    ]


#: The calibration labels of ``structural-nested-mean`` that license each key, by setup.
_STRUCTURAL_LABELS: Final[Mapping[str, tuple[str, ...]]] = {
    "s2": ("ate_r11", "blip_a1_l1", "blip_a2_l1", "ey0", "ey_r10", "ey_r11"),
    "h": (
        "ate_r10",
        "ate_r11",
        "blip_a1_l1",
        "blip_a1_l1_w",
        "blip_a2_l1",
        "blip_a2_l1_a1",
        "blip_a2_l1_w",
        "ey0",
        "ey_r11",
    ),
    "p": ("ate_r11", "blip_a1_l1", "blip_a2_l1", "blip_a2_l1_a1", "ey0", "ey_r01"),
    "m": ("ate_r11", "blip_a1_l1", "blip_a2_l1", "ey0"),
    "s3": ("ate_r101", "ate_r111", "blip_a1_l1", "blip_a2_l1", "blip_a3_l1", "ey0"),
    "k": (
        "ate_r12",
        "ate_r21",
        "blip_a1_l1",
        "blip_a1_l2",
        "blip_a2_l1",
        "blip_a2_l2",
        "ey0",
        "ey_r12",
    ),
    "c": ("ate_dose_high", "ate_dose_mid", "blip_a1", "blip_a2", "ey0", "ey_dose_mid"),
    "l": ("blip_a1_l1", "blip_a2_l1", "blip_a2_l1_s2", "ey0"),
    "s2w": ("ate_r11", "blip_a1_l1", "ey0", "ey_r11"),
    "s2c": ("ate_r11", "blip_a1_l1", "ey0", "ey_r11"),
}

#: The size-ladder labels of ``structural-nested-mean`` and their three sizes, by setup.
_STRUCTURAL_LADDERS: Final[Mapping[str, tuple[tuple[str, ...], tuple[int, ...]]]] = {
    "s2": (("ate_r11",), (500, 2000, 8000)),
    "h": (("ate_r11",), (500, 2000, 8000)),
    "k": (("ate_r12",), (1000, 4000, 16000)),
    "c": (("ate_dose_high",), (500, 2000, 8000)),
}


def _structural_cells(*setups: str) -> list[str]:
    """The gated positive cells of ``structural-nested-mean`` that license the given setups.

    The calibration controls, the negative controls ``LU`` and ``VB``, the weak-positivity
    report ``SP`` and the model-check cells carry no row: none of them licenses an estimate.
    """
    cells: list[str] = []
    for setup in setups:
        cells += [
            f"interval_calibration/{setup}_{label}__correctly_specified"
            for label in _STRUCTURAL_LABELS[setup]
        ]
        labels, sizes = _STRUCTURAL_LADDERS.get(setup, ((), ()))
        for label in labels:
            cells += [f"root_n_and_efficiency/{setup}_{label}__n_{n}" for n in sizes]
            cells += [
                f"root_n_rate/{setup}_{label}__{leg}" for leg in ("empirical_sd", "reported_se")
            ]
    return cells


def _structural_rows() -> list[SupportRow]:
    """The rows of the structural nested mean model (``G42``): one per key the study fits.

    Every fit of ``structural-nested-mean`` is cross-fitted, with a Gaussian outcome and no
    declared bounds, complete data and the Rademacher multiplier. The keys follow the support
    records of the declared run. ``S2``, ``H``, ``P`` and ``M`` share one key. The study fits
    no simultaneous band and no derived contrast, so each row licenses the point and the Wald
    interval only.

    Law ``N`` has no row. A row names a learner class and not its settings. The study measures
    one early-stopped booster, and a booster of fixed size has a bias that does not vanish at
    n = 512,000, so a row for the class would license a configuration the study refutes.
    """
    study = ("structural-nested-mean",)
    glm: dict[str, type | None] = {"outcome": GLM, "treatment": GLM, "regression": None}
    limits = (
        "The intervals are valid under the linear structural nested mean model with the "
        "declared blip features. The study laws satisfy that model.",
        "The EconML DynamicDML comparator runs are pending.",
    )

    def row(
        row_id: str,
        cells: list[str],
        *,
        treatment: tuple[str, ...] = _TWO,
        weighting: str = "none",
        dependence: str = "iid",
        regimens: bool = True,
        covariate_features: bool = False,
        extra: tuple[str, ...] = (),
    ) -> SupportRow:
        key = SupportKey(
            kind="longitudinal",
            engine=("snmm",),
            family=frozenset(
                ("end_of_study", "blips", *(("regimens", "static") if regimens else ()))
            ),
            outcome=CONTINUOUS,
            missing=("complete",),
            treatment=treatment,
            weighting=weighting,
            splitting=("cross_fitted",),
            dependence=dependence,
            qualifiers=frozenset(),
            options=(("covariate_features", covariate_features), ("multiplier_kind", "rademacher")),
        )
        return SupportRow(
            row_id, key, _LONG_WALD, study, tuple(cells), dict(glm), (*limits, *extra)
        )

    return [
        row("G42.two_nodes", _structural_cells("s2", "h", "p", "m")),
        row("G42.three_nodes", _structural_cells("s3"), treatment=("2", "2", "2")),
        row("G42.categorical", _structural_cells("k"), treatment=("3", "3")),
        row("G42.dose", _structural_cells("c"), treatment=("continuous", "continuous")),
        row(
            "G42.covariate_feature",
            _structural_cells("l"),
            regimens=False,
            covariate_features=True,
            extra=(
                "A feature that reads a time-varying covariate reports the blips and the "
                "reference mean only. A static level or contrast is refused (X72).",
            ),
        ),
        row("G42.weighted", _structural_cells("s2w"), weighting="fixed"),
        row("G42.clustered", _structural_cells("s2c"), dependence="clustered"),
    ]


def _long_options(normalization: str) -> tuple[tuple[str, Any], ...]:
    return (
        ("multiplier_kind", "rademacher"),
        ("normalization", normalization),
        ("ratio", "density"),
        ("survival_model", "iterated_mean"),
        ("targeting", "iterative"),
        ("isotonic", False),
        *_x33_options(),
    )


def _categorical_long_rows() -> tuple[SupportRow, ...]:
    """The longitudinal rows of the study ``categorical-incremental`` (X43): ``G43``."""
    slug = ["categorical-incremental"]
    ipsi: dict[str, Any] = {
        "engine": "ipsi",
        "regimens": ("tilt",),
        "outcome": ("gaussian", "bounded"),
        "treatment": ("3", "3"),
        "splitting": ("cross_fitted",),
    }
    return (
        long_row(
            "G43.cross_fitted",
            slug,
            **ipsi,
            cells=["longitudinal_dropout_folds_5", "longitudinal_dropout_contrast_t_a"],
        ),
        long_row(
            "G43.complete",
            slug,
            **ipsi,
            missing=("complete",),
            cells=["longitudinal_folds_5", "longitudinal_contrast_t_a"],
        ),
    )


#: The longitudinal rows: ``LTMLE`` (``G01`` to ``G21``, ``G27``), ``SDR`` (``G23`` to ``G26``),
#: the sequential baselines (``G28``), the parametric g-formula (``G29``), repeated-outcome
#: ``IPSI`` (``G37``), the continuous-dose price and tuned-cap rows (``G39`` to ``G41``), the
#: structural nested mean model (``G42``) and the categorical tilt (``G43``).
LONGITUDINAL_ROWS: Final[tuple[SupportRow, ...]] = _with_shortfalls(
    (*_long_rows(), *_categorical_long_rows())
)


# ------------------------------------------------------------------ exemptions and anchors

#: The registered studies that no row names, each with the reason. A study that measures no
#: estimate cannot license a configuration, so it is not the evidence of a row.
EXEMPT_STUDIES: Final[Mapping[str, str]] = {
    "omitted-variable-bound-se": (
        "It measures the standard error of a sensitivity bound. The bound carries its own "
        "status and is no fit of a row."
    ),
    "calibration-slope-warning": (
        "It measures the rate of a diagnostic warning. The warning is no estimate."
    ),
    "learned-rule-cvtmle-boundary": (
        "It reports truth rows at the exceptional and weak-blip laws, has no property cell, "
        "and licenses no configuration (F27)."
    ),
}

#: The constructor options that stay out of a key, each with a file that states the option. An
#: option here changes how a nuisance is fitted or how a run is seeded and scheduled, and a
#: study covers the target at every value. A row names a learner class only where it sets
#: ``learners``.
OPTION_ALLOWLIST: Final[Mapping[str, str]] = {
    "outcome_learner": "src/cleverly/estimators/tmle.py",
    "treatment_learner": "src/cleverly/estimators/tmle.py",
    "missingness_learner": "src/cleverly/estimators/tmle.py",
    "intermediate_learner": "src/cleverly/estimators/tmle.py",
    "pseudo_learner": "docs/technical-reference/longitudinal-tmle.md",
    "censoring_learner": "docs/technical-reference/longitudinal-tmle.md",
    "g_bounds": "docs/technical-reference/point-treatment-tmle.md",
    "n_folds": "docs/technical-reference/cv-tmle.md",
    "learner_folds": "docs/technical-reference/longitudinal-tmle.md",
    "random_state": "src/cleverly/estimators/tmle.py",
    "n_jobs": "src/cleverly/estimators/tmle.py",
    "run_id": "src/cleverly/estimators/tmle.py",
}


@dataclass(frozen=True)
class TheoryAnchor:
    """Where the theory of one target family is checked.

    Parameters
    ----------
    row : str
        The first cell of the row of ``docs/technical-reference/evidence.md`` that covers the
        family. Empty when that page has no row for the family yet.
    instruments : tuple of str
        The exact-law and Gateaux files of the family, relative to the repository root.
    """

    row: str
    instruments: tuple[str, ...]


def _anchor(row: str, *instruments: str) -> TheoryAnchor:
    return TheoryAnchor(row, instruments)


#: The theory anchor of each family a key can hold. A family with no anchor has no exact-law or
#: theory row, and a fit that reports it is experimental.
FAMILY_THEORY: Final[Mapping[str, TheoryAnchor]] = {
    "P1": _anchor("`ate`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux.py"),
    "P2": _anchor("`rr`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux.py"),
    "P3": _anchor("`att`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux.py"),
    "P4": _anchor("`par`", "tests/discrete_law.py", "tests/discrete_law_mar.py"),
    "P5": _anchor(
        "`ey_regime`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux_regime.py"
    ),
    "P6": _anchor(
        "`ey_regime`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux_regime.py"
    ),
    "P7": _anchor(
        "`ey_policy`", "tests/discrete_law_shift.py", "tests/unit/test_influence_gateaux_shift.py"
    ),
    "P8": _anchor(
        "`ey_rr_tilt`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux_rr_tilt.py"
    ),
    "P9": _anchor(
        "`ey_ipsi`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux_ipsi.py"
    ),
    "P10": _anchor("`msm`", "tests/discrete_law.py", "tests/unit/test_influence_gateaux_msm.py"),
    "P11": _anchor("`ey_learned_rule`", "tests/discrete_law.py"),
    "blips": _anchor(
        "structural nested mean model (`StructuralNestedMean`)",
        "tests/discrete_law_structural.py",
        "tests/unit/test_structural_nested_mean.py",
    ),
    "end_of_study": _anchor(
        "end-of-study, static and dynamic regimens",
        "tests/discrete_law_longitudinal.py",
        "tests/unit/test_influence_gateaux_longitudinal.py",
    ),
    "survival": _anchor(
        "absorbing survival curve",
        "tests/discrete_law_survival.py",
        "tests/unit/test_influence_gateaux_survival.py",
    ),
    "competing": _anchor(
        "competing-risks cumulative incidence",
        "tests/discrete_law_competing.py",
        "tests/unit/test_influence_gateaux_competing.py",
    ),
    "repeated": _anchor(
        "repeated outcome means over horizons",
        "tests/discrete_law_repeated.py",
        "tests/unit/test_influence_gateaux_repeated.py",
    ),
    "regimens": _anchor(
        "end-of-study, static and dynamic regimens",
        "tests/discrete_law_longitudinal.py",
        "tests/unit/test_influence_gateaux_longitudinal.py",
    ),
    "static": _anchor(
        "end-of-study, static and dynamic regimens",
        "tests/discrete_law_longitudinal.py",
        "tests/unit/test_influence_gateaux_longitudinal.py",
    ),
    "msm": _anchor(
        "working model over regimen/horizon cells",
        "tests/discrete_law_longitudinal.py",
        "tests/unit/test_influence_gateaux_longitudinal_msm.py",
    ),
    "stochastic": _anchor(
        "known stochastic categorical policies",
        "tests/discrete_law_longitudinal_policy.py",
        "tests/unit/test_influence_gateaux_longitudinal_policy.py",
    ),
    "mtp": _anchor(
        "",
        "tests/discrete_law_longitudinal_mtp.py",
        "tests/unit/test_influence_gateaux_longitudinal_mtp.py",
    ),
    "tilt": _anchor(
        "",
        "tests/discrete_law_longitudinal_ipsi.py",
        "tests/unit/test_influence_gateaux_ipsi.py",
    ),
}
