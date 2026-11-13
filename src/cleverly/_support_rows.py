"""The support rows of the point-treatment estimators.

Each row names one configuration that a registered study covers, the kinds of inference the
study licenses there, and the study cells that license it. The rows are explicit tuples: a
row covers a set of values on one axis only where :func:`rows_over` expands it into one row
per value, and each expanded row has a case in ``tests/studies/support_cases.py``.

The row ids follow ``docs/roadmap.md`` and the support inventory: ``T`` for ordinary TMLE,
``D`` for DR-TMLE, ``C`` for C-TMLE and ``AI`` for the adaptive-propensity IPTW. A suffix names
the expanded value. The longitudinal rows (``G``) are a later addition.

A leaf module: it imports :mod:`cleverly._support` only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from ._support import (
    DEFAULT_BAND_OPTION,
    KEY_AXES,
    T_BAND_OPTION,
    Kind,
    SupportKey,
    SupportRow,
)
from .learners.glm import GLM

__all__ = [
    "BAND_CELLS",
    "BAND_OPTION_DRAWS",
    "DEFAULT_BAND_DRAWS",
    "EXEMPT_STUDIES",
    "FAMILY_THEORY",
    "LONGITUDINAL_ROWS",
    "MEASURED_LIMITS",
    "OPTION_ALLOWLIST",
    "POINT_ROWS",
    "SHORTFALLS",
    "T_BAND_CELLS",
    "T_BAND_STUDY",
    "UNMAPPED_BAND_CELLS",
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
    """The options a support row of ``engine`` carries in its key unless it overrides them.

    For ``"tmle"`` and ``"collaborative_tmle"`` these are the options of a default fit.  For
    ``"drtmle"`` they differ from a default fit in one option.  The key names
    ``reduced_crossfit="pooled"``, which :func:`cleverly._support.point_key` records for
    every fit that runs the in-sample or unguarded path, and which the stacked rows ``D01``
    to ``D03`` measured against R ``drtmle``.  A default cross-fitted ``DRTMLE`` fit runs
    ``"foldwise"``, and only a row that overrides ``reduced_crossfit`` covers it.

    Parameters
    ----------
    engine : str
        ``"tmle"``, ``"drtmle"``, ``"collaborative_tmle"`` or ``"adaptive_iptw"``.

    Returns
    -------
    dict of str to Any
        The option names and values of :func:`cleverly._support.point_key`.
    """
    options: dict[str, Any] = {
        "fluctuation": "logistic",
        "targeting": "iterative",
        "target_weights": False,
        "stopping": "relative",
        "step_bound": "none",
        "iterate": "last",
        "outcome_by_arm": False,
        "patience": False,
        "treatment_adjustment": False,
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
    if engine == "adaptive_iptw":
        options |= {
            "form": "targeted",
            "mechanism_learner": "adaptive",
            "reduced_learner": "selecting",
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
        The target families the row covers, as ``P1`` to ``P14``.
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
    # Roadmap: X26.
    "mechanism that is a LogisticRegression or is declared known. They do not calibrate "
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
            # A clustered fold-evaluated estimate takes t with J - V degrees of freedom at
            # every cluster count.
            kinds=_WALD_T,
            cells=["cluster_robust_fold_evaluated_cluster_covariate"],
            limits=[
                "Equal cluster sizes only: cv_evaluation=True refuses clusters whose size (or "
                "total weight, or treated count for att) differs.",
            ],
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
            limits=[
                "The fold-evaluated cells use equal cluster sizes only, because "
                "cv_evaluation=True refuses clusters whose size (or total weight, or treated "
                "count for att) differs."
            ],
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
            kinds=_WALD | {"band"},
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
            # Roadmap: X8.
            limits=["The logit slope at the smallest stratum is red."],
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
            # Roadmap: X8.
            limits=["One paired cell at the smallest stratum is red."],
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
            # Roadmap: RM18.
            limits=["One paired calibration leg is red."],
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
            # Roadmap: F27.
            limits=["The exceptional and weak-blip laws are red."],
        )
    )
    # X11 (h), (k) and (l): the learned-rule value with fold targeting, known weights and
    # clusters. Unequal cluster sizes key the qualifier ``unequal_clusters``, which only the
    # stacked row T42.clustered declares, and a weighted clustered fit has no row either.
    equal_sizes = (
        "The study draws equal cluster sizes only. A fit with unequal sizes is experimental, "
        "because sizes that depend on the outcome put a ratio bias in the fold-evaluated point. "
        "The stacked report (cv_evaluation=False) at unequal sizes has its own row, T42.clustered."
    )
    rows.append(
        point_row(
            "T37.fold_targeted",
            ["learned-rule-compositions"],
            families=("P11",),
            splitting=("fold_targeted_evaluated",),
            cells=["fold_targeted"],
            # Roadmap: X11.
            limits=[
                "The study measures coverage with a logistic outcome learner at one law. An "
                "outcome learner that omits W1 is measured for bias only."
            ],
        )
    )
    rows.append(
        point_row(
            "T37.weighted",
            ["learned-rule-compositions"],
            families=("P11",),
            weighting="fixed",
            splitting=("fold_evaluated",),
            cells=["weighted"],
            # Roadmap: X11.
            limits=[
                "The study measures a known observation weight only. Estimated weights refuse.",
                "A weighted fit with clusters is experimental.",
            ],
        )
    )
    rows.append(
        point_row(
            "T37.clustered",
            ["learned-rule-compositions"],
            families=("P11",),
            splitting=("fold_evaluated",),
            dependence="clustered",
            # J - V degrees of freedom at every cluster count: t_190 at 200 clusters in 10 folds.
            kinds=_WALD_T,
            cells=["clustered"],
            # Roadmap: X11.
            limits=[equal_sizes],
        )
    )
    rows.append(
        point_row(
            "T37.few_cluster",
            ["learned-rule-compositions"],
            families=("P11",),
            splitting=("fold_evaluated",),
            dependence="t",
            kinds=_WALD_T,
            cells=["clustered_few__declared"],
            # Roadmap: X11.
            limits=[
                "The study measures 30 clusters of 34 rows in 10 folds, with 20 degrees of "
                "freedom.",
                equal_sizes,
            ],
        )
    )
    # X11 (e): the stacked report of the learned-rule value keys ``stacked``. The rows T42 of
    # the ``learned-rule-stacked`` study cover it on rows, with a known weight and with clusters.
    # X11 (a): the in-sample fit of one rule learned on every row keys ``in_sample`` and the
    # learner families (``outcome_learner_family``, ``treatment_learner_family`` and, for a rule
    # learner, ``rule_learner_family``). The row T43 of the ``learned-rule-in-sample`` study
    # covers the plug-in rule with a parametric outcome and a parametric treatment learner.
    # X11 (c): known regimes beside a learned rule (``interventions=``) key the qualifier
    # ``known_regimes``. The rows T44 of the ``learned-rule-contrast`` study cover such a fit on
    # rows under each report, and every other composition with known regimes matches no row.
    # X11 (d): the plug-in rule over K > 2 arms keys ``treatment=("arms{K}",)``. The row T45 of
    # the ``learned-rule-arms`` study covers the fold-evaluated report on rows at three arms, and
    # every other composition over more than two arms matches no row.
    # X11 (f): the value of the optimal rule (``ey_optimal_rule``) keys family ``P12`` in place
    # of ``P11``. The row T46 of the ``learned-rule-optimal-value`` study covers the
    # fold-evaluated report on rows with a known treatment mechanism (``known_mechanism``), and
    # every other composition of the optimal value, an estimated g among them, matches no row.
    # X11 (j): the plug-in rule with the intermediate held at each level (``intermediate=``) keys
    # ``missing=("intermediate",)``. The row T47 of the ``learned-rule-intermediate`` study covers
    # the fold-evaluated report on rows at each level, and the row T48 covers a known regime
    # through the same fold-evaluated controlled-direct-effect path. Every other composition with
    # an intermediate matches no row.
    # X11 (i): the learned-rule value as the mean over R splits. The key reads ``repeated``
    # beside ``fold_evaluated``, so fold targeting, weights and clusters, which key other
    # splitting, weighting and dependence axes, match no row.
    rows.append(
        point_row(
            "T38.split_mean",
            ["learned-rule-split-mean"],
            families=("P11",),
            splitting=("fold_evaluated", "repeated"),
            cells=["non_exceptional", "misspecified_limit"],
            # Roadmap: X11.
            limits=[
                "The study measures unweighted rows with pooled targeting and no id, at ten "
                "folds and repeats=5. A fit with weights, fold targeting or clusters at "
                "repeats above 1 matches no row and is experimental, and clusters refuse.",
                "The key holds neither the fold count nor the repeat count, so a fit at "
                "another repeat count reads supported on the linearity argument, which covers "
                "any fixed repeat count, and no cell measures it.",
                "The band cannot separate the declared variance from the mean of the split "
                "variances, which was 1.2% to 1.6% higher in the pilot. The formula and the "
                "per-split truth rest on the unit witnesses.",
                "The exceptional and weak-blip laws are not measured.",
            ],
        )
    )
    # X11 (g): the learned-rule value when a blip or a weighted-classification rule learner
    # sets each fold's rule. The qualifier ``rule_learner_<loss>`` keys the loss, so the plug-in
    # row T24 never covers these fits and each loss has its own row. The splitting axis reads
    # ``fold_evaluated`` without ``repeated``, and weights, clusters and fold targeting key
    # other axes, so a fit that composes a rule learner with any of them matches no row.
    # SE ratio and coverage of each loss at n = 4,000 (``interval_calibration``), 2,000 (primary)
    # and 1,000 (``small_sample_diagnostic``), from the committed results of the study.
    measured = {
        "blip": (
            "1.0080 and 0.9501 at n = 4,000, 0.9644 and 0.9375 at n = 2,000, and 0.9615 and "
            "0.9410 at n = 1,000"
        ),
        "weighted_classification": (
            "0.9886 and 0.9456 at n = 4,000, 0.9525 and 0.9380 at n = 2,000, and 0.9557 and "
            "0.9325 at n = 1,000"
        ),
    }
    for loss, reading in measured.items():
        rows.append(
            point_row(
                f"T39.{loss}",
                ["learned-rule-learners"],
                families=("P11",),
                splitting=("fold_evaluated",),
                qualifiers=[f"rule_learner_{loss}"],
                cells=[loss],
                # Roadmap: X11.
                limits=[
                    "The study measures unweighted rows with pooled targeting, repeats=1 and no "
                    "id. A rule-learner fit with weights, fold targeting, clusters or repeats "
                    "above 1 matches no row and is experimental.",
                    "The interval is calibrated at n = 4,000. At n = 1,000 and 2,000 the "
                    "standard error is about 3.5% to 5% small and the coverage about 0.93 to "
                    f"0.94. The SE ratio and the coverage are {reading}. The key holds no sample "
                    "size, so a smaller fit also reads supported.",
                    "The deficit is the covariance between folds, because each fold's rule is "
                    "fitted on the other folds' validation rows. It is a second-order remainder "
                    "of relative order about 1/n, larger for a pseudo-outcome rule learner than "
                    "for the plug-in rule, and it grows with the learner's variability.",
                    "The study measures a linear regression or a logistic regression on a "
                    "quadratic basis of two covariates, at one law. A more flexible rule learner "
                    "rests on the theory alone and may need a larger n for the same calibration.",
                    "The fold-locality control shows that the instrument resolves a rule learned "
                    "on the validation rows. The subtler leak, outer nuisances reused for the "
                    "pseudo-outcome, rests on the unit witnesses W-g3 and W-g4.",
                    "The value gain over the plug-in rule is descriptive and carries no test.",
                ],
            )
        )
    # X28 (X11 PR E1): fold-evaluated CV-TMLE with baseline strata. The qualifier ``strata``
    # and the splitting ``fold_evaluated`` key the fit, so the unstratified T03 never covers it.
    # The study fits the arm means and the ATE (P1). A regime mean with strata keys P5 and
    # matches no row, and ``repeats`` above 1 keys ``repeated``. The rows run over the outcome
    # kind and the weighting, as T03 does, because each outcome has a weighted cell.
    strata_limits = [
        "The study measures the arm means and the ATE on rows with pooled targeting, complete "
        "data, no id and repeats=1. The regime means and their contrasts, which the "
        "construction also reports, are not measured and match no row.",
        "Measured at n = 1,000 with 5 folds and three strata whose smallest share is 0.2, so "
        "about 200 rows and about 17 treated rows per stratum-fold. The key holds neither the "
        "fold count nor the stratum shares. With 10 folds at n = 500 the fit refuses about 14% "
        "of samples for an empty stratum-fold arm, and that refusal is an error, never a silent "
        "estimate. More folds or smaller strata leave fewer rows per fold plug-in and were not "
        "measured.",
        "Every learner of the study is a correctly specified parametric model. A flexible "
        "learner rests on the theory alone.",
        "The intervals are pointwise (simultaneous=False). The simultaneous band is not "
        "measured, so the row lists no band.",
        "With repeats above 1 the marginal estimate is the median of the draws' share "
        "mixtures, and such a fit matches no row.",
        "The unnormalised-curve control shows that the instrument resolves a gross "
        "normalisation error. The fold-local stratum share, against the whole-sample share, "
        "rests on the unit witnesses of tests/unit/test_fold_evaluated_strata_exact.py.",
        "The targeting control's learner sees neither W1 nor the stratum. It shows that "
        "targeting repairs a stratum-blind initial fit, not a subtle misspecification within a "
        "stratum.",
    ]
    outcome_limits = {
        "binary": [
            "On the binary outcome the smallest stratum-arm holds about 38 rows of the rarer "
            "outcome value at n = 1,000 (5th percentile about 30). With fewer, the Wald interval "
            "misses on the side of the rarer value: with about 6 such rows on average, ey[1] in "
            "the smallest stratum covered 0.909 and the stacked stratified fit 0.9155 on the same "
            "draws (sparse_event_diagnostic, no verdict).",
            "The share term of the marginal curve carries too little of the variance on a "
            "binary outcome to be resolved there. It is tested on the bounded outcome, and its "
            "formula does not read the outcome type.",
        ],
        "continuous": [
            "The bounded outcome is a beta law on (0, 1) with q_bounds=(0, 1). The share term "
            "of the marginal curve carries 27% of the marginal ATE variance there, and a fit "
            "without it fails its band.",
        ],
    }
    for weighted in (False, True):
        for kind, outcome in (("binary", BINARY), ("continuous", BOUNDED)):
            scenario = ("binary_" if kind == "binary" else "") + (
                "weighted" if weighted else "unweighted"
            )
            weight_limits = (
                [
                    "The study measures the known weight 1 + 1.5 W2, which varies inside every "
                    "stratum."
                ]
                if weighted
                else []
            )
            rows.append(
                point_row(
                    f"T40.{kind}_weighted" if weighted else f"T40.{kind}",
                    ["fold-evaluated-strata"],
                    families=("P1",),
                    outcome=outcome,
                    weighting="fixed" if weighted else "none",
                    splitting=("fold_evaluated",),
                    qualifiers=["strata"],
                    cells=[scenario],
                    # Roadmap: X28.
                    limits=[*strata_limits, *outcome_limits[kind], *weight_limits],
                )
            )
    # X11 (m) (X11 PR E2): the learned-rule value inside baseline strata, E1's construction
    # with the fold rule as the regime. The qualifier ``strata`` keys it, so T24 never covers
    # it. A rule learner keys ``rule_learner_<loss>`` beside ``strata``, fold targeting and
    # clusters key other axes and ``repeats`` above 1 keys ``repeated``, so each of those
    # compositions matches no row. The rows run over the outcome kind and the weighting,
    # because each outcome has a weighted cell. The numbers are the committed results of the
    # study (``tests/canonical/learned_rule_strata``).
    learned_strata_limits = [
        "The study measures the plug-in rule learner (LearnedRule()) on rows or with a known "
        "weight, with pooled targeting, complete data, no id and repeats=1. A rule-learner "
        "loss, fold targeting, clusters or repeats above 1 with strata matches no row.",
        "The target is data-adaptive: the boundary stratum's value varies with the learned "
        "rules, and a stratum the rules treat alike has a fixed target. Intervals are "
        "pointwise (simultaneous=False); the band and repeats > 1 are not measured. The rule "
        "boundary was never placed in the smallest stratum.",
        "Every outcome learner of the study is a correctly specified parametric model on "
        "(A, W1, W2, S). A flexible learner rests on the theory alone.",
        "The unnormalised-curve control and, on the bounded outcome, the share-dropped "
        "control show that the instrument resolves a gross normalisation error and a missing "
        "share term. The fold-local stratum share rests on the unit witnesses.",
        "The targeting control's learner sees neither W1 nor the stratum, and each of its "
        "truths is the harness refit with that learner. It shows that targeting repairs a "
        "stratum-blind initial fit at a learned rule, not a subtle misspecification within a "
        "stratum.",
    ]
    learned_outcome_limits = {
        "binary": [
            "Measured at n = 1,000 with 5 folds and three strata whose smallest share is 0.2, "
            "on laws whose blip has a nonzero slope at the rule boundary: 1.5 on a bounded "
            "outcome and 3.0 on the logit scale on a binary outcome. The boundary fell in a "
            "stratum of share 0.35 (bounded) or 0.45 (binary), and the five fold rules "
            "disagreed on about 8% and 14% of that stratum's rows. With a weaker margin the "
            "fold rules disagree on more of the sample, the target moves with them, and the "
            "interval under-covers because the variance of learning the rule is not yet "
            "negligible. At the binary blip 0.1 - 0.3 W1 the study's diagnostic cell reads "
            "coverage 0.915 to 0.939 in every stratum and marginally, with the fold rules "
            "disagreeing on 44% of the boundary stratum. At -0.6 + 1.5 W1 the boundary "
            "stratum reads an SE ratio of 0.968 while the other strata are calibrated. The "
            "unstratified learned-rule fit under-covers alike on the same draws, so this is "
            "the finite-sample weak-margin limit of the learned-rule value (roadmap F27, the "
            "weak_blip row of the learned-rule boundary study), not of the stratified "
            "construction.",
            "The share term of the marginal curve carries too little of the variance on the "
            "binary outcome to be resolved there. It is tested on the bounded outcome.",
        ],
        "continuous": [
            "Measured at n = 1,000 with 5 folds and three strata whose smallest share is 0.2, "
            "on a beta law on (0, 1) with q_bounds=(0, 1) whose blip has slope 1.5 at the rule "
            "boundary. The boundary fell in a stratum of share 0.35, and the five fold rules "
            "disagreed on about 8% of that stratum's rows. With a weaker margin the fold rules "
            "disagree on more of the sample and the interval under-covers, as the "
            "unstratified learned-rule fit does (roadmap F27, the weak_blip row of the "
            "learned-rule boundary study).",
        ],
    }
    for weighted in (False, True):
        for kind, outcome in (("binary", BINARY), ("continuous", BOUNDED)):
            scenario = ("binary_steep" if kind == "binary" else "") + (
                ("_weighted" if kind == "binary" else "weighted")
                if weighted
                else ("" if kind == "binary" else "unweighted")
            )
            weight_limits = (
                [
                    "The study measures the known weight 1 + 1.5 W2, which varies inside every "
                    "stratum."
                ]
                if weighted
                else []
            )
            rows.append(
                point_row(
                    f"T41.{kind}_weighted" if weighted else f"T41.{kind}",
                    ["learned-rule-strata"],
                    families=("P11",),
                    outcome=outcome,
                    weighting="fixed" if weighted else "none",
                    splitting=("fold_evaluated",),
                    qualifiers=["strata"],
                    cells=[scenario],
                    # Roadmap: X11, F27.
                    limits=[
                        *learned_strata_limits,
                        *learned_outcome_limits[kind],
                        *weight_limits,
                    ],
                )
            )
    # X11 (e) (X11 PR F1): the stacked report of the learned-rule value, ``cv_evaluation=False``
    # with pooled targeting, keyed ``stacked``. Fold targeting keys ``fold_targeted``, strata
    # key ``strata``, a rule learner keys ``rule_learner_<loss>`` and ``repeats`` above 1 keys
    # ``repeated``, so each of those compositions matches no row, and neither does a weighted
    # clustered fit or one under the t reference. The study's cluster cell passed every clause
    # of its pre-declared ``UNEQUAL_CLUSTERS_RULE`` (``tests/studies/learned_rule_stacked.py``),
    # so ``T42.clustered`` declares ``unequal_clusters`` with the measured scope in its limits,
    # and ``point_key`` still adds the qualifier, which keeps every fold-evaluated row off such
    # a fit. The numbers are the committed results (``tests/canonical/learned_rule_stacked``).
    stacked_limits = [
        "The study measures the pointwise stacked interval of the plug-in rule (LearnedRule()) "
        "at nominal n = 2,000 with 10 folds, repeats=1, pooled targeting, complete data and "
        "the RM30 logistic learners, in three settings: equal folds on rows, a known weight, "
        "and 200 clusters. Strata, fold targeting, a rule-learner loss, repeats above 1 and "
        "weights with clusters under the stacked report match no row. The key holds neither "
        "the fold count, the sample size nor the learner, so another n, V or learner reads "
        "supported and was not measured.",
        "The target is data-adaptive: sum_v (W_v/W) Psi_{d_v}(P0), each fold weighted by its "
        "share of the rows (of the weight mass under weights=). At unequal folds it is not the "
        "fold-evaluated target. The calibration verdicts cannot tell the two targets apart in "
        "these cells. The weighting rests on the unit witnesses and the harness's fold-weight "
        "check.",
        "The stacked variance is not guaranteed to be at least the fold-evaluated second "
        "moment, because the cross term of the variance identity can be negative. Over the "
        "primary replications the mean cross term is -0.09% to -0.16% and the between-fold "
        "term +0.02% to +0.03% of that second moment, which moves the SE by under 0.1%, far "
        "below the study's resolution, so no calibration reading is attributed to them. Both "
        "terms scale as V/n and are unmeasured at a smaller n or a larger V.",
        "The omitted-variable bound of a stacked fit refuses (X11 part (n2)).",
        "No comparator: R polle's Algorithm 4 is the construction's source, not a pinned "
        "comparator.",
    ]
    unequal_clusters_limits = [
        "The cluster cell passed the study's pre-declared unequal_clusters rule, so this row "
        "declares unequal cluster sizes. It claims only the measured scope: about 200 clusters "
        "of size 5 to 15 in 10 folds, the size rising with the cluster's outcome effect "
        "(size-curve correlation 0.891, curve ICC 0.077), with no weights. Fits with 40 to 199 "
        "clusters share the key. The law's oracle puts the stacked ratio bias at about -0.03 SD "
        "at 40 clusters, but the interval was not measured there. A fit under the t reference "
        "(fewer than 40 clusters) matches no row.",
        "The fold-evaluated report keeps unequal_clusters by theory, not by a failed verdict: "
        "its ratio bias grows as V/sqrt(J) in SDs. On the same draws its error minus the "
        "stacked report's error is -0.00213 (MC SE 0.000015), about -0.11 stacked SDs, and the "
        "oracle puts its bias at about -0.29 SD at 40 clusters. At 200 clusters its interval "
        "would pass the same bands.",
    ]
    for suffix, weighting, dependence, qualifiers, cell, extra in (
        ("rows", "none", "iid", (), "rows", []),
        (
            "weighted",
            "fixed",
            "iid",
            (),
            "weighted",
            [
                "The study measures the known weight w = 1 + 1.5 * 1{W1 > 0} at the "
                "misspecified_limit law. Its calibration SE ratio below 1 (0.986) matches the "
                "fold-evaluated compositions cell of the same law (0.969) and is a property of "
                "that law. Estimated weights refuse."
            ],
        ),
        (
            "clustered",
            "none",
            "clustered",
            ("unequal_clusters",),
            "clustered",
            unequal_clusters_limits,
        ),
        (
            "clustered_equal_sizes",
            "none",
            "clustered",
            (),
            "clustered",
            [
                "The cluster cell draws unequal sizes that rise with the outcome effect. Equal "
                "sizes are the case in which neither report carries a ratio bias, and the same "
                "variance rule applies, but no cell draws them under the stacked report. A "
                "weighted clustered fit or a fit under the t reference matches no row.",
            ],
        ),
    ):
        rows.append(
            point_row(
                f"T42.{suffix}",
                ["learned-rule-stacked"],
                families=("P11",),
                weighting=weighting,
                splitting=("stacked",),
                dependence=dependence,
                qualifiers=qualifiers,
                cells=[cell],
                # Roadmap: X11.
                limits=[*stacked_limits, *extra],
            )
        )
    # X11 (a) (X11 PR F2): the in-sample fit of one plug-in rule learned on every row,
    # ``cross_fit=False``, keyed ``in_sample`` with the learner families. The row declares both
    # families ``parametric`` and nothing else, because the study's declared support condition
    # held: both primary tests, every ``interval_calibration`` verdict and the
    # ``targeting_necessity`` verdicts passed. An adaptive learner keys ``adaptive``, known
    # treatment probabilities key ``known`` and ``known_mechanism``, and a rule learner keys
    # ``rule_learner_family`` and ``rule_learner_<loss>``, so each of those fits matches no row.
    # The numbers are the committed results (``tests/canonical/learned_rule_in_sample``).
    rows.append(
        point_row(
            "T43",
            ["learned-rule-in-sample"],
            families=("P11",),
            splitting=("in_sample",),
            options={
                "outcome_learner_family": "parametric",
                "treatment_learner_family": "parametric",
            },
            cells=["non_exceptional", "misspecified_limit"],
            # Roadmap: X11.
            limits=[
                "The study measures the pointwise in-sample interval of the plug-in rule "
                "(LearnedRule(), cross_fit=False) at n = 2,000 on two RM30 laws: one whose "
                "interaction logistic outcome model is correct (non_exceptional) and one whose "
                "model cannot represent the blip (misspecified_limit), both marked parametric, "
                "with a correct logistic treatment model. Binary outcome and treatment, iid "
                "rows, no weights. Rule learners, known treatment probabilities, weights, "
                "clusters, strata, continuous outcomes and more than two arms match no row. "
                "The key holds neither n nor the learner beyond its family, so another n or "
                "another parametric learner reads supported and was not measured.",
                "The target is Psi_{d_n}(P0), the value of the one rule learned on every row. "
                "At this size the calibration verdicts cannot tell it from the fold-average "
                "target of the cross-fitted fit: the SD of the gap between the two targets is "
                "0.008 (non_exceptional) and 0.015 (misspecified_limit) error SDs. The unit "
                "witnesses establish which rule the truth reads, and that known treatment "
                "probabilities enter the clever covariate; no verdict here sees either.",
                "The in-sample point is optimistic: the primary standardized bias is +0.16 "
                "(non_exceptional) and +0.21 (misspecified_limit) error SDs. A passed bias leg "
                "means within 0.25 SD at n = 2,000, not unbiased. The bias was measured at "
                "n = 2,000 only. Theory makes it vanish beside the SE as n grows once the rule "
                "settles (C3a), and smaller samples lean further.",
                "The interval needs the learned rule to settle (C3a). For a fixed-form outcome "
                "model whose limit benefit is near zero, the rule changes between samples and "
                "the point leans further: with RM30's drop_w1 learner the rule flipped in "
                "19.4% of draws, and the targeted bias was +0.13 SD on settled draws, +1.07 SD "
                "on flipped ones and +0.31 SD overall, with coverage 0.938. The row covers such "
                "fits, and no check in the package detects them.",
                "The necessity learner WithoutW1MainLogistic replaced the plan's drop_w1 "
                "learner before the run, because the drop_w1 rule does not settle at this n "
                "(C3a): from the law it flips in about 17.5% of samples. Cross-fitted, the same "
                "drop_w1 learner on the same law reads -0.009 SD (learned-rule-cvtmle).",
                "Adaptive in-sample outcome learners are outside the row: they key adaptive "
                "and warn (InSampleLearnerWarning), and the interpolating control's SE ratio "
                "of 0.391, with coverage 0, shows why.",
                "One declared misspecified_limit calibration draw of 7,200 raised two lbfgs "
                "convergence warnings, in the fit and in its truth refit, and was kept. No "
                "primary fit warned.",
                "No comparator: no maintained package fits the in-sample learned-rule TMLE.",
            ],
        )
    )
    # X11 (c) (X11 PR G): the learned rule beside known regimes (``interventions=``), with each
    # known value and each contrast ``ate_learned_rule``, keyed ``known_regimes``. One row for
    # each report, because the study's declared support condition held for both: the five
    # primary tests, every ``interval_calibration``, ``conservative_coverage`` and
    # ``contrast_covariance`` verdict of that report's labels, and the ``targeting_necessity``
    # verdicts passed. A row labels every estimate of a fit, so each covers the three P11 names.
    # Weights, clusters, strata, fold targeting, a rule learner, repeats above 1, the in-sample
    # fit and known treatment probabilities key other axes or qualifiers, so each of those
    # compositions matches no row. The numbers are the committed results
    # (``tests/canonical/learned_rule_contrast``).
    contrast_limits = [
        "The study measures the pointwise intervals of the plug-in rule (LearnedRule()) beside "
        "Static(1) and Static(0) at n = 2,000 with 10 equal folds, repeats=1, pooled "
        "targeting, the RM30 interaction logistic outcome model and a correct logistic "
        "treatment model fitted by maximum likelihood, on two RM30 laws. Binary outcome and "
        "treatment, iid rows, no weights. The key holds neither n, the fold count nor the "
        "learner, so another n, V or learner reads supported and was not measured.",
        "The key also covers compositions no cell measured: reference= naming the learned "
        "rule (sign-flipped contrasts), known dynamic Rule regimes and more than two known "
        "regimes. Each rests on the unit witnesses W-c1 and W-c2 and the linearity argument, as "
        "T38.split_mean does for its repeat count.",
        "On the misspecified_limit law (conservative_coverage) the cells claim coverage of each "
        "parameter at or above 0.92 when the outcome model cannot represent the treated arm, "
        "not SE calibration, and the study does not show that these intervals are "
        "conservative. The reported curves omit the treatment model's projection term. "
        "Corollary 2 (eq. 11) of van der Laan and Luedtke (2015) makes that variance "
        "conservative for a maximum-likelihood g in a correct model, and the paper extends "
        "this to the cross-validated TMLE by analogy only; the contrasts' version is the "
        "research session's own derivation, not a statement in the paper. On this law the "
        "omitted term is negligible (predicted SE ratio at most 1.0005), so the study tests "
        "neither conservativeness nor the analogy.",
        "The known_0 cells read an outcome arm the model represents correctly (the two laws "
        "share Qbar0(0, W)), so they are not misspecified-outcome evidence.",
        "The misspecified-outcome coverage was measured only with a correct parametric g "
        "fitted by maximum likelihood. A known g (treatment_probabilities=, which keys "
        "known_mechanism and matches no row) and a super-learner g were not measured.",
        "The necessity pair reads the contrast against Static(0) only. Against Static(1) the "
        "drop_w1 learner's limit rule is Static(1), so that contrast is degenerate: it was "
        "exactly zero in 62% of the 4,000 pilot draws.",
        "Pointwise intervals: the simultaneous band of a contrast fit refuses. No comparator: "
        "no maintained package fits a cross-validated TMLE of a learned rule beside known "
        "regimes.",
        "The harness counts solver warnings: one necessity draw of 1,200 raised two and was "
        "kept, and no primary, calibration or coverage draw warned. No rule reads the counts.",
    ]
    # The realized SE ratio and coverage of each positive, in the order value, Static(1),
    # Static(0), the contrast against Static(1), the contrast against Static(0).
    contrast_measured = {
        "fold_evaluated": (
            "The non_exceptional calibration positives read SE ratios of 0.975, 0.999, 1.004, "
            "0.996 and 0.993 with coverage 0.946, 0.950, 0.956, 0.948 and 0.953 (5,600 draws). "
            "The misspecified_limit positives read SE ratios of 0.997, 1.004, 1.003, 0.996 and "
            "0.991 with coverage 0.950, 0.951, 0.951, 0.947 and 0.942 (3,200 draws). The lowest "
            "primary cell, the contrast against Static(0), read SE ratio 0.970 and coverage "
            "0.937 at 1,200 draws, inside the primary rules; its calibration cell reads 0.993 "
            "and 0.953."
        ),
        "stacked": (
            "The non_exceptional calibration positives read SE ratios of 0.975, 0.999, 1.004, "
            "0.996 and 0.993 with coverage 0.946, 0.950, 0.956, 0.948 and 0.953 (5,600 draws). "
            "The misspecified_limit positives read SE ratios of 0.997, 1.004, 1.003, 0.995 and "
            "0.991 with coverage 0.950, 0.951, 0.951, 0.947 and 0.942 (3,200 draws). The "
            "stacked point equals the fold-evaluated one at equal folds without weights, as in "
            "every draw here, so the primary and necessity verdicts read it too. Without weights "
            "or clusters the folds of a covered fit differ by a few rows, and only the variance "
            "rule differs."
        ),
    }
    for report in ("fold_evaluated", "stacked"):
        rows.append(
            point_row(
                f"T44.{report}",
                ["learned-rule-contrast"],
                families=("P11",),
                splitting=(report,),
                qualifiers=["known_regimes"],
                cells=["non_exceptional", "misspecified_limit"],
                # Roadmap: X11.
                limits=[contrast_measured[report], *contrast_limits],
            )
        )
    # X11 (d) (X11 PR H): the plug-in rule over three arms, the argmax over every arm with a tie
    # to the lowest arm code, keyed ``treatment=("arms3",)``. One row, for the fold-evaluated
    # report on rows, because the study's declared support condition held: the primary bias
    # leg, coverage floor and SE sanity, every ``interval_calibration`` verdict, both
    # ``conservative_coverage`` verdicts and the ``targeting_necessity`` verdicts passed. Four or
    # more arms key ``arms4`` and above, and weights, clusters, strata, fold targeting, repeats
    # above 1, the stacked report, the in-sample fit and known regimes beside the rule key other
    # axes or qualifiers, so each of those compositions matches no row. The numbers are the
    # committed results (``tests/canonical/learned_rule_arms``).
    rows.append(
        point_row(
            "T45",
            ["learned-rule-arms"],
            families=("P11",),
            treatment=("arms3",),
            splitting=("fold_evaluated",),
            cells=["three_arms", "missing_quadratic"],
            # Roadmap: X11.
            limits=[
                "The study measures the pointwise interval of the plug-in rule (LearnedRule()) "
                "over three arms at n = 2,000 on one law, with 10 equal folds, repeats=1, pooled "
                "targeting, an arm-interacted logistic outcome model and a correct multinomial "
                "logistic treatment model fitted by maximum likelihood. Binary outcome, iid "
                "rows, no weights. The key holds neither n, the fold count nor the learner, so "
                "another n, V or learner reads supported and was not measured.",
                "With the correct outcome model the calibration positive reads SE ratio 0.986 "
                "and coverage 0.9425 (3,200 draws), and the primary cell 0.997 and 0.947 (1,200 "
                "draws).",
                "The margin condition holds at both outcome learners' limits: every arm's share "
                "is at least 0.175 and 0.2455, and the top two arms lie within 0.01 of each "
                "other on 1.6% and 1.8% of the population. Over the draws the fitted rules "
                "differ from the limit rule on 3.6% and 3.8% of rows on average, the smallest "
                "arm share averages 0.177 and 0.245, and no draw leaves an arm without rows. "
                "The no-tie form of C3 for the argmax is the package's own step.",
                "When arm 2's outcome regression omits its quadratic term (arms 0 and 1 are "
                "exact at the limit), the cells claim coverage at or above 0.92, not SE "
                "calibration: coverage 0.940 at SE ratio 0.996 (1,200 draws). The reported "
                "curve omits the treatment model's projection term, which Corollary 2 of van "
                "der Laan and Luedtke (2015) makes conservative for a maximum-likelihood g in a "
                "correct model, extended to the cross-validated TMLE by analogy only. On this "
                "law the predicted SE ratio of that term is 1.0004, so the study does not test "
                "conservativeness.",
                "The necessity pair (an outcome model without W1) reads bias only. Its fold "
                "rules have not settled at n = 2,000, so its targeted SE ratio 0.948 and "
                "coverage 0.932 are a weak-margin reading, not calibration evidence or valid "
                "inference under a misspecified outcome model.",
                "The coverage was measured only with a correct parametric g fitted by maximum "
                "likelihood. A known g (treatment_probabilities=, which keys known_mechanism "
                "and matches no row) and a super-learner g were not measured.",
                "The harness counts solver warnings: one necessity draw of 1,200 raised two and "
                "was kept, and no primary, calibration or coverage draw warned. No rule reads "
                "the counts. No comparator: no maintained package fits a cross-validated TMLE "
                "of a learned rule over more than two arms.",
            ],
        )
    )
    # X11 (f) (X11 PR I): the value of the optimal rule, the fixed parameter Psi_d0, read from the
    # fold-evaluated CV-TMLE of the learned-rule value, keyed family ``P12``. One row, for the
    # fold-evaluated report on rows with the known treatment mechanism, because the study's
    # declared support condition held: the primary bias leg, coverage floor and SE sanity, every
    # ``interval_calibration`` verdict and the ``optimal_target_control`` verdict passed. An
    # estimated g drops ``known_mechanism``, and repeats above 1, clusters, fold targeting, a
    # rule learner, the stacked report and more than two arms key other axes or qualifiers, so
    # each of those compositions matches no row. The numbers are the committed results
    # (``tests/canonical/learned_rule_optimal_value``).
    rows.append(
        point_row(
            "T46",
            ["learned-rule-optimal-value"],
            families=("P12",),
            splitting=("fold_evaluated",),
            qualifiers=("known_mechanism",),
            cells=["non_exceptional", "wrong_limit"],
            # Roadmap: X11.
            limits=[
                "The study measures the pointwise interval of the value of the optimal rule "
                "(OptimalRuleValue(), estimands=['ey_optimal_rule']) at n = 2,000 on one law: a "
                "binary treatment and outcome, ten equal folds, pooled targeting, the plug-in "
                "rule, repeats=1, iid rows, no weights, strata or known regimes. The treatment "
                "mechanism is known (treatment_probabilities=). The outcome model is a "
                "correctly specified parametric logistic model, so its limit rule is the "
                "optimal rule, and the blip 0.1 + W1 has no tie set and a bounded density at 0 "
                "(margin exponent 1, verified analytically). The key sees neither the learner, "
                "the law nor n, so a misspecified or flexible learner, a law with ties, a "
                "steeper margin or another n reads supported and was not measured.",
                "The interval covers the optimal value only up to the regret, the gap between "
                "the optimal value and the value of the fold rules, which is O_P(1/n) under "
                "these conditions. The calibration cell's mean regret is 0.00138, 0.09 "
                "estimate SDs: against the optimal value it reads standardized bias -0.088, SE "
                "ratio 0.997 and coverage 0.949 (13,200 draws), and against the realized rule "
                "value +0.004, 1.006 and 0.952. The primary bias interval, -0.0026 to -0.0007 "
                "(1,600 draws), excludes zero because of the regret, and the regret's spread "
                "lowers the SE ratio against the optimal value by about 0.9%. One n is "
                "measured, so the rate is not shown.",
                "The target control runs at n = 2,500, not 2,000. With a learner whose limit "
                "rule treats nobody while the optimal rule treats 29.3%, the interval covers the "
                "value of the learner's limit rule and misses the optimal value: coverage 0.09 "
                "of the optimal value, 0.958 of the realized rule value, mean regret 0.0450.",
                "An estimated g is a reading, not a claim, and has no row: at 1,200 draws its "
                "bias interval is -0.0026 to -0.0004, its SE ratio 0.975 and its coverage 0.948 "
                "(99% interval 0.930 to 0.963). Such a fit drops known_mechanism and reads "
                "experimental.",
                "On a law whose blip is zero on 60% of the population no claim is made. The "
                "interval under-covers there, coverage 0.912 with SE ratio 0.888 against the "
                "optimal value and 0.919 with 0.893 against the realized rule value, because "
                "the fold rules do not settle on the tie set (disagreement 0 to 0.735, mean "
                "0.280), not because of the regret (mean 0.00086).",
                "A rule learner, clusters, repeats above 1, fold targeting, the stacked report, "
                "more than two arms and an estimated g stay experimental. Weights, strata, the "
                "in-sample fit and contrasts with known regimes refuse. No comparator: no "
                "maintained package fits a cross-validated TMLE of the value of the optimal "
                "rule. No draw raised a solver warning.",
            ],
        )
    )
    # X11 (j) (X11 PR J): the plug-in rule at A with a binary intermediate held at each level
    # (``intermediate=``), keyed ``missing=("intermediate",)``, and a known regime through the
    # same fold-evaluated controlled-direct-effect path. Two rows, for the fold-evaluated report
    # on rows, because the study's declared support conditions held: for T47 the primary bias
    # leg, coverage floor and SE sanity at both levels, every learned ``interval_calibration``
    # verdict and the ``targeting_necessity`` verdicts; for T48 every known-rule
    # ``interval_calibration`` verdict. Weights, clusters, strata, fold targeting, repeats above
    # 1, the stacked and in-sample reports, more than two arms, a known g and a rule learner key
    # other axes or qualifiers or refuse, so each of those compositions matches no row. The
    # numbers are the committed results (``tests/canonical/learned_rule_intermediate``).
    intermediate_limits = [
        "The study measures one law at n = 2,000: a binary treatment, outcome and intermediate, "
        "ten equal folds, pooled targeting, repeats=1, iid rows, and no weights, clusters, "
        "strata, fold targeting, stacked report or contrasts. The outcome, treatment and "
        "intermediate models are correctly specified parametric logistic models, fitted in "
        "every cell. The key holds neither the learner, the law nor n, so a flexible learner, "
        "another law or another n reads supported and rests on the theory alone.",
        "Positivity: the smallest g0 q0 on the support is 0.045, and along the learned rules it "
        "is 0.11 (z = 0) and 0.19 (z = 1). A law with q0(z | a, W) nearer 0 is not measured.",
        "A rule learner with intermediate= refuses (X11 (j2)), and so do a non-binary "
        "intermediate (X63), strata under cv_evaluation=True (X28), the omitted-variable bound "
        "(X11 (n2)) and the placebo of a level in refute. Weights, clusters, fold targeting, "
        "repeats, the stacked and in-sample reports, three or more arms and a known g stay "
        "experimental. No comparator: no maintained package fits a learned rule with an "
        "intermediate held fixed. No draw failed or raised a solver warning.",
    ]
    rows.append(
        point_row(
            "T47",
            ["learned-rule-intermediate"],
            families=("P11",),
            missing=("intermediate",),
            splitting=("fold_evaluated",),
            cells=["cde_z0", "cde_z1", "learned_z0", "learned_z1", "drop_w1_z0", "drop_w1_z1"],
            # Roadmap: X11.
            limits=[
                "The study measures the pointwise interval of the plug-in rule (LearnedRule()) "
                "fitted with intermediate=, one result per level of the intermediate. Each "
                "level's interval targets the value of that level's own fold rules, V^-1 sum_v "
                "E0 Qbar0(d^z_nv(W), z, W). That is neither the value of the optimal rule at "
                "that level nor the jointly optimal pair of treatment and intermediate. The "
                "optimal rules at the two levels differ on 30% of the population.",
                intermediate_limits[0],
                "The calibration positive reads SE ratio 0.998 (99% interval 0.985 to 1.010) "
                "and coverage 0.949 at z = 0, and 0.983 (0.971 to 0.996) and 0.946 at z = 1 "
                "(20,000 draws). The primary cells read coverage 0.946 and 0.931 and SE ratio "
                "1.002 and 0.977 (1,200 draws). Plug-in learned rules read slightly below 1 at "
                "n = 2,000, the between-fold covariance that T39 describes.",
                "The control without the factor 1{Z = z} / q of the clever covariate reads SE "
                "ratio 0.591 and 0.731, so the instrument resolves a variance missing that "
                "factor. With an outcome model that omits W1, a confounder of A, Z and Y, the "
                "untargeted plug-in is biased by +0.099 and -0.024 and targeting removes it; its "
                "limit rules treat everyone at z = 0 and nobody at z = 1, so the pair tests a "
                "gross misspecification, not a subtle one.",
                intermediate_limits[1],
                intermediate_limits[2],
            ],
        )
    )
    rows.append(
        point_row(
            "T48",
            ["learned-rule-intermediate"],
            families=("P5",),
            missing=("intermediate",),
            splitting=("fold_evaluated",),
            cells=["known_z0", "known_z1"],
            # Roadmap: X11.
            limits=[
                "The study measures one known rule, 1{W1 > 0}, and only its value, through the "
                "fold-evaluated controlled-direct-effect path at each level of the "
                "intermediate. The key also covers static regimes, contrasts between known "
                "regimes and any deterministic rule with a binary intermediate on that path, "
                "and none of those is measured. This is the measured path the learned fit "
                "reduces to bit for bit at a constant rule (W-j1). T30 (in sample, missing "
                "outcome) is a different configuration.",
                intermediate_limits[0],
                "The calibration positive reads SE ratio 1.007 (99% interval 0.990 to 1.025) "
                "and coverage 0.952 at z = 0, and 0.997 (0.980 to 1.015) and 0.949 at z = 1 "
                "(10,000 draws). The control without the factor 1{Z = z} / q reads SE ratio "
                "0.607 and 0.770.",
                "Positivity along the known rule: the smallest g0 q0 is 0.11 (z = 0) and 0.21 "
                "(z = 1).",
                intermediate_limits[2],
            ],
        )
    )
    # P1 (c): the learned-rule value when the EP curve sets each fold's rule
    # (``LearnedRule(loss="ep", modifiers=...)``), keyed ``rule_learner_ep``. The plan named these
    # rows T49; the conditional-projection rows took T49 and T50 first. Two rows, for the
    # fold-evaluated report on rows, because every declared verdict of ``learned-rule-ep``
    # passed: the primary bias leg, coverage floor and SE sanity at both laws, every
    # ``interval_calibration`` positive at n = 16,000, the shrunken-SE and noise controls, and the
    # fold-locality control. The sizing reading admitted the rows (``T49_ADMITTED`` of
    # ``tests/studies/learned_rule_ep_properties.py``), and every row carries its ``T49_LIMITS``
    # sentence. The support key holds neither the sieve nor the final learner, so
    # ``ep_rule_condition`` of ``cleverly.interventions.learned`` labels a declared one
    # experimental. Weights, clusters, fold targeting, repeats above 1, known regimes, a
    # continuous outcome and an estimated g for the optimal value key other axes or qualifiers,
    # and three or more arms, an intermediate and strata refuse. The numbers are the committed
    # results (``tests/canonical/learned_rule_ep``).
    ep_limits = [
        "The study measures LearnedRule(loss='ep', modifiers=Modifiers(['W1', 'W2'], ...)) with "
        "sieve=None, the default sieve of six cosines of W1 plus the indicator of W2 with its "
        "size chosen by cross-validation over the inner folds, and the least-squares final "
        "stage, on unweighted rows with ten folds, pooled targeting, repeats=1 and no id, at "
        "two RM30 laws with the main-terms logistic outcome learner and a correct logistic "
        "treatment learner. A declared sieve or final learner reads experimental. The key holds "
        "neither the modifiers, the law, the learners nor n, so another modifier set, law, "
        "learner or n reads supported and rests on the theory alone.",
        "Calibrated at n = 16,000 at these laws; at n = 4,000 the sizing pilot read an SE ratio "
        "of 0.970 and coverage 0.9365, about 6.5% excess error variance from learning the "
        "rule. The size moved from 4,000 to 16,000 after that pilot and before any declared "
        "draw.",
        "The exceptional law, where the effect is zero everywhere, under-covers: at n = 2,000 "
        "the coverage is 0.901 (99% interval 0.885 to 0.916) and the SE ratio 0.833. The fold "
        "curves have no fixed sign there, so condition C3 fails, and no check in the package "
        "detects such a law. The row makes no claim there.",
        "The fold-locality control, each fold's EP curve fitted on its own validation rows, "
        "covers 0.008 with a bias of 4.6 error SDs, so the instrument resolves a gross leak. "
        "The subtle leaks, the outer nuisances at the training rows and a size step that sees "
        "its inner fold, rest on the unit witnesses W-c2 and W-c3. No comparator: no maintained "
        "package fits a cross-validated TMLE of the value of an EP rule.",
    ]
    rows.append(
        point_row(
            "T51",
            ["learned-rule-ep"],
            families=("P11",),
            splitting=("fold_evaluated",),
            qualifiers=["rule_learner_ep"],
            cells=[
                "non_exceptional",
                "misspecified_limit",
                "non_exceptional__declared",
                "misspecified_limit__declared",
            ],
            # Roadmap: P1.
            limits=[
                "The target is the fold average of the value of the rules the fit learned, not "
                "the value of the optimal rule. Only the T51.optimal cell sees whether the EP "
                "rule is a good rule: a learner that treats nobody still covers its own value.",
                *ep_limits[:2],
                "At n = 16,000 the SE ratio is 0.999 (99% interval 0.978 to 1.020) at "
                "non_exceptional and 1.008 (0.988 to 1.030) at misspecified_limit, with "
                "coverage 0.951 and 0.954 (7,200 draws each). At n = 2,000 it is 0.994 and "
                "0.983 with coverage 0.951 and 0.944, and at n = 1,000 it is 0.987 and 0.911 "
                "with coverage 0.948 and 0.918 (2,000 draws each), so at n = 1,000 at the "
                "misspecified law the SE is about 9% small. The key holds no sample size, so a "
                "smaller fit also reads supported.",
                *ep_limits[2:],
                "The value gain over the plug-in rule is descriptive and carries no test.",
            ],
        )
    )
    rows.append(
        point_row(
            "T51.optimal",
            ["learned-rule-ep"],
            families=("P12",),
            splitting=("fold_evaluated",),
            qualifiers=["known_mechanism", "rule_learner_ep"],
            cells=["optimal_known_g__declared"],
            # Roadmap: P1.
            limits=[
                "The study measures the value of the optimal rule (estimands=['ey_optimal_rule']) "
                "with the EP rule given both adjustment columns and the treatment mechanism "
                "known (treatment_probabilities=), at the non_exceptional law only, at "
                "n = 16,000. An estimated g drops known_mechanism and reads experimental.",
                *ep_limits[:2],
                "The interval covers the optimal value only up to the regret of the EP limit "
                "rule, the sign of the projection of the effect on the largest default member, "
                "which differs from the optimal rule on 0.98% of the population and gives up "
                "2.4e-5 of value, 0.005 SE at n = 16,000. The cell reads SE ratio 0.985 (99% "
                "interval 0.965 to 1.007), coverage 0.947 and standardized bias -0.070 (7,200 "
                "draws); the bias interval, -0.00053 to -0.00021, excludes zero and lies inside "
                "the margin. On a law where the sign of the effect is not that of a function in "
                "the sieve, the regret does not vanish and the interval covers the value of the "
                "limit rule instead.",
                *ep_limits[2:],
            ],
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
    # The band kind follows BAND_CELLS: the study has no band cell of the continuous or the
    # stacked three-arm configuration (mar_attributable_properties.BAND_SCENARIOS).
    for suffix, kwargs, cell, banded in (
        ("binary", {}, "binary_mar_attributable", True),
        ("continuous", {"outcome": BOUNDED}, "continuous_mar_attributable", False),
        ("three_arm", {"treatment": ("arms3",)}, "three_arm_mar_attributable", True),
        ("binary_stacked", {"splitting": ("stacked",)}, "binary_mar_attributable_cvtmle", True),
        (
            "three_arm_stacked",
            {"treatment": ("arms3",), "splitting": ("stacked",)},
            "three_arm_mar_attributable_cvtmle",
            False,
        ),
        ("weighted", {"weighting": "fixed"}, "binary_mar_attributable_weighted", True),
        ("clustered", {"dependence": "clustered"}, "binary_mar_attributable_clustered", True),
    ):
        rows.append(
            point_row(
                f"T29.{suffix}",
                ["mar-attributable-tmle"],
                families=attributable,
                missing=("outcome",),
                kinds=frozenset({"point", "wald", "band"}) if banded else _WALD,
                cells=[cell],
                **kwargs,
            )
        )
    rows.append(
        point_row(
            "T30",
            # default_bands:cde_z0 refits cde-tmle's z = 0 fit with its default band
            # (default_band_properties.SOURCES), so it measures this configuration's band.
            ["cde-tmle", "default-simultaneous-bands"],
            families=("P1", "P2"),
            missing=("outcome", "intermediate"),
            cells=["binary_cde_z0_mar", "binary_cde_z1_mar"],
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


#: The limit line of a stacked DR-TMLE row whose evidence ran the pooled reduced regressions.
LEAK = (
    "Pooled reduced regressions: R drtmle cvFolds parity. The drtmle-foldwise study "
    "measured the fold leak on shared draws at n = 1,000: ATE bias 0.400 SD pooled against "
    "0.156 SD foldwise, a paired displacement of 0.266 SD. Row D11 covers the binary "
    "unstratified foldwise default."
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
            # Roadmap: RM18.
            limits=[
                "The one-inconsistent-nuisance cells are red.",
                LEAK,
                "The red one-inconsistent-nuisance cells fit the pooled reduced regressions. "
                "The foldwise default passes on their own draws at n = 1,500 "
                "(drtmle-foldwise-n1500), and in drtmle-foldwise for treatment_correct at "
                "n = 1,000 and for both regimes at n = 3,000; row D11 covers it.",
            ],
        ),
        point_row(
            "D02",
            ["canonical-multi-arm-drtmle"],
            families=("P1", "P2"),
            engine=dr,
            treatment=("arms3",),
            splitting=("stacked",),
            cells=["multi_arm_binary_drtmle"],
            # Roadmap: RM18.
            limits=[
                "The boundary and one-sided-bias cells are red.",
                LEAK,
                "The red treatment_correct cell fits the pooled reduced regressions. The "
                "foldwise default passes on its draws at n = 2,000 (drtmle-foldwise) and on the "
                "primary law and the outcome_correct cell (multi-arm-drtmle-foldwise); row D12 "
                "covers it.",
            ],
        ),
        point_row(
            "D03",
            ["canonical-stratified-drtmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            qualifiers=["strata"],
            cells=["stratified_both_correct"],
            # Roadmap: X8.
            limits=[
                "Two truth cells and the small-stratum cell are red.",
                LEAK,
                "The red treatment_correct cells fit the pooled reduced regressions. Under the "
                "foldwise default the marginal ATE bias reads 0.40, 0.27 and 0.20 SD at n = "
                "2,000, 4,000 and 8,000 and contracts faster than n^-1/2, consistent with second "
                "order (log-bias slope -1.07, 99% interval -1.22 to -0.93; "
                "stratified-drtmle-foldwise); row D13 covers it.",
            ],
        ),
        point_row(
            "D04",
            ["clustered-unequal-cvtmle"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            dependence="clustered",
            options={"reduced_crossfit": "foldwise"},
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
            options={"reduced_crossfit": "foldwise"},
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
            options={"reduced_crossfit": "foldwise"},
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
        # X83: the default cross-fitted fit, foldwise reduced regressions on the
        # canonical-drtmle law. Its key is the key of D07.estimated_g_stacked, so the two merge.
        # Fix-6 adds the default at n = 1,500 on the RM18 cells' draws.
        point_row(
            "D11",
            ["drtmle-foldwise", "drtmle-foldwise-n1500"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            options={"reduced_crossfit": "foldwise"},
            cells=["both_correct", "outcome_correct", "treatment_correct"],
        ),
        # Fix-6: the default cross-fitted three-arm fit, foldwise reduced regressions on the
        # canonical-multi-arm-drtmle law, and the X83 multi-arm treatment_correct cell.
        point_row(
            "D12",
            ["multi-arm-drtmle-foldwise", "drtmle-foldwise"],
            families=("P1", "P2"),
            engine=dr,
            treatment=("arms3",),
            splitting=("stacked",),
            options={"reduced_crossfit": "foldwise"},
            cells=[
                "multi_arm_binary_drtmle",
                "double_robustness/outcome_correct",
                "double_robustness/multi_arm_drtmle__treatment_correct",
            ],
            limits=[
                "The foldwise outcome_correct cell runs at n = 1,000 only "
                "(multi-arm-drtmle-foldwise); no foldwise multi-arm outcome_correct cell runs at "
                "n = 2,000, the size of the primary rows and of the treatment_correct cell "
                "(drtmle-foldwise)."
            ],
        ),
        # Fix-6: the default cross-fitted stratified fit, foldwise reduced regressions on the
        # canonical-stratified-drtmle law, and the X83 stratified outcome_correct cells.
        point_row(
            "D13",
            ["stratified-drtmle-foldwise", "drtmle-foldwise"],
            families=("P1",),
            engine=dr,
            splitting=("stacked",),
            qualifiers=["strata"],
            options={"reduced_crossfit": "foldwise"},
            cells=[
                "stratified_both_correct",
                "stratified_contraction",
                "double_robustness/marginal_ate__stratified",
                "double_robustness/v0_ate__stratified",
                "double_robustness/v1_ate__stratified",
                "double_robustness/v2_ate__stratified",
            ],
            limits=[
                "The smallest-stratum truth cell ey[1][V=2] is red and a defect suspect "
                "(X8-drtmle-small-stratum): its SE is under-reported by about 12% (SE ratio "
                "0.882, 99% interval 0.858 to 0.908) and its coverage is 0.907, pending Fix-3's "
                "small-stratum correction.",
                "The treatment_correct bias is a finite-sample limit: the marginal ATE reads "
                "0.40, 0.27 and 0.20 SD at n = 2,000, 4,000 and 8,000, and the marginal and v1 "
                "biases contract faster than n^-1/2, consistent with second order "
                "(stratified-drtmle-foldwise; X8-drtmle-one-sided-bias).",
            ],
        ),
    ]
    return _merged(
        rows,
        {
            "D06.qg": ["D06.qg", "D07.known_g"],
            "D11": ["D11", "D07.estimated_g_stacked"],
        },
    )


#: The limit of ``G31.isotonic``: what the binding projection was measured to do (study
#: ``isotonic-binding``, Fix-8 plan 3.8).
ISOTONIC_BINDING_LIMIT: Final[str] = (
    "On the isotonic-binding law (two nodes, a continuous baseline covariate, a strictly "
    "increasing risk) the projection moved the never regimen's horizon-1 level on 0.354 of "
    "draws, by at least 0.1 SE on 0.193 and by at most 0.564 SE, and the interval at the "
    "projected level read coverage 0.942 (SE ratio 0.98). The projected and unprojected "
    "intervals disagreed on 0.0064 of draws, so the projection changed coverage by at most that "
    "(difference +0.0008, 99% interval -0.0027 to +0.0040). On the larger-move law (treatment "
    "slope 2.0) the moves reached 0.994 SE and the coverage difference read -0.0033 (99% "
    "interval -0.0081 to +0.0021), but there the unprojected interval itself under-covers "
    "(coverage 0.938, SE ratio 0.95). The moves arise because the two horizons' node-1 "
    "regressions use different learner classes; a flat truth and other learner pairings are "
    "untested."
)


#: The limit of ``C01.greedy`` and ``C12.greedy``: the greedy point estimate on a law where
#: both selected nuisances are wrong (study ``option-patience``, Fix-8 plan 2.3).
GREEDY_LATE_CONFOUNDER_LIMIT: Final[str] = (
    "On a law where the outcome regression omits a confounder and the selected treatment model "
    "is misspecified given the selected set, the greedy point estimate keeps a bias (study "
    "option-patience, greedy whole path on LC3: 0.42 SD at n = 2,000)."
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
                # Roadmap: F18.
                limits=[
                    "The selector plug-in interval is withheld.",
                    *([GREEDY_LATE_CONFOUNDER_LIMIT] if strategy == "greedy" else []),
                ],
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
                # Roadmap: F18.
                limits=["The selector plug-in interval is withheld."],
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
            # Roadmap: F19.
            limits=["The generated-design interval is withheld."],
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
        # The R ctmle type-3 comparison covers this row's fit: binary, two-arm, complete-outcome,
        # in-sample ATE.  Its two GLM scenarios are equivalent to R.
        point_row(
            "C10.in_sample",
            [study],
            families=("P1",),
            engine=derivative,
            cells=["in_sample", "binary_declared_pair", "binary_pipeline"],
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
    ("T16.logit_binary", "wald"): (
        "Row T16.logit_binary is red for the logit slope at the smallest stratum: coverage "
        # Roadmap: X8.
        "0.9289 to 0.9560 and SE ratio 0.7818 to 0.9387."
    ),
    ("T16.dose_logit", "wald"): (
        "Row T16.dose_logit is red for the logit slope at the smallest stratum: coverage "
        # Roadmap: X8.
        "0.9289 to 0.9560 and SE ratio 0.7818 to 0.9387."
    ),
    # X40: the owner chose this label for the four red price_down cells of the five-node price
    # study (2026-10-16). To flip it to a stated limit, move each sentence to MEASURED_LIMITS
    # under the same row id, and change the four dispositions of ``mtp-pricing`` in
    # ``tests/studies/evidence/support_dispositions.py`` (see ``_X40_PRICE_DOWN`` there).
    ("D13", "wald"): (
        "Row D13 is red for the smallest-stratum treated mean, a defect suspect pending Fix-3's "
        # Roadmap: X8-drtmle-small-stratum.
        "small-stratum correction: coverage 0.9073 and SE ratio 0.8820 (99% interval 0.8584 to "
        "0.9079)."
    ),
    ("G39.ltmle.tau5", "wald"): (
        "Row G39.ltmle.tau5 is red for the price_down plan at five nodes and n = 5,000: "
        "coverage 0.9160 to 0.9170 and SE ratio 0.9310 to 0.9479 (mtp-pricing-calibration)."
    ),
    ("G39.sdr.tau5", "wald"): (
        "Row G39.sdr.tau5 is red for the price_down plan at five nodes and n = 5,000: "
        "coverage 0.9190 to 0.9230 and SE ratio 0.9339 to 0.9518 (mtp-pricing-calibration)."
    ),
    # band-finite-sample: default_bands:categorical_ltmle measures this configuration's default
    # band; stochastic-categorical-ltmle's primary band cell, which BAND_CELLS maps here, passes.
    # A band shortfall binds the default band only: the t band has cells of its own
    # (T_BAND_CELLS), so it does not read the sentence.
    ("G08.in_sample", "band"): (
        "Row G08.in_sample is red for the default band of its configuration: joint coverage "
        "0.9191 to 0.9458 (default-simultaneous-bands, categorical_ltmle; band-finite-sample). "
        "A plain band with band_reference='t' is supported on this row (band-t-reference)."
    ),
    # band-finite-sample: default_bands:cde_z0 measures this configuration's default band. T30
    # licenses the t band alone (T_BAND_CELLS).
    ("T30", "band"): (
        "Row T30 is red for the default band of its configuration at z = 0: joint coverage "
        "0.9119 to 0.9397 (default-simultaneous-bands, cde_z0; band-finite-sample). A plain band "
        "with band_reference='t' is supported on this row (band-t-reference)."
    ),
    # strata-boundary-mean: canonical-stratified-tmle's strata band cell measures this
    # configuration's default band. T13 licenses the t band alone (T_BAND_CELLS).
    ("T13", "band"): (
        "Row T13 is red for the default band of its configuration: joint coverage 0.9196 to "
        "0.9462 (canonical-stratified-tmle, strata; strata-boundary-mean). A plain band with "
        "band_reference='t' is supported on this row (band-t-reference)."
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
    "D01": (
        "RM18-boundary: coverage 0.8864 to 0.9385, bias 0.0115",
        "RM18-one-sided-bias: bias 0.0026 to 0.0075, margin 0.0067, SE ratio 0.9888",
        "RM18-one-sided-bias: bias 0.0068 to 0.0121, margin 0.0072, SE ratio 0.9671",
    ),
    "D02": (
        "RM18-boundary: coverage 0.8988 to 0.9542, bias 0.000900",
        "RM18-one-sided-bias: bias 0.0015 to 0.0072, margin 0.0068, SE ratio 0.9998",
        "RM18-boundary: coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123",
        "RM18-boundary: bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888",
    ),
    "D03": (
        "X8-drtmle-small-stratum: bias 0.0023 to 0.0112, coverage 0.9000, SE ratio 0.9388",
        "X8-drtmle-one-sided-bias: bias 0.0136 to 0.0185, margin 0.0067, SE ratio 0.9902",
        "X8-drtmle-one-sided-bias: bias 0.0094 to 0.0162, margin 0.0093, SE ratio 0.9505",
        "X8-drtmle-one-sided-bias: bias 0.0123 to 0.0214, margin 0.0124, SE ratio 0.9364",
        "X8-drtmle-one-sided-bias: bias 0.0179 to 0.0286, margin 0.0147, SE ratio 1.0627",
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
    "G10.continuous": (
        (
            "mtp-longitudinal-limits: coverage 0.9289 to 0.9560, SE ratio 0.9432 to 1.0214, "
            "empirical efficiency ratio 1.0746 to 1.1643, reported efficiency ratio 1.0948 to"
            " 1.1006"
        ),
        "mtp-pricing-learned-density: coverage 0.9530 to 0.9747, SE ratio 1.0265 to 1.1162",
    ),
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
            "X39-below-tail-size: the Wald interval is validated only at n of at least twice the "
            "exact-law tail size of the quantity. At n = 2,000, below the tail size 2,326 of "
            "ate_regimen[always vs never], coverage is 0.911; bias 0.000011 to 0.0112, SE ratio "
            "0.9459"
        ),
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
    "AI.points.cross_fit": (
        "adaptive-iptw-finite-sample: bias 0.000882 to 0.0046, margin 0.0044",
        "adaptive-iptw-finite-sample: bias -0.0032 to -0.000466, margin 0.0033",
        "adaptive-iptw-finite-sample: bias -0.000589 to 0.0024, margin 0.0035",
    ),
    "AI.points_composite_three_arm.cross_fit": (
        "adaptive-iptw-composite-l3: bias -0.0075 to -0.000620, margin 0.0066",
    ),
    "AI.known_clustered": (
        "adaptive-iptw-finite-sample: coverage 0.9394 to 0.9643, SE ratio 0.9902 to 1.0722",
    ),
    "AI.points_missing_outcome.cross_fit": (
        "adaptive-iptw-finite-sample: bias -0.0031 to 0.0014, margin 0.0043",
    ),
    "AI.points_regime": (
        "adaptive-iptw-finite-sample: bias -0.000687 to 0.0037, margin 0.0043",
        "adaptive-iptw-finite-sample: bias -0.000669 to 0.0029, margin 0.0034",
    ),
    "AI.points_regime_missing": (
        "adaptive-iptw-finite-sample: bias -0.0025 to 0.0010, margin 0.0048",
    ),
}


#: The multiplier draws a registered band cell measured unless its study pins fewer: the default
#: ``n_multiplier`` since band-finite-sample (``tests/unit/test_band_defaults.py`` ties the two).
DEFAULT_BAND_DRAWS: Final[int] = 10_000

#: The rows whose band cells measured fewer than :data:`DEFAULT_BAND_DRAWS` draws, by row id.  A
#: band with :data:`~cleverly._support.DEFAULT_BAND_OPTION` and fewer draws than its row's count
#: is experimental.
#:
#: Every row in :data:`BAND_CELLS` lists :data:`~cleverly._support.DEFAULT_BAND_OPTION`, the
#: configuration band-finite-sample Wave 2 regenerated every band cell with (Mammen, logit), and
#: every cell :data:`BAND_CELLS` maps to such a row passes on the regenerated rows (the Wave 2
#: decision record).  A red cell that measures a row's default band without licensing it still
#: binds the row as a band shortfall (:data:`SHORTFALLS`), which a listed ``t`` band does not
#: read: ``G08.in_sample`` and ``T30`` carry the ``categorical_ltmle`` and ``cde_z0`` readings
#: of default-simultaneous-bands, and ``T13`` the ``strata`` reading of
#: canonical-stratified-tmle.  ``T30`` and ``T13`` license the ``t`` band alone
#: (:data:`T_BAND_CELLS`).  Any other multiplier or scale reads experimental with
#: :data:`~cleverly._support.UNMEASURED_BAND_OPTION`.
BAND_OPTION_DRAWS: Final[Mapping[str, int]] = {
    # multi-arm-mar-drtmle keeps N_MULTIPLIER = 1_000 (canonical_multi_arm_mar_drtmle.py, a
    # fixed design), so its arms band cell measured the Mammen logit band at 1,000 draws.
    "D09": 1_000,
}


def _band(study: str, setting: str) -> str:
    return f"{study}:simultaneous_coverage/{setting}__simultaneous_band"


def _t_band(study: str, setting: str) -> str:
    return f"{study}:simultaneous_coverage/{setting}__t_band"


#: The study whose cells license :data:`~cleverly._support.T_BAND_OPTION`.
T_BAND_STUDY: Final[str] = "band-t-reference"

#: The ``band-t-reference`` cells that license :data:`~cleverly._support.T_BAND_OPTION` on each
#: row, written by :func:`_t_band` (Fix-2 plan, section 1.1 item 4). Every cell passed its joint
#: rule, and so did its gated pointwise control; the paired gain over the normal-reference
#: control is reported and gates nothing. Each licence covers the whole row key, as a
#: :data:`BAND_CELLS` licence does. A row named here licenses ``band``, cites
#: ``band-t-reference`` and lists the ``t`` triple; a row with no :data:`BAND_CELLS` entry lists
#: the ``t`` triple alone, so its default band reads experimental. The held-out
#: ``strata_n1000`` cell of ``band-t-reference-held-out`` licenses no row.
T_BAND_CELLS: Final[Mapping[str, tuple[str, ...]]] = {
    # default_band_properties.SOURCES["categorical_ltmle"]: canonical-categorical-ltmle's
    # in-sample fit, at n = 2,000 and n = 8,000.
    "G08.in_sample": (
        _t_band("band_t_reference", "categorical_ltmle"),
        _t_band("band_t_reference", "categorical_ltmle_n8000"),
    ),
    # SOURCES["categorical_ltmle_crossfit"]: the cross-fitted categorical LTMLE fit.
    "G08.cross_fitted": (_t_band("band_t_reference", "categorical_ltmle_crossfit"),),
    # band_t_reference_properties._six_regimens: longitudinal-sdr's cross-fitted band fit.
    "G23.cross_fitted": (_t_band("band_t_reference", "six_regimens"),),
    # SOURCES["ordinary_binary"]: ordinary in-sample TMLE over the binary estimands.
    "T01.binary": (_t_band("band_t_reference", "ordinary_binary"),),
    # band_t_reference_properties._strata: canonical-stratified-tmle's in-sample fit.
    "T13": (
        _t_band("band_t_reference", "strata"),
        _t_band("band_t_reference", "strata_n8000"),
    ),
    # _strata(crossfit=True): the cross-fitted stratified fit.
    "T14": (_t_band("band_t_reference", "crossfit_strata"),),
    # SOURCES["point_msm"]: the in-sample point-treatment MSM fit.
    "T18": (_t_band("band_t_reference", "point_msm"),),
    # SOURCES["cde_z0"]: cde-tmle's z = 0 fit with missing outcomes, at n = 2,000 and 8,000.
    "T30": (
        _t_band("band_t_reference", "cde_z0"),
        _t_band("band_t_reference", "cde_z0_n8000"),
    ),
}


def _with_shortfalls(rows: Iterable[SupportRow]) -> tuple[SupportRow, ...]:
    """Attach the shortfall, limit, ``t`` band and band-draw tables to their rows.

    The tables are :data:`SHORTFALLS`, :data:`MEASURED_LIMITS`, :data:`T_BAND_CELLS` and
    :data:`BAND_OPTION_DRAWS`. A ``t`` band cell measured the band at
    :data:`DEFAULT_BAND_DRAWS` draws.
    """
    out = []
    for row in rows:
        pairs = tuple(
            (kind, text) for (row_id, kind), text in SHORTFALLS.items() if row_id == row.id
        )
        limits = tuple(dict.fromkeys((*row.limits, *MEASURED_LIMITS.get(row.id, ()))))
        if T_BAND_CELLS.get(row.id):
            options = (*row.band_options, T_BAND_OPTION) if "band" in row.kinds else ()
            row = replace(
                row,
                kinds=row.kinds | {"band"},
                evidence=tuple(dict.fromkeys((*row.evidence, T_BAND_STUDY))),
                band_options=options or (T_BAND_OPTION,),
            )
        measured = {
            DEFAULT_BAND_OPTION: BAND_OPTION_DRAWS.get(row.id, DEFAULT_BAND_DRAWS),
            T_BAND_OPTION: DEFAULT_BAND_DRAWS,
        }
        draws = (
            tuple((option, measured[option]) for option in row.band_options)
            if "band" in row.kinds
            else ()
        )
        out.append(replace(row, shortfalls=pairs, limits=limits, band_option_draws=draws))
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


# ----------------------------------------------------------- adaptive-propensity IPTW (X21)

_AI_SLUG: Final[tuple[str, ...]] = ("adaptive-iptw",)
_AI_STACKED: Final[tuple[str, ...]] = ("stacked",)

_AI_COMPARATOR: Final[str] = (
    "The study's drtmle comparator runs are pending, so the evidence is Python property "
    "evidence only."
)
_AI_LEARNERS: Final[str] = (
    "The key reads the learner families. The row covers a fixed-basis reduced regression and, "
    "under an estimated mechanism, a parametric mechanism learner. A default super learner, a "
    "tree or a boosted learner in either role matches no row."
)
_AI_PLUGIN: Final[str] = (
    "Under an estimated mechanism the row licenses the point estimate only. The plug-in "
    "standard error and interval of the fit are a diagnostic."
)
_AI_EPSILON_PATH: Final[str] = (
    "Under cross-fitting, the interval argument needs the epsilon-path condition of S5 in "
    "docs/technical-reference/adaptive-iptw.md. That condition stays open."
)


def _ai_cells(family: str, cell: str, labels: Sequence[str]) -> list[str]:
    """The cells ``<label>__<cell>`` of one property family of ``adaptive-iptw``."""
    return [f"{family}/{label}__{cell}" for label in labels]


def _ai_calibration(cell: str, labels: Sequence[str]) -> list[str]:
    return _ai_cells("interval_calibration", f"{cell}__correctly_specified", labels)


def _ai_bias(cell: str, labels: Sequence[str]) -> list[str]:
    return _ai_cells("bias", cell, labels)


def _ai_necessity(*labels: str) -> list[str]:
    return [f"targeting_necessity/{label}__targeted" for label in labels]


def _adaptive_iptw_rows() -> tuple[SupportRow, ...]:
    """The rows of the study ``adaptive-iptw`` (X21), one per configuration of its cells.

    A known mechanism, declared for the whole fit, licenses the Wald interval where the
    calibration cells pass. An estimated mechanism licenses the point only, because the
    estimator withholds the interval there. The key reads the learner families
    (:func:`cleverly._support.learner_family`). A known row needs a fixed-basis reduced
    regression, and an estimated row also needs a parametric mechanism learner, as every cell
    fitted. A default, tree or boosted learner matches no row.
    """
    engine = ("adaptive_iptw",)
    arms = ("ey0", "ey1", "ate")
    regimes = ("ey_d", "ey_s", "ey_never", "ate_d", "ate_s")
    ratios = ("rr", "or")
    known_options = {"mechanism_learner": "known", "reduced_learner": "fixed_basis"}
    known: dict[str, Any] = {"engine": engine, "splitting": _AI_STACKED, "options": known_options}
    one_step: dict[str, Any] = known | {"options": known_options | {"form": "one_step"}}
    known_limits = [_AI_COMPARATOR, _AI_LEARNERS, _AI_EPSILON_PATH]
    points: dict[str, Any] = {
        "engine": engine,
        "kinds": _POINT,
        "options": {"mechanism_learner": "parametric", "reduced_learner": "fixed_basis"},
        "limits": [_AI_COMPARATOR, _AI_LEARNERS, _AI_PLUGIN],
    }
    cross: dict[str, Any] = points | {
        "splitting": _AI_STACKED,
        "limits": [*points["limits"], _AI_EPSILON_PATH],
    }

    rows = [
        point_row(
            "AI.known_arms.targeted",
            _AI_SLUG,
            **known,
            families=("P1", "P2"),
            qualifiers=("known_mechanism",),
            kinds=frozenset({"point", "wald", "band"}),
            cells=[
                *_ai_calibration("known_arms", (*arms, *ratios)),
                "simultaneous_coverage/known_band__simultaneous_band",
            ],
            limits=known_limits,
        ),
        point_row(
            "AI.known_arms.one_step",
            _AI_SLUG,
            **one_step,
            families=("P1", "P2"),
            qualifiers=("known_mechanism",),
            cells=_ai_calibration("one_step_known_arms", (*arms, *ratios)),
            limits=[*known_limits, "The one-step form has no simultaneous-band cell."],
        ),
        point_row(
            "AI.known_regimes.targeted",
            _AI_SLUG,
            **known,
            families=("P5", "P6"),
            qualifiers=("known_mechanism",),
            cells=_ai_calibration("known_regimes", regimes),
            limits=known_limits,
        ),
        point_row(
            "AI.known_regimes.one_step",
            _AI_SLUG,
            **one_step,
            families=("P5", "P6"),
            qualifiers=("known_mechanism",),
            cells=_ai_calibration("one_step_known_regimes", regimes),
            limits=known_limits,
        ),
        point_row(
            "AI.known_weighted",
            _AI_SLUG,
            **known,
            families=("P1",),
            weighting="fixed",
            qualifiers=("known_mechanism",),
            cells=_ai_calibration("known_weighted", arms),
            limits=known_limits,
        ),
        point_row(
            "AI.known_clustered",
            _AI_SLUG,
            **known,
            families=("P1",),
            dependence="clustered",
            qualifiers=("known_mechanism",),
            cells=_ai_calibration("known_clustered", arms),
            limits=known_limits,
        ),
        point_row(
            "AI.known_stratified",
            _AI_SLUG,
            **known,
            families=("P1",),
            qualifiers=("known_mechanism", "strata"),
            cells=_ai_calibration("known_stratified", ("ate_v0", "ate_v1", "ate_v2")),
            limits=known_limits,
        ),
        point_row(
            "AI.points.cross_fit",
            _AI_SLUG,
            **cross,
            families=("P1",),
            cells=[
                *_ai_bias("correct", arms),
                *_ai_bias("rate", arms),
                *_ai_bias("level_set_breaking_h1", ("ey1",)),
                *_ai_bias("parametric_q_ne_qr", ("ey1",)),
                *(f"root_n_points/n_{n}" for n in (500, 1500, 4500)),
                *_ai_necessity("p_ey0", "p_ey1", "p_ate"),
            ],
        ),
        point_row(
            "AI.points.in_sample",
            _AI_SLUG,
            **points,
            families=("P1",),
            cells=_ai_bias("insample_correct", arms),
        ),
        point_row(
            "AI.points_three_arm",
            _AI_SLUG,
            **cross,
            families=("P1",),
            treatment=("arms3",),
            cells=[
                *_ai_bias("three_arm", ("ey_low", "ey_medium", "ey_high", "ate_low", "ate_medium")),
                *_ai_bias(
                    "rate_reduced", ("ey_low", "ey_medium", "ey_high", "ate_low", "ate_medium")
                ),
            ],
        ),
        merge_rows(
            "AI.points_missing_outcome.cross_fit",
            point_row(
                "AI.points_composite",
                _AI_SLUG,
                **cross,
                families=("P1",),
                missing=("outcome",),
                cells=_ai_bias("composite_l1", arms),
            ),
            point_row(
                "AI.points_natural_course",
                _AI_SLUG,
                **cross,
                families=("P1", "P4"),
                missing=("outcome",),
                cells=[
                    *_ai_bias("natural_course", ("ey0", "ey_obs", "par", "paf")),
                    *_ai_necessity("cn_ey_obs"),
                ],
            ),
        ),
        point_row(
            "AI.points_missing_outcome.in_sample",
            _AI_SLUG,
            **points,
            families=("P1",),
            missing=("outcome",),
            cells=_ai_bias("insample_composite_l1", arms),
        ),
        point_row(
            "AI.points_composite_three_arm.cross_fit",
            _AI_SLUG,
            **cross,
            families=("P1",),
            treatment=("arms3",),
            missing=("outcome",),
            cells=_ai_bias("composite_l3", ("ey_low", "ey_mid", "ey_high", "ate_low", "ate_mid")),
        ),
        point_row(
            "AI.points_composite_three_arm.in_sample",
            _AI_SLUG,
            **points,
            families=("P1",),
            treatment=("arms3",),
            missing=("outcome",),
            cells=_ai_bias(
                "insample_composite_l3", ("ey_low", "ey_mid", "ey_high", "ate_low", "ate_mid")
            ),
        ),
        point_row(
            "AI.points_weighted",
            _AI_SLUG,
            **cross,
            families=("P1",),
            weighting="fixed",
            cells=_ai_bias("weighted", arms),
        ),
        point_row(
            "AI.points_clustered",
            _AI_SLUG,
            **cross,
            families=("P1",),
            dependence="clustered",
            cells=_ai_bias("clustered", arms),
        ),
        point_row(
            "AI.points_stratified",
            _AI_SLUG,
            **cross,
            families=("P1",),
            qualifiers=("strata",),
            cells=_ai_bias("stratified", ("ate_v0", "ate_v1", "ate_v2")),
        ),
        point_row(
            "AI.points_regime",
            _AI_SLUG,
            **cross,
            families=("P5", "P6"),
            cells=[*_ai_bias("regime", regimes), *_ai_necessity("pd_d", "pd_s")],
        ),
        point_row(
            "AI.points_regime_missing",
            _AI_SLUG,
            **points,
            families=("P5", "P6"),
            missing=("outcome",),
            cells=[
                *_ai_bias("insample_regime_missing", ("ey_d", "ey_s", "ate_s_vs_d")),
                *_ai_necessity("rm_d"),
            ],
        ),
        point_row(
            "AI.points_missing_treatment.cross_fit",
            _AI_SLUG,
            **cross,
            families=("P1",),
            missing=("outcome", "treatment"),
            cells=_ai_bias("missing_treatment_crossfit", arms),
        ),
        point_row(
            "AI.points_missing_treatment.in_sample",
            _AI_SLUG,
            **points,
            families=("P1", "P5"),
            missing=("outcome", "treatment"),
            cells=[
                *_ai_bias("insample_missing_treatment", arms),
                *_ai_bias("insample_regime_missing_treatment", ("ey_d",)),
            ],
        ),
    ]
    return tuple(rows)


#: The point-treatment rows: ordinary TMLE (``T``), DR-TMLE (``D``), C-TMLE (``C``), the
#: categorical tilt (``T35``) and the adaptive-propensity IPTW (``AI``).
def _projection_rows() -> tuple[SupportRow, ...]:
    """The projection rows of ``conditional-projection`` (P1 part (a)): ``T49`` and ``T50``.

    A ``ConditionalContrast`` with a fixed basis keys family ``P13`` (the CATE projection) or
    ``P14`` (the log-CRR projection) on the effect-modification axis.  The rows were added
    because every declared verdict passed: the primary tests (bias leg, coverage floor, SE
    sanity), every ``interval_calibration`` positive and both row-level SE controls, every
    ``bias`` positive and control, and the heterogeneity test's ``type_i_error`` and ``power``
    cells.  Each row is the key of one calibration group of the study, so a composition no
    group measured, such as weights with clusters, matches no row.  The numbers are the
    committed results (``tests/canonical/conditional_projection``).
    """
    common = [
        "The study measures the pointwise Wald intervals of the coefficients of the projection "
        "on Linear(['W1', 'W3']), with continuous modifiers on [-1, 1], binary outcomes, "
        "default targeting, stacked cross-fitting and one outcome regression per arm "
        "(outcome_by_arm=True; a pooled outcome regression on this axis reads experimental). "
        "The basis, the modifiers, the law and n are not in the key, so another basis "
        "(Indicators, Cosine, BSpline, Additive, Tensor), another law or another n reads "
        "supported and rests on the theory alone.",
        "The interval covers the coefficients of the least-squares projection of the CATE on "
        "the basis over the sample's modifier distribution, not the CATE where it lies outside "
        "the span of the basis.",
        "Every calibration group fits consistent learners: a spline GLM per arm for the outcome "
        "and logistic GLMs for the treatment and observation mechanisms, and the ensemble for "
        "the weighted group. The bias ladder (both nuisances off by 5 n^(-1/3)) and two laws "
        "with a declared wrong outcome regression measure bias, not coverage; a misspecified "
        "treatment model is not measured.",
        "The simultaneous band of effect_curve(grid=) and the held-out curve checks of "
        "curve.diagnostics are not measured and stay experimental. The plug-in learner "
        "(method='ep'), model=None, strata=, an intermediate variable, missing treatment values, "
        "repeats and the bootstrap refuse. No comparator: no maintained package fits a "
        "projection TMLE of the CATE on a declared basis.",
    ]
    cells = {
        "": ["obs_1k", "obs_4k", "slow_1k", "slow_4k", "slow_16k", "noweight_dr_4k"],
        "known": ["trial_2k", "null_2k", "power_4k"],
        "arms3": ["multi_4k"],
        "weighted": ["weighted_sl_2k"],
        "clustered": ["clustered_4k"],
        "missing": ["obs_missing_4k", "missing_dr_4k"],
    }
    variants: dict[str, dict[str, Any]] = {
        "": {},
        "known": {"qualifiers": ("known_mechanism",)},
        "arms3": {"treatment": ("arms3",)},
        "weighted": {"weighting": "fixed"},
        "clustered": {"dependence": "clustered"},
        "missing": {"missing": ("outcome",)},
    }
    variant_limits = {
        "": [],
        "known": ["The trial groups declare P(A = 1) = 0.5 through treatment_probabilities=."],
        "arms3": ["The study measures three arms and the contrast of arm 2 with arm 0."],
        "weighted": [
            "The study measures one weight, 0.25 + 1.75 expit(6 W3), which moves the intercept "
            "by about half an SD, with the ensemble learner and five outer folds."
        ],
        "clustered": [
            "The study measures 800 equal clusters of 5 rows with W1 and the treatment at "
            "cluster level; the row-level SE reads ratios of 0.78 against it."
        ],
        "missing": [
            "The study measures outcomes missing at random with an observation probability in "
            "[0.60, 0.93]."
        ],
    }
    rows = []
    for family, row_id, scale_cells in (
        ("P13", "T49", None),
        ("P14", "T50", {"": ["crr_1k", "crr_4k"], "missing": ["obs_missing_crr_4k"]}),
    ):
        for suffix, settings in variants.items():
            measured = cells[suffix] if scale_cells is None else scale_cells.get(suffix)
            if not measured:
                continue
            rows.append(
                point_row(
                    row_id if not suffix else f"{row_id}.{suffix}",
                    ["conditional-projection"],
                    families=(family,),
                    splitting=("stacked",),
                    options={"outcome_by_arm": True},
                    cells=measured,
                    # Roadmap: P1.
                    limits=[*common, *variant_limits[suffix]],
                    **settings,
                )
            )
    return tuple(rows)


# The point rows of the X76 option studies, one function per part, each registered from its
# committed run: a declared stopping rule (T52), outcome_by_arm=True (T53), patience= on the
# greedy path (C12) and treatment_adjustment= (T54).
#: The limit every ``T52`` and ``G55`` row states first: what the interval is under a rule.
_RULE_INTERVAL_LIMIT = (
    "Under a declared stopping rule the interval adds the remaining score to its half-width, "
    "psi +/- (z se + |r_n|). The plain interval of the same fit has no level guarantee: where "
    "the rule binds hard it covers 0.837 and the widened one 0.977 (study "
    "option-stopping-point, cells residual_necessity, every threshold times 8). The bootstrap "
    "is refused under a rule, and a step bound or iterate='best' under a rule is not covered."
)


def _rule_newton_limit(largest: str) -> str:
    """The limit of a row whose cells read the Newton walk (plan 3.2, design review S2).

    ``largest`` is the largest absolute residual of the row's cell over its draws, in SE units.
    """
    return (
        f"On the Newton walk the rule leaves a residual of at most {largest} SE, and no draw "
        "reaches 0.01 SE (cells residual_witness), so these cells measure an estimator "
        "indistinguishable from the default: their coverage and SE ratio equal the relative "
        "fit's on the same draws."
    )


#: The share of draws whose residual reaches 0.01 SE on each one-step cell of
#: ``option-stopping-point`` (cells ``residual_witness``).
_POINT_WITNESS_SHARE = {
    ("binary", "scaled_var"): "0.854",
    ("binary", "sample_size"): "0.303",
    ("continuous", "scaled_var"): "0.999",
    ("continuous", "sample_size"): "0.751",
}

_RULE_LABELS = {"scaled_var": "sv", "sample_size": "ss"}


def _x76_stopping_point_rows() -> tuple[SupportRow, ...]:
    """A declared stopping rule on ordinary TMLE (``T52``, study ``option-stopping-point``).

    One row per rule, walk and outcome over the in-sample ``T01`` key, with the rule in the
    options and the default step bound and iterate, so a bound or ``iterate="best"`` under a
    rule matches no row.
    """
    rows = []
    for rule, label in _RULE_LABELS.items():
        for targeting, walk in (("iterative", "newton"), ("one_step", "one_step")):
            for outcome, axis in (("binary", BINARY), ("continuous", CONTINUOUS)):
                if targeting == "iterative":
                    walk_limit = _rule_newton_limit("1e-10" if outcome == "binary" else "0.0014")
                else:
                    share = _POINT_WITNESS_SHARE[(outcome, rule)]
                    walk_limit = (
                        f"On the one-step walk the rule binds: the residual reaches 0.01 SE "
                        f"on a share {share} of draws (cell "
                        f"residual_witness/{outcome}_{label}_one_step__witness)."
                    )
                rows.append(
                    point_row(
                        f"T52.{rule}.{walk}.{outcome}",
                        ["option-stopping-point"],
                        families=("P1",),
                        outcome=axis,
                        options={"stopping": rule, "targeting": targeting},
                        cells=[f"interval_calibration/{outcome}_{label}_{walk}__widened"],
                        limits=[
                            _RULE_INTERVAL_LIMIT,
                            walk_limit,
                            "Measured in sample at n = 2,000 on the canonical-tmle binary and "
                            "continuous laws, for the ATE and the arm means.",
                        ],
                    )
                )
    return tuple(rows)


#: The limit of every ``T53`` row (study ``option-outcome-by-arm``).
_BY_ARM_LIMIT = (
    "The study option-outcome-by-arm measures the interval of a fit with one main-terms "
    "regression per arm. No study cell would read differently if outcome_by_arm=True were "
    "ignored, because under the study's correct treatment mechanism the pooled fit is also "
    "unbiased (cells pooled_outcome). The evidence that each arm's regression reads only its "
    "own arm is the exact-law witness in tests/unit/test_option_outcome_by_arm_witness.py."
)


def _x76_by_arm_rows() -> tuple[SupportRow, ...]:
    """``outcome_by_arm=True`` on ordinary TMLE (``T53``, study ``option-outcome-by-arm``)."""
    slug = ["option-outcome-by-arm"]
    by_arm = {"outcome_by_arm": True}
    families = ("P1", "P2")
    return (
        *rows_over(
            point_row(
                "T53.in_sample",
                slug,
                families=families,
                options=by_arm,
                cells=[
                    "interval_calibration/in_sample_binary__correctly_specified",
                    "interval_calibration/in_sample_continuous__correctly_specified",
                    "double_robustness/treatment_correct",
                ],
                limits=[_BY_ARM_LIMIT],
            ),
            "outcome",
            {"binary": BINARY, "continuous": CONTINUOUS},
        ),
        *rows_over(
            point_row(
                "T53.stacked",
                slug,
                families=families,
                splitting=("stacked",),
                options=by_arm,
                cells=[
                    "interval_calibration/stacked_binary__correctly_specified",
                    "interval_calibration/stacked_bounded__correctly_specified",
                ],
                limits=[_BY_ARM_LIMIT],
            ),
            "outcome",
            {"binary": BINARY, "bounded": BOUNDED},
        ),
    )


def _x76_patience_rows() -> tuple[SupportRow, ...]:
    """``patience=`` on the greedy C-TMLE path (``C12``, study ``option-patience``).

    The key of ``C01.greedy`` with ``patience`` declared. A pre-ordered path with ``patience``
    keeps its permanent condition (``_option_support.patience_conditions``) and matches no row.
    """
    return (
        point_row(
            "C12.greedy",
            ["option-patience"],
            families=("P1",),
            engine=_ctmle("greedy"),
            kinds=_POINT,
            options={"patience": True},
            cells=["patience_bias/lc1_greedy", "greedy_late_confounder/lc3_greedy"],
            # Roadmap: F18.
            limits=[
                "The selector plug-in interval is withheld.",
                "Measured at patience 1 on LC1 and patience 2 on LC3. The counter stopped the "
                "greedy path early on 85% of LC1 draws, with the point estimate unbiased (cells "
                "patience_witness and patience_bias), and on LC3 the patient estimate equals the "
                "whole path's on 98% of draws (cell greedy_late_confounder).",
                GREEDY_LATE_CONFOUNDER_LIMIT,
            ],
        ),
    )


#: The limit of both ``T54`` rows (study ``option-treatment-adjustment``).
_ADJUSTMENT_LIMIT = (
    "The interval holds under the declared restriction that treatment depends on the "
    "adjustment covariates only through the subset. A regression of the treatment on the "
    "left-out covariates can check it, for the dependence that regression represents. When "
    "treatment also depends on a left-out covariate, the estimate keeps the term the "
    "restriction hides: with an outcome regression that converges slower than root-n its "
    "standardized bias grows with n, to 1.88 SD at n = 16,000 (study "
    "option-treatment-adjustment, cell instrument_slow_outcome), and with a correct "
    "parametric outcome regression the interval under-covers (cell instrument_parametric: "
    "SE ratio 0.91, coverage 0.925). Measured on a binary outcome with logistic regressions."
)


def _x76_adjustment_rows() -> tuple[SupportRow, ...]:
    """``treatment_adjustment=`` on ordinary TMLE (``T54``, ``option-treatment-adjustment``)."""
    slug = ["option-treatment-adjustment"]
    declared = {"treatment_adjustment": True}
    families = ("P1", "P2")
    return (
        point_row(
            "T54.in_sample",
            slug,
            families=families,
            options=declared,
            cells=[
                "interval_calibration/in_sample__correctly_specified",
                "double_robustness/treatment_correct",
            ],
            limits=[_ADJUSTMENT_LIMIT],
        ),
        point_row(
            "T54.stacked",
            slug,
            families=families,
            splitting=("stacked",),
            options=declared,
            cells=["interval_calibration/stacked__correctly_specified"],
            limits=[_ADJUSTMENT_LIMIT],
        ),
    )


POINT_ROWS: Final[tuple[SupportRow, ...]] = _with_shortfalls(
    (
        *_t_rows(),
        *_d_rows(),
        *_c_rows(),
        *_categorical_point_rows(),
        *_adaptive_iptw_rows(),
        *_projection_rows(),
        *_x76_stopping_point_rows(),
        *_x76_by_arm_rows(),
        *_x76_patience_rows(),
        *_x76_adjustment_rows(),
    )
)


# --------------------------------------------------------------------------- longitudinal

_LONG_WALD: Final[frozenset[Kind]] = frozenset({"point", "wald"})
_LONG_BAND: Final[frozenset[Kind]] = frozenset({"point", "wald", "band"})
_LONG_T: Final[frozenset[Kind]] = frozenset({"point", "wald_t"})
_LONG_DERIVED: Final[frozenset[Kind]] = frozenset({"point", "wald", "derived"})
_LONG_BOOTSTRAP: Final[frozenset[Kind]] = frozenset(
    {"point", "wald", "band", "bootstrap", "derived"}
)
#: The bootstrap kinds of a row that no band cell measures (``BAND_CELLS``).
_LONG_BOOTSTRAP_UNBANDED: Final[frozenset[Kind]] = _LONG_BOOTSTRAP - {"band"}
_TWO: Final[tuple[str, ...]] = ("2", "2")
_OUTCOME_KINDS: Final[frozenset[str]] = frozenset(
    {"end_of_study", "survival", "competing", "repeated"}
)


#: The limit of ``G45.hazard_model``: on the study law its valid bounds seldom change the fit.
#: ``docs/technical-reference/method-evidence/survival-node-bounds.md`` reports the 10.1% share.
HAZARD_BOUNDS_RARELY_BIND: Final[str] = (
    "The bounds rarely bind (the estimate moved on 10% of draws), so the study reports the "
    "binding witness of this fit and does not gate it."
)


def _x33_options(
    *,
    node_bounds: bool = False,
    hazard_pooling: str = "node",
    ratio_trim: bool = False,
    markov_order: bool = False,
    tuned_cap: bool = False,
    stopping: str = "relative",
    step_bound: str = "none",
    iterate: str = "last",
) -> tuple[tuple[str, Any], ...]:
    """The fitting options of a longitudinal key that change the fit's arithmetic.

    The defaults are those of a fit that declares none of them, in the order
    :func:`cleverly._support.longitudinal_key` records them. Only the ``mtp-tuning`` rows
    (``G41``) declare a cap tuned on the data.
    """
    return (
        ("node_bounds", node_bounds),
        ("hazard_pooling", hazard_pooling),
        ("ratio_trim", ratio_trim),
        ("markov_order", markov_order),
        ("tuned_cap", tuned_cap),
        ("stopping", stopping),
        ("step_bound", step_bound),
        ("iterate", iterate),
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
    engine: str | tuple[str, ...] = "tmle",
    normalization: str | None = None,
    survival_model: str = "iterated_mean",
    targeting: str = "iterative",
    isotonic: bool = False,
    ratio: str = "density",
    node_bounds: bool = False,
    hazard_pooling: str = "node",
    tuned_cap: bool = False,
    ratio_trim: bool = False,
    markov_order: bool = False,
    stopping: str = "relative",
    step_bound: str = "none",
    iterate: str = "last",
    kinds: frozenset[Kind] = _LONG_WALD,
    cells: Sequence[str] = (),
    limits: Sequence[str] = (),
    learners: Mapping[str, type] | None = None,
    kind_families: Mapping[Kind, Iterable[str]] | None = None,
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
    engine : str or tuple of str, default "tmle"
        ``tmle``, ``sdr``, ``ipsi``, ``gcomp`` or ``ipw``, or the whole engine tuple of a
        class that records one, such as ``("collaborative_tmle", "greedy")``.
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
    tuned_cap : bool, default False
        Whether the row's fits declare a policy cap tuned on the data.
    ratio_trim : bool, default False
        Whether the row's fits declare ``ratio_trim=``.
    markov_order : bool, default False
        Whether the row's data declare a Markov order.
    stopping : {"relative", "scaled_var", "sample_size"}, default "relative"
        The stopping rule of the row's fits.
    step_bound : {"none", "search", "fixed"}, default "none"
        The step bound of the row's fits under a declared stopping rule: none, a bound with a
        line search, or the whole bound. It reads ``"none"`` under ``"relative"``.
    iterate : {"last", "best"}, default "last"
        The iterate the row's fits report under a declared stopping rule. It reads ``"last"``
        under ``"relative"``.
    kinds : frozenset of str, default Wald
        What the row licenses.
    cells : sequence of str, default ()
        The licensing study cells.
    limits : sequence of str, default ()
        The stated limits of the row.
    learners : mapping of str to type or None, default None
        The exact learner class per nuisance role, where the study fixes one.
    kind_families : mapping of str to iterable of str or None, default None
        The target families that a listed kind covers. A kind with no entry covers every family
        of the row.

    Returns
    -------
    SupportRow
        The row.
    """
    key = SupportKey(
        kind="longitudinal",
        engine=(engine,) if isinstance(engine, str) else tuple(engine),
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
            ("normalization", normalization),
            ("ratio", ratio),
            ("survival_model", survival_model),
            ("targeting", targeting),
            ("isotonic", isotonic),
            *_x33_options(
                node_bounds=node_bounds,
                hazard_pooling=hazard_pooling,
                ratio_trim=ratio_trim,
                markov_order=markov_order,
                tuned_cap=tuned_cap,
                stopping=stopping,
                step_bound=step_bound,
                iterate=iterate,
            ),
        ),
    )
    return SupportRow(
        row_id,
        key,
        kinds,
        tuple(evidence),
        tuple(cells),
        learners,
        tuple(limits),
        kind_families=tuple(
            (kind, frozenset(covered)) for kind, covered in (kind_families or {}).items()
        ),
    )


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
            # Roadmap: X20.
            limits=["The full-refit bootstrap reads an SE ratio of 1.17 to 1.33 here."],
        ),
        long_row(
            "G03",
            ["weighted-ltmle"],
            weighting="fixed",
            kinds=_LONG_WALD,
            cells=["interval_calibration"],
        ),
        long_row(
            "G04",
            ["weighted-ltmle-crossfit"],
            weighting="fixed",
            splitting=xf,
            kinds=_LONG_WALD,
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
            # Roadmap: X20.
            limits=["Only the full-refit bootstrap measures this configuration."],
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
        merged = merge_rows(categorical.id, categorical, policy)
        if suffix == "in_sample":
            # default_bands:categorical_ltmle refits canonical-categorical-ltmle's in-sample fit
            # with its default band (default_band_properties.SOURCES), so it measures this
            # configuration's band, and it is red: SHORTFALLS states the reading.
            merged = replace(merged, evidence=(*merged.evidence, "default-simultaneous-bands"))
        rows.append(merged)
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
                # Only the in-sample continuous fit of longitudinal_mtp_properties._band_rows
                # measures the band (BAND_CELLS).
                kinds=_LONG_BAND if suffix == "continuous" else _LONG_WALD,
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
            kinds=_LONG_BOOTSTRAP_UNBANDED,
            cells=["boot_risk2_survival", "boot_risk4_survival", "rmst_survival"],
        ),
        long_row(
            "G12",
            ["canonical-ltmle-survival-crossfit"],
            outcome_kind="survival",
            splitting=xf,
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
            cells=["competing_t4"],
        ),
        long_row(
            "G14.three_arm",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("3",) * 3,
            kinds=_LONG_WALD,
            cells=["three_arm_t3"],
        ),
        long_row(
            "G14.continuous_time",
            [survival],
            outcome_kind="survival",
            held=True,
            treatment=("2",) * 4,
            kinds=_LONG_WALD,
            cells=["continuous_t4"],
        ),
        long_row(
            "G14.end_of_study",
            [survival],
            held=True,
            kinds=_LONG_WALD,
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
            # Roadmap: X13.
            limits=["The fixed-weight cell reads a finite-sample shortfall."],
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
            kinds=_LONG_WALD,
            cells=["three_arm_t3"],
            # Roadmap: X13.
            limits=["The three-arm cell reads a finite-sample shortfall."],
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
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
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
            [curve, "isotonic-binding"],
            outcome_kind="survival",
            treatment=("2", "2"),
            isotonic=True,
            kinds=_LONG_WALD,
            cells=[
                "isotonic_t2",
                "interval_calibration/projected_t1__correctly_specified",
                "binding_witness/never_t1",
                "material_move/never_t1",
            ],
            limits=[ISOTONIC_BINDING_LIMIT],
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
            kinds=_LONG_WALD,
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
            kinds=_LONG_WALD,
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
        # The band is the default simultaneous band of the in-sample calibration fits, which
        # estimate the mechanisms, over all fifteen reported parameters (cell all_reported).
        long_row(
            "G27",
            ["longitudinal-repeated-outcome"],
            outcome_kind="repeated",
            treatment=("2",) * 3,
            kinds=_LONG_BAND,
            cells=["interval_calibration", "all_reported"],
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
        # The band cell six_regimens is cross-fitted (longitudinal_sdr_properties._sdr), so
        # only G23.cross_fitted licenses the band (BAND_CELLS).
        long_row("G23.in_sample", sdr, engine="sdr", cells=["static"]),
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
            cells=["survival_folds_1", "survival_contrast_x2_h3_in_sample"],
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
        "The paired lmtp comparison uses the known mechanisms and the cell-means regressions "
        "of the study."
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
    # Node bounds on a held survival fit (`survival-node-bounds`): every node bounded, the
    # censoring model pooled over the nodes, and node 1's retention declared known (the
    # qualifier ``time>=1``), in sample with GLM learners on one five-visit law at n = 2,000.
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
                "true hazard and one pooled logistic hazard. " + bounds_limit,
                HAZARD_BOUNDS_RARELY_BIND,
            ],
        ),
    ]
    return tuple(rows)


def _continuous_dose_rows() -> list[SupportRow]:
    """The rows of the continuous-policy studies (X40): ``G39`` to ``G41``.

    The key reads the node count, the engine, the splitting, the outcome scale, the ratio
    route and whether a cap is tuned on the data. It does not read a declared support, an
    atom or a branch. The two-node
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
                tuned_cap=True,
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
            options=(("covariate_features", covariate_features),),
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
                # Roadmap: X72.
                "reference mean only. A static level or contrast is refused.",
            ),
        ),
        row("G42.weighted", _structural_cells("s2w"), weighting="fixed"),
        row("G42.clustered", _structural_cells("s2c"), dependence="clustered"),
    ]


def _long_options(normalization: str) -> tuple[tuple[str, Any], ...]:
    return (
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


_LH_BCA: Final[frozenset[Kind]] = frozenset({"point", "wald", "bootstrap_bca"})
_LH_BAND: Final[frozenset[Kind]] = frozenset({"point", "wald", "bootstrap_bca", "band"})
_LH_BOOT_BAND: Final[frozenset[Kind]] = frozenset(
    {"point", "wald", "bootstrap_bca", "bootstrap_band"}
)
#: The scenario sizes at which the study gates the Wald and the BCa interval of each row that
#: reads the Wald note: ``tests/unit/test_long_horizon_support.py`` derives both from
#: ``tests/long_horizon_design.py``.
_LH_GATED_N: Final[Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]]] = {
    "binary_k3": ((4_000,), (4_000,)),
    "binary_k4": ((16_000,), (16_000,)),
    "binary_k5": ((76_000,), (76_000,)),
    "categorical_k3": ((28_000,), (28_000,)),
    "categorical_k4": ((128_000,), (108_000, 124_000, 128_000)),
    "categorical_k5": ((116_000,), (64_000, 72_000, 96_000, 116_000, 124_000)),
    "sdr_binary_k3": ((4_000,), (4_000,)),
    "sdr_binary_k4": ((16_000,), (16_000,)),
    "sdr_binary_k5": ((76_000,), (76_000,)),
    "sdr_categorical_k5": (
        (44_000, 116_000),
        (44_000, 64_000, 72_000, 76_000, 96_000, 116_000, 124_000),
    ),
}
#: The rows whose gated BCa quantities all use static or dynamic plans, so that the row licenses
#: the BCa interval for those plans only.
_LH_STATIC_BCA_ROWS: Final[frozenset[str]] = frozenset(
    {
        "binary_k3",
        "binary_k4",
        "categorical_k3",
        "categorical_k4",
        "sdr_binary_k3",
        "sdr_binary_k4",
    }
)
_LH_STATIC_BCA_FAMILIES: Final[frozenset[str]] = frozenset({"end_of_study", "regimens", "static"})


def _lh_sizes(sizes: Sequence[int]) -> str:
    """Scenario sizes as a list, such as ``n = 4,000 and 16,000``."""
    text = [f"{n:,}" for n in sizes]
    return text[0] if len(text) == 1 else f"{', '.join(text[:-1])} and {text[-1]}"


def _lh_note(wald: Sequence[int], bca: Sequence[int], *, static_bca: bool) -> str:
    """The stated limit of a long-horizon row that reads a Wald interval and a BCa interval."""
    note = (
        "The Wald interval is gated for a quantity only where n is at least twice its n_kappa. "
        f"The study gates it at n = {_lh_sizes(wald)}. "
        "Where the Wald interval is not gated, the BCa interval at B = 199 is gated instead, "
        f"at n = {_lh_sizes(bca)}, and the BCa row of a Wald-gated quantity reports. "
        "The row grain checks neither n nor the quantity. "
        "A fit at another n or on another quantity reads as this row."
    )
    if static_bca:
        note += (
            " The study gates BCa on plans without a known stochastic node only, "
            "so the row licenses BCa for those plans."
        )
    return note


_LH_SWITCH_LIMIT: Final[str] = (
    "X39-below-tail-size: the Wald interval is validated only at n of at least twice the "
    "exact-law tail size of the quantity. At n = 72,000, below the tail size 134,244 of "
    "ey_regimen[switch], coverage is 0.9133; bias -0.0015 to 0.0038, SE ratio 0.9624"
)
# Roadmap: X20.
_LH_CROSSFIT_NOTE: Final[str] = (
    "The cross-fitted bootstrap interval is experimental: the full-refit bootstrap reads an SE "
    "ratio of 1.17 to 1.33 here, and six of the Wald rows lose their gate below twice "
    "n_kappa."
)
_LH_MSM_NOTE: Final[str] = (
    "Both coefficients are gated at n = 96,000, which is at least twice the n_kappa of the "
    "intercept (94,226) and of the duration (50,908). The MSM fit runs no bootstrap."
)
_LH_SDR_BAND_NOTE: Final[str] = "The SDR bootstrap band is experimental and gates no row."
_LH_NO_PRIMARY_NOTE: Final[str] = (
    "The study fits this setting in its property cells only, so no BCa row gates it and the "
    "Wald interval rests on the calibration cells."
)
_LH_LEARNED_NOTE: Final[str] = (
    "The cell is a bias check of the necessity family: it states no interval, and the "
    "treatment of the third arm is learned."
)


def _long_horizon_rows() -> tuple[SupportRow, ...]:
    """The rows of the three long-horizon studies (X39): ``G46``.

    One row for each estimator, decision count, arm count, splitting and working model that a
    primary scenario covers. The row ids read ``G46.tmle.<name>`` and ``G46.sdr.<name>``.

    Every row licenses ``point``. It licenses ``wald`` when the study gates a Wald row of a
    member scenario, or an ``interval_calibration`` cell of the row's setting. It licenses
    ``bootstrap_bca`` when the study gates a BCa row of a member scenario. The study gates BCa
    only where the Wald row is not gated (n below twice n_kappa). The six rows whose gated BCa
    rows all use plans without a known stochastic node license BCa for those plans only
    (``kind_families``). The row grain checks neither n nor the quantity, so each row states
    its gated n and that limit in its first note. The cross-fitted rows and every SDR row
    license no bootstrap band: the cross-fitted bootstrap is experimental (X20), and the SDR
    bootstrap band is experimental. A row licenses ``band`` when the study gates the Wald-curve
    band of its setting (``binary_k5``) and ``bootstrap_band`` when it gates the studentized
    sup-t band at B = 199 (``categorical_k5``).
    """
    tmle2 = ("long-horizon-ltmle", "tmle", 2)
    tmle3 = ("long-horizon-categorical-ltmle", "tmle", 3)
    sdr2 = ("long-horizon-sdr", "sdr", 2)
    sdr3 = ("long-horizon-sdr", "sdr", 3)
    #: name, study, decisions, folds, msm, learned, kinds, stochastic, cells, limits
    declared: tuple[
        tuple[
            str,
            tuple[str, str, int],
            int,
            int,
            bool,
            bool,
            frozenset[Kind],
            bool,
            tuple[str, ...],
            tuple[str, ...],
        ],
        ...,
    ] = (
        ("binary_k3", tmle2, 3, 1, False, False, _LH_BCA, True, ("binary_k3",), ()),
        ("binary_k4", tmle2, 4, 1, False, False, _LH_BCA, True, ("binary_k4",), ()),
        ("binary_k5", tmle2, 5, 1, False, False, _LH_BAND, True, ("binary_k5",), ()),
        (
            "binary_k5_msm",
            tmle2,
            5,
            1,
            True,
            False,
            _LONG_WALD,
            False,
            ("binary_k5_msm",),
            (_LH_MSM_NOTE,),
        ),
        (
            "categorical_k3",
            tmle3,
            3,
            1,
            False,
            False,
            _LH_BCA,
            True,
            ("categorical_k3",),
            (),
        ),
        (
            "categorical_k4",
            tmle3,
            4,
            1,
            False,
            False,
            _LH_BCA,
            True,
            ("categorical_k4", "categorical_k4_high", "categorical_k4_static"),
            (),
        ),
        (
            "categorical_k5",
            tmle3,
            5,
            1,
            False,
            False,
            _LH_BOOT_BAND,
            True,
            (
                "categorical_k5",
                "categorical_k5_mix",
                "categorical_k5_static",
                "categorical_k5_switch",
                "categorical_k5_high",
            ),
            (),
        ),
        (
            "categorical_k5_crossfit",
            tmle3,
            5,
            5,
            False,
            False,
            _LONG_WALD,
            True,
            ("categorical_k5_crossfit",),
            (_LH_CROSSFIT_NOTE,),
        ),
        (
            "sdr_binary_k3",
            sdr2,
            3,
            1,
            False,
            False,
            _LH_BCA,
            True,
            ("sdr_binary_k3",),
            (),
        ),
        (
            "sdr_binary_k4",
            sdr2,
            4,
            1,
            False,
            False,
            _LH_BCA,
            True,
            ("sdr_binary_k4",),
            (),
        ),
        (
            "sdr_binary_k5",
            sdr2,
            5,
            1,
            False,
            False,
            _LH_BCA,
            True,
            ("sdr_binary_k5",),
            (),
        ),
        (
            "sdr_categorical_k5",
            sdr3,
            5,
            1,
            False,
            False,
            _LH_BCA,
            True,
            (
                "sdr_categorical_k5",
                "sdr_categorical_k5_mix",
                "sdr_categorical_k5_static",
                "sdr_categorical_k5_switch",
                "sdr_categorical_k5_high",
                "sdr_policy_k5",
                "sdr_policy_k5_mix",
                "sdr_policy_k5_taper",
            ),
            (_LH_SWITCH_LIMIT, _LH_SDR_BAND_NOTE),
        ),
        (
            "sdr_categorical_k5_crossfit",
            sdr3,
            5,
            5,
            False,
            False,
            _LONG_WALD,
            True,
            ("sdr_categorical_k5_crossfit",),
            (_LH_CROSSFIT_NOTE,),
        ),
        (
            "sdr_categorical_k3",
            sdr3,
            3,
            1,
            False,
            False,
            _LONG_WALD,
            True,
            ("interval_calibration/categorical_respond_vs_ref__k3__known_mechanism",),
            (_LH_NO_PRIMARY_NOTE,),
        ),
        (
            "sdr_categorical_k4",
            sdr3,
            4,
            1,
            False,
            False,
            _LONG_WALD,
            True,
            (
                "interval_calibration/categorical_respond_vs_ref__k4__known_mechanism",
                "root_n_and_efficiency/respond_vs_ref__k4__n_4000",
            ),
            (_LH_NO_PRIMARY_NOTE,),
        ),
        (
            "categorical_k5_learned",
            tmle3,
            5,
            1,
            False,
            True,
            frozenset({"point"}),
            True,
            ("categorical_probability_necessity/third_arm__k5__binary_complement",),
            (_LH_LEARNED_NOTE,),
        ),
        (
            "sdr_categorical_k5_learned",
            sdr3,
            5,
            1,
            False,
            True,
            frozenset({"point"}),
            True,
            ("categorical_probability_necessity/third_arm__k5__binary_complement",),
            (_LH_LEARNED_NOTE,),
        ),
    )
    rows: list[SupportRow] = []
    for name, (
        slug,
        engine,
        arms,
    ), nodes, folds, msm, learned, kinds, stochastic, cells, limits in declared:
        roles = "C" if learned else "AC"
        if name in _LH_GATED_N:
            wald, bca = _LH_GATED_N[name]
            limits = (_lh_note(wald, bca, static_bca=name in _LH_STATIC_BCA_ROWS), *limits)
        rows.append(
            long_row(
                f"G46.{engine}.{name}",
                [slug],
                regimens=("static", "stochastic") if stochastic else ("static",),
                msm=msm,
                treatment=tuple(str(arms) for _ in range(nodes)),
                splitting=("cross_fitted",) if folds > 1 else ("in_sample",),
                qualifiers=[f"{role}{k}" for role in roles for k in range(1, nodes + 1)],
                engine=engine,
                kinds=kinds,
                cells=cells,
                limits=limits,
                kind_families=(
                    {"bootstrap_bca": _LH_STATIC_BCA_FAMILIES}
                    if name in _LH_STATIC_BCA_ROWS
                    else None
                ),
            )
        )
    return tuple(rows)


#: The longitudinal rows: ``LTMLE`` (``G01`` to ``G21``, ``G27``), ``SDR`` (``G23`` to ``G26``),
#: the sequential baselines (``G28``), the parametric g-formula (``G29``), repeated-outcome
#: ``IPSI`` (``G37``), the continuous-dose price and tuned-cap rows (``G39`` to ``G41``), the
#: structural nested mean model (``G42``), the categorical tilt (``G43``) and the long-horizon
#: studies of ``LTMLE`` and ``SDR`` at three to five decisions (``G46``) and the learned two-node
#: regimen (``G47``), the eliminated competing event (``G48``), separable effects (``G49``),
#: collaborative TMLE of a risk curve (``G50``), the drift-corrected survival TMLE (``G51``) and
#: the working model over the hazard curve (``G60``).
#:
def _learned_regimen_rows() -> tuple[SupportRow, ...]:
    """The learned two-node regimen of ``learned-regimen-ltmle`` (X11 (b), X11 PR K): ``G47``.

    The row keys the regimen kind ``learned`` on the cross-fitted censored two-node fit with
    every option at its default. It was added because every declared verdict passed: the
    primary tests (bias leg, coverage floor, SE sanity), ``interval_calibration`` (the positive
    and both controls), ``conservative_coverage`` (the positive and its control),
    ``targeting_necessity`` (both labels) and ``fold_locality``. The numbers are the committed
    results (``tests/canonical/learned_regimen_ltmle``).
    """
    return (
        long_row(
            "G47",
            ["learned-regimen-ltmle"],
            regimens=("learned",),
            splitting=("cross_fitted",),
            cells=["two_node", "squared_w1"],
            # Roadmap: X11.
            limits=[
                "The interval targets the stacked value of the fit's own learned rule pairs, "
                "sum_v (n_v / n) Psi_{d_nv}(P_0), not the value of the optimal pair.",
                "The study measures one law at n = 2,000 with ten folds, two binary treatment "
                "nodes, censoring at each node and an end-of-study binary outcome. Neither n nor "
                "the fold count is in the key, so another n, another fold count or another law "
                "reads supported and rests on the theory alone.",
                "The treatment and censoring models are correct main-terms logistic models in "
                "every cell, and coverage rests on them: the curve omits the treatment "
                "mechanism's term and is conservative when they are correct. The node-2 learner "
                "is a correct model of each stratum in the Q-learning that learns the rules. "
                "The estimator's own outcome regressions pool the followers' strata with no "
                "treatment column, so the logistic and the cubic node-1 learner are "
                "approximations there. A misspecified treatment or censoring model is not "
                "measured.",
                "The primary cell reads coverage 0.954 and SE ratio 1.017 (1,200 draws). The "
                "calibration positive reads SE ratio 0.979 (99% interval 0.960 to 0.998) and "
                "coverage 0.939 (9,200 draws), and its controls read SE ratios 0.783 and 0.702.",
                "A misspecified node-1 learner on (W1^2, W2), with g correct, claims coverage "
                "only: coverage 0.940 over 2,400 draws, against a control at 0.865.",
                "Targeting removes the plug-in's bias for that misspecified node-1 learner (a "
                "limit bias, untargeted bias 0.060) and for the declared learners (a "
                "finite-sample bias at n = 2,000 whose limit is about 0, untargeted bias 0.016).",
                "A learner whose limit rule has no margin gives a non-regular target, and its "
                "interval under-covers; no cell measures that regime. The fold-locality control "
                "(rules learned on the rows that evaluate them) reads bias 0.090, so the "
                "instrument resolves in-sample rule learning.",
                "A continuous outcome with q_bounds=, declared node bounds, ratio_trim=, an "
                "uncensored panel and a known mechanism are keyed apart and read experimental; a "
                "stopping rule is keyed apart and reads experimental. More than two "
                "nodes, repeats above 1, n_folds=1, weights and clusters refuse. No comparator: "
                "no maintained package fits a learned two-node regimen by cross-validated TMLE.",
            ],
        ),
    )


def _eliminated_rows() -> tuple[SupportRow, ...]:
    """The eliminated competing event of ``eliminated-competing-events`` (X91 part A): ``G48``.

    A fit with ``eliminate=`` keys the missing axis ``("censoring", "eliminated")``. The rows
    were added because every declared verdict passed: 31 of 31 primary tests and all ten
    property cells (``double_robustness`` and ``interval_calibration`` with its controls), with
    no failed fit. The numbers are the committed results
    (``tests/canonical/eliminated_competing_events``).
    """
    slug = ["eliminated-competing-events"]
    shared = (
        "Every nuisance in the study is a saturated cell mean on a discrete law, n = 16,000, so "
        "a flexible learner, a continuous covariate and another n rest on the theory alone. "
        "One prevented cause and one remaining cause; no weights, clusters or comparator (R "
        "lmtp with a composite censoring column is the matching implementation, and its runs "
        "are pending)."
    )
    return (
        long_row(
            "G48.in_sample",
            slug,
            outcome_kind="survival",
            missing=("censoring", "eliminated"),
            cells=["ltmle_time_varying", "tv_rule_t2", "tv_never_t2", "ltmle_always_t2"],
            limits=[
                "Two binary treatment nodes with a time-varying covariate, two static plans "
                "and a dynamic one, horizons 1 and 2. The primary rows read coverage 0.945 to "
                "0.951 over 2,800 draws; the calibration cell reads SE ratio 0.990 and coverage "
                "0.947 over 4,800 draws.",
                shared,
            ],
        ),
        long_row(
            "G48.cross_fitted",
            slug,
            outcome_kind="survival",
            missing=("censoring", "eliminated"),
            splitting=("cross_fitted",),
            cells=["ltmle_time_varying_crossfit"],
            limits=[
                "Two outer folds, with the prevented-cause factor fitted by the censoring "
                "learner's clone. The same law and estimands as G48.in_sample, coverage 0.945 to "
                "0.956 over 2,800 draws. No property cell reads the cross-fitted fit; at "
                "n = 16,000 its spread sits up to 1.32 times the efficient one while its "
                "reported SE tracks it (SE ratio 0.949 to 1.033).",
                shared,
            ],
        ),
        long_row(
            "G48.hazard",
            slug,
            outcome_kind="survival",
            missing=("censoring", "eliminated"),
            held=True,
            treatment=("3",) * 3,
            survival_model="hazard",
            cells=["hazard_held_baseline", "hb_a1_t2", "hb_a2_t1", "hazard_a1_t3"],
            limits=[
                "A three-level baseline treatment held over three nodes, horizons 1 to 3. At "
                "t = 3 the influence curve rests on Kish sizes near 30 to 60 and the fit warns "
                "SparseInfluenceWarning. The primary rows read coverage 0.938 to 0.953 over 2,800 "
                "draws, and the calibration cell at t = 3 reads SE ratio 0.992 and "
                "coverage 0.945 over 4,800 draws.",
                shared,
            ],
        ),
    )


def _separable_rows() -> tuple[SupportRow, ...]:
    """Separable effects on hazard ``LTMLE`` of ``separable-effects`` (X91 part D): ``G49``.

    A fit with a ``Separable`` regimen keys the regimen kind ``separable``, beside its static
    arms, which the factor-wise engine also targets, and the position of the cause it counts
    first among the data's causes, so a fit that counts the other cause first matches no row.
    The rows were added because every
    declared verdict passed: 82 of 82 primary tests and all ten property cells
    (``double_robustness`` and ``interval_calibration`` with its controls), with no failed fit.
    The numbers are the committed results (``tests/canonical/separable_effects``).
    """
    slug = ["separable-effects"]
    shared = (
        "A binary baseline covariate, a three-level baseline treatment held over three nodes, "
        "censoring, and relapse and death listed in that order with death counted first; "
        "horizons 1 to 3. A fit that counts the first-listed cause first is not covered. "
        "Every nuisance is a saturated cell mean, n = 16,000, so a flexible learner, a "
        "continuous covariate and another n rest on the theory alone. Two causes, baseline "
        "covariates only; no weights, clusters or comparator (none was located)."
    )
    common: dict[str, Any] = {
        "outcome_kind": "competing",
        "regimens": ("separable", "static"),
        "held": True,
        "treatment": ("3",) * 3,
        "survival_model": "hazard",
        "targeting": "one_step",
        # The study lists relapse then death and counts death first.
        "qualifiers": ("separable_first_cause_2",),
    }
    return (
        long_row(
            "G49.in_sample",
            slug,
            cells=["in_sample", "sep10_relapse_t3", "in_sample_sde_relapse_t3"],
            limits=[
                "Two static and two separable arms, 41 risks and contrasts. The primary rows "
                "read coverage 0.945 to 0.954 over 8,400 draws; the separable direct effect's "
                "calibration cell reads SE ratio 0.998 and coverage 0.952 over 4,400 draws. The "
                "robustness cells measure the factor engine with the exact mechanism and one "
                "wrong hazard at a time.",
                shared,
            ],
            **common,
        ),
        long_row(
            "G49.cross_fitted",
            slug,
            splitting=("cross_fitted",),
            cells=["cross_fitted", "cross_fitted_sep01_death_t3"],
            limits=[
                "Two outer folds. The primary rows read coverage 0.948 to 0.955 over 8,400 "
                "draws, and the calibration cell reads SE ratio 0.990 and coverage 0.945 over "
                "4,400 draws. In 2.0% of the fits the targeting stopped short of tol at a "
                "relative score near 1e-3 and warned; those fits are kept in every row.",
                shared,
            ],
            **common,
        ),
    )


def _survival_ctmle_rows() -> tuple[SupportRow, ...]:
    """Collaborative TMLE of a risk curve of ``survival-ctmle`` (X91 part B): ``G50``.

    ``SurvivalCTMLE`` records the engine ``("collaborative_tmle", "greedy")`` on the hazard
    ``LTMLE`` key, with a two-level baseline treatment held over the two nodes and pooled hazard
    and censoring regressions. Every estimate takes the ``working_mechanism_plugin`` status, so
    the rows license the point alone (F18). They were added because every declared verdict
    passed: 14 of 14 bias tests, the RMSE comparison against the full treatment model at OR 15,
    and no failed fit. The study publishes under ``reporting`` because it declares no property
    cell. The numbers are the committed results (``tests/canonical/survival_ctmle``).
    """
    slug = ["survival-ctmle"]
    shared = (
        "The laws of Stitelman and van der Laan (2010), Sections 12 to 14: five clipped "
        "correlated normal covariates, a binary baseline treatment, light censoring (about 4.7% "
        "per interval), two grid times and the risk at t = 2. The outcome model is a correct "
        "main-term GLM with hazard_pooling='pooled' in every cell, and the candidates are the "
        "paper's main-term logistic models. A misspecified outcome model is reported only: in "
        "scenario 2 the risk difference then reads a standardized bias of -0.98 at OR 10. "
        "Another law, learner or n rests on the paper alone. No comparator: no maintained "
        "package fits this construction."
    )
    common: dict[str, Any] = {
        "outcome_kind": "survival",
        "engine": ("collaborative_tmle", "greedy"),
        "held": True,
        "survival_model": "hazard",
        "hazard_pooling": "pooled",
        "kinds": _POINT,
    }
    return (
        long_row(
            "G50.in_sample",
            slug,
            cells=[
                "or1_qcorrect",
                "or10_qcorrect",
                "or15_qcorrect",
                "s2_or10_qcorrect",
                "s2_or15_qcorrect",
                "or15_qcorrect_n2000",
            ],
            limits=[
                "Points only: the plug-in interval is withheld (F18). Six in-sample cells at "
                "OR 1, 10 and 15, scenario 2 and n = 2,000; the standardized bias of the risk of "
                "arm 1 and of the risk difference stays within 0.075 over 1,600 draws each, and "
                "the RMSE of the risk of arm 1 at OR 15 is 0.54 times that of hazard LTMLE with "
                "the full treatment model.",
                shared,
            ],
            **common,
        ),
        long_row(
            "G50.cross_fitted",
            slug,
            splitting=("cross_fitted",),
            cells=["or10_qcorrect_crossfit"],
            limits=[
                "Points only: the plug-in interval is withheld (F18). Two outer folds for the "
                "initial outcome fit at OR 10; the standardized bias reads 0.047 for the risk of "
                "arm 1 and -0.010 for the risk difference over 1,600 draws.",
                shared,
            ],
            **common,
        ),
    )


def _survival_dr_tmle_rows() -> tuple[SupportRow, ...]:
    """The drift-corrected survival TMLE of ``survival-dr-tmle`` (X91 part F): ``G51``.

    ``SurvivalDRTMLE`` records the engine ``("drtmle", "univariate")`` on the hazard ``LTMLE``
    key, with a two-level baseline treatment held over the five nodes.  The rows were added
    because every declared verdict passed: 18 of 18 primary tests and all six property cells
    (``double_robustness`` and ``interval_calibration`` with its controls), with no failed fit.
    The numbers are the committed results (``tests/canonical/survival_dr_tmle``).
    """
    slug = ["survival-dr-tmle"]
    shared = (
        "Law L-S: two normal covariates, a binary baseline treatment, censoring and five grid "
        "times, the risk at t = 5. Per-node logistic models, ten outer folds, n = 1,000 to "
        "16,000; another law, a flexible learner or an in-sample fit rests on the theory alone "
        "(an in-sample gradient-boosting mechanism read a standardized bias near -4). One event "
        "type, baseline covariates only; no weights, clusters or comparator (none was located)."
    )
    common: dict[str, Any] = {
        "outcome_kind": "survival",
        "engine": ("drtmle", "univariate"),
        "held": True,
        "treatment": ("2",) * 5,
        "survival_model": "hazard",
        "splitting": ("cross_fitted",),
    }
    return (
        long_row(
            "G51.estimated",
            slug,
            cells=["both_1k", "both_4k", "both_16k", "a1_t5", "both_rd"],
            limits=[
                "Every nuisance estimated. The primary rows read coverage 0.942 to 0.953 over "
                "6,000 draws per size, and the risk difference's calibration cell reads SE ratio "
                "0.995 and coverage 0.946 over 3,600 draws. The drift loop stops at max_iter in "
                "10.7% of the fits at n = 1,000 and 2.8% at 16,000, with a warning; those fits are "
                "kept in every row.",
                shared,
            ],
            **common,
        ),
        long_row(
            "G51.declared",
            slug,
            qualifiers=("A", "time>=2", "time>=3", "time>=4", "time>=5"),
            cells=["known_1k", "known_4k", "known_16k"],
            limits=[
                "The treatment and retention probabilities declared on a randomized law, with "
                "the hazard model wrong. The arm rows read coverage 0.946 to 0.955 over 6,000 "
                "draws per size. With every factor declared the drift block is zero and the fit "
                "equals hazard LTMLE.",
                shared,
            ],
            **common,
        ),
    )


def _hazard_msm_rows() -> tuple[SupportRow, ...]:
    """The working model over the hazard curve of ``hazard-msm`` (X70 part (d)): ``G60``.

    ``LTMLE(msm=..., survival_model="hazard", targeting="one_step")`` keys ``msm`` on the hazard
    ``LTMLE`` key, with a two-level baseline treatment held over the visits.  The rows were added
    because every declared verdict passed: 12 of 12 primary tests and 51 of 51 property cells
    with a verdict (calibration with its controls, the simultaneous band, robustness on both
    laws, targeting necessity, the root-n rate and power), with no failed fit.  The numbers are
    the committed results (``tests/canonical/hazard_msm``).
    """
    slug = ["hazard-msm"]
    shared = (
        "Finite laws with W1 binary and W2 on four levels, a logistic arm and visit dropout, "
        "and logistic pseudo-hazards with an a x W1 term; the logit working model "
        "(1, a, t, a W1) with uniform cell weights and V = W1, and one identity-link "
        "coefficient. Correctly specified low-dimensional logistic learners at n = 2,000, one "
        "fold; a wrong hazard learner or wrong mechanisms only in the robustness cells. A "
        "flexible learner, a weight that varies with V or the cell, cross-fitting, observation "
        "weights or clusters rest on the exact-law tests alone. No comparator: no maintained "
        "package targets these coefficient scores."
    )
    common: dict[str, Any] = {
        "held": True,
        "msm": True,
        "survival_model": "hazard",
        "targeting": "one_step",
    }
    return (
        long_row(
            "G60.survival",
            slug,
            outcome_kind="survival",
            treatment=("2",) * 5,
            kinds=_LONG_BAND,
            cells=["survival"],
            limits=[
                "One event type over five visits. The primary rows read coverage 0.948 to 0.954 "
                "over 2,400 draws, the calibration cells SE ratio 0.988 to 1.014 over 5,200, and "
                "the max-t band over the four coefficients joint coverage 0.946.",
                shared,
            ],
            **common,
        ),
        long_row(
            "G60.competing",
            slug,
            outcome_kind="competing",
            treatment=("2",) * 4,
            cells=["competing"],
            limits=[
                "Relapse and death over four visits, each cause with its own projection. The "
                "primary rows read coverage 0.940 to 0.956 over 2,400 draws; no band is "
                "validated on this law.",
                shared,
            ],
            **common,
        ),
    )


# The longitudinal rows of the X76 option studies, one function per part, each registered from
# its committed run: a declared stopping rule (G55), ratio_trim= (G56) and a Markov order (G57).
def _x76_stopping_long_rows() -> tuple[SupportRow, ...]:
    """A declared stopping rule on ``LTMLE`` (``G55``, study ``option-stopping-ltmle``).

    ``G55.<rule>.end_of_study`` is the key of ``G01``, whose walk is the per-node Newton
    fluctuation, so a one-step end-of-study walk under a rule matches no row (design review
    S2). ``G55.<rule>.survival`` is the key of ``G31.survival``, the one-step hazard walk.
    """
    slug = ["option-stopping-ltmle"]
    witness = {"scaled_var": "0.816", "sample_size": "0.248"}
    rows = []
    for rule, label in _RULE_LABELS.items():
        rows.append(
            long_row(
                f"G55.{rule}.end_of_study",
                slug,
                stopping=rule,
                kinds=_LONG_DERIVED,
                cells=[f"interval_calibration/l3_{label}_newton__widened"],
                limits=[
                    _RULE_INTERVAL_LIMIT,
                    _rule_newton_limit("1.3e-7"),
                    "Measured in sample at n = 2,000 on the canonical-ltmle censored two-node "
                    "law, for the ATE of the always and never regimens.",
                ],
            )
        )
        rows.append(
            long_row(
                f"G55.{rule}.survival",
                slug,
                outcome_kind="survival",
                held=True,
                treatment=("2",) * 5,
                survival_model="hazard",
                targeting="one_step",
                stopping=rule,
                kinds=_LONG_WALD,
                cells=[f"interval_calibration/l4_{label}_one_step__widened"],
                limits=[
                    _RULE_INTERVAL_LIMIT,
                    f"On the one-step hazard walk the rule binds: the residual reaches 0.01 SE "
                    f"on a share {witness[rule]} of draws (cell "
                    f"residual_witness/l4_{label}_one_step__witness). Where it binds hard the "
                    "plain interval of the arm-1 risk covers 0.874 and the widened one 0.979 "
                    "(cells residual_necessity).",
                    "Measured at n = 2,000 on the point-survival-curve five-visit law, for the "
                    "risk difference at visit 5. The simultaneous band under a rule is not "
                    "measured.",
                ],
            )
        )
    return tuple(rows)


#: The limit of every ``G56`` row (study ``option-ratio-trim``).
_TRIM_LIMIT = (
    "The trimmed point estimate is consistent when every iterated outcome regression is. "
    "The trim removes the double robustness on the trimmed share: with an outcome regression "
    "that ignores W the trimmed fit is biased by 1.90 SD for LTMLE and 2.43 SD for SDR, where "
    "the untrimmed fit is not (study option-ratio-trim, cells trim_necessity). Measured only "
    "with LinearRegression outcome and pseudo-outcome regressions on a law where they are "
    "correct; any other outcome learner reads experimental, and pinning the class does not "
    "make a linear regression correct. No interval is given (F30): the plug-in standard "
    "error is a diagnostic, 0.55 to 0.58 of the spread of the trimmed estimate, with "
    "coverage 0.72 to 0.73."
)


def _x76_trim_rows() -> tuple[SupportRow, ...]:
    """``ratio_trim=`` on ``LTMLE`` and ``SDR``: the point estimate (``G56``)."""
    from sklearn.linear_model import LinearRegression

    slug = ["option-ratio-trim"]
    linear: dict[str, type] = {"outcome": LinearRegression, "pseudo": LinearRegression}
    rows = []
    for engine, label in (("tmle", "ltmle"), ("sdr", "sdr")):
        for splitting, cell in (("in_sample", "in_sample"), ("cross_fitted", "stacked")):
            rows.append(
                long_row(
                    f"G56.{label}.{splitting}",
                    slug,
                    engine=engine,
                    splitting=(splitting,),
                    ratio_trim=True,
                    kinds=frozenset({"point"}),
                    cells=[f"trim_point/{label}_{cell}"],
                    limits=[_TRIM_LIMIT],
                    learners=linear,
                )
            )
    return tuple(rows)


def _x76_markov_rows() -> tuple[SupportRow, ...]:
    """A Markov-order window on ``LTMLE`` (``G57``, study ``option-markov-order``)."""
    return (
        long_row(
            "G57",
            ["option-markov-order"],
            treatment=("2", "2", "2", "2"),
            markov_order=True,
            kinds=_LONG_WALD,
            cells=[
                "interval_calibration/order1_ey__correctly_specified",
                "interval_calibration/order1_ate__correctly_specified",
                "interval_calibration/order2_ate__correctly_specified",
                "order_double_robustness/ey_never",
            ],
            limits=[
                "The interval holds under the declared restriction: at every node the outcome "
                "regression and the treatment and censoring mechanisms depend on the history "
                "only through W and the last k + 1 nodes. The data can test it, and the fit "
                "does not. When it fails for the outcome regression and the mechanisms, the "
                "estimate is biased (study option-markov-order, cell order_necessity: 1.32 SD "
                "at order 0). The interval is claimed valid, not efficient. Measured at orders "
                "1 and 2 on four binary treatment nodes with censoring, in sample, with linear "
                "outcome regressions and logistic mechanisms."
            ],
        ),
    )


#: A learned two-node regimen (``LearnedRegimen``, X11 (b)) keys the regimen kind ``learned``.
#: The row ``G47`` of the ``learned-regimen-ltmle`` study covers the cross-fitted censored
#: two-node fit with every option at its default; every other composition of that kind matches
#: no row and reads experimental.
LONGITUDINAL_ROWS: Final[tuple[SupportRow, ...]] = _with_shortfalls(
    (
        *_long_rows(),
        *_categorical_long_rows(),
        *_long_horizon_rows(),
        *_learned_regimen_rows(),
        *_eliminated_rows(),
        *_separable_rows(),
        *_survival_ctmle_rows(),
        *_survival_dr_tmle_rows(),
        *_hazard_msm_rows(),
        *_x76_stopping_long_rows(),
        *_x76_trim_rows(),
        *_x76_markov_rows(),
    )
)


# ------------------------------------------------------------------ exemptions and anchors

#: The registered studies that no row names, each with the reason. A study that measures no
#: estimate cannot license a configuration, so it is not the evidence of a row. A study of an
#: engine that no support key reads is the other case: its reason names what a row would need.
EXEMPT_STUDIES: Final[Mapping[str, str]] = {
    "band-t-reference-held-out": (
        "It holds out one t band cell, the strata law at n = 1,000, to test the rule out of "
        "sample. The cell licenses no row: band-t-reference licenses the t band, and this study "
        "is a reported check of it."
    ),
    "adaptive-tmle-trial": (
        "It measures the intervals of AdaptiveTMLEMethod on a trial pooled with external data. "
        "No support key reads a TrialWithExternalData fit, so no row can name the "
        "configurations the study covers. A row would not change the label: every estimate "
        "carries the shown status working_model_projection, which keeps an estimate "
        "experimental inside any row until F40 closes."
    ),
    "adaptive-tmle-selective": (
        "It measures the intervals of AdaptiveTMLEMethod under selective enrollment with Super "
        "Learner nuisances. That engine is no point-treatment row: every estimate carries the "
        "working_model_projection status, whose open condition F40 holds, and no support row "
        "was declared before the run, so the fit keeps the experimental label."
    ),
    "ep-cate": (
        "It measures the EP curve, which carries no interval and is no fit of a point row. "
        "cleverly.estimators.ep.EP_MEASURED, not a support row, records what the study licenses."
    ),
    "ep-crr": (
        "It measures the log-CRR EP curve, which carries no interval and is no fit of a point "
        "row. cleverly.estimators.ep.EP_MEASURED, not a support row, records what the study "
        "licenses."
    ),
    "ep-assessment": (
        "It measures the held-out readings of an EP curve. A reading is no estimate and carries "
        "no status."
    ),
    "targeting-curves": (
        "It measures the targeting curves of a priority score, which are no Target and no fit "
        "of a point row. cleverly.targeting_curves.TARGETING_MEASURED, not a support row, "
        "records the configurations the study licenses."
    ),
    "score-assessment": (
        "It measures the score assessment of an external score, which is no Target and no fit "
        "of a point row. cleverly.effect_curves.SCORE_ASSESSMENT_MEASURED and "
        "SCORE_PAIRED_MEASURED, not a support row, record the configurations the study "
        "licenses."
    ),
    "small-sample-wald": (
        "It measures the small-sample Wald interval rule, computed from the public inference "
        "functions, while the shipped intervals stay unchanged. No shipped fit reports the "
        "interval it measures, so no row can name it until the default reads the rule."
    ),
    "dose-response": (
        "It measures the kernel-smoothed dose-response curve of DoseResponseTMLE, which is no "
        "Target and no fit of a point row. cleverly.estimators.dose_response."
        "DOSE_RESPONSE_MEASURED, not a support row, records the configurations the study "
        "licenses, and a pair below the Kish floor reads experimental (low_dose_support_plugin)."
    ),
    "missingness-tilt-inference": (
        "It measures the targeted missingness tilt's intervals off gamma = 0, a sensitivity "
        "reading of a fit and no fit of a point row. The missingness_tilt_interval instrument, "
        "not a support row, records the configurations the study licenses."
    ),
    "dose-response-npcausal": (
        "It compares the kernel-smoothed dose-response curve of DoseResponseTMLE with npcausal "
        "ctseff, which is no Target and no fit of a point row. Each pair's interval takes its "
        "status from the fit's own support report, not from a support row."
    ),
    "omitted-variable-bound-se": (
        "It measures the standard error of a sensitivity bound. The bound carries its own "
        "status and is no fit of a row."
    ),
    "ovb-limit-coverage": (
        "It measures the coverage of the omitted-variable bound limits of X82 part G. A limit "
        "is no estimate, and VERDICT_EVIDENCE, not a support row, records what the study "
        "licenses."
    ),
    "marginal-bound-limit-coverage": (
        "It measures the coverage of the marginal sensitivity bound limits of X82 part H. A "
        "limit is no estimate, and VERDICT_EVIDENCE, not a support row, records what the study "
        "licenses."
    ),
    "evalue-limit-coverage": (
        "It measures the coverage of the risk-ratio intervals that two E-value limits map. "
        "VERDICT_EVIDENCE, not a support row, records what the study licenses."
    ),
    "calibration-slope-warning": (
        "It measures the rate of a diagnostic warning. The warning is no estimate."
    ),
    "refuter-calibration-longitudinal": (
        "It measures the size and power of refutation verdict rules. A verdict is no estimate, "
        "and VERDICT_EVIDENCE, not a support row, records what the study licenses."
    ),
    "refuter-calibration-point": (
        "It measures the size and power of refutation verdict rules. A verdict is no estimate, "
        "and VERDICT_EVIDENCE, not a support row, records what the study licenses."
    ),
    "refuter-calibration-band": (
        "It measures the size and power of refutation verdict rules. A verdict is no estimate, "
        "and VERDICT_EVIDENCE, not a support row, records what the study licenses."
    ),
    "learned-rule-cvtmle-boundary": (
        "It reports truth rows at the exceptional and weak-blip laws, has no property cell, "
        # Roadmap: F27.
        "and licenses no configuration."
    ),
    "continuous-time-survival": (
        "It measures ContinuousSurvivalTMLE, which no support key reads: the continuous-time fit "
        "has no node structure to classify. cleverly.survival.result.continuous_survival_support, "
        "not a support row, records the configurations the study licenses."
    ),
    "adaptive-iptw-pairing": (
        "It pairs the gate study's fits with R drtmle replicate by replicate, has no property "
        "cell, and licenses no configuration. The gate study adaptive-iptw is the evidence of "
        "the AI rows."
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
    "reduced_outcome_learner": "src/cleverly/estimators/iptw.py",
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
    "P12": _anchor("`ey_optimal_rule`", "tests/unit/test_learned_rule_optimal.py"),
    # P1 part (a): the exact-law point and influence-curve checks of both projections.
    "P13": _anchor(
        "`cate_projection`", "tests/discrete_law.py", "tests/unit/test_effect_modification_exact.py"
    ),
    "P14": _anchor(
        "`log_crr_projection`",
        "tests/discrete_law.py",
        "tests/unit/test_effect_modification_exact.py",
    ),
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
    "separable": _anchor(
        "separable effects on competing events",
        "tests/discrete_law_point_survival.py",
        "tests/unit/test_separable_effects_law.py",
    ),
    # X11 (b): with each fold's rule pair held fixed the learned fit is the known
    # ``DynamicRegimen`` fit bit for bit, so the dynamic-rule oracle covers it fold by fold.
    "learned": _anchor(
        "a learned two-node regimen (`LearnedRegimen`), the stacked value "
        "$\\sum_v (n_v/n)\\Psi_{d_{nv}}$",
        "tests/discrete_law_longitudinal.py",
        "tests/unit/test_influence_gateaux_longitudinal.py",
        "tests/unit/test_longitudinal_learned_regimen.py",
    ),
}


# ------------------------------------------------------------------ band cells


#: The band cells that license each row's ``band`` kind, as ``"<study directory>:<cell>"``
#: (band-finite-sample plan, section 4.5). A cell maps to a row only where the fit that
#: produces the cell falls under the row's key, checked against the study's design module
#: named beside each entry. ``tests/unit/test_band_cell_mapping.py`` requires an entry for every
#: row that licenses ``band``, and a positive band cell of a cited study behind each entry.
BAND_CELLS: Final[Mapping[str, tuple[str, ...]]] = {
    # default_bands._default_fit: the shipped TMLE() default, inherited through the study
    # default-simultaneous-bands.
    "T02.binary": (_band("default_bands", "default_fit"),),
    # mar_attributable_properties.BAND_SCENARIOS: each label refits its scenario with
    # canonical_mar_attributable.fit_cleverly, in sample unless the scenario is a CV-TMLE one.
    "T25": (_band("tmle_mar_attributable", "attributable_binary"),),
    # mar_arm_indexed_cvtmle_properties._fit_replication: the band reads the stacked
    # interval_calibration fit of law l1 to l4 (mar_arm_indexed_laws.LAWS).
    "T26.binary": (
        _band("tmle_mar_arm_indexed_cvtmle", "l1"),
        _band("tmle_mar_attributable", "attributable_binary_cvtmle"),
    ),
    "T26.continuous": (_band("tmle_mar_arm_indexed_cvtmle", "l2"),),
    "T26.three_arm": (_band("tmle_mar_arm_indexed_cvtmle", "l3"),),
    "T26.three_arm_continuous": (_band("tmle_mar_arm_indexed_cvtmle", "l4"),),
    "T29.three_arm": (_band("tmle_mar_attributable", "attributable_three_arm"),),
    "T29.weighted": (_band("tmle_mar_attributable", "attributable_weighted"),),
    "T29.clustered": (_band("tmle_mar_attributable", "attributable_clustered"),),
    # multi_arm_mar_drtmle_properties._fit_drawn: canonical_multi_arm_mar_drtmle.fit_cleverly,
    # randomized in-sample DRTMLE at three arms.
    "D09": (_band("drtmle_mar_multi_arm", "arms"),),
    # ctmle_oat_per_arm_parametric_extensions_properties._joint_rows: one band per fit group.
    "C11.binary": (_band("ctmle_oat_per_arm_parametric_extensions", "binary_population"),),
    "C11.k_arms": (_band("ctmle_oat_per_arm_parametric_extensions", "three_arm"),),
    "C11.stacked_k_arms": (_band("ctmle_oat_per_arm_parametric_extensions", "three_arm_crossfit"),),
    # adaptive_iptw_properties._band_row: the targeted form with known arms.
    "AI.known_arms.targeted": (_band("adaptive_iptw", "known_band"),),
    # ltmle_properties._fit_replication: the in-sample correctly specified fit.
    "G01": (_band("ltmle", "regimens"),),
    # clustered_crossfit_ltmle_properties._band_rows: cross-fitted LTMLE on clusters.
    "G05.cross_fitted": (_band("lmtp_clustered_ltmle", "clustered_regimens"),),
    # stochastic_categorical_ltmle_properties._band_rows: in-sample static and known stochastic
    # categorical regimens.
    "G08.in_sample": (_band("stochastic_categorical_ltmle", "primary"),),
    # longitudinal_mtp_properties._band_rows: in-sample two-node continuous MTP.
    "G10.continuous": (_band("longitudinal_mtp", "primary"),),
    # ltmle_survival_properties.fit and fit_curve: in-sample survival LTMLE.
    "G11": (
        _band("ltmle_survival", "all_reported"),
        _band("ltmle_survival", "curve_always"),
    ),
    # ltmle_competing_properties.fit, n_folds=1, with joint_cells=True for this study.
    "G13.in_sample": (
        _band("lmtp_ltmle_competing", "all_reported"),
        _band("lmtp_ltmle_competing", "curve_always"),
    ),
    # The point-survival studies band only the survival_t5 fit (``joint = label ==
    # "survival_t5"`` in each properties module).
    "G14.survival": (_band("point_survival", "all_reported"),),
    "G17.survival": (_band("point_survival_crossfit", "all_reported"),),
    "G30.survival": (_band("point_survival_hazard", "all_reported"),),
    "G31.survival": (
        _band("point_survival_curve", "all_reported"),
        _band("point_survival_curve", "curve_arm0"),
        _band("point_survival_curve", "curve_arm1"),
    ),
    "G32.survival": (
        _band("point_survival_curve_crossfit", "all_reported"),
        _band("point_survival_curve_crossfit", "curve_arm0"),
        _band("point_survival_curve_crossfit", "curve_arm1"),
    ),
    # crossfit_longitudinal_msm_properties._band_rows: the cross-fitted MSM projection.
    "G20": (_band("lmtp_ltmle_msm", "cross_fitted_msm"),),
    # ltmle_repeated_properties._fit_drawn: the in-sample estimated_mechanisms fit.
    "G27": (_band("ltmle_repeated", "all_reported"),),
    # ipsi_repeated_properties._fit_drawn: the in_sample configuration.
    "G37.in_sample": (_band("ipsi_repeated", "all_reported"),),
    # longitudinal_sdr_properties._band_rows: discrete_fit runs _sdr with n_folds=N_FOLDS, a
    # cross-fitted SDR fit, so G23.in_sample has no band cell.
    "G23.cross_fitted": (_band("longitudinal_sdr", "six_regimens"),),
    # longitudinal_incremental_properties._band_rows: cross-fitted IPSI.
    "G33.cross_fitted": (_band("npcausal_longitudinal_incremental", "nine_tilts"),),
    # longitudinal_incremental_dropout_properties._band_drawn: the band reads the five-fold
    # fit; the one-fold fit there gives a pointwise cell only.
    "G34.survival.cross_fitted": (_band("longitudinal_incremental_dropout", "survival_grid"),),
    # sdr_repeated_properties._fit_drawn: the in_sample configuration.
    "G38.in_sample": (_band("sdr_repeated", "all_reported"),),
    # longitudinal_repeated_crossfit_properties.BANDS: each band label's fit set and scenario.
    "G44.law_d.tmle": (_band("longitudinal_repeated_crossfit", "d_ltmle"),),
    "G44.law_d.sdr": (_band("longitudinal_repeated_crossfit", "d_sdr"),),
    "G44.law_s": (_band("longitudinal_repeated_crossfit", "s_ltmle"),),
    "G44.law_c": (_band("longitudinal_repeated_crossfit", "c_ltmle"),),
    "G44.law_m": (_band("longitudinal_repeated_crossfit", "m_ltmle"),),
    "G44.law_i": (_band("longitudinal_repeated_crossfit", "i_ipsi"),),
    # long_horizon_properties.fit_key: the joint arm of the K = 5 binary fit set.
    "G46.tmle.binary_k5": (_band("long_horizon_ltmle", "k5"),),
    # hazard_msm_properties: the joint fit of the four survival-law coefficients.
    "G60.survival": (_band("hazard_msm", "survival_logit_coefficients"),),
    # stratified_tmle_properties._calibration with crossfit=True: the default cross-fitted fit
    # with baseline strata, ("ey", "ate"), under the row's key (stacked, strata).
    "T14": (_band("tmle3_stratified", "crossfit_strata"),),
}

_DEFAULT_BANDS_REASON = (
    "It measures the shipped default band of the configuration of row {row}, which does not "
    "cite default-simultaneous-bands."
)

#: The positive band cells of the studies cited by a band row that license no row, with the
#: reason. Each default_bands cell measures another configuration's default band
#: (default_band_properties.SOURCES); the row named is the one that configuration classifies onto.
UNMAPPED_BAND_CELLS: Final[Mapping[str, str]] = {
    _band("default_bands", setting): _DEFAULT_BANDS_REASON.format(row=row)
    for setting, row in (
        ("ordinary_binary", "T01.binary"),
        ("fold_evaluated", "T03.binary"),
        ("weighted", "T06"),
        ("learned_weighted", "T07"),
        ("missing_outcome", "T25"),
        ("missing_outcome_drtmle", "D08"),
        ("drtmle", "D01"),
        ("multi_arm_drtmle", "D02"),
        ("cde_z1", "T30"),
        ("point_msm", "T18"),
        ("clustered", "T10.unweighted"),
        ("shift_grid", "T21"),
        ("incremental_grid", "T23"),
        ("stochastic_regimes", "T20"),
        ("deterministic_regimes", "T19"),
        ("ltmle_crossfit", "G02"),
        ("survival_crossfit", "G12"),
        ("competing_crossfit", "G13.cross_fitted"),
        ("weighted_ltmle", "G03"),
        ("weighted_ltmle_crossfit", "G04"),
        ("categorical_ltmle_crossfit", "G08.cross_fitted"),
        ("longitudinal_msm", "G19"),
    )
} | {
    # Rows that cite default-simultaneous-bands for a red reading of their own default band.
    _band("default_bands", "cde_z0"): (
        "It measures the default band of the configuration of row T30, which licenses the t band "
        "alone, and it is red; the row carries the reading as a band shortfall."
    ),
    _band("default_bands", "categorical_ltmle"): (
        "It measures the default band of the configuration of row G08.in_sample and it is red; "
        "the row carries the reading as a band shortfall."
    ),
    # band-t-reference-held-out: the held-out cell of the t reference (Fix-2 plan 5.2).
    _t_band("band_t_reference_held_out", "strata_n1000"): (
        "Held out and declared at risk: it tests the rule out of sample on the strata law at "
        "n = 1,000 and licenses no row."
    ),
    # canonical-stratified-tmle is cited by T14's band since its crossfit_strata cell licenses it.
    _band("tmle3_stratified", "strata"): (
        "It measures the default band of the configuration of row T13, which licenses the t band "
        "alone, and it is red; the row carries the reading as a band shortfall."
    ),
}
