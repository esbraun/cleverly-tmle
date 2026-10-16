"""The support rows and cases of the three long-horizon studies (plan rev 4.3).

A support row of :mod:`cleverly._support_rows` names a registered study as its evidence, and a
registered study commits its summaries.  The three long-horizon studies commit none until the
declared run finishes (``tests/unit/test_long_horizon_design.py`` skips until then), so their
rows cannot enter ``_support_rows`` yet.  This module holds them as prepared data:

* :data:`ROWS` is one :class:`~cleverly._support.SupportRow` for each estimator, decision count,
  arm count, splitting and working model that a primary scenario covers.  A row cites its study
  by slug.
* :data:`MEMBERS` names, for each row, the scenarios whose fits the row covers, and :data:`META`
  its estimator setting.  The test derives the licensed kinds from the committed gate lists
  (:func:`licensed_kinds`) and requires them to equal the kinds declared here.
* :data:`CASES` is one case for each member scenario, and one for each of the two bands.  A case
  runs the study's own fit (:func:`tests.studies.long_horizon_primary.fit`, and the band fit of
  :mod:`tests.studies.long_horizon_properties`) up to the support classifier and holds the key it
  reads.

**Kinds.**  Every row licenses ``point``.  It licenses ``wald`` when the study gates a Wald row
of a member scenario or an ``interval_calibration`` cell of the row's setting.  It licenses
``bootstrap_bca`` when the study gates a BCa row of a member scenario.  The cross-fitted rows and
every SDR row license no bootstrap band: the cross-fitted bootstrap is experimental (X20), and the
SDR bootstrap band is experimental (plan rev 4.3).  A row licenses ``bootstrap_band`` when the
study gates the studentized sup-t band of its setting (S1 and S2, K = 5).  No row lists the
``band`` kind, because the Wald-curve band is the reporting cell ``k5__ic_band``.

**Registration.**  Once a study registers, move its rows into ``_support_rows.py`` and its cases
into ``support_cases.py``, delete this module, and let ``tests/unit/test_support_rows.py`` carry
the checks.  ``tests/unit/test_long_horizon_support.py`` fails the day all three studies register
while this module still exists.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Final

from cleverly._support import Kind, SupportRow
from cleverly._support_rows import long_row
from tests import discrete_law_longitudinal_horizon as law_
from tests import long_horizon_design as D
from tests.studies import long_horizon_primary as primary
from tests.studies import long_horizon_properties as properties
from tests.studies.support_cases import Observed, RowCase, capture

__all__ = ["CASES", "MEMBERS", "META", "ROWS", "Setting", "licensed_kinds"]

#: The size of the sample a case draws.  The key reads the structure of a fit, not a value, and
#: 4,000 units pass the fit's preflight for every plan of every scenario.
CASE_N: Final = 4_000

_BASE: Final[frozenset[Kind]] = frozenset({"point", "wald"})
_BCA: Final[frozenset[Kind]] = frozenset({"point", "wald", "bootstrap_bca"})
_BAND: Final[frozenset[Kind]] = frozenset({"point", "wald", "bootstrap_bca", "bootstrap_band"})


@dataclass(frozen=True)
class Setting:
    """The estimator setting of one row.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    K : int
        The decision count.
    m : int
        The arm count of each node.
    folds : int
        1 for the in-sample fit, 5 for the cross-fitted one.
    msm : bool
        Whether the row reports working-model coefficients.
    learned : bool
        Whether the fit learns the treatment mechanism of one arm (the binary-complement route of
        the ``probability_necessity`` cells) instead of taking all of it as known.
    """

    study: str
    K: int
    m: int
    folds: int = 1
    msm: bool = False
    learned: bool = False


def _row(
    name: str,
    setting: Setting,
    slug: str,
    kinds: frozenset[Kind],
    *,
    stochastic: bool,
    limits: tuple[str, ...] = (),
) -> SupportRow:
    engine = "sdr" if setting.study == "S3" else "tmle"
    arms = str(setting.m)
    nodes = tuple(arms for _ in range(setting.K))
    roles = "C" if setting.learned else "AC"
    qualifiers = [f"{role}{k}" for role in roles for k in range(1, setting.K + 1)]
    return long_row(
        f"X39.{engine}.{name}",
        [slug],
        regimens=("static", "stochastic") if stochastic else ("static",),
        msm=setting.msm,
        treatment=nodes,
        splitting=("cross_fitted",) if setting.folds > 1 else ("in_sample",),
        qualifiers=qualifiers,
        engine=engine,
        kinds=kinds,
        cells=[name],
        limits=limits,
    )


_WALD_NOTE = (
    "The Wald interval is gated for a quantity only where n is at least twice its n_kappa; below "
    "that it is a reporting row (plan rev 4.3, decision C1)."
)
_CROSSFIT_NOTE = (
    "The cross-fitted bootstrap interval is experimental: the full-refit bootstrap reads an SE "
    "ratio of 1.17 to 1.33 here (X20), and six of the Wald rows lose their gate below twice "
    "n_kappa."
)
_MSM_NOTE = (
    "Only the duration coefficient has a gated calibration cell.  The intercept needs n of "
    "94,226 for the Wald regime, and the study fits n = 48,000."
)
_SDR_BAND_NOTE = "The SDR bootstrap band is experimental and gates no row."
_NO_PRIMARY_NOTE = (
    "The study fits this setting in its property cells only, so no BCa row gates it and the "
    "Wald interval rests on the calibration cells."
)
_LEARNED_NOTE = (
    "The cell is a bias check of the necessity family: it states no interval, and the treatment "
    "of the third arm is learned."
)
_POINT: Final[frozenset[Kind]] = frozenset({"point"})

LTMLE_BINARY: Final = "long-horizon-ltmle"
LTMLE_CATEGORICAL: Final = "long-horizon-categorical-ltmle"
SDR: Final = "long-horizon-sdr"

#: Every row with its setting, declared kinds and the covered scenarios.
_DECLARED: Final = (
    ("binary_k3", Setting("S1", 3, 2), LTMLE_BINARY, _BCA, True, ("binary_k3",), (_WALD_NOTE,)),
    ("binary_k4", Setting("S1", 4, 2), LTMLE_BINARY, _BCA, True, ("binary_k4",), (_WALD_NOTE,)),
    ("binary_k5", Setting("S1", 5, 2), LTMLE_BINARY, _BAND, True, ("binary_k5",), (_WALD_NOTE,)),
    (
        "binary_k5_msm",
        Setting("S1", 5, 2, msm=True),
        LTMLE_BINARY,
        _BASE,
        False,
        ("binary_k5_msm",),
        (_MSM_NOTE,),
    ),
    (
        "categorical_k3",
        Setting("S2", 3, 3),
        LTMLE_CATEGORICAL,
        _BCA,
        True,
        ("categorical_k3",),
        (_WALD_NOTE,),
    ),
    (
        "categorical_k4",
        Setting("S2", 4, 3),
        LTMLE_CATEGORICAL,
        _BCA,
        True,
        ("categorical_k4", "categorical_k4_high", "categorical_k4_static"),
        (_WALD_NOTE,),
    ),
    (
        "categorical_k5",
        Setting("S2", 5, 3),
        LTMLE_CATEGORICAL,
        _BAND,
        True,
        (
            "categorical_k5",
            "categorical_k5_mix",
            "categorical_k5_static",
            "categorical_k5_switch",
            "categorical_k5_high",
        ),
        (_WALD_NOTE,),
    ),
    (
        "categorical_k5_crossfit",
        Setting("S2", 5, 3, folds=5),
        LTMLE_CATEGORICAL,
        _BASE,
        True,
        ("categorical_k5_crossfit",),
        (_CROSSFIT_NOTE,),
    ),
    ("sdr_binary_k3", Setting("S3", 3, 2), SDR, _BCA, True, ("sdr_binary_k3",), (_WALD_NOTE,)),
    ("sdr_binary_k4", Setting("S3", 4, 2), SDR, _BCA, True, ("sdr_binary_k4",), (_WALD_NOTE,)),
    ("sdr_binary_k5", Setting("S3", 5, 2), SDR, _BCA, True, ("sdr_binary_k5",), (_WALD_NOTE,)),
    (
        "sdr_categorical_k5",
        Setting("S3", 5, 3),
        SDR,
        _BCA,
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
        (_WALD_NOTE, _SDR_BAND_NOTE),
    ),
    (
        "sdr_categorical_k5_crossfit",
        Setting("S3", 5, 3, folds=5),
        SDR,
        _BASE,
        True,
        ("sdr_categorical_k5_crossfit",),
        (_CROSSFIT_NOTE,),
    ),
    ("sdr_categorical_k3", Setting("S3", 3, 3), SDR, _BASE, True, (), (_NO_PRIMARY_NOTE,)),
    ("sdr_categorical_k4", Setting("S3", 4, 3), SDR, _BASE, True, (), (_NO_PRIMARY_NOTE,)),
    (
        "categorical_k5_learned",
        Setting("S2", 5, 3, learned=True),
        LTMLE_CATEGORICAL,
        _POINT,
        True,
        (),
        (_LEARNED_NOTE,),
    ),
    (
        "sdr_categorical_k5_learned",
        Setting("S3", 5, 3, learned=True),
        SDR,
        _POINT,
        True,
        (),
        (_LEARNED_NOTE,),
    ),
)

ROWS: Final[tuple[SupportRow, ...]] = tuple(
    _row(name, setting, slug, kinds, stochastic=stochastic, limits=limits)
    for name, setting, slug, kinds, stochastic, _, limits in _DECLARED
)
#: Each row's id to the scenarios whose fits it covers.
MEMBERS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    row.id: tuple((setting.study, scenario) for scenario in scenarios)
    for row, (_, setting, _, _, _, scenarios, _) in zip(ROWS, _DECLARED, strict=True)
}
#: Each row's id to its estimator setting.
META: Final[dict[str, Setting]] = {
    row.id: setting for row, (_, setting, *_) in zip(ROWS, _DECLARED, strict=True)
}


def licensed_kinds(row_id: str) -> frozenset[Kind]:
    """The kinds the committed gate lists license for the row ``row_id`` (see the module text)."""
    setting = META[row_id]
    members = {scenario for _, scenario in MEMBERS[row_id]}
    gated = [(name, estimand) for study, name, estimand in D.GATED_ROWS if study == setting.study]
    kinds: set[Kind] = {"point"}
    in_members = [(name, estimand) for name, estimand in gated if name in members]
    calibrated = not setting.learned and any(
        cell.study == setting.study
        and (cell.quantity.K, cell.quantity.m, cell.folds, cell.msm)
        == (setting.K, setting.m, setting.folds, setting.msm)
        for cell in D.calibration_cells()
    )
    if calibrated or any(not estimand.endswith(primary.BCA_SUFFIX) for _, estimand in in_members):
        kinds.add("wald")
    if any(estimand.endswith(primary.BCA_SUFFIX) for _, estimand in in_members):
        kinds.add("bootstrap_bca")
    if (
        setting.study in D.BAND_STUDIES
        and setting.folds == 1
        and not setting.msm
        and not setting.learned
        and setting.K == 5
        and D.band_config(setting.study)[0].m == setting.m
    ):
        kinds.add("bootstrap_band")
    return frozenset(kinds)


# --------------------------------------------------------------------------------------- cases
def _scenario(study: str, name: str) -> tuple[object, primary.Scenario]:
    module = D.STUDIES[study]
    return module, next(s for s in module.SCENARIOS if s.name == name)


def _primary_case(row: SupportRow, study: str, name: str) -> RowCase:
    module, scenario = _scenario(study, name)
    slug = row.evidence[0]
    line = inspect.getsourcelines(primary.fit)[1]

    def build() -> Observed:
        frame = law_.draw(scenario.K, scenario.m, CASE_N, 0, scenario.law)
        return capture(lambda: primary.fit(frame, scenario, module.ESTIMATOR))  # type: ignore[attr-defined]

    return RowCase(row.id, slug, name, f"long_horizon_primary.py:{line}", build)


def _band_case(row: SupportRow, study: str) -> RowCase:
    config, plans, _ = D.band_config(study)
    arm = properties.Arm("k5", "positive", config, read="band", plans=plans)
    line = inspect.getsourcelines(properties._call)[1]

    def build() -> Observed:
        drawn = law_.draw(config.K, config.m, CASE_N, 0, config.law)
        return capture(
            lambda: properties._call(
                arm, drawn, plans, simultaneous=True, reference=None, n_bootstrap=D.B_GATE
            )
        )

    return RowCase(
        row.id,
        row.evidence[0],
        "k5__simultaneous_band",
        f"long_horizon_properties.py:{line}",
        build,
    )


def _row_of(study: str, config: D.Config) -> str | None:
    engine = "sdr" if config.estimator == "SDR" else "tmle"
    shape = (config.K, config.m, config.n_folds, config.msm, bool(config.complement_nodes))
    for row_id, s in META.items():
        if (
            s.study == study
            and row_id.split(".")[1] == engine
            and (s.K, s.m, s.folds, s.msm, s.learned) == shape
        ):
            return row_id
    return None


def _cell_cases() -> list[RowCase]:
    """One property-cell case for each row that no primary scenario covers, per cell family."""
    bare = {row_id for row_id, members in MEMBERS.items() if not members}
    out: list[RowCase] = []
    for study in D.STUDIES:
        seen: set[tuple[str, str]] = set()
        for fit_set in properties.fit_sets(study):
            for arm in fit_set.arms:
                if arm.read == "band" or arm.config.estimator not in ("LTMLE", "SDR"):
                    continue
                row_id = _row_of(study, arm.config)
                family = (row_id or "", fit_set.family)
                if row_id not in bare or family in seen:
                    continue
                seen.add(family)
                out.append(_cell_case(next(r for r in ROWS if r.id == row_id), arm))
    return out


def _cell_case(row: SupportRow, arm: properties.Arm) -> RowCase:
    config = arm.config
    line = inspect.getsourcelines(D.Config.fit_sample)[1]

    def build() -> Observed:
        drawn = law_.draw(config.K, config.m, CASE_N, 0, config.law)
        return capture(lambda: config.fit_sample(drawn))

    return RowCase(row.id, row.evidence[0], arm.cell, f"long_horizon_design.py:{line}", build)


def _cases() -> tuple[RowCase, ...]:
    out: list[RowCase] = []
    for row in ROWS:
        out += [_primary_case(row, study, name) for study, name in MEMBERS[row.id]]
        if "bootstrap_band" in row.kinds:
            out.append(_band_case(row, META[row.id].study))
    return tuple(out + _cell_cases())


#: One case for each member scenario, and one for each of the S1 and S2 bands.
CASES: Final[tuple[RowCase, ...]] = _cases()
