r"""Property families of ``parametric-gformula`` (X45).

Every fit is :func:`tests.studies.canonical_parametric_gformula.fit`.  Each draw is shared by
every fit and every cell of its fit set, so the methods are paired within a law.  The
estimator's ``random_state`` comes from the replicate's own stream
(:func:`simulation_seed`), so the simulation error is independent across replicates.

=============================  ===================================================================
family                         cells, size and replications
=============================  ===================================================================
``interval_calibration``       the 15 labels of :data:`study.CALIBRATION_LABELS` at n = 2,000 with
                               ``n_draws`` 10, :data:`CALIBRATION_REPLICATES` of each law, and the
                               two derived controls of each label (a shrunken standard error and
                               added noise)
``root_n_and_efficiency``      the four contrast labels of :data:`study.LADDER_LABELS` at n = 2,000, 4,000
                               and 8,000, :data:`LADDER_REPLICATES` each
``monte_carlo_accounting``     the levels of :data:`study.MC_LABELS` at ``n_draws`` 1.  The positive cell
                               passes the calibration rule.  The ``mc_removed`` cell keeps the
                               estimate of the one-history fit and takes the standard error of an
                               independent fit with :data:`~tests.studies.canonical_parametric_
                               gformula.M_PRIME` histories per unit, so the simulation term is
                               absent from the curve.  Its SE ratio is predicted to fall below
                               the calibration band
``ice_necessity``              the labels of :data:`study.ICE_GATED` on ``G2`` at n = 32,000: the
                               parametric g-formula is unbiased, and the main-terms ICE fit is
                               biased by a declared margin, on the same draws
``ice_reported``               the other paired labels of the ICE fit, reported
``shared_target``              ``LTMLE`` (and ``IPSI`` for the odds tilt) beside the parametric
                               g-formula on the same draws of ``G1`` (n = 2,000) and ``G2`` (n = 8,000), reported
                               in fit sets of their own, so an ``LTMLE`` failure costs no other family
``extrapolation_report``       ``G4Q``: the main-terms fit extrapolates to a fixed dose, reported
=============================  ===================================================================

**Rules.**  ``interval_calibration`` reads the shared calibration rule: the 99% bootstrap
interval of the SE ratio inside (0.93, 1.07) and the 99% coverage interval inside (0.92, 0.98),
with the bias leg on each positive cell.  There is no efficiency band: a parametric plug-in is
not the efficient estimator of the nonparametric model, so
:attr:`StudyRecord.calibration_efficiency_ratio` is ``False``, and the noise control's scale is
:data:`NOISE_SD`, one pilot sampling SD at the cell's size.  ``root_n_and_efficiency`` reads the
shared ladder rule.  ``monte_carlo_accounting`` positive cells read the calibration rule, and a
``mc_removed`` cell in :data:`GATED_DIRECTIONS` passes when its whole SE-ratio interval leaves
the band below.  The design test gates a cell only where its predictive power is at least 0.95.
``ice_necessity`` reads :data:`ICE_NECESSITY`.  The reported families carry no verdict.

Every fit's inference status is checked before its row is written.

**Failures.**  A replicate whose fit raises a recordable error is recorded by
:func:`~tests.studies.evidence.replication.fit_properties` and never redrawn.  It costs each
cell its fit set writes one replicate (:data:`FIT_SET_CELLS`), and the accounting judges each
cell at the declared bound :data:`FAILURE_BOUND` under :data:`PROPERTY_RULES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_parametric_gformula as study
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    Necessity,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_rules,
    necessity_verdicts,
    se_ratio_leaves_band,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CALIBRATION_N = study.CALIBRATION_N
LADDER = study.LADDER
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70

#: The replicates of each law's calibration fit set: the smallest multiple of 200 at which every
#: label of the law passes the calibration rule with chance at least 0.90
#: (``tests/unit/test_parametric_gformula_design.py``).
CALIBRATION_REPLICATES: dict[str, int] = {"G1": 2_400, "G2": 2_400, "G3": 2_400, "G4": 2_400}
LADDER_REPLICATES = 1_000
#: The replicates of each law's ``n_draws`` 1 fit set.
ACCOUNTING_REPLICATES: dict[str, int] = {"G1": 2_400, "G2": 6_400, "G4": 2_400}
#: The replicates of the paired fit sets.
PAIRED_REPLICATES: dict[str, int] = {"G1": 1_200, "G2": 1_200}
EXTRAPOLATION_REPLICATES = 1_000
#: The size of the ``G2`` paired fit set.  The ICE bias is a fixed amount, so its standardized
#: size grows with the square root of n, and the design test shows the power at this size.
PAIRED_N = {"G1": CALIBRATION_N, "G2": 32_000}
#: The size of the shared-target fit sets, which report ``LTMLE`` and ``IPSI`` beside NICE.
SHARED_N = {"G1": CALIBRATION_N, "G2": LADDER[-1]}

#: The empirical SD of each calibration label in the design pilot times :math:`\sqrt{2000}`, so
#: the noise control adds one sampling SD at the cell's size.  It scales the noise control only.
#: The design test pins these values.
NOISE_SD: dict[str, float] = {
    "nice_g1_treat_all": 0.815,
    "nice_g1_treat_none": 1.244,
    "nice_g1_rule": 0.64,
    "nice_g1_odds_x2": 0.635,
    "nice_g1_contrast": 1.672,
    "nice_g2_treat_all": 7.586,
    "nice_g2_treat_none": 8.963,
    "nice_g2_policy_b": 5.929,
    "nice_g2_contrast": 10.74,
    "nice_g3_relapse_h2": 0.583,
    "nice_g3_relapse_h4": 0.832,
    "nice_g3_death_h4": 0.645,
    "nice_g3_contrast_h4": 1.725,
    "nice_g4_shift": 3.889,
    "nice_g4_dose": 1.76,
    "nice_g4_contrast": 1.168,
}
#: Every ``mc_removed`` cell whose predicted side of the calibration band the design test shows
#: powered (predictive power >= 0.95 at :data:`ACCOUNTING_REPLICATES`).  The others report.
GATED_DIRECTIONS: dict[str, Literal["below", "above"]] = {"nice_g2_treat_all__mc_removed": "below"}

# ------------------------------------------------------------------------------ the truths


def _targets() -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for law, keys in study.KEYS.items():
        truth = study.truths(law)
        for key, name in keys.items():
            out[study.label_of(law, key)] = (name, float(truth[name]))
    for law, keys in study.PAIR_KEYS.items():
        truth = study.truths(law)
        for key, name in keys.items():
            out[f"pair_{law.lower()}_{key}"] = (name, float(truth[name]))
    for law, shared in study.SHARED_KEYS.items():
        truth = study.truths(law)
        for key, (name, _) in shared.items():
            out[f"target_{law.lower()}_{key}"] = (name, float(truth[name]))
    name = "ey_regimen[dose_half]"
    out[study.EXTRAPOLATION_LABEL] = (name, float(study.truths("G4Q")[name]))
    return out


#: ``label -> (estimand, truth)`` for every label of every family.
TARGETS: dict[str, tuple[str, float]] = _targets()

# ------------------------------------------------------------------------------- fit sets

#: Every fit set: ``name -> (law, n, replicates)``.  One draw of the law per replicate.
FIT_SETS: dict[str, tuple[str, int, int]] = {
    **{
        f"calibration_{law.lower()}": (law, CALIBRATION_N, CALIBRATION_REPLICATES[law])
        for law in study.LAWS
    },
    **{
        f"accounting_{law.lower()}": (law, CALIBRATION_N, ACCOUNTING_REPLICATES[law])
        for law in study.MC_KEYS
    },
    **{
        f"ladder_{label.split('_')[1]}_n_{size}": (
            label.split("_")[1].upper(),
            size,
            LADDER_REPLICATES,
        )
        for label in study.LADDER_LABELS
        for size in LADDER
    },
    **{
        f"paired_{law.lower()}": (law, PAIRED_N[law], PAIRED_REPLICATES[law])
        for law in study.PAIR_KEYS
    },
    **{
        f"shared_{law.lower()}": (law, SHARED_N[law], PAIRED_REPLICATES[law])
        for law in study.SHARED_KEYS
    },
    "extrapolation_g4q": ("G4Q", CALIBRATION_N, EXTRAPOLATION_REPLICATES),
}


def sample_seed(fit_set: str, replicate: int) -> int:
    """The seed of one draw: this study's own ``property_sample`` stream."""
    return stream_seed(STUDY, "property_sample", fit_set, replicate)


def simulation_seed(fit_set: str, replicate: int, stream: str = "estimate") -> int:
    """The seed of one fit's simulation: a stream that no sample or other fit reads."""
    return study.simulation_seed(
        stream_seed(STUDY, "property_simulation", fit_set, replicate, stream)
    )


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One draw of ``law``."""
    return study.law_of(law).sample(n, seed)


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    est: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=est,
        alpha=STUDY.margins.alpha,
    )


def _names(labels: Any) -> list[str]:
    return [TARGETS[label][0] for label in labels]


def _check_dose_reliance(result: Any) -> None:
    """A fixed dose carries the model-reliance statement and a support diagnostic."""
    report = result.simulation
    if report.model_reliance is None or not report.support:
        raise study.StatusMismatch("a fixed dose fit carries no model-reliance statement")


def fit_set_rows(fit_set: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes.

    Each fit's inference status is checked before a row is written
    (:func:`~tests.studies.canonical_parametric_gformula.require_status`).  A mismatch stops the
    run rather than calibrating the wrong curve.
    """
    law, n, requested = FIT_SETS[fit_set]
    kind = fit_set.split("_")[0]
    rows: list[dict[str, Any]] = []
    seed = simulation_seed(fit_set, replicate)

    def add(family: str, cell: str, role: str, label: str, result: Any) -> None:
        name, truth = TARGETS[label]
        rows.append(_row(family, cell, role, replicate, n, requested, truth, result[name]))

    if kind == "calibration":
        labels = [study.label_of(law, key) for key in study.KEYS[law]]
        result = study.fit("nice", law, frame, seed=seed)
        study.require_status(result, _names(labels))
        if law == "G4":
            _check_dose_reliance(result)
        for label in labels:
            add("interval_calibration", f"{label}__correctly_specified", "positive", label, result)
    elif kind == "ladder":
        label = next(x for x in study.LADDER_LABELS if x.split("_")[1] == law.lower())
        result = study.fit("nice", law, frame, seed=seed)
        study.require_status(result, _names([label]))
        add("root_n_and_efficiency", f"{label}__n_{n}", "positive", label, result)
    elif kind == "accounting":
        static = law != "G4"
        one = study.fit(
            "nice", law, frame, seed=seed, n_draws=study.N_DRAWS_ONE, static_only=static
        )
        many = study.fit(
            "nice",
            law,
            frame,
            seed=simulation_seed(fit_set, replicate, "m_prime"),
            n_draws=study.M_PRIME,
            n_derivative_draws=study.N_DRAWS,
            static_only=static,
        )
        for label in (study.label_of(law, key) for key in study.MC_KEYS[law]):
            name, truth = TARGETS[label]
            study.require_status(one, [name])
            study.require_status(many, [name])
            add("monte_carlo_accounting", f"{label}__n_draws_1", "positive", label, one)
            rows.append(
                control_row(
                    property_name="monte_carlo_accounting",
                    cell=f"{label}__mc_removed",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=float(one[name].psi),
                    standard_error=float(many[name].std_error),
                    critical=CRITICAL,
                    role=DIAGNOSTIC_ROLE,
                )
            )
    elif kind == "paired":
        nice = study.fit("nice", law, frame, seed=seed, static_only=False)
        ice = study.fit("ice", law, frame)
        study.require_status(nice, _names(f"pair_{law.lower()}_{k}" for k in study.PAIR_KEYS[law]))
        study.require_status(ice, _names(f"pair_{law.lower()}_{k}" for k in study.PAIR_KEYS[law]))
        for key in study.PAIR_KEYS[law]:
            label = f"pair_{law.lower()}_{key}"
            gated = label in study.ICE_GATED
            family = "ice_necessity" if gated else "ice_reported"
            add(family, f"{label}__nice", "positive" if gated else DIAGNOSTIC_ROLE, label, nice)
            add(family, f"{label}__ice", "control" if gated else DIAGNOSTIC_ROLE, label, ice)
    elif kind == "shared":
        nice = study.fit("nice", law, frame, seed=seed, static_only=False)
        others = {
            method: study.fit(method, law, frame)
            for method in sorted({m for _, m in study.SHARED_KEYS[law].values()})
        }
        for key, (name, method) in study.SHARED_KEYS[law].items():
            label = f"target_{law.lower()}_{key}"
            other = others[method]
            study.require_status(other, [name])
            study.require_status(nice, [name])
            add("shared_target", f"{label}__nice", DIAGNOSTIC_ROLE, label, nice)
            add("shared_target", f"{label}__{method}", DIAGNOSTIC_ROLE, label, other)
    elif kind == "extrapolation":
        result = study.fit("nice", law, frame, seed=seed)
        study.require_status(result, _names([study.EXTRAPOLATION_LABEL]))
        add(
            "extrapolation_report",
            f"{study.EXTRAPOLATION_LABEL}__main_terms",
            DIAGNOSTIC_ROLE,
            study.EXTRAPOLATION_LABEL,
            result,
        )
    else:
        raise KeyError(fit_set)
    return rows


# ------------------------------------------------------------------------------ the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        The law's name.  A family that reads the fits of another family carries the family in
        its name, so the collision gate keys its stream apart from the claim.
    law : str
        The law whose truths :func:`~tests.studies.canonical_parametric_gformula.truths`
        returns.
    """

    name: str
    law: str

    def truth(self) -> dict[str, float]:
        """Return ``estimand -> truth`` for every estimand of this law."""
        return dict(study.truths(self.law))


def _cells_and_fit_sets() -> list[tuple[PropertyCell, str]]:
    """Every sampled cell beside the fit set whose draws it reads."""
    cells: list[tuple[PropertyCell, str]] = []
    owners: dict[str, str] = {}

    def add(family: str, cell: str, role: str, fit_set: str, label: str) -> None:
        law, n, replicates = FIT_SETS[fit_set]
        owner = owners.setdefault(fit_set, family)
        name = law if owner == family else f"{law}_{family}"
        cells.append(
            (
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=DeclaredLaw(name, law),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=n,
                    replicates=replicates,
                    seed=sample_seed(fit_set, 0),
                    role=role,
                    estimand=TARGETS[label][0],
                ),
                fit_set,
            )
        )

    for label in study.CALIBRATION_LABELS:
        law = label.split("_")[1].upper()
        for kind, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{kind}",
                role,
                f"calibration_{law.lower()}",
                label,
            )
    for label in study.LADDER_LABELS:
        law = label.split("_")[1]
        for size in LADDER:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "positive",
                f"ladder_{law}_n_{size}",
                label,
            )
    for label in study.MC_LABELS:
        law = label.split("_")[1]
        add("monte_carlo_accounting", f"{label}__n_draws_1", "positive", f"accounting_{law}", label)
        add(
            "monte_carlo_accounting",
            f"{label}__mc_removed",
            DIAGNOSTIC_ROLE,
            f"accounting_{law}",
            label,
        )
    for label in study.PAIR_LABELS:
        law = label.split("_")[1]
        gated = label in study.ICE_GATED
        family = "ice_necessity" if gated else "ice_reported"
        add(
            family,
            f"{label}__nice",
            "positive" if gated else DIAGNOSTIC_ROLE,
            f"paired_{law}",
            label,
        )
        add(
            family, f"{label}__ice", "control" if gated else DIAGNOSTIC_ROLE, f"paired_{law}", label
        )
    for law, shared in study.SHARED_KEYS.items():
        for key, (_, method) in shared.items():
            label = f"target_{law.lower()}_{key}"
            for arm in ("nice", method):
                add(
                    "shared_target",
                    f"{label}__{arm}",
                    DIAGNOSTIC_ROLE,
                    f"shared_{law.lower()}",
                    label,
                )
    add(
        "extrapolation_report",
        f"{study.EXTRAPOLATION_LABEL}__main_terms",
        DIAGNOSTIC_ROLE,
        "extrapolation_g4q",
        study.EXTRAPOLATION_LABEL,
    )
    return cells


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its law, estimand, size and stream root."""
    return tuple(cell for cell, _ in _cells_and_fit_sets())


# ------------------------------------------------------------------------------- the loop


def _draw_key(key: tuple[str, int]) -> pd.DataFrame:
    fit_set, replicate = key
    law, n, _ = FIT_SETS[fit_set]
    return draw(law, n, sample_seed(fit_set, replicate))


def _fit_key(key: tuple[str, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_set_rows(key[0], key[1], frame)


#: Each calibration positive arm to the controls derived from it.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(study.CALIBRATION_LABELS)


def _fit_set_cells() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cell, fit_set in _cells_and_fit_sets():
        out.setdefault(fit_set, []).append(f"{cell.property}/{cell.cell}")
    return out


#: The ``property/cell`` names that each fit set's rows and derived controls fill.
FIT_SET_CELLS: dict[str, list[str]] = _fit_set_cells()


def _cells_of(key: tuple[str, int]) -> list[str]:
    """The ``property/cell`` names a key's rows and derived controls fill."""
    return FIT_SET_CELLS[key[0]]


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def run_fit_set(fit_set: str, *, n_jobs: int, replicates: int | None = None) -> pd.DataFrame:
    """Every row of one fit set; ``replicates`` caps the budget for a smoke or a pilot.

    A recordable failure costs each cell its fit writes one replicate, and is never redrawn.
    The driver records it, and the accounting judges each cell at :data:`FAILURE_BOUND`.
    """
    _, _, declared = FIT_SETS[fit_set]
    count = declared if replicates is None else replicates
    outcomes = fit_properties(
        [(fit_set, replicate) for replicate in range(count)],
        _fit_key,
        cells_of=_cells_of,
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        n_jobs=n_jobs,
        label=f"properties/{fit_set}",
        draw=_draw_key,
        derived=DERIVED_CELLS,
        seed_of=lambda key: sample_seed(key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    if replicates is not None:
        rows["requested_replicates"] = count
    return rows


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, fit_sets: Any = None
) -> pd.DataFrame:
    """Fit every property replicate; ``budget`` caps each fit set for a smoke or a pilot."""
    names = list(FIT_SETS) if fit_sets is None else list(fit_sets)
    rows = pd.concat(
        [run_fit_set(name, n_jobs=n_jobs, replicates=budget) for name in names],
        ignore_index=True,
    )
    labels = [
        label
        for label in study.CALIBRATION_LABELS
        if f"{label}__correctly_specified" in set(rows["cell"])
    ]
    if labels:
        rows = pd.concat(
            [
                rows,
                calibration_controls(
                    rows,
                    STUDY,
                    labels=labels,
                    efficiency_bounds=NOISE_SD,
                    calibration_n=CALIBRATION_N,
                    shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                    critical=CRITICAL,
                ),
            ],
            ignore_index=True,
        )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts

#: The ``ice_necessity`` family's declaration, read by its verdict and its rule.  The control
#: arm is the main-terms ICE fit, whose bias the design test shows.  The displacement threshold
#: is 0.35 of the positive arm's SD: the design test shows each gated pair differs by more.
ICE_DISPLACEMENT = 0.35
ICE_NECESSITY = Necessity(
    family="ice_necessity",
    labels=study.ICE_GATED,
    arms=("nice", "ice"),
    column="ice_displacement",
    threshold=ICE_DISPLACEMENT,
)

#: The families that carry a verdict beyond the shared calibration and ladder rules.
OWN_FAMILIES = ("monte_carlo_accounting",)
REPORTED_FAMILIES = ("ice_reported", "shared_target", "extrapolation_report")


def _own_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    margins = STUDY.margins
    for index in summary.index[summary["property"].isin(OWN_FAMILIES)]:
        family = str(summary.loc[index, "property"])
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        ratio = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, family, cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        if cell.endswith("__n_draws_1"):
            coverage = (
                float(summary.loc[index, "coverage_ci_lower"]),
                float(summary.loc[index, "coverage_ci_upper"]),
            )
            summary.loc[index, "passed"] = bool(
                ratio.within(*margins.calibration_se_ratio)
                and margins.calibration_coverage[0] <= coverage[0]
                and coverage[1] <= margins.calibration_coverage[1]
                and bool(summary.loc[index, "bias_equivalent"])
            )
            continue
        side = GATED_DIRECTIONS.get(cell)
        # A gated cell must leave the calibration band on its predicted side, so it shows that
        # the calibration instrument would catch a curve without the simulation term.
        summary.loc[index, "passed"] = (
            True
            if side is None
            else se_ratio_leaves_band(ratio, margins.calibration_se_ratio, side)
        )
    reported = summary["property"].isin(REPORTED_FAMILIES)
    summary.loc[reported, "passed"] = True
    # An unpowered ICE label reports: its two arms are diagnostic rows.


def _own_seed(ctx: pf.Context, row: pd.Series) -> int:
    return stream_seed(ctx.record, str(row["property"]), str(row["cell"]))


def _calibration_band(ctx: pf.Context) -> tuple[float, float]:
    return ctx.margins.calibration_se_ratio


def _calibration_rule() -> pf.FamilyRule:
    """The shared calibration rule, with the bias leg on each positive cell."""
    shared = property_verdicts.calibration_rule(efficiency_band=None)
    return pf.FamilyRule(
        families=shared.families,
        shared="calibration_verdicts",
        clauses=(
            *shared.clauses,
            pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
        ),
    )


def _accounting_rule() -> pf.FamilyRule:
    """The simulation term's rule: the positive calibration rule, then the gated directions."""

    def leaves(side: Literal["below", "above"]) -> pf.Judge:
        def judge(ctx: pf.Context, row: pd.Series, iv: Any) -> pf.Judged:
            low, high = ctx.margins.calibration_se_ratio
            slack = low - iv.high if side == "below" else iv.low - high
            return pf.Judged(slack, se_ratio_leaves_band(iv, (low, high), side))

        return judge

    return pf.FamilyRule(
        families=frozenset({"monte_carlo_accounting"}),
        shared="summarize_properties",
        clauses=(
            pf.se_interval_clause(
                "se ratio in calibration band",
                which="se_ratio",
                judge=pf.band_judge(_calibration_band),
                seed=_own_seed,
                applies=pf.roles("positive"),
            ),
            pf.proportion_clause(
                "coverage in calibration band",
                statistic="S2",
                judge=pf.band_judge(lambda ctx: ctx.margins.calibration_coverage),
                applies=pf.roles("positive"),
            ),
            pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
            *(
                pf.se_interval_clause(
                    f"se ratio {side} band",
                    which="se_ratio",
                    judge=leaves(side),
                    seed=_own_seed,
                    applies=pf.cells(*(cell for cell, s in GATED_DIRECTIONS.items() if s == side)),
                )
                for side in sorted(set(GATED_DIRECTIONS.values()))
            ),
        ),
    )


#: The failure accounting's declared rules.
PROPERTY_RULES = (_calibration_rule(), _accounting_rule(), *necessity_rules(ICE_NECESSITY))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    labels = tuple(
        label for label in study.LADDER_LABELS if (rows["cell"] == f"{label}__n_{LADDER[0]}").any()
    )
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=labels, extra_columns=(ICE_NECESSITY.column,)
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    # The shared calibration rule reads the SE ratio and coverage only, so every positive
    # calibration cell also needs its 99% bias interval inside 0.25 SD.
    positive = (summary["property"] == "interval_calibration") & summary["cell"].str.endswith(
        "__correctly_specified"
    )
    summary.loc[positive, "passed"] = summary.loc[positive, "passed"].astype(bool) & summary.loc[
        positive, "bias_equivalent"
    ].astype(bool)
    necessity_verdicts(summary, rows, ICE_NECESSITY)
    _own_verdicts(summary, rows)
    return finish(summary, rates)


def cell_label(cell: str) -> str:
    """The label of a cell name, before ``__``."""
    return cell.split("__", 1)[0]


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
